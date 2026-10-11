//! Brush painting on the CPU, in texture space.
//!
//! For every triangle near the brush we walk the texels it covers in its UV
//! layout, reconstruct each texel's 3D position, and ask the brush how much it
//! covers that position. This is the same shape as GPU texture painting
//! (render the mesh in UV space, evaluate the brush per fragment), so it can
//! move to a shader later without changing behaviour.
//!
//! Coverage is accumulated per stroke with `max`, and pixels are always
//! blended from the image as it was when the stroke started. Overlapping
//! segments of one stroke therefore never darken each other.

use crate::mesh::{Image, ImageRegion, Mesh, linear_to_srgb, srgb_to_linear};
use crate::painting::brush::Brush;
use crate::renderer::camera::OrbitCamera;
use egui::{Pos2, Rect};
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::cell::OnceCell;

const TILE_SIZE: f32 = 32.0;
/// Texels whose centre lies this many texels outside a triangle's UV footprint
/// are still painted, so UV seams don't show a hairline gap.
const SEAM_MARGIN_TEXELS: f32 = 1.0;
/// The surface brush skips triangles facing more than ~100 degrees away from
/// the surface under the cursor: it wraps around edges but not through walls.
const SURFACE_MIN_FACING: f32 = -0.2;
/// Relative depth tolerance when deciding whether a texel is hidden.
const DEPTH_TOLERANCE: f32 = 1e-3;
/// Two triangles sharing an edge are one smooth surface when their vertex
/// normals along it agree this closely (cosine); otherwise it's a hard edge.
const SMOOTH_EDGE_COSINE: f32 = 0.99;
/// Number of points in the projected brush outline.
const OUTLINE_POINTS: usize = 48;

struct Triangle {
    position: [Vec3; 3],
    uv: [Vec2; 3],
    /// Face normal.
    normal: Vec3,
    /// Shading normals at the corners (the face normal where the mesh has none).
    corner_normals: [Vec3; 3],
    material: usize,
}

impl Triangle {
    /// Shading normal at a point on the triangle. Falls back to the face
    /// normal when the mesh's normals are missing or point somewhere odd.
    fn smooth_normal(&self, position: Vec3) -> Vec3 {
        barycentric_3d(position, self.position)
            .map(|weights| interpolate(self.corner_normals, weights).normalize_or_zero())
            .filter(|normal| normal.dot(self.normal) > 0.5)
            .unwrap_or(self.normal)
    }
}

/// A shared edge, identified by the exact bits of its two end positions.
type EdgeKey = ([u32; 3], [u32; 3]);

fn position_key(position: Vec3) -> [u32; 3] {
    // `+ 0.0` turns -0.0 into 0.0 so both get the same bits.
    position.to_array().map(|value| (value + 0.0).to_bits())
}

fn edge_key(a: Vec3, b: Vec3) -> EdgeKey {
    let (a, b) = (position_key(a), position_key(b));
    if a <= b { (a, b) } else { (b, a) }
}

/// What the cursor is over.
struct SurfaceHit {
    triangle: u32,
    position: Vec3,
    normal: Vec3,
}

/// One continuous brush stroke on one texture set.
pub struct Stroke {
    material: usize,
    /// The texture as it was before the stroke started.
    original: Vec<u8>,
    /// Highest brush coverage (0..1) each texel received during this stroke.
    coverage: Vec<f32>,
    /// When the brush may not paint over edges: which triangles belong to the
    /// face the stroke started on. Found on the first segment that hits the model.
    face: Option<Vec<bool>>,
}

impl Stroke {
    pub fn begin(mesh: &Mesh, material: usize) -> Option<Self> {
        let image = &mesh.materials.get(material)?.base_color;
        Some(Self {
            material,
            original: image.rgba.clone(),
            coverage: vec![0.0; image.rgba.len() / 4],
            face: None,
        })
    }

    pub fn material(&self) -> usize {
        self.material
    }

    fn allows(&self, triangle: u32) -> bool {
        self.face
            .as_ref()
            .is_none_or(|face| face.get(triangle as usize).copied().unwrap_or(false))
    }
}

/// The mesh as seen from one camera: world-space triangles plus a screen tile
/// grid of the visible ones, for picking and occlusion tests.
pub struct StrokePainter {
    viewport: Rect,
    view_proj: Mat4,
    inverse_view_proj: Mat4,
    eye: Vec3,
    forward: Vec3,
    /// World units per screen point at a view depth of 1.
    world_per_point: f32,
    triangles: Vec<Triangle>,
    tiles_x: usize,
    tiles_y: usize,
    /// Indices of front-facing triangles overlapping each screen tile.
    tiles: Vec<Vec<u32>>,
    /// Every triangle edge, sorted by key, to find neighbours. Built on first use.
    edges: OnceCell<Vec<(EdgeKey, u32)>>,
}

impl StrokePainter {
    pub fn new(mesh: &Mesh, camera: &OrbitCamera, viewport: Rect) -> Self {
        let view_proj = camera.view_proj(aspect(viewport));
        let eye = camera.eye();
        let tiles_x = (viewport.width() / TILE_SIZE).ceil().max(1.0) as usize;
        let tiles_y = (viewport.height() / TILE_SIZE).ceil().max(1.0) as usize;
        let mut painter = Self {
            viewport,
            view_proj,
            inverse_view_proj: view_proj.inverse(),
            eye,
            forward: (camera.target - eye).normalize_or_zero(),
            world_per_point: 2.0 * (camera.fov_y * 0.5).tan() / viewport.height().max(1.0),
            triangles: Vec::new(),
            tiles_x,
            tiles_y,
            tiles: vec![Vec::new(); tiles_x * tiles_y],
            edges: OnceCell::new(),
        };

        for submesh in &mesh.submeshes {
            let range = submesh.indices.start as usize..submesh.indices.end as usize;
            let Some(indices) = mesh.indices.get(range) else {
                continue;
            };
            for corners in indices.chunks_exact(3) {
                let Some(vertices) = corners
                    .iter()
                    .map(|&i| mesh.vertices.get(i as usize))
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                let position = [0, 1, 2].map(|i| Vec3::from(vertices[i].position));
                let uv = [0, 1, 2].map(|i| Vec2::from(vertices[i].uv));
                let normal = (position[1] - position[0])
                    .cross(position[2] - position[0])
                    .normalize_or_zero();
                if normal == Vec3::ZERO || !position.iter().all(|p| p.is_finite()) {
                    continue;
                }
                let corner_normals = [0, 1, 2].map(|i| {
                    let corner = Vec3::from(vertices[i].normal).normalize_or_zero();
                    if corner.is_finite() && corner != Vec3::ZERO {
                        corner
                    } else {
                        normal
                    }
                });
                let index = painter.triangles.len() as u32;
                painter.triangles.push(Triangle {
                    position,
                    uv,
                    normal,
                    corner_normals,
                    material: submesh.material,
                });
                painter.bin_triangle(index);
            }
        }
        painter
    }

    /// True if this painter was built for the given view and can be reused.
    pub fn matches(&self, camera: &OrbitCamera, viewport: Rect) -> bool {
        self.viewport == viewport && self.view_proj == camera.view_proj(aspect(viewport))
    }

    /// Paints one segment of `stroke` between two screen positions (in points)
    /// and returns the texel region that changed.
    pub fn paint_segment(
        &self,
        stroke: &mut Stroke,
        mesh: &mut Mesh,
        from: Pos2,
        to: Pos2,
        brush: &Brush,
    ) -> Option<ImageRegion> {
        let image = &mut mesh.materials.get_mut(stroke.material)?.base_color;
        if image.rgba.len() != stroke.original.len() || self.viewport.height() <= 0.0 {
            return None;
        }
        let from = self.to_local(from);
        let to = self.to_local(to);
        if brush.paint_over_edges {
            stroke.face = None;
        } else if stroke.face.is_none() {
            let start = self.surface_hit(from).or_else(|| self.surface_hit(to))?;
            stroke.face = Some(self.connected_face(start.triangle));
        }
        let radius = (brush.size * 0.5).max(0.5);
        let mut target = PaintTarget {
            image,
            stroke,
            color: brush.color.map(srgb_float_to_linear),
            opacity: brush.opacity.clamp(0.0, 1.0),
            region: None,
        };

        if brush.screen_space {
            self.paint_screen_segment(&mut target, from, to, radius, brush.hardness);
        } else {
            self.paint_surface_segment(&mut target, from, to, radius, brush.hardness);
        }
        target.region
    }

    /// Outline of the surface brush under the cursor: a disc lying on the
    /// surface, in screen points. `None` when the cursor isn't over the model.
    pub fn brush_outline(&self, cursor: Pos2, brush: &Brush) -> Option<Vec<Pos2>> {
        let hit = self.surface_hit(self.to_local(cursor))?;
        let world_radius = self.world_radius(hit.position, (brush.size * 0.5).max(0.5));
        let (axis_a, axis_b) = hit.normal.any_orthonormal_pair();
        (0..OUTLINE_POINTS)
            .map(|i| {
                let angle = i as f32 / OUTLINE_POINTS as f32 * std::f32::consts::TAU;
                let (sin, cos) = angle.sin_cos();
                let point = hit.position + (axis_a * cos + axis_b * sin) * world_radius;
                let clip = self.view_proj * point.extend(1.0);
                if clip.w <= 0.0 || clip.z < 0.0 {
                    return None;
                }
                let screen = self.clip_to_screen(clip);
                Some(Pos2::new(
                    screen.x + self.viewport.min.x,
                    screen.y + self.viewport.min.y,
                ))
            })
            .collect()
    }

    /// Projection painting: the brush is a circle on screen, projected onto
    /// whatever is visible under it.
    fn paint_screen_segment(
        &self,
        target: &mut PaintTarget,
        from: Vec2,
        to: Vec2,
        radius: f32,
        hardness: f32,
    ) {
        let min = from.min(to) - Vec2::splat(radius);
        let max = from.max(to) + Vec2::splat(radius);
        let corners = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)];
        let corner_rays = corners.map(|corner| self.ray(corner));

        for index in self.triangles_in_screen_rect(min, max) {
            let triangle = &self.triangles[index as usize];
            if triangle.material != target.stroke.material || !target.stroke.allows(index) {
                continue;
            }
            // Where the brush rectangle lands on this triangle's plane, in UV.
            let bounds = corner_rays
                .iter()
                .map(|ray| {
                    let point =
                        ray_plane(self.eye, (*ray)?, triangle.position[0], triangle.normal)?;
                    Some(interpolate(
                        triangle.uv,
                        barycentric_3d(point, triangle.position)?,
                    ))
                })
                .collect::<Option<Vec<_>>>()
                .map(|uvs| uv_bounds(&uvs));

            target.paint_triangle(triangle, bounds, |position, texel_reach| {
                let clip = self.view_proj * position.extend(1.0);
                if clip.w <= 0.0 || clip.z < 0.0 {
                    return 0.0;
                }
                let screen = self.clip_to_screen(clip);
                let depth = (position - self.eye).dot(self.forward).max(1e-6);
                let reach = texel_reach / (self.world_per_point * depth);
                let distance = distance_to_segment(screen, from, to) - reach;
                let coverage = falloff(distance.max(0.0), radius, hardness);
                if coverage <= 0.0 || !self.is_visible(position, screen, index) {
                    return 0.0;
                }
                coverage
            });
        }
    }

    /// Surface painting: the brush is a sphere around the surface under the
    /// cursor, sized so it matches the screen circle there. It keeps a constant
    /// size on the model and wraps around edges.
    fn paint_surface_segment(
        &self,
        target: &mut PaintTarget,
        from: Vec2,
        to: Vec2,
        radius: f32,
        hardness: f32,
    ) {
        let start_hit = self.surface_hit(from);
        let Some(end_hit) = self.surface_hit(to).or_else(|| self.surface_hit(from)) else {
            return;
        };
        let (end, normal) = (end_hit.position, end_hit.normal);
        let world_radius = self.world_radius(end, radius);
        let mut start = start_hit.map_or(end, |hit| hit.position);
        // Sweeping across a silhouette can make the two ends far apart in 3D;
        // a capsule between them would cut through empty space. Use a dab.
        let screen_length = self.world_radius(end, from.distance(to));
        if start.distance(end) > 2.0 * screen_length + world_radius {
            start = end;
        }

        let lo = start.min(end) - Vec3::splat(world_radius);
        let hi = start.max(end) + Vec3::splat(world_radius);
        for (index, triangle) in self.triangles.iter().enumerate() {
            if triangle.material != target.stroke.material
                || triangle.normal.dot(normal) < SURFACE_MIN_FACING
                || !target.stroke.allows(index as u32)
            {
                continue;
            }
            let tri_lo = triangle.position[0]
                .min(triangle.position[1])
                .min(triangle.position[2]);
            let tri_hi = triangle.position[0]
                .max(triangle.position[1])
                .max(triangle.position[2]);
            if tri_lo.cmpgt(hi).any() || tri_hi.cmplt(lo).any() {
                continue;
            }

            // The brush on this plane fits inside discs around both ends.
            let (axis_a, axis_b) = triangle.normal.any_orthonormal_pair();
            let plane =
                |p: Vec3| p - triangle.normal * (p - triangle.position[0]).dot(triangle.normal);
            let bounds = [plane(start), plane(end)]
                .into_iter()
                .flat_map(|center| {
                    [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
                        .map(|(a, b)| center + (axis_a * a + axis_b * b) * world_radius)
                })
                .map(|point| {
                    Some(interpolate(
                        triangle.uv,
                        barycentric_3d(point, triangle.position)?,
                    ))
                })
                .collect::<Option<Vec<_>>>()
                .map(|uvs| uv_bounds(&uvs));

            target.paint_triangle(triangle, bounds, |position, texel_reach| {
                let distance = distance_to_segment_3d(position, start, end) - texel_reach;
                falloff(distance.max(0.0), world_radius, hardness)
            });
        }
    }

    /// All triangles reachable from `start` without crossing a hard edge.
    fn connected_face(&self, start: u32) -> Vec<bool> {
        let edges = self.edges.get_or_init(|| {
            let mut edges: Vec<(EdgeKey, u32)> = self
                .triangles
                .iter()
                .enumerate()
                .flat_map(|(index, triangle)| {
                    (0..3).map(move |corner| {
                        let a = triangle.position[corner];
                        let b = triangle.position[(corner + 1) % 3];
                        (edge_key(a, b), index as u32)
                    })
                })
                .collect();
            edges.sort_unstable();
            edges
        });

        let mut inside = vec![false; self.triangles.len()];
        let mut pending = vec![start];
        inside[start as usize] = true;
        while let Some(current) = pending.pop() {
            let triangle = &self.triangles[current as usize];
            for corner in 0..3 {
                let (a, b) = (
                    triangle.position[corner],
                    triangle.position[(corner + 1) % 3],
                );
                let key = edge_key(a, b);
                let first = edges.partition_point(|(edge, _)| *edge < key);
                for &(_, neighbour) in edges[first..].iter().take_while(|(edge, _)| *edge == key) {
                    if !inside[neighbour as usize] && self.is_smooth_edge(current, neighbour, a, b)
                    {
                        inside[neighbour as usize] = true;
                        pending.push(neighbour);
                    }
                }
            }
        }
        inside
    }

    /// Whether the shading is continuous across the edge `a`-`b` shared by two triangles.
    fn is_smooth_edge(&self, first: u32, second: u32, a: Vec3, b: Vec3) -> bool {
        let normal_at = |triangle: &Triangle, point: Vec3| {
            let key = position_key(point);
            (0..3)
                .find(|&corner| position_key(triangle.position[corner]) == key)
                .map(|corner| triangle.corner_normals[corner])
        };
        let (first, second) = (
            &self.triangles[first as usize],
            &self.triangles[second as usize],
        );
        [a, b].into_iter().all(
            |point| match (normal_at(first, point), normal_at(second, point)) {
                (Some(n1), Some(n2)) => n1.dot(n2) >= SMOOTH_EDGE_COSINE,
                _ => false,
            },
        )
    }

    fn bin_triangle(&mut self, index: u32) {
        let triangle = &self.triangles[index as usize];
        // Back faces are culled in the viewport, so they can't be seen or hit.
        if triangle.normal.dot(self.eye - triangle.position[0]) <= 0.0 {
            return;
        }
        let Some((min, max)) = self.screen_bounds(triangle.position) else {
            return;
        };
        let (width, height) = (self.viewport.width(), self.viewport.height());
        if max.x < 0.0 || max.y < 0.0 || min.x >= width || min.y >= height {
            return;
        }
        let tile =
            |value: f32, tiles: usize| ((value.max(0.0) / TILE_SIZE) as usize).min(tiles - 1);
        for y in tile(min.y, self.tiles_y)..=tile(max.y, self.tiles_y) {
            for x in tile(min.x, self.tiles_x)..=tile(max.x, self.tiles_x) {
                self.tiles[y * self.tiles_x + x].push(index);
            }
        }
    }

    /// Screen bounding box of a triangle, clipped to the near plane so
    /// triangles that pass behind the camera still count where they're visible.
    fn screen_bounds(&self, position: [Vec3; 3]) -> Option<(Vec2, Vec2)> {
        let clip = position.map(|p| self.view_proj * p.extend(1.0));
        let mut min = Vec2::splat(f32::INFINITY);
        let mut max = Vec2::splat(f32::NEG_INFINITY);
        let mut add = |point: Vec4| {
            let screen = self.clip_to_screen(point);
            min = min.min(screen);
            max = max.max(screen);
        };
        for i in 0..3 {
            let (a, b) = (clip[i], clip[(i + 1) % 3]);
            if a.z >= 0.0 {
                add(a);
            }
            if (a.z >= 0.0) != (b.z >= 0.0) {
                add(a + (b - a) * (a.z / (a.z - b.z)));
            }
        }
        (min.is_finite() && max.is_finite()).then_some((min, max))
    }

    fn triangles_in_screen_rect(&self, min: Vec2, max: Vec2) -> Vec<u32> {
        let (width, height) = (self.viewport.width(), self.viewport.height());
        if max.x < 0.0 || max.y < 0.0 || min.x >= width || min.y >= height {
            return Vec::new();
        }
        let tile =
            |value: f32, tiles: usize| ((value.max(0.0) / TILE_SIZE) as usize).min(tiles - 1);
        let mut found = Vec::new();
        for y in tile(min.y, self.tiles_y)..=tile(max.y, self.tiles_y) {
            for x in tile(min.x, self.tiles_x)..=tile(max.x, self.tiles_x) {
                found.extend_from_slice(&self.tiles[y * self.tiles_x + x]);
            }
        }
        found.sort_unstable();
        found.dedup();
        found
    }

    fn to_local(&self, point: Pos2) -> Vec2 {
        Vec2::new(point.x - self.viewport.min.x, point.y - self.viewport.min.y)
    }

    /// World size of `points` screen points at the depth of `position`.
    fn world_radius(&self, position: Vec3, points: f32) -> f32 {
        let depth = (position - self.eye).dot(self.forward).max(1e-6);
        points * self.world_per_point * depth
    }

    fn clip_to_screen(&self, clip: Vec4) -> Vec2 {
        Vec2::new(
            (clip.x / clip.w * 0.5 + 0.5) * self.viewport.width(),
            (0.5 - clip.y / clip.w * 0.5) * self.viewport.height(),
        )
    }

    /// Direction of the camera ray through a viewport-local screen point.
    fn ray(&self, screen: Vec2) -> Option<Vec3> {
        let clip_x = screen.x / self.viewport.width() * 2.0 - 1.0;
        let clip_y = 1.0 - screen.y / self.viewport.height() * 2.0;
        let near = self
            .inverse_view_proj
            .project_point3(Vec3::new(clip_x, clip_y, 0.0));
        let far = self
            .inverse_view_proj
            .project_point3(Vec3::new(clip_x, clip_y, 1.0));
        let direction = (far - near).normalize_or_zero();
        (direction != Vec3::ZERO && direction.is_finite()).then_some(direction)
    }

    /// Closest visible triangle under a screen point and its distance from the eye.
    fn hit(&self, screen: Vec2) -> Option<(u32, f32)> {
        if screen.x < 0.0
            || screen.y < 0.0
            || screen.x >= self.viewport.width()
            || screen.y >= self.viewport.height()
        {
            return None;
        }
        let direction = self.ray(screen)?;
        let tile = (screen.y / TILE_SIZE) as usize * self.tiles_x + (screen.x / TILE_SIZE) as usize;
        let mut closest: Option<(u32, f32)> = None;
        for &index in self.tiles.get(tile)? {
            let triangle = &self.triangles[index as usize];
            if let Some(distance) = intersect_triangle(self.eye, direction, triangle.position)
                && closest.is_none_or(|(_, best)| distance < best)
            {
                closest = Some((index, distance));
            }
        }
        closest
    }

    fn surface_hit(&self, screen: Vec2) -> Option<SurfaceHit> {
        let (triangle, distance) = self.hit(screen)?;
        let position = self.eye + self.ray(screen)? * distance;
        Some(SurfaceHit {
            triangle,
            position,
            normal: self.triangles[triangle as usize].smooth_normal(position),
        })
    }

    /// Whether `position` (on triangle `index`, seen at `screen`) is unoccluded.
    fn is_visible(&self, position: Vec3, screen: Vec2, index: u32) -> bool {
        match self.hit(screen) {
            Some((hit, _)) if hit == index => true,
            Some((_, distance)) => {
                let own = position.distance(self.eye);
                distance >= own * (1.0 - DEPTH_TOLERANCE)
            }
            // Nothing in front: a seam-margin texel just past a silhouette.
            None => true,
        }
    }
}

/// The texture being painted plus the stroke's bookkeeping.
struct PaintTarget<'a> {
    image: &'a mut Image,
    stroke: &'a mut Stroke,
    color: [f32; 3],
    opacity: f32,
    region: Option<ImageRegion>,
}

impl PaintTarget<'_> {
    /// Applies `coverage(world position, texel reach)` to every texel of
    /// `triangle` inside `bounds` (UV), or the whole triangle when bounds
    /// couldn't be computed. Texel reach is the world distance from a texel
    /// centre to its corners, so the brush can test footprint overlap.
    fn paint_triangle(
        &mut self,
        triangle: &Triangle,
        bounds: Option<(Vec2, Vec2)>,
        coverage: impl Fn(Vec3, f32) -> f32,
    ) {
        let size = Vec2::new(self.image.width as f32, self.image.height as f32);
        // Work in texel units so the seam margin is the same on both axes.
        let corners = triangle.uv.map(|uv| uv * size);
        let Some(margins) = edge_margins(corners, SEAM_MARGIN_TEXELS) else {
            return;
        };
        let world_area = (triangle.position[1] - triangle.position[0])
            .cross(triangle.position[2] - triangle.position[0])
            .length();
        let texel_area = cross(corners[1] - corners[0], corners[2] - corners[0]).abs();
        let texel_reach = (world_area / texel_area).sqrt() * std::f32::consts::FRAC_1_SQRT_2;
        let (mut lo, mut hi) = uv_bounds(&corners);
        lo -= Vec2::splat(SEAM_MARGIN_TEXELS);
        hi += Vec2::splat(SEAM_MARGIN_TEXELS);
        if let Some((brush_lo, brush_hi)) = bounds {
            // Grow by a texel so texels larger than the brush are still found.
            lo = lo.max(brush_lo * size - Vec2::ONE);
            hi = hi.min(brush_hi * size + Vec2::ONE);
        }
        if lo.x > hi.x || lo.y > hi.y {
            return;
        }

        let (min_x, max_x) = texel_range(lo.x, hi.x, self.image.width);
        let (min_y, max_y) = texel_range(lo.y, hi.y, self.image.height);
        for raw_y in min_y..=max_y {
            for raw_x in min_x..=max_x {
                let center = Vec2::new(raw_x as f32 + 0.5, raw_y as f32 + 0.5);
                let Some(weights) = barycentric_2d(center, corners) else {
                    return;
                };
                if weights.iter().zip(margins).any(|(w, margin)| *w < -margin) {
                    continue;
                }
                let amount = coverage(interpolate(triangle.position, weights), texel_reach);
                if amount > 0.0 {
                    let x = raw_x.rem_euclid(self.image.width as i64) as u32;
                    let y = raw_y.rem_euclid(self.image.height as i64) as u32;
                    self.apply(x, y, amount);
                }
            }
        }
    }

    fn apply(&mut self, x: u32, y: u32, amount: f32) {
        let texel = (y * self.image.width + x) as usize;
        let amount = amount.min(1.0);
        if amount <= self.stroke.coverage[texel] {
            return;
        }
        self.stroke.coverage[texel] = amount;

        let alpha = amount * self.opacity;
        let offset = texel * 4;
        for (channel, color) in self.color.iter().enumerate() {
            let original = srgb_to_linear(self.stroke.original[offset + channel]);
            self.image.rgba[offset + channel] =
                linear_to_srgb(original + (color - original) * alpha);
        }
        let pixel = ImageRegion::pixel(x, y);
        self.region = Some(self.region.map_or(pixel, |region| region.union(pixel)));
    }
}

fn aspect(viewport: Rect) -> f32 {
    viewport.width() / viewport.height().max(1.0)
}

/// 1 inside the hard core, easing to 0 at `radius`.
fn falloff(distance: f32, radius: f32, hardness: f32) -> f32 {
    if distance >= radius {
        return 0.0;
    }
    let inner = radius * hardness.clamp(0.0, 1.0);
    if distance <= inner {
        return 1.0;
    }
    let t = 1.0 - (distance - inner) / (radius - inner);
    t * t * (3.0 - 2.0 * t)
}

/// How far below zero each barycentric weight may go for a point to stay
/// within `margin` (same units as `corners`) of the triangle.
fn edge_margins(corners: [Vec2; 3], margin: f32) -> Option<[f32; 3]> {
    let doubled_area = cross(corners[1] - corners[0], corners[2] - corners[0]).abs();
    if doubled_area < 1e-12 {
        return None;
    }
    // Weight i is the distance to the opposite edge divided by that edge's height.
    Some([0, 1, 2].map(|i| {
        let edge = corners[(i + 2) % 3] - corners[(i + 1) % 3];
        margin * edge.length() / doubled_area
    }))
}

fn uv_bounds(points: &[Vec2]) -> (Vec2, Vec2) {
    points.iter().fold(
        (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
        |(lo, hi), point| (lo.min(*point), hi.max(*point)),
    )
}

fn texel_range(min: f32, max: f32, size: u32) -> (i64, i64) {
    let min = min.floor() as i64;
    let max = max.ceil() as i64;
    if max - min + 1 > size as i64 {
        let center = (min + max) / 2;
        (center - size as i64 / 2, center + size as i64 / 2)
    } else {
        (min, max)
    }
}

fn barycentric_2d(point: Vec2, triangle: [Vec2; 3]) -> Option<[f32; 3]> {
    let denominator = cross(triangle[1] - triangle[0], triangle[2] - triangle[0]);
    if denominator.abs() < 1e-12 {
        return None;
    }
    let b = cross(point - triangle[0], triangle[2] - triangle[0]) / denominator;
    let c = cross(triangle[1] - triangle[0], point - triangle[0]) / denominator;
    Some([1.0 - b - c, b, c])
}

/// Barycentric weights of `point` projected onto the triangle's plane.
fn barycentric_3d(point: Vec3, triangle: [Vec3; 3]) -> Option<[f32; 3]> {
    let v0 = triangle[1] - triangle[0];
    let v1 = triangle[2] - triangle[0];
    let v2 = point - triangle[0];
    let (d00, d01, d11) = (v0.dot(v0), v0.dot(v1), v1.dot(v1));
    let (d20, d21) = (v2.dot(v0), v2.dot(v1));
    let denominator = d00 * d11 - d01 * d01;
    if denominator.abs() < 1e-20 {
        return None;
    }
    let b = (d11 * d20 - d01 * d21) / denominator;
    let c = (d00 * d21 - d01 * d20) / denominator;
    Some([1.0 - b - c, b, c])
}

fn interpolate<T>(values: [T; 3], weights: [f32; 3]) -> T
where
    T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T> + Copy,
{
    values[0] * weights[0] + values[1] * weights[1] + values[2] * weights[2]
}

fn ray_plane(origin: Vec3, direction: Vec3, plane_point: Vec3, normal: Vec3) -> Option<Vec3> {
    let denominator = direction.dot(normal);
    if denominator.abs() < 1e-6 {
        return None;
    }
    let distance = (plane_point - origin).dot(normal) / denominator;
    (distance > 0.0).then_some(origin + direction * distance)
}

fn cross(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}

fn distance_to_segment(point: Vec2, from: Vec2, to: Vec2) -> f32 {
    let segment = to - from;
    let length_squared = segment.length_squared();
    if length_squared == 0.0 {
        return point.distance(from);
    }
    let t = ((point - from).dot(segment) / length_squared).clamp(0.0, 1.0);
    point.distance(from + segment * t)
}

fn distance_to_segment_3d(point: Vec3, from: Vec3, to: Vec3) -> f32 {
    let segment = to - from;
    let length_squared = segment.length_squared();
    if length_squared == 0.0 {
        return point.distance(from);
    }
    let t = ((point - from).dot(segment) / length_squared).clamp(0.0, 1.0);
    point.distance(from + segment * t)
}

/// Möller–Trumbore; returns the distance along the ray.
fn intersect_triangle(origin: Vec3, direction: Vec3, [a, b, c]: [Vec3; 3]) -> Option<f32> {
    let edge1 = b - a;
    let edge2 = c - a;
    let cross = direction.cross(edge2);
    let determinant = edge1.dot(cross);
    if determinant.abs() < 1e-12 {
        return None;
    }
    let inverse_determinant = determinant.recip();
    let from_a = origin - a;
    let u = from_a.dot(cross) * inverse_determinant;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = from_a.cross(edge1);
    let v = direction.dot(q) * inverse_determinant;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let distance = edge2.dot(q) * inverse_determinant;
    (distance >= 0.0).then_some(distance)
}

fn srgb_float_to_linear(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::{Stroke, StrokePainter};
    use crate::mesh::{Image, ImageRegion, Material, Mesh, SubMesh, Vertex};
    use crate::painting::brush::Brush;
    use crate::renderer::camera::OrbitCamera;
    use egui::{Pos2, Rect};
    use glam::Vec3;

    const TEXTURE_SIZE: u32 = 128;

    fn test_camera() -> OrbitCamera {
        OrbitCamera {
            target: Vec3::ZERO,
            distance: 3.0,
            radius: 1.0,
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 45.0_f32.to_radians(),
        }
    }

    fn viewport() -> Rect {
        Rect::from_min_max(Pos2::ZERO, Pos2::new(256.0, 256.0))
    }

    fn blank_material(name: &str) -> Material {
        Material {
            name: name.into(),
            base_color: Image::filled(TEXTURE_SIZE, TEXTURE_SIZE, [0, 0, 0, 255]),
        }
    }

    /// Corner positions and UVs of one triangle.
    type TestTriangle = ([[f32; 3]; 3], [[f32; 2]; 3]);

    fn mesh_from_triangles(triangles: &[TestTriangle]) -> Mesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (positions, uvs) in triangles {
            let base = vertices.len() as u32;
            let [a, b, c] = positions.map(Vec3::from);
            let normal = (b - a).cross(c - a).normalize().to_array();
            for index in 0..3 {
                vertices.push(Vertex {
                    position: positions[index],
                    normal,
                    uv: uvs[index],
                });
            }
            indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
        Mesh {
            submeshes: vec![SubMesh {
                indices: 0..indices.len() as u32,
                material: 0,
            }],
            vertices,
            indices,
            materials: vec![blank_material("Test")],
        }
    }

    /// Two triangles facing the camera, in separate UV islands.
    fn test_mesh() -> Mesh {
        mesh_from_triangles(&[
            (
                [[-1.0, -1.0, 0.0], [0.0, -1.0, 0.0], [-0.5, 1.0, 0.0]],
                [[0.0, 0.0], [0.45, 0.0], [0.225, 1.0]],
            ),
            (
                [[0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.5, 1.0, 0.0]],
                [[0.55, 0.0], [1.0, 0.0], [0.775, 1.0]],
            ),
        ])
    }

    /// A large floor (y = 0) with UVs spanning 0..1.
    fn floor_mesh(half_size: f32) -> Mesh {
        let s = half_size;
        mesh_from_triangles(&[
            (
                [[-s, 0.0, s], [s, 0.0, s], [s, 0.0, -s]],
                [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            ),
            (
                [[-s, 0.0, s], [s, 0.0, -s], [-s, 0.0, -s]],
                [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            ),
        ])
    }

    fn brush(size: f32, screen_space: bool) -> Brush {
        Brush {
            size,
            hardness: 1.0,
            opacity: 1.0,
            color: [1.0, 0.0, 0.0],
            screen_space,
            paint_over_edges: true,
        }
    }

    fn paint(
        mesh: &mut Mesh,
        camera: &OrbitCamera,
        points: &[Pos2],
        brush: &Brush,
    ) -> Option<ImageRegion> {
        let painter = StrokePainter::new(mesh, camera, viewport());
        let mut stroke = Stroke::begin(mesh, 0).unwrap();
        let mut region: Option<ImageRegion> = None;
        let segments: Vec<_> = if points.len() == 1 {
            vec![(points[0], points[0])]
        } else {
            points.windows(2).map(|pair| (pair[0], pair[1])).collect()
        };
        for (from, to) in segments {
            if let Some(changed) = painter.paint_segment(&mut stroke, mesh, from, to, brush) {
                region = Some(region.map_or(changed, |r| r.union(changed)));
            }
        }
        region
    }

    /// (x, y, red) of every painted texel.
    fn painted(mesh: &Mesh) -> Vec<(u32, u32, u8)> {
        mesh.materials[0]
            .base_color
            .rgba
            .chunks_exact(4)
            .enumerate()
            .filter(|(_, pixel)| pixel[0] > 0)
            .map(|(i, pixel)| (i as u32 % TEXTURE_SIZE, i as u32 / TEXTURE_SIZE, pixel[0]))
            .collect()
    }

    #[test]
    fn segment_paints_a_continuous_stroke_and_reports_its_region() {
        for screen_space in [true, false] {
            let mut mesh = test_mesh();
            let region = paint(
                &mut mesh,
                &test_camera(),
                &[Pos2::new(85.0, 128.0), Pos2::new(171.0, 128.0)],
                &brush(16.0, screen_space),
            )
            .expect("stroke should paint");

            let texels = painted(&mesh);
            let min_x = texels.iter().map(|t| t.0).min().unwrap();
            let max_x = texels.iter().map(|t| t.0).max().unwrap();
            assert!(min_x < 50 && max_x > 80, "screen_space={screen_space}");
            for (x, y, _) in texels {
                assert!(x >= region.x && x < region.x + region.width);
                assert!(y >= region.y && y < region.y + region.height);
            }
        }
    }

    #[test]
    fn overlapping_segments_do_not_build_up_into_dots() {
        let mut mesh = test_mesh();
        let mut soft = brush(24.0, true);
        soft.hardness = 0.0;
        soft.opacity = 0.5;
        // Many short segments, like a mouse stroke sampled every frame.
        let points: Vec<_> = (0..20)
            .map(|i| Pos2::new(60.0 + i as f32 * 2.0, 128.0))
            .collect();
        paint(&mut mesh, &test_camera(), &points, &soft);

        let half = crate::mesh::linear_to_srgb(0.5);
        let max_red = painted(&mesh).iter().map(|t| t.2).max().unwrap();
        assert!(
            max_red <= half,
            "max {max_red} exceeds a single pass at 50% ({half})"
        );
    }

    #[test]
    fn large_surface_brush_does_not_bleed_to_adjacent_uv_island() {
        let mut mesh = test_mesh();
        paint(
            &mut mesh,
            &test_camera(),
            &[Pos2::new(91.0, 128.0)],
            &brush(36.0, false),
        );

        let texels = painted(&mesh);
        assert!(texels.iter().any(|t| t.0 < 58));
        assert!(!texels.iter().any(|t| t.0 >= 70));
    }

    #[test]
    fn brush_does_not_paint_a_selected_material_hidden_behind_another_surface() {
        for screen_space in [true, false] {
            let mut mesh = test_mesh();
            for vertex in &mut mesh.vertices[..3] {
                vertex.position[2] = -0.1;
            }
            for index in 0..3 {
                mesh.vertices[index + 3].position = mesh.vertices[index].position;
                mesh.vertices[index + 3].position[2] = 0.0;
            }
            mesh.submeshes = vec![
                SubMesh {
                    indices: 0..3,
                    material: 0,
                },
                SubMesh {
                    indices: 3..6,
                    material: 1,
                },
            ];
            mesh.materials.push(blank_material("Front"));

            let region = paint(
                &mut mesh,
                &test_camera(),
                &[Pos2::new(110.0, 128.0)],
                &brush(6.0, screen_space),
            );
            assert!(region.is_none(), "screen_space={screen_space}");
            assert!(painted(&mesh).is_empty());
        }
    }

    #[test]
    fn paints_when_the_camera_is_close_to_a_large_surface() {
        // The floor extends far behind the camera, and the depth across each
        // triangle varies a lot: both used to make the brush miss.
        let camera = OrbitCamera {
            distance: 0.3,
            pitch: 0.5,
            radius: 10.0,
            ..test_camera()
        };
        for screen_space in [true, false] {
            let mut mesh = floor_mesh(10.0);
            let region = paint(
                &mut mesh,
                &camera,
                &[Pos2::new(128.0, 128.0)],
                &brush(20.0, screen_space),
            );
            assert!(region.is_some(), "screen_space={screen_space}");
        }
    }

    #[test]
    fn surface_brush_stays_round_on_surfaces_seen_at_an_angle() {
        // Looking along a floor at a shallow angle: a screen-space circle
        // stretches into a long ellipse on the floor, the surface brush doesn't.
        let camera = OrbitCamera {
            distance: 3.0,
            pitch: 0.25,
            ..test_camera()
        };
        let aspect_ratio = |screen_space: bool| {
            let mut mesh = floor_mesh(2.0);
            paint(
                &mut mesh,
                &camera,
                &[Pos2::new(128.0, 128.0)],
                &brush(30.0, screen_space),
            );
            let texels = painted(&mesh);
            let width = texels.iter().map(|t| t.0).max().unwrap()
                - texels.iter().map(|t| t.0).min().unwrap();
            let depth = texels.iter().map(|t| t.1).max().unwrap()
                - texels.iter().map(|t| t.1).min().unwrap();
            depth as f32 / width as f32
        };
        let screen = aspect_ratio(true);
        let surface = aspect_ratio(false);
        assert!(screen > 2.5, "screen-space footprint ratio {screen}");
        assert!(
            (0.8..1.25).contains(&surface),
            "surface footprint ratio {surface}"
        );
    }

    /// Which cells of the cube's 3x2 UV atlas (one per face) got paint.
    fn painted_cube_cells(mesh: &Mesh) -> Vec<(u32, u32)> {
        let image = &mesh.materials[0].base_color;
        let mut cells: Vec<_> = image
            .rgba
            .chunks_exact(4)
            .enumerate()
            .filter(|(_, pixel)| pixel[0] == 255 && pixel[1] == 0)
            .map(|(i, _)| {
                let (x, y) = (i as u32 % image.width, i as u32 / image.width);
                (x * 3 / image.width, y * 2 / image.height)
            })
            .collect();
        cells.sort_unstable();
        cells.dedup();
        cells
    }

    #[test]
    fn stroke_across_a_hard_edge_stays_on_its_face_when_asked() {
        // Looking at the cube's +Z / +X edge; drag from the +Z face over the edge.
        let camera = OrbitCamera {
            yaw: 0.6,
            ..test_camera()
        };
        let points = [
            Pos2::new(90.0, 128.0),
            Pos2::new(140.0, 128.0),
            Pos2::new(190.0, 128.0),
        ];
        for screen_space in [true, false] {
            let mut over_edges = brush(20.0, screen_space);
            let mut mesh = crate::mesh::primitives::cube();
            paint(&mut mesh, &camera, &points, &over_edges);
            assert_eq!(
                painted_cube_cells(&mesh),
                vec![(0, 0), (2, 0)],
                "screen_space={screen_space}"
            );

            over_edges.paint_over_edges = false;
            let mut mesh = crate::mesh::primitives::cube();
            paint(&mut mesh, &camera, &points, &over_edges);
            assert_eq!(
                painted_cube_cells(&mesh),
                vec![(0, 0)],
                "screen_space={screen_space}"
            );
        }
    }

    #[test]
    fn face_restriction_follows_smooth_surfaces_across_triangles() {
        // The floor's two triangles share smooth normals along their diagonal,
        // so they are one face: a restricted stroke must cross the diagonal.
        let camera = OrbitCamera {
            pitch: 1.2,
            ..test_camera()
        };
        let mut restricted = brush(16.0, false);
        restricted.paint_over_edges = false;
        let mut mesh = floor_mesh(2.0);
        paint(
            &mut mesh,
            &camera,
            &[Pos2::new(90.0, 90.0), Pos2::new(166.0, 166.0)],
            &restricted,
        );
        let sides: Vec<bool> = painted(&mesh)
            .iter()
            .map(|(x, y, _)| x + y < TEXTURE_SIZE)
            .collect();
        assert!(sides.contains(&true) && sides.contains(&false));
    }

    #[test]
    fn surface_brush_outline_lies_on_the_surface() {
        let mesh = test_mesh();
        let painter = StrokePainter::new(&mesh, &test_camera(), viewport());
        let cursor = Pos2::new(91.0, 128.0);

        // Facing the camera, the disc looks like the screen circle.
        let outline = painter.brush_outline(cursor, &brush(30.0, false)).unwrap();
        for point in &outline {
            let distance = point.distance(cursor);
            assert!(
                (distance - 15.0).abs() < 1.0,
                "outline point {distance} from cursor"
            );
        }
        // Off the model there is nothing to project onto.
        assert!(
            painter
                .brush_outline(Pos2::new(5.0, 5.0), &brush(30.0, false))
                .is_none()
        );

        // Seen at a shallow angle, the disc is squashed vertically.
        let camera = OrbitCamera {
            pitch: 0.25,
            ..test_camera()
        };
        let floor = floor_mesh(2.0);
        let painter = StrokePainter::new(&floor, &camera, viewport());
        let outline = painter
            .brush_outline(Pos2::new(128.0, 128.0), &brush(30.0, false))
            .unwrap();
        let height = outline.iter().map(|p| p.y).fold(f32::MIN, f32::max)
            - outline.iter().map(|p| p.y).fold(f32::MAX, f32::min);
        let width = outline.iter().map(|p| p.x).fold(f32::MIN, f32::max)
            - outline.iter().map(|p| p.x).fold(f32::MAX, f32::min);
        assert!(height < width * 0.5, "outline {width} x {height}");
    }
}
