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

use crate::document::{CLEAR, Change, Document, Layer, Plane, Tile};
use crate::mesh::{Channel, Encoding, Image, ImageRegion, Mesh};
use crate::painting::brush::{Brush, FillMode, Tool};
use crate::painting::history;
use crate::painting::raster::{
    SEAM_MARGIN_TEXELS, barycentric_2d, barycentric_3d, cross, edge_margins, interpolate,
    texel_range, uv_bounds,
};
use crate::renderer::camera::OrbitCamera;
use egui::{Pos2, Rect};
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

const TILE_SIZE: f32 = 32.0;
/// The surface brush skips triangles facing more than ~100 degrees away from
/// the surface under the cursor: it wraps around edges but not through walls.
const SURFACE_MIN_FACING: f32 = -0.2;
/// Relative depth tolerance when deciding whether a texel is hidden.
const DEPTH_TOLERANCE: f32 = 1e-3;
/// Two triangles sharing an edge are one smooth surface when their vertex
/// normals along it agree this closely (cosine); otherwise it's a hard edge.
const SMOOTH_EDGE_COSINE: f32 = 0.99;
/// How far the blur brush spreads paint, as a share of the brush radius.
const BLUR_SHARE: f32 = 0.25;
/// The widest blur, as the Gaussian's sigma in texels.
const MAX_BLUR_TEXELS: f32 = 8.0;
/// Blur works on square tiles of this many texels (plus the kernel around them).
const BLUR_TILE: u32 = 32;
/// Most steps one smear segment is split into (each moves paint up to a texel).
const MAX_SMEAR_STEPS: usize = 32;
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
/// Which texels of a texture the model's UVs use, row by row.
type UvMask = Arc<Vec<bool>>;
/// A UV edge within one material slot, by the exact bits of its two end UVs.
type UvEdgeKey = (usize, [u32; 2], [u32; 2]);

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

/// What a stroke remembers about one material slot it painted on.
struct SlotStroke {
    width: u32,
    height: u32,
    /// The layer texture as it was before the stroke started.
    original: Vec<u8>,
    /// Which history tiles the stroke changed, row by row.
    touched_tiles: Vec<bool>,
    /// Highest brush coverage (0..1) each texel received during this stroke.
    coverage: Vec<f32>,
}

impl SlotStroke {
    fn new(image: &Image) -> Self {
        Self {
            width: image.width,
            height: image.height,
            original: image.rgba.clone(),
            touched_tiles: vec![false; (tiles_x(image.width) * tiles_y(image.height)) as usize],
            coverage: vec![0.0; image.rgba.len() / 4],
        }
    }

    /// The pixels of every touched tile, as they were before the stroke.
    fn into_tiles(self) -> Vec<Tile> {
        let columns = tiles_x(self.width);
        self.touched_tiles
            .iter()
            .enumerate()
            .filter(|(_, touched)| **touched)
            .map(|(index, _)| {
                let (tile_x, tile_y) = (index as u32 % columns, index as u32 / columns);
                let region = ImageRegion {
                    x: tile_x * history::TILE_SIZE,
                    y: tile_y * history::TILE_SIZE,
                    width: history::TILE_SIZE.min(self.width - tile_x * history::TILE_SIZE),
                    height: history::TILE_SIZE.min(self.height - tile_y * history::TILE_SIZE),
                };
                let row_bytes = (region.width * 4) as usize;
                let mut pixels = Vec::with_capacity(row_bytes * region.height as usize);
                for y in region.y..region.y + region.height {
                    let start = ((y * self.width + region.x) * 4) as usize;
                    pixels.extend_from_slice(&self.original[start..start + row_bytes]);
                }
                Tile { region, pixels }
            })
            .collect()
    }
}

/// One continuous brush stroke into one layer, on every slot the layer
/// targets: into every channel the brush paints, or into the layer's mask.
pub struct Stroke {
    layer: u64,
    /// Paints the layer's mask instead of its channels.
    on_mask: bool,
    /// Per material slot and plane, created when the stroke first reaches it.
    slots: Vec<[Option<SlotStroke>; Plane::COUNT]>,
    /// When the brush may not paint over edges: which triangles belong to the
    /// face the stroke started on. Found on the first segment that hits the model.
    face: Option<Vec<bool>>,
}

impl Stroke {
    /// Starts a stroke on the layer at `layer_index`. The base layer can't be painted.
    /// `None` for the base layer, and for fill layers, which are painted
    /// through their mask.
    pub fn begin(document: &Document, layer_index: usize) -> Option<Self> {
        Self::start(document, layer_index, false).filter(|_| {
            document
                .layers
                .get(layer_index)
                .is_some_and(|layer| layer.fill.is_none())
        })
    }

    fn start(document: &Document, layer_index: usize, on_mask: bool) -> Option<Self> {
        let layer = document.layers.get(layer_index)?;
        if layer.is_base() {
            return None;
        }
        Some(Self {
            layer: layer.id,
            on_mask,
            slots: (0..document.mesh.materials.len())
                .map(|_| Default::default())
                .collect(),
            face: None,
        })
    }

    /// Starts a stroke on the mask of the layer at `layer_index`: painting
    /// shows the layer there, erasing hides it. `None` if it has no mask, or
    /// its mask is made by a generator.
    pub fn begin_on_mask(document: &Document, layer_index: usize) -> Option<Self> {
        let layer = document.layers.get(layer_index)?;
        layer.mask?;
        if layer.mask_generator.is_some() {
            return None;
        }
        Self::start(document, layer_index, true)
    }

    /// Whether the stroke paints the layer's mask.
    pub fn on_mask(&self) -> bool {
        self.on_mask
    }

    /// Id of the layer being painted.
    pub fn layer(&self) -> u64 {
        self.layer
    }

    /// Whether the stroke has changed any pixels yet.
    pub fn has_changes(&self) -> bool {
        self.slots
            .iter()
            .flatten()
            .flatten()
            .any(|slot| slot.touched_tiles.contains(&true))
    }

    /// Ends the stroke. Returns the changes that undo it (empty if nothing changed).
    pub fn finish(self) -> Vec<Change> {
        let layer = self.layer;
        let mut changes = Vec::new();
        for (material, planes) in self.slots.into_iter().enumerate() {
            for (plane, slot) in Plane::ALL.into_iter().zip(planes) {
                let tiles = slot.map(SlotStroke::into_tiles).unwrap_or_default();
                if !tiles.is_empty() {
                    changes.push(Change::Pixels {
                        layer,
                        material,
                        plane,
                        tiles,
                    });
                }
            }
        }
        changes
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
    /// View depth of the orbit centre, where a brush that keeps its size on
    /// the model is measured.
    focus_depth: f32,
    triangles: Vec<Triangle>,
    tiles_x: usize,
    tiles_y: usize,
    /// Indices of front-facing triangles overlapping each screen tile.
    tiles: Vec<Vec<u32>>,
    /// Every triangle edge, sorted by key, to find neighbours. Built on first use.
    edges: OnceLock<Vec<(EdgeKey, u32)>>,
    /// Per (material, width, height): which texels the model's UVs use.
    /// Built when the blur brush first needs it.
    uv_masks: Mutex<HashMap<(usize, u32, u32), UvMask>>,
    /// What `hit` finds at the centre of each viewport pixel, once asked
    /// (see `pixel_hit`). Texels on the same pixel share one lookup, also
    /// across the threads that paint a segment together.
    pixel_hits: OnceLock<Vec<AtomicU64>>,
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
            focus_depth: camera.distance.max(1e-6),
            triangles: Vec::new(),
            tiles_x,
            tiles_y,
            tiles: vec![Vec::new(); tiles_x * tiles_y],
            edges: OnceLock::new(),
            uv_masks: Mutex::default(),
            pixel_hits: OnceLock::new(),
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

    /// Paints one segment of `stroke` into `layer` between two screen
    /// positions (in points). `slot_sizes` gives each slot's texture size.
    /// Returns the texel region that changed in each slot and channel.
    pub fn paint_segment(
        &self,
        stroke: &mut Stroke,
        layer: &mut Layer,
        slot_sizes: &[(u32, u32)],
        from: Pos2,
        to: Pos2,
        brush: &Brush,
    ) -> Vec<(usize, Plane, ImageRegion)> {
        if layer.id != stroke.layer || self.viewport.height() <= 0.0 {
            return Vec::new();
        }
        let from = self.to_local(from);
        let to = self.to_local(to);
        if brush.paint_over_edges {
            stroke.face = None;
        } else if stroke.face.is_none() {
            let Some(start) = self.surface_hit(from).or_else(|| self.surface_hit(to)) else {
                return Vec::new();
            };
            stroke.face = Some(self.connected_face(start.triangle));
        }
        let radius = (brush.size * 0.5).max(0.5);
        let mut target = LayerTarget::new(layer, stroke, slot_sizes, brush);
        match brush.tool {
            Tool::Smear => {
                target.effect = Effect::Smear(self.smear_motion(from, to, radius, brush))
            }
            Tool::Blur => {
                let reach = self.surface_hit(to).map_or(0.0, |hit| {
                    BLUR_SHARE * self.brush_world_radius(hit.position, radius, brush)
                });
                // The blur builds up with the distance the brush travels (a
                // full dab per brush radius), not with how often the mouse
                // reports a move. A click is one full dab. Blur passes add up
                // as variance, hence the square root.
                let travel = if from == to {
                    1.0
                } else {
                    (from.distance(to) / radius).min(1.0)
                };
                target.effect = Effect::Blur(reach * travel.sqrt());
                // Blur only within the UV islands, so paint doesn't drain
                // into the unused texture space around them.
                target.uv_masks = slot_sizes
                    .iter()
                    .enumerate()
                    .map(|(material, &(width, height))| {
                        (target.layer.targets.get(material) == Some(&true))
                            .then(|| self.uv_mask(material, width, height))
                    })
                    .collect();
            }
            Tool::Paint | Tool::Fill => {}
        }
        if brush.screen_space {
            self.paint_screen_segment(&mut target, from, to, radius, brush.hardness);
        } else {
            self.paint_surface_segment(&mut target, from, to, radius, brush);
        }
        target.filter();
        target.changed()
    }

    /// Fills part of `layer` with the brush's values, as chosen by the
    /// brush's fill mode; `at` (screen points) picks the material or UV
    /// island. Returns the texel region that changed in each slot and channel.
    pub fn fill(
        &self,
        stroke: &mut Stroke,
        layer: &mut Layer,
        slot_sizes: &[(u32, u32)],
        at: Pos2,
        brush: &Brush,
    ) -> Vec<(usize, Plane, ImageRegion)> {
        if layer.id != stroke.layer {
            return Vec::new();
        }
        let hit = self.surface_hit(self.to_local(at));
        let mut target = LayerTarget::new(layer, stroke, slot_sizes, brush);
        match brush.fill_mode {
            FillMode::Layer => {
                for material in 0..slot_sizes.len() {
                    if let Some(mut slot) = target.slot(material) {
                        slot.fill_all();
                    }
                }
            }
            FillMode::Material => {
                if let Some(hit) = hit
                    && let Some(mut slot) =
                        target.slot(self.triangles[hit.triangle as usize].material)
                {
                    slot.fill_all();
                }
            }
            FillMode::UvIsland => {
                let Some(hit) = hit else {
                    return Vec::new();
                };
                let material = self.triangles[hit.triangle as usize].material;
                let Some(mut slot) = target.slot(material) else {
                    return Vec::new();
                };
                for index in self.uv_island(hit.triangle) {
                    slot.paint_triangle(&self.triangles[index as usize], None, |_, _| 1.0);
                }
            }
        }
        target.changed()
    }

    /// Which texels of a `width` x `height` texture the triangles of
    /// `material` cover (with the seam margin), as row-major flags.
    fn uv_mask(&self, material: usize, width: u32, height: u32) -> UvMask {
        let key = (material, width, height);
        let masks = || self.uv_masks.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(mask) = masks().get(&key) {
            return Arc::clone(mask);
        }
        let size = Vec2::new(width as f32, height as f32);
        let mut mask = vec![false; width as usize * height as usize];
        for triangle in self.triangles.iter().filter(|t| t.material == material) {
            let corners = triangle.uv.map(|uv| uv * size);
            let Some(margins) = edge_margins(corners, SEAM_MARGIN_TEXELS) else {
                continue;
            };
            let (lo, hi) = uv_bounds(&corners);
            let (min_x, max_x) =
                texel_range(lo.x - SEAM_MARGIN_TEXELS, hi.x + SEAM_MARGIN_TEXELS, width);
            let (min_y, max_y) =
                texel_range(lo.y - SEAM_MARGIN_TEXELS, hi.y + SEAM_MARGIN_TEXELS, height);
            for raw_y in min_y..=max_y {
                for raw_x in min_x..=max_x {
                    let center = Vec2::new(raw_x as f32 + 0.5, raw_y as f32 + 0.5);
                    let Some(weights) = barycentric_2d(center, corners) else {
                        continue;
                    };
                    if weights.iter().zip(margins).all(|(w, margin)| *w >= -margin) {
                        let x = raw_x.rem_euclid(width as i64) as usize;
                        let y = raw_y.rem_euclid(height as i64) as usize;
                        mask[y * width as usize + x] = true;
                    }
                }
            }
        }
        let mask = Arc::new(mask);
        masks().insert(key, Arc::clone(&mask));
        mask
    }

    /// The triangles in the same UV island as `start`: same material slot,
    /// joined through edges that share both UV coordinates.
    fn uv_island(&self, start: u32) -> Vec<u32> {
        let key = |uv: Vec2| [uv.x.to_bits(), uv.y.to_bits()];
        let mut edges: Vec<(UvEdgeKey, u32)> = Vec::new();
        for (index, triangle) in self.triangles.iter().enumerate() {
            for corner in 0..3 {
                let a = key(triangle.uv[corner]);
                let b = key(triangle.uv[(corner + 1) % 3]);
                edges.push(((triangle.material, a.min(b), a.max(b)), index as u32));
            }
        }
        edges.sort_unstable();

        // Union-find over triangles that share an edge.
        let mut parent: Vec<u32> = (0..self.triangles.len() as u32).collect();
        fn root(parent: &mut [u32], mut i: u32) -> u32 {
            while parent[i as usize] != i {
                parent[i as usize] = parent[parent[i as usize] as usize];
                i = parent[i as usize];
            }
            i
        }
        for group in edges.chunk_by(|a, b| a.0 == b.0) {
            for pair in group.windows(2) {
                let (a, b) = (root(&mut parent, pair[0].1), root(&mut parent, pair[1].1));
                parent[a as usize] = b;
            }
        }
        let island = root(&mut parent, start);
        (0..self.triangles.len() as u32)
            .filter(|&index| root(&mut parent, index) == island)
            .collect()
    }

    /// How far the paint moves on the model during a smear segment: from
    /// the surface under `from` to the surface under `to`. Zero when either
    /// misses, or when they lie on different surfaces (far apart in 3D).
    fn smear_motion(&self, from: Vec2, to: Vec2, radius: f32, brush: &Brush) -> Vec3 {
        let (Some(start), Some(end)) = (self.surface_hit(from), self.surface_hit(to)) else {
            return Vec3::ZERO;
        };
        let motion = end.position - start.position;
        let reach = self.brush_world_radius(end.position, radius + from.distance(to), brush);
        if motion.length() > 2.0 * reach {
            Vec3::ZERO
        } else {
            motion
        }
    }

    /// Outline of the surface brush under the cursor: a disc lying on the
    /// surface, in screen points. `None` when the cursor isn't over the model.
    pub fn brush_outline(&self, cursor: Pos2, brush: &Brush) -> Option<Vec<Pos2>> {
        let hit = self.surface_hit(self.to_local(cursor))?;
        let world_radius =
            self.brush_world_radius(hit.position, (brush.size * 0.5).max(0.5), brush);
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
        target: &mut LayerTarget,
        from: Vec2,
        to: Vec2,
        radius: f32,
        hardness: f32,
    ) {
        let brush = ScreenBrush {
            from,
            to,
            radius,
            hardness,
        };
        let min = from.min(to) - Vec2::splat(radius);
        let max = from.max(to) + Vec2::splat(radius);
        // Painting and erasing need no neighbouring texels, so their coverage
        // is worked out on all CPU cores; blur and smear go one by one.
        let in_parallel = matches!(target.effect, Effect::Paint | Effect::Erase);
        let mut jobs = Vec::new();
        for index in self.triangles_in_screen_rect(min, max) {
            let triangle = &self.triangles[index as usize];
            if !target.stroke.allows(index) {
                continue;
            }
            // Only the part of the triangle under the brush rectangle matters.
            // Triangles can cover far more of the texture than the brush does.
            let Some(bounds) = self.uv_bounds_in_screen_rect(triangle, min, max) else {
                continue;
            };
            if in_parallel {
                // Textures are only made for slots that get painted, later.
                let Some((width, height)) = target.slot_size(triangle.material) else {
                    continue;
                };
                let (lo, hi) = bounds;
                let area = (hi - lo) * Vec2::new(width as f32, height as f32);
                jobs.push(ScreenJob {
                    index,
                    bounds,
                    texels: (area.x.max(1.0) * area.y.max(1.0)).min(1e9) as usize,
                });
            } else if let Some(mut slot) = target.slot(triangle.material) {
                slot.paint_triangle(triangle, Some(bounds), |position, texel_reach| {
                    self.screen_coverage(&brush, position, texel_reach, index)
                });
            }
        }
        if in_parallel {
            self.paint_screen_jobs(target, jobs, &brush);
        }
    }

    /// How much of the screen brush covers `position` on triangle `index`.
    fn screen_coverage(
        &self,
        brush: &ScreenBrush,
        position: Vec3,
        texel_reach: f32,
        index: u32,
    ) -> f32 {
        let clip = self.view_proj * position.extend(1.0);
        if clip.w <= 0.0 || clip.z < 0.0 {
            return 0.0;
        }
        let screen = self.clip_to_screen(clip);
        let depth = (position - self.eye).dot(self.forward).max(1e-6);
        let reach = texel_reach / (self.world_per_point * depth);
        let distance = distance_to_segment(screen, brush.from, brush.to) - reach;
        let coverage = falloff(distance.max(0.0), brush.radius, brush.hardness);
        if coverage <= 0.0 || !self.is_visible(position, screen, index) {
            return 0.0;
        }
        coverage
    }

    /// Paints the screen brush onto the triangles of `jobs`. Triangles that
    /// cover much of the texture are cut into pieces, so that the cores share
    /// them; the pieces are done in batches to keep the memory they need small.
    fn paint_screen_jobs(
        &self,
        target: &mut LayerTarget,
        jobs: Vec<ScreenJob>,
        brush: &ScreenBrush,
    ) {
        /// Texels worth one unit of work, and the most a batch holds.
        const PIECE_TEXELS: usize = 1 << 16;
        const BATCH_TEXELS: usize = 1 << 22;
        let mut pieces = Vec::new();
        for job in jobs {
            let count = job.texels.div_ceil(PIECE_TEXELS).clamp(1, 64);
            let (lo, hi) = job.bounds;
            for part in 0..count {
                let v = |k: usize| lo.y + (hi.y - lo.y) * k as f32 / count as f32;
                pieces.push(ScreenJob {
                    bounds: (Vec2::new(lo.x, v(part)), Vec2::new(hi.x, v(part + 1))),
                    texels: job.texels.div_ceil(count),
                    ..job
                });
            }
        }
        let mut start = 0;
        while start < pieces.len() {
            let mut end = start + 1;
            let mut total = pieces[start].texels;
            while end < pieces.len() && total + pieces[end].texels <= BATCH_TEXELS {
                total += pieces[end].texels;
                end += 1;
            }
            self.paint_screen_batch(target, &pieces[start..end], brush);
            start = end;
        }
    }

    fn paint_screen_batch(
        &self,
        target: &mut LayerTarget,
        batch: &[ScreenJob],
        brush: &ScreenBrush,
    ) {
        let sizes = target.slot_sizes;
        let planes: Vec<Plane> = target.values.iter().map(|(plane, _)| *plane).collect();
        // What the stroke already covered, per slot and painted plane. A slot
        // whose stroke buffers don't exist yet has nothing to skip.
        let covered: Vec<Vec<&[f32]>> = target
            .stroke
            .slots
            .iter()
            .map(|slot| {
                planes
                    .iter()
                    .map(|plane| slot[plane.index()].as_ref().map(|s| s.coverage.as_slice()))
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_default()
            })
            .collect();
        let results = parallel_filter_map_dynamic(batch, 4, |piece| {
            let triangle = &self.triangles[piece.index as usize];
            let (width, height) = *sizes.get(triangle.material)?;
            let scan = TriangleScan::new(width, height, triangle, Some(piece.bounds))?;
            let covered = &covered[triangle.material];
            let mut texels = Vec::new();
            for texel in scan.texels() {
                let x = texel.raw_x.rem_euclid(width as i64) as u32;
                let y = texel.raw_y.rem_euclid(height as i64) as u32;
                let offset = (y * width + x) as usize;
                if !covered.is_empty() && covered.iter().all(|plane| plane[offset] >= 1.0) {
                    continue;
                }
                let position = interpolate(triangle.position, texel.weights);
                let amount = self.screen_coverage(brush, position, scan.reach, piece.index);
                if amount > 0.0 {
                    texels.push((x, y, amount));
                }
            }
            (!texels.is_empty()).then_some((piece.index, texels))
        });
        // Apply them in the order the triangles come, as one by one would.
        for (index, texels) in results {
            let material = self.triangles[index as usize].material;
            if let Some(mut slot) = target.slot(material) {
                for (x, y, amount) in texels {
                    slot.apply(x, y, amount);
                }
            }
        }
    }

    /// Surface painting: the brush is a sphere around the surface under the
    /// cursor, sized so it matches the screen circle there. It keeps a constant
    /// size on the model and wraps around edges.
    fn paint_surface_segment(
        &self,
        target: &mut LayerTarget,
        from: Vec2,
        to: Vec2,
        radius: f32,
        brush: &Brush,
    ) {
        let hardness = brush.hardness;
        let start_hit = self.surface_hit(from);
        let Some(end_hit) = self.surface_hit(to).or_else(|| self.surface_hit(from)) else {
            return;
        };
        let (end, normal) = (end_hit.position, end_hit.normal);
        let world_radius = self.brush_world_radius(end, radius, brush);
        let mut start = start_hit.as_ref().map_or(end, |hit| hit.position);
        // Sweeping across a silhouette can make the two ends far apart in 3D;
        // a capsule between them would cut through empty space. Use a dab.
        let screen_length = self.world_radius(end, from.distance(to));
        if start.distance(end) > 2.0 * screen_length + world_radius {
            start = end;
        }

        let mut seeds = vec![end_hit.triangle];
        if let Some(hit) = start_hit.filter(|_| start != end) {
            seeds.push(hit.triangle);
        }
        for index in self.surface_patch(&seeds, start, end, world_radius, normal) {
            let triangle = &self.triangles[index as usize];
            if !target.stroke.allows(index)
                || target.layer.targets.get(triangle.material) != Some(&true)
            {
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

            let Some(mut slot) = target.slot(triangle.material) else {
                continue;
            };
            slot.paint_triangle(triangle, bounds, |position, texel_reach| {
                let distance = distance_to_segment_3d(position, start, end) - texel_reach;
                falloff(distance.max(0.0), world_radius, hardness)
            });
        }
    }

    /// All triangles reachable from `start` without crossing a hard edge.
    fn connected_face(&self, start: u32) -> Vec<bool> {
        let mut inside = vec![false; self.triangles.len()];
        let mut pending = vec![start];
        inside[start as usize] = true;
        while let Some(current) = pending.pop() {
            self.for_each_neighbour(current, |neighbour, a, b| {
                if !inside[neighbour as usize] && self.is_smooth_edge(current, neighbour, a, b) {
                    inside[neighbour as usize] = true;
                    pending.push(neighbour);
                }
            });
        }
        inside
    }

    /// Triangles the surface brush may paint: those reachable from `seeds`
    /// across shared edges without leaving the brush, and facing the same way
    /// as the surface under the cursor. Keeps a big brush from jumping to
    /// separate or hidden parts of the model that happen to be nearby.
    fn surface_patch(
        &self,
        seeds: &[u32],
        start: Vec3,
        end: Vec3,
        radius: f32,
        normal: Vec3,
    ) -> Vec<u32> {
        let near = |index: u32| {
            let triangle = &self.triangles[index as usize];
            let center = interpolate(triangle.position, [1.0 / 3.0; 3]);
            let reach = triangle
                .position
                .iter()
                .map(|corner| corner.distance(center))
                .fold(0.0, f32::max);
            triangle.normal.dot(normal) >= SURFACE_MIN_FACING
                && distance_to_segment_3d(center, start, end) <= radius + reach
        };
        let mut patch: Vec<u32> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut pending: Vec<u32> = seeds
            .iter()
            .copied()
            .filter(|&seed| seen.insert(seed))
            .collect();
        while let Some(current) = pending.pop() {
            patch.push(current);
            self.for_each_neighbour(current, |neighbour, _, _| {
                if !seen.contains(&neighbour) && near(neighbour) {
                    seen.insert(neighbour);
                    pending.push(neighbour);
                }
            });
        }
        patch
    }

    /// Calls `visit(neighbour, edge start, edge end)` for every triangle
    /// sharing an edge (by position) with `index`.
    fn for_each_neighbour(&self, index: u32, mut visit: impl FnMut(u32, Vec3, Vec3)) {
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
        let triangle = &self.triangles[index as usize];
        for corner in 0..3 {
            let (a, b) = (
                triangle.position[corner],
                triangle.position[(corner + 1) % 3],
            );
            let key = edge_key(a, b);
            let first = edges.partition_point(|(edge, _)| *edge < key);
            for &(_, neighbour) in edges[first..].iter().take_while(|(edge, _)| *edge == key) {
                if neighbour != index {
                    visit(neighbour, a, b);
                }
            }
        }
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

    /// The UV rectangle holding the part of `triangle` that is on screen
    /// inside `min..max` (points), or `None` if none of it is. The triangle is
    /// clipped in clip space, so this also holds for triangles seen edge-on
    /// or reaching behind the camera.
    fn uv_bounds_in_screen_rect(
        &self,
        triangle: &Triangle,
        min: Vec2,
        max: Vec2,
    ) -> Option<(Vec2, Vec2)> {
        let (width, height) = (self.viewport.width(), self.viewport.height());
        let low = Vec2::new(min.x / width * 2.0 - 1.0, 1.0 - max.y / height * 2.0);
        let high = Vec2::new(max.x / width * 2.0 - 1.0, 1.0 - min.y / height * 2.0);
        // A point of the clipped polygon: clip position and barycentric weights.
        let mut polygon: Vec<(Vec4, [f32; 3])> = (0..3)
            .map(|i| {
                let mut weights = [0.0; 3];
                weights[i] = 1.0;
                (self.view_proj * triangle.position[i].extend(1.0), weights)
            })
            .collect();
        // Keep what is on the positive side of each plane (dot with clip position).
        let planes = [
            Vec4::new(0.0, 0.0, 1.0, 0.0),
            Vec4::new(1.0, 0.0, 0.0, -low.x),
            Vec4::new(-1.0, 0.0, 0.0, high.x),
            Vec4::new(0.0, 1.0, 0.0, -low.y),
            Vec4::new(0.0, -1.0, 0.0, high.y),
        ];
        for plane in planes {
            let mut kept = Vec::with_capacity(polygon.len() + 1);
            for (i, &(a, weights_a)) in polygon.iter().enumerate() {
                let (b, weights_b) = polygon[(i + 1) % polygon.len()];
                let (side_a, side_b) = (plane.dot(a), plane.dot(b));
                if side_a >= 0.0 {
                    kept.push((a, weights_a));
                }
                if (side_a >= 0.0) != (side_b >= 0.0) {
                    let t = side_a / (side_a - side_b);
                    let weights =
                        [0, 1, 2].map(|k| weights_a[k] + (weights_b[k] - weights_a[k]) * t);
                    kept.push((a + (b - a) * t, weights));
                }
            }
            polygon = kept;
            if polygon.is_empty() {
                return None;
            }
        }
        let uvs: Vec<Vec2> = polygon
            .iter()
            .map(|(_, weights)| interpolate(triangle.uv, *weights))
            .collect();
        Some(uv_bounds(&uvs))
    }

    fn to_local(&self, point: Pos2) -> Vec2 {
        Vec2::new(point.x - self.viewport.min.x, point.y - self.viewport.min.y)
    }

    /// World radius of the surface brush at `position` for a screen radius
    /// of `points`: measured there, or at the orbit centre when the brush
    /// keeps its size on the model.
    fn brush_world_radius(&self, position: Vec3, points: f32, brush: &Brush) -> f32 {
        if brush.keep_size_on_model {
            points * self.world_per_point * self.focus_depth
        } else {
            self.world_radius(position, points)
        }
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

    /// `hit` for the pixel `screen` lies in, measured at its centre and
    /// remembered. A texel only needs to know what is in front of its pixel,
    /// and with stretched UVs thousands of texels share one.
    fn pixel_hit(&self, screen: Vec2) -> Option<(u32, f32)> {
        // A slot holds the triangle in the high half and the distance in the
        // low half. These two triangle numbers mean "not asked yet" and
        // "nothing there" (no model has that many triangles).
        const UNKNOWN: u64 = u64::MAX;
        const NOTHING: u64 = (u32::MAX as u64) << 32;
        let (width, height) = (self.viewport.width().ceil(), self.viewport.height().ceil());
        if screen.x < 0.0 || screen.y < 0.0 || screen.x >= width || screen.y >= height {
            return None;
        }
        let (x, y) = (screen.x.floor(), screen.y.floor());
        let hits = self.pixel_hits.get_or_init(|| {
            (0..width as usize * height as usize)
                .map(|_| AtomicU64::new(UNKNOWN))
                .collect()
        });
        let slot = &hits[y as usize * width as usize + x as usize];
        let mut value = slot.load(Ordering::Relaxed);
        if value == UNKNOWN {
            // Two threads may both work this out; they get the same answer.
            value = match self.hit(Vec2::new(x + 0.5, y + 0.5)) {
                Some((triangle, distance)) => (triangle as u64) << 32 | distance.to_bits() as u64,
                None => NOTHING,
            };
            slot.store(value, Ordering::Relaxed);
        }
        (value != NOTHING).then(|| ((value >> 32) as u32, f32::from_bits(value as u32)))
    }

    /// Whether `position` (on triangle `index`, seen at `screen`) is unoccluded.
    fn is_visible(&self, position: Vec3, screen: Vec2, index: u32) -> bool {
        match self.pixel_hit(screen) {
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

/// What the brush puts into each plane it paints: linear colour for base
/// colour, the value itself for data channels. On a mask: white shows the
/// layer, and erasing paints black, which hides it.
fn brush_values(brush: &Brush, on_mask: bool) -> Vec<(Plane, [f32; 3])> {
    if on_mask {
        let value = if brush.erases() { 0.0 } else { 1.0 };
        return vec![(Plane::Mask, [value; 3])];
    }
    let mut values = Vec::new();
    if brush.paint_color {
        let color = brush.color.map(srgb_float_to_linear);
        values.push((Plane::Channel(Channel::BaseColor), color));
    }
    if brush.paint_roughness {
        let roughness = [brush.roughness.clamp(0.0, 1.0); 3];
        values.push((Plane::Channel(Channel::Roughness), roughness));
    }
    if brush.paint_metallic {
        let metallic = [brush.metallic.clamp(0.0, 1.0); 3];
        values.push((Plane::Channel(Channel::Metallic), metallic));
    }
    values
}

/// The screen-space brush: a circle dragged along a segment, in viewport points.
struct ScreenBrush {
    from: Vec2,
    to: Vec2,
    radius: f32,
    hardness: f32,
}

/// A triangle (or a strip of it) the screen brush paints.
#[derive(Clone, Copy)]
struct ScreenJob {
    index: u32,
    /// The part of the triangle to paint, as UV bounds.
    bounds: (Vec2, Vec2),
    /// Rough number of texels it holds.
    texels: usize,
}

/// The layer being painted plus the stroke's bookkeeping.
struct LayerTarget<'a> {
    layer: &'a mut Layer,
    stroke: &'a mut Stroke,
    slot_sizes: &'a [(u32, u32)],
    values: Vec<(Plane, [f32; 3])>,
    opacity: f32,
    effect: Effect,
    /// Per slot, the texels the blur may use (see `StrokePainter::uv_mask`).
    uv_masks: Vec<Option<UvMask>>,
    /// Per slot, the texels blur or smear covered in this segment.
    covered: Vec<Vec<Covered>>,
    /// Changed region per slot and plane.
    regions: Vec<[Option<ImageRegion>; Plane::COUNT]>,
}

/// What a stroke does to the texels it covers.
#[derive(Clone, Copy)]
enum Effect {
    /// Blend the brush's values in.
    Paint,
    /// Remove paint.
    Erase,
    /// Average with the texels around, this far (world units).
    Blur(f32),
    /// Pull in the paint from this far back along the stroke (world units).
    Smear(Vec3),
}

impl<'a> LayerTarget<'a> {
    fn new(
        layer: &'a mut Layer,
        stroke: &'a mut Stroke,
        slot_sizes: &'a [(u32, u32)],
        brush: &Brush,
    ) -> Self {
        let on_mask = stroke.on_mask;
        Self {
            layer,
            stroke,
            slot_sizes,
            values: brush_values(brush, on_mask),
            opacity: brush.opacity.clamp(0.0, 1.0),
            effect: match brush.tool {
                Tool::Blur => Effect::Blur(0.0),
                Tool::Smear => Effect::Smear(Vec3::ZERO),
                // A mask is erased by painting black (see `brush_values`).
                Tool::Paint | Tool::Fill if brush.erases() && !on_mask => Effect::Erase,
                Tool::Paint | Tool::Fill => Effect::Paint,
            },
            uv_masks: Vec::new(),
            covered: vec![Vec::new(); slot_sizes.len()],
            regions: vec![Default::default(); slot_sizes.len()],
        }
    }

    /// Blurs or smears everything the segment covered, slot by slot.
    fn filter(&mut self) {
        for material in 0..self.covered.len() {
            let covered = std::mem::take(&mut self.covered[material]);
            if covered.is_empty() {
                continue;
            }
            if let Some(mut slot) = self.slot(material) {
                slot.filter(covered);
            }
        }
    }

    /// The texel region that changed, per slot and plane.
    fn changed(self) -> Vec<(usize, Plane, ImageRegion)> {
        let mut changed = Vec::new();
        for (material, regions) in self.regions.into_iter().enumerate() {
            for (plane, region) in Plane::ALL.into_iter().zip(regions) {
                if let Some(region) = region {
                    changed.push((material, plane, region));
                }
            }
        }
        changed
    }

    /// The size of `material`'s textures if the layer paints on that slot.
    /// Unlike `slot`, this creates nothing.
    fn slot_size(&self, material: usize) -> Option<(u32, u32)> {
        (self.layer.targets.get(material) == Some(&true))
            .then(|| self.slot_sizes.get(material).copied())
            .flatten()
    }

    /// The textures of `material` to paint into (one per painted channel),
    /// created on first use. `None` if the layer doesn't paint on that slot.
    fn slot(&mut self, material: usize) -> Option<SlotTarget<'_>> {
        if self.layer.targets.get(material) != Some(&true) {
            return None;
        }
        let (width, height) = *self.slot_sizes.get(material)?;
        // Where a mask hasn't been painted, it has its starting value.
        let unpainted_mask = self.layer.mask.unwrap_or(255);
        let textures = self.layer.textures.get_mut(material)?;
        let strokes = self.stroke.slots.get_mut(material)?;
        let regions = self.regions.get_mut(material)?;
        let covered = self.covered.get_mut(material)?;
        let mut channels = Vec::new();
        for (((plane, texture), stroke), region) in Plane::ALL
            .into_iter()
            .zip(textures.iter_mut())
            .zip(strokes.iter_mut())
            .zip(regions.iter_mut())
        {
            let Some(&(_, value)) = self.values.iter().find(|(p, _)| *p == plane) else {
                continue;
            };
            let image = texture.get_or_insert_with(|| match plane {
                Plane::Mask => Image::filled(
                    width,
                    height,
                    [unpainted_mask, unpainted_mask, unpainted_mask, 255],
                ),
                Plane::Channel(_) => Image::filled(width, height, CLEAR),
            });
            if (image.width, image.height) != (width, height) {
                continue;
            }
            let stroke = stroke.get_or_insert_with(|| SlotStroke::new(image));
            let encoding = plane.encoding();
            channels.push(ChannelTarget {
                image,
                stroke,
                region,
                value,
                encoded_value: value.map(|v| encoding.encode(v)),
                encoding,
            });
        }
        (!channels.is_empty()).then_some(SlotTarget {
            channels,
            width,
            height,
            opacity: self.opacity,
            effect: self.effect,
            uv_mask: self.uv_masks.get(material).cloned().flatten(),
            covered,
        })
    }
}

/// One channel texture of a slot being painted.
struct ChannelTarget<'a> {
    image: &'a mut Image,
    stroke: &'a mut SlotStroke,
    region: &'a mut Option<ImageRegion>,
    value: [f32; 3],
    /// `value` as stored bytes, for texels where the brush replaces what's there.
    encoded_value: [u8; 3],
    encoding: Encoding,
}

impl ChannelTarget<'_> {
    /// Stores a new pixel and records the change for undo and upload.
    fn write(&mut self, x: u32, y: u32, pixel: [u8; 4]) {
        let width = self.image.width;
        let offset = ((y * width + x) * 4) as usize;
        if self.image.rgba[offset..offset + 4] == pixel {
            return;
        }
        self.image.rgba[offset..offset + 4].copy_from_slice(&pixel);
        let tile = (y / history::TILE_SIZE) * tiles_x(width) + x / history::TILE_SIZE;
        self.stroke.touched_tiles[tile as usize] = true;
        let pixel = ImageRegion::pixel(x, y);
        *self.region = Some(self.region.map_or(pixel, |region| region.union(pixel)));
    }

    /// Texel (x, y), wrapping around the edges, as linear premultiplied RGBA.
    fn premultiplied(&self, x: i64, y: i64) -> Vec4 {
        let x = x.rem_euclid(self.image.width as i64) as u32;
        let y = y.rem_euclid(self.image.height as i64) as u32;
        let offset = ((y * self.image.width + x) * 4) as usize;
        let pixel = &self.image.rgba[offset..offset + 4];
        let alpha = pixel[3] as f32 / 255.0;
        Vec4::new(
            self.encoding.decode(pixel[0]) * alpha,
            self.encoding.decode(pixel[1]) * alpha,
            self.encoding.decode(pixel[2]) * alpha,
            alpha,
        )
    }

    /// Back from linear premultiplied RGBA to the stored pixel.
    fn encode(&self, value: Vec4) -> [u8; 4] {
        let alpha = value.w.clamp(0.0, 1.0);
        if alpha <= 0.0 {
            return [0; 4];
        }
        [
            self.encoding.encode(value.x / alpha),
            self.encoding.encode(value.y / alpha),
            self.encoding.encode(value.z / alpha),
            (alpha * 255.0).round() as u8,
        ]
    }

    /// Blurs the `texels` of one tile; returns their (x, y, new pixel).
    /// Works on a copy of the tile plus `radius` texels around it, as linear
    /// premultiplied colour: a horizontal pass, then a vertical one. Texels
    /// `mask` marks as unused don't count (normalised convolution).
    #[allow(clippy::too_many_arguments)]
    fn blur_tile(
        &self,
        tile_x: u32,
        tile_y: u32,
        radius: i64,
        kernel: &[f32],
        mask: Option<&[bool]>,
        texels: &[Covered],
        opacity: f32,
    ) -> Vec<(u32, u32, [u8; 4])> {
        let mut writes = Vec::with_capacity(texels.len());
        let (width, height) = (self.image.width as i64, self.image.height as i64);
        let tile = BLUR_TILE as i64;
        let (x0, y0) = (tile_x as i64 * tile - radius, tile_y as i64 * tile - radius);
        let span = tile + 2 * radius;

        // The region, with unused texels zeroed and their weight (w) zero.
        let mut source = Vec::with_capacity((span * span) as usize);
        for y in y0..y0 + span {
            for x in x0..x0 + span {
                let index = (y.rem_euclid(height) * width + x.rem_euclid(width)) as usize;
                let used = mask.is_none_or(|mask| mask.get(index).copied().unwrap_or(false));
                source.push(if used {
                    let value = self.premultiplied(x, y);
                    (value, 1.0)
                } else {
                    (Vec4::ZERO, 0.0)
                });
            }
        }
        // Horizontal pass, for the tile's columns only.
        let mut across = vec![(Vec4::ZERO, 0.0f32); (span * tile) as usize];
        for row in 0..span {
            for column in 0..tile {
                let mut sum = (Vec4::ZERO, 0.0);
                for (k, weight) in kernel.iter().enumerate() {
                    let (value, used) = source[(row * span + column + k as i64) as usize];
                    sum.0 += value * *weight;
                    sum.1 += used * weight;
                }
                across[(row * tile + column) as usize] = sum;
            }
        }
        // Vertical pass, for the covered texels only.
        for c in texels {
            let (column, row) = (
                c.x as i64 - tile_x as i64 * tile,
                c.y as i64 - tile_y as i64 * tile,
            );
            let mut sum = (Vec4::ZERO, 0.0);
            for (k, weight) in kernel.iter().enumerate() {
                let (value, used) = across[((row + k as i64) * tile + column) as usize];
                sum.0 += value * *weight;
                sum.1 += used * weight;
            }
            let current = self.premultiplied(c.x as i64, c.y as i64);
            if sum.1 <= 1e-6 {
                continue;
            }
            let blurred = sum.0 / sum.1;
            let strength = (c.amount.min(1.0) * opacity).clamp(0.0, 1.0);
            writes.push((c.x, c.y, self.encode(current.lerp(blurred, strength))));
        }
        writes
    }

    /// Bilinear sample at a texel-space position (texel centres at integers).
    fn sample(&self, texel: Vec2) -> Option<Vec4> {
        if !texel.is_finite() {
            return None;
        }
        let base = texel.floor();
        let f = texel - base;
        let (x, y) = (base.x as i64, base.y as i64);
        let top = self
            .premultiplied(x, y)
            .lerp(self.premultiplied(x + 1, y), f.x);
        let bottom = self
            .premultiplied(x, y + 1)
            .lerp(self.premultiplied(x + 1, y + 1), f.x);
        Some(top.lerp(bottom, f.y))
    }
}

/// One slot's textures being painted, all the same size.
struct SlotTarget<'a> {
    channels: Vec<ChannelTarget<'a>>,
    width: u32,
    height: u32,
    opacity: f32,
    effect: Effect,
    /// Texels the blur may use; `None` means all.
    uv_mask: Option<UvMask>,
    /// Blur and smear: the texels covered so far in this segment.
    covered: &'a mut Vec<Covered>,
}

/// A texel the blur or smear brush covers during one segment.
#[derive(Clone, Copy)]
struct Covered {
    x: u32,
    y: u32,
    amount: f32,
    /// Smear: where the paint comes from, relative to this texel (texels).
    shift: Vec2,
    /// Blur: how far the paint spreads here (texels; the Gaussian's sigma).
    spread: f32,
}

/// The texels of one triangle in a `width` x `height` texture: the ones on
/// it plus a seam margin, limited to the part inside some UV bounds.
struct TriangleScan {
    /// Corners in texel units, so the seam margin is the same on both axes.
    corners: [Vec2; 3],
    margins: [f32; 3],
    /// World distance from a texel centre to its corners.
    reach: f32,
    /// Ranges of texel columns and rows to look at (not yet wrapped).
    x: (i64, i64),
    y: (i64, i64),
}

/// A texel found by `TriangleScan::texels`.
struct ScannedTexel {
    /// Texel coordinates; outside the texture when the margin wraps around.
    raw_x: i64,
    raw_y: i64,
    center: Vec2,
    weights: [f32; 3],
}

impl TriangleScan {
    /// `None` if the triangle is degenerate in UV or `bounds` miss it.
    fn new(
        width: u32,
        height: u32,
        triangle: &Triangle,
        bounds: Option<(Vec2, Vec2)>,
    ) -> Option<Self> {
        let size = Vec2::new(width as f32, height as f32);
        let corners = triangle.uv.map(|uv| uv * size);
        let margins = edge_margins(corners, SEAM_MARGIN_TEXELS)?;
        let world_area = (triangle.position[1] - triangle.position[0])
            .cross(triangle.position[2] - triangle.position[0])
            .length();
        let texel_area = cross(corners[1] - corners[0], corners[2] - corners[0]).abs();
        let reach = (world_area / texel_area).sqrt() * std::f32::consts::FRAC_1_SQRT_2;
        let (mut lo, mut hi) = uv_bounds(&corners);
        lo -= Vec2::splat(SEAM_MARGIN_TEXELS);
        hi += Vec2::splat(SEAM_MARGIN_TEXELS);
        if let Some((brush_lo, brush_hi)) = bounds {
            // Grow by a texel so texels larger than the brush are still found.
            lo = lo.max(brush_lo * size - Vec2::ONE);
            hi = hi.min(brush_hi * size + Vec2::ONE);
        }
        if lo.x > hi.x || lo.y > hi.y {
            return None;
        }
        Some(Self {
            corners,
            margins,
            reach,
            x: texel_range(lo.x, hi.x, width),
            y: texel_range(lo.y, hi.y, height),
        })
    }

    /// The texels that lie on the triangle or in its seam margin.
    fn texels(&self) -> impl Iterator<Item = ScannedTexel> + '_ {
        (self.y.0..=self.y.1).flat_map(move |raw_y| {
            (self.x.0..=self.x.1).filter_map(move |raw_x| {
                let center = Vec2::new(raw_x as f32 + 0.5, raw_y as f32 + 0.5);
                let weights = barycentric_2d(center, self.corners)?;
                weights
                    .iter()
                    .zip(self.margins)
                    .all(|(w, margin)| *w >= -margin)
                    .then_some(ScannedTexel {
                        raw_x,
                        raw_y,
                        center,
                        weights,
                    })
            })
        })
    }
}

impl SlotTarget<'_> {
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
        let Some(scan) = TriangleScan::new(self.width, self.height, triangle, bounds) else {
            return;
        };
        let texel_size = scan.reach * std::f32::consts::SQRT_2;
        for texel in scan.texels() {
            // Painting can't add to a texel this stroke already fully covered.
            if self.saturated(texel.raw_x, texel.raw_y) {
                continue;
            }
            let position = interpolate(triangle.position, texel.weights);
            let amount = coverage(position, scan.reach);
            if amount > 0.0 {
                let x = texel.raw_x.rem_euclid(self.width as i64) as u32;
                let y = texel.raw_y.rem_euclid(self.height as i64) as u32;
                // Blur and smear read neighbouring texels: they're done for the
                // whole segment at once, after every triangle (see `filter`).
                let (shift, spread) = match self.effect {
                    Effect::Paint | Effect::Erase => {
                        self.apply(x, y, amount);
                        continue;
                    }
                    Effect::Blur(reach) => (Vec2::ZERO, reach / texel_size),
                    Effect::Smear(motion) => {
                        let Some(source) = barycentric_3d(position - motion, triangle.position)
                        else {
                            continue;
                        };
                        (interpolate(scan.corners, source) - texel.center, 0.0)
                    }
                };
                self.covered.push(Covered {
                    x,
                    y,
                    amount,
                    shift,
                    spread,
                });
            }
        }
    }

    /// Whether painting or erasing already gave the texel full coverage in
    /// this stroke, so more of it changes nothing. Blur and smear read the
    /// texels around, so they always run.
    fn saturated(&self, raw_x: i64, raw_y: i64) -> bool {
        if !matches!(self.effect, Effect::Paint | Effect::Erase) || self.channels.is_empty() {
            return false;
        }
        let x = raw_x.rem_euclid(self.width as i64) as usize;
        let y = raw_y.rem_euclid(self.height as i64) as usize;
        let texel = y * self.width as usize + x;
        self.channels
            .iter()
            .all(|target| target.stroke.coverage[texel] >= 1.0)
    }

    /// Blurs or smears this segment's covered texels.
    fn filter(&mut self, mut covered: Vec<Covered>) {
        // Neighbouring triangles share their seam-margin texels: keep each
        // texel once, with its strongest coverage.
        covered.sort_unstable_by_key(|c| (c.y, c.x));
        covered.dedup_by(|later, kept| {
            let same = (later.x, later.y) == (kept.x, kept.y);
            if same && later.amount > kept.amount {
                *kept = *later;
            }
            same
        });
        match self.effect {
            Effect::Blur(_) => self.blur(&covered),
            Effect::Smear(_) => self.smear(&covered),
            Effect::Paint | Effect::Erase => {}
        }
    }

    /// A Gaussian blur of the covered texels, done horizontally then
    /// vertically on tiles around them. Texels outside the UV islands
    /// (`mask`) are left out of the average.
    fn blur(&mut self, covered: &[Covered]) {
        let spread = covered.iter().map(|c| c.spread).sum::<f32>() / covered.len() as f32;
        // Narrower than a texel can't be sampled: blur one texel, more weakly.
        let (sigma, opacity) = if spread < 1.0 {
            (1.0, self.opacity * spread * spread)
        } else {
            (spread.min(MAX_BLUR_TEXELS), self.opacity)
        };
        if opacity <= 0.0 {
            return;
        }
        let radius = (sigma * 3.0).ceil() as i64;
        let kernel: Vec<f32> = (-radius..=radius)
            .map(|i| (-((i * i) as f32) / (2.0 * sigma * sigma)).exp())
            .collect();

        // Covered texels by tile (they are sorted by row, so group explicitly).
        let mut tiles: HashMap<(u32, u32), Vec<Covered>> = HashMap::new();
        for c in covered {
            tiles
                .entry((c.x / BLUR_TILE, c.y / BLUR_TILE))
                .or_default()
                .push(*c);
        }
        let mask = self.uv_mask.clone();
        let tiles: Vec<_> = tiles.into_iter().collect();
        let mask = mask.as_deref().map(Vec::as_slice);
        for target in &mut self.channels {
            // A tile is a lot of work: a few are worth spreading out.
            let blurred = parallel_filter_map(&tiles, 4, |((tile_x, tile_y), texels)| {
                Some(target.blur_tile(*tile_x, *tile_y, radius, &kernel, mask, texels, opacity))
            });
            for (x, y, pixel) in blurred.into_iter().flatten() {
                target.write(x, y, pixel);
            }
        }
    }

    /// Pushes paint along the stroke: each covered texel takes the paint
    /// from `shift` back. Long moves go in steps of at most a texel, so the
    /// paint flows instead of jumping (and never reads far outside its island).
    fn smear(&mut self, covered: &[Covered]) {
        let longest = covered.iter().map(|c| c.shift.length()).fold(0.0, f32::max);
        if longest <= 0.0 {
            return;
        }
        let steps = (longest.ceil() as usize).clamp(1, MAX_SMEAR_STEPS);
        // Split the strength too, so the result doesn't depend on the steps.
        let opacity = 1.0 - (1.0 - self.opacity.clamp(0.0, 1.0)).powf(1.0 / steps as f32);
        for target in &mut self.channels {
            for _ in 0..steps {
                let new = parallel_filter_map(covered, 4096, |c| {
                    let from = Vec2::new(c.x as f32, c.y as f32) + c.shift / steps as f32;
                    let source = target.sample(from)?;
                    let current = target.premultiplied(c.x as i64, c.y as i64);
                    let strength = c.amount.min(1.0) * opacity;
                    Some((c.x, c.y, target.encode(current.lerp(source, strength))))
                });
                for (x, y, pixel) in new {
                    target.write(x, y, pixel);
                }
            }
        }
    }

    /// Covers every texel of the slot's textures.
    fn fill_all(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                self.apply(x, y, 1.0);
            }
        }
    }

    fn apply(&mut self, x: u32, y: u32, amount: f32) {
        let texel = (y * self.width + x) as usize;
        let amount = amount.min(1.0);
        let alpha = amount * self.opacity;
        let tile = (y / history::TILE_SIZE) * tiles_x(self.width) + x / history::TILE_SIZE;
        for target in &mut self.channels {
            if amount <= target.stroke.coverage[texel] {
                continue;
            }
            target.stroke.coverage[texel] = amount;

            // Blend from the pixel as it was before the stroke, with straight alpha.
            let offset = texel * 4;
            let original = &target.stroke.original[offset..offset + 4];
            let original_alpha = original[3] as f32 / 255.0;
            let out = &mut target.image.rgba[offset..offset + 4];
            if matches!(self.effect, Effect::Erase) {
                out.copy_from_slice(original);
                out[3] = (original_alpha * (1.0 - alpha) * 255.0).round() as u8;
            } else {
                let out_alpha = alpha + original_alpha * (1.0 - alpha);
                if original_alpha == 0.0 || alpha >= 1.0 {
                    // Nothing below shows through: the brush's own value.
                    out[..3].copy_from_slice(&target.encoded_value);
                } else {
                    for (channel, value) in target.value.iter().enumerate() {
                        let below = target.encoding.decode(original[channel])
                            * original_alpha
                            * (1.0 - alpha);
                        out[channel] = target.encoding.encode((value * alpha + below) / out_alpha);
                    }
                }
                out[3] = (out_alpha * 255.0).round() as u8;
            }
            target.stroke.touched_tiles[tile as usize] = true;
            let pixel = ImageRegion::pixel(x, y);
            *target.region = Some(target.region.map_or(pixel, |region| region.union(pixel)));
        }
    }
}

/// `items.iter().filter_map(f).collect()`, spread over the CPU cores when
/// there are at least `min_items` (fewer aren't worth starting threads for).
/// Keeps the order.
fn parallel_filter_map<T: Sync, R: Send>(
    items: &[T],
    min_items: usize,
    f: impl Fn(&T) -> Option<R> + Sync,
) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    if threads < 2 || items.len() < min_items {
        return items.iter().filter_map(f).collect();
    }
    let chunk = items.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let jobs: Vec<_> = items
            .chunks(chunk)
            .map(|part| scope.spawn(|| part.iter().filter_map(&f).collect::<Vec<R>>()))
            .collect();
        jobs.into_iter()
            .flat_map(|job| job.join().expect("brush worker thread panicked"))
            .collect()
    })
}

/// Like `parallel_filter_map`, but threads take one item at a time as they
/// finish, so items of very different cost still keep every core busy.
fn parallel_filter_map_dynamic<T: Sync, R: Send>(
    items: &[T],
    min_items: usize,
    f: impl Fn(&T) -> Option<R> + Sync,
) -> Vec<R> {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    if threads < 2 || items.len() < min_items {
        return items.iter().filter_map(f).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut found: Vec<(usize, R)> = std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..threads.min(items.len()))
            .map(|_| {
                scope.spawn(|| {
                    let mut found = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(index) else {
                            break found;
                        };
                        if let Some(result) = f(item) {
                            found.push((index, result));
                        }
                    }
                })
            })
            .collect();
        jobs.into_iter()
            .flat_map(|job| job.join().expect("brush worker thread panicked"))
            .collect()
    });
    found.sort_unstable_by_key(|(index, _)| *index);
    found.into_iter().map(|(_, result)| result).collect()
}

fn tiles_x(width: u32) -> u32 {
    width.div_ceil(history::TILE_SIZE)
}

fn tiles_y(height: u32) -> u32 {
    height.div_ceil(history::TILE_SIZE)
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
    use crate::document::{Document, Plane};
    use crate::mesh::Channel;
    use crate::mesh::{Image, ImageRegion, Material, Mesh, SubMesh, Vertex};
    use crate::painting::brush::{Brush, FillMode, Tool};
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
        Material::new(
            name,
            Image::filled(TEXTURE_SIZE, TEXTURE_SIZE, [0, 0, 0, 255]),
        )
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
            ..Brush::default()
        }
    }

    /// Paints through a `Document` (new layer, slot 0 only), then writes the
    /// composites back into `mesh` so tests can inspect the result there.
    fn paint(
        mesh: &mut Mesh,
        camera: &OrbitCamera,
        points: &[Pos2],
        brush: &Brush,
    ) -> Option<ImageRegion> {
        paint_strokes(mesh, camera, &[(points, brush)])
    }

    /// Like `paint`, with several strokes one after another on the same layer.
    fn paint_strokes(
        mesh: &mut Mesh,
        camera: &OrbitCamera,
        strokes: &[(&[Pos2], &Brush)],
    ) -> Option<ImageRegion> {
        let painter = StrokePainter::new(mesh, camera, viewport());
        let empty = Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
            submeshes: Vec::new(),
            materials: Vec::new(),
        };
        let mut document = Document::new(std::mem::replace(mesh, empty));
        for (slot, target) in document.layers[1].targets.iter_mut().enumerate() {
            *target = slot == 0;
        }
        let sizes = document.slot_sizes();
        let mut region: Option<ImageRegion> = None;
        for &(points, brush) in strokes {
            let mut stroke = Stroke::begin(&document, 1).unwrap();
            let segments: Vec<_> = if points.len() == 1 {
                vec![(points[0], points[0])]
            } else {
                points.windows(2).map(|pair| (pair[0], pair[1])).collect()
            };
            for (from, to) in segments {
                let layer = &mut document.layers[1];
                for (_, _, changed) in
                    painter.paint_segment(&mut stroke, layer, &sizes, from, to, brush)
                {
                    region = Some(region.map_or(changed, |r| r.union(changed)));
                }
            }
        }

        let composites: Vec<_> = (0..document.mesh.materials.len())
            .map(|material| document.composite(material, Channel::BaseColor).unwrap())
            .collect();
        *mesh = document.mesh;
        for (material, composite) in mesh.materials.iter_mut().zip(composites) {
            material.base_color = composite;
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

        // Alpha is stored in 8 bits, so allow one step of rounding.
        let half = crate::mesh::linear_to_srgb(0.5) + 1;
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

    #[test]
    fn finished_stroke_undoes_to_the_original_texture() {
        let mut document = Document::new(crate::mesh::primitives::cube());
        let before = document
            .composite(0, Channel::BaseColor)
            .unwrap()
            .rgba
            .clone();
        let camera = OrbitCamera {
            yaw: 0.6,
            ..test_camera()
        };
        let painter = StrokePainter::new(&document.mesh, &camera, viewport());
        let sizes = document.slot_sizes();
        let mut stroke = Stroke::begin(&document, 1).unwrap();
        for (from, to) in [
            ((90.0, 128.0), (140.0, 110.0)),
            ((140.0, 110.0), (190.0, 140.0)),
        ] {
            painter.paint_segment(
                &mut stroke,
                &mut document.layers[1],
                &sizes,
                Pos2::new(from.0, from.1),
                Pos2::new(to.0, to.1),
                &brush(40.0, false),
            );
        }
        let after = document
            .composite(0, Channel::BaseColor)
            .unwrap()
            .rgba
            .clone();
        assert_ne!(before, after);

        let mut history = crate::painting::history::History::default();
        history.push("Test", stroke.finish());
        history.undo(&mut document);
        assert_eq!(
            document.composite(0, Channel::BaseColor).unwrap().rgba,
            before
        );
        history.redo(&mut document);
        assert_eq!(
            document.composite(0, Channel::BaseColor).unwrap().rgba,
            after
        );
    }

    #[test]
    fn eraser_removes_paint_from_the_layer() {
        let mut document = Document::new(test_mesh());
        let painter = StrokePainter::new(&document.mesh, &test_camera(), viewport());
        let sizes = document.slot_sizes();
        let point = Pos2::new(91.0, 128.0);
        let mut paint_brush = brush(20.0, false);
        for erase in [false, true] {
            paint_brush.erase = erase;
            let mut stroke = Stroke::begin(&document, 1).unwrap();
            painter.paint_segment(
                &mut stroke,
                &mut document.layers[1],
                &sizes,
                point,
                point,
                &paint_brush,
            );
        }
        let layer = document.layers[1].textures[0][0].as_ref().unwrap();
        assert!(layer.rgba.chunks_exact(4).any(|pixel| pixel[0] == 255));
        assert!(layer.rgba.chunks_exact(4).all(|pixel| pixel[3] == 0));
    }

    #[test]
    fn base_layer_cannot_be_painted() {
        let document = Document::new(test_mesh());
        assert!(Stroke::begin(&document, 0).is_none());
    }

    #[test]
    fn stroke_that_misses_the_model_leaves_no_undo_step() {
        let mut document = Document::new(test_mesh());
        let painter = StrokePainter::new(&document.mesh, &test_camera(), viewport());
        let sizes = document.slot_sizes();
        let mut stroke = Stroke::begin(&document, 1).unwrap();
        let corner = Pos2::new(3.0, 3.0);
        painter.paint_segment(
            &mut stroke,
            &mut document.layers[1],
            &sizes,
            corner,
            corner,
            &brush(10.0, false),
        );
        assert!(stroke.finish().is_empty());
    }

    #[test]
    fn big_surface_brush_does_not_reach_separate_surfaces_behind() {
        // A front panel, and a smaller separate panel 0.1 behind it (fully
        // hidden) in its own UV island.
        // The brush sphere is far bigger than the gap, but the hidden panel isn't
        // connected to the painted surface, so it must stay clean.
        let mut mesh = mesh_from_triangles(&[
            (
                [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0], [0.45, 0.0], [0.225, 1.0]],
            ),
            (
                [[-0.8, -0.8, -0.1], [0.8, -0.8, -0.1], [0.0, 0.8, -0.1]],
                [[0.55, 0.0], [1.0, 0.0], [0.775, 1.0]],
            ),
        ]);
        for screen_space in [false, true] {
            let region = paint(
                &mut mesh,
                &test_camera(),
                &[Pos2::new(128.0, 140.0)],
                &brush(120.0, screen_space),
            );
            assert!(region.is_some());
            let texels = painted(&mesh);
            assert!(texels.iter().any(|t| t.0 < 58), "front panel painted");
            assert!(
                !texels.iter().any(|t| t.0 >= 70),
                "hidden panel painted (screen_space={screen_space})"
            );
        }
    }

    #[test]
    fn brush_paints_only_the_channels_it_is_set_to() {
        let mut document = Document::new(test_mesh());
        let painter = StrokePainter::new(&document.mesh, &test_camera(), viewport());
        let sizes = document.slot_sizes();
        let mut roughness_only = brush(20.0, false);
        roughness_only.paint_color = false;
        roughness_only.paint_roughness = true;
        roughness_only.roughness = 0.9;
        let point = Pos2::new(91.0, 128.0);
        let mut stroke = Stroke::begin(&document, 1).unwrap();
        let changed = painter.paint_segment(
            &mut stroke,
            &mut document.layers[1],
            &sizes,
            point,
            point,
            &roughness_only,
        );
        assert!(
            changed
                .iter()
                .all(|(_, plane, _)| *plane == Plane::Channel(Channel::Roughness))
        );
        assert!(!changed.is_empty());

        let textures = &document.layers[1].textures[0];
        assert!(textures[Channel::BaseColor.index()].is_none());
        assert!(textures[Channel::Metallic.index()].is_none());
        let roughness = document.composite(0, Channel::Roughness).unwrap();
        let painted = crate::mesh::Encoding::Linear.encode(0.9);
        assert!(
            roughness
                .rgba
                .chunks_exact(4)
                .any(|pixel| pixel[0] == painted)
        );
        // Base colour is untouched.
        let base = document.composite(0, Channel::BaseColor).unwrap();
        assert_eq!(base.rgba, document.mesh.materials[0].base_color.rgba);
        // One undo step, for the roughness texture.
        let changes = stroke.finish();
        assert!(matches!(
            changes.as_slice(),
            [crate::document::Change::Pixels {
                plane: Plane::Channel(Channel::Roughness),
                ..
            }]
        ));
    }

    /// Fills through a `Document` like `paint`, clicking at `at`. Only slot 0
    /// is targeted. Returns how many slot/channel regions changed.
    fn fill(mesh: &mut Mesh, at: Pos2, mode: FillMode) -> usize {
        let painter = StrokePainter::new(mesh, &test_camera(), viewport());
        let empty = Mesh {
            vertices: Vec::new(),
            indices: Vec::new(),
            submeshes: Vec::new(),
            materials: Vec::new(),
        };
        let mut document = Document::new(std::mem::replace(mesh, empty));
        for (slot, target) in document.layers[1].targets.iter_mut().enumerate() {
            *target = slot == 0;
        }
        let sizes = document.slot_sizes();
        let mut stroke = Stroke::begin(&document, 1).unwrap();
        let brush = Brush {
            tool: Tool::Fill,
            fill_mode: mode,
            ..brush(1.0, false)
        };
        let changed = painter.fill(&mut stroke, &mut document.layers[1], &sizes, at, &brush);
        let composite = document.composite(0, Channel::BaseColor).unwrap();
        *mesh = document.mesh;
        mesh.materials[0].base_color = composite;
        changed.len()
    }

    #[test]
    fn fill_covers_the_whole_layer() {
        let mut mesh = test_mesh();
        // Clicking empty space is fine for a whole-layer fill.
        assert_eq!(fill(&mut mesh, Pos2::new(5.0, 5.0), FillMode::Layer), 1);
        assert_eq!(painted(&mesh).len(), (TEXTURE_SIZE * TEXTURE_SIZE) as usize);
    }

    #[test]
    fn fill_covers_only_the_clicked_uv_island() {
        // test_mesh: two triangles in separate islands, left one at u < 0.5.
        let mut mesh = test_mesh();
        assert_eq!(
            fill(&mut mesh, Pos2::new(76.0, 145.0), FillMode::UvIsland),
            1
        );
        let texels = painted(&mesh);
        assert!(texels.len() > 1000);
        // (The seam margin left of u = 0 wraps round to the last column.)
        assert!(
            texels
                .iter()
                .all(|(x, _, _)| *x < TEXTURE_SIZE / 2 || *x == TEXTURE_SIZE - 1)
        );

        // A quad whose two triangles share a UV edge: one island, all of it.
        let mut quad = mesh_from_triangles(&[
            (
                [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            ),
            (
                [[-1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]],
                [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]],
            ),
        ]);
        assert_eq!(
            fill(&mut quad, Pos2::new(160.0, 150.0), FillMode::UvIsland),
            1
        );
        assert_eq!(painted(&quad).len(), (TEXTURE_SIZE * TEXTURE_SIZE) as usize);
    }

    #[test]
    fn fill_material_needs_a_material_the_layer_paints_on() {
        // Right triangle uses slot 1, which the layer doesn't paint on.
        let two_materials = || {
            let mut mesh = test_mesh();
            mesh.materials.push(blank_material("Other"));
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
            mesh
        };
        let mut other = two_materials();
        assert_eq!(
            fill(&mut other, Pos2::new(180.0, 145.0), FillMode::Material),
            0
        );
        assert!(painted(&other).is_empty());
        let mut mesh = two_materials();
        assert_eq!(
            fill(&mut mesh, Pos2::new(76.0, 145.0), FillMode::Material),
            1
        );
        assert_eq!(painted(&mesh).len(), (TEXTURE_SIZE * TEXTURE_SIZE) as usize);
    }

    #[test]
    fn brush_can_keep_one_size_on_near_and_far_parts() {
        // Looking down at a floor at an angle: the far dab is deeper in view.
        let camera = OrbitCamera {
            pitch: 30.0_f32.to_radians(),
            ..test_camera()
        };
        let dab = |at: Pos2, keep_size_on_model: bool| {
            let mut floor = floor_mesh(3.0);
            let brush = Brush {
                keep_size_on_model,
                ..brush(60.0, false)
            };
            paint(&mut floor, &camera, &[at], &brush).expect("dab should paint");
            painted(&floor).len() as f32
        };
        let (near, far) = (Pos2::new(128.0, 220.0), Pos2::new(128.0, 60.0));
        let ratio = dab(far, true) / dab(near, true);
        assert!(
            (0.8..1.25).contains(&ratio),
            "kept size: far/near = {ratio}"
        );
        let ratio = dab(far, false) / dab(near, false);
        assert!(ratio > 3.0, "screen size: far/near = {ratio}");
    }

    /// A hard red dab on the left triangle of `test_mesh`, at (76, 145).
    fn red_dab() -> (Vec<Pos2>, Brush) {
        (vec![Pos2::new(76.0, 145.0)], brush(30.0, false))
    }

    #[test]
    fn blur_softens_the_edge_of_paint() {
        let (dab, paint_brush) = red_dab();
        let mut hard = test_mesh();
        paint(&mut hard, &test_camera(), &dab, &paint_brush);

        let blur = Brush {
            tool: Tool::Blur,
            ..brush(60.0, false)
        };
        // A few short segments, like holding the mouse over the spot.
        let wiggle: Vec<_> = (0..12)
            .map(|i| Pos2::new(76.0 + (i % 2) as f32, 145.0))
            .collect();
        let mut soft = test_mesh();
        paint_strokes(
            &mut soft,
            &test_camera(),
            &[(&dab, &paint_brush), (&wiggle, &blur)],
        );

        // Partly covered texels: the hard dab has (nearly) none, the blurred one many.
        let partial = |mesh: &Mesh| {
            painted(mesh)
                .iter()
                .filter(|(_, _, red)| (20..230).contains(red))
                .count()
        };
        assert!(
            partial(&soft) > partial(&hard) + 50,
            "{} vs {}",
            partial(&soft),
            partial(&hard)
        );
        // Blurring spreads paint out; it doesn't add or remove much of it.
        // (Summed as linear light; the stored values are sRGB-encoded.)
        let total = |mesh: &Mesh| {
            painted(mesh)
                .iter()
                .map(|t| crate::mesh::Encoding::Srgb.decode(t.2))
                .sum::<f32>()
        };
        let ratio = total(&soft) / total(&hard);
        assert!((0.8..1.25).contains(&ratio), "total paint ratio {ratio}");
    }

    #[test]
    fn smear_pushes_paint_along_the_stroke() {
        let (dab, paint_brush) = red_dab();
        let smear = Brush {
            tool: Tool::Smear,
            ..brush(30.0, false)
        };
        // Drag from the middle of the dab to the right, in small steps.
        let drag: Vec<_> = (0..=20)
            .map(|i| Pos2::new(76.0 + i as f32, 145.0))
            .collect();
        let mut dabbed = test_mesh();
        paint(&mut dabbed, &test_camera(), &dab, &paint_brush);
        let mut smeared = test_mesh();
        paint_strokes(
            &mut smeared,
            &test_camera(),
            &[(&dab, &paint_brush), (&drag, &smear)],
        );

        let right_edge = |mesh: &Mesh| painted(mesh).iter().map(|t| t.0).max().unwrap();
        assert!(
            right_edge(&smeared) > right_edge(&dabbed) + 3,
            "{} vs {}",
            right_edge(&smeared),
            right_edge(&dabbed)
        );
        // Nothing appears on the left, where the stroke came from.
        let left_edge = |mesh: &Mesh| painted(mesh).iter().map(|t| t.0).min().unwrap();
        assert!(left_edge(&smeared) >= left_edge(&dabbed));
    }

    #[test]
    fn erase_fill_clears_only_the_clicked_island() {
        let painter = StrokePainter::new(&test_mesh(), &test_camera(), viewport());
        let mut document = Document::new(test_mesh());
        let sizes = document.slot_sizes();
        let mut fill = Brush {
            tool: Tool::Fill,
            fill_mode: FillMode::Layer,
            ..brush(1.0, false)
        };
        for (erase, mode) in [(false, FillMode::Layer), (true, FillMode::UvIsland)] {
            fill.erase = erase;
            fill.fill_mode = mode;
            let mut stroke = Stroke::begin(&document, 1).unwrap();
            let at = Pos2::new(76.0, 145.0);
            painter.fill(&mut stroke, &mut document.layers[1], &sizes, at, &fill);
        }
        let layer = document.layers[1].textures[0][0].as_ref().unwrap();
        let alpha = |x: u32, y: u32| layer.rgba[((y * TEXTURE_SIZE + x) * 4 + 3) as usize];
        // Left island (u < 0.45) erased; the right island and empty space stay filled.
        assert_eq!(alpha(20, 64), 0);
        assert_eq!(alpha(100, 64), 255);
        assert_eq!(alpha(64, 120), 255);
    }

    #[test]
    fn painting_a_mask_changes_only_the_mask() {
        let painter = StrokePainter::new(&test_mesh(), &test_camera(), viewport());
        let mut document = Document::new(test_mesh());
        let sizes = document.slot_sizes();
        document.set_mask(1, Some(255)).unwrap();
        // With erase on, a mask stroke hides the layer (paints black).
        let mut hide = brush(30.0, false);
        hide.erase = true;
        let mut stroke = Stroke::begin_on_mask(&document, 1).unwrap();
        let at = Pos2::new(76.0, 145.0);
        let changed =
            painter.paint_segment(&mut stroke, &mut document.layers[1], &sizes, at, at, &hide);
        assert!(!changed.is_empty());
        assert!(changed.iter().all(|(_, plane, _)| *plane == Plane::Mask));
        let layer = &document.layers[1];
        assert!(layer.textures[0][0].is_none(), "colour untouched");
        let mask = layer.textures[0][Plane::Mask.index()].as_ref().unwrap();
        assert!(
            mask.rgba.chunks_exact(4).any(|pixel| pixel[0] == 0),
            "hidden somewhere"
        );
        assert!(
            mask.rgba.chunks_exact(4).any(|pixel| pixel[0] == 255),
            "shown elsewhere"
        );
        assert!(
            mask.rgba.chunks_exact(4).all(|pixel| pixel[3] == 255),
            "masks stay opaque"
        );

        // Undo restores the unpainted mask.
        let mut history = crate::painting::history::History::default();
        history.push("Test", stroke.finish());
        history.undo(&mut document);
        let mask = document.layers[1].textures[0][Plane::Mask.index()]
            .as_ref()
            .unwrap();
        assert!(mask.rgba.chunks_exact(4).all(|pixel| pixel[0] == 255));

        // A layer without a mask can't be painted on as one.
        assert!(Stroke::begin_on_mask(&document, 0).is_none());
        document.set_mask(1, None).unwrap();
        assert!(Stroke::begin_on_mask(&document, 1).is_none());
    }

    #[test]
    fn fill_reveals_one_island_of_a_hide_all_mask() {
        let painter = StrokePainter::new(&test_mesh(), &test_camera(), viewport());
        let mut document = Document::new(test_mesh());
        let sizes = document.slot_sizes();
        document.set_mask(1, Some(0)).unwrap();
        let reveal = Brush {
            tool: Tool::Fill,
            fill_mode: FillMode::UvIsland,
            ..brush(1.0, false)
        };
        let mut stroke = Stroke::begin_on_mask(&document, 1).unwrap();
        let at = Pos2::new(76.0, 145.0);
        painter.fill(&mut stroke, &mut document.layers[1], &sizes, at, &reveal);
        let mask = document.layers[1].textures[0][Plane::Mask.index()]
            .as_ref()
            .unwrap();
        let value = |x: u32, y: u32| mask.rgba[((y * TEXTURE_SIZE + x) * 4) as usize];
        // The left island is revealed; the right one and empty space stay hidden.
        assert_eq!(value(20, 64), 255);
        assert_eq!(value(100, 64), 0);
        assert_eq!(value(64, 120), 0);
    }

    #[test]
    fn brush_rect_clips_a_triangle_to_the_part_it_covers() {
        use crate::painting::raster::uv_bounds;
        use glam::Vec2;
        let mesh = crate::mesh::primitives::cube();
        let painter = StrokePainter::new(&mesh, &test_camera(), viewport());
        // The front face's first triangle: seen head-on, it fills the middle.
        let triangle = painter
            .triangles
            .iter()
            .find(|t| t.normal.z > 0.9)
            .expect("front face");
        let (full_lo, full_hi) = uv_bounds(&triangle.uv);

        let everything = (Vec2::splat(-10.0), Vec2::splat(300.0));
        let (lo, hi) = painter
            .uv_bounds_in_screen_rect(triangle, everything.0, everything.1)
            .unwrap();
        assert!((lo - full_lo).length() < 1e-4 && (hi - full_hi).length() < 1e-4);

        // A small rectangle at the middle of the screen: a smaller part of the UVs.
        let (lo, hi) = painter
            .uv_bounds_in_screen_rect(triangle, Vec2::new(120.0, 120.0), Vec2::new(136.0, 136.0))
            .unwrap();
        assert!((hi - lo).max_element() < (full_hi - full_lo).max_element() * 0.5);
        assert!(lo.x >= full_lo.x - 1e-4 && hi.x <= full_hi.x + 1e-4);

        // Rectangles that miss the cube altogether, or are behind the screen edge.
        for (min, max) in [
            (Vec2::new(0.0, 0.0), Vec2::new(8.0, 8.0)),
            (Vec2::new(300.0, 300.0), Vec2::new(400.0, 400.0)),
        ] {
            assert!(
                painter
                    .uv_bounds_in_screen_rect(triangle, min, max)
                    .is_none()
            );
        }
    }

    /// Timings on a real model: `cargo test --release profile_car -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs car.glb in the project folder"]
    fn profile_car() {
        use glam::Vec2;
        use std::time::Instant;
        let mut mesh = crate::mesh::gltf_import::load(std::path::Path::new("car.glb")).unwrap();
        println!(
            "{} vertices, {} triangles, {} materials",
            mesh.vertices.len(),
            mesh.indices.len() / 3,
            mesh.materials.len()
        );
        for material in &mut mesh.materials {
            let image = &material.base_color;
            println!("  {}: {}x{}", material.name, image.width, image.height);
            if (image.width, image.height) == (1, 1) {
                let pixel = [image.rgba[0], image.rgba[1], image.rgba[2], image.rgba[3]];
                material.base_color = Image::filled(2048, 2048, pixel);
            }
        }
        let mut areas: Vec<f32> = mesh
            .indices
            .chunks_exact(3)
            .map(|t| {
                let uv = [0, 1, 2].map(|i| Vec2::from(mesh.vertices[t[i] as usize].uv) * 2048.0);
                crate::painting::raster::cross(uv[1] - uv[0], uv[2] - uv[0]).abs() * 0.5
            })
            .collect();
        areas.sort_by(f32::total_cmp);
        println!(
            "UV area in texels of 2048x2048: median {:.0}, 99% {:.0}, max {:.0}; {} triangles over 100k",
            areas[areas.len() / 2],
            areas[areas.len() * 99 / 100],
            areas[areas.len() - 1],
            areas.iter().filter(|a| **a > 100_000.0).count()
        );
        let (center, radius) = mesh.bounds();
        let mut camera = test_camera();
        camera.frame(center, radius);
        let viewport = Rect::from_min_max(Pos2::ZERO, Pos2::new(1000.0, 700.0));

        let start = Instant::now();
        let painter = StrokePainter::new(&mesh, &camera, viewport);
        println!("StrokePainter::new: {:?}", start.elapsed());

        let start = Instant::now();
        let mut document = Document::new(mesh);
        println!("Document::new: {:?}", start.elapsed());
        let start = Instant::now();
        let bytes = crate::project::encode(&document).unwrap();
        println!(
            "project::encode: {:?} ({} MB)",
            start.elapsed(),
            bytes.len() >> 20
        );
        let sizes = document.slot_sizes();
        {
            use crate::generator::{Generator, Pattern, bake::Bake};
            let start = Instant::now();
            let bake = Bake::new(&document.mesh, 3, 2048, 2048);
            println!(
                "Bake::new (2048^2, incl. convex edges and creases): {:?}",
                start.elapsed()
            );
            let start = Instant::now();
            let wear = Generator {
                pattern: Pattern::EdgeWear,
                ..Generator::default()
            };
            let values = bake.generate(&wear);
            println!("first edge wear generate: {:?}", start.elapsed());
            let start = Instant::now();
            bake.generate(&wear);
            println!("next edge wear generate: {:?}", start.elapsed());
            println!(
                "   lit texels: {}",
                values.iter().filter(|v| **v > 128).count()
            );
            let start = Instant::now();
            let cavity = bake.generate(&Generator {
                pattern: Pattern::Cavity,
                ..Generator::default()
            });
            println!("first cavity generate: {:?}", start.elapsed());
            println!(
                "   lit texels: {}",
                cavity.iter().filter(|v| **v > 128).count()
            );
        }
        let start = Instant::now();
        let hit = painter.surface_hit(Vec2::new(500.0, 350.0));
        println!("one surface_hit: {:?}", start.elapsed());
        if let Some(hit) = hit {
            let start = Instant::now();
            painter.connected_face(hit.triangle);
            println!("connected_face: {:?}", start.elapsed());
        }

        for screen_space in [false, true] {
            for textures in &mut document.layers[1].textures {
                *textures = Default::default();
            }
            let brush = Brush {
                size: 40.0,
                screen_space,
                color: [1.0, 0.0, 0.0],
                ..Brush::default()
            };
            let mut stroke = Stroke::begin(&document, 1).unwrap();
            let at = Pos2::new(500.0, 350.0);
            let start = Instant::now();
            painter.paint_segment(&mut stroke, &mut document.layers[1], &sizes, at, at, &brush);
            let elapsed = start.elapsed();
            let painted: usize = document.layers[1]
                .textures
                .iter()
                .flatten()
                .flatten()
                .map(|image| image.rgba.chunks_exact(4).filter(|p| p[3] > 0).count())
                .sum();
            println!("single dab screen_space={screen_space}: {painted} texels in {elapsed:?}");
            for (slot, planes) in document.layers[1].textures.iter().enumerate() {
                if let Some(image) = &planes[0] {
                    let count = image.rgba.chunks_exact(4).filter(|p| p[3] > 0).count();
                    if count > 0 {
                        println!("     slot {slot}: {count}");
                    }
                }
            }
        }
        for (size, screen_space, over_edges) in [
            (40.0, false, true),
            (150.0, false, true),
            (40.0, true, true),
            (150.0, true, true),
            (40.0, false, false),
        ] {
            let brush = Brush {
                size,
                screen_space,
                paint_over_edges: over_edges,
                color: [1.0, 0.0, 0.0],
                ..Brush::default()
            };
            for textures in &mut document.layers[1].textures {
                *textures = Default::default();
            }
            let mut stroke = Stroke::begin(&document, 1).unwrap();
            let mut times = Vec::new();
            for step in 0..6 {
                let from = Pos2::new(400.0 + step as f32 * 12.0, 350.0);
                let to = Pos2::new(from.x + 12.0, 350.0 + (step % 3) as f32);
                let start = Instant::now();
                painter.paint_segment(
                    &mut stroke,
                    &mut document.layers[1],
                    &sizes,
                    from,
                    to,
                    &brush,
                );
                times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            let painted: usize = document.layers[1]
                .textures
                .iter()
                .flatten()
                .flatten()
                .map(|image| image.rgba.chunks_exact(4).filter(|p| p[3] > 0).count())
                .sum();
            println!("   painted texels: {painted}");
            times.sort_by(f64::total_cmp);
            println!(
                "size {size:>5} screen_space={screen_space:<5} over_edges={over_edges:<5}: \
                 median {:.1} ms, worst {:.1} ms",
                times[times.len() / 2],
                times[times.len() - 1]
            );
        }
    }
}
