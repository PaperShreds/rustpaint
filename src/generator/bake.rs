//! Where on the model each texel of a material's textures is.
//!
//! Drawing a triangle into UV space and writing its 3D position instead of a
//! colour gives a "position map". A pattern evaluated at those positions
//! doesn't care where the UV islands lie, so it has no seams.

use super::edges::{Edges, Fold};
use super::{Frame, Recipe};
use crate::mesh::{Image, LEVEL, Mesh};
use crate::painting::raster::{
    SEAM_MARGIN_TEXELS, barycentric_2d, edge_margins, interpolate, texel_range, uv_bounds,
};
use glam::{Vec2, Vec3};
use std::sync::{Arc, Mutex, OnceLock};

/// How sure we are that a texel belongs to the model.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Cover {
    None,
    /// Just outside a triangle: the seam margin that stops hairline gaps.
    Margin,
    Inside,
}

/// The model-space position of every texel of one material slot.
pub struct Bake {
    pub width: u32,
    pub height: u32,
    pub frame: Frame,
    positions: Vec<Vec3>,
    /// The surface's direction at each texel: unit length, or zero where
    /// the model has no normal.
    normals: Vec<Vec3>,
    /// How far one UV unit along u and along v reaches across the model at
    /// each texel, in the model's units.
    stretch: Vec<[f32; 2]>,
    cover: Vec<Cover>,
    edges: Edges,
    /// Distance from each texel to the nearest convex edge and the nearest
    /// crease, each found when a pattern first needs it.
    fold_distance: [OnceLock<Vec<f32>>; 2],
    /// The latest results of `generate_cached`, newest last.
    recent: Mutex<Vec<(Recipe, Arc<Vec<u8>>)>>,
}

/// How many generated results a bake keeps for reuse.
const RECENT_RESULTS: usize = 4;

impl Bake {
    pub fn new(mesh: &Mesh, material: usize, width: u32, height: u32) -> Self {
        let count = width as usize * height as usize;
        let mut bake = Self {
            width,
            height,
            frame: Frame::of(mesh),
            edges: Edges::new(mesh, &Frame::of(mesh)),
            fold_distance: [OnceLock::new(), OnceLock::new()],
            recent: Mutex::new(Vec::new()),
            positions: vec![Vec3::ZERO; count],
            normals: vec![Vec3::ZERO; count],
            stretch: vec![[0.0; 2]; count],
            cover: vec![Cover::None; count],
        };
        let size = Vec2::new(width as f32, height as f32);
        for submesh in mesh.submeshes.iter().filter(|s| s.material == material) {
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
                let normal = [0, 1, 2].map(|i| Vec3::from(vertices[i].normal));
                let uv = [0, 1, 2].map(|i| Vec2::from(vertices[i].uv) * size);
                if position.iter().all(|v| v.is_finite()) {
                    bake.rasterize(position, normal, uv);
                }
            }
        }
        bake
    }

    fn rasterize(&mut self, position: [Vec3; 3], normal: [Vec3; 3], corners: [Vec2; 3]) {
        let Some(margins) = edge_margins(corners, SEAM_MARGIN_TEXELS) else {
            return;
        };
        let stretch = uv_stretch(position, corners, self.width, self.height);
        let (lo, hi) = uv_bounds(&corners);
        let (min_x, max_x) = texel_range(
            lo.x - SEAM_MARGIN_TEXELS,
            hi.x + SEAM_MARGIN_TEXELS,
            self.width,
        );
        let (min_y, max_y) = texel_range(
            lo.y - SEAM_MARGIN_TEXELS,
            hi.y + SEAM_MARGIN_TEXELS,
            self.height,
        );
        for raw_y in min_y..=max_y {
            for raw_x in min_x..=max_x {
                let center = Vec2::new(raw_x as f32 + 0.5, raw_y as f32 + 0.5);
                let Some(weights) = barycentric_2d(center, corners) else {
                    return;
                };
                let cover = if weights.iter().all(|w| *w >= 0.0) {
                    Cover::Inside
                } else if weights.iter().zip(margins).all(|(w, margin)| *w >= -margin) {
                    Cover::Margin
                } else {
                    continue;
                };
                let x = raw_x.rem_euclid(self.width as i64) as usize;
                let y = raw_y.rem_euclid(self.height as i64) as usize;
                let texel = y * self.width as usize + x;
                // A texel on a triangle always beats one in a neighbour's margin.
                if cover > self.cover[texel] {
                    self.cover[texel] = cover;
                    self.positions[texel] = interpolate(position, weights);
                    let normal = interpolate(normal, weights).normalize_or_zero();
                    self.normals[texel] = if normal.is_finite() {
                        normal
                    } else {
                        Vec3::ZERO
                    };
                    self.stretch[texel] = stretch;
                }
            }
        }
    }

    #[cfg(test)]
    /// Whether the texel at (`x`, `y`) lies on the model (or in a seam margin).
    pub fn covers(&self, x: u32, y: u32) -> bool {
        self.cover[y as usize * self.width as usize + x as usize] != Cover::None
    }

    #[cfg(test)]
    /// How far one UV unit along u and along v reaches across the model at
    /// the texel at (`x`, `y`), in the model's units; zero off the model.
    pub fn stretch(&self, x: u32, y: u32) -> [f32; 2] {
        self.stretch[y as usize * self.width as usize + x as usize]
    }

    #[cfg(test)]
    /// The model-space position of a covered texel.
    pub fn position(&self, x: u32, y: u32) -> Vec3 {
        self.positions[y as usize * self.width as usize + x as usize]
    }

    #[cfg(test)]
    /// `generate_recipe` for a single pattern.
    pub fn generate(&self, generator: &super::Generator) -> Vec<u8> {
        self.generate_recipe(&Recipe::from(*generator))
    }

    /// Runs `recipe` on every covered texel: its value as a byte (0..255),
    /// row by row. Texels off the model are 0. Spread over the CPU cores.
    pub fn generate_recipe(&self, recipe: &Recipe) -> Vec<u8> {
        let folds = [Fold::Convex, Fold::Concave].map(|fold| {
            recipe.uses(fold).then(|| {
                self.fold_distance[fold as usize].get_or_init(|| self.measure_fold_distance(fold))
            })
        });
        let mut values = vec![0u8; self.cover.len()];
        self.for_each_texel(&mut values, |texel, value| {
            let near = folds.map(|distances| distances.map_or(f32::INFINITY, |d| d[texel]));
            let v = recipe.sample(
                self.positions[texel],
                self.normals[texel],
                &self.frame,
                near,
            );
            *value = (v * 255.0).round() as u8;
        });
        values
    }

    /// A tangent-space normal map of `height` (an image of this bake's size,
    /// value in red): the bumps the heights make on the surface, where a
    /// height of 1 stands `depth` model units out from one of 0. As glTF
    /// expects: red to +u, green "up" in the image (to -v), blue out of
    /// the surface. Texels off the model face straight out.
    pub fn normal_map(&self, height: &Image, depth: f32) -> Image {
        const FLAT: [u8; 4] = [128, 128, 255, 255];
        let mut normals = Image::filled(self.width, self.height, FLAT);
        if (height.width, height.height) != (self.width, self.height) {
            return normals;
        }
        let (width, rows) = (self.width as i64, self.height as i64);
        let value = |x: i64, y: i64| {
            let pixel = &height.rgba[(y * width + x) as usize * 4..][..4];
            if pixel[3] == 0 {
                LEVEL
            } else {
                pixel[0] as f32 / 255.0
            }
        };
        self.for_each_texel(normals.rgba.as_chunks_mut::<4>().0, |texel, out| {
            let (x, y) = ((texel as i64) % width, (texel as i64) / width);
            let [along_u, along_v] = self.stretch[texel];
            if along_u <= 0.0 || along_v <= 0.0 {
                return;
            }
            let center = value(x, y);
            // Off the model a neighbour's height means nothing: use this one's.
            let at = |dx: i64, dy: i64| {
                let (nx, ny) = ((x + dx).rem_euclid(width), (y + dy).rem_euclid(rows));
                if self.cover[(ny * width + nx) as usize] == Cover::None {
                    center
                } else {
                    value(nx, ny)
                }
            };
            // Sobel: the slope per texel, smoothed across, so 8-bit steps
            // don't show as stripes. Opposite texels are subtracted first,
            // so level ground gives exactly zero.
            let slope_x =
                ((at(1, -1) - at(-1, -1)) + 2.0 * (at(1, 0) - at(-1, 0)) + (at(1, 1) - at(-1, 1)))
                    / 8.0;
            let slope_y =
                ((at(-1, 1) - at(-1, -1)) + 2.0 * (at(0, 1) - at(0, -1)) + (at(1, 1) - at(1, -1)))
                    / 8.0;
            // Per UV unit, then per model unit along the surface.
            let per_u = slope_x * self.width as f32 / along_u;
            let per_v = slope_y * self.height as f32 / along_v;
            let normal = Vec3::new(-depth * per_u, depth * per_v, 1.0).normalize();
            let encode = |c: f32| ((c * 0.5 + 0.5) * 255.0).round() as u8;
            *out = [encode(normal.x), encode(normal.y), encode(normal.z), 255];
        });
        normals
    }

    /// `generate_recipe`, but remembers the last few results: changing only
    /// what is made *from* a pattern (a colour, say) doesn't run it again.
    pub fn generate_cached(&self, recipe: &Recipe) -> Arc<Vec<u8>> {
        let lock = || self.recent.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, values)) = lock().iter().find(|(r, _)| r == recipe) {
            return values.clone();
        }
        let values = Arc::new(self.generate_recipe(recipe));
        let mut recent = lock();
        if recent.len() >= RECENT_RESULTS {
            recent.remove(0);
        }
        recent.push((*recipe, values.clone()));
        values
    }

    fn measure_fold_distance(&self, fold: Fold) -> Vec<f32> {
        let mut distances = vec![f32::INFINITY; self.cover.len()];
        self.for_each_texel(&mut distances, |texel, distance| {
            *distance = self.edges.distance(self.positions[texel], fold);
        });
        distances
    }

    /// Calls `f(texel index, its slot in `out`)` for every covered texel,
    /// spread over the CPU cores.
    fn for_each_texel<T: Send>(&self, out: &mut [T], f: impl Fn(usize, &mut T) + Sync) {
        let width = self.width as usize;
        let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
        let rows = (self.height as usize).div_ceil(threads).max(1);
        let f = &f;
        std::thread::scope(|scope| {
            for (part, chunk) in out.chunks_mut(rows * width).enumerate() {
                scope.spawn(move || {
                    for (offset, slot) in chunk.iter_mut().enumerate() {
                        let texel = part * rows * width + offset;
                        if self.cover[texel] != Cover::None {
                            f(texel, slot);
                        }
                    }
                });
            }
        });
    }
}

/// How far a triangle reaches across the model per UV unit along u and
/// along v: the lengths of its position's derivatives. `corners` are in
/// texels of a `width` x `height` texture.
fn uv_stretch(position: [Vec3; 3], corners: [Vec2; 3], width: u32, height: u32) -> [f32; 2] {
    let scale = Vec2::new(width as f32, height as f32);
    let (e1, e2) = (position[1] - position[0], position[2] - position[0]);
    let (d1, d2) = (
        (corners[1] - corners[0]) / scale,
        (corners[2] - corners[0]) / scale,
    );
    let area = d1.x * d2.y - d2.x * d1.y;
    if area.abs() < 1e-12 {
        return [0.0; 2];
    }
    let along_u = (e1 * d2.y - e2 * d1.y) / area;
    let along_v = (e2 * d1.x - e1 * d2.x) / area;
    [along_u.length(), along_v.length()].map(|l| if l.is_finite() { l } else { 0.0 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::{Axis, Generator, Pattern};
    use crate::mesh::primitives;

    fn bake() -> Bake {
        // The cube's atlas is 3x2 cells of 256 texels.
        Bake::new(&primitives::cube(), 0, 768, 512)
    }

    #[test]
    fn texels_on_the_model_know_their_position() {
        let bake = bake();
        let covered = (0..bake.height)
            .flat_map(|y| (0..bake.width).map(move |x| (x, y)))
            .filter(|&(x, y)| bake.covers(x, y))
            .count();
        // Six 248-texel-wide faces plus margins, out of 3x2 cells.
        assert!(covered > 6 * 240 * 240 && covered < 768 * 512, "{covered}");
        // Every covered texel sits on the cube's surface (some coordinate +-0.5).
        for y in (0..bake.height).step_by(7) {
            for x in (0..bake.width).step_by(7) {
                if bake.covers(x, y) {
                    let p = bake.position(x, y);
                    assert!(
                        (p.abs().max_element() - 0.5).abs() < 0.02,
                        "{p:?} is off the cube"
                    );
                }
            }
        }
        // The padding between faces is not covered far from any face.
        assert!(!bake.covers(0, 0));
    }

    #[test]
    fn generating_fills_the_model_and_nothing_else() {
        let bake = bake();
        let values = bake.generate(&Generator::default());
        assert_eq!(values.len(), 768 * 512);
        let mut seen = std::collections::HashSet::new();
        for y in 0..bake.height {
            for x in 0..bake.width {
                let value = values[(y * bake.width + x) as usize];
                if !bake.covers(x, y) {
                    assert_eq!(value, 0, "value outside the UV islands");
                } else {
                    seen.insert(value);
                }
            }
        }
        assert!(
            seen.len() > 50,
            "the noise should vary: {} values",
            seen.len()
        );
    }

    #[test]
    fn the_same_generator_gives_the_same_texels() {
        let bake = bake();
        let generator = Generator {
            pattern: Pattern::Cells,
            seed: 9,
            ..Generator::default()
        };
        assert_eq!(bake.generate(&generator), bake.generate(&generator));
        let other = Generator {
            seed: 10,
            ..generator
        };
        assert_ne!(bake.generate(&generator), bake.generate(&other));
    }

    #[test]
    fn a_repeated_pattern_is_reused() {
        let bake = Bake::new(&primitives::cube(), 0, 64, 64);
        let generator = Generator::default();
        let recipe = Recipe::from(generator);
        let first = bake.generate_cached(&recipe);
        assert!(Arc::ptr_eq(&first, &bake.generate_cached(&recipe)));
        assert_eq!(*first, bake.generate(&generator));
        let other = Generator {
            seed: 9,
            ..generator
        };
        assert_ne!(first, bake.generate_cached(&Recipe::from(other)));
    }

    #[test]
    fn edge_wear_hugs_the_edges_of_the_model() {
        let bake = bake();
        let wear = Generator {
            pattern: Pattern::EdgeWear,
            edge_width: 0.05,
            breakup: 0.0,
            ..Generator::default()
        };
        let values = bake.generate(&wear);
        let radius = bake.frame.radius();
        let (mut lit, mut checked) = (0, 0);
        for y in (0..bake.height).step_by(3) {
            for x in (0..bake.width).step_by(3) {
                if !bake.covers(x, y) {
                    continue;
                }
                let p = bake.position(x, y);
                // How far this texel is from the nearest of the cube's edges
                // (where two of the coordinates are +-0.5).
                let from_face = p.abs().to_array().map(|c| 0.5 - c);
                let mut sorted = from_face;
                sorted.sort_by(f32::total_cmp);
                let distance = (sorted[0] * sorted[0] + sorted[1] * sorted[1]).sqrt();
                let value = values[(y * bake.width + x) as usize];
                if distance < 0.02 * radius {
                    assert!(value > 150, "{value} near an edge ({distance})");
                    lit += 1;
                } else if distance > 0.07 * radius {
                    assert_eq!(value, 0, "{value} away from the edges ({distance})");
                }
                checked += 1;
            }
        }
        assert!(lit > 100 && checked > lit, "{lit} of {checked}");
        // The noise eats into the wear, so it covers less.
        let broken = bake.generate(&Generator {
            breakup: 0.6,
            ..wear
        });
        let sum = |v: &[u8]| v.iter().map(|&b| b as u64).sum::<u64>();
        assert_ne!(sum(&broken), sum(&values));
    }

    #[test]
    fn cavity_lights_the_creases_and_edge_wear_the_edges() {
        let lit = |mesh: &Mesh, pattern| {
            let bake = Bake::new(mesh, 0, 768, 512);
            let values = bake.generate(&Generator {
                pattern,
                edge_width: 0.05,
                breakup: 0.0,
                ..Generator::default()
            });
            values.iter().filter(|&&value| value > 150).count()
        };
        let outside = primitives::cube();
        let mut inside = primitives::cube();
        for triangle in inside.indices.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
        // A box has edges and no creases; turned inside out, the other way round.
        assert!(lit(&outside, Pattern::EdgeWear) > 1000);
        assert_eq!(lit(&outside, Pattern::Cavity), 0);
        assert_eq!(lit(&inside, Pattern::EdgeWear), 0);
        assert!(lit(&inside, Pattern::Cavity) > 1000);
    }

    #[test]
    fn bricks_cover_every_face_with_mortar_between_them() {
        let bake = bake();
        let values = bake.generate(&Generator {
            pattern: Pattern::Bricks,
            bevel: 0.0,
            ..Generator::default()
        });
        // Count brick and mortar texels on each of the cube's six faces.
        let mut faces = [[0usize; 2]; 6];
        for y in 0..bake.height {
            for x in 0..bake.width {
                if !bake.covers(x, y) {
                    continue;
                }
                let p = bake.position(x, y);
                let axis = (0..3)
                    .max_by(|&a, &b| p[a].abs().total_cmp(&p[b].abs()))
                    .unwrap();
                let face = axis * 2 + (p[axis] > 0.0) as usize;
                let value = values[(y * bake.width + x) as usize];
                assert!(value == 0 || value == 255, "a hard edge: {value}");
                faces[face][(value == 255) as usize] += 1;
            }
        }
        for [mortar, bricks] in faces {
            // Walls: mortar on 0.1 of each row and of each brick's length,
            // about 15%. The top and bottom cut through a row, so only show
            // the joints between bricks, about 10%.
            let share = mortar as f32 / (mortar + bricks) as f32;
            assert!(
                (0.05..0.25).contains(&share),
                "{mortar} mortar, {bricks} brick"
            );
        }
    }

    #[test]
    fn the_stretch_is_how_far_the_texture_spans_across_the_model() {
        let bake = bake();
        // Each face is a 1x1 square in a 256-texel cell of a 768x512 atlas
        // (with a few texels of padding): about 3 and 2 per UV unit.
        let [along_u, along_v] = bake.stretch(128, 128);
        assert!((along_u - 3.0).abs() < 0.2, "{along_u}");
        assert!((along_v - 2.0).abs() < 0.2, "{along_v}");
        assert_eq!(bake.stretch(0, 0), [0.0; 2], "off the model");
    }

    #[test]
    fn a_normal_map_leans_away_from_rising_ground() {
        let bake = bake();
        let ramp = |slope_x: f32, slope_y: f32| {
            let mut height = Image::filled(bake.width, bake.height, [0, 0, 0, 255]);
            for (texel, pixel) in height.rgba.chunks_exact_mut(4).enumerate() {
                let (x, y) = (texel as u32 % bake.width, texel as u32 / bake.width);
                let value = 0.5 + slope_x * x as f32 + slope_y * y as f32;
                pixel[..3].fill((value.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
            bake.normal_map(&height, 0.5)
        };
        let at = |image: &Image, x: u32, y: u32| {
            let i = ((y * image.width + x) * 4) as usize;
            [image.rgba[i], image.rgba[i + 1], image.rgba[i + 2]]
        };
        // Level ground faces straight out.
        assert_eq!(at(&ramp(0.0, 0.0), 128, 128), [128, 128, 255]);
        // Rising 1.2 per UV unit to the right (+u), which spans 3 model
        // units: a slope of 0.4 * depth 0.5. The surface leans left: red
        // about 0.5 - 0.2 / 2.
        let [red, green, _] = at(&ramp(0.4 / 256.0, 0.0), 128, 128);
        assert!((95..115).contains(&red) && green == 128, "{red} {green}");
        // Rising downwards in the image (+v): it leans up, green above half.
        let [red, green, _] = at(&ramp(0.0, 0.4 / 256.0), 128, 128);
        assert!(red == 128 && (140..160).contains(&green), "{red} {green}");
        assert_eq!(at(&ramp(0.4, 0.4), 0, 0), [128, 128, 255], "off the model");
    }

    #[test]
    fn a_gradient_follows_the_model_not_the_uv_layout() {
        let bake = bake();
        let values = bake.generate(&Generator {
            pattern: Pattern::Gradient,
            axis: Axis::Y,
            ..Generator::default()
        });
        for y in (0..bake.height).step_by(5) {
            for x in (0..bake.width).step_by(5) {
                if bake.covers(x, y) {
                    let expected = (bake.position(x, y).y + 0.5) * 255.0;
                    let got = values[(y * bake.width + x) as usize] as f32;
                    assert!((got - expected).abs() <= 1.0, "{got} vs {expected}");
                }
            }
        }
    }
}
