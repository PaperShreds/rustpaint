//! Procedural patterns that fill layers and masks.
//!
//! A `Generator` is a small set of parameters that turns a point on the model
//! into a number from 0 to 1. Patterns are evaluated in object space, from
//! `Bake`d positions, so they run across UV seams without a break and don't
//! stretch with the UV layout.

pub mod bake;
pub mod edges;
pub mod noise;

use glam::Vec3;

/// The model's bounding box: what patterns are measured against, so a
/// pattern looks the same on a model whatever its size or position.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub min: Vec3,
    pub max: Vec3,
}

impl Frame {
    pub fn of(mesh: &crate::mesh::Mesh) -> Self {
        let mut frame = Self {
            min: Vec3::splat(f32::MAX),
            max: Vec3::splat(f32::MIN),
        };
        for vertex in &mesh.vertices {
            let position = Vec3::from(vertex.position);
            if position.is_finite() {
                frame.min = frame.min.min(position);
                frame.max = frame.max.max(position);
            }
        }
        if frame.min.x > frame.max.x {
            frame = Self {
                min: Vec3::NEG_ONE,
                max: Vec3::ONE,
            };
        }
        frame
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half the box's diagonal, never zero.
    pub fn radius(&self) -> f32 {
        ((self.max - self.min) * 0.5).length().max(1e-4)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Pattern {
    /// Soft blobs.
    Value,
    /// Cloudy, even noise.
    #[default]
    Perlin,
    /// Cells: dark in the middle of each, light at the borders.
    Cells,
    /// A ramp from one end of the model to the other.
    Gradient,
    /// Light along the model's convex edges, broken up by noise: worn paint.
    EdgeWear,
}

impl Pattern {
    pub const ALL: [Pattern; 5] = [
        Pattern::Value,
        Pattern::Perlin,
        Pattern::Cells,
        Pattern::Gradient,
        Pattern::EdgeWear,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Pattern::Value => "Value noise",
            Pattern::Perlin => "Perlin noise",
            Pattern::Cells => "Cells",
            Pattern::Gradient => "Gradient",
            Pattern::EdgeWear => "Edge wear",
        }
    }

    /// Stable number for files.
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|p| *p == self).unwrap_or(0) as u8
    }

    pub fn from_index(index: u8) -> Option<Self> {
        Self::ALL.get(index as usize).copied()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Axis {
    X,
    #[default]
    Y,
    Z,
}

impl Axis {
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    pub fn name(self) -> &'static str {
        match self {
            Axis::X => "X",
            Axis::Y => "Y",
            Axis::Z => "Z",
        }
    }

    pub fn index(self) -> u8 {
        self as u8
    }

    pub fn from_index(index: u8) -> Option<Self> {
        Self::ALL.get(index as usize).copied()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Generator {
    pub pattern: Pattern,
    /// Noise: how many features fit across the model.
    pub scale: f32,
    /// Noise: how many layers of finer detail are added.
    pub octaves: u32,
    pub seed: u32,
    /// Gradient: the direction it runs in (from 0 at the model's lowest point
    /// on this axis to 1 at its highest).
    pub axis: Axis,
    /// Edge wear: how far from an edge it reaches, in model radii.
    pub edge_width: f32,
    /// Edge wear: how much the noise (set by scale, detail and seed) eats
    /// into the wear, 0 for none.
    pub breakup: f32,
    /// The pattern values that become 0 and 1: raising `low` or lowering
    /// `high` adds contrast; moving them together shifts the threshold.
    pub low: f32,
    pub high: f32,
    pub invert: bool,
}

impl Default for Generator {
    fn default() -> Self {
        Self {
            pattern: Pattern::default(),
            scale: 4.0,
            octaves: 4,
            seed: 0,
            axis: Axis::default(),
            edge_width: 0.03,
            breakup: 0.4,
            low: 0.0,
            high: 1.0,
            invert: false,
        }
    }
}

impl Generator {
    /// The pattern's value (0..1) at `point`, a position in the model's own
    /// space that `frame` encloses. `edge_distance` is how far the point is
    /// from the nearest convex edge (see `ConvexEdges::distance`); only edge
    /// wear looks at it.
    pub fn sample(&self, point: Vec3, frame: &Frame, edge_distance: f32) -> f32 {
        let noise_at = |base: fn(Vec3, u32) -> f32| {
            // Measured in model-radius units, so scale 1 is about one
            // feature across the whole model.
            let local = (point - frame.center()) / frame.radius() * self.scale;
            noise::fractal(base, local, self.octaves, self.seed)
        };
        let raw = match self.pattern {
            Pattern::EdgeWear => {
                let width = (self.edge_width * frame.radius()).max(1e-6);
                // 1 on an edge, falling to 0 at `width`; the noise moves that line.
                let near = 1.0 - edge_distance / width;
                (near + (noise_at(noise::perlin) - 0.5) * 2.0 * self.breakup).clamp(0.0, 1.0)
            }
            Pattern::Gradient => {
                let axis = self.axis as usize;
                let range = frame.max[axis] - frame.min[axis];
                if range > 1e-6 {
                    (point[axis] - frame.min[axis]) / range
                } else {
                    0.0
                }
            }
            noise_pattern => noise_at(match noise_pattern {
                Pattern::Value => noise::value,
                Pattern::Cells => noise::cells,
                _ => noise::perlin,
            }),
        };
        let leveled = ((raw - self.low) / (self.high - self.low).max(1e-4)).clamp(0.0, 1.0);
        if self.invert { 1.0 - leveled } else { leveled }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: Frame = Frame {
        min: Vec3::new(-2.0, 0.0, -1.0),
        max: Vec3::new(2.0, 4.0, 1.0),
    };

    #[test]
    fn gradient_runs_from_one_end_of_the_model_to_the_other() {
        let generator = Generator {
            pattern: Pattern::Gradient,
            axis: Axis::Y,
            ..Generator::default()
        };
        assert_eq!(
            generator.sample(Vec3::new(0.0, 0.0, 0.0), &FRAME, f32::INFINITY),
            0.0
        );
        assert_eq!(
            generator.sample(Vec3::new(9.0, 2.0, 0.0), &FRAME, f32::INFINITY),
            0.5
        );
        assert_eq!(
            generator.sample(Vec3::new(0.0, 4.0, 0.0), &FRAME, f32::INFINITY),
            1.0
        );
        let inverted = Generator {
            invert: true,
            ..generator
        };
        assert_eq!(
            inverted.sample(Vec3::new(0.0, 4.0, 0.0), &FRAME, f32::INFINITY),
            0.0
        );
    }

    #[test]
    fn levels_add_contrast_and_every_value_stays_in_range() {
        let soft = Generator::default();
        let hard = Generator {
            low: 0.45,
            high: 0.55,
            ..soft
        };
        let mut extremes = 0;
        for i in 0..400 {
            let p = Vec3::new(i as f32 * 0.031, i as f32 * 0.017, i as f32 * 0.009);
            let (a, b) = (
                soft.sample(p, &FRAME, f32::INFINITY),
                hard.sample(p, &FRAME, f32::INFINITY),
            );
            assert!((0.0..=1.0).contains(&a) && (0.0..=1.0).contains(&b));
            if b == 0.0 || b == 1.0 {
                extremes += 1;
            }
        }
        assert!(extremes > 200, "levels barely change anything: {extremes}");
    }

    #[test]
    fn the_pattern_does_not_depend_on_the_model_size() {
        let generator = Generator::default();
        let big = Frame {
            min: FRAME.min * 10.0,
            max: FRAME.max * 10.0,
        };
        for i in 0..50 {
            let p = Vec3::new(i as f32 * 0.03, i as f32 * 0.05, i as f32 * 0.01);
            let (a, b) = (
                generator.sample(p, &FRAME, f32::INFINITY),
                generator.sample(p * 10.0, &big, f32::INFINITY),
            );
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn pattern_and_axis_numbers_round_trip() {
        for pattern in Pattern::ALL {
            assert_eq!(Pattern::from_index(pattern.index()), Some(pattern));
        }
        for axis in Axis::ALL {
            assert_eq!(Axis::from_index(axis.index()), Some(axis));
        }
        assert_eq!(Pattern::from_index(99), None);
    }
}
