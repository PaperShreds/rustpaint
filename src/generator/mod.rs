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
    /// Light in the model's concave creases, broken up by noise: dirt.
    Cavity,
}

impl Pattern {
    pub const ALL: [Pattern; 6] = [
        Pattern::Value,
        Pattern::Perlin,
        Pattern::Cells,
        Pattern::Gradient,
        Pattern::EdgeWear,
        Pattern::Cavity,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Pattern::Value => "Value noise",
            Pattern::Perlin => "Perlin noise",
            Pattern::Cells => "Cells",
            Pattern::Gradient => "Gradient",
            Pattern::EdgeWear => "Edge wear",
            Pattern::Cavity => "Cavity",
        }
    }

    /// Stable number for files.
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|p| *p == self).unwrap_or(0) as u8
    }

    pub fn from_index(index: u8) -> Option<Self> {
        Self::ALL.get(index as usize).copied()
    }

    /// The kind of fold the pattern follows, if it follows one.
    pub fn fold(self) -> Option<edges::Fold> {
        match self {
            Pattern::EdgeWear => Some(edges::Fold::Convex),
            Pattern::Cavity => Some(edges::Fold::Concave),
            _ => None,
        }
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
    /// Edge wear and cavity: how far from an edge or crease it reaches, in
    /// model radii.
    pub edge_width: f32,
    /// Edge wear and cavity: how much the noise (set by scale, detail and
    /// seed) eats into the pattern, 0 for none.
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
    /// from the nearest fold of the kind `Pattern::fold` names (see
    /// `Edges::distance`); only patterns with a fold look at it.
    pub fn sample(&self, point: Vec3, frame: &Frame, edge_distance: f32) -> f32 {
        let noise_at = |base: fn(Vec3, u32) -> f32| {
            // Measured in model-radius units, so scale 1 is about one
            // feature across the whole model.
            let local = (point - frame.center()) / frame.radius() * self.scale;
            noise::fractal(base, local, self.octaves, self.seed)
        };
        let raw = match self.pattern {
            Pattern::EdgeWear | Pattern::Cavity => {
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

/// How a pattern is combined with the result so far.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Op {
    /// Keeps only what both have: the second pattern cuts holes in the first.
    #[default]
    Multiply,
    Add,
    Subtract,
    /// The brighter of the two at every point.
    Lighter,
    /// The darker of the two at every point.
    Darker,
}

impl Op {
    pub const ALL: [Op; 5] = [Op::Multiply, Op::Add, Op::Subtract, Op::Lighter, Op::Darker];

    pub fn name(self) -> &'static str {
        match self {
            Op::Multiply => "Multiply",
            Op::Add => "Add",
            Op::Subtract => "Subtract",
            Op::Lighter => "Lighter",
            Op::Darker => "Darker",
        }
    }

    /// Stable number for files.
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|op| *op == self).unwrap_or(0) as u8
    }

    pub fn from_index(index: u8) -> Option<Self> {
        Self::ALL.get(index as usize).copied()
    }

    /// Combines the result so far (`below`) with a pattern's value (`top`).
    pub fn apply(self, below: f32, top: f32) -> f32 {
        match self {
            Op::Multiply => below * top,
            Op::Add => (below + top).min(1.0),
            Op::Subtract => (below - top).max(0.0),
            Op::Lighter => below.max(top),
            Op::Darker => below.min(top),
        }
    }
}

/// A pattern laid over the ones before it in a `Recipe`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step {
    pub op: Op,
    /// 0..1: how much of the combined result replaces what was there.
    pub amount: f32,
    pub generator: Generator,
}

/// The most patterns that go on top of a recipe's first one.
pub const MAX_STEPS: usize = 3;

/// What decides a value from 0 to 1 at every point of the model: one pattern,
/// with up to `MAX_STEPS` more combined into it one after the other.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Recipe {
    pub base: Generator,
    /// Filled from the front.
    steps: [Option<Step>; MAX_STEPS],
}

impl From<Generator> for Recipe {
    fn from(base: Generator) -> Self {
        Self {
            base,
            steps: [None; MAX_STEPS],
        }
    }
}

impl Recipe {
    /// A recipe from a first pattern and the patterns combined into it
    /// (those beyond `MAX_STEPS` are left out).
    pub fn with_steps(base: Generator, steps: impl IntoIterator<Item = Step>) -> Self {
        let mut recipe = Self::from(base);
        for (slot, step) in recipe.steps.iter_mut().zip(steps) {
            *slot = Some(step);
        }
        recipe
    }

    pub fn steps(&self) -> impl Iterator<Item = &Step> {
        self.steps.iter().flatten()
    }

    pub fn step_count(&self) -> usize {
        self.steps().count()
    }

    pub fn step_mut(&mut self, index: usize) -> Option<&mut Step> {
        self.steps.get_mut(index)?.as_mut()
    }

    /// Adds a pattern: finer noise that cuts into the result, to start from.
    /// Returns false if the recipe is full.
    pub fn add_step(&mut self) -> bool {
        let count = self.step_count();
        let Some(slot) = self.steps.get_mut(count) else {
            return false;
        };
        *slot = Some(Step {
            op: Op::Multiply,
            amount: 1.0,
            generator: Generator {
                scale: 16.0,
                seed: self.base.seed.wrapping_add(count as u32 + 1),
                ..Generator::default()
            },
        });
        true
    }

    pub fn remove_step(&mut self, index: usize) {
        if index < MAX_STEPS {
            self.steps[index] = None;
            // Close the gap.
            self.steps[index..].rotate_left(1);
        }
    }

    /// Whether any pattern follows folds of this kind (so needs distances).
    pub fn uses(&self, fold: edges::Fold) -> bool {
        std::iter::once(&self.base)
            .chain(self.steps().map(|step| &step.generator))
            .any(|generator| generator.pattern.fold() == Some(fold))
    }

    /// The value (0..1) at `point`; `near` is how far it is from the nearest
    /// convex edge and the nearest crease (see `Generator::sample`).
    pub fn sample(&self, point: Vec3, frame: &Frame, near: [f32; 2]) -> f32 {
        let value = |generator: &Generator| {
            let distance = generator
                .pattern
                .fold()
                .map_or(f32::INFINITY, |fold| near[fold as usize]);
            generator.sample(point, frame, distance)
        };
        self.steps().fold(value(&self.base), |below, step| {
            let combined = step.op.apply(below, value(&step.generator));
            below + (combined - below) * step.amount.clamp(0.0, 1.0)
        })
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

    #[test]
    fn a_second_pattern_cuts_into_the_first() {
        let blotches = Generator {
            low: 0.4,
            high: 0.6,
            ..Generator::default()
        };
        let mut recipe = Recipe::from(blotches);
        assert!(recipe.add_step());
        assert_eq!(recipe.step_count(), 1);
        let none = [f32::INFINITY; 2];
        let (mut cut, mut equal) = (0, 0);
        for i in 0..400 {
            let p = Vec3::new(i as f32 * 0.031, i as f32 * 0.017, i as f32 * 0.009);
            let alone = Recipe::from(blotches).sample(p, &FRAME, none);
            let combined = recipe.sample(p, &FRAME, none);
            assert!(combined <= alone + 1e-6, "multiplying never adds");
            if combined < alone - 0.01 {
                cut += 1;
            } else {
                equal += 1;
            }
        }
        assert!(cut > 20 && equal > 20, "{cut} cut, {equal} untouched");
    }

    #[test]
    fn steps_combine_in_order_and_by_their_amount() {
        let flat = |value: f32| Generator {
            pattern: Pattern::Gradient,
            axis: Axis::Y,
            // A gradient squeezed by its levels to 1 wherever it is above `value`.
            low: value,
            high: value,
            ..Generator::default()
        };
        let at = Vec3::new(0.0, 4.0, 0.0); // the top: the gradient is 1, so above 0.5
        let near = [f32::INFINITY; 2];
        let make = |op, amount| {
            Recipe::with_steps(
                flat(0.5),
                [Step {
                    op,
                    amount,
                    generator: Generator {
                        invert: true,
                        ..flat(0.5)
                    },
                }],
            )
        };
        // Base 1, step 0 (the inverted gradient).
        assert_eq!(make(Op::Multiply, 1.0).sample(at, &FRAME, near), 0.0);
        assert_eq!(make(Op::Add, 1.0).sample(at, &FRAME, near), 1.0);
        assert_eq!(make(Op::Subtract, 1.0).sample(at, &FRAME, near), 1.0);
        assert_eq!(make(Op::Darker, 1.0).sample(at, &FRAME, near), 0.0);
        assert_eq!(make(Op::Lighter, 1.0).sample(at, &FRAME, near), 1.0);
        // Half the amount: halfway between the base and the combined result.
        assert_eq!(make(Op::Multiply, 0.5).sample(at, &FRAME, near), 0.5);
        assert_eq!(make(Op::Multiply, 0.0).sample(at, &FRAME, near), 1.0);
    }

    #[test]
    fn steps_can_be_added_removed_and_are_limited() {
        let mut recipe = Recipe::default();
        for _ in 0..MAX_STEPS {
            assert!(recipe.add_step());
        }
        assert!(!recipe.add_step(), "full");
        let seeds: Vec<u32> = recipe.steps().map(|s| s.generator.seed).collect();
        assert!(seeds.windows(2).all(|pair| pair[0] != pair[1]), "{seeds:?}");
        recipe.remove_step(0);
        let left: Vec<u32> = recipe.steps().map(|s| s.generator.seed).collect();
        assert_eq!(left, seeds[1..]);
        assert!(recipe.add_step());
        assert_eq!(recipe.step_count(), MAX_STEPS);
        recipe.remove_step(MAX_STEPS + 5); // nothing there
        assert_eq!(Recipe::from(Generator::default()).step_count(), 0);
        assert!(recipe.step_mut(MAX_STEPS).is_none());
        assert!(!Recipe::default().uses(edges::Fold::Concave));
    }

    #[test]
    fn op_numbers_round_trip() {
        for op in Op::ALL {
            assert_eq!(Op::from_index(op.index()), Some(op));
        }
        assert_eq!(Op::from_index(99), None);
    }
}
