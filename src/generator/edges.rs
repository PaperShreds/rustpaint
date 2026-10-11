//! Where the model's edges and creases are, for the edge wear and cavity
//! patterns.
//!
//! An edge is convex where the two faces that meet at it fold away from each
//! other, like the corner of a box, and concave (a crease) where they fold
//! towards each other, like the inside of one. Sharper folds count for more.
//! Flat surfaces have neither.

use super::Frame;
use crate::mesh::Mesh;
use glam::Vec3;
use std::collections::HashMap;

/// Edges that fold less than this (in degrees) are tessellation, not edges.
const MIN_ANGLE: f32 = 10.0;
/// Edges that fold this much or more count fully.
const FULL_ANGLE: f32 = 60.0;
/// The most a pattern can reach from an edge, in model radii. Sets the grid.
pub const MAX_REACH: f32 = 0.1;

/// Which kind of fold to look for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold {
    Convex,
    Concave,
}

struct Segment {
    a: Vec3,
    b: Vec3,
    /// 0..1: how sharp the fold is.
    weight: f32,
}

/// The model's folds, to find the ones near a point quickly: a grid for each
/// kind.
pub struct Edges {
    convex: Grid,
    concave: Grid,
}

struct Grid {
    segments: Vec<Segment>,
    cell: f32,
    cells: HashMap<[i32; 3], Vec<u32>>,
}

/// An edge by its end positions (so edges split by UV seams or normals meet).
type EdgeKey = ([u32; 3], [u32; 3]);

/// A triangle's side of an edge: its normal and the corner opposite the edge.
struct Side {
    normal: Vec3,
    opposite: Vec3,
}

fn key(position: Vec3) -> [u32; 3] {
    position.to_array().map(f32::to_bits)
}

impl Edges {
    pub fn new(mesh: &Mesh, frame: &Frame) -> Self {
        let mut open: HashMap<EdgeKey, (Vec3, Vec3, Side)> = HashMap::new();
        let (mut convex, mut concave) = (Vec::new(), Vec::new());
        for submesh in &mesh.submeshes {
            let range = submesh.indices.start as usize..submesh.indices.end as usize;
            let Some(indices) = mesh.indices.get(range) else {
                continue;
            };
            for corners in indices.chunks_exact(3) {
                let Some(position) = corners
                    .iter()
                    .map(|&i| {
                        mesh.vertices
                            .get(i as usize)
                            .map(|v| Vec3::from(v.position))
                    })
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                let normal = (position[1] - position[0])
                    .cross(position[2] - position[0])
                    .normalize_or_zero();
                if normal == Vec3::ZERO || !normal.is_finite() {
                    continue;
                }
                for side in 0..3 {
                    let (a, b) = (position[side], position[(side + 1) % 3]);
                    let this = Side {
                        normal,
                        opposite: position[(side + 2) % 3],
                    };
                    let edge = if key(a) < key(b) {
                        (key(a), key(b))
                    } else {
                        (key(b), key(a))
                    };
                    match open.remove(&edge) {
                        None => {
                            open.insert(edge, (a, b, this));
                        }
                        Some((a, b, other)) => match fold_weight(a, &other, &this) {
                            Some((Fold::Convex, weight)) => convex.push(Segment { a, b, weight }),
                            Some((Fold::Concave, weight)) => concave.push(Segment { a, b, weight }),
                            None => {}
                        },
                    }
                }
            }
        }
        let cell = (MAX_REACH * frame.radius()).max(1e-6);
        Self {
            convex: Grid::new(convex, cell),
            concave: Grid::new(concave, cell),
        }
    }

    /// Distance from `point` to the nearest fold of this kind, with weaker
    /// folds counting as further away (a fold of weight 0.5 counts double).
    /// `INFINITY` when none is within a reach (`MAX_REACH` model radii) of it.
    pub fn distance(&self, point: Vec3, fold: Fold) -> f32 {
        match fold {
            Fold::Convex => self.convex.distance(point),
            Fold::Concave => self.concave.distance(point),
        }
    }

    #[cfg(test)]
    pub fn len(&self, fold: Fold) -> usize {
        match fold {
            Fold::Convex => self.convex.segments.len(),
            Fold::Concave => self.concave.segments.len(),
        }
    }
}

impl Grid {
    fn new(segments: Vec<Segment>, cell: f32) -> Self {
        let mut cells: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
        for (index, segment) in segments.iter().enumerate() {
            let cell_of = |p: Vec3| (p / cell).floor().to_array().map(|c| c as i32);
            let (lo, hi) = (
                cell_of(segment.a.min(segment.b)),
                cell_of(segment.a.max(segment.b)),
            );
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        cells.entry([x, y, z]).or_default().push(index as u32);
                    }
                }
            }
        }
        Self {
            segments,
            cell,
            cells,
        }
    }

    fn distance(&self, point: Vec3) -> f32 {
        let [cx, cy, cz] = (point / self.cell).floor().to_array().map(|c| c as i32);
        let mut nearest = f32::INFINITY;
        for x in cx - 1..=cx + 1 {
            for y in cy - 1..=cy + 1 {
                for z in cz - 1..=cz + 1 {
                    let Some(found) = self.cells.get(&[x, y, z]) else {
                        continue;
                    };
                    for &index in found {
                        let segment = &self.segments[index as usize];
                        let distance = distance_to_segment(point, segment.a, segment.b);
                        nearest = nearest.min(distance / segment.weight);
                    }
                }
            }
        }
        nearest
    }
}

/// How the edge from `a` between two faces folds, and how much it counts, if
/// it folds enough to count.
fn fold_weight(a: Vec3, first: &Side, second: &Side) -> Option<(Fold, f32)> {
    // Convex: the second face's far corner lies behind the first face's
    // plane. Concave: in front of it.
    let side = (second.opposite - a).dot(first.normal);
    let fold = if side < -1e-6 {
        Fold::Convex
    } else if side > 1e-6 {
        Fold::Concave
    } else {
        return None;
    };
    let angle = first
        .normal
        .dot(second.normal)
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    (angle >= MIN_ANGLE).then(|| (fold, (angle / FULL_ANGLE).clamp(0.0, 1.0)))
}

fn distance_to_segment(point: Vec3, a: Vec3, b: Vec3) -> f32 {
    let edge = b - a;
    let length_squared = edge.length_squared();
    let t = if length_squared > 0.0 {
        ((point - a).dot(edge) / length_squared).clamp(0.0, 1.0)
    } else {
        0.0
    };
    point.distance(a + edge * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::primitives;

    fn edges(mesh: &Mesh) -> Edges {
        Edges::new(mesh, &Frame::of(mesh))
    }

    fn inside_out(mut mesh: Mesh) -> Mesh {
        for triangle in mesh.indices.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
        mesh
    }

    #[test]
    fn a_cube_has_twelve_convex_edges_and_no_creases() {
        let cube = primitives::cube();
        let edges = edges(&cube);
        // Each edge is two triangle sides, and the faces' own diagonals are flat.
        assert_eq!(edges.len(Fold::Convex), 12);
        assert_eq!(edges.len(Fold::Concave), 0);
        // On an edge, and halfway to the middle of a face (the cube spans -0.5..0.5).
        assert!(edges.distance(Vec3::new(0.5, 0.5, 0.1), Fold::Convex) < 1e-5);
        let near = edges.distance(Vec3::new(0.5, 0.45, 0.0), Fold::Convex);
        assert!((near - 0.05).abs() < 1e-4, "{near}");
        // A face's centre is 0.5 from its nearest edge, which is out of reach
        // for a model this size (reach is a tenth of its radius).
        assert_eq!(
            edges.distance(Vec3::new(0.0, 0.0, 0.5), Fold::Convex),
            f32::INFINITY
        );
    }

    #[test]
    fn an_inside_out_box_has_only_concave_edges() {
        let edges = edges(&inside_out(primitives::cube()));
        assert_eq!(edges.len(Fold::Convex), 0);
        assert_eq!(edges.len(Fold::Concave), 12);
        let corner = Vec3::new(0.5, 0.5, 0.1);
        assert_eq!(edges.distance(corner, Fold::Convex), f32::INFINITY);
        assert!(edges.distance(corner, Fold::Concave) < 1e-5);
    }

    #[test]
    fn weaker_folds_count_as_further_away() {
        let first = Side {
            normal: Vec3::Z,
            opposite: Vec3::new(0.0, -1.0, 0.0),
        };
        let fold = |degrees: f32, up: bool| {
            let angle = degrees.to_radians();
            // The other face tilts away from the first, down (convex) or up.
            let tilt = if up { -1.0 } else { 1.0 };
            Side {
                normal: Vec3::new(0.0, angle.sin() * tilt, angle.cos()),
                opposite: Vec3::new(0.0, angle.cos(), -angle.sin() * tilt),
            }
        };
        let a = Vec3::ZERO;
        assert_eq!(
            fold_weight(a, &first, &fold(5.0, false)),
            None,
            "too shallow"
        );
        let (kind, half) = fold_weight(a, &first, &fold(30.0, false)).unwrap();
        assert_eq!(kind, Fold::Convex);
        assert!((half - 0.5).abs() < 1e-4, "{half}");
        assert_eq!(
            fold_weight(a, &first, &fold(90.0, false)),
            Some((Fold::Convex, 1.0))
        );
        assert_eq!(
            fold_weight(a, &first, &fold(90.0, true)),
            Some((Fold::Concave, 1.0))
        );
    }
}
