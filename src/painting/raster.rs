//! Triangle and texel geometry shared by the brush and the generators.

use glam::{Vec2, Vec3};

/// Texels whose centre lies this many texels outside a triangle's UV footprint
/// are still painted, so UV seams don't show a hairline gap.
pub(crate) const SEAM_MARGIN_TEXELS: f32 = 1.0;

/// How far below zero each barycentric weight may go for a point to stay
/// within `margin` (same units as `corners`) of the triangle.
pub(crate) fn edge_margins(corners: [Vec2; 3], margin: f32) -> Option<[f32; 3]> {
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

pub(crate) fn uv_bounds(points: &[Vec2]) -> (Vec2, Vec2) {
    points.iter().fold(
        (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
        |(lo, hi), point| (lo.min(*point), hi.max(*point)),
    )
}

pub(crate) fn texel_range(min: f32, max: f32, size: u32) -> (i64, i64) {
    let min = min.floor() as i64;
    let max = max.ceil() as i64;
    if max - min + 1 > size as i64 {
        let center = (min + max) / 2;
        (center - size as i64 / 2, center + size as i64 / 2)
    } else {
        (min, max)
    }
}

pub(crate) fn barycentric_2d(point: Vec2, triangle: [Vec2; 3]) -> Option<[f32; 3]> {
    let denominator = cross(triangle[1] - triangle[0], triangle[2] - triangle[0]);
    if denominator.abs() < 1e-12 {
        return None;
    }
    let b = cross(point - triangle[0], triangle[2] - triangle[0]) / denominator;
    let c = cross(triangle[1] - triangle[0], point - triangle[0]) / denominator;
    Some([1.0 - b - c, b, c])
}

/// Barycentric weights of `point` projected onto the triangle's plane.
pub(crate) fn barycentric_3d(point: Vec3, triangle: [Vec3; 3]) -> Option<[f32; 3]> {
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

pub(crate) fn interpolate<T>(values: [T; 3], weights: [f32; 3]) -> T
where
    T: std::ops::Mul<f32, Output = T> + std::ops::Add<Output = T> + Copy,
{
    values[0] * weights[0] + values[1] * weights[1] + values[2] * weights[2]
}

pub(crate) fn cross(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - a.y * b.x
}
