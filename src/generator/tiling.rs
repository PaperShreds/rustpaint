//! Repeating shapes: checkers, stripes, bricks and tiles. Like the noise they
//! are built in 3D, so they wrap around the model without seams. Each is a
//! lattice of planes; where the surface cuts a plane it shows a line.
//!
//! Distances to a plane are measured along the surface, not straight through
//! space: a surface that meets a plane at a shallow angle would otherwise
//! show a smeared, extra-wide line, and one lying in the plane would be all
//! line. `normal` is the surface's direction at the point.

use glam::Vec3;

/// How far along the surface a point is from a plane `distance` away in
/// space, where the plane is perpendicular to an axis and `along` is the
/// normal's component on that axis. A surface lying in the plane is never
/// near it (it never crosses it).
fn on_surface(distance: f32, along: f32) -> f32 {
    let slope = (1.0 - along * along).max(0.0).sqrt();
    if slope < 1e-3 {
        f32::INFINITY
    } else {
        distance / slope
    }
}

/// 0 at `edge`, 1 at `edge + soft` and beyond; a hard step when `soft` is 0.
fn ramp(distance: f32, edge: f32, soft: f32) -> f32 {
    if soft <= 1e-6 {
        return if distance >= edge { 1.0 } else { 0.0 };
    }
    let t = ((distance - edge) / soft).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Distance from `t` to the nearest whole number.
fn to_whole(t: f32) -> f32 {
    let f = t - t.floor();
    f.min(1.0 - f)
}

/// A 3D checkerboard of unit cubes: 1 and 0 alternating. `bevel` softens
/// the change over that distance either side of each border.
pub fn checker(point: Vec3, normal: Vec3, bevel: f32) -> f32 {
    let product: f32 = (0..3)
        .map(|axis| {
            let t = point[axis];
            let sign = if (t.floor() as i64).rem_euclid(2) == 0 {
                1.0
            } else {
                -1.0
            };
            sign * ramp(on_surface(to_whole(t), normal[axis]), 0.0, bevel)
        })
        .product();
    0.5 + 0.5 * product
}

/// Lines across `axis`, one per unit: 1 on a line `width` wide (a fraction
/// of the spacing), 0 between.
pub fn stripes(point: Vec3, normal: Vec3, axis: usize, width: f32, bevel: f32) -> f32 {
    let distance = on_surface(to_whole(point[axis]), normal[axis]);
    1.0 - ramp(distance, width * 0.5, bevel)
}

/// Rows of blocks stacked along `axis`, one unit high and `length` long in
/// the other two directions. Each row is pushed `offset` of a block along
/// from the one below (0.5 for a brick bond). 1 on a block's face, 0 in the
/// `mortar` (as a fraction of a block's height) between blocks, with a
/// `bevel` from one to the other.
pub fn blocks(
    point: Vec3,
    normal: Vec3,
    axis: usize,
    length: f32,
    offset: f32,
    mortar: f32,
    bevel: f32,
) -> f32 {
    let up = point[axis];
    let shift = up.floor() * offset * length;
    let edge = (0..3)
        .map(|k| {
            let distance = if k == axis {
                to_whole(up)
            } else {
                to_whole((point[k] + shift) / length) * length
            };
            on_surface(distance, normal[k])
        })
        .fold(f32::INFINITY, f32::min);
    ramp(edge, mortar * 0.5, bevel)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FACING_Z: Vec3 = Vec3::Z;

    #[test]
    fn checker_alternates_across_a_face() {
        let at = |x: f32, y: f32| checker(Vec3::new(x, y, 0.3), FACING_Z, 0.0);
        assert_eq!(at(0.5, 0.5), at(1.5, 1.5));
        assert_ne!(at(0.5, 0.5), at(1.5, 0.5));
        assert_ne!(at(0.5, 0.5), at(0.5, -0.5));
        // A face lying exactly on a border still shows a checkerboard.
        let on_border = |x: f32, y: f32| checker(Vec3::new(x, y, 1.0), FACING_Z, 0.1);
        assert_ne!(on_border(0.5, 0.5), on_border(1.5, 0.5));
    }

    #[test]
    fn a_bevel_makes_a_soft_change() {
        let hard = checker(Vec3::new(1.02, 0.5, 0.5), FACING_Z, 0.0);
        let soft = checker(Vec3::new(1.02, 0.5, 0.5), FACING_Z, 0.2);
        assert!(hard == 0.0 || hard == 1.0);
        assert!(soft > 0.0 && soft < 1.0, "{soft}");
    }

    #[test]
    fn stripes_are_as_wide_as_asked() {
        let lit = (0..1000)
            .filter(|i| {
                stripes(
                    Vec3::new(0.3, *i as f32 / 100.0, 0.0),
                    FACING_Z,
                    1,
                    0.2,
                    0.0,
                ) > 0.5
            })
            .count();
        assert!((190..=210).contains(&lit), "{lit}");
    }

    #[test]
    fn lines_keep_their_width_on_a_slanted_surface() {
        // A surface tilted 60 degrees from the stripes: in space, a point
        // along it moves up only half as fast.
        let normal = Vec3::new(0.0, 60f32.to_radians().sin(), 60f32.to_radians().cos());
        let along = Vec3::new(0.0, normal.z, -normal.y);
        let lit = (0..1000)
            .filter(|i| stripes(along * (*i as f32 / 100.0), normal, 1, 0.2, 0.0) > 0.5)
            .count();
        // 10 units along it climb 5, so cross 5 lines, each still 0.2 wide
        // along the surface (straight through space they would be 0.4).
        assert!((95..=105).contains(&lit), "{lit}");
    }

    #[test]
    fn bricks_have_mortar_between_them_and_rows_are_offset() {
        let at = |x: f32, y: f32| blocks(Vec3::new(x, y, 0.3), FACING_Z, 1, 2.0, 0.5, 0.1, 0.0);
        assert_eq!(at(1.0, 0.5), 1.0, "middle of a brick");
        assert_eq!(at(1.0, 0.01), 0.0, "mortar between rows");
        assert_eq!(at(0.01, 0.5), 0.0, "mortar between bricks");
        // In the next row up the joint has moved half a brick along.
        assert_eq!(at(0.01, 1.5), 1.0);
        assert_eq!(at(1.01, 1.5), 0.0);
        // A wall in the plane of a joint is still bricks, not all mortar.
        let in_joint = blocks(Vec3::new(1.0, 0.5, 2.0), FACING_Z, 1, 2.0, 0.5, 0.1, 0.0);
        assert_eq!(in_joint, 1.0);
    }
}
