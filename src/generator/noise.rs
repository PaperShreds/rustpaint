//! Seeded 3D noise. Every function is deterministic: the same point and seed
//! always give the same value, on every machine.

use glam::Vec3;

/// A pseudo-random number from an integer lattice point and a seed.
fn hash(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut h = seed
        ^ (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^ (h >> 15)
}

/// `hash` as a number in 0..1.
fn unit(x: i32, y: i32, z: i32, seed: u32) -> f32 {
    (hash(x, y, z, seed) >> 8) as f32 / (1u32 << 24) as f32
}

/// Smooth 0..1 easing with zero slope at both ends.
fn fade(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// The lattice cell holding `point`, and the position inside it (0..1).
fn cell(point: Vec3) -> ([i32; 3], Vec3) {
    let floor = point.floor();
    (
        [floor.x as i32, floor.y as i32, floor.z as i32],
        point - floor,
    )
}

/// Value noise: random values on a lattice, smoothly blended. Range 0..1.
pub fn value(point: Vec3, seed: u32) -> f32 {
    let ([x, y, z], t) = cell(point);
    let (u, v, w) = (fade(t.x), fade(t.y), fade(t.z));
    let corner = |dx, dy, dz| unit(x + dx, y + dy, z + dz, seed);
    let along_x = |dy, dz| lerp(corner(0, dy, dz), corner(1, dy, dz), u);
    lerp(
        lerp(along_x(0, 0), along_x(1, 0), v),
        lerp(along_x(0, 1), along_x(1, 1), v),
        w,
    )
}

/// Gradient (Perlin) noise. Range 0..1, centred on 0.5.
pub fn perlin(point: Vec3, seed: u32) -> f32 {
    const GRADIENTS: [[f32; 3]; 12] = [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, -1.0],
    ];
    let ([x, y, z], t) = cell(point);
    let (u, v, w) = (fade(t.x), fade(t.y), fade(t.z));
    let corner = |dx: i32, dy: i32, dz: i32| {
        let gradient = GRADIENTS[hash(x + dx, y + dy, z + dz, seed) as usize % 12];
        let offset = t - Vec3::new(dx as f32, dy as f32, dz as f32);
        gradient[0] * offset.x + gradient[1] * offset.y + gradient[2] * offset.z
    };
    let along_x = |dy, dz| lerp(corner(0, dy, dz), corner(1, dy, dz), u);
    let noise = lerp(
        lerp(along_x(0, 0), along_x(1, 0), v),
        lerp(along_x(0, 1), along_x(1, 1), v),
        w,
    );
    (0.5 + noise * 0.5).clamp(0.0, 1.0)
}

/// Worley noise: the distance to the nearest of a random point per lattice
/// cell. 0 at the points, rising to about 1 between them.
pub fn cells(point: Vec3, seed: u32) -> f32 {
    let ([x, y, z], t) = cell(point);
    let mut nearest = f32::MAX;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (cx, cy, cz) = (x + dx, y + dy, z + dz);
                let feature = Vec3::new(
                    dx as f32 + unit(cx, cy, cz, seed),
                    dy as f32 + unit(cx, cy, cz, seed ^ 0x68bc_21eb),
                    dz as f32 + unit(cx, cy, cz, seed ^ 0x02e9_9a4d),
                );
                nearest = nearest.min((feature - t).length_squared());
            }
        }
    }
    nearest.sqrt().min(1.0)
}

/// Layers `octaves` of `base`, each twice as fine and half as strong as the
/// one before. Stays in 0..1.
pub fn fractal(base: fn(Vec3, u32) -> f32, point: Vec3, octaves: u32, seed: u32) -> f32 {
    let mut sum = 0.0;
    let mut total = 0.0;
    let mut strength = 1.0;
    let mut frequency = 1.0;
    for octave in 0..octaves.max(1) {
        sum += strength * base(point * frequency, seed.wrapping_add(octave * 0x9e37));
        total += strength;
        strength *= 0.5;
        frequency *= 2.0;
    }
    sum / total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn points() -> impl Iterator<Item = Vec3> {
        (0..500).map(|i| {
            let i = i as f32;
            Vec3::new(i * 0.37, i * 0.113 - 9.0, (i * 0.71).sin() * 20.0)
        })
    }

    #[test]
    fn noise_stays_in_range_and_varies() {
        for base in [value, perlin, cells] {
            let samples: Vec<f32> = points().map(|p| base(p, 7)).collect();
            assert!(samples.iter().all(|v| (0.0..=1.0).contains(v)));
            let lo = samples.iter().copied().fold(1.0, f32::min);
            let hi = samples.iter().copied().fold(0.0, f32::max);
            assert!(hi - lo > 0.4, "flat noise: {lo}..{hi}");
        }
    }

    #[test]
    fn the_same_seed_gives_the_same_noise_and_another_seed_a_different_one() {
        for base in [value, perlin, cells] {
            let a: Vec<f32> = points().map(|p| base(p, 1)).collect();
            let again: Vec<f32> = points().map(|p| base(p, 1)).collect();
            let other: Vec<f32> = points().map(|p| base(p, 2)).collect();
            assert_eq!(a, again);
            assert_ne!(a, other);
        }
    }

    #[test]
    fn noise_is_continuous() {
        for base in [value, perlin, cells] {
            for p in points() {
                let step = Vec3::splat(0.001);
                assert!((base(p, 3) - base(p + step, 3)).abs() < 0.05);
            }
        }
    }

    #[test]
    fn fractal_noise_stays_in_range() {
        for p in points() {
            let v = fractal(perlin, p, 6, 5);
            assert!((0.0..=1.0).contains(&v));
        }
    }
}
