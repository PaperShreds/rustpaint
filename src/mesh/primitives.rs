use super::{Mesh, Vertex};
use glam::Vec3;

/// A unit cube with 24 vertices (4 per face, so each face gets its own normals and UVs).
pub fn cube() -> Mesh {
    // (normal, u axis, v axis). u × v == normal gives counter-clockwise winding.
    let faces = [
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y),
        (Vec3::X, Vec3::NEG_Z, Vec3::Y),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::X, Vec3::NEG_Z),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
    ];
    // (u sign, v sign, uv coordinate); image-style UVs have v pointing down.
    let corners = [
        (-1.0_f32, -1.0_f32, [0.0_f32, 1.0_f32]),
        (1.0, -1.0, [1.0, 1.0]),
        (1.0, 1.0, [1.0, 0.0]),
        (-1.0, 1.0, [0.0, 0.0]),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (normal, u, v) in faces {
        let base = vertices.len() as u32;
        for (su, sv, uv) in corners {
            let position = (normal + su * u + sv * v) * 0.5;
            vertices.push(Vertex {
                position: position.to_array(),
                normal: normal.to_array(),
                uv,
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    Mesh { vertices, indices }
}
