use super::{Image, Material, Mesh, SubMesh, Vertex};
use glam::Vec3;

/// Pixels per cube face in the texture atlas.
const FACE_SIZE: u32 = 256;
/// Empty border around each face's UV cell, so filtering doesn't bleed between faces.
const FACE_PADDING: f32 = 4.0 / FACE_SIZE as f32;

/// A unit cube with 24 vertices and one checkerboard material.
/// Each face has its own cell in a 3x2 texture atlas, so faces can be painted independently.
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
    let corners = [
        (-1.0_f32, -1.0_f32, [0.0_f32, 1.0_f32]),
        (1.0, -1.0, [1.0, 1.0]),
        (1.0, 1.0, [1.0, 0.0]),
        (-1.0, 1.0, [0.0, 0.0]),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (face, (normal, u, v)) in faces.into_iter().enumerate() {
        let base = vertices.len() as u32;
        let cell = [(face % 3) as f32, (face / 3) as f32];
        for (su, sv, uv) in corners {
            let position = (normal + su * u + sv * v) * 0.5;
            let inset = uv.map(|c| FACE_PADDING + c * (1.0 - 2.0 * FACE_PADDING));
            vertices.push(Vertex {
                position: position.to_array(),
                normal: normal.to_array(),
                uv: [(cell[0] + inset[0]) / 3.0, (cell[1] + inset[1]) / 2.0],
            });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    let index_count = indices.len() as u32;
    Mesh {
        vertices,
        indices,
        submeshes: vec![SubMesh {
            indices: 0..index_count,
            material: 0,
        }],
        materials: vec![Material {
            name: "Checker".to_string(),
            base_color: checker_image(FACE_SIZE * 3, FACE_SIZE * 2, FACE_SIZE / 8),
        }],
    }
}

fn checker_image(width: u32, height: u32, cell: u32) -> Image {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        for x in 0..width {
            let light = (x / cell + y / cell).is_multiple_of(2);
            let v = if light { 217 } else { 115 };
            rgba.extend_from_slice(&[v, v, v, 255]);
        }
    }
    Image {
        width,
        height,
        rgba,
    }
}
#[cfg(test)]
mod tests {
    use super::cube;

    #[test]
    fn cube_faces_have_separate_uv_cells() {
        let mesh = cube();
        let cells: Vec<_> = mesh
            .vertices
            .chunks_exact(4)
            .map(|face| {
                let u = face.iter().map(|v| v.uv[0]).sum::<f32>() / 4.0;
                let v = face.iter().map(|v| v.uv[1]).sum::<f32>() / 4.0;
                ((u * 3.0) as u32, (v * 2.0) as u32)
            })
            .collect();
        for (i, a) in cells.iter().enumerate() {
            assert!(
                cells[i + 1..].iter().all(|b| a != b),
                "faces share a UV cell"
            );
        }
        for vertex in &mesh.vertices {
            assert!(vertex.uv.iter().all(|c| (0.0..=1.0).contains(c)));
        }
    }
}
