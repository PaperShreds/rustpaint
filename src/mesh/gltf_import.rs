use super::{Mesh, Vertex};
use glam::{Mat3, Mat4, Vec3};
use std::path::Path;

/// Loads every triangle mesh in the default scene into a single `Mesh`.
pub fn load(path: &Path) -> Result<Mesh, String> {
    let (document, buffers, _images) = gltf::import(path).map_err(|e| e.to_string())?;

    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or("file contains no scenes")?;

    let mut mesh = Mesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for node in scene.nodes() {
        append_node(&node, Mat4::IDENTITY, &buffers, &mut mesh);
    }

    if mesh.vertices.is_empty() {
        return Err("no triangle meshes found in file".into());
    }
    Ok(mesh)
}

fn append_node(
    node: &gltf::Node,
    parent_transform: Mat4,
    buffers: &[gltf::buffer::Data],
    out: &mut Mesh,
) {
    let world = parent_transform * Mat4::from_cols_array_2d(&node.transform().matrix());

    if let Some(gltf_mesh) = node.mesh() {
        for primitive in gltf_mesh.primitives() {
            append_primitive(&primitive, world, buffers, out);
        }
    }
    for child in node.children() {
        append_node(&child, world, buffers, out);
    }
}

fn append_primitive(
    primitive: &gltf::Primitive,
    world: Mat4,
    buffers: &[gltf::buffer::Data],
    out: &mut Mesh,
) {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        return; // points, lines and strips are skipped for now
    }

    let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
    let Some(positions) = reader.read_positions() else {
        return;
    };
    let positions: Vec<[f32; 3]> = positions.collect();

    let indices: Vec<u32> = match reader.read_indices() {
        Some(indices) => indices.into_u32().collect(),
        None => (0..positions.len() as u32).collect(),
    };
    let normals: Vec<[f32; 3]> = match reader.read_normals() {
        Some(normals) => normals.collect(),
        None => compute_normals(&positions, &indices),
    };
    let uvs: Option<Vec<[f32; 2]>> = reader
        .read_tex_coords(0)
        .map(|coords| coords.into_f32().collect());

    // Normals need the inverse-transpose so non-uniform scale doesn't skew them.
    let linear = Mat3::from_cols(
        world.x_axis.truncate(),
        world.y_axis.truncate(),
        world.z_axis.truncate(),
    );
    let normal_matrix = linear.inverse().transpose();
    // A mirrored transform flips triangle winding, which would get culled.
    let flip_winding = world.determinant() < 0.0;

    let base = out.vertices.len() as u32;
    for (i, position) in positions.iter().enumerate() {
        let position = world.transform_point3(Vec3::from(*position));
        let normal = (normal_matrix * Vec3::from(normals[i])).normalize_or_zero();
        let uv = uvs
            .as_ref()
            .and_then(|uvs| uvs.get(i))
            .copied()
            .unwrap_or([0.0, 0.0]);
        out.vertices.push(Vertex {
            position: position.to_array(),
            normal: normal.to_array(),
            uv,
        });
    }

    for tri in indices.chunks_exact(3) {
        if flip_winding {
            out.indices
                .extend_from_slice(&[base + tri[0], base + tri[2], base + tri[1]]);
        } else {
            out.indices
                .extend_from_slice(&[base + tri[0], base + tri[1], base + tri[2]]);
        }
    }
}

/// Smooth per-vertex normals for files that don't ship any.
fn compute_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut sums = vec![Vec3::ZERO; positions.len()];
    for tri in indices.chunks_exact(3) {
        let (a, b, c) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let (pa, pb, pc) = (
            Vec3::from(positions[a]),
            Vec3::from(positions[b]),
            Vec3::from(positions[c]),
        );
        let face_normal = (pb - pa).cross(pc - pa); // length = area, so bigger faces weigh more
        sums[a] += face_normal;
        sums[b] += face_normal;
        sums[c] += face_normal;
    }
    sums.into_iter()
        .map(|n| n.normalize_or_zero().to_array())
        .collect()
}
