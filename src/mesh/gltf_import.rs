use super::{linear_to_srgb, srgb_to_linear, Image, Material, Mesh, SubMesh, Vertex};
use glam::{Mat3, Mat4, Vec3};
use std::path::Path;

const NO_MATERIAL_NAME: &str = "(no material)";

/// Loads every triangle mesh in the default scene. Each glTF material becomes
/// a `Material`; each primitive becomes a `SubMesh` that uses one of them.
pub fn load(path: &Path) -> Result<Mesh, String> {
    let (document, buffers, images) = gltf::import(path).map_err(|e| e.to_string())?;

    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .ok_or("file contains no scenes")?;

    let mut mesh = Mesh {
        vertices: Vec::new(),
        indices: Vec::new(),
        submeshes: Vec::new(),
        materials: document
            .materials()
            .map(|m| convert_material(&m, &images))
            .collect(),
    };

    for node in scene.nodes() {
        append_node(&node, Mat4::IDENTITY, &buffers, &mut mesh);
    }

    if mesh.vertices.is_empty() {
        return Err("no triangle meshes found in file".into());
    }
    Ok(mesh)
}

fn convert_material(material: &gltf::Material, images: &[gltf::image::Data]) -> Material {
    let pbr = material.pbr_metallic_roughness();
    let factor = pbr.base_color_factor();

    let name = material
        .name()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("Material {}", material.index().unwrap_or(0)));

    let texture_image = pbr
        .base_color_texture()
        .and_then(|info| images.get(info.texture().source().index()))
        .and_then(|data| {
            Some(Image {
                width: data.width,
                height: data.height,
                rgba: to_rgba8(data)?,
            })
        });

    let base_color = match texture_image {
        Some(mut image) => {
            if factor != [1.0; 4] {
                tint(&mut image, factor);
            }
            image
        }
        None => Image::solid(factor),
    };

    Material { name, base_color }
}

/// Converts decoded glTF image data to 8-bit RGBA. Returns None for exotic formats.
fn to_rgba8(data: &gltf::image::Data) -> Option<Vec<u8>> {
    use gltf::image::Format;

    let src = &data.pixels;
    let mut out = Vec::with_capacity((data.width * data.height * 4) as usize);
    match data.format {
        Format::R8G8B8A8 => out.extend_from_slice(src),
        Format::R8G8B8 => {
            for c in src.chunks_exact(3) {
                out.extend_from_slice(&[c[0], c[1], c[2], 255]);
            }
        }
        Format::R8G8 => {
            for c in src.chunks_exact(2) {
                out.extend_from_slice(&[c[0], c[0], c[0], c[1]]);
            }
        }
        Format::R8 => {
            for &v in src {
                out.extend_from_slice(&[v, v, v, 255]);
            }
        }
        // 16-bit: keep the high byte of each little-endian channel.
        Format::R16G16B16A16 => {
            for c in src.chunks_exact(8) {
                out.extend_from_slice(&[c[1], c[3], c[5], c[7]]);
            }
        }
        Format::R16G16B16 => {
            for c in src.chunks_exact(6) {
                out.extend_from_slice(&[c[1], c[3], c[5], 255]);
            }
        }
        _ => return None,
    }
    Some(out)
}

/// Multiplies the image by glTF's baseColorFactor (done in linear space).
fn tint(image: &mut Image, factor: [f32; 4]) {
    for px in image.rgba.chunks_exact_mut(4) {
        for c in 0..3 {
            px[c] = linear_to_srgb(srgb_to_linear(px[c]) * factor[c]);
        }
        px[3] = (px[3] as f32 * factor[3]).round().clamp(0.0, 255.0) as u8;
    }
}

/// Index of the fallback material for primitives that don't specify one.
fn no_material_index(mesh: &mut Mesh) -> usize {
    if let Some(i) = mesh.materials.iter().position(|m| m.name == NO_MATERIAL_NAME) {
        return i;
    }
    mesh.materials.push(Material {
        name: NO_MATERIAL_NAME.to_string(),
        base_color: Image::solid([0.8, 0.8, 0.8, 1.0]),
    });
    mesh.materials.len() - 1
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
        return;
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

    let material = match primitive.material().index() {
        Some(index) => index,
        None => no_material_index(out),
    };

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

    let start = out.indices.len() as u32;
    for tri in indices.chunks_exact(3) {
        if flip_winding {
            out.indices
                .extend_from_slice(&[base + tri[0], base + tri[2], base + tri[1]]);
        } else {
            out.indices
                .extend_from_slice(&[base + tri[0], base + tri[1], base + tri[2]]);
        }
    }
    let end = out.indices.len() as u32;

    if end > start {
        out.submeshes.push(SubMesh {
            indices: start..end,
            material,
        });
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
        let face_normal = (pb - pa).cross(pc - pa);
        sums[a] += face_normal;
        sums[b] += face_normal;
        sums[c] += face_normal;
    }
    sums.into_iter()
        .map(|n| n.normalize_or_zero().to_array())
        .collect()
}