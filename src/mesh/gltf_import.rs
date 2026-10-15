use super::{
    Channel, Image, LEVEL, Material, Mesh, SubMesh, Vertex, data_image, linear_to_srgb,
    srgb_to_linear,
};
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

    // glTF packs roughness into green and metallic into blue, scaled by factors.
    let packed = pbr
        .metallic_roughness_texture()
        .and_then(|info| images.get(info.texture().source().index()))
        .and_then(|data| Some((data.width, data.height, to_rgba8(data)?)));
    let channel = |byte: usize, factor: f32| match &packed {
        Some((width, height, rgba)) => {
            let encoding = Channel::Roughness.encoding();
            let mut image = data_image(*width, *height, 0.0);
            for (pixel, source) in image.rgba.chunks_exact_mut(4).zip(rgba.chunks_exact(4)) {
                pixel[..3].fill(encoding.encode(encoding.decode(source[byte]) * factor));
            }
            image
        }
        None => data_image(1, 1, factor),
    };
    // Sizes may differ from the base colour; `Document` makes them match.
    Material {
        name,
        roughness: channel(1, pbr.roughness_factor()),
        metallic: channel(2, pbr.metallic_factor()),
        height: data_image(1, 1, LEVEL),
        base_color,
    }
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
    if let Some(i) = mesh
        .materials
        .iter()
        .position(|m| m.name == NO_MATERIAL_NAME)
    {
        return i;
    }
    mesh.materials.push(Material::new(
        NO_MATERIAL_NAME.to_string(),
        Image::solid([0.8, 0.8, 0.8, 1.0]),
    ));
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
#[cfg(test)]
mod tests {
    use super::load;
    use crate::mesh::{Channel, Encoding};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn imports_roughness_and_metallic_from_the_packed_texture() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("rustpaint-gltf-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();

        // One triangle with UVs.
        let floats: [f32; 15] = [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // positions
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0, // uvs
        ];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(directory.join("data.bin"), &bytes).unwrap();
        // A 4x4 base colour, and a 2x2 packed texture: roughness in green, metallic in blue.
        image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]))
            .save(directory.join("base.png"))
            .unwrap();
        image::RgbaImage::from_pixel(2, 2, image::Rgba([0, 200, 100, 255]))
            .save(directory.join("packed.png"))
            .unwrap();
        let gltf = r#"{
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "material": 0}]}],
            "materials": [{"name": "Paint", "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicRoughnessTexture": {"index": 1},
                "roughnessFactor": 0.5,
                "metallicFactor": 1.0
            }}],
            "textures": [{"source": 0}, {"source": 1}],
            "images": [{"uri": "base.png"}, {"uri": "packed.png"}],
            "buffers": [{"uri": "data.bin", "byteLength": 60}],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": 36},
                {"buffer": 0, "byteOffset": 36, "byteLength": 24}
            ],
            "accessors": [
                {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
                 "min": [0, 0, 0], "max": [1, 1, 0]},
                {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"}
            ]
        }"#;
        let path = directory.join("model.gltf");
        std::fs::write(&path, gltf).unwrap();

        let mut mesh = load(&path).unwrap();
        std::fs::remove_dir_all(&directory).unwrap();
        let material = &mut mesh.materials[0];
        assert_eq!(material.roughness.width, 2, "imported at its own size");
        material.conform_channels();
        for channel in Channel::ALL {
            let image = material.channel(channel);
            assert_eq!((image.width, image.height), (4, 4), "{}", channel.name());
        }
        let value = |channel: Channel| material.channel(channel).rgba[0];
        assert_eq!(
            value(Channel::Roughness),
            Encoding::Linear.encode(200.0 / 255.0 * 0.5)
        );
        assert_eq!(value(Channel::Metallic), 100);
    }
}
