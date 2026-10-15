//! Writes the model with its painted textures as one binary glTF (`.glb`)
//! file, ready for Blender, Godot, Unity, Unreal or a web viewer.
//!
//! Each material slot becomes a glTF material with a base colour texture and
//! a metallic-roughness texture (roughness in green, metallic in blue, as
//! glTF expects; red, which some engines read as ambient occlusion, is white).
//! A slot whose height channel isn't level everywhere also gets a normal map,
//! and the mesh then carries tangents, so engines read it the way it was made.
//! Every texture is the slot's visible layers combined, embedded as a PNG.

use crate::document::Document;
use crate::files;
use crate::mesh::{Channel, Image};
use serde_json::{Value, json};
use std::path::Path;

// glTF constants.
const ARRAY_BUFFER: u32 = 34962;
const ELEMENT_ARRAY_BUFFER: u32 = 34963;
const FLOAT: u32 = 5126;
const UNSIGNED_INT: u32 = 5125;
const LINEAR: u32 = 9729;
const LINEAR_MIPMAP_LINEAR: u32 = 9987;
const REPEAT: u32 = 10497;

pub fn write(path: &Path, document: &Document, name: &str) -> Result<(), String> {
    files::write_atomically(path, &encode(document, name)?)
}

/// The `.glb` file's contents.
pub fn encode(document: &Document, name: &str) -> Result<Vec<u8>, String> {
    let mesh = &document.mesh;
    let textures = encode_textures(document)?;

    let mut buffer = Buffer::default();
    let mut accessors = Vec::new();

    // Vertex attributes, one tightly packed view each.
    let positions: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
    let (min, max) = bounds(&positions);
    let view = buffer.push(bytemuck::cast_slice(&positions), Some(ARRAY_BUFFER));
    accessors.push(json!({
        "bufferView": view, "componentType": FLOAT, "count": positions.len(),
        "type": "VEC3", "min": min, "max": max,
    }));
    let normals: Vec<[f32; 3]> = mesh
        .vertices
        .iter()
        .map(|v| unit_normal(v.normal))
        .collect();
    let view = buffer.push(bytemuck::cast_slice(&normals), Some(ARRAY_BUFFER));
    accessors.push(json!({
        "bufferView": view, "componentType": FLOAT, "count": normals.len(), "type": "VEC3",
    }));
    let uvs: Vec<[f32; 2]> = mesh.vertices.iter().map(|v| v.uv).collect();
    let view = buffer.push(bytemuck::cast_slice(&uvs), Some(ARRAY_BUFFER));
    accessors.push(json!({
        "bufferView": view, "componentType": FLOAT, "count": uvs.len(), "type": "VEC2",
    }));
    let mut attributes = json!({ "POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2 });
    if textures.iter().any(|t| t.normal.is_some()) {
        let tangents = tangents(mesh);
        let view = buffer.push(bytemuck::cast_slice(&tangents), Some(ARRAY_BUFFER));
        accessors.push(json!({
            "bufferView": view, "componentType": FLOAT, "count": tangents.len(), "type": "VEC4",
        }));
        attributes["TANGENT"] = json!(accessors.len() - 1);
    }

    // One primitive per submesh, all sharing the vertex attributes.
    let indices_view = buffer.push(
        bytemuck::cast_slice(&mesh.indices),
        Some(ELEMENT_ARRAY_BUFFER),
    );
    let mut primitives = Vec::new();
    for submesh in &mesh.submeshes {
        let (start, end) = (submesh.indices.start as usize, submesh.indices.end as usize);
        if start >= end || end > mesh.indices.len() || submesh.material >= mesh.materials.len() {
            continue;
        }
        accessors.push(json!({
            "bufferView": indices_view, "byteOffset": start * 4,
            "componentType": UNSIGNED_INT, "count": end - start, "type": "SCALAR",
        }));
        primitives.push(json!({
            "attributes": attributes,
            "indices": accessors.len() - 1,
            "material": submesh.material,
        }));
    }
    if primitives.is_empty() {
        return Err("the model has no triangles to export".into());
    }

    // Per material: base colour, metallic-roughness and maybe a normal map.
    let mut images = Vec::new();
    let mut materials = Vec::new();
    for (material, textures) in mesh.materials.iter().zip(textures) {
        let mut add_image = |png: Vec<u8>, suffix: &str| {
            let view = buffer.push(&png, None);
            images.push(json!({
                "bufferView": view,
                "mimeType": "image/png",
                "name": format!("{}_{suffix}", material.name),
            }));
            images.len() - 1
        };
        let base_color = add_image(textures.base_color, "BaseColor");
        let metal_rough = add_image(textures.metal_rough, "MetallicRoughness");
        let mut entry = json!({
            "name": material.name,
            "pbrMetallicRoughness": {
                "baseColorTexture": { "index": base_color },
                "metallicRoughnessTexture": { "index": metal_rough },
                "metallicFactor": 1.0,
                "roughnessFactor": 1.0,
            },
        });
        if let Some(normal) = textures.normal {
            entry["normalTexture"] = json!({ "index": add_image(normal, "Normal") });
        }
        materials.push(entry);
    }
    let textures: Vec<Value> = (0..images.len())
        .map(|source| json!({ "sampler": 0, "source": source }))
        .collect();

    let document_json = json!({
        "asset": { "version": "2.0", "generator": "RustPaint" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "mesh": 0, "name": name }],
        "meshes": [{ "name": name, "primitives": primitives }],
        "materials": materials,
        "textures": textures,
        "images": images,
        "samplers": [{
            "magFilter": LINEAR, "minFilter": LINEAR_MIPMAP_LINEAR,
            "wrapS": REPEAT, "wrapT": REPEAT,
        }],
        "accessors": accessors,
        "bufferViews": buffer.views,
        "buffers": [{ "byteLength": buffer.bytes.len() }],
    });
    let json = serde_json::to_vec(&document_json).map_err(|e| e.to_string())?;
    container(json, buffer.bytes)
}

/// The binary data of the file, and the views into it.
#[derive(Default)]
struct Buffer {
    bytes: Vec<u8>,
    views: Vec<Value>,
}

impl Buffer {
    /// Appends `data` as a new buffer view (4-byte aligned) and returns its index.
    fn push(&mut self, data: &[u8], target: Option<u32>) -> usize {
        while !self.bytes.len().is_multiple_of(4) {
            self.bytes.push(0);
        }
        let mut view = json!({
            "buffer": 0,
            "byteOffset": self.bytes.len(),
            "byteLength": data.len(),
        });
        if let Some(target) = target {
            view["target"] = json!(target);
        }
        self.bytes.extend_from_slice(data);
        self.views.push(view);
        self.views.len() - 1
    }
}

/// Wraps the JSON and binary chunks in the `.glb` container.
fn container(mut json: Vec<u8>, mut binary: Vec<u8>) -> Result<Vec<u8>, String> {
    // Chunks are 4-byte aligned: JSON padded with spaces, binary with zeros.
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    while !binary.len().is_multiple_of(4) {
        binary.push(0);
    }
    let total = 12 + 8 + json.len() + 8 + binary.len();
    let total = u32::try_from(total).map_err(|_| "the model is too large for a .glb file")?;

    let mut out = Vec::with_capacity(total as usize);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&total.to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json);
    out.extend_from_slice(&(binary.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&binary);
    Ok(out)
}

/// One material's PNGs.
struct MaterialTextures {
    base_color: Vec<u8>,
    metal_rough: Vec<u8>,
    /// Only for a material with bumps.
    normal: Option<Vec<u8>>,
}

/// Per material: the base colour PNG, the metallic-roughness PNG and the
/// normal map if it has bumps. Combining the layers and compressing the
/// PNGs runs one thread per material.
fn encode_textures(document: &Document) -> Result<Vec<MaterialTextures>, String> {
    let count = document.mesh.materials.len();
    std::thread::scope(|scope| {
        let jobs: Vec<_> = (0..count)
            .map(|material| scope.spawn(move || material_textures(document, material)))
            .collect();
        jobs.into_iter()
            .map(|job| {
                job.join()
                    .unwrap_or_else(|_| Err("texture export thread failed".into()))
            })
            .collect()
    })
}

fn material_textures(document: &Document, material: usize) -> Result<MaterialTextures, String> {
    let composite = |channel| {
        document
            .composite(material, channel)
            .ok_or_else(|| "material slot not found".to_string())
    };
    let base_color = composite(Channel::BaseColor)?;
    let roughness = composite(Channel::Roughness)?;
    let metallic = composite(Channel::Metallic)?;
    if (roughness.width, roughness.height) != (metallic.width, metallic.height) {
        return Err("material channels differ in size".into());
    }
    // Data channels keep their value in red (see `Material`).
    let packed: Vec<u8> = roughness
        .rgba
        .chunks_exact(4)
        .zip(metallic.rgba.chunks_exact(4))
        .flat_map(|(r, m)| [255, r[0], m[0]])
        .collect();
    let height = composite(Channel::Height)?;
    let level = height.rgba.chunks_exact(4).all(|p| p[0] == height.rgba[0]);
    let normal = if level {
        None
    } else {
        let normals = document
            .normal_map(material, &height)
            .ok_or("material slot not found")?;
        let rgb = normals
            .rgba
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]]);
        let normals = Image {
            rgba: rgb.collect(),
            ..normals
        };
        Some(png(&normals, image::ColorType::Rgb8)?)
    };
    Ok(MaterialTextures {
        base_color: png(&base_color, image::ColorType::Rgba8)?,
        metal_rough: png(
            &Image {
                width: roughness.width,
                height: roughness.height,
                rgba: packed,
            },
            image::ColorType::Rgb8,
        )?,
        normal,
    })
}

/// A tangent per vertex, as glTF wants: xyz along increasing u, w the sign
/// that turns `cross(normal, tangent)` into the direction of decreasing v
/// (up in the image, the normal map's green).
fn tangents(mesh: &crate::mesh::Mesh) -> Vec<[f32; 4]> {
    use glam::{Vec2, Vec3};
    let mut along_u = vec![Vec3::ZERO; mesh.vertices.len()];
    let mut along_v = vec![Vec3::ZERO; mesh.vertices.len()];
    for corners in mesh.indices.chunks_exact(3) {
        let Some(v) = corners
            .iter()
            .map(|&i| mesh.vertices.get(i as usize))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let p = [0, 1, 2].map(|i| Vec3::from(v[i].position));
        let uv = [0, 1, 2].map(|i| Vec2::from(v[i].uv));
        let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
        let (d1, d2) = (uv[1] - uv[0], uv[2] - uv[0]);
        let area = d1.x * d2.y - d2.x * d1.y;
        if area.abs() < 1e-12 {
            continue;
        }
        // Weighted by the triangle's size in space, like a smooth normal.
        let weight = e1.cross(e2).length();
        let u = (e1 * d2.y - e2 * d1.y) / area * weight;
        let v = (e2 * d1.x - e1 * d2.x) / area * weight;
        if !(u.is_finite() && v.is_finite()) {
            continue;
        }
        for &i in corners {
            along_u[i as usize] += u;
            along_v[i as usize] += v;
        }
    }
    mesh.vertices
        .iter()
        .zip(along_u.iter().zip(&along_v))
        .map(|(vertex, (&u, &v))| {
            let normal = Vec3::from(unit_normal(vertex.normal));
            // Gram-Schmidt: the part of u lying in the surface.
            let mut tangent = (u - normal * normal.dot(u)).normalize_or_zero();
            if !tangent.is_finite() || tangent == Vec3::ZERO {
                tangent = normal.any_orthonormal_vector();
            }
            let up = -v;
            let sign = if normal.cross(tangent).dot(up) < 0.0 {
                -1.0
            } else {
                1.0
            };
            [tangent.x, tangent.y, tangent.z, sign]
        })
        .collect()
}

/// PNG-compresses `image`, whose `rgba` holds pixels of `color` type.
fn png(image: &Image, color: image::ColorType) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    image::write_buffer_with_format(
        &mut std::io::Cursor::new(&mut bytes),
        &image.rgba,
        image.width,
        image.height,
        color,
        image::ImageFormat::Png,
    )
    .map_err(|e| e.to_string())?;
    Ok(bytes)
}

fn bounds(positions: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for position in positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(position[axis]);
            max[axis] = max[axis].max(position[axis]);
        }
    }
    (min, max)
}

/// glTF requires unit-length normals; a missing (zero) one becomes "up".
fn unit_normal(normal: [f32; 3]) -> [f32; 3] {
    let normal = glam::Vec3::from(normal);
    if normal.length_squared() > 0.25 {
        normal.normalize().to_array()
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Document;
    use crate::mesh::{gltf_import, primitives};

    #[test]
    fn exported_model_loads_back_with_its_painted_textures() {
        let mut document = Document::new(primitives::cube());
        // A layer that paints the top-left corner with a colour, roughness and metallic.
        let index = document.layers.len();
        let _ = document.insert_new_layer(index);
        let (width, height) = document.slot_sizes()[0];
        let layer = &mut document.layers[index];
        let values = [
            (Channel::BaseColor, [200, 40, 10, 255]),
            (Channel::Roughness, [77, 77, 77, 255]),
            (Channel::Metallic, [255, 255, 255, 255]),
        ];
        for (channel, value) in values {
            let mut image = Image::filled(width, height, [0, 0, 0, 0]);
            image.rgba[..4].copy_from_slice(&value);
            layer.textures[0][channel.index()] = Some(image);
        }

        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("rustpaint-glb-{}-{nonce}.glb", std::process::id()));
        write(&path, &document, "cube").unwrap();
        let loaded = gltf_import::load(&path);
        std::fs::remove_file(&path).unwrap();
        let loaded = loaded.unwrap();

        let original = &document.mesh;
        assert_eq!(loaded.vertices.len(), original.vertices.len());
        assert_eq!(loaded.indices, original.indices);
        assert_eq!(loaded.materials.len(), 1);
        let material = &loaded.materials[0];
        assert_eq!(material.name, original.materials[0].name);
        assert_eq!(
            (material.base_color.width, material.base_color.height),
            (width, height)
        );
        assert_eq!(material.base_color.rgba[..4], [200, 40, 10, 255]);
        assert_eq!(material.roughness.rgba[0], 77);
        assert_eq!(material.metallic.rgba[0], 255);
        // Unpainted texels keep the base: the checker, roughness 0.5, metallic 0.
        assert_eq!(
            material.base_color.rgba[4..8],
            original.materials[0].base_color.rgba[4..8]
        );
        assert_eq!(material.roughness.rgba[4], 128);
        assert_eq!(material.metallic.rgba[4], 0);
    }

    #[test]
    fn bumps_are_exported_as_a_normal_map_with_tangents() {
        let flat = Document::new(primitives::cube());
        let file = gltf::Gltf::from_slice(&encode(&flat, "cube").unwrap()).unwrap();
        assert!(file.materials().all(|m| m.normal_texture().is_none()));
        let primitive = file.meshes().next().unwrap().primitives().next().unwrap();
        assert!(primitive.get(&gltf::Semantic::Tangents).is_none());

        // A raised square in one corner of the texture.
        let mut document = Document::new(primitives::cube());
        let (width, height) = document.slot_sizes()[0];
        let mut bump = Image::filled(width, height, [0, 0, 0, 0]);
        for y in 100..140 {
            for x in 100..140 {
                let i = ((y * width + x) * 4) as usize;
                bump.rgba[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        document.layers[1].textures[0][Channel::Height.index()] = Some(bump);
        let bytes = encode(&document, "cube").unwrap();
        let file = gltf::Gltf::from_slice(&bytes).unwrap();
        let material = file.materials().next().unwrap();
        assert!(material.normal_texture().is_some());
        let primitive = file.meshes().next().unwrap().primitives().next().unwrap();
        assert!(primitive.get(&gltf::Semantic::Tangents).is_some());

        // Each tangent lies in the surface along increasing u.
        let mesh = &document.mesh;
        let tangents = tangents(mesh);
        for triangle in mesh.indices.as_chunks::<3>().0 {
            let v = triangle.map(|i| &mesh.vertices[i as usize]);
            let p = v.map(|v| glam::Vec3::from(v.position));
            let uv = v.map(|v| glam::Vec2::from(v.uv));
            let (e1, e2) = (p[1] - p[0], p[2] - p[0]);
            let (d1, d2) = (uv[1] - uv[0], uv[2] - uv[0]);
            let along_u = (e1 * d2.y - e2 * d1.y) / (d1.x * d2.y - d2.x * d1.y);
            let along_v = (e2 * d1.x - e1 * d2.x) / (d1.x * d2.y - d2.x * d1.y);
            for (&index, vertex) in triangle.iter().zip(v) {
                let [x, y, z, w] = tangents[index as usize];
                let tangent = glam::Vec3::new(x, y, z);
                let normal = glam::Vec3::from(vertex.normal);
                assert!((tangent.length() - 1.0).abs() < 1e-4);
                assert!(tangent.dot(normal).abs() < 1e-4);
                assert!(tangent.dot(along_u.normalize()) > 0.99);
                // The bitangent points up the image: towards decreasing v.
                assert!((normal.cross(tangent) * w).dot(along_v) < 0.0);
            }
        }
    }
}
