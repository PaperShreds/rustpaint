//! The `.rustpaint` project file.
//!
//! Layout: magic, the mesh with its own textures (the base layer), then the
//! paint layers. Layer textures are mostly transparent, so only their 64x64
//! tiles that contain paint are stored. Version 1 files have no layer section.
//! All counts are validated before allocating, so a damaged or hostile file
//! produces an error instead of a crash.

use crate::document::{CLEAR, Document, Layer, LayerProps};
use crate::mesh::{Image, ImageRegion, Material, Mesh, SubMesh, Vertex};
use crate::painting::history::TILE_SIZE;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC_V1: &[u8; 8] = b"RPAINT01";
const MAGIC_V2: &[u8; 8] = b"RPAINT02";
const MAX_PROJECT_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const SIZE_LIMIT_ERROR: &str = "project file exceeds the 4 GiB size limit";
const MAX_LAYERS: usize = 4096;
/// Memory all decoded layer textures may take together.
const MAX_LAYER_TEXTURE_BYTES: usize = 4 * 1024 * 1024 * 1024;
const MAX_VERTICES: usize = 20_000_000;
const MAX_INDICES: usize = 60_000_000;
const MAX_SUBMESHES: usize = MAX_INDICES / 3;
const MAX_MATERIALS: usize = 4096;
const MAX_TEXTURE_DIMENSION: u32 = 16_384;
const MAX_NAME_BYTES: usize = 1024 * 1024;

pub fn load(path: &Path) -> Result<Document, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    if size > MAX_PROJECT_BYTES {
        return Err(SIZE_LIMIT_ERROR.into());
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size as usize)
        .map_err(|e| format!("could not allocate project file: {e}"))?;
    file.take(MAX_PROJECT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_PROJECT_BYTES {
        return Err(SIZE_LIMIT_ERROR.into());
    }
    decode(&bytes)
}

pub fn save(path: &Path, document: &Document) -> Result<(), String> {
    let bytes = encode(document)?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("project path has no valid file name")?;
    let temporary_path = path.with_file_name(format!(".{file_name}.{}.tmp", std::process::id()));

    let mut temporary_file_created = false;
    let write_result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary_path)
            .map_err(|e| e.to_string())?;
        temporary_file_created = true;
        file.write_all(&bytes).map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(&temporary_path, path).map_err(|e| e.to_string())
    })();

    if write_result.is_err() && temporary_file_created {
        let _ = std::fs::remove_file(&temporary_path);
    }
    write_result
}

fn encode(document: &Document) -> Result<Vec<u8>, String> {
    let mut bytes = encode_mesh(&document.mesh)?;
    bytes[..MAGIC_V2.len()].copy_from_slice(MAGIC_V2);
    encode_layers(&mut bytes, document)?;
    if bytes.len() as u64 > MAX_PROJECT_BYTES {
        return Err(SIZE_LIMIT_ERROR.into());
    }
    Ok(bytes)
}

/// The mesh and its textures in the version 1 layout, magic included.
fn encode_mesh(mesh: &Mesh) -> Result<Vec<u8>, String> {
    validate_mesh(mesh)?;
    let mut encoded_size = 24_u64
        .checked_add((mesh.vertices.len() as u64) * 32)
        .and_then(|size| size.checked_add((mesh.indices.len() as u64) * 4))
        .and_then(|size| size.checked_add((mesh.submeshes.len() as u64) * 12))
        .ok_or_else(|| "project file size overflows".to_string())?;
    for material in &mesh.materials {
        encoded_size = encoded_size
            .checked_add(12 + material.name.len() as u64)
            .and_then(|size| size.checked_add(material.base_color.rgba.len() as u64))
            .ok_or_else(|| "project file size overflows".to_string())?;
    }
    if encoded_size > MAX_PROJECT_BYTES {
        return Err(SIZE_LIMIT_ERROR.into());
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(encoded_size as usize)
        .map_err(|e| format!("could not allocate project file: {e}"))?;
    bytes.extend_from_slice(MAGIC_V1);
    for count in [
        mesh.vertices.len(),
        mesh.indices.len(),
        mesh.submeshes.len(),
        mesh.materials.len(),
    ] {
        write_count(&mut bytes, count)?;
    }

    for vertex in &mesh.vertices {
        for value in vertex
            .position
            .iter()
            .chain(&vertex.normal)
            .chain(&vertex.uv)
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for index in &mesh.indices {
        bytes.extend_from_slice(&index.to_le_bytes());
    }
    for submesh in &mesh.submeshes {
        write_count(&mut bytes, submesh.indices.start as usize)?;
        write_count(&mut bytes, submesh.indices.end as usize)?;
        write_count(&mut bytes, submesh.material)?;
    }
    for material in &mesh.materials {
        write_count(&mut bytes, material.name.len())?;
        bytes.extend_from_slice(material.name.as_bytes());
        bytes.extend_from_slice(&material.base_color.width.to_le_bytes());
        bytes.extend_from_slice(&material.base_color.height.to_le_bytes());
        bytes.extend_from_slice(&material.base_color.rgba);
    }

    if bytes.len() as u64 > MAX_PROJECT_BYTES {
        return Err(SIZE_LIMIT_ERROR.into());
    }
    Ok(bytes)
}

fn decode(bytes: &[u8]) -> Result<Document, String> {
    let mut reader = Reader { bytes, offset: 0 };
    let magic = reader.take(MAGIC_V1.len())?;
    let has_layers = if magic == MAGIC_V2 {
        true
    } else if magic == MAGIC_V1 {
        false
    } else {
        return Err("not a supported RustPaint project file".into());
    };
    let mesh = decode_mesh(&mut reader)?;
    let document = if has_layers {
        decode_layers(&mut reader, mesh)?
    } else {
        Document::new(mesh)
    };
    if reader.offset != bytes.len() {
        return Err("project file contains unexpected trailing data".into());
    }
    Ok(document)
}

fn decode_mesh(reader: &mut Reader) -> Result<Mesh, String> {
    let vertex_count = reader.count(MAX_VERTICES, "vertex")?;
    let index_count = reader.count(MAX_INDICES, "index")?;
    let submesh_count = reader.count(MAX_SUBMESHES, "submesh")?;
    let material_count = reader.count(MAX_MATERIALS, "material")?;
    reader.ensure_remaining(
        vertex_count
            .checked_mul(32)
            .ok_or("project vertex size overflows")?,
    )?;
    let mut vertices = Vec::new();
    vertices
        .try_reserve_exact(vertex_count)
        .map_err(|e| format!("could not allocate project vertices: {e}"))?;
    for _ in 0..vertex_count {
        let mut values = [0.0; 8];
        for value in &mut values {
            *value = f32::from_le_bytes(reader.array()?);
        }
        vertices.push(Vertex {
            position: values[0..3].try_into().unwrap(),
            normal: values[3..6].try_into().unwrap(),
            uv: values[6..8].try_into().unwrap(),
        });
    }

    reader.ensure_remaining(
        index_count
            .checked_mul(4)
            .ok_or("project index size overflows")?,
    )?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(index_count)
        .map_err(|e| format!("could not allocate project indices: {e}"))?;
    for _ in 0..index_count {
        indices.push(reader.u32()?);
    }

    reader.ensure_remaining(
        submesh_count
            .checked_mul(12)
            .ok_or("project submesh size overflows")?,
    )?;
    let mut submeshes = Vec::new();
    submeshes
        .try_reserve_exact(submesh_count)
        .map_err(|e| format!("could not allocate project submeshes: {e}"))?;
    for _ in 0..submesh_count {
        submeshes.push(SubMesh {
            indices: reader.u32()?..reader.u32()?,
            material: reader.u32()? as usize,
        });
    }

    let mut materials = Vec::new();
    materials
        .try_reserve_exact(material_count)
        .map_err(|e| format!("could not allocate project materials: {e}"))?;
    for _ in 0..material_count {
        let name_len = reader.count(MAX_NAME_BYTES, "material name byte")?;
        let name = std::str::from_utf8(reader.take(name_len)?)
            .map_err(|e| format!("invalid material name: {e}"))?
            .to_owned();
        let width = reader.u32()?;
        let height = reader.u32()?;
        let pixel_bytes = texture_byte_len(width, height)?;
        let rgba = reader.take(pixel_bytes)?.to_vec();
        materials.push(Material {
            name,
            base_color: Image {
                width,
                height,
                rgba,
            },
        });
    }
    let mesh = Mesh {
        vertices,
        indices,
        submeshes,
        materials,
    };
    validate_mesh(&mesh)?;
    Ok(mesh)
}

fn encode_layers(bytes: &mut Vec<u8>, document: &Document) -> Result<(), String> {
    let slots = document.mesh.materials.len();
    write_count(bytes, document.layers.len())?;
    for layer in &document.layers {
        if layer.targets.len() != slots {
            return Err("layer does not match the model's material slots".into());
        }
        bytes.push(layer.is_base() as u8);
        write_count(bytes, layer.name.len())?;
        bytes.extend_from_slice(layer.name.as_bytes());
        bytes.push(layer.visible as u8);
        bytes.extend_from_slice(&layer.opacity.to_le_bytes());
        bytes.extend(layer.targets.iter().map(|&on| on as u8));
        if layer.is_base() {
            continue;
        }
        for slot in 0..slots {
            let Some(image) = layer.textures.get(slot).and_then(Option::as_ref) else {
                write_count(bytes, 0)?;
                continue;
            };
            let painted: Vec<(usize, ImageRegion)> = (0..tile_count(image.width, image.height))
                .map(|index| (index, tile_region(index, image.width, image.height)))
                .filter(|(_, region)| region_has_paint(image, *region))
                .collect();
            write_count(bytes, painted.len())?;
            for (index, region) in painted {
                write_count(bytes, index)?;
                for y in region.y..region.y + region.height {
                    let start = ((y * image.width + region.x) * 4) as usize;
                    bytes
                        .extend_from_slice(&image.rgba[start..start + (region.width * 4) as usize]);
                }
            }
        }
    }
    Ok(())
}

fn decode_layers(reader: &mut Reader, mesh: Mesh) -> Result<Document, String> {
    let count = reader.count(MAX_LAYERS, "layer")?;
    if count == 0 {
        return Err("project has no base layer".into());
    }
    let sizes: Vec<(u32, u32)> = mesh
        .materials
        .iter()
        .map(|m| (m.base_color.width, m.base_color.height))
        .collect();
    let mut texture_budget = MAX_LAYER_TEXTURE_BYTES;
    let mut base = None;
    let mut layers = Vec::new();

    for index in 0..count {
        let is_base = reader.flag("layer kind")?;
        if is_base != (index == 0) {
            return Err("project layers are out of order".into());
        }
        let name_len = reader.count(MAX_NAME_BYTES, "layer name byte")?;
        let name = std::str::from_utf8(reader.take(name_len)?)
            .map_err(|e| format!("invalid layer name: {e}"))?
            .to_owned();
        let visible = reader.flag("layer visibility")?;
        let opacity = f32::from_le_bytes(reader.array()?);
        if !opacity.is_finite() {
            return Err("project contains an invalid layer opacity".into());
        }
        let targets = (0..sizes.len())
            .map(|_| reader.flag("layer slot"))
            .collect::<Result<Vec<_>, _>>()?;
        let props = LayerProps {
            name,
            visible,
            opacity: opacity.clamp(0.0, 1.0),
            targets,
        };
        if is_base {
            base = Some(props);
            continue;
        }

        let mut textures = Vec::with_capacity(sizes.len());
        for &(width, height) in &sizes {
            let tiles = tile_count(width, height);
            let painted = reader.count(tiles, "layer tile")?;
            if painted == 0 {
                textures.push(None);
                continue;
            }
            let bytes = (width as usize) * (height as usize) * 4;
            texture_budget = texture_budget
                .checked_sub(bytes)
                .ok_or("project layers need more memory than supported")?;
            let mut image = Image::filled(width, height, CLEAR);
            for _ in 0..painted {
                let tile = reader.count(tiles - 1, "layer tile index")?;
                let region = tile_region(tile, width, height);
                for y in region.y..region.y + region.height {
                    let row = reader.take((region.width * 4) as usize)?;
                    let start = ((y * width + region.x) * 4) as usize;
                    image.rgba[start..start + row.len()].copy_from_slice(row);
                }
            }
            textures.push(Some(image));
        }
        layers.push(Layer {
            id: index as u64,
            name: props.name,
            visible: props.visible,
            opacity: props.opacity,
            targets: props.targets,
            textures,
        });
    }

    let mut document = Document::with_layers(mesh, layers);
    if let Some(mut props) = base {
        props.name = document.layers[0].name.clone();
        document.set_base_props(props);
    }
    Ok(document)
}

fn tile_count(width: u32, height: u32) -> usize {
    (width.div_ceil(TILE_SIZE) * height.div_ceil(TILE_SIZE)) as usize
}

fn tile_region(index: usize, width: u32, height: u32) -> ImageRegion {
    let columns = width.div_ceil(TILE_SIZE);
    let (x, y) = (
        index as u32 % columns * TILE_SIZE,
        index as u32 / columns * TILE_SIZE,
    );
    ImageRegion {
        x,
        y,
        width: TILE_SIZE.min(width - x),
        height: TILE_SIZE.min(height - y),
    }
}

fn region_has_paint(image: &Image, region: ImageRegion) -> bool {
    (region.y..region.y + region.height).any(|y| {
        let start = ((y * image.width + region.x) * 4) as usize;
        image.rgba[start..start + (region.width * 4) as usize]
            .chunks_exact(4)
            .any(|pixel| pixel[3] != 0)
    })
}

fn validate_mesh(mesh: &Mesh) -> Result<(), String> {
    if mesh.vertices.is_empty() || mesh.vertices.len() > MAX_VERTICES {
        return Err("project has an invalid vertex count".into());
    }
    if mesh.indices.is_empty()
        || mesh.indices.len() > MAX_INDICES
        || !mesh.indices.len().is_multiple_of(3)
    {
        return Err("project has an invalid index count".into());
    }
    if mesh.materials.is_empty() || mesh.materials.len() > MAX_MATERIALS {
        return Err("project has an invalid material count".into());
    }
    if mesh.submeshes.is_empty() {
        return Err("project has no submeshes".into());
    }
    if mesh.submeshes.len() > MAX_SUBMESHES {
        return Err("project has too many submeshes".into());
    }
    for vertex in &mesh.vertices {
        if vertex
            .position
            .iter()
            .chain(&vertex.normal)
            .chain(&vertex.uv)
            .any(|value| !value.is_finite())
        {
            return Err("project contains a non-finite vertex value".into());
        }
    }
    if mesh
        .indices
        .iter()
        .any(|&index| index as usize >= mesh.vertices.len())
    {
        return Err("project contains an out-of-range mesh index".into());
    }
    for submesh in &mesh.submeshes {
        if submesh.indices.start >= submesh.indices.end
            || submesh.indices.end as usize > mesh.indices.len()
            || submesh.indices.start % 3 != 0
            || submesh.indices.end % 3 != 0
            || submesh.material >= mesh.materials.len()
        {
            return Err("project contains an invalid submesh".into());
        }
    }
    for material in &mesh.materials {
        let expected_size =
            texture_byte_len(material.base_color.width, material.base_color.height)?;
        if material.base_color.rgba.len() != expected_size {
            return Err("project contains an invalid texture buffer".into());
        }
        if material.name.len() > MAX_NAME_BYTES {
            return Err("project material name is too long".into());
        }
    }
    Ok(())
}

fn texture_byte_len(width: u32, height: u32) -> Result<usize, String> {
    if width == 0 || height == 0 || width > MAX_TEXTURE_DIMENSION || height > MAX_TEXTURE_DIMENSION
    {
        return Err("project contains invalid texture dimensions".into());
    }
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|pixels| pixels.checked_mul(4))
        .ok_or_else(|| "project texture size overflows".into())
}

fn write_count(bytes: &mut Vec<u8>, count: usize) -> Result<(), String> {
    let count = u32::try_from(count).map_err(|_| "project value is too large".to_string())?;
    bytes.extend_from_slice(&count.to_le_bytes());
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl Reader<'_> {
    fn ensure_remaining(&self, count: usize) -> Result<(), String> {
        if count > self.bytes.len().saturating_sub(self.offset) {
            return Err("project file is truncated".into());
        }
        Ok(())
    }

    fn take(&mut self, count: usize) -> Result<&[u8], String> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or_else(|| "project file offset overflows".to_string())?;
        let result = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| "project file is truncated".to_string())?;
        self.offset = end;
        Ok(result)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], String> {
        self.take(N)?
            .try_into()
            .map_err(|_| "project file is truncated".to_string())
    }

    fn flag(&mut self, what: &str) -> Result<bool, String> {
        match self.take(1)?[0] {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(format!("project contains an invalid {what}")),
        }
    }

    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn count(&mut self, maximum: usize, kind: &str) -> Result<usize, String> {
        let count = self.u32()? as usize;
        if count > maximum {
            return Err(format!("project {kind} count exceeds the supported limit"));
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::{decode, encode, encode_mesh, load, save};
    use crate::document::{CLEAR, Document};
    use crate::mesh::{Image, primitives};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn project_round_trip_preserves_mesh_and_textures() {
        let document = Document::new(primitives::cube());
        let decoded = decode(&encode(&document).unwrap()).unwrap();
        let (mesh, decoded) = (&document.mesh, &decoded.mesh);

        assert_eq!(decoded.vertices.len(), mesh.vertices.len());
        assert_eq!(decoded.indices, mesh.indices);
        assert_eq!(decoded.submeshes.len(), mesh.submeshes.len());
        assert_eq!(decoded.materials.len(), mesh.materials.len());
        for (actual, expected) in decoded.materials.iter().zip(&mesh.materials) {
            assert_eq!(actual.name, expected.name);
            assert_eq!(actual.base_color.width, expected.base_color.width);
            assert_eq!(actual.base_color.height, expected.base_color.height);
            assert_eq!(actual.base_color.rgba, expected.base_color.rgba);
        }
    }

    #[test]
    fn project_reader_rejects_truncated_data() {
        let mut bytes = encode(&Document::new(primitives::cube())).unwrap();
        bytes.pop();

        assert!(decode(&bytes).is_err());
    }

    #[test]
    fn project_file_can_be_saved_reopened_and_overwritten() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "rustpaint-test-{}-{nonce}.rustpaint",
            std::process::id()
        ));
        let mut document = Document::new(primitives::cube());

        save(&path, &document).unwrap();
        document.mesh.materials[0].base_color.rgba[0] ^= 0xff;
        save(&path, &document).unwrap();
        let reopened = load(&path).unwrap().mesh;
        let mesh = &document.mesh;

        assert_eq!(
            reopened.materials[0].base_color.rgba,
            mesh.materials[0].base_color.rgba
        );
        std::fs::remove_file(path).unwrap();
    }

    /// A cube document with a second, hidden paint layer holding one red pixel.
    fn layered_document() -> Document {
        let mut document = Document::new(primitives::cube());
        let (w, h) = document.slot_sizes()[0];
        let mut image = Image::filled(w, h, CLEAR);
        let last = image.rgba.len() - 4;
        image.rgba[last..].copy_from_slice(&[255, 0, 0, 255]);
        document.layers[1].textures[0] = Some(image);
        document.layers[1].name = "Rust".into();
        document.layers[1].opacity = 0.25;
        document.insert_new_layer(2);
        document.layers[2].visible = false;
        document.layers[0].targets[0] = false;
        document
    }

    #[test]
    fn layers_round_trip() {
        let document = layered_document();
        let bytes = encode(&document).unwrap();
        let decoded = decode(&bytes).unwrap();

        assert_eq!(decoded.layers.len(), 3);
        for (actual, expected) in decoded.layers.iter().zip(&document.layers) {
            assert_eq!(actual.name, expected.name);
            assert_eq!(actual.visible, expected.visible);
            assert_eq!(actual.opacity, expected.opacity);
            assert_eq!(actual.targets, expected.targets);
            assert_eq!(actual.is_base(), expected.is_base());
        }
        let painted = |document: &Document| {
            document.layers[1].textures[0]
                .as_ref()
                .map(|image| image.rgba.clone())
        };
        assert_eq!(painted(&decoded), painted(&document));
        assert!(decoded.layers[2].textures.iter().all(Option::is_none));
        assert_eq!(
            decoded.composite(0).unwrap().rgba,
            document.composite(0).unwrap().rgba
        );
        // Only the one painted tile is stored, not the whole layer texture.
        let base_only = encode_mesh(&document.mesh).unwrap().len();
        assert!(bytes.len() < base_only + 64 * 64 * 4 + 1024);
    }

    #[test]
    fn version_1_files_open_with_the_texture_as_base_layer() {
        let bytes = encode_mesh(&primitives::cube()).unwrap();
        let document = decode(&bytes).unwrap();
        assert_eq!(document.layers.len(), 2);
        assert!(document.layers[0].is_base());
        assert_eq!(
            document.composite(0).unwrap().rgba,
            primitives::cube().materials[0].base_color.rgba
        );
    }

    #[test]
    fn damaged_layer_data_is_rejected() {
        let bytes = encode(&layered_document()).unwrap();
        // Every truncation fails cleanly instead of panicking.
        for cut in [1, 10, 100, 1000, 10_000] {
            assert!(decode(&bytes[..bytes.len() - cut]).is_err());
        }
        // Append junk.
        let mut extended = bytes.clone();
        extended.push(0);
        assert!(decode(&extended).is_err());
    }
}
