//! The `.rustpaint` project file.
//!
//! Layout: magic, the mesh with its base colour textures, then (version 3)
//! each material's roughness and metallic, then the paint layers. Layer
//! textures are mostly transparent, so only their 64x64 tiles that contain
//! paint are stored. Version 4 adds layer masks: the value where the mask
//! is unpainted, then the tiles that differ from it, one byte per pixel.
//! Older versions still open: version 1 has no layers, version 2 has base
//! colour layers only and no blend modes, version 3 has no masks.
//! All counts are validated before allocating, so a damaged or hostile file
//! produces an error instead of a crash.

use crate::document::{BlendMode, CLEAR, Channel, Document, Layer, LayerProps, Plane};
use crate::files;
use crate::mesh::{Image, ImageRegion, Material, Mesh, SubMesh, Vertex};
use crate::painting::history::TILE_SIZE;
use std::io::Read;
use std::path::Path;

const MAGIC_V1: &[u8; 8] = b"RPAINT01";
const MAGIC_V2: &[u8; 8] = b"RPAINT02";
const MAGIC_V3: &[u8; 8] = b"RPAINT03";
const MAGIC_V4: &[u8; 8] = b"RPAINT04";
/// Channels after base colour, stored one byte per pixel per material.
const DATA_CHANNELS: [Channel; 2] = [Channel::Roughness, Channel::Metallic];
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
    files::write_atomically(path, &encode(document)?)
}

/// The project file's contents for `document`.
pub fn encode(document: &Document) -> Result<Vec<u8>, String> {
    let mut bytes = encode_mesh(&document.mesh)?;
    bytes[..MAGIC_V4.len()].copy_from_slice(MAGIC_V4);
    for material in &document.mesh.materials {
        for channel in DATA_CHANNELS {
            let image = material.channel(channel);
            if (image.width, image.height)
                != (material.base_color.width, material.base_color.height)
            {
                return Err("material channels differ in size".into());
            }
            bytes.extend(image.rgba.chunks_exact(4).map(|pixel| pixel[0]));
        }
    }
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
    let version = [MAGIC_V1, MAGIC_V2, MAGIC_V3, MAGIC_V4]
        .iter()
        .position(|known| magic == *known)
        .ok_or("not a supported RustPaint project file")?
        + 1;
    let mut mesh = decode_mesh(&mut reader)?;
    if version >= 3 {
        for material in &mut mesh.materials {
            let (width, height) = (material.base_color.width, material.base_color.height);
            for channel in DATA_CHANNELS {
                let values = reader.take(width as usize * height as usize)?;
                let image = material.channel_mut(channel);
                for (pixel, &value) in image.rgba.chunks_exact_mut(4).zip(values) {
                    pixel[..3].fill(value);
                }
            }
        }
    }
    let document = match version {
        1 => Document::new(mesh),
        _ => decode_layers(&mut reader, mesh, version)?,
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
        // Data channels follow later in the file (version 3) or default.
        materials.push(Material::new(
            name,
            Image {
                width,
                height,
                rgba,
            },
        ));
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
        bytes.push(layer.blend.index());
        bytes.extend(layer.targets.iter().map(|&on| on as u8));
        if layer.is_base() {
            continue;
        }
        bytes.push(layer.mask.is_some() as u8);
        bytes.push(layer.mask.unwrap_or(0));
        for (slot, channel) in
            (0..slots).flat_map(|slot| Channel::ALL.map(|channel| (slot, channel)))
        {
            let image = layer
                .textures
                .get(slot)
                .and_then(|textures| textures[channel.index()].as_ref());
            let Some(image) = image else {
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
        if let Some(unpainted) = layer.mask {
            for slot in 0..slots {
                let image = layer
                    .textures
                    .get(slot)
                    .and_then(|planes| planes[Plane::Mask.index()].as_ref());
                encode_mask(bytes, image, unpainted)?;
            }
        }
    }
    Ok(())
}

/// A mask image's tiles that differ from its unpainted value, one byte per pixel.
fn encode_mask(bytes: &mut Vec<u8>, image: Option<&Image>, unpainted: u8) -> Result<(), String> {
    let Some(image) = image else {
        return write_count(bytes, 0);
    };
    let differs = |region: ImageRegion| {
        (region.y..region.y + region.height).any(|y| {
            let start = ((y * image.width + region.x) * 4) as usize;
            image.rgba[start..start + (region.width * 4) as usize]
                .chunks_exact(4)
                .any(|pixel| pixel[0] != unpainted)
        })
    };
    let painted: Vec<(usize, ImageRegion)> = (0..tile_count(image.width, image.height))
        .map(|index| (index, tile_region(index, image.width, image.height)))
        .filter(|(_, region)| differs(*region))
        .collect();
    write_count(bytes, painted.len())?;
    for (index, region) in painted {
        write_count(bytes, index)?;
        for y in region.y..region.y + region.height {
            let start = ((y * image.width + region.x) * 4) as usize;
            let row = &image.rgba[start..start + (region.width * 4) as usize];
            bytes.extend(row.chunks_exact(4).map(|pixel| pixel[0]));
        }
    }
    Ok(())
}

/// One mask image: its painted tiles over the unpainted value, or `None` when
/// nothing was painted.
fn decode_mask(
    reader: &mut Reader,
    width: u32,
    height: u32,
    unpainted: u8,
    budget: &mut usize,
) -> Result<Option<Image>, String> {
    let tiles = tile_count(width, height);
    let painted = reader.count(tiles, "mask tile")?;
    if painted == 0 {
        return Ok(None);
    }
    let bytes = (width as usize) * (height as usize) * 4;
    *budget = budget
        .checked_sub(bytes)
        .ok_or("project layers need more memory than supported")?;
    let mut image = Image::filled(width, height, [unpainted, unpainted, unpainted, 255]);
    for _ in 0..painted {
        let tile = reader.count(tiles - 1, "mask tile index")?;
        let region = tile_region(tile, width, height);
        for y in region.y..region.y + region.height {
            let row = reader.take(region.width as usize)?;
            let start = ((y * width + region.x) * 4) as usize;
            for (pixel, &value) in image.rgba[start..].chunks_exact_mut(4).zip(row) {
                pixel.copy_from_slice(&[value, value, value, 255]);
            }
        }
    }
    Ok(Some(image))
}

fn decode_layers(reader: &mut Reader, mesh: Mesh, version: usize) -> Result<Document, String> {
    // Version 2 had no blend modes and only base colour paint.
    let channels: &[Channel] = if version >= 3 {
        &Channel::ALL
    } else {
        &[Channel::BaseColor]
    };
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
        let blend = if version >= 3 {
            BlendMode::from_index(reader.take(1)?[0])
                .ok_or("project contains an unknown blend mode")?
        } else {
            BlendMode::Normal
        };
        let targets = (0..sizes.len())
            .map(|_| reader.flag("layer slot"))
            .collect::<Result<Vec<_>, _>>()?;
        let props = LayerProps {
            name,
            visible,
            opacity: opacity.clamp(0.0, 1.0),
            blend,
            targets,
        };
        if is_base {
            base = Some(props);
            continue;
        }

        let mask = if version >= 4 {
            let has_mask = reader.flag("layer mask")?;
            let unpainted = reader.take(1)?[0];
            has_mask.then_some(unpainted)
        } else {
            None
        };
        let mut textures: Vec<[Option<Image>; Plane::COUNT]> = Vec::with_capacity(sizes.len());
        for &(width, height) in &sizes {
            let mut slot: [Option<Image>; Plane::COUNT] = Default::default();
            for channel in channels {
                slot[channel.index()] =
                    decode_layer_texture(reader, width, height, &mut texture_budget)?;
            }
            textures.push(slot);
        }
        if let Some(unpainted) = mask {
            for (slot, &(width, height)) in textures.iter_mut().zip(&sizes) {
                slot[Plane::Mask.index()] =
                    decode_mask(reader, width, height, unpainted, &mut texture_budget)?;
            }
        }
        layers.push(Layer {
            id: index as u64,
            name: props.name,
            visible: props.visible,
            opacity: props.opacity,
            blend: props.blend,
            targets: props.targets,
            textures,
            mask,
        });
    }

    let mut document = Document::with_layers(mesh, layers);
    if let Some(mut props) = base {
        props.name = document.layers[0].name.clone();
        document.set_base_props(props);
    }
    Ok(document)
}

/// One layer texture: its painted tiles, or `None` when it has none.
fn decode_layer_texture(
    reader: &mut Reader,
    width: u32,
    height: u32,
    budget: &mut usize,
) -> Result<Option<Image>, String> {
    let tiles = tile_count(width, height);
    let painted = reader.count(tiles, "layer tile")?;
    if painted == 0 {
        return Ok(None);
    }
    let bytes = (width as usize) * (height as usize) * 4;
    *budget = budget
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
    Ok(Some(image))
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
    use crate::mesh::Channel;
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
        document.layers[1].textures[0][0] = Some(image.clone());
        // The same pixel painted into roughness too.
        image.rgba[last..].copy_from_slice(&[40, 40, 40, 255]);
        document.layers[1].textures[0][Channel::Roughness.index()] = Some(image);
        document.layers[1].name = "Rust".into();
        document.layers[1].opacity = 0.25;
        document.layers[1].blend = crate::document::BlendMode::Multiply;
        // A metallic value on the model itself.
        document.mesh.materials[0].metallic.rgba[0..3].fill(200);
        document.insert_new_layer(2);
        document.layers[2].visible = false;
        // A mask that hides everything except one painted corner.
        document.set_mask(1, Some(0)).unwrap();
        let mut mask = Image::filled(w, h, [0, 0, 0, 255]);
        mask.rgba[last..].copy_from_slice(&[200, 200, 200, 255]);
        document.layers[1].textures[0][crate::document::Plane::Mask.index()] = Some(mask);
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
            assert_eq!(actual.blend, expected.blend);
            assert_eq!(actual.visible, expected.visible);
            assert_eq!(actual.opacity, expected.opacity);
            assert_eq!(actual.targets, expected.targets);
            assert_eq!(actual.is_base(), expected.is_base());
        }
        let painted = |document: &Document| {
            document.layers[1].textures[0][0]
                .as_ref()
                .map(|image| image.rgba.clone())
        };
        assert_eq!(painted(&decoded), painted(&document));
        let mask = |document: &Document| {
            let layer = &document.layers[1];
            let image = layer.textures[0][crate::document::Plane::Mask.index()].as_ref();
            (layer.mask, image.map(|image| image.rgba.clone()))
        };
        assert_eq!(mask(&decoded), mask(&document));
        assert_eq!(decoded.layers[2].mask, None);
        assert!(
            decoded.layers[2]
                .textures
                .iter()
                .flatten()
                .all(Option::is_none)
        );
        assert_eq!(
            decoded.composite(0, Channel::BaseColor).unwrap().rgba,
            document.composite(0, Channel::BaseColor).unwrap().rgba
        );
        // Only the one painted tile is stored, not the whole layer texture.
        for channel in Channel::ALL {
            assert_eq!(
                decoded.composite(0, channel).unwrap().rgba,
                document.composite(0, channel).unwrap().rgba,
                "{}",
                channel.name()
            );
        }
        // Only the painted tiles (one per painted channel, and one of the
        // mask at a byte per pixel) are stored, plus one byte per pixel for
        // each data channel of the model.
        let (w, h) = document.slot_sizes()[0];
        let base = encode_mesh(&document.mesh).unwrap().len() + 2 * (w * h) as usize;
        assert!(bytes.len() < base + 2 * 64 * 64 * 4 + 64 * 64 + 1024);
    }

    #[test]
    fn version_1_files_open_with_the_texture_as_base_layer() {
        let bytes = encode_mesh(&primitives::cube()).unwrap();
        let document = decode(&bytes).unwrap();
        assert_eq!(document.layers.len(), 2);
        assert!(document.layers[0].is_base());
        assert_eq!(
            document.composite(0, Channel::BaseColor).unwrap().rgba,
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
