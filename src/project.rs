use crate::mesh::{Image, Material, Mesh, SubMesh, Vertex};
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"RPAINT01";
const MAX_PROJECT_BYTES: u64 = 512 * 1024 * 1024;
const MAX_VERTICES: usize = 20_000_000;
const MAX_INDICES: usize = 60_000_000;
const MAX_SUBMESHES: usize = MAX_INDICES / 3;
const MAX_MATERIALS: usize = 4096;
const MAX_TEXTURE_DIMENSION: u32 = 16_384;
const MAX_NAME_BYTES: usize = 1024 * 1024;

pub fn load(path: &Path) -> Result<Mesh, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    if size > MAX_PROJECT_BYTES {
        return Err("project file exceeds the 512 MiB size limit".into());
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size as usize)
        .map_err(|e| format!("could not allocate project file: {e}"))?;
    file.take(MAX_PROJECT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    if bytes.len() as u64 > MAX_PROJECT_BYTES {
        return Err("project file exceeds the 512 MiB size limit".into());
    }
    decode(&bytes)
}

pub fn save(path: &Path, mesh: &Mesh) -> Result<(), String> {
    let bytes = encode(mesh)?;
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

fn encode(mesh: &Mesh) -> Result<Vec<u8>, String> {
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
        return Err("project file exceeds the 512 MiB size limit".into());
    }
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(encoded_size as usize)
        .map_err(|e| format!("could not allocate project file: {e}"))?;
    bytes.extend_from_slice(MAGIC);
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
        return Err("project file exceeds the 512 MiB size limit".into());
    }
    Ok(bytes)
}

fn decode(bytes: &[u8]) -> Result<Mesh, String> {
    let mut reader = Reader { bytes, offset: 0 };
    if reader.take(MAGIC.len())? != MAGIC {
        return Err("not a supported RustPaint project file".into());
    }
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
    if reader.offset != bytes.len() {
        return Err("project file contains unexpected trailing data".into());
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
    use super::{decode, encode, load, save};
    use crate::mesh::primitives;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn project_round_trip_preserves_mesh_and_textures() {
        let mesh = primitives::cube();
        let decoded = decode(&encode(&mesh).unwrap()).unwrap();

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
        let mut bytes = encode(&primitives::cube()).unwrap();
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
        let mut mesh = primitives::cube();

        save(&path, &mesh).unwrap();
        mesh.materials[0].base_color.rgba[0] ^= 0xff;
        save(&path, &mesh).unwrap();
        let reopened = load(&path).unwrap();

        assert_eq!(
            reopened.materials[0].base_color.rgba,
            mesh.materials[0].base_color.rgba
        );
        std::fs::remove_file(path).unwrap();
    }
}
