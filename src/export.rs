//! Exporting painted textures as image files.

use crate::document::Document;
use crate::mesh::Image;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// One texture file to write.
pub struct ExportFile {
    pub path: PathBuf,
    /// The material slot whose combined layers go into this file.
    pub material: usize,
}

/// The PNG files exporting `document` into `directory` would write, one per
/// material slot with its visible layers combined: `<model>_<material>_BaseColor.png`.
pub fn plan(directory: &Path, model_name: &str, document: &Document) -> Vec<ExportFile> {
    let model = sanitize(model_name);
    let mut used = HashSet::new();
    document
        .mesh
        .materials
        .iter()
        .enumerate()
        .map(|(index, material)| {
            let base = format!("{model}_{}_BaseColor", sanitize(&material.name));
            // Two materials can sanitize to the same name; keep files distinct.
            let mut name = base.clone();
            let mut suffix = 2;
            while !used.insert(name.to_lowercase()) {
                name = format!("{base}_{suffix}");
                suffix += 1;
            }
            ExportFile {
                path: directory.join(format!("{name}.png")),
                material: index,
            }
        })
        .collect()
}

/// Blends each slot's layers and writes the files, one thread per file. Each
/// is written to a temporary file first and then renamed, so an interrupted
/// export never leaves a half-written PNG behind.
pub fn write(files: &[ExportFile], document: &Document) -> Result<(), String> {
    std::thread::scope(|scope| {
        let jobs: Vec<_> = files
            .iter()
            .map(|file| {
                scope.spawn(move || {
                    let image = document
                        .composite(file.material)
                        .ok_or("material slot not found")?;
                    write_png(&file.path, &image)
                })
            })
            .collect();
        for (file, job) in files.iter().zip(jobs) {
            job.join()
                .unwrap_or_else(|_| Err("export thread failed".into()))
                .map_err(|error| format!("{}: {error}", display_name(&file.path)))?;
        }
        Ok(())
    })
}

fn write_png(path: &Path, image: &Image) -> Result<(), String> {
    let buffer = image::RgbaImage::from_raw(image.width, image.height, image.rgba.clone())
        .ok_or("texture data does not match its size")?;
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        display_name(path),
        std::process::id()
    ));

    let result = (|| {
        let file = std::fs::File::create(&temporary).map_err(|e| e.to_string())?;
        let mut writer = std::io::BufWriter::new(file);
        buffer
            .write_to(&mut writer, image::ImageFormat::Png)
            .map_err(|e| e.to_string())?;
        let file = writer.into_inner().map_err(|e| e.to_string())?;
        file.sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(&temporary, path).map_err(|e| e.to_string())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    result
}

fn display_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path.display().to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Makes a name safe to use in a file name on every platform.
fn sanitize(name: &str) -> String {
    let cleaned: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    let cleaned = cleaned.trim_matches('_');
    if cleaned.is_empty() {
        "Untitled".to_string()
    } else {
        cleaned.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::{plan, sanitize, write};
    use crate::document::Document;
    use crate::mesh::{Image, Material, primitives};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn names_are_safe_and_unique() {
        assert_eq!(sanitize("Body / Paint: v2"), "Body___Paint__v2");
        assert_eq!(sanitize("  (no material)  "), "no_material");
        assert_eq!(sanitize("???"), "Untitled");

        let mut mesh = primitives::cube();
        mesh.materials = ["Metal", "metal", "Metal?"]
            .into_iter()
            .map(|name| Material {
                name: name.into(),
                base_color: Image::filled(1, 1, [0, 0, 0, 255]),
            })
            .collect();
        let document = Document::new(mesh);
        let names: Vec<_> = plan("out".as_ref(), "My Model", &document)
            .iter()
            .map(|file| {
                file.path
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(
            names,
            [
                "My_Model_Metal_BaseColor.png",
                "My_Model_metal_BaseColor_2.png",
                "My_Model_Metal_BaseColor_3.png",
            ]
        );
    }

    #[test]
    fn exported_png_matches_the_texture() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory =
            std::env::temp_dir().join(format!("rustpaint-export-{}-{nonce}", std::process::id()));
        std::fs::create_dir(&directory).unwrap();

        let mut mesh = primitives::cube();
        mesh.materials[0].base_color.rgba[0..4].copy_from_slice(&[255, 0, 0, 128]);
        let document = Document::new(mesh);
        let files = plan(&directory, "cube", &document);
        write(&files, &document).unwrap();

        let saved = image::open(&files[0].path).unwrap().to_rgba8();
        let texture = &document.composite(0).unwrap();
        assert_eq!(saved.dimensions(), (texture.width, texture.height));
        assert_eq!(saved.as_raw(), &texture.rgba);
        // Only the PNG is left: no temporary files.
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
