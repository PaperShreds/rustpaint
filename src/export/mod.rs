//! Exporting painted textures as image files, or the model with its
//! textures as a `.glb` (see `glb`).

pub mod glb;

use crate::document::Document;
use crate::files;
use crate::mesh::{Channel, Encoding, Image};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// One texture file to write.
pub struct ExportFile {
    pub path: PathBuf,
    /// The material slot whose combined layers go into this file.
    pub material: usize,
    pub texture: Texture,
}

/// What an exported file holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Texture {
    Channel(Channel),
    /// The bumps of the height channel as a tangent-space normal map
    /// (OpenGL convention, as glTF, Blender and Godot expect).
    Normal,
}

impl Texture {
    pub const ALL: [Texture; Channel::COUNT + 1] = [
        Texture::Channel(Channel::BaseColor),
        Texture::Channel(Channel::Roughness),
        Texture::Channel(Channel::Metallic),
        Texture::Channel(Channel::Height),
        Texture::Normal,
    ];

    fn file_suffix(self) -> &'static str {
        match self {
            Texture::Channel(channel) => channel.file_suffix(),
            Texture::Normal => "Normal",
        }
    }
}

/// The PNG files exporting `document` into `directory` would write: one per
/// material slot and channel, plus a normal map, with its visible layers
/// combined, named `<model>_<material>_<Channel>.png` (e.g. `_BaseColor`,
/// `_Roughness`, `_Normal`).
pub fn plan(directory: &Path, model_name: &str, document: &Document) -> Vec<ExportFile> {
    let model = sanitize(model_name);
    let mut used = HashSet::new();
    document
        .mesh
        .materials
        .iter()
        .enumerate()
        .flat_map(|(index, material)| {
            let base = format!("{model}_{}", sanitize(&material.name));
            // Two materials can sanitize to the same name; keep files distinct.
            let mut stem = base.clone();
            let mut suffix = 2;
            while !used.insert(stem.to_lowercase()) {
                stem = format!("{base}_{suffix}");
                suffix += 1;
            }
            Texture::ALL.map(|texture| ExportFile {
                path: directory.join(format!("{stem}_{}.png", texture.file_suffix())),
                material: index,
                texture,
            })
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
                    let channel = match file.texture {
                        Texture::Channel(channel) => channel,
                        Texture::Normal => Channel::Height,
                    };
                    let image = document
                        .composite(file.material, channel)
                        .ok_or("material slot not found")?;
                    match file.texture {
                        Texture::Channel(channel) => {
                            write_png(&file.path, &image, Some(channel.encoding()))
                        }
                        Texture::Normal => {
                            let normals = document
                                .normal_map(file.material, &image)
                                .ok_or("material slot not found")?;
                            write_png(&file.path, &normals, None)
                        }
                    }
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

/// Colour is written as RGBA; data channels as greyscale, as engines
/// expect; a normal map (no encoding) as RGB.
fn write_png(path: &Path, image: &Image, encoding: Option<Encoding>) -> Result<(), String> {
    let buffer: image::DynamicImage = match encoding {
        Some(Encoding::Srgb) => {
            image::RgbaImage::from_raw(image.width, image.height, image.rgba.clone())
                .map(image::DynamicImage::ImageRgba8)
        }
        Some(Encoding::Linear) => {
            let values = image.rgba.chunks_exact(4).map(|pixel| pixel[0]).collect();
            image::GrayImage::from_raw(image.width, image.height, values)
                .map(image::DynamicImage::ImageLuma8)
        }
        None => {
            let values = image.rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]);
            image::RgbImage::from_raw(image.width, image.height, values.collect())
                .map(image::DynamicImage::ImageRgb8)
        }
    }
    .ok_or("texture data does not match its size")?;
    let mut png = Vec::new();
    buffer
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    files::write_atomically(path, &png)
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
    use super::{Texture, plan, sanitize, write};
    use crate::document::Document;
    use crate::mesh::Channel;
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
            .map(|name| Material::new(name, Image::filled(1, 1, [0, 0, 0, 255])))
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
                "My_Model_Metal_Roughness.png",
                "My_Model_Metal_Metallic.png",
                "My_Model_Metal_Height.png",
                "My_Model_Metal_Normal.png",
                "My_Model_metal_2_BaseColor.png",
                "My_Model_metal_2_Roughness.png",
                "My_Model_metal_2_Metallic.png",
                "My_Model_metal_2_Height.png",
                "My_Model_metal_2_Normal.png",
                "My_Model_Metal_3_BaseColor.png",
                "My_Model_Metal_3_Roughness.png",
                "My_Model_Metal_3_Metallic.png",
                "My_Model_Metal_3_Height.png",
                "My_Model_Metal_3_Normal.png",
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
        let texture = &document.composite(0, Channel::BaseColor).unwrap();
        assert_eq!(saved.dimensions(), (texture.width, texture.height));
        assert_eq!(saved.as_raw(), &texture.rgba);

        // Data channels are greyscale, holding the raw values.
        assert_eq!(files[1].texture, Texture::Channel(Channel::Roughness));
        let roughness = image::open(&files[1].path).unwrap();
        assert!(matches!(roughness, image::DynamicImage::ImageLuma8(_)));
        let expected = crate::mesh::Encoding::Linear.encode(0.5);
        assert!(
            roughness
                .to_luma8()
                .pixels()
                .all(|pixel| pixel.0[0] == expected)
        );

        // Nothing raised: the normal map faces straight out everywhere.
        assert_eq!(files[4].texture, Texture::Normal);
        let normals = image::open(&files[4].path).unwrap();
        assert!(matches!(normals, image::DynamicImage::ImageRgb8(_)));
        let odd: std::collections::HashSet<[u8; 3]> =
            normals.to_rgb8().pixels().map(|pixel| pixel.0).collect();
        assert_eq!(odd, [[128, 128, 255]].into());

        // Only the PNGs are left: no temporary files.
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 5);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
