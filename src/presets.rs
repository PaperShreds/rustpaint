//! Presets: saved stacks of fill layers, such as "Rust".
//!
//! A preset holds the settings of fill layers (values, second look, mask
//! generator, blend and opacity), never pixels, so it is tiny and fits any
//! model: its patterns are worked out again from each model's own shape.
//! Painted layers and painted masks aren't part of one.
//!
//! Presets you save are files in the program's data folder; the built-in ones
//! live in the code.

use crate::document::{BlendMode, FillMix, FillValues, Layer};
use crate::generator::{Axis, Generator, Pattern};
use crate::project;
use std::path::{Path, PathBuf};

/// Extension of preset files.
pub const EXTENSION: &str = "rustpreset";
/// Preset files are a few hundred bytes; anything much bigger isn't one.
const MAX_FILE_BYTES: u64 = 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
pub struct PresetLayer {
    pub name: String,
    pub opacity: f32,
    pub blend: BlendMode,
    pub fill: FillValues,
    /// The pattern that decides where the layer shows (as its mask).
    pub generator: Option<Generator>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Preset {
    pub name: String,
    /// Bottom layer first.
    pub layers: Vec<PresetLayer>,
}

impl Preset {
    /// A preset from the fill layers among `layers` (bottom first). A painted
    /// mask can't be kept, so that layer shows everywhere. `None` if none of
    /// them is a fill layer.
    pub fn from_layers<'a>(
        name: &str,
        layers: impl IntoIterator<Item = &'a Layer>,
    ) -> Option<Self> {
        let layers: Vec<PresetLayer> = layers
            .into_iter()
            .filter_map(|layer| {
                Some(PresetLayer {
                    name: layer.name.clone(),
                    opacity: layer.opacity,
                    blend: layer.blend,
                    fill: layer.fill?,
                    generator: layer.mask_generator.filter(|_| layer.mask.is_some()),
                })
            })
            .collect();
        (!layers.is_empty()).then(|| Self {
            name: name.trim().to_string(),
            layers,
        })
    }
}

/// The presets available: the built-in ones, then the ones you saved.
pub struct Library {
    built_in: Vec<Preset>,
    saved: Vec<Preset>,
    directory: Option<PathBuf>,
}

impl Library {
    /// Reads the saved presets from `directory` (when there is one).
    pub fn load(directory: Option<PathBuf>) -> Self {
        let saved = directory.as_deref().map(load_all).unwrap_or_default();
        Self {
            built_in: built_in(),
            saved,
            directory,
        }
    }

    pub fn built_in(&self) -> &[Preset] {
        &self.built_in
    }

    pub fn saved(&self) -> &[Preset] {
        &self.saved
    }

    /// Preset `index`, counting the built-in ones first.
    pub fn get(&self, index: usize) -> Option<&Preset> {
        match index.checked_sub(self.built_in.len()) {
            None => self.built_in.get(index),
            Some(saved) => self.saved.get(saved),
        }
    }

    /// Whether a saved preset has this name (saving would replace it).
    pub fn has_saved(&self, name: &str) -> bool {
        self.saved.iter().any(|preset| preset.name == name.trim())
    }

    /// Saves `preset`, replacing a saved one of the same name.
    pub fn save(&mut self, mut preset: Preset) -> Result<(), String> {
        preset.name = preset.name.trim().to_string();
        let directory = self
            .directory
            .as_deref()
            .ok_or("there is nowhere to keep presets on this system")?;
        std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
        let bytes = project::encode_preset(&preset)?;
        std::fs::write(file_for(directory, &preset.name), bytes).map_err(|e| e.to_string())?;
        self.saved = load_all(directory);
        Ok(())
    }

    /// Deletes the saved preset at `index` in `saved()`.
    pub fn delete(&mut self, index: usize) -> Result<(), String> {
        let (Some(directory), Some(preset)) = (self.directory.as_deref(), self.saved.get(index))
        else {
            return Err("no such preset".into());
        };
        std::fs::remove_file(file_for(directory, &preset.name)).map_err(|e| e.to_string())?;
        self.saved = load_all(directory);
        Ok(())
    }
}

/// The file a preset called `name` is kept in.
fn file_for(directory: &Path, name: &str) -> PathBuf {
    let stem: String = name
        .trim()
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, ' ' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    directory.join(format!("{stem}.{EXTENSION}"))
}

/// Every readable preset in `directory`, by name. Files that can't be read
/// are skipped, so one bad file never hides the others.
fn load_all(directory: &Path) -> Vec<Preset> {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return Vec::new();
    };
    let mut presets: Vec<Preset> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|e| e.to_str()) == Some(EXTENSION))
        .filter(|path| path.metadata().is_ok_and(|m| m.len() <= MAX_FILE_BYTES))
        .filter_map(|path| project::decode_preset(&std::fs::read(path).ok()?).ok())
        .collect();
    presets.sort_by_key(|preset| preset.name.to_lowercase());
    presets
}

fn layer(name: &str, fill: FillValues, generator: Generator) -> PresetLayer {
    PresetLayer {
        name: name.to_string(),
        opacity: 1.0,
        blend: BlendMode::Normal,
        fill,
        generator: Some(generator),
    }
}

/// Paint that is the same on every channel it fills.
fn paint(color: [u8; 3], roughness: f32, metallic: f32) -> FillValues {
    FillValues {
        color,
        roughness,
        metallic,
        channels: [true; 3],
        mix: None,
    }
}

fn built_in() -> Vec<Preset> {
    let patches = FillValues {
        mix: Some(FillMix {
            color: [86, 38, 14],
            roughness: 0.75,
            metallic: 0.0,
            generator: Generator {
                pattern: Pattern::Perlin,
                scale: 7.0,
                octaves: 5,
                seed: 1,
                low: 0.35,
                high: 0.65,
                ..Generator::default()
            },
        }),
        ..paint([156, 78, 34], 0.9, 0.0)
    };
    vec![
        Preset {
            name: "Rust".to_string(),
            layers: vec![
                layer(
                    "Rust patches",
                    patches,
                    Generator {
                        scale: 2.5,
                        seed: 7,
                        low: 0.5,
                        high: 0.62,
                        ..Generator::default()
                    },
                ),
                layer(
                    "Rust in creases",
                    paint([98, 46, 20], 0.85, 0.0),
                    Generator {
                        pattern: Pattern::Cavity,
                        edge_width: 0.06,
                        breakup: 0.6,
                        scale: 10.0,
                        octaves: 5,
                        low: 0.1,
                        high: 0.7,
                        ..Generator::default()
                    },
                ),
            ],
        },
        Preset {
            name: "Dirt in creases".to_string(),
            layers: vec![PresetLayer {
                opacity: 0.9,
                ..layer(
                    "Dirt",
                    paint([28, 20, 14], 0.95, 0.0),
                    Generator {
                        pattern: Pattern::Cavity,
                        edge_width: 0.08,
                        breakup: 0.5,
                        scale: 12.0,
                        octaves: 5,
                        low: 0.05,
                        high: 0.8,
                        ..Generator::default()
                    },
                )
            }],
        },
        Preset {
            name: "Worn edges".to_string(),
            layers: vec![layer(
                "Bare metal edges",
                paint([190, 190, 195], 0.35, 1.0),
                Generator {
                    pattern: Pattern::EdgeWear,
                    edge_width: 0.02,
                    breakup: 0.5,
                    scale: 16.0,
                    octaves: 5,
                    low: 0.2,
                    high: 0.6,
                    ..Generator::default()
                },
            )],
        },
        Preset {
            name: "Dust from below".to_string(),
            layers: vec![PresetLayer {
                opacity: 0.8,
                ..layer(
                    "Dust",
                    paint([150, 138, 120], 1.0, 0.0),
                    Generator {
                        pattern: Pattern::Gradient,
                        axis: Axis::Y,
                        invert: true,
                        low: 0.0,
                        high: 0.5,
                        ..Generator::default()
                    },
                )
            }],
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Document;
    use crate::mesh::primitives;

    fn directory(name: &str) -> PathBuf {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        std::env::temp_dir().join(format!("rustpaint-{name}-{}-{nonce}", std::process::id()))
    }

    #[test]
    fn built_in_presets_make_layers_with_patterns() {
        let mut document = Document::new(primitives::cube());
        let library = Library::load(None);
        assert!(library.built_in().len() >= 3);
        for preset in library.built_in() {
            let layers = document.new_preset_layers(preset);
            assert_eq!(layers.len(), preset.layers.len());
            for layer in &layers {
                assert!(layer.fill.is_some() && layer.mask.is_some());
                assert!(layer.mask_generator.is_some());
                assert!(layer.textures[0][crate::document::Plane::Mask.index()].is_some());
            }
        }
    }

    #[test]
    fn saved_presets_survive_a_restart_and_can_be_deleted() {
        let directory = directory("presets");
        let mut library = Library::load(Some(directory.clone()));
        let rust = library.built_in()[0].clone();
        assert!(!library.has_saved("My rust"));
        library
            .save(Preset {
                name: " My rust ".to_string(),
                ..rust.clone()
            })
            .unwrap();
        let reopened = Library::load(Some(directory.clone()));
        assert_eq!(reopened.saved().len(), 1);
        let saved = reopened.saved()[0].clone();
        assert_eq!(saved.name, "My rust");
        assert_eq!(saved.layers, rust.layers);
        let first_saved = reopened.built_in().len();
        assert_eq!(reopened.get(first_saved), Some(&saved));
        assert_eq!(reopened.get(0), Some(&reopened.built_in()[0]));

        // Saving under a name again replaces it; a broken file is skipped.
        let mut library = reopened;
        library.save(saved.clone()).unwrap();
        assert_eq!(library.saved().len(), 1);
        std::fs::write(directory.join("broken.rustpreset"), b"nonsense").unwrap();
        let mut library = Library::load(Some(directory.clone()));
        assert_eq!(library.saved().len(), 1);
        library.delete(0).unwrap();
        assert!(library.saved().is_empty());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn only_fill_layers_go_into_a_preset() {
        let mut document = Document::new(primitives::cube());
        let fill = document.new_fill_layer(FillValues::default());
        let painted = document.new_layer();
        assert!(Preset::from_layers("None", [&painted]).is_none());
        let preset = Preset::from_layers("Both", [&painted, &fill]).unwrap();
        assert_eq!(preset.layers.len(), 1);
        assert_eq!(preset.layers[0].fill, FillValues::default());
        assert_eq!(preset.layers[0].generator, None, "no mask");
    }

    #[test]
    fn a_preset_is_added_and_undone_as_one_step() {
        use crate::document::Change;
        use crate::painting::history::History;
        let mut document = Document::new(primitives::cube());
        let preset = Library::load(None).built_in()[0].clone();
        let mut history = History::default();
        let mut changes = Vec::new();
        for (offset, layer) in document.new_preset_layers(&preset).into_iter().enumerate() {
            let mut change = Change::Layer {
                index: 2 + offset,
                stored: Some(layer),
            };
            document.apply(&mut change).unwrap();
            changes.push(change);
        }
        history.push("Add preset", changes);
        let names = |document: &Document| {
            // Layer 1 is the paint layer every document starts with.
            document.layers[2..]
                .iter()
                .map(|layer| layer.name.clone())
                .collect::<Vec<_>>()
        };
        let added: Vec<_> = preset.layers.iter().map(|l| l.name.clone()).collect();
        assert_eq!(names(&document), added);
        history.undo(&mut document);
        assert!(names(&document).is_empty());
        history.redo(&mut document);
        assert_eq!(names(&document), added);
    }

    #[test]
    fn names_make_safe_file_names() {
        let file = file_for(Path::new("/presets"), "../a/b: c");
        assert_eq!(file, Path::new("/presets/___a_b_ c.rustpreset"));
        assert_eq!(file.parent(), Some(Path::new("/presets")));
    }
}
