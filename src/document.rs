//! The open project: the model plus its paint layers.
//!
//! Every material slot of the mesh has its own UV layout, so a layer holds
//! one texture per slot it paints on. Layer 0 is the model's own textures
//! ("Base"); it can be hidden but not painted on, moved or deleted.
//!
//! The viewport blends the visible layers on the GPU. `composite` does the same
//! on the CPU, for export.

pub use crate::mesh::{Channel, Encoding};
use crate::mesh::{Image, ImageRegion, Mesh};

/// A transparent pixel, the starting content of a paint layer.
pub const CLEAR: [u8; 4] = [0, 0, 0, 0];

/// How a layer combines with the layers below it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendMode {
    #[default]
    Normal,
    Multiply,
    Screen,
    Overlay,
    Darken,
    Lighten,
    Add,
}

impl BlendMode {
    pub const ALL: [BlendMode; 7] = [
        BlendMode::Normal,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Overlay,
        BlendMode::Darken,
        BlendMode::Lighten,
        BlendMode::Add,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BlendMode::Normal => "Normal",
            BlendMode::Multiply => "Multiply",
            BlendMode::Screen => "Screen",
            BlendMode::Overlay => "Overlay",
            BlendMode::Darken => "Darken",
            BlendMode::Lighten => "Lighten",
            BlendMode::Add => "Add",
        }
    }

    /// Stable number for files and shaders.
    pub fn index(self) -> u8 {
        Self::ALL.iter().position(|mode| *mode == self).unwrap_or(0) as u8
    }

    pub fn from_index(index: u8) -> Option<Self> {
        Self::ALL.get(index as usize).copied()
    }

    /// Blends one channel of `top` onto `below` (straight values in 0..1).
    /// Must match `blend_channel` in `composite.wgsl`.
    pub fn apply(self, below: f32, top: f32) -> f32 {
        let value = match self {
            BlendMode::Normal => top,
            BlendMode::Multiply => below * top,
            BlendMode::Screen => below + top - below * top,
            BlendMode::Overlay => {
                if below <= 0.5 {
                    2.0 * below * top
                } else {
                    1.0 - 2.0 * (1.0 - below) * (1.0 - top)
                }
            }
            BlendMode::Darken => below.min(top),
            BlendMode::Lighten => below.max(top),
            BlendMode::Add => below + top,
        };
        value.clamp(0.0, 1.0)
    }
}

pub struct Layer {
    /// Stable identity, so undo steps and UI state survive reordering.
    pub id: u64,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    /// Which material slots the layer paints on (and shows on), by slot index.
    pub targets: Vec<bool>,
    /// Paint per material slot and channel; `None` until first painted.
    /// Always empty for the base layer, whose pixels are the mesh's textures.
    pub textures: Vec<[Option<Image>; Channel::COUNT]>,
}

impl Layer {
    pub fn is_base(&self) -> bool {
        self.id == BASE_LAYER_ID
    }

    pub fn props(&self) -> LayerProps {
        LayerProps {
            name: self.name.clone(),
            visible: self.visible,
            opacity: self.opacity,
            blend: self.blend,
            targets: self.targets.clone(),
        }
    }

    fn set_props(&mut self, props: LayerProps) {
        self.name = props.name;
        self.visible = props.visible;
        self.opacity = props.opacity;
        self.blend = props.blend;
        self.targets = props.targets;
    }

    fn bytes(&self) -> usize {
        self.textures
            .iter()
            .flatten()
            .flatten()
            .map(|image| image.rgba.len())
            .sum()
    }
}

pub const BASE_LAYER_ID: u64 = 0;

/// A layer as it contributes to one slot's composite.
pub struct VisibleLayer<'a> {
    pub id: u64,
    pub image: &'a Image,
    pub opacity: f32,
    pub blend: BlendMode,
}

/// Settings of a layer that can be changed (and undone) as one.
#[derive(Clone, PartialEq)]
pub struct LayerProps {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    pub targets: Vec<bool>,
}

/// Pixels of one square tile of a layer texture, tightly packed RGBA.
pub struct Tile {
    pub region: ImageRegion,
    pub pixels: Vec<u8>,
}

/// One undoable change. Applying it performs the change and turns it into its
/// own inverse, so the same value serves undo and redo.
pub enum Change {
    /// Swap pixels of a layer texture with the stored ones.
    Pixels {
        layer: u64,
        material: usize,
        channel: Channel,
        tiles: Vec<Tile>,
    },
    /// With `stored` set: insert it at `index`. Without: remove the layer at
    /// `index` and store it.
    Layer { index: usize, stored: Option<Layer> },
    /// Move the layer at `from` to `to`.
    Move { from: usize, to: usize },
    /// Swap a layer's settings with the stored ones.
    Props { layer: u64, props: LayerProps },
}

impl Change {
    pub fn bytes(&self) -> usize {
        match self {
            Change::Pixels { tiles, .. } => tiles.iter().map(|tile| tile.pixels.len()).sum(),
            Change::Layer {
                stored: Some(layer),
                ..
            } => layer.bytes(),
            _ => 0,
        }
    }
}

/// What a change touched, so the GPU copy can be brought up to date.
pub enum Dirty {
    /// Pixels of one layer texture changed.
    Pixels {
        layer: u64,
        material: usize,
        channel: Channel,
        region: ImageRegion,
    },
    /// Layers were added, removed, reordered, or their settings changed.
    Layers,
}

pub struct Document {
    pub mesh: Mesh,
    /// Bottom to top; `layers[0]` is the base layer.
    pub layers: Vec<Layer>,
    next_layer_id: u64,
}

impl Document {
    /// A document for a freshly imported model: its textures as the base
    /// layer, plus an empty layer to paint on.
    pub fn new(mesh: Mesh) -> Self {
        let mut document = Self::with_layers(mesh, Vec::new());
        document.insert_new_layer(1);
        document
    }

    /// `layers` excludes the base layer, which is created here.
    pub fn with_layers(mut mesh: Mesh, layers: Vec<Layer>) -> Self {
        // Every texture of a slot must be the same size, whatever its source.
        for material in &mut mesh.materials {
            material.conform_channels();
        }
        let slots = mesh.materials.len();
        let base = Layer {
            id: BASE_LAYER_ID,
            name: "Base (from model)".to_string(),
            visible: true,
            opacity: 1.0,
            blend: BlendMode::Normal,
            targets: vec![true; slots],
            textures: Vec::new(),
        };
        let next_layer_id = layers.iter().map(|layer| layer.id).max().unwrap_or(0) + 1;
        Self {
            mesh,
            layers: std::iter::once(base).chain(layers).collect(),
            next_layer_id,
        }
    }

    pub fn set_base_props(&mut self, props: LayerProps) {
        self.layers[0].set_props(props);
    }

    pub fn layer_index(&self, id: u64) -> Option<usize> {
        self.layers.iter().position(|layer| layer.id == id)
    }

    /// Size of each slot's textures (the same for every layer).
    pub fn slot_sizes(&self) -> Vec<(u32, u32)> {
        self.mesh
            .materials
            .iter()
            .map(|material| (material.base_color.width, material.base_color.height))
            .collect()
    }

    /// The pixels of a layer on a slot: the mesh's texture for the base
    /// layer, otherwise the layer's paint (if any).
    pub fn layer_image<'a>(
        &'a self,
        layer: &'a Layer,
        material: usize,
        channel: Channel,
    ) -> Option<&'a Image> {
        if layer.is_base() {
            self.mesh
                .materials
                .get(material)
                .map(|m| m.channel(channel))
        } else {
            layer.textures.get(material)?[channel.index()].as_ref()
        }
    }

    /// What shows on `material`, bottom to top: each visible layer that targets
    /// the slot and has pixels there.
    pub fn visible_layers(&self, material: usize, channel: Channel) -> Vec<VisibleLayer<'_>> {
        let size = self.slot_sizes().get(material).copied();
        self.layers
            .iter()
            .filter(|layer| layer.visible && layer.targets.get(material) == Some(&true))
            .filter_map(|layer| {
                let image = self.layer_image(layer, material, channel)?;
                (Some((image.width, image.height)) == size).then_some(VisibleLayer {
                    id: layer.id,
                    image,
                    opacity: layer.opacity.clamp(0.0, 1.0),
                    blend: layer.blend,
                })
            })
            .collect()
    }

    /// A new, empty layer that paints on every slot. Not yet in the stack.
    pub fn new_layer(&mut self) -> Layer {
        let id = self.next_layer_id;
        self.next_layer_id += 1;
        let slots = self.mesh.materials.len();
        Layer {
            id,
            name: format!("Layer {id}"),
            visible: true,
            opacity: 1.0,
            blend: BlendMode::Normal,
            targets: vec![true; slots],
            textures: (0..slots).map(|_| Default::default()).collect(),
        }
    }

    /// Adds a new layer at `index` and returns the change that removes it again.
    pub fn insert_new_layer(&mut self, index: usize) -> Change {
        let layer = self.new_layer();
        let mut change = Change::Layer {
            index: index.clamp(1, self.layers.len()),
            stored: Some(layer),
        };
        self.apply(&mut change);
        change
    }

    /// Applies `change` (making it its own inverse). Returns what it touched,
    /// or `None` if the change doesn't fit this document.
    pub fn apply(&mut self, change: &mut Change) -> Option<Dirty> {
        match change {
            Change::Pixels {
                layer,
                material,
                channel,
                tiles,
            } => {
                let index = self.layer_index(*layer)?;
                let image =
                    self.layers[index].textures.get_mut(*material)?[channel.index()].as_mut()?;
                let fits = tiles.iter().all(|tile| {
                    tile.region.x + tile.region.width <= image.width
                        && tile.region.y + tile.region.height <= image.height
                        && tile.pixels.len()
                            == (tile.region.width * tile.region.height * 4) as usize
                });
                if !fits {
                    return None;
                }
                for tile in tiles.iter_mut() {
                    swap_tile(image, tile);
                }
                let region = tiles
                    .iter()
                    .map(|tile| tile.region)
                    .reduce(ImageRegion::union)?;
                Some(Dirty::Pixels {
                    layer: *layer,
                    material: *material,
                    channel: *channel,
                    region,
                })
            }
            Change::Layer { index, stored } => {
                match stored.take() {
                    Some(layer) => {
                        if *index == 0 || *index > self.layers.len() {
                            *stored = Some(layer);
                            return None;
                        }
                        self.layers.insert(*index, layer);
                    }
                    None => {
                        if *index == 0 || *index >= self.layers.len() {
                            return None;
                        }
                        *stored = Some(self.layers.remove(*index));
                    }
                }
                Some(Dirty::Layers)
            }
            Change::Move { from, to } => {
                let len = self.layers.len();
                if *from == 0 || *to == 0 || *from >= len || *to >= len {
                    return None;
                }
                let layer = self.layers.remove(*from);
                self.layers.insert(*to, layer);
                std::mem::swap(from, to);
                Some(Dirty::Layers)
            }
            Change::Props { layer, props } => {
                let index = self.layer_index(*layer)?;
                if props.targets.len() != self.mesh.materials.len() {
                    return None;
                }
                let previous = self.layers[index].props();
                self.layers[index].set_props(std::mem::replace(props, previous));
                Some(Dirty::Layers)
            }
        }
    }

    /// The visible layers of one channel of `material`, blended into one image (on the CPU).
    pub fn composite(&self, material: usize, channel: Channel) -> Option<Image> {
        let (width, height) = *self.slot_sizes().get(material)?;
        let sources = self.visible_layers(material, channel);
        let mut composite = Image::filled(width, height, CLEAR);
        for (offset, pixel) in composite.rgba.chunks_exact_mut(4).enumerate() {
            pixel.copy_from_slice(&composite_pixel(&sources, offset * 4, channel.encoding()));
        }
        Some(composite)
    }
}

/// Blends one pixel through the layer stack. Same maths as `fs_layer` in
/// `composite.wgsl`: premultiplied, with the W3C formula for blend modes.
fn composite_pixel(sources: &[VisibleLayer], offset: usize, encoding: Encoding) -> [u8; 4] {
    let contributes =
        |layer: &&VisibleLayer| layer.image.rgba[offset + 3] != 0 && layer.opacity > 0.0;
    // A fully opaque Normal layer hides everything below it.
    let start = sources
        .iter()
        .rposition(|layer| {
            layer.blend == BlendMode::Normal
                && layer.opacity == 1.0
                && layer.image.rgba[offset + 3] == 255
        })
        .unwrap_or(0);
    let mut contributors = sources[start..].iter().filter(contributes);
    let Some(first) = contributors.next() else {
        return CLEAR;
    };
    if contributors.next().is_none() && first.opacity == 1.0 && first.blend == BlendMode::Normal {
        // A single Normal layer shows through unchanged.
        let mut pixel = [0; 4];
        pixel.copy_from_slice(&first.image.rgba[offset..offset + 4]);
        return pixel;
    }

    let mut color = [0.0f32; 3]; // premultiplied
    let mut coverage = 0.0f32;
    for layer in &sources[start..] {
        let alpha = layer.image.rgba[offset + 3] as f32 / 255.0 * layer.opacity;
        if alpha == 0.0 {
            continue;
        }
        for (channel, value) in color.iter_mut().enumerate() {
            let top = encoding.decode(layer.image.rgba[offset + channel]);
            let below = if coverage > 0.0 {
                *value / coverage
            } else {
                0.0
            };
            let mixed = layer.blend.apply(below, top);
            *value =
                top * alpha * (1.0 - coverage) + *value * (1.0 - alpha) + alpha * coverage * mixed;
        }
        coverage = alpha + coverage * (1.0 - alpha);
    }
    if coverage <= 0.0 {
        return CLEAR;
    }
    [
        encoding.encode(color[0] / coverage),
        encoding.encode(color[1] / coverage),
        encoding.encode(color[2] / coverage),
        (coverage * 255.0).round() as u8,
    ]
}

fn swap_tile(image: &mut Image, tile: &mut Tile) {
    let region = tile.region;
    let row_bytes = (region.width * 4) as usize;
    for row in 0..region.height {
        let start = (((region.y + row) * image.width + region.x) * 4) as usize;
        let stored = row as usize * row_bytes;
        image.rgba[start..start + row_bytes]
            .swap_with_slice(&mut tile.pixels[stored..stored + row_bytes]);
    }
}

#[cfg(test)]
mod tests {
    use super::{BlendMode, CLEAR, Change, Document};
    use crate::mesh::Channel;
    use crate::mesh::{Image, ImageRegion, Material, primitives};

    /// A cube with two 4x4 slots: red and blue.
    fn document() -> Document {
        let mut mesh = primitives::cube();
        mesh.materials = [("Red", [255, 0, 0, 255]), ("Blue", [0, 0, 255, 255])]
            .into_iter()
            .map(|(name, pixel)| Material::new(name, Image::filled(4, 4, pixel)))
            .collect();
        Document::new(mesh)
    }

    fn paint(document: &mut Document, layer: usize, material: usize, pixel: [u8; 4]) {
        let (w, h) = document.slot_sizes()[material];
        let image = document.layers[layer].textures[material][0]
            .get_or_insert_with(|| Image::filled(w, h, CLEAR));
        image.rgba[0..4].copy_from_slice(&pixel);
    }

    #[test]
    fn new_document_shows_the_model_textures() {
        let document = document();
        assert_eq!(document.layers.len(), 2);
        assert!(document.layers[0].is_base());
        assert_eq!(
            &document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            &[255, 0, 0, 255]
        );
        assert_eq!(
            &document.composite(1, Channel::BaseColor).unwrap().rgba[0..4],
            &[0, 0, 255, 255]
        );
    }

    #[test]
    fn layers_blend_by_alpha_and_opacity() {
        let mut document = document();
        paint(&mut document, 1, 0, [0, 255, 0, 255]);
        assert_eq!(
            &document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            &[0, 255, 0, 255]
        );
        // Pixels the layer didn't paint still show the base.
        assert_eq!(
            &document.composite(0, Channel::BaseColor).unwrap().rgba[4..8],
            &[255, 0, 0, 255]
        );

        document.layers[1].opacity = 0.5;
        let composite = document.composite(0, Channel::BaseColor).unwrap();
        let pixel = &composite.rgba[0..4];
        // Half green over red, mixed in linear light: both channels well above 50%.
        assert!(pixel[0] > 180 && pixel[1] > 180 && pixel[2] == 0 && pixel[3] == 255);
    }

    #[test]
    fn hidden_layers_and_untargeted_slots_do_not_show() {
        let mut document = document();
        paint(&mut document, 1, 0, [0, 255, 0, 255]);

        document.layers[1].visible = false;
        assert_eq!(
            &document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            &[255, 0, 0, 255]
        );

        document.layers[1].visible = true;
        document.layers[1].targets[0] = false;
        assert_eq!(
            &document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            &[255, 0, 0, 255]
        );
        // The paint is kept and comes back when the slot is ticked again.
        document.layers[1].targets[0] = true;
        assert_eq!(
            &document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            &[0, 255, 0, 255]
        );
    }

    #[test]
    fn layer_changes_undo_by_applying_again() {
        let mut document = document();
        paint(&mut document, 1, 0, [0, 255, 0, 255]);

        // Remove, then undo the removal.
        let mut remove = Change::Layer {
            index: 1,
            stored: None,
        };
        document.apply(&mut remove).unwrap();
        assert_eq!(document.layers.len(), 1);
        assert_eq!(
            &document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            &[255, 0, 0, 255]
        );
        document.apply(&mut remove).unwrap();
        assert_eq!(document.layers.len(), 2);
        assert_eq!(
            &document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            &[0, 255, 0, 255]
        );

        // Reorder two paint layers and back.
        let mut add = document.insert_new_layer(2);
        let top = document.layers[2].id;
        let mut reorder = Change::Move { from: 2, to: 1 };
        document.apply(&mut reorder).unwrap();
        assert_eq!(document.layers[1].id, top);
        document.apply(&mut reorder).unwrap();
        assert_eq!(document.layers[2].id, top);
        document.apply(&mut add).unwrap();
        assert_eq!(document.layers.len(), 2);

        // The base layer can't be removed or moved.
        let mut remove_base = Change::Layer {
            index: 0,
            stored: None,
        };
        assert!(document.apply(&mut remove_base).is_none());
        assert!(
            document
                .apply(&mut Change::Move { from: 0, to: 1 })
                .is_none()
        );
    }

    #[test]
    fn pixel_changes_swap_and_report_their_region() {
        let mut document = document();
        paint(&mut document, 1, 0, CLEAR);
        let id = document.layers[1].id;
        let region = ImageRegion::pixel(1, 1);
        let mut change = Change::Pixels {
            layer: id,
            material: 0,
            channel: Channel::BaseColor,
            tiles: vec![super::Tile {
                region,
                pixels: vec![0, 255, 0, 255],
            }],
        };
        document.apply(&mut change).unwrap();
        assert_eq!(
            &document.composite(0, Channel::BaseColor).unwrap().rgba[20..24],
            &[0, 255, 0, 255]
        );
        document.apply(&mut change).unwrap();
        assert_eq!(
            &document.composite(0, Channel::BaseColor).unwrap().rgba[20..24],
            &[255, 0, 0, 255]
        );
    }

    #[test]
    fn srgb_round_trip_is_exact() {
        for value in 0..=255u8 {
            let linear = crate::mesh::srgb_to_linear(value);
            assert_eq!(crate::mesh::linear_to_srgb(linear), value);
        }
    }

    #[test]
    fn blend_modes_combine_with_the_layers_below() {
        let mut document = document();
        // The base is pure red; paint mid grey on the layer above.
        paint(&mut document, 1, 0, [128, 128, 128, 255]);
        let grey = crate::mesh::srgb_to_linear(128);
        let pixel = |document: &Document| {
            let composite = document.composite(0, Channel::BaseColor).unwrap();
            [0, 1, 2].map(|c| crate::mesh::srgb_to_linear(composite.rgba[c]))
        };
        for mode in BlendMode::ALL {
            document.layers[1].blend = mode;
            let expected = [
                mode.apply(1.0, grey),
                mode.apply(0.0, grey),
                mode.apply(0.0, grey),
            ];
            let actual = pixel(&document);
            for (a, e) in actual.iter().zip(expected) {
                assert!(
                    (a - e).abs() < 0.01,
                    "{}: {actual:?} vs {expected:?}",
                    mode.name()
                );
            }
        }
        // Spot checks against the definitions.
        document.layers[1].blend = BlendMode::Multiply;
        assert!(
            pixel(&document)[1] < 0.01,
            "multiply onto black stays black"
        );
        document.layers[1].blend = BlendMode::Screen;
        assert!(pixel(&document)[0] > 0.99, "screen onto white stays white");
    }
}
