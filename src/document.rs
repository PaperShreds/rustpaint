//! The open project: the model plus its paint layers.
//!
//! Every material slot of the mesh has its own UV layout, so a layer holds
//! one texture per slot it paints on. Layer 0 is the model's own textures
//! ("Base"); it can be hidden but not painted on, moved or deleted.
//!
//! A layer can have a mask: a greyscale image per slot that hides (black) or
//! shows (white) the layer's paint, in all channels. The images live next to
//! the channels as the layer's `Plane::Mask`.
//!
//! A layer can also be procedural. A fill layer's images are one value per
//! channel, everywhere; a mask can come from a `Generator`. Their pixels are
//! made from the settings (`refresh_layer`), so undo only stores the settings.
//!
//! The viewport blends the visible layers on the GPU. `composite` does the same
//! on the CPU, for export.

use crate::generator::Generator;
use crate::generator::bake::Bake;
pub use crate::mesh::{Channel, Encoding};
use crate::mesh::{Image, ImageRegion, Mesh};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A transparent pixel, the starting content of a paint layer.
pub const CLEAR: [u8; 4] = [0, 0, 0, 0];

/// One of a layer's images on a slot: a material channel, or its mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Plane {
    Channel(Channel),
    Mask,
}

impl Plane {
    pub const COUNT: usize = Channel::COUNT + 1;
    pub const ALL: [Plane; Plane::COUNT] = [
        Plane::Channel(Channel::BaseColor),
        Plane::Channel(Channel::Roughness),
        Plane::Channel(Channel::Metallic),
        Plane::Mask,
    ];

    /// Position in a layer's per-slot image array (channels first).
    pub fn index(self) -> usize {
        match self {
            Plane::Channel(channel) => channel.index(),
            Plane::Mask => Channel::COUNT,
        }
    }

    /// Masks hold a plain value, like the data channels.
    pub fn encoding(self) -> Encoding {
        match self {
            Plane::Channel(channel) => channel.encoding(),
            Plane::Mask => Encoding::Linear,
        }
    }
}

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

/// What a fill layer puts on the model: one value per channel, everywhere.
/// Where it shows is up to the layer's mask.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillValues {
    /// Base colour, sRGB-encoded.
    pub color: [u8; 3],
    pub roughness: f32,
    pub metallic: f32,
    /// Which channels the layer fills (the others are left to the layers below).
    pub channels: [bool; Channel::COUNT],
    /// A second set of values and the pattern that mixes them in.
    pub mix: Option<FillMix>,
}

/// A fill layer's second look: where its pattern is 1 the layer has these
/// values, where it is 0 the fill's own, and in between a blend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FillMix {
    /// Base colour, sRGB-encoded.
    pub color: [u8; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub generator: Generator,
}

impl Default for FillMix {
    fn default() -> Self {
        Self {
            color: [90, 90, 90],
            roughness: 0.5,
            metallic: 0.0,
            generator: Generator::default(),
        }
    }
}

impl Default for FillValues {
    fn default() -> Self {
        Self {
            color: [200, 200, 200],
            roughness: 0.5,
            metallic: 0.0,
            channels: [true, false, false],
            mix: None,
        }
    }
}

impl FillValues {
    /// The pixel a fill layer has everywhere on `channel`.
    pub fn pixel(&self, channel: Channel) -> [u8; 4] {
        let data = |value: f32| {
            let byte = Encoding::Linear.encode(value);
            [byte, byte, byte, 255]
        };
        match channel {
            Channel::BaseColor => [self.color[0], self.color[1], self.color[2], 255],
            Channel::Roughness => data(self.roughness),
            Channel::Metallic => data(self.metallic),
        }
    }

    /// The pixel for every pattern value from 0 to 255: the fill's own
    /// values at 0, the mix's at 255. Colours blend in linear light.
    pub fn ramp(&self, channel: Channel) -> [[u8; 4]; 256] {
        let from = self.pixel(channel);
        let Some(mix) = self.mix else {
            return [from; 256];
        };
        let to = FillValues {
            color: mix.color,
            roughness: mix.roughness,
            metallic: mix.metallic,
            ..*self
        }
        .pixel(channel);
        let encoding = match channel {
            Channel::BaseColor => Encoding::Srgb,
            _ => Encoding::Linear,
        };
        std::array::from_fn(|step| {
            let t = step as f32 / 255.0;
            let blend = |c: usize| {
                let (a, b) = (encoding.decode(from[c]), encoding.decode(to[c]));
                encoding.encode(a + (b - a) * t)
            };
            [blend(0), blend(1), blend(2), 255]
        })
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
    /// Images per material slot and plane (`Plane::index`); `None` until
    /// first painted. Always empty for the base layer, whose pixels are the
    /// mesh's textures.
    pub textures: Vec<[Option<Image>; Plane::COUNT]>,
    /// If the layer has a mask: its value where it hasn't been painted (255
    /// shows the layer there, 0 hides it). Painted masks are `Plane::Mask`
    /// images: opaque, the value in every colour channel.
    pub mask: Option<u8>,
    /// A fill layer: its channel images are made from these values, not painted.
    pub fill: Option<FillValues>,
    /// If set, the mask's images are made by this generator, not painted.
    /// Needs a mask.
    pub mask_generator: Option<Generator>,
}

impl Layer {
    pub fn is_base(&self) -> bool {
        self.id == BASE_LAYER_ID
    }

    /// Whether some of the layer's images are made from settings.
    pub fn is_procedural(&self) -> bool {
        self.fill.is_some() || self.mask_generator.is_some()
    }

    pub fn props(&self) -> LayerProps {
        LayerProps {
            name: self.name.clone(),
            visible: self.visible,
            opacity: self.opacity,
            blend: self.blend,
            targets: self.targets.clone(),
            fill: self.fill,
            generator: self.mask_generator,
        }
    }

    fn set_props(&mut self, props: LayerProps) {
        self.name = props.name;
        self.visible = props.visible;
        self.opacity = props.opacity;
        self.blend = props.blend;
        self.targets = props.targets;
        self.fill = props.fill;
        self.mask_generator = props.generator;
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
    pub mask: LayerMask<'a>,
}

/// How much of a layer shows, texel by texel.
#[derive(Clone, Copy)]
pub enum LayerMask<'a> {
    /// No mask: all of it.
    None,
    /// A mask nobody painted on this slot: the same everywhere (0..255).
    Uniform(u8),
    /// A painted mask; the value is in red.
    Image(&'a Image),
}

impl LayerMask<'_> {
    /// The mask value (0..255) at a pixel's byte offset.
    pub fn at(&self, offset: usize) -> u8 {
        match self {
            LayerMask::None => 255,
            LayerMask::Uniform(value) => *value,
            LayerMask::Image(image) => image.rgba[offset],
        }
    }
}

/// Settings of a layer that can be changed (and undone) as one.
#[derive(Clone, PartialEq)]
pub struct LayerProps {
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: BlendMode,
    pub targets: Vec<bool>,
    /// Changing these remakes the layer's procedural images.
    pub fill: Option<FillValues>,
    pub generator: Option<Generator>,
}

/// Which procedural images to make again.
#[derive(Clone, Copy)]
pub struct Remake {
    pub fill: bool,
    pub mask: bool,
    /// Only the images a targeted slot doesn't have yet.
    pub only_missing: bool,
}

impl Remake {
    pub const ALL: Remake = Remake {
        fill: true,
        mask: true,
        only_missing: false,
    };
    pub const FILL: Remake = Remake {
        fill: true,
        mask: false,
        only_missing: false,
    };
}

/// Pixels of one square tile of a layer texture, tightly packed RGBA.
pub struct Tile {
    pub region: ImageRegion,
    pub pixels: Vec<u8>,
}

/// One undoable change. Applying it performs the change and turns it into its
/// own inverse, so the same value serves undo and redo.
pub enum Change {
    /// Swap pixels of a layer image with the stored ones.
    Pixels {
        layer: u64,
        material: usize,
        plane: Plane,
        tiles: Vec<Tile>,
    },
    /// With `stored` set: insert it at `index`. Without: remove the layer at
    /// `index` and store it.
    Layer { index: usize, stored: Option<Layer> },
    /// Move the layer at `from` to `to`.
    Move { from: usize, to: usize },
    /// Swap a layer's settings with the stored ones.
    Props { layer: u64, props: LayerProps },
    /// Swap a layer's mask (`Layer::mask` plus its images, one per slot)
    /// with the stored one. Adds, removes or replaces the mask.
    Mask {
        layer: u64,
        mask: Option<u8>,
        images: Vec<Option<Image>>,
    },
}

impl Change {
    pub fn bytes(&self) -> usize {
        match self {
            Change::Pixels { tiles, .. } => tiles.iter().map(|tile| tile.pixels.len()).sum(),
            Change::Layer {
                stored: Some(layer),
                ..
            } => layer.bytes(),
            Change::Mask { images, .. } => images.iter().flatten().map(|i| i.rgba.len()).sum(),
            _ => 0,
        }
    }
}

/// What a change touched, so the GPU copy can be brought up to date.
pub enum Dirty {
    /// Pixels of one layer image changed.
    Pixels {
        layer: u64,
        material: usize,
        plane: Plane,
        region: ImageRegion,
    },
    /// Layers were added, removed, reordered, or their settings changed.
    Layers,
    /// A layer's mask was added, removed or replaced as a whole.
    Mask { layer: u64 },
    /// The images of a procedural layer were remade, all of its planes.
    Generated { layer: u64 },
}

pub struct Document {
    pub mesh: Mesh,
    /// Bottom to top; `layers[0]` is the base layer.
    pub layers: Vec<Layer>,
    next_layer_id: u64,
    /// Texel positions per (material slot, width, height), made when a
    /// generator first needs them.
    bakes: BakeCache,
}

type BakeCache = Mutex<HashMap<(usize, u32, u32), Arc<Bake>>>;

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
            mask: None,
            fill: None,
            mask_generator: None,
        };
        let next_layer_id = layers.iter().map(|layer| layer.id).max().unwrap_or(0) + 1;
        let mut document = Self {
            mesh,
            layers: std::iter::once(base).chain(layers).collect(),
            next_layer_id,
            bakes: BakeCache::default(),
        };
        // Procedural layers are stored as settings only.
        for index in 1..document.layers.len() {
            if document.layers[index].is_procedural() {
                document.regenerate(index, Remake::ALL);
            }
        }
        document
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

    /// How much of `layer` shows on `material`.
    pub fn layer_mask<'a>(&'a self, layer: &'a Layer, material: usize) -> LayerMask<'a> {
        let Some(value) = layer.mask else {
            return LayerMask::None;
        };
        let size = self.slot_sizes().get(material).copied();
        let image = layer
            .textures
            .get(material)
            .and_then(|planes| planes[Plane::Mask.index()].as_ref())
            .filter(|image| Some((image.width, image.height)) == size);
        image.map_or(LayerMask::Uniform(value), LayerMask::Image)
    }

    /// Adds a mask to the layer at `index` (showing it all, or hiding it
    /// all), or removes its mask with `value` `None`. Returns the change
    /// that undoes it, or `None` if there's nothing to do.
    pub fn set_mask(&mut self, index: usize, value: Option<u8>) -> Option<Change> {
        let layer = self.layers.get(index).filter(|layer| !layer.is_base())?;
        if layer.mask.is_none() && value.is_none() {
            return None;
        }
        let mut change = Change::Mask {
            layer: layer.id,
            mask: value,
            images: vec![None; self.mesh.materials.len()],
        };
        self.apply(&mut change)?;
        Some(change)
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
                    mask: self.layer_mask(layer, material),
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
            mask: None,
            fill: None,
            mask_generator: None,
        }
    }

    /// A new fill layer on every slot, made from `fill`. Not yet in the stack.
    pub fn new_fill_layer(&mut self, fill: FillValues) -> Layer {
        let mut layer = self.new_layer();
        layer.name = format!("Fill {}", layer.id);
        layer.fill = Some(fill);
        refresh_layer(&mut layer, &self.mesh, &self.bakes, Remake::ALL);
        layer
    }

    /// Remakes the procedural images of the layer at `index` from its
    /// settings. A mask that lost its generator keeps its pixels.
    pub fn regenerate(&mut self, index: usize, remake: Remake) {
        if let Some(layer) = self.layers.get_mut(index).filter(|l| !l.is_base()) {
            refresh_layer(layer, &self.mesh, &self.bakes, remake);
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
                plane,
                tiles,
            } => {
                let index = self.layer_index(*layer)?;
                let image =
                    self.layers[index].textures.get_mut(*material)?[plane.index()].as_mut()?;
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
                    plane: *plane,
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
            Change::Props {
                layer: layer_id,
                props,
            } => {
                let index = self.layer_index(*layer_id)?;
                if props.targets.len() != self.mesh.materials.len() {
                    return None;
                }
                let before = self.layers[index].props();
                self.layers[index].set_props(std::mem::replace(props, before.clone()));
                let layer = &self.layers[index];
                let remake = Remake {
                    fill: before.fill != layer.fill,
                    mask: before.generator != layer.mask_generator,
                    only_missing: false,
                };
                let retargeted = before.targets != layer.targets && layer.is_procedural();
                if remake.fill || remake.mask || retargeted {
                    let scope = if retargeted && !(remake.fill || remake.mask) {
                        Remake {
                            only_missing: true,
                            ..Remake::ALL
                        }
                    } else {
                        remake
                    };
                    self.regenerate(index, scope);
                    return Some(Dirty::Generated { layer: *layer_id });
                }
                Some(Dirty::Layers)
            }
            Change::Mask {
                layer,
                mask,
                images,
            } => {
                let index = self.layer_index(*layer)?;
                let target = &mut self.layers[index];
                if target.is_base() || images.len() != target.textures.len() {
                    return None;
                }
                std::mem::swap(&mut target.mask, mask);
                for (planes, image) in target.textures.iter_mut().zip(images.iter_mut()) {
                    std::mem::swap(&mut planes[Plane::Mask.index()], image);
                }
                Some(Dirty::Mask { layer: *layer })
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
    let shown = |layer: &VisibleLayer| layer.mask.at(offset);
    let contributes = |layer: &&VisibleLayer| {
        layer.image.rgba[offset + 3] != 0 && layer.opacity > 0.0 && shown(layer) != 0
    };
    // A fully opaque Normal layer hides everything below it.
    let start = sources
        .iter()
        .rposition(|layer| {
            layer.blend == BlendMode::Normal
                && layer.opacity == 1.0
                && layer.image.rgba[offset + 3] == 255
                && shown(layer) == 255
        })
        .unwrap_or(0);
    let mut contributors = sources[start..].iter().filter(contributes);
    let Some(first) = contributors.next() else {
        return CLEAR;
    };
    if contributors.next().is_none()
        && first.opacity == 1.0
        && first.blend == BlendMode::Normal
        && shown(first) == 255
    {
        // A single Normal layer shows through unchanged.
        let mut pixel = [0; 4];
        pixel.copy_from_slice(&first.image.rgba[offset..offset + 4]);
        return pixel;
    }

    let mut color = [0.0f32; 3]; // premultiplied
    let mut coverage = 0.0f32;
    for layer in &sources[start..] {
        let alpha = layer.image.rgba[offset + 3] as f32 / 255.0
            * layer.opacity
            * (shown(layer) as f32 / 255.0);
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

/// Makes the images of `layer`'s fill and generated mask that `remake` asks
/// for. Images on slots the layer doesn't target are dropped, so they can't
/// go stale.
fn refresh_layer(layer: &mut Layer, mesh: &Mesh, bakes: &BakeCache, remake: Remake) {
    let only_missing = remake.only_missing;
    let generator = layer.mask_generator.filter(|_| layer.mask.is_some());
    for (slot, material) in mesh.materials.iter().enumerate() {
        let (width, height) = (material.base_color.width, material.base_color.height);
        let targeted = layer.targets.get(slot) == Some(&true);
        let Some(planes) = layer.textures.get_mut(slot) else {
            continue;
        };
        let current = |image: &Option<Image>| {
            image
                .as_ref()
                .is_some_and(|image| (image.width, image.height) == (width, height))
        };
        let bake = || {
            bakes
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .entry((slot, width, height))
                .or_insert_with(|| Arc::new(Bake::new(mesh, slot, width, height)))
                .clone()
        };
        if let Some(fill) = layer.fill.filter(|_| remake.fill) {
            // Only a mixed fill needs the pattern, and only on a slot it fills.
            let mut mixed = None;
            for channel in Channel::ALL {
                let plane = &mut planes[channel.index()];
                let wanted = targeted && fill.channels[channel.index()];
                if !wanted {
                    *plane = None;
                } else if !(only_missing && current(plane)) {
                    *plane = Some(match fill.mix {
                        None => Image::filled(width, height, fill.pixel(channel)),
                        Some(mix) => {
                            let values =
                                mixed.get_or_insert_with(|| bake().generate_cached(&mix.generator));
                            let ramp = fill.ramp(channel);
                            Image {
                                width,
                                height,
                                rgba: values
                                    .iter()
                                    .flat_map(|&value| ramp[value as usize])
                                    .collect(),
                            }
                        }
                    });
                }
            }
        }
        if let Some(generator) = generator.filter(|_| remake.mask) {
            let plane = &mut planes[Plane::Mask.index()];
            if !targeted {
                *plane = None;
            } else if !(only_missing && current(plane)) {
                let rgba = bake()
                    .generate_cached(&generator)
                    .iter()
                    .flat_map(|&value| [value, value, value, 255])
                    .collect();
                *plane = Some(Image {
                    width,
                    height,
                    rgba,
                });
            }
        }
    }
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
    use super::{BlendMode, CLEAR, Change, Document, FillMix, FillValues, LayerProps, Plane};
    use crate::generator::Generator;
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
            plane: Plane::Channel(Channel::BaseColor),
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

    #[test]
    fn a_mask_hides_the_layer_where_it_is_black() {
        let mut document = document();
        // Green over the red slot, on every pixel.
        let (w, h) = document.slot_sizes()[0];
        document.layers[1].textures[0][0] = Some(Image::filled(w, h, [0, 255, 0, 255]));
        let undo = document.set_mask(1, Some(255)).unwrap();
        // A painted mask: black at pixel 0, half at pixel 1; white elsewhere.
        let mut mask = Image::filled(w, h, [255, 255, 255, 255]);
        mask.rgba[0..4].copy_from_slice(&[0, 0, 0, 255]);
        mask.rgba[4..8].copy_from_slice(&[128, 128, 128, 255]);
        document.layers[1].textures[0][Plane::Mask.index()] = Some(mask);

        let composite = document.composite(0, Channel::BaseColor).unwrap();
        assert_eq!(
            composite.rgba[0..4],
            [255, 0, 0, 255],
            "hidden: the red below"
        );
        assert_eq!(
            composite.rgba[8..12],
            [0, 255, 0, 255],
            "shown: the green layer"
        );
        let half = &composite.rgba[4..8];
        assert!(
            half[0] > 100 && half[1] > 100,
            "half shown: a mix, got {half:?}"
        );

        // The other slot has no painted mask: the unpainted value applies.
        document.layers[1].textures[1][0] = Some(Image::filled(w, h, [0, 255, 0, 255]));
        assert_eq!(
            document.composite(1, Channel::BaseColor).unwrap().rgba[0..4],
            [0, 255, 0, 255]
        );

        // Undoing the "add mask" removes it, painted image included.
        let mut undo = undo;
        document.apply(&mut undo);
        assert_eq!(document.layers[1].mask, None);
        assert!(document.layers[1].textures[0][Plane::Mask.index()].is_none());
        assert_eq!(
            document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            [0, 255, 0, 255]
        );
        // ...and redoing brings it back as it was.
        document.apply(&mut undo);
        assert_eq!(document.layers[1].mask, Some(255));
        assert_eq!(
            document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            [255, 0, 0, 255]
        );
    }

    #[test]
    fn a_hide_all_mask_hides_the_whole_layer() {
        let mut document = document();
        let (w, h) = document.slot_sizes()[0];
        document.layers[1].textures[0][0] = Some(Image::filled(w, h, [0, 255, 0, 255]));
        document.set_mask(1, Some(0)).unwrap();
        assert_eq!(
            document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            [255, 0, 0, 255]
        );
        // Removing it shows the layer again; the base layer has no mask.
        document.set_mask(1, None).unwrap();
        assert_eq!(
            document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            [0, 255, 0, 255]
        );
        assert!(document.set_mask(0, Some(255)).is_none());
        assert!(document.set_mask(1, None).is_none(), "nothing to remove");
    }

    /// The checkered cube, whose atlas is big enough for generators.
    fn cube_document() -> Document {
        Document::new(primitives::cube())
    }

    fn with_props(document: &mut Document, index: usize, edit: impl Fn(&mut LayerProps)) -> Change {
        let mut props = document.layers[index].props();
        edit(&mut props);
        let mut change = Change::Props {
            layer: document.layers[index].id,
            props,
        };
        document.apply(&mut change).unwrap();
        change
    }

    fn mask_pixels(document: &Document, layer: usize) -> Option<Vec<u8>> {
        document.layers[layer].textures[0][Plane::Mask.index()]
            .as_ref()
            .map(|image| image.rgba.clone())
    }

    #[test]
    fn a_fill_layer_covers_the_layers_below_with_its_values() {
        let mut document = document();
        let fill = FillValues {
            color: [0, 255, 0],
            roughness: 1.0,
            channels: [true, true, false],
            ..FillValues::default()
        };
        let layer = document.new_fill_layer(fill);
        let mut add = Change::Layer {
            index: 2,
            stored: Some(layer),
        };
        document.apply(&mut add).unwrap();

        let color = document.composite(0, Channel::BaseColor).unwrap();
        assert_eq!(&color.rgba[0..4], &[0, 255, 0, 255]);
        assert_eq!(&color.rgba[60..64], &[0, 255, 0, 255], "everywhere");
        let roughness = document.composite(1, Channel::Roughness).unwrap();
        assert_eq!(&roughness.rgba[0..3], &[255, 255, 255]);
        // Metallic isn't filled: the model's own value shows.
        let metallic = document.composite(0, Channel::Metallic).unwrap();
        assert_eq!(&metallic.rgba[0..3], &[0, 0, 0]);
        assert!(document.layers[2].textures[0][Channel::Metallic.index()].is_none());

        // Changing the values remakes the images; undoing restores the old ones.
        let mut change = with_props(&mut document, 2, |p| {
            p.fill = Some(FillValues {
                color: [0, 0, 255],
                ..fill
            })
        });
        assert_eq!(
            &document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            &[0, 0, 255, 255]
        );
        document.apply(&mut change).unwrap();
        assert_eq!(
            &document.composite(0, Channel::BaseColor).unwrap().rgba[0..4],
            &[0, 255, 0, 255]
        );
    }

    #[test]
    fn a_fill_layer_is_only_painted_through_its_mask() {
        use crate::painting::stroke::Stroke;
        let mut document = document();
        let layer = document.new_fill_layer(FillValues::default());
        document
            .apply(&mut Change::Layer {
                index: 2,
                stored: Some(layer),
            })
            .unwrap();
        assert!(Stroke::begin(&document, 2).is_none());
        assert!(Stroke::begin_on_mask(&document, 2).is_none(), "no mask yet");
        document.set_mask(2, Some(0)).unwrap();
        assert!(Stroke::begin_on_mask(&document, 2).is_some());
        // Paint layers are unaffected.
        assert!(Stroke::begin(&document, 1).is_some());
    }

    #[test]
    fn a_generated_mask_follows_its_settings_through_undo_and_redo() {
        let mut document = cube_document();
        document.layers[1].textures[0][0] = Some(Image::filled(768, 512, [0, 255, 0, 255]));
        document.set_mask(1, Some(255)).unwrap();
        assert!(
            mask_pixels(&document, 1).is_none(),
            "no image until generated"
        );

        let generator = Generator::default();
        with_props(&mut document, 1, |p| p.generator = Some(generator));
        let first = mask_pixels(&document, 1).expect("generated");
        assert!(
            first.chunks_exact(4).any(|p| p[0] != first[0]),
            "the noise varies"
        );

        let other = Generator {
            seed: 5,
            ..generator
        };
        let mut change = with_props(&mut document, 1, |p| p.generator = Some(other));
        let second = mask_pixels(&document, 1).unwrap();
        assert_ne!(first, second);
        document.apply(&mut change).unwrap();
        assert_eq!(mask_pixels(&document, 1).unwrap(), first, "undone");
        document.apply(&mut change).unwrap();
        assert_eq!(mask_pixels(&document, 1).unwrap(), second, "redone");

        // Dropping the generator keeps the pixels, as a painted mask...
        let mut convert = with_props(&mut document, 1, |p| p.generator = None);
        assert_eq!(mask_pixels(&document, 1).unwrap(), second);
        // ...and undoing that makes it procedural again.
        document.apply(&mut convert).unwrap();
        assert_eq!(document.layers[1].mask_generator, Some(other));
        assert_eq!(mask_pixels(&document, 1).unwrap(), second);
    }

    #[test]
    fn a_generated_mask_only_shows_the_layer_where_the_pattern_is_bright() {
        let mut document = cube_document();
        document.layers[1].textures[0][0] = Some(Image::filled(768, 512, [0, 255, 0, 255]));
        document.set_mask(1, Some(255)).unwrap();
        with_props(&mut document, 1, |p| {
            p.generator = Some(Generator {
                low: 0.45,
                high: 0.55,
                ..Generator::default()
            })
        });
        let mask = mask_pixels(&document, 1).unwrap();
        let composite = document.composite(0, Channel::BaseColor).unwrap();
        let (mut green, mut base) = (0, 0);
        for (index, pixel) in composite.rgba.chunks_exact(4).enumerate() {
            if mask[index * 4] == 255 && pixel == [0, 255, 0, 255] {
                green += 1;
            }
            if mask[index * 4] == 0 && pixel != [0, 255, 0, 255] {
                base += 1;
            }
        }
        assert!(
            green > 10_000 && base > 10_000,
            "{green} shown, {base} hidden"
        );
    }

    #[test]
    fn moving_a_generated_mask_to_another_slot_makes_its_pixels_there() {
        let mut document = Document::new({
            let mut mesh = primitives::cube();
            mesh.materials
                .push(Material::new("Second", Image::filled(8, 8, [9; 4])));
            mesh
        });
        document.layers[1].targets[1] = false;
        document.set_mask(1, Some(255)).unwrap();
        with_props(&mut document, 1, |p| {
            p.generator = Some(Generator::default())
        });
        assert!(document.layers[1].textures[0][Plane::Mask.index()].is_some());
        assert!(document.layers[1].textures[1][Plane::Mask.index()].is_none());
        with_props(&mut document, 1, |p| p.targets[1] = true);
        assert!(document.layers[1].textures[1][Plane::Mask.index()].is_some());
    }

    #[test]
    fn a_fill_mix_blends_between_two_looks_in_linear_light() {
        let plain = FillValues {
            color: [255, 255, 255],
            roughness: 0.2,
            ..FillValues::default()
        };
        assert!(
            plain
                .ramp(Channel::BaseColor)
                .iter()
                .all(|p| *p == plain.pixel(Channel::BaseColor))
        );
        let mixed = FillValues {
            mix: Some(FillMix {
                color: [0, 0, 0],
                roughness: 0.8,
                ..FillMix::default()
            }),
            ..plain
        };
        let colors = mixed.ramp(Channel::BaseColor);
        assert_eq!(colors[0], [255, 255, 255, 255], "pattern 0: the fill's own");
        assert_eq!(colors[255], [0, 0, 0, 255], "pattern 1: the second look");
        // Half of white's light is brighter than half of its sRGB number.
        assert!(colors[128][0] > 170, "{:?}", colors[128]);
        let roughness = mixed.ramp(Channel::Roughness);
        assert_eq!(roughness[0][0], 51);
        assert_eq!(roughness[255][0], 204);
    }

    #[test]
    fn a_mixed_fill_layer_varies_over_the_model_and_follows_undo() {
        let mut document = cube_document();
        let fill = FillValues {
            color: [255, 0, 0],
            mix: Some(FillMix {
                color: [0, 0, 255],
                generator: Generator {
                    low: 0.4,
                    high: 0.6,
                    ..Generator::default()
                },
                ..FillMix::default()
            }),
            ..FillValues::default()
        };
        let layer = document.new_fill_layer(fill);
        document
            .apply(&mut Change::Layer {
                index: 2,
                stored: Some(layer),
            })
            .unwrap();
        let image = document.layers[2].textures[0][Channel::BaseColor.index()]
            .as_ref()
            .expect("filled");
        let count = |color: [u8; 4]| image.rgba.chunks_exact(4).filter(|p| *p == color).count();
        let (red, blue) = (count([255, 0, 0, 255]), count([0, 0, 255, 255]));
        assert!(red > 10_000 && blue > 10_000, "{red} red, {blue} blue");

        let mut change = with_props(&mut document, 2, |p| p.fill = Some(FillValues::default()));
        let image = document.layers[2].textures[0][Channel::BaseColor.index()]
            .as_ref()
            .unwrap();
        assert!(
            image
                .rgba
                .chunks_exact(4)
                .all(|p| p == [200, 200, 200, 255])
        );
        document.apply(&mut change).unwrap();
        let image = document.layers[2].textures[0][Channel::BaseColor.index()]
            .as_ref()
            .unwrap();
        assert!(image.rgba.chunks_exact(4).any(|p| p == [0, 0, 255, 255]));
    }
}
