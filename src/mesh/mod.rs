use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::ops::Range;

pub mod gltf_import;
pub mod primitives;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

/// 8-bit RGBA, tightly packed, sRGB-encoded colour.
#[derive(Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

impl Image {
    /// A 1x1 image from a *linear* colour (like glTF's baseColorFactor).
    pub fn solid(color: [f32; 4]) -> Self {
        Self {
            width: 1,
            height: 1,
            rgba: vec![
                linear_to_srgb(color[0]),
                linear_to_srgb(color[1]),
                linear_to_srgb(color[2]),
                (color[3].clamp(0.0, 1.0) * 255.0).round() as u8,
            ],
        }
    }

    /// A `width` x `height` image where every pixel is `pixel` (already sRGB-encoded).
    pub fn filled(width: u32, height: u32, pixel: [u8; 4]) -> Self {
        Self {
            width,
            height,
            rgba: pixel.repeat((width * height) as usize),
        }
    }
}

/// A rectangle of pixels inside an `Image`, used to upload only what changed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl ImageRegion {
    pub fn pixel(x: u32, y: u32) -> Self {
        Self {
            x,
            y,
            width: 1,
            height: 1,
        }
    }

    pub fn union(self, other: Self) -> Self {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        let right = (self.x + self.width).max(other.x + other.width);
        let bottom = (self.y + self.height).max(other.y + other.height);
        Self {
            x,
            y,
            width: right - x,
            height: bottom - y,
        }
    }
}

/// How a texture's bytes map to values that blend linearly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// Colour: sRGB-encoded, blended in linear light.
    Srgb,
    /// Data such as roughness: the bytes are the values.
    Linear,
}

impl Encoding {
    pub fn decode(self, byte: u8) -> f32 {
        match self {
            Encoding::Srgb => srgb_to_linear(byte),
            Encoding::Linear => byte as f32 / 255.0,
        }
    }

    pub fn encode(self, value: f32) -> u8 {
        match self {
            Encoding::Srgb => linear_to_srgb(value),
            Encoding::Linear => (value.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }
}

/// A material property that can be painted, each with its own texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Channel {
    BaseColor,
    Roughness,
    Metallic,
}

impl Channel {
    pub const COUNT: usize = 3;
    pub const ALL: [Channel; Channel::COUNT] =
        [Channel::BaseColor, Channel::Roughness, Channel::Metallic];

    pub fn index(self) -> usize {
        self as usize
    }

    pub fn name(self) -> &'static str {
        match self {
            Channel::BaseColor => "Base color",
            Channel::Roughness => "Roughness",
            Channel::Metallic => "Metallic",
        }
    }

    /// Suffix of exported files, as game engines expect.
    pub fn file_suffix(self) -> &'static str {
        match self {
            Channel::BaseColor => "BaseColor",
            Channel::Roughness => "Roughness",
            Channel::Metallic => "Metallic",
        }
    }

    pub fn encoding(self) -> Encoding {
        match self {
            Channel::BaseColor => Encoding::Srgb,
            Channel::Roughness | Channel::Metallic => Encoding::Linear,
        }
    }
}

/// A grey image holding one data value (such as roughness) in every pixel.
pub fn data_image(width: u32, height: u32, value: f32) -> Image {
    let byte = Encoding::Linear.encode(value);
    Image::filled(width, height, [byte, byte, byte, 255])
}

/// One material slot of the model: a texture per channel, all the same size.
pub struct Material {
    pub name: String,
    pub base_color: Image,
    /// Data channels hold their value in the red (and green, blue) bytes.
    pub roughness: Image,
    pub metallic: Image,
}

impl Material {
    /// A material with uniform, mid roughness and no metal.
    pub fn new(name: impl Into<String>, base_color: Image) -> Self {
        Self::with_values(name, base_color, 0.5, 0.0)
    }

    pub fn with_values(
        name: impl Into<String>,
        base_color: Image,
        roughness: f32,
        metallic: f32,
    ) -> Self {
        let (width, height) = (base_color.width, base_color.height);
        Self {
            name: name.into(),
            base_color,
            roughness: data_image(width, height, roughness),
            metallic: data_image(width, height, metallic),
        }
    }

    pub fn channel(&self, channel: Channel) -> &Image {
        match channel {
            Channel::BaseColor => &self.base_color,
            Channel::Roughness => &self.roughness,
            Channel::Metallic => &self.metallic,
        }
    }

    pub fn channel_mut(&mut self, channel: Channel) -> &mut Image {
        match channel {
            Channel::BaseColor => &mut self.base_color,
            Channel::Roughness => &mut self.roughness,
            Channel::Metallic => &mut self.metallic,
        }
    }

    /// Resizes the data channels to the base colour's size, which every
    /// layer of this slot uses.
    pub fn conform_channels(&mut self) {
        let (width, height) = (self.base_color.width, self.base_color.height);
        for channel in [Channel::Roughness, Channel::Metallic] {
            let image = self.channel_mut(channel);
            if (image.width, image.height) != (width, height) {
                *image = resize(image, width, height);
            }
        }
    }
}

/// Smoothly resamples an image to a new size.
pub fn resize(image: &Image, width: u32, height: u32) -> Image {
    if (image.width, image.height) == (1, 1) {
        let pixel = [image.rgba[0], image.rgba[1], image.rgba[2], image.rgba[3]];
        return Image::filled(width, height, pixel);
    }
    let source = ::image::RgbaImage::from_raw(image.width, image.height, image.rgba.clone());
    match source {
        Some(source) => {
            let resized = ::image::imageops::resize(
                &source,
                width,
                height,
                ::image::imageops::FilterType::Triangle,
            );
            Image {
                width,
                height,
                rgba: resized.into_raw(),
            }
        }
        None => Image::filled(width, height, [0, 0, 0, 255]),
    }
}

/// A range of indices drawn with one material.
#[derive(Clone)]
pub struct SubMesh {
    pub indices: Range<u32>,
    pub material: usize,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub submeshes: Vec<SubMesh>,
    pub materials: Vec<Material>,
}

impl Mesh {
    /// Center and radius of a sphere enclosing the mesh (via its bounding box).
    pub fn bounds(&self) -> (Vec3, f32) {
        if self.vertices.is_empty() {
            return (Vec3::ZERO, 1.0);
        }
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for v in &self.vertices {
            let p = Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        let center = (min + max) * 0.5;
        let radius = ((max - min) * 0.5).length().max(1e-4);
        (center, radius)
    }
}

pub fn srgb_to_linear(value: u8) -> f32 {
    // Called per channel per layer when compositing, so use a table.
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|i| {
            let c = i as f32 / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        })
    })[value as usize]
}

pub fn linear_to_srgb(linear: f32) -> u8 {
    let c = linear.clamp(0.0, 1.0);
    let s = if c <= 0.003_130_8 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round() as u8
}
