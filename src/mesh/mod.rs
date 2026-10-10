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
}

/// One material slot of the model. This will become a "texture set" you paint on.
pub struct Material {
    pub name: String,
    pub base_color: Image,
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
    let c = value as f32 / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
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