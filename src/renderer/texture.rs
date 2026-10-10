use crate::mesh::Image;
use eframe::egui_wgpu::wgpu::{self, util::DeviceExt as _};

/// sRGB format: the GPU decodes to linear when sampling, so lighting math is correct.
pub const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub fn create_texture(device: &wgpu::Device, queue: &wgpu::Queue, image: &Image) -> wgpu::Texture {
    let (mip_levels, data) = build_mip_chain(image);

    device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("material_base_color"),
            size: wgpu::Extent3d {
                width: image.width,
                height: image.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &data,
    )
}

pub fn create_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("material_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        anisotropy_clamp: 8,
        ..Default::default()
    })
}

/// Builds all mip levels with a simple 2x2 box filter, concatenated level after level.
fn build_mip_chain(image: &Image) -> (u32, Vec<u8>) {
    let mut data = image.rgba.clone();
    let mut previous = image.rgba.clone();
    let (mut w, mut h) = (image.width, image.height);
    let mut levels = 1;

    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let mut next = Vec::with_capacity((nw * nh * 4) as usize);

        for y in 0..nh {
            for x in 0..nw {
                let x0 = (x * 2).min(w - 1);
                let x1 = (x * 2 + 1).min(w - 1);
                let y0 = (y * 2).min(h - 1);
                let y1 = (y * 2 + 1).min(h - 1);
                for c in 0..4 {
                    let at = |xx: u32, yy: u32| previous[((yy * w + xx) * 4 + c) as usize] as u32;
                    let sum = at(x0, y0) + at(x1, y0) + at(x0, y1) + at(x1, y1);
                    next.push(((sum + 2) / 4) as u8);
                }
            }
        }

        data.extend_from_slice(&next);
        previous = next;
        w = nw;
        h = nh;
        levels += 1;
    }

    (levels, data)
}