use crate::mesh::{Image, ImageRegion};
use eframe::egui_wgpu::wgpu;

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

/// Uploads `region` of `image` into mip level 0 of a same-sized `texture`.
/// A region outside the image is ignored: it would be a GPU validation
/// error, which panics.
pub fn write_region(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    image: &Image,
    region: ImageRegion,
) {
    let fits = region.width > 0
        && region.height > 0
        && region
            .x
            .checked_add(region.width)
            .is_some_and(|end| end <= image.width)
        && region
            .y
            .checked_add(region.height)
            .is_some_and(|end| end <= image.height);
    let size = texture.size();
    if !fits || (size.width, size.height) != (image.width, image.height) {
        return;
    }
    let row_bytes = (region.width * 4) as usize;
    let mut data = Vec::with_capacity(row_bytes * region.height as usize);
    for y in region.y..region.y + region.height {
        let start = ((y * image.width + region.x) * 4) as usize;
        data.extend_from_slice(&image.rgba[start..start + row_bytes]);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d {
                x: region.x,
                y: region.y,
                z: 0,
            },
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(region.width * 4),
            rows_per_image: Some(region.height),
        },
        wgpu::Extent3d {
            width: region.width,
            height: region.height,
            depth_or_array_layers: 1,
        },
    );
}
