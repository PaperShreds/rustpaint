use crate::mesh::{Image, ImageRegion};
use eframe::egui_wgpu::wgpu::{self, util::DeviceExt as _};

/// sRGB format: the GPU decodes to linear when sampling, so lighting math is correct.
pub const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// One downsampled level of a mip chain.
struct MipLevel {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

/// A material texture that can be partially updated while painting.
///
/// Level 0 lives in the material's `Image`; levels 1.. are kept here on the CPU
/// so a painted region can be re-filtered without rebuilding the whole chain.
pub struct PaintableTexture {
    texture: wgpu::Texture,
    lower_mips: Vec<MipLevel>,
}

impl PaintableTexture {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, image: &Image) -> Self {
        let lower_mips = build_lower_mips(image);
        let mut data = image.rgba.clone();
        for level in &lower_mips {
            data.extend_from_slice(&level.rgba);
        }

        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("material_base_color"),
                size: wgpu::Extent3d {
                    width: image.width,
                    height: image.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: lower_mips.len() as u32 + 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: TEXTURE_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &data,
        );
        Self {
            texture,
            lower_mips,
        }
    }

    pub fn create_view(&self) -> wgpu::TextureView {
        self.texture
            .create_view(&wgpu::TextureViewDescriptor::default())
    }

    /// Uploads `region` of `image` (the new level 0) and refreshes the mips it affects.
    pub fn update_region(&mut self, queue: &wgpu::Queue, image: &Image, region: ImageRegion) {
        let size = self.texture.size();
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
        if !fits || (size.width, size.height) != (image.width, image.height) {
            // A bad region would be a GPU validation error, which panics; skip it.
            return;
        }
        write_region(queue, &self.texture, 0, image.width, &image.rgba, region);
        for (level, region) in update_lower_mips(&mut self.lower_mips, image, region) {
            let mip = &self.lower_mips[level - 1];
            write_region(
                queue,
                &self.texture,
                level as u32,
                mip.width,
                &mip.rgba,
                region,
            );
        }
    }
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

/// Builds mip levels 1.. with a simple 2x2 box filter.
fn build_lower_mips(image: &Image) -> Vec<MipLevel> {
    let mut levels: Vec<MipLevel> = Vec::new();
    let (mut w, mut h) = (image.width, image.height);

    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let previous = levels
            .last()
            .map_or(&image.rgba[..], |level| &level.rgba[..]);
        let mut next = Vec::with_capacity((nw * nh * 4) as usize);
        for y in 0..nh {
            for x in 0..nw {
                next.extend_from_slice(&downsample(previous, w, h, x, y));
            }
        }
        levels.push(MipLevel {
            width: nw,
            height: nh,
            rgba: next,
        });
        w = nw;
        h = nh;
    }
    levels
}

/// Re-filters the parts of `lower_mips` that depend on `region` of level 0.
/// Returns the changed region of every level, keyed by mip level (1..).
fn update_lower_mips(
    lower_mips: &mut [MipLevel],
    image: &Image,
    region: ImageRegion,
) -> Vec<(usize, ImageRegion)> {
    let mut changed = Vec::with_capacity(lower_mips.len());
    let mut region = region;
    let (mut src_w, mut src_h) = (image.width, image.height);

    for index in 0..lower_mips.len() {
        let (before, after) = lower_mips.split_at_mut(index);
        let src = before
            .last()
            .map_or(&image.rgba[..], |level| &level.rgba[..]);
        let dst = &mut after[0];

        // Clamp both ends: with an odd source size, the last row/column maps
        // past the end of the smaller level.
        let x0 = (region.x / 2).min(dst.width - 1);
        let y0 = (region.y / 2).min(dst.height - 1);
        let x1 = ((region.x + region.width - 1) / 2).min(dst.width - 1);
        let y1 = ((region.y + region.height - 1) / 2).min(dst.height - 1);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let offset = ((y * dst.width + x) * 4) as usize;
                dst.rgba[offset..offset + 4].copy_from_slice(&downsample(src, src_w, src_h, x, y));
            }
        }

        region = ImageRegion {
            x: x0,
            y: y0,
            width: x1 - x0 + 1,
            height: y1 - y0 + 1,
        };
        changed.push((index + 1, region));
        src_w = dst.width;
        src_h = dst.height;
    }
    changed
}

/// Averages the 2x2 block of `src` (size `w` x `h`) that maps to pixel (`x`, `y`) one level down.
fn downsample(src: &[u8], w: u32, h: u32, x: u32, y: u32) -> [u8; 4] {
    let x0 = (x * 2).min(w - 1);
    let x1 = (x * 2 + 1).min(w - 1);
    let y0 = (y * 2).min(h - 1);
    let y1 = (y * 2 + 1).min(h - 1);
    let mut out = [0; 4];
    for (c, value) in out.iter_mut().enumerate() {
        let at = |xx: u32, yy: u32| src[((yy * w + xx) * 4) as usize + c] as u32;
        let sum = at(x0, y0) + at(x1, y0) + at(x0, y1) + at(x1, y1);
        *value = ((sum + 2) / 4) as u8;
    }
    out
}

fn write_region(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip_level: u32,
    image_width: u32,
    rgba: &[u8],
    region: ImageRegion,
) {
    let row_bytes = (region.width * 4) as usize;
    let mut data = Vec::with_capacity(row_bytes * region.height as usize);
    for y in region.y..region.y + region.height {
        let start = ((y * image_width + region.x) * 4) as usize;
        data.extend_from_slice(&rgba[start..start + row_bytes]);
    }
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
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

#[cfg(test)]
mod tests {
    use super::{build_lower_mips, update_lower_mips};
    use crate::mesh::{Image, ImageRegion};

    fn gradient(width: u32, height: u32) -> Image {
        let rgba = (0..width * height)
            .flat_map(|i| [(i * 7) as u8, (i * 13) as u8, (i * 29) as u8, 255])
            .collect();
        Image {
            width,
            height,
            rgba,
        }
    }

    #[test]
    fn partial_mip_update_matches_full_rebuild() {
        // Odd sizes exercise the edge clamping in the box filter; 768 x 512 is the
        // default cube, whose mips go 3 -> 1 wide. Regions touch every edge.
        for (width, height) in [(64, 64), (37, 21), (1, 9), (768, 512)] {
            let regions = [
                ImageRegion {
                    x: width / 3,
                    y: height / 4,
                    width: (width / 3).max(1),
                    height: (height / 2).max(1),
                },
                ImageRegion::pixel(width - 1, height - 1),
                ImageRegion::pixel(0, 0),
                ImageRegion {
                    x: width / 2,
                    y: 0,
                    width: width - width / 2,
                    height,
                },
            ];
            for region in regions {
                check_partial_update(width, height, region);
            }
        }
    }

    fn check_partial_update(width: u32, height: u32, region: ImageRegion) {
        {
            let mut image = gradient(width, height);
            let mut mips = build_lower_mips(&image);

            for y in region.y..region.y + region.height {
                for x in region.x..region.x + region.width {
                    let offset = ((y * width + x) * 4) as usize;
                    image.rgba[offset..offset + 3].copy_from_slice(&[250, 10, 90]);
                }
            }
            update_lower_mips(&mut mips, &image, region);

            let expected = build_lower_mips(&image);
            assert_eq!(mips.len(), expected.len());
            for (actual, expected) in mips.iter().zip(&expected) {
                assert_eq!(actual.rgba, expected.rgba, "{width}x{height} {region:?}");
            }
        }
    }
}
