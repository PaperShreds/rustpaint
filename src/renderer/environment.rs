//! The HDR environment that lights the viewport (image-based lighting).
//!
//! An equirectangular HDR image is uploaded as a float texture with a mip
//! chain. Rougher surfaces sample blurrier mips for their reflections, and a
//! very blurry mip stands in for diffuse light from all around.

use eframe::egui_wgpu::wgpu::{self, util::DeviceExt as _};
use std::path::{Path, PathBuf};

/// Larger maps are scaled down: reflections are blurred anyway, and this
/// keeps memory and loading time small.
const MAX_WIDTH: u32 = 2048;

/// Environments that ship with the program (Poly Haven, CC0; see
/// assets/environments/README.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuiltIn {
    Outdoor,
    Studio,
}

impl BuiltIn {
    pub const ALL: [BuiltIn; 2] = [BuiltIn::Outdoor, BuiltIn::Studio];

    pub fn name(self) -> &'static str {
        match self {
            BuiltIn::Outdoor => "Outdoor",
            BuiltIn::Studio => "Studio",
        }
    }

    /// Stable name for the settings file.
    pub fn key(self) -> &'static str {
        match self {
            BuiltIn::Outdoor => "outdoor",
            BuiltIn::Studio => "studio",
        }
    }

    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|built_in| built_in.key() == key)
    }

    fn hdr_file(self) -> &'static [u8] {
        match self {
            BuiltIn::Outdoor => include_bytes!("../../assets/environments/je_gray_02_1k.hdr"),
            BuiltIn::Studio => include_bytes!("../../assets/environments/studio_small_09_1k.hdr"),
        }
    }
}

/// Where the environment comes from.
#[derive(Clone, Debug, PartialEq)]
pub enum EnvironmentSource {
    BuiltIn(BuiltIn),
    File(PathBuf),
}

impl Default for EnvironmentSource {
    fn default() -> Self {
        EnvironmentSource::BuiltIn(BuiltIn::Outdoor)
    }
}

impl EnvironmentSource {
    pub fn load(&self) -> Result<HdrImage, String> {
        match self {
            EnvironmentSource::BuiltIn(built_in) => HdrImage::built_in(*built_in),
            EnvironmentSource::File(path) => HdrImage::load(path),
        }
    }

    pub fn name(&self) -> String {
        match self {
            EnvironmentSource::BuiltIn(built_in) => built_in.name().to_string(),
            EnvironmentSource::File(path) => path.file_name().map_or_else(
                || path.display().to_string(),
                |name| name.to_string_lossy().into(),
            ),
        }
    }
}

/// An equirectangular HDR image: linear RGBA floats.
pub struct HdrImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<f32>,
}

impl HdrImage {
    pub fn built_in(built_in: BuiltIn) -> Result<Self, String> {
        let image =
            image::load_from_memory_with_format(built_in.hdr_file(), image::ImageFormat::Hdr)
                .map_err(|e| e.to_string())?;
        Ok(Self::from_dynamic(image))
    }

    /// Loads a `.hdr` or `.exr` file.
    pub fn load(path: &Path) -> Result<Self, String> {
        let image = image::open(path).map_err(|e| e.to_string())?;
        Ok(Self::from_dynamic(image))
    }

    fn from_dynamic(image: image::DynamicImage) -> Self {
        let image = if image.width() > MAX_WIDTH {
            let height = (MAX_WIDTH as u64 * image.height() as u64 / image.width() as u64).max(1);
            image.resize_exact(
                MAX_WIDTH,
                height as u32,
                image::imageops::FilterType::Triangle,
            )
        } else {
            image
        };
        let image = image.into_rgba32f();
        let (width, height) = image.dimensions();
        let mut rgba = image.into_raw();
        // Infinite or NaN pixels (seen in some HDRs) would poison every mip.
        for value in &mut rgba {
            if !value.is_finite() || *value < 0.0 {
                *value = 0.0;
            }
        }
        Self {
            width,
            height,
            rgba,
        }
    }

    /// Each level half the size of the previous one, down to 1x1. Levels
    /// are filtered with a tent (1 3 3 1) rather than a 2x2 box, so the
    /// blurry levels come out smooth instead of blocky. The map wraps around
    /// horizontally.
    fn mip_chain(&self) -> Vec<(u32, u32, Vec<f32>)> {
        let mut levels = vec![(self.width, self.height, self.rgba.clone())];
        while let Some((w, h, previous)) = levels.last().filter(|(w, h, _)| *w > 1 || *h > 1) {
            let (w, h) = (*w as i64, *h as i64);
            let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
            const WEIGHTS: [f32; 4] = [1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0];
            // Taps -1..=2 around 2x (just 0 when the axis doesn't shrink).
            let taps = |x: i64, shrinks: bool| -> Vec<(i64, f32)> {
                if shrinks {
                    (0..4)
                        .map(|i| (x * 2 + i - 1, WEIGHTS[i as usize]))
                        .collect()
                } else {
                    vec![(x, 1.0)]
                }
            };
            // Horizontal pass: wraps around.
            let mut across = vec![0.0f32; (nw * h * 4) as usize];
            for y in 0..h {
                for x in 0..nw {
                    for (sx, weight) in taps(x, w > 1) {
                        let from = ((y * w + sx.rem_euclid(w)) * 4) as usize;
                        let to = ((y * nw + x) * 4) as usize;
                        for c in 0..4 {
                            across[to + c] += previous[from + c] * weight;
                        }
                    }
                }
            }
            // Vertical pass: clamps at the poles.
            let mut next = vec![0.0f32; (nw * nh * 4) as usize];
            for y in 0..nh {
                for (sy, weight) in taps(y, h > 1) {
                    let sy = sy.clamp(0, h - 1);
                    for x in 0..nw {
                        let from = ((sy * nw + x) * 4) as usize;
                        let to = ((y * nw + x) * 4) as usize;
                        for c in 0..4 {
                            next[to + c] += across[from + c] * weight;
                        }
                    }
                }
            }
            levels.push((nw as u32, nh as u32, next));
        }
        levels
    }
}

/// Uploads `image` as a half-float texture with its full mip chain.
/// Returns the texture and its number of mip levels.
pub fn create_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    image: &HdrImage,
) -> (wgpu::Texture, u32) {
    let levels = image.mip_chain();
    let data: Vec<u8> = levels
        .iter()
        .flat_map(|(_, _, values)| values.iter())
        .flat_map(|value| half::f16::from_f32(*value).to_le_bytes())
        .collect();
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("environment"),
            size: wgpu::Extent3d {
                width: image.width,
                height: image.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &data,
    );
    (texture, levels.len() as u32)
}

/// Wraps around horizontally, clamps at the poles, blends between mips.
pub fn create_sampler(device: &wgpu::Device) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("environment_sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    })
}

#[cfg(test)]
mod tests {
    use super::{BuiltIn, HdrImage};

    #[test]
    fn default_environment_decodes_with_a_full_mip_chain() {
        let image = HdrImage::built_in(BuiltIn::Studio).unwrap();
        assert_eq!((image.width, image.height), (1024, 512));
        assert!(image.rgba.iter().all(|v| v.is_finite() && *v >= 0.0));
        // A studio: some pixels are brighter than white (lights), which is
        // what makes it HDR.
        assert!(image.rgba.iter().any(|v| *v > 1.0));

        let mips = image.mip_chain();
        assert_eq!(mips.len(), 11); // 1024 wide -> 1x1
        let (w, h, last) = mips.last().unwrap();
        assert_eq!((*w, *h), (1, 1));
        // The 1x1 level is the image's average brightness.
        let average = image.rgba.iter().step_by(4).sum::<f32>() / (1024.0 * 512.0);
        assert!((last[0] - average).abs() / average < 0.05);
    }

    #[test]
    fn every_built_in_environment_loads() {
        for built_in in BuiltIn::ALL {
            let image = HdrImage::built_in(built_in).unwrap();
            assert!(image.width >= 512, "{}", built_in.name());
            assert_eq!(BuiltIn::from_key(built_in.key()), Some(built_in));
        }
    }

    #[test]
    fn blurry_levels_wrap_around_horizontally() {
        // One bright column at the left edge of an 8x2 map.
        let mut rgba = vec![0.0; 8 * 2 * 4];
        for y in 0..2 {
            rgba[y * 8 * 4..y * 8 * 4 + 4].copy_from_slice(&[8.0; 4]);
        }
        let image = HdrImage {
            width: 8,
            height: 2,
            rgba,
        };
        let mips = image.mip_chain();
        let (w, _, level) = &mips[1];
        assert_eq!(*w, 4);
        // It spreads to the right edge too (its neighbour across the seam),
        // and the total brightness is kept.
        assert!(level[0] > 0.0 && level[3 * 4] > 0.0);
        assert_eq!(level[2 * 4], 0.0);
        let total: f32 = level.iter().step_by(4).sum();
        assert!((total * 4.0 - 16.0).abs() < 1e-4, "{total}");
    }
}
