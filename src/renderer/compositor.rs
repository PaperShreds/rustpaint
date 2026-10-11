//! Layer compositing on the GPU.
//!
//! Each material slot keeps its layer textures on the GPU and a composite
//! texture the mesh shader samples. Recompositing blends the visible layers
//! one at a time into a float scratch texture (ping-ponging between two, since
//! blend modes need to read what's below), copies the result into the
//! composite, and renders its mip chain. Changing opacity, visibility or blend
//! mode costs a few draw calls instead of a pass over every pixel on the CPU.

use crate::document::{BlendMode, Encoding};
use crate::mesh::{Image, ImageRegion};
use crate::renderer::texture;
use eframe::egui_wgpu::wgpu::{self, util::DeviceExt as _};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;
use std::sync::Mutex;

/// Precise, linear format layers are blended in.
const SCRATCH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Most layers one slot can blend in a single composite.
const MAX_DRAWS: usize = 1024;
/// Spacing of per-draw entries in the parameter buffer (the dynamic-offset alignment).
const PARAMS_STRIDE: u64 = 256;

/// Texture format for a channel: colour is sRGB (decoded to linear when
/// read), data such as roughness is stored as is.
fn format_for(encoding: Encoding) -> wgpu::TextureFormat {
    match encoding {
        Encoding::Srgb => wgpu::TextureFormat::Rgba8UnormSrgb,
        Encoding::Linear => wgpu::TextureFormat::Rgba8Unorm,
    }
}

/// One layer to blend, bottom to top.
pub struct Draw {
    pub layer: u64,
    pub opacity: f32,
    pub blend: BlendMode,
}

pub struct Compositor {
    /// One texture, no sampler: layer textures and the "below" scratch.
    texture_layout: wgpu::BindGroupLayout,
    /// Texture plus sampler: sources of the resolve and mip passes.
    source_layout: wgpu::BindGroupLayout,
    layer_pipeline: wgpu::RenderPipeline,
    /// Resolve/mip pipeline per composite format.
    copy_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    params_buffer: wgpu::Buffer,
    params_bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    max_texture_size: u32,
    /// Scratch texture pairs, shared by all slots of the same size.
    scratch: Mutex<HashMap<(u32, u32), Scratch>>,
}

struct Scratch {
    views: [wgpu::TextureView; 2],
    /// Reads scratch `i` as the "below" input of a layer pass.
    below: [wgpu::BindGroup; 2],
    /// Reads scratch `i` as the source of the resolve pass.
    source: [wgpu::BindGroup; 2],
}

/// The GPU side of one channel of one material slot.
pub struct GpuSlot {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    /// False when the slot is too big for this GPU; it then shows a 1x1 blank.
    usable: bool,
    /// Kept alive by its views; read directly only by the read-back test.
    #[cfg_attr(not(test), allow(dead_code))]
    composite: wgpu::Texture,
    /// All mips of the composite, for the mesh shader.
    pub view: wgpu::TextureView,
    level_views: Vec<wgpu::TextureView>,
    /// `mip_sources[i]` samples level `i`, to render level `i + 1`.
    mip_sources: Vec<wgpu::BindGroup>,
    layers: HashMap<u64, GpuLayer>,
}

struct GpuLayer {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
}

impl Compositor {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("composite_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/composite.wgsl").into()),
        });
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                multisampled: false,
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_texture_layout"),
            entries: &[texture_entry(0)],
        });
        let params_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_params_layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(16),
                },
                count: None,
            }],
        });
        let source_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_source_layout"),
            entries: &[
                texture_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline = |label, layouts: &[Option<&wgpu::BindGroupLayout>], entry, format| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_full_screen"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let layer_pipeline = pipeline(
            "composite_layer_pipeline",
            &[
                Some(&texture_layout),
                Some(&params_layout),
                Some(&texture_layout),
            ],
            "fs_layer",
            SCRATCH_FORMAT,
        );
        let copy_pipelines = [Encoding::Srgb, Encoding::Linear]
            .into_iter()
            .map(|encoding| {
                let format = format_for(encoding);
                let pipeline = pipeline(
                    "composite_copy_pipeline",
                    &[Some(&source_layout)],
                    "fs_copy",
                    format,
                );
                (format, pipeline)
            })
            .collect();

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite_params"),
            size: MAX_DRAWS as u64 * PARAMS_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_params_bind_group"),
            layout: &params_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &params_buffer,
                    offset: 0,
                    size: NonZeroU64::new(16),
                }),
            }],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("composite_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            texture_layout,
            source_layout,
            layer_pipeline,
            copy_pipelines,
            params_buffer,
            params_bind_group,
            sampler,
            max_texture_size: device.limits().max_texture_dimension_2d,
            scratch: Mutex::new(HashMap::new()),
        }
    }

    fn texture_bind_group(
        &self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_texture"),
            layout: &self.texture_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(view),
            }],
        })
    }

    fn source_bind_group(
        &self,
        device: &wgpu::Device,
        view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_source"),
            layout: &self.source_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    /// A slot whose layers are `width` x `height`, stored with `encoding`.
    pub fn create_slot(
        &self,
        device: &wgpu::Device,
        width: u32,
        height: u32,
        encoding: Encoding,
    ) -> GpuSlot {
        let usable = width > 0
            && height > 0
            && width <= self.max_texture_size
            && height <= self.max_texture_size;
        let (texture_width, texture_height) = if usable { (width, height) } else { (1, 1) };
        let format = format_for(encoding);
        let levels = 32 - texture_width.max(texture_height).leading_zeros();
        let composite = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("slot_composite"),
            size: wgpu::Extent3d {
                width: texture_width,
                height: texture_height,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let level_views: Vec<_> = (0..levels)
            .map(|level| {
                composite.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let mip_sources = level_views[..level_views.len() - 1]
            .iter()
            .map(|view| self.source_bind_group(device, view))
            .collect();
        GpuSlot {
            width,
            height,
            format,
            usable,
            view: composite.create_view(&wgpu::TextureViewDescriptor::default()),
            composite,
            level_views,
            mip_sources,
            layers: HashMap::new(),
        }
    }

    /// Copies `region` of `image` into the slot's texture for layer `id`
    /// (all of it if the layer is new on the GPU).
    pub fn upload_layer(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: &mut GpuSlot,
        id: u64,
        image: &Image,
        region: Option<ImageRegion>,
    ) {
        if !slot.usable || (image.width, image.height) != (slot.width, slot.height) {
            return;
        }
        if let Some(layer) = slot.layers.get(&id) {
            let region = region.unwrap_or(ImageRegion {
                x: 0,
                y: 0,
                width: image.width,
                height: image.height,
            });
            texture::write_region(queue, &layer.texture, image, region);
            return;
        }
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("layer_texture"),
                size: wgpu::Extent3d {
                    width: image.width,
                    height: image.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: slot.format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &image.rgba,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = self.texture_bind_group(device, &view);
        slot.layers.insert(
            id,
            GpuLayer {
                texture,
                bind_group,
            },
        );
    }

    /// Frees GPU textures of layers not in `keep`.
    pub fn retain_layers(&self, slot: &mut GpuSlot, keep: &HashSet<u64>) {
        slot.layers.retain(|id, _| keep.contains(id));
    }

    pub fn has_layer(&self, slot: &GpuSlot, id: u64) -> bool {
        slot.layers.contains_key(&id)
    }

    /// Blends `draws` (bottom to top) into the slot's composite and rebuilds
    /// its mips. Layers missing on the GPU are skipped.
    pub fn composite(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: &GpuSlot,
        draws: &[Draw],
    ) {
        let draws: Vec<(&GpuLayer, &Draw)> = draws
            .iter()
            .filter(|_| slot.usable)
            .filter_map(|draw| Some((slot.layers.get(&draw.layer)?, draw)))
            .take(MAX_DRAWS)
            .collect();
        let mut params = vec![0u8; draws.len() * PARAMS_STRIDE as usize];
        for (index, (_, draw)) in draws.iter().enumerate() {
            let start = index * PARAMS_STRIDE as usize;
            params[start..start + 4].copy_from_slice(&draw.opacity.to_le_bytes());
            params[start + 4..start + 8]
                .copy_from_slice(&(draw.blend.index() as f32).to_le_bytes());
        }
        if !params.is_empty() {
            queue.write_buffer(&self.params_buffer, 0, &params);
        }

        let mut pool = self
            .scratch
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let size = (
            slot.level_views[0].texture().width(),
            slot.level_views[0].texture().height(),
        );
        let scratch = pool
            .entry(size)
            .or_insert_with(|| self.create_scratch(device, size.0, size.1));

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("composite"),
        });
        // Start from transparent, then each layer reads one scratch and writes the other.
        drop(begin_pass(
            &mut encoder,
            &scratch.views[0],
            "composite_clear",
        ));
        for (index, (layer, _)) in draws.iter().enumerate() {
            let (read, write) = (index % 2, (index + 1) % 2);
            let mut pass = begin_pass(&mut encoder, &scratch.views[write], "composite_layer");
            pass.set_pipeline(&self.layer_pipeline);
            pass.set_bind_group(0, &layer.bind_group, &[]);
            let offset = (index as u64 * PARAMS_STRIDE) as u32;
            pass.set_bind_group(1, &self.params_bind_group, &[offset]);
            pass.set_bind_group(2, &scratch.below[read], &[]);
            pass.draw(0..3, 0..1);
        }
        let result = draws.len() % 2;

        let copy = &self.copy_pipelines[&slot.format];
        let sources = std::iter::once(&scratch.source[result]).chain(&slot.mip_sources);
        for (source, target) in sources.zip(&slot.level_views) {
            let mut pass = begin_pass(&mut encoder, target, "composite_copy");
            pass.set_pipeline(copy);
            pass.set_bind_group(0, source, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }

    fn create_scratch(&self, device: &wgpu::Device, width: u32, height: u32) -> Scratch {
        let views = [0, 1].map(|_| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("composite_scratch"),
                    size: wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: SCRATCH_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        });
        Scratch {
            below: [0, 1].map(|i| self.texture_bind_group(device, &views[i])),
            source: [0, 1].map(|i| self.source_bind_group(device, &views[i])),
            views,
        }
    }
}

fn begin_pass<'a>(
    encoder: &'a mut wgpu::CommandEncoder,
    target: &'a wgpu::TextureView,
    label: &'static str,
) -> wgpu::RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

#[cfg(test)]
mod tests {
    use super::{Compositor, Draw};
    use crate::document::{BlendMode, CLEAR, Document, Encoding};
    use crate::mesh::Channel;
    use crate::mesh::{Image, Material, primitives};
    use eframe::egui_wgpu::wgpu;

    /// A headless GPU device, or `None` on machines without one.
    fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    /// Reads mip level 0 of the slot's composite back as straight-alpha RGBA.
    fn read_back(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: &super::GpuSlot,
        encoding: Encoding,
    ) -> Vec<u8> {
        let row = (slot.width * 4).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * slot.height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            slot.composite.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(slot.height),
                },
            },
            wgpu::Extent3d {
                width: slot.width,
                height: slot.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap();
        let mut pixels = Vec::new();
        for y in 0..slot.height as usize {
            let start = y * row as usize;
            pixels.extend_from_slice(&data[start..start + slot.width as usize * 4]);
        }
        // The composite is premultiplied; the CPU reference is straight alpha.
        for pixel in pixels.chunks_exact_mut(4) {
            let alpha = pixel[3] as f32 / 255.0;
            if alpha > 0.0 {
                for channel in &mut pixel[..3] {
                    *channel = encoding.encode(encoding.decode(*channel) / alpha);
                }
            }
        }
        pixels
    }

    #[test]
    fn gpu_composite_matches_the_cpu_blend_in_every_mode() {
        let Some((device, queue)) = gpu() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        // An odd size, to exercise the mip chain's edges.
        let (width, height) = (37, 21);
        let mut mesh = primitives::cube();
        mesh.materials = vec![Material::new(
            "Test",
            Image::filled(width, height, [200, 40, 40, 255]),
        )];
        let mut document = Document::new(mesh);
        // Colour and roughness paint on two layers, partly transparent.
        let mut paint = Image::filled(width, height, CLEAR);
        for (index, pixel) in paint.rgba.chunks_exact_mut(4).enumerate() {
            if index % 3 == 0 {
                pixel.copy_from_slice(&[20, 220, 90, 160]);
            }
        }
        document.layers[1].textures[0] = [Some(paint.clone()), Some(paint), None];
        document.layers[1].opacity = 0.6;
        let top = document.layers.len();
        document.insert_new_layer(top);
        let wash = Image::filled(width, height, [10, 10, 250, 60]);
        document.layers[2].textures[0] = [Some(wash.clone()), Some(wash), None];

        let compositor = Compositor::new(&device);
        for channel in [Channel::BaseColor, Channel::Roughness] {
            let encoding = channel.encoding();
            let mut slot = compositor.create_slot(&device, width, height, encoding);
            for layer in document.visible_layers(0, channel) {
                compositor.upload_layer(&device, &queue, &mut slot, layer.id, layer.image, None);
            }

            for mode in BlendMode::ALL {
                document.layers[1].blend = mode;
                document.layers[2].blend = mode;
                let draws: Vec<Draw> = document
                    .visible_layers(0, channel)
                    .into_iter()
                    .map(|layer| Draw {
                        layer: layer.id,
                        opacity: layer.opacity,
                        blend: layer.blend,
                    })
                    .collect();
                assert_eq!(draws.len(), 3);
                compositor.composite(&device, &queue, &slot, &draws);

                let gpu = read_back(&device, &queue, &slot, encoding);
                let cpu = document.composite(0, channel).unwrap().rgba;
                let worst = gpu
                    .iter()
                    .zip(&cpu)
                    .map(|(a, b)| a.abs_diff(*b))
                    .max()
                    .unwrap();
                // The 8-bit composite stores premultiplied values, so rounding differs slightly.
                assert!(
                    worst <= 3,
                    "{} / {}: GPU and CPU differ by up to {worst}",
                    channel.name(),
                    mode.name()
                );
            }
        }
    }
}
