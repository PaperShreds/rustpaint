//! Layer compositing on the GPU.
//!
//! Each material slot keeps its layer textures on the GPU and a composite
//! texture the mesh shader samples. Recompositing draws the visible layers
//! into the composite (one full-screen triangle each) and then renders its
//! mip chain, so changing opacity or visibility costs a few draw calls instead
//! of a pass over every pixel on the CPU.

use crate::mesh::{Image, ImageRegion};
use crate::renderer::texture;
use eframe::egui_wgpu::wgpu::{self, util::DeviceExt as _};
use std::collections::{HashMap, HashSet};
use std::num::NonZeroU64;

/// sRGB, so blending and filtering happen in linear light.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// Most layers one slot can blend in a single composite.
const MAX_DRAWS: usize = 1024;
/// Spacing of per-draw entries in the opacity buffer (the dynamic-offset alignment).
const OPACITY_STRIDE: u64 = 256;

pub struct Compositor {
    layer_layout: wgpu::BindGroupLayout,
    mip_layout: wgpu::BindGroupLayout,
    layer_pipeline: wgpu::RenderPipeline,
    mip_pipeline: wgpu::RenderPipeline,
    opacity_buffer: wgpu::Buffer,
    opacity_bind_group: wgpu::BindGroup,
    mip_sampler: wgpu::Sampler,
    max_texture_size: u32,
}

/// The GPU side of one material slot.
pub struct GpuSlot {
    width: u32,
    height: u32,
    /// False when the slot is too big for this GPU; it then shows a 1x1 grey.
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
        let layer_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_layer_layout"),
            entries: &[texture_entry(0)],
        });
        let opacity_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_opacity_layout"),
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
        let mip_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite_mip_layout"),
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

        let premultiplied_over = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = |label, layouts: &[Option<&wgpu::BindGroupLayout>], entry, blend| {
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
                        format: FORMAT,
                        blend,
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
            &[Some(&layer_layout), Some(&opacity_layout)],
            "fs_layer",
            Some(wgpu::BlendState {
                color: premultiplied_over,
                alpha: premultiplied_over,
            }),
        );
        let mip_pipeline = pipeline(
            "composite_mip_pipeline",
            &[Some(&mip_layout)],
            "fs_mip",
            None,
        );

        let opacity_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("composite_opacity"),
            size: MAX_DRAWS as u64 * OPACITY_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let opacity_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("composite_opacity_bind_group"),
            layout: &opacity_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &opacity_buffer,
                    offset: 0,
                    size: NonZeroU64::new(16),
                }),
            }],
        });
        let mip_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("composite_mip_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            layer_layout,
            mip_layout,
            layer_pipeline,
            mip_pipeline,
            opacity_buffer,
            opacity_bind_group,
            mip_sampler,
            max_texture_size: device.limits().max_texture_dimension_2d,
        }
    }

    /// A slot whose layers are `width` x `height`.
    pub fn create_slot(&self, device: &wgpu::Device, width: u32, height: u32) -> GpuSlot {
        let usable = width > 0
            && height > 0
            && width <= self.max_texture_size
            && height <= self.max_texture_size;
        let (texture_width, texture_height) = if usable { (width, height) } else { (1, 1) };
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
            format: FORMAT,
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
            .map(|view| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("composite_mip_source"),
                    layout: &self.mip_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.mip_sampler),
                        },
                    ],
                })
            })
            .collect();
        GpuSlot {
            width,
            height,
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
                format: FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &image.rgba,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_bind_group"),
            layout: &self.layer_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
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

    /// Blends `draws` (layer id, opacity; bottom to top) into the slot's
    /// composite and rebuilds its mips. Layers missing on the GPU are skipped.
    pub fn composite(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: &GpuSlot,
        draws: &[(u64, f32)],
    ) {
        let draws: Vec<(&GpuLayer, f32)> = draws
            .iter()
            .filter(|_| slot.usable)
            .filter_map(|(id, opacity)| Some((slot.layers.get(id)?, *opacity)))
            .take(MAX_DRAWS)
            .collect();
        let mut opacities = vec![0u8; draws.len() * OPACITY_STRIDE as usize];
        for (index, (_, opacity)) in draws.iter().enumerate() {
            let start = index * OPACITY_STRIDE as usize;
            opacities[start..start + 4].copy_from_slice(&opacity.to_le_bytes());
        }
        if !opacities.is_empty() {
            queue.write_buffer(&self.opacity_buffer, 0, &opacities);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("composite"),
        });
        {
            let mut pass = begin_pass(&mut encoder, &slot.level_views[0], "composite_layers");
            pass.set_pipeline(&self.layer_pipeline);
            for (index, (layer, _)) in draws.iter().enumerate() {
                pass.set_bind_group(0, &layer.bind_group, &[]);
                let offset = (index as u64 * OPACITY_STRIDE) as u32;
                pass.set_bind_group(1, &self.opacity_bind_group, &[offset]);
                pass.draw(0..3, 0..1);
            }
        }
        for (source, target) in slot.mip_sources.iter().zip(&slot.level_views[1..]) {
            let mut pass = begin_pass(&mut encoder, target, "composite_mip");
            pass.set_pipeline(&self.mip_pipeline);
            pass.set_bind_group(0, source, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
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
    use super::Compositor;
    use crate::document::{CLEAR, Document};
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
    fn read_back(device: &wgpu::Device, queue: &wgpu::Queue, slot: &super::GpuSlot) -> Vec<u8> {
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
                    let linear = crate::mesh::srgb_to_linear(*channel) / alpha;
                    *channel = crate::mesh::linear_to_srgb(linear);
                }
            }
        }
        pixels
    }

    #[test]
    fn gpu_composite_matches_the_cpu_blend() {
        let Some((device, queue)) = gpu() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        // An odd size, to exercise the mip chain's edges.
        let (width, height) = (37, 21);
        let mut mesh = primitives::cube();
        mesh.materials = vec![Material {
            name: "Test".into(),
            base_color: Image::filled(width, height, [200, 40, 40, 255]),
        }];
        let mut document = Document::new(mesh);
        let mut paint = Image::filled(width, height, CLEAR);
        for (index, pixel) in paint.rgba.chunks_exact_mut(4).enumerate() {
            if index % 3 == 0 {
                pixel.copy_from_slice(&[20, 220, 90, 160]);
            }
        }
        document.layers[1].textures[0] = Some(paint);
        document.layers[1].opacity = 0.6;
        let top = document.layers.len();
        document.insert_new_layer(top);
        document.layers[2].textures[0] = Some(Image::filled(width, height, [10, 10, 250, 60]));

        let compositor = Compositor::new(&device);
        let mut slot = compositor.create_slot(&device, width, height);
        let draws: Vec<(u64, f32)> = document
            .visible_layers(0)
            .into_iter()
            .map(|(id, image, opacity)| {
                compositor.upload_layer(&device, &queue, &mut slot, id, image, None);
                (id, opacity)
            })
            .collect();
        assert_eq!(draws.len(), 3);
        compositor.composite(&device, &queue, &slot, &draws);

        let gpu = read_back(&device, &queue, &slot);
        let cpu = document.composite(0).unwrap().rgba;
        let worst = gpu
            .iter()
            .zip(&cpu)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        // 8-bit storage and premultiplication round slightly differently.
        assert!(worst <= 3, "GPU and CPU composites differ by up to {worst}");
    }
}
