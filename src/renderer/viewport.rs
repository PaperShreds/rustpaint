use crate::document::{Dirty, Document};
use crate::mesh::{Mesh, SubMesh, Vertex};
use crate::renderer::compositor::{Compositor, GpuSlot};
use crate::renderer::texture;
use eframe::egui;
use eframe::egui_wgpu::wgpu::util::DeviceExt as _;
use eframe::egui_wgpu::{self, wgpu};
use std::collections::{BTreeSet, HashSet};

/// Depth buffer bits requested from eframe (see `main.rs`).
/// The pipeline's depth format is derived from the same value, so they can't drift apart.
pub const DEPTH_BUFFER_BITS: u8 = 32;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

/// One material slot: its layers and composite on the GPU, and the bind
/// group the mesh shader samples the composite through.
struct GpuMaterial {
    slot: GpuSlot,
    bind_group: wgpu::BindGroup,
}

/// GPU resources for the viewport. Lives inside egui's callback resources.
pub struct ViewportRenderer {
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    material_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    compositor: Compositor,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    submeshes: Vec<SubMesh>,
    materials: Vec<GpuMaterial>,
}

fn create_mesh_buffers(device: &wgpu::Device, mesh: &Mesh) -> (wgpu::Buffer, wgpu::Buffer) {
    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_vertices"),
        contents: bytemuck::cast_slice(&mesh.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh_indices"),
        contents: bytemuck::cast_slice(&mesh.indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    (vertex_buffer, index_buffer)
}

fn create_gpu_materials(
    render_state: &egui_wgpu::RenderState,
    compositor: &Compositor,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    document: &Document,
) -> Vec<GpuMaterial> {
    let device = &render_state.device;
    document
        .slot_sizes()
        .into_iter()
        .enumerate()
        .map(|(material, (width, height))| {
            let mut slot = compositor.create_slot(device, width, height);
            sync_slot(render_state, compositor, &mut slot, document, material);
            composite_slot(render_state, compositor, &slot, document, material);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material_bind_group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&slot.view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            GpuMaterial { slot, bind_group }
        })
        .collect()
}

/// Makes the slot hold exactly the layers that have pixels on `material`
/// (hidden ones too, so showing them again needs no upload).
fn sync_slot(
    render_state: &egui_wgpu::RenderState,
    compositor: &Compositor,
    slot: &mut GpuSlot,
    document: &Document,
    material: usize,
) {
    let mut keep = HashSet::new();
    for layer in &document.layers {
        if let Some(image) = document.layer_image(layer, material) {
            keep.insert(layer.id);
            if !compositor.has_layer(slot, layer.id) {
                let (device, queue) = (&render_state.device, &render_state.queue);
                compositor.upload_layer(device, queue, slot, layer.id, image, None);
            }
        }
    }
    compositor.retain_layers(slot, &keep);
}

fn composite_slot(
    render_state: &egui_wgpu::RenderState,
    compositor: &Compositor,
    slot: &GpuSlot,
    document: &Document,
    material: usize,
) {
    let draws: Vec<(u64, f32)> = document
        .visible_layers(material)
        .into_iter()
        .map(|(id, _, opacity)| (id, opacity))
        .collect();
    compositor.composite(&render_state.device, &render_state.queue, slot, &draws);
}

/// Replaces the model shown in the viewport.
pub fn show_document(render_state: &egui_wgpu::RenderState, document: &Document) {
    let (vertex_buffer, index_buffer) = create_mesh_buffers(&render_state.device, &document.mesh);

    let mut renderer = render_state.renderer.write();
    let Some(viewport) = renderer.callback_resources.get_mut::<ViewportRenderer>() else {
        return;
    };
    viewport.materials = create_gpu_materials(
        render_state,
        &viewport.compositor,
        &viewport.material_layout,
        &viewport.sampler,
        document,
    );
    viewport.vertex_buffer = vertex_buffer;
    viewport.index_buffer = index_buffer;
    viewport.submeshes = document.mesh.submeshes.clone();
}

/// Brings the GPU copy of `document` up to date after `dirty` changes and
/// recomposites the slots they touched.
pub fn apply_changes(render_state: &egui_wgpu::RenderState, document: &Document, dirty: &[Dirty]) {
    if dirty.is_empty() {
        return;
    }
    let mut renderer = render_state.renderer.write();
    let Some(viewport) = renderer.callback_resources.get_mut::<ViewportRenderer>() else {
        return;
    };
    let compositor = &viewport.compositor;
    let (device, queue) = (&render_state.device, &render_state.queue);

    if dirty.iter().any(|change| matches!(change, Dirty::Layers)) {
        for (material, gpu) in viewport.materials.iter_mut().enumerate() {
            sync_slot(render_state, compositor, &mut gpu.slot, document, material);
        }
    }
    let mut touched = BTreeSet::new();
    for change in dirty {
        match change {
            Dirty::Pixels {
                layer,
                material,
                region,
            } => {
                let Some(gpu) = viewport.materials.get_mut(*material) else {
                    continue;
                };
                let image = document
                    .layer_index(*layer)
                    .and_then(|index| document.layer_image(&document.layers[index], *material));
                if let Some(image) = image {
                    compositor.upload_layer(
                        device,
                        queue,
                        &mut gpu.slot,
                        *layer,
                        image,
                        Some(*region),
                    );
                    touched.insert(*material);
                }
            }
            Dirty::Layers => touched.extend(0..viewport.materials.len()),
        }
    }
    for material in touched {
        if let Some(gpu) = viewport.materials.get(material) {
            composite_slot(render_state, compositor, &gpu.slot, document, material);
        }
    }
}

/// Call once at startup, from the eframe creation closure, with the initial document.
pub fn init(cc: &eframe::CreationContext<'_>, document: &Document) {
    let Some(render_state) = cc.wgpu_render_state.as_ref() else {
        eprintln!("No wgpu render state available - is eframe's wgpu feature enabled?");
        return;
    };
    let device = &render_state.device;

    let depth_format = egui_wgpu::depth_format_from_bits(DEPTH_BUFFER_BITS, 0)
        .expect("unsupported depth buffer bit count");

    // --- Camera uniform ---
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("camera_uniform"),
        contents: bytemuck::bytes_of(&CameraUniform {
            view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera_bind_group"),
        layout: &camera_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });

    // --- Material layout: one texture + one sampler per material ---
    let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("material_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let sampler = texture::create_sampler(device);

    // --- Pipeline ---
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/mesh.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewport_pipeline_layout"),
        bind_group_layouts: &[Some(&camera_layout), Some(&material_layout)],
        immediate_size: 0,
    });

    let attributes = wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2];
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &attributes,
    };

    let fragment_entry = if render_state.target_format.is_srgb() {
        "fs_main_linear_framebuffer"
    } else {
        "fs_main_gamma_framebuffer"
    };

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("viewport_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[Some(vertex_layout)],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some(fragment_entry),
            targets: &[Some(render_state.target_format.into())],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // --- Initial mesh ---
    let (vertex_buffer, index_buffer) = create_mesh_buffers(device, &document.mesh);
    let compositor = Compositor::new(device);
    let materials = create_gpu_materials(
        render_state,
        &compositor,
        &material_layout,
        &sampler,
        document,
    );

    render_state
        .renderer
        .write()
        .callback_resources
        .insert(ViewportRenderer {
            pipeline,
            camera_buffer,
            camera_bind_group,
            material_layout,
            sampler,
            compositor,
            vertex_buffer,
            index_buffer,
            submeshes: document.mesh.submeshes.clone(),
            materials,
        });
}

/// Per-frame data handed to the GPU for the viewport rectangle.
pub struct ViewportCallback {
    pub view_proj: [[f32; 4]; 4],
}

impl egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(renderer) = resources.get::<ViewportRenderer>() {
            queue.write_buffer(
                &renderer.camera_buffer,
                0,
                bytemuck::bytes_of(&CameraUniform {
                    view_proj: self.view_proj,
                }),
            );
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(renderer) = resources.get::<ViewportRenderer>() else {
            return;
        };
        render_pass.set_pipeline(&renderer.pipeline);
        render_pass.set_bind_group(0, &renderer.camera_bind_group, &[]);
        render_pass.set_vertex_buffer(0, renderer.vertex_buffer.slice(..));
        render_pass.set_index_buffer(renderer.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        for submesh in &renderer.submeshes {
            if let Some(material) = renderer.materials.get(submesh.material) {
                render_pass.set_bind_group(1, &material.bind_group, &[]);
                render_pass.draw_indexed(submesh.indices.clone(), 0, 0..1);
            }
        }
    }
}
