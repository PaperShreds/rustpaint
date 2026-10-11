use crate::document::{Channel, Dirty, Document, LayerMask, Plane};
use crate::mesh::{Mesh, SubMesh, Vertex};
use crate::renderer::compositor::{Compositor, Draw, DrawMask, GpuMasks, GpuSlot};
use crate::renderer::environment::{self, HdrImage};
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
    /// For the background: screen position back to a view direction.
    inverse_view_proj: [[f32; 4]; 4],
    /// xyz = camera position.
    eye: [f32; 4],
    /// x = `ViewMode` as a number, y = environment rotation (radians),
    /// z = environment intensity, w = environment mip count.
    settings: [f32; 4],
    /// x = background blur, 0..1.
    background: [f32; 4],
}

/// How the environment lights and shows in the viewport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvironmentSettings {
    /// Turns the environment around the vertical axis, in degrees.
    pub rotation_degrees: f32,
    pub intensity: f32,
    /// Draw the (blurred) environment behind the model.
    pub show_background: bool,
    /// How blurry the background is: 0 sharp, 1 very blurry.
    pub background_blur: f32,
}

impl Default for EnvironmentSettings {
    fn default() -> Self {
        Self {
            rotation_degrees: 0.0,
            intensity: 1.0,
            show_background: true,
            background_blur: 0.5,
        }
    }
}

/// What the viewport shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ViewMode {
    /// The material, lit.
    #[default]
    Lit,
    /// One channel, unlit.
    Channel(Channel),
}

impl ViewMode {
    pub const ALL: [ViewMode; 4] = [
        ViewMode::Lit,
        ViewMode::Channel(Channel::BaseColor),
        ViewMode::Channel(Channel::Roughness),
        ViewMode::Channel(Channel::Metallic),
    ];

    pub fn name(self) -> &'static str {
        match self {
            ViewMode::Lit => "Material",
            ViewMode::Channel(channel) => channel.name(),
        }
    }

    /// Matches `settings.x` in `mesh.wgsl`.
    fn index(self) -> f32 {
        match self {
            ViewMode::Lit => 0.0,
            ViewMode::Channel(channel) => 1.0 + channel.index() as f32,
        }
    }
}

/// One material slot: per channel, its layers and composite on the GPU; and
/// the bind group the mesh shader samples the composites through.
struct GpuMaterial {
    slots: [GpuSlot; Channel::COUNT],
    /// Layer masks, shared by the channels.
    masks: GpuMasks,
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
    background_pipeline: wgpu::RenderPipeline,
    environment_layout: wgpu::BindGroupLayout,
    environment_sampler: wgpu::Sampler,
    environment_bind_group: wgpu::BindGroup,
    environment_levels: u32,
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
            let mut masks = compositor.create_masks(width, height);
            sync_masks(
                render_state,
                compositor,
                &mut masks,
                document,
                material,
                None,
            );
            let slots = Channel::ALL.map(|channel| {
                let mut slot = compositor.create_slot(device, width, height, channel.encoding());
                sync_slot(
                    render_state,
                    compositor,
                    &mut slot,
                    document,
                    material,
                    channel,
                );
                composite_slot(
                    render_state,
                    compositor,
                    &slot,
                    &masks,
                    document,
                    material,
                    channel,
                );
                slot
            });
            let texture = |channel: Channel| wgpu::BindGroupEntry {
                binding: channel.index() as u32,
                resource: wgpu::BindingResource::TextureView(&slots[channel.index()].view),
            };
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("material_bind_group"),
                layout,
                entries: &[
                    texture(Channel::BaseColor),
                    texture(Channel::Roughness),
                    texture(Channel::Metallic),
                    wgpu::BindGroupEntry {
                        binding: Channel::COUNT as u32,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            });
            GpuMaterial {
                slots,
                masks,
                bind_group,
            }
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
    channel: Channel,
) {
    let mut keep = HashSet::new();
    for layer in &document.layers {
        if let Some(image) = document.layer_image(layer, material, channel) {
            keep.insert(layer.id);
            if !compositor.has_layer(slot, layer.id) {
                let (device, queue) = (&render_state.device, &render_state.queue);
                compositor.upload_layer(device, queue, slot, layer.id, image, None);
            }
        }
    }
    compositor.retain_layers(slot, &keep);
}

/// Makes `masks` hold the painted masks of every layer on `material`. The
/// mask of layer `replaced` (swapped out as a whole) is uploaded again in full.
fn sync_masks(
    render_state: &egui_wgpu::RenderState,
    compositor: &Compositor,
    masks: &mut GpuMasks,
    document: &Document,
    material: usize,
    replaced: Option<u64>,
) {
    let mut keep = HashSet::new();
    for layer in &document.layers {
        if let LayerMask::Image(image) = document.layer_mask(layer, material) {
            keep.insert(layer.id);
            if replaced == Some(layer.id) || !compositor.has_mask(masks, layer.id) {
                let (device, queue) = (&render_state.device, &render_state.queue);
                compositor.upload_mask(device, queue, masks, layer.id, image, None);
            }
        }
    }
    compositor.retain_masks(masks, &keep);
}

fn composite_slot(
    render_state: &egui_wgpu::RenderState,
    compositor: &Compositor,
    slot: &GpuSlot,
    masks: &GpuMasks,
    document: &Document,
    material: usize,
    channel: Channel,
) {
    let draws: Vec<Draw> = document
        .visible_layers(material, channel)
        .into_iter()
        .map(|layer| Draw {
            layer: layer.id,
            opacity: layer.opacity,
            blend: layer.blend,
            mask: match layer.mask {
                LayerMask::None => DrawMask::Uniform(1.0),
                LayerMask::Uniform(value) => DrawMask::Uniform(value as f32 / 255.0),
                LayerMask::Image(_) => DrawMask::Texture,
            },
        })
        .collect();
    let (device, queue) = (&render_state.device, &render_state.queue);
    compositor.composite(device, queue, slot, masks, &draws);
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

    let replaced: Vec<u64> = dirty
        .iter()
        .filter_map(|change| match change {
            Dirty::Mask { layer } => Some(*layer),
            _ => None,
        })
        .collect();
    let layers_changed = dirty.iter().any(|change| matches!(change, Dirty::Layers));
    if layers_changed || !replaced.is_empty() {
        for (material, gpu) in viewport.materials.iter_mut().enumerate() {
            for (slot, channel) in gpu.slots.iter_mut().zip(Channel::ALL) {
                sync_slot(render_state, compositor, slot, document, material, channel);
            }
            sync_masks(
                render_state,
                compositor,
                &mut gpu.masks,
                document,
                material,
                None,
            );
            for layer in &replaced {
                sync_masks(
                    render_state,
                    compositor,
                    &mut gpu.masks,
                    document,
                    material,
                    Some(*layer),
                );
            }
        }
    }
    let mut touched = BTreeSet::new();
    for change in dirty {
        match change {
            Dirty::Pixels {
                layer,
                material,
                plane: Plane::Mask,
                region,
            } => {
                let Some(gpu) = viewport.materials.get_mut(*material) else {
                    continue;
                };
                let mask = document
                    .layer_index(*layer)
                    .map(|index| document.layer_mask(&document.layers[index], *material));
                if let Some(LayerMask::Image(image)) = mask {
                    compositor.upload_mask(
                        device,
                        queue,
                        &mut gpu.masks,
                        *layer,
                        image,
                        Some(*region),
                    );
                    // A mask shows in every channel.
                    touched.extend(Channel::ALL.map(|channel| (*material, channel)));
                }
            }
            Dirty::Pixels {
                layer,
                material,
                plane: Plane::Channel(channel),
                region,
            } => {
                let Some(gpu) = viewport.materials.get_mut(*material) else {
                    continue;
                };
                let image = document.layer_index(*layer).and_then(|index| {
                    document.layer_image(&document.layers[index], *material, *channel)
                });
                if let Some(image) = image {
                    compositor.upload_layer(
                        device,
                        queue,
                        &mut gpu.slots[channel.index()],
                        *layer,
                        image,
                        Some(*region),
                    );
                    touched.insert((*material, *channel));
                }
            }
            Dirty::Layers | Dirty::Mask { .. } => {
                for material in 0..viewport.materials.len() {
                    touched.extend(Channel::ALL.map(|channel| (material, channel)));
                }
            }
        }
    }
    for (material, channel) in touched {
        if let Some(gpu) = viewport.materials.get(material) {
            let slot = &gpu.slots[channel.index()];
            composite_slot(
                render_state,
                compositor,
                slot,
                &gpu.masks,
                document,
                material,
                channel,
            );
        }
    }
}

fn create_environment(
    render_state: &egui_wgpu::RenderState,
    layout: &wgpu::BindGroupLayout,
    sampler: &wgpu::Sampler,
    image: &HdrImage,
) -> (wgpu::BindGroup, u32) {
    let device = &render_state.device;
    let (texture, levels) = environment::create_texture(device, &render_state.queue, image);
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("environment_bind_group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    });
    (bind_group, levels)
}

/// Replaces the environment that lights the viewport.
pub fn set_environment(render_state: &egui_wgpu::RenderState, image: &HdrImage) {
    let mut renderer = render_state.renderer.write();
    let Some(viewport) = renderer.callback_resources.get_mut::<ViewportRenderer>() else {
        return;
    };
    let (bind_group, levels) = create_environment(
        render_state,
        &viewport.environment_layout,
        &viewport.environment_sampler,
        image,
    );
    viewport.environment_bind_group = bind_group;
    viewport.environment_levels = levels;
}

/// Call once at startup, from the eframe creation closure, with the initial
/// document and environment.
pub fn init(cc: &eframe::CreationContext<'_>, document: &Document, environment: &HdrImage) {
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
            inverse_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            eye: [0.0; 4],
            settings: [0.0; 4],
            background: [0.0; 4],
        }),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("camera_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
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
    // One composite per channel (base colour, roughness, metallic), then the sampler.
    let channel_texture = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    };
    let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("material_layout"),
        entries: &[
            channel_texture(0),
            channel_texture(1),
            channel_texture(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let sampler = texture::create_sampler(device);
    let environment_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("environment_layout"),
        entries: &[
            channel_texture(0),
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    // --- Pipeline ---
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mesh_shader"),
        source: wgpu::ShaderSource::Wgsl(
            concat!(
                include_str!("shaders/common.wgsl"),
                include_str!("shaders/mesh.wgsl")
            )
            .into(),
        ),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("viewport_pipeline_layout"),
        bind_group_layouts: &[
            Some(&camera_layout),
            Some(&material_layout),
            Some(&environment_layout),
        ],
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

    // The environment behind the model: drawn first, at the far plane.
    let background_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("background_shader"),
        source: wgpu::ShaderSource::Wgsl(
            concat!(
                include_str!("shaders/common.wgsl"),
                include_str!("shaders/background.wgsl")
            )
            .into(),
        ),
    });
    let background_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("background_pipeline_layout"),
        bind_group_layouts: &[Some(&camera_layout), Some(&environment_layout)],
        immediate_size: 0,
    });
    let background_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("background_pipeline"),
        layout: Some(&background_layout),
        vertex: wgpu::VertexState {
            module: &background_shader,
            entry_point: Some("vs_background"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &background_shader,
            entry_point: Some(fragment_entry),
            targets: &[Some(render_state.target_format.into())],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: depth_format,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });
    let environment_sampler = environment::create_sampler(device);
    let (environment_bind_group, environment_levels) = create_environment(
        render_state,
        &environment_layout,
        &environment_sampler,
        environment,
    );

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
            background_pipeline,
            environment_layout,
            environment_sampler,
            environment_bind_group,
            environment_levels,
            vertex_buffer,
            index_buffer,
            submeshes: document.mesh.submeshes.clone(),
            materials,
        });
}

/// Per-frame data handed to the GPU for the viewport rectangle.
pub struct ViewportCallback {
    pub view_proj: glam::Mat4,
    pub eye: [f32; 3],
    pub view_mode: ViewMode,
    pub environment: EnvironmentSettings,
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
                    view_proj: self.view_proj.to_cols_array_2d(),
                    inverse_view_proj: self.view_proj.inverse().to_cols_array_2d(),
                    eye: [self.eye[0], self.eye[1], self.eye[2], 1.0],
                    settings: [
                        self.view_mode.index(),
                        self.environment.rotation_degrees.to_radians(),
                        self.environment.intensity.max(0.0),
                        renderer.environment_levels as f32,
                    ],
                    background: [
                        self.environment.background_blur.clamp(0.0, 1.0),
                        0.0,
                        0.0,
                        0.0,
                    ],
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
        render_pass.set_bind_group(0, &renderer.camera_bind_group, &[]);
        if self.environment.show_background {
            render_pass.set_pipeline(&renderer.background_pipeline);
            render_pass.set_bind_group(1, &renderer.environment_bind_group, &[]);
            render_pass.draw(0..3, 0..1);
        }
        render_pass.set_pipeline(&renderer.pipeline);
        render_pass.set_bind_group(2, &renderer.environment_bind_group, &[]);
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
