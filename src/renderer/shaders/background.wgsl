// The environment drawn behind the model, blurred as much as the user likes.

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var environment_texture: texture_2d<f32>;
@group(1) @binding(1) var environment_sampler: sampler;

struct BackgroundOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_background(@builtin(vertex_index) index: u32) -> BackgroundOutput {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u)) * 2.0 - 1.0;
    var out: BackgroundOutput;
    out.position = vec4<f32>(corner, 1.0, 1.0);
    out.ndc = corner;
    return out;
}

fn background(in: BackgroundOutput) -> vec3<f32> {
    let far = camera.inverse_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let direction = far.xyz / far.w - camera.eye.xyz;
    let uv = environment_uv(direction, camera.settings.y);
    // Blurred, so the model stands out from it. At most 6 levels down (1/64 size).
    let lod = camera.background.x * min(6.0, camera.settings.w - 1.0);
    let color = sample_environment(environment_texture, environment_sampler, uv, lod);
    return tone_map(color * camera.settings.z);
}

@fragment
fn fs_main_gamma_framebuffer(in: BackgroundOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(linear_to_srgb(background(in)), 1.0);
}

@fragment
fn fs_main_linear_framebuffer(in: BackgroundOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(background(in), 1.0);
}
