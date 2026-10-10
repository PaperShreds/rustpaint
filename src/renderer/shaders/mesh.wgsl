struct Camera {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var base_color_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.uv = in.uv;
    return out;
}

// Linear-space colour of the lit surface.
fn shade(in: VertexOutput) -> vec3<f32> {
    let n = normalize(in.normal);
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.6));
    let diffuse = max(dot(n, light_dir), 0.0);
    let light = 0.25 + 0.75 * diffuse;

    let albedo = textureSample(base_color_texture, base_color_sampler, in.uv).rgb;
    return albedo * light;
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let low = c * 12.92;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, c <= vec3<f32>(0.0031308));
}

// Framebuffer stores gamma-encoded values (what egui prefers): encode ourselves.
@fragment
fn fs_main_gamma_framebuffer(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(linear_to_srgb(shade(in)), 1.0);
}

// sRGB framebuffer: the hardware encodes for us.
@fragment
fn fs_main_linear_framebuffer(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(shade(in), 1.0);
}