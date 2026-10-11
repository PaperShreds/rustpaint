// Draws the model. `Camera`, `PI`, `environment_uv`, `tone_map` and
// `linear_to_srgb` come from common.wgsl.

@group(0) @binding(0) var<uniform> camera: Camera;

// Layer composites, premultiplied (see composite.wgsl).
@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var roughness_texture: texture_2d<f32>;
@group(1) @binding(2) var metallic_texture: texture_2d<f32>;
@group(1) @binding(3) var material_sampler: sampler;

// Equirectangular HDR environment with a mip chain (blurrier with each level).
@group(2) @binding(0) var environment_texture: texture_2d<f32>;
@group(2) @binding(1) var environment_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world_position: vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.uv = in.uv;
    out.world_position = in.position;
    return out;
}

// Composites hold premultiplied colour; return the straight value.
fn sample_channel(texture: texture_2d<f32>, uv: vec2<f32>) -> vec3<f32> {
    let texel = textureSample(texture, material_sampler, uv);
    return select(texel.rgb / texel.a, vec3<f32>(0.0), texel.a <= 0.0);
}

// Environment light arriving from `direction`, blurred by `blur` (0 sharp, 1 fully).
fn environment(direction: vec3<f32>, blur: f32) -> vec3<f32> {
    let levels = max(camera.settings.w - 1.0, 0.0);
    let uv = environment_uv(direction, camera.settings.y);
    return sample_environment(environment_texture, environment_sampler, uv, blur * levels)
        * camera.settings.z;
}

// How much light a surface reflects, averaged over the environment
// (analytic fit of the split-sum lookup, Karis 2014).
fn environment_brdf(f0: vec3<f32>, roughness: f32, n_dot_v: f32) -> vec3<f32> {
    let c0 = vec4<f32>(-1.0, -0.0275, -0.572, 0.022);
    let c1 = vec4<f32>(1.0, 0.0425, 1.04, -0.04);
    let r = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    let ab = vec2<f32>(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

// Linear-space colour of the surface, ready for display.
fn shade(in: VertexOutput) -> vec3<f32> {
    let albedo = sample_channel(base_color_texture, in.uv);
    let roughness_value = sample_channel(roughness_texture, in.uv).r;
    let metallic = sample_channel(metallic_texture, in.uv).r;

    let mode = u32(camera.settings.x + 0.5);
    if mode == 1u {
        return albedo;
    }
    if mode == 2u {
        return vec3<f32>(roughness_value);
    }
    if mode == 3u {
        return vec3<f32>(metallic);
    }

    let roughness = clamp(roughness_value, 0.02, 1.0);
    let n = normalize(in.normal);
    let v = normalize(camera.eye.xyz - in.world_position);
    let n_dot_v = max(dot(n, v), 1e-4);
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Diffuse: light from the whole hemisphere, i.e. a very blurred environment.
    let diffuse = (1.0 - metallic) * albedo * environment(n, 0.85);
    // Specular: the reflection, blurrier the rougher the surface.
    let specular = environment(reflect(-v, n), sqrt(roughness)) * environment_brdf(f0, roughness, n_dot_v);
    return tone_map(diffuse + specular);
}

// Framebuffer stores gamma-encoded values (what egui prefers): encode ourselves.
@fragment
fn fs_main_gamma_framebuffer(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(linear_to_srgb(clamp(shade(in), vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}

// sRGB framebuffer: the hardware encodes for us.
@fragment
fn fs_main_linear_framebuffer(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(shade(in), 1.0);
}
