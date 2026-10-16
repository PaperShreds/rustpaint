// Draws the model. `Camera`, `PI`, `environment_uv`, `tone_map` and
// `linear_to_srgb` come from common.wgsl.

@group(0) @binding(0) var<uniform> camera: Camera;

// Layer composites, premultiplied (see composite.wgsl).
@group(1) @binding(0) var base_color_texture: texture_2d<f32>;
@group(1) @binding(1) var roughness_texture: texture_2d<f32>;
@group(1) @binding(2) var metallic_texture: texture_2d<f32>;
@group(1) @binding(3) var height_texture: texture_2d<f32>;
@group(1) @binding(4) var material_sampler: sampler;

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

// The straight height of texel `at` in mip `level` (the texture repeats).
fn texel_height(at: vec2<i32>, size: vec2<i32>, level: i32) -> f32 {
    let t = textureLoad(height_texture, ((at % size) + size) % size, level);
    return select(t.r / t.a, 0.5, t.a <= 0.0);
}

// Cubic B-spline weights of the four texels around a position `f` of the
// way between the middle two.
fn cubic_weights(f: f32) -> vec4<f32> {
    let f2 = f * f;
    let f3 = f2 * f;
    return vec4<f32>(
        (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0,
        (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0,
        (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0,
        f3 / 6.0,
    );
}

// Weights of the three steps between those texels in the cubic's slope.
fn slope_weights(f: f32) -> vec3<f32> {
    return vec3<f32>((1.0 - f) * (1.0 - f), 1.0 + 2.0 * f - 2.0 * f * f, f * f) * 0.5;
}

// The slope (per UV unit, along u and v) of the heights in mip `level`
// smoothed by a cubic B-spline: it changes gently from texel to texel, so
// magnified bumps look round instead of faceted. Exact texels and steps
// between them rather than filtered taps, so level ground gives exactly
// zero instead of the filtering's rounding noise.
fn height_slope(uv: vec2<f32>, level: i32) -> vec2<f32> {
    let size_i = vec2<i32>(textureDimensions(height_texture, level));
    let size = vec2<f32>(size_i);
    let p = uv * size - 0.5;
    let first = vec2<i32>(floor(p)) - 1;
    let f = p - floor(p);
    var h: array<array<f32, 4>, 4>;
    for (var y = 0; y < 4; y++) {
        for (var x = 0; x < 4; x++) {
            h[y][x] = texel_height(first + vec2<i32>(x, y), size_i, level);
        }
    }
    let across_u = cubic_weights(f.y);
    let across_v = cubic_weights(f.x);
    let along_u = slope_weights(f.x);
    let along_v = slope_weights(f.y);
    var slope = vec2<f32>(0.0);
    for (var a = 0; a < 4; a++) {
        for (var b = 0; b < 3; b++) {
            slope.x += across_u[a] * along_u[b] * (h[a][b + 1] - h[a][b]);
            slope.y += across_v[a] * along_v[b] * (h[b + 1][a] - h[b][a]);
        }
    }
    return slope * size;
}

// The surface normal bent by the bumps of the height channel, using the
// surface gradient (Mikkelsen 2010): no tangents needed. Same slopes as
// `Bake::normal_map`, so the viewport matches the exported normal map.
fn bumped_normal(in: VertexOutput) -> vec3<f32> {
    let n = normalize(in.normal);
    let dp_x = dpdx(in.world_position);
    let dp_y = dpdy(in.world_position);
    let duv_x = dpdx(in.uv);
    let duv_y = dpdy(in.uv);
    let depth = camera.background.y;
    if depth <= 0.0 {
        return n;
    }
    // The mip whose texels are about a pixel, so distant bumps don't shimmer.
    let size = vec2<f32>(textureDimensions(height_texture));
    let footprint = max(length(duv_x * size), length(duv_y * size));
    let top = f32(textureNumLevels(height_texture) - 1u);
    let level = i32(clamp(log2(max(footprint, 1.0)), 0.0, top));
    let slope = height_slope(in.uv, level);
    let slope_u = slope.x;
    let slope_v = slope.y;
    // How u and v change across the surface, from their screen derivatives.
    let r1 = cross(dp_y, n);
    let r2 = cross(n, dp_x);
    let det = dot(dp_x, r1);
    if abs(det) < 1e-20 {
        return n;
    }
    let grad_u = (r1 * duv_x.x + r2 * duv_y.x) / det;
    let grad_v = (r1 * duv_x.y + r2 * duv_y.y) / det;
    let bumped = n - depth * (slope_u * grad_u + slope_v * grad_v);
    return select(normalize(bumped), n, dot(bumped, bumped) < 1e-12);
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
    if mode == 4u {
        return vec3<f32>(sample_channel(height_texture, in.uv).r);
    }
    let n = bumped_normal(in);
    if mode == 5u {
        // Shown as stored colour, so undo the display encoding to come.
        let color = n * 0.5 + 0.5;
        return color * color;
    }

    let roughness = clamp(roughness_value, 0.02, 1.0);
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
