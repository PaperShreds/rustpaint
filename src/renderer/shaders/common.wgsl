// Shared by mesh.wgsl and background.wgsl (prepended to each).

struct Camera {
    view_proj: mat4x4<f32>,
    // Screen position back to world, for the background.
    inverse_view_proj: mat4x4<f32>,
    // xyz = camera position.
    eye: vec4<f32>,
    // x = view mode (0 lit, 1 base colour, 2 roughness, 3 metallic),
    // y = environment rotation (radians), z = intensity, w = environment mip count.
    settings: vec4<f32>,
    // x = background blur (0 sharp .. 1 very blurry).
    background: vec4<f32>,
};

const PI: f32 = 3.14159265;

// Where a world direction lands on the equirectangular environment map.
fn environment_uv(direction: vec3<f32>, rotation: f32) -> vec2<f32> {
    let d = normalize(direction);
    let angle = atan2(d.z, d.x) + rotation;
    let u = fract(angle / (2.0 * PI) + 0.5);
    let v = acos(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2<f32>(u, v);
}

// One mip level, filtered with a cubic B-spline instead of bilinear, so a
// small (blurry) level magnified across the screen looks smooth rather than
// blocky. Four bilinear taps (GPU Gems 2, chapter 20).
fn sample_level_smooth(t: texture_2d<f32>, s: sampler, uv: vec2<f32>, level: f32) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(t, i32(level)));
    let p = uv * size - 0.5;
    let i = floor(p);
    let f = p - i;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    // Texel centres to sample, between the pairs of texels each tap covers.
    let a = (i - 0.5 + w1 / g0) / size;
    let b = (i + 1.5 + w3 / g1) / size;
    let top = textureSampleLevel(t, s, vec2<f32>(a.x, a.y), level).rgb * g0.x
        + textureSampleLevel(t, s, vec2<f32>(b.x, a.y), level).rgb * g1.x;
    let bottom = textureSampleLevel(t, s, vec2<f32>(a.x, b.y), level).rgb * g0.x
        + textureSampleLevel(t, s, vec2<f32>(b.x, b.y), level).rgb * g1.x;
    return top * g0.y + bottom * g1.y;
}

// The environment at a fractional mip level (higher is blurrier).
fn sample_environment(t: texture_2d<f32>, s: sampler, uv: vec2<f32>, lod: f32) -> vec3<f32> {
    let last = f32(textureNumLevels(t) - 1u);
    let level = clamp(lod, 0.0, last);
    let low = floor(level);
    let high = min(low + 1.0, last);
    return mix(
        sample_level_smooth(t, s, uv, low),
        sample_level_smooth(t, s, uv, high),
        level - low,
    );
}

// Compresses HDR values into the displayable range (ACES fit by Narkowicz).
fn tone_map(color: vec3<f32>) -> vec3<f32> {
    let x = max(color, vec3<f32>(0.0));
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    let low = c * 12.92;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, c <= vec3<f32>(0.0031308));
}
