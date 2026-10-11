// Blends layer textures into a slot's composite texture, and builds its mips.
//
// Layers are combined one at a time into a 16-bit float scratch texture
// holding premultiplied, linear values: each layer pass reads the result so
// far from one scratch texture and writes the next into the other. A resolve
// pass then copies the result into the 8-bit composite, and mip passes shrink it.

// Covers the whole target with one triangle; `uv` is (0,0) top-left.
struct FullScreen {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_full_screen(@builtin(vertex_index) index: u32) -> FullScreen {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: FullScreen;
    out.position = vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return out;
}

// --- Layer pass

@group(0) @binding(0) var layer_texture: texture_2d<f32>;
// x = opacity, y = blend mode (`BlendMode::index`).
@group(1) @binding(0) var<uniform> layer_params: vec4<f32>;
// Everything below this layer, premultiplied.
@group(2) @binding(0) var below_texture: texture_2d<f32>;

// Must match `BlendMode::apply` in document.rs.
fn blend_channel(mode: u32, below: f32, top: f32) -> f32 {
    var value = top;
    switch mode {
        case 1u: { value = below * top; }
        case 2u: { value = below + top - below * top; }
        case 3u: {
            value = select(1.0 - 2.0 * (1.0 - below) * (1.0 - top), 2.0 * below * top, below <= 0.5);
        }
        case 4u: { value = min(below, top); }
        case 5u: { value = max(below, top); }
        case 6u: { value = below + top; }
        default: {}
    }
    return clamp(value, 0.0, 1.0);
}

@fragment
fn fs_layer(in: FullScreen) -> @location(0) vec4<f32> {
    // All textures are the target's size: read matching texels exactly.
    let texel = vec2<i32>(in.position.xy);
    let top = textureLoad(layer_texture, texel, 0);
    let below = textureLoad(below_texture, texel, 0);
    let alpha = top.a * layer_params.x;
    let mode = u32(layer_params.y + 0.5);

    let below_straight = select(vec3<f32>(0.0), below.rgb / below.a, below.a > 0.0);
    let mixed = vec3<f32>(
        blend_channel(mode, below_straight.r, top.r),
        blend_channel(mode, below_straight.g, top.g),
        blend_channel(mode, below_straight.b, top.b),
    );
    // W3C compositing: the blend mode applies only where both are present.
    let color = top.rgb * alpha * (1.0 - below.a) + below.rgb * (1.0 - alpha) + alpha * below.a * mixed;
    return vec4<f32>(color, alpha + below.a * (1.0 - alpha));
}

// --- Resolve and mip passes: sample the source at this pixel's position.
// Same size as the target gives an exact copy; double size gives a 2x2 average.

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

@fragment
fn fs_copy(in: FullScreen) -> @location(0) vec4<f32> {
    return textureSample(source_texture, source_sampler, in.uv);
}
