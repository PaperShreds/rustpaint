// Blends layer textures into a slot's composite texture, and builds its mips.
//
// The composite is sRGB, so the GPU blends in linear light. It holds
// premultiplied colour, which keeps mip averaging correct where paint meets
// transparency.

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

// --- Layer pass: drawn once per visible layer, blended "over" what's below.

@group(0) @binding(0) var layer_texture: texture_2d<f32>;
// x = layer opacity.
@group(1) @binding(0) var<uniform> layer_opacity: vec4<f32>;

@fragment
fn fs_layer(in: FullScreen) -> @location(0) vec4<f32> {
    // Same size as the target: read the matching texel exactly.
    let texel = textureLoad(layer_texture, vec2<i32>(in.position.xy), 0);
    let alpha = texel.a * layer_opacity.x;
    return vec4<f32>(texel.rgb * alpha, alpha);
}

// --- Mip pass: each level is the bilinear average of the level above.

@group(0) @binding(0) var source_texture: texture_2d<f32>;
@group(0) @binding(1) var source_sampler: sampler;

@fragment
fn fs_mip(in: FullScreen) -> @location(0) vec4<f32> {
    return textureSample(source_texture, source_sampler, in.uv);
}
