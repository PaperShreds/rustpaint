struct Camera {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;

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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let light_dir = normalize(vec3<f32>(0.5, 0.8, 0.6));
    let diffuse = max(dot(n, light_dir), 0.0);
    let light = 0.25 + 0.75 * diffuse;

    // Checkerboard from UVs: 8x8 cells per face.
    let cell = floor(in.uv * 8.0);
    let checker = (cell.x + cell.y) % 2.0;
    let base = mix(vec3<f32>(0.85), vec3<f32>(0.45), checker);

    return vec4<f32>(base * light, 1.0);
}