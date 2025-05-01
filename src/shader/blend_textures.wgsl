@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    // Create a full-screen triangle: https://wallisc.github.io/rendering/2021/04/18/Fullscreen-Pass.html
    let uv = vec2<f32>(
        f32((vertex_index << 1u) & 2u),
        f32(vertex_index & 2u)
    );
    return vec4<f32>(
        uv.x * 2.0 - 1.0,
        -(uv.y * 2.0 - 1.0),
        0.0,
        1.0
    );
}

@group(0) @binding(0) var tex_below: texture_2d<f32>;
@group(0) @binding(1) var tex_top: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    // Normalized texture coordinates
    let tex_width = f32(textureDimensions(tex_below).x);
    let tex_height = f32(textureDimensions(tex_below).y);
    let tex_coord = vec2<f32>(
        position.x / tex_width,
        position.y / tex_height
    );

    let color_below = textureSample(tex_below, tex_sampler, tex_coord);
    let color_top = textureSample(tex_top, tex_sampler, tex_coord);

    // Alpha blending
    return vec4<f32>(
        mix(color_below.rgb, color_top.rgb, color_top.a),
        1.0
    );
}
