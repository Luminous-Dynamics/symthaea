#import bevy_ui::ui_vertex_output::UiVertexOutput

@group(1) @binding(0) var imagination_texture: texture_2d<f32>;
@group(1) @binding(1) var imagination_sampler: sampler;

fn inferno(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.0, 0.0, 0.0);
    let c1 = vec3<f32>(0.18, 0.05, 0.25);
    let c2 = vec3<f32>(0.47, 0.1, 0.37);
    let c3 = vec3<f32>(0.74, 0.2, 0.31);
    let c4 = vec3<f32>(0.95, 0.47, 0.15);
    let c5 = vec3<f32>(0.98, 0.85, 0.36);
    
    if (t < 0.2) {
        return mix(c0, c1, t / 0.2);
    } else if (t < 0.4) {
        return mix(c1, c2, (t - 0.2) / 0.2);
    } else if (t < 0.6) {
        return mix(c2, c3, (t - 0.4) / 0.2);
    } else if (t < 0.8) {
        return mix(c3, c4, (t - 0.6) / 0.2);
    } else {
        return mix(c4, c5, (t - 0.8) / 0.2);
    }
}

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    // 1. Sample the single R8Unorm channel (lives in .r)
    let intensity: f32 = textureSample(imagination_texture, imagination_sampler, in.uv).r;
    
    // 2. Map intensity to the Inferno color map
    let base_color = inferno(intensity);
    
    // 3. Procedural Upscaling Enhancer (Subtle CRT Scanline simulation to mask low-res)
    let scanline: f32 = sin(in.uv.y * 500.0) * 0.1 + 0.9;
    let mapped_color = base_color * scanline;
    
    // 4. Threshold-Free Faux Bloom Generation
    let bloom_threshold = 0.75;
    let bloom_intensity = max(0.0, intensity - bloom_threshold) / (1.0 - bloom_threshold);
    let bloom_glow = vec3<f32>(0.98, 0.6, 0.2) * pow(bloom_intensity, 3.0) * 2.5;
    
    // 5. Composite the final emissive fragments
    let final_rgb = mapped_color + bloom_glow;
    
    return vec4<f32>(final_rgb, intensity * 0.95 + 0.05); // Alpha tracks activation density
}
