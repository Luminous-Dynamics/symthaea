#import bevy_ui::ui_vertex_output::UiVertexOutput

@group(1) @binding(0) var imagination_texture: texture_2d<f32>;
@group(1) @binding(1) var imagination_sampler: sampler;

@fragment
fn fragment(in: UiVertexOutput) -> @location(0) vec4<f32> {
    // 1. Sample the single R8Unorm channel (lives in .r)
    let intensity: f32 = textureSample(imagination_texture, imagination_sampler, in.uv).r;
    
    // 2. Define a base synaptic color palette (e.g., Vivid Cyan/Teal)
    let base_color = vec3<f32>(0.0, 0.95, 0.85);
    
    // 3. Procedural Upscaling Enhancer (Subtle CRT Scanline simulation to mask low-res)
    let scanline: f32 = sin(in.uv.y * 500.0) * 0.1 + 0.9;
    let mapped_color = base_color * intensity * scanline;
    
    // 4. Threshold-Free Faux Bloom Generation
    // We pass the intensity through an exponential curve to isolate and amplify high-activation areas
    let bloom_threshold = 0.75;
    let bloom_intensity = max(0.0, intensity - bloom_threshold) / (1.0 - bloom_threshold);
    let bloom_glow = vec3<f32>(0.3, 1.0, 0.9) * pow(bloom_intensity, 3.0) * 2.5;
    
    // 5. Composite the final emissive fragments
    let final_rgb = mapped_color + bloom_glow;
    
    return vec4<f32>(final_rgb, intensity * 0.95); // Alpha tracks activation density
}
