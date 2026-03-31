#import bevy_pbr::{
    forward_io::VertexOutput,
    pbr_fragment::pbr_input_from_vertex_output,
    pbr_functions,
}

struct HolographicSettings {
    fresnel_color: vec4<f32>,
    fresnel_power: f32,
    scanline_speed: f32,
    scanline_density: f32,
    hologram_alpha: f32,
    time: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
};

@group(2) @binding(100) var<uniform> settings: HolographicSettings;

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Get view direction from camera
    let view_dir = normalize(in.world_position.xyz - view.world_position.xyz);
    let normal = normalize(in.world_normal);

    // === FRESNEL EFFECT ===
    // dot(normal, view) = 1 at center, 0 at edges
    let ndotv = abs(dot(normal, -view_dir));
    let fresnel = pow(1.0 - ndotv, settings.fresnel_power);

    // === SCANLINES ===
    // Horizontal bands sweeping downward
    let scan_pos = in.world_position.y * settings.scanline_density + settings.time * settings.scanline_speed;
    let scanline = smoothstep(0.4, 0.6, fract(scan_pos));

    // === NOISE PATTERN ===
    // Simple deterministic noise from position
    let noise_val = fract(sin(dot(in.world_position.xz, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let noise = mix(0.85, 1.0, noise_val); // subtle texture

    // === COMBINE ===
    // Base: very transparent at center, opaque at edges (Fresnel drives alpha)
    let base_alpha = settings.hologram_alpha * (0.3 + 0.7 * fresnel);

    // Fresnel glow color
    let glow = settings.fresnel_color.rgb * fresnel * 1.5;

    // Scanline modulation (dims slightly on scanline)
    let scan_mod = mix(1.0, 0.7, scanline * 0.3);

    // Final color: base PBR + Fresnel glow + scanline + noise
    var pbr_input = pbr_input_from_vertex_output(in, false, false);
    let base_color = pbr_input.material.base_color.rgb;

    let final_color = (base_color * scan_mod * noise + glow) * base_alpha;
    let final_alpha = base_alpha * scan_mod;

    return vec4<f32>(final_color, final_alpha);
}
