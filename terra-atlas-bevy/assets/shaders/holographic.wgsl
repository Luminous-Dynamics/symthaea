#import bevy_pbr::{
    forward_io::VertexOutput,
    mesh_view_bindings::view,
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
    // View direction from camera to fragment
    let view_dir = normalize(in.world_position.xyz - view.world_position.xyz);
    let normal = normalize(in.world_normal);

    // === FRESNEL EFFECT ===
    let ndotv = abs(dot(normal, -view_dir));
    let fresnel = pow(1.0 - ndotv, settings.fresnel_power);

    // === SCANLINES ===
    let scan_pos = in.world_position.y * settings.scanline_density + settings.time * settings.scanline_speed;
    let scanline = smoothstep(0.4, 0.6, fract(scan_pos));

    // === NOISE ===
    let noise_val = fract(sin(dot(in.world_position.xz, vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let noise = mix(0.85, 1.0, noise_val);

    // === COMBINE ===
    let base_alpha = settings.hologram_alpha * (0.3 + 0.7 * fresnel);
    let glow = settings.fresnel_color.rgb * fresnel * 1.5;
    let scan_mod = mix(1.0, 0.7, scanline * 0.3);

    // Sample the base color from the texture (via PBR input)
    let base_color = vec3<f32>(0.12, 0.18, 0.22) * noise * scan_mod;

    let final_color = base_color + glow;
    let final_alpha = base_alpha * scan_mod;

    return vec4<f32>(final_color * final_alpha, final_alpha);
}
