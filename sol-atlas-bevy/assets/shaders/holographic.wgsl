#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
    mesh_view_bindings::view,
}
#endif

struct HolographicSettings {
    fresnel_color: vec4<f32>,
    fresnel_power: f32,
    scanline_speed: f32,
    scanline_density: f32,
    hologram_alpha: f32,
    time: f32,
    enable_holographic: f32,  // 1.0 = full effects, 0.0 = PBR only
    _padding: f32,
    _padding2: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> holographic: HolographicSettings;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // Generate standard PBR input
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // View direction
    let world_pos = pbr_input.world_position.xyz;
    let view_pos = view.world_position.xyz;
    let view_dir = normalize(world_pos - view_pos);
    let normal = pbr_input.world_normal;

    // === FRESNEL EFFECT ===
    let ndotv = abs(dot(normalize(normal), -view_dir));
    let fresnel = pow(1.0 - ndotv, holographic.fresnel_power);

    // Conditionally apply holographic effects (Satellite mode = PBR only)
    if (holographic.enable_holographic > 0.5) {
        // === SCANLINES ===
        let scan_pos = world_pos.y * holographic.scanline_density + holographic.time * holographic.scanline_speed;
        let scanline = smoothstep(0.4, 0.6, fract(scan_pos));

        // === NOISE ===
        let noise = fract(sin(dot(world_pos.xz, vec2<f32>(12.9898, 78.233))) * 43758.5453);

        // Modulate base color with holographic effects
        let scan_mod = mix(1.0, 0.85, scanline * 0.2);
        let noise_mod = mix(0.85, 1.0, noise);
        pbr_input.material.base_color = vec4<f32>(
            pbr_input.material.base_color.rgb * scan_mod * noise_mod,
            pbr_input.material.base_color.a * holographic.hologram_alpha * (0.15 + 0.85 * pow(fresnel, 1.5))
        );

        // Add Fresnel glow to emissive
        pbr_input.material.emissive = pbr_input.material.emissive +
            vec4<f32>(holographic.fresnel_color.rgb * fresnel * 1.5, 0.0);
    } else {
        // PBR mode — subtle atmospheric rim light only
        let rim = pow(1.0 - ndotv, 2.0) * 0.3;
        pbr_input.material.emissive = pbr_input.material.emissive +
            vec4<f32>(0.3 * rim, 0.5 * rim, 0.8 * rim, 0.0);
    }

    // Alpha discard
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;
    out.color = apply_pbr_lighting(pbr_input);
    out.color = main_pass_post_lighting_processing(pbr_input, out.color);
#endif

    return out;
}
