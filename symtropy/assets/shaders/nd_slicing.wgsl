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

struct NdSlicingSettings {
    w_pos: f32,
    w_slice: f32,
    slice_thickness: f32,
    edge_fade: f32, // 0.0 = hard cut, 1.0 = smooth fade
    time: f32,
    phi_global: f32,
    surprise_global: f32,
    harmony_global: f32,
    energy_level: f32,
    _padding: f32,
    _padding2: f32,
    _padding3: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> settings: NdSlicingSettings;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // 1. Calculate distance from 4D slice plane
    // For now, use w_pos as the object's 4th coordinate
    let dist = abs(settings.w_pos - settings.w_slice);
    
    // 2. Discard if totally outside slice
    if (dist > settings.slice_thickness) {
        discard;
    }
    
    // 3. Holographic Edge Effect
    let edge_proximity = clamp(1.0 - (dist / settings.slice_thickness), 0.0, 1.0);
    
    // Generate standard PBR input
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // 4. Phi Heatmap Modulation
    // Low Phi (0.0) -> Cool/Cyan
    // High Phi (1.0) -> Hot/Orange-Red
    // Energy level adds brightness/bloom potential
    
    let low_phi_color = vec3<f32>(0.0, 0.8, 1.0); // Cyan
    let high_phi_color = vec3<f32>(1.0, 0.2, 0.0); // Orange-Red
    
    let phi_tint = mix(low_phi_color, high_phi_color, settings.phi_global);
    let energy_glow = settings.energy_level * 0.5;
    
    // Mix the PBR color with the Phi heatmap
    // We keep some of the original texture/base color but tint it
    pbr_input.material.base_color = vec4<f32>(
        mix(pbr_input.material.base_color.rgb, phi_tint, 0.6) + phi_tint * energy_glow,
        pbr_input.material.base_color.a * edge_proximity
    );

    // Add holographic "scanline" effect based on time
    let scanline = sin(in.world_position.y * 10.0 + settings.time * 5.0) * 0.1 + 0.9;
    pbr_input.material.base_color.rgb *= mix(1.0, scanline, dist / settings.slice_thickness);

    // Alpha discard (standard Bevy behavior)
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
