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
    let dist = abs(settings.w_pos - settings.w_slice);
    
    // 2. Discard or fade based on distance
    if (dist > settings.slice_thickness) {
        discard;
    }
    
    let alpha_mult = if (settings.edge_fade > 0.001) {
        clamp(1.0 - (dist / settings.slice_thickness), 0.0, 1.0)
    } else {
        1.0
    };

    // Generate standard PBR input
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Apply 4D alpha modulation
    pbr_input.material.base_color.a = pbr_input.material.base_color.a * alpha_mult;

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
