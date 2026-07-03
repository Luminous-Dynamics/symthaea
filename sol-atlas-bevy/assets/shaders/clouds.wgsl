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

struct CloudSettings {
    time: f32,
    scroll_speed_1: f32,
    scroll_speed_2: f32,
    contrast: f32,
    coverage: f32,
    silver_lining: f32,
    _padding: f32,
    _padding2: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> clouds: CloudSettings;
@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var cloud_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var cloud_sampler: sampler;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);

    // Two independently-drifting samples of the same cloud texture,
    // combined — a cheap stand-in for volumetric depth. Real raymarched
    // 3D noise would look better still but is a much larger shader
    // investment; this reads convincingly at planet-orbit distance.
    //
    // Deliberately offset-only, no UV scaling: scaling V (latitude) pushes
    // it out of [0,1] for a wide southern band, and if the sampler clamps
    // rather than repeats there, that whole band all reads the same
    // clamped edge texel — which happened to be a bright strip in this
    // texture, rendering as a large fake "cloud mass" stuck near one pole.
    let uv1 = in.uv + vec2<f32>(clouds.time * clouds.scroll_speed_1, 0.0);
    let uv2 = in.uv + vec2<f32>(
        clouds.time * clouds.scroll_speed_2,
        clouds.time * clouds.scroll_speed_2 * 0.3,
    );

    let sample1 = textureSample(cloud_texture, cloud_sampler, uv1).r;
    let sample2 = textureSample(cloud_texture, cloud_sampler, uv2).r;
    let combined_luma = max(sample1, sample2 * 0.8);

    // Contrast-enhance so formations read as distinct puffy clouds rather
    // than a flat translucent haze.
    let shaped = pow(clamp(combined_luma, 0.0, 1.0), clouds.contrast);

    let world_pos = pbr_input.world_position.xyz;
    let view_pos = view.world_position.xyz;
    let view_dir = normalize(world_pos - view_pos);
    let normal = normalize(pbr_input.world_normal);
    let ndotv = abs(dot(normal, -view_dir));
    let rim = pow(1.0 - ndotv, 2.5);

    pbr_input.material.base_color = vec4<f32>(vec3<f32>(1.0, 1.0, 1.0), shaped * clouds.coverage);

    // Silver-lining: bright rim glow at the cloud silhouette edge — the
    // signature look of real satellite-photographed cloud cover.
    let lining = rim * clouds.silver_lining * shaped;
    pbr_input.material.emissive = pbr_input.material.emissive
        + vec4<f32>(lining, lining, lining * 0.95, 0.0);

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
