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
    outline_color: vec4<f32>,
    outline_intensity: f32,   // 0.0 disables the coastline-outline effect
    outline_threshold: f32,   // luminance-gradient sensitivity
    _padding: f32,
    _padding2: f32,
    _padding3: f32,
};

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> holographic: HolographicSettings;
@group(#{MATERIAL_BIND_GROUP}) @binding(101)
var surface_texture: texture_2d<f32>;
@group(#{MATERIAL_BIND_GROUP}) @binding(102)
var surface_sampler: sampler;

/// Coastline-outline glow: samples the surface texture's four neighbors and
/// lights up wherever the color gradient crosses `outline_threshold` — i.e.
/// land/ocean boundaries — with a bright vector-art-style line. Reads
/// clearly regardless of how much the base color is tinted, unlike relying
/// on the tinted photo texture's own (often washed-out) color contrast.
///
/// Two things this needed to actually catch land like the Americas, not
/// just Antarctica's stark white-ice-vs-ocean edge:
/// 1. Full RGB color distance, not just luminance — green/tan land vs blue
///    ocean often differs far more in hue than in raw brightness, which a
///    luminance-only gradient mostly misses.
/// 2. A much wider sample offset (was 2 texels, now a fixed UV-space
///    epsilon) — at typical planet-orbit zoom the sphere covers a small
///    fraction of the screen, so the underlying 8k texture is heavily
///    minified/mipmapped before this even samples it. A 2-texel gap is
///    far smaller than the minification blur radius, so on all but the
///    most extreme boundaries (Antarctica) the four samples were already
///    blurred to nearly the same color before the gradient was computed.
fn coastline_outline(uv: vec2<f32>) -> f32 {
    let texel = vec2<f32>(0.0018, 0.0018);

    let c_up = textureSample(surface_texture, surface_sampler, uv + vec2<f32>(0.0, texel.y)).rgb;
    let c_down = textureSample(surface_texture, surface_sampler, uv - vec2<f32>(0.0, texel.y)).rgb;
    let c_left = textureSample(surface_texture, surface_sampler, uv - vec2<f32>(texel.x, 0.0)).rgb;
    let c_right = textureSample(surface_texture, surface_sampler, uv + vec2<f32>(texel.x, 0.0)).rgb;

    let dv = c_up - c_down;
    let dh = c_left - c_right;
    let gradient = length(dv) + length(dh);
    return smoothstep(holographic.outline_threshold, holographic.outline_threshold * 2.0, gradient);
}

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

        // Modulate base color with holographic effects. Alpha used to be
        // gated by the Fresnel term (0.15 + 0.85*fresnel^1.5), which is
        // near-zero face-on (fresnel -> 0 where the viewer looks straight
        // at the globe) and only opens up at the grazing silhouette edge —
        // that crushed the visible landmass to near-total transparency
        // exactly where users look. Fresnel now only drives the edge-glow
        // emissive below, not whether the surface data is visible at all.
        let scan_mod = mix(1.0, 0.85, scanline * 0.2);
        let noise_mod = mix(0.85, 1.0, noise);
        pbr_input.material.base_color = vec4<f32>(
            pbr_input.material.base_color.rgb * scan_mod * noise_mod,
            pbr_input.material.base_color.a * holographic.hologram_alpha
        );

        // Add Fresnel glow to emissive
        pbr_input.material.emissive = pbr_input.material.emissive +
            vec4<f32>(holographic.fresnel_color.rgb * fresnel * 1.5, 0.0);

        // === COASTLINE OUTLINE ===
        // A bright glowing line along land/ocean boundaries, independent
        // of the (tinted, sometimes washed-out) base texture color — this
        // is what actually makes landmass legible in a heavily-stylized
        // hologram tint, rather than relying on tint-and-hope.
        if (holographic.outline_intensity > 0.001) {
            let edge = coastline_outline(in.uv);
            pbr_input.material.emissive = pbr_input.material.emissive +
                vec4<f32>(holographic.outline_color.rgb * edge * holographic.outline_intensity, 0.0);
        }
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
