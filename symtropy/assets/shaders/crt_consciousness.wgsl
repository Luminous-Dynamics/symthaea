// CRT + Consciousness Post-Processing Shader
// Driven by allostatic_load, arousal, and Leviathan danger.
//
// Effects:
// - CRT scanlines (intensity modulated by consciousness)
// - Phosphor glow (warm when calm, cold when stressed)
// - Chromatic aberration (driven by allostatic_load)
// - Vignette (darkens edges, deepens with stress)
// - Noise/grain (increases with arousal)
// - Color grading (shifts hue from gold→teal→red with danger)

#import bevy_core_pipeline::fullscreen_vertex_shader::FullscreenVertexOutput

@group(0) @binding(0) var screen_texture: texture_2d<f32>;
@group(0) @binding(1) var texture_sampler: sampler;

struct ConsciousnessUniforms {
    // Allostatic load [0, 1] — drives aberration + vignette
    allostatic_load: f32,
    // Arousal [0, 1] — drives noise + scanline intensity
    arousal: f32,
    // Danger level [0, 1] — drives color shift + distortion
    danger: f32,
    // Time (seconds) — for animation
    time: f32,
    // Screen resolution
    resolution: vec2<f32>,
    // Padding for alignment
    _pad: vec2<f32>,
}

@group(0) @binding(2) var<uniform> consciousness: ConsciousnessUniforms;

// Pseudo-random hash
fn hash(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453);
}

@fragment
fn fragment(in: FullscreenVertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let load = consciousness.allostatic_load;
    let arousal = consciousness.arousal;
    let danger = consciousness.danger;
    let t = consciousness.time;

    // ── Chromatic Aberration ──
    // Offset increases with allostatic load
    let aberration = load * 0.008 + danger * 0.004;
    let r = textureSample(screen_texture, texture_sampler, uv + vec2<f32>(aberration, 0.0)).r;
    let g = textureSample(screen_texture, texture_sampler, uv).g;
    let b = textureSample(screen_texture, texture_sampler, uv - vec2<f32>(aberration, 0.0)).b;
    var color = vec3<f32>(r, g, b);

    // ── CRT Scanlines ──
    let scanline_freq = consciousness.resolution.y * 0.5;
    let scanline = sin(uv.y * scanline_freq * 3.14159) * 0.5 + 0.5;
    let scanline_intensity = 0.03 + arousal * 0.08; // subtle when calm, visible when stressed
    color = color * (1.0 - scanline_intensity * (1.0 - scanline));

    // ── Phosphor Glow ──
    // Warm gold when calm, cold teal when stressed
    let calm_tint = vec3<f32>(1.02, 1.01, 0.98); // warm
    let stress_tint = vec3<f32>(0.95, 1.0, 1.05);  // cold
    let tint = mix(calm_tint, stress_tint, load);
    color = color * tint;

    // ── Vignette ──
    let center = uv - vec2<f32>(0.5, 0.5);
    let dist = length(center);
    let vignette_strength = 0.3 + load * 0.4 + danger * 0.3; // deeper with stress
    let vignette = 1.0 - smoothstep(0.3, 0.9, dist) * vignette_strength;
    color = color * vignette;

    // ── Film Grain ──
    let grain = hash(uv * consciousness.resolution + vec2<f32>(t * 100.0, t * 73.0));
    let grain_intensity = 0.02 + arousal * 0.06;
    color = color + (grain - 0.5) * grain_intensity;

    // ── Danger Color Shift ──
    // Shift toward red when danger is high
    if danger > 0.1 {
        let red_shift = danger * 0.15;
        color.r = color.r + red_shift;
        color.g = color.g * (1.0 - danger * 0.2);
        color.b = color.b * (1.0 - danger * 0.3);
    }

    // ── Consciousness Glow ──
    // When allostatic load is very low (deep calm), add subtle golden bloom
    if load < 0.1 {
        let calm_glow = (1.0 - load * 10.0) * 0.05;
        color = color + vec3<f32>(calm_glow * 0.8, calm_glow * 0.6, calm_glow * 0.1);
    }

    return vec4<f32>(clamp(color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
