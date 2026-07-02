// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! WebGL2 consciousness visualization engine.
//!
//! Renders the Eight Harmonies as luminous orbs in binaural space,
//! with Phi driving visual coherence and spectral features driving particles.
//! Uses raw WebGL2 via web-sys for clean Leptos integration.

use wasm_bindgen::JsCast;
use web_sys::{
    HtmlCanvasElement, WebGl2RenderingContext as GL, WebGlProgram, WebGlShader,
    WebGlUniformLocation,
};

use crate::audio_bridge::AudioAnalysis;

// --- Harmony binaural positions (azimuth degrees, elevation degrees, distance) ---
// From symthaea-muse binaural.rs spatial layout

const HARMONY_POSITIONS: [(f32, f32, f32); 8] = [
    (0.0, 0.0, 1.0),      // ResonantCoherence — center, close
    (28.0, 6.0, 1.5),     // PanSentientFlourishing — right-front
    (-23.0, -6.0, 1.5),   // IntegralWisdom — left-front
    (-132.0, -9.0, 2.5),  // InfinitePlay — left-rear
    (69.0, 0.0, 2.0),     // UniversalInterconnectedness — wide right
    (171.0, 0.0, 2.0),    // SacredReciprocity — behind
    (17.0, 23.0, 1.8),    // EvolutionaryProgression — ascending
    (0.0, 68.0, 1.5),     // SacredStillness — above
];

const HARMONY_NAMES: [&str; 8] = [
    "Resonant Coherence",
    "Pan-Sentient Flourishing",
    "Integral Wisdom",
    "Infinite Play",
    "Universal Interconnectedness",
    "Sacred Reciprocity",
    "Evolutionary Progression",
    "Sacred Stillness",
];

/// Base colors for each harmony (R, G, B) — warm to cool spectrum.
const HARMONY_COLORS: [(f32, f32, f32); 8] = [
    (1.0, 0.95, 0.8),   // ResonantCoherence — warm white
    (1.0, 0.7, 0.3),    // PanSentient — warm gold
    (0.3, 0.8, 0.5),    // IntegralWisdom — green
    (0.9, 0.4, 0.8),    // InfinitePlay — magenta
    (0.3, 0.6, 1.0),    // UniversalInterconnectedness — blue
    (0.7, 0.5, 1.0),    // SacredReciprocity — purple
    (1.0, 0.5, 0.3),    // EvolutionaryProgression — orange
    (0.6, 0.9, 1.0),    // SacredStillness — ice blue
];

// --- Vertex shader: fullscreen quad ---

const VERTEX_SHADER_SRC: &str = r#"#version 300 es
precision highp float;
in vec2 a_position;
out vec2 v_uv;
void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
"#;

// --- Fragment shader: consciousness field ---

const FRAGMENT_SHADER_SRC: &str = r#"#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform float u_phi;
uniform float u_arousal;
uniform float u_valence;
uniform float u_rms;
uniform float u_centroid;
uniform float u_flux;

uniform vec2 u_orbs[8];
uniform vec3 u_orb_colors[8];
uniform float u_orb_energy[8];

// --- Noise functions ---
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    mat2 rot = mat2(0.8, 0.6, -0.6, 0.8);
    for (int i = 0; i < 6; i++) {
        v += a * noise(p);
        p = rot * p * 2.0;
        a *= 0.5;
    }
    return v;
}

// --- Star field ---
float stars(vec2 uv, float density, float brightness) {
    vec2 cell = floor(uv * density);
    vec2 local = fract(uv * density);
    float r = hash21(cell);
    vec2 center = vec2(r, hash21(cell + 100.0));
    float d = length(local - center);
    float size = 0.01 + r * 0.02;
    float twinkle = sin(u_time * (2.0 + r * 4.0) + r * 100.0) * 0.3 + 0.7;
    return smoothstep(size, 0.0, d) * brightness * twinkle * step(0.85, r);
}

void main() {
    vec2 uv = v_uv;
    vec2 centered = uv - 0.5;
    float radial = length(centered);
    float angle = atan(centered.y, centered.x);

    // --- Star field background (drifts slowly) ---
    vec2 star_uv = uv + vec2(u_time * 0.003, u_time * 0.001);
    float star_layer1 = stars(star_uv, 30.0, 0.4);
    float star_layer2 = stars(star_uv * 1.5 + 0.5, 50.0, 0.2);
    vec3 star_color = vec3(0.7, 0.8, 1.0) * (star_layer1 + star_layer2) * (1.0 - u_arousal * 0.5);

    // --- Nebula background ---
    float time_slow = u_time * 0.04 * (0.5 + u_arousal * 0.5);
    vec2 nebula_uv = centered * 3.0 + vec2(time_slow, time_slow * 0.7);
    float nebula = fbm(nebula_uv);
    float nebula2 = fbm(nebula_uv + vec2(5.2, 1.3) + time_slow * 0.3);
    float nebula3 = fbm(nebula_uv * 0.5 + vec2(u_time * 0.01, 0.0));

    // Valence color: negative=deep blue, neutral=purple, positive=warm amber
    vec3 nebula_neg = vec3(0.03, 0.05, 0.15);
    vec3 nebula_neu = vec3(0.06, 0.03, 0.10);
    vec3 nebula_pos = vec3(0.12, 0.06, 0.02);
    float val_mix = u_valence * 0.5 + 0.5;
    vec3 bg_color = mix(nebula_neg, nebula_neu, smoothstep(0.0, 0.5, val_mix));
    bg_color = mix(bg_color, nebula_pos, smoothstep(0.5, 1.0, val_mix));

    float nebula_brightness = 0.02 + u_arousal * 0.08;
    bg_color += nebula_brightness * vec3(nebula * 0.7, nebula2 * 0.5, nebula3 * 0.6);
    bg_color += star_color;

    // --- Phi coherence structures ---
    float phi_s = smoothstep(0.3, 0.9, u_phi);

    // Coherence ring (expands with phi)
    float ring_r = 0.22 + phi_s * 0.08;
    float ring_w = 0.008 + u_rms * 0.004;
    float coherence_ring = exp(-pow(radial - ring_r, 2.0) / (ring_w * ring_w));
    bg_color += coherence_ring * vec3(0.4, 0.6, 1.0) * phi_s * 0.12;

    // Subtle radial grid at low phi (fragmented consciousness)
    float grid_fade = 1.0 - smoothstep(0.0, 0.4, u_phi);
    if (grid_fade > 0.01) {
        float grid = smoothstep(0.98, 1.0, abs(sin(radial * 40.0))) * 0.03;
        grid += smoothstep(0.98, 1.0, abs(sin(angle * 8.0))) * 0.02 * smoothstep(0.1, 0.3, radial);
        bg_color += vec3(0.2, 0.3, 0.4) * grid * grid_fade;
    }

    vec3 color = bg_color;

    // --- Harmony orbs ---
    for (int i = 0; i < 8; i++) {
        vec2 orb_pos = u_orbs[i];
        vec3 orb_color = u_orb_colors[i];
        float energy = u_orb_energy[i];

        vec2 delta = uv - orb_pos;
        float dist = length(delta);

        // Breathing core (phi-synced)
        float breath = 1.0 + sin(u_time * 1.5 + float(i) * 0.9) * 0.15 * u_phi;
        float core_size = (0.012 + energy * 0.018 + u_rms * 0.008) * breath;
        float core = exp(-dist * dist / (core_size * core_size));

        // Inner halo
        float halo_size = core_size * 3.0;
        float halo = exp(-dist * dist / (halo_size * halo_size)) * 0.35;

        // Outer aura (very soft, wide)
        float aura_size = 0.07 + energy * 0.05 + u_phi * 0.04;
        float aura = exp(-dist * dist / (aura_size * aura_size)) * 0.15;

        // Audio-reactive pulse
        float pulse = 1.0 + sin(u_time * 2.5 + float(i) * 0.8) * u_rms * 0.25;

        // Spectral shimmer rings
        float shimmer = 0.0;
        if (energy > 0.25) {
            shimmer = sin(dist * 60.0 - u_time * 4.0 + float(i) * 1.3) * 0.1 * u_flux;
            shimmer *= exp(-dist * dist / (aura_size * aura_size * 0.5));
        }

        // Chromatic aberration at high energy (slight color shift per channel)
        vec3 orb_contrib = orb_color * (core * 2.0 + halo + aura + shimmer) * pulse * (0.4 + energy * 0.8);
        if (energy > 0.5) {
            float ca = 0.003 * energy;
            float dist_r = length(delta + vec2(ca, 0.0));
            float dist_b = length(delta - vec2(ca, 0.0));
            orb_contrib.r *= 1.0 + exp(-dist_r * dist_r / (core_size * core_size)) * 0.2;
            orb_contrib.b *= 1.0 + exp(-dist_b * dist_b / (core_size * core_size)) * 0.2;
        }

        color += orb_contrib;
    }

    // --- Tendrils (Phi-gated, richer) ---
    if (u_phi > 0.25) {
        float tendril_strength = smoothstep(0.25, 0.85, u_phi) * 0.18;
        for (int i = 0; i < 8; i++) {
            for (int j = i + 1; j < 8; j++) {
                vec2 a = u_orbs[i];
                vec2 b = u_orbs[j];
                float orb_dist = length(b - a);
                if (orb_dist < 0.55) {
                    vec2 ab = b - a;
                    float t = clamp(dot(uv - a, ab) / dot(ab, ab), 0.0, 1.0);
                    vec2 closest = a + t * ab;

                    // Wavy tendril (sine displacement perpendicular to line)
                    vec2 perp = normalize(vec2(-ab.y, ab.x));
                    float wave = sin(t * 15.0 + u_time * 2.0 + float(i + j)) * 0.008 * u_phi;
                    closest += perp * wave;

                    float line_dist = length(uv - closest);
                    float width = 0.002 + u_rms * 0.002;
                    float tendril = exp(-line_dist * line_dist / (width * width));

                    vec3 tendril_color = mix(u_orb_colors[i], u_orb_colors[j], t);
                    float energy_avg = (u_orb_energy[i] + u_orb_energy[j]) * 0.5;

                    // Flowing particles along tendril
                    float flow = sin(t * 25.0 - u_time * 4.0) * 0.5 + 0.5;
                    float particles = pow(flow, 3.0); // sharper peaks = particle-like

                    color += tendril_color * tendril * tendril_strength * energy_avg * (0.3 + particles * 0.7);
                }
            }
        }
    }

    // --- Central phi orb (breathing, multi-layered) ---
    float phi_breath = 1.0 + sin(u_time * 0.8) * 0.1 * u_phi;
    float phi_core_r = (0.008 + u_phi * 0.025) * phi_breath;
    float phi_core = exp(-radial * radial / (phi_core_r * phi_core_r));

    float phi_halo_r = phi_core_r * 4.0;
    float phi_halo = exp(-radial * radial / (phi_halo_r * phi_halo_r)) * 0.4;

    float phi_outer_r = 0.08 + u_phi * 0.06;
    float phi_outer = exp(-radial * radial / (phi_outer_r * phi_outer_r)) * 0.1;

    vec3 phi_color = mix(vec3(0.25, 0.25, 0.5), vec3(1.0, 0.92, 0.5), u_phi);
    vec3 phi_edge_color = mix(vec3(0.4, 0.4, 0.8), vec3(1.0, 0.8, 0.3), u_phi);
    color += phi_color * (phi_core * 2.5 + phi_halo) * (0.3 + u_phi * 0.7);
    color += phi_edge_color * phi_outer * u_phi;

    // --- Spectral energy ring ---
    float spec_ring_r = 0.32 + u_centroid * 0.12;
    float spec_ring_w = 0.004 + u_rms * 0.006;
    float spec_ring = exp(-pow(radial - spec_ring_r, 2.0) / (spec_ring_w * spec_ring_w));
    // Ring rotates with audio flux
    float ring_angle_mod = sin(angle * 3.0 + u_time * u_flux * 2.0) * 0.5 + 0.5;
    vec3 ring_color = mix(vec3(0.15, 0.4, 0.8), vec3(0.8, 0.25, 0.6), u_centroid);
    color += ring_color * spec_ring * u_rms * 0.6 * (0.5 + ring_angle_mod * 0.5);

    // --- Drifting motes (consciousness particles) ---
    float mote_sum = 0.0;
    for (int i = 0; i < 12; i++) {
        float fi = float(i);
        float speed = 0.02 + fi * 0.005;
        vec2 mote_pos = vec2(
            0.5 + sin(u_time * speed + fi * 2.1) * (0.2 + fi * 0.02),
            0.5 + cos(u_time * speed * 0.7 + fi * 1.7) * (0.15 + fi * 0.015)
        );
        float d = length(uv - mote_pos);
        float mote_r = 0.003 + u_phi * 0.002;
        mote_sum += exp(-d * d / (mote_r * mote_r)) * (0.3 + u_phi * 0.5);
    }
    color += vec3(0.6, 0.7, 0.9) * mote_sum * 0.15;

    // --- Vignette (softer, consciousness-aware) ---
    float vignette = 1.0 - smoothstep(0.25, 0.72, radial);
    float phi_brighten = u_phi * 0.15; // high phi lifts the vignette
    color *= (0.25 + phi_brighten) + vignette * (0.75 - phi_brighten);

    // --- Tone mapping (ACES-like filmic) ---
    color = color * (2.51 * color + 0.03) / (color * (2.43 * color + 0.59) + 0.14);

    fragColor = vec4(color, 1.0);
}
"#;

/// Consciousness visualization state.
pub struct ConsciousnessViz {
    gl: GL,
    program: WebGlProgram,
    // Uniforms
    u_time: WebGlUniformLocation,
    u_phi: WebGlUniformLocation,
    u_arousal: WebGlUniformLocation,
    u_valence: WebGlUniformLocation,
    u_rms: WebGlUniformLocation,
    u_centroid: WebGlUniformLocation,
    u_flux: WebGlUniformLocation,
    u_orbs: WebGlUniformLocation,
    u_orb_colors: WebGlUniformLocation,
    u_orb_energy: WebGlUniformLocation,
    // State
    width: f32,
    height: f32,
}

impl ConsciousnessViz {
    /// Initialize WebGL2 on the given canvas element.
    pub fn new(canvas: &HtmlCanvasElement) -> Result<Self, String> {
        let gl = canvas
            .get_context("webgl2")
            .map_err(|e| format!("getContext failed: {:?}", e))?
            .ok_or("WebGL2 not supported")?
            .dyn_into::<GL>()
            .map_err(|_| "Failed to cast to WebGL2")?;

        let w = canvas.width() as f32;
        let h = canvas.height() as f32;
        gl.viewport(0, 0, w as i32, h as i32);

        // Compile shaders
        let vert = compile_shader(&gl, GL::VERTEX_SHADER, VERTEX_SHADER_SRC)?;
        let frag = compile_shader(&gl, GL::FRAGMENT_SHADER, FRAGMENT_SHADER_SRC)?;
        let program = link_program(&gl, &vert, &frag)?;
        gl.use_program(Some(&program));

        // Create fullscreen quad VAO
        let vao = gl.create_vertex_array().ok_or("Failed to create VAO")?;
        gl.bind_vertex_array(Some(&vao));

        let vertices: [f32; 12] = [
            -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0,
        ];

        let buffer = gl.create_buffer().ok_or("Failed to create buffer")?;
        gl.bind_buffer(GL::ARRAY_BUFFER, Some(&buffer));
        unsafe {
            let vert_array = js_sys::Float32Array::view(&vertices);
            gl.buffer_data_with_array_buffer_view(
                GL::ARRAY_BUFFER,
                &vert_array,
                GL::STATIC_DRAW,
            );
        }

        let pos_loc = gl.get_attrib_location(&program, "a_position") as u32;
        gl.enable_vertex_attrib_array(pos_loc);
        gl.vertex_attrib_pointer_with_i32(pos_loc, 2, GL::FLOAT, false, 0, 0);

        // Get uniform locations
        let u = |name: &str| -> Result<WebGlUniformLocation, String> {
            gl.get_uniform_location(&program, name)
                .ok_or_else(|| format!("Missing uniform: {name}"))
        };

        Ok(Self {
            u_time: u("u_time")?,
            u_phi: u("u_phi")?,
            u_arousal: u("u_arousal")?,
            u_valence: u("u_valence")?,
            u_rms: u("u_rms")?,
            u_centroid: u("u_centroid")?,
            u_flux: u("u_flux")?,
            u_orbs: u("u_orbs[0]")?,
            u_orb_colors: u("u_orb_colors[0]")?,
            u_orb_energy: u("u_orb_energy[0]")?,
            gl,
            program,
            width: w,
            height: h,
        })
    }

    /// Render one frame of the consciousness visualization.
    pub fn render(&self, params: &VizParams) {
        let gl = &self.gl;
        gl.use_program(Some(&self.program));

        // Set scalar uniforms
        gl.uniform1f(Some(&self.u_time), params.time);
        gl.uniform1f(Some(&self.u_phi), params.phi);
        gl.uniform1f(Some(&self.u_arousal), params.arousal);
        gl.uniform1f(Some(&self.u_valence), params.valence);
        gl.uniform1f(Some(&self.u_rms), params.rms);
        gl.uniform1f(Some(&self.u_centroid), params.centroid);
        gl.uniform1f(Some(&self.u_flux), params.flux);

        // Compute orb screen positions from binaural coordinates
        let orb_positions = compute_orb_positions(params.phi);
        let mut orb_flat = [0.0f32; 16]; // 8 * vec2
        for (i, (x, y)) in orb_positions.iter().enumerate() {
            orb_flat[i * 2] = *x;
            orb_flat[i * 2 + 1] = *y;
        }
        gl.uniform2fv_with_f32_array(Some(&self.u_orbs), &orb_flat);

        // Orb colors
        let mut colors_flat = [0.0f32; 24]; // 8 * vec3
        for (i, (r, g, b)) in HARMONY_COLORS.iter().enumerate() {
            colors_flat[i * 3] = *r;
            colors_flat[i * 3 + 1] = *g;
            colors_flat[i * 3 + 2] = *b;
        }
        gl.uniform3fv_with_f32_array(Some(&self.u_orb_colors), &colors_flat);

        // Orb energy from frequency data (map bins to harmonies)
        let energies = compute_orb_energies(&params.analysis);
        gl.uniform1fv_with_f32_array(Some(&self.u_orb_energy), &energies);

        // Draw fullscreen quad
        gl.draw_arrays(GL::TRIANGLES, 0, 6);
    }
}

/// Parameters for one visualization frame.
pub struct VizParams {
    pub time: f32,
    pub phi: f32,
    pub arousal: f32,
    pub valence: f32,
    pub rms: f32,
    pub centroid: f32,
    pub flux: f32,
    pub analysis: AudioAnalysis,
}

/// Project binaural harmony positions to screen-space UV [0,1].
/// Phi drives convergence: high phi = all orbs move toward center.
fn compute_orb_positions(phi: f32) -> [(f32, f32); 8] {
    let convergence = phi * phi; // quadratic convergence

    let mut positions = [(0.0f32, 0.0f32); 8];
    for (i, (az_deg, el_deg, dist)) in HARMONY_POSITIONS.iter().enumerate() {
        let az = az_deg.to_radians();
        let el = el_deg.to_radians();

        // Simple azimuth/elevation to screen projection
        // x: azimuth maps to horizontal (0° = center, ±180° = edges)
        // y: elevation maps to vertical (0° = center, +90° = top)
        let raw_x = 0.5 + az.sin() * 0.35 / dist;
        let raw_y = 0.5 - el.sin() * 0.35 / dist + az.cos().abs() * 0.05; // slight depth cue

        // Phi convergence: blend toward center (0.5, 0.5)
        let x = (raw_x + (0.5 - raw_x) * convergence).clamp(0.05, 0.95);
        let y = (raw_y + (0.5 - raw_y) * convergence).clamp(0.05, 0.95);

        positions[i] = (x, y);
    }
    positions
}

/// Map frequency bins to 8 harmony orb energies.
/// Each harmony gets a frequency band, normalized [0, 1].
fn compute_orb_energies(analysis: &AudioAnalysis) -> [f32; 8] {
    let bins = &analysis.frequency_data;
    let n = bins.len();
    if n == 0 {
        return [0.0; 8];
    }

    let mut energies = [0.0f32; 8];
    let band_size = n / 8;

    for i in 0..8 {
        let start = i * band_size;
        let end = ((i + 1) * band_size).min(n);
        let sum: u32 = bins[start..end].iter().map(|&b| b as u32).sum();
        let avg = sum as f32 / ((end - start) as f32 * 255.0);
        energies[i] = avg;
    }
    energies
}

// --- WebGL helpers ---

fn compile_shader(gl: &GL, shader_type: u32, source: &str) -> Result<WebGlShader, String> {
    let shader = gl
        .create_shader(shader_type)
        .ok_or("Failed to create shader")?;
    gl.shader_source(&shader, source);
    gl.compile_shader(&shader);

    if gl
        .get_shader_parameter(&shader, GL::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        let log = gl
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| "Unknown error".into());
        gl.delete_shader(Some(&shader));
        Err(format!("Shader compile error: {log}"))
    }
}

fn link_program(gl: &GL, vert: &WebGlShader, frag: &WebGlShader) -> Result<WebGlProgram, String> {
    let program = gl.create_program().ok_or("Failed to create program")?;
    gl.attach_shader(&program, vert);
    gl.attach_shader(&program, frag);
    gl.link_program(&program);

    if gl
        .get_program_parameter(&program, GL::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        let log = gl
            .get_program_info_log(&program)
            .unwrap_or_else(|| "Unknown error".into());
        gl.delete_program(Some(&program));
        Err(format!("Program link error: {log}"))
    }
}
