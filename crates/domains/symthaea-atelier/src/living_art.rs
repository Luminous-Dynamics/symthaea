// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Living art: self-evolving artworks with embedded temporal dynamics.
//!
//! Generates an HTML page containing a `<canvas>` element and a JavaScript
//! CfC kernel that continues evolving the artwork after generation. Each piece
//! is a living organism with its own temporal dynamics, gradually changing
//! based on the closed-form CfC solution running in the browser.
//!
//! The artwork has a finite lifespan — controlled by an energy budget from
//! the substrate model. When energy runs out, the art fades and dies.
//!
//! # Technical Approach
//!
//! The CfC closed-form solution is simple enough to implement in JavaScript:
//! ```javascript
//! x[i] = x_inf[i] + (x[i] - x_inf[i]) * Math.exp(-dt / tau);
//! ```
//!
//! We export the network's initial state (weights, hidden state, tau) and
//! the projection vectors as JSON embedded in the HTML. The JS runtime
//! evolves the state and repaints the canvas at ~30fps.

use crate::pixel_canvas::{Brushstroke, NeuralPainter};
use symthaea_canvas::CognitiveSnapshot;
use symthaea_core::genesis::GenesisSeed;

/// Configuration for living art generation.
pub struct LivingArtConfig {
    /// Canvas width in pixels.
    pub width: u32,
    /// Canvas height in pixels.
    pub height: u32,
    /// Initial number of brushstrokes (before the art starts living).
    pub initial_strokes: usize,
    /// Lifespan in seconds (how long the art evolves before fading).
    pub lifespan_secs: f32,
    /// Evolution speed (dt per animation frame).
    pub evolution_dt: f32,
    /// Maximum strokes per frame during evolution.
    pub strokes_per_frame: usize,
}

impl Default for LivingArtConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 512,
            initial_strokes: 40,
            lifespan_secs: 120.0, // 2 minutes
            evolution_dt: 0.02,
            strokes_per_frame: 5,
        }
    }
}

/// Generate a living art HTML page.
///
/// The page contains:
/// 1. A `<canvas>` element with the initial painting
/// 2. JavaScript CfC kernel that continues evolving
/// 3. Energy budget countdown (art fades when energy depletes)
/// 4. The consciousness metadata that birthed this piece
pub fn generate_living_art(snapshot: &CognitiveSnapshot, config: &LivingArtConfig) -> String {
    let genesis = GenesisSeed::from_phrase("living-art");
    let mut painter = NeuralPainter::new(&genesis);

    // Generate initial painting
    let (canvas, initial_strokes) = painter.paint(
        snapshot,
        config.width,
        config.height,
        config.initial_strokes,
    );

    // Export initial pixel data as base64
    let png_bytes = canvas.to_png_bytes();
    let png_b64 = base64_encode(&png_bytes);

    // Export CfC parameters for the JS kernel
    let harmony_json = format!(
        "[{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}]",
        snapshot.harmony_activations[0],
        snapshot.harmony_activations[1],
        snapshot.harmony_activations[2],
        snapshot.harmony_activations[3],
        snapshot.harmony_activations[4],
        snapshot.harmony_activations[5],
        snapshot.harmony_activations[6],
        snapshot.harmony_activations[7],
    );

    // Export stroke history for JS replay/continuation
    let strokes_json = strokes_to_json(&initial_strokes);

    generate_html(&png_b64, &harmony_json, &strokes_json, snapshot, config)
}

fn strokes_to_json(strokes: &[Brushstroke]) -> String {
    let entries: Vec<String> = strokes
        .iter()
        .map(|s| {
            format!(
                "{{x:{:.3},y:{:.3},r:{:.1},p:{:.2},cr:{:.2},cg:{:.2},cb:{:.2},a:{:.2},an:{:.2},el:{:.2}}}",
                s.x, s.y, s.radius, s.pressure, s.r, s.g, s.b, s.alpha, s.angle, s.elongation
            )
        })
        .collect();
    format!("[{}]", entries.join(","))
}

fn generate_html(
    png_b64: &str,
    harmony_json: &str,
    _strokes_json: &str,
    snapshot: &CognitiveSnapshot,
    config: &LivingArtConfig,
) -> String {
    format!(
        r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Living Art — Symthaea</title>
<style>
  body {{ margin: 0; background: #1a1a2e; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; font-family: monospace; color: #888; }}
  canvas {{ border: 1px solid #222; }}
  .info {{ margin-top: 12px; font-size: 12px; text-align: center; }}
  .energy {{ color: #4a9; }}
  .fading {{ color: #a44; }}
</style>
</head>
<body>
<canvas id="art" width="{w}" height="{h}"></canvas>
<div class="info">
  <div>Ψ = {psi:.2} | ♡ = {valence:.2} | ↑ = {arousal:.2}</div>
  <div class="energy" id="energy">Energy: 100%</div>
  <div id="status">Evolving...</div>
</div>
<script>
// ═══ Living Art: Consciousness-Driven Intelligent Painting ═══
const W = {w}, H = {h};
const LIFESPAN = {lifespan};
const DT = {dt};
const STROKES_PER_FRAME = {spf};

// Consciousness parameters
const harmonies = {harmony_json};
const psi = {psi};
const valence = {valence};
const arousal = {arousal};
const dopamine = {dopamine};
const serotonin = {serotonin};
const noradrenaline = {noradrenaline};

// ── Painting Temperament (computed ONCE — consciousness defines character) ──
const temperament = {{
  stroke_density: 0.15 + arousal * 0.6 + (1.0 - psi) * 0.25,
  position_variance: 0.05 + harmonies[3] * 0.3 + (1.0 - harmonies[7]) * 0.15,
  center_bias: 0.3 + harmonies[0] * 0.3 + harmonies[7] * 0.4,
  color_range: 20 + harmonies[3] * 200 + (1.0 - harmonies[7]) * 80,
  color_warmth: (valence + 1.0) / 2.0 * 0.6 + serotonin * 0.4,
  stroke_boldness: 0.2 + dopamine * 0.3 + noradrenaline * 0.2 + (1.0 - psi) * 0.1,
  stroke_softness: 0.3 + serotonin * 0.3 + harmonies[7] * 0.2 + (1.0 - arousal) * 0.2,
  restraint: harmonies[7] * 0.5 + psi * 0.3 + (1.0 - arousal) * 0.2,
  base_hue: valence > 0 ? 20 + (valence + 1.0) / 2.0 * 40 : 200 + (1.0 - (valence + 1.0) / 2.0) * 60
}};

// ── Compositional Phases ──
function get_phase(e) {{
  if (e > 0.80) return {{ name:'emergence',   dm:0.4,  bm:0.5, rm:0.6, ra:0.2  }};
  if (e > 0.60) return {{ name:'development', dm:0.7,  bm:0.8, rm:0.8, ra:0.0  }};
  if (e > 0.40) return {{ name:'peak',        dm:1.0,  bm:1.2, rm:1.0, ra:-0.1 }};
  if (e > 0.20) return {{ name:'resolution',  dm:0.5,  bm:0.7, rm:0.4, ra:0.15 }};
  return                {{ name:'stillness',   dm:0.15, bm:0.3, rm:0.2, ra:0.4  }};
}}

// ── Canvas Memory ──
const GRID = 4;
const mem = {{
  coverage: 0, cx: 0.5, cy: 0.5, tw: 0.001,
  dom_hue: temperament.base_hue, count: 0,
  hits: new Float32Array(GRID * GRID)
}};
function update_mem(sx, sy, hue, radius) {{
  const w = radius * radius;
  mem.tw += w;
  mem.cx += (sx / W - mem.cx) * w / mem.tw;
  mem.cy += (sy / H - mem.cy) * w / mem.tw;
  mem.dom_hue += (hue - mem.dom_hue) * 0.05;
  mem.count++;
  mem.coverage = Math.min(1.0, mem.count * radius * radius / (W * H) * 4);
  const gx = Math.min(GRID - 1, Math.floor(sx / W * GRID));
  const gy = Math.min(GRID - 1, Math.floor(sy / H * GRID));
  mem.hits[gy * GRID + gx]++;
}}
function least_dense() {{
  let mv = Infinity, mi = 0;
  for (let i = 0; i < GRID * GRID; i++) if (mem.hits[i] < mv) {{ mv = mem.hits[i]; mi = i; }}
  return {{ x: ((mi % GRID) + 0.5) / GRID, y: (Math.floor(mi / GRID) + 0.5) / GRID }};
}}

// ── CfC State ──
const STATE_DIM = 16;
let state = new Float32Array(STATE_DIM);
for (let i = 0; i < 8; i++) state[i] = harmonies[i];
state[8] = psi; state[9] = valence; state[10] = arousal;
state[11] = Math.random(); state[12] = Math.random();
state[13] = Math.random(); state[14] = Math.random(); state[15] = Math.random();
const tau = 0.05 + psi * 0.1;
const backbone = 0.3;

const canvas = document.getElementById('art');
const ctx = canvas.getContext('2d');
const img = new Image();
img.onload = () => {{ ctx.drawImage(img, 0, 0); startEvolution(); }};
img.src = 'data:image/png;base64,{png_b64}';

let energy = 1.0, elapsed = 0, strokeCount = 0;

function cfc_step() {{
  for (let i = 0; i < STATE_DIM; i++) {{
    const j = (i + 1) % STATE_DIM, k = (i + 5) % STATE_DIM;
    const x_inf = Math.tanh(state[i] * backbone + state[j] * 0.4 + state[k] * 0.2);
    const tau_eff = tau * (1.0 + Math.abs(state[i]) * backbone);
    state[i] = x_inf + (state[i] - x_inf) * Math.exp(-DT / tau_eff);
    state[i] += Math.sin(elapsed * (0.3 + i * 0.17) + i * 1.618) * 0.12 * energy;
    state[i] += (Math.random() - 0.5) * 0.04 * energy;
    if (Math.abs(state[i]) > 0.85) state[i] *= 0.6;
  }}
}}

// ── Intent-Driven Stroke Decode ──
function decode_stroke() {{
  const phase = get_phase(energy);
  const eff_density = temperament.stroke_density * phase.dm;
  const eff_restraint = Math.min(1.0, temperament.restraint + phase.ra);

  // 1. RESTRAINT: should we paint at all?
  const skip_prob = 1.0 - eff_density * (1.0 - mem.coverage * eff_restraint);
  if (Math.random() < skip_prob) return null;

  // 2. POSITION: intent-driven
  let x = mem.cx + state[0] * temperament.position_variance;
  let y = mem.cy + state[1] * temperament.position_variance;
  x += (0.5 - x) * temperament.center_bias * 0.5;
  y += (0.5 - y) * temperament.center_bias * 0.5;

  // Redirect away from saturated regions
  const gx = Math.min(GRID-1, Math.max(0, Math.floor(x * GRID)));
  const gy = Math.min(GRID-1, Math.max(0, Math.floor(y * GRID)));
  const local_d = mem.hits[gy * GRID + gx];
  const mean_d = mem.count / (GRID * GRID) + 1;
  if (local_d > mean_d * 2.0 && Math.random() < 0.5) {{
    const ld = least_dense();
    x = x * 0.3 + ld.x * 0.7;
    y = y * 0.3 + ld.y * 0.7;
  }}
  x = Math.max(0.02, Math.min(0.98, x));
  y = Math.max(0.02, Math.min(0.98, y));

  // 3. COLOR: temperament-constrained
  const eff_range = temperament.color_range * phase.rm;
  const hue_off = (state[4] * 0.5 + state[5] * 0.3) * eff_range - eff_range * 0.5;
  let hue = (temperament.base_hue + hue_off + elapsed * 2.5) % 360;
  if (hue < 0) hue += 360;
  const sat = 0.2 + temperament.stroke_boldness * 0.5 + Math.abs(state[5]) * 0.15;
  const lit = 0.25 + psi * 0.15 + Math.abs(state[6]) * 0.15;
  const [r, g, b] = hslToRgb(hue, sat, lit);

  // 4. SIZE: phase-scaled
  const eff_bold = temperament.stroke_boldness * phase.bm;
  const radius = 6 + eff_bold * 50 + Math.abs(state[2]) * 15;
  const alpha = (0.12 + eff_bold * 0.35) * (0.3 + energy * 0.7)
                * (1.0 - temperament.stroke_softness * 0.3);

  return {{ x: x*W, y: y*H, radius, pressure: 0.4 + eff_bold * 0.4, r, g, b, alpha, hue }};
}}

function hslToRgb(h, s, l) {{
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs((h / 60) % 2 - 1));
  const m = l - c / 2;
  let r, g, b;
  if (h < 60)      {{ r = c; g = x; b = 0; }}
  else if (h < 120) {{ r = x; g = c; b = 0; }}
  else if (h < 180) {{ r = 0; g = c; b = x; }}
  else if (h < 240) {{ r = 0; g = x; b = c; }}
  else if (h < 300) {{ r = x; g = 0; b = c; }}
  else              {{ r = c; g = 0; b = x; }}
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}}

function paint_stroke(s) {{
  ctx.globalAlpha = s.alpha * s.pressure;
  ctx.fillStyle = `rgb(${{Math.floor(s.r)}},${{Math.floor(s.g)}},${{Math.floor(s.b)}})`;
  ctx.beginPath();
  ctx.arc(s.x, s.y, s.radius * 0.6, 0, Math.PI * 2);
  ctx.fill();
  const gradient = ctx.createRadialGradient(s.x, s.y, s.radius * 0.3, s.x, s.y, s.radius);
  gradient.addColorStop(0, `rgba(${{Math.floor(s.r)}},${{Math.floor(s.g)}},${{Math.floor(s.b)}},0.5)`);
  gradient.addColorStop(1, `rgba(${{Math.floor(s.r)}},${{Math.floor(s.g)}},${{Math.floor(s.b)}},0)`);
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(s.x, s.y, s.radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
}}

function startEvolution() {{
  function frame() {{
    elapsed += DT;
    energy = Math.max(0, 1.0 - elapsed / LIFESPAN);
    cfc_step();
    for (let i = 0; i < STROKES_PER_FRAME; i++) {{
      cfc_step();
      const s = decode_stroke();
      if (s && s.alpha > 0.005) {{
        paint_stroke(s);
        update_mem(s.x, s.y, s.hue || 0, s.radius);
        strokeCount++;
      }}
    }}
    const ph = get_phase(energy).name;
    const energyEl = document.getElementById('energy');
    const statusEl = document.getElementById('status');
    energyEl.textContent = `Energy: ${{(energy * 100).toFixed(1)}}%`;
    energyEl.className = energy > 0.3 ? 'energy' : 'fading';
    statusEl.textContent = energy > 0
      ? `${{ph}} (${{strokeCount}} strokes, ${{elapsed.toFixed(1)}}s)`
      : `resting (${{strokeCount}} strokes)`;
    if (energy > 0) requestAnimationFrame(frame);
  }}
  requestAnimationFrame(frame);
}}
</script>
</body>
</html>"##,
        w = config.width,
        h = config.height,
        psi = snapshot.consciousness_level,
        valence = snapshot.valence,
        arousal = snapshot.arousal,
        dopamine = snapshot.dopamine,
        serotonin = snapshot.serotonin,
        noradrenaline = snapshot.noradrenaline,
        lifespan = config.lifespan_secs,
        dt = config.evolution_dt,
        spf = config.strokes_per_frame,
        harmony_json = harmony_json,
        png_b64 = png_b64,
    )
}

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 63) as usize] as char);
        result.push(CHARS[((triple >> 12) & 63) as usize] as char);
        result.push(if chunk.len() > 1 {
            CHARS[((triple >> 6) & 63) as usize] as char
        } else {
            '='
        });
        result.push(if chunk.len() > 2 {
            CHARS[(triple & 63) as usize] as char
        } else {
            '='
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_snapshot() -> CognitiveSnapshot {
        CognitiveSnapshot {
            consciousness_level: 0.7,
            valence: 0.3,
            arousal: 0.5,
            dopamine: 0.6,
            serotonin: 0.5,
            noradrenaline: 0.4,
            harmony_activations: [0.5, 0.6, 0.4, 0.7, 0.3, 0.5, 0.8, 0.2],
            prediction_error: 0.1,
            thought_vector: vec![0.3, -0.2],
            ..CognitiveSnapshot::dormant()
        }
    }

    #[test]
    fn generates_valid_html() {
        let config = LivingArtConfig {
            width: 64,
            height: 64,
            initial_strokes: 10,
            lifespan_secs: 30.0,
            ..Default::default()
        };
        let html = generate_living_art(&test_snapshot(), &config);

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<canvas"));
        assert!(html.contains("cfc_step"));
        assert!(html.contains("requestAnimationFrame"));
        assert!(html.contains("data:image/png;base64,"));
    }

    #[test]
    fn html_contains_consciousness_params() {
        let config = LivingArtConfig {
            width: 64,
            height: 64,
            initial_strokes: 5,
            ..Default::default()
        };
        let snap = test_snapshot();
        let html = generate_living_art(&snap, &config);

        assert!(html.contains("const psi = 0.7"));
        assert!(html.contains("const valence = 0.3"));
        assert!(html.contains("const arousal = 0.5"));
    }

    #[test]
    fn different_states_different_art() {
        let config = LivingArtConfig {
            width: 32,
            height: 32,
            initial_strokes: 5,
            ..Default::default()
        };

        let html1 = generate_living_art(
            &CognitiveSnapshot {
                consciousness_level: 0.9,
                valence: 0.8,
                ..CognitiveSnapshot::dormant()
            },
            &config,
        );
        let html2 = generate_living_art(
            &CognitiveSnapshot {
                consciousness_level: 0.3,
                valence: -0.8,
                ..CognitiveSnapshot::dormant()
            },
            &config,
        );

        assert_ne!(html1, html2);
    }
}
