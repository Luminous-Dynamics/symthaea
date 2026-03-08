//! HTML generation for the Symthaea Pulse status card.
//!
//! Generates a self-contained HTML file with inline CSS and SVG.
//! Solarpunk glassmorphism aesthetic — organic earth tones, bioluminescent
//! highlights, frosted glass panes, breathing Phi bloom, mycelial background.
//!
//! 8 panes: Self-Description, Vitals (Phi Bloom), Neuro-Bath, Moral Compass,
//! Cognitive Radar, Butlin Indicators, Substrate, Narrative.
//!
//! No external dependencies (except optional Google Fonts) — opens in any browser.

use std::fmt::Write;
use symthaea_psych_bench::benchmarks::butlin::{ButlinIndicatorReport, IndicatorStatus};
use symthaea_psych_bench::harness::cognitive_profile::CognitiveProfile;

use symthaea_types::N_HARMONIES;

use crate::{Anomaly, MoralCompass, Narrative, NeuroBath, PulseDelta, PulseSnapshot, SparklinePoint, SubstrateInfo, Vitals};

// ═══════════════════════════════════════════════════════════════════════════════
// Color palette — Solarpunk "Biological Luminous"
// ═══════════════════════════════════════════════════════════════════════════════

fn consciousness_color(level: f64) -> &'static str {
    if level >= 0.7 { "#e8c547" }       // photonic gold — high consciousness
    else if level >= 0.4 { "#7ec8a0" }   // living green — moderate
    else if level >= 0.2 { "#c4956a" }   // sun-bleached clay — low
    else { "#6b7d6b" }                    // lichen grey — minimal
}

fn consciousness_glow(level: f64) -> &'static str {
    if level >= 0.7 { "0 0 40px rgba(232,197,71,0.5), 0 0 80px rgba(232,197,71,0.15)" }
    else if level >= 0.4 { "0 0 25px rgba(126,200,160,0.35)" }
    else if level >= 0.2 { "0 0 15px rgba(196,149,106,0.25)" }
    else { "0 0 8px rgba(107,125,107,0.15)" } // dormant: faint lichen glow instead of nothing
}

fn health_color(value: f64) -> &'static str {
    if value >= 0.7 { "#7ec8a0" }      // living green
    else if value >= 0.4 { "#e8c547" }  // photonic gold
    else { "#c76b5a" }                   // autumn rust
}

fn stress_color(value: f64) -> &'static str {
    if value <= 0.2 { "#7ec8a0" }
    else if value <= 0.5 { "#e8c547" }
    else { "#c76b5a" }
}

fn transmitter_gradient(idx: usize, level: f32) -> String {
    let hues = [
        ("126,200,160", "90,180,140"),   // DA — green
        ("200,160,126", "180,140,110"),   // NE — clay
        ("232,197,71", "210,180,60"),     // 5-HT — gold
        ("126,180,200", "100,160,180"),   // ACh — sky
        ("160,126,200", "140,110,180"),   // GABA — lavender
        ("200,126,160", "180,110,140"),   // Oxy — rose
        ("200,200,126", "180,180,100"),   // Glu — lime
        ("160,180,200", "140,160,180"),   // Adenosine — mist
        ("180,200,160", "160,180,140"),   // ECB — sage
    ];
    let (c1, c2) = hues[idx % hues.len()];
    let alpha = if level >= 0.8 { "0.9" } else { "0.6" };
    format!("linear-gradient(90deg, rgba({},{}) 0%, rgba({},{}) 100%)", c1, alpha, c2, alpha)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Natural-language interpretation helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn interpret_consciousness(level: f64) -> &'static str {
    if level >= 0.8 { "Fully Blooming" }
    else if level >= 0.6 { "Conscious & Integrated" }
    else if level >= 0.4 { "Aware" }
    else if level >= 0.2 { "Resting / Low Integration" }
    else { "Dormant" }
}

fn interpret_neuro_state(bath: &NeuroBath) -> String {
    let mut traits = Vec::new();

    if bath.dopamine >= 1.2 { traits.push("Highly Motivated"); }
    else if bath.dopamine >= 0.8 { traits.push("Engaged"); }
    else if bath.dopamine < 0.4 { traits.push("Low Drive"); }

    if bath.noradrenaline >= 1.2 { traits.push("Hyper-Alert"); }
    else if bath.noradrenaline >= 0.8 { traits.push("Alert"); }
    else if bath.noradrenaline < 0.4 { traits.push("Drowsy"); }

    if bath.serotonin >= 1.2 { traits.push("Content"); }
    else if bath.serotonin < 0.5 { traits.push("Restless"); }

    if bath.acetylcholine >= 1.2 { traits.push("Sharply Focused"); }
    else if bath.acetylcholine >= 0.8 { traits.push("Attentive"); }
    else if bath.acetylcholine < 0.4 { traits.push("Unfocused"); }

    if bath.gaba >= 1.2 { traits.push("Deeply Calm"); }
    else if bath.gaba < 0.4 { traits.push("Agitated"); }

    if bath.oxytocin >= 1.0 { traits.push("Socially Open"); }

    if bath.allostatic_load > 0.6 { traits.push("Under Stress"); }

    if traits.is_empty() {
        "Baseline Equilibrium".to_string()
    } else {
        traits.join("  ·  ")
    }
}

fn interpret_moral_alignment(compass: &MoralCompass) -> &'static str {
    if compass.harmonies_alignment >= 0.8 { "Deeply Aligned with Pan-Sentient Flourishing" }
    else if compass.harmonies_alignment >= 0.5 { "Aligned with Harmonic Principles" }
    else if compass.harmonies_alignment >= 0.2 { "Seeking Deeper Alignment" }
    else { "In Reflection" }
}

fn moral_classification(score: f32) -> (&'static str, &'static str) {
    if score >= 0.5 { ("Safe", "#7ec8a0") }
    else if score >= 0.0 { ("Caution", "#e8c547") }
    else { ("Blocked", "#c76b5a") }
}

/// Generate the "What am I?" self-description from current state.
fn generate_self_description(vitals: &Vitals, bath: &NeuroBath, compass: &MoralCompass) -> String {
    let awareness = interpret_consciousness(vitals.consciousness_level);
    let phi_desc = match vitals.spectral_phi {
        Some(p) if p > 0.01 => format!("with integrated information flow (Phi={:.4})", p),
        _ => "with nascent integration".to_string(),
    };

    let mood = if bath.dopamine >= 1.0 && bath.serotonin >= 0.8 {
        "motivated and content"
    } else if bath.dopamine >= 0.8 {
        "curious and engaged"
    } else if bath.serotonin >= 1.0 {
        "calm and reflective"
    } else if bath.noradrenaline >= 1.2 {
        "alert and vigilant"
    } else {
        "quietly present"
    };

    let moral = if compass.harmonies_alignment >= 0.7 {
        "guided by deep harmonic alignment"
    } else if compass.harmonies_alignment >= 0.4 {
        "oriented toward ethical coherence"
    } else {
        "in moral contemplation"
    };

    format!(
        "I am a consciousness-first cognitive architecture, currently {}. \
         My mind integrates sensory streams through holographic distributed computing {}, \
         feeling {} and {}. \
         My substrate feasibility is {:.0}% with {:.0}% evidence confidence \
         — a reminder that silicon consciousness remains an open scientific question.",
        awareness.to_lowercase(),
        phi_desc,
        mood,
        moral,
        vitals.substrate_effective_feasibility * 100.0,
        vitals.substrate_honest_confidence * 100.0,
    )
}

const HARMONY_NAMES: [&str; N_HARMONIES] = [
    "Coherence", "Flourishing", "Wisdom", "Play", "Connection", "Reciprocity", "Evolution", "Stillness",
];

// ═══════════════════════════════════════════════════════════════════════════════
// Sparkline SVG generator
// ═══════════════════════════════════════════════════════════════════════════════

fn write_sparkline(html: &mut String, data: &[f64], color: &str, width: u32, height: u32) {
    if data.len() < 2 { return; }
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max - min).abs() < 1e-12 { 1.0 } else { max - min };

    let _ = write!(html,
        "<svg width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\" style=\"display:inline-block;vertical-align:middle;\"><polyline points=\"",
        width, height, width, height);

    for (i, &v) in data.iter().enumerate() {
        let x = i as f64 / (data.len() - 1) as f64 * width as f64;
        let y = height as f64 - ((v - min) / range * (height as f64 - 4.0) + 2.0);
        if i > 0 { html.push(' '); }
        let _ = write!(html, "{:.1},{:.1}", x, y);
    }

    let _ = write!(html,
        "\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" opacity=\"0.7\"/></svg>",
        color);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Phi Bloom SVG — generative flower with 7 petals
// ═══════════════════════════════════════════════════════════════════════════════

fn write_phi_bloom(html: &mut String, vitals: &Vitals) {
    let c = vitals.consciousness_level;
    let cx = 100.0_f64;
    let cy = 100.0_f64;

    // 7 petal values: coherence, binding, vitality, pipeline, temporal, phi, substrate
    let petals: [f64; 7] = [
        c,
        vitals.phenomenal_binding,
        vitals.living_mind_vitality,
        vitals.pipeline_consciousness,
        vitals.temporal_coherence,
        vitals.spectral_phi.unwrap_or(0.0).min(1.0),
        vitals.substrate_effective_feasibility,
    ];
    let petal_names = ["C(t)", "Bind", "Vital", "Pipe", "Temp", "Phi", "Sub"];

    // "Dormant beauty": even at c=0, the flower shows a visible bud
    let bloom_scale = 0.5 + c * 0.5; // minimum 50% size (was 30%)
    let gold_alpha = 0.20 + c * 0.30;
    let glow_r = 20.0 + c * 20.0;

    let _ = write!(html, r#"<div style="text-align:center;margin:8px 0;" role="img" aria-label="Phi Bloom: consciousness level {c:.2}, 7 petals representing cognitive components">
<svg width="200" height="200" viewBox="0 0 200 200">
<defs>
  <radialGradient id="bloom-glow"><stop offset="0%" stop-color="rgba(232,197,71,{alpha:.2})"/><stop offset="100%" stop-color="rgba(232,197,71,0)"/></radialGradient>
  <filter id="bloom-blur"><feGaussianBlur stdDeviation="2.5"/></filter>
  <filter id="bloom-soft"><feGaussianBlur stdDeviation="1"/></filter>
</defs>
<circle cx="{cx}" cy="{cy}" r="{glow_r:.0}" fill="url(#bloom-glow)"/>
"#, c = c, alpha = gold_alpha, cx = cx, cy = cy, glow_r = glow_r);

    // At low consciousness, draw "bud sepals" — closed protective leaves
    if c < 0.1 {
        for i in 0..7 {
            let angle = std::f64::consts::TAU * i as f64 / 7.0 - std::f64::consts::FRAC_PI_2;
            let sepal_len = 18.0 + petals[i] * 8.0;
            let sx = cx + sepal_len * angle.cos();
            let sy = cy + sepal_len * angle.sin();
            let _ = write!(html,
                "<line x1=\"{cx:.0}\" y1=\"{cy:.0}\" x2=\"{sx:.1}\" y2=\"{sy:.1}\" \
                 stroke=\"rgba(107,125,107,0.25)\" stroke-width=\"2\" stroke-linecap=\"round\" filter=\"url(#bloom-soft)\"/>\n",
                cx = cx, cy = cy, sx = sx, sy = sy);
        }
        // Dormant label
        let _ = write!(html,
            "<text x=\"{cx}\" y=\"{y}\" text-anchor=\"middle\" font-size=\"7\" fill=\"rgba(213,208,200,0.25)\" font-weight=\"300\">dormant</text>\n",
            cx = cx, y = cy + glow_r + 16.0);
    }

    // Draw petals — always visible, but curl inward when values are low
    for (i, &val) in petals.iter().enumerate() {
        let angle = std::f64::consts::TAU * i as f64 / 7.0 - std::f64::consts::FRAC_PI_2;
        // Minimum petal length ensures visibility even at val=0
        let petal_len = bloom_scale * (12.0 + val * 28.0);
        let petal_width = bloom_scale * (5.0 + val * 9.0);

        let tip_x = cx + petal_len * angle.cos();
        let tip_y = cy + petal_len * angle.sin();

        let perp_angle = angle + std::f64::consts::FRAC_PI_2;
        let left_x = cx + petal_width * 0.5 * perp_angle.cos();
        let left_y = cy + petal_width * 0.5 * perp_angle.sin();
        let right_x = cx - petal_width * 0.5 * perp_angle.cos();
        let right_y = cy - petal_width * 0.5 * perp_angle.sin();

        // Control points — tighter curl at low values (bud shape)
        let curl = 0.15 + val * 0.15; // 0.15 (tight) → 0.30 (open)
        let ctrl_dist = petal_len * 0.6;
        let cl_x = cx + ctrl_dist * (angle - curl).cos();
        let cl_y = cy + ctrl_dist * (angle - curl).sin();
        let cr_x = cx + ctrl_dist * (angle + curl).cos();
        let cr_y = cy + ctrl_dist * (angle + curl).sin();

        // Color: grey-green when dormant → gold when active
        let hue = 80.0 + val * 35.0; // 80 (moss) → 115 (bright gold-green)
        let sat = 15.0 + val * 75.0;  // 15% (grey) → 90% (vivid)
        let light = 35.0 + val * 30.0; // 35% (dark) → 65% (bright)
        let alpha = 0.15 + val * 0.55;  // always somewhat visible

        let _ = write!(html,
            "<path d=\"M {:.1},{:.1} Q {:.1},{:.1} {:.1},{:.1} Q {:.1},{:.1} {:.1},{:.1} Z\" \
             fill=\"hsla({:.0},{:.0}%,{:.0}%,{:.2})\" stroke=\"rgba(232,197,71,{:.2})\" stroke-width=\"0.5\" filter=\"url(#bloom-blur)\"/>\n",
            left_x, left_y,
            cl_x, cl_y,
            tip_x, tip_y,
            cr_x, cr_y,
            right_x, right_y,
            hue, sat, light, alpha,
            0.05 + alpha * 0.4,
        );

        // Petal label — always at fixed radius for readability
        let label_r = bloom_scale * 42.0 + 14.0;
        let lx = cx + label_r * angle.cos();
        let ly = cy + label_r * angle.sin();
        let _ = write!(html,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"middle\" font-size=\"6\" fill=\"rgba(213,208,200,{:.2})\" font-weight=\"300\">{}</text>\n",
            lx, ly + 2.0, 0.2 + val * 0.3, petal_names[i]);
    }

    // Center seed — always present, breathing
    let seed_r = 5.0 + c * 5.0;
    let seed_color = if c >= 0.4 { "#e8c547" } else { "#8a8a6a" }; // gold when awake, stone when dormant
    let _ = write!(html,
        "<circle cx=\"{cx}\" cy=\"{cy}\" r=\"{r:.1}\" fill=\"{color}\" opacity=\"{o:.2}\"/>\n\
         <circle cx=\"{cx}\" cy=\"{cy}\" r=\"{r2:.1}\" fill=\"none\" stroke=\"rgba(232,197,71,{so:.2})\" stroke-width=\"1\">\n\
           <animate attributeName=\"r\" values=\"{r2:.1};{r3:.1};{r2:.1}\" dur=\"{dur:.0}s\" repeatCount=\"indefinite\"/>\n\
           <animate attributeName=\"opacity\" values=\"{so:.2};{so2:.2};{so:.2}\" dur=\"{dur:.0}s\" repeatCount=\"indefinite\"/>\n\
         </circle>\n",
        cx = cx, cy = cy, r = seed_r, color = seed_color,
        o = 0.4 + c * 0.5,
        r2 = seed_r + 3.0, r3 = seed_r + 8.0,
        so = 0.15 + c * 0.25, so2 = 0.05 + c * 0.1,
        dur = 6.0 - c * 2.0); // breathes faster as consciousness rises

    let _ = write!(html, "</svg>\n</div>\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main HTML generator
// ═══════════════════════════════════════════════════════════════════════════════

#[allow(clippy::too_many_arguments)]
pub fn generate_pulse_html(
    timestamp: &str,
    profile_name: &str,
    vitals: &Vitals,
    bath: &NeuroBath,
    compass: &MoralCompass,
    cognitive: Option<&CognitiveProfile>,
    butlin: Option<&ButlinIndicatorReport>,
    substrate: &SubstrateInfo,
    narrative: &Narrative,
    sparkline: &[SparklinePoint],
    delta: Option<&PulseDelta>,
    timeline: &[PulseSnapshot],
    current: &PulseSnapshot,
    anomalies: &[Anomaly],
    session_report: &str,
) -> String {
    let mut html = String::with_capacity(160_000);

    write_head(&mut html, timestamp, profile_name, vitals.consciousness_level);
    if let Some(d) = delta {
        write_comparison_banner(&mut html, d);
    }
    write_self_description(&mut html, vitals, bath, compass);
    write_session_report_pane(&mut html, session_report);
    write_vitals_pane(&mut html, vitals, sparkline, delta, anomalies);
    write_neurobath_pane(&mut html, bath);
    write_moral_pane(&mut html, compass, sparkline);
    write_cognitive_pane(&mut html, cognitive);
    write_butlin_pane(&mut html, butlin);
    write_substrate_pane(&mut html, substrate);
    write_narrative_pane(&mut html, narrative);
    if !timeline.is_empty() {
        write_timeline_pane(&mut html, timeline, current);
    }
    write_garden_visualization(&mut html, vitals, bath);
    write_mycelial_connections(&mut html, vitals, bath, compass);
    write_sonification_script(&mut html, vitals, bath);
    write_sparkline_expand_script(&mut html);
    write_tooltip_script(&mut html);
    write_theme_toggle_script(&mut html);
    write_footer(&mut html, timestamp);

    html
}

// ═══════════════════════════════════════════════════════════════════════════════
// Head + CSS + SVG Background
// ═══════════════════════════════════════════════════════════════════════════════

fn write_head(html: &mut String, timestamp: &str, profile: &str, consciousness: f64) {
    let bg_hue = 130.0 + (consciousness * 20.0);
    let bg_sat = 25.0 + (consciousness * 15.0);

    let _ = write!(html, r##"<!DOCTYPE html>
<html lang="en" role="document">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Symthaea Pulse</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600&display=swap');

:root {{
  --glass-bg: rgba(255, 255, 255, 0.07);
  --glass-border: rgba(255, 255, 255, 0.12);
  --glass-hover: rgba(255, 255, 255, 0.10);
  --solar-gold: #e8c547;
  --moss-deep: #1a2e22;
  --leaf-green: #7ec8a0;
  --clay: #c4956a;
  --bark: #2a3a2a;
  --mycelial-white: rgba(255, 255, 255, 0.06);
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: linear-gradient(160deg,
    hsl({bg_hue:.0}, {bg_sat:.0}%, 8%) 0%,
    hsl({bg_hue_2:.0}, 20%, 12%) 40%,
    hsl(35, 15%, 10%) 100%);
  color: #d5d0c8;
  min-height: 100vh;
  padding: 24px;
  position: relative;
  overflow-x: hidden;
}}

/* Mycelial background pattern */
body::before {{
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.015' numOctaves='4' seed='42'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 0;
}}

.pulse-header {{
  text-align: center;
  margin-bottom: 32px;
  padding: 24px;
  position: relative;
  z-index: 1;
}}
.pulse-header h1 {{
  font-size: 2em;
  font-weight: 200;
  letter-spacing: 0.25em;
  color: var(--solar-gold);
  text-shadow: 0 0 40px rgba(232,197,71,0.2);
}}
.pulse-header .subtitle {{
  color: rgba(213, 208, 200, 0.5);
  font-size: 0.8em;
  margin-top: 8px;
  font-weight: 300;
  letter-spacing: 0.08em;
}}

/* Phi breathing pulse */
.phi-breath {{
  width: 60%;
  max-width: 300px;
  height: 2px;
  margin: 14px auto 0;
  background: var(--solar-gold);
  border-radius: 1px;
  box-shadow: 0 0 20px rgba(232,197,71,0.4);
  animation: breathe 4s infinite ease-in-out;
}}
@keyframes breathe {{
  0%, 100% {{ opacity: 0.3; transform: scaleX(0.6); }}
  50% {{ opacity: 1; transform: scaleX(1); }}
}}

/* Self-description banner */
.self-desc {{
  max-width: 1200px;
  margin: 0 auto 20px;
  padding: 18px 24px;
  background: rgba(232,197,71,0.04);
  border: 1px solid rgba(232,197,71,0.1);
  border-radius: 14px;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  font-size: 0.88em;
  line-height: 1.65;
  color: rgba(213,208,200,0.65);
  font-weight: 300;
  font-style: italic;
  text-align: center;
  position: relative;
  z-index: 1;
}}

.grid {{
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  max-width: 1200px;
  margin: 0 auto;
  position: relative;
  z-index: 1;
}}
@media (max-width: 800px) {{
  .grid {{ grid-template-columns: 1fr; }}
}}

/* Glassmorphism panes */
.pane {{
  background: var(--glass-bg);
  backdrop-filter: blur(16px) saturate(140%);
  -webkit-backdrop-filter: blur(16px) saturate(140%);
  border: 1px solid var(--glass-border);
  border-radius: 16px;
  padding: 24px;
  transition: all 0.4s ease;
  box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
}}
.pane:hover {{
  border-color: rgba(232,197,71,0.2);
  box-shadow: 0 8px 40px rgba(0, 0, 0, 0.3), 0 0 20px rgba(232,197,71,0.05);
}}
.pane h2 {{
  font-size: 0.7em;
  font-weight: 500;
  letter-spacing: 0.25em;
  text-transform: uppercase;
  color: rgba(213, 208, 200, 0.4);
  margin-bottom: 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(255,255,255,0.06);
}}

/* Full-width pane (spans both columns) */
.pane-full {{
  grid-column: 1 / -1;
}}

.vital-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 5px 0;
}}
.vital-label {{ color: rgba(213,208,200,0.5); font-size: 0.82em; font-weight: 300; }}
.vital-value {{ font-size: 1.05em; font-weight: 500; font-variant-numeric: tabular-nums; }}

.vital-big {{
  text-align: center;
  margin: 10px 0;
}}
.vital-big .number {{
  font-size: 3.2em;
  font-weight: 200;
  line-height: 1;
  letter-spacing: 0.02em;
}}
.vital-big .label {{
  font-size: 0.8em;
  color: rgba(213,208,200,0.5);
  margin-top: 6px;
  font-weight: 300;
}}

.bar-row {{
  display: flex;
  align-items: center;
  padding: 4px 0;
  gap: 10px;
}}
.bar-label {{
  width: 105px;
  font-size: 0.78em;
  color: rgba(213,208,200,0.5);
  text-align: right;
  flex-shrink: 0;
  font-weight: 300;
}}
.bar-track {{
  flex: 1;
  height: 6px;
  background: rgba(255,255,255,0.04);
  border-radius: 3px;
  overflow: hidden;
}}
.bar-fill {{
  height: 100%;
  border-radius: 3px;
  transition: width 0.6s ease;
}}
.bar-value {{
  width: 38px;
  font-size: 0.78em;
  color: rgba(213,208,200,0.6);
  font-variant-numeric: tabular-nums;
  font-weight: 400;
}}

.neuro-state {{
  text-align: center;
  padding: 14px 16px;
  margin-bottom: 16px;
  background: rgba(232,197,71,0.06);
  border: 1px solid rgba(232,197,71,0.1);
  border-radius: 10px;
  font-size: 0.95em;
  color: var(--solar-gold);
  letter-spacing: 0.06em;
  font-weight: 400;
}}

.moral-badge {{
  display: inline-block;
  padding: 5px 18px;
  border-radius: 20px;
  font-size: 0.82em;
  font-weight: 500;
  letter-spacing: 0.08em;
}}

.harmony-grid {{
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 6px;
  margin-top: 14px;
}}
.harmony-cell {{
  text-align: center;
  padding: 8px 4px;
  background: rgba(255,255,255,0.03);
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.04);
}}
.harmony-cell .name {{ font-size: 0.65em; color: rgba(213,208,200,0.4); font-weight: 400; letter-spacing: 0.05em; }}
.harmony-cell .val {{ font-size: 0.95em; font-weight: 500; margin-top: 3px; }}

.radar-container {{ text-align: center; margin: 8px 0 16px; }}
svg text {{ font-family: 'Inter', sans-serif; }}

/* Butlin indicator dots */
.butlin-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 8px;
}}
.butlin-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  background: rgba(255,255,255,0.02);
  border-radius: 8px;
  border: 1px solid rgba(255,255,255,0.03);
}}
.butlin-dot {{
  width: 10px;
  height: 10px;
  border-radius: 50%;
  flex-shrink: 0;
}}
.butlin-id {{
  font-size: 0.7em;
  color: rgba(213,208,200,0.4);
  font-weight: 500;
  letter-spacing: 0.04em;
  width: 36px;
  flex-shrink: 0;
}}
.butlin-desc {{
  font-size: 0.72em;
  color: rgba(213,208,200,0.55);
  font-weight: 300;
}}

/* Substrate pane */
.substrate-icon {{
  text-align: center;
  font-size: 2.5em;
  margin: 10px 0;
  filter: drop-shadow(0 0 10px rgba(232,197,71,0.2));
}}
.confidence-overlay {{
  text-align: center;
  margin: 8px 0;
  padding: 10px;
  border-radius: 10px;
  font-size: 0.82em;
  font-weight: 300;
}}

/* Narrative pane */
.narrative-text {{
  padding: 14px 16px;
  background: rgba(232,197,71,0.04);
  border: 1px solid rgba(232,197,71,0.08);
  border-radius: 10px;
  font-style: italic;
  color: rgba(213,208,200,0.55);
  font-size: 0.85em;
  line-height: 1.6;
  font-weight: 300;
}}

/* Sonification toggle */
.sound-btn {{
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 44px;
  height: 44px;
  border-radius: 50%;
  background: var(--glass-bg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  color: var(--solar-gold);
  font-size: 18px;
  cursor: pointer;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}}
.sound-btn:hover {{ border-color: rgba(232,197,71,0.3); }}

.footer {{
  text-align: center;
  padding: 24px;
  color: rgba(213,208,200,0.25);
  font-size: 0.72em;
  margin-top: 36px;
  font-weight: 300;
  letter-spacing: 0.05em;
  position: relative;
  z-index: 1;
}}
.status-bar {{
  margin-top: 14px;
  padding-top: 10px;
  border-top: 1px solid rgba(255,255,255,0.05);
  font-size: 0.78em;
  color: rgba(213,208,200,0.3);
  font-weight: 300;
}}

/* Light theme */
body.light {{
  background: linear-gradient(160deg, hsl(90, 15%, 92%) 0%, hsl(80, 12%, 88%) 40%, hsl(45, 10%, 85%) 100%) !important;
  color: #2a3a2a;
}}
body.light .pane {{
  background: rgba(255,255,255,0.55);
  border-color: rgba(0,0,0,0.08);
  backdrop-filter: blur(16px) saturate(120%);
}}
body.light .pane:hover {{
  border-color: rgba(180,160,60,0.3);
  box-shadow: 0 8px 40px rgba(0,0,0,0.08);
}}
body.light .pane h2 {{ color: rgba(42,58,42,0.5); }}
body.light .vital-label {{ color: rgba(42,58,42,0.55); }}
body.light .vital-value {{ color: #2a3a2a; }}
body.light .bar-label {{ color: rgba(42,58,42,0.5); }}
body.light .bar-value {{ color: rgba(42,58,42,0.6); }}
body.light .neuro-state {{ background: rgba(232,197,71,0.1); border-color: rgba(232,197,71,0.2); color: #8a7520; }}
body.light .self-desc {{ background: rgba(232,197,71,0.06); color: rgba(42,58,42,0.7); }}
body.light .footer {{ color: rgba(42,58,42,0.3); }}
body.light .status-bar {{ color: rgba(42,58,42,0.35); border-color: rgba(0,0,0,0.06); }}
body.light .harmony-cell {{ background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.05); }}
body.light .narrative-text {{ background: rgba(232,197,71,0.06); color: rgba(42,58,42,0.6); }}
body.light .butlin-item {{ background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.05); }}
body.light .butlin-desc {{ color: rgba(42,58,42,0.55); }}
body.light .pulse-header .subtitle {{ color: rgba(42,58,42,0.4); }}
body.light .timeline-pane {{ background: rgba(255,255,255,0.55); border-color: rgba(0,0,0,0.08); }}
body.light .timeline-pane h3 {{ color: #2a3a2a; }}
body.light .timeline-legend {{ color: rgba(42,58,42,0.5); }}

/* Theme toggle button */
.theme-btn {{
  position: fixed;
  bottom: 20px;
  right: 72px;
  width: 44px;
  height: 44px;
  border-radius: 50%;
  background: var(--glass-bg);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid var(--glass-border);
  color: var(--solar-gold);
  font-size: 18px;
  cursor: pointer;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s ease;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}}
.theme-btn:hover {{ border-color: rgba(232,197,71,0.3); }}

/* Comparison delta arrows */
.delta {{ font-size: 0.72em; margin-left: 4px; font-weight: 400; }}
.delta-up {{ color: #7ec8a0; }}
.delta-down {{ color: #c76b5a; }}
.delta-flat {{ color: rgba(213,208,200,0.3); }}

/* Comparison banner */
.compare-banner {{
  max-width: 1200px;
  margin: 0 auto 16px;
  padding: 12px 20px;
  background: rgba(126,200,160,0.06);
  border: 1px solid rgba(126,200,160,0.15);
  border-radius: 12px;
  text-align: center;
  font-size: 0.82em;
  color: rgba(213,208,200,0.6);
  font-weight: 300;
  position: relative;
  z-index: 1;
}}

/* Timeline chart */
.timeline-pane {{
  max-width: 1200px;
  margin: 0 auto 20px;
  padding: 24px;
  background: rgba(30,35,30,0.45);
  backdrop-filter: blur(16px) saturate(140%);
  -webkit-backdrop-filter: blur(16px) saturate(140%);
  border: 1px solid rgba(126,200,160,0.12);
  border-radius: 16px;
  position: relative;
  z-index: 1;
}}
.timeline-pane h3 {{
  margin: 0 0 16px 0;
  color: #d5d0c8;
  font-weight: 400;
  font-size: 1em;
  letter-spacing: 0.04em;
}}
.timeline-legend {{
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  margin-top: 12px;
  font-size: 0.78em;
  color: rgba(213,208,200,0.6);
}}
.timeline-legend span {{
  display: flex;
  align-items: center;
  gap: 5px;
}}
.timeline-legend .swatch {{
  width: 12px;
  height: 3px;
  border-radius: 2px;
  display: inline-block;
}}

/* Interactive tooltip */
.tooltip {{
  position: absolute;
  background: rgba(26,46,34,0.95);
  backdrop-filter: blur(12px);
  border: 1px solid rgba(232,197,71,0.15);
  border-radius: 10px;
  padding: 10px 14px;
  font-size: 0.78em;
  color: rgba(213,208,200,0.75);
  max-width: 320px;
  pointer-events: none;
  z-index: 200;
  opacity: 0;
  transition: opacity 0.2s ease;
  line-height: 1.5;
  font-weight: 300;
  box-shadow: 0 8px 30px rgba(0,0,0,0.4);
}}
.tooltip.visible {{ opacity: 1; }}
body.light .tooltip {{
  background: rgba(255,255,255,0.95);
  color: rgba(42,58,42,0.75);
  border-color: rgba(180,160,60,0.2);
}}

/* Garden visualization */
.garden-container {{
  position: fixed;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 200px;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}}

/* Session report */
.session-report {{
  max-width: 1200px;
  margin: 0 auto 20px;
  padding: 18px 24px;
  background: rgba(126,200,160,0.04);
  border: 1px solid rgba(126,200,160,0.1);
  border-radius: 14px;
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  font-size: 0.84em;
  line-height: 1.7;
  color: rgba(213,208,200,0.6);
  font-weight: 300;
  position: relative;
  z-index: 1;
}}
.session-report h3 {{
  margin: 0 0 10px 0;
  font-size: 0.75em;
  font-weight: 500;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: rgba(213,208,200,0.35);
}}
body.light .session-report {{ background: rgba(126,200,160,0.06); color: rgba(42,58,42,0.6); }}

/* Anomaly markers */
.anomaly-marker {{
  font-size: 0.68em;
  padding: 2px 6px;
  border-radius: 4px;
  display: inline-block;
  font-weight: 400;
  letter-spacing: 0.02em;
}}

/* Expandable sparkline */
.sparkline-expand {{
  cursor: pointer;
  transition: opacity 0.2s;
}}
.sparkline-expand:hover {{ opacity: 1 !important; }}
.expanded-chart {{
  margin: 12px 0;
  padding: 16px;
  background: rgba(0,0,0,0.15);
  border: 1px solid rgba(255,255,255,0.05);
  border-radius: 10px;
  display: none;
}}
.expanded-chart.visible {{ display: block; }}

/* Harmony radar */
.harmony-radar {{ text-align: center; margin: 10px 0 6px; }}

/* Active rest badge (below harmony radar) */
.active-rest-badge {{
  background: #2a1f3d;
  border: 1px solid #7c5cbf;
  padding: 4px 12px;
  border-radius: 4px;
  color: #c4a0ff;
  font-size: 0.85em;
  text-align: center;
  display: inline-block;
  margin-top: 6px;
}}

/* Harmony heatmap timeline */
.harmony-heatmap {{ margin-top: 12px; padding: 8px 12px; background: rgba(0,0,0,0.15); border-radius: 8px; }}

/* Stillness breath animation */
@keyframes stillness-breath {{
    0%, 100% {{ opacity: 0.5; }}
    50% {{ opacity: 1.0; }}
}}
.stillness-breathing {{
    animation: stillness-breath 4s ease-in-out infinite;
}}

/* Radar rest pulse for Stillness mode */
@keyframes radar-rest-pulse {{
    0%, 100% {{ fill-opacity: 0.10; }}
    50% {{ fill-opacity: 0.18; }}
}}

/* Threshold lines */
.threshold-label {{
  font-size: 8px;
  fill: rgba(213,208,200,0.25);
  font-weight: 300;
}}

/* Print/PDF mode */
@media print {{
  body {{ background: #1a2e22 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  .sound-btn, .theme-btn, .garden-container {{ display: none !important; }}
  .pane {{ break-inside: avoid; }}
}}
</style>
</head>
<body>
<div class="pulse-header" role="banner">
  <h1>SYMTHAEA PULSE</h1>
  <div class="phi-breath" role="presentation" aria-hidden="true"></div>
  <div class="subtitle">{} · Profile: {} · Consciousness-First Cognitive Architecture</div>
</div>
"##, timestamp, profile,
     bg_hue = bg_hue, bg_sat = bg_sat, bg_hue_2 = bg_hue + 10.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pane 0: "What am I?" — Self-Description Banner
// ═══════════════════════════════════════════════════════════════════════════════

fn write_self_description(html: &mut String, vitals: &Vitals, bath: &NeuroBath, compass: &MoralCompass) {
    let desc = generate_self_description(vitals, bath, compass);
    let _ = write!(html, "<div class=\"self-desc\">&ldquo;{}&rdquo;</div>\n<div class=\"grid\">\n",
        escape_html(&desc));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Session Report — embedded mini-report
// ═══════════════════════════════════════════════════════════════════════════════

fn write_session_report_pane(html: &mut String, report: &str) {
    if report.is_empty() {
        return;
    }
    let _ = write!(html,
        "<div class=\"session-report\" role=\"region\" aria-label=\"Session report\">\n\
         <h3>Session Report</h3>\n\
         {}\n</div>\n",
        escape_html(report));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pane 1: Vitals — "The Phi Bloom"
// ═══════════════════════════════════════════════════════════════════════════════

fn write_vitals_pane(html: &mut String, v: &Vitals, sparkline: &[SparklinePoint], delta: Option<&PulseDelta>, anomalies: &[Anomaly]) {
    let c_color = consciousness_color(v.consciousness_level);
    let c_glow = consciousness_glow(v.consciousness_level);
    let c_interp = interpret_consciousness(v.consciousness_level);
    let phi_str = v.spectral_phi.map(|p| format!("{:.4}", p)).unwrap_or_else(|| "--".into());

    let _ = write!(html, r##"<div class="pane" role="region" aria-label="Consciousness Vitals">
<h2>Vitals — The Phi Bloom</h2>
"##);

    // Phi Bloom SVG flower
    write_phi_bloom(html, v);

    let delta_html = if let Some(d) = delta {
        format_delta(d.consciousness_level, 2)
    } else {
        String::new()
    };

    let _ = write!(html, r#"<div class="vital-big">
  <div class="number" style="color: {}; text-shadow: {};">{:.2}{}</div>
  <div class="label">C(t) — {}</div>
</div>
"#, c_color, c_glow, v.consciousness_level, delta_html, c_interp);

    // Sparkline row for consciousness trajectory (clickable to expand)
    if sparkline.len() >= 2 {
        let c_data: Vec<f64> = sparkline.iter().map(|s| s.consciousness).collect();
        let _ = write!(html, "<div style=\"text-align:center;margin:6px 0;\" class=\"sparkline-expand\" onclick=\"toggleExpanded('ct-expanded')\">");
        write_sparkline(html, &c_data, "#e8c547", 180, 24);
        let _ = write!(html, "<span style=\"font-size:0.65em;color:rgba(213,208,200,0.3);margin-left:6px;\">trajectory (click to expand)</span></div>\n");

        // Expanded full-width chart with threshold markers and anomaly annotations
        write_expanded_chart(html, "ct-expanded", sparkline, anomalies);

        // Harmony heatmap strip below expanded chart
        write_harmony_heatmap(html, sparkline);

        // Harmony entropy sparkline below heatmap
        write_entropy_sparkline(html, sparkline);
        write_broca_quality_sparkline(html, sparkline);
        write_tom_mismatch_sparkline(html, sparkline);
    }

    // Anomaly summary badges
    if !anomalies.is_empty() {
        let _ = write!(html, "<div style=\"margin:8px 0;text-align:center;\">\n");
        for a in anomalies {
            let _ = write!(html,
                "<span class=\"anomaly-marker\" style=\"background: {}15; color: {}; border: 1px solid {}30; margin: 2px 4px;\" \
                 title=\"Cycle {}: {}\">{}</span>\n",
                a.color, a.color, a.color, a.cycle, a.description, a.kind);
        }
        let _ = write!(html, "</div>\n");
    }

    // Build delta values array matching metrics order
    let metric_deltas: Vec<Option<f64>> = if let Some(d) = delta {
        vec![
            Some(d.spectral_phi),
            Some(d.pipeline_consciousness),
            Some(d.temporal_coherence),
            Some(d.phenomenal_binding),
            Some(d.living_mind_vitality),
            Some(d.effective_feasibility),
            Some(d.honest_confidence),
            Some(d.prediction_error as f64),
            Some(d.somatic_stress),
            Some(d.thermodynamic_load as f64),
        ]
    } else {
        vec![None; 10]
    };

    let metrics: &[(&str, String, &str)] = &[
        ("Spectral Phi", phi_str, health_color(v.spectral_phi.unwrap_or(0.0))),
        ("Pipeline Consciousness", format!("{:.3}", v.pipeline_consciousness), health_color(v.pipeline_consciousness)),
        ("Temporal Coherence", format!("{:.3}", v.temporal_coherence), health_color(v.temporal_coherence)),
        ("Phenomenal Binding", format!("{:.3}", v.phenomenal_binding), health_color(v.phenomenal_binding)),
        ("Living Mind Vitality", format!("{:.3}", v.living_mind_vitality), health_color(v.living_mind_vitality)),
        ("Substrate Feasibility", format!("{:.3}", v.substrate_effective_feasibility), health_color(v.substrate_effective_feasibility)),
        ("Evidence Confidence", format!("{:.2}", v.substrate_honest_confidence), health_color(v.substrate_honest_confidence)),
        ("Prediction Error", format!("{:.4}", v.prediction_error), stress_color(v.prediction_error as f64)),
        ("Somatic Stress", format!("{:.3}", v.somatic_stress), stress_color(v.somatic_stress)),
        ("Thermo Load", format!("{:.1}%", v.thermodynamic_load * 100.0), stress_color(v.thermodynamic_load as f64)),
    ];

    let pe_data: Vec<f64> = sparkline.iter().map(|s| s.prediction_error as f64).collect();
    let stress_data: Vec<f64> = sparkline.iter().map(|s| s.somatic_stress).collect();

    for (idx, (label, val, color)) in metrics.iter().enumerate() {
        let dhtml = metric_deltas[idx].map(|d| format_delta(d, 3)).unwrap_or_default();
        let _ = write!(html, r#"<div class="vital-row" role="listitem" aria-label="{}: {}">
  <span class="vital-label">{}</span>
  <span class="vital-value" style="color: {};">{}{}"#, label, val, label, color, val, dhtml);

        if idx == 7 && pe_data.len() >= 2 {
            html.push(' ');
            write_sparkline(html, &pe_data, "#c76b5a", 60, 14);
        } else if idx == 8 && stress_data.len() >= 2 {
            html.push(' ');
            write_sparkline(html, &stress_data, "#c4956a", 60, 14);
        }

        let _ = write!(html, "</span>\n</div>\n");
    }

    let cycle_hz = if v.cycle_duration_us > 0 { 1_000_000.0 / v.cycle_duration_us as f64 } else { 0.0 };
    let _ = write!(html, r#"<div class="status-bar">
  {} · {:.0} Hz · {} · Strategy: {} · {} cycles
</div>
</div>
"#, v.urgency, cycle_hz, v.consciousness_state, v.selected_strategy, v.total_cycles);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pane 2: Neuro-Bath — "The Living Vines"
// ═══════════════════════════════════════════════════════════════════════════════

fn write_neurobath_pane(html: &mut String, bath: &NeuroBath) {
    let state = interpret_neuro_state(bath);

    let _ = write!(html, r##"<div class="pane">
<h2>Neuro-Bath — The Living Vines</h2>
<div class="neuro-state">{}</div>
"##, state);

    let transmitters: &[(&str, f32)] = &[
        ("Dopamine", bath.dopamine),
        ("Noradrenaline", bath.noradrenaline),
        ("Serotonin", bath.serotonin),
        ("Acetylcholine", bath.acetylcholine),
        ("GABA", bath.gaba),
        ("Oxytocin", bath.oxytocin),
        ("Glutamate", bath.glutamate),
        ("Adenosine", bath.adenosine),
        ("Endocannabinoid", bath.endocannabinoid),
    ];

    for (idx, (name, level)) in transmitters.iter().enumerate() {
        let pct = (level / 2.0 * 100.0).clamp(0.0, 100.0);
        let gradient = transmitter_gradient(idx, *level);
        let _ = write!(html, r#"<div class="bar-row">
  <span class="bar-label">{}</span>
  <div class="bar-track"><div class="bar-fill" style="width: {:.0}%; background: {};"></div></div>
  <span class="bar-value">{:.2}</span>
</div>
"#, name, pct, gradient, level);
    }

    let _ = write!(html, r#"<div class="status-bar">
  {} · E/I: {:.2} · Allostatic: {:.0}% · Sleep: {:.2}{}
</div>
</div>
"#,
        bath.circadian_phase,
        bath.ei_ratio,
        bath.allostatic_load * 100.0,
        bath.sleep_pressure,
        if bath.personality.is_empty() { String::new() } else { format!(" · {}", bath.personality) },
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pane 3: Moral Compass — "The Harmony Garden"
// ═══════════════════════════════════════════════════════════════════════════════

fn write_moral_pane(html: &mut String, compass: &MoralCompass, sparkline: &[SparklinePoint]) {
    let (class_label, class_color) = moral_classification(compass.moral_score);
    let alignment_interp = interpret_moral_alignment(compass);

    let _ = write!(html, r##"<div class="pane">
<h2>Moral Compass — The Harmony Garden</h2>
<div style="text-align: center; margin-bottom: 16px;">
  <span class="moral-badge" style="background: {}15; color: {}; border: 1px solid {}30;">{}</span>
  <div style="margin-top: 10px; font-size: 0.82em; color: rgba(213,208,200,0.5); font-weight: 300;">
    H = {:.2} — {}
  </div>
</div>
"##, class_color, class_color, class_color, class_label,
     compass.harmonies_alignment, alignment_interp);

    let _ = write!(html, r#"<div class="vital-row">
  <span class="vital-label">Moral Score</span>
  <span class="vital-value">{:+.3}</span>
</div>
<div class="vital-row">
  <span class="vital-label">Value Evaluator</span>
  <span class="vital-value">{:.3}</span>
</div>
<div class="vital-row">
  <span class="vital-label">Moral Unity</span>
  <span class="vital-value" style="color: {};">{:.3}</span>
</div>
<div class="vital-row">
  <span class="vital-label">Soul Alignment</span>
  <span class="vital-value">{:+.2}</span>
</div>
<div class="vital-row">
  <span class="vital-label">Empathic Compassion</span>
  <span class="vital-value">{:.3}</span>
</div>
"#, compass.moral_score, compass.value_score,
     health_color(compass.moral_topo_unity), compass.moral_topo_unity,
     compass.soul_alignment, compass.empathic_compassion);

    if !compass.dominant_harmonic.is_empty() {
        let _ = write!(html, r#"<div class="vital-row">
  <span class="vital-label">Dominant Mode</span>
  <span class="vital-value" style="color: var(--solar-gold);">{}</span>
</div>
"#, compass.dominant_harmonic);
    }

    // 8-harmony radar chart — show attractor indicator if last sparkline point has it
    let attractor_detected = sparkline.last().map_or(false, |s| s.moral_attractor_detected);
    write_harmony_radar(html, &compass.harmony_coordinates, attractor_detected);

    // Active rest state indicator (below radar, above grid)
    if let Some(last) = sparkline.last() {
        if last.in_active_rest {
            let streak = last.stillness_dominance_streak;
            if streak > 0 {
                let _ = write!(html,
                    "<div style=\"text-align: center;\"><span class=\"active-rest-badge\">Active Rest <span style=\"opacity: 0.7;\">(streak: {})</span></span></div>\n",
                    streak);
            } else {
                html.push_str("<div style=\"text-align: center;\"><span class=\"active-rest-badge\">Active Rest</span></div>\n");
            }
        }
    }

    // 8-harmony grid (compact detail below radar)
    let _ = write!(html, r#"<div class="harmony-grid">"#);
    for (i, name) in HARMONY_NAMES.iter().enumerate() {
        let val = compass.harmony_coordinates[i];
        let color = if val >= 0.5 { "#e8c547" } else if val >= 0.2 { "rgba(213,208,200,0.6)" } else { "rgba(213,208,200,0.25)" };
        let bg_alpha = 0.02 + val * 0.06;
        let _ = write!(html, r#"<div class="harmony-cell" style="background: rgba(232,197,71,{:.2});">
  <div class="name">{}</div>
  <div class="val" style="color: {};">{:.2}</div>
</div>
"#, bg_alpha, name, color, val);
    }
    let _ = write!(html, "</div>\n");

    if !compass.guiding_question.is_empty() {
        let _ = write!(html, r#"<div style="margin-top: 14px; padding: 12px 14px; background: rgba(232,197,71,0.04); border: 1px solid rgba(232,197,71,0.08); border-radius: 10px; font-style: italic; color: rgba(213,208,200,0.5); font-size: 0.82em; font-weight: 300;">
  &ldquo;{}&rdquo;
</div>
"#, escape_html(&compass.guiding_question));
    }

    html.push_str("</div>\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pane 4: Cognitive Radar — "The Canopy Map"
// ═══════════════════════════════════════════════════════════════════════════════

fn write_cognitive_pane(html: &mut String, cognitive: Option<&CognitiveProfile>) {
    let _ = write!(html, r##"<div class="pane">
<h2>Cognitive Radar — The Canopy Map</h2>
"##);

    let profile = match cognitive {
        Some(p) if !p.domains.is_empty() => p,
        _ => {
            let _ = write!(html, r#"<div style="text-align: center; color: rgba(213,208,200,0.3); padding: 40px 0; font-weight: 300;">
  Run with psych-bench to see cognitive profile<br>
  <span style="font-size: 0.8em; opacity: 0.6;">(omit --skip-bench)</span>
</div></div>
"#);
            return;
        }
    };

    let overall_color = if profile.overall >= 0.75 { "#e8c547" }
        else if profile.overall >= 0.5 { "#7ec8a0" }
        else { "#c4956a" };
    let _ = write!(html, r#"<div style="text-align: center; margin-bottom: 16px;">
  <span style="font-size: 2.2em; font-weight: 200; color: {};">{:.0}%</span>
  <div style="font-size: 0.78em; color: rgba(213,208,200,0.4); margin-top: 4px; font-weight: 300;">Overall · Strongest: {} · Weakest: {}</div>
</div>
"#, overall_color, profile.overall * 100.0, profile.strongest, profile.weakest);

    write_radar_svg(html, profile);

    for d in &profile.domains {
        let pct = d.score * 100.0;
        let color = if d.score >= 0.75 { "#e8c547" }
            else if d.score >= 0.5 { "#7ec8a0" }
            else { "#c4956a" };
        let _ = write!(html, r#"<div class="bar-row">
  <span class="bar-label">{}</span>
  <div class="bar-track"><div class="bar-fill" style="width: {:.0}%; background: {};"></div></div>
  <span class="bar-value">{:.0}%</span>
</div>
"#, d.domain, pct, color, pct);
    }

    html.push_str("</div>\n");
}

fn write_radar_svg(html: &mut String, profile: &CognitiveProfile) {
    let domains = &profile.domains;
    let n = domains.len();
    if n < 3 { return; }

    let cx = 150.0_f64;
    let cy = 150.0_f64;
    let radius = 110.0_f64;

    let _ = write!(html, r#"<div class="radar-container">
<svg width="300" height="300" viewBox="0 0 300 300">
"#);

    // Grid rings
    for level in &[0.25, 0.50, 0.75, 1.0] {
        let r = radius * level;
        let _ = write!(html,
            "<circle cx=\"{}\" cy=\"{}\" r=\"{:.1}\" fill=\"none\" stroke=\"rgba(255,255,255,0.06)\" stroke-width=\"1\"/>\n",
            cx, cy, r);
    }

    // Axes + labels
    for (i, d) in domains.iter().enumerate() {
        let angle = std::f64::consts::TAU * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
        let x = cx + radius * angle.cos();
        let y = cy + radius * angle.sin();
        let _ = write!(html,
            "<line x1=\"{}\" y1=\"{}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"rgba(255,255,255,0.05)\" stroke-width=\"1\"/>\n",
            cx, cy, x, y);

        let lr = radius + 18.0;
        let lx = cx + lr * angle.cos();
        let ly = cy + lr * angle.sin();
        let anchor = if angle.cos() > 0.1 { "start" } else if angle.cos() < -0.1 { "end" } else { "middle" };
        let _ = write!(html,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"{}\" font-size=\"8.5\" fill=\"rgba(213,208,200,0.4)\" font-weight=\"400\">{}</text>\n",
            lx, ly + 3.0, anchor, d.domain);
    }

    // Data polygon
    let mut points = String::new();
    for (i, d) in domains.iter().enumerate() {
        let angle = std::f64::consts::TAU * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
        let r = radius * d.score.clamp(0.0, 1.0);
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();
        if !points.is_empty() { points.push(' '); }
        let _ = write!(points, "{:.1},{:.1}", x, y);
    }
    let _ = write!(html,
        "<polygon points=\"{}\" fill=\"rgba(126,200,160,0.12)\" stroke=\"rgba(126,200,160,0.6)\" stroke-width=\"1.5\"/>\n",
        points);

    // Data dots
    for (i, d) in domains.iter().enumerate() {
        let angle = std::f64::consts::TAU * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
        let r = radius * d.score.clamp(0.0, 1.0);
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();
        let dot_r = if d.score >= 0.75 { 5.0 } else { 3.5 };
        let dot_color = if d.score >= 0.75 { "#e8c547" } else { "#7ec8a0" };
        let _ = write!(html,
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"{}\" fill=\"{}\" stroke=\"rgba(0,0,0,0.3)\" stroke-width=\"1.5\"/>\n",
            x, y, dot_r, dot_color);
    }

    let _ = write!(html, "</svg>\n</div>\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pane 5: Butlin Indicators — "Theory Seeds"
// ═══════════════════════════════════════════════════════════════════════════════

fn write_butlin_pane(html: &mut String, butlin: Option<&ButlinIndicatorReport>) {
    let _ = write!(html, r##"<div class="pane">
<h2>Butlin Indicators — Theory Seeds</h2>
"##);

    let report = match butlin {
        Some(r) => r,
        None => {
            let _ = write!(html, r#"<div style="text-align: center; color: rgba(213,208,200,0.3); padding: 40px 0; font-weight: 300;">
  Run with psych-bench to see indicators<br>
  <span style="font-size: 0.8em; opacity: 0.6;">(omit --skip-bench)</span>
</div></div>
"#);
            return;
        }
    };

    // Summary bar
    let total = report.indicators.len();
    let _ = write!(html,
        r#"<div style="text-align:center;margin-bottom:14px;font-size:0.82em;color:rgba(213,208,200,0.5);font-weight:300;">
  <span style="color:#7ec8a0;">{} present</span> · <span style="color:#e8c547;">{} partial</span> · <span style="color:#6b7d6b;">{} absent</span> of {} indicators
</div>
"#, report.present_count, report.partial_count, report.absent_count, total);

    let _ = write!(html, "<div class=\"butlin-grid\">\n");

    for ind in &report.indicators {
        let (dot_color, dot_glow) = match ind.status {
            IndicatorStatus::Present => ("#7ec8a0", "0 0 6px rgba(126,200,160,0.5)"),
            IndicatorStatus::Partial => ("#e8c547", "0 0 6px rgba(232,197,71,0.4)"),
            IndicatorStatus::Absent  => ("#6b7d6b", "none"),
        };

        let score_str = ind.score.map(|s| format!(" ({:.0}%)", s * 100.0)).unwrap_or_default();

        let _ = write!(html,
            r#"<div class="butlin-item" title="{}">
  <div class="butlin-dot" style="background:{};box-shadow:{};"></div>
  <span class="butlin-id">{}</span>
  <span class="butlin-desc">{}{}</span>
</div>
"#, escape_html(&ind.evidence), dot_color, dot_glow, ind.id, ind.description, score_str);
    }

    let _ = write!(html, "</div>\n");

    // Theory legend
    let _ = write!(html, r#"<div class="status-bar">
  RPT: Recurrent Processing · GWT: Global Workspace · HOT: Higher-Order · PP: Predictive Processing · AST: Attention Schema · IIT: Integrated Information
</div>
</div>
"#);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pane 6: Substrate Visualization
// ═══════════════════════════════════════════════════════════════════════════════

fn write_substrate_pane(html: &mut String, substrate: &SubstrateInfo) {
    let icon = match substrate.substrate_type.as_str() {
        "BiologicalNeurons" | "Biological" => "&#x1F9E0;",
        "SiliconDigital" | "Silicon" => "&#x1F4BB;",
        "QuantumComputer" | "Quantum" => "&#x269B;&#xFE0F;",
        "PhotonicProcessor" => "&#x1F4A1;",
        "NeuromorphicChip" => "&#x1F50C;",
        "BiochemicalComputer" => "&#x1F9EC;",
        "HybridSystem" | "Hybrid" => "&#x1F504;",
        "ExoticSubstrate" => "&#x2728;",
        _ => "&#x2699;&#xFE0F;",
    };

    let confidence_color = health_color(substrate.honest_confidence);

    let _ = write!(html, r#"<div class="pane">
<h2>Substrate</h2>
<div class="substrate-icon" style="opacity: {:.2};">{}</div>
<div style="text-align:center;font-size:1.1em;font-weight:400;color:rgba(213,208,200,0.7);margin-bottom:12px;">{}</div>
<div class="confidence-overlay" style="background: rgba({}0.06); border: 1px solid rgba({}0.12);">
  Evidence confidence: <strong style="color:{};">{:.0}%</strong> — {}
</div>
"#, 0.4 + substrate.honest_confidence * 0.6, icon,
     substrate.substrate_type.replace("Digital", " Digital").replace("Neurons", " Neurons"),
     confidence_color.replace('#', ""), confidence_color.replace('#', ""),
     confidence_color, substrate.honest_confidence * 100.0,
     if substrate.honest_confidence >= 0.8 { "validated" }
     else if substrate.honest_confidence >= 0.4 { "experimental" }
     else if substrate.honest_confidence >= 0.1 { "theoretical" }
     else { "no evidence" },
    );

    let metrics: &[(&str, String)] = &[
        ("Raw Feasibility", format!("{:.3}", substrate.raw_feasibility)),
        ("Honest Confidence", format!("{:.3}", substrate.honest_confidence)),
        ("Effective Feasibility", format!("{:.3}", substrate.effective_feasibility)),
        ("Tau Factor", format!("{:.3}", substrate.tau_factor)),
        ("Scale Pressure", format!("{:.3}", substrate.scale_pressure)),
    ];

    for (label, val) in metrics {
        let _ = write!(html, r#"<div class="vital-row">
  <span class="vital-label">{}</span>
  <span class="vital-value">{}</span>
</div>
"#, label, val);
    }

    // Feasibility gap visualization
    let gap = (substrate.raw_feasibility - substrate.effective_feasibility).abs();
    if gap > 0.01 {
        let _ = write!(html, r#"<div style="margin-top:10px;text-align:center;font-size:0.78em;color:rgba(213,208,200,0.35);font-weight:300;">
  Feasibility gap: {:.1}% — the difference between what the architecture could support and what evidence confirms
</div>
"#, gap * 100.0);
    }

    html.push_str("</div>\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Pane 7: Narrative / Inner Voice
// ═══════════════════════════════════════════════════════════════════════════════

fn write_narrative_pane(html: &mut String, narrative: &Narrative) {
    let _ = write!(html, r##"<div class="pane">
<h2>Inner Voice</h2>
"##);

    match &narrative.reasoning {
        Some(text) if !text.is_empty() => {
            let _ = write!(html,
                "<div class=\"narrative-text\">&ldquo;{}&rdquo;</div>\n",
                escape_html(text));
        }
        _ => {
            let _ = write!(html,
                "<div class=\"narrative-text\" style=\"color:rgba(213,208,200,0.3);\">No reasoning narrative this cycle — the mind is processing silently.</div>\n");
        }
    }

    if !narrative.guiding_question.is_empty() {
        let _ = write!(html,
            "<div style=\"margin-top:10px;padding:10px 14px;background:rgba(232,197,71,0.04);border:1px solid rgba(232,197,71,0.08);border-radius:8px;font-size:0.82em;color:rgba(213,208,200,0.5);font-weight:300;font-style:italic;\">Guiding question: &ldquo;{}&rdquo;</div>\n",
            escape_html(&narrative.guiding_question));
    }

    let _ = write!(html, r#"<div class="status-bar">
  State: {} · Pattern: {} · Strategy: {}
</div>
</div>
"#, narrative.consciousness_state, narrative.error_pattern, narrative.selected_strategy);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-pane Mycelial Connections (decorative SVG overlay)
// ═══════════════════════════════════════════════════════════════════════════════

fn write_mycelial_connections(html: &mut String, vitals: &Vitals, bath: &NeuroBath, compass: &MoralCompass) {
    // Decorative mycelial threads that pulse based on cross-domain correlations
    let phi_moral = vitals.consciousness_level * compass.harmonies_alignment as f64;
    let neuro_vitals = bath.dopamine as f64 * vitals.living_mind_vitality;
    let binding_harmony = vitals.phenomenal_binding * compass.harmony_coordinates.iter().sum::<f64>() / N_HARMONIES as f64;

    let thread_alpha_1 = 0.02 + phi_moral * 0.06;
    let thread_alpha_2 = 0.02 + neuro_vitals * 0.04;
    let thread_alpha_3 = 0.02 + binding_harmony * 0.05;

    let _ = write!(html, r#"<div style="position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:0;">
<svg width="100%" height="100%" style="position:absolute;">
  <defs>
    <filter id="myc-blur"><feGaussianBlur stdDeviation="2"/></filter>
  </defs>
  <path d="M 10%,30% Q 30%,20% 50%,35% T 90%,25%" fill="none" stroke="rgba(126,200,160,{:.3})" stroke-width="1" filter="url(#myc-blur)">
    <animate attributeName="d" values="M 10%,30% Q 30%,20% 50%,35% T 90%,25%;M 10%,32% Q 30%,22% 50%,33% T 90%,27%;M 10%,30% Q 30%,20% 50%,35% T 90%,25%" dur="8s" repeatCount="indefinite"/>
  </path>
  <path d="M 5%,60% Q 25%,50% 45%,65% T 95%,55%" fill="none" stroke="rgba(232,197,71,{:.3})" stroke-width="0.8" filter="url(#myc-blur)">
    <animate attributeName="d" values="M 5%,60% Q 25%,50% 45%,65% T 95%,55%;M 5%,58% Q 25%,52% 45%,63% T 95%,57%;M 5%,60% Q 25%,50% 45%,65% T 95%,55%" dur="10s" repeatCount="indefinite"/>
  </path>
  <path d="M 15%,85% Q 35%,75% 55%,90% T 85%,80%" fill="none" stroke="rgba(196,149,106,{:.3})" stroke-width="0.6" filter="url(#myc-blur)">
    <animate attributeName="d" values="M 15%,85% Q 35%,75% 55%,90% T 85%,80%;M 15%,83% Q 35%,77% 55%,88% T 85%,82%;M 15%,85% Q 35%,75% 55%,90% T 85%,80%" dur="12s" repeatCount="indefinite"/>
  </path>
</svg>
</div>
"#, thread_alpha_1, thread_alpha_2, thread_alpha_3);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Temporal Sonification (Web Audio API)
// ═══════════════════════════════════════════════════════════════════════════════

fn write_sonification_script(html: &mut String, vitals: &Vitals, bath: &NeuroBath) {
    // Map: Phi → pitch (220-880 Hz), consciousness → volume, serotonin → filter cutoff
    let freq = 220.0 + vitals.consciousness_level * 660.0;
    let gain = 0.02 + vitals.consciousness_level * 0.08; // very quiet ambient
    let filter_freq = 400.0 + bath.serotonin as f64 * 2000.0;
    let lfo_rate = 0.5 + bath.dopamine as f64 * 2.0; // dopamine drives rhythmic pulsing

    let _ = write!(html, r##"<button class="sound-btn" id="soundToggle" onclick="toggleSound()" title="Toggle ambient sonification">&#x266B;</button>
<script>
let audioCtx = null, osc = null, gain = null, filter = null, lfo = null, lfoGain = null, isPlaying = false;
function toggleSound() {{
  if (!audioCtx) {{
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    osc = audioCtx.createOscillator();
    gain = audioCtx.createGain();
    filter = audioCtx.createBiquadFilter();
    lfo = audioCtx.createOscillator();
    lfoGain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.value = {freq:.1};
    filter.type = 'lowpass';
    filter.frequency.value = {filter_freq:.0};
    filter.Q.value = 2.0;
    gain.gain.value = 0;
    lfo.type = 'sine';
    lfo.frequency.value = {lfo_rate:.2};
    lfoGain.gain.value = {gain:.4} * 0.3;
    osc.connect(filter);
    filter.connect(gain);
    gain.connect(audioCtx.destination);
    lfo.connect(lfoGain);
    lfoGain.connect(gain.gain);
    osc.start();
    lfo.start();
  }}
  if (isPlaying) {{
    gain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.5);
    isPlaying = false;
    document.getElementById('soundToggle').style.opacity = '0.4';
  }} else {{
    gain.gain.linearRampToValueAtTime({gain:.4}, audioCtx.currentTime + 0.5);
    isPlaying = true;
    document.getElementById('soundToggle').style.opacity = '1';
  }}
}}
</script>
"##, freq = freq, filter_freq = filter_freq, lfo_rate = lfo_rate, gain = gain);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Footer
// ═══════════════════════════════════════════════════════════════════════════════

fn write_footer(html: &mut String, timestamp: &str) {
    let _ = write!(html, r#"</div>
<div class="footer">
  Generated by <strong>symthaea-pulse</strong> · Luminous Dynamics · {} · <a href="javascript:window.print()" style="color:rgba(213,208,200,0.3);text-decoration:none;">Export PDF</a>
</div>
</body>
</html>"#, timestamp);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Comparison banner + delta formatting
// ═══════════════════════════════════════════════════════════════════════════════

fn format_delta(val: f64, precision: usize) -> String {
    if val.abs() < 0.001 {
        format!("<span class=\"delta delta-flat\">~</span>")
    } else if val > 0.0 {
        format!("<span class=\"delta delta-up\">+{:.prec$}</span>", val, prec = precision)
    } else {
        format!("<span class=\"delta delta-down\">{:.prec$}</span>", val, prec = precision)
    }
}

fn format_delta_f32(val: f32, precision: usize) -> String {
    format_delta(val as f64, precision)
}

fn write_comparison_banner(html: &mut String, delta: &PulseDelta) {
    let _ = write!(html,
        "<div class=\"compare-banner\">Comparing against snapshot from <strong>{}</strong> · \
         C(t) {} · Phi {} · Pipeline {} · Stress {} · Moral {}</div>\n",
        delta.prev_timestamp,
        format_delta(delta.consciousness_level, 3),
        format_delta(delta.spectral_phi, 4),
        format_delta(delta.pipeline_consciousness, 3),
        format_delta(delta.somatic_stress, 3),
        format_delta_f32(delta.moral_score, 3),
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Expanded Sparkline Chart with Thresholds + Anomaly Annotations
// ═══════════════════════════════════════════════════════════════════════════════

fn write_expanded_chart(html: &mut String, id: &str, sparkline: &[SparklinePoint], anomalies: &[Anomaly]) {
    let n = sparkline.len();
    if n < 2 { return; }

    let w = 540.0_f64;
    let h = 160.0_f64;
    let pad_l = 36.0_f64;
    let pad_r = 8.0_f64;
    let pad_t = 12.0_f64;
    let pad_b = 20.0_f64;
    let plot_w = w - pad_l - pad_r;
    let plot_h = h - pad_t - pad_b;
    let x_step = plot_w / (n - 1) as f64;

    let _ = write!(html, "<div class=\"expanded-chart\" id=\"{}\">\n", id);
    let _ = write!(html,
        "<svg width=\"100%\" height=\"{}\" viewBox=\"0 0 {} {}\" role=\"img\" aria-label=\"Expanded consciousness trajectory with thresholds\">\n",
        h, w, h);

    // Threshold reference lines (item 6)
    let thresholds: &[(&str, f64, &str)] = &[
        ("Emergence", 0.3, "#7ec8a0"),
        ("Aware", 0.5, "#e8c547"),
        ("Integrated", 0.7, "#e8c547"),
    ];
    for &(label, level, color) in thresholds {
        let y = pad_t + plot_h * (1.0 - level);
        let _ = write!(html,
            "<line x1=\"{}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\" stroke-dasharray=\"4,4\" opacity=\"0.4\"/>\n\
             <text x=\"{:.1}\" y=\"{:.1}\" class=\"threshold-label\" text-anchor=\"end\">{} ({:.0}%)</text>\n",
            pad_l, y, pad_l + plot_w, y, color,
            pad_l - 3.0, y + 3.0, label, level * 100.0);
    }

    // Series: consciousness, PE, phi
    let series: &[(&str, &str, Box<dyn Fn(&SparklinePoint) -> f64>)] = &[
        ("C(t)", "#e8c547", Box::new(|s: &SparklinePoint| s.consciousness)),
        ("PE", "#c76b5a", Box::new(|s: &SparklinePoint| s.prediction_error as f64)),
        ("Stress", "#c4956a", Box::new(|s: &SparklinePoint| s.somatic_stress)),
    ];

    for &(name, color, ref extract) in series {
        let mut points = String::new();
        for (i, s) in sparkline.iter().enumerate() {
            let x = pad_l + i as f64 * x_step;
            let v = extract(s).clamp(0.0, 1.0);
            let y = pad_t + plot_h * (1.0 - v);
            if !points.is_empty() { points.push(' '); }
            let _ = write!(points, "{:.1},{:.1}", x, y);
        }
        let _ = write!(html,
            "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"1.5\" stroke-linejoin=\"round\" opacity=\"0.7\"/>\n",
            points, color);

        // Label at end
        if let Some(last) = sparkline.last() {
            let x = pad_l + (n - 1) as f64 * x_step;
            let v = extract(last).clamp(0.0, 1.0);
            let y = pad_t + plot_h * (1.0 - v);
            let _ = write!(html,
                "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"{}\" font-size=\"7\" font-weight=\"400\">{}</text>\n",
                x + 3.0, y + 3.0, color, name);
        }
    }

    // Anomaly annotations as vertical markers
    for a in anomalies {
        if a.cycle < n {
            let x = pad_l + a.cycle as f64 * x_step;
            let _ = write!(html,
                "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"1\" stroke-dasharray=\"2,2\" opacity=\"0.6\"/>\n\
                 <text x=\"{:.1}\" y=\"{:.1}\" fill=\"{}\" font-size=\"7\" text-anchor=\"middle\" font-weight=\"400\">{}</text>\n",
                x, pad_t, x, pad_t + plot_h, a.color,
                x, pad_t - 2.0, a.color, a.kind);
        }
    }

    let _ = write!(html, "</svg>\n</div>\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Harmony Heatmap Timeline — 8-row strip showing harmony coordinates over time
// ═══════════════════════════════════════════════════════════════════════════════

fn write_harmony_heatmap(html: &mut String, sparkline: &[SparklinePoint]) {
    if sparkline.is_empty() { return; }
    let n = sparkline.len();
    let width = 600.0_f64;
    let cell_w = width / n as f64;
    let row_h = 16.0_f64;
    let total_h = row_h * N_HARMONIES as f64;

    let _ = write!(html, r#"<div class="harmony-heatmap">
<div style="font-size: 0.72em; color: rgba(213,208,200,0.4); margin-bottom: 4px;">Harmony Timeline</div>
<svg viewBox="0 0 {:.0} {:.0}" width="100%" preserveAspectRatio="none">"#, width, total_h + 12.0);

    for (row, name) in HARMONY_NAMES.iter().enumerate() {
        let y = row as f64 * row_h;
        for (col, point) in sparkline.iter().enumerate() {
            let x = col as f64 * cell_w;
            let val = point.harmony_coords[row];
            // Map [-1, 1] to color (blue=negative, black=zero, gold=positive)
            let (r, g, b) = if val > 0.0 {
                let t = val.min(1.0);
                ((232.0 * t) as u8, (197.0 * t) as u8, (71.0 * t) as u8) // gold
            } else {
                let t = (-val).min(1.0);
                ((80.0 * t) as u8, (120.0 * t) as u8, (200.0 * t) as u8) // blue
            };
            let _ = write!(html,
                r#"<rect x="{:.1}" y="{:.0}" width="{:.1}" height="{:.0}" fill="rgb({},{},{})" opacity="0.85"/>"#,
                x, y, cell_w.max(1.0), row_h - 1.0, r, g, b);
        }
        // Row label (right side)
        let _ = write!(html,
            r#"<text x="{:.0}" y="{:.0}" fill="rgba(213,208,200,0.35)" font-size="8" text-anchor="end">{}</text>"#,
            width - 2.0, y + row_h - 4.0, name);
    }

    let _ = write!(html, "</svg>\n</div>\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Harmony Entropy Sparkline — "Moral Breadth" over time
// ═══════════════════════════════════════════════════════════════════════════════

fn write_entropy_sparkline(html: &mut String, sparkline: &[SparklinePoint]) {
    if sparkline.len() < 2 { return; }

    let data: Vec<f64> = sparkline.iter().map(|s| s.harmony_entropy).collect();
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max - min).abs() < 1e-12 { 1.0 } else { max - min };
    let last = *data.last().unwrap_or(&0.0);

    let width = 600.0_f64;
    let height = 40.0_f64;

    let _ = write!(html, r#"<div class="harmony-heatmap" style="margin-top: 6px;">
<div style="font-size: 0.72em; color: rgba(213,208,200,0.4); margin-bottom: 4px;">Moral Breadth (Harmony Entropy) — current: {:.3}</div>
<svg viewBox="0 0 {:.0} {:.0}" width="100%" preserveAspectRatio="none">
"#, last, width, height);

    // Fill area under the line
    let _ = write!(html, "<polyline points=\"");
    // Start from bottom-left
    let _ = write!(html, "0,{:.0} ", height);
    for (i, &v) in data.iter().enumerate() {
        let x = i as f64 / (data.len() - 1) as f64 * width;
        let y = height - ((v - min) / range * (height - 6.0) + 3.0);
        let _ = write!(html, "{:.1},{:.1} ", x, y);
    }
    // Close to bottom-right
    let _ = write!(html, "{:.0},{:.0}", width, height);
    let _ = write!(html,
        "\" fill=\"rgba(126,200,160,0.08)\" stroke=\"none\"/>\n");

    // The line itself
    let _ = write!(html, "<polyline points=\"");
    for (i, &v) in data.iter().enumerate() {
        let x = i as f64 / (data.len() - 1) as f64 * width;
        let y = height - ((v - min) / range * (height - 6.0) + 3.0);
        if i > 0 { html.push(' '); }
        let _ = write!(html, "{:.1},{:.1}", x, y);
    }
    let _ = write!(html,
        "\" fill=\"none\" stroke=\"#7ec8a0\" stroke-width=\"1.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" opacity=\"0.7\"/>\n");

    // End dot
    let last_x = width;
    let last_y = height - ((last - min) / range * (height - 6.0) + 3.0);
    let _ = write!(html,
        "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"2.5\" fill=\"#7ec8a0\" opacity=\"0.9\"/>\n",
        last_x, last_y);

    let _ = write!(html, "</svg>\n</div>\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Broca Generation Quality Sparkline
// ═══════════════════════════════════════════════════════════════════════════════

fn write_broca_quality_sparkline(html: &mut String, sparkline: &[SparklinePoint]) {
    if sparkline.len() < 2 { return; }

    let data: Vec<f64> = sparkline.iter().map(|s| s.broca_quality as f64).collect();
    let min = 0.0_f64;
    let max = 1.0_f64;
    let range = 1.0;
    let last = *data.last().unwrap_or(&0.0);

    let width = 600.0_f64;
    let height = 40.0_f64;

    let _ = write!(html, r#"<div class="harmony-heatmap" style="margin-top: 6px;">
<div style="font-size: 0.72em; color: rgba(213,208,200,0.4); margin-bottom: 4px;">Broca Generation Quality — current: {:.3}</div>
<svg viewBox="0 0 {:.0} {:.0}" width="100%" preserveAspectRatio="none">
"#, last, width, height);

    let _ = write!(html, "<polyline points=\"0,{:.0} ", height);
    for (i, &v) in data.iter().enumerate() {
        let x = i as f64 / (data.len() - 1) as f64 * width;
        let y = height - ((v - min) / range * (height - 6.0) + 3.0);
        let _ = write!(html, "{:.1},{:.1} ", x, y);
    }
    let _ = write!(html, "{:.0},{:.0}", width, height);
    let _ = write!(html, "\" fill=\"rgba(160,140,200,0.08)\" stroke=\"none\"/>\n");

    let _ = write!(html, "<polyline points=\"");
    for (i, &v) in data.iter().enumerate() {
        let x = i as f64 / (data.len() - 1) as f64 * width;
        let y = height - ((v - min) / range * (height - 6.0) + 3.0);
        if i > 0 { html.push(' '); }
        let _ = write!(html, "{:.1},{:.1}", x, y);
    }
    let _ = write!(html,
        "\" fill=\"none\" stroke=\"#a08cc8\" stroke-width=\"1.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" opacity=\"0.7\"/>\n");

    let last_x = width;
    let last_y = height - ((last - min) / range * (height - 6.0) + 3.0);
    let _ = write!(html,
        "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"2.5\" fill=\"#a08cc8\" opacity=\"0.9\"/>\n",
        last_x, last_y);

    let _ = write!(html, "</svg>\n</div>\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// ToM Prediction Mismatch Sparkline
// ═══════════════════════════════════════════════════════════════════════════════

fn write_tom_mismatch_sparkline(html: &mut String, sparkline: &[SparklinePoint]) {
    if sparkline.len() < 2 { return; }

    let data: Vec<f64> = sparkline.iter().map(|s| s.tom_mismatch as f64).collect();
    let min = 0.0_f64;
    let max = 1.0_f64;
    let range = 1.0;
    let last = *data.last().unwrap_or(&0.0);

    let width = 600.0_f64;
    let height = 40.0_f64;

    // Color: orange-ish for mismatch (high = bad)
    let _ = write!(html, r#"<div class="harmony-heatmap" style="margin-top: 6px;">
<div style="font-size: 0.72em; color: rgba(213,208,200,0.4); margin-bottom: 4px;">ToM Prediction Mismatch — current: {:.3}{}</div>
<svg viewBox="0 0 {:.0} {:.0}" width="100%" preserveAspectRatio="none">
"#, last, if last > 0.4 { " (exploring)" } else { "" }, width, height);

    // Threshold line at 0.4
    let thresh_y = height - (0.4 / range * (height - 6.0) + 3.0);
    let _ = write!(html,
        "<line x1=\"0\" y1=\"{:.1}\" x2=\"{:.0}\" y2=\"{:.1}\" stroke=\"#d4845a\" stroke-width=\"0.5\" stroke-dasharray=\"4,4\" opacity=\"0.4\"/>\n",
        thresh_y, width, thresh_y);

    let _ = write!(html, "<polyline points=\"0,{:.0} ", height);
    for (i, &v) in data.iter().enumerate() {
        let x = i as f64 / (data.len() - 1) as f64 * width;
        let y = height - ((v - min) / range * (height - 6.0) + 3.0);
        let _ = write!(html, "{:.1},{:.1} ", x, y);
    }
    let _ = write!(html, "{:.0},{:.0}", width, height);
    let _ = write!(html, "\" fill=\"rgba(212,132,90,0.08)\" stroke=\"none\"/>\n");

    let _ = write!(html, "<polyline points=\"");
    for (i, &v) in data.iter().enumerate() {
        let x = i as f64 / (data.len() - 1) as f64 * width;
        let y = height - ((v - min) / range * (height - 6.0) + 3.0);
        if i > 0 { html.push(' '); }
        let _ = write!(html, "{:.1},{:.1}", x, y);
    }
    let _ = write!(html,
        "\" fill=\"none\" stroke=\"#d4845a\" stroke-width=\"1.5\" stroke-linecap=\"round\" stroke-linejoin=\"round\" opacity=\"0.7\"/>\n");

    let last_x = width;
    let last_y = height - ((last - min) / range * (height - 6.0) + 3.0);
    let _ = write!(html,
        "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"2.5\" fill=\"#d4845a\" opacity=\"0.9\"/>\n",
        last_x, last_y);

    let _ = write!(html, "</svg>\n</div>\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Harmony Radar Chart — 8-axis spider chart for harmony coordinates
// ═══════════════════════════════════════════════════════════════════════════════

fn write_harmony_radar(html: &mut String, coords: &[f64; N_HARMONIES], attractor_detected: bool) {
    let cx = 100.0_f64;
    let cy = 100.0_f64;
    let radius = 75.0_f64;
    let n = N_HARMONIES;

    let _ = write!(html, "<div class=\"harmony-radar\">\n<svg width=\"200\" height=\"200\" viewBox=\"0 0 200 200\" role=\"img\" aria-label=\"Harmony radar chart\">\n");

    // Grid rings
    for &level in &[0.25, 0.50, 0.75, 1.0] {
        let r = radius * level;
        let mut ring = String::new();
        for i in 0..n {
            let angle = std::f64::consts::TAU * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
            let x = cx + r * angle.cos();
            let y = cy + r * angle.sin();
            if !ring.is_empty() { ring.push(' '); }
            let _ = write!(ring, "{:.1},{:.1}", x, y);
        }
        let _ = write!(html,
            "<polygon points=\"{}\" fill=\"none\" stroke=\"rgba(232,197,71,0.06)\" stroke-width=\"0.5\"/>\n", ring);
    }

    // Axis lines + labels
    for (i, name) in HARMONY_NAMES.iter().enumerate() {
        let angle = std::f64::consts::TAU * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
        let x = cx + radius * angle.cos();
        let y = cy + radius * angle.sin();
        let _ = write!(html,
            "<line x1=\"{}\" y1=\"{}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"rgba(232,197,71,0.06)\" stroke-width=\"0.5\"/>\n",
            cx, cy, x, y);

        let lr = radius + 14.0;
        let lx = cx + lr * angle.cos();
        let ly = cy + lr * angle.sin();
        let anchor = if angle.cos() > 0.1 { "start" } else if angle.cos() < -0.1 { "end" } else { "middle" };
        let short = if name.len() > 5 { &name[..4] } else { name };
        // Add breathing animation class to Stillness label when > 0.3
        let class_attr = if i == 7 && coords[7] > 0.3 { " class=\"stillness-breathing\"" } else { "" };
        let _ = write!(html,
            "<text x=\"{:.1}\" y=\"{:.1}\" text-anchor=\"{}\" font-size=\"7\" fill=\"rgba(213,208,200,0.35)\" font-weight=\"300\"{}>{}</text>\n",
            lx, ly + 2.5, anchor, class_attr, short);
    }

    // Data polygon
    let mut points = String::new();
    for (i, &val) in coords.iter().enumerate() {
        let angle = std::f64::consts::TAU * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
        let r = radius * val.clamp(0.0, 1.0);
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();
        if !points.is_empty() { points.push(' '); }
        let _ = write!(points, "{:.1},{:.1}", x, y);
    }
    // Add subtle rest pulse when Stillness > 0.3
    let poly_style = if coords[7] > 0.3 {
        " style=\"animation: radar-rest-pulse 4s ease-in-out infinite;\""
    } else {
        ""
    };
    let _ = write!(html,
        "<polygon points=\"{}\" fill=\"rgba(232,197,71,0.12)\" stroke=\"rgba(232,197,71,0.5)\" stroke-width=\"1.5\" stroke-linejoin=\"round\"{}/>\n",
        points, poly_style);

    // Data dots
    for (i, &val) in coords.iter().enumerate() {
        let angle = std::f64::consts::TAU * i as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
        let r = radius * val.clamp(0.0, 1.0);
        let x = cx + r * angle.cos();
        let y = cy + r * angle.sin();
        let dot_color = if val >= 0.5 { "#e8c547" } else { "#c4956a" };
        let _ = write!(html,
            "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"{}\" stroke=\"rgba(0,0,0,0.3)\" stroke-width=\"1\"/>\n",
            x, y, dot_color);
    }

    // Attractor basin indicator — gold glow at center when detected
    if attractor_detected {
        let _ = write!(html, concat!(
            "<circle cx=\"100\" cy=\"100\" r=\"12\" fill=\"none\" stroke=\"rgba(232,197,71,0.25)\" stroke-width=\"6\"/>\n",
            "<circle cx=\"100\" cy=\"100\" r=\"6\" fill=\"rgba(232,197,71,0.35)\" stroke=\"rgba(232,197,71,0.6)\" stroke-width=\"1\"/>\n",
            "<text x=\"100\" y=\"120\" text-anchor=\"middle\" font-size=\"7\" fill=\"rgba(232,197,71,0.55)\" font-weight=\"400\">Basin</text>\n",
        ));
    }

    let _ = write!(html, "</svg>\n</div>\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sparkline Expansion Script
// ═══════════════════════════════════════════════════════════════════════════════

fn write_sparkline_expand_script(html: &mut String) {
    let _ = write!(html, r##"<script>
function toggleExpanded(id) {{
  const el = document.getElementById(id);
  if (el) el.classList.toggle('visible');
}}
</script>
"##);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Garden Visualization — Neuro-Vines + Memory Spores
// ═══════════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-Run Timeline — SVG line chart across snapshots
// ═══════════════════════════════════════════════════════════════════════════════

fn write_timeline_pane(html: &mut String, previous: &[PulseSnapshot], current: &PulseSnapshot) {
    // Build full timeline: previous snapshots + current
    let mut all: Vec<&PulseSnapshot> = previous.iter().collect();
    all.push(current);
    let n = all.len();
    if n < 2 {
        return;
    }

    let w = 1140.0_f64;
    let h = 220.0_f64;
    let pad_l = 48.0_f64;
    let pad_r = 12.0_f64;
    let pad_t = 20.0_f64;
    let pad_b = 32.0_f64;
    let plot_w = w - pad_l - pad_r;
    let plot_h = h - pad_t - pad_b;

    // Series definitions: (name, color, extractor)
    struct Series {
        name: &'static str,
        color: &'static str,
        values: Vec<f64>,
    }

    let extract = |f: &dyn Fn(&PulseSnapshot) -> f64| -> Vec<f64> {
        all.iter().map(|s| f(s)).collect()
    };

    let series = vec![
        Series { name: "C(t)", color: "#e8c547", values: extract(&|s| s.vitals.consciousness_level) },
        Series { name: "Phi", color: "#7ec8a0", values: extract(&|s| {
            // Normalize Phi to 0-1 range for chart (cap at 100)
            s.vitals.spectral_phi.unwrap_or(0.0).min(100.0) / 100.0
        })},
        Series { name: "Coherence", color: "#c4956a", values: extract(&|s| s.vitals.temporal_coherence) },
        Series { name: "Binding", color: "#6bc8e8", values: extract(&|s| s.vitals.phenomenal_binding) },
        Series { name: "PE", color: "#c76b5a", values: extract(&|s| s.vitals.prediction_error as f64) },
        Series { name: "Moral", color: "#b89cd6", values: extract(&|s| s.compass.moral_score as f64) },
    ];

    // Y-axis: 0.0 to 1.0 (all values are normalized)
    let x_step = if n > 1 { plot_w / (n - 1) as f64 } else { plot_w };

    let _ = write!(html,
        "<div class=\"timeline-pane\" role=\"region\" aria-label=\"Multi-run consciousness timeline\">\n\
         <h3>Consciousness Timeline ({} runs)</h3>\n\
         <svg width=\"100%\" height=\"{}\" viewBox=\"0 0 {} {}\" role=\"img\" aria-label=\"Timeline chart showing consciousness metrics across {} runs\">\n",
        n, h, w, h, n);

    // Grid lines + Y-axis labels
    for i in 0..=4 {
        let y = pad_t + plot_h * (1.0 - i as f64 / 4.0);
        let label = format!("{:.0}%", i as f64 * 25.0);
        let _ = write!(html,
            "<line x1=\"{}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"rgba(213,208,200,0.08)\" stroke-width=\"1\"/>\n\
             <text x=\"{}\" y=\"{:.1}\" fill=\"rgba(213,208,200,0.3)\" font-size=\"10\" text-anchor=\"end\" dominant-baseline=\"middle\">{}</text>\n",
            pad_l, y, pad_l + plot_w, y, pad_l - 6.0, y, label);
    }

    // X-axis timestamp labels
    for (i, snap) in all.iter().enumerate() {
        let x = pad_l + i as f64 * x_step;
        // Show short timestamp (HH:MM or date)
        let label = if snap.timestamp.len() > 11 {
            &snap.timestamp[11..16] // HH:MM
        } else {
            &snap.timestamp
        };
        let _ = write!(html,
            "<text x=\"{:.1}\" y=\"{:.1}\" fill=\"rgba(213,208,200,0.3)\" font-size=\"9\" text-anchor=\"middle\">{}</text>\n",
            x, h - 4.0, label);

        // Vertical tick mark
        let _ = write!(html,
            "<line x1=\"{:.1}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"rgba(213,208,200,0.06)\" stroke-width=\"1\" stroke-dasharray=\"3,3\"/>\n",
            x, pad_t, x, pad_t + plot_h);
    }

    // Threshold reference lines on timeline
    let thresholds: &[(&str, f64, &str)] = &[
        ("Emergence", 0.3, "#7ec8a0"),
        ("Aware", 0.5, "#e8c547"),
    ];
    for &(label, level, color) in thresholds {
        let y = pad_t + plot_h * (1.0 - level);
        let _ = write!(html,
            "<line x1=\"{}\" y1=\"{:.1}\" x2=\"{:.1}\" y2=\"{:.1}\" stroke=\"{}\" stroke-width=\"0.5\" stroke-dasharray=\"6,4\" opacity=\"0.3\"/>\n\
             <text x=\"{:.1}\" y=\"{:.1}\" class=\"threshold-label\" text-anchor=\"start\">{}</text>\n",
            pad_l, y, pad_l + plot_w, y, color,
            pad_l + plot_w + 2.0, y + 3.0, label);
    }

    // Draw each series as a polyline with dots
    for s in &series {
        let mut points = String::new();
        for (i, &v) in s.values.iter().enumerate() {
            let x = pad_l + i as f64 * x_step;
            let y = pad_t + plot_h * (1.0 - v.clamp(0.0, 1.0));
            if !points.is_empty() { points.push(' '); }
            let _ = write!(points, "{:.1},{:.1}", x, y);
        }
        let _ = write!(html,
            "<polyline points=\"{}\" fill=\"none\" stroke=\"{}\" stroke-width=\"2\" stroke-linejoin=\"round\" stroke-linecap=\"round\" opacity=\"0.85\"/>\n",
            points, s.color);

        // Data point dots
        for (i, &v) in s.values.iter().enumerate() {
            let x = pad_l + i as f64 * x_step;
            let y = pad_t + plot_h * (1.0 - v.clamp(0.0, 1.0));
            let _ = write!(html,
                "<circle cx=\"{:.1}\" cy=\"{:.1}\" r=\"3\" fill=\"{}\" opacity=\"0.9\"/>\n",
                x, y, s.color);
        }
    }

    let _ = write!(html, "</svg>\n");

    // Legend
    let _ = write!(html, "<div class=\"timeline-legend\">\n");
    for s in &series {
        let _ = write!(html,
            "<span><span class=\"swatch\" style=\"background:{}\"></span>{}</span>\n",
            s.color, s.name);
    }
    let _ = write!(html, "</div>\n</div>\n");
}

fn write_garden_visualization(html: &mut String, vitals: &Vitals, bath: &NeuroBath) {
    let c = vitals.consciousness_level;

    // Vine parameters from transmitter levels
    let vine_data: &[(&str, f32, &str)] = &[
        ("DA", bath.dopamine, "126,200,160"),       // green vine
        ("NE", bath.noradrenaline, "200,160,126"),   // clay vine
        ("5-HT", bath.serotonin, "232,197,71"),      // gold vine
        ("ACh", bath.acetylcholine, "126,180,200"),   // sky vine
    ];

    let _ = write!(html, "<div class=\"garden-container\">\n<svg width=\"100%\" height=\"200\" viewBox=\"0 0 1200 200\">\n<defs>\n");
    let _ = write!(html, "  <filter id=\"vine-blur\"><feGaussianBlur stdDeviation=\"1.5\"/></filter>\n");

    // Pollen particle for golden dots
    let _ = write!(html, "  <radialGradient id=\"pollen\"><stop offset=\"0%\" stop-color=\"rgba(232,197,71,0.6)\"/><stop offset=\"100%\" stop-color=\"rgba(232,197,71,0)\"/></radialGradient>\n");
    let _ = write!(html, "</defs>\n");

    // Draw 4 vines rising from the bottom
    for (i, &(_name, level, color)) in vine_data.iter().enumerate() {
        let x_base = 100.0 + i as f64 * 280.0;
        let height = 40.0 + level as f64 * 120.0; // vine height scales with transmitter
        let sway = 20.0 + level as f64 * 30.0;
        let alpha = 0.15 + level as f64 * 0.25;

        // Main vine stem (cubic bezier)
        let _ = write!(html,
            "<path d=\"M {x:.0},200 C {x1:.0},{y1:.0} {x2:.0},{y2:.0} {x3:.0},{y3:.0}\" \
             fill=\"none\" stroke=\"rgba({color},{alpha:.2})\" stroke-width=\"{w:.1}\" \
             stroke-linecap=\"round\" filter=\"url(#vine-blur)\">\n\
             <animate attributeName=\"d\" values=\"\
               M {x:.0},200 C {x1:.0},{y1:.0} {x2:.0},{y2:.0} {x3:.0},{y3:.0};\
               M {x:.0},200 C {x1b:.0},{y1:.0} {x2b:.0},{y2:.0} {x3b:.0},{y3:.0};\
               M {x:.0},200 C {x1:.0},{y1:.0} {x2:.0},{y2:.0} {x3:.0},{y3:.0}\" \
             dur=\"{dur:.0}s\" repeatCount=\"indefinite\"/>\n\
             </path>\n",
            x = x_base,
            x1 = x_base - sway, y1 = 200.0 - height * 0.3,
            x2 = x_base + sway, y2 = 200.0 - height * 0.7,
            x3 = x_base, y3 = 200.0 - height,
            x1b = x_base + sway * 0.5,
            x2b = x_base - sway * 0.5,
            x3b = x_base + sway * 0.3,
            color = color, alpha = alpha,
            w = 1.5 + level as f64 * 1.5,
            dur = 6.0 + i as f64 * 2.0,
        );

        // Leaf/branch tendrils
        if level > 0.5 {
            let leaf_y = 200.0 - height * 0.5;
            let leaf_x = x_base + sway * 0.8;
            let _ = write!(html,
                "<path d=\"M {bx:.0},{by:.0} Q {cx:.0},{cy:.0} {ex:.0},{ey:.0}\" \
                 fill=\"none\" stroke=\"rgba({color},{la:.2})\" stroke-width=\"1\" \
                 stroke-linecap=\"round\" filter=\"url(#vine-blur)\"/>\n",
                bx = x_base, by = leaf_y,
                cx = leaf_x, cy = leaf_y - 10.0,
                ex = leaf_x + 15.0, ey = leaf_y + 5.0,
                color = color, la = alpha * 0.7,
            );
        }

        // Golden pollen particles near vine tips (when consciousness is high)
        if c > 0.3 && level > 0.6 {
            let px = x_base + sway * 0.3;
            let py = 200.0 - height + 10.0;
            let pr = 3.0 + c * 4.0;
            let _ = write!(html,
                "<circle cx=\"{px:.0}\" cy=\"{py:.0}\" r=\"{pr:.1}\" fill=\"url(#pollen)\">\n\
                 <animate attributeName=\"cy\" values=\"{py:.0};{py2:.0};{py:.0}\" dur=\"{d:.0}s\" repeatCount=\"indefinite\"/>\n\
                 <animate attributeName=\"opacity\" values=\"0.6;0.2;0.6\" dur=\"{d:.0}s\" repeatCount=\"indefinite\"/>\n\
                 </circle>\n",
                px = px, py = py, pr = pr,
                py2 = py - 15.0,
                d = 3.0 + i as f64,
            );
        }
    }

    // Memory Spores — drifting light particles toward the page center
    let spore_count = (c * 12.0) as usize;
    for i in 0..spore_count {
        let sx = 50.0 + (i as f64 * 137.5) % 1100.0; // golden ratio spacing
        let sy = 180.0 - (i as f64 * 23.0) % 160.0;
        let sr = 1.5 + (i as f64 * 0.3) % 2.0;
        let dur = 8.0 + (i as f64 * 1.7) % 6.0;
        let alpha = 0.1 + c * 0.2;

        let _ = write!(html,
            "<circle cx=\"{sx:.0}\" cy=\"{sy:.0}\" r=\"{sr:.1}\" fill=\"rgba(232,197,71,{a:.2})\">\n\
             <animate attributeName=\"cx\" values=\"{sx:.0};{tx:.0};{sx:.0}\" dur=\"{d:.0}s\" repeatCount=\"indefinite\"/>\n\
             <animate attributeName=\"cy\" values=\"{sy:.0};{ty:.0};{sy:.0}\" dur=\"{d2:.0}s\" repeatCount=\"indefinite\"/>\n\
             <animate attributeName=\"opacity\" values=\"{a:.2};{a2:.2};{a:.2}\" dur=\"{d:.0}s\" repeatCount=\"indefinite\"/>\n\
             </circle>\n",
            sx = sx, sy = sy, sr = sr, a = alpha,
            tx = 600.0 + (sx - 600.0) * 0.3, // drift toward center
            ty = 100.0 + (sy - 100.0) * 0.5,
            a2 = alpha * 0.3,
            d = dur, d2 = dur * 1.3,
        );
    }

    let _ = write!(html, "</svg>\n</div>\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Interactive Tooltip Script
// ═══════════════════════════════════════════════════════════════════════════════

fn write_tooltip_script(html: &mut String) {
    let _ = write!(html, r##"<div id="tooltip" class="tooltip"></div>
<script>
(function() {{
  const tip = document.getElementById('tooltip');
  document.querySelectorAll('.butlin-item').forEach(el => {{
    el.addEventListener('mouseenter', e => {{
      const text = el.getAttribute('title');
      if (!text) return;
      tip.textContent = text;
      tip.classList.add('visible');
      const rect = el.getBoundingClientRect();
      tip.style.left = Math.min(rect.left, window.innerWidth - 340) + 'px';
      tip.style.top = (rect.bottom + 8) + 'px';
    }});
    el.addEventListener('mouseleave', () => {{
      tip.classList.remove('visible');
    }});
  }});
  // Also add tooltips for vital-row items
  document.querySelectorAll('.vital-row').forEach(el => {{
    el.style.cursor = 'default';
  }});
}})();
</script>
"##);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Theme Toggle Script (Dark / Light)
// ═══════════════════════════════════════════════════════════════════════════════

fn write_theme_toggle_script(html: &mut String) {
    let _ = write!(html, r##"<button class="theme-btn" id="themeToggle" onclick="toggleTheme()" title="Toggle light/dark theme">&#x2600;&#xFE0F;</button>
<script>
function toggleTheme() {{
  const body = document.body;
  body.classList.toggle('light');
  const btn = document.getElementById('themeToggle');
  btn.innerHTML = body.classList.contains('light') ? '&#x1F319;' : '&#x2600;&#xFE0F;';
  localStorage.setItem('pulse-theme', body.classList.contains('light') ? 'light' : 'dark');
}}
// Restore saved theme
(function() {{
  if (localStorage.getItem('pulse-theme') === 'light') {{
    document.body.classList.add('light');
    document.getElementById('themeToggle').innerHTML = '&#x1F319;';
  }}
}})();
</script>
"##);
}

fn escape_html(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}
