// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Consciousness-driven music generation page.
//!
//! Provides sliders for arousal, valence, and phi (consciousness level),
//! a play/stop toggle, and a Canvas2D visualization showing:
//! - Waveform oscilloscope (simulated from consciousness params)
//! - Spectrum bars (64 frequency bins)
//! - Consciousness orbit trail (V-A trajectory)
//! - Phi meter gauge
//!
//! Also listens on the Holochain app WebSocket (`ws://localhost:8888`) for
//! `consciousness_composition` signals emitted by the music-bridge zome and
//! displays the latest composition metadata in real time.

use leptos::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, MessageEvent, WebSocket};

use crate::live_synth::{LiveSynth, LiveSynthRunner};

/// Consciousness bridge WebSocket relay.
/// Falls back to the Holochain conductor if the bridge isn't running.
/// Override per-deployment via `window.__HC_CONDUCTOR_URL` in index.html.
const BRIDGE_WS: &str = "ws://localhost:8893";
const HC_APP_WS: &str = "ws://localhost:8888";

/// Metadata extracted from a `consciousness_composition` Holochain signal.
#[derive(Clone, Debug, Default)]
struct CompositionMeta {
    tempo_bpm: f32,
    scale_name: String,
    note_count: u32,
    quality_score: f32,
    phi_score: f32,
    valence: f32,
    arousal: f32,
    narrative_tags: Vec<String>,
}

/// Maximum orbit trail length.
const ORBIT_TRAIL_LEN: usize = 200;

/// Consciousness-driven music generation page.
#[component]
pub fn ConsciousnessPage() -> impl IntoView {
    // Consciousness parameter signals
    let (arousal, set_arousal) = signal(0.5_f64);
    let (valence, set_valence) = signal(0.0_f64);
    let (phi, set_phi) = signal(0.5_f64);
    let (is_playing, set_playing) = signal(false);

    // Latest consciousness composition from Holochain signal
    let (last_comp, set_last_comp) = signal(Option::<CompositionMeta>::None);
    let (hc_status, set_hc_status) = signal("Connecting…".to_string());

    // Keeps the WebSocket alive for the component lifetime (WebSocket is !Send → new_local)
    let _ws_handle = StoredValue::new_local(Option::<WebSocket>::None);

    // Live synthesis engine — creates Web Audio graph, renders consciousness-driven audio
    let synth_runner: StoredValue<Option<Rc<RefCell<LiveSynthRunner>>>, LocalStorage> =
        StoredValue::new_local(None);

    let canvas_ref = NodeRef::<leptos::html::Canvas>::new();

    // ── Holochain signal listener ──────────────────────────────────────────────
    // Opens a WebSocket to the shared ecosystem conductor and listens for
    // `consciousness_composition` signals emitted by the music-bridge zome.
    // Updates `last_comp` and mirrors arousal/valence/phi into the viz.
    Effect::new(move |_| {
        // Try bridge relay first (has signal broadcast); fall back to conductor.
        // Allow override via window.__HC_CONDUCTOR_URL for production deployments.
        let url = js_sys::Reflect::get(
            &wasm_bindgen::JsValue::from(web_sys::window().unwrap()),
            &wasm_bindgen::JsValue::from_str("__HC_CONDUCTOR_URL"),
        )
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| {
            // Try bridge relay first, fall back to conductor
            if WebSocket::new(BRIDGE_WS).is_ok() {
                BRIDGE_WS.to_string()
            } else {
                HC_APP_WS.to_string()
            }
        });

        let ws = match WebSocket::new(&url) {
            Ok(ws) => ws,
            Err(_) => {
                set_hc_status.set("⚠ no signal source".to_string());
                return;
            }
        };

        // onopen
        {
            let set_status = set_hc_status.clone();
            let on_open = Closure::wrap(Box::new(move |_: web_sys::Event| {
                set_status.set("Connected".to_string());
            }) as Box<dyn FnMut(web_sys::Event)>);
            ws.set_onopen(Some(on_open.as_ref().unchecked_ref()));
            on_open.forget();
        }

        // onerror / onclose
        {
            let set_status = set_hc_status.clone();
            let on_close = Closure::wrap(Box::new(move |_: web_sys::CloseEvent| {
                set_status.set("Disconnected".to_string());
            }) as Box<dyn FnMut(web_sys::CloseEvent)>);
            ws.set_onclose(Some(on_close.as_ref().unchecked_ref()));
            on_close.forget();
        }

        // onmessage — parse Holochain signal envelope and extract composition metadata
        {
            let set_comp = set_last_comp.clone();
            let set_arousal2 = set_arousal.clone();
            let set_valence2 = set_valence.clone();
            let set_phi2 = set_phi.clone();
            let on_msg = Closure::wrap(Box::new(move |ev: MessageEvent| {
                let Some(text) = ev.data().as_string() else { return };
                // Holochain signal envelope: {"type":"Signal","data":{...}}
                // We look for the payload object containing signal_type.
                // Use a simple JSON scan rather than pulling in a full parser.
                if !text.contains("consciousness_composition") { return }

                // Parse with js_sys::JSON for zero-dependency JSON access
                let Ok(val) = js_sys::JSON::parse(&text) else { return };
                // Drill: .data.App.payload  OR  .data.payload  (conductor version-dependent)
                let payload = find_payload(&val);
                let Some(obj) = payload.as_ref().and_then(|v| v.dyn_ref::<js_sys::Object>()) else { return };

                let meta = CompositionMeta {
                    tempo_bpm: get_f32(obj, "tempo_bpm"),
                    scale_name: get_string(obj, "scale_name"),
                    note_count: get_f32(obj, "note_count") as u32,
                    quality_score: get_f32(obj, "quality_score"),
                    phi_score: get_f32(obj, "phi_score"),
                    valence: get_f32(obj, "valence"),
                    arousal: get_f32(obj, "arousal"),
                    narrative_tags: get_string_array(obj, "narrative_tags"),
                };

                // Mirror signal values into the viz sliders (triggers synth state update via Effect)
                set_arousal2.set(meta.arousal as f64);
                set_valence2.set(meta.valence as f64);
                set_phi2.set(meta.phi_score as f64);
                set_comp.set(Some(meta));

                // Auto-start synth on first signal if not already playing
                if !is_playing.get_untracked() {
                    set_playing.set(true);
                }
            }) as Box<dyn FnMut(MessageEvent)>);
            ws.set_onmessage(Some(on_msg.as_ref().unchecked_ref()));
            on_msg.forget();
        }

        // Keep WebSocket alive for the component lifetime
        _ws_handle.set_value(Some(ws));
    });

    // Reactive synth lifecycle — start/stop the Web Audio engine when is_playing changes.
    // This lets both the button and auto-start (from HC signals) control audio.
    Effect::new(move |_| {
        let playing = is_playing.get();
        synth_runner.update_value(|r| {
            if playing && r.is_none() {
                // Start
                if let Ok(ls) = LiveSynth::new() {
                    let engine = ls.engine();
                    engine.borrow_mut().set_state(
                        arousal.get_untracked() as f32,
                        valence.get_untracked() as f32,
                        phi.get_untracked() as f32,
                    );
                    *r = Some(ls.start());
                }
            } else if !playing && r.is_some() {
                // Stop
                if let Some(runner) = r.as_ref() {
                    runner.borrow_mut().stop();
                }
                *r = None;
            }
        });
    });

    // Reactive synth state update — whenever arousal/valence/phi change (slider or signal),
    // push the new state into the running synth engine so audio tracks consciousness live.
    Effect::new(move |_| {
        let a = arousal.get() as f32;
        let v = valence.get() as f32;
        let p = phi.get() as f32;
        synth_runner.with_value(|r| {
            if let Some(runner) = r.as_ref() {
                runner.borrow().engine().borrow_mut().set_state(a, v, p);
            }
        });
    });

    // Derived emotion label from V-A quadrant
    let emotion_label = move || {
        let a = arousal.get();
        let v = valence.get();
        if a > 0.6 && v > 0.3 {
            "Excitement / Joy"
        } else if a > 0.6 && v < -0.3 {
            "Panic / Anger"
        } else if a < 0.4 && v > 0.3 {
            "Contentment / Peace"
        } else if a < 0.4 && v < -0.3 {
            "Sadness / Grief"
        } else {
            "Neutral / Exploratory"
        }
    };

    // Phi tier label
    let phi_tier = move || {
        let p = phi.get();
        if p >= 0.8 {
            "Guardian"
        } else if p >= 0.6 {
            "Contributor"
        } else if p >= 0.4 {
            "Participant"
        } else if p >= 0.2 {
            "Observer"
        } else {
            "Dormant"
        }
    };

    // Canvas animation loop — reads live signals for arousal/valence/phi
    Effect::new(move |_| {
        let Some(canvas_el) = canvas_ref.get() else {
            return;
        };
        let canvas: HtmlCanvasElement = canvas_el.into();

        let w = 800_u32;
        let h = 400_u32;
        canvas.set_width(w);
        canvas.set_height(h);

        let ctx = canvas
            .get_context("2d")
            .unwrap()
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()
            .unwrap();

        let width = w as f64;
        let height = h as f64;

        // Orbit trail buffer
        let orbit_trail: Rc<RefCell<Vec<(f64, f64)>>> = Rc::new(RefCell::new(Vec::new()));

        let frame_ref: Rc<RefCell<Option<Closure<dyn FnMut()>>>> =
            Rc::new(RefCell::new(None));
        let frame_ref_clone = frame_ref.clone();
        let orbit_trail_clone = orbit_trail.clone();

        let closure = Closure::wrap(Box::new(move || {
            let time = web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now() / 1000.0)
                .unwrap_or(0.0);

            let a = arousal.get_untracked();
            let v = valence.get_untracked();
            let p = phi.get_untracked();
            let playing = is_playing.get_untracked();

            // Clear canvas
            ctx.set_fill_style_str("#0f1d14");
            ctx.fill_rect(0.0, 0.0, width, height);

            // Layout: 4 quadrants
            // Top-left: Waveform (400x200)
            // Top-right: Spectrum (400x200)
            // Bottom-left: Orbit trail (400x200)
            // Bottom-right: Phi meter (400x200)
            let hw = width / 2.0;
            let hh = height / 2.0;

            // ── Waveform Oscilloscope (top-left) ──
            draw_waveform(&ctx, 0.0, 0.0, hw, hh, time, a, v, p, playing);

            // ── Spectrum Bars (top-right) ──
            draw_spectrum(&ctx, hw, 0.0, hw, hh, time, a, v, p, playing);

            // ── Consciousness Orbit (bottom-left) ──
            {
                let mut trail = orbit_trail_clone.borrow_mut();
                trail.push((v, a));
                if trail.len() > ORBIT_TRAIL_LEN {
                    trail.remove(0);
                }
                draw_orbit(&ctx, 0.0, hh, hw, hh, &trail, v, a, p);
            }

            // ── Phi Meter (bottom-right) ──
            draw_phi_meter(&ctx, hw, hh, hw, hh, p, phi_tier_str(p));

            // Section labels
            ctx.set_fill_style_str("rgba(126, 200, 160, 0.5)");
            ctx.set_font("11px monospace");
            ctx.fill_text("WAVEFORM", 8.0, 14.0).ok();
            ctx.fill_text("SPECTRUM", hw + 8.0, 14.0).ok();
            ctx.fill_text("V-A ORBIT", 8.0, hh + 14.0).ok();
            ctx.fill_text("PHI METER", hw + 8.0, hh + 14.0).ok();

            // Separator lines
            ctx.set_stroke_style_str("rgba(126, 200, 160, 0.15)");
            ctx.set_line_width(1.0);
            ctx.begin_path();
            ctx.move_to(hw, 0.0);
            ctx.line_to(hw, height);
            ctx.move_to(0.0, hh);
            ctx.line_to(width, hh);
            ctx.stroke();

            // Schedule next frame
            if let Some(window) = web_sys::window() {
                if let Some(ref cb) = *frame_ref_clone.borrow() {
                    let _ = window.request_animation_frame(cb.as_ref().unchecked_ref());
                }
            }
        }) as Box<dyn FnMut()>);

        if let Some(window) = web_sys::window() {
            let _ = window.request_animation_frame(closure.as_ref().unchecked_ref());
        }

        *frame_ref.borrow_mut() = Some(closure);
    });

    view! {
        <div class="page consciousness-page">
            <h1>"Consciousness Music"</h1>
            <p class="page-subtitle">
                "Generate music from consciousness state. "
                "Adjust arousal, valence, and phi to shape the sound."
            </p>

            // Parameter controls
            <div class="consciousness-controls">
                <div class="control-group">
                    <label class="control-label">
                        "Valence "
                        <span class="control-value">{move || format!("{:.2}", valence.get())}</span>
                    </label>
                    <div class="slider-row">
                        <span class="slider-label-left">"Negative"</span>
                        <input
                            type="range"
                            class="consciousness-slider"
                            min="-1"
                            max="1"
                            step="0.01"
                            prop:value={move || valence.get().to_string()}
                            on:input=move |ev| {
                                let val: f64 = event_target_value(&ev).parse().unwrap_or(0.0);
                                set_valence.set(val);
                            }
                        />
                        <span class="slider-label-right">"Positive"</span>
                    </div>
                </div>

                <div class="control-group">
                    <label class="control-label">
                        "Arousal "
                        <span class="control-value">{move || format!("{:.2}", arousal.get())}</span>
                    </label>
                    <div class="slider-row">
                        <span class="slider-label-left">"Calm"</span>
                        <input
                            type="range"
                            class="consciousness-slider"
                            min="0"
                            max="1"
                            step="0.01"
                            prop:value={move || arousal.get().to_string()}
                            on:input=move |ev| {
                                let val: f64 = event_target_value(&ev).parse().unwrap_or(0.5);
                                set_arousal.set(val);
                            }
                        />
                        <span class="slider-label-right">"Intense"</span>
                    </div>
                </div>

                <div class="control-group">
                    <label class="control-label">
                        "Consciousness (Phi) "
                        <span class="control-value">{move || format!("{:.2}", phi.get())}</span>
                        <span class="phi-tier">" \u{2014} " {phi_tier}</span>
                    </label>
                    <div class="slider-row">
                        <span class="slider-label-left">"Dormant"</span>
                        <input
                            type="range"
                            class="consciousness-slider"
                            min="0"
                            max="1"
                            step="0.01"
                            prop:value={move || phi.get().to_string()}
                            on:input=move |ev| {
                                let val: f64 = event_target_value(&ev).parse().unwrap_or(0.5);
                                set_phi.set(val);
                            }
                        />
                        <span class="slider-label-right">"Guardian"</span>
                    </div>
                </div>
            </div>

            // Play/Stop button — toggles is_playing; the lifecycle Effect handles the rest
            <div class="consciousness-playback">
                <button
                    class="btn btn-primary consciousness-play-btn"
                    on:click=move |_| set_playing.update(|p| *p = !*p)
                >
                    {move || if is_playing.get() { "Stop Synthesis" } else { "Start Synthesis" }}
                </button>
                <span class="playback-status">
                    {move || if is_playing.get() {
                        "Generating..."
                    } else {
                        "Idle"
                    }}
                </span>
            </div>

            // Emotion state display
            <div class="consciousness-state">
                <div class="state-label">
                    <span class="state-icon">
                        {move || {
                            let a = arousal.get();
                            let v = valence.get();
                            if a > 0.6 && v > 0.3 { "\u{1f525}" }
                            else if a > 0.6 && v < -0.3 { "\u{26a1}" }
                            else if a < 0.4 && v > 0.3 { "\u{1f33f}" }
                            else if a < 0.4 && v < -0.3 { "\u{1f327}" }
                            else { "\u{1f30a}" }
                        }}
                    </span>
                    <span class="emotion-text">{emotion_label}</span>
                </div>
            </div>

            // Canvas2D visualization (waveform, spectrum, orbit, phi meter)
            <div class="visualization-container">
                <canvas node_ref=canvas_ref width="800" height="400"
                    style="width: 100%; max-width: 800px; border-radius: 8px; border: 1px solid rgba(126, 200, 160, 0.2);"
                ></canvas>
            </div>

            // Holochain signal panel: latest consciousness composition
            <div class="consciousness-signal-panel">
                <div class="signal-header">
                    <span class="signal-title">"Latest Consciousness Composition"</span>
                    <span class="signal-status"
                        style=move || format!(
                            "color: {}",
                            if hc_status.get().starts_with("Connected") { "#7ec8a0" }
                            else if hc_status.get().starts_with("⚠") { "#e8c547" }
                            else { "#888" }
                        )
                    >
                        {hc_status}
                    </span>
                </div>
                {move || match last_comp.get() {
                    None => view! {
                        <p class="signal-empty">
                            "No composition received yet. "
                            "Start Symthaea's cognitive loop with the Music modality enabled "
                            "to see live compositions appear here."
                        </p>
                    }.into_any(),
                    Some(meta) => view! {
                        <div class="signal-meta-grid">
                            <div class="signal-meta-item">
                                <span class="meta-label">"Tempo"</span>
                                <span class="meta-value">{format!("{:.0} BPM", meta.tempo_bpm)}</span>
                            </div>
                            <div class="signal-meta-item">
                                <span class="meta-label">"Scale"</span>
                                <span class="meta-value">{meta.scale_name.replace('_', " ")}</span>
                            </div>
                            <div class="signal-meta-item">
                                <span class="meta-label">"Notes"</span>
                                <span class="meta-value">{meta.note_count.to_string()}</span>
                            </div>
                            <div class="signal-meta-item">
                                <span class="meta-label">"Quality"</span>
                                <span class="meta-value">{format!("{:.2}", meta.quality_score)}</span>
                            </div>
                            <div class="signal-meta-item">
                                <span class="meta-label">"Phi (Φ)"</span>
                                <span class="meta-value">{format!("{:.3}", meta.phi_score)}</span>
                            </div>
                            <div class="signal-meta-item">
                                <span class="meta-label">"Valence / Arousal"</span>
                                <span class="meta-value">
                                    {format!("{:+.2} / {:.2}", meta.valence, meta.arousal)}
                                </span>
                            </div>
                            {if meta.narrative_tags.is_empty() {
                                view! { <div></div> }.into_any()
                            } else {
                                view! {
                                    <div class="signal-meta-item signal-tags">
                                        <span class="meta-label">"Narrative"</span>
                                        <span class="meta-value">
                                            {meta.narrative_tags.join(" · ")}
                                        </span>
                                    </div>
                                }.into_any()
                            }}
                        </div>
                    }.into_any(),
                }}
            </div>

            // Info section
            <div class="consciousness-info">
                <h2>"How It Works"</h2>
                <div class="info-grid">
                    <div class="info-card">
                        <h3>"Valence"</h3>
                        <p>"Emotional polarity from negative (minor keys, dissonance) to positive (major keys, consonance)."</p>
                    </div>
                    <div class="info-card">
                        <h3>"Arousal"</h3>
                        <p>"Energy level from calm (slow tempo, soft dynamics) to intense (fast tempo, loud dynamics)."</p>
                    </div>
                    <div class="info-card">
                        <h3>"Phi"</h3>
                        <p>"Consciousness integration level. Higher phi produces more complex, self-referential musical structures."</p>
                    </div>
                </div>
            </div>
        </div>
    }
}

// ── Holochain signal helpers ───────────────────────────────────────────────────

/// Drill into a Holochain signal envelope to find the zome payload.
/// Supports: { type: "Signal", data: { App: { payload: ... } } }
/// and simpler forms emitted by some conductor versions.
fn find_payload(val: &JsValue) -> Option<JsValue> {
    let obj = val.dyn_ref::<js_sys::Object>()?;

    // Try .data.App.payload
    let data = js_sys::Reflect::get(obj, &"data".into()).ok()?;
    if let Some(data_obj) = data.dyn_ref::<js_sys::Object>() {
        if let Ok(app) = js_sys::Reflect::get(data_obj, &"App".into()) {
            if let Some(app_obj) = app.dyn_ref::<js_sys::Object>() {
                if let Ok(payload) = js_sys::Reflect::get(app_obj, &"payload".into()) {
                    return Some(payload);
                }
            }
        }
        // Try .data.payload (flat envelope)
        if let Ok(payload) = js_sys::Reflect::get(data_obj, &"payload".into()) {
            return Some(payload);
        }
        // .data itself might be the payload
        return Some(data);
    }
    None
}

fn get_f32(obj: &js_sys::Object, key: &str) -> f32 {
    js_sys::Reflect::get(obj, &key.into())
        .ok()
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0) as f32
}

fn get_string(obj: &js_sys::Object, key: &str) -> String {
    js_sys::Reflect::get(obj, &key.into())
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_default()
}

fn get_string_array(obj: &js_sys::Object, key: &str) -> Vec<String> {
    js_sys::Reflect::get(obj, &key.into())
        .ok()
        .and_then(|v| v.dyn_into::<js_sys::Array>().ok())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_string())
                .collect()
        })
        .unwrap_or_default()
}

// ── Drawing functions ──

fn phi_tier_str(p: f64) -> &'static str {
    if p >= 0.8 { "Guardian" }
    else if p >= 0.6 { "Contributor" }
    else if p >= 0.4 { "Participant" }
    else if p >= 0.2 { "Observer" }
    else { "Dormant" }
}

/// Waveform oscilloscope — simulates audio waveform from consciousness params.
fn draw_waveform(
    ctx: &CanvasRenderingContext2d,
    x: f64, y: f64, w: f64, h: f64,
    time: f64, arousal: f64, valence: f64, phi: f64, playing: bool,
) {
    let mid_y = y + h / 2.0;
    let amplitude = if playing { 0.3 + arousal * 0.6 } else { 0.05 };
    let freq_base = 1.0 + arousal * 4.0; // Higher arousal = more cycles
    let harmonics = 1 + (phi * 5.0) as i32; // Phi adds harmonic complexity

    ctx.set_stroke_style_str(if valence > 0.0 {
        "rgba(126, 200, 160, 0.9)" // green for positive
    } else {
        "rgba(232, 197, 71, 0.9)" // gold for negative
    });
    ctx.set_line_width(1.5);
    ctx.begin_path();

    let padding = 20.0;
    let samples = 256;
    for i in 0..=samples {
        let t = i as f64 / samples as f64;
        let px = x + padding + t * (w - 2.0 * padding);

        // Composite waveform from harmonics
        let mut sample = 0.0;
        for h_idx in 1..=harmonics {
            let h_f = h_idx as f64;
            let freq = freq_base * h_f;
            let phase = time * freq * std::f64::consts::TAU + h_f * 0.7;
            let harmonic_amp = amplitude / h_f;
            // Valence shifts between sine (positive) and sawtooth-like (negative)
            if valence > 0.0 {
                sample += harmonic_amp * (phase).sin();
            } else {
                // More angular waveform for negative valence
                sample += harmonic_amp * (phase).sin() * (1.0 + (-valence) * (phase * 2.0).sin().abs());
            }
        }

        let py = mid_y - sample * (h / 2.0 - padding);
        if i == 0 {
            ctx.move_to(px, py);
        } else {
            ctx.line_to(px, py);
        }
    }
    ctx.stroke();

    // Amplitude guide lines
    ctx.set_stroke_style_str("rgba(126, 200, 160, 0.1)");
    ctx.set_line_width(0.5);
    ctx.begin_path();
    ctx.move_to(x + padding, mid_y);
    ctx.line_to(x + w - padding, mid_y);
    ctx.stroke();
}

/// Spectrum bars — 64 frequency bins derived from consciousness state.
fn draw_spectrum(
    ctx: &CanvasRenderingContext2d,
    x: f64, y: f64, w: f64, h: f64,
    time: f64, arousal: f64, valence: f64, phi: f64, playing: bool,
) {
    let bins = 64;
    let padding = 20.0;
    let bar_area_w = w - 2.0 * padding;
    let bar_w = bar_area_w / bins as f64 - 1.0;
    let max_bar_h = h - 2.0 * padding;

    for i in 0..bins {
        let freq_norm = i as f64 / bins as f64;

        // Simulated spectrum: arousal controls overall energy,
        // valence shifts spectral center, phi adds high-freq detail
        let center = 0.3 + valence * 0.2; // Spectral centroid
        let spread = 0.15 + phi * 0.2;
        let envelope = (-(freq_norm - center).powi(2) / (2.0 * spread * spread)).exp();
        let noise = ((time * 3.0 + i as f64 * 0.4).sin() * 0.3 + 0.7).abs();
        let base_level = if playing { arousal * 0.8 + 0.1 } else { 0.02 };
        let level = (envelope * base_level * noise).min(1.0);

        let bar_h = level * max_bar_h;
        let bx = x + padding + i as f64 * (bar_w + 1.0);
        let by = y + h - padding - bar_h;

        // Color gradient: low freq = warm, high freq = cool
        let r = ((1.0 - freq_norm) * 200.0 + 55.0) as u32;
        let g = (160.0 + freq_norm * 40.0) as u32;
        let b = (100.0 + freq_norm * 100.0) as u32;
        ctx.set_fill_style_str(&format!("rgb({r},{g},{b})"));
        ctx.fill_rect(bx, by, bar_w, bar_h);
    }
}

/// Consciousness orbit trail — plots V-A trajectory over time.
fn draw_orbit(
    ctx: &CanvasRenderingContext2d,
    x: f64, y: f64, w: f64, h: f64,
    trail: &[(f64, f64)], // (valence, arousal) pairs
    current_v: f64, current_a: f64, phi: f64,
) {
    let padding = 30.0;
    let cx = x + w / 2.0;
    let cy = y + h / 2.0;
    let scale_x = (w - 2.0 * padding) / 2.0; // valence: -1..1
    let scale_y = (h - 2.0 * padding) / 2.0; // arousal: 0..1, centered

    // Axis lines
    ctx.set_stroke_style_str("rgba(126, 200, 160, 0.15)");
    ctx.set_line_width(0.5);
    ctx.begin_path();
    ctx.move_to(x + padding, cy);
    ctx.line_to(x + w - padding, cy);
    ctx.move_to(cx, y + padding);
    ctx.line_to(cx, y + h - padding);
    ctx.stroke();

    // Axis labels
    ctx.set_fill_style_str("rgba(126, 200, 160, 0.4)");
    ctx.set_font("9px monospace");
    ctx.fill_text("V+", x + w - padding - 10.0, cy - 4.0).ok();
    ctx.fill_text("V-", x + padding + 2.0, cy - 4.0).ok();
    ctx.fill_text("A+", cx + 4.0, y + padding + 10.0).ok();
    ctx.fill_text("A-", cx + 4.0, y + h - padding - 2.0).ok();

    // Draw trail with fading alpha
    if trail.len() >= 2 {
        for i in 1..trail.len() {
            let alpha = (i as f64 / trail.len() as f64) * 0.8;
            let (v0, a0) = trail[i - 1];
            let (v1, a1) = trail[i];
            let px0 = cx + v0 * scale_x;
            let py0 = cy - (a0 - 0.5) * scale_y * 2.0;
            let px1 = cx + v1 * scale_x;
            let py1 = cy - (a1 - 0.5) * scale_y * 2.0;

            ctx.set_stroke_style_str(&format!("rgba(232, 197, 71, {:.2})", alpha));
            ctx.set_line_width(1.0 + alpha);
            ctx.begin_path();
            ctx.move_to(px0, py0);
            ctx.line_to(px1, py1);
            ctx.stroke();
        }
    }

    // Current position dot
    let curr_px = cx + current_v * scale_x;
    let curr_py = cy - (current_a - 0.5) * scale_y * 2.0;
    let dot_r = 4.0 + phi * 4.0; // Bigger dot = more consciousness

    ctx.set_fill_style_str("rgba(232, 197, 71, 0.9)");
    ctx.begin_path();
    ctx.arc(curr_px, curr_py, dot_r, 0.0, std::f64::consts::TAU).ok();
    ctx.fill();

    // Glow
    ctx.set_fill_style_str("rgba(232, 197, 71, 0.2)");
    ctx.begin_path();
    ctx.arc(curr_px, curr_py, dot_r * 2.5, 0.0, std::f64::consts::TAU).ok();
    ctx.fill();
}

/// Phi meter — vertical gauge showing consciousness level with tier thresholds.
fn draw_phi_meter(
    ctx: &CanvasRenderingContext2d,
    x: f64, y: f64, w: f64, h: f64,
    phi: f64, tier: &str,
) {
    let padding = 30.0;
    let bar_w = 40.0;
    let bar_h = h - 2.0 * padding;
    let bar_x = x + w / 2.0 - bar_w / 2.0;
    let bar_y = y + padding;

    // Background bar
    ctx.set_fill_style_str("rgba(255, 255, 255, 0.05)");
    ctx.fill_rect(bar_x, bar_y, bar_w, bar_h);

    // Filled portion
    let fill_h = phi * bar_h;
    let fill_y = bar_y + bar_h - fill_h;

    // Color gradient based on phi
    let color = if phi >= 0.8 {
        "rgba(232, 197, 71, 0.9)" // gold
    } else if phi >= 0.6 {
        "rgba(126, 200, 160, 0.9)" // green
    } else if phi >= 0.4 {
        "rgba(90, 184, 160, 0.8)" // teal
    } else if phi >= 0.2 {
        "rgba(91, 155, 213, 0.7)" // blue
    } else {
        "rgba(100, 100, 100, 0.5)" // grey
    };

    ctx.set_fill_style_str(color);
    ctx.fill_rect(bar_x, fill_y, bar_w, fill_h);

    // Tier threshold lines
    let thresholds = [(0.2, "Observer"), (0.4, "Participant"), (0.6, "Contributor"), (0.8, "Guardian")];
    for (thresh, label) in &thresholds {
        let ty = bar_y + bar_h * (1.0 - thresh);
        ctx.set_stroke_style_str("rgba(126, 200, 160, 0.3)");
        ctx.set_line_width(0.5);
        ctx.begin_path();
        ctx.move_to(bar_x - 5.0, ty);
        ctx.line_to(bar_x + bar_w + 5.0, ty);
        ctx.stroke();

        ctx.set_fill_style_str("rgba(126, 200, 160, 0.4)");
        ctx.set_font("9px monospace");
        ctx.fill_text(label, bar_x + bar_w + 10.0, ty + 3.0).ok();
    }

    // Current value + tier label
    ctx.set_fill_style_str("rgba(232, 197, 71, 1.0)");
    ctx.set_font("bold 16px monospace");
    ctx.set_text_align("center");
    ctx.fill_text(
        &format!("{:.2}", phi),
        x + w / 2.0,
        y + h - 8.0,
    ).ok();

    ctx.set_fill_style_str("rgba(126, 200, 160, 0.7)");
    ctx.set_font("12px monospace");
    ctx.fill_text(tier, x + w / 2.0, y + padding - 8.0).ok();
    ctx.set_text_align("start"); // reset
}
