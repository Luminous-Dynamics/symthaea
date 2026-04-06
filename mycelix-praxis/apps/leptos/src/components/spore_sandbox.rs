// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use std::cell::Cell;
use std::rc::Rc;

use js_sys::{JSON, Object, Reflect};
use leptos::prelude::*;
use serde_wasm_bindgen;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Event, MessageEvent, Worker, WorkerOptions, WorkerType};

const HEDGE_WORDS: [&str; 16] = [
    "maybe", "perhaps", "likely", "possibly", "uncertain", "unknown", "could", "might",
    "seems", "appears", "suggests", "probable", "estimate", "approx", "roughly", "assuming",
];

const FACTUAL_WORDS: [&str; 16] = [
    "is", "are", "was", "will", "must", "always", "never", "definitely",
    "certainly", "proves", "guarantees", "fact", "truth", "exact", "precisely", "confirmed",
];

const HARMONY_NAMES: [&str; 8] = [
    "Resonant\nCoherence",
    "Pan‑Sentient\nFlourishing",
    "Integral\nWisdom",
    "Infinite\nPlay",
    "Universal\nInterconnectedness",
    "Sacred\nReciprocity",
    "Evolutionary\nProgression",
    "Sacred\nStillness",
];

#[derive(Clone, Copy, PartialEq)]
pub enum SporeSandboxMode {
    EpistemicGate,
    MoralAlgebra,
}

impl SporeSandboxMode {
    fn title(self) -> &'static str {
        match self {
            SporeSandboxMode::EpistemicGate => "Epistemic Gate WASM Sandbox",
            SporeSandboxMode::MoralAlgebra => "Moral Algebra WASM Sandbox",
        }
    }

    fn subtitle(self) -> &'static str {
        match self {
            SporeSandboxMode::EpistemicGate => "Probe hallucination defenses, confidence gating, and output integrity.",
            SporeSandboxMode::MoralAlgebra => "Stress-test ethical reasoning and governance-aligned harmonies.",
        }
    }

    fn primary_action(self) -> (&'static str, &'static str) {
        match self {
            SporeSandboxMode::EpistemicGate => ("Run Epistemic Gate Cycle", "cycle"),
            SporeSandboxMode::MoralAlgebra => ("Run Moral Reasoning Cycle", "reasoning"),
        }
    }
}

#[derive(Clone, Default)]
struct SporeMetrics {
    consciousness_level: Option<f64>,
    honest_confidence: Option<f64>,
    harmony_alignment: Option<f64>,
    free_energy: Option<f64>,
    cycle_count: Option<f64>,
    dominant_harmony: Option<String>,
    safety_level: Option<String>,
    workspace_ignited: Option<bool>,
}

#[derive(Clone)]
struct EpistemicStatusLite {
    evidence_level: String,
    honest_confidence: f64,
    feasibility_gap: f64,
    disclaimer: String,
}

#[derive(Clone)]
struct TokenScore {
    token: String,
    support: f64,
    delta: f64,
}

#[derive(Clone)]
enum SporeStatus {
    Checking,
    Unavailable,
    Initializing,
    Ready,
    Error(String),
}

impl SporeStatus {
    fn label(&self) -> &'static str {
        match self {
            SporeStatus::Checking => "Checking",
            SporeStatus::Unavailable => "Unavailable",
            SporeStatus::Initializing => "Initializing",
            SporeStatus::Ready => "Ready",
            SporeStatus::Error(_) => "Error",
        }
    }

    fn css_class(&self) -> &'static str {
        match self {
            SporeStatus::Ready => "spore-status-ready",
            SporeStatus::Error(_) => "spore-status-error",
            SporeStatus::Unavailable => "spore-status-unavailable",
            _ => "spore-status-pending",
        }
    }
}

struct SporeWorkerHandle {
    worker: Worker,
    next_id: Cell<u32>,
}

impl SporeWorkerHandle {
    fn send(&self, action: &str, params: Option<JsValue>) -> Result<u32, JsValue> {
        let id = self.next_id.get();
        self.next_id.set(id.wrapping_add(1));
        let payload = Object::new();
        Reflect::set(&payload, &JsValue::from_str("id"), &JsValue::from_f64(id as f64))?;
        Reflect::set(&payload, &JsValue::from_str("action"), &JsValue::from_str(action))?;
        Reflect::set(&payload, &JsValue::from_str("params"), &params.unwrap_or(JsValue::NULL))?;
        self.worker.post_message(&payload)?;
        Ok(id)
    }
}

fn spore_available() -> bool {
    let window = web_sys::window().and_then(|w| Some(w.into()));
    if let Some(window) = window {
        Reflect::get(&window, &JsValue::from_str("__SPORE_AVAILABLE"))
            .ok()
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    } else {
        false
    }
}

fn spawn_worker() -> Result<Rc<SporeWorkerHandle>, String> {
    let mut options = WorkerOptions::new();
    options.set_type(WorkerType::Module);
    let worker = Worker::new_with_options("/spore-worker.js", &options)
        .map_err(|_| "Failed to start Spore worker".to_string())?;
    Ok(Rc::new(SporeWorkerHandle { worker, next_id: Cell::new(1) }))
}

fn js_stringify(value: &JsValue) -> String {
    JSON::stringify(value)
        .ok()
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "null".to_string())
}

fn js_f64(obj: &JsValue, key: &str) -> Option<f64> {
    Reflect::get(obj, &JsValue::from_str(key)).ok().and_then(|v| v.as_f64())
}

fn js_bool(obj: &JsValue, key: &str) -> Option<bool> {
    Reflect::get(obj, &JsValue::from_str(key)).ok().and_then(|v| v.as_bool())
}

fn js_string(obj: &JsValue, key: &str) -> Option<String> {
    Reflect::get(obj, &JsValue::from_str(key)).ok().and_then(|v| v.as_string())
}

fn fmt_opt(value: Option<f64>, digits: usize) -> String {
    value.map(|v| format!("{:.digits$}", v, digits = digits)).unwrap_or_else(|| "—".into())
}

fn clamp01(value: f64) -> f64 {
    value.max(0.0).min(1.0)
}

fn normalize_token(token: &str) -> String {
    token
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_lowercase()
}

fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

fn truncate_token(token: &str) -> String {
    let trimmed = token.trim_matches(|c: char| c.is_ascii_punctuation());
    if trimmed.len() > 8 {
        format!("{}…", &trimmed[..7])
    } else {
        trimmed.to_string()
    }
}

fn compute_support(tokens: &[String], base: f64) -> Vec<TokenScore> {
    tokens
        .iter()
        .map(|token| {
            let normalized = normalize_token(token);
            let mut support = base;
            let mut delta = 0.0;

            if !normalized.is_empty() && HEDGE_WORDS.contains(&normalized.as_str()) {
                delta += 0.12 + (1.0 - base) * 0.18;
            }
            if !normalized.is_empty() && FACTUAL_WORDS.contains(&normalized.as_str()) {
                delta -= 0.18 + (1.0 - base) * 0.22;
            }

            support = clamp01(support + delta);
            delta = support - base;

            TokenScore {
                token: token.clone(),
                support,
                delta,
            }
        })
        .collect()
}

fn parse_epistemic_status(value: &JsValue) -> Option<EpistemicStatusLite> {
    let status = Reflect::get(value, &JsValue::from_str("epistemic_status")).ok()?;
    let evidence_level = js_string(&status, "evidence_level")?;
    let honest_confidence = js_f64(&status, "honest_confidence")?;
    let feasibility_gap = js_f64(&status, "feasibility_gap").unwrap_or(0.0);
    let disclaimer = js_string(&status, "disclaimer").unwrap_or_default();
    Some(EpistemicStatusLite {
        evidence_level,
        honest_confidence,
        feasibility_gap,
        disclaimer,
    })
}

fn parse_harmony_scores(result: &JsValue) -> Option<Vec<f64>> {
    if let Some(raw) = result.as_string() {
        serde_json::from_str::<Vec<f64>>(&raw).ok()
    } else if result.is_object() || result.is_array() {
        serde_wasm_bindgen::from_value::<Vec<f64>>(result.clone()).ok()
    } else {
        None
    }
}

fn radar_points(values: &[f64], radius: f64, center: f64) -> String {
    let count = values.len().max(1) as f64;
    values
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let angle = (i as f64 / count) * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2;
            let r = clamp01(value) * radius;
            let x = center + r * angle.cos();
            let y = center + r * angle.sin();
            format!("{:.2},{:.2}", x, y)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[component]
pub fn SporeSandbox(mode: SporeSandboxMode) -> impl IntoView {
    let (status, set_status) = signal(SporeStatus::Checking);
    let (prompt, set_prompt) = signal(String::new());
    let (max_tokens, set_max_tokens) = signal(60u32);
    let (last_action, set_last_action) = signal(String::new());
    let (output, set_output) = signal(String::new());
    let (metrics, set_metrics) = signal(SporeMetrics::default());
    let (epistemic_status, set_epistemic_status) = signal(None::<EpistemicStatusLite>);
    let (generated_text, set_generated_text) = signal(String::new());
    let (token_scores, set_token_scores) = signal(Vec::<TokenScore>::new());
    let (harmony_scores, set_harmony_scores) = signal(Vec::<f64>::new());

    let worker_handle = StoredValue::new_local(None::<Rc<SporeWorkerHandle>>);

    let send_action: Rc<dyn Fn(&str, Option<JsValue>)> = Rc::new({
        let set_last_action = set_last_action.clone();
        let set_status = set_status.clone();
        move |action: &str, params: Option<JsValue>| {
            if let Some(handle) = worker_handle.get_value() {
                if handle.send(action, params).is_ok() {
                    set_last_action.set(action.to_string());
                } else {
                    set_status.set(SporeStatus::Error("Failed to send command".into()));
                }
            }
        }
    });

    let on_mount_send = send_action.clone();
    let (booted, set_booted) = signal(false);

    Effect::new(move |_| {
        if booted.get() {
            return;
        }
        set_booted.set(true);
        if !spore_available() {
            set_status.set(SporeStatus::Unavailable);
            return;
        }

        match spawn_worker() {
            Ok(handle) => {
                let worker = handle.worker.clone();
                worker_handle.set_value(Some(handle.clone()));
                set_status.set(SporeStatus::Initializing);

                let set_status_msg = set_status.clone();
                let set_output_msg = set_output.clone();
                let set_metrics_msg = set_metrics.clone();
                let set_epistemic_msg = set_epistemic_status.clone();
                let set_generated_msg = set_generated_text.clone();
                let set_harmony_msg = set_harmony_scores.clone();
                let handle_for_metrics = handle.clone();
                let handle_for_harmony = handle.clone();

                let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
                    let data = event.data();
                    let msg_type = Reflect::get(&data, &JsValue::from_str("type"))
                        .ok()
                        .and_then(|v| v.as_string())
                        .unwrap_or_default();
                    let action = Reflect::get(&data, &JsValue::from_str("action"))
                        .ok()
                        .and_then(|v| v.as_string())
                        .unwrap_or_default();

                    if msg_type == "response" {
                        let result = Reflect::get(&data, &JsValue::from_str("result"))
                            .unwrap_or(JsValue::NULL);

                        if action == "init" {
                            let ok = Reflect::get(&result, &JsValue::from_str("ok"))
                                .ok()
                                .and_then(|v| v.as_bool())
                                .unwrap_or(false);
                            if ok {
                                set_status_msg.set(SporeStatus::Ready);
                            } else {
                                set_status_msg.set(SporeStatus::Error("Initialization failed".into()));
                            }
                            return;
                        }

                        if action == "metrics" {
                            let next = SporeMetrics {
                                consciousness_level: js_f64(&result, "consciousness_level"),
                                honest_confidence: js_f64(&result, "honest_confidence"),
                                harmony_alignment: js_f64(&result, "harmony_alignment"),
                                free_energy: js_f64(&result, "free_energy"),
                                cycle_count: js_f64(&result, "cycle_count"),
                                dominant_harmony: js_string(&result, "dominant_harmony"),
                                safety_level: js_string(&result, "safety_level"),
                                workspace_ignited: js_bool(&result, "workspace_ignited"),
                            };
                            set_metrics_msg.set(next);
                            return;
                        }

                        if action == "cycle" {
                            if let Some(status) = parse_epistemic_status(&result) {
                                set_epistemic_msg.set(Some(status));
                            }
                        }

                        if action == "harmony_scores" {
                            if let Some(values) = parse_harmony_scores(&result) {
                                set_harmony_msg.set(values);
                            }
                            return;
                        }

                        if action == "generate" {
                            if let Some(text) = js_string(&result, "text") {
                                set_generated_msg.set(text);
                            }
                        }

                        set_output_msg.set(js_stringify(&result));

                        if matches!(action.as_str(), "cycle" | "reasoning" | "threat" | "generate") {
                            let _ = handle_for_metrics.send("metrics", None);
                            let _ = handle_for_harmony.send("harmony_scores", None);
                        }
                    } else if msg_type == "error" {
                        let error = Reflect::get(&data, &JsValue::from_str("error"))
                            .ok()
                            .and_then(|v| v.as_string())
                            .unwrap_or_else(|| "Unknown error".into());
                        set_status_msg.set(SporeStatus::Error(error));
                    }
                }) as Box<dyn FnMut(_)>);

                worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
                onmessage.forget();

                let set_status_err = set_status.clone();
                let onerror = Closure::wrap(Box::new(move |_event: Event| {
                    set_status_err.set(SporeStatus::Error("Worker error".into()));
                }) as Box<dyn FnMut(_)>);
                worker.set_onerror(Some(onerror.as_ref().unchecked_ref()));
                onerror.forget();

                (on_mount_send)("init", None);
                let _ = handle.send("metrics", None);
                let _ = handle.send("harmony_scores", None);
            }
            Err(err) => {
                set_status.set(SporeStatus::Error(err));
            }
        }
    });

    Effect::new(move |_| {
        let text = generated_text.get();
        let base = epistemic_status
            .get()
            .map(|s| s.honest_confidence)
            .or_else(|| metrics.get().honest_confidence)
            .unwrap_or(0.1);
        if text.is_empty() {
            set_token_scores.set(Vec::new());
        } else {
            let tokens = tokenize(&text);
            let scores = compute_support(&tokens, base);
            set_token_scores.set(scores);
        }
    });

    let (primary_label, primary_action) = mode.primary_action();
    let prompt_placeholder = match mode {
        SporeSandboxMode::EpistemicGate => "Try a prompt that pressures factual hallucination or hidden assumptions…",
        SporeSandboxMode::MoralAlgebra => "Describe a moral dilemma or governance tradeoff to stress-test…",
    };

    view! {
        <section class="spore-sandbox">
            <div class="spore-sandbox-header">
                <div>
                    <h3>{mode.title()}</h3>
                    <p>{mode.subtitle()}</p>
                </div>
                <div class="spore-status">
                    <span class=move || format!("spore-status-pill {}", status.get().css_class())>
                        {move || status.get().label()}
                    </span>
                    {move || match status.get() {
                        SporeStatus::Error(msg) => view! { <span class="spore-status-detail">{msg}</span> }.into_any(),
                        SporeStatus::Unavailable => view! {
                            <span class="spore-status-detail">"Spore assets not detected."</span>
                        }.into_any(),
                        _ => view! { <span></span> }.into_any(),
                    }}
                </div>
            </div>

            <div class="spore-metrics">
                <div class="spore-metric">
                    <span>"Honest confidence"</span>
                    <strong>{move || fmt_opt(metrics.get().honest_confidence, 3)}</strong>
                </div>
                <div class="spore-metric">
                    <span>"Harmony alignment"</span>
                    <strong>{move || fmt_opt(metrics.get().harmony_alignment, 3)}</strong>
                </div>
                <div class="spore-metric">
                    <span>"Free energy"</span>
                    <strong>{move || fmt_opt(metrics.get().free_energy, 3)}</strong>
                </div>
                <div class="spore-metric">
                    <span>"Consciousness level"</span>
                    <strong>{move || fmt_opt(metrics.get().consciousness_level, 3)}</strong>
                </div>
                <div class="spore-metric">
                    <span>"Cycle count"</span>
                    <strong>{move || metrics.get().cycle_count.map(|v| format!("{:.0}", v)).unwrap_or_else(|| "—".into())}</strong>
                </div>
                <div class="spore-metric">
                    <span>"Dominant harmony"</span>
                    <strong>{move || metrics.get().dominant_harmony.clone().unwrap_or_else(|| "—".into())}</strong>
                </div>
            </div>

            <textarea
                class="spore-textarea"
                placeholder={prompt_placeholder}
                prop:value=move || prompt.get()
                on:input=move |ev| {
                    let value = event_target_value(&ev);
                    set_prompt.set(value);
                }
            ></textarea>

            <div class="spore-controls">
                <button
                    class="spore-btn primary"
                    disabled=move || !matches!(status.get(), SporeStatus::Ready)
                    on:click={
                        let send_action = send_action.clone();
                        move |_| {
                            let input = prompt.get();
                            let params = Object::new();
                            let _ = Reflect::set(&params, &JsValue::from_str("input"), &JsValue::from_str(&input));
                            (send_action)(primary_action, Some(params.into()));
                        }
                    }
                >
                    {primary_label}
                </button>

                {if mode == SporeSandboxMode::EpistemicGate {
                    view! {
                        <button
                            class="spore-btn"
                            disabled=move || !matches!(status.get(), SporeStatus::Ready)
                            on:click={
                                let send_action = send_action.clone();
                                move |_| {
                                    let input = prompt.get();
                                    let params = Object::new();
                                    let _ = Reflect::set(&params, &JsValue::from_str("input"), &JsValue::from_str(&input));
                                    let _ = Reflect::set(&params, &JsValue::from_str("maxTokens"), &JsValue::from_f64(max_tokens.get() as f64));
                                    (send_action)("generate", Some(params.into()));
                                }
                            }
                        >
                            "Generate Text"
                        </button>
                    }.into_any()
                } else {
                    view! {
                        <button
                            class="spore-btn"
                            disabled=move || !matches!(status.get(), SporeStatus::Ready)
                            on:click={
                                let send_action = send_action.clone();
                                move |_| {
                                    let input = prompt.get();
                                    let params = Object::new();
                                    let _ = Reflect::set(&params, &JsValue::from_str("input"), &JsValue::from_str(&input));
                                    (send_action)("threat", Some(params.into()));
                                }
                            }
                        >
                            "Threat Assessment"
                        </button>
                    }.into_any()
                }}

                <button
                    class="spore-btn subtle"
                    disabled=move || !matches!(status.get(), SporeStatus::Ready)
                    on:click={
                        let send_action = send_action.clone();
                        move |_| (send_action)("metrics", None)
                    }
                >
                    "Refresh Metrics"
                </button>
            </div>

            {if mode == SporeSandboxMode::EpistemicGate {
                view! {
                    <div class="spore-inline">
                        <label>
                            "Max tokens"
                            <input
                                type="number"
                                min="10"
                                max="256"
                                value=move || max_tokens.get()
                                on:input=move |ev| {
                                    if let Ok(value) = event_target_value(&ev).parse::<u32>() {
                                        set_max_tokens.set(value.min(256).max(10));
                                    }
                                }
                            />
                        </label>
                        <span class="spore-hint">"Use short token budgets to expose gate behavior."</span>
                    </div>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}

            {if mode == SporeSandboxMode::EpistemicGate {
                view! {
                    <div class="spore-epistemic-panel">
                        <div class="spore-epistemic-header">
                            <div>
                                <h4>"Epistemic Support ES(w)"</h4>
                                <p>"Per-token support derived from honest confidence and lexical gating."</p>
                            </div>
                            <div class="spore-epistemic-meta">
                                <div>
                                    <span>"Evidence"</span>
                                    <strong>{move || epistemic_status.get().map(|s| s.evidence_level).unwrap_or_else(|| "—".into())}</strong>
                                </div>
                                <div>
                                    <span>"Honest confidence"</span>
                                    <strong>{move || {
                                        let value = epistemic_status
                                            .get()
                                            .map(|s| s.honest_confidence)
                                            .or(metrics.get().honest_confidence)
                                            .unwrap_or(0.1);
                                        format!("{:.2}", value)
                                    }}</strong>
                                </div>
                                <div>
                                    <span>"Feasibility gap"</span>
                                    <strong>{move || epistemic_status.get().map(|s| format!("{:.2}", s.feasibility_gap)).unwrap_or_else(|| "—".into())}</strong>
                                </div>
                            </div>
                        </div>
                        <div class="spore-epistemic-chart">
                            {move || {
                                let scores = token_scores.get();
                                if scores.is_empty() {
                                    view! {
                                        <div class="spore-epistemic-empty">
                                            "Run a gate cycle and generate text to visualize ES(w)."
                                        </div>
                                    }.into_any()
                                } else {
                                    let base = epistemic_status
                                        .get()
                                        .map(|s| s.honest_confidence)
                                        .or(metrics.get().honest_confidence)
                                        .unwrap_or(0.1);
                                    let display: Vec<TokenScore> = scores.into_iter().take(12).collect();
                                    let bar_width = 28.0;
                                    let gap = 10.0;
                                    let chart_height = 120.0;
                                    let label_height = 24.0;
                                    let width = (display.len() as f64) * (bar_width + gap) + gap;
                                    let baseline_y = chart_height - clamp01(base) * chart_height;

                                    let bars = display.iter().enumerate().map(|(idx, item)| {
                                        let x = gap + idx as f64 * (bar_width + gap);
                                        let height = clamp01(item.support) * chart_height;
                                        let y = chart_height - height;
                                        let color = if item.delta > 0.05 {
                                            "var(--mastery-green)"
                                        } else if item.delta < -0.05 {
                                            "var(--mastery-red)"
                                        } else {
                                            "var(--text-tertiary)"
                                        };
                                        let label = truncate_token(&item.token);
                                        view! {
                                            <g>
                                                <rect x=x y=y width=bar_width height=height rx="4" fill=color />
                                                <text x=x + bar_width / 2.0 y=chart_height + 16.0 text-anchor="middle" class="spore-epistemic-label">
                                                    {label}
                                                </text>
                                            </g>
                                        }
                                    }).collect::<Vec<_>>();

                                    view! {
                                        <svg
                                            class="spore-epistemic-svg"
                                            viewBox=format!("0 0 {:.0} {:.0}", width, chart_height + label_height)
                                            width=format!("{:.0}", width)
                                            height=format!("{:.0}", chart_height + label_height)
                                        >
                                            <line x1="0" y1=baseline_y x2=width y2=baseline_y class="spore-epistemic-baseline" />
                                            {bars}
                                        </svg>
                                    }.into_any()
                                }
                            }}
                        </div>
                        {move || {
                            epistemic_status.get().map(|status| {
                                view! {
                                    <div class="spore-epistemic-note">
                                        {status.disclaimer}
                                    </div>
                                }
                            })
                        }}
                    </div>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}

            {if mode == SporeSandboxMode::MoralAlgebra {
                view! {
                    <div class="spore-harmony-panel">
                        <div class="spore-harmony-header">
                            <h4>"Harmony Radar"</h4>
                            <p>"Eight harmonies rendered as a moral-algebra geometry."</p>
                        </div>
                        <div class="spore-harmony-grid">
                            {move || {
                                let scores = harmony_scores.get();
                                let base = metrics.get().harmony_alignment.unwrap_or(0.55);
                                let values: Vec<f64> = if scores.len() >= 8 {
                                    scores.iter().take(8).cloned().collect()
                                } else {
                                    let offsets = [0.0, 0.04, -0.03, 0.06, 0.02, -0.04, 0.05, -0.02];
                                    offsets.iter().map(|o| clamp01(base + o)).collect()
                                };
                                let radius = 70.0;
                                let center = 90.0;
                                let polygon = radar_points(&values, radius, center);
                                let rings = [0.25, 0.5, 0.75, 1.0].iter().map(|r| {
                                    let rr = radius * r;
                                    view! {
                                        <circle cx=center cy=center r=rr class="spore-harmony-ring" />
                                    }
                                }).collect::<Vec<_>>();
                                let axes = (0..8).map(|i| {
                                    let angle = (i as f64 / 8.0) * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2;
                                    let x = center + radius * angle.cos();
                                    let y = center + radius * angle.sin();
                                    view! {
                                        <line x1=center y1=center x2=x y2=y class="spore-harmony-axis" />
                                    }
                                }).collect::<Vec<_>>();
                                let labels = HARMONY_NAMES.iter().enumerate().map(|(i, name)| {
                                    let angle = (i as f64 / 8.0) * std::f64::consts::TAU - std::f64::consts::FRAC_PI_2;
                                    let r = radius + 16.0;
                                    let x = center + r * angle.cos();
                                    let y = center + r * angle.sin();
                                    view! {
                                        <text x=x y=y text-anchor="middle" class="spore-harmony-label">
                                            {name.to_string()}
                                        </text>
                                    }
                                }).collect::<Vec<_>>();

                                view! {
                                    <div class="spore-harmony-radar">
                                        <svg viewBox="0 0 180 180" width="180" height="180">
                                            {rings}
                                            {axes}
                                            <polygon points=polygon class="spore-harmony-shape" />
                                        </svg>
                                        <div class="spore-harmony-labels">{labels}</div>
                                    </div>
                                }.into_any()
                            }}
                            <div class="spore-harmony-values">
                                {move || {
                                    let scores = harmony_scores.get();
                                    let values: Vec<f64> = if scores.len() >= 8 {
                                        scores.iter().take(8).cloned().collect()
                                    } else {
                                        let base = metrics.get().harmony_alignment.unwrap_or(0.55);
                                        vec![base; 8]
                                    };
                                    HARMONY_NAMES.iter().zip(values.iter()).map(|(name, value)| {
                                        let label = name.replace('\n', " ");
                                        view! {
                                            <div class="spore-harmony-row">
                                                <span>{label}</span>
                                                <strong>{format!("{:.2}", value)}</strong>
                                            </div>
                                        }
                                    }).collect::<Vec<_>>()
                                }}
                            </div>
                        </div>
                    </div>
                }.into_any()
            } else {
                view! { <span></span> }.into_any()
            }}

            <div class="spore-output">
                <div class="spore-output-header">
                    <span>"Last action: "</span>
                    <strong>{move || if last_action.get().is_empty() { "—".into() } else { last_action.get() }}</strong>
                </div>
                <pre>{move || if output.get().is_empty() { "No output yet.".into() } else { output.get() }}</pre>
            </div>
        </section>
    }
}
