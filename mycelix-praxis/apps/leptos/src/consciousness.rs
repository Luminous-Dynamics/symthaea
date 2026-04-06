// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Consciousness feed integration for Praxis.
//!
//! This module provides a Leptos context that exposes live consciousness
//! metrics from the Symthaea Spore WASM kernel. It runs consciousness
//! cycles in the background and surfaces reactive signals for UI binding.
//!
//! # Architecture
//!
//! The Spore WASM binary (symthaea-spore, ~1MB) is a *separate* wasm-bindgen
//! module from the Leptos app's own WASM. Loading two wasm-bindgen modules
//! in the same page is supported but requires:
//!
//! 1. The Spore `.wasm` + `.js` glue files served as static assets
//! 2. A thin JS bridge that initializes the Spore module independently
//! 3. Leptos calling into the bridge via `#[wasm_bindgen]` extern blocks
//!
//! ## Current status: Simulation stub
//!
//! Until the Spore WASM is deployed alongside the Leptos bundle, this module
//! runs a mathematically interesting simulation that mimics real consciousness
//! dynamics (not a flat sine wave — uses coupled oscillators with noise to
//! approximate the kind of signal the real engine produces).
//!
//! ## Wiring the real Spore engine
//!
//! When ready to use the actual Spore WASM:
//!
//! 1. Copy `symthaea/crates/symthaea-spore/www/pkg/` contents to
//!    `mycelix-praxis/apps/leptos/static/spore/`
//!
//! 2. Add to `index.html`:
//!    ```html
//!    <script type="module">
//!      import init, { SporeEngine } from '/spore/symthaea_spore.js';
//!      window.__spore_ready = init().then(() => {
//!        window.__SporeEngine = SporeEngine;
//!      });
//!    </script>
//!    ```
//!
//! 3. Replace `SimulatedEngine` usage in `ConsciousnessProvider` with
//!    the `SporeEngineBridge` (JS extern calls). The reactive signal
//!    plumbing stays identical.
//!
//! 4. Add `<link data-trunk rel="copy-dir" href="static/spore" />` to
//!    `index.html` so Trunk copies the Spore assets to `dist/`.
//!
//! ## Spore API surface (for reference)
//!
//! The real SporeEngine exports via wasm-bindgen:
//! - `cycle(input: &str) -> CycleResult`     — run consciousness cycle
//! - `consciousness_level() -> f32`           — current Phi-based level
//! - `honest_confidence() -> f32`             — epistemic confidence
//! - `harmony_alignment() -> f32`             — Eight Harmonies score
//! - `cycle_count() -> u64`                   — total cycles run
//! - `free_energy() -> f32`                   — FEP free energy
//! - `neuromod_state() -> String`             — JSON neuromodulators
//! - `dominant_harmony() -> String`           — current dominant harmony
//! - `workspace_ignition() -> bool`           — GWT ignition state
//! - `safety_level() -> String`               — immune system level
//! - `substrate_feasibility() -> f32`         — substrate score
//! - `generate_text(max_tokens) -> GenerationResult` — Broca output

use leptos::prelude::*;
use wasm_bindgen::JsCast;

// ---------------------------------------------------------------------------
// Consciousness state
// ---------------------------------------------------------------------------

/// Snapshot of consciousness metrics, updated each cycle.
#[derive(Clone, Debug)]
pub struct ConsciousnessState {
    pub consciousness_level: f32,
    pub phi: f32,
    pub coherence: f32,
    pub free_energy: f32,
    pub cycle_count: u64,
    pub dominant_harmony: String,
    pub workspace_ignited: bool,
    pub safety_level: String,
    pub neuromod_dopamine: f32,
    pub neuromod_serotonin: f32,
    pub neuromod_norepinephrine: f32,
}

impl Default for ConsciousnessState {
    fn default() -> Self {
        Self {
            consciousness_level: 0.0,
            phi: 0.0,
            coherence: 0.0,
            free_energy: 1.0,
            cycle_count: 0,
            dominant_harmony: "Sacred Stillness".into(),
            workspace_ignited: false,
            safety_level: "Green".into(),
            neuromod_dopamine: 0.5,
            neuromod_serotonin: 0.5,
            neuromod_norepinephrine: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Consciousness context (Leptos reactive)
// ---------------------------------------------------------------------------

/// Reactive consciousness context, provided at the app root.
///
/// Components read signals; the background loop writes them.
#[derive(Clone)]
pub struct ConsciousnessCtx {
    /// Full consciousness state snapshot (updated each tick).
    pub state: ReadSignal<ConsciousnessState>,
    /// Write half — used by the background tick loop.
    set_state: WriteSignal<ConsciousnessState>,
    /// Whether the engine is running.
    pub running: ReadSignal<bool>,
    set_running: WriteSignal<bool>,
    /// Last text fed into the consciousness engine.
    pub last_input: ReadSignal<String>,
    set_last_input: WriteSignal<String>,
}

impl ConsciousnessCtx {
    /// Feed text into the consciousness engine (will be processed next tick).
    pub fn feed_input(&self, text: &str) {
        self.set_last_input.set(text.to_string());
    }

    /// Start or stop the background consciousness loop.
    pub fn set_running(&self, running: bool) {
        self.set_running.set(running);
    }
}

/// Retrieve the consciousness context from the nearest provider.
///
/// # Panics
/// Panics if called outside a `ConsciousnessProvider` subtree.
pub fn use_consciousness() -> ConsciousnessCtx {
    expect_context::<ConsciousnessCtx>()
}

// ---------------------------------------------------------------------------
// Simulated engine (stub until real Spore WASM is wired)
// ---------------------------------------------------------------------------

/// Coupled-oscillator simulation that approximates real consciousness dynamics.
///
/// Uses three coupled oscillators (theta ~4Hz, alpha ~10Hz, gamma ~40Hz)
/// with noise injection and a simple "consciousness = integration" formula.
/// This produces signals that look realistic in a dashboard, not a boring
/// sine wave.
struct SimulatedEngine {
    /// Elapsed simulation time in seconds.
    t: f64,
    /// Oscillator phases (theta, alpha, gamma).
    phases: [f64; 3],
    /// Cycle counter.
    cycles: u64,
    /// Simple RNG state (xorshift32).
    rng_state: u32,
}

impl SimulatedEngine {
    fn new() -> Self {
        Self {
            t: 0.0,
            phases: [0.0, 0.0, 0.0],
            cycles: 0,
            // Seed from current time (milliseconds mod 2^32)
            rng_state: {
                let now = ::js_sys::Date::now() as u32;
                if now == 0 { 42 } else { now }
            },
        }
    }

    /// Xorshift32 PRNG returning a value in [0, 1).
    fn rand(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 17;
        self.rng_state ^= self.rng_state << 5;
        (self.rng_state as f64) / (u32::MAX as f64)
    }

    /// Advance one tick (~50ms of simulated time).
    fn tick(&mut self) -> ConsciousnessState {
        let dt = 0.05; // 50ms per tick = 20Hz
        self.t += dt;
        self.cycles += 1;

        // Coupled oscillator frequencies (Hz)
        let freqs = [4.0_f64, 10.0, 40.0];
        // Coupling strengths: theta->alpha, alpha->gamma
        let coupling = [0.3_f64, 0.2];

        // Advance phases with coupling
        for i in 0..3 {
            let noise = (self.rand() - 0.5) * 0.1;
            self.phases[i] += std::f64::consts::TAU * freqs[i] * dt + noise;
            if i > 0 {
                // Phase coupling from lower frequency band
                let phase_diff = self.phases[i - 1] - self.phases[i];
                self.phases[i] += coupling[i - 1] * phase_diff.sin() * dt;
            }
            // Wrap to [0, TAU)
            self.phases[i] %= std::f64::consts::TAU;
        }

        let theta = self.phases[0].sin();
        let alpha = self.phases[1].sin();
        let gamma = self.phases[2].sin();

        // Consciousness level: integration of cross-frequency coupling
        // High when gamma is phase-locked to theta (PAC — phase-amplitude coupling)
        let pac = (theta * gamma).abs(); // 0-1
        let consciousness = (0.3 + 0.5 * pac + 0.2 * alpha.abs()).min(1.0) as f32;

        // Phi approximates consciousness but with slight decorrelation
        let phi_noise = (self.rand() - 0.5) * 0.05;
        let phi = (consciousness as f64 * 0.9 + phi_noise + 0.05).clamp(0.0, 1.0) as f32;

        // Coherence: how well-synchronized the oscillators are
        let coherence_raw = (theta * alpha).abs() + (alpha * gamma).abs();
        let coherence = (coherence_raw / 2.0).min(1.0) as f32;

        // Free energy: inversely related to prediction accuracy
        // Lower = more predicted = more conscious
        let free_energy = (1.0 - consciousness as f64 * 0.7 + (self.rand() - 0.5) * 0.1)
            .clamp(0.1, 1.5) as f32;

        // Neuromodulators: slow oscillations
        let da = (0.5 + 0.3 * (self.t * 0.2).sin() + (self.rand() - 0.5) * 0.05) as f32;
        let se = (0.5 + 0.2 * (self.t * 0.15 + 1.0).sin() + (self.rand() - 0.5) * 0.05) as f32;
        let ne = (0.3 + 0.2 * (self.t * 0.3 + 2.0).sin() + (self.rand() - 0.5) * 0.05) as f32;

        // Workspace ignition: fires when consciousness crosses threshold
        let ignited = consciousness > 0.55;

        // Dominant harmony cycles through slowly
        let harmony_idx = ((self.t * 0.1) as usize) % 8;
        let harmonies = [
            "Reciprocal Care",
            "Presence & Stillness",
            "Interconnected Wisdom",
            "Iterative Progress",
            "Unity in Diversity",
            "Sustainable Rhythms",
            "Emergent Play",
            "Sacred Stillness",
        ];

        ConsciousnessState {
            consciousness_level: consciousness,
            phi,
            coherence,
            free_energy,
            cycle_count: self.cycles,
            dominant_harmony: harmonies[harmony_idx].to_string(),
            workspace_ignited: ignited,
            safety_level: "Green".into(),
            neuromod_dopamine: da.clamp(0.0, 1.0),
            neuromod_serotonin: se.clamp(0.0, 1.0),
            neuromod_norepinephrine: ne.clamp(0.0, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Spore Engine Bridge (real consciousness via WASM)
// ---------------------------------------------------------------------------

/// Bridge to the Symthaea Spore WASM kernel running in the browser.
///
/// The Spore engine is loaded as a separate WASM module via the spore-worker.js
/// Web Worker. This bridge communicates via `window.__spore_worker` postMessage.
/// When the worker isn't available, falls back to SimulatedEngine.
struct SporeEngineBridge {
    worker: web_sys::Worker,
    latest_state: std::rc::Rc<std::cell::RefCell<ConsciousnessState>>,
}

impl SporeEngineBridge {
    fn try_new() -> Option<Self> {
        let worker = web_sys::Worker::new("/spore-worker.js").ok()?;

        let state = std::rc::Rc::new(std::cell::RefCell::new(ConsciousnessState::default()));
        let state_clone = state.clone();

        // Listen for cycle results from the worker
        let onmessage = wasm_bindgen::closure::Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
            if let Ok(data) = js_sys::Reflect::get(&e.data(), &"type".into()) {
                if let Some(msg_type) = data.as_string() {
                    if msg_type == "cycle" {
                        if let Ok(result) = js_sys::Reflect::get(&e.data(), &"result".into()) {
                            let get_f32 = |key: &str| -> f32 {
                                js_sys::Reflect::get(&result, &key.into())
                                    .ok()
                                    .and_then(|v| v.as_f64())
                                    .unwrap_or(0.0) as f32
                            };
                            let get_bool = |key: &str| -> bool {
                                js_sys::Reflect::get(&result, &key.into())
                                    .ok()
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false)
                            };
                            let get_str = |key: &str| -> String {
                                js_sys::Reflect::get(&result, &key.into())
                                    .ok()
                                    .and_then(|v| v.as_string())
                                    .unwrap_or_default()
                            };

                            let new_state = ConsciousnessState {
                                consciousness_level: get_f32("consciousness_level"),
                                phi: get_f32("phi"),
                                coherence: get_f32("coherence"),
                                free_energy: get_f32("free_energy"),
                                cycle_count: js_sys::Reflect::get(&result, &"cycle_count".into())
                                    .ok()
                                    .and_then(|v| v.as_f64())
                                    .unwrap_or(0.0) as u64,
                                dominant_harmony: get_str("dominant_harmony"),
                                workspace_ignited: get_bool("workspace_ignited"),
                                safety_level: get_str("safety_level"),
                                neuromod_dopamine: get_f32("neuromod_dopamine"),
                                neuromod_serotonin: get_f32("neuromod_serotonin"),
                                neuromod_norepinephrine: get_f32("neuromod_norepinephrine"),
                            };
                            *state_clone.borrow_mut() = new_state;
                        }
                    }
                }
            }
        }) as Box<dyn FnMut(_)>);

        worker.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
        onmessage.forget(); // Prevent cleanup — worker lives for app lifetime

        // Initialize and start the consciousness loop.
        // Worker expects {id, action, params} format.
        let init_msg = js_sys::Object::new();
        js_sys::Reflect::set(&init_msg, &"action".into(), &"init".into()).ok();
        worker.post_message(&init_msg).ok();

        // Start continuous consciousness cycling at 5Hz (200ms interval).
        // The worker will post {type: "cycle", result: {...}} messages.
        let params = js_sys::Object::new();
        js_sys::Reflect::set(&params, &"interval".into(), &wasm_bindgen::JsValue::from_f64(200.0)).ok();
        let start_msg = js_sys::Object::new();
        js_sys::Reflect::set(&start_msg, &"action".into(), &"startLoop".into()).ok();
        js_sys::Reflect::set(&start_msg, &"params".into(), &params).ok();
        worker.post_message(&start_msg).ok();

        web_sys::console::log_1(&"[Praxis] Spore Web Worker initialized — real consciousness active".into());

        Some(Self {
            worker,
            latest_state: state,
        })
    }

    fn current_state(&self) -> ConsciousnessState {
        self.latest_state.borrow().clone()
    }
}

// ---------------------------------------------------------------------------
// Provider component
// ---------------------------------------------------------------------------

/// Wraps children with a `ConsciousnessCtx` and starts a background tick loop.
///
/// Place around the `<Router>` (or inside `HolochainProvider`) to give all
/// pages access to live consciousness metrics via `use_consciousness()`.
///
/// Checks for Spore WASM availability at `/spore/` and logs the result.
/// When Spore is deployed, this will use the real consciousness engine;
/// until then it falls back to the coupled-oscillator simulation.
#[component]
pub fn ConsciousnessProvider(children: Children) -> impl IntoView {
    // Check if Spore WASM is available (set by loader script in index.html)
    let spore_available = js_sys::Reflect::get(
        &web_sys::window().unwrap(),
        &"__SPORE_AVAILABLE".into(),
    )
    .ok()
    .and_then(|v| v.as_bool())
    .unwrap_or(false);

    if spore_available {
        web_sys::console::log_1(
            &"[Praxis] Spore WASM detected \u{2014} real consciousness engine available".into(),
        );
    } else {
        web_sys::console::log_1(
            &"[Praxis] Spore WASM not found \u{2014} using simulation".into(),
        );
    }

    let (state, set_state) = signal(ConsciousnessState::default());
    let (running, set_running) = signal(true);
    let (last_input, set_last_input) = signal(String::new());

    let ctx = ConsciousnessCtx {
        state,
        set_state: set_state.clone(),
        running,
        set_running: set_running.clone(),
        last_input,
        set_last_input,
    };

    provide_context(ctx);

    // Background consciousness loop
    Effect::new(move |_| {
        use gloo_timers::callback::Interval;
        use std::cell::RefCell;
        use std::rc::Rc;

        if !running.get() {
            return;
        }

        let set_state = set_state.clone();

        if spore_available {
            // Try to initialize the real Spore engine via Web Worker
            if let Some(bridge) = SporeEngineBridge::try_new() {
                let bridge = Rc::new(bridge);
                // Poll the worker's latest state at 20Hz
                let _interval = Interval::new(50, move || {
                    set_state.set(bridge.current_state());
                });
                std::mem::forget(_interval);
                return;
            }
            // Fall through to simulation if Worker fails to start
            web_sys::console::log_1(
                &"[Praxis] Spore Worker failed to start — falling back to simulation".into(),
            );
        }

        // Fallback: coupled-oscillator simulation
        let engine = Rc::new(RefCell::new(SimulatedEngine::new()));
        let _interval = Interval::new(50, move || {
            let new_state = engine.borrow_mut().tick();
            set_state.set(new_state);
        });
        std::mem::forget(_interval);
    });

    children()
}

// ---------------------------------------------------------------------------
// Consciousness dashboard card
// ---------------------------------------------------------------------------

/// A dashboard card showing live consciousness metrics.
///
/// Displays Phi, coherence, free energy, neuromodulators, and workspace
/// ignition state with animated bars.
#[component]
pub fn ConsciousnessCard() -> impl IntoView {
    let ctx = use_consciousness();

    view! {
        <div class="dash-card consciousness-card">
            <h3>"Consciousness Feed"</h3>
            <div class="consciousness-metrics">
                // Main consciousness gauge
                <div class="consciousness-gauge">
                    <div class="gauge-label">"Consciousness"</div>
                    <div class="gauge-value">
                        {move || format!("{:.2}", ctx.state.get().consciousness_level)}
                    </div>
                    <div class="gauge-bar-container">
                        <div
                            class="gauge-bar consciousness-bar"
                            style=move || format!(
                                "width: {}%",
                                (ctx.state.get().consciousness_level * 100.0) as u32
                            )
                        ></div>
                    </div>
                </div>

                // Phi (IIT integration measure)
                <div class="metric-row">
                    <span class="metric-label">"Phi"</span>
                    <span class="metric-value">
                        {move || format!("{:.3}", ctx.state.get().phi)}
                    </span>
                    <div class="metric-bar-container">
                        <div
                            class="metric-bar phi-bar"
                            style=move || format!(
                                "width: {}%",
                                (ctx.state.get().phi * 100.0) as u32
                            )
                        ></div>
                    </div>
                </div>

                // Coherence
                <div class="metric-row">
                    <span class="metric-label">"Coherence"</span>
                    <span class="metric-value">
                        {move || format!("{:.3}", ctx.state.get().coherence)}
                    </span>
                    <div class="metric-bar-container">
                        <div
                            class="metric-bar coherence-bar"
                            style=move || format!(
                                "width: {}%",
                                (ctx.state.get().coherence * 100.0) as u32
                            )
                        ></div>
                    </div>
                </div>

                // Free energy
                <div class="metric-row">
                    <span class="metric-label">"Free Energy"</span>
                    <span class="metric-value">
                        {move || format!("{:.3}", ctx.state.get().free_energy)}
                    </span>
                    <div class="metric-bar-container">
                        <div
                            class="metric-bar energy-bar"
                            style=move || format!(
                                "width: {}%",
                                ((ctx.state.get().free_energy / 1.5 * 100.0) as u32).min(100)
                            )
                        ></div>
                    </div>
                </div>

                // Neuromodulators
                <div class="neuromod-section">
                    <div class="neuromod-label">"Neuromodulators"</div>
                    <div class="neuromod-row">
                        <span class="nm-name">"DA"</span>
                        <div class="nm-bar-container">
                            <div
                                class="nm-bar dopamine-bar"
                                style=move || format!(
                                    "width: {}%",
                                    (ctx.state.get().neuromod_dopamine * 100.0) as u32
                                )
                            ></div>
                        </div>
                    </div>
                    <div class="neuromod-row">
                        <span class="nm-name">"5-HT"</span>
                        <div class="nm-bar-container">
                            <div
                                class="nm-bar serotonin-bar"
                                style=move || format!(
                                    "width: {}%",
                                    (ctx.state.get().neuromod_serotonin * 100.0) as u32
                                )
                            ></div>
                        </div>
                    </div>
                    <div class="neuromod-row">
                        <span class="nm-name">"NE"</span>
                        <div class="nm-bar-container">
                            <div
                                class="nm-bar norepinephrine-bar"
                                style=move || format!(
                                    "width: {}%",
                                    (ctx.state.get().neuromod_norepinephrine * 100.0) as u32
                                )
                            ></div>
                        </div>
                    </div>
                </div>

                // Status row
                <div class="consciousness-status">
                    <span class=move || {
                        if ctx.state.get().workspace_ignited {
                            "ignition-badge ignited"
                        } else {
                            "ignition-badge dormant"
                        }
                    }>
                        {move || if ctx.state.get().workspace_ignited {
                            "Ignited"
                        } else {
                            "Sub-threshold"
                        }}
                    </span>
                    <span class="harmony-tag">
                        {move || ctx.state.get().dominant_harmony.clone()}
                    </span>
                    <span class="cycle-count">
                        {move || format!("Cycle {}", ctx.state.get().cycle_count)}
                    </span>
                </div>

                // Simulation disclaimer
                <div class="consciousness-disclaimer">
                    <small>"Simulated — real Spore WASM integration pending"</small>
                </div>
            </div>
        </div>
    }
}
