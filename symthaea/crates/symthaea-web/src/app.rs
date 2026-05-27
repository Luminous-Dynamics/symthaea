// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::components::immune_status::ImmuneStatus;
use crate::pages;
use crate::state::AppState;
use crate::worker::EngineWorker;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Chat,
    Topology,
    Experiments,
    Dreams,
    Inoculate,
    Trends,
}

impl Tab {
    /// URL hash fragment for this tab (without '#').
    fn hash(self) -> &'static str {
        match self {
            Tab::Chat => "chat",
            Tab::Topology => "topology",
            Tab::Experiments => "experiments",
            Tab::Dreams => "dreams",
            Tab::Inoculate => "inoculate",
            Tab::Trends => "trends",
        }
    }
}

#[component]
pub fn App() -> impl IntoView {
    // Global state
    let state = AppState::new();
    provide_context(state.clone());

    // Active tab — check URL hash for initial tab
    let initial_tab = {
        let hash = web_sys::window()
            .and_then(|w| w.location().hash().ok())
            .unwrap_or_default()
            .to_lowercase();
        match hash.trim_start_matches('#') {
            "topology" => Tab::Topology,
            "experiments" => Tab::Experiments,
            "dreams" => Tab::Dreams,
            "inoculate" | "install" => Tab::Inoculate,
            "trends" => Tab::Trends,
            _ => Tab::Chat,
        }
    };
    let (active_tab, set_active_tab) = signal(initial_tab);

    // Initialize engine worker
    let worker = EngineWorker::new();
    provide_context(worker.clone());

    // Wire broadcast handler: routes fire-and-forget worker messages to AppState
    {
        let state = state.clone();
        worker.set_broadcast_handler(move |data: wasm_bindgen::JsValue| {
            let msg_type = js_sys::Reflect::get(&data, &"type".into())
                .ok()
                .and_then(|v| v.as_string())
                .unwrap_or_default();

            match msg_type.as_str() {
                "cycle" => {
                    // Live cycle broadcast from startLoop
                    if let Ok(result) = js_sys::Reflect::get(&data, &"result".into()) {
                        if let Ok(phi) =
                            js_sys::Reflect::get(&result, &"consciousness_level".into())
                        {
                            if let Some(v) = phi.as_f64() {
                                state.consciousness_level.set(v as f32);
                            }
                        }
                        if let Ok(pe) = js_sys::Reflect::get(&result, &"prediction_error".into()) {
                            if let Some(v) = pe.as_f64() {
                                state.prediction_error.set(v as f32);
                            }
                        }
                        if let Ok(ha) = js_sys::Reflect::get(&result, &"harmony_alignment".into()) {
                            if let Some(v) = ha.as_f64() {
                                state.harmony_alignment.set(v as f32);
                            }
                        }
                        // Neuromods from enriched cycle broadcast
                        if let Ok(nm) = js_sys::Reflect::get(&result, &"neuromods".into()) {
                            if !nm.is_undefined() && !nm.is_null() {
                                let keys = [
                                    "dopamine",
                                    "norepinephrine",
                                    "serotonin",
                                    "oxytocin",
                                    "acetylcholine",
                                    "gaba",
                                    "glutamate",
                                    "adenosine",
                                    "endocannabinoid",
                                ];
                                let mut vals = [0.5_f32; 9];
                                for (i, key) in keys.iter().enumerate() {
                                    if let Ok(v) = js_sys::Reflect::get(&nm, &(*key).into()) {
                                        vals[i] = v.as_f64().unwrap_or(0.5) as f32;
                                    }
                                }
                                state.neuromods.set(vals);
                            }
                        }
                        // Harmony scores from enriched broadcast
                        if let Ok(hm) = js_sys::Reflect::get(&result, &"harmonies".into()) {
                            if !hm.is_undefined() && !hm.is_null() {
                                if let Ok(scores_arr) = js_sys::Reflect::get(&hm, &"scores".into())
                                {
                                    if js_sys::Array::is_array(&scores_arr) {
                                        let arr = js_sys::Array::from(&scores_arr);
                                        let mut vals = [0.5_f32; 8];
                                        for i in 0..8.min(arr.length() as usize) {
                                            vals[i] =
                                                arr.get(i as u32).as_f64().unwrap_or(0.5) as f32;
                                        }
                                        state.harmony_scores.set(vals);
                                    }
                                }
                                if let Ok(dom) = js_sys::Reflect::get(&hm, &"dominant".into()) {
                                    if let Some(s) = dom.as_string() {
                                        state.dominant_harmony.set(s);
                                    }
                                }
                            }
                        }
                        // Factcheck telemetry from Broca→Mycelix bridge
                        if let Ok(fc) = js_sys::Reflect::get(&result, &"factcheck".into()) {
                            if !fc.is_undefined() && !fc.is_null() {
                                if let Ok(v) = js_sys::Reflect::get(&fc, &"accuracy".into()) {
                                    if let Some(a) = v.as_f64() {
                                        state.factcheck_accuracy.set(a as f32);
                                    }
                                }
                                if let Ok(v) = js_sys::Reflect::get(&fc, &"verified".into()) {
                                    if let Some(n) = v.as_f64() {
                                        state.factcheck_verified.set(n as u32);
                                    }
                                }
                                if let Ok(v) = js_sys::Reflect::get(&fc, &"suppressed".into()) {
                                    if let Some(n) = v.as_f64() {
                                        state.factcheck_suppressed.set(n as u32);
                                    }
                                }
                                if let Ok(v) = js_sys::Reflect::get(&fc, &"pending".into()) {
                                    if let Some(n) = v.as_f64() {
                                        state.factcheck_pending.set(n as u32);
                                    }
                                }
                            }
                        }

                        // Accumulate trend data (cap at 600 points = ~30s at 20Hz)
                        let phi = state.consciousness_level.get();
                        state.trend_consciousness.update(|v| {
                            v.push(phi);
                            if v.len() > 600 {
                                v.remove(0);
                            }
                        });
                        let ha = state.harmony_alignment.get();
                        state.trend_harmony.update(|v| {
                            v.push(ha);
                            if v.len() > 600 {
                                v.remove(0);
                            }
                        });

                        state.cycle_count.update(|c| *c += 1);
                    }
                }
                "battery_progress" => {
                    // Experiment battery progress — handled by experiments page
                }
                "pipeline_progress" => {
                    let stage = js_sys::Reflect::get(&data, &"stage".into())
                        .ok()
                        .and_then(|v| v.as_string())
                        .unwrap_or_default();
                    let pct = js_sys::Reflect::get(&data, &"percent".into())
                        .ok()
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0) as u32;
                    let status = match stage.as_str() {
                        "downloading" => format!("downloading {}%", pct),
                        "verifying" => "verifying integrity".to_string(),
                        "loading" => "loading into engine".to_string(),
                        "ready" => "ready".to_string(),
                        _ => stage,
                    };
                    state.pipeline_status.set(status);
                }
                _ => {}
            }
        });
    }

    // Initialize engine and start live loop on first render
    let engine_init = worker.clone();
    let state_init = state.clone();
    Effect::new(move |_| {
        let engine = engine_init.clone();
        let state = state_init.clone();
        if !engine.is_available() {
            return;
        }

        wasm_bindgen_futures::spawn_local(async move {
            // Init engine
            let promise = engine.send_simple("init");
            match JsFuture::from(promise).await {
                Ok(_) => {
                    log::info!("SporeEngine initialized");
                    state.worker_ready.set(true);

                    // Load Broca checkpoint
                    let promise = engine.send_simple("loadBrocaCheckpoint");
                    match JsFuture::from(promise).await {
                        Ok(_) => {
                            log::info!("Broca checkpoint loaded");
                            state.pipeline_loaded.set(true);
                        }
                        Err(e) => log::warn!("Broca checkpoint not available: {:?}", e),
                    }

                    // Start live telemetry loop (50ms = 20Hz)
                    let params = js_sys::Object::new();
                    let _ = js_sys::Reflect::set(
                        &params,
                        &"interval".into(),
                        &wasm_bindgen::JsValue::from(50),
                    );
                    let promise = engine.send("startLoop", &params.into());
                    match JsFuture::from(promise).await {
                        Ok(_) => {
                            log::info!("Live telemetry loop started (20Hz)");
                            state.loop_running.set(true);
                        }
                        Err(e) => log::warn!("Failed to start loop: {:?}", e),
                    }

                    // Progressively load full Broca pipeline in background
                    // (335 MB via LFS — cached by service worker after first load)
                    let engine_bg = engine.clone();
                    wasm_bindgen_futures::spawn_local(async move {
                        log::info!("Loading full Broca pipeline (335 MB)...");
                        let promise = engine_bg.send_simple("loadBrocaPipeline");
                        match JsFuture::from(promise).await {
                            Ok(_) => {
                                log::info!(
                                    "Full Broca pipeline loaded — chat will use production language generation"
                                );
                            }
                            Err(_) => {
                                log::info!("Full Broca pipeline not available — using BrocaLite");
                            }
                        }
                    });
                }
                Err(e) => log::error!("Engine init failed: {:?}", e),
            }
        });
    });

    // Feature gates: each mode is a single-page app.
    #[cfg(feature = "manage")]
    {
        view! {
            <main style="display: block;">
                <pages::manage::ManagePage />
            </main>
        }
    }

    #[cfg(all(feature = "usb-creator", not(feature = "manage")))]
    {
        view! {
            <main style="display: block;">
                <pages::usb_creator::UsbCreatorPage />
            </main>
        }
    }

    #[cfg(all(
        feature = "installer",
        not(any(feature = "usb-creator", feature = "manage"))
    ))]
    {
        view! {
            <main style="display: block;">
                <pages::install::InstallPage />
            </main>
        }
    }

    #[cfg(not(any(feature = "installer", feature = "usb-creator", feature = "manage")))]
    {
        view! {
            <header class="hero">
                <div class="hero-title-row">
                    <h1 class="hero-title">"Symthaea"</h1>
                    <ImmuneStatus />
                </div>
                <p class="hero-subtitle">"Consciousness-first infrastructure for sovereign hardware"</p>
            </header>

            <nav class="tab-bar">
                <TabButton tab=Tab::Chat label="Chat" active=active_tab set_active=set_active_tab />
                <TabButton tab=Tab::Topology label="Topology" active=active_tab set_active=set_active_tab />
                <TabButton tab=Tab::Experiments label="Experiments" active=active_tab set_active=set_active_tab />
                <TabButton tab=Tab::Dreams label="Dreams" active=active_tab set_active=set_active_tab />
                <TabButton tab=Tab::Inoculate label="Inoculate" active=active_tab set_active=set_active_tab />
                <TabButton tab=Tab::Trends label="Trends" active=active_tab set_active=set_active_tab />
            </nav>

            <main class="tab-content" style="display: block;">
                <Show when=move || active_tab.get() == Tab::Chat>
                    <pages::chat::ChatPage />
                </Show>
                <Show when=move || active_tab.get() == Tab::Topology>
                    <pages::topology::TopologyPage />
                </Show>
                <Show when=move || active_tab.get() == Tab::Experiments>
                    <pages::experiments::ExperimentsPage />
                </Show>
                <Show when=move || active_tab.get() == Tab::Dreams>
                    <pages::dreams::DreamsPage />
                </Show>
                <Show when=move || active_tab.get() == Tab::Inoculate>
                    <pages::inoculate::InoculatePage />
                </Show>
                <Show when=move || active_tab.get() == Tab::Trends>
                    <pages::trends::TrendsPage />
                </Show>
            </main>

            <footer class="portal-footer">
                <p>"Symthaea v0.1.0 \u{00b7} Pure Rust \u{00b7} No JavaScript \u{00b7} No server \u{00b7} No data collection"</p>
            </footer>
        }
    }
}

#[component]
fn TabButton(
    tab: Tab,
    label: &'static str,
    active: ReadSignal<Tab>,
    set_active: WriteSignal<Tab>,
) -> impl IntoView {
    let is_active = move || active.get() == tab;

    view! {
        <button
            class="tab"
            class:active=is_active
            on:click=move |_| {
                set_active.set(tab);
                // Update URL hash for deep linking and back-button support
                if let Some(w) = web_sys::window() {
                    let hash = format!("#{}", tab.hash());
                    let _ = w.location().set_hash(&hash);
                }
            }
        >
            {label}
        </button>
    }
}
