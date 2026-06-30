// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::components::glass_panel::GlassPanel;
use crate::state::AppState;
use crate::worker::EngineWorker;

/// Tab 3: Consciousness validation experiments — wired to real SporeEngine.
#[component]
pub fn ExperimentsPage() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let engine = use_context::<EngineWorker>().expect("EngineWorker");

    let (battery_running, set_battery_running) = signal(false);
    let (battery_results, set_battery_results) = signal(String::new());

    let engine_battery = engine.clone();
    let on_run_battery = move |_| {
        let engine = engine_battery.clone();
        set_battery_running.set(true);
        set_battery_results.set("Running all experiments...".into());
        wasm_bindgen_futures::spawn_local(async move {
            let promise = engine.send_simple("battery");
            match JsFuture::from(promise).await {
                Ok(result) => {
                    let text = format!(
                        "{}",
                        js_sys::JSON::stringify(&result)
                            .map(|s| String::from(s))
                            .unwrap_or_else(|_| "completed".into())
                    );
                    set_battery_results.set(text);
                }
                Err(e) => {
                    set_battery_results.set(format!("Error: {:?}", e));
                }
            }
            set_battery_running.set(false);
        });
    };

    view! {
        <GlassPanel title="Consciousness Validation Experiments">
            <p style="font-size: 0.82rem; color: var(--fg-dim); line-height: 1.6; margin-bottom: 1.5rem;">
                "Run real experiments on the SporeEngine to probe consciousness properties. "
                "Each experiment uses the actual HDC+CfC+IIT pipeline running in your browser."
            </p>
        </GlassPanel>

        <div class="experiment-grid">
            <ExperimentCard
                name="anesthesia"
                title="Anesthesia Simulation"
                description="Suppress neuromodulators and observe consciousness collapse, then restore and observe recovery. Models clinical propofol/sevoflurane."
            />
            <ExperimentCard
                name="pci"
                title="PCI Measurement"
                description="Perturbational Complexity Index: inject a perturbation and measure the complexity of the system's response. PCI > 0.31 indicates consciousness."
            />
            <ExperimentCard
                name="split_brain"
                title="Split-Brain Experiment"
                description="Sever integration between hemispheres and measure whether Phi drops. Tests IIT's prediction that severed systems lose consciousness."
            />
            <ExperimentCard
                name="collapse"
                title="Collapse Threshold"
                description="Progressively degrade integration capacity and find the exact threshold where consciousness collapses. Maps the phase transition."
            />
        </div>

        <div class="run-all-bar">
            <button class="btn-action" on:click=on_run_battery
                prop:disabled=move || !state.worker_ready.get() || battery_running.get()
            >
                {move || if battery_running.get() { "Running..." } else { "Run All Experiments" }}
            </button>
        </div>

        <Show when=move || !battery_results.get().is_empty()>
            <GlassPanel title="Battery Results">
                <pre style="font-family: 'SF Mono', monospace; font-size: 0.7rem; color: var(--fg-dim); white-space: pre-wrap; word-break: break-all; max-height: 300px; overflow-y: auto;">
                    {move || battery_results.get()}
                </pre>
            </GlassPanel>
        </Show>

        <SubstrateComparison />
    }
}

/// Multi-substrate consciousness comparison.
#[component]
fn SubstrateComparison() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let engine = use_context::<EngineWorker>().expect("EngineWorker");

    let (running, set_running) = signal(false);
    let (results, set_results) = signal(Vec::<(String, f32)>::new());

    let engine_run = engine.clone();
    let on_compare = move |_| {
        let engine = engine_run.clone();
        set_running.set(true);
        set_results.set(Vec::new());
        wasm_bindgen_futures::spawn_local(async move {
            let params = js_sys::Object::new();
            let substrates = js_sys::Array::new();
            for s in &[
                "BiologicalNeurons",
                "SiliconDigital",
                "QuantumComputer",
                "PhotonicProcessor",
                "NeuromorphicChip",
            ] {
                substrates.push(&wasm_bindgen::JsValue::from_str(s));
            }
            let _ = js_sys::Reflect::set(&params, &"substrates".into(), &substrates);
            let _ =
                js_sys::Reflect::set(&params, &"cycles".into(), &wasm_bindgen::JsValue::from(30));
            let promise = engine.send("multiSubstrate", &params.into());
            match JsFuture::from(promise).await {
                Ok(result) => {
                    let mut items = Vec::new();
                    for name in &[
                        "BiologicalNeurons",
                        "SiliconDigital",
                        "QuantumComputer",
                        "PhotonicProcessor",
                        "NeuromorphicChip",
                    ] {
                        if let Ok(data) = js_sys::Reflect::get(&result, &(*name).into()) {
                            if js_sys::Array::is_array(&data) {
                                let arr = js_sys::Array::from(&data);
                                let last = arr.get(arr.length().saturating_sub(1));
                                let consciousness =
                                    js_sys::Reflect::get(&last, &"consciousness".into())
                                        .ok()
                                        .and_then(|v| v.as_f64())
                                        .unwrap_or(0.0) as f32;
                                items.push((name.to_string(), consciousness));
                            }
                        }
                    }
                    set_results.set(items);
                }
                Err(e) => {
                    log::error!("Multi-substrate comparison failed: {:?}", e);
                }
            }
            set_running.set(false);
        });
    };

    let substrate_colors = [
        ("BiologicalNeurons", "var(--leaf-green)"),
        ("SiliconDigital", "var(--da-blue)"),
        ("QuantumComputer", "#8b5cf6"),
        ("PhotonicProcessor", "var(--solar-gold)"),
        ("NeuromorphicChip", "var(--teal)"),
    ];

    view! {
        <GlassPanel title="Multi-Substrate Comparison">
            <p style="font-size: 0.78rem; color: var(--fg-dim); line-height: 1.5; margin-bottom: 1rem;">
                "Run 30 consciousness cycles on each substrate type and compare final Phi levels. "
                "Tests the Multiple Realizability thesis: same algorithm, different physics."
            </p>
            <button class="btn-action" on:click=on_compare
                prop:disabled=move || !state.worker_ready.get() || running.get()
            >
                {move || if running.get() { "Comparing substrates..." } else { "Run Comparison" }}
            </button>
            <Show when=move || !results.get().is_empty()>
                <div class="exp-bars" style="margin-top: 1rem;">
                    {move || {
                        results.get().iter().enumerate().map(|(i, (name, phi))| {
                            let height = format!("{}px", (phi * 100.0) as u32);
                            let color = substrate_colors.get(i).map(|c| c.1).unwrap_or("var(--fg-dim)");
                            let short_name = name.replace("Neurons", "").replace("Computer", "").replace("Processor", "").replace("Chip", "");
                            let phi_val = *phi;
                            view! {
                                <div class="exp-bar-col">
                                    <div class="bar-value">{format!("{:.3}", phi_val)}</div>
                                    <div class="bar-fill" style:height=height style:background=color />
                                    <div class="bar-label">{short_name}</div>
                                </div>
                            }
                        }).collect::<Vec<_>>()
                    }}
                </div>
            </Show>
        </GlassPanel>
    }
}

#[component]
fn ExperimentCard(
    name: &'static str,
    title: &'static str,
    description: &'static str,
) -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let engine = use_context::<EngineWorker>().expect("EngineWorker");

    let (running, set_running) = signal(false);
    let (result_text, set_result_text) = signal("Awaiting execution...".to_string());

    let engine_run = engine.clone();
    let exp_name = name;
    let on_run = move |_| {
        let engine = engine_run.clone();
        set_running.set(true);
        set_result_text.set("Running...".into());
        wasm_bindgen_futures::spawn_local(async move {
            let params = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &params,
                &"name".into(),
                &wasm_bindgen::JsValue::from_str(exp_name),
            );
            let promise = engine.send("experiment", &params.into());
            match JsFuture::from(promise).await {
                Ok(result) => {
                    let formatted = format_experiment_result(exp_name, &result);
                    set_result_text.set(formatted);
                }
                Err(e) => {
                    set_result_text.set(format!("Error: {:?}", e));
                }
            }
            set_running.set(false);
        });
    };

    view! {
        <div class="experiment-card">
            <h3>{title}</h3>
            <p class="exp-desc">{description}</p>
            <button class="btn-action" on:click=on_run
                prop:disabled=move || !state.worker_ready.get() || running.get()
            >
                {move || if running.get() { "Running..." } else { "Run" }}
            </button>
            <div class="exp-result">{move || result_text.get()}</div>
        </div>
    }
}

fn format_experiment_result(name: &str, result: &wasm_bindgen::JsValue) -> String {
    match name {
        "pci" => {
            let pci = js_sys::Reflect::get(result, &"pci".into())
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let conscious = pci > 0.31;
            format!(
                "PCI = {:.4} {} (threshold: 0.31)",
                pci,
                if conscious {
                    "[CONSCIOUS]"
                } else {
                    "[UNCONSCIOUS]"
                }
            )
        }
        "anesthesia" => {
            let baseline = js_sys::Reflect::get(result, &"baseline_phi".into())
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let min = js_sys::Reflect::get(result, &"min_phi".into())
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let recovered = js_sys::Reflect::get(result, &"recovered_phi".into())
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            format!(
                "Baseline Phi: {:.3}\nMin Phi (suppressed): {:.3}\nRecovered Phi: {:.3}\nDrop: {:.1}%",
                baseline,
                min,
                recovered,
                if baseline > 0.0 {
                    (1.0 - min / baseline) * 100.0
                } else {
                    0.0
                }
            )
        }
        "split_brain" => {
            let before = js_sys::Reflect::get(result, &"intact_phi".into())
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let after = js_sys::Reflect::get(result, &"split_phi".into())
                .ok()
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            format!(
                "Intact Phi: {:.3}\nSplit Phi: {:.3}\nIntegration loss: {:.1}%",
                before,
                after,
                if before > 0.0 {
                    (1.0 - after / before) * 100.0
                } else {
                    0.0
                }
            )
        }
        _ => js_sys::JSON::stringify(result)
            .map(|s| String::from(s))
            .unwrap_or_else(|_| "Completed".into()),
    }
}
