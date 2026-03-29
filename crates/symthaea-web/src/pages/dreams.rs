// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::components::glass_panel::GlassPanel;
use crate::state::AppState;
use crate::worker::EngineWorker;

/// Tab 4: Dream engine — counterfactual learning + FEP exploration.
#[component]
pub fn DreamsPage() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let engine = use_context::<EngineWorker>().expect("EngineWorker");

    let (dream_running, set_dream_running) = signal(false);
    let (dream_cycles_input, set_dream_cycles_input) = signal(5_u32);
    let (dream_log, set_dream_log) = signal(Vec::<String>::new());
    let (consolidations, set_consolidations) = signal(0_u32);

    // Fetch initial FEP/dream stats when worker is ready
    let engine_stats = engine.clone();
    let state_stats = state.clone();
    Effect::new(move |_| {
        if !state_stats.worker_ready.get() {
            return;
        }
        let engine = engine_stats.clone();
        let state = state_stats.clone();
        wasm_bindgen_futures::spawn_local(async move {
            // Get dream stats
            let promise = engine.send_simple("dreamStats");
            if let Ok(result) = JsFuture::from(promise).await {
                if let Ok(dc) = js_sys::Reflect::get(&result, &"dream_cycles".into()) {
                    state.dream_cycles.set(dc.as_f64().unwrap_or(0.0) as u32);
                }
                if let Ok(wc) = js_sys::Reflect::get(&result, &"wisdom_count".into()) {
                    state.wisdom_count.set(wc.as_f64().unwrap_or(0.0) as u32);
                }
            }

            // Get FEP state
            let promise = engine.send_simple("freeEnergy");
            if let Ok(result) = JsFuture::from(promise).await {
                if let Some(fe) = result.as_f64() {
                    state.free_energy.set(fe as f32);
                } else if let Ok(fe) = js_sys::Reflect::get(&result, &"free_energy".into()) {
                    state.free_energy.set(fe.as_f64().unwrap_or(0.0) as f32);
                }
            }
        });
    });

    let engine_dream = engine.clone();
    let state_dream = state.clone();
    let on_dream = move |_| {
        let engine = engine_dream.clone();
        let state = state_dream.clone();
        let cycles = dream_cycles_input.get();
        set_dream_running.set(true);
        set_dream_log
            .update(|log| log.push(format!("Starting dream session ({cycles} cycles)...")));

        wasm_bindgen_futures::spawn_local(async move {
            let params = js_sys::Object::new();
            let _ = js_sys::Reflect::set(
                &params,
                &"cycles".into(),
                &wasm_bindgen::JsValue::from(cycles),
            );
            let promise = engine.send("dreamSession", &params.into());
            match JsFuture::from(promise).await {
                Ok(result) => {
                    // Parse dream session results
                    if let Ok(wc) = js_sys::Reflect::get(&result, &"wisdom_gained".into()) {
                        let w = wc.as_f64().unwrap_or(0.0) as u32;
                        state.wisdom_count.update(|c| *c += w);
                        set_dream_log.update(|log| log.push(format!("Gained {w} wisdom entries")));
                    }
                    if let Ok(pe) = js_sys::Reflect::get(&result, &"final_prediction_error".into())
                    {
                        state
                            .prediction_error
                            .set(pe.as_f64().unwrap_or(0.0) as f32);
                    }
                    if let Ok(cons) = js_sys::Reflect::get(&result, &"consolidations".into()) {
                        set_consolidations.set(cons.as_f64().unwrap_or(0.0) as u32);
                    }
                    state.dream_cycles.update(|c| *c += cycles);
                    set_dream_log.update(|log| log.push("Dream session complete.".into()));
                }
                Err(e) => {
                    set_dream_log.update(|log| log.push(format!("Error: {:?}", e)));
                }
            }

            // Refresh FEP state
            let promise = engine.send_simple("freeEnergy");
            if let Ok(result) = JsFuture::from(promise).await {
                if let Some(fe) = result.as_f64() {
                    state.free_energy.set(fe as f32);
                } else if let Ok(fe) = js_sys::Reflect::get(&result, &"free_energy".into()) {
                    state.free_energy.set(fe.as_f64().unwrap_or(0.0) as f32);
                }
            }

            set_dream_running.set(false);
        });
    };

    view! {
        <GlassPanel title="Dream Engine">
            <p style="font-size: 0.82rem; color: var(--fg-dim); line-height: 1.6; margin-bottom: 1rem;">
                "Counterfactual learning through dreaming: the system replays experiences, "
                "perturbs them, and consolidates wisdom. FEP drives explore/exploit balance."
            </p>

            <div class="dream-controls">
                <span style="font-size: 0.8rem; color: var(--fg-dim);">"Cycles:"</span>
                <input
                    type="number"
                    min="1" max="50"
                    prop:value=move || dream_cycles_input.get().to_string()
                    on:input=move |ev| {
                        if let Ok(v) = event_target_value(&ev).parse::<u32>() {
                            set_dream_cycles_input.set(v.clamp(1, 50));
                        }
                    }
                />
                <button class="btn-action" on:click=on_dream
                    prop:disabled=move || !state.worker_ready.get() || dream_running.get()
                >
                    {move || if dream_running.get() { "Dreaming..." } else { "Start Dream Session" }}
                </button>
            </div>

            <div class="dream-stats-grid">
                <div class="dream-stat">
                    <div class="ds-val">{move || state.dream_cycles.get().to_string()}</div>
                    <div class="ds-label">"dream cycles"</div>
                </div>
                <div class="dream-stat">
                    <div class="ds-val">{move || state.wisdom_count.get().to_string()}</div>
                    <div class="ds-label">"wisdom entries"</div>
                </div>
                <div class="dream-stat">
                    <div class="ds-val">
                        {move || format!("{:.3}", state.prediction_error.get())}
                    </div>
                    <div class="ds-label">"prediction error"</div>
                </div>
                <div class="dream-stat">
                    <div class="ds-val">{move || consolidations.get().to_string()}</div>
                    <div class="ds-label">"consolidations"</div>
                </div>
            </div>
        </GlassPanel>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
            <GlassPanel title="FEP State">
                <div class="fep-display" style="grid-template-columns: 1fr;">
                    <div class="fep-big">
                        <div class="fep-val">
                            {move || format!("{:.3}", state.free_energy.get())}
                        </div>
                        <div class="fep-label">"free energy"</div>
                        <div class=move || {
                            if state.free_energy.get() > 0.5 { "fep-mode exploring" } else { "fep-mode exploiting" }
                        }>
                            {move || if state.free_energy.get() > 0.5 { "exploring" } else { "exploiting" }}
                        </div>
                    </div>
                </div>
            </GlassPanel>

            <GlassPanel title="Dream Journal">
                <div class="wisdom-journal">
                    <Show when=move || dream_log.get().is_empty()>
                        <div class="wisdom-empty">
                            "No dream sessions yet. Start one to begin counterfactual learning."
                        </div>
                    </Show>
                    <For
                        each=move || dream_log.get()
                        key=|entry| entry.clone()
                        children=move |entry: String| {
                            view! {
                                <div class="wisdom-entry">
                                    <div class="w-context">{entry}</div>
                                </div>
                            }
                        }
                    />
                </div>
            </GlassPanel>
        </div>
    }
}
