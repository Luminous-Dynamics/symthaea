// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::components::glass_panel::GlassPanel;
use crate::state::AppState;
use crate::worker::EngineWorker;

/// Visualizes the Global Workspace Theory (GWT) attention competition.
/// Shows ignition events and broadcast strength.
#[component]
pub fn WorkspaceViz() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let engine = use_context::<EngineWorker>().expect("EngineWorker");

    let (ws_contents, set_ws_contents) = signal(Vec::<(String, f32)>::new());

    let engine_ignite = engine.clone();
    let state_ignite = state.clone();
    let on_ignite = move |_| {
        let engine = engine_ignite.clone();
        let state = state_ignite.clone();
        let set_contents = set_ws_contents;
        wasm_bindgen_futures::spawn_local(async move {
            // Trigger ignition
            let promise = engine.send_simple("workspaceIgnition");
            if let Ok(result) = JsFuture::from(promise).await {
                if let Ok(ignited) = js_sys::Reflect::get(&result, &"ignited".into()) {
                    state
                        .workspace_ignited
                        .set(ignited.as_bool().unwrap_or(false));
                }
                if let Ok(strength) = js_sys::Reflect::get(&result, &"broadcast_strength".into()) {
                    state
                        .workspace_broadcast_strength
                        .set(strength.as_f64().unwrap_or(0.0) as f32);
                }
            }

            // Get workspace state
            let promise = engine.send_simple("workspaceState");
            if let Ok(result) = JsFuture::from(promise).await {
                if let Ok(contents) = js_sys::Reflect::get(&result, &"contents".into()) {
                    if js_sys::Array::is_array(&contents) {
                        let arr = js_sys::Array::from(&contents);
                        let mut items = Vec::new();
                        for i in 0..arr.length().min(8) {
                            let item = arr.get(i);
                            let name = js_sys::Reflect::get(&item, &"region".into())
                                .ok()
                                .and_then(|v| v.as_string())
                                .unwrap_or_else(|| format!("region_{i}"));
                            let activation = js_sys::Reflect::get(&item, &"activation".into())
                                .ok()
                                .and_then(|v| v.as_f64())
                                .unwrap_or(0.0) as f32;
                            items.push((name, activation));
                        }
                        set_contents.set(items);
                    }
                }
            }
        });
    };

    view! {
        <GlassPanel title="Global Workspace">
            <div style="display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.8rem;">
                <button class="btn-action" on:click=on_ignite
                    prop:disabled=move || !state.worker_ready.get()
                >
                    "Trigger Ignition"
                </button>
                <span class="ws-status" style:color=move || {
                    if state.workspace_ignited.get() { "var(--solar-gold)" } else { "var(--fg-muted)" }
                }>
                    {move || if state.workspace_ignited.get() { "IGNITED" } else { "dormant" }}
                </span>
                <span class="ws-strength" style="font-family: 'SF Mono', monospace; font-size: 0.7rem; color: var(--fg-dim);">
                    {move || format!("broadcast: {:.2}", state.workspace_broadcast_strength.get())}
                </span>
            </div>
            <div class="ws-contents">
                <Show when=move || ws_contents.get().is_empty()>
                    <div class="ws-empty" style="color: var(--fg-muted); font-size: 0.8rem; font-style: italic;">
                        "Trigger ignition to see workspace competition"
                    </div>
                </Show>
                <For
                    each=move || ws_contents.get()
                    key=|(name, _)| name.clone()
                    children=move |(name, activation): (String, f32)| {
                        let width = format!("{}%", (activation * 100.0) as u32);
                        view! {
                            <div class="neuro-bar-row">
                                <span class="nb-label" style="width: 4.5rem;">{name}</span>
                                <div class="nb-track">
                                    <div class="nb-fill"
                                        style:width=width
                                        style:background="var(--solar-gold)"
                                    />
                                </div>
                                <span class="nb-val">{format!("{:.2}", activation)}</span>
                            </div>
                        }
                    }
                />
            </div>
        </GlassPanel>
    }
}
