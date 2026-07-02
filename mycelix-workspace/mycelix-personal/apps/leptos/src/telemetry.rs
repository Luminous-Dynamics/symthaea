// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use mycelix_leptos_core::TelemetryLine;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeMetricsSnapshot {
    pub total_success: u64,
    pub total_errors: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstellationVitals {
    pub identity: Option<BridgeMetricsSnapshot>,
    pub finance: Option<BridgeMetricsSnapshot>,
    pub civic: Option<BridgeMetricsSnapshot>,
    pub knowledge: Option<BridgeMetricsSnapshot>,
    pub symthaea: Option<BridgeMetricsSnapshot>,
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "__TAURI__", "event"])]
    async fn listen(event: &str, handler: &js_sys::Function) -> JsValue;
}

#[component]
pub fn ConstellationTelemetry() -> impl IntoView {
    let cultural = crate::context::use_cultural();
    let symbols = cultural.symbols;

    let identity_history = RwSignal::new(vec![1.0; 10]);
    let finance_history = RwSignal::new(vec![1.0; 10]);
    let civic_history = RwSignal::new(vec![1.0; 10]);
    let knowledge_history = RwSignal::new(vec![1.0; 10]);
    let symthaea_history = RwSignal::new(vec![1.0; 10]);

    // THERMODYNAMIC IGNITION (Vector 2)
    let genesis_pulse = RwSignal::new(false);
    let genesis_text = RwSignal::new(String::new());

    // Use a resource to subscribe to Tauri events
    #[cfg(feature = "hydrate")]
    {
        use wasm_bindgen::closure::Closure;

        spawn_local(async move {
            let callback = Closure::wrap(Box::new(move |event: JsValue| {
                let vitals: ConstellationVitals = serde_wasm_bindgen::from_value(
                    js_sys::Reflect::get(&event, &"payload".into()).unwrap(),
                )
                .unwrap();

                // Detect Thermodynamic Genesis
                if let Some(ref m) = vitals.finance {
                    if m.total_success > 100 {
                        // Threshold for pulse
                        genesis_pulse.set(true);
                        let s = symbols.get_untracked();
                        genesis_text.set(format!(
                            "{}. 10 SAP flowed into {}.",
                            s.genesis_alias, s.hearth_alias
                        ));
                        // [Audio chime would play here]
                        set_timeout(move || genesis_pulse.set(false), 5000);
                    }
                }

                let update_history =
                    |history: RwSignal<Vec<f64>>, metrics: Option<BridgeMetricsSnapshot>| {
                        let mut current = history.get_untracked();
                        current.remove(0);
                        let health = if let Some(m) = metrics {
                            let total = m.total_success + m.total_errors;
                            if total > 0 {
                                m.total_success as f64 / total as f64
                            } else {
                                1.0
                            }
                        } else {
                            0.0 // Substrate offline
                        };
                        current.push(health);
                        history.set(current);
                    };

                update_history(identity_history, vitals.identity);
                update_history(finance_history, vitals.finance);
                update_history(civic_history, vitals.civic);
                update_history(knowledge_history, vitals.knowledge);
                update_history(symthaea_history, vitals.symthaea);
            }) as Box<dyn FnMut(JsValue)>);

            listen("constellation-vitals", callback.as_ref().unchecked_ref()).await;
            callback.forget();
        });
    }

    view! {
        // GOLDEN PULSE OVERLAY (Vector 2)
        <Show when=move || genesis_pulse.get()>
            <div style="position: fixed; inset: 0; pointer-events: none; z-index: 1000;
                        background: radial-gradient(circle, rgba(255,215,0,0.15) 0%, transparent 70%); 
                        animation: pulse-gold 3s ease-out;">
                <div style="position: absolute; bottom: 20%; width: 100%; text-align: center;
                            color: var(--md-signal); font-family: var(--md-mono); font-size: 1.2rem;
                            text-shadow: 0 0 20px rgba(255,215,0,0.5);">
                    {move || genesis_text.get()}
                </div>
            </div>
        </Show>

        <section class="vault-card" style="margin-top: 2rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem;">
                <h3 style="margin: 0; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: var(--md-fg-muted);">
                    {move || format!("{} Liveness", symbols.get().hearth_alias)}
                </h3>
                <span style="font-size: 0.75rem; color: var(--md-signal); font-family: var(--md-mono);">
                    {move || symbols.get().orientation.clone()}
                </span>
            </div>

            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                <TelemetryLine
                    label="Identity"
                    values=identity_history.into()
                    unit="%"
                    min=0.0
                    max=1.0
                />
                <TelemetryLine
                    label="Finance"
                    values=finance_history.into()
                    unit="%"
                    min=0.0
                    max=1.0
                />
                <TelemetryLine
                    label="Civic"
                    values=civic_history.into()
                    unit="%"
                    min=0.0
                    max=1.0
                />
                <TelemetryLine
                    label="Knowledge"
                    values=knowledge_history.into()
                    unit="%"
                    min=0.0
                    max=1.0
                />
                <TelemetryLine
                    label="Symthaea (Moral)"
                    values=symthaea_history.into()
                    unit="%"
                    min=0.0
                    max=1.0
                />
            </div>

            <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid var(--md-divider, rgba(255,255,255,0.05));">
                <div style="display: flex; gap: 1.5rem; font-size: 0.7rem; color: var(--md-fg-muted); font-family: var(--md-mono);">
                    <div style="display: flex; align-items: center; gap: 0.4rem;">
                        <div style="width: 6px; height: 6px; border-radius: 50%; background: var(--md-signal);"></div>
                        "Active coordination"
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.4rem;">
                        <div style="width: 6px; height: 6px; border-radius: 50%; background: var(--md-fg-muted); opacity: 0.3;"></div>
                        {move || format!("Standby {}", symbols.get().hearth_alias)}
                    </div>
                </div>
            </div>
        </section>
    }
}
