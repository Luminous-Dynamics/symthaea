// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Boltzmann Distribution Explorer — P(E) ∝ exp(-E/kT)
//!
//! Students see how temperature controls energy level populations.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn BoltzmannExplorer(node_id: String) -> impl IntoView {
    let k_b = 1.381e-23_f64; // J/K
    let k_b_ev = 8.617e-5_f64; // eV/K

    let (temperature, set_temperature) = signal(300.0_f64); // K

    // Energy levels (0 to 5 kT units)
    let populations = move || {
        let t = temperature.get().max(1.0);
        let kt = k_b_ev * t; // kT in eV
        (0..10)
            .map(|i| {
                let e = i as f64 * 0.1; // Energy in eV
                let p = (-e / kt).exp();
                (e, p)
            })
            .collect::<Vec<_>>()
    };

    let kt_ev = Memo::new(move |_| k_b_ev * temperature.get());
    let avg_energy = Memo::new(move |_| 1.5 * k_b_ev * temperature.get()); // 3/2 kT for 3D

    view! {
        <div style="padding: 1rem;">
            <h2 style="font-size: 1.5rem; margin-bottom: 0.5rem; color: var(--accent-color, #6366f1);">"Boltzmann Distribution"</h2>
            <p style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 1rem;">
                <code style="color: #10b981;">"P(E) ∝ exp(-E/kT)"</code>
                " — Higher temperature = more particles at high energy"
            </p>

            <div style="display: flex; gap: 0.375rem; margin-bottom: 0.5rem; flex-wrap: wrap;">
                {vec![("Liquid N₂", 77.0), ("Room", 300.0), ("Boiling", 373.0), ("Lava", 1500.0), ("Sun surface", 5778.0), ("Fusion", 1.5e8)].into_iter().map(|(name, t)| view! {
                    <button
                        style="padding: 0.25rem 0.5rem; background: #1f2937; color: #e2e8f0; border: 1px solid #374151; border-radius: 0.25rem; cursor: pointer; font-size: 0.65rem;"
                        on:click=move |_| set_temperature.set(t)
                    >{name}</button>
                }).collect::<Vec<_>>()}
            </div>

            <div style="margin-bottom: 1rem;">
                <label style="font-size: 0.8rem; color: #94a3b8;">"Temperature: " {move || format!("{:.0} K", temperature.get())} " (" {move || format!("{:.0} °C", temperature.get() - 273.15)} ")"</label>
                <input type="range" min="1" max="8" step="0.01" style="width: 100%;"
                    prop:value=move || temperature.get().log10().to_string()
                    on:input=move |ev| { let t: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into(); set_temperature.set(10.0_f64.powf(t.value().parse().unwrap_or(2.5))); }
                />
            </div>

            // Population bar chart
            <div style="display: flex; align-items: end; gap: 4px; height: 120px; padding: 0 1rem; margin-bottom: 1rem;">
                {move || populations().into_iter().map(|(e, p)| {
                    let h = (p * 110.0).min(110.0).max(2.0);
                    let color = if p > 0.5 { "#6366f1" } else if p > 0.1 { "#3b82f6" } else { "#1e3a5f" };
                    view! {
                        <div style="flex: 1; display: flex; flex-direction: column; align-items: center;">
                            <div style=format!("width: 100%; height: {}px; background: {}; border-radius: 2px 2px 0 0; min-height: 2px;", h, color)
                                title=format!("E={:.2} eV, P={:.3}", e, p)
                            ></div>
                            <div style="font-size: 0.55rem; color: #64748b; margin-top: 2px;">{format!("{:.1}", e)}</div>
                        </div>
                    }
                }).collect::<Vec<_>>()}
            </div>
            <div style="text-align: center; font-size: 0.7rem; color: #64748b; margin-bottom: 1rem;">"Energy (eV) →"</div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
                <div style="padding: 0.75rem; background: #111827; border: 1px solid rgba(99,102,241,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.7rem; color: #94a3b8;">"kT"</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #6366f1;">{move || format!("{:.4} eV", kt_ev.get())}</div>
                </div>
                <div style="padding: 0.75rem; background: #111827; border: 1px solid rgba(16,185,129,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.7rem; color: #94a3b8;">"Average Energy (3D)"</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #10b981;">{move || format!("{:.4} eV", avg_energy.get())}</div>
                </div>
            </div>
        </div>
    }
}
