// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ideal Gas Law Explorer — PV = nRT
//!
//! Students adjust P, V, n, T and see the equation balance live.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn IdealGasExplorer(node_id: String) -> impl IntoView {
    let r_gas = 8.314_f64; // J/(mol·K)

    let (pressure, set_pressure) = signal(101325.0_f64);  // Pa (1 atm)
    let (volume, set_volume) = signal(0.0224_f64);          // m³ (22.4 L at STP)
    let (moles, set_moles) = signal(1.0_f64);
    let (temperature, set_temperature) = signal(273.15_f64); // K (0°C)
    let (locked, set_locked) = signal("T".to_string()); // Which variable to compute

    // Computed values based on which is locked
    let computed_t = Memo::new(move |_| pressure.get() * volume.get() / (moles.get() * r_gas));
    let computed_p = Memo::new(move |_| moles.get() * r_gas * temperature.get() / volume.get());
    let computed_v = Memo::new(move |_| moles.get() * r_gas * temperature.get() / pressure.get());

    // PV and nRT for balance display
    let pv = Memo::new(move |_| pressure.get() * volume.get());
    let nrt = Memo::new(move |_| moles.get() * r_gas * temperature.get());
    let balance = Memo::new(move |_| {
        let diff = (pv.get() - nrt.get()).abs();
        let avg = (pv.get() + nrt.get()) / 2.0;
        if avg < 0.001 { 100.0 } else { (1.0 - diff / avg) * 100.0 }
    });

    view! {
        <div style="padding: 1rem;">
            <h2 style="font-size: 1.5rem; margin-bottom: 0.5rem; color: var(--accent-color, #6366f1);">"Ideal Gas Law"</h2>
            <p style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 1rem;">
                <code style="color: #10b981;">"PV = nRT"</code>
                " — Adjust any three, the fourth is computed"
            </p>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                <div>
                    <label style="font-size: 0.8rem; color: #94a3b8;">"P (Pa): " {move || format!("{:.0}", pressure.get())} " (" {move || format!("{:.2}", pressure.get() / 101325.0)} " atm)"</label>
                    <input type="range" min="1000" max="1000000" step="1000" style="width: 100%;"
                        prop:value=move || pressure.get().to_string()
                        on:input=move |ev| {
                            let target: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into();
                            set_pressure.set(target.value().parse().unwrap_or(101325.0));
                        }
                    />
                </div>
                <div>
                    <label style="font-size: 0.8rem; color: #94a3b8;">"V (L): " {move || format!("{:.1}", volume.get() * 1000.0)}</label>
                    <input type="range" min="0.001" max="0.1" step="0.001" style="width: 100%;"
                        prop:value=move || volume.get().to_string()
                        on:input=move |ev| {
                            let target: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into();
                            set_volume.set(target.value().parse().unwrap_or(0.0224));
                        }
                    />
                </div>
                <div>
                    <label style="font-size: 0.8rem; color: #94a3b8;">"n (mol): " {move || format!("{:.2}", moles.get())}</label>
                    <input type="range" min="0.1" max="10" step="0.1" style="width: 100%;"
                        prop:value=move || moles.get().to_string()
                        on:input=move |ev| {
                            let target: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into();
                            set_moles.set(target.value().parse().unwrap_or(1.0));
                        }
                    />
                </div>
                <div>
                    <label style="font-size: 0.8rem; color: #94a3b8;">"T (K): " {move || format!("{:.0}", temperature.get())} " (" {move || format!("{:.0}", temperature.get() - 273.15)} " °C)"</label>
                    <input type="range" min="100" max="1000" step="1" style="width: 100%;"
                        prop:value=move || temperature.get().to_string()
                        on:input=move |ev| {
                            let target: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into();
                            set_temperature.set(target.value().parse().unwrap_or(273.15));
                        }
                    />
                </div>
            </div>

            // Balance display
            <div style="display: grid; grid-template-columns: 1fr auto 1fr; gap: 1rem; align-items: center; margin-bottom: 1rem;">
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(99,102,241,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94a3b8;">"PV"</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #6366f1;">{move || format!("{:.1} J", pv.get())}</div>
                </div>
                <div style="font-size: 1.5rem; color: #94a3b8;">"="</div>
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(16,185,129,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94a3b8;">"nRT"</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #10b981;">{move || format!("{:.1} J", nrt.get())}</div>
                </div>
            </div>

            <p style="font-size: 0.75rem; color: #64748b; font-style: italic;">
                "R = 8.314 J/(mol·K). At STP (0°C, 1 atm), 1 mol = 22.4 L."
            </p>
        </div>
    }
}
