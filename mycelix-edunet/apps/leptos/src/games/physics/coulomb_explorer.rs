// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coulomb's Law Explorer — F = kq₁q₂/r²
//!
//! Students compare electrostatic and gravitational forces to see why
//! electricity is 10^36 times stronger than gravity.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn CoulombExplorer(node_id: String) -> impl IntoView {
    let k_e = 8.988e9_f64; // N·m²/C²
    let g = 6.674e-11_f64;
    let e_charge = 1.602e-19_f64;
    let m_proton = 1.673e-27_f64;

    let (q1_e, set_q1_e) = signal(1.0_f64);    // Charge 1 in units of e
    let (q2_e, set_q2_e) = signal(-1.0_f64);   // Charge 2 in units of e
    let (distance_nm, set_distance_nm) = signal(0.053_f64); // Bohr radius in nm

    let f_coulomb = Memo::new(move |_| {
        let r = distance_nm.get() * 1e-9;
        if r < 1e-15 { return 0.0; }
        k_e * q1_e.get() * e_charge * q2_e.get() * e_charge / (r * r)
    });

    let f_gravity = Memo::new(move |_| {
        let r = distance_nm.get() * 1e-9;
        if r < 1e-15 { return 0.0; }
        g * m_proton * m_proton / (r * r)
    });

    let ratio = Memo::new(move |_| {
        let fg = f_gravity.get().abs();
        if fg < 1e-100 { return 0.0; }
        f_coulomb.get().abs() / fg
    });

    view! {
        <div style="padding: 1rem;">
            <h2 style="font-size: 1.5rem; margin-bottom: 0.5rem; color: var(--accent-color, #6366f1);">"Coulomb's Law"</h2>
            <p style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 1rem;">
                <code style="color: #10b981;">"F = kq₁q₂/r²"</code>
                " — Same 1/r² structure as gravity, but 10³⁶× stronger"
            </p>

            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                <div>
                    <label style="font-size: 0.8rem; color: #94a3b8;">"q₁: " {move || format!("{:+.0}e", q1_e.get())}</label>
                    <input type="range" min="-10" max="10" step="1" style="width: 100%;"
                        prop:value=move || q1_e.get().to_string()
                        on:input=move |ev| { let t: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into(); set_q1_e.set(t.value().parse().unwrap_or(1.0)); }
                    />
                </div>
                <div>
                    <label style="font-size: 0.8rem; color: #94a3b8;">"q₂: " {move || format!("{:+.0}e", q2_e.get())}</label>
                    <input type="range" min="-10" max="10" step="1" style="width: 100%;"
                        prop:value=move || q2_e.get().to_string()
                        on:input=move |ev| { let t: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into(); set_q2_e.set(t.value().parse().unwrap_or(-1.0)); }
                    />
                </div>
                <div>
                    <label style="font-size: 0.8rem; color: #94a3b8;">"r: " {move || format!("{:.3} nm", distance_nm.get())}</label>
                    <input type="range" min="0.01" max="1.0" step="0.001" style="width: 100%;"
                        prop:value=move || distance_nm.get().to_string()
                        on:input=move |ev| { let t: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into(); set_distance_nm.set(t.value().parse().unwrap_or(0.053)); }
                    />
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem;">
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(99,102,241,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.7rem; color: #94a3b8;">"Coulomb Force"</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #6366f1;">{move || format!("{:.2e} N", f_coulomb.get())}</div>
                </div>
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(16,185,129,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.7rem; color: #94a3b8;">"Gravity (proton-proton)"</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #10b981;">{move || format!("{:.2e} N", f_gravity.get())}</div>
                </div>
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(239,68,68,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.7rem; color: #94a3b8;">"EM/Gravity Ratio"</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #ef4444;">{move || format!("{:.2e}×", ratio.get())}</div>
                </div>
            </div>
        </div>
    }
}
