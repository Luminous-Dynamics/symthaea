// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lorentz Factor Explorer — γ = 1/√(1-v²/c²)
//!
//! Students see time dilation, length contraction, and relativistic mass
//! as velocity approaches c.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn LorentzExplorer(node_id: String) -> impl IntoView {
    let c = 299_792_458.0_f64;

    let (v_fraction, set_v_fraction) = signal(0.0_f64); // v/c (0 to 0.9999)

    let gamma = Memo::new(move |_| {
        let beta = v_fraction.get();
        if beta >= 1.0 { return f64::INFINITY; }
        1.0 / (1.0 - beta * beta).sqrt()
    });

    let time_dilation = Memo::new(move |_| gamma.get()); // t' = γt
    let length_contraction = Memo::new(move |_| 1.0 / gamma.get()); // L' = L/γ
    let v_ms = Memo::new(move |_| v_fraction.get() * c);

    let presets = vec![
        ("Walking", 0.000000005),
        ("Jet (Mach 2)", 0.0000023),
        ("ISS orbit", 0.0000257),
        ("1% c", 0.01),
        ("10% c", 0.1),
        ("50% c", 0.5),
        ("90% c", 0.9),
        ("99% c", 0.99),
        ("99.99% c", 0.9999),
    ];

    view! {
        <div style="padding: 1rem;">
            <h2 style="font-size: 1.5rem; margin-bottom: 0.5rem; color: var(--accent-color, #6366f1);">"Lorentz Factor"</h2>
            <p style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 1rem;">
                <code style="color: #10b981;">"γ = 1/√(1 - v²/c²)"</code>
                " — Time slows, space contracts, mass grows"
            </p>

            <div style="display: flex; gap: 0.375rem; margin-bottom: 1rem; flex-wrap: wrap;">
                {presets.into_iter().map(|(name, v)| view! {
                    <button
                        style="padding: 0.25rem 0.5rem; background: #1f2937; color: #e2e8f0; border: 1px solid #374151; border-radius: 0.25rem; cursor: pointer; font-size: 0.65rem;"
                        on:click=move |_| set_v_fraction.set(v)
                    >{name}</button>
                }).collect::<Vec<_>>()}
            </div>

            <div style="margin-bottom: 1rem;">
                <label style="font-size: 0.8rem; color: #94a3b8;">"Velocity: " {move || format!("{:.4}c", v_fraction.get())} " (" {move || format!("{:.0} m/s", v_ms.get())} ")"</label>
                <input type="range" min="0" max="0.9999" step="0.0001" style="width: 100%;"
                    prop:value=move || v_fraction.get().to_string()
                    on:input=move |ev| { let t: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into(); set_v_fraction.set(t.value().parse().unwrap_or(0.0)); }
                />
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(99,102,241,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.7rem; color: #94a3b8;">"Lorentz Factor γ"</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #6366f1;">{move || format!("{:.4}", gamma.get())}</div>
                </div>
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(245,158,11,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.7rem; color: #94a3b8;">"Time Dilation"</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #f59e0b;">{move || format!("{:.4}×", time_dilation.get())}</div>
                    <div style="font-size: 0.65rem; color: #94a3b8;">"1 sec → " {move || format!("{:.4} sec", time_dilation.get())}</div>
                </div>
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(16,185,129,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.7rem; color: #94a3b8;">"Length Contraction"</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #10b981;">{move || format!("{:.4}×", length_contraction.get())}</div>
                    <div style="font-size: 0.65rem; color: #94a3b8;">"1 m → " {move || format!("{:.4} m", length_contraction.get())}</div>
                </div>
            </div>
        </div>
    }
}
