// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Newton's Gravitation Explorer — F = GMm/r²
//!
//! Students adjust masses and distance, see gravitational force update live.
//! Compares gravity with Coulomb force to show structural similarity.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn GravityExplorer(node_id: String) -> impl IntoView {
    let g_const = 6.674e-11_f64; // m³/(kg·s²)

    let (mass1, set_mass1) = signal(5.97e24_f64);  // Earth mass (kg)
    let (mass2, set_mass2) = signal(7.35e22_f64);   // Moon mass (kg)
    let (distance, set_distance) = signal(3.84e8_f64); // Earth-Moon distance (m)

    let force = Memo::new(move |_| {
        let r = distance.get();
        if r < 1.0 { return 0.0; }
        g_const * mass1.get() * mass2.get() / (r * r)
    });

    let acceleration = Memo::new(move |_| {
        let m = mass2.get();
        if m < 0.01 { return 0.0; }
        force.get() / m
    });

    // Presets
    let presets = vec![
        ("Earth-Moon", 5.97e24, 7.35e22, 3.84e8),
        ("Earth-Sun", 5.97e24, 1.989e30, 1.496e11),
        ("Earth-Apple", 5.97e24, 0.1, 6.371e6),
        ("Two People", 70.0, 70.0, 1.0),
    ];

    view! {
        <div style="padding: 1rem;">
            <h2 style="font-size: 1.5rem; margin-bottom: 0.5rem; color: var(--accent-color, #6366f1);">"Newton's Gravitation"</h2>
            <p style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 1rem;">
                <code style="color: #10b981;">"F = GMm/r²"</code>
                " — Every mass attracts every other mass"
            </p>

            // Presets
            <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
                {presets.into_iter().map(|(name, m1, m2, r)| view! {
                    <button
                        style="padding: 0.375rem 0.75rem; background: #1f2937; color: #e2e8f0; border: 1px solid #374151; border-radius: 0.375rem; cursor: pointer; font-size: 0.75rem;"
                        on:click=move |_| { set_mass1.set(m1); set_mass2.set(m2); set_distance.set(r); }
                    >{name}</button>
                }).collect::<Vec<_>>()}
            </div>

            // Sliders
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                <div>
                    <label style="font-size: 0.8rem; color: #94a3b8;">"Mass 1 (kg): " {move || format!("{:.2e}", mass1.get())}</label>
                    <input type="range" min="0.01" max="30" step="0.1" style="width: 100%;"
                        prop:value=move || (mass1.get().log10() + 1.0).to_string()
                        on:input=move |ev| {
                            let target: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into();
                            let log_val: f64 = target.value().parse().unwrap_or(1.0);
                            set_mass1.set(10.0_f64.powf(log_val - 1.0));
                        }
                    />
                </div>
                <div>
                    <label style="font-size: 0.8rem; color: #94a3b8;">"Mass 2 (kg): " {move || format!("{:.2e}", mass2.get())}</label>
                    <input type="range" min="0.01" max="30" step="0.1" style="width: 100%;"
                        prop:value=move || (mass2.get().log10() + 1.0).to_string()
                        on:input=move |ev| {
                            let target: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into();
                            let log_val: f64 = target.value().parse().unwrap_or(1.0);
                            set_mass2.set(10.0_f64.powf(log_val - 1.0));
                        }
                    />
                </div>
                <div>
                    <label style="font-size: 0.8rem; color: #94a3b8;">"Distance (m): " {move || format!("{:.2e}", distance.get())}</label>
                    <input type="range" min="0" max="12" step="0.1" style="width: 100%;"
                        prop:value=move || distance.get().log10().to_string()
                        on:input=move |ev| {
                            let target: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into();
                            let log_val: f64 = target.value().parse().unwrap_or(1.0);
                            set_distance.set(10.0_f64.powf(log_val));
                        }
                    />
                </div>
            </div>

            // Results
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(99,102,241,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94a3b8;">"Gravitational Force"</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #6366f1;">{move || format!("{:.3e} N", force.get())}</div>
                </div>
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(16,185,129,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94a3b8;">"Acceleration on M2"</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #10b981;">{move || format!("{:.3e} m/s²", acceleration.get())}</div>
                </div>
            </div>

            <p style="font-size: 0.75rem; color: #64748b; margin-top: 1rem; font-style: italic;">
                "G = 6.674 × 10⁻¹¹ m³/(kg·s²). Same 1/r² structure as Coulomb's law — HDC skeleton similarity confirms."
            </p>
        </div>
    }
}
