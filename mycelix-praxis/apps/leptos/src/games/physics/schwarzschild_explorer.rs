// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Schwarzschild Radius Explorer — r_s = 2GM/c²
//!
//! Students adjust mass and see the event horizon, comparing with known objects.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn SchwarzschildExplorer(node_id: String) -> impl IntoView {
    let g = 6.674e-11_f64;
    let c = 2.998e8_f64;
    let c2 = c * c;
    let m_sun = 1.989e30_f64;

    let (mass_solar, set_mass_solar) = signal(1.0_f64); // In solar masses

    let r_s = Memo::new(move |_| 2.0 * g * mass_solar.get() * m_sun / c2);
    let r_s_km = Memo::new(move |_| r_s.get() / 1000.0);

    // Comparison objects
    let comparisons = move || {
        let rs = r_s.get();
        let mut comps = Vec::new();
        if rs < 0.01 { comps.push(("Atom", 1e-10)); }
        if rs < 1.0 { comps.push(("Human hair", 5e-5)); }
        if rs < 100.0 { comps.push(("Tennis ball", 0.033)); }
        comps.push(("Earth radius", 6.371e6));
        comps.push(("Sun radius", 6.96e8));
        comps.push(("Earth orbit", 1.496e11));
        comps
    };

    // Known black holes
    let known = vec![
        ("Sagittarius A*", 4.0e6),
        ("M87*", 6.5e9),
        ("TON 618", 6.6e10),
        ("Stellar (typical)", 10.0),
    ];

    view! {
        <div style="padding: 1rem;">
            <h2 style="font-size: 1.5rem; margin-bottom: 0.5rem; color: var(--accent-color, #6366f1);">"Schwarzschild Radius"</h2>
            <p style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 1rem;">
                <code style="color: #10b981;">"r_s = 2GM/c²"</code>
                " — The point of no return"
            </p>

            // Presets
            <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
                {known.into_iter().map(|(name, m)| view! {
                    <button
                        style="padding: 0.375rem 0.75rem; background: #1f2937; color: #e2e8f0; border: 1px solid #374151; border-radius: 0.375rem; cursor: pointer; font-size: 0.75rem;"
                        on:click=move |_| set_mass_solar.set(m)
                    >{name}</button>
                }).collect::<Vec<_>>()}
            </div>

            <div style="margin-bottom: 1rem;">
                <label style="font-size: 0.8rem; color: #94a3b8;">"Mass: " {move || format!("{:.2e}", mass_solar.get())} " M☉"</label>
                <input type="range" min="-1" max="11" step="0.1" style="width: 100%;"
                    prop:value=move || mass_solar.get().log10().to_string()
                    on:input=move |ev| {
                        let target: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into();
                        let log_val: f64 = target.value().parse().unwrap_or(0.0);
                        set_mass_solar.set(10.0_f64.powf(log_val));
                    }
                />
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(99,102,241,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.75rem; color: #94a3b8;">"Event Horizon Radius"</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #6366f1;">{move || format!("{:.3e} m", r_s.get())}</div>
                    <div style="font-size: 0.875rem; color: #94a3b8;">{move || format!("({:.1} km)", r_s_km.get())}</div>
                </div>
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(16,185,129,0.2); border-radius: 0.5rem;">
                    <div style="font-size: 0.75rem; color: #94a3b8; margin-bottom: 0.5rem;">"Size Comparison"</div>
                    {move || comparisons().into_iter().map(|(name, size)| {
                        let ratio = r_s.get() / size;
                        let color = if ratio > 1.0 { "#ef4444" } else { "#10b981" };
                        view! {
                            <div style="font-size: 0.75rem; display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                                <span style="color: #94a3b8;">{name}</span>
                                <span style=format!("color: {};", color)>{format!("{:.2e}×", ratio)}</span>
                            </div>
                        }
                    }).collect::<Vec<_>>()}
                </div>
            </div>

            <p style="font-size: 0.75rem; color: #64748b; font-style: italic;">
                "For the Sun: r_s = 2.95 km. You'd have to compress Earth to 8.87 mm to make it a black hole."
            </p>
        </div>
    }
}
