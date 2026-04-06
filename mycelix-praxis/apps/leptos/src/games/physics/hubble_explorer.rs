// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hubble's Law Explorer — v = H₀d
//!
//! Students see how the expansion of the universe means more distant
//! galaxies recede faster, and estimate the age of the universe.

use leptos::prelude::*;
use wasm_bindgen::JsCast;

#[component]
pub fn HubbleExplorer(node_id: String) -> impl IntoView {
    let (h0, set_h0) = signal(70.0_f64); // km/s/Mpc (Hubble constant)
    let (distance_mpc, set_distance_mpc) = signal(100.0_f64); // Megaparsecs

    let recession_velocity = Memo::new(move |_| h0.get() * distance_mpc.get()); // km/s
    let recession_c = Memo::new(move |_| recession_velocity.get() / 299792.458); // fraction of c
    let age_gyr = Memo::new(move |_| {
        if h0.get() < 0.1 { return 0.0; }
        // t = 1/H₀ (Hubble time)
        // H₀ in km/s/Mpc → convert to 1/s → 1/H₀ in seconds → Gyr
        let h0_per_s = h0.get() / (3.086e19); // Mpc in km
        1.0 / h0_per_s / (3.156e16) // seconds to Gyr
    });

    let objects = vec![
        ("Andromeda (M31)", 0.78),
        ("Virgo Cluster", 16.5),
        ("Coma Cluster", 100.0),
        ("Hubble Deep Field", 3000.0),
        ("CMB (observable edge)", 14000.0),
    ];

    view! {
        <div style="padding: 1rem;">
            <h2 style="font-size: 1.5rem; margin-bottom: 0.5rem; color: var(--accent-color, #6366f1);">"Hubble's Law"</h2>
            <p style="font-size: 0.875rem; color: #94a3b8; margin-bottom: 1rem;">
                <code style="color: #10b981;">"v = H₀d"</code>
                " — The universe is expanding. Farther = faster."
            </p>

            <div style="display: flex; gap: 0.375rem; margin-bottom: 1rem; flex-wrap: wrap;">
                {objects.into_iter().map(|(name, d)| view! {
                    <button
                        style="padding: 0.25rem 0.5rem; background: #1f2937; color: #e2e8f0; border: 1px solid #374151; border-radius: 0.25rem; cursor: pointer; font-size: 0.65rem;"
                        on:click=move |_| set_distance_mpc.set(d)
                    >{name}</button>
                }).collect::<Vec<_>>()}
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem;">
                <div>
                    <label style="font-size: 0.8rem; color: #94a3b8;">"H₀: " {move || format!("{:.1}", h0.get())} " km/s/Mpc"</label>
                    <input type="range" min="50" max="90" step="0.5" style="width: 100%;"
                        prop:value=move || h0.get().to_string()
                        on:input=move |ev| { let t: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into(); set_h0.set(t.value().parse().unwrap_or(70.0)); }
                    />
                    <div style="font-size: 0.65rem; color: #64748b;">"(Hubble tension: 67 vs 73)"</div>
                </div>
                <div>
                    <label style="font-size: 0.8rem; color: #94a3b8;">"Distance: " {move || format!("{:.0}", distance_mpc.get())} " Mpc"</label>
                    <input type="range" min="1" max="14000" step="10" style="width: 100%;"
                        prop:value=move || distance_mpc.get().to_string()
                        on:input=move |ev| { let t: web_sys::HtmlInputElement = ev.target().unwrap().unchecked_into(); set_distance_mpc.set(t.value().parse().unwrap_or(100.0)); }
                    />
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(99,102,241,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.7rem; color: #94a3b8;">"Recession Velocity"</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #6366f1;">{move || format!("{:.0} km/s", recession_velocity.get())}</div>
                    <div style="font-size: 0.65rem; color: #94a3b8;">{move || format!("({:.4}c)", recession_c.get())}</div>
                </div>
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(16,185,129,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.7rem; color: #94a3b8;">"Hubble Time (Universe Age)"</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #10b981;">{move || format!("{:.1} Gyr", age_gyr.get())}</div>
                </div>
                <div style="padding: 1rem; background: #111827; border: 1px solid rgba(245,158,11,0.2); border-radius: 0.5rem; text-align: center;">
                    <div style="font-size: 0.7rem; color: #94a3b8;">"Light Travel Time"</div>
                    <div style="font-size: 1.25rem; font-weight: 700; color: #f59e0b;">{move || format!("{:.0} Mly", distance_mpc.get() * 3.26)}</div>
                </div>
            </div>
        </div>
    }
}
