// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Toggleable metrics panel for consciousness research mode.
//! Shows Phi numerically, 8 harmony activations, and spectral features.

use leptos::prelude::*;

/// Harmony names for display.
const HARMONY_NAMES: [&str; 8] = [
    "Coherence", "Flourishing", "Wisdom", "Play",
    "Interconnect", "Reciprocity", "Progress", "Stillness",
];

const HARMONY_COLORS: [&str; 8] = [
    "#ffe6b3", "#ffb34d", "#4dcc80", "#e666cc",
    "#4d99ff", "#b380ff", "#ff804d", "#99e6ff",
];

/// Research metrics overlay.
#[component]
pub fn MetricsPanel(
    phi: RwSignal<f64>,
    arousal: RwSignal<f64>,
    valence: RwSignal<f64>,
    visible: RwSignal<bool>,
) -> impl IntoView {
    view! {
        <Show when=move || visible.get()>
            <div class="metrics-panel">
                // Phi display
                <div class="metric-phi">
                    <span class="metric-phi-value">{move || format!("{:.3}", phi.get())}</span>
                    <span class="metric-phi-label">"Phi"</span>
                </div>

                // V/A readout
                <div class="metric-row">
                    <span class="metric-label">"V"</span>
                    <span class="metric-value">{move || format!("{:+.2}", valence.get())}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">"A"</span>
                    <span class="metric-value">{move || format!("{:.2}", arousal.get())}</span>
                </div>

                // Harmony activations (computed from V/A/Phi)
                <div class="metric-harmonies">
                    {(0..8).map(|i| {
                        let color = HARMONY_COLORS[i];
                        let name = HARMONY_NAMES[i];
                        view! {
                            <div class="harmony-bar-row">
                                <span class="harmony-bar-label">{name}</span>
                                <div class="harmony-bar-track">
                                    <div class="harmony-bar-fill"
                                        style=move || {
                                            let a = arousal.get() as f32;
                                            let v = valence.get() as f32;
                                            let p = phi.get() as f32;
                                            let activation = compute_harmony(i, a, v, p);
                                            format!("width: {}%; background: {};", (activation * 100.0) as u32, color)
                                        }
                                    />
                                </div>
                            </div>
                        }
                    }).collect_view()}
                </div>
            </div>
        </Show>
    }
}

/// Compute harmony activation (mirrors SynthState::auto_harmonies).
fn compute_harmony(i: usize, arousal: f32, valence: f32, phi: f32) -> f32 {
    match i {
        0 => 0.5 + phi * 0.5,
        1 => 0.3 + valence.max(0.0) * 0.7,
        2 => 0.4 + phi * 0.3,
        3 => 0.2 + arousal * 0.5 + (-valence).max(0.0) * 0.3,
        4 => 0.3 + (1.0 - arousal) * 0.4,
        5 => 0.3 + valence.abs() * 0.3 + phi * 0.2,
        6 => arousal * 0.8,
        7 => (1.0 - arousal) * 0.6 + phi * 0.2,
        _ => 0.0,
    }
    .clamp(0.0, 1.0)
}
