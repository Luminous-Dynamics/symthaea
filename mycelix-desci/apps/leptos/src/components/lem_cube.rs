// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! LEM Cube visualization — 3-axis epistemic classification display.

use leptos::prelude::*;

/// Render the LEM Cube as three horizontal progress bars.
#[component]
pub fn LemCube(
    #[prop(into)] empirical: u8,
    #[prop(into)] normative: u8,
    #[prop(into)] materiality: u8,
) -> impl IntoView {
    let e_pct = (empirical as f32 / 4.0 * 100.0) as u32;
    let n_pct = (normative as f32 / 3.0 * 100.0) as u32;
    let m_pct = (materiality as f32 / 3.0 * 100.0) as u32;

    let e_labels = ["E0: Null", "E1: Testimonial", "E2: Verifiable", "E3: Cryptographic", "E4: Reproducible"];
    let n_labels = ["N0: Personal", "N1: Communal", "N2: Network", "N3: Axiomatic"];
    let m_labels = ["M0: Ephemeral", "M1: Temporal", "M2: Persistent", "M3: Foundational"];

    let e_label = e_labels.get(empirical as usize).unwrap_or(&"?");
    let n_label = n_labels.get(normative as usize).unwrap_or(&"?");
    let m_label = m_labels.get(materiality as usize).unwrap_or(&"?");

    view! {
        <div class="glass-panel" style="padding: 1rem;">
            <h4 style="font-size: 0.875rem; margin-bottom: 0.75rem; color: var(--accent-indigo);">"LEM Cube Classification"</h4>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 0.25rem;">
                        <span style="color: var(--text-secondary);">"Empirical"</span>
                        <span>{*e_label}</span>
                    </div>
                    <div style="height: 6px; background: var(--bg-secondary); border-radius: 3px;">
                        <div style=format!("width: {}%; height: 100%; background: linear-gradient(90deg, var(--tier-e0), var(--tier-e4)); border-radius: 3px;", e_pct)></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 0.25rem;">
                        <span style="color: var(--text-secondary);">"Normative"</span>
                        <span>{*n_label}</span>
                    </div>
                    <div style="height: 6px; background: var(--bg-secondary); border-radius: 3px;">
                        <div style=format!("width: {}%; height: 100%; background: var(--accent-indigo); border-radius: 3px;", n_pct)></div>
                    </div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-bottom: 0.25rem;">
                        <span style="color: var(--text-secondary);">"Materiality"</span>
                        <span>{*m_label}</span>
                    </div>
                    <div style="height: 6px; background: var(--bg-secondary); border-radius: 3px;">
                        <div style=format!("width: {}%; height: 100%; background: var(--accent-emerald); border-radius: 3px;", m_pct)></div>
                    </div>
                </div>
            </div>
        </div>
    }
}
