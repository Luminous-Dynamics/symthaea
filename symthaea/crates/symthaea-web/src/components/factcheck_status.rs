// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

use crate::state::AppState;

/// Factcheck bridge status panel.
///
/// Displays Broca→Mycelix knowledge verification metrics:
/// - Accuracy EMA (color-coded bar)
/// - Claims verified / suppressed counters
/// - Pending verification queue depth
#[component]
pub fn FactcheckStatus() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");

    let accuracy_pct = move || (state.factcheck_accuracy.get() * 100.0) as u32;
    let accuracy_color = move || {
        let a = state.factcheck_accuracy.get();
        if a >= 0.8 {
            "var(--leaf-green)"
        } else if a >= 0.5 {
            "#eab308"
        } else {
            "#ef4444"
        }
    };

    let bar_width = move || format!("{}%", accuracy_pct());

    view! {
        <div class="factcheck-panel glass-panel">
            <h3 class="panel-title">"Factcheck Bridge"</h3>

            <div class="factcheck-accuracy">
                <div class="factcheck-label">
                    <span>"Accuracy"</span>
                    <span class="factcheck-value">{move || format!("{}%", accuracy_pct())}</span>
                </div>
                <div class="factcheck-bar-bg">
                    <div
                        class="factcheck-bar-fill"
                        style:width=bar_width
                        style:background=accuracy_color
                    />
                </div>
            </div>

            <div class="factcheck-counters">
                <div class="factcheck-counter verified">
                    <span class="counter-icon">"+"</span>
                    <span class="counter-value">{move || state.factcheck_verified.get()}</span>
                    <span class="counter-label">"verified"</span>
                </div>
                <div class="factcheck-counter suppressed">
                    <span class="counter-icon">"-"</span>
                    <span class="counter-value">{move || state.factcheck_suppressed.get()}</span>
                    <span class="counter-label">"suppressed"</span>
                </div>
                <div class="factcheck-counter pending">
                    <span class="counter-icon">"~"</span>
                    <span class="counter-value">{move || state.factcheck_pending.get()}</span>
                    <span class="counter-label">"pending"</span>
                </div>
            </div>
        </div>
    }
}
