// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

use crate::state::AppState;

/// Neuromodulator bar chart: all 9 transmitters from the consciousness bath.
#[component]
pub fn NeuroBars() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let neuromods = state.neuromods;

    let labels = ["DA", "NE", "5-HT", "OT", "ACh", "GABA", "Glu", "Ade", "eCB"];
    let colors = [
        "var(--da-blue)",
        "var(--ne-red)",
        "var(--sht-green)",
        "var(--ot-pink)",
        "#8b5cf6", // ACh — violet
        "#f59e0b", // GABA — amber
        "#ef4444", // Glutamate — red-orange
        "#6366f1", // Adenosine — indigo
        "#10b981", // Endocannabinoid — emerald
    ];

    view! {
        <div>
            {labels
                .iter()
                .zip(colors.iter())
                .enumerate()
                .map(|(i, (label, color))| {
                    let label = *label;
                    let color = *color;
                    view! {
                        <div class="neuro-bar-row">
                            <span class="nb-label">{label}</span>
                            <div class="nb-track">
                                <div
                                    class="nb-fill"
                                    style:width=move || {
                                        format!("{}%", (neuromods.get()[i] * 100.0) as u32)
                                    }
                                    style:background=color
                                />
                            </div>
                            <span class="nb-val">
                                {move || format!("{:.2}", neuromods.get()[i])}
                            </span>
                        </div>
                    }
                })
                .collect::<Vec<_>>()}
        </div>
    }
}
