// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

use crate::state::AppState;

/// Eight-point SVG radar chart for the Eight Harmonies — reactive.
#[component]
pub fn HarmonyRadar() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");
    let labels = ["RC", "PSF", "IW", "IP", "UI", "SR", "EP", "SS"];

    let cx = 90.0_f32;
    let cy = 90.0_f32;
    let r = 70.0_f32;

    // Label positions (static — outside the chart)
    let label_elems: Vec<_> = labels
        .iter()
        .enumerate()
        .map(|(i, label)| {
            let angle = (i as f32 / 8.0) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
            let x = cx + angle.cos() * (r + 14.0);
            let y = cy + angle.sin() * (r + 14.0);
            (x, y, *label)
        })
        .collect();

    view! {
        <div class="harmony-radar">
            <svg viewBox="0 0 180 180" xmlns="http://www.w3.org/2000/svg">
                // Grid circles
                <circle cx={cx.to_string()} cy={cy.to_string()} r={(r * 0.33).to_string()} fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="0.5" />
                <circle cx={cx.to_string()} cy={cy.to_string()} r={(r * 0.66).to_string()} fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="0.5" />
                <circle cx={cx.to_string()} cy={cy.to_string()} r={r.to_string()} fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="0.5" />
                // Reactive data polygon
                <polygon
                    points=move || {
                        let scores = state.harmony_scores.get();
                        scores.iter().enumerate().map(|(i, v)| {
                            let angle = (i as f32 / 8.0) * std::f32::consts::TAU - std::f32::consts::FRAC_PI_2;
                            let x = cx + angle.cos() * r * v;
                            let y = cy + angle.sin() * r * v;
                            format!("{x:.1},{y:.1}")
                        }).collect::<Vec<_>>().join(" ")
                    }
                    fill="rgba(126,200,160,0.15)"
                    stroke="var(--leaf-green)"
                    stroke-width="1.5"
                    style="transition: all 0.5s ease;"
                />
                // Labels
                {label_elems
                    .into_iter()
                    .map(|(x, y, label)| {
                        view! {
                            <text
                                x={x.to_string()}
                                y={y.to_string()}
                                text-anchor="middle"
                                dominant-baseline="central"
                                fill="var(--fg-muted)"
                                font-size="7"
                            >
                                {label}
                            </text>
                        }
                    })
                    .collect::<Vec<_>>()}
            </svg>
            <div style="text-align: center; font-size: 0.6rem; color: var(--fg-muted); margin-top: 0.2rem;">
                {move || state.dominant_harmony.get()}
            </div>
        </div>
    }
}
