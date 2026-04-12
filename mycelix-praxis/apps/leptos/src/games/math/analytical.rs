// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Analytical Geometry Workbench — place points, see distance/midpoint/gradient live.

use leptos::prelude::*;
use crate::games::shared::graph_renderer::{GraphArea, GraphConfig};
use crate::curriculum::{use_set_progress, ProgressStatus};

#[component]
pub fn AnalyticalGeometryExplorer(node_id: String) -> impl IntoView {
    let set_progress = use_set_progress();

    let (x1, set_x1) = signal(1.0_f64);
    let (y1, set_y1) = signal(2.0_f64);
    let (x2, set_x2) = signal(5.0_f64);
    let (y2, set_y2) = signal(6.0_f64);
    let (challenge_idx, set_challenge_idx) = signal(0_usize);
    let (show_success, set_show_success) = signal(false);

    let distance = Memo::new(move |_| {
        let dx = x2.get() - x1.get();
        let dy = y2.get() - y1.get();
        (dx * dx + dy * dy).sqrt()
    });

    let midpoint_x = Memo::new(move |_| (x1.get() + x2.get()) / 2.0);
    let midpoint_y = Memo::new(move |_| (y1.get() + y2.get()) / 2.0);

    let gradient = Memo::new(move |_| {
        let dx = x2.get() - x1.get();
        if dx.abs() < 0.01 { f64::INFINITY } else { (y2.get() - y1.get()) / dx }
    });

    let config = GraphConfig { x_min: -8.0, x_max: 8.0, y_min: -8.0, y_max: 8.0, show_grid: true, grid_step: 2.0 };

    let check_challenge = {
        let node_id = node_id.clone();
        move || {
            let idx = challenge_idx.get_untracked();
            let d = distance.get_untracked();
            let mx = midpoint_x.get_untracked();
            let my = midpoint_y.get_untracked();
            let g = gradient.get_untracked();

            let passed = match idx {
                0 => (d - 5.0).abs() < 0.3,              // Distance = 5
                1 => (mx - 0.0).abs() < 0.3 && (my - 0.0).abs() < 0.3, // Midpoint at origin
                2 => (g - 2.0).abs() < 0.2,              // Gradient = 2
                3 => g.abs() > 100.0 || g.is_infinite(),  // Vertical line (undefined gradient)
                _ => false,
            };

            if passed {
                set_show_success.set(true);
                let next = idx + 1;
                let node_id = node_id.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    gloo_timers::future::sleep(std::time::Duration::from_millis(1500)).await;
                    set_show_success.set(false);
                    set_challenge_idx.set(next);
                    if next >= 4 {
                        set_progress.update(|p| {
                            let e = p.nodes.entry(node_id).or_default();
                            e.mastery_permille = e.mastery_permille.max(800);
                            if e.status == ProgressStatus::NotStarted { e.status = ProgressStatus::Studying; }
                        });
                    }
                });
            }
        }
    };

    view! {
        <div class="game-container">
            <GraphArea config=config>
                // Line between points
                <line x1=move || x1.get() y1=move || -y1.get() x2=move || x2.get() y2=move || -y2.get()
                    stroke="var(--primary)" stroke-width="0.08" />

                // Point A
                <circle cx=move || x1.get() cy=move || -y1.get() r="0.25" fill="var(--info)" stroke="var(--bg)" stroke-width="0.06" />
                <text x=move || x1.get() + 0.4 y=move || -y1.get() - 0.4 font-size="0.5" fill="var(--info)">
                    {move || format!("A({:.0},{:.0})", x1.get(), y1.get())}
                </text>

                // Point B
                <circle cx=move || x2.get() cy=move || -y2.get() r="0.25" fill="var(--error)" stroke="var(--bg)" stroke-width="0.06" />
                <text x=move || x2.get() + 0.4 y=move || -y2.get() - 0.4 font-size="0.5" fill="var(--error)">
                    {move || format!("B({:.0},{:.0})", x2.get(), y2.get())}
                </text>

                // Midpoint
                <circle cx=move || midpoint_x.get() cy=move || -midpoint_y.get() r="0.18"
                    fill="var(--success)" stroke="var(--bg)" stroke-width="0.04" />
                <text x=move || midpoint_x.get() + 0.3 y=move || -midpoint_y.get() + 0.6 font-size="0.4" fill="var(--success)">
                    "M"
                </text>
            </GraphArea>

            // Values display
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.5rem; text-align: center; padding: 0.75rem 0">
                <div>
                    <div style="font-size: 0.7rem; color: var(--text-tertiary)">"Distance"</div>
                    <div style="font-size: 1.1rem; font-weight: 700">{move || format!("{:.2}", distance.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--text-tertiary)">"Midpoint"</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: var(--success)">
                        {move || format!("({:.1},{:.1})", midpoint_x.get(), midpoint_y.get())}
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--text-tertiary)">"Gradient"</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: var(--primary)">
                        {move || { let g = gradient.get(); if g.is_infinite() { "undef".to_string() } else { format!("{:.2}", g) } }}
                    </div>
                </div>
            </div>

            // Point sliders
            <div class="game-sliders">
                <div class="game-slider-row">
                    <label class="game-slider-label">"Ax"</label>
                    <input type="range" min="-7" max="7" step="0.5" class="game-slider"
                        prop:value=move || x1.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_x1.set(v); } } />
                    <span class="game-slider-value">{move || format!("{:.0}", x1.get())}</span>
                </div>
                <div class="game-slider-row">
                    <label class="game-slider-label">"Ay"</label>
                    <input type="range" min="-7" max="7" step="0.5" class="game-slider"
                        prop:value=move || y1.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_y1.set(v); } } />
                    <span class="game-slider-value">{move || format!("{:.0}", y1.get())}</span>
                </div>
                <div class="game-slider-row">
                    <label class="game-slider-label">"Bx"</label>
                    <input type="range" min="-7" max="7" step="0.5" class="game-slider"
                        prop:value=move || x2.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_x2.set(v); } } />
                    <span class="game-slider-value">{move || format!("{:.0}", x2.get())}</span>
                </div>
                <div class="game-slider-row">
                    <label class="game-slider-label">"By"</label>
                    <input type="range" min="-7" max="7" step="0.5" class="game-slider"
                        prop:value=move || y2.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_y2.set(v); } } />
                    <span class="game-slider-value">{move || format!("{:.0}", y2.get())}</span>
                </div>
            </div>

            // Challenges
            <div class="game-challenge">
                {move || {
                    let idx = challenge_idx.get();
                    if idx >= 4 {
                        view! { <div class="game-complete"><div class="game-complete-icon">"\u{2714}"</div><div class="game-complete-text">"All challenges complete!"</div></div> }.into_any()
                    } else if show_success.get() {
                        view! { <div class="game-success"><span class="game-success-icon">"\u{2714}"</span>" Correct!"</div> }.into_any()
                    } else {
                        let instructions = ["Place the points so the distance AB = 5", "Place the points so the midpoint is at the origin (0, 0)", "Place the points so the gradient = 2", "Place the points to make a vertical line (undefined gradient)"];
                        let hints = ["Distance = \u{221a}((x\u{2082}-x\u{2081})\u{00b2} + (y\u{2082}-y\u{2081})\u{00b2}). Try (0,0) and (3,4).", "Midpoint = ((x\u{2081}+x\u{2082})/2, (y\u{2081}+y\u{2082})/2). Set them symmetric about the origin.", "Gradient = (y\u{2082}-y\u{2081})/(x\u{2082}-x\u{2081}). Rise must be double the run.", "A vertical line has the same x-coordinate for both points."];
                        let check = check_challenge.clone();
                        view! {
                            <div class="game-challenge-active">
                                <div class="game-challenge-number">"Challenge "{idx + 1}" of 4"</div>
                                <div class="game-challenge-instruction">{instructions[idx]}</div>
                                <div class="game-challenge-actions">
                                    <button class="praxis-filter-btn active" on:click=move |_| check()>"Check"</button>
                                </div>
                                <div class="game-challenge-hint">{hints[idx]}</div>
                            </div>
                        }.into_any()
                    }
                }}
            </div>
        </div>
    }
}
