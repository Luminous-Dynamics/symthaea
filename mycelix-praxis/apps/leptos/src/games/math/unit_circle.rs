// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unit Circle Explorer — interactive trigonometry visualization.
//!
//! Students rotate a point around the unit circle and see sin, cos, tan
//! values update in real-time with visual projections.

use leptos::prelude::*;
use crate::games::shared::graph_renderer::{GraphArea, GraphConfig};
use crate::curriculum::{use_set_progress, ProgressStatus};

struct Challenge {
    instruction: &'static str,
    check: fn(f64) -> bool,
    hint: &'static str,
}

fn challenges() -> Vec<Challenge> {
    vec![
        Challenge {
            instruction: "Find an angle where sin \u{03b8} = 0",
            check: |angle| {
                let sin = (angle.to_radians()).sin();
                sin.abs() < 0.08
            },
            hint: "sin = 0 when the point is on the x-axis. Try 0\u{00b0} or 180\u{00b0}.",
        },
        Challenge {
            instruction: "Find an angle where cos \u{03b8} = 0",
            check: |angle| {
                let cos = (angle.to_radians()).cos();
                cos.abs() < 0.08
            },
            hint: "cos = 0 when the point is on the y-axis. Try 90\u{00b0} or 270\u{00b0}.",
        },
        Challenge {
            instruction: "Find an angle in Quadrant 2 where sin is positive",
            check: |angle| {
                let rad = angle.to_radians();
                angle > 90.0 && angle < 180.0 && rad.sin() > 0.3
            },
            hint: "Quadrant 2 is between 90\u{00b0} and 180\u{00b0}. Sin is positive here.",
        },
        Challenge {
            instruction: "Find an angle where sin \u{03b8} = cos \u{03b8}",
            check: |angle| {
                let rad = angle.to_radians();
                (rad.sin() - rad.cos()).abs() < 0.08
            },
            hint: "sin = cos at 45\u{00b0} and 225\u{00b0}.",
        },
        Challenge {
            instruction: "Find an angle where sin \u{03b8} is negative and cos \u{03b8} is negative",
            check: |angle| {
                let rad = angle.to_radians();
                rad.sin() < -0.2 && rad.cos() < -0.2
            },
            hint: "Both sin and cos are negative in Quadrant 3 (180\u{00b0} to 270\u{00b0}).",
        },
    ]
}

#[component]
pub fn UnitCircleExplorer(node_id: String) -> impl IntoView {
    let set_progress = use_set_progress();
    let (angle_deg, set_angle_deg) = signal(45.0_f64);
    let (challenge_idx, set_challenge_idx) = signal(0_usize);
    let (show_success, set_show_success) = signal(false);
    let (completed_count, set_completed_count) = signal(0_u32);

    let angle_rad = Memo::new(move |_| angle_deg.get().to_radians());
    let cos_val = Memo::new(move |_| angle_rad.get().cos());
    let sin_val = Memo::new(move |_| angle_rad.get().sin());
    let tan_val = Memo::new(move |_| {
        let c = cos_val.get();
        if c.abs() < 0.01 { f64::NAN } else { sin_val.get() / c }
    });

    let point_x = Memo::new(move |_| cos_val.get());
    let point_y = Memo::new(move |_| -sin_val.get()); // SVG inverted

    let quadrant = Memo::new(move |_| {
        let a = angle_deg.get() % 360.0;
        if a < 90.0 { "Q1" } else if a < 180.0 { "Q2" } else if a < 270.0 { "Q3" } else { "Q4" }
    });

    // Unit circle path (full circle)
    let circle_path = "M 1 0 A 1 1 0 0 1 -1 0 A 1 1 0 0 1 1 0";

    let config = GraphConfig { x_min: -1.8, x_max: 1.8, y_min: -1.8, y_max: 1.8, show_grid: false, grid_step: 0.5 };

    let check_challenge = {
        let node_id = node_id.clone();
        move || {
            let idx = challenge_idx.get_untracked();
            let ch = challenges();
            if idx >= ch.len() { return; }
            if (ch[idx].check)(angle_deg.get_untracked()) {
                set_show_success.set(true);
                set_completed_count.update(|c| *c += 1);
                let next = idx + 1;
                let node_id = node_id.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    gloo_timers::future::sleep(std::time::Duration::from_millis(1500)).await;
                    set_show_success.set(false);
                    set_challenge_idx.set(next);
                    if next >= 5 {
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
                // Unit circle
                <path d=circle_path stroke="var(--text-secondary)" stroke-width="0.02" fill="none" />

                // Axes through the circle
                <line x1="-1.5" y1="0" x2="1.5" y2="0" class="graph-axis" />
                <line x1="0" y1="-1.5" x2="0" y2="1.5" class="graph-axis" />

                // Radius line from origin to point
                <line x1="0" y1="0" x2=move || point_x.get() y2=move || point_y.get()
                    stroke="var(--primary)" stroke-width="0.03" />

                // Sin projection (vertical line from point to x-axis)
                <line x1=move || point_x.get() y1=move || point_y.get() x2=move || point_x.get() y2="0"
                    stroke="var(--error)" stroke-width="0.03" stroke-dasharray="0.05 0.03" />

                // Cos projection (horizontal line from point to y-axis)
                <line x1="0" y1=move || point_y.get() x2=move || point_x.get() y2=move || point_y.get()
                    stroke="var(--info)" stroke-width="0.03" stroke-dasharray="0.05 0.03" />

                // Point on circle
                <circle cx=move || point_x.get() cy=move || point_y.get() r="0.06" class="game-point-vertex" />

                // Quadrant labels
                <text x="0.7" y="-0.7" font-size="0.12" fill="var(--text-tertiary)">"Q1"</text>
                <text x="-0.9" y="-0.7" font-size="0.12" fill="var(--text-tertiary)">"Q2"</text>
                <text x="-0.9" y="0.8" font-size="0.12" fill="var(--text-tertiary)">"Q3"</text>
                <text x="0.7" y="0.8" font-size="0.12" fill="var(--text-tertiary)">"Q4"</text>
            </GraphArea>

            // Values display
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 0.5rem; text-align: center; padding: 0.75rem 0">
                <div>
                    <div style="font-size: 0.7rem; color: var(--text-tertiary)">"Angle"</div>
                    <div style="font-size: 1.1rem; font-weight: 700">{move || format!("{:.0}\u{00b0}", angle_deg.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--error)">"sin \u{03b8}"</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: var(--error)">{move || format!("{:.3}", sin_val.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--info)">"cos \u{03b8}"</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: var(--info)">{move || format!("{:.3}", cos_val.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--warning)">"tan \u{03b8}"</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: var(--warning)">
                        {move || { let t = tan_val.get(); if t.is_nan() { "undef".to_string() } else { format!("{:.3}", t) } }}
                    </div>
                </div>
            </div>

            <div style="text-align: center; font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 0.5rem">
                "Quadrant: "{move || quadrant.get()}
            </div>

            // Angle slider
            <div class="game-sliders">
                <div class="game-slider-row">
                    <label class="game-slider-label">"\u{03b8} ="</label>
                    <input type="range" min="0" max="360" step="1" class="game-slider"
                        prop:value=move || angle_deg.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = leptos::prelude::event_target_value(&ev).parse::<f64>() { set_angle_deg.set(v); }
                        }
                    />
                    <span class="game-slider-value">{move || format!("{:.0}\u{00b0}", angle_deg.get())}</span>
                </div>
            </div>

            // Challenges
            <div class="game-challenge">
                {move || {
                    let idx = challenge_idx.get();
                    let ch = challenges();
                    if idx >= ch.len() {
                        view! { <div class="game-complete"><div class="game-complete-icon">"\u{2714}"</div><div class="game-complete-text">"All challenges complete!"</div></div> }.into_any()
                    } else if show_success.get() {
                        view! { <div class="game-success"><span class="game-success-icon">"\u{2714}"</span>" Correct!"</div> }.into_any()
                    } else {
                        let instruction = ch[idx].instruction;
                        let hint = ch[idx].hint;
                        let check = check_challenge.clone();
                        view! {
                            <div class="game-challenge-active">
                                <div class="game-challenge-number">"Challenge "{idx + 1}" of 5"</div>
                                <div class="game-challenge-instruction">{instruction}</div>
                                <div class="game-challenge-actions">
                                    <button class="praxis-filter-btn active" on:click=move |_| check()>"Check"</button>
                                </div>
                                <div class="game-challenge-hint">{hint}</div>
                            </div>
                        }.into_any()
                    }
                }}
            </div>
        </div>
    }
}
