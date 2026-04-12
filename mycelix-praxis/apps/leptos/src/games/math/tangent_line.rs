// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tangent Line Explorer — interactive calculus visualization.
//!
//! Students drag a point along a cubic curve and see the tangent line
//! drawn in real-time. The gradient value updates live.
//! Challenges: find turning points, identify increasing/decreasing intervals.

use leptos::prelude::*;
use crate::games::shared::graph_renderer::{GraphArea, GraphConfig};
use crate::curriculum::{use_set_progress, ProgressStatus};

/// f(x) = x³ - 3x + 1
fn f(x: f64) -> f64 { x * x * x - 3.0 * x + 1.0 }

/// f'(x) = 3x² - 3
fn f_prime(x: f64) -> f64 { 3.0 * x * x - 3.0 }

fn curve_path(x_min: f64, x_max: f64) -> String {
    let steps = 200;
    let dx = (x_max - x_min) / steps as f64;
    let mut path = String::new();
    for i in 0..=steps {
        let x = x_min + i as f64 * dx;
        let y = -f(x);
        if y < -12.0 || y > 12.0 { continue; }
        if path.is_empty() { path = format!("M {:.2} {:.2}", x, y); }
        else { path.push_str(&format!(" L {:.2} {:.2}", x, y)); }
    }
    path
}

fn tangent_line_path(x0: f64, x_min: f64, x_max: f64) -> String {
    let y0 = f(x0);
    let m = f_prime(x0);
    // Tangent line: y - y0 = m(x - x0) => y = m*x - m*x0 + y0
    let x1 = x_min.max(x0 - 4.0);
    let x2 = x_max.min(x0 + 4.0);
    let y1 = m * (x1 - x0) + y0;
    let y2 = m * (x2 - x0) + y0;
    format!("M {:.2} {:.2} L {:.2} {:.2}", x1, -y1, x2, -y2)
}

struct Challenge {
    instruction: &'static str,
    check: fn(f64) -> bool,
    hint: &'static str,
}

fn challenges() -> Vec<Challenge> {
    vec![
        Challenge {
            instruction: "Find a point where the gradient is zero (a turning point)",
            check: |x| f_prime(x).abs() < 0.3,
            hint: "Turning points are where f'(x) = 0. Try x = 1 or x = -1.",
        },
        Challenge {
            instruction: "Find a point where the gradient is positive (function increasing)",
            check: |x| f_prime(x) > 1.0,
            hint: "The function increases when f'(x) > 0. Try x > 1 or x < -1.",
        },
        Challenge {
            instruction: "Find a point where the gradient is negative (function decreasing)",
            check: |x| f_prime(x) < -1.0,
            hint: "The function decreases when f'(x) < 0. Try -1 < x < 1.",
        },
        Challenge {
            instruction: "Find the local maximum (turning point with gradient changing from + to -)",
            check: |x| (x - (-1.0)).abs() < 0.3,
            hint: "The local max is at x = -1 where f'(x) changes from positive to negative.",
        },
    ]
}

#[component]
pub fn TangentLineExplorer(node_id: String) -> impl IntoView {
    let set_progress = use_set_progress();
    let (x_pos, set_x_pos) = signal(0.0_f64);
    let (challenge_idx, set_challenge_idx) = signal(0_usize);
    let (show_success, set_show_success) = signal(false);
    let (completed_count, set_completed_count) = signal(0_u32);

    let gradient = Memo::new(move |_| f_prime(x_pos.get()));
    let point_y = Memo::new(move |_| f(x_pos.get()));
    let tangent_path = Memo::new(move |_| tangent_line_path(x_pos.get(), -5.0, 5.0));

    let gradient_color = Memo::new(move |_| {
        let g = gradient.get();
        if g.abs() < 0.3 { "var(--warning)" }
        else if g > 0.0 { "var(--success)" }
        else { "var(--error)" }
    });

    let static_curve = curve_path(-5.0, 5.0);
    let config = GraphConfig { x_min: -5.0, x_max: 5.0, y_min: -5.0, y_max: 5.0, show_grid: true, grid_step: 1.0 };

    let check_challenge = {
        let node_id = node_id.clone();
        move || {
            let idx = challenge_idx.get_untracked();
            let ch = challenges();
            if idx >= ch.len() { return; }
            if (ch[idx].check)(x_pos.get_untracked()) {
                set_show_success.set(true);
                set_completed_count.update(|c| *c += 1);
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
                <path d=static_curve.clone() class="game-curve" fill="none" />
                <path d=move || tangent_path.get() stroke="var(--warning)" stroke-width="0.08" fill="none" stroke-dasharray="0.2 0.1" />
                <circle cx=move || x_pos.get() cy=move || -point_y.get() r="0.2" class="game-point-vertex" />
            </GraphArea>

            <div class="game-equation">
                "f(x) = x\u{00b3} \u{2212} 3x + 1"
            </div>

            <div style="text-align: center; margin-bottom: 0.5rem">
                <span style="font-size: 0.9rem">"Gradient at x = "{move || format!("{:.1}", x_pos.get())}": "</span>
                <span style=move || format!("font-size: 1.2rem; font-weight: 700; color: {}", gradient_color.get())>
                    {move || format!("{:.2}", gradient.get())}
                </span>
            </div>

            <div class="game-sliders">
                <div class="game-slider-row">
                    <label class="game-slider-label">"x ="</label>
                    <input type="range" min="-4" max="4" step="0.05" class="game-slider"
                        prop:value=move || x_pos.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = leptos::prelude::event_target_value(&ev).parse::<f64>() { set_x_pos.set(v); }
                        }
                    />
                    <span class="game-slider-value">{move || format!("{:.1}", x_pos.get())}</span>
                </div>
            </div>

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
                                <div class="game-challenge-number">"Challenge "{idx + 1}" of 4"</div>
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
