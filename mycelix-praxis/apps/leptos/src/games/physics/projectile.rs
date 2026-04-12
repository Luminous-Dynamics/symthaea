// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Projectile Motion Simulator — vertical projectile under gravity.
//!
//! Students set initial velocity and observe position, velocity,
//! and acceleration graphs in real-time. Covers CAPS Gr12 P1.MECH2.

use leptos::prelude::*;
use crate::games::shared::graph_renderer::{GraphArea, GraphConfig};
use crate::curriculum::{use_set_progress, ProgressStatus};

const G: f64 = 9.8; // m/s²

/// Calculate height at time t for vertical throw: h = v₀t - ½gt²
fn height(v0: f64, t: f64) -> f64 {
    v0 * t - 0.5 * G * t * t
}

/// Calculate velocity at time t: v = v₀ - gt
fn velocity(v0: f64, t: f64) -> f64 {
    v0 - G * t
}

/// Time to reach max height: t = v₀/g
fn time_to_peak(v0: f64) -> f64 {
    v0 / G
}

/// Maximum height: h = v₀²/(2g)
fn max_height(v0: f64) -> f64 {
    v0 * v0 / (2.0 * G)
}

/// Total flight time: t = 2v₀/g
fn total_time(v0: f64) -> f64 {
    2.0 * v0 / G
}

fn trajectory_path(v0: f64, t_max: f64) -> String {
    let steps = 100;
    let dt = t_max / steps as f64;
    let mut path = String::new();
    for i in 0..=steps {
        let t = i as f64 * dt;
        let h = height(v0, t);
        if h < 0.0 && i > 0 { break; }
        // Map: x = time (0 to t_max), y = height (SVG inverted)
        let svg_x = t;
        let svg_y = -h;
        if path.is_empty() { path = format!("M {:.3} {:.3}", svg_x, svg_y); }
        else { path.push_str(&format!(" L {:.3} {:.3}", svg_x, svg_y)); }
    }
    path
}

#[component]
pub fn ProjectileExplorer(node_id: String) -> impl IntoView {
    let set_progress = use_set_progress();
    let (v0, set_v0) = signal(20.0_f64);
    let (time_slider, set_time_slider) = signal(0.0_f64);
    let (challenge_idx, set_challenge_idx) = signal(0_usize);
    let (show_success, set_show_success) = signal(false);

    let t_total = Memo::new(move |_| total_time(v0.get()));
    let h_max = Memo::new(move |_| max_height(v0.get()));
    let t_peak = Memo::new(move |_| time_to_peak(v0.get()));

    let current_h = Memo::new(move |_| {
        let t = time_slider.get() * t_total.get();
        height(v0.get(), t).max(0.0)
    });

    let current_v = Memo::new(move |_| {
        let t = time_slider.get() * t_total.get();
        velocity(v0.get(), t)
    });

    let path = Memo::new(move |_| {
        trajectory_path(v0.get(), t_total.get())
    });

    // Graph config: x = time (0 to t_total), y = height (0 to h_max)
    let graph_config = Memo::new(move |_| {
        let tt = t_total.get().max(1.0);
        let hm = h_max.get().max(5.0);
        GraphConfig {
            x_min: -0.2, x_max: tt + 0.5,
            y_min: -(hm + 2.0), y_max: 2.0,
            show_grid: true, grid_step: if tt > 6.0 { 2.0 } else { 1.0 },
        }
    });

    let check_challenge = {
        let node_id = node_id.clone();
        move || {
            let idx = challenge_idx.get_untracked();
            let v = v0.get_untracked();
            let passed = match idx {
                0 => (max_height(v) - 20.0).abs() < 1.5,       // Max height ≈ 20m
                1 => (total_time(v) - 4.0).abs() < 0.3,        // Total time ≈ 4s
                2 => (v - 30.0).abs() < 2.0,                   // v₀ = 30 m/s (discover relationship)
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
                    if next >= 3 {
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
            // Height-time graph
            {move || {
                let cfg = graph_config.get();
                let path_d = path.get();
                let t_now = time_slider.get() * t_total.get();
                let h_now = current_h.get();
                view! {
                    <GraphArea config=cfg>
                        // Trajectory curve
                        <path d=path_d class="game-curve" fill="none" />

                        // Current position marker
                        <circle cx=t_now cy={-h_now} r="0.3" class="game-point-vertex" />

                        // Axis labels
                        <text x={t_total.get() / 2.0} y="1.5" font-size="0.5" fill="var(--text-secondary)" text-anchor="middle">"time (s)"</text>
                    </GraphArea>
                }
            }}

            // Values display
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 0.5rem; text-align: center; padding: 0.75rem 0">
                <div>
                    <div style="font-size: 0.65rem; color: var(--text-tertiary)">"Height"</div>
                    <div style="font-size: 1rem; font-weight: 700; color: var(--info)">{move || format!("{:.1}m", current_h.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.65rem; color: var(--text-tertiary)">"Velocity"</div>
                    <div style=move || format!("font-size: 1rem; font-weight: 700; color: {}", if current_v.get() >= 0.0 { "var(--success)" } else { "var(--error)" })>
                        {move || format!("{:.1}m/s", current_v.get())}
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.65rem; color: var(--text-tertiary)">"Max Height"</div>
                    <div style="font-size: 1rem; font-weight: 700">{move || format!("{:.1}m", h_max.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.65rem; color: var(--text-tertiary)">"Total Time"</div>
                    <div style="font-size: 1rem; font-weight: 700">{move || format!("{:.2}s", t_total.get())}</div>
                </div>
            </div>

            // Sliders
            <div class="game-sliders">
                <div class="game-slider-row">
                    <label class="game-slider-label">"v\u{2080}"</label>
                    <input type="range" min="5" max="50" step="0.5" class="game-slider"
                        prop:value=move || v0.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_v0.set(v); } }
                    />
                    <span class="game-slider-value">{move || format!("{:.0}", v0.get())}"m/s"</span>
                </div>
                <div class="game-slider-row">
                    <label class="game-slider-label">"t"</label>
                    <input type="range" min="0" max="1" step="0.005" class="game-slider"
                        prop:value=move || time_slider.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_time_slider.set(v); } }
                    />
                    <span class="game-slider-value">{move || format!("{:.1}s", time_slider.get() * t_total.get())}</span>
                </div>
            </div>

            // Acceleration note
            <div style="text-align: center; font-size: 0.8rem; color: var(--text-tertiary); margin: 0.5rem 0">
                "Acceleration = \u{2212}9.8 m/s\u{00b2} (constant, always downward)"
            </div>

            // Challenges
            <div class="game-challenge">
                {move || {
                    let idx = challenge_idx.get();
                    if idx >= 3 {
                        view! { <div class="game-complete"><div class="game-complete-icon">"\u{2714}"</div><div class="game-complete-text">"All challenges complete!"</div></div> }.into_any()
                    } else if show_success.get() {
                        view! { <div class="game-success"><span class="game-success-icon">"\u{2714}"</span>" Correct!"</div> }.into_any()
                    } else {
                        let instructions = [
                            "Set the initial velocity so the ball reaches a maximum height of 20m",
                            "Set the initial velocity so the total flight time is 4 seconds",
                            "Set v\u{2080} = 30 m/s and observe: what is the velocity at the peak?",
                        ];
                        let hints = [
                            "h_max = v\u{2080}\u{00b2}/(2g). Solve: 20 = v\u{2080}\u{00b2}/19.6",
                            "Total time = 2v\u{2080}/g. Solve: 4 = 2v\u{2080}/9.8",
                            "At the peak, the ball momentarily stops. v = 0 m/s at t = v\u{2080}/g",
                        ];
                        let check = check_challenge.clone();
                        view! {
                            <div class="game-challenge-active">
                                <div class="game-challenge-number">"Challenge "{idx + 1}" of 3"</div>
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
