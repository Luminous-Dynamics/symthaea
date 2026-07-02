// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Parabola Explorer — interactive function graph game.
//!
//! Students manipulate sliders for a, p, q to explore y = a(x-p)² + q.
//! Five challenges guide discovery of parameter effects.
//! Proven mechanic for building function intuition (PhET/Desmos research).

use leptos::prelude::*;
use crate::games::shared::graph_renderer::{GraphArea, GraphConfig};
use crate::curriculum::{use_set_progress, ProgressStatus};
use crate::social_proof;

/// Generate SVG path data for a parabola y = a(x-p)² + q
fn parabola_path(a: f64, p: f64, q: f64, x_min: f64, x_max: f64) -> String {
    let steps = 200;
    let dx = (x_max - x_min) / steps as f64;
    let mut path = String::new();

    for i in 0..=steps {
        let x = x_min + i as f64 * dx;
        let y = a * (x - p).powi(2) + q;
        let svg_y = -y; // SVG y-axis is inverted

        if svg_y < -12.0 || svg_y > 12.0 { continue; } // clip

        if path.is_empty() {
            path = format!("M {:.2} {:.2}", x, svg_y);
        } else {
            path.push_str(&format!(" L {:.2} {:.2}", x, svg_y));
        }
    }
    path
}

struct Challenge {
    instruction: &'static str,
    check: fn(f64, f64, f64) -> bool,
    hint: &'static str,
}

fn challenges() -> Vec<Challenge> {
    vec![
        Challenge {
            instruction: "Make the parabola open downward",
            check: |a, _, _| a < 0.0,
            hint: "What happens when 'a' is negative?",
        },
        Challenge {
            instruction: "Move the vertex to (2, 3)",
            check: |_, p, q| (p - 2.0).abs() < 0.3 && (q - 3.0).abs() < 0.3,
            hint: "The vertex is at (p, q). Adjust the p and q sliders.",
        },
        Challenge {
            instruction: "Make a narrow parabola (steep sides) opening upward",
            check: |a, _, _| a > 2.0,
            hint: "A larger |a| makes the parabola narrower. Try a > 2.",
        },
        Challenge {
            instruction: "Make a wide, flat parabola opening downward",
            check: |a, _, _| a < 0.0 && a > -0.5,
            hint: "A small |a| makes it wider. Try a between -0.5 and 0.",
        },
        Challenge {
            instruction: "Place the vertex at (-1, -2) opening upward",
            check: |a, p, q| a > 0.0 && (p - (-1.0)).abs() < 0.3 && (q - (-2.0)).abs() < 0.3,
            hint: "Set p = -1, q = -2, and make sure a > 0.",
        },
    ]
}

#[component]
pub fn ParabolaExplorer(node_id: String) -> impl IntoView {
    let set_progress = use_set_progress();

    // Parameters
    let (a, set_a) = signal(1.0_f64);
    let (p, set_p) = signal(0.0_f64);
    let (q, set_q) = signal(0.0_f64);

    // Challenge state
    let (challenge_idx, set_challenge_idx) = signal(0_usize);
    let (show_success, set_show_success) = signal(false);
    let (show_error_msg, set_show_error_msg) = signal(String::new());
    let (attempt_count, set_attempt_count) = signal(0_u32);
    let (completed_count, set_completed_count) = signal(0_u32);
    let all_challenges = challenges();
    let total_challenges = all_challenges.len();

    // Derived: SVG path
    let curve_path = Memo::new(move |_| {
        parabola_path(a.get(), p.get(), q.get(), -10.0, 10.0)
    });

    // Derived: vertex position (SVG coords)
    let vertex_x = Memo::new(move |_| p.get());
    let vertex_y = Memo::new(move |_| -q.get()); // SVG inverted

    // Derived: axis of symmetry
    let axis_x = Memo::new(move |_| p.get());

    // Derived: y-intercept
    let y_intercept = Memo::new(move |_| {
        let av = a.get();
        let pv = p.get();
        let qv = q.get();
        av * pv * pv + qv
    });

    // Derived: equation text
    let equation = Memo::new(move |_| {
        let av = a.get();
        let pv = p.get();
        let qv = q.get();
        let p_sign = if pv >= 0.0 { format!("- {:.1}", pv) } else { format!("+ {:.1}", -pv) };
        let q_sign = if qv >= 0.0 { format!("+ {:.1}", qv) } else { format!("- {:.1}", -qv) };
        format!("y = {:.1}(x {})² {}", av, p_sign, q_sign)
    });

    // Check challenge
    let check_challenge = {
        let node_id = node_id.clone();
        move || {
            let idx = challenge_idx.get_untracked();
            let challenges = challenges();
            if idx >= challenges.len() { return; }

            let passed = (challenges[idx].check)(
                a.get_untracked(),
                p.get_untracked(),
                q.get_untracked(),
            );

            set_attempt_count.update(|c| *c += 1);

            if passed {
                set_show_error_msg.set(String::new());
                set_show_success.set(true);
                set_completed_count.update(|c| *c += 1);

                // Auto-advance after 1.5 seconds
                let next_idx = idx + 1;
                let node_id = node_id.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    gloo_timers::future::sleep(std::time::Duration::from_millis(1500)).await;
                    set_show_success.set(false);
                    set_challenge_idx.set(next_idx);

                    // If all challenges complete, update mastery
                    if next_idx >= 5 {
                        set_progress.update(|p| {
                            let entry = p.nodes.entry(node_id).or_default();
                            entry.mastery_permille = entry.mastery_permille.max(800);
                            if entry.status == ProgressStatus::NotStarted {
                                entry.status = ProgressStatus::Studying;
                            }
                        });
                    }
                });
            } else {
                // Wise feedback on failure
                let att = attempt_count.get_untracked();
                let msg = social_proof::wise_error_message("function parameters", att, true);
                set_show_error_msg.set(msg);
                // Auto-dismiss after 3 seconds
                wasm_bindgen_futures::spawn_local(async move {
                    gloo_timers::future::sleep(std::time::Duration::from_millis(3000)).await;
                    set_show_error_msg.set(String::new());
                });
            }
        }
    };

    let reset = move || {
        set_a.set(1.0);
        set_p.set(0.0);
        set_q.set(0.0);
    };

    let config = GraphConfig::default();

    view! {
        <div class="game-container">
            // Graph
            <GraphArea config=config>
                // The parabola curve
                <path
                    d=move || curve_path.get()
                    class="game-curve"
                    fill="none"
                />

                // Axis of symmetry (dashed)
                <line
                    x1=move || axis_x.get()
                    y1="-10"
                    x2=move || axis_x.get()
                    y2="10"
                    class="game-dashed-line"
                />

                // Vertex point
                <circle
                    cx=move || vertex_x.get()
                    cy=move || vertex_y.get()
                    r="0.25"
                    class="game-point-vertex"
                />

                // Vertex label
                <text
                    x=move || vertex_x.get() + 0.4
                    y=move || vertex_y.get() - 0.4
                    class="game-point-label"
                >
                    {move || format!("({:.1}, {:.1})", p.get(), q.get())}
                </text>

                // Y-intercept point
                <circle
                    cx="0"
                    cy=move || -y_intercept.get()
                    r="0.2"
                    class="game-point-intercept"
                />
            </GraphArea>

            // Equation display
            <div class="game-equation">
                {move || equation.get()}
            </div>

            // Sliders
            <div class="game-sliders">
                <div class="game-slider-row">
                    <label class="game-slider-label">"a ="</label>
                    <input
                        type="range"
                        min="-3" max="3" step="0.1"
                        class="game-slider"
                        prop:value=move || a.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = leptos::prelude::event_target_value(&ev).parse::<f64>() {
                                set_a.set(v);
                            }
                        }
                    />
                    <span class="game-slider-value">{move || format!("{:.1}", a.get())}</span>
                </div>
                <div class="game-slider-row">
                    <label class="game-slider-label">"p ="</label>
                    <input
                        type="range"
                        min="-5" max="5" step="0.1"
                        class="game-slider"
                        prop:value=move || p.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = leptos::prelude::event_target_value(&ev).parse::<f64>() {
                                set_p.set(v);
                            }
                        }
                    />
                    <span class="game-slider-value">{move || format!("{:.1}", p.get())}</span>
                </div>
                <div class="game-slider-row">
                    <label class="game-slider-label">"q ="</label>
                    <input
                        type="range"
                        min="-5" max="5" step="0.1"
                        class="game-slider"
                        prop:value=move || q.get().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = leptos::prelude::event_target_value(&ev).parse::<f64>() {
                                set_q.set(v);
                            }
                        }
                    />
                    <span class="game-slider-value">{move || format!("{:.1}", q.get())}</span>
                </div>
            </div>

            // Challenge section
            <div class="game-challenge">
                {move || {
                    let idx = challenge_idx.get();
                    let challenges = challenges();

                    if idx >= challenges.len() {
                        let count = completed_count.get();
                        view! {
                            <div class="game-complete">
                                <div class="game-complete-icon">"\u{2714}"</div>
                                <div class="game-complete-text">
                                    "All challenges complete! "{count}"/" {total_challenges}" solved."
                                </div>
                            </div>
                        }.into_any()
                    } else if show_success.get() {
                        view! {
                            <div class="game-success">
                                <span class="game-success-icon">"\u{2714}"</span>
                                " Correct!"
                            </div>
                        }.into_any()
                    } else {
                        let instruction = challenges[idx].instruction;
                        let hint = challenges[idx].hint;
                        let check = check_challenge.clone();
                        view! {
                            <div class="game-challenge-active">
                                <div class="game-challenge-number">
                                    "Challenge "{idx + 1}" of "{total_challenges}
                                </div>
                                <div class="game-challenge-instruction">{instruction}</div>
                                <div class="game-challenge-actions">
                                    <button class="praxis-filter-btn active" on:click=move |_| check()>
                                        "Check"
                                    </button>
                                    <button class="praxis-filter-btn" on:click=move |_| reset()>
                                        "Reset"
                                    </button>
                                </div>
                                <div class="game-challenge-hint">{hint}</div>
                                // Wise feedback on incorrect attempt
                                {move || {
                                    let msg = show_error_msg.get();
                                    if msg.is_empty() {
                                        view! { <span></span> }.into_any()
                                    } else {
                                        view! {
                                            <div style="margin-top: 0.75rem; padding: 0.5rem 0.75rem; border-left: 3px solid var(--info); background: rgba(59, 130, 246, 0.05); border-radius: 0 6px 6px 0; font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5">
                                                {msg}
                                            </div>
                                        }.into_any()
                                    }
                                }}
                            </div>
                        }.into_any()
                    }
                }}
            </div>
        </div>
    }
}
