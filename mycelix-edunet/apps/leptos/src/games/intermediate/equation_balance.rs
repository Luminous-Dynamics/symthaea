// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Equation Balance — Gr7-8. Solve equations by keeping both sides equal.
//! Visual balance scale metaphor for algebraic thinking.

use leptos::prelude::*;

#[component]
pub fn EquationBalance(node_id: String) -> impl IntoView {
    let (target_x, _set_target) = signal(5i32);
    let (coefficient, _set_coeff) = signal(2i32);
    let (constant, _set_const) = signal(3i32);
    let (rhs, _set_rhs) = signal(13i32); // 2x + 3 = 13

    let (user_steps, set_user_steps) = signal(Vec::<String>::new());
    let (current_lhs_x, set_lhs_x) = signal(2i32); // coefficient of x
    let (current_lhs_c, set_lhs_c) = signal(3i32); // constant on left
    let (current_rhs, set_rhs_val) = signal(13i32); // right side
    let (solved, set_solved) = signal(false);
    let (message, set_message) = signal(String::new());

    let is_balanced = move || {
        current_lhs_x.get() * target_x.get() + current_lhs_c.get() == current_rhs.get()
    };

    view! {
        <div class="game-container equation-balance-game">
            <h3>"Equation Balance"</h3>
            <p class="game-instructions">"Keep both sides equal! Find the value of x."</p>

            // Current equation display
            <div style="font-size: 1.8rem; text-align: center; margin: 1rem 0; font-weight: 700; font-family: monospace">
                {move || {
                    let cx = current_lhs_x.get();
                    let cc = current_lhs_c.get();
                    let r = current_rhs.get();
                    if cx == 1 && cc == 0 {
                        format!("x = {}", r)
                    } else if cc == 0 {
                        format!("{}x = {}", cx, r)
                    } else if cc > 0 {
                        format!("{}x + {} = {}", cx, cc, r)
                    } else {
                        format!("{}x - {} = {}", cx, -cc, r)
                    }
                }}
            </div>

            // Balance visualization
            <svg viewBox="0 0 400 150" class="game-svg" aria-label="Equation balance scale">
                // Fulcrum
                <polygon points="200,140 185,100 215,100" fill="var(--text-secondary)" />
                // Beam
                <line x1="50" y1="100" x2="350" y2="100" stroke="var(--text)" stroke-width="3" />
                // Left pan
                <rect x="60" y="80" width="120" height="20" rx="4" fill="var(--primary, #7c3aed)" opacity="0.3" />
                <text x="120" y="75" text-anchor="middle" font-size="14" fill="var(--primary)" font-weight="bold">
                    {move || {
                        let cx = current_lhs_x.get();
                        let cc = current_lhs_c.get();
                        if cc == 0 { format!("{}x", cx) }
                        else if cc > 0 { format!("{}x+{}", cx, cc) }
                        else { format!("{}x{}", cx, cc) }
                    }}
                </text>
                // Right pan
                <rect x="220" y="80" width="120" height="20" rx="4" fill="var(--success, #22c55e)" opacity="0.3" />
                <text x="280" y="75" text-anchor="middle" font-size="14" fill="var(--success)" font-weight="bold">
                    {move || current_rhs.get().to_string()}
                </text>
                // Balance indicator
                <text x="200" y="65" text-anchor="middle" font-size="20">
                    {move || if is_balanced() { "⚖️" } else { "⚠️" }}
                </text>
            </svg>

            // Operation buttons
            {move || if !solved.get() {
                view! {
                    <div class="game-controls" style="display: flex; flex-wrap: wrap; gap: 0.5rem; justify-content: center">
                        <button class="btn-secondary" on:click=move |_| {
                            let c = current_lhs_c.get();
                            set_lhs_c.set(0);
                            set_rhs_val.update(|r| *r -= c);
                            set_user_steps.update(|s| s.push(format!("Subtract {} from both sides", c)));
                            if current_lhs_x.get() == 1 {
                                set_solved.set(true);
                                set_message.set(format!("x = {}!", current_rhs.get()));
                            }
                        }>"Subtract " {move || current_lhs_c.get()} " from both sides"</button>

                        <button class="btn-secondary" on:click=move |_| {
                            let cx = current_lhs_x.get();
                            if cx != 0 {
                                set_lhs_x.set(1);
                                set_rhs_val.update(|r| *r /= cx);
                                set_user_steps.update(|s| s.push(format!("Divide both sides by {}", cx)));
                                if current_lhs_c.get() == 0 {
                                    set_solved.set(true);
                                    set_message.set(format!("x = {}!", current_rhs.get()));
                                }
                            }
                        }>"Divide both sides by " {move || current_lhs_x.get()}</button>
                    </div>
                }.into_any()
            } else {
                view! {
                    <div style="text-align: center; font-size: 1.5rem; color: var(--success); margin: 1rem 0">
                        "Solved! x = " {move || current_rhs.get()}
                    </div>
                }.into_any()
            }}

            // Steps taken
            <div class="game-steps" style="margin-top: 1rem">
                <h4>"Your steps:"</h4>
                <ol>
                    {move || user_steps.get().iter().map(|s| {
                        view! { <li>{s.clone()}</li> }
                    }).collect_view()}
                </ol>
            </div>

            <div class="game-message" role="status" aria-live="polite" style="text-align: center; color: var(--success)">
                {move || message.get()}
            </div>
        </div>
    }
}
