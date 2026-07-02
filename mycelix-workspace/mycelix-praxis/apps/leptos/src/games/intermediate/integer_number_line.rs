// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integer Number Line — Gr6. Visualise positive and negative numbers.
//! Drag a marker to answer addition/subtraction with integers.

use leptos::prelude::*;

#[component]
pub fn IntegerNumberLine(node_id: String) -> impl IntoView {
    let (a, set_a) = signal(3i32);
    let (b, set_b) = signal(-5i32);
    let (user_answer, set_user_answer) = signal(0i32);
    let (message, set_message) = signal(String::new());
    let (score, set_score) = signal(0u32);

    let correct_answer = move || a.get() + b.get();
    let range = -10..=10;

    let new_question = move || {
        let new_a = (js_sys::Math::random() * 15.0) as i32 - 7;
        let new_b = (js_sys::Math::random() * 15.0) as i32 - 7;
        set_a.set(new_a);
        set_b.set(new_b);
        set_user_answer.set(0);
        set_message.set(String::new());
    };

    view! {
        <div class="game-container integer-line-game">
            <h3>"Integer Number Line"</h3>

            <div class="game-question" style="font-size: 1.8rem; text-align: center; margin: 1rem 0; font-weight: 700">
                "(" {move || a.get()} ") + (" {move || b.get()} ") = ?"
            </div>

            <svg viewBox="0 0 500 120" class="game-svg" aria-label="Number line from -10 to 10">
                // Number line
                <line x1="25" y1="60" x2="475" y2="60" stroke="var(--text-secondary)" stroke-width="2" />

                // Tick marks and labels
                {(-10..=10).map(|n| {
                    let x = 25.0 + (n + 10) as f64 * 21.4;
                    let is_zero = n == 0;
                    view! {
                        <line x1=x y1="50" x2=x y2="70"
                            stroke=if is_zero { "var(--primary)" } else { "var(--text-secondary)" }
                            stroke-width=if is_zero { "3" } else { "1" }
                        />
                        <text x=x y="90" text-anchor="middle" font-size="10"
                            fill=if is_zero { "var(--primary)" } else { "var(--text-secondary)" }
                            font-weight=if is_zero { "bold" } else { "normal" }
                        >
                            {n.to_string()}
                        </text>
                    }
                }).collect_view()}

                // Start point (a)
                {move || {
                    let ax = 25.0 + (a.get() + 10) as f64 * 21.4;
                    view! {
                        <circle cx=ax cy="60" r="8" fill="var(--info, #3b82f6)" />
                        <text x=ax y="35" text-anchor="middle" font-size="12" fill="var(--info)" font-weight="bold">
                            {"Start: "}{a.get().to_string()}
                        </text>
                    }
                }}

                // User answer marker (draggable via click)
                {move || {
                    let ux = 25.0 + (user_answer.get() + 10) as f64 * 21.4;
                    view! {
                        <circle cx=ux cy="60" r="10" fill="var(--warning, #f59e0b)" stroke="var(--text)" stroke-width="2" />
                        <text x=ux y="110" text-anchor="middle" font-size="11" fill="var(--warning)" font-weight="bold">
                            {"Answer: "}{user_answer.get().to_string()}
                        </text>
                    }
                }}
            </svg>

            // Answer controls
            <div class="game-controls" style="display: flex; justify-content: center; gap: 0.5rem; margin: 0.5rem 0">
                <button class="btn-secondary" on:click=move |_| set_user_answer.update(|a| *a = (*a - 1).max(-10))>"◀ Left"</button>
                <button class="btn-primary" on:click=move |_| {
                    if user_answer.get() == correct_answer() {
                        set_score.update(|s| *s += 1);
                        set_message.set(format!("Correct! ({}) + ({}) = {}", a.get(), b.get(), correct_answer()));
                        new_question();
                    } else {
                        set_message.set(format!("Not quite. Try moving to {}.", correct_answer()));
                    }
                }>"Check"</button>
                <button class="btn-secondary" on:click=move |_| set_user_answer.update(|a| *a = (*a + 1).min(10))>"Right ▶"</button>
            </div>

            <div class="game-message" role="status" aria-live="polite" style="text-align: center">
                {move || message.get()}
            </div>
            <div class="game-score" style="text-align: center">"Score: " {move || score.get()}</div>
        </div>
    }
}
