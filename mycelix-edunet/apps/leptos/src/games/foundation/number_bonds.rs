// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Number Bonds Game — Gr1-3. Find pairs that add to 10 (or other targets).
//! Interactive SVG with draggable number cards.

use leptos::prelude::*;

#[component]
pub fn NumberBondsGame(node_id: String) -> impl IntoView {
    let (target, set_target) = signal(10u32);
    let (score, set_score) = signal(0u32);
    let (selected, set_selected) = signal(Option::<u32>::None);
    let (message, set_message) = signal(String::new());
    let (pairs_found, set_pairs_found) = signal(Vec::<(u32, u32)>::new());

    let numbers: Vec<u32> = (0..=10).collect();

    view! {
        <div class="game-container number-bonds-game">
            <h3>"Number Bonds of " {move || target.get()}</h3>
            <p class="game-instructions">
                "Tap two numbers that add up to " {move || target.get()} "!"
            </p>

            <div class="game-score">
                "Score: " {move || score.get()} " / 5"
            </div>

            <svg viewBox="0 0 400 300" class="game-svg" aria-label="Number bonds interactive game">
                // Number cards arranged in a circle
                {numbers.iter().map(|&n| {
                    let x = 200.0 + 140.0 * ((n as f64) * std::f64::consts::TAU / 11.0).cos();
                    let y = 150.0 + 110.0 * ((n as f64) * std::f64::consts::TAU / 11.0).sin();
                    let is_found = move || {
                        pairs_found.get().iter().any(|(a,b)| *a == n || *b == n)
                    };
                    let is_selected = move || selected.get() == Some(n);

                    view! {
                        <g
                            class="number-card"
                            style="cursor: pointer"
                            on:click=move |_| {
                                if is_found() { return; }
                                match selected.get() {
                                    None => {
                                        set_selected.set(Some(n));
                                        set_message.set(format!("{} + ? = {}", n, target.get()));
                                    }
                                    Some(first) => {
                                        if first == n {
                                            set_selected.set(None);
                                            set_message.set(String::new());
                                        } else if first + n == target.get() {
                                            set_pairs_found.update(|p| p.push((first, n)));
                                            set_score.update(|s| *s += 1);
                                            set_selected.set(None);
                                            set_message.set(format!("{} + {} = {}!", first, n, target.get()));
                                        } else {
                                            set_selected.set(None);
                                            set_message.set(format!("{} + {} = {}. Try again!", first, n, first + n));
                                        }
                                    }
                                }
                            }
                        >
                            <circle
                                cx=x cy=y r="24"
                                fill=move || {
                                    if is_found() { "var(--success, #22c55e)" }
                                    else if is_selected() { "var(--primary, #7c3aed)" }
                                    else { "var(--surface, #1e293b)" }
                                }
                                stroke="var(--border, #334155)"
                                stroke-width="2"
                            />
                            <text
                                x=x y=y
                                text-anchor="middle" dominant-baseline="central"
                                font-size="18" font-weight="bold"
                                fill=move || {
                                    if is_found() || is_selected() { "#fff" } else { "var(--text, #e2e8f0)" }
                                }
                            >
                                {n.to_string()}
                            </text>
                        </g>
                    }
                }).collect_view()}

                // Message display
                <text x="200" y="280" text-anchor="middle" font-size="16" fill="var(--primary, #7c3aed)">
                    {move || message.get()}
                </text>
            </svg>

            // Target selector
            <div class="game-controls">
                <label>"Bonds of: "</label>
                {[5, 10, 15, 20].iter().map(|&t| {
                    view! {
                        <button
                            class=move || if target.get() == t { "btn-primary" } else { "btn-secondary" }
                            on:click=move |_| {
                                set_target.set(t);
                                set_score.set(0);
                                set_selected.set(None);
                                set_pairs_found.set(Vec::new());
                                set_message.set(String::new());
                            }
                        >
                            {t.to_string()}
                        </button>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
