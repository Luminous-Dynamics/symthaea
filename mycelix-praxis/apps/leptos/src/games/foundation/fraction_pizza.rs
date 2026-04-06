// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fraction Pizza Game — Gr3-5. Visualise fractions as pizza slices.
//! Tap slices to shade, match the target fraction.

use leptos::prelude::*;

#[component]
pub fn FractionPizzaGame(node_id: String) -> impl IntoView {
    let (denominator, set_denominator) = signal(4u32);
    let (target_numerator, set_target) = signal(1u32);
    let (shaded, set_shaded) = signal(Vec::<u32>::new());
    let (message, set_message) = signal(String::new());

    let check_answer = move || {
        let num_shaded = shaded.get().len() as u32;
        let target = target_numerator.get();
        if num_shaded == target {
            set_message.set(format!("Correct! {}/{} of the pizza!", target, denominator.get()));
        }
    };

    view! {
        <div class="game-container fraction-pizza-game">
            <h3>"Fraction Pizza"</h3>
            <p class="game-instructions">
                "Shade " {move || target_numerator.get()} "/" {move || denominator.get()} " of the pizza!"
            </p>

            <svg viewBox="0 0 300 300" class="game-svg" aria-label="Fraction pizza game">
                // Pizza circle
                <circle cx="150" cy="150" r="120" fill="#f59e0b" stroke="#92400e" stroke-width="3" />

                // Slices
                {move || {
                    let d = denominator.get();
                    let shaded_slices = shaded.get();
                    (0..d).map(|i| {
                        let angle_start = (i as f64) * std::f64::consts::TAU / (d as f64) - std::f64::consts::FRAC_PI_2;
                        let angle_end = ((i + 1) as f64) * std::f64::consts::TAU / (d as f64) - std::f64::consts::FRAC_PI_2;
                        let x1 = 150.0 + 120.0 * angle_start.cos();
                        let y1 = 150.0 + 120.0 * angle_start.sin();
                        let x2 = 150.0 + 120.0 * angle_end.cos();
                        let y2 = 150.0 + 120.0 * angle_end.sin();
                        let large_arc = if d <= 2 { 1 } else { 0 };
                        let is_shaded = shaded_slices.contains(&i);

                        let path = format!(
                            "M 150 150 L {} {} A 120 120 0 {} 1 {} {} Z",
                            x1, y1, large_arc, x2, y2
                        );

                        view! {
                            <path
                                d=path
                                fill=if is_shaded { "#ef4444" } else { "#f59e0b" }
                                stroke="#92400e"
                                stroke-width="2"
                                style="cursor: pointer"
                                on:click=move |_| {
                                    set_shaded.update(|s| {
                                        if s.contains(&i) { s.retain(|&x| x != i); }
                                        else { s.push(i); }
                                    });
                                    check_answer();
                                }
                            />
                        }
                    }).collect_view()
                }}

                // Center dot
                <circle cx="150" cy="150" r="8" fill="#92400e" />
            </svg>

            <div class="game-message" role="status" aria-live="polite">
                {move || message.get()}
            </div>

            <div class="game-controls">
                <label>"Slices: "</label>
                {[2, 3, 4, 6, 8].iter().map(|&d| {
                    view! {
                        <button
                            class=move || if denominator.get() == d { "btn-primary" } else { "btn-secondary" }
                            on:click=move |_| {
                                set_denominator.set(d);
                                set_target.set(1.min(d));
                                set_shaded.set(Vec::new());
                                set_message.set(String::new());
                            }
                        >
                            {d.to_string()}
                        </button>
                    }
                }).collect_view()}
                <label style="margin-left: 1rem">"Show: "</label>
                {move || {
                    let d = denominator.get();
                    (1..d).map(|n| {
                        view! {
                            <button
                                class=move || if target_numerator.get() == n { "btn-primary" } else { "btn-secondary" }
                                on:click=move |_| {
                                    set_target.set(n);
                                    set_shaded.set(Vec::new());
                                    set_message.set(String::new());
                                }
                            >
                                {format!("{}/{}", n, d)}
                            </button>
                        }
                    }).collect_view()
                }}
            </div>
        </div>
    }
}
