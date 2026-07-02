// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Times Tables Speed Challenge — Gr3-5. Beat your own record!

use leptos::prelude::*;

#[component]
pub fn TimesTablesGame(node_id: String) -> impl IntoView {
    let (table, set_table) = signal(2u32);
    let (current_q, set_current_q) = signal(1u32);
    let (answer, set_answer) = signal(String::new());
    let (score, set_score) = signal(0u32);
    let (total, set_total) = signal(0u32);
    let (message, set_message) = signal(String::new());
    let (streak, set_streak) = signal(0u32);

    let check = move || {
        let t = table.get();
        let q = current_q.get();
        let correct = t * q;
        let user_answer: u32 = answer.get().trim().parse().unwrap_or(0);

        set_total.update(|t| *t += 1);

        if user_answer == correct {
            set_score.update(|s| *s += 1);
            set_streak.update(|s| *s += 1);
            let s = streak.get() + 1;
            set_message.set(if s >= 5 {
                format!("{}! {} × {} = {} — {} in a row!", 
                    if s >= 10 { "AMAZING" } else { "Great" }, t, q, correct, s)
            } else {
                format!("{} × {} = {} ✓", t, q, correct)
            });
        } else {
            set_streak.set(0);
            set_message.set(format!("{} × {} = {} (you said {})", t, q, correct, user_answer));
        }

        // Next question
        let next = if q >= 12 { 1 } else { q + 1 };
        set_current_q.set(next);
        set_answer.set(String::new());
    };

    view! {
        <div class="game-container times-tables-game">
            <h3>"Times Tables Challenge"</h3>

            <div class="game-score">
                {move || format!("Score: {}/{}", score.get(), total.get())}
                {move || if streak.get() >= 3 {
                    format!(" | Streak: {}!", streak.get())
                } else { String::new() }}
            </div>

            <div class="times-question" style="font-size: 2.5rem; text-align: center; margin: 1.5rem 0; font-weight: 700">
                {move || table.get()} " × " {move || current_q.get()} " = "
                <input
                    type="number"
                    class="times-input"
                    style="width: 80px; font-size: 2rem; text-align: center; border: 2px solid var(--primary); border-radius: 8px; background: var(--surface); color: var(--text)"
                    prop:value=move || answer.get()
                    on:input=move |ev| {
                        use wasm_bindgen::JsCast;
                        let val = ev.target().unwrap().dyn_into::<web_sys::HtmlInputElement>().unwrap().value();
                        set_answer.set(val);
                    }
                    on:keydown=move |ev| {
                        if ev.key() == "Enter" { check(); }
                    }
                    autofocus=true
                />
            </div>

            <div class="game-message" role="status" aria-live="polite" style="text-align: center; min-height: 1.5rem">
                {move || message.get()}
            </div>

            <div class="game-controls" style="margin-top: 1rem">
                <label>"Table: "</label>
                {(2..=12).map(|t| {
                    view! {
                        <button
                            class=move || if table.get() == t { "btn-primary" } else { "btn-secondary" }
                            on:click=move |_| {
                                set_table.set(t);
                                set_current_q.set(1);
                                set_score.set(0);
                                set_total.set(0);
                                set_streak.set(0);
                                set_answer.set(String::new());
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
