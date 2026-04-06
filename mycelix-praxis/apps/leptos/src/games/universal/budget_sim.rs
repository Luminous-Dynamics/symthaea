// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Budget Simulator — interactive financial literacy game.
//!
//! The student allocates a monthly income across categories (needs, wants, savings)
//! and sees the impact of their choices via a visual budget chart. Teaches the
//! 50/30/20 rule and compound interest fundamentals.

use leptos::prelude::*;

#[component]
pub fn BudgetSimulator(node_id: String) -> impl IntoView {
    let income = 10_000.0_f64; // R10,000 (SA Rand, relatable)
    let (needs, set_needs) = signal(50_u32); // percentage
    let (wants, set_wants) = signal(30_u32);
    let savings = Memo::new(move |_| 100_u32.saturating_sub(needs.get() + wants.get()));

    let needs_amount = Memo::new(move |_| income * needs.get() as f64 / 100.0);
    let wants_amount = Memo::new(move |_| income * wants.get() as f64 / 100.0);
    let savings_amount = Memo::new(move |_| income * savings.get() as f64 / 100.0);

    // Compound interest: savings invested at 8% for 10 years
    let future_value = Memo::new(move |_| {
        let monthly = savings_amount.get();
        let rate = 0.08_f64 / 12.0;
        let months = 120.0_f64;
        if rate > 0.0 {
            monthly * ((1.0 + rate).powf(months) - 1.0) / rate
        } else {
            monthly * months
        }
    });

    let health = Memo::new(move |_| {
        let s = savings.get();
        let n = needs.get();
        if s >= 20 && n >= 40 && n <= 60 { "Healthy balance" }
        else if s < 10 { "Warning: too little savings" }
        else if n > 70 { "Warning: needs are too high" }
        else { "Adjust to find your balance" }
    });

    let health_color = Memo::new(move |_| {
        let s = savings.get();
        if s >= 20 { "var(--mastery-green)" }
        else if s >= 10 { "var(--warning)" }
        else { "var(--error)" }
    });

    view! {
        <div class="game-container">
            <h3 style="font-size: 1rem; margin-bottom: 0.75rem">"Budget Simulator"</h3>
            <p style="font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 1rem">
                "Monthly income: R"{income as u32}". Allocate between needs, wants, and savings."
            </p>

            <div class="game-sliders">
                <div class="game-slider-row">
                    <span class="game-slider-label" style="width: 5rem">"Needs "{needs.get()}"%"</span>
                    <input type="range" min="0" max="100" step="5"
                        prop:value=move || needs.get().to_string()
                        on:input=move |ev| {
                            let v: u32 = leptos::prelude::event_target_value(&ev).parse().unwrap_or(50);
                            let max = 100_u32.saturating_sub(wants.get_untracked());
                            set_needs.set(v.min(max));
                        }
                        style="flex: 1"
                    />
                    <span style="font-size: 0.8rem; min-width: 4rem; text-align: right">"R"{needs_amount.get() as u32}</span>
                </div>
                <div class="game-slider-row">
                    <span class="game-slider-label" style="width: 5rem">"Wants "{wants.get()}"%"</span>
                    <input type="range" min="0" max="100" step="5"
                        prop:value=move || wants.get().to_string()
                        on:input=move |ev| {
                            let v: u32 = leptos::prelude::event_target_value(&ev).parse().unwrap_or(30);
                            let max = 100_u32.saturating_sub(needs.get_untracked());
                            set_wants.set(v.min(max));
                        }
                        style="flex: 1"
                    />
                    <span style="font-size: 0.8rem; min-width: 4rem; text-align: right">"R"{wants_amount.get() as u32}</span>
                </div>
                <div class="game-slider-row">
                    <span class="game-slider-label" style="width: 5rem; color: var(--mastery-green)">"Savings "{savings.get()}"%"</span>
                    <div style="flex: 1; height: 6px; background: var(--border); border-radius: 3px; position: relative">
                        <div style=move || format!("width: {}%; height: 100%; background: var(--mastery-green); border-radius: 3px; transition: width 0.3s", savings.get())></div>
                    </div>
                    <span style="font-size: 0.8rem; min-width: 4rem; text-align: right; color: var(--mastery-green)">"R"{savings_amount.get() as u32}</span>
                </div>
            </div>

            // Visual budget bar
            <svg viewBox="0 0 300 30" style="width: 100%; margin: 1rem 0; border-radius: 6px; overflow: hidden">
                <rect x="0" y="0" width=move || (needs.get() as f64 * 3.0) height="30" fill="var(--info)" />
                <rect x=move || (needs.get() as f64 * 3.0) y="0" width=move || (wants.get() as f64 * 3.0) height="30" fill="var(--warning)" />
                <rect x=move || ((needs.get() + wants.get()) as f64 * 3.0) y="0" width=move || (savings.get() as f64 * 3.0) height="30" fill="var(--mastery-green)" />
            </svg>

            <div style="display: flex; gap: 1rem; font-size: 0.75rem; color: var(--text-secondary); margin-bottom: 1rem">
                <span style="display: flex; align-items: center; gap: 0.25rem">
                    <span style="width: 8px; height: 8px; background: var(--info); border-radius: 2px"></span>"Needs"
                </span>
                <span style="display: flex; align-items: center; gap: 0.25rem">
                    <span style="width: 8px; height: 8px; background: var(--warning); border-radius: 2px"></span>"Wants"
                </span>
                <span style="display: flex; align-items: center; gap: 0.25rem">
                    <span style="width: 8px; height: 8px; background: var(--mastery-green); border-radius: 2px"></span>"Savings"
                </span>
            </div>

            // Compound interest projection
            <div style="padding: 0.75rem; background: var(--soil-rich); border-radius: 8px; margin-bottom: 0.75rem">
                <div style="font-size: 0.8rem; color: var(--text-secondary)">"If you invest your savings at 8% per year for 10 years:"</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: var(--mastery-green); margin-top: 0.25rem">
                    "R"{move || format!("{:.0}", future_value.get())}
                </div>
                <div style="font-size: 0.7rem; color: var(--text-tertiary)">
                    "Total contributed: R"{move || format!("{:.0}", savings_amount.get() * 120.0)}
                    " | Interest earned: R"{move || format!("{:.0}", future_value.get() - savings_amount.get() * 120.0)}
                </div>
            </div>

            // Health indicator
            <div style=move || format!("font-size: 0.85rem; font-weight: 600; color: {}; text-align: center", health_color.get())>
                {health.get()}
            </div>
            <div style="font-size: 0.7rem; color: var(--text-tertiary); text-align: center; margin-top: 0.25rem">
                "The 50/30/20 rule: 50% needs, 30% wants, 20% savings"
            </div>
        </div>
    }
}
