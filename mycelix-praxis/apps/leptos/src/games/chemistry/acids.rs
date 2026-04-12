// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Acid-Base Explorer — pH calculations and strong/weak acid visualization.
//!
//! Students adjust concentration and acid type, see pH, [H₃O⁺], [OH⁻] update.
//! Covers CAPS Gr12 P2.ACID (30 marks).

use leptos::prelude::*;
use crate::curriculum::{use_set_progress, ProgressStatus};
use crate::social_proof;

#[component]
pub fn AcidBaseExplorer(node_id: String) -> impl IntoView {
    let set_progress = use_set_progress();

    let (concentration, set_concentration) = signal(0.1_f64);  // mol/dm³
    let (is_strong, set_is_strong) = signal(true);
    let (is_acid, set_is_acid) = signal(true);
    let (ka_exponent, set_ka_exponent) = signal(-5.0_f64);  // Ka = 10^ka_exponent (for weak acids)

    // Calculations
    let h3o_concentration = Memo::new(move |_| {
        let c = concentration.get();
        if is_acid.get() {
            if is_strong.get() {
                c  // fully ionised
            } else {
                let ka = 10.0_f64.powf(ka_exponent.get());
                // [H₃O⁺] = √(Ka × c) for weak acid
                (ka * c).sqrt().min(c)
            }
        } else {
            // Base: [OH⁻] = c (strong) or √(Kb×c) (weak), then [H₃O⁺] = Kw/[OH⁻]
            let oh = if is_strong.get() { c } else {
                let kb = 10.0_f64.powf(ka_exponent.get());
                (kb * c).sqrt().min(c)
            };
            1.0e-14 / oh
        }
    });

    let oh_concentration = Memo::new(move |_| {
        1.0e-14 / h3o_concentration.get()
    });

    let ph = Memo::new(move |_| {
        let h = h3o_concentration.get();
        if h <= 0.0 { 14.0 } else { -h.log10() }
    });

    let poh = Memo::new(move |_| 14.0 - ph.get());

    let ph_color = Memo::new(move |_| {
        let p = ph.get();
        if p < 3.0 { "var(--error)" }
        else if p < 6.0 { "#f97316" }
        else if p < 8.0 { "var(--success)" }
        else if p < 11.0 { "var(--info)" }
        else { "#8b5cf6" }
    });

    let (challenge_idx, set_challenge_idx) = signal(0_usize);
    let (show_success, set_show_success) = signal(false);
    let (show_error_msg, set_show_error_msg) = signal(String::new());
    let (attempt_count, set_attempt_count) = signal(0_u32);

    let check_challenge = {
        let node_id = node_id.clone();
        move || {
            let idx = challenge_idx.get_untracked();
            let p = ph.get_untracked();
            let strong = is_strong.get_untracked();
            let acid = is_acid.get_untracked();

            set_attempt_count.update(|c| *c += 1);

            let passed = match idx {
                0 => (p - 1.0).abs() < 0.2 && acid && strong,   // pH = 1 with strong acid
                1 => (p - 13.0).abs() < 0.3 && !acid && strong, // pH = 13 with strong base
                2 => !strong && acid && p > 2.0 && p < 5.0,      // Weak acid pH (not fully ionised)
                3 => (p - 7.0).abs() < 0.5,                      // Neutral pH
                _ => false,
            };

            if passed {
                set_show_error_msg.set(String::new());
                set_show_success.set(true);
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
            } else {
                let att = attempt_count.get_untracked();
                let msg = social_proof::wise_error_message("pH calculations", att, true);
                set_show_error_msg.set(msg);
                wasm_bindgen_futures::spawn_local(async move {
                    gloo_timers::future::sleep(std::time::Duration::from_millis(3000)).await;
                    set_show_error_msg.set(String::new());
                });
            }
        }
    };

    view! {
        <div class="game-container">
            // pH scale visualization
            <div style="position: relative; height: 40px; margin-bottom: 1.5rem; border-radius: 8px; overflow: hidden; background: linear-gradient(to right, #ef4444, #f97316, #facc15, #22c55e, #3b82f6, #7c3aed)">
                // pH indicator needle
                <div style=move || format!("position: absolute; top: -4px; left: calc({}% - 2px); width: 4px; height: 48px; background: white; border-radius: 2px; transition: left 0.3s; box-shadow: 0 0 8px rgba(255,255,255,0.5)", ph.get() / 14.0 * 100.0)></div>
                // Scale labels
                <div style="position: absolute; bottom: -20px; left: 0; width: 100%; display: flex; justify-content: space-between; font-size: 0.65rem; color: var(--text-tertiary)">
                    <span>"0"</span><span>"7"</span><span>"14"</span>
                </div>
            </div>

            // Values display
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 0.5rem; text-align: center; padding: 1.5rem 0 0.75rem">
                <div>
                    <div style="font-size: 0.65rem; color: var(--text-tertiary)">"pH"</div>
                    <div style=move || format!("font-size: 1.5rem; font-weight: 700; color: {}", ph_color.get())>
                        {move || format!("{:.2}", ph.get())}
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.65rem; color: var(--text-tertiary)">"pOH"</div>
                    <div style="font-size: 1.1rem; font-weight: 700">{move || format!("{:.2}", poh.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.65rem; color: var(--text-tertiary)">"[H\u{2083}O\u{207a}]"</div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: var(--error)">{move || format!("{:.2e}", h3o_concentration.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.65rem; color: var(--text-tertiary)">"[OH\u{207b}]"</div>
                    <div style="font-size: 0.85rem; font-weight: 600; color: var(--info)">{move || format!("{:.2e}", oh_concentration.get())}</div>
                </div>
            </div>

            // Type toggles
            <div style="display: flex; gap: 0.5rem; justify-content: center; margin-bottom: 1rem">
                <div class="praxis-filter-group">
                    <button class=move || if is_acid.get() { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                        on:click=move |_| set_is_acid.set(true)>"Acid"</button>
                    <button class=move || if !is_acid.get() { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                        on:click=move |_| set_is_acid.set(false)>"Base"</button>
                </div>
                <div class="praxis-filter-group">
                    <button class=move || if is_strong.get() { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                        on:click=move |_| set_is_strong.set(true)>"Strong"</button>
                    <button class=move || if !is_strong.get() { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                        on:click=move |_| set_is_strong.set(false)>"Weak"</button>
                </div>
            </div>

            // Formula display
            <div class="game-equation" style="font-size: 0.85rem">
                {move || if is_acid.get() {
                    if is_strong.get() { "pH = \u{2212}log[H\u{2083}O\u{207a}] = \u{2212}log(c)" }
                    else { "pH = \u{2212}log(\u{221a}(Ka \u{00d7} c))" }
                } else {
                    if is_strong.get() { "pOH = \u{2212}log[OH\u{207b}], pH = 14 \u{2212} pOH" }
                    else { "pOH = \u{2212}log(\u{221a}(Kb \u{00d7} c)), pH = 14 \u{2212} pOH" }
                }}
            </div>

            // Sliders
            <div class="game-sliders">
                <div class="game-slider-row">
                    <label class="game-slider-label">"c"</label>
                    <input type="range" min="-3" max="0" step="0.1" class="game-slider"
                        prop:value=move || concentration.get().log10().to_string()
                        on:input=move |ev| {
                            if let Ok(v) = leptos::prelude::event_target_value(&ev).parse::<f64>() {
                                set_concentration.set(10.0_f64.powf(v));
                            }
                        } />
                    <span class="game-slider-value">{move || format!("{:.3}", concentration.get())}</span>
                </div>
                {move || if !is_strong.get() {
                    view! {
                        <div class="game-slider-row">
                            <label class="game-slider-label">"Ka"</label>
                            <input type="range" min="-8" max="-1" step="0.5" class="game-slider"
                                prop:value=move || ka_exponent.get().to_string()
                                on:input=move |ev| {
                                    if let Ok(v) = leptos::prelude::event_target_value(&ev).parse::<f64>() {
                                        set_ka_exponent.set(v);
                                    }
                                } />
                            <span class="game-slider-value">{move || format!("10\u{207b}{}", (-ka_exponent.get()) as i32)}</span>
                        </div>
                    }.into_any()
                } else {
                    view! { <span></span> }.into_any()
                }}
            </div>

            // Challenges
            <div class="game-challenge">
                {move || {
                    let idx = challenge_idx.get();
                    if idx >= 4 {
                        view! { <div class="game-complete"><div class="game-complete-icon">"\u{2714}"</div><div class="game-complete-text">"All challenges complete!"</div></div> }.into_any()
                    } else if show_success.get() {
                        view! { <div class="game-success"><span class="game-success-icon">"\u{2714}"</span>" Correct!"</div> }.into_any()
                    } else {
                        let instructions = [
                            "Create a strong acid solution with pH = 1",
                            "Create a strong base solution with pH = 13",
                            "Switch to a weak acid and observe: pH is higher than a strong acid at the same concentration",
                            "Find a concentration that gives a nearly neutral pH (\u{2248} 7)",
                        ];
                        let hints = [
                            "For a strong acid, pH = \u{2212}log(c). What concentration gives pH = 1?",
                            "For a strong base, pOH = \u{2212}log(c), pH = 14 \u{2212} pOH.",
                            "A weak acid only partially ionises, so [H\u{2083}O\u{207a}] < c. The pH is higher.",
                            "Very dilute solutions approach pH 7. Try a very low concentration.",
                        ];
                        let check = check_challenge.clone();
                        view! {
                            <div class="game-challenge-active">
                                <div class="game-challenge-number">"Challenge "{idx + 1}" of 4"</div>
                                <div class="game-challenge-instruction">{instructions[idx]}</div>
                                <div class="game-challenge-actions">
                                    <button class="praxis-filter-btn active" on:click=move |_| check()>"Check"</button>
                                </div>
                                <div class="game-challenge-hint">{hints[idx]}</div>
                                {move || {
                                    let msg = show_error_msg.get();
                                    if msg.is_empty() { view! { <span></span> }.into_any() }
                                    else { view! {
                                        <div style="margin-top: 0.75rem; padding: 0.5rem 0.75rem; border-left: 3px solid var(--info); background: rgba(59,130,246,0.05); border-radius: 0 6px 6px 0; font-size: 0.85rem; color: var(--text-secondary)">
                                            {msg}
                                        </div>
                                    }.into_any() }
                                }}
                            </div>
                        }.into_any()
                    }
                }}
            </div>
        </div>
    }
}
