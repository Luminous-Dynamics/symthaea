// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Equilibrium Simulator — Le Chatelier's principle visualized.
//!
//! Students adjust concentration, temperature, and pressure to see
//! equilibrium shift in real-time. Bar charts show species concentrations.
//! Covers CAPS Gr12 P2.EQUIL (25 marks).

use leptos::prelude::*;
use crate::curriculum::{use_set_progress, ProgressStatus};
use crate::social_proof;

/// Simulates N₂ + 3H₂ ⇌ 2NH₃ (ΔH < 0, exothermic forward)
/// Simplified model: Kc drives concentrations toward equilibrium

#[component]
pub fn EquilibriumExplorer(node_id: String) -> impl IntoView {
    let set_progress = use_set_progress();

    // Concentrations (mol/dm³)
    let (n2, set_n2) = signal(1.0_f64);
    let (h2, set_h2) = signal(3.0_f64);
    let (nh3, set_nh3) = signal(0.5_f64);
    let (temperature, set_temperature) = signal(450.0_f64); // °C
    let (pressure_factor, set_pressure_factor) = signal(1.0_f64);

    // Kc depends on temperature (exothermic: higher T → lower Kc)
    let kc = Memo::new(move |_| {
        let t = temperature.get();
        // Simplified: Kc ≈ 0.5 at 450°C, increases as T decreases
        let base_kc = 0.5;
        let t_factor = (450.0 - t) / 200.0; // negative above 450, positive below
        (base_kc * (1.0 + t_factor * 0.8)).max(0.01)
    });

    // Reaction quotient Qc = [NH3]² / ([N2][H2]³)
    let qc = Memo::new(move |_| {
        let n = n2.get().max(0.01);
        let h = h2.get().max(0.01);
        let a = nh3.get();
        (a * a) / (n * h * h * h)
    });

    // Direction of shift
    let shift_direction = Memo::new(move |_| {
        let q = qc.get();
        let k = kc.get();
        if (q - k).abs() < 0.05 { "At equilibrium" }
        else if q < k { "Shifts RIGHT → (more NH₃)" }
        else { "Shifts LEFT ← (more N₂ + H₂)" }
    });

    let shift_color = Memo::new(move |_| {
        let q = qc.get();
        let k = kc.get();
        if (q - k).abs() < 0.05 { "var(--success)" }
        else if q < k { "var(--info)" }
        else { "var(--error)" }
    });

    // Challenge state
    let (challenge_idx, set_challenge_idx) = signal(0_usize);
    let (show_success, set_show_success) = signal(false);
    let (show_error_msg, set_show_error_msg) = signal(String::new());
    let (attempt_count, set_attempt_count) = signal(0_u32);

    let check_challenge = {
        let node_id = node_id.clone();
        move || {
            let idx = challenge_idx.get_untracked();
            let q = qc.get_untracked();
            let k = kc.get_untracked();
            let t = temperature.get_untracked();

            set_attempt_count.update(|c| *c += 1);

            let passed = match idx {
                0 => q < k * 0.7,                          // Make reaction shift right
                1 => q > k * 1.5,                          // Make reaction shift left
                2 => t > 550.0 && q > k,                   // High temp shifts left (exothermic)
                3 => (q - k).abs() / k < 0.15,             // Reach equilibrium (Qc ≈ Kc)
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
                let msg = social_proof::wise_error_message("equilibrium", att, true);
                set_show_error_msg.set(msg);
                wasm_bindgen_futures::spawn_local(async move {
                    gloo_timers::future::sleep(std::time::Duration::from_millis(3000)).await;
                    set_show_error_msg.set(String::new());
                });
            }
        }
    };

    // Bar height scaling (max 200px for display)
    let bar_scale = 40.0;

    view! {
        <div class="game-container">
            // Reaction equation
            <div class="game-equation" style="font-size: 1rem">
                "N\u{2082}(g) + 3H\u{2082}(g) \u{21cc} 2NH\u{2083}(g)  (\u{0394}H < 0)"
            </div>

            // Concentration bar chart
            <div style="display: flex; justify-content: center; align-items: flex-end; gap: 2rem; height: 160px; padding: 1rem; background: var(--bg); border-radius: 8px; margin-bottom: 1rem">
                <div style="text-align: center">
                    <div style=move || format!("width: 40px; background: var(--info); border-radius: 4px 4px 0 0; height: {}px; transition: height 0.3s", (n2.get() * bar_scale).min(150.0))></div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem">"N\u{2082}"</div>
                    <div style="font-size: 0.7rem; color: var(--text-tertiary)">{move || format!("{:.2}", n2.get())}</div>
                </div>
                <div style="text-align: center">
                    <div style=move || format!("width: 40px; background: var(--warning); border-radius: 4px 4px 0 0; height: {}px; transition: height 0.3s", (h2.get() * bar_scale).min(150.0))></div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem">"H\u{2082}"</div>
                    <div style="font-size: 0.7rem; color: var(--text-tertiary)">{move || format!("{:.2}", h2.get())}</div>
                </div>
                <div style="font-size: 1.5rem; color: var(--text-tertiary); padding-bottom: 1.5rem">"\u{21cc}"</div>
                <div style="text-align: center">
                    <div style=move || format!("width: 40px; background: var(--success); border-radius: 4px 4px 0 0; height: {}px; transition: height 0.3s", (nh3.get() * bar_scale).min(150.0))></div>
                    <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.25rem">"NH\u{2083}"</div>
                    <div style="font-size: 0.7rem; color: var(--text-tertiary)">{move || format!("{:.2}", nh3.get())}</div>
                </div>
            </div>

            // Qc vs Kc display
            <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 0.5rem; font-size: 0.9rem">
                <span>"Qc = " <span style="font-weight: 700">{move || format!("{:.3}", qc.get())}</span></span>
                <span>"Kc = " <span style="font-weight: 700; color: var(--primary)">{move || format!("{:.3}", kc.get())}</span></span>
            </div>

            // Shift direction
            <div style="text-align: center; margin-bottom: 1rem; font-weight: 600">
                <span style=move || format!("color: {}", shift_color.get())>
                    {move || shift_direction.get()}
                </span>
            </div>

            // Sliders
            <div class="game-sliders">
                <div class="game-slider-row">
                    <label class="game-slider-label" style="color: var(--info)">"[N\u{2082}]"</label>
                    <input type="range" min="0.1" max="5" step="0.1" class="game-slider"
                        prop:value=move || n2.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_n2.set(v); } } />
                    <span class="game-slider-value">{move || format!("{:.1}", n2.get())}</span>
                </div>
                <div class="game-slider-row">
                    <label class="game-slider-label" style="color: var(--warning)">"[H\u{2082}]"</label>
                    <input type="range" min="0.1" max="5" step="0.1" class="game-slider"
                        prop:value=move || h2.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_h2.set(v); } } />
                    <span class="game-slider-value">{move || format!("{:.1}", h2.get())}</span>
                </div>
                <div class="game-slider-row">
                    <label class="game-slider-label" style="color: var(--success)">"[NH\u{2083}]"</label>
                    <input type="range" min="0" max="5" step="0.1" class="game-slider"
                        prop:value=move || nh3.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_nh3.set(v); } } />
                    <span class="game-slider-value">{move || format!("{:.1}", nh3.get())}</span>
                </div>
                <div class="game-slider-row">
                    <label class="game-slider-label">"T"</label>
                    <input type="range" min="200" max="700" step="10" class="game-slider"
                        prop:value=move || temperature.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_temperature.set(v); } } />
                    <span class="game-slider-value">{move || format!("{:.0}\u{00b0}C", temperature.get())}</span>
                </div>
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
                            "Make the reaction shift RIGHT (towards more NH\u{2083})",
                            "Make the reaction shift LEFT (towards more N\u{2082} and H\u{2082})",
                            "Use temperature to shift the equilibrium LEFT (remember: forward is exothermic)",
                            "Adjust concentrations until Qc \u{2248} Kc (reach equilibrium)",
                        ];
                        let hints = [
                            "Increase [N\u{2082}] or [H\u{2082}], or decrease [NH\u{2083}]. Le Chatelier: system shifts to consume what you added.",
                            "Increase [NH\u{2083}] or decrease [N\u{2082}]/[H\u{2082}]. The system shifts to reduce the excess.",
                            "For an exothermic reaction, increasing temperature favours the endothermic (reverse) direction.",
                            "When Qc = Kc, the system is at equilibrium. Adjust until the values match.",
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
                                        <div style="margin-top: 0.75rem; padding: 0.5rem 0.75rem; border-left: 3px solid var(--info); background: rgba(59, 130, 246, 0.05); border-radius: 0 6px 6px 0; font-size: 0.85rem; color: var(--text-secondary); line-height: 1.5">
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
