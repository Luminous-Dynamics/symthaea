// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Circuit Explorer — interactive Ohm's law and series/parallel circuits.
//!
//! Students adjust resistance and EMF, see current and voltage update live.
//! Covers internal resistance (ε = I(R + r)) for Gr12.

use leptos::prelude::*;
use crate::curriculum::{use_set_progress, ProgressStatus};

#[component]
pub fn CircuitExplorer(node_id: String) -> impl IntoView {
    let set_progress = use_set_progress();

    let (emf, set_emf) = signal(12.0_f64);       // Battery EMF (V)
    let (r_int, set_r_int) = signal(0.5_f64);     // Internal resistance (Ω)
    let (r1, set_r1) = signal(4.0_f64);            // Resistor 1 (Ω)
    let (r2, set_r2) = signal(6.0_f64);            // Resistor 2 (Ω)
    let (series, set_series) = signal(true);        // true = series, false = parallel

    // Calculations
    let r_external = Memo::new(move |_| {
        if series.get() {
            r1.get() + r2.get()
        } else {
            let r1v = r1.get();
            let r2v = r2.get();
            if r1v + r2v < 0.01 { 0.01 } else { (r1v * r2v) / (r1v + r2v) }
        }
    });

    let current = Memo::new(move |_| {
        let r_total = r_external.get() + r_int.get();
        if r_total < 0.01 { 0.0 } else { emf.get() / r_total }
    });

    let v_terminal = Memo::new(move |_| emf.get() - current.get() * r_int.get());
    let v_lost = Memo::new(move |_| current.get() * r_int.get());
    let v_r1 = Memo::new(move |_| {
        if series.get() { current.get() * r1.get() }
        else { v_terminal.get() } // parallel: same voltage across both
    });
    let v_r2 = Memo::new(move |_| {
        if series.get() { current.get() * r2.get() }
        else { v_terminal.get() }
    });
    let i_r1 = Memo::new(move |_| {
        if series.get() { current.get() }
        else if r1.get() < 0.01 { 0.0 } else { v_terminal.get() / r1.get() }
    });
    let i_r2 = Memo::new(move |_| {
        if series.get() { current.get() }
        else if r2.get() < 0.01 { 0.0 } else { v_terminal.get() / r2.get() }
    });
    let power_total = Memo::new(move |_| emf.get() * current.get());

    let (challenge_idx, set_challenge_idx) = signal(0_usize);
    let (show_success, set_show_success) = signal(false);

    let check_challenge = {
        let node_id = node_id.clone();
        move || {
            let idx = challenge_idx.get_untracked();
            let i = current.get_untracked();
            let vt = v_terminal.get_untracked();
            let passed = match idx {
                0 => (i - 2.0).abs() < 0.15,                    // Current = 2A
                1 => (vt - 10.0).abs() < 0.3,                   // Terminal voltage = 10V
                2 => !series.get_untracked() && (i - 3.0).abs() < 0.3, // Parallel, I ≈ 3A
                3 => (v_lost.get_untracked() - 2.0).abs() < 0.3,      // Lost volts = 2V
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
                    if next >= 4 {
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
            // Circuit type toggle
            <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; justify-content: center">
                <button
                    class=move || if series.get() { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                    on:click=move |_| set_series.set(true)
                >"Series"</button>
                <button
                    class=move || if !series.get() { "praxis-filter-btn active" } else { "praxis-filter-btn" }
                    on:click=move |_| set_series.set(false)
                >"Parallel"</button>
            </div>

            // Circuit diagram (simplified SVG)
            <svg viewBox="0 0 300 180" class="game-graph" style="aspect-ratio: 5/3; max-height: 250px; background: var(--bg)">
                // Battery
                <rect x="20" y="60" width="40" height="60" rx="4" fill="none" stroke="var(--warning)" stroke-width="2" />
                <text x="40" y="55" text-anchor="middle" fill="var(--warning)" font-size="10">"Battery"</text>
                <text x="40" y="85" text-anchor="middle" fill="var(--text)" font-size="9">
                    {move || format!("{}V", emf.get())}
                </text>
                <text x="40" y="100" text-anchor="middle" fill="var(--text-tertiary)" font-size="7">
                    {move || format!("r={}Ω", r_int.get())}
                </text>

                // Wires from battery
                <line x1="60" y1="75" x2="120" y2="75" stroke="var(--text-secondary)" stroke-width="1.5" />
                <line x1="60" y1="105" x2="120" y2="105" stroke="var(--text-secondary)" stroke-width="1.5" />

                // Resistors (series or parallel)
                {move || if series.get() {
                    view! {
                        // R1
                        <rect x="120" y="65" width="60" height="20" rx="3" fill="none" stroke="var(--info)" stroke-width="2" />
                        <text x="150" y="60" text-anchor="middle" fill="var(--info)" font-size="9">
                            {format!("R1={}Ω", r1.get_untracked())}
                        </text>
                        // Wire between
                        <line x1="180" y1="75" x2="200" y2="75" stroke="var(--text-secondary)" stroke-width="1.5" />
                        // R2
                        <rect x="200" y="65" width="60" height="20" rx="3" fill="none" stroke="var(--error)" stroke-width="2" />
                        <text x="230" y="60" text-anchor="middle" fill="var(--error)" font-size="9">
                            {format!("R2={}Ω", r2.get_untracked())}
                        </text>
                        // Return wire
                        <line x1="260" y1="75" x2="280" y2="75" stroke="var(--text-secondary)" stroke-width="1.5" />
                        <line x1="280" y1="75" x2="280" y2="105" stroke="var(--text-secondary)" stroke-width="1.5" />
                        <line x1="120" y1="105" x2="280" y2="105" stroke="var(--text-secondary)" stroke-width="1.5" />
                    }.into_any()
                } else {
                    view! {
                        // Parallel split
                        <line x1="120" y1="75" x2="120" y2="55" stroke="var(--text-secondary)" stroke-width="1.5" />
                        <line x1="120" y1="75" x2="120" y2="95" stroke="var(--text-secondary)" stroke-width="1.5" />
                        // R1 (top branch)
                        <rect x="140" y="45" width="80" height="20" rx="3" fill="none" stroke="var(--info)" stroke-width="2" />
                        <text x="180" y="42" text-anchor="middle" fill="var(--info)" font-size="9">
                            {format!("R1={}Ω", r1.get_untracked())}
                        </text>
                        <line x1="120" y1="55" x2="140" y2="55" stroke="var(--text-secondary)" stroke-width="1.5" />
                        <line x1="220" y1="55" x2="260" y2="55" stroke="var(--text-secondary)" stroke-width="1.5" />
                        // R2 (bottom branch)
                        <rect x="140" y="85" width="80" height="20" rx="3" fill="none" stroke="var(--error)" stroke-width="2" />
                        <text x="180" y="118" text-anchor="middle" fill="var(--error)" font-size="9">
                            {format!("R2={}Ω", r2.get_untracked())}
                        </text>
                        <line x1="120" y1="95" x2="140" y2="95" stroke="var(--text-secondary)" stroke-width="1.5" />
                        <line x1="220" y1="95" x2="260" y2="95" stroke="var(--text-secondary)" stroke-width="1.5" />
                        // Rejoin
                        <line x1="260" y1="55" x2="260" y2="95" stroke="var(--text-secondary)" stroke-width="1.5" />
                        <line x1="260" y1="75" x2="280" y2="75" stroke="var(--text-secondary)" stroke-width="1.5" />
                        <line x1="280" y1="75" x2="280" y2="105" stroke="var(--text-secondary)" stroke-width="1.5" />
                        <line x1="120" y1="105" x2="280" y2="105" stroke="var(--text-secondary)" stroke-width="1.5" />
                    }.into_any()
                }}

                // Current arrow
                <text x="90" y="70" fill="var(--success)" font-size="9">
                    {move || format!("I={:.2}A", current.get())}
                </text>
            </svg>

            // Readings
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.5rem; text-align: center; padding: 0.75rem 0; font-size: 0.85rem">
                <div>
                    <div style="font-size: 0.65rem; color: var(--text-tertiary)">"V terminal"</div>
                    <div style="font-weight: 700; color: var(--success)">{move || format!("{:.2}V", v_terminal.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.65rem; color: var(--text-tertiary)">"V lost (Ir)"</div>
                    <div style="font-weight: 700; color: var(--error)">{move || format!("{:.2}V", v_lost.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.65rem; color: var(--text-tertiary)">"R external"</div>
                    <div style="font-weight: 700">{move || format!("{:.2}\u{03a9}", r_external.get())}</div>
                </div>
            </div>

            // Formula
            <div class="game-equation" style="font-size: 0.9rem">
                "\u{03b5} = I(R + r) \u{2192} "
                {move || format!("{:.1} = {:.2}({:.2} + {:.1})", emf.get(), current.get(), r_external.get(), r_int.get())}
            </div>

            // Sliders
            <div class="game-sliders">
                <div class="game-slider-row">
                    <label class="game-slider-label">"\u{03b5}"</label>
                    <input type="range" min="1" max="24" step="0.5" class="game-slider"
                        prop:value=move || emf.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_emf.set(v); } } />
                    <span class="game-slider-value">{move || format!("{:.0}V", emf.get())}</span>
                </div>
                <div class="game-slider-row">
                    <label class="game-slider-label">"r"</label>
                    <input type="range" min="0.1" max="3" step="0.1" class="game-slider"
                        prop:value=move || r_int.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_r_int.set(v); } } />
                    <span class="game-slider-value">{move || format!("{:.1}\u{03a9}", r_int.get())}</span>
                </div>
                <div class="game-slider-row">
                    <label class="game-slider-label">"R1"</label>
                    <input type="range" min="1" max="20" step="0.5" class="game-slider"
                        prop:value=move || r1.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_r1.set(v); } } />
                    <span class="game-slider-value">{move || format!("{:.0}\u{03a9}", r1.get())}</span>
                </div>
                <div class="game-slider-row">
                    <label class="game-slider-label">"R2"</label>
                    <input type="range" min="1" max="20" step="0.5" class="game-slider"
                        prop:value=move || r2.get().to_string()
                        on:input=move |ev| { if let Ok(v) = leptos::prelude::event_target_value(&ev).parse() { set_r2.set(v); } } />
                    <span class="game-slider-value">{move || format!("{:.0}\u{03a9}", r2.get())}</span>
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
                            "Set the circuit so the current is exactly 2A",
                            "Adjust the circuit so the terminal voltage is 10V",
                            "Switch to parallel and make the total current approximately 3A",
                            "Make the lost volts (Ir) equal to 2V",
                        ];
                        let hints = [
                            "\u{03b5} = I(R+r). If I = 2A, then R+r = \u{03b5}/2.",
                            "V_terminal = \u{03b5} - Ir. Adjust R or r until V_terminal = 10V.",
                            "In parallel, R_total = R1\u{00b7}R2/(R1+R2). Lower total R means more current.",
                            "Lost volts = I \u{00d7} r. Adjust internal resistance and current.",
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
                            </div>
                        }.into_any()
                    }
                }}
            </div>
        </div>
    }
}
