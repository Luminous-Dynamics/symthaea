// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Statistics Explorer — drag data points, see mean/median/SD update live.

use leptos::prelude::*;
use crate::curriculum::{use_set_progress, ProgressStatus};

#[component]
pub fn StatsExplorer(node_id: String) -> impl IntoView {
    let set_progress = use_set_progress();

    // 8 data points the student can adjust
    let initial = vec![3.0, 5.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0];
    let (data, set_data) = signal(initial);
    let (challenge_idx, set_challenge_idx) = signal(0_usize);
    let (show_success, set_show_success) = signal(false);

    let mean = Memo::new(move |_| {
        let d = data.get();
        if d.is_empty() { return 0.0; }
        d.iter().sum::<f64>() / d.len() as f64
    });

    let median = Memo::new(move |_| {
        let mut d = data.get();
        d.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = d.len();
        if n == 0 { return 0.0; }
        if n % 2 == 0 { (d[n/2 - 1] + d[n/2]) / 2.0 } else { d[n/2] }
    });

    let std_dev = Memo::new(move |_| {
        let d = data.get();
        let m = mean.get();
        let n = d.len() as f64;
        if n == 0.0 { return 0.0; }
        let variance = d.iter().map(|x| (x - m).powi(2)).sum::<f64>() / n;
        variance.sqrt()
    });

    let range = Memo::new(move |_| {
        let d = data.get();
        if d.is_empty() { return 0.0; }
        let min = d.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = d.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        max - min
    });

    struct Challenge { instruction: &'static str, check: fn(f64, f64, f64) -> bool, hint: &'static str }
    let challenges = vec![
        Challenge { instruction: "Make the mean equal to 10", check: |mean, _, _| (mean - 10.0).abs() < 0.5, hint: "The mean is the sum divided by the count. Adjust values so they average to 10." },
        Challenge { instruction: "Make the standard deviation less than 1", check: |_, _, sd| sd < 1.0, hint: "Standard deviation measures spread. Make all values very close together." },
        Challenge { instruction: "Make the median greater than the mean", check: |mean, median, _| median > mean + 0.5, hint: "Add some very low values to pull the mean down while keeping the middle values high." },
        Challenge { instruction: "Make the range exactly 10", check: |_, _, _| false, hint: "Range = max - min. Set the highest value 10 more than the lowest." }, // Note: we'll check range separately
    ];
    let total_challenges = challenges.len();

    // Override challenge 3 check to use range
    let check_challenge = {
        let node_id = node_id.clone();
        move || {
            let idx = challenge_idx.get_untracked();
            if idx >= 4 { return; }
            let m = mean.get_untracked();
            let med = median.get_untracked();
            let sd = std_dev.get_untracked();
            let r = range.get_untracked();
            let passed = match idx {
                0 => (m - 10.0).abs() < 0.5,
                1 => sd < 1.0,
                2 => med > m + 0.5,
                3 => (r - 10.0).abs() < 0.5,
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
            // Stats display
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 0.5rem; text-align: center; margin-bottom: 1rem">
                <div>
                    <div style="font-size: 0.7rem; color: var(--text-tertiary)">"Mean"</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: var(--info)">{move || format!("{:.1}", mean.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--text-tertiary)">"Median"</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: var(--success)">{move || format!("{:.1}", median.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--text-tertiary)">"Std Dev"</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: var(--warning)">{move || format!("{:.2}", std_dev.get())}</div>
                </div>
                <div>
                    <div style="font-size: 0.7rem; color: var(--text-tertiary)">"Range"</div>
                    <div style="font-size: 1.2rem; font-weight: 700; color: var(--error)">{move || format!("{:.1}", range.get())}</div>
                </div>
            </div>

            // Data point sliders
            <div class="game-sliders">
                {(0..8).map(|i| {
                    view! {
                        <div class="game-slider-row">
                            <label class="game-slider-label" style="width: 1.5rem">{format!("x{}", i+1)}</label>
                            <input type="range" min="0" max="20" step="0.5" class="game-slider"
                                prop:value=move || data.get()[i].to_string()
                                on:input=move |ev| {
                                    if let Ok(v) = leptos::prelude::event_target_value(&ev).parse::<f64>() {
                                        set_data.update(|d| d[i] = v);
                                    }
                                }
                            />
                            <span class="game-slider-value" style="width: 2rem">{move || format!("{:.0}", data.get()[i])}</span>
                        </div>
                    }
                }).collect::<Vec<_>>()}
            </div>

            // Challenges
            <div class="game-challenge">
                {move || {
                    let idx = challenge_idx.get();
                    if idx >= total_challenges {
                        view! { <div class="game-complete"><div class="game-complete-icon">"\u{2714}"</div><div class="game-complete-text">"All challenges complete!"</div></div> }.into_any()
                    } else if show_success.get() {
                        view! { <div class="game-success"><span class="game-success-icon">"\u{2714}"</span>" Correct!"</div> }.into_any()
                    } else {
                        let instructions = ["Make the mean equal to 10", "Make the standard deviation less than 1", "Make the median greater than the mean", "Make the range exactly 10"];
                        let hints = ["Adjust values so they average to 10.", "Make all values very close together.", "Add low outliers to pull the mean down.", "Set max - min = 10."];
                        let check = check_challenge.clone();
                        view! {
                            <div class="game-challenge-active">
                                <div class="game-challenge-number">"Challenge "{idx + 1}" of "{total_challenges}</div>
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
