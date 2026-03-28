// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

/// Interactive earnings calculator comparing Spotify vs Mycelix per-play rates.
#[component]
pub fn EarningsCalculator() -> impl IntoView {
    let plays = RwSignal::new(10_000u64);

    let spotify_earnings = move || {
        let p = plays.get() as f64;
        p * 0.003 // ~$0.003/play average
    };

    let mycelix_earnings = move || {
        let p = plays.get() as f64;
        p * 0.007 // ~$0.007/play (70% of $0.01)
    };

    let multiplier = move || {
        let s = spotify_earnings();
        if s > 0.0 {
            mycelix_earnings() / s
        } else {
            0.0
        }
    };

    view! {
        <div class="earnings-calculator">
            <h3>"Earnings Calculator"</h3>
            <div class="earnings-input">
                <label>"Monthly plays: " <strong>{move || plays.get().to_string()}</strong></label>
                <input
                    type="range"
                    min="1000"
                    max="100000"
                    step="1000"
                    prop:value=move || plays.get().to_string()
                    on:input=move |ev| {
                        if let Ok(v) = event_target_value(&ev).parse::<u64>() {
                            plays.set(v);
                        }
                    }
                />
            </div>
            <div class="earnings-comparison">
                <div class="earnings-column spotify">
                    <h4>"Spotify"</h4>
                    <span class="earnings-value">{move || format!("${:.2}", spotify_earnings())}</span>
                </div>
                <div class="earnings-column mycelix">
                    <h4>"Mycelix"</h4>
                    <span class="earnings-value">{move || format!("${:.2}", mycelix_earnings())}</span>
                </div>
            </div>
            <p class="earnings-multiplier">
                {move || format!("{:.1}x more on Mycelix", multiplier())}
            </p>
        </div>
    }
}
