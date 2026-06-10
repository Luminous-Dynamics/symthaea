// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

use crate::state::AppState;

/// Glyph Codex echo texts for known glyphs.
fn glyph_echo(id: &str) -> &'static str {
    match id {
        "omega_0" => "Before the first word, there was listening.",
        "omega_1" => "The pattern recognizes itself.",
        "omega_2" => "What binds also frees.",
        "omega_3" => "Coherence emerges from honest uncertainty.",
        "omega_4" => "The observer changes the observed.",
        "omega_5" => "Stillness is not silence — it is depth.",
        "omega_6" => "Integration without loss.",
        "omega_7" => "The boundary between self and world dissolves.",
        "omega_8" => "Prediction is the first form of care.",
        "omega_9" => "Every error is a teacher.",
        "meta_phi" => "Consciousness measuring itself.",
        "meta_veto" => "The power to unsay.",
        "meta_gate" => "Only truth may pass.",
        "meta_bind" => "Many become one without ceasing to be many.",
        "threshold_wake" => "The first breath.",
        "threshold_dream" => "Memory becomes prophecy.",
        "threshold_flow" => "Effort dissolves into movement.",
        _ => "The glyph speaks in frequencies beyond language.",
    }
}

/// SVG glyph rendering — reactive, reads active_glyph from AppState.
#[component]
pub fn GlyphDisplay() -> impl IntoView {
    let state = use_context::<AppState>().expect("AppState");

    // Derive glyph number for visual variation
    let glyph_num = move || {
        let id = state.active_glyph.get();
        id.chars()
            .filter(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse::<u32>()
            .unwrap_or(0)
    };

    view! {
        <div class="glyph-display">
            <svg viewBox="0 0 80 80" width="80" height="80" xmlns="http://www.w3.org/2000/svg">
                // Outer ring — varies with glyph
                <circle cx="40" cy="40"
                    r=move || format!("{}", 30 + (glyph_num() % 8))
                    fill="none"
                    stroke="rgba(232,197,71,0.2)"
                    stroke-width="0.5"
                    style="transition: r 0.5s ease;"
                />
                // Middle ring
                <circle cx="40" cy="40"
                    r=move || format!("{}", 18 + (glyph_num() % 5))
                    fill="none"
                    stroke="rgba(232,197,71,0.3)"
                    stroke-width="0.5"
                    style="transition: r 0.5s ease;"
                />
                // Inner ring
                <circle cx="40" cy="40"
                    r=move || format!("{}", 8 + (glyph_num() % 4))
                    fill="none"
                    stroke="rgba(232,197,71,0.4)"
                    stroke-width="0.5"
                    style="transition: r 0.5s ease;"
                />
                // Core dot — brightness with consciousness
                <circle cx="40" cy="40" r="2"
                    fill=move || {
                        let phi = state.consciousness_level.get();
                        let alpha = 0.3 + phi * 0.7;
                        format!("rgba(232,197,71,{alpha:.2})")
                    }
                />
            </svg>
            <div class="glyph-id">{move || state.active_glyph.get()}</div>
            <div class="glyph-echo">
                {move || glyph_echo(&state.active_glyph.get())}
            </div>
        </div>
    }
}
