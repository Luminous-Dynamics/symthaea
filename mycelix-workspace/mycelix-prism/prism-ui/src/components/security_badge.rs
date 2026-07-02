// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Security badge: safety dot + threat count.

use crate::state::BrowserState;
use leptos::prelude::*;
use prism_common::SafetyLevel;

#[component]
pub fn SecurityBadge() -> impl IntoView {
    let state = expect_context::<BrowserState>();

    let safety_class = move || match state.safety.get() {
        SafetyLevel::Green => "safety-dot green",
        SafetyLevel::Yellow => "safety-dot yellow",
        SafetyLevel::Orange => "safety-dot orange",
        SafetyLevel::Red => "safety-dot red",
    };

    let tc = move || state.threat_count.get();

    view! {
        <div class=safety_class></div>
        {move || {
            let count = tc();
            if count > 0 {
                Some(view! { <span class="threat-badge">{format!("{} threats", count)}</span> })
            } else {
                None
            }
        }}
    }
}
