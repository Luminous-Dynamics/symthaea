// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Consciousness tier gating component.
//!
//! Shows children when the user meets the required tier, or shows
//! the growth distance when they don't. Uses CSS display toggling
//! so children are rendered once.

use leptos::prelude::*;
use personal_leptos_types::TrustTier;
use mycelix_leptos_core::use_consciousness;

#[component]
pub fn TierGate(
    min_tier: TrustTier,
    #[prop(into)]
    action: String,
    children: Children,
) -> impl IntoView {
    let consciousness = use_consciousness();

    let meets_tier = Memo::new(move |_| {
        consciousness.tier.get() >= min_tier
    });

    let rendered_children = children();

    view! {
        <div style=move || if meets_tier.get() { "display: contents" } else { "display: none" }>
            {rendered_children}
        </div>
        <div
            class="tier-gate-blocked"
            style=move || if meets_tier.get() { "display: none" } else { "display: block" }
        >
            {move || {
                let profile = consciousness.profile.get();
                let score = profile.combined_score();
                let needed = min_tier.min_score();
                let gap = needed - score;
                view! {
                    <div class="gate-message">
                        <span class="gate-action">{action.clone()}</span>
                        " requires "
                        <span class=format!("tier-badge tier-{}", min_tier.css_class())>
                            {min_tier.label()}
                        </span>
                        " tier"
                    </div>
                    <div class="gate-growth">
                        <div class="growth-bar">
                            <div
                                class="growth-fill"
                                style=format!("width: {}%", (score / needed * 100.0).min(100.0))
                            ></div>
                        </div>
                        <span class="growth-label">
                            {format!("{:.0}% of the way — grow {:.2} more", score / needed * 100.0, gap)}
                        </span>
                    </div>
                    <div class="gate-dimensions">
                        <span class="dim">{format!("identity: {:.2}", profile.identity)}</span>
                        <span class="dim">{format!("reputation: {:.2}", profile.reputation)}</span>
                        <span class="dim">{format!("community: {:.2}", profile.community)}</span>
                        <span class="dim">{format!("engagement: {:.2}", profile.engagement)}</span>
                    </div>
                }
            }}
        </div>
    }
}
