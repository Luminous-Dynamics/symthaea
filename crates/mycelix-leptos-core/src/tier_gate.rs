// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Civic tier gating component (8D Sovereign Profile).
//!
//! Shows children when the user meets the required tier, or shows
//! a growth progress view with the 8-dimensional breakdown when
//! they don't. Uses CSS display toggling so children are rendered once.

use crate::consciousness::{
    combined_score, use_consciousness, SovereignDimension, DIMENSION_LABELS,
};
use leptos::prelude::*;
use personal_leptos_types::TrustTier;

/// Gate content behind a minimum civic tier.
///
/// When the user's tier is below `min_tier`, shows a progress bar
/// with the 8D dimension breakdown and growth distance.
#[component]
pub fn TierGate(
    min_tier: TrustTier,
    #[prop(into)] action: String,
    children: Children,
) -> impl IntoView {
    let consciousness = use_consciousness();

    let meets_tier = Memo::new(move |_| consciousness.tier.get() >= min_tier);

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
                let score = combined_score(&profile);
                let needed = min_tier.min_score();
                let gap = (needed - score).max(0.0);
                let pct = if needed > 0.0 { (score / needed * 100.0).min(100.0) } else { 100.0 };

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
                                style=format!("width: {pct:.0}%")
                            ></div>
                        </div>
                        <span class="growth-label">
                            {format!("{pct:.0}% of the way \u{2014} grow {gap:.2} more")}
                        </span>
                    </div>
                    <div class="gate-dimensions gate-dimensions-8d">
                        {SovereignDimension::ALL.iter().map(|dim| {
                            let val = profile.get(*dim);
                            let label = &DIMENSION_LABELS[dim.index()];
                            let bar_width = (val * 100.0).clamp(0.0, 100.0);
                            view! {
                                <div class="dim-row" title={label.description_en}>
                                    <span class="dim-icon">{label.icon}</span>
                                    <span class="dim-name">{label.name_en}</span>
                                    <div class="dim-bar">
                                        <div
                                            class="dim-fill"
                                            style=format!("width: {bar_width:.0}%")
                                        ></div>
                                    </div>
                                    <span class="dim-value">{format!("{val:.0}%")}</span>
                                </div>
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                }
            }}
        </div>
    }
}
