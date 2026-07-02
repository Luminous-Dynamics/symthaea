// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use crate::hearth_context::{use_hearth, member_name};
use crate::components::TierGate;
use hearth_leptos_types::*;
use personal_leptos_types::TrustTier;

/// All autonomy tiers in order, for the progress visualization.
const TIERS: &[AutonomyTier] = &[
    AutonomyTier::Dependent,
    AutonomyTier::Supervised,
    AutonomyTier::Guided,
    AutonomyTier::SemiAutonomous,
    AutonomyTier::Autonomous,
];

fn tier_index(tier: &AutonomyTier) -> usize {
    TIERS.iter().position(|t| t == tier).unwrap_or(0)
}

#[component]
pub fn AutonomyPage() -> impl IntoView {
    let hearth = use_hearth();

    view! {
        <div class="page autonomy-page">
            <h1 class="page-title">"autonomy"</h1>
            <p class="page-subtitle">"graduated independence and capability growth"</p>

            <TierGate min_tier=TrustTier::Elevated action_label="manage autonomy profiles">
                <button class="action-btn" aria-label="review capability requests">"review requests"</button>
            </TierGate>

            {move || {
                let members = hearth.members.get();
                let profiles = hearth.autonomy_profiles.get();

                if profiles.is_empty() {
                    view! { <div class="empty-state">"independence grows slowly, like roots. no profiles yet."</div> }.into_any()
                } else {
                    view! {
                        <div class="autonomy-list">
                            {profiles.iter().map(|p| {
                                let who = member_name(&members, &p.member);
                                let tier = p.current_tier.label().to_string();
                                let idx = tier_index(&p.current_tier);
                                let progress_pct = ((idx as f64 + 1.0) / TIERS.len() as f64 * 100.0) as u32;
                                let capabilities = p.capabilities.clone();
                                let restrictions = p.restrictions.clone();
                                view! {
                                    <div class="autonomy-card">
                                        <div class="autonomy-header">
                                            <span class="autonomy-member">{who}</span>
                                            <span class="autonomy-tier">{tier}</span>
                                        </div>

                                        // Progress visualization
                                        <div class="autonomy-progress" aria-label="autonomy progress">
                                            <div class="autonomy-track">
                                                {TIERS.iter().enumerate().map(|(i, t)| {
                                                    let is_reached = i <= idx;
                                                    let label = t.label();
                                                    view! {
                                                        <div class=format!("autonomy-step {}", if is_reached { "reached" } else { "future" })>
                                                            <div class="step-dot"></div>
                                                            <span class="step-label">{label}</span>
                                                        </div>
                                                    }
                                                }).collect_view()}
                                            </div>
                                            <div class="autonomy-bar-container">
                                                <div class="autonomy-bar" style=format!("width: {}%", progress_pct)></div>
                                            </div>
                                        </div>

                                        <div class="autonomy-details">
                                            <div class="autonomy-can">
                                                <h4>"can"</h4>
                                                {capabilities.iter().map(|c| {
                                                    let cap = c.replace('_', " ");
                                                    view! { <span class="capability-tag">{cap}</span> }
                                                }).collect_view()}
                                            </div>
                                            <div class="autonomy-cant">
                                                <h4>"not yet"</h4>
                                                {restrictions.iter().map(|r| {
                                                    let res = r.replace('_', " ");
                                                    view! { <span class="restriction-tag">{res}</span> }
                                                }).collect_view()}
                                            </div>
                                        </div>
                                    </div>
                                }
                            }).collect_view()}
                        </div>
                    }.into_any()
                }
            }}
        </div>
    }
}
