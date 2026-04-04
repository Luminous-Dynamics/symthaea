// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Trust / researcher reputation page.

use leptos::prelude::*;
use crate::types::*;
use crate::components::trust_card::TrustCard;

fn demo_trust() -> TrustScore {
    TrustScore {
        participant: "symthaea-nuclear-engine".into(),
        score: 0.82,
        confidence: 0.65,
        interaction_count: 47,
        expertise_domains: vec!["Nuclear Physics".into(), "Materials Science".into(), "Computational Physics".into()],
    }
}

fn demo_claims_by_tier() -> Vec<(String, usize)> {
    vec![("E0".into(), 2), ("E1".into(), 5), ("E2".into(), 12), ("E3".into(), 3), ("E4".into(), 1)]
}

#[component]
pub fn TrustProfilePage() -> impl IntoView {
    let trust = demo_trust();
    let tier_dist = demo_claims_by_tier();

    view! {
        <div class="page-container">
            <h1 class="page-title">"Trust Profile"</h1>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem;">
                <TrustCard trust=trust />

                <div class="glass-panel">
                    <h3 style="font-size: 1rem; margin-bottom: 0.75rem;">"Claims by Epistemic Tier"</h3>
                    {tier_dist.iter().map(|(tier, count)| {
                        let color = match tier.as_str() {
                            "E0" => "var(--tier-e0)", "E1" => "var(--tier-e1)",
                            "E2" => "var(--tier-e2)", "E3" => "var(--tier-e3)",
                            "E4" => "var(--tier-e4)", _ => "var(--text-secondary)",
                        };
                        view! {
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                                <span style=format!("color: {}; font-weight: 600; font-size: 0.875rem;", color)>{tier.clone()}</span>
                                <div style="flex: 1; margin: 0 0.75rem; height: 6px; background: var(--bg-secondary); border-radius: 3px;">
                                    <div style=format!("width: {}%; height: 100%; background: {}; border-radius: 3px;", count * 5, color)></div>
                                </div>
                                <span style="font-size: 0.875rem; font-weight: 600;">{*count}</span>
                            </div>
                        }
                    }).collect::<Vec<_>>()}
                </div>
            </div>
        </div>
    }
}
