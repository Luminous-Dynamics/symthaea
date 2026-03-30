// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;
use crate::hearth_context::{use_hearth, member_name};
use crate::components::TierGate;
use personal_leptos_types::TrustTier;

#[component]
pub fn AutonomyPage() -> impl IntoView {
    let hearth = use_hearth();

    view! {
        <div class="page autonomy-page">
            <h1 class="page-title">"Autonomy"</h1>
            <p class="page-subtitle">"Graduated independence and capability growth"</p>

            // Guardian-only: approve capability requests
            <TierGate min_tier=TrustTier::Elevated action_label="manage autonomy profiles">
                <button class="action-btn">"Review Requests"</button>
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
                                let capabilities = p.capabilities.join(", ");
                                let restrictions = p.restrictions.join(", ");
                                view! {
                                    <div class="autonomy-card">
                                        <div class="autonomy-header">
                                            <span class="autonomy-member">{who}</span>
                                            <span class="autonomy-tier">{tier}</span>
                                        </div>
                                        <div class="autonomy-details">
                                            <div class="autonomy-can">
                                                <h4>"Can"</h4>
                                                <p>{capabilities}</p>
                                            </div>
                                            <div class="autonomy-cant">
                                                <h4>"Not yet"</h4>
                                                <p>{restrictions}</p>
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
