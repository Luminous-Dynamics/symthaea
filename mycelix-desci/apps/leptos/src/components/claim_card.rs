// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use crate::types::ClaimResponse;
use crate::components::lem_badge::LemBadge;

/// Claim summary card for grid display.
#[component]
pub fn ClaimCard(claim: ClaimResponse) -> impl IntoView {
    let description = claim.content.description.clone();
    let category = claim.content.category.clone();
    let tier = claim.tier;
    let verifications = claim.verifications_count;
    let id = claim.id.clone();
    let created = claim.created_at.chars().take(10).collect::<String>();

    view! {
        <a href=format!("/claims/{}", id) style="text-decoration: none; color: inherit;">
            <div class="glass-panel claim-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <LemBadge tier=tier />
                    <span style="font-size: 0.75rem; color: var(--text-secondary);">{category}</span>
                </div>
                <p class="claim-description">{description}</p>
                <div class="claim-meta">
                    <span>{verifications} " verifications"</span>
                    <span>" | "</span>
                    <span>{created}</span>
                </div>
            </div>
        </a>
    }
}
