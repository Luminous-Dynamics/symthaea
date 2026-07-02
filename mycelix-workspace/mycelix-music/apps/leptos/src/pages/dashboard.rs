// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;
use mycelix_leptos_core::{StatCard, SovereignRadar, SovereignRadarSize};

#[component]
pub fn DashboardPage() -> impl IntoView {
    // TODO: Replace with use_zome_call for real data
    let total_plays = RwSignal::new(0u64);
    let total_earnings = RwSignal::new(0u64);
    let unsettled_plays = RwSignal::new(0u64);
    let trust_score = RwSignal::new(0u32);

    view! {
        <div class="page dashboard-page">
            <h1>"Artist Dashboard"</h1>

            <div class="stats-grid">
                <StatCard label="Total Plays" value=total_plays.get().to_string() />
                <StatCard label="Total Earnings" value=format!("{} wei", total_earnings.get()) />
                <StatCard label="Unsettled Plays" value=unsettled_plays.get().to_string() />
                <StatCard label="Trust Score" value=format!("{}/1000", trust_score.get()) />
            </div>

            <section class="settlement-section">
                <h2>"Settlement"</h2>
                <p>"Batch your unsettled plays for on-chain settlement."</p>
                <button class="btn btn-primary" disabled=move || unsettled_plays.get() == 0>
                    "Create Settlement Batch"
                </button>
                <p class="help-text">
                    "Settlement batches aggregate your plays into a single on-chain transaction, "
                    "minimizing gas fees while ensuring you get paid for every play."
                </p>
            </section>

            <section class="sovereign-section">
                <h2>"Civic Profile"</h2>
                <SovereignRadar size=SovereignRadarSize::Small />
            </section>

            <section class="verification-section">
                <h2>"Verification"</h2>
                <p>"Build your trust score through community vouches."</p>
                <div class="tier-progress">
                    <div class="tier-bar">
                        <span class="tier-label">"Unverified"</span>
                        <span class="tier-label">"Community"</span>
                        <span class="tier-label">"Trusted"</span>
                        <span class="tier-label">"Platform"</span>
                    </div>
                </div>
            </section>
        </div>
    }
}
