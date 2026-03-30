// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use leptos::prelude::*;

#[component]
pub fn DisclosurePage() -> impl IntoView {
    view! {
        <div class="page disclosure-page">
            <h1 class="page-title">"disclosure settings"</h1>
            <p class="page-subtitle">"control what you share across clusters"</p>

            <div class="disclosure-info">
                <p>"Your personal data lives on your source chain. Other clusters can only see what you explicitly allow."</p>
            </div>

            <section>
                <h2>"Cross-Cluster Sharing"</h2>

                <div class="disclosure-row">
                    <span class="disclosure-label">"Identity (Profile)"</span>
                    <span class="disclosure-scope">"Selected Fields"</span>
                    <span class="disclosure-detail">"display_name, pronouns"</span>
                </div>

                <div class="disclosure-row">
                    <span class="disclosure-label">"Health Records"</span>
                    <span class="disclosure-scope">"Existence Only"</span>
                    <span class="disclosure-detail">"Other clusters can only see that records exist"</span>
                </div>

                <div class="disclosure-row">
                    <span class="disclosure-label">"Governance Credentials"</span>
                    <span class="disclosure-scope">"Full"</span>
                    <span class="disclosure-detail">"Required for voting eligibility"</span>
                </div>

                <div class="disclosure-row">
                    <span class="disclosure-label">"K-Vector Trust"</span>
                    <span class="disclosure-scope">"Tier Only"</span>
                    <span class="disclosure-detail">"Tier is visible; exact score range is private"</span>
                </div>
            </section>
        </div>
    }
}
