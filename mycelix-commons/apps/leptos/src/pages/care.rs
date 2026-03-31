// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;

#[component]
pub fn CarePage() -> impl IntoView {
    view! {
        <div class="page care-page">
            <h1>"Mutual Care Network"</h1>
            <p class="page-desc">"Give and receive care within your community. Every commitment is recorded on your local chain."</p>

            <section class="care-stats">
                <div class="stat-card">
                    <span class="stat-value">"1,203"</span>
                    <span class="stat-label">"Active Commitments"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">"456"</span>
                    <span class="stat-label">"Hours Given This Month"</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">"98%"</span>
                    <span class="stat-label">"Fulfillment Rate"</span>
                </div>
            </section>

            <section class="care-categories">
                <h2>"Care Categories"</h2>
                <div class="category-grid">
                    <div class="category-card">"Childcare — 89 active"</div>
                    <div class="category-card">"Elder Care — 67 active"</div>
                    <div class="category-card">"Health Support — 45 active"</div>
                    <div class="category-card">"Education — 112 active"</div>
                    <div class="category-card">"Emergency — 23 active"</div>
                    <div class="category-card">"Household — 156 active"</div>
                </div>
            </section>

            <section class="recent-activity">
                <h2>"Recent Care Activity"</h2>
                <div class="activity-list">
                    <div class="activity-item">
                        <span class="activity-type">"Childcare"</span>
                        <span>"After-school pickup — 3 hours"</span>
                        <span class="activity-status fulfilled">"Fulfilled"</span>
                    </div>
                    <div class="activity-item">
                        <span class="activity-type">"Elder Care"</span>
                        <span>"Weekly shopping assistance"</span>
                        <span class="activity-status pending">"Pending"</span>
                    </div>
                </div>
            </section>
        </div>
    }
}
