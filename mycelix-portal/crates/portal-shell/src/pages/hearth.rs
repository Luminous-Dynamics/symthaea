// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Hearth domain pages — kinship and care dashboard.

use leptos::prelude::*;

#[component]
pub fn HearthOverview() -> impl IntoView {
    view! {
        <div class="hearth-content">
            <div class="governance-nav">
                <button class="domain-nav-btn active">"Kinship"</button>
                <button class="domain-nav-btn">"Gratitude"</button>
                <button class="domain-nav-btn">"Care Circle"</button>
                <button class="domain-nav-btn">"Stories"</button>
                <button class="domain-nav-btn">"Milestones"</button>
            </div>

            <div class="commons-stats-grid">
                <div class="thought-card">
                    <div class="thought-type" style="color: #DB2777">"BONDS"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"14"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Active kinship bonds"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #F472B6">"GRATITUDE"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"8"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Notes this month"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #ec4899">"STORIES"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"23"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"In the family archive"</p>
                </div>
                <div class="thought-card">
                    <div class="thought-type" style="color: #f9a8d4">"MILESTONES"</div>
                    <p class="thought-content" style="font-size: 1.8rem; font-weight: 700">"5"</p>
                    <p style="font-size: 0.7rem; color: var(--text-muted)">"Upcoming celebrations"</p>
                </div>
            </div>

            <div class="thought-card" style="margin-top: 1rem;">
                <h3 class="section-title">"Recent Gratitude"</h3>
                <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                    <p style="font-size: 0.8rem; line-height: 1.5;">
                        <span style="color: #F472B6; font-weight: 600">"Elena"</span>
                        " \u{2192} "
                        <span style="color: #DB2777; font-weight: 600">"Marcus"</span>
                        ": \"Thank you for watching the kids while I was at the council meeting.\""
                    </p>
                    <p style="font-size: 0.8rem; line-height: 1.5;">
                        <span style="color: #F472B6; font-weight: 600">"Ama"</span>
                        " \u{2192} "
                        <span style="color: #DB2777; font-weight: 600">"Community"</span>
                        ": \"The food distribution yesterday was beautiful. We are fed.\""
                    </p>
                </div>
            </div>
        </div>
    }
}
