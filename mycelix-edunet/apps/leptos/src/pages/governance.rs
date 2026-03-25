// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use leptos::prelude::*;

#[component]
pub fn GovernancePage() -> impl IntoView {
    view! {
        <div class="stub-page">
            <h2>"Community Governance"</h2>
            <p>
                "Participate in curriculum decisions through quadratic voting. "
                "Propose new courses, vote on quality standards, and help shape "
                "the direction of the learning community."
            </p>
            <div class="planned-features">
                <h3>"Planned Features"</h3>
                <ul>
                    <li>"Proposal creation and discussion"</li>
                    <li>"Quadratic voting on curriculum changes"</li>
                    <li>"Three-tier proposal tracks (Fast/Normal/Slow)"</li>
                    <li>"Quorum tracking and vote results"</li>
                    <li>"DAO treasury overview"</li>
                </ul>
            </div>
        </div>
    }
}
