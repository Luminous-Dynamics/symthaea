// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Applications page — track job application lifecycle.

use leptos::prelude::*;

#[component]
pub fn ApplicationsPage() -> impl IntoView {
    view! {
        <div class="page applications-page">
            <h1>"My Applications"</h1>

            <div class="app-pipeline" role="list" aria-label="Application pipeline">
                <div class="pipeline-stage" role="listitem">
                    <h3>"Draft"</h3>
                    <p class="stage-count">"0"</p>
                </div>
                <div class="pipeline-stage" role="listitem">
                    <h3>"Submitted"</h3>
                    <p class="stage-count">"0"</p>
                </div>
                <div class="pipeline-stage" role="listitem">
                    <h3>"Under Review"</h3>
                    <p class="stage-count">"0"</p>
                </div>
                <div class="pipeline-stage" role="listitem">
                    <h3>"Interview"</h3>
                    <p class="stage-count">"0"</p>
                </div>
                <div class="pipeline-stage" role="listitem">
                    <h3>"Offered"</h3>
                    <p class="stage-count">"0"</p>
                </div>
            </div>

            <div class="applications-list">
                <p class="empty-state">"No applications yet. Browse jobs to apply."</p>
            </div>
        </div>
    }
}
