// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_energy_context;

#[component]
pub fn RegenerativePage() -> impl IntoView {
    let ctx = use_energy_context();

    view! {
        <div class="page-regenerative">
            <h1>"Regenerative Ownership"</h1>
            <p class="subtitle">"Community ownership transition for energy projects."</p>

            {move || ctx.contracts.get().iter().map(|c| {
                let project = c.project_id.clone();
                let community = c.community_did.clone();
                let current = c.current_ownership_percentage;
                let target = c.target_ownership_percentage;
                let progress = current / target * 100.0;
                let reserve = c.reserve_account_balance;
                let currency = c.currency.clone();
                let status = c.status.label();
                view! {
                    <div class="regen-card">
                        <div class="regen-header">
                            <h3>{format!("Project {project}")}</h3>
                            <span class="badge">{status}</span>
                        </div>
                        <div class="regen-progress">
                            <span class="regen-label">"Ownership Progress"</span>
                            <div class="progress-bar-outer">
                                <div class="progress-bar-fill" style=format!("width: {progress:.0}%")></div>
                            </div>
                            <span class="regen-pct">{format!("{current:.0}% / {target:.0}%")}</span>
                        </div>
                        <div class="regen-meta">
                            <span>{format!("Reserve: {currency} {reserve:.0}")}</span>
                            <span class="regen-community">{format!("Community: {community}")}</span>
                        </div>
                    </div>
                }
            }).collect_view()}
        </div>
    }
}
