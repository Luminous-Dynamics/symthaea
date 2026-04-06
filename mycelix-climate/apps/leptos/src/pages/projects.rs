// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_climate_context;

#[component]
pub fn ProjectsPage() -> impl IntoView {
    let ctx = use_climate_context();

    view! {
        <div class="page-projects">
            <h1>"Climate Projects"</h1>
            <p class="subtitle">"Active and proposed climate mitigation projects."</p>

            <div class="summary-bar">
                <div class="summary-item">
                    <span class="summary-label">"Total"</span>
                    <span class="summary-value">{move || ctx.projects_summary.get().total_projects.to_string()}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">"Active"</span>
                    <span class="summary-value accent">{move || ctx.projects_summary.get().active_count.to_string()}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">"Expected Credits"</span>
                    <span class="summary-value">{move || format!("{:.0} t", ctx.projects_summary.get().total_expected_credits)}</span>
                </div>
            </div>

            <div class="project-grid">
                {move || ctx.projects.get().iter().map(|p| {
                    let name = p.name.clone();
                    let icon = p.project_type.icon();
                    let type_label = p.project_type.label();
                    let status = p.status.label();
                    let status_css = p.status.css_class();
                    let region = p.location.region.clone().unwrap_or_default();
                    let country = p.location.country_code.clone();
                    let credits = p.expected_credits;
                    let verified = p.verifier_did.is_some();
                    view! {
                        <div class=format!("project-card {status_css}")>
                            <div class="project-header">
                                <span class="project-icon">{icon}</span>
                                <div class="project-title-group">
                                    <h3 class="project-name">{name}</h3>
                                    <span class="project-type">{type_label}</span>
                                </div>
                                <span class=format!("badge badge-{status_css}")>{status}</span>
                            </div>
                            <div class="project-body">
                                <span class="project-location">{format!("{region}, {country}")}</span>
                                <span class="project-credits">{format!("{credits:.0} tCO2e expected")}</span>
                            </div>
                            <div class="project-footer">
                                {verified.then(|| view! { <span class="badge badge-success">"Verified"</span> })}
                            </div>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
