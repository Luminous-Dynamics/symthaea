// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_energy_context;

#[component]
pub fn ProjectsPage() -> impl IntoView {
    let ctx = use_energy_context();

    view! {
        <div class="page-projects">
            <h1>"Energy Projects"</h1>
            <p class="subtitle">"Renewable energy projects across South Africa."</p>

            <div class="project-grid">
                {move || ctx.projects.get().iter().map(|p| {
                    let name = p.name.clone();
                    let icon = p.project_type.icon();
                    let type_label = p.project_type.label();
                    let status = p.status.label();
                    let status_css = p.status.css_class();
                    let region = p.location.region.clone();
                    let mw = p.capacity_mw;
                    let funded_pct = if p.financials.total_cost > 0.0 {
                        p.financials.funded_amount / p.financials.total_cost * 100.0
                    } else { 0.0 };
                    let phi = p.phi_score;
                    view! {
                        <div class="project-card">
                            <div class="project-header">
                                <span class="project-icon">{icon}</span>
                                <div class="project-title-group">
                                    <h3 class="project-name">{name}</h3>
                                    <span class="project-type">{type_label}</span>
                                </div>
                                <span class=format!("badge badge-{status_css}")>{status}</span>
                            </div>
                            <div class="project-body">
                                <span class="project-stat">{format!("{mw:.0} MW | {region}")}</span>
                                <span class="project-stat">{format!("Funded: {funded_pct:.0}%")}</span>
                                {phi.map(|p| view! { <span class="project-stat">{format!("Phi: {p:.2}")}</span> })}
                            </div>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
