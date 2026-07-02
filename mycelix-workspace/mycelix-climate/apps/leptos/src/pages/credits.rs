// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_climate_context;
use crate::actions;

#[component]
pub fn CreditsPage() -> impl IntoView {
    let ctx = use_climate_context();

    view! {
        <div class="page-credits">
            <h1>"Carbon Credits"</h1>
            <p class="subtitle">"Your carbon credit portfolio."</p>

            <div class="summary-bar">
                <div class="summary-item">
                    <span class="summary-label">"Total"</span>
                    <span class="summary-value">{move || format!("{:.0} t", ctx.credit_summary.get().total_tonnes)}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">"Active"</span>
                    <span class="summary-value accent">{move || format!("{:.0} t", ctx.credit_summary.get().active_tonnes)}</span>
                </div>
                <div class="summary-item">
                    <span class="summary-label">"Retired"</span>
                    <span class="summary-value">{move || format!("{:.0} t", ctx.credit_summary.get().retired_tonnes)}</span>
                </div>
            </div>

            <div class="credit-list">
                {move || ctx.credits.get().iter().map(|c| {
                    let id = c.id.clone();
                    let retire_id = id.clone();
                    let project = c.project_id.clone();
                    let tonnes = c.tonnes_co2e;
                    let year = c.vintage_year;
                    let status = c.status.label();
                    let css = c.status.css_class();
                    view! {
                        <div class="credit-card">
                            <div class="credit-header">
                                <span class="credit-id">{id}</span>
                                <span class=format!("badge badge-{css}")>{status}</span>
                            </div>
                            <div class="credit-body">
                                <span class="credit-tonnes">{format!("{tonnes:.0} tCO2e")}</span>
                                <span class="credit-meta">{format!("Vintage {year} | Project {project}")}</span>
                            </div>
                            {(css == "active").then(|| {
                                let retire_id = retire_id.clone();
                                view! {
                                    <button class="btn btn-ghost" on:click=move |_| {
                                        actions::retire_credit(retire_id.clone());
                                    }>
                                        "Retire this credit"
                                    </button>
                                }
                            })}
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
