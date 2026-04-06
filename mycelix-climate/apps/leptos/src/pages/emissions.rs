// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_climate_context;

#[component]
pub fn EmissionsPage() -> impl IntoView {
    let ctx = use_climate_context();

    view! {
        <div class="page-emissions">
            <h1>"Emissions Tracking"</h1>
            <p class="subtitle">"Scope 1, 2, and 3 emissions reporting."</p>

            {move || {
                let footprints = ctx.footprints.get();
                if footprints.is_empty() {
                    view! {
                        <div class="empty-state">
                            <div class="empty-state-icon">"📊"</div>
                            <h3 class="empty-state-title">"No emissions data"</h3>
                            <p class="empty-state-desc">"Record your first carbon footprint to start tracking."</p>
                        </div>
                    }.into_any()
                } else {
                    footprints.iter().map(|f| {
                        let total = f.total();
                        let s1 = f.scope1;
                        let s2 = f.scope2;
                        let s3 = f.scope3;
                        let method = f.methodology.clone();
                        let verified = f.verified_by.is_some();
                        view! {
                            <div class="emissions-card">
                                <div class="emissions-total">
                                    <span class="emissions-value">{format!("{total:.1}")}</span>
                                    <span class="emissions-unit">"tCO2e total"</span>
                                </div>
                                <div class="scope-breakdown">
                                    <div class="scope-bar">
                                        <div class="scope-fill scope1" style=format!("width: {}%", s1 / total * 100.0)></div>
                                        <div class="scope-fill scope2" style=format!("width: {}%", s2 / total * 100.0)></div>
                                        <div class="scope-fill scope3" style=format!("width: {}%", s3 / total * 100.0)></div>
                                    </div>
                                    <div class="scope-legend">
                                        <span class="scope-item"><span class="dot scope1-dot"></span>{format!("Scope 1: {s1:.1}t")}</span>
                                        <span class="scope-item"><span class="dot scope2-dot"></span>{format!("Scope 2: {s2:.1}t")}</span>
                                        <span class="scope-item"><span class="dot scope3-dot"></span>{format!("Scope 3: {s3:.1}t")}</span>
                                    </div>
                                </div>
                                <div class="emissions-meta">
                                    <span>{format!("Methodology: {method}")}</span>
                                    {verified.then(|| view! { <span class="badge badge-success">"Verified"</span> })}
                                </div>
                            </div>
                        }
                    }).collect_view().into_any()
                }
            }}
        </div>
    }
}
