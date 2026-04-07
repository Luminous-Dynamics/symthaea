// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use crate::context::use_climate_context;
use crate::actions;

#[component]
pub fn EmissionsPage() -> impl IntoView {
    let ctx = use_climate_context();

    // Footprint form state
    let (s1, set_s1) = signal(String::new());
    let (s2, set_s2) = signal(String::new());
    let (s3, set_s3) = signal(String::new());
    let (method, set_method) = signal("GHG Protocol".to_string());
    let (show_form, set_show_form) = signal(false);

    view! {
        <div class="page-emissions">
            <h1>"Emissions Tracking"</h1>
            <p class="subtitle">"Scope 1, 2, and 3 emissions reporting."</p>

            <div class="action-section">
                <button class="btn btn-primary"
                    on:click=move |_| set_show_form.update(|s| *s = !*s)
                >
                    {move || if show_form.get() { "Cancel" } else { "Record Footprint" }}
                </button>

                <div style=move || if show_form.get() { "display: block" } else { "display: none" }>
                    <form class="create-form" on:submit=move |ev: leptos::ev::SubmitEvent| {
                        ev.prevent_default();
                        let scope1 = s1.get_untracked().parse::<f64>().unwrap_or(0.0);
                        let scope2 = s2.get_untracked().parse::<f64>().unwrap_or(0.0);
                        let scope3 = s3.get_untracked().parse::<f64>().unwrap_or(0.0);
                        if scope1 + scope2 + scope3 > 0.0 {
                            actions::record_footprint(scope1, scope2, scope3, method.get_untracked());
                            set_s1.set(String::new());
                            set_s2.set(String::new());
                            set_s3.set(String::new());
                            set_show_form.set(false);
                        }
                    }>
                        <div class="form-field">
                            <label>"Scope 1 (direct, tCO2e)"</label>
                            <input class="form-input" type="number" step="0.1" min="0" placeholder="12.5"
                                prop:value=move || s1.get()
                                on:input=move |ev| set_s1.set(event_target_value(&ev))
                            />
                        </div>
                        <div class="form-field">
                            <label>"Scope 2 (energy, tCO2e)"</label>
                            <input class="form-input" type="number" step="0.1" min="0" placeholder="8.3"
                                prop:value=move || s2.get()
                                on:input=move |ev| set_s2.set(event_target_value(&ev))
                            />
                        </div>
                        <div class="form-field">
                            <label>"Scope 3 (indirect, tCO2e)"</label>
                            <input class="form-input" type="number" step="0.1" min="0" placeholder="45.2"
                                prop:value=move || s3.get()
                                on:input=move |ev| set_s3.set(event_target_value(&ev))
                            />
                        </div>
                        <div class="form-field">
                            <label>"Methodology"</label>
                            <select class="form-select"
                                on:change=move |ev| set_method.set(event_target_value(&ev))
                            >
                                <option value="GHG Protocol">"GHG Protocol"</option>
                                <option value="ISO 14064">"ISO 14064"</option>
                                <option value="CDP">"CDP"</option>
                            </select>
                        </div>
                        <button type="submit" class="btn btn-primary">"Record footprint"</button>
                    </form>
                </div>
            </div>

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
