// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use climate_leptos_types::ProjectType;
use crate::context::use_climate_context;
use crate::actions;

#[component]
pub fn ProjectsPage() -> impl IntoView {
    let ctx = use_climate_context();

    // Form state
    let (name, set_name) = signal(String::new());
    let (project_type, set_project_type) = signal("Reforestation".to_string());
    let (region, set_region) = signal(String::new());
    let (credits, set_credits) = signal(String::new());
    let (show_form, set_show_form) = signal(false);

    let can_submit = Memo::new(move |_| {
        !name.get().trim().is_empty()
            && !region.get().trim().is_empty()
            && credits.get().parse::<f64>().unwrap_or(0.0) > 0.0
    });

    let on_submit = move |ev: leptos::ev::SubmitEvent| {
        ev.prevent_default();
        if can_submit.get_untracked() {
            let pt = match project_type.get_untracked().as_str() {
                "RenewableEnergy" => ProjectType::RenewableEnergy,
                "MethaneCapture" => ProjectType::MethaneCapture,
                "OceanRestoration" => ProjectType::OceanRestoration,
                "DirectAirCapture" => ProjectType::DirectAirCapture,
                _ => ProjectType::Reforestation,
            };
            actions::create_project(
                name.get_untracked(),
                pt,
                "ZA".into(),
                region.get_untracked(),
                -30.0, 25.0, // default SA coordinates
                credits.get_untracked().parse().unwrap_or(1000.0),
            );
            set_name.set(String::new());
            set_region.set(String::new());
            set_credits.set(String::new());
            set_show_form.set(false);
        }
    };

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

            // Create project button + form
            <div class="action-section">
                <button
                    class="btn btn-primary"
                    on:click=move |_| set_show_form.update(|s| *s = !*s)
                >
                    {move || if show_form.get() { "Cancel" } else { "Create Project" }}
                </button>

                <div style=move || if show_form.get() { "display: block" } else { "display: none" }>
                    <form class="create-form" on:submit=on_submit>
                        <div class="form-field">
                            <label>"Project Name"</label>
                            <input class="form-input" type="text" placeholder="e.g. Karoo Reforestation"
                                prop:value=move || name.get()
                                on:input=move |ev| set_name.set(event_target_value(&ev))
                            />
                        </div>
                        <div class="form-field">
                            <label>"Type"</label>
                            <select class="form-select"
                                on:change=move |ev| set_project_type.set(event_target_value(&ev))
                            >
                                <option value="Reforestation">"Reforestation"</option>
                                <option value="RenewableEnergy">"Renewable Energy"</option>
                                <option value="MethaneCapture">"Methane Capture"</option>
                                <option value="OceanRestoration">"Ocean Restoration"</option>
                                <option value="DirectAirCapture">"Direct Air Capture"</option>
                            </select>
                        </div>
                        <div class="form-field">
                            <label>"Region"</label>
                            <input class="form-input" type="text" placeholder="e.g. Northern Cape"
                                prop:value=move || region.get()
                                on:input=move |ev| set_region.set(event_target_value(&ev))
                            />
                        </div>
                        <div class="form-field">
                            <label>"Expected Credits (tCO2e)"</label>
                            <input class="form-input" type="number" step="100" min="1" placeholder="10000"
                                prop:value=move || credits.get()
                                on:input=move |ev| set_credits.set(event_target_value(&ev))
                            />
                        </div>
                        <button type="submit" class="btn btn-primary" disabled=move || !can_submit.get()>
                            "Plant this project"
                        </button>
                    </form>
                </div>
            </div>

            // Project list
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
