// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use leptos::prelude::*;
use climate_leptos_types::ProjectType;
use crate::context::use_climate_context;
use crate::actions;

#[component]
pub fn ProjectsPage() -> impl IntoView {
    let ctx = use_climate_context();
    let hc = mycelix_leptos_core::holochain_provider::use_holochain();

    // Credential check — async query to bridge zome
    let (has_credential, set_has_credential) = signal(true); // default: allow in mock
    wasm_bindgen_futures::spawn_local({
        let hc = hc.clone();
        async move {
            if !hc.is_mock() {
                let result = hc.call_zome_default::<String, bool>(
                    "bridge", "check_environmental_credential",
                    &"did:mycelix:user-001".to_string(),
                ).await.unwrap_or(true);
                set_has_credential.set(result);
            }
        }
    });

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

            // World map with project dots
            <div class="project-map-container world-map">
                <svg viewBox="0 0 800 400" class="project-map" xmlns="http://www.w3.org/2000/svg">
                    // Simplified world coastline (rectangles for continents)
                    <rect class="continent" x="120" y="60" width="180" height="120" rx="8" /> // Americas
                    <rect class="continent" x="340" y="40" width="120" height="100" rx="8" /> // Europe/Africa
                    <rect class="continent" x="340" y="150" width="100" height="140" rx="8" /> // Africa
                    <rect class="continent" x="480" y="50" width="180" height="120" rx="8" /> // Asia
                    <rect class="continent" x="580" y="220" width="100" height="80" rx="8" /> // Oceania
                    // Equator line
                    <line x1="50" y1="200" x2="750" y2="200" class="equator" />
                    // Project dots (reactive, Mercator-like projection)
                    {move || ctx.projects.get().iter().map(|p| {
                        // Map lat/lon to SVG: lon -180→180 → x 50→750, lat 80→-60 → y 20→380
                        let lat = p.location.latitude;
                        let lon = p.location.longitude;
                        let x = ((lon + 180.0) / 360.0 * 700.0 + 50.0) as f32;
                        let y = ((80.0 - lat) / 140.0 * 360.0 + 20.0) as f32;
                        let r = match p.status {
                            climate_leptos_types::ProjectStatus::Active => 8.0,
                            climate_leptos_types::ProjectStatus::Completed => 7.0,
                            climate_leptos_types::ProjectStatus::Proposed => 5.0,
                            _ => 6.0,
                        };
                        let color = match p.project_type {
                            climate_leptos_types::ProjectType::Reforestation => "#6abf69",
                            climate_leptos_types::ProjectType::RenewableEnergy => "#e8c547",
                            climate_leptos_types::ProjectType::OceanRestoration => "#5ba0c9",
                            climate_leptos_types::ProjectType::MethaneCapture => "#d4a574",
                            _ => "#4ecdc4",
                        };
                        let name = p.name.clone();
                        view! {
                            <circle cx=x.to_string() cy=y.to_string() r=r.to_string() fill=color opacity="0.85" class="project-dot">
                                <title>{name}</title>
                            </circle>
                            <circle cx=x.to_string() cy=y.to_string() r=(r + 3.0).to_string() fill="none" stroke=color stroke-width="1" opacity="0.3" class="project-dot-ring" />
                        }
                    }).collect_view()}
                </svg>
            </div>

            // Create project button + form (gated by Praxis credential)
            <div class="action-section">
                {move || if !has_credential.get() {
                    Some(view! {
                        <div class="credential-gate">
                            <span class="gate-icon">"🔒"</span>
                            <span>"Complete environmental science in "</span>
                            <a href="http://localhost:8107" class="cross-cluster-link">"Praxis"</a>
                            <span>" to create projects."</span>
                        </div>
                    })
                } else {
                    None
                }}
                <button
                    class="btn btn-primary"
                    disabled=move || !has_credential.get()
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
