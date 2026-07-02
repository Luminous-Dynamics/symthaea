// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Climate action dispatch: mock branch (instant local mutation) vs
//! real branch (spawn_local zome call).

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use climate_leptos_types::*;
use mycelix_leptos_core::holochain_provider::use_holochain;
use mycelix_leptos_core::{use_toasts, ToastKind};
use crate::context::use_climate_context;

/// Create a new climate project.
pub fn create_project(
    name: String,
    project_type: ProjectType,
    country_code: String,
    region: String,
    latitude: f64,
    longitude: f64,
    expected_credits: f64,
) {
    let hc = use_holochain();
    let ctx = use_climate_context();
    let toasts = use_toasts();

    if hc.is_mock() {
        let id = format!("PRJ-{:03}", ctx.projects.get_untracked().len() + 1);
        ctx.projects.update(|projects| {
            projects.push(ClimateProjectView {
                id: id.clone(),
                name: name.clone(),
                project_type: project_type.clone(),
                location: LocationView {
                    country_code: country_code.clone(),
                    region: Some(region.clone()),
                    latitude,
                    longitude,
                },
                expected_credits,
                start_date: (js_sys::Date::now() / 1000.0) as i64,
                verifier_did: None,
                status: ProjectStatus::Proposed,
            });
        });
        ctx.projects_summary.update(|s| {
            s.total_projects += 1;
            s.proposed_count += 1;
            s.total_expected_credits += expected_credits;
        });
        toasts.push(
            format!("Project \"{name}\" has been planted"),
            ToastKind::Custom("climate".into()),
        );
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input {
                id: String,
                name: String,
                project_type: ProjectType,
                location: LocationView,
                expected_credits: f64,
                start_date: i64,
            }
            let input = Input {
                id: format!("PRJ-{}", (js_sys::Date::now() / 1000.0) as u64),
                name: name.clone(),
                project_type,
                location: LocationView {
                    country_code,
                    region: Some(region),
                    latitude,
                    longitude,
                },
                expected_credits,
                start_date: (js_sys::Date::now() / 1000.0) as i64,
            };
            match hc.call_zome_default::<_, serde_json::Value>(
                "projects", "create_climate_project", &input,
            ).await {
                Ok(_) => toasts.push(
                    format!("Project \"{name}\" has been planted"),
                    ToastKind::Custom("climate".into()),
                ),
                Err(e) => {
                    web_sys::console::log_1(&format!("create project failed: {e}").into());
                    toasts.push("Project could not take root", ToastKind::Error);
                }
            }
        });
    }
}

/// Retire a carbon credit (permanently offset).
pub fn retire_credit(credit_id: String) {
    let hc = use_holochain();
    let ctx = use_climate_context();
    let toasts = use_toasts();

    if hc.is_mock() {
        ctx.credits.update(|credits| {
            if let Some(c) = credits.iter_mut().find(|c| c.id == credit_id) {
                c.status = CreditStatus::Retired;
                c.retired_at = Some((js_sys::Date::now() / 1000.0) as i64);
            }
        });
        ctx.credit_summary.update(|s| {
            s.retired_tonnes += 1.0; // simplified
        });
        toasts.push("Credit retired — carbon offset permanent", ToastKind::Success);
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input { credit_action_hash: String }
            match hc.call_zome_default::<_, serde_json::Value>(
                "carbon", "retire_credit", &Input { credit_action_hash: credit_id },
            ).await {
                Ok(_) => toasts.push("Credit retired — carbon offset permanent", ToastKind::Success),
                Err(e) => {
                    web_sys::console::log_1(&format!("retire credit failed: {e}").into());
                    toasts.push("Retirement could not be recorded", ToastKind::Error);
                }
            }
        });
    }
}

/// Record a carbon footprint measurement.
pub fn record_footprint(
    scope1: f64,
    scope2: f64,
    scope3: f64,
    methodology: String,
) {
    let hc = use_holochain();
    let ctx = use_climate_context();
    let toasts = use_toasts();
    let total = scope1 + scope2 + scope3;

    if hc.is_mock() {
        ctx.footprints.update(|fps| {
            fps.push(CarbonFootprintView {
                entity_did: mycelix_leptos_core::local_did(),
                period_start: (js_sys::Date::now() / 1000.0) as i64,
                period_end: (js_sys::Date::now() / 1000.0) as i64 + 31536000,
                scope1, scope2, scope3,
                methodology: methodology.clone(),
                verified_by: None,
            });
        });
        toasts.push(
            format!("{total:.1} tCO2e recorded — awareness is the first step"),
            ToastKind::Custom("climate".into()),
        );
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input {
                entity_did: String,
                period_start: i64,
                period_end: i64,
                scope1: f64, scope2: f64, scope3: f64,
                methodology: String,
                verified_by: Option<String>,
            }
            let now = (js_sys::Date::now() / 1000.0) as i64;
            let input = Input {
                entity_did: mycelix_leptos_core::local_did(),
                period_start: now, period_end: now + 31536000,
                scope1, scope2, scope3,
                methodology,
                verified_by: None,
            };
            match hc.call_zome_default::<_, serde_json::Value>(
                "carbon", "create_carbon_footprint", &input,
            ).await {
                Ok(_) => toasts.push(
                    format!("{total:.1} tCO2e recorded"),
                    ToastKind::Custom("climate".into()),
                ),
                Err(e) => {
                    web_sys::console::log_1(&format!("record footprint failed: {e}").into());
                    toasts.push("Footprint could not be recorded", ToastKind::Error);
                }
            }
        });
    }
}
