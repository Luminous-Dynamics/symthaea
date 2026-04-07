// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Energy action dispatch: mock branch (instant local mutation) vs
//! real branch (spawn_local zome call).

use leptos::prelude::*;
use wasm_bindgen_futures::spawn_local;
use energy_leptos_types::*;
use mycelix_leptos_core::holochain_provider::use_holochain;
use mycelix_leptos_core::{use_toasts, ToastKind};
use crate::context::use_energy_context;

/// Register a new energy project.
pub fn register_project(
    name: String,
    project_type: ProjectType,
    region: String,
    capacity_mw: f64,
) {
    let hc = use_holochain();
    let ctx = use_energy_context();
    let toasts = use_toasts();

    if hc.is_mock() {
        let id = format!("ENR-{:03}", ctx.projects.get_untracked().len() + 1);
        ctx.projects.update(|projects| {
            projects.push(EnergyProjectView {
                id: id.clone(),
                name: name.clone(),
                project_type: project_type.clone(),
                location: ProjectLocationView {
                    latitude: 0.0,
                    longitude: 0.0,
                    country: String::new(),
                    region: region.clone(),
                },
                capacity_mw,
                status: ProjectStatus::Proposed,
                developer_did: "did:mycelix:user-001".into(),
                community_did: None,
                financials: ProjectFinancialsView {
                    total_cost: 0.0,
                    funded_amount: 0.0,
                    currency: "SAP".into(),
                    target_irr: 0.0,
                    payback_years: 0.0,
                    annual_revenue_estimate: 0.0,
                },
                phi_score: None,
            });
        });
        toasts.push(
            format!("{} MW {name} project registered", capacity_mw),
            ToastKind::Custom("energy".into()),
        );
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input {
                name: String,
                project_type: ProjectType,
                region: String,
                capacity_mw: f64,
            }
            let input = Input { name: name.clone(), project_type, region, capacity_mw };
            match hc.call_zome_default::<_, serde_json::Value>(
                "projects", "register_project", &input,
            ).await {
                Ok(_) => toasts.push(
                    format!("{capacity_mw} MW {name} project registered"),
                    ToastKind::Custom("energy".into()),
                ),
                Err(e) => {
                    web_sys::console::log_1(&format!("register project failed: {e}").into());
                    toasts.push("Project could not be registered", ToastKind::Error);
                }
            }
        });
    }
}

/// Create a trade offer on the energy grid.
pub fn create_trade_offer(amount_kwh: f64, price_per_kwh: f64) {
    let hc = use_holochain();
    let ctx = use_energy_context();
    let toasts = use_toasts();

    if hc.is_mock() {
        let id = format!("OFR-{:04}", ctx.offers.get_untracked().len() + 1);
        ctx.offers.update(|offers| {
            offers.push(TradeOfferView {
                id: id.clone(),
                seller_did: "did:mycelix:user-001".into(),
                project_id: None,
                amount_kwh,
                price_per_kwh,
                currency: "SAP".into(),
                status: OfferStatus::Active,
            });
        });
        toasts.push(
            format!("{amount_kwh} kWh offered at {price_per_kwh}/kWh"),
            ToastKind::Custom("energy".into()),
        );
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input {
                amount_kwh: f64,
                price_per_kwh: f64,
            }
            let input = Input { amount_kwh, price_per_kwh };
            match hc.call_zome_default::<_, serde_json::Value>(
                "grid", "create_trade_offer", &input,
            ).await {
                Ok(_) => toasts.push(
                    format!("{amount_kwh} kWh offered at {price_per_kwh}/kWh"),
                    ToastKind::Custom("energy".into()),
                ),
                Err(e) => {
                    web_sys::console::log_1(&format!("create trade offer failed: {e}").into());
                    toasts.push("Trade offer could not be created", ToastKind::Error);
                }
            }
        });
    }
}

/// Pledge an investment into an energy project.
pub fn pledge_investment(
    project_id: String,
    amount: f64,
    investment_type: InvestmentType,
) {
    let hc = use_holochain();
    let ctx = use_energy_context();
    let toasts = use_toasts();

    if hc.is_mock() {
        let id = format!("INV-{:04}", ctx.investments.get_untracked().len() + 1);
        ctx.investments.update(|investments| {
            investments.push(InvestmentView {
                id: id.clone(),
                project_id: project_id.clone(),
                investor_did: "did:mycelix:user-001".into(),
                amount,
                currency: "SAP".into(),
                shares: 0.0,
                share_percentage: 0.0,
                investment_type: investment_type.clone(),
                status: InvestmentStatus::Pledged,
            });
        });
        toasts.push(
            format!("{amount:.2} SAP pledged to {project_id}"),
            ToastKind::Custom("energy".into()),
        );
    } else {
        let hc = hc.clone();
        spawn_local(async move {
            #[derive(serde::Serialize)]
            struct Input {
                project_id: String,
                amount: f64,
                investment_type: InvestmentType,
            }
            let input = Input { project_id: project_id.clone(), amount, investment_type };
            match hc.call_zome_default::<_, serde_json::Value>(
                "investments", "pledge_investment", &input,
            ).await {
                Ok(_) => toasts.push(
                    format!("{amount:.2} SAP pledged to {project_id}"),
                    ToastKind::Custom("energy".into()),
                ),
                Err(e) => {
                    web_sys::console::log_1(&format!("pledge investment failed: {e}").into());
                    toasts.push("Investment pledge could not be recorded", ToastKind::Error);
                }
            }
        });
    }
}
