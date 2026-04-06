// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain context for Energy.

use leptos::prelude::*;
use energy_leptos_types::*;

#[derive(Clone)]
pub struct EnergyCtx {
    pub projects: RwSignal<Vec<EnergyProjectView>>,
    pub investments: RwSignal<Vec<InvestmentView>>,
    pub offers: RwSignal<Vec<TradeOfferView>>,
    pub trades: RwSignal<Vec<TradeView>>,
    pub contracts: RwSignal<Vec<RegenerativeContractView>>,
}

pub fn provide_energy_context() {
    let state = EnergyCtx {
        projects: RwSignal::new(mock_projects()),
        investments: RwSignal::new(mock_investments()),
        offers: RwSignal::new(mock_offers()),
        trades: RwSignal::new(vec![]),
        contracts: RwSignal::new(mock_contracts()),
    };
    provide_context(state);
}

pub fn use_energy_context() -> EnergyCtx {
    expect_context::<EnergyCtx>()
}

fn mock_projects() -> Vec<EnergyProjectView> {
    vec![
        EnergyProjectView {
            id: "EP-001".into(), name: "Komati Solar Farm".into(), project_type: ProjectType::Solar,
            location: ProjectLocationView { latitude: -25.4, longitude: 29.4, country: "ZA".into(), region: "Mpumalanga".into() },
            capacity_mw: 150.0, status: ProjectStatus::Operational, developer_did: "did:mycelix:dev-001".into(),
            community_did: Some("did:mycelix:komati-community".into()),
            financials: ProjectFinancialsView { total_cost: 2_500_000.0, funded_amount: 2_200_000.0, currency: "ZAR".into(), target_irr: 12.5, payback_years: 8.0, annual_revenue_estimate: 450_000.0 },
            phi_score: Some(0.72),
        },
        EnergyProjectView {
            id: "EP-002".into(), name: "West Coast Wind".into(), project_type: ProjectType::Wind,
            location: ProjectLocationView { latitude: -32.8, longitude: 17.9, country: "ZA".into(), region: "Western Cape".into() },
            capacity_mw: 280.0, status: ProjectStatus::Construction, developer_did: "did:mycelix:dev-002".into(),
            community_did: None,
            financials: ProjectFinancialsView { total_cost: 5_000_000.0, funded_amount: 3_500_000.0, currency: "ZAR".into(), target_irr: 14.0, payback_years: 7.0, annual_revenue_estimate: 900_000.0 },
            phi_score: None,
        },
        EnergyProjectView {
            id: "EP-003".into(), name: "Koeberg Battery Storage".into(), project_type: ProjectType::BatteryStorage,
            location: ProjectLocationView { latitude: -33.7, longitude: 18.4, country: "ZA".into(), region: "Western Cape".into() },
            capacity_mw: 60.0, status: ProjectStatus::Financing, developer_did: "did:mycelix:dev-001".into(),
            community_did: None,
            financials: ProjectFinancialsView { total_cost: 800_000.0, funded_amount: 200_000.0, currency: "ZAR".into(), target_irr: 10.0, payback_years: 10.0, annual_revenue_estimate: 120_000.0 },
            phi_score: None,
        },
    ]
}

fn mock_investments() -> Vec<InvestmentView> {
    vec![
        InvestmentView { id: "INV-001".into(), project_id: "EP-001".into(), investor_did: "did:mycelix:user-001".into(), amount: 50_000.0, currency: "ZAR".into(), shares: 100.0, share_percentage: 2.0, investment_type: InvestmentType::CommunityShare, status: InvestmentStatus::Confirmed },
        InvestmentView { id: "INV-002".into(), project_id: "EP-002".into(), investor_did: "did:mycelix:user-001".into(), amount: 25_000.0, currency: "ZAR".into(), shares: 50.0, share_percentage: 0.5, investment_type: InvestmentType::Equity, status: InvestmentStatus::Pledged },
    ]
}

fn mock_offers() -> Vec<TradeOfferView> {
    vec![
        TradeOfferView { id: "OFF-001".into(), seller_did: "did:mycelix:producer-001".into(), project_id: Some("EP-001".into()), amount_kwh: 500.0, price_per_kwh: 1.85, currency: "ZAR".into(), status: OfferStatus::Active },
        TradeOfferView { id: "OFF-002".into(), seller_did: "did:mycelix:producer-002".into(), project_id: None, amount_kwh: 200.0, price_per_kwh: 2.10, currency: "ZAR".into(), status: OfferStatus::Active },
    ]
}

fn mock_contracts() -> Vec<RegenerativeContractView> {
    vec![
        RegenerativeContractView { id: "RC-001".into(), project_id: "EP-001".into(), community_did: "did:mycelix:komati-community".into(), current_ownership_percentage: 35.0, target_ownership_percentage: 100.0, reserve_account_balance: 180_000.0, currency: "ZAR".into(), status: ContractStatus::Active },
    ]
}
