// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mock data for Energy cluster — global renewable projects and P2P trading.

use energy_leptos_types::*;

pub fn projects() -> Vec<EnergyProjectView> {
    vec![
        EnergyProjectView {
            id: "EP-001".into(),
            name: "Sahara Desert Solar Array".into(),
            project_type: ProjectType::Solar,
            location: ProjectLocationView {
                latitude: 23.4,
                longitude: 3.8,
                country: "DZ".into(),
                region: "Adrar".into(),
            },
            capacity_mw: 500.0,
            status: ProjectStatus::Operational,
            developer_did: "did:mycelix:dev-001".into(),
            community_did: Some("did:mycelix:adrar-community".into()),
            financials: ProjectFinancialsView {
                total_cost: 12_000_000.0,
                funded_amount: 11_000_000.0,
                currency: "USD".into(),
                target_irr: 14.0,
                payback_years: 7.0,
                annual_revenue_estimate: 2_200_000.0,
            },
            phi_score: Some(0.82),
        },
        EnergyProjectView {
            id: "EP-002".into(),
            name: "Patagonia Wind Farm".into(),
            project_type: ProjectType::Wind,
            location: ProjectLocationView {
                latitude: -51.6,
                longitude: -69.2,
                country: "AR".into(),
                region: "Santa Cruz".into(),
            },
            capacity_mw: 320.0,
            status: ProjectStatus::Construction,
            developer_did: "did:mycelix:dev-002".into(),
            community_did: None,
            financials: ProjectFinancialsView {
                total_cost: 8_000_000.0,
                funded_amount: 5_500_000.0,
                currency: "USD".into(),
                target_irr: 16.0,
                payback_years: 6.0,
                annual_revenue_estimate: 1_800_000.0,
            },
            phi_score: None,
        },
        EnergyProjectView {
            id: "EP-003".into(),
            name: "Kerala Tidal Battery".into(),
            project_type: ProjectType::BatteryStorage,
            location: ProjectLocationView {
                latitude: 9.9,
                longitude: 76.3,
                country: "IN".into(),
                region: "Kerala".into(),
            },
            capacity_mw: 80.0,
            status: ProjectStatus::Financing,
            developer_did: "did:mycelix:dev-003".into(),
            community_did: Some("did:mycelix:kerala-coop".into()),
            financials: ProjectFinancialsView {
                total_cost: 3_200_000.0,
                funded_amount: 800_000.0,
                currency: "USD".into(),
                target_irr: 11.0,
                payback_years: 9.0,
                annual_revenue_estimate: 450_000.0,
            },
            phi_score: None,
        },
        EnergyProjectView {
            id: "EP-004".into(),
            name: "Norwegian Fjord Hydro".into(),
            project_type: ProjectType::Hydro,
            location: ProjectLocationView {
                latitude: 61.2,
                longitude: 7.1,
                country: "NO".into(),
                region: "Sogn og Fjordane".into(),
            },
            capacity_mw: 200.0,
            status: ProjectStatus::Operational,
            developer_did: "did:mycelix:dev-004".into(),
            community_did: Some("did:mycelix:fjord-commons".into()),
            financials: ProjectFinancialsView {
                total_cost: 6_000_000.0,
                funded_amount: 6_000_000.0,
                currency: "USD".into(),
                target_irr: 10.0,
                payback_years: 12.0,
                annual_revenue_estimate: 800_000.0,
            },
            phi_score: Some(0.91),
        },
    ]
}

pub fn investments() -> Vec<InvestmentView> {
    vec![
        InvestmentView {
            id: "INV-001".into(),
            project_id: "EP-001".into(),
            investor_did: "did:mycelix:user-001".into(),
            amount: 5_000.0,
            currency: "USD".into(),
            shares: 100.0,
            share_percentage: 0.04,
            investment_type: InvestmentType::CommunityShare,
            status: InvestmentStatus::Confirmed,
        },
        InvestmentView {
            id: "INV-002".into(),
            project_id: "EP-002".into(),
            investor_did: "did:mycelix:user-001".into(),
            amount: 2_500.0,
            currency: "USD".into(),
            shares: 50.0,
            share_percentage: 0.03,
            investment_type: InvestmentType::Equity,
            status: InvestmentStatus::Pledged,
        },
    ]
}

pub fn offers() -> Vec<TradeOfferView> {
    vec![
        TradeOfferView {
            id: "OFF-001".into(),
            seller_did: "did:mycelix:producer-001".into(),
            project_id: Some("EP-001".into()),
            amount_kwh: 5000.0,
            price_per_kwh: 0.08,
            currency: "USD".into(),
            status: OfferStatus::Active,
        },
        TradeOfferView {
            id: "OFF-002".into(),
            seller_did: "did:mycelix:producer-002".into(),
            project_id: Some("EP-004".into()),
            amount_kwh: 2000.0,
            price_per_kwh: 0.12,
            currency: "USD".into(),
            status: OfferStatus::Active,
        },
    ]
}

pub fn contracts() -> Vec<RegenerativeContractView> {
    vec![RegenerativeContractView {
        id: "RC-001".into(),
        project_id: "EP-001".into(),
        community_did: "did:mycelix:adrar-community".into(),
        current_ownership_percentage: 35.0,
        target_ownership_percentage: 100.0,
        reserve_account_balance: 180_000.0,
        currency: "USD".into(),
        status: ContractStatus::Active,
    }]
}
