// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Domain context for Climate.

use leptos::prelude::*;
use climate_leptos_types::*;

#[derive(Clone)]
pub struct ClimateCtx {
    pub projects: RwSignal<Vec<ClimateProjectView>>,
    pub credits: RwSignal<Vec<CarbonCreditView>>,
    pub footprints: RwSignal<Vec<CarbonFootprintView>>,
    pub credit_summary: RwSignal<CreditSummaryView>,
    pub projects_summary: RwSignal<ProjectsSummaryView>,
}

pub fn provide_climate_context() {
    let state = ClimateCtx {
        projects: RwSignal::new(mock_projects()),
        credits: RwSignal::new(mock_credits()),
        footprints: RwSignal::new(mock_footprints()),
        credit_summary: RwSignal::new(CreditSummaryView {
            total_credits: 12, total_tonnes: 4850.0,
            active_tonnes: 3200.0, retired_tonnes: 1650.0, transferred_count: 3,
        }),
        projects_summary: RwSignal::new(ProjectsSummaryView {
            total_projects: 4, proposed_count: 1, verified_count: 1,
            active_count: 1, completed_count: 1, total_expected_credits: 25000.0,
        }),
    };
    provide_context(state);
}

pub fn use_climate_context() -> ClimateCtx {
    expect_context::<ClimateCtx>()
}

fn mock_projects() -> Vec<ClimateProjectView> {
    vec![
        ClimateProjectView {
            id: "PRJ-001".into(), name: "Great Karoo Reforestation".into(),
            project_type: ProjectType::Reforestation,
            location: LocationView { country_code: "ZA".into(), region: Some("Northern Cape".into()), latitude: -31.5, longitude: 19.0 },
            expected_credits: 12000.0, start_date: 1704067200,
            verifier_did: Some("did:mycelix:verifier-001".into()), status: ProjectStatus::Active,
        },
        ClimateProjectView {
            id: "PRJ-002".into(), name: "Cape Town Solar Farm".into(),
            project_type: ProjectType::RenewableEnergy,
            location: LocationView { country_code: "ZA".into(), region: Some("Western Cape".into()), latitude: -33.9, longitude: 18.4 },
            expected_credits: 8000.0, start_date: 1711929600,
            verifier_did: None, status: ProjectStatus::Proposed,
        },
        ClimateProjectView {
            id: "PRJ-003".into(), name: "Durban Kelp Restoration".into(),
            project_type: ProjectType::OceanRestoration,
            location: LocationView { country_code: "ZA".into(), region: Some("KwaZulu-Natal".into()), latitude: -29.8, longitude: 31.0 },
            expected_credits: 3000.0, start_date: 1696118400,
            verifier_did: Some("did:mycelix:verifier-002".into()), status: ProjectStatus::Completed,
        },
        ClimateProjectView {
            id: "PRJ-004".into(), name: "Joburg Landfill Methane Capture".into(),
            project_type: ProjectType::MethaneCapture,
            location: LocationView { country_code: "ZA".into(), region: Some("Gauteng".into()), latitude: -26.2, longitude: 28.0 },
            expected_credits: 2000.0, start_date: 1719792000,
            verifier_did: Some("did:mycelix:verifier-001".into()), status: ProjectStatus::Verified,
        },
    ]
}

fn mock_credits() -> Vec<CarbonCreditView> {
    vec![
        CarbonCreditView { id: "CC-001".into(), project_id: "PRJ-001".into(), vintage_year: 2024, tonnes_co2e: 500.0, status: CreditStatus::Active, owner_did: "did:mycelix:user-001".into(), retired_at: None },
        CarbonCreditView { id: "CC-002".into(), project_id: "PRJ-001".into(), vintage_year: 2024, tonnes_co2e: 350.0, status: CreditStatus::Retired, owner_did: "did:mycelix:user-001".into(), retired_at: Some(1714521600) },
        CarbonCreditView { id: "CC-003".into(), project_id: "PRJ-003".into(), vintage_year: 2023, tonnes_co2e: 200.0, status: CreditStatus::Active, owner_did: "did:mycelix:user-001".into(), retired_at: None },
    ]
}

fn mock_footprints() -> Vec<CarbonFootprintView> {
    vec![CarbonFootprintView {
        entity_did: "did:mycelix:user-001".into(), period_start: 1704067200, period_end: 1735689600,
        scope1: 12.5, scope2: 8.3, scope3: 45.2, methodology: "GHG Protocol".into(), verified_by: None,
    }]
}
