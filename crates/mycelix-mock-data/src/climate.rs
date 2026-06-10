// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Mock data for Climate cluster — global projects and credits.

use climate_leptos_types::*;

pub fn projects() -> Vec<ClimateProjectView> {
    vec![
        ClimateProjectView {
            id: "PRJ-001".into(), name: "Amazon Basin Reforestation".into(),
            project_type: ProjectType::Reforestation,
            location: LocationView { country_code: "BR".into(), region: Some("Par\u{00E1}".into()), latitude: -3.1, longitude: -52.0 },
            expected_credits: 45000.0, start_date: 1704067200,
            verifier_did: Some("did:mycelix:verra-001".into()), status: ProjectStatus::Active,
        },
        ClimateProjectView {
            id: "PRJ-002".into(), name: "North Sea Offshore Wind".into(),
            project_type: ProjectType::RenewableEnergy,
            location: LocationView { country_code: "DK".into(), region: Some("Jutland".into()), latitude: 55.7, longitude: 7.5 },
            expected_credits: 28000.0, start_date: 1711929600,
            verifier_did: Some("did:mycelix:gold-standard-001".into()), status: ProjectStatus::Active,
        },
        ClimateProjectView {
            id: "PRJ-003".into(), name: "Great Barrier Reef Kelp Restoration".into(),
            project_type: ProjectType::OceanRestoration,
            location: LocationView { country_code: "AU".into(), region: Some("Queensland".into()), latitude: -18.3, longitude: 147.7 },
            expected_credits: 12000.0, start_date: 1696118400,
            verifier_did: Some("did:mycelix:verifier-002".into()), status: ProjectStatus::Completed,
        },
        ClimateProjectView {
            id: "PRJ-004".into(), name: "Sahel Solar Corridor".into(),
            project_type: ProjectType::RenewableEnergy,
            location: LocationView { country_code: "SN".into(), region: Some("Saint-Louis".into()), latitude: 16.0, longitude: -16.5 },
            expected_credits: 35000.0, start_date: 1719792000,
            verifier_did: None, status: ProjectStatus::Proposed,
        },
        ClimateProjectView {
            id: "PRJ-005".into(), name: "Karoo Reforestation".into(),
            project_type: ProjectType::Reforestation,
            location: LocationView { country_code: "ZA".into(), region: Some("Northern Cape".into()), latitude: -31.5, longitude: 19.0 },
            expected_credits: 12000.0, start_date: 1704067200,
            verifier_did: Some("did:mycelix:verifier-001".into()), status: ProjectStatus::Active,
        },
        ClimateProjectView {
            id: "PRJ-006".into(), name: "Iceland Direct Air Capture".into(),
            project_type: ProjectType::DirectAirCapture,
            location: LocationView { country_code: "IS".into(), region: Some("Hellishei\u{00F0}i".into()), latitude: 64.0, longitude: -21.3 },
            expected_credits: 8000.0, start_date: 1711929600,
            verifier_did: Some("did:mycelix:climeworks-001".into()), status: ProjectStatus::Verified,
        },
        ClimateProjectView {
            id: "PRJ-007".into(), name: "India Methane Recovery".into(),
            project_type: ProjectType::MethaneCapture,
            location: LocationView { country_code: "IN".into(), region: Some("Maharashtra".into()), latitude: 19.1, longitude: 72.9 },
            expected_credits: 15000.0, start_date: 1696118400,
            verifier_did: Some("did:mycelix:verifier-003".into()), status: ProjectStatus::Active,
        },
    ]
}

pub fn credits() -> Vec<CarbonCreditView> {
    vec![
        CarbonCreditView { id: "CC-001".into(), project_id: "PRJ-001".into(), vintage_year: 2024, tonnes_co2e: 500.0, status: CreditStatus::Active, owner_did: "did:mycelix:user-001".into(), retired_at: None },
        CarbonCreditView { id: "CC-002".into(), project_id: "PRJ-002".into(), vintage_year: 2024, tonnes_co2e: 350.0, status: CreditStatus::Retired, owner_did: "did:mycelix:user-001".into(), retired_at: Some(1714521600) },
        CarbonCreditView { id: "CC-003".into(), project_id: "PRJ-003".into(), vintage_year: 2023, tonnes_co2e: 200.0, status: CreditStatus::Active, owner_did: "did:mycelix:user-001".into(), retired_at: None },
    ]
}

pub fn footprints() -> Vec<CarbonFootprintView> {
    vec![CarbonFootprintView {
        entity_did: "did:mycelix:user-001".into(), period_start: 1704067200, period_end: 1735689600,
        scope1: 12.5, scope2: 8.3, scope3: 45.2, methodology: "GHG Protocol".into(), verified_by: None,
    }]
}

pub fn credit_summary() -> CreditSummaryView {
    CreditSummaryView {
        total_credits: 12, total_tonnes: 4850.0,
        active_tonnes: 3200.0, retired_tonnes: 1650.0, transferred_count: 3,
    }
}

pub fn projects_summary() -> ProjectsSummaryView {
    ProjectsSummaryView {
        total_projects: 7, proposed_count: 1, verified_count: 1,
        active_count: 4, completed_count: 1, total_expected_credits: 155000.0,
    }
}
