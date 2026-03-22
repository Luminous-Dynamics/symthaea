// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root//! Energy Bridge — typed interface to mycelix-energy zome functions.
//!
//! Maps Tauri commands to Holochain zome calls for energy project
//! management, investment tracking, and consciousness scoring.

use serde::{Deserialize, Serialize};

/// Summary of an energy project for list views.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectSummary {
    pub id: String,
    pub terra_atlas_id: Option<String>,
    pub name: String,
    pub project_type: String,
    pub capacity_mw: f64,
    pub status: String,
    pub latitude: f64,
    pub longitude: f64,
    pub funded_percentage: f64,
    pub phi_score: Option<f64>,
    pub harmony_alignment: Option<f64>,
}

/// Consciousness score for a project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessScore {
    pub project_id: String,
    pub phi_score: Option<f64>,
    pub harmony_alignment: Option<f64>,
    pub assessed_at: Option<String>,
}

/// Investment summary for portfolio views.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InvestmentSummary {
    pub id: String,
    pub project_id: String,
    pub project_name: String,
    pub amount: f64,
    pub currency: String,
    pub status: String,
    pub pledged_at: String,
    pub confirmed_at: Option<String>,
}

/// Field verification evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldVerification {
    pub project_id: String,
    pub verifier_did: String,
    pub evidence_cid: Option<String>,
    pub gps_latitude: f64,
    pub gps_longitude: f64,
    pub gps_accuracy_m: f32,
    pub verification_type: String,
    pub notes: Option<String>,
}

/// Impact summary for QOL dashboard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactSummary {
    pub project_id: String,
    pub total_co2_avoided: f64,
    pub total_jobs_created: u32,
    pub total_trust_delta: f64,
    pub peak_households: u32,
    pub report_count: u32,
    pub net_humanity_benefit: f64,
}

/// Allocation pledge for the Conscious Allocation Network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PledgeSummary {
    pub id: String,
    pub project_id: String,
    pub pledger_did: String,
    pub amount: u64,
    pub currency: String,
    pub consciousness_tier: String,
    pub harmony_intent: String,
    pub status: String,
    pub expires_at: String,
}

/// Match result from the consciousness-weighted matching engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchSummary {
    pub id: String,
    pub pledge_id: String,
    pub project_id: String,
    pub pledger_did: String,
    pub amount: u64,
    pub match_score: f64,
    pub consciousness_weight: f64,
    pub status: String,
}

/// Full allocation dashboard combining consciousness + pledges + matches + impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationDashboard {
    pub project_id: String,
    pub consciousness: Option<ConsciousnessScore>,
    pub impact: Option<ImpactSummary>,
    pub pending_pledges: u32,
    pub matched_pledges: u32,
    pub total_pledged: u64,
    pub net_humanity_benefit: f64,
    pub discovery_score: f64,
}

/// Demo projects for when the conductor isn't fully wired.
/// These match real USACE/SMR data structure.
pub fn demo_projects() -> Vec<ProjectSummary> {
    vec![
        ProjectSummary {
            id: "demo-001".into(),
            terra_atlas_id: Some("USACE-DAM-1234".into()),
            name: "Hoover Dam Retrofit — Turbine Upgrade".into(),
            project_type: "Hydro".into(),
            capacity_mw: 2080.0,
            status: "Operational".into(),
            latitude: 36.0160,
            longitude: -114.7377,
            funded_percentage: 85.0,
            phi_score: Some(0.72),
            harmony_alignment: Some(0.68),
        },
        ProjectSummary {
            id: "demo-002".into(),
            terra_atlas_id: Some("SMR-NUSCALE-01".into()),
            name: "NuScale VOYGR — Idaho National Lab".into(),
            project_type: "Nuclear".into(),
            capacity_mw: 462.0,
            status: "Construction".into(),
            latitude: 43.5153,
            longitude: -112.9484,
            funded_percentage: 62.0,
            phi_score: Some(0.58),
            harmony_alignment: Some(0.45),
        },
        ProjectSummary {
            id: "demo-003".into(),
            terra_atlas_id: Some("FERC-SOLAR-TX-5500".into()),
            name: "West Texas Community Solar Cooperative".into(),
            project_type: "Solar".into(),
            capacity_mw: 150.0,
            status: "Financing".into(),
            latitude: 31.9686,
            longitude: -99.9018,
            funded_percentage: 35.0,
            phi_score: Some(0.81),
            harmony_alignment: Some(0.89),
        },
    ]
}
