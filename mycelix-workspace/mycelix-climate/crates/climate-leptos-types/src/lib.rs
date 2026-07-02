// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM-safe types for the Climate cluster frontend.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Carbon
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonFootprintView {
    pub entity_did: String,
    pub period_start: i64,
    pub period_end: i64,
    pub scope1: f64,
    pub scope2: f64,
    pub scope3: f64,
    pub methodology: String,
    pub verified_by: Option<String>,
}

impl CarbonFootprintView {
    pub fn total(&self) -> f64 {
        self.scope1 + self.scope2 + self.scope3
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum CreditStatus {
    Active,
    Transferred,
    Retired,
}

impl CreditStatus {
    pub fn label(&self) -> &'static str {
        match self {
            CreditStatus::Active => "Active",
            CreditStatus::Transferred => "Transferred",
            CreditStatus::Retired => "Retired",
        }
    }
    pub fn css_class(&self) -> &'static str {
        match self {
            CreditStatus::Active => "active",
            CreditStatus::Transferred => "transferred",
            CreditStatus::Retired => "retired",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CarbonCreditView {
    pub id: String,
    pub project_id: String,
    pub vintage_year: u32,
    pub tonnes_co2e: f64,
    pub status: CreditStatus,
    pub owner_did: String,
    pub retired_at: Option<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CreditSummaryView {
    pub total_credits: u64,
    pub total_tonnes: f64,
    pub active_tonnes: f64,
    pub retired_tonnes: f64,
    pub transferred_count: u64,
}

// ---------------------------------------------------------------------------
// Projects
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProjectType {
    Reforestation,
    RenewableEnergy,
    MethaneCapture,
    OceanRestoration,
    DirectAirCapture,
}

impl ProjectType {
    pub fn label(&self) -> &'static str {
        match self {
            ProjectType::Reforestation => "Reforestation",
            ProjectType::RenewableEnergy => "Renewable Energy",
            ProjectType::MethaneCapture => "Methane Capture",
            ProjectType::OceanRestoration => "Ocean Restoration",
            ProjectType::DirectAirCapture => "Direct Air Capture",
        }
    }
    pub fn icon(&self) -> &'static str {
        match self {
            ProjectType::Reforestation => "🌳",
            ProjectType::RenewableEnergy => "☀\u{fe0f}",
            ProjectType::MethaneCapture => "🏭",
            ProjectType::OceanRestoration => "🌊",
            ProjectType::DirectAirCapture => "💨",
        }
    }
    pub fn all() -> &'static [ProjectType] {
        &[
            ProjectType::Reforestation,
            ProjectType::RenewableEnergy,
            ProjectType::MethaneCapture,
            ProjectType::OceanRestoration,
            ProjectType::DirectAirCapture,
        ]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProjectStatus {
    Proposed,
    Verified,
    Active,
    Completed,
}

impl ProjectStatus {
    pub fn label(&self) -> &'static str {
        match self {
            ProjectStatus::Proposed => "Proposed",
            ProjectStatus::Verified => "Verified",
            ProjectStatus::Active => "Active",
            ProjectStatus::Completed => "Completed",
        }
    }
    pub fn css_class(&self) -> &'static str {
        match self {
            ProjectStatus::Proposed => "phase-proposed",
            ProjectStatus::Verified => "phase-verified",
            ProjectStatus::Active => "phase-active",
            ProjectStatus::Completed => "phase-completed",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LocationView {
    pub country_code: String,
    pub region: Option<String>,
    pub latitude: f64,
    pub longitude: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClimateProjectView {
    pub id: String,
    pub name: String,
    pub project_type: ProjectType,
    pub location: LocationView,
    pub expected_credits: f64,
    pub start_date: i64,
    pub verifier_did: Option<String>,
    pub status: ProjectStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectMilestoneView {
    pub project_id: String,
    pub title: String,
    pub description: String,
    pub target_date: i64,
    pub completed_at: Option<i64>,
    pub credits_issued: Option<f64>,
    pub verified_by: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectsSummaryView {
    pub total_projects: u64,
    pub proposed_count: u64,
    pub verified_count: u64,
    pub active_count: u64,
    pub completed_count: u64,
    pub total_expected_credits: f64,
}

// ---------------------------------------------------------------------------
// Marketplace
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MarketplaceListingView {
    pub listing_id: String,
    pub credit_id: String,
    pub project_id: String,
    pub seller_did: String,
    pub price_per_tonne: u64,
    pub currency: String,
    pub min_purchase: f64,
    pub available_tonnes: f64,
    pub expires_at: i64,
    pub is_active: bool,
}
