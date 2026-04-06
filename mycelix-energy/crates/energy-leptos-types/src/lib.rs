// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM-safe types for the Energy cluster frontend.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Projects
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProjectType {
    Solar, Wind, Hydro, Nuclear, Geothermal, BatteryStorage, PumpedHydro, Hydrogen, Biomass,
}
impl ProjectType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Solar => "Solar", Self::Wind => "Wind", Self::Hydro => "Hydro",
            Self::Nuclear => "Nuclear", Self::Geothermal => "Geothermal",
            Self::BatteryStorage => "Battery", Self::PumpedHydro => "Pumped Hydro",
            Self::Hydrogen => "Hydrogen", Self::Biomass => "Biomass",
        }
    }
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Solar => "☀\u{fe0f}", Self::Wind => "🌬\u{fe0f}", Self::Hydro => "💧",
            Self::Nuclear => "⚛\u{fe0f}", Self::Geothermal => "🌋",
            Self::BatteryStorage => "🔋", Self::PumpedHydro => "🏔\u{fe0f}",
            Self::Hydrogen => "💨", Self::Biomass => "🌿",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProjectStatus {
    Proposed, Planning, Permitting, Financing, Construction, Operational, Decommissioned,
}
impl ProjectStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Proposed => "Proposed", Self::Planning => "Planning",
            Self::Permitting => "Permitting", Self::Financing => "Financing",
            Self::Construction => "Construction", Self::Operational => "Operational",
            Self::Decommissioned => "Decommissioned",
        }
    }
    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Proposed => "proposed", Self::Planning | Self::Permitting => "planning",
            Self::Financing => "financing", Self::Construction => "construction",
            Self::Operational => "operational", Self::Decommissioned => "decommissioned",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectLocationView {
    pub latitude: f64,
    pub longitude: f64,
    pub country: String,
    pub region: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectFinancialsView {
    pub total_cost: f64,
    pub funded_amount: f64,
    pub currency: String,
    pub target_irr: f64,
    pub payback_years: f64,
    pub annual_revenue_estimate: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnergyProjectView {
    pub id: String,
    pub name: String,
    pub project_type: ProjectType,
    pub location: ProjectLocationView,
    pub capacity_mw: f64,
    pub status: ProjectStatus,
    pub developer_did: String,
    pub community_did: Option<String>,
    pub financials: ProjectFinancialsView,
    pub phi_score: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectMilestoneView {
    pub project_id: String,
    pub name: String,
    pub description: String,
    pub target_date: i64,
    pub completed_date: Option<i64>,
}

// ---------------------------------------------------------------------------
// Investments
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum InvestmentType { Equity, Debt, ConvertibleNote, RevenueShare, CommunityShare }
impl InvestmentType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Equity => "Equity", Self::Debt => "Debt", Self::ConvertibleNote => "Convertible",
            Self::RevenueShare => "Revenue Share", Self::CommunityShare => "Community Share",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum InvestmentStatus { Pledged, PendingPayment, Confirmed, Cancelled, Transferred }
impl InvestmentStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Pledged => "Pledged", Self::PendingPayment => "Pending",
            Self::Confirmed => "Confirmed", Self::Cancelled => "Cancelled",
            Self::Transferred => "Transferred",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InvestmentView {
    pub id: String,
    pub project_id: String,
    pub investor_did: String,
    pub amount: f64,
    pub currency: String,
    pub shares: f64,
    pub share_percentage: f64,
    pub investment_type: InvestmentType,
    pub status: InvestmentStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DividendView {
    pub project_id: String,
    pub amount: f64,
    pub currency: String,
    pub period_start: i64,
    pub period_end: i64,
}

// ---------------------------------------------------------------------------
// Grid Trading
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum OfferStatus { Active, PartiallyFilled, Filled, Expired, Cancelled }
impl OfferStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Active => "Active", Self::PartiallyFilled => "Partial",
            Self::Filled => "Filled", Self::Expired => "Expired", Self::Cancelled => "Cancelled",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeOfferView {
    pub id: String,
    pub seller_did: String,
    pub project_id: Option<String>,
    pub amount_kwh: f64,
    pub price_per_kwh: f64,
    pub currency: String,
    pub status: OfferStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TradeView {
    pub id: String,
    pub seller_did: String,
    pub buyer_did: String,
    pub amount_kwh: f64,
    pub total_price: f64,
    pub currency: String,
    pub settled: bool,
}

// ---------------------------------------------------------------------------
// Regenerative
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContractStatus { Active, TransitionInProgress, TransitionComplete, Paused, Terminated }
impl ContractStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Active => "Active", Self::TransitionInProgress => "Transitioning",
            Self::TransitionComplete => "Complete", Self::Paused => "Paused",
            Self::Terminated => "Terminated",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegenerativeContractView {
    pub id: String,
    pub project_id: String,
    pub community_did: String,
    pub current_ownership_percentage: f64,
    pub target_ownership_percentage: f64,
    pub reserve_account_balance: f64,
    pub currency: String,
    pub status: ContractStatus,
}
