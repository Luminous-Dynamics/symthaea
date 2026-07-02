// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM-safe mirror types for the Mycelix Finance cluster.
//!
//! Three currencies: MYCEL (soulbound reputation 0-1), SAP (transferable
//! with 2%/yr demurrage), TEND (mutual credit, 1 hour = 1 TEND).
//!
//! No HDI/HDK dependencies. `AgentPubKey` → `String`, `Timestamp` → `i64`.

use serde::{Deserialize, Serialize};

// ============================================================================
// TEND (Mutual Credit)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum OracleTier {
    Normal,
    Elevated,
    High,
    Emergency,
}

impl OracleTier {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Normal => "Normal",
            Self::Elevated => "Elevated",
            Self::High => "High",
            Self::Emergency => "Emergency",
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Normal => "oracle-normal",
            Self::Elevated => "oracle-elevated",
            Self::High => "oracle-high",
            Self::Emergency => "oracle-emergency",
        }
    }

    pub fn credit_limit(&self) -> i32 {
        match self {
            Self::Normal => 40,
            Self::Elevated => 60,
            Self::High => 80,
            Self::Emergency => 120,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TendBalanceView {
    pub member_did: String,
    pub dao_did: String,
    pub balance: i32,
    pub total_provided: f32,
    pub total_received: f32,
    pub exchange_count: u32,
    pub last_activity: i64,
}

impl TendBalanceView {
    pub fn equilibrium_label(&self) -> &'static str {
        if self.balance == 0 {
            "in equilibrium"
        } else if self.balance > 0 {
            "the community owes you care"
        } else {
            "you owe the community care"
        }
    }

    pub fn balance_class(&self) -> &'static str {
        if self.balance == 0 {
            "tend-equilibrium"
        } else if self.balance > 0 {
            "tend-positive"
        } else {
            "tend-negative"
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExchangeStatus {
    Proposed,
    Confirmed,
    Disputed,
    Cancelled,
    Resolved,
}

impl ExchangeStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Proposed => "Awaiting confirmation",
            Self::Confirmed => "Confirmed",
            Self::Disputed => "In dispute",
            Self::Cancelled => "Cancelled",
            Self::Resolved => "Resolved",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServiceCategory {
    CareWork,
    HomeServices,
    FoodServices,
    Transportation,
    Education,
    GeneralAssistance,
    Administrative,
    Creative,
    TechSupport,
    Wellness,
    Gardening,
    Custom(String),
}

impl ServiceCategory {
    pub fn label(&self) -> &str {
        match self {
            Self::CareWork => "Care Work",
            Self::HomeServices => "Home Services",
            Self::FoodServices => "Food Services",
            Self::Transportation => "Transportation",
            Self::Education => "Education",
            Self::GeneralAssistance => "General Assistance",
            Self::Administrative => "Administrative",
            Self::Creative => "Creative",
            Self::TechSupport => "Tech Support",
            Self::Wellness => "Wellness",
            Self::Gardening => "Gardening",
            Self::Custom(s) => s.as_str(),
        }
    }

    pub fn all_standard() -> &'static [ServiceCategory] {
        &[
            Self::CareWork,
            Self::HomeServices,
            Self::FoodServices,
            Self::Transportation,
            Self::Education,
            Self::GeneralAssistance,
            Self::Administrative,
            Self::Creative,
            Self::TechSupport,
            Self::Wellness,
            Self::Gardening,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TendExchangeView {
    pub hash: String,
    pub id: String,
    pub provider_did: String,
    pub receiver_did: String,
    pub hours: f32,
    pub service_description: String,
    pub service_category: ServiceCategory,
    pub status: ExchangeStatus,
    pub created: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceListingView {
    pub hash: String,
    pub id: String,
    pub provider_did: String,
    pub dao_did: String,
    pub title: String,
    pub description: String,
    pub category: ServiceCategory,
    pub estimated_hours: f32,
    pub active: bool,
    pub created: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ServiceRequestView {
    pub hash: String,
    pub id: String,
    pub requester_did: String,
    pub dao_did: String,
    pub title: String,
    pub description: String,
    pub category: ServiceCategory,
    pub estimated_hours: f32,
    pub open: bool,
    pub created: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OracleStateView {
    pub vitality: u32,
    pub tier: OracleTier,
    pub updated_at: i64,
}

// ============================================================================
// SAP (Transferable Currency with Demurrage)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SapBalanceView {
    pub member_did: String,
    pub balance: u64,
    pub last_demurrage_at: i64,
    pub demurrage_pending: u64,
}

impl SapBalanceView {
    /// Display balance in human-readable SAP units (1 SAP = 1_000_000 micro-SAP).
    pub fn display_balance(&self) -> f64 {
        self.balance as f64 / 1_000_000.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaymentStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
    Refunded,
}

impl PaymentStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Pending => "Pending",
            Self::Processing => "Processing",
            Self::Completed => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
            Self::Refunded => "Refunded",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SapPaymentView {
    pub hash: String,
    pub id: String,
    pub from_did: String,
    pub to_did: String,
    pub amount: u64,
    pub fee: u64,
    pub memo: Option<String>,
    pub status: PaymentStatus,
    pub created: i64,
}

// ============================================================================
// MYCEL (Soulbound Reputation)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MycelTier {
    Newcomer,
    Apprentice,
    Member,
    Steward,
    Elder,
}

impl MycelTier {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Newcomer => "Newcomer",
            Self::Apprentice => "Apprentice",
            Self::Member => "Member",
            Self::Steward => "Steward",
            Self::Elder => "Elder",
        }
    }

    pub fn from_score(score: f64) -> Self {
        if score >= 0.8 {
            Self::Elder
        } else if score >= 0.6 {
            Self::Steward
        } else if score >= 0.4 {
            Self::Member
        } else if score >= 0.2 {
            Self::Apprentice
        } else {
            Self::Newcomer
        }
    }

    pub fn css_class(&self) -> &'static str {
        match self {
            Self::Newcomer => "mycel-newcomer",
            Self::Apprentice => "mycel-apprentice",
            Self::Member => "mycel-member",
            Self::Steward => "mycel-steward",
            Self::Elder => "mycel-elder",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MycelScoreView {
    pub member_did: String,
    pub score: f64,
    pub participation: f64,
    pub recognition: f64,
    pub validation: f64,
    pub longevity: f64,
    pub active_months: u32,
    pub tier: MycelTier,
    pub last_updated: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContributionType {
    Technical,
    Community,
    Care,
    Governance,
    Creative,
    Education,
    General,
}

impl ContributionType {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Technical => "Technical",
            Self::Community => "Community",
            Self::Care => "Care",
            Self::Governance => "Governance",
            Self::Creative => "Creative",
            Self::Education => "Education",
            Self::General => "General",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecognitionEventView {
    pub hash: String,
    pub recognizer_did: String,
    pub recipient_did: String,
    pub contribution_type: ContributionType,
    pub weight: f64,
    pub cycle_id: String,
    pub created: i64,
}

// ============================================================================
// Treasury
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TreasuryView {
    pub hash: String,
    pub id: String,
    pub name: String,
    pub balance: u64,
    pub reserve_ratio: f64,
    pub inalienable_reserve: u64,
    pub available: u64,
    pub currency: String,
    pub created: i64,
}

impl TreasuryView {
    pub fn display_balance(&self) -> f64 {
        self.balance as f64 / 1_000_000.0
    }

    pub fn display_reserve(&self) -> f64 {
        self.inalienable_reserve as f64 / 1_000_000.0
    }

    pub fn display_available(&self) -> f64 {
        self.available as f64 / 1_000_000.0
    }

    pub fn reserve_health(&self) -> f64 {
        if self.balance == 0 {
            return 0.0;
        }
        self.inalienable_reserve as f64 / self.balance as f64
    }
}

// ============================================================================
// Staking
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StakeStatus {
    Active,
    Unbonding,
    Withdrawn,
    Slashed,
}

impl StakeStatus {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Active => "Active",
            Self::Unbonding => "Unbonding",
            Self::Withdrawn => "Withdrawn",
            Self::Slashed => "Slashed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StakeView {
    pub hash: String,
    pub id: String,
    pub staker_did: String,
    pub sap_amount: u64,
    pub mycel_score: f32,
    pub stake_weight: f32,
    pub status: StakeStatus,
    pub created: i64,
    pub unbonding_until: Option<i64>,
}

// ============================================================================
// Finance Summary (aggregated view)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinanceSummaryView {
    pub tend_balance: i32,
    pub sap_balance: u64,
    pub mycel_score: f64,
    pub oracle_tier: OracleTier,
    pub tend_limit: i32,
}
