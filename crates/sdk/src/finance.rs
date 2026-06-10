// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Finance Module
//!
//! Client-side types and helpers for interacting with the Mycelix Finance cluster.
//!
//! Provides:
//! - Re-exports of canonical finance types from `mycelix_finance_types`
//! - Response types matching the finance bridge coordinator's wire format
//! - `FinanceBridgeClient` for ergonomic cross-cluster queries
//!
//! ## Example
//!
//! ```rust
//! use mycelix_sdk::finance::{FinanceBridgeClient, Currency, FeeTier};
//!
//! let client = FinanceBridgeClient::new("did:mycelix:alice");
//! let tier = FeeTier::from_mycel(0.75);
//! assert_eq!(tier, FeeTier::Steward);
//! assert!(Currency::Sap.is_transferable());
//! ```

use serde::{Deserialize, Serialize};

// =============================================================================
// Re-exports from mycelix_finance_types (canonical shared types)
// =============================================================================

pub use mycelix_finance_types::{
    ContributionType, Currency, CurrencyAlias, CurrencyStatus, FeeTier, MetabolicState,
    MintedCurrencyParams, SapMintSource, SuccessionPreference, TendLimitTier,
};

// =============================================================================
// Response types (mirror the finance bridge coordinator wire format)
// =============================================================================

/// SAP balance response from the finance bridge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BalanceResponse {
    /// Member's decentralized identifier
    pub member_did: String,
    /// Currency queried (always "SAP" for this response)
    pub currency: String,
    /// Current SAP balance in micro-SAP
    pub balance: u64,
    /// Whether the balance data was successfully retrieved
    pub available: bool,
}

/// TEND balance response from the finance bridge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TendBalanceResponse {
    /// Member's decentralized identifier
    pub member_did: String,
    /// Current TEND balance (signed: positive = credits, negative = debits)
    pub balance: i32,
    /// Member's MYCEL reputation score
    pub mycel_score: f64,
    /// Whether the balance data was successfully retrieved
    pub available: bool,
}

/// Fee tier response from the finance bridge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeeTierResponse {
    /// Member's decentralized identifier
    pub member_did: String,
    /// Member's MYCEL reputation score
    pub mycel_score: f64,
    /// Human-readable tier name (e.g., "Newcomer", "Member", "Steward")
    pub tier_name: String,
    /// Base fee rate as a fraction (e.g., 0.001 = 0.1%)
    pub base_fee_rate: f64,
}

/// TEND limit response from the finance bridge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TendLimitResponse {
    /// Member's decentralized identifier
    pub member_did: String,
    /// Current network vitality score
    pub vitality: u32,
    /// Human-readable tier name (e.g., "Normal", "Emergency")
    pub tier_name: String,
    /// Effective TEND balance limit (positive/negative)
    pub effective_limit: i32,
}

/// Unified financial summary across all currencies.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FinanceSummaryResponse {
    /// Member's decentralized identifier
    pub member_did: String,
    /// SAP balance in micro-SAP
    pub sap_balance: u64,
    /// TEND balance (signed mutual credit)
    pub tend_balance: i32,
    /// MYCEL reputation score (0.0 - 1.0)
    pub mycel_score: f64,
    /// Fee tier name
    pub fee_tier: String,
    /// Current fee rate
    pub fee_rate: f64,
    /// Effective TEND limit
    pub tend_limit: i32,
    /// TEND limit tier name
    pub tend_tier: String,
}

/// Health check response from the finance bridge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FinanceBridgeHealth {
    /// Whether the finance bridge is healthy
    pub healthy: bool,
    /// Agent DID of the responding node
    pub agent: String,
    /// List of available zome names
    pub zomes: Vec<String>,
}

// =============================================================================
// FinanceBridgeClient
// =============================================================================

/// Client for cross-cluster finance bridge calls.
///
/// This is a lightweight struct that builds the serialized request payloads
/// for dispatching via `CallTargetCell::OtherRole("finance")` in Holochain,
/// or for use in SDK tests and simulations.
///
/// In a Holochain zome context, methods return the expected response type
/// that callers should deserialize from the `ZomeCallResponse`. Outside
/// Holochain, this client can be used with a mock transport or for
/// type-safe request construction.
///
/// ## Example
///
/// ```rust
/// use mycelix_sdk::finance::FinanceBridgeClient;
///
/// let client = FinanceBridgeClient::new("did:mycelix:alice");
/// assert_eq!(client.member_did(), "did:mycelix:alice");
///
/// // Build a mock response for testing
/// let health = client.mock_health_check();
/// assert!(health.healthy);
/// ```
#[derive(Debug, Clone)]
pub struct FinanceBridgeClient {
    member_did: String,
}

impl FinanceBridgeClient {
    /// Create a new finance bridge client for the given member DID.
    pub fn new(member_did: impl Into<String>) -> Self {
        Self {
            member_did: member_did.into(),
        }
    }

    /// Get the member DID this client is configured for.
    pub fn member_did(&self) -> &str {
        &self.member_did
    }

    /// Target zome name for cross-cluster calls.
    pub fn zome_name() -> &'static str {
        "finance_bridge"
    }

    /// Target role name for `CallTargetCell::OtherRole`.
    pub fn role_name() -> &'static str {
        "finance"
    }

    // -------------------------------------------------------------------------
    // Request builders: return (function_name, serialized_payload) tuples
    // -------------------------------------------------------------------------

    /// Build the request for querying SAP balance.
    ///
    /// Zome function: `query_sap_balance`
    /// Expected response: [`BalanceResponse`]
    pub fn query_sap_balance_request(&self) -> (&'static str, String) {
        ("query_sap_balance", self.member_did.clone())
    }

    /// Build the request for querying TEND balance.
    ///
    /// Zome function: `query_tend_balance`
    /// Expected response: [`TendBalanceResponse`]
    pub fn query_tend_balance_request(&self) -> (&'static str, String) {
        ("query_tend_balance", self.member_did.clone())
    }

    /// Build the request for getting the member's fee tier.
    ///
    /// Zome function: `get_member_fee_tier`
    /// Expected response: [`FeeTierResponse`]
    pub fn get_fee_tier_request(&self) -> (&'static str, String) {
        ("get_member_fee_tier", self.member_did.clone())
    }

    /// Build the request for getting the member's TEND limit.
    ///
    /// Zome function: `get_member_tend_limit`
    /// Expected response: [`TendLimitResponse`]
    pub fn get_tend_limit_request(&self) -> (&'static str, String) {
        ("get_member_tend_limit", self.member_did.clone())
    }

    /// Build the request for getting a unified finance summary.
    ///
    /// Zome function: `get_finance_summary`
    /// Expected response: [`FinanceSummaryResponse`]
    pub fn get_finance_summary_request(&self) -> (&'static str, String) {
        ("get_finance_summary", self.member_did.clone())
    }

    /// Build the request for the health check.
    ///
    /// Zome function: `health_check`
    /// Expected response: [`FinanceBridgeHealth`]
    pub fn health_check_request() -> (&'static str, ()) {
        ("health_check", ())
    }

    // -------------------------------------------------------------------------
    // Mock builders: useful for testing without a live conductor
    // -------------------------------------------------------------------------

    /// Build a mock SAP balance response.
    pub fn mock_sap_balance(&self, balance: u64) -> BalanceResponse {
        BalanceResponse {
            member_did: self.member_did.clone(),
            currency: "SAP".into(),
            balance,
            available: true,
        }
    }

    /// Build a mock TEND balance response.
    pub fn mock_tend_balance(&self, balance: i32, mycel_score: f64) -> TendBalanceResponse {
        TendBalanceResponse {
            member_did: self.member_did.clone(),
            balance,
            mycel_score,
            available: true,
        }
    }

    /// Build a mock fee tier response from a MYCEL score.
    pub fn mock_fee_tier(&self, mycel_score: f64) -> FeeTierResponse {
        let tier = FeeTier::from_mycel(mycel_score);
        FeeTierResponse {
            member_did: self.member_did.clone(),
            mycel_score,
            tier_name: format!("{:?}", tier),
            base_fee_rate: tier.base_fee_rate(),
        }
    }

    /// Build a mock TEND limit response from a vitality score.
    pub fn mock_tend_limit(&self, vitality: u32) -> TendLimitResponse {
        let tier = TendLimitTier::from_vitality(vitality);
        TendLimitResponse {
            member_did: self.member_did.clone(),
            vitality,
            tier_name: format!("{:?}", tier),
            effective_limit: tier.limit(),
        }
    }

    /// Build a mock finance summary.
    pub fn mock_finance_summary(
        &self,
        sap_balance: u64,
        tend_balance: i32,
        mycel_score: f64,
        vitality: u32,
    ) -> FinanceSummaryResponse {
        let fee_tier = FeeTier::from_mycel(mycel_score);
        let tend_tier = TendLimitTier::from_vitality(vitality);
        FinanceSummaryResponse {
            member_did: self.member_did.clone(),
            sap_balance,
            tend_balance,
            mycel_score,
            fee_tier: format!("{:?}", fee_tier),
            fee_rate: fee_tier.base_fee_rate(),
            tend_limit: tend_tier.limit(),
            tend_tier: format!("{:?}", tend_tier),
        }
    }

    /// Build a mock health check response.
    pub fn mock_health_check(&self) -> FinanceBridgeHealth {
        FinanceBridgeHealth {
            healthy: true,
            agent: self.member_did.clone(),
            zomes: vec![
                "payments".into(),
                "treasury".into(),
                "tend".into(),
                "staking".into(),
                "recognition".into(),
                "currency_mint".into(),
            ],
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = FinanceBridgeClient::new("did:mycelix:alice");
        assert_eq!(client.member_did(), "did:mycelix:alice");
        assert_eq!(FinanceBridgeClient::zome_name(), "finance_bridge");
        assert_eq!(FinanceBridgeClient::role_name(), "finance");
    }

    #[test]
    fn test_request_builders() {
        let client = FinanceBridgeClient::new("did:mycelix:bob");

        let (fn_name, payload) = client.query_sap_balance_request();
        assert_eq!(fn_name, "query_sap_balance");
        assert_eq!(payload, "did:mycelix:bob");

        let (fn_name, payload) = client.query_tend_balance_request();
        assert_eq!(fn_name, "query_tend_balance");
        assert_eq!(payload, "did:mycelix:bob");

        let (fn_name, payload) = client.get_fee_tier_request();
        assert_eq!(fn_name, "get_member_fee_tier");
        assert_eq!(payload, "did:mycelix:bob");

        let (fn_name, payload) = client.get_tend_limit_request();
        assert_eq!(fn_name, "get_member_tend_limit");
        assert_eq!(payload, "did:mycelix:bob");

        let (fn_name, payload) = client.get_finance_summary_request();
        assert_eq!(fn_name, "get_finance_summary");
        assert_eq!(payload, "did:mycelix:bob");

        let (fn_name, _) = FinanceBridgeClient::health_check_request();
        assert_eq!(fn_name, "health_check");
    }

    #[test]
    fn test_mock_sap_balance() {
        let client = FinanceBridgeClient::new("did:mycelix:alice");
        let resp = client.mock_sap_balance(5_000_000_000);
        assert_eq!(resp.member_did, "did:mycelix:alice");
        assert_eq!(resp.currency, "SAP");
        assert_eq!(resp.balance, 5_000_000_000);
        assert!(resp.available);
    }

    #[test]
    fn test_mock_tend_balance() {
        let client = FinanceBridgeClient::new("did:mycelix:alice");
        let resp = client.mock_tend_balance(-15, 0.5);
        assert_eq!(resp.balance, -15);
        assert_eq!(resp.mycel_score, 0.5);
        assert!(resp.available);
    }

    #[test]
    fn test_mock_fee_tier_newcomer() {
        let client = FinanceBridgeClient::new("did:mycelix:alice");
        let resp = client.mock_fee_tier(0.1);
        assert_eq!(resp.tier_name, "Newcomer");
        assert!((resp.base_fee_rate - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mock_fee_tier_steward() {
        let client = FinanceBridgeClient::new("did:mycelix:alice");
        let resp = client.mock_fee_tier(0.8);
        assert_eq!(resp.tier_name, "Steward");
        assert!((resp.base_fee_rate - 0.0001).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mock_tend_limit_emergency() {
        let client = FinanceBridgeClient::new("did:mycelix:alice");
        let resp = client.mock_tend_limit(5);
        assert_eq!(resp.tier_name, "Emergency");
        assert_eq!(resp.effective_limit, 120);
    }

    #[test]
    fn test_mock_tend_limit_normal() {
        let client = FinanceBridgeClient::new("did:mycelix:alice");
        let resp = client.mock_tend_limit(50);
        assert_eq!(resp.tier_name, "Normal");
        assert_eq!(resp.effective_limit, 40);
    }

    #[test]
    fn test_mock_finance_summary() {
        let client = FinanceBridgeClient::new("did:mycelix:alice");
        let resp = client.mock_finance_summary(1_000_000, -5, 0.5, 30);
        assert_eq!(resp.sap_balance, 1_000_000);
        assert_eq!(resp.tend_balance, -5);
        assert_eq!(resp.mycel_score, 0.5);
        assert_eq!(resp.fee_tier, "Member");
        assert_eq!(resp.tend_tier, "Elevated");
        assert_eq!(resp.tend_limit, 60);
    }

    #[test]
    fn test_mock_health_check() {
        let client = FinanceBridgeClient::new("did:mycelix:alice");
        let resp = client.mock_health_check();
        assert!(resp.healthy);
        assert_eq!(resp.zomes.len(), 6);
        assert!(resp.zomes.contains(&"payments".to_string()));
        assert!(resp.zomes.contains(&"currency_mint".to_string()));
    }

    #[test]
    fn test_response_serde_roundtrip() {
        let balance = BalanceResponse {
            member_did: "did:mycelix:test".into(),
            currency: "SAP".into(),
            balance: 42,
            available: true,
        };
        let json = serde_json::to_string(&balance).unwrap();
        let parsed: BalanceResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, balance);

        let summary = FinanceSummaryResponse {
            member_did: "did:mycelix:test".into(),
            sap_balance: 1000,
            tend_balance: -10,
            mycel_score: 0.5,
            fee_tier: "Member".into(),
            fee_rate: 0.0003,
            tend_limit: 40,
            tend_tier: "Normal".into(),
        };
        let json = serde_json::to_string(&summary).unwrap();
        let parsed: FinanceSummaryResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, summary);
    }

    #[test]
    fn test_reexported_types() {
        // Verify canonical types are accessible through this module
        assert_eq!(Currency::Sap.display_name(), "SAP");
        assert!(!Currency::Mycel.is_transferable());
        assert_eq!(FeeTier::from_mycel(0.5), FeeTier::Member);
        assert_eq!(TendLimitTier::from_vitality(5), TendLimitTier::Emergency);
        assert_eq!(
            MetabolicState::from_vitality(90.0),
            MetabolicState::Thriving
        );
    }
}
