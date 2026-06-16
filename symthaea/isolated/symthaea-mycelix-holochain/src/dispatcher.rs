// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! High-level dispatch functions for governance and finance zome calls.
//!
//! These dispatchers provide typed, ergonomic wrappers around raw `call_zome`
//! for the most commonly used Mycelix coordinator zome functions.

use crate::conductor::HolochainConductor;
use crate::error::{BridgeError, Result};
use crate::types::{BalanceResponse, FeeTierResponse, TendBalanceResponse};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Governance dispatcher
// ---------------------------------------------------------------------------

/// Dispatches governance-related zome calls (proposals, voting).
///
/// Targets the `governance` role and `agora` coordinator zome in the
/// mycelix-governance cluster DNA.
pub struct GovernanceDispatcher {
    conductor: Arc<HolochainConductor>,
}

impl GovernanceDispatcher {
    /// Create a new governance dispatcher backed by the given conductor.
    pub fn new(conductor: Arc<HolochainConductor>) -> Self {
        Self { conductor }
    }

    /// Submit a new proposal to the governance system.
    ///
    /// Returns the action hash of the created proposal entry.
    pub async fn submit_proposal(
        &self,
        proposal_text: &str,
        proposer_did: &str,
    ) -> Result<String> {
        let payload = serde_json::json!({
            "proposal_text": proposal_text,
            "proposer_did": proposer_did,
        });

        let result = self
            .conductor
            .call_zome("governance", "agora", "create_proposal", payload)
            .await?;

        result
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| {
                BridgeError::DeserializationError(
                    "Expected string action hash from create_proposal".into(),
                )
            })
    }

    /// Cast a vote on an active proposal.
    ///
    /// Returns the action hash of the vote entry.
    pub async fn cast_vote(
        &self,
        proposal_id: &str,
        voter_did: &str,
        approve: bool,
        rationale: &str,
    ) -> Result<String> {
        let payload = serde_json::json!({
            "proposal_id": proposal_id,
            "voter_did": voter_did,
            "approve": approve,
            "rationale": rationale,
        });

        let result = self
            .conductor
            .call_zome("governance", "agora", "cast_vote", payload)
            .await?;

        result
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| {
                BridgeError::DeserializationError(
                    "Expected string action hash from cast_vote".into(),
                )
            })
    }

    /// Get all currently active proposals.
    ///
    /// Returns a vector of proposal JSON objects as returned by the
    /// coordinator zome. The caller is responsible for interpreting
    /// the fields (title, description, vote counts, etc.).
    pub async fn get_active_proposals(&self) -> Result<Vec<serde_json::Value>> {
        let result = self
            .conductor
            .call_zome(
                "governance",
                "agora",
                "get_active_proposals",
                serde_json::json!({}),
            )
            .await?;

        result
            .as_array()
            .cloned()
            .ok_or_else(|| {
                BridgeError::DeserializationError(
                    "Expected array from get_active_proposals".into(),
                )
            })
    }
}

// ---------------------------------------------------------------------------
// Finance dispatcher
// ---------------------------------------------------------------------------

/// Dispatches finance-related zome calls (SAP balance, TEND balance, fee tiers).
///
/// Targets the `finance` role and `finance_bridge` coordinator zome in the
/// mycelix-finance cluster DNA.
pub struct FinanceDispatcher {
    conductor: Arc<HolochainConductor>,
}

impl FinanceDispatcher {
    /// Create a new finance dispatcher backed by the given conductor.
    pub fn new(conductor: Arc<HolochainConductor>) -> Self {
        Self { conductor }
    }

    /// Query the SAP (community currency) balance for a member.
    pub async fn query_sap_balance(&self, member_did: &str) -> Result<BalanceResponse> {
        let payload = serde_json::json!({
            "member_did": member_did,
        });

        let result = self
            .conductor
            .call_zome("finance", "finance_bridge", "query_sap_balance", payload)
            .await?;

        serde_json::from_value(result).map_err(|e| {
            BridgeError::DeserializationError(format!(
                "Failed to parse BalanceResponse: {e}"
            ))
        })
    }

    /// Query the TEND (time-exchange) balance for a member.
    pub async fn query_tend_balance(&self, member_did: &str) -> Result<TendBalanceResponse> {
        let payload = serde_json::json!({
            "member_did": member_did,
        });

        let result = self
            .conductor
            .call_zome("finance", "finance_bridge", "query_tend_balance", payload)
            .await?;

        serde_json::from_value(result).map_err(|e| {
            BridgeError::DeserializationError(format!(
                "Failed to parse TendBalanceResponse: {e}"
            ))
        })
    }

    /// Get the fee tier for a member based on their MYCEL score.
    pub async fn get_fee_tier(&self, member_did: &str) -> Result<FeeTierResponse> {
        let payload = serde_json::json!({
            "member_did": member_did,
        });

        let result = self
            .conductor
            .call_zome(
                "finance",
                "finance_bridge",
                "get_member_fee_tier",
                payload,
            )
            .await?;

        serde_json::from_value(result).map_err(|e| {
            BridgeError::DeserializationError(format!(
                "Failed to parse FeeTierResponse: {e}"
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_governance_dispatcher_creation() {
        let conductor = Arc::new(HolochainConductor::new(
            "ws://localhost:8888",
            "mycelix-unified",
        ));
        let dispatcher = GovernanceDispatcher::new(conductor.clone());
        assert_eq!(dispatcher.conductor.app_id(), "mycelix-unified");
    }

    #[test]
    fn test_finance_dispatcher_creation() {
        let conductor = Arc::new(HolochainConductor::new(
            "ws://localhost:8888",
            "mycelix-unified",
        ));
        let dispatcher = FinanceDispatcher::new(conductor.clone());
        assert_eq!(dispatcher.conductor.app_id(), "mycelix-unified");
    }
}
