// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Conservative value-of-information gate for adaptive memory retrieval.
//!
//! A remaining retrieval call is not automatically worth making. Retrieval can
//! improve claim coverage, have no effect, distract the reader, introduce stale
//! or conflicting evidence, and consume tokens/latency. This module consumes a
//! calibrated estimate of those marginal effects and decides whether another
//! call is worth its measured cost.
//!
//! The score is expressed in integer micro-utility units under a versioned
//! utility schema. The caller must calibrate estimates on held-out data and
//! supply a conservative lower bound for benefit and an upper bound for harm.
//! This module is the deterministic decision rule; it does not train or
//! authenticate the upstream estimator.
//!
//! The retrieval decision is separate from proof validity. It must never
//! upgrade an unsupported claim or create a proof. When it returns
//! RetrieveMore, the orchestrator performs another retrieval step, rebuilds
//! claim coverage and invokes the proof solver again. When it returns Stop,
//! the lifecycle action controller selects WARN_STALE / EXPLAIN_CONFLICT /
//! ABSTAIN (or ANSWER if a verified proof already exists).

use crate::memory_lifecycle_policy::LifecycleRetrievalProgress;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// Versioned assumptions defining one micro-utility unit and cost conversion.
pub const RETRIEVAL_UTILITY_POLICY_VERSION: &str = "BROCA-RETRIEVAL-VOI-V1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalUtilityEstimate {
    pub estimate_id: String,
    pub decision_id: String,
    pub principal_id: String,
    pub scope_id: String,
    pub valid_at_unix_ms: u64,
    pub known_at_unix_ms: u64,
    /// Digest of the exact unresolved-claim state used by the estimator.
    pub unresolved_claims_sha256: String,
    pub unresolved_required_claim_ids: BTreeSet<String>,
    /// Identity and digest of the held-out-calibrated estimator.
    pub estimator_id: String,
    pub estimator_version: String,
    pub estimator_sha256: String,
    pub calibration_split_sha256: String,
    pub feature_schema_sha256: String,
    /// Defines the utility unit and all cost conversion coefficients.
    pub utility_schema_sha256: String,
    /// Conservative lower confidence bound on marginal correctness gain.
    pub correctness_gain_lcb_micro: u64,
    /// Conservative lower bound on avoided lifecycle decision loss
    /// (e.g. stale-only or unresolved-conflict failure).
    pub lifecycle_risk_reduction_lcb_micro: u64,
    /// Upper confidence bound on retrieval-induced answer harm/distraction.
    pub retrieval_harm_ucb_micro: u64,
    /// Measured/inferred cost after converting tokens, latency and serving cost
    /// into the same utility unit under utility_schema_sha256.
    pub retrieval_cost_micro: u64,
    /// Current remaining call budget before making another retrieval call.
    pub remaining_calls: u32,
    /// Remaining total cost budget in the same micro-utility units.
    pub remaining_budget_micro: u64,
    /// Minimum positive net value required to continue.
    pub minimum_net_value_micro: u64,
}

impl RetrievalUtilityEstimate {
    pub fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("estimate_id", self.estimate_id.as_str()),
            ("decision_id", self.decision_id.as_str()),
            ("principal_id", self.principal_id.as_str()),
            ("scope_id", self.scope_id.as_str()),
            ("estimator_id", self.estimator_id.as_str()),
            ("estimator_version", self.estimator_version.as_str()),
        ] {
            require_non_empty(value, field)?;
        }
        for (field, value) in [
            ("unresolved_claims_sha256", self.unresolved_claims_sha256.as_str()),
            ("estimator_sha256", self.estimator_sha256.as_str()),
            ("calibration_split_sha256", self.calibration_split_sha256.as_str()),
            ("feature_schema_sha256", self.feature_schema_sha256.as_str()),
            ("utility_schema_sha256", self.utility_schema_sha256.as_str()),
        ] {
            validate_sha256(value, field)?;
        }
        for claim_id in &self.unresolved_required_claim_ids {
            require_non_empty(claim_id, "unresolved_required_claim_id")?;
        }
        if self.remaining_calls == 0 && self.remaining_budget_micro > 0 {
            // An empty call budget is valid; it simply forces Stop. The cost
            // budget is retained to explain which constraint was binding.
        }
        Ok(())
    }

    pub fn unresolved_claims_digest(
        claim_ids: &BTreeSet<String>,
    ) -> Result<String, String> {
        for claim_id in claim_ids {
            require_non_empty(claim_id, "unresolved_required_claim_id")?;
        }
        let bytes = serde_json::to_vec(claim_ids)
            .map_err(|error| format!("cannot serialize unresolved-claim set: {error}"))?;
        Ok(sha256_hex(&bytes))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalValueAction {
    RetrieveMore,
    Stop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalValueReason {
    PositiveConservativeNetValue,
    NoUnresolvedRequiredClaims,
    NoRemainingCalls,
    RetrievalCostExceedsRemainingBudget,
    NetValueBelowMinimumMargin,
}

/// Auditable decision over one calibrated marginal retrieval-value estimate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalValueDecision {
    pub policy_version: String,
    pub estimate: RetrievalUtilityEstimate,
    /// Benefit lower-bound minus harm upper-bound and retrieval cost.
    pub conservative_net_value_micro: i128,
    pub action: RetrievalValueAction,
    pub reason: RetrievalValueReason,
    pub decision_sha256: String,
}

#[derive(Debug, Serialize)]
struct RetrievalValueDigestPayload<'a> {
    policy_version: &'a str,
    estimate: &'a RetrievalUtilityEstimate,
    conservative_net_value_micro: i128,
    action: RetrievalValueAction,
    reason: RetrievalValueReason,
}

impl RetrievalValueDecision {
    pub fn verify_internal_consistency(&self) -> Result<(), String> {
        if self.policy_version != RETRIEVAL_UTILITY_POLICY_VERSION {
            return Err("unsupported retrieval value policy version".into());
        }
        self.estimate.validate()?;
        let (expected_net, expected_action, expected_reason) =
            calculate_decision(&self.estimate);
        if self.conservative_net_value_micro != expected_net
            || self.action != expected_action
            || self.reason != expected_reason
        {
            return Err("retrieval value decision does not match deterministic policy".into());
        }
        let expected_digest = retrieval_value_digest(
            &self.policy_version,
            &self.estimate,
            self.conservative_net_value_micro,
            self.action,
            self.reason,
        )?;
        if expected_digest != self.decision_sha256 {
            return Err("retrieval value decision SHA-256 mismatch".into());
        }
        Ok(())
    }

    /// Ensure the estimate is not reused for another identity, scope or time.
    pub fn verify_request_binding(
        &self,
        decision_id: &str,
        principal_id: &str,
        scope_id: &str,
        valid_at_unix_ms: u64,
        known_at_unix_ms: u64,
        unresolved_claims: &BTreeSet<String>,
    ) -> Result<(), String> {
        if self.estimate.decision_id != decision_id
            || self.estimate.principal_id != principal_id
            || self.estimate.scope_id != scope_id
            || self.estimate.valid_at_unix_ms != valid_at_unix_ms
            || self.estimate.known_at_unix_ms != known_at_unix_ms
        {
            return Err("retrieval value estimate identity/time binding mismatch".into());
        }
        if self.estimate.unresolved_required_claim_ids != *unresolved_claims {
            return Err("retrieval value estimate unresolved-claim set mismatch".into());
        }
        let digest = RetrievalUtilityEstimate::unresolved_claims_digest(unresolved_claims)?;
        if self.estimate.unresolved_claims_sha256 != digest {
            return Err("retrieval value estimate unresolved-claim digest mismatch".into());
        }
        Ok(())
    }

    /// Convert the VOI decision into the V1 lifecycle policy's continuation
    /// field. A stopped value estimate always closes continuation; a positive
    /// decision carries the remaining call budget forward.
    pub fn to_lifecycle_progress(&self) -> Result<LifecycleRetrievalProgress, String> {
        self.verify_internal_consistency()?;
        Ok(LifecycleRetrievalProgress {
            complete: self.estimate.remaining_calls == 0,
            can_continue: self.action == RetrievalValueAction::RetrieveMore,
            // Preserve unused calls when the rational stopping threshold is
            // reached. "Stopped early" is not the same as "budget exhausted".
            remaining_calls: self.estimate.remaining_calls,
            value_decision_sha256: Some(self.decision_sha256.clone()),
        })
    }
}

/// Decide whether another retrieval call has positive conservative expected
/// value after accounting for answer harm and the full retrieval cost.
///
/// Net value = correctness-gain lower bound
///           + lifecycle-risk-reduction lower bound
///           - retrieval-harm upper bound
///           - retrieval cost.
///
/// Continuation is allowed only when net value is strictly greater than the
/// configured margin, a call remains, and the expected cost fits the remaining
/// budget. Equality stops: this is a conservative stopping policy.
pub fn evaluate_retrieval_value(
    estimate: RetrievalUtilityEstimate,
) -> Result<RetrievalValueDecision, String> {
    estimate.validate()?;
    let (net, action, reason) = calculate_decision(&estimate);
    let policy_version = RETRIEVAL_UTILITY_POLICY_VERSION.to_string();
    let decision_sha256 = retrieval_value_digest(
        &policy_version,
        &estimate,
        net,
        action,
        reason,
    )?;
    let decision = RetrievalValueDecision {
        policy_version,
        estimate,
        conservative_net_value_micro: net,
        action,
        reason,
        decision_sha256,
    };
    decision.verify_internal_consistency()?;
    Ok(decision)
}

fn calculate_decision(
    estimate: &RetrievalUtilityEstimate,
) -> (i128, RetrievalValueAction, RetrievalValueReason) {
    let benefit = i128::from(estimate.correctness_gain_lcb_micro)
        + i128::from(estimate.lifecycle_risk_reduction_lcb_micro);
    let penalty = i128::from(estimate.retrieval_harm_ucb_micro)
        + i128::from(estimate.retrieval_cost_micro);
    let net = benefit - penalty;

    if estimate.unresolved_required_claim_ids.is_empty() {
        return (
            net,
            RetrievalValueAction::Stop,
            RetrievalValueReason::NoUnresolvedRequiredClaims,
        );
    }
    if estimate.remaining_calls == 0 {
        return (
            net,
            RetrievalValueAction::Stop,
            RetrievalValueReason::NoRemainingCalls,
        );
    }
    if estimate.retrieval_cost_micro > estimate.remaining_budget_micro {
        return (
            net,
            RetrievalValueAction::Stop,
            RetrievalValueReason::RetrievalCostExceedsRemainingBudget,
        );
    }
    if net <= i128::from(estimate.minimum_net_value_micro) {
        return (
            net,
            RetrievalValueAction::Stop,
            RetrievalValueReason::NetValueBelowMinimumMargin,
        );
    }
    (
        net,
        RetrievalValueAction::RetrieveMore,
        RetrievalValueReason::PositiveConservativeNetValue,
    )
}

fn retrieval_value_digest(
    policy_version: &str,
    estimate: &RetrievalUtilityEstimate,
    conservative_net_value_micro: i128,
    action: RetrievalValueAction,
    reason: RetrievalValueReason,
) -> Result<String, String> {
    let payload = RetrievalValueDigestPayload {
        policy_version,
        estimate,
        conservative_net_value_micro,
        action,
        reason,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize retrieval value decision: {error}"))?;
    Ok(sha256_hex(&bytes))
}

fn require_non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
        })
    {
        return Err(format!("{field} must be a 64-character lowercase SHA-256 digest"));
    }
    Ok(())
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    fn estimate() -> RetrievalUtilityEstimate {
        let unresolved = BTreeSet::from(["C1".to_string(), "C2".to_string()]);
        RetrievalUtilityEstimate {
            estimate_id: "estimate-1".into(),
            decision_id: "decision-1".into(),
            principal_id: "principal-1".into(),
            scope_id: "scope-1".into(),
            valid_at_unix_ms: 200,
            known_at_unix_ms: 300,
            unresolved_claims_sha256: RetrievalUtilityEstimate::unresolved_claims_digest(&unresolved).unwrap(),
            unresolved_required_claim_ids: unresolved,
            estimator_id: "utility-estimator".into(),
            estimator_version: "v1".into(),
            estimator_sha256: digest("estimator"),
            calibration_split_sha256: digest("heldout calibration split"),
            feature_schema_sha256: digest("features"),
            utility_schema_sha256: digest("micro-utility-v1"),
            correctness_gain_lcb_micro: 900,
            lifecycle_risk_reduction_lcb_micro: 300,
            retrieval_harm_ucb_micro: 100,
            retrieval_cost_micro: 400,
            remaining_calls: 2,
            remaining_budget_micro: 1000,
            minimum_net_value_micro: 100,
        }
    }

    #[test]
    fn retrieves_when_conservative_net_value_exceeds_margin_and_budgets() {
        let decision = evaluate_retrieval_value(estimate()).unwrap();
        assert_eq!(decision.conservative_net_value_micro, 700);
        assert_eq!(decision.action, RetrievalValueAction::RetrieveMore);
        assert_eq!(decision.reason, RetrievalValueReason::PositiveConservativeNetValue);
        assert!(decision.verify_internal_consistency().is_ok());
        assert_eq!(
            decision.to_lifecycle_progress().unwrap(),
            LifecycleRetrievalProgress {
                complete: false,
                can_continue: true,
                remaining_calls: 2,
            }
        );
    }

    #[test]
    fn stops_when_marginal_utility_is_equal_to_or_below_margin() {
        let mut value = estimate();
        value.correctness_gain_lcb_micro = 300;
        value.lifecycle_risk_reduction_lcb_micro = 200;
        value.retrieval_harm_ucb_micro = 100;
        value.retrieval_cost_micro = 300;
        value.minimum_net_value_micro = 100;
        // net = 100, equality to the required margin is not enough.
        let decision = evaluate_retrieval_value(value).unwrap();
        assert_eq!(decision.conservative_net_value_micro, 100);
        assert_eq!(decision.action, RetrievalValueAction::Stop);
        assert_eq!(decision.reason, RetrievalValueReason::NetValueBelowMinimumMargin);
    }

    #[test]
    fn stops_when_retrieval_cost_exceeds_remaining_budget_even_if_beneficial() {
        let mut value = estimate();
        value.retrieval_cost_micro = 1_200;
        value.remaining_budget_micro = 1_000;
        let decision = evaluate_retrieval_value(value).unwrap();
        assert_eq!(decision.action, RetrievalValueAction::Stop);
        assert_eq!(decision.reason, RetrievalValueReason::RetrievalCostExceedsRemainingBudget);
    }

    #[test]
    fn stops_when_no_call_budget_remains() {
        let mut value = estimate();
        value.remaining_calls = 0;
        let decision = evaluate_retrieval_value(value).unwrap();
        assert_eq!(decision.action, RetrievalValueAction::Stop);
        assert_eq!(decision.reason, RetrievalValueReason::NoRemainingCalls);
    }

    #[test]
    fn stops_when_retrieval_harm_exceeds_expected_benefit() {
        let mut value = estimate();
        value.correctness_gain_lcb_micro = 50;
        value.lifecycle_risk_reduction_lcb_micro = 20;
        value.retrieval_harm_ucb_micro = 500;
        value.retrieval_cost_micro = 100;
        let decision = evaluate_retrieval_value(value).unwrap();
        assert!(decision.conservative_net_value_micro < 0);
        assert_eq!(decision.action, RetrievalValueAction::Stop);
        assert_eq!(decision.reason, RetrievalValueReason::NetValueBelowMinimumMargin);
    }

    #[test]
    fn refuses_to_reuse_estimate_for_different_claims_or_time() {
        let decision = evaluate_retrieval_value(estimate()).unwrap();
        let unresolved = BTreeSet::from(["C1".to_string(), "C2".to_string()]);
        decision.verify_request_binding(
            "decision-1",
            "principal-1",
            "scope-1",
            200,
            300,
            &unresolved,
        ).unwrap();
        assert!(decision.verify_request_binding(
            "decision-1",
            "principal-1",
            "scope-1",
            201,
            300,
            &unresolved,
        ).unwrap_err().contains("identity/time"));
        let different = BTreeSet::from(["C1".to_string()]);
        assert!(decision.verify_request_binding(
            "decision-1",
            "principal-1",
            "scope-1",
            200,
            300,
            &different,
        ).unwrap_err().contains("claim set mismatch"));
    }

    #[test]
    fn receipt_detects_mutation_after_policy_evaluation() {
        let mut decision = evaluate_retrieval_value(estimate()).unwrap();
        decision.action = RetrievalValueAction::Stop;
        assert!(decision.verify_internal_consistency().unwrap_err().contains("deterministic policy"));
    }

    #[test]
    fn positive_voi_receipt_can_populate_lifecycle_retrieval_progress() {
        let decision = evaluate_retrieval_value(estimate()).unwrap();
        assert_eq!(decision.to_lifecycle_progress().unwrap().can_continue, true);

        let mut low_value = estimate();
        low_value.correctness_gain_lcb_micro = 1;
        low_value.lifecycle_risk_reduction_lcb_micro = 0;
        let stop = evaluate_retrieval_value(low_value).unwrap();
        let progress = stop.to_lifecycle_progress().unwrap();
        assert!(progress.complete);
        assert!(!progress.can_continue);
        assert_eq!(progress.remaining_calls, 0);
    }
}
