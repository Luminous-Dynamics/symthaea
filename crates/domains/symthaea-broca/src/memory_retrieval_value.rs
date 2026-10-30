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

use crate::memory_evidence::EvidenceIndexSnapshotReceipt;
use crate::memory_lifecycle_policy::{
    ClaimLifecycleLedgerEntry, ClaimLifecycleState, LifecycleAction, LifecycleActionDecision,
    LifecyclePolicyRequest, LifecycleRetrievalProgress, ProofVerificationState,
    VerifiedLifecycleProofToken, decide_lifecycle_action,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// Versioned assumptions defining one micro-utility unit and cost conversion.
pub const RETRIEVAL_UTILITY_POLICY_VERSION: &str = "BROCA-RETRIEVAL-VOI-V1";
pub const RETRIEVAL_UTILITY_CALIBRATION_VERSION: &str = "BROCA-GROUP-CONFORMAL-VOI-V1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalUtilityEstimate {
    pub estimate_id: String,
    pub decision_id: String,
    pub principal_id: String,
    pub scope_id: String,
    pub valid_at_unix_ms: u64,
    pub known_at_unix_ms: u64,
    /// Digest of the exact EvidenceIndex state used by retrieval and the estimator.
    pub evidence_index_state_sha256: String,
    /// Digest of the exact unresolved-claim state used by the estimator.
    pub unresolved_claims_sha256: String,
    pub unresolved_required_claim_ids: BTreeSet<String>,
    /// Digest of the calibration algorithm and group-conformal procedure.
    pub calibration_method_sha256: String,
    /// Familywise one-sided coverage target for the three calibrated bounds.
    pub calibration_joint_coverage_percent: u8,
    /// Identity and digest of the held-out-calibrated estimator.
    pub estimator_id: String,
    pub estimator_version: String,
    pub estimator_sha256: String,
    pub calibration_split_sha256: String,
    pub feature_schema_sha256: String,
    /// Defines the utility unit and all cost conversion coefficients.
    pub utility_schema_sha256: String,
    /// Point prediction before held-out calibration.
    pub correctness_gain_prediction_micro: i64,
    /// Conservative lower confidence bound on marginal correctness gain.
    pub correctness_gain_lcb_micro: i64,
    /// Point prediction before held-out calibration.
    pub lifecycle_risk_reduction_prediction_micro: i64,
    /// Conservative lower bound on avoided lifecycle decision loss
    /// (e.g. stale-only or unresolved-conflict failure); may be negative.
    pub lifecycle_risk_reduction_lcb_micro: i64,
    /// Point prediction before held-out calibration.
    pub retrieval_harm_prediction_micro: i64,
    /// Upper confidence bound on retrieval-induced answer harm/distraction;
    /// signed because retrieval can reduce harm on some workloads.
    pub retrieval_harm_ucb_micro: i64,
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
            ("evidence_index_state_sha256", self.evidence_index_state_sha256.as_str()),
            ("unresolved_claims_sha256", self.unresolved_claims_sha256.as_str()),
            ("calibration_method_sha256", self.calibration_method_sha256.as_str()),
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
        let expected_claims_digest =
            Self::unresolved_claims_digest(&self.unresolved_required_claim_ids)?;
        if expected_claims_digest != self.unresolved_claims_sha256 {
            return Err("unresolved-claim set digest mismatch".into());
        }
        if self.calibration_joint_coverage_percent != 90 {
            return Err("retrieval value calibration must use the pinned 90% joint-coverage target".into());
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
        evidence_index_state_sha256: &str,
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
        if self.estimate.evidence_index_state_sha256 != evidence_index_state_sha256 {
            return Err("retrieval value estimate evidence-index snapshot mismatch".into());
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

/// Canonical adapter from a calibrated VOI result into lifecycle action policy.
///
/// It derives the unresolved claim set from the request ledger, binds the value
/// estimate to the exact decision/principal/scope/time and EvidenceIndex state,
/// replaces retrieval progress with the deterministic VOI result, and then
/// invokes the lifecycle action controller. Production orchestrators should
/// use this instead of copying a can_continue boolean manually.
pub fn decide_lifecycle_action_with_voi(
    mut request: LifecyclePolicyRequest,
    verified_proof: Option<&VerifiedLifecycleProofToken>,
    value_decision: &RetrievalValueDecision,
    evidence_index_snapshot: &EvidenceIndexSnapshotReceipt,
) -> Result<LifecycleActionDecision, String> {
    let unresolved_claim_ids: BTreeSet<String> = request
        .required_claims
        .iter()
        .filter(|claim| claim.state != ClaimLifecycleState::SupportedCurrent)
        .map(|claim| claim.conclusion_id.clone())
        .collect();

    if evidence_index_snapshot.known_at_unix_ms != request.known_at_unix_ms {
        return Err("retrieval value snapshot known_at does not match lifecycle request".into());
    }
    value_decision.verify_request_binding(
        &request.decision_id,
        &request.principal_id,
        &request.scope_id,
        request.valid_at_unix_ms,
        request.known_at_unix_ms,
        &evidence_index_snapshot.index_state_sha256,
        &unresolved_claim_ids,
    )?;
    request.retrieval = value_decision.to_lifecycle_progress()?;
    decide_lifecycle_action(request, verified_proof)
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

    fn estimate_for_claims(
        unresolved: BTreeSet<String>,
    ) -> RetrievalUtilityEstimate {
        let mut value = estimate();
        value.unresolved_claims_sha256 =
            RetrievalUtilityEstimate::unresolved_claims_digest(&unresolved).unwrap();
        value.unresolved_required_claim_ids = unresolved;
        value
    }

    fn lifecycle_request_for_missing_claims(
        unresolved: &BTreeSet<String>,
    ) -> LifecyclePolicyRequest {
        LifecyclePolicyRequest {
            decision_id: "decision-1".into(),
            principal_id: "principal-1".into(),
            scope_id: "scope-1".into(),
            valid_at_unix_ms: 200,
            known_at_unix_ms: 300,
            proof_status: ProofVerificationState::NoEligibleProof,
            proof_packet_sha256: None,
            retrieval: LifecycleRetrievalProgress {
                complete: false,
                can_continue: true,
                remaining_calls: 2,
                value_decision_sha256: None,
            },
            required_claims: unresolved
                .iter()
                .map(|claim_id| ClaimLifecycleLedgerEntry {
                    conclusion_id: claim_id.clone(),
                    state: ClaimLifecycleState::Missing,
                    selected_justification_id: None,
                    current_evidence_ids: BTreeSet::new(),
                    stale_evidence_ids: BTreeSet::new(),
                    conflicting_evidence_ids: BTreeSet::new(),
                    rejected_evidence_ids: BTreeSet::new(),
                })
                .collect(),
        }
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
            evidence_index_state_sha256: digest("evidence index state"),
            unresolved_claims_sha256: RetrievalUtilityEstimate::unresolved_claims_digest(&unresolved).unwrap(),
            unresolved_required_claim_ids: unresolved,
            calibration_method_sha256: digest("BROCA-GROUP-CONFORMAL-VOI-V1"),
            calibration_joint_coverage_percent: 90,
            estimator_id: "utility-estimator".into(),
            estimator_version: "v1".into(),
            estimator_sha256: digest("estimator"),
            calibration_split_sha256: digest("heldout calibration split"),
            feature_schema_sha256: digest("features"),
            utility_schema_sha256: digest("micro-utility-v1"),
            correctness_gain_prediction_micro: 900,
            correctness_gain_lcb_micro: 900,
            lifecycle_risk_reduction_prediction_micro: 300,
            lifecycle_risk_reduction_lcb_micro: 300,
            retrieval_harm_prediction_micro: 100,
            retrieval_harm_ucb_micro: 100,
            retrieval_cost_micro: 400,
            remaining_calls: 2,
            remaining_budget_micro: 1000,
            minimum_net_value_micro: 100,
        }
    }

    #[derive(serde::Deserialize)]
    struct RetrievalValueFixtureFile {
        schema_version: String,
        policy_version: String,
        case_count: usize,
        cases: Vec<RetrievalValueFixtureCase>,
    }

    #[derive(serde::Deserialize)]
    struct RetrievalValueFixtureCase {
        case_id: String,
        correctness_gain_lcb_micro: i64,
        lifecycle_risk_reduction_lcb_micro: i64,
        retrieval_harm_ucb_micro: i64,
        retrieval_cost_micro: u64,
        remaining_calls: u32,
        remaining_budget_micro: u64,
        minimum_net_value_micro: u64,
        unresolved_required_claim_ids: BTreeSet<String>,
        expected_net_value_micro: i128,
        expected_action: RetrievalValueAction,
        expected_reason: RetrievalValueReason,
    }

    #[test]
    fn canonical_voi_adapter_selects_retrieve_more_for_bound_unresolved_claims() {
        let unresolved = BTreeSet::from(["C1".to_string()]);
        let estimate = estimate_for_claims(unresolved.clone());
        let value_decision = evaluate_retrieval_value(estimate).unwrap();
        let request = lifecycle_request_for_missing_claims(&unresolved);

        let action = decide_lifecycle_action_with_voi(
            request,
            None,
            &value_decision,
            &EvidenceIndexSnapshotReceipt {
                known_at_unix_ms: 300,
                index_state_sha256: digest("evidence index state"),
            },
        ).unwrap();
        assert_eq!(action.action, LifecycleAction::RetrieveMore);
        assert_eq!(
            action.retrieval.value_decision_sha256.as_deref(),
            Some(value_decision.decision_sha256.as_str())
        );
        action.verify_internal_consistency().unwrap();
    }

    #[test]
    fn canonical_voi_adapter_stops_and_abstains_when_next_call_is_not_worthwhile() {
        let unresolved = BTreeSet::from(["C1".to_string()]);
        let mut estimate = estimate_for_claims(unresolved.clone());
        estimate.correctness_gain_lcb_micro = 1;
        estimate.lifecycle_risk_reduction_lcb_micro = 0;
        let value_decision = evaluate_retrieval_value(estimate).unwrap();
        let request = lifecycle_request_for_missing_claims(&unresolved);

        let action = decide_lifecycle_action_with_voi(
            request,
            None,
            &value_decision,
            &EvidenceIndexSnapshotReceipt {
                known_at_unix_ms: 300,
                index_state_sha256: digest("evidence index state"),
            },
        ).unwrap();
        assert_eq!(action.action, LifecycleAction::Abstain);
        assert!(!action.retrieval.complete);
        assert!(!action.retrieval.can_continue);
        assert_eq!(action.retrieval.remaining_calls, 2);
        assert!(action.retrieval.value_decision_sha256.is_some());
    }

    #[test]
    fn canonical_voi_adapter_rejects_snapshot_known_at_mismatch() {
        let unresolved = BTreeSet::from(["C1".to_string()]);
        let estimate = estimate_for_claims(unresolved.clone());
        let value_decision = evaluate_retrieval_value(estimate).unwrap();
        let request = lifecycle_request_for_missing_claims(&unresolved);

        let error = decide_lifecycle_action_with_voi(
            request,
            None,
            &value_decision,
            &EvidenceIndexSnapshotReceipt {
                known_at_unix_ms: 301,
                index_state_sha256: digest("evidence index state"),
            },
        ).unwrap_err();
        assert!(error.contains("snapshot known_at"));
    }

    #[test]
    fn canonical_voi_adapter_rejects_changed_evidence_index_snapshot() {
        let unresolved = BTreeSet::from(["C1".to_string()]);
        let estimate = estimate_for_claims(unresolved.clone());
        let value_decision = evaluate_retrieval_value(estimate).unwrap();
        let request = lifecycle_request_for_missing_claims(&unresolved);

        let error = decide_lifecycle_action_with_voi(
            request,
            None,
            &value_decision,
            &EvidenceIndexSnapshotReceipt {
                known_at_unix_ms: 300,
                index_state_sha256: digest("different evidence index"),
            },
        ).unwrap_err();
        assert!(error.contains("evidence-index snapshot mismatch"));
    }

    #[test]
    fn versioned_retrieval_value_fixture_corpus_matches_gate() {
        let raw = include_str!(
            "../../../../research/broca-bitemporal-evaluation-v1/retrieval-value-fixtures-v1.json"
        );
        let fixture: RetrievalValueFixtureFile = serde_json::from_str(raw).unwrap();
        assert_eq!(fixture.schema_version, "broca-retrieval-value-fixtures-v1");
        assert_eq!(fixture.policy_version, RETRIEVAL_UTILITY_POLICY_VERSION);
        assert_eq!(fixture.case_count, fixture.cases.len());
        assert_eq!(fixture.case_count, 10);

        for case in fixture.cases {
            let mut value = estimate();
            value.estimate_id = case.case_id.clone();
            value.correctness_gain_lcb_micro = case.correctness_gain_lcb_micro;
            value.lifecycle_risk_reduction_lcb_micro = case.lifecycle_risk_reduction_lcb_micro;
            value.retrieval_harm_ucb_micro = case.retrieval_harm_ucb_micro;
            value.retrieval_cost_micro = case.retrieval_cost_micro;
            value.remaining_calls = case.remaining_calls;
            value.remaining_budget_micro = case.remaining_budget_micro;
            value.minimum_net_value_micro = case.minimum_net_value_micro;
            value.unresolved_required_claim_ids = case.unresolved_required_claim_ids;
            value.unresolved_claims_sha256 =
                RetrievalUtilityEstimate::unresolved_claims_digest(
                    &value.unresolved_required_claim_ids
                ).unwrap();

            let decision = evaluate_retrieval_value(value).unwrap();
            assert_eq!(
                decision.conservative_net_value_micro,
                case.expected_net_value_micro,
                "net value mismatch for fixture case {}",
                case.case_id
            );
            assert_eq!(
                decision.action,
                case.expected_action,
                "action mismatch for fixture case {}",
                case.case_id
            );
            assert_eq!(
                decision.reason,
                case.expected_reason,
                "reason mismatch for fixture case {}",
                case.case_id
            );
        }
    }

    #[test]
    fn retrieves_when_conservative_net_value_exceeds_margin_and_budgets() {
        let decision = evaluate_retrieval_value(estimate()).unwrap();
        assert_eq!(decision.conservative_net_value_micro, 700);
        assert_eq!(decision.action, RetrievalValueAction::RetrieveMore);
        assert_eq!(decision.reason, RetrievalValueReason::PositiveConservativeNetValue);
        assert!(decision.verify_internal_consistency().is_ok());
        let progress = decision.to_lifecycle_progress().unwrap();
        assert_eq!(progress.complete, false);
        assert_eq!(progress.can_continue, true);
        assert_eq!(progress.remaining_calls, 2);
        assert_eq!(progress.value_decision_sha256.as_deref(), Some(decision.decision_sha256.as_str()));
    }

    #[test]
    fn preserves_negative_conservative_benefit_bounds_without_clamping() {
        let mut value = estimate();
        value.correctness_gain_prediction_micro = 100;
        value.correctness_gain_lcb_micro = -25;
        value.lifecycle_risk_reduction_prediction_micro = 50;
        value.lifecycle_risk_reduction_lcb_micro = -10;
        value.retrieval_harm_prediction_micro = 20;
        value.retrieval_harm_ucb_micro = 30;
        value.retrieval_cost_micro = 0;
        value.minimum_net_value_micro = 0;

        let decision = evaluate_retrieval_value(value).unwrap();
        assert_eq!(decision.conservative_net_value_micro, -65);
        assert_eq!(decision.action, RetrievalValueAction::Stop);
        assert_eq!(decision.reason, RetrievalValueReason::NetValueBelowMinimumMargin);
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
            &digest("evidence index state"),
            &unresolved,
        ).unwrap();
        assert!(decision.verify_request_binding(
            "decision-1",
            "principal-1",
            "scope-1",
            201,
            300,
            &digest("evidence index state"),
            &unresolved,
        ).unwrap_err().contains("identity/time"));
        let different = BTreeSet::from(["C1".to_string()]);
        assert!(decision.verify_request_binding(
            "decision-1",
            "principal-1",
            "scope-1",
            200,
            300,
            &digest("evidence index state"),
            &different,
        ).unwrap_err().contains("claim set mismatch"));
    }

    #[test]
    fn rejects_unresolved_claim_digest_tampering() {
        let mut value = estimate();
        value.unresolved_claims_sha256 = digest("different claim set");
        let error = evaluate_retrieval_value(value).unwrap_err();
        assert!(error.contains("unresolved-claim set digest mismatch"));
    }

    #[test]
    fn refuses_to_reuse_a_value_estimate_for_a_different_evidence_index_snapshot() {
        let decision = evaluate_retrieval_value(estimate()).unwrap();
        let unresolved = BTreeSet::from(["C1".to_string(), "C2".to_string()]);
        let error = decision.verify_request_binding(
            "decision-1",
            "principal-1",
            "scope-1",
            200,
            300,
            &digest("different evidence index"),
            &unresolved,
        ).unwrap_err();
        assert!(error.contains("evidence-index snapshot mismatch"));
    }

    #[test]
    fn receipt_detects_mutation_after_policy_evaluation() {
        let mut decision = evaluate_retrieval_value(estimate()).unwrap();
        decision.action = RetrievalValueAction::Stop;
        assert!(decision.verify_internal_consistency().unwrap_err().contains("deterministic policy"));
    }

    #[test]
    fn does_not_retrieve_when_all_required_claims_are_already_covered() {
        let mut value = estimate();
        value.unresolved_required_claim_ids.clear();
        value.unresolved_claims_sha256 =
            RetrievalUtilityEstimate::unresolved_claims_digest(
                &value.unresolved_required_claim_ids
            ).unwrap();
        let decision = evaluate_retrieval_value(value).unwrap();
        assert_eq!(decision.action, RetrievalValueAction::Stop);
        assert_eq!(decision.reason, RetrievalValueReason::NoUnresolvedRequiredClaims);
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
        assert!(!progress.complete, "the retrieval budget is still present");
        assert!(!progress.can_continue, "the value gate stopped early");
        assert_eq!(progress.remaining_calls, 2);
        assert_eq!(progress.value_decision_sha256.as_deref(), Some(stop.decision_sha256.as_str()));
    }
}
