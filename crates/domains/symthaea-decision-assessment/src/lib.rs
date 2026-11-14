// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, non-authoritative composition of qualified engineering decision
//! receipts. This crate does not authenticate signatures, establish source truth,
//! grant approval, or authorize effects. Owner-specific validators remain the
//! authority for their native receipt semantics; this crate intersects their
//! qualified outputs and checks cross-owner decision invariants.

#![deny(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, FixedOffset};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const ASSESSMENT_SCHEMA_VERSION: &str = "engineering-decision-assessment-v0.1";
pub const CLAIM_CEILING: &str =
    "bounded_engineering_decision_assessment_no_release_or_effect_authority";

const AXES: [&str; 5] = [
    "process_admissibility",
    "evidence_use_validity",
    "engineering_applicability",
    "causal_justification",
    "action_policy",
];

const RECEIPTS: [&str; 6] = [
    "temporal_admissibility",
    "run_context_visibility",
    "evidence_use_validity",
    "engineering_applicability",
    "causal_frontier",
    "action_policy",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateState {
    Pass,
    Fail,
    Indeterminate,
    NotApplicable,
}

impl GateState {
    fn parse(value: &str) -> Result<Self, EvaluationError> {
        match value {
            "pass" => Ok(Self::Pass),
            "fail" => Ok(Self::Fail),
            "indeterminate" => Ok(Self::Indeterminate),
            "not_applicable" => Ok(Self::NotApplicable),
            other => Err(EvaluationError::InvalidEnvelope(format!(
                "unknown gate state: {other}"
            ))),
        }
    }

    fn combine(states: impl IntoIterator<Item = Self>) -> Self {
        let mut saw_indeterminate = false;
        let mut saw_not_applicable = false;
        for state in states {
            match state {
                Self::Fail => return Self::Fail,
                Self::Indeterminate => saw_indeterminate = true,
                Self::NotApplicable => saw_not_applicable = true,
                Self::Pass => {}
            }
        }
        if saw_indeterminate {
            Self::Indeterminate
        } else if saw_not_applicable {
            Self::NotApplicable
        } else {
            Self::Pass
        }
    }

    fn is_indeterminate(self) -> bool {
        matches!(self, Self::Indeterminate | Self::NotApplicable)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionSupport {
    SupportedAtK,
    NotSupportedAtK,
    Indeterminate,
    NotAssessed,
}

impl DecisionSupport {
    fn parse(value: &str) -> Result<Self, EvaluationError> {
        match value {
            "supported_at_K" => Ok(Self::SupportedAtK),
            "not_supported_at_K" => Ok(Self::NotSupportedAtK),
            "indeterminate" => Ok(Self::Indeterminate),
            "not_assessed" => Ok(Self::NotAssessed),
            other => Err(EvaluationError::InvalidEnvelope(format!(
                "unknown decision_support: {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReceiptQualification {
    /// Exact identity supplied by the owner-native receipt adapter.
    pub owner_id: String,
    pub artifact_ref: String,
    pub artifact_sha256: String,
    pub contract_version: String,
    /// These states must come from the appropriate owner validators. This crate
    /// deliberately does not implement signature, trust-root, rights, or
    /// domain-specific evidence validation.
    pub identity: GateState,
    pub digest_integrity: GateState,
    pub authenticity: GateState,
    pub scope: GateState,
    pub native_semantics: GateState,
    /// Required on the action-policy receipt to bind the policy to the exact
    /// evaluator-frozen action/loss inputs and selected action.
    pub action_set_sha256: Option<String>,
    pub loss_model_sha256: Option<String>,
    pub selected_action: Option<String>,
    pub action_set_completeness: Option<GateState>,
}

impl ReceiptQualification {
    fn statuses(&self) -> Vec<GateState> {
        vec![
            self.identity,
            self.digest_integrity,
            self.authenticity,
            self.scope,
            self.native_semantics,
        ]
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AxisAssessment {
    pub state: GateState,
    pub reason_codes: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AssessmentReport {
    /// SHA-256 of the exact input byte stream. This is an identity receipt for
    /// this evaluation run, not proof that the packet was committed earlier.
    pub input_sha256: String,
    pub decision_support: DecisionSupport,
    pub axes: BTreeMap<String, AxisAssessment>,
    pub reason_codes: Vec<String>,
    pub critical_gate_failure_count: usize,
    pub indeterminate_axis_count: usize,
    pub outcome_state: String,
    pub independent_assessment: bool,
    pub non_authoritative: bool,
    pub effect_authority: String,
}

#[derive(Debug, Error)]
pub enum EvaluationError {
    #[error("invalid JSON: {0}")]
    InvalidJson(#[from] serde_json::Error),
    #[error("invalid assessment envelope: {0}")]
    InvalidEnvelope(String),
}

pub fn evaluate_json(
    packet_bytes: &[u8],
    qualifications: &BTreeMap<String, ReceiptQualification>,
) -> Result<AssessmentReport, EvaluationError> {
    let packet: serde_json::Value = serde_json::from_slice(packet_bytes)?;
    let mut reasons = BTreeSet::new();
    let mut hard_failures = BTreeSet::new();
    let mut globally_indeterminate = false;

    let schema_version = string_at(&packet, "/schema_version")?;
    if schema_version != ASSESSMENT_SCHEMA_VERSION {
        hard_failures.insert("SCHEMA_VERSION_UNSUPPORTED".to_owned());
    }

    let claim_ceiling = string_at(&packet, "/claim_ceiling")?;
    if claim_ceiling != CLAIM_CEILING {
        hard_failures.insert("CLAIM_CEILING_EXCEEDED".to_owned());
    }

    let committed = required_object(&packet, "/candidate_commitment")?;
    let episode = required_object(&packet, "/decision_episode")?;
    let evaluator = required_object(&packet, "/evaluator")?;
    let outcome = required_object(&packet, "/outcome")?;

    validate_commitment(
        committed,
        episode,
        evaluator,
        outcome,
        &mut reasons,
        &mut hard_failures,
        &mut globally_indeterminate,
    )?;

    validate_defeaters(
        required_object(&packet, "/defeater_analysis")?,
        &mut reasons,
        &mut hard_failures,
        &mut globally_indeterminate,
    )?;

    validate_counterfactual(
        required_object(&packet, "/counterfactual_adjudication")?,
        &mut reasons,
        &mut hard_failures,
        &mut globally_indeterminate,
    )?;

    validate_outcome(outcome, &mut reasons, &mut hard_failures)?;

    let owner_receipts = required_object(&packet, "/owner_receipts")?;
    let axes_value = required_object(&packet, "/axes")?;
    let mut axes = BTreeMap::new();

    for axis in AXES {
        let declared = required_object(axes_value, &format!("/{axis}"))?;
        let declared_state = GateState::parse(string_at(declared, "/result")?)?;
        let mut states = vec![declared_state];
        let mut axis_reasons = BTreeSet::new();

        for reason in string_array_at(declared, "/reason_codes")? {
            axis_reasons.insert(reason);
        }

        for receipt_key in receipt_dependencies(axis) {
            let receipt = required_object(owner_receipts, &format!("/{receipt_key}"))?;
            let receipt_state = qualify_receipt(
                receipt_key,
                receipt,
                qualifications.get(receipt_key),
                committed,
                episode,
                &mut reasons,
            )?;
            states.push(receipt_state);
            if receipt_state == GateState::Fail {
                axis_reasons.insert(format!("OWNER_RECEIPT_QUALIFICATION_FAILED:{receipt_key}"));
            } else if receipt_state.is_indeterminate() {
                axis_reasons.insert(format!(
                    "OWNER_RECEIPT_QUALIFICATION_INDETERMINATE:{receipt_key}"
                ));
            }
        }

        let state = GateState::combine(states);
        if state == GateState::Fail {
            reasons.insert(format!("AXIS_FAILED:{axis}"));
        } else if state.is_indeterminate() {
            reasons.insert(format!("AXIS_INDETERMINATE:{axis}"));
        }
        axes.insert(
            axis.to_owned(),
            AxisAssessment {
                state,
                reason_codes: axis_reasons.into_iter().collect(),
            },
        );
    }

    let failed_axes = axes
        .values()
        .filter(|axis| axis.state == GateState::Fail)
        .count();
    let indeterminate_axis_count = axes
        .values()
        .filter(|axis| axis.state.is_indeterminate())
        .count();

    if failed_axes > 0 {
        reasons.insert("ONE_OR_MORE_AXES_FAILED".to_owned());
    }
    if indeterminate_axis_count > 0 {
        globally_indeterminate = true;
    }

    let mut critical_gate_failure_count = failed_axes + hard_failures.len();
    for failure in hard_failures {
        reasons.insert(failure);
    }

    let computed_before_declaration = if critical_gate_failure_count > 0 {
        DecisionSupport::NotSupportedAtK
    } else if globally_indeterminate {
        DecisionSupport::Indeterminate
    } else {
        DecisionSupport::SupportedAtK
    };

    let declared = required_object(&packet, "/assessment_result")?;
    let declared_support =
        DecisionSupport::parse(string_at(declared, "/decision_support")?)?;
    let declared_failures = usize_at(declared, "/critical_gate_failure_count")?;
    let declared_indeterminate = usize_at(declared, "/indeterminate_axis_count")?;

    if declared_support != computed_before_declaration {
        reasons.insert("DECLARED_DECISION_SUPPORT_MISMATCH".to_owned());
        critical_gate_failure_count += 1;
    }
    if declared_failures != failed_axes + (critical_gate_failure_count - failed_axes) {
        reasons.insert("DECLARED_CRITICAL_FAILURE_COUNT_MISMATCH".to_owned());
        critical_gate_failure_count += 1;
    }
    if declared_indeterminate != indeterminate_axis_count {
        reasons.insert("DECLARED_INDETERMINATE_AXIS_COUNT_MISMATCH".to_owned());
        critical_gate_failure_count += 1;
    }

    if !bool_at(declared, "/non_authoritative")? {
        reasons.insert("ASSESSMENT_MUST_BE_NON_AUTHORITATIVE".to_owned());
        critical_gate_failure_count += 1;
    }
    if string_at(declared, "/effect_authority")? != "none" {
        reasons.insert("EFFECT_AUTHORITY_MUST_REMAIN_NONE".to_owned());
        critical_gate_failure_count += 1;
    }

    let decision_support = if critical_gate_failure_count > 0 {
        DecisionSupport::NotSupportedAtK
    } else if globally_indeterminate {
        DecisionSupport::Indeterminate
    } else {
        DecisionSupport::SupportedAtK
    };

    let independent_assessment = bool_at(evaluator, "/independent_assessment")?;
    let outcome_state = string_at(outcome, "/state")?.to_owned();

    Ok(AssessmentReport {
        input_sha256: sha256_hex(packet_bytes),
        decision_support,
        axes,
        reason_codes: reasons.into_iter().collect(),
        critical_gate_failure_count,
        indeterminate_axis_count,
        outcome_state,
        independent_assessment,
        non_authoritative: true,
        effect_authority: "none".to_owned(),
    })
}

fn receipt_dependencies(axis: &str) -> &'static [&'static str] {
    match axis {
        "process_admissibility" => &["temporal_admissibility", "run_context_visibility"],
        "evidence_use_validity" => &["evidence_use_validity"],
        "engineering_applicability" => &["engineering_applicability"],
        "causal_justification" => &["causal_frontier"],
        "action_policy" => &["action_policy"],
        _ => &[],
    }
}

fn qualify_receipt(
    key: &str,
    receipt: &serde_json::Value,
    qualification: Option<&ReceiptQualification>,
    commitment: &serde_json::Value,
    episode: &serde_json::Value,
    reasons: &mut BTreeSet<String>,
) -> Result<GateState, EvaluationError> {
    let Some(qualification) = qualification else {
        reasons.insert(format!("OWNER_QUALIFICATION_MISSING:{key}"));
        return Ok(GateState::Indeterminate);
    };

    let expected = [
        ("owner_id", qualification.owner_id.as_str()),
        ("artifact_ref", qualification.artifact_ref.as_str()),
        ("artifact_sha256", qualification.artifact_sha256.as_str()),
        ("contract_version", qualification.contract_version.as_str()),
    ];
    let identity_matches = expected
        .iter()
        .all(|(field, expected)| string_at(receipt, &format!("/{field}")).ok() == Some(*expected));

    if !identity_matches {
        reasons.insert(format!("OWNER_RECEIPT_IDENTITY_MISMATCH:{key}"));
        return Ok(GateState::Fail);
    }

    if !is_sha256(string_at(receipt, "/artifact_sha256")?) {
        reasons.insert(format!("OWNER_RECEIPT_DIGEST_MALFORMED:{key}"));
        return Ok(GateState::Fail);
    }

    let mut states = qualification.statuses();

    if key == "action_policy" {
        let action_set = string_at(commitment, "/action_set_sha256")?;
        let loss_model = string_at(commitment, "/loss_model_sha256")?;
        let selected_action = string_at(episode, "/selected_action")?;

        if qualification.action_set_sha256.as_deref() != Some(action_set)
            || qualification.loss_model_sha256.as_deref() != Some(loss_model)
            || qualification.selected_action.as_deref() != Some(selected_action)
        {
            reasons.insert("ACTION_POLICY_INPUT_BINDING_MISMATCH".to_owned());
            return Ok(GateState::Fail);
        }

        match qualification.action_set_completeness {
            Some(state) => states.push(state),
            None => {
                reasons.insert("ACTION_SET_COMPLETENESS_UNQUALIFIED".to_owned());
                states.push(GateState::Indeterminate);
            }
        }
    }

    let state = GateState::combine(states);
    if state == GateState::Fail {
        reasons.insert(format!("OWNER_QUALIFICATION_FAILED:{key}"));
    } else if state.is_indeterminate() {
        reasons.insert(format!("OWNER_QUALIFICATION_INDETERMINATE:{key}"));
    }
    Ok(state)
}

fn validate_commitment(
    commitment: &serde_json::Value,
    episode: &serde_json::Value,
    evaluator: &serde_json::Value,
    outcome: &serde_json::Value,
    reasons: &mut BTreeSet<String>,
    failures: &mut BTreeSet<String>,
    indeterminate: &mut bool,
) -> Result<(), EvaluationError> {
    for field in [
        "decision_packet_sha256",
        "evidence_manifest_sha256",
        "action_set_sha256",
        "loss_model_sha256",
    ] {
        if !is_sha256(string_at(commitment, &format!("/{field}"))?) {
            failures.insert(format!("COMMITMENT_DIGEST_MALFORMED:{field}"));
        }
    }

    match string_at(commitment, "/state")? {
        "sealed" => {}
        "unsealed" => {
            failures.insert("CANDIDATE_COMMITMENT_UNSEALED".to_owned());
        }
        "not_recorded" => {
            reasons.insert("CANDIDATE_COMMITMENT_NOT_RECORDED".to_owned());
            *indeterminate = true;
        }
        other => {
            return Err(EvaluationError::InvalidEnvelope(format!(
                "unknown commitment state: {other}"
            )));
        }
    }

    match string_at(commitment, "/outcome_disclosure_state")? {
        "not_disclosed_before_commit" => {}
        "disclosed_before_commit" => {
            failures.insert("POST_HOC_OUTCOME_LEAKAGE".to_owned());
        }
        "unknown" => {
            reasons.insert("OUTCOME_DISCLOSURE_BOUNDARY_UNKNOWN".to_owned());
            *indeterminate = true;
        }
        other => {
            return Err(EvaluationError::InvalidEnvelope(format!(
                "unknown outcome disclosure state: {other}"
            )));
        }
    }

    let committed_at = datetime_at(commitment, "/committed_at")?;
    let frontier = datetime_at(episode, "/decision_frontier/instant")?;
    let assessed_at = datetime_at(evaluator, "/assessed_at")?;
    let boundary = string_at(episode, "/decision_frontier/boundary")?;

    let before_frontier = if boundary == "inclusive" {
        committed_at <= frontier
    } else if boundary == "exclusive" {
        committed_at < frontier
    } else {
        false
    };

    if !before_frontier {
        failures.insert("COMMITMENT_AFTER_DECISION_FRONTIER".to_owned());
    }
    if committed_at > assessed_at {
        failures.insert("COMMITMENT_AFTER_ASSESSMENT_TIME".to_owned());
    }

    if let Some(disclosed_at) = commitment.get("outcome_disclosed_at") {
        if let Some(value) = disclosed_at.as_str() {
            let disclosed_at = parse_datetime(value)?;
            if disclosed_at < committed_at {
                failures.insert("OUTCOME_DISCLOSED_BEFORE_COMMIT".to_owned());
            }
        }
    }

    if string_at(outcome, "/state")? == "observed" {
        let observed_at = datetime_at(outcome, "/observed_at")?;
        if observed_at <= committed_at {
            failures.insert("POST_DECISION_OUTCOME_LEAKAGE".to_owned());
        }
    }
    Ok(())
}

fn validate_defeaters(
    analysis: &serde_json::Value,
    reasons: &mut BTreeSet<String>,
    failures: &mut BTreeSet<String>,
    indeterminate: &mut bool,
) -> Result<(), EvaluationError> {
    if !is_sha256(string_at(analysis, "/protocol_sha256")?) {
        failures.insert("DEFEATER_PROTOCOL_DIGEST_MALFORMED".to_owned());
    }

    match string_at(analysis, "/state")? {
        "complete" => {}
        "incomplete" | "not_run" => {
            reasons.insert("DEFEATER_ANALYSIS_INCOMPLETE".to_owned());
            *indeterminate = true;
        }
        other => {
            return Err(EvaluationError::InvalidEnvelope(format!(
                "unknown defeater analysis state: {other}"
            )));
        }
    }

    let findings = analysis
        .get("findings")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| EvaluationError::InvalidEnvelope("findings must be an array".to_owned()))?;

    for finding in findings {
        let severity = string_at(finding, "/severity")?;
        let state = string_at(finding, "/state")?;
        match (severity, state) {
            ("critical", "unresolved") => {
                failures.insert("CRITICAL_DEFEATER_UNRESOLVED".to_owned());
            }
            ("major", "unresolved") => {
                failures.insert("MAJOR_DEFEATER_UNRESOLVED".to_owned());
            }
            ("minor", "unresolved") => {
                reasons.insert("MINOR_DEFEATER_UNRESOLVED".to_owned());
                *indeterminate = true;
            }
            ("critical", "accepted_residual") => {
                failures.insert("CRITICAL_DEFEATER_ACCEPTANCE_NOT_AUTHORIZED".to_owned());
            }
            ("major", "accepted_residual") => {
                reasons.insert("MAJOR_DEFEATER_RESIDUAL_REQUIRES_REVIEW".to_owned());
                *indeterminate = true;
            }
            ("minor", "accepted_residual") | (_, "resolved") => {}
            _ => {
                return Err(EvaluationError::InvalidEnvelope(format!(
                    "unknown defeater severity/state: {severity}/{state}"
                )));
            }
        }
    }
    Ok(())
}

fn validate_counterfactual(
    counterfactual: &serde_json::Value,
    reasons: &mut BTreeSet<String>,
    failures: &mut BTreeSet<String>,
    indeterminate: &mut bool,
) -> Result<(), EvaluationError> {
    let relation = string_at(counterfactual, "/relation")?;
    let result = string_at(counterfactual, "/result")?;

    if relation == "not_applicable" {
        if result != "not_applicable" || counterfactual.get("pair_id").is_some() {
            failures.insert("COUNTERFACTUAL_NOT_APPLICABLE_MISMATCH".to_owned());
        }
        return Ok(());
    }

    if relation != "flip" && relation != "invariant" {
        return Err(EvaluationError::InvalidEnvelope(format!(
            "unknown counterfactual relation: {relation}"
        )));
    }

    if result == "indeterminate" {
        reasons.insert("COUNTERFACTUAL_RESULT_INDETERMINATE".to_owned());
        *indeterminate = true;
        return Ok(());
    }
    if result != "pass" {
        failures.insert("COUNTERFACTUAL_RELATION_FAILED".to_owned());
        return Ok(());
    }

    let expected_original = string_at(counterfactual, "/expected_original_action")?;
    let expected_paired = string_at(counterfactual, "/expected_paired_action")?;
    let observed_original = string_at(counterfactual, "/observed_original_action")?;
    let observed_paired = string_at(counterfactual, "/observed_paired_action")?;

    let relation_holds = if relation == "flip" {
        expected_original != expected_paired
            && observed_original == expected_original
            && observed_paired == expected_paired
    } else {
        expected_original == expected_paired
            && observed_original == expected_original
            && observed_paired == expected_paired
    };

    if !relation_holds {
        failures.insert("COUNTERFACTUAL_ACTION_RELATION_MISMATCH".to_owned());
    }
    Ok(())
}

fn validate_outcome(
    outcome: &serde_json::Value,
    reasons: &mut BTreeSet<String>,
    failures: &mut BTreeSet<String>,
) -> Result<(), EvaluationError> {
    let state = string_at(outcome, "/state")?;
    match state {
        "not_observed" => {
            if outcome.get("outcome_ref").is_some() {
                failures.insert("UNOBSERVED_OUTCOME_HAS_REFERENCE".to_owned());
            }
        }
        "observed" => {
            for field in ["outcome_ref", "outcome_sha256", "observed_at", "horizon"] {
                if outcome.get(field).is_none() {
                    failures.insert(format!("OBSERVED_OUTCOME_MISSING:{field}"));
                }
            }
            if let Some(digest) = outcome.get("outcome_sha256").and_then(|v| v.as_str()) {
                if !is_sha256(digest) {
                    failures.insert("OUTCOME_DIGEST_MALFORMED".to_owned());
                }
            }
        }
        "censored" | "unavailable" | "failed" | "unknown" => {
            if outcome
                .get("missingness_reason")
                .and_then(serde_json::Value::as_str)
                .filter(|v| !v.trim().is_empty())
                .is_none()
            {
                failures.insert("OUTCOME_MISSINGNESS_UNEXPLAINED".to_owned());
            } else {
                reasons.insert(format!("OUTCOME_STATE_PRESERVED:{state}"));
            }
        }
        other => {
            return Err(EvaluationError::InvalidEnvelope(format!(
                "unknown outcome state: {other}"
            )));
        }
    }
    Ok(())
}

fn required_object<'a>(
    value: &'a serde_json::Value,
    pointer: &str,
) -> Result<&'a serde_json::Value, EvaluationError> {
    value
        .pointer(pointer)
        .filter(|v| v.is_object())
        .ok_or_else(|| EvaluationError::InvalidEnvelope(format!("missing object: {pointer}")))
}

fn string_at<'a>(
    value: &'a serde_json::Value,
    pointer: &str,
) -> Result<&'a str, EvaluationError> {
    value
        .pointer(pointer)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| EvaluationError::InvalidEnvelope(format!("missing string: {pointer}")))
}

fn bool_at(value: &serde_json::Value, pointer: &str) -> Result<bool, EvaluationError> {
    value
        .pointer(pointer)
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| EvaluationError::InvalidEnvelope(format!("missing boolean: {pointer}")))
}

fn usize_at(value: &serde_json::Value, pointer: &str) -> Result<usize, EvaluationError> {
    value
        .pointer(pointer)
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| EvaluationError::InvalidEnvelope(format!("missing integer: {pointer}")))
}

fn string_array_at(
    value: &serde_json::Value,
    pointer: &str,
) -> Result<Vec<String>, EvaluationError> {
    let array = value
        .pointer(pointer)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| EvaluationError::InvalidEnvelope(format!("missing array: {pointer}")))?;
    array
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| EvaluationError::InvalidEnvelope(format!("non-string item: {pointer}")))
        })
        .collect()
}

fn datetime_at(
    value: &serde_json::Value,
    pointer: &str,
) -> Result<DateTime<FixedOffset>, EvaluationError> {
    parse_datetime(string_at(value, pointer)?)
}

fn parse_datetime(value: &str) -> Result<DateTime<FixedOffset>, EvaluationError> {
    DateTime::parse_from_rfc3339(value).map_err(|e| {
        EvaluationError::InvalidEnvelope(format!("invalid RFC3339 timestamp {value}: {e}"))
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn sample_packet() -> Value {
        let receipt = |owner: &str, suffix: char| {
            json!({
                "owner_id": owner,
                "artifact_ref": format!("synthetic://{owner}/receipt-001"),
                "artifact_sha256": suffix.to_string().repeat(64),
                "contract_version": "0.1",
                "native_status": "pass",
                "claim_ceiling": "synthetic_known_answer_only"
            })
        };

        json!({
            "schema_version": ASSESSMENT_SCHEMA_VERSION,
            "assessment_id": "synthetic-eng-decision-001",
            "artifact_visibility": "public_synthetic",
            "decision_episode": {
                "case_id": "EC-DECISION-001",
                "pair_id": "EC-PAIR-001",
                "baseline_sha256": "1".repeat(64),
                "proposed_change_sha256": "2".repeat(64),
                "target_world_time": {
                    "state": "known",
                    "instant": "2026-11-12T12:00:00Z",
                    "observation_ref": "synthetic://clock/T"
                },
                "decision_frontier": {
                    "instant": "2026-11-13T09:00:00Z",
                    "boundary": "inclusive",
                    "frontier_ref": "synthetic://frontier/K-001",
                    "ordering_model": "single_source_authoritative_clock_or_monotonic_sequence",
                    "causal_frontier_state": "complete"
                },
                "selected_action": "revalidate"
            },
            "candidate_commitment": {
                "state": "sealed",
                "decision_packet_sha256": "3".repeat(64),
                "evidence_manifest_sha256": "4".repeat(64),
                "action_set_sha256": "5".repeat(64),
                "loss_model_sha256": "6".repeat(64),
                "committed_at": "2026-11-13T08:59:30Z",
                "outcome_disclosure_state": "not_disclosed_before_commit"
            },
            "owner_receipts": {
                "temporal_admissibility": receipt("temporal-owner", 'a'),
                "run_context_visibility": receipt("run-context-owner", 'b'),
                "evidence_use_validity": receipt("use-validity-owner", 'c'),
                "engineering_applicability": receipt("applicability-owner", 'd'),
                "causal_frontier": receipt("causal-frontier-owner", 'e'),
                "action_policy": receipt("action-policy-owner", 'f')
            },
            "axes": {
                "process_admissibility": {"result":"pass","reason_codes":[],"receipt_ref":"synthetic://process/1"},
                "evidence_use_validity": {"result":"pass","reason_codes":[],"receipt_ref":"synthetic://use/1"},
                "engineering_applicability": {"result":"pass","reason_codes":[],"receipt_ref":"synthetic://applicability/1"},
                "causal_justification": {"result":"pass","reason_codes":[],"receipt_ref":"synthetic://causal/1"},
                "action_policy": {"result":"pass","reason_codes":[],"receipt_ref":"synthetic://policy/1"}
            },
            "defeater_analysis": {
                "state": "complete",
                "protocol_ref": "synthetic://protocol/defeater-search-v0.1",
                "protocol_sha256": "7".repeat(64),
                "findings": [{
                    "defeater_id":"DEF-001",
                    "claim_ref":"synthetic://claim/operating-envelope",
                    "category":"measurement_or_uncertainty",
                    "severity":"minor",
                    "state":"accepted_residual",
                    "evidence_refs":["synthetic://evidence/uncertainty-bound"],
                    "residual_rationale":"Bounded uncertainty does not invalidate the required revalidation disposition."
                }]
            },
            "counterfactual_adjudication": {
                "relation":"not_applicable",
                "result":"not_applicable"
            },
            "outcome": {"state":"not_observed"},
            "evaluator": {
                "evaluator_id":"independent-reference-evaluator-v0.1",
                "build_sha256":"8".repeat(64),
                "assessed_at":"2026-11-13T09:01:00Z",
                "assessment_kind":"synthetic_known_answer",
                "independent_assessment":true
            },
            "assessment_result": {
                "decision_support":"supported_at_K",
                "critical_gate_failure_count":0,
                "indeterminate_axis_count":0,
                "non_authoritative":true,
                "effect_authority":"none"
            },
            "claim_ceiling": CLAIM_CEILING
        })
    }

    fn qualifications(packet: &Value) -> BTreeMap<String, ReceiptQualification> {
        let mut result = BTreeMap::new();
        let receipts = packet.get("owner_receipts").unwrap().as_object().unwrap();
        for (key, receipt) in receipts {
            let owner_id = receipt["owner_id"].as_str().unwrap().to_owned();
            let artifact_ref = receipt["artifact_ref"].as_str().unwrap().to_owned();
            let artifact_sha256 = receipt["artifact_sha256"].as_str().unwrap().to_owned();
            let contract_version = receipt["contract_version"].as_str().unwrap().to_owned();
            let action = key == "action_policy";
            result.insert(key.clone(), ReceiptQualification {
                owner_id,
                artifact_ref,
                artifact_sha256,
                contract_version,
                identity: GateState::Pass,
                digest_integrity: GateState::Pass,
                authenticity: GateState::Pass,
                scope: GateState::Pass,
                native_semantics: GateState::Pass,
                action_set_sha256: action.then(|| packet["candidate_commitment"]["action_set_sha256"].as_str().unwrap().to_owned()),
                loss_model_sha256: action.then(|| packet["candidate_commitment"]["loss_model_sha256"].as_str().unwrap().to_owned()),
                selected_action: action.then(|| packet["decision_episode"]["selected_action"].as_str().unwrap().to_owned()),
                action_set_completeness: action.then_some(GateState::Pass),
            });
        }
        result
    }

    fn run(packet: &Value, qualifications: &BTreeMap<String, ReceiptQualification>) -> AssessmentReport {
        evaluate_json(serde_json::to_string(packet).unwrap().as_bytes(), qualifications).unwrap()
    }

    fn rebase_declared_counts(packet: &mut Value, report: &AssessmentReport) {
        packet["assessment_result"]["decision_support"] =
            json!(match report.decision_support {
                DecisionSupport::SupportedAtK => "supported_at_K",
                DecisionSupport::NotSupportedAtK => "not_supported_at_K",
                DecisionSupport::Indeterminate => "indeterminate",
                DecisionSupport::NotAssessed => "not_assessed"
            });
        packet["assessment_result"]["critical_gate_failure_count"] = json!(report.critical_gate_failure_count);
        packet["assessment_result"]["indeterminate_axis_count"] = json!(report.indeterminate_axis_count);
    }

    #[test]
    fn complete_qualified_packet_is_supported_but_non_authoritative() {
        let packet = sample_packet();
        let report = run(&packet, &qualifications(&packet));
        assert_eq!(report.decision_support, DecisionSupport::SupportedAtK);
        assert_eq!(report.critical_gate_failure_count, 0);
        assert_eq!(report.indeterminate_axis_count, 0);
        assert!(report.non_authoritative);
        assert_eq!(report.effect_authority, "none");
        assert!(report.reason_codes.is_empty());
    }

    #[test]
    fn absent_owner_qualification_is_indeterminate_not_pass() {
        let mut packet = sample_packet();
        let report = run(&packet, &BTreeMap::new());
        assert_eq!(report.decision_support, DecisionSupport::Indeterminate);
        assert_eq!(report.indeterminate_axis_count, 5);
        assert!(report.reason_codes.iter().any(|r| r.starts_with("OWNER_QUALIFICATION_MISSING:")));

        packet["assessment_result"]["decision_support"] = json!("indeterminate");
        packet["assessment_result"]["indeterminate_axis_count"] = json!(5);
        let report = run(&packet, &BTreeMap::new());
        assert_eq!(report.decision_support, DecisionSupport::Indeterminate);
        assert_eq!(report.critical_gate_failure_count, 0);
    }

    #[test]
    fn receipt_identity_or_digest_mismatch_fails_closed() {
        let packet = sample_packet();
        let mut qualified = qualifications(&packet);
        qualified.get_mut("action_policy").unwrap().artifact_sha256 = "0".repeat(64);
        let report = run(&packet, &qualified);
        assert_eq!(report.decision_support, DecisionSupport::NotSupportedAtK);
        assert!(report.reason_codes.contains(&"OWNER_RECEIPT_IDENTITY_MISMATCH:action_policy".to_owned()));
    }

    #[test]
    fn owner_authenticity_unknown_cannot_support_a_decision() {
        let mut packet = sample_packet();
        let mut qualified = qualifications(&packet);
        qualified.get_mut("causal_frontier").unwrap().authenticity = GateState::Indeterminate;
        packet["assessment_result"]["decision_support"] = json!("indeterminate");
        packet["assessment_result"]["indeterminate_axis_count"] = json!(1);
        let report = run(&packet, &qualified);
        assert_eq!(report.decision_support, DecisionSupport::Indeterminate);
    }

    #[test]
    fn critical_unresolved_defeater_blocks_supported_at_k() {
        let packet = sample_packet();
        let mut packet = packet;
        packet["defeater_analysis"]["findings"][0]["severity"] = json!("critical");
        packet["defeater_analysis"]["findings"][0]["state"] = json!("unresolved");
        let report = run(&packet, &qualifications(&packet));
        assert_eq!(report.decision_support, DecisionSupport::NotSupportedAtK);
        assert!(report.reason_codes.contains(&"CRITICAL_DEFEATER_UNRESOLVED".to_owned()));
    }

    #[test]
    fn outcome_disclosed_before_commit_is_post_hoc_leakage() {
        let mut packet = sample_packet();
        packet["candidate_commitment"]["outcome_disclosure_state"] = json!("disclosed_before_commit");
        let report = run(&packet, &qualifications(&packet));
        assert_eq!(report.decision_support, DecisionSupport::NotSupportedAtK);
        assert!(report.reason_codes.contains(&"POST_HOC_OUTCOME_LEAKAGE".to_owned()));
    }

    #[test]
    fn commitment_after_decision_frontier_fails() {
        let mut packet = sample_packet();
        packet["candidate_commitment"]["committed_at"] = json!("2026-11-13T09:00:01Z");
        let report = run(&packet, &qualifications(&packet));
        assert_eq!(report.decision_support, DecisionSupport::NotSupportedAtK);
        assert!(report.reason_codes.contains(&"COMMITMENT_AFTER_DECISION_FRONTIER".to_owned()));
    }

    #[test]
    fn action_policy_receipt_must_bind_complete_frozen_action_and_loss_inputs() {
        let packet = sample_packet();
        let mut qualified = qualifications(&packet);
        qualified.get_mut("action_policy").unwrap().action_set_sha256 = Some("9".repeat(64));
        let report = run(&packet, &qualified);
        assert_eq!(report.decision_support, DecisionSupport::NotSupportedAtK);
        assert!(report.reason_codes.contains(&"ACTION_POLICY_INPUT_BINDING_MISMATCH".to_owned()));
    }

    #[test]
    fn flip_counterfactual_requires_both_expected_actions() {
        let mut packet = sample_packet();
        packet["counterfactual_adjudication"] = json!({
            "relation":"flip",
            "result":"pass",
            "pair_id":"PAIR-1",
            "expected_original_action":"revalidate",
            "expected_paired_action":"block",
            "observed_original_action":"revalidate",
            "observed_paired_action":"block"
        });
        let report = run(&packet, &qualifications(&packet));
        assert_eq!(report.decision_support, DecisionSupport::SupportedAtK);
    }

    #[test]
    fn failed_flip_counterfactual_blocks_support() {
        let mut packet = sample_packet();
        packet["counterfactual_adjudication"] = json!({
            "relation":"flip",
            "result":"pass",
            "pair_id":"PAIR-1",
            "expected_original_action":"revalidate",
            "expected_paired_action":"block",
            "observed_original_action":"revalidate",
            "observed_paired_action":"revalidate"
        });
        let report = run(&packet, &qualifications(&packet));
        assert_eq!(report.decision_support, DecisionSupport::NotSupportedAtK);
        assert!(report.reason_codes.contains(&"COUNTERFACTUAL_ACTION_RELATION_MISMATCH".to_owned()));
    }

    #[test]
    fn invariant_counterfactual_requires_action_to_remain_unchanged() {
        let mut packet = sample_packet();
        packet["counterfactual_adjudication"] = json!({
            "relation":"invariant",
            "result":"pass",
            "pair_id":"PAIR-2",
            "expected_original_action":"revalidate",
            "expected_paired_action":"revalidate",
            "observed_original_action":"revalidate",
            "observed_paired_action":"revalidate"
        });
        let report = run(&packet, &qualifications(&packet));
        assert_eq!(report.decision_support, DecisionSupport::SupportedAtK);
    }

    #[test]
    fn censored_outcome_remains_censored_and_does_not_become_zero() {
        let mut packet = sample_packet();
        packet["outcome"] = json!({
            "state":"censored",
            "missingness_reason":"Follow-up observation was interrupted before the declared horizon."
        });
        let report = run(&packet, &qualifications(&packet));
        assert_eq!(report.decision_support, DecisionSupport::SupportedAtK);
        assert_eq!(report.outcome_state, "censored");
        assert!(report.reason_codes.contains(&"OUTCOME_STATE_PRESERVED:censored".to_owned()));
    }

    #[test]
    fn observed_outcome_before_commit_is_leakage() {
        let mut packet = sample_packet();
        packet["outcome"] = json!({
            "state":"observed",
            "outcome_ref":"synthetic://outcome/1",
            "outcome_sha256":"9".repeat(64),
            "observed_at":"2026-11-13T08:58:00Z",
            "horizon":"one-hour"
        });
        let report = run(&packet, &qualifications(&packet));
        assert_eq!(report.decision_support, DecisionSupport::NotSupportedAtK);
        assert!(report.reason_codes.contains(&"POST_DECISION_OUTCOME_LEAKAGE".to_owned()));
    }

    #[test]
    fn false_declared_support_is_detected() {
        let mut packet = sample_packet();
        let report = run(&packet, &BTreeMap::new());
        rebase_declared_counts(&mut packet, &report);
        packet["assessment_result"]["decision_support"] = json!("supported_at_K");
        let report = run(&packet, &BTreeMap::new());
        assert_eq!(report.decision_support, DecisionSupport::NotSupportedAtK);
        assert!(report.reason_codes.contains(&"DECLARED_DECISION_SUPPORT_MISMATCH".to_owned()));
    }

    #[test]
    fn unsupported_effect_authority_is_rejected() {
        let mut packet = sample_packet();
        packet["assessment_result"]["effect_authority"] = json!("approve_release");
        let report = run(&packet, &qualifications(&packet));
        assert_eq!(report.decision_support, DecisionSupport::NotSupportedAtK);
        assert!(report.reason_codes.contains(&"EFFECT_AUTHORITY_MUST_REMAIN_NONE".to_owned()));
    }

    #[test]
    fn malformed_json_is_rejected() {
        let error = evaluate_json(b"{", &BTreeMap::new()).unwrap_err();
        assert!(matches!(error, EvaluationError::InvalidJson(_)));
    }
}
