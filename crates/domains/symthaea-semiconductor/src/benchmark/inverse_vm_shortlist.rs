// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-qualified inverse virtual-metrology candidate screening.
//!
//! This module screens caller-proposed recipe candidates. It does not generate
//! recipe values, estimate causal effects, authorize experiments, or control
//! equipment. Diversity is applied only after hard feasibility/support gates.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const INVERSE_VM_SHORTLIST_INPUT_SCHEMA_V1: &str =
    "luminous.semi-bench.inverse-vm-shortlist-input.v1";
pub const INVERSE_VM_SHORTLIST_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.inverse-vm-shortlist-policy.v1";
pub const INVERSE_VM_SHORTLIST_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.inverse-vm-shortlist-report.v1";
pub const FORWARD_EVALUATION_RECEIPT_SCHEMA_V1: &str =
    "luminous.semi-bench.inverse-vm-forward-evaluation-receipt.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForwardEvidenceStateV1 {
    Qualified,
    Unknown,
    Rejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateScreenStateV1 {
    Feasible,
    Rejected,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShortlistEvaluationStateV1 {
    Complete,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateScreenReasonV1 {
    ContextSnapshotMismatch,
    MissingRecipeKnob,
    DuplicateRecipeKnob,
    UnknownRecipeKnob,
    RecipeValueNonFinite,
    RecipeValueOutsideHardBound,
    ConstraintSourceUnavailable,
    HistoricalSupportReceiptMissing,
    HistoricalSupportCountInsufficient,
    HistoricalSupportDigestMismatch,
    HistoricalSupportDistanceNonFinite,
    HistoricalSupportOutsideEnvelope,
    DuplicateSupportNeighbour,
    InvalidSupportNeighbourId,
    ForwardModelDigestMismatch,
    ForwardEvaluationReceiptMissing,
    ForwardEvaluationReceiptSchemaMismatch,
    ForwardEvaluationReceiptDigestMismatch,
    ForwardEvaluationReceiptContextMismatch,
    ForwardEvaluationReceiptRecipeMismatch,
    ForwardEvaluationReceiptPredictionMismatch,
    ForwardEvaluationReceiptStatusMismatch,
    ForwardEvaluationReceiptNotYetAvailable,
    ForwardEvaluationReceiptClockDomainMismatch,
    ForwardEvidenceUnknown,
    ForwardEvidenceRejected,
    MissingRequiredPrediction,
    DuplicatePredictedMeasurand,
    PredictionUnitMismatch,
    PredictionIntervalInvalid,
    TargetIntervalNotContained,
    SideConstraintIntervalNotContained,
    CandidateEvidenceDigestMismatch,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NumericRangeV1 {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecipeKnobConstraintV1 {
    pub knob_id: String,
    pub unit: String,
    pub allowed: NumericRangeV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecipeSettingV1 {
    pub knob_id: String,
    pub unit: String,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetrologyTargetV1 {
    pub measurand_id: String,
    pub unit: String,
    pub target_value: f64,
    pub acceptable: NumericRangeV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SideConstraintV1 {
    pub constraint_id: String,
    pub measurand_id: String,
    pub unit: String,
    pub allowed: NumericRangeV1,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictedMetricIntervalV1 {
    pub measurand_id: String,
    pub unit: String,
    pub lower: f64,
    pub upper: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistoricalSupportEvidenceV1 {
    pub support_manifest_sha256: String,
    pub neighbour_ids: Vec<String>,
    /// Normalized distance in the declared historical-support feature space.
    pub normalized_distance: f64,
    pub support_policy_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ForwardEvaluationReceiptV1 {
    pub schema: String,
    pub evaluator_id: String,
    pub clock_domain_id: String,
    pub clock_mapping_sha256: String,
    pub process_snapshot_sha256: String,
    pub forward_model_sha256: String,
    pub model_qualification_report_sha256: Option<String>,
    pub recipe_settings_sha256: String,
    pub predicted_metrics_sha256: String,
    pub evidence_state: ForwardEvidenceStateV1,
    pub evaluated_at_unix_seconds: i64,
    pub available_at_unix_seconds: i64,
    pub receipt_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmCandidateV1 {
    pub candidate_id: String,
    pub process_snapshot_sha256: String,
    pub recipe_settings: Vec<RecipeSettingV1>,
    pub historical_support: Option<HistoricalSupportEvidenceV1>,
    pub forward_model_sha256: String,
    pub forward_evidence_state: ForwardEvidenceStateV1,
    pub forward_evaluation_receipt: Option<ForwardEvaluationReceiptV1>,
    pub predicted_metrics: Vec<PredictedMetricIntervalV1>,
    pub candidate_evidence_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmShortlistPolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub recipe_constraint_source_sha256: String,
    pub recipe_constraints_complete: bool,
    pub recipe_knob_constraints: Vec<RecipeKnobConstraintV1>,
    pub side_constraints_complete: bool,
    pub side_constraints: Vec<SideConstraintV1>,
    pub target: MetrologyTargetV1,
    pub required_forward_model_sha256: String,
    pub support_manifest_sha256: String,
    pub support_policy_sha256: String,
    pub minimum_support_neighbours: usize,
    pub maximum_normalized_support_distance: f64,
    pub maximum_interval_width_fraction: f64,
    /// Lower is better. This scales target error, interval width and support distance.
    pub target_error_weight: f64,
    pub interval_width_weight: f64,
    pub support_distance_weight: f64,
    /// Diversity bonus applied only among already-feasible candidates.
    pub diversity_weight: f64,
    pub minimum_recipe_diversity_distance: f64,
    pub top_k: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmShortlistInputV1 {
    pub schema: String,
    pub input_id: String,
    pub process_snapshot_sha256: String,
    /// Decision knowledge frontier; forward receipts available later cannot be used.
    pub decision_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub clock_mapping_sha256: String,
    pub policy: InverseVmShortlistPolicyV1,
    pub candidates: Vec<InverseVmCandidateV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmCandidateEvaluationV1 {
    pub candidate_id: String,
    pub state: CandidateScreenStateV1,
    pub reasons: Vec<CandidateScreenReasonV1>,
    pub base_score: Option<f64>,
    pub selected_score: Option<f64>,
    pub selected_rank: Option<usize>,
    pub candidate_evidence_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmShortlistReportV1 {
    pub schema: String,
    pub input_id: String,
    pub process_snapshot_sha256: String,
    pub decision_at_unix_seconds: i64,
    pub policy_sha256: String,
    pub constraint_source_sha256: String,
    pub state: ShortlistEvaluationStateV1,
    pub evaluations: Vec<InverseVmCandidateEvaluationV1>,
    pub top_k_candidate_ids: Vec<String>,
    pub feasible_candidate_count: usize,
    pub rejected_candidate_count: usize,
    pub unknown_candidate_count: usize,
    pub candidate_generation_executed: bool,
    pub recipe_write_executed: bool,
    pub physical_experiment_authorized: bool,
    pub equipment_control_authority: bool,
    /// Forward receipts are content-bound but not issuer-authenticated in this version.
    pub forward_evaluation_receipt_authenticity_proven: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct InverseVmShortlistErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), InverseVmShortlistErrorV1> {
    if value.trim().is_empty() {
        return Err(InverseVmShortlistErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, InverseVmShortlistErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        InverseVmShortlistErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn range_is_valid(range: &NumericRangeV1) -> bool {
    range.lower.is_finite() && range.upper.is_finite() && range.lower <= range.upper
}

fn range_contains(outer: &NumericRangeV1, inner_lower: f64, inner_upper: f64) -> bool {
    inner_lower >= outer.lower && inner_upper <= outer.upper
}

fn recipe_settings_digest(
    settings: &[RecipeSettingV1],
) -> Result<String, InverseVmShortlistErrorV1> {
    let mut canonical = settings.to_vec();
    canonical.sort_by(|left, right| left.knob_id.cmp(&right.knob_id));
    json_sha256(&canonical)
}

fn predicted_metrics_digest(
    metrics: &[PredictedMetricIntervalV1],
) -> Result<String, InverseVmShortlistErrorV1> {
    let mut canonical = metrics.to_vec();
    canonical.sort_by(|left, right| left.measurand_id.cmp(&right.measurand_id));
    json_sha256(&canonical)
}

fn forward_receipt_digest(
    receipt: &ForwardEvaluationReceiptV1,
) -> Result<String, InverseVmShortlistErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

/// Seals a forward-evaluation receipt after the evaluator has frozen the
/// process snapshot, model, recipe vector and prediction interval outputs.
/// A content digest provides replayable identity, not issuer authentication.
pub fn seal_forward_evaluation_receipt_v1(
    receipt: &ForwardEvaluationReceiptV1,
) -> Result<String, InverseVmShortlistErrorV1> {
    forward_receipt_digest(receipt)
}

fn candidate_digest(candidate: &InverseVmCandidateV1) -> Result<String, InverseVmShortlistErrorV1> {
    let mut canonical = candidate.clone();
    canonical.candidate_evidence_sha256.clear();
    json_sha256(&canonical)
}

fn policy_digest(policy: &InverseVmShortlistPolicyV1) -> Result<String, InverseVmShortlistErrorV1> {
    let mut canonical = policy.clone();
    canonical
        .recipe_knob_constraints
        .sort_by(|a, b| a.knob_id.cmp(&b.knob_id));
    canonical
        .side_constraints
        .sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
    json_sha256(&canonical)
}

fn report_digest(report: &InverseVmShortlistReportV1) -> Result<String, InverseVmShortlistErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

/// Calculates the canonical candidate payload digest. This is a content
/// identity, not proof that the named source system authored the candidate.
pub fn seal_inverse_vm_candidate_v1(
    candidate: &InverseVmCandidateV1,
) -> Result<String, InverseVmShortlistErrorV1> {
    candidate_digest(candidate)
}

/// Calculates a policy digest after canonical sorting of knob/constraint rows.
pub fn seal_inverse_vm_shortlist_policy_v1(
    policy: &InverseVmShortlistPolicyV1,
) -> Result<String, InverseVmShortlistErrorV1> {
    policy_digest(policy)
}

fn validate_policy(policy: &InverseVmShortlistPolicyV1) -> Result<(), InverseVmShortlistErrorV1> {
    if policy.schema != INVERSE_VM_SHORTLIST_POLICY_SCHEMA_V1 {
        return Err(InverseVmShortlistErrorV1(
            "shortlist policy schema is unsupported".into(),
        ));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if policy.revision == 0
        || policy.recipe_knob_constraints.is_empty()
        || policy.minimum_support_neighbours == 0
        || !valid_sha256(&policy.recipe_constraint_source_sha256)
        || !valid_sha256(&policy.required_forward_model_sha256)
        || !valid_sha256(&policy.support_manifest_sha256)
        || !valid_sha256(&policy.support_policy_sha256)
    {
        return Err(InverseVmShortlistErrorV1(
            "policy identity or required evidence digest is malformed".into(),
        ));
    }
    if !range_is_valid(&policy.target.acceptable)
        || !policy.target.target_value.is_finite()
        || policy.target.target_value < policy.target.acceptable.lower
        || policy.target.target_value > policy.target.acceptable.upper
    {
        return Err(InverseVmShortlistErrorV1(
            "target value or acceptable range is invalid".into(),
        ));
    }
    require_text(&policy.target.measurand_id, "policy.target.measurand_id")?;
    require_text(&policy.target.unit, "policy.target.unit")?;
    if !policy.maximum_normalized_support_distance.is_finite()
        || policy.maximum_normalized_support_distance <= 0.0
        || !policy.maximum_interval_width_fraction.is_finite()
        || policy.maximum_interval_width_fraction <= 0.0
        || !policy.target_error_weight.is_finite()
        || policy.target_error_weight < 0.0
        || !policy.interval_width_weight.is_finite()
        || policy.interval_width_weight < 0.0
        || !policy.support_distance_weight.is_finite()
        || policy.support_distance_weight < 0.0
        || !policy.diversity_weight.is_finite()
        || policy.diversity_weight < 0.0
        || !policy.minimum_recipe_diversity_distance.is_finite()
        || policy.minimum_recipe_diversity_distance < 0.0
        || policy.top_k == 0
    {
        return Err(InverseVmShortlistErrorV1(
            "policy numeric threshold is invalid".into(),
        ));
    }

    let mut knob_ids = BTreeSet::new();
    for knob in &policy.recipe_knob_constraints {
        require_text(&knob.knob_id, "recipe_constraint.knob_id")?;
        require_text(&knob.unit, "recipe_constraint.unit")?;
        if !range_is_valid(&knob.allowed) || !knob_ids.insert(knob.knob_id.as_str()) {
            return Err(InverseVmShortlistErrorV1(
                "recipe constraint has invalid range or duplicate knob ID".into(),
            ));
        }
    }
    let mut constraint_ids = BTreeSet::new();
    for constraint in &policy.side_constraints {
        require_text(&constraint.constraint_id, "side_constraint.constraint_id")?;
        require_text(&constraint.measurand_id, "side_constraint.measurand_id")?;
        require_text(&constraint.unit, "side_constraint.unit")?;
        if !range_is_valid(&constraint.allowed)
            || !constraint_ids.insert(constraint.constraint_id.as_str())
        {
            return Err(InverseVmShortlistErrorV1(
                "side constraint has invalid range or duplicate constraint ID".into(),
            ));
        }
    }
    Ok(())
}

fn normalized_recipe_distance(
    left: &InverseVmCandidateV1,
    right: &InverseVmCandidateV1,
    constraints: &BTreeMap<String, RecipeKnobConstraintV1>,
) -> f64 {
    let right_settings: BTreeMap<&str, &RecipeSettingV1> = right
        .recipe_settings
        .iter()
        .map(|setting| (setting.knob_id.as_str(), setting))
        .collect();
    let mut sum = 0.0;
    let mut count = 0_usize;
    for a in &left.recipe_settings {
        if let (Some(b), Some(bound)) = (
            right_settings.get(a.knob_id.as_str()),
            constraints.get(&a.knob_id),
        ) {
            let span = (bound.allowed.upper - bound.allowed.lower).max(f64::EPSILON);
            let delta = (a.value - b.value) / span;
            sum += delta * delta;
            count += 1;
        }
    }
    if count == 0 {
        f64::INFINITY
    } else {
        (sum / count as f64).sqrt()
    }
}

fn validate_forward_evaluation_receipt(
    input: &InverseVmShortlistInputV1,
    candidate: &InverseVmCandidateV1,
) -> Result<Vec<CandidateScreenReasonV1>, InverseVmShortlistErrorV1> {
    let mut reasons = Vec::new();
    let Some(receipt) = candidate.forward_evaluation_receipt.as_ref() else {
        reasons.push(CandidateScreenReasonV1::ForwardEvaluationReceiptMissing);
        return Ok(reasons);
    };

    if receipt.schema != FORWARD_EVALUATION_RECEIPT_SCHEMA_V1 {
        reasons.push(CandidateScreenReasonV1::ForwardEvaluationReceiptSchemaMismatch);
    }
    require_text(&receipt.evaluator_id, "forward_receipt.evaluator_id")?;
    require_text(&receipt.clock_domain_id, "forward_receipt.clock_domain_id")?;
    if !valid_sha256(&receipt.clock_mapping_sha256)
        || receipt.clock_mapping_sha256 != input.clock_mapping_sha256
        || receipt.clock_domain_id != input.clock_domain_id
    {
        reasons.push(CandidateScreenReasonV1::ForwardEvaluationReceiptClockDomainMismatch);
    }
    if !valid_sha256(&receipt.receipt_sha256)
        || forward_receipt_digest(receipt)? != receipt.receipt_sha256
    {
        reasons.push(CandidateScreenReasonV1::ForwardEvaluationReceiptDigestMismatch);
    }

    let recipe_digest = recipe_settings_digest(&candidate.recipe_settings)?;
    let metrics_digest = predicted_metrics_digest(&candidate.predicted_metrics)?;
    if receipt.process_snapshot_sha256 != input.process_snapshot_sha256
        || receipt.process_snapshot_sha256 != candidate.process_snapshot_sha256
        || receipt.forward_model_sha256 != candidate.forward_model_sha256
    {
        reasons.push(CandidateScreenReasonV1::ForwardEvaluationReceiptContextMismatch);
    }
    if receipt.recipe_settings_sha256 != recipe_digest {
        reasons.push(CandidateScreenReasonV1::ForwardEvaluationReceiptRecipeMismatch);
    }
    if receipt.predicted_metrics_sha256 != metrics_digest {
        reasons.push(CandidateScreenReasonV1::ForwardEvaluationReceiptPredictionMismatch);
    }
    if receipt.evidence_state != candidate.forward_evidence_state {
        reasons.push(CandidateScreenReasonV1::ForwardEvaluationReceiptStatusMismatch);
    }
    if receipt.evaluated_at_unix_seconds < 0
        || receipt.available_at_unix_seconds < receipt.evaluated_at_unix_seconds
    {
        reasons.push(CandidateScreenReasonV1::ForwardEvaluationReceiptContextMismatch);
    }
    if receipt.available_at_unix_seconds > input.decision_at_unix_seconds {
        reasons.push(CandidateScreenReasonV1::ForwardEvaluationReceiptNotYetAvailable);
    }
    match receipt.evidence_state {
        ForwardEvidenceStateV1::Qualified => {
            if receipt
                .model_qualification_report_sha256
                .as_deref()
                .is_none_or(|digest| !valid_sha256(digest))
            {
                reasons.push(CandidateScreenReasonV1::ForwardEvaluationReceiptDigestMismatch);
            }
        }
        ForwardEvidenceStateV1::Unknown | ForwardEvidenceStateV1::Rejected => {}
    }
    reasons.sort();
    reasons.dedup();
    Ok(reasons)
}

fn screen_candidate(
    input: &InverseVmShortlistInputV1,
    candidate: &InverseVmCandidateV1,
    knob_constraints: &BTreeMap<String, RecipeKnobConstraintV1>,
    side_constraints: &BTreeMap<String, SideConstraintV1>,
) -> Result<InverseVmCandidateEvaluationV1, InverseVmShortlistErrorV1> {
    let policy = &input.policy;
    let mut reasons = Vec::new();
    let mut state = CandidateScreenStateV1::Feasible;

    if candidate.process_snapshot_sha256 != input.process_snapshot_sha256 {
        reasons.push(CandidateScreenReasonV1::ContextSnapshotMismatch);
        state = CandidateScreenStateV1::Rejected;
    }
    if !candidate
        .candidate_evidence_sha256
        .eq(&candidate_digest(candidate)?)
    {
        reasons.push(CandidateScreenReasonV1::CandidateEvidenceDigestMismatch);
        state = CandidateScreenStateV1::Rejected;
    }

    if !policy.recipe_constraints_complete || !policy.side_constraints_complete {
        reasons.push(CandidateScreenReasonV1::ConstraintSourceUnavailable);
        if state != CandidateScreenStateV1::Rejected {
            state = CandidateScreenStateV1::Unknown;
        }
    }

    let mut seen_knobs = BTreeSet::new();
    for setting in &candidate.recipe_settings {
        require_text(&setting.knob_id, "candidate.recipe_setting.knob_id")?;
        if !setting.value.is_finite() {
            reasons.push(CandidateScreenReasonV1::RecipeValueNonFinite);
            state = CandidateScreenStateV1::Rejected;
            continue;
        }
        if !seen_knobs.insert(setting.knob_id.as_str()) {
            reasons.push(CandidateScreenReasonV1::DuplicateRecipeKnob);
            state = CandidateScreenStateV1::Rejected;
            continue;
        }
        let Some(constraint) = knob_constraints.get(&setting.knob_id) else {
            reasons.push(CandidateScreenReasonV1::UnknownRecipeKnob);
            state = CandidateScreenStateV1::Rejected;
            continue;
        };
        if setting.unit != constraint.unit {
            reasons.push(CandidateScreenReasonV1::UnknownRecipeKnob);
            state = CandidateScreenStateV1::Rejected;
            continue;
        }
        if setting.value < constraint.allowed.lower || setting.value > constraint.allowed.upper {
            reasons.push(CandidateScreenReasonV1::RecipeValueOutsideHardBound);
            state = CandidateScreenStateV1::Rejected;
        }
    }
    for knob_id in knob_constraints.keys() {
        if !seen_knobs.contains(knob_id.as_str()) {
            reasons.push(CandidateScreenReasonV1::MissingRecipeKnob);
            state = CandidateScreenStateV1::Rejected;
        }
    }

    match candidate.historical_support.as_ref() {
        None => {
            reasons.push(CandidateScreenReasonV1::HistoricalSupportReceiptMissing);
            if state != CandidateScreenStateV1::Rejected {
                state = CandidateScreenStateV1::Unknown;
            }
        }
        Some(support) => {
            if support.support_manifest_sha256 != policy.support_manifest_sha256
                || support.support_policy_sha256 != policy.support_policy_sha256
                || !valid_sha256(&support.support_manifest_sha256)
                || !valid_sha256(&support.support_policy_sha256)
            {
                reasons.push(CandidateScreenReasonV1::HistoricalSupportDigestMismatch);
                state = CandidateScreenStateV1::Rejected;
            }
            if support.neighbour_ids.iter().collect::<BTreeSet<_>>().len()
                != support.neighbour_ids.len()
            {
                reasons.push(CandidateScreenReasonV1::DuplicateSupportNeighbour);
                state = CandidateScreenStateV1::Rejected;
            }
            if support
                .neighbour_ids
                .iter()
                .any(|neighbour_id| neighbour_id.trim().is_empty())
            {
                reasons.push(CandidateScreenReasonV1::InvalidSupportNeighbourId);
                state = CandidateScreenStateV1::Rejected;
            }
            if support.neighbour_ids.len() < policy.minimum_support_neighbours {
                reasons.push(CandidateScreenReasonV1::HistoricalSupportCountInsufficient);
                if state != CandidateScreenStateV1::Rejected {
                    state = CandidateScreenStateV1::Unknown;
                }
            }
            if !support.normalized_distance.is_finite() || support.normalized_distance < 0.0 {
                reasons.push(CandidateScreenReasonV1::HistoricalSupportDistanceNonFinite);
                state = CandidateScreenStateV1::Rejected;
            } else if support.normalized_distance > policy.maximum_normalized_support_distance {
                reasons.push(CandidateScreenReasonV1::HistoricalSupportOutsideEnvelope);
                state = CandidateScreenStateV1::Rejected;
            }
        }
    }

    if candidate.forward_model_sha256 != policy.required_forward_model_sha256
        || !valid_sha256(&candidate.forward_model_sha256)
    {
        reasons.push(CandidateScreenReasonV1::ForwardModelDigestMismatch);
        state = CandidateScreenStateV1::Rejected;
    }
    let forward_receipt_reasons = validate_forward_evaluation_receipt(input, candidate)?;
    for reason in forward_receipt_reasons {
        let receipt_is_late =
            reason == CandidateScreenReasonV1::ForwardEvaluationReceiptNotYetAvailable;
        reasons.push(reason);
        if state != CandidateScreenStateV1::Rejected {
            state = if receipt_is_late
                || reason == CandidateScreenReasonV1::ForwardEvaluationReceiptMissing
            {
                CandidateScreenStateV1::Unknown
            } else {
                CandidateScreenStateV1::Rejected
            };
        }
    }
    match candidate.forward_evidence_state {
        ForwardEvidenceStateV1::Qualified => {}
        ForwardEvidenceStateV1::Unknown => {
            reasons.push(CandidateScreenReasonV1::ForwardEvidenceUnknown);
            if state != CandidateScreenStateV1::Rejected {
                state = CandidateScreenStateV1::Unknown;
            }
        }
        ForwardEvidenceStateV1::Rejected => {
            reasons.push(CandidateScreenReasonV1::ForwardEvidenceRejected);
            state = CandidateScreenStateV1::Rejected;
        }
    }

    let mut predictions = BTreeMap::new();
    for metric in &candidate.predicted_metrics {
        require_text(
            &metric.measurand_id,
            "candidate.predicted_metric.measurand_id",
        )?;
        if predictions
            .insert(metric.measurand_id.as_str(), metric)
            .is_some()
        {
            reasons.push(CandidateScreenReasonV1::DuplicatePredictedMeasurand);
            state = CandidateScreenStateV1::Rejected;
        }
        if !metric.lower.is_finite() || !metric.upper.is_finite() || metric.lower > metric.upper {
            reasons.push(CandidateScreenReasonV1::PredictionIntervalInvalid);
            state = CandidateScreenStateV1::Rejected;
        }
    }

    let mut target_error = f64::INFINITY;
    let mut normalized_width = f64::INFINITY;
    if let Some(prediction) = predictions.get(policy.target.measurand_id.as_str()) {
        if prediction.unit != policy.target.unit {
            reasons.push(CandidateScreenReasonV1::PredictionUnitMismatch);
            state = CandidateScreenStateV1::Rejected;
        } else if !range_contains(
            &policy.target.acceptable,
            prediction.lower,
            prediction.upper,
        ) {
            reasons.push(CandidateScreenReasonV1::TargetIntervalNotContained);
            state = CandidateScreenStateV1::Rejected;
        } else {
            let midpoint = (prediction.lower + prediction.upper) / 2.0;
            let target_span =
                (policy.target.acceptable.upper - policy.target.acceptable.lower).max(f64::EPSILON);
            target_error = (midpoint - policy.target.target_value).abs() / target_span;
            normalized_width = (prediction.upper - prediction.lower) / target_span;
            if normalized_width > policy.maximum_interval_width_fraction {
                reasons.push(CandidateScreenReasonV1::PredictionIntervalInvalid);
                state = CandidateScreenStateV1::Rejected;
            }
        }
    } else {
        reasons.push(CandidateScreenReasonV1::MissingRequiredPrediction);
        if state != CandidateScreenStateV1::Rejected {
            state = CandidateScreenStateV1::Unknown;
        }
    }

    for constraint in side_constraints.values() {
        match predictions.get(constraint.measurand_id.as_str()) {
            None => {
                reasons.push(CandidateScreenReasonV1::MissingRequiredPrediction);
                if state != CandidateScreenStateV1::Rejected {
                    state = CandidateScreenStateV1::Unknown;
                }
            }
            Some(prediction) if prediction.unit != constraint.unit => {
                reasons.push(CandidateScreenReasonV1::PredictionUnitMismatch);
                state = CandidateScreenStateV1::Rejected;
            }
            Some(prediction)
                if !range_contains(&constraint.allowed, prediction.lower, prediction.upper) =>
            {
                reasons.push(CandidateScreenReasonV1::SideConstraintIntervalNotContained);
                state = CandidateScreenStateV1::Rejected;
            }
            Some(_) => {}
        }
    }

    reasons.sort();
    reasons.dedup();
    let score = if state == CandidateScreenStateV1::Feasible {
        let support_distance = candidate
            .historical_support
            .as_ref()
            .map(|support| support.normalized_distance)
            .unwrap_or(f64::INFINITY);
        Some(
            policy.target_error_weight * target_error
                + policy.interval_width_weight * normalized_width
                + policy.support_distance_weight * support_distance
                    / policy.maximum_normalized_support_distance,
        )
    } else {
        None
    };
    Ok(InverseVmCandidateEvaluationV1 {
        candidate_id: candidate.candidate_id.clone(),
        state,
        reasons,
        base_score: score,
        selected_score: None,
        selected_rank: None,
        candidate_evidence_sha256: candidate.candidate_evidence_sha256.clone(),
    })
}

fn build_shortlist(
    candidates: &[InverseVmCandidateV1],
    evaluations: &mut [InverseVmCandidateEvaluationV1],
    policy: &InverseVmShortlistPolicyV1,
) -> Vec<String> {
    let candidate_by_id: BTreeMap<&str, &InverseVmCandidateV1> = candidates
        .iter()
        .map(|candidate| (candidate.candidate_id.as_str(), candidate))
        .collect();
    let knob_constraints: BTreeMap<String, RecipeKnobConstraintV1> = policy
        .recipe_knob_constraints
        .iter()
        .cloned()
        .map(|constraint| (constraint.knob_id.clone(), constraint))
        .collect();

    let mut remaining: BTreeSet<String> = evaluations
        .iter()
        .filter(|evaluation| evaluation.state == CandidateScreenStateV1::Feasible)
        .map(|evaluation| evaluation.candidate_id.clone())
        .collect();
    let mut selected: Vec<String> = Vec::new();

    while !remaining.is_empty() && selected.len() < policy.top_k {
        let mut best: Option<(String, f64)> = None;
        for candidate_id in &remaining {
            let evaluation = evaluations
                .iter()
                .find(|evaluation| evaluation.candidate_id == *candidate_id)
                .expect("candidate evaluation exists");
            let base_score = evaluation.base_score.unwrap_or(f64::INFINITY);
            let min_distance = selected
                .iter()
                .map(|selected_id| {
                    normalized_recipe_distance(
                        candidate_by_id[candidate_id.as_str()],
                        candidate_by_id[selected_id.as_str()],
                        &knob_constraints,
                    )
                })
                .fold(f64::INFINITY, f64::min);
            if !selected.is_empty() && min_distance < policy.minimum_recipe_diversity_distance {
                continue;
            }
            let diversity_bonus = if selected.is_empty() {
                0.0
            } else {
                policy.diversity_weight * min_distance
            };
            let selected_score = base_score - diversity_bonus;
            let replace = best.as_ref().is_none_or(|(best_id, best_score)| {
                match selected_score.total_cmp(best_score) {
                    std::cmp::Ordering::Less => true,
                    std::cmp::Ordering::Equal => candidate_id < best_id,
                    std::cmp::Ordering::Greater => false,
                }
            });
            if replace {
                best = Some((candidate_id.clone(), selected_score));
            }
        }
        let Some((candidate_id, selected_score)) = best else {
            break;
        };
        remaining.remove(&candidate_id);
        let rank = selected.len() + 1;
        if let Some(evaluation) = evaluations
            .iter_mut()
            .find(|evaluation| evaluation.candidate_id == candidate_id)
        {
            evaluation.selected_rank = Some(rank);
            evaluation.selected_score = Some(selected_score);
        }
        selected.push(candidate_id);
    }
    selected
}

fn unknown_report(
    input: &InverseVmShortlistInputV1,
    reason: CandidateScreenReasonV1,
) -> Result<InverseVmShortlistReportV1, InverseVmShortlistErrorV1> {
    let mut evaluations: Vec<InverseVmCandidateEvaluationV1> = input
        .candidates
        .iter()
        .map(|candidate| InverseVmCandidateEvaluationV1 {
            candidate_id: candidate.candidate_id.clone(),
            state: CandidateScreenStateV1::Unknown,
            reasons: vec![reason],
            base_score: None,
            selected_score: None,
            selected_rank: None,
            candidate_evidence_sha256: candidate.candidate_evidence_sha256.clone(),
        })
        .collect();
    evaluations.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
    let mut report = InverseVmShortlistReportV1 {
        schema: INVERSE_VM_SHORTLIST_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        process_snapshot_sha256: input.process_snapshot_sha256.clone(),
        decision_at_unix_seconds: input.decision_at_unix_seconds,
        policy_sha256: policy_digest(&input.policy)?,
        constraint_source_sha256: input.policy.recipe_constraint_source_sha256.clone(),
        state: ShortlistEvaluationStateV1::Unknown,
        evaluations,
        top_k_candidate_ids: Vec::new(),
        feasible_candidate_count: 0,
        rejected_candidate_count: 0,
        unknown_candidate_count: input.candidates.len(),
        candidate_generation_executed: false,
        recipe_write_executed: false,
        physical_experiment_authorized: false,
        equipment_control_authority: false,
        forward_evaluation_receipt_authenticity_proven: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

/// Screens and ranks caller-proposed recipe candidates. The input inventory
/// must be complete; missing constraints yield UNKNOWN, not implicit bounds.
/// Only candidates that pass every hard support/interval/recipe constraint
/// are eligible for top-K diversity selection.
pub fn evaluate_inverse_vm_shortlist_v1(
    input: &InverseVmShortlistInputV1,
) -> Result<InverseVmShortlistReportV1, InverseVmShortlistErrorV1> {
    if input.schema != INVERSE_VM_SHORTLIST_INPUT_SCHEMA_V1 {
        return Err(InverseVmShortlistErrorV1(
            "shortlist input schema is unsupported".into(),
        ));
    }
    require_text(&input.input_id, "input.input_id")?;
    require_text(&input.clock_domain_id, "input.clock_domain_id")?;
    if !valid_sha256(&input.clock_mapping_sha256) {
        return Err(InverseVmShortlistErrorV1(
            "decision clock mapping digest is malformed".into(),
        ));
    }
    if input.decision_at_unix_seconds < 0 {
        return Err(InverseVmShortlistErrorV1(
            "decision knowledge frontier must be a non-negative Unix timestamp".into(),
        ));
    }
    if !valid_sha256(&input.process_snapshot_sha256) {
        return Err(InverseVmShortlistErrorV1(
            "process snapshot digest is malformed".into(),
        ));
    }
    validate_policy(&input.policy)?;

    let mut candidate_ids = BTreeSet::new();
    for candidate in &input.candidates {
        require_text(&candidate.candidate_id, "candidate.candidate_id")?;
        if !candidate_ids.insert(candidate.candidate_id.as_str()) {
            return Err(InverseVmShortlistErrorV1("duplicate candidate ID".into()));
        }
    }

    if !input.policy.recipe_constraints_complete || !input.policy.side_constraints_complete {
        return unknown_report(input, CandidateScreenReasonV1::ConstraintSourceUnavailable);
    }

    let knob_constraints: BTreeMap<String, RecipeKnobConstraintV1> = input
        .policy
        .recipe_knob_constraints
        .iter()
        .cloned()
        .map(|constraint| (constraint.knob_id.clone(), constraint))
        .collect();
    let side_constraints: BTreeMap<String, SideConstraintV1> = input
        .policy
        .side_constraints
        .iter()
        .cloned()
        .map(|constraint| (constraint.constraint_id.clone(), constraint))
        .collect();

    let mut candidates = input.candidates.clone();
    candidates.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
    let mut evaluations = Vec::with_capacity(candidates.len());
    for candidate in &candidates {
        evaluations.push(screen_candidate(
            input,
            candidate,
            &knob_constraints,
            &side_constraints,
        )?);
    }

    let top_k_candidate_ids = build_shortlist(&candidates, &mut evaluations, &input.policy);
    evaluations.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
    let feasible_candidate_count = evaluations
        .iter()
        .filter(|evaluation| evaluation.state == CandidateScreenStateV1::Feasible)
        .count();
    let rejected_candidate_count = evaluations
        .iter()
        .filter(|evaluation| evaluation.state == CandidateScreenStateV1::Rejected)
        .count();
    let unknown_candidate_count = evaluations
        .iter()
        .filter(|evaluation| evaluation.state == CandidateScreenStateV1::Unknown)
        .count();
    let mut report = InverseVmShortlistReportV1 {
        schema: INVERSE_VM_SHORTLIST_REPORT_SCHEMA_V1.into(),
        input_id: input.input_id.clone(),
        process_snapshot_sha256: input.process_snapshot_sha256.clone(),
        decision_at_unix_seconds: input.decision_at_unix_seconds,
        policy_sha256: policy_digest(&input.policy)?,
        constraint_source_sha256: input.policy.recipe_constraint_source_sha256.clone(),
        state: ShortlistEvaluationStateV1::Complete,
        evaluations,
        top_k_candidate_ids,
        feasible_candidate_count,
        rejected_candidate_count,
        unknown_candidate_count,
        candidate_generation_executed: false,
        recipe_write_executed: false,
        physical_experiment_authorized: false,
        equipment_control_authority: false,
        forward_evaluation_receipt_authenticity_proven: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

/// Recomputes the complete shortlist and compares the entire replayable receipt.
pub fn verify_inverse_vm_shortlist_v1(
    input: &InverseVmShortlistInputV1,
    supplied: &InverseVmShortlistReportV1,
) -> Result<bool, InverseVmShortlistErrorV1> {
    if report_digest(supplied)? != supplied.report_sha256 {
        return Ok(false);
    }
    let expected = evaluate_inverse_vm_shortlist_v1(input)?;
    Ok(expected == *supplied)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn policy() -> InverseVmShortlistPolicyV1 {
        InverseVmShortlistPolicyV1 {
            schema: INVERSE_VM_SHORTLIST_POLICY_SCHEMA_V1.into(),
            policy_id: "epi-offline-recommendation-policy.v1".into(),
            revision: 1,
            recipe_constraint_source_sha256: digest('a'),
            recipe_constraints_complete: true,
            recipe_knob_constraints: vec![
                RecipeKnobConstraintV1 {
                    knob_id: "rf_power".into(),
                    unit: "W".into(),
                    allowed: NumericRangeV1 {
                        lower: 100.0,
                        upper: 500.0,
                    },
                },
                RecipeKnobConstraintV1 {
                    knob_id: "pressure".into(),
                    unit: "mTorr".into(),
                    allowed: NumericRangeV1 {
                        lower: 10.0,
                        upper: 100.0,
                    },
                },
            ],
            side_constraints_complete: true,
            side_constraints: vec![SideConstraintV1 {
                constraint_id: "uniformity".into(),
                measurand_id: "uniformity".into(),
                unit: "%".into(),
                allowed: NumericRangeV1 {
                    lower: 0.0,
                    upper: 3.0,
                },
            }],
            target: MetrologyTargetV1 {
                measurand_id: "epi_thickness".into(),
                unit: "nm".into(),
                target_value: 100.0,
                acceptable: NumericRangeV1 {
                    lower: 98.0,
                    upper: 102.0,
                },
            },
            required_forward_model_sha256: digest('b'),
            support_manifest_sha256: digest('c'),
            support_policy_sha256: digest('d'),
            minimum_support_neighbours: 2,
            maximum_normalized_support_distance: 0.75,
            maximum_interval_width_fraction: 0.75,
            target_error_weight: 1.0,
            interval_width_weight: 0.5,
            support_distance_weight: 0.25,
            diversity_weight: 0.05,
            minimum_recipe_diversity_distance: 0.02,
            top_k: 3,
        }
    }

    fn candidate(
        id: &str,
        rf: f64,
        pressure: f64,
        pred: (f64, f64),
        support_distance: f64,
    ) -> InverseVmCandidateV1 {
        let mut value = InverseVmCandidateV1 {
            candidate_id: id.into(),
            process_snapshot_sha256: digest('e'),
            recipe_settings: vec![
                RecipeSettingV1 {
                    knob_id: "rf_power".into(),
                    unit: "W".into(),
                    value: rf,
                },
                RecipeSettingV1 {
                    knob_id: "pressure".into(),
                    unit: "mTorr".into(),
                    value: pressure,
                },
            ],
            historical_support: Some(HistoricalSupportEvidenceV1 {
                support_manifest_sha256: digest('c'),
                neighbour_ids: vec![format!("n-{id}-1"), format!("n-{id}-2")],
                normalized_distance: support_distance,
                support_policy_sha256: digest('d'),
            }),
            forward_model_sha256: digest('b'),
            forward_evidence_state: ForwardEvidenceStateV1::Qualified,
            forward_evaluation_receipt: None,
            predicted_metrics: vec![
                PredictedMetricIntervalV1 {
                    measurand_id: "epi_thickness".into(),
                    unit: "nm".into(),
                    lower: pred.0,
                    upper: pred.1,
                },
                PredictedMetricIntervalV1 {
                    measurand_id: "uniformity".into(),
                    unit: "%".into(),
                    lower: 1.0,
                    upper: 2.0,
                },
            ],
            candidate_evidence_sha256: String::new(),
        };
        value.forward_evaluation_receipt = Some(forward_receipt(&value));
        value.candidate_evidence_sha256 = candidate_digest(&value).unwrap();
        value
    }

    fn forward_receipt(candidate: &InverseVmCandidateV1) -> ForwardEvaluationReceiptV1 {
        let mut receipt = ForwardEvaluationReceiptV1 {
            schema: FORWARD_EVALUATION_RECEIPT_SCHEMA_V1.into(),
            evaluator_id: "forward-evaluator.v1".into(),
            clock_domain_id: "fab-clock".into(),
            clock_mapping_sha256: digest('9'),
            process_snapshot_sha256: candidate.process_snapshot_sha256.clone(),
            forward_model_sha256: candidate.forward_model_sha256.clone(),
            model_qualification_report_sha256: Some(digest('f')),
            recipe_settings_sha256: recipe_settings_digest(&candidate.recipe_settings).unwrap(),
            predicted_metrics_sha256: predicted_metrics_digest(&candidate.predicted_metrics).unwrap(),
            evidence_state: candidate.forward_evidence_state,
            evaluated_at_unix_seconds: 100,
            available_at_unix_seconds: 110,
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = forward_receipt_digest(&receipt).unwrap();
        receipt
    }

    fn input(candidates: Vec<InverseVmCandidateV1>) -> InverseVmShortlistInputV1 {
        InverseVmShortlistInputV1 {
            schema: INVERSE_VM_SHORTLIST_INPUT_SCHEMA_V1.into(),
            input_id: "inverse-epi-input-001".into(),
            process_snapshot_sha256: digest('e'),
            decision_at_unix_seconds: 2_000,
            clock_domain_id: "fab-clock".into(),
            clock_mapping_sha256: digest('9'),
            policy: policy(),
            candidates,
        }
    }

    #[test]
    fn missing_forward_evaluation_receipt_is_unknown() {
        let mut candidate = candidate("missing-forward-receipt", 200.0, 50.0, (99.0, 101.0), 0.2);
        candidate.forward_evaluation_receipt = None;
        candidate.candidate_evidence_sha256 = candidate_digest(&candidate).unwrap();
        let report = evaluate_inverse_vm_shortlist_v1(&input(vec![candidate])).unwrap();
        assert_eq!(report.evaluations[0].state, CandidateScreenStateV1::Unknown);
        assert!(
            report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::ForwardEvaluationReceiptMissing)
        );
        assert!(report.top_k_candidate_ids.is_empty());
        assert!(!report.forward_evaluation_receipt_authenticity_proven);
    }

    #[test]
    fn forward_evaluation_receipt_after_decision_frontier_is_unknown() {
        let mut input = input(vec![candidate("late-forward-receipt", 200.0, 50.0, (99.0, 101.0), 0.2)]);
        input.decision_at_unix_seconds = 105;
        let report = evaluate_inverse_vm_shortlist_v1(&input).unwrap();
        assert_eq!(report.evaluations[0].state, CandidateScreenStateV1::Unknown);
        assert!(
            report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::ForwardEvaluationReceiptNotYetAvailable)
        );
        assert!(report.top_k_candidate_ids.is_empty());
    }

    #[test]
    fn forward_receipt_recipe_or_prediction_mutation_is_rejected() {
        let mut wrong_recipe = candidate("wrong-recipe-receipt", 200.0, 50.0, (99.0, 101.0), 0.2);
        wrong_recipe.recipe_settings[0].value += 1.0;
        wrong_recipe.candidate_evidence_sha256 = candidate_digest(&wrong_recipe).unwrap();
        let recipe_report = evaluate_inverse_vm_shortlist_v1(&input(vec![wrong_recipe])).unwrap();
        assert_eq!(recipe_report.evaluations[0].state, CandidateScreenStateV1::Rejected);
        assert!(
            recipe_report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::ForwardEvaluationReceiptRecipeMismatch)
        );

        let mut wrong_prediction = candidate("wrong-prediction-receipt", 200.0, 50.0, (99.0, 101.0), 0.2);
        wrong_prediction.predicted_metrics[0].upper = 100.5;
        wrong_prediction.candidate_evidence_sha256 = candidate_digest(&wrong_prediction).unwrap();
        let prediction_report =
            evaluate_inverse_vm_shortlist_v1(&input(vec![wrong_prediction])).unwrap();
        assert_eq!(prediction_report.evaluations[0].state, CandidateScreenStateV1::Rejected);
        assert!(
            prediction_report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::ForwardEvaluationReceiptPredictionMismatch)
        );
    }

    #[test]
    fn forward_receipt_context_and_digest_tampering_are_rejected() {
        let mut wrong_context = candidate("wrong-forward-context", 200.0, 50.0, (99.0, 101.0), 0.2);
        let receipt = wrong_context.forward_evaluation_receipt.as_mut().unwrap();
        receipt.process_snapshot_sha256 = digest('f');
        receipt.receipt_sha256 = forward_receipt_digest(receipt).unwrap();
        wrong_context.candidate_evidence_sha256 = candidate_digest(&wrong_context).unwrap();
        let context_report = evaluate_inverse_vm_shortlist_v1(&input(vec![wrong_context])).unwrap();
        assert_eq!(context_report.evaluations[0].state, CandidateScreenStateV1::Rejected);
        assert!(
            context_report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::ForwardEvaluationReceiptContextMismatch)
        );

        let mut tampered = candidate("tampered-forward", 200.0, 50.0, (99.0, 101.0), 0.2);
        tampered.forward_evaluation_receipt.as_mut().unwrap().receipt_sha256 = digest('0');
        tampered.candidate_evidence_sha256 = candidate_digest(&tampered).unwrap();
        let tampered_report = evaluate_inverse_vm_shortlist_v1(&input(vec![tampered])).unwrap();
        assert_eq!(tampered_report.evaluations[0].state, CandidateScreenStateV1::Rejected);
        assert!(
            tampered_report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::ForwardEvaluationReceiptDigestMismatch)
        );
    }

    #[test]
    fn forward_receipt_clock_domain_or_mapping_mismatch_is_rejected() {
        let mut candidate = candidate("wrong-clock-map", 200.0, 50.0, (99.0, 101.0), 0.2);
        let receipt = candidate.forward_evaluation_receipt.as_mut().unwrap();
        receipt.clock_mapping_sha256 = digest('8');
        receipt.receipt_sha256 = forward_receipt_digest(receipt).unwrap();
        candidate.candidate_evidence_sha256 = candidate_digest(&candidate).unwrap();
        let report = evaluate_inverse_vm_shortlist_v1(&input(vec![candidate])).unwrap();
        assert_eq!(report.evaluations[0].state, CandidateScreenStateV1::Rejected);
        assert!(
            report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::ForwardEvaluationReceiptClockDomainMismatch)
        );
    }

    #[test]
    fn qualification_digest_is_required_for_forward_receipt_marked_qualified() {
        let mut candidate = candidate("missing-qual-digest", 200.0, 50.0, (99.0, 101.0), 0.2);
        candidate.forward_evaluation_receipt.as_mut().unwrap().model_qualification_report_sha256 = None;
        let receipt = candidate.forward_evaluation_receipt.as_mut().unwrap();
        receipt.receipt_sha256 = forward_receipt_digest(receipt).unwrap();
        candidate.candidate_evidence_sha256 = candidate_digest(&candidate).unwrap();
        let report = evaluate_inverse_vm_shortlist_v1(&input(vec![candidate])).unwrap();
        assert_eq!(report.evaluations[0].state, CandidateScreenStateV1::Rejected);
        assert!(
            report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::ForwardEvaluationReceiptDigestMismatch)
        );
    }

    #[test]
    fn report_binds_decision_frontier_and_disclaims_forward_issuer_authenticity() {
        let mut input = input(vec![candidate("one", 200.0, 50.0, (99.0, 101.0), 0.2)]);
        input.decision_at_unix_seconds = 2_001;
        let report = evaluate_inverse_vm_shortlist_v1(&input).unwrap();
        assert_eq!(report.decision_at_unix_seconds, 2_001);
        assert!(!report.forward_evaluation_receipt_authenticity_proven);
        assert!(verify_inverse_vm_shortlist_v1(&input, &report).unwrap());
        let mut changed = input;
        changed.decision_at_unix_seconds = 2_002;
        assert!(!verify_inverse_vm_shortlist_v1(&changed, &report).unwrap());
    }

    #[test]
    fn feasible_candidates_are_ranked_and_replayed() {
        let input = input(vec![
            candidate("c", 200.0, 50.0, (99.0, 101.0), 0.30),
            candidate("a", 250.0, 45.0, (99.5, 100.5), 0.20),
            candidate("b", 450.0, 80.0, (98.5, 101.5), 0.40),
        ]);
        let report = evaluate_inverse_vm_shortlist_v1(&input).unwrap();
        assert_eq!(report.state, ShortlistEvaluationStateV1::Complete);
        assert_eq!(report.feasible_candidate_count, 3);
        assert_eq!(
            report.top_k_candidate_ids,
            vec!["a".to_string(), "c".to_string(), "b".to_string(),]
        );
        assert!(verify_inverse_vm_shortlist_v1(&input, &report).unwrap());
        assert!(!report.candidate_generation_executed);
        assert!(!report.recipe_write_executed);
        assert!(!report.physical_experiment_authorized);
        assert!(!report.equipment_control_authority);
    }

    #[test]
    fn out_of_range_recipe_candidate_is_rejected_before_ranking() {
        let input = input(vec![candidate("bad", 900.0, 50.0, (99.0, 101.0), 0.2)]);
        let report = evaluate_inverse_vm_shortlist_v1(&input).unwrap();
        assert!(report.top_k_candidate_ids.is_empty());
        assert_eq!(
            report.evaluations[0].state,
            CandidateScreenStateV1::Rejected
        );
        assert!(
            report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::RecipeValueOutsideHardBound)
        );
    }

    #[test]
    fn unsupported_candidate_is_rejected_even_with_attractive_prediction() {
        let input = input(vec![candidate("ood", 200.0, 50.0, (100.0, 100.1), 1.2)]);
        let report = evaluate_inverse_vm_shortlist_v1(&input).unwrap();
        assert!(report.top_k_candidate_ids.is_empty());
        assert!(
            report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::HistoricalSupportOutsideEnvelope)
        );
    }

    #[test]
    fn uncertainty_interval_crossing_target_or_side_constraint_is_rejected() {
        let mut wide_target = candidate("wide", 200.0, 50.0, (97.0, 101.0), 0.2);
        wide_target.candidate_evidence_sha256 = candidate_digest(&wide_target).unwrap();
        let input = input(vec![wide_target]);
        let report = evaluate_inverse_vm_shortlist_v1(&input).unwrap();
        assert_eq!(
            report.evaluations[0].state,
            CandidateScreenStateV1::Rejected
        );
        assert!(
            report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::TargetIntervalNotContained)
        );
    }

    #[test]
    fn missing_constraint_inventory_returns_unknown_and_empty_shortlist() {
        let mut input = input(vec![candidate("one", 200.0, 50.0, (99.0, 101.0), 0.2)]);
        input.policy.recipe_constraints_complete = false;
        let report = evaluate_inverse_vm_shortlist_v1(&input).unwrap();
        assert_eq!(report.state, ShortlistEvaluationStateV1::Unknown);
        assert!(report.top_k_candidate_ids.is_empty());
        assert_eq!(report.unknown_candidate_count, 1);
    }

    #[test]
    fn missing_support_receipt_is_unknown_not_feasible() {
        let mut candidate = candidate("no-support", 200.0, 50.0, (99.0, 101.0), 0.2);
        candidate.historical_support = None;
        candidate.candidate_evidence_sha256 = candidate_digest(&candidate).unwrap();
        let report = evaluate_inverse_vm_shortlist_v1(&input(vec![candidate])).unwrap();
        assert_eq!(report.evaluations[0].state, CandidateScreenStateV1::Unknown);
        assert!(report.top_k_candidate_ids.is_empty());
    }

    #[test]
    fn malformed_support_neighbour_id_is_rejected() {
        let mut candidate = candidate("bad-support-id", 200.0, 50.0, (99.0, 101.0), 0.2);
        candidate.historical_support.as_mut().unwrap().neighbour_ids[0] = " ".into();
        candidate.candidate_evidence_sha256 = candidate_digest(&candidate).unwrap();
        let report = evaluate_inverse_vm_shortlist_v1(&input(vec![candidate])).unwrap();
        assert_eq!(
            report.evaluations[0].state,
            CandidateScreenStateV1::Rejected
        );
        assert!(
            report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::InvalidSupportNeighbourId)
        );
    }

    #[test]
    fn unknown_forward_evidence_never_becomes_a_recommendation() {
        let mut candidate = candidate("unknown-forward", 200.0, 50.0, (99.0, 101.0), 0.2);
        candidate.forward_evidence_state = ForwardEvidenceStateV1::Unknown;
        let receipt = candidate.forward_evaluation_receipt.as_mut().unwrap();
        receipt.evidence_state = ForwardEvidenceStateV1::Unknown;
        receipt.receipt_sha256 = forward_receipt_digest(receipt).unwrap();
        candidate.candidate_evidence_sha256 = candidate_digest(&candidate).unwrap();
        let report = evaluate_inverse_vm_shortlist_v1(&input(vec![candidate])).unwrap();
        assert_eq!(report.evaluations[0].state, CandidateScreenStateV1::Unknown);
        assert!(report.top_k_candidate_ids.is_empty());
    }

    #[test]
    fn incomplete_support_or_wrong_context_cannot_hide_behind_candidate_digest() {
        let mut candidate = candidate("wrong-context", 200.0, 50.0, (99.0, 101.0), 0.2);
        candidate.process_snapshot_sha256 = digest('f');
        candidate.candidate_evidence_sha256 = candidate_digest(&candidate).unwrap();
        let report = evaluate_inverse_vm_shortlist_v1(&input(vec![candidate])).unwrap();
        assert_eq!(
            report.evaluations[0].state,
            CandidateScreenStateV1::Rejected
        );
        assert!(
            report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::ContextSnapshotMismatch)
        );
    }

    #[test]
    fn duplicate_candidate_and_duplicate_recipe_knobs_are_rejected() {
        let duplicate_ids = vec![
            candidate("same", 200.0, 50.0, (99.0, 101.0), 0.2),
            candidate("same", 250.0, 55.0, (99.0, 101.0), 0.2),
        ];
        assert!(evaluate_inverse_vm_shortlist_v1(&input(duplicate_ids)).is_err());

        let mut duplicated = candidate("dupe-knob", 200.0, 50.0, (99.0, 101.0), 0.2);
        duplicated.recipe_settings.push(RecipeSettingV1 {
            knob_id: "rf_power".into(),
            unit: "W".into(),
            value: 220.0,
        });
        duplicated.candidate_evidence_sha256 = candidate_digest(&duplicated).unwrap();
        let report = evaluate_inverse_vm_shortlist_v1(&input(vec![duplicated])).unwrap();
        assert_eq!(
            report.evaluations[0].state,
            CandidateScreenStateV1::Rejected
        );
        assert!(
            report.evaluations[0]
                .reasons
                .contains(&CandidateScreenReasonV1::DuplicateRecipeKnob)
        );
    }

    #[test]
    fn input_order_and_candidate_order_do_not_change_receipt() {
        let a = candidate("a", 200.0, 50.0, (99.0, 101.0), 0.2);
        let b = candidate("b", 250.0, 55.0, (99.0, 101.0), 0.3);
        let c = candidate("c", 280.0, 60.0, (99.0, 101.0), 0.4);
        let first = input(vec![a.clone(), b.clone(), c.clone()]);
        let mut second = input(vec![c, b, a]);
        second.policy.recipe_knob_constraints.reverse();
        let left = evaluate_inverse_vm_shortlist_v1(&first).unwrap();
        let right = evaluate_inverse_vm_shortlist_v1(&second).unwrap();
        assert_eq!(left.report_sha256, right.report_sha256);
    }

    #[test]
    fn minimum_diversity_distance_can_leave_shortlist_shorter_than_k() {
        let a = candidate("a", 200.0, 50.0, (99.0, 101.0), 0.2);
        let b = candidate("b", 201.0, 50.1, (99.0, 101.0), 0.2);
        let mut policy = policy();
        policy.top_k = 2;
        policy.minimum_recipe_diversity_distance = 0.10;
        let input = InverseVmShortlistInputV1 {
            schema: INVERSE_VM_SHORTLIST_INPUT_SCHEMA_V1.into(),
            input_id: "diversity-minimum-test".into(),
            process_snapshot_sha256: digest('e'),
            decision_at_unix_seconds: 2_000,
            clock_domain_id: "fab-clock".into(),
            clock_mapping_sha256: digest('9'),
            policy,
            candidates: vec![a, b],
        };
        let report = evaluate_inverse_vm_shortlist_v1(&input).unwrap();
        assert_eq!(report.feasible_candidate_count, 2);
        assert_eq!(report.top_k_candidate_ids.len(), 1);
    }

    #[test]
    fn tampered_shortlist_receipt_is_rejected_by_independent_replay() {
        let input = input(vec![candidate("a", 200.0, 50.0, (99.0, 101.0), 0.2)]);
        let mut report = evaluate_inverse_vm_shortlist_v1(&input).unwrap();
        report.top_k_candidate_ids.clear();
        report.report_sha256 = report_digest(&report).unwrap();
        assert!(!verify_inverse_vm_shortlist_v1(&input, &report).unwrap());
    }

    #[test]
    fn shortlist_matches_independent_exhaustive_lexicographic_path_oracle() {
        #[derive(Clone)]
        struct OraclePath {
            ids: Vec<String>,
            priority: Vec<(f64, String)>,
        }

        fn distance_oracle(
            left: &InverseVmCandidateV1,
            right: &InverseVmCandidateV1,
            policy: &InverseVmShortlistPolicyV1,
        ) -> f64 {
            let right_by_id: BTreeMap<&str, &RecipeSettingV1> = right
                .recipe_settings
                .iter()
                .map(|setting| (setting.knob_id.as_str(), setting))
                .collect();
            let mut squared = 0.0;
            let mut dimensions = 0_usize;
            for left_setting in &left.recipe_settings {
                let Some(right_setting) = right_by_id.get(left_setting.knob_id.as_str()) else {
                    continue;
                };
                let Some(bound) = policy
                    .recipe_knob_constraints
                    .iter()
                    .find(|constraint| constraint.knob_id == left_setting.knob_id)
                else {
                    continue;
                };
                let width = (bound.allowed.upper - bound.allowed.lower).max(f64::EPSILON);
                let normalized = (left_setting.value - right_setting.value) / width;
                squared += normalized * normalized;
                dimensions += 1;
            }
            if dimensions == 0 {
                f64::INFINITY
            } else {
                (squared / dimensions as f64).sqrt()
            }
        }

        fn enumerate_paths(
            selected: &mut Vec<String>,
            priority: &mut Vec<(f64, String)>,
            remaining: &mut Vec<String>,
            candidates: &BTreeMap<String, InverseVmCandidateV1>,
            base_scores: &BTreeMap<String, f64>,
            policy: &InverseVmShortlistPolicyV1,
            paths: &mut Vec<OraclePath>,
        ) {
            if selected.len() == policy.top_k {
                paths.push(OraclePath {
                    ids: selected.clone(),
                    priority: priority.clone(),
                });
                return;
            }

            let mut extensions = Vec::new();
            for candidate_id in remaining.iter() {
                let candidate = &candidates[candidate_id];
                let min_distance = selected
                    .iter()
                    .map(|selected_id| distance_oracle(candidate, &candidates[selected_id], policy))
                    .fold(f64::INFINITY, f64::min);
                if !selected.is_empty() && min_distance < policy.minimum_recipe_diversity_distance {
                    continue;
                }
                let diversity_bonus = if selected.is_empty() {
                    0.0
                } else {
                    policy.diversity_weight * min_distance
                };
                extensions.push((
                    candidate_id.clone(),
                    base_scores[candidate_id] - diversity_bonus,
                ));
            }

            if extensions.is_empty() {
                paths.push(OraclePath {
                    ids: selected.clone(),
                    priority: priority.clone(),
                });
                return;
            }

            extensions.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            for (candidate_id, selected_score) in extensions {
                let index = remaining
                    .iter()
                    .position(|id| id == &candidate_id)
                    .expect("oracle candidate remains available");
                let removed = remaining.remove(index);
                selected.push(candidate_id.clone());
                priority.push((selected_score, candidate_id));
                enumerate_paths(
                    selected,
                    priority,
                    remaining,
                    candidates,
                    base_scores,
                    policy,
                    paths,
                );
                priority.pop();
                selected.pop();
                remaining.insert(index, removed);
            }
        }

        fn compare_paths(left: &OraclePath, right: &OraclePath) -> std::cmp::Ordering {
            for (left_step, right_step) in left.priority.iter().zip(&right.priority) {
                let order = left_step
                    .0
                    .total_cmp(&right_step.0)
                    .then_with(|| left_step.1.cmp(&right_step.1));
                if order != std::cmp::Ordering::Equal {
                    return order;
                }
            }
            left.priority.len().cmp(&right.priority.len())
        }

        let mut input = input(vec![
            candidate("a", 200.0, 25.0, (99.7, 100.3), 0.15),
            candidate("b", 205.0, 25.5, (100.0, 100.2), 0.30),
            candidate("c", 350.0, 55.0, (99.5, 100.5), 0.10),
            candidate("d", 480.0, 95.0, (99.3, 100.7), 0.20),
        ]);
        input.policy.top_k = 3;
        input.policy.minimum_recipe_diversity_distance = 0.02;

        let report = evaluate_inverse_vm_shortlist_v1(&input).unwrap();
        assert_eq!(report.feasible_candidate_count, 4);

        let candidates: BTreeMap<String, InverseVmCandidateV1> = input
            .candidates
            .iter()
            .cloned()
            .map(|candidate| (candidate.candidate_id.clone(), candidate))
            .collect();
        let base_scores: BTreeMap<String, f64> = report
            .evaluations
            .iter()
            .map(|evaluation| {
                (
                    evaluation.candidate_id.clone(),
                    evaluation
                        .base_score
                        .expect("all fixture candidates are feasible"),
                )
            })
            .collect();
        let mut remaining: Vec<String> = candidates.keys().cloned().collect();
        let mut paths = Vec::new();
        enumerate_paths(
            &mut Vec::new(),
            &mut Vec::new(),
            &mut remaining,
            &candidates,
            &base_scores,
            &input.policy,
            &mut paths,
        );
        let oracle = paths
            .iter()
            .min_by(|left, right| compare_paths(left, right))
            .expect("finite candidate pool has at least one path");

        assert_eq!(report.top_k_candidate_ids, oracle.ids);
        for (rank, (score, candidate_id)) in oracle.priority.iter().enumerate() {
            let evaluation = report
                .evaluations
                .iter()
                .find(|evaluation| evaluation.candidate_id == *candidate_id)
                .unwrap();
            assert_eq!(evaluation.selected_rank, Some(rank + 1));
            assert!(
                (evaluation.selected_score.unwrap() - score).abs() < 1.0e-12,
                "selected score differs for {candidate_id}"
            );
        }
    }

    #[test]
    fn diversity_only_ranks_candidates_after_feasibility_gate() {
        let mut good = candidate("good", 200.0, 50.0, (99.0, 101.0), 0.2);
        good.recipe_settings[0].value = 240.0;
        good.candidate_evidence_sha256 = candidate_digest(&good).unwrap();
        let mut unsafe_far = candidate("unsafe", 490.0, 99.0, (95.0, 106.0), 0.2);
        unsafe_far.candidate_evidence_sha256 = candidate_digest(&unsafe_far).unwrap();
        let report = evaluate_inverse_vm_shortlist_v1(&input(vec![good, unsafe_far])).unwrap();
        assert_eq!(report.top_k_candidate_ids, vec!["good".to_string()]);
        assert_eq!(report.rejected_candidate_count, 1);
    }
}
