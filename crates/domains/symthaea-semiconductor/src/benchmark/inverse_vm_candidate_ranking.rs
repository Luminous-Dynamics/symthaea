// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-qualified offline inverse-VM candidate ranking.
//!
//! This module ranks declared candidate control vectors only. It does not
//! generate recipes, execute corrections, write APC/R2R settings or authorize
//! equipment actions. Robust interval feasibility and historical support are
//! hard gates, not soft score penalties.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const INVERSE_VM_RANKING_SCHEMA_V1: &str =
    "luminous.semi-bench.inverse-vm-ranking.v1";
pub const INVERSE_VM_CANDIDATE_SCHEMA_V1: &str =
    "luminous.semi-bench.inverse-vm-candidate.v1";
pub const INVERSE_VM_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.inverse-vm-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateEvidenceStatusV1 {
    CurrentEligible,
    Unknown,
    Blocked,
    HistoricalOnly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateDispositionV1 {
    Eligible,
    Unknown,
    Unsupported,
    Infeasible,
    Rejected,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateRejectionReasonV1 {
    CandidateSchemaUnsupported,
    CandidateDigestMismatch,
    RecipeSchemaMismatch,
    CandidateIdEmpty,
    DuplicateCandidateId,
    MalformedModelArtifactDigest,
    MalformedPredictionReceiptDigest,
    MalformedSupportReceiptDigest,
    ModelEvidenceUnknown,
    ModelEvidenceBlocked,
    ModelEvidenceHistoricalOnly,
    ModelEvidenceUnavailableAtCutoff,
    SupportManifestMismatch,
    SupportEvidenceInsufficient,
    SupportDistanceInvalid,
    OutsideHistoricalSupport,
    ControlDimensionMissing,
    ControlDimensionMismatch,
    ControlValueNonFinite,
    ControlValueOutsideDeclaredRange,
    TargetPredictionMissing,
    TargetPredictionDuplicate,
    TargetUnitMismatch,
    TargetPredictionNonFinite,
    TargetUncertaintyInvalid,
    RobustTargetWindowNotSatisfied,
    ConstraintPredictionMissing,
    ConstraintPredictionDuplicate,
    ConstraintUnitMismatch,
    ConstraintPredictionNonFinite,
    ConstraintUncertaintyInvalid,
    RobustSideConstraintViolated,
    InsufficientCandidateCount,
    InputSchemaUnsupported,
    InputDigestMismatch,
    RankingPolicyInvalid,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecipeControlBoundV1 {
    pub control_id: String,
    pub unit: String,
    pub minimum: f64,
    pub maximum: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmTargetSpecV1 {
    pub target_id: String,
    pub unit: String,
    pub desired_value: f64,
    /// Maximum permitted absolute target deviation in the declared unit.
    pub tolerance: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmSideConstraintV1 {
    pub constraint_id: String,
    pub unit: String,
    pub minimum: f64,
    pub maximum: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateTargetPredictionV1 {
    pub target_id: String,
    pub unit: String,
    pub predicted_value: f64,
    pub standard_uncertainty: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CandidateConstraintPredictionV1 {
    pub constraint_id: String,
    pub unit: String,
    pub predicted_value: f64,
    pub standard_uncertainty: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmCandidateV1 {
    pub schema: String,
    pub candidate_id: String,
    pub recipe_schema_sha256: String,
    pub control_values: BTreeMap<String, f64>,
    pub model_artifact_sha256: String,
    pub forward_prediction_receipt_sha256: String,
    pub model_evidence_receipt_sha256: String,
    pub model_evidence_status: CandidateEvidenceStatusV1,
    pub model_evidence_available_at_unix_seconds: i64,
    pub support_manifest_sha256: String,
    pub support_receipt_sha256: String,
    /// Declared distance in the frozen historical-support metric space.
    pub historical_support_distance: f64,
    pub historical_support_radius: f64,
    pub historical_support_count: usize,
    pub target_predictions: Vec<CandidateTargetPredictionV1>,
    pub constraint_predictions: Vec<CandidateConstraintPredictionV1>,
    pub candidate_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmRankingInputV1 {
    pub schema: String,
    pub ranking_id: String,
    pub recipe_schema_sha256: String,
    pub historical_support_manifest_sha256: String,
    pub knowledge_cutoff_unix_seconds: i64,
    pub confidence_multiplier: f64,
    pub minimum_support_count: usize,
    pub requested_top_k: usize,
    /// Lower is better. Target risk and support ratio are separately exposed.
    pub support_score_weight: f64,
    /// Greedy normalized control-space separation bonus.
    pub diversity_weight: f64,
    pub minimum_diversity_distance: f64,
    pub control_bounds: Vec<RecipeControlBoundV1>,
    pub target_specs: Vec<InverseVmTargetSpecV1>,
    pub side_constraints: Vec<InverseVmSideConstraintV1>,
    pub candidates: Vec<InverseVmCandidateV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmCandidateEvaluationV1 {
    pub candidate_id: String,
    pub disposition: CandidateDispositionV1,
    pub reasons: Vec<CandidateRejectionReasonV1>,
    pub robust_target_risk: Option<f64>,
    pub maximum_support_ratio: Option<f64>,
    pub minimum_constraint_margin_ratio: Option<f64>,
    pub base_score: Option<f64>,
    pub diversity_bonus: f64,
    pub selection_score: Option<f64>,
    pub selected_rank: Option<usize>,
    pub candidate_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InverseVmRankingReportV1 {
    pub schema: String,
    pub ranking_id: String,
    pub input_sha256: String,
    pub selected_candidate_ids: Vec<String>,
    pub evaluations: Vec<InverseVmCandidateEvaluationV1>,
    pub eligible_count: usize,
    pub unknown_count: usize,
    pub unsupported_count: usize,
    pub infeasible_count: usize,
    pub rejected_count: usize,
    pub requested_top_k: usize,
    pub top_k_filled: bool,
    pub source_authenticity_proven: bool,
    pub physical_success_proven: bool,
    pub recipe_write_executed: bool,
    pub apc_r2r_authority: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct InverseVmRankingErrorV1(pub String);

fn require_text(value: &str, field: &str) -> Result<(), InverseVmRankingErrorV1> {
    if value.trim().is_empty() {
        return Err(InverseVmRankingErrorV1(format!(
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

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, InverseVmRankingErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        InverseVmRankingErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn candidate_digest(
    candidate: &InverseVmCandidateV1,
) -> Result<String, InverseVmRankingErrorV1> {
    let mut canonical = candidate.clone();
    canonical.candidate_sha256.clear();
    canonical.target_predictions.sort_by(|a, b| a.target_id.cmp(&b.target_id));
    canonical
        .constraint_predictions
        .sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
    json_sha256(&canonical)
}

fn input_digest(input: &InverseVmRankingInputV1) -> Result<String, InverseVmRankingErrorV1> {
    let mut canonical = input.clone();
    canonical.candidates.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
    for candidate in &mut canonical.candidates {
        candidate.target_predictions.sort_by(|a, b| a.target_id.cmp(&b.target_id));
        candidate
            .constraint_predictions
            .sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
    }
    canonical.control_bounds.sort_by(|a, b| a.control_id.cmp(&b.control_id));
    canonical.target_specs.sort_by(|a, b| a.target_id.cmp(&b.target_id));
    canonical
        .side_constraints
        .sort_by(|a, b| a.constraint_id.cmp(&b.constraint_id));
    json_sha256(&canonical)
}

fn report_digest(
    report: &InverseVmRankingReportV1,
) -> Result<String, InverseVmRankingErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    canonical.evaluations.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
    json_sha256(&canonical)
}

fn validate_input(input: &InverseVmRankingInputV1) -> Result<(), InverseVmRankingErrorV1> {
    if input.schema != INVERSE_VM_RANKING_SCHEMA_V1 {
        return Err(InverseVmRankingErrorV1("ranking input schema is unsupported".into()));
    }
    require_text(&input.ranking_id, "input.ranking_id")?;
    if !valid_sha256(&input.recipe_schema_sha256)
        || !valid_sha256(&input.historical_support_manifest_sha256)
        || input.requested_top_k == 0
        || input.minimum_support_count == 0
        || !input.confidence_multiplier.is_finite()
        || input.confidence_multiplier <= 0.0
        || !input.support_score_weight.is_finite()
        || input.support_score_weight < 0.0
        || !input.diversity_weight.is_finite()
        || input.diversity_weight < 0.0
        || !input.minimum_diversity_distance.is_finite()
        || input.minimum_diversity_distance < 0.0
        || (input.diversity_weight > 0.0 && input.minimum_diversity_distance == 0.0)
    {
        return Err(InverseVmRankingErrorV1(
            "ranking input contains invalid identity or policy values".into(),
        ));
    }
    if input.control_bounds.is_empty() || input.target_specs.is_empty() {
        return Err(InverseVmRankingErrorV1(
            "at least one control bound and target specification are required".into(),
        ));
    }
    let mut controls = BTreeSet::new();
    for control in &input.control_bounds {
        require_text(&control.control_id, "control.control_id")?;
        require_text(&control.unit, "control.unit")?;
        if !control.minimum.is_finite()
            || !control.maximum.is_finite()
            || control.minimum >= control.maximum
        {
            return Err(InverseVmRankingErrorV1(format!(
                "control {} has invalid bounds",
                control.control_id
            )));
        }
        if !controls.insert(control.control_id.as_str()) {
            return Err(InverseVmRankingErrorV1(format!(
                "duplicate control dimension {}",
                control.control_id
            )));
        }
    }
    let mut targets = BTreeSet::new();
    for target in &input.target_specs {
        require_text(&target.target_id, "target.target_id")?;
        require_text(&target.unit, "target.unit")?;
        if !target.desired_value.is_finite() || !target.tolerance.is_finite() || target.tolerance <= 0.0 {
            return Err(InverseVmRankingErrorV1(format!(
                "target {} has invalid target or tolerance",
                target.target_id
            )));
        }
        if !targets.insert(target.target_id.as_str()) {
            return Err(InverseVmRankingErrorV1(format!(
                "duplicate target specification {}",
                target.target_id
            )));
        }
    }
    let mut constraints = BTreeSet::new();
    for constraint in &input.side_constraints {
        require_text(&constraint.constraint_id, "constraint.constraint_id")?;
        require_text(&constraint.unit, "constraint.unit")?;
        if !constraint.minimum.is_finite()
            || !constraint.maximum.is_finite()
            || constraint.minimum >= constraint.maximum
        {
            return Err(InverseVmRankingErrorV1(format!(
                "constraint {} has invalid bounds",
                constraint.constraint_id
            )));
        }
        if !constraints.insert(constraint.constraint_id.as_str()) {
            return Err(InverseVmRankingErrorV1(format!(
                "duplicate side constraint {}",
                constraint.constraint_id
            )));
        }
    }
    Ok(())
}

pub fn seal_inverse_vm_candidate_v1(
    candidate: &InverseVmCandidateV1,
) -> Result<String, InverseVmRankingErrorV1> {
    candidate_digest(candidate)
}

#[derive(Clone, Debug)]
struct FeasibleCandidate {
    evaluation_index: usize,
    candidate: InverseVmCandidateV1,
    base_score: f64,
}

fn evaluate_candidate(
    input: &InverseVmRankingInputV1,
    candidate: &InverseVmCandidateV1,
) -> Result<InverseVmCandidateEvaluationV1, InverseVmRankingErrorV1> {
    let mut reasons = Vec::new();
    let mut disposition = CandidateDispositionV1::Eligible;

    let mut set = |next: CandidateDispositionV1, reason: CandidateRejectionReasonV1| {
        reasons.push(reason);
        let rank = |value| match value {
            CandidateDispositionV1::Eligible => 0,
            CandidateDispositionV1::Unknown => 1,
            CandidateDispositionV1::Unsupported => 2,
            CandidateDispositionV1::Infeasible => 3,
            CandidateDispositionV1::Rejected => 4,
        };
        if rank(next) > rank(disposition) {
            disposition = next;
        }
    };

    if candidate.schema != INVERSE_VM_CANDIDATE_SCHEMA_V1 {
        set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::CandidateSchemaUnsupported);
    }
    if candidate.candidate_id.trim().is_empty() {
        set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::CandidateIdEmpty);
    }
    if candidate_digest(candidate)? != candidate.candidate_sha256 {
        set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::CandidateDigestMismatch);
    }
    if !valid_sha256(&candidate.recipe_schema_sha256)
        || candidate.recipe_schema_sha256 != input.recipe_schema_sha256
    {
        set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::RecipeSchemaMismatch);
    }
    if !valid_sha256(&candidate.model_artifact_sha256) {
        set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::MalformedModelArtifactDigest);
    }
    if !valid_sha256(&candidate.forward_prediction_receipt_sha256) {
        set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::MalformedPredictionReceiptDigest);
    }
    if !valid_sha256(&candidate.model_evidence_receipt_sha256) {
        set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::MalformedPredictionReceiptDigest);
    }
    if !valid_sha256(&candidate.support_manifest_sha256)
        || candidate.support_manifest_sha256 != input.historical_support_manifest_sha256
    {
        set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::SupportManifestMismatch);
    }
    if !valid_sha256(&candidate.support_receipt_sha256) {
        set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::MalformedSupportReceiptDigest);
    }

    match candidate.model_evidence_status {
        CandidateEvidenceStatusV1::CurrentEligible => {}
        CandidateEvidenceStatusV1::Unknown => set(
            CandidateDispositionV1::Unknown,
            CandidateRejectionReasonV1::ModelEvidenceUnknown,
        ),
        CandidateEvidenceStatusV1::Blocked => set(
            CandidateDispositionV1::Rejected,
            CandidateRejectionReasonV1::ModelEvidenceBlocked,
        ),
        CandidateEvidenceStatusV1::HistoricalOnly => set(
            CandidateDispositionV1::Rejected,
            CandidateRejectionReasonV1::ModelEvidenceHistoricalOnly,
        ),
    }
    if candidate.model_evidence_available_at_unix_seconds > input.knowledge_cutoff_unix_seconds {
        set(
            CandidateDispositionV1::Unknown,
            CandidateRejectionReasonV1::ModelEvidenceUnavailableAtCutoff,
        );
    }
    if !candidate.historical_support_distance.is_finite()
        || !candidate.historical_support_radius.is_finite()
        || candidate.historical_support_distance < 0.0
        || candidate.historical_support_radius <= 0.0
    {
        set(
            CandidateDispositionV1::Rejected,
            CandidateRejectionReasonV1::SupportDistanceInvalid,
        );
    } else if candidate.historical_support_distance > candidate.historical_support_radius {
        set(
            CandidateDispositionV1::Unsupported,
            CandidateRejectionReasonV1::OutsideHistoricalSupport,
        );
    }
    if candidate.historical_support_count < input.minimum_support_count {
        set(
            CandidateDispositionV1::Unknown,
            CandidateRejectionReasonV1::SupportEvidenceInsufficient,
        );
    }

    let control_map: BTreeMap<&str, &RecipeControlBoundV1> = input
        .control_bounds
        .iter()
        .map(|item| (item.control_id.as_str(), item))
        .collect();
    for bound in &input.control_bounds {
        match candidate.control_values.get(&bound.control_id) {
            None => set(
                CandidateDispositionV1::Rejected,
                CandidateRejectionReasonV1::ControlDimensionMissing,
            ),
            Some(value) if !value.is_finite() => set(
                CandidateDispositionV1::Rejected,
                CandidateRejectionReasonV1::ControlValueNonFinite,
            ),
            Some(value) if *value < bound.minimum || *value > bound.maximum => set(
                CandidateDispositionV1::Infeasible,
                CandidateRejectionReasonV1::ControlValueOutsideDeclaredRange,
            ),
            Some(_) => {}
        }
    }
    if candidate.control_values.len() != input.control_bounds.len()
        || candidate.control_values.keys().any(|key| !control_map.contains_key(key.as_str()))
    {
        set(
            CandidateDispositionV1::Rejected,
            CandidateRejectionReasonV1::ControlDimensionMismatch,
        );
    }

    let target_predictions: BTreeMap<&str, &CandidateTargetPredictionV1> = {
        let mut map = BTreeMap::new();
        for prediction in &candidate.target_predictions {
            if map.insert(prediction.target_id.as_str(), prediction).is_some() {
                set(
                    CandidateDispositionV1::Rejected,
                    CandidateRejectionReasonV1::TargetPredictionDuplicate,
                );
            }
        }
        map
    };
    let mut robust_target_risk = 0.0;
    for spec in &input.target_specs {
        let Some(prediction) = target_predictions.get(spec.target_id.as_str()) else {
            set(
                CandidateDispositionV1::Rejected,
                CandidateRejectionReasonV1::TargetPredictionMissing,
            );
            continue;
        };
        if prediction.unit != spec.unit {
            set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::TargetUnitMismatch);
            continue;
        }
        if !prediction.predicted_value.is_finite() {
            set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::TargetPredictionNonFinite);
            continue;
        }
        if !prediction.standard_uncertainty.is_finite() || prediction.standard_uncertainty < 0.0 {
            set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::TargetUncertaintyInvalid);
            continue;
        }
        let robust_error = (prediction.predicted_value - spec.desired_value).abs()
            + input.confidence_multiplier * prediction.standard_uncertainty;
        let ratio = robust_error / spec.tolerance;
        robust_target_risk += ratio;
        if ratio > 1.0 {
            set(
                CandidateDispositionV1::Infeasible,
                CandidateRejectionReasonV1::RobustTargetWindowNotSatisfied,
            );
        }
    }
    if !input.target_specs.is_empty() {
        robust_target_risk /= input.target_specs.len() as f64;
    }
    if target_predictions.len() != input.target_specs.len() {
        set(
            CandidateDispositionV1::Rejected,
            CandidateRejectionReasonV1::TargetPredictionMissing,
        );
    }

    let constraint_predictions: BTreeMap<&str, &CandidateConstraintPredictionV1> = {
        let mut map = BTreeMap::new();
        for prediction in &candidate.constraint_predictions {
            if map.insert(prediction.constraint_id.as_str(), prediction).is_some() {
                set(
                    CandidateDispositionV1::Rejected,
                    CandidateRejectionReasonV1::ConstraintPredictionDuplicate,
                );
            }
        }
        map
    };
    let mut minimum_constraint_margin_ratio = f64::INFINITY;
    for spec in &input.side_constraints {
        let Some(prediction) = constraint_predictions.get(spec.constraint_id.as_str()) else {
            set(
                CandidateDispositionV1::Rejected,
                CandidateRejectionReasonV1::ConstraintPredictionMissing,
            );
            continue;
        };
        if prediction.unit != spec.unit {
            set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::ConstraintUnitMismatch);
            continue;
        }
        if !prediction.predicted_value.is_finite() {
            set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::ConstraintPredictionNonFinite);
            continue;
        }
        if !prediction.standard_uncertainty.is_finite() || prediction.standard_uncertainty < 0.0 {
            set(CandidateDispositionV1::Rejected, CandidateRejectionReasonV1::ConstraintUncertaintyInvalid);
            continue;
        }
        let lower = prediction.predicted_value
            - input.confidence_multiplier * prediction.standard_uncertainty;
        let upper = prediction.predicted_value
            + input.confidence_multiplier * prediction.standard_uncertainty;
        if lower < spec.minimum || upper > spec.maximum {
            set(
                CandidateDispositionV1::Infeasible,
                CandidateRejectionReasonV1::RobustSideConstraintViolated,
            );
        }
        let range = spec.maximum - spec.minimum;
        let margin = (lower - spec.minimum).min(spec.maximum - upper);
        minimum_constraint_margin_ratio = minimum_constraint_margin_ratio.min(margin / range);
    }
    if constraint_predictions.len() != input.side_constraints.len() {
        set(
            CandidateDispositionV1::Rejected,
            CandidateRejectionReasonV1::ConstraintPredictionMissing,
        );
    }
    if minimum_constraint_margin_ratio == f64::INFINITY {
        minimum_constraint_margin_ratio = 0.0;
    }

    let support_ratio = if candidate.historical_support_radius > 0.0 {
        candidate.historical_support_distance / candidate.historical_support_radius
    } else {
        f64::INFINITY
    };
    let base_score = if robust_target_risk.is_finite() && support_ratio.is_finite() {
        robust_target_risk + input.support_score_weight * support_ratio
    } else {
        f64::INFINITY
    };

    reasons.sort();
    reasons.dedup();
    Ok(InverseVmCandidateEvaluationV1 {
        candidate_id: candidate.candidate_id.clone(),
        disposition,
        reasons,
        robust_target_risk: robust_target_risk.is_finite().then_some(robust_target_risk),
        maximum_support_ratio: support_ratio.is_finite().then_some(support_ratio),
        minimum_constraint_margin_ratio: minimum_constraint_margin_ratio.is_finite().then_some(minimum_constraint_margin_ratio),
        base_score: (disposition == CandidateDispositionV1::Eligible).then_some(base_score),
        diversity_bonus: 0.0,
        selection_score: None,
        selected_rank: None,
        candidate_sha256: candidate.candidate_sha256.clone(),
    })
}

fn normalized_control_distance(
    left: &InverseVmCandidateV1,
    right: &InverseVmCandidateV1,
    bounds: &[RecipeControlBoundV1],
) -> Result<f64, InverseVmRankingErrorV1> {
    let mut squared = 0.0;
    for bound in bounds {
        let left_value = left.control_values.get(&bound.control_id).ok_or_else(|| {
            InverseVmRankingErrorV1("eligible candidate lost a declared control".into())
        })?;
        let right_value = right.control_values.get(&bound.control_id).ok_or_else(|| {
            InverseVmRankingErrorV1("eligible candidate lost a declared control".into())
        })?;
        let normalized = (left_value - right_value) / (bound.maximum - bound.minimum);
        squared += normalized * normalized;
    }
    Ok((squared / bounds.len() as f64).sqrt())
}

/// Filters candidates against robust target intervals, side constraints,
/// current model evidence and historical support, then returns a deterministic
/// diverse top-K shortlist. No recipe write or control authority is conferred.
pub fn rank_feasible_inverse_vm_candidates_v1(
    input: &InverseVmRankingInputV1,
) -> Result<InverseVmRankingReportV1, InverseVmRankingErrorV1> {
    validate_input(input)?;
    let input_sha256 = input_digest(input)?;

    let mut seen_ids = BTreeSet::new();
    let mut evaluations = Vec::with_capacity(input.candidates.len());
    let mut eligible = Vec::new();
    for candidate in &input.candidates {
        if !seen_ids.insert(candidate.candidate_id.as_str()) {
            return Err(InverseVmRankingErrorV1(format!(
                "duplicate candidate ID {}",
                candidate.candidate_id
            )));
        }
        let evaluation = evaluate_candidate(input, candidate)?;
        if evaluation.disposition == CandidateDispositionV1::Eligible {
            eligible.push(FeasibleCandidate {
                evaluation_index: evaluations.len(),
                candidate: candidate.clone(),
                base_score: evaluation.base_score.unwrap_or(f64::INFINITY),
            });
        }
        evaluations.push(evaluation);
    }

    eligible.sort_by(|a, b| {
        a.base_score
            .total_cmp(&b.base_score)
            .then_with(|| a.candidate.candidate_id.cmp(&b.candidate.candidate_id))
    });

    let mut selected: Vec<usize> = Vec::new();
    let mut remaining: BTreeSet<usize> = (0..eligible.len()).collect();
    while selected.len() < input.requested_top_k && !remaining.is_empty() {
        let mut best: Option<(usize, f64, f64)> = None;
        for index in &remaining {
            let item = &eligible[*index];
            let min_distance = selected
                .iter()
                .map(|selected_index| {
                    normalized_control_distance(
                        &item.candidate,
                        &eligible[*selected_index].candidate,
                        &input.control_bounds,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .fold(1.0_f64, f64::min);
            let diversity_bonus = if selected.is_empty() {
                0.0
            } else {
                input.diversity_weight
                    * (min_distance / input.minimum_diversity_distance).min(1.0)
            };
            let score = item.base_score - diversity_bonus;
            match best {
                None => best = Some((*index, score, diversity_bonus)),
                Some((best_index, best_score, _))
                    if score < best_score
                        || (score == best_score
                            && item.candidate.candidate_id
                                < eligible[best_index].candidate.candidate_id) =>
                {
                    best = Some((*index, score, diversity_bonus));
                }
                _ => {}
            }
        }
        let (best_index, selection_score, diversity_bonus) =
            best.expect("remaining candidate set is non-empty");
        remaining.remove(&best_index);
        let evaluation_index = eligible[best_index].evaluation_index;
        let evaluation = &mut evaluations[evaluation_index];
        evaluation.diversity_bonus = diversity_bonus;
        evaluation.selection_score = Some(selection_score);
        evaluation.selected_rank = Some(selected.len() + 1);
        selected.push(best_index);
    }

    let selected_candidate_ids = selected
        .iter()
        .map(|index| eligible[*index].candidate.candidate_id.clone())
        .collect::<Vec<_>>();
    evaluations.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));

    let eligible_count = evaluations
        .iter()
        .filter(|item| item.disposition == CandidateDispositionV1::Eligible)
        .count();
    let unknown_count = evaluations
        .iter()
        .filter(|item| item.disposition == CandidateDispositionV1::Unknown)
        .count();
    let unsupported_count = evaluations
        .iter()
        .filter(|item| item.disposition == CandidateDispositionV1::Unsupported)
        .count();
    let infeasible_count = evaluations
        .iter()
        .filter(|item| item.disposition == CandidateDispositionV1::Infeasible)
        .count();
    let rejected_count = evaluations
        .iter()
        .filter(|item| item.disposition == CandidateDispositionV1::Rejected)
        .count();

    let mut report = InverseVmRankingReportV1 {
        schema: INVERSE_VM_REPORT_SCHEMA_V1.into(),
        ranking_id: input.ranking_id.clone(),
        input_sha256,
        selected_candidate_ids,
        evaluations,
        eligible_count,
        unknown_count,
        unsupported_count,
        infeasible_count,
        rejected_count,
        requested_top_k: input.requested_top_k,
        top_k_filled: selected.len() == input.requested_top_k,
        source_authenticity_proven: false,
        physical_success_proven: false,
        recipe_write_executed: false,
        apc_r2r_authority: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn input(candidates: Vec<InverseVmCandidateV1>) -> InverseVmRankingInputV1 {
        InverseVmRankingInputV1 {
            schema: INVERSE_VM_RANKING_SCHEMA_V1.into(),
            ranking_id: "epi-topk-001".into(),
            recipe_schema_sha256: digest('a'),
            historical_support_manifest_sha256: digest('b'),
            knowledge_cutoff_unix_seconds: 1_900_000_000,
            confidence_multiplier: 2.0,
            minimum_support_count: 20,
            requested_top_k: 2,
            support_score_weight: 0.2,
            diversity_weight: 0.1,
            minimum_diversity_distance: 0.2,
            control_bounds: vec![
                RecipeControlBoundV1 { control_id: "pressure".into(), unit: "kPa".into(), minimum: 10.0, maximum: 20.0 },
                RecipeControlBoundV1 { control_id: "power".into(), unit: "W".into(), minimum: 100.0, maximum: 200.0 },
            ],
            target_specs: vec![
                InverseVmTargetSpecV1 { target_id: "thickness".into(), unit: "nm".into(), desired_value: 100.0, tolerance: 2.0 },
            ],
            side_constraints: vec![
                InverseVmSideConstraintV1 { constraint_id: "uniformity".into(), unit: "%".into(), minimum: 0.0, maximum: 5.0 },
            ],
            candidates,
        }
    }

    fn candidate(id: &str, pressure: f64, power: f64, target: f64, uncertainty: f64, support_distance: f64) -> InverseVmCandidateV1 {
        let mut candidate = InverseVmCandidateV1 {
            schema: INVERSE_VM_CANDIDATE_SCHEMA_V1.into(),
            candidate_id: id.into(),
            recipe_schema_sha256: digest('a'),
            control_values: BTreeMap::from([
                ("pressure".into(), pressure),
                ("power".into(), power),
            ]),
            model_artifact_sha256: digest('c'),
            forward_prediction_receipt_sha256: digest('d'),
            model_evidence_receipt_sha256: digest('e'),
            model_evidence_status: CandidateEvidenceStatusV1::CurrentEligible,
            model_evidence_available_at_unix_seconds: 1_800_000_000,
            support_manifest_sha256: digest('b'),
            support_receipt_sha256: digest('f'),
            historical_support_distance: support_distance,
            historical_support_radius: 1.0,
            historical_support_count: 100,
            target_predictions: vec![
                CandidateTargetPredictionV1 {
                    target_id: "thickness".into(),
                    unit: "nm".into(),
                    predicted_value: target,
                    standard_uncertainty: uncertainty,
                },
            ],
            constraint_predictions: vec![
                CandidateConstraintPredictionV1 {
                    constraint_id: "uniformity".into(),
                    unit: "%".into(),
                    predicted_value: 2.0,
                    standard_uncertainty: 0.2,
                },
            ],
            candidate_sha256: String::new(),
        };
        candidate.candidate_sha256 = seal_inverse_vm_candidate_v1(&candidate).unwrap();
        candidate
    }

    #[test]
    fn robust_target_and_side_constraint_intervals_gate_candidates() {
        let good = candidate("good", 12.0, 120.0, 100.5, 0.2, 0.1);
        let target_bad = candidate("target-uncertain", 13.0, 130.0, 101.0, 0.8, 0.1);
        let mut constraint_bad = candidate("constraint-bad", 14.0, 140.0, 100.5, 0.2, 0.1);
        constraint_bad.constraint_predictions[0].predicted_value = 4.8;
        constraint_bad.candidate_sha256 = seal_inverse_vm_candidate_v1(&constraint_bad).unwrap();

        let report = rank_feasible_inverse_vm_candidates_v1(&input(vec![good, target_bad, constraint_bad])).unwrap();
        assert_eq!(report.eligible_count, 1);
        assert_eq!(report.infeasible_count, 2);
        assert_eq!(report.selected_candidate_ids, vec!["good"]);
    }

    #[test]
    fn unsupported_or_unknown_evidence_never_enters_topk() {
        let supported = candidate("supported", 12.0, 120.0, 100.0, 0.2, 0.1);
        let outside = candidate("outside", 13.0, 130.0, 100.0, 0.2, 1.1);
        let mut unknown = candidate("unknown", 14.0, 140.0, 100.0, 0.2, 0.1);
        unknown.model_evidence_status = CandidateEvidenceStatusV1::Unknown;
        unknown.candidate_sha256 = seal_inverse_vm_candidate_v1(&unknown).unwrap();
        let report = rank_feasible_inverse_vm_candidates_v1(&input(vec![supported, outside, unknown])).unwrap();
        assert_eq!(report.selected_candidate_ids, vec!["supported"]);
        assert_eq!(report.unsupported_count, 1);
        assert_eq!(report.unknown_count, 1);
    }

    #[test]
    fn historical_support_count_below_floor_is_unknown() {
        let mut item = candidate("low-support", 12.0, 120.0, 100.0, 0.2, 0.1);
        item.historical_support_count = 3;
        item.candidate_sha256 = seal_inverse_vm_candidate_v1(&item).unwrap();
        let report = rank_feasible_inverse_vm_candidates_v1(&input(vec![item])).unwrap();
        assert_eq!(report.unknown_count, 1);
        assert!(report.selected_candidate_ids.is_empty());
    }

    #[test]
    fn model_evidence_unavailable_at_cutoff_is_unknown() {
        let mut item = candidate("future-evidence", 12.0, 120.0, 100.0, 0.2, 0.1);
        item.model_evidence_available_at_unix_seconds = 1_950_000_000;
        item.candidate_sha256 = seal_inverse_vm_candidate_v1(&item).unwrap();
        let report = rank_feasible_inverse_vm_candidates_v1(&input(vec![item])).unwrap();
        assert_eq!(report.unknown_count, 1);
    }

    #[test]
    fn control_outside_declared_recipe_domain_is_infeasible() {
        let mut item = candidate("out-of-range", 25.0, 120.0, 100.0, 0.2, 0.1);
        item.candidate_sha256 = seal_inverse_vm_candidate_v1(&item).unwrap();
        let report = rank_feasible_inverse_vm_candidates_v1(&input(vec![item])).unwrap();
        assert_eq!(report.infeasible_count, 1);
        assert!(report.selected_candidate_ids.is_empty());
    }

    #[test]
    fn digest_tampering_rejects_candidate() {
        let mut item = candidate("tampered", 12.0, 120.0, 100.0, 0.2, 0.1);
        item.control_values.insert("pressure".into(), 17.0);
        let report = rank_feasible_inverse_vm_candidates_v1(&input(vec![item])).unwrap();
        assert_eq!(report.rejected_count, 1);
        assert!(report.selected_candidate_ids.is_empty());
    }

    #[test]
    fn diversity_changes_topk_without_changing_feasibility_scores() {
        let a = candidate("a", 12.0, 120.0, 100.0, 0.2, 0.1);
        let b = candidate("b", 12.1, 121.0, 100.0, 0.2, 0.1);
        let c = candidate("c", 19.0, 190.0, 100.2, 0.2, 0.1);
        let mut value = input(vec![a, b, c]);
        value.diversity_weight = 0.5;
        let report = rank_feasible_inverse_vm_candidates_v1(&value).unwrap();
        assert_eq!(report.eligible_count, 3);
        assert_eq!(report.selected_candidate_ids[0], "a");
        assert_eq!(report.selected_candidate_ids[1], "c");
        let b_eval = report.evaluations.iter().find(|item| item.candidate_id == "b").unwrap();
        assert_eq!(b_eval.disposition, CandidateDispositionV1::Eligible);
        assert_eq!(b_eval.selected_rank, None);
    }

    #[test]
    fn candidate_and_input_order_do_not_change_ranking_digest() {
        let a = candidate("a", 12.0, 120.0, 100.0, 0.2, 0.1);
        let b = candidate("b", 18.0, 180.0, 100.0, 0.2, 0.2);
        let first = input(vec![a.clone(), b.clone()]);
        let second = input(vec![b, a]);
        let first_report = rank_feasible_inverse_vm_candidates_v1(&first).unwrap();
        let second_report = rank_feasible_inverse_vm_candidates_v1(&second).unwrap();
        assert_eq!(first_report.report_sha256, second_report.report_sha256);
    }

    #[test]
    fn topk_is_human_review_only_and_reports_partial_fill() {
        let mut data = input(vec![candidate("one", 12.0, 120.0, 100.0, 0.2, 0.1)]);
        data.requested_top_k = 3;
        let report = rank_feasible_inverse_vm_candidates_v1(&data).unwrap();
        assert!(!report.top_k_filled);
        assert_eq!(report.selected_candidate_ids, vec!["one"]);
        assert!(!report.recipe_write_executed);
        assert!(!report.apc_r2r_authority);
        assert!(!report.equipment_control_authority);
    }
}
