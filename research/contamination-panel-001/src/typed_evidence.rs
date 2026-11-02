//! Typed evidence ledger and abstaining detector comparison.
//!
//! Different contamination methods measure different units and make different
//! assumptions. This module preserves evidence kind, unit of analysis, model,
//! split, scope and provenance. It computes pairwise rank agreement only when
//! two detector batches are compatible on the same unit universe.
//!
//! It deliberately emits no scalar or majority-vote contamination verdict.

use crate::hash_domain;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const EVIDENCE_BUNDLE_SCHEMA: &str = "symthaea.typed_evidence_bundle.v1";
pub const EVIDENCE_REPORT_SCHEMA: &str = "symthaea.typed_evidence_report.v1";
pub const MAX_EVIDENCE_BATCHES: usize = 256;
pub const MAX_EVIDENCE_UNITS: usize = 1_000_000;
pub const MIN_PAIRWISE_SHARED_UNITS: usize = 3;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceKindV1 {
    DetectorScore,
    CausalEffect,
    ExecutionAttestation,
    AdjudicationDecision,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceUnitV1 {
    Item,
    Dataset,
    SourceFamily,
    ModelCheckpoint,
    TrainingIntervention,
    EvaluationRun,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AccessTierV1 {
    BlackBox,
    GrayBox,
    WhiteBox,
    ExecutionTrace,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScoreDirectionV1 {
    HigherIsPositive,
    LowerIsPositive,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservationStatusV1 {
    Observed,
    Missing,
    Invalid,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AssumptionStatusV1 {
    Supported,
    Violated,
    Unknown,
    NotApplicable,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStatusV1 {
    Calibrated,
    NotCalibrated,
    Unknown,
    Invalid,
    NotApplicable,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceObservationV1 {
    pub unit_id: String,
    pub status: ObservationStatusV1,
    pub value: Option<f64>,
    pub reason_code: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AssumptionAssessmentV1 {
    pub assumption_id: String,
    pub status: AssumptionStatusV1,
    pub evidence_sha256: Option<String>,
    pub note: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationEvidenceV1 {
    pub status: CalibrationStatusV1,
    pub calibration_manifest_sha256: Option<String>,
    /// Permille: 10 = 1%, 50 = 5%, 100 = 10%.
    pub target_fpr_permille: Option<u16>,
    pub threshold_oriented: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBatchV1 {
    pub batch_id: String,
    pub method_id: String,
    pub evidence_kind: EvidenceKindV1,
    pub unit_type: EvidenceUnitV1,
    pub access_tier: AccessTierV1,
    pub model_checkpoint_sha256: String,
    pub metric_definition_sha256: String,
    pub scope_manifest_sha256: String,
    pub evaluation_split_sha256: String,
    pub unit_manifest_sha256: String,
    pub expected_unit_count: u64,
    pub score_direction: Option<ScoreDirectionV1>,
    pub calibration: CalibrationEvidenceV1,
    pub declared_independent_authentication: bool,
    pub authentication_receipt_sha256: Option<String>,
    pub assumptions: Vec<AssumptionAssessmentV1>,
    /// Must include every expected unit, including explicit Missing/Invalid rows.
    pub observations: Vec<EvidenceObservationV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TypedEvidenceBundleV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub batches: Vec<EvidenceBatchV1>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PairwiseComparisonStatusV1 {
    Comparable,
    DifferentModelCheckpoint,
    DifferentEvaluationSplit,
    DifferentScopeManifest,
    DifferentUnitManifest,
    InsufficientSharedUnits,
    DegenerateRanks,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PairwiseEvidenceComparisonV1 {
    pub left_batch_id: String,
    pub right_batch_id: String,
    pub status: PairwiseComparisonStatusV1,
    pub left_observed_count: u64,
    pub right_observed_count: u64,
    pub shared_observed_unit_count: u64,
    pub union_observed_unit_count: u64,
    pub shared_coverage_permille: f64,
    pub spearman_rank_correlation: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceBatchSummaryV1 {
    pub batch_id: String,
    pub method_id: String,
    pub evidence_kind: EvidenceKindV1,
    pub unit_type: EvidenceUnitV1,
    pub access_tier: AccessTierV1,
    pub expected_unit_count: u64,
    pub observed_unit_count: u64,
    pub missing_unit_count: u64,
    pub invalid_unit_count: u64,
    pub coverage_permille: f64,
    pub calibration_status: CalibrationStatusV1,
    pub required_assumptions_supported: bool,
    pub unsupported_critical_assumptions: Vec<String>,
    pub declared_independent_authentication: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceScopeReportV1 {
    pub evidence_kind: EvidenceKindV1,
    pub unit_type: EvidenceUnitV1,
    pub batch_ids: Vec<String>,
    pub pairwise_comparisons: Vec<PairwiseEvidenceComparisonV1>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TypedEvidenceReportV1 {
    pub schema: String,
    pub study_id: String,
    pub benchmark_id: String,
    pub benchmark_version: String,
    pub bundle_sha256: String,
    pub batch_summaries: Vec<EvidenceBatchSummaryV1>,
    pub scope_reports: Vec<EvidenceScopeReportV1>,
    pub cross_scope_pair_count_not_compared: u64,
    pub scalar_fusion_permitted: bool,
    pub fusion_recommendation: String,
    pub abstain_from_confirmatory_claims: bool,
    pub independently_authenticated: bool,
    pub confirmatory_eligible: bool,
    pub analysis_status: String,
    pub issues: Vec<String>,
    pub claim_boundary: String,
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_digest(value: &str, label: &str) -> Result<(), String> {
    if valid_digest(value) {
        Ok(())
    } else {
        Err(format!("{label} must be a lowercase SHA-256 digest"))
    }
}

fn valid_id(value: &str, maximum: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= maximum
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._:-".contains(&byte))
}

pub fn compute_evidence_unit_manifest_sha256(ids: &[String]) -> Result<String, String> {
    if ids.is_empty() || ids.iter().any(|id| !valid_id(id, 128)) {
        return Err("evidence unit IDs must be nonempty valid identifiers".to_owned());
    }
    let mut sorted = ids.to_vec();
    sorted.sort();
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("evidence unit IDs must be unique".to_owned());
    }
    let bytes = serde_json::to_vec(&sorted)
        .map_err(|error| format!("cannot serialize evidence unit IDs: {error}"))?;
    Ok(hash_domain(b"typed-evidence-unit-manifest.v1", &bytes))
}

fn required_assumptions(kind: EvidenceKindV1) -> &'static [&'static str] {
    match kind {
        EvidenceKindV1::DetectorScore => &[
            "score_semantics_declared",
            "evaluation_split_frozen",
            "target_unit_manifest_bound",
        ],
        EvidenceKindV1::CausalEffect => &[
            "randomization_manifest_bound",
            "independent_training_pair_unit",
            "outcome_denominator_reconciled",
        ],
        EvidenceKindV1::ExecutionAttestation => &[
            "independent_monitoring_receipt",
            "boundary_coverage_declared",
            "receipt_authenticity_verified",
        ],
        EvidenceKindV1::AdjudicationDecision => &[
            "candidate_denominator_frozen",
            "reviewer_coverage_declared",
            "decision_records_reconciled",
        ],
    }
}

fn validate_batch(batch: &EvidenceBatchV1) -> Result<EvidenceBatchSummaryV1, String> {
    for (value, label, max) in [
        (&batch.batch_id, "batch_id", 128),
        (&batch.method_id, "method_id", 128),
    ] {
        if !valid_id(value, max) {
            return Err(format!("{label} must be a valid bounded identifier"));
        }
    }
    for (digest, label) in [
        (&batch.model_checkpoint_sha256, "model_checkpoint_sha256"),
        (&batch.metric_definition_sha256, "metric_definition_sha256"),
        (&batch.scope_manifest_sha256, "scope_manifest_sha256"),
        (&batch.evaluation_split_sha256, "evaluation_split_sha256"),
        (&batch.unit_manifest_sha256, "unit_manifest_sha256"),
    ] {
        require_digest(digest, label)?;
    }
    if batch.expected_unit_count == 0
        || batch.expected_unit_count as usize > MAX_EVIDENCE_UNITS
        || batch.observations.len() as u64 != batch.expected_unit_count
    {
        return Err("every expected evidence unit must have exactly one observation row".to_owned());
    }
    let unit_ids: Vec<String> = batch.observations.iter().map(|observation| observation.unit_id.clone()).collect();
    let expected_manifest = compute_evidence_unit_manifest_sha256(&unit_ids)?;
    if expected_manifest != batch.unit_manifest_sha256 {
        return Err("evidence unit manifest digest does not match observation IDs".to_owned());
    }

    match batch.evidence_kind {
        EvidenceKindV1::DetectorScore => {
            if batch.score_direction.is_none() {
                return Err("detector score batches require an explicit score direction".to_owned());
            }
            match batch.calibration.status {
                CalibrationStatusV1::Calibrated => {
                    let digest = batch
                        .calibration
                        .calibration_manifest_sha256
                        .as_deref()
                        .ok_or_else(|| "calibrated detector evidence requires a calibration manifest digest".to_owned())?;
                    require_digest(digest, "calibration_manifest_sha256")?;
                    if batch.calibration.target_fpr_permille.is_none_or(|value| !(1..=100).contains(&value)) {
                        return Err("calibrated detector target FPR must be in 1..=100 permille".to_owned());
                    }
                    if batch.calibration.threshold_oriented.is_none_or(|value| !value.is_finite()) {
                        return Err("calibrated detector evidence requires a finite oriented threshold".to_owned());
                    }
                }
                CalibrationStatusV1::NotApplicable => {
                    return Err("detector calibration cannot be marked not_applicable".to_owned());
                }
                CalibrationStatusV1::NotCalibrated
                | CalibrationStatusV1::Unknown
                | CalibrationStatusV1::Invalid => {}
            }
        }
        _ => {
            if batch.score_direction.is_some()
                || batch.calibration.status != CalibrationStatusV1::NotApplicable
                || batch.calibration.calibration_manifest_sha256.is_some()
                || batch.calibration.target_fpr_permille.is_some()
                || batch.calibration.threshold_oriented.is_some()
            {
                return Err("non-detector evidence must not claim detector direction or calibration fields".to_owned());
            }
        }
    }

    if batch.declared_independent_authentication {
        let receipt = batch
            .authentication_receipt_sha256
            .as_deref()
            .ok_or_else(|| "declared independent authentication requires a receipt digest".to_owned())?;
        require_digest(receipt, "authentication_receipt_sha256")?;
    } else if batch.authentication_receipt_sha256.is_some() {
        return Err("authentication receipt digest cannot be attached when independent authentication is not declared".to_owned());
    }

    let mut assumptions = BTreeMap::new();
    for assessment in &batch.assumptions {
        if !valid_id(&assessment.assumption_id, 128) {
            return Err("assumption IDs must be valid bounded identifiers".to_owned());
        }
        if assumptions.insert(assessment.assumption_id.as_str(), assessment).is_some() {
            return Err("assumption IDs must be unique within a batch".to_owned());
        }
        if let Some(digest) = assessment.evidence_sha256.as_deref() {
            require_digest(digest, "assumption evidence digest")?;
        }
    }

    let mut unsupported = Vec::new();
    for assumption in required_assumptions(batch.evidence_kind) {
        match assumptions.get(assumption) {
            Some(assessment)
                if assessment.status == AssumptionStatusV1::Supported
                    && assessment.evidence_sha256.is_some() => {}
            Some(_) => unsupported.push((*assumption).to_owned()),
            None => unsupported.push((*assumption).to_owned()),
        }
    }
    for assessment in &batch.assumptions {
        if assessment.status == AssumptionStatusV1::Violated
            || assessment.status == AssumptionStatusV1::Unknown
            || assessment.status == AssumptionStatusV1::NotApplicable && required_assumptions(batch.evidence_kind).contains(&assessment.assumption_id.as_str())
        {
            if !unsupported.contains(&assessment.assumption_id) {
                unsupported.push(assessment.assumption_id.clone());
            }
        }
    }

    let mut ids = BTreeSet::new();
    let mut observed = 0_u64;
    let mut missing = 0_u64;
    let mut invalid = 0_u64;
    for observation in &batch.observations {
        if !valid_id(&observation.unit_id, 128) || !ids.insert(observation.unit_id.as_str()) {
            return Err("evidence observation unit IDs must be valid and unique".to_owned());
        }
        match observation.status {
            ObservationStatusV1::Observed => {
                if observation.value.is_none_or(|value| !value.is_finite())
                    || observation.reason_code.is_some()
                {
                    return Err("observed evidence requires a finite value and no failure reason".to_owned());
                }
                observed += 1;
            }
            ObservationStatusV1::Missing => {
                if observation.value.is_some() || !observation.reason_code.as_deref().is_some_and(|value| valid_id(value, 128)) {
                    return Err("missing evidence requires no value and a valid reason code".to_owned());
                }
                missing += 1;
            }
            ObservationStatusV1::Invalid => {
                if observation.value.is_some() || !observation.reason_code.as_deref().is_some_and(|value| valid_id(value, 128)) {
                    return Err("invalid evidence requires no value and a valid reason code".to_owned());
                }
                invalid += 1;
            }
        }
    }

    Ok(EvidenceBatchSummaryV1 {
        batch_id: batch.batch_id.clone(),
        method_id: batch.method_id.clone(),
        evidence_kind: batch.evidence_kind,
        unit_type: batch.unit_type,
        access_tier: batch.access_tier,
        expected_unit_count: batch.expected_unit_count,
        observed_unit_count: observed,
        missing_unit_count: missing,
        invalid_unit_count: invalid,
        coverage_permille: observed as f64 * 1_000.0 / batch.expected_unit_count as f64,
        calibration_status: batch.calibration.status,
        required_assumptions_supported: unsupported.is_empty(),
        unsupported_critical_assumptions: unsupported,
        declared_independent_authentication: batch.declared_independent_authentication,
    })
}

fn oriented_value(value: f64, direction: ScoreDirectionV1) -> f64 {
    match direction {
        ScoreDirectionV1::HigherIsPositive => value,
        ScoreDirectionV1::LowerIsPositive => -value,
    }
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut ordered: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    ordered.sort_by(|left, right| left.1.total_cmp(&right.1));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < ordered.len() {
        let mut end = start + 1;
        while end < ordered.len() && ordered[end].1 == ordered[start].1 {
            end += 1;
        }
        let average_rank = ((start + 1 + end) as f64) / 2.0;
        for (original_index, _) in &ordered[start..end] {
            ranks[*original_index] = average_rank;
        }
        start = end;
    }
    ranks
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() < MIN_PAIRWISE_SHARED_UNITS {
        return None;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut numerator = 0.0;
    let mut left_ss = 0.0;
    let mut right_ss = 0.0;
    for (left_value, right_value) in left.iter().zip(right) {
        let left_delta = left_value - left_mean;
        let right_delta = right_value - right_mean;
        numerator += left_delta * right_delta;
        left_ss += left_delta * left_delta;
        right_ss += right_delta * right_delta;
    }
    if left_ss <= f64::EPSILON || right_ss <= f64::EPSILON {
        None
    } else {
        Some((numerator / (left_ss * right_ss).sqrt()).clamp(-1.0, 1.0))
    }
}

fn pairwise_comparison(
    left: &EvidenceBatchV1,
    right: &EvidenceBatchV1,
) -> PairwiseEvidenceComparisonV1 {
    let mut status = PairwiseComparisonStatusV1::Comparable;
    if left.model_checkpoint_sha256 != right.model_checkpoint_sha256 {
        status = PairwiseComparisonStatusV1::DifferentModelCheckpoint;
    } else if left.evaluation_split_sha256 != right.evaluation_split_sha256 {
        status = PairwiseComparisonStatusV1::DifferentEvaluationSplit;
    } else if left.scope_manifest_sha256 != right.scope_manifest_sha256 {
        status = PairwiseComparisonStatusV1::DifferentScopeManifest;
    } else if left.unit_manifest_sha256 != right.unit_manifest_sha256 {
        status = PairwiseComparisonStatusV1::DifferentUnitManifest;
    }

    let left_observed: BTreeMap<&str, f64> = left
        .observations
        .iter()
        .filter_map(|observation| {
            (observation.status == ObservationStatusV1::Observed)
                .then(|| (observation.unit_id.as_str(), observation.value.unwrap_or(f64::NAN)))
        })
        .collect();
    let right_observed: BTreeMap<&str, f64> = right
        .observations
        .iter()
        .filter_map(|observation| {
            (observation.status == ObservationStatusV1::Observed)
                .then(|| (observation.unit_id.as_str(), observation.value.unwrap_or(f64::NAN)))
        })
        .collect();
    let shared_ids: Vec<&str> = left_observed
        .keys()
        .filter(|unit_id| right_observed.contains_key(**unit_id))
        .copied()
        .collect();
    let union_observed = left_observed.len() + right_observed.len() - shared_ids.len();
    let shared_coverage = if union_observed == 0 {
        0.0
    } else {
        shared_ids.len() as f64 * 1_000.0 / union_observed as f64
    };

    let mut spearman = None;
    if status == PairwiseComparisonStatusV1::Comparable {
        if shared_ids.len() < MIN_PAIRWISE_SHARED_UNITS {
            status = PairwiseComparisonStatusV1::InsufficientSharedUnits;
        } else {
            let left_direction = left.score_direction.unwrap_or(ScoreDirectionV1::HigherIsPositive);
            let right_direction = right.score_direction.unwrap_or(ScoreDirectionV1::HigherIsPositive);
            let left_values: Vec<f64> = shared_ids
                .iter()
                .map(|unit_id| oriented_value(left_observed[unit_id], left_direction))
                .collect();
            let right_values: Vec<f64> = shared_ids
                .iter()
                .map(|unit_id| oriented_value(right_observed[unit_id], right_direction))
                .collect();
            let left_ranks = average_ranks(&left_values);
            let right_ranks = average_ranks(&right_values);
            spearman = pearson_correlation(&left_ranks, &right_ranks);
            if spearman.is_none() {
                status = PairwiseComparisonStatusV1::DegenerateRanks;
            }
        }
    }
    PairwiseEvidenceComparisonV1 {
        left_batch_id: left.batch_id.clone(),
        right_batch_id: right.batch_id.clone(),
        status,
        left_observed_count: left_observed.len() as u64,
        right_observed_count: right_observed.len() as u64,
        shared_observed_unit_count: shared_ids.len() as u64,
        union_observed_unit_count: union_observed as u64,
        shared_coverage_permille: shared_coverage,
        spearman_rank_correlation: spearman,
    }
}

/// Build a typed evidence report. Only detector batches with identical
/// checkpoint, split, scope manifest and unit manifest receive pairwise
/// rank-comparison statistics. No cross-unit or scalar fusion is performed.
pub fn analyze_typed_evidence_bundle(
    bundle: &TypedEvidenceBundleV1,
) -> Result<TypedEvidenceReportV1, String> {
    if bundle.schema != EVIDENCE_BUNDLE_SCHEMA {
        return Err(format!("bundle schema must be {EVIDENCE_BUNDLE_SCHEMA}"));
    }
    if !valid_id(&bundle.study_id, 128)
        || !valid_id(&bundle.benchmark_id, 128)
        || bundle.benchmark_version.trim().is_empty()
        || bundle.benchmark_version.len() > 64
    {
        return Err("study and benchmark identities must be valid bounded identifiers".to_owned());
    }
    if bundle.batches.is_empty() || bundle.batches.len() > MAX_EVIDENCE_BATCHES {
        return Err(format!("evidence batch count must be in 1..={MAX_EVIDENCE_BATCHES}"));
    }
    let mut batch_ids = BTreeSet::new();
    let mut summaries = Vec::with_capacity(bundle.batches.len());
    let mut scope_groups = BTreeMap::<(EvidenceKindV1, EvidenceUnitV1), Vec<&EvidenceBatchV1>>::new();
    let mut issues = Vec::new();
    let mut abstain = false;
    for batch in &bundle.batches {
        if !batch_ids.insert(batch.batch_id.as_str()) {
            return Err("evidence batch IDs must be unique".to_owned());
        }
        let summary = validate_batch(batch)?;
        if !summary.unsupported_critical_assumptions.is_empty() {
            abstain = true;
            issues.push(format!(
                "batch {} has unsupported required assumptions: {}",
                batch.batch_id,
                summary.unsupported_critical_assumptions.join(", ")
            ));
        }
        if summary.missing_unit_count > 0 || summary.invalid_unit_count > 0 {
            abstain = true;
            issues.push(format!(
                "batch {} retains {} missing and {} invalid units in its denominator",
                batch.batch_id,
                summary.missing_unit_count,
                summary.invalid_unit_count
            ));
        }
        if batch.evidence_kind == EvidenceKindV1::DetectorScore
            && batch.calibration.status != CalibrationStatusV1::Calibrated
        {
            abstain = true;
            issues.push(format!(
                "detector batch {} is not calibrated",
                batch.batch_id
            ));
        }
        if !batch.declared_independent_authentication {
            issues.push(format!(
                "batch {} has no independently authenticated execution receipt",
                batch.batch_id
            ));
            abstain = true;
        }
        summaries.push(summary);
        scope_groups
            .entry((batch.evidence_kind, batch.unit_type))
            .or_default()
            .push(batch);
    }

    let mut scope_reports = Vec::new();
    for ((kind, unit_type), batches) in scope_groups {
        let mut comparisons = Vec::new();
        if kind == EvidenceKindV1::DetectorScore {
            for left_index in 0..batches.len() {
                for right_index in left_index + 1..batches.len() {
                    comparisons.push(pairwise_comparison(
                        batches[left_index],
                        batches[right_index],
                    ));
                }
            }
        }
        scope_reports.push(EvidenceScopeReportV1 {
            evidence_kind: kind,
            unit_type,
            batch_ids: batches.iter().map(|batch| batch.batch_id.clone()).collect(),
            pairwise_comparisons: comparisons,
        });
    }

    let total_pairs = bundle.batches.len() * (bundle.batches.len() - 1) / 2;
    let mut comparable_scope_pairs = 0_usize;
    for scope in &scope_reports {
        comparable_scope_pairs += scope.batch_ids.len() * (scope.batch_ids.len() - 1) / 2;
    }
    let cross_scope_pair_count_not_compared = total_pairs.saturating_sub(comparable_scope_pairs) as u64;
    let bundle_bytes = serde_json::to_vec(bundle)
        .map_err(|error| format!("cannot serialize typed evidence bundle: {error}"))?;
    let bundle_sha256 = hash_domain(b"typed-evidence-bundle.v1", &bundle_bytes);

    if !issues.is_empty() {
        issues.push("all results remain exploratory until an independent trust-root and execution-custody verifier authenticates the supplied receipts".to_owned());
    }
    if issues.is_empty() {
        issues.push("all declared required assumptions are supported and detector batches are calibrated; no scalar fusion is performed".to_owned());
    }

    Ok(TypedEvidenceReportV1 {
        schema: EVIDENCE_REPORT_SCHEMA.to_owned(),
        study_id: bundle.study_id.clone(),
        benchmark_id: bundle.benchmark_id.clone(),
        benchmark_version: bundle.benchmark_version.clone(),
        bundle_sha256,
        batch_summaries: summaries,
        scope_reports,
        cross_scope_pair_count_not_compared,
        scalar_fusion_permitted: false,
        fusion_recommendation: "structured_evidence_report_only_no_scalar_or_majority_vote".to_owned(),
        abstain_from_confirmatory_claims: abstain,
        independently_authenticated: false,
        confirmatory_eligible: false,
        analysis_status: if abstain {
            "abstain_from_confirmatory_claims".to_owned()
        } else {
            "exploratory_typed_evidence".to_owned()
        },
        issues,
        claim_boundary: "This module organizes heterogeneous evidence and computes tie-aware Spearman rank agreement only for detector batches sharing the exact unit type, model checkpoint, evaluation split, scope manifest and unit manifest. It does not fuse item-level and dataset-level metrics, detector scores and causal effects, or detector evidence and execution attestations. Pairwise agreement is not truth, majority consensus is not emitted, and no scalar contamination probability is produced. Declared authentication receipts are not independently verified by this module; reports remain unauthenticated and non-confirmatory.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat_n(byte, 64).collect()
    }

    fn required_assumptions(kind: EvidenceKindV1) -> Vec<AssumptionAssessmentV1> {
        required_assumptions(kind)
            .iter()
            .map(|assumption| AssumptionAssessmentV1 {
                assumption_id: (*assumption).to_owned(),
                status: AssumptionStatusV1::Supported,
                evidence_sha256: Some(digest('a')),
                note: "synthetic fixture".to_owned(),
            })
            .collect()
    }

    fn detector_batch(
        batch_id: &str,
        unit_type: EvidenceUnitV1,
        direction: ScoreDirectionV1,
        values: &[f64],
    ) -> EvidenceBatchV1 {
        let ids: Vec<String> = (0..values.len()).map(|index| format!("unit-{index}")).collect();
        EvidenceBatchV1 {
            batch_id: batch_id.to_owned(),
            method_id: batch_id.to_owned(),
            evidence_kind: EvidenceKindV1::DetectorScore,
            unit_type,
            access_tier: AccessTierV1::GrayBox,
            model_checkpoint_sha256: digest('b'),
            metric_definition_sha256: digest('c'),
            scope_manifest_sha256: digest('d'),
            evaluation_split_sha256: digest('e'),
            unit_manifest_sha256: compute_evidence_unit_manifest_sha256(&ids).unwrap(),
            expected_unit_count: ids.len() as u64,
            score_direction: Some(direction),
            calibration: CalibrationEvidenceV1 {
                status: CalibrationStatusV1::Calibrated,
                calibration_manifest_sha256: Some(digest('f')),
                target_fpr_permille: Some(10),
                threshold_oriented: Some(0.5),
            },
            declared_independent_authentication: true,
            authentication_receipt_sha256: Some(digest('1')),
            assumptions: required_assumptions(EvidenceKindV1::DetectorScore),
            observations: ids
                .into_iter()
                .zip(values)
                .map(|(unit_id, value)| EvidenceObservationV1 {
                    unit_id,
                    status: ObservationStatusV1::Observed,
                    value: Some(*value),
                    reason_code: None,
                })
                .collect(),
        }
    }

    fn bundle(batches: Vec<EvidenceBatchV1>) -> TypedEvidenceBundleV1 {
        TypedEvidenceBundleV1 {
            schema: EVIDENCE_BUNDLE_SCHEMA.to_owned(),
            study_id: "study-001".to_owned(),
            benchmark_id: "benchmark-001".to_owned(),
            benchmark_version: "v1".to_owned(),
            batches,
        }
    }

    #[test]
    fn compatible_detector_batches_get_tie_aware_spearman_comparison() {
        let report = analyze_typed_evidence_bundle(&bundle(vec![
            detector_batch(
                "method-a",
                EvidenceUnitV1::Item,
                ScoreDirectionV1::HigherIsPositive,
                &[0.1, 0.2, 0.8, 0.9],
            ),
            detector_batch(
                "method-b",
                EvidenceUnitV1::Item,
                ScoreDirectionV1::LowerIsPositive,
                &[0.9, 0.8, 0.2, 0.1],
            ),
        ]))
        .unwrap();
        let comparison = &report.scope_reports[0].pairwise_comparisons[0];
        assert_eq!(comparison.status, PairwiseComparisonStatusV1::Comparable);
        assert_eq!(comparison.shared_observed_unit_count, 4);
        assert!((comparison.spearman_rank_correlation.unwrap() - 1.0).abs() < 1.0e-12);
        assert!(!report.scalar_fusion_permitted);
        assert!(!report.confirmatory_eligible);
    }

    #[test]
    fn item_and_dataset_evidence_are_never_compared() {
        let report = analyze_typed_evidence_bundle(&bundle(vec![
            detector_batch(
                "item-method",
                EvidenceUnitV1::Item,
                ScoreDirectionV1::HigherIsPositive,
                &[0.1, 0.2, 0.8, 0.9],
            ),
            detector_batch(
                "dataset-method",
                EvidenceUnitV1::Dataset,
                ScoreDirectionV1::HigherIsPositive,
                &[0.1, 0.2, 0.8, 0.9],
            ),
        ]))
        .unwrap();
        assert_eq!(report.scope_reports.len(), 2);
        assert_eq!(report.cross_scope_pair_count_not_compared, 1);
        assert!(report.scope_reports.iter().all(|scope| scope.pairwise_comparisons.is_empty()));
    }

    #[test]
    fn scope_manifest_mismatch_prevents_pairwise_correlation() {
        let mut right = detector_batch(
            "method-b",
            EvidenceUnitV1::Item,
            ScoreDirectionV1::HigherIsPositive,
            &[0.1, 0.2, 0.8, 0.9],
        );
        right.scope_manifest_sha256 = digest('9');
        let report = analyze_typed_evidence_bundle(&bundle(vec![
            detector_batch(
                "method-a",
                EvidenceUnitV1::Item,
                ScoreDirectionV1::HigherIsPositive,
                &[0.1, 0.2, 0.8, 0.9],
            ),
            right,
        ]))
        .unwrap();
        let comparison = &report.scope_reports[0].pairwise_comparisons[0];
        assert_eq!(
            comparison.status,
            PairwiseComparisonStatusV1::DifferentScopeManifest
        );
        assert!(comparison.spearman_rank_correlation.is_none());
    }

    #[test]
    fn missing_observations_remain_in_denominator_and_force_abstention() {
        let mut batch = detector_batch(
            "method-a",
            EvidenceUnitV1::Item,
            ScoreDirectionV1::HigherIsPositive,
            &[0.1, 0.2, 0.8, 0.9],
        );
        batch.observations[3] = EvidenceObservationV1 {
            unit_id: "unit-3".to_owned(),
            status: ObservationStatusV1::Missing,
            value: None,
            reason_code: Some("backend_logprob_missing".to_owned()),
        };
        let report = analyze_typed_evidence_bundle(&bundle(vec![batch])).unwrap();
        assert_eq!(report.batch_summaries[0].expected_unit_count, 4);
        assert_eq!(report.batch_summaries[0].observed_unit_count, 3);
        assert_eq!(report.batch_summaries[0].missing_unit_count, 1);
        assert!(report.abstain_from_confirmatory_claims);
    }

    #[test]
    fn not_applicable_cannot_bypass_required_critical_assumption() {
        let mut batch = detector_batch(
            "method-a",
            EvidenceUnitV1::Item,
            ScoreDirectionV1::HigherIsPositive,
            &[0.1, 0.2, 0.8, 0.9],
        );
        batch.assumptions[0].status = AssumptionStatusV1::NotApplicable;
        let report = analyze_typed_evidence_bundle(&bundle(vec![batch])).unwrap();
        assert!(report.abstain_from_confirmatory_claims);
        assert!(!report.confirmatory_eligible);
    }
}
