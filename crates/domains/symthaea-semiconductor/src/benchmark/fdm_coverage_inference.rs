// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Paper-specific short-memory realized-coverage inference under functional
//! dependence. This module is separate from the Newey-West/HAC baseline.
//!
//! The implementation consumes a process-model-provided upper envelope for
//! the indicator functional-dependence coefficients. These coefficients are
//! properties of the data-generating process and cannot be proven from one
//! observed wafer stream. The report therefore says assumptions are declared,
//! never proven. The fixed predictor, calibration window and adjacent test
//! window must be frozen and independently identified.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

use super::gated_streaming_adapter::StreamingPredictionReceiptV1;
use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const FDM_COVERAGE_SCHEMA_V1: &str = "luminous.semi-bench.fdm-coverage.v1";
pub const FDM_COVERAGE_SCORE_SCHEMA_V1: &str = "luminous.semi-bench.fdm-coverage-score.v1";
pub const FDM_COVERAGE_ASSUMPTION_SCHEMA_V1: &str =
    "luminous.semi-bench.fdm-coverage-assumption.v1";
pub const FDM_COVERAGE_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.fdm-coverage-policy.v1";
pub const FDM_COVERAGE_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.fdm-coverage-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FdmCoverageStatusV1 {
    Complete,
    InsufficientCalibration,
    InsufficientBlocks,
    FullIntervalCutoff,
    AssumptionsNotSupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FdmCoverageInferenceMethodV1 {
    ZhaiChengWuBlockMeansShortMemory,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FdmAssumptionStateV1 {
    DeclaredByProcessModelNotProven,
    MissingProcessModelAssumptions,
    MissingQuantileRegularity,
    IndicatorFdmTailConditionNotMet,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FdmCoverageAssumptionEvidenceV1 {
    pub schema: String,
    pub evidence_id: String,
    /// Digest of the stationary causal process/innovation model used to bound
    /// the indicator FDM. This digest binds a claim; it does not authenticate it.
    pub source_process_model_sha256: String,
    pub predictor_artifact_sha256: String,
    pub score_definition_sha256: String,
    pub assumption_evidence_sha256: String,
    pub stationary_causal_innovation_representation_declared: bool,
    pub iid_innovations_declared: bool,
    pub fixed_predictor_independent_of_calibration_and_test_declared: bool,
    pub continuous_score_cdf_declared: bool,
    pub quantile_density_regularity_declared: bool,
    pub no_pairwise_score_ties_declared: bool,
    /// Upper bounds for theta_0,...,theta_K. Each coefficient is in [0,1].
    pub indicator_theta_upper_bounds: Vec<f64>,
    /// For every k > K: theta_k <= tail_power_constant * k^(-tail_power_exponent).
    /// The exponent must exceed 3 for the paper's block-variance tail condition.
    pub tail_power_constant: f64,
    pub tail_power_exponent: f64,
    pub quantile_density_lower_bound: Option<f64>,
    pub quantile_density_upper_bound: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FdmCoveragePolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub target_coverage: f64,
    /// Non-overlapping calibration-indicator block length ell.
    pub calibration_block_length: usize,
    pub confidence_z: f64,
    pub minimum_calibration_scores: usize,
    pub minimum_test_scores: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FdmQualifiedScoreObservationV1 {
    pub schema: String,
    pub prediction_receipt: StreamingPredictionReceiptV1,
    pub label_id: String,
    pub label_revision: u64,
    pub label_payload_sha256: String,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub delivered_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
    pub observation_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FdmCoverageInputV1 {
    pub schema: String,
    pub run_id: String,
    pub cutoff_at_unix_seconds: i64,
    pub source_stream_sha256: String,
    pub calibration_manifest_sha256: String,
    pub test_manifest_sha256: String,
    pub policy: FdmCoveragePolicyV1,
    pub assumptions: FdmCoverageAssumptionEvidenceV1,
    /// Calibration scores are the first n observations. The test window must
    /// be immediately adjacent in prediction-sequence order.
    pub calibration: Vec<FdmQualifiedScoreObservationV1>,
    pub test: Vec<FdmQualifiedScoreObservationV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FdmCoverageReportV1 {
    pub schema: String,
    pub run_id: String,
    pub input_sha256: String,
    pub source_stream_sha256: String,
    pub calibration_manifest_sha256: String,
    pub test_manifest_sha256: String,
    pub status: FdmCoverageStatusV1,
    pub inference_method: FdmCoverageInferenceMethodV1,
    pub assumption_state: FdmAssumptionStateV1,
    pub assumptions_declared_but_not_proven: bool,
    pub predictor_artifact_sha256: String,
    pub score_definition_sha256: String,
    pub target_coverage: f64,
    pub calibration_count: usize,
    pub test_count: usize,
    pub cutoff_rank: usize,
    /// None denotes the exact conformal +infinity cutoff when rank > n.
    pub calibration_cutoff: Option<f64>,
    pub finite_cutoff: bool,
    pub observed_test_coverage: Option<f64>,
    pub covered_test_count: usize,
    pub missed_test_count: usize,
    pub block_length: usize,
    pub complete_calibration_block_count: usize,
    pub discarded_calibration_remainder: usize,
    pub block_indicator_mean: Option<f64>,
    pub estimated_long_run_variance: Option<f64>,
    pub estimated_long_run_standard_deviation: Option<f64>,
    pub realized_coverage_standard_error: Option<f64>,
    /// Prediction interval for future realized coverage, centered at nominal
    /// coverage, as in Theorem 3 / Proposition 4 of arXiv:2609.33868.
    pub realized_coverage_reference_lower: Option<f64>,
    pub realized_coverage_reference_upper: Option<f64>,
    pub realized_coverage_test_statistic: Option<f64>,
    pub rejects_nominal_coverage: Option<bool>,
    pub indicator_fdm_theta_sum_upper_bound: Option<f64>,
    pub indicator_fdm_tail_condition_upper_bound: Option<f64>,
    pub marginal_coverage_error_bound_density_free: Option<f64>,
    pub marginal_bound_condition_met: bool,
    pub long_memory_supported: bool,
    pub iid_inference_claimed: bool,
    pub population_claim_authorized_by_qualified_stream_alone: bool,
    pub source_authenticity_proven: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct FdmCoverageErrorV1(pub String);

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, FdmCoverageErrorV1> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| FdmCoverageErrorV1(format!("canonical JSON serialization failed: {error}")))?;
    Ok(sha256_hex(&bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), FdmCoverageErrorV1> {
    if value.trim().is_empty() {
        return Err(FdmCoverageErrorV1(format!("{field}: expected non-empty text")));
    }
    Ok(())
}

fn score_observation_digest(
    row: &FdmQualifiedScoreObservationV1,
) -> Result<String, FdmCoverageErrorV1> {
    let mut canonical = row.clone();
    canonical.observation_sha256.clear();
    json_sha256(&canonical)
}

fn prediction_digest(
    receipt: &StreamingPredictionReceiptV1,
) -> Result<String, FdmCoverageErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn eligibility_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, FdmCoverageErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn input_digest(input: &FdmCoverageInputV1) -> Result<String, FdmCoverageErrorV1> {
    let mut canonical = input.clone();
    canonical.calibration.sort_by_key(|row| row.prediction_receipt.sequence);
    canonical.test.sort_by_key(|row| row.prediction_receipt.sequence);
    json_sha256(&canonical)
}

fn report_digest(report: &FdmCoverageReportV1) -> Result<String, FdmCoverageErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn validate_policy(policy: &FdmCoveragePolicyV1) -> Result<(), FdmCoverageErrorV1> {
    if policy.schema != FDM_COVERAGE_POLICY_SCHEMA_V1 {
        return Err(FdmCoverageErrorV1("FDM coverage policy schema is unsupported".into()));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if policy.revision == 0
        || !policy.target_coverage.is_finite()
        || !(0.0..1.0).contains(&policy.target_coverage)
        || !policy.confidence_z.is_finite()
        || policy.confidence_z <= 0.0
        || policy.calibration_block_length < 2
        || policy.minimum_calibration_scores < 2
        || policy.minimum_test_scores == 0
    {
        return Err(FdmCoverageErrorV1("FDM policy parameters are invalid".into()));
    }
    Ok(())
}

fn validate_assumptions(
    evidence: &FdmCoverageAssumptionEvidenceV1,
) -> Result<(), FdmCoverageErrorV1> {
    if evidence.schema != FDM_COVERAGE_ASSUMPTION_SCHEMA_V1 {
        return Err(FdmCoverageErrorV1("FDM assumption schema is unsupported".into()));
    }
    require_text(&evidence.evidence_id, "assumptions.evidence_id")?;
    for (field, value) in [
        ("source_process_model_sha256", evidence.source_process_model_sha256.as_str()),
        ("predictor_artifact_sha256", evidence.predictor_artifact_sha256.as_str()),
        ("score_definition_sha256", evidence.score_definition_sha256.as_str()),
        ("assumption_evidence_sha256", evidence.assumption_evidence_sha256.as_str()),
    ] {
        if !valid_sha256(value) {
            return Err(FdmCoverageErrorV1(format!("{field}: malformed SHA-256")));
        }
    }
    if evidence.indicator_theta_upper_bounds.is_empty()
        || evidence
            .indicator_theta_upper_bounds
            .iter()
            .any(|theta| !theta.is_finite() || !(0.0..=1.0).contains(theta))
        || !evidence.tail_power_constant.is_finite()
        || evidence.tail_power_constant < 0.0
        || !evidence.tail_power_exponent.is_finite()
    {
        return Err(FdmCoverageErrorV1("indicator FDM envelope is malformed".into()));
    }
    Ok(())
}

/// Computes conservative bounds on Theta=sum(theta_k) and on
/// sum_{k>=1}(sum_{j>=k}theta_j)^(1/2). The latter is the stronger tail
/// condition used by Proposition 4 for block-variance consistency.
///
/// The finite vector supplies theta_0,...,theta_K. For k>K the certificate
/// supplies theta_k <= C*k^(-a). Requiring a>3 is sufficient for the tail
/// series in Proposition 4.
fn fdm_envelope_bounds(
    evidence: &FdmCoverageAssumptionEvidenceV1,
) -> Result<(f64, f64), FdmCoverageErrorV1> {
    let finite_len = evidence.indicator_theta_upper_bounds.len();
    let tail_start = finite_len as f64;
    let exponent = evidence.tail_power_exponent;
    if exponent <= 3.0 {
        return Err(FdmCoverageErrorV1(
            "indicator FDM tail exponent must exceed 3 for the paper's variance-estimator condition"
                .into(),
        ));
    }

    let constant = evidence.tail_power_constant;
    let tail_sum = constant
        * (tail_start.powf(-exponent) + tail_start.powf(1.0 - exponent) / (exponent - 1.0));
    let theta_sum = evidence.indicator_theta_upper_bounds.iter().sum::<f64>() + tail_sum;

    let mut tail_condition = 0.0;
    for lag in 1..finite_len {
        let finite_tail = evidence.indicator_theta_upper_bounds[lag..]
            .iter()
            .sum::<f64>();
        let remaining = finite_tail + tail_sum;
        tail_condition += remaining.sqrt();
    }

    let power = (exponent - 1.0) / 2.0;
    let tail_coefficient = (constant * exponent / (exponent - 1.0)).sqrt();
    let tail_series = tail_coefficient
        * (tail_start.powf(-power) + tail_start.powf(1.0 - power) / (power - 1.0));
    tail_condition += tail_series;

    if !theta_sum.is_finite() || !tail_condition.is_finite() {
        return Err(FdmCoverageErrorV1("FDM envelope bound overflowed".into()));
    }
    Ok((theta_sum, tail_condition))
}

fn assumptions_supported(
    evidence: &FdmCoverageAssumptionEvidenceV1,
) -> Result<Option<FdmAssumptionStateV1>, FdmCoverageErrorV1> {
    let (_, tail_condition) = fdm_envelope_bounds(evidence)?;
    if !evidence.stationary_causal_innovation_representation_declared
        || !evidence.iid_innovations_declared
        || !evidence.fixed_predictor_independent_of_calibration_and_test_declared
        || !evidence.continuous_score_cdf_declared
    {
        return Ok(Some(FdmAssumptionStateV1::MissingProcessModelAssumptions));
    }
    let (Some(lower), Some(upper)) = (
        evidence.quantile_density_lower_bound,
        evidence.quantile_density_upper_bound,
    ) else {
        return Ok(Some(FdmAssumptionStateV1::MissingQuantileRegularity));
    };
    if !evidence.quantile_density_regularity_declared
        || !evidence.no_pairwise_score_ties_declared
        || !lower.is_finite()
        || !upper.is_finite()
        || lower <= 0.0
        || upper < lower
    {
        return Ok(Some(FdmAssumptionStateV1::MissingQuantileRegularity));
    }
    if tail_condition.is_infinite() {
        return Ok(Some(FdmAssumptionStateV1::IndicatorFdmTailConditionNotMet));
    }
    Ok(None)
}

fn validate_score_observation(
    row: &FdmQualifiedScoreObservationV1,
    cutoff: i64,
) -> Result<f64, FdmCoverageErrorV1> {
    if row.schema != FDM_COVERAGE_SCORE_SCHEMA_V1 {
        return Err(FdmCoverageErrorV1("FDM score observation schema is unsupported".into()));
    }
    require_text(&row.label_id, "score.label_id")?;
    require_text(&row.clock_domain_id, "score.clock_domain_id")?;
    if row.label_revision == 0 || !valid_sha256(&row.label_payload_sha256) {
        return Err(FdmCoverageErrorV1("score label revision/payload digest is invalid".into()));
    }
    if row.observation_sha256 != score_observation_digest(row)? {
        return Err(FdmCoverageErrorV1("score observation digest mismatch".into()));
    }
    let prediction = &row.prediction_receipt;
    if prediction.schema != "luminous.semi-bench.streaming-prediction-receipt.v1"
        || !valid_sha256(&prediction.receipt_sha256)
        || prediction_digest(prediction)? != prediction.receipt_sha256
        || !valid_sha256(&prediction.global_model_artifact_sha256)
        || !valid_sha256(&prediction.feature_schema_sha256)
        || !valid_sha256(&prediction.policy_sha256)
        || prediction.adapter_state_before_sha256 != prediction.adapter_state_after_sha256
        || !prediction.prediction_value.is_finite()
    {
        return Err(FdmCoverageErrorV1(
            "FDM score requires a valid frozen prediction receipt and unchanged adapter state"
                .into(),
        ));
    }
    let report = &row.eligibility_report;
    if !valid_sha256(&report.report_sha256)
        || eligibility_digest(report)? != report.report_sha256
        || report.frontier != LabelEvidenceFrontierV1::CurrentAtK1
        || report.eligibility != LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
        || !report.downstream_consumption_allowed
        || report.label_id != row.label_id
        || report.label_revision != row.label_revision
        || report.label_sha256 != row.label_payload_sha256
        || report.measurement_event_id != prediction.measurement_event_id
        || report.equipment_control_authority
        || report.training_snapshot_mutated
        || !report.qualified_corrected_value.is_finite()
        || !report.qualified_standard_uncertainty_upper_bound.is_some_and(|value| {
            value.is_finite() && value >= 0.0
        })
    {
        return Err(FdmCoverageErrorV1(
            "FDM score requires exact current K1 numeric label qualification".into(),
        ));
    }
    if row.clock_domain_id != prediction.clock_domain_id
        || row.label_available_at_unix_seconds < prediction.prediction_at_unix_seconds
        || row.eligibility_report_available_at_unix_seconds < row.label_available_at_unix_seconds
        || row.delivered_at_unix_seconds < row.eligibility_report_available_at_unix_seconds
        || row.delivered_at_unix_seconds > cutoff
    {
        return Err(FdmCoverageErrorV1(
            "FDM score label/report availability is incompatible with the declared cutoff".into(),
        ));
    }
    Ok((report.qualified_corrected_value - prediction.prediction_value).abs())
}

fn validate_adjacent_stream(
    calibration: &[FdmQualifiedScoreObservationV1],
    test: &[FdmQualifiedScoreObservationV1],
    assumptions: &FdmCoverageAssumptionEvidenceV1,
    cutoff: i64,
) -> Result<(Vec<f64>, Vec<f64>), FdmCoverageErrorV1> {
    let mut calibration_rows = calibration.to_vec();
    let mut test_rows = test.to_vec();
    calibration_rows.sort_by_key(|row| row.prediction_receipt.sequence);
    test_rows.sort_by_key(|row| row.prediction_receipt.sequence);

    let all_rows = calibration_rows.iter().chain(test_rows.iter());
    let mut samples = BTreeSet::new();
    let mut events = BTreeSet::new();
    let mut label_revisions = BTreeSet::new();
    let mut sequences = BTreeSet::new();
    let mut previous_event_time = None;
    let mut model_artifact = None;
    let mut feature_schema = None;
    let mut policy_digest = None;
    let mut adapter_state = None;
    let mut sequence_values = Vec::new();
    let mut all_scores = Vec::new();

    for row in all_rows {
        let score = validate_score_observation(row, cutoff)?;
        let prediction = &row.prediction_receipt;
        if !samples.insert(prediction.sample_id.as_str())
            || !events.insert(prediction.measurement_event_id.as_str())
            || !label_revisions.insert((row.label_id.as_str(), row.label_revision))
            || !sequences.insert(prediction.sequence)
        {
            return Err(FdmCoverageErrorV1(
                "duplicate sample/event/label revision/prediction sequence in calibration-test stream"
                    .into(),
            ));
        }
        if previous_event_time.is_some_and(|time| prediction.event_time_unix_seconds <= time) {
            return Err(FdmCoverageErrorV1("FDM event time must strictly increase".into()));
        }
        previous_event_time = Some(prediction.event_time_unix_seconds);

        for (stored, candidate, field) in [
            (
                &mut model_artifact,
                &prediction.global_model_artifact_sha256,
                "global model artifact",
            ),
            (
                &mut feature_schema,
                &prediction.feature_schema_sha256,
                "feature schema",
            ),
            (&mut policy_digest, &prediction.policy_sha256, "prediction policy"),
            (
                &mut adapter_state,
                &prediction.adapter_state_before_sha256,
                "frozen adapter state",
            ),
        ] {
            if stored.as_ref().is_some_and(|value| value != candidate) {
                return Err(FdmCoverageErrorV1(format!(
                    "{field} changed across the calibration/test stream"
                )));
            }
            *stored = Some(candidate.clone());
        }
        sequence_values.push(prediction.sequence);
        all_scores.push(score);
    }

    if calibration_rows.is_empty() || test_rows.is_empty() {
        return Err(FdmCoverageErrorV1(
            "calibration and adjacent test windows must both be non-empty".into(),
        ));
    }
    let calibration_last = calibration_rows.last().unwrap().prediction_receipt.sequence;
    let test_first = test_rows.first().unwrap().prediction_receipt.sequence;
    if calibration_last.checked_add(1) != Some(test_first)
        || sequence_values.windows(2).any(|pair| pair[1] != pair[0] + 1)
    {
        return Err(FdmCoverageErrorV1(
            "calibration and test windows must form one contiguous adjacent sequence".into(),
        ));
    }

    // Bind the declared process-model predictor to the actual frozen predictor.
    if model_artifact.as_deref() != Some(assumptions.predictor_artifact_sha256.as_str()) {
        return Err(FdmCoverageErrorV1(
            "FDM assumption evidence references a different predictor artifact".into(),
        ));
    }

    let calibration_scores = all_scores[..calibration_rows.len()].to_vec();
    let test_scores = all_scores[calibration_rows.len()..].to_vec();
    Ok((calibration_scores, test_scores))
}

fn input_digest(input: &FdmCoverageInputV1) -> Result<String, FdmCoverageErrorV1> {
    let mut canonical = input.clone();
    canonical.calibration.sort_by_key(|row| row.prediction_receipt.sequence);
    canonical.test.sort_by_key(|row| row.prediction_receipt.sequence);
    json_sha256(&canonical)
}

fn report_digest(report: &FdmCoverageReportV1) -> Result<String, FdmCoverageErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

/// Paper-specific block-means estimator from Proposition 4 of
/// arXiv:2609.33868, not a Newey-West/HAC estimator.
fn block_means_long_run_variance(
    calibration_indicators: &[f64],
    block_length: usize,
) -> Result<(f64, f64, usize, usize), FdmCoverageErrorV1> {
    let n = calibration_indicators.len();
    if block_length < 2 || block_length >= n {
        return Err(FdmCoverageErrorV1("block length must be in [2,n)".into()));
    }
    let blocks = n / block_length;
    if blocks < 2 {
        return Err(FdmCoverageErrorV1(
            "at least two complete non-overlapping calibration blocks are required".into(),
        ));
    }
    let remainder = n % block_length;
    let means = (0..blocks)
        .map(|block| {
            let start = block * block_length;
            calibration_indicators[start..start + block_length]
                .iter()
                .sum::<f64>()
                / block_length as f64
        })
        .collect::<Vec<_>>();
    let mean = means.iter().sum::<f64>() / blocks as f64;
    let sigma_squared = block_length as f64
        / blocks as f64
        * means.iter().map(|value| (value - mean).powi(2)).sum::<f64>();
    if !sigma_squared.is_finite() || sigma_squared < 0.0 {
        return Err(FdmCoverageErrorV1(
            "paper block-means long-run variance is negative or non-finite".into(),
        ));
    }
    Ok((mean, sigma_squared, blocks, remainder))
}

/// Evaluates the paper-specific short-memory block-SE method.
///
/// The input must contain a calibration window followed immediately by a test
/// window. Every score must be K1-qualified and available by cutoff. This is a
/// selected/qualified observed-stream estimand, not population coverage.
pub fn evaluate_fdm_coverage_inference_v1(
    input: &FdmCoverageInputV1,
) -> Result<FdmCoverageReportV1, FdmCoverageErrorV1> {
    if input.schema != FDM_COVERAGE_SCHEMA_V1 {
        return Err(FdmCoverageErrorV1("FDM coverage input schema is unsupported".into()));
    }
    require_text(&input.run_id, "input.run_id")?;
    for (field, value) in [
        ("source_stream_sha256", input.source_stream_sha256.as_str()),
        ("calibration_manifest_sha256", input.calibration_manifest_sha256.as_str()),
        ("test_manifest_sha256", input.test_manifest_sha256.as_str()),
    ] {
        if !valid_sha256(value) {
            return Err(FdmCoverageErrorV1(format!("{field}: malformed SHA-256")));
        }
    }
    validate_policy(&input.policy)?;
    validate_assumptions(&input.assumptions)?;

    let mut report = FdmCoverageReportV1 {
        schema: FDM_COVERAGE_REPORT_SCHEMA_V1.into(),
        run_id: input.run_id.clone(),
        input_sha256: input_digest(input)?,
        source_stream_sha256: input.source_stream_sha256.clone(),
        calibration_manifest_sha256: input.calibration_manifest_sha256.clone(),
        test_manifest_sha256: input.test_manifest_sha256.clone(),
        status: FdmCoverageStatusV1::AssumptionsNotSupported,
        inference_method: FdmCoverageInferenceMethodV1::ZhaiChengWuBlockMeansShortMemory,
        assumption_state: FdmAssumptionStateV1::MissingProcessModelAssumptions,
        assumptions_declared_but_not_proven: true,
        predictor_artifact_sha256: input.assumptions.predictor_artifact_sha256.clone(),
        score_definition_sha256: input.assumptions.score_definition_sha256.clone(),
        target_coverage: input.policy.target_coverage,
        calibration_count: input.calibration.len(),
        test_count: input.test.len(),
        cutoff_rank: 0,
        calibration_cutoff: None,
        finite_cutoff: false,
        observed_test_coverage: None,
        covered_test_count: 0,
        missed_test_count: 0,
        block_length: input.policy.calibration_block_length,
        complete_calibration_block_count: 0,
        discarded_calibration_remainder: 0,
        block_indicator_mean: None,
        estimated_long_run_variance: None,
        estimated_long_run_standard_deviation: None,
        realized_coverage_standard_error: None,
        realized_coverage_reference_lower: None,
        realized_coverage_reference_upper: None,
        realized_coverage_test_statistic: None,
        rejects_nominal_coverage: None,
        indicator_fdm_theta_sum_upper_bound: None,
        indicator_fdm_tail_condition_upper_bound: None,
        marginal_coverage_error_bound_density_free: None,
        marginal_bound_condition_met: false,
        long_memory_supported: false,
        iid_inference_claimed: false,
        population_claim_authorized_by_qualified_stream_alone: false,
        source_authenticity_proven: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };

    let (theta_sum, tail_condition) = fdm_envelope_bounds(&input.assumptions)?;
    report.indicator_fdm_theta_sum_upper_bound = Some(theta_sum);
    report.indicator_fdm_tail_condition_upper_bound = Some(tail_condition);

    if let Some(reason) = assumptions_supported(&input.assumptions)? {
        report.assumption_state = reason;
        report.report_sha256 = report_digest(&report)?;
        return Ok(report);
    }
    report.assumption_state = FdmAssumptionStateV1::DeclaredByProcessModelNotProven;

    let (calibration_scores, test_scores) = validate_adjacent_stream(
        &input.calibration,
        &input.test,
        &input.assumptions,
        input.cutoff_at_unix_seconds,
    )?;
    let n = calibration_scores.len();
    let m = test_scores.len();
    if n < input.policy.minimum_calibration_scores || m < input.policy.minimum_test_scores {
        report.status = FdmCoverageStatusV1::InsufficientCalibration;
        report.report_sha256 = report_digest(&report)?;
        return Ok(report);
    }

    let rank = ((n as f64 + 1.0) * input.policy.target_coverage).ceil() as usize;
    report.cutoff_rank = rank;
    if rank > n {
        // Exact finite-sample split-conformal convention: q_hat = +infinity.
        report.status = FdmCoverageStatusV1::FullIntervalCutoff;
        report.covered_test_count = m;
        report.observed_test_coverage = Some(1.0);
        report.report_sha256 = report_digest(&report)?;
        return Ok(report);
    }
    let mut sorted = calibration_scores.clone();
    sorted.sort_by(f64::total_cmp);
    let cutoff = sorted[rank - 1];
    report.calibration_cutoff = Some(cutoff);
    report.finite_cutoff = true;

    let calibration_indicators = calibration_scores
        .iter()
        .map(|score| if *score <= cutoff { 1.0 } else { 0.0 })
        .collect::<Vec<_>>();
    let covered = test_scores.iter().filter(|score| **score <= cutoff).count();
    let misses = m - covered;
    report.covered_test_count = covered;
    report.missed_test_count = misses;
    report.observed_test_coverage = Some(covered as f64 / m as f64);

    let block_length = input.policy.calibration_block_length;
    let blocks = n / block_length;
    if blocks < 2 {
        report.status = FdmCoverageStatusV1::InsufficientBlocks;
        report.report_sha256 = report_digest(&report)?;
        return Ok(report);
    }
    let (block_mean, sigma_squared, blocks, remainder) =
        block_means_long_run_variance(&calibration_indicators, block_length)?;
    report.block_indicator_mean = Some(block_mean);
    report.estimated_long_run_variance = Some(sigma_squared);
    report.estimated_long_run_standard_deviation = Some(sigma_squared.sqrt());
    report.complete_calibration_block_count = blocks;
    report.discarded_calibration_remainder = remainder;

    let standard_error = sigma_squared.sqrt() * (1.0 / n as f64 + 1.0 / m as f64).sqrt();
    if !standard_error.is_finite() {
        return Err(FdmCoverageErrorV1("paper block-SE result is non-finite".into()));
    }
    if standard_error > 0.0 {
        let observed = report.observed_test_coverage.unwrap();
        let statistic = (observed - input.policy.target_coverage) / standard_error;
        report.realized_coverage_standard_error = Some(standard_error);
        report.realized_coverage_reference_lower =
            Some((input.policy.target_coverage - input.policy.confidence_z * standard_error).max(0.0));
        report.realized_coverage_reference_upper =
            Some((input.policy.target_coverage + input.policy.confidence_z * standard_error).min(1.0));
        report.realized_coverage_test_statistic = Some(statistic);
        report.rejects_nominal_coverage = Some(statistic.abs() > input.policy.confidence_z);
    }

    let alpha = 1.0 - input.policy.target_coverage;
    let condition = 2.0 / n as f64 + (2.0 * theta_sum.powi(2) / n as f64).cbrt()
        < alpha.min(1.0 - alpha);
    report.marginal_bound_condition_met = condition;
    if condition {
        report.marginal_coverage_error_bound_density_free =
            Some(2.0 / n as f64 + 3.0 * (theta_sum.powi(2) / (4.0 * n as f64)).cbrt());
    }

    report.status = FdmCoverageStatusV1::Complete;
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn assumptions() -> FdmCoverageAssumptionEvidenceV1 {
        FdmCoverageAssumptionEvidenceV1 {
            schema: FDM_COVERAGE_ASSUMPTION_SCHEMA_V1.into(),
            evidence_id: "stationary-fixed-vm-process-v1".into(),
            source_process_model_sha256: digest('a'),
            predictor_artifact_sha256: digest('b'),
            score_definition_sha256: digest('c'),
            assumption_evidence_sha256: digest('d'),
            stationary_causal_innovation_representation_declared: true,
            iid_innovations_declared: true,
            fixed_predictor_independent_of_calibration_and_test_declared: true,
            continuous_score_cdf_declared: true,
            quantile_density_regularity_declared: true,
            no_pairwise_score_ties_declared: true,
            indicator_theta_upper_bounds: vec![1.0, 0.5, 0.25],
            tail_power_constant: 0.01,
            tail_power_exponent: 4.0,
            quantile_density_lower_bound: Some(0.1),
            quantile_density_upper_bound: Some(1.0),
        }
    }

    fn policy(block_length: usize) -> FdmCoveragePolicyV1 {
        FdmCoveragePolicyV1 {
            schema: FDM_COVERAGE_POLICY_SCHEMA_V1.into(),
            policy_id: "fdm-paper-short-memory-v1".into(),
            revision: 1,
            target_coverage: 0.90,
            calibration_block_length: block_length,
            confidence_z: 1.959963984540054,
            minimum_calibration_scores: 4,
            minimum_test_scores: 2,
        }
    }

    fn score(
        sequence: u64,
        sample_id: &str,
        event_id: &str,
        score: f64,
        model: char,
        event_time: i64,
    ) -> FdmQualifiedScoreObservationV1 {
        let mut prediction = StreamingPredictionReceiptV1 {
            schema: "luminous.semi-bench.streaming-prediction-receipt.v1".into(),
            sequence,
            sample_id: sample_id.into(),
            measurement_event_id: event_id.into(),
            event_time_unix_seconds: event_time,
            feature_available_at_unix_seconds: event_time + 1,
            prediction_at_unix_seconds: event_time + 2,
            clock_domain_id: "fab-clock".into(),
            global_model_artifact_sha256: digest(model),
            feature_schema_sha256: digest('e'),
            policy_sha256: digest('f'),
            context_key: Some("tool-a/ch-1".into()),
            feature_vector_sha256: digest('1'),
            selected_feature_indices: vec![0],
            selected_feature_vector: vec![1.0],
            global_baseline_prediction: 0.0,
            feature_adapter_prediction: 0.0,
            context_correction: 0.0,
            temporal_correction: 0.0,
            base_prediction: 0.0,
            prediction_value: 0.0,
            adapter_state_before_sha256: digest('2'),
            adapter_state_after_sha256: digest('2'),
            receipt_sha256: String::new(),
        };
        prediction.receipt_sha256 = prediction_digest(&prediction).unwrap();
        let label_id = format!("label-{sequence}");
        let payload = digest('3');
        let mut eligibility = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.clone(),
            label_revision: 1,
            measurement_event_id: event_id.into(),
            calibration_certificate_id: "certificate-1".into(),
            frontier: LabelEvidenceFrontierV1::CurrentAtK1,
            eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            downstream_consumption_allowed: true,
            reasons: Vec::new(),
            label_sha256: payload.clone(),
            measurement_sha256: digest('4'),
            qualified_corrected_value: score,
            qualified_standard_uncertainty_upper_bound: Some(0.05),
            qualification_report_sha256: digest('5'),
            requalification_report_sha256: digest('6'),
            requalification_frontier_sha256: digest('7'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        eligibility.report_sha256 = eligibility_digest(&eligibility).unwrap();
        let mut row = FdmQualifiedScoreObservationV1 {
            schema: FDM_COVERAGE_SCORE_SCHEMA_V1.into(),
            prediction_receipt: prediction,
            label_id,
            label_revision: 1,
            label_payload_sha256: payload,
            label_available_at_unix_seconds: event_time + 3,
            eligibility_report_available_at_unix_seconds: event_time + 4,
            delivered_at_unix_seconds: event_time + 5,
            clock_domain_id: "fab-clock".into(),
            eligibility_report: eligibility,
            observation_sha256: String::new(),
        };
        row.observation_sha256 = score_observation_digest(&row).unwrap();
        row
    }

    fn input() -> FdmCoverageInputV1 {
        let all = (0..32)
            .map(|index| {
                let value = if index % 5 == 0 { 1.5 } else { 0.5 };
                score(
                    index + 1,
                    &format!("sample-{index}"),
                    &format!("event-{index}"),
                    value,
                    'b',
                    index as i64 * 10,
                )
            })
            .collect::<Vec<_>>();
        FdmCoverageInputV1 {
            schema: FDM_COVERAGE_SCHEMA_V1.into(),
            run_id: "fdm-run-001".into(),
            cutoff_at_unix_seconds: 10_000,
            source_stream_sha256: digest('8'),
            calibration_manifest_sha256: digest('9'),
            test_manifest_sha256: digest('a'),
            policy: policy(4),
            assumptions: assumptions(),
            calibration: all[..24].to_vec(),
            test: all[24..].to_vec(),
        }
    }

    #[test]
    fn paper_block_means_result_matches_manual_non_overlapping_calibration_blocks() {
        let value = input();
        let report = evaluate_fdm_coverage_inference_v1(&value).unwrap();
        assert_eq!(report.status, FdmCoverageStatusV1::Complete);
        assert_eq!(report.inference_method, FdmCoverageInferenceMethodV1::ZhaiChengWuBlockMeansShortMemory);
        assert_eq!(report.calibration_count, 24);
        assert_eq!(report.test_count, 8);
        assert_eq!(report.complete_calibration_block_count, 6);
        assert_eq!(report.discarded_calibration_remainder, 0);
        assert!(report.realized_coverage_standard_error.is_some());
        assert!(!report.long_memory_supported);
        assert!(report.assumptions_declared_but_not_proven);
    }

    #[test]
    fn exponent_not_above_three_is_rejected_for_proposition_four_tail_condition() {
        let mut value = input();
        value.assumptions.tail_power_exponent = 3.0;
        assert!(evaluate_fdm_coverage_inference_v1(&value).is_err());
    }

    #[test]
    fn changed_predictor_or_updated_adapter_invalidates_paper_assumptions() {
        let mut value = input();
        value.test[0].prediction_receipt.adapter_state_after_sha256 = digest('9');
        value.test[0].prediction_receipt.receipt_sha256 =
            prediction_digest(&value.test[0].prediction_receipt).unwrap();
        value.test[0].observation_sha256 = score_observation_digest(&value.test[0]).unwrap();
        assert!(evaluate_fdm_coverage_inference_v1(&value).is_err());

        let mut value = input();
        value.test[0].prediction_receipt.global_model_artifact_sha256 = digest('7');
        value.test[0].prediction_receipt.receipt_sha256 =
            prediction_digest(&value.test[0].prediction_receipt).unwrap();
        value.test[0].observation_sha256 = score_observation_digest(&value.test[0]).unwrap();
        assert!(evaluate_fdm_coverage_inference_v1(&value).is_err());
    }

    #[test]
    fn non_adjacent_calibration_and_test_stream_is_rejected() {
        let mut value = input();
        value.test[0].prediction_receipt.sequence += 1;
        value.test[0].prediction_receipt.receipt_sha256 =
            prediction_digest(&value.test[0].prediction_receipt).unwrap();
        value.test[0].observation_sha256 = score_observation_digest(&value.test[0]).unwrap();
        assert!(evaluate_fdm_coverage_inference_v1(&value).is_err());
    }

    #[test]
    fn input_order_does_not_change_the_sealed_report() {
        let mut value = input();
        let first = evaluate_fdm_coverage_inference_v1(&value).unwrap();
        value.calibration.reverse();
        value.test.reverse();
        let second = evaluate_fdm_coverage_inference_v1(&value).unwrap();
        assert_eq!(first.report_sha256, second.report_sha256);
    }
}
