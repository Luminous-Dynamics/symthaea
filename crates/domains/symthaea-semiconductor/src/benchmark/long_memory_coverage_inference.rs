// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Appendix C long-memory block-sampling inference for realized conformal coverage.
//!
//! This is deliberately separate from short-memory FDM batch-means inference
//! and from Newey-West/HAC. It implements the Gaussian linear, regularly
//! varying coefficient model studied in Section 4 / Appendix C of
//! arXiv:2609.33868. Model assumptions are declared and digest-bound, not
//! established by the finite observed wafer stream.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

use super::fdm_coverage_inference::{
    FdmQualifiedScoreObservationV1, validate_score_observation,
};

pub const LONG_MEMORY_COVERAGE_SCHEMA_V1: &str =
    "luminous.semi-bench.long-memory-coverage.v1";
pub const LONG_MEMORY_COVERAGE_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.long-memory-coverage-policy.v1";
pub const LONG_MEMORY_ASSUMPTION_SCHEMA_V1: &str =
    "luminous.semi-bench.long-memory-assumption.v1";
pub const LONG_MEMORY_COVERAGE_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.long-memory-coverage-report.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LongMemoryCoverageStatusV1 {
    Complete,
    AssumptionsNotSupported,
    InsufficientInput,
    BlockConstraintsNotSatisfied,
    FullIntervalCutoff,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LongMemoryLimitRegimeV1 {
    GaussianShortMemoryWithinLinearModel,
    CriticalGaussianAtThreeQuarter,
    RosenblattNonGaussian,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GaussianLinearLongMemoryEvidenceV1 {
    pub schema: String,
    pub evidence_id: String,
    pub source_process_model_sha256: String,
    pub predictor_artifact_sha256: String,
    pub absolute_error_score_definition_sha256: String,
    pub assumption_evidence_sha256: String,
    pub gaussian_iid_innovations_declared: bool,
    pub regularly_varying_coefficients_declared: bool,
    pub fixed_predictor_independent_of_calibration_and_test_declared: bool,
    /// Declared positive leading coefficient scale c in a_j ~ c*j^(-beta).
    pub coefficient_scale_lower_bound: f64,
    /// Declared positive remainder exponent phi in a_j=j^(-beta){c+O(j^(-phi))}.
    pub coefficient_refinement_exponent: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LongMemoryCoveragePolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    pub target_coverage: f64,
    /// b=floor(n^zeta) in Appendix C.1.
    pub block_exponent_zeta: f64,
    pub significance_level: f64,
    pub minimum_calibration_count: usize,
    pub minimum_test_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LongMemoryCoverageInputV1 {
    pub schema: String,
    pub run_id: String,
    pub cutoff_at_unix_seconds: i64,
    pub source_stream_sha256: String,
    pub calibration_manifest_sha256: String,
    pub test_manifest_sha256: String,
    pub policy: LongMemoryCoveragePolicyV1,
    pub assumptions: GaussianLinearLongMemoryEvidenceV1,
    pub calibration: Vec<FdmQualifiedScoreObservationV1>,
    pub test: Vec<FdmQualifiedScoreObservationV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LongMemoryCoverageReportV1 {
    pub schema: String,
    pub run_id: String,
    pub input_sha256: String,
    pub source_stream_sha256: String,
    pub calibration_manifest_sha256: String,
    pub test_manifest_sha256: String,
    pub status: LongMemoryCoverageStatusV1,
    pub limit_regime: Option<LongMemoryLimitRegimeV1>,
    pub assumptions_declared_but_not_proven: bool,
    pub predictor_artifact_sha256: String,
    pub target_coverage: f64,
    pub significance_level: f64,
    pub calibration_count: usize,
    pub test_count: usize,
    pub calibration_rank: Option<usize>,
    pub full_calibration_cutoff: Option<f64>,
    pub observed_test_coverage: Option<f64>,
    pub covered_test_count: usize,
    pub missed_test_count: usize,
    pub pilot_block_length_h: usize,
    pub coverage_block_length_b: usize,
    pub adjacent_test_block_length_ell_b: usize,
    pub paired_block_count: usize,
    pub beta_estimate: Option<f64>,
    pub q_h: Option<f64>,
    pub q_2h: Option<f64>,
    pub beta_used_zero_variance_convention: bool,
    pub normalization_r_b: Option<f64>,
    pub normalization_r_n: Option<f64>,
    pub block_distribution_lower_quantile: Option<f64>,
    pub block_distribution_upper_quantile: Option<f64>,
    pub normalized_full_window_statistic: Option<f64>,
    pub realized_coverage_reference_lower: Option<f64>,
    pub realized_coverage_reference_upper: Option<f64>,
    pub rejects_nominal_coverage: Option<bool>,
    pub normalized_block_distribution_sha256: Option<String>,
    pub long_memory_supported: bool,
    pub iid_inference_claimed: bool,
    pub population_claim_authorized_by_qualified_stream_alone: bool,
    pub source_authenticity_proven: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct LongMemoryCoverageErrorV1(pub String);

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(value: &T) -> Result<String, LongMemoryCoverageErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        LongMemoryCoverageErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_text(value: &str, field: &str) -> Result<(), LongMemoryCoverageErrorV1> {
    if value.trim().is_empty() {
        return Err(LongMemoryCoverageErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn input_digest(input: &LongMemoryCoverageInputV1) -> Result<String, LongMemoryCoverageErrorV1> {
    let mut canonical = input.clone();
    canonical.calibration.sort_by_key(|row| row.prediction_receipt.sequence);
    canonical.test.sort_by_key(|row| row.prediction_receipt.sequence);
    json_sha256(&canonical)
}

fn report_digest(
    report: &LongMemoryCoverageReportV1,
) -> Result<String, LongMemoryCoverageErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn expected_score_definition_sha256() -> String {
    sha256_hex(b"abs(qualified_corrected_value - frozen_prediction_receipt.prediction_value)")
}

fn validate_input_policy(policy: &LongMemoryCoveragePolicyV1) -> Result<(), LongMemoryCoverageErrorV1> {
    if policy.schema != LONG_MEMORY_COVERAGE_POLICY_SCHEMA_V1 {
        return Err(LongMemoryCoverageErrorV1(
            "long-memory coverage policy schema is unsupported".into(),
        ));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if policy.revision == 0
        || !policy.target_coverage.is_finite()
        || !(0.0..1.0).contains(&policy.target_coverage)
        || !policy.block_exponent_zeta.is_finite()
        || !(0.0..1.0).contains(&policy.block_exponent_zeta)
        || !policy.significance_level.is_finite()
        || !(0.0..1.0).contains(&policy.significance_level)
        || policy.minimum_calibration_count < 8
        || policy.minimum_test_count < 1
    {
        return Err(LongMemoryCoverageErrorV1(
            "long-memory policy has invalid coverage, zeta, significance or sample limits".into(),
        ));
    }
    Ok(())
}

fn validate_model_evidence(
    evidence: &GaussianLinearLongMemoryEvidenceV1,
) -> Result<(), LongMemoryCoverageErrorV1> {
    if evidence.schema != LONG_MEMORY_ASSUMPTION_SCHEMA_V1 {
        return Err(LongMemoryCoverageErrorV1(
            "long-memory assumption schema is unsupported".into(),
        ));
    }
    require_text(&evidence.evidence_id, "assumptions.evidence_id")?;
    for (field, value) in [
        ("source_process_model_sha256", evidence.source_process_model_sha256.as_str()),
        ("predictor_artifact_sha256", evidence.predictor_artifact_sha256.as_str()),
        (
            "absolute_error_score_definition_sha256",
            evidence.absolute_error_score_definition_sha256.as_str(),
        ),
        ("assumption_evidence_sha256", evidence.assumption_evidence_sha256.as_str()),
    ] {
        if !valid_sha256(value) {
            return Err(LongMemoryCoverageErrorV1(format!("{field}: malformed SHA-256")));
        }
    }
    if !evidence.coefficient_scale_lower_bound.is_finite()
        || evidence.coefficient_scale_lower_bound <= 0.0
        || !evidence.coefficient_refinement_exponent.is_finite()
        || evidence.coefficient_refinement_exponent <= 0.0
    {
        return Err(LongMemoryCoverageErrorV1(
            "Gaussian-linear coefficient assumptions are not positive and finite".into(),
        ));
    }
    Ok(())
}

fn validate_adjacent_scores(
    calibration: &[FdmQualifiedScoreObservationV1],
    test: &[FdmQualifiedScoreObservationV1],
    evidence: &GaussianLinearLongMemoryEvidenceV1,
    cutoff: i64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), LongMemoryCoverageErrorV1> {
    let mut calibration_rows = calibration.to_vec();
    let mut test_rows = test.to_vec();
    calibration_rows.sort_by_key(|row| row.prediction_receipt.sequence);
    test_rows.sort_by_key(|row| row.prediction_receipt.sequence);

    if calibration_rows.is_empty() || test_rows.is_empty() {
        return Err(LongMemoryCoverageErrorV1(
            "calibration and test windows must both be non-empty".into(),
        ));
    }

    let mut seen_samples = BTreeSet::new();
    let mut seen_events = BTreeSet::new();
    let mut seen_labels = BTreeSet::new();
    let mut seen_sequences = BTreeSet::new();
    let mut previous_time = None;
    let mut common_model: Option<String> = None;
    let mut common_schema: Option<String> = None;
    let mut common_policy: Option<String> = None;
    let mut common_adapter_state: Option<String> = None;

    let mut scores = Vec::with_capacity(calibration_rows.len() + test_rows.len());
    let mut signed_residuals = Vec::with_capacity(calibration_rows.len() + test_rows.len());
    let mut sequences = Vec::with_capacity(calibration_rows.len() + test_rows.len());

    for row in calibration_rows.iter().chain(test_rows.iter()) {
        let score = validate_score_observation(row, cutoff)
            .map_err(|error| LongMemoryCoverageErrorV1(error.to_string()))?;
        let prediction = &row.prediction_receipt;
        let report = &row.eligibility_report;
        if !seen_samples.insert(prediction.sample_id.as_str())
            || !seen_events.insert(prediction.measurement_event_id.as_str())
            || !seen_labels.insert((row.label_id.as_str(), row.label_revision))
            || !seen_sequences.insert(prediction.sequence)
        {
            return Err(LongMemoryCoverageErrorV1(
                "duplicate sample, event, label revision or prediction sequence".into(),
            ));
        }
        if previous_time.is_some_and(|value| prediction.event_time_unix_seconds <= value) {
            return Err(LongMemoryCoverageErrorV1(
                "event time must strictly increase through the calibration/test windows".into(),
            ));
        }
        previous_time = Some(prediction.event_time_unix_seconds);

        for (stored, observed, field) in [
            (
                &mut common_model,
                &prediction.global_model_artifact_sha256,
                "global model artifact",
            ),
            (
                &mut common_schema,
                &prediction.feature_schema_sha256,
                "feature schema",
            ),
            (&mut common_policy, &prediction.policy_sha256, "prediction policy"),
            (
                &mut common_adapter_state,
                &prediction.adapter_state_before_sha256,
                "adapter state",
            ),
        ] {
            if stored.as_ref().is_some_and(|value| value != observed) {
                return Err(LongMemoryCoverageErrorV1(format!(
                    "{field} changes inside the frozen calibration/test stream"
                )));
            }
            *stored = Some(observed.clone());
        }
        signed_residuals.push(report.qualified_corrected_value - prediction.prediction_value);
        scores.push(score);
        sequences.push(prediction.sequence);
    }

    if common_model.as_deref() != Some(evidence.predictor_artifact_sha256.as_str()) {
        return Err(LongMemoryCoverageErrorV1(
            "process-model evidence references a different frozen predictor artifact".into(),
        ));
    }
    if evidence.absolute_error_score_definition_sha256 != expected_score_definition_sha256() {
        return Err(LongMemoryCoverageErrorV1(
            "long-memory score definition must be absolute frozen-predictor residual".into(),
        ));
    }
    if !evidence.gaussian_iid_innovations_declared
        || !evidence.regularly_varying_coefficients_declared
        || !evidence.fixed_predictor_independent_of_calibration_and_test_declared
    {
        return Err(LongMemoryCoverageErrorV1(
            "Gaussian-linear long-memory model assumptions are not declared".into(),
        ));
    }

    let calibration_last = calibration_rows.last().unwrap().prediction_receipt.sequence;
    let test_first = test_rows.first().unwrap().prediction_receipt.sequence;
    if calibration_last.checked_add(1) != Some(test_first)
        || sequences.windows(2).any(|pair| pair[0].checked_add(1) != Some(pair[1]))
    {
        return Err(LongMemoryCoverageErrorV1(
            "calibration and test windows must be immediately adjacent and complete".into(),
        ));
    }

    let n = calibration_rows.len();
    let calibration_scores = scores[..n].to_vec();
    let test_scores = scores[n..].to_vec();
    let calibration_residuals = signed_residuals[..n].to_vec();
    Ok((calibration_scores, test_scores, calibration_residuals))
}

fn two_scale_block_variance(
    signed_residuals: &[f64],
    block_length: usize,
) -> Result<f64, LongMemoryCoverageErrorV1> {
    let n = signed_residuals.len();
    if block_length == 0 || block_length >= n {
        return Err(LongMemoryCoverageErrorV1(
            "two-scale pilot block length is invalid".into(),
        ));
    }
    let blocks = n - block_length + 1;
    let mut total = 0.0;
    for start in 0..blocks {
        let sum = signed_residuals[start..start + block_length]
            .iter()
            .sum::<f64>();
        total += sum * sum;
    }
    let q = total / blocks as f64;
    if !q.is_finite() || q < 0.0 {
        return Err(LongMemoryCoverageErrorV1(
            "two-scale block variance is non-finite or negative".into(),
        ));
    }
    Ok(q)
}

fn estimate_memory_parameter(
    signed_residuals: &[f64],
    h: usize,
) -> Result<(f64, f64, f64, bool), LongMemoryCoverageErrorV1> {
    let q_h = two_scale_block_variance(signed_residuals, h)?;
    let q_2h = two_scale_block_variance(signed_residuals, 2 * h)?;
    if q_h == 0.0 || q_2h == 0.0 {
        return Ok((0.75, q_h, q_2h, true));
    }
    let beta = 1.5 - (q_2h / q_h).ln() / (2.0 * std::f64::consts::LN_2);
    Ok((beta, q_h, q_2h, false))
}

/// R_N(u) = sqrt((1 + integral_1^N x^(2-4u) dx)/N).
/// exp_m1 avoids cancellation close to the critical u=3/4 point.
fn long_memory_normalization(n: usize, beta: f64) -> Result<f64, LongMemoryCoverageErrorV1> {
    if n < 1 || !beta.is_finite() || !(0.5..1.0).contains(&beta) {
        return Err(LongMemoryCoverageErrorV1(
            "normalization requires N>=1 and beta in (1/2,1)".into(),
        ));
    }
    let log_n = (n as f64).ln();
    let exponent = 3.0 - 4.0 * beta;
    let integral = if exponent.abs() < 1.0e-8 {
        log_n
    } else {
        (exponent * log_n).exp_m1() / exponent
    };
    let squared = (1.0 + integral) / n as f64;
    if !squared.is_finite() || squared <= 0.0 {
        return Err(LongMemoryCoverageErrorV1(
            "long-memory normalization is non-positive or non-finite".into(),
        ));
    }
    Ok(squared.sqrt())
}

fn conformal_cutoff(scores: &[f64], target: f64) -> Result<Option<f64>, LongMemoryCoverageErrorV1> {
    if scores.is_empty() {
        return Err(LongMemoryCoverageErrorV1("cannot calibrate an empty score vector".into()));
    }
    let rank = ((scores.len() as f64 + 1.0) * target).ceil() as usize;
    if rank > scores.len() {
        return Ok(None);
    }
    let mut ordered = scores.to_vec();
    ordered.sort_by(f64::total_cmp);
    Ok(Some(ordered[rank - 1]))
}

fn empirical_quantile(values: &[f64], probability: f64) -> Result<f64, LongMemoryCoverageErrorV1> {
    if values.is_empty() || !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
        return Err(LongMemoryCoverageErrorV1("empirical quantile input is invalid".into()));
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let rank = ((probability * ordered.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(ordered.len() - 1);
    Ok(ordered[rank])
}

fn report_digest(
    report: &LongMemoryCoverageReportV1,
) -> Result<String, LongMemoryCoverageErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

/// Implements Section 4 / Appendix C.1 block-sampling inference under the
/// declared Gaussian linear, regularly varying residual model.
pub fn evaluate_long_memory_coverage_inference_v1(
    input: &LongMemoryCoverageInputV1,
) -> Result<LongMemoryCoverageReportV1, LongMemoryCoverageErrorV1> {
    if input.schema != LONG_MEMORY_COVERAGE_SCHEMA_V1 {
        return Err(LongMemoryCoverageErrorV1(
            "long-memory coverage input schema is unsupported".into(),
        ));
    }
    require_text(&input.run_id, "input.run_id")?;
    for (field, value) in [
        ("source_stream_sha256", input.source_stream_sha256.as_str()),
        ("calibration_manifest_sha256", input.calibration_manifest_sha256.as_str()),
        ("test_manifest_sha256", input.test_manifest_sha256.as_str()),
        (
            "source_process_model_sha256",
            input.assumptions.source_process_model_sha256.as_str(),
        ),
        ("predictor_artifact_sha256", input.assumptions.predictor_artifact_sha256.as_str()),
        (
            "absolute_error_score_definition_sha256",
            input.assumptions.absolute_error_score_definition_sha256.as_str(),
        ),
        ("assumption_evidence_sha256", input.assumptions.assumption_evidence_sha256.as_str()),
    ] {
        if !valid_sha256(value) {
            return Err(LongMemoryCoverageErrorV1(format!("{field}: malformed SHA-256")));
        }
    }
    validate_input_policy(&input.policy)?;
    validate_model_evidence(&input.assumptions)?;

    let n = input.calibration.len();
    let m = input.test.len();
    let h = (n as f64).sqrt().floor() as usize;
    let b = (n as f64).powf(input.policy.block_exponent_zeta).floor() as usize;
    let ell_b = if n > 0 {
        ((b as u128 * m as u128) / n as u128) as usize
    } else {
        0
    };

    let mut report = LongMemoryCoverageReportV1 {
        schema: LONG_MEMORY_COVERAGE_REPORT_SCHEMA_V1.into(),
        run_id: input.run_id.clone(),
        input_sha256: input_digest(input)?,
        source_stream_sha256: input.source_stream_sha256.clone(),
        calibration_manifest_sha256: input.calibration_manifest_sha256.clone(),
        test_manifest_sha256: input.test_manifest_sha256.clone(),
        status: LongMemoryCoverageStatusV1::AssumptionsNotSupported,
        limit_regime: None,
        assumptions_declared_but_not_proven: true,
        predictor_artifact_sha256: input.assumptions.predictor_artifact_sha256.clone(),
        target_coverage: input.policy.target_coverage,
        significance_level: input.policy.significance_level,
        calibration_count: n,
        test_count: m,
        calibration_rank: None,
        full_calibration_cutoff: None,
        observed_test_coverage: None,
        covered_test_count: 0,
        missed_test_count: 0,
        pilot_block_length_h: h,
        coverage_block_length_b: b,
        adjacent_test_block_length_ell_b: ell_b,
        paired_block_count: 0,
        beta_estimate: None,
        q_h: None,
        q_2h: None,
        beta_used_zero_variance_convention: false,
        normalization_r_b: None,
        normalization_r_n: None,
        block_distribution_lower_quantile: None,
        block_distribution_upper_quantile: None,
        normalized_full_window_statistic: None,
        realized_coverage_reference_lower: None,
        realized_coverage_reference_upper: None,
        rejects_nominal_coverage: None,
        normalized_block_distribution_sha256: None,
        long_memory_supported: false,
        iid_inference_claimed: false,
        population_claim_authorized_by_qualified_stream_alone: false,
        source_authenticity_proven: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };

    if n < input.policy.minimum_calibration_count || m < input.policy.minimum_test_count {
        report.status = LongMemoryCoverageStatusV1::InsufficientInput;
        report.report_sha256 = report_digest(&report)?;
        return Ok(report);
    }

    let (calibration_scores, test_scores, signed_residuals) = validate_adjacent_scores(
        &input.calibration,
        &input.test,
        &input.assumptions,
        input.cutoff_at_unix_seconds,
    )?;
    let calibration_rank = ((n as f64 + 1.0) * input.policy.target_coverage).ceil() as usize;
    report.calibration_rank = Some(calibration_rank);
    if calibration_rank > n {
        report.status = LongMemoryCoverageStatusV1::FullIntervalCutoff;
        report.covered_test_count = m;
        report.observed_test_coverage = Some(1.0);
        report.report_sha256 = report_digest(&report)?;
        return Ok(report);
    }
    report.full_calibration_cutoff = conformal_cutoff(&calibration_scores, input.policy.target_coverage)?;

    if h == 0
        || b < 2
        || ell_b == 0
        || b.checked_add(ell_b).is_none_or(|value| value > n)
        || 2_usize.checked_mul(h).is_none_or(|value| value > n)
    {
        report.status = LongMemoryCoverageStatusV1::BlockConstraintsNotSatisfied;
        report.report_sha256 = report_digest(&report)?;
        return Ok(report);
    }
    let block_rank = ((b as f64 + 1.0) * input.policy.target_coverage).ceil() as usize;
    if block_rank > b {
        report.status = LongMemoryCoverageStatusV1::BlockConstraintsNotSatisfied;
        report.report_sha256 = report_digest(&report)?;
        return Ok(report);
    }

    let (beta, q_h, q_2h, zero_convention) = estimate_memory_parameter(&signed_residuals, h)?;
    report.beta_estimate = Some(beta);
    report.q_h = Some(q_h);
    report.q_2h = Some(q_2h);
    report.beta_used_zero_variance_convention = zero_convention;
    if !beta.is_finite() || !(0.5..1.0).contains(&beta) {
        report.status = LongMemoryCoverageStatusV1::AssumptionsNotSupported;
        report.report_sha256 = report_digest(&report)?;
        return Ok(report);
    }
    report.limit_regime = Some(if beta > 0.75 {
        LongMemoryLimitRegimeV1::GaussianShortMemoryWithinLinearModel
    } else if beta == 0.75 {
        LongMemoryLimitRegimeV1::CriticalGaussianAtThreeQuarter
    } else {
        LongMemoryLimitRegimeV1::RosenblattNonGaussian
    });

    let r_b = long_memory_normalization(b, beta)?;
    let r_n = long_memory_normalization(n, beta)?;
    report.normalization_r_b = Some(r_b);
    report.normalization_r_n = Some(r_n);

    let full_cutoff = report.full_calibration_cutoff.unwrap();
    let covered = test_scores
        .iter()
        .filter(|score| **score <= full_cutoff)
        .count();
    let full_coverage = covered as f64 / m as f64;
    report.covered_test_count = covered;
    report.missed_test_count = m - covered;
    report.observed_test_coverage = Some(full_coverage);

    let mut normalized_blocks = Vec::with_capacity(n - b - ell_b + 1);
    for start in 0..=n - b - ell_b {
        let block_scores = &calibration_scores[start..start + b];
        let cutoff = conformal_cutoff(block_scores, input.policy.target_coverage)?
            .ok_or_else(|| LongMemoryCoverageErrorV1(
                "paired calibration block has an infinite conformal cutoff".into()
            ))?;
        let block_test_scores = &calibration_scores[start + b..start + b + ell_b];
        let block_coverage = block_test_scores
            .iter()
            .filter(|score| **score <= cutoff)
            .count() as f64
            / ell_b as f64;
        normalized_blocks.push((block_coverage - input.policy.target_coverage) / r_b);
    }
    report.paired_block_count = normalized_blocks.len();
    report.normalized_block_distribution_sha256 = Some(json_sha256(&normalized_blocks)?);

    let low_probability = input.policy.significance_level / 2.0;
    let high_probability = 1.0 - low_probability;
    let block_low = empirical_quantile(&normalized_blocks, low_probability)?;
    let block_high = empirical_quantile(&normalized_blocks, high_probability)?;
    report.block_distribution_lower_quantile = Some(block_low);
    report.block_distribution_upper_quantile = Some(block_high);

    let full_statistic = (full_coverage - input.policy.target_coverage) / r_n;
    let future_lower =
        (input.policy.target_coverage + r_n * block_low).clamp(0.0, 1.0);
    let future_upper =
        (input.policy.target_coverage + r_n * block_high).clamp(0.0, 1.0);
    report.normalized_full_window_statistic = Some(full_statistic);
    report.realized_coverage_reference_lower = Some(future_lower);
    report.realized_coverage_reference_upper = Some(future_upper);
    report.rejects_nominal_coverage = Some(full_statistic < block_low || full_statistic > block_high);
    report.long_memory_supported = true;
    report.status = LongMemoryCoverageStatusV1::Complete;
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critical_beta_normalization_uses_log_n_continuous_limit() {
        let n = 1024;
        let expected = ((1.0 + (n as f64).ln()) / n as f64).sqrt();
        let actual = long_memory_normalization(n, 0.75).unwrap();
        assert!((actual - expected).abs() < 1.0e-12);
    }

    #[test]
    fn normalization_is_stable_on_both_sides_of_critical_beta() {
        let below = long_memory_normalization(1 << 16, 0.74999999).unwrap();
        let critical = long_memory_normalization(1 << 16, 0.75).unwrap();
        let above = long_memory_normalization(1 << 16, 0.75000001).unwrap();
        assert!(below.is_finite() && critical.is_finite() && above.is_finite());
        assert!((below - critical).abs() < 1.0e-6);
        assert!((above - critical).abs() < 1.0e-6);
    }

    #[test]
    fn zero_two_scale_variance_uses_exact_three_quarter_convention() {
        let zero = vec![0.0; 32];
        let (beta, q_h, q_2h, used) = estimate_memory_parameter(&zero, 4).unwrap();
        assert_eq!(beta, 0.75);
        assert_eq!(q_h, 0.0);
        assert_eq!(q_2h, 0.0);
        assert!(used);
    }

    #[test]
    fn invalid_beta_is_not_clipped_into_the_supported_interval() {
        let residuals = (0..32)
            .map(|index| if index % 2 == 0 { 1.0 } else { -1.0 })
            .collect::<Vec<_>>();
        let (beta, _, _, used) = estimate_memory_parameter(&residuals, 4).unwrap();
        if !used && !(0.5..1.0).contains(&beta) {
            assert!(!beta.is_finite() || beta <= 0.5 || beta >= 1.0);
        }
    }

    #[test]
    fn quantile_uses_left_empirical_order_statistic() {
        let values = vec![-4.0, -1.0, 0.0, 2.0, 10.0];
        assert_eq!(empirical_quantile(&values, 0.2).unwrap(), -1.0);
        assert_eq!(empirical_quantile(&values, 0.8).unwrap(), 2.0);
    }

    #[test]
    fn q_variance_is_mean_of_overlapping_block_sum_squares() {
        let values = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let q = two_scale_block_variance(&values, 2).unwrap();
        // Four rolling 2-wide sums are 0, 0, 0, 0.
        assert_eq!(q, 0.0);
    }
}
