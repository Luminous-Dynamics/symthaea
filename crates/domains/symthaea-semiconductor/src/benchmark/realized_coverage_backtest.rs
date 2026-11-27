// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dependence-aware realized-coverage backtesting for frozen VM intervals.
//!
//! This module intentionally implements a named Newey-West/HAC Bartlett
//! baseline. It is not the paper-specific functional-dependence estimator and
//! does not provide long-memory inference. Its estimand is coverage among
//! K1-qualified outcomes in the supplied observed stream, not the wafer
//! population under selective metrology.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use thiserror::Error;

use super::gated_streaming_adapter::StreamingPredictionReceiptV1;
use super::label_consumption_guard::{
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
};

pub const REALIZED_COVERAGE_BACKTEST_SCHEMA_V1: &str =
    "luminous.semi-bench.realized-coverage-backtest.v1";
pub const REALIZED_COVERAGE_OBSERVATION_SCHEMA_V1: &str =
    "luminous.semi-bench.realized-coverage-observation.v1";
pub const REALIZED_COVERAGE_REPORT_SCHEMA_V1: &str =
    "luminous.semi-bench.realized-coverage-report.v1";
pub const REALIZED_COVERAGE_POLICY_SCHEMA_V1: &str =
    "luminous.semi-bench.realized-coverage-policy.v1";

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageEstimandV1 {
    QualifiedObservedStream,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageInferenceMethodV1 {
    HacBartlettAsymptotic,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageBacktestStatusV1 {
    Complete,
    InsufficientQualifiedOutcomes,
    AssumptionNotSupported,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageAssumptionStateV1 {
    ShortMemoryHacAssumptionDeclaredButNotProven,
    HacAssumptionNotDeclared,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoverageExclusionReasonV1 {
    K1Unknown,
    K1Blocked,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealizedCoveragePolicyV1 {
    pub schema: String,
    pub policy_id: String,
    pub revision: u64,
    /// Target marginal coverage in (0, 1).
    pub target_coverage: f64,
    /// Bartlett/Newey-West truncation lag. Must be less than n after filtering.
    pub hac_max_lag: usize,
    /// Two-sided normal critical value; 1.959963984540054 is conventional for 95%.
    pub confidence_z: f64,
    /// Minimum K1-eligible observations needed before inferential output is emitted.
    pub minimum_qualified_observations: usize,
    /// Explicitly selects the short-memory HAC interpretation. This is a declaration,
    /// not a statistical proof that the observed process is short-memory.
    pub short_memory_assumption_declared: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealizedCoverageObservationV1 {
    pub schema: String,
    pub prediction_receipt: StreamingPredictionReceiptV1,
    /// Frozen at prediction time. It is never reconstructed from a later model state.
    pub interval_lower: f64,
    pub interval_upper: f64,
    pub label_id: String,
    pub label_revision: u64,
    pub label_payload_sha256: String,
    pub outcome_value: f64,
    pub label_available_at_unix_seconds: i64,
    pub eligibility_report_available_at_unix_seconds: i64,
    pub delivered_at_unix_seconds: i64,
    pub clock_domain_id: String,
    pub eligibility_report: LabelConsumptionEligibilityReportV1,
    pub observation_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoverageBacktestInputV1 {
    pub schema: String,
    pub run_id: String,
    pub cutoff_at_unix_seconds: i64,
    pub source_stream_sha256: String,
    pub policy: RealizedCoveragePolicyV1,
    pub observations: Vec<RealizedCoverageObservationV1>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RealizedCoverageReportV1 {
    pub schema: String,
    pub run_id: String,
    pub input_sha256: String,
    pub source_stream_sha256: String,
    pub status: CoverageBacktestStatusV1,
    pub estimand: CoverageEstimandV1,
    pub inference_method: CoverageInferenceMethodV1,
    pub assumption_state: CoverageAssumptionStateV1,
    pub target_coverage: f64,
    pub observed_coverage: Option<f64>,
    pub qualified_count: usize,
    pub covered_count: usize,
    pub missed_count: usize,
    pub excluded_unknown_count: usize,
    pub excluded_blocked_count: usize,
    pub late_or_unavailable_count: usize,
    pub iid_standard_error_diagnostic: Option<f64>,
    pub hac_standard_error: Option<f64>,
    pub hac_confidence_interval_lower: Option<f64>,
    pub hac_confidence_interval_upper: Option<f64>,
    pub hac_test_statistic: Option<f64>,
    pub hac_critical_value: f64,
    pub hac_rejects_target: Option<bool>,
    pub hac_max_lag: usize,
    pub qualified_prediction_sequence_first: Option<u64>,
    pub qualified_prediction_sequence_last: Option<u64>,
    pub iid_inference_claimed: bool,
    pub long_memory_supported: bool,
    pub population_claim_authorized_by_qualified_subset_alone: bool,
    pub source_authenticity_proven: bool,
    pub equipment_control_authority: bool,
    pub report_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Error)]
#[error("{0}")]
pub struct RealizedCoverageBacktestErrorV1(pub String);

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn json_sha256<T: Serialize + ?Sized>(
    value: &T,
) -> Result<String, RealizedCoverageBacktestErrorV1> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        RealizedCoverageBacktestErrorV1(format!("canonical JSON serialization failed: {error}"))
    })?;
    Ok(sha256_hex(&bytes))
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn prediction_receipt_digest(
    receipt: &StreamingPredictionReceiptV1,
) -> Result<String, RealizedCoverageBacktestErrorV1> {
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    json_sha256(&canonical)
}

fn eligibility_report_digest(
    report: &LabelConsumptionEligibilityReportV1,
) -> Result<String, RealizedCoverageBacktestErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn observation_digest(
    observation: &RealizedCoverageObservationV1,
) -> Result<String, RealizedCoverageBacktestErrorV1> {
    let mut canonical = observation.clone();
    canonical.observation_sha256.clear();
    json_sha256(&canonical)
}

fn input_digest(input: &CoverageBacktestInputV1) -> Result<String, RealizedCoverageBacktestErrorV1> {
    let mut canonical = input.clone();
    canonical.observations.sort_by_key(|row| row.prediction_receipt.sequence);
    json_sha256(&canonical)
}

fn report_digest(
    report: &RealizedCoverageReportV1,
) -> Result<String, RealizedCoverageBacktestErrorV1> {
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    json_sha256(&canonical)
}

fn require_text(value: &str, field: &str) -> Result<(), RealizedCoverageBacktestErrorV1> {
    if value.trim().is_empty() {
        return Err(RealizedCoverageBacktestErrorV1(format!(
            "{field}: expected non-empty text"
        )));
    }
    Ok(())
}

fn validate_policy(policy: &RealizedCoveragePolicyV1) -> Result<(), RealizedCoverageBacktestErrorV1> {
    if policy.schema != REALIZED_COVERAGE_POLICY_SCHEMA_V1 {
        return Err(RealizedCoverageBacktestErrorV1(
            "coverage policy schema is unsupported".into(),
        ));
    }
    require_text(&policy.policy_id, "policy.policy_id")?;
    if policy.revision == 0
        || !policy.target_coverage.is_finite()
        || !(0.0..1.0).contains(&policy.target_coverage)
        || !policy.confidence_z.is_finite()
        || policy.confidence_z <= 0.0
        || policy.minimum_qualified_observations < 2
    {
        return Err(RealizedCoverageBacktestErrorV1(
            "coverage policy contains an invalid target, critical value, revision or minimum n"
                .into(),
        ));
    }
    Ok(())
}

fn validate_observation(
    row: &RealizedCoverageObservationV1,
) -> Result<(), RealizedCoverageBacktestErrorV1> {
    if row.schema != REALIZED_COVERAGE_OBSERVATION_SCHEMA_V1 {
        return Err(RealizedCoverageBacktestErrorV1(
            "coverage observation schema is unsupported".into(),
        ));
    }
    require_text(&row.label_id, "observation.label_id")?;
    require_text(&row.clock_domain_id, "observation.clock_domain_id")?;
    if row.label_revision == 0
        || !valid_sha256(&row.label_payload_sha256)
        || !row.interval_lower.is_finite()
        || !row.interval_upper.is_finite()
        || row.interval_lower > row.interval_upper
        || !row.outcome_value.is_finite()
    {
        return Err(RealizedCoverageBacktestErrorV1(
            "coverage observation has invalid interval, value, revision or payload digest".into(),
        ));
    }
    if row.observation_sha256 != observation_digest(row)? {
        return Err(RealizedCoverageBacktestErrorV1(
            "coverage observation digest does not match its content".into(),
        ));
    }
    let prediction = &row.prediction_receipt;
    if !valid_sha256(&prediction.receipt_sha256)
        || prediction_receipt_digest(prediction)? != prediction.receipt_sha256
        || !valid_sha256(&prediction.global_model_artifact_sha256)
        || !valid_sha256(&prediction.feature_schema_sha256)
        || !valid_sha256(&prediction.policy_sha256)
    {
        return Err(RealizedCoverageBacktestErrorV1(
            "frozen prediction receipt or model/schema/policy digest is invalid".into(),
        ));
    }
    let report = &row.eligibility_report;
    if !valid_sha256(&report.report_sha256)
        || eligibility_report_digest(report)? != report.report_sha256
        || !valid_sha256(&report.label_sha256)
        || report.label_sha256 != row.label_payload_sha256
        || report.label_id != row.label_id
        || report.label_revision != row.label_revision
        || report.measurement_event_id != prediction.measurement_event_id
    {
        return Err(RealizedCoverageBacktestErrorV1(
            "K1 eligibility receipt does not bind the delivered label, revision, event and payload"
                .into(),
        ));
    }
    if row.clock_domain_id != prediction.clock_domain_id
        || prediction.event_time_unix_seconds > prediction.feature_available_at_unix_seconds
        || prediction.feature_available_at_unix_seconds > prediction.prediction_at_unix_seconds
        || row.label_available_at_unix_seconds < prediction.prediction_at_unix_seconds
        || row.eligibility_report_available_at_unix_seconds
            < row.label_available_at_unix_seconds
        || row.delivered_at_unix_seconds < row.eligibility_report_available_at_unix_seconds
        || row.delivered_at_unix_seconds < row.label_available_at_unix_seconds
        || row.delivered_at_unix_seconds < prediction.prediction_at_unix_seconds
    {
        return Err(RealizedCoverageBacktestErrorV1(
            "coverage outcome/report availability chronology or cutoff is invalid".into(),
        ));
    }
    if !report.qualified_corrected_value.is_finite()
        || report.qualified_corrected_value != row.outcome_value
        || report.equipment_control_authority
        || report.training_snapshot_mutated
        || report
            .qualified_standard_uncertainty_upper_bound
            .is_none_or(|value| !value.is_finite() || value < 0.0)
    {
        return Err(RealizedCoverageBacktestErrorV1(
            "outcome differs from the qualified corrected measurement value".into(),
        ));
    }
    Ok(())
}

fn normal_two_sided_critical_value(z: f64) -> f64 {
    z
}

/// Newey-West long-run variance of the Bernoulli miss-indicator mean with
/// Bartlett weights. This is a named HAC baseline, not the functional-
—dependence estimator in arXiv:2609.33868.
fn hac_bartlett_standard_error(
    misses: &[f64],
    maximum_lag: usize,
) -> Result<f64, RealizedCoverageBacktestErrorV1> {
    let n = misses.len();
    if n < 2 || maximum_lag >= n {
        return Err(RealizedCoverageBacktestErrorV1(
            "HAC lag must be smaller than the qualified sample size".into(),
        ));
    }
    let mean = misses.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = misses.iter().map(|value| value - mean).collect();
    let mut long_run_variance = centered.iter().map(|value| value * value).sum::<f64>() / n as f64;
    for lag in 1..=maximum_lag {
        let covariance = centered
            .iter()
            .skip(lag)
            .zip(centered.iter())
            .map(|(left, right)| left * right)
            .sum::<f64>()
            / n as f64;
        let weight = 1.0 - lag as f64 / (maximum_lag as f64 + 1.0);
        long_run_variance += 2.0 * weight * covariance;
    }
    if !long_run_variance.is_finite() {
        return Err(RealizedCoverageBacktestErrorV1(
            "HAC long-run variance is non-finite".into(),
        ));
    }
    // A negative truncated estimate is possible in finite samples. Preserve
    // the raw estimate as a no-inference state rather than silently clipping it.
    if long_run_variance < 0.0 {
        return Err(RealizedCoverageBacktestErrorV1(
            "HAC long-run variance is negative; no inferential result emitted".into(),
        ));
    }
    Ok((long_run_variance / n as f64).sqrt())
}

fn iid_standard_error(coverage: f64, n: usize) -> f64 {
    (coverage * (1.0 - coverage) / n as f64).sqrt()
}

/// Computes qualified-stream realized coverage from immutable prediction
/// intervals and K1-eligible outcomes. Non-qualified outcomes are counted but
/// excluded from the qualified-stream estimand.
pub fn evaluate_realized_coverage_backtest_v1(
    input: &CoverageBacktestInputV1,
) -> Result<RealizedCoverageReportV1, RealizedCoverageBacktestErrorV1> {
    if input.schema != REALIZED_COVERAGE_BACKTEST_SCHEMA_V1 {
        return Err(RealizedCoverageBacktestErrorV1(
            "coverage backtest input schema is unsupported".into(),
        ));
    }
    require_text(&input.run_id, "input.run_id")?;
    if !valid_sha256(&input.source_stream_sha256) {
        return Err(RealizedCoverageBacktestErrorV1(
            "source stream digest is malformed".into(),
        ));
    }
    validate_policy(&input.policy)?;

    let mut sample_ids = BTreeSet::new();
    let mut event_ids = BTreeSet::new();
    let mut label_revisions = BTreeSet::new();
    let mut observations = input.observations.clone();
    observations.sort_by_key(|row| row.prediction_receipt.sequence);

    let mut qualified = Vec::new();
    let mut excluded_unknown_count = 0;
    let mut excluded_blocked_count = 0;
    let mut late_or_unavailable_count = 0;

    for row in &observations {
        validate_observation(row)?;
        if !sample_ids.insert(row.prediction_receipt.sample_id.as_str())
            || !event_ids.insert(row.prediction_receipt.measurement_event_id.as_str())
            || !label_revisions.insert((row.label_id.as_str(), row.label_revision))
            || !prediction_sequences.insert(row.prediction_receipt.sequence)
        {
            return Err(RealizedCoverageBacktestErrorV1(
                "duplicate sample, measurement event, label revision or prediction sequence".into(),
            ));
        }
        if previous_event_time.is_some_and(|time| {
            row.prediction_receipt.event_time_unix_seconds < time
        }) {
            return Err(RealizedCoverageBacktestErrorV1(
                "prediction event time regresses in sequence order".into(),
            ));
        }
        previous_event_time = Some(row.prediction_receipt.event_time_unix_seconds);
        if row.delivered_at_unix_seconds > input.cutoff_at_unix_seconds {
            late_or_unavailable_count += 1;
            continue;
        }
        match row.eligibility_report.eligibility {
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy
                if row.eligibility_report.frontier == LabelEvidenceFrontierV1::CurrentAtK1
                    && row.eligibility_report.downstream_consumption_allowed =>
            {
                qualified.push(row);
            }
            LabelConsumptionEligibilityV1::Unknown => excluded_unknown_count += 1,
            LabelConsumptionEligibilityV1::Blocked => excluded_blocked_count += 1,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy => {
                late_or_unavailable_count += 1
            }
        }
    }

    let n = qualified.len();
    let mut covered_count = 0;
    let mut misses = Vec::with_capacity(n);
    let mut first_sequence = None;
    let mut last_sequence = None;
    for row in &qualified {
        let prediction = &row.prediction_receipt;
        if first_sequence.is_none() {
            first_sequence = Some(prediction.sequence);
        }
        last_sequence = Some(prediction.sequence);
        let covered = row.outcome_value >= row.interval_lower
            && row.outcome_value <= row.interval_upper;
        if covered {
            covered_count += 1;
            misses.push(0.0);
        } else {
            misses.push(1.0);
        }
    }

    let mut report = RealizedCoverageReportV1 {
        schema: REALIZED_COVERAGE_REPORT_SCHEMA_V1.into(),
        run_id: input.run_id.clone(),
        input_sha256: input_digest(input)?,
        source_stream_sha256: input.source_stream_sha256.clone(),
        status: if !input.policy.short_memory_assumption_declared {
            CoverageBacktestStatusV1::AssumptionNotSupported
        } else if n < input.policy.minimum_qualified_observations
            || input.policy.hac_max_lag >= n
        {
            CoverageBacktestStatusV1::InsufficientQualifiedOutcomes
        } else {
            CoverageBacktestStatusV1::Complete
        },
        estimand: CoverageEstimandV1::QualifiedObservedStream,
        inference_method: CoverageInferenceMethodV1::HacBartlettAsymptotic,
        assumption_state: if input.policy.short_memory_assumption_declared {
            CoverageAssumptionStateV1::ShortMemoryHacAssumptionDeclaredButNotProven
        } else {
            CoverageAssumptionStateV1::HacAssumptionNotDeclared
        },
        target_coverage: input.policy.target_coverage,
        observed_coverage: None,
        qualified_count: n,
        covered_count,
        missed_count: n - covered_count,
        excluded_unknown_count,
        excluded_blocked_count,
        late_or_unavailable_count,
        iid_standard_error_diagnostic: None,
        hac_standard_error: None,
        hac_confidence_interval_lower: None,
        hac_confidence_interval_upper: None,
        hac_test_statistic: None,
        hac_critical_value: normal_two_sided_critical_value(input.policy.confidence_z),
        hac_rejects_target: None,
        hac_max_lag: input.policy.hac_max_lag,
        qualified_prediction_sequence_first: first_sequence,
        qualified_prediction_sequence_last: last_sequence,
        iid_inference_claimed: false,
        long_memory_supported: false,
        population_claim_authorized_by_qualified_subset_alone: false,
        source_authenticity_proven: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };

    if report.status == CoverageBacktestStatusV1::Complete {
        let coverage = covered_count as f64 / n as f64;
        let se_iid = iid_standard_error(coverage, n);
        let se_hac = hac_bartlett_standard_error(&misses, input.policy.hac_max_lag)?;
        let lower = (coverage - report.hac_critical_value * se_hac).max(0.0);
        let upper = (coverage + report.hac_critical_value * se_hac).min(1.0);
        let statistic = if se_hac > 0.0 {
            Some((coverage - input.policy.target_coverage) / se_hac)
        } else {
            None
        };
        report.observed_coverage = Some(coverage);
        report.iid_standard_error_diagnostic = Some(se_iid);
        report.hac_standard_error = Some(se_hac);
        report.hac_confidence_interval_lower = Some(lower);
        report.hac_confidence_interval_upper = Some(upper);
        report.hac_test_statistic = statistic;
        report.hac_rejects_target =
            statistic.map(|value| value.abs() > report.hac_critical_value);
    }
    report.report_sha256 = report_digest(&report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn prediction(
        sequence: u64,
        sample: &str,
        event: &str,
        event_time: i64,
    ) -> StreamingPredictionReceiptV1 {
        let mut receipt = StreamingPredictionReceiptV1 {
            schema: super::super::gated_streaming_adapter::STREAMING_PREDICTION_RECEIPT_SCHEMA_V1
                .into(),
            sequence,
            sample_id: sample.into(),
            measurement_event_id: event.into(),
            event_time_unix_seconds: event_time,
            feature_available_at_unix_seconds: event_time + 1,
            prediction_at_unix_seconds: event_time + 2,
            clock_domain_id: "fab-clock".into(),
            global_model_artifact_sha256: digest('a'),
            feature_schema_sha256: digest('b'),
            policy_sha256: digest('c'),
            context_key: Some("tool-a/ch-1".into()),
            feature_vector_sha256: digest('d'),
            selected_feature_indices: vec![0],
            selected_feature_vector: vec![1.0],
            global_baseline_prediction: 10.0,
            feature_adapter_prediction: 0.0,
            context_correction: 0.0,
            temporal_correction: 0.0,
            base_prediction: 10.0,
            prediction_value: 10.0,
            adapter_state_before_sha256: digest('e'),
            adapter_state_after_sha256: digest('e'),
            receipt_sha256: String::new(),
        };
        receipt.receipt_sha256 = prediction_receipt_digest(&receipt).unwrap();
        receipt
    }

    fn eligibility(
        label_id: &str,
        revision: u64,
        event: &str,
        label_value: f64,
        payload: &str,
        status: LabelConsumptionEligibilityV1,
        frontier: LabelEvidenceFrontierV1,
    ) -> LabelConsumptionEligibilityReportV1 {
        let mut report = LabelConsumptionEligibilityReportV1 {
            schema: "luminous.semi-bench.label-consumption-report.v1".into(),
            label_id: label_id.into(),
            label_revision: revision,
            measurement_event_id: event.into(),
            calibration_certificate_id: "certificate-1".into(),
            frontier,
            eligibility: status,
            downstream_consumption_allowed:
                status == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
            reasons: Vec::new(),
            label_sha256: payload.into(),
            measurement_sha256: digest('f'),
            qualified_corrected_value: label_value,
            qualified_standard_uncertainty_upper_bound: Some(0.1),
            qualification_report_sha256: digest('a'),
            requalification_report_sha256: digest('b'),
            requalification_frontier_sha256: digest('c'),
            source_authenticity_proven: false,
            training_snapshot_mutated: false,
            equipment_control_authority: false,
            report_sha256: String::new(),
        };
        report.report_sha256 = eligibility_report_digest(&report).unwrap();
        report
    }

    fn observation(
        sequence: u64,
        status: LabelConsumptionEligibilityV1,
        value: f64,
    ) -> RealizedCoverageObservationV1 {
        let sample = format!("sample-{sequence}");
        let event = format!("event-{sequence}");
        let label_id = format!("label-{sequence}");
        let payload = digest('d');
        let mut row = RealizedCoverageObservationV1 {
            schema: REALIZED_COVERAGE_OBSERVATION_SCHEMA_V1.into(),
            prediction_receipt: prediction(sequence, &sample, &event, sequence as i64 * 10),
            interval_lower: 9.0,
            interval_upper: 11.0,
            label_id: label_id.clone(),
            label_revision: 1,
            label_payload_sha256: payload.clone(),
            outcome_value: value,
            label_available_at_unix_seconds: sequence as i64 * 10 + 5,
            eligibility_report_available_at_unix_seconds: sequence as i64 * 10 + 6,
            delivered_at_unix_seconds: sequence as i64 * 10 + 7,
            clock_domain_id: "fab-clock".into(),
            eligibility_report: eligibility(
                &label_id,
                1,
                &event,
                value,
                &payload,
                status,
                LabelEvidenceFrontierV1::CurrentAtK1,
            ),
            observation_sha256: String::new(),
        };
        row.observation_sha256 = observation_digest(&row).unwrap();
        row
    }

    fn input(rows: Vec<RealizedCoverageObservationV1>, hac_max_lag: usize) -> CoverageBacktestInputV1 {
        CoverageBacktestInputV1 {
            schema: REALIZED_COVERAGE_BACKTEST_SCHEMA_V1.into(),
            run_id: "coverage-run-001".into(),
            cutoff_at_unix_seconds: 100_000,
            source_stream_sha256: digest('f'),
            policy: RealizedCoveragePolicyV1 {
                schema: REALIZED_COVERAGE_POLICY_SCHEMA_V1.into(),
                policy_id: "hac-bartlett-v1".into(),
                revision: 1,
                target_coverage: 0.95,
                hac_max_lag,
                confidence_z: 1.959963984540054,
                minimum_qualified_observations: 2,
                short_memory_assumption_declared: true,
            },
            observations: rows,
        }
    }

    #[test]
    fn all_covered_stream_returns_one_hundred_percent_with_explicit_zero_hac_error() {
        let rows = (1..=12)
            .map(|sequence| observation(sequence, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0))
            .collect();
        let report = evaluate_realized_coverage_backtest_v1(&input(rows, 2)).unwrap();
        assert_eq!(report.status, CoverageBacktestStatusV1::Complete);
        assert_eq!(report.observed_coverage, Some(1.0));
        assert_eq!(report.covered_count, 12);
        assert_eq!(report.missed_count, 0);
        assert_eq!(report.hac_standard_error, Some(0.0));
        assert_eq!(report.hac_rejects_target, None);
    }

    #[test]
    fn frozen_interval_misses_are_scored_without_reconstructing_predictions() {
        let rows = vec![
            observation(1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0),
            observation(2, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 12.0),
            observation(3, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 8.0),
            observation(4, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.5),
            observation(5, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0),
            observation(6, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 9.5),
        ];
        let report = evaluate_realized_coverage_backtest_v1(&input(rows, 1)).unwrap();
        assert_eq!(report.qualified_count, 6);
        assert_eq!(report.covered_count, 4);
        assert_eq!(report.missed_count, 2);
        assert!((report.observed_coverage.unwrap() - 4.0 / 6.0).abs() < 1.0e-12);
        assert!(!report.population_claim_authorized_by_qualified_subset_alone);
        assert!(!report.iid_inference_claimed);
    }

    #[test]
    fn positive_serial_dependence_increases_hac_uncertainty_over_iid_diagnostic() {
        let misses = vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        let coverage = 1.0 - misses.iter().sum::<f64>() / misses.len() as f64;
        let iid = iid_standard_error(coverage, misses.len());
        let hac = hac_bartlett_standard_error(&misses, 3).unwrap();
        assert!(hac > iid);
    }

    #[test]
    fn unknown_and_blocked_labels_are_excluded_from_qualified_stream_denominator() {
        let rows = vec![
            observation(1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0),
            observation(2, LabelConsumptionEligibilityV1::Unknown, 12.0),
            observation(3, LabelConsumptionEligibilityV1::Blocked, 10.0),
            observation(4, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0),
            observation(5, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0),
        ];
        let report = evaluate_realized_coverage_backtest_v1(&input(rows, 1)).unwrap();
        assert_eq!(report.qualified_count, 3);
        assert_eq!(report.excluded_unknown_count, 1);
        assert_eq!(report.excluded_blocked_count, 1);
        assert_eq!(report.observed_coverage, Some(1.0));
        assert_eq!(report.estimand, CoverageEstimandV1::QualifiedObservedStream);
    }

    #[test]
    fn underpowered_qualified_stream_returns_insufficient_not_iid_fallback() {
        let rows = vec![
            observation(1, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0),
            observation(2, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0),
        ];
        let mut value = input(rows, 2);
        value.policy.minimum_qualified_observations = 5;
        let report = evaluate_realized_coverage_backtest_v1(&value).unwrap();
        assert_eq!(report.status, CoverageBacktestStatusV1::InsufficientQualifiedOutcomes);
        assert_eq!(report.hac_standard_error, None);
        assert_eq!(report.observed_coverage, None);
    }

    #[test]
    fn long_memory_claim_is_not_emitted_by_hac_baseline() {
        let rows = (1..=10)
            .map(|sequence| observation(sequence, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0))
            .collect();
        let mut value = input(rows, 2);
        value.policy.short_memory_assumption_declared = false;
        let report = evaluate_realized_coverage_backtest_v1(&value).unwrap();
        assert_eq!(report.status, CoverageBacktestStatusV1::AssumptionNotSupported);
        assert_eq!(report.assumption_state, CoverageAssumptionStateV1::HacAssumptionNotDeclared);
        assert_eq!(report.hac_standard_error, None);
        assert!(!report.long_memory_supported);
    }

    #[test]
    fn tampered_interval_receipt_or_label_qualification_receipt_is_rejected() {
        let rows = (1..=6)
            .map(|sequence| observation(sequence, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0))
            .collect();
        let mut value = input(rows, 1);
        value.observations[0].interval_upper = 12.0;
        assert!(evaluate_realized_coverage_backtest_v1(&value).is_err());

        let rows = (1..=6)
            .map(|sequence| observation(sequence, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0))
            .collect();
        let mut value = input(rows, 1);
        value.observations[0].eligibility_report.report_sha256 = digest('0');
        value.observations[0].observation_sha256 = observation_digest(&value.observations[0]).unwrap();
        assert!(evaluate_realized_coverage_backtest_v1(&value).is_err());
    }

    #[test]
    fn report_availability_must_follow_physical_label_availability() {
        let rows = (1..=6)
            .map(|sequence| observation(sequence, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0))
            .collect();
        let mut value = input(rows, 1);
        value.observations[0].eligibility_report_available_at_unix_seconds =
            value.observations[0].label_available_at_unix_seconds - 1;
        value.observations[0].observation_sha256 = observation_digest(&value.observations[0]).unwrap();
        assert!(evaluate_realized_coverage_backtest_v1(&value).is_err());
    }

    #[test]
    fn outcomes_delivered_after_the_frozen_cutoff_are_counted_but_not_used() {
        let mut rows = (1..=6)
            .map(|sequence| {
                observation(
                    sequence,
                    LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
                    10.0,
                )
            })
            .collect::<Vec<_>>();
        rows[5].delivered_at_unix_seconds = 100_001;
        rows[5].eligibility_report_available_at_unix_seconds = 100_000;
        rows[5].observation_sha256 = observation_digest(&rows[5]).unwrap();
        let mut value = input(rows, 1);
        value.cutoff_at_unix_seconds = 100_000;
        let report = evaluate_realized_coverage_backtest_v1(&value).unwrap();
        assert_eq!(report.qualified_count, 5);
        assert_eq!(report.late_or_unavailable_count, 1);
    }

    #[test]
    fn duplicate_sample_event_or_label_revision_is_rejected() {
        let mut rows = (1..=6)
            .map(|sequence| observation(sequence, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0))
            .collect::<Vec<_>>();
        rows[1].prediction_receipt.sample_id = rows[0].prediction_receipt.sample_id.clone();
        rows[1].prediction_receipt.receipt_sha256 =
            prediction_receipt_digest(&rows[1].prediction_receipt).unwrap();
        rows[1].observation_sha256 = observation_digest(&rows[1]).unwrap();
        assert!(evaluate_realized_coverage_backtest_v1(&input(rows, 1)).is_err());
    }

    #[test]
    fn deterministic_input_permutation_has_identical_receipt() {
        let rows = (1..=10)
            .map(|sequence| observation(sequence, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0 + f64::from(sequence % 3 == 0)))
            .collect::<Vec<_>>();
        let first = evaluate_realized_coverage_backtest_v1(&input(rows.clone(), 2)).unwrap();
        let mut permuted = rows;
        permuted.reverse();
        let second = evaluate_realized_coverage_backtest_v1(&input(permuted, 2)).unwrap();
        assert_eq!(first.report_sha256, second.report_sha256);
    }

    #[test]
    fn hac_standard_error_is_never_reported_as_the_paper_specific_estimator() {
        let rows = (1..=12)
            .map(|sequence| observation(sequence, LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy, 10.0))
            .collect();
        let report = evaluate_realized_coverage_backtest_v1(&input(rows, 2)).unwrap();
        assert_eq!(
            report.inference_method,
            CoverageInferenceMethodV1::HacBartlettAsymptotic
        );
        assert_eq!(
            report.assumption_state,
            CoverageAssumptionStateV1::ShortMemoryHacAssumptionDeclaredButNotProven
        );
    }
}
