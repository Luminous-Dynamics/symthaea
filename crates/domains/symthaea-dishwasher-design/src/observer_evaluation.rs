//! Run-level risk/coverage evaluation for outcome observers.
//!
//! Evaluates frozen per-item/per-endpoint pass probabilities against
//! independently adjudicated physical-run outcomes. A physical wash run is the
//! statistical unit: items and endpoints are nested observations.
//!
//! This module does not train or calibrate a model, authenticate source
//! archives, prove that assay labels are correct, estimate cluster correlation,
//! or authorize appliance release. Its zero-event Clopper-Pearson diagnostic is
//! valid only under an IID challenge-run assumption; grouped holdout checks do
//! not make runs within a partition independent.

use std::collections::{BTreeMap, BTreeSet};

use crate::quality::{Decision, Endpoint};
use crate::validation::{
    audit_grouped_holdout, GroupedValidationError, GroupedValidationReport, HoldoutAxis,
    RunPartitionRecord, ValidationPartition,
};

#[derive(Debug, Clone, PartialEq)]
pub struct EndpointPrediction {
    pub item_id: String,
    pub endpoint: Endpoint,
    /// Calibrated probability that this item/endpoint will pass its criterion.
    /// Calibration must be established on separate data by the caller.
    pub pass_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EndpointOutcome {
    pub item_id: String,
    pub endpoint: Endpoint,
    /// Adjudicated outcome from the independent endpoint assay.
    pub decision: Decision,
    /// Opaque ID of the upstream assay record; it is not authenticated here.
    pub assay_record_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObserverRunScores {
    pub run_id: String,
    pub item_ids: Vec<String>,
    /// Exact frozen model identity shared by every run in this evaluation.
    pub model_id: String,
    pub model_sha256: String,
    pub calibration_bundle_sha256: String,
    /// One probability for every item × required endpoint pair.
    pub predictions: Vec<EndpointPrediction>,
    /// One independently adjudicated outcome for every item × required endpoint pair.
    pub outcomes: Vec<EndpointOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunTruth {
    Conforming,
    Nonconforming,
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObserverEvaluationError {
    GroupedValidation(GroupedValidationError),
    UnsupportedTargetPartition,
    EmptyRequiredEndpoints,
    DuplicateRequiredEndpoint(Endpoint),
    EmptyThresholds,
    InvalidThreshold,
    DuplicateThreshold,
    InvalidAlpha,
    InvalidIdentity(&'static str),
    InvalidDigest(&'static str),
    ModelIdentityMismatch,
    DuplicateEvaluationRun(String),
    MissingEvaluationRun(String),
    UnknownEvaluationRun(String),
    EvaluationRunWrongPartition(String),
    EmptyItemSet(String),
    DuplicateItem { run_id: String, item_id: String },
    UnexpectedItem { run_id: String, item_id: String },
    UnexpectedEndpoint { run_id: String, endpoint: Endpoint },
    DuplicatePrediction { run_id: String, item_id: String, endpoint: Endpoint },
    DuplicateOutcome { run_id: String, item_id: String, endpoint: Endpoint },
    MissingPrediction { run_id: String, item_id: String, endpoint: Endpoint },
    MissingOutcome { run_id: String, item_id: String, endpoint: Endpoint },
    InvalidProbability { run_id: String, item_id: String, endpoint: Endpoint },
}

#[derive(Debug, Clone, PartialEq)]
pub struct EndpointThresholdMetrics {
    pub endpoint: Endpoint,
    /// A run fails this endpoint if any required item fails it.
    pub nonconforming_run_count: usize,
    /// Such a run is a false accept when the overall run is released.
    pub false_accept_run_count: usize,
    pub unresolved_run_count: usize,
    pub challenge_false_accept_rate: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObserverThresholdMetrics {
    pub threshold: f64,
    pub physical_run_count: usize,
    pub conforming_run_count: usize,
    pub nonconforming_run_count: usize,
    pub unresolved_run_count: usize,
    /// A run is released when every item × endpoint probability is >= threshold.
    pub released_run_count: usize,
    pub verified_released_run_count: usize,
    pub conforming_released_run_count: usize,
    pub false_accept_run_count: usize,
    /// The model would release a run whose independent assay result is unresolved.
    pub unverifiable_release_run_count: usize,
    pub conforming_not_released_run_count: usize,
    /// Released runs / all target-partition physical runs, including unresolved runs.
    pub release_coverage: f64,
    /// False-accepted known-failing challenge runs / all known-failing challenge runs.
    pub challenge_false_accept_rate: Option<f64>,
    /// False-accepted runs / released runs with adjudicable ground truth.
    /// This is prevalence-dependent and interpretable operationally only on a
    /// representative operating-population sample.
    pub false_release_rate_among_verified_releases: Option<f64>,
    /// Conforming runs released / all conforming runs.
    pub conforming_release_rate: Option<f64>,
    /// Exact one-sided zero-event Clopper-Pearson upper bound for the
    /// challenge false-accept probability, returned only when zero events occur.
    /// IID challenge runs are required. This is not a cluster-adjusted bound.
    pub iid_zero_event_challenge_upper_bound: Option<f64>,
    pub endpoint_metrics: Vec<EndpointThresholdMetrics>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObserverBenchmarkReport {
    pub target_partition: ValidationPartition,
    pub target_run_count: usize,
    pub alpha: f64,
    pub model_id: String,
    pub model_sha256: String,
    pub calibration_bundle_sha256: String,
    pub grouped_validation: GroupedValidationReport,
    /// Sorted thresholds, suitable for constructing a risk/coverage curve.
    pub points: Vec<ObserverThresholdMetrics>,
}

#[derive(Debug, Clone)]
struct PreparedRun {
    truth: RunTruth,
    minimum_endpoint_probability: f64,
    endpoint_truth: BTreeMap<Endpoint, RunTruth>,
}

fn canonical_non_empty(value: &str) -> bool {
    !value.trim().is_empty() && value.trim() == value
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

/// Evaluate a frozen observer against one locked or external physical-run split.
///
/// The run manifest must contain development, calibration, and target records
/// (plus locked confirmation when the target is external replication). The
/// caller explicitly names holdout axes; all declared groups must be disjoint
/// across partitions. Evaluation scores must contain exactly one complete
/// item × endpoint prediction/outcome matrix for every target-partition run.
///
/// A run is released at threshold t only if every item × endpoint pass
/// probability is at least t. The minimum endpoint probability is a conservative
/// decision rule; it is NOT the joint probability that the whole load passes.
///
/// The function reports challenge false acceptance separately from false
/// releases among adjudicable released runs. Unresolved assay outcomes are
/// never silently treated as passes. A run with unresolved truth that crosses
/// the threshold is counted as an unverifiable release.
///
/// The zero-event confidence bound is exact Clopper-Pearson for a binomial
/// count, but the IID assumption is strong. Holdout group separation prevents
/// cross-partition leakage; it does not remove within-partition clustering.
/// Use this bound only when the target challenge-run sampling plan justifies
/// independent physical runs. For clustered studies, use preregistered
/// cluster-aware inference instead.
pub fn evaluate_observer_risk_coverage(
    run_manifest: &[RunPartitionRecord],
    evaluation_runs: &[ObserverRunScores],
    holdout_axes: &[HoldoutAxis],
    target_partition: ValidationPartition,
    required_endpoints: &[Endpoint],
    thresholds: &[f64],
    alpha: f64,
) -> Result<ObserverBenchmarkReport, ObserverEvaluationError> {
    if !matches!(
        target_partition,
        ValidationPartition::LockedConfirmation | ValidationPartition::ExternalReplication
    ) {
        return Err(ObserverEvaluationError::UnsupportedTargetPartition);
    }
    if required_endpoints.is_empty() {
        return Err(ObserverEvaluationError::EmptyRequiredEndpoints);
    }
    if thresholds.is_empty() {
        return Err(ObserverEvaluationError::EmptyThresholds);
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(ObserverEvaluationError::InvalidAlpha);
    }

    let mut required = BTreeSet::new();
    for endpoint in required_endpoints {
        if !required.insert(*endpoint) {
            return Err(ObserverEvaluationError::DuplicateRequiredEndpoint(*endpoint));
        }
    }

    let mut sorted_thresholds = thresholds.to_vec();
    for threshold in &mut sorted_thresholds {
        if !threshold.is_finite() || !(0.0..=1.0).contains(threshold) {
            return Err(ObserverEvaluationError::InvalidThreshold);
        }
        // Normalize negative zero so duplicate detection is deterministic.
        if *threshold == 0.0 {
            *threshold = 0.0;
        }
    }
    sorted_thresholds.sort_by(f64::total_cmp);
    if sorted_thresholds.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err(ObserverEvaluationError::DuplicateThreshold);
    }

    let required_partitions = if target_partition == ValidationPartition::ExternalReplication {
        vec![
            ValidationPartition::Development,
            ValidationPartition::Calibration,
            ValidationPartition::LockedConfirmation,
            ValidationPartition::ExternalReplication,
        ]
    } else {
        vec![
            ValidationPartition::Development,
            ValidationPartition::Calibration,
            ValidationPartition::LockedConfirmation,
        ]
    };
    let grouped_validation = audit_grouped_holdout(
        run_manifest,
        holdout_axes,
        &required_partitions,
    )
    .map_err(ObserverEvaluationError::GroupedValidation)?;

    let mut partition_by_run = BTreeMap::new();
    let mut target_run_ids = BTreeSet::new();
    for record in run_manifest {
        partition_by_run.insert(record.run_id.as_str(), record.partition);
        if record.partition == target_partition {
            target_run_ids.insert(record.run_id.as_str());
        }
    }

    for run in evaluation_runs {
        if !canonical_non_empty(&run.run_id) {
            return Err(ObserverEvaluationError::InvalidIdentity("run_id"));
        }
    }

    let mut seen_evaluation_runs = BTreeSet::new();
    let mut prepared = Vec::with_capacity(evaluation_runs.len());
    let mut frozen_model: Option<(String, String, String)> = None;

    for run in evaluation_runs {
        if !canonical_non_empty(&run.model_id) {
            return Err(ObserverEvaluationError::InvalidIdentity("model_id"));
        }
        if !valid_digest(&run.model_sha256) {
            return Err(ObserverEvaluationError::InvalidDigest("model_sha256"));
        }
        if !valid_digest(&run.calibration_bundle_sha256) {
            return Err(ObserverEvaluationError::InvalidDigest("calibration_bundle_sha256"));
        }
        if !seen_evaluation_runs.insert(run.run_id.as_str()) {
            return Err(ObserverEvaluationError::DuplicateEvaluationRun(run.run_id.clone()));
        }
        let observed_partition = partition_by_run
            .get(run.run_id.as_str())
            .copied()
            .ok_or_else(|| ObserverEvaluationError::UnknownEvaluationRun(run.run_id.clone()))?;
        if observed_partition != target_partition {
            return Err(ObserverEvaluationError::EvaluationRunWrongPartition(
                run.run_id.clone(),
            ));
        }

        match &frozen_model {
            Some((model_id, model_sha, calibration_sha))
                if model_id != &run.model_id
                    || model_sha != &run.model_sha256
                    || calibration_sha != &run.calibration_bundle_sha256 =>
            {
                return Err(ObserverEvaluationError::ModelIdentityMismatch);
            }
            None => {
                frozen_model = Some((
                    run.model_id.clone(),
                    run.model_sha256.clone(),
                    run.calibration_bundle_sha256.clone(),
                ));
            }
            _ => {}
        }

        if run.item_ids.is_empty() {
            return Err(ObserverEvaluationError::EmptyItemSet(run.run_id.clone()));
        }
        let mut item_ids = BTreeSet::new();
        for item_id in &run.item_ids {
            if !canonical_non_empty(item_id) {
                return Err(ObserverEvaluationError::InvalidIdentity("item_id"));
            }
            if !item_ids.insert(item_id.as_str()) {
                return Err(ObserverEvaluationError::DuplicateItem {
                    run_id: run.run_id.clone(),
                    item_id: item_id.clone(),
                });
            }
        }

        let mut predictions = BTreeMap::new();
        for prediction in &run.predictions {
            if !item_ids.contains(prediction.item_id.as_str()) {
                return Err(ObserverEvaluationError::UnexpectedItem {
                    run_id: run.run_id.clone(),
                    item_id: prediction.item_id.clone(),
                });
            }
            if !required.contains(&prediction.endpoint) {
                return Err(ObserverEvaluationError::UnexpectedEndpoint {
                    run_id: run.run_id.clone(),
                    endpoint: prediction.endpoint,
                });
            }
            if !prediction.pass_probability.is_finite()
                || !(0.0..=1.0).contains(&prediction.pass_probability)
            {
                return Err(ObserverEvaluationError::InvalidProbability {
                    run_id: run.run_id.clone(),
                    item_id: prediction.item_id.clone(),
                    endpoint: prediction.endpoint,
                });
            }
            let key = (prediction.item_id.clone(), prediction.endpoint);
            if predictions.insert(key.clone(), prediction.pass_probability).is_some() {
                return Err(ObserverEvaluationError::DuplicatePrediction {
                    run_id: run.run_id.clone(),
                    item_id: key.0,
                    endpoint: key.1,
                });
            }
        }

        let mut outcomes = BTreeMap::new();
        for outcome in &run.outcomes {
            if !canonical_non_empty(&outcome.assay_record_id) {
                return Err(ObserverEvaluationError::InvalidIdentity("assay_record_id"));
            }
            if !item_ids.contains(outcome.item_id.as_str()) {
                return Err(ObserverEvaluationError::UnexpectedItem {
                    run_id: run.run_id.clone(),
                    item_id: outcome.item_id.clone(),
                });
            }
            if !required.contains(&outcome.endpoint) {
                return Err(ObserverEvaluationError::UnexpectedEndpoint {
                    run_id: run.run_id.clone(),
                    endpoint: outcome.endpoint,
                });
            }
            let key = (outcome.item_id.clone(), outcome.endpoint);
            if outcomes.insert(key.clone(), outcome.decision).is_some() {
                return Err(ObserverEvaluationError::DuplicateOutcome {
                    run_id: run.run_id.clone(),
                    item_id: key.0,
                    endpoint: key.1,
                });
            }
        }

        let expected_pair_count = item_ids
            .len()
            .checked_mul(required.len())
            .ok_or(ObserverEvaluationError::InvalidIdentity("item_endpoint_count"))?;
        if predictions.len() != expected_pair_count {
            for item_id in &item_ids {
                for endpoint in &required {
                    if !predictions.contains_key(&(item_id.to_string(), *endpoint)) {
                        return Err(ObserverEvaluationError::MissingPrediction {
                            run_id: run.run_id.clone(),
                            item_id: (*item_id).to_owned(),
                            endpoint: *endpoint,
                        });
                    }
                }
            }
        }
        if outcomes.len() != expected_pair_count {
            for item_id in &item_ids {
                for endpoint in &required {
                    if !outcomes.contains_key(&(item_id.to_string(), *endpoint)) {
                        return Err(ObserverEvaluationError::MissingOutcome {
                            run_id: run.run_id.clone(),
                            item_id: (*item_id).to_owned(),
                            endpoint: *endpoint,
                        });
                    }
                }
            }
        }

        let minimum_endpoint_probability = predictions.values().copied()
            .fold(f64::INFINITY, f64::min);
        let mut endpoint_truth = BTreeMap::new();
        for endpoint in &required {
            let mut endpoint_failed = false;
            let mut endpoint_indeterminate = false;
            for ((_, observed_endpoint), decision) in &outcomes {
                if observed_endpoint == endpoint {
                    match decision {
                        Decision::Pass => {}
                        Decision::Fail => endpoint_failed = true,
                        Decision::Indeterminate => endpoint_indeterminate = true,
                    }
                }
            }
            let endpoint_status = if endpoint_failed {
                RunTruth::Nonconforming
            } else if endpoint_indeterminate {
                RunTruth::Unresolved
            } else {
                RunTruth::Conforming
            };
            endpoint_truth.insert(*endpoint, endpoint_status);
        }
        let truth = if endpoint_truth.values().any(|status| *status == RunTruth::Nonconforming) {
            RunTruth::Nonconforming
        } else if endpoint_truth.values().any(|status| *status == RunTruth::Unresolved) {
            RunTruth::Unresolved
        } else {
            RunTruth::Conforming
        };
        prepared.push(PreparedRun {
            truth,
            minimum_endpoint_probability,
            endpoint_truth,
        });
    }

    for run_id in &target_run_ids {
        if !seen_evaluation_runs.contains(*run_id) {
            return Err(ObserverEvaluationError::MissingEvaluationRun((*run_id).to_owned()));
        }
    }
    if evaluation_runs.len() != target_run_ids.len() {
        for run in evaluation_runs {
            if !target_run_ids.contains(run.run_id.as_str()) {
                return Err(ObserverEvaluationError::UnknownEvaluationRun(run.run_id.clone()));
            }
        }
    }

    let (model_id, model_sha256, calibration_bundle_sha256) = frozen_model
        .ok_or_else(|| ObserverEvaluationError::MissingEvaluationRun("target partition is empty".to_owned()))?;

    let physical_run_count = prepared.len();
    let mut points = Vec::with_capacity(sorted_thresholds.len());
    for threshold in sorted_thresholds {
        let conforming_run_count = prepared.iter()
            .filter(|run| run.truth == RunTruth::Conforming).count();
        let nonconforming_run_count = prepared.iter()
            .filter(|run| run.truth == RunTruth::Nonconforming).count();
        let unresolved_run_count = prepared.iter()
            .filter(|run| run.truth == RunTruth::Unresolved).count();

        let mut released_run_count = 0_usize;
        let mut verified_released_run_count = 0_usize;
        let mut conforming_released_run_count = 0_usize;
        let mut false_accept_run_count = 0_usize;
        let mut unverifiable_release_run_count = 0_usize;
        let mut conforming_not_released_run_count = 0_usize;

        for run in &prepared {
            let released = run.minimum_endpoint_probability >= threshold;
            if released {
                released_run_count += 1;
                match run.truth {
                    RunTruth::Conforming => {
                        verified_released_run_count += 1;
                        conforming_released_run_count += 1;
                    }
                    RunTruth::Nonconforming => {
                        verified_released_run_count += 1;
                        false_accept_run_count += 1;
                    }
                    RunTruth::Unresolved => unverifiable_release_run_count += 1,
                }
            } else if run.truth == RunTruth::Conforming {
                conforming_not_released_run_count += 1;
            }
        }

        let iid_zero_event_challenge_upper_bound = if false_accept_run_count == 0
            && nonconforming_run_count > 0
        {
            Some(1.0 - alpha.powf(1.0 / nonconforming_run_count as f64))
        } else {
            None
        };

        points.push(ObserverThresholdMetrics {
            threshold,
            physical_run_count,
            conforming_run_count,
            nonconforming_run_count,
            unresolved_run_count,
            released_run_count,
            verified_released_run_count,
            conforming_released_run_count,
            false_accept_run_count,
            unverifiable_release_run_count,
            conforming_not_released_run_count,
            release_coverage: ratio(released_run_count, physical_run_count).unwrap_or(0.0),
            challenge_false_accept_rate: ratio(
                false_accept_run_count,
                nonconforming_run_count,
            ),
            false_release_rate_among_verified_releases: ratio(
                false_accept_run_count,
                verified_released_run_count,
            ),
            conforming_release_rate: ratio(
                conforming_released_run_count,
                conforming_run_count,
            ),
            iid_zero_event_challenge_upper_bound,
            endpoint_metrics: endpoint_metrics(&prepared, &required, threshold),
        });
    }

    Ok(ObserverBenchmarkReport {
        target_partition,
        target_run_count: physical_run_count,
        alpha,
        model_id,
        model_sha256,
        calibration_bundle_sha256,
        grouped_validation,
        points,
    })
}

fn endpoint_metrics(
    runs: &[PreparedRun],
    required: &BTreeSet<Endpoint>,
    threshold: f64,
) -> Vec<EndpointThresholdMetrics> {
    required.iter().map(|endpoint| {
        let mut nonconforming_run_count = 0_usize;
        let mut false_accept_run_count = 0_usize;
        let mut unresolved_run_count = 0_usize;
        for run in runs {
            match run.endpoint_truth.get(endpoint).copied() {
                Some(RunTruth::Nonconforming) => {
                    nonconforming_run_count += 1;
                    if run.minimum_endpoint_probability >= threshold {
                        false_accept_run_count += 1;
                    }
                }
                Some(RunTruth::Unresolved) => unresolved_run_count += 1,
                Some(RunTruth::Conforming) => {}
                None => unresolved_run_count += 1,
            }
        }
        EndpointThresholdMetrics {
            endpoint: *endpoint,
            nonconforming_run_count,
            false_accept_run_count,
            unresolved_run_count,
            challenge_false_accept_rate: ratio(
                false_accept_run_count,
                nonconforming_run_count,
            ),
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUIRED: [Endpoint; 3] = [
        Endpoint::SoilRemovalResidual,
        Endpoint::FinalRinseResidue,
        Endpoint::CrossContamination,
    ];

    fn manifest_record(
        run_id: &str,
        partition: ValidationPartition,
        suffix: &str,
    ) -> RunPartitionRecord {
        RunPartitionRecord {
            run_id: run_id.to_owned(),
            partition,
            machine_unit_id: format!("machine-{suffix}"),
            test_session_id: format!("session-{suffix}"),
            soil_batch_id: format!("soil-{suffix}"),
            detergent_batch_id: format!("detergent-{suffix}"),
            water_batch_id: format!("water-{suffix}"),
            operator_id: format!("operator-{suffix}"),
            assay_lab_id: format!("lab-{suffix}"),
            test_site_id: format!("site-{suffix}"),
        }
    }

    fn scores(run_id: &str, decision: Decision, probability: f64) -> ObserverRunScores {
        let item_id = "cup-1".to_owned();
        ObserverRunScores {
            run_id: run_id.to_owned(),
            item_ids: vec![item_id.clone()],
            model_id: "observer-v1".to_owned(),
            model_sha256: "a".repeat(64),
            calibration_bundle_sha256: "b".repeat(64),
            predictions: REQUIRED.iter().map(|endpoint| EndpointPrediction {
                item_id: item_id.clone(),
                endpoint: *endpoint,
                pass_probability: probability,
            }).collect(),
            outcomes: REQUIRED.iter().map(|endpoint| EndpointOutcome {
                item_id: item_id.clone(),
                endpoint: *endpoint,
                decision,
                assay_record_id: format!("assay-{run_id}-{endpoint:?}"),
            }).collect(),
        }
    }

    fn manifest_for_target(
        target: ValidationPartition,
        target_ids: &[&str],
    ) -> Vec<RunPartitionRecord> {
        let mut records = vec![
            manifest_record("development-run", ValidationPartition::Development, "dev"),
            manifest_record("calibration-run", ValidationPartition::Calibration, "cal"),
        ];
        if target == ValidationPartition::ExternalReplication {
            records.push(manifest_record("confirmation-run", ValidationPartition::LockedConfirmation, "confirm"));
        }
        records.extend(target_ids.iter().enumerate().map(|(index, run_id)| {
            manifest_record(run_id, target, &format!("target-{index}"))
        }));
        records
    }

    #[test]
    fn reports_run_level_risk_coverage_without_counting_items_as_runs() {
        let target_ids = ["pass-high", "fail-high", "unknown-high", "pass-low"];
        let manifest = manifest_for_target(
            ValidationPartition::LockedConfirmation,
            &target_ids,
        );
        let runs = vec![
            scores("pass-high", Decision::Pass, 0.95),
            scores("fail-high", Decision::Fail, 0.98),
            scores("unknown-high", Decision::Indeterminate, 0.99),
            scores("pass-low", Decision::Pass, 0.20),
        ];
        let report = evaluate_observer_risk_coverage(
            &manifest,
            &runs,
            &[
                HoldoutAxis::MachineUnit,
                HoldoutAxis::TestSession,
                HoldoutAxis::SoilBatch,
            ],
            ValidationPartition::LockedConfirmation,
            &REQUIRED,
            &[0.99, 0.90],
            0.05,
        ).unwrap();

        assert_eq!(report.target_run_count, 4);
        assert_eq!(report.grouped_validation.physical_run_count, 6);
        assert_eq!(report.points[0].threshold, 0.90);
        assert_eq!(report.points[0].released_run_count, 3);
        assert_eq!(report.points[0].false_accept_run_count, 1);
        assert_eq!(report.points[0].unverifiable_release_run_count, 1);
        assert_eq!(report.points[0].challenge_false_accept_rate, Some(1.0));
        assert_eq!(report.points[0].false_release_rate_among_verified_releases, Some(0.5));
        assert_eq!(report.points[0].release_coverage, 0.75);

        assert_eq!(report.points[1].threshold, 0.99);
        assert_eq!(report.points[1].released_run_count, 1);
        assert_eq!(report.points[1].false_accept_run_count, 0);
        assert_eq!(report.points[1].unverifiable_release_run_count, 1);
        assert!((report.points[1].iid_zero_event_challenge_upper_bound.unwrap() - 0.95).abs() < 1e-12);
    }

    #[test]
    fn requires_every_item_endpoint_prediction_and_assay_outcome() {
        let manifest = manifest_for_target(
            ValidationPartition::LockedConfirmation,
            &["run-1"],
        );
        let mut run = scores("run-1", Decision::Pass, 0.99);
        run.predictions.pop();
        assert!(matches!(
            evaluate_observer_risk_coverage(
                &manifest,
                &[run],
                &[HoldoutAxis::MachineUnit],
                ValidationPartition::LockedConfirmation,
                &REQUIRED,
                &[0.9],
                0.05,
            ),
            Err(ObserverEvaluationError::MissingPrediction { .. })
        ));
    }

    #[test]
    fn rejects_group_leakage_before_scoring() {
        let mut manifest = manifest_for_target(
            ValidationPartition::LockedConfirmation,
            &["run-1"],
        );
        manifest[2].machine_unit_id = manifest[0].machine_unit_id.clone();
        let runs = vec![scores("run-1", Decision::Pass, 0.99)];
        assert!(matches!(
            evaluate_observer_risk_coverage(
                &manifest,
                &runs,
                &[HoldoutAxis::MachineUnit],
                ValidationPartition::LockedConfirmation,
                &REQUIRED,
                &[0.9],
                0.05,
            ),
            Err(ObserverEvaluationError::GroupedValidation(
                GroupedValidationError::GroupLeaksAcrossPartitions { .. }
            ))
        ));
    }

    #[test]
    fn requires_complete_external_replication_partition_stack() {
        let manifest = vec![
            manifest_record("development-run", ValidationPartition::Development, "dev"),
            manifest_record("calibration-run", ValidationPartition::Calibration, "cal"),
            manifest_record("external-run", ValidationPartition::ExternalReplication, "external"),
        ];
        assert!(matches!(
            evaluate_observer_risk_coverage(
                &manifest,
                &[scores("external-run", Decision::Pass, 0.99)],
                &[HoldoutAxis::MachineUnit],
                ValidationPartition::ExternalReplication,
                &REQUIRED,
                &[0.9],
                0.05,
            ),
            Err(ObserverEvaluationError::GroupedValidation(
                GroupedValidationError::MissingRequiredPartition(
                    ValidationPartition::LockedConfirmation
                )
            ))
        ));
    }

    #[test]
    fn rejects_inconsistent_model_or_calibration_identity() {
        let manifest = manifest_for_target(
            ValidationPartition::LockedConfirmation,
            &["run-1", "run-2"],
        );
        let mut second = scores("run-2", Decision::Pass, 0.99);
        second.calibration_bundle_sha256 = "c".repeat(64);
        assert_eq!(
            evaluate_observer_risk_coverage(
                &manifest,
                &[scores("run-1", Decision::Pass, 0.99), second],
                &[HoldoutAxis::MachineUnit],
                ValidationPartition::LockedConfirmation,
                &REQUIRED,
                &[0.9],
                0.05,
            ),
            Err(ObserverEvaluationError::ModelIdentityMismatch)
        );
    }

    #[test]
    fn unresolved_truth_is_counted_not_promoted_to_success() {
        let manifest = manifest_for_target(
            ValidationPartition::LockedConfirmation,
            &["unknown-run"],
        );
        let report = evaluate_observer_risk_coverage(
            &manifest,
            &[scores("unknown-run", Decision::Indeterminate, 0.99)],
            &[HoldoutAxis::MachineUnit],
            ValidationPartition::LockedConfirmation,
            &REQUIRED,
            &[0.9],
            0.05,
        ).unwrap();
        let point = &report.points[0];
        assert_eq!(point.unresolved_run_count, 1);
        assert_eq!(point.unverifiable_release_run_count, 1);
        assert_eq!(point.verified_released_run_count, 0);
        assert_eq!(point.false_release_rate_among_verified_releases, None);
        assert_eq!(point.iid_zero_event_challenge_upper_bound, None);
    }

    #[test]
    fn rejects_proxy_scores_outside_probability_range() {
        let manifest = manifest_for_target(
            ValidationPartition::LockedConfirmation,
            &["run-1"],
        );
        let mut run = scores("run-1", Decision::Pass, 0.99);
        run.predictions[0].pass_probability = 1.01;
        assert!(matches!(
            evaluate_observer_risk_coverage(
                &manifest,
                &[run],
                &[HoldoutAxis::MachineUnit],
                ValidationPartition::LockedConfirmation,
                &REQUIRED,
                &[0.9],
                0.05,
            ),
            Err(ObserverEvaluationError::InvalidProbability { .. })
        ));
    }

    #[test]
    fn rejects_missing_evaluation_run_and_unsupported_partition() {
        let manifest = manifest_for_target(
            ValidationPartition::LockedConfirmation,
            &["run-1"],
        );
        assert!(matches!(
            evaluate_observer_risk_coverage(
                &manifest,
                &[],
                &[HoldoutAxis::MachineUnit],
                ValidationPartition::LockedConfirmation,
                &REQUIRED,
                &[0.9],
                0.05,
            ),
            Err(ObserverEvaluationError::MissingEvaluationRun(_))
        ));
        assert_eq!(
            evaluate_observer_risk_coverage(
                &manifest,
                &[scores("development-run", Decision::Pass, 0.99)],
                &[HoldoutAxis::MachineUnit],
                ValidationPartition::Development,
                &REQUIRED,
                &[0.9],
                0.05,
            ),
            Err(ObserverEvaluationError::UnsupportedTargetPartition)
        );
    }
}
