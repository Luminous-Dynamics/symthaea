//! Runnable scalar-error baselines for the cognitive-loop drift benchmark.
//!
//! The first implementation intentionally covers the valid-PE-only access track:
//! fixed threshold and a calibration-centred EWMA control chart. Invalid/missing PE
//! is absence of evidence, never a numeric sentinel. These are reference controls,
//! not claims of state-of-the-art performance.

use crate::cognitive_drift_score::{score_trace, Detection, DriftEvent, DriftScore, DriftTrace};
use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarDriftRunInput {
    pub schema_version: u32,
    pub stream_id: String,
    pub detector_version: String,
    /// Full commit SHA of the code that produced this run.
    pub repository_head_sha: String,
    pub seed: u64,
    pub label_delay_cycles: u64,
    pub max_detection_delay_cycles: u64,
    /// Disjoint calibration observations; never included in the scored evaluation stream.
    pub calibration_values: Vec<f64>,
    /// One entry per evaluation cycle, including explicit missing-evidence cycles.
    pub observations: Vec<ScalarObservation>,
    pub drift_events: Vec<DriftEvent>,
    pub detector: ScalarDetectorConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarObservation {
    pub cycle_index: u64,
    pub value: Option<f64>,
    pub evidence_valid: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ScalarDetectorConfig {
    pub method: ScalarDetectorMethod,
    /// Required only for fixed-threshold. Threshold is selected on disjoint calibration data.
    pub fixed_threshold: Option<f64>,
    /// Required only for EWMA. Must be in (0, 1].
    pub ewma_alpha: Option<f64>,
    /// Required only for EWMA; multiplier applied to calibration standard deviation.
    pub ewma_threshold_sigma: Option<f64>,
    /// Required only for EWMA; prevents a zero-variance calibration window from zeroing the limit.
    pub calibration_stddev_floor: Option<f64>,
    /// Required only for Page-Hinkley. Tolerated mean movement per valid sample.
    pub page_hinkley_delta: Option<f64>,
    /// Required only for Page-Hinkley. Positive cumulative-deviation alarm limit.
    pub page_hinkley_threshold: Option<f64>,
    /// Suppress alarms until this many valid evaluation samples have been seen.
    pub warmup_valid_observations: usize,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ScalarDetectorMethod {
    FixedThreshold,
    EwmaControlChart,
    PageHinkley,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ScalarDriftRunResult {
    pub result_schema_version: u32,
    pub metric_version: String,
    pub canonical_input_sha256: String,
    /// SHA-256 of the serialized result with this field set to null.
    pub result_sha256: Option<String>,
    pub stream_id: String,
    pub detector_id: String,
    pub detector_version: String,
    pub repository_head_sha: String,
    pub configuration_digest: String,
    pub comparison_track: String,
    pub seed: u64,
    pub calibration_observation_count: usize,
    pub evaluation_cycle_count: usize,
    pub valid_evaluation_observation_count: usize,
    pub detections: Vec<Detection>,
    pub score: DriftScore,
    pub interpretation_guard: String,
}

pub fn evaluate_file(path: &Path) -> anyhow::Result<ScalarDriftRunResult> {
    let bytes = fs::read(path)
        .with_context(|| format!("reading scalar drift input {}", path.display()))?;
    let input: ScalarDriftRunInput = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing scalar drift input {}", path.display()))?;
    run_scalar_detector(input)
}

pub fn run_scalar_detector(mut input: ScalarDriftRunInput) -> anyhow::Result<ScalarDriftRunResult> {
    ensure!(input.schema_version == 1, "scalar drift input schema_version must be 1");
    ensure!(!input.stream_id.trim().is_empty(), "stream_id must not be empty");
    ensure!(
        !input.detector_version.trim().is_empty(),
        "detector_version must not be empty"
    );
    validate_commit_sha(&input.repository_head_sha)?;
    ensure!(
        input.max_detection_delay_cycles > 0,
        "max_detection_delay_cycles must be positive"
    );
    ensure!(
        !input.calibration_values.is_empty(),
        "calibration_values must not be empty"
    );
    ensure!(
        input.calibration_values.iter().all(|x| x.is_finite()),
        "calibration_values must all be finite"
    );
    ensure!(
        !input.observations.is_empty(),
        "observations must not be empty"
    );
    for (index, observation) in input.observations.iter().enumerate() {
        ensure!(
            observation.cycle_index == index as u64,
            "evaluation observations must cover a contiguous cycle panel starting at zero"
        );
        match (observation.evidence_valid, observation.value) {
            (true, Some(value)) => ensure!(
                value.is_finite(),
                "valid evidence at cycle {} must be finite",
                observation.cycle_index
            ),
            (true, None) => anyhow::bail!(
                "valid evidence at cycle {} must include a numeric value",
                observation.cycle_index
            ),
            (false, None) => {}
            (false, Some(_)) => anyhow::bail!(
                "invalid evidence at cycle {} must be represented as missing, not a sentinel",
                observation.cycle_index
            ),
        }
    }

    // Ground-truth events are a set of intervals, not an ordered stream. Normalize
    // their serialization before hashing while preserving calibration/sample order.
    input.drift_events.sort_by(|left, right| {
        (left.start_cycle, left.end_cycle, &left.id)
            .cmp(&(right.start_cycle, right.end_cycle, &right.id))
    });
    let input_digest = sha256(&serde_json::to_vec(&input)?);
    let configuration_digest = sha256(&serde_json::to_vec(&input.detector)?);
    let (detector_id, detections) = detect(&input)?;
    let valid_count = input
        .observations
        .iter()
        .filter(|sample| sample.evidence_valid)
        .count();

    let trace = DriftTrace {
        schema_version: 1,
        stream_id: input.stream_id.clone(),
        detector_id: detector_id.clone(),
        detector_version: input.detector_version.clone(),
        comparison_track: "valid-pe-only".to_owned(),
        label_delay_cycles: input.label_delay_cycles,
        repository_head_sha: input.repository_head_sha.clone(),
        configuration_digest: configuration_digest.clone(),
        seed: input.seed,
        stream_length_cycles: input.observations.len() as u64,
        valid_observations: valid_count as u64,
        max_detection_delay_cycles: input.max_detection_delay_cycles,
        drift_events: input.drift_events,
        detections: detections.clone(),
    };
    let score = score_trace(trace)?;

    let mut result = ScalarDriftRunResult {
        result_schema_version: 1,
        metric_version: "scalar-drift-baselines-v1".to_owned(),
        canonical_input_sha256: input_digest,
        result_sha256: None,
        stream_id: input.stream_id,
        detector_id,
        detector_version: input.detector_version,
        repository_head_sha: input.repository_head_sha,
        configuration_digest,
        comparison_track: "valid-pe-only".to_owned(),
        seed: input.seed,
        calibration_observation_count: input.calibration_values.len(),
        evaluation_cycle_count: input.observations.len(),
        valid_evaluation_observation_count: valid_count,
        detections,
        score,
        interpretation_guard: "Fixed threshold and EWMA are reference controls for the valid-PE-only track, not a complete KDD 2026 reproduction or SOTA claim. Calibration observations must be disjoint from the scored evaluation stream. Invalid evidence is skipped and cannot update detector state. Detection F1 must be reported alongside false alarms, delay, task impact, and adaptation cost.".to_owned(),
    };
    result.result_sha256 = Some(result_digest(&result)?);
    Ok(result)
}

fn detect(input: &ScalarDriftRunInput) -> anyhow::Result<(String, Vec<Detection>)> {
    ensure!(
        input.detector.warmup_valid_observations > 0,
        "warmup_valid_observations must be positive"
    );
    match input.detector.method {
        ScalarDetectorMethod::FixedThreshold => {
            let threshold = input
                .detector
                .fixed_threshold
                .context("fixed_threshold is required for fixed-threshold")?;
            ensure!(
                threshold.is_finite() && threshold >= 0.0,
                "fixed_threshold must be finite and non-negative"
            );
            ensure!(
                input.detector.ewma_alpha.is_none()
                    && input.detector.ewma_threshold_sigma.is_none()
                    && input.detector.calibration_stddev_floor.is_none()
                    && input.detector.page_hinkley_delta.is_none()
                    && input.detector.page_hinkley_threshold.is_none(),
                "EWMA and Page-Hinkley parameters must be absent for fixed-threshold"
            );
            let mut detections = Vec::new();
            let mut latched = false;
            let mut valid_count = 0usize;
            for observation in &input.observations {
                if !observation.evidence_valid {
                    continue;
                }
                valid_count += 1;
                let above = observation.value.expect("validated numeric PE") > threshold;
                if valid_count >= input.detector.warmup_valid_observations && above && !latched {
                    detections.push(Detection {
                        cycle_index: observation.cycle_index,
                        evidence_valid: true,
                    });
                }
                // Invalid cycles do not rearm the detector; only valid below-threshold evidence does.
                if valid_count >= input.detector.warmup_valid_observations && !above {
                    latched = false;
                } else if above && valid_count >= input.detector.warmup_valid_observations {
                    latched = true;
                }
            }
            Ok(("fixed-threshold".to_owned(), detections))
        }
        ScalarDetectorMethod::EwmaControlChart => {
            ensure!(
                input.detector.fixed_threshold.is_none()
                    && input.detector.page_hinkley_delta.is_none()
                    && input.detector.page_hinkley_threshold.is_none(),
                "fixed-threshold and Page-Hinkley parameters must be absent for ewma-control-chart"
            );
            let alpha = input
                .detector
                .ewma_alpha
                .context("ewma_alpha is required for ewma-control-chart")?;
            let sigma_multiplier = input
                .detector
                .ewma_threshold_sigma
                .context("ewma_threshold_sigma is required for ewma-control-chart")?;
            let stddev_floor = input
                .detector
                .calibration_stddev_floor
                .context("calibration_stddev_floor is required for ewma-control-chart")?;
            ensure!(
                alpha.is_finite() && alpha > 0.0 && alpha <= 1.0,
                "ewma_alpha must be finite and in (0, 1]"
            );
            ensure!(
                sigma_multiplier.is_finite() && sigma_multiplier > 0.0,
                "ewma_threshold_sigma must be finite and positive"
            );
            ensure!(
                stddev_floor.is_finite() && stddev_floor > 0.0,
                "calibration_stddev_floor must be finite and positive"
            );
            let (center, calibration_stddev) = mean_stddev(&input.calibration_values)?;
            let effective_stddev = calibration_stddev.max(stddev_floor);
            let limit = sigma_multiplier * effective_stddev;
            ensure!(limit.is_finite(), "EWMA control limit overflowed");
            let mut ewma = center;
            let mut detections = Vec::new();
            let mut latched = false;
            let mut valid_count = 0usize;
            for observation in &input.observations {
                if !observation.evidence_valid {
                    continue;
                }
                valid_count += 1;
                let value = observation.value.expect("validated numeric PE");
                ewma = alpha * value + (1.0 - alpha) * ewma;
                ensure!(ewma.is_finite(), "EWMA state became non-finite");
                let deviation = (ewma - center).abs();
                ensure!(deviation.is_finite(), "EWMA deviation overflowed");
                let outside = deviation > limit;
                if valid_count >= input.detector.warmup_valid_observations && outside && !latched {
                    detections.push(Detection {
                        cycle_index: observation.cycle_index,
                        evidence_valid: true,
                    });
                }
                if valid_count >= input.detector.warmup_valid_observations && !outside {
                    latched = false;
                } else if outside && valid_count >= input.detector.warmup_valid_observations {
                    latched = true;
                }
            }
            Ok(("ewma-control-chart".to_owned(), detections))
        }
        ScalarDetectorMethod::PageHinkley => {
            ensure!(
                input.detector.fixed_threshold.is_none()
                    && input.detector.ewma_alpha.is_none()
                    && input.detector.ewma_threshold_sigma.is_none()
                    && input.detector.calibration_stddev_floor.is_none(),
                "fixed-threshold and EWMA parameters must be absent for Page-Hinkley"
            );
            let delta = input
                .detector
                .page_hinkley_delta
                .context("page_hinkley_delta is required for Page-Hinkley")?;
            let threshold = input
                .detector
                .page_hinkley_threshold
                .context("page_hinkley_threshold is required for Page-Hinkley")?;
            ensure!(
                delta.is_finite() && delta >= 0.0,
                "page_hinkley_delta must be finite and non-negative"
            );
            ensure!(
                threshold.is_finite() && threshold > 0.0,
                "page_hinkley_threshold must be finite and positive"
            );
            let (calibration_mean, _) = mean_stddev(&input.calibration_values)?;
            let rearm_level = calibration_mean + delta;
            ensure!(rearm_level.is_finite(), "Page-Hinkley rearm level overflowed");

            // Page-Hinkley on upward shifts in valid PE. Calibration mean initializes
            // the sequential mean; an alarm is latched until valid PE returns near the
            // calibration regime, avoiding repeated alarms during one sustained shift.
            let mut running_mean = calibration_mean;
            let mut running_count = input.calibration_values.len();
            let mut cumulative = 0.0_f64;
            let mut minimum_cumulative = 0.0_f64;
            let mut detections = Vec::new();
            let mut valid_count = 0usize;
            let mut latched = false;
            for observation in &input.observations {
                if !observation.evidence_valid {
                    continue;
                }
                let value = observation.value.expect("validated numeric PE");
                valid_count += 1;
                running_count += 1;
                let delta_mean = value - running_mean;
                ensure!(delta_mean.is_finite(), "Page-Hinkley mean deviation overflowed");
                running_mean += delta_mean / running_count as f64;
                ensure!(running_mean.is_finite(), "Page-Hinkley running mean became non-finite");

                if latched {
                    if value <= rearm_level {
                        latched = false;
                        cumulative = 0.0;
                        minimum_cumulative = 0.0;
                    }
                    continue;
                }

                cumulative += value - running_mean - delta;
                ensure!(cumulative.is_finite(), "Page-Hinkley cumulative statistic overflowed");
                minimum_cumulative = minimum_cumulative.min(cumulative);
                let deviation = cumulative - minimum_cumulative;
                ensure!(deviation.is_finite(), "Page-Hinkley deviation overflowed");
                if valid_count >= input.detector.warmup_valid_observations
                    && deviation > threshold
                {
                    detections.push(Detection {
                        cycle_index: observation.cycle_index,
                        evidence_valid: true,
                    });
                    latched = true;
                    cumulative = 0.0;
                    minimum_cumulative = 0.0;
                }
            }
            Ok(("page-hinkley".to_owned(), detections))
        }
    }
}

fn mean_stddev(values: &[f64]) -> anyhow::Result<(f64, f64)> {
    ensure!(values.len() >= 2, "EWMA calibration requires at least two observations");
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    for (index, &value) in values.iter().enumerate() {
        let count = (index + 1) as f64;
        let delta = value - mean;
        ensure!(delta.is_finite(), "calibration mean update overflowed");
        mean += delta / count;
        ensure!(mean.is_finite(), "calibration mean became non-finite");
        let delta_after = value - mean;
        let increment = delta * delta_after;
        ensure!(increment.is_finite(), "calibration variance update overflowed");
        m2 += increment;
        ensure!(m2.is_finite(), "calibration variance accumulator overflowed");
    }
    let variance = (m2 / (values.len() - 1) as f64).max(0.0);
    ensure!(variance.is_finite(), "calibration variance is not finite");
    let stddev = variance.sqrt();
    ensure!(stddev.is_finite(), "calibration standard deviation is not finite");
    Ok((mean, stddev))
}

fn validate_commit_sha(value: &str) -> anyhow::Result<()> {
    ensure!(
        value.len() == 40 || value.len() == 64,
        "repository_head_sha must be a full 40- or 64-character commit hash"
    );
    ensure!(
        value.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "repository_head_sha must be hexadecimal"
    );
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn result_digest(result: &ScalarDriftRunResult) -> anyhow::Result<String> {
    let mut copy = result.clone();
    copy.result_sha256 = None;
    Ok(sha256(&serde_json::to_vec(&copy)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(method: ScalarDetectorMethod) -> ScalarDriftRunInput {
        let detector = match method {
            ScalarDetectorMethod::FixedThreshold => ScalarDetectorConfig {
                method,
                fixed_threshold: Some(0.5),
                ewma_alpha: None,
                ewma_threshold_sigma: None,
                calibration_stddev_floor: None,
                page_hinkley_delta: None,
                page_hinkley_threshold: None,
                warmup_valid_observations: 1,
            },
            ScalarDetectorMethod::EwmaControlChart => ScalarDetectorConfig {
                method,
                fixed_threshold: None,
                ewma_alpha: Some(0.5),
                ewma_threshold_sigma: Some(3.0),
                calibration_stddev_floor: Some(0.01),
                page_hinkley_delta: None,
                page_hinkley_threshold: None,
                warmup_valid_observations: 1,
            },
            ScalarDetectorMethod::PageHinkley => ScalarDetectorConfig {
                method,
                fixed_threshold: None,
                ewma_alpha: None,
                ewma_threshold_sigma: None,
                calibration_stddev_floor: None,
                page_hinkley_delta: Some(0.05),
                page_hinkley_threshold: Some(0.15),
                warmup_valid_observations: 1,
            },
        };
        ScalarDriftRunInput {
            schema_version: 1,
            stream_id: "synthetic-stream-not-measured".to_owned(),
            detector_version: "xtask-reference-1".to_owned(),
            repository_head_sha: "a".repeat(40),
            seed: 17,
            label_delay_cycles: 0,
            max_detection_delay_cycles: 2,
            calibration_values: vec![0.1, 0.12, 0.09, 0.11],
            observations: vec![
                ScalarObservation { cycle_index: 0, value: Some(0.1), evidence_valid: true },
                ScalarObservation { cycle_index: 1, value: Some(0.8), evidence_valid: true },
                ScalarObservation { cycle_index: 2, value: Some(0.9), evidence_valid: true },
                ScalarObservation { cycle_index: 3, value: None, evidence_valid: false },
                ScalarObservation { cycle_index: 4, value: Some(0.7), evidence_valid: true },
                ScalarObservation { cycle_index: 5, value: Some(0.1), evidence_valid: true },
                ScalarObservation { cycle_index: 6, value: Some(0.8), evidence_valid: true },
                ScalarObservation { cycle_index: 7, value: Some(0.2), evidence_valid: true },
            ],
            drift_events: vec![
                DriftEvent { id: "shift-one".to_owned(), start_cycle: 1, end_cycle: 2 },
                DriftEvent { id: "shift-two".to_owned(), start_cycle: 6, end_cycle: 6 },
            ],
            detector,
        }
    }

    #[test]
    fn fixed_threshold_rearms_only_on_valid_below_threshold_evidence() {
        let result = run_scalar_detector(fixture(ScalarDetectorMethod::FixedThreshold))
            .expect("fixed threshold should run");
        assert_eq!(
            result.detections.iter().map(|d| d.cycle_index).collect::<Vec<_>>(),
            vec![1, 6]
        );
        assert_eq!(result.valid_evaluation_observation_count, 7);
        assert_eq!(result.score.true_positive_events, 2);
        assert_eq!(result.score.false_positive_alarms, 0);
        assert!(result.score.hard_gate_pass);
    }

    #[test]
    fn ewma_control_chart_uses_disjoint_calibration_and_finds_a_shift() {
        let result = run_scalar_detector(fixture(ScalarDetectorMethod::EwmaControlChart))
            .expect("EWMA should run");
        assert!(result.detections.iter().any(|d| d.cycle_index == 1));
        assert_eq!(result.score.invalid_evidence_alarms, 0);
    }

    #[test]
    fn page_hinkley_detects_separate_upward_shifts_without_repeated_alarm_storms() {
        let result = run_scalar_detector(fixture(ScalarDetectorMethod::PageHinkley))
            .expect("Page-Hinkley should run");
        assert_eq!(
            result.detections.iter().map(|d| d.cycle_index).collect::<Vec<_>>(),
            vec![1, 6]
        );
        assert_eq!(result.score.true_positive_events, 2);
        assert_eq!(result.score.false_positive_alarms, 0);
        assert_eq!(result.score.missed_events, 0);
    }

    #[test]
    fn invalid_samples_never_update_or_rearm_detector() {
        let mut input = fixture(ScalarDetectorMethod::FixedThreshold);
        input.observations[3].value = Some(1000.0);
        let error = run_scalar_detector(input).unwrap_err().to_string();
        assert!(error.contains("represented as missing, not a sentinel"));
    }

    #[test]
    fn rejects_bad_cycles_configuration_and_nonfinite_calibration() {
        let mut bad_cycle = fixture(ScalarDetectorMethod::FixedThreshold);
        bad_cycle.observations[0].cycle_index = 7;
        assert!(run_scalar_detector(bad_cycle).is_err());

        let mut bad_config = fixture(ScalarDetectorMethod::FixedThreshold);
        bad_config.detector.ewma_alpha = Some(0.5);
        assert!(run_scalar_detector(bad_config).is_err());

        let mut bad_calibration = fixture(ScalarDetectorMethod::EwmaControlChart);
        bad_calibration.calibration_values[1] = f64::NAN;
        assert!(run_scalar_detector(bad_calibration).is_err());
    }

    #[test]
    fn canonical_input_digest_is_invariant_to_ground_truth_event_order() {
        let mut first = fixture(ScalarDetectorMethod::FixedThreshold);
        let mut second = first.clone();
        second.drift_events.reverse();
        let a = run_scalar_detector(first.clone()).expect("first run");
        let b = run_scalar_detector(second).expect("reordered ground truth run");
        assert_eq!(a.canonical_input_sha256, b.canonical_input_sha256);
        assert_eq!(a.score.event_f1, b.score.event_f1);
        assert_eq!(a.result_sha256, b.result_sha256);
        // Observation order is semantic and must not be normalized away.
        first.observations.swap(0, 1);
        assert!(run_scalar_detector(first).is_err());
    }

    #[test]
    fn canonical_digests_and_results_are_reproducible() {
        let input = fixture(ScalarDetectorMethod::FixedThreshold);
        let first = run_scalar_detector(input.clone()).expect("first run scores");
        let second = run_scalar_detector(input).expect("second run scores");
        assert_eq!(first.canonical_input_sha256, second.canonical_input_sha256);
        assert_eq!(first.configuration_digest, second.configuration_digest);
        assert_eq!(first.result_sha256, second.result_sha256);
    }
}
