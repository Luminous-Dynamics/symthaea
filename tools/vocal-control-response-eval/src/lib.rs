//! VCRB-001: a dependency-free evaluator for score-conditioned vocal controls.
//!
//! The evaluator consumes measurements produced by independent acoustic analyzers.
//! It does not render audio, extract F0, infer phonemes, or judge naturalness.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvaluationError {
    pub message: String,
}

impl EvaluationError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for EvaluationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for EvaluationError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetricContract {
    pub metric_id: String,
    pub unit: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollateralLimit {
    pub metric: MetricContract,
    pub maximum_absolute_delta: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlSpec {
    pub control_id: String,
    pub requested_delta: f64,
    pub requested_unit: String,
    pub target: MetricContract,
    pub expected_target_delta_min: f64,
    pub expected_target_delta_max: f64,
    pub minimum_target_in_window_fraction: f64,
    pub minimum_pairs: usize,
    pub collateral_limits: Vec<CollateralLimit>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricObservation {
    pub metric_id: String,
    pub unit: String,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PairedObservation {
    /// Identifies matched conditions, such as the same seed and source take.
    pub pair_id: String,
    pub baseline: Vec<MetricObservation>,
    pub intervention: Vec<MetricObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlTrial {
    pub source_case_id: String,
    pub control_id: String,
    pub pairs: Vec<PairedObservation>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MetricDeltaSummary {
    pub metric: MetricContract,
    pub mean_delta: f64,
    pub sample_standard_deviation: f64,
    pub minimum_delta: f64,
    pub maximum_delta: f64,
    pub maximum_absolute_delta: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TargetResponse {
    pub summary: MetricDeltaSummary,
    pub expected_delta_min: f64,
    pub expected_delta_max: f64,
    pub in_window_fraction: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CollateralResponse {
    pub summary: MetricDeltaSummary,
    pub maximum_allowed_absolute_delta: f64,
    pub tolerance_utilization: Option<f64>,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlResponseRow {
    pub control_id: String,
    pub requested_delta: f64,
    pub requested_unit: String,
    /// Target acoustic delta divided by requested control delta.
    pub empirical_response_gain: f64,
    pub target: TargetResponse,
    pub collateral: Vec<CollateralResponse>,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ControlResponseMatrixReport {
    pub source_case_id: String,
    pub pair_count: usize,
    pub metrics: Vec<MetricContract>,
    pub controls: Vec<ControlResponseRow>,
    pub passed: bool,
}

/// Evaluate a complete one-factor-at-a-time matrix for one frozen source case.
///
/// Every row must use the same ordered pair IDs and monitored metric universe.
/// Each control has one target metric and explicit tolerances for every other metric.
/// Missing, extra, duplicate, non-finite, or unit-mismatched observations fail closed.
pub fn evaluate_control_response_matrix(
    specs: &[ControlSpec],
    trials: &[ControlTrial],
) -> Result<ControlResponseMatrixReport, EvaluationError> {
    if specs.is_empty() {
        return Err(EvaluationError::new("at least one control specification is required"));
    }
    if specs.len() != trials.len() {
        return Err(EvaluationError::new(format!(
            "one trial is required per control specification: {} specs, {} trials",
            specs.len(), trials.len()
        )));
    }

    let mut specs_by_id = HashMap::new();
    let mut common_metrics: Option<BTreeMap<String, String>> = None;
    for spec in specs {
        validate_control_spec(spec)?;
        if specs_by_id.insert(spec.control_id.as_str(), spec).is_some() {
            return Err(EvaluationError::new(format!(
                "duplicate control specification: {}", spec.control_id
            )));
        }
        let metrics = expected_metrics(spec);
        if let Some(expected) = &common_metrics {
            if expected != &metrics {
                return Err(EvaluationError::new(
                    "all control rows must observe the same metric IDs and units"
                ));
            }
        } else {
            common_metrics = Some(metrics);
        }
    }

    let mut trials_by_id = HashMap::new();
    for trial in trials {
        require_non_empty(&trial.source_case_id, "source_case_id")?;
        if trials_by_id.insert(trial.control_id.as_str(), trial).is_some() {
            return Err(EvaluationError::new(format!(
                "duplicate trial for control: {}", trial.control_id
            )));
        }
        if !specs_by_id.contains_key(trial.control_id.as_str()) {
            return Err(EvaluationError::new(format!(
                "trial has no matching specification: {}", trial.control_id
            )));
        }
    }

    let first_trial = trials.first().expect("non-empty trials checked above");
    let source_case_id = first_trial.source_case_id.clone();
    let pair_ids = validate_pair_ids(first_trial)?;
    let pair_count = pair_ids.len();

    let mut controls = Vec::with_capacity(specs.len());
    for spec in specs {
        let trial = trials_by_id.get(spec.control_id.as_str()).copied().ok_or_else(|| {
            EvaluationError::new(format!("missing trial for control: {}", spec.control_id))
        })?;

        if trial.source_case_id != source_case_id {
            return Err(EvaluationError::new(format!(
                "source case mismatch for control {}: expected {}, got {}",
                spec.control_id, source_case_id, trial.source_case_id
            )));
        }

        if validate_pair_ids(trial)? != pair_ids {
            return Err(EvaluationError::new(format!(
                "matched pair IDs/order differ for control {}", spec.control_id
            )));
        }
        if pair_count < spec.minimum_pairs {
            return Err(EvaluationError::new(format!(
                "control {} requires at least {} matched pairs; got {}",
                spec.control_id, spec.minimum_pairs, pair_count
            )));
        }
        controls.push(evaluate_one_control(spec, trial)?);
    }

    let metrics = common_metrics.expect("non-empty specifications checked above")
        .into_iter()
        .map(|(metric_id, unit)| MetricContract {
            metric_id,
            unit,
        })
        .collect::<Vec<_>>();
    let passed = controls.iter().all(|row| row.passed);

    Ok(ControlResponseMatrixReport {
        source_case_id,
        pair_count,
        metrics,
        controls,
        passed,
    })
}

fn validate_control_spec(spec: &ControlSpec) -> Result<(), EvaluationError> {
    require_non_empty(&spec.control_id, "control_id")?;
    require_non_empty(&spec.requested_unit, "requested_unit")?;
    if !spec.requested_delta.is_finite() || spec.requested_delta == 0.0 {
        return Err(EvaluationError::new(format!(
            "control {} requested_delta must be finite and non-zero", spec.control_id
        )));
    }
    require_non_empty(&spec.target.metric_id, "target metric_id")?;
    require_non_empty(&spec.target.unit, "target unit")?;
    if !spec.expected_target_delta_min.is_finite()
        || !spec.expected_target_delta_max.is_finite()
        || spec.expected_target_delta_min > spec.expected_target_delta_max
    {
        return Err(EvaluationError::new(format!(
            "control {} has an invalid expected target-delta interval", spec.control_id
        )));
    }
    if spec.expected_target_delta_min <= 0.0 && spec.expected_target_delta_max >= 0.0 {
        return Err(EvaluationError::new(format!(
            "control {} expected target-delta interval must exclude zero", spec.control_id
        )));
    }
    if !spec.minimum_target_in_window_fraction.is_finite()
        || spec.minimum_target_in_window_fraction <= 0.0
        || spec.minimum_target_in_window_fraction > 1.0
    {
        return Err(EvaluationError::new(format!(
            "control {} minimum_target_in_window_fraction must be in (0, 1]",
            spec.control_id
        )));
    }
    if spec.minimum_pairs < 2 {
        return Err(EvaluationError::new(format!(
            "control {} minimum_pairs must be at least two", spec.control_id
        )));
    }
    if spec.collateral_limits.is_empty() {
        return Err(EvaluationError::new(format!(
            "control {} must declare at least one collateral metric", spec.control_id
        )));
    }

    let mut ids = HashSet::new();
    ids.insert(spec.target.metric_id.as_str());
    for collateral in &spec.collateral_limits {
        require_non_empty(&collateral.metric.metric_id, "collateral metric_id")?;
        require_non_empty(&collateral.metric.unit, "collateral metric unit")?;
        if !ids.insert(collateral.metric.metric_id.as_str()) {
            return Err(EvaluationError::new(format!(
                "control {} has duplicate or target-overlapping metric {}",
                spec.control_id, collateral.metric.metric_id
            )));
        }
        if !collateral.maximum_absolute_delta.is_finite()
            || collateral.maximum_absolute_delta < 0.0
        {
            return Err(EvaluationError::new(format!(
                "control {} has an invalid collateral tolerance for {}",
                spec.control_id, collateral.metric.metric_id
            )));
        }
    }
    Ok(())
}

fn require_non_empty(value: &str, field: &str) -> Result<(), EvaluationError> {
    if value.trim().is_empty() {
        Err(EvaluationError::new(format!("{field} must not be empty")))
    } else {
        Ok(())
    }
}

fn expected_metrics(spec: &ControlSpec) -> BTreeMap<String, String> {
    let mut metrics = BTreeMap::new();
    metrics.insert(spec.target.metric_id.clone(), spec.target.unit.clone());
    for collateral in &spec.collateral_limits {
        metrics.insert(collateral.metric.metric_id.clone(), collateral.metric.unit.clone());
    }
    metrics
}

fn validate_pair_ids(trial: &ControlTrial) -> Result<Vec<String>, EvaluationError> {
    if trial.pairs.is_empty() {
        return Err(EvaluationError::new(format!(
            "control {} has no paired observations", trial.control_id
        )));
    }
    let mut seen = HashSet::new();
    let mut ids = Vec::with_capacity(trial.pairs.len());
    for pair in &trial.pairs {
        require_non_empty(&pair.pair_id, "pair_id")?;
        if !seen.insert(pair.pair_id.as_str()) {
            return Err(EvaluationError::new(format!(
                "duplicate pair_id {} for control {}", pair.pair_id, trial.control_id
            )));
        }
        ids.push(pair.pair_id.clone());
    }
    Ok(ids)
}

fn evaluate_one_control(
    spec: &ControlSpec,
    trial: &ControlTrial,
) -> Result<ControlResponseRow, EvaluationError> {
    let expected = expected_metrics(spec);
    let mut deltas: HashMap<String, Vec<f64>> = expected.keys()
        .map(|metric_id| (metric_id.clone(), Vec::with_capacity(trial.pairs.len())))
        .collect();

    for pair in &trial.pairs {
        let baseline = validate_observations(&pair.baseline, &expected, "baseline")?;
        let intervention = validate_observations(&pair.intervention, &expected, "intervention")?;

        for (metric_id, values) in &mut deltas {
            let before = baseline[metric_id.as_str()];
            let after = intervention[metric_id.as_str()];
            let delta = after - before;
            if !delta.is_finite() {
                return Err(EvaluationError::new(format!(
                    "non-finite delta for metric {metric_id} in pair {}", pair.pair_id
                )));
            }
            values.push(delta);
        }
    }

    let target_deltas = &deltas[&spec.target.metric_id];
    let target_summary = summarize(&spec.target, target_deltas)?;
    let in_window_count = target_deltas.iter().filter(|delta| {
        **delta >= spec.expected_target_delta_min
            && **delta <= spec.expected_target_delta_max
    }).count();
    let in_window_fraction = in_window_count as f64 / target_deltas.len() as f64;
    let target_passed = target_summary.mean_delta >= spec.expected_target_delta_min
        && target_summary.mean_delta <= spec.expected_target_delta_max
        && in_window_fraction >= spec.minimum_target_in_window_fraction;

    let target = TargetResponse {
        summary: target_summary,
        expected_delta_min: spec.expected_target_delta_min,
        expected_delta_max: spec.expected_target_delta_max,
        in_window_fraction,
        passed: target_passed,
    };

    let empirical_response_gain = target.summary.mean_delta / spec.requested_delta;
    if !empirical_response_gain.is_finite() {
        return Err(EvaluationError::new(format!(
            "non-finite empirical response gain for control {}", spec.control_id
        )));
    }

    let mut collateral = Vec::with_capacity(spec.collateral_limits.len());
    for limit in &spec.collateral_limits {
        let summary = summarize(&limit.metric, &deltas[&limit.metric.metric_id])?;
        let passed = summary.maximum_absolute_delta <= limit.maximum_absolute_delta;
        let tolerance_utilization = (limit.maximum_absolute_delta > 0.0)
            .then_some(summary.maximum_absolute_delta / limit.maximum_absolute_delta);
        collateral.push(CollateralResponse {
            summary,
            maximum_allowed_absolute_delta: limit.maximum_absolute_delta,
            tolerance_utilization,
            passed,
        });
    }

    let passed = target.passed && collateral.iter().all(|metric| metric.passed);
    Ok(ControlResponseRow {
        control_id: spec.control_id.clone(),
        requested_delta: spec.requested_delta,
        requested_unit: spec.requested_unit.clone(),
        empirical_response_gain,
        target,
        collateral,
        passed,
    })
}

fn validate_observations<'a>(
    observations: &'a [MetricObservation],
    expected: &BTreeMap<String, String>,
    phase: &str,
) -> Result<HashMap<&'a str, f64>, EvaluationError> {
    if observations.len() != expected.len() {
        return Err(EvaluationError::new(format!(
            "{phase} observation count mismatch: expected {}, got {}",
            expected.len(), observations.len()
        )));
    }

    let mut values = HashMap::with_capacity(observations.len());
    for observation in observations {
        require_non_empty(&observation.metric_id, "observation metric_id")?;
        require_non_empty(&observation.unit, "observation unit")?;
        if !observation.value.is_finite() {
            return Err(EvaluationError::new(format!(
                "non-finite observation for metric {}", observation.metric_id
            )));
        }
        let expected_unit = expected.get(&observation.metric_id).ok_or_else(|| {
            EvaluationError::new(format!(
                "unexpected {phase} metric {}", observation.metric_id
            ))
        })?;
        if expected_unit != &observation.unit {
            return Err(EvaluationError::new(format!(
                "unit mismatch for metric {}: expected {}, got {}",
                observation.metric_id, expected_unit, observation.unit
            )));
        }
        if values.insert(observation.metric_id.as_str(), observation.value).is_some() {
            return Err(EvaluationError::new(format!(
                "duplicate {phase} observation for metric {}", observation.metric_id
            )));
        }
    }
    Ok(values)
}

fn summarize(
    metric: &MetricContract,
    deltas: &[f64],
) -> Result<MetricDeltaSummary, EvaluationError> {
    if deltas.len() < 2 {
        return Err(EvaluationError::new(format!(
            "metric {} requires at least two paired deltas", metric.metric_id
        )));
    }

    let mut mean = 0.0f64;
    let mut sum_squared_deviations = 0.0f64;
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut maximum_absolute: f64 = 0.0;

    for (index, delta) in deltas.iter().copied().enumerate() {
        if !delta.is_finite() {
            return Err(EvaluationError::new(format!(
                "non-finite delta for metric {}", metric.metric_id
            )));
        }
        minimum = minimum.min(delta);
        maximum = maximum.max(delta);
        maximum_absolute = maximum_absolute.max(delta.abs());

        let count = (index + 1) as f64;
        let difference = delta - mean;
        mean += difference / count;
        let difference_after_update = delta - mean;
        sum_squared_deviations += difference * difference_after_update;
        if !mean.is_finite() || !sum_squared_deviations.is_finite() {
            return Err(EvaluationError::new(format!(
                "numeric overflow summarizing metric {}", metric.metric_id
            )));
        }
    }

    let sample_variance = (sum_squared_deviations / (deltas.len() - 1) as f64).max(0.0);
    let sample_standard_deviation = sample_variance.sqrt();
    if !sample_standard_deviation.is_finite() || !maximum_absolute.is_finite() {
        return Err(EvaluationError::new(format!(
            "non-finite summary for metric {}", metric.metric_id
        )));
    }

    Ok(MetricDeltaSummary {
        metric: metric.clone(),
        mean_delta: mean,
        sample_standard_deviation,
        minimum_delta: minimum,
        maximum_delta: maximum,
        maximum_absolute_delta: maximum_absolute,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(metric_id: &str, unit: &str, value: f64) -> MetricObservation {
        MetricObservation {
            metric_id: metric_id.into(),
            unit: unit.into(),
            value,
        }
    }

    fn pair(
        pair_id: &str,
        base_pitch: f64,
        edit_pitch: f64,
        base_vibrato: f64,
        edit_vibrato: f64,
        base_energy: f64,
        edit_energy: f64,
    ) -> PairedObservation {
        PairedObservation {
            pair_id: pair_id.into(),
            baseline: vec![
                metric("pitch_error_cents", "cents", base_pitch),
                metric("vibrato_depth_cents", "cents", base_vibrato),
                metric("phrase_energy_db", "dB", base_energy),
            ],
            intervention: vec![
                metric("pitch_error_cents", "cents", edit_pitch),
                metric("vibrato_depth_cents", "cents", edit_vibrato),
                metric("phrase_energy_db", "dB", edit_energy),
            ],
        }
    }

    fn pitch_spec() -> ControlSpec {
        ControlSpec {
            control_id: "pitch_offset".into(),
            requested_delta: 200.0,
            requested_unit: "cents".into(),
            target: MetricContract {
                metric_id: "pitch_error_cents".into(),
                unit: "cents".into(),
            },
            expected_target_delta_min: 180.0,
            expected_target_delta_max: 220.0,
            minimum_target_in_window_fraction: 0.8,
            minimum_pairs: 3,
            collateral_limits: vec![
                CollateralLimit {
                    metric: MetricContract {
                        metric_id: "vibrato_depth_cents".into(),
                        unit: "cents".into(),
                    },
                    maximum_absolute_delta: 0.5,
                },
                CollateralLimit {
                    metric: MetricContract {
                        metric_id: "phrase_energy_db".into(),
                        unit: "dB".into(),
                    },
                    maximum_absolute_delta: 0.1,
                },
            ],
        }
    }

    fn vibrato_spec() -> ControlSpec {
        ControlSpec {
            control_id: "vibrato_depth".into(),
            requested_delta: 10.0,
            requested_unit: "cents".into(),
            target: MetricContract {
                metric_id: "vibrato_depth_cents".into(),
                unit: "cents".into(),
            },
            expected_target_delta_min: 8.0,
            expected_target_delta_max: 12.0,
            minimum_target_in_window_fraction: 0.8,
            minimum_pairs: 3,
            collateral_limits: vec![
                CollateralLimit {
                    metric: MetricContract {
                        metric_id: "pitch_error_cents".into(),
                        unit: "cents".into(),
                    },
                    maximum_absolute_delta: 0.5,
                },
                CollateralLimit {
                    metric: MetricContract {
                        metric_id: "phrase_energy_db".into(),
                        unit: "dB".into(),
                    },
                    maximum_absolute_delta: 0.1,
                },
            ],
        }
    }

    fn valid_trials() -> Vec<ControlTrial> {
        vec![
            ControlTrial {
                source_case_id: "heldout-phrase-001".into(),
                control_id: "pitch_offset".into(),
                pairs: vec![
                    pair("seed-1", 0.0, 200.0, 30.0, 30.0, -10.0, -10.0),
                    pair("seed-2", 0.0, 210.0, 30.0, 30.0, -10.0, -10.0),
                    pair("seed-3", 0.0, 190.0, 30.0, 30.0, -10.0, -10.0),
                ],
            },
            ControlTrial {
                source_case_id: "heldout-phrase-001".into(),
                control_id: "vibrato_depth".into(),
                pairs: vec![
                    pair("seed-1", 0.0, 0.0, 30.0, 39.0, -10.0, -10.0),
                    pair("seed-2", 0.0, 0.0, 30.0, 40.0, -10.0, -10.0),
                    pair("seed-3", 0.0, 0.0, 30.0, 41.0, -10.0, -10.0),
                ],
            },
        ]
    }

    #[test]
    fn valid_matrix_passes_and_exposes_cross_coupling_rows() {
        let report = evaluate_control_response_matrix(
            &[pitch_spec(), vibrato_spec()], &valid_trials()
        ).unwrap();
        assert!(report.passed);
        assert_eq!(report.pair_count, 3);
        assert_eq!(report.metrics.len(), 3);
        assert_eq!(report.controls.len(), 2);
        assert!((report.controls[0].target.summary.mean_delta - 2.0).abs() < 1e-12);
        assert!((report.controls[0].empirical_response_gain - 1.0).abs() < 1e-12);
        assert!((report.controls[1].target.summary.mean_delta - 10.0).abs() < 1e-12);
        assert!(report.controls.iter().all(|row| row.passed));
    }

    #[test]
    fn wrong_direction_target_response_fails() {
        let mut trials = valid_trials();
        for pair in &mut trials[0].pairs {
            for observation in &mut pair.intervention {
                if observation.metric_id == "pitch_error_cents" {
                    observation.value = -200.0;
                }
            }
        }
        let report = evaluate_control_response_matrix(
            &[pitch_spec(), vibrato_spec()], &trials
        ).unwrap();
        assert!(!report.passed);
        assert!(!report.controls[0].target.passed);
    }

    #[test]
    fn no_op_target_response_fails() {
        let mut trials = valid_trials();
        for pair in &mut trials[1].pairs {
            for observation in &mut pair.intervention {
                if observation.metric_id == "vibrato_depth_cents" {
                    observation.value = 30.0;
                }
            }
        }
        let report = evaluate_control_response_matrix(
            &[pitch_spec(), vibrato_spec()], &trials
        ).unwrap();
        assert!(!report.controls[1].target.passed);
    }

    #[test]
    fn off_diagonal_acoustic_change_fails_even_when_target_is_correct() {
        let mut trials = valid_trials();
        for pair in &mut trials[0].pairs {
            for observation in &mut pair.intervention {
                if observation.metric_id == "vibrato_depth_cents" {
                    observation.value = 31.0;
                }
            }
        }
        let report = evaluate_control_response_matrix(
            &[pitch_spec(), vibrato_spec()], &trials
        ).unwrap();
        assert!(!report.controls[0].passed);
        assert!(report.controls[0].target.passed);
        assert!(!report.controls[0].collateral[0].passed);
    }

    #[test]
    fn target_pair_fraction_is_enforced_not_only_the_mean() {
        let pairs = vec![
            pair("seed-1", 0.0, 0.0, 30.0, 30.0, -10.0, -10.0),
            pair("seed-2", 0.0, 200.0, 30.0, 30.0, -10.0, -10.0),
            pair("seed-3", 0.0, 200.0, 30.0, 30.0, -10.0, -10.0),
            pair("seed-4", 0.0, 200.0, 30.0, 30.0, -10.0, -10.0),
        ];
        let trial = ControlTrial {
            source_case_id: "heldout-phrase-001".into(),
            control_id: "pitch_offset".into(),
            pairs,
        };
        let report = evaluate_control_response_matrix(&[pitch_spec()], &[trial]).unwrap();
        assert!(!report.controls[0].target.passed);
        assert!((report.controls[0].target.in_window_fraction - 0.75).abs() < 1e-12);
    }

    #[test]
    fn missing_or_extra_metrics_fail_closed() {
        let mut trials = valid_trials();
        trials[0].pairs[0].baseline.pop();
        assert!(evaluate_control_response_matrix(
            &[pitch_spec(), vibrato_spec()], &trials
        ).is_err());

        let mut trials = valid_trials();
        trials[0].pairs[0].intervention.push(metric("untracked", "unit", 1.0));
        assert!(evaluate_control_response_matrix(
            &[pitch_spec(), vibrato_spec()], &trials
        ).is_err());
    }

    #[test]
    fn duplicate_and_non_finite_observations_fail_closed() {
        let mut trials = valid_trials();
        trials[0].pairs[0].baseline.push(metric("pitch_error_cents", "cents", 2.0));
        assert!(evaluate_control_response_matrix(
            &[pitch_spec(), vibrato_spec()], &trials
        ).is_err());

        let mut trials = valid_trials();
        trials[0].pairs[0].intervention[0].value = f64::NAN;
        assert!(evaluate_control_response_matrix(
            &[pitch_spec(), vibrato_spec()], &trials
        ).is_err());
    }

    #[test]
    fn unit_mismatch_is_rejected() {
        let mut trials = valid_trials();
        trials[1].pairs[0].intervention[1].unit = "Hz".into();
        assert!(evaluate_control_response_matrix(
            &[pitch_spec(), vibrato_spec()], &trials
        ).is_err());
    }

    #[test]
    fn mismatched_source_or_pair_ids_are_rejected() {
        let mut trials = valid_trials();
        trials[1].source_case_id = "different-phrase".into();
        assert!(evaluate_control_response_matrix(
            &[pitch_spec(), vibrato_spec()], &trials
        ).is_err());

        let mut trials = valid_trials();
        trials[1].pairs[0].pair_id = "different-seed".into();
        assert!(evaluate_control_response_matrix(
            &[pitch_spec(), vibrato_spec()], &trials
        ).is_err());
    }

    #[test]
    fn malformed_specs_and_missing_trials_are_rejected() {
        let mut bad_spec = pitch_spec();
        bad_spec.requested_delta = 0.0;
        assert!(evaluate_control_response_matrix(
            &[bad_spec], &[valid_trials()[0].clone()]
        ).is_err());

        let mut bad_spec = pitch_spec();
        bad_spec.expected_target_delta_min = -1.0;
        bad_spec.expected_target_delta_max = 1.0;
        assert!(evaluate_control_response_matrix(
            &[bad_spec], &[valid_trials()[0].clone()]
        ).is_err());

        assert!(evaluate_control_response_matrix(
            &[pitch_spec(), vibrato_spec()], &[valid_trials()[0].clone()]
        ).is_err());
    }

    #[test]
    fn insufficient_repetitions_are_rejected() {
        let mut trial = valid_trials().remove(0);
        trial.pairs.pop();
        assert!(evaluate_control_response_matrix(&[pitch_spec()], &[trial]).is_err());
    }

    #[test]
    fn zero_collateral_tolerance_requires_exact_metric_preservation() {
        let mut spec = pitch_spec();
        spec.collateral_limits[0].maximum_absolute_delta = 0.0;
        let mut trial = valid_trials().remove(0);
        for pair in &mut trial.pairs {
            for observation in &mut pair.intervention {
                if observation.metric_id == "vibrato_depth_cents" {
                    observation.value += 0.25;
                }
            }
        }
        let report = evaluate_control_response_matrix(&[spec], &[trial]).unwrap();
        assert!(!report.controls[0].collateral[0].passed);
        assert_eq!(report.controls[0].collateral[0].tolerance_utilization, None);
    }
}
