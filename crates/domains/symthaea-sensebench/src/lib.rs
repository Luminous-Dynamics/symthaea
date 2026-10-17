// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, hardware-free starter benchmark for Symthaea Sense.
//!
//! This harness validates generator and evaluation plumbing. Synthetic results
//! are not evidence that any physical phenomenon exists and are not a substitute
//! for reference instruments or independently collected data.

use serde::{Deserialize, Serialize};

pub mod experiment_design;
pub mod causal_benchmark;
pub mod scm_graph_benchmark;
pub mod blind_world_benchmark;
pub mod geophysics;
pub mod robustness_benchmark;

pub const GENERATOR_VERSION: &str = "sensebench-0.1.0";
pub const MANIFEST_SCHEMA_VERSION: u16 = 1;
/// Keep CLI and test fixtures bounded so a malformed invocation cannot
/// unexpectedly allocate unbounded benchmark memory.
pub const MAX_SAMPLE_COUNT: usize = 1_000_000;

/// Scenario labels belong to the evaluator, not the detector-facing stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioKind {
    Null,
    StepChange,
    TransientSpike,
    BiasDrift,
    Dropout,
}

impl ScenarioKind {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "null" => Some(Self::Null),
            "step" | "step_change" => Some(Self::StepChange),
            "spike" | "transient_spike" => Some(Self::TransientSpike),
            "drift" | "bias_drift" => Some(Self::BiasDrift),
            "dropout" => Some(Self::Dropout),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Null => "null",
            Self::StepChange => "step_change",
            Self::TransientSpike => "transient_spike",
            Self::BiasDrift => "bias_drift",
            Self::Dropout => "dropout",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    StepChange,
    TransientSpike,
    BiasDrift,
    Dropout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationSplit {
    Development,
    Tuning,
    BlindEvaluation,
}

/// Evaluator-only metadata. Do not pass this object, its seed, or its ground
/// truth fields to a detector intended to simulate blind deployment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScenarioManifest {
    pub schema_version: u16,
    pub generator_version: String,
    pub scenario_id: String,
    pub scenario_kind: ScenarioKind,
    pub seed: u64,
    pub sample_count: usize,
    pub sample_interval_ms: u64,
    pub split: EvaluationSplit,
}

/// Detector-visible input: no event labels, scenario kind, or random seed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamPoint {
    pub tick: u64,
    /// None is a missing measurement, not a zero-valued observation.
    pub value: Option<f64>,
}

/// Evaluator-only ground truth. end_tick is inclusive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TruthEvent {
    pub kind: EventKind,
    pub start_tick: u64,
    pub end_tick: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scenario {
    pub manifest: ScenarioManifest,
    pub observations: Vec<StreamPoint>,
    pub truth_events: Vec<TruthEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnomalyScore {
    pub tick: u64,
    /// None means no numeric sample was observed at this tick.
    pub score: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetectionMetrics {
    pub evaluated_samples: usize,
    pub missing_samples: usize,
    pub true_positive_samples: usize,
    pub false_positive_samples: usize,
    pub false_negative_samples: usize,
    pub true_negative_samples: usize,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    /// Average precision (area under the stepwise precision-recall curve).
    pub average_precision: Option<f64>,
    pub event_recall: Option<f64>,
    pub false_alarms_per_sensor_hour: f64,
    pub mean_detection_delay_ticks: Option<f64>,
}

/// Generate deterministic scenarios using a local, versioned SplitMix64 stream.
///
/// The initial implementation intentionally keeps the generator dependency-free
/// apart from serialization. Identical version, kind, seed, count and interval
/// produce byte-stable field values; JSON byte stability is tested in this crate.
pub fn generate_scenario(
    kind: ScenarioKind,
    seed: u64,
    sample_count: usize,
    sample_interval_ms: u64,
    split: EvaluationSplit,
) -> Result<Scenario, &'static str> {
    if sample_count < 64 {
        return Err("sample_count must be >= 64 to preserve a baseline window");
    }
    if sample_count > MAX_SAMPLE_COUNT {
        return Err("sample_count exceeds MAX_SAMPLE_COUNT");
    }
    if sample_interval_ms == 0 {
        return Err("sample_interval_ms must be > 0");
    }

    let mut rng = SplitMix64::new(seed);
    let start = sample_count / 2;
    let spike_tick = start + (sample_count / 8).max(1);
    let dropout_end = (start + (sample_count / 16).max(1)).min(sample_count - 1);
    let mut observations = Vec::with_capacity(sample_count);

    for tick in 0..sample_count {
        let noise = rng.noise(0.4);
        let mut value = 10.0 + noise;
        match kind {
            ScenarioKind::Null => {}
            ScenarioKind::StepChange if tick >= start => value += 2.5,
            ScenarioKind::TransientSpike if tick == spike_tick => value += 5.0,
            ScenarioKind::BiasDrift if tick >= start => {
                value += 3.0 * (tick - start) as f64 / (sample_count - start) as f64;
            }
            ScenarioKind::Dropout if (start..=dropout_end).contains(&tick) => {
                observations.push(StreamPoint {
                    tick: tick as u64,
                    value: None,
                });
                continue;
            }
            _ => {}
        }

        observations.push(StreamPoint {
            tick: tick as u64,
            value: Some(value),
        });
    }

    let truth_events = match kind {
        ScenarioKind::Null => Vec::new(),
        ScenarioKind::StepChange => vec![TruthEvent {
            kind: EventKind::StepChange,
            start_tick: start as u64,
            end_tick: (sample_count - 1) as u64,
        }],
        ScenarioKind::TransientSpike => vec![TruthEvent {
            kind: EventKind::TransientSpike,
            start_tick: spike_tick as u64,
            end_tick: spike_tick as u64,
        }],
        ScenarioKind::BiasDrift => vec![TruthEvent {
            kind: EventKind::BiasDrift,
            start_tick: start as u64,
            end_tick: (sample_count - 1) as u64,
        }],
        ScenarioKind::Dropout => vec![TruthEvent {
            kind: EventKind::Dropout,
            start_tick: start as u64,
            end_tick: dropout_end as u64,
        }],
    };

    Ok(Scenario {
        manifest: ScenarioManifest {
            schema_version: MANIFEST_SCHEMA_VERSION,
            generator_version: GENERATOR_VERSION.to_owned(),
            scenario_id: format!("{}-seed-{seed:016x}-n-{sample_count}", kind.as_str()),
            scenario_kind: kind,
            seed,
            sample_count,
            sample_interval_ms,
            split,
        },
        observations,
        truth_events,
    })
}

/// Score each observed value against a fixed initial reference window.
///
/// This is an intentionally simple baseline, not a state-of-the-art detector:
/// it makes leakage and benchmark comparisons easy to inspect. Missing samples
/// receive no numeric score and must be evaluated as an integrity/fault channel.
pub fn z_score_baseline(points: &[StreamPoint], baseline_samples: usize) -> Vec<AnomalyScore> {
    if points.is_empty() {
        return Vec::new();
    }
    let baseline_limit = baseline_samples.min(points.len()).max(1);
    let base: Vec<f64> = points
        .iter()
        .take(baseline_limit)
        .filter_map(|point| point.value)
        .filter(|value| value.is_finite())
        .collect();

    if base.is_empty() {
        return points
            .iter()
            .map(|point| AnomalyScore { tick: point.tick, score: None })
            .collect();
    }

    let mean = base.iter().sum::<f64>() / base.len() as f64;
    let variance = base.iter().map(|value| (value - mean).powi(2)).sum::<f64>()
        / base.len() as f64;
    let sigma = variance.sqrt().max(1.0e-9);

    points.iter().map(|point| AnomalyScore {
        tick: point.tick,
        score: point.value.filter(|value| value.is_finite())
            .map(|value| (value - mean).abs() / sigma),
    }).collect()
}

/// Evaluate numeric anomalies and missing-measurement events separately.
///
/// Missing data can detect a dropout, but it is not coerced into an artificial
/// numeric anomaly score. Numeric point metrics exclude missing samples and
/// event recall uses a separate missingness rule for dropout events.
pub fn evaluate(
    scenario: &Scenario,
    scores: &[AnomalyScore],
    threshold: f64,
) -> DetectionMetrics {
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut fn_ = 0usize;
    let mut tn = 0usize;
    let mut missing = 0usize;
    let mut evaluated = 0usize;
    let mut false_alarm_episodes = 0usize;
    let mut last_false_alarm_tick: Option<u64> = None;
    let mut ranked: Vec<(f64, bool, u64)> = Vec::new();

    for point in &scenario.observations {
        let Some(value) = point.value else {
            missing += 1;
            last_false_alarm_tick = None;
            continue;
        };
        if !value.is_finite() {
            continue;
        }
        evaluated += 1;
        let positive = scenario.truth_events.iter().any(|event| {
            event.kind != EventKind::Dropout
                && (event.start_tick..=event.end_tick).contains(&point.tick)
        });
        let score = scores.iter().find(|score| score.tick == point.tick)
            .and_then(|score| score.score).unwrap_or(0.0);
        let predicted = score >= threshold;
        ranked.push((score, positive, point.tick));
        match (positive, predicted) {
            (true, true) => {
                tp += 1;
                last_false_alarm_tick = None;
            }
            (false, true) => {
                fp += 1;
                if last_false_alarm_tick != Some(point.tick.saturating_sub(1)) {
                    false_alarm_episodes += 1;
                }
                last_false_alarm_tick = Some(point.tick);
            }
            (true, false) => {
                fn_ += 1;
                last_false_alarm_tick = None;
            }
            (false, false) => {
                tn += 1;
                last_false_alarm_tick = None;
            }
        }
    }

    let precision = ratio(tp, tp + fp);
    let recall = ratio(tp, tp + fn_);
    let f1 = match (precision, recall) {
        (Some(p), Some(r)) if p + r > 0.0 => Some(2.0 * p * r / (p + r)),
        (Some(_), Some(_)) => Some(0.0),
        _ => None,
    };
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let positives = ranked.iter().filter(|item| item.1).count();
    let average_precision = if positives == 0 {
        None
    } else {
        // Process tied anomaly scores as a group; otherwise arbitrary row order
        // could change average precision even though the detector scores match.
        let mut index = 0usize;
        let mut cumulative_tp = 0usize;
        let mut cumulative_fp = 0usize;
        let mut previous_recall = 0.0;
        let mut area = 0.0;
        while index < ranked.len() {
            let score = ranked[index].0;
            let mut next = index;
            let mut group_tp = 0usize;
            let mut group_fp = 0usize;
            while next < ranked.len() && ranked[next].0.total_cmp(&score).is_eq() {
                if ranked[next].1 {
                    group_tp += 1;
                } else {
                    group_fp += 1;
                }
                next += 1;
            }
            cumulative_tp += group_tp;
            cumulative_fp += group_fp;
            let recall_at_threshold = cumulative_tp as f64 / positives as f64;
            let precision_at_threshold = cumulative_tp as f64
                / (cumulative_tp + cumulative_fp) as f64;
            area += (recall_at_threshold - previous_recall) * precision_at_threshold;
            previous_recall = recall_at_threshold;
            index = next;
        }
        Some(area)
    };

    let mut detected_events = 0usize;
    let mut delays = Vec::new();
    for event in &scenario.truth_events {
        let detected_tick = if event.kind == EventKind::Dropout {
            scenario.observations.iter()
                .filter(|point| (event.start_tick..=event.end_tick).contains(&point.tick))
                .find(|point| point.value.is_none())
                .map(|point| point.tick)
        } else {
            scores.iter()
                .filter(|score| (event.start_tick..=event.end_tick).contains(&score.tick))
                .filter(|score| score.score.is_some_and(|score| score >= threshold))
                .map(|score| score.tick)
                .min()
        };

        if let Some(tick) = detected_tick {
            detected_events += 1;
            delays.push(tick.saturating_sub(event.start_tick) as f64);
        }
    }

    let event_recall = ratio(detected_events, scenario.truth_events.len());
    let mean_detection_delay_ticks = if delays.is_empty() {
        None
    } else {
        Some(delays.iter().sum::<f64>() / delays.len() as f64)
    };
    let duration_hours = (scenario.manifest.sample_count as f64
        * scenario.manifest.sample_interval_ms as f64) / 3_600_000.0;
    // Count contiguous false-positive runs as alarm episodes, not every
    // above-threshold sample as a separate alarm.
    let false_alarms_per_sensor_hour = if duration_hours > 0.0 {
        false_alarm_episodes as f64 / duration_hours
    } else {
        0.0
    };

    DetectionMetrics {
        evaluated_samples: evaluated,
        missing_samples: missing,
        true_positive_samples: tp,
        false_positive_samples: fp,
        false_negative_samples: fn_,
        true_negative_samples: tn,
        precision,
        recall,
        f1,
        average_precision,
        event_recall,
        false_alarms_per_sensor_hour,
        mean_detection_delay_ticks,
    }
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then_some(numerator as f64 / denominator as f64)
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn noise(&mut self, peak_to_peak: f64) -> f64 {
        let unit = self.next_u64() as f64 / u64::MAX as f64;
        (unit - 0.5) * peak_to_peak
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generated(kind: ScenarioKind, seed: u64) -> Scenario {
        generate_scenario(kind, seed, 512, 1_000, EvaluationSplit::BlindEvaluation).unwrap()
    }

    #[test]
    fn same_manifest_produces_identical_observations_and_json() {
        let a = generated(ScenarioKind::StepChange, 42);
        let b = generated(ScenarioKind::StepChange, 42);
        assert_eq!(a, b);
        assert_eq!(serde_json::to_vec(&a).unwrap(), serde_json::to_vec(&b).unwrap());
    }

    #[test]
    fn null_suite_has_no_truth_events_and_baseline_has_no_false_alarms() {
        let scenario = generated(ScenarioKind::Null, 123);
        assert!(scenario.truth_events.is_empty());
        let scores = z_score_baseline(&scenario.observations, 64);
        let metrics = evaluate(&scenario, &scores, 4.0);
        assert_eq!(metrics.false_positive_samples, 0);
        assert_eq!(metrics.event_recall, None);
    }

    #[test]
    fn contiguous_false_positive_samples_count_as_one_alarm_episode() {
        let scenario = generated(ScenarioKind::Null, 123);
        let scores: Vec<AnomalyScore> = scenario.observations.iter()
            .map(|point| AnomalyScore { tick: point.tick, score: Some(10.0) })
            .collect();
        let metrics = evaluate(&scenario, &scores, 4.0);
        assert_eq!(metrics.false_positive_samples, 512);
        let expected_alarms_per_hour = 3_600_000.0 / (512.0 * 1_000.0);
        assert!(
            (metrics.false_alarms_per_sensor_hour - expected_alarms_per_hour).abs() < 1.0e-12
        );
    }

    #[test]
    fn step_change_is_detected_by_simple_reference_baseline() {
        let scenario = generated(ScenarioKind::StepChange, 123);
        let scores = z_score_baseline(&scenario.observations, 64);
        let metrics = evaluate(&scenario, &scores, 4.0);
        assert_eq!(metrics.event_recall, Some(1.0));
        assert!(metrics.recall.unwrap_or(0.0) > 0.99);
    }

    #[test]
    fn gradual_bias_drift_is_detected_and_delay_is_reported() {
        let scenario = generated(ScenarioKind::BiasDrift, 123);
        let scores = z_score_baseline(&scenario.observations, 64);
        let metrics = evaluate(&scenario, &scores, 4.0);
        assert_eq!(metrics.event_recall, Some(1.0));
        let delay = metrics.mean_detection_delay_ticks.unwrap();
        assert!(delay > 0.0, "gradual drift should take time to cross the threshold");
        assert!(delay < 128.0, "drift detection should occur before the event ends");
    }

    #[test]
    fn isolated_spike_is_detected() {
        let scenario = generated(ScenarioKind::TransientSpike, 123);
        let scores = z_score_baseline(&scenario.observations, 64);
        let metrics = evaluate(&scenario, &scores, 4.0);
        assert_eq!(metrics.event_recall, Some(1.0));
        assert_eq!(metrics.true_positive_samples, 1);
        assert_eq!(metrics.average_precision, Some(1.0));
    }

    #[test]
    fn average_precision_groups_tied_scores_without_order_bias() {
        let scenario = generated(ScenarioKind::StepChange, 123);
        let tied_scores: Vec<AnomalyScore> = scenario.observations.iter()
            .map(|point| AnomalyScore { tick: point.tick, score: Some(1.0) })
            .collect();
        let metrics = evaluate(&scenario, &tied_scores, 4.0);
        assert!((metrics.average_precision.unwrap() - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn dropout_remains_missing_and_is_measured_as_integrity_event() {
        let scenario = generated(ScenarioKind::Dropout, 123);
        let scores = z_score_baseline(&scenario.observations, 64);
        let metrics = evaluate(&scenario, &scores, 4.0);
        assert!(metrics.missing_samples > 0);
        assert_eq!(metrics.event_recall, Some(1.0));
        assert_eq!(metrics.average_precision, None);
    }

    #[test]
    fn invalid_generator_settings_are_rejected() {
        assert!(
            generate_scenario(ScenarioKind::Null, 1, 10, 1_000, EvaluationSplit::Development)
                .is_err()
        );
        assert!(
            generate_scenario(
                ScenarioKind::Null,
                1,
                MAX_SAMPLE_COUNT + 1,
                1_000,
                EvaluationSplit::Development
            )
            .is_err()
        );
        assert!(
            generate_scenario(ScenarioKind::Null, 1, 64, 0, EvaluationSplit::Development)
                .is_err()
        );
    }

    #[test]
    fn schema_is_versioned_and_seeded() {
        let scenario = generated(ScenarioKind::Null, 0xA5A5);
        assert_eq!(scenario.manifest.schema_version, MANIFEST_SCHEMA_VERSION);
        assert_eq!(scenario.manifest.generator_version, GENERATOR_VERSION);
        assert_eq!(scenario.manifest.seed, 0xA5A5);
    }
}
