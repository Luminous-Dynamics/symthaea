//! Timing-aware scoring of detector alarms against known drift episodes.
//!
//! This scores drift-event detection, not downstream predictive quality. Ground-truth
//! intervals and alarm times must come from the same held-out stream.

use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs, path::Path};

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriftTrace {
    pub schema_version: u32,
    pub stream_id: String,
    pub detector_id: String,
    pub repository_head_sha: String,
    pub configuration_digest: String,
    pub seed: u64,
    pub stream_length_cycles: u64,
    pub valid_observations: u64,
    pub max_detection_delay_cycles: u64,
    pub drift_events: Vec<DriftEvent>,
    pub detections: Vec<Detection>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DriftEvent {
    pub id: String,
    pub start_cycle: u64,
    pub end_cycle: u64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Detection {
    pub cycle_index: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EventMatch {
    pub event_id: String,
    pub start_cycle: u64,
    pub end_cycle: u64,
    pub detected_cycle: Option<u64>,
    pub delay_cycles: Option<u64>,
    /// Delay / (transition width + allowed post-transition delay), clipped to [0, 1].
    /// A missed event receives a penalty of 1.0.
    pub normalized_delay_penalty: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DriftScore {
    pub metric_version: String,
    pub stream_id: String,
    pub detector_id: String,
    pub repository_head_sha: String,
    pub configuration_digest: String,
    pub seed: u64,
    pub stream_length_cycles: u64,
    pub valid_observations: u64,
    pub ground_truth_events: usize,
    pub detections_total: usize,
    pub true_positive_events: usize,
    pub false_positive_alarms: usize,
    pub missed_events: usize,
    pub precision: f64,
    pub recall: f64,
    pub event_f1: f64,
    pub false_alarms_per_1000_valid_observations: f64,
    pub missed_drift_rate: f64,
    pub mean_normalized_detection_delay_penalized: f64,
    pub event_matches: Vec<EventMatch>,
    pub false_alarm_cycles: Vec<u64>,
}

pub fn evaluate_file(path: &Path) -> anyhow::Result<DriftScore> {
    let bytes = fs::read(path).with_context(|| format!("reading drift trace {}", path.display()))?;
    let trace: DriftTrace = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing drift trace {}", path.display()))?;
    score_trace(trace)
}

pub fn score_trace(mut trace: DriftTrace) -> anyhow::Result<DriftScore> {
    ensure!(trace.schema_version == 1, "drift trace schema_version must be 1");
    ensure!(!trace.stream_id.trim().is_empty(), "stream_id must not be empty");
    ensure!(!trace.detector_id.trim().is_empty(), "detector_id must not be empty");
    ensure!(
        trace.repository_head_sha.len() == 40 || trace.repository_head_sha.len() == 64,
        "repository_head_sha must be a full 40- or 64-character commit hash"
    );
    ensure!(
        trace.repository_head_sha.bytes().all(|b| b.is_ascii_hexdigit()),
        "repository_head_sha must be hexadecimal"
    );
    ensure!(
        trace.configuration_digest.len() == 64
            && trace.configuration_digest.bytes().all(|b| b.is_ascii_hexdigit()),
        "configuration_digest must be a 64-character hexadecimal digest"
    );
    ensure!(trace.stream_length_cycles > 0, "stream_length_cycles must be positive");
    ensure!(
        trace.valid_observations > 0,
        "valid_observations must be positive to report false alarms per 1,000 observations"
    );
    ensure!(
        trace.valid_observations <= trace.stream_length_cycles,
        "valid_observations cannot exceed stream_length_cycles"
    );
    ensure!(
        trace.max_detection_delay_cycles > 0,
        "max_detection_delay_cycles must be positive"
    );

    let mut event_ids = BTreeSet::new();
    for event in &trace.drift_events {
        ensure!(!event.id.trim().is_empty(), "drift event id must not be empty");
        ensure!(event_ids.insert(event.id.clone()), "duplicate drift event id: {}", event.id);
        ensure!(
            event.start_cycle <= event.end_cycle,
            "drift event {} starts after it ends",
            event.id
        );
        ensure!(
            event.end_cycle < trace.stream_length_cycles,
            "drift event {} lies outside stream_length_cycles",
            event.id
        );
    }
    for detection in &trace.detections {
        ensure!(
            detection.cycle_index < trace.stream_length_cycles,
            "detection cycle {} lies outside stream_length_cycles",
            detection.cycle_index
        );
    }

    trace.drift_events.sort_by_key(|event| (event.start_cycle, event.end_cycle, event.id.clone()));
    trace.detections.sort_by_key(|detection| detection.cycle_index);

    let mut used_detection = vec![false; trace.detections.len()];
    let mut event_matches = Vec::with_capacity(trace.drift_events.len());
    let mut missed_events = 0usize;
    let mut total_normalized_delay = 0.0f64;

    for event in &trace.drift_events {
        let deadline = event.end_cycle.saturating_add(trace.max_detection_delay_cycles);
        let found = trace.detections.iter().enumerate().find(|(index, detection)| {
            !used_detection[*index]
                && detection.cycle_index >= event.start_cycle
                && detection.cycle_index <= deadline
        });
        if let Some((index, detection)) = found {
            used_detection[index] = true;
            let delay = detection.cycle_index.saturating_sub(event.start_cycle);
            let allowance = event
                .end_cycle
                .saturating_sub(event.start_cycle)
                .saturating_add(trace.max_detection_delay_cycles)
                .max(1);
            let normalized = (delay as f64 / allowance as f64).clamp(0.0, 1.0);
            total_normalized_delay += normalized;
            event_matches.push(EventMatch {
                event_id: event.id.clone(),
                start_cycle: event.start_cycle,
                end_cycle: event.end_cycle,
                detected_cycle: Some(detection.cycle_index),
                delay_cycles: Some(delay),
                normalized_delay_penalty: normalized,
            });
        } else {
            missed_events += 1;
            total_normalized_delay += 1.0;
            event_matches.push(EventMatch {
                event_id: event.id.clone(),
                start_cycle: event.start_cycle,
                end_cycle: event.end_cycle,
                detected_cycle: None,
                delay_cycles: None,
                normalized_delay_penalty: 1.0,
            });
        }
    }

    let true_positive_events = event_matches.iter().filter(|m| m.detected_cycle.is_some()).count();
    let false_alarm_cycles = trace
        .detections
        .iter()
        .enumerate()
        .filter_map(|(index, detection)| (!used_detection[index]).then_some(detection.cycle_index))
        .collect::<Vec<_>>();
    let false_positive_alarms = false_alarm_cycles.len();

    let precision = if trace.detections.is_empty() {
        if trace.drift_events.is_empty() { 1.0 } else { 0.0 }
    } else {
        true_positive_events as f64 / trace.detections.len() as f64
    };
    let recall = if trace.drift_events.is_empty() {
        1.0
    } else {
        true_positive_events as f64 / trace.drift_events.len() as f64
    };
    let event_f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    let false_alarms_per_1000_valid_observations =
        false_positive_alarms as f64 * 1000.0 / trace.valid_observations as f64;
    let missed_drift_rate = if trace.drift_events.is_empty() {
        0.0
    } else {
        missed_events as f64 / trace.drift_events.len() as f64
    };
    let mean_normalized_detection_delay_penalized = if trace.drift_events.is_empty() {
        0.0
    } else {
        total_normalized_delay / trace.drift_events.len() as f64
    };

    Ok(DriftScore {
        metric_version: "cognitive-drift-event/v1".to_owned(),
        stream_id: trace.stream_id,
        detector_id: trace.detector_id,
        repository_head_sha: trace.repository_head_sha,
        configuration_digest: trace.configuration_digest,
        seed: trace.seed,
        stream_length_cycles: trace.stream_length_cycles,
        valid_observations: trace.valid_observations,
        ground_truth_events: trace.drift_events.len(),
        detections_total: trace.detections.len(),
        true_positive_events,
        false_positive_alarms,
        missed_events,
        precision,
        recall,
        event_f1,
        false_alarms_per_1000_valid_observations,
        missed_drift_rate,
        mean_normalized_detection_delay_penalized,
        event_matches,
        false_alarm_cycles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(events: Vec<DriftEvent>, detections: Vec<Detection>) -> DriftTrace {
        DriftTrace {
            schema_version: 1,
            stream_id: "synthetic-test-stream".to_owned(),
            detector_id: "test-detector".to_owned(),
            repository_head_sha: "a".repeat(40),
            configuration_digest: "b".repeat(64),
            seed: 17,
            stream_length_cycles: 1000,
            valid_observations: 900,
            max_detection_delay_cycles: 20,
            drift_events: events,
            detections,
        }
    }

    #[test]
    fn matches_one_alarm_to_one_drift_episode() {
        let result = score_trace(trace(
            vec![DriftEvent { id: "d1".to_owned(), start_cycle: 100, end_cycle: 120 }],
            vec![Detection { cycle_index: 110 }],
        )).expect("trace scores");
        assert_eq!(result.true_positive_events, 1);
        assert_eq!(result.false_positive_alarms, 0);
        assert_eq!(result.missed_events, 0);
        assert_eq!(result.event_f1, 1.0);
        assert!((result.mean_normalized_detection_delay_penalized - 0.25).abs() < 1e-12);
    }

    #[test]
    fn duplicate_alarm_and_pre_drift_alarm_count_as_false_positives() {
        let result = score_trace(trace(
            vec![DriftEvent { id: "d1".to_owned(), start_cycle: 100, end_cycle: 100 }],
            vec![
                Detection { cycle_index: 20 },
                Detection { cycle_index: 105 },
                Detection { cycle_index: 106 },
            ],
        )).expect("trace scores");
        assert_eq!(result.true_positive_events, 1);
        assert_eq!(result.false_positive_alarms, 2);
        assert_eq!(result.false_alarm_cycles, vec![20, 106]);
        assert!((result.false_alarms_per_1000_valid_observations - (2000.0 / 900.0)).abs() < 1e-12);
        assert!((result.event_f1 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn missed_shift_receives_full_delay_penalty() {
        let result = score_trace(trace(
            vec![DriftEvent { id: "d1".to_owned(), start_cycle: 100, end_cycle: 130 }],
            vec![Detection { cycle_index: 20 }],
        )).expect("trace scores");
        assert_eq!(result.missed_events, 1);
        assert_eq!(result.missed_drift_rate, 1.0);
        assert_eq!(result.mean_normalized_detection_delay_penalized, 1.0);
    }

    #[test]
    fn rejects_invalid_provenance_and_out_of_range_events() {
        let mut bad = trace(
            vec![DriftEvent { id: "d1".to_owned(), start_cycle: 900, end_cycle: 1100 }],
            vec![],
        );
        bad.configuration_digest = "short".to_owned();
        assert!(score_trace(bad).is_err());
    }

    #[test]
    fn no_drift_no_alarms_is_a_perfect_no_alarm_result() {
        let result = score_trace(trace(vec![], vec![])).expect("trace scores");
        assert_eq!(result.event_f1, 1.0);
        assert_eq!(result.false_positive_alarms, 0);
        assert_eq!(result.mean_normalized_detection_delay_penalized, 0.0);
    }
}
