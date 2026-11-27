//! Timing-aware scoring of detector alarms against known drift episodes.
//!
//! This scores drift-event detection, not downstream predictive quality. Ground-truth
//! intervals and alarm times must come from the same held-out stream.

use anyhow::{ensure, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::Path};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DriftTrace {
    pub schema_version: u32,
    pub stream_id: String,
    pub detector_id: String,
    pub detector_version: String,
    pub comparison_track: String,
    /// Delay between an outcome becoming available and the event being observed, in cycles.
    pub label_delay_cycles: u64,
    pub repository_head_sha: String,
    pub configuration_digest: String,
    pub seed: u64,
    pub stream_length_cycles: u64,
    pub valid_observations: u64,
    pub max_detection_delay_cycles: u64,
    pub drift_events: Vec<DriftEvent>,
    pub detections: Vec<Detection>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DriftEvent {
    pub id: String,
    pub start_cycle: u64,
    pub end_cycle: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Detection {
    pub cycle_index: u64,
    /// Whether the detector had valid evidence from its declared access track at this cycle.
    pub evidence_valid: bool,
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
    pub canonical_trace_sha256: String,
    /// SHA-256 of the serialized score with this field set to null.
    pub result_sha256: Option<String>,
    pub stream_id: String,
    pub detector_id: String,
    pub detector_version: String,
    pub comparison_track: String,
    pub label_delay_cycles: u64,
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
    pub invalid_evidence_alarms: usize,
    pub hard_gate_pass: bool,
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
    ensure!(!trace.detector_version.trim().is_empty(), "detector_version must not be empty");
    ensure!(
        ["valid-pe-only", "feature-only-unsupervised", "auxiliary-loss-unsupervised", "observed-outcome"].contains(&trace.comparison_track.as_str()),
        "comparison_track must be one of the declared evidence-access tracks"
    );
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

    // Canonical digest is invariant to array ordering in the JSON input.
    let mut canonical_trace = trace.clone();
    canonical_trace.drift_events.sort_by_key(|event| (
        event.start_cycle,
        event.end_cycle,
        event.id.clone(),
    ));
    canonical_trace.detections.sort_by_key(|detection| (
        detection.cycle_index,
        detection.evidence_valid,
    ));
    let canonical_trace_bytes = serde_json::to_vec(&canonical_trace)?;
    let canonical_trace_sha256 = sha256_hex(&canonical_trace_bytes);

    // Earliest-deadline interval matching maximizes event/alarm matches for interval-compatible
    // events; alarms still prefer the earliest valid timestamp within each window.
    let delay_budget = trace.max_detection_delay_cycles;
    trace.drift_events.sort_by_key(|event| (
        event.end_cycle.saturating_add(delay_budget),
        event.start_cycle,
        event.id.clone(),
    ));
    trace.detections.sort_by_key(|detection| detection.cycle_index);

    let mut used_detection = vec![false; trace.detections.len()];
    let mut event_matches = Vec::with_capacity(trace.drift_events.len());
    let mut missed_events = 0usize;
    let mut total_normalized_delay = 0.0f64;

    for event in &trace.drift_events {
        let deadline = event.end_cycle.saturating_add(trace.max_detection_delay_cycles);
        let found = trace.detections.iter().enumerate().find(|(index, detection)| {
            !used_detection[*index]
                && detection.evidence_valid
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

    event_matches.sort_by_key(|m| (m.start_cycle, m.end_cycle, m.event_id.clone()));
    let true_positive_events = event_matches.iter().filter(|m| m.detected_cycle.is_some()).count();
    let invalid_evidence_alarms = trace.detections.iter().filter(|d| !d.evidence_valid).count();
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

    let mut score = DriftScore {
        metric_version: "cognitive-drift-event/v1".to_owned(),
        canonical_trace_sha256,
        result_sha256: None,
        stream_id: trace.stream_id,
        detector_id: trace.detector_id,
        detector_version: trace.detector_version,
        comparison_track: trace.comparison_track,
        label_delay_cycles: trace.label_delay_cycles,
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
        invalid_evidence_alarms,
        hard_gate_pass: invalid_evidence_alarms == 0,
        precision,
        recall,
        event_f1,
        false_alarms_per_1000_valid_observations,
        missed_drift_rate,
        mean_normalized_detection_delay_penalized,
        event_matches,
        false_alarm_cycles,
    };
    let score_payload = serde_json::to_vec(&score)?;
    score.result_sha256 = Some(sha256_hex(&score_payload));
    Ok(score)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest.iter() {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(events: Vec<DriftEvent>, detections: Vec<Detection>) -> DriftTrace {
        DriftTrace {
            schema_version: 1,
            stream_id: "synthetic-test-stream".to_owned(),
            detector_id: "test-detector".to_owned(),
            detector_version: "test-fixture-1".to_owned(),
            comparison_track: "valid-pe-only".to_owned(),
            label_delay_cycles: 0,
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
            vec![Detection { cycle_index: 110, evidence_valid: true }],
        )).expect("trace scores");
        assert_eq!(result.true_positive_events, 1);
        assert_eq!(result.false_positive_alarms, 0);
        assert_eq!(result.missed_events, 0);
        assert_eq!(result.event_f1, 1.0);
        assert!(result.hard_gate_pass);
        assert_eq!(result.canonical_trace_sha256.len(), 64);
        assert_eq!(result.result_sha256.as_ref().map(String::len), Some(64));
        assert!((result.mean_normalized_detection_delay_penalized - 0.25).abs() < 1e-12);
    }

    #[test]
    fn duplicate_alarm_and_pre_drift_alarm_count_as_false_positives() {
        let result = score_trace(trace(
            vec![DriftEvent { id: "d1".to_owned(), start_cycle: 100, end_cycle: 100 }],
            vec![
                Detection { cycle_index: 20, evidence_valid: true },
                Detection { cycle_index: 105, evidence_valid: true },
                Detection { cycle_index: 106, evidence_valid: true },
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
            vec![Detection { cycle_index: 20, evidence_valid: true }],
        )).expect("trace scores");
        assert_eq!(result.missed_events, 1);
        assert_eq!(result.missed_drift_rate, 1.0);
        assert_eq!(result.mean_normalized_detection_delay_penalized, 1.0);
    }

    #[test]
    fn rejects_invalid_provenance_and_out_of_range_events() {
        let mut bad_digest = trace(vec![], vec![]);
        bad_digest.configuration_digest = "short".to_owned();
        assert!(score_trace(bad_digest).is_err());

        let bad_event = trace(
            vec![DriftEvent { id: "d1".to_owned(), start_cycle: 900, end_cycle: 1100 }],
            vec![],
        );
        assert!(score_trace(bad_event).is_err());
    }

    #[test]
    fn rejects_undeclared_evidence_access_track() {
        let mut bad = trace(vec![], vec![]);
        bad.comparison_track = "secret-label-leak".to_owned();
        assert!(score_trace(bad).is_err());
    }

    #[test]
    fn earliest_deadline_matching_prevents_avoidable_missed_events() {
        let result = score_trace(trace(
            vec![
                DriftEvent { id: "long-window".to_owned(), start_cycle: 100, end_cycle: 200 },
                DriftEvent { id: "short-window".to_owned(), start_cycle: 150, end_cycle: 150 },
            ],
            vec![
                Detection { cycle_index: 160, evidence_valid: true },
                Detection { cycle_index: 210, evidence_valid: true },
            ],
        )).expect("trace scores");
        assert_eq!(result.true_positive_events, 2);
        assert_eq!(result.missed_events, 0);
        assert_eq!(result.false_positive_alarms, 0);
    }

    #[test]
    fn trace_digest_is_invariant_to_event_and_alarm_order() {
        let first = trace(
            vec![
                DriftEvent { id: "d2".to_owned(), start_cycle: 300, end_cycle: 305 },
                DriftEvent { id: "d1".to_owned(), start_cycle: 100, end_cycle: 105 },
            ],
            vec![
                Detection { cycle_index: 301, evidence_valid: true },
                Detection { cycle_index: 101, evidence_valid: true },
            ],
        );
        let mut second = first.clone();
        second.drift_events.reverse();
        second.detections.reverse();
        let a = score_trace(first).expect("first trace scores");
        let b = score_trace(second).expect("reordered trace scores");
        assert_eq!(a.canonical_trace_sha256, b.canonical_trace_sha256);
        assert_eq!(a.result_sha256, b.result_sha256);
    }

    #[test]
    fn alarm_without_valid_evidence_cannot_match_and_fails_hard_gate() {
        let result = score_trace(trace(
            vec![DriftEvent { id: "d1".to_owned(), start_cycle: 100, end_cycle: 100 }],
            vec![Detection { cycle_index: 105, evidence_valid: false }],
        )).expect("trace scores");
        assert_eq!(result.true_positive_events, 0);
        assert_eq!(result.false_positive_alarms, 1);
        assert_eq!(result.missed_events, 1);
        assert_eq!(result.invalid_evidence_alarms, 1);
        assert!(!result.hard_gate_pass);
    }

    #[test]
    fn no_drift_no_alarms_is_a_perfect_no_alarm_result() {
        let result = score_trace(trace(vec![], vec![])).expect("trace scores");
        assert_eq!(result.event_f1, 1.0);
        assert_eq!(result.false_positive_alarms, 0);
        assert_eq!(result.mean_normalized_detection_delay_penalized, 0.0);
    }
}
