use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: &str = "mel-musical-timing-panel-v2";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MusicalTimingPanelV2 {
    schema_version: String,
    session_id: String,
    case_id: String,
    track: TrackKind,
    clock_domain_id: String,
    instrument_scope: String,
    reference_time_map_id: String,
    reference_time_map_version: String,
    reference_time_map_kind: String,
    reference_time_map_receipt_sha256: String,
    onset_capture_receipt_sha256: String,
    clock_calibration_receipt_sha256: String,
    policy: TimingPolicyV2,
    target_events: Vec<TargetEventV2>,
    tempo_changes: Vec<TempoChangeV2>,
    tempo_estimates: Vec<TempoEstimateV2>,
    extra_onsets: Vec<ExtraOnsetV2>,
    claim_ceiling: Vec<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TrackKind {
    Midi,
    RenderedAudio,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TimingPolicyV2 {
    maximum_relative_period_error: f64,
    required_consecutive_stable_estimates: usize,
    minimum_estimate_spacing_ns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetEventV2 {
    event_id: String,
    instrument_id: String,
    beat_index: u64,
    subdivision_index: u32,
    subdivision_count: u32,
    expected_onset_ns: u64,
    local_beat_period_ns: u64,
    outcome: String,
    observed_onset_ns: Option<u64>,
    capture_receipt_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TempoChangeV2 {
    change_id: String,
    change_observed_ns: u64,
    reference_beat_period_ns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TempoEstimateV2 {
    estimate_id: String,
    change_id: String,
    observed_at_ns: u64,
    evidence_window_end_ns: u64,
    estimated_period_ns: u64,
    confidence: f64,
    provenance_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtraOnsetV2 {
    event_id: String,
    instrument_id: String,
    observed_onset_ns: u64,
    capture_receipt_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Distribution {
    pub count: usize,
    pub median: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
    pub maximum: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstrumentTimingSummary {
    pub instrument_id: String,
    pub target_count: usize,
    pub observed_count: usize,
    pub missed_count: usize,
    pub absolute_phase_error_beats: Distribution,
    pub within_005_beat_rate: Option<f64>,
    pub within_010_beat_rate: Option<f64>,
    pub within_025_beat_rate: Option<f64>,
    pub within_050_beat_rate: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TempoAdaptationSummary {
    pub change_id: String,
    pub reference_beat_period_ns: u64,
    pub estimate_count: usize,
    pub stable_estimates_required: usize,
    pub recovered: bool,
    pub adaptation_time_ns: Option<u64>,
    pub adaptation_time_beats: Option<f64>,
    pub stable_run_start_ns: Option<u64>,
    pub last_estimated_period_ns: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EventTimingFinding {
    pub event_id: String,
    pub instrument_id: String,
    pub beat_index: u64,
    pub expected_onset_ns: u64,
    pub observed_onset_ns: Option<u64>,
    pub signed_error_ns: Option<i128>,
    pub signed_phase_error_beats: Option<f64>,
    pub absolute_phase_error_beats: Option<f64>,
    pub within_005_beat: Option<bool>,
    pub within_010_beat: Option<bool>,
    pub within_025_beat: Option<bool>,
    pub within_050_beat: Option<bool>,
    pub outcome: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MusicalTimingReport {
    pub schema_version: String,
    pub session_id: String,
    pub case_id: String,
    pub track: TrackKind,
    pub reference_time_map_id: String,
    pub reference_time_map_version: String,
    pub reference_time_map_kind: String,
    pub target_count: usize,
    pub observed_count: usize,
    pub missed_count: usize,
    pub extra_onset_count: usize,
    pub target_accounting_completeness: f64,
    pub observed_target_coverage_rate: f64,
    pub signed_phase_error_beats: Distribution,
    pub absolute_phase_error_beats: Distribution,
    pub absolute_error_ms: Distribution,
    pub signed_phase_bias_beats: Option<f64>,
    pub phase_drift_beats_per_beat: Option<f64>,
    pub within_005_beat_rate: Option<f64>,
    pub within_010_beat_rate: Option<f64>,
    pub within_025_beat_rate: Option<f64>,
    pub within_050_beat_rate: Option<f64>,
    pub per_instrument: Vec<InstrumentTimingSummary>,
    pub tempo_adaptation: Vec<TempoAdaptationSummary>,
    pub future_information_violations: usize,
    pub qualified_for_declared_timing_contract: bool,
    pub qualification_findings: Vec<String>,
    pub event_findings: Vec<EventTimingFinding>,
    pub claim_ceiling: Vec<String>,
}

#[derive(Debug)]
struct ComputedEvent {
    instrument_id: String,
    beat_index: u64,
    signed_phase_error: f64,
    absolute_phase_error: f64,
    absolute_error_ms: f64,
}

pub fn parse_and_score(input: &str) -> Result<MusicalTimingReport, String> {
    let panel: MusicalTimingPanelV2 = serde_json::from_str(input)
        .map_err(|error| format!("invalid musical timing panel JSON: {error}"))?;
    score_panel(&panel)
}

fn score_panel(panel: &MusicalTimingPanelV2) -> Result<MusicalTimingReport, String> {
    validate_header(panel)?;
    let changes = validate_tempo_changes(&panel.tempo_changes)?;
    let estimates = validate_estimates(panel, &changes)?;
    let mut findings = Vec::new();
    let mut event_findings = Vec::new();
    let mut computed = Vec::new();
    let mut event_ids = BTreeSet::new();
    let mut observed_count = 0usize;
    let mut missed_count = 0usize;

    for target in &panel.target_events {
        require_text(&target.event_id, "target event_id")?;
        require_text(&target.instrument_id, "target instrument_id")?;
        if !event_ids.insert(target.event_id.as_str()) {
            return Err(format!("duplicate target event_id: {}", target.event_id));
        }
        if target.subdivision_count == 0 || target.subdivision_index >= target.subdivision_count {
            return Err(format!(
                "event {} has an invalid rational subdivision",
                target.event_id
            ));
        }
        if target.local_beat_period_ns == 0 {
            return Err(format!(
                "event {} has zero canonical local beat period",
                target.event_id
            ));
        }
        match target.outcome.as_str() {
            "observed" => {
                let observed = target.observed_onset_ns.ok_or_else(|| {
                    format!(
                        "observed event {} is missing observed_onset_ns",
                        target.event_id
                    )
                })?;
                let receipt = target.capture_receipt_sha256.as_deref().ok_or_else(|| {
                    format!(
                        "observed event {} is missing capture receipt",
                        target.event_id
                    )
                })?;
                require_digest(receipt, "target onset capture receipt")?;
                let signed_error_ns = i128::from(observed) - i128::from(target.expected_onset_ns);
                let signed_phase = signed_error_ns as f64 / target.local_beat_period_ns as f64;
                let absolute_phase = signed_phase.abs();
                let absolute_ms = signed_error_ns.abs() as f64 / 1_000_000.0;
                let within_005 = absolute_phase <= 0.05;
                let within_010 = absolute_phase <= 0.10;
                let within_025 = absolute_phase <= 0.25;
                let within_050 = absolute_phase <= 0.50;
                event_findings.push(EventTimingFinding {
                    event_id: target.event_id.clone(),
                    instrument_id: target.instrument_id.clone(),
                    beat_index: target.beat_index,
                    expected_onset_ns: target.expected_onset_ns,
                    observed_onset_ns: Some(observed),
                    signed_error_ns: Some(signed_error_ns),
                    signed_phase_error_beats: Some(signed_phase),
                    absolute_phase_error_beats: Some(absolute_phase),
                    within_005_beat: Some(within_005),
                    within_010_beat: Some(within_010),
                    within_025_beat: Some(within_025),
                    within_050_beat: Some(within_050),
                    outcome: "observed".to_owned(),
                });
                computed.push(ComputedEvent {
                    instrument_id: target.instrument_id.clone(),
                    beat_index: target.beat_index,
                    signed_phase_error: signed_phase,
                    absolute_phase_error: absolute_phase,
                    absolute_error_ms: absolute_ms,
                });
                observed_count += 1;
            }
            "missed" => {
                if target.observed_onset_ns.is_some() || target.capture_receipt_sha256.is_some() {
                    return Err(format!(
                        "missed event {} must not contain an onset or capture receipt",
                        target.event_id
                    ));
                }
                event_findings.push(EventTimingFinding {
                    event_id: target.event_id.clone(),
                    instrument_id: target.instrument_id.clone(),
                    beat_index: target.beat_index,
                    expected_onset_ns: target.expected_onset_ns,
                    observed_onset_ns: None,
                    signed_error_ns: None,
                    signed_phase_error_beats: None,
                    absolute_phase_error_beats: None,
                    within_005_beat: None,
                    within_010_beat: None,
                    within_025_beat: None,
                    within_050_beat: None,
                    outcome: "missed".to_owned(),
                });
                missed_count += 1;
                findings.push(format!("target event {} was missed", target.event_id));
            }
            other => return Err(format!("unsupported target outcome {other}")),
        }
    }

    let mut extra_ids = BTreeSet::new();
    for extra in &panel.extra_onsets {
        require_text(&extra.event_id, "extra onset event_id")?;
        require_text(&extra.instrument_id, "extra onset instrument_id")?;
        require_digest(&extra.capture_receipt_sha256, "extra onset capture receipt")?;
        if !extra_ids.insert(extra.event_id.as_str()) || event_ids.contains(extra.event_id.as_str())
        {
            return Err(format!(
                "duplicate or target-colliding extra onset ID: {}",
                extra.event_id
            ));
        }
    }
    if !panel.extra_onsets.is_empty() {
        findings.push(format!(
            "{} unmatched extra onsets were supplied",
            panel.extra_onsets.len()
        ));
    }

    let signed: Vec<f64> = computed
        .iter()
        .map(|item| item.signed_phase_error)
        .collect();
    let absolute: Vec<f64> = computed
        .iter()
        .map(|item| item.absolute_phase_error)
        .collect();
    let absolute_ms: Vec<f64> = computed.iter().map(|item| item.absolute_error_ms).collect();
    let signed_bias = mean(&signed);
    let drift = regression_slope(
        computed
            .iter()
            .map(|item| (item.beat_index as f64, item.signed_phase_error))
            .collect(),
    );
    let within_005 = threshold_rate(&absolute, 0.05);
    let within_010 = threshold_rate(&absolute, 0.10);
    let within_025 = threshold_rate(&absolute, 0.25);
    let within_050 = threshold_rate(&absolute, 0.50);

    let mut by_instrument = BTreeMap::<String, Vec<&ComputedEvent>>::new();
    let mut target_counts = BTreeMap::<String, usize>::new();
    let mut missed_counts = BTreeMap::<String, usize>::new();
    for target in &panel.target_events {
        *target_counts
            .entry(target.instrument_id.clone())
            .or_default() += 1;
        if target.outcome == "missed" {
            *missed_counts
                .entry(target.instrument_id.clone())
                .or_default() += 1;
        }
    }
    for item in &computed {
        by_instrument
            .entry(item.instrument_id.clone())
            .or_default()
            .push(item);
    }
    let mut per_instrument = Vec::new();
    for (instrument, target_count) in target_counts {
        let items = by_instrument.get(&instrument).cloned().unwrap_or_default();
        let errors: Vec<f64> = items.iter().map(|item| item.absolute_phase_error).collect();
        per_instrument.push(InstrumentTimingSummary {
            instrument_id: instrument.clone(),
            target_count,
            observed_count: items.len(),
            missed_count: missed_counts.get(&instrument).copied().unwrap_or(0),
            absolute_phase_error_beats: distribution(&errors),
            within_005_beat_rate: threshold_rate(&errors, 0.05),
            within_010_beat_rate: threshold_rate(&errors, 0.10),
            within_025_beat_rate: threshold_rate(&errors, 0.25),
            within_050_beat_rate: threshold_rate(&errors, 0.50),
        });
    }

    let tempo_adaptation = score_tempo_adaptation(panel, &estimates);
    for summary in &tempo_adaptation {
        if !summary.recovered {
            findings.push(format!(
                "tempo change {} did not reach {} consecutive stable estimates",
                summary.change_id, summary.stable_estimates_required
            ));
        }
    }

    let total_targets = panel.target_events.len();
    let target_accounting_completeness = if total_targets == 0 {
        0.0
    } else {
        (observed_count + missed_count) as f64 / total_targets as f64
    };
    if observed_count + missed_count != total_targets {
        findings.push("target event accounting is incomplete".to_owned());
    }

    Ok(MusicalTimingReport {
        schema_version: "mel-musical-timing-report-v2".to_owned(),
        session_id: panel.session_id.clone(),
        case_id: panel.case_id.clone(),
        track: panel.track,
        reference_time_map_id: panel.reference_time_map_id.clone(),
        reference_time_map_version: panel.reference_time_map_version.clone(),
        reference_time_map_kind: panel.reference_time_map_kind.clone(),
        target_count: total_targets,
        observed_count,
        missed_count,
        extra_onset_count: panel.extra_onsets.len(),
        target_accounting_completeness,
        observed_target_coverage_rate: if total_targets == 0 {
            0.0
        } else {
            observed_count as f64 / total_targets as f64
        },
        signed_phase_error_beats: distribution(&signed),
        absolute_phase_error_beats: distribution(&absolute),
        absolute_error_ms: distribution(&absolute_ms),
        signed_phase_bias_beats: signed_bias,
        phase_drift_beats_per_beat: drift,
        within_005_beat_rate: within_005,
        within_010_beat_rate: within_010,
        within_025_beat_rate: within_025,
        within_050_beat_rate: within_050,
        per_instrument,
        tempo_adaptation,
        future_information_violations: 0,
        qualified_for_declared_timing_contract: findings.is_empty(),
        qualification_findings: findings,
        event_findings,
        claim_ceiling: panel.claim_ceiling.clone(),
    })
}

fn validate_header(panel: &MusicalTimingPanelV2) -> Result<(), String> {
    if panel.schema_version != SCHEMA_VERSION {
        return Err("unsupported musical timing schema_version".to_owned());
    }
    require_text(&panel.session_id, "session_id")?;
    require_text(&panel.case_id, "case_id")?;
    require_text(&panel.clock_domain_id, "clock_domain_id")?;
    require_text(&panel.instrument_scope, "instrument_scope")?;
    require_text(&panel.reference_time_map_id, "reference_time_map_id")?;
    require_text(
        &panel.reference_time_map_version,
        "reference_time_map_version",
    )?;
    if !matches!(
        panel.reference_time_map_kind.as_str(),
        "notation_tempo" | "performance_time"
    ) {
        return Err("unsupported canonical reference time-map kind".to_owned());
    }
    require_digest(
        &panel.reference_time_map_receipt_sha256,
        "canonical reference time-map receipt",
    )?;
    require_digest(&panel.onset_capture_receipt_sha256, "onset capture receipt")?;
    require_digest(
        &panel.clock_calibration_receipt_sha256,
        "clock calibration receipt",
    )?;
    if panel.claim_ceiling.is_empty() || panel.claim_ceiling.iter().any(|s| s.trim().is_empty()) {
        return Err("claim_ceiling must contain explicit non-empty limits".to_owned());
    }
    let policy = &panel.policy;
    if !policy.maximum_relative_period_error.is_finite()
        || policy.maximum_relative_period_error <= 0.0
        || policy.maximum_relative_period_error > 1.0
        || policy.required_consecutive_stable_estimates == 0
        || policy.minimum_estimate_spacing_ns == 0
    {
        return Err("invalid tempo adaptation policy".to_owned());
    }
    if panel.target_events.is_empty() {
        return Err("target_events must be non-empty".to_owned());
    }
    Ok(())
}

fn validate_tempo_changes(
    changes: &[TempoChangeV2],
) -> Result<BTreeMap<&str, &TempoChangeV2>, String> {
    let mut result = BTreeMap::new();
    let mut previous_time = None;
    for change in changes {
        require_text(&change.change_id, "tempo change ID")?;
        if change.reference_beat_period_ns == 0 {
            return Err(format!(
                "tempo change {} has zero reference beat period",
                change.change_id
            ));
        }
        if previous_time.is_some_and(|time| change.change_observed_ns <= time) {
            return Err("tempo changes must have strictly increasing observation times".to_owned());
        }
        previous_time = Some(change.change_observed_ns);
        if result.insert(change.change_id.as_str(), change).is_some() {
            return Err(format!("duplicate tempo change ID: {}", change.change_id));
        }
    }
    Ok(result)
}

fn validate_estimates<'a>(
    panel: &'a MusicalTimingPanelV2,
    changes: &BTreeMap<&str, &'a TempoChangeV2>,
) -> Result<BTreeMap<&'a str, Vec<&'a TempoEstimateV2>>, String> {
    let mut result = BTreeMap::<&str, Vec<&TempoEstimateV2>>::new();
    let mut ids = BTreeSet::new();
    let mut previous_time = None;
    let mut previous_time_by_change = BTreeMap::<&str, u64>::new();

    for estimate in &panel.tempo_estimates {
        require_text(&estimate.estimate_id, "tempo estimate ID")?;
        require_digest(&estimate.provenance_sha256, "tempo estimate provenance")?;
        if !ids.insert(estimate.estimate_id.as_str()) {
            return Err(format!(
                "duplicate tempo estimate ID: {}",
                estimate.estimate_id
            ));
        }
        if estimate.estimated_period_ns == 0
            || !estimate.confidence.is_finite()
            || !(0.0..=1.0).contains(&estimate.confidence)
        {
            return Err(format!(
                "tempo estimate {} has invalid values",
                estimate.estimate_id
            ));
        }
        if estimate.evidence_window_end_ns > estimate.observed_at_ns {
            return Err(format!(
                "tempo estimate {} references future evidence",
                estimate.estimate_id
            ));
        }
        if previous_time.is_some_and(|time| estimate.observed_at_ns <= time) {
            return Err("tempo estimate observations must be strictly monotonic".to_owned());
        }
        previous_time = Some(estimate.observed_at_ns);

        let change = changes.get(estimate.change_id.as_str()).ok_or_else(|| {
            format!(
                "estimate references unknown tempo change {}",
                estimate.change_id
            )
        })?;
        if estimate.observed_at_ns < change.change_observed_ns {
            return Err(format!(
                "tempo estimate {} was observed before the tempo change became available",
                estimate.estimate_id
            ));
        }
        if previous_time_by_change
            .get(estimate.change_id.as_str())
            .is_some_and(|time| {
                estimate.observed_at_ns - *time < panel.policy.minimum_estimate_spacing_ns
            })
        {
            return Err(format!(
                "tempo estimate {} violates minimum spacing",
                estimate.estimate_id
            ));
        }
        previous_time_by_change.insert(estimate.change_id.as_str(), estimate.observed_at_ns);
        result
            .entry(estimate.change_id.as_str())
            .or_default()
            .push(estimate);
    }
    Ok(result)
}

fn score_tempo_adaptation<'a>(
    panel: &MusicalTimingPanelV2,
    estimates: &BTreeMap<&str, Vec<&'a TempoEstimateV2>>,
) -> Vec<TempoAdaptationSummary> {
    let required = panel.policy.required_consecutive_stable_estimates;
    let tolerance = panel.policy.maximum_relative_period_error;
    let mut output = Vec::new();

    for change in &panel.tempo_changes {
        let values = estimates
            .get(change.change_id.as_str())
            .cloned()
            .unwrap_or_default();
        let mut stable_run = 0usize;
        let mut stable_start = None;
        let mut recovered_at = None;

        for estimate in &values {
            let relative_error = (estimate.estimated_period_ns as f64
                - change.reference_beat_period_ns as f64)
                .abs()
                / change.reference_beat_period_ns as f64;
            if relative_error <= tolerance {
                stable_run += 1;
                if stable_run == 1 {
                    stable_start = Some(estimate.observed_at_ns);
                }
                if stable_run >= required {
                    recovered_at = Some(estimate.observed_at_ns);
                    break;
                }
            } else {
                stable_run = 0;
                stable_start = None;
            }
        }

        let adaptation_time_ns = recovered_at.map(|at| at - change.change_observed_ns);
        output.push(TempoAdaptationSummary {
            change_id: change.change_id.clone(),
            reference_beat_period_ns: change.reference_beat_period_ns,
            estimate_count: values.len(),
            stable_estimates_required: required,
            recovered: recovered_at.is_some(),
            adaptation_time_ns,
            adaptation_time_beats: adaptation_time_ns
                .map(|time| time as f64 / change.reference_beat_period_ns as f64),
            stable_run_start_ns: stable_start,
            last_estimated_period_ns: values.last().map(|estimate| estimate.estimated_period_ns),
        });
    }
    output
}

fn distribution(values: &[f64]) -> Distribution {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Distribution {
        count: sorted.len(),
        median: percentile(&sorted, 50),
        p95: percentile(&sorted, 95),
        p99: percentile(&sorted, 99),
        maximum: sorted.last().copied(),
    }
}

fn percentile(sorted: &[f64], p: usize) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (sorted.len() * p).div_ceil(100).max(1);
    sorted.get(rank - 1).copied()
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

fn threshold_rate(values: &[f64], threshold: f64) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(
            values.iter().filter(|value| **value <= threshold).count() as f64 / values.len() as f64,
        )
    }
}

fn regression_slope(points: Vec<(f64, f64)>) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let x_mean = points.iter().map(|point| point.0).sum::<f64>() / points.len() as f64;
    let y_mean = points.iter().map(|point| point.1).sum::<f64>() / points.len() as f64;
    let numerator = points
        .iter()
        .map(|point| (point.0 - x_mean) * (point.1 - y_mean))
        .sum::<f64>();
    let denominator = points
        .iter()
        .map(|point| (point.0 - x_mean).powi(2))
        .sum::<f64>();
    if denominator == 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn require_text(value: &str, name: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{name} must be non-empty"))
    } else {
        Ok(())
    }
}

fn require_digest(value: &str, name: &str) -> Result<(), String> {
    if value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        Ok(())
    } else {
        Err(format!(
            "{name} must be a lowercase 64-character SHA-256 digest"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = include_str!("../fixtures/valid-musical-timing-v2.json");

    fn panel() -> Value {
        serde_json::from_str(FIXTURE).expect("known-answer timing fixture JSON")
    }

    fn score(value: &Value) -> Result<MusicalTimingReport, String> {
        parse_and_score(&value.to_string())
    }

    #[test]
    fn known_answer_reports_phase_error_and_tempo_recovery_from_canonical_map() {
        let report = parse_and_score(FIXTURE).expect("known-answer panel should score");
        assert_eq!(report.target_count, 4);
        assert_eq!(report.observed_count, 4);
        assert_eq!(report.missed_count, 0);
        assert_eq!(report.extra_onset_count, 0);
        assert_eq!(report.within_025_beat_rate, Some(1.0));
        assert!(report.tempo_adaptation.iter().all(|item| item.recovered));
        assert_eq!(report.reference_time_map_id, "shared-muse-time-map");
        assert!(report.qualified_for_declared_timing_contract);
    }

    #[test]
    fn estimate_before_tempo_change_is_rejected() {
        let mut value = panel();
        value["tempo_estimates"][5]["observed_at_ns"] = serde_json::json!(1_999_999_999_u64);
        value["tempo_estimates"][5]["evidence_window_end_ns"] =
            serde_json::json!(1_999_999_999_u64);
        let error = score(&value).expect_err("no estimate can precede change availability");
        assert!(error.contains("observed before the tempo change became available"));
    }

    #[test]
    fn tempo_estimate_cannot_use_future_audio_evidence() {
        let mut value = panel();
        value["tempo_estimates"][5]["evidence_window_end_ns"] =
            serde_json::json!(2_200_000_000_u64);
        let error = score(&value).expect_err("tempo estimate may not inspect future audio");
        assert!(error.contains("future evidence"));
    }

    #[test]
    fn tempo_estimates_cannot_be_repeated_at_the_same_timestamp() {
        let mut value = panel();
        value["tempo_estimates"][6]["observed_at_ns"] =
            value["tempo_estimates"][5]["observed_at_ns"].clone();
        value["tempo_estimates"][6]["evidence_window_end_ns"] =
            value["tempo_estimates"][6]["observed_at_ns"].clone();
        let error = score(&value).expect_err("same-timestamp estimates must not fake stability");
        assert!(error.contains("strictly monotonic"));
    }

    #[test]
    fn invalid_subdivision_is_rejected() {
        let mut value = panel();
        value["target_events"][0]["subdivision_index"] = serde_json::json!(4);
        let error = score(&value).expect_err("subdivision numerator must be in range");
        assert!(error.contains("invalid rational subdivision"));
    }

    #[test]
    fn clock_or_capture_receipt_digest_must_be_present_and_well_formed() {
        let mut value = panel();
        value["target_events"][0]["capture_receipt_sha256"] = serde_json::json!("bad");
        let error = score(&value).expect_err("bad capture receipt digest must fail");
        assert!(error.contains("target onset capture receipt"));
    }

    #[test]
    fn constant_phase_offset_is_visible_even_when_inter_event_intervals_match() {
        let mut value = panel();
        for target in value["target_events"].as_array_mut().unwrap() {
            if target["outcome"] == "observed" {
                let onset = target["observed_onset_ns"].as_u64().unwrap();
                target["observed_onset_ns"] = serde_json::json!(onset + 50_000_000);
            }
        }
        let report = score(&value).expect("constant offset should remain scoreable");
        assert!(report.signed_phase_bias_beats.unwrap() > 0.10);
    }

    #[test]
    fn missed_events_and_extra_onsets_fail_qualification_even_when_phase_is_good() {
        let mut value = panel();
        value["target_events"][3]["outcome"] = serde_json::json!("missed");
        value["target_events"][3]["observed_onset_ns"] = serde_json::Value::Null;
        value["target_events"][3]["capture_receipt_sha256"] = serde_json::Value::Null;
        value["extra_onsets"] = serde_json::json!([{
            "event_id":"extra-1",
            "instrument_id":"hi_hat",
            "observed_onset_ns":2_650_000_000_u64,
            "capture_receipt_sha256":"2222222222222222222222222222222222222222222222222222222222222222"
        }]);
        let report = score(&value).expect("missed/extra events should be reported");
        assert_eq!(report.missed_count, 1);
        assert_eq!(report.extra_onset_count, 1);
        assert_eq!(report.target_accounting_completeness, 1.0);
        assert_eq!(report.observed_target_coverage_rate, 0.75);
        assert!(!report.qualified_for_declared_timing_contract);
    }

    #[test]
    fn unstable_tempo_estimates_never_qualify() {
        let mut value = panel();
        for estimate in value["tempo_estimates"].as_array_mut().unwrap() {
            if estimate["change_id"] == "change-2" {
                estimate["estimated_period_ns"] = serde_json::json!(500_000_000_u64);
            }
        }
        let report = score(&value).expect("unstable estimates remain scoreable");
        assert!(!report.tempo_adaptation[1].recovered);
        assert!(!report.qualified_for_declared_timing_contract);
    }

    #[test]
    fn missing_tempo_recovery_remains_a_failure_not_a_pass() {
        let mut value = panel();
        value["tempo_estimates"]
            .as_array_mut()
            .unwrap()
            .retain(|estimate| estimate["change_id"] != "change-2");
        let report = score(&value).expect("no recovery is a reportable result");
        assert!(!report.tempo_adaptation[1].recovered);
        assert!(!report.qualified_for_declared_timing_contract);
    }

    #[test]
    fn expected_onset_is_consumed_from_the_canonical_map_not_recomputed() {
        let mut value = panel();
        value["target_events"][0]["expected_onset_ns"] = serde_json::json!(5_000_000);
        let report = score(&value).expect("canonical mapper output should be directly consumed");
        assert_eq!(report.event_findings[0].expected_onset_ns, 5_000_000);
    }

    #[test]
    fn unsupported_time_map_kind_is_rejected() {
        let mut value = panel();
        value["reference_time_map_kind"] = serde_json::json!("self_inferred_output_map");
        let error = score(&value).expect_err("only canonical mapper outputs are admissible");
        assert!(error.contains("unsupported canonical reference time-map kind"));
    }
}
