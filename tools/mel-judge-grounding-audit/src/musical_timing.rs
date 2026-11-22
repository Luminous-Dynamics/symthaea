use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: &str = "mel-musical-timing-panel-v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MusicalTimingPanelV1 {
    schema_version: String,
    session_id: String,
    case_id: String,
    track: TrackKind,
    clock_domain_id: String,
    instrument_scope: String,
    reference_grid_receipt_sha256: String,
    onset_capture_receipt_sha256: String,
    clock_calibration_receipt_sha256: String,
    policy: TimingPolicyV1,
    tempo_segments: Vec<TempoSegmentV1>,
    target_events: Vec<TargetEventV1>,
    tempo_estimates: Vec<TempoEstimateV1>,
    extra_onsets: Vec<ExtraOnsetV1>,
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
struct TimingPolicyV1 {
    maximum_relative_period_error: f64,
    required_consecutive_stable_estimates: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TempoSegmentV1 {
    segment_id: String,
    start_beat_index: u64,
    phase_zero_ns: u64,
    change_observed_ns: u64,
    beat_period_ns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TargetEventV1 {
    event_id: String,
    instrument_id: String,
    segment_id: String,
    beat_index: u64,
    subdivision_index: u32,
    subdivision_count: u32,
    outcome: String,
    observed_onset_ns: Option<u64>,
    capture_receipt_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TempoEstimateV1 {
    estimate_id: String,
    segment_id: String,
    observed_at_ns: u64,
    evidence_window_end_ns: u64,
    estimated_period_ns: u64,
    confidence: f64,
    provenance_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExtraOnsetV1 {
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
    pub segment_id: String,
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
    pub segment_id: String,
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
    pub target_count: usize,
    pub observed_count: usize,
    pub missed_count: usize,
    pub extra_onset_count: usize,
    pub event_match_completeness: f64,
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
    let panel: MusicalTimingPanelV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid musical timing panel JSON: {error}"))?;
    score_panel(&panel)
}

fn score_panel(panel: &MusicalTimingPanelV1) -> Result<MusicalTimingReport, String> {
    validate_header(panel)?;

    let segments = validate_segments(&panel.tempo_segments)?;
    let estimates = validate_estimates(panel, &segments)?;
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
        let segment = segments
            .get(target.segment_id.as_str())
            .ok_or_else(|| format!("event {} references unknown tempo segment", target.event_id))?;
        if target.beat_index < segment.start_beat_index {
            return Err(format!(
                "event {} precedes its declared tempo segment",
                target.event_id
            ));
        }
        if let Some(next_start) = next_segment_start(&panel.tempo_segments, segment) {
            if target.beat_index >= next_start {
                return Err(format!(
                    "event {} crosses the next tempo segment boundary",
                    target.event_id
                ));
            }
        }
        let expected = expected_onset_ns(segment, target)?;
        match target.outcome.as_str() {
            "observed" => {
                let observed = target.observed_onset_ns.ok_or_else(|| {
                    format!("observed event {} is missing observed_onset_ns", target.event_id)
                })?;
                let receipt = target.capture_receipt_sha256.as_deref().ok_or_else(|| {
                    format!("observed event {} is missing capture receipt", target.event_id)
                })?;
                require_digest(receipt, "target onset capture receipt")?;
                if observed < segment.change_observed_ns {
                    return Err(format!(
                        "event {} onset predates the tempo change becoming observable",
                        target.event_id
                    ));
                }
                let signed_error_ns = i128::from(observed) - i128::from(expected);
                let signed_phase = signed_error_ns as f64 / segment.beat_period_ns as f64;
                let absolute_phase = signed_phase.abs();
                let absolute_ms = (signed_error_ns.abs() as f64) / 1_000_000.0;
                let within_005 = absolute_phase <= 0.05;
                let within_010 = absolute_phase <= 0.10;
                let within_025 = absolute_phase <= 0.25;
                let within_050 = absolute_phase <= 0.50;
                event_findings.push(EventTimingFinding {
                    event_id: target.event_id.clone(),
                    instrument_id: target.instrument_id.clone(),
                    segment_id: target.segment_id.clone(),
                    expected_onset_ns: expected,
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
                    segment_id: target.segment_id.clone(),
                    expected_onset_ns: expected,
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
        if !extra_ids.insert(extra.event_id.as_str()) || event_ids.contains(extra.event_id.as_str()) {
            return Err(format!("duplicate or target-colliding extra onset ID: {}", extra.event_id));
        }
    }
    if !panel.extra_onsets.is_empty() {
        findings.push(format!("{} unmatched extra onsets were supplied", panel.extra_onsets.len()));
    }

    let signed: Vec<f64> = computed.iter().map(|item| item.signed_phase_error).collect();
    let absolute: Vec<f64> = computed.iter().map(|item| item.absolute_phase_error).collect();
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
        *target_counts.entry(target.instrument_id.clone()).or_default() += 1;
        if target.outcome == "missed" {
            *missed_counts.entry(target.instrument_id.clone()).or_default() += 1;
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

    let tempo_adaptation = score_tempo_adaptation(panel, &estimates)?;
    for summary in &tempo_adaptation {
        if !summary.recovered {
            findings.push(format!(
                "tempo segment {} did not reach {} consecutive stable estimates",
                summary.segment_id, summary.stable_estimates_required
            ));
        }
    }
    let future_information_violations = 0usize;
    let total_targets = panel.target_events.len();
    let event_match_completeness = if total_targets == 0 {
        0.0
    } else {
        (observed_count + missed_count) as f64 / total_targets as f64
    };
    if observed_count + missed_count != total_targets {
        findings.push("target event accounting is incomplete".to_owned());
    }

    Ok(MusicalTimingReport {
        schema_version: "mel-musical-timing-report-v1".to_owned(),
        session_id: panel.session_id.clone(),
        case_id: panel.case_id.clone(),
        track: panel.track,
        target_count: total_targets,
        observed_count,
        missed_count,
        extra_onset_count: panel.extra_onsets.len(),
        event_match_completeness,
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
        future_information_violations,
        qualified_for_declared_timing_contract: findings.is_empty(),
        qualification_findings: findings,
        event_findings,
        claim_ceiling: panel.claim_ceiling.clone(),
    })
}

fn validate_header(panel: &MusicalTimingPanelV1) -> Result<(), String> {
    if panel.schema_version != SCHEMA_VERSION {
        return Err("unsupported musical timing schema_version".to_owned());
    }
    require_text(&panel.session_id, "session_id")?;
    require_text(&panel.case_id, "case_id")?;
    require_text(&panel.clock_domain_id, "clock_domain_id")?;
    require_text(&panel.instrument_scope, "instrument_scope")?;
    require_digest(&panel.reference_grid_receipt_sha256, "reference grid receipt")?;
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
    {
        return Err("invalid tempo adaptation policy".to_owned());
    }
    if panel.tempo_segments.is_empty() {
        return Err("tempo_segments must be non-empty".to_owned());
    }
    if panel.target_events.is_empty() {
        return Err("target_events must be non-empty".to_owned());
    }
    Ok(())
}

fn validate_segments(segments: &[TempoSegmentV1]) -> Result<BTreeMap<&str, &TempoSegmentV1>, String> {
    let mut result = BTreeMap::new();
    let mut previous_beat = None;
    let mut previous_phase = None;
    let mut previous_observed = None;
    for segment in segments {
        require_text(&segment.segment_id, "tempo segment ID")?;
        if segment.beat_period_ns == 0 {
            return Err(format!("tempo segment {} has zero beat period", segment.segment_id));
        }
        if segment.change_observed_ns > segment.phase_zero_ns {
            return Err(format!(
                "tempo segment {} phase zero predates the change becoming observable",
                segment.segment_id
            ));
        }
        if previous_beat.is_some_and(|beat| segment.start_beat_index <= beat)
            || previous_phase.is_some_and(|time| segment.phase_zero_ns <= time)
            || previous_observed.is_some_and(|time| segment.change_observed_ns < time)
        {
            return Err("tempo segments must be ordered by beat and monotonic time".to_owned());
        }
        previous_beat = Some(segment.start_beat_index);
        previous_phase = Some(segment.phase_zero_ns);
        previous_observed = Some(segment.change_observed_ns);
        if result.insert(segment.segment_id.as_str(), segment).is_some() {
            return Err(format!("duplicate tempo segment ID: {}", segment.segment_id));
        }
    }
    Ok(result)
}

fn next_segment_start(segments: &[TempoSegmentV1], current: &TempoSegmentV1) -> Option<u64> {
    segments
        .iter()
        .filter(|segment| segment.start_beat_index > current.start_beat_index)
        .map(|segment| segment.start_beat_index)
        .min()
}

fn expected_onset_ns(segment: &TempoSegmentV1, target: &TargetEventV1) -> Result<u64, String> {
    let beat_offset = u128::from(target.beat_index - segment.start_beat_index);
    let period = u128::from(segment.beat_period_ns);
    let whole_beats = beat_offset.checked_mul(period).ok_or_else(|| "beat time overflow".to_owned())?;
    let subdivision = period
        .checked_mul(u128::from(target.subdivision_index))
        .ok_or_else(|| "subdivision time overflow".to_owned())?
        / u128::from(target.subdivision_count);
    let expected = u128::from(segment.phase_zero_ns)
        .checked_add(whole_beats)
        .and_then(|value| value.checked_add(subdivision))
        .ok_or_else(|| "expected onset timestamp overflow".to_owned())?;
    u64::try_from(expected).map_err(|_| "expected onset timestamp exceeds u64".to_owned())
}

fn validate_estimates<'a>(
    panel: &'a MusicalTimingPanelV1,
    segments: &BTreeMap<&str, &'a TempoSegmentV1>,
) -> Result<BTreeMap<&'a str, Vec<&'a TempoEstimateV1>>, String> {
    let mut result = BTreeMap::<&str, Vec<&TempoEstimateV1>>::new();
    let mut ids = BTreeSet::new();
    let mut previous_time = None;
    for estimate in &panel.tempo_estimates {
        require_text(&estimate.estimate_id, "tempo estimate ID")?;
        require_digest(&estimate.provenance_sha256, "tempo estimate provenance")?;
        if !ids.insert(estimate.estimate_id.as_str()) {
            return Err(format!("duplicate tempo estimate ID: {}", estimate.estimate_id));
        }
        if estimate.estimated_period_ns == 0
            || !estimate.confidence.is_finite()
            || !(0.0..=1.0).contains(&estimate.confidence)
        {
            return Err(format!("tempo estimate {} has invalid values", estimate.estimate_id));
        }
        if estimate.evidence_window_end_ns > estimate.observed_at_ns {
            return Err(format!(
                "tempo estimate {} references future evidence",
                estimate.estimate_id
            ));
        }
        if previous_time.is_some_and(|time| estimate.observed_at_ns < time) {
            return Err("tempo estimate observations must be monotonic".to_owned());
        }
        previous_time = Some(estimate.observed_at_ns);
        let segment = segments
            .get(estimate.segment_id.as_str())
            .ok_or_else(|| format!("estimate {} references unknown segment", estimate.estimate_id))?;
        if estimate.observed_at_ns < segment.change_observed_ns {
            return Err(format!(
                "tempo estimate {} was observed before its segment became available",
                estimate.estimate_id
            ));
        }
        result
            .entry(estimate.segment_id.as_str())
            .or_default()
            .push(estimate);
    }
    Ok(result)
}

fn score_tempo_adaptation<'a>(
    panel: &MusicalTimingPanelV1,
    estimates: &BTreeMap<&str, Vec<&'a TempoEstimateV1>>,
) -> Result<Vec<TempoAdaptationSummary>, String> {
    let required = panel.policy.required_consecutive_stable_estimates;
    let tolerance = panel.policy.maximum_relative_period_error;
    let mut output = Vec::new();
    for segment in &panel.tempo_segments {
        let values = estimates.get(segment.segment_id.as_str()).cloned().unwrap_or_default();
        let mut stable_run = 0usize;
        let mut stable_start = None;
        let mut recovered_at = None;
        for estimate in &values {
            let relative_error = (estimate.estimated_period_ns as f64 - segment.beat_period_ns as f64).abs()
                / segment.beat_period_ns as f64;
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
        let adaptation_time_ns = recovered_at.map(|at| at - segment.change_observed_ns);
        output.push(TempoAdaptationSummary {
            segment_id: segment.segment_id.clone(),
            reference_beat_period_ns: segment.beat_period_ns,
            estimate_count: values.len(),
            stable_estimates_required: required,
            recovered: recovered_at.is_some(),
            adaptation_time_ns,
            adaptation_time_beats: adaptation_time_ns.map(|time| time as f64 / segment.beat_period_ns as f64),
            stable_run_start_ns: stable_start,
            last_estimated_period_ns: values.last().map(|estimate| estimate.estimated_period_ns),
        });
    }
    Ok(output)
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
        Some(values.iter().filter(|value| **value <= threshold).count() as f64 / values.len() as f64)
    }
}

fn regression_slope(points: Vec<(f64, f64)>) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let x_mean = points.iter().map(|point| point.0).sum::<f64>() / points.len() as f64;
    let y_mean = points.iter().map(|point| point.1).sum::<f64>() / points.len() as f64;
    let numerator = points.iter().map(|point| (point.0 - x_mean) * (point.1 - y_mean)).sum::<f64>();
    let denominator = points.iter().map(|point| (point.0 - x_mean).powi(2)).sum::<f64>();
    if denominator == 0.0 { None } else { Some(numerator / denominator) }
}

fn require_text(value: &str, name: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{name} must be non-empty"))
    } else {
        Ok(())
    }
}

fn require_digest(value: &str, name: &str) -> Result<(), String> {
    if value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        Ok(())
    } else {
        Err(format!("{name} must be a lowercase 64-character SHA-256 digest"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = include_str!("../fixtures/valid-musical-timing-v1.json");

    fn panel() -> Value {
        serde_json::from_str(FIXTURE).expect("known-answer timing fixture JSON")
    }

    fn score(value: &Value) -> Result<MusicalTimingReport, String> {
        parse_and_score(&value.to_string())
    }

    #[test]
    fn known_answer_reports_phase_error_and_tempo_recovery() {
        let report = parse_and_score(FIXTURE).expect("known-answer panel should score");
        assert_eq!(report.target_count, 4);
        assert_eq!(report.observed_count, 4);
        assert_eq!(report.missed_count, 0);
        assert_eq!(report.extra_onset_count, 0);
        assert_eq!(report.within_025_beat_rate, Some(1.0));
        assert!(report.tempo_adaptation.iter().all(|item| item.recovered));
        assert!(report.qualified_for_declared_timing_contract);
    }

    #[test]
    fn backdated_tempo_segment_is_rejected() {
        let mut value = panel();
        value["tempo_segments"][1]["change_observed_ns"] = serde_json::json!(2_100_000_000_u64);
        let error = score(&value).expect_err("tempo event cannot be backdated");
        assert!(error.contains("phase zero predates"));
    }

    #[test]
    fn estimate_before_tempo_change_is_rejected() {
        let mut value = panel();
        value["tempo_estimates"][5]["observed_at_ns"] = serde_json::json!(1_999_999_999_u64);
        let error = score(&value).expect_err("no estimate can precede change availability");
        assert!(error.contains("observed before its segment became available"));
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
    fn invalid_subdivision_is_rejected() {
        let mut value = panel();
        value["target_events"][0]["subdivision_index"] = serde_json::json!(4);
        let error = score(&value).expect_err("subdivision numerator must be in range");
        assert!(error.contains("invalid rational subdivision"));
    }

    #[test]
    fn beat_index_cannot_cross_segment_boundary() {
        let mut value = panel();
        value["target_events"][0]["beat_index"] = serde_json::json!(4);
        let error = score(&value).expect_err("target beat belongs in next tempo segment");
        assert!(error.contains("crosses the next tempo segment"));
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
        assert_eq!(report.event_match_completeness, 1.0);
        assert!(!report.qualified_for_declared_timing_contract);
    }

    #[test]
    fn unstable_tempo_estimates_never_qualify() {
        let mut value = panel();
        for estimate in value["tempo_estimates"].as_array_mut().unwrap() {
            if estimate["segment_id"] == "tempo-2" {
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
        value["tempo_estimates"].as_array_mut().unwrap().retain(|estimate| estimate["segment_id"] != "tempo-2");
        let report = score(&value).expect("no recovery is a reportable result");
        assert!(!report.tempo_adaptation[1].recovered);
        assert!(!report.qualified_for_declared_timing_contract);
    }

    #[test]
    fn segment_boundary_beat_zero_is_scored_on_the_new_period() {
        let mut value = panel();
        value["target_events"][2]["beat_index"] = serde_json::json!(4);
        value["target_events"][2]["subdivision_index"] = serde_json::json!(0);
        value["target_events"][2]["observed_onset_ns"] = serde_json::json!(2_000_000_000_u64);
        let report = score(&value).expect("new segment boundary event should score");
        assert_eq!(report.event_findings[2].expected_onset_ns, 2_000_000_000);
        assert_eq!(report.event_findings[2].absolute_phase_error_beats, Some(0.0));
    }
}
