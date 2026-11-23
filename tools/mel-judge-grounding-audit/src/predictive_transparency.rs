use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: &str = "mel-musical-forecast-panel-v2";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ForecastPanelV1 {
    schema_version: String,
    session_id: String,
    case_id: String,
    clock_domain_id: String,
    reference_time_map_id: String,
    reference_time_map_version: String,
    reference_time_map_kind: String,
    reference_time_map_clock_domain_id: String,
    reference_time_map_receipt_sha256: String,
    time_alignment_source_clock_domain_id: String,
    time_alignment_target_clock_domain_id: String,
    time_alignment_receipt_sha256: String,
    time_alignment_valid_from_ns: u64,
    time_alignment_valid_until_ns: u64,
    time_alignment_max_uncertainty_ns: u64,
    input_snapshots: Vec<InputSnapshotV1>,
    policy: ForecastPolicyV1,
    forecasts: Vec<ForecastV1>,
    displays: Vec<ForecastDisplayV1>,
    dispositions: Vec<ForecastDispositionV1>,
    realizations: Vec<ForecastRealizationV1>,
    observation_closures: Vec<ObservationClosureV1>,
    claim_ceiling: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputSnapshotV1 {
    input_snapshot_id: String,
    available_at_ns: u64,
    input_history_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ForecastPolicyV1 {
    maximum_onset_error_beats: f64,
    minimum_anticipation_lead_ns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ForecastV1 {
    forecast_id: String,
    commitment_series_id: String,
    revision: u32,
    supersedes_forecast_id: Option<String>,
    input_snapshot_id: String,
    committed_at_ns: u64,
    target_start_ns: u64,
    target_end_ns: u64,
    expires_at_ns: u64,
    input_history_sha256: String,
    policy_sha256: String,
    events: Vec<ForecastedEventV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ForecastedEventV1 {
    forecast_event_id: String,
    role: String,
    beat_index: u64,
    expected_onset_ns: u64,
    local_beat_period_ns: u64,
    presence_probability: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ForecastDisplayV1 {
    forecast_id: String,
    delivered_at_ns: u64,
    displayed_at_ns: u64,
    display_receipt_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ForecastDispositionV1 {
    forecast_id: String,
    action: String,
    acted_at_ns: u64,
    actor_scope: String,
    action_receipt_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ForecastRealizationV1 {
    realization_id: String,
    forecast_id: String,
    started_at_ns: u64,
    completed_at_ns: u64,
    artifact_sha256: String,
    observation_capture_id: String,
    observed_event_manifest_sha256: String,
    events: Vec<ProducedEventV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProducedEventV1 {
    event_id: String,
    forecast_event_id: Option<String>,
    role: String,
    #[serde(rename = "beat_index")]
    _beat_index: u64,
    onset_ns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservedEventV1 {
    event_id: String,
    forecast_event_id: Option<String>,
    role: String,
    beat_index: u64,
    onset_ns: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ObservationClosureV1 {
    forecast_id: String,
    capture_id: String,
    source_kind: String,
    observer_id: String,
    producer_id: String,
    event_observer_id: String,
    event_observer_qualification_id: String,
    event_observer_build_sha256: String,
    event_observer_config_sha256: String,
    event_observer_qualification_receipt_sha256: String,
    observed_event_manifest_sha256: String,
    event_observer_receipt_sha256: String,
    observed_events: Vec<ObservedEventV1>,
    clock_domain_id: String,
    coverage_start_ns: u64,
    coverage_end_ns: u64,
    terminal_watermark_ns: u64,
    closed_at_ns: u64,
    first_sequence: u64,
    end_sequence_exclusive: u64,
    expected_sequence_count: u64,
    observed_sequence_count: u64,
    missing_sequence_count: u64,
    duplicate_sequence_count: u64,
    sample_rate_hz: u32,
    channel_count: u16,
    expected_sample_frames: u64,
    observed_sample_frames: u64,
    dropped_sample_frames: u64,
    xrun_count: u32,
    heartbeat_interval_ns: u64,
    expected_heartbeat_count: u64,
    observed_heartbeat_count: u64,
    missing_heartbeat_count: u64,
    capture_route_sha256: String,
    capture_payload_sha256: String,
    observer_build_sha256: String,
    closure_receipt_sha256: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct EventForecastFinding {
    pub forecast_id: String,
    pub forecast_event_id: String,
    pub observed_event_id: Option<String>,
    pub predicted_role: String,
    pub observed_role: Option<String>,
    pub predicted_beat_index: u64,
    pub observed_beat_index: Option<u64>,
    pub beat_index_match: Option<bool>,
    pub expected_onset_ns: u64,
    pub observed_onset_ns: Option<u64>,
    pub signed_onset_error_ns: Option<i128>,
    pub absolute_onset_error_beats: Option<f64>,
    pub worst_case_absolute_onset_error_beats: Option<f64>,
    pub within_onset_tolerance: Option<bool>,
    pub presence_probability: f64,
    pub presence_observed: Option<bool>,
    pub presence_brier_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ForecastSeriesSummary {
    pub commitment_series_id: String,
    pub revision_count: usize,
    pub latest_forecast_id: String,
    pub has_supersession_chain: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ForecastAuditReport {
    pub schema_version: String,
    pub session_id: String,
    pub case_id: String,
    pub reference_time_map_id: String,
    pub reference_time_map_version: String,
    pub reference_time_map_kind: String,
    pub reference_time_map_clock_domain_id: String,
    pub time_alignment_source_clock_domain_id: String,
    pub time_alignment_target_clock_domain_id: String,
    pub time_alignment_valid_from_ns: u64,
    pub time_alignment_valid_until_ns: u64,
    pub time_alignment_max_uncertainty_ns: u64,
    pub forecast_count: usize,
    pub forecast_event_count: usize,
    pub display_count: usize,
    pub displayed_before_target_count: usize,
    pub forecasts_without_display: usize,
    pub disposition_count: usize,
    pub realization_count: usize,
    pub observation_closure_count: usize,
    pub complete_observation_closure_count: usize,
    pub incomplete_observation_closure_count: usize,
    pub forecasts_without_observation_closure: usize,
    pub output_observation_structurally_eligible: bool,
    pub producer_reported_event_record_count: usize,
    pub observed_event_record_count: usize,
    pub qualified_observed_event_record_count: usize,
    pub matched_forecast_event_count: usize,
    pub unfulfilled_forecast_event_count: usize,
    pub unobserved_forecast_event_count: usize,
    pub presence_brier_sample_count: usize,
    pub unforecasted_observed_event_count: usize,
    pub role_match_rate: Option<f64>,
    pub beat_index_match_rate: Option<f64>,
    pub conditional_presence_brier_score: Option<f64>,
    pub absolute_onset_error_beats: Vec<f64>,
    pub median_absolute_onset_error_beats: Option<f64>,
    pub p95_absolute_onset_error_beats: Option<f64>,
    pub within_onset_tolerance_rate: Option<f64>,
    pub median_anticipation_lead_ns: Option<u64>,
    pub series: Vec<ForecastSeriesSummary>,
    pub structurally_valid_anticipatory_trace: bool,
    pub qualification_findings: Vec<String>,
    pub event_findings: Vec<EventForecastFinding>,
    pub claim_ceiling: Vec<String>,
}

pub fn parse_and_audit(input: &str) -> Result<ForecastAuditReport, String> {
    let panel: ForecastPanelV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid musical forecast panel JSON: {error}"))?;
    audit_panel(&panel)
}

fn audit_panel(panel: &ForecastPanelV1) -> Result<ForecastAuditReport, String> {
    validate_header(panel)?;

    let mut findings = Vec::new();
    let mut forecast_ids = BTreeMap::<&str, &ForecastV1>::new();
    let mut forecast_events = BTreeMap::<(&str, &str), &ForecastedEventV1>::new();
    let mut series_members = BTreeMap::<&str, Vec<&ForecastV1>>::new();

    for forecast in &panel.forecasts {
        validate_forecast(forecast, panel)?;
        if forecast_ids
            .insert(forecast.forecast_id.as_str(), forecast)
            .is_some()
        {
            return Err(format!("duplicate forecast_id: {}", forecast.forecast_id));
        }
        series_members
            .entry(forecast.commitment_series_id.as_str())
            .or_default()
            .push(forecast);
        for event in &forecast.events {
            if forecast_events
                .insert(
                    (
                        forecast.forecast_id.as_str(),
                        event.forecast_event_id.as_str(),
                    ),
                    event,
                )
                .is_some()
            {
                return Err(format!(
                    "duplicate forecast_event_id {} in forecast {}",
                    event.forecast_event_id, forecast.forecast_id
                ));
            }
        }
    }

    let series_summaries = validate_series(&mut series_members, &forecast_ids)?;
    let latest_by_series: BTreeMap<&str, &ForecastV1> = series_members
        .iter()
        .filter_map(|(series_id, members)| {
            members
                .iter()
                .max_by_key(|forecast| forecast.revision)
                .map(|forecast| (*series_id, *forecast))
        })
        .collect();

    let mut displays_by_forecast = BTreeMap::<&str, Vec<&ForecastDisplayV1>>::new();
    let mut display_ids = BTreeSet::new();
    let mut displayed_before_target_count = 0usize;
    let mut lead_times = Vec::new();

    for display in &panel.displays {
        let forecast = forecast_ids
            .get(display.forecast_id.as_str())
            .copied()
            .ok_or_else(|| {
                format!(
                    "display references unknown forecast {}",
                    display.forecast_id
                )
            })?;
        require_digest(&display.display_receipt_sha256, "display receipt digest")?;
        if display.displayed_at_ns < display.delivered_at_ns
            || display.delivered_at_ns < forecast.committed_at_ns
        {
            return Err(format!(
                "display timestamps violate commit/delivery/display order for {}",
                display.forecast_id
            ));
        }
        if !display_ids.insert(display.forecast_id.as_str()) {
            return Err(format!(
                "duplicate display receipt for forecast {}",
                display.forecast_id
            ));
        }
        let superseded_at = series_members
            .get(forecast.commitment_series_id.as_str())
            .and_then(|members| {
                members
                    .iter()
                    .filter(|candidate| {
                        candidate.supersedes_forecast_id.as_deref()
                            == Some(forecast.forecast_id.as_str())
                    })
                    .map(|candidate| candidate.committed_at_ns)
                    .min()
            });
        if superseded_at.is_some_and(|time| display.displayed_at_ns >= time) {
            findings.push(format!(
                "superseded forecast {} was displayed after its successor was committed",
                display.forecast_id
            ));
        }
        if display.displayed_at_ns < forecast.target_start_ns
            && display.displayed_at_ns < forecast.expires_at_ns
        {
            displayed_before_target_count += 1;
            lead_times.push(forecast.target_start_ns - display.displayed_at_ns);
        } else {
            findings.push(format!(
                "forecast {} was displayed too late or after expiry",
                display.forecast_id
            ));
        }
        displays_by_forecast
            .entry(display.forecast_id.as_str())
            .or_default()
            .push(display);
    }

    let mut dispositions_by_forecast = BTreeMap::<&str, &ForecastDispositionV1>::new();
    for disposition in &panel.dispositions {
        let forecast = forecast_ids
            .get(disposition.forecast_id.as_str())
            .copied()
            .ok_or_else(|| {
                format!(
                    "disposition references unknown forecast {}",
                    disposition.forecast_id
                )
            })?;
        require_text(&disposition.actor_scope, "disposition actor_scope")?;
        require_digest(&disposition.action_receipt_sha256, "action receipt digest")?;
        let display = displays_by_forecast
            .get(disposition.forecast_id.as_str())
            .and_then(|items| items.first())
            .ok_or_else(|| {
                format!(
                    "disposition for {} has no preceding display",
                    disposition.forecast_id
                )
            })?;
        if disposition.acted_at_ns < display.displayed_at_ns
            || disposition.acted_at_ns >= forecast.target_start_ns
        {
            findings.push(format!(
                "disposition for {} was not actionable before the target window",
                disposition.forecast_id
            ));
        }
        if dispositions_by_forecast
            .insert(disposition.forecast_id.as_str(), disposition)
            .is_some()
        {
            return Err(format!(
                "duplicate disposition for forecast {}",
                disposition.forecast_id
            ));
        }
        if !matches!(
            disposition.action.as_str(),
            "accept" | "steer" | "veto" | "branch" | "ignore" | "abstain"
        ) {
            return Err(format!(
                "unsupported forecast disposition {}",
                disposition.action
            ));
        }
    }

    for forecast in &panel.forecasts {
        let child_commit_time = series_members
            .get(forecast.commitment_series_id.as_str())
            .and_then(|members| {
                members
                    .iter()
                    .find(|candidate| {
                        candidate.supersedes_forecast_id.as_deref()
                            == Some(forecast.forecast_id.as_str())
                    })
                    .map(|candidate| candidate.committed_at_ns)
            });
        if let Some(disposition) = dispositions_by_forecast.get(forecast.forecast_id.as_str())
            && matches!(disposition.action.as_str(), "steer" | "branch")
        {
            match child_commit_time {
                None => findings.push(format!(
                    "{} disposition for {} has no superseding forecast",
                    disposition.action, forecast.forecast_id
                )),
                Some(committed_at_ns) if committed_at_ns <= disposition.acted_at_ns => {
                    findings.push(format!(
                        "{} disposition for {} occurred after its successor was already committed",
                        disposition.action, forecast.forecast_id
                    ));
                }
                _ => {}
            }
        }
    }

    let mut realization_ids = BTreeSet::new();
    let mut realized_forecast_ids = BTreeSet::new();
    let mut producer_event_ids = BTreeSet::new();
    let mut produced_forecast_event_ids = BTreeSet::new();
    let mut producer_reported_event_record_count = 0usize;

    for realization in &panel.realizations {
        if !realization_ids.insert(realization.realization_id.as_str()) {
            return Err(format!(
                "duplicate realization_id: {}",
                realization.realization_id
            ));
        }
        if !realized_forecast_ids.insert(realization.forecast_id.as_str()) {
            return Err(format!(
                "forecast {} has more than one realization record",
                realization.forecast_id
            ));
        }
        let forecast = forecast_ids
            .get(realization.forecast_id.as_str())
            .copied()
            .ok_or_else(|| {
                format!(
                    "realization references unknown forecast {}",
                    realization.forecast_id
                )
            })?;
        require_digest(&realization.artifact_sha256, "realization artifact digest")?;
        require_text(
            &realization.observation_capture_id,
            "realization observation_capture_id",
        )?;
        require_digest(
            &realization.observed_event_manifest_sha256,
            "realization observed event manifest digest",
        )?;
        if realization.started_at_ns < forecast.committed_at_ns
            || realization.completed_at_ns < realization.started_at_ns
        {
            return Err(format!(
                "realization {} violates commit/start/completion order",
                realization.realization_id
            ));
        }
        if realization.started_at_ns < forecast.target_start_ns {
            findings.push(format!(
                "forecast {} realization started before its target window",
                forecast.forecast_id
            ));
        }
        if realization.started_at_ns > forecast.target_end_ns {
            findings.push(format!(
                "forecast {} realization started after its target window ended",
                forecast.forecast_id
            ));
        }
        if realization.started_at_ns > forecast.expires_at_ns {
            findings.push(format!(
                "stale forecast {} was realized after expiry",
                forecast.forecast_id
            ));
        }
        if series_members
            .get(forecast.commitment_series_id.as_str())
            .is_some_and(|members| {
                members.iter().any(|candidate| {
                    candidate.supersedes_forecast_id.as_deref()
                        == Some(forecast.forecast_id.as_str())
                        && candidate.committed_at_ns <= realization.started_at_ns
                })
            })
        {
            findings.push(format!(
                "superseded forecast {} was used after its successor was committed",
                forecast.forecast_id
            ));
        }
        if dispositions_by_forecast
            .get(forecast.forecast_id.as_str())
            .is_some_and(|disposition| disposition.action == "veto")
        {
            findings.push(format!(
                "vetoed forecast {} still produced a realization",
                forecast.forecast_id
            ));
        }
        if latest_by_series
            .get(forecast.commitment_series_id.as_str())
            .is_none_or(|latest| latest.forecast_id != forecast.forecast_id)
            && realization.started_at_ns
                >= latest_by_series[forecast.commitment_series_id.as_str()].committed_at_ns
        {
            findings.push(format!(
                "realization {} uses a non-current forecast version",
                realization.realization_id
            ));
        }

        // Producer-side event claims remain visible, but they are never used as
        // the observation ground truth for forecast presence or timing scores.
        for event in &realization.events {
            require_text(&event.event_id, "producer event_id")?;
            require_text(&event.role, "producer event role")?;
            if !producer_event_ids.insert(event.event_id.as_str()) {
                return Err(format!("duplicate producer event_id: {}", event.event_id));
            }
            if event.onset_ns < realization.started_at_ns
                || event.onset_ns > realization.completed_at_ns
            {
                return Err(format!(
                    "producer event {} falls outside its realization interval",
                    event.event_id
                ));
            }
            producer_reported_event_record_count += 1;
            if let Some(forecast_event_id) = event.forecast_event_id.as_deref() {
                if !forecast_events
                    .contains_key(&(forecast.forecast_id.as_str(), forecast_event_id))
                {
                    return Err(format!(
                        "producer event {} references unknown forecast event {}",
                        event.event_id, forecast_event_id
                    ));
                }
                if !produced_forecast_event_ids
                    .insert((forecast.forecast_id.as_str(), forecast_event_id))
                {
                    return Err(format!(
                        "forecast event {} was emitted more than once",
                        forecast_event_id
                    ));
                }
                if event.onset_ns < forecast.target_start_ns
                    || event.onset_ns >= forecast.target_end_ns
                {
                    findings.push(format!(
                        "producer event {} falls outside forecast {} target window",
                        event.event_id, forecast.forecast_id
                    ));
                }
            }
        }
    }

    let mut closure_by_forecast = BTreeMap::<&str, bool>::new();
    let mut observed_by_forecast_event = BTreeMap::<(&str, &str), &ObservedEventV1>::new();
    let mut observed_event_ids = BTreeSet::new();
    let mut unforecasted_observed_by_forecast = BTreeMap::<&str, usize>::new();
    let mut observed_event_record_count = 0usize;
    let mut qualified_observed_event_record_count = 0usize;
    let mut complete_observation_closure_count = 0usize;
    let mut incomplete_observation_closure_count = 0usize;

    for closure in &panel.observation_closures {
        let forecast = forecast_ids
            .get(closure.forecast_id.as_str())
            .copied()
            .ok_or_else(|| {
                format!(
                    "observation closure references unknown forecast {}",
                    closure.forecast_id
                )
            })?;
        if closure_by_forecast.contains_key(closure.forecast_id.as_str()) {
            return Err(format!(
                "forecast {} has more than one observation closure",
                closure.forecast_id
            ));
        }
        validate_observation_closure_shape(closure)?;
        for event in &closure.observed_events {
            require_text(&event.event_id, "observed event_id")?;
            require_text(&event.role, "observed event role")?;
            if !observed_event_ids.insert(event.event_id.as_str()) {
                return Err(format!("duplicate observed event_id: {}", event.event_id));
            }
            if event.onset_ns < closure.coverage_start_ns
                || event.onset_ns >= closure.coverage_end_ns
            {
                return Err(format!(
                    "observed event {} falls outside its capture interval",
                    event.event_id
                ));
            }
            if event.onset_ns < forecast.target_start_ns || event.onset_ns >= forecast.target_end_ns
            {
                return Err(format!(
                    "observed event {} falls outside forecast {} target window",
                    event.event_id, forecast.forecast_id
                ));
            }
            observed_event_record_count += 1;
            if let Some(forecast_event_id) = event.forecast_event_id.as_deref() {
                if !forecast_events
                    .contains_key(&(forecast.forecast_id.as_str(), forecast_event_id))
                {
                    return Err(format!(
                        "observed event {} references unknown forecast event {}",
                        event.event_id, forecast_event_id
                    ));
                }
                if observed_by_forecast_event
                    .insert((forecast.forecast_id.as_str(), forecast_event_id), event)
                    .is_some()
                {
                    return Err(format!(
                        "forecast event {} was observed more than once",
                        forecast_event_id
                    ));
                }
            } else {
                *unforecasted_observed_by_forecast
                    .entry(forecast.forecast_id.as_str())
                    .or_default() += 1;
            }
        }

        let closure_findings = observation_closure_incompleteness(closure, forecast, panel);
        let complete = closure_findings.is_empty();
        if complete {
            complete_observation_closure_count += 1;
            qualified_observed_event_record_count += closure.observed_events.len();
        } else {
            incomplete_observation_closure_count += 1;
            for reason in closure_findings {
                findings.push(format!(
                    "forecast {} observation closure {} is incomplete: {}",
                    closure.forecast_id, closure.capture_id, reason
                ));
            }
        }
        closure_by_forecast.insert(closure.forecast_id.as_str(), complete);
    }
    let forecasts_without_observation_closure = panel
        .forecasts
        .iter()
        .filter(|forecast| !closure_by_forecast.contains_key(forecast.forecast_id.as_str()))
        .count();
    for forecast in &panel.forecasts {
        if !closure_by_forecast.contains_key(forecast.forecast_id.as_str()) {
            findings.push(format!(
                "forecast {} has no observation closure; event absence remains unobserved",
                forecast.forecast_id
            ));
        }
    }

    let unforecasted_observed_event_count = unforecasted_observed_by_forecast
        .iter()
        .filter_map(|(forecast_id, count)| {
            closure_by_forecast
                .get(*forecast_id)
                .and_then(|complete| if *complete { Some(*count) } else { None })
        })
        .sum::<usize>();

    let mut event_findings = Vec::new();
    let mut brier_sum = 0.0;
    let mut brier_count = 0usize;
    let mut role_matches = 0usize;
    let mut beat_index_matches = 0usize;
    let mut matched_count = 0usize;
    let mut unfulfilled_count = 0usize;
    let mut unobserved_count = 0usize;
    let mut onset_errors = Vec::new();

    for forecast in &panel.forecasts {
        let observation_complete = closure_by_forecast
            .get(forecast.forecast_id.as_str())
            .copied()
            .unwrap_or(false);
        for predicted in &forecast.events {
            let observed_candidate = observed_by_forecast_event.get(&(
                forecast.forecast_id.as_str(),
                predicted.forecast_event_id.as_str(),
            ));
            let observed = if observation_complete {
                observed_candidate
            } else {
                None
            };
            let (
                observed_event_id,
                observed_role,
                observed_beat_index,
                beat_index_match,
                observed_onset_ns,
                signed_error_ns,
                phase_error,
            ) = if let Some(actual) = observed {
                let signed = i128::from(actual.onset_ns) - i128::from(predicted.expected_onset_ns);
                let phase = signed as f64 / predicted.local_beat_period_ns as f64;
                role_matches += usize::from(actual.role == predicted.role);
                let beat_match = actual.beat_index == predicted.beat_index;
                beat_index_matches += usize::from(beat_match);
                matched_count += 1;
                onset_errors.push(phase.abs());
                (
                    Some(actual.event_id.clone()),
                    Some(actual.role.clone()),
                    Some(actual.beat_index),
                    Some(beat_match),
                    Some(actual.onset_ns),
                    Some(signed),
                    Some(phase),
                )
            } else {
                if observation_complete {
                    unfulfilled_count += 1;
                } else {
                    unobserved_count += 1;
                }
                (None, None, None, None, None, None, None)
            };
            let presence_observed = observation_complete.then_some(observed.is_some());
            let brier = presence_observed.map(|observed| {
                let observed_probability = if observed { 1.0 } else { 0.0 };
                (predicted.presence_probability - observed_probability).powi(2)
            });
            if let Some(value) = brier {
                brier_sum += value;
                brier_count += 1;
            }
            let worst_case_phase_error = signed_error_ns.map(|signed| {
                (signed.unsigned_abs() as f64 + panel.time_alignment_max_uncertainty_ns as f64)
                    / predicted.local_beat_period_ns as f64
            });
            let within =
                worst_case_phase_error.map(|value| value <= panel.policy.maximum_onset_error_beats);
            if within == Some(false) {
                findings.push(format!(
                    "forecast event {} exceeded onset tolerance including clock-alignment uncertainty",
                    predicted.forecast_event_id
                ));
            }
            event_findings.push(EventForecastFinding {
                forecast_id: forecast.forecast_id.clone(),
                forecast_event_id: predicted.forecast_event_id.clone(),
                observed_event_id,
                predicted_role: predicted.role.clone(),
                observed_role,
                predicted_beat_index: predicted.beat_index,
                observed_beat_index,
                beat_index_match,
                expected_onset_ns: predicted.expected_onset_ns,
                observed_onset_ns,
                signed_onset_error_ns: signed_error_ns,
                absolute_onset_error_beats: phase_error.map(f64::abs),
                worst_case_absolute_onset_error_beats: worst_case_phase_error,
                within_onset_tolerance: within,
                presence_probability: predicted.presence_probability,
                presence_observed,
                presence_brier_score: brier,
            });
        }
    }

    let forecasts_without_display = panel
        .forecasts
        .iter()
        .filter(|forecast| !displays_by_forecast.contains_key(forecast.forecast_id.as_str()))
        .count();
    if forecasts_without_display > 0 {
        findings.push(format!(
            "{forecasts_without_display} forecast(s) were never rendered to the collaborator"
        ));
    }

    let mut absolute_sorted = onset_errors.clone();
    absolute_sorted.sort_by(f64::total_cmp);
    let within_count = event_findings
        .iter()
        .filter(|finding| finding.within_onset_tolerance == Some(true))
        .count();
    let onset_observed_count = matched_count;
    let median_absolute_onset_error_beats = percentile(&absolute_sorted, 50);
    let p95_absolute_onset_error_beats = percentile(&absolute_sorted, 95);
    let structurally_valid = findings.is_empty();
    Ok(ForecastAuditReport {
        schema_version: "mel-musical-forecast-report-v2".to_owned(),
        session_id: panel.session_id.clone(),
        case_id: panel.case_id.clone(),
        reference_time_map_id: panel.reference_time_map_id.clone(),
        reference_time_map_version: panel.reference_time_map_version.clone(),
        reference_time_map_kind: panel.reference_time_map_kind.clone(),
        reference_time_map_clock_domain_id: panel.reference_time_map_clock_domain_id.clone(),
        time_alignment_source_clock_domain_id: panel.time_alignment_source_clock_domain_id.clone(),
        time_alignment_target_clock_domain_id: panel.time_alignment_target_clock_domain_id.clone(),
        time_alignment_valid_from_ns: panel.time_alignment_valid_from_ns,
        time_alignment_valid_until_ns: panel.time_alignment_valid_until_ns,
        time_alignment_max_uncertainty_ns: panel.time_alignment_max_uncertainty_ns,
        forecast_count: panel.forecasts.len(),
        forecast_event_count: forecast_events.len(),
        display_count: panel.displays.len(),
        displayed_before_target_count,
        forecasts_without_display,
        disposition_count: panel.dispositions.len(),
        realization_count: panel.realizations.len(),
        observation_closure_count: panel.observation_closures.len(),
        complete_observation_closure_count,
        incomplete_observation_closure_count,
        forecasts_without_observation_closure,
        output_observation_structurally_eligible: complete_observation_closure_count
            == panel.forecasts.len(),
        producer_reported_event_record_count,
        observed_event_record_count,
        qualified_observed_event_record_count,
        matched_forecast_event_count: matched_count,
        unfulfilled_forecast_event_count: unfulfilled_count,
        unobserved_forecast_event_count: unobserved_count,
        presence_brier_sample_count: brier_count,
        unforecasted_observed_event_count,
        role_match_rate: rate(role_matches, matched_count),
        beat_index_match_rate: rate(beat_index_matches, matched_count),
        conditional_presence_brier_score: mean(brier_sum, brier_count),
        absolute_onset_error_beats: absolute_sorted,
        median_absolute_onset_error_beats,
        p95_absolute_onset_error_beats,
        within_onset_tolerance_rate: rate(within_count, onset_observed_count),
        median_anticipation_lead_ns: percentile_u64(&mut lead_times, 50),
        series: series_summaries,
        structurally_valid_anticipatory_trace: structurally_valid,
        qualification_findings: findings,
        event_findings,
        claim_ceiling: panel.claim_ceiling.clone(),
    })
}

fn validate_header(panel: &ForecastPanelV1) -> Result<(), String> {
    if panel.schema_version != SCHEMA_VERSION {
        return Err("unsupported musical forecast schema_version".to_owned());
    }
    require_text(&panel.session_id, "session_id")?;
    require_text(&panel.case_id, "case_id")?;
    require_text(&panel.clock_domain_id, "clock_domain_id")?;
    require_text(&panel.reference_time_map_id, "reference_time_map_id")?;
    require_text(
        &panel.reference_time_map_version,
        "reference_time_map_version",
    )?;
    if !matches!(
        panel.reference_time_map_kind.as_str(),
        "notation_tempo" | "performance_time"
    ) {
        return Err("unsupported canonical time-map kind".to_owned());
    }
    require_text(
        &panel.reference_time_map_clock_domain_id,
        "reference_time_map_clock_domain_id",
    )?;
    require_text(
        &panel.time_alignment_source_clock_domain_id,
        "time_alignment_source_clock_domain_id",
    )?;
    require_text(
        &panel.time_alignment_target_clock_domain_id,
        "time_alignment_target_clock_domain_id",
    )?;
    if panel.time_alignment_source_clock_domain_id != panel.reference_time_map_clock_domain_id
        || panel.time_alignment_target_clock_domain_id != panel.clock_domain_id
    {
        return Err("time alignment source/target domains do not match the canonical map and evaluator clock".to_owned());
    }
    if panel.time_alignment_valid_from_ns >= panel.time_alignment_valid_until_ns {
        return Err("time alignment validity interval must have positive duration".to_owned());
    }
    require_digest(
        &panel.reference_time_map_receipt_sha256,
        "canonical time-map receipt digest",
    )?;
    require_digest(
        &panel.time_alignment_receipt_sha256,
        "time alignment receipt digest",
    )?;
    if panel.input_snapshots.is_empty() {
        return Err("input_snapshots must be non-empty".to_owned());
    }
    let mut snapshot_ids = BTreeSet::new();
    let mut previous_snapshot_time = None;
    for snapshot in &panel.input_snapshots {
        require_text(&snapshot.input_snapshot_id, "input_snapshot_id")?;
        require_digest(&snapshot.input_history_sha256, "input snapshot digest")?;
        if !snapshot_ids.insert(snapshot.input_snapshot_id.as_str()) {
            return Err("duplicate input_snapshot_id".to_owned());
        }
        if previous_snapshot_time.is_some_and(|time| snapshot.available_at_ns < time) {
            return Err("input snapshots must be ordered by availability".to_owned());
        }
        previous_snapshot_time = Some(snapshot.available_at_ns);
    }
    if panel.forecasts.is_empty() {
        return Err("forecasts must be non-empty".to_owned());
    }
    if !panel.policy.maximum_onset_error_beats.is_finite()
        || panel.policy.maximum_onset_error_beats <= 0.0
        || panel.policy.minimum_anticipation_lead_ns == 0
    {
        return Err("invalid musical forecast policy".to_owned());
    }
    if panel.claim_ceiling.is_empty()
        || panel
            .claim_ceiling
            .iter()
            .any(|claim| claim.trim().is_empty())
    {
        return Err("claim_ceiling must contain explicit non-empty limits".to_owned());
    }
    Ok(())
}

fn validate_observation_closure_shape(closure: &ObservationClosureV1) -> Result<(), String> {
    require_text(&closure.forecast_id, "observation closure forecast_id")?;
    require_text(&closure.capture_id, "observation closure capture_id")?;
    require_text(&closure.observer_id, "observation closure observer_id")?;
    require_text(&closure.producer_id, "observation closure producer_id")?;
    require_text(
        &closure.event_observer_id,
        "observation closure event_observer_id",
    )?;
    require_text(
        &closure.event_observer_qualification_id,
        "event observer qualification ID",
    )?;
    require_text(
        &closure.clock_domain_id,
        "observation closure clock_domain_id",
    )?;
    for (digest, name) in [
        (&closure.capture_route_sha256, "capture route digest"),
        (&closure.capture_payload_sha256, "capture payload digest"),
        (&closure.observer_build_sha256, "observer build digest"),
        (
            &closure.event_observer_build_sha256,
            "event observer build digest",
        ),
        (
            &closure.event_observer_config_sha256,
            "event observer config digest",
        ),
        (
            &closure.event_observer_qualification_receipt_sha256,
            "event observer qualification receipt digest",
        ),
        (
            &closure.observed_event_manifest_sha256,
            "observed event manifest digest",
        ),
        (
            &closure.event_observer_receipt_sha256,
            "event observer receipt digest",
        ),
        (
            &closure.closure_receipt_sha256,
            "observation closure receipt digest",
        ),
    ] {
        require_digest(digest, name)?;
    }
    if closure.coverage_start_ns >= closure.coverage_end_ns {
        return Err(format!(
            "observation closure {} must have a positive coverage interval",
            closure.capture_id
        ));
    }
    if closure.closed_at_ns < closure.coverage_end_ns
        || closure.terminal_watermark_ns > closure.closed_at_ns
    {
        return Err(format!(
            "observation closure {} has an impossible terminal watermark or close time",
            closure.capture_id
        ));
    }
    let sequence_span = closure
        .end_sequence_exclusive
        .checked_sub(closure.first_sequence)
        .ok_or_else(|| "observation sequence range is reversed".to_owned())?;
    if sequence_span != closure.expected_sequence_count
        || closure
            .observed_sequence_count
            .checked_add(closure.missing_sequence_count)
            != Some(closure.expected_sequence_count)
    {
        return Err(format!(
            "observation closure {} has inconsistent sequence accounting",
            closure.capture_id
        ));
    }

    let duration_ns = closure.coverage_end_ns - closure.coverage_start_ns;
    match closure.source_kind.as_str() {
        "rendered_audio_loopback" => {
            if closure.sample_rate_hz < 8_000
                || closure.channel_count == 0
                || closure.expected_sample_frames == 0
                || closure.heartbeat_interval_ns != 0
                || closure.expected_heartbeat_count != 0
                || closure.observed_heartbeat_count != 0
                || closure.missing_heartbeat_count != 0
            {
                return Err(format!(
                    "audio-loopback closure {} has invalid sample or heartbeat accounting",
                    closure.capture_id
                ));
            }
            let expected_frames =
                (u128::from(duration_ns) * u128::from(closure.sample_rate_hz)) / 1_000_000_000;
            if expected_frames != u128::from(closure.expected_sample_frames)
                || closure
                    .observed_sample_frames
                    .checked_add(closure.dropped_sample_frames)
                    != Some(closure.expected_sample_frames)
            {
                return Err(format!(
                    "audio-loopback closure {} has inconsistent sample-frame accounting",
                    closure.capture_id
                ));
            }
        }
        "midi_output_monitor" => {
            if closure.sample_rate_hz != 0
                || closure.channel_count != 0
                || closure.expected_sample_frames != 0
                || closure.observed_sample_frames != 0
                || closure.dropped_sample_frames != 0
                || closure.xrun_count != 0
                || closure.heartbeat_interval_ns == 0
                || closure.expected_heartbeat_count == 0
                || closure
                    .observed_heartbeat_count
                    .checked_add(closure.missing_heartbeat_count)
                    != Some(closure.expected_heartbeat_count)
            {
                return Err(format!(
                    "MIDI monitor closure {} has invalid heartbeat or sample accounting",
                    closure.capture_id
                ));
            }
            let expected_heartbeats = duration_ns
                .div_ceil(closure.heartbeat_interval_ns)
                .checked_add(1)
                .ok_or_else(|| "MIDI heartbeat schedule count overflows u64".to_owned())?;
            if expected_heartbeats != closure.expected_heartbeat_count {
                return Err(format!(
                    "MIDI monitor closure {} heartbeat schedule does not cover its declared interval",
                    closure.capture_id
                ));
            }
        }
        _ => {
            return Err(format!(
                "unsupported observation source_kind {}",
                closure.source_kind
            ));
        }
    }
    Ok(())
}

fn observation_closure_incompleteness(
    closure: &ObservationClosureV1,
    forecast: &ForecastV1,
    panel: &ForecastPanelV1,
) -> Vec<String> {
    let mut reasons = Vec::new();
    if closure.observer_id == closure.producer_id {
        reasons.push("observer identity is not independent of the output producer".to_owned());
    }
    if closure.event_observer_id == closure.producer_id {
        reasons
            .push("event observer identity is not independent of the output producer".to_owned());
    }
    if let Some(realization) = panel
        .realizations
        .iter()
        .find(|item| item.forecast_id == forecast.forecast_id)
    {
        if realization.observation_capture_id != closure.capture_id {
            reasons.push("realization references a different observation capture".to_owned());
        }
        if realization.observed_event_manifest_sha256 != closure.observed_event_manifest_sha256 {
            reasons
                .push("realization event manifest does not match the observer receipt".to_owned());
        }
    }
    if closure.clock_domain_id != panel.clock_domain_id {
        reasons.push("capture clock domain does not match the evaluator clock domain".to_owned());
    }
    if closure.coverage_start_ns > forecast.target_start_ns {
        reasons.push("capture begins after the forecast target window starts".to_owned());
    }
    if closure.coverage_end_ns < forecast.target_end_ns {
        reasons.push("capture ends before the forecast target window ends".to_owned());
    }
    if closure.terminal_watermark_ns < closure.coverage_end_ns {
        reasons.push("terminal watermark does not close the complete captured interval".to_owned());
    }
    if closure.missing_sequence_count > 0 || closure.duplicate_sequence_count > 0 {
        reasons.push("capture sequence has missing or duplicated records".to_owned());
    }
    if closure.observed_sequence_count != closure.expected_sequence_count {
        reasons.push("capture sequence count does not match the expected count".to_owned());
    }
    match closure.source_kind.as_str() {
        "rendered_audio_loopback" => {
            if closure.observed_sample_frames != closure.expected_sample_frames
                || closure.dropped_sample_frames > 0
                || closure.xrun_count > 0
            {
                reasons.push("audio capture contains missing frames or an xrun".to_owned());
            }
        }
        "midi_output_monitor"
            if closure.missing_heartbeat_count > 0
                || closure.observed_heartbeat_count != closure.expected_heartbeat_count =>
        {
            reasons.push("MIDI monitor heartbeat coverage contains a gap".to_owned());
        }
        "midi_output_monitor" => {}
        _ => {}
    }
    reasons
}

fn validate_forecast(forecast: &ForecastV1, panel: &ForecastPanelV1) -> Result<(), String> {
    require_text(&forecast.forecast_id, "forecast_id")?;
    require_text(&forecast.commitment_series_id, "commitment_series_id")?;
    require_text(&forecast.input_snapshot_id, "input_snapshot_id")?;
    require_digest(
        &forecast.input_history_sha256,
        "forecast input history digest",
    )?;
    let input_snapshot = panel
        .input_snapshots
        .iter()
        .find(|snapshot| snapshot.input_snapshot_id == forecast.input_snapshot_id)
        .ok_or_else(|| {
            format!(
                "forecast {} references unknown or yoked input snapshot",
                forecast.forecast_id
            )
        })?;
    if input_snapshot.input_history_sha256 != forecast.input_history_sha256 {
        return Err(format!(
            "forecast {} input-history digest does not match its snapshot",
            forecast.forecast_id
        ));
    }
    if input_snapshot.available_at_ns > forecast.committed_at_ns {
        return Err(format!(
            "forecast {} uses input not available at commit time",
            forecast.forecast_id
        ));
    }
    require_digest(&forecast.policy_sha256, "forecast policy digest")?;
    if forecast.target_start_ns < panel.time_alignment_valid_from_ns
        || forecast.target_end_ns > panel.time_alignment_valid_until_ns
    {
        return Err(format!(
            "forecast {} target window is outside the time-alignment validity interval",
            forecast.forecast_id
        ));
    }
    if forecast.revision == 0 || forecast.target_start_ns >= forecast.target_end_ns {
        return Err(format!(
            "forecast {} has invalid revision or target window",
            forecast.forecast_id
        ));
    }
    if forecast.expires_at_ns < forecast.target_end_ns {
        return Err(format!(
            "forecast {} expires before its target window ends",
            forecast.forecast_id
        ));
    }
    if forecast.committed_at_ns >= forecast.target_start_ns {
        return Err(format!(
            "forecast {} was committed after its target window began (post-hoc)",
            forecast.forecast_id
        ));
    }
    if forecast
        .target_start_ns
        .saturating_sub(forecast.committed_at_ns)
        < panel.policy.minimum_anticipation_lead_ns
    {
        return Err(format!(
            "forecast {} does not meet the minimum anticipation lead",
            forecast.forecast_id
        ));
    }
    if forecast.events.is_empty() {
        return Err(format!(
            "forecast {} has no predicted events",
            forecast.forecast_id
        ));
    }
    let mut ids = BTreeSet::new();
    for event in &forecast.events {
        require_text(&event.forecast_event_id, "forecast_event_id")?;
        require_text(&event.role, "predicted event role")?;
        if !ids.insert(event.forecast_event_id.as_str()) {
            return Err(format!(
                "forecast {} contains duplicate forecast_event_id",
                forecast.forecast_id
            ));
        }
        if event.expected_onset_ns < forecast.target_start_ns
            || event.expected_onset_ns >= forecast.target_end_ns
            || event.local_beat_period_ns == 0
            || !event.presence_probability.is_finite()
            || !(0.0..=1.0).contains(&event.presence_probability)
        {
            return Err(format!(
                "forecast event {} has invalid time, beat period or probability",
                event.forecast_event_id
            ));
        }
    }
    Ok(())
}

fn validate_series(
    series: &mut BTreeMap<&str, Vec<&ForecastV1>>,
    forecast_ids: &BTreeMap<&str, &ForecastV1>,
) -> Result<Vec<ForecastSeriesSummary>, String> {
    let mut summaries = Vec::new();
    for (series_id, members) in series.iter_mut() {
        members.sort_by_key(|forecast| forecast.revision);
        let mut prior: Option<&ForecastV1> = None;
        for (index, forecast) in members.iter().enumerate() {
            if index == 0 {
                if forecast.revision != 1 || forecast.supersedes_forecast_id.is_some() {
                    return Err(format!(
                        "series {series_id} must start at revision 1 without a superseded forecast"
                    ));
                }
            } else {
                let previous = prior.expect("previous revision is present");
                if forecast.revision != previous.revision + 1
                    || forecast.supersedes_forecast_id.as_deref()
                        != Some(previous.forecast_id.as_str())
                {
                    return Err(format!(
                        "series {series_id} has a forked, skipped or invalid supersession chain"
                    ));
                }
                if forecast.committed_at_ns <= previous.committed_at_ns {
                    return Err(format!(
                        "series {series_id} revisions must be committed in increasing time"
                    ));
                }
                if forecast.target_start_ns != previous.target_start_ns
                    || forecast.target_end_ns != previous.target_end_ns
                {
                    return Err(format!(
                        "series {series_id} changed its target window across revisions"
                    ));
                }
                if forecast.policy_sha256 != previous.policy_sha256 {
                    return Err(format!(
                        "series {series_id} changed policy identity across revisions"
                    ));
                }
            }
            if let Some(parent_id) = forecast.supersedes_forecast_id.as_deref() {
                let parent = forecast_ids.get(parent_id).copied().ok_or_else(|| {
                    format!(
                        "forecast {} supersedes unknown forecast {parent_id}",
                        forecast.forecast_id
                    )
                })?;
                if parent.commitment_series_id != forecast.commitment_series_id {
                    return Err(format!(
                        "forecast {} supersedes a forecast from another series",
                        forecast.forecast_id
                    ));
                }
            }
            prior = Some(forecast);
        }
        let latest = members.last().expect("series is non-empty");
        summaries.push(ForecastSeriesSummary {
            commitment_series_id: (*series_id).to_owned(),
            revision_count: members.len(),
            latest_forecast_id: latest.forecast_id.clone(),
            has_supersession_chain: members.len() > 1,
        });
    }
    Ok(summaries)
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
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(format!(
            "{name} must be a lowercase 64-character SHA-256 digest"
        ))
    }
}

fn mean(total: f64, count: usize) -> Option<f64> {
    if count == 0 {
        None
    } else {
        Some(total / count as f64)
    }
}

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn percentile(values: &[f64], percentile: usize) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let rank = (values.len() * percentile).div_ceil(100).max(1);
    values.get(rank - 1).copied()
}

fn percentile_u64(values: &mut [u64], percentile: usize) -> Option<u64> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable();
    let rank = (values.len() * percentile).div_ceil(100).max(1);
    values.get(rank - 1).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const FIXTURE: &str = include_str!("../fixtures/valid-musical-forecast-v2.json");

    fn panel() -> Value {
        serde_json::from_str(FIXTURE).expect("known-answer forecast fixture")
    }

    fn audit(value: &Value) -> Result<ForecastAuditReport, String> {
        parse_and_audit(&value.to_string())
    }

    #[test]
    fn known_answer_forecast_is_visible_before_realization_and_well_calibrated() {
        let report = parse_and_audit(FIXTURE).expect("forecast fixture should validate");
        assert!(report.structurally_valid_anticipatory_trace);
        assert_eq!(report.forecast_count, 1);
        assert_eq!(
            report.reference_time_map_clock_domain_id,
            "muse-performance-time-v1"
        );
        assert_eq!(report.displayed_before_target_count, 1);
        assert_eq!(report.producer_reported_event_record_count, 1);
        assert_eq!(report.qualified_observed_event_record_count, 1);
        assert_eq!(report.matched_forecast_event_count, 1);
        assert_eq!(report.observation_closure_count, 1);
        assert_eq!(report.complete_observation_closure_count, 1);
        assert_eq!(report.incomplete_observation_closure_count, 0);
        assert!(report.output_observation_structurally_eligible);
        assert_eq!(report.unobserved_forecast_event_count, 0);
        assert_eq!(report.presence_brier_sample_count, 1);
        assert_eq!(report.event_findings[0].presence_observed, Some(true));
        assert_eq!(report.role_match_rate, Some(1.0));
        assert_eq!(report.beat_index_match_rate, Some(1.0));
        assert!((report.conditional_presence_brier_score.unwrap() - 0.0025).abs() < 1e-9);
        assert!(report.within_onset_tolerance_rate.unwrap() >= 1.0);
    }

    #[test]
    fn time_alignment_receipt_is_required_to_compare_score_time_with_runner_time() {
        let mut value = panel();
        value["time_alignment_receipt_sha256"] = serde_json::json!("invalid");
        let error = audit(&value).expect_err("clock-domain alignment receipt is required");
        assert!(error.contains("time alignment receipt digest"));
    }

    #[test]
    fn post_hoc_forecast_is_rejected() {
        let mut value = panel();
        value["forecasts"][0]["committed_at_ns"] = serde_json::json!(1_010_000_000_u64);
        let error = audit(&value).expect_err("post-hoc forecast must not qualify");
        assert!(error.contains("post-hoc"));
    }

    #[test]
    fn yoked_input_history_is_rejected() {
        let mut value = panel();
        value["forecasts"][0]["input_history_sha256"] =
            serde_json::json!("9999999999999999999999999999999999999999999999999999999999999999");
        let error = audit(&value).expect_err("yoked forecast must not appear contingent");
        assert!(error.contains("does not match its snapshot"));
    }

    #[test]
    fn time_alignment_domain_mismatch_is_rejected() {
        let mut value = panel();
        value["time_alignment_target_clock_domain_id"] = serde_json::json!("another-runner-clock");
        let error = audit(&value).expect_err("clock domains must be bound explicitly");
        assert!(error.contains("source/target domains"));
    }

    #[test]
    fn forecast_window_must_fit_time_alignment_validity() {
        let mut value = panel();
        value["time_alignment_valid_until_ns"] = serde_json::json!(900_000_000_u64);
        let error = audit(&value).expect_err("forecast may not outlive alignment fit");
        assert!(error.contains("validity interval"));
    }

    #[test]
    fn alignment_uncertainty_is_added_to_worst_case_phase_error() {
        let mut value = panel();
        value["time_alignment_max_uncertainty_ns"] = serde_json::json!(40_000_000_u64);
        let report = audit(&value).expect("uncertainty should produce a scored finding");
        assert!(!report.structurally_valid_anticipatory_trace);
        assert_eq!(report.event_findings[0].within_onset_tolerance, Some(false));
        assert!(
            (report.event_findings[0]
                .worst_case_absolute_onset_error_beats
                .unwrap()
                - 0.12)
                .abs()
                < 1e-9
        );
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|finding| finding.contains("clock-alignment uncertainty"))
        );
    }

    #[test]
    fn forecast_cannot_cite_an_input_snapshot_from_the_future() {
        let mut value = panel();
        value["input_snapshots"][0]["available_at_ns"] = serde_json::json!(600_000_000_u64);
        let error = audit(&value).expect_err("future input cannot support an earlier forecast");
        assert!(error.contains("not available at commit time"));
    }

    #[test]
    fn stale_forecast_realization_is_not_qualified() {
        let mut value = panel();
        value["forecasts"][0]["expires_at_ns"] = serde_json::json!(1_100_000_000_u64);
        value["forecasts"][0]["target_end_ns"] = serde_json::json!(1_100_000_000_u64);
        value["realizations"][0]["started_at_ns"] = serde_json::json!(1_200_000_000_u64);
        value["realizations"][0]["completed_at_ns"] = serde_json::json!(1_230_000_000_u64);
        value["realizations"][0]["events"][0]["onset_ns"] = serde_json::json!(1_220_000_000_u64);
        let report = audit(&value).expect("stale output remains reportable");
        assert!(!report.structurally_valid_anticipatory_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|finding| finding.contains("stale forecast"))
        );
    }

    #[test]
    fn display_after_target_start_is_rejected() {
        let mut value = panel();
        value["displays"][0]["displayed_at_ns"] = serde_json::json!(1_000_000_001_u64);
        let report = audit(&value).expect("late display should produce an audit finding");
        assert!(!report.structurally_valid_anticipatory_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|x| x.contains("displayed too late"))
        );
    }

    #[test]
    fn revision_history_must_be_contiguous_and_immutable() {
        let mut value = panel();
        let mut second = value["forecasts"][0].clone();
        second["forecast_id"] = serde_json::json!("forecast-2");
        second["revision"] = serde_json::json!(2);
        second["supersedes_forecast_id"] = serde_json::json!("forecast-001");
        second["input_snapshot_id"] = serde_json::json!("snapshot-2");
        second["input_history_sha256"] =
            serde_json::json!("9999999999999999999999999999999999999999999999999999999999999999");
        second["committed_at_ns"] = serde_json::json!(900_000_000_u64);
        value["input_snapshots"].as_array_mut().unwrap().push(serde_json::json!({
            "input_snapshot_id":"snapshot-2",
            "available_at_ns":850_000_000_u64,
            "input_history_sha256":"9999999999999999999999999999999999999999999999999999999999999999"
        }));
        value["forecasts"].as_array_mut().unwrap().push(second);
        let report = audit(&value).expect("superseded realization remains reportable");
        assert!(!report.structurally_valid_anticipatory_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|finding| finding.contains("superseded forecast"))
        );
    }

    #[test]
    fn steer_after_successor_commit_is_stale_and_unqualified() {
        let mut value = panel();
        let mut second = value["forecasts"][0].clone();
        second["forecast_id"] = serde_json::json!("forecast-2");
        second["revision"] = serde_json::json!(2);
        second["supersedes_forecast_id"] = serde_json::json!("forecast-001");
        second["input_snapshot_id"] = serde_json::json!("snapshot-2");
        second["input_history_sha256"] =
            serde_json::json!("9999999999999999999999999999999999999999999999999999999999999999");
        second["committed_at_ns"] = serde_json::json!(900_000_000_u64);
        value["input_snapshots"].as_array_mut().unwrap().push(serde_json::json!({
            "input_snapshot_id":"snapshot-2",
            "available_at_ns":850_000_000_u64,
            "input_history_sha256":"9999999999999999999999999999999999999999999999999999999999999999"
        }));
        value["forecasts"].as_array_mut().unwrap().push(second);
        value["dispositions"][0]["action"] = serde_json::json!("steer");
        value["dispositions"][0]["acted_at_ns"] = serde_json::json!(950_000_000_u64);
        let report = audit(&value).expect("late steer remains reportable");
        assert!(!report.structurally_valid_anticipatory_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|finding| finding.contains("successor was already committed"))
        );
    }

    #[test]
    fn realization_after_target_window_is_rejected_even_before_expiry() {
        let mut value = panel();
        value["realizations"][0]["started_at_ns"] = serde_json::json!(1_550_000_000_u64);
        value["realizations"][0]["completed_at_ns"] = serde_json::json!(1_560_000_000_u64);
        value["realizations"][0]["events"][0]["onset_ns"] = serde_json::json!(1_555_000_000_u64);
        let report = audit(&value).expect("late realization remains reportable");
        assert!(!report.structurally_valid_anticipatory_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|finding| finding.contains("target window ended"))
        );
    }

    #[test]
    fn vetoed_forecast_cannot_produce_output() {
        let mut value = panel();
        value["dispositions"][0]["action"] = serde_json::json!("veto");
        let report = audit(&value).expect("veto violation is an audit finding");
        assert!(!report.structurally_valid_anticipatory_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|x| x.contains("vetoed forecast"))
        );
    }

    #[test]
    fn user_deviation_is_reported_but_not_scored_as_disobedience() {
        let mut value = panel();
        value["observation_closures"][0]["observed_events"][0]["forecast_event_id"] =
            serde_json::Value::Null;
        value["realizations"][0]["events"] = serde_json::json!([]);
        let report = audit(&value).expect("unforecasted contribution is allowed");
        assert!(report.structurally_valid_anticipatory_trace);
        assert_eq!(report.unforecasted_observed_event_count, 1);
        assert_eq!(report.qualified_observed_event_record_count, 1);
        assert_eq!(report.matched_forecast_event_count, 0);
    }

    #[test]
    fn steer_requires_a_new_superseding_forecast() {
        let mut value = panel();
        value["dispositions"][0]["action"] = serde_json::json!("steer");
        let report = audit(&value).expect("missing steer revision remains reportable");
        assert!(!report.structurally_valid_anticipatory_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|x| x.contains("steer disposition"))
        );
    }

    #[test]
    fn duplicate_forecast_event_realization_is_rejected() {
        let mut value = panel();
        let extra = value["realizations"][0]["events"][0].clone();
        value["realizations"][0]["events"]
            .as_array_mut()
            .unwrap()
            .push(extra);
        let error = audit(&value).expect_err("forecast event cannot be used twice");
        assert!(error.contains("duplicate producer event_id"));
    }

    #[test]
    fn off_beat_realization_is_reported_even_when_onset_is_close() {
        let mut value = panel();
        value["observation_closures"][0]["observed_events"][0]["beat_index"] = serde_json::json!(5);
        let report = audit(&value).expect("off-beat structure should be reported");
        assert!(report.structurally_valid_anticipatory_trace);
        assert_eq!(report.beat_index_match_rate, Some(0.0));
        assert_eq!(report.absolute_onset_error_beats, vec![0.04]);
        assert_eq!(report.event_findings[0].beat_index_match, Some(false));
    }

    #[test]
    fn low_presence_probability_penalizes_a_realized_event() {
        let mut value = panel();
        value["forecasts"][0]["events"][0]["presence_probability"] = serde_json::json!(0.2);
        let report = audit(&value).expect("low-confidence but correct forecast is reportable");
        assert!((report.conditional_presence_brier_score.unwrap() - 0.64).abs() < 1e-9);
    }

    #[test]
    fn incomplete_capture_makes_absence_unknown_not_false() {
        let mut value = panel();
        value["observation_closures"][0]["coverage_end_ns"] = serde_json::json!(1_400_000_000_u64);
        value["observation_closures"][0]["terminal_watermark_ns"] =
            serde_json::json!(1_400_000_000_u64);
        value["observation_closures"][0]["closed_at_ns"] = serde_json::json!(1_410_000_000_u64);
        value["observation_closures"][0]["expected_sample_frames"] = serde_json::json!(19_680_u64);
        value["observation_closures"][0]["observed_sample_frames"] = serde_json::json!(19_680_u64);
        let report = audit(&value).expect("partial capture remains reportable");
        assert_eq!(report.event_findings[0].presence_observed, None);
        assert_eq!(report.event_findings[0].presence_brier_score, None);
        assert_eq!(report.matched_forecast_event_count, 0);
        assert_eq!(report.producer_reported_event_record_count, 1);
        assert_eq!(report.observed_event_record_count, 1);
        assert_eq!(report.qualified_observed_event_record_count, 0);
        assert_eq!(report.unfulfilled_forecast_event_count, 0);
        assert_eq!(report.unobserved_forecast_event_count, 1);
        assert_eq!(report.presence_brier_sample_count, 0);
        assert_eq!(report.conditional_presence_brier_score, None);
        assert!(!report.structurally_valid_anticipatory_trace);
    }

    #[test]
    fn missing_capture_receipt_does_not_turn_a_missing_event_into_a_negative() {
        let mut value = panel();
        value["observation_closures"] = serde_json::json!([]);
        let report = audit(&value).expect("missing observation remains reportable");
        assert_eq!(report.forecasts_without_observation_closure, 1);
        assert!(!report.output_observation_structurally_eligible);
        assert_eq!(report.event_findings[0].presence_observed, None);
        assert_eq!(report.unobserved_forecast_event_count, 1);
        assert_eq!(report.unfulfilled_forecast_event_count, 0);
        assert_eq!(report.conditional_presence_brier_score, None);
    }

    #[test]
    fn complete_capture_without_observer_event_scores_absence_despite_producer_claim() {
        let mut value = panel();
        value["observation_closures"][0]["observed_events"] = serde_json::json!([]);
        let report = audit(&value).expect("closed complete capture should score absence");
        assert_eq!(report.event_findings[0].presence_observed, Some(false));
        assert!((report.event_findings[0].presence_brier_score.unwrap() - 0.9025).abs() < 1e-9);
        assert_eq!(report.unfulfilled_forecast_event_count, 1);
        assert_eq!(report.unobserved_forecast_event_count, 0);
    }

    #[test]
    fn dropped_audio_frames_make_the_observation_window_incomplete() {
        let mut value = panel();
        value["observation_closures"][0]["observed_sample_frames"] = serde_json::json!(24_959_u64);
        value["observation_closures"][0]["dropped_sample_frames"] = serde_json::json!(1_u64);
        let report = audit(&value).expect("dropped audio frame is reportable");
        assert_eq!(report.event_findings[0].presence_observed, None);
        assert_eq!(report.event_findings[0].presence_brier_score, None);
        assert_eq!(report.incomplete_observation_closure_count, 1);
        assert!(!report.output_observation_structurally_eligible);
    }

    #[test]
    fn missing_midi_monitor_heartbeat_makes_absence_unobservable() {
        let mut value = panel();
        let closure = &mut value["observation_closures"][0];
        closure["source_kind"] = serde_json::json!("midi_output_monitor");
        closure["sample_rate_hz"] = serde_json::json!(0);
        closure["channel_count"] = serde_json::json!(0);
        closure["expected_sample_frames"] = serde_json::json!(0);
        closure["observed_sample_frames"] = serde_json::json!(0);
        closure["dropped_sample_frames"] = serde_json::json!(0);
        closure["xrun_count"] = serde_json::json!(0);
        closure["heartbeat_interval_ns"] = serde_json::json!(10_000_000_u64);
        closure["expected_heartbeat_count"] = serde_json::json!(53_u64);
        closure["observed_heartbeat_count"] = serde_json::json!(52_u64);
        closure["missing_heartbeat_count"] = serde_json::json!(1_u64);
        closure["expected_sequence_count"] = serde_json::json!(0);
        closure["observed_sequence_count"] = serde_json::json!(0);
        closure["missing_sequence_count"] = serde_json::json!(0);
        closure["duplicate_sequence_count"] = serde_json::json!(0);
        closure["first_sequence"] = serde_json::json!(0);
        closure["end_sequence_exclusive"] = serde_json::json!(0);
        let report = audit(&value).expect("heartbeat gap is a qualification finding");
        assert_eq!(report.event_findings[0].presence_observed, None);
        assert_eq!(report.unobserved_forecast_event_count, 1);
    }

    #[test]
    fn producer_cannot_certify_its_own_output_observation() {
        let mut value = panel();
        value["observation_closures"][0]["observer_id"] =
            value["observation_closures"][0]["producer_id"].clone();
        let report = audit(&value).expect("common-mode observer is reportable but ineligible");
        assert_eq!(report.event_findings[0].presence_observed, None);
        assert!(!report.structurally_valid_anticipatory_trace);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|finding| finding.contains("not independent"))
        );
    }

    #[test]
    fn closure_receipt_digest_is_required() {
        let mut value = panel();
        value["observation_closures"][0]["closure_receipt_sha256"] = serde_json::json!("bad");
        let error = audit(&value).expect_err("malformed closure receipts must be rejected");
        assert!(error.contains("observation closure receipt digest"));
    }

    #[test]
    fn duplicate_observation_closures_for_one_forecast_are_rejected() {
        let mut value = panel();
        let duplicate = value["observation_closures"][0].clone();
        value["observation_closures"]
            .as_array_mut()
            .unwrap()
            .push(duplicate);
        let error = audit(&value).expect_err("one forecast may have only one closure decision");
        assert!(error.contains("more than one observation closure"));
    }

    #[test]
    fn mismatched_event_manifest_cannot_establish_presence_or_absence() {
        let mut value = panel();
        value["realizations"][0]["observed_event_manifest_sha256"] =
            serde_json::json!("9999999999999999999999999999999999999999999999999999999999999999");
        let report = audit(&value).expect("unbound event manifest remains reportable");
        assert_eq!(report.event_findings[0].presence_observed, None);
        assert_eq!(report.event_findings[0].presence_brier_score, None);
        assert_eq!(report.unobserved_forecast_event_count, 1);
        assert!(!report.output_observation_structurally_eligible);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|finding| finding.contains("event manifest does not match"))
        );
    }

    #[test]
    fn missing_event_observer_receipt_cannot_establish_absence() {
        let mut value = panel();
        value["observation_closures"][0]["event_observer_qualification_id"] = serde_json::json!("");
        let error = audit(&value).expect_err("missing observer qualification reference must fail");
        assert!(error.contains("event observer qualification ID"));
    }

    #[test]
    fn producer_cannot_claim_to_be_independent_event_detector() {
        let mut value = panel();
        value["observation_closures"][0]["event_observer_id"] =
            value["observation_closures"][0]["producer_id"].clone();
        let report =
            audit(&value).expect("producer detector claim remains reportable but ineligible");
        assert_eq!(report.event_findings[0].presence_observed, None);
        assert!(!report.output_observation_structurally_eligible);
        assert!(
            report
                .qualification_findings
                .iter()
                .any(|finding| finding.contains("event observer identity is not independent"))
        );
    }

    #[test]
    fn complete_capture_manifest_is_sufficient_without_producer_realization() {
        let mut value = panel();
        value["realizations"] = serde_json::json!([]);
        let report = audit(&value).expect("independent observer evidence stands alone");
        assert!(report.output_observation_structurally_eligible);
        assert_eq!(report.event_findings[0].presence_observed, Some(true));
        assert!((report.event_findings[0].presence_brier_score.unwrap() - 0.0025).abs() < 1e-9);
        assert_eq!(report.producer_reported_event_record_count, 0);
        assert_eq!(report.observed_event_record_count, 1);
        assert_eq!(report.qualified_observed_event_record_count, 1);
        assert_eq!(report.unobserved_forecast_event_count, 0);
        assert!((report.conditional_presence_brier_score.unwrap() - 0.0025).abs() < 1e-9);
    }

    #[test]
    fn complete_empty_capture_manifest_scores_absence_without_producer_realization() {
        let mut value = panel();
        value["realizations"] = serde_json::json!([]);
        value["observation_closures"][0]["observed_events"] = serde_json::json!([]);
        let report =
            audit(&value).expect("complete empty observer manifest is evidence of absence");
        assert!(report.output_observation_structurally_eligible);
        assert_eq!(report.event_findings[0].presence_observed, Some(false));
        assert!((report.event_findings[0].presence_brier_score.unwrap() - 0.9025).abs() < 1e-9);
        assert_eq!(report.producer_reported_event_record_count, 0);
        assert_eq!(report.observed_event_record_count, 0);
        assert_eq!(report.qualified_observed_event_record_count, 0);
        assert_eq!(report.unfulfilled_forecast_event_count, 1);
        assert_eq!(report.unobserved_forecast_event_count, 0);
    }

    #[test]
    fn independent_observer_manifest_scores_event_without_producer_event_claim() {
        let mut value = panel();
        value["realizations"][0]["events"] = serde_json::json!([]);
        let report = audit(&value).expect("independent observation should be scored");
        assert_eq!(report.event_findings[0].presence_observed, Some(true));
        assert_eq!(report.producer_reported_event_record_count, 0);
        assert_eq!(report.observed_event_record_count, 1);
        assert_eq!(report.qualified_observed_event_record_count, 1);
        assert_eq!(report.matched_forecast_event_count, 1);
    }

    #[test]
    fn independent_event_detector_may_differ_from_capture_recorder() {
        let mut value = panel();
        value["observation_closures"][0]["event_observer_id"] =
            serde_json::json!("separate-independent-event-detector-v1");
        let report = audit(&value).expect("distinct independent event detector is valid");
        assert!(report.output_observation_structurally_eligible);
        assert_eq!(report.event_findings[0].presence_observed, Some(true));
    }
}
