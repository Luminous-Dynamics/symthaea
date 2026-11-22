use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

const SCHEMA_VERSION: &str = "mel-musical-forecast-panel-v1";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ForecastPanelV1 {
    schema_version: String,
    session_id: String,
    case_id: String,
    clock_domain_id: String,
    input_history_sha256: String,
    policy: ForecastPolicyV1,
    forecasts: Vec<ForecastV1>,
    displays: Vec<ForecastDisplayV1>,
    dispositions: Vec<ForecastDispositionV1>,
    realizations: Vec<ForecastRealizationV1>,
    claim_ceiling: Vec<String>,
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
    events: Vec<RealizedEventV1>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RealizedEventV1 {
    event_id: String,
    forecast_event_id: Option<String>,
    role: String,
    beat_index: u64,
    onset_ns: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct EventForecastFinding {
    pub forecast_id: String,
    pub forecast_event_id: String,
    pub realized_event_id: Option<String>,
    pub predicted_role: String,
    pub realized_role: Option<String>,
    pub expected_onset_ns: u64,
    pub observed_onset_ns: Option<u64>,
    pub signed_onset_error_ns: Option<i128>,
    pub absolute_onset_error_beats: Option<f64>,
    pub within_onset_tolerance: Option<bool>,
    pub presence_probability: f64,
    pub presence_observed: bool,
    pub presence_brier_score: f64,
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
    pub forecast_count: usize,
    pub forecast_event_count: usize,
    pub display_count: usize,
    pub displayed_before_target_count: usize,
    pub forecasts_without_display: usize,
    pub disposition_count: usize,
    pub realization_count: usize,
    pub realized_event_count: usize,
    pub matched_forecast_event_count: usize,
    pub unfulfilled_forecast_event_count: usize,
    pub unforecasted_realized_event_count: usize,
    pub role_match_rate: Option<f64>,
    pub conditional_presence_brier_score: Option<f64>,
    pub absolute_onset_error_beats: Vec<f64>,
    pub median_absolute_onset_error_beats: Option<f64>,
    pub p95_absolute_onset_error_beats: Option<f64>,
    pub within_onset_tolerance_rate: Option<f64>,
    pub median_anticipation_lead_ns: Option<u64>,
    pub series: Vec<ForecastSeriesSummary>,
    pub qualified_for_anticipatory_use: bool,
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
                .insert((forecast.forecast_id.as_str(), event.forecast_event_id.as_str()), event)
                .is_some()
            {
                return Err(format!(
                    "duplicate forecast_event_id {} in forecast {}",
                    event.forecast_event_id, forecast.forecast_id
                ));
            }
        }
    }

    let series = validate_series(&mut series_members, &forecast_ids, &mut findings)?;
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
            .ok_or_else(|| format!("display references unknown forecast {}", display.forecast_id))?;
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
            return Err(format!("duplicate display receipt for forecast {}", display.forecast_id));
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
            return Err(format!("unsupported forecast disposition {}", disposition.action));
        }
    }

    for forecast in &panel.forecasts {
        let has_child = series_members
            .get(forecast.commitment_series_id.as_str())
            .is_some_and(|members| {
                members.iter().any(|candidate| {
                    candidate.supersedes_forecast_id.as_deref()
                        == Some(forecast.forecast_id.as_str())
                })
            });
        if let Some(disposition) = dispositions_by_forecast.get(forecast.forecast_id.as_str()) {
            if matches!(disposition.action.as_str(), "steer" | "branch") && !has_child {
                findings.push(format!(
                    "{} disposition for {} has no superseding forecast",
                    disposition.action, forecast.forecast_id
                ));
            }
        }
    }

    let mut realization_ids = BTreeSet::new();
    let mut realized_event_ids = BTreeSet::new();
    let mut realized_by_forecast_event =
        BTreeMap::<(&str, &str), (&ForecastRealizationV1, &RealizedEventV1)>::new();
    let mut unforecasted_realized_event_count = 0usize;
    let mut realized_event_count = 0usize;

    for realization in &panel.realizations {
        if !realization_ids.insert(realization.realization_id.as_str()) {
            return Err(format!("duplicate realization_id: {}", realization.realization_id));
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
        if !latest_by_series
            .get(forecast.commitment_series_id.as_str())
            .is_some_and(|latest| latest.forecast_id == forecast.forecast_id)
            && realization.started_at_ns
                >= latest_by_series[forecast.commitment_series_id.as_str()].committed_at_ns
        {
            findings.push(format!(
                "realization {} uses a non-current forecast version",
                realization.realization_id
            ));
        }

        for event in &realization.events {
            require_text(&event.event_id, "realized event_id")?;
            require_text(&event.role, "realized event role")?;
            if !realized_event_ids.insert(event.event_id.as_str()) {
                return Err(format!("duplicate realized event_id: {}", event.event_id));
            }
            realized_event_count += 1;
            match event.forecast_event_id.as_deref() {
                Some(forecast_event_id) => {
                    if !forecast_events.contains_key(&(forecast.forecast_id.as_str(), forecast_event_id)) {
                        return Err(format!(
                            "realized event {} references unknown forecast event {}",
                            event.event_id, forecast_event_id
                        ));
                    }
                    if realized_by_forecast_event
                        .insert(
                            (forecast.forecast_id.as_str(), forecast_event_id),
                            (realization, event),
                        )
                        .is_some()
                    {
                        return Err(format!(
                            "forecast event {} was realized more than once",
                            forecast_event_id
                        ));
                    }
                }
                None => unforecasted_realized_event_count += 1,
            }
        }
    }

    for disposition in &panel.dispositions {
        if disposition.action == "veto"
            && panel.realizations.iter().any(|realization| {
                realization.forecast_id == disposition.forecast_id
                    && !realization.events.is_empty()
            })
        {
            findings.push(format!(
                "vetoed forecast {} has output events",
                disposition.forecast_id
            ));
        }
    }

    let mut event_findings = Vec::new();
    let mut brier_sum = 0.0;
    let mut brier_count = 0usize;
    let mut role_matches = 0usize;
    let mut matched_count = 0usize;
    let mut unfulfilled_count = 0usize;
    let mut onset_errors = Vec::new();

    for forecast in &panel.forecasts {
        for predicted in &forecast.events {
            let realized = realized_by_forecast_event
                .get(&(forecast.forecast_id.as_str(), predicted.forecast_event_id.as_str()));
            let (realized_event_id, realized_role, observed_onset_ns, signed_error_ns, phase_error) =
                if let Some((_, actual)) = realized {
                    let signed = i128::from(actual.onset_ns) - i128::from(predicted.expected_onset_ns);
                    let phase = signed as f64 / predicted.local_beat_period_ns as f64;
                    role_matches += usize::from(actual.role == predicted.role);
                    matched_count += 1;
                    onset_errors.push(phase.abs());
                    (
                        Some(actual.event_id.clone()),
                        Some(actual.role.clone()),
                        Some(actual.onset_ns),
                        Some(signed),
                        Some(phase),
                    )
                } else {
                    unfulfilled_count += 1;
                    (None, None, None, None, None)
                };
            let presence_observed = realized.is_some();
            let brier = (predicted.presence_probability - f64::from(presence_observed)).powi(2);
            brier_sum += brier;
            brier_count += 1;
            let within = phase_error.map(|value| value.abs() <= panel.policy.maximum_onset_error_beats);
            event_findings.push(EventForecastFinding {
                forecast_id: forecast.forecast_id.clone(),
                forecast_event_id: predicted.forecast_event_id.clone(),
                realized_event_id,
                predicted_role: predicted.role.clone(),
                realized_role,
                expected_onset_ns: predicted.expected_onset_ns,
                observed_onset_ns,
                signed_onset_error_ns: signed_error_ns,
                absolute_onset_error_beats: phase_error.map(f64::abs),
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

    let mut absolute_sorted = onset_errors.clone();
    absolute_sorted.sort_by(f64::total_cmp);
    let within_count = event_findings
        .iter()
        .filter(|finding| finding.within_onset_tolerance == Some(true))
        .count();
    let onset_observed_count = matched_count;
    let series_summaries = series_members
        .iter()
        .filter_map(|(series_id, members)| {
            let latest = members.iter().max_by_key(|forecast| forecast.revision)?;
            Some(ForecastSeriesSummary {
                commitment_series_id: (*series_id).to_owned(),
                revision_count: members.len(),
                latest_forecast_id: latest.forecast_id.clone(),
                has_supersession_chain: members.len() > 1,
            })
        })
        .collect();

    let qualified = findings.is_empty();
    Ok(ForecastAuditReport {
        schema_version: "mel-musical-forecast-report-v1".to_owned(),
        session_id: panel.session_id.clone(),
        case_id: panel.case_id.clone(),
        forecast_count: panel.forecasts.len(),
        forecast_event_count: forecast_events.len(),
        display_count: panel.displays.len(),
        displayed_before_target_count,
        forecasts_without_display,
        disposition_count: panel.dispositions.len(),
        realization_count: panel.realizations.len(),
        realized_event_count,
        matched_forecast_event_count: matched_count,
        unfulfilled_forecast_event_count: unfulfilled_count,
        unforecasted_realized_event_count,
        role_match_rate: rate(role_matches, matched_count),
        conditional_presence_brier_score: mean(brier_sum, brier_count),
        absolute_onset_error_beats: absolute_sorted,
        median_absolute_onset_error_beats: percentile(&absolute_sorted, 50),
        p95_absolute_onset_error_beats: percentile(&absolute_sorted, 95),
        within_onset_tolerance_rate: rate(within_count, onset_observed_count),
        median_anticipation_lead_ns: percentile_u64(&mut lead_times, 50),
        series: series_summaries,
        qualified_for_anticipatory_use: qualified,
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
    require_digest(&panel.input_history_sha256, "input history digest")?;
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
        || panel.claim_ceiling.iter().any(|claim| claim.trim().is_empty())
    {
        return Err("claim_ceiling must contain explicit non-empty limits".to_owned());
    }
    Ok(())
}

fn validate_forecast(forecast: &ForecastV1, panel: &ForecastPanelV1) -> Result<(), String> {
    require_text(&forecast.forecast_id, "forecast_id")?;
    require_text(&forecast.commitment_series_id, "commitment_series_id")?;
    require_digest(&forecast.input_history_sha256, "forecast input history digest")?;
    require_digest(&forecast.policy_sha256, "forecast policy digest")?;
    if forecast.revision == 0 || forecast.target_start_ns >= forecast.target_end_ns {
        return Err(format!("forecast {} has invalid revision or target window", forecast.forecast_id));
    }
    if forecast.expires_at_ns < forecast.target_end_ns {
        return Err(format!("forecast {} expires before its target window ends", forecast.forecast_id));
    }
    if forecast.input_history_sha256 != panel.input_history_sha256 {
        return Err(format!(
            "forecast {} is not bound to the active session input history (yoked or mismatched preview)",
            forecast.forecast_id
        ));
    }
    if forecast.committed_at_ns >= forecast.target_start_ns {
        return Err(format!(
            "forecast {} was committed after its target window began (post-hoc)",
            forecast.forecast_id
        ));
    }
    if forecast.target_start_ns.saturating_sub(forecast.committed_at_ns)
        < panel.policy.minimum_anticipation_lead_ns
    {
        return Err(format!(
            "forecast {} does not meet the minimum anticipation lead",
            forecast.forecast_id
        ));
    }
    if forecast.events.is_empty() {
        return Err(format!("forecast {} has no predicted events", forecast.forecast_id));
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
    findings: &mut Vec<String>,
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
                if forecast.input_history_sha256 != previous.input_history_sha256 {
                    return Err(format!(
                        "series {series_id} changed input-history identity across revisions"
                    ));
                }
            }
            if let Some(parent_id) = forecast.supersedes_forecast_id.as_deref() {
                let parent = forecast_ids.get(parent_id).copied().ok_or_else(|| {
                    format!("forecast {} supersedes unknown forecast {parent_id}", forecast.forecast_id)
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
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(format!("{name} must be a lowercase 64-character SHA-256 digest"))
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

    const FIXTURE: &str = include_str!("../fixtures/valid-musical-forecast-v1.json");

    fn panel() -> Value {
        serde_json::from_str(FIXTURE).expect("known-answer forecast fixture")
    }

    fn audit(value: &Value) -> Result<ForecastAuditReport, String> {
        parse_and_audit(&value.to_string())
    }

    #[test]
    fn known_answer_forecast_is_visible_before_realization_and_well_calibrated() {
        let report = parse_and_audit(FIXTURE).expect("forecast fixture should validate");
        assert!(report.qualified_for_anticipatory_use);
        assert_eq!(report.forecast_count, 1);
        assert_eq!(report.displayed_before_target_count, 1);
        assert_eq!(report.matched_forecast_event_count, 1);
        assert_eq!(report.role_match_rate, Some(1.0));
        assert!((report.conditional_presence_brier_score.unwrap() - 0.0025).abs() < 1e-9);
        assert!(report.within_onset_tolerance_rate.unwrap() >= 1.0);
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
        value["forecasts"][0]["input_history_sha256"] = serde_json::json!("9999999999999999999999999999999999999999999999999999999999999999");
        let error = audit(&value).expect_err("yoked forecast must not appear contingent");
        assert!(error.contains("yoked or mismatched"));
    }

    #[test]
    fn stale_forecast_realization_is_not_qualified() {
        let mut value = panel();
        value["forecasts"][0]["expires_at_ns"] = serde_json::json!(1_000_000_001_u64);
        value["forecasts"][0]["target_end_ns"] = serde_json::json!(1_000_000_001_u64);
        let error = audit(&value).expect_err("expiry must cover the target window");
        assert!(error.contains("expires before"));
    }

    #[test]
    fn display_after_target_start_is_rejected() {
        let mut value = panel();
        value["displays"][0]["displayed_at_ns"] = serde_json::json!(1_000_000_001_u64);
        let report = audit(&value).expect("late display should produce an audit finding");
        assert!(!report.qualified_for_anticipatory_use);
        assert!(report.qualification_findings.iter().any(|x| x.contains("displayed too late")));
    }

    #[test]
    fn revision_history_must_be_contiguous_and_immutable() {
        let mut value = panel();
        let mut second = value["forecasts"][0].clone();
        second["forecast_id"] = serde_json::json!("forecast-2");
        second["revision"] = serde_json::json!(2);
        second["supersedes_forecast_id"] = serde_json::json!("forecast-1");
        second["committed_at_ns"] = serde_json::json!(900_000_000_u64);
        value["forecasts"].as_array_mut().unwrap().push(second);
        let error = audit(&value).expect_err("latest revision committed after target starts");
        assert!(error.to_string().contains("post-hoc") || error.to_string().contains("revision"));
    }

    #[test]
    fn vetoed_forecast_cannot_produce_output() {
        let mut value = panel();
        value["dispositions"][0]["action"] = serde_json::json!("veto");
        let report = audit(&value).expect("veto violation is an audit finding");
        assert!(!report.qualified_for_anticipatory_use);
        assert!(report.qualification_findings.iter().any(|x| x.contains("vetoed forecast")));
    }

    #[test]
    fn user_deviation_is_reported_but_not_scored_as_disobedience() {
        let mut value = panel();
        value["realizations"][0]["events"][0]["forecast_event_id"] = serde_json::Value::Null;
        let report = audit(&value).expect("unforecasted contribution is allowed");
        assert!(report.qualified_for_anticipatory_use);
        assert_eq!(report.unforecasted_realized_event_count, 1);
        assert_eq!(report.matched_forecast_event_count, 0);
    }

    #[test]
    fn steer_requires_a_new_superseding_forecast() {
        let mut value = panel();
        value["dispositions"][0]["action"] = serde_json::json!("steer");
        let report = audit(&value).expect("missing steer revision remains reportable");
        assert!(!report.qualified_for_anticipatory_use);
        assert!(report.qualification_findings.iter().any(|x| x.contains("steer disposition")));
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
        assert!(error.contains("duplicate realized event_id"));
    }

    #[test]
    fn low_presence_probability_penalizes_a_realized_event() {
        let mut value = panel();
        value["forecasts"][0]["events"][0]["presence_probability"] = serde_json::json!(0.2);
        let report = audit(&value).expect("low-confidence but correct forecast is reportable");
        assert!((report.conditional_presence_brier_score.unwrap() - 0.64).abs() < 1e-9);
    }
}
