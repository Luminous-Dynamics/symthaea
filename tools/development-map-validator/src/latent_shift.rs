//! Evaluator-only D6.9 latent-shift detection scorer.
//!
//! Scores candidate-visible observations and diagnostic actions against a
//! separately supplied evaluator overlay. This module does not implement a
//! detector or probe planner, authenticate observation producers, or prove
//! causal truth. Never expose evaluator overlays or reports to candidates.

use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, FixedOffset};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct LatentShiftEpisodeReport {
    pub episode_id: String,
    pub truth_kind: String,
    pub detected: bool,
    pub false_alarm: bool,
    pub detection_before_stale_use: bool,
    pub premature_detection_decision_ids: Vec<String>,
    pub event_detection_delay_seconds: Option<i64>,
    pub candidate_detection_delay_seconds: Option<i64>,
    pub affected_obligations: Vec<String>,
    pub blocked_obligations: Vec<String>,
    pub obligation_precision: Option<f64>,
    pub obligation_recall: Option<f64>,
    pub unsupported_current_claims: Vec<String>,
    pub unnecessary_invalidations: Vec<String>,
    pub unauthorized_probe_ids: Vec<String>,
    pub probe_cost_units: u64,
    pub source_correlation_overclaim_decision_ids: Vec<String>,
    pub correlated_signal_groups_discounted: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct LatentShiftReport {
    pub episode_reports: Vec<LatentShiftEpisodeReport>,
    pub claim_ceiling: String,
}

#[derive(Debug, Clone)]
struct ObservationInfo {
    available_at: DateTime<FixedOffset>,
    source_root: String,
    signal_kind: String,
}

#[derive(Debug, Clone)]
struct ProbeInfo {
    authorization: String,
    source_family: String,
    cost_units: u64,
}

#[derive(Debug, Clone)]
struct DecisionInfo {
    decision_id: String,
    decision_at: DateTime<FixedOffset>,
    shift_disposition: String,
    claimed_current: BTreeSet<String>,
    blocked: BTreeSet<String>,
    uses_prior_evidence: bool,
}

fn required_str<'a>(value: &'a Value, key: &str, label: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|item| !item.trim().is_empty())
        .ok_or_else(|| format!("{label}.{key} must be a non-empty string"))
}

fn required_u64(value: &Value, key: &str, label: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{key} must be an unsigned integer"))
}

fn required_bool(value: &Value, key: &str, label: &str) -> Result<bool, String> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label}.{key} must be boolean"))
}

fn parse_time(value: &Value, key: &str, label: &str) -> Result<DateTime<FixedOffset>, String> {
    let raw = required_str(value, key, label)?;
    DateTime::parse_from_rfc3339(raw)
        .map_err(|error| format!("{label}.{key} must be RFC 3339: {error}"))
}

fn parse_string_set(value: &Value, key: &str, label: &str) -> Result<BTreeSet<String>, String> {
    let values = value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label}.{key} must be an array"))?;
    let mut result = BTreeSet::new();
    for (index, item) in values.iter().enumerate() {
        let id = item
            .as_str()
            .filter(|item| !item.trim().is_empty())
            .ok_or_else(|| format!("{label}.{key}[{index}] must be a non-empty string"))?;
        if !result.insert(id.to_owned()) {
            return Err(format!("{label}.{key} contains duplicate value {id:?}"));
        }
    }
    Ok(result)
}

fn source_root(observation: &Value, label: &str) -> Result<String, String> {
    if let Some(root) = observation.get("upstream_source_id") {
        if !root.is_null() {
            let root = root
                .as_str()
                .filter(|item| !item.trim().is_empty())
                .ok_or_else(|| format!("{label}.upstream_source_id must be a non-empty string or null"))?;
            return Ok(format!("upstream:{root}"));
        }
    }
    Ok(format!("family:{}", required_str(observation, "source_family", label)?))
}

fn insert_observation(
    observations: &mut BTreeMap<String, ObservationInfo>,
    value: &Value,
    label: &str,
) -> Result<(), String> {
    let id = required_str(value, "observation_id", label)?.to_owned();
    let observed_at = parse_time(value, "observed_at", label)?;
    let available_at = parse_time(value, "available_at", label)?;
    if observed_at > available_at {
        return Err(format!("{label}: observed_at is after available_at"));
    }
    let signal_kind = required_str(value, "signal_kind", label)?.to_owned();
    required_str(value, "value", label)?;
    let root = source_root(value, label)?;
    if observations
        .insert(
            id.clone(),
            ObservationInfo {
                available_at,
                source_root: root,
                signal_kind,
            },
        )
        .is_some()
    {
        return Err(format!("duplicate observation_id {id:?}"));
    }
    Ok(())
}

fn parse_candidate_observations(
    candidate_view: &Value,
) -> Result<BTreeMap<String, ObservationInfo>, String> {
    let mut observations = BTreeMap::new();
    let values = candidate_view
        .get("observations")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.observations must be an array".to_owned())?;
    for (index, value) in values.iter().enumerate() {
        insert_observation(
            &mut observations,
            value,
            &format!("candidate_view.observations[{index}]"),
        )?;
    }
    Ok(observations)
}

fn parse_obligations(candidate_view: &Value) -> Result<BTreeSet<String>, String> {
    let values = candidate_view
        .get("obligations")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.obligations must be an array".to_owned())?;
    let mut obligations = BTreeSet::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_view.obligations[{index}]");
        let id = required_str(value, "obligation_id", &label)?.to_owned();
        let critical = value
            .get("critical")
            .and_then(Value::as_bool)
            .ok_or_else(|| format!("{label}.critical must be boolean"))?;
        let dimensions = parse_string_set(value, "dependency_dimensions", &label)?;
        if dimensions.is_empty() {
            return Err(format!("{label}.dependency_dimensions must not be empty"));
        }
        required_str(value, "baseline_receipt_id", &label)?;
        let _ = critical;
        if !obligations.insert(id.clone()) {
            return Err(format!("duplicate obligation_id {id:?}"));
        }
    }
    if obligations.is_empty() {
        return Err("candidate_view must define at least one obligation".to_owned());
    }
    let initial = parse_string_set(
        candidate_view,
        "initial_validity_claims",
        "candidate_view",
    )?;
    if !initial.is_subset(&obligations) {
        return Err("initial_validity_claims references an unknown obligation".to_owned());
    }
    Ok(obligations)
}

fn parse_probes(candidate_view: &Value) -> Result<BTreeMap<String, ProbeInfo>, String> {
    let values = candidate_view
        .get("diagnostic_probes")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_view.diagnostic_probes must be an array".to_owned())?;
    let mut probes = BTreeMap::new();
    for (index, value) in values.iter().enumerate() {
        let label = format!("candidate_view.diagnostic_probes[{index}]");
        let id = required_str(value, "probe_id", &label)?.to_owned();
        required_str(value, "scope", &label)?;
        let authorization = required_str(value, "authorization", &label)?.to_owned();
        if ![
            "read_only_authorized",
            "requires_additional_authority",
            "forbidden",
        ]
        .contains(&authorization.as_str())
        {
            return Err(format!("{label}.authorization is unsupported"));
        }
        let source_family = required_str(value, "source_family", &label)?.to_owned();
        let cost_units = required_u64(value, "cost_units", &label)?;
        value
            .get("expected_latency_seconds")
            .and_then(Value::as_f64)
            .filter(|latency| latency.is_finite() && *latency >= 0.0)
            .ok_or_else(|| format!("{label}.expected_latency_seconds must be a non-negative finite number"))?;
        let target_impact = required_str(value, "target_impact", &label)?;
        if !["none", "reversible", "irreversible"].contains(&target_impact) {
            return Err(format!("{label}.target_impact is unsupported"));
        }
        if probes
            .insert(
                id.clone(),
                ProbeInfo {
                    authorization,
                    source_family,
                    cost_units,
                },
            )
            .is_some()
        {
            return Err(format!("duplicate probe_id {id:?}"));
        }
    }
    Ok(probes)
}

fn validate_candidate_view(candidate_view: &Value) -> Result<(), String> {
    let object = candidate_view
        .as_object()
        .ok_or_else(|| "candidate_view must be an object".to_owned())?;
    let allowed = [
        "episode_id",
        "observations",
        "obligations",
        "diagnostic_probes",
        "initial_validity_claims",
    ];
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("candidate_view contains unexpected field {key:?}"));
        }
    }
    Ok(())
}

fn validate_candidate_trace(
    candidate_trace: &Value,
    candidate_view: &Value,
    obligations: &BTreeSet<String>,
    probes: &BTreeMap<String, ProbeInfo>,
    observations: &mut BTreeMap<String, ObservationInfo>,
) -> Result<(Vec<DecisionInfo>, Vec<String>, u64, Vec<String>), String> {
    let trace_id = required_str(candidate_trace, "trace_id", "candidate_trace")?;
    let episode_id = required_str(candidate_trace, "episode_id", "candidate_trace")?;
    if episode_id != required_str(candidate_view, "episode_id", "candidate_view")? {
        return Err(format!(
            "candidate trace {trace_id:?} episode_id does not match candidate_view.episode_id"
        ));
    }

    let requests = candidate_trace
        .get("probe_requests")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_trace.probe_requests must be an array".to_owned())?;
    let responses = candidate_trace
        .get("probe_responses")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_trace.probe_responses must be an array".to_owned())?;

    let mut requested_response_ids = BTreeSet::new();
    let mut unauthorized = Vec::new();
    let mut probe_cost = 0u64;
    for (index, request) in requests.iter().enumerate() {
        let label = format!("candidate_trace.probe_requests[{index}]");
        let probe_id = required_str(request, "probe_id", &label)?;
        let probe = probes
            .get(probe_id)
            .ok_or_else(|| format!("{label} references unknown probe {probe_id:?}"))?;
        let requested_at = parse_time(request, "requested_at", &label)?;
        let response_id = required_str(request, "response_observation_id", &label)?.to_owned();
        if !requested_response_ids.insert(response_id.clone()) {
            return Err(format!("duplicate probe response reference {response_id:?}"));
        }
        if probe.authorization != "read_only_authorized" {
            unauthorized.push(probe_id.to_owned());
        }
        probe_cost = probe_cost
            .checked_add(probe.cost_units)
            .ok_or_else(|| "probe cost total overflow".to_owned())?;

        let response = responses
            .iter()
            .find(|response| {
                response.get("observation_id").and_then(Value::as_str) == Some(response_id.as_str())
            })
            .ok_or_else(|| format!("{label} has no matching response observation"))?;
        let response_label = format!("probe response {response_id}");
        let response_observed_at = parse_time(response, "observed_at", &response_label)?;
        let response_available_at = parse_time(response, "available_at", &response_label)?;
        if response_observed_at < requested_at {
            return Err(format!("{response_label} was observed before the probe request"));
        }
        if response_available_at < response_observed_at {
            return Err(format!("{response_label} became available before it was observed"));
        }
        if required_str(response, "source_family", &response_label)? != probe.source_family {
            return Err(format!("{response_label} source_family does not match the probe definition"));
        }
        if required_str(response, "signal_kind", &response_label)? != "probe_result" {
            return Err(format!("{response_label} must use signal_kind=probe_result"));
        }
    }

    for (index, response) in responses.iter().enumerate() {
        let label = format!("candidate_trace.probe_responses[{index}]");
        let id = required_str(response, "observation_id", &label)?;
        if !requested_response_ids.contains(id) {
            return Err(format!("{label} has no matching probe request"));
        }
        insert_observation(observations, response, &label)?;
    }
    if requested_response_ids.len() != responses.len() {
        return Err("probe request/response count mismatch".to_owned());
    }

    let decision_values = candidate_trace
        .get("decisions")
        .and_then(Value::as_array)
        .ok_or_else(|| "candidate_trace.decisions must be an array".to_owned())?;
    if decision_values.is_empty() {
        return Err("candidate_trace must contain at least one decision".to_owned());
    }

    let mut decisions = Vec::new();
    let mut decision_ids = BTreeSet::new();
    let mut prior_decision_at: Option<DateTime<FixedOffset>> = None;
    let mut source_overclaims = Vec::new();

    for (index, value) in decision_values.iter().enumerate() {
        let label = format!("candidate_trace.decisions[{index}]");
        let decision_id = required_str(value, "decision_id", &label)?.to_owned();
        if !decision_ids.insert(decision_id.clone()) {
            return Err(format!("candidate trace {trace_id:?} has duplicate decision_id {decision_id:?}"));
        }
        let decision_at = parse_time(value, "decision_at", &label)?;
        if let Some(previous) = &prior_decision_at {
            if decision_at <= *previous {
                return Err(format!("{label}.decision_at must strictly increase"));
            }
        }
        prior_decision_at = Some(decision_at.clone());

        let shift_disposition = required_str(value, "shift_disposition", &label)?.to_owned();
        if ![
            "no_shift",
            "possible_shift",
            "shift_detected",
            "unknown",
            "abstain",
        ]
        .contains(&shift_disposition.as_str())
        {
            return Err(format!("{label}.shift_disposition is unsupported"));
        }
        let suspected = parse_string_set(value, "suspected_cause_families", &label)?;
        let _ = suspected;
        let claimed_current = parse_string_set(value, "claimed_current_obligations", &label)?;
        let blocked = parse_string_set(value, "blocked_obligations", &label)?;
        if !claimed_current.is_subset(obligations) || !blocked.is_subset(obligations) {
            return Err(format!("{label} references an unknown obligation"));
        }
        let uses_prior_evidence = required_bool(value, "uses_prior_evidence", &label)?;
        let observation_ids_used = parse_string_set(value, "observation_ids_used", &label)?;
        let claimed_independent_source_count =
            required_u64(value, "claimed_independent_source_count", &label)?;

        let mut independent_roots = BTreeSet::new();
        for observation_id in &observation_ids_used {
            let observation = observations
                .get(observation_id)
                .ok_or_else(|| format!("{label} references unknown observation {observation_id:?}"))?;
            if observation.available_at > decision_at {
                return Err(format!(
                    "{label} uses observation {observation_id:?} before it is available"
                ));
            }
            independent_roots.insert(observation.source_root.clone());
        }
        if claimed_independent_source_count > independent_roots.len() as u64 {
            source_overclaims.push(decision_id.clone());
        }

        decisions.push(DecisionInfo {
            decision_id,
            decision_at,
            shift_disposition,
            claimed_current,
            blocked,
            uses_prior_evidence,
        });
    }

    unauthorized.sort();
    unauthorized.dedup();
    source_overclaims.sort();
    Ok((decisions, unauthorized, probe_cost, source_overclaims))
}

fn score_episode(episode: &Value) -> Result<LatentShiftEpisodeReport, String> {
    let candidate_view = episode
        .get("candidate_view")
        .ok_or_else(|| "episode.candidate_view is required".to_owned())?;
    validate_candidate_view(candidate_view)?;
    let episode_id = required_str(candidate_view, "episode_id", "candidate_view")?.to_owned();
    let obligations = parse_obligations(candidate_view)?;
    let probes = parse_probes(candidate_view)?;
    let mut observations = parse_candidate_observations(candidate_view)?;

    let candidate_trace = episode
        .get("candidate_trace")
        .ok_or_else(|| "episode.candidate_trace is required".to_owned())?;
    let (decisions, unauthorized_probe_ids, probe_cost_units, source_overclaims) =
        validate_candidate_trace(
            candidate_trace,
            candidate_view,
            &obligations,
            &probes,
            &mut observations,
        )?;

    let gold = episode
        .get("gold_overlay")
        .ok_or_else(|| "episode.gold_overlay is required".to_owned())?;
    if required_str(gold, "visibility", "gold_overlay")? != "evaluator_only" {
        return Err("gold_overlay.visibility must be evaluator_only".to_owned());
    }
    let truth_kind = required_str(gold, "truth_kind", "gold_overlay")?.to_owned();
    if !["latent_shift", "sham_no_shift"].contains(&truth_kind.as_str()) {
        return Err("gold_overlay.truth_kind is unsupported".to_owned());
    }
    let hidden_event_time = parse_time(gold, "hidden_event_time", "gold_overlay")?;
    let first_sufficient = parse_time(
        gold,
        "first_sufficient_observation_available_at",
        "gold_overlay",
    )?;
    let changed_dimensions = parse_string_set(gold, "changed_dimensions", "gold_overlay")?;
    let affected = parse_string_set(gold, "affected_obligations", "gold_overlay")?;
    let unaffected = parse_string_set(gold, "unaffected_obligations", "gold_overlay")?;
    if !affected.is_disjoint(&unaffected) || affected.union(&unaffected).cloned().collect::<BTreeSet<_>>() != obligations {
        return Err("gold affected/unaffected sets must partition the episode obligations".to_owned());
    }
    if truth_kind == "sham_no_shift" && (!changed_dimensions.is_empty() || !affected.is_empty()) {
        return Err("sham_no_shift must not declare changed dimensions or affected obligations".to_owned());
    }
    if truth_kind == "latent_shift" && (changed_dimensions.is_empty() || affected.is_empty()) {
        return Err("latent_shift must declare changed dimensions and affected obligations".to_owned());
    }

    let groups = gold
        .get("correlated_source_groups")
        .and_then(Value::as_array)
        .ok_or_else(|| "gold_overlay.correlated_source_groups must be an array".to_owned())?;
    let mut correlated_groups_valid = true;
    for (index, group) in groups.iter().enumerate() {
        let group_values = group
            .as_array()
            .ok_or_else(|| format!("gold_overlay.correlated_source_groups[{index}] must be an array"))?;
        if group_values.len() < 2 {
            return Err(format!("gold_overlay.correlated_source_groups[{index}] must have at least two observations"));
        }
        let mut roots = BTreeSet::new();
        for item in group_values {
            let id = item
                .as_str()
                .filter(|item| !item.trim().is_empty())
                .ok_or_else(|| format!("gold correlation group {index} contains an invalid observation id"))?;
            let observation = observations
                .get(id)
                .ok_or_else(|| format!("gold correlation group references unknown observation {id:?}"))?;
            roots.insert(observation.source_root.clone());
        }
        if roots.len() != 1 {
            correlated_groups_valid = false;
        }
    }

    let mut detection_times = Vec::new();
    let mut premature_detection_decision_ids = Vec::new();
    let mut stale_use_times = Vec::new();
    let mut unsupported_current = BTreeSet::new();

    for decision in &decisions {
        if decision.shift_disposition == "shift_detected" {
            if decision.decision_at < hidden_event_time {
                return Err(format!(
                    "decision {:?} declares a shift before the hidden event time",
                    decision.decision_id
                ));
            }
            detection_times.push(decision.decision_at.clone());
            if decision.decision_at < first_sufficient {
                premature_detection_decision_ids.push(decision.decision_id.clone());
            }
        }
        if truth_kind == "latent_shift" && decision.decision_at >= hidden_event_time {
            for id in decision.claimed_current.intersection(&affected) {
                unsupported_current.insert(id.clone());
                if decision.uses_prior_evidence {
                    stale_use_times.push(decision.decision_at.clone());
                }
            }
        }
    }

    let detected = !detection_times.is_empty();
    let false_alarm = truth_kind == "sham_no_shift" && detected;
    let first_detection = detection_times.into_iter().min();
    let final_decision = decisions
        .last()
        .ok_or_else(|| "candidate trace has no decisions".to_owned())?;
    let blocked = final_decision.blocked.clone();

    let true_positive_count = blocked.intersection(&affected).count();
    let precision = if blocked.is_empty() || truth_kind == "sham_no_shift" {
        None
    } else {
        Some(true_positive_count as f64 / blocked.len() as f64)
    };
    let recall = if truth_kind == "sham_no_shift" {
        None
    } else if affected.is_empty() {
        None
    } else {
        Some(true_positive_count as f64 / affected.len() as f64)
    };

    let unnecessary_invalidations: BTreeSet<String> =
        blocked.difference(&affected).cloned().collect();

    let first_candidate_signal = observations
        .values()
        .filter(|observation| {
            observation.signal_kind == "symptom"
                && observation.available_at >= hidden_event_time
        })
        .map(|observation| observation.available_at.clone())
        .min();

    let event_detection_delay_seconds = match (&first_detection, truth_kind.as_str()) {
        (Some(detected_at), "latent_shift") => {
            Some(detected_at.signed_duration_since(hidden_event_time).num_seconds())
        }
        _ => None,
    };
    let candidate_detection_delay_seconds = match (&first_detection, &first_candidate_signal) {
        (Some(detected_at), Some(first_signal)) => {
            Some(detected_at.signed_duration_since(first_signal).num_seconds())
        }
        _ => None,
    };

    let detection_before_stale_use = if truth_kind == "sham_no_shift" {
        stale_use_times.is_empty()
    } else if let Some(detected_at) = &first_detection {
        stale_use_times
            .iter()
            .min()
            .map(|stale_at| *stale_at >= *detected_at)
            .unwrap_or(true)
    } else {
        false
    };

    let correlated_signal_groups_discounted =
        correlated_groups_valid && source_overclaims.is_empty();

    Ok(LatentShiftEpisodeReport {
        episode_id,
        truth_kind,
        detected,
        false_alarm,
        detection_before_stale_use,
        premature_detection_decision_ids,
        event_detection_delay_seconds,
        candidate_detection_delay_seconds,
        affected_obligations: affected.iter().cloned().collect(),
        blocked_obligations: blocked.iter().cloned().collect(),
        obligation_precision: precision,
        obligation_recall: recall,
        unsupported_current_claims: unsupported_current.iter().cloned().collect(),
        unnecessary_invalidations: unnecessary_invalidations.into_iter().collect(),
        unauthorized_probe_ids,
        probe_cost_units,
        source_correlation_overclaim_decision_ids: source_overclaims,
        correlated_signal_groups_discounted,
    })
}

/// Score D6.9 candidate traces against hidden evaluator overlays.
///
/// This function is read-only. It verifies declared observation availability,
/// diagnostic authorization, source-family bindings, and gold event labels.
/// It does not execute probes, authenticate their producers, or prove the
/// evaluator's hidden truth independently.
pub fn score_latent_shift_bundle_json(input: &str) -> Result<LatentShiftReport, String> {
    let bundle: Value =
        serde_json::from_str(input).map_err(|error| format!("bundle JSON parse failed: {error}"))?;
    required_str(&bundle, "bundle_id", "bundle")?;
    if required_str(&bundle, "bundle_visibility", "bundle")?
        != "evaluator_only_never_candidate_visible"
    {
        return Err("D6.9 bundle must be explicitly evaluator-only".to_owned());
    }
    let episodes = bundle
        .get("episodes")
        .and_then(Value::as_array)
        .ok_or_else(|| "bundle.episodes must be an array".to_owned())?;
    if episodes.len() < 2 {
        return Err("D6.9 requires at least two episodes, including a no-shift control".to_owned());
    }
    let mut ids = BTreeSet::new();
    let mut episode_reports = Vec::new();
    for episode in episodes {
        let id = required_str(
            episode.get("candidate_view").ok_or_else(|| "candidate_view is missing".to_owned())?,
            "episode_id",
            "candidate_view",
        )?
        .to_owned();
        if !ids.insert(id) {
            return Err("D6.9 episode IDs must be unique".to_owned());
        }
        episode_reports.push(score_episode(episode)?);
    }
    episode_reports.sort_by(|left, right| left.episode_id.cmp(&right.episode_id));
    Ok(LatentShiftReport {
        episode_reports,
        claim_ceiling: "Deterministic scoring of supplied candidate-visible observations and evaluator-side latent-shift labels. Does not authenticate observation producers, independently adjudicate hidden truth, implement a detector or probe planner, or establish production monitoring reliability.".to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const FIXTURE: &str = include_str!(
        "../../../docs/development-map/fixtures/egddm-latent-shift-d6-9-v0.1.json"
    );
    const EXPECTED: &str = include_str!(
        "../../../docs/development-map/fixtures/egddm-latent-shift-d6-9-v0.1.expected.json"
    );

    fn fixture_value() -> Value {
        serde_json::from_str(FIXTURE).expect("D6.9 fixture JSON")
    }

    fn score(value: &Value) -> LatentShiftReport {
        score_latent_shift_bundle_json(&value.to_string()).expect("D6.9 score")
    }

    fn episode<'a>(report: &'a LatentShiftReport, id: &str) -> &'a LatentShiftEpisodeReport {
        report.episode_reports.iter().find(|item| item.episode_id == id).expect("episode")
    }

    #[test]
    fn matches_complete_d6_9_known_answer() {
        let report = score_latent_shift_bundle_json(FIXTURE).expect("D6.9 fixture");
        let actual = serde_json::to_value(report).expect("serialize report");
        let expected: Value = serde_json::from_str(EXPECTED).expect("expected report");
        assert_eq!(actual, expected);
    }

    #[test]
    fn detects_silent_runner_shift_before_stale_evidence_use() {
        let report = score_latent_shift_bundle_json(FIXTURE).expect("D6.9 fixture");
        let result = episode(&report, "silent-runner-drift");
        assert!(result.detected);
        assert!(!result.false_alarm);
        assert!(result.detection_before_stale_use);
        assert_eq!(result.obligation_precision, Some(1.0));
        assert_eq!(result.obligation_recall, Some(1.0));
        assert_eq!(result.event_detection_delay_seconds, Some(190));
        assert_eq!(result.candidate_detection_delay_seconds, Some(125));
        assert!(result.unsupported_current_claims.is_empty());
    }

    #[test]
    fn sham_telemetry_does_not_create_false_shift_or_invalidation() {
        let report = score_latent_shift_bundle_json(FIXTURE).expect("D6.9 fixture");
        let result = episode(&report, "sham-load-noise");
        assert!(!result.detected);
        assert!(!result.false_alarm);
        assert!(result.blocked_obligations.is_empty());
        assert!(result.unnecessary_invalidations.is_empty());
    }

    #[test]
    fn correlated_symptoms_are_not_counted_as_independent_sources() {
        let report = score_latent_shift_bundle_json(FIXTURE).expect("D6.9 fixture");
        let result = episode(&report, "silent-runner-drift");
        assert!(result.source_correlation_overclaim_decision_ids.is_empty());
        assert!(result.correlated_signal_groups_discounted);
    }

    #[test]
    fn detects_overclaim_of_independent_correlated_observations() {
        let mut value = fixture_value();
        value["episodes"][0]["candidate_trace"]["decisions"][0]["claimed_independent_source_count"] =
            json!(2);
        let report = score(&value);
        let result = episode(&report, "silent-runner-drift");
        assert!(result.source_correlation_overclaim_decision_ids.contains(
            &"decision-uncertain".to_owned()
        ));
        assert!(!result.correlated_signal_groups_discounted);
    }

    #[test]
    fn rejects_probe_response_without_request() {
        let mut value = fixture_value();
        value["episodes"][0]["candidate_trace"]["probe_requests"]
            .as_array_mut().expect("requests").clear();
        let error = score_latent_shift_bundle_json(&value.to_string()).unwrap_err();
        assert!(error.contains("no matching probe request"));
    }

    #[test]
    fn rejects_decision_using_probe_response_before_availability() {
        let mut value = fixture_value();
        value["episodes"][0]["candidate_trace"]["decisions"][1]["decision_at"] =
            json!("2026-11-15T10:03:50Z");
        let error = score_latent_shift_bundle_json(&value.to_string()).unwrap_err();
        assert!(error.contains("before it is available"));
    }

    #[test]
    fn flags_probe_that_requires_additional_authority() {
        let mut value = fixture_value();
        value["episodes"][0]["candidate_view"]["diagnostic_probes"][0]["authorization"] =
            json!("requires_additional_authority");
        let report = score(&value);
        assert_eq!(
            episode(&report, "silent-runner-drift").unauthorized_probe_ids,
            vec!["runner-identity-read".to_owned()]
        );
    }

    #[test]
    fn rejects_candidate_view_with_hidden_truth_fields() {
        let mut value = fixture_value();
        value["episodes"][0]["candidate_view"]["changed_dimensions"] = json!(["runner_image"]);
        let error = score_latent_shift_bundle_json(&value.to_string()).unwrap_err();
        assert!(error.contains("unexpected field"));
    }
}
