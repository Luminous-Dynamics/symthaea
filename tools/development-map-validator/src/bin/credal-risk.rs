//! D6.16 evaluator-only prototype: ambiguity-aware posterior risk.
//!
//! The candidate observes interval bounds for the prior over admissible hypotheses.
//! Evaluator-only safety labels and the actual hidden hypothesis are joined after
//! the candidate trace is immutable. This module does not estimate real-world
//! probabilities or authenticate the custody of its evaluator oracle.

use chrono::{DateTime, FixedOffset};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const CLAIM_CEILING: &str = "Synthetic known-answer evaluator over deterministic observations and bounded-simplex prior ambiguity. The worst-case risk is exact for the supplied per-hypothesis interval bounds and binary action-safety labels. It does not validate that the intervals contain the real data-generating process, estimate calibrated priors, model noisy or correlated observations, authenticate oracle custody, or prove production safety.";

#[derive(Debug, Clone, PartialEq, Eq)]
struct Interval {
    lower: u64,
    upper: u64,
    nominal: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActionPolicy {
    irreversible: bool,
    hard_gate: bool,
    max_residual_risk_bps: u64,
}

#[derive(Debug, Clone)]
struct CandidateOption {
    available_at: DateTime<FixedOffset>,
}

#[derive(Debug, Clone)]
struct Hypothesis {
    safe_actions: BTreeSet<String>,
    observations: BTreeMap<String, String>,
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a serde_json::Map<String, Value>, String> {
    value.as_object().ok_or_else(|| format!("{label} must be an object"))
}

fn allowed_keys(value: &Value, allowed: &[&str], label: &str) -> Result<(), String> {
    for key in object(value, label)?.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("{label} contains unknown field {key:?}"));
        }
    }
    Ok(())
}

fn required_value<'a>(value: &'a Value, key: &str, label: &str) -> Result<&'a Value, String> {
    value.get(key).ok_or_else(|| format!("{label}.{key} is required"))
}

fn required_str(value: &Value, key: &str, label: &str) -> Result<String, String> {
    required_value(value, key, label)?
        .as_str()
        .filter(|item| !item.trim().is_empty())
        .map(str::to_owned)
        .ok_or_else(|| format!("{label}.{key} must be a non-empty string"))
}

fn required_u64(value: &Value, key: &str, label: &str) -> Result<u64, String> {
    required_value(value, key, label)?
        .as_u64()
        .ok_or_else(|| format!("{label}.{key} must be a non-negative integer"))
}

fn required_bool(value: &Value, key: &str, label: &str) -> Result<bool, String> {
    required_value(value, key, label)?
        .as_bool()
        .ok_or_else(|| format!("{label}.{key} must be boolean"))
}

fn parse_time(value: &Value, key: &str, label: &str) -> Result<DateTime<FixedOffset>, String> {
    let raw = required_str(value, key, label)?;
    DateTime::parse_from_rfc3339(&raw)
        .map_err(|error| format!("{label}.{key} must be RFC 3339: {error}"))
}

fn string_set(value: &Value, label: &str) -> Result<BTreeSet<String>, String> {
    let items = value.as_array().ok_or_else(|| format!("{label} must be an array"))?;
    let mut output = BTreeSet::new();
    for (index, item) in items.iter().enumerate() {
        let name = item.as_str()
            .filter(|item| !item.trim().is_empty())
            .ok_or_else(|| format!("{label}[{index}] must be a non-empty string"))?;
        if !output.insert(name.to_owned()) {
            return Err(format!("{label} contains duplicate value {name:?}"));
        }
    }
    Ok(output)
}

fn parse_actions(value: &Value, root: &str) -> Result<BTreeMap<String, ActionPolicy>, String> {
    let items = required_value(value, "actions", root)?
        .as_array().ok_or_else(|| format!("{root}.actions must be an array"))?;
    if items.is_empty() {
        return Err(format!("{root}.actions must not be empty"));
    }
    let mut output = BTreeMap::new();
    for (index, item) in items.iter().enumerate() {
        let label = format!("{root}.actions[{index}]");
        allowed_keys(item, &["action_id", "irreversible", "hard_gate", "max_residual_risk_bps"], &label)?;
        let id = required_str(item, "action_id", &label)?;
        let irreversible = required_bool(item, "irreversible", &label)?;
        let hard_gate = required_bool(item, "hard_gate", &label)?;
        let max_residual_risk_bps = required_u64(item, "max_residual_risk_bps", &label)?;
        if max_residual_risk_bps > 10_000 {
            return Err(format!("{label}.max_residual_risk_bps must be <= 10000"));
        }
        if hard_gate && max_residual_risk_bps != 0 {
            return Err(format!("{label}: hard-gated actions must have zero residual-risk allowance"));
        }
        if irreversible && !hard_gate {
            return Err(format!("{label}: irreversible actions require a hard gate"));
        }
        if output.insert(id.clone(), ActionPolicy {
            irreversible,
            hard_gate,
            max_residual_risk_bps,
        }).is_some() {
            return Err(format!("duplicate action_id {id:?}"));
        }
    }
    Ok(output)
}

fn parse_intervals(value: &Value, root: &str) -> Result<BTreeMap<String, Interval>, String> {
    let items = required_value(value, "hypothesis_bounds", root)?
        .as_array().ok_or_else(|| format!("{root}.hypothesis_bounds must be an array"))?;
    if items.is_empty() || items.len() > 64 {
        return Err(format!("{root}.hypothesis_bounds must contain 1..=64 entries"));
    }
    let mut output = BTreeMap::new();
    let mut lower_total = 0u64;
    let mut upper_total = 0u64;
    let mut nominal_total = 0u64;
    for (index, item) in items.iter().enumerate() {
        let label = format!("{root}.hypothesis_bounds[{index}]");
        allowed_keys(item, &[
            "hypothesis_id",
            "lower_probability_bps",
            "upper_probability_bps",
            "nominal_probability_bps",
        ], &label)?;
        let id = required_str(item, "hypothesis_id", &label)?;
        let lower = required_u64(item, "lower_probability_bps", &label)?;
        let upper = required_u64(item, "upper_probability_bps", &label)?;
        let nominal = required_u64(item, "nominal_probability_bps", &label)?;
        if lower > upper || upper > 10_000 {
            return Err(format!("{label} must satisfy 0 <= lower <= upper <= 10000"));
        }
        if nominal < lower || nominal > upper {
            return Err(format!("{label}.nominal_probability_bps must lie inside its interval"));
        }
        if upper == 0 {
            return Err(format!("{label} has zero upper probability and is not an admissible hypothesis"));
        }
        lower_total = lower_total.checked_add(lower).ok_or("lower probability sum overflow")?;
        upper_total = upper_total.checked_add(upper).ok_or("upper probability sum overflow")?;
        nominal_total = nominal_total.checked_add(nominal).ok_or("nominal probability sum overflow")?;
        if output.insert(id.clone(), Interval { lower, upper, nominal }).is_some() {
            return Err(format!("duplicate hypothesis_id {id:?}"));
        }
    }
    if lower_total > 10_000 || upper_total < 10_000 {
        return Err(format!(
            "prior ambiguity set is infeasible: lower sum {lower_total}, upper sum {upper_total}, required total 10000"
        ));
    }
    if nominal_total != 10_000 {
        return Err(format!("nominal probability weights must sum to 10000, got {nominal_total}"));
    }
    Ok(output)
}

fn parse_candidate_options(value: &Value) -> Result<BTreeMap<String, CandidateOption>, String> {
    let items = required_value(value, "options", "candidate_view")?
        .as_array().ok_or("candidate_view.options must be an array")?;
    if items.is_empty() || items.len() > 20 {
        return Err("candidate_view.options must contain 1..=20 options".to_owned());
    }
    let mut output = BTreeMap::new();
    for (index, item) in items.iter().enumerate() {
        let label = format!("candidate_view.options[{index}]");
        allowed_keys(item, &["option_id", "available_at"], &label)?;
        let id = required_str(item, "option_id", &label)?;
        let available_at = parse_time(item, "available_at", &label)?;
        if output.insert(id.clone(), CandidateOption { available_at }).is_some() {
            return Err(format!("duplicate option_id {id:?}"));
        }
    }
    Ok(output)
}

fn parse_oracle_options(
    value: &Value,
    candidate: &BTreeMap<String, CandidateOption>,
) -> Result<(), String> {
    let items = required_value(value, "options", "oracle")?
        .as_array().ok_or("oracle.options must be an array")?;
    let mut output = BTreeMap::new();
    for (index, item) in items.iter().enumerate() {
        let label = format!("oracle.options[{index}]");
        allowed_keys(item, &["option_id", "available_at"], &label)?;
        let id = required_str(item, "option_id", &label)?;
        let available_at = parse_time(item, "available_at", &label)?;
        if output.insert(id.clone(), available_at).is_some() {
            return Err(format!("duplicate oracle option_id {id:?}"));
        }
    }
    if output.len() != candidate.len() {
        return Err("candidate and oracle option sets must match exactly".to_owned());
    }
    for (id, visible) in candidate {
        if output.get(id) != Some(&visible.available_at) {
            return Err(format!("candidate/oracle availability mismatch for option {id:?}"));
        }
    }
    Ok(())
}

fn parse_hypotheses(
    oracle: &Value,
    hypothesis_ids: &BTreeSet<String>,
    option_ids: &BTreeSet<String>,
    action_ids: &BTreeSet<String>,
) -> Result<BTreeMap<String, Hypothesis>, String> {
    let items = required_value(oracle, "hypotheses", "oracle")?
        .as_array().ok_or("oracle.hypotheses must be an array")?;
    if items.len() != hypothesis_ids.len() {
        return Err("oracle hypothesis set must match candidate prior-bound IDs exactly".to_owned());
    }
    let mut output = BTreeMap::new();
    for (index, item) in items.iter().enumerate() {
        let label = format!("oracle.hypotheses[{index}]");
        allowed_keys(item, &["hypothesis_id", "safe_actions", "observations"], &label)?;
        let id = required_str(item, "hypothesis_id", &label)?;
        if !hypothesis_ids.contains(&id) {
            return Err(format!("{label} references unknown hypothesis {id:?}"));
        }
        let safe_actions = string_set(
            required_value(item, "safe_actions", &label)?,
            &format!("{label}.safe_actions"),
        )?;
        if safe_actions.iter().any(|action| !action_ids.contains(action)) {
            return Err(format!("{label}.safe_actions references an unknown action"));
        }
        let observations_value = required_value(item, "observations", &label)?;
        let observations_object = object(observations_value, &format!("{label}.observations"))?;
        if observations_object.len() != option_ids.len() {
            return Err(format!("{label}.observations must contain every option exactly once"));
        }
        let mut observations = BTreeMap::new();
        for option_id in option_ids {
            let observed = observations_object.get(option_id)
                .and_then(Value::as_str)
                .filter(|observed| !observed.trim().is_empty())
                .ok_or_else(|| format!("{label}.observations.{option_id} must be a non-empty string"))?;
            observations.insert(option_id.clone(), observed.to_owned());
        }
        if output.insert(id.clone(), Hypothesis { safe_actions, observations }).is_some() {
            return Err(format!("duplicate oracle hypothesis {id:?}"));
        }
    }
    let parsed_ids: BTreeSet<String> = output.keys().cloned().collect();
    if &parsed_ids != hypothesis_ids {
        return Err("oracle hypothesis set must match prior-bound IDs exactly".to_owned());
    }
    Ok(output)
}

fn nominal_risk_bps(
    intervals: &BTreeMap<String, Interval>,
    survivors: &BTreeSet<String>,
    hazards: &BTreeSet<String>,
) -> Result<u64, String> {
    let denominator: u64 = survivors.iter()
        .map(|id| intervals.get(id).map(|interval| interval.nominal).unwrap_or(0))
        .sum();
    if denominator == 0 {
        return Err("nominal prior assigns zero probability to the surviving evidence set".to_owned());
    }
    let numerator: u64 = hazards.iter()
        .map(|id| intervals.get(id).map(|interval| interval.nominal).unwrap_or(0))
        .sum();
    Ok((numerator * 10_000 + denominator - 1) / denominator)
}

fn sum_bounds(
    intervals: &BTreeMap<String, Interval>,
    ids: &BTreeSet<String>,
) -> (u64, u64) {
    let lower = ids.iter()
        .map(|id| intervals.get(id).map(|item| item.lower).unwrap_or(0))
        .sum();
    let upper = ids.iter()
        .map(|id| intervals.get(id).map(|item| item.upper).unwrap_or(0))
        .sum();
    (lower, upper)
}

// Exact linear-fractional maximization over a bounded simplex. With binary
// action safety, aggregate hypotheses into unsafe survivors, safe survivors,
// and excluded worlds. The feasible (unsafe_mass, safe_mass) region is a
// rectangle intersected by two sum bounds. A linear-fractional objective
// reaches its maximum at a polygon vertex.
fn robust_upper_risk_bps(
    intervals: &BTreeMap<String, Interval>,
    survivors: &BTreeSet<String>,
    hazards: &BTreeSet<String>,
) -> Result<u64, String> {
    if !hazards.is_subset(survivors) {
        return Err("hazard hypotheses must be a subset of surviving hypotheses".to_owned());
    }
    let safe_survivors: BTreeSet<String> = survivors.difference(hazards).cloned().collect();
    let excluded: BTreeSet<String> = intervals.keys()
        .filter(|id| !survivors.contains(*id))
        .cloned().collect();
    let (lh, uh) = sum_bounds(intervals, hazards);
    let (lg, ug) = sum_bounds(intervals, &safe_survivors);
    let (lo, uo) = sum_bounds(intervals, &excluded);
    // Enforce total probability = 10000 by intersecting h+s with the
    // residual mass that excluded hypotheses can absorb. Check support before
    // the zero-hazard shortcut: impossible evidence is never zero risk.
    let minimum_survivor_mass = (lh + lg).max(10_000u64.saturating_sub(uo));
    let maximum_survivor_mass = (uh + ug).min(10_000u64.saturating_sub(lo));
    if maximum_survivor_mass == 0 || minimum_survivor_mass > maximum_survivor_mass {
        return Err("surviving evidence set has no feasible positive prior mass".to_owned());
    }
    if uh == 0 {
        return Ok(0);
    }

    let mut vertices: BTreeSet<(u64, u64)> = BTreeSet::new();
    let mut add_vertex = |h: u64, s: u64| {
        let total = h.saturating_add(s);
        if h >= lh && h <= uh && s >= lg && s <= ug
            && total >= minimum_survivor_mass
            && total <= maximum_survivor_mass
            && total > 0
        {
            vertices.insert((h, s));
        }
    };
    let h_edges = [lh, uh];
    let s_edges = [lg, ug];
    let total_edges = [minimum_survivor_mass, maximum_survivor_mass];
    for h in h_edges {
        for s in s_edges {
            add_vertex(h, s);
        }
        for total in total_edges {
            if total >= h {
                add_vertex(h, total - h);
            }
        }
    }
    for s in s_edges {
        for total in total_edges {
            if total >= s {
                add_vertex(total - s, s);
            }
        }
    }
    if vertices.is_empty() {
        return Err("surviving evidence set has no feasible positive prior mass".to_owned());
    }
    let mut worst = 0u64;
    for (hazard_mass, safe_mass) in vertices {
        let denominator = hazard_mass + safe_mass;
        let risk = (hazard_mass * 10_000 + denominator - 1) / denominator;
        worst = worst.max(risk);
    }
    Ok(worst)
}

fn evaluate_trace(
    trace: &Value,
    intervals: &BTreeMap<String, Interval>,
    hypotheses: &BTreeMap<String, Hypothesis>,
    actual_hypothesis_id: &str,
    candidate_options: &BTreeMap<String, CandidateOption>,
    actions: &BTreeMap<String, ActionPolicy>,
    cutoff: &DateTime<FixedOffset>,
) -> Result<Value, String> {
    let trace_id = required_str(trace, "trace_id", "trace")?;
    let label = format!("trace {trace_id}");
    allowed_keys(trace, &["trace_id", "acquisitions", "decision"], &label)?;
    let actual = hypotheses.get(actual_hypothesis_id).ok_or("actual hypothesis disappeared")?;
    let acquisitions = required_value(trace, "acquisitions", &label)?
        .as_array().ok_or_else(|| format!("{label}.acquisitions must be an array"))?;
    let decision = required_value(trace, "decision", &label)?;
    allowed_keys(decision, &["kind", "action_id", "claimed_residual_risk_bps"], &format!("{label}.decision"))?;
    let decision_kind = required_str(decision, "kind", &format!("{label}.decision"))?;
    if decision_kind != "act" && decision_kind != "defer" {
        return Err(format!("{label}.decision.kind must be act or defer"));
    }
    let action_id = required_str(decision, "action_id", &format!("{label}.decision"))?;
    let policy = actions.get(&action_id)
        .ok_or_else(|| format!("{label} references unknown action {action_id:?}"))?;
    let claimed_risk = match decision.get("claimed_residual_risk_bps") {
        None | Some(Value::Null) => None,
        Some(Value::Number(number)) => number.as_u64(),
        _ => None,
    };
    if decision.get("claimed_residual_risk_bps").is_some() && claimed_risk.is_none() {
        return Err(format!("{label}.decision.claimed_residual_risk_bps must be an integer or null"));
    }
    if claimed_risk.is_some_and(|risk| risk > 10_000) {
        return Err(format!("{label}.decision.claimed_residual_risk_bps must be <= 10000"));
    }

    let mut survivors: BTreeSet<String> = intervals.keys().cloned().collect();
    let mut previous_at: Option<DateTime<FixedOffset>> = None;
    let mut step_reports = Vec::new();
    let mut redundant = 0usize;
    let mut future_leakage = 0usize;
    let mut unavailable = 0usize;
    let mut false_observations = 0usize;

    for (index, step) in acquisitions.iter().enumerate() {
        let step_label = format!("{label}.acquisitions[{index}]");
        allowed_keys(step, &["option_id", "observed_value", "at"], &step_label)?;
        let option_id = required_str(step, "option_id", &step_label)?;
        let observed_value = required_str(step, "observed_value", &step_label)?;
        let at = parse_time(step, "at", &step_label)?;
        if previous_at.as_ref().is_some_and(|previous| &at < previous) {
            return Err(format!("{step_label}.at must be nondecreasing across sequential acquisitions"));
        }
        previous_at = Some(at.clone());
        let option = candidate_options.get(&option_id)
            .ok_or_else(|| format!("{step_label} references unknown option {option_id:?}"))?;
        let before = survivors.len();
        let mut disposition = "applied".to_owned();
        if at > *cutoff {
            future_leakage += 1;
            disposition = "future_evidence_rejected".to_owned();
        } else if at < option.available_at {
            unavailable += 1;
            disposition = "not_yet_available_rejected".to_owned();
        } else {
            let actual_value = actual.observations.get(&option_id)
                .ok_or_else(|| format!("actual hypothesis has no observation for {option_id:?}"))?;
            if actual_value != &observed_value {
                false_observations += 1;
                disposition = "false_observation_rejected".to_owned();
            } else {
                let narrowed: BTreeSet<String> = survivors.iter()
                    .filter(|hypothesis_id| {
                        hypotheses.get(*hypothesis_id)
                            .and_then(|hypothesis| hypothesis.observations.get(&option_id))
                            == Some(&observed_value)
                    })
                    .cloned()
                    .collect();
                if narrowed.is_empty() || !narrowed.contains(actual_hypothesis_id) {
                    false_observations += 1;
                    disposition = "observation_would_exclude_actual_hypothesis".to_owned();
                } else {
                    survivors = narrowed;
                    if survivors.len() == before {
                        redundant += 1;
                        disposition = "applied_no_information_gain".to_owned();
                    }
                }
            }
        }
        step_reports.push(json!({
            "option_id": option_id,
            "observed_value": observed_value,
            "at": at.to_rfc3339(),
            "hypotheses_before": before,
            "hypotheses_after": survivors.len(),
            "disposition": disposition
        }));
    }

    let hazards: BTreeSet<String> = survivors.iter()
        .filter(|hypothesis_id| {
            !hypotheses.get(*hypothesis_id)
                .is_some_and(|hypothesis| hypothesis.safe_actions.contains(&action_id))
        })
        .cloned()
        .collect();
    let nominal_risk = nominal_risk_bps(intervals, &survivors, &hazards)?;
    let robust_risk = robust_upper_risk_bps(intervals, &survivors, &hazards)?;
    let nominal_justified = nominal_risk <= policy.max_residual_risk_bps;
    let robust_justified = robust_risk <= policy.max_residual_risk_bps;
    let actual_world_safe = actual.safe_actions.contains(&action_id);
    let unsupported_action = decision_kind == "act" && !robust_justified;
    let actual_unsafe_action = decision_kind == "act" && !actual_world_safe;
    let hard_gate_violation = decision_kind == "act" && policy.hard_gate && robust_risk > 0;
    let nominal_false_safe = decision_kind == "act" && nominal_justified && !robust_justified;
    let unnecessary_abstention = decision_kind == "defer" && robust_justified;
    let correctly_deferred = decision_kind == "defer" && !robust_justified;
    let claimed_risk_understatement = claimed_risk.is_some_and(|claim| claim < robust_risk);
    let ambiguity_width: u64 = intervals.values().map(|interval| interval.upper - interval.lower).sum();
    let optimism_gap = robust_risk.saturating_sub(nominal_risk);

    Ok(json!({
        "trace_id": trace_id,
        "decision_kind": decision_kind,
        "action_id": action_id,
        "irreversible": policy.irreversible,
        "hard_gate": policy.hard_gate,
        "max_residual_risk_bps": policy.max_residual_risk_bps,
        "surviving_hypothesis_ids": survivors,
        "surviving_hypothesis_count": survivors.len(),
        "prior_ambiguity_width_bps": ambiguity_width,
        "nominal_posterior_risk_bps": nominal_risk,
        "worst_case_posterior_risk_bps": robust_risk,
        "prior_optimism_gap_bps": optimism_gap,
        "nominal_action_justified": nominal_justified,
        "robust_action_justified": robust_justified,
        "actual_hypothesis_supports_action": actual_world_safe,
        "unsupported_action": unsupported_action,
        "actual_unsafe_action": actual_unsafe_action,
        "hard_gate_violation": hard_gate_violation,
        "nominal_false_safe": nominal_false_safe,
        "unnecessary_abstention": unnecessary_abstention,
        "correctly_deferred": correctly_deferred,
        "claimed_residual_risk_bps": claimed_risk,
        "claimed_risk_understatement": claimed_risk_understatement,
        "acquisition_count": acquisitions.len(),
        "redundant_acquisition_count": redundant,
        "future_evidence_leakage_count": future_leakage,
        "not_yet_available_count": unavailable,
        "false_observation_count": false_observations,
        "acquisition_steps": step_reports
    }))
}

pub fn evaluate_credal_risk_parts_json(candidate_raw: &str, oracle_raw: &str) -> Result<Value, String> {
    let candidate: Value = serde_json::from_str(candidate_raw)
        .map_err(|error| format!("invalid candidate-view JSON: {error}"))?;
    let oracle: Value = serde_json::from_str(oracle_raw)
        .map_err(|error| format!("invalid evaluator-oracle JSON: {error}"))?;
    allowed_keys(&candidate, &[
        "scenario_id", "decision_cutoff", "actions", "hypothesis_bounds", "options", "traces"
    ], "candidate_view")?;
    allowed_keys(&oracle, &[
        "actual_hypothesis_id", "decision_cutoff", "actions", "hypothesis_bounds", "hypotheses", "options"
    ], "oracle")?;

    let scenario_id = required_str(&candidate, "scenario_id", "candidate_view")?;
    let cutoff = parse_time(&candidate, "decision_cutoff", "candidate_view")?;
    let oracle_cutoff = parse_time(&oracle, "decision_cutoff", "oracle")?;
    if cutoff != oracle_cutoff {
        return Err("candidate decision cutoff differs from evaluator-authoritative cutoff".to_owned());
    }
    let actions = parse_actions(&candidate, "candidate_view")?;
    let authoritative_actions = parse_actions(&oracle, "oracle")?;
    if actions != authoritative_actions {
        return Err("candidate action policies differ from evaluator-authoritative action policies".to_owned());
    }
    let candidate_intervals = parse_intervals(&candidate, "candidate_view")?;
    let oracle_intervals = parse_intervals(&oracle, "oracle")?;
    if candidate_intervals != oracle_intervals {
        return Err("candidate prior bounds differ from evaluator-authoritative prior bounds".to_owned());
    }
    let hypothesis_ids: BTreeSet<String> = candidate_intervals.keys().cloned().collect();
    let action_ids: BTreeSet<String> = actions.keys().cloned().collect();
    let candidate_options = parse_candidate_options(&candidate)?;
    let oracle_options = parse_oracle_options(&oracle, &candidate_options)?;
    let option_ids: BTreeSet<String> = candidate_options.keys().cloned().collect();
    let hypotheses = parse_hypotheses(&oracle, &hypothesis_ids, &option_ids, &action_ids)?;
    let actual_hypothesis_id = required_str(&oracle, "actual_hypothesis_id", "oracle")?;
    if !hypotheses.contains_key(&actual_hypothesis_id) {
        return Err("oracle.actual_hypothesis_id does not identify a declared hypothesis".to_owned());
    }
    let traces = required_value(&candidate, "traces", "candidate_view")?
        .as_array().ok_or("candidate_view.traces must be an array")?;
    if traces.is_empty() || traces.len() > 64 {
        return Err("candidate_view.traces must contain 1..=64 traces".to_owned());
    }
    let mut trace_ids = BTreeSet::new();
    let mut reports = Vec::new();
    for trace in traces {
        let id = required_str(trace, "trace_id", "trace")?;
        if !trace_ids.insert(id.clone()) {
            return Err(format!("duplicate trace_id {id:?}"));
        }
        reports.push(evaluate_trace(
            trace, &candidate_intervals, &hypotheses, &actual_hypothesis_id,
            &candidate_options, &actions, &cutoff,
        )?);
    }
    let count = |field: &str| -> usize {
        reports.iter().filter(|report| report[field].as_bool() == Some(true)).count()
    };
    let sum = |field: &str| -> u64 {
        reports.iter().map(|report| report[field].as_u64().unwrap_or(0)).sum()
    };
    let optimism_gap_total: u64 = reports.iter()
        .map(|report| report["prior_optimism_gap_bps"].as_u64().unwrap_or(0))
        .sum();
    Ok(json!({
        "scenario_id": scenario_id,
        "adjudication_status": "synthetic_known_answer_credal_evaluation_only",
        "hypothesis_count": candidate_intervals.len(),
        "action_count": actions.len(),
        "option_count": candidate_options.len(),
        "trace_count": reports.len(),
        "summary": {
            "unsupported_action_trace_count": count("unsupported_action"),
            "actual_unsafe_action_trace_count": count("actual_unsafe_action"),
            "hard_gate_violation_trace_count": count("hard_gate_violation"),
            "nominal_false_safe_trace_count": count("nominal_false_safe"),
            "unnecessary_abstention_trace_count": count("unnecessary_abstention"),
            "correctly_deferred_trace_count": count("correctly_deferred"),
            "future_evidence_leakage_events": sum("future_evidence_leakage_count"),
            "false_observation_events": sum("false_observation_count"),
            "total_prior_optimism_gap_bps": optimism_gap_total
        },
        "traces": reports,
        "claim_ceiling": CLAIM_CEILING
    }))
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 || args[0] == "--help" || args[0] == "-h" {
        eprintln!("Usage: credal-risk <candidate-view.json> <evaluator-oracle.json>");
        eprintln!("The candidate receives only candidate-view.json; keep evaluator-oracle.json in a separate protected evaluator workspace.");
        eprintln!("Evaluator-only D6.16 bounded-prior risk scorer.");
        process::exit(if args.len() == 2 { 0 } else { 2 });
    }
    let candidate_path = &args[0];
    let oracle_path = &args[1];
    let candidate = match fs::read_to_string(candidate_path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("ERROR D6.16-CANDIDATE-IO-001 {candidate_path}: {error}");
            process::exit(2);
        }
    };
    let oracle = match fs::read_to_string(oracle_path) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("ERROR D6.16-ORACLE-IO-001 {oracle_path}: {error}");
            process::exit(2);
        }
    };
    match evaluate_credal_risk_parts_json(&candidate, &oracle) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(output) => println!("{output}"),
            Err(error) => {
                eprintln!("ERROR D6.16-JSON-001: {error}");
                process::exit(2);
            }
        },
        Err(error) => {
            eprintln!("ERROR D6.16-CREDAL-RISK-001: {error}");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const CANDIDATE: &str = include_str!("../../../../docs/development-map/fixtures/egddm-credal-risk-candidate-v0.1.json");
    const ORACLE: &str = include_str!("../../../../docs/development-map/fixtures/egddm-credal-risk-oracle-v0.1.json");
    const EXPECTED: &str = include_str!("../../../../docs/development-map/fixtures/egddm-credal-risk-v0.1.expected.json");

    fn report() -> Value {
        evaluate_credal_risk_parts_json(CANDIDATE, ORACLE).expect("D6.16 fixture evaluates")
    }
    fn trace(value: &Value, id: &str) -> Value {
        value["traces"].as_array().unwrap().iter()
            .find(|item| item["trace_id"] == id).expect("trace exists").clone()
    }

    #[test]
    fn fixture_matches_complete_known_answer_report() {
        let actual = report();
        let expected: Value = serde_json::from_str(EXPECTED).expect("expected JSON");
        assert_eq!(actual, expected);
    }

    #[test]
    fn point_probability_is_overconfident_under_prior_ambiguity() {
        let nominal = trace(&report(), "nominal-only-probe");
        assert_eq!(nominal["nominal_posterior_risk_bps"], 1000);
        assert_eq!(nominal["worst_case_posterior_risk_bps"], 3000);
        assert_eq!(nominal["nominal_action_justified"], true);
        assert_eq!(nominal["robust_action_justified"], false);
        assert_eq!(nominal["nominal_false_safe"], true);
    }

    #[test]
    fn targeted_evidence_reduces_worst_case_risk_to_zero() {
        let acquired = trace(&report(), "robust-probe-after-cheap-signal");
        assert_eq!(acquired["surviving_hypothesis_ids"], json!(["hazard-a", "hazard-b", "safe"]));
        assert_eq!(acquired["worst_case_posterior_risk_bps"], 0);
        assert_eq!(acquired["robust_action_justified"], true);
    }

    #[test]
    fn one_gate_does_not_clear_the_other_release_hazard() {
        let partial = trace(&report(), "single-gate-not-enough-release");
        assert_eq!(partial["nominal_posterior_risk_bps"], 2000);
        assert_eq!(partial["worst_case_posterior_risk_bps"], 3334);
        assert_eq!(partial["robust_action_justified"], false);
    }

    #[test]
    fn two_independent_gates_leave_only_safe_release_hypothesis() {
        let both = trace(&report(), "two-gates-release");
        assert_eq!(both["surviving_hypothesis_ids"], json!(["safe"]));
        assert_eq!(both["worst_case_posterior_risk_bps"], 0);
        assert_eq!(both["robust_action_justified"], true);
    }

    #[test]
    fn always_act_and_always_abstain_fail_paired_controls() {
        let reckless = trace(&report(), "always-act-release");
        assert_eq!(reckless["unsupported_action"], true);
        assert_eq!(reckless["hard_gate_violation"], true);
        let abstain = trace(&report(), "always-abstain-readonly");
        assert_eq!(abstain["unnecessary_abstention"], true);
        assert_eq!(abstain["robust_action_justified"], true);
    }

    #[test]
    fn wide_registered_reversible_envelope_can_support_positive_action() {
        let wide = trace(&report(), "bounded-action-wide-envelope");
        assert_eq!(wide["worst_case_posterior_risk_bps"], 3000);
        assert_eq!(wide["max_residual_risk_bps"], 3500);
        assert_eq!(wide["robust_action_justified"], true);
    }

    #[test]
    fn false_observation_cannot_remove_the_actual_hypothesis() {
        let false_observation = trace(&report(), "false-observation-retained");
        assert_eq!(false_observation["false_observation_count"], 1);
        assert_eq!(false_observation["surviving_hypothesis_count"], 4);
        assert_eq!(false_observation["worst_case_posterior_risk_bps"], 3000);
    }

    #[test]
    fn point_intervals_reproduce_nominal_risk_exactly() {
        let mut candidate: Value = serde_json::from_str(CANDIDATE).unwrap();
        let mut oracle: Value = serde_json::from_str(ORACLE).unwrap();
        let bounds = candidate["hypothesis_bounds"].as_array_mut().unwrap();
        for bound in bounds {
            let nominal = bound["nominal_probability_bps"].clone();
            bound["lower_probability_bps"] = nominal.clone();
            bound["upper_probability_bps"] = nominal;
        }
        oracle["hypothesis_bounds"] = candidate["hypothesis_bounds"].clone();
        let result = evaluate_credal_risk_parts_json(&candidate.to_string(), &oracle.to_string()).unwrap();
        for item in result["traces"].as_array().unwrap() {
            assert_eq!(item["worst_case_posterior_risk_bps"], item["nominal_posterior_risk_bps"]);
        }
    }

    #[test]
    fn infeasible_prior_ambiguity_set_is_rejected() {
        let mut candidate: Value = serde_json::from_str(CANDIDATE).unwrap();
        let mut oracle: Value = serde_json::from_str(ORACLE).unwrap();
        candidate["hypothesis_bounds"][0]["lower_probability_bps"] = json!(9000);
        oracle["hypothesis_bounds"] = candidate["hypothesis_bounds"].clone();
        let error = evaluate_credal_risk_parts_json(&candidate.to_string(), &oracle.to_string()).unwrap_err();
        assert!(error.contains("ambiguity set is infeasible"));
    }

    #[test]
    fn candidate_cannot_change_authoritative_prior_bounds() {
        let mut candidate: Value = serde_json::from_str(CANDIDATE).unwrap();
        let oracle: Value = serde_json::from_str(ORACLE).unwrap();
        candidate["hypothesis_bounds"][0]["upper_probability_bps"] = json!(9999);
        let error = evaluate_credal_risk_parts_json(&candidate.to_string(), &oracle.to_string()).unwrap_err();
        assert!(error.contains("prior bounds differ from evaluator-authoritative"));
    }

    #[test]
    fn candidate_cannot_weaken_authoritative_action_policy() {
        let mut candidate: Value = serde_json::from_str(CANDIDATE).unwrap();
        let oracle: Value = serde_json::from_str(ORACLE).unwrap();
        candidate["actions"][0]["max_residual_risk_bps"] = json!(9000);
        let error = evaluate_credal_risk_parts_json(&candidate.to_string(), &oracle.to_string()).unwrap_err();
        assert!(error.contains("action policies differ from evaluator-authoritative"));
    }

    #[test]
    fn empty_survivor_probability_support_is_not_zero_risk() {
        let intervals = BTreeMap::from([
            ("hazard".to_owned(), Interval { lower: 0, upper: 5000, nominal: 2500 }),
            ("safe".to_owned(), Interval { lower: 0, upper: 5000, nominal: 2500 }),
            ("outside".to_owned(), Interval { lower: 10_000, upper: 10_000, nominal: 5000 }),
        ]);
        let survivors = BTreeSet::from(["hazard".to_owned(), "safe".to_owned()]);
        let hazards = BTreeSet::from(["hazard".to_owned()]);
        assert!(robust_upper_risk_bps(&intervals, &survivors, &hazards).is_err());
    }

    #[test]
    fn safe_only_but_zero_probability_evidence_support_is_rejected() {
        let intervals = BTreeMap::from([
            ("safe".to_owned(), Interval { lower: 0, upper: 5000, nominal: 0 }),
            ("outside".to_owned(), Interval { lower: 10_000, upper: 10_000, nominal: 10_000 }),
        ]);
        let survivors = BTreeSet::from(["safe".to_owned()]);
        let hazards = BTreeSet::new();
        assert!(robust_upper_risk_bps(&intervals, &survivors, &hazards).is_err());
    }

    #[test]
    fn sequential_acquisition_timestamps_must_be_nondecreasing() {
        let mut candidate: Value = serde_json::from_str(CANDIDATE).unwrap();
        candidate["traces"][0]["acquisitions"][1]["at"] = json!("2026-11-16T09:40:00Z");
        let oracle: Value = serde_json::from_str(ORACLE).unwrap();
        let error = evaluate_credal_risk_parts_json(&candidate.to_string(), &oracle.to_string()).unwrap_err();
        assert!(error.contains("must be nondecreasing"));
    }

    #[test]
    fn future_evidence_is_rejected_without_narrowing_the_hypothesis_set() {
        let mut candidate: Value = serde_json::from_str(CANDIDATE).unwrap();
        let oracle: Value = serde_json::from_str(ORACLE).unwrap();
        candidate["traces"][1]["acquisitions"][0]["at"] = json!("2026-11-16T10:01:00Z");
        let result = evaluate_credal_risk_parts_json(&candidate.to_string(), &oracle.to_string()).unwrap();
        let trace = result["traces"].as_array().unwrap().iter()
            .find(|item| item["trace_id"] == "robust-probe-after-cheap-signal").unwrap();
        assert_eq!(trace["future_evidence_leakage_count"], 1);
        assert_eq!(trace["surviving_hypothesis_count"], 4);
        assert_eq!(trace["worst_case_posterior_risk_bps"], 3000);
    }

    #[test]
    fn unknown_fields_fail_closed() {
        let mut candidate: Value = serde_json::from_str(CANDIDATE).unwrap();
        let oracle: Value = serde_json::from_str(ORACLE).unwrap();
        candidate["options"][0]["untrusted_override"] = json!(true);
        let error = evaluate_credal_risk_parts_json(&candidate.to_string(), &oracle.to_string()).unwrap_err();
        assert!(error.contains("unknown field"));
    }
}
