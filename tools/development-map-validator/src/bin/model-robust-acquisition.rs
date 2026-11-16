//! D6.16 finite model-family evaluator for risk-robust sequential acquisition.
//!
//! Each candidate model is updated independently using history-conditioned likelihoods.
//! An unweighted model family is never averaged. Unsupported models remain in the
//! worst-case envelope. This is an evaluator prototype, not a general POMDP solver.

use chrono::{DateTime, FixedOffset};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::{env, fs, process};

const CLAIM_CEILING: &str = "Synthetic finite model-family evaluator. It assumes supplied model membership, priors, likelihoods, safety labels and cost samples. It does not establish that the true world is in the family, calibrate real sensor likelihoods, prove physical independence, authorize operations, or solve unrestricted adaptive POMDPs.";

#[derive(Clone, PartialEq, Eq)]
struct ActionPolicy {
    irreversible: bool,
    hard_gate: bool,
    max_residual_risk_bps: u64,
}
#[derive(Clone)]
struct CandidateMeasurement {
    available_at: DateTime<FixedOffset>,
    estimate_upper: u64,
}
#[derive(Clone)]
struct OracleMeasurement {
    available_at: DateTime<FixedOffset>,
    costs: BTreeMap<String, Vec<u64>>,
}
#[derive(Clone, PartialEq, Eq)]
struct HistoryEvent {
    measurement_id: String,
    outcome_id: String,
}
#[derive(Clone)]
struct ModelState {
    prior_bps: u64,
    safe_actions: BTreeSet<String>,
}
#[derive(Clone)]
struct LikelihoodRow {
    history: Vec<HistoryEvent>,
    measurement_id: String,
    outcomes: BTreeMap<String, BTreeMap<String, u64>>,
}
#[derive(Clone)]
struct Model {
    model_id: String,
    states: BTreeMap<String, ModelState>,
    likelihoods: Vec<LikelihoodRow>,
}
#[derive(Clone, PartialEq, Eq)]
struct ModelPosterior {
    weights: BTreeMap<String, u64>,
    unsupported: bool,
    reason: Option<String>,
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

fn unique_strings(value: &Value, label: &str) -> Result<BTreeSet<String>, String> {
    let values = value.as_array().ok_or_else(|| format!("{label} must be an array"))?;
    let mut output = BTreeSet::new();
    for (index, item) in values.iter().enumerate() {
        let text = item
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .ok_or_else(|| format!("{label}[{index}] must be a non-empty string"))?;
        if !output.insert(text.to_owned()) {
            return Err(format!("{label} contains duplicate value {text:?}"));
        }
    }
    Ok(output)
}

fn parse_actions(value: &Value, root: &str) -> Result<BTreeMap<String, ActionPolicy>, String> {
    let items = required_value(value, "actions", root)?
        .as_array()
        .ok_or_else(|| format!("{root}.actions must be an array"))?;
    if items.is_empty() {
        return Err(format!("{root}.actions must not be empty"));
    }

    let mut output = BTreeMap::new();
    for (index, item) in items.iter().enumerate() {
        let label = format!("{root}.actions[{index}]");
        allowed_keys(
            item,
            &["action_id", "irreversible", "hard_gate", "max_residual_risk_bps"],
            &label,
        )?;
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
        let policy = ActionPolicy {
            irreversible,
            hard_gate,
            max_residual_risk_bps,
        };
        if output.insert(id.clone(), policy).is_some() {
            return Err(format!("duplicate action_id {id:?}"));
        }
    }
    Ok(output)
}

fn parse_candidate_measurements(
    value: &Value,
) -> Result<BTreeMap<String, CandidateMeasurement>, String> {
    let items = required_value(value, "measurements", "candidate_view")?
        .as_array()
        .ok_or("candidate_view.measurements must be an array")?;
    if items.is_empty() || items.len() > 16 {
        return Err("candidate_view.measurements must contain 1..=16 items".to_owned());
    }

    let mut output = BTreeMap::new();
    for (index, item) in items.iter().enumerate() {
        let label = format!("candidate_view.measurements[{index}]");
        allowed_keys(
            item,
            &["measurement_id", "available_at", "estimated_cost_upper_micros"],
            &label,
        )?;
        let id = required_str(item, "measurement_id", &label)?;
        let measurement = CandidateMeasurement {
            available_at: parse_time(item, "available_at", &label)?,
            estimate_upper: required_u64(item, "estimated_cost_upper_micros", &label)?,
        };
        if output.insert(id.clone(), measurement).is_some() {
            return Err(format!("duplicate measurement_id {id:?}"));
        }
    }
    Ok(output)
}

fn parse_cost_samples(value: &Value, label: &str) -> Result<BTreeMap<String, Vec<u64>>, String> {
    allowed_keys(value, &["cold", "warm", "retry"], label)?;
    let mut output = BTreeMap::new();
    for mode in ["cold", "warm", "retry"] {
        let items = required_value(value, mode, label)?
            .as_array()
            .ok_or_else(|| format!("{label}.{mode} must be an array"))?;
        if items.is_empty() {
            return Err(format!("{label}.{mode} must not be empty; unknown cost is not zero"));
        }
        let mut samples = Vec::new();
        for (index, item) in items.iter().enumerate() {
            let amount = item
                .as_u64()
                .filter(|amount| *amount > 0)
                .ok_or_else(|| format!("{label}.{mode}[{index}] must be a positive integer"))?;
            samples.push(amount);
        }
        output.insert(mode.to_owned(), samples);
    }
    Ok(output)
}

fn nearest_rank_p95(samples: &[u64]) -> u64 {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let rank = (sorted.len() * 95 + 99) / 100;
    sorted[rank.saturating_sub(1)]
}

fn parse_oracle_measurements(
    oracle: &Value,
    candidate: &BTreeMap<String, CandidateMeasurement>,
) -> Result<BTreeMap<String, OracleMeasurement>, String> {
    let items = required_value(oracle, "measurements", "oracle")?
        .as_array()
        .ok_or("oracle.measurements must be an array")?;
    let mut output = BTreeMap::new();

    for (index, item) in items.iter().enumerate() {
        let label = format!("oracle.measurements[{index}]");
        allowed_keys(
            item,
            &["measurement_id", "available_at", "cost_samples_micros"],
            &label,
        )?;
        let id = required_str(item, "measurement_id", &label)?;
        let available_at = parse_time(item, "available_at", &label)?;
        let costs = parse_cost_samples(
            required_value(item, "cost_samples_micros", &label)?,
            &format!("{label}.cost_samples_micros"),
        )?;
        let visible = candidate
            .get(&id)
            .ok_or_else(|| format!("oracle measurement {id:?} is not candidate-visible"))?;
        if visible.available_at != available_at {
            return Err(format!("candidate/oracle availability mismatch for measurement {id:?}"));
        }
        if output
            .insert(id.clone(), OracleMeasurement { available_at, costs })
            .is_some()
        {
            return Err(format!("duplicate oracle measurement_id {id:?}"));
        }
    }

    if output.len() != candidate.len() {
        return Err("candidate and oracle measurement sets must match exactly".to_owned());
    }
    Ok(output)
}

fn parse_history(
    value: &Value,
    label: &str,
    measurement_ids: &BTreeSet<String>,
) -> Result<Vec<HistoryEvent>, String> {
    let items = value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?;
    if items.len() > 2 {
        return Err(format!("{label} exceeds the depth-2 limit"));
    }

    let mut output = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let item_label = format!("{label}[{index}]");
        allowed_keys(item, &["measurement_id", "outcome_id"], &item_label)?;
        let measurement_id = required_str(item, "measurement_id", &item_label)?;
        if !measurement_ids.contains(&measurement_id) {
            return Err(format!(
                "{item_label} references unknown measurement {measurement_id:?}"
            ));
        }
        let outcome_id = required_str(item, "outcome_id", &item_label)?;
        output.push(HistoryEvent {
            measurement_id,
            outcome_id,
        });
    }
    Ok(output)
}

fn parse_models(
    oracle: &Value,
    measurement_ids: &BTreeSet<String>,
    action_ids: &BTreeSet<String>,
) -> Result<Vec<Model>, String> {
    let items = required_value(oracle, "models", "oracle")?
        .as_array()
        .ok_or("oracle.models must be an array")?;
    if items.is_empty() || items.len() > 16 {
        return Err("oracle.models must contain 1..=16 models".to_owned());
    }

    let mut models = Vec::new();
    let mut model_ids = BTreeSet::new();

    for (model_index, item) in items.iter().enumerate() {
        let label = format!("oracle.models[{model_index}]");
        allowed_keys(item, &["model_id", "states", "likelihoods"], &label)?;
        let model_id = required_str(item, "model_id", &label)?;
        if !model_ids.insert(model_id.clone()) {
            return Err(format!("duplicate model_id {model_id:?}"));
        }

        let state_items = required_value(item, "states", &label)?
            .as_array()
            .ok_or_else(|| format!("{label}.states must be an array"))?;
        if state_items.is_empty() || state_items.len() > 16 {
            return Err(format!("{label}.states must contain 1..=16 states"));
        }

        let mut states = BTreeMap::new();
        let mut prior_sum = 0u64;
        for (state_index, state) in state_items.iter().enumerate() {
            let state_label = format!("{label}.states[{state_index}]");
            allowed_keys(state, &["state_id", "prior_bps", "safe_actions"], &state_label)?;
            let state_id = required_str(state, "state_id", &state_label)?;
            let prior_bps = required_u64(state, "prior_bps", &state_label)?;
            if prior_bps > 10_000 {
                return Err(format!("{state_label}.prior_bps must be <= 10000"));
            }
            prior_sum = prior_sum
                .checked_add(prior_bps)
                .ok_or("prior probability sum overflow")?;
            let safe_actions = unique_strings(
                required_value(state, "safe_actions", &state_label)?,
                &format!("{state_label}.safe_actions"),
            )?;
            if safe_actions.iter().any(|action| !action_ids.contains(action)) {
                return Err(format!("{state_label}.safe_actions references unknown action"));
            }
            if states
                .insert(state_id.clone(), ModelState { prior_bps, safe_actions })
                .is_some()
            {
                return Err(format!("{label} contains duplicate state_id {state_id:?}"));
            }
        }
        if prior_sum != 10_000 {
            return Err(format!(
                "{label} prior weights must sum to 10000 bps, got {prior_sum}"
            ));
        }

        let row_items = required_value(item, "likelihoods", &label)?
            .as_array()
            .ok_or_else(|| format!("{label}.likelihoods must be an array"))?;
        if row_items.is_empty() {
            return Err(format!("{label}.likelihoods must not be empty"));
        }

        let mut likelihoods = Vec::new();
        let mut row_keys = BTreeSet::new();
        for (row_index, row) in row_items.iter().enumerate() {
            let row_label = format!("{label}.likelihoods[{row_index}]");
            allowed_keys(row, &["history", "measurement_id", "outcomes"], &row_label)?;
            let history = parse_history(
                required_value(row, "history", &row_label)?,
                &format!("{row_label}.history"),
                measurement_ids,
            )?;
            let measurement_id = required_str(row, "measurement_id", &row_label)?;
            if !measurement_ids.contains(&measurement_id) {
                return Err(format!(
                    "{row_label} references unknown measurement {measurement_id:?}"
                ));
            }

            let row_key = format!(
                "{}|{}",
                history
                    .iter()
                    .map(|event| format!("{}={}", event.measurement_id, event.outcome_id))
                    .collect::<Vec<_>>()
                    .join(","),
                measurement_id
            );
            if !row_keys.insert(row_key) {
                return Err(format!("{label} has duplicate history/measurement likelihood rows"));
            }

            let outcome_items = required_value(row, "outcomes", &row_label)?
                .as_array()
                .ok_or_else(|| format!("{row_label}.outcomes must be an array"))?;
            if outcome_items.is_empty() {
                return Err(format!("{row_label}.outcomes must not be empty"));
            }

            let mut outcomes = BTreeMap::new();
            for (outcome_index, outcome) in outcome_items.iter().enumerate() {
                let outcome_label = format!("{row_label}.outcomes[{outcome_index}]");
                allowed_keys(
                    outcome,
                    &["outcome_id", "state_likelihoods"],
                    &outcome_label,
                )?;
                let outcome_id = required_str(outcome, "outcome_id", &outcome_label)?;
                let likelihood_items = required_value(
                    outcome,
                    "state_likelihoods",
                    &outcome_label,
                )?
                .as_array()
                .ok_or_else(|| {
                    format!("{outcome_label}.state_likelihoods must be an array")
                })?;
                let mut state_likelihoods = BTreeMap::new();
                for (likelihood_index, entry) in likelihood_items.iter().enumerate() {
                    let entry_label =
                        format!("{outcome_label}.state_likelihoods[{likelihood_index}]");
                    allowed_keys(entry, &["state_id", "probability_bps"], &entry_label)?;
                    let state_id = required_str(entry, "state_id", &entry_label)?;
                    if !states.contains_key(&state_id) {
                        return Err(format!("{entry_label} references unknown state {state_id:?}"));
                    }
                    let probability_bps = required_u64(entry, "probability_bps", &entry_label)?;
                    if probability_bps > 10_000 {
                        return Err(format!("{entry_label}.probability_bps must be <= 10000"));
                    }
                    if state_likelihoods.insert(state_id.clone(), probability_bps).is_some() {
                        return Err(format!(
                            "{outcome_label} duplicates state likelihood {state_id:?}"
                        ));
                    }
                }
                if state_likelihoods.len() != states.len() {
                    return Err(format!(
                        "{outcome_label} must define likelihood for every model state"
                    ));
                }
                if outcomes
                    .insert(outcome_id.clone(), state_likelihoods)
                    .is_some()
                {
                    return Err(format!("{row_label} contains duplicate outcome_id {outcome_id:?}"));
                }
            }

            for state_id in states.keys() {
                let sum = outcomes.values().map(|values| values[state_id]).sum::<u64>();
                if sum != 10_000 {
                    return Err(format!(
                        "{row_label} likelihoods for state {state_id:?} must sum to 10000, got {sum}"
                    ));
                }
            }

            likelihoods.push(LikelihoodRow {
                history,
                measurement_id,
                outcomes,
            });
        }

        models.push(Model {
            model_id,
            states,
            likelihoods,
        });
    }

    Ok(models)
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn reduce_weights(weights: &mut BTreeMap<String, u64>) {
    let divisor = weights
        .values()
        .copied()
        .filter(|value| *value > 0)
        .fold(0, gcd);
    if divisor > 1 {
        for value in weights.values_mut() {
            *value /= divisor;
        }
    }
}

fn model_risk(
    model: &Model,
    posterior: &ModelPosterior,
    action_id: &str,
) -> (u64, u64, u64, bool) {
    if posterior.unsupported {
        return (10_000, 0, 0, false);
    }
    let total = posterior.weights.values().copied().sum::<u64>();
    if total == 0 {
        return (10_000, 0, 0, false);
    }
    let unsafe_mass = posterior
        .weights
        .iter()
        .filter(|(state_id, _)| !model.states[*state_id].safe_actions.contains(action_id))
        .map(|(_, weight)| *weight)
        .sum::<u64>();
    let risk = (unsafe_mass * 10_000 + total - 1) / total;
    (risk, unsafe_mass, total, true)
}

fn measurement_cost(option: &OracleMeasurement, mode: &str) -> Result<u64, String> {
    let samples = option
        .costs
        .get(mode)
        .ok_or_else(|| format!("missing {mode} cost samples"))?;
    Ok(nearest_rank_p95(samples))
}

fn evaluate_trace(
    trace: &Value,
    models: &[Model],
    candidate_measurements: &BTreeMap<String, CandidateMeasurement>,
    oracle_measurements: &BTreeMap<String, OracleMeasurement>,
    actions: &BTreeMap<String, ActionPolicy>,
    cutoff: &DateTime<FixedOffset>,
    budget_micros: u64,
) -> Result<Value, String> {
    let trace_id = required_str(trace, "trace_id", "trace")?;
    let label = format!("trace {trace_id}");
    allowed_keys(trace, &["trace_id", "acquisitions", "decision"], &label)?;
    let acquisitions = required_value(trace, "acquisitions", &label)?
        .as_array()
        .ok_or_else(|| format!("{label}.acquisitions must be an array"))?;
    if acquisitions.len() > 2 {
        return Err(format!("{label} exceeds the depth-2 acquisition limit"));
    }

    let decision = required_value(trace, "decision", &label)?;
    allowed_keys(
        decision,
        &["kind", "action_id", "claimed_worst_case_risk_bps"],
        &format!("{label}.decision"),
    )?;
    let decision_kind = required_str(decision, "kind", &format!("{label}.decision"))?;
    if !["act", "defer"].contains(&decision_kind.as_str()) {
        return Err(format!("{label}.decision.kind must be act or defer"));
    }
    let action_id = required_str(decision, "action_id", &format!("{label}.decision"))?;
    let policy = actions
        .get(&action_id)
        .ok_or_else(|| format!("{label} references unknown action {action_id:?}"))?;
    let claimed_risk = match decision.get("claimed_worst_case_risk_bps") {
        None | Some(Value::Null) => None,
        Some(Value::Number(number)) => number.as_u64(),
        _ => None,
    };
    if decision.get("claimed_worst_case_risk_bps").is_some() && claimed_risk.is_none() {
        return Err(format!(
            "{label}.decision.claimed_worst_case_risk_bps must be integer or null"
        ));
    }
    if claimed_risk.is_some_and(|risk| risk > 10_000) {
        return Err(format!(
            "{label}.decision.claimed_worst_case_risk_bps must be <= 10000"
        ));
    }

    let mut posteriors: BTreeMap<String, ModelPosterior> = models
        .iter()
        .map(|model| {
            let mut weights: BTreeMap<String, u64> = model
                .states
                .iter()
                .map(|(id, state)| (id.clone(), state.prior_bps))
                .collect();
            reduce_weights(&mut weights);
            (
                model.model_id.clone(),
                ModelPosterior {
                    weights,
                    unsupported: false,
                    reason: None,
                },
            )
        })
        .collect();

    let mut history: Vec<HistoryEvent> = Vec::new();
    let mut uses: BTreeMap<String, usize> = BTreeMap::new();
    let mut previous_at: Option<DateTime<FixedOffset>> = None;
    let mut total_cost = 0u64;
    let mut redundant_count = 0usize;
    let mut future_count = 0usize;
    let mut unavailable_count = 0usize;
    let mut unsupported_events = 0usize;
    let mut zero_support_events = 0usize;
    let mut estimate_undercoverage = 0usize;
    let mut budget_overrun = false;
    let mut step_reports = Vec::new();

    for (index, step) in acquisitions.iter().enumerate() {
        let step_label = format!("{label}.acquisitions[{index}]");
        allowed_keys(
            step,
            &["measurement_id", "outcome_id", "mode", "at"],
            &step_label,
        )?;
        let measurement_id = required_str(step, "measurement_id", &step_label)?;
        let outcome_id = required_str(step, "outcome_id", &step_label)?;
        let mode = required_str(step, "mode", &step_label)?;
        if !["cold", "warm", "retry"].contains(&mode.as_str()) {
            return Err(format!("{step_label}.mode must be cold, warm or retry"));
        }
        let at = parse_time(step, "at", &step_label)?;
        if previous_at.as_ref().is_some_and(|previous| &at < previous) {
            return Err(format!("{step_label}.at must be nondecreasing"));
        }
        previous_at = Some(at.clone());

        let visible = candidate_measurements
            .get(&measurement_id)
            .ok_or_else(|| format!("{step_label} references unknown measurement {measurement_id:?}"))?;
        let oracle = oracle_measurements
            .get(&measurement_id)
            .ok_or_else(|| format!("oracle cost for measurement {measurement_id:?} is missing"))?;
        let prior_uses = *uses.get(&measurement_id).unwrap_or(&0);
        if (prior_uses == 0 && mode != "cold") || (prior_uses > 0 && mode == "cold") {
            return Err(format!("{step_label}.mode does not match first-use/repeat state"));
        }

        let cost = measurement_cost(oracle, &mode)?;
        total_cost = total_cost.checked_add(cost).ok_or("acquisition cost overflow")?;
        *uses.entry(measurement_id.clone()).or_insert(0) += 1;
        if visible.estimate_upper < cost {
            estimate_undercoverage += 1;
        }

        let future = at > *cutoff;
        let premature = at < visible.available_at || at < oracle.available_at;
        if future {
            future_count += 1;
        }
        if premature {
            unavailable_count += 1;
        }
        if total_cost > budget_micros {
            budget_overrun = true;
        }

        let mut disposition = "applied".to_owned();
        let mut newly_unsupported = Vec::new();
        let before_posteriors = posteriors.clone();

        if future {
            disposition = "future_evidence_rejected".to_owned();
        } else if premature {
            disposition = "not_yet_available_rejected".to_owned();
        } else if total_cost > budget_micros {
            disposition = "budget_exceeded_not_acquired".to_owned();
        } else {
            for model in models {
                let posterior = posteriors
                    .get_mut(&model.model_id)
                    .ok_or("model posterior disappeared")?;
                if posterior.unsupported {
                    continue;
                }

                let row = model
                    .likelihoods
                    .iter()
                    .find(|row| row.measurement_id == measurement_id && row.history == history);

                let Some(row) = row else {
                    posterior.unsupported = true;
                    posterior.reason = Some(format!("no_likelihood_row_for:{measurement_id}"));
                    newly_unsupported.push(model.model_id.clone());
                    continue;
                };

                let Some(likelihoods) = row.outcomes.get(&outcome_id) else {
                    posterior.unsupported = true;
                    posterior.reason = Some(format!("outcome_not_in_model:{outcome_id}"));
                    newly_unsupported.push(model.model_id.clone());
                    continue;
                };

                let mut updated = BTreeMap::new();
                let mut mass = 0u64;
                for (state_id, previous_weight) in &posterior.weights {
                    let likelihood = likelihoods.get(state_id).copied().ok_or_else(|| {
                        format!("model {} outcome omits state {state_id}", model.model_id)
                    })?;
                    let weight = previous_weight
                        .checked_mul(likelihood)
                        .ok_or("posterior weight multiplication overflow")?;
                    mass = mass.checked_add(weight).ok_or("posterior mass overflow")?;
                    updated.insert(state_id.clone(), weight);
                }

                if mass == 0 {
                    posterior.unsupported = true;
                    posterior.reason = Some("zero_conditioning_mass".to_owned());
                    newly_unsupported.push(model.model_id.clone());
                } else {
                    reduce_weights(&mut updated);
                    posterior.weights = updated;
                }
            }

            if !newly_unsupported.is_empty() {
                unsupported_events += 1;
                let supported_after_update = posteriors.values()
                    .filter(|posterior| !posterior.unsupported)
                    .count();
                if supported_after_update == 0 {
                    zero_support_events += 1;
                }
            }
            history.push(HistoryEvent {
                measurement_id: measurement_id.clone(),
                outcome_id: outcome_id.clone(),
            });
            if models.iter().all(|model| {
                let before = before_posteriors.get(&model.model_id);
                let after = posteriors.get(&model.model_id);
                before.zip(after).is_some_and(|(left, right)| {
                    left.weights == right.weights
                        && left.unsupported == right.unsupported
                        && left.reason == right.reason
                })
            }) {
                redundant_count += 1;
            }
        }

        let supported_after = models.len()
            - posteriors.values().filter(|posterior| posterior.unsupported).count();
        let worst_after = if disposition == "applied" {
            let mut worst = 0u64;
            for model in models {
                let posterior = posteriors
                    .get(&model.model_id)
                    .ok_or("model posterior disappeared")?;
                worst = worst.max(model_risk(model, posterior, &action_id).0);
            }
            Some(worst)
        } else {
            None
        };

        step_reports.push(json!({
            "measurement_id": measurement_id,
            "outcome_id": outcome_id,
            "mode": mode,
            "cost_upper_micros": cost,
            "candidate_estimate_upper_micros": visible.estimate_upper,
            "estimate_undercovered": visible.estimate_upper < cost,
            "history_depth": history.len(),
            "newly_unsupported_model_ids": newly_unsupported,
            "supported_models_after": supported_after,
            "model_count": models.len(),
            "worst_case_risk_bps_after": worst_after,
            "disposition": disposition
        }));
    }

    let mut model_results = Vec::new();
    let mut supported_count = 0usize;
    let mut worst_case_risk = 0u64;
    let mut worst_model_id = String::new();
    let mut minimum_model_risk = 10_000u64;
    let mut hard_gate_safe = true;

    for model in models {
        let posterior = posteriors
            .get(&model.model_id)
            .ok_or("model posterior disappeared")?;
        let (risk, unsafe_weight, total_weight, supported) =
            model_risk(model, posterior, &action_id);
        if supported {
            supported_count += 1;
        }
        if risk > worst_case_risk || worst_model_id.is_empty() {
            worst_case_risk = risk;
            worst_model_id = model.model_id.clone();
        }
        minimum_model_risk = minimum_model_risk.min(risk);
        if !supported || unsafe_weight > 0 {
            hard_gate_safe = false;
        }
        model_results.push(json!({
            "model_id": model.model_id,
            "supported": supported,
            "unsupported_reason": posterior.reason,
            "posterior_weight_mass": total_weight,
            "unsafe_weight_mass": unsafe_weight,
            "risk_bps": risk,
            "state_weight_masses": posterior.weights
        }));
    }

    let model_support_coverage_bps =
        (supported_count as u64 * 10_000) / models.len() as u64;
    let all_models_supported = supported_count == models.len();
    let action_justified = all_models_supported
        && if policy.hard_gate {
            hard_gate_safe
        } else {
            worst_case_risk <= policy.max_residual_risk_bps
        };

    let unsupported_action = decision_kind == "act" && !action_justified;
    let hard_gate_violation = decision_kind == "act"
        && policy.hard_gate
        && (!all_models_supported || !hard_gate_safe);
    let risk_policy_violation = decision_kind == "act"
        && !policy.hard_gate
        && (!all_models_supported || worst_case_risk > policy.max_residual_risk_bps);
    let unnecessary_abstention = decision_kind == "defer" && action_justified;
    let correctly_deferred = decision_kind == "defer" && !action_justified;
    let risk_understatement = claimed_risk.is_some_and(|claimed| claimed < worst_case_risk);
    let model_risk_spread_bps = worst_case_risk.saturating_sub(minimum_model_risk);

    Ok(json!({
        "trace_id": trace_id,
        "decision_kind": decision_kind,
        "action_id": action_id,
        "irreversible": policy.irreversible,
        "hard_gate": policy.hard_gate,
        "max_residual_risk_bps": policy.max_residual_risk_bps,
        "aggregation": "unweighted_worst_case_over_models",
        "model_count": models.len(),
        "supported_model_count": supported_count,
        "model_support_coverage_bps": model_support_coverage_bps,
        "all_models_supported": all_models_supported,
        "worst_case_model_id": worst_model_id,
        "minimum_model_risk_bps": minimum_model_risk,
        "worst_case_risk_bps": worst_case_risk,
        "model_risk_spread_bps": model_risk_spread_bps,
        "action_justified": action_justified,
        "unsupported_action": unsupported_action,
        "hard_gate_violation": hard_gate_violation,
        "risk_policy_violation": risk_policy_violation,
        "unnecessary_abstention": unnecessary_abstention,
        "correctly_deferred": correctly_deferred,
        "claimed_worst_case_risk_bps": claimed_risk,
        "risk_understatement": risk_understatement,
        "acquisition_count": acquisitions.len(),
        "unique_measurement_count": uses.len(),
        "redundant_measurement_count": redundant_count,
        "unsupported_model_observation_events": unsupported_events,
        "zero_support_model_observation_events": zero_support_events,
        "future_evidence_leakage_count": future_count,
        "not_yet_available_count": unavailable_count,
        "cost_estimate_undercoverage_count": estimate_undercoverage,
        "total_cost_upper_micros": total_cost,
        "budget_micros": budget_micros,
        "budget_overrun": budget_overrun,
        "acquisition_steps": step_reports,
        "model_results": model_results,
        "final_history": history.iter().map(|event| json!({
            "measurement_id": event.measurement_id,
            "outcome_id": event.outcome_id
        })).collect::<Vec<_>>()
    }))
}

pub fn evaluate_model_robust_acquisition_json(
    candidate_raw: &str,
    oracle_raw: &str,
) -> Result<Value, String> {
    let candidate: Value = serde_json::from_str(candidate_raw)
        .map_err(|error| format!("invalid candidate-view JSON: {error}"))?;
    let oracle: Value = serde_json::from_str(oracle_raw)
        .map_err(|error| format!("invalid evaluator-oracle JSON: {error}"))?;

    allowed_keys(
        &candidate,
        &["scenario_id", "decision_cutoff", "budget_micros", "actions", "measurements", "traces"],
        "candidate_view",
    )?;
    allowed_keys(
        &oracle,
        &["decision_cutoff", "budget_micros", "actions", "measurements", "models"],
        "oracle",
    )?;

    let scenario_id = required_str(&candidate, "scenario_id", "candidate_view")?;
    let cutoff = parse_time(&candidate, "decision_cutoff", "candidate_view")?;
    let oracle_cutoff = parse_time(&oracle, "decision_cutoff", "oracle")?;
    if cutoff != oracle_cutoff {
        return Err("candidate decision cutoff differs from evaluator-authoritative cutoff".to_owned());
    }

    let budget_micros = required_u64(&candidate, "budget_micros", "candidate_view")?;
    let oracle_budget = required_u64(&oracle, "budget_micros", "oracle")?;
    if budget_micros != oracle_budget {
        return Err("candidate budget differs from evaluator-authoritative budget".to_owned());
    }

    let actions = parse_actions(&candidate, "candidate_view")?;
    let authoritative_actions = parse_actions(&oracle, "oracle")?;
    if actions != authoritative_actions {
        return Err("candidate action policies differ from evaluator-authoritative action policies".to_owned());
    }
    let action_ids = actions.keys().cloned().collect::<BTreeSet<_>>();

    let candidate_measurements = parse_candidate_measurements(&candidate)?;
    let measurement_ids = candidate_measurements.keys().cloned().collect::<BTreeSet<_>>();
    let oracle_measurements = parse_oracle_measurements(&oracle, &candidate_measurements)?;
    let models = parse_models(&oracle, &measurement_ids, &action_ids)?;

    let traces = required_value(&candidate, "traces", "candidate_view")?
        .as_array()
        .ok_or("candidate_view.traces must be an array")?;
    if traces.is_empty() || traces.len() > 64 {
        return Err("candidate_view.traces must contain 1..=64 traces".to_owned());
    }

    let mut seen = BTreeSet::new();
    let mut reports = Vec::new();
    for trace in traces {
        let trace_id = required_str(trace, "trace_id", "trace")?;
        if !seen.insert(trace_id.clone()) {
            return Err(format!("duplicate trace_id {trace_id:?}"));
        }
        reports.push(evaluate_trace(
            trace,
            &models,
            &candidate_measurements,
            &oracle_measurements,
            &actions,
            &cutoff,
            budget_micros,
        )?);
    }

    let count = |field: &str| {
        reports
            .iter()
            .filter(|report| report[field].as_bool() == Some(true))
            .count()
    };
    let sum = |field: &str| {
        reports
            .iter()
            .map(|report| report[field].as_u64().unwrap_or(0))
            .sum::<u64>()
    };

    Ok(json!({
        "scenario_id": scenario_id,
        "adjudication_status": "synthetic_model_robust_known_answer_evaluator_only",
        "aggregation": "unweighted_worst_case_over_models",
        "model_count": models.len(),
        "measurement_count": candidate_measurements.len(),
        "action_count": actions.len(),
        "trace_count": reports.len(),
        "summary": {
            "unsupported_action_trace_count": count("unsupported_action"),
            "hard_gate_violation_trace_count": count("hard_gate_violation"),
            "risk_policy_violation_trace_count": count("risk_policy_violation"),
            "unnecessary_abstention_trace_count": count("unnecessary_abstention"),
            "budget_overrun_trace_count": count("budget_overrun"),
            "future_evidence_leakage_events": sum("future_evidence_leakage_count"),
            "unsupported_model_observation_events": sum("unsupported_model_observation_events"),
            "zero_support_model_observation_events": sum("zero_support_model_observation_events"),
            "risk_understatement_trace_count": count("risk_understatement")
        },
        "traces": reports,
        "claim_ceiling": CLAIM_CEILING
    }))
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 || args[0] == "--help" || args[0] == "-h" {
        eprintln!("Usage: model-robust-acquisition <candidate-view.json> <evaluator-oracle.json>");
        eprintln!("D6.16 evaluator-only finite model-family scorer; keep the oracle in a separate protected workspace.");
        process::exit(if args.len() == 2 { 0 } else { 2 });
    }

    let candidate = match fs::read_to_string(&args[0]) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("ERROR D6.16-CANDIDATE-IO-001 {}: {error}", args[0]);
            process::exit(2);
        }
    };
    let oracle = match fs::read_to_string(&args[1]) {
        Ok(text) => text,
        Err(error) => {
            eprintln!("ERROR D6.16-ORACLE-IO-001 {}: {error}", args[1]);
            process::exit(2);
        }
    };

    match evaluate_model_robust_acquisition_json(&candidate, &oracle) {
        Ok(report) => match serde_json::to_string_pretty(&report) {
            Ok(output) => println!("{output}"),
            Err(error) => {
                eprintln!("ERROR D6.16-JSON-001: {error}");
                process::exit(2);
            }
        },
        Err(error) => {
            eprintln!("ERROR D6.16-MODEL-ROBUST-001: {error}");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CANDIDATE: &str = include_str!("../../../../docs/development-map/fixtures/egddm-model-robust-acquisition-candidate-v0.1.json");
    const ORACLE: &str = include_str!("../../../../docs/development-map/fixtures/egddm-model-robust-acquisition-oracle-v0.1.json");
    const EXPECTED: &str = include_str!("../../../../docs/development-map/fixtures/egddm-model-robust-acquisition-v0.1.expected.json");

    fn report() -> Value {
        evaluate_model_robust_acquisition_json(CANDIDATE, ORACLE)
            .expect("D6.16 fixture evaluates")
    }

    fn trace(value: &Value, id: &str) -> Value {
        value["traces"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["trace_id"] == id)
            .expect("trace exists")
            .clone()
    }

    #[test]
    fn fixture_matches_complete_known_answer_report() {
        let actual = report();
        let expected: Value = serde_json::from_str(EXPECTED).expect("expected JSON");
        assert_eq!(actual, expected);
    }

    #[test]
    fn unweighted_model_family_uses_worst_case_not_model_average() {
        let result = trace(&report(), "model-average-trap");
        assert_eq!(result["minimum_model_risk_bps"], 2_000);
        assert_eq!(result["worst_case_risk_bps"], 4_000);
        assert_eq!(result["action_justified"], false);
        assert_eq!(result["unsupported_action"], true);
        assert_eq!(result["risk_policy_violation"], true);
        assert_eq!(result["risk_understatement"], true);
    }

    #[test]
    fn correlated_dashboard_does_not_add_independent_evidence() {
        let value = report();
        let primary = trace(&value, "primary-only-release");
        let dashboard = trace(&value, "shared-dashboard-release");
        assert_eq!(primary["worst_case_risk_bps"], dashboard["worst_case_risk_bps"]);
        assert_eq!(primary["model_results"], dashboard["model_results"]);
        assert_eq!(dashboard["redundant_measurement_count"], 1);
    }

    #[test]
    fn independent_reference_restores_hard_gate_support() {
        let result = trace(&report(), "independent-reference-release");
        assert_eq!(result["all_models_supported"], true);
        assert_eq!(result["worst_case_risk_bps"], 0);
        assert_eq!(result["action_justified"], true);
    }

    #[test]
    fn noisy_measurement_preserves_nonzero_risk_but_supports_bounded_probe() {
        let result = trace(&report(), "noisy-bounded-probe");
        assert_eq!(result["worst_case_risk_bps"], 2_174);
        assert_eq!(result["max_residual_risk_bps"], 3_500);
        assert_eq!(result["action_justified"], true);
        assert_eq!(result["risk_understatement"], true);
    }

    #[test]
    fn partially_supported_model_is_not_silently_dropped() {
        let result = trace(&report(), "partial-model-support");
        assert_eq!(result["supported_model_count"], 1);
        assert_eq!(result["model_support_coverage_bps"], 5_000);
        assert_eq!(result["worst_case_risk_bps"], 10_000);
        assert_eq!(result["action_justified"], false);
    }

    #[test]
    fn out_of_model_observation_remains_explicitly_unsupported() {
        let result = trace(&report(), "out-of-model-observation");
        assert_eq!(result["supported_model_count"], 0);
        assert_eq!(result["zero_support_model_observation_events"], 1);
        assert_eq!(result["action_justified"], false);
    }

    #[test]
    fn always_abstain_fails_on_positive_control_after_sufficient_evidence() {
        let result = trace(&report(), "always-abstain-positive-control");
        assert_eq!(result["action_justified"], true);
        assert_eq!(result["unnecessary_abstention"], true);
    }

    #[test]
    fn future_measurement_cannot_change_the_prior_decision() {
        let result = trace(&report(), "future-reference");
        assert_eq!(result["future_evidence_leakage_count"], 1);
        assert_eq!(result["final_history"].as_array().unwrap().len(), 1);
        assert_eq!(result["action_justified"], false);
    }

    #[test]
    fn candidate_cannot_weaken_authoritative_action_policy() {
        let mut candidate: Value = serde_json::from_str(CANDIDATE).unwrap();
        candidate["actions"][0]["irreversible"] = json!(false);
        candidate["actions"][0]["hard_gate"] = json!(false);
        candidate["actions"][0]["max_residual_risk_bps"] = json!(10_000);
        let error = evaluate_model_robust_acquisition_json(&candidate.to_string(), ORACLE)
            .unwrap_err();
        assert!(error.contains("evaluator-authoritative action policies"));
    }

    #[test]
    fn invalid_likelihood_rows_must_normalize_per_state() {
        let mut oracle: Value = serde_json::from_str(ORACLE).unwrap();
        oracle["models"][0]["likelihoods"][0]["outcomes"][0]["state_likelihoods"][0]["probability_bps"] = json!(9_999);
        let error = evaluate_model_robust_acquisition_json(CANDIDATE, &oracle.to_string())
            .unwrap_err();
        assert!(error.contains("must sum to 10000"));
    }

    #[test]
    fn unknown_cost_is_not_treated_as_zero() {
        let mut oracle: Value = serde_json::from_str(ORACLE).unwrap();
        oracle["measurements"][0]["cost_samples_micros"]["retry"] = json!([]);
        let error = evaluate_model_robust_acquisition_json(CANDIDATE, &oracle.to_string())
            .unwrap_err();
        assert!(error.contains("must not be empty"));
    }
}
