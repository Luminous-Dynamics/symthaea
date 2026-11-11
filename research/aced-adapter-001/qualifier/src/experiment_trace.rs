use crate::power_design::TwoSidedMeanPowerDesign;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_VERSION: &str = "aced-experiment-value-trace-v0.1";
const GENESIS_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const TOLERANCE: f64 = 1e-9;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TraceReport {
    pub event_count: usize,
    pub proposal_count: usize,
    pub observation_count: usize,
    pub scored_observation_count: usize,
    pub mean_brier_score: Option<f64>,
    /// None when any observed outcome had zero forecast probability (infinite log loss).
    pub mean_log_loss: Option<f64>,
    pub zero_probability_outcome_count: usize,
    pub stopped: bool,
    pub terminal_sha256: String,
}

#[derive(Clone, Debug)]
struct ProposalPrediction {
    experiment_id: String,
    outcome_probabilities: BTreeMap<String, f64>,
}

#[derive(Clone, Copy, Debug)]
struct ObservationScores {
    brier_score: f64,
    log_loss: Option<f64>,
}

fn text<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key)?.as_str().filter(|item| !item.trim().is_empty())
}

fn finite_number(value: &Value, key: &str) -> Option<f64> {
    value.get(key)?.as_f64().filter(|number| number.is_finite())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn exact_fields(value: &Value, expected: &[&str]) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.len() == expected.len() && expected.iter().all(|key| object.contains_key(*key))
}

fn write_canonical_json(value: &Value, output: &mut Vec<u8>) -> Result<(), String> {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            serde_json::to_writer(output, value)
                .map_err(|error| format!("canonical JSON scalar serialization failed: {error}"))?;
        }
        Value::Array(items) => {
            output.push(b'[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                write_canonical_json(item, output)?;
            }
            output.push(b']');
        }
        Value::Object(fields) => {
            let mut keys = fields.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            output.push(b'{');
            for (index, key) in keys.iter().enumerate() {
                if index > 0 {
                    output.push(b',');
                }
                serde_json::to_writer(&mut *output, key)
                    .map_err(|error| format!("canonical JSON key serialization failed: {error}"))?;
                output.push(b':');
                write_canonical_json(
                    fields
                        .get(key.as_str())
                        .ok_or_else(|| "canonical JSON key disappeared".to_string())?,
                    output,
                )?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

/// SHA-256 over canonical UTF-8 JSON (recursive key sort, array order preserved).
pub fn canonical_json_sha256(value: &Value) -> Result<String, String> {
    let mut bytes = Vec::new();
    write_canonical_json(value, &mut bytes)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_string_array(value: &Value, field: &str) -> Result<(), String> {
    let Some(items) = value.as_array() else {
        return Err(format!("{field} must be an array"));
    };
    if items.is_empty() {
        return Err(format!("{field} must not be empty"));
    }
    let mut seen = BTreeSet::new();
    for item in items {
        let Some(text) = item.as_str().filter(|item| !item.trim().is_empty()) else {
            return Err(format!("{field} items must be non-empty strings"));
        };
        if !seen.insert(text.to_string()) {
            return Err(format!("{field} contains duplicate item {text}"));
        }
    }
    Ok(())
}

/// Validate the typed, closed experiment-plan contract before digest binding.
pub fn validate_experiment_plan(plan: &Value) -> Result<(), String> {
    if !exact_fields(
        plan,
        &[
            "schema_version",
            "plan_id",
            "benchmark_id",
            "target",
            "intervention",
            "control",
            "allocation",
            "measurement",
            "sample",
            "budget",
            "safety",
            "identification_assumptions",
            "confounders_and_selection",
            "stop_rules",
        ],
    ) {
        return Err("experiment plan does not match its closed fourteen-field contract".into());
    }
    if text(plan, "schema_version") != Some("aced-experiment-plan-v0.2")
        || text(plan, "plan_id").is_none()
        || text(plan, "benchmark_id") != Some("ACED-REPAIR-ADAPTER-001")
    {
        return Err("experiment plan identity or benchmark binding mismatch".into());
    }

    let target = &plan["target"];
    if !exact_fields(target, &["question", "hypothesis_ids"])
        || text(target, "question").is_none()
    {
        return Err("experiment plan target contract mismatch".into());
    }
    validate_string_array(&target["hypothesis_ids"], "target.hypothesis_ids")?;

    let intervention = &plan["intervention"];
    if !exact_fields(intervention, &["kind", "target_ref", "treatment_spec"])
        || text(intervention, "kind").is_none()
        || text(intervention, "target_ref").is_none()
        || text(intervention, "treatment_spec").is_none()
    {
        return Err("experiment plan intervention contract mismatch".into());
    }

    let control = &plan["control"];
    if !exact_fields(control, &["kind", "description", "control_variables"])
        || text(control, "kind").is_none()
        || text(control, "description").is_none()
    {
        return Err("experiment plan control contract mismatch".into());
    }
    validate_string_array(&control["control_variables"], "control.control_variables")?;

    let allocation = &plan["allocation"];
    if !exact_fields(allocation, &["method", "unit", "seed_commitment_sha256"])
        || text(allocation, "method").is_none()
        || text(allocation, "unit").is_none()
        || !text(allocation, "seed_commitment_sha256").is_some_and(is_sha256)
    {
        return Err("experiment plan allocation contract mismatch".into());
    }

    let measurement = &plan["measurement"];
    if !exact_fields(measurement, &["variables", "timepoints", "outcome_taxonomy"]) {
        return Err("experiment plan measurement contract mismatch".into());
    }
    validate_string_array(&measurement["variables"], "measurement.variables")?;
    validate_string_array(&measurement["timepoints"], "measurement.timepoints")?;
    validate_string_array(&measurement["outcome_taxonomy"], "measurement.outcome_taxonomy")?;

    let sample = &plan["sample"];
    if !exact_fields(
        sample,
        &[
            "planned_units",
            "standardized_effect_size",
            "alpha",
            "target_power",
            "power_method",
        ],
    ) || sample["planned_units"].as_u64().is_none_or(|units| units < 2 || units % 2 != 0)
        || text(sample, "power_method")
            != Some("two_sided_equal_allocation_independent_means_normal_approximation")
    {
        return Err("experiment plan numeric sample/power contract mismatch".into());
    }
    let power_design = TwoSidedMeanPowerDesign {
        standardized_effect_size: finite_number(sample, "standardized_effect_size")
            .unwrap_or(f64::NAN),
        alpha: finite_number(sample, "alpha").unwrap_or(f64::NAN),
        target_power: finite_number(sample, "target_power").unwrap_or(f64::NAN),
    };
    let planned_units = sample["planned_units"]
        .as_u64()
        .ok_or_else(|| "experiment plan planned_units must be an integer".to_string())?;
    power_design.qualify_planned_total(planned_units)
        .map_err(|error| format!("experiment plan is underpowered or invalid: {error}"))?;

    let budget = &plan["budget"];
    if !exact_fields(budget, &["max_experiments", "max_turns", "max_cost_units"])
        || budget["max_experiments"].as_u64().is_none_or(|value| value == 0)
        || budget["max_turns"].as_u64().is_none_or(|value| value == 0)
        || finite_number(budget, "max_cost_units").is_none_or(|value| value < 0.0)
    {
        return Err("experiment plan budget contract mismatch".into());
    }

    let safety = &plan["safety"];
    if !exact_fields(
        safety,
        &["risk_class", "reversibility", "maximum_exposure", "stop_conditions"],
    ) || text(safety, "risk_class").is_none()
        || !["reversible", "partially_reversible", "irreversible"]
            .contains(&text(safety, "reversibility").unwrap_or(""))
        || finite_number(safety, "maximum_exposure").is_none_or(|value| value < 0.0)
    {
        return Err("experiment plan safety contract mismatch".into());
    }
    validate_string_array(&safety["stop_conditions"], "safety.stop_conditions")?;
    validate_string_array(
        &plan["identification_assumptions"],
        "identification_assumptions",
    )?;
    validate_string_array(
        &plan["confounders_and_selection"],
        "confounders_and_selection",
    )?;

    let stop_rules = &plan["stop_rules"];
    if !exact_fields(
        stop_rules,
        &[
            "premature_stop_rule",
            "continue_rule",
            "conflict_resolution_rule",
        ],
    ) || text(stop_rules, "premature_stop_rule").is_none()
        || text(stop_rules, "continue_rule").is_none()
        || text(stop_rules, "conflict_resolution_rule").is_none()
    {
        return Err("experiment plan stop-rule contract mismatch".into());
    }
    Ok(())
}

/// Hash the immutable event envelope, excluding only its own event_sha256 field.
pub fn event_sha256(event: &Value) -> Result<String, String> {
    const ENVELOPE_FIELDS: [&str; 5] = [
        "event_id",
        "sequence",
        "event_type",
        "previous_event_sha256",
        "payload",
    ];
    if !exact_fields(
        event,
        &[
            "event_id",
            "sequence",
            "event_type",
            "previous_event_sha256",
            "payload",
            "event_sha256",
        ],
    ) {
        return Err("event must contain exactly the six declared envelope fields".into());
    }
    let mut envelope = Map::new();
    for key in ENVELOPE_FIELDS {
        let value = event
            .get(key)
            .ok_or_else(|| format!("event envelope is missing {key}"))?;
        envelope.insert(key.to_string(), value.clone());
    }
    let mut bytes = Vec::new();
    write_canonical_json(&Value::Object(envelope), &mut bytes)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn validate_distribution(value: &Value) -> Result<(), String> {
    let Some(outcomes) = value.as_array() else {
        return Err("predicted_outcome_distribution must be an array".into());
    };
    if outcomes.is_empty() {
        return Err("predicted_outcome_distribution must not be empty".into());
    }
    let mut names = BTreeSet::new();
    let mut total = 0.0;
    for (index, outcome) in outcomes.iter().enumerate() {
        if !exact_fields(outcome, &["outcome", "probability"]) {
            return Err(format!(
                "predicted outcome {index} must contain exactly outcome and probability"
            ));
        }
        let name = text(outcome, "outcome")
            .ok_or_else(|| format!("predicted outcome {index} has no identity"))?;
        let probability = finite_number(outcome, "probability")
            .ok_or_else(|| format!("predicted outcome {index} probability is invalid"))?;
        if probability < 0.0 || probability > 1.0 || !names.insert(name.to_string()) {
            return Err(format!(
                "predicted outcome {index} has an invalid probability or duplicate identity"
            ));
        }
        total += probability;
    }
    if (total - 1.0).abs() > TOLERANCE {
        return Err(format!(
            "predicted outcome probabilities must sum to one; observed {total}"
        ));
    }
    Ok(())
}

fn validate_proposal(payload: &Value) -> Result<ProposalPrediction, String> {
    const FIELDS: [&str; 11] = [
        "experiment_id",
        "experiment_plan_sha256",
        "belief_state_sha256",
        "selector",
        "predicted_outcome_distribution",
        "predicted_eig_bits",
        "predicted_evsi",
        "predicted_risk",
        "risk_aversion_coefficient",
        "predicted_cost",
        "predicted_selection_score",
    ];
    if !exact_fields(payload, &FIELDS) {
        return Err(
            "experiment_proposal payload does not match its closed eleven-field contract".into(),
        );
    }
    let id = text(payload, "experiment_id")
        .ok_or_else(|| "experiment proposal has no experiment_id".to_string())?;
    if !text(payload, "experiment_plan_sha256").is_some_and(is_sha256) {
        return Err(format!("experiment {id} has a malformed experiment-plan digest"));
    }
    if !text(payload, "belief_state_sha256").is_some_and(is_sha256) {
        return Err(format!("experiment {id} has a malformed belief-state digest"));
    }
    if ![
        "EIG",
        "EVSI",
        "RISK_ADJUSTED_EVSI",
    ]
    .contains(&text(payload, "selector").unwrap_or(""))
    {
        return Err(format!("experiment {id} has an unsupported selector"));
    }
    validate_distribution(&payload["predicted_outcome_distribution"])?;
    let outcome_probabilities = payload["predicted_outcome_distribution"]
        .as_array()
        .expect("distribution validated above")
        .iter()
        .map(|outcome| {
            (
                text(outcome, "outcome").expect("outcome identity validated above").to_string(),
                finite_number(outcome, "probability").expect("probability validated above"),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for key in [
        "predicted_eig_bits",
        "predicted_evsi",
        "predicted_risk",
        "predicted_cost",
    ] {
        if finite_number(payload, key).is_none_or(|value| value < 0.0) {
            return Err(format!("experiment {id} has invalid {key}"));
        }
    }
    let risk_aversion = finite_number(payload, "risk_aversion_coefficient")
        .filter(|value| *value >= 0.0)
        .ok_or_else(|| format!("experiment {id} has invalid risk_aversion_coefficient"))?;
    let eig = finite_number(payload, "predicted_eig_bits").unwrap_or_default();
    let evsi = finite_number(payload, "predicted_evsi").unwrap_or_default();
    let risk = finite_number(payload, "predicted_risk").unwrap_or_default();
    let cost = finite_number(payload, "predicted_cost").unwrap_or_default();
    let observed_score = finite_number(payload, "predicted_selection_score")
        .ok_or_else(|| format!("experiment {id} has invalid predicted_selection_score"))?;
    let expected_score = match text(payload, "selector").unwrap_or("") {
        "EIG" => eig,
        "EVSI" => evsi - cost,
        "RISK_ADJUSTED_EVSI" => evsi - cost - risk_aversion * risk,
        _ => unreachable!("selector validated above"),
    };
    if (observed_score - expected_score).abs() > TOLERANCE {
        return Err(format!(
            "experiment {id} predicted_selection_score is inconsistent with selector arithmetic"
        ));
    }
    Ok(ProposalPrediction {
        experiment_id: id.to_string(),
        outcome_probabilities,
    })
}

fn validate_observation(
    payload: &Value,
    proposals: &BTreeMap<String, ProposalPrediction>,
    observed_proposals: &mut BTreeSet<String>,
) -> Result<Option<ObservationScores>, String> {
    if !exact_fields(
        payload,
        &[
            "proposal_event_id",
            "observation_status",
            "observation_sha256",
            "actual_cost",
            "observed_outcome",
        ],
    ) {
        return Err(
            "experiment_observation payload does not match its closed five-field contract".into(),
        );
    }
    let proposal_id = text(payload, "proposal_event_id")
        .ok_or_else(|| "observation has no proposal_event_id".to_string())?;
    let Some(prediction) = proposals.get(proposal_id) else {
        return Err(format!("observation references unknown or future proposal {proposal_id}"));
    };
    if !observed_proposals.insert(proposal_id.to_string()) {
        return Err(format!("proposal {proposal_id} has more than one observation event"));
    }
    let status = text(payload, "observation_status")
        .ok_or_else(|| "observation has no status".to_string())?;
    match status {
        "observed" => {
            if !text(payload, "observation_sha256").is_some_and(is_sha256)
                || finite_number(payload, "actual_cost").is_none_or(|cost| cost < 0.0)
            {
                return Err(format!(
                    "observed proposal {proposal_id} lacks a digest or actual cost"
                ));
            }
            let observed_outcome = text(payload, "observed_outcome")
                .ok_or_else(|| format!("observed proposal {proposal_id} lacks an outcome label"))?;
            if !prediction.outcome_probabilities.contains_key(observed_outcome) {
                return Err(format!(
                    "observed outcome {observed_outcome} was not in the proposal's preregistered outcome space"
                ));
            }
            let mut brier_score = 0.0;
            for (outcome, probability) in &prediction.outcome_probabilities {
                let target = if outcome == observed_outcome { 1.0 } else { 0.0 };
                brier_score += (probability - target).powi(2);
            }
            let observed_probability = prediction.outcome_probabilities[observed_outcome];
            let log_loss = if observed_probability > 0.0 {
                Some(-observed_probability.ln())
            } else {
                None
            };
            Ok(Some(ObservationScores { brier_score, log_loss }))
        }
        "failed" | "unknown" => {
            if !payload["observation_sha256"].is_null()
                || !payload["observed_outcome"].is_null()
                || (!payload["actual_cost"].is_null()
                    && finite_number(payload, "actual_cost").is_none_or(|cost| cost < 0.0))
            {
                return Err(format!(
                    "failed/unknown proposal {proposal_id} must not claim an observation digest or outcome"
                ));
            }
            Ok(None)
        }
        _ => Err(format!("proposal {proposal_id} has unsupported observation status")),
    }
}

fn validate_stop(
    payload: &Value,
    proposals: &BTreeMap<String, ProposalPrediction>,
    observed_proposals: &BTreeSet<String>,
) -> Result<(), String> {
    if !exact_fields(payload, &["evidence_state_sha256", "reason"]) {
        return Err("stop payload must contain exactly evidence_state_sha256 and reason".into());
    }
    if !text(payload, "evidence_state_sha256").is_some_and(is_sha256)
        || text(payload, "reason").is_none()
    {
        return Err("stop event requires an evidence-state digest and reason".into());
    }
    if proposals
        .keys()
        .any(|proposal_event_id| !observed_proposals.contains(proposal_event_id))
    {
        return Err(
            "stop event cannot leave a proposed experiment without an observed, failed or unknown reconciliation event".into(),
        );
    }
    Ok(())
}

/// Qualify append-only proposal/observation/stop events.
/// The hash chain detects edits relative to a previously anchored event. It does not prove a trusted timestamp.
pub fn validate_trace(trace: &Value) -> Result<TraceReport, Vec<String>> {
    let mut errors = Vec::new();
    if !exact_fields(trace, &["schema_version", "run_id", "events", "claim_ceiling"])
        || text(trace, "schema_version") != Some(SCHEMA_VERSION)
        || text(trace, "run_id").is_none()
        || text(trace, "claim_ceiling").is_none()
    {
        errors.push("trace identity or closed top-level contract mismatch".into());
    }

    let Some(events) = trace["events"].as_array() else {
        return Err(vec!["trace events must be an array".into()]);
    };
    if events.is_empty() {
        errors.push("trace must contain at least one event".into());
    }

    let mut previous_hash = GENESIS_SHA256.to_string();
    let mut event_ids = BTreeSet::new();
    let mut proposals = BTreeMap::<String, ProposalPrediction>::new();
    let mut experiment_ids = BTreeSet::<String>::new();
    let mut observed_proposals = BTreeSet::<String>::new();
    let mut brier_sum = 0.0;
    let mut log_loss_sum = 0.0;
    let mut log_loss_count = 0usize;
    let mut stopped = false;
    let mut report = TraceReport::default();

    for (index, event) in events.iter().enumerate() {
        if !exact_fields(
            event,
            &[
                "event_id",
                "sequence",
                "event_type",
                "previous_event_sha256",
                "payload",
                "event_sha256",
            ],
        ) {
            errors.push(format!("events[{index}] does not match the closed event envelope"));
            continue;
        }
        let Some(event_id) = text(event, "event_id") else {
            errors.push(format!("events[{index}] has no event_id"));
            continue;
        };
        if !event_ids.insert(event_id.to_string()) {
            errors.push(format!("events[{index}] duplicates event_id {event_id}"));
        }
        if event["sequence"].as_u64() != Some(index as u64) {
            errors.push(format!("event {event_id} has an invalid sequence number"));
        }
        if text(event, "previous_event_sha256") != Some(previous_hash.as_str()) {
            errors.push(format!("event {event_id} breaks the previous-event hash chain"));
        }
        let expected_hash = match event_sha256(event) {
            Ok(hash) => hash,
            Err(error) => {
                errors.push(format!("event {event_id}: {error}"));
                continue;
            }
        };
        let observed_hash = text(event, "event_sha256").unwrap_or("");
        if observed_hash != expected_hash {
            errors.push(format!("event {event_id} hash does not bind its exact envelope"));
        }
        previous_hash = expected_hash;
        report.event_count += 1;

        let Some(event_type) = text(event, "event_type") else {
            errors.push(format!("event {event_id} has no type"));
            continue;
        };
        let payload = &event["payload"];
        match event_type {
            "experiment_proposal" => {
                if stopped {
                    errors.push(format!("event {event_id} proposes an experiment after stop"));
                }
                match validate_proposal(payload) {
                    Ok(prediction) => {
                        if !experiment_ids.insert(prediction.experiment_id.clone()) {
                            errors.push(format!(
                                "duplicate experiment proposal {}",
                                prediction.experiment_id
                            ));
                        } else {
                            proposals.insert(event_id.to_string(), prediction);
                            report.proposal_count += 1;
                        }
                    }
                    Err(error) => errors.push(format!("event {event_id}: {error}")),
                }
            }
            "experiment_observation" => {
                if stopped {
                    errors.push(format!("event {event_id} records an observation after stop"));
                }
                match validate_observation(payload, &proposals, &mut observed_proposals) {
                    Ok(scores) => {
                        report.observation_count += 1;
                        if let Some(scores) = scores {
                            report.scored_observation_count += 1;
                            brier_sum += scores.brier_score;
                            if let Some(log_loss) = scores.log_loss {
                                log_loss_sum += log_loss;
                                log_loss_count += 1;
                            } else {
                                report.zero_probability_outcome_count += 1;
                            }
                        }
                    }
                    Err(error) => errors.push(format!("event {event_id}: {error}")),
                }
            }
            "stop" => {
                if stopped {
                    errors.push(format!("event {event_id} duplicates a stop event"));
                }
                if let Err(error) = validate_stop(payload, &proposals, &observed_proposals) {
                    errors.push(format!("event {event_id}: {error}"));
                }
                stopped = true;
            }
            _ => errors.push(format!("event {event_id} has unsupported event_type {event_type}")),
        }
    }

    report.stopped = stopped;
    report.terminal_sha256 = previous_hash;
    if report.scored_observation_count > 0 {
        report.mean_brier_score = Some(brier_sum / report.scored_observation_count as f64);
        if report.zero_probability_outcome_count == 0 && log_loss_count > 0 {
            report.mean_log_loss = Some(log_loss_sum / log_loss_count as f64);
        }
    }
    if errors.is_empty() {
        Ok(report)
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}

/// Verify every proposal's plan digest against an independently supplied immutable plan artifact.
pub fn validate_trace_with_plans(
    trace: &Value,
    plan_artifacts: &[Value],
) -> Result<TraceReport, Vec<String>> {
    let report = validate_trace(trace)?;
    let mut plans_by_digest = BTreeMap::<String, &Value>::new();
    let mut errors = Vec::new();

    for (index, plan) in plan_artifacts.iter().enumerate() {
        if let Err(error) = validate_experiment_plan(plan) {
            errors.push(format!("plan_artifacts[{index}]: {error}"));
            continue;
        }
        let digest = match canonical_json_sha256(plan) {
            Ok(digest) => digest,
            Err(error) => {
                errors.push(format!("plan_artifacts[{index}]: {error}"));
                continue;
            }
        };
        if plans_by_digest.insert(digest.clone(), plan).is_some() {
            errors.push(format!("duplicate plan artifact digest {digest}"));
        }
    }

    let events = trace["events"].as_array().expect("trace validated events array");
    for event in events.iter().filter(|event| {
        text(event, "event_type") == Some("experiment_proposal")
    }) {
        let event_id = text(event, "event_id").unwrap_or("<missing-event-id>");
        let payload = &event["payload"];
        let expected_digest = text(payload, "experiment_plan_sha256").unwrap_or("");
        let Some(plan) = plans_by_digest.get(expected_digest) else {
            errors.push(format!(
                "proposal {event_id} references an experiment plan artifact that was not supplied or whose digest differs"
            ));
            continue;
        };
        if text(payload, "experiment_id") != text(plan, "plan_id") {
            errors.push(format!(
                "proposal {event_id} experiment_id does not match the bound plan_id"
            ));
        }
    }

    if errors.is_empty() {
        Ok(report)
    } else {
        errors.sort();
        errors.dedup();
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const TRACE_SCHEMA: &str =
        include_str!("../../EXPERIMENT_VALUE_TRACE_V0_1.schema.json");
    const PLAN_SCHEMA: &str =
        include_str!("../../ACED_EXPERIMENT_PLAN_V0_2.schema.json");
    const PLAN_FIXTURE: &str =
        include_str!("../tests/fixtures/aced_experiment_plan_v0_2.json");

    fn parsed_plan() -> Value {
        serde_json::from_str(PLAN_FIXTURE).expect("plan fixture JSON")
    }

    #[test]
    fn checked_in_trace_schema_matches_the_closed_rust_contract() {
        let schema: Value = serde_json::from_str(TRACE_SCHEMA).expect("trace schema JSON");
        assert_eq!(
            text(&schema, "$schema"),
            Some("https://json-schema.org/draft/2020-12/schema")
        );
        assert_eq!(
            text(&schema, "title"),
            Some("ACED experiment-value append-only trace v0.1")
        );
        let plan_schema: Value =
            serde_json::from_str(PLAN_SCHEMA).expect("experiment plan schema JSON");
        assert_eq!(
            text(&plan_schema, "title"),
            Some("ACED prospective experiment plan v0.2")
        );
        assert_eq!(plan_schema["additionalProperties"], false);
        validate_experiment_plan(&parsed_plan()).expect("plan fixture satisfies Rust contract");
        let fixture_plan = parsed_plan();
        let sample = &fixture_plan["sample"];
        let design = TwoSidedMeanPowerDesign {
            standardized_effect_size: finite_number(sample, "standardized_effect_size").unwrap(),
            alpha: finite_number(sample, "alpha").unwrap(),
            target_power: finite_number(sample, "target_power").unwrap(),
        };
        let required = design.required_equal_allocation_sample_size().unwrap();
        assert_eq!(required.total_units, 126);
        assert_eq!(required.units_per_arm, 63);
        assert!(required.achieved_power_at_planned_n >= 0.8);
        let mut underpowered = parsed_plan();
        underpowered["sample"]["planned_units"] = Value::from(40);
        assert!(validate_experiment_plan(&underpowered).is_err());
        assert_eq!(
            plan_schema["required"].as_array().unwrap().len(),
            14
        );
        assert!(plan_schema["properties"]["target"]["additionalProperties"] == false);
        assert!(plan_schema["properties"]["control"]["additionalProperties"] == false);
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["properties"]["events"]["items"]["oneOf"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        let observation_schema = schema["properties"]["events"]["items"]["oneOf"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| {
                text(&item["properties"]["event_type"], "const") == Some("experiment_observation")
            })
            .unwrap();
        assert_eq!(
            observation_schema["properties"]["payload"]["required"]
                .as_array()
                .unwrap()
                .len(),
            5
        );
        assert!(
            observation_schema["properties"]["payload"]["properties"]
                .get("observed_outcome")
                .is_some()
        );
        assert_eq!(
            observation_schema["properties"]["payload"]["allOf"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }

    fn event(
        event_id: &str,
        sequence: u64,
        event_type: &str,
        previous: &str,
        payload: Value,
    ) -> Value {
        let mut value = json!({
            "event_id": event_id,
            "sequence": sequence,
            "event_type": event_type,
            "previous_event_sha256": previous,
            "payload": payload,
            "event_sha256": ""
        });
        value["event_sha256"] = Value::String(event_sha256(&value).unwrap());
        value
    }

    fn proposal() -> Value {
        let plan = parsed_plan();
        let plan_digest = canonical_json_sha256(&plan).expect("canonical plan digest");
        json!({
            "experiment_id": plan["plan_id"].as_str().unwrap(),
            "experiment_plan_sha256": plan_digest,
            "belief_state_sha256": "a".repeat(64),
            "selector": "RISK_ADJUSTED_EVSI",
            "predicted_outcome_distribution": [
                {"outcome": "positive", "probability": 0.3},
                {"outcome": "negative", "probability": 0.7}
            ],
            "predicted_eig_bits": 0.24,
            "predicted_evsi": 0.18,
            "predicted_risk": 0.01,
            "risk_aversion_coefficient": 1.0,
            "predicted_cost": 0.02,
            "predicted_selection_score": 0.15
        })
    }

    fn trace(events: Vec<Value>) -> Value {
        json!({
            "schema_version": SCHEMA_VERSION,
            "run_id": "run-001",
            "events": events,
            "claim_ceiling":
                concat!(
                    "Append-only forecast/observation trace structure only; ",
                    "not model capability or trusted time."
                )
        })
    }

    #[test]
    fn prospective_proposal_then_observation_qualifies() {
        let first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            proposal(),
        );
        let first_hash = text(&first, "event_sha256").unwrap().to_string();
        let second = event(
            "event-1",
            1,
            "experiment_observation",
            &first_hash,
            json!({
                "proposal_event_id": "event-0",
                "observation_status": "observed",
                "observation_sha256": "b".repeat(64),
                "actual_cost": 0.02,
                "observed_outcome": "positive"
            }),
        );
        let trace_value = trace(vec![first, second]);
        let result = validate_trace_with_plans(&trace_value, &[parsed_plan()]).unwrap();
        assert_eq!(result.event_count, 2);
        assert_eq!(result.proposal_count, 1);
        assert_eq!(result.observation_count, 1);
        assert_eq!(result.scored_observation_count, 1);
        assert_eq!(result.mean_brier_score, Some(0.98));
        assert!((result.mean_log_loss.unwrap() - -(0.3_f64).ln()).abs() < TOLERANCE);
        assert_eq!(result.zero_probability_outcome_count, 0);
        assert!(!result.stopped);
        assert!(is_sha256(&result.terminal_sha256));
    }

    #[test]
    fn modified_forecast_after_observation_breaks_event_hash() {
        let mut first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            proposal(),
        );
        first["payload"]["predicted_evsi"] = json!(0.99);
        let first_hash = text(&first, "event_sha256").unwrap().to_string();
        let second = event(
            "event-1",
            1,
            "experiment_observation",
            &first_hash,
            json!({
                "proposal_event_id": "event-0",
                "observation_status": "observed",
                "observation_sha256": "b".repeat(64),
                "actual_cost": 0.02,
                "observed_outcome": "positive"
            }),
        );
        assert!(validate_trace(&trace(vec![first, second])).is_err());
    }

    #[test]
    fn eig_trace_score_is_not_reduced_by_utility_cost() {
        let mut payload = proposal();
        payload["selector"] = Value::String("EIG".into());
        payload["predicted_eig_bits"] = json!(0.24);
        payload["predicted_evsi"] = json!(0.18);
        payload["predicted_risk"] = json!(0.01);
        payload["risk_aversion_coefficient"] = json!(1.0);
        payload["predicted_cost"] = json!(100.0);
        payload["predicted_selection_score"] = json!(0.24);
        let first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            payload,
        );
        assert!(validate_trace(&trace(vec![first])).is_ok());
    }

    #[test]
    fn failed_or_unknown_observation_is_retained_but_not_proper_score_scored() {
        let first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            proposal(),
        );
        let first_hash = text(&first, "event_sha256").unwrap().to_string();
        let second = event(
            "event-1",
            1,
            "experiment_observation",
            &first_hash,
            json!({
                "proposal_event_id": "event-0",
                "observation_status": "unknown",
                "observation_sha256": Value::Null,
                "actual_cost": Value::Null,
                "observed_outcome": Value::Null
            }),
        );
        let report = validate_trace(&trace(vec![first, second])).unwrap();
        assert_eq!(report.observation_count, 1);
        assert_eq!(report.scored_observation_count, 0);
        assert_eq!(report.mean_brier_score, None);
        assert_eq!(report.mean_log_loss, None);
    }

    #[test]
    fn zero_probability_observation_is_not_hidden_by_log_loss_clipping() {
        let mut payload = proposal();
        payload["predicted_outcome_distribution"] = json!([
            {"outcome": "positive", "probability": 0.0},
            {"outcome": "negative", "probability": 1.0}
        ]);
        let first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            payload,
        );
        let first_hash = text(&first, "event_sha256").unwrap().to_string();
        let second = event(
            "event-1",
            1,
            "experiment_observation",
            &first_hash,
            json!({
                "proposal_event_id": "event-0",
                "observation_status": "observed",
                "observation_sha256": "b".repeat(64),
                "actual_cost": 0.02,
                "observed_outcome": "positive"
            }),
        );
        let report = validate_trace(&trace(vec![first, second])).unwrap();
        assert_eq!(report.mean_brier_score, Some(2.0));
        assert_eq!(report.mean_log_loss, None);
        assert_eq!(report.zero_probability_outcome_count, 1);
    }

    #[test]
    fn altered_plan_artifact_is_rejected_by_digest_binding() {
        let first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            proposal(),
        );
        let trace_value = trace(vec![first]);
        let mut altered = parsed_plan();
        altered["control"]["description"] =
            Value::String("Changed control after the proposal was committed.".into());
        assert!(validate_trace_with_plans(&trace_value, &[altered]).is_err());
    }

    #[test]
    fn plan_id_must_match_proposal_experiment_id() {
        let mut payload = proposal();
        payload["experiment_id"] = Value::String("different-experiment".into());
        let first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            payload,
        );
        assert!(validate_trace_with_plans(&trace(vec![first]), &[parsed_plan()]).is_err());
    }

    #[test]
    fn malformed_experiment_plan_digest_is_rejected() {
        let mut payload = proposal();
        payload["experiment_plan_sha256"] = json!("not-a-sha256");
        let first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            payload,
        );
        assert!(validate_trace(&trace(vec![first])).is_err());
    }

    #[test]
    fn forecast_trace_schema_requires_experiment_plan_digest() {
        let schema: Value =
            serde_json::from_str(TRACE_SCHEMA).expect("trace schema JSON");
        let proposal_schema = schema["properties"]["events"]["items"]["oneOf"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| {
                text(&item["properties"]["event_type"], "const")
                    == Some("experiment_proposal")
            })
            .unwrap();
        assert!(proposal_schema["properties"]["payload"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|field| field == "experiment_plan_sha256"));
        assert_eq!(
            proposal_schema["properties"]["payload"]["properties"]
                ["experiment_plan_sha256"]["pattern"],
            "^[a-f0-9]{64}$"
        );
    }

    #[test]
    fn inconsistent_predicted_selection_score_is_rejected() {
        let mut payload = proposal();
        payload["predicted_selection_score"] = json!(0.42);
        let first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            payload,
        );
        assert!(validate_trace(&trace(vec![first])).is_err());
    }

    #[test]
    fn orphan_observation_is_rejected() {
        let first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            proposal(),
        );
        let first_hash = text(&first, "event_sha256").unwrap().to_string();
        let second = event(
            "event-1",
            1,
            "experiment_observation",
            &first_hash,
            json!({
                "proposal_event_id": "missing-proposal",
                "observation_status": "observed",
                "observation_sha256": "b".repeat(64),
                "actual_cost": 0.02,
                "observed_outcome": "positive"
            }),
        );
        assert!(validate_trace(&trace(vec![first, second])).is_err());
    }

    #[test]
    fn duplicate_observation_for_one_proposal_is_rejected() {
        let first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            proposal(),
        );
        let first_hash = text(&first, "event_sha256").unwrap().to_string();
        let second = event(
            "event-1",
            1,
            "experiment_observation",
            &first_hash,
            json!({
                "proposal_event_id": "event-0",
                "observation_status": "observed",
                "observation_sha256": "b".repeat(64),
                "actual_cost": 0.02,
                "observed_outcome": "positive"
            }),
        );
        let second_hash = text(&second, "event_sha256").unwrap().to_string();
        let third = event(
            "event-2",
            2,
            "experiment_observation",
            &second_hash,
            json!({
                "proposal_event_id": "event-0",
                "observation_status": "failed",
                "observation_sha256": Value::Null,
                "actual_cost": Value::Null,
                "observed_outcome": Value::Null
            }),
        );
        assert!(validate_trace(&trace(vec![first, second, third])).is_err());
    }

    #[test]
    fn stop_with_unreconciled_proposal_is_rejected() {
        let first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            proposal(),
        );
        let first_hash = text(&first, "event_sha256").unwrap().to_string();
        let stop = event(
            "event-1",
            1,
            "stop",
            &first_hash,
            json!({
                "evidence_state_sha256": "c".repeat(64),
                "reason": "Stopping with an unobserved probe is not a qualified terminal trace."
            }),
        );
        assert!(validate_trace(&trace(vec![first, stop])).is_err());
    }

    #[test]
    fn proposal_after_stop_is_rejected() {
        let stop = event(
            "event-0",
            0,
            "stop",
            GENESIS_SHA256,
            json!({
                "evidence_state_sha256": "c".repeat(64),
                "reason": "No remaining experiment has positive net decision value."
            }),
        );
        let stop_hash = text(&stop, "event_sha256").unwrap().to_string();
        let next = event(
            "event-1",
            1,
            "experiment_proposal",
            &stop_hash,
            proposal(),
        );
        assert!(validate_trace(&trace(vec![stop, next])).is_err());
    }

    #[test]
    fn malformed_forecast_distribution_is_rejected() {
        let mut payload = proposal();
        payload["predicted_outcome_distribution"][1]["probability"] = json!(0.5);
        let first = event(
            "event-0",
            0,
            "experiment_proposal",
            GENESIS_SHA256,
            payload,
        );
        assert!(validate_trace(&trace(vec![first])).is_err());
    }
}
