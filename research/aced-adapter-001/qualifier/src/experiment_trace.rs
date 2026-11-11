use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const SCHEMA_VERSION: &str = "aced-experiment-value-trace-v0.1";
const GENESIS_SHA256: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const TOLERANCE: f64 = 1e-9;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TraceReport {
    pub event_count: usize,
    pub proposal_count: usize,
    pub observation_count: usize,
    pub stopped: bool,
    pub terminal_sha256: String,
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

fn validate_proposal(payload: &Value) -> Result<String, String> {
    const FIELDS: [&str; 9] = [
        "experiment_id",
        "belief_state_sha256",
        "selector",
        "predicted_outcome_distribution",
        "predicted_eig_bits",
        "predicted_evsi",
        "predicted_risk",
        "predicted_cost",
        "predicted_net_value",
    ];
    if !exact_fields(payload, &FIELDS) {
        return Err("experiment_proposal payload does not match its closed nine-field contract".into());
    }
    let id = text(payload, "experiment_id")
        .ok_or_else(|| "experiment proposal has no experiment_id".to_string())?;
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
    if finite_number(payload, "predicted_net_value").is_none() {
        return Err(format!("experiment {id} has invalid predicted_net_value"));
    }
    Ok(id.to_string())
}

fn validate_observation(
    payload: &Value,
    proposals: &BTreeMap<String, String>,
    observed_proposals: &mut BTreeSet<String>,
) -> Result<(), String> {
    if !exact_fields(
        payload,
        &[
            "proposal_event_id",
            "observation_status",
            "observation_sha256",
            "actual_cost",
        ],
    ) {
        return Err("experiment_observation payload does not match its closed four-field contract".into());
    }
    let proposal_id = text(payload, "proposal_event_id")
        .ok_or_else(|| "observation has no proposal_event_id".to_string())?;
    if !proposals.contains_key(proposal_id) {
        return Err(format!("observation references unknown or future proposal {proposal_id}"));
    }
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
                return Err(format!("observed proposal {proposal_id} lacks a digest or actual cost"));
            }
        }
        "failed" | "unknown" => {
            if !payload["observation_sha256"].is_null()
                || (!payload["actual_cost"].is_null()
                    && finite_number(payload, "actual_cost").is_none_or(|cost| cost < 0.0))
            {
                return Err(format!(
                    "failed/unknown proposal {proposal_id} must not claim an observation digest"
                ));
            }
        }
        _ => return Err(format!("proposal {proposal_id} has unsupported observation status")),
    }
    Ok(())
}

fn validate_stop(payload: &Value) -> Result<(), String> {
    if !exact_fields(payload, &["evidence_state_sha256", "reason"]) {
        return Err("stop payload must contain exactly evidence_state_sha256 and reason".into());
    }
    if !text(payload, "evidence_state_sha256").is_some_and(is_sha256)
        || text(payload, "reason").is_none()
    {
        return Err("stop event requires an evidence-state digest and reason".into());
    }
    Ok(())
}

/// Qualify append-only proposal/observation/stop events.
/// The hash chain detects edits relative to a previously anchored event; by itself it is not a trusted timestamp.
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
    let mut proposals = BTreeMap::<String, String>::new();
    let mut observed_proposals = BTreeSet::<String>::new();
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
                    Ok(experiment_id) => {
                        if proposals
                            .insert(experiment_id.clone(), event_id.to_string())
                            .is_some()
                        {
                            errors.push(format!("duplicate experiment proposal {experiment_id}"));
                        }
                        report.proposal_count += 1;
                    }
                    Err(error) => errors.push(format!("event {event_id}: {error}")),
                }
            }
            "experiment_observation" => {
                if stopped {
                    errors.push(format!("event {event_id} records an observation after stop"));
                }
                match validate_observation(payload, &proposals, &mut observed_proposals) {
                    Ok(()) => report.observation_count += 1,
                    Err(error) => errors.push(format!("event {event_id}: {error}")),
                }
            }
            "stop" => {
                if stopped {
                    errors.push(format!("event {event_id} duplicates a stop event"));
                }
                if let Err(error) = validate_stop(payload) {
                    errors.push(format!("event {event_id}: {error}"));
                }
                stopped = true;
            }
            _ => errors.push(format!("event {event_id} has unsupported event_type {event_type}")),
        }
    }

    report.stopped = stopped;
    report.terminal_sha256 = previous_hash;
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
        json!({
            "experiment_id": "probe-1",
            "belief_state_sha256": "a".repeat(64),
            "selector": "EVSI",
            "predicted_outcome_distribution": [
                {"outcome": "positive", "probability": 0.3},
                {"outcome": "negative", "probability": 0.7}
            ],
            "predicted_eig_bits": 0.24,
            "predicted_evsi": 0.18,
            "predicted_risk": 0.01,
            "predicted_cost": 0.02,
            "predicted_net_value": 0.15
        })
    }

    fn trace(events: Vec<Value>) -> Value {
        json!({
            "schema_version": SCHEMA_VERSION,
            "run_id": "run-001",
            "events": events,
            "claim_ceiling": "Append-only forecast/observation trace structure only; not model capability or trusted time."
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
                "actual_cost": 0.02
            }),
        );
        let result = validate_trace(&trace(vec![first, second])).unwrap();
        assert_eq!(result.event_count, 2);
        assert_eq!(result.proposal_count, 1);
        assert_eq!(result.observation_count, 1);
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
                "actual_cost": 0.02
            }),
        );
        assert!(validate_trace(&trace(vec![first, second])).is_err());
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
                "actual_cost": 0.02
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
                "actual_cost": 0.02
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
                "actual_cost": Value::Null
            }),
        );
        assert!(validate_trace(&trace(vec![first, second, third])).is_err());
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
