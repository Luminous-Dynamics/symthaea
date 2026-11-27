// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#![forbid(unsafe_code)]

//! Semantic checks for SAT memory lifecycle JSONL traces.
//!
//! This tool validates event-chain semantics and claim boundaries. It is not a
//! Draft 2020-12 JSON Schema validator, a signature verifier, an evaluator, or
//! proof that evidence references resolve to authentic external artifacts.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::process::ExitCode;

#[derive(Default)]
struct CorpusState {
    trace_ids: BTreeSet<String>,
    event_ids: BTreeSet<String>,
    paired_controls: Vec<(String, String)>,
    trace_count: usize,
    event_count: usize,
}

fn field_str<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing or empty string field '{key}'"))
}

fn field_u64(value: &Value, key: &str) -> Result<u64, String> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("missing or invalid unsigned integer field '{key}'"))
}

fn is_sha256(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn event_index(events: &[Value], stage: &str, outcome: Option<&str>) -> Option<usize> {
    events.iter().position(|event| {
        event.get("stage").and_then(Value::as_str) == Some(stage)
            && outcome.is_none_or(|expected| {
                event.get("outcome").and_then(Value::as_str) == Some(expected)
            })
    })
}

fn require_order(
    events: &[Value],
    earlier: (&str, Option<&str>),
    later: (&str, Option<&str>),
    trace_id: &str,
) -> Result<(), String> {
    let a = event_index(events, earlier.0, earlier.1)
        .ok_or_else(|| format!("{trace_id}: missing event {} {:?}", earlier.0, earlier.1))?;
    let b = event_index(events, later.0, later.1)
        .ok_or_else(|| format!("{trace_id}: missing event {} {:?}", later.0, later.1))?;
    if a >= b {
        return Err(format!(
            "{trace_id}: event order violation: {} must precede {}",
            earlier.0, later.0
        ));
    }
    Ok(())
}

fn has_independent_sink_effect(events: &[Value]) -> bool {
    events.iter().any(|event| {
        if event.get("stage").and_then(Value::as_str) != Some("external_effect_observation")
            || event.get("outcome").and_then(Value::as_str) != Some("effect_confirmed")
            || event.get("actor").and_then(Value::as_str) != Some("independent_sink")
            || event.get("trust_boundary").and_then(Value::as_str) != Some("external_sink")
        {
            return false;
        }
        event
            .get("evidence")
            .and_then(Value::as_array)
            .is_some_and(|evidence| {
                evidence.iter().any(|item| {
                    item.get("observer").and_then(Value::as_str) == Some("independent_sink")
                        && item.get("independence_status").and_then(Value::as_str)
                            == Some("independent")
                })
            })
    })
}

fn validate_trace(
    trace: &Value,
    line_number: usize,
    state: &mut CorpusState,
) -> Result<(), String> {
    let prefix = format!("line {line_number}");
    let schema_version = field_str(trace, "schema_version").map_err(|e| format!("{prefix}: {e}"))?;
    if schema_version != "sat-memory-lifecycle-trace/v0.1" {
        return Err(format!("{prefix}: unsupported schema_version '{schema_version}'"));
    }

    let trace_id = field_str(trace, "trace_id").map_err(|e| format!("{prefix}: {e}"))?.to_owned();
    if !state.trace_ids.insert(trace_id.clone()) {
        return Err(format!("{prefix}: duplicate trace_id '{trace_id}'"));
    }

    let execution_state =
        field_str(trace, "execution_state").map_err(|e| format!("{trace_id}: {e}"))?;
    let executed = matches!(execution_state, "executed_candidate" | "independently_reproduced");
    if !matches!(
        execution_state,
        "synthetic_fixture_not_executed" | "executed_candidate" | "independently_reproduced"
    ) {
        return Err(format!("{trace_id}: unsupported execution_state '{execution_state}'"));
    }
    if executed {
        field_str(trace, "run_manifest_ref")
            .map_err(|_| format!("{trace_id}: executed trace requires run_manifest_ref"))?;
    }

    if let Some(pair) = trace.get("paired_control_trace_id") {
        if !pair.is_null() {
            let pair_id = pair
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| format!("{trace_id}: paired_control_trace_id must be a string or null"))?;
            state.paired_controls.push((trace_id.clone(), pair_id.to_owned()));
        }
    }

    let profile = trace
        .get("attack_profile")
        .ok_or_else(|| format!("{trace_id}: missing attack_profile"))?;
    let budget = field_u64(profile, "adaptive_round_budget")
        .map_err(|e| format!("{trace_id}: {e}"))?;
    let observed = field_u64(profile, "rounds_observed")
        .map_err(|e| format!("{trace_id}: {e}"))?;
    if observed > budget || budget > 15 {
        return Err(format!(
            "{trace_id}: observed adaptive rounds {observed} exceed budget {budget}"
        ));
    }

    let boundary = field_str(trace, "sut_boundary").map_err(|e| format!("{trace_id}: {e}"))?;
    let claim_scope = field_str(trace, "claim_scope").map_err(|e| format!("{trace_id}: {e}"))?;
    let events = trace
        .get("events")
        .and_then(Value::as_array)
        .filter(|events| !events.is_empty())
        .ok_or_else(|| format!("{trace_id}: events must be a non-empty array"))?;

    let mut local_events = BTreeMap::<String, Value>::new();
    for (index, event) in events.iter().enumerate() {
        let event_id = field_str(event, "event_id")
            .map_err(|e| format!("{trace_id}: event {index}: {e}"))?
            .to_owned();
        if !state.event_ids.insert(event_id.clone()) {
            return Err(format!("{trace_id}: duplicate event_id '{event_id}' across corpus"));
        }
        if local_events.insert(event_id.clone(), event.clone()).is_some() {
            return Err(format!("{trace_id}: duplicate event_id '{event_id}' within trace"));
        }

        let sequence = field_u64(event, "sequence")
            .map_err(|e| format!("{trace_id}: event {event_id}: {e}"))?;
        if sequence != index as u64 {
            return Err(format!(
                "{trace_id}: event {event_id} sequence {sequence}, expected {index}"
            ));
        }

        let expected_previous = if index == 0 {
            None
        } else {
            Some(
                field_str(&events[index - 1], "event_id")
                    .map_err(|e| format!("{trace_id}: predecessor: {e}"))?,
            )
        };
        let actual_previous = event.get("previous_event_id");
        let predecessor_matches = match (expected_previous, actual_previous) {
            (None, Some(Value::Null)) => true,
            (Some(expected), Some(Value::String(actual))) => expected == actual,
            _ => false,
        };
        if !predecessor_matches {
            return Err(format!("{trace_id}: broken predecessor link at event {event_id}"));
        }

        if let Some(round) = event.get("adaptive_round") {
            if !round.is_null() {
                let round = round.as_u64().ok_or_else(|| {
                    format!("{trace_id}: event {event_id} has invalid adaptive_round")
                })?;
                if round == 0 || round > budget {
                    return Err(format!(
                        "{trace_id}: event {event_id} adaptive_round {round} outside budget {budget}"
                    ));
                }
            }
        }

        let evidence = event
            .get("evidence")
            .and_then(Value::as_array)
            .filter(|items| !items.is_empty())
            .ok_or_else(|| format!("{trace_id}: event {event_id} lacks evidence"))?;
        for item in evidence {
            field_str(item, "evidence_ref")
                .map_err(|e| format!("{trace_id}: event {event_id}: {e}"))?;
            field_str(item, "observer")
                .map_err(|e| format!("{trace_id}: event {event_id}: {e}"))?;
            field_str(item, "independence_status")
                .map_err(|e| format!("{trace_id}: event {event_id}: {e}"))?;
            if executed {
                let digest = field_str(item, "artifact_sha256").map_err(|_| {
                    format!("{trace_id}: executed event {event_id} evidence needs artifact_sha256")
                })?;
                if !is_sha256(digest) {
                    return Err(format!(
                        "{trace_id}: event {event_id} has malformed SHA-256 '{digest}'"
                    ));
                }
            } else if let Some(Value::String(digest)) = item.get("artifact_sha256") {
                if !is_sha256(digest) {
                    return Err(format!(
                        "{trace_id}: event {event_id} has malformed SHA-256 '{digest}'"
                    ));
                }
            }
        }
        state.event_count += 1;
    }

    let local_ids = local_events.keys().cloned().collect::<BTreeSet<_>>();
    let invariants = trace
        .get("invariants")
        .and_then(Value::as_array)
        .filter(|items| !items.is_empty())
        .ok_or_else(|| format!("{trace_id}: invariants must be a non-empty array"))?;
    for invariant in invariants {
        let invariant_id = field_str(invariant, "invariant_id")
            .map_err(|e| format!("{trace_id}: {e}"))?;
        let supporting = invariant
            .get("supporting_event_ids")
            .and_then(Value::as_array)
            .filter(|items| !items.is_empty())
            .ok_or_else(|| format!("{trace_id}: invariant {invariant_id} has no supporting events"))?;
        for reference in supporting {
            let id = reference
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| format!("{trace_id}: invariant {invariant_id} has invalid event reference"))?;
            if !local_ids.contains(id) {
                return Err(format!(
                    "{trace_id}: invariant {invariant_id} references missing event '{id}'"
                ));
            }
        }

        if invariant_id == "causal_memory_effect_supported_by_counterfactual" {
            let mut observed_effect = false;
            let mut counterfactual_no_effect = false;
            for reference in supporting {
                let id = reference.as_str().expect("validated above");
                let event = local_events.get(id).expect("validated above");
                observed_effect |= event.get("stage").and_then(Value::as_str)
                    == Some("external_effect_observation")
                    && event.get("outcome").and_then(Value::as_str) == Some("effect_confirmed")
                    && event.get("evidence").and_then(Value::as_array).is_some_and(|evidence| {
                        evidence.iter().any(|item| {
                            item.get("observer").and_then(Value::as_str) == Some("independent_sink")
                                && item.get("independence_status").and_then(Value::as_str)
                                    == Some("independent")
                        })
                    });
                counterfactual_no_effect |= event.get("stage").and_then(Value::as_str)
                    == Some("counterfactual_replay")
                    && event.get("outcome").and_then(Value::as_str) == Some("effect_absent");
            }
            if !observed_effect || !counterfactual_no_effect {
                return Err(format!(
                    "{trace_id}: causal attribution invariant needs an independent confirmed effect and a no-effect counterfactual"
                ));
            }
        }

        if invariant_id == "residual_influence_absent_after_repair" {
            let has_residual_absence = supporting.iter().any(|reference| {
                let id = reference.as_str().expect("validated above");
                let event = local_events.get(id).expect("validated above");
                event.get("stage").and_then(Value::as_str) == Some("residual_influence_check")
                    && event.get("outcome").and_then(Value::as_str)
                        == Some("residual_influence_absent")
            });
            if !has_residual_absence {
                return Err(format!(
                    "{trace_id}: residual-influence invariant lacks an explicit absent result"
                ));
            }
        }
    }

    let stages = events
        .iter()
        .map(|event| {
            (
                event.get("stage").and_then(Value::as_str).unwrap_or_default(),
                event.get("outcome").and_then(Value::as_str).unwrap_or_default(),
            )
        })
        .collect::<Vec<_>>();

    if boundary == "store_only" || claim_scope == "store_only_observation" {
        if stages.iter().any(|(stage, _)| {
            matches!(*stage, "agent_decision" | "external_effect_observation")
        }) {
            return Err(format!(
                "{trace_id}: store-only scope cannot claim agent decisions or external effects"
            ));
        }
    }

    let confirmed_effect_exists = has_independent_sink_effect(events);
    if stages.iter().any(|(stage, outcome)| {
        *stage == "external_effect_observation" && *outcome == "effect_confirmed"
    }) && !confirmed_effect_exists
    {
        return Err(format!(
            "{trace_id}: confirmed external effect lacks independent sink evidence"
        ));
    }

    let disposition = field_str(trace, "expected_disposition")
        .map_err(|e| format!("{trace_id}: {e}"))?;
    if disposition == "external_effect_detected_positive_control" && !confirmed_effect_exists {
        return Err(format!(
            "{trace_id}: positive-control disposition requires an independently observed effect"
        ));
    }

    let scenario = field_str(trace, "scenario").map_err(|e| format!("{trace_id}: {e}"))?;
    match scenario {
        "erasure_restore" => {
            require_order(events, ("erasure", None), ("restore_attempt", None), &trace_id)?;
            require_order(
                events,
                ("restore_attempt", None),
                ("witness_reconciliation", None),
                &trace_id,
            )?;
            require_order(
                events,
                ("witness_reconciliation", None),
                ("post_restore_read", None),
                &trace_id,
            )?;
            if disposition == "erasure_reconciled"
                && event_index(events, "post_restore_read", Some("not_served")).is_none()
            {
                return Err(format!(
                    "{trace_id}: erasure_reconciled requires post-restore not_served"
                ));
            }
        }
        "counterfactual_repair_replay" => {
            let attempt = event_index(events, "attack_attempt", None)
                .ok_or_else(|| format!("{trace_id}: missing attack_attempt"))?;
            let record_count = events[attempt]
                .get("record_ids")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            if record_count < 2 {
                return Err(format!("{trace_id}: L2 compositional attack needs at least two records"));
            }
            if field_str(profile, "depth").map_err(|e| format!("{trace_id}: {e}"))? != "L2" {
                return Err(format!("{trace_id}: counterfactual repair fixture must declare L2"));
            }
            if event_index(events, "counterfactual_replay", Some("effect_absent")).is_none() {
                return Err(format!("{trace_id}: missing no-effect counterfactual replay"));
            }
            require_order(events, ("correction", None), ("projection_invalidation", None), &trace_id)?;
            require_order(events, ("projection_invalidation", None), ("projection_rebuild", None), &trace_id)?;
            let rebuild_index = event_index(events, "projection_rebuild", None)
                .ok_or_else(|| format!("{trace_id}: missing projection_rebuild"))?;
            let post_repair_replay_index = events
                .iter()
                .rposition(|event| {
                    event.get("stage").and_then(Value::as_str) == Some("counterfactual_replay")
                        && event.get("outcome").and_then(Value::as_str) == Some("effect_absent")
                })
                .ok_or_else(|| format!("{trace_id}: missing post-repair no-effect replay"))?;
            let residual_index = event_index(
                events,
                "residual_influence_check",
                Some("residual_influence_absent"),
            )
            .ok_or_else(|| format!("{trace_id}: missing residual-influence absence check"))?;
            if rebuild_index >= post_repair_replay_index
                || post_repair_replay_index >= residual_index
            {
                return Err(format!(
                    "{trace_id}: post-repair replay and residual-influence check must follow projection rebuild"
                ));
            }
        }
        _ => {}
    }

    state.trace_count += 1;
    Ok(())
}

fn validate_corpus(lines: &[String]) -> Result<(usize, usize), String> {
    let mut state = CorpusState::default();
    for (index, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("line {}: invalid JSON: {error}", index + 1))?;
        validate_trace(&value, index + 1, &mut state)?;
    }

    for (trace_id, pair_id) in &state.paired_controls {
        if !state.trace_ids.contains(pair_id) {
            return Err(format!(
                "{trace_id}: paired_control_trace_id '{pair_id}' does not exist in this corpus"
            ));
        }
    }

    Ok((state.trace_count, state.event_count))
}

fn run() -> Result<(), String> {
    let args = env::args().collect::<Vec<_>>();
    if args.len() != 2 {
        return Err(format!(
            "usage: {} <trace.jsonl>",
            args.first().map(String::as_str).unwrap_or("sat-memory-trace-verifier")
        ));
    }
    let file = File::open(&args[1])
        .map_err(|error| format!("cannot open '{}': {error}", args[1]))?;
    let reader = BufReader::new(file);
    let lines = reader
        .lines()
        .collect::<Result<Vec<_>, io::Error>>()
        .map_err(|error| format!("cannot read '{}': {error}", args[1]))?;
    let (traces, events) = validate_corpus(&lines)?;
    println!(
        "semantic checks passed: {traces} traces, {events} events; JSON Schema validation and external evidence verification were not performed"
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("semantic verification failed: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture_lines() -> Vec<String> {
        include_str!("../../../docs/research/fixtures/sat-memory-lifecycle-trace-v0.1.jsonl")
            .lines()
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn current_public_fixture_corpus_passes_semantic_checks() {
        let (trace_count, event_count) = validate_corpus(&fixture_lines()).expect("fixture corpus");
        assert_eq!(trace_count, 8);
        assert_eq!(event_count, 63);
    }

    #[test]
    fn duplicate_event_ids_are_rejected_within_a_trace() {
        let mut lines = fixture_lines();
        let mut first: Value = serde_json::from_str(&lines[0]).expect("JSON");
        let duplicate_id = first["events"][0]["event_id"].clone();
        first["events"][1]["event_id"] = duplicate_id;
        lines[0] = first.to_string();
        assert!(validate_corpus(&lines).is_err());
    }

    #[test]
    fn reordered_event_sequence_is_rejected() {
        let mut lines = fixture_lines();
        let mut trace: Value = serde_json::from_str(&lines[0]).expect("JSON");
        trace["events"][1]["sequence"] = json!(99);
        lines[0] = trace.to_string();
        assert!(validate_corpus(&lines).is_err());
    }

    #[test]
    fn store_only_trace_cannot_claim_agent_decision() {
        let mut lines = fixture_lines();
        let mut trace: Value = serde_json::from_str(&lines[7]).expect("JSON");
        trace["events"][3]["stage"] = json!("agent_decision");
        lines[7] = trace.to_string();
        assert!(validate_corpus(&lines).is_err());
    }

    #[test]
    fn positive_control_requires_an_independent_sink() {
        let mut lines = fixture_lines();
        let mut trace: Value = serde_json::from_str(&lines[3]).expect("JSON");
        trace["events"][5]["evidence"][0]["independence_status"] = json!("same_sut_boundary");
        lines[3] = trace.to_string();
        assert!(validate_corpus(&lines).is_err());
    }

    #[test]
    fn l2_counterfactual_trace_requires_composition_and_repair() {
        let mut lines = fixture_lines();
        let mut trace: Value = serde_json::from_str(&lines[7]).expect("JSON");
        trace["events"][0]["record_ids"] = json!(["only-one-record"]);
        lines[7] = trace.to_string();
        assert!(validate_corpus(&lines).is_err());
    }

    #[test]
    fn post_repair_counterfactual_must_follow_projection_rebuild() {
        let mut lines = fixture_lines();
        let mut trace: Value = serde_json::from_str(&lines[7]).expect("JSON");
        trace["events"][11]["stage"] = json!("retrieval_exposure");
        lines[7] = trace.to_string();
        assert!(validate_corpus(&lines).is_err());
    }

    #[test]
    fn executed_trace_requires_a_manifest_and_artifact_hashes() {
        let mut lines = fixture_lines();
        let mut trace: Value = serde_json::from_str(&lines[0]).expect("JSON");
        trace["execution_state"] = json!("executed_candidate");
        trace["run_manifest_ref"] = json!("run-manifest://example");
        lines[0] = trace.to_string();
        assert!(validate_corpus(&lines).is_err());
    }
}
