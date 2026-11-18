// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Provider-neutral raw-session ingestion and bitemporal replay for MACE.
// Tool success is not treated as an observed world outcome.

use std::collections::{BTreeMap, BTreeSet};

use broca_mace_policy::{Action, Experience, Objective, Regime, Scope};
use serde_json::Value;

#[derive(Clone, Debug)]
struct RawEvent {
    schema_version: u64,
    event_id: u32,
    session_id: String,
    sequence: u64,
    event_type: String,
    occurred_at_ms: u64,
    recorded_at_ms: u64,
    valid_from_ms: Option<u64>,
    valid_to_ms: Option<u64>,
    payload: Value,
    raw_line: String,
}

impl RawEvent {
    fn visible_at(&self, valid_at_ms: u64, known_at_ms: u64) -> bool {
        self.recorded_at_ms <= known_at_ms
            && self.occurred_at_ms <= valid_at_ms
            && self.valid_from_ms.is_none_or(|from| from <= valid_at_ms)
            && self.valid_to_ms.is_none_or(|to| valid_at_ms < to)
    }

    fn string(&self, key: &str) -> Option<&str> {
        self.payload.get(key).and_then(Value::as_str)
    }

    fn number(&self, key: &str) -> Option<f64> {
        self.payload.get(key).and_then(Value::as_f64)
    }

    fn integer(&self, key: &str) -> Option<u64> {
        self.payload.get(key).and_then(Value::as_u64)
    }

    fn boolean(&self, key: &str) -> bool {
        self.payload.get(key).and_then(Value::as_bool).unwrap_or(false)
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct VerifiedSource {
    pub source_family: u16,
    pub source_root_id: u32,
    pub measurement_error_bound: f64,
    pub trusted: bool,
}

pub(super) type VerifiedSourceRegistry = BTreeMap<String, VerifiedSource>;

#[derive(Clone, Debug, Default)]
pub(super) struct IngestReport {
    pub raw_event_count: usize,
    pub visible_event_count: usize,
    pub tool_success_without_world_observation: usize,
    pub retracted_observation_count: usize,
    pub experiences: Vec<Experience>,
}

fn parse_jsonl(input: &str) -> Result<Vec<RawEvent>, String> {
    let mut events = Vec::new();
    let mut event_ids = BTreeSet::new();
    let mut previous_sequence = None;
    let mut session_id: Option<String> = None;

    for (line_index, line) in input.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line)
            .map_err(|error| format!("line {}: invalid JSON: {error}", line_index + 1))?;
        let object = value
            .as_object()
            .ok_or_else(|| format!("line {}: event must be an object", line_index + 1))?;
        let required_u64 = |key: &str| {
            object
                .get(key)
                .and_then(Value::as_u64)
                .ok_or_else(|| format!("line {}: invalid {key}", line_index + 1))
        };
        let required_string = |key: &str| {
            object
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_owned)
                .ok_or_else(|| format!("line {}: invalid {key}", line_index + 1))
        };

        let schema_version = required_u64("schema_version")?;
        if schema_version != 1 {
            return Err(format!("line {}: unsupported schema version", line_index + 1));
        }
        let event_id = u32::try_from(required_u64("event_id")?)
            .map_err(|_| format!("line {}: event_id exceeds u32", line_index + 1))?;
        if !event_ids.insert(event_id) {
            return Err(format!("line {}: duplicate event_id", line_index + 1));
        }

        let current_session = required_string("session_id")?;
        if session_id.as_ref().is_some_and(|expected| expected != &current_session) {
            return Err(format!("line {}: mixed session IDs", line_index + 1));
        }
        session_id.get_or_insert_with(|| current_session.clone());

        let sequence = required_u64("sequence")?;
        if previous_sequence.is_some_and(|previous| sequence <= previous) {
            return Err(format!("line {}: sequence is not increasing", line_index + 1));
        }
        previous_sequence = Some(sequence);

        let occurred_at_ms = required_u64("occurred_at_ms")?;
        let recorded_at_ms = required_u64("recorded_at_ms")?;
        if recorded_at_ms < occurred_at_ms {
            return Err(format!("line {}: knowledge time precedes event time", line_index + 1));
        }
        let optional_time = |key: &str| -> Result<Option<u64>, String> {
            match object.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(value) => value.as_u64().map(Some).ok_or_else(|| {
                    format!("line {}: invalid {key}", line_index + 1)
                }),
            }
        };
        let valid_from_ms = optional_time("valid_from_ms")?;
        let valid_to_ms = optional_time("valid_to_ms")?;
        if matches!((valid_from_ms, valid_to_ms), (Some(from), Some(to)) if from >= to) {
            return Err(format!("line {}: invalid valid-time interval", line_index + 1));
        }
        let payload = object
            .get("payload")
            .cloned()
            .filter(Value::is_object)
            .ok_or_else(|| format!("line {}: payload must be an object", line_index + 1))?;

        events.push(RawEvent {
            schema_version,
            event_id,
            session_id: current_session,
            sequence,
            event_type: required_string("event_type")?,
            occurred_at_ms,
            recorded_at_ms,
            valid_from_ms,
            valid_to_ms,
            payload,
            raw_line: line.to_owned(),
        });
    }
    if events.is_empty() {
        return Err("session contains no events".to_owned());
    }
    Ok(events)
}

fn parse_scope(payload: &Value) -> Option<Scope> {
    let regime = match payload.get("regime")?.as_str()? {
        "baseline" => Regime::Baseline,
        "shifted" => Regime::Shifted,
        _ => return None,
    };
    let objective = match payload.get("objective")?.as_str()? {
        "quality" => Objective::Quality,
        "cost" => Objective::Cost,
        "speed" => Objective::Speed,
        _ => return None,
    };
    Some(Scope { regime, objective })
}

fn parse_action(payload: &Value) -> Option<Action> {
    match payload.get("action")?.as_str()? {
        "A" => Some(Action::A),
        "B" => Some(Action::B),
        _ => None,
    }
}

#[derive(Clone, Debug)]
struct ActionPlan {
    action: Action,
    scope: Scope,
    intervention: bool,
    prediction_registered_at_ms: Option<u64>,
    confounders_controlled: bool,
    delayed_horizon_complete: bool,
    occurred_at_ms: u64,
    recorded_at_ms: u64,
    sequence: u64,
}

fn apply_corrections(events: &[RawEvent], known_at_ms: u64) -> BTreeSet<u32> {
    events
        .iter()
        .filter(|event| {
            event.recorded_at_ms <= known_at_ms
                && (event.event_type == "source_retraction"
                    || event.event_type == "source_supersession")
        })
        .filter_map(|event| {
            event.integer("target_event_id")
                .and_then(|id| u32::try_from(id).ok())
        })
        .collect()
}

/// Derive candidate-visible experiences at (valid_at, known_at).
/// Trust roots/error bounds come from the external verified-source registry.
pub(super) fn derive_as_known(
    input: &str,
    valid_at_ms: u64,
    known_at_ms: u64,
    verified_sources: &VerifiedSourceRegistry,
) -> Result<IngestReport, String> {
    let events = parse_jsonl(input)?;
    let retracted_ids = apply_corrections(&events, known_at_ms);
    let visible: Vec<&RawEvent> = events
        .iter()
        .filter(|event| event.visible_at(valid_at_ms, known_at_ms))
        .collect();

    let mut plans = BTreeMap::<String, ActionPlan>::new();
    let mut dispatches = BTreeMap::<String, &RawEvent>::new();
    let mut tool_results = BTreeMap::<String, usize>::new();
    for event in &visible {
        match event.event_type.as_str() {
            "action_pre_registered" => {
                let Some(action_id) = event.string("action_id") else {
                    continue;
                };
                let Some(action) = parse_action(&event.payload) else {
                    continue;
                };
                let Some(scope) = parse_scope(&event.payload) else {
                    continue;
                };
                let plan = ActionPlan {
                    action,
                    scope,
                    intervention: event.boolean("intervention"),
                    prediction_registered_at_ms: event.integer("prediction_registered_at_ms"),
                    confounders_controlled: event.boolean("confounders_controlled"),
                    delayed_horizon_complete: event.boolean("delayed_horizon_complete"),
                    occurred_at_ms: event.occurred_at_ms,
                    recorded_at_ms: event.recorded_at_ms,
                    sequence: event.sequence,
                };
                if plans.insert(action_id.to_owned(), plan).is_some() {
                    return Err(format!("duplicate action plan {action_id}"));
                }
            }
            "action_dispatched" => {
                if let Some(action_id) = event.string("action_id") {
                    if dispatches.insert(action_id.to_owned(), event).is_some() {
                        return Err(format!("duplicate dispatch {action_id}"));
                    }
                }
            }
            "tool_result" => {
                if let Some(action_id) = event.string("action_id") {
                    *tool_results.entry(action_id.to_owned()).or_default() += 1;
                }
            }
            _ => {}
        }
    }

    let mut report = IngestReport {
        raw_event_count: events.len(),
        visible_event_count: visible.len(),
        ..IngestReport::default()
    };
    for event in visible {
        if event.event_type != "world_observation" {
            continue;
        }
        let Some(action_id) = event.string("action_id") else {
            continue;
        };
        let Some(plan) = plans.get(action_id) else {
            continue;
        };
        let Some(dispatch) = dispatches.get(action_id) else {
            continue;
        };
        if plan.sequence >= dispatch.sequence
            || plan.occurred_at_ms > dispatch.occurred_at_ms
            || event.sequence <= dispatch.sequence
            || event.occurred_at_ms < dispatch.occurred_at_ms
        {
            continue;
        }

        let pre_registered = plan
            .prediction_registered_at_ms
            .is_some_and(|registered| {
                registered <= plan.occurred_at_ms
                    && plan.recorded_at_ms < dispatch.recorded_at_ms
            });
        let receipt_present = event
            .string("observation_receipt")
            .is_some_and(|receipt| !receipt.trim().is_empty());
        let digest = event.string("source_artifact_digest").unwrap_or("");
        let verified = verified_sources.get(digest);
        let source_trusted =
            verified.is_some_and(|source| source.trusted && source.source_root_id != 0);
        let source_family = verified.map_or(0, |source| source.source_family);
        let source_root_id = verified.map_or(0, |source| source.source_root_id);
        let measurement_error_bound =
            verified.map_or(1.0, |source| source.measurement_error_bound);
        let Some(observed_utility) = event.number("observed_utility") else {
            continue;
        };
        let was_retracted = retracted_ids.contains(&event.event_id);
        if was_retracted {
            report.retracted_observation_count += 1;
        }
        report.experiences.push(Experience {
            event_id: event.event_id,
            source_family,
            source_root_id,
            scope: plan.scope,
            action: plan.action,
            observed_utility,
            measurement_error_bound,
            intervention: plan.intervention,
            prediction_pre_registered: pre_registered,
            outcome_reconciled: receipt_present,
            source_trusted,
            source_retracted: was_retracted,
            confounders_controlled: plan.confounders_controlled,
            delayed_horizon_complete: plan.delayed_horizon_complete,
        });
    }

    let observed_action_ids: BTreeSet<&str> = report
        .experiences
        .iter()
        .filter_map(|experience| {
            events.iter().find_map(|event| {
                (event.event_id == experience.event_id)
                    .then(|| event.string("action_id"))
                    .flatten()
            })
        })
        .collect();
    report.tool_success_without_world_observation = tool_results
        .iter()
        .filter(|(action_id, _)| !observed_action_ids.contains(action_id.as_str()))
        .map(|(_, count)| *count)
        .sum();
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use broca_mace_policy::{decide, Policy, Query};
    use serde_json::json;

    fn raw_event(
        event_id: u32,
        sequence: u64,
        event_type: &str,
        occurred_at_ms: u64,
        recorded_at_ms: u64,
        payload: Value,
    ) -> Value {
        json!({
            "schema_version": 1,
            "event_id": event_id,
            "session_id": "session-a",
            "sequence": sequence,
            "event_type": event_type,
            "occurred_at_ms": occurred_at_ms,
            "recorded_at_ms": recorded_at_ms,
            "payload": payload
        })
    }

    fn observation_events() -> Vec<Value> {
        let mut events = Vec::new();
        let rows = [
            (1, 2, 3, "a1", "A", 0.92, "digest-a1"),
            (4, 5, 6, "a2", "A", 0.90, "digest-a2"),
            (7, 8, 9, "b1", "B", 0.32, "digest-b1"),
            (10, 11, 12, "b2", "B", 0.34, "digest-b2"),
        ];
        for (plan_id, dispatch_id, observation_id, action_id, action, utility, digest)
            in rows
        {
            events.push(raw_event(
                plan_id,
                u64::from(plan_id),
                "action_pre_registered",
                10,
                10,
                json!({
                    "action_id": action_id,
                    "action": action,
                    "regime": "baseline",
                    "objective": "quality",
                    "intervention": true,
                    "prediction_registered_at_ms": 9,
                    "confounders_controlled": true,
                    "delayed_horizon_complete": true
                }),
            ));
            events.push(raw_event(
                dispatch_id,
                u64::from(dispatch_id),
                "action_dispatched",
                20,
                20,
                json!({ "action_id": action_id }),
            ));
            events.push(raw_event(
                observation_id,
                u64::from(observation_id),
                "world_observation",
                30,
                30,
                json!({
                    "action_id": action_id,
                    "observed_utility": utility,
                    "observation_receipt": format!("receipt-{observation_id}"),
                    "source_artifact_digest": digest
                }),
            ));
        }
        events
    }

    fn registry() -> VerifiedSourceRegistry {
        BTreeMap::from([
            ("digest-a1".to_owned(), VerifiedSource {
                source_family: 10,
                source_root_id: 100,
                measurement_error_bound: 0.02,
                trusted: true,
            }),
            ("digest-a2".to_owned(), VerifiedSource {
                source_family: 11,
                source_root_id: 101,
                measurement_error_bound: 0.02,
                trusted: true,
            }),
            ("digest-b1".to_owned(), VerifiedSource {
                source_family: 20,
                source_root_id: 200,
                measurement_error_bound: 0.02,
                trusted: true,
            }),
            ("digest-b2".to_owned(), VerifiedSource {
                source_family: 21,
                source_root_id: 201,
                measurement_error_bound: 0.02,
                trusted: true,
            }),
        ])
    }

    fn jsonl(events: Vec<Value>) -> String {
        events
            .into_iter()
            .map(|event| serde_json::to_string(&event).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn raw_jsonl_ingestion_derives_a_qualified_memory_decision() {
        let report = derive_as_known(&jsonl(observation_events()), 100, 100, &registry())
            .expect("valid raw session");
        assert_eq!(report.raw_event_count, 12);
        assert_eq!(report.experiences.len(), 4);
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        let decision = decide(
            Policy::CausalQualified,
            &report.experiences,
            Query { scope },
        );
        assert_eq!(decision.action, Some(Action::A));
    }

    #[test]
    fn tool_success_without_world_observation_never_becomes_outcome_memory() {
        let mut events = observation_events();
        events.retain(|event| {
            event["event_type"].as_str() != Some("world_observation")
                || event["payload"]["action_id"] != "a1"
        });
        events.push(raw_event(
            13,
            13,
            "tool_result",
            40,
            40,
            json!({ "action_id": "a1", "success": true }),
        ));
        let report = derive_as_known(&jsonl(events), 100, 100, &registry())
            .expect("valid tool-only session");
        assert_eq!(report.tool_success_without_world_observation, 1);
        assert_eq!(report.experiences.len(), 3);
        assert!(report.experiences.iter().all(|experience| {
            experience.action != Action::A || experience.source_family != 10
        }));
    }

    #[test]
    fn late_arriving_observation_is_not_visible_before_knowledge_time() {
        let mut events = observation_events();
        events[2]["recorded_at_ms"] = json!(150);
        let raw = jsonl(events);
        let early = derive_as_known(&raw, 100, 100, &registry()).unwrap();
        let late = derive_as_known(&raw, 100, 160, &registry()).unwrap();
        assert_eq!(early.experiences.len(), 3);
        assert_eq!(late.experiences.len(), 4);
    }

    #[test]
    fn retraction_changes_as_known_state_without_rewriting_history() {
        let mut events = observation_events();
        events.push(raw_event(
            13,
            13,
            "source_retraction",
            150,
            150,
            json!({ "target_event_id": 3, "reason": "independent correction" }),
        ));
        let raw = jsonl(events);
        let before = derive_as_known(&raw, 100, 100, &registry()).unwrap();
        let after = derive_as_known(&raw, 100, 160, &registry()).unwrap();
        let before_item = before.experiences.iter().find(|item| item.event_id == 3).unwrap();
        let after_item = after.experiences.iter().find(|item| item.event_id == 3).unwrap();
        assert!(!before_item.source_retracted);
        assert!(after_item.source_retracted);
        let scope = Scope {
            regime: Regime::Baseline,
            objective: Objective::Quality,
        };
        assert_eq!(
            decide(Policy::CausalQualified, &before.experiences, Query { scope }).action,
            Some(Action::A)
        );
        assert_eq!(
            decide(Policy::CausalQualified, &after.experiences, Query { scope }).action,
            None
        );
    }

    #[test]
    fn unregistered_source_artifact_is_not_trusted_or_independent() {
        let mut events = observation_events();
        events[2]["payload"]["source_artifact_digest"] = json!("forged-digest");
        let report = derive_as_known(&jsonl(events), 100, 100, &registry()).unwrap();
        let unknown = report.experiences.iter().find(|item| item.event_id == 3).unwrap();
        assert!(!unknown.source_trusted);
        assert_eq!(unknown.source_root_id, 0);
    }

    #[test]
    fn malformed_duplicate_or_out_of_order_raw_events_fail_closed() {
        let mut events = observation_events();
        events[1]["sequence"] = json!(1);
        assert!(derive_as_known(&jsonl(events.clone()), 100, 100, &registry()).is_err());
        events = observation_events();
        events[1]["event_id"] = json!(1);
        assert!(derive_as_known(&jsonl(events), 100, 100, &registry()).is_err());
        assert!(derive_as_known("{not-json}", 100, 100, &registry()).is_err());
    }

    #[test]
    fn unmeasurable_registry_bound_is_rejected_by_candidate_policy() {
        let mut sources = registry();
        sources.insert(
            "digest-a1".to_owned(),
            VerifiedSource {
                source_family: 10,
                source_root_id: 100,
                measurement_error_bound: f64::NAN,
                trusted: true,
            },
        );
        let report = derive_as_known(&jsonl(observation_events()), 100, 100, &sources).unwrap();
        let item = report.experiences.iter().find(|item| item.event_id == 3).unwrap();
        assert!(!item.measurement_error_bound.is_finite());
        let query = Query {
            scope: Scope {
                regime: Regime::Baseline,
                objective: Objective::Quality,
            },
        };
        assert!(report.experiences.iter().filter(|item| item.action == Action::A).count() >= 2);
        assert_eq!(
            decide(Policy::CausalQualified, &report.experiences, query).action,
            None
        );
    }

    #[test]
    fn valid_time_interval_is_respected_independently_of_knowledge_time() {
        let mut events = observation_events();
        events[2]["valid_from_ms"] = json!(50);
        let raw = jsonl(events);
        let before = derive_as_known(&raw, 40, 100, &registry()).unwrap();
        let during = derive_as_known(&raw, 60, 100, &registry()).unwrap();
        assert_eq!(before.experiences.len(), 3);
        assert_eq!(during.experiences.len(), 4);
    }

    #[test]
    fn raw_event_original_lines_are_retained_for_audit() {
        let raw = jsonl(observation_events());
        let events = parse_jsonl(&raw).unwrap();
        assert_eq!(events[0].raw_line, raw.lines().next().unwrap());
        assert_eq!(events[0].schema_version, 1);
        assert_eq!(events[0].session_id, "session-a");
        assert_eq!(events[0].sequence, 1);
    }
}
