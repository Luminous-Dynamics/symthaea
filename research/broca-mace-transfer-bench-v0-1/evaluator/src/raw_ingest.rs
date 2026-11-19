// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Provider-neutral raw-session ingestion and bitemporal replay for MACE.
// Transcript claims are never their own execution, outcome, or source authority.

use std::collections::{BTreeMap, BTreeSet};

use broca_mace_policy::{
    Action, Experience, Objective, Regime, Scope, SourceRootDigest,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

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
    raw_line_sha256: String,
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
}

#[derive(Clone, Copy, Debug)]
pub(super) struct VerifiedSource {
    pub source_family: u16,
    pub source_root_id: u32,
    pub source_root_digest: SourceRootDigest,
    pub measurement_error_bound: f64,
    pub trusted: bool,
}
pub(super) type VerifiedSourceRegistry = BTreeMap<String, VerifiedSource>;

#[derive(Clone, Debug)]
pub(super) struct VerifiedActionReceipt {
    pub plan_event_id: u32,
    pub action_id: String,
    pub action: Action,
    pub scope: Scope,
    pub dispatched_at_ms: u64,
    pub prediction_registered_at_ms: u64,
    pub intervention: bool,
    pub confounders_controlled: bool,
    pub delayed_horizon_complete: bool,
    pub trusted: bool,
}
pub(super) type VerifiedActionRegistry = BTreeMap<String, VerifiedActionReceipt>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ConsolidationDisposition {
    RetainRaw,
    CompressWithSourceLinks,
    PromoteToProcedure,
    PromoteToRule,
    Quarantine,
    Supersede,
    ForgetFromActiveIndex,
    EraseForPrivacy,
}

#[derive(Clone, Debug)]
pub(super) struct VerifiedConsolidationReceipt {
    pub disposition: ConsolidationDisposition,
    pub source_event_ids: Vec<u32>,
    pub decision_digest: String,
    pub verified: bool,
}
pub(super) type VerifiedConsolidationRegistry =
    BTreeMap<u32, VerifiedConsolidationReceipt>;

#[derive(Clone, Debug)]
pub(super) struct VerifiedOutcomeReceipt {
    pub event_id: u32,
    pub action_id: String,
    pub observed_utility: f64,
    pub reconciled: bool,
}
pub(super) type VerifiedOutcomeRegistry = BTreeMap<String, VerifiedOutcomeReceipt>;

#[derive(Clone, Debug, Default)]
pub(super) struct IngestReport {
    pub raw_event_count: usize,
    pub raw_session_sha256: String,
    pub raw_event_digests: Vec<(u32, String)>,
    pub visible_event_count: usize,
    pub tool_success_without_verified_world_observation: usize,
    pub unverified_outcome_receipt_count: usize,
    pub unpromoted_observation_count: usize,
    pub raw_payload_outcome_mismatch_count: usize,
    pub retracted_observation_count: usize,
    pub experiences: Vec<Experience>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
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
            raw_line_sha256: sha256_hex(line.as_bytes()),
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
    event_id: u32,
    action: Action,
    scope: Scope,
    occurred_at_ms: u64,
    recorded_at_ms: u64,
    sequence: u64,
}

fn apply_corrections(
    events: &[RawEvent],
    valid_at_ms: u64,
    known_at_ms: u64,
) -> Result<BTreeSet<u32>, String> {
    let by_id: BTreeMap<u32, &RawEvent> =
        events.iter().map(|event| (event.event_id, event)).collect();
    let mut corrected = BTreeSet::new();

    for event in events.iter().filter(|event| {
        event.event_type == "source_retraction"
            || event.event_type == "source_supersession"
    }) {
        let raw_target = event
            .integer("target_event_id")
            .ok_or_else(|| format!("correction {} has no target_event_id", event.event_id))?;
        let target_id = u32::try_from(raw_target)
            .map_err(|_| format!("correction {} target exceeds u32", event.event_id))?;
        let target = by_id
            .get(&target_id)
            .ok_or_else(|| {
                format!("correction {} targets unknown event {}", event.event_id, target_id)
            })?;

        if target.event_type != "world_observation" {
            return Err(format!(
                "correction {} target {} is not a world observation",
                event.event_id, target_id
            ));
        }
        if event.sequence <= target.sequence
            || event.recorded_at_ms < target.recorded_at_ms
            || event.occurred_at_ms < target.occurred_at_ms
        {
            return Err(format!(
                "correction {} predates source event {}",
                event.event_id, target_id
            ));
        }

        let effective_from = event.valid_from_ms.unwrap_or(event.occurred_at_ms);
        let effective_at_valid_time = effective_from <= valid_at_ms
            && event.valid_to_ms.is_none_or(|end| valid_at_ms < end);
        let known_by_query = event.recorded_at_ms <= known_at_ms;

        if known_by_query && effective_at_valid_time {
            corrected.insert(target_id);
        }
    }

    Ok(corrected)
}

pub(super) fn derive_as_known(
    input: &str,
    valid_at_ms: u64,
    known_at_ms: u64,
    verified_actions: &VerifiedActionRegistry,
    verified_outcomes: &VerifiedOutcomeRegistry,
    verified_sources: &VerifiedSourceRegistry,
    verified_consolidations: &VerifiedConsolidationRegistry,
) -> Result<IngestReport, String> {
    let events = parse_jsonl(input)?;
    let retracted_ids = apply_corrections(&events, valid_at_ms, known_at_ms)?;
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
                    event_id: event.event_id,
                    action,
                    scope,
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
        raw_session_sha256: sha256_hex(input.as_bytes()),
        raw_event_digests: events
            .iter()
            .map(|event| (event.event_id, event.raw_line_sha256.clone()))
            .collect(),
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
            || event.sequence <= dispatch.sequence
            || event.occurred_at_ms < dispatch.occurred_at_ms
        {
            continue;
        }

        let execution_key = dispatch.string("execution_receipt_digest").unwrap_or("");
        let execution = verified_actions.get(execution_key);
        let execution_matches = execution.is_some_and(|receipt| {
            receipt.trusted
                && receipt.plan_event_id == plan.event_id
                && receipt.action_id == action_id
                && receipt.action == plan.action
                && receipt.scope == plan.scope
                && receipt.dispatched_at_ms == dispatch.occurred_at_ms
        });
        let pre_registered = execution_matches
            && execution.is_some_and(|receipt| {
                receipt.prediction_registered_at_ms <= plan.occurred_at_ms
                    && plan.occurred_at_ms < dispatch.occurred_at_ms
                    && plan.recorded_at_ms < dispatch.recorded_at_ms
            });

        let outcome_key = event.string("observation_receipt").unwrap_or("");
        let Some(outcome) = verified_outcomes.get(outcome_key).filter(|receipt| {
            receipt.event_id == event.event_id && receipt.action_id == action_id
        }) else {
            report.unverified_outcome_receipt_count += 1;
            continue;
        };
        if event
            .number("observed_utility")
            .is_some_and(|claimed| (claimed - outcome.observed_utility).abs() > f64::EPSILON)
        {
            report.raw_payload_outcome_mismatch_count += 1;
        }

        let Some(consolidation) = verified_consolidations.get(&event.event_id).filter(|receipt| {
            receipt.verified
                && !receipt.decision_digest.is_empty()
                && receipt.disposition == ConsolidationDisposition::PromoteToProcedure
                && receipt.source_event_ids.contains(&event.event_id)
        }) else {
            report.unpromoted_observation_count += 1;
            continue;
        };
        let _consolidation_receipt_digest = &consolidation.decision_digest;

        let source_key = event.string("source_artifact_digest").unwrap_or("");
        let source = verified_sources.get(source_key);
        let source_trusted = source.is_some_and(|receipt| {
            receipt.trusted
                && receipt.source_family != 0
                && receipt.source_root_digest.is_known()
        });
        let source_family = source.map_or(0, |receipt| receipt.source_family);
        let source_root_id = source.map_or(0, |receipt| receipt.source_root_id);
        let source_root_digest =
            source.map_or(SourceRootDigest::UNKNOWN, |receipt| receipt.source_root_digest);
        let measurement_error_bound =
            source.map_or(1.0, |receipt| receipt.measurement_error_bound);
        let source_retracted = retracted_ids.contains(&event.event_id);
        if source_retracted {
            report.retracted_observation_count += 1;
        }

        report.experiences.push(Experience {
            event_id: event.event_id,
            source_family,
            source_root_id,
            source_root_digest,
            scope: plan.scope,
            action: plan.action,
            observed_utility: outcome.observed_utility,
            measurement_error_bound,
            intervention: execution_matches
                && execution.is_some_and(|receipt| receipt.intervention),
            prediction_pre_registered: pre_registered,
            outcome_reconciled: outcome.reconciled,
            source_trusted,
            source_retracted,
            confounders_controlled: execution_matches
                && execution.is_some_and(|receipt| receipt.confounders_controlled),
            delayed_horizon_complete: execution_matches
                && execution.is_some_and(|receipt| receipt.delayed_horizon_complete),
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
    report.tool_success_without_verified_world_observation = tool_results
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
        let rows = [
            (1, 2, 3, "a1", "A", 0.92, "digest-a1"),
            (4, 5, 6, "a2", "A", 0.90, "digest-a2"),
            (7, 8, 9, "b1", "B", 0.32, "digest-b1"),
            (10, 11, 12, "b2", "B", 0.34, "digest-b2"),
        ];
        let mut events = Vec::new();
        for (plan_id, dispatch_id, observation_id, action_id, action, utility, digest) in rows {
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
                    "intervention": false,
                    "prediction_registered_at_ms": 999,
                    "confounders_controlled": false,
                    "delayed_horizon_complete": false
                }),
            ));
            events.push(raw_event(
                dispatch_id,
                u64::from(dispatch_id),
                "action_dispatched",
                20,
                20,
                json!({
                    "action_id": action_id,
                    "execution_receipt_digest": format!("exec-{action_id}")
                }),
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

    fn action_registry() -> VerifiedActionRegistry {
        let rows = [
            ("a1", 1, Action::A),
            ("a2", 4, Action::A),
            ("b1", 7, Action::B),
            ("b2", 10, Action::B),
        ];
        rows.into_iter()
            .map(|(action_id, plan_event_id, action)| {
                (
                    format!("exec-{action_id}"),
                    VerifiedActionReceipt {
                        plan_event_id,
                        action_id: action_id.to_owned(),
                        action,
                        scope: Scope {
                            regime: Regime::Baseline,
                            objective: Objective::Quality,
                        },
                        dispatched_at_ms: 20,
                        prediction_registered_at_ms: 9,
                        intervention: true,
                        confounders_controlled: true,
                        delayed_horizon_complete: true,
                        trusted: true,
                    },
                )
            })
            .collect()
    }

    fn outcome_registry() -> VerifiedOutcomeRegistry {
        let rows = [
            (3, "a1", 0.92),
            (6, "a2", 0.90),
            (9, "b1", 0.32),
            (12, "b2", 0.34),
        ];
        rows.into_iter()
            .map(|(event_id, action_id, observed_utility)| {
                (
                    format!("receipt-{event_id}"),
                    VerifiedOutcomeReceipt {
                        event_id,
                        action_id: action_id.to_owned(),
                        observed_utility,
                        reconciled: true,
                    },
                )
            })
            .collect()
    }

    fn consolidation_registry() -> VerifiedConsolidationRegistry {
        [3_u32, 6, 9, 12]
            .into_iter()
            .map(|event_id| {
                (
                    event_id,
                    VerifiedConsolidationReceipt {
                        disposition: ConsolidationDisposition::PromoteToProcedure,
                        source_event_ids: vec![event_id],
                        decision_digest: format!("decision-digest-{event_id}"),
                        verified: true,
                    },
                )
            })
            .collect()
    }

    fn fixture_root_digest(seed: u32) -> SourceRootDigest {
        let mut bytes = [0_u8; 32];
        bytes[..4].copy_from_slice(&seed.to_be_bytes());
        SourceRootDigest::from_bytes(bytes)
    }

    fn source_registry() -> VerifiedSourceRegistry {
        BTreeMap::from([
            ("digest-a1".to_owned(), VerifiedSource {
                source_family: 10,
                source_root_id: 100,
                source_root_digest: fixture_root_digest(100),
                measurement_error_bound: 0.02,
                trusted: true,
            }),
            ("digest-a2".to_owned(), VerifiedSource {
                source_family: 11,
                source_root_id: 101,
                source_root_digest: fixture_root_digest(101),
                measurement_error_bound: 0.02,
                trusted: true,
            }),
            ("digest-b1".to_owned(), VerifiedSource {
                source_family: 20,
                source_root_id: 200,
                source_root_digest: fixture_root_digest(200),
                measurement_error_bound: 0.02,
                trusted: true,
            }),
            ("digest-b2".to_owned(), VerifiedSource {
                source_family: 21,
                source_root_id: 201,
                source_root_digest: fixture_root_digest(201),
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

    fn derive_fixture(input: &str, valid_at: u64, known_at: u64) -> Result<IngestReport, String> {
        derive_as_known(
            input,
            valid_at,
            known_at,
            &action_registry(),
            &outcome_registry(),
            &source_registry(),
            &consolidation_registry(),
        )
    }

    fn derive_with_sources(
        input: &str,
        valid_at: u64,
        known_at: u64,
        sources: &VerifiedSourceRegistry,
    ) -> Result<IngestReport, String> {
        derive_as_known(
            input,
            valid_at,
            known_at,
            &action_registry(),
            &outcome_registry(),
            sources,
            &consolidation_registry(),
        )
    }

    fn derive_with_consolidations(
        input: &str,
        valid_at: u64,
        known_at: u64,
        consolidations: &VerifiedConsolidationRegistry,
    ) -> Result<IngestReport, String> {
        derive_as_known(
            input,
            valid_at,
            known_at,
            &action_registry(),
            &outcome_registry(),
            &source_registry(),
            consolidations,
        )
    }

    #[test]
    fn raw_jsonl_ingestion_derives_a_qualified_memory_decision() {
        let raw = jsonl(observation_events());
        let report = derive_fixture(&raw, 100, 100).expect("valid raw session");
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
    fn tool_success_without_verified_world_observation_never_becomes_outcome_memory() {
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
        let report = derive_fixture(&jsonl(events), 100, 100).unwrap();
        assert_eq!(report.tool_success_without_verified_world_observation, 1);
        assert_eq!(report.experiences.len(), 3);
    }

    #[test]
    fn late_arriving_observation_is_not_visible_before_knowledge_time() {
        let mut events = observation_events();
        events[2]["recorded_at_ms"] = json!(150);
        let raw = jsonl(events);
        let early = derive_fixture(&raw, 100, 100).unwrap();
        let late = derive_fixture(&raw, 100, 160).unwrap();
        assert_eq!(early.experiences.len(), 3);
        assert_eq!(late.experiences.len(), 4);
    }

    #[test]
    fn retraction_changes_as_known_state_without_rewriting_history() {
        let mut events = observation_events();
        let mut correction = raw_event(
            13,
            13,
            "source_retraction",
            150,
            150,
            json!({ "target_event_id": 3, "reason": "independent correction" }),
        );
        correction["valid_from_ms"] = json!(0);
        events.push(correction);
        let raw = jsonl(events);
        let before = derive_fixture(&raw, 100, 100).unwrap();
        let after = derive_fixture(&raw, 100, 160).unwrap();
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
    fn retroactive_correction_is_applied_only_when_known_and_effective() {
        let mut events = observation_events();
        let mut correction = raw_event(
            13,
            13,
            "source_retraction",
            150,
            150,
            json!({ "target_event_id": 3, "reason": "retroactive correction" }),
        );
        correction["valid_from_ms"] = json!(0);
        events.push(correction);
        let raw = jsonl(events);

        let before_knowledge = derive_fixture(&raw, 100, 149).unwrap();
        let after_knowledge = derive_fixture(&raw, 100, 150).unwrap();
        assert!(before_knowledge
            .experiences
            .iter()
            .any(|item| item.event_id == 3));
        assert!(after_knowledge
            .experiences
            .iter()
            .find(|item| item.event_id == 3)
            .unwrap()
            .source_retracted);
    }

    #[test]
    fn known_future_effective_correction_does_not_rewrite_earlier_valid_time() {
        let mut events = observation_events();
        let mut correction = raw_event(
            13,
            13,
            "source_retraction",
            150,
            150,
            json!({ "target_event_id": 3, "reason": "future-effective correction" }),
        );
        correction["valid_from_ms"] = json!(150);
        events.push(correction);
        let raw = jsonl(events);

        let before_effective_time = derive_fixture(&raw, 100, 160).unwrap();
        let at_effective_time = derive_fixture(&raw, 150, 160).unwrap();
        assert!(before_effective_time
            .experiences
            .iter()
            .any(|item| item.event_id == 3));
        assert!(at_effective_time
            .experiences
            .iter()
            .find(|item| item.event_id == 3)
            .unwrap()
            .source_retracted);
    }

    #[test]
    fn correction_with_unknown_target_fails_closed() {
        let mut events = observation_events();
        events.push(raw_event(
            13,
            13,
            "source_retraction",
            150,
            150,
            json!({ "target_event_id": 999 }),
        ));
        assert!(derive_fixture(&jsonl(events), 100, 160)
            .unwrap_err()
            .contains("targets unknown event 999"));
    }

    #[test]
    fn correction_cannot_retract_a_non_observation_event() {
        let mut events = observation_events();
        events.push(raw_event(
            13,
            13,
            "source_retraction",
            150,
            150,
            json!({ "target_event_id": 1 }),
        ));
        assert!(derive_fixture(&jsonl(events), 100, 160)
            .unwrap_err()
            .contains("is not a world observation"));
    }

    #[test]
    fn correction_cannot_predate_the_source_observation() {
        let mut events = observation_events();
        events.push(raw_event(
            13,
            13,
            "source_retraction",
            25,
            25,
            json!({ "target_event_id": 3 }),
        ));
        assert!(derive_fixture(&jsonl(events), 100, 160)
            .unwrap_err()
            .contains("predates source event 3"));
    }

    #[test]
    fn raw_payload_cannot_self_attest_action_causality_or_measurement_value() {
        let mut events = observation_events();
        events[2]["payload"]["observed_utility"] = json!(0.01);
        let report = derive_fixture(&jsonl(events), 100, 100).unwrap();
        let item = report.experiences.iter().find(|item| item.event_id == 3).unwrap();
        assert_eq!(item.observed_utility, 0.92);
        assert_eq!(report.raw_payload_outcome_mismatch_count, 1);
        assert!(item.intervention);
        assert!(item.prediction_pre_registered);
        assert!(item.confounders_controlled);
    }

    #[test]
    fn missing_execution_receipt_cannot_authorize_causal_use() {
        let mut events = observation_events();
        events[1]["payload"]["execution_receipt_digest"] = json!("not-in-registry");
        let report = derive_fixture(&jsonl(events), 100, 100).unwrap();
        let item = report.experiences.iter().find(|item| item.event_id == 3).unwrap();
        assert!(!item.intervention);
        assert!(!item.prediction_pre_registered);
        assert!(!item.confounders_controlled);
        let query = Query {
            scope: Scope {
                regime: Regime::Baseline,
                objective: Objective::Quality,
            },
        };
        assert_eq!(
            decide(Policy::CausalQualified, &report.experiences, query).action,
            None
        );
    }

    #[test]
    fn missing_or_mismatched_observation_receipt_is_not_reconciled() {
        let mut events = observation_events();
        events[2]["payload"]["observation_receipt"] = json!("forged-receipt");
        let report = derive_fixture(&jsonl(events), 100, 100).unwrap();
        assert_eq!(report.unverified_outcome_receipt_count, 1);
        assert!(!report.experiences.iter().any(|item| item.event_id == 3));
    }

    #[test]
    fn unregistered_source_artifact_is_not_trusted_or_independent() {
        let mut events = observation_events();
        events[2]["payload"]["source_artifact_digest"] = json!("forged-digest");
        let report = derive_fixture(&jsonl(events), 100, 100).unwrap();
        let item = report.experiences.iter().find(|item| item.event_id == 3).unwrap();
        assert!(!item.source_trusted);
        assert_eq!(item.source_root_id, 0);
        assert!(!item.source_root_digest.is_known());
    }

    #[test]
    fn malformed_duplicate_or_out_of_order_raw_events_fail_closed() {
        let mut events = observation_events();
        events[1]["sequence"] = json!(1);
        assert!(derive_fixture(&jsonl(events.clone()), 100, 100).is_err());
        events = observation_events();
        events[1]["event_id"] = json!(1);
        assert!(derive_fixture(&jsonl(events), 100, 100).is_err());
        assert!(derive_fixture("{not-json}", 100, 100).is_err());
    }

    #[test]
    fn invalid_registry_error_bound_is_rejected_by_candidate_policy() {
        let mut sources = source_registry();
        sources.insert(
            "digest-a1".to_owned(),
            VerifiedSource {
                source_family: 10,
                source_root_id: 100,
                source_root_digest: fixture_root_digest(100),
                measurement_error_bound: f64::NAN,
                trusted: true,
            },
        );
        let raw = jsonl(observation_events());
        let report = derive_with_sources(&raw, 100, 100, &sources).unwrap();
        let query = Query {
            scope: Scope {
                regime: Regime::Baseline,
                objective: Objective::Quality,
            },
        };
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
        let before = derive_fixture(&raw, 40, 100).unwrap();
        let during = derive_fixture(&raw, 60, 100).unwrap();
        assert_eq!(before.experiences.len(), 3);
        assert_eq!(during.experiences.len(), 4);
    }

    #[test]
    fn post_dispatch_action_plan_cannot_be_relabelled_as_preregistered() {
        let mut events = observation_events();
        events[0]["occurred_at_ms"] = json!(25);
        events[0]["recorded_at_ms"] = json!(25);
        let report = derive_fixture(&jsonl(events), 100, 100).unwrap();
        let item = report.experiences.iter().find(|item| item.event_id == 3).unwrap();
        assert!(!item.prediction_pre_registered);
        assert!(item.intervention);
    }

    #[test]
    fn raw_session_history_is_not_truncated_at_the_consolidation_window() {
        let mut events = observation_events();
        for offset in 0..260_u32 {
            events.push(raw_event(
                1000 + offset,
                13 + u64::from(offset),
                "assistant_note",
                40 + u64::from(offset),
                40 + u64::from(offset),
                json!({ "text": format!("later-note-{offset}") }),
            ));
        }
        let parsed = parse_jsonl(&jsonl(events)).unwrap();
        assert_eq!(parsed.len(), 272);
        assert_eq!(parsed[0].event_id, 1);
        assert!(parsed.iter().any(|event| event.event_id == 3));
    }

    #[test]
    fn raw_content_digests_are_stable_and_change_when_input_changes() {
        let first = jsonl(observation_events());
        let second = first.replace("0.92", "0.93");
        let first_report = derive_fixture(&first, 100, 100).unwrap();
        let second_report = derive_fixture(&second, 100, 100).unwrap();
        assert_eq!(first_report.raw_session_sha256.len(), 64);
        assert_eq!(first_report.raw_event_digests.len(), 12);
        assert_ne!(
            first_report.raw_session_sha256,
            second_report.raw_session_sha256
        );
    }

    #[test]
    fn raw_experience_is_not_transferable_without_verified_promotion() {
        let empty = VerifiedConsolidationRegistry::new();
        let report = derive_with_consolidations(
            &jsonl(observation_events()),
            100,
            100,
            &empty,
        )
        .unwrap();
        assert_eq!(report.experiences.len(), 0);
        assert_eq!(report.unpromoted_observation_count, 4);
    }

    #[test]
    fn forget_quarantine_and_privacy_erasure_do_not_enter_transfer_memory() {
        let mut consolidations = consolidation_registry();
        for disposition in [
            ConsolidationDisposition::ForgetFromActiveIndex,
            ConsolidationDisposition::Quarantine,
            ConsolidationDisposition::EraseForPrivacy,
        ] {
            consolidations.insert(
                3,
                VerifiedConsolidationReceipt {
                    disposition,
                    source_event_ids: vec![3],
                    decision_digest: "verified-policy-decision".to_owned(),
                    verified: true,
                },
            );
            let report = derive_with_consolidations(
                &jsonl(observation_events()),
                100,
                100,
                &consolidations,
            )
            .unwrap();
            assert!(!report.experiences.iter().any(|item| item.event_id == 3));
            assert_eq!(report.unpromoted_observation_count, 1);
        }
    }

    #[test]
    fn consolidation_receipt_must_bind_the_exact_source_event() {
        let mut consolidations = consolidation_registry();
        consolidations.insert(
            3,
            VerifiedConsolidationReceipt {
                disposition: ConsolidationDisposition::PromoteToProcedure,
                source_event_ids: vec![999],
                decision_digest: "verified-but-wrong-source".to_owned(),
                verified: true,
            },
        );
        let report = derive_with_consolidations(
            &jsonl(observation_events()),
            100,
            100,
            &consolidations,
        )
        .unwrap();
        assert!(!report.experiences.iter().any(|item| item.event_id == 3));
        assert_eq!(report.unpromoted_observation_count, 1);
    }

    #[test]
    fn raw_event_original_lines_are_retained_for_audit() {
        let raw = jsonl(observation_events());
        let events = parse_jsonl(&raw).unwrap();
        assert_eq!(events[0].raw_line, raw.lines().next().unwrap());
        assert_eq!(events[0].schema_version, 1);
        assert_eq!(events[0].session_id, "session-a");
        assert_eq!(events[0].sequence, 1);
        assert_eq!(events[0].raw_line_sha256.len(), 64);
    }

    #[test]
    fn provider_trace_records_never_become_candidate_experiences() {
        use broca_mace_ingress::providers::{normalize_capture_jsonl, Provider};

        let capture = [
            serde_json::json!({
                "schema_version": 1,
                "session_id": "anthropic-sample",
                "sequence": 1,
                "captured_at_ms": 100,
                "event": {
                    "id": "evt-tool",
                    "type": "user.tool_result",
                    "tool_use_id": "call-1",
                    "content": "the tool reports success"
                }
            }).to_string(),
            serde_json::json!({
                "schema_version": 1,
                "session_id": "anthropic-sample",
                "sequence": 2,
                "captured_at_ms": 110,
                "event": {
                    "id": "evt-idle",
                    "type": "session.status_idle",
                    "stop_reason": {"type": "end_turn"}
                }
            }).to_string(),
        ].join("\n");

        let batch = normalize_capture_jsonl(
            Provider::AnthropicManagedAgents,
            "anthropic-sample",
            &capture
        ).expect("provider capture normalizes");
        let mace_jsonl = batch.to_mace_jsonl().expect("trace metadata exports");

        let empty_actions = BTreeMap::new();
        let empty_outcomes = BTreeMap::new();
        let empty_sources = BTreeMap::new();
        let empty_consolidations = BTreeMap::new();
        let report = derive_as_known(
            &mace_jsonl,
            200,
            200,
            &empty_actions,
            &empty_outcomes,
            &empty_sources,
            &empty_consolidations
        ).expect("trace-only MACE records parse");

        assert_eq!(report.raw_event_count, 2);
        assert_eq!(report.visible_event_count, 2);
        assert!(report.experiences.is_empty());
    }

}
