// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit cue-driven state transitions for ensemble performance.
//!
//! This module models the protocol between a directing layer and an ensemble:
//! a cue is emitted by a declared layer, the ensemble changes state, and a
//! designated response layer performs a declared response action within a
//! bounded time window. It is useful for studying leader/follower practice
//! (including Juju directing) without assuming one instrument, gesture,
//! vocabulary or cue convention is universal.
//!
//! The module verifies an authored cue/state/action trace. It does not infer
//! cues from audio/video, generate a performance, or claim authenticity.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Wire schema for the ensemble cue graph.
pub const ENSEMBLE_CUE_GRAPH_SCHEMA_V1: &str = "ensemble-cue-graph-v1";

/// Wire schema for a cue event sequence.
pub const ENSEMBLE_CUE_SEQUENCE_SCHEMA_V1: &str = "ensemble-cue-sequence-v1";

/// Maximum number of named states in one cue graph.
pub const MAX_CUE_STATES: usize = 128;

/// Maximum number of layers referenced by one cue graph.
pub const MAX_CUE_LAYERS: usize = 64;

/// Maximum number of transition rules in one cue graph.
pub const MAX_CUE_TRANSITIONS: usize = 1_024;

/// Maximum number of cue, response and state-change events in one sequence.
pub const MAX_CUE_SEQUENCE_EVENTS: usize = 20_000;

/// A declared ensemble state, such as full, soft, break, solo, chorus or ending.
///
/// Names are authored by the profile owner. The verifier does not interpret
/// their cultural meaning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CueStateV1 {
    pub id: String,
}

/// A transition rule from a cue source to an expected ensemble state and
/// response. The key (from_state, cue_type, source_layer_id) must be unique.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CueTransitionRuleV1 {
    pub from_state: String,
    pub cue_type: String,
    pub source_layer_id: String,
    pub to_state: String,
    pub response_layer_id: String,
    pub response_action: String,
    pub minimum_response_delay_ticks: u32,
    pub maximum_response_delay_ticks: u32,
    pub maximum_state_change_latency_ticks: u32,
}

/// Directed finite-state cue protocol.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnsembleCueGraphV1 {
    pub schema: String,
    pub initial_state: String,
    pub states: Vec<CueStateV1>,
    pub layer_ids: Vec<String>,
    pub transitions: Vec<CueTransitionRuleV1>,
}

/// A cue emitted by a named ensemble layer at an absolute tick in the sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CueEventV1 {
    pub id: String,
    pub cue_type: String,
    pub source_layer_id: String,
    pub onset_tick: u64,
}

/// A response action performed by a layer for a particular cue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CueResponseEventV1 {
    pub cue_id: String,
    pub layer_id: String,
    pub action: String,
    pub onset_tick: u64,
}

/// The ensemble state change actually realized in response to a cue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CueStateChangeEventV1 {
    pub cue_id: String,
    pub to_state: String,
    pub onset_tick: u64,
}

/// Finite, absolute-time cue sequence. Events do not wrap around the horizon;
/// callers representing a repeated cycle should tile it into this timeline.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnsembleCueSequenceV1 {
    pub schema: String,
    pub horizon_ticks: u64,
    pub cues: Vec<CueEventV1>,
    pub responses: Vec<CueResponseEventV1>,
    pub state_changes: Vec<CueStateChangeEventV1>,
}

/// Structural validation issue. Invalid graphs/sequences cannot be evaluated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CueValidationIssue {
    pub code: String,
    pub path: String,
    pub message: String,
}

/// Runtime evidence for one cue. A failed result is an evaluated failure, not
/// a malformed-input error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CueTransitionResultV1 {
    pub cue_id: String,
    pub from_state: Option<String>,
    pub expected_to_state: Option<String>,
    pub observed_to_state: Option<String>,
    pub state_change_latency_ticks: Option<u64>,
    pub response_delay_ticks: Option<u64>,
    pub response_found: bool,
    pub passed: bool,
}

/// Full verification report. Every cue is represented, including cues that
/// could not be resolved from the current state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnsembleCueVerificationV1 {
    pub schema: String,
    pub initial_state: String,
    pub final_state: String,
    pub results: Vec<CueTransitionResultV1>,
    pub all_transitions_passed: bool,
}

/// Validate a cue graph and return stable, machine-readable issues.
pub fn validate_cue_graph(
    graph: &EnsembleCueGraphV1,
) -> Result<(), Vec<CueValidationIssue>> {
    let mut issues = Vec::new();
    if graph.schema != ENSEMBLE_CUE_GRAPH_SCHEMA_V1 {
        issues.push(issue(
            "unsupported_graph_schema",
            "schema",
            format!("expected {ENSEMBLE_CUE_GRAPH_SCHEMA_V1}"),
        ));
    }
    if graph.states.is_empty() {
        issues.push(issue("empty_states", "states", "at least one state is required"));
    }
    if graph.states.len() > MAX_CUE_STATES {
        issues.push(issue(
            "too_many_states",
            "states",
            format!("state count exceeds limit {MAX_CUE_STATES}"),
        ));
    }
    if graph.layer_ids.is_empty() {
        issues.push(issue("empty_layers", "layer_ids", "at least one layer is required"));
    }
    if graph.layer_ids.len() > MAX_CUE_LAYERS {
        issues.push(issue(
            "too_many_layers",
            "layer_ids",
            format!("layer count exceeds limit {MAX_CUE_LAYERS}"),
        ));
    }
    if graph.transitions.is_empty() {
        issues.push(issue(
            "empty_transitions",
            "transitions",
            "at least one cue transition is required",
        ));
    }
    if graph.transitions.len() > MAX_CUE_TRANSITIONS {
        issues.push(issue(
            "too_many_transitions",
            "transitions",
            format!("transition count exceeds limit {MAX_CUE_TRANSITIONS}"),
        ));
    }

    let mut states = HashSet::new();
    for (index, state) in graph.states.iter().enumerate() {
        if state.id.trim().is_empty() {
            issues.push(issue(
                "empty_state_id",
                format!("states[{index}].id"),
                "state ID must not be empty",
            ));
        } else if !states.insert(state.id.as_str()) {
            issues.push(issue(
                "duplicate_state_id",
                format!("states[{index}].id"),
                format!("duplicate state {}", state.id),
            ));
        }
    }
    if !states.contains(graph.initial_state.as_str()) {
        issues.push(issue(
            "unknown_initial_state",
            "initial_state",
            format!("unknown initial state {}", graph.initial_state),
        ));
    }

    let mut layers = HashSet::new();
    for (index, layer_id) in graph.layer_ids.iter().enumerate() {
        if layer_id.trim().is_empty() {
            issues.push(issue(
                "empty_layer_id",
                format!("layer_ids[{index}]"),
                "layer ID must not be empty",
            ));
        } else if !layers.insert(layer_id.as_str()) {
            issues.push(issue(
                "duplicate_layer_id",
                format!("layer_ids[{index}]"),
                format!("duplicate layer {layer_id}"),
            ));
        }
    }

    let mut keys = HashSet::new();
    for (index, rule) in graph.transitions.iter().enumerate() {
        let path = format!("transitions[{index}]");
        if !states.contains(rule.from_state.as_str()) {
            issues.push(issue(
                "unknown_from_state",
                format!("{path}.from_state"),
                format!("unknown state {}", rule.from_state),
            ));
        }
        if !states.contains(rule.to_state.as_str()) {
            issues.push(issue(
                "unknown_to_state",
                format!("{path}.to_state"),
                format!("unknown state {}", rule.to_state),
            ));
        }
        if rule.from_state == rule.to_state {
            issues.push(issue(
                "self_state_transition",
                &path,
                "cue transition must change state",
            ));
        }
        if rule.cue_type.trim().is_empty() {
            issues.push(issue(
                "empty_cue_type",
                format!("{path}.cue_type"),
                "cue type must not be empty",
            ));
        }
        if !layers.contains(rule.source_layer_id.as_str()) {
            issues.push(issue(
                "unknown_source_layer",
                format!("{path}.source_layer_id"),
                format!("unknown source layer {}", rule.source_layer_id),
            ));
        }
        if !layers.contains(rule.response_layer_id.as_str()) {
            issues.push(issue(
                "unknown_response_layer",
                format!("{path}.response_layer_id"),
                format!("unknown response layer {}", rule.response_layer_id),
            ));
        }
        if rule.response_action.trim().is_empty() {
            issues.push(issue(
                "empty_response_action",
                format!("{path}.response_action"),
                "response action must not be empty",
            ));
        }
        if rule.minimum_response_delay_ticks == 0
            || rule.maximum_response_delay_ticks < rule.minimum_response_delay_ticks
        {
            issues.push(issue(
                "invalid_response_window",
                format!("{path}.minimum_response_delay_ticks"),
                "response window must be positive and ordered",
            ));
        }
        if rule.maximum_state_change_latency_ticks == 0 {
            issues.push(issue(
                "invalid_state_change_latency",
                format!("{path}.maximum_state_change_latency_ticks"),
                "state-change latency must be positive",
            ));
        }

        let key = (
            rule.from_state.as_str(),
            rule.cue_type.as_str(),
            rule.source_layer_id.as_str(),
        );
        if !keys.insert(key) {
            issues.push(issue(
                "duplicate_transition_key",
                &path,
                "only one rule is allowed for a state/cue/source-layer combination",
            ));
        }
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

/// Verify cue-triggered state changes and layer responses.
///
/// A cue is eligible only when its declared source emits it in the current
/// state. The realized state change must reach the rule's target within its
/// latency bound, and the designated response layer must perform the required
/// action within its response window. Cues are processed in absolute-time
/// order; same-tick cues are ordered by stable cue ID.
pub fn verify_ensemble_cues(
    graph: &EnsembleCueGraphV1,
    sequence: &EnsembleCueSequenceV1,
) -> Result<EnsembleCueVerificationV1, Vec<CueValidationIssue>> {
    validate_cue_graph(graph)?;
    validate_sequence(graph, sequence)?;

    let rules: HashMap<(&str, &str, &str), &CueTransitionRuleV1> = graph
        .transitions
        .iter()
        .map(|rule| {
            (
                (
                    rule.from_state.as_str(),
                    rule.cue_type.as_str(),
                    rule.source_layer_id.as_str(),
                ),
                rule,
            )
        })
        .collect();

    let mut cues = sequence.cues.iter().collect::<Vec<_>>();
    cues.sort_by(|left, right| {
        left.onset_tick
            .cmp(&right.onset_tick)
            .then_with(|| left.id.cmp(&right.id))
    });

    let mut state = graph.initial_state.clone();
    let mut results = Vec::with_capacity(cues.len());

    for cue in cues {
        let key = (state.as_str(), cue.cue_type.as_str(), cue.source_layer_id.as_str());
        let Some(rule) = rules.get(&key).copied() else {
            results.push(CueTransitionResultV1 {
                cue_id: cue.id.clone(),
                from_state: Some(state.clone()),
                expected_to_state: None,
                observed_to_state: sequence
                    .state_changes
                    .iter()
                    .find(|event| event.cue_id == cue.id)
                    .map(|event| event.to_state.clone()),
                state_change_latency_ticks: None,
                response_delay_ticks: None,
                response_found: false,
                passed: false,
            });
            continue;
        };

        let state_change = sequence
            .state_changes
            .iter()
            .find(|event| event.cue_id == cue.id);
        let observed_to_state = state_change.map(|event| event.to_state.clone());
        let state_change_latency =
            state_change.and_then(|event| event.onset_tick.checked_sub(cue.onset_tick));
        let state_change_ok = state_change.is_some_and(|event| {
            event.to_state == rule.to_state
                && event.onset_tick >= cue.onset_tick
                && event.onset_tick - cue.onset_tick
                    <= rule.maximum_state_change_latency_ticks as u64
        });

        let response = sequence
            .responses
            .iter()
            .filter(|event| {
                event.cue_id == cue.id
                    && event.layer_id == rule.response_layer_id
                    && event.action == rule.response_action
                    && event.onset_tick >= cue.onset_tick
            })
            .filter(|event| {
                let delay = event.onset_tick - cue.onset_tick;
                delay >= rule.minimum_response_delay_ticks as u64
                    && delay <= rule.maximum_response_delay_ticks as u64
            })
            .min_by_key(|event| event.onset_tick);
        let response_delay = response.map(|event| event.onset_tick - cue.onset_tick);
        let response_found = response.is_some();
        let passed = state_change_ok && response_found;

        results.push(CueTransitionResultV1 {
            cue_id: cue.id.clone(),
            from_state: Some(state.clone()),
            expected_to_state: Some(rule.to_state.clone()),
            observed_to_state,
            state_change_latency_ticks: state_change_latency,
            response_delay_ticks: response_delay,
            response_found,
            passed,
        });

        if state_change_ok {
            state = rule.to_state.clone();
        }
    }

    let all_transitions_passed = results.iter().all(|result| result.passed);
    Ok(EnsembleCueVerificationV1 {
        schema: "ensemble-cue-verification-v1".into(),
        initial_state: graph.initial_state.clone(),
        final_state: state,
        results,
        all_transitions_passed,
    })
}

fn validate_sequence(
    graph: &EnsembleCueGraphV1,
    sequence: &EnsembleCueSequenceV1,
) -> Result<(), Vec<CueValidationIssue>> {
    let mut issues = Vec::new();
    if sequence.schema != ENSEMBLE_CUE_SEQUENCE_SCHEMA_V1 {
        issues.push(issue(
            "unsupported_sequence_schema",
            "schema",
            format!("expected {ENSEMBLE_CUE_SEQUENCE_SCHEMA_V1}"),
        ));
    }
    if sequence.horizon_ticks == 0 {
        issues.push(issue(
            "zero_horizon",
            "horizon_ticks",
            "cue sequence horizon must be positive",
        ));
    }
    let event_count = sequence
        .cues
        .len()
        .saturating_add(sequence.responses.len())
        .saturating_add(sequence.state_changes.len());
    if event_count > MAX_CUE_SEQUENCE_EVENTS {
        issues.push(issue(
            "too_many_sequence_events",
            "cues",
            format!("event count exceeds limit {MAX_CUE_SEQUENCE_EVENTS}"),
        ));
    }

    let layers = graph.layer_ids.iter().map(String::as_str).collect::<HashSet<_>>();
    let states = graph.states.iter().map(|state| state.id.as_str()).collect::<HashSet<_>>();
    let mut cue_ids = HashSet::new();
    for (index, cue) in sequence.cues.iter().enumerate() {
        let path = format!("cues[{index}]");
        if cue.id.trim().is_empty() || !cue_ids.insert(cue.id.as_str()) {
            issues.push(issue(
                "empty_or_duplicate_cue_id",
                format!("{path}.id"),
                "cue ID must be non-empty and unique",
            ));
        }
        if cue.cue_type.trim().is_empty() {
            issues.push(issue(
                "empty_cue_type",
                format!("{path}.cue_type"),
                "cue type must not be empty",
            ));
        }
        if !layers.contains(cue.source_layer_id.as_str()) {
            issues.push(issue(
                "unknown_cue_source",
                format!("{path}.source_layer_id"),
                format!("unknown layer {}", cue.source_layer_id),
            ));
        }
        if cue.onset_tick >= sequence.horizon_ticks {
            issues.push(issue(
                "cue_outside_horizon",
                format!("{path}.onset_tick"),
                "cue must occur before the sequence horizon",
            ));
        }
    }

    let mut state_change_cues = HashSet::new();
    for (index, event) in sequence.state_changes.iter().enumerate() {
        let path = format!("state_changes[{index}]");
        if !cue_ids.contains(event.cue_id.as_str()) {
            issues.push(issue(
                "unknown_state_change_cue",
                format!("{path}.cue_id"),
                format!("unknown cue {}", event.cue_id),
            ));
        }
        if !state_change_cues.insert(event.cue_id.as_str()) {
            issues.push(issue(
                "duplicate_state_change",
                format!("{path}.cue_id"),
                "at most one realized state change is allowed per cue",
            ));
        }
        if !states.contains(event.to_state.as_str()) {
            issues.push(issue(
                "unknown_observed_state",
                format!("{path}.to_state"),
                format!("unknown state {}", event.to_state),
            ));
        }
        if event.onset_tick >= sequence.horizon_ticks {
            issues.push(issue(
                "state_change_outside_horizon",
                format!("{path}.onset_tick"),
                "state change must occur before the sequence horizon",
            ));
        }
    }

    for (index, event) in sequence.responses.iter().enumerate() {
        let path = format!("responses[{index}]");
        if !cue_ids.contains(event.cue_id.as_str()) {
            issues.push(issue(
                "unknown_response_cue",
                format!("{path}.cue_id"),
                format!("unknown cue {}", event.cue_id),
            ));
        }
        if !layers.contains(event.layer_id.as_str()) {
            issues.push(issue(
                "unknown_response_layer",
                format!("{path}.layer_id"),
                format!("unknown layer {}", event.layer_id),
            ));
        }
        if event.action.trim().is_empty() {
            issues.push(issue(
                "empty_response_action",
                format!("{path}.action"),
                "response action must not be empty",
            ));
        }
        if event.onset_tick >= sequence.horizon_ticks {
            issues.push(issue(
                "response_outside_horizon",
                format!("{path}.onset_tick"),
                "response must occur before the sequence horizon",
            ));
        }
    }

    if issues.is_empty() {
        Ok(())
    } else {
        Err(issues)
    }
}

fn issue(
    code: impl Into<String>,
    path: impl Into<String>,
    message: impl Into<String>,
) -> CueValidationIssue {
    CueValidationIssue {
        code: code.into(),
        path: path.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn graph() -> EnsembleCueGraphV1 {
        EnsembleCueGraphV1 {
            schema: ENSEMBLE_CUE_GRAPH_SCHEMA_V1.into(),
            initial_state: "full".into(),
            states: ["full", "soft", "solo"]
                .into_iter()
                .map(|id| CueStateV1 { id: id.into() })
                .collect(),
            layer_ids: ["captain_guitar", "talking_drum", "rhythm_section"]
                .into_iter()
                .map(str::to_owned)
                .collect(),
            transitions: vec![
                CueTransitionRuleV1 {
                    from_state: "full".into(),
                    cue_type: "soften".into(),
                    source_layer_id: "captain_guitar".into(),
                    to_state: "soft".into(),
                    response_layer_id: "talking_drum".into(),
                    response_action: "lower_intensity".into(),
                    minimum_response_delay_ticks: 2,
                    maximum_response_delay_ticks: 12,
                    maximum_state_change_latency_ticks: 4,
                },
                CueTransitionRuleV1 {
                    from_state: "soft".into(),
                    cue_type: "solo_entry".into(),
                    source_layer_id: "captain_guitar".into(),
                    to_state: "solo".into(),
                    response_layer_id: "rhythm_section".into(),
                    response_action: "thin_accompaniment".into(),
                    minimum_response_delay_ticks: 2,
                    maximum_response_delay_ticks: 16,
                    maximum_state_change_latency_ticks: 6,
                },
            ],
        }
    }

    fn sequence() -> EnsembleCueSequenceV1 {
        EnsembleCueSequenceV1 {
            schema: ENSEMBLE_CUE_SEQUENCE_SCHEMA_V1.into(),
            horizon_ticks: 128,
            cues: vec![
                CueEventV1 {
                    id: "cue_1".into(),
                    cue_type: "soften".into(),
                    source_layer_id: "captain_guitar".into(),
                    onset_tick: 16,
                },
                CueEventV1 {
                    id: "cue_2".into(),
                    cue_type: "solo_entry".into(),
                    source_layer_id: "captain_guitar".into(),
                    onset_tick: 64,
                },
            ],
            responses: vec![
                CueResponseEventV1 {
                    cue_id: "cue_1".into(),
                    layer_id: "talking_drum".into(),
                    action: "lower_intensity".into(),
                    onset_tick: 20,
                },
                CueResponseEventV1 {
                    cue_id: "cue_2".into(),
                    layer_id: "rhythm_section".into(),
                    action: "thin_accompaniment".into(),
                    onset_tick: 70,
                },
            ],
            state_changes: vec![
                CueStateChangeEventV1 {
                    cue_id: "cue_1".into(),
                    to_state: "soft".into(),
                    onset_tick: 18,
                },
                CueStateChangeEventV1 {
                    cue_id: "cue_2".into(),
                    to_state: "solo".into(),
                    onset_tick: 68,
                },
            ],
        }
    }

    #[test]
    fn verifies_ordered_cue_state_response_chain() {
        let report = verify_ensemble_cues(&graph(), &sequence()).unwrap();
        assert!(report.all_transitions_passed);
        assert_eq!(report.final_state, "solo");
        assert_eq!(report.results.len(), 2);
        assert_eq!(report.results[0].state_change_latency_ticks, Some(2));
        assert_eq!(report.results[0].response_delay_ticks, Some(4));
        assert_eq!(report.results[1].response_delay_ticks, Some(6));
    }

    #[test]
    fn missing_response_fails_without_forgiving_state_change() {
        let mut input = sequence();
        input.responses.remove(0);
        let report = verify_ensemble_cues(&graph(), &input).unwrap();
        assert!(!report.all_transitions_passed);
        assert!(!report.results[0].response_found);
        assert_eq!(report.final_state, "solo");
    }

    #[test]
    fn incorrect_state_change_does_not_advance_the_state_machine() {
        let mut input = sequence();
        input.state_changes[0].to_state = "solo".into();
        let report = verify_ensemble_cues(&graph(), &input).unwrap();
        assert!(!report.results[0].passed);
        assert_eq!(report.results[1].from_state.as_deref(), Some("full"));
        assert_eq!(report.final_state, "full");
    }

    #[test]
    fn response_outside_the_declared_window_fails() {
        let mut input = sequence();
        input.responses[0].onset_tick = 40;
        let report = verify_ensemble_cues(&graph(), &input).unwrap();
        assert!(!report.results[0].passed);
        assert!(!report.results[0].response_found);
    }

    #[test]
    fn state_change_before_cue_is_a_failed_transition_not_time_underflow() {
        let mut input = sequence();
        input.state_changes[0].onset_tick = 12;
        let report = verify_ensemble_cues(&graph(), &input).unwrap();
        assert!(!report.results[0].passed);
        assert_eq!(report.results[0].state_change_latency_ticks, None);
        assert_eq!(report.final_state, "full");
    }

    #[test]
    fn cue_array_order_does_not_change_verification_order() {
        let expected = verify_ensemble_cues(&graph(), &sequence()).unwrap();
        let mut reversed = sequence();
        reversed.cues.reverse();
        let actual = verify_ensemble_cues(&graph(), &reversed).unwrap();
        assert_eq!(expected.results, actual.results);
        assert_eq!(expected.final_state, actual.final_state);
    }

    #[test]
    fn duplicate_transition_keys_are_rejected() {
        let mut value = graph();
        value.transitions.push(value.transitions[0].clone());
        let issues = validate_cue_graph(&value).unwrap_err();
        assert!(issues.iter().any(|entry| entry.code == "duplicate_transition_key"));
    }

    #[test]
    fn cues_without_a_current_state_rule_fail_closed() {
        let mut input = sequence();
        input.cues[0].cue_type = "unregistered_signal".into();
        let report = verify_ensemble_cues(&graph(), &input).unwrap();
        assert!(!report.results[0].passed);
        assert!(report.results[0].expected_to_state.is_none());
    }
}
