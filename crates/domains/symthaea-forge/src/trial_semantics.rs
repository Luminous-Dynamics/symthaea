// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strong semantic validation for Forge attempt-scoped observations.
//!
//! The ordinary trace validator proves lifecycle, attempt pairing, and observation coverage. This
//! module adds the proposition needed before search history can be used for learning: event kind,
//! observation schema, attempt identity, candidate identity, transformation identity, and the
//! observed rejection/selection shape must all agree.

use crate::trace::{validate_forge_trace_observations, ForgeTraceError, ForgeTraceEvent};
use serde_json::Value;
use std::collections::BTreeMap;
use symthaea_algorithms::ledger::DiscoveryEventKind;
use symthaea_algorithms::observation::{ObservationEncoding, ObservationObject, ObservationStore};
use thiserror::Error;

pub const CANDIDATE_GENERATED_SCHEMA: &str = "symthaea.forge.candidate-generated.v2";
pub const NO_CANDIDATE_SCHEMA: &str = "symthaea.forge.no-candidate.v2";
pub const CORRECTNESS_GATES_SCHEMA: &str = "symthaea.forge.correctness-gates.v2";
pub const BENCHMARK_FAILURE_SCHEMA: &str = "symthaea.forge.benchmark-failure.v2";
pub const SELECTION_SCHEMA: &str = "symthaea.forge.selection.v2";
pub const CANDIDATE_DECISION_SCHEMA: &str = "symthaea.forge.candidate-decision.v2";
pub const SEARCH_SUMMARY_SCHEMA: &str = "symthaea.forge.search-summary.v1";
pub const SEARCH_ABORTED_SCHEMA: &str = "symthaea.forge.search-aborted.v1";

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ForgeTrialSemanticError {
    #[error("Forge trace/observation relationship is invalid: {0}")]
    Trace(String),
    #[error("Forge observation must use canonical JSON encoding")]
    NonJsonObservation,
    #[error("Forge observation payload is not valid JSON")]
    InvalidJson,
    #[error("Forge event kind does not match observation schema")]
    SchemaMismatch,
    #[error("Forge observation attempt identity does not match trace event")]
    AttemptMismatch,
    #[error("Forge observation generation does not match trace event")]
    GenerationMismatch,
    #[error("Forge observation candidate artifact does not match trace event")]
    CandidateArtifactMismatch,
    #[error("Forge observation transformation does not match candidate generation")]
    TransformationMismatch,
    #[error("Forge observation does not describe the rejection/selection event it is attached to")]
    OutcomeShapeMismatch,
    #[error("Forge candidate terminal event has no generated transformation")]
    MissingGeneratedTransformation,
    #[error("CandidateArchived is not a Forge-local trial event")]
    UnsupportedArchivedEvent,
}

impl From<ForgeTraceError> for ForgeTrialSemanticError {
    fn from(error: ForgeTraceError) -> Self {
        Self::Trace(error.to_string())
    }
}

/// Validate a complete Forge trace strongly enough for per-transformation trial extraction.
///
/// This intentionally does not promote Forge-local gate or benchmark results into independent
/// correctness/performance evidence. It proves only that Forge's own event and observation records
/// are internally consistent and attributable to the exact attempt that produced them.
pub fn validate_forge_trial_semantics(
    trace: &[ForgeTraceEvent],
    observations: &ObservationStore,
) -> Result<(), ForgeTrialSemanticError> {
    validate_forge_trace_observations(trace, observations)?;

    let mut transformations = BTreeMap::<String, String>::new();

    for event in trace {
        let object = observations
            .get(&event.observation_id)
            .ok_or_else(|| ForgeTrialSemanticError::Trace("missing observation after coverage validation".into()))?;
        let payload = json_payload(object)?;

        match event.kind {
            DiscoveryEventKind::CandidateGenerated => {
                require_schema(object, &[CANDIDATE_GENERATED_SCHEMA])?;
                require_attempt(event, &payload)?;
                require_generation(event, &payload)?;
                require_candidate_artifact(event, &payload)?;
                let transformation = required_string(&payload, "transformation_id")?;
                let attempt = event
                    .attempt_id
                    .as_ref()
                    .ok_or(ForgeTrialSemanticError::AttemptMismatch)?;
                transformations.insert(
                    attempt.as_content_id().as_str().to_string(),
                    transformation.to_string(),
                );
            }
            DiscoveryEventKind::GeneratorNoOp => {
                require_schema(object, &[NO_CANDIDATE_SCHEMA])?;
                require_attempt(event, &payload)?;
                let reason = required_string(&payload, "reason")?;
                if reason.is_empty() {
                    return Err(ForgeTrialSemanticError::OutcomeShapeMismatch);
                }
            }
            DiscoveryEventKind::RejectedCompilation => {
                require_schema(object, &[CORRECTNESS_GATES_SCHEMA])?;
                require_attempt(event, &payload)?;
                require_transformation(event, &payload, &transformations)?;
                validate_compile_rejection(&payload)?;
            }
            DiscoveryEventKind::RejectedCorrectness => {
                require_schema(object, &[CORRECTNESS_GATES_SCHEMA])?;
                require_attempt(event, &payload)?;
                require_transformation(event, &payload, &transformations)?;
                validate_correctness_rejection(&payload)?;
            }
            DiscoveryEventKind::RejectedEvaluation => {
                require_schema(object, &[BENCHMARK_FAILURE_SCHEMA])?;
                require_attempt(event, &payload)?;
                require_transformation(event, &payload, &transformations)?;
                if required_string(&payload, "error")?.is_empty() {
                    return Err(ForgeTrialSemanticError::OutcomeShapeMismatch);
                }
            }
            DiscoveryEventKind::ValidNotSelected => {
                require_schema(object, &[SELECTION_SCHEMA, CANDIDATE_DECISION_SCHEMA])?;
                require_attempt(event, &payload)?;
                require_transformation(event, &payload, &transformations)?;
                match object.schema() {
                    SELECTION_SCHEMA => {
                        if required_string(&payload, "decision")? != "not-better-than-parent" {
                            return Err(ForgeTrialSemanticError::OutcomeShapeMismatch);
                        }
                    }
                    CANDIDATE_DECISION_SCHEMA => {
                        require_candidate_artifact(event, &payload)?;
                        match required_string(&payload, "decision")? {
                            "superseded-within-generation"
                            | "generation-winner-remained-better" => {}
                            _ => return Err(ForgeTrialSemanticError::OutcomeShapeMismatch),
                        }
                    }
                    _ => return Err(ForgeTrialSemanticError::SchemaMismatch),
                }
            }
            DiscoveryEventKind::SelectedForContinuation => {
                require_schema(object, &[CANDIDATE_DECISION_SCHEMA])?;
                require_attempt(event, &payload)?;
                require_candidate_artifact(event, &payload)?;
                require_transformation(event, &payload, &transformations)?;
                if required_string(&payload, "decision")? != "selected-for-next-generation" {
                    return Err(ForgeTrialSemanticError::OutcomeShapeMismatch);
                }
            }
            DiscoveryEventKind::SearchCompleted => {
                require_schema(object, &[SEARCH_SUMMARY_SCHEMA])?;
            }
            DiscoveryEventKind::SearchAborted => {
                require_schema(object, &[SEARCH_ABORTED_SCHEMA])?;
            }
            DiscoveryEventKind::CandidateArchived => {
                return Err(ForgeTrialSemanticError::UnsupportedArchivedEvent);
            }
        }
    }

    Ok(())
}

fn json_payload(object: &ObservationObject) -> Result<Value, ForgeTrialSemanticError> {
    if object.encoding() != ObservationEncoding::Json {
        return Err(ForgeTrialSemanticError::NonJsonObservation);
    }
    serde_json::from_slice(object.payload()).map_err(|_| ForgeTrialSemanticError::InvalidJson)
}

fn require_schema(
    object: &ObservationObject,
    allowed: &[&str],
) -> Result<(), ForgeTrialSemanticError> {
    if allowed.iter().any(|schema| *schema == object.schema()) {
        Ok(())
    } else {
        Err(ForgeTrialSemanticError::SchemaMismatch)
    }
}

fn require_attempt(
    event: &ForgeTraceEvent,
    payload: &Value,
) -> Result<(), ForgeTrialSemanticError> {
    let expected = event
        .attempt_id
        .as_ref()
        .ok_or(ForgeTrialSemanticError::AttemptMismatch)?;
    if payload.get("attempt_id").and_then(Value::as_str)
        == Some(expected.as_content_id().as_str())
    {
        Ok(())
    } else {
        Err(ForgeTrialSemanticError::AttemptMismatch)
    }
}

fn require_generation(
    event: &ForgeTraceEvent,
    payload: &Value,
) -> Result<(), ForgeTrialSemanticError> {
    if payload.get("generation").and_then(Value::as_u64) == event.generation {
        Ok(())
    } else {
        Err(ForgeTrialSemanticError::GenerationMismatch)
    }
}

fn require_candidate_artifact(
    event: &ForgeTraceEvent,
    payload: &Value,
) -> Result<(), ForgeTrialSemanticError> {
    let expected = event
        .candidate_artifact_id
        .as_ref()
        .ok_or(ForgeTrialSemanticError::CandidateArtifactMismatch)?;
    if payload
        .get("candidate_artifact_id")
        .and_then(Value::as_str)
        == Some(expected.as_str())
    {
        Ok(())
    } else {
        Err(ForgeTrialSemanticError::CandidateArtifactMismatch)
    }
}

fn require_transformation<'a>(
    event: &ForgeTraceEvent,
    payload: &'a Value,
    transformations: &'a BTreeMap<String, String>,
) -> Result<(), ForgeTrialSemanticError> {
    let attempt = event
        .attempt_id
        .as_ref()
        .ok_or(ForgeTrialSemanticError::AttemptMismatch)?;
    let expected = transformations
        .get(attempt.as_content_id().as_str())
        .ok_or(ForgeTrialSemanticError::MissingGeneratedTransformation)?;
    if required_string(payload, "transformation_id")? == expected {
        Ok(())
    } else {
        Err(ForgeTrialSemanticError::TransformationMismatch)
    }
}

fn required_string<'a>(
    payload: &'a Value,
    field: &str,
) -> Result<&'a str, ForgeTrialSemanticError> {
    payload
        .get(field)
        .and_then(Value::as_str)
        .ok_or(ForgeTrialSemanticError::OutcomeShapeMismatch)
}

fn validate_compile_rejection(payload: &Value) -> Result<(), ForgeTrialSemanticError> {
    let gates = payload
        .get("gates")
        .and_then(Value::as_array)
        .ok_or(ForgeTrialSemanticError::OutcomeShapeMismatch)?;
    if gates.len() != 1 || !gate_matches(&gates[0], "compile", false) {
        return Err(ForgeTrialSemanticError::OutcomeShapeMismatch);
    }
    Ok(())
}

fn validate_correctness_rejection(payload: &Value) -> Result<(), ForgeTrialSemanticError> {
    let gates = payload
        .get("gates")
        .and_then(Value::as_array)
        .ok_or(ForgeTrialSemanticError::OutcomeShapeMismatch)?;
    if gates.len() != 2
        || !gate_matches(&gates[0], "compile", true)
        || !gate_matches(&gates[1], "test", false)
    {
        return Err(ForgeTrialSemanticError::OutcomeShapeMismatch);
    }
    Ok(())
}

fn gate_matches(value: &Value, name: &str, passed: bool) -> bool {
    value.get("gate").and_then(Value::as_str) == Some(name)
        && value.get("passed").and_then(Value::as_bool) == Some(passed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::{full_source_artifact_id, MutationRecord};
    use crate::fitness::{Gate, GateResult};
    use crate::observations as forge_observations;
    use crate::trace::ForgeAttemptId;
    use std::time::Duration;
    use symthaea_algorithms::observation::ObservationStore;

    fn fixture() -> (ForgeAttemptId, MutationRecord) {
        let baseline = full_source_artifact_id("fn f() -> i32 { 1 }\n");
        let candidate = full_source_artifact_id("fn f() -> i32 { 2 }\n");
        (
            ForgeAttemptId::derive(&baseline, 7, 0, 0),
            MutationRecord::new(0, "literal", "1 -> 2", baseline, candidate),
        )
    }

    fn correctness_failure() -> Vec<GateResult> {
        vec![
            GateResult {
                gate: Gate::Compile,
                passed: true,
                output_tail: String::new(),
                duration: Duration::from_nanos(10),
            },
            GateResult {
                gate: Gate::Test,
                passed: false,
                output_tail: "counterexample".into(),
                duration: Duration::from_nanos(20),
            },
        ]
    }

    #[test]
    fn canonical_generated_then_correctness_rejected_trial_validates() {
        let (attempt, mutation) = fixture();
        let generated = forge_observations::candidate_generated(&attempt, &mutation).unwrap();
        let rejected =
            forge_observations::gates(&attempt, &mutation, &correctness_failure()).unwrap();
        let summary = forge_observations::search_summary(1, 0, 0, 1, 0, 0, 0, None, None).unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt.clone(),
                0,
                DiscoveryEventKind::CandidateGenerated,
                mutation.candidate_artifact_id.clone(),
                generated.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt,
                0,
                DiscoveryEventKind::RejectedCorrectness,
                mutation.candidate_artifact_id.clone(),
                rejected.id().clone(),
            ),
            ForgeTraceEvent::completed(summary.id().clone()),
        ];
        let store = ObservationStore::from_objects(vec![generated, rejected, summary]).unwrap();
        assert!(validate_forge_trial_semantics(&trace, &store).is_ok());
    }

    #[test]
    fn wrong_schema_cannot_be_attached_to_rejection_event() {
        let (attempt, mutation) = fixture();
        let generated = forge_observations::candidate_generated(&attempt, &mutation).unwrap();
        let wrong = forge_observations::selection(
            &attempt,
            &mutation,
            None,
            None,
            "not-better-than-parent",
        )
        .unwrap();
        let abort = forge_observations::search_abort(
            "test",
            "synthetic abort",
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
        )
        .unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt.clone(),
                0,
                DiscoveryEventKind::CandidateGenerated,
                mutation.candidate_artifact_id.clone(),
                generated.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt,
                0,
                DiscoveryEventKind::RejectedCorrectness,
                mutation.candidate_artifact_id.clone(),
                wrong.id().clone(),
            ),
            ForgeTraceEvent::aborted(abort.id().clone()),
        ];
        let store = ObservationStore::from_objects(vec![generated, wrong, abort]).unwrap();
        assert_eq!(
            validate_forge_trial_semantics(&trace, &store).unwrap_err(),
            ForgeTrialSemanticError::SchemaMismatch
        );
    }

    #[test]
    fn terminal_transformation_must_match_generation_observation() {
        let (attempt, mutation) = fixture();
        let alternative = MutationRecord::new(
            mutation.generation,
            "different-operator",
            "same bytes different claimed transform",
            mutation.parent_artifact_id.clone(),
            mutation.candidate_artifact_id.clone(),
        );
        let generated = forge_observations::candidate_generated(&attempt, &mutation).unwrap();
        let rejected =
            forge_observations::gates(&attempt, &alternative, &correctness_failure()).unwrap();
        let summary = forge_observations::search_summary(1, 0, 0, 1, 0, 0, 0, None, None).unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt.clone(),
                0,
                DiscoveryEventKind::CandidateGenerated,
                mutation.candidate_artifact_id.clone(),
                generated.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt,
                0,
                DiscoveryEventKind::RejectedCorrectness,
                mutation.candidate_artifact_id.clone(),
                rejected.id().clone(),
            ),
            ForgeTraceEvent::completed(summary.id().clone()),
        ];
        let store = ObservationStore::from_objects(vec![generated, rejected, summary]).unwrap();
        assert_eq!(
            validate_forge_trial_semantics(&trace, &store).unwrap_err(),
            ForgeTrialSemanticError::TransformationMismatch
        );
    }
}
