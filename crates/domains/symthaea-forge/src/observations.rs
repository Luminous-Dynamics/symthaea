// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical observation payloads emitted by Forge search.
//!
//! These objects make `observation_id` reconstructable without turning a Forge-local observation
//! into correctness, performance, replication, promotion, or runtime authority.

use crate::certificate::{CertificateError, ForgeCandidate, MutationRecord};
use crate::fitness::{BenchmarkResult, GateResult};
use serde::Serialize;
use symthaea_algorithms::observation::{ObservationEncoding, ObservationError, ObservationObject};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeObservationError {
    #[error(transparent)]
    Certificate(#[from] CertificateError),
    #[error(transparent)]
    Object(#[from] ObservationError),
    #[error("Forge observation JSON serialization failed: {0}")]
    Json(String),
}

fn json_object(
    schema: &'static str,
    value: &impl Serialize,
) -> Result<ObservationObject, ForgeObservationError> {
    let payload = serde_json::to_vec(value)
        .map_err(|error| ForgeObservationError::Json(error.to_string()))?;
    Ok(ObservationObject::new(
        schema,
        ObservationEncoding::Json,
        payload,
    )?)
}

#[derive(Serialize)]
struct CandidateGeneratedPayload<'a> {
    generation: usize,
    transformation_id: &'a str,
    operator: &'a str,
    detail: &'a str,
    parent_artifact_id: &'a str,
    candidate_artifact_id: &'a str,
}

pub fn candidate_generated(
    mutation: &MutationRecord,
) -> Result<ObservationObject, ForgeObservationError> {
    json_object(
        "symthaea.forge.candidate-generated.v1",
        &CandidateGeneratedPayload {
            generation: mutation.generation,
            transformation_id: mutation.transformation_id.as_content_id().as_str(),
            operator: &mutation.operator,
            detail: &mutation.detail,
            parent_artifact_id: mutation.parent_artifact_id.as_str(),
            candidate_artifact_id: mutation.candidate_artifact_id.as_str(),
        },
    )
}

#[derive(Serialize)]
struct NoCandidatePayload<'a> {
    reason: &'a str,
    parent_artifact_id: &'a str,
}

pub fn no_candidate(
    reason: &str,
    parent_artifact_id: &symthaea_algorithms::ContentId,
) -> Result<ObservationObject, ForgeObservationError> {
    json_object(
        "symthaea.forge.no-candidate.v1",
        &NoCandidatePayload {
            reason,
            parent_artifact_id: parent_artifact_id.as_str(),
        },
    )
}

#[derive(Serialize)]
struct GatePayload<'a> {
    gate: &'a str,
    passed: bool,
    duration_ns: u128,
    output_tail: &'a str,
}

#[derive(Serialize)]
struct GateObservationPayload<'a> {
    transformation_id: &'a str,
    gates: Vec<GatePayload<'a>>,
}

pub fn gates(
    mutation: &MutationRecord,
    gate_results: &[GateResult],
) -> Result<ObservationObject, ForgeObservationError> {
    let gates = gate_results
        .iter()
        .map(|gate| GatePayload {
            gate: gate.gate.label(),
            passed: gate.passed,
            duration_ns: gate.duration.as_nanos(),
            output_tail: &gate.output_tail,
        })
        .collect();
    json_object(
        "symthaea.forge.correctness-gates.v1",
        &GateObservationPayload {
            transformation_id: mutation.transformation_id.as_content_id().as_str(),
            gates,
        },
    )
}

#[derive(Serialize)]
struct BenchmarkFailurePayload<'a> {
    transformation_id: &'a str,
    error: &'a str,
}

pub fn benchmark_failure(
    mutation: &MutationRecord,
    error: &str,
) -> Result<ObservationObject, ForgeObservationError> {
    json_object(
        "symthaea.forge.benchmark-failure.v1",
        &BenchmarkFailurePayload {
            transformation_id: mutation.transformation_id.as_content_id().as_str(),
            error,
        },
    )
}

#[derive(Serialize)]
struct SelectionPayload<'a> {
    transformation_id: &'a str,
    decision: &'a str,
    metric_name: Option<&'a str>,
    parent_score_bits: Option<u64>,
    candidate_score_bits: Option<u64>,
}

pub fn selection(
    mutation: &MutationRecord,
    benchmark: Option<&BenchmarkResult>,
    parent_score: Option<f64>,
    decision: &str,
) -> Result<ObservationObject, ForgeObservationError> {
    json_object(
        "symthaea.forge.selection.v1",
        &SelectionPayload {
            transformation_id: mutation.transformation_id.as_content_id().as_str(),
            decision,
            metric_name: benchmark.map(|result| result.metric_name.as_str()),
            parent_score_bits: parent_score.map(f64::to_bits),
            candidate_score_bits: benchmark.map(|result| result.score.to_bits()),
        },
    )
}

#[derive(Serialize)]
struct CandidateDecisionPayload<'a> {
    candidate_artifact_id: &'a str,
    transformation_id: &'a str,
    decision: &'a str,
    metric_name: Option<&'a str>,
    candidate_score_bits: Option<u64>,
}

pub fn candidate_decision(
    candidate: &ForgeCandidate,
    decision: &str,
) -> Result<ObservationObject, ForgeObservationError> {
    candidate.validate()?;
    let certificate = candidate.certificate();
    let final_mutation = certificate
        .mutation_history
        .last()
        .expect("validated Forge candidate contains a mutation");
    json_object(
        "symthaea.forge.candidate-decision.v1",
        &CandidateDecisionPayload {
            candidate_artifact_id: candidate.artifact_id().as_str(),
            transformation_id: final_mutation.transformation_id.as_content_id().as_str(),
            decision,
            metric_name: certificate
                .benchmark
                .as_ref()
                .map(|benchmark| benchmark.metric_name.as_str()),
            candidate_score_bits: certificate
                .benchmark
                .as_ref()
                .map(|benchmark| benchmark.candidate_score.to_bits()),
        },
    )
}

#[derive(Serialize)]
struct SearchSummaryPayload<'a> {
    candidates_attempted: usize,
    candidates_no_eligible_mutation: usize,
    candidates_failed_compile: usize,
    candidates_failed_test: usize,
    candidates_failed_benchmark: usize,
    candidates_passed_correctness: usize,
    candidates_selected_by_search: usize,
    baseline_score_bits: Option<u64>,
    best_artifact_id: Option<&'a str>,
}

#[allow(clippy::too_many_arguments)]
pub fn search_summary(
    candidates_attempted: usize,
    candidates_no_eligible_mutation: usize,
    candidates_failed_compile: usize,
    candidates_failed_test: usize,
    candidates_failed_benchmark: usize,
    candidates_passed_correctness: usize,
    candidates_selected_by_search: usize,
    baseline_score: Option<f64>,
    best: Option<&ForgeCandidate>,
) -> Result<ObservationObject, ForgeObservationError> {
    json_object(
        "symthaea.forge.search-summary.v1",
        &SearchSummaryPayload {
            candidates_attempted,
            candidates_no_eligible_mutation,
            candidates_failed_compile,
            candidates_failed_test,
            candidates_failed_benchmark,
            candidates_passed_correctness,
            candidates_selected_by_search,
            baseline_score_bits: baseline_score.map(f64::to_bits),
            best_artifact_id: best.map(|candidate| candidate.artifact_id().as_str()),
        },
    )
}

#[derive(Serialize)]
struct SearchAbortPayload<'a> {
    phase: &'a str,
    detail: &'a str,
    candidates_attempted: usize,
    candidates_no_eligible_mutation: usize,
    candidates_failed_compile: usize,
    candidates_failed_test: usize,
    candidates_failed_benchmark: usize,
    candidates_passed_correctness: usize,
    candidates_selected_by_search: usize,
    baseline_score_bits: Option<u64>,
    best_artifact_id: Option<&'a str>,
}

/// Record an apparatus/search failure without classifying any interrupted candidate as rejected.
#[allow(clippy::too_many_arguments)]
pub fn search_abort(
    phase: &str,
    detail: &str,
    candidates_attempted: usize,
    candidates_no_eligible_mutation: usize,
    candidates_failed_compile: usize,
    candidates_failed_test: usize,
    candidates_failed_benchmark: usize,
    candidates_passed_correctness: usize,
    candidates_selected_by_search: usize,
    baseline_score: Option<f64>,
    best: Option<&ForgeCandidate>,
) -> Result<ObservationObject, ForgeObservationError> {
    json_object(
        "symthaea.forge.search-aborted.v1",
        &SearchAbortPayload {
            phase,
            detail,
            candidates_attempted,
            candidates_no_eligible_mutation,
            candidates_failed_compile,
            candidates_failed_test,
            candidates_failed_benchmark,
            candidates_passed_correctness,
            candidates_selected_by_search,
            baseline_score_bits: baseline_score.map(f64::to_bits),
            best_artifact_id: best.map(|candidate| candidate.artifact_id().as_str()),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::full_source_artifact_id;

    #[test]
    fn candidate_generated_object_changes_with_transformation() {
        let parent = full_source_artifact_id("fn f() {}\n");
        let child_a = full_source_artifact_id("fn f() { let _a = 1; }\n");
        let child_b = full_source_artifact_id("fn f() { let _b = 2; }\n");
        let a = MutationRecord::new(0, "literal", "a", parent.clone(), child_a);
        let b = MutationRecord::new(0, "literal", "b", parent, child_b);
        assert_ne!(candidate_generated(&a).unwrap().id(), candidate_generated(&b).unwrap().id());
    }

    #[test]
    fn gate_object_preserves_output_and_duration() {
        let parent = full_source_artifact_id("fn f() {}\n");
        let child = full_source_artifact_id("fn f() { let _a = 1; }\n");
        let mutation = MutationRecord::new(0, "literal", "a", parent, child);
        let mut gate = GateResult {
            gate: crate::fitness::Gate::Compile,
            passed: false,
            output_tail: "error A".into(),
            duration: std::time::Duration::from_nanos(10),
        };
        let a = gates(&mutation, &[gate.clone()]).unwrap();
        gate.output_tail = "error B".into();
        let b = gates(&mutation, &[gate]).unwrap();
        assert_ne!(a.id(), b.id());
    }

    #[test]
    fn abort_object_binds_phase_and_detail() {
        let a = search_abort(
            "candidate-evaluation",
            "runner lost",
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            None,
            None,
        )
        .unwrap();
        let b = search_abort(
            "candidate-restoration",
            "runner lost",
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            None,
            None,
        )
        .unwrap();
        assert_ne!(a.id(), b.id());
    }
}
