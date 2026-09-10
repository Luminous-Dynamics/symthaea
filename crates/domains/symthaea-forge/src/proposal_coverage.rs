// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-record validation between Forge's search trace and raw proposal archive.
//!
//! The trace says what lifecycle event occurred; the raw archive says what proposal opportunity
//! and draw preceded that lifecycle. This module proves those two generator-local records agree
//! before either may be used for later semantic policy-learning qualification.

use crate::proposal_trace::{
    ForgeRawProposalArchive, ForgeRawProposalDecision, ForgeRawProposalError,
};
use crate::trace::{
    validate_forge_trace_observations, ForgeTraceError, ForgeTraceEvent,
};
use crate::trial_semantics::{CANDIDATE_GENERATED_SCHEMA, NO_CANDIDATE_SCHEMA};
use serde_json::Value;
use std::collections::BTreeSet;
use symthaea_algorithms::ledger::DiscoveryEventKind;
use symthaea_algorithms::observation::{ObservationEncoding, ObservationStore};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalCoverageError {
    #[error(transparent)]
    Raw(#[from] ForgeRawProposalError),
    #[error(transparent)]
    Trace(#[from] ForgeTraceError),
    #[error("proposal-stage trace event has no matching raw proposal record")]
    MissingRawProposal,
    #[error("raw proposal archive contains a record with no proposal-stage trace event")]
    ExtraRawProposal,
    #[error("proposal-stage observation must be canonical JSON")]
    NonJsonObservation,
    #[error("proposal-stage observation payload is invalid JSON")]
    InvalidJson,
    #[error("proposal-stage observation schema is inconsistent with its event")]
    SchemaMismatch,
    #[error("raw proposal parent artifact disagrees with the trace observation")]
    ParentArtifactMismatch,
    #[error("raw proposal decision disagrees with the trace event")]
    DecisionMismatch,
    #[error("raw proposal mutation effect disagrees with CandidateGenerated observation")]
    MutationEffectMismatch,
    #[error("pre-proposal failure was incorrectly represented as a proposal-stage trace event")]
    PreProposalTrace,
}

/// Prove complete, one-to-one coverage between proposal-stage trace events and raw proposal records.
pub fn validate_forge_raw_proposal_coverage(
    trace: &[ForgeTraceEvent],
    observations: &ObservationStore,
    raw: &ForgeRawProposalArchive,
) -> Result<(), ForgeProposalCoverageError> {
    validate_forge_trace_observations(trace, observations)?;
    raw.validate()?;

    let mut expected = BTreeSet::new();
    for event in trace {
        match event.kind {
            DiscoveryEventKind::CandidateGenerated => {
                let attempt = event
                    .attempt_id
                    .as_ref()
                    .ok_or(ForgeProposalCoverageError::MissingRawProposal)?;
                let record = raw
                    .get(attempt)
                    .ok_or(ForgeProposalCoverageError::MissingRawProposal)?;
                expected.insert(attempt.as_content_id().as_str().to_string());

                let object = observations
                    .get(&event.observation_id)
                    .ok_or(ForgeProposalCoverageError::InvalidJson)?;
                if object.schema() != CANDIDATE_GENERATED_SCHEMA {
                    return Err(ForgeProposalCoverageError::SchemaMismatch);
                }
                let payload = json_payload(object)?;
                require_parent(record.parent_artifact_id().as_str(), &payload)?;
                if !matches!(record.decision(), ForgeRawProposalDecision::Selected { .. }) {
                    return Err(ForgeProposalCoverageError::DecisionMismatch);
                }
                let effect = record
                    .mutation_effect()
                    .ok_or(ForgeProposalCoverageError::MutationEffectMismatch)?;
                if payload.get("operator").and_then(Value::as_str) != Some(effect.operator())
                    || payload.get("detail").and_then(Value::as_str) != Some(effect.detail())
                {
                    return Err(ForgeProposalCoverageError::MutationEffectMismatch);
                }
            }
            DiscoveryEventKind::GeneratorNoOp => {
                let attempt = event
                    .attempt_id
                    .as_ref()
                    .ok_or(ForgeProposalCoverageError::MissingRawProposal)?;
                let object = observations
                    .get(&event.observation_id)
                    .ok_or(ForgeProposalCoverageError::InvalidJson)?;
                if object.schema() != NO_CANDIDATE_SCHEMA {
                    return Err(ForgeProposalCoverageError::SchemaMismatch);
                }
                let payload = json_payload(object)?;
                let reason = payload
                    .get("reason")
                    .and_then(Value::as_str)
                    .ok_or(ForgeProposalCoverageError::InvalidJson)?;
                if reason == "current-best-not-syn-parseable" {
                    return Err(ForgeProposalCoverageError::PreProposalTrace);
                }
                let record = raw
                    .get(attempt)
                    .ok_or(ForgeProposalCoverageError::MissingRawProposal)?;
                expected.insert(attempt.as_content_id().as_str().to_string());
                require_parent(record.parent_artifact_id().as_str(), &payload)?;
                match reason {
                    "no-eligible-ast-mutation" => {
                        if !matches!(record.decision(), ForgeRawProposalDecision::NoEligibleSites)
                            || record.mutation_effect().is_some()
                        {
                            return Err(ForgeProposalCoverageError::DecisionMismatch);
                        }
                    }
                    "mutation-rendered-identical-source" => {
                        if !matches!(record.decision(), ForgeRawProposalDecision::Selected { .. }) {
                            return Err(ForgeProposalCoverageError::DecisionMismatch);
                        }
                    }
                    _ => return Err(ForgeProposalCoverageError::DecisionMismatch),
                }
            }
            _ => {}
        }
    }

    if raw.records().len() != expected.len()
        || raw.records().iter().any(|record| {
            !expected.contains(record.attempt_id().as_content_id().as_str())
        })
    {
        return Err(ForgeProposalCoverageError::ExtraRawProposal);
    }
    Ok(())
}

fn json_payload(
    object: &symthaea_algorithms::observation::ObservationObject,
) -> Result<Value, ForgeProposalCoverageError> {
    if object.encoding() != ObservationEncoding::Json {
        return Err(ForgeProposalCoverageError::NonJsonObservation);
    }
    serde_json::from_slice(object.payload()).map_err(|_| ForgeProposalCoverageError::InvalidJson)
}

fn require_parent(expected: &str, payload: &Value) -> Result<(), ForgeProposalCoverageError> {
    if payload.get("parent_artifact_id").and_then(Value::as_str) == Some(expected) {
        Ok(())
    } else {
        Err(ForgeProposalCoverageError::ParentArtifactMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::{full_source_artifact_id, MutationRecord};
    use crate::mutations::{ComparisonOperatorSwap, Mutator};
    use crate::observations as forge_observations;
    use crate::proposal_trace::ForgeRawProposalRecord;
    use crate::trace::ForgeAttemptId;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn generated_candidate_requires_matching_raw_effect() {
        let parent = full_source_artifact_id("fn f(x: i32) -> bool { x < 5 }\n");
        let attempt = ForgeAttemptId::derive(&parent, 7, 0, 0);
        let mut body: syn::Block = syn::parse_str("{ x < 5 }").unwrap();
        let mut rng = StdRng::seed_from_u64(4);
        let recorded = Mutator::new(vec![Box::new(ComparisonOperatorSwap)])
            .mutate_one_recorded(&mut body, &mut rng);
        let raw_record = ForgeRawProposalRecord::from_recorded(
            attempt.clone(),
            parent.clone(),
            &recorded,
        )
        .unwrap();
        let effect = raw_record.mutation_effect().unwrap();
        let child = full_source_artifact_id("fn f(x: i32) -> bool { x <= 5 }\n");
        let mutation = MutationRecord::new(
            0,
            effect.operator(),
            effect.detail(),
            parent,
            child.clone(),
        );
        let generated = forge_observations::candidate_generated(&attempt, &mutation).unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt,
                0,
                DiscoveryEventKind::CandidateGenerated,
                child,
                generated.id().clone(),
            ),
            ForgeTraceEvent::aborted(
                forge_observations::search_abort(
                    "fixture",
                    "interrupted after generation",
                    1, 0, 0, 0, 0, 0, 0, None, None,
                )
                .unwrap()
                .id()
                .clone(),
            ),
        ];
        let aborted = forge_observations::search_abort(
            "fixture",
            "interrupted after generation",
            1, 0, 0, 0, 0, 0, 0, None, None,
        )
        .unwrap();
        let observations = ObservationStore::from_objects(vec![generated, aborted]).unwrap();
        let archive = ForgeRawProposalArchive::from_records(vec![raw_record]).unwrap();
        validate_forge_raw_proposal_coverage(&trace, &observations, &archive).unwrap();
    }

    #[test]
    fn no_eligible_trace_requires_no_eligible_raw_decision() {
        let parent = full_source_artifact_id("fn f() -> &'static str { \"x\" }\n");
        let attempt = ForgeAttemptId::derive(&parent, 7, 0, 0);
        let mut body: syn::Block = syn::parse_str("{ \"x\" }").unwrap();
        let mut rng = StdRng::seed_from_u64(2);
        let recorded = Mutator::new(vec![Box::new(ComparisonOperatorSwap)])
            .mutate_one_recorded(&mut body, &mut rng);
        let raw_record = ForgeRawProposalRecord::from_recorded(
            attempt.clone(),
            parent.clone(),
            &recorded,
        )
        .unwrap();
        let no_candidate = forge_observations::no_candidate(
            &attempt,
            "no-eligible-ast-mutation",
            &parent,
        )
        .unwrap();
        let summary = forge_observations::search_summary(
            1, 1, 0, 0, 0, 0, 0, None, None,
        )
        .unwrap();
        let trace = vec![
            ForgeTraceEvent::no_candidate(attempt, 0, no_candidate.id().clone()),
            ForgeTraceEvent::completed(summary.id().clone()),
        ];
        let observations = ObservationStore::from_objects(vec![no_candidate, summary]).unwrap();
        let archive = ForgeRawProposalArchive::from_records(vec![raw_record]).unwrap();
        validate_forge_raw_proposal_coverage(&trace, &observations, &archive).unwrap();
    }

    #[test]
    fn proposal_record_without_trace_event_is_rejected() {
        let parent = full_source_artifact_id("fn f() -> &'static str { \"x\" }\n");
        let attempt = ForgeAttemptId::derive(&parent, 7, 0, 0);
        let mut body: syn::Block = syn::parse_str("{ \"x\" }").unwrap();
        let mut rng = StdRng::seed_from_u64(2);
        let recorded = Mutator::new(vec![Box::new(ComparisonOperatorSwap)])
            .mutate_one_recorded(&mut body, &mut rng);
        let archive = ForgeRawProposalArchive::from_records(vec![
            ForgeRawProposalRecord::from_recorded(attempt, parent, &recorded).unwrap(),
        ])
        .unwrap();
        let summary = forge_observations::search_summary(
            0, 0, 0, 0, 0, 0, 0, None, None,
        )
        .unwrap();
        let trace = vec![ForgeTraceEvent::completed(summary.id().clone())];
        let observations = ObservationStore::from_objects(vec![summary]).unwrap();
        assert!(matches!(
            validate_forge_raw_proposal_coverage(&trace, &observations, &archive),
            Err(ForgeProposalCoverageError::ExtraRawProposal)
        ));
    }
}
