// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Semantic verification for persisted Forge bundles.
//!
//! [`bundle`](crate::bundle) proves exact file integrity and canonical file sets. This module adds
//! cross-file propositions: trace terminal must agree with manifest outcome, every observation must
//! resolve, summary/abort counters must agree with the event history, and any persisted survivor
//! must be the exact last `SelectedForContinuation` artifact.

use crate::bundle::{
    read_completed_manifest, BundleError, ForgeBundleManifest, ForgeBundleOutcome, ABORT_FILE,
    CANDIDATE_FILE, CERTIFICATE_FILE, MANIFEST_FILE, OBSERVATIONS_FILE, REPORT_FILE, TRACE_FILE,
};
use crate::certificate::{ForgeCandidate, ForgeCertificate};
use crate::search::{SearchFailure, SearchStats};
use crate::trace::{validate_forge_trace_observations, ForgeTraceEvent};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;
use symthaea_algorithms::ledger::DiscoveryEventKind;
use symthaea_algorithms::observation::{ObservationObject, ObservationStore};
use symthaea_algorithms::ContentId;
use thiserror::Error;

const COMPLETED_SCHEMA: &str = "symthaea.forge.search-summary.v1";
const ABORTED_SCHEMA: &str = "symthaea.forge.search-aborted.v1";
const MAX_ABORT_DETAIL_BYTES: usize = 4_020;

#[derive(Debug, Error)]
pub enum BundleSemanticError {
    #[error(transparent)]
    Manifest(#[from] BundleError),
    #[error("Forge bundle JSON decode failed for {file}: {detail}")]
    Json { file: &'static str, detail: String },
    #[error("Forge bundle I/O failed for {file}: {detail}")]
    Io { file: &'static str, detail: String },
    #[error("Forge trace/observation relationship is invalid: {0}")]
    Trace(String),
    #[error("Forge observation archive is invalid: {0}")]
    Observation(String),
    #[error("Forge survivor is invalid: {0}")]
    Survivor(String),
    #[error("manifest outcome disagrees with the persisted trace terminal")]
    OutcomeTerminalMismatch,
    #[error("reserved Forge file exists but is not committed by the manifest: {0}")]
    UnmanifestedReservedFile(&'static str),
    #[error("persisted survivor disagrees with the last SelectedForContinuation event")]
    SurvivorSelectionMismatch,
    #[error("completed no-winner bundle contains a selected continuation")]
    NoWinnerHasSelection,
    #[error("terminal observation schema does not match the terminal event")]
    TerminalObservationSchemaMismatch,
    #[error("terminal observation payload disagrees with the persisted trace or bundle")]
    TerminalObservationMismatch,
    #[error("aborted bundle is missing or has invalid abort metadata")]
    InvalidAbortRecord,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeAbortStats {
    pub candidates_attempted: usize,
    pub candidates_no_eligible_mutation: usize,
    pub candidates_failed_compile: usize,
    pub candidates_failed_test: usize,
    pub candidates_failed_benchmark: usize,
    pub candidates_passed_correctness: usize,
    pub candidates_selected_by_search: usize,
}

impl From<&SearchStats> for ForgeAbortStats {
    fn from(stats: &SearchStats) -> Self {
        Self {
            candidates_attempted: stats.candidates_attempted,
            candidates_no_eligible_mutation: stats.candidates_no_eligible_mutation,
            candidates_failed_compile: stats.candidates_failed_compile,
            candidates_failed_test: stats.candidates_failed_test,
            candidates_failed_benchmark: stats.candidates_failed_benchmark,
            candidates_passed_correctness: stats.candidates_passed_correctness,
            candidates_selected_by_search: stats.candidates_selected_by_search,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForgeAbortRecord {
    id: ContentId,
    phase: String,
    detail: String,
    stats: ForgeAbortStats,
    baseline_score_bits: Option<u64>,
    best_artifact_id: Option<ContentId>,
}

impl ForgeAbortRecord {
    pub fn from_failure(failure: &SearchFailure) -> Self {
        let stats = ForgeAbortStats::from(&failure.stats);
        let best_artifact_id = failure.best.as_ref().map(|candidate| candidate.artifact_id().clone());
        let baseline_score_bits = failure.baseline_benchmark_score.map(f64::to_bits);
        let id = derive_abort_record_id(
            &failure.phase,
            &failure.detail,
            &stats,
            baseline_score_bits,
            best_artifact_id.as_ref(),
        );
        Self {
            id,
            phase: failure.phase.clone(),
            detail: failure.detail.clone(),
            stats,
            baseline_score_bits,
            best_artifact_id,
        }
    }

    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn phase(&self) -> &str {
        &self.phase
    }

    pub fn detail(&self) -> &str {
        &self.detail
    }

    pub fn stats(&self) -> &ForgeAbortStats {
        &self.stats
    }

    pub fn best_artifact_id(&self) -> Option<&ContentId> {
        self.best_artifact_id.as_ref()
    }

    pub fn validate(&self) -> Result<(), BundleSemanticError> {
        if self.phase.trim().is_empty()
            || self.phase.trim() != self.phase
            || self.phase.chars().any(char::is_control)
            || self.detail.is_empty()
            || self.detail.len() > MAX_ABORT_DETAIL_BYTES
        {
            return Err(BundleSemanticError::InvalidAbortRecord);
        }
        let expected = derive_abort_record_id(
            &self.phase,
            &self.detail,
            &self.stats,
            self.baseline_score_bits,
            self.best_artifact_id.as_ref(),
        );
        if expected != self.id {
            return Err(BundleSemanticError::InvalidAbortRecord);
        }
        Ok(())
    }

    pub fn to_json_pretty(&self) -> Result<String, BundleSemanticError> {
        serde_json::to_string_pretty(self).map_err(|error| BundleSemanticError::Json {
            file: ABORT_FILE,
            detail: error.to_string(),
        })
    }

    fn observation_payload(&self) -> Value {
        json!({
            "phase": self.phase,
            "detail": self.detail,
            "candidates_attempted": self.stats.candidates_attempted,
            "candidates_no_eligible_mutation": self.stats.candidates_no_eligible_mutation,
            "candidates_failed_compile": self.stats.candidates_failed_compile,
            "candidates_failed_test": self.stats.candidates_failed_test,
            "candidates_failed_benchmark": self.stats.candidates_failed_benchmark,
            "candidates_passed_correctness": self.stats.candidates_passed_correctness,
            "candidates_selected_by_search": self.stats.candidates_selected_by_search,
            "baseline_score_bits": self.baseline_score_bits,
            "best_artifact_id": self.best_artifact_id.as_ref().map(ContentId::as_str),
        })
    }
}

fn derive_abort_record_id(
    phase: &str,
    detail: &str,
    stats: &ForgeAbortStats,
    baseline_score_bits: Option<u64>,
    best_artifact_id: Option<&ContentId>,
) -> ContentId {
    let counts = [
        stats.candidates_attempted as u128,
        stats.candidates_no_eligible_mutation as u128,
        stats.candidates_failed_compile as u128,
        stats.candidates_failed_test as u128,
        stats.candidates_failed_benchmark as u128,
        stats.candidates_passed_correctness as u128,
        stats.candidates_selected_by_search as u128,
    ];
    let mut parts = vec![phase.as_bytes().to_vec(), detail.as_bytes().to_vec()];
    parts.extend(counts.into_iter().map(|count| count.to_be_bytes().to_vec()));
    parts.push(
        baseline_score_bits
            .map(u64::to_be_bytes)
            .unwrap_or(u64::MAX.to_be_bytes())
            .to_vec(),
    );
    parts.push(
        best_artifact_id
            .map(ContentId::as_str)
            .unwrap_or("")
            .as_bytes()
            .to_vec(),
    );
    ContentId::derive(
        "symthaea.forge-abort-record.v1",
        parts.iter().map(Vec::as_slice),
    )
}

#[derive(Debug)]
pub struct VerifiedForgeBundle {
    pub manifest: ForgeBundleManifest,
    pub trace: Vec<ForgeTraceEvent>,
    pub observations: ObservationStore,
    pub survivor: Option<ForgeCandidate>,
    pub abort: Option<ForgeAbortRecord>,
}

pub fn read_completed_bundle(root: &Path) -> Result<VerifiedForgeBundle, BundleSemanticError> {
    let manifest = read_completed_manifest(root)?;
    reject_unmanifested_reserved_files(root, &manifest)?;

    let trace: Vec<ForgeTraceEvent> = read_json(root, TRACE_FILE)?;
    let objects: Vec<ObservationObject> = read_json(root, OBSERVATIONS_FILE)?;
    let observations = ObservationStore::from_objects(objects)
        .map_err(|error| BundleSemanticError::Observation(error.to_string()))?;
    validate_forge_trace_observations(&trace, &observations)
        .map_err(|error| BundleSemanticError::Trace(error.to_string()))?;

    let terminal = trace
        .last()
        .ok_or(BundleSemanticError::OutcomeTerminalMismatch)?;
    let terminal_object = observations
        .get(&terminal.observation_id)
        .ok_or(BundleSemanticError::TerminalObservationMismatch)?;
    let terminal_payload: Value = serde_json::from_slice(terminal_object.payload()).map_err(|error| {
        BundleSemanticError::Json {
            file: OBSERVATIONS_FILE,
            detail: error.to_string(),
        }
    })?;

    let last_selected = trace
        .iter()
        .rev()
        .find(|event| event.kind == DiscoveryEventKind::SelectedForContinuation)
        .and_then(|event| event.candidate_artifact_id.as_ref());
    let derived_stats = derive_stats(&trace);
    let survivor = read_survivor_if_present(root)?;

    let abort = match manifest.outcome {
        ForgeBundleOutcome::Winner => {
            if terminal.kind != DiscoveryEventKind::SearchCompleted {
                return Err(BundleSemanticError::OutcomeTerminalMismatch);
            }
            if terminal_object.schema() != COMPLETED_SCHEMA {
                return Err(BundleSemanticError::TerminalObservationSchemaMismatch);
            }
            let survivor = survivor
                .as_ref()
                .ok_or(BundleSemanticError::SurvivorSelectionMismatch)?;
            if last_selected != Some(survivor.artifact_id()) {
                return Err(BundleSemanticError::SurvivorSelectionMismatch);
            }
            validate_completed_payload(&terminal_payload, &derived_stats, Some(survivor.artifact_id()))?;
            None
        }
        ForgeBundleOutcome::NoWinner => {
            if terminal.kind != DiscoveryEventKind::SearchCompleted {
                return Err(BundleSemanticError::OutcomeTerminalMismatch);
            }
            if terminal_object.schema() != COMPLETED_SCHEMA {
                return Err(BundleSemanticError::TerminalObservationSchemaMismatch);
            }
            if survivor.is_some() || last_selected.is_some() {
                return Err(BundleSemanticError::NoWinnerHasSelection);
            }
            validate_completed_payload(&terminal_payload, &derived_stats, None)?;
            None
        }
        ForgeBundleOutcome::Aborted => {
            if terminal.kind != DiscoveryEventKind::SearchAborted {
                return Err(BundleSemanticError::OutcomeTerminalMismatch);
            }
            if terminal_object.schema() != ABORTED_SCHEMA {
                return Err(BundleSemanticError::TerminalObservationSchemaMismatch);
            }
            let abort: ForgeAbortRecord = read_json(root, ABORT_FILE)?;
            abort.validate()?;
            if abort.stats != derived_stats || abort.observation_payload() != terminal_payload {
                return Err(BundleSemanticError::TerminalObservationMismatch);
            }
            match (abort.best_artifact_id(), survivor.as_ref(), last_selected) {
                (None, None, None) => {}
                (Some(expected), Some(candidate), Some(selected))
                    if expected == candidate.artifact_id() && expected == selected => {}
                _ => return Err(BundleSemanticError::SurvivorSelectionMismatch),
            }
            Some(abort)
        }
    };

    Ok(VerifiedForgeBundle {
        manifest,
        trace,
        observations,
        survivor,
        abort,
    })
}

fn read_survivor_if_present(root: &Path) -> Result<Option<ForgeCandidate>, BundleSemanticError> {
    if !root.join(CANDIDATE_FILE).is_file() {
        return Ok(None);
    }
    let source = fs::read_to_string(root.join(CANDIDATE_FILE)).map_err(|error| {
        BundleSemanticError::Io {
            file: CANDIDATE_FILE,
            detail: error.to_string(),
        }
    })?;
    let certificate: ForgeCertificate = read_json(root, CERTIFICATE_FILE)?;
    ForgeCandidate::new(certificate, source)
        .map(Some)
        .map_err(|error| BundleSemanticError::Survivor(error.to_string()))
}

fn read_json<T: for<'de> Deserialize<'de>>(
    root: &Path,
    file: &'static str,
) -> Result<T, BundleSemanticError> {
    let bytes = fs::read(root.join(file)).map_err(|error| BundleSemanticError::Io {
        file,
        detail: error.to_string(),
    })?;
    serde_json::from_slice(&bytes).map_err(|error| BundleSemanticError::Json {
        file,
        detail: error.to_string(),
    })
}

fn reject_unmanifested_reserved_files(
    root: &Path,
    manifest: &ForgeBundleManifest,
) -> Result<(), BundleSemanticError> {
    let manifested: std::collections::BTreeSet<&str> =
        manifest.files.iter().map(|file| file.name.as_str()).collect();
    for name in [
        TRACE_FILE,
        OBSERVATIONS_FILE,
        ABORT_FILE,
        CANDIDATE_FILE,
        CERTIFICATE_FILE,
        REPORT_FILE,
    ] {
        if root.join(name).exists() && !manifested.contains(name) {
            return Err(BundleSemanticError::UnmanifestedReservedFile(name));
        }
    }
    if !root.join(MANIFEST_FILE).is_file() {
        return Err(BundleSemanticError::Manifest(BundleError::MissingFile(
            MANIFEST_FILE.to_string(),
        )));
    }
    Ok(())
}

fn derive_stats(trace: &[ForgeTraceEvent]) -> ForgeAbortStats {
    let mut generated = 0usize;
    let mut no_op = 0usize;
    let mut compile = 0usize;
    let mut correctness = 0usize;
    let mut evaluation = 0usize;
    let mut selected = 0usize;
    let mut valid_not_selected = 0usize;
    for event in trace {
        match event.kind {
            DiscoveryEventKind::CandidateGenerated => generated += 1,
            DiscoveryEventKind::GeneratorNoOp => no_op += 1,
            DiscoveryEventKind::RejectedCompilation => compile += 1,
            DiscoveryEventKind::RejectedCorrectness => correctness += 1,
            DiscoveryEventKind::RejectedEvaluation => evaluation += 1,
            DiscoveryEventKind::ValidNotSelected => valid_not_selected += 1,
            DiscoveryEventKind::SelectedForContinuation => selected += 1,
            DiscoveryEventKind::CandidateArchived
            | DiscoveryEventKind::SearchCompleted
            | DiscoveryEventKind::SearchAborted => {}
        }
    }
    ForgeAbortStats {
        candidates_attempted: generated + no_op,
        candidates_no_eligible_mutation: no_op,
        candidates_failed_compile: compile,
        candidates_failed_test: correctness,
        candidates_failed_benchmark: evaluation,
        candidates_passed_correctness: evaluation + valid_not_selected + selected,
        candidates_selected_by_search: selected,
    }
}

fn validate_completed_payload(
    payload: &Value,
    stats: &ForgeAbortStats,
    best: Option<&ContentId>,
) -> Result<(), BundleSemanticError> {
    let expected = [
        ("candidates_attempted", stats.candidates_attempted),
        (
            "candidates_no_eligible_mutation",
            stats.candidates_no_eligible_mutation,
        ),
        ("candidates_failed_compile", stats.candidates_failed_compile),
        ("candidates_failed_test", stats.candidates_failed_test),
        (
            "candidates_failed_benchmark",
            stats.candidates_failed_benchmark,
        ),
        (
            "candidates_passed_correctness",
            stats.candidates_passed_correctness,
        ),
        (
            "candidates_selected_by_search",
            stats.candidates_selected_by_search,
        ),
    ];
    for (field, value) in expected {
        if payload.get(field).and_then(Value::as_u64) != Some(value as u64) {
            return Err(BundleSemanticError::TerminalObservationMismatch);
        }
    }
    let observed_best = payload.get("best_artifact_id").and_then(Value::as_str);
    if observed_best != best.map(ContentId::as_str) {
        return Err(BundleSemanticError::TerminalObservationMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_stats_count_only_actual_continuations_as_selected() {
        let artifact = ContentId::derive("artifact", [b"a".as_slice()]);
        let obs = |name: &str| ContentId::derive("obs", [name.as_bytes()]);
        let trace = vec![
            ForgeTraceEvent::candidate(
                0,
                DiscoveryEventKind::CandidateGenerated,
                artifact.clone(),
                obs("generated"),
            ),
            ForgeTraceEvent::candidate(
                0,
                DiscoveryEventKind::ValidNotSelected,
                artifact,
                obs("not-selected"),
            ),
            ForgeTraceEvent::completed(obs("complete")),
        ];
        let stats = derive_stats(&trace);
        assert_eq!(stats.candidates_attempted, 1);
        assert_eq!(stats.candidates_passed_correctness, 1);
        assert_eq!(stats.candidates_selected_by_search, 0);
    }
}
