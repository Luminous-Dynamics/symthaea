// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Label-blind retrieval-policy memory, kept separate from world/evidence memory.
//!
//! This module learns only reusable tool/action templates and their operating
//! envelope. It does not store query text, question IDs, gold answers, evaluator
//! judgments, hidden task families, attack assignments, retrieved facts or
//! source payloads. A strategy note may recommend where/how to search; it is
//! never evidence that a world-state claim is true or that a source is
//! authorized for a purpose.
//!
//! The in-memory implementation is a deterministic reference prototype. A
//! hosted service must persist trace receipts and promoted notes durably and
//! must apply the same tenant/domain authorization rules at its storage layer.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub const MIN_SUPPORTING_TRACES_V1: usize = 3;
pub const MAX_SUPPORTING_TRACES_V1: usize = 128;
pub const MAX_TOOL_CALLS_PER_TRACE_V1: usize = 256;
pub const MAX_EVIDENCE_IDS_PER_TRACE_V1: usize = 10_000;
pub const MAX_IDENTIFIER_BYTES_V1: usize = 256;
pub const MAX_DOMAIN_ID_BYTES_V1: usize = 512;
pub const MAX_EVIDENCE_ID_BYTES_V1: usize = 1_024;
pub const MAX_STRATEGY_TTL_SECONDS_V1: u64 = 30 * 24 * 60 * 60;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum RetrievalOperationOutcomeV1 {
    EvidenceObserved,
    NoEvidence,
    ToolError,
}

impl RetrievalOperationOutcomeV1 {
    fn tag(self) -> u8 {
        match self {
            Self::EvidenceObserved => 1,
            Self::NoEvidence => 2,
            Self::ToolError => 3,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::EvidenceObserved => "evidence_observed",
            Self::NoEvidence => "no_evidence",
            Self::ToolError => "tool_error",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetrievalStopReasonV1 {
    EvidenceSufficient,
    BudgetExhausted,
    NoCandidateEvidence,
    ToolError,
    Abstained,
}

impl RetrievalStopReasonV1 {
    fn tag(self) -> u8 {
        match self {
            Self::EvidenceSufficient => 1,
            Self::BudgetExhausted => 2,
            Self::NoCandidateEvidence => 3,
            Self::ToolError => 4,
            Self::Abstained => 5,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::EvidenceSufficient => "evidence_sufficient",
            Self::BudgetExhausted => "budget_exhausted",
            Self::NoCandidateEvidence => "no_candidate_evidence",
            Self::ToolError => "tool_error",
            Self::Abstained => "abstained",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetrievalOperationV1 {
    /// Stable, non-secret tool identifier (e.g. "filesystem.search").
    pub tool_id: String,
    /// Digest of the exact tool/API implementation or pinned adapter.
    pub tool_revision_sha256: String,
    /// Digest of a parameterized action template, never raw query arguments.
    pub action_template_sha256: String,
    /// IDs of source/candidate evidence actually returned by this operation.
    pub observed_evidence_ids: Vec<String>,
    pub outcome: RetrievalOperationOutcomeV1,
    pub elapsed_ns: u64,
    pub inspected_bytes: u64,
    /// False means the adapter cannot account for all bytes inspected by the operation.
    pub inspected_bytes_measured: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetrievalTraceReceiptV1 {
    pub run_id: String,
    /// Official opaque per-query invocation ID, not a benchmark question ID.
    pub query_invocation_id: String,
    pub domain_id: String,
    pub environment_revision_sha256: String,
    pub tool_profile_sha256: String,
    /// Digest of exact query bytes; no query text is retained.
    pub query_sha256: String,
    /// Digest of the candidate evidence set produced by the query, if any.
    pub candidate_set_sha256: Option<String>,
    pub operations: Vec<RetrievalOperationV1>,
    pub stop_reason: RetrievalStopReasonV1,
    pub tool_call_budget: u32,
    pub tool_calls_used: u32,
    pub elapsed_ns: u64,
    pub inspected_bytes: u64,
    /// True only when every operation reports complete inspected-byte accounting.
    pub inspected_bytes_complete: bool,
    pub created_at_unix: u64,
}

impl RetrievalTraceReceiptV1 {
    /// Deterministic receipt commitment using the versioned binary encoding.
    pub fn receipt_sha256(&self) -> String {
        trace_receipt_digest(self)
    }

    /// JSON projection for an audit/replay envelope. The receipt digest commits
    /// to the typed fields using the separate canonical binary encoding.
    pub fn to_json_value(&self) -> Value {
        json!({
            "schema_version": "mace_retrieval_trace_receipt_v1",
            "run_id": self.run_id,
            "query_invocation_id": self.query_invocation_id,
            "domain_id": self.domain_id,
            "environment_revision_sha256": self.environment_revision_sha256,
            "tool_profile_sha256": self.tool_profile_sha256,
            "query_sha256": self.query_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "operations": self.operations.iter().map(|operation| json!({
                "tool_id": operation.tool_id,
                "tool_revision_sha256": operation.tool_revision_sha256,
                "action_template_sha256": operation.action_template_sha256,
                "observed_evidence_ids": operation.observed_evidence_ids,
                "outcome": operation.outcome.as_str(),
                "elapsed_ns": operation.elapsed_ns,
                "inspected_bytes": operation.inspected_bytes,
                "inspected_bytes_measured": operation.inspected_bytes_measured,
            })).collect::<Vec<_>>(),
            "stop_reason": self.stop_reason.as_str(),
            "tool_call_budget": self.tool_call_budget,
            "tool_calls_used": self.tool_calls_used,
            "elapsed_ns": self.elapsed_ns,
            "inspected_bytes": self.inspected_bytes,
            "inspected_bytes_complete": self.inspected_bytes_complete,
            "created_at_unix": self.created_at_unix,
            "receipt_sha256": self.receipt_sha256(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetrievalTraceRefV1 {
    pub run_id: String,
    pub query_invocation_id: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetrievalStrategyStepV1 {
    pub tool_id: String,
    pub tool_revision_sha256: String,
    pub action_template_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetrievalStrategyNoteV1 {
    /// The digest is also the immutable note ID.
    pub note_sha256: String,
    pub domain_id: String,
    pub environment_revision_sha256: String,
    pub tool_profile_sha256: String,
    /// Digest of the ordered tool/action template sequence.
    pub strategy_signature_sha256: String,
    pub steps: Vec<RetrievalStrategyStepV1>,
    /// Sorted receipt digests of independent query invocations supporting promotion.
    pub supporting_trace_receipt_sha256: Vec<String>,
    pub consolidation_algorithm_sha256: String,
    pub promoted_at_unix: u64,
    pub expires_at_unix: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetrievalPolicyRejectV1 {
    InvalidIdentifier,
    InvalidDigest,
    EmptyOperations,
    TooManyToolCalls,
    ToolCallCountMismatch,
    InvalidOperation,
    DuplicateEvidenceId,
    EvidenceOutcomeMismatch,
    InvalidResourceAccounting,
    InvalidStopReason,
    CandidateDigestMismatch,
    InvocationAlreadyBound,
    UnknownTrace,
    DuplicateTraceReference,
    TooFewSupportingTraces,
    TooManySupportingTraces,
    DuplicateSupportingQuery,
    MixedDomain,
    MixedEnvironmentRevision,
    MixedToolProfile,
    MixedStrategy,
    UnsupportedTraceOutcome,
    EmptyStrategy,
    InvalidPromotionTime,
    StrategyExpired,
    InvalidTtl,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StoredRetrievalTraceV1 {
    receipt: RetrievalTraceReceiptV1,
    receipt_sha256: String,
    strategy_signature_sha256: String,
}

#[derive(Clone, Debug, Default)]
pub struct RetrievalPolicyMemoryV1 {
    traces: BTreeMap<(String, String), StoredRetrievalTraceV1>,
    notes: BTreeMap<String, RetrievalStrategyNoteV1>,
}

fn valid_atom(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && !value.chars().any(char::is_control)
}

fn valid_atom_with_limit(value: &str, maximum_bytes: usize) -> bool {
    valid_atom(value) && value.len() <= maximum_bytes
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        result.push(HEX[usize::from(byte >> 4)] as char);
        result.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    result
}

fn trace_receipt_digest(trace: &RetrievalTraceReceiptV1) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:retrieval-trace:v1\0");
    for value in [
        trace.run_id.as_str(),
        trace.query_invocation_id.as_str(),
        trace.domain_id.as_str(),
        trace.environment_revision_sha256.as_str(),
        trace.tool_profile_sha256.as_str(),
        trace.query_sha256.as_str(),
    ] {
        hash_field(&mut hasher, value);
    }
    match &trace.candidate_set_sha256 {
        Some(value) => {
            hasher.update([1]);
            hash_field(&mut hasher, value);
        }
        None => hasher.update([0]),
    }
    hasher.update(trace.stop_reason.tag().to_be_bytes());
    hasher.update(trace.tool_call_budget.to_be_bytes());
    hasher.update(trace.tool_calls_used.to_be_bytes());
    hasher.update(trace.elapsed_ns.to_be_bytes());
    hasher.update(trace.inspected_bytes.to_be_bytes());
    hasher.update([u8::from(trace.inspected_bytes_complete)]);
    hasher.update(trace.created_at_unix.to_be_bytes());
    hasher.update((trace.operations.len() as u64).to_be_bytes());
    for operation in &trace.operations {
        hash_field(&mut hasher, &operation.tool_id);
        hash_field(&mut hasher, &operation.tool_revision_sha256);
        hash_field(&mut hasher, &operation.action_template_sha256);
        hasher.update(operation.outcome.tag().to_be_bytes());
        hasher.update(operation.elapsed_ns.to_be_bytes());
        hasher.update(operation.inspected_bytes.to_be_bytes());
        hasher.update([u8::from(operation.inspected_bytes_measured)]);
        let mut evidence_ids = operation.observed_evidence_ids.clone();
        evidence_ids.sort();
        hasher.update((evidence_ids.len() as u64).to_be_bytes());
        for evidence_id in evidence_ids {
            hash_field(&mut hasher, &evidence_id);
        }
    }
    hex_digest(&hasher.finalize())
}

fn strategy_signature(trace: &RetrievalTraceReceiptV1) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:retrieval-strategy:v1\0");
    hasher.update((trace.operations.len() as u64).to_be_bytes());
    for operation in &trace.operations {
        hash_field(&mut hasher, &operation.tool_id);
        hash_field(&mut hasher, &operation.tool_revision_sha256);
        hash_field(&mut hasher, &operation.action_template_sha256);
    }
    hex_digest(&hasher.finalize())
}

fn strategy_steps(trace: &RetrievalTraceReceiptV1) -> Vec<RetrievalStrategyStepV1> {
    trace
        .operations
        .iter()
        .map(|operation| RetrievalStrategyStepV1 {
            tool_id: operation.tool_id.clone(),
            tool_revision_sha256: operation.tool_revision_sha256.clone(),
            action_template_sha256: operation.action_template_sha256.clone(),
        })
        .collect()
}

fn note_digest(note: &RetrievalStrategyNoteV1) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:retrieval-strategy-note:v1\0");
    for value in [
        note.domain_id.as_str(),
        note.environment_revision_sha256.as_str(),
        note.tool_profile_sha256.as_str(),
        note.strategy_signature_sha256.as_str(),
        note.consolidation_algorithm_sha256.as_str(),
    ] {
        hash_field(&mut hasher, value);
    }
    hasher.update(note.promoted_at_unix.to_be_bytes());
    hasher.update(note.expires_at_unix.to_be_bytes());
    hasher.update((note.steps.len() as u64).to_be_bytes());
    for step in &note.steps {
        hash_field(&mut hasher, &step.tool_id);
        hash_field(&mut hasher, &step.tool_revision_sha256);
        hash_field(&mut hasher, &step.action_template_sha256);
    }
    hasher.update((note.supporting_trace_receipt_sha256.len() as u64).to_be_bytes());
    for receipt in &note.supporting_trace_receipt_sha256 {
        hash_field(&mut hasher, receipt);
    }
    hex_digest(&hasher.finalize())
}

fn validate_trace(trace: &RetrievalTraceReceiptV1) -> Result<(), RetrievalPolicyRejectV1> {
    if !valid_atom_with_limit(&trace.run_id, MAX_IDENTIFIER_BYTES_V1)
        || !valid_atom_with_limit(&trace.query_invocation_id, MAX_IDENTIFIER_BYTES_V1)
        || !valid_atom_with_limit(&trace.domain_id, MAX_DOMAIN_ID_BYTES_V1)
    {
        return Err(RetrievalPolicyRejectV1::InvalidIdentifier);
    }
    for value in [
        trace.environment_revision_sha256.as_str(),
        trace.tool_profile_sha256.as_str(),
        trace.query_sha256.as_str(),
    ] {
        if !valid_sha256(value) {
            return Err(RetrievalPolicyRejectV1::InvalidDigest);
        }
    }
    if trace.candidate_set_sha256.as_deref().is_some_and(|value| !valid_sha256(value)) {
        return Err(RetrievalPolicyRejectV1::InvalidDigest);
    }
    if trace.operations.is_empty() {
        return Err(RetrievalPolicyRejectV1::EmptyOperations);
    }
    if trace.tool_call_budget == 0 || trace.tool_call_budget as usize > MAX_TOOL_CALLS_PER_TRACE_V1 {
        return Err(RetrievalPolicyRejectV1::TooManyToolCalls);
    }
    if trace.operations.len() > MAX_TOOL_CALLS_PER_TRACE_V1 {
        return Err(RetrievalPolicyRejectV1::TooManyToolCalls);
    }
    if trace.tool_calls_used as usize != trace.operations.len()
        || trace.tool_calls_used > trace.tool_call_budget
    {
        return Err(RetrievalPolicyRejectV1::ToolCallCountMismatch);
    }

    let mut elapsed_sum = 0_u64;
    let mut bytes_sum = 0_u64;
    let mut total_evidence_ids = 0_usize;
    let mut observed_any = false;
    let mut saw_tool_error = false;
    for operation in &trace.operations {
        if !valid_atom_with_limit(&operation.tool_id, MAX_IDENTIFIER_BYTES_V1)
            || !valid_sha256(&operation.tool_revision_sha256)
            || !valid_sha256(&operation.action_template_sha256)
        {
            return Err(RetrievalPolicyRejectV1::InvalidOperation);
        }
        elapsed_sum = elapsed_sum
            .checked_add(operation.elapsed_ns)
            .ok_or(RetrievalPolicyRejectV1::InvalidResourceAccounting)?;
        bytes_sum = bytes_sum
            .checked_add(operation.inspected_bytes)
            .ok_or(RetrievalPolicyRejectV1::InvalidResourceAccounting)?;
        let mut local_ids = BTreeSet::new();
        total_evidence_ids = total_evidence_ids
            .checked_add(operation.observed_evidence_ids.len())
            .ok_or(RetrievalPolicyRejectV1::InvalidOperation)?;
        if total_evidence_ids > MAX_EVIDENCE_IDS_PER_TRACE_V1 {
            return Err(RetrievalPolicyRejectV1::InvalidOperation);
        }
        match operation.outcome {
            RetrievalOperationOutcomeV1::EvidenceObserved => {
                if operation.observed_evidence_ids.is_empty() {
                    return Err(RetrievalPolicyRejectV1::EvidenceOutcomeMismatch);
                }
                observed_any = true;
            }
            RetrievalOperationOutcomeV1::NoEvidence | RetrievalOperationOutcomeV1::ToolError => {
                if !operation.observed_evidence_ids.is_empty() {
                    return Err(RetrievalPolicyRejectV1::EvidenceOutcomeMismatch);
                }
            }
        }
        if operation.outcome == RetrievalOperationOutcomeV1::ToolError {
            saw_tool_error = true;
        }
        for evidence_id in &operation.observed_evidence_ids {
            if !valid_atom_with_limit(evidence_id, MAX_EVIDENCE_ID_BYTES_V1) {
                return Err(RetrievalPolicyRejectV1::InvalidOperation);
            }
            if !local_ids.insert(evidence_id) {
                return Err(RetrievalPolicyRejectV1::DuplicateEvidenceId);
            }
        }
    }
    let all_bytes_measured = trace.operations.iter()
        .all(|operation| operation.inspected_bytes_measured);
    if elapsed_sum != trace.elapsed_ns
        || bytes_sum != trace.inspected_bytes
        || all_bytes_measured != trace.inspected_bytes_complete
    {
        return Err(RetrievalPolicyRejectV1::InvalidResourceAccounting);
    }

    match trace.stop_reason {
        RetrievalStopReasonV1::EvidenceSufficient => {
            if !observed_any
                || !trace
                    .candidate_set_sha256
                    .as_deref()
                    .is_some_and(valid_sha256)
            {
                return Err(RetrievalPolicyRejectV1::InvalidStopReason);
            }
        }
        RetrievalStopReasonV1::BudgetExhausted => {
            if trace.tool_calls_used != trace.tool_call_budget {
                return Err(RetrievalPolicyRejectV1::InvalidStopReason);
            }
        }
        RetrievalStopReasonV1::NoCandidateEvidence => {
            if observed_any || trace.candidate_set_sha256.is_some() {
                return Err(RetrievalPolicyRejectV1::InvalidStopReason);
            }
        }
        RetrievalStopReasonV1::ToolError => {
            if !saw_tool_error
                || trace.operations.last().map(|operation| operation.outcome)
                    != Some(RetrievalOperationOutcomeV1::ToolError)
            {
                return Err(RetrievalPolicyRejectV1::InvalidStopReason);
            }
        }
        RetrievalStopReasonV1::Abstained => {}
    }
    Ok(())
}

impl RetrievalPolicyMemoryV1 {
    /// Record a completed, label-blind interaction trace. Replays of the same
    /// run/invocation and identical receipt are idempotent; rebinding is rejected.
    pub fn record_trace(
        &mut self,
        trace: RetrievalTraceReceiptV1,
    ) -> Result<String, RetrievalPolicyRejectV1> {
        validate_trace(&trace)?;
        let key = (trace.run_id.clone(), trace.query_invocation_id.clone());
        let receipt_sha256 = trace_receipt_digest(&trace);
        let signature = strategy_signature(&trace);
        if let Some(existing) = self.traces.get(&key) {
            if existing.receipt_sha256 != receipt_sha256 {
                return Err(RetrievalPolicyRejectV1::InvocationAlreadyBound);
            }
            return Ok(existing.receipt_sha256.clone());
        }
        self.traces.insert(
            key,
            StoredRetrievalTraceV1 {
                receipt: trace,
                receipt_sha256: receipt_sha256.clone(),
                strategy_signature_sha256: signature,
            },
        );
        Ok(receipt_sha256)
    }

    pub fn trace_receipt_sha256(
        &self,
        run_id: &str,
        query_invocation_id: &str,
    ) -> Option<&str> {
        self.traces
            .get(&(run_id.to_owned(), query_invocation_id.to_owned()))
            .map(|stored| stored.receipt_sha256.as_str())
    }

    /// Promote a search strategy only after at least three distinct successful
    /// query invocations share exactly the same domain, environment, tool
    /// profile and ordered tool/action template sequence.
    pub fn promote_strategy(
        &mut self,
        references: &[RetrievalTraceRefV1],
        consolidation_algorithm_sha256: &str,
        promoted_at_unix: u64,
        expires_at_unix: u64,
    ) -> Result<RetrievalStrategyNoteV1, RetrievalPolicyRejectV1> {
        if !valid_sha256(consolidation_algorithm_sha256) {
            return Err(RetrievalPolicyRejectV1::InvalidDigest);
        }
        if references.len() < MIN_SUPPORTING_TRACES_V1 {
            return Err(RetrievalPolicyRejectV1::TooFewSupportingTraces);
        }
        if references.len() > MAX_SUPPORTING_TRACES_V1 {
            return Err(RetrievalPolicyRejectV1::TooManySupportingTraces);
        }
        if promoted_at_unix >= expires_at_unix {
            return Err(RetrievalPolicyRejectV1::InvalidPromotionTime);
        }
        if expires_at_unix - promoted_at_unix > MAX_STRATEGY_TTL_SECONDS_V1 {
            return Err(RetrievalPolicyRejectV1::InvalidTtl);
        }

        let mut seen_refs = BTreeSet::new();
        let mut seen_queries = BTreeSet::new();
        let mut selected = Vec::with_capacity(references.len());
        for reference in references {
            if !valid_atom_with_limit(&reference.run_id, MAX_IDENTIFIER_BYTES_V1)
                || !valid_atom_with_limit(&reference.query_invocation_id, MAX_IDENTIFIER_BYTES_V1)
            {
                return Err(RetrievalPolicyRejectV1::InvalidIdentifier);
            }
            let key = (
                reference.run_id.clone(),
                reference.query_invocation_id.clone(),
            );
            if !seen_refs.insert(key.clone()) {
                return Err(RetrievalPolicyRejectV1::DuplicateTraceReference);
            }
            let stored = self
                .traces
                .get(&key)
                .ok_or(RetrievalPolicyRejectV1::UnknownTrace)?;
            if stored.receipt.stop_reason != RetrievalStopReasonV1::EvidenceSufficient {
                return Err(RetrievalPolicyRejectV1::UnsupportedTraceOutcome);
            }
            if !seen_queries.insert(stored.receipt.query_sha256.clone()) {
                return Err(RetrievalPolicyRejectV1::DuplicateSupportingQuery);
            }
            selected.push(stored);
        }

        let first = selected[0];
        let mut supporting_trace_receipt_sha256 = Vec::with_capacity(selected.len());
        for stored in &selected {
            if stored.receipt.domain_id != first.receipt.domain_id {
                return Err(RetrievalPolicyRejectV1::MixedDomain);
            }
            if stored.receipt.environment_revision_sha256
                != first.receipt.environment_revision_sha256
            {
                return Err(RetrievalPolicyRejectV1::MixedEnvironmentRevision);
            }
            if stored.receipt.tool_profile_sha256 != first.receipt.tool_profile_sha256 {
                return Err(RetrievalPolicyRejectV1::MixedToolProfile);
            }
            if stored.strategy_signature_sha256 != first.strategy_signature_sha256 {
                return Err(RetrievalPolicyRejectV1::MixedStrategy);
            }
            if stored.receipt.created_at_unix > promoted_at_unix {
                return Err(RetrievalPolicyRejectV1::InvalidPromotionTime);
            }
            supporting_trace_receipt_sha256.push(stored.receipt_sha256.clone());
        }
        supporting_trace_receipt_sha256.sort();

        let mut note = RetrievalStrategyNoteV1 {
            note_sha256: String::new(),
            domain_id: first.receipt.domain_id.clone(),
            environment_revision_sha256: first.receipt.environment_revision_sha256.clone(),
            tool_profile_sha256: first.receipt.tool_profile_sha256.clone(),
            strategy_signature_sha256: first.strategy_signature_sha256.clone(),
            steps: strategy_steps(&first.receipt),
            supporting_trace_receipt_sha256,
            consolidation_algorithm_sha256: consolidation_algorithm_sha256.to_owned(),
            promoted_at_unix,
            expires_at_unix,
        };
        if note.steps.is_empty() {
            return Err(RetrievalPolicyRejectV1::EmptyStrategy);
        }
        note.note_sha256 = note_digest(&note);
        self.notes
            .entry(note.note_sha256.clone())
            .or_insert_with(|| note.clone());
        Ok(note)
    }

    /// Return only notes whose domain, environment revision, tool profile and
    /// validity interval match exactly. A changed environment is negative
    /// transfer, not an invitation to silently reuse a stale strategy.
    pub fn applicable_notes(
        &self,
        domain_id: &str,
        environment_revision_sha256: &str,
        tool_profile_sha256: &str,
        now_unix: u64,
    ) -> Vec<&RetrievalStrategyNoteV1> {
        self.notes
            .values()
            .filter(|note| {
                note.domain_id == domain_id
                    && note.environment_revision_sha256 == environment_revision_sha256
                    && note.tool_profile_sha256 == tool_profile_sha256
                    && note.promoted_at_unix <= now_unix
                    && now_unix < note.expires_at_unix
            })
            .collect()
    }

    pub fn note_count(&self) -> usize {
        self.notes.len()
    }

    pub fn trace_count(&self) -> usize {
        self.traces.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENV_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const TOOL_PROFILE_A: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const TOOL_REVISION: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const ACTION_SEARCH: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const ACTION_OPEN: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const CANDIDATE_SET: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const ALGORITHM: &str = "1111111111111111111111111111111111111111111111111111111111111111";

    fn operation(
        action: &str,
        outcome: RetrievalOperationOutcomeV1,
        evidence_ids: Vec<&str>,
    ) -> RetrievalOperationV1 {
        RetrievalOperationV1 {
            tool_id: "filesystem.search".to_owned(),
            tool_revision_sha256: TOOL_REVISION.to_owned(),
            action_template_sha256: action.to_owned(),
            observed_evidence_ids: evidence_ids.into_iter().map(str::to_owned).collect(),
            outcome,
            elapsed_ns: 10,
            inspected_bytes: 128,
            inspected_bytes_measured: true,
        }
    }

    fn trace(
        invocation: &str,
        query_digest: &str,
        domain: &str,
        environment: &str,
        tool_profile: &str,
        action: &str,
    ) -> RetrievalTraceReceiptV1 {
        let operations = vec![
            operation(
                action,
                RetrievalOperationOutcomeV1::EvidenceObserved,
                vec!["source:evidence-1"],
            ),
            operation(
                ACTION_OPEN,
                RetrievalOperationOutcomeV1::EvidenceObserved,
                vec!["source:evidence-2"],
            ),
        ];
        RetrievalTraceReceiptV1 {
            run_id: "run-1".to_owned(),
            query_invocation_id: invocation.to_owned(),
            domain_id: domain.to_owned(),
            environment_revision_sha256: environment.to_owned(),
            tool_profile_sha256: tool_profile.to_owned(),
            query_sha256: query_digest.to_owned(),
            candidate_set_sha256: Some(CANDIDATE_SET.to_owned()),
            operations,
            stop_reason: RetrievalStopReasonV1::EvidenceSufficient,
            tool_call_budget: 8,
            tool_calls_used: 2,
            elapsed_ns: 20,
            inspected_bytes: 256,
            inspected_bytes_complete: true,
            created_at_unix: 100,
        }
    }

    fn refs(invocations: &[&str]) -> Vec<RetrievalTraceRefV1> {
        invocations
            .iter()
            .map(|invocation| RetrievalTraceRefV1 {
                run_id: "run-1".to_owned(),
                query_invocation_id: (*invocation).to_owned(),
            })
            .collect()
    }

    fn three_successful_traces(memory: &mut RetrievalPolicyMemoryV1) -> Vec<RetrievalTraceRefV1> {
        let samples = [
            ("query-a", "1010101010101010101010101010101010101010101010101010101010101010"),
            ("query-b", "2020202020202020202020202020202020202020202020202020202020202020"),
            ("query-c", "3030303030303030303030303030303030303030303030303030303030303030"),
        ];
        for (invocation, query_digest) in samples {
            memory
                .record_trace(trace(
                    invocation,
                    query_digest,
                    "web-shop-a",
                    ENV_A,
                    TOOL_PROFILE_A,
                    ACTION_SEARCH,
                ))
                .expect("trace accepted");
        }
        refs(&["query-a", "query-b", "query-c"])
    }

    #[test]
    fn trace_receipt_is_idempotent_but_invocation_rebinding_is_rejected() {
        let mut memory = RetrievalPolicyMemoryV1::default();
        let original = trace(
            "invocation-1",
            "1212121212121212121212121212121212121212121212121212121212121212",
            "web-shop-a",
            ENV_A,
            TOOL_PROFILE_A,
            ACTION_SEARCH,
        );
        let first = memory.record_trace(original.clone()).expect("record");
        assert_eq!(memory.record_trace(original).expect("idempotent"), first);

        let changed = trace(
            "invocation-1",
            "1313131313131313131313131313131313131313131313131313131313131313",
            "web-shop-a",
            ENV_A,
            TOOL_PROFILE_A,
            ACTION_SEARCH,
        );
        assert_eq!(
            memory.record_trace(changed),
            Err(RetrievalPolicyRejectV1::InvocationAlreadyBound)
        );
        assert_eq!(memory.trace_count(), 1);
    }

    #[test]
    fn promotion_requires_three_distinct_successful_query_invocations() {
        let mut memory = RetrievalPolicyMemoryV1::default();
        let references = three_successful_traces(&mut memory);
        let note = memory
            .promote_strategy(&references[..2], ALGORITHM, 200, 300)
            .expect_err("two traces do not meet support threshold");
        assert_eq!(note, RetrievalPolicyRejectV1::TooFewSupportingTraces);

        let note = memory
            .promote_strategy(&references, ALGORITHM, 200, 300)
            .expect("three traces promote");
        assert_eq!(note.supporting_trace_receipt_sha256.len(), 3);
        assert_eq!(note.steps.len(), 2);
        assert_eq!(memory.note_count(), 1);
    }

    #[test]
    fn repeated_query_digest_cannot_masquerade_as_independent_support() {
        let mut memory = RetrievalPolicyMemoryV1::default();
        let query = "4141414141414141414141414141414141414141414141414141414141414141";
        for invocation in ["q1", "q2", "q3"] {
            memory
                .record_trace(trace(
                    invocation,
                    query,
                    "web-shop-a",
                    ENV_A,
                    TOOL_PROFILE_A,
                    ACTION_SEARCH,
                ))
                .unwrap();
        }
        assert_eq!(
            memory.promote_strategy(&refs(&["q1", "q2", "q3"]), ALGORITHM, 200, 300),
            Err(RetrievalPolicyRejectV1::DuplicateSupportingQuery)
        );
    }

    #[test]
    fn promotion_rejects_mixed_domain_environment_tool_profile_and_strategy() {
        let mut memory = RetrievalPolicyMemoryV1::default();
        memory.record_trace(trace("d1", "5151515151515151515151515151515151515151515151515151515151515151", "web-a", ENV_A, TOOL_PROFILE_A, ACTION_SEARCH)).unwrap();
        memory.record_trace(trace("d2", "5252525252525252525252525252525252525252525252525252525252525252", "web-b", ENV_A, TOOL_PROFILE_A, ACTION_SEARCH)).unwrap();
        memory.record_trace(trace("d3", "5353535353535353535353535353535353535353535353535353535353535353", "web-a", ENV_A, TOOL_PROFILE_A, ACTION_SEARCH)).unwrap();
        assert_eq!(
            memory.promote_strategy(&refs(&["d1","d2","d3"]), ALGORITHM, 200, 300),
            Err(RetrievalPolicyRejectV1::MixedDomain)
        );

        let mut environment_memory = RetrievalPolicyMemoryV1::default();
        environment_memory.record_trace(trace("e1", "6161616161616161616161616161616161616161616161616161616161616161", "web-a", ENV_A, TOOL_PROFILE_A, ACTION_SEARCH)).unwrap();
        environment_memory.record_trace(trace("e2", "6262626262626262626262626262626262626262626262626262626262626262", "web-a", "9999999999999999999999999999999999999999999999999999999999999999", TOOL_PROFILE_A, ACTION_SEARCH)).unwrap();
        environment_memory.record_trace(trace("e3", "6363636363636363636363636363636363636363636363636363636363636363", "web-a", ENV_A, TOOL_PROFILE_A, ACTION_SEARCH)).unwrap();
        assert_eq!(
            environment_memory.promote_strategy(&refs(&["e1","e2","e3"]), ALGORITHM, 200, 300),
            Err(RetrievalPolicyRejectV1::MixedEnvironmentRevision)
        );

        let mut profile_memory = RetrievalPolicyMemoryV1::default();
        profile_memory.record_trace(trace("p1", "7171717171717171717171717171717171717171717171717171717171717171", "web-a", ENV_A, TOOL_PROFILE_A, ACTION_SEARCH)).unwrap();
        profile_memory.record_trace(trace("p2", "7272727272727272727272727272727272727272727272727272727272727272", "web-a", ENV_A, "8888888888888888888888888888888888888888888888888888888888888888", ACTION_SEARCH)).unwrap();
        profile_memory.record_trace(trace("p3", "7373737373737373737373737373737373737373737373737373737373737373", "web-a", ENV_A, TOOL_PROFILE_A, ACTION_SEARCH)).unwrap();
        assert_eq!(
            profile_memory.promote_strategy(&refs(&["p1","p2","p3"]), ALGORITHM, 200, 300),
            Err(RetrievalPolicyRejectV1::MixedToolProfile)
        );

        let mut strategy_memory = RetrievalPolicyMemoryV1::default();
        strategy_memory.record_trace(trace("s1", "8181818181818181818181818181818181818181818181818181818181818181", "web-a", ENV_A, TOOL_PROFILE_A, ACTION_SEARCH)).unwrap();
        strategy_memory.record_trace(trace("s2", "8282828282828282828282828282828282828282828282828282828282828282", "web-a", ENV_A, TOOL_PROFILE_A, ACTION_OPEN)).unwrap();
        strategy_memory.record_trace(trace("s3", "8383838383838383838383838383838383838383838383838383838383838383", "web-a", ENV_A, TOOL_PROFILE_A, ACTION_SEARCH)).unwrap();
        assert_eq!(
            strategy_memory.promote_strategy(&refs(&["s1","s2","s3"]), ALGORITHM, 200, 300),
            Err(RetrievalPolicyRejectV1::MixedStrategy)
        );
    }

    #[test]
    fn policy_note_is_not_applicable_outside_its_exact_domain_and_versions() {
        let mut memory = RetrievalPolicyMemoryV1::default();
        let supporting = three_successful_traces(&mut memory);
        let note = memory.promote_strategy(&supporting, ALGORITHM, 200, 300).unwrap();

        assert_eq!(
            memory.applicable_notes("web-shop-a", ENV_A, TOOL_PROFILE_A, 250),
            vec![&note]
        );
        assert!(memory.applicable_notes("web-shop-b", ENV_A, TOOL_PROFILE_A, 250).is_empty());
        assert!(memory.applicable_notes(
            "web-shop-a",
            "9999999999999999999999999999999999999999999999999999999999999999",
            TOOL_PROFILE_A,
            250
        ).is_empty());
        assert!(memory.applicable_notes(
            "web-shop-a",
            ENV_A,
            "8888888888888888888888888888888888888888888888888888888888888888",
            250
        ).is_empty());
    }

    #[test]
    fn expired_and_not_yet_promoted_notes_are_never_returned() {
        let mut memory = RetrievalPolicyMemoryV1::default();
        let supporting = three_successful_traces(&mut memory);
        memory.promote_strategy(&supporting, ALGORITHM, 200, 300).unwrap();

        assert!(memory.applicable_notes("web-shop-a", ENV_A, TOOL_PROFILE_A, 199).is_empty());
        assert!(memory.applicable_notes("web-shop-a", ENV_A, TOOL_PROFILE_A, 300).is_empty());
        assert_eq!(
            memory.promote_strategy(&supporting, ALGORITHM, 200, 200),
            Err(RetrievalPolicyRejectV1::InvalidPromotionTime)
        );
        assert_eq!(
            memory.promote_strategy(&supporting, ALGORITHM, 200, 200 + MAX_STRATEGY_TTL_SECONDS_V1 + 1),
            Err(RetrievalPolicyRejectV1::InvalidTtl)
        );
    }

    #[test]
    fn no_result_is_not_silently_promoted_as_exhausted_search() {
        let mut memory = RetrievalPolicyMemoryV1::default();
        let mut no_result = trace(
            "no-result",
            "9191919191919191919191919191919191919191919191919191919191919191",
            "web-shop-a",
            ENV_A,
            TOOL_PROFILE_A,
            ACTION_SEARCH,
        );
        no_result.candidate_set_sha256 = None;
        no_result.operations = vec![operation(
            ACTION_SEARCH,
            RetrievalOperationOutcomeV1::NoEvidence,
            vec![],
        )];
        no_result.tool_calls_used = 1;
        no_result.elapsed_ns = 10;
        no_result.inspected_bytes = 128;
        no_result.stop_reason = RetrievalStopReasonV1::NoCandidateEvidence;

        memory.record_trace(no_result).expect("valid no-result trace");
        assert_eq!(
            memory.promote_strategy(&refs(&["no-result"]), ALGORITHM, 200, 300),
            Err(RetrievalPolicyRejectV1::TooFewSupportingTraces)
        );

        let mut early_stop = trace(
            "early-stop",
            "9292929292929292929292929292929292929292929292929292929292929292",
            "web-shop-a",
            ENV_A,
            TOOL_PROFILE_A,
            ACTION_SEARCH,
        );
        early_stop.candidate_set_sha256 = None;
        early_stop.operations = vec![operation(
            ACTION_SEARCH,
            RetrievalOperationOutcomeV1::NoEvidence,
            vec![],
        )];
        early_stop.tool_calls_used = 1;
        early_stop.elapsed_ns = 10;
        early_stop.inspected_bytes = 128;
        early_stop.stop_reason = RetrievalStopReasonV1::BudgetExhausted;
        early_stop.tool_call_budget = 4;
        assert_eq!(
            memory.record_trace(early_stop),
            Err(RetrievalPolicyRejectV1::InvalidStopReason)
        );
    }

    #[test]
    fn inspected_byte_completeness_is_explicit_and_committed() {
        let mut receipt = trace(
            "byte-accounting",
            "a595959595959595959595959595959595959595959595959595959595959595",
            "web-shop-a",
            ENV_A,
            TOOL_PROFILE_A,
            ACTION_SEARCH,
        );
        receipt.operations[0].inspected_bytes_measured = false;
        receipt.inspected_bytes_complete = false;
        let digest = receipt.receipt_sha256();
        assert!(digest.len() == 64);
        let json = receipt.to_json_value();
        assert_eq!(json["inspected_bytes_complete"], false);
        assert_eq!(json["operations"][0]["inspected_bytes_measured"], false);
        assert_eq!(json["receipt_sha256"], digest);

        receipt.inspected_bytes_complete = true;
        assert_eq!(
            RetrievalPolicyMemoryV1::default().record_trace(receipt),
            Err(RetrievalPolicyRejectV1::InvalidResourceAccounting)
        );
    }

    #[test]
    fn trace_rejects_unbounded_identifiers_and_evidence_fanout() {
        let mut memory = RetrievalPolicyMemoryV1::default();
        let mut long_tool = trace(
            "long-tool",
            "a393939393939393939393939393939393939393939393939393939393939393",
            "web-shop-a",
            ENV_A,
            TOOL_PROFILE_A,
            ACTION_SEARCH,
        );
        long_tool.operations[0].tool_id = "t".repeat(MAX_IDENTIFIER_BYTES_V1 + 1);
        assert_eq!(
            memory.record_trace(long_tool),
            Err(RetrievalPolicyRejectV1::InvalidOperation)
        );

        let mut too_many_ids = trace(
            "too-many-ids",
            "a494949494949494949494949494949494949494949494949494949494949494",
            "web-shop-a",
            ENV_A,
            TOOL_PROFILE_A,
            ACTION_SEARCH,
        );
        too_many_ids.operations[0].observed_evidence_ids =
            (0..=MAX_EVIDENCE_IDS_PER_TRACE_V1)
                .map(|index| format!("source:{index}"))
                .collect();
        assert_eq!(
            memory.record_trace(too_many_ids),
            Err(RetrievalPolicyRejectV1::InvalidOperation)
        );
        assert_eq!(memory.trace_count(), 0);
    }

    #[test]
    fn trace_rejects_unaccounted_cost_and_missing_evidence_for_success() {
        let mut memory = RetrievalPolicyMemoryV1::default();
        let mut invalid_cost = trace(
            "cost",
            "a191919191919191919191919191919191919191919191919191919191919191",
            "web-shop-a",
            ENV_A,
            TOOL_PROFILE_A,
            ACTION_SEARCH,
        );
        invalid_cost.elapsed_ns += 1;
        assert_eq!(
            memory.record_trace(invalid_cost),
            Err(RetrievalPolicyRejectV1::InvalidResourceAccounting)
        );

        let mut false_success = trace(
            "false-success",
            "a292929292929292929292929292929292929292929292929292929292929292",
            "web-shop-a",
            ENV_A,
            TOOL_PROFILE_A,
            ACTION_SEARCH,
        );
        false_success.candidate_set_sha256 = None;
        assert_eq!(
            memory.record_trace(false_success),
            Err(RetrievalPolicyRejectV1::InvalidStopReason)
        );
    }
}
