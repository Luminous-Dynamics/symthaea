// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Exact, budget-aware selection of minimum-cost sufficient evidence proofs.
//!
//! Conventional retrieval ranks passages by similarity and returns top-k. This
//! module solves a different problem: every requested conclusion must be
//! justified by one admissible OR-of-AND proof, and the union of selected source
//! records must fit an explicit context-token budget.
//!
//! The solver is intentionally exact and bounded. It never returns a packet as
//! optimal if the search-node limit was reached before exhaustive completion.
//! A later approximate solver must return its own lower/upper-bound certificate
//! and must not reuse the exact Supported status.
//!
//! Trust boundary:
//! - authorization_accepted and resolved_span_sha256 are outputs from external
//!   trusted adapters. This module binds their digests and values but does not
//!   verify signatures, read source files, or authorize principals.
//! - token_cost is supplied by the pinned tokenizer/packet renderer. The solver
//!   binds it but does not re-tokenize source content.
//! - A packet digest is an integrity commitment, not a digital signature.

use crate::memory_evidence::{
    EvidenceIndex, EvidenceSpeakerRole, ExactLexicalEnumerationResult,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

fn is_empty_string_set(value: &BTreeSet<String>) -> bool {
    value.is_empty()
}

fn require_non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

fn valid_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{field} must be 64 lowercase hexadecimal characters"));
    }
    Ok(())
}

fn sha256_json<T: Serialize>(value: &T, field: &str) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("failed to serialize {field}: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// One immutable, source-anchored evidence candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofEvidenceCandidate {
    pub evidence_id: String,
    pub source_revision_id: String,
    pub source_pointer: String,
    /// UTF-8 excerpt resolved from the source pointer and byte offsets.
    pub source_span_text: String,
    pub source_span_start: u64,
    pub source_span_end: u64,
    pub source_family_id: String,
    /// Trusted session identity; distinct from source-family independence.
    /// Required for any obligation with minimum_distinct_sessions > 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_session_id: Option<String>,
    /// Speaker role derived from the immutable source step identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_speaker: Option<EvidenceSpeakerRole>,
    pub scope_id: String,
    pub payload_sha256: String,
    pub source_span_sha256: String,
    /// Digest recomputed by the trusted source-span resolver.
    pub resolved_span_sha256: String,
    /// Symmetric conflict edges from a trusted contradiction resolver.
    pub conflicts_with: BTreeSet<String>,
    /// Transaction/knowledge time at which this exact revision was admitted.
    pub recorded_at_unix_ms: u64,
    /// Transaction time at which this revision was retracted, if any.
    pub retracted_at_unix_ms: Option<u64>,
    /// Inclusive world-valid start.
    pub valid_from_unix_ms: Option<u64>,
    /// Exclusive world-valid end.
    pub valid_until_unix_ms: Option<u64>,
    pub authorized_principal_id: String,
    /// Result from the trusted authorization adapter; not self-attested by the solver.
    pub authorization_accepted: bool,
    pub authorization_receipt_sha256: String,
    /// Rank from the broad candidate retriever; 1 is best. It is diagnostic,
    /// not the sufficiency objective or an exclusion threshold.
    pub retrieval_rank: u32,
    /// Exact rendered evidence-block cost, including locator metadata and excerpt text.
    pub token_cost: u32,
}

impl ProofEvidenceCandidate {
    fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("evidence_id", self.evidence_id.as_str()),
            ("source_revision_id", self.source_revision_id.as_str()),
            ("source_pointer", self.source_pointer.as_str()),
            ("source_family_id", self.source_family_id.as_str()),
            ("scope_id", self.scope_id.as_str()),
            ("authorized_principal_id", self.authorized_principal_id.as_str()),
        ] {
            require_non_empty(value, field)?;
        }
        if self.source_span_text.is_empty() {
            return Err("source_span_text must not be empty".into());
        }
        if let Some(session_id) = &self.source_session_id {
            require_non_empty(session_id, "source_session_id")?;
        }
        valid_sha256(&self.payload_sha256, "payload_sha256")?;
        valid_sha256(&self.source_span_sha256, "source_span_sha256")?;
        valid_sha256(&self.resolved_span_sha256, "resolved_span_sha256")?;
        if self.conflicts_with.contains(&self.evidence_id) {
            return Err("evidence cannot conflict with itself".into());
        }
        for conflicting_id in &self.conflicts_with {
            require_non_empty(conflicting_id, "conflicting evidence ID")?;
        }
        valid_sha256(
            &self.authorization_receipt_sha256,
            "authorization_receipt_sha256",
        )?;
        if self.source_span_end <= self.source_span_start {
            return Err("source_span_end must be greater than source_span_start".into());
        }
        if self.source_span_end - self.source_span_start != self.source_span_text.len() as u64 {
            return Err("source span byte offsets do not match resolved UTF-8 excerpt length".into());
        }
        if self.token_cost == 0 {
            return Err("token_cost must be greater than zero".into());
        }
        if self.retrieval_rank == 0 {
            return Err("retrieval_rank is one-based and must be greater than zero".into());
        }
        if self.valid_from_unix_ms.zip(self.valid_until_unix_ms)
            .is_some_and(|(start, end)| end <= start)
        {
            return Err("valid_until must be strictly after valid_from".into());
        }
        if self.retracted_at_unix_ms
            .is_some_and(|retracted| retracted < self.recorded_at_unix_ms)
        {
            return Err("retracted_at cannot precede recorded_at".into());
        }
        Ok(())
    }

    fn rejection_reasons(
        &self,
        request: &ProofBudgetRequest,
    ) -> BTreeSet<ProofCandidateRejection> {
        let mut reasons = BTreeSet::new();
        if self.recorded_at_unix_ms > request.known_at_unix_ms {
            reasons.insert(ProofCandidateRejection::NotKnownAtQueryTime);
        }
        if self.retracted_at_unix_ms
            .is_some_and(|retracted| retracted <= request.known_at_unix_ms)
        {
            reasons.insert(ProofCandidateRejection::RetractedAtQueryTime);
        }
        if self.valid_from_unix_ms
            .is_some_and(|start| request.valid_at_unix_ms < start)
            || self.valid_until_unix_ms
                .is_some_and(|end| request.valid_at_unix_ms >= end)
        {
            reasons.insert(ProofCandidateRejection::OutsideValidTime);
        }
        if self.scope_id != request.scope_id {
            reasons.insert(ProofCandidateRejection::OutOfScope);
        }
        if !self.authorization_accepted
            || self.authorized_principal_id != request.principal_id
        {
            reasons.insert(ProofCandidateRejection::Unauthorized);
        }
        let resolved_digest = sha256_hex(self.source_span_text.as_bytes());
        if self.source_span_sha256 != self.resolved_span_sha256
            || self.source_span_sha256 != resolved_digest
        {
            reasons.insert(ProofCandidateRejection::SourceSpanMismatch);
        }
        reasons
    }
}

/// One conjunctive proof. Every dependency must be present and admissible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofJustification {
    pub justification_id: String,
    pub evidence_dependencies: Vec<String>,
}

/// One required conclusion. Any one of its justifications is sufficient (OR);
/// dependencies inside a justification are conjunctive (AND).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofObligation {
    pub conclusion_id: String,
    pub alternatives: Vec<ProofJustification>,
    /// Count distinct source families within the selected proof, not revisions.
    pub minimum_independent_families: u32,
    /// Require evidence from this many distinct trusted conversation sessions.
    /// Zero disables the session constraint; it never changes source-family checks.
    /// Defaults to zero when deserializing legacy proof requests.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub minimum_distinct_sessions: u32,
    /// Minimum number of distinct evidence dependencies that must be
    /// attributable to the user. Assistant/unknown sources never count.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub minimum_user_authored_evidence: u32,
    /// Every evidence ID returned by this verified exhaustive enumeration
    /// must be included in the selected proof. Empty is the legacy/non-count path.
    #[serde(default, skip_serializing_if = "is_empty_string_set")]
    pub required_complete_evidence_ids: BTreeSet<String>,
    /// Digest of the exact enumeration receipt that produced the required IDs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enumeration_receipt_sha256: Option<String>,
    /// Token cost of this selected proof's conclusion/dependency envelope.
    pub selection_overhead_tokens: u32,
}

impl ProofObligation {
    /// Bind this required claim to a complete, independently reverified lexical
    /// enumeration. The generated proof alternative contains every match ID;
    /// the optimizer and packet verifier enforce that none can be omitted.
    ///
    /// Empty enumerations are rejected here: Broca does not yet have a
    /// separate proof-of-absence capability for zero-match answers.
    pub fn bind_verified_complete_enumeration(
        &mut self,
        index: &EvidenceIndex,
        enumeration: &ExactLexicalEnumerationResult,
        justification_id: &str,
    ) -> Result<(), String> {
        require_non_empty(justification_id, "justification_id")?;
        index.verify_exact_lexical_enumeration(enumeration)?;
        if enumeration.matches.is_empty() {
            return Err(
                "empty enumeration cannot support an answer until proof-of-absence is implemented"
                    .into(),
            );
        }
        if !self.required_complete_evidence_ids.is_empty()
            || self.enumeration_receipt_sha256.is_some()
        {
            return Err("proof obligation already has a complete-enumeration binding".into());
        }
        if self.alternatives.iter().any(|item| item.justification_id == justification_id) {
            return Err("complete-enumeration justification ID is already in use".into());
        }
        let required_ids: BTreeSet<String> = enumeration.matches
            .iter()
            .map(|item| item.evidence_id.clone())
            .collect();
        if required_ids.len() != enumeration.matches.len() {
            return Err("exact enumeration contains duplicate evidence IDs".into());
        }
        if enumeration.receipt.required_speaker_role == Some(EvidenceSpeakerRole::User) {
            let required_user_matches = u32::try_from(required_ids.len())
                .map_err(|_| "user-authored enumeration match count exceeds u32".to_string())?;
            self.minimum_user_authored_evidence =
                self.minimum_user_authored_evidence.max(required_user_matches);
        }
        self.required_complete_evidence_ids = required_ids.clone();
        self.enumeration_receipt_sha256 =
            Some(enumeration.receipt.receipt_sha256.clone());
        self.alternatives.push(ProofJustification {
            justification_id: justification_id.into(),
            evidence_dependencies: required_ids.into_iter().collect(),
        });
        Ok(())
    }
}

/// Query contract for exact minimum-cost proof selection.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofBudgetRequest {
    pub decision_id: String,
    pub principal_id: String,
    pub scope_id: String,
    pub valid_at_unix_ms: u64,
    pub known_at_unix_ms: u64,
    pub policy_id: String,
    pub policy_sha256: String,
    /// Fixed packet header/framing tokens emitted by the pinned renderer.
    pub fixed_packet_overhead_tokens: u64,
    pub token_budget: u64,
    pub max_search_nodes: u64,
    /// Tokenizer identity/version/digest used to calculate token_cost.
    pub tokenizer_id: String,
    pub tokenizer_version: String,
    pub tokenizer_sha256: String,
    /// Digest of the packet serialization/rendering implementation.
    pub packet_renderer_sha256: String,
    pub candidates: Vec<ProofEvidenceCandidate>,
    pub obligations: Vec<ProofObligation>,
}

impl ProofBudgetRequest {
    fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("decision_id", self.decision_id.as_str()),
            ("principal_id", self.principal_id.as_str()),
            ("scope_id", self.scope_id.as_str()),
            ("policy_id", self.policy_id.as_str()),
            ("tokenizer_id", self.tokenizer_id.as_str()),
            ("tokenizer_version", self.tokenizer_version.as_str()),
        ] {
            require_non_empty(value, field)?;
        }
        valid_sha256(&self.policy_sha256, "policy_sha256")?;
        valid_sha256(&self.tokenizer_sha256, "tokenizer_sha256")?;
        valid_sha256(&self.packet_renderer_sha256, "packet_renderer_sha256")?;
        if self.token_budget == 0 {
            return Err("token_budget must be greater than zero".into());
        }
        if self.max_search_nodes == 0 {
            return Err("max_search_nodes must be greater than zero".into());
        }
        if self.candidates.is_empty() {
            return Err("at least one evidence candidate is required".into());
        }
        if self.obligations.is_empty() {
            return Err("at least one proof obligation is required".into());
        }

        let mut evidence_ids = BTreeSet::new();
        for candidate in &self.candidates {
            candidate.validate()?;
            if !evidence_ids.insert(candidate.evidence_id.as_str()) {
                return Err(format!("duplicate evidence ID {:?}", candidate.evidence_id));
            }
        }

        let candidate_ids: BTreeSet<&str> = self.candidates.iter()
            .map(|candidate| candidate.evidence_id.as_str())
            .collect();
        for candidate in &self.candidates {
            for conflicting_id in &candidate.conflicts_with {
                if !candidate_ids.contains(conflicting_id.as_str()) {
                    return Err(format!("conflict edge references unknown evidence ID {conflicting_id:?}"));
                }
                let Some(conflicting) = self.candidates.iter()
                    .find(|item| item.evidence_id == *conflicting_id) else {
                    return Err(format!("conflict edge references unknown evidence ID {conflicting_id:?}"));
                };
                if !conflicting.conflicts_with.contains(&candidate.evidence_id) {
                    return Err(format!(
                        "conflict relation between {:?} and {:?} must be symmetric",
                        candidate.evidence_id, conflicting_id
                    ));
                }
            }
        }

        let mut conclusions = BTreeSet::new();
        for obligation in &self.obligations {
            require_non_empty(&obligation.conclusion_id, "conclusion_id")?;
            if !conclusions.insert(obligation.conclusion_id.as_str()) {
                return Err(format!("duplicate conclusion {:?}", obligation.conclusion_id));
            }
            if obligation.minimum_independent_families == 0 {
                return Err("minimum_independent_families must be greater than zero".into());
            }
            if obligation.minimum_distinct_sessions > 0
                && obligation.minimum_distinct_sessions as usize > self.candidates.len()
            {
                return Err(format!(
                    "minimum_distinct_sessions for {:?} exceeds candidate count",
                    obligation.conclusion_id
                ));
            }
            if obligation.minimum_user_authored_evidence > 0
                && obligation.minimum_user_authored_evidence as usize > self.candidates.len()
            {
                return Err(format!(
                    "minimum_user_authored_evidence for {:?} exceeds candidate count",
                    obligation.conclusion_id
                ));
            }
            if obligation.required_complete_evidence_ids.is_empty() {
                if obligation.enumeration_receipt_sha256.is_some() {
                    return Err("enumeration receipt digest cannot exist without required evidence IDs".into());
                }
            } else {
                let receipt_digest = obligation.enumeration_receipt_sha256.as_deref()
                    .ok_or_else(|| "complete enumeration IDs require an enumeration receipt digest".to_string())?;
                valid_sha256(receipt_digest, "enumeration_receipt_sha256")?;
                for evidence_id in &obligation.required_complete_evidence_ids {
                    if !candidate_ids.contains(evidence_id.as_str()) {
                        return Err(format!(
                            "complete enumeration evidence {:?} is absent from proof candidates",
                            evidence_id
                        ));
                    }
                }
            }
            if obligation.alternatives.is_empty() {
                return Err(format!(
                    "proof obligation {:?} must have at least one alternative",
                    obligation.conclusion_id
                ));
            }
            let mut justification_ids = BTreeSet::new();
            for justification in &obligation.alternatives {
                require_non_empty(&justification.justification_id, "justification_id")?;
                if !justification_ids.insert(justification.justification_id.as_str()) {
                    return Err(format!(
                        "duplicate justification {:?} for conclusion {:?}",
                        justification.justification_id, obligation.conclusion_id
                    ));
                }
                if justification.evidence_dependencies.is_empty() {
                    return Err(format!(
                        "justification {:?} must have at least one evidence dependency",
                        justification.justification_id
                    ));
                }
                let mut dependencies = BTreeSet::new();
                for evidence_id in &justification.evidence_dependencies {
                    require_non_empty(evidence_id, "evidence dependency ID")?;
                    if !dependencies.insert(evidence_id.as_str()) {
                        return Err(format!(
                            "duplicate evidence dependency {:?} in justification {:?}",
                            evidence_id, justification.justification_id
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofCandidateRejection {
    NotKnownAtQueryTime,
    RetractedAtQueryTime,
    OutsideValidTime,
    OutOfScope,
    Unauthorized,
    SourceSpanMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JustificationRejectionReason {
    MissingEvidenceReference,
    CandidateIneligible,
    IndependentFamilyCountInsufficient,
    SessionProvenanceMissing,
    DistinctSessionCountInsufficient,
    UserEvidenceCountInsufficient,
    CompleteEnumerationEvidenceOmitted,
    ContradictoryEvidence,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct JustificationRejection {
    pub conclusion_id: String,
    pub justification_id: String,
    pub reason: JustificationRejectionReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofBudgetStatus {
    Supported,
    NoEligibleProof,
    BudgetInsufficient,
    SearchLimitReached,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedProof {
    pub conclusion_id: String,
    pub justification_id: String,
    /// Conjunctive dependency edges carried in the packet, not inferred later.
    pub evidence_dependencies: Vec<String>,
    /// Exact pinned-renderer token overhead for this conclusion and proof edges.
    pub selection_overhead_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofPacketEvidenceRef {
    pub evidence_id: String,
    pub source_revision_id: String,
    pub source_pointer: String,
    pub source_span_text: String,
    pub source_span_start: u64,
    pub source_span_end: u64,
    pub source_family_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_speaker: Option<EvidenceSpeakerRole>,
    pub payload_sha256: String,
    pub source_span_sha256: String,
    pub authorization_receipt_sha256: String,
    pub retrieval_rank: u32,
    pub token_cost: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofBudgetResult {
    pub status: ProofBudgetStatus,
    pub request_sha256: String,
    pub selected_proofs: Vec<SelectedProof>,
    pub evidence: Vec<ProofPacketEvidenceRef>,
    pub total_token_cost: u64,
    pub search_nodes: u64,
    /// Number of admissible lower-bound calculations performed by the optimizer.
    #[serde(default)]
    pub lower_bound_evaluations: u64,
    /// Number of branches pruned because selected cost plus an admissible
    /// remaining-cost lower bound exceeded the best complete packet.
    #[serde(default)]
    pub bound_pruned_nodes: u64,
    pub candidate_rejections: BTreeMap<String, Vec<ProofCandidateRejection>>,
    pub justification_rejections: Vec<JustificationRejection>,
    pub packet_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct ProofPacketPayload<'a> {
    decision_id: &'a str,
    principal_id: &'a str,
    scope_id: &'a str,
    valid_at_unix_ms: u64,
    known_at_unix_ms: u64,
    request_sha256: &'a str,
    policy_id: &'a str,
    policy_sha256: &'a str,
    tokenizer_id: &'a str,
    tokenizer_version: &'a str,
    tokenizer_sha256: &'a str,
    packet_renderer_sha256: &'a str,
    fixed_packet_overhead_tokens: u64,
    token_budget: u64,
    selected_proofs: &'a [SelectedProof],
    evidence: &'a [ProofPacketEvidenceRef],
    total_token_cost: u64,
}

#[derive(Debug, Clone)]
struct CandidatePlan {
    evidence_ids: Vec<String>,
    selected_proofs: Vec<SelectedProof>,
    total_token_cost: u64,
}

fn canonicalize_request(mut request: ProofBudgetRequest) -> ProofBudgetRequest {
    request.candidates.sort_by(|left, right| left.evidence_id.cmp(&right.evidence_id));
    request.obligations.sort_by(|left, right| left.conclusion_id.cmp(&right.conclusion_id));
    for obligation in &mut request.obligations {
        obligation.alternatives.sort_by(|left, right| {
            left.justification_id.cmp(&right.justification_id)
        });
        for alternative in &mut obligation.alternatives {
            alternative.evidence_dependencies.sort();
        }
    }
    request
}

struct ExactSearch<'a> {
    request: &'a ProofBudgetRequest,
    candidate_by_id: BTreeMap<&'a str, &'a ProofEvidenceCandidate>,
    eligible_by_id: BTreeMap<&'a str, &'a ProofEvidenceCandidate>,
    candidate_rejections: BTreeMap<String, Vec<ProofCandidateRejection>>,
    justification_rejections: BTreeSet<JustificationRejection>,
    nodes: u64,
    lower_bound_evaluations: u64,
    bound_pruned_nodes: u64,
    budget_blocked: bool,
    arithmetic_error: bool,
    search_limit_reached: bool,
    best: Option<CandidatePlan>,
}

impl<'a> ExactSearch<'a> {
    fn new(request: &'a ProofBudgetRequest) -> Self {
        let candidate_by_id: BTreeMap<_, _> = request
            .candidates
            .iter()
            .map(|candidate| (candidate.evidence_id.as_str(), candidate))
            .collect();
        let mut eligible_by_id = BTreeMap::new();
        let mut candidate_rejections = BTreeMap::new();
        for candidate in &request.candidates {
            let reasons = candidate.rejection_reasons(request);
            if reasons.is_empty() {
                eligible_by_id.insert(candidate.evidence_id.as_str(), candidate);
            } else {
                candidate_rejections.insert(
                    candidate.evidence_id.clone(),
                    reasons.into_iter().collect(),
                );
            }
        }
        Self {
            request,
            candidate_by_id,
            eligible_by_id,
            candidate_rejections,
            justification_rejections: BTreeSet::new(),
            nodes: 0,
            lower_bound_evaluations: 0,
            bound_pruned_nodes: 0,
            budget_blocked: false,
            arithmetic_error: false,
            search_limit_reached: false,
            best: None,
        }
    }

    fn explore(
        &mut self,
        obligation_index: usize,
        selected_evidence_ids: &BTreeSet<String>,
        selected_proofs: &[SelectedProof],
    ) {
        if self.search_limit_reached {
            return;
        }
        if self.nodes >= self.request.max_search_nodes {
            self.search_limit_reached = true;
            return;
        }
        self.nodes += 1;

        if obligation_index == self.request.obligations.len() {
            let mut evidence_ids: Vec<_> = selected_evidence_ids.iter().cloned().collect();
            evidence_ids.sort();
            let Some(total_token_cost) = self.cost_of_plan(selected_evidence_ids, selected_proofs) else {
                self.arithmetic_error = true;
                return;
            };
            if total_token_cost > self.request.token_budget {
                self.budget_blocked = true;
                return;
            }
            let plan = CandidatePlan {
                total_token_cost,
                evidence_ids,
                selected_proofs: selected_proofs.to_vec(),
            };
            self.consider(plan);
            return;
        }

        // Clone the small immutable branch point so recursive mutable search
        // does not retain an immutable borrow into self.request.
        let conclusion_id = self.request.obligations[obligation_index].conclusion_id.clone();
        let minimum_independent_families =
            self.request.obligations[obligation_index].minimum_independent_families;
        let minimum_distinct_sessions =
            self.request.obligations[obligation_index].minimum_distinct_sessions;
        let minimum_user_authored_evidence =
            self.request.obligations[obligation_index].minimum_user_authored_evidence;
        let alternatives = self.request.obligations[obligation_index].alternatives.clone();
        for alternative in &alternatives {
            if self.search_limit_reached {
                return;
            }

            let mut missing = false;
            let mut ineligible = false;
            let mut missing_session_provenance = false;
            let mut families = BTreeSet::<String>::new();
            let mut sessions = BTreeSet::<String>::new();
            let mut user_authored_evidence = BTreeSet::<String>::new();
            for evidence_id in &alternative.evidence_dependencies {
                let Some(candidate) = self.candidate_by_id.get(evidence_id.as_str()) else {
                    missing = true;
                    continue;
                };
                if !self.eligible_by_id.contains_key(evidence_id.as_str()) {
                    ineligible = true;
                    continue;
                }
                families.insert(candidate.source_family_id.clone());
                if minimum_distinct_sessions > 0 {
                    if let Some(session_id) = &candidate.source_session_id {
                        sessions.insert(session_id.clone());
                    } else {
                        missing_session_provenance = true;
                    }
                }
                if candidate.source_speaker == Some(EvidenceSpeakerRole::User) {
                    user_authored_evidence.insert(candidate.evidence_id.clone());
                }
            }

            if missing || ineligible {
                self.justification_rejections.insert(JustificationRejection {
                    conclusion_id: conclusion_id.clone(),
                    justification_id: alternative.justification_id.clone(),
                    reason: if missing {
                        JustificationRejectionReason::MissingEvidenceReference
                    } else {
                        JustificationRejectionReason::CandidateIneligible
                    },
                });
                continue;
            }
            if families.len() < minimum_independent_families as usize {
                self.justification_rejections.insert(JustificationRejection {
                    conclusion_id: conclusion_id.clone(),
                    justification_id: alternative.justification_id.clone(),
                    reason: JustificationRejectionReason::IndependentFamilyCountInsufficient,
                });
                continue;
            }
            if minimum_distinct_sessions > 0 && missing_session_provenance {
                self.justification_rejections.insert(JustificationRejection {
                    conclusion_id: conclusion_id.clone(),
                    justification_id: alternative.justification_id.clone(),
                    reason: JustificationRejectionReason::SessionProvenanceMissing,
                });
                continue;
            }
            if sessions.len() < minimum_distinct_sessions as usize {
                self.justification_rejections.insert(JustificationRejection {
                    conclusion_id: conclusion_id.clone(),
                    justification_id: alternative.justification_id.clone(),
                    reason: JustificationRejectionReason::DistinctSessionCountInsufficient,
                });
                continue;
            }
            if user_authored_evidence.len() < minimum_user_authored_evidence as usize {
                self.justification_rejections.insert(JustificationRejection {
                    conclusion_id: conclusion_id.clone(),
                    justification_id: alternative.justification_id.clone(),
                    reason: JustificationRejectionReason::UserEvidenceCountInsufficient,
                });
                continue;
            }
            if self.request.obligations[obligation_index]
                .required_complete_evidence_ids
                .iter()
                .any(|required_id| !alternative.evidence_dependencies.contains(required_id))
            {
                self.justification_rejections.insert(JustificationRejection {
                    conclusion_id: conclusion_id.clone(),
                    justification_id: alternative.justification_id.clone(),
                    reason: JustificationRejectionReason::CompleteEnumerationEvidenceOmitted,
                });
                continue;
            }

            let mut next_evidence = selected_evidence_ids.clone();
            next_evidence.extend(alternative.evidence_dependencies.iter().cloned());
            if self.has_internal_conflict(&next_evidence) {
                self.justification_rejections.insert(JustificationRejection {
                    conclusion_id: conclusion_id.clone(),
                    justification_id: alternative.justification_id.clone(),
                    reason: JustificationRejectionReason::ContradictoryEvidence,
                });
                continue;
            }
            // Do not prune partial over-budget branches. Only a complete
            // admissible proof combination can establish BudgetInsufficient.
            let mut next_proofs = selected_proofs.to_vec();
            next_proofs.push(SelectedProof {
                conclusion_id: conclusion_id.clone(),
                justification_id: alternative.justification_id.clone(),
                evidence_dependencies: alternative.evidence_dependencies.clone(),
                selection_overhead_tokens: self.request.obligations[obligation_index].selection_overhead_tokens,
            });
            let Some(next_cost) = self.cost_of_plan(&next_evidence, &next_proofs) else {
                self.arithmetic_error = true;
                continue;
            };
            if self.best.as_ref()
                .is_some_and(|best| next_cost > best.total_token_cost)
            {
                continue;
            }

            let next_obligation_index = obligation_index + 1;
            if next_obligation_index < self.request.obligations.len() {
                if let Some(best_cost) = self.best.as_ref().map(|best| best.total_token_cost) {
                    let Some(remaining_lower_bound) =
                        self.lower_bound_for_remaining(next_obligation_index, &next_evidence)
                    else {
                        // Arithmetic overflow is reported by solve_proof_budget
                        // after the search returns; never treat it as a valid bound.
                        continue;
                    };
                    let Some(total_lower_bound) = next_cost.checked_add(remaining_lower_bound) else {
                        self.arithmetic_error = true;
                        continue;
                    };
                    if total_lower_bound > best_cost {
                        self.bound_pruned_nodes = self.bound_pruned_nodes.saturating_add(1);
                        // Skip only this alternative. Other alternatives at the
                        // current obligation remain candidates for the optimum.
                        continue;
                    }
                }
            }
            self.explore(next_obligation_index, &next_evidence, &next_proofs);
        }
    }

    /// Admissible lower bound for the additional cost of all remaining
    /// obligations. For each obligation, compute its cheapest currently
    /// admissible alternative relative to the evidence already selected. The
    /// maximum of these per-obligation minima is a safe bound: the eventual
    /// shared evidence union must cost at least as much as every individual
    /// remaining obligation, but summing the minima would overcount evidence
    /// that future obligations may share.
    fn lower_bound_for_remaining(
        &mut self,
        first_obligation_index: usize,
        selected_evidence_ids: &BTreeSet<String>,
    ) -> Option<u64> {
        self.lower_bound_evaluations = match self.lower_bound_evaluations.checked_add(1) {
            Some(value) => value,
            None => {
                self.arithmetic_error = true;
                return None;
            }
        };

        // Own the small immutable suffix so the bound can use the search maps
        // without holding a borrow into self.request across helper calls.
        let remaining_obligations: Vec<_> = self.request.obligations
            .iter()
            .skip(first_obligation_index)
            .cloned()
            .collect();
        let mut maximum_minimum = 0u64;

        for obligation in remaining_obligations {
            let mut minimum_for_obligation: Option<u64> = None;
            for alternative in &obligation.alternatives {
                let mut families = BTreeSet::new();
                let mut sessions = BTreeSet::new();
                let mut user_authored_evidence = BTreeSet::new();
                let mut admissible = true;
                for evidence_id in &alternative.evidence_dependencies {
                    let Some(candidate) = self.candidate_by_id.get(evidence_id.as_str()) else {
                        admissible = false;
                        break;
                    };
                    if !self.eligible_by_id.contains_key(evidence_id.as_str()) {
                        admissible = false;
                        break;
                    }
                    families.insert(candidate.source_family_id.as_str());
                    if obligation.minimum_distinct_sessions > 0 {
                        if let Some(session_id) = &candidate.source_session_id {
                            sessions.insert(session_id.as_str());
                        } else {
                            admissible = false;
                            break;
                        }
                    }
                    if candidate.source_speaker == Some(EvidenceSpeakerRole::User) {
                        user_authored_evidence.insert(candidate.evidence_id.as_str());
                    }
                }
                if !admissible
                    || families.len() < obligation.minimum_independent_families as usize
                    || sessions.len() < obligation.minimum_distinct_sessions as usize
                    || user_authored_evidence.len() < obligation.minimum_user_authored_evidence as usize
                    || obligation.required_complete_evidence_ids
                        .iter()
                        .any(|required_id| !alternative.evidence_dependencies.contains(required_id))
                {
                    continue;
                }

                let mut candidate_union = selected_evidence_ids.clone();
                candidate_union.extend(alternative.evidence_dependencies.iter().cloned());
                if self.has_internal_conflict(&candidate_union) {
                    continue;
                }

                let mut incremental_cost = u64::from(obligation.selection_overhead_tokens);
                for evidence_id in &alternative.evidence_dependencies {
                    if selected_evidence_ids.contains(evidence_id) {
                        continue;
                    }
                    let Some(candidate) = self.candidate_by_id.get(evidence_id.as_str()) else {
                        admissible = false;
                        break;
                    };
                    let Some(next_cost) =
                        incremental_cost.checked_add(u64::from(candidate.token_cost))
                    else {
                        self.arithmetic_error = true;
                        return None;
                    };
                    incremental_cost = next_cost;
                }
                if !admissible {
                    continue;
                }

                minimum_for_obligation = Some(
                    minimum_for_obligation
                        .map_or(incremental_cost, |current| current.min(incremental_cost)),
                );
            }

            // No individually admissible continuation means this branch will
            // eventually fail proof construction. A zero lower bound is
            // conservative and lets the ordinary search record its rejection.
            if let Some(minimum) = minimum_for_obligation {
                maximum_minimum = maximum_minimum.max(minimum);
            }
        }

        Some(maximum_minimum)
    }

    fn has_internal_conflict(&self, evidence_ids: &BTreeSet<String>) -> bool {
        for evidence_id in evidence_ids {
            let Some(candidate) = self.candidate_by_id.get(evidence_id.as_str()) else {
                return true;
            };
            if candidate.conflicts_with.iter().any(|other| evidence_ids.contains(other)) {
                return true;
            }
        }
        false
    }

    fn cost_of_set(&self, evidence_ids: &BTreeSet<String>) -> Option<u64> {
        evidence_ids.iter().try_fold(0u64, |total, id| {
            let candidate = self.candidate_by_id.get(id.as_str())?;
            total.checked_add(candidate.token_cost as u64)
        })
    }

    fn cost_of_plan(
        &self,
        evidence_ids: &BTreeSet<String>,
        selected_proofs: &[SelectedProof],
    ) -> Option<u64> {
        let mut total = self.request.fixed_packet_overhead_tokens
            .checked_add(self.cost_of_set(evidence_ids)?)?;
        for selected in selected_proofs {
            let obligation = self.request.obligations.iter()
                .find(|obligation| obligation.conclusion_id == selected.conclusion_id)?;
            total = total.checked_add(obligation.selection_overhead_tokens as u64)?;
        }
        Some(total)
    }

    fn consider(&mut self, candidate: CandidatePlan) {
        let better = self.best.as_ref().is_none_or(|best| {
            let candidate_proofs: Vec<_> = candidate.selected_proofs
                .iter()
                .map(|proof| (&proof.conclusion_id, &proof.justification_id))
                .collect();
            let best_proofs: Vec<_> = best.selected_proofs
                .iter()
                .map(|proof| (&proof.conclusion_id, &proof.justification_id))
                .collect();
            (candidate.total_token_cost, &candidate.evidence_ids, candidate_proofs)
                < (best.total_token_cost, &best.evidence_ids, best_proofs)
        });
        if better {
            self.best = Some(candidate);
        }
    }
}

/// Solve all obligations exactly within the supplied search-node limit.
///
/// A Supported packet is returned only after the entire bounded search space
/// has been exhausted, proving minimum token cost under this request contract.
pub fn solve_proof_budget(
    request: ProofBudgetRequest,
) -> Result<ProofBudgetResult, String> {
    request.validate()?;
    let request = canonicalize_request(request);
    let request_sha256 = sha256_json(&request, "canonical proof budget request")?;
    let mut search = ExactSearch::new(&request);
    search.explore(0, &BTreeSet::new(), &[]);
    if search.arithmetic_error {
        return Err("proof packet token-cost arithmetic overflow or inconsistent selected plan".into());
    }

    let candidate_rejections = search.candidate_rejections;
    let justification_rejections: Vec<_> =
        search.justification_rejections.into_iter().collect();

    if search.search_limit_reached {
        return Ok(ProofBudgetResult {
            status: ProofBudgetStatus::SearchLimitReached,
            request_sha256,
            selected_proofs: Vec::new(),
            evidence: Vec::new(),
            total_token_cost: 0,
            search_nodes: search.nodes,
            lower_bound_evaluations: search.lower_bound_evaluations,
            bound_pruned_nodes: search.bound_pruned_nodes,
            candidate_rejections,
            justification_rejections,
            packet_sha256: None,
        });
    }

    let Some(plan) = search.best else {
        return Ok(ProofBudgetResult {
            status: if search.budget_blocked {
                ProofBudgetStatus::BudgetInsufficient
            } else {
                ProofBudgetStatus::NoEligibleProof
            },
            request_sha256,
            selected_proofs: Vec::new(),
            evidence: Vec::new(),
            total_token_cost: 0,
            search_nodes: search.nodes,
            lower_bound_evaluations: search.lower_bound_evaluations,
            bound_pruned_nodes: search.bound_pruned_nodes,
            candidate_rejections,
            justification_rejections,
            packet_sha256: None,
        });
    };

    let candidate_map: BTreeMap<_, _> = request.candidates
        .iter()
        .map(|candidate| (candidate.evidence_id.as_str(), candidate))
        .collect();
    let evidence: Vec<_> = plan.evidence_ids
        .iter()
        .map(|evidence_id| {
            let candidate = candidate_map
                .get(evidence_id.as_str())
                .ok_or_else(|| format!("selected evidence {evidence_id:?} is absent from candidate map"))?;
            Ok(ProofPacketEvidenceRef {
                evidence_id: candidate.evidence_id.clone(),
                source_revision_id: candidate.source_revision_id.clone(),
                source_pointer: candidate.source_pointer.clone(),
                source_span_text: candidate.source_span_text.clone(),
                source_span_start: candidate.source_span_start,
                source_span_end: candidate.source_span_end,
                source_family_id: candidate.source_family_id.clone(),
                source_session_id: candidate.source_session_id.clone(),
                source_speaker: candidate.source_speaker,
                payload_sha256: candidate.payload_sha256.clone(),
                source_span_sha256: candidate.source_span_sha256.clone(),
                authorization_receipt_sha256: candidate.authorization_receipt_sha256.clone(),
                retrieval_rank: candidate.retrieval_rank,
                token_cost: candidate.token_cost,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let payload = ProofPacketPayload {
        request_sha256: &request_sha256,
        decision_id: &request.decision_id,
        principal_id: &request.principal_id,
        scope_id: &request.scope_id,
        valid_at_unix_ms: request.valid_at_unix_ms,
        known_at_unix_ms: request.known_at_unix_ms,
        policy_id: &request.policy_id,
        policy_sha256: &request.policy_sha256,
        tokenizer_id: &request.tokenizer_id,
        tokenizer_version: &request.tokenizer_version,
        tokenizer_sha256: &request.tokenizer_sha256,
        packet_renderer_sha256: &request.packet_renderer_sha256,
        fixed_packet_overhead_tokens: request.fixed_packet_overhead_tokens,
        token_budget: request.token_budget,
        selected_proofs: &plan.selected_proofs,
        evidence: &evidence,
        total_token_cost: plan.total_token_cost,
    };
    let packet_sha256 = sha256_json(&payload, "proof packet")?;

    Ok(ProofBudgetResult {
        status: ProofBudgetStatus::Supported,
        request_sha256,
        selected_proofs: plan.selected_proofs,
        evidence,
        total_token_cost: plan.total_token_cost,
        search_nodes: search.nodes,
        lower_bound_evaluations: search.lower_bound_evaluations,
        bound_pruned_nodes: search.bound_pruned_nodes,
        candidate_rejections,
        justification_rejections,
        packet_sha256: Some(packet_sha256),
    })
}

/// Independently verify packet sufficiency without invoking the optimizer.
///
/// This checks that every requested conclusion has exactly one selected
/// admissible alternative, that all conjunctive dependencies are present,
/// that independent-family thresholds are met, and that the exact source set,
/// token cost and packet digest match. It does not prove minimum cost.
pub fn verify_proof_packet(
    request: &ProofBudgetRequest,
    result: &ProofBudgetResult,
) -> Result<(), String> {
    request.validate()?;
    let request = canonicalize_request(request.clone());
    if result.status != ProofBudgetStatus::Supported {
        return Err("only a Supported result can be verified as a proof packet".into());
    }
    let request_sha256 = sha256_json(&request, "canonical proof budget request")?;
    if result.request_sha256 != request_sha256 {
        return Err("proof packet request digest mismatch".into());
    }
    if result.selected_proofs.len() != request.obligations.len() {
        return Err("proof packet does not select exactly one proof per obligation".into());
    }
    let mut canonical_selected = result.selected_proofs.clone();
    canonical_selected.sort_by(|left, right| left.conclusion_id.cmp(&right.conclusion_id));
    if canonical_selected != result.selected_proofs {
        return Err("selected proof order is not canonical by conclusion ID".into());
    }

    let candidate_by_id: BTreeMap<_, _> = request.candidates
        .iter()
        .map(|candidate| (candidate.evidence_id.as_str(), candidate))
        .collect();
    let mut selected_evidence_ids = BTreeSet::new();
    let mut seen_conclusions = BTreeSet::new();

    for selected in &result.selected_proofs {
        if !seen_conclusions.insert(selected.conclusion_id.as_str()) {
            return Err(format!("duplicate selected proof for {:?}", selected.conclusion_id));
        }
        let obligation = request.obligations
            .iter()
            .find(|obligation| obligation.conclusion_id == selected.conclusion_id)
            .ok_or_else(|| format!("unknown selected conclusion {:?}", selected.conclusion_id))?;
        let justification = obligation.alternatives
            .iter()
            .find(|alternative| alternative.justification_id == selected.justification_id)
            .ok_or_else(|| format!("unknown selected justification {:?}", selected.justification_id))?;
        if selected.evidence_dependencies != justification.evidence_dependencies {
            return Err(format!(
                "selected proof {:?} does not carry its exact conjunctive dependency edges",
                selected.justification_id
            ));
        }
        if selected.selection_overhead_tokens != obligation.selection_overhead_tokens {
            return Err(format!(
                "selected proof {:?} has incorrect rendered-envelope token overhead",
                selected.justification_id
            ));
        }
        if obligation.required_complete_evidence_ids
            .iter()
            .any(|required_id| !justification.evidence_dependencies.contains(required_id))
        {
            return Err(format!(
                "selected proof {:?} omits evidence required by its exact enumeration receipt",
                selected.justification_id
            ));
        }

        let mut independent_families = BTreeSet::new();
        let mut distinct_sessions = BTreeSet::new();
        let mut user_authored_evidence = BTreeSet::new();
        for evidence_id in &justification.evidence_dependencies {
            let candidate = candidate_by_id
                .get(evidence_id.as_str())
                .ok_or_else(|| format!("selected proof references missing evidence {evidence_id:?}"))?;
            let reasons = candidate.rejection_reasons(&request);
            if !reasons.is_empty() {
                return Err(format!(
                    "selected evidence {evidence_id:?} is ineligible: {reasons:?}"
                ));
            }
            independent_families.insert(candidate.source_family_id.as_str());
            if obligation.minimum_distinct_sessions > 0 {
                let session_id = candidate.source_session_id.as_deref().ok_or_else(|| {
                    format!("selected proof {:?} lacks trusted session provenance", selected.justification_id)
                })?;
                distinct_sessions.insert(session_id);
            }
            if candidate.source_speaker == Some(EvidenceSpeakerRole::User) {
                user_authored_evidence.insert(candidate.evidence_id.as_str());
            }
            selected_evidence_ids.insert(evidence_id.clone());
        }
        if independent_families.len() < obligation.minimum_independent_families as usize {
            return Err(format!(
                "selected proof {:?} does not meet the independent-family threshold",
                selected.justification_id
            ));
        }
        if distinct_sessions.len() < obligation.minimum_distinct_sessions as usize {
            return Err(format!(
                "selected proof {:?} does not meet the distinct-session threshold",
                selected.justification_id
            ));
        }
        if user_authored_evidence.len() < obligation.minimum_user_authored_evidence as usize {
            return Err(format!(
                "selected proof {:?} does not meet the user-authored evidence threshold",
                selected.justification_id
            ));
        }
    }

    if seen_conclusions.len() != request.obligations.len() {
        return Err("proof packet omits one or more required conclusions".into());
    }
    for evidence_id in &selected_evidence_ids {
        let candidate = candidate_by_id
            .get(evidence_id.as_str())
            .ok_or_else(|| format!("selected evidence {evidence_id:?} is absent from candidate map"))?;
        if candidate.conflicts_with.iter().any(|other| selected_evidence_ids.contains(other)) {
            return Err(format!("proof packet combines contradictory evidence at {:?}", evidence_id));
        }
    }

    let mut expected_evidence = Vec::new();
    let mut total_token_cost = request.fixed_packet_overhead_tokens;
    for evidence_id in &selected_evidence_ids {
        let candidate = candidate_by_id
            .get(evidence_id.as_str())
            .ok_or_else(|| format!("selected evidence {evidence_id:?} is absent from candidate map"))?;
        total_token_cost = total_token_cost
            .checked_add(candidate.token_cost as u64)
            .ok_or_else(|| "proof packet token cost overflow".to_string())?;
        expected_evidence.push(ProofPacketEvidenceRef {
            evidence_id: candidate.evidence_id.clone(),
            source_revision_id: candidate.source_revision_id.clone(),
            source_pointer: candidate.source_pointer.clone(),
            source_span_text: candidate.source_span_text.clone(),
            source_span_start: candidate.source_span_start,
            source_span_end: candidate.source_span_end,
            source_family_id: candidate.source_family_id.clone(),
            source_session_id: candidate.source_session_id.clone(),
            source_speaker: candidate.source_speaker,
            payload_sha256: candidate.payload_sha256.clone(),
            source_span_sha256: candidate.source_span_sha256.clone(),
            authorization_receipt_sha256: candidate.authorization_receipt_sha256.clone(),
            retrieval_rank: candidate.retrieval_rank,
            token_cost: candidate.token_cost,
        });
    }

    if result.evidence != expected_evidence {
        return Err("proof packet evidence set or source digests do not match selected proofs".into());
    }
    for selected in &result.selected_proofs {
        let obligation = request.obligations.iter()
            .find(|obligation| obligation.conclusion_id == selected.conclusion_id)
            .ok_or_else(|| format!("unknown selected conclusion {:?}", selected.conclusion_id))?;
        total_token_cost = total_token_cost
            .checked_add(obligation.selection_overhead_tokens as u64)
            .ok_or_else(|| "proof envelope token cost overflow".to_string())?;
    }

    if result.total_token_cost != total_token_cost {
        return Err("proof packet token cost does not match its unique evidence set".into());
    }
    if total_token_cost > request.token_budget {
        return Err("proof packet exceeds the requested token budget".into());
    }

    let payload = ProofPacketPayload {
        request_sha256: &request_sha256,
        decision_id: &request.decision_id,
        principal_id: &request.principal_id,
        scope_id: &request.scope_id,
        valid_at_unix_ms: request.valid_at_unix_ms,
        known_at_unix_ms: request.known_at_unix_ms,
        policy_id: &request.policy_id,
        policy_sha256: &request.policy_sha256,
        tokenizer_id: &request.tokenizer_id,
        tokenizer_version: &request.tokenizer_version,
        tokenizer_sha256: &request.tokenizer_sha256,
        packet_renderer_sha256: &request.packet_renderer_sha256,
        fixed_packet_overhead_tokens: request.fixed_packet_overhead_tokens,
        token_budget: request.token_budget,
        selected_proofs: &result.selected_proofs,
        evidence: &result.evidence,
        total_token_cost,
    };
    let expected_packet_sha256 = sha256_json(&payload, "proof packet")?;
    if result.packet_sha256.as_deref() != Some(expected_packet_sha256.as_str()) {
        return Err("proof packet digest mismatch".into());
    }
    Ok(())
}

/// Verify both packet sufficiency and exact minimum-cost optimality.
///
/// Sufficiency is checked by the separate verifier above; optimality is then
/// checked by exhaustive recomputation under the same bounded exact solver.
pub fn verify_proof_budget_result(
    request: &ProofBudgetRequest,
    result: &ProofBudgetResult,
) -> Result<(), String> {
    crate::memory_proof_optimality::verify_minimum_cost_optimality(request, result)
        .map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_evidence::{EvidenceIndex, EvidenceRecord, EvidenceSource, EvidencePayload, EvidenceLifecycle};

    fn digest(byte: char) -> String {
        std::iter::repeat(byte).take(64).collect()
    }

    fn candidate(id: &str, family: &str, token_cost: u32) -> ProofEvidenceCandidate {
        ProofEvidenceCandidate {
            evidence_id: id.into(),
            source_revision_id: format!("revision-{id}"),
            source_pointer: format!("/messages/{id}/content"),
            source_span_text: "0123456789".into(),
            source_span_start: 0,
            source_span_end: 10,
            source_family_id: family.into(),
            source_session_id: Some(format!("session-{id}")),
            source_speaker: Some(EvidenceSpeakerRole::User),
            scope_id: "scope-1".into(),
            payload_sha256: digest('a'),
            source_span_sha256: sha256_hex(b"0123456789"),
            resolved_span_sha256: sha256_hex(b"0123456789"),
            conflicts_with: BTreeSet::new(),
            recorded_at_unix_ms: 100,
            retracted_at_unix_ms: None,
            valid_from_unix_ms: Some(10),
            valid_until_unix_ms: None,
            authorized_principal_id: "principal-1".into(),
            authorization_accepted: true,
            authorization_receipt_sha256: digest('c'),
            retrieval_rank: 50,
            token_cost,
        }
    }

    fn justification(id: &str, dependencies: &[&str]) -> ProofJustification {
        ProofJustification {
            justification_id: id.into(),
            evidence_dependencies: dependencies.iter().map(|value| (*value).into()).collect(),
        }
    }

    fn obligation(id: &str, alternatives: Vec<ProofJustification>) -> ProofObligation {
        ProofObligation {
            conclusion_id: id.into(),
            alternatives,
            minimum_independent_families: 1,
            minimum_distinct_sessions: 0,
            minimum_user_authored_evidence: 0,
            required_complete_evidence_ids: BTreeSet::new(),
            enumeration_receipt_sha256: None,
            selection_overhead_tokens: 0,
        }
    }

    fn request(
        candidates: Vec<ProofEvidenceCandidate>,
        obligations: Vec<ProofObligation>,
        budget: u64,
    ) -> ProofBudgetRequest {
        ProofBudgetRequest {
            decision_id: "decision-1".into(),
            principal_id: "principal-1".into(),
            scope_id: "scope-1".into(),
            valid_at_unix_ms: 200,
            known_at_unix_ms: 200,
            policy_id: "policy-v1".into(),
            policy_sha256: digest('d'),
            tokenizer_id: "broca-tokenizer".into(),
            tokenizer_version: "2026-10-30".into(),
            tokenizer_sha256: digest('e'),
            packet_renderer_sha256: digest('f'),
            fixed_packet_overhead_tokens: 0,
            token_budget: budget,
            max_search_nodes: 10_000,
            candidates,
            obligations,
        }
    }

    #[test]
    fn admissible_remaining_cost_bound_prunes_branches_without_changing_optimum() {
        let req = request(
            vec![
                candidate("good-a", "family-good", 5),
                candidate("worse-a", "family-worse", 6),
                candidate("shared-b", "family-b", 5),
            ],
            vec![
                obligation(
                    "claim-a",
                    vec![
                        justification("a-best", &["good-a"]),
                        justification("z-worse", &["worse-a"]),
                    ],
                ),
                obligation(
                    "claim-b",
                    vec![justification("only-b", &["shared-b"])],
                ),
            ],
            64,
        );
        let result = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(result.total_token_cost, 10);
        assert_eq!(result.selected_proofs[0].justification_id, "a-best");
        assert!(result.lower_bound_evaluations > 0);
        assert_eq!(result.bound_pruned_nodes, 1);

        // The independent exhaustive verifier still checks the complete
        // Cartesian product, including the branch pruned by the optimizer.
        verify_proof_budget_result(&req, &result).unwrap();
    }

    #[test]
    fn shared_future_evidence_is_not_double_counted_in_the_lower_bound() {
        let req = request(
            vec![
                candidate("A", "family-A", 1),
                candidate("E", "family-E", 5),
                candidate("Z", "family-Z", 1),
            ],
            vec![
                obligation(
                    "claim-1",
                    vec![
                        justification("a-high-evidence-id", &["Z"]),
                        justification("z-low-evidence-id", &["A"]),
                    ],
                ),
                obligation("claim-2", vec![justification("shared-2", &["E"])]),
                obligation("claim-3", vec![justification("shared-3", &["E"])]),
            ],
            64,
        );

        let result = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(result.total_token_cost, 6);

        // The second alternative has equal cost but a canonical evidence set
        // beginning with "A", which wins the deterministic tie-break. Both
        // remaining obligations share E, so their lower-bound contribution is
        // max(5, 5)=5, not 10. Equality with the incumbent must not be pruned.
        assert_eq!(result.selected_proofs[0].justification_id, "z-low-evidence-id");
        assert_eq!(
            result.evidence.iter().map(|item| item.evidence_id.as_str()).collect::<Vec<_>>(),
            vec!["A", "E"],
        );
        assert_eq!(result.bound_pruned_nodes, 0);
        verify_proof_budget_result(&req, &result).unwrap();
    }

    #[test]
    fn selects_minimum_cost_sufficient_or_of_and_proof() {
        let req = request(
            vec![
                candidate("A", "family-A", 2),
                candidate("B", "family-B", 3),
                candidate("C", "family-C", 1),
                candidate("D", "family-D", 1),
            ],
            vec![obligation(
                "claim-1",
                vec![
                    justification("proof-AB", &["A", "B"]),
                    justification("proof-CD", &["C", "D"]),
                ],
            )],
            2,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(result.total_token_cost, 2);
        assert_eq!(result.evidence.iter().map(|item| item.evidence_id.as_str()).collect::<Vec<_>>(), vec!["C", "D"]);
        assert_eq!(result.selected_proofs[0].justification_id, "proof-CD");
    }

    #[test]
    fn shared_evidence_is_charged_once_across_obligations() {
        let req = request(
            vec![
                candidate("A", "family-A", 2),
                candidate("B", "family-B", 1),
                candidate("C", "family-C", 2),
            ],
            vec![
                obligation("claim-1", vec![justification("p1", &["A", "B"])]),
                obligation("claim-2", vec![justification("p2", &["B", "C"])]),
            ],
            5,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(result.total_token_cost, 5);
        assert_eq!(result.evidence.len(), 3);
    }

    #[test]
    fn low_relevance_premise_is_kept_when_it_completes_required_proof() {
        let mut distractor = candidate("high-score-distractor", "family-X", 1);
        distractor.retrieval_rank = 1;
        let mut required = candidate("required-premise", "family-Y", 2);
        required.retrieval_rank = 99;
        let mut required_second = candidate("required-second-premise", "family-Z", 2);
        required_second.retrieval_rank = 100;
        let req = request(
            vec![distractor, required, required_second],
            vec![obligation(
                "claim-1",
                vec![justification("complete-proof", &["required-premise", "required-second-premise"])],
            )],
            4,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert!(result.evidence.iter().any(|item| item.evidence_id == "required-premise"));
        assert!(!result.evidence.iter().any(|item| item.evidence_id == "high-score-distractor"));
    }

    #[test]
    fn multiple_revisions_from_one_family_do_not_satisfy_independence() {
        let mut req = request(
            vec![
                candidate("A-rev1", "family-A", 1),
                candidate("A-rev2", "family-A", 1),
            ],
            vec![obligation(
                "claim-1",
                vec![justification("two-copies", &["A-rev1", "A-rev2"])],
            )],
            5,
        );
        req.obligations[0].minimum_independent_families = 2;
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.justification_rejections.iter().any(|reason| {
            reason.reason == JustificationRejectionReason::IndependentFamilyCountInsufficient
        }));
    }

    #[test]
    fn optimizer_selects_higher_cost_cross_session_proof_over_cheaper_single_session_proof() {
        let mut same_session_a = candidate("A", "family-A", 1);
        same_session_a.source_session_id = Some("session-1".into());
        let mut same_session_b = candidate("B", "family-B", 1);
        same_session_b.source_session_id = Some("session-1".into());
        let mut cross_session_c = candidate("C", "family-C", 5);
        cross_session_c.source_session_id = Some("session-2".into());

        let mut obligation = obligation(
            "claim-cross-session",
            vec![
                justification("cheap-single-session", &["A", "B"]),
                justification("verified-cross-session", &["A", "C"]),
            ],
        );
        obligation.minimum_independent_families = 2;
        obligation.minimum_distinct_sessions = 2;

        let req = request(
            vec![same_session_a, same_session_b, cross_session_c],
            vec![obligation],
            64,
        );
        let result = solve_proof_budget(req.clone()).unwrap();

        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(result.total_token_cost, 6);
        assert_eq!(
            result.selected_proofs[0].justification_id,
            "verified-cross-session"
        );
        assert_eq!(
            result.evidence
                .iter()
                .map(|item| item.source_session_id.as_deref().unwrap())
                .collect::<Vec<_>>(),
            vec!["session-1", "session-2"]
        );
        assert!(result.justification_rejections.iter().any(|rejection| {
            rejection.justification_id == "cheap-single-session"
                && rejection.reason == JustificationRejectionReason::DistinctSessionCountInsufficient
        }));

        // The independent, exhaustive verifier must agree with the constrained
        // optimizer and validate the session provenance carried by the packet.
        verify_proof_budget_result(&req, &result).unwrap();
    }

    #[test]
    fn multiple_independent_sources_from_one_session_do_not_satisfy_multisession_requirement() {
        let mut a = candidate("A", "family-A", 1);
        let mut b = candidate("B", "family-B", 1);
        a.source_session_id = Some("session-one".into());
        b.source_session_id = Some("session-one".into());

        let mut obligation = obligation(
            "claim-1",
            vec![justification("two-source-one-session", &["A", "B"])],
        );
        obligation.minimum_independent_families = 2;
        obligation.minimum_distinct_sessions = 2;

        let result = solve_proof_budget(
            request(vec![a, b], vec![obligation], 10)
        ).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.justification_rejections.iter().any(|rejection| {
            rejection.reason == JustificationRejectionReason::DistinctSessionCountInsufficient
        }));
    }

    #[test]
    fn sessions_do_not_substitute_for_independent_source_families() {
        let mut a = candidate("A", "family-same", 1);
        let mut b = candidate("B", "family-same", 1);
        a.source_session_id = Some("session-one".into());
        b.source_session_id = Some("session-two".into());

        let mut obligation = obligation(
            "claim-1",
            vec![justification("two-sessions-one-family", &["A", "B"])],
        );
        obligation.minimum_independent_families = 2;
        obligation.minimum_distinct_sessions = 2;

        let result = solve_proof_budget(
            request(vec![a, b], vec![obligation], 10)
        ).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.justification_rejections.iter().any(|rejection| {
            rejection.reason == JustificationRejectionReason::IndependentFamilyCountInsufficient
        }));
    }

    #[test]
    fn missing_trusted_session_provenance_cannot_satisfy_session_requirement() {
        let mut a = candidate("A", "family-A", 1);
        a.source_session_id = None;
        let mut obligation = obligation(
            "claim-1",
            vec![justification("missing-session", &["A"])],
        );
        obligation.minimum_distinct_sessions = 1;

        let result = solve_proof_budget(
            request(vec![a], vec![obligation], 10)
        ).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.justification_rejections.iter().any(|rejection| {
            rejection.reason == JustificationRejectionReason::SessionProvenanceMissing
        }));
    }





    #[derive(Deserialize)]
    struct CrossSessionFixtureFile {
        schema_version: String,
        protocol_version: String,
        case_count: usize,
        cases: Vec<CrossSessionFixtureCase>,
    }

    #[derive(Deserialize)]
    struct CrossSessionFixtureCase {
        case_id: String,
        minimum_independent_families: u32,
        minimum_distinct_sessions: u32,
        candidates: Vec<CrossSessionFixtureCandidate>,
        alternatives: Vec<CrossSessionFixtureAlternative>,
        expected_status: ProofBudgetStatus,
        expected_selected_justification_id: Option<String>,
        expected_rejection: Option<JustificationRejectionReason>,
    }

    #[derive(Deserialize)]
    struct CrossSessionFixtureCandidate {
        evidence_id: String,
        source_family_id: String,
        source_session_id: Option<String>,
        token_cost: u32,
    }

    #[derive(Deserialize)]
    struct CrossSessionFixtureAlternative {
        justification_id: String,
        evidence_dependencies: Vec<String>,
    }

    #[test]
    fn legacy_single_session_serialization_omits_new_zero_and_absent_fields() {
        let mut legacy_candidate = candidate("A", "family-A", 2);
        legacy_candidate.source_session_id = None;
        let candidate_json = serde_json::to_value(&legacy_candidate).unwrap();
        assert!(candidate_json.get("source_session_id").is_none());

        let req = request(
            vec![legacy_candidate],
            vec![obligation(
                "claim-1",
                vec![justification("legacy-proof", &["A"])],
            )],
            32,
        );
        let request_json = serde_json::to_value(&req).unwrap();
        assert!(
            request_json["obligations"][0]
                .get("minimum_distinct_sessions")
                .is_none()
        );
        assert_eq!(
            request_json["obligations"][0]["minimum_independent_families"],
            serde_json::json!(1)
        );
    }

    #[test]
    fn proof_packet_rejects_mutated_source_session_identity() {
        let mut a = candidate("A", "family-A", 2);
        let mut b = candidate("B", "family-B", 4);
        a.source_session_id = Some("session-1".into());
        b.source_session_id = Some("session-2".into());
        let mut obligation = obligation(
            "claim-1",
            vec![justification("two-session-proof", &["A", "B"])],
        );
        obligation.minimum_independent_families = 2;
        obligation.minimum_distinct_sessions = 2;
        let req = request(vec![a, b], vec![obligation], 32);
        let mut result = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);

        result.evidence[0].source_session_id = Some("forged-session".into());
        assert!(verify_proof_budget_result(&req, &result).is_err());
    }

    #[test]
    fn versioned_cross_session_proof_fixtures_match_optimizer_and_verifier() {
        let raw = include_str!(
            "../../../../research/broca-bitemporal-evaluation-v1/cross-session-proof-fixtures-v1.json"
        );
        let fixture: CrossSessionFixtureFile = serde_json::from_str(raw).unwrap();
        assert_eq!(fixture.schema_version, "broca-cross-session-proof-fixtures-v1");
        assert_eq!(fixture.protocol_version, "BROCA-CROSS-SESSION-PROOF-V1");
        assert_eq!(fixture.case_count, fixture.cases.len());
        assert_eq!(fixture.case_count, 4);

        for case in fixture.cases {
            let candidates = case.candidates
                .into_iter()
                .map(|source| {
                    let mut item = candidate(
                        &source.evidence_id,
                        &source.source_family_id,
                        source.token_cost,
                    );
                    item.source_session_id = source.source_session_id;
                    item
                })
                .collect();
            let alternatives = case.alternatives
                .into_iter()
                .map(|alternative| ProofJustification {
                    justification_id: alternative.justification_id,
                    evidence_dependencies: alternative.evidence_dependencies,
                })
                .collect();
            let mut proof_obligation = obligation(&case.case_id, alternatives);
            proof_obligation.minimum_independent_families =
                case.minimum_independent_families;
            proof_obligation.minimum_distinct_sessions =
                case.minimum_distinct_sessions;
            let req = request(candidates, vec![proof_obligation], 128);
            let result = solve_proof_budget(req.clone()).unwrap();

            assert_eq!(
                result.status,
                case.expected_status,
                "proof status mismatch in fixture {}",
                case.case_id
            );
            if let Some(expected_id) = case.expected_selected_justification_id {
                assert_eq!(result.selected_proofs.len(), 1);
                assert_eq!(
                    result.selected_proofs[0].justification_id,
                    expected_id,
                    "selected proof mismatch in fixture {}",
                    case.case_id
                );
                verify_proof_budget_result(&req, &result).unwrap();
            } else {
                assert!(result.selected_proofs.is_empty());
                let expected_rejection = case.expected_rejection.unwrap();
                assert!(
                    result.justification_rejections.iter().any(|rejection| {
                        rejection.reason == expected_rejection
                    }),
                    "expected rejection {:?} in fixture {}",
                    expected_rejection,
                    case.case_id
                );
            }
        }
    }

    #[test]
    fn legacy_single_session_requests_and_packets_keep_canonical_json_shape() {
        let mut candidate_without_session = candidate("legacy", "family-legacy", 2);
        candidate_without_session.source_session_id = None;
        let req = request(
            vec![candidate_without_session],
            vec![obligation(
                "legacy-claim",
                vec![justification("legacy-proof", &["legacy"])],
            )],
            64,
        );

        let request_json = serde_json::to_value(&req).unwrap();
        assert!(
            request_json["candidates"][0].get("source_session_id").is_none(),
            "absent session metadata must not change legacy canonical request JSON"
        );
        assert!(
            request_json["obligations"][0]
                .get("minimum_distinct_sessions")
                .is_none(),
            "zero session threshold must not change legacy canonical request JSON"
        );

        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        let packet_json = serde_json::to_value(&result).unwrap();
        assert!(
            packet_json["evidence"][0].get("source_session_id").is_none(),
            "absent session metadata must not change legacy packet JSON"
        );
    }


    #[test]
    fn optimizer_prefers_user_authored_evidence_over_cheaper_assistant_only_match() {
        let mut assistant = candidate("assistant", "family-assistant", 1);
        assistant.source_speaker = Some(EvidenceSpeakerRole::Assistant);
        let mut user = candidate("user", "family-user", 5);
        user.source_speaker = Some(EvidenceSpeakerRole::User);
        let mut required = obligation(
            "claim-user-source",
            vec![
                justification("cheap-assistant-only", &["assistant"]),
                justification("user-authored-proof", &["user"]),
            ],
        );
        required.minimum_user_authored_evidence = 1;

        let req = request(vec![assistant, user], vec![required], 32);
        let result = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(result.total_token_cost, 5);
        assert_eq!(result.selected_proofs[0].justification_id, "user-authored-proof");
        assert_eq!(result.evidence[0].source_speaker, Some(EvidenceSpeakerRole::User));
        assert!(result.justification_rejections.iter().any(|rejection| {
            rejection.justification_id == "cheap-assistant-only"
                && rejection.reason == JustificationRejectionReason::UserEvidenceCountInsufficient
        }));
        verify_proof_budget_result(&req, &result).unwrap();
    }

    #[test]
    fn assistant_and_unknown_speaker_evidence_never_satisfy_user_requirement() {
        for role in [Some(EvidenceSpeakerRole::Assistant), None] {
            let mut item = candidate("only", "family-only", 1);
            item.source_speaker = role;
            let mut required = obligation(
                "claim-user-source",
                vec![justification("non-user-proof", &["only"])],
            );
            required.minimum_user_authored_evidence = 1;
            let result = solve_proof_budget(
                request(vec![item], vec![required], 16)
            ).unwrap();

            assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
            assert!(result.justification_rejections.iter().any(|rejection| {
                rejection.reason == JustificationRejectionReason::UserEvidenceCountInsufficient
            }));
        }
    }

    #[test]
    fn proof_packet_rejects_speaker_role_tampering() {
        let mut user = candidate("user", "family-user", 2);
        user.source_speaker = Some(EvidenceSpeakerRole::User);
        let mut required = obligation(
            "claim-user-source",
            vec![justification("user-proof", &["user"])],
        );
        required.minimum_user_authored_evidence = 1;
        let req = request(vec![user], vec![required], 16);
        let mut result = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        result.evidence[0].source_speaker = Some(EvidenceSpeakerRole::Assistant);
        assert!(verify_proof_budget_result(&req, &result).is_err());
    }

    fn insert_enumeration_test_record(
        index: &mut EvidenceIndex,
        hdc_id: u64,
        evidence_id: &str,
        session_id: &str,
        step_id: &str,
        text: &str,
    ) {
        let source_bytes = serde_json::to_vec(&serde_json::json!({"evidence": text})).unwrap();
        let record = EvidenceRecord {
            schema_version: crate::memory_evidence::EVIDENCE_SCHEMA_VERSION.into(),
            evidence_id: evidence_id.into(),
            hdc_id,
            scope_id: "scope-1".into(),
            source: EvidenceSource {
                trajectory_id: session_id.into(),
                step_id: Some(step_id.into()),
                step_index: step_id.rsplit('-').next().and_then(|s| s.parse().ok()),
                source_sha256: sha256_hex(&source_bytes),
                source_pointer: "/evidence".into(),
            },
            payload: EvidencePayload::Text { value: text.into() },
            content_sha256: sha256_hex(text.as_bytes()),
            lifecycle: EvidenceLifecycle::Active,
            occurred_at_unix_ms: Some(100),
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            superseded_by: None,
        };
        index.insert(record, &source_bytes).unwrap();
    }

    #[test]
    fn complete_enumeration_receipt_requires_every_match_in_the_selected_proof() {
        let mut index = EvidenceIndex::new();
        insert_enumeration_test_record(&mut index, 1, "E1", "S1", "user-0", "invoice paid in March");
        insert_enumeration_test_record(&mut index, 2, "E2", "S2", "assistant-1", "invoice paid suggestion");
        insert_enumeration_test_record(&mut index, 3, "E3", "S3", "user-2", "invoice paid in April");
        insert_enumeration_test_record(&mut index, 4, "E4", "S4", "credit issued in May");

        let enumeration = index.enumerate_exact_lexical_all_terms_at_snapshot(
            "invoice paid",
            "scope-1",
            200,
            300,
            None,
        ).unwrap();
        assert_eq!(enumeration.receipt.matching_record_count, 3);

        let mut obligation = ProofObligation {
            conclusion_id: "count-invoice-events".into(),
            alternatives: Vec::new(),
            minimum_independent_families: 2,
            minimum_distinct_sessions: 2,
            minimum_user_authored_evidence: 2,
            required_complete_evidence_ids: BTreeSet::new(),
            enumeration_receipt_sha256: None,
            selection_overhead_tokens: 1,
        };
        obligation.bind_verified_complete_enumeration(
            &index,
            &enumeration,
            "all-invoice-events",
        ).unwrap();
        assert_eq!(
            obligation.required_complete_evidence_ids,
            BTreeSet::from(["E1".into(), "E2".into(), "E3".into()])
        );
        assert_eq!(
            obligation.enumeration_receipt_sha256.as_deref(),
            Some(enumeration.receipt.receipt_sha256.as_str())
        );

        let mut e1 = candidate("E1", "family-1", 1);
        e1.source_session_id = Some("S1".into());
        e1.source_speaker = Some(EvidenceSpeakerRole::User);
        let mut e2 = candidate("E2", "family-2", 1);
        e2.source_session_id = Some("S2".into());
        e2.source_speaker = Some(EvidenceSpeakerRole::Assistant);
        let mut e3 = candidate("E3", "family-3", 1);
        e3.source_session_id = Some("S3".into());
        e3.source_speaker = Some(EvidenceSpeakerRole::User);

        let req = request(vec![e1, e2, e3], vec![obligation.clone()], 32);
        let result = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(
            result.evidence.iter().map(|item| item.evidence_id.as_str()).collect::<Vec<_>>(),
            vec!["E1", "E2", "E3"]
        );
        verify_proof_budget_result(&req, &result).unwrap();

        // A tempting partial packet still satisfies the family, session and
        // user-source floors, but omits one exact match from the enumeration.
        // Both optimizer and independent verifier must refuse to support it.
        let mut partial = obligation;
        partial.alternatives[0].evidence_dependencies = vec!["E1".into(), "E3".into()];
        let partial_req = request(
            req.candidates.clone(),
            vec![partial],
            32,
        );
        let partial_result = solve_proof_budget(partial_req).unwrap();
        assert_eq!(partial_result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(partial_result.justification_rejections.iter().any(|rejection| {
            rejection.reason == JustificationRejectionReason::CompleteEnumerationEvidenceOmitted
        }));
    }

    #[test]
    fn complete_enumeration_binding_rejects_tampered_or_empty_enumerations() {
        let mut index = EvidenceIndex::new();
        insert_enumeration_test_record(&mut index, 1, "E1", "S1", "user-0", "invoice paid");
        let valid = index.enumerate_exact_lexical_all_terms_at_snapshot(
            "invoice paid", "scope-1", 200, 300, None,
        ).unwrap();

        let mut tampered = valid.clone();
        tampered.matches.clear();
        let mut obligation = ProofObligation {
            conclusion_id: "claim-1".into(),
            alternatives: Vec::new(),
            minimum_independent_families: 1,
            minimum_distinct_sessions: 0,
            minimum_user_authored_evidence: 0,
            required_complete_evidence_ids: BTreeSet::new(),
            enumeration_receipt_sha256: None,
            selection_overhead_tokens: 0,
        };
        assert!(obligation.bind_verified_complete_enumeration(
            &index, &tampered, "all-matches"
        ).is_err());

        let empty = index.enumerate_exact_lexical_all_terms_at_snapshot(
            "never-present", "scope-1", 200, 300, None,
        ).unwrap();
        assert!(obligation.bind_verified_complete_enumeration(
            &index, &empty, "empty-enumeration"
        ).unwrap_err().contains("proof-of-absence"));
    }

    #[test]
    fn out_of_scope_candidate_is_not_eligible() {
        let mut item = candidate("A", "family-A", 1);
        item.scope_id = "scope-2".into();
        let req = request(
            vec![item],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.candidate_rejections["A"].contains(&ProofCandidateRejection::OutOfScope));
    }

    #[test]
    fn retracted_candidate_is_not_eligible_after_known_time() {
        let mut item = candidate("A", "family-A", 1);
        item.retracted_at_unix_ms = Some(150);
        let req = request(
            vec![item],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.candidate_rejections["A"].contains(&ProofCandidateRejection::RetractedAtQueryTime));
    }

    #[test]
    fn not_yet_known_candidate_cannot_leak_from_future_transaction_time() {
        let mut item = candidate("A", "family-A", 1);
        item.recorded_at_unix_ms = 201;
        let req = request(
            vec![item],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.candidate_rejections["A"].contains(&ProofCandidateRejection::NotKnownAtQueryTime));
    }

    #[test]
    fn world_validity_is_separate_from_knowledge_time() {
        let mut item = candidate("A", "family-A", 1);
        item.valid_from_unix_ms = Some(201);
        let req = request(
            vec![item],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.candidate_rejections["A"].contains(&ProofCandidateRejection::OutsideValidTime));
    }

    #[test]
    fn exclusive_valid_until_boundary_is_enforced() {
        let mut item = candidate("A", "family-A", 1);
        item.valid_until_unix_ms = Some(200);
        let req = request(
            vec![item],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.candidate_rejections["A"].contains(&ProofCandidateRejection::OutsideValidTime));
    }

    #[test]
    fn rejected_authorization_receipt_blocks_evidence() {
        let mut item = candidate("A", "family-A", 1);
        item.authorization_accepted = false;
        let req = request(
            vec![item],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.candidate_rejections["A"].contains(&ProofCandidateRejection::Unauthorized));
    }

    #[test]
    fn principal_mismatch_blocks_evidence_even_with_accepted_receipt() {
        let mut item = candidate("A", "family-A", 1);
        item.authorized_principal_id = "principal-2".into();
        let req = request(
            vec![item],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.candidate_rejections["A"].contains(&ProofCandidateRejection::Unauthorized));
    }

    #[test]
    fn mutated_source_span_is_rejected() {
        let mut item = candidate("A", "family-A", 1);
        item.resolved_span_sha256 = digest('e');
        let req = request(
            vec![item],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.candidate_rejections["A"].contains(&ProofCandidateRejection::SourceSpanMismatch));
    }

    #[test]
    fn one_token_over_budget_returns_insufficient_not_truncated_proof() {
        let req = request(
            vec![candidate("A", "family-A", 2), candidate("B", "family-B", 2)],
            vec![obligation("claim-1", vec![justification("p1", &["A", "B"])])],
            3,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::BudgetInsufficient);
        assert!(result.evidence.is_empty());
        assert_eq!(result.packet_sha256, None);
    }

    #[test]
    fn partial_over_budget_branch_without_complete_proof_is_no_eligible_proof() {
        let req = request(
            vec![
                candidate("A-expensive", "family-A", 10),
                candidate("B", "family-B", 1),
            ],
            vec![
                obligation("claim-1", vec![justification("p1", &["A-expensive"])]),
                obligation("claim-2", vec![justification("p2", &["missing-source"])]),
            ],
            2,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.evidence.is_empty());
        assert_eq!(result.packet_sha256, None);
    }

    #[test]
    fn missing_dependency_does_not_produce_a_partial_packet() {
        let req = request(
            vec![candidate("A", "family-A", 1)],
            vec![obligation("claim-1", vec![justification("p1", &["A", "missing-B"])])],
            5,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.evidence.is_empty());
        assert!(result.justification_rejections.iter().any(|reason| {
            reason.reason == JustificationRejectionReason::MissingEvidenceReference
        }));
    }

    #[test]
    fn candidate_and_alternative_order_do_not_change_packet_digest() {
        let mut req = request(
            vec![
                candidate("A", "family-A", 2),
                candidate("B", "family-B", 1),
                candidate("C", "family-C", 1),
            ],
            vec![obligation(
                "claim-1",
                vec![
                    justification("proof-AB", &["A", "B"]),
                    justification("proof-C", &["C"]),
                ],
            )],
            5,
        );
        let left = solve_proof_budget(req.clone()).unwrap();
        req.candidates.reverse();
        req.obligations[0].alternatives.reverse();
        req.obligations[0].alternatives[0].evidence_dependencies.reverse();
        let right = solve_proof_budget(req).unwrap();
        assert_eq!(left, right);
    }

    #[test]
    fn independent_verifier_rejects_packet_mutation() {
        let req = request(
            vec![candidate("A", "family-A", 1)],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        let mut result = solve_proof_budget(req.clone()).unwrap();
        verify_proof_budget_result(&req, &result).unwrap();
        result.evidence[0].payload_sha256 = digest('f');
        assert!(verify_proof_budget_result(&req, &result).is_err());
    }

    #[test]
    fn rendered_packet_header_and_proof_envelope_costs_are_included() {
        let mut claim = obligation("claim-1", vec![justification("p1", &["A", "B"])]);
        claim.selection_overhead_tokens = 2;
        let mut req = request(
            vec![candidate("A", "family-A", 1), candidate("B", "family-B", 1)],
            vec![claim],
            5,
        );
        req.fixed_packet_overhead_tokens = 1;
        let result = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(result.total_token_cost, 5);
        verify_proof_budget_result(&req, &result).unwrap();

        req.token_budget = 4;
        let insufficient = solve_proof_budget(req).unwrap();
        assert_eq!(insufficient.status, ProofBudgetStatus::BudgetInsufficient);
        assert!(insufficient.evidence.is_empty());
    }

    #[test]
    fn changed_resolved_excerpt_is_rejected_by_recomputed_span_digest() {
        let mut item = candidate("A", "family-A", 1);
        item.source_span_text = "tampered".into();
        let req = request(
            vec![item],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        assert!(solve_proof_budget(req).unwrap_err().contains("source span byte offsets"));
    }

    #[test]
    fn packet_sufficiency_and_minimum_cost_optimality_are_distinct_checks() {
        let req = request(
            vec![
                candidate("A-cheap", "family-A", 1),
                candidate("B-valid", "family-B", 2),
            ],
            vec![obligation(
                "claim-1",
                vec![
                    justification("proof-cheap", &["A-cheap"]),
                    justification("proof-valid-but-costlier", &["B-valid"]),
                ],
            )],
            5,
        );
        let mut non_minimal = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(non_minimal.selected_proofs[0].justification_id, "proof-cheap");

        let canonical = canonicalize_request(req.clone());
        let candidate = canonical.candidates.iter()
            .find(|item| item.evidence_id == "B-valid").unwrap();
        non_minimal.selected_proofs = vec![SelectedProof {
            conclusion_id: "claim-1".into(),
            justification_id: "proof-valid-but-costlier".into(),
            evidence_dependencies: vec!["B-valid".into()],
            selection_overhead_tokens: 0,
        }];
        non_minimal.evidence = vec![ProofPacketEvidenceRef {
            evidence_id: candidate.evidence_id.clone(),
            source_revision_id: candidate.source_revision_id.clone(),
            source_pointer: candidate.source_pointer.clone(),
            source_span_text: candidate.source_span_text.clone(),
            source_span_start: candidate.source_span_start,
            source_span_end: candidate.source_span_end,
            source_family_id: candidate.source_family_id.clone(),
            source_session_id: candidate.source_session_id.clone(),
            source_speaker: candidate.source_speaker,
            payload_sha256: candidate.payload_sha256.clone(),
            source_span_sha256: candidate.source_span_sha256.clone(),
            authorization_receipt_sha256: candidate.authorization_receipt_sha256.clone(),
            retrieval_rank: candidate.retrieval_rank,
            token_cost: candidate.token_cost,
        }];
        non_minimal.total_token_cost = candidate.token_cost as u64;
        let canonical_request_sha256 = sha256_json(&canonical, "canonical proof budget request").unwrap();
        non_minimal.request_sha256 = canonical_request_sha256.clone();
        let payload = ProofPacketPayload {
            request_sha256: &canonical_request_sha256,
            decision_id: &canonical.decision_id,
            principal_id: &canonical.principal_id,
            scope_id: &canonical.scope_id,
            valid_at_unix_ms: canonical.valid_at_unix_ms,
            known_at_unix_ms: canonical.known_at_unix_ms,
            policy_id: &canonical.policy_id,
            policy_sha256: &canonical.policy_sha256,
            tokenizer_id: &canonical.tokenizer_id,
            tokenizer_version: &canonical.tokenizer_version,
            tokenizer_sha256: &canonical.tokenizer_sha256,
            packet_renderer_sha256: &canonical.packet_renderer_sha256,
            fixed_packet_overhead_tokens: canonical.fixed_packet_overhead_tokens,
            token_budget: canonical.token_budget,
            selected_proofs: &non_minimal.selected_proofs,
            evidence: &non_minimal.evidence,
            total_token_cost: non_minimal.total_token_cost,
        };
        non_minimal.packet_sha256 = Some(sha256_json(&payload, "test proof packet").unwrap());

        verify_proof_packet(&req, &non_minimal).unwrap();
        assert!(verify_proof_budget_result(&req, &non_minimal).is_err());
    }

    #[test]
    fn exact_solver_refuses_to_claim_optimality_after_node_limit() {
        let mut req = request(
            vec![
                candidate("A", "family-A", 1),
                candidate("B", "family-B", 1),
            ],
            vec![obligation(
                "claim-1",
                vec![justification("pA", &["A"]), justification("pB", &["B"])],
            )],
            5,
        );
        req.max_search_nodes = 1;
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::SearchLimitReached);
        assert_eq!(result.packet_sha256, None);
        assert!(result.evidence.is_empty());
    }

    #[test]
    fn exact_solver_matches_exhaustive_oracle_for_small_shared_proof_graph() {
        let req = request(
            vec![
                candidate("A", "family-A", 2),
                candidate("B", "family-B", 3),
                candidate("C", "family-C", 1),
                candidate("D", "family-D", 2),
                candidate("E", "family-E", 1),
            ],
            vec![
                obligation(
                    "claim-1",
                    vec![
                        justification("p1-AB", &["A", "B"]),
                        justification("p1-C", &["C"]),
                    ],
                ),
                obligation(
                    "claim-2",
                    vec![
                        justification("p2-CD", &["C", "D"]),
                        justification("p2-E", &["E"]),
                    ],
                ),
            ],
            20,
        );
        // Exhaustive four-combination oracle:
        // AB+CD=8, AB+E=6, C+CD=3, C+E=2.
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(result.total_token_cost, 2);
        assert_eq!(
            result.selected_proofs.iter().map(|proof| proof.justification_id.as_str()).collect::<Vec<_>>(),
            vec!["p1-C", "p2-E"]
        );
    }

    #[test]
    fn contradictory_candidates_cannot_be_combined_across_obligations() {
        let mut first = candidate("A", "family-A", 1);
        let mut second = candidate("B", "family-B", 1);
        first.conflicts_with.insert("B".into());
        second.conflicts_with.insert("A".into());
        let req = request(
            vec![first, second],
            vec![
                obligation("claim-1", vec![justification("p1", &["A"])]),
                obligation("claim-2", vec![justification("p2", &["B"])]),
            ],
            5,
        );
        let result = solve_proof_budget(req).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.justification_rejections.iter().any(|item| {
            item.reason == JustificationRejectionReason::ContradictoryEvidence
        }));
    }

    #[test]
    fn conflict_graph_must_be_symmetric_and_reference_known_candidates() {
        let mut first = candidate("A", "family-A", 1);
        first.conflicts_with.insert("B".into());
        let req = request(
            vec![first, candidate("B", "family-B", 1)],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        assert!(solve_proof_budget(req).unwrap_err().contains("must be symmetric"));

        let mut first = candidate("A", "family-A", 1);
        first.conflicts_with.insert("missing".into());
        let req = request(
            vec![first],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        assert!(solve_proof_budget(req).unwrap_err().contains("unknown evidence ID"));
    }

    #[test]
    fn packet_toolchain_digests_are_bound_into_request_and_packet() {
        let req = request(
            vec![candidate("A", "family-A", 1)],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        let result = solve_proof_budget(req.clone()).unwrap();
        let mut changed = req;
        changed.tokenizer_sha256 = digest('9');
        assert!(verify_proof_packet(&changed, &result).is_err());
    }

    #[test]
    fn fixed_and_per_obligation_proof_envelope_costs_are_counted() {
        let mut req = request(
            vec![
                candidate("A", "family-A", 1),
                candidate("B", "family-B", 2),
                candidate("C", "family-C", 3),
            ],
            vec![
                obligation("claim-1", vec![justification("p1", &["A"])]),
                obligation("claim-2", vec![justification("p2", &["B", "C"])]),
            ],
            20,
        );
        req.fixed_packet_overhead_tokens = 3;
        req.obligations[0].selection_overhead_tokens = 2;
        req.obligations[1].selection_overhead_tokens = 4;
        let result = solve_proof_budget(req.clone()).unwrap();
        // Evidence 1 + 2 + 3, proof envelopes 2 + 4, fixed header 3.
        assert_eq!(result.total_token_cost, 15);
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        verify_proof_packet(&req, &result).unwrap();

        req.token_budget = 14;
        let insufficient = solve_proof_budget(req).unwrap();
        assert_eq!(insufficient.status, ProofBudgetStatus::BudgetInsufficient);
        assert!(insufficient.evidence.is_empty());
    }
    #[test]
    fn independent_verifier_rejects_dependency_edge_tampering() {
        let req = request(
            vec![candidate("A", "family-A", 1), candidate("B", "family-B", 1)],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        let mut result = solve_proof_budget(req.clone()).unwrap();
        result.selected_proofs[0].evidence_dependencies = vec!["B".into()];
        assert!(verify_proof_packet(&req, &result).unwrap_err().contains("exact conjunctive dependency edges"));
    }

    #[test]
    fn independent_verifier_rejects_source_locator_tampering() {
        let req = request(
            vec![candidate("A", "family-A", 1)],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        let mut result = solve_proof_budget(req.clone()).unwrap();
        result.evidence[0].source_pointer = "/messages/999/content".into();
        assert!(verify_proof_packet(&req, &result).unwrap_err().contains("source digests"));
    }

    #[test]
    fn source_span_offsets_must_match_the_resolved_excerpt_byte_length() {
        let mut item = candidate("A", "family-A", 1);
        item.source_span_end = 11;
        let req = request(
            vec![item],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        assert!(solve_proof_budget(req).unwrap_err().contains("byte offsets"));
    }

    #[test]
    fn exact_solver_matches_cartesian_product_oracle_over_64_small_graphs() {
        for seed in 0u32..64 {
            let mut req = request(
                vec![
                    candidate("A", "family-A", 1),
                    candidate("B", "family-B", 1),
                    candidate("C", "family-C", 1),
                    candidate("D", "family-D", 1),
                ],
                vec![
                    obligation(
                        "claim-a",
                        vec![
                            justification("a-AB", &["A", "B"]),
                            justification("a-C", &["C"]),
                        ],
                    ),
                    obligation(
                        "claim-b",
                        vec![
                            justification("b-BD", &["B", "D"]),
                            justification("b-CA", &["C", "A"]),
                        ],
                    ),
                ],
                100,
            );
            req.fixed_packet_overhead_tokens = (seed % 4) as u64;
            req.obligations[0].selection_overhead_tokens = seed % 3;
            req.obligations[1].selection_overhead_tokens = (seed.wrapping_mul(3) % 5) as u32;
            for (index, item) in req.candidates.iter_mut().enumerate() {
                item.token_cost = ((seed + (index as u32 * 3)) % 7) + 1;
            }

            // Independent Cartesian-product oracle enumerates all four proof
            // combinations, deduplicates shared evidence, and adds renderer
            // header/proof-envelope costs.
            let mut oracle_minimum = u64::MAX;
            for first in &req.obligations[0].alternatives {
                for second in &req.obligations[1].alternatives {
                    let mut evidence_ids = BTreeSet::new();
                    evidence_ids.extend(first.evidence_dependencies.iter().cloned());
                    evidence_ids.extend(second.evidence_dependencies.iter().cloned());
                    let mut cost = req.fixed_packet_overhead_tokens
                        + req.obligations[0].selection_overhead_tokens as u64
                        + req.obligations[1].selection_overhead_tokens as u64;
                    for evidence_id in evidence_ids {
                        let item = req.candidates.iter()
                            .find(|item| item.evidence_id == evidence_id)
                            .expect("test oracle references only defined evidence");
                        cost += item.token_cost as u64;
                    }
                    oracle_minimum = oracle_minimum.min(cost);
                }
            }

            let result = solve_proof_budget(req.clone()).unwrap();
            assert_eq!(result.status, ProofBudgetStatus::Supported, "seed={seed}");
            assert_eq!(result.total_token_cost, oracle_minimum, "seed={seed}");
            verify_proof_budget_result(&req, &result)
                .unwrap_or_else(|error| panic!("seed={seed}: {error}"));
        }
    }

    #[test]
    fn invalid_source_digest_is_rejected_before_search() {
        let mut item = candidate("A", "family-A", 1);
        item.payload_sha256 = "bad".into();
        let req = request(
            vec![item],
            vec![obligation("claim-1", vec![justification("p1", &["A"])])],
            5,
        );
        assert!(solve_proof_budget(req).unwrap_err().contains("payload_sha256"));
    }
}
