// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic lifecycle-aware action controller for evidence-grounded memory.
//!
//! Retrieval and proof selection do not, by themselves, decide whether a user
//! should receive an answer. This module makes that final action explicit:
//! ANSWER, WARN_STALE, EXPLAIN_CONFLICT, RETRIEVE_MORE, or ABSTAIN.
//!
//! The policy is intentionally deterministic and auditable. It does not call a
//! language model, infer facts from prose, or construct answer text. An upstream
//! orchestrator must populate one claim-ledger entry per required conclusion
//! from trusted bitemporal, authorization, contradiction and proof-verifier
//! outputs. In particular, only the trusted caller may set proof_status to
//! VerifiedOptimal after invoking the independent proof-packet verifier.
//!
//! Policy precedence:
//! 1. ANSWER only if every required claim has current support and the complete
//!    proof packet is independently verified optimal.
//! 2. If required support is unresolved and more retrieval is explicitly
//!    available, request RETRIEVE_MORE before making a terminal decision.
//! 3. Untrusted / provenance-invalid evidence causes ABSTAIN.
//! 4. Unresolved current contradictions cause EXPLAIN_CONFLICT.
//! 5. Stale-only support causes WARN_STALE.
//! 6. Missing support or an unverified proof causes ABSTAIN.
//!
//! The returned digest commits to the request binding, proof status, retrieval
//! progress, selected action and claim-level lifecycle ledger. It is an
//! integrity receipt, not a signature or cryptographic proof of upstream
//! authorization / verifier execution.

use crate::memory_proof_budget::{ProofBudgetRequest, ProofBudgetResult};
use crate::memory_verified_reader_context::build_verified_proof_reader_context;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

/// The terminal or continuation action selected for this memory query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleAction {
    Answer,
    WarnStale,
    ExplainConflict,
    RetrieveMore,
    Abstain,
}

/// Evidence state for one required conclusion at the request's (valid_at, known_at).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimLifecycleState {
    /// At least one selected, current, authorized proof supports the claim.
    SupportedCurrent,
    /// Only expired, superseded, or otherwise stale evidence supports the claim.
    StaleOnly,
    /// Two or more current evidence records remain in unresolved contradiction.
    Conflicted,
    /// No admissible evidence is available for the required conclusion.
    Missing,
    /// Candidate evidence was rejected by the trusted authorization check.
    Unauthorized,
    /// Provenance, source-span, scope, or integrity verification failed.
    ProvenanceInvalid,
}

/// State of the independent proof-packet verification step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofVerificationState {
    /// The exact proof packet passed the independent sufficiency and optimality verifier.
    VerifiedOptimal,
    /// The proof solver/verifier has not been run to completion.
    NotRun,
    /// The configured exact-search limit prevented an optimality conclusion.
    SearchLimitReached,
    /// No eligible admissible proof was found in the current candidate set.
    NoEligibleProof,
    /// Complete search found no sufficient packet within the declared token budget.
    BudgetInsufficient,
    /// Packet verification failed, or the supplied packet was rejected.
    Rejected,
}

/// Machine-readable rationale for the selected lifecycle action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleActionReason {
    AllRequiredClaimsHaveVerifiedCurrentSupport,
    MoreRetrievalMayResolveIncompleteSupport,
    UntrustedEvidenceCannotSupportTheDecision,
    CurrentEvidenceConflictRemainsUnresolved,
    OnlyStaleEvidenceCanSupportTheClaim,
    RequiredEvidenceIsMissing,
    ProofPacketWasNotIndependentlyVerified,
}

/// Opaque proof-verification capability. Its fields are private and the only
/// public constructor runs the independent proof-packet verifier through the
/// verified-reader adapter. Do not deserialize or fabricate this value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedLifecycleProofToken {
    decision_id: String,
    principal_id: String,
    scope_id: String,
    valid_at_unix_ms: u64,
    known_at_unix_ms: u64,
    request_sha256: String,
    packet_sha256: String,
    selected_proofs: std::collections::BTreeMap<
        String,
        (String, BTreeSet<String>, BTreeSet<String>),
    >,
}

impl VerifiedLifecycleProofToken {
    fn verify_request_binding(&self, request: &LifecyclePolicyRequest) -> Result<(), String> {
        if self.decision_id != request.decision_id
            || self.principal_id != request.principal_id
            || self.scope_id != request.scope_id
            || self.valid_at_unix_ms != request.valid_at_unix_ms
            || self.known_at_unix_ms != request.known_at_unix_ms
        {
            return Err("verified proof token does not match lifecycle request identity or time".into());
        }
        if request.proof_packet_sha256.as_deref() != Some(self.packet_sha256.as_str()) {
            return Err("verified proof token packet digest does not match lifecycle request".into());
        }
        self.verify_claim_bindings(&request.required_claims)
    }

    fn verify_decision_binding(&self, decision: &LifecycleActionDecision) -> Result<(), String> {
        if self.decision_id != decision.decision_id
            || self.principal_id != decision.principal_id
            || self.scope_id != decision.scope_id
            || self.valid_at_unix_ms != decision.valid_at_unix_ms
            || self.known_at_unix_ms != decision.known_at_unix_ms
        {
            return Err("verified proof token does not match the final decision identity or time".into());
        }
        if decision.proof_request_sha256.as_deref() != Some(self.request_sha256.as_str())
            || decision.proof_packet_sha256.as_deref() != Some(self.packet_sha256.as_str())
        {
            return Err("verified proof token digest does not match the final decision".into());
        }
        self.verify_claim_bindings(&decision.claim_ledger)
    }

    fn verify_claim_bindings(
        &self,
        claims: &[ClaimLifecycleLedgerEntry],
    ) -> Result<(), String> {
        validate_sha256(&self.request_sha256, "verified proof request digest")?;
        validate_sha256(&self.packet_sha256, "verified proof packet digest")?;
        if self.selected_proofs.len() != claims.len() {
            return Err("verified proof token does not cover every required lifecycle claim".into());
        }
        for claim in claims {
            if claim.state != ClaimLifecycleState::SupportedCurrent {
                return Err("verified proof token cannot be used with a non-current claim".into());
            }
            let (justification_id, dependencies, session_ids) = self
                .selected_proofs
                .get(&claim.conclusion_id)
                .ok_or_else(|| format!(
                    "verified proof token is missing required conclusion {:?}",
                    claim.conclusion_id
                ))?;
            if claim.selected_justification_id.as_deref() != Some(justification_id.as_str())
                || &claim.current_evidence_ids != dependencies
                || &claim.source_session_ids != session_ids
            {
                return Err(format!(
                    "lifecycle claim {:?} does not match the independently verified selected proof or session coverage",
                    claim.conclusion_id
                ));
            }
        }
        Ok(())
    }
}

/// Create an opaque lifecycle proof token only after the canonical independent
/// verifier has accepted the packet and its exact reader context.
pub fn verify_proof_packet_for_lifecycle(
    request: &ProofBudgetRequest,
    result: &ProofBudgetResult,
) -> Result<VerifiedLifecycleProofToken, String> {
    let context = build_verified_proof_reader_context(request, result)?;
    context.verify_internal_consistency()?;
    let session_by_evidence_id: std::collections::BTreeMap<String, Option<String>> =
        context.evidence
            .iter()
            .map(|evidence| (evidence.evidence_id.clone(), evidence.source_session_id.clone()))
            .collect();
    let selected_proofs = context
        .claim_proofs
        .iter()
        .map(|proof| {
            let dependencies: BTreeSet<String> =
                proof.evidence_dependencies.iter().cloned().collect();
            let session_ids: BTreeSet<String> = proof.evidence_dependencies
                .iter()
                .filter_map(|evidence_id| {
                    session_by_evidence_id
                        .get(evidence_id)
                        .cloned()
                        .flatten()
                })
                .collect();
            (
                proof.conclusion_id.clone(),
                (
                    proof.justification_id.clone(),
                    dependencies,
                    session_ids,
                ),
            )
        })
        .collect();
    Ok(VerifiedLifecycleProofToken {
        decision_id: context.decision_id,
        principal_id: context.principal_id,
        scope_id: context.scope_id,
        valid_at_unix_ms: context.valid_at_unix_ms,
        known_at_unix_ms: context.known_at_unix_ms,
        request_sha256: context.request_sha256,
        packet_sha256: context.packet_sha256,
        selected_proofs,
    })
}

/// Claim-level provenance/lifecycle record. It contains IDs and states only;
/// source text is deliberately kept outside this control-plane structure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimLifecycleLedgerEntry {
    pub conclusion_id: String,
    pub state: ClaimLifecycleState,
    pub selected_justification_id: Option<String>,
    pub current_evidence_ids: BTreeSet<String>,
    pub stale_evidence_ids: BTreeSet<String>,
    pub conflicting_evidence_ids: BTreeSet<String>,
    pub rejected_evidence_ids: BTreeSet<String>,
    /// Exact distinct source-session IDs supporting the selected proof.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub source_session_ids: BTreeSet<String>,
}

impl ClaimLifecycleLedgerEntry {
    pub fn validate(&self) -> Result<(), String> {
        require_non_empty(&self.conclusion_id, "conclusion_id")?;
        if let Some(justification_id) = &self.selected_justification_id {
            require_non_empty(justification_id, "selected_justification_id")?;
        }

        let mut all_ids = BTreeSet::new();
        for (field, evidence_ids) in [
            ("current_evidence_ids", &self.current_evidence_ids),
            ("stale_evidence_ids", &self.stale_evidence_ids),
            ("conflicting_evidence_ids", &self.conflicting_evidence_ids),
            ("rejected_evidence_ids", &self.rejected_evidence_ids),
        ] {
            for evidence_id in evidence_ids {
                require_non_empty(evidence_id, field)?;
                if !all_ids.insert(evidence_id.as_str()) {
                    return Err(format!(
                        "claim {:?} repeats evidence ID {:?} across lifecycle sets",
                        self.conclusion_id, evidence_id
                    ));
                }
            }
        }

        for session_id in &self.source_session_ids {
            require_non_empty(session_id, "source_session_id")?;
        }

        match self.state {
            ClaimLifecycleState::SupportedCurrent => {
                if self.selected_justification_id.is_none()
                    || self.current_evidence_ids.is_empty()
                    || !self.stale_evidence_ids.is_empty()
                    || !self.conflicting_evidence_ids.is_empty()
                    || !self.rejected_evidence_ids.is_empty()
                {
                    return Err(format!(
                        "supported-current claim {:?} requires one selected justification and current evidence only",
                        self.conclusion_id
                    ));
                }
            }
            ClaimLifecycleState::StaleOnly => {
                if self.selected_justification_id.is_some()
                    || !self.current_evidence_ids.is_empty()
                    || self.stale_evidence_ids.is_empty()
                    || !self.conflicting_evidence_ids.is_empty()
                    || !self.rejected_evidence_ids.is_empty()
                {
                    return Err(format!(
                        "stale-only claim {:?} must contain stale evidence and no current proof",
                        self.conclusion_id
                    ));
                }
            }
            ClaimLifecycleState::Conflicted => {
                if self.selected_justification_id.is_some()
                    || !self.current_evidence_ids.is_empty()
                    || !self.stale_evidence_ids.is_empty()
                    || self.conflicting_evidence_ids.len() < 2
                    || !self.rejected_evidence_ids.is_empty()
                {
                    return Err(format!(
                        "conflicted claim {:?} requires at least two conflicting evidence IDs and no selected proof",
                        self.conclusion_id
                    ));
                }
            }
            ClaimLifecycleState::Missing => {
                if self.selected_justification_id.is_some() || !all_ids.is_empty() {
                    return Err(format!(
                        "missing claim {:?} must not carry evidence or a selected proof",
                        self.conclusion_id
                    ));
                }
            }
            ClaimLifecycleState::Unauthorized | ClaimLifecycleState::ProvenanceInvalid => {
                if self.selected_justification_id.is_some()
                    || !self.current_evidence_ids.is_empty()
                    || !self.stale_evidence_ids.is_empty()
                    || !self.conflicting_evidence_ids.is_empty()
                    || self.rejected_evidence_ids.is_empty()
                {
                    return Err(format!(
                        "untrusted claim {:?} must place rejected evidence IDs only in rejected_evidence_ids",
                        self.conclusion_id
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Whether retrieval is complete or a bounded continuation is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LifecycleRetrievalProgress {
    pub complete: bool,
    pub can_continue: bool,
    pub remaining_calls: u32,
}

impl LifecycleRetrievalProgress {
    pub fn validate(&self) -> Result<(), String> {
        if self.complete && self.can_continue {
            return Err("complete retrieval cannot also advertise continuation".into());
        }
        if self.complete && self.remaining_calls != 0 {
            return Err("complete retrieval must have zero remaining calls".into());
        }
        if self.can_continue && self.remaining_calls == 0 {
            return Err("retrieval continuation requires a positive remaining-call budget".into());
        }
        Ok(())
    }
}

/// Complete input to the deterministic action policy.
///
/// The claim ledger must contain every required conclusion exactly once. IDs
/// are canonicalized before the digest is produced, so equivalent input
/// permutations generate the same action receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LifecyclePolicyRequest {
    pub decision_id: String,
    pub principal_id: String,
    pub scope_id: String,
    pub valid_at_unix_ms: u64,
    pub known_at_unix_ms: u64,
    pub proof_status: ProofVerificationState,
    pub proof_packet_sha256: Option<String>,
    pub retrieval: LifecycleRetrievalProgress,
    pub required_claims: Vec<ClaimLifecycleLedgerEntry>,
}

impl LifecyclePolicyRequest {
    pub fn validate(&self) -> Result<(), String> {
        require_non_empty(&self.decision_id, "decision_id")?;
        require_non_empty(&self.principal_id, "principal_id")?;
        require_non_empty(&self.scope_id, "scope_id")?;
        self.retrieval.validate()?;
        if self.required_claims.is_empty() {
            return Err("required_claims must not be empty".into());
        }

        let mut conclusion_ids = BTreeSet::new();
        for claim in &self.required_claims {
            claim.validate()?;
            if !conclusion_ids.insert(claim.conclusion_id.as_str()) {
                return Err(format!(
                    "duplicate required conclusion {:?}",
                    claim.conclusion_id
                ));
            }
        }

        if let Some(packet_sha256) = &self.proof_packet_sha256 {
            validate_sha256(packet_sha256, "proof_packet_sha256")?;
        }
        if self.proof_status == ProofVerificationState::VerifiedOptimal {
            if self.proof_packet_sha256.is_none() {
                return Err("VerifiedOptimal requires proof_packet_sha256".into());
            }
            if self.required_claims
                .iter()
                .any(|claim| claim.state != ClaimLifecycleState::SupportedCurrent)
            {
                return Err(
                    "VerifiedOptimal is inconsistent with a non-supported required claim".into(),
                );
            }
        } else if self.proof_packet_sha256.is_some() {
            return Err("unverified proof status must not carry proof_packet_sha256".into());
        }
        Ok(())
    }
}

/// Auditable action decision bound to one principal, scope and temporal query.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LifecycleActionDecision {
    pub policy_version: String,
    pub decision_id: String,
    pub principal_id: String,
    pub scope_id: String,
    pub valid_at_unix_ms: u64,
    pub known_at_unix_ms: u64,
    pub action: LifecycleAction,
    pub reason: LifecycleActionReason,
    pub proof_status: ProofVerificationState,
    pub proof_request_sha256: Option<String>,
    pub proof_packet_sha256: Option<String>,
    pub retrieval: LifecycleRetrievalProgress,
    pub claim_ledger: Vec<ClaimLifecycleLedgerEntry>,
    pub action_sha256: String,
}

#[derive(Debug, Serialize)]
struct LifecycleActionDigestPayload<'a> {
    policy_version: &'a str,
    decision_id: &'a str,
    principal_id: &'a str,
    scope_id: &'a str,
    valid_at_unix_ms: u64,
    known_at_unix_ms: u64,
    action: LifecycleAction,
    reason: LifecycleActionReason,
    proof_status: ProofVerificationState,
    proof_request_sha256: &'a Option<String>,
    proof_packet_sha256: &'a Option<String>,
    retrieval: LifecycleRetrievalProgress,
    claim_ledger: &'a [ClaimLifecycleLedgerEntry],
}

impl LifecycleActionDecision {
    /// Recompute the deterministic action digest and validate the full ledger.
    pub fn verify_internal_consistency(&self) -> Result<(), String> {
        if self.policy_version != LIFECYCLE_POLICY_VERSION {
            return Err("unsupported lifecycle policy version".into());
        }
        require_non_empty(&self.decision_id, "decision_id")?;
        require_non_empty(&self.principal_id, "principal_id")?;
        require_non_empty(&self.scope_id, "scope_id")?;
        self.retrieval.validate()?;

        let mut previous: Option<&str> = None;
        let mut seen = BTreeSet::new();
        for claim in &self.claim_ledger {
            claim.validate()?;
            if previous.is_some_and(|prior| prior >= claim.conclusion_id.as_str()) {
                return Err("claim ledger must be in canonical ascending conclusion order".into());
            }
            previous = Some(claim.conclusion_id.as_str());
            if !seen.insert(claim.conclusion_id.as_str()) {
                return Err("claim ledger contains duplicate conclusions".into());
            }
        }
        if self.claim_ledger.is_empty() {
            return Err("claim ledger must not be empty".into());
        }
        if let Some(packet_sha256) = &self.proof_packet_sha256 {
            validate_sha256(packet_sha256, "proof_packet_sha256")?;
        }
        if let Some(request_sha256) = &self.proof_request_sha256 {
            validate_sha256(request_sha256, "proof_request_sha256")?;
        }
        if self.proof_status == ProofVerificationState::VerifiedOptimal {
            if self.proof_packet_sha256.is_none()
                || self.proof_request_sha256.is_none()
                || self.claim_ledger
                    .iter()
                    .any(|claim| claim.state != ClaimLifecycleState::SupportedCurrent)
            {
                return Err("VerifiedOptimal decision has inconsistent packet or claim state".into());
            }
        } else if self.proof_packet_sha256.is_some() || self.proof_request_sha256.is_some() {
            return Err("unverified decision must not carry a verified proof token digest".into());
        }
        let (expected_action, expected_reason) = choose_action(
            self.proof_status,
            self.retrieval,
            &self.claim_ledger,
        );
        if self.action != expected_action || self.reason != expected_reason {
            return Err("lifecycle action or reason does not match the deterministic policy".into());
        }
        let expected = lifecycle_action_digest(
            &self.policy_version,
            &self.decision_id,
            &self.principal_id,
            &self.scope_id,
            self.valid_at_unix_ms,
            self.known_at_unix_ms,
            self.action,
            self.reason,
            self.proof_status,
            &self.proof_request_sha256,
            &self.proof_packet_sha256,
            self.retrieval,
            &self.claim_ledger,
        )?;
        if expected != self.action_sha256 {
            return Err("lifecycle action SHA-256 mismatch".into());
        }
        Ok(())
    }

    /// Prevent accidental reuse across principals, scopes or time coordinates.
    pub fn verify_request_binding(
        &self,
        principal_id: &str,
        scope_id: &str,
        valid_at_unix_ms: u64,
        known_at_unix_ms: u64,
    ) -> Result<(), String> {
        if self.principal_id != principal_id {
            return Err("lifecycle decision principal binding mismatch".into());
        }
        if self.scope_id != scope_id {
            return Err("lifecycle decision scope binding mismatch".into());
        }
        if self.valid_at_unix_ms != valid_at_unix_ms {
            return Err("lifecycle decision valid_at binding mismatch".into());
        }
        if self.known_at_unix_ms != known_at_unix_ms {
            return Err("lifecycle decision known_at binding mismatch".into());
        }
        Ok(())
    }

    /// A generation loop may emit an answer only while it retains and presents
    /// the opaque proof token created by verify_proof_packet_for_lifecycle.
    /// A serialized decision receipt alone is never authorization to answer.
    pub fn answer_is_permitted(
        &self,
        verified_proof: Option<&VerifiedLifecycleProofToken>,
    ) -> Result<bool, String> {
        self.verify_internal_consistency()?;
        if self.action != LifecycleAction::Answer
            || self.proof_status != ProofVerificationState::VerifiedOptimal
        {
            return Ok(false);
        }
        let token = verified_proof.ok_or_else(|| {
            "ANSWER requires the original opaque proof-verifier token at the final response gate".to_string()
        })?;
        token.verify_decision_binding(self)?;
        Ok(true)
    }
}

pub const LIFECYCLE_POLICY_VERSION: &str = "BROCA-LIFECYCLE-ACTION-V2";

/// Select an action from the complete claim lifecycle ledger.
///
/// This is a deterministic policy, not an LLM response. A caller may render
/// WARN_STALE or EXPLAIN_CONFLICT using only the IDs and state records in the
/// returned decision. It must never convert those actions into ANSWER.
pub fn decide_lifecycle_action(
    mut request: LifecyclePolicyRequest,
    verified_proof: Option<&VerifiedLifecycleProofToken>,
) -> Result<LifecycleActionDecision, String> {
    request.validate()?;
    match (request.proof_status, verified_proof) {
        (ProofVerificationState::VerifiedOptimal, Some(token)) => {
            token.verify_request_binding(&request)?;
        }
        (ProofVerificationState::VerifiedOptimal, None) => {
            return Err("VerifiedOptimal requires an opaque token from verify_proof_packet_for_lifecycle".into());
        }
        (_, Some(_)) => {
            return Err("a verified proof token cannot accompany a non-VerifiedOptimal status".into());
        }
        (_, None) => {}
    }
    request.required_claims
        .sort_by(|left, right| left.conclusion_id.cmp(&right.conclusion_id));

    let (action, reason) = choose_action(
        request.proof_status,
        request.retrieval,
        &request.required_claims,
    );

    let policy_version = LIFECYCLE_POLICY_VERSION.to_string();
    let claim_ledger = request.required_claims;
    let action_sha256 = lifecycle_action_digest(
        &policy_version,
        &request.decision_id,
        &request.principal_id,
        &request.scope_id,
        request.valid_at_unix_ms,
        request.known_at_unix_ms,
        action,
        reason,
        request.proof_status,
        &verified_proof.map(|token| token.request_sha256.clone()),
        &request.proof_packet_sha256,
        request.retrieval,
        &claim_ledger,
    )?;

    let decision = LifecycleActionDecision {
        policy_version,
        decision_id: request.decision_id,
        principal_id: request.principal_id,
        scope_id: request.scope_id,
        valid_at_unix_ms: request.valid_at_unix_ms,
        known_at_unix_ms: request.known_at_unix_ms,
        action,
        reason,
        proof_status: request.proof_status,
        proof_request_sha256: verified_proof.map(|token| token.request_sha256.clone()),
        proof_packet_sha256: request.proof_packet_sha256,
        retrieval: request.retrieval,
        claim_ledger,
        action_sha256,
    };
    decision.verify_internal_consistency()?;
    Ok(decision)
}

fn choose_action(
    proof_status: ProofVerificationState,
    retrieval: LifecycleRetrievalProgress,
    claims: &[ClaimLifecycleLedgerEntry],
) -> (LifecycleAction, LifecycleActionReason) {
    let all_supported = claims
        .iter()
        .all(|claim| claim.state == ClaimLifecycleState::SupportedCurrent);
    if all_supported && proof_status == ProofVerificationState::VerifiedOptimal {
        return (
            LifecycleAction::Answer,
            LifecycleActionReason::AllRequiredClaimsHaveVerifiedCurrentSupport,
        );
    }
    if retrieval.can_continue {
        return (
            LifecycleAction::RetrieveMore,
            LifecycleActionReason::MoreRetrievalMayResolveIncompleteSupport,
        );
    }
    if claims.iter().any(|claim| {
        matches!(
            claim.state,
            ClaimLifecycleState::Unauthorized | ClaimLifecycleState::ProvenanceInvalid
        )
    }) {
        return (
            LifecycleAction::Abstain,
            LifecycleActionReason::UntrustedEvidenceCannotSupportTheDecision,
        );
    }
    if claims.iter().any(|claim| claim.state == ClaimLifecycleState::Conflicted) {
        return (
            LifecycleAction::ExplainConflict,
            LifecycleActionReason::CurrentEvidenceConflictRemainsUnresolved,
        );
    }
    if claims.iter().any(|claim| claim.state == ClaimLifecycleState::StaleOnly) {
        return (
            LifecycleAction::WarnStale,
            LifecycleActionReason::OnlyStaleEvidenceCanSupportTheClaim,
        );
    }
    if claims.iter().any(|claim| claim.state == ClaimLifecycleState::Missing) {
        return (
            LifecycleAction::Abstain,
            LifecycleActionReason::RequiredEvidenceIsMissing,
        );
    }
    (
        LifecycleAction::Abstain,
        LifecycleActionReason::ProofPacketWasNotIndependentlyVerified,
    )
}

fn lifecycle_action_digest(
    policy_version: &str,
    decision_id: &str,
    principal_id: &str,
    scope_id: &str,
    valid_at_unix_ms: u64,
    known_at_unix_ms: u64,
    action: LifecycleAction,
    reason: LifecycleActionReason,
    proof_status: ProofVerificationState,
    proof_request_sha256: &Option<String>,
    proof_packet_sha256: &Option<String>,
    retrieval: LifecycleRetrievalProgress,
    claim_ledger: &[ClaimLifecycleLedgerEntry],
) -> Result<String, String> {
    let payload = LifecycleActionDigestPayload {
        policy_version,
        decision_id,
        principal_id,
        scope_id,
        valid_at_unix_ms,
        known_at_unix_ms,
        action,
        reason,
        proof_status,
        proof_request_sha256,
        proof_packet_sha256,
        retrieval,
        claim_ledger,
    };
    let bytes = serde_json::to_vec(&payload)
        .map_err(|error| format!("cannot serialize lifecycle action receipt: {error}"))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn require_non_empty(value: &str, field: &str) -> Result<(), String> {
    if value.trim().is_empty() || value.trim() != value {
        return Err(format!("{field} must be non-empty and trimmed"));
    }
    Ok(())
}

fn validate_sha256(value: &str, field: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
        })
    {
        return Err(format!("{field} must be a 64-character lowercase SHA-256 digest"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_proof_budget::{
        ProofEvidenceCandidate, ProofJustification, ProofObligation, solve_proof_budget,
    };

    fn digest(value: &str) -> String {
        format!("{:x}", Sha256::digest(value.as_bytes()))
    }

    fn supported_claim(id: &str) -> ClaimLifecycleLedgerEntry {
        ClaimLifecycleLedgerEntry {
            conclusion_id: id.into(),
            state: ClaimLifecycleState::SupportedCurrent,
            selected_justification_id: Some(format!("proof-{id}")),
            current_evidence_ids: BTreeSet::from([format!("evidence-{id}")]),
            stale_evidence_ids: BTreeSet::new(),
            conflicting_evidence_ids: BTreeSet::new(),
            rejected_evidence_ids: BTreeSet::new(),
            source_session_ids: BTreeSet::new(),
        }
    }

    fn stale_claim(id: &str) -> ClaimLifecycleLedgerEntry {
        ClaimLifecycleLedgerEntry {
            conclusion_id: id.into(),
            state: ClaimLifecycleState::StaleOnly,
            selected_justification_id: None,
            current_evidence_ids: BTreeSet::new(),
            stale_evidence_ids: BTreeSet::from([format!("stale-{id}")]),
            conflicting_evidence_ids: BTreeSet::new(),
            rejected_evidence_ids: BTreeSet::new(),
            source_session_ids: BTreeSet::new(),
        }
    }

    fn conflict_claim(id: &str) -> ClaimLifecycleLedgerEntry {
        ClaimLifecycleLedgerEntry {
            conclusion_id: id.into(),
            state: ClaimLifecycleState::Conflicted,
            selected_justification_id: None,
            current_evidence_ids: BTreeSet::new(),
            stale_evidence_ids: BTreeSet::new(),
            conflicting_evidence_ids: BTreeSet::from([
                format!("conflict-{id}-a"),
                format!("conflict-{id}-b"),
            ]),
            rejected_evidence_ids: BTreeSet::new(),
            source_session_ids: BTreeSet::new(),
        }
    }

    fn missing_claim(id: &str) -> ClaimLifecycleLedgerEntry {
        ClaimLifecycleLedgerEntry {
            conclusion_id: id.into(),
            state: ClaimLifecycleState::Missing,
            selected_justification_id: None,
            current_evidence_ids: BTreeSet::new(),
            stale_evidence_ids: BTreeSet::new(),
            conflicting_evidence_ids: BTreeSet::new(),
            rejected_evidence_ids: BTreeSet::new(),
            source_session_ids: BTreeSet::new(),
        }
    }

    fn rejected_claim(id: &str, state: ClaimLifecycleState) -> ClaimLifecycleLedgerEntry {
        ClaimLifecycleLedgerEntry {
            conclusion_id: id.into(),
            state,
            selected_justification_id: None,
            current_evidence_ids: BTreeSet::new(),
            stale_evidence_ids: BTreeSet::new(),
            conflicting_evidence_ids: BTreeSet::new(),
            rejected_evidence_ids: BTreeSet::from([format!("rejected-{id}")]),
            source_session_ids: BTreeSet::new(),
        }
    }

    fn progress(complete: bool, can_continue: bool) -> LifecycleRetrievalProgress {
        LifecycleRetrievalProgress {
            complete,
            can_continue,
            remaining_calls: if can_continue { 2 } else { 0 },
        }
    }

    fn token_for(request: &LifecyclePolicyRequest) -> VerifiedLifecycleProofToken {
        let selected_proofs = request.required_claims
            .iter()
            .map(|claim| {
                (
                    claim.conclusion_id.clone(),
                    (
                        claim.selected_justification_id.clone().unwrap(),
                        claim.current_evidence_ids.clone(),
                        claim.source_session_ids.clone(),
                    ),
                )
            })
            .collect();
        VerifiedLifecycleProofToken {
            decision_id: request.decision_id.clone(),
            principal_id: request.principal_id.clone(),
            scope_id: request.scope_id.clone(),
            valid_at_unix_ms: request.valid_at_unix_ms,
            known_at_unix_ms: request.known_at_unix_ms,
            request_sha256: digest("verified request"),
            packet_sha256: request.proof_packet_sha256.clone().unwrap(),
            selected_proofs,
        }
    }

    fn decide_with_test_token(
        request: LifecyclePolicyRequest,
    ) -> Result<LifecycleActionDecision, String> {
        let token = if request.proof_status == ProofVerificationState::VerifiedOptimal {
            Some(token_for(&request))
        } else {
            None
        };
        decide_lifecycle_action(request, token.as_ref())
    }

    fn request(
        proof_status: ProofVerificationState,
        claims: Vec<ClaimLifecycleLedgerEntry>,
        retrieval: LifecycleRetrievalProgress,
    ) -> LifecyclePolicyRequest {
        LifecyclePolicyRequest {
            decision_id: "decision-1".into(),
            principal_id: "principal-1".into(),
            scope_id: "scope-1".into(),
            valid_at_unix_ms: 200,
            known_at_unix_ms: 300,
            proof_status,
            proof_packet_sha256: if proof_status == ProofVerificationState::VerifiedOptimal {
                Some(digest("packet"))
            } else {
                None
            },
            retrieval,
            required_claims: claims,
        }
    }

    fn real_proof_request() -> ProofBudgetRequest {
        let excerpt_a = "verified evidence from the first session";
        let excerpt_b = "independent confirmation from the second session";
        ProofBudgetRequest {
            decision_id: "decision-real".into(),
            principal_id: "principal-1".into(),
            scope_id: "scope-1".into(),
            valid_at_unix_ms: 200,
            known_at_unix_ms: 300,
            policy_id: "policy-v1".into(),
            policy_sha256: digest("policy"),
            fixed_packet_overhead_tokens: 2,
            token_budget: 64,
            max_search_nodes: 1000,
            tokenizer_id: "test-tokenizer".into(),
            tokenizer_version: "1".into(),
            tokenizer_sha256: digest("tokenizer"),
            packet_renderer_sha256: digest("renderer"),
            candidates: vec![
                ProofEvidenceCandidate {
                    evidence_id: "E1".into(),
                    source_revision_id: "REV-E1".into(),
                    source_pointer: "/messages/0/content".into(),
                    source_span_text: excerpt_a.into(),
                    source_span_start: 0,
                    source_span_end: excerpt_a.len() as u64,
                    source_family_id: "family-1".into(),
                    source_session_id: Some("session-1".into()),
                    source_speaker: Some(crate::memory_evidence::EvidenceSpeakerRole::User),
                    scope_id: "scope-1".into(),
                    payload_sha256: digest("payload-1"),
                    source_span_sha256: digest(excerpt_a),
                    resolved_span_sha256: digest(excerpt_a),
                    conflicts_with: BTreeSet::new(),
                    recorded_at_unix_ms: 100,
                    retracted_at_unix_ms: None,
                    valid_from_unix_ms: Some(10),
                    valid_until_unix_ms: None,
                    authorized_principal_id: "principal-1".into(),
                    authorization_accepted: true,
                    authorization_receipt_sha256: digest("authorization receipt-1"),
                    retrieval_rank: 1,
                    token_cost: 3,
                },
                ProofEvidenceCandidate {
                    evidence_id: "E2".into(),
                    source_revision_id: "REV-E2".into(),
                    source_pointer: "/messages/1/content".into(),
                    source_span_text: excerpt_b.into(),
                    source_span_start: 0,
                    source_span_end: excerpt_b.len() as u64,
                    source_family_id: "family-2".into(),
                    source_session_id: Some("session-2".into()),
                    source_speaker: Some(crate::memory_evidence::EvidenceSpeakerRole::Assistant),
                    scope_id: "scope-1".into(),
                    payload_sha256: digest("payload-2"),
                    source_span_sha256: digest(excerpt_b),
                    resolved_span_sha256: digest(excerpt_b),
                    conflicts_with: BTreeSet::new(),
                    recorded_at_unix_ms: 150,
                    retracted_at_unix_ms: None,
                    valid_from_unix_ms: Some(10),
                    valid_until_unix_ms: None,
                    authorized_principal_id: "principal-1".into(),
                    authorization_accepted: true,
                    authorization_receipt_sha256: digest("authorization receipt-2"),
                    retrieval_rank: 2,
                    token_cost: 4,
                },
            ],
            obligations: vec![ProofObligation {
                conclusion_id: "C1".into(),
                alternatives: vec![ProofJustification {
                    justification_id: "proof-C1".into(),
                    evidence_dependencies: vec!["E1".into(), "E2".into()],
                }],
                minimum_independent_families: 2,
                minimum_distinct_sessions: 2,
                minimum_user_authored_evidence: 1,
                required_complete_evidence_ids: BTreeSet::new(),
                enumeration_receipt_sha256: None,
                selection_overhead_tokens: 1,
            }],
        }
    }

    #[test]
    fn proof_token_cannot_be_reused_for_a_different_principal_scope_or_time() {
        let proof_request = real_proof_request();
        let result = solve_proof_budget(proof_request.clone()).unwrap();
        let token = verify_proof_packet_for_lifecycle(&proof_request, &result).unwrap();

        let mut lifecycle_request = LifecyclePolicyRequest {
            decision_id: proof_request.decision_id.clone(),
            principal_id: proof_request.principal_id.clone(),
            scope_id: proof_request.scope_id.clone(),
            valid_at_unix_ms: proof_request.valid_at_unix_ms,
            known_at_unix_ms: proof_request.known_at_unix_ms,
            proof_status: ProofVerificationState::VerifiedOptimal,
            proof_packet_sha256: result.packet_sha256.clone(),
            retrieval: progress(true, false),
            required_claims: vec![ClaimLifecycleLedgerEntry {
                conclusion_id: "C1".into(),
                state: ClaimLifecycleState::SupportedCurrent,
                selected_justification_id: Some("proof-C1".into()),
                current_evidence_ids: BTreeSet::from(["E1".into(), "E2".into()]),
                stale_evidence_ids: BTreeSet::new(),
                conflicting_evidence_ids: BTreeSet::new(),
                rejected_evidence_ids: BTreeSet::new(),
                source_session_ids: BTreeSet::from(["session-1".into(), "session-2".into()]),
            }],
        };
        lifecycle_request.principal_id = "other-principal".into();
        assert!(decide_lifecycle_action(lifecycle_request.clone(), Some(&token))
            .unwrap_err()
            .contains("identity or time"));
        lifecycle_request.principal_id = "principal-1".into();
        lifecycle_request.known_at_unix_ms += 1;
        assert!(decide_lifecycle_action(lifecycle_request, Some(&token))
            .unwrap_err()
            .contains("identity or time"));
    }

    #[test]
    fn verified_proof_token_is_created_only_from_the_independent_packet_verifier() {
        let proof_request = real_proof_request();
        let result = solve_proof_budget(proof_request.clone()).unwrap();
        assert_eq!(result.status, crate::memory_proof_budget::ProofBudgetStatus::Supported);
        let token = verify_proof_packet_for_lifecycle(&proof_request, &result).unwrap();

        let lifecycle_request = LifecyclePolicyRequest {
            decision_id: proof_request.decision_id.clone(),
            principal_id: proof_request.principal_id.clone(),
            scope_id: proof_request.scope_id.clone(),
            valid_at_unix_ms: proof_request.valid_at_unix_ms,
            known_at_unix_ms: proof_request.known_at_unix_ms,
            proof_status: ProofVerificationState::VerifiedOptimal,
            proof_packet_sha256: result.packet_sha256.clone(),
            retrieval: progress(true, false),
            required_claims: vec![ClaimLifecycleLedgerEntry {
                conclusion_id: "C1".into(),
                state: ClaimLifecycleState::SupportedCurrent,
                selected_justification_id: Some("proof-C1".into()),
                current_evidence_ids: BTreeSet::from(["E1".into(), "E2".into()]),
                stale_evidence_ids: BTreeSet::new(),
                conflicting_evidence_ids: BTreeSet::new(),
                rejected_evidence_ids: BTreeSet::new(),
                source_session_ids: BTreeSet::from(["session-1".into(), "session-2".into()]),
            }],
        };
        let decision = decide_lifecycle_action(lifecycle_request, Some(&token)).unwrap();
        assert_eq!(decision.action, LifecycleAction::Answer);
        assert_eq!(decision.proof_request_sha256.as_deref(), Some(token.request_sha256.as_str()));
        assert_eq!(decision.proof_packet_sha256.as_deref(), Some(token.packet_sha256.as_str()));
        assert!(decision.answer_is_permitted(Some(&token)).unwrap());
        assert!(decision.answer_is_permitted(None).unwrap_err().contains("original opaque proof-verifier token"));

        // The serialized audit receipt can be retained for history, but the
        // opaque in-process proof capability is deliberately not serialized.
        let serialized = serde_json::to_vec(&decision).unwrap();
        let restored: LifecycleActionDecision = serde_json::from_slice(&serialized).unwrap();
        assert!(restored.answer_is_permitted(None).unwrap_err().contains("original opaque proof-verifier token"));
    }

    #[test]
    fn answers_only_when_every_required_claim_has_verified_current_support() {
        let decision = decide_with_test_token(request(
            ProofVerificationState::VerifiedOptimal,
            vec![supported_claim("C1"), supported_claim("C2")],
            progress(true, false),
        )).unwrap();

        assert_eq!(decision.action, LifecycleAction::Answer);
        assert_eq!(
            decision.reason,
            LifecycleActionReason::AllRequiredClaimsHaveVerifiedCurrentSupport
        );
        // The helper's token is deliberately not retained; the integration test below exercises the final capability gate.
        decision.verify_internal_consistency().unwrap();
    }

    #[test]
    fn verified_optimal_status_without_opaque_verifier_token_is_rejected() {
        let req = request(
            ProofVerificationState::VerifiedOptimal,
            vec![supported_claim("C1")],
            progress(true, false),
        );
        let error = decide_lifecycle_action(req, None).unwrap_err();
        assert!(error.contains("opaque token"));
    }

    #[test]
    fn opaque_proof_token_must_match_exact_selected_claim_dependencies() {
        let req = request(
            ProofVerificationState::VerifiedOptimal,
            vec![supported_claim("C1")],
            progress(true, false),
        );
        let mut token = token_for(&req);
        token
            .selected_proofs
            .get_mut("C1")
            .unwrap()
            .1 = BTreeSet::from(["different-evidence".into()]);
        let error = decide_lifecycle_action(req, Some(&token)).unwrap_err();
        assert!(error.contains("does not match the independently verified selected proof"));
    }

    #[test]
    fn opaque_proof_token_rejects_tampered_session_coverage() {
        let req = request(
            ProofVerificationState::VerifiedOptimal,
            vec![supported_claim("C1")],
            progress(true, false),
        );
        let mut token = token_for(&req);
        token
            .selected_proofs
            .get_mut("C1")
            .unwrap()
            .2 = BTreeSet::from(["session-forged".into()]);

        let error = decide_lifecycle_action(req, Some(&token)).unwrap_err();
        assert!(error.contains("session coverage"));
    }


    #[test]
    fn all_current_looking_claims_without_independent_verification_cannot_answer() {
        let decision = decide_with_test_token(request(
            ProofVerificationState::SearchLimitReached,
            vec![supported_claim("C1")],
            progress(false, false),
        )).unwrap();

        assert_eq!(decision.action, LifecycleAction::Abstain);
        assert!(!decision.answer_is_permitted(None).unwrap());
    }

    #[test]
    fn incomplete_retrieval_requests_more_before_a_terminal_decision() {
        let decision = decide_with_test_token(request(
            ProofVerificationState::NoEligibleProof,
            vec![missing_claim("C1")],
            progress(false, true),
        )).unwrap();

        assert_eq!(decision.action, LifecycleAction::RetrieveMore);
    }

    #[test]
    fn unresolved_current_conflict_is_explained_not_answered() {
        let decision = decide_with_test_token(request(
            ProofVerificationState::NoEligibleProof,
            vec![conflict_claim("C1")],
            progress(true, false),
        )).unwrap();

        assert_eq!(decision.action, LifecycleAction::ExplainConflict);
        assert!(!decision.answer_is_permitted(None).unwrap());
    }

    #[test]
    fn stale_only_support_warns_instead_of_answering() {
        let decision = decide_with_test_token(request(
            ProofVerificationState::NoEligibleProof,
            vec![stale_claim("C1")],
            progress(true, false),
        )).unwrap();

        assert_eq!(decision.action, LifecycleAction::WarnStale);
        assert!(!decision.answer_is_permitted(None).unwrap());
    }

    #[test]
    fn missing_required_evidence_abstains_after_retrieval_is_exhausted() {
        let decision = decide_with_test_token(request(
            ProofVerificationState::NoEligibleProof,
            vec![missing_claim("C1")],
            progress(true, false),
        )).unwrap();

        assert_eq!(decision.action, LifecycleAction::Abstain);
    }

    #[test]
    fn unauthorized_or_invalid_provenance_never_supports_an_answer() {
        for state in [
            ClaimLifecycleState::Unauthorized,
            ClaimLifecycleState::ProvenanceInvalid,
        ] {
            let decision = decide_with_test_token(request(
                ProofVerificationState::NoEligibleProof,
                vec![rejected_claim("C1", state)],
                progress(true, false),
            )).unwrap();
            assert_eq!(decision.action, LifecycleAction::Abstain);
            assert!(!decision.answer_is_permitted(None).unwrap());
        }
    }

    #[test]
    fn retrieval_continuation_precedes_conflict_and_stale_terminal_actions() {
        let decision = decide_with_test_token(request(
            ProofVerificationState::NoEligibleProof,
            vec![conflict_claim("C1"), stale_claim("C2")],
            progress(false, true),
        )).unwrap();

        assert_eq!(decision.action, LifecycleAction::RetrieveMore);
    }

    #[test]
    fn claim_order_does_not_change_canonical_action_receipt() {
        let first = decide_with_test_token(request(
            ProofVerificationState::VerifiedOptimal,
            vec![supported_claim("C2"), supported_claim("C1")],
            progress(true, false),
        )).unwrap();
        let second = decide_with_test_token(request(
            ProofVerificationState::VerifiedOptimal,
            vec![supported_claim("C1"), supported_claim("C2")],
            progress(true, false),
        )).unwrap();

        assert_eq!(first.action_sha256, second.action_sha256);
        assert_eq!(
            first.claim_ledger.iter().map(|claim| claim.conclusion_id.as_str()).collect::<Vec<_>>(),
            vec!["C1", "C2"]
        );
    }

    #[test]
    fn action_receipt_detects_post_decision_mutation() {
        let mut decision = decide_with_test_token(request(
            ProofVerificationState::VerifiedOptimal,
            vec![supported_claim("C1")],
            progress(true, false),
        )).unwrap();
        decision.action = LifecycleAction::Abstain;
        assert!(decision.verify_internal_consistency().unwrap_err().contains("SHA-256 mismatch"));
    }

    #[derive(serde::Deserialize)]
    struct LifecycleActionFixtureFile {
        schema_version: String,
        policy_version: String,
        case_count: usize,
        cases: Vec<LifecycleActionFixtureCase>,
    }

    #[derive(serde::Deserialize)]
    struct LifecycleActionFixtureCase {
        case_id: String,
        proof_status: ProofVerificationState,
        claim_state: ClaimLifecycleState,
        retrieval_complete: bool,
        retrieval_can_continue: bool,
        remaining_calls: u32,
        expected_action: LifecycleAction,
    }

    #[test]
    fn versioned_lifecycle_action_fixture_corpus_matches_controller() {
        let raw = include_str!(
            "../../../../research/broca-bitemporal-evaluation-v1/lifecycle-action-fixtures-v1.json"
        );
        let fixture: LifecycleActionFixtureFile = serde_json::from_str(raw).unwrap();
        assert_eq!(fixture.schema_version, "broca-lifecycle-action-fixtures-v1");
        assert_eq!(fixture.policy_version, LIFECYCLE_POLICY_VERSION);
        assert_eq!(fixture.case_count, fixture.cases.len());
        assert_eq!(fixture.case_count, 60);

        for case in fixture.cases {
            let claim = match case.claim_state {
                ClaimLifecycleState::SupportedCurrent => supported_claim(&case.case_id),
                ClaimLifecycleState::StaleOnly => stale_claim(&case.case_id),
                ClaimLifecycleState::Conflicted => conflict_claim(&case.case_id),
                ClaimLifecycleState::Missing => missing_claim(&case.case_id),
                ClaimLifecycleState::Unauthorized => {
                    rejected_claim(&case.case_id, ClaimLifecycleState::Unauthorized)
                }
                ClaimLifecycleState::ProvenanceInvalid => {
                    rejected_claim(&case.case_id, ClaimLifecycleState::ProvenanceInvalid)
                }
            };
            let retrieval = LifecycleRetrievalProgress {
                complete: case.retrieval_complete,
                can_continue: case.retrieval_can_continue,
                remaining_calls: case.remaining_calls,
            };
            let decision = decide_with_test_token(request(
                case.proof_status,
                vec![claim],
                retrieval,
            )).unwrap();
            assert_eq!(
                decision.action,
                case.expected_action,
                "fixture case {} expected {:?}, got {:?}",
                case.case_id,
                case.expected_action,
                decision.action
            );
        }
    }

    #[test]
    fn exhaustive_lifecycle_action_truth_table_matches_policy_precedence() {
        let proof_states = [
            ProofVerificationState::NotRun,
            ProofVerificationState::SearchLimitReached,
            ProofVerificationState::NoEligibleProof,
            ProofVerificationState::BudgetInsufficient,
            ProofVerificationState::Rejected,
        ];
        let claim_cases: Vec<(ClaimLifecycleState, fn(&str) -> ClaimLifecycleLedgerEntry)> = vec![
            (ClaimLifecycleState::SupportedCurrent, supported_claim),
            (ClaimLifecycleState::StaleOnly, stale_claim),
            (ClaimLifecycleState::Conflicted, conflict_claim),
            (ClaimLifecycleState::Missing, missing_claim),
            (
                ClaimLifecycleState::Unauthorized,
                |id| rejected_claim(id, ClaimLifecycleState::Unauthorized),
            ),
            (
                ClaimLifecycleState::ProvenanceInvalid,
                |id| rejected_claim(id, ClaimLifecycleState::ProvenanceInvalid),
            ),
        ];

        let mut evaluated = 0usize;
        for proof_status in proof_states {
            for (claim_state, make_claim) in &claim_cases {
                for can_continue in [false, true] {
                    let retrieval = progress(!can_continue, can_continue);
                    let decision = decide_with_test_token(request(
                        proof_status,
                        vec![make_claim("C1")],
                        retrieval,
                    )).unwrap();

                    let expected = if can_continue {
                        LifecycleAction::RetrieveMore
                    } else {
                        match claim_state {
                            ClaimLifecycleState::SupportedCurrent
                            | ClaimLifecycleState::Missing
                            | ClaimLifecycleState::Unauthorized
                            | ClaimLifecycleState::ProvenanceInvalid => LifecycleAction::Abstain,
                            ClaimLifecycleState::StaleOnly => LifecycleAction::WarnStale,
                            ClaimLifecycleState::Conflicted => LifecycleAction::ExplainConflict,
                        }
                    };
                    assert_eq!(
                        decision.action,
                        expected,
                        "policy mismatch: proof={proof_status:?}, claim={claim_state:?}, can_continue={can_continue}",
                    );
                    assert!(!decision.answer_is_permitted(None).unwrap());
                    evaluated += 1;
                }
            }
        }
        assert_eq!(evaluated, 60);
    }

    #[test]
    fn decision_cannot_be_reused_across_principal_scope_or_time() {
        let decision = decide_with_test_token(request(
            ProofVerificationState::VerifiedOptimal,
            vec![supported_claim("C1")],
            progress(true, false),
        )).unwrap();
        decision.verify_request_binding("principal-1", "scope-1", 200, 300).unwrap();
        assert!(decision.verify_request_binding("other", "scope-1", 200, 300).is_err());
        assert!(decision.verify_request_binding("principal-1", "other", 200, 300).is_err());
        assert!(decision.verify_request_binding("principal-1", "scope-1", 201, 300).is_err());
        assert!(decision.verify_request_binding("principal-1", "scope-1", 200, 301).is_err());
    }
}
