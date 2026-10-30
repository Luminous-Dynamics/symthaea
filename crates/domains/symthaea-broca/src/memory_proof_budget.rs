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

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

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

/// One immutable, source-anchored evidence candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProofEvidenceCandidate {
    pub evidence_id: String,
    pub source_family_id: String,
    pub scope_id: String,
    pub payload_sha256: String,
    pub source_span_sha256: String,
    /// Digest recomputed by the trusted source-span resolver.
    pub resolved_span_sha256: String,
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
    /// Exact cost from the pinned packet tokenizer/renderer.
    pub token_cost: u32,
}

impl ProofEvidenceCandidate {
    fn validate(&self) -> Result<(), String> {
        for (field, value) in [
            ("evidence_id", self.evidence_id.as_str()),
            ("source_family_id", self.source_family_id.as_str()),
            ("scope_id", self.scope_id.as_str()),
            ("authorized_principal_id", self.authorized_principal_id.as_str()),
        ] {
            require_non_empty(value, field)?;
        }
        valid_sha256(&self.payload_sha256, "payload_sha256")?;
        valid_sha256(&self.source_span_sha256, "source_span_sha256")?;
        valid_sha256(&self.resolved_span_sha256, "resolved_span_sha256")?;
        valid_sha256(
            &self.authorization_receipt_sha256,
            "authorization_receipt_sha256",
        )?;
        if self.token_cost == 0 {
            return Err("token_cost must be greater than zero".into());
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
        if self.source_span_sha256 != self.resolved_span_sha256 {
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
    pub token_budget: u64,
    pub max_search_nodes: u64,
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
        ] {
            require_non_empty(value, field)?;
        }
        valid_sha256(&self.policy_sha256, "policy_sha256")?;
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

        let mut conclusions = BTreeSet::new();
        for obligation in &self.obligations {
            require_non_empty(&obligation.conclusion_id, "conclusion_id")?;
            if !conclusions.insert(obligation.conclusion_id.as_str()) {
                return Err(format!("duplicate conclusion {:?}", obligation.conclusion_id));
            }
            if obligation.minimum_independent_families == 0 {
                return Err("minimum_independent_families must be greater than zero".into());
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofPacketEvidenceRef {
    pub evidence_id: String,
    pub source_family_id: String,
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
    policy_id: &'a str,
    policy_sha256: &'a str,
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
    budget_blocked: bool,
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
            budget_blocked: false,
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
            let plan = CandidatePlan {
                total_token_cost: self.cost_of(&evidence_ids),
                evidence_ids,
                selected_proofs: selected_proofs.to_vec(),
            };
            self.consider(plan);
            return;
        }

        let obligation = &self.request.obligations[obligation_index];
        for alternative in &obligation.alternatives {
            if self.search_limit_reached {
                return;
            }

            let mut missing = false;
            let mut ineligible = false;
            let mut families = BTreeSet::new();
            for evidence_id in &alternative.evidence_dependencies {
                let Some(candidate) = self.candidate_by_id.get(evidence_id.as_str()) else {
                    missing = true;
                    continue;
                };
                if !self.eligible_by_id.contains_key(evidence_id.as_str()) {
                    ineligible = true;
                    continue;
                }
                families.insert(candidate.source_family_id.as_str());
            }

            if missing || ineligible {
                self.justification_rejections.insert(JustificationRejection {
                    conclusion_id: obligation.conclusion_id.clone(),
                    justification_id: alternative.justification_id.clone(),
                    reason: if missing {
                        JustificationRejectionReason::MissingEvidenceReference
                    } else {
                        JustificationRejectionReason::CandidateIneligible
                    },
                });
                continue;
            }
            if families.len() < obligation.minimum_independent_families as usize {
                self.justification_rejections.insert(JustificationRejection {
                    conclusion_id: obligation.conclusion_id.clone(),
                    justification_id: alternative.justification_id.clone(),
                    reason: JustificationRejectionReason::IndependentFamilyCountInsufficient,
                });
                continue;
            }

            let mut next_evidence = selected_evidence_ids.clone();
            next_evidence.extend(alternative.evidence_dependencies.iter().cloned());
            let next_cost = self.cost_of_set(&next_evidence);
            if next_cost > self.request.token_budget {
                self.budget_blocked = true;
                continue;
            }
            if self.best.as_ref()
                .is_some_and(|best| next_cost > best.total_token_cost)
            {
                continue;
            }

            let mut next_proofs = selected_proofs.to_vec();
            next_proofs.push(SelectedProof {
                conclusion_id: obligation.conclusion_id.clone(),
                justification_id: alternative.justification_id.clone(),
            });
            self.explore(obligation_index + 1, &next_evidence, &next_proofs);
        }
    }

    fn cost_of_set(&self, evidence_ids: &BTreeSet<String>) -> u64 {
        evidence_ids
            .iter()
            .filter_map(|id| self.candidate_by_id.get(id.as_str()))
            .map(|candidate| candidate.token_cost as u64)
            .sum()
    }

    fn cost_of(&self, evidence_ids: &[String]) -> u64 {
        let set: BTreeSet<_> = evidence_ids.iter().cloned().collect();
        self.cost_of_set(&set)
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
            let candidate = candidate_map[evidence_id.as_str()];
            ProofPacketEvidenceRef {
                evidence_id: candidate.evidence_id.clone(),
                source_family_id: candidate.source_family_id.clone(),
                payload_sha256: candidate.payload_sha256.clone(),
                source_span_sha256: candidate.source_span_sha256.clone(),
                authorization_receipt_sha256: candidate.authorization_receipt_sha256.clone(),
                retrieval_rank: candidate.retrieval_rank,
                token_cost: candidate.token_cost,
            }
        })
        .collect();
    let payload = ProofPacketPayload {
        decision_id: &request.decision_id,
        principal_id: &request.principal_id,
        scope_id: &request.scope_id,
        valid_at_unix_ms: request.valid_at_unix_ms,
        known_at_unix_ms: request.known_at_unix_ms,
        policy_id: &request.policy_id,
        policy_sha256: &request.policy_sha256,
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

        let mut independent_families = BTreeSet::new();
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
            selected_evidence_ids.insert(evidence_id.clone());
        }
        if independent_families.len() < obligation.minimum_independent_families as usize {
            return Err(format!(
                "selected proof {:?} does not meet the independent-family threshold",
                selected.justification_id
            ));
        }
    }

    if seen_conclusions.len() != request.obligations.len() {
        return Err("proof packet omits one or more required conclusions".into());
    }

    let mut expected_evidence = Vec::new();
    let mut total_token_cost = 0u64;
    for evidence_id in &selected_evidence_ids {
        let candidate = candidate_by_id[evidence_id.as_str()];
        total_token_cost = total_token_cost
            .checked_add(candidate.token_cost as u64)
            .ok_or_else(|| "proof packet token cost overflow".to_string())?;
        expected_evidence.push(ProofPacketEvidenceRef {
            evidence_id: candidate.evidence_id.clone(),
            source_family_id: candidate.source_family_id.clone(),
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
    if result.total_token_cost != total_token_cost {
        return Err("proof packet token cost does not match its unique evidence set".into());
    }
    if total_token_cost > request.token_budget {
        return Err("proof packet exceeds the requested token budget".into());
    }

    let payload = ProofPacketPayload {
        decision_id: &request.decision_id,
        principal_id: &request.principal_id,
        scope_id: &request.scope_id,
        valid_at_unix_ms: request.valid_at_unix_ms,
        known_at_unix_ms: request.known_at_unix_ms,
        policy_id: &request.policy_id,
        policy_sha256: &request.policy_sha256,
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
    verify_proof_packet(request, result)?;
    let recomputed = solve_proof_budget(request.clone())?;
    if &recomputed != result {
        return Err("proof-budget result is valid but differs from the independently recomputed optimum".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: char) -> String {
        std::iter::repeat(byte).take(64).collect()
    }

    fn candidate(id: &str, family: &str, token_cost: u32) -> ProofEvidenceCandidate {
        ProofEvidenceCandidate {
            evidence_id: id.into(),
            source_family_id: family.into(),
            scope_id: "scope-1".into(),
            payload_sha256: digest('a'),
            source_span_sha256: digest('b'),
            resolved_span_sha256: digest('b'),
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
            token_budget: budget,
            max_search_nodes: 10_000,
            candidates,
            obligations,
        }
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
