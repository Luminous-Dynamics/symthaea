// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Independent exhaustive verifier for minimum-cost proof packets.
//!
//! The proof-budget optimizer uses branch-and-bound. This module deliberately
//! does not call that optimizer or reuse its search state. It enumerates the
//! admissible Cartesian product of proof alternatives, recomputes each union
//! cost, and compares the globally minimal canonical choice with the packet.
//!
//! This is an independently implemented verifier, not a succinct proof of
//! optimality: verification can be exponential. It fails closed at the
//! request's explicit node limit. A future certificate-based verifier can
//! replace exhaustive enumeration without changing the packet sufficiency
//! contract.
//!
//! Trust boundary: verify_proof_packet checks packet structure and the shared
//! request contract first. Source resolution, authorization receipt signature
//! verification, contradiction discovery, tokenization, and rendering remain
//! responsibilities of their separately identified trusted adapters.

use crate::memory_evidence::EvidenceSpeakerRole;
use crate::memory_proof_budget::{
    ProofBudgetRequest, ProofBudgetResult, ProofBudgetStatus, ProofEvidenceCandidate,
    SelectedProof, verify_proof_packet,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

const VERIFIER_ID: &str = "broca-independent-exhaustive-optimality-v1";

/// Diagnostic receipt returned only after independent exhaustive verification.
///
/// This receipt is not a signature and is not itself a succinct mathematical
/// certificate. It records the independent verifier implementation identity,
/// the exact packet/request binding, and the search coverage it performed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProofOptimalityVerificationReceipt {
    pub verifier_id: String,
    pub request_sha256: String,
    pub packet_sha256: String,
    pub verifier_search_nodes: u64,
    pub complete_combinations: u64,
    pub feasible_combinations: u64,
    pub minimum_token_cost: u64,
}

#[derive(Debug, Clone)]
struct IndependentPlan {
    evidence_ids: Vec<String>,
    selected_proofs: Vec<SelectedProof>,
    choice_key: Vec<(String, String)>,
    total_token_cost: u64,
}

fn canonicalize_request(mut request: ProofBudgetRequest) -> ProofBudgetRequest {
    request
        .candidates
        .sort_by(|left, right| left.evidence_id.cmp(&right.evidence_id));
    request
        .obligations
        .sort_by(|left, right| left.conclusion_id.cmp(&right.conclusion_id));
    for obligation in &mut request.obligations {
        obligation
            .alternatives
            .sort_by(|left, right| left.justification_id.cmp(&right.justification_id));
        for alternative in &mut obligation.alternatives {
            alternative.evidence_dependencies.sort();
        }
    }
    request
}

fn source_span_digest_matches(candidate: &ProofEvidenceCandidate) -> bool {
    let resolved = format!("{:x}", Sha256::digest(candidate.source_span_text.as_bytes()));
    candidate.source_span_sha256 == candidate.resolved_span_sha256
        && candidate.source_span_sha256 == resolved
}

fn independently_admissible(
    candidate: &ProofEvidenceCandidate,
    request: &ProofBudgetRequest,
) -> bool {
    let known_in_time = candidate.recorded_at_unix_ms <= request.known_at_unix_ms
        && candidate
            .retracted_at_unix_ms
            .is_none_or(|retracted| retracted > request.known_at_unix_ms);
    let valid_in_world = candidate
        .valid_from_unix_ms
        .is_none_or(|start| request.valid_at_unix_ms >= start)
        && candidate
            .valid_until_unix_ms
            .is_none_or(|end| request.valid_at_unix_ms < end);
    let authorized = candidate.authorization_accepted
        && candidate.authorized_principal_id == request.principal_id;

    known_in_time
        && valid_in_world
        && candidate.scope_id == request.scope_id
        && authorized
        && source_span_digest_matches(candidate)
}

fn has_conflict(
    selected_ids: &BTreeSet<String>,
    candidates: &BTreeMap<String, ProofEvidenceCandidate>,
) -> bool {
    selected_ids.iter().any(|evidence_id| {
        candidates
            .get(evidence_id)
            .is_none_or(|candidate| {
                candidate
                    .conflicts_with
                    .iter()
                    .any(|other| selected_ids.contains(other))
            })
    })
}

fn checked_plan_cost(
    request: &ProofBudgetRequest,
    selected_ids: &BTreeSet<String>,
    selected_proofs: &[SelectedProof],
    candidates: &BTreeMap<String, ProofEvidenceCandidate>,
) -> Result<u64, String> {
    let mut total = request.fixed_packet_overhead_tokens;
    for evidence_id in selected_ids {
        let candidate = candidates
            .get(evidence_id)
            .ok_or_else(|| format!("independent verifier lost evidence {evidence_id:?}"))?;
        total = total
            .checked_add(u64::from(candidate.token_cost))
            .ok_or_else(|| "independent verifier evidence cost overflow".to_string())?;
    }
    for proof in selected_proofs {
        total = total
            .checked_add(u64::from(proof.selection_overhead_tokens))
            .ok_or_else(|| "independent verifier proof-envelope cost overflow".to_string())?;
    }
    Ok(total)
}

fn compare_plans(left: &IndependentPlan, right: &IndependentPlan) -> Ordering {
    left.total_token_cost
        .cmp(&right.total_token_cost)
        .then_with(|| left.evidence_ids.cmp(&right.evidence_ids))
        .then_with(|| left.choice_key.cmp(&right.choice_key))
}

struct Enumeration<'a> {
    request: &'a ProofBudgetRequest,
    candidates: &'a BTreeMap<String, ProofEvidenceCandidate>,
    eligible_ids: &'a BTreeSet<String>,
    nodes: u64,
    complete_combinations: u64,
    feasible_combinations: u64,
    best: Option<IndependentPlan>,
    max_search_nodes: u64,
}

impl Enumeration<'_> {
    fn visit(
        &mut self,
        obligation_index: usize,
        selected_ids: &BTreeSet<String>,
        selected_proofs: &mut Vec<SelectedProof>,
    ) -> Result<(), String> {
        if self.nodes >= self.max_search_nodes {
            return Err(format!(
                "independent optimality verifier exhausted max_search_nodes={} before exhaustive completion",
                self.max_search_nodes
            ));
        }
        self.nodes += 1;

        if obligation_index == self.request.obligations.len() {
            self.complete_combinations = self
                .complete_combinations
                .checked_add(1)
                .ok_or_else(|| "independent verifier combination counter overflow".to_string())?;
            let evidence_ids: Vec<String> = selected_ids.iter().cloned().collect();
            let total_token_cost = checked_plan_cost(
                self.request,
                selected_ids,
                selected_proofs,
                self.candidates,
            )?;
            if total_token_cost <= self.request.token_budget {
                self.feasible_combinations = self
                    .feasible_combinations
                    .checked_add(1)
                    .ok_or_else(|| "independent verifier feasible counter overflow".to_string())?;
            }
            let choice_key = selected_proofs
                .iter()
                .map(|proof| (proof.conclusion_id.clone(), proof.justification_id.clone()))
                .collect();
            let candidate = IndependentPlan {
                evidence_ids,
                selected_proofs: selected_proofs.clone(),
                choice_key,
                total_token_cost,
            };
            if self
                .best
                .as_ref()
                .is_none_or(|best| compare_plans(&candidate, best) == Ordering::Less)
            {
                self.best = Some(candidate);
            }
            return Ok(());
        }

        // Own the branch point so recursive mutable enumeration does not
        // retain a borrow into the request stored on self.
        let conclusion_id = self.request.obligations[obligation_index].conclusion_id.clone();
        let minimum_independent_families =
            self.request.obligations[obligation_index].minimum_independent_families;
        let minimum_distinct_sessions =
            self.request.obligations[obligation_index].minimum_distinct_sessions;
        let minimum_user_authored_evidence =
            self.request.obligations[obligation_index].minimum_user_authored_evidence;
        let selection_overhead_tokens =
            self.request.obligations[obligation_index].selection_overhead_tokens;
        let alternatives = self.request.obligations[obligation_index].alternatives.clone();

        for alternative in &alternatives {
            let mut families = BTreeSet::<String>::new();
            let mut sessions = BTreeSet::<String>::new();
            let mut user_authored_evidence = BTreeSet::<String>::new();
            let mut alternative_is_admissible = true;

            for evidence_id in &alternative.evidence_dependencies {
                let Some(candidate) = self.candidates.get(evidence_id) else {
                    alternative_is_admissible = false;
                    break;
                };
                if !self.eligible_ids.contains(evidence_id) {
                    alternative_is_admissible = false;
                    break;
                }
                families.insert(candidate.source_family_id.clone());
                if minimum_distinct_sessions > 0 {
                    let Some(session_id) = &candidate.source_session_id else {
                        alternative_is_admissible = false;
                        break;
                    };
                    sessions.insert(session_id.clone());
                }
                if candidate.source_speaker == Some(EvidenceSpeakerRole::User) {
                    user_authored_evidence.insert(candidate.evidence_id.clone());
                }
            }

            if !alternative_is_admissible
                || families.len() < minimum_independent_families as usize
                || sessions.len() < minimum_distinct_sessions as usize
                || user_authored_evidence.len() < minimum_user_authored_evidence as usize
                || obligation.required_complete_evidence_ids
                    .iter()
                    .any(|required_id| !alternative.evidence_dependencies.contains(required_id))
            {
                continue;
            }

            let mut next_ids = selected_ids.clone();
            next_ids.extend(alternative.evidence_dependencies.iter().cloned());
            if has_conflict(&next_ids, self.candidates) {
                continue;
            }

            selected_proofs.push(SelectedProof {
                conclusion_id: conclusion_id.clone(),
                justification_id: alternative.justification_id.clone(),
                evidence_dependencies: alternative.evidence_dependencies.clone(),
                selection_overhead_tokens,
            });

            self.visit(obligation_index + 1, &next_ids, selected_proofs)?;
            selected_proofs.pop();
        }
        Ok(())
    }
}

/// Verify that a Supported packet is globally minimum-cost using a separate
/// exhaustive implementation. The optimizer's branch-and-bound function is
/// never called by this verifier.
///
/// Returns an audit receipt on success. Returns an error if the packet is
/// insufficient, if the independent search is bounded before completion, or
/// if any cheaper or deterministically preferred equal-cost proof exists.
pub fn verify_minimum_cost_optimality(
    request: &ProofBudgetRequest,
    result: &ProofBudgetResult,
) -> Result<ProofOptimalityVerificationReceipt, String> {
    verify_minimum_cost_optimality_with_limit(request, result, request.max_search_nodes)
}

/// Verify minimum-cost optimality with a verifier-specific node limit.
///
/// This permits the caller to impose a stricter verification budget than the
/// optimizer used to produce the packet. Hitting the limit is always an error,
/// never an optimality receipt.
pub fn verify_minimum_cost_optimality_with_limit(
    request: &ProofBudgetRequest,
    result: &ProofBudgetResult,
    verifier_max_search_nodes: u64,
) -> Result<ProofOptimalityVerificationReceipt, String> {
    if verifier_max_search_nodes == 0 {
        return Err("verifier_max_search_nodes must be greater than zero".into());
    }
    if result.status != ProofBudgetStatus::Supported {
        return Err("only a Supported result can be checked for minimum-cost optimality".into());
    }

    // The existing verifier checks source eligibility, exact proof edges,
    // source-set equality, renderer/token cost, and the packet digest.
    verify_proof_packet(request, result)?;

    let canonical_request = canonicalize_request(request.clone());
    let candidates: BTreeMap<String, ProofEvidenceCandidate> = canonical_request
        .candidates
        .iter()
        .cloned()
        .map(|candidate| (candidate.evidence_id.clone(), candidate))
        .collect();

    let eligible_ids: BTreeSet<String> = candidates
        .iter()
        .filter_map(|(evidence_id, candidate)| {
            independently_admissible(candidate, &canonical_request)
                .then(|| evidence_id.clone())
        })
        .collect();

    let mut enumeration = Enumeration {
        request: &canonical_request,
        candidates: &candidates,
        eligible_ids: &eligible_ids,
        nodes: 0,
        complete_combinations: 0,
        feasible_combinations: 0,
        best: None,
        max_search_nodes: verifier_max_search_nodes,
    };
    enumeration.visit(0, &BTreeSet::new(), &mut Vec::new())?;

    let optimum = enumeration
        .best
        .ok_or_else(|| "independent verifier found no admissible complete proof combination".to_string())?;

    if optimum.total_token_cost > canonical_request.token_budget {
        return Err("independent optimum exceeds the requested packet token budget".into());
    }
    if optimum.total_token_cost != result.total_token_cost
        || optimum.evidence_ids
            != result.evidence.iter().map(|item| item.evidence_id.clone()).collect::<Vec<_>>()
        || optimum.selected_proofs != result.selected_proofs
    {
        return Err(format!(
            "packet is sufficient but not the independently verified canonical optimum (packet_cost={}, optimum_cost={})",
            result.total_token_cost, optimum.total_token_cost
        ));
    }

    let packet_sha256 = result
        .packet_sha256
        .clone()
        .ok_or_else(|| "Supported result is missing packet_sha256".to_string())?;

    Ok(ProofOptimalityVerificationReceipt {
        verifier_id: VERIFIER_ID.to_string(),
        request_sha256: result.request_sha256.clone(),
        packet_sha256,
        verifier_search_nodes: enumeration.nodes,
        complete_combinations: enumeration.complete_combinations,
        feasible_combinations: enumeration.feasible_combinations,
        minimum_token_cost: optimum.total_token_cost,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_proof_budget::{
        ProofBudgetStatus, ProofEvidenceCandidate, ProofJustification, ProofObligation,
        solve_proof_budget,
    };

    fn digest(text: &str) -> String {
        format!("{:x}", Sha256::digest(text.as_bytes()))
    }

    fn candidate(id: &str, family: &str, token_cost: u32) -> ProofEvidenceCandidate {
        let excerpt = format!("source excerpt {id}");
        let span_digest = digest(&excerpt);
        ProofEvidenceCandidate {
            evidence_id: id.into(),
            source_revision_id: format!("revision-{id}"),
            source_pointer: format!("/messages/{id}/content"),
            source_span_text: excerpt.clone(),
            source_span_start: 0,
            source_span_end: excerpt.len() as u64,
            source_family_id: family.into(),
            source_session_id: Some(format!("session-{id}")),
            source_speaker: Some(EvidenceSpeakerRole::User),
            scope_id: "scope-1".into(),
            payload_sha256: digest("payload"),
            source_span_sha256: span_digest.clone(),
            resolved_span_sha256: span_digest,
            conflicts_with: BTreeSet::new(),
            recorded_at_unix_ms: 100,
            retracted_at_unix_ms: None,
            valid_from_unix_ms: Some(10),
            valid_until_unix_ms: None,
            authorized_principal_id: "principal-1".into(),
            authorization_accepted: true,
            authorization_receipt_sha256: digest("authorization receipt"),
            retrieval_rank: 50,
            token_cost,
        }
    }

    fn request() -> ProofBudgetRequest {
        ProofBudgetRequest {
            decision_id: "decision-1".into(),
            principal_id: "principal-1".into(),
            scope_id: "scope-1".into(),
            valid_at_unix_ms: 200,
            known_at_unix_ms: 200,
            policy_id: "policy-v1".into(),
            policy_sha256: digest("policy"),
            fixed_packet_overhead_tokens: 3,
            token_budget: 64,
            max_search_nodes: 10_000,
            tokenizer_id: "test-tokenizer".into(),
            tokenizer_version: "1".into(),
            tokenizer_sha256: digest("tokenizer"),
            packet_renderer_sha256: digest("renderer"),
            candidates: vec![
                candidate("expensive", "family-expensive", 10),
                candidate("cheap", "family-cheap", 2),
            ],
            obligations: vec![ProofObligation {
                conclusion_id: "claim-1".into(),
                alternatives: vec![
                    ProofJustification {
                        justification_id: "proof-expensive".into(),
                        evidence_dependencies: vec!["expensive".into()],
                    },
                    ProofJustification {
                        justification_id: "proof-cheap".into(),
                        evidence_dependencies: vec!["cheap".into()],
                    },
                ],
                minimum_independent_families: 1,
                minimum_distinct_sessions: 0,
                minimum_user_authored_evidence: 0,
                required_complete_evidence_ids: BTreeSet::new(),
                enumeration_receipt_sha256: None,
                selection_overhead_tokens: 2,
            }],
        }
    }

    #[test]
    fn independent_verifier_confirms_optimizer_result_and_reports_coverage() {
        let req = request();
        let result = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        let receipt = verify_minimum_cost_optimality(&req, &result).unwrap();

        assert_eq!(receipt.verifier_id, VERIFIER_ID);
        assert_eq!(receipt.minimum_token_cost, 7);
        assert_eq!(receipt.complete_combinations, 2);
        assert_eq!(receipt.feasible_combinations, 2);
        assert!(receipt.verifier_search_nodes >= receipt.complete_combinations);
        assert_eq!(result.selected_proofs[0].justification_id, "proof-cheap");
    }

    #[test]
    fn independent_verifier_rejects_packet_digest_mutation() {
        let req = request();
        let mut result = solve_proof_budget(req.clone()).unwrap();
        result.packet_sha256 = Some(digest("tampered packet"));
        assert!(verify_minimum_cost_optimality(&req, &result).is_err());
    }

    #[derive(Serialize)]
    struct PacketPayloadForTest<'a> {
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
        evidence: &'a [crate::memory_proof_budget::ProofPacketEvidenceRef],
        total_token_cost: u64,
    }

    fn valid_but_nonminimal_packet(request: &ProofBudgetRequest) -> ProofBudgetResult {
        let mut expensive_only = request.clone();
        expensive_only.candidates.retain(|candidate| candidate.evidence_id == "expensive");
        expensive_only.obligations[0]
            .alternatives
            .retain(|alternative| alternative.justification_id == "proof-expensive");
        let mut result = solve_proof_budget(expensive_only).unwrap();

        let canonical = canonicalize_request(request.clone());
        let request_bytes = serde_json::to_vec(&canonical).unwrap();
        result.request_sha256 = format!("{:x}", Sha256::digest(request_bytes));

        let candidate = canonical
            .candidates
            .iter()
            .find(|candidate| candidate.evidence_id == "expensive")
            .unwrap();
        result.selected_proofs = vec![SelectedProof {
            conclusion_id: "claim-1".into(),
            justification_id: "proof-expensive".into(),
            evidence_dependencies: vec!["expensive".into()],
            selection_overhead_tokens: 2,
        }];
        result.evidence = vec![crate::memory_proof_budget::ProofPacketEvidenceRef {
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
        result.total_token_cost = canonical.fixed_packet_overhead_tokens
            + u64::from(candidate.token_cost)
            + u64::from(canonical.obligations[0].selection_overhead_tokens);

        let payload = PacketPayloadForTest {
            decision_id: &canonical.decision_id,
            principal_id: &canonical.principal_id,
            scope_id: &canonical.scope_id,
            valid_at_unix_ms: canonical.valid_at_unix_ms,
            known_at_unix_ms: canonical.known_at_unix_ms,
            request_sha256: &result.request_sha256,
            policy_id: &canonical.policy_id,
            policy_sha256: &canonical.policy_sha256,
            tokenizer_id: &canonical.tokenizer_id,
            tokenizer_version: &canonical.tokenizer_version,
            tokenizer_sha256: &canonical.tokenizer_sha256,
            packet_renderer_sha256: &canonical.packet_renderer_sha256,
            fixed_packet_overhead_tokens: canonical.fixed_packet_overhead_tokens,
            token_budget: canonical.token_budget,
            selected_proofs: &result.selected_proofs,
            evidence: &result.evidence,
            total_token_cost: result.total_token_cost,
        };
        let packet_bytes = serde_json::to_vec(&payload).unwrap();
        result.packet_sha256 = Some(format!("{:x}", Sha256::digest(packet_bytes)));
        result
    }

    #[test]
    fn independent_verifier_rejects_a_sufficient_but_nonminimal_packet() {
        let req = request();
        let result = valid_but_nonminimal_packet(&req);

        // Prove this is not merely a malformed-packet rejection.
        verify_proof_packet(&req, &result).unwrap();
        assert_eq!(result.total_token_cost, 15);
        let error = verify_minimum_cost_optimality(&req, &result).unwrap_err();
        assert!(error.contains("not the independently verified canonical optimum"));
        let routed_error = crate::memory_proof_budget::verify_proof_budget_result(&req, &result)
            .unwrap_err();
        assert!(routed_error.contains("not the independently verified canonical optimum"));
    }

    #[test]
    fn independent_verifier_fails_closed_when_its_own_search_budget_is_exhausted() {
        let req = request();
        let result = solve_proof_budget(req.clone()).unwrap();
        let error = verify_minimum_cost_optimality_with_limit(&req, &result, 1).unwrap_err();
        assert!(error.contains("exhausted max_search_nodes=1"));
    }

    #[test]
    fn independent_verifier_prefers_a_valid_distinct_family_proof_over_correlated_revisions() {
        let mut req = request();
        req.candidates = vec![
            candidate("revision-a1", "family-a", 1),
            candidate("revision-a2", "family-a", 1),
            candidate("source-b", "family-b", 5),
            candidate("source-c", "family-c", 6),
        ];
        req.obligations[0].minimum_independent_families = 2;
        req.obligations[0].alternatives = vec![
            ProofJustification {
                justification_id: "correlated-cheap".into(),
                evidence_dependencies: vec!["revision-a1".into(), "revision-a2".into()],
            },
            ProofJustification {
                justification_id: "independent-support".into(),
                evidence_dependencies: vec!["source-b".into(), "source-c".into()],
            },
        ];

        let result = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(result.selected_proofs[0].justification_id, "independent-support");
        let receipt = verify_minimum_cost_optimality(&req, &result).unwrap();
        assert_eq!(receipt.minimum_token_cost, 16);
        assert_eq!(receipt.complete_combinations, 1);
    }
}
