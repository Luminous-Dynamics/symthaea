// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fail-closed adapter from independently verified proof packets to reader context.
//!
//! No retrieval, reranking, truncation, or excerpt rewriting occurs here. The
//! adapter verifies sufficiency and optimality first, then exposes exact source
//! spans as MemoryContextItems while retaining provenance and proof edges in a
//! separate control-plane structure.
//!
//! This module does not prove natural-language entailment. A downstream claim
//! extractor must map answer claims to registered conclusion IDs; this module
//! then checks the declared selected justification and its complete evidence set.
//! Source resolution, authorization, contradiction discovery, tokenization and
//! rendering remain the responsibility of separately identified trusted adapters.

use crate::memory_evidence::{EvidenceSpeakerRole, MemoryContextItem, MemoryContextType};
use crate::memory_proof_budget::{
    ProofBudgetRequest, ProofBudgetResult, ProofBudgetStatus, SelectedProof,
    verify_proof_budget_result,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Provenance retained outside reader-visible text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedReaderEvidence {
    pub evidence_id: String,
    pub source_revision_id: String,
    pub source_pointer: String,
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

/// The selected OR-of-AND proof for one required conclusion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedReaderClaimProof {
    pub conclusion_id: String,
    pub justification_id: String,
    pub selection_overhead_tokens: u32,
    /// Complete conjunctive dependency set in canonical order.
    pub evidence_dependencies: Vec<String>,
}

/// Claim/evidence declaration supplied by a downstream answer adapter.
///
/// The struct records a mapping made by the caller; it is not a natural-language
/// entailment proof and does not discover claims in generated prose.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReaderClaimEvidenceUse {
    pub conclusion_id: String,
    pub justification_id: String,
    pub cited_evidence_ids: BTreeSet<String>,
}

/// Reader context built only from an independently verified proof packet.
///
/// context_items are the reader-visible source excerpts. evidence and
/// claim_proofs are control-plane metadata and must not be silently serialized
/// as additional memory text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedProofReaderContext {
    pub decision_id: String,
    pub principal_id: String,
    pub scope_id: String,
    pub valid_at_unix_ms: u64,
    pub known_at_unix_ms: u64,
    pub request_sha256: String,
    pub packet_sha256: String,
    /// Canonical digest over this exact reader/control-plane context, including
    /// source session provenance and claim-proof dependencies.
    #[serde(default)]
    pub context_sha256: String,
    pub tokenizer_id: String,
    pub tokenizer_version: String,
    pub tokenizer_sha256: String,
    pub packet_renderer_sha256: String,
    pub fixed_packet_overhead_tokens: u64,
    pub total_token_cost: u64,
    pub token_budget: u64,
    pub context_items: Vec<MemoryContextItem>,
    pub evidence: Vec<VerifiedReaderEvidence>,
    pub claim_proofs: Vec<VerifiedReaderClaimProof>,
}

impl VerifiedProofReaderContext {
    /// Canonical integrity digest for the exact reader context. The digest
    /// field is blanked before serialization so the operation is stable.
    pub fn computed_context_sha256(&self) -> Result<String, String> {
        let mut canonical = self.clone();
        canonical.context_sha256.clear();
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| format!("cannot serialize verified reader context: {error}"))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    /// Borrow the exact reader context without changing its order or contents.
    pub fn memory_context_items(&self) -> &[MemoryContextItem] {
        &self.context_items
    }

    /// Require the caller to match the exact request identity and time coordinates.
    pub fn verify_request_binding(
        &self,
        principal_id: &str,
        scope_id: &str,
        valid_at_unix_ms: u64,
        known_at_unix_ms: u64,
    ) -> Result<(), String> {
        if self.principal_id != principal_id {
            return Err("reader context principal does not match request".into());
        }
        if self.scope_id != scope_id {
            return Err("reader context scope does not match request".into());
        }
        if self.valid_at_unix_ms != valid_at_unix_ms {
            return Err("reader context valid_at does not match request".into());
        }
        if self.known_at_unix_ms != known_at_unix_ms {
            return Err("reader context known_at does not match request".into());
        }
        Ok(())
    }
    /// Check each declared claim against the exact selected proof dependencies.
    ///
    /// This rejects unknown conclusions, wrong alternative proofs, omitted
    /// premises, added distractors and evidence absent from reader context.
    /// It does not establish that the natural-language claim follows from them.
    pub fn verify_claim_evidence_uses(
        &self,
        uses: &[ReaderClaimEvidenceUse],
    ) -> Result<(), String> {
        let proofs_by_conclusion: BTreeMap<&str, &VerifiedReaderClaimProof> = self
            .claim_proofs
            .iter()
            .map(|proof| (proof.conclusion_id.as_str(), proof))
            .collect();
        let context_evidence_ids: BTreeSet<&str> = self
            .evidence
            .iter()
            .map(|evidence| evidence.evidence_id.as_str())
            .collect();

        let mut seen_conclusions = BTreeSet::new();
        for usage in uses {
            if !seen_conclusions.insert(usage.conclusion_id.as_str()) {
                return Err(format!(
                    "duplicate evidence-use declarations for conclusion {:?}",
                    usage.conclusion_id
                ));
            }
            let proof = proofs_by_conclusion
                .get(usage.conclusion_id.as_str())
                .ok_or_else(|| format!(
                    "answer claim references unknown or unselected conclusion {:?}",
                    usage.conclusion_id
                ))?;
            if usage.justification_id != proof.justification_id {
                return Err(format!(
                    "answer claim for {:?} names justification {:?}, but packet selected {:?}",
                    usage.conclusion_id, usage.justification_id, proof.justification_id
                ));
            }

            let expected: BTreeSet<&str> = proof
                .evidence_dependencies
                .iter()
                .map(String::as_str)
                .collect();
            let cited: BTreeSet<&str> = usage
                .cited_evidence_ids
                .iter()
                .map(String::as_str)
                .collect();
            if cited != expected {
                return Err(format!(
                    "answer claim for {:?} must cite the complete exact selected dependency set",
                    usage.conclusion_id
                ));
            }
            if let Some(missing) = cited
                .iter()
                .find(|evidence_id| !context_evidence_ids.contains(*evidence_id))
            {
                return Err(format!(
                    "answer claim cites evidence {:?} absent from verified reader context",
                    missing
                ));
            }
        }
        Ok(())
    }

    /// Fail closed if the control-plane metadata and reader context ever drift.
    pub fn verify_internal_consistency(&self) -> Result<(), String> {
        fn valid_sha256(value: &str) -> bool {
            value.len() == 64
                && value.bytes().all(|byte| {
                    byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
                })
        }
        fn sha256(bytes: &[u8]) -> String {
            format!("{:x}", Sha256::digest(bytes))
        }

        for (field, value) in [
            ("request_sha256", self.request_sha256.as_str()),
            ("packet_sha256", self.packet_sha256.as_str()),
            ("context_sha256", self.context_sha256.as_str()),
            ("tokenizer_sha256", self.tokenizer_sha256.as_str()),
            ("packet_renderer_sha256", self.packet_renderer_sha256.as_str()),
        ] {
            if !valid_sha256(value) {
                return Err(format!("{field} must be a 64-character lowercase SHA-256 digest"));
            }
        }
        for (field, value) in [
            ("decision_id", self.decision_id.as_str()),
            ("principal_id", self.principal_id.as_str()),
            ("scope_id", self.scope_id.as_str()),
            ("tokenizer_id", self.tokenizer_id.as_str()),
            ("tokenizer_version", self.tokenizer_version.as_str()),
        ] {
            if value.trim().is_empty() || value.trim() != value {
                return Err(format!("{field} must be non-empty and trimmed"));
            }
        }
        if self.context_items.len() != self.evidence.len() {
            return Err(format!(
                "reader item count {} differs from evidence metadata count {}",
                self.context_items.len(),
                self.evidence.len()
            ));
        }
        if self.context_items.is_empty() {
            return Err("verified reader context must contain at least one evidence item".into());
        }
        if self.total_token_cost > self.token_budget {
            return Err("verified reader context exceeds its declared token budget".into());
        }

        let mut evidence_ids = BTreeSet::new();
        let mut total_cost = self.fixed_packet_overhead_tokens;
        let mut previous_evidence_id: Option<&str> = None;
        for (index, (item, evidence)) in self
            .context_items
            .iter()
            .zip(&self.evidence)
            .enumerate()
        {
            if item.kind != MemoryContextType::Text || item.value.trim().is_empty() {
                return Err(format!("reader evidence item {index} is not non-empty text"));
            }
            if sha256(item.value.as_bytes()) != evidence.source_span_sha256 {
                return Err(format!(
                    "reader evidence item {index} no longer matches its verified source-span digest"
                ));
            }
            if evidence.evidence_id.trim().is_empty()
                || !evidence_ids.insert(evidence.evidence_id.as_str())
            {
                return Err(format!("reader evidence item {index} has an empty or duplicate ID"));
            }
            if previous_evidence_id.is_some_and(|previous| previous >= evidence.evidence_id.as_str()) {
                return Err("reader evidence items are not in canonical ascending evidence-ID order".into());
            }
            previous_evidence_id = Some(evidence.evidence_id.as_str());
            if evidence.source_revision_id.trim().is_empty()
                || evidence.source_pointer.trim().is_empty()
                || evidence.source_family_id.trim().is_empty()
            {
                return Err(format!("reader evidence item {index} has incomplete provenance"));
            }
            if evidence.source_session_id.as_ref().is_some_and(|session_id| {
                session_id.trim().is_empty() || session_id.trim() != session_id
            }) {
                return Err(format!("reader evidence item {index} has invalid session provenance"));
            }
            if evidence.source_span_end <= evidence.source_span_start
                || evidence.source_span_end - evidence.source_span_start != item.value.len() as u64
            {
                return Err(format!("reader evidence item {index} has inconsistent UTF-8 byte offsets"));
            }
            if !valid_sha256(&evidence.payload_sha256)
                || !valid_sha256(&evidence.source_span_sha256)
                || !valid_sha256(&evidence.authorization_receipt_sha256)
            {
                return Err(format!("reader evidence item {index} has malformed provenance digest"));
            }
            if evidence.retrieval_rank == 0 || evidence.token_cost == 0 {
                return Err(format!("reader evidence item {index} has invalid retrieval rank or token cost"));
            }
            total_cost = total_cost
                .checked_add(u64::from(evidence.token_cost))
                .ok_or_else(|| "reader evidence token cost overflow".to_string())?;
        }

        let mut conclusions = BTreeSet::new();
        let mut previous_conclusion_id: Option<&str> = None;
        for proof in &self.claim_proofs {
            if proof.conclusion_id.trim().is_empty()
                || proof.justification_id.trim().is_empty()
                || !conclusions.insert(proof.conclusion_id.as_str())
            {
                return Err("claim proof metadata has an empty or duplicate conclusion".into());
            }
            if previous_conclusion_id.is_some_and(|previous| previous >= proof.conclusion_id.as_str()) {
                return Err("claim proof metadata is not in canonical ascending conclusion order".into());
            }
            previous_conclusion_id = Some(proof.conclusion_id.as_str());
            if proof.evidence_dependencies.is_empty() {
                return Err(format!("selected proof {:?} has no dependencies", proof.justification_id));
            }
            let mut dependencies = BTreeSet::new();
            for evidence_id in &proof.evidence_dependencies {
                if !evidence_ids.contains(evidence_id.as_str()) {
                    return Err(format!(
                        "selected proof {:?} references absent evidence {:?}",
                        proof.justification_id, evidence_id
                    ));
                }
                if !dependencies.insert(evidence_id.as_str()) {
                    return Err(format!(
                        "selected proof {:?} repeats evidence {:?}",
                        proof.justification_id, evidence_id
                    ));
                }
            }
            total_cost = total_cost
                .checked_add(u64::from(proof.selection_overhead_tokens))
                .ok_or_else(|| "reader proof-envelope token cost overflow".to_string())?;
        }

        if total_cost != self.total_token_cost {
            return Err(format!(
                "reader context token cost mismatch: metadata says {}, recomputed {}",
                self.total_token_cost, total_cost
            ));
        }
        let expected_context_sha256 = self.computed_context_sha256()?;
        if expected_context_sha256 != self.context_sha256 {
            return Err("verified reader context SHA-256 mismatch".into());
        }
        Ok(())
    }
}

/// Independently verify a proof result and construct its exact reader context.
/// A non-Supported, insufficient, non-optimal, tampered or over-budget packet
/// is rejected before any reader-visible item is returned.
pub fn build_verified_proof_reader_context(
    request: &ProofBudgetRequest,
    result: &ProofBudgetResult,
) -> Result<VerifiedProofReaderContext, String> {
    if result.status != ProofBudgetStatus::Supported {
        return Err(format!("cannot build reader context from proof status {:?}", result.status));
    }
    verify_proof_budget_result(request, result)?;
    if result.evidence.is_empty() || result.selected_proofs.is_empty() {
        return Err("Supported packet must contain evidence and selected proofs".into());
    }
    if result.total_token_cost > request.token_budget {
        return Err("verified packet exceeds request token budget".into());
    }

    let context_items = result
        .evidence
        .iter()
        .map(|evidence| MemoryContextItem {
            kind: MemoryContextType::Text,
            value: evidence.source_span_text.clone(),
        })
        .collect();
    let evidence = result
        .evidence
        .iter()
        .map(|evidence| VerifiedReaderEvidence {
            evidence_id: evidence.evidence_id.clone(),
            source_revision_id: evidence.source_revision_id.clone(),
            source_pointer: evidence.source_pointer.clone(),
            source_span_start: evidence.source_span_start,
            source_span_end: evidence.source_span_end,
            source_family_id: evidence.source_family_id.clone(),
            source_session_id: evidence.source_session_id.clone(),
            source_speaker: evidence.source_speaker,
            payload_sha256: evidence.payload_sha256.clone(),
            source_span_sha256: evidence.source_span_sha256.clone(),
            authorization_receipt_sha256: evidence.authorization_receipt_sha256.clone(),
            retrieval_rank: evidence.retrieval_rank,
            token_cost: evidence.token_cost,
        })
        .collect();
    let claim_proofs = result
        .selected_proofs
        .iter()
        .map(|proof: &SelectedProof| VerifiedReaderClaimProof {
            conclusion_id: proof.conclusion_id.clone(),
            justification_id: proof.justification_id.clone(),
            selection_overhead_tokens: proof.selection_overhead_tokens,
            evidence_dependencies: proof.evidence_dependencies.clone(),
        })
        .collect();

    let mut context = VerifiedProofReaderContext {
        decision_id: request.decision_id.clone(),
        principal_id: request.principal_id.clone(),
        scope_id: request.scope_id.clone(),
        valid_at_unix_ms: request.valid_at_unix_ms,
        known_at_unix_ms: request.known_at_unix_ms,
        request_sha256: result.request_sha256.clone(),
        packet_sha256: result
            .packet_sha256
            .clone()
            .ok_or_else(|| "Supported packet is missing packet_sha256".to_string())?,
        context_sha256: String::new(),
        tokenizer_id: request.tokenizer_id.clone(),
        tokenizer_version: request.tokenizer_version.clone(),
        tokenizer_sha256: request.tokenizer_sha256.clone(),
        packet_renderer_sha256: request.packet_renderer_sha256.clone(),
        fixed_packet_overhead_tokens: request.fixed_packet_overhead_tokens,
        total_token_cost: result.total_token_cost,
        token_budget: request.token_budget,
        context_items,
        evidence,
        claim_proofs,
    };
    context.context_sha256 = context.computed_context_sha256()?;
    context.verify_internal_consistency()?;
    Ok(context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_proof_budget::{
        ProofBudgetRequest, ProofEvidenceCandidate, ProofJustification, ProofObligation,
        solve_proof_budget,
    };
    use sha2::{Digest, Sha256};

    fn digest(value: &str) -> String {
        format!("{:x}", Sha256::digest(value.as_bytes()))
    }

    fn candidate(id: &str, token_cost: u32) -> ProofEvidenceCandidate {
        let excerpt = format!("source excerpt {id}");
        let span_sha256 = digest(&excerpt);
        ProofEvidenceCandidate {
            evidence_id: id.into(),
            source_revision_id: format!("revision-{id}"),
            source_pointer: format!("/messages/{id}/content"),
            source_span_text: excerpt.clone(),
            source_span_start: 0,
            source_span_end: excerpt.len() as u64,
            source_family_id: format!("family-{id}"),
            source_session_id: Some(format!("session-{id}")),
            source_speaker: Some(EvidenceSpeakerRole::User),
            scope_id: "scope-1".into(),
            payload_sha256: digest("payload"),
            source_span_sha256: span_sha256.clone(),
            resolved_span_sha256: span_sha256,
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
            candidates: vec![candidate("A", 2), candidate("B", 3)],
            obligations: vec![ProofObligation {
                conclusion_id: "claim-1".into(),
                alternatives: vec![
                    ProofJustification {
                        justification_id: "proof-A".into(),
                        evidence_dependencies: vec!["A".into()],
                    },
                    ProofJustification {
                        justification_id: "proof-B".into(),
                        evidence_dependencies: vec!["B".into()],
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
    fn reader_context_is_built_only_from_independently_verified_packet() {
        let req = request();
        let result = solve_proof_budget(req.clone()).unwrap();
        let context = build_verified_proof_reader_context(&req, &result).unwrap();

        assert_eq!(context.context_items.len(), 1);
        assert_eq!(context.context_items[0].kind, MemoryContextType::Text);
        assert_eq!(context.context_items[0].value, "source excerpt A");
        assert_eq!(context.evidence[0].evidence_id, "A");
        assert_eq!(context.claim_proofs[0].justification_id, "proof-A");
        assert_eq!(context.total_token_cost, 7);
        context.verify_internal_consistency().unwrap();
    }

    #[test]
    fn verified_reader_context_retains_exact_cross_session_proof_coverage() {
        let mut req = request();
        req.obligations[0].alternatives = vec![ProofJustification {
            justification_id: "cross-session-proof".into(),
            evidence_dependencies: vec!["A".into(), "B".into()],
        }];
        req.obligations[0].minimum_independent_families = 2;
        req.obligations[0].minimum_distinct_sessions = 2;

        let result = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);

        let context = build_verified_proof_reader_context(&req, &result).unwrap();
        assert_eq!(
            context.evidence
                .iter()
                .map(|evidence| evidence.source_session_id.as_deref().unwrap())
                .collect::<Vec<_>>(),
            vec!["session-A", "session-B"]
        );
        assert_eq!(
            context.claim_proofs[0].evidence_dependencies,
            vec!["A", "B"]
        );
        context.verify_internal_consistency().unwrap();
    }


    #[test]
    fn reader_context_digest_detects_session_provenance_mutation() {
        let mut req = request();
        req.obligations[0].alternatives = vec![ProofJustification {
            justification_id: "cross-session-proof".into(),
            evidence_dependencies: vec!["A".into(), "B".into()],
        }];
        req.obligations[0].minimum_independent_families = 2;
        req.obligations[0].minimum_distinct_sessions = 2;
        let result = solve_proof_budget(req.clone()).unwrap();
        let mut context = build_verified_proof_reader_context(&req, &result).unwrap();
        context.evidence[0].source_session_id = Some("forged-session".into());

        assert!(context
            .verify_internal_consistency()
            .unwrap_err()
            .contains("reader context SHA-256 mismatch"));
    }

    #[test]
    fn verified_reader_context_preserves_user_and_assistant_speaker_provenance() {
        let mut req = request();
        req.obligations[0].alternatives = vec![ProofJustification {
            justification_id: "mixed-speaker-proof".into(),
            evidence_dependencies: vec!["A".into(), "B".into()],
        }];
        req.obligations[0].minimum_user_authored_evidence = 1;
        req.candidates[1].source_speaker = Some(EvidenceSpeakerRole::Assistant);

        let result = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        let context = build_verified_proof_reader_context(&req, &result).unwrap();

        let speakers: BTreeMap<&str, Option<EvidenceSpeakerRole>> = context.evidence
            .iter()
            .map(|item| (item.evidence_id.as_str(), item.source_speaker))
            .collect();
        assert_eq!(speakers["A"], Some(EvidenceSpeakerRole::User));
        assert_eq!(speakers["B"], Some(EvidenceSpeakerRole::Assistant));
        context.verify_internal_consistency().unwrap();
    }

    #[test]
    fn reader_context_requires_matching_request_identity_and_time() {
        let req = request();
        let result = solve_proof_budget(req.clone()).unwrap();
        let context = build_verified_proof_reader_context(&req, &result).unwrap();
        context.verify_request_binding("principal-1", "scope-1", 200, 200).unwrap();
        assert!(context.verify_request_binding("other-principal", "scope-1", 200, 200).is_err());
        assert!(context.verify_request_binding("principal-1", "other-scope", 200, 200).is_err());
        assert!(context.verify_request_binding("principal-1", "scope-1", 201, 200).is_err());
        assert!(context.verify_request_binding("principal-1", "scope-1", 200, 199).is_err());
    }
    #[test]
    fn reader_context_rejects_non_supported_search_results() {
        let mut req = request();
        req.max_search_nodes = 1;
        let result = solve_proof_budget(req.clone()).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::SearchLimitReached);
        assert!(build_verified_proof_reader_context(&req, &result)
            .unwrap_err()
            .contains("SearchLimitReached"));
    }

    #[test]
    fn answer_claim_must_cite_exact_selected_and_dependencies() {
        let req = request();
        let result = solve_proof_budget(req.clone()).unwrap();
        let context = build_verified_proof_reader_context(&req, &result).unwrap();

        let valid_use = ReaderClaimEvidenceUse {
            conclusion_id: "claim-1".into(),
            justification_id: "proof-A".into(),
            cited_evidence_ids: BTreeSet::from(["A".into()]),
        };
        context.verify_claim_evidence_uses(&[valid_use.clone()]).unwrap();

        let missing_premise = ReaderClaimEvidenceUse {
            cited_evidence_ids: BTreeSet::new(),
            ..valid_use.clone()
        };
        assert!(context.verify_claim_evidence_uses(&[missing_premise]).is_err());

        let wrong_alternative = ReaderClaimEvidenceUse {
            justification_id: "proof-B".into(),
            ..valid_use.clone()
        };
        assert!(context.verify_claim_evidence_uses(&[wrong_alternative]).is_err());

        let invented_claim = ReaderClaimEvidenceUse {
            conclusion_id: "not-required".into(),
            ..valid_use
        };
        assert!(context.verify_claim_evidence_uses(&[invented_claim]).is_err());
    }

    #[test]
    fn reader_context_detects_excerpt_mutation_after_packet_verification() {
        let req = request();
        let result = solve_proof_budget(req.clone()).unwrap();
        let mut context = build_verified_proof_reader_context(&req, &result).unwrap();
        context.context_items[0].value.push_str(" tampered");
        assert!(context
            .verify_internal_consistency()
            .unwrap_err()
            .contains("source-span digest"));
    }

    #[test]
    fn reader_context_recomputes_full_token_cost_after_serialization_boundary() {
        let req = request();
        let result = solve_proof_budget(req.clone()).unwrap();
        let mut context = build_verified_proof_reader_context(&req, &result).unwrap();
        context.total_token_cost += 1;
        assert!(context
            .verify_internal_consistency()
            .unwrap_err()
            .contains("token cost mismatch"));
    }

    #[test]
    fn context_never_truncates_or_changes_packet_evidence_order() {
        let mut req = request();
        req.obligations[0].alternatives = vec![ProofJustification {
            justification_id: "proof-AB".into(),
            evidence_dependencies: vec!["B".into(), "A".into()],
        }];
        let result = solve_proof_budget(req.clone()).unwrap();
        let context = build_verified_proof_reader_context(&req, &result).unwrap();

        assert_eq!(
            context.evidence.iter().map(|item| item.evidence_id.as_str()).collect::<Vec<_>>(),
            vec!["A", "B"]
        );
        assert_eq!(context.context_items.len(), 2);
        assert_eq!(context.context_items[0].value, "source excerpt A");
        assert_eq!(context.context_items[1].value, "source excerpt B");
        assert_eq!(
            context.claim_proofs[0].evidence_dependencies,
            vec!["A".to_string(), "B".to_string()]
        );
    }
}
