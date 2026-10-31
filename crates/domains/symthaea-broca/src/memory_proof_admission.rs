// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Admission boundary from lexical retrieval candidates to exact proof search.
//!
//! BM25 is a broad candidate generator, not an authority source. It ranks text
//! records; it does not establish source revision identity, historical
//! transaction time, authorization, contradiction edges, or exact rendered
//! token cost. Those values must arrive through a separately trusted admission
//! envelope. This module joins the two without inventing missing provenance.
//!
//! The resulting ProofBudgetRequest is still validated by the proof solver.
//! A failed/false authorization decision is preserved as an ineligible
//! candidate so the solver can report the rejection; it is never promoted into
//! usable evidence. A caller must bind the lexical index snapshot to the same
//! known_at coordinate as the request or this adapter refuses the batch.

use crate::memory_evidence::{
    evidence_speaker_role_from_step_id, EvidenceIndex, EvidenceIndexSnapshotReceipt,
    EvidenceLifecycle, ExactLexicalEnumerationResult, LexicalEvidenceHit, MemoryContextType,
};
use crate::memory_proof_budget::{
    ProofBudgetRequest, ProofEvidenceCandidate,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Trusted admission metadata for one lexical hit.
///
/// The authorizer must verify the underlying authorization evidence before
/// setting authorization_accepted. The source resolver / bitemporal store must
/// supply revision identity, source-family identity and transaction timestamps.
/// The tokenizer/renderer must supply token_cost for the exact final evidence
/// block. BM25 rank is deliberately supplied by the hit order, not this record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofCandidateAdmission {
    pub evidence_id: String,
    pub source_revision_id: String,
    /// Exact source identity fields copied from the immutable EvidenceIndex hit.
    pub source_trajectory_id: String,
    pub source_step_id: Option<String>,
    pub source_step_index: Option<u64>,
    pub source_sha256: String,
    pub source_pointer: String,
    pub source_family_id: String,
    pub scope_id: String,
    /// Digest of the exact EvidenceIndex state that produced this hit.
    pub index_state_sha256: String,
    /// Transaction time when this immutable source revision was admitted.
    pub recorded_at_unix_ms: u64,
    /// Transaction time when this revision was retracted, if any.
    pub retracted_at_unix_ms: Option<u64>,
    /// Inclusive beginning of world-time validity.
    pub valid_from_unix_ms: Option<u64>,
    /// Exclusive end of world-time validity.
    pub valid_until_unix_ms: Option<u64>,
    pub authorized_principal_id: String,
    /// Set only by a trusted authorization adapter after policy evaluation.
    pub authorization_accepted: bool,
    /// Digest of the authorization decision receipt; not a signature itself.
    pub authorization_receipt_sha256: String,
    /// Symmetric contradiction edges from the trusted contradiction resolver.
    pub conflicts_with: BTreeSet<String>,
    /// Exact token cost from the pinned packet renderer and tokenizer.
    pub token_cost: u32,
}

/// Join BM25 hits with trusted admission envelopes to populate a proof request.
///
/// Requirements:
/// - request.candidates must be empty; candidate replacement is never silent;
/// - index_snapshot_known_at must exactly equal request.known_at;
/// - every hit has one and only one admission, and no unused admission exists;
/// - each hit must be non-empty text whose payload digest matches the immutable
///   EvidenceIndex content digest;
/// - source revision, family, scope, transaction time, authorization and cost
///   are copied only from the admission envelope;
/// - BM25 result order becomes one-based retrieval_rank.
///
/// The adapter intentionally does not infer a bitemporal revision from a
/// trajectory ID, trust an evidence_id as proof of source authenticity, or
/// estimate tokens from character counts. The caller still has to ensure the
/// EvidenceIndex is the exact immutable snapshot represented by the supplied
/// known_at coordinate.
pub fn admit_lexical_hits_for_proof_budget(
    mut request: ProofBudgetRequest,
    hits: &[LexicalEvidenceHit],
    admissions: &[ProofCandidateAdmission],
    snapshot: &EvidenceIndexSnapshotReceipt,
) -> Result<ProofBudgetRequest, String> {
    if !request.candidates.is_empty() {
        return Err("proof request candidates must be empty before lexical admission".into());
    }
    if hits.is_empty() {
        return Err("lexical retrieval returned no evidence candidates".into());
    }
    if snapshot.known_at_unix_ms != request.known_at_unix_ms {
        return Err(format!(
            "lexical index snapshot known_at {} does not match proof request known_at {}",
            snapshot.known_at_unix_ms, request.known_at_unix_ms
        ));
    }
    if snapshot.index_state_sha256.len() != 64
        || !snapshot.index_state_sha256.bytes().all(|byte| {
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
        })
    {
        return Err("lexical index snapshot has a malformed state digest".into());
    }

    let mut admission_by_id = BTreeMap::<&str, &ProofCandidateAdmission>::new();
    for admission in admissions {
        if admission.evidence_id.trim().is_empty() {
            return Err("admission evidence_id must be non-empty".into());
        }
        if admission_by_id
            .insert(admission.evidence_id.as_str(), admission)
            .is_some()
        {
            return Err(format!(
                "duplicate admission for evidence ID {:?}",
                admission.evidence_id
            ));
        }
        if admission.index_state_sha256 != snapshot.index_state_sha256 {
            return Err(format!(
                "admission {:?} is bound to a different lexical index state",
                admission.evidence_id
            ));
        }
        if admission.scope_id != request.scope_id {
            return Err(format!(
                "admission {:?} scope does not match the proof request",
                admission.evidence_id
            ));
        }
    }

    let mut seen_hits = BTreeSet::new();
    let mut seen_hdc_ids = BTreeSet::new();
    let mut candidates = Vec::with_capacity(hits.len());
    for (index, hit) in hits.iter().enumerate() {
        if hit.context.kind != MemoryContextType::Text {
            return Err(format!("lexical hit {} is not text evidence", index + 1));
        }
        if hit.evidence_id.trim().is_empty() || hit.context.value.trim().is_empty() {
            return Err(format!("lexical hit {} has an empty ID or payload", index + 1));
        }
        if !seen_hits.insert(hit.evidence_id.as_str()) {
            return Err(format!("duplicate lexical hit {:?}", hit.evidence_id));
        }
        if !seen_hdc_ids.insert(hit.hdc_id) {
            return Err(format!("duplicate HDC candidate ID {}", hit.hdc_id));
        }
        if !hit.bm25_score.is_finite() || hit.bm25_score <= 0.0 {
            return Err(format!("lexical hit {:?} has an invalid BM25 score", hit.evidence_id));
        }
        let admission = admission_by_id
            .get(hit.evidence_id.as_str())
            .ok_or_else(|| format!(
                "lexical hit {:?} has no trusted admission envelope",
                hit.evidence_id
            ))?;

        if hit.lifecycle != EvidenceLifecycle::Active {
            return Err(format!(
                "lexical hit {:?} is not active in its index snapshot",
                hit.evidence_id
            ));
        }
        if hit.snapshot_receipt.as_ref() != Some(snapshot) {
            return Err(format!(
                "lexical hit {:?} is not bound to the supplied index snapshot receipt",
                hit.evidence_id
            ));
        }
        if hit.scope_id != request.scope_id || hit.scope_id != admission.scope_id {
            return Err(format!(
                "lexical hit {:?} scope does not match its request/admission",
                hit.evidence_id
            ));
        }
        if hit.valid_from_unix_ms != admission.valid_from_unix_ms
            || hit.valid_until_unix_ms != admission.valid_until_unix_ms
        {
            return Err(format!(
                "lexical hit {:?} validity interval does not match its admission",
                hit.evidence_id
            ));
        }

        if hit.source.trajectory_id.trim().is_empty()
            || !hit.source.source_pointer.starts_with('/')
            || hit.source.source_pointer.trim() == "/"
            || hit.source.source_sha256.len() != 64
            || !hit.source.source_sha256.bytes().all(|byte| {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            })
        {
            return Err(format!(
                "lexical hit {:?} has incomplete source provenance",
                hit.evidence_id
            ));
        }

        let resolved_digest = sha256_hex(hit.context.value.as_bytes());
        if hit.content_sha256 != resolved_digest {
            return Err(format!(
                "lexical hit {:?} payload digest does not match returned text",
                hit.evidence_id
            ));
        }
        if hit.content_sha256.len() != 64
            || !hit.content_sha256.bytes().all(|byte| {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            })
        {
            return Err(format!(
                "lexical hit {:?} has a malformed payload digest",
                hit.evidence_id
            ));
        }

        if admission.source_revision_id.trim().is_empty()
            || admission.source_family_id.trim().is_empty()
            || admission.authorized_principal_id.trim().is_empty()
        {
            return Err(format!(
                "admission {:?} is missing revision, family or principal identity",
                admission.evidence_id
            ));
        }
        if admission.source_trajectory_id != hit.source.trajectory_id
            || admission.source_step_id != hit.source.step_id
            || admission.source_step_index != hit.source.step_index
            || admission.source_sha256 != hit.source.source_sha256
            || admission.source_pointer != hit.source.source_pointer
        {
            return Err(format!(
                "admission {:?} source identity does not match the immutable lexical hit",
                admission.evidence_id
            ));
        }
        if admission.token_cost == 0 {
            return Err(format!(
                "admission {:?} has no exact rendered token cost",
                admission.evidence_id
            ));
        }
        if admission.valid_from_unix_ms
            .zip(admission.valid_until_unix_ms)
            .is_some_and(|(start, end)| end <= start)
        {
            return Err(format!(
                "admission {:?} has an invalid world-time interval",
                admission.evidence_id
            ));
        }
        if admission.retracted_at_unix_ms
            .is_some_and(|retracted| retracted < admission.recorded_at_unix_ms)
        {
            return Err(format!(
                "admission {:?} has retracted_at before recorded_at",
                admission.evidence_id
            ));
        }
        if admission.authorization_receipt_sha256.len() != 64
            || !admission.authorization_receipt_sha256.bytes().all(|byte| {
                byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)
            })
        {
            return Err(format!(
                "admission {:?} has a malformed authorization receipt digest",
                admission.evidence_id
            ));
        }

        let span_len = hit.context.value.len() as u64;
        if span_len == 0 {
            return Err(format!("lexical hit {:?} has an empty source span", hit.evidence_id));
        }

        candidates.push(ProofEvidenceCandidate {
            evidence_id: hit.evidence_id.clone(),
            source_revision_id: admission.source_revision_id.clone(),
            source_pointer: hit.source.source_pointer.clone(),
            source_span_text: hit.context.value.clone(),
            source_span_start: 0,
            source_span_end: span_len,
            source_family_id: admission.source_family_id.clone(),
            // Session identity is derived from the exact source trajectory
            // returned by the index and matched against the trusted admission.
            // AML ingestion sets trajectory_id directly from Add.session_id.
            source_session_id: Some(admission.source_trajectory_id.clone()),
            // Speaker role is derived from the exact source step ID returned
            // by the snapshot-bound EvidenceIndex hit, not a caller label.
            source_speaker: evidence_speaker_role_from_step_id(hit.source.step_id.as_deref()),
            scope_id: admission.scope_id.clone(),
            payload_sha256: hit.content_sha256.clone(),
            source_span_sha256: resolved_digest.clone(),
            resolved_span_sha256: resolved_digest,
            conflicts_with: admission.conflicts_with.clone(),
            recorded_at_unix_ms: admission.recorded_at_unix_ms,
            retracted_at_unix_ms: admission.retracted_at_unix_ms,
            valid_from_unix_ms: admission.valid_from_unix_ms,
            valid_until_unix_ms: admission.valid_until_unix_ms,
            authorized_principal_id: admission.authorized_principal_id.clone(),
            authorization_accepted: admission.authorization_accepted,
            authorization_receipt_sha256: admission.authorization_receipt_sha256.clone(),
            retrieval_rank: u32::try_from(index + 1)
                .map_err(|_| "retrieval rank exceeds u32".to_string())?,
            token_cost: admission.token_cost,
        });
    }

    if admission_by_id.len() != seen_hits.len() {
        let unused: Vec<&str> = admission_by_id
            .keys()
            .copied()
            .filter(|evidence_id| !seen_hits.contains(evidence_id))
            .collect();
        return Err(format!(
            "unused admission envelopes are not allowed: {:?}",
            unused
        ));
    }

    request.candidates = candidates;
    Ok(request)
}

/// Admit every exact match from a verified enumeration and bind the complete
/// match set to one required proof obligation. The index re-verifies the
/// enumeration and materializes every hit; the existing trusted admission
/// envelopes then bind source identity, authorization, lifecycle and token cost.
///
/// This is the supported path for count / all-instances questions. A zero-match
/// enumeration is refused until the system has a dedicated proof-of-absence
/// capability.
pub fn admit_complete_enumeration_for_proof_budget(
    request: ProofBudgetRequest,
    index: &EvidenceIndex,
    enumeration: &ExactLexicalEnumerationResult,
    admissions: &[ProofCandidateAdmission],
    payload_root: &std::path::Path,
    conclusion_id: &str,
    justification_id: &str,
) -> Result<ProofBudgetRequest, String> {
    if enumeration.matches.is_empty() {
        return Err(
            "empty enumeration cannot support an answer until proof-of-absence is implemented"
                .into(),
        );
    }
    if request.scope_id != enumeration.receipt.scope_id {
        return Err("enumeration scope does not match proof request".into());
    }
    if request.valid_at_unix_ms != enumeration.receipt.valid_at_unix_ms {
        return Err("enumeration valid_at does not match proof request".into());
    }
    if request.known_at_unix_ms != enumeration.receipt.known_at_unix_ms {
        return Err("enumeration known_at does not match proof request".into());
    }

    let hits = index.materialize_exact_lexical_enumeration_hits(
        enumeration,
        payload_root,
    )?;
    let snapshot = hits
        .first()
        .and_then(|hit| hit.snapshot_receipt.as_ref())
        .cloned()
        .ok_or_else(|| "exact enumeration materialized no snapshot-bound evidence hits".to_string())?;

    let mut admitted = admit_lexical_hits_for_proof_budget(
        request,
        &hits,
        admissions,
        &snapshot,
    )?;
    let obligation = admitted.obligations.iter_mut()
        .find(|obligation| obligation.conclusion_id == conclusion_id)
        .ok_or_else(|| format!("unknown proof conclusion {conclusion_id:?}"))?;
    obligation.bind_verified_complete_enumeration(
        index,
        enumeration,
        justification_id,
    )?;
    Ok(admitted)
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_evidence::{
        EvidenceIndex, EvidenceIndexSnapshotReceipt, EvidenceLifecycle, EvidencePayload,
        EvidenceRecord, EvidenceSource, LexicalEvidenceHit, MemoryContextItem, Bm25Config,
    };
    use std::path::Path;
    use crate::memory_proof_budget::{
        ProofBudgetStatus, ProofCandidateRejection, ProofJustification, ProofObligation,
        solve_proof_budget,
    };

    fn digest(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    fn hit(id: &str, text: &str, rank_score: f64) -> LexicalEvidenceHit {
        LexicalEvidenceHit {
            hdc_id: id.as_bytes().first().copied().map(u64::from).unwrap_or(0),
            evidence_id: id.into(),
            scope_id: "scope-1".into(),
            source: EvidenceSource {
                trajectory_id: "trajectory-1".into(),
                step_id: Some("step-2".into()),
                step_index: Some(2),
                source_sha256: digest("canonical source"),
                source_pointer: "/messages/2/content".into(),
            },
            content_sha256: digest(text),
            lifecycle: EvidenceLifecycle::Active,
            valid_from_unix_ms: Some(50),
            valid_until_unix_ms: Some(500),
            snapshot_receipt: Some(snapshot_receipt()),
            bm25_score: rank_score,
            context: MemoryContextItem {
                kind: MemoryContextType::Text,
                value: text.into(),
            },
        }
    }

    fn admission(id: &str) -> ProofCandidateAdmission {
        ProofCandidateAdmission {
            evidence_id: id.into(),
            source_revision_id: format!("revision-{id}"),
            source_trajectory_id: "trajectory-1".into(),
            source_step_id: Some("step-2".into()),
            source_step_index: Some(2),
            source_sha256: digest("canonical source"),
            source_pointer: "/messages/2/content".into(),
            source_family_id: format!("family-{id}"),
            scope_id: "scope-1".into(),
            index_state_sha256: digest("index state"),
            recorded_at_unix_ms: 100,
            retracted_at_unix_ms: None,
            valid_from_unix_ms: Some(50),
            valid_until_unix_ms: Some(500),
            authorized_principal_id: "principal-1".into(),
            authorization_accepted: true,
            authorization_receipt_sha256: digest("authorization receipt"),
            conflicts_with: BTreeSet::new(),
            token_cost: 5,
        }
    }

    fn snapshot_receipt() -> EvidenceIndexSnapshotReceipt {
        EvidenceIndexSnapshotReceipt {
            known_at_unix_ms: 200,
            index_state_sha256: digest("index state"),
        }
    }

    fn request() -> ProofBudgetRequest {
        ProofBudgetRequest {
            decision_id: "decision-1".into(),
            principal_id: "principal-1".into(),
            scope_id: "scope-1".into(),
            valid_at_unix_ms: 250,
            known_at_unix_ms: 200,
            policy_id: "policy-v1".into(),
            policy_sha256: digest("policy"),
            fixed_packet_overhead_tokens: 2,
            token_budget: 64,
            max_search_nodes: 1000,
            tokenizer_id: "pinned-tokenizer".into(),
            tokenizer_version: "1".into(),
            tokenizer_sha256: digest("tokenizer"),
            packet_renderer_sha256: digest("renderer"),
            candidates: Vec::new(),
            obligations: vec![ProofObligation {
                conclusion_id: "claim-1".into(),
                alternatives: vec![ProofJustification {
                    justification_id: "proof-A".into(),
                    evidence_dependencies: vec!["A".into()],
                }],
                minimum_independent_families: 1,
                minimum_distinct_sessions: 0,
                minimum_user_authored_evidence: 0,
                required_complete_evidence_ids: BTreeSet::new(),
                enumeration_receipt_sha256: None,
                selection_overhead_tokens: 3,
            }],
        }
    }

    fn indexed_record(text: &str) -> (EvidenceRecord, Vec<u8>) {
        let source_bytes = serde_json::to_vec(&serde_json::json!({
            "messages": [{"content": text}]
        })).unwrap();
        let record = EvidenceRecord {
            schema_version: crate::memory_evidence::EVIDENCE_SCHEMA_VERSION.to_string(),
            evidence_id: "A".into(),
            hdc_id: 65,
            scope_id: "scope-1".into(),
            source: EvidenceSource {
                trajectory_id: "trajectory-1".into(),
                step_id: Some("step-2".into()),
                step_index: Some(2),
                source_sha256: digest_bytes(&source_bytes),
                source_pointer: "/messages/0/content".into(),
            },
            payload: EvidencePayload::Text { value: text.into() },
            content_sha256: digest(text),
            lifecycle: EvidenceLifecycle::Active,
            occurred_at_unix_ms: Some(100),
            valid_from_unix_ms: Some(50),
            valid_until_unix_ms: Some(500),
            superseded_by: None,
        };
        (record, source_bytes)
    }

    fn digest_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[test]
    fn lexical_hits_join_only_with_exact_snapshot_admissions() {
        let proof_request = admit_lexical_hits_for_proof_budget(
            request(),
            &[hit("A", "verified evidence excerpt", 1.0)],
            &[admission("A")],
            &snapshot_receipt(),
        ).unwrap();

        assert_eq!(proof_request.candidates.len(), 1);
        let candidate = &proof_request.candidates[0];
        assert_eq!(candidate.source_revision_id, "revision-A");
        assert_eq!(candidate.source_span_text, "verified evidence excerpt");
        assert_eq!(candidate.retrieval_rank, 1);
        assert_eq!(candidate.recorded_at_unix_ms, 100);
        assert_eq!(candidate.token_cost, 5);

        let result = solve_proof_budget(proof_request).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(result.total_token_cost, 10);
    }

    #[test]
    fn actual_bm25_index_hits_join_to_a_state_bound_proof_request() {
        let text = "verified evidence excerpt about project funding";
        let (record, source_bytes) = indexed_record(text);
        let mut index = EvidenceIndex::new();
        index.insert(record, &source_bytes).unwrap();

        let (snapshot, hits) = index.search_bm25_at_snapshot(
            "verified evidence",
            "scope-1",
            250,
            200,
            Path::new("."),
            Bm25Config::default(),
        ).unwrap();
        assert_eq!(hits.len(), 1);

        let mut admission = admission("A");
        admission.source_trajectory_id = hits[0].source.trajectory_id.clone();
        admission.source_step_id = hits[0].source.step_id.clone();
        admission.source_step_index = hits[0].source.step_index;
        admission.source_sha256 = hits[0].source.source_sha256.clone();
        admission.source_pointer = hits[0].source.source_pointer.clone();
        admission.index_state_sha256 = snapshot.index_state_sha256.clone();
        let proof_request = admit_lexical_hits_for_proof_budget(
            request(),
            &hits,
            &[admission],
            &snapshot,
        ).unwrap();
        let result = solve_proof_budget(proof_request).unwrap();

        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(result.evidence[0].source_span_text, text);
    }

    #[test]
    fn trusted_session_provenance_flows_from_admission_into_multisession_proof_packet() {
        let mut proof_request = request();
        proof_request.obligations[0].minimum_independent_families = 2;
        proof_request.obligations[0].minimum_distinct_sessions = 2;
        proof_request.obligations[0].alternatives = vec![ProofJustification {
            justification_id: "two-session-proof".into(),
            evidence_dependencies: vec!["A".into(), "B".into()],
        }];

        let mut hit_a = hit("A", "evidence from first session", 1.0);
        hit_a.source.trajectory_id = "session-1".into();
        let mut hit_b = hit("B", "evidence from second session", 0.9);
        hit_b.source.trajectory_id = "session-2".into();

        // AML ingestion binds trajectory_id to the Add request's session_id.
        // The trusted admission envelope must preserve that exact mapping.
        let mut admission_a = admission("A");
        admission_a.source_trajectory_id = hit_a.source.trajectory_id.clone();
        let mut admission_b = admission("B");
        admission_b.source_trajectory_id = "session-2".into();
        admission_b.source_sha256 = hit_b.source.source_sha256.clone();
        admission_b.source_pointer = hit_b.source.source_pointer.clone();
        admission_b.source_step_id = hit_b.source.step_id.clone();
        admission_b.source_step_index = hit_b.source.step_index;

        let admitted = admit_lexical_hits_for_proof_budget(
            proof_request,
            &[hit_a, hit_b],
            &[admission_a, admission_b],
            &snapshot_receipt(),
        ).unwrap();
        let result = solve_proof_budget(admitted).unwrap();

        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(
            result.evidence
                .iter()
                .map(|evidence| evidence.source_session_id.as_deref().unwrap())
                .collect::<Vec<_>>(),
            vec!["session-1", "session-2"]
        );
    }

    #[test]
    fn session_provenance_is_derived_from_indexed_trajectory_not_admission_guess() {
        let mut proof_request = request();
        proof_request.obligations[0].minimum_distinct_sessions = 1;
        let indexed_hit = hit("A", "evidence", 1.0);
        let mut mismatched_admission = admission("A");
        mismatched_admission.source_trajectory_id = "forged-other-session".into();

        let error = admit_lexical_hits_for_proof_budget(
            proof_request.clone(),
            &[indexed_hit],
            &[mismatched_admission],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(error.contains("trajectory"));

        let admitted = admit_lexical_hits_for_proof_budget(
            proof_request,
            &[hit("A", "evidence", 1.0)],
            &[admission("A")],
            &snapshot_receipt(),
        ).unwrap();
        let result = solve_proof_budget(admitted).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::Supported);
        assert_eq!(result.evidence[0].source_session_id.as_deref(), Some("trajectory-1"));
    }

    #[test]
    fn admission_derives_speaker_from_snapshot_bound_source_step_and_enforces_user_threshold() {
        for (step_id, expected_role, expected_status) in [
            (
                "user-0",
                Some(crate::memory_evidence::EvidenceSpeakerRole::User),
                ProofBudgetStatus::Supported,
            ),
            (
                "assistant-1",
                Some(crate::memory_evidence::EvidenceSpeakerRole::Assistant),
                ProofBudgetStatus::NoEligibleProof,
            ),
            (
                "turn-2",
                None,
                ProofBudgetStatus::NoEligibleProof,
            ),
        ] {
            let mut proof_request = request();
            proof_request.obligations[0].minimum_user_authored_evidence = 1;

            let mut source_hit = hit("A", "invoice paid by user", 1.0);
            source_hit.source.step_id = Some(step_id.into());
            let mut trusted_admission = admission("A");
            trusted_admission.source_step_id = Some(step_id.into());

            let admitted = admit_lexical_hits_for_proof_budget(
                proof_request,
                &[source_hit],
                &[trusted_admission],
                &snapshot_receipt(),
            ).unwrap();
            assert_eq!(admitted.candidates[0].source_speaker, expected_role);

            let result = solve_proof_budget(admitted).unwrap();
            assert_eq!(result.status, expected_status, "source step {step_id}");
            if expected_status == ProofBudgetStatus::NoEligibleProof {
                assert!(result.justification_rejections.iter().any(|rejection| {
                    rejection.reason
                        == crate::memory_proof_budget::JustificationRejectionReason::UserEvidenceCountInsufficient
                }));
            }
        }
    }

    #[test]
    fn refuses_snapshot_known_at_mismatch() {
        let error = admit_lexical_hits_for_proof_budget(
            request(),
            &[hit("A", "evidence", 1.0)],
            &[admission("A")],
            &EvidenceIndexSnapshotReceipt { known_at_unix_ms: 199, index_state_sha256: digest("index state") },
        ).unwrap_err();
        assert!(error.contains("snapshot known_at"));
    }

    #[test]
    fn refuses_hits_without_a_matching_snapshot_receipt() {
        let mut unsnapshotted = hit("A", "evidence", 1.0);
        unsnapshotted.snapshot_receipt = None;
        let error = admit_lexical_hits_for_proof_budget(
            request(),
            &[unsnapshotted],
            &[admission("A")],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(error.contains("not bound to the supplied index snapshot"));

        let mut mismatched = hit("A", "evidence", 1.0);
        mismatched.snapshot_receipt = Some(EvidenceIndexSnapshotReceipt {
            known_at_unix_ms: 200,
            index_state_sha256: digest("another index state"),
        });
        let error = admit_lexical_hits_for_proof_budget(
            request(),
            &[mismatched],
            &[admission("A")],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(error.contains("not bound to the supplied index snapshot"));
    }

    #[test]
    fn refuses_admission_from_a_different_index_state() {
        let mut wrong_state = admission("A");
        wrong_state.index_state_sha256 = digest("different index state");
        let error = admit_lexical_hits_for_proof_budget(
            request(),
            &[hit("A", "evidence", 1.0)],
            &[wrong_state],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(error.contains("different lexical index state"));
    }

    #[test]
    fn refuses_missing_or_unused_admission_envelopes() {
        let missing = admit_lexical_hits_for_proof_budget(
            request(),
            &[hit("A", "evidence", 1.0)],
            &[],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(missing.contains("no trusted admission"));

        let unused = admit_lexical_hits_for_proof_budget(
            request(),
            &[hit("A", "evidence", 1.0)],
            &[admission("A"), admission("B")],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(unused.contains("unused admission"));
    }

    #[test]
    fn refuses_duplicate_hits_and_digest_mismatch() {
        let duplicate = admit_lexical_hits_for_proof_budget(
            request(),
            &[hit("A", "evidence", 1.0), hit("A", "evidence", 0.9)],
            &[admission("A")],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(duplicate.contains("duplicate lexical hit"));

        let mut corrupt = hit("A", "evidence", 1.0);
        corrupt.content_sha256 = digest("different content");
        let mismatch = admit_lexical_hits_for_proof_budget(
            request(),
            &[corrupt],
            &[admission("A")],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(mismatch.contains("payload digest"));
    }

    #[test]
    fn preserves_unauthorized_candidate_as_a_solver_rejection() {
        let mut denied = admission("A");
        denied.authorization_accepted = false;
        let proof_request = admit_lexical_hits_for_proof_budget(
            request(),
            &[hit("A", "evidence", 1.0)],
            &[denied],
            &snapshot_receipt(),
        ).unwrap();

        let result = solve_proof_budget(proof_request).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.candidate_rejections["A"].contains(&ProofCandidateRejection::Unauthorized));
    }

    #[test]
    fn rejects_scope_and_validity_metadata_drift_between_hit_and_admission() {
        let mut wrong_scope = hit("A", "evidence", 1.0);
        wrong_scope.scope_id = "scope-2".into();
        let error = admit_lexical_hits_for_proof_budget(
            request(),
            &[wrong_scope],
            &[admission("A")],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(error.contains("scope does not match"));

        let mut wrong_validity = hit("A", "evidence", 1.0);
        wrong_validity.valid_until_unix_ms = Some(400);
        let error = admit_lexical_hits_for_proof_budget(
            request(),
            &[wrong_validity],
            &[admission("A")],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(error.contains("validity interval does not match"));
    }

    #[test]
    fn rejects_cross_scope_admission_before_constructing_candidates() {
        let mut wrong_scope = admission("A");
        wrong_scope.scope_id = "scope-2".into();
        let error = admit_lexical_hits_for_proof_budget(
            request(),
            &[hit("A", "evidence", 1.0)],
            &[wrong_scope],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(error.contains("scope does not match"));
    }

    #[test]
    fn solver_reports_future_revoked_and_expired_candidates_without_promoting_them() {
        let mut future = admission("A");
        future.recorded_at_unix_ms = 201;

        let mut revoked = admission("B");
        revoked.retracted_at_unix_ms = Some(200);

        let mut expired = admission("C");
        expired.valid_until_unix_ms = Some(250);

        let mut proof_request = request();
        proof_request.obligations[0].alternatives = vec![
            ProofJustification {
                justification_id: "future".into(),
                evidence_dependencies: vec!["A".into()],
            },
            ProofJustification {
                justification_id: "revoked".into(),
                evidence_dependencies: vec!["B".into()],
            },
            ProofJustification {
                justification_id: "expired".into(),
                evidence_dependencies: vec!["C".into()],
            },
        ];

        let proof_request = admit_lexical_hits_for_proof_budget(
            proof_request,
            &[
                hit("A", "future evidence", 3.0),
                hit("B", "revoked evidence", 2.0),
                hit("C", "expired evidence", 1.0),
            ],
            &[future, revoked, expired],
            &snapshot_receipt(),
        ).unwrap();

        let result = solve_proof_budget(proof_request).unwrap();
        assert_eq!(result.status, ProofBudgetStatus::NoEligibleProof);
        assert!(result.candidate_rejections["A"]
            .contains(&ProofCandidateRejection::NotKnownAtQueryTime));
        assert!(result.candidate_rejections["B"]
            .contains(&ProofCandidateRejection::RetractedAtQueryTime));
        assert!(result.candidate_rejections["C"]
            .contains(&ProofCandidateRejection::OutsideValidTime));
    }

    #[test]
    fn rejects_non_finite_scores_and_duplicate_hdc_ids() {
        let mut non_finite = hit("A", "evidence", 1.0);
        non_finite.bm25_score = f64::NAN;
        let error = admit_lexical_hits_for_proof_budget(
            request(),
            &[non_finite],
            &[admission("A")],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(error.contains("invalid BM25 score"));

        let mut aliased = hit("B", "other evidence", 0.9);
        aliased.hdc_id = hit("A", "evidence", 1.0).hdc_id;
        let error = admit_lexical_hits_for_proof_budget(
            request(),
            &[hit("A", "evidence", 1.0), aliased],
            &[admission("A"), admission("B")],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(error.contains("duplicate HDC candidate ID"));
    }

    #[test]
    fn refuses_prepopulated_request_candidates_to_avoid_silent_replacement() {
        let mut proof_request = request();
        proof_request.candidates.push(crate::memory_proof_budget::ProofEvidenceCandidate {
            evidence_id: "old".into(),
            source_revision_id: "revision-old".into(),
            source_pointer: "/old".into(),
            source_span_text: "old".into(),
            source_span_start: 0,
            source_span_end: 3,
            source_family_id: "family-old".into(),
            source_session_id: Some("session-old".into()),
            source_speaker: None,
            scope_id: "scope-1".into(),
            payload_sha256: digest("old"),
            source_span_sha256: digest("old"),
            resolved_span_sha256: digest("old"),
            conflicts_with: BTreeSet::new(),
            recorded_at_unix_ms: 100,
            retracted_at_unix_ms: None,
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            authorized_principal_id: "principal-1".into(),
            authorization_accepted: true,
            authorization_receipt_sha256: digest("receipt"),
            retrieval_rank: 1,
            token_cost: 1,
        });
        let error = admit_lexical_hits_for_proof_budget(
            proof_request,
            &[hit("A", "evidence", 1.0)],
            &[admission("A")],
            &snapshot_receipt(),
        ).unwrap_err();
        assert!(error.contains("candidates must be empty"));
    }
}
