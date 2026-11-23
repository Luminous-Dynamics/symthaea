// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Projects LongMemEval-V2 retrieval hits into the MACE candidate-evidence
//! envelope without flattening original screenshot evidence into text.
//!
//! This adapter creates structurally bound candidates only. It does not
//! authenticate the benchmark publisher, prove semantic truth, or authorize
//! consequential use. Screenshot payload references are opaque workspace
//! resolver keys, not filesystem paths.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

use crate::candidate_evidence::{
    CandidateEvidenceBatchV1, CandidateEvidenceReject, CandidateEvidenceV1,
    EvidenceModalityV1, encode_candidate_evidence_jsonl,
};
use crate::longmemeval_v2::{
    LongMemEvalV2HitV1, LongMemEvalV2SearchResultV1,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LongMemEvalV2CandidateIdentityV1 {
    pub run_id: String,
    pub candidate_id: String,
    pub backend_id: String,
    /// Exact build/configuration identity supplied by the reproducible runner.
    pub backend_revision_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LongMemEvalV2CandidateReject {
    EmptyQuery,
    EmptyHits,
    InvalidDomain,
    InvalidHit,
    InvalidRank,
    InvalidScore,
    InvalidSourceDigest,
    MissingScreenshotDigest,
    ScreenshotProjectionMismatch,
    Candidate(CandidateEvidenceReject),
}

impl From<CandidateEvidenceReject> for LongMemEvalV2CandidateReject {
    fn from(value: CandidateEvidenceReject) -> Self {
        Self::Candidate(value)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn state_parent_id(trajectory_id: &str, ordinal: usize, state_sha256: &str) -> String {
    format!("lmev2:state:{trajectory_id}:{ordinal}:{state_sha256}")
}

fn source_state_evidence(
    identity: &LongMemEvalV2CandidateIdentityV1,
    query_scope: &str,
    query_purpose: &str,
    query_sha256: &str,
    domain: &str,
    trajectory_id: &str,
    trajectory_sha256: &str,
    state: &LongMemEvalV2StateV1,
    retrieval_rank: u32,
) -> CandidateEvidenceV1 {
    CandidateEvidenceV1 {
        run_id: identity.run_id.clone(),
        candidate_id: identity.candidate_id.clone(),
        backend_id: identity.backend_id.clone(),
        backend_revision_sha256: identity.backend_revision_sha256.clone(),
        query_scope: query_scope.to_owned(),
        query_purpose: query_purpose.to_owned(),
        query_sha256: query_sha256.to_owned(),
        evidence_id: state_parent_id(trajectory_id, state.source_ordinal, &state.raw_state_sha256),
        source_id: trajectory_id.to_owned(),
        source_sha256: trajectory_sha256.to_owned(),
        payload_sha256: state.raw_state_sha256.clone(),
        payload_ref: format!(
            "lmev2://{domain}/trajectories/{trajectory_id}/states/{}/raw-json",
            state.source_ordinal
        ),
        modality: EvidenceModalityV1::Structured,
        retrieval_channel: "source_state_lineage".to_owned(),
        retrieval_rank,
        retrieval_score: None,
        parent_evidence_ids: Vec::new(),
        derivation_receipt_sha256: None,
    }
}

fn screenshot_derivation_receipt(
    state_sha256: &str,
    image_sha256: &str,
    screenshot_reference: &str,
) -> String {
    let material = format!(
        "luminous:mace:lmev2:screenshot-binding:v1\\0{state_sha256}\\0{image_sha256}\\0{screenshot_reference}"
    );
    sha256_hex(material.as_bytes())
}

fn screenshot_evidence(
    identity: &LongMemEvalV2CandidateIdentityV1,
    query_scope: &str,
    query_purpose: &str,
    query_sha256: &str,
    domain: &str,
    trajectory_id: &str,
    trajectory_sha256: &str,
    state: &LongMemEvalV2StateV1,
    retrieval_rank: u32,
) -> CandidateEvidenceV1 {
    let digest = state.screenshot_sha256.as_deref().unwrap_or_default();
    let parent_id = state_parent_id(trajectory_id, state.source_ordinal, &state.raw_state_sha256);
    CandidateEvidenceV1 {
        run_id: identity.run_id.clone(),
        candidate_id: identity.candidate_id.clone(),
        backend_id: identity.backend_id.clone(),
        backend_revision_sha256: identity.backend_revision_sha256.clone(),
        query_scope: query_scope.to_owned(),
        query_purpose: query_purpose.to_owned(),
        query_sha256: query_sha256.to_owned(),
        evidence_id: format!(
            "lmev2:image:{trajectory_id}:{}:{digest}",
            state.source_ordinal
        ),
        source_id: trajectory_id.to_owned(),
        source_sha256: trajectory_sha256.to_owned(),
        payload_sha256: digest.to_owned(),
        payload_ref: format!(
            "lmev2://{domain}/trajectories/{trajectory_id}/states/{}/screenshots/{digest}",
            state.source_ordinal
        ),
        modality: EvidenceModalityV1::Image,
        retrieval_channel: "attached_image_evidence".to_owned(),
        retrieval_rank,
        retrieval_score: None,
        parent_evidence_ids: vec![parent_id],
        derivation_receipt_sha256: Some(screenshot_derivation_receipt(
            &state.raw_state_sha256,
            digest,
            &state.screenshot_reference,
        )),
    }
}

/// Convert retrieved V2 hits to canonical MACE candidate evidence.
///
/// Every hit contributes one text candidate whose parent IDs reference the
/// exact source-state digests and whose derivation receipt is the adapter's
/// deterministic chunk digest. Each distinct original screenshot contributes
/// an independent image candidate only when its pixel digest is present.
///
/// `attached_image_evidence` ranks express deterministic attachment order,
/// not image relevance. No image-query or image-ranking capability is implied.
pub fn encode_longmemeval_v2_candidate_evidence_v1(
    identity: &LongMemEvalV2CandidateIdentityV1,
    query: &str,
    domain: &str,
    hits: &[LongMemEvalV2HitV1],
) -> Result<CandidateEvidenceBatchV1, LongMemEvalV2CandidateReject> {
    if query.trim().is_empty() {
        return Err(LongMemEvalV2CandidateReject::EmptyQuery);
    }
    if !matches!(domain, "web" | "enterprise") {
        return Err(LongMemEvalV2CandidateReject::InvalidDomain);
    }
    if hits.is_empty() {
        return Err(LongMemEvalV2CandidateReject::EmptyHits);
    }

    let query_sha256 = sha256_hex(query.as_bytes());
    let query_scope = format!("longmemeval-v2:domain:{domain}");
    let query_purpose = "benchmark:memory-context";
    let mut records = Vec::new();
    let mut seen_state_ids = BTreeSet::<String>::new();
    let mut seen_screenshots = BTreeSet::<(String, usize, String)>::new();
    let mut state_rank = 0_u32;
    let mut image_rank = 0_u32;

    for hit in hits {
        if hit.domain != domain
            || hit.trajectory_id.trim().is_empty()
            || hit.chunk_id.trim().is_empty()
            || hit.state_window.is_empty()
            || hit.evidence_text.trim().is_empty()
        {
            return Err(LongMemEvalV2CandidateReject::InvalidHit);
        }
        if !valid_sha256(&hit.source_trajectory_sha256) || !valid_sha256(&hit.chunk_id) {
            return Err(LongMemEvalV2CandidateReject::InvalidSourceDigest);
        }
        if hit.rank == 0 {
            return Err(LongMemEvalV2CandidateReject::InvalidRank);
        }
        let retrieval_rank = u32::try_from(hit.rank)
            .map_err(|_| LongMemEvalV2CandidateReject::InvalidRank)?;
        if !hit.score.is_finite() || hit.score < 0.0 {
            return Err(LongMemEvalV2CandidateReject::InvalidScore);
        }
        if hit.screenshot_references.len() != hit.state_window.len()
            || hit.screenshot_sha256s.len() != hit.state_window.len()
        {
            return Err(LongMemEvalV2CandidateReject::ScreenshotProjectionMismatch);
        }

        let mut parent_evidence_ids = Vec::with_capacity(hit.state_window.len());
        for (index, state) in hit.state_window.iter().enumerate() {
            if !valid_sha256(&state.raw_state_sha256) {
                return Err(LongMemEvalV2CandidateReject::InvalidSourceDigest);
            }
            if hit.screenshot_references[index] != state.screenshot_reference
                || hit.screenshot_sha256s[index] != state.screenshot_sha256
            {
                return Err(LongMemEvalV2CandidateReject::ScreenshotProjectionMismatch);
            }
            let image_digest = state
                .screenshot_sha256
                .as_deref()
                .ok_or(LongMemEvalV2CandidateReject::MissingScreenshotDigest)?;
            if !valid_sha256(image_digest) {
                return Err(LongMemEvalV2CandidateReject::InvalidSourceDigest);
            }

            let state_id = state_parent_id(
                &hit.trajectory_id,
                state.source_ordinal,
                &state.raw_state_sha256,
            );
            parent_evidence_ids.push(state_id.clone());
            if seen_state_ids.insert(state_id) {
                state_rank = state_rank
                    .checked_add(1)
                    .ok_or(LongMemEvalV2CandidateReject::InvalidRank)?;
                records.push(source_state_evidence(
                    identity,
                    &query_scope,
                    query_purpose,
                    &query_sha256,
                    domain,
                    &hit.trajectory_id,
                    &hit.source_trajectory_sha256,
                    state,
                    state_rank,
                ));
            }

            let image_key = (
                hit.trajectory_id.clone(),
                state.source_ordinal,
                image_digest.to_owned(),
            );
            if seen_screenshots.insert(image_key) {
                image_rank = image_rank
                    .checked_add(1)
                    .ok_or(LongMemEvalV2CandidateReject::InvalidRank)?;
                records.push(screenshot_evidence(
                    identity,
                    &query_scope,
                    query_purpose,
                    &query_sha256,
                    domain,
                    &hit.trajectory_id,
                    &hit.source_trajectory_sha256,
                    state,
                    image_rank,
                ));
            }
        }

        records.push(CandidateEvidenceV1 {
            run_id: identity.run_id.clone(),
            candidate_id: identity.candidate_id.clone(),
            backend_id: identity.backend_id.clone(),
            backend_revision_sha256: identity.backend_revision_sha256.clone(),
            query_scope: query_scope.clone(),
            query_purpose: query_purpose.to_owned(),
            query_sha256: query_sha256.clone(),
            evidence_id: format!("lmev2:text:{}:{}", hit.trajectory_id, hit.chunk_id),
            source_id: hit.trajectory_id.clone(),
            source_sha256: hit.source_trajectory_sha256.clone(),
            payload_sha256: hit.chunk_id.clone(),
            payload_ref: format!(
                "lmev2://{domain}/trajectories/{}/chunks/{}",
                hit.trajectory_id, hit.chunk_id
            ),
            modality: EvidenceModalityV1::Text,
            retrieval_channel: "trajectory_bm25_text".to_owned(),
            retrieval_rank,
            retrieval_score: Some(hit.score),
            parent_evidence_ids,
            // The chunk digest commits to source-state digests, window bounds,
            // and rendered text. It is not an authenticated issuer signature.
            derivation_receipt_sha256: Some(hit.chunk_id.clone()),
        });
    }

    encode_candidate_evidence_jsonl(&records).map_err(Into::into)
}

/// Encode the complete V2 search result, including unscored temporal-neighbor
/// context. Neighbor records use their own retrieval channel and never inherit
/// the BM25 score of the hit that caused their expansion.
pub fn encode_longmemeval_v2_search_candidate_evidence_v1(
    identity: &LongMemEvalV2CandidateIdentityV1,
    query: &str,
    domain: &str,
    result: &LongMemEvalV2SearchResultV1,
) -> Result<CandidateEvidenceBatchV1, LongMemEvalV2CandidateReject> {
    if result.hits.is_empty() {
        return Err(LongMemEvalV2CandidateReject::EmptyHits);
    }

    let hit_batch = encode_longmemeval_v2_candidate_evidence_v1(
        identity,
        query,
        domain,
        &result.hits,
    )?;
    let query_sha256 = sha256_hex(query.as_bytes());
    let query_scope = format!("longmemeval-v2:domain:{domain}");
    let query_purpose = "benchmark:memory-context";
    let mut records = hit_batch.evidence.clone();
    let mut seen_state_ids = records
        .iter()
        .filter(|item| item.retrieval_channel == "source_state_lineage")
        .map(|item| item.evidence_id.clone())
        .collect::<BTreeSet<_>>();
    let mut seen_image_ids = records
        .iter()
        .filter(|item| item.modality == EvidenceModalityV1::Image)
        .map(|item| item.evidence_id.clone())
        .collect::<BTreeSet<_>>();
    let mut state_rank = records
        .iter()
        .filter(|item| item.retrieval_channel == "source_state_lineage")
        .map(|item| item.retrieval_rank)
        .max()
        .unwrap_or(0);
    let mut image_rank = records
        .iter()
        .filter(|item| item.retrieval_channel == "attached_image_evidence")
        .map(|item| item.retrieval_rank)
        .max()
        .unwrap_or(0);

    for (neighbor_index, neighbor) in result.neighbor_context.iter().enumerate() {
        let chunk = &neighbor.chunk;
        if chunk.domain != domain
            || chunk.trajectory_id.trim().is_empty()
            || chunk.state_window.is_empty()
            || chunk.evidence_text().trim().is_empty()
            || !valid_sha256(&chunk.source_trajectory_sha256)
            || !valid_sha256(&chunk.chunk_sha256)
        {
            return Err(LongMemEvalV2CandidateReject::InvalidHit);
        }
        if chunk.screenshot_references.len() != chunk.state_window.len() {
            return Err(LongMemEvalV2CandidateReject::ScreenshotProjectionMismatch);
        }

        let mut parents = Vec::with_capacity(chunk.state_window.len());
        for (state_index, state) in chunk.state_window.iter().enumerate() {
            if chunk.screenshot_references[state_index] != state.screenshot_reference {
                return Err(LongMemEvalV2CandidateReject::ScreenshotProjectionMismatch);
            }
            if !valid_sha256(&state.raw_state_sha256) {
                return Err(LongMemEvalV2CandidateReject::InvalidSourceDigest);
            }
            let image_digest = state
                .screenshot_sha256
                .as_deref()
                .ok_or(LongMemEvalV2CandidateReject::MissingScreenshotDigest)?;
            if !valid_sha256(image_digest) {
                return Err(LongMemEvalV2CandidateReject::InvalidSourceDigest);
            }

            let state_id = state_parent_id(
                &chunk.trajectory_id,
                state.source_ordinal,
                &state.raw_state_sha256,
            );
            parents.push(state_id.clone());
            if seen_state_ids.insert(state_id) {
                state_rank = state_rank
                    .checked_add(1)
                    .ok_or(LongMemEvalV2CandidateReject::InvalidRank)?;
                records.push(source_state_evidence(
                    identity,
                    &query_scope,
                    query_purpose,
                    &query_sha256,
                    domain,
                    &chunk.trajectory_id,
                    &chunk.source_trajectory_sha256,
                    state,
                    state_rank,
                ));
            }

            let image_id = format!(
                "lmev2:image:{}:{}:{}",
                chunk.trajectory_id, state.source_ordinal, image_digest
            );
            if seen_image_ids.insert(image_id) {
                image_rank = image_rank
                    .checked_add(1)
                    .ok_or(LongMemEvalV2CandidateReject::InvalidRank)?;
                records.push(screenshot_evidence(
                    identity,
                    &query_scope,
                    query_purpose,
                    &query_sha256,
                    domain,
                    &chunk.trajectory_id,
                    &chunk.source_trajectory_sha256,
                    state,
                    image_rank,
                ));
            }
        }

        let neighbor_rank = u32::try_from(neighbor_index + 1)
            .map_err(|_| LongMemEvalV2CandidateReject::InvalidRank)?;
        records.push(CandidateEvidenceV1 {
            run_id: identity.run_id.clone(),
            candidate_id: identity.candidate_id.clone(),
            backend_id: identity.backend_id.clone(),
            backend_revision_sha256: identity.backend_revision_sha256.clone(),
            query_scope: query_scope.clone(),
            query_purpose: query_purpose.to_owned(),
            query_sha256: query_sha256.clone(),
            evidence_id: format!(
                "lmev2:neighbor:{}:{}",
                chunk.trajectory_id, chunk.chunk_sha256
            ),
            source_id: chunk.trajectory_id.clone(),
            source_sha256: chunk.source_trajectory_sha256.clone(),
            payload_sha256: chunk.chunk_sha256.clone(),
            payload_ref: format!(
                "lmev2://{domain}/trajectories/{}/chunks/{}",
                chunk.trajectory_id, chunk.chunk_sha256
            ),
            modality: EvidenceModalityV1::Text,
            retrieval_channel: "trajectory_neighbor_expansion".to_owned(),
            retrieval_rank: neighbor_rank,
            retrieval_score: None,
            parent_evidence_ids: parents,
            derivation_receipt_sha256: Some(chunk.chunk_sha256.clone()),
        });
    }

    encode_candidate_evidence_jsonl(&records).map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::longmemeval_v2::LongMemEvalV2StateV1;

    const SOURCE_DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const CHUNK_DIGEST: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const STATE_DIGEST_A: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const STATE_DIGEST_B: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const IMAGE_DIGEST_A: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const IMAGE_DIGEST_B: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";
    const BACKEND_DIGEST: &str = "1111111111111111111111111111111111111111111111111111111111111111";

    fn state(
        ordinal: usize,
        state_digest: &str,
        image_digest: Option<&str>,
    ) -> LongMemEvalV2StateV1 {
        LongMemEvalV2StateV1 {
            source_ordinal: ordinal,
            source_state_index: ordinal as i64,
            step: Some(ordinal as i64),
            url: format!("https://example.test/state/{ordinal}"),
            action_on_arrival: None,
            thoughts: None,
            accessibility_tree: format!("State {ordinal}"),
            screenshot_reference: format!("trajectory-1/state-{ordinal}.png"),
            screenshot_sha256: image_digest.map(str::to_owned),
            raw_state_json: format!("{{\"state\":{ordinal}}}"),
            raw_state_sha256: state_digest.to_owned(),
        }
    }

    fn hit() -> LongMemEvalV2HitV1 {
        let state_a = state(1, STATE_DIGEST_A, Some(IMAGE_DIGEST_A));
        let state_b = state(2, STATE_DIGEST_B, Some(IMAGE_DIGEST_B));
        LongMemEvalV2HitV1 {
            chunk_id: CHUNK_DIGEST.to_owned(),
            trajectory_id: "trajectory-1".to_owned(),
            domain: "web".to_owned(),
            source_trajectory_sha256: SOURCE_DIGEST.to_owned(),
            center_state_ordinal: 1,
            score: 2.5,
            rank: 1,
            evidence_text: "Customer Ada submitted order 42".to_owned(),
            screenshot_references: vec![
                state_a.screenshot_reference.clone(),
                state_b.screenshot_reference.clone(),
            ],
            screenshot_sha256s: vec![
                state_a.screenshot_sha256.clone(),
                state_b.screenshot_sha256.clone(),
            ],
            state_window: vec![state_a, state_b],
        }
    }

    fn identity() -> LongMemEvalV2CandidateIdentityV1 {
        LongMemEvalV2CandidateIdentityV1 {
            run_id: "run-1".to_owned(),
            candidate_id: "candidate-1".to_owned(),
            backend_id: "mace_lmev2_bm25".to_owned(),
            backend_revision_sha256: BACKEND_DIGEST.to_owned(),
        }
    }

    #[test]
    fn converts_text_and_images_with_query_source_and_state_lineage() {
        let batch = encode_longmemeval_v2_candidate_evidence_v1(
            &identity(),
            "Which order did Ada submit?",
            "web",
            &[hit()],
        )
        .expect("candidate evidence");

        assert_eq!(batch.evidence.len(), 5);
        assert_eq!(
            batch.query_sha256,
            sha256_hex("Which order did Ada submit?".as_bytes())
        );
        let text = batch.evidence.iter()
            .find(|item| item.modality == EvidenceModalityV1::Text)
            .expect("text candidate");
        assert_eq!(text.source_sha256, SOURCE_DIGEST);
        assert_eq!(text.payload_sha256, CHUNK_DIGEST);
        assert_eq!(text.parent_evidence_ids.len(), 2);
        assert_eq!(text.derivation_receipt_sha256.as_deref(), Some(CHUNK_DIGEST));
        assert!(text.parent_evidence_ids.iter().all(|parent| batch.evidence.iter()
            .any(|item| item.evidence_id == *parent
                && item.retrieval_channel == "source_state_lineage")));
        let source_states = batch.evidence.iter()
            .filter(|item| item.retrieval_channel == "source_state_lineage")
            .collect::<Vec<_>>();
        assert_eq!(source_states.len(), 2);
        assert!(source_states.iter().all(|item| item.modality == EvidenceModalityV1::Structured));

        let images = batch.evidence.iter()
            .filter(|item| item.modality == EvidenceModalityV1::Image)
            .collect::<Vec<_>>();
        assert_eq!(images.len(), 2);
        assert!(images.iter().any(|item| item.payload_sha256 == IMAGE_DIGEST_A));
        assert!(images.iter().any(|item| item.payload_sha256 == IMAGE_DIGEST_B));
        assert!(images.iter().all(|item| item.retrieval_score.is_none()));
        assert!(images.iter().all(|item| item.retrieval_channel == "attached_image_evidence"));
        assert!(images.iter().all(|item| item.parent_evidence_ids.len() == 1
            && item.derivation_receipt_sha256.is_some()));
    }

    #[test]
    fn deduplicates_original_images_repeated_by_overlapping_hits() {
        let first = hit();
        let mut second = hit();
        second.chunk_id = "2222222222222222222222222222222222222222222222222222222222222222".to_owned();
        second.rank = 2;
        second.center_state_ordinal = 2;
        let batch = encode_longmemeval_v2_candidate_evidence_v1(
            &identity(),
            "query",
            "web",
            &[first, second],
        )
        .expect("overlap deduplication");
        assert_eq!(
            batch.evidence.iter()
                .filter(|item| item.modality == EvidenceModalityV1::Image)
                .count(),
            2
        );
        assert_eq!(
            batch.evidence.iter()
                .filter(|item| item.modality == EvidenceModalityV1::Text)
                .count(),
            2
        );
    }

    #[test]
    fn search_result_encoder_includes_neighbors_without_inheriting_bm25_scores() {
        let hit = hit();
        let mut neighbor_state = state(
            3,
            "4444444444444444444444444444444444444444444444444444444444444444",
            Some("5555555555555555555555555555555555555555555555555555555555555555"),
        );
        neighbor_state.screenshot_reference = "trajectory-2/state-3.png".to_owned();
        let neighbor_chunk = crate::longmemeval_v2::LongMemEvalV2EvidenceChunkV1 {
            chunk_id: "2222222222222222222222222222222222222222222222222222222222222222".to_owned(),
            trajectory_id: "trajectory-2".to_owned(),
            domain: "web".to_owned(),
            goal: "Find customer Ada's order".to_owned(),
            start_url: "https://example.test".to_owned(),
            center_state_ordinal: 3,
            window_start_ordinal: 3,
            window_end_ordinal_exclusive: 4,
            source_trajectory_sha256: "3333333333333333333333333333333333333333333333333333333333333333".to_owned(),
            state_window: vec![neighbor_state],
            screenshot_references: vec!["trajectory-2/state-3.png".to_owned()],
            chunk_sha256: "2222222222222222222222222222222222222222222222222222222222222222".to_owned(),
            evidence_text: "Earlier order workflow state".to_owned(),
        };
        let result = LongMemEvalV2SearchResultV1 {
            hits: vec![hit],
            neighbor_context: vec![crate::longmemeval_v2::LongMemEvalV2NeighborV1 {
                chunk: neighbor_chunk,
                expanded_from_hit_rank: 1,
                relative_state_offset: -1,
            }],
            search_latency_ns: 10,
        };
        let batch = encode_longmemeval_v2_search_candidate_evidence_v1(
            &identity(),
            "query",
            "web",
            &result,
        )
        .expect("complete candidate set");
        let neighbor = batch.evidence.iter()
            .find(|item| item.retrieval_channel == "trajectory_neighbor_expansion")
            .expect("neighbor record");
        assert_eq!(neighbor.retrieval_score, None);
        assert_eq!(neighbor.retrieval_rank, 1);
        assert_eq!(neighbor.parent_evidence_ids.len(), 1);
        assert!(batch.evidence.iter().any(|item| {
            item.evidence_id == neighbor.parent_evidence_ids[0]
                && item.retrieval_channel == "source_state_lineage"
        }));
        assert!(batch.evidence.iter().filter(|item| item.retrieval_channel == "source_state_lineage").count() == 3);
    }

    #[test]
    fn refuses_image_evidence_without_a_pixel_digest() {
        let mut value = hit();
        value.state_window[0].screenshot_sha256 = None;
        value.screenshot_sha256s[0] = None;
        assert_eq!(
            encode_longmemeval_v2_candidate_evidence_v1(
                &identity(),
                "query",
                "web",
                &[value],
            ),
            Err(LongMemEvalV2CandidateReject::MissingScreenshotDigest)
        );
    }

    #[test]
    fn rejects_wrong_domain_and_non_finite_backend_score() {
        assert_eq!(
            encode_longmemeval_v2_candidate_evidence_v1(
                &identity(),
                "query",
                "enterprise",
                &[hit()],
            ),
            Err(LongMemEvalV2CandidateReject::InvalidHit)
        );
        let mut value = hit();
        value.score = f64::NAN;
        assert_eq!(
            encode_longmemeval_v2_candidate_evidence_v1(
                &identity(),
                "query",
                "web",
                &[value],
            ),
            Err(LongMemEvalV2CandidateReject::InvalidScore)
        );
    }
}
