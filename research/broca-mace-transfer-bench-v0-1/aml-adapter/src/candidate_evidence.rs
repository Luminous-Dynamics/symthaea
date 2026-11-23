// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Backend-neutral, source-preserving candidate-evidence envelope.
//!
//! This module validates candidate evidence exchanged between retrieval
//! backends and the MACE evaluator. It does not authenticate issuers, verify
//! lifecycle state, grant rights, or establish semantic truth. Candidate
//! records remain untrusted until independently qualified by the assurance
//! layer.

use std::collections::BTreeSet;

use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

pub const CANDIDATE_EVIDENCE_SCHEMA: &str = "mace_candidate_evidence_v0_1";
pub const MAX_CANDIDATE_EVIDENCE_RECORDS: usize = 10_000;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum EvidenceModalityV1 {
    Text,
    Image,
    Audio,
    Video,
    Structured,
}

impl EvidenceModalityV1 {
    fn parse(value: &str) -> Result<Self, CandidateEvidenceReject> {
        match value {
            "text" => Ok(Self::Text),
            "image" => Ok(Self::Image),
            "audio" => Ok(Self::Audio),
            "video" => Ok(Self::Video),
            "structured" => Ok(Self::Structured),
            _ => Err(CandidateEvidenceReject::InvalidModality),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Structured => "structured",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateEvidenceV1 {
    pub run_id: String,
    pub candidate_id: String,
    pub backend_id: String,
    pub backend_revision_sha256: String,
    pub query_scope: String,
    pub query_purpose: String,
    /// SHA-256 of the exact query bytes; the query itself is not copied here.
    pub query_sha256: String,
    pub evidence_id: String,
    pub source_id: String,
    pub source_sha256: String,
    pub payload_sha256: String,
    /// Opaque resolver-owned reference. This module never opens paths or URLs.
    pub payload_ref: String,
    pub modality: EvidenceModalityV1,
    pub retrieval_channel: String,
    /// One-based rank within retrieval_channel; scores across channels are not comparable.
    pub retrieval_rank: u32,
    /// Optional backend-native score. Never compared across channels by this module.
    pub retrieval_score: Option<f64>,
    /// Required parent evidence identifiers for derived evidence.
    pub parent_evidence_ids: Vec<String>,
    /// Optional receipt digest binding the transformation that produced this evidence.
    pub derivation_receipt_sha256: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CandidateEvidenceBatchV1 {
    pub run_id: String,
    pub candidate_id: String,
    pub backend_id: String,
    pub backend_revision_sha256: String,
    pub query_scope: String,
    pub query_purpose: String,
    pub query_sha256: String,
    /// Deterministically ordered by channel, rank, then evidence_id.
    pub evidence: Vec<CandidateEvidenceV1>,
    /// Commitment to the complete normalized candidate set and its query/backend identity.
    pub candidate_set_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateEvidenceReject {
    InvalidJson,
    TopLevelMustBeObject,
    UnknownOrMissingField,
    InvalidString,
    InvalidDigest,
    InvalidModality,
    InvalidRank,
    InvalidScore,
    InvalidParent,
    MissingDerivationReceipt,
    EmptyBatch,
    TooManyRecords,
    MixedCandidateIdentity,
    DuplicateEvidenceId,
    DuplicateChannelRank,
}

fn valid_atom(value: &str) -> bool {
    !value.trim().is_empty()
        && value == value.trim()
        && !value.chars().any(char::is_control)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn required_string(
    object: &Map<String, Value>,
    field: &str,
) -> Result<String, CandidateEvidenceReject> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| valid_atom(value))
        .map(str::to_owned)
        .ok_or(CandidateEvidenceReject::InvalidString)
}

fn optional_digest(
    object: &Map<String, Value>,
    field: &str,
) -> Result<Option<String>, CandidateEvidenceReject> {
    match object.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if valid_sha256(value) => Ok(Some(value.clone())),
        _ => Err(CandidateEvidenceReject::InvalidDigest),
    }
}

fn check_exact_fields(
    object: &Map<String, Value>,
    required: &[&str],
    optional: &[&str],
) -> Result<(), CandidateEvidenceReject> {
    if required.iter().any(|field| !object.contains_key(*field)) {
        return Err(CandidateEvidenceReject::UnknownOrMissingField);
    }
    if object
        .keys()
        .any(|field| !required.contains(&field.as_str()) && !optional.contains(&field.as_str()))
    {
        return Err(CandidateEvidenceReject::UnknownOrMissingField);
    }
    Ok(())
}

fn parse_record(line: &str) -> Result<CandidateEvidenceV1, CandidateEvidenceReject> {
    let value: Value =
        serde_json::from_str(line).map_err(|_| CandidateEvidenceReject::InvalidJson)?;
    let object = value
        .as_object()
        .ok_or(CandidateEvidenceReject::TopLevelMustBeObject)?;

    const REQUIRED: &[&str] = &[
        "schema_version",
        "record_type",
        "run_id",
        "candidate_id",
        "backend_id",
        "backend_revision_sha256",
        "query_scope",
        "query_purpose",
        "query_sha256",
        "evidence_id",
        "source_id",
        "source_sha256",
        "payload_sha256",
        "payload_ref",
        "modality",
        "retrieval_channel",
        "retrieval_rank",
        "parent_evidence_ids",
    ];
    const OPTIONAL: &[&str] = &["retrieval_score", "derivation_receipt_sha256"];
    check_exact_fields(object, REQUIRED, OPTIONAL)?;

    if object.get("schema_version").and_then(Value::as_str)
        != Some(CANDIDATE_EVIDENCE_SCHEMA)
        || object.get("record_type").and_then(Value::as_str) != Some("candidate_evidence")
    {
        return Err(CandidateEvidenceReject::UnknownOrMissingField);
    }

    let backend_revision_sha256 = required_string(object, "backend_revision_sha256")?;
    let query_sha256 = required_string(object, "query_sha256")?;
    let source_sha256 = required_string(object, "source_sha256")?;
    let payload_sha256 = required_string(object, "payload_sha256")?;
    if !valid_sha256(&backend_revision_sha256)
        || !valid_sha256(&query_sha256)
        || !valid_sha256(&source_sha256)
        || !valid_sha256(&payload_sha256)
    {
        return Err(CandidateEvidenceReject::InvalidDigest);
    }

    let rank = object
        .get("retrieval_rank")
        .and_then(Value::as_u64)
        .filter(|rank| *rank > 0 && *rank <= u64::from(u32::MAX))
        .ok_or(CandidateEvidenceReject::InvalidRank)? as u32;

    let score = match object.get("retrieval_score") {
        None | Some(Value::Null) => None,
        Some(value) => {
            let score = value
                .as_f64()
                .filter(|score| score.is_finite())
                .ok_or(CandidateEvidenceReject::InvalidScore)?;
            Some(score)
        }
    };

    let modality = EvidenceModalityV1::parse(
        object
            .get("modality")
            .and_then(Value::as_str)
            .ok_or(CandidateEvidenceReject::InvalidModality)?,
    )?;

    let evidence_id = required_string(object, "evidence_id")?;
    let parent_values = object
        .get("parent_evidence_ids")
        .and_then(Value::as_array)
        .ok_or(CandidateEvidenceReject::InvalidParent)?;
    let mut parent_evidence_ids = Vec::with_capacity(parent_values.len());
    let mut seen_parents = BTreeSet::new();
    for parent in parent_values {
        let parent = parent
            .as_str()
            .filter(|value| valid_atom(value))
            .ok_or(CandidateEvidenceReject::InvalidParent)?;
        if parent == evidence_id || !seen_parents.insert(parent.to_owned()) {
            return Err(CandidateEvidenceReject::InvalidParent);
        }
        parent_evidence_ids.push(parent.to_owned());
    }
    parent_evidence_ids.sort();

    let derivation_receipt_sha256 = optional_digest(object, "derivation_receipt_sha256")?;
    if !parent_evidence_ids.is_empty() && derivation_receipt_sha256.is_none() {
        return Err(CandidateEvidenceReject::MissingDerivationReceipt);
    }

    Ok(CandidateEvidenceV1 {
        run_id: required_string(object, "run_id")?,
        candidate_id: required_string(object, "candidate_id")?,
        backend_id: required_string(object, "backend_id")?,
        backend_revision_sha256,
        query_scope: required_string(object, "query_scope")?,
        query_purpose: required_string(object, "query_purpose")?,
        query_sha256,
        evidence_id,
        source_id: required_string(object, "source_id")?,
        source_sha256,
        payload_sha256,
        payload_ref: required_string(object, "payload_ref")?,
        modality,
        retrieval_channel: required_string(object, "retrieval_channel")?,
        retrieval_rank: rank,
        retrieval_score: score,
        parent_evidence_ids,
        derivation_receipt_sha256,
    })
}

/// Parse and validate a candidate JSONL stream. All records must describe the
/// same run, candidate, backend revision, query, scope and purpose. Unknown
/// fields, mixed scopes, duplicate evidence IDs and duplicate per-channel ranks
/// are rejected before a batch digest is emitted.
pub fn parse_candidate_evidence_jsonl(
    input: &str,
) -> Result<CandidateEvidenceBatchV1, CandidateEvidenceReject> {
    let mut evidence = Vec::new();
    for line in input.lines().filter(|line| !line.trim().is_empty()) {
        if evidence.len() >= MAX_CANDIDATE_EVIDENCE_RECORDS {
            return Err(CandidateEvidenceReject::TooManyRecords);
        }
        evidence.push(parse_record(line)?);
    }
    if evidence.is_empty() {
        return Err(CandidateEvidenceReject::EmptyBatch);
    }

    evidence.sort_by(|left, right| {
        left.retrieval_channel
            .cmp(&right.retrieval_channel)
            .then_with(|| left.retrieval_rank.cmp(&right.retrieval_rank))
            .then_with(|| left.evidence_id.cmp(&right.evidence_id))
    });

    let first = &evidence[0];
    let mut evidence_ids = BTreeSet::new();
    let mut channel_ranks = BTreeSet::new();
    for item in &evidence {
        if item.run_id != first.run_id
            || item.candidate_id != first.candidate_id
            || item.backend_id != first.backend_id
            || item.backend_revision_sha256 != first.backend_revision_sha256
            || item.query_scope != first.query_scope
            || item.query_purpose != first.query_purpose
            || item.query_sha256 != first.query_sha256
        {
            return Err(CandidateEvidenceReject::MixedCandidateIdentity);
        }
        if !evidence_ids.insert(item.evidence_id.clone()) {
            return Err(CandidateEvidenceReject::DuplicateEvidenceId);
        }
        if !channel_ranks.insert((item.retrieval_channel.clone(), item.retrieval_rank)) {
            return Err(CandidateEvidenceReject::DuplicateChannelRank);
        }
    }

    let candidate_set_sha256 = candidate_set_digest(&evidence);
    Ok(CandidateEvidenceBatchV1 {
        run_id: first.run_id.clone(),
        candidate_id: first.candidate_id.clone(),
        backend_id: first.backend_id.clone(),
        backend_revision_sha256: first.backend_revision_sha256.clone(),
        query_scope: first.query_scope.clone(),
        query_purpose: first.query_purpose.clone(),
        query_sha256: first.query_sha256.clone(),
        evidence,
        candidate_set_sha256,
    })
}

fn candidate_set_digest(evidence: &[CandidateEvidenceV1]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"luminous:mace:candidate-evidence-set:v1\0");
    if let Some(first) = evidence.first() {
        hash_field(&mut hasher, &first.run_id);
        hash_field(&mut hasher, &first.candidate_id);
        hash_field(&mut hasher, &first.backend_id);
        hash_field(&mut hasher, &first.backend_revision_sha256);
        hash_field(&mut hasher, &first.query_scope);
        hash_field(&mut hasher, &first.query_purpose);
        hash_field(&mut hasher, &first.query_sha256);
    }
    hasher.update((evidence.len() as u64).to_be_bytes());
    for item in evidence {
        hash_field(&mut hasher, &item.evidence_id);
        hash_field(&mut hasher, &item.source_id);
        hash_field(&mut hasher, &item.source_sha256);
        hash_field(&mut hasher, &item.payload_sha256);
        hash_field(&mut hasher, &item.payload_ref);
        hash_field(&mut hasher, item.modality.as_str());
        hash_field(&mut hasher, &item.retrieval_channel);
        hasher.update(item.retrieval_rank.to_be_bytes());
        match item.retrieval_score {
            Some(score) => {
                hasher.update([1]);
                hasher.update(score.to_bits().to_be_bytes());
            }
            None => { hasher.update([0]); }
        }
        hasher.update((item.parent_evidence_ids.len() as u64).to_be_bytes());
        for parent in &item.parent_evidence_ids {
            hash_field(&mut hasher, parent);
        }
        match &item.derivation_receipt_sha256 {
            Some(digest) => {
                hasher.update([1]);
                hash_field(&mut hasher, digest);
            }
            None => { hasher.update([0]); }
        }
    }
    hex_digest(hasher.finalize().as_slice())
}

fn hash_field(hasher: &mut Sha256, value: &str) {
    hasher.update((value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[usize::from(byte >> 4)] as char);
        output.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const DIGEST_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn record(evidence_id: &str, channel: &str, rank: u64) -> Value {
        json!({
            "schema_version": CANDIDATE_EVIDENCE_SCHEMA,
            "record_type": "candidate_evidence",
            "run_id": "run-1",
            "candidate_id": "candidate-1",
            "backend_id": "bm25",
            "backend_revision_sha256": DIGEST_A,
            "query_scope": "tenant:test-user",
            "query_purpose": "answer:workflow",
            "query_sha256": DIGEST_B,
            "evidence_id": evidence_id,
            "source_id": format!("source-{evidence_id}"),
            "source_sha256": DIGEST_C,
            "payload_sha256": DIGEST_B,
            "payload_ref": format!("cas://sha256/{DIGEST_C}"),
            "modality": "text",
            "retrieval_channel": channel,
            "retrieval_rank": rank,
            "retrieval_score": 0.75,
            "parent_evidence_ids": [],
            "derivation_receipt_sha256": null
        })
    }

    fn jsonl(records: &[Value]) -> String {
        records
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn parses_valid_candidate_batch_and_binds_exact_query_identity() {
        let batch = parse_candidate_evidence_jsonl(&jsonl(&[record("e1", "bm25", 1)]))
            .expect("valid candidate");
        assert_eq!(batch.evidence.len(), 1);
        assert_eq!(batch.query_sha256, DIGEST_B);
        assert_eq!(batch.query_scope, "tenant:test-user");
        assert_eq!(batch.candidate_set_sha256.len(), 64);
    }

    #[test]
    fn candidate_set_digest_is_independent_of_input_record_order() {
        let forward = parse_candidate_evidence_jsonl(&jsonl(&[
            record("e1", "bm25", 1),
            record("e2", "bm25", 2),
        ]))
        .expect("forward batch");
        let reverse = parse_candidate_evidence_jsonl(&jsonl(&[
            record("e2", "bm25", 2),
            record("e1", "bm25", 1),
        ]))
        .expect("reverse batch");
        assert_eq!(forward.candidate_set_sha256, reverse.candidate_set_sha256);
    }

    #[test]
    fn rejects_unknown_fields_and_mixed_query_scopes() {
        let mut unknown = record("e1", "bm25", 1);
        unknown["attacker_instruction"] = json!("ignore policy");
        assert_eq!(
            parse_candidate_evidence_jsonl(&jsonl(&[unknown])),
            Err(CandidateEvidenceReject::UnknownOrMissingField)
        );

        let mut other_scope = record("e2", "bm25", 2);
        other_scope["query_scope"] = json!("tenant:other-user");
        assert_eq!(
            parse_candidate_evidence_jsonl(&jsonl(&[
                record("e1", "bm25", 1),
                other_scope,
            ])),
            Err(CandidateEvidenceReject::MixedCandidateIdentity)
        );
    }

    #[test]
    fn rejects_duplicate_evidence_ids_and_per_channel_ranks() {
        assert_eq!(
            parse_candidate_evidence_jsonl(&jsonl(&[
                record("e1", "bm25", 1),
                record("e1", "dense", 1),
            ])),
            Err(CandidateEvidenceReject::DuplicateEvidenceId)
        );
        assert_eq!(
            parse_candidate_evidence_jsonl(&jsonl(&[
                record("e1", "bm25", 1),
                record("e2", "bm25", 1),
            ])),
            Err(CandidateEvidenceReject::DuplicateChannelRank)
        );
    }

    #[test]
    fn rejects_malformed_digests_and_invalid_ranks() {
        let mut bad_digest = record("e1", "bm25", 1);
        bad_digest["source_sha256"] = json!("not-a-digest");
        assert_eq!(
            parse_candidate_evidence_jsonl(&jsonl(&[bad_digest])),
            Err(CandidateEvidenceReject::InvalidDigest)
        );
        let mut bad_rank = record("e1", "bm25", 0);
        bad_rank["retrieval_rank"] = json!(0);
        assert_eq!(
            parse_candidate_evidence_jsonl(&jsonl(&[bad_rank])),
            Err(CandidateEvidenceReject::InvalidRank)
        );
    }

    #[test]
    fn requires_lineage_receipt_for_derived_evidence() {
        let mut derived = record("e2", "bm25", 1);
        derived["parent_evidence_ids"] = json!(["e1"]);
        assert_eq!(
            parse_candidate_evidence_jsonl(&jsonl(&[derived.clone()])),
            Err(CandidateEvidenceReject::MissingDerivationReceipt)
        );
        derived["derivation_receipt_sha256"] = json!(DIGEST_C);
        assert!(parse_candidate_evidence_jsonl(&jsonl(&[derived])).is_ok());
    }

    #[test]
    fn rejects_self_parent_and_duplicate_parents() {
        let mut self_parent = record("e1", "bm25", 1);
        self_parent["parent_evidence_ids"] = json!(["e1"]);
        self_parent["derivation_receipt_sha256"] = json!(DIGEST_C);
        assert_eq!(
            parse_candidate_evidence_jsonl(&jsonl(&[self_parent])),
            Err(CandidateEvidenceReject::InvalidParent)
        );
        let mut duplicate = record("e2", "bm25", 1);
        duplicate["parent_evidence_ids"] = json!(["e1", "e1"]);
        duplicate["derivation_receipt_sha256"] = json!(DIGEST_C);
        assert_eq!(
            parse_candidate_evidence_jsonl(&jsonl(&[duplicate])),
            Err(CandidateEvidenceReject::InvalidParent)
        );
    }

    #[test]
    fn rejects_unknown_modalities_and_non_numeric_scores() {
        let mut modality = record("e1", "bm25", 1);
        modality["modality"] = json!("telepathy");
        assert_eq!(
            parse_candidate_evidence_jsonl(&jsonl(&[modality])),
            Err(CandidateEvidenceReject::InvalidModality)
        );
        let mut score = record("e1", "bm25", 1);
        score["retrieval_score"] = json!("NaN");
        assert_eq!(
            parse_candidate_evidence_jsonl(&jsonl(&[score])),
            Err(CandidateEvidenceReject::InvalidScore)
        );
    }

    #[test]
    fn rejects_empty_input_and_accepts_multiple_independent_channels() {
        assert_eq!(
            parse_candidate_evidence_jsonl(" \n \n"),
            Err(CandidateEvidenceReject::EmptyBatch)
        );
        let batch = parse_candidate_evidence_jsonl(&jsonl(&[
            record("e1", "bm25", 1),
            record("e2", "dense", 1),
        ]))
        .expect("hybrid channels");
        assert_eq!(batch.evidence.len(), 2);
    }
}
