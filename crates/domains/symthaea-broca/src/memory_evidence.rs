// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Evidence payloads for memory systems whose approximate index stores vectors only.
//!
//! LongMemEval-V2 requires evidence-bearing text or existing image paths. HDC
//! candidate IDs are not themselves evidence. This module defines a companion
//! payload contract and process-local resolver that filters scope, validity and
//! lifecycle and verifies content digests before context reaches a reader.
//!
//! A deterministic BM25 lexical baseline is included for raw-text queries.
//! It is not a semantic encoder or complete LongMemEval-V2 backend: trajectory
//! ingestion, multimodal query handling, and benchmark-worker wiring remain open.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};

pub const EVIDENCE_SCHEMA_VERSION: &str = "BROCA-EVIDENCE-ITEM-V1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceLifecycle {
    Active,
    Superseded,
    Revoked,
    Deleted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EvidencePayload {
    Text { value: String },
    /// Path to an image relative to payload_root or absolute beneath it.
    /// The digest binds file bytes, not the path string.
    Image { path: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceSource {
    pub trajectory_id: String,
    pub step_id: Option<String>,
    pub step_index: Option<u64>,
    /// SHA-256 of the exact canonical source bytes emitted by the ingestion adapter.
    pub source_sha256: String,
}

/// Evidence linked to the numeric ID used by an HDC candidate index.
///
/// content_sha256 hashes UTF-8 bytes for Text and file bytes for Image.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub schema_version: String,
    /// Stable opaque evidence identity, never a benchmark question ID.
    pub evidence_id: String,
    pub hdc_id: u64,
    /// Namespace/environment scope provided by trusted configuration.
    pub scope_id: String,
    pub source: EvidenceSource,
    pub payload: EvidencePayload,
    pub content_sha256: String,
    pub lifecycle: EvidenceLifecycle,
    /// Optional source event time, Unix epoch milliseconds.
    pub occurred_at_unix_ms: Option<u64>,
    /// Inclusive beginning of validity, Unix epoch milliseconds.
    pub valid_from_unix_ms: Option<u64>,
    /// Exclusive end of validity, Unix epoch milliseconds.
    pub valid_until_unix_ms: Option<u64>,
    /// Required only while lifecycle is Superseded.
    pub superseded_by: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryContextType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "image")]
    Image,
}

/// Shape consumed by the official LongMemEval-V2 Memory.query adapter.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryContextItem {
    #[serde(rename = "type")]
    pub kind: MemoryContextType,
    pub value: String,
}

/// A resolved context item with the exact record and trajectory location that
/// supplied it. Benchmark adapters may project `item` to the official
/// `type/value` ABI while retaining this envelope for attribution and audits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedMemoryContextItem {
    pub hdc_id: u64,
    pub evidence_id: String,
    pub source: EvidenceSource,
    pub item: MemoryContextItem,
}

/// A seed or an adjacent state selected for temporal/workflow context.
///
/// Seed candidates retain their retrieval identity. An adjacent candidate records
/// which seed caused its inclusion; no relevance score is fabricated for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AdjacentStateCandidate {
    pub(crate) hdc_id: u64,
    /// None for ranked retrieval seeds; Some(seed HDC ID) for an expanded neighbor.
    pub(crate) anchor_hdc_id: Option<u64>,
}

/// Why an input candidate was not emitted by an audited evidence resolution.
///
/// These dispositions describe deterministic operational checks; they do not
/// establish whether emitted text semantically supports the user's question.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceCandidateDisposition {
    Emitted,
    DuplicateCandidate,
    UnknownCandidate,
    InvalidRecord,
    InactiveLifecycle,
    ScopeMismatch,
    NotYetValid,
    Expired,
    MetadataEligibleBeyondOutputLimit,
}

/// Per-input-candidate disposition, retained in the original hit order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceCandidateAudit {
    pub hdc_id: u64,
    pub disposition: EvidenceCandidateDisposition,
}

/// Minimum operational coverage required before a caller proceeds to answer/action.
///
/// This is not a semantic sufficiency model. Configure the floor for the application
/// and validate its abstention/coverage trade-off against labeled held-out tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceFloorPolicy {
    pub minimum_items: usize,
    pub minimum_distinct_trajectories: usize,
}

/// Observable outcome of applying the configured operational coverage floor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceFloorAudit {
    pub required_items: usize,
    pub required_distinct_trajectories: usize,
    pub emitted_items: usize,
    pub distinct_trajectories: usize,
    pub meets_floor: bool,
}

/// Resolution output plus an audit for every input HDC hit.
///
/// If an eligible payload is corrupt or cannot be verified, resolution returns an
/// error instead of disguising an integrity problem as insufficient evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditedMemoryResolution {
    pub items: Vec<ResolvedMemoryContextItem>,
    pub candidate_audit: Vec<EvidenceCandidateAudit>,
    pub floor: EvidenceFloorAudit,
}

/// One evidence item selected for a query-specific set-cover target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryEvidenceSetMember {
    pub hdc_id: u64,
    /// Original retrieval score, retained as metadata rather than proof of support.
    pub retrieval_score: f32,
    /// Declared cost from a pinned tokenizer and context-item framing contract.
    pub declared_token_cost: usize,
    /// Required roles this member newly covers in the greedy selection step.
    pub newly_covered_roles: Vec<String>,
}

/// Greedy query-specific evidence-set selection result.
///
/// The complete flag means all caller-declared roles have been covered by role
/// annotations; it does not mean that evidence semantically entails an answer.
/// Role extraction and query requirement planning must be independently source-
/* grounded and evaluated. The final serialized context still needs a tokenizer
budget check because token costs may not be additive across items. */
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryEvidenceSetSelection {
    pub required_roles: Vec<String>,
    pub covered_roles: Vec<String>,
    pub missing_roles: Vec<String>,
    pub members: Vec<QueryEvidenceSetMember>,
    pub declared_token_cost: usize,
    pub token_budget: usize,
    pub complete: bool,
}



impl EvidenceRecord {
    /// Structural validation; image existence and bytes are checked at use time.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != EVIDENCE_SCHEMA_VERSION {
            return Err("unsupported evidence schema version".into());
        }
        if self.evidence_id.trim().is_empty() {
            return Err("evidence_id must be non-empty".into());
        }
        if self.scope_id.trim().is_empty() {
            return Err("scope_id must be non-empty".into());
        }
        if self.source.trajectory_id.trim().is_empty() {
            return Err("source trajectory_id must be non-empty".into());
        }
        if self.source.step_id.as_ref().is_some_and(|id| id.trim().is_empty()) {
            return Err("source step_id must be non-empty when present".into());
        }
        if !is_sha256(&self.source.source_sha256) {
            return Err("source_sha256 must be 64 lowercase hexadecimal characters".into());
        }
        if !is_sha256(&self.content_sha256) {
            return Err("content_sha256 must be 64 lowercase hexadecimal characters".into());
        }
        if let (Some(start), Some(end)) = (self.valid_from_unix_ms, self.valid_until_unix_ms) {
            if start >= end {
                return Err("valid_from_unix_ms must be earlier than valid_until_unix_ms".into());
            }
        }
        match &self.payload {
            EvidencePayload::Text { value } if value.trim().is_empty() => {
                return Err("text evidence must be non-empty".into());
            }
            EvidencePayload::Image { path } => {
                if path.trim().is_empty() {
                    return Err("image path must be non-empty".into());
                }
                if Path::new(path).components().any(|part| matches!(part, Component::ParentDir)) {
                    return Err("image path must not contain parent-directory traversal".into());
                }
            }
            _ => {}
        }
        match &self.lifecycle {
            EvidenceLifecycle::Superseded => {
                let replacement = self
                    .superseded_by
                    .as_deref()
                    .filter(|value| !value.trim().is_empty())
                    .ok_or_else(|| "superseded evidence requires superseded_by".to_string())?;
                if replacement == self.evidence_id {
                    return Err("evidence cannot supersede itself".into());
                }
            }
            _ if self.superseded_by.is_some() => {
                return Err("superseded_by is only valid for superseded evidence".into());
            }
            _ => {}
        }
        Ok(())
    }

    /// Verify the provenance digest against the exact canonical source bytes
    /// supplied by the ingestion adapter. No implicit JSON reserialization is
    /// performed here: the adapter must define and pin its canonical encoding.
    pub fn verify_source_bytes(&self, source_bytes: &[u8]) -> Result<(), String> {
        self.validate()?;
        if sha256_hex(source_bytes) != self.source.source_sha256 {
            return Err("source_sha256 mismatch".into());
        }
        Ok(())
    }

    /// Whether this record is eligible at now_ms in the requested scope.
    ///
    /// Revoked, superseded, deleted, expired, not-yet-valid, malformed and
    /// cross-scope records are never eligible. Tombstones remain indexed so
    /// old vector hits cannot silently resurrect deleted evidence.
    pub fn is_eligible_for(&self, scope_id: &str, now_ms: u64) -> bool {
        self.validate().is_ok()
            && self.lifecycle == EvidenceLifecycle::Active
            && self.scope_id == scope_id
            && self.valid_from_unix_ms.is_none_or(|start| now_ms >= start)
            && self.valid_until_unix_ms.is_none_or(|end| now_ms < end)
    }

    /// Verify payload bytes and produce the official {type, value} context item.
    ///
    /// Images are canonicalized under payload_root, preventing symlink/path
    /// aliases from escaping it. Returned image paths are absolute and canonical.
    fn context_item(&self, payload_root: &Path) -> Result<MemoryContextItem, String> {
        self.validate()?;
        match &self.payload {
            EvidencePayload::Text { value } => {
                if sha256_hex(value.as_bytes()) != self.content_sha256 {
                    return Err("text content_sha256 mismatch".into());
                }
                Ok(MemoryContextItem {
                    kind: MemoryContextType::Text,
                    value: value.clone(),
                })
            }
            EvidencePayload::Image { path } => {
                let resolved = resolve_image_path(payload_root, path)?;
                let bytes = fs::read(&resolved)
                    .map_err(|error| format!("failed to read image payload: {error}"))?;
                if sha256_hex(&bytes) != self.content_sha256 {
                    return Err("image content_sha256 mismatch".into());
                }
                Ok(MemoryContextItem {
                    kind: MemoryContextType::Image,
                    value: resolved.to_string_lossy().into_owned(),
                })
            }
        }
    }
}

/// Versioned snapshot envelope for restoring the payload resolver after restart.
///
/// The unkeyed digest detects accidental/truncated/corrupted record bytes; it is
/// not an authenticity signature. A hostile-storage deployment must additionally
/// pin or authenticate the snapshot digest outside this file.
const EVIDENCE_INDEX_SNAPSHOT_SCHEMA_VERSION: &str = "BROCA-EVIDENCE-INDEX-SNAPSHOT-V1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceIndexSnapshot {
    schema_version: String,
    /// Generation supplied by the persistence coordinator; starts at 1.
    generation: u64,
    /// HDC-ID-sorted records include revoked, superseded and deleted tombstones.
    records: Vec<EvidenceRecord>,
    /// Digest covers schema version, generation, and the ordered records.
    snapshot_sha256: String,
}

/// Binds the exact vector-index bytes and evidence snapshot to one generation.
///
/// This detects accidentally mixing an old vector index with a newer payload
/// snapshot (or vice versa). It is an unkeyed consistency manifest, not a
/// signature: authenticate or externally pin its bytes when storage is hostile.
pub const EVIDENCE_COMMIT_MANIFEST_SCHEMA_VERSION: &str = "BROCA-EVIDENCE-COMMIT-MANIFEST-V3";
const LEGACY_EVIDENCE_COMMIT_MANIFEST_SCHEMA_VERSION: &str = "BROCA-EVIDENCE-COMMIT-MANIFEST-V2";

/// Identity of the actual on-disk HDC store committed alongside evidence.
/// The CRC checksum checks logical parity; exact HDC file bytes are separately
/// bound by SHA-256 in the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HdcStoreIdentity {
    pub store_generation: u64,
    pub live_count: u64,
    pub checksum_version: u32,
    pub checksum_hex: String,
}

impl HdcStoreIdentity {
    pub(crate) fn validate_for_recovery(&self) -> Result<(), String> {
        if self.store_generation == 0 || self.checksum_version == 0 {
            return Err("HDC store identity generation and checksum version must be nonzero".into());
        }
        if self.checksum_hex.len() != 16
            || !self.checksum_hex.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err("HDC store checksum must be 16 lowercase hexadecimal characters".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceCommitManifest {
    schema_version: String,
    /// Monotonic envelope generation assigned by the persistence coordinator, starting at 1.
    generation: u64,
    hdc_index_sha256: String,
    evidence_snapshot_sha256: String,
    /// Present only for the typed production commit path.
    hdc_store_identity: Option<HdcStoreIdentity>,
    /// SHA-256 of the immutable generation-bound lexical inverted index, when present.
    /// Missing in legacy V2 manifests; those generations rebuild the index from the verified snapshot.
    #[serde(default)]
    lexical_index_sha256: Option<String>,
}

impl EvidenceCommitManifest {
    /// Construct a low-level pair-binding manifest for fixtures that are not a
    /// real HdcStore. Production callers should prefer new_with_hdc_identity.
    pub fn new(generation: u64, hdc_index_bytes: &[u8], evidence_snapshot_bytes: &[u8]) -> Result<Self, String> {
        Self::new_with_optional_hdc_identity(generation, hdc_index_bytes, evidence_snapshot_bytes, None)
    }

    /// Construct a legacy V2 typed manifest that binds the HDC identity but has
    /// no committed lexical artifact. New production commits must use
    /// new_with_hdc_identity_and_lexical_index instead.
    pub fn new_with_hdc_identity(
        generation: u64,
        hdc_index_bytes: &[u8],
        evidence_snapshot_bytes: &[u8],
        identity: HdcStoreIdentity,
    ) -> Result<Self, String> {
        identity.validate_for_recovery()?;
        let mut manifest = Self::new_with_optional_hdc_identity(
            generation, hdc_index_bytes, evidence_snapshot_bytes, Some(identity),
        )?;
        manifest.schema_version = LEGACY_EVIDENCE_COMMIT_MANIFEST_SCHEMA_VERSION.to_string();
        manifest.validate_envelope()?;
        Ok(manifest)
    }

    /// Construct a typed commit manifest that also binds a serialized lexical index.
    pub fn new_with_hdc_identity_and_lexical_index(
        generation: u64,
        hdc_index_bytes: &[u8],
        evidence_snapshot_bytes: &[u8],
        identity: HdcStoreIdentity,
        lexical_index_bytes: &[u8],
    ) -> Result<Self, String> {
        if lexical_index_bytes.is_empty() {
            return Err("serialized lexical index bytes must be non-empty".into());
        }
        let mut manifest = Self::new_with_hdc_identity(
            generation, hdc_index_bytes, evidence_snapshot_bytes, identity,
        )?;
        manifest.schema_version = EVIDENCE_COMMIT_MANIFEST_SCHEMA_VERSION.to_string();
        manifest.lexical_index_sha256 = Some(sha256_hex(lexical_index_bytes));
        manifest.validate_envelope()?;
        Ok(manifest)
    }

    pub fn lexical_index_sha256(&self) -> Option<&str> {
        self.lexical_index_sha256.as_deref()
    }

    /// Verify whether the optional lexical index is present exactly as declared.
    pub fn verify_lexical_index(&self, lexical_index_bytes: Option<&[u8]>) -> Result<(), String> {
        self.validate_envelope()?;
        match (self.lexical_index_sha256.as_deref(), lexical_index_bytes) {
            (Some(expected), Some(bytes)) if sha256_hex(bytes) == expected => Ok(()),
            (Some(_), Some(_)) => Err("lexical index digest does not match commit manifest".into()),
            (Some(_), None) => Err("committed lexical index is missing".into()),
            (None, Some(_)) => Err("unmanifested lexical index artifact is present".into()),
            (None, None) => Ok(()),
        }
    }

    fn new_with_optional_hdc_identity(
        generation: u64,
        hdc_index_bytes: &[u8],
        evidence_snapshot_bytes: &[u8],
        identity: Option<HdcStoreIdentity>,
    ) -> Result<Self, String> {
        if generation == 0 {
            return Err("evidence commit generation must start at 1".into());
        }
        let snapshot: EvidenceIndexSnapshot = serde_json::from_slice(evidence_snapshot_bytes)
            .map_err(|error| format!("invalid evidence snapshot: {error}"))?;
        if snapshot.generation != generation {
            return Err("evidence snapshot generation does not match commit manifest".into());
        }
        EvidenceIndex::restore_snapshot(evidence_snapshot_bytes)?;
        if let Some(identity) = &identity {
            identity.validate_for_recovery()?;
        }
        Ok(Self {
            schema_version: EVIDENCE_COMMIT_MANIFEST_SCHEMA_VERSION.to_string(),
            generation,
            hdc_index_sha256: sha256_hex(hdc_index_bytes),
            evidence_snapshot_sha256: sha256_hex(evidence_snapshot_bytes),
            hdc_store_identity: identity,
            lexical_index_sha256: None,
        })
    }

    pub fn hdc_store_identity(&self) -> Option<&HdcStoreIdentity> {
        self.hdc_store_identity.as_ref()
    }

    fn validate_envelope(&self) -> Result<(), String> {
        match self.schema_version.as_str() {
            EVIDENCE_COMMIT_MANIFEST_SCHEMA_VERSION => {
                if self.hdc_store_identity.is_some() && self.lexical_index_sha256.is_none() {
                    return Err("V3 typed manifest must bind a lexical-index digest".into());
                }
            }
            LEGACY_EVIDENCE_COMMIT_MANIFEST_SCHEMA_VERSION => {
                if self.lexical_index_sha256.is_some() {
                    return Err("legacy V2 manifest cannot declare a lexical-index digest".into());
                }
            }
            _ => return Err("unsupported evidence commit manifest schema version".into()),
        }
        if self.generation == 0 {
            return Err("evidence commit generation must start at 1".into());
        }
        if !is_sha256(&self.hdc_index_sha256) || !is_sha256(&self.evidence_snapshot_sha256) {
            return Err("evidence commit manifest contains an invalid SHA-256 digest".into());
        }
        if let Some(identity) = &self.hdc_store_identity {
            identity.validate_for_recovery()?;
        }
        if self.lexical_index_sha256.as_deref().is_some_and(|digest| !is_sha256(digest)) {
            return Err("evidence commit manifest contains an invalid lexical-index SHA-256 digest".into());
        }
        Ok(())
    }

    /// Check both artifact digests and validate the evidence snapshot itself.
    pub fn verify(&self, hdc_index_bytes: &[u8], evidence_snapshot_bytes: &[u8]) -> Result<(), String> {
        self.validate_envelope()?;
        if sha256_hex(hdc_index_bytes) != self.hdc_index_sha256 {
            return Err("HDC index digest does not match evidence commit manifest".into());
        }
        if sha256_hex(evidence_snapshot_bytes) != self.evidence_snapshot_sha256 {
            return Err("evidence snapshot digest does not match commit manifest".into());
        }
        let snapshot: EvidenceIndexSnapshot = serde_json::from_slice(evidence_snapshot_bytes)
            .map_err(|error| format!("invalid evidence snapshot: {error}"))?;
        if snapshot.generation != self.generation {
            return Err("evidence snapshot generation does not match commit manifest".into());
        }
        EvidenceIndex::restore_snapshot(evidence_snapshot_bytes)?;
        Ok(())
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate_envelope()?;
        serde_json::to_vec(self).map_err(|error| format!("failed to serialize commit manifest: {error}"))
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let manifest: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid evidence commit manifest: {error}"))?;
        manifest.validate_envelope()?;
        Ok(manifest)
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// Map from HDC candidate IDs to evidence payloads.
///
/// Snapshot/restore preserves lifecycle tombstones across process restarts, but
/// this is not a disk transaction and cannot atomically commit with the HDC
/// vector index. Callers must coordinate both stores and durably pin/authenticate
/// the returned snapshot bytes as required by their threat model.
/// Versioned immutable inverted index derived from one evidence snapshot.
///
/// The manifest binds the exact serialized bytes; generation and structural
/// validation prevent accidentally using an index from another commit.
pub const LEXICAL_INDEX_SCHEMA_VERSION: &str = "BROCA-LEXICAL-INDEX-V1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LexicalIndex {
    schema_version: String,
    generation: u64,
    /// Text payload ID -> token count. IDs include text records with no tokens.
    documents: BTreeMap<u64, LexicalDocument>,
    /// Token -> ascending HDC-ID postings.
    postings: BTreeMap<String, Vec<LexicalPosting>>,
    total_token_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LexicalDocument {
    token_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LexicalPosting {
    hdc_id: u64,
    term_frequency: u32,
}

#[derive(Debug, Default)]
pub struct EvidenceIndex {
    records: BTreeMap<u64, EvidenceRecord>,
    evidence_ids: BTreeMap<String, u64>,
    /// Derived adjacency index includes tombstones; query-time eligibility is
    /// always rechecked against the canonical record before a neighbor is emitted.
    trajectory_state_ids: BTreeMap<(String, u64), BTreeSet<u64>>,
}

impl LexicalIndex {
    /// Build an inverted index from verified text payloads in a specific
    /// evidence generation. This is done during commit, never per query.
    pub fn build(evidence: &EvidenceIndex, generation: u64) -> Result<Self, String> {
        if generation == 0 {
            return Err("lexical index generation must start at 1".into());
        }
        let mut documents = BTreeMap::new();
        let mut postings: BTreeMap<String, Vec<LexicalPosting>> = BTreeMap::new();
        let mut total_token_count = 0u64;

        for (hdc_id, record) in &evidence.records {
            let EvidencePayload::Text { value } = &record.payload else {
                continue;
            };
            if sha256_hex(value.as_bytes()) != record.content_sha256 {
                return Err(format!("cannot index text with invalid content digest at HDC ID {hdc_id}"));
            }
            let tokens = lexical_tokens(value);
            let mut frequencies = BTreeMap::<String, u32>::new();
            for token in &tokens {
                let count = frequencies.entry(token.clone()).or_default();
                *count = count.checked_add(1)
                    .ok_or_else(|| format!("lexical term frequency overflow at HDC ID {hdc_id}"))?;
            }
            let token_count = u64::try_from(tokens.len())
                .map_err(|_| "lexical document token count overflow".to_string())?;
            total_token_count = total_token_count.checked_add(token_count)
                .ok_or_else(|| "lexical index token count overflow".to_string())?;
            documents.insert(*hdc_id, LexicalDocument { token_count });
            for (term, term_frequency) in frequencies {
                postings.entry(term).or_default().push(LexicalPosting {
                    hdc_id: *hdc_id,
                    term_frequency,
                });
            }
        }

        let index = Self {
            schema_version: LEXICAL_INDEX_SCHEMA_VERSION.to_string(),
            generation,
            documents,
            postings,
            total_token_count,
        };
        index.validate_structure()?;
        Ok(index)
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn to_json_bytes(&self) -> Result<Vec<u8>, String> {
        self.validate_structure()?;
        serde_json::to_vec(self).map_err(|error| format!("failed to encode lexical index: {error}"))
    }

    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let index: Self = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid lexical index: {error}"))?;
        index.validate_structure()?;
        Ok(index)
    }

    /// Ensure the index is for this exact logical evidence generation and has
    /// exactly one document entry for each text record (including zero-token text).
    pub fn validate_for_evidence(&self, evidence: &EvidenceIndex, generation: u64) -> Result<(), String> {
        self.validate_structure()?;
        if self.generation != generation {
            return Err("lexical index generation does not match evidence generation".into());
        }
        let expected: BTreeSet<u64> = evidence.records.iter()
            .filter_map(|(id, record)| matches!(&record.payload, EvidencePayload::Text { .. }).then_some(*id))
            .collect();
        let actual: BTreeSet<u64> = self.documents.keys().copied().collect();
        if expected != actual {
            return Err("lexical index text-document IDs do not match evidence snapshot".into());
        }
        Ok(())
    }

    fn validate_structure(&self) -> Result<(), String> {
        if self.schema_version != LEXICAL_INDEX_SCHEMA_VERSION {
            return Err("unsupported lexical index schema version".into());
        }
        if self.generation == 0 {
            return Err("lexical index generation must start at 1".into());
        }
        let mut summed_lengths = 0u64;
        let mut per_document_frequency = BTreeMap::<u64, u64>::new();
        for (id, document) in &self.documents {
            summed_lengths = summed_lengths.checked_add(document.token_count)
                .ok_or_else(|| "lexical index token count overflow".to_string())?;
            per_document_frequency.insert(*id, 0);
        }
        if summed_lengths != self.total_token_count {
            return Err("lexical index total token count does not match document lengths".into());
        }
        for (term, list) in &self.postings {
            let canonical_term = lexical_tokens(term);
            if canonical_term.len() != 1 || canonical_term[0] != *term {
                return Err("lexical index contains a non-canonical term".into());
            }
            if list.is_empty() {
                return Err("lexical index contains an empty posting list".into());
            }
            let mut previous = None;
            for posting in list {
                if posting.term_frequency == 0 {
                    return Err("lexical posting term frequency must be positive".into());
                }
                if previous.is_some_and(|id| posting.hdc_id <= id) {
                    return Err("lexical postings must be strictly ordered by HDC ID".into());
                }
                previous = Some(posting.hdc_id);
                let document = self.documents.get(&posting.hdc_id)
                    .ok_or_else(|| "lexical posting references an unknown document".to_string())?;
                if u64::from(posting.term_frequency) > document.token_count {
                    return Err("lexical term frequency exceeds document token count".into());
                }
                let count = per_document_frequency.get_mut(&posting.hdc_id)
                    .ok_or_else(|| "lexical posting references a non-text document".to_string())?;
                *count = count.checked_add(u64::from(posting.term_frequency))
                    .ok_or_else(|| "lexical document frequency sum overflow".to_string())?;
            }
        }
        for (id, document) in &self.documents {
            if per_document_frequency[id] != document.token_count {
                return Err("lexical postings do not account for each document token".into());
            }
        }
        Ok(())
    }
}

impl EvidenceIndex {
    fn index_trajectory_state(&mut self, record: &EvidenceRecord) {
        if let Some(step_index) = record.source.step_index {
            self.trajectory_state_ids
                .entry((record.source.trajectory_id.clone(), step_index))
                .or_default()
                .insert(record.hdc_id);
        }
    }

    pub fn new() -> Self {
        Self::default()
    }

    /// Encode a deterministic versioned snapshot.
    ///
    /// Records are serialized in ascending HDC ID order. The embedded digest
    /// covers the canonical serialized record vector and detects partial or
    /// accidental changes; it does not authenticate the writer.
    pub fn snapshot_bytes(&self, generation: u64) -> Result<Vec<u8>, String> {
        if generation == 0 {
            return Err("evidence snapshot generation must start at 1".into());
        }
        let records: Vec<EvidenceRecord> = self.records.values().cloned().collect();
        let mut evidence_ids = BTreeSet::new();

        for record in &records {
            record.validate()?;
            if !evidence_ids.insert(record.evidence_id.clone()) {
                return Err("snapshot contains duplicate evidence_id".into());
            }
            if let EvidencePayload::Text { value } = &record.payload {
                if sha256_hex(value.as_bytes()) != record.content_sha256 {
                    return Err("snapshot contains text content_sha256 mismatch".into());
                }
            }
        }

        // Every supersession edge must remain resolvable within this snapshot.
        for record in &records {
            if record.lifecycle == EvidenceLifecycle::Superseded {
                let replacement = record.superseded_by.as_deref()
                    .ok_or_else(|| "superseded evidence requires superseded_by".to_string())?;
                let replacement_record = records.iter()
                    .find(|candidate| candidate.evidence_id == replacement)
                    .ok_or_else(|| "snapshot is missing superseding evidence".to_string())?;
                if replacement_record.scope_id != record.scope_id {
                    return Err("snapshot supersession crosses scopes".into());
                }
            }
        }

        // Bind schema and generation into the digest, not only the record vector.
        let snapshot_body = serde_json::to_vec(&(
            EVIDENCE_INDEX_SNAPSHOT_SCHEMA_VERSION,
            generation,
            &records,
        )).map_err(|error| format!("failed to serialize evidence snapshot body: {error}"))?;
        let snapshot = EvidenceIndexSnapshot {
            schema_version: EVIDENCE_INDEX_SNAPSHOT_SCHEMA_VERSION.to_string(),
            generation,
            records,
            snapshot_sha256: sha256_hex(&snapshot_body),
        };
        serde_json::to_vec(&snapshot)
            .map_err(|error| format!("failed to serialize evidence snapshot: {error}"))
    }

    /// Restore a snapshot while rebuilding all derived ID maps from records.
    ///
    /// This validates the format, record structure, text payload digests,
    /// duplicate identities and supersession references. Source provenance
    /// digests are retained but cannot be rechecked without the original source
    /// bytes; image payload bytes are rechecked under payload_root at resolve.
    pub fn restore_snapshot(bytes: &[u8]) -> Result<Self, String> {
        let snapshot: EvidenceIndexSnapshot = serde_json::from_slice(bytes)
            .map_err(|error| format!("invalid evidence snapshot: {error}"))?;
        if snapshot.schema_version != EVIDENCE_INDEX_SNAPSHOT_SCHEMA_VERSION {
            return Err("unsupported evidence snapshot schema version".into());
        }
        if snapshot.generation == 0 {
            return Err("evidence snapshot generation must start at 1".into());
        }
        if !is_sha256(&snapshot.snapshot_sha256) {
            return Err("snapshot_sha256 must be lowercase SHA-256".into());
        }
        let snapshot_body = serde_json::to_vec(&(
            &snapshot.schema_version,
            snapshot.generation,
            &snapshot.records,
        )).map_err(|error| format!("failed to serialize restored snapshot body: {error}"))?;
        if sha256_hex(&snapshot_body) != snapshot.snapshot_sha256 {
            return Err("evidence snapshot digest mismatch".into());
        }

        let mut index = Self::new();
        for record in snapshot.records {
            record.validate()?;
            if let EvidencePayload::Text { value } = &record.payload {
                if sha256_hex(value.as_bytes()) != record.content_sha256 {
                    return Err("restored text content_sha256 mismatch".into());
                }
            }
            if index.records.contains_key(&record.hdc_id) {
                return Err("snapshot contains duplicate hdc_id".into());
            }
            if index.evidence_ids.contains_key(&record.evidence_id) {
                return Err("snapshot contains duplicate evidence_id".into());
            }
            index.index_trajectory_state(&record);
            index.evidence_ids.insert(record.evidence_id.clone(), record.hdc_id);
            index.records.insert(record.hdc_id, record);
        }

        for record in index.records.values() {
            if record.lifecycle == EvidenceLifecycle::Superseded {
                let replacement = record.superseded_by.as_deref()
                    .ok_or_else(|| "superseded evidence requires superseded_by".to_string())?;
                let replacement_hdc_id = index.evidence_ids.get(replacement)
                    .ok_or_else(|| "snapshot is missing superseding evidence".to_string())?;
                let replacement_record = index.records.get(replacement_hdc_id)
                    .ok_or_else(|| "snapshot supersession index is inconsistent".to_string())?;
                if replacement_record.scope_id != record.scope_id {
                    return Err("snapshot supersession crosses scopes".into());
                }
            }
        }
        Ok(index)
    }

    /// IDs whose evidence lifecycle is active. This set must exactly match the
    /// live IDs in the real HDC index for a production commit.
    pub fn active_hdc_ids(&self) -> BTreeSet<u64> {
        self.records.iter()
            .filter_map(|(hdc_id, record)| (record.lifecycle == EvidenceLifecycle::Active).then_some(*hdc_id))
            .collect()
    }

    /// IDs retained as payload lineage, including inactive tombstones.
    pub fn all_hdc_ids(&self) -> BTreeSet<u64> {
        self.records.keys().copied().collect()
    }

    pub fn insert(&mut self, record: EvidenceRecord, source_bytes: &[u8]) -> Result<(), String> {
        // Verify provenance before the record enters the candidate resolver.
        record.verify_source_bytes(source_bytes)?;
        if self.records.contains_key(&record.hdc_id) {
            return Err("duplicate hdc_id; use an explicit lifecycle transition".into());
        }
        if self.evidence_ids.contains_key(&record.evidence_id) {
            return Err("duplicate evidence_id".into());
        }
        self.index_trajectory_state(&record);
        self.evidence_ids.insert(record.evidence_id.clone(), record.hdc_id);
        self.records.insert(record.hdc_id, record);
        Ok(())
    }

    pub fn revoke(&mut self, hdc_id: u64) -> Result<(), String> {
        let record = self.records.get_mut(&hdc_id).ok_or_else(|| "unknown hdc_id".to_string())?;
        if record.lifecycle == EvidenceLifecycle::Deleted {
            return Err("deleted evidence cannot transition back to revoked".into());
        }
        record.lifecycle = EvidenceLifecycle::Revoked;
        record.superseded_by = None;
        Ok(())
    }

    /// Retain a tombstone rather than removing the ID-to-evidence mapping.
    pub fn mark_deleted(&mut self, hdc_id: u64) -> Result<(), String> {
        let record = self.records.get_mut(&hdc_id).ok_or_else(|| "unknown hdc_id".to_string())?;
        record.lifecycle = EvidenceLifecycle::Deleted;
        record.superseded_by = None;
        Ok(())
    }

    /// Replace active evidence with another active record in the same scope.
    pub fn supersede(&mut self, old_hdc_id: u64, new_hdc_id: u64) -> Result<(), String> {
        if old_hdc_id == new_hdc_id {
            return Err("old and new hdc_id must differ".into());
        }
        let (new_scope, new_evidence_id, new_lifecycle) = self
            .records
            .get(&new_hdc_id)
            .map(|record| (
                record.scope_id.clone(),
                record.evidence_id.clone(),
                record.lifecycle.clone(),
            ))
            .ok_or_else(|| "unknown replacement hdc_id".to_string())?;
        if new_lifecycle != EvidenceLifecycle::Active {
            return Err("replacement evidence must be active".into());
        }
        let old = self.records.get_mut(&old_hdc_id).ok_or_else(|| "unknown old hdc_id".to_string())?;
        if old.lifecycle != EvidenceLifecycle::Active {
            return Err("only active evidence can be superseded".into());
        }
        if old.scope_id != new_scope {
            return Err("evidence cannot be superseded across scopes".into());
        }
        old.lifecycle = EvidenceLifecycle::Superseded;
        old.superseded_by = Some(new_evidence_id);
        old.validate()
    }

    /// BM25 over a previously built inverted index. This avoids re-tokenizing
    /// and re-hashing every evidence payload on each query. Eligibility and
    /// corpus statistics are still evaluated against current evidence metadata,
    /// because lifecycle and validity windows can change between queries.
    pub(crate) fn rank_text_indexed(
        &self,
        index: &LexicalIndex,
        expected_generation: u64,
        query_text: &str,
        scope_id: &str,
        now_ms: u64,
        max_candidates: usize,
    ) -> Result<Vec<(u64, f32)>, String> {
        if scope_id.trim().is_empty() {
            return Err("scope_id must be non-empty".into());
        }
        if expected_generation == 0 || index.generation != expected_generation {
            return Err("lexical index generation does not match committed evidence generation".into());
        }
        if max_candidates == 0 {
            return Ok(Vec::new());
        }
        let query_terms: BTreeSet<String> = lexical_tokens(query_text).into_iter().collect();
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        const K1: f64 = 1.2;
        const B: f64 = 0.75;
        let mut eligible_documents = 0usize;
        let mut total_length = 0u64;
        for (hdc_id, record) in &self.records {
            if !record.is_eligible_for(scope_id, now_ms) {
                continue;
            }
            let EvidencePayload::Text { .. } = &record.payload else {
                continue;
            };
            let document = index.documents.get(hdc_id)
                .ok_or_else(|| format!("lexical index missing eligible text HDC ID {hdc_id}"))?;
            if document.token_count == 0 {
                continue;
            }
            eligible_documents += 1;
            total_length = total_length.checked_add(document.token_count)
                .ok_or_else(|| "eligible lexical token count overflow".to_string())?;
        }
        if eligible_documents == 0 {
            return Ok(Vec::new());
        }
        let average_length = total_length as f64 / eligible_documents as f64;
        if average_length <= 0.0 {
            return Ok(Vec::new());
        }

        let eligible = |id: u64| -> bool {
            self.records.get(&id).is_some_and(|record| record.is_eligible_for(scope_id, now_ms))
                && index.documents.contains_key(&id)
                && index.documents.get(&id).is_some_and(|document| document.token_count > 0)
        };
        let mut document_frequency = BTreeMap::<String, usize>::new();
        let mut scores = BTreeMap::<u64, f64>::new();
        for term in &query_terms {
            let Some(postings) = index.postings.get(term) else {
                continue;
            };
            let mut df = 0usize;
            for posting in postings {
                if !eligible(posting.hdc_id) {
                    continue;
                }
                df += 1;
                *scores.entry(posting.hdc_id).or_default();
            }
            document_frequency.insert(term.clone(), df);
        }

        // Iterate terms in lexical order so each document's floating-point
        // accumulation is deterministic and matches the reference implementation.
        for term in &query_terms {
            let Some(postings) = index.postings.get(term) else {
                continue;
            };
            let df = *document_frequency.get(term).unwrap_or(&0) as f64;
            if df == 0.0 {
                continue;
            }
            let n = eligible_documents as f64;
            let inverse_document_frequency = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for posting in postings {
                if !eligible(posting.hdc_id) {
                    continue;
                }
                let document_length = index.documents[&posting.hdc_id].token_count as f64;
                let term_frequency = f64::from(posting.term_frequency);
                let length_norm = 1.0 - B + B * (document_length / average_length);
                let contribution = inverse_document_frequency
                    * (term_frequency * (K1 + 1.0))
                    / (term_frequency + K1 * length_norm);
                *scores.entry(posting.hdc_id).or_default() += contribution;
            }
        }
        let mut ranked: Vec<(u64, f32)> = scores.into_iter()
            .filter_map(|(id, score)| (score > 0.0).then_some((id, score as f32)))
            .collect();
        ranked.sort_unstable_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(max_candidates);
        Ok(ranked)
    }

    /// Diversify a previously ranked lexical candidate pool by trajectory.
    ///
    /// The caller should oversample the BM25 ranking before applying this policy.
    /// The selector preserves score order (score descending, HDC ID ascending),
    /// emits at most `per_trajectory_limit` text records per source trajectory,
    /// and rechecks scope, lifecycle and time eligibility at selection time.
    ///
    /// This is an opt-in retrieval policy, not part of the BM25 score or an
    /// accuracy claim. It can reduce duplicate sibling-state context, but it may
    /// also hurt questions that need several states from one trajectory; evaluate
    /// by memory ability and equal context budget before selecting a default.
    pub(crate) fn diversify_text_candidates_by_trajectory(
        &self,
        ranked_candidates: &[(u64, f32)],
        scope_id: &str,
        now_ms: u64,
        max_candidates: usize,
        per_trajectory_limit: usize,
    ) -> Result<Vec<(u64, f32)>, String> {
        if scope_id.trim().is_empty() {
            return Err("scope_id must be non-empty".into());
        }
        if per_trajectory_limit == 0 {
            return Err("per_trajectory_limit must be positive".into());
        }
        if max_candidates == 0 || ranked_candidates.is_empty() {
            return Ok(Vec::new());
        }

        let mut ordered = ranked_candidates.to_vec();
        if ordered.iter().any(|(_, score)| !score.is_finite() || *score < 0.0) {
            return Err("candidate scores must be finite and non-negative".into());
        }
        ordered.sort_unstable_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });

        let mut seen_ids = BTreeSet::new();
        let mut trajectory_counts = BTreeMap::<String, usize>::new();
        let mut selected = Vec::with_capacity(max_candidates.min(ordered.len()));
        for (hdc_id, score) in ordered {
            if !seen_ids.insert(hdc_id) {
                continue;
            }
            let record = self.records.get(&hdc_id)
                .ok_or_else(|| format!("ranked lexical candidate references unknown HDC ID {hdc_id}"))?;
            if !record.is_eligible_for(scope_id, now_ms)
                || !matches!(&record.payload, EvidencePayload::Text { .. })
            {
                continue;
            }
            let trajectory_id = record.source.trajectory_id.clone();
            let count = trajectory_counts.entry(trajectory_id).or_default();
            if *count >= per_trajectory_limit {
                continue;
            }
            *count += 1;
            selected.push((hdc_id, score));
            if selected.len() == max_candidates {
                break;
            }
        }
        Ok(selected)
    }

    /// Pack ranked lexical candidates within a caller-supplied context-token budget.
    ///
    /// Costs must be produced by the pinned model tokenizer and canonical context-item
    /// serializer, not inferred from UTF-8 length. Because tokenization across item
    /// boundaries may not be additive, callers must also check the final serialized
    /// context token count before sending it to a reader. This method enforces the
    /// declared additive cost contract; it does not replace that final guard.
    ///
    /// Candidates are considered by score descending, then HDC ID ascending. An item
    /// that does not fit is skipped rather than ending the scan, allowing smaller
    /// lower-ranked evidence to use remaining budget. Optional trajectory caps are
    /// opt-in and count only selected, eligible text items.
    ///
    /// The caller should oversample the ranking before packing. No quality gain is
    /// assumed: packing may trade a top-ranked long item for several lower-ranked
    /// short items and must be evaluated under an equal final serialized-token budget.
    pub(crate) fn pack_text_candidates_under_token_budget(
        &self,
        ranked_candidates: &[(u64, f32)],
        candidate_token_costs: &BTreeMap<u64, usize>,
        scope_id: &str,
        now_ms: u64,
        token_budget: usize,
        max_candidates: usize,
        per_trajectory_limit: Option<usize>,
    ) -> Result<Vec<(u64, f32)>, String> {
        if scope_id.trim().is_empty() {
            return Err("scope_id must be non-empty".into());
        }
        if per_trajectory_limit.is_some_and(|limit| limit == 0) {
            return Err("per_trajectory_limit must be positive when present".into());
        }

        let mut ordered = ranked_candidates.to_vec();
        if ordered.iter().any(|(_, score)| !score.is_finite() || *score < 0.0) {
            return Err("candidate scores must be finite and non-negative".into());
        }
        ordered.sort_unstable_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });

        // Validate the entire unique candidate pool before selection. An unknown ID
        // or missing/zero declared cost must not become harmless merely because its
        // rank, scope, lifecycle, or size would otherwise exclude it.
        let mut validated_ids = BTreeSet::new();
        for (hdc_id, _) in &ordered {
            if !validated_ids.insert(*hdc_id) {
                continue;
            }
            if !self.records.contains_key(hdc_id) {
                return Err(format!("ranked lexical candidate references unknown HDC ID {hdc_id}"));
            }
            let cost = candidate_token_costs.get(hdc_id)
                .ok_or_else(|| format!("missing token cost for HDC ID {hdc_id}"))?;
            if *cost == 0 {
                return Err(format!("token cost must be positive for HDC ID {hdc_id}"));
            }
        }
        if token_budget == 0 || max_candidates == 0 || ordered.is_empty() {
            return Ok(Vec::new());
        }

        let mut seen_ids = BTreeSet::new();
        let mut trajectory_counts = BTreeMap::<String, usize>::new();
        let mut used_cost = 0usize;
        let mut selected = Vec::with_capacity(max_candidates.min(ordered.len()));
        for (hdc_id, score) in ordered {
            if !seen_ids.insert(hdc_id) {
                continue;
            }
            let record = self.records.get(&hdc_id)
                .ok_or_else(|| format!("ranked lexical candidate references unknown HDC ID {hdc_id}"))?;
            if !record.is_eligible_for(scope_id, now_ms)
                || !matches!(&record.payload, EvidencePayload::Text { .. })
            {
                continue;
            }

            if let Some(limit) = per_trajectory_limit {
                let count = trajectory_counts.get(&record.source.trajectory_id).copied().unwrap_or(0);
                if count >= limit {
                    continue;
                }
            }

            let token_cost = *candidate_token_costs.get(&hdc_id)
                .ok_or_else(|| format!("missing token cost for HDC ID {hdc_id}"))?;
            let remaining = token_budget - used_cost;
            if token_cost > remaining {
                continue;
            }

            used_cost += token_cost;
            if per_trajectory_limit.is_some() {
                *trajectory_counts.entry(record.source.trajectory_id.clone()).or_default() += 1;
            }
            selected.push((hdc_id, score));
            if selected.len() == max_candidates {
                break;
            }
        }
        Ok(selected)
    }

    /// Expand ranked seeds with immediate same-trajectory predecessor/successor states.
    ///
    /// Seeds remain first and retain their rank order. Adjacent records are appended
    /// in seed order, predecessor before successor, and carry the seed ID that caused
    /// their inclusion. The method never invents an adjacent record's relevance score.
    ///
    /// Only an eligible text record at the exact neighboring state index may expand.
    /// Missing states are not bridged, cross-trajectory and cross-scope records are
    /// excluded, and multiple eligible records at one trajectory/state coordinate
    /// cause a fail-closed ambiguity error. Tombstones remain in the index but are
    /// filtered at selection time. The caller must apply the final token-budget guard.
    ///
    /// This is an opt-in policy for temporal-dynamics and workflow questions; static
    /// state recall may regress if unnecessary neighbors are included. Measure by
    /// ability and equal serialized-token budget before enabling by default.
    pub(crate) fn expand_text_candidates_with_adjacent_states(
        &self,
        ranked_seeds: &[(u64, f32)],
        scope_id: &str,
        now_ms: u64,
        max_candidates: usize,
        max_neighbors_per_seed: usize,
    ) -> Result<Vec<AdjacentStateCandidate>, String> {
        if scope_id.trim().is_empty() {
            return Err("scope_id must be non-empty".into());
        }
        if ranked_seeds.iter().any(|(_, score)| !score.is_finite() || *score < 0.0) {
            return Err("candidate scores must be finite and non-negative".into());
        }

        let mut ordered = ranked_seeds.to_vec();
        ordered.sort_unstable_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });

        // Validate every unique seed ID before truncation; malformed lower-ranked
        // candidates must not be hidden by max_candidates.
        let mut seen_seeds = BTreeSet::new();
        for (hdc_id, _) in &ordered {
            if !seen_seeds.insert(*hdc_id) {
                continue;
            }
            if !self.records.contains_key(hdc_id) {
                return Err(format!("ranked memory seed references unknown HDC ID {hdc_id}"));
            }
        }
        if max_candidates == 0 || ordered.is_empty() {
            return Ok(Vec::new());
        }

        let mut selected = Vec::with_capacity(max_candidates.min(ordered.len()));
        let mut selected_ids = BTreeSet::new();

        // Preserve the strongest eligible seeds before spending any slots on
        // expansion. An ineligible seed is skipped and cannot authorize neighbors.
        for (hdc_id, _) in &ordered {
            if !selected_ids.insert(*hdc_id) {
                continue;
            }
            let record = self.records.get(hdc_id)
                .ok_or_else(|| format!("ranked memory seed references unknown HDC ID {hdc_id}"))?;
            if !record.is_eligible_for(scope_id, now_ms)
                || !matches!(&record.payload, EvidencePayload::Text { .. })
            {
                selected_ids.remove(hdc_id);
                continue;
            }
            selected.push(AdjacentStateCandidate {
                hdc_id: *hdc_id,
                anchor_hdc_id: None,
            });
            if selected.len() == max_candidates {
                return Ok(selected);
            }
        }

        if max_neighbors_per_seed == 0 || selected.len() == max_candidates {
            return Ok(selected);
        }

        // Neighbors are appended only after all retrieval seeds, so expansion
        // cannot displace a higher-ranked seed. The ordered state index is derived
        // on insert/restore; lifecycle is still checked below at query time.
        let seed_ids: Vec<u64> = selected.iter().map(|candidate| candidate.hdc_id).collect();
        for seed_id in seed_ids {
            if selected.len() == max_candidates {
                break;
            }
            let seed = self.records.get(&seed_id)
                .ok_or_else(|| format!("selected seed disappeared: {seed_id}"))?;
            let Some(step_index) = seed.source.step_index else {
                continue;
            };
            let trajectory_id = seed.source.trajectory_id.clone();
            let mut neighbor_steps = Vec::with_capacity(2);
            if let Some(previous) = step_index.checked_sub(1) {
                neighbor_steps.push(previous);
            }
            if let Some(next) = step_index.checked_add(1) {
                neighbor_steps.push(next);
            }

            let mut added_for_seed = 0usize;
            for neighbor_step in neighbor_steps {
                if selected.len() == max_candidates || added_for_seed == max_neighbors_per_seed {
                    break;
                }
                let key = (trajectory_id.clone(), neighbor_step);
                let Some(ids) = self.trajectory_state_ids.get(&key) else {
                    continue;
                };

                let mut eligible_text_ids = ids.iter().copied().filter(|candidate_id| {
                    self.records.get(candidate_id).is_some_and(|record| {
                        record.is_eligible_for(scope_id, now_ms)
                            && matches!(&record.payload, EvidencePayload::Text { .. })
                    })
                });
                let Some(neighbor_id) = eligible_text_ids.next() else {
                    continue;
                };
                if eligible_text_ids.next().is_some() {
                    return Err(format!(
                        "ambiguous eligible adjacent states at trajectory {:?}, state index {}",
                        trajectory_id, neighbor_step
                    ));
                }
                if !selected_ids.insert(neighbor_id) {
                    continue;
                }

                selected.push(AdjacentStateCandidate {
                    hdc_id: neighbor_id,
                    anchor_hdc_id: Some(seed_id),
                });
                added_for_seed += 1;
            }
        }
        Ok(selected)
    }

    /// Deterministic BM25 lexical retrieval over eligible text evidence.
    ///
    /// This is a transparent lexical baseline, not a semantic encoder. It reads
    /// only validated active records in the requested scope and validity window;
    /// revoked, superseded, deleted, expired, and cross-scope items do not
    /// influence document statistics or ranking. Text payload digests are
    /// verified before indexing. Image payloads are excluded from this arm.
    pub fn rank_text(
        &self,
        query_text: &str,
        scope_id: &str,
        now_ms: u64,
        max_candidates: usize,
    ) -> Result<Vec<(u64, f32)>, String> {
        if scope_id.trim().is_empty() {
            return Err("scope_id must be non-empty".into());
        }
        if max_candidates == 0 {
            return Ok(Vec::new());
        }
        let query_terms: BTreeSet<String> = lexical_tokens(query_text).into_iter().collect();
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        const K1: f64 = 1.2;
        const B: f64 = 0.75;
        let mut documents: Vec<(u64, BTreeMap<String, usize>, usize)> = Vec::new();
        let mut document_frequency: BTreeMap<String, usize> = BTreeMap::new();
        let mut total_length = 0usize;

        for (hdc_id, record) in &self.records {
            if !record.is_eligible_for(scope_id, now_ms) {
                continue;
            }
            let EvidencePayload::Text { value } = &record.payload else {
                continue;
            };
            if sha256_hex(value.as_bytes()) != record.content_sha256 {
                return Err(format!("text content_sha256 mismatch for eligible evidence HDC ID {hdc_id}"));
            }
            let tokens = lexical_tokens(value);
            if tokens.is_empty() {
                continue;
            }
            let mut frequencies = BTreeMap::<String, usize>::new();
            for token in &tokens {
                *frequencies.entry(token.clone()).or_default() += 1;
            }
            for token in frequencies.keys() {
                *document_frequency.entry(token.clone()).or_default() += 1;
            }
            total_length = total_length.saturating_add(tokens.len());
            documents.push((*hdc_id, frequencies, tokens.len()));
        }

        let document_count = documents.len();
        if document_count == 0 {
            return Ok(Vec::new());
        }
        let average_length = total_length as f64 / document_count as f64;
        if average_length <= 0.0 {
            return Ok(Vec::new());
        }

        let mut ranked = Vec::with_capacity(documents.len());
        for (hdc_id, frequencies, document_length) in documents {
            let length_norm = 1.0 - B + B * (document_length as f64 / average_length);
            let mut score = 0.0f64;
            for term in &query_terms {
                let Some(&tf) = frequencies.get(term) else {
                    continue;
                };
                let df = *document_frequency.get(term).unwrap_or(&0) as f64;
                let n = document_count as f64;
                let inverse_document_frequency = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                let tf = tf as f64;
                score += inverse_document_frequency * (tf * (K1 + 1.0)) / (tf + K1 * length_norm);
            }
            if score > 0.0 {
                ranked.push((hdc_id, score as f32));
            }
        }
        ranked.sort_unstable_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(max_candidates);
        Ok(ranked)
    }

    /// Resolve ranked HDC IDs to safe context items with source provenance.
    ///
    /// Missing IDs, duplicates, inactive records, invalid times and cross-scope
    /// records are skipped. A present but corrupt payload is an error, not
    /// accepted evidence. The order follows candidate ranking after ineligible
    /// records are omitted.
    pub fn resolve_with_provenance(
        &self,
        hdc_hits: &[u64],
        scope_id: &str,
        now_ms: u64,
        payload_root: &Path,
        max_items: usize,
    ) -> Result<Vec<ResolvedMemoryContextItem>, String> {
        if scope_id.trim().is_empty() {
            return Err("scope_id must be non-empty".into());
        }
        if max_items == 0 {
            return Ok(Vec::new());
        }
        let mut seen = BTreeSet::new();
        let mut output = Vec::new();
        for hdc_id in hdc_hits {
            if output.len() >= max_items {
                break;
            }
            if !seen.insert(*hdc_id) {
                continue;
            }
            let Some(record) = self.records.get(hdc_id) else {
                // A stale vector hit with no payload mapping must never turn
                // into evidence via a fallback to caller-supplied text.
                continue;
            };
            if !record.is_eligible_for(scope_id, now_ms) {
                continue;
            }
            output.push(ResolvedMemoryContextItem {
                hdc_id: *hdc_id,
                evidence_id: record.evidence_id.clone(),
                source: record.source.clone(),
                item: record.context_item(payload_root)?,
            });
        }
        Ok(output)
    }

    /// Resolve candidates and report why every input hit was emitted or excluded.
    ///
    /// The configured floor measures operational evidence coverage (item count and
    /// distinct source trajectories) only. A passing floor does not prove the context
    /// answers the question; the downstream caller should treat a failed floor as a
    /// reason to abstain, ask for clarification, or perform a separately authorized
    /// retrieval step, and must validate that policy on held-out tasks.
    ///
    /// Candidate decisions are reported in input order, including duplicates. Unknown,
    /// invalid, inactive, cross-scope, not-yet-valid, expired and over-limit candidates
    /// are never emitted. An eligible payload whose bytes fail verification is an
    /// integrity error and aborts the call; it is never mislabeled as an empty result.
    pub fn resolve_with_audit(
        &self,
        hdc_hits: &[u64],
        scope_id: &str,
        now_ms: u64,
        payload_root: &Path,
        max_items: usize,
        floor_policy: EvidenceFloorPolicy,
    ) -> Result<AuditedMemoryResolution, String> {
        if scope_id.trim().is_empty() {
            return Err("scope_id must be non-empty".into());
        }
        if floor_policy.minimum_items == 0 {
            return Err("minimum_items must be positive".into());
        }
        if floor_policy.minimum_distinct_trajectories == 0 {
            return Err("minimum_distinct_trajectories must be positive".into());
        }

        let mut seen = BTreeSet::new();
        let mut items = Vec::new();
        let mut candidate_audit = Vec::with_capacity(hdc_hits.len());
        for hdc_id in hdc_hits {
            let disposition = if !seen.insert(*hdc_id) {
                EvidenceCandidateDisposition::DuplicateCandidate
            } else if let Some(record) = self.records.get(hdc_id) {
                if record.lifecycle != EvidenceLifecycle::Active {
                    EvidenceCandidateDisposition::InactiveLifecycle
                } else if record.validate().is_err() {
                    EvidenceCandidateDisposition::InvalidRecord
                } else if record.scope_id != scope_id {
                    EvidenceCandidateDisposition::ScopeMismatch
                } else if record.valid_from_unix_ms.is_some_and(|start| now_ms < start) {
                    EvidenceCandidateDisposition::NotYetValid
                } else if record.valid_until_unix_ms.is_some_and(|end| now_ms >= end) {
                    EvidenceCandidateDisposition::Expired
                } else if items.len() >= max_items {
                    EvidenceCandidateDisposition::MetadataEligibleBeyondOutputLimit
                } else {
                    // Any payload integrity/path failure must abort rather than
                    // silently lowering the evidence floor.
                    let item = record.context_item(payload_root)?;
                    items.push(ResolvedMemoryContextItem {
                        hdc_id: *hdc_id,
                        evidence_id: record.evidence_id.clone(),
                        source: record.source.clone(),
                        item,
                    });
                    EvidenceCandidateDisposition::Emitted
                }
            } else {
                EvidenceCandidateDisposition::UnknownCandidate
            };
            candidate_audit.push(EvidenceCandidateAudit {
                hdc_id: *hdc_id,
                disposition,
            });
        }

        let distinct_trajectories = items.iter()
            .map(|item| item.source.trajectory_id.as_str())
            .collect::<BTreeSet<_>>()
            .len();
        let meets_floor = items.len() >= floor_policy.minimum_items
            && distinct_trajectories >= floor_policy.minimum_distinct_trajectories;
        let floor = EvidenceFloorAudit {
            required_items: floor_policy.minimum_items,
            required_distinct_trajectories: floor_policy.minimum_distinct_trajectories,
            emitted_items: items.len(),
            distinct_trajectories,
            meets_floor,
        };

        Ok(AuditedMemoryResolution {
            items,
            candidate_audit,
            floor,
        })
    }

    /// Select a query-specific evidence set using caller-supplied role annotations.
    ///
    /// This is a deterministic set-cover heuristic, not a semantic entailment engine.
    /// required_roles is the query planner's explicit information contract and
    /// candidate_roles must come from a separately evaluated, source-grounded
    /// reification stage. Both are inputs, not truths established by retrieval.
    ///
    /// At each step the selector chooses the eligible text candidate with the most
    /// newly-covered required roles per declared token cost. Ties prefer higher
    /// retrieval score, then lower HDC ID. Retrieval score is a tie-break only; it is
    /// not treated as calibrated evidence strength. Selection stops when all roles
    /// are covered, the item limit is reached, or no fitting candidate covers a
    /// remaining role. The result lists uncovered roles explicitly.
    ///
    /// Every unique ranked candidate must have a known evidence record, a positive
    /// declared cost, and an explicit role-set entry (possibly empty). This prevents
    /// missing annotations from silently behaving like semantic irrelevance. Eligible
    /// text digests are verified before they can participate in the selection.
    ///
    /// The final prompt/context serializer must re-count the actual serialized token
    /// total and fail closed if it exceeds token_budget; this method enforces only
    /// the declared additive-cost budget.
    pub(crate) fn select_query_evidence_set(
        &self,
        ranked_candidates: &[(u64, f32)],
        candidate_token_costs: &BTreeMap<u64, usize>,
        candidate_roles: &BTreeMap<u64, BTreeSet<String>>,
        required_roles: &BTreeSet<String>,
        scope_id: &str,
        now_ms: u64,
        token_budget: usize,
        max_candidates: usize,
    ) -> Result<QueryEvidenceSetSelection, String> {
        if scope_id.trim().is_empty() {
            return Err("scope_id must be non-empty".into());
        }
        if required_roles.is_empty() {
            return Err("required_roles must not be empty".into());
        }
        for role in required_roles {
            if role.trim().is_empty() || role.trim() != role {
                return Err("required role labels must be non-empty and trimmed".into());
            }
        }

        let mut ordered = ranked_candidates.to_vec();
        if ordered.iter().any(|(_, score)| !score.is_finite() || *score < 0.0) {
            return Err("candidate scores must be finite and non-negative".into());
        }
        ordered.sort_unstable_by(|left, right| {
            right.1.total_cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });

        // Validate each unique candidate's metadata before eligibility/budget checks.
        // Otherwise a malformed low-ranked item could be silently hidden by truncation.
        let mut unique = Vec::new();
        let mut seen = BTreeSet::new();
        for (hdc_id, score) in ordered {
            if !seen.insert(hdc_id) {
                continue;
            }
            if !self.records.contains_key(&hdc_id) {
                return Err(format!("ranked evidence candidate references unknown HDC ID {hdc_id}"));
            }
            let cost = candidate_token_costs.get(&hdc_id)
                .ok_or_else(|| format!("missing token cost for HDC ID {hdc_id}"))?;
            if *cost == 0 {
                return Err(format!("token cost must be positive for HDC ID {hdc_id}"));
            }
            let roles = candidate_roles.get(&hdc_id)
                .ok_or_else(|| format!("missing explicit evidence-role annotation for HDC ID {hdc_id}"))?;
            for role in roles {
                if role.trim().is_empty() || role.trim() != role {
                    return Err(format!("evidence role labels must be non-empty and trimmed for HDC ID {hdc_id}"));
                }
            }
            unique.push((hdc_id, score));
        }

        let role_keys: BTreeSet<u64> = candidate_roles.keys().copied().collect();
        let candidate_keys: BTreeSet<u64> = unique.iter().map(|(id, _)| *id).collect();
        if role_keys != candidate_keys {
            return Err("candidate role annotations must exactly match the unique ranked candidate IDs".into());
        }

        let mut eligible = Vec::new();
        for (hdc_id, score) in unique {
            let record = self.records.get(&hdc_id)
                .ok_or_else(|| format!("ranked evidence candidate disappeared: {hdc_id}"))?;
            if !record.is_eligible_for(scope_id, now_ms) {
                continue;
            }
            let EvidencePayload::Text { value } = &record.payload else {
                continue;
            };
            if sha256_hex(value.as_bytes()) != record.content_sha256 {
                return Err(format!("text content digest mismatch for eligible evidence HDC ID {hdc_id}"));
            }
            eligible.push((hdc_id, score));
        }

        let mut remaining_roles = required_roles.clone();
        let mut members = Vec::new();
        let mut used_ids = BTreeSet::new();
        let mut used_cost = 0usize;

        while !remaining_roles.is_empty() && members.len() < max_candidates {
            let remaining_budget = token_budget.saturating_sub(used_cost);
            let mut best: Option<(u64, f32, usize, BTreeSet<String>)> = None;

            for (hdc_id, score) in &eligible {
                if used_ids.contains(hdc_id) {
                    continue;
                }
                let cost = *candidate_token_costs.get(hdc_id)
                    .ok_or_else(|| format!("missing token cost for HDC ID {hdc_id}"))?;
                if cost > remaining_budget {
                    continue;
                }
                let roles = candidate_roles.get(hdc_id)
                    .ok_or_else(|| format!("missing evidence-role annotation for HDC ID {hdc_id}"))?;
                let newly_covered: BTreeSet<String> = roles
                    .intersection(&remaining_roles)
                    .cloned()
                    .collect();
                if newly_covered.is_empty() {
                    continue;
                }

                let replace = match &best {
                    None => true,
                    Some((best_id, best_score, best_cost, best_roles)) => {
                        let left = (newly_covered.len() as u128) * (*best_cost as u128);
                        let right = (best_roles.len() as u128) * (cost as u128);
                        left > right
                            || (left == right
                                && (score.total_cmp(best_score).is_gt()
                                    || (score.total_cmp(best_score).is_eq() && *hdc_id < *best_id)))
                    }
                };
                if replace {
                    best = Some((*hdc_id, *score, cost, newly_covered));
                }
            }

            let Some((hdc_id, score, cost, newly_covered)) = best else {
                break;
            };
            used_ids.insert(hdc_id);
            used_cost = used_cost.checked_add(cost)
                .ok_or_else(|| "selected evidence token cost overflow".to_string())?;
            for role in &newly_covered {
                remaining_roles.remove(role);
            }
            members.push(QueryEvidenceSetMember {
                hdc_id,
                retrieval_score: score,
                declared_token_cost: cost,
                newly_covered_roles: newly_covered.into_iter().collect(),
            });
        }

        let covered_roles: BTreeSet<String> = required_roles
            .difference(&remaining_roles)
            .cloned()
            .collect();
        Ok(QueryEvidenceSetSelection {
            required_roles: required_roles.iter().cloned().collect(),
            covered_roles: covered_roles.iter().cloned().collect(),
            missing_roles: remaining_roles.iter().cloned().collect(),
            members,
            declared_token_cost: used_cost,
            token_budget,
            complete: remaining_roles.is_empty(),
        })
    }

    /// ABI projection for consumers that require only official `type/value`
    /// context items. Use `resolve_with_provenance` when an audit trail is needed.
    pub fn resolve(
        &self,
        hdc_hits: &[u64],
        scope_id: &str,
        now_ms: u64,
        payload_root: &Path,
        max_items: usize,
    ) -> Result<Vec<MemoryContextItem>, String> {
        Ok(self.resolve_with_provenance(hdc_hits, scope_id, now_ms, payload_root, max_items)?
            .into_iter()
            .map(|resolved| resolved.item)
            .collect())
    }
}

/// Lowercase alphanumeric tokenization, deterministic and dependency-free.
/// No stemming, learned embeddings, external corpus statistics, or query labels
/// are used; keep this arm labeled as lexical retrieval in benchmark reports.
fn lexical_tokens(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in input.chars().flat_map(char::to_lowercase) {
        if character.is_alphanumeric() {
            current.push(character);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn resolve_image_path(payload_root: &Path, path: &str) -> Result<PathBuf, String> {
    let root = fs::canonicalize(payload_root)
        .map_err(|error| format!("invalid payload root: {error}"))?;
    let supplied = Path::new(path);
    if supplied.components().any(|part| matches!(part, Component::ParentDir)) {
        return Err("image path must not contain parent-directory traversal".into());
    }
    let candidate = if supplied.is_absolute() { supplied.to_path_buf() } else { root.join(supplied) };
    let resolved = fs::canonicalize(&candidate)
        .map_err(|error| format!("image payload does not exist: {error}"))?;
    if !resolved.starts_with(&root) {
        return Err("image payload resolves outside payload_root".into());
    }
    if !resolved.is_file() {
        return Err("image payload must resolve to a regular file".into());
    }
    Ok(resolved)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn hash(bytes: &[u8]) -> String {
        sha256_hex(bytes)
    }

    fn text_record(id: u64, evidence_id: &str, scope: &str, text: &str) -> EvidenceRecord {
        EvidenceRecord {
            schema_version: EVIDENCE_SCHEMA_VERSION.to_string(),
            evidence_id: evidence_id.to_string(),
            hdc_id: id,
            scope_id: scope.to_string(),
            source: EvidenceSource {
                trajectory_id: "trajectory-001".to_string(),
                step_id: Some("step-4".to_string()),
                step_index: Some(4),
                source_sha256: hash(b"canonical trajectory bytes"),
            },
            payload: EvidencePayload::Text { value: text.to_string() },
            content_sha256: hash(text.as_bytes()),
            lifecycle: EvidenceLifecycle::Active,
            occurred_at_unix_ms: Some(100),
            valid_from_unix_ms: None,
            valid_until_unix_ms: None,
            superseded_by: None,
        }
    }

    fn text_record_for_trajectory(
        id: u64,
        evidence_id: &str,
        scope: &str,
        trajectory_id: &str,
        text: &str,
    ) -> (EvidenceRecord, Vec<u8>) {
        let source_bytes = format!("canonical trajectory bytes for {trajectory_id}, state {id}").into_bytes();
        let mut record = text_record(id, evidence_id, scope, text);
        record.source.trajectory_id = trajectory_id.to_string();
        record.source.step_id = Some(format!("step-{id}"));
        record.source.step_index = Some(id);
        record.source.source_sha256 = hash(&source_bytes);
        (record, source_bytes)
    }


    #[test]
    fn trajectory_diversification_caps_sibling_states_and_preserves_score_order() {
        let mut evidence = EvidenceIndex::new();
        for (id, evidence_id, trajectory_id, source) in [
            (1, "state-a1", "trajectory-a", "goal repeated shared state"),
            (2, "state-a2", "trajectory-a", "goal repeated second state"),
            (3, "state-a3", "trajectory-a", "goal repeated third state"),
            (4, "state-b1", "trajectory-b", "goal separate environment"),
            (5, "state-c1", "trajectory-c", "goal another environment"),
        ] {
            let (record, source_bytes) =
                text_record_for_trajectory(id, evidence_id, "web", trajectory_id, source);
            evidence.insert(record, &source_bytes).unwrap();
        }

        // Deliberately unsorted input and a repeated ID must still produce a
        // deterministic score/ID order and never duplicate a context candidate.
        let ranked = [(5, 0.70), (2, 0.95), (1, 0.99), (1, 0.99), (4, 0.80), (3, 0.90)];
        let selected = evidence
            .diversify_text_candidates_by_trajectory(&ranked, "web", 150, 3, 1)
            .unwrap();
        assert_eq!(selected, vec![(1, 0.99), (4, 0.80), (5, 0.70)]);

        let two_per_trajectory = evidence
            .diversify_text_candidates_by_trajectory(&ranked, "web", 150, 4, 2)
            .unwrap();
        assert_eq!(
            two_per_trajectory,
            vec![(1, 0.99), (2, 0.95), (4, 0.80), (5, 0.70)]
        );
    }

    #[test]
    fn trajectory_diversification_rechecks_scope_lifecycle_and_validity() {
        let mut evidence = EvidenceIndex::new();

        let (active, active_source) =
            text_record_for_trajectory(1, "active", "web", "trajectory-a", "current active text");
        evidence.insert(active, &active_source).unwrap();

        let (private, private_source) = text_record_for_trajectory(
            2, "private", "enterprise", "trajectory-b", "cross scope text",
        );
        evidence.insert(private, &private_source).unwrap();

        let (mut expired, expired_source) = text_record_for_trajectory(
            3, "expired", "web", "trajectory-c", "expired text",
        );
        expired.valid_until_unix_ms = Some(150);
        evidence.insert(expired, &expired_source).unwrap();

        let (revoked, revoked_source) = text_record_for_trajectory(
            4, "revoked", "web", "trajectory-d", "revoked text",
        );
        evidence.insert(revoked, &revoked_source).unwrap();
        evidence.revoke(4).unwrap();

        let selected = evidence
            .diversify_text_candidates_by_trajectory(
                &[(1, 1.0), (2, 0.99), (3, 0.98), (4, 0.97)],
                "web",
                150,
                10,
                1,
            )
            .unwrap();
        assert_eq!(selected, vec![(1, 1.0)]);
        assert!(evidence
            .diversify_text_candidates_by_trajectory(&[(99, 0.5)], "web", 150, 1, 1)
            .is_err());
        assert!(evidence
            .diversify_text_candidates_by_trajectory(&[(1, f32::NAN)], "web", 150, 1, 1)
            .is_err());
        assert!(evidence
            .diversify_text_candidates_by_trajectory(&[(1, 1.0)], "web", 150, 1, 0)
            .is_err());
    }

    #[test]
    fn lexical_index_round_trip_and_indexed_bm25_match_reference_ranking() {
        let mut evidence = EvidenceIndex::new();
        evidence.insert(text_record(7, "evidence-relevant", "web", "cart checkout button opens shopping cart"), b"canonical trajectory bytes").unwrap();
        evidence.insert(text_record(8, "evidence-unrelated", "web", "mountain trails and hiking boots"), b"canonical trajectory bytes").unwrap();
        let mut expired = text_record(9, "evidence-expired", "web", "cart checkout button opens shopping cart");
        expired.valid_until_unix_ms = Some(150);
        evidence.insert(expired, b"canonical trajectory bytes").unwrap();
        let mut other_scope = text_record(10, "evidence-other-scope", "enterprise", "cart checkout button opens shopping cart");
        evidence.insert(other_scope.clone(), b"canonical trajectory bytes").unwrap();
        other_scope.hdc_id = 11;
        other_scope.evidence_id = "evidence-revoked".into();
        other_scope.scope_id = "web".into();
        evidence.insert(other_scope, b"canonical trajectory bytes").unwrap();
        evidence.revoke(11).unwrap();

        let index = LexicalIndex::build(&evidence, 4).unwrap();
        let bytes = index.to_json_bytes().unwrap();
        let restored = LexicalIndex::from_json_bytes(&bytes).unwrap();
        restored.validate_for_evidence(&evidence, 4).unwrap();
        assert_eq!(restored.to_json_bytes().unwrap(), bytes);

        let query = "Where is the cart checkout button?";
        let reference = evidence.rank_text(query, "web", 150, 20).unwrap();
        let indexed = evidence.rank_text_indexed(&restored, 4, query, "web", 150, 20).unwrap();
        assert_eq!(indexed, reference, "indexed BM25 must match the reference ranker exactly");
        assert!(evidence.rank_text_indexed(&restored, 5, query, "web", 150, 20).is_err());
    }

    #[test]
    fn lexical_index_rejects_wrong_generation_and_document_set() {
        let mut evidence = EvidenceIndex::new();
        evidence.insert(text_record(7, "evidence-text", "web", "lexical index generation test"), b"canonical trajectory bytes").unwrap();
        let mut index = LexicalIndex::build(&evidence, 3).unwrap();
        assert!(index.validate_for_evidence(&evidence, 4).is_err());
        index.documents.remove(&7);
        assert!(index.validate_for_evidence(&evidence, 3).is_err());
    }

    #[test]
    fn lexical_bm25_ranks_relevant_text_and_filters_scope_time_and_lifecycle() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-relevant", "web", "cart checkout button opens the shopping cart"), b"canonical trajectory bytes").unwrap();
        index.insert(text_record(8, "evidence-unrelated", "web", "mountain trails and hiking boots"), b"canonical trajectory bytes").unwrap();

        let mut private = text_record(9, "evidence-private", "enterprise", "cart checkout button opens shopping cart");
        index.insert(private.clone(), b"canonical trajectory bytes").unwrap();

        private.hdc_id = 10;
        private.evidence_id = "evidence-expired".into();
        private.scope_id = "web".into();
        private.valid_until_unix_ms = Some(150);
        private.content_sha256 = hash(b"cart checkout button opens shopping cart");
        index.insert(private, b"canonical trajectory bytes").unwrap();

        let ranked = index.rank_text("Where is the cart checkout button?", "web", 150, 10).unwrap();
        assert_eq!(ranked.first().map(|item| item.0), Some(7));
        assert!(ranked.iter().all(|(id, _)| *id != 9 && *id != 10));
        assert!(ranked.iter().all(|(_, score)| score.is_finite() && *score > 0.0));
        assert!(index.rank_text("!!!", "web", 150, 10).unwrap().is_empty());
    }

    #[test]
    fn token_budget_packing_skips_oversize_items_and_preserves_rank_order() {
        let mut evidence = EvidenceIndex::new();
        for (id, evidence_id, trajectory_id, value) in [
            (1, "long-high-score", "trajectory-a", "longest context"),
            (2, "medium-score", "trajectory-b", "medium context"),
            (3, "small-score", "trajectory-c", "small"),
        ] {
            let (record, source_bytes) = text_record_for_trajectory(
                id, evidence_id, "web", trajectory_id, value,
            );
            evidence.insert(record, &source_bytes).unwrap();
        }

        let ranked = [(3, 0.80), (1, 0.99), (2, 0.90), (1, 0.99)];
        let costs = BTreeMap::from([(1, 6usize), (2, 4usize), (3, 1usize)]);
        let selected = evidence.pack_text_candidates_under_token_budget(
            &ranked, &costs, "web", 150, 5, 3, None,
        ).unwrap();

        // The top-ranked item cannot fit. The packer skips it, keeps the next
        // highest-ranked item that fits, then fills the final token with item 3.
        assert_eq!(selected, vec![(2, 0.90), (3, 0.80)]);
        assert!(selected.iter().map(|(id, _)| costs[id]).sum::<usize>() <= 5);
    }

    #[test]
    fn token_budget_packing_rechecks_scope_lifecycle_validity_and_trajectory_cap() {
        let mut evidence = EvidenceIndex::new();
        for (id, evidence_id, scope, trajectory_id, value) in [
            (1, "active-a1", "web", "trajectory-a", "active first"),
            (2, "active-a2", "web", "trajectory-a", "active sibling"),
            (3, "private", "enterprise", "trajectory-b", "private context"),
            (4, "expired", "web", "trajectory-c", "expired context"),
            (5, "revoked", "web", "trajectory-d", "revoked context"),
            (6, "active-e", "web", "trajectory-e", "another current context"),
        ] {
            let (mut record, source_bytes) = text_record_for_trajectory(
                id, evidence_id, scope, trajectory_id, value,
            );
            if id == 4 {
                record.valid_until_unix_ms = Some(150);
            }
            evidence.insert(record, &source_bytes).unwrap();
        }
        evidence.revoke(5).unwrap();

        let ranked = [(1, 0.99), (2, 0.98), (3, 0.97), (4, 0.96), (5, 0.95), (6, 0.80)];
        let costs = BTreeMap::from([
            (1, 2usize), (2, 2usize), (3, 2usize),
            (4, 2usize), (5, 2usize), (6, 2usize),
        ]);
        let selected = evidence.pack_text_candidates_under_token_budget(
            &ranked, &costs, "web", 150, 4, 5, Some(1),
        ).unwrap();

        assert_eq!(selected, vec![(1, 0.99), (6, 0.80)]);
        assert!(selected.iter().map(|(id, _)| costs[id]).sum::<usize>() <= 4);
    }

    #[test]
    fn token_budget_packing_fails_closed_on_malformed_scores_ids_and_costs() {
        let mut evidence = EvidenceIndex::new();
        evidence.insert(
            text_record(7, "evidence-seven", "web", "candidate evidence"),
            b"canonical trajectory bytes",
        ).unwrap();
        let costs = BTreeMap::from([(7, 1usize)]);

        assert!(evidence.pack_text_candidates_under_token_budget(
            &[(7, f32::NAN)], &costs, "web", 150, 5, 2, None,
        ).is_err());
        assert!(evidence.pack_text_candidates_under_token_budget(
            &[(7, -0.1)], &costs, "web", 150, 5, 2, None,
        ).is_err());
        assert!(evidence.pack_text_candidates_under_token_budget(
            &[(404, 0.5)], &BTreeMap::from([(404, 1usize)]), "web", 150, 5, 2, None,
        ).is_err());
        assert!(evidence.pack_text_candidates_under_token_budget(
            &[(7, 0.5)], &BTreeMap::new(), "web", 150, 5, 2, None,
        ).is_err());
        assert!(evidence.pack_text_candidates_under_token_budget(
            &[(7, 0.5)], &BTreeMap::from([(7, 0usize)]), "web", 150, 5, 2, None,
        ).is_err());
        assert!(evidence.pack_text_candidates_under_token_budget(
            &[(7, 0.5)], &costs, "", 150, 5, 2, None,
        ).is_err());
        assert!(evidence.pack_text_candidates_under_token_budget(
            &[(7, 0.5)], &costs, "web", 150, 5, 2, Some(0),
        ).is_err());
    }

    #[test]
    fn adjacent_state_expansion_keeps_seeds_and_adds_contiguous_neighbors() {
        let mut evidence = EvidenceIndex::new();
        for (id, evidence_id, trajectory_id, step, scope, text) in [
            (1, "a-state-4", "trajectory-a", 4, "web", "previous state"),
            (2, "a-state-5", "trajectory-a", 5, "web", "retrieved target state"),
            (3, "a-state-6", "trajectory-a", 6, "web", "following state"),
            (4, "b-state-4", "trajectory-b", 4, "web", "other trajectory"),
            (5, "private-state-6", "trajectory-a", 6, "enterprise", "private neighbor"),
            (6, "a-state-8", "trajectory-a", 8, "web", "not contiguous"),
        ] {
            let (mut record, source_bytes) = text_record_for_trajectory(
                id, evidence_id, scope, trajectory_id, text,
            );
            record.source.step_index = Some(step);
            evidence.insert(record, &source_bytes).unwrap();
        }

        let expanded = evidence.expand_text_candidates_with_adjacent_states(
            &[(2, 0.99)], "web", 150, 3, 2,
        ).unwrap();
        assert_eq!(expanded, vec![
            AdjacentStateCandidate { hdc_id: 2, anchor_hdc_id: None },
            AdjacentStateCandidate { hdc_id: 1, anchor_hdc_id: Some(2) },
            AdjacentStateCandidate { hdc_id: 3, anchor_hdc_id: Some(2) },
        ]);
    }

    #[test]
    fn adjacent_state_expansion_filters_lifecycle_and_preserves_seed_priority() {
        let mut evidence = EvidenceIndex::new();
        for (id, evidence_id, step, text) in [
            (1, "previous", 4, "previous active state"),
            (2, "seed", 5, "retrieval seed"),
            (3, "next-revoked", 6, "revoked next state"),
            (4, "other-seed", 9, "second ranked seed"),
            (5, "other-next", 10, "neighbor of second seed"),
        ] {
            let (mut record, source_bytes) = text_record_for_trajectory(
                id, evidence_id, "web", "trajectory-a", text,
            );
            record.source.step_index = Some(step);
            evidence.insert(record, &source_bytes).unwrap();
        }
        evidence.revoke(3).unwrap();

        let expanded = evidence.expand_text_candidates_with_adjacent_states(
            &[(2, 0.99), (4, 0.90)], "web", 150, 3, 1,
        ).unwrap();
        assert_eq!(expanded, vec![
            AdjacentStateCandidate { hdc_id: 2, anchor_hdc_id: None },
            AdjacentStateCandidate { hdc_id: 4, anchor_hdc_id: None },
            AdjacentStateCandidate { hdc_id: 1, anchor_hdc_id: Some(2) },
        ]);
    }

    #[test]
    fn adjacent_state_expansion_fails_closed_on_ambiguous_active_neighbors() {
        let mut evidence = EvidenceIndex::new();
        for (id, evidence_id, step, text) in [
            (1, "seed", 5, "retrieval seed"),
            (2, "neighbor-a", 6, "first possible neighbor"),
            (3, "neighbor-b", 6, "ambiguous possible neighbor"),
        ] {
            let (mut record, source_bytes) = text_record_for_trajectory(
                id, evidence_id, "web", "trajectory-a", text,
            );
            record.source.step_index = Some(step);
            evidence.insert(record, &source_bytes).unwrap();
        }
        assert!(evidence.expand_text_candidates_with_adjacent_states(
            &[(1, 0.99)], "web", 150, 3, 2,
        ).is_err());
    }

    #[test]
    fn adjacent_state_index_rebuilds_after_snapshot_restore_and_filters_tombstones() {
        let mut evidence = EvidenceIndex::new();
        for (id, evidence_id, step, text) in [
            (1, "previous", 4, "previous active state"),
            (2, "seed", 5, "retrieval seed"),
            (3, "next-deleted", 6, "deleted next state"),
        ] {
            let (mut record, source_bytes) = text_record_for_trajectory(
                id, evidence_id, "web", "trajectory-a", text,
            );
            record.source.step_index = Some(step);
            evidence.insert(record, &source_bytes).unwrap();
        }
        evidence.mark_deleted(3).unwrap();

        let snapshot = evidence.snapshot_bytes(7).unwrap();
        let restored = EvidenceIndex::restore_snapshot(&snapshot).unwrap();
        let expanded = restored.expand_text_candidates_with_adjacent_states(
            &[(2, 0.99)], "web", 150, 3, 2,
        ).unwrap();

        assert_eq!(expanded, vec![
            AdjacentStateCandidate { hdc_id: 2, anchor_hdc_id: None },
            AdjacentStateCandidate { hdc_id: 1, anchor_hdc_id: Some(2) },
        ]);
        assert!(expanded.iter().all(|candidate| candidate.hdc_id != 3));
    }

    #[test]
    fn adjacent_state_expansion_rejects_unknown_seed_and_malformed_score() {
        let evidence = EvidenceIndex::new();
        assert!(evidence.expand_text_candidates_with_adjacent_states(
            &[(404, 0.99)], "web", 150, 3, 2,
        ).is_err());
        assert!(evidence.expand_text_candidates_with_adjacent_states(
            &[(404, f32::INFINITY)], "web", 150, 3, 2,
        ).is_err());
        assert!(evidence.expand_text_candidates_with_adjacent_states(
            &[], "", 150, 3, 2,
        ).is_err());
    }

    #[test]
    fn audited_resolution_explains_rejections_and_fails_the_coverage_floor() {
        let mut evidence = EvidenceIndex::new();
        let (active, active_source) = text_record_for_trajectory(
            1, "active", "web", "trajectory-a", "current valid context",
        );
        evidence.insert(active, &active_source).unwrap();

        let (private, private_source) = text_record_for_trajectory(
            2, "private", "enterprise", "trajectory-private", "private context",
        );
        evidence.insert(private, &private_source).unwrap();

        let (mut future, future_source) = text_record_for_trajectory(
            3, "future", "web", "trajectory-future", "future context",
        );
        future.valid_from_unix_ms = Some(200);
        evidence.insert(future, &future_source).unwrap();

        let (mut expired, expired_source) = text_record_for_trajectory(
            4, "expired", "web", "trajectory-expired", "expired context",
        );
        expired.valid_until_unix_ms = Some(150);
        evidence.insert(expired, &expired_source).unwrap();

        let (revoked, revoked_source) = text_record_for_trajectory(
            5, "revoked", "web", "trajectory-revoked", "revoked context",
        );
        evidence.insert(revoked, &revoked_source).unwrap();
        evidence.revoke(5).unwrap();

        let result = evidence.resolve_with_audit(
            &[1, 1, 2, 3, 4, 5, 404],
            "web", 150, Path::new("."), 10,
            EvidenceFloorPolicy { minimum_items: 2, minimum_distinct_trajectories: 2 },
        ).unwrap();

        assert_eq!(result.items.len(), 1);
        assert_eq!(result.candidate_audit.iter().map(|decision| decision.disposition).collect::<Vec<_>>(), vec![
            EvidenceCandidateDisposition::Emitted,
            EvidenceCandidateDisposition::DuplicateCandidate,
            EvidenceCandidateDisposition::ScopeMismatch,
            EvidenceCandidateDisposition::NotYetValid,
            EvidenceCandidateDisposition::Expired,
            EvidenceCandidateDisposition::InactiveLifecycle,
            EvidenceCandidateDisposition::UnknownCandidate,
        ]);
        assert_eq!(result.floor.emitted_items, 1);
        assert_eq!(result.floor.distinct_trajectories, 1);
        assert!(!result.floor.meets_floor);
    }

    #[test]
    fn audited_resolution_requires_distinct_trajectory_coverage_not_just_item_count() {
        let mut evidence = EvidenceIndex::new();
        for (id, evidence_id, trajectory_id, text) in [
            (1, "state-a1", "trajectory-a", "first state"),
            (2, "state-a2", "trajectory-a", "second state"),
            (3, "state-b1", "trajectory-b", "independent state"),
        ] {
            let (record, source_bytes) = text_record_for_trajectory(
                id, evidence_id, "web", trajectory_id, text,
            );
            evidence.insert(record, &source_bytes).unwrap();
        }
        let policy = EvidenceFloorPolicy {
            minimum_items: 2,
            minimum_distinct_trajectories: 2,
        };

        let one_trajectory = evidence.resolve_with_audit(
            &[1, 2], "web", 150, Path::new("."), 2, policy,
        ).unwrap();
        assert_eq!(one_trajectory.items.len(), 2);
        assert_eq!(one_trajectory.floor.distinct_trajectories, 1);
        assert!(!one_trajectory.floor.meets_floor);

        let diverse = evidence.resolve_with_audit(
            &[1, 3], "web", 150, Path::new("."), 2, policy,
        ).unwrap();
        assert_eq!(diverse.items.len(), 2);
        assert_eq!(diverse.floor.distinct_trajectories, 2);
        assert!(diverse.floor.meets_floor);
    }

    #[test]
    fn audited_resolution_marks_eligible_candidates_beyond_output_limit() {
        let mut evidence = EvidenceIndex::new();
        for (id, evidence_id, trajectory_id) in [
            (1, "first", "trajectory-a"),
            (2, "second", "trajectory-b"),
        ] {
            let (record, source_bytes) = text_record_for_trajectory(
                id, evidence_id, "web", trajectory_id, "eligible context",
            );
            evidence.insert(record, &source_bytes).unwrap();
        }
        let result = evidence.resolve_with_audit(
            &[1, 2], "web", 150, Path::new("."), 1,
            EvidenceFloorPolicy { minimum_items: 1, minimum_distinct_trajectories: 1 },
        ).unwrap();
        assert_eq!(result.items.len(), 1);
        assert_eq!(result.candidate_audit[1].disposition, EvidenceCandidateDisposition::MetadataEligibleBeyondOutputLimit);
        assert!(result.floor.meets_floor);
    }

    #[test]
    fn audited_resolution_fails_closed_on_corrupt_payload_and_invalid_floor() {
        let mut evidence = EvidenceIndex::new();
        evidence.insert(
            text_record(7, "evidence-corrupt", "web", "verified text"),
            b"canonical trajectory bytes",
        ).unwrap();
        evidence.records.get_mut(&7).unwrap().content_sha256 = hash(b"tampered text");

        let policy = EvidenceFloorPolicy { minimum_items: 1, minimum_distinct_trajectories: 1 };
        assert!(evidence.resolve_with_audit(
            &[7], "web", 150, Path::new("."), 1, policy,
        ).is_err());
        assert!(evidence.resolve_with_audit(
            &[7], "web", 150, Path::new("."), 1,
            EvidenceFloorPolicy { minimum_items: 0, minimum_distinct_trajectories: 1 },
        ).is_err());
        assert!(evidence.resolve_with_audit(
            &[7], "web", 150, Path::new("."), 1,
            EvidenceFloorPolicy { minimum_items: 1, minimum_distinct_trajectories: 0 },
        ).is_err());
    }

    #[test]
    fn query_evidence_set_covers_complementary_roles_within_budget() {
        let mut evidence = EvidenceIndex::new();
        for (id, evidence_id, trajectory_id, text) in [
            (1, "ranked-high", "trajectory-a", "initially relevant observation"),
            (2, "multi-role", "trajectory-b", "procedure and outcome evidence"),
            (3, "procedure-only", "trajectory-c", "procedure context"),
            (4, "outcome-only", "trajectory-d", "observed result"),
        ] {
            let (record, source_bytes) = text_record_for_trajectory(
                id, evidence_id, "web", trajectory_id, text,
            );
            evidence.insert(record, &source_bytes).unwrap();
        }

        let ranked = [(1, 0.99), (2, 0.60), (3, 0.80), (4, 0.70)];
        let costs = BTreeMap::from([(1, 2usize), (2, 3usize), (3, 2usize), (4, 2usize)]);
        let roles = BTreeMap::from([
            (1, BTreeSet::from(["context".to_string()])),
            (2, BTreeSet::from(["procedure".to_string(), "outcome".to_string()])),
            (3, BTreeSet::from(["procedure".to_string()])),
            (4, BTreeSet::from(["outcome".to_string()])),
        ]);
        let required = BTreeSet::from(["procedure".to_string(), "outcome".to_string()]);
        let selected = evidence.select_query_evidence_set(
            &ranked, &costs, &roles, &required, "web", 150, 3, 2,
        ).unwrap();

        assert!(selected.complete);
        assert_eq!(selected.members.iter().map(|member| member.hdc_id).collect::<Vec<_>>(), vec![2]);
        assert_eq!(selected.covered_roles, vec!["outcome".to_string(), "procedure".to_string()]);
        assert!(selected.missing_roles.is_empty());
        assert_eq!(selected.declared_token_cost, 3);
    }

    #[test]
    fn query_evidence_set_returns_uncovered_roles_instead_of_claiming_sufficiency() {
        let mut evidence = EvidenceIndex::new();
        let (record, source_bytes) = text_record_for_trajectory(
            1, "single-role", "web", "trajectory-a", "only the first fact is present",
        );
        evidence.insert(record, &source_bytes).unwrap();
        let ranked = [(1, 0.9)];
        let costs = BTreeMap::from([(1, 2usize)]);
        let roles = BTreeMap::from([(
            1, BTreeSet::from(["identity".to_string()]),
        )]);
        let required = BTreeSet::from(["identity".to_string(), "date".to_string()]);

        let selected = evidence.select_query_evidence_set(
            &ranked, &costs, &roles, &required, "web", 150, 8, 4,
        ).unwrap();
        assert!(!selected.complete);
        assert_eq!(selected.covered_roles, vec!["identity".to_string()]);
        assert_eq!(selected.missing_roles, vec!["date".to_string()]);
    }

    #[test]
    fn query_evidence_set_rejects_missing_roles_unknown_ids_and_malformed_costs() {
        let mut evidence = EvidenceIndex::new();
        evidence.insert(
            text_record(1, "one", "web", "candidate"),
            b"canonical trajectory bytes",
        ).unwrap();
        let ranked = [(1, 0.9)];
        let required = BTreeSet::from(["claim".to_string()]);

        assert!(evidence.select_query_evidence_set(
            &ranked, &BTreeMap::from([(1, 1usize)]), &BTreeMap::new(),
            &required, "web", 150, 4, 2,
        ).is_err());
        assert!(evidence.select_query_evidence_set(
            &[(404, 0.9)], &BTreeMap::from([(404, 1usize)]),
            &BTreeMap::from([(404, BTreeSet::from(["claim".to_string()]))]),
            &required, "web", 150, 4, 2,
        ).is_err());
        assert!(evidence.select_query_evidence_set(
            &ranked, &BTreeMap::from([(1, 0usize)]),
            &BTreeMap::from([(1, BTreeSet::from(["claim".to_string()]))]),
            &required, "web", 150, 4, 2,
        ).is_err());
        assert!(evidence.select_query_evidence_set(
            &ranked, &BTreeMap::from([(1, 1usize)]),
            &BTreeMap::from([(1, BTreeSet::from([" claim ".to_string()]))]),
            &required, "web", 150, 4, 2,
        ).is_err());
    }

    #[test]
    fn query_evidence_set_never_uses_ineligible_or_corrupt_evidence() {
        let mut evidence = EvidenceIndex::new();
        let (active, active_source) = text_record_for_trajectory(
            1, "active", "web", "trajectory-active", "current evidence",
        );
        evidence.insert(active, &active_source).unwrap();
        let (private, private_source) = text_record_for_trajectory(
            2, "private", "enterprise", "trajectory-private", "private evidence",
        );
        evidence.insert(private, &private_source).unwrap();
        let (mut expired, expired_source) = text_record_for_trajectory(
            3, "expired", "web", "trajectory-expired", "expired evidence",
        );
        expired.valid_until_unix_ms = Some(150);
        evidence.insert(expired, &expired_source).unwrap();

        let ranked = [(1, 0.9), (2, 0.99), (3, 0.95)];
        let costs = BTreeMap::from([(1, 1usize), (2, 1usize), (3, 1usize)]);
        let roles = BTreeMap::from([
            (1, BTreeSet::from(["claim".to_string()])),
            (2, BTreeSet::from(["claim".to_string()])),
            (3, BTreeSet::from(["claim".to_string()])),
        ]);
        let required = BTreeSet::from(["claim".to_string()]);
        let selected = evidence.select_query_evidence_set(
            &ranked, &costs, &roles, &required, "web", 150, 2, 2,
        ).unwrap();
        assert_eq!(selected.members.iter().map(|member| member.hdc_id).collect::<Vec<_>>(), vec![1]);

        evidence.records.get_mut(&1).unwrap().content_sha256 = hash(b"tampered");
        assert!(evidence.select_query_evidence_set(
            &ranked, &costs, &roles, &required, "web", 150, 2, 2,
        ).is_err());
    }

    #[test]
    fn lexical_bm25_fails_closed_on_corrupt_eligible_text() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-valid", "web", "canonical answer"), b"canonical trajectory bytes").unwrap();
        index.records.get_mut(&7).unwrap().content_sha256 = hash(b"wrong bytes");
        assert!(index.rank_text("canonical", "web", 150, 10).is_err());
    }

    #[test]
    fn lexical_bm25_ties_are_deterministic_by_hdc_id() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(9, "evidence-nine", "web", "identical lexical terms"), b"canonical trajectory bytes").unwrap();
        index.insert(text_record(3, "evidence-three", "web", "identical lexical terms"), b"canonical trajectory bytes").unwrap();
        let first = index.rank_text("identical lexical terms", "web", 150, 10).unwrap();
        let second = index.rank_text("identical lexical terms", "web", 150, 10).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.iter().map(|pair| pair.0).collect::<Vec<_>>(), vec![3, 9]);
    }

    #[test]
    fn snapshot_round_trip_preserves_tombstones_and_blocks_resurrection() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-deleted", "web", "old route"), b"canonical trajectory bytes").unwrap();
        index.insert(text_record(8, "evidence-active", "web", "current route"), b"canonical trajectory bytes").unwrap();
        index.mark_deleted(7).unwrap();

        let snapshot = index.snapshot_bytes(3).unwrap();
        let restored = EvidenceIndex::restore_snapshot(&snapshot).unwrap();
        let items = restored.resolve(&[7, 8, 7], "web", 150, Path::new("."), 10).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].value, "current route");
        assert!(restored.insert(
            text_record(7, "evidence-resurrection", "web", "resurrected route"),
            b"canonical trajectory bytes",
        ).is_err());
        assert_eq!(restored.snapshot_bytes(3).unwrap(), snapshot);
    }

    #[test]
    fn snapshot_rejects_record_tampering_and_noncanonical_envelopes() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "trusted route"), b"canonical trajectory bytes").unwrap();
        let snapshot = index.snapshot_bytes(3).unwrap();

        let mut value: serde_json::Value = serde_json::from_slice(&snapshot).unwrap();
        value["records"][0]["payload"]["value"] = serde_json::Value::String("tampered route".into());
        let tampered = serde_json::to_vec(&value).unwrap();
        assert!(EvidenceIndex::restore_snapshot(&tampered).is_err());

        value["schema_version"] = serde_json::Value::String("BROCA-EVIDENCE-INDEX-SNAPSHOT-V999".into());
        assert!(EvidenceIndex::restore_snapshot(&serde_json::to_vec(&value).unwrap()).is_err());

        // Recomputing the outer snapshot digest must not hide a bad payload digest.
        let mut crafted: EvidenceIndexSnapshot = serde_json::from_slice(&snapshot).unwrap();
        if let EvidencePayload::Text { value } = &mut crafted.records[0].payload {
            *value = "tampered route".into();
        }
        crafted.snapshot_sha256 = hash(&serde_json::to_vec(&(
            &crafted.schema_version,
            crafted.generation,
            &crafted.records,
        )).unwrap());
        assert!(EvidenceIndex::restore_snapshot(&serde_json::to_vec(&crafted).unwrap()).is_err());
    }

    #[test]
    fn snapshot_rebuild_rejects_duplicate_ids_even_with_recomputed_digest() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "route one"), b"canonical trajectory bytes").unwrap();
        let snapshot = index.snapshot_bytes(3).unwrap();
        let mut envelope: EvidenceIndexSnapshot = serde_json::from_slice(&snapshot).unwrap();
        envelope.records.push(envelope.records[0].clone());
        envelope.snapshot_sha256 = hash(&serde_json::to_vec(&(
            &envelope.schema_version,
            envelope.generation,
            &envelope.records,
        )).unwrap());
        let crafted = serde_json::to_vec(&envelope).unwrap();
        assert!(EvidenceIndex::restore_snapshot(&crafted).is_err());
    }

    #[test]
    fn legacy_v2_manifest_remains_verifiable_without_lexical_artifact() {
        let mut evidence = EvidenceIndex::new();
        evidence.insert(
            text_record(7, "evidence-legacy", "web", "legacy committed evidence"),
            b"canonical trajectory bytes",
        ).unwrap();
        let snapshot = evidence.snapshot_bytes(5).unwrap();
        let identity = HdcStoreIdentity {
            store_generation: 3,
            live_count: 1,
            checksum_version: 1,
            checksum_hex: "0123456789abcdef".into(),
        };
        let legacy = EvidenceCommitManifest::new_with_hdc_identity(
            5, b"canonical hdc bytes", &snapshot, identity,
        ).unwrap();
        let mut legacy_value: serde_json::Value =
            serde_json::from_slice(&legacy.to_json_bytes().unwrap()).unwrap();
        legacy_value.as_object_mut().unwrap().remove("lexical_index_sha256");
        let legacy_bytes = serde_json::to_vec(&legacy_value).unwrap();

        let recovered = EvidenceCommitManifest::from_json_bytes(&legacy_bytes).unwrap();
        recovered.verify(b"canonical hdc bytes", &snapshot).unwrap();
        recovered.verify_lexical_index(None).unwrap();
        assert!(recovered.verify_lexical_index(Some(b"unmanifested index")).is_err());
        assert_eq!(recovered.lexical_index_sha256(), None);

        let mut invalid_v3 = legacy_value;
        invalid_v3["schema_version"] =
            serde_json::Value::String(EVIDENCE_COMMIT_MANIFEST_SCHEMA_VERSION.into());
        assert!(EvidenceCommitManifest::from_json_bytes(&serde_json::to_vec(&invalid_v3).unwrap()).is_err(),
            "V3 typed manifest without a lexical digest must be rejected");
    }

    #[test]
    fn v3_commit_manifest_binds_exact_lexical_index_bytes() {
        let mut evidence = EvidenceIndex::new();
        evidence.insert(
            text_record(7, "evidence-manifest", "web", "manifest bound lexical document"),
            b"canonical trajectory bytes",
        ).unwrap();
        let snapshot = evidence.snapshot_bytes(5).unwrap();
        let lexical_index = LexicalIndex::build(&evidence, 5).unwrap();
        let lexical_bytes = lexical_index.to_json_bytes().unwrap();
        let identity = HdcStoreIdentity {
            store_generation: 3,
            live_count: 1,
            checksum_version: 1,
            checksum_hex: "0123456789abcdef".into(),
        };
        let manifest = EvidenceCommitManifest::new_with_hdc_identity_and_lexical_index(
            5, b"canonical hdc bytes", &snapshot, identity, &lexical_bytes,
        ).unwrap();
        manifest.verify(b"canonical hdc bytes", &snapshot).unwrap();
        manifest.verify_lexical_index(Some(&lexical_bytes)).unwrap();
        assert!(manifest.verify_lexical_index(Some(b"different lexical index")).is_err());
        assert!(manifest.verify_lexical_index(None).is_err());

        let encoded = manifest.to_json_bytes().unwrap();
        let recovered = EvidenceCommitManifest::from_json_bytes(&encoded).unwrap();
        assert_eq!(recovered.lexical_index_sha256(), Some(sha256_hex(&lexical_bytes).as_str()));
        recovered.verify_lexical_index(Some(&lexical_bytes)).unwrap();
    }

    #[test]
    fn lexical_index_rejects_unknown_schema_and_malformed_postings() {
        let mut evidence = EvidenceIndex::new();
        evidence.insert(
            text_record(7, "evidence-index-structure", "web", "alpha beta alpha"),
            b"canonical trajectory bytes",
        ).unwrap();
        let lexical = LexicalIndex::build(&evidence, 2).unwrap();
        let bytes = lexical.to_json_bytes().unwrap();

        let mut unknown: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        unknown["schema_version"] = serde_json::Value::String("BROCA-LEXICAL-INDEX-V999".into());
        assert!(LexicalIndex::from_json_bytes(&serde_json::to_vec(&unknown).unwrap()).is_err());

        let mut malformed: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        malformed["postings"]["alpha"][0]["term_frequency"] = serde_json::Value::from(0);
        assert!(LexicalIndex::from_json_bytes(&serde_json::to_vec(&malformed).unwrap()).is_err());
    }

    #[test]
    fn commit_manifest_binds_matching_vector_index_and_evidence_snapshot() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "current route"), b"canonical trajectory bytes").unwrap();
        index.mark_deleted(7).unwrap();
        let evidence_bytes = index.snapshot_bytes(3).unwrap();
        let hdc_bytes = b"canonical HDC index generation 3";

        let manifest = EvidenceCommitManifest::new(3, hdc_bytes, &evidence_bytes).unwrap();
        assert_eq!(manifest.generation(), 3);
        manifest.verify(hdc_bytes, &evidence_bytes).unwrap();

        let encoded = manifest.to_json_bytes().unwrap();
        let restored = EvidenceCommitManifest::from_json_bytes(&encoded).unwrap();
        restored.verify(hdc_bytes, &evidence_bytes).unwrap();
        assert!(restored.verify(b"stale HDC index bytes", &evidence_bytes).is_err());

        let mut other_index = EvidenceIndex::new();
        other_index.insert(text_record(7, "evidence-a", "web", "different route"), b"canonical trajectory bytes").unwrap();
        let mismatched_evidence = other_index.snapshot_bytes(3).unwrap();
        assert!(restored.verify(hdc_bytes, &mismatched_evidence).is_err());
    }

    #[test]
    fn commit_manifest_rejects_zero_generation_and_invalid_snapshot() {
        assert!(EvidenceCommitManifest::new(0, b"index", b"not a snapshot").is_err());
        assert!(EvidenceCommitManifest::new(1, b"index", b"not a snapshot").is_err());
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "current route"), b"canonical trajectory bytes").unwrap();
        assert!(index.snapshot_bytes(0).is_err());
        let evidence_bytes = index.snapshot_bytes(3).unwrap();
        assert!(EvidenceCommitManifest::new(0, b"index", &evidence_bytes).is_err());
        assert!(EvidenceCommitManifest::new(4, b"index", &evidence_bytes).is_err());
    }

    #[test]
    fn serializes_context_items_to_official_type_value_shape() {
        let item = MemoryContextItem {
            kind: MemoryContextType::Text,
            value: "button opens cart".to_string(),
        };
        let json = serde_json::to_value(item).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["value"], "button opens cart");
        assert!(json.get("kind").is_none());
    }

    #[test]
    fn resolves_text_with_digest_and_preserves_hit_order() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "cart icon opens cart"), b"canonical trajectory bytes").unwrap();
        index.insert(text_record(8, "evidence-b", "web", "checkout uses address panel"), b"canonical trajectory bytes").unwrap();
        let items = index.resolve(&[8, 7, 8, 404], "web", 150, Path::new("."), 3).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].value, "checkout uses address panel");
        assert_eq!(items[1].value, "cart icon opens cart");
        assert_eq!(items[0].kind, MemoryContextType::Text);
    }

    #[test]
    fn refuses_cross_scope_and_time_invalid_evidence() {
        let mut index = EvidenceIndex::new();
        let mut record = text_record(7, "evidence-a", "enterprise", "admin panel uses roles");
        record.valid_from_unix_ms = Some(200);
        record.valid_until_unix_ms = Some(300);
        index.insert(record, b"canonical trajectory bytes").unwrap();
        assert!(index.resolve(&[7], "web", 250, Path::new("."), 10).unwrap().is_empty());
        assert!(index.resolve(&[7], "enterprise", 199, Path::new("."), 10).unwrap().is_empty());
        assert!(index.resolve(&[7], "enterprise", 300, Path::new("."), 10).unwrap().is_empty());
        assert_eq!(index.resolve(&[7], "enterprise", 250, Path::new("."), 10).unwrap().len(), 1);
    }

    #[test]
    fn revoked_and_deleted_records_never_resolve_or_reinsert() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "old route"), b"canonical trajectory bytes").unwrap();
        index.insert(text_record(8, "evidence-b", "web", "new route"), b"canonical trajectory bytes").unwrap();
        index.revoke(7).unwrap();
        index.mark_deleted(8).unwrap();
        assert!(index.resolve(&[7, 8], "web", 150, Path::new("."), 10).unwrap().is_empty());
        assert!(index.insert(text_record(7, "evidence-c", "web", "resurrected route"), b"canonical trajectory bytes").is_err());
        assert!(index.revoke(8).is_err());
    }

    #[test]
    fn supersession_hides_old_record_and_requires_same_scope() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-old", "web", "old selector"), b"canonical trajectory bytes").unwrap();
        index.insert(text_record(8, "evidence-new", "web", "new selector"), b"canonical trajectory bytes").unwrap();
        index.supersede(7, 8).unwrap();
        let items = index.resolve(&[7, 8], "web", 150, Path::new("."), 10).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].value, "new selector");

        let mut other = EvidenceIndex::new();
        other.insert(text_record(1, "web-record", "web", "web payload"), b"canonical trajectory bytes").unwrap();
        other.insert(text_record(2, "enterprise-record", "enterprise", "enterprise payload"), b"canonical trajectory bytes").unwrap();
        assert!(other.supersede(1, 2).is_err());
    }

    #[test]
    fn rejects_provenance_digest_mismatch_before_indexing() {
        let mut index = EvidenceIndex::new();
        let record = text_record(9, "evidence-provenance", "web", "source-bound evidence");
        assert!(index.insert(record.clone(), b"tampered source bytes").is_err());
        // A failed insert must not reserve either ID.
        assert!(index.insert(record, b"canonical trajectory bytes").is_ok());
    }

    #[test]
    fn refuses_duplicate_vector_and_evidence_ids() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "payload"), b"canonical trajectory bytes").unwrap();
        assert!(index.insert(text_record(7, "evidence-b", "web", "other payload"), b"canonical trajectory bytes").is_err());
        assert!(index.insert(text_record(8, "evidence-a", "web", "other payload"), b"canonical trajectory bytes").is_err());
    }

    #[test]
    fn tampered_text_digest_fails_closed() {
        let mut index = EvidenceIndex::new();
        let mut record = text_record(7, "evidence-a", "web", "trusted text");
        record.content_sha256 = hash(b"different text");
        index.insert(record, b"canonical trajectory bytes").unwrap();
        assert!(index.resolve(&[7], "web", 150, Path::new("."), 10).is_err());
    }

    #[test]
    fn invalid_hashes_and_lifecycle_shapes_are_rejected() {
        let mut record = text_record(7, "evidence-a", "web", "payload");
        record.source.source_sha256 = "not-a-digest".to_string();
        assert!(record.validate().is_err());

        let mut record = text_record(8, "evidence-b", "web", "payload");
        record.lifecycle = EvidenceLifecycle::Superseded;
        assert!(record.validate().is_err());
        record.superseded_by = Some("evidence-b".to_string());
        assert!(record.validate().is_err());
    }

    #[test]
    fn verifies_image_digest_and_emits_canonical_path() {
        let root = tempdir().unwrap();
        let image_bytes = b"fake PNG bytes for contract test";
        fs::write(root.path().join("step.png"), image_bytes).unwrap();

        let mut record = text_record(7, "evidence-image", "web", "placeholder");
        record.payload = EvidencePayload::Image { path: "step.png".to_string() };
        record.content_sha256 = hash(image_bytes);
        let item = record.context_item(root.path()).unwrap();
        assert_eq!(item.kind, MemoryContextType::Image);
        assert_eq!(item.value, fs::canonicalize(root.path().join("step.png")).unwrap().to_string_lossy().into_owned());
    }

    #[test]
    fn rejects_path_traversal_and_image_digest_mismatch() {
        let root = tempdir().unwrap();
        fs::write(root.path().join("step.png"), b"bytes").unwrap();

        let mut traversal = text_record(7, "evidence-a", "web", "placeholder");
        traversal.payload = EvidencePayload::Image { path: "../step.png".to_string() };
        assert!(traversal.validate().is_err());

        let mut tampered = text_record(8, "evidence-b", "web", "placeholder");
        tampered.payload = EvidencePayload::Image { path: "step.png".to_string() };
        tampered.content_sha256 = hash(b"different bytes");
        assert!(tampered.context_item(root.path()).is_err());
    }

    #[test]
    fn resolve_with_provenance_preserves_source_and_abi_projection() {
        let mut index = EvidenceIndex::new();
        index.insert(
            text_record(7, "evidence-attributed", "web", "grounded answer"),
            b"canonical trajectory bytes",
        ).unwrap();

        let resolved = index.resolve_with_provenance(
            &[7, 7], "web", 150, Path::new("."), 5,
        ).unwrap();
        assert_eq!(resolved.len(), 1, "duplicate candidate IDs must not duplicate context");
        assert_eq!(resolved[0].hdc_id, 7);
        assert_eq!(resolved[0].evidence_id, "evidence-attributed");
        assert_eq!(resolved[0].source.trajectory_id, "trajectory-001");
        assert_eq!(resolved[0].source.step_id.as_deref(), Some("step-4"));
        assert_eq!(resolved[0].item.value, "grounded answer");

        let abi = index.resolve(&[7, 7], "web", 150, Path::new("."), 5).unwrap();
        assert_eq!(abi, resolved.into_iter().map(|entry| entry.item).collect::<Vec<_>>());
    }

}
