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
pub const EVIDENCE_COMMIT_MANIFEST_SCHEMA_VERSION: &str = "BROCA-EVIDENCE-COMMIT-MANIFEST-V2";

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
}

impl EvidenceCommitManifest {
    /// Construct a low-level pair-binding manifest for fixtures that are not a
    /// real HdcStore. Production callers should prefer new_with_hdc_identity.
    pub fn new(generation: u64, hdc_index_bytes: &[u8], evidence_snapshot_bytes: &[u8]) -> Result<Self, String> {
        Self::new_with_optional_hdc_identity(generation, hdc_index_bytes, evidence_snapshot_bytes, None)
    }

    /// Construct a manifest that commits the actual HDC store's logical identity
    /// in addition to the exact on-disk file-byte digest.
    pub fn new_with_hdc_identity(
        generation: u64,
        hdc_index_bytes: &[u8],
        evidence_snapshot_bytes: &[u8],
        identity: HdcStoreIdentity,
    ) -> Result<Self, String> {
        identity.validate_for_recovery()?;
        Self::new_with_optional_hdc_identity(generation, hdc_index_bytes, evidence_snapshot_bytes, Some(identity))
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
        })
    }

    pub fn hdc_store_identity(&self) -> Option<&HdcStoreIdentity> {
        self.hdc_store_identity.as_ref()
    }

    fn validate_envelope(&self) -> Result<(), String> {
        if self.schema_version != EVIDENCE_COMMIT_MANIFEST_SCHEMA_VERSION {
            return Err("unsupported evidence commit manifest schema version".into());
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
#[derive(Debug, Default)]
pub struct EvidenceIndex {
    records: BTreeMap<u64, EvidenceRecord>,
    evidence_ids: BTreeMap<String, u64>,
}

impl EvidenceIndex {
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
