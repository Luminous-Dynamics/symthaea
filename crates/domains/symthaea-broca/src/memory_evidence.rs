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
//! This is not yet a persistent payload store, trajectory ingester, query-text
//! encoder, or complete LongMemEval-V2 backend.

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
    /// HDC-ID-sorted records include revoked, superseded and deleted tombstones.
    records: Vec<EvidenceRecord>,
    records_sha256: String,
}

/// Binds the exact vector-index bytes and evidence snapshot to one generation.
///
/// This detects accidentally mixing an old vector index with a newer payload
/// snapshot (or vice versa). It is an unkeyed consistency manifest, not a
/// signature: authenticate or externally pin its bytes when storage is hostile.
pub const EVIDENCE_COMMIT_MANIFEST_SCHEMA_VERSION: &str = "BROCA-EVIDENCE-COMMIT-MANIFEST-V1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceCommitManifest {
    schema_version: String,
    /// Monotonic generation assigned by the persistence coordinator, starting at 1.
    generation: u64,
    hdc_index_sha256: String,
    evidence_snapshot_sha256: String,
}

impl EvidenceCommitManifest {
    /// Construct a pair-binding manifest for exact persisted bytes.
    pub fn new(generation: u64, hdc_index_bytes: &[u8], evidence_snapshot_bytes: &[u8]) -> Result<Self, String> {
        if generation == 0 {
            return Err("evidence commit generation must start at 1".into());
        }
        // Refuse to bind a manifest to an invalid or corrupt evidence snapshot.
        EvidenceIndex::restore_snapshot(evidence_snapshot_bytes)?;
        Ok(Self {
            schema_version: EVIDENCE_COMMIT_MANIFEST_SCHEMA_VERSION.to_string(),
            generation,
            hdc_index_sha256: sha256_hex(hdc_index_bytes),
            evidence_snapshot_sha256: sha256_hex(evidence_snapshot_bytes),
        })
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
    pub fn snapshot_bytes(&self) -> Result<Vec<u8>, String> {
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

        let records_json = serde_json::to_vec(&records)
            .map_err(|error| format!("failed to serialize evidence records: {error}"))?;
        let snapshot = EvidenceIndexSnapshot {
            schema_version: EVIDENCE_INDEX_SNAPSHOT_SCHEMA_VERSION.to_string(),
            records,
            records_sha256: sha256_hex(&records_json),
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
        if !is_sha256(&snapshot.records_sha256) {
            return Err("snapshot records_sha256 must be lowercase SHA-256".into());
        }
        let records_json = serde_json::to_vec(&snapshot.records)
            .map_err(|error| format!("failed to serialize restored records: {error}"))?;
        if sha256_hex(&records_json) != snapshot.records_sha256 {
            return Err("evidence snapshot record digest mismatch".into());
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

    /// Resolve ranked HDC IDs to safe context items. Missing IDs, duplicates,
    /// inactive records, invalid times and cross-scope records are skipped.
    /// A present but corrupt payload is an error, not accepted evidence.
    pub fn resolve(
        &self,
        hdc_hits: &[u64],
        scope_id: &str,
        now_ms: u64,
        payload_root: &Path,
        max_items: usize,
    ) -> Result<Vec<MemoryContextItem>, String> {
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
            output.push(record.context_item(payload_root)?);
        }
        Ok(output)
    }
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
    fn snapshot_round_trip_preserves_tombstones_and_blocks_resurrection() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-deleted", "web", "old route"), b"canonical trajectory bytes").unwrap();
        index.insert(text_record(8, "evidence-active", "web", "current route"), b"canonical trajectory bytes").unwrap();
        index.mark_deleted(7).unwrap();

        let snapshot = index.snapshot_bytes().unwrap();
        let restored = EvidenceIndex::restore_snapshot(&snapshot).unwrap();
        let items = restored.resolve(&[7, 8, 7], "web", 150, Path::new("."), 10).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].value, "current route");
        assert!(restored.insert(
            text_record(7, "evidence-resurrection", "web", "resurrected route"),
            b"canonical trajectory bytes",
        ).is_err());
        assert_eq!(restored.snapshot_bytes().unwrap(), snapshot);
    }

    #[test]
    fn snapshot_rejects_record_tampering_and_noncanonical_envelopes() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "trusted route"), b"canonical trajectory bytes").unwrap();
        let snapshot = index.snapshot_bytes().unwrap();

        let mut value: serde_json::Value = serde_json::from_slice(&snapshot).unwrap();
        value["records"][0]["payload"]["value"] = serde_json::Value::String("tampered route".into());
        let tampered = serde_json::to_vec(&value).unwrap();
        assert!(EvidenceIndex::restore_snapshot(&tampered).is_err());

        value["schema_version"] = serde_json::Value::String("BROCA-EVIDENCE-INDEX-SNAPSHOT-V999".into());
        assert!(EvidenceIndex::restore_snapshot(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn snapshot_rebuild_rejects_duplicate_ids_even_with_recomputed_digest() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "route one"), b"canonical trajectory bytes").unwrap();
        let snapshot = index.snapshot_bytes().unwrap();
        let mut envelope: EvidenceIndexSnapshot = serde_json::from_slice(&snapshot).unwrap();
        envelope.records.push(envelope.records[0].clone());
        envelope.records_sha256 = hash(&serde_json::to_vec(&envelope.records).unwrap());
        let crafted = serde_json::to_vec(&envelope).unwrap();
        assert!(EvidenceIndex::restore_snapshot(&crafted).is_err());
    }

    #[test]
    fn commit_manifest_binds_matching_vector_index_and_evidence_snapshot() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "current route"), b"canonical trajectory bytes").unwrap();
        index.mark_deleted(7).unwrap();
        let evidence_bytes = index.snapshot_bytes().unwrap();
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
        let mismatched_evidence = other_index.snapshot_bytes().unwrap();
        assert!(restored.verify(hdc_bytes, &mismatched_evidence).is_err());
    }

    #[test]
    fn commit_manifest_rejects_zero_generation_and_invalid_snapshot() {
        assert!(EvidenceCommitManifest::new(0, b"index", b"not a snapshot").is_err());
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "current route"), b"canonical trajectory bytes").unwrap();
        let evidence_bytes = index.snapshot_bytes().unwrap();
        assert!(EvidenceCommitManifest::new(0, b"index", &evidence_bytes).is_err());
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
}
