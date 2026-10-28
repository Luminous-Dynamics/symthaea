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
    /// RFC 6901 JSON Pointer to the exact string payload inside the source trajectory.
    pub source_pointer: String,
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

/// Deterministic BM25 retrieval settings for the lexical evidence baseline.
/// This is a transparent baseline, not a semantic encoder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bm25Config {
    /// Term-frequency saturation parameter; must be finite and positive.
    pub k1: f64,
    /// Document-length normalization in [0, 1].
    pub b: f64,
    /// Maximum number of evidence items returned.
    pub max_items: usize,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            max_items: 5,
        }
    }
}

impl Bm25Config {
    fn validate(&self) -> Result<(), String> {
        if !self.k1.is_finite() || self.k1 <= 0.0 {
            return Err("BM25 k1 must be finite and positive".into());
        }
        if !self.b.is_finite() || !(0.0..=1.0).contains(&self.b) {
            return Err("BM25 b must be finite and in [0, 1]".into());
        }
        if self.max_items == 0 {
            return Err("BM25 max_items must be positive".into());
        }
        Ok(())
    }
}

/// A scored evidence candidate. Keep this metadata outside the reader context;
/// the adapter should pass only `context` as a benchmark MemoryContextItem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LexicalEvidenceHit {
    pub hdc_id: u64,
    pub evidence_id: String,
    pub source: EvidenceSource,
    pub content_sha256: String,
    pub bm25_score: f64,
    pub context: MemoryContextItem,
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
        if !self.source.source_pointer.starts_with('/')
            || self.source.source_pointer.trim() == "/"
        {
            return Err("source_pointer must identify a non-root JSON value".into());
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
            EvidencePayload::Text { value } => {
                if value.trim().is_empty() {
                    return Err("text evidence must be non-empty".into());
                }
                // Text is immutable after insertion, so reject a mismatched digest
                // before it can populate cached lexical postings or reach a reader.
                if sha256_hex(value.as_bytes()) != self.content_sha256 {
                    return Err("text content_sha256 mismatch".into());
                }
            }
            EvidencePayload::Image { path } => {
                if path.trim().is_empty() {
                    return Err("image path must be non-empty".into());
                }
                if Path::new(path).components().any(|part| matches!(part, Component::ParentDir)) {
                    return Err("image path must not contain parent-directory traversal".into());
                }
            }
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
        let source_json: serde_json::Value = serde_json::from_slice(source_bytes)
            .map_err(|error| format!("source bytes are not valid JSON: {error}"))?;
        self.verify_source_value(&source_json)
    }

    fn verify_source_value(&self, source_json: &serde_json::Value) -> Result<(), String> {
        let source_value = source_json
            .pointer(&self.source.source_pointer)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "source_pointer must resolve to a string value".to_string())?;
        let expected_value = match &self.payload {
            EvidencePayload::Text { value } => value.as_str(),
            EvidencePayload::Image { path } => path.as_str(),
        };
        if source_value != expected_value {
            return Err("payload does not match the source string at source_pointer".into());
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

/// Process-local map from HDC candidate IDs to evidence payloads.
///
/// Production persistence must commit lifecycle/tombstone state atomically with
/// the durable vector index. This in-memory resolver does not solve that yet.
#[derive(Debug, Default)]
pub struct EvidenceIndex {
    records: BTreeMap<u64, EvidenceRecord>,
    evidence_ids: BTreeMap<String, u64>,
    // Text is digest-checked and tokenized once at ingestion. Postings keep each
    // term once per vocabulary key instead of duplicating term strings per doc.
    lexical_postings: BTreeMap<String, Vec<(u64, usize)>>,
    lexical_document_lengths: BTreeMap<u64, usize>,
}

impl EvidenceIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, record: EvidenceRecord, source_bytes: &[u8]) -> Result<(), String> {
        self.insert_batch_atomic(vec![record], source_bytes).map(|_| ())
    }

    /// Validate the entire ingestion batch before mutating the index.
    ///
    /// This prevents a malformed final field or an ID collision from leaving a
    /// partially indexed trajectory. All records must point into the same exact
    /// source bytes and begin in the Active lifecycle state.
    pub fn insert_batch_atomic(
        &mut self,
        records: Vec<EvidenceRecord>,
        source_bytes: &[u8],
    ) -> Result<usize, String> {
        let source_sha256 = sha256_hex(source_bytes);
        let source_json: serde_json::Value = serde_json::from_slice(source_bytes)
            .map_err(|error| format!("source bytes are not valid JSON: {error}"))?;
        let mut incoming_hdc_ids = BTreeSet::new();
        let mut incoming_evidence_ids = BTreeSet::new();
        let mut pending_lexical: Vec<(u64, Option<(usize, Vec<(String, usize)>)>)> =
            Vec::with_capacity(records.len());

        for record in &records {
            if record.lifecycle != EvidenceLifecycle::Active {
                return Err("new evidence must enter with active lifecycle".into());
            }
            record.validate()?;
            if record.source.source_sha256 != source_sha256 {
                return Err("source_sha256 mismatch".into());
            }
            record.verify_source_value(&source_json)?;

            let lexical_document = match &record.payload {
                EvidencePayload::Text { value } => {
                    if sha256_hex(value.as_bytes()) != record.content_sha256 {
                        return Err("text content_sha256 mismatch".into());
                    }
                    let tokens = tokenize_lexical(value);
                    if tokens.is_empty() {
                        None
                    } else {
                        let mut terms = BTreeMap::new();
                        for token in &tokens {
                            *terms.entry(token.clone()).or_insert(0usize) += 1;
                        }
                        Some((tokens.len(), terms.into_iter().collect()))
                    }
                }
                EvidencePayload::Image { .. } => None,
            };
            pending_lexical.push((record.hdc_id, lexical_document));

            if self.records.contains_key(&record.hdc_id)
                || !incoming_hdc_ids.insert(record.hdc_id)
            {
                return Err("duplicate hdc_id in store or ingestion batch".into());
            }
            if self.evidence_ids.contains_key(&record.evidence_id)
                || !incoming_evidence_ids.insert(record.evidence_id.clone())
            {
                return Err("duplicate evidence_id in store or ingestion batch".into());
            }
        }

        let count = records.len();
        for record in records {
            self.evidence_ids.insert(record.evidence_id.clone(), record.hdc_id);
            self.records.insert(record.hdc_id, record);
        }
        for (hdc_id, document) in pending_lexical {
            if let Some((length, terms)) = document {
                self.lexical_document_lengths.insert(hdc_id, length);
                for (term, term_frequency) in terms {
                    self.lexical_postings
                        .entry(term)
                        .or_default()
                        .push((hdc_id, term_frequency));
                }
            }
        }
        Ok(count)
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

    /// Revoke an evidence item by its stable external evidence ID.
    /// Numeric mapping remains as a tombstone so stale candidates cannot resurrect it.
    pub fn revoke_evidence(&mut self, evidence_id: &str) -> Result<(), String> {
        let hdc_id = self
            .evidence_ids
            .get(evidence_id)
            .copied()
            .ok_or_else(|| "unknown evidence_id".to_string())?;
        self.revoke(hdc_id)
    }

    /// Mark evidence deleted while retaining a non-resolvable tombstone.
    pub fn delete_evidence(&mut self, evidence_id: &str) -> Result<(), String> {
        let hdc_id = self
            .evidence_ids
            .get(evidence_id)
            .copied()
            .ok_or_else(|| "unknown evidence_id".to_string())?;
        self.mark_deleted(hdc_id)
    }

    /// Supersede evidence by stable IDs, enforcing the existing same-scope rule.
    pub fn supersede_evidence(
        &mut self,
        old_evidence_id: &str,
        new_evidence_id: &str,
    ) -> Result<(), String> {
        let old_hdc_id = self
            .evidence_ids
            .get(old_evidence_id)
            .copied()
            .ok_or_else(|| "unknown old evidence_id".to_string())?;
        let new_hdc_id = self
            .evidence_ids
            .get(new_evidence_id)
            .copied()
            .ok_or_else(|| "unknown replacement evidence_id".to_string())?;
        self.supersede(old_hdc_id, new_hdc_id)
    }

    /// Revoke an evidence item by its stable external evidence ID.
    /// Numeric mapping remains as a tombstone so stale candidates cannot resurrect it.
    pub fn revoke_evidence(&mut self, evidence_id: &str) -> Result<(), String> {
        let hdc_id = self
            .evidence_ids
            .get(evidence_id)
            .copied()
            .ok_or_else(|| "unknown evidence_id".to_string())?;
        self.revoke(hdc_id)
    }

    /// Mark evidence deleted while retaining a non-resolvable tombstone.
    pub fn delete_evidence(&mut self, evidence_id: &str) -> Result<(), String> {
        let hdc_id = self
            .evidence_ids
            .get(evidence_id)
            .copied()
            .ok_or_else(|| "unknown evidence_id".to_string())?;
        self.mark_deleted(hdc_id)
    }

    /// Supersede evidence by stable IDs, enforcing the existing same-scope rule.
    pub fn supersede_evidence(
        &mut self,
        old_evidence_id: &str,
        new_evidence_id: &str,
    ) -> Result<(), String> {
        let old_hdc_id = self
            .evidence_ids
            .get(old_evidence_id)
            .copied()
            .ok_or_else(|| "unknown old evidence_id".to_string())?;
        let new_hdc_id = self
            .evidence_ids
            .get(new_evidence_id)
            .copied()
            .ok_or_else(|| "unknown replacement evidence_id".to_string())?;
        self.supersede(old_hdc_id, new_hdc_id)
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


    /// Search active, in-scope text evidence using deterministic BM25 ranking.
    ///
    /// The candidate corpus is filtered by lifecycle, scope, validity window and
    /// structural validation before document frequency statistics are computed.
    /// Text digests are verified before they can affect ranking. Images are
    /// intentionally excluded from this lexical arm. Unknown query terms return
    /// no hit; ties use ascending HDC ID so repeated runs are reproducible.
    ///
    /// This establishes a simple non-semantic baseline before any HDC/text encoder
    /// is evaluated. It does not use gold answers or benchmark question metadata.
    pub fn search_bm25(
        &self,
        query: &str,
        scope_id: &str,
        now_ms: u64,
        _payload_root: &Path,
        config: Bm25Config,
    ) -> Result<Vec<LexicalEvidenceHit>, String> {
        config.validate()?;
        if scope_id.trim().is_empty() {
            return Err("scope_id must be non-empty".into());
        }

        let query_terms: BTreeSet<String> = tokenize_lexical(query).into_iter().collect();
        if query_terms.is_empty() {
            return Ok(Vec::new());
        }

        // Count the eligible lexical corpus and its average length from compact
        // metadata only. Full text is not tokenized or hashed again per query.
        let mut document_count = 0usize;
        let mut total_document_length = 0usize;
        for (hdc_id, length) in &self.lexical_document_lengths {
            let Some(record) = self.records.get(hdc_id) else {
                continue;
            };
            if !record.is_eligible_for(scope_id, now_ms) {
                continue;
            }
            document_count += 1;
            total_document_length += length;
        }
        if document_count == 0 {
            return Ok(Vec::new());
        }
        let average_length = total_document_length as f64 / document_count as f64;
        if !average_length.is_finite() || average_length <= 0.0 {
            return Ok(Vec::new());
        }

        // Only postings for query terms are visited. Lifecycle/scope/time
        // eligibility is applied before document frequency and candidates, so
        // revoked or out-of-scope postings cannot distort the BM25 statistics.
        let mut document_frequency: BTreeMap<String, usize> =
            query_terms.iter().cloned().map(|term| (term, 0)).collect();
        let mut candidate_term_frequencies: BTreeMap<u64, BTreeMap<String, usize>> =
            BTreeMap::new();
        for term in &query_terms {
            let Some(postings) = self.lexical_postings.get(term) else {
                continue;
            };
            for (hdc_id, term_frequency) in postings {
                let Some(record) = self.records.get(hdc_id) else {
                    continue;
                };
                if !record.is_eligible_for(scope_id, now_ms) {
                    continue;
                }
                *document_frequency
                    .get_mut(term)
                    .expect("query term frequency entry was initialized") += 1;
                candidate_term_frequencies
                    .entry(*hdc_id)
                    .or_default()
                    .insert(term.clone(), *term_frequency);
            }
        }

        let mut hits = Vec::new();
        for (hdc_id, term_frequencies) in candidate_term_frequencies {
            let Some(record) = self.records.get(&hdc_id) else {
                continue;
            };
            let Some(length) = self.lexical_document_lengths.get(&hdc_id) else {
                continue;
            };
            let mut score = 0.0f64;
            for term in &query_terms {
                let Some(&term_frequency) = term_frequencies.get(term) else {
                    continue;
                };
                let df = *document_frequency.get(term).unwrap_or(&0) as f64;
                let idf =
                    (1.0 + (document_count as f64 - df + 0.5) / (df + 0.5)).ln();
                let tf = term_frequency as f64;
                let length_norm = config.k1
                    * (1.0 - config.b + config.b * *length as f64 / average_length);
                score += idf * (tf * (config.k1 + 1.0)) / (tf + length_norm);
            }

            if score <= 0.0 || !score.is_finite() {
                continue;
            }
            let EvidencePayload::Text { value } = &record.payload else {
                continue;
            };
            hits.push(LexicalEvidenceHit {
                hdc_id,
                evidence_id: record.evidence_id.clone(),
                source: record.source.clone(),
                content_sha256: record.content_sha256.clone(),
                bm25_score: score,
                context: MemoryContextItem {
                    kind: MemoryContextType::Text,
                    value: value.clone(),
                },
            });
        }

        hits.sort_by(|left, right| {
            right
                .bm25_score
                .total_cmp(&left.bm25_score)
                .then_with(|| left.hdc_id.cmp(&right.hdc_id))
        });
        hits.truncate(config.max_items);
        Ok(hits)
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


/// Lowercase Unicode alphanumeric tokens; punctuation and whitespace delimit terms.
/// Kept intentionally small and deterministic so the baseline has no model dependency.
fn tokenize_lexical(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for character in text.chars() {
        if character.is_alphanumeric() {
            current.extend(character.to_lowercase());
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

    fn source_bytes_for_text(text: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({"evidence": text})).unwrap()
    }

    fn text_record(id: u64, evidence_id: &str, scope: &str, text: &str) -> EvidenceRecord {
        let source_bytes = source_bytes_for_text(text);
        EvidenceRecord {
            schema_version: EVIDENCE_SCHEMA_VERSION.to_string(),
            evidence_id: evidence_id.to_string(),
            hdc_id: id,
            scope_id: scope.to_string(),
            source: EvidenceSource {
                trajectory_id: "trajectory-001".to_string(),
                step_id: Some("step-4".to_string()),
                step_index: Some(4),
                source_sha256: hash(&source_bytes),
                source_pointer: "/evidence".to_string(),
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


    fn insert_text(index: &mut EvidenceIndex, id: u64, text: &str, scope: &str) {
        index
            .insert(
                text_record(id, &format!("evidence-{id}"), scope, text),
                &source_bytes_for_text(text),
            )
            .unwrap();
    }

    #[test]
    fn bm25_ranks_matching_evidence_and_returns_auditable_hits() {
        let mut index = EvidenceIndex::new();
        insert_text(&mut index, 10, "The checkout payment configuration uses a billing panel", "web");
        insert_text(&mut index, 11, "The account profile has a display name", "web");
        insert_text(&mut index, 12, "Billing configuration for enterprise invoices", "web");

        let hits = index
            .search_bm25(
                "billing configuration",
                "web",
                150,
                Path::new("."),
                Bm25Config::default(),
            )
            .unwrap();

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].hdc_id, 12);
        assert_eq!(hits[1].hdc_id, 10);
        assert!(hits[0].bm25_score > 0.0);
        assert_eq!(hits[0].context.kind, MemoryContextType::Text);
        assert_eq!(hits[0].source.trajectory_id, "trajectory-001");
    }

    #[test]
    fn bm25_filters_scope_and_lifecycle_before_scoring() {
        let mut index = EvidenceIndex::new();
        insert_text(&mut index, 1, "private billing billing", "enterprise");
        insert_text(&mut index, 2, "public billing", "web");
        insert_text(&mut index, 3, "obsolete billing", "web");
        index.revoke(3).unwrap();

        let hits = index
            .search_bm25("billing", "web", 150, Path::new("."), Bm25Config::default())
            .unwrap();

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hdc_id, 2);
    }

    #[test]
    fn bm25_is_deterministic_for_ties_and_empty_or_unknown_queries() {
        let mut index = EvidenceIndex::new();
        insert_text(&mut index, 9, "cart opens overlay", "web");
        insert_text(&mut index, 4, "cart opens overlay", "web");

        let first = index
            .search_bm25("cart", "web", 150, Path::new("."), Bm25Config::default())
            .unwrap();
        let second = index
            .search_bm25("cart", "web", 150, Path::new("."), Bm25Config::default())
            .unwrap();

        assert_eq!(first, second);
        assert_eq!(first.iter().map(|hit| hit.hdc_id).collect::<Vec<_>>(), vec![4, 9]);
        assert!(index
            .search_bm25("   !!! ", "web", 150, Path::new("."), Bm25Config::default())
            .unwrap()
            .is_empty());
        assert!(index
            .search_bm25("nonexistentterm", "web", 150, Path::new("."), Bm25Config::default())
            .unwrap()
            .is_empty());
    }

    #[test]
    fn bm25_rejects_invalid_configuration_and_tampered_candidate_text() {
        let mut index = EvidenceIndex::new();
        insert_text(&mut index, 1, "trusted invoice route", "web");
        let invalid = Bm25Config { k1: 1.2, b: 1.1, max_items: 5 };
        assert!(index
            .search_bm25("invoice", "web", 150, Path::new("."), invalid)
            .is_err());
        assert!(index
            .search_bm25(
                "invoice",
                "web",
                150,
                Path::new("."),
                Bm25Config { max_items: 0, ..Bm25Config::default() },
            )
            .is_err());

        let mut corrupt = text_record(2, "corrupt-evidence", "web", "invoice route");
        corrupt.content_sha256 = hash(b"tampered text");
        assert!(index
            .insert(corrupt, &source_bytes_for_text("invoice route"))
            .is_err());
        let hits = index
            .search_bm25("invoice", "web", 150, Path::new("."), Bm25Config::default())
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].hdc_id, 1);
    }


    #[test]
    fn rejects_payload_not_present_at_its_source_pointer() {
        let mut index = EvidenceIndex::new();
        let record = text_record(21, "evidence-source-pointer", "web", "trusted source string");
        let mismatched_source = serde_json::to_vec(&serde_json::json!({
            "evidence": "different string"
        })).unwrap();
        assert!(index.insert(record.clone(), &mismatched_source).is_err());
        assert!(index
            .insert(record, &source_bytes_for_text("trusted source string"))
            .is_ok());
    }

    #[test]
    fn ordinary_insert_rejects_pre_revoked_or_deleted_records() {
        let mut index = EvidenceIndex::new();
        let mut record = text_record(22, "evidence-revoked", "web", "blocked evidence");
        record.lifecycle = EvidenceLifecycle::Revoked;
        assert!(index
            .insert(record, &source_bytes_for_text("blocked evidence"))
            .is_err());
    }


    #[test]
    fn evidence_id_lifecycle_transitions_are_query_visible_and_stable() {
        let mut index = EvidenceIndex::new();
        let old = text_record(101, "old-evidence", "web", "billing panel moved");
        let new = text_record(102, "new-evidence", "web", "billing panel now here");
        index.insert(old, &source_bytes_for_text("billing panel moved")).unwrap();
        index.insert(new, &source_bytes_for_text("billing panel now here")).unwrap();

        index.supersede_evidence("old-evidence", "new-evidence").unwrap();
        let hits = index
            .search_bm25("billing panel", "web", 150, Path::new("."), Bm25Config::default())
            .unwrap();
        assert!(hits.iter().all(|hit| hit.evidence_id != "old-evidence"));

        index.revoke_evidence("new-evidence").unwrap();
        assert!(index
            .search_bm25("billing panel", "web", 150, Path::new("."), Bm25Config::default())
            .unwrap()
            .is_empty());
        index.delete_evidence("new-evidence").unwrap();
        assert!(index.revoke_evidence("new-evidence").is_err());
    }


    #[test]
    fn evidence_id_lifecycle_transitions_are_query_visible_and_stable() {
        let mut index = EvidenceIndex::new();
        let old = text_record(101, "old-evidence", "web", "billing panel moved");
        let new = text_record(102, "new-evidence", "web", "billing panel now here");
        index.insert(old, &source_bytes_for_text("billing panel moved")).unwrap();
        index.insert(new, &source_bytes_for_text("billing panel now here")).unwrap();

        index.supersede_evidence("old-evidence", "new-evidence").unwrap();
        let hits = index
            .search_bm25("billing panel", "web", 150, Path::new("."), Bm25Config::default())
            .unwrap();
        assert!(hits.iter().all(|hit| hit.evidence_id != "old-evidence"));

        index.revoke_evidence("new-evidence").unwrap();
        assert!(index
            .search_bm25("billing panel", "web", 150, Path::new("."), Bm25Config::default())
            .unwrap()
            .is_empty());
        index.delete_evidence("new-evidence").unwrap();
        assert!(index.revoke_evidence("new-evidence").is_err());
    }

    #[test]
    fn batch_ingestion_is_atomic_when_the_final_source_pointer_is_invalid() {
        let mut index = EvidenceIndex::new();
        let source_value = serde_json::json!({
            "evidence": "first source string",
            "second": "different string"
        });
        let source_bytes = serde_json::to_vec(&source_value).unwrap();
        let source_digest = hash(&source_bytes);

        let mut valid = text_record(31, "evidence-valid", "web", "first source string");
        valid.source.source_sha256 = source_digest.clone();
        let mut invalid = text_record(32, "evidence-invalid", "web", "expected source string");
        invalid.source.source_sha256 = source_digest;
        invalid.source.source_pointer = "/second".to_string();

        assert!(index
            .insert_batch_atomic(vec![valid.clone(), invalid], &source_bytes)
            .is_err());
        // The valid first item must not be indexed before the second was checked.
        assert!(index.records.is_empty());
        assert!(index.lexical_postings.is_empty());
        assert!(index.lexical_document_lengths.is_empty());
        assert!(index.insert(valid, &source_bytes).is_ok());
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
        index.insert(text_record(7, "evidence-a", "web", "cart icon opens cart"), &source_bytes_for_text("cart icon opens cart")).unwrap();
        index.insert(text_record(8, "evidence-b", "web", "checkout uses address panel"), &source_bytes_for_text("checkout uses address panel")).unwrap();
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
        index.insert(record, &source_bytes_for_text("admin panel uses roles")).unwrap();
        assert!(index.resolve(&[7], "web", 250, Path::new("."), 10).unwrap().is_empty());
        assert!(index.resolve(&[7], "enterprise", 199, Path::new("."), 10).unwrap().is_empty());
        assert!(index.resolve(&[7], "enterprise", 300, Path::new("."), 10).unwrap().is_empty());
        assert_eq!(index.resolve(&[7], "enterprise", 250, Path::new("."), 10).unwrap().len(), 1);
    }

    #[test]
    fn revoked_and_deleted_records_never_resolve_or_reinsert() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "old route"), &source_bytes_for_text("old route")).unwrap();
        index.insert(text_record(8, "evidence-b", "web", "new route"), &source_bytes_for_text("new route")).unwrap();
        index.revoke(7).unwrap();
        index.mark_deleted(8).unwrap();
        assert!(index.resolve(&[7, 8], "web", 150, Path::new("."), 10).unwrap().is_empty());
        assert!(index.insert(text_record(7, "evidence-c", "web", "resurrected route"), &source_bytes_for_text("resurrected route")).is_err());
        assert!(index.revoke(8).is_err());
    }

    #[test]
    fn supersession_hides_old_record_and_requires_same_scope() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-old", "web", "old selector"), &source_bytes_for_text("old selector")).unwrap();
        index.insert(text_record(8, "evidence-new", "web", "new selector"), &source_bytes_for_text("new selector")).unwrap();
        index.supersede(7, 8).unwrap();
        let items = index.resolve(&[7, 8], "web", 150, Path::new("."), 10).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].value, "new selector");

        let mut other = EvidenceIndex::new();
        other.insert(text_record(1, "web-record", "web", "web payload"), &source_bytes_for_text("web payload")).unwrap();
        other.insert(text_record(2, "enterprise-record", "enterprise", "enterprise payload"), &source_bytes_for_text("enterprise payload")).unwrap();
        assert!(other.supersede(1, 2).is_err());
    }

    #[test]
    fn rejects_provenance_digest_mismatch_before_indexing() {
        let mut index = EvidenceIndex::new();
        let record = text_record(9, "evidence-provenance", "web", "source-bound evidence");
        assert!(index.insert(record.clone(), b"tampered source bytes").is_err());
        // A failed insert must not reserve either ID.
        assert!(index.insert(record, &source_bytes_for_text("source-bound evidence")).is_ok());
    }

    #[test]
    fn refuses_duplicate_vector_and_evidence_ids() {
        let mut index = EvidenceIndex::new();
        index.insert(text_record(7, "evidence-a", "web", "payload"), &source_bytes_for_text("payload")).unwrap();
        assert!(index.insert(text_record(7, "evidence-b", "web", "other payload"), &source_bytes_for_text("other payload")).is_err());
        assert!(index.insert(text_record(8, "evidence-a", "web", "other payload"), &source_bytes_for_text("other payload")).is_err());
    }

    #[test]
    fn tampered_text_digest_is_rejected_before_indexing() {
        let mut index = EvidenceIndex::new();
        let mut record = text_record(7, "evidence-a", "web", "trusted text");
        record.content_sha256 = hash(b"different text");
        assert!(index
            .insert(record, &source_bytes_for_text("trusted text"))
            .is_err());
        assert!(index.records.is_empty());
        assert!(index.lexical_postings.is_empty());
        assert!(index.lexical_document_lengths.is_empty());
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
