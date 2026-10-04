// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Knowledge Persistence — SQLite Storage
//!
//! Persists knowledge graph facts and causal edges to SQLite so that
//! knowledge survives process restarts. BinaryHV vectors are stored as
//! 2048-byte BLOBs for efficient Hamming similarity on reload.
//!
//! Schema:
//! - knowledge_facts: id, vector_blob, source_text, confidence, domain, cycle, is_causal
//! - knowledge_causal_edges: cause, effect, strength, cycle
//! - knowledge_ontology: name, vector_blob, usage_count, utility, cycle
//! - knowledge_snapshot_receipts: generation, canonical_digest_hex, receipt_digest_hex
//! - knowledge_snapshot_validation_receipts: validation_sequence, validation_event, generation, validator/profile metadata, outcome
//!
//! Science: Ebbinghaus (1885) memory consolidation across sessions

use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;
use rusqlite::{OptionalExtension, TransactionBehavior};
use symthaea_core::hdc::unified_hv::BinaryHV;
use symthaea_epistemic_types::{ProvenanceRelation, ProvenanceRelationKind};


/// Knowledge persistence layer backed by SQLite.
///
/// Provides save/load for the full knowledge graph state.
/// Uses rusqlite for direct SQL access (no ORM overhead).
pub struct KnowledgePersistence {
    /// Path to the SQLite database file
    db_path: String,
    /// Whether the schema has been initialized
    initialized: bool,
    /// Statistics
    total_saved: u64,
    total_loaded: u64,
}

/// A serializable fact record for persistence
#[derive(Debug, Clone)]
pub struct FactRecord {
    /// Stable memory identity preserved across persistence/reload.
    pub memory_id: String,
    /// Optional canonical identity; absent until an external admission boundary assigns one.
    pub canonical_identity: Option<String>,
    /// Provenance family shared by representations of the same source lineage.
    pub provenance_family: Option<String>,
    /// BinaryHV encoded as raw bytes (2048 bytes for 16,384 bits)
    pub vector_bytes: Vec<u8>,
    /// Source text of the fact
    pub source_text: String,
    /// Confidence score
    pub confidence: f32,
    /// Domain tag (optional)
    pub domain: Option<String>,
    /// Cycle when fact was inserted
    pub cycle: u64,
    /// Whether the fact contains causal relations
    pub is_causal: bool,
}

/// A serializable provenance relation record.
#[derive(Debug, Clone)]
pub struct ProvenanceRelationRecord {
    pub source_memory_id: String,
    pub target_memory_id: String,
    pub kind: ProvenanceRelationKind,
    pub created_at: String,
}

impl From<ProvenanceRelationRecord> for ProvenanceRelation {
    fn from(record: ProvenanceRelationRecord) -> Self {
        Self {
            source_memory_id: record.source_memory_id,
            target_memory_id: record.target_memory_id,
            kind: record.kind,
            created_at: record.created_at,
        }
    }
}

impl From<ProvenanceRelation> for ProvenanceRelationRecord {
    fn from(relation: ProvenanceRelation) -> Self {
        Self {
            source_memory_id: relation.source_memory_id,
            target_memory_id: relation.target_memory_id,
            kind: relation.kind,
            created_at: relation.created_at,
        }
    }
}

/// A serializable causal edge record
#[derive(Debug, Clone)]
pub struct CausalEdgeRecord {
    pub cause: String,
    pub effect: String,
    pub strength: f32,
    pub is_inhibitory: bool,
    pub cycle: u64,
}

impl CausalEdgeRecord {
    fn validate(&self) -> Result<(), String> {
        if self.cause.trim().is_empty() {
            return Err("CausalEdgeRecord cause must be non-empty".into());
        }
        if self.effect.trim().is_empty() {
            return Err("CausalEdgeRecord effect must be non-empty".into());
        }
        if !self.strength.is_finite() || !(-1.0..=1.0).contains(&self.strength) {
            return Err("CausalEdgeRecord strength must be finite and in [-1, 1]".into());
        }
        if (self.is_inhibitory && self.strength > 0.0)
            || (!self.is_inhibitory && self.strength < 0.0)
        {
            return Err(
                "CausalEdgeRecord strength sign must match is_inhibitory metadata".into(),
            );
        }
        if self.cycle > i64::MAX as u64 {
            return Err("CausalEdgeRecord cycle exceeds SQLite INTEGER range".into());
        }
        Ok(())
    }
}

/// A serializable ontology primitive record
#[derive(Debug, Clone)]
pub struct OntologyRecord {
    pub name: String,
    pub vector_bytes: Vec<u8>,
    pub usage_count: u64,
    pub utility: f64,
    pub created_at_cycle: u64,
    pub last_used_cycle: u64,
    /// IS-A parent concept name, if any (NULL in SQLite when absent).
    /// Science: Quillian (1967) — semantic networks; Collins & Loftus (1975).
    pub is_a_parent: Option<String>,
}

/// A complete, single-generation persistence read.
#[derive(Debug, Clone)]
pub struct KnowledgePersistenceSnapshot {
    pub facts: Vec<FactRecord>,
    pub provenance_relations: Vec<ProvenanceRelationRecord>,
    pub causal_edges: Vec<CausalEdgeRecord>,
    pub ontology: Vec<OntologyRecord>,
}

/// Immutable commit metadata for a complete snapshot committed by `save_snapshot`.
///
/// The generation identifies the committed persistence event; the digest identifies
/// the exact canonical content of the snapshot. This is provenance metadata, not
/// an assertion that the snapshot is true or semantically correct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeSnapshotReceipt {
    pub generation: u64,
    pub canonical_digest_hex: String,
    /// Self-digest over temporal receipt metadata; tamper-evidence only.
    pub receipt_digest_hex: String,
}

fn is_hex_digest(value: &str) -> bool {
    value.len() == 64 && value.as_bytes().iter().all(u8::is_ascii_hexdigit)
}

impl KnowledgeSnapshotReceipt {
    /// Recompute the exact v1 self-digest used by persistence verification.
    ///
    /// This exposes the canonical receipt primitive so external evidence tooling can
    /// independently validate a persisted snapshot receipt without reimplementing
    /// the digest algorithm.
    pub fn canonical_receipt_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.epf-011.snapshot-receipt.v1");
        digest_u64(&mut hasher, self.generation);
        digest_str(&mut hasher, &self.canonical_digest_hex);
        hasher.finalize().to_hex().to_string()
    }

    /// The self-digest recomputed from the receipt's public fields.
    pub fn recomputed_receipt_digest_hex(&self) -> String {
        self.canonical_receipt_digest_hex()
    }

    /// Whether the stored self-digest exactly matches the canonical receipt digest.
    pub fn verify_self_digest(&self) -> bool {
        self.receipt_digest_hex == self.recomputed_receipt_digest_hex()
    }

    /// Whether the receipt satisfies its structural metadata invariants and self-digest.
    ///
    /// Unlike verify_self_digest, this rejects malformed generation/digest fields as well.
    pub fn verify_integrity(&self) -> bool {
        self.generation > 0
            && is_hex_digest(&self.canonical_digest_hex)
            && is_hex_digest(&self.receipt_digest_hex)
            && self.verify_self_digest()
    }

    /// Compute a canonical commitment over an already verified receipt history.
    ///
    /// The ordered generation, canonical snapshot digest, and per-receipt self-digest are
    /// all bound into the history commitment. The commitment is deterministic and intended
    /// for external evidence anchoring; it is not an authenticated signature.
    pub fn canonical_history_digest_hex(history: &[Self]) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.epf-011.snapshot-receipt-history.v1");
        digest_u64(&mut hasher, history.len() as u64);
        for receipt in history {
            digest_u64(&mut hasher, receipt.generation);
            digest_str(&mut hasher, &receipt.canonical_digest_hex);
            digest_str(&mut hasher, &receipt.receipt_digest_hex);
        }
        hasher.finalize().to_hex().to_string()
    }

}

/// A non-persistent evidence checkpoint for an externally anchored snapshot-receipt
/// history. The checkpoint does not assert truth or authenticity by itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeSnapshotReceiptHistoryCheckpoint {
    pub receipt_count: u64,
    pub latest_generation: u64,
    pub history_digest_hex: String,
}

impl KnowledgeSnapshotReceiptHistoryCheckpoint {
    /// Build a checkpoint from a receipt history that has already passed ledger verification.
    pub fn from_history(history: &[KnowledgeSnapshotReceipt]) -> Self {
        Self {
            receipt_count: history.len() as u64,
            latest_generation: history.last().map_or(0, |receipt| receipt.generation),
            history_digest_hex: KnowledgeSnapshotReceipt::canonical_history_digest_hex(history),
        }
    }

    /// Whether the checkpoint exactly matches a structurally valid receipt history.
    pub fn verify_against_history(&self, history: &[KnowledgeSnapshotReceipt]) -> bool {
        if !is_hex_digest(&self.history_digest_hex)
            || self.receipt_count != history.len() as u64
            || self.latest_generation != history.last().map_or(0, |receipt| receipt.generation)
        {
            return false;
        }
        if history
            .iter()
            .enumerate()
            .any(|(index, receipt)| {
                receipt.generation != index as u64 + 1 || !receipt.verify_integrity()
            })
        {
            return false;
        }
        self.history_digest_hex == KnowledgeSnapshotReceipt::canonical_history_digest_hex(history)
    }

    /// Whether the checkpoint exactly matches the prefix of a structurally valid, longer receipt history.
    pub fn verify_prefix_against_history(&self, history: &[KnowledgeSnapshotReceipt]) -> bool {
        if self.receipt_count > history.len() as u64 {
            return false;
        }
        self.verify_against_history(&history[..self.receipt_count as usize])
    }
}

/// Immutable record that a named validator evaluated the currently committed
/// complete knowledge snapshot under a specific validator/profile version.
///
/// This records validation provenance only. A conforming result is not a claim
/// that the underlying knowledge is true, and a validator is not implicitly granted
/// authority to assign canonical identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeSnapshotValidationReceipt {
    pub validation_event: String,
    pub generation: u64,
    pub snapshot_digest_hex: String,
    pub validator_ref: String,
    pub validator_version: String,
    pub validation_profile: String,
    pub conforms: bool,
    pub report_digest_hex: Option<String>,
}

impl KnowledgeSnapshotValidationReceipt {
    /// Self-digest used by the current validation ledger schema.
    ///
    /// Version 2 binds the internal append sequence as well as the public receipt
    /// fields, so direct sequence mutation cannot remain self-consistent.
    /// Recompute the sequence-bound v2 self-digest for this validation receipt.
    ///
    /// This exposes the exact canonical digest primitive used by persistence verification
    /// so external evidence tooling can independently reproduce the stored digest.
    pub fn canonical_digest_hex_for_sequence(&self, validation_sequence: u64) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.epf-011.snapshot-validation-receipt.v2");
        digest_u64(&mut hasher, validation_sequence);
        digest_str(&mut hasher, &self.validation_event);
        digest_u64(&mut hasher, self.generation);
        digest_str(&mut hasher, &self.snapshot_digest_hex);
        digest_str(&mut hasher, &self.validator_ref);
        digest_str(&mut hasher, &self.validator_version);
        digest_str(&mut hasher, &self.validation_profile);
        digest_bool(&mut hasher, self.conforms);
        digest_opt_str(&mut hasher, self.report_digest_hex.as_deref());
        hasher.finalize().to_hex().to_string()
    }

    /// Digest format used by the immediately preceding EPF-011 validation-receipt
    /// tranche. Migration accepts only an exact legacy self-digest and rewrites it
    /// deterministically to the sequence-bound v2 form.
    fn legacy_canonical_digest_hex(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.epf-011.snapshot-validation-receipt.v1");
        digest_str(&mut hasher, &self.validation_event);
        digest_u64(&mut hasher, self.generation);
        digest_str(&mut hasher, &self.snapshot_digest_hex);
        digest_str(&mut hasher, &self.validator_ref);
        digest_str(&mut hasher, &self.validator_version);
        digest_str(&mut hasher, &self.validation_profile);
        digest_bool(&mut hasher, self.conforms);
        digest_opt_str(&mut hasher, self.report_digest_hex.as_deref());
        hasher.finalize().to_hex().to_string()
    }

    fn validate_input(&self) -> Result<(), String> {
        if self.validation_event.trim().is_empty() {
            return Err("Snapshot validation event must be non-empty".into());
        }
        if self.generation == 0 {
            return Err("Snapshot validation generation must be positive".into());
        }
        if !is_hex_digest(&self.snapshot_digest_hex) {
            return Err(
                "Snapshot validation digest must be a 64-character hexadecimal digest".into(),
            );
        }
        if self.validator_ref.trim().is_empty() {
            return Err("Snapshot validator reference must be non-empty".into());
        }
        if self.validator_version.trim().is_empty() {
            return Err("Snapshot validator version must be non-empty".into());
        }
        if self.validation_profile.trim().is_empty() {
            return Err("Snapshot validation profile must be non-empty".into());
        }
        if let Some(report_digest) = self.report_digest_hex.as_deref() {
            if !is_hex_digest(report_digest) {
                return Err(
                    "Snapshot validation report digest must be a 64-character hexadecimal digest when present"
                        .into(),
                );
            }
        }
        Ok(())
    }
}

/// A persisted validation receipt together with the append sequence that is
/// cryptographically bound into its self-digest.
///
/// The input type remains sequence-free because the sequence is assigned atomically by
/// SQLite at append time. This record type exposes the assigned sequence for audit/export
/// without allowing callers to choose or spoof it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeSnapshotValidationReceiptRecord {
    pub validation_sequence: u64,
    pub receipt: KnowledgeSnapshotValidationReceipt,
    /// The self-digest stored in SQLite. Audit callers can compare this value with
    /// the digest recomputed from `receipt` and `validation_sequence` without raw SQL.
    pub stored_receipt_digest_hex: String,
}

impl KnowledgeSnapshotValidationReceiptRecord {
    /// Recompute the self-digest from the persisted receipt fields and assigned sequence.
    pub fn recomputed_receipt_digest_hex(&self) -> String {
        self.receipt
            .canonical_digest_hex_for_sequence(self.validation_sequence)
    }

    /// Whether the stored self-digest exactly matches the canonical v2 digest.
    pub fn verify_self_digest(&self) -> bool {
        self.stored_receipt_digest_hex == self.recomputed_receipt_digest_hex()
    }

    /// Whether the receipt record satisfies metadata and self-digest invariants.
    ///
    /// This includes the linked receipt's structural input validation as well as the
    /// sequence-bound stored self-digest.
    pub fn verify_integrity(&self) -> bool {
        self.receipt.validate_input().is_ok() && self.verify_self_digest()
    }

    /// Compute a canonical commitment over an already verified validation history.
    ///
    /// The ordered append sequence, complete receipt fields, and stored v2 self-digest are
    /// all bound into the history commitment. The commitment is deterministic and intended
    /// for external evidence anchoring; it is not an authenticated signature.
    pub fn canonical_history_digest_hex(history: &[Self]) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.epf-011.snapshot-validation-receipt-history.v1");
        digest_u64(&mut hasher, history.len() as u64);
        for record in history {
            digest_u64(&mut hasher, record.validation_sequence);
            digest_str(&mut hasher, &record.receipt.validation_event);
            digest_u64(&mut hasher, record.receipt.generation);
            digest_str(&mut hasher, &record.receipt.snapshot_digest_hex);
            digest_str(&mut hasher, &record.receipt.validator_ref);
            digest_str(&mut hasher, &record.receipt.validator_version);
            digest_str(&mut hasher, &record.receipt.validation_profile);
            digest_bool(&mut hasher, record.receipt.conforms);
            digest_opt_str(
                &mut hasher,
                record.receipt.report_digest_hex.as_deref(),
            );
            digest_str(&mut hasher, &record.stored_receipt_digest_hex);
        }
        hasher.finalize().to_hex().to_string()
    }
}

/// A non-persistent evidence checkpoint for an externally anchored validation-receipt
/// history. The checkpoint does not assert truth, validator authority, or authenticity by itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KnowledgeSnapshotValidationReceiptHistoryCheckpoint {
    pub receipt_count: u64,
    pub latest_validation_sequence: u64,
    pub latest_generation: u64,
    pub history_digest_hex: String,
}

impl KnowledgeSnapshotValidationReceiptHistoryCheckpoint {
    /// Build a checkpoint from validation records that have already passed ledger verification.
    pub fn from_history(history: &[KnowledgeSnapshotValidationReceiptRecord]) -> Self {
        Self {
            receipt_count: history.len() as u64,
            latest_validation_sequence: history
                .last()
                .map_or(0, |record| record.validation_sequence),
            latest_generation: history.last().map_or(0, |record| record.receipt.generation),
            history_digest_hex:
                KnowledgeSnapshotValidationReceiptRecord::canonical_history_digest_hex(history),
        }
    }

    /// Whether the checkpoint exactly matches a structurally valid validation history.
    pub fn verify_against_history(
        &self,
        history: &[KnowledgeSnapshotValidationReceiptRecord],
    ) -> bool {
        if !is_hex_digest(&self.history_digest_hex)
            || self.receipt_count != history.len() as u64
            || self.latest_validation_sequence
                != history.last().map_or(0, |record| record.validation_sequence)
            || self.latest_generation != history.last().map_or(0, |record| record.receipt.generation)
        {
            return false;
        }

        let mut previous_generation = None;
        if history.iter().enumerate().any(|(index, record)| {
            if record.validation_sequence != index as u64 + 1 || !record.verify_integrity() {
                return true;
            }
            if previous_generation.is_some_and(|previous| record.receipt.generation < previous) {
                return true;
            }
            previous_generation = Some(record.receipt.generation);
            false
        }) {
            return false;
        }

        self.history_digest_hex
            == KnowledgeSnapshotValidationReceiptRecord::canonical_history_digest_hex(history)
    }

    /// Whether the checkpoint exactly matches the prefix of a structurally valid, longer validation history.
    pub fn verify_prefix_against_history(
        &self,
        history: &[KnowledgeSnapshotValidationReceiptRecord],
    ) -> bool {
        if self.receipt_count > history.len() as u64 {
            return false;
        }
        self.verify_against_history(&history[..self.receipt_count as usize])
    }
}

impl KnowledgePersistenceSnapshot {
    /// Compute a versioned, order-independent digest of the complete persisted
    /// cognitive snapshot. This is an integrity/evidence identifier, not a claim
    /// about epistemic truth or semantic correctness.
    pub fn canonical_digest(&self) -> blake3::Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea.epf-011.knowledge-snapshot.v1");

        let mut facts: Vec<&FactRecord> = self.facts.iter().collect();
        facts.sort_by(|a, b| {
            a.memory_id
                .cmp(&b.memory_id)
                .then_with(|| a.canonical_identity.cmp(&b.canonical_identity))
                .then_with(|| a.provenance_family.cmp(&b.provenance_family))
                .then_with(|| a.vector_bytes.cmp(&b.vector_bytes))
                .then_with(|| a.source_text.cmp(&b.source_text))
                .then_with(|| a.confidence.to_bits().cmp(&b.confidence.to_bits()))
                .then_with(|| a.domain.cmp(&b.domain))
                .then_with(|| a.cycle.cmp(&b.cycle))
                .then_with(|| a.is_causal.cmp(&b.is_causal))
        });
        digest_section_tag(&mut hasher, b"facts");
        digest_u64(&mut hasher, facts.len() as u64);
        for fact in facts {
            digest_str(&mut hasher, &fact.memory_id);
            digest_opt_str(&mut hasher, fact.canonical_identity.as_deref());
            digest_opt_str(&mut hasher, fact.provenance_family.as_deref());
            digest_bytes(&mut hasher, &fact.vector_bytes);
            digest_str(&mut hasher, &fact.source_text);
            digest_f32(&mut hasher, fact.confidence);
            digest_opt_str(&mut hasher, fact.domain.as_deref());
            digest_u64(&mut hasher, fact.cycle);
            digest_bool(&mut hasher, fact.is_causal);
        }

        let mut relations: Vec<&ProvenanceRelationRecord> =
            self.provenance_relations.iter().collect();
        relations.sort_by(|a, b| {
            a.source_memory_id
                .cmp(&b.source_memory_id)
                .then_with(|| a.target_memory_id.cmp(&b.target_memory_id))
                .then_with(|| provenance_kind_tag(a.kind).cmp(provenance_kind_tag(b.kind)))
                .then_with(|| a.created_at.cmp(&b.created_at))
        });
        digest_section_tag(&mut hasher, b"provenance");
        digest_u64(&mut hasher, relations.len() as u64);
        for relation in relations {
            digest_str(&mut hasher, &relation.source_memory_id);
            digest_str(&mut hasher, &relation.target_memory_id);
            digest_str(&mut hasher, provenance_kind_tag(relation.kind));
            digest_str(&mut hasher, &relation.created_at);
        }

        let mut edges: Vec<&CausalEdgeRecord> = self.causal_edges.iter().collect();
        edges.sort_by(|a, b| {
            a.cause
                .cmp(&b.cause)
                .then_with(|| a.effect.cmp(&b.effect))
                .then_with(|| a.cycle.cmp(&b.cycle))
                .then_with(|| a.strength.to_bits().cmp(&b.strength.to_bits()))
                .then_with(|| a.is_inhibitory.cmp(&b.is_inhibitory))
        });
        digest_section_tag(&mut hasher, b"causal");
        digest_u64(&mut hasher, edges.len() as u64);
        for edge in edges {
            digest_str(&mut hasher, &edge.cause);
            digest_str(&mut hasher, &edge.effect);
            digest_f32(&mut hasher, edge.strength);
            digest_bool(&mut hasher, edge.is_inhibitory);
            digest_u64(&mut hasher, edge.cycle);
        }

        let mut ontology: Vec<&OntologyRecord> = self.ontology.iter().collect();
        ontology.sort_by(|a, b| {
            a.name
                .cmp(&b.name)
                .then_with(|| a.vector_bytes.cmp(&b.vector_bytes))
                .then_with(|| a.usage_count.cmp(&b.usage_count))
                .then_with(|| a.utility.to_bits().cmp(&b.utility.to_bits()))
                .then_with(|| a.created_at_cycle.cmp(&b.created_at_cycle))
                .then_with(|| a.last_used_cycle.cmp(&b.last_used_cycle))
                .then_with(|| a.is_a_parent.cmp(&b.is_a_parent))
        });
        digest_section_tag(&mut hasher, b"ontology");
        digest_u64(&mut hasher, ontology.len() as u64);
        for record in ontology {
            digest_str(&mut hasher, &record.name);
            digest_bytes(&mut hasher, &record.vector_bytes);
            digest_u64(&mut hasher, record.usage_count);
            digest_f64(&mut hasher, record.utility);
            digest_u64(&mut hasher, record.created_at_cycle);
            digest_u64(&mut hasher, record.last_used_cycle);
            digest_opt_str(&mut hasher, record.is_a_parent.as_deref());
        }

        hasher.finalize()
    }

    /// Return the canonical snapshot digest as lowercase hexadecimal.
    pub fn canonical_digest_hex(&self) -> String {
        self.canonical_digest().to_hex().to_string()
    }
}

fn digest_section_tag(hasher: &mut blake3::Hasher, tag: &[u8]) {
    hasher.update(&[0xA5]);
    digest_bytes(hasher, tag);
}

fn digest_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn digest_str(hasher: &mut blake3::Hasher, value: &str) {
    digest_bytes(hasher, value.as_bytes());
}

fn digest_opt_str(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            digest_str(hasher, value);
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn digest_bool(hasher: &mut blake3::Hasher, value: bool) {
    hasher.update(&[u8::from(value)]);
}

fn digest_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn digest_f32(hasher: &mut blake3::Hasher, value: f32) {
    digest_u64(hasher, value.to_bits() as u64);
}

fn digest_f64(hasher: &mut blake3::Hasher, value: f64) {
    digest_u64(hasher, value.to_bits());
}

/// Frozen representation used by the version-1 snapshot digest. Do not switch this
/// to the storage wire code without versioning the snapshot digest domain.
fn provenance_kind_tag(kind: ProvenanceRelationKind) -> &'static str {
    match kind {
        ProvenanceRelationKind::DerivedFrom => "DerivedFrom",
        ProvenanceRelationKind::RevisedFrom => "RevisedFrom",
        ProvenanceRelationKind::Supersedes => "Supersedes",
        ProvenanceRelationKind::Contradicts => "Contradicts",
        ProvenanceRelationKind::Corroborates => "Corroborates",
        ProvenanceRelationKind::RepresentationOf => "RepresentationOf",
    }
}

fn provenance_kind_storage_code(kind: ProvenanceRelationKind) -> &'static str {
    kind.stable_code()
}

fn provenance_kind_from_persisted(value: &str) -> Option<ProvenanceRelationKind> {
    match value {
        "derived_from" | "DerivedFrom" => Some(ProvenanceRelationKind::DerivedFrom),
        "revised_from" | "RevisedFrom" => Some(ProvenanceRelationKind::RevisedFrom),
        "supersedes" | "Supersedes" => Some(ProvenanceRelationKind::Supersedes),
        "contradicts" | "Contradicts" => Some(ProvenanceRelationKind::Contradicts),
        "corroborates" | "Corroborates" => Some(ProvenanceRelationKind::Corroborates),
        "representation_of" | "RepresentationOf" => Some(ProvenanceRelationKind::RepresentationOf),
        _ => None,
    }
}

impl Default for KnowledgePersistence {
    fn default() -> Self {
        Self {
            db_path: String::new(),
            initialized: false,
            total_saved: 0,
            total_loaded: 0,
        }
    }
}

const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(5);

const CURRENT_SCHEMA_USER_VERSION: i64 = 1;

impl KnowledgePersistence {
    /// Create a new persistence layer with the given database path.
    ///
    /// The database file is created on first save if it doesn't exist.
    pub fn new(db_path: impl AsRef<Path>) -> Self {
        Self {
            db_path: db_path.as_ref().to_string_lossy().to_string(),
            initialized: false,
            total_saved: 0,
            total_loaded: 0,
        }
    }

    /// Whether a database path has been configured
    pub fn is_configured(&self) -> bool {
        !self.db_path.is_empty()
    }

    /// Save a batch of fact records to the database.
    ///
    /// Uses a single transaction for efficiency.
    /// Returns the number of facts saved.
    pub fn save_facts(&mut self, facts: &[FactRecord]) -> Result<usize, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        if facts.iter().any(|fact| fact.memory_id.trim().is_empty()) {
            return Err("FactRecord memory_id must be non-empty".into());
        }
        let mut fact_ids = HashSet::with_capacity(facts.len());
        if facts
            .iter()
            .any(|fact| !fact_ids.insert(fact.memory_id.as_str()))
        {
            return Err("Snapshot contains duplicate FactRecord memory_id".into());
        }
        if facts.iter().any(|fact| fact.vector_bytes.len() != BinaryHV::BYTES) {
            return Err(format!(
                "FactRecord vector_bytes must be exactly {} bytes",
                BinaryHV::BYTES
            ));
        }
        if facts.iter().any(|fact| !fact.confidence.is_finite() || !(0.0..=1.0).contains(&fact.confidence)) {
            return Err("FactRecord confidence must be finite and in [0, 1]".into());
        }
        if facts.iter().any(|fact| fact.cycle > i64::MAX as u64) {
            return Err("FactRecord cycle exceeds SQLite INTEGER range".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        conn.execute_batch("BEGIN TRANSACTION")
            .map_err(|e| e.to_string())?;

        let mut count = 0;
        for fact in facts {
            conn.execute(
                "INSERT INTO knowledge_facts (memory_id, canonical_identity, provenance_family, vector_blob, source_text, confidence, domain, cycle, is_causal)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(memory_id) DO UPDATE SET
                    canonical_identity = excluded.canonical_identity,
                    provenance_family = excluded.provenance_family,
                    vector_blob = excluded.vector_blob,
                    source_text = excluded.source_text,
                    confidence = excluded.confidence,
                    domain = excluded.domain,
                    cycle = excluded.cycle,
                    is_causal = excluded.is_causal",
                rusqlite::params![
                    fact.memory_id,
                    fact.canonical_identity,
                    fact.provenance_family,
                    fact.vector_bytes,
                    fact.source_text,
                    fact.confidence,
                    fact.domain,
                    i64::try_from(fact.cycle).expect("fact cycle preflighted for SQLite INTEGER range"),
                    fact.is_causal,
                ],
            )
            .map_err(|e| e.to_string())?;
            count += 1;
        }

        conn.execute_batch("COMMIT").map_err(|e| e.to_string())?;

        self.total_saved += count as u64;
        Ok(count)
    }

    /// Atomically persist a complete knowledge snapshot across all identity-bearing domains.
    ///
    /// Facts, provenance, causal edges, and ontology are committed in one SQLite
    /// transaction after all in-memory inputs have been preflighted.
    pub fn save_snapshot(
        &mut self,
        facts: &[FactRecord],
        relations: &[ProvenanceRelationRecord],
        edges: &[CausalEdgeRecord],
        ontology: &[OntologyRecord],
    ) -> Result<(), String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }
        if facts.iter().any(|fact| fact.memory_id.trim().is_empty()) {
            return Err("FactRecord memory_id must be non-empty".into());
        }
        let mut fact_ids = HashSet::with_capacity(facts.len());
        if facts
            .iter()
            .any(|fact| !fact_ids.insert(fact.memory_id.as_str()))
        {
            return Err("Snapshot contains duplicate FactRecord memory_id".into());
        }
        if facts.iter().any(|fact| fact.vector_bytes.len() != BinaryHV::BYTES) {
            return Err(format!(
                "FactRecord vector_bytes must be exactly {} bytes",
                BinaryHV::BYTES
            ));
        }
        if facts.iter().any(|fact| {
            !fact.confidence.is_finite() || !(0.0..=1.0).contains(&fact.confidence)
        }) {
            return Err("FactRecord confidence must be finite and in [0, 1]".into());
        }
        if facts.iter().any(|fact| fact.cycle > i64::MAX as u64) {
            return Err("FactRecord cycle exceeds SQLite INTEGER range".into());
        }

        for relation in relations {
            ProvenanceRelation::from(relation.clone())
                .validate()
                .map_err(|e| format!("Invalid provenance relation: {e}"))?;
        }
        let mut provenance_keys = HashSet::with_capacity(relations.len());
        if relations.iter().any(|relation| {
            !provenance_keys.insert((
                relation.source_memory_id.as_str(),
                relation.target_memory_id.as_str(),
                relation.kind,
                relation.created_at.as_str(),
            ))
        }) {
            return Err("Snapshot contains duplicate ProvenanceRelationRecord key".into());
        }

        for edge in edges {
            edge.validate()
                .map_err(|e| format!("Invalid causal edge: {e}"))?;
        }
        let mut causal_keys = HashSet::with_capacity(edges.len());
        if edges
            .iter()
            .any(|edge| !causal_keys.insert((edge.cause.as_str(), edge.effect.as_str())))
        {
            return Err("Snapshot contains duplicate CausalEdgeRecord key".into());
        }
        if edges.iter().any(|edge| edge.cycle > i64::MAX as u64) {
            return Err("CausalEdgeRecord cycle exceeds SQLite INTEGER range".into());
        }

        if ontology.iter().any(|record| record.name.trim().is_empty()) {
            return Err("OntologyRecord name must be non-empty".into());
        }
        let mut ontology_names = HashSet::with_capacity(ontology.len());
        if ontology
            .iter()
            .any(|record| !ontology_names.insert(record.name.as_str()))
        {
            return Err("Snapshot contains duplicate OntologyRecord name".into());
        }
        if ontology
            .iter()
            .any(|record| record.vector_bytes.len() != BinaryHV::BYTES)
        {
            return Err(format!(
                "OntologyRecord vector_bytes must be exactly {} bytes",
                BinaryHV::BYTES
            ));
        }
        if ontology.iter().any(|record| !record.utility.is_finite()) {
            return Err("OntologyRecord utility must be finite".into());
        }
        if ontology.iter().any(|record| {
            record.usage_count > i64::MAX as u64
                || record.created_at_cycle > i64::MAX as u64
                || record.last_used_cycle > i64::MAX as u64
        }) {
            return Err("OntologyRecord integer field exceeds SQLite INTEGER range".into());
        }

        let snapshot_digest_hex = KnowledgePersistenceSnapshot {
            facts: facts.to_vec(),
            provenance_relations: relations.to_vec(),
            causal_edges: edges.to_vec(),
            ontology: ontology.to_vec(),
        }
        .canonical_digest_hex();

        let mut conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| format!("Begin snapshot transaction: {e}"))?;
        verify_snapshot_receipts_in_tx(&tx)?;

        // A snapshot represents the current resident projection. Reconcile the
        // projection tables before upserting so facts/causal edges/ontology that
        // were pruned in memory cannot be resurrected on the next startup.
        // Provenance remains append-only below because it is historical lineage.
        delete_absent_keys(
            &tx,
            "knowledge_facts",
            "memory_id",
            facts.iter().map(|f| f.memory_id.as_str()).collect(),
        )?;
        delete_absent_composite_keys(
            &tx,
            "knowledge_causal_edges",
            "cause",
            "effect",
            edges.iter().map(|e| (e.cause.as_str(), e.effect.as_str())).collect(),
        )?;
        delete_absent_keys(
            &tx,
            "knowledge_ontology",
            "name",
            ontology.iter().map(|o| o.name.as_str()).collect(),
        )?;

        let mut saved_count = 0usize;

        for fact in facts {
            tx.execute(
                "INSERT INTO knowledge_facts (memory_id, canonical_identity, provenance_family, vector_blob, source_text, confidence, domain, cycle, is_causal)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT(memory_id) DO UPDATE SET
                    canonical_identity = excluded.canonical_identity,
                    provenance_family = excluded.provenance_family,
                    vector_blob = excluded.vector_blob,
                    source_text = excluded.source_text,
                    confidence = excluded.confidence,
                    domain = excluded.domain,
                    cycle = excluded.cycle,
                    is_causal = excluded.is_causal",
                rusqlite::params![
                    fact.memory_id,
                    fact.canonical_identity,
                    fact.provenance_family,
                    fact.vector_bytes,
                    fact.source_text,
                    fact.confidence,
                    i64::try_from(fact.cycle).expect("fact cycle preflighted for SQLite INTEGER range"),
                    fact.is_causal,
                ],
            )
            .map_err(|e| format!("Snapshot fact: {e}"))?;
            saved_count += 1;
        }

        for relation in relations {
            saved_count += tx
                .execute(
                    "INSERT INTO knowledge_provenance_relations
                     (source_memory_id, target_memory_id, kind, created_at)
                     VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(source_memory_id, target_memory_id, kind, created_at) DO NOTHING",
                    rusqlite::params![
                        relation.source_memory_id,
                        relation.target_memory_id,
                        provenance_kind_storage_code(relation.kind),
                        relation.created_at
                    ],
                )
                .map_err(|e| format!("Snapshot provenance: {e}"))?;
        }

        for edge in edges {
            tx.execute(
                "INSERT INTO knowledge_causal_edges (cause, effect, strength, is_inhibitory, cycle)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(cause, effect) DO UPDATE SET
                    strength = excluded.strength,
                    is_inhibitory = excluded.is_inhibitory,
                    cycle = excluded.cycle",
                rusqlite::params![
                    edge.cause,
                    edge.effect,
                    edge.strength,
                    edge.is_inhibitory,
                    i64::try_from(edge.cycle)
                        .expect("causal edge cycle preflighted for SQLite INTEGER range"),
                ],
            )
            .map_err(|e| format!("Snapshot causal edge: {e}"))?;
            saved_count += 1;
        }

        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT INTO knowledge_ontology
                     (name, vector_blob, usage_count, utility, created_at_cycle, last_used_cycle, is_a_parent)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                     ON CONFLICT(name) DO UPDATE SET
                        vector_blob = excluded.vector_blob,
                        usage_count = excluded.usage_count,
                        utility = excluded.utility,
                        created_at_cycle = excluded.created_at_cycle,
                        last_used_cycle = excluded.last_used_cycle,
                        is_a_parent = excluded.is_a_parent",
                )
                .map_err(|e| format!("Prepare snapshot ontology: {e}"))?;

            for record in ontology {
                stmt.execute(rusqlite::params![
                    record.name,
                    record.vector_bytes,
                    i64::try_from(record.usage_count)
                        .expect("ontology usage count preflighted for SQLite INTEGER range"),
                    record.utility,
                    i64::try_from(record.created_at_cycle)
                        .expect("ontology creation cycle preflighted for SQLite INTEGER range"),
                    i64::try_from(record.last_used_cycle)
                        .expect("ontology last-used cycle preflighted for SQLite INTEGER range"),
                    record.is_a_parent,
                ])
                .map_err(|e| format!("Snapshot ontology: {e}"))?;
                saved_count += 1;
            }
        }

        let actual_snapshot = read_snapshot_from_transaction(&tx)?;
        let actual_digest = actual_snapshot.canonical_digest_hex();
        if actual_digest != snapshot_digest_hex {
            return Err(format!(
                "Snapshot reconciliation digest mismatch: requested {}, persisted {}",
                snapshot_digest_hex, actual_digest
            ));
        }

        let previous_generation = tx
            .query_row(
                "SELECT MAX(generation) FROM knowledge_snapshot_receipts",
                [],
                |row| row.get::<_, Option<i64>>(0),
            )
            .map_err(|e| format!("Load snapshot receipt generation: {e}"))?;
        let generation = match previous_generation {
            None => 1_i64,
            Some(value) => value
                .checked_add(1)
                .ok_or("Snapshot receipt generation exhausted SQLite INTEGER range")?,
        };
        if generation <= 0 {
            return Err("Snapshot receipt generation must be positive".into());
        }

        let receipt = KnowledgeSnapshotReceipt {
            generation: u64::try_from(generation)
                .map_err(|_| "Snapshot receipt generation exceeds SQLite INTEGER range")?,
            canonical_digest_hex: actual_digest.clone(),
            receipt_digest_hex: String::new(),
        };
        let receipt_digest_hex = receipt.canonical_receipt_digest_hex();

        tx.execute(
            "INSERT INTO knowledge_snapshot_receipts
             (generation, canonical_digest_hex, receipt_digest_hex)
             VALUES (?1, ?2, ?3)",
            rusqlite::params![generation, actual_digest, receipt_digest_hex],
        )
        .map_err(|e| format!("Snapshot receipt: {e}"))?;

        tx.commit()
            .map_err(|e| format!("Commit snapshot transaction: {e}"))?;
        self.total_saved += saved_count as u64;
        Ok(())
    }

    /// Load the complete append-only history of committed snapshot receipts.
    ///
    /// Every returned receipt is self-digest verified in the same SQLite transaction.
    /// Historical receipt digests identify the exact snapshot representation committed
    /// at each generation; they do not reconstruct historical projection contents.
    pub fn snapshot_receipt_history(&mut self) -> Result<Vec<KnowledgeSnapshotReceipt>, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin snapshot receipt history verification: {e}"))?;
        verify_snapshot_receipts_in_tx(&tx)?;

        let mut stmt = tx
            .prepare(
                "SELECT generation, canonical_digest_hex, receipt_digest_hex
                 FROM knowledge_snapshot_receipts
                 ORDER BY generation ASC",
            )
            .map_err(|e| format!("Prepare snapshot receipt history: {e}"))?;
        let receipts = stmt
            .query_map([], |row| {
                let generation = row.get::<_, i64>(0)?;
                Ok(KnowledgeSnapshotReceipt {
                    generation: u64::try_from(generation).map_err(|_| {
                        rusqlite::Error::IntegralValueOutOfRange(0, generation)
                    })?,
                    canonical_digest_hex: row.get(1)?,
                    receipt_digest_hex: row.get(2)?,
                })
            })
            .map_err(|e| format!("Query snapshot receipt history: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Load snapshot receipt history row: {e}"))?;

        tx.commit()
            .map_err(|e| format!("Commit snapshot receipt history verification: {e}"))?;
        Ok(receipts)
    }

    /// Compute one deterministic commitment over the complete verified snapshot-receipt history.
    ///
    /// This is an evidence/anchoring primitive. It does not replace per-record verification and
    /// does not provide authenticity without an independent authenticated external anchor.
    pub fn snapshot_receipt_history_digest_hex(&mut self) -> Result<String, String> {
        self.snapshot_receipt_history()
            .map(|history| KnowledgeSnapshotReceipt::canonical_history_digest_hex(&history))
    }

    /// Return a verified, non-persistent checkpoint suitable for external anchoring.
    pub fn snapshot_receipt_history_checkpoint(
        &mut self,
    ) -> Result<KnowledgeSnapshotReceiptHistoryCheckpoint, String> {
        self.snapshot_receipt_history()
            .map(|history| KnowledgeSnapshotReceiptHistoryCheckpoint::from_history(&history))
    }

    /// Verify the current complete snapshot-receipt history against an external checkpoint.
    pub fn verify_snapshot_receipt_history_checkpoint(
        &mut self,
        checkpoint: &KnowledgeSnapshotReceiptHistoryCheckpoint,
    ) -> Result<(), String> {
        let history = self.snapshot_receipt_history()?;
        if checkpoint.verify_against_history(&history) {
            Ok(())
        } else {
            Err("Snapshot receipt history checkpoint mismatch".into())
        }
    }

    /// Verify that the current snapshot-receipt history is a valid append-only extension of an external checkpoint.
    pub fn verify_snapshot_receipt_history_checkpoint_prefix(
        &mut self,
        checkpoint: &KnowledgeSnapshotReceiptHistoryCheckpoint,
    ) -> Result<(), String> {
        let history = self.snapshot_receipt_history()?;
        if checkpoint.verify_prefix_against_history(&history) {
            Ok(())
        } else {
            Err("Snapshot receipt history checkpoint prefix mismatch".into())
        }
    }

    /// Load the latest committed complete-snapshot receipt.
    ///
    /// This receipt is append-only and is only advanced by successful
    /// `save_snapshot` transactions. Individual-domain save methods do not
    /// create receipts because they do not establish a complete snapshot boundary.
    pub fn latest_snapshot_receipt(&mut self) -> Result<Option<KnowledgeSnapshotReceipt>, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin latest snapshot receipt verification: {e}"))?;
        verify_snapshot_receipts_in_tx(&tx)?;
        verify_current_snapshot_matches_latest_receipt_in_tx(&tx)?;

        let receipt = tx
            .query_row(
                "SELECT generation, canonical_digest_hex, receipt_digest_hex
                 FROM knowledge_snapshot_receipts
                 ORDER BY generation DESC
                 LIMIT 1",
                [],
                |row| {
                    let generation = row.get::<_, i64>(0)?;
                    Ok(KnowledgeSnapshotReceipt {
                        generation: u64::try_from(generation).map_err(|_| {
                            rusqlite::Error::IntegralValueOutOfRange(0, generation)
                        })?,
                        canonical_digest_hex: row.get(1)?,
                        receipt_digest_hex: row.get(2)?,
                    })
                },
            )
            .optional()
            .map_err(|e| format!("Load latest snapshot receipt: {e}"))?;

        tx.commit()
            .map_err(|e| format!("Commit latest snapshot receipt verification: {e}"))?;
        Ok(receipt)
    }

    /// Load the latest committed complete snapshot together with the receipt that
    /// certifies that exact content, from one SQLite read transaction.
    ///
    /// This is the preferred API when callers need temporal provenance. The receipt
    /// and all four knowledge domains are observed from one SQLite snapshot, and the
    /// canonical digest is checked before the transaction is released.
    pub fn load_snapshot_with_receipt(
        &mut self,
    ) -> Result<Option<(KnowledgePersistenceSnapshot, KnowledgeSnapshotReceipt)>, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;


        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin verified persistence snapshot read: {e}"))?;

        verify_snapshot_receipts_in_tx(&tx)?;

        let receipt = tx
            .query_row(
                "SELECT generation, canonical_digest_hex, receipt_digest_hex
                 FROM knowledge_snapshot_receipts
                 ORDER BY generation DESC
                 LIMIT 1",
                [],
                |row| {
                    let generation = row.get::<_, i64>(0)?;
                    Ok(KnowledgeSnapshotReceipt {
                        generation: u64::try_from(generation).map_err(|_| {
                            rusqlite::Error::IntegralValueOutOfRange(0, generation)
                        })?,
                        canonical_digest_hex: row.get(1)?,
                        receipt_digest_hex: row.get(2)?,
                    })
                },
            )
            .optional()
            .map_err(|e| format!("Load verified snapshot receipt: {e}"))?;

        let Some(receipt) = receipt else {
            tx.commit()
                .map_err(|e| format!("Commit empty verified snapshot read: {e}"))?;
            return Ok(None);
        };

        let snapshot = read_snapshot_from_transaction(&tx)?;
        let actual_digest = snapshot.canonical_digest_hex();
        if actual_digest != receipt.canonical_digest_hex {
            return Err(format!(
                "Snapshot receipt digest mismatch: generation {} records {}, observed {}",
                receipt.generation, receipt.canonical_digest_hex, actual_digest
            ));
        }

        tx.commit()
            .map_err(|e| format!("Commit verified persistence snapshot read: {e}"))?;

        self.total_loaded += (
            snapshot.facts.len()
            + snapshot.provenance_relations.len()
            + snapshot.causal_edges.len()
            + snapshot.ontology.len()
        ) as u64;

        Ok(Some((snapshot, receipt)))
    }

    /// Record validation of the currently committed complete snapshot.
    ///
    /// Validation is intentionally fail-closed against temporal drift: the requested
    /// generation must still be the latest committed generation, and the current
    /// snapshot digest must equal that generation's receipt. Historical generations
    /// are not reconstructable from the projection tables alone.
    pub fn record_snapshot_validation(
        &mut self,
        validation: KnowledgeSnapshotValidationReceipt,
    ) -> Result<(), String> {
        validation.validate_input()?;
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let mut conn = self.open_connection()?;
        self.ensure_schema(&conn)?;


        // This transaction both observes the snapshot and appends the validation receipt.
        // Starting it as IMMEDIATE avoids the deferred read→write upgrade race that can
        // otherwise surface as SQLITE_BUSY_SNAPSHOT under concurrent writers.
        let tx = conn
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| format!("Begin snapshot validation transaction: {e}"))?;

        verify_snapshot_receipts_in_tx(&tx)?;
        verify_snapshot_validation_receipts_in_tx(&tx)?;

        let latest = tx
            .query_row(
                "SELECT generation, canonical_digest_hex, receipt_digest_hex
                 FROM knowledge_snapshot_receipts
                 ORDER BY generation DESC
                 LIMIT 1",
                [],
                |row| {
                    let generation = row.get::<_, i64>(0)?;
                    Ok(KnowledgeSnapshotReceipt {
                        generation: u64::try_from(generation).map_err(|_| {
                            rusqlite::Error::IntegralValueOutOfRange(0, generation)
                        })?,
                        canonical_digest_hex: row.get(1)?,
                        receipt_digest_hex: row.get(2)?,
                    })
                },
            )
            .optional()
            .map_err(|e| format!("Load latest snapshot receipt for validation: {e}"))?;

        let Some(latest) = latest else {
            return Err("Cannot validate snapshot without a committed receipt".into());
        };

        if validation.generation != latest.generation {
            return Err(format!(
                "Snapshot validation generation is not current: requested {}, current {}",
                validation.generation, latest.generation
            ));
        }

        if validation.snapshot_digest_hex != latest.canonical_digest_hex {
            return Err(format!(
                "Snapshot validation digest does not match committed receipt: requested {}, current {}",
                validation.snapshot_digest_hex, latest.canonical_digest_hex
            ));
        }

        let snapshot = read_snapshot_from_transaction(&tx)?;
        let actual_digest = snapshot.canonical_digest_hex();
        if actual_digest != latest.canonical_digest_hex {
            return Err(format!(
                "Snapshot validation digest mismatch: committed {}, observed {}",
                latest.canonical_digest_hex, actual_digest
            ));
        }

        let previous_validation_sequence = tx
            .query_row(
                "SELECT MAX(validation_sequence)
                 FROM knowledge_snapshot_validation_receipts",
                [],
                |row| row.get::<_, Option<i64>>(0),
            )
            .map_err(|e| format!("Load validation sequence for append: {e}"))?;
        let next_validation_sequence = match previous_validation_sequence {
            None => 1_i64,
            Some(value) => value
                .checked_add(1)
                .ok_or("Validation sequence exhausted SQLite INTEGER range")?,
        };

        tx.execute(
            "INSERT INTO knowledge_snapshot_validation_receipts
             (validation_event, validation_sequence, generation, snapshot_digest_hex, validator_ref,
              validator_version, validation_profile, conforms, report_digest_hex, receipt_digest_hex)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                validation.validation_event,
                next_validation_sequence,
                i64::try_from(validation.generation)
                    .map_err(|_| "Snapshot validation generation exceeds SQLite INTEGER range")?,
                validation.snapshot_digest_hex,
                validation.validator_ref,
                validation.validator_version,
                validation.validation_profile,
                validation.conforms,
                validation.report_digest_hex,
                validation.canonical_digest_hex_for_sequence(
                    u64::try_from(next_validation_sequence)
                        .expect("validation sequence preflighted for SQLite INTEGER range"),
                ),
            ],
        )
        .map_err(|e| format!("Persist snapshot validation receipt: {e}"))?;

        tx.commit()
            .map_err(|e| format!("Commit snapshot validation receipt: {e}"))?;
        Ok(())
    }

    /// Verify the self-digest of every persisted validation receipt.
    ///
    /// This detects accidental/corrupt mutation of validation-event fields. It is
    /// tamper-evidence only: without a separate authenticated key, it does not prove
    /// who made the mutation or establish validator authenticity.
    pub fn verify_snapshot_validation_receipts(&mut self) -> Result<(), String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin validation receipt verification: {e}"))?;
        // Validation receipts are cross-bound to snapshot receipts, so the verifier must
        // validate both ledgers before declaring validation provenance internally consistent.
        verify_snapshot_receipts_in_tx(&tx)?;
        verify_snapshot_validation_receipts_in_tx(&tx)?;
        tx.commit()
            .map_err(|e| format!("Commit validation receipt verification: {e}"))?;
        Ok(())
    }

    /// Load validation receipts for the latest committed snapshot generation.
    ///
    /// The legacy API preserves its historical event-name ordering and receipt-only shape.
    /// Use latest_snapshot_validation_receipt_records when the append sequence is required
    /// for audit/export or independent re-computation of the v2 self-digest.
    pub fn latest_snapshot_validation_receipts(
        &mut self,
    ) -> Result<Vec<KnowledgeSnapshotValidationReceipt>, String> {
        self.load_snapshot_validation_receipt_records(false, true)
            .map(|records| records.into_iter().map(|record| record.receipt).collect())
    }

    /// Load every append-only validation receipt in ledger sequence order.
    ///
    /// Unlike the latest-generation API, this exposes historical validation records as well,
    /// including the assigned sequence and stored v2 self-digest. The entire validation ledger
    /// is integrity-verified in the same SQLite transaction before any records are returned.
    /// Historical generations are intentionally not reconstructed from the current projection;
    /// their ledger bindings remain independently auditable.
    pub fn snapshot_validation_receipt_records(
        &mut self,
    ) -> Result<Vec<KnowledgeSnapshotValidationReceiptRecord>, String> {
        self.load_snapshot_validation_receipt_records(true, false)
    }

    /// Load latest-generation validation receipts together with their SQLite append sequence.
    ///
    /// The returned sequence is the exact value bound into each receipt's v2 self-digest.
    /// Ordering is deterministic by that append sequence.
    pub fn latest_snapshot_validation_receipt_records(
        &mut self,
    ) -> Result<Vec<KnowledgeSnapshotValidationReceiptRecord>, String> {
        self.load_snapshot_validation_receipt_records(true, true)
    }

    /// Compute one deterministic commitment over the complete verified validation-receipt history.
    ///
    /// This is an evidence/anchoring primitive. It does not replace per-record verification and
    /// does not provide authenticity without an independent authenticated external anchor.
    pub fn snapshot_validation_receipt_history_digest_hex(&mut self) -> Result<String, String> {
        self.snapshot_validation_receipt_records().map(|history| {
            KnowledgeSnapshotValidationReceiptRecord::canonical_history_digest_hex(&history)
        })
    }

    /// Return a verified, non-persistent validation-history checkpoint suitable for external anchoring.
    pub fn snapshot_validation_receipt_history_checkpoint(
        &mut self,
    ) -> Result<KnowledgeSnapshotValidationReceiptHistoryCheckpoint, String> {
        self.snapshot_validation_receipt_records()
            .map(|history| KnowledgeSnapshotValidationReceiptHistoryCheckpoint::from_history(&history))
    }

    /// Verify the current complete validation-receipt history against an external checkpoint.
    pub fn verify_snapshot_validation_receipt_history_checkpoint(
        &mut self,
        checkpoint: &KnowledgeSnapshotValidationReceiptHistoryCheckpoint,
    ) -> Result<(), String> {
        let history = self.snapshot_validation_receipt_records()?;
        if checkpoint.verify_against_history(&history) {
            Ok(())
        } else {
            Err("Snapshot validation receipt history checkpoint mismatch".into())
        }
    }

    /// Verify that the current validation-receipt history is a valid append-only extension of an external checkpoint.
    pub fn verify_snapshot_validation_receipt_history_checkpoint_prefix(
        &mut self,
        checkpoint: &KnowledgeSnapshotValidationReceiptHistoryCheckpoint,
    ) -> Result<(), String> {
        let history = self.snapshot_validation_receipt_records()?;
        if checkpoint.verify_prefix_against_history(&history) {
            Ok(())
        } else {
            Err("Snapshot validation receipt history checkpoint prefix mismatch".into())
        }
    }

    fn load_snapshot_validation_receipt_records(
        &mut self,
        order_by_sequence: bool,
        latest_generation_only: bool,
    ) -> Result<Vec<KnowledgeSnapshotValidationReceiptRecord>, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        // Keep verification and retrieval in the same transaction snapshot.
        // A second transaction after verification would re-open a TOCTOU window in
        // which another writer could mutate validation history between the integrity
        // check and the returned rows.
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin latest validation verification: {e}"))?;
        verify_snapshot_receipts_in_tx(&tx)?;
        if latest_generation_only {
            verify_current_snapshot_matches_latest_receipt_in_tx(&tx)?;
        }
        verify_snapshot_validation_receipts_in_tx(&tx)?;

        let order_clause = if order_by_sequence {
            "v.validation_sequence ASC"
        } else {
            "v.validation_event ASC"
        };
        let generation_filter = if latest_generation_only {
            " WHERE v.generation = (
                 SELECT generation
                 FROM knowledge_snapshot_receipts
                 ORDER BY generation DESC
                 LIMIT 1
             )"
        } else {
            ""
        };
        let query = format!(
            "SELECT v.validation_sequence, v.validation_event, v.generation, v.snapshot_digest_hex,
                    v.validator_ref, v.validator_version, v.validation_profile, v.conforms,
                    v.report_digest_hex, v.receipt_digest_hex
             FROM knowledge_snapshot_validation_receipts v
             {generation_filter}
             ORDER BY {order_clause}"
        );
        let mut stmt = tx
            .prepare(&query)
            .map_err(|e| format!("Prepare latest snapshot validations: {e}"))?;

        let records = stmt
            .query_map([], |row| {
                let validation_sequence_i64 = row.get::<_, i64>(0)?;
                let validation_sequence = u64::try_from(validation_sequence_i64).map_err(|_| {
                    rusqlite::Error::IntegralValueOutOfRange(0, validation_sequence_i64)
                })?;
                let generation = row.get::<_, i64>(2)?;
                Ok(KnowledgeSnapshotValidationReceiptRecord {
                    validation_sequence,
                    receipt: KnowledgeSnapshotValidationReceipt {
                        validation_event: row.get(1)?,
                        generation: u64::try_from(generation).map_err(|_| {
                            rusqlite::Error::IntegralValueOutOfRange(2, generation)
                        })?,
                        snapshot_digest_hex: row.get(3)?,
                        validator_ref: row.get(4)?,
                        validator_version: row.get(5)?,
                        validation_profile: row.get(6)?,
                        conforms: row.get(7)?,
                        report_digest_hex: row.get(8)?,
                    },
                    stored_receipt_digest_hex: row.get(9)?,
                })
            })
            .map_err(|e| format!("Query latest snapshot validations: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Load latest snapshot validation row: {e}"))?;

        tx.commit()
            .map_err(|e| format!("Commit latest validation verification: {e}"))?;

        Ok(records)
    }

    /// Load all persistence domains from one SQLite read transaction.
    ///
    /// The returned records are all observed from a single database snapshot. When a
    /// committed snapshot receipt exists, the live projection is verified against it
    /// before records are returned; older databases without receipts remain readable.
    /// This prevents startup restore from combining facts/provenance/causal/ontology
    /// rows committed by different snapshot generations or silently accepting receipt drift.
    pub fn load_snapshot(&mut self) -> Result<KnowledgePersistenceSnapshot, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;


        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin persistence snapshot read: {e}"))?;

        // Preserve the legacy API shape, but fail closed when a committed snapshot
        // receipt exists and the live projection no longer matches it. Databases that
        // have never used save_snapshot remain readable without inventing a receipt.
        verify_snapshot_receipts_in_tx(&tx)?;
        verify_current_snapshot_matches_latest_receipt_in_tx(&tx)?;

        let snapshot = read_snapshot_from_transaction(&tx)?;

        tx.commit()
            .map_err(|e| format!("Commit persistence snapshot read: {e}"))?;

        self.total_loaded += (
            snapshot.facts.len()
            + snapshot.provenance_relations.len()
            + snapshot.causal_edges.len()
            + snapshot.ontology.len()
        ) as u64;

        Ok(snapshot)
    }

    /// Load all fact records from the database.
    pub fn load_facts(&mut self) -> Result<Vec<FactRecord>, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;


        let mut stmt = conn
            .prepare(
                "SELECT id, memory_id, canonical_identity, provenance_family, vector_blob, source_text, confidence, domain, cycle, is_causal
                 FROM knowledge_facts ORDER BY cycle DESC, memory_id ASC, id ASC",
            )
            .map_err(|e| e.to_string())?;

        let facts: Vec<FactRecord> = stmt
            .query_map([], |row| {
                Ok(FactRecord {
                    memory_id: row.get(1)?,
                    canonical_identity: row.get(2)?,
                    provenance_family: row.get(3)?,
                    vector_bytes: row.get(4)?,
                    source_text: row.get(5)?,
                    confidence: row.get(6)?,
                    domain: row.get(7)?,
                    cycle: u64::try_from(row.get::<_, i64>(8)?).map_err(|_| rusqlite::Error::InvalidColumnType(8, "cycle".into(), rusqlite::types::Type::Integer))?,
                    is_causal: row.get(9)?,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Load fact row: {e}"))?;

        self.total_loaded += facts.len() as u64;
        Ok(facts)
    }

    /// Save typed provenance relations append-only. Replaying an existing relation is idempotent.
    pub fn save_provenance_relations(&mut self, relations: &[ProvenanceRelationRecord]) -> Result<usize, String> {
        if !self.is_configured() { return Err("No database path configured".into()); }
        for relation in relations {
            ProvenanceRelation::from(relation.clone())
                .validate()
                .map_err(|e| format!("Invalid provenance relation: {e}"))?;
        }
        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin provenance transaction: {e}"))?;
        let mut count = 0;
        for relation in relations {
            let inserted = tx
                .execute(
                    "INSERT INTO knowledge_provenance_relations
                     (source_memory_id, target_memory_id, kind, created_at)
                     VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(source_memory_id, target_memory_id, kind, created_at) DO NOTHING",
                    rusqlite::params![
                        relation.source_memory_id,
                        relation.target_memory_id,
                        provenance_kind_storage_code(relation.kind),
                        relation.created_at
                    ],
                )
                .map_err(|e| e.to_string())?;
            count += inserted;
        }
        tx.commit()
            .map_err(|e| format!("Commit provenance transaction: {e}"))?;
        self.total_saved += count as u64;
        Ok(count)
    }

    /// Load typed provenance relations from SQLite.
    pub fn load_provenance_relations(&mut self) -> Result<Vec<ProvenanceRelationRecord>, String> {
        if !self.is_configured() { return Err("No database path configured".into()); }
        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let mut stmt = conn.prepare("SELECT source_memory_id, target_memory_id, kind, created_at FROM knowledge_provenance_relations ORDER BY created_at, source_memory_id, target_memory_id, kind").map_err(|e| e.to_string())?;
        let relations = stmt.query_map([], |row| {
            let kind: String = row.get(2)?;
            let kind = provenance_kind_from_persisted(&kind).ok_or_else(|| {
                rusqlite::Error::InvalidColumnType(2, "kind".into(), rusqlite::types::Type::Text)
            })?;
            Ok(ProvenanceRelationRecord { source_memory_id: row.get(0)?, target_memory_id: row.get(1)?, kind, created_at: row.get(3)? })
        }).map_err(|e| e.to_string())?;
        let mut loaded = Vec::new();
        for row in relations {
            let record = row.map_err(|e| e.to_string())?;
            ProvenanceRelation::from(record.clone())
                .validate()
                .map_err(|e| format!("Invalid persisted provenance relation: {e}"))?;
            loaded.push(record);
        }
        let relations = loaded;
        self.total_loaded += relations.len() as u64;
        Ok(relations)
    }

    /// Save causal edge records.
    pub fn save_causal_edges(&mut self, edges: &[CausalEdgeRecord]) -> Result<usize, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }
        for edge in edges {
            edge.validate()?;
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        conn.execute_batch("BEGIN TRANSACTION")
            .map_err(|e| e.to_string())?;

        let mut count = 0;
        for edge in edges {
            conn.execute(
                "INSERT INTO knowledge_causal_edges (cause, effect, strength, is_inhibitory, cycle)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(cause, effect) DO UPDATE SET
                    strength = excluded.strength,
                    is_inhibitory = excluded.is_inhibitory,
                    cycle = excluded.cycle",
                rusqlite::params![
                    edge.cause,
                    edge.effect,
                    edge.strength,
                    edge.is_inhibitory,
                    i64::try_from(edge.cycle).expect("causal edge cycle preflighted for SQLite INTEGER range"),
                ],
            )
            .map_err(|e| e.to_string())?;
            count += 1;
        }

        conn.execute_batch("COMMIT").map_err(|e| e.to_string())?;

        self.total_saved += count as u64;
        Ok(count)
    }

    /// Load causal edge records.
    pub fn load_causal_edges(&mut self) -> Result<Vec<CausalEdgeRecord>, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        let mut stmt = conn
            .prepare(
                "SELECT cause, effect, strength, is_inhibitory, cycle FROM knowledge_causal_edges\n                 ORDER BY cycle DESC, cause ASC, effect ASC",
            )
            .map_err(|e| e.to_string())?;

        let edges: Vec<CausalEdgeRecord> = stmt
            .query_map([], |row| {
                Ok(CausalEdgeRecord {
                    cause: row.get(0)?,
                    effect: row.get(1)?,
                    strength: row.get(2)?,
                    is_inhibitory: row.get(3)?,
                    cycle: u64::try_from(row.get::<_, i64>(4)?).map_err(|_| rusqlite::Error::InvalidColumnType(4, "cycle".into(), rusqlite::types::Type::Integer))?,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Load causal edge row: {e}"))?;

        for edge in &edges {
            edge.validate()
                .map_err(|e| format!("Invalid persisted causal edge: {e}"))?;
        }

        self.total_loaded += edges.len() as u64;
        Ok(edges)
    }

    /// Save ontology primitives to the database.
    ///
    /// Uses an explicit PRIMARY KEY conflict target so the persisted name
    /// remains the identity being updated rather than using REPLACE semantics.
    /// Returns the number of primitives saved.
    pub fn save_ontology(&mut self, records: &[OntologyRecord]) -> Result<usize, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }
        if records.iter().any(|record| record.name.trim().is_empty()) {
            return Err("OntologyRecord name must be non-empty".into());
        }
        if records.iter().any(|record| record.vector_bytes.len() != BinaryHV::BYTES) {
            return Err(format!(
                "OntologyRecord vector_bytes must be exactly {} bytes",
                BinaryHV::BYTES
            ));
        }
        if records.iter().any(|record| !record.utility.is_finite()) {
            return Err("OntologyRecord utility must be finite".into());
        }
        if records.iter().any(|record| {
            record.usage_count > i64::MAX as u64
                || record.created_at_cycle > i64::MAX as u64
                || record.last_used_cycle > i64::MAX as u64
        }) {
            return Err("OntologyRecord integer field exceeds SQLite INTEGER range".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Tx: {e}"))?;
        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT INTO knowledge_ontology
                     (name, vector_blob, usage_count, utility, created_at_cycle, last_used_cycle, is_a_parent)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                     ON CONFLICT(name) DO UPDATE SET
                        vector_blob = excluded.vector_blob,
                        usage_count = excluded.usage_count,
                        utility = excluded.utility,
                        created_at_cycle = excluded.created_at_cycle,
                        last_used_cycle = excluded.last_used_cycle,
                        is_a_parent = excluded.is_a_parent",
                )
                .map_err(|e| format!("Prepare: {e}"))?;

            for r in records {
                stmt.execute(rusqlite::params![
                    r.name,
                    r.vector_bytes,
                    i64::try_from(r.usage_count).expect("ontology usage count preflighted for SQLite INTEGER range"),
                    r.utility,
                    i64::try_from(r.created_at_cycle).expect("ontology creation cycle preflighted for SQLite INTEGER range"),
                    i64::try_from(r.last_used_cycle).expect("ontology last-used cycle preflighted for SQLite INTEGER range"),
                    r.is_a_parent,
                ])
                .map_err(|e| format!("Insert ontology: {e}"))?;
            }
        }
        tx.commit().map_err(|e| format!("Commit: {e}"))?;

        let count = records.len();
        self.total_saved += count as u64;
        Ok(count)
    }

    /// Load ontology primitives from the database.
    ///
    /// Returns all stored primitives ordered by utility descending.
    pub fn load_ontology(&mut self) -> Result<Vec<OntologyRecord>, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }
        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        let mut stmt = conn
            .prepare(
                "SELECT name, vector_blob, usage_count, utility, created_at_cycle, last_used_cycle, is_a_parent
                 FROM knowledge_ontology ORDER BY utility DESC, name ASC, created_at_cycle ASC, last_used_cycle ASC",
            )
            .map_err(|e| format!("Prepare: {e}"))?;

        let records: Vec<OntologyRecord> = stmt
            .query_map([], |row| {
                Ok(OntologyRecord {
                    name: row.get(0)?,
                    vector_bytes: row.get(1)?,
                    usage_count: u64::try_from(row.get::<_, i64>(2)?).map_err(|_| rusqlite::Error::InvalidColumnType(2, "usage_count".into(), rusqlite::types::Type::Integer))?,
                    utility: row.get(3)?,
                    created_at_cycle: u64::try_from(row.get::<_, i64>(4)?).map_err(|_| rusqlite::Error::InvalidColumnType(4, "created_at_cycle".into(), rusqlite::types::Type::Integer))?,
                    last_used_cycle: u64::try_from(row.get::<_, i64>(5)?).map_err(|_| rusqlite::Error::InvalidColumnType(5, "last_used_cycle".into(), rusqlite::types::Type::Integer))?,
                    is_a_parent: row.get(6)?,
                })
            })
            .map_err(|e| format!("Query: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Load ontology row: {e}"))?;

        self.total_loaded += records.len() as u64;
        Ok(records)
    }

    /// Total records saved across all calls.
    pub fn total_saved(&self) -> u64 {
        self.total_saved
    }

    /// Total records loaded across all calls.
    pub fn total_loaded(&self) -> u64 {
        self.total_loaded
    }

    // ── Internal ────────────────────────────────────────────────────────

    fn open_connection(&self) -> Result<rusqlite::Connection, String> {
        let conn =
            rusqlite::Connection::open(&self.db_path).map_err(|e| format!("SQLite open: {e}"))?;
        conn.busy_timeout(SQLITE_BUSY_TIMEOUT)
            .map_err(|e| format!("SQLite busy timeout: {e}"))?;
        conn.execute_batch("PRAGMA foreign_keys = ON;")
            .map_err(|e| format!("SQLite foreign keys: {e}"))?;
        Ok(conn)
    }

    fn ensure_schema(&mut self, conn: &rusqlite::Connection) -> Result<(), String> {
        if self.initialized {
            // `initialized` is process-local state. Another connection/process can still
            // mutate SQLite schema objects after this instance has initialized them, so do
            // not let the fast path mask loss of the invariants that enforce stable identity,
            // append-only receipts, and validation sequencing. Hold the same write lock used
            // by migration so the entire attestation observes one stable schema state.
            verify_schema_user_version(conn)?;
            let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)
                .map_err(|e| format!("Begin schema attestation transaction: {e}"))?;
            verify_schema_user_version(&conn)?;
            verify_initialized_schema_integrity(conn)?;
            tx.commit()
                .map_err(|e| format!("Commit schema attestation transaction: {e}"))?;
            return Ok(());
        }

        // Once a database has completed the current migration, the user-version marker makes
        // the schema state explicit across process restarts. Do not rerun repair-style migration
        // against a marked current database: verify it fail-closed instead.
        let user_version: i64 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(|e| format!("Schema user-version check: {e}"))?;
        if user_version == CURRENT_SCHEMA_USER_VERSION {
            let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)
                .map_err(|e| format!("Begin schema attestation transaction: {e}"))?;
            verify_schema_user_version(conn)?;
            verify_initialized_schema_integrity(conn)?;
            tx.commit()
                .map_err(|e| format!("Commit schema attestation transaction: {e}"))?;
            self.initialized = true;
            return Ok(());
        }
        if user_version != 0 {
            return Err(format!(
                "Unsupported knowledge SQLite schema user_version {}; expected 0 for legacy or {} for current",
                user_version, CURRENT_SCHEMA_USER_VERSION
            ));
        }

        // Serialize schema initialization across connections and keep the additive
        // migration atomic. Without a write transaction, two first-time openers can
        // race on ALTER TABLE / index creation and a failure midway can expose an
        // intermediate schema to another connection.
        // `ensure_schema` intentionally accepts a shared connection reference because
        // it is invoked by read-only callers as well as writers. `new_unchecked` gives
        // us the same IMMEDIATE behavior without requiring every call site to borrow the
        // connection mutably; rusqlite still rejects a nested transaction at runtime.
        let tx = rusqlite::Transaction::new_unchecked(conn, TransactionBehavior::Immediate)
            .map_err(|e| format!("Begin schema migration transaction: {e}"))?;

        // Re-read after acquiring the write lock. Another opener may have completed the
        // migration between the initial user_version check and BEGIN IMMEDIATE; in that case,
        // attest the now-current schema without rerunning repair-style migration.
        let transaction_user_version: i64 = tx
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .map_err(|e| format!("Schema transaction user-version check: {e}"))?;
        if transaction_user_version == CURRENT_SCHEMA_USER_VERSION {
            verify_initialized_schema_integrity(conn)?;
            tx.commit()
                .map_err(|e| format!("Commit schema verification transaction: {e}"))?;
            self.initialized = true;
            return Ok(());
        }
        if transaction_user_version != 0 {
            return Err(format!(
                "Unsupported knowledge SQLite schema user_version {}; expected 0 for legacy or {} for current",
                transaction_user_version, CURRENT_SCHEMA_USER_VERSION
            ));
        }

        tx.execute_batch(
            "CREATE TABLE IF NOT EXISTS knowledge_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                canonical_identity TEXT,
                provenance_family TEXT,
                vector_blob BLOB NOT NULL,
                source_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                domain TEXT,
                cycle INTEGER NOT NULL,
                is_causal INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS knowledge_provenance_relations (
                source_memory_id TEXT NOT NULL,
                target_memory_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_memory_id, target_memory_id, kind, created_at)
            );
            CREATE TABLE IF NOT EXISTS knowledge_causal_edges (
                cause TEXT NOT NULL,
                effect TEXT NOT NULL,
                strength REAL NOT NULL,
                is_inhibitory INTEGER NOT NULL DEFAULT 0,
                cycle INTEGER NOT NULL,
                PRIMARY KEY (cause, effect)
            );
            CREATE TABLE IF NOT EXISTS knowledge_ontology (
                name TEXT PRIMARY KEY,
                vector_blob BLOB NOT NULL,
                usage_count INTEGER NOT NULL,
                utility REAL NOT NULL,
                created_at_cycle INTEGER NOT NULL,
                last_used_cycle INTEGER NOT NULL,
                is_a_parent TEXT
            );
            CREATE TABLE IF NOT EXISTS knowledge_snapshot_receipts (
                generation INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_digest_hex TEXT NOT NULL,
                receipt_digest_hex TEXT NOT NULL CHECK (length(receipt_digest_hex) = 64)
            );
            CREATE TABLE IF NOT EXISTS knowledge_snapshot_validation_receipts (
                validation_event TEXT PRIMARY KEY,
                validation_sequence INTEGER NOT NULL UNIQUE CHECK (validation_sequence > 0),
                generation INTEGER NOT NULL,
                snapshot_digest_hex TEXT NOT NULL,
                validator_ref TEXT NOT NULL,
                validator_version TEXT NOT NULL,
                validation_profile TEXT NOT NULL,
                conforms INTEGER NOT NULL CHECK (conforms IN (0, 1)),
                report_digest_hex TEXT,
                receipt_digest_hex TEXT NOT NULL CHECK (length(receipt_digest_hex) = 64),
                FOREIGN KEY (generation) REFERENCES knowledge_snapshot_receipts(generation)
            );
            CREATE INDEX IF NOT EXISTS idx_snapshot_validation_receipts_generation
                ON knowledge_snapshot_validation_receipts(generation, validation_event);
            CREATE INDEX IF NOT EXISTS idx_facts_domain ON knowledge_facts(domain);
            CREATE INDEX IF NOT EXISTS idx_facts_cycle ON knowledge_facts(cycle);",
        )
        .map_err(|e| format!("Schema init: {e}"))?;

        // Backward-compatible migration for databases created before EPF-011.
        // SQLite UNIQUE indexes permit multiple NULLs, so legacy rows without a memory_id
        // remain compatible while stable memory_id becomes the idempotency key for new rows.
        let columns: Vec<String> = tx
            .prepare("PRAGMA table_info(knowledge_facts)")
            .map_err(|e| format!("Schema inspect: {e}"))?
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|e| format!("Schema inspect query: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Schema inspect row: {e}"))?;
        for (name, ty) in [("memory_id", "TEXT"), ("canonical_identity", "TEXT"), ("provenance_family", "TEXT")] {
            if !columns.iter().any(|c| c == name) {
                tx.execute(&format!("ALTER TABLE knowledge_facts ADD COLUMN {name} {ty}"), [])
                    .map_err(|e| format!("Schema migration {name}: {e}"))?;
            }
        }

        // Materialize deterministic identities as part of the atomic migration rather
        // than performing an autocommit write from individual read APIs. This makes
        // legacy identity normalization happen exactly once, behind the same migration
        // boundary as the uniqueness constraint.
        tx.execute(
            "UPDATE knowledge_facts
             SET memory_id = 'legacy-fact:' || id
             WHERE memory_id IS NULL",
            [],
        )
        .map_err(|e| format!("Schema legacy memory identity backfill: {e}"))?;

        let legacy_memory_ids = {
            let mut stmt = tx
                .prepare(
                    "SELECT rowid, memory_id
                     FROM knowledge_facts
                     WHERE memory_id IS NOT NULL",
                )
                .map_err(|e| format!("Schema legacy memory identity validation prepare: {e}"))?;
            stmt.query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| format!("Schema legacy memory identity validation query: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Schema legacy memory identity validation row: {e}"))?
        };
        for (rowid, memory_id) in legacy_memory_ids {
            if memory_id.trim().is_empty() {
                return Err(format!(
                    "Schema legacy memory identity validation failed: row {} has blank memory_id",
                    rowid
                ));
            }
        }

        // Canonicalize persisted provenance relation kinds to the stable shared wire codes.
        // The loader accepts the legacy Rust Debug spellings above for compatibility, but after
        // migration the database has one representation that is independent of Rust variant names.
        let legacy_kind_rows: Vec<(i64, String)> = {
            let mut stmt = tx
                .prepare(
                    "SELECT rowid, kind
                     FROM knowledge_provenance_relations
                     WHERE kind IN (
                         'DerivedFrom', 'RevisedFrom', 'Supersedes',
                         'Contradicts', 'Corroborates', 'RepresentationOf'
                     )
                     ORDER BY rowid ASC",
                )
                .map_err(|e| format!("Provenance kind normalization prepare: {e}"))?;
            stmt.query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
                .map_err(|e| format!("Provenance kind normalization query: {e}"))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Provenance kind normalization row: {e}"))?
        };
        for (rowid, legacy_kind) in legacy_kind_rows {
            let kind = provenance_kind_from_persisted(&legacy_kind)
                .ok_or_else(|| format!("Unknown persisted provenance kind: {legacy_kind}"))?;
            tx.execute(
                "UPDATE knowledge_provenance_relations
                 SET kind = ?1
                 WHERE rowid = ?2",
                rusqlite::params![provenance_kind_storage_code(kind), rowid],
            )
            .map_err(|e| format!("Provenance kind normalization update: {e}"))?;
        }

        // Validate the complete persisted relation-kind domain after normalization so an
        // out-of-band unknown kind cannot survive migration and merely surface later at load.
        let mut invalid_kind_stmt = tx
            .prepare(
                "SELECT rowid, kind
                 FROM knowledge_provenance_relations
                 WHERE kind NOT IN (
                     'derived_from', 'revised_from', 'supersedes',
                     'contradicts', 'corroborates', 'representation_of'
                 )
                 ORDER BY rowid ASC",
            )
            .map_err(|e| format!("Provenance kind integrity prepare: {e}"))?;
        let invalid_kind_rows = invalid_kind_stmt
            .query_map([], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)))
            .map_err(|e| format!("Provenance kind integrity query: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Provenance kind integrity row: {e}"))?;
        if let Some((rowid, kind)) = invalid_kind_rows.first() {
            return Err(format!(
                "Persisted provenance relation row {} has unknown kind: {}",
                rowid, kind
            ));
        }

        // Add snapshot-receipt self-digest support to databases created by the
        // earlier EPF-011 receipt tranche, then deterministically backfill legacy rows.
        let snapshot_receipt_columns: Vec<String> = tx
            .prepare("PRAGMA table_info(knowledge_snapshot_receipts)")
            .map_err(|e| format!("Snapshot receipt schema inspect: {e}"))?
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|e| format!("Snapshot receipt schema inspect query: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Schema inspect row: {e}"))?;
        if !snapshot_receipt_columns.iter().any(|c| c == "receipt_digest_hex") {
            tx.execute(
                "ALTER TABLE knowledge_snapshot_receipts
                 ADD COLUMN receipt_digest_hex TEXT",
                [],
            )
            .map_err(|e| format!("Snapshot receipt schema migration receipt_digest_hex: {e}"))?;
        }

        let legacy_snapshot_receipts = {
            let mut stmt = tx
                .prepare(
                    "SELECT rowid, generation, canonical_digest_hex
                     FROM knowledge_snapshot_receipts
                     WHERE receipt_digest_hex IS NULL
                     ORDER BY generation ASC",
                )
                .map_err(|e| format!("Snapshot receipt backfill prepare: {e}"))?;
            stmt.query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    KnowledgeSnapshotReceipt {
                        generation: u64::try_from(row.get::<_, i64>(1)?).map_err(|_| {
                            rusqlite::Error::InvalidColumnType(
                                1,
                                "generation".into(),
                                rusqlite::types::Type::Integer,
                            )
                        })?,
                        canonical_digest_hex: row.get(2)?,
                        receipt_digest_hex: String::new(),
                    },
                ))
            })
            .map_err(|e| format!("Snapshot receipt backfill query: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Snapshot receipt backfill row: {e}"))?
        };

        for (rowid, receipt) in legacy_snapshot_receipts {
            tx.execute(
                "UPDATE knowledge_snapshot_receipts
                 SET receipt_digest_hex = ?1
                 WHERE rowid = ?2",
                rusqlite::params![receipt.canonical_receipt_digest_hex(), rowid],
            )
            .map_err(|e| format!("Snapshot receipt backfill update: {e}"))?;
        }

        // Add validation append sequence to databases created before this hardening tranche.
        // Legacy rows are assigned deterministic sequence numbers in existing rowid order.
        let validation_sequence_columns: Vec<String> = tx
            .prepare("PRAGMA table_info(knowledge_snapshot_validation_receipts)")
            .map_err(|e| format!("Validation sequence schema inspect: {e}"))?
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|e| format!("Validation sequence schema inspect query: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Schema inspect row: {e}"))?;
        if !validation_sequence_columns.iter().any(|c| c == "validation_sequence") {
            tx.execute(
                "ALTER TABLE knowledge_snapshot_validation_receipts
                 ADD COLUMN validation_sequence INTEGER",
                [],
            )
            .map_err(|e| format!("Validation schema migration validation_sequence: {e}"))?;
        }

        let starting_sequence = tx
            .query_row(
                "SELECT COALESCE(MAX(validation_sequence), 0)
                 FROM knowledge_snapshot_validation_receipts",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map_err(|e| format!("Load validation sequence watermark: {e}"))?;
        if starting_sequence < 0 {
            return Err("Validation sequence contains a negative value".into());
        }

        let legacy_validation_sequence_rows = {
            let mut stmt = tx
                .prepare(
                    "SELECT rowid
                     FROM knowledge_snapshot_validation_receipts
                     WHERE validation_sequence IS NULL
                     ORDER BY rowid ASC",
                )
                .map_err(|e| format!("Validation sequence backfill prepare: {e}"))?;
            stmt.query_map([], |row| row.get::<_, i64>(0))
                .map_err(|e| format!("Validation sequence backfill query: {e}"))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Validation sequence backfill row: {e}"))?
        };

        let mut next_sequence = u64::try_from(starting_sequence)
            .map_err(|_| "Validation sequence exceeds supported range")?;
        for rowid in legacy_validation_sequence_rows {
            next_sequence = next_sequence
                .checked_add(1)
                .ok_or("Validation sequence exhausted SQLite INTEGER range")?;
            if next_sequence > i64::MAX as u64 {
                return Err("Validation sequence exceeds SQLite INTEGER range".into());
            }
            tx.execute(
                "UPDATE knowledge_snapshot_validation_receipts
                 SET validation_sequence = ?1
                 WHERE rowid = ?2",
                rusqlite::params![i64::try_from(next_sequence).expect("sequence preflighted"), rowid],
            )
            .map_err(|e| format!("Validation sequence backfill update: {e}"))?;
        }

        // Add validation-receipt self-digest support to databases created by the
        // earlier validation-ledger tranche, then deterministically backfill legacy rows.
        let validation_columns: Vec<String> = tx
            .prepare("PRAGMA table_info(knowledge_snapshot_validation_receipts)")
            .map_err(|e| format!("Validation schema inspect: {e}"))?
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|e| format!("Validation schema inspect query: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Schema inspect row: {e}"))?;
        if !validation_columns.iter().any(|c| c == "receipt_digest_hex") {
            tx.execute(
                "ALTER TABLE knowledge_snapshot_validation_receipts
                 ADD COLUMN receipt_digest_hex TEXT",
                [],
            )
            .map_err(|e| format!("Validation schema migration receipt_digest_hex: {e}"))?;
        }

        // Existing databases may already have the append-only validation trigger from
        // the immediately preceding tranche. Drop that trigger inside this same atomic
        // migration transaction so a legacy v1 self-digest can be upgraded; if any later
        // migration step fails, the transaction rollback restores the trigger unchanged.
        tx.execute_batch(
            "DROP TRIGGER IF EXISTS trg_knowledge_snapshot_receipts_required_insert;
             DROP TRIGGER IF EXISTS trg_knowledge_snapshot_validation_receipts_required_insert;
             DROP TRIGGER IF EXISTS trg_knowledge_snapshot_validation_receipts_no_update;",
        )
        .map_err(|e| format!("Schema validation receipt migration guard: {e}"))?;

        // Normalize every persisted validation receipt to the current sequence-bound
        // v2 self-digest. Rows from the immediately preceding tranche use v1; rows
        // without a digest are deterministically initialized from their existing fields.
        // Any present digest that matches neither exact representation fails the migration
        // closed instead of silently repairing potentially corrupted history.
        let validation_receipt_rows = {
            let mut stmt = tx
                .prepare(
                    "SELECT rowid, validation_sequence, validation_event, generation,
                            snapshot_digest_hex, validator_ref, validator_version,
                            validation_profile, conforms, report_digest_hex,
                            receipt_digest_hex
                     FROM knowledge_snapshot_validation_receipts
                     ORDER BY rowid ASC",
                )
                .map_err(|e| format!("Validation receipt integrity normalization prepare: {e}"))?;
            stmt.query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    u64::try_from(row.get::<_, i64>(1)?).map_err(|_| {
                        rusqlite::Error::InvalidColumnType(
                            1,
                            "validation_sequence".into(),
                            rusqlite::types::Type::Integer,
                        )
                    })?,
                    KnowledgeSnapshotValidationReceipt {
                        validation_event: row.get(2)?,
                        generation: u64::try_from(row.get::<_, i64>(3)?).map_err(|_| {
                            rusqlite::Error::InvalidColumnType(
                                3,
                                "generation".into(),
                                rusqlite::types::Type::Integer,
                            )
                        })?,
                        snapshot_digest_hex: row.get(4)?,
                        validator_ref: row.get(5)?,
                        validator_version: row.get(6)?,
                        validation_profile: row.get(7)?,
                        conforms: row.get(8)?,
                        report_digest_hex: row.get(9)?,
                    },
                    row.get::<_, Option<String>>(10)?,
                ))
            })
            .map_err(|e| format!("Validation receipt integrity normalization query: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Validation receipt integrity normalization row: {e}"))?
        };

        for (rowid, validation_sequence, receipt, stored_digest) in validation_receipt_rows {
            receipt
                .validate_input()
                .map_err(|e| format!("Invalid persisted snapshot validation receipt: {e}"))?;

            if validation_sequence == 0 {
                return Err("Validation receipt sequence must be positive".into());
            }

            let v2_digest = receipt.canonical_digest_hex_for_sequence(validation_sequence);
            let needs_v2_backfill = match stored_digest.as_deref() {
                None => true,
                Some(digest) if digest == v2_digest => false,
                Some(digest) if digest == receipt.legacy_canonical_digest_hex() => true,
                Some(_) => {
                    return Err(format!(
                        "Validation receipt self-digest is not a recognized EPF-011 integrity digest: {}",
                        receipt.validation_event
                    ));
                }
            };

            if needs_v2_backfill {
                tx.execute(
                    "UPDATE knowledge_snapshot_validation_receipts
                     SET receipt_digest_hex = ?1
                     WHERE rowid = ?2",
                    rusqlite::params![v2_digest, rowid],
                )
                .map_err(|e| format!("Validation receipt integrity normalization update: {e}"))?;
            }
        }

        // Create the identity index only after the additive columns exist on legacy databases.
        tx.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_memory_id_unique ON knowledge_facts(memory_id)",
            [],
        )
        .map_err(|e| format!("Schema identity index: {e}"))?;

        tx.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_validation_receipts_sequence_unique
             ON knowledge_snapshot_validation_receipts(validation_sequence)",
            [],
        )
        .map_err(|e| format!("Schema validation sequence index: {e}"))?;

        // Existing legacy databases may already contain orphaned validation rows because
        // SQLite foreign-key enforcement is connection-local. Fail the migration closed rather
        // than committing a repaired schema around an invalid historical ledger.
        let mut foreign_key_check = tx
            .prepare("PRAGMA foreign_key_check")
            .map_err(|e| format!("Schema foreign-key check prepare: {e}"))?;
        let foreign_key_violations = foreign_key_check
            .query_map([], |_| Ok(()))
            .map_err(|e| format!("Schema foreign-key check query: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Schema foreign-key check row: {e}"))?;
        if !foreign_key_violations.is_empty() {
            return Err(format!(
                "Schema foreign-key check failed: {} violation(s)",
                foreign_key_violations.len()
            ));
        }

        // Before committing the migrated schema, require the historical ledgers to satisfy
        // the same invariants used by runtime verification. This prevents malformed legacy
        // metadata from being converted into a merely self-consistent but structurally invalid
        // provenance history.
        verify_snapshot_receipts_in_tx(&tx)?;
        verify_snapshot_validation_receipts_in_tx(&tx)?;

        // Enforce stable fact identity at the SQLite boundary. Because the migration above
        // materializes every legacy NULL identity before committing, newly inserted or updated
        // fact rows must always carry a non-blank identity.
        tx.execute_batch(
            "DROP TRIGGER IF EXISTS trg_knowledge_facts_memory_id_no_blank_insert;
             DROP TRIGGER IF EXISTS trg_knowledge_facts_memory_id_no_blank_update;
             CREATE TRIGGER IF NOT EXISTS trg_knowledge_facts_memory_id_required_insert
             BEFORE INSERT ON knowledge_facts
             WHEN NEW.memory_id IS NULL OR trim(NEW.memory_id) = ''
             BEGIN
                 SELECT RAISE(ABORT, 'knowledge_facts memory_id must be non-empty');
             END;
             CREATE TRIGGER IF NOT EXISTS trg_knowledge_facts_memory_id_required_update
             BEFORE UPDATE OF memory_id ON knowledge_facts
             WHEN NEW.memory_id IS NULL OR trim(NEW.memory_id) = ''
             BEGIN
                 SELECT RAISE(ABORT, 'knowledge_facts memory_id must be non-empty');
             END;
             CREATE TRIGGER IF NOT EXISTS trg_knowledge_provenance_relation_required_insert
             BEFORE INSERT ON knowledge_provenance_relations
             WHEN NEW.source_memory_id IS NULL
                  OR trim(NEW.source_memory_id) = ''
                  OR NEW.target_memory_id IS NULL
                  OR trim(NEW.target_memory_id) = ''
                  OR NEW.source_memory_id = NEW.target_memory_id
                  OR NEW.created_at IS NULL
                  OR trim(NEW.created_at) = ''
                  OR NEW.kind IS NULL
                  OR NEW.kind NOT IN (
                      'derived_from', 'revised_from', 'supersedes',
                      'contradicts', 'corroborates', 'representation_of'
                  )
             BEGIN
                 SELECT RAISE(ABORT, 'knowledge_provenance_relations requires valid identities, timestamp, and stable kind');
             END;
             CREATE TRIGGER IF NOT EXISTS trg_knowledge_provenance_relation_required_update
             BEFORE UPDATE OF source_memory_id, target_memory_id, kind, created_at ON knowledge_provenance_relations
             WHEN NEW.source_memory_id IS NULL
                  OR trim(NEW.source_memory_id) = ''
                  OR NEW.target_memory_id IS NULL
                  OR trim(NEW.target_memory_id) = ''
                  OR NEW.source_memory_id = NEW.target_memory_id
                  OR NEW.created_at IS NULL
                  OR trim(NEW.created_at) = ''
                  OR NEW.kind IS NULL
                  OR NEW.kind NOT IN (
                      'derived_from', 'revised_from', 'supersedes',
                      'contradicts', 'corroborates', 'representation_of'
                  )
             BEGIN
                 SELECT RAISE(ABORT, 'knowledge_provenance_relations requires valid identities, timestamp, and stable kind');
             END;
             CREATE TRIGGER IF NOT EXISTS trg_knowledge_snapshot_receipts_required_insert
             BEFORE INSERT ON knowledge_snapshot_receipts
             WHEN NEW.generation IS NULL
                  OR NEW.generation <= 0
                  OR NEW.canonical_digest_hex IS NULL
                  OR length(NEW.canonical_digest_hex) <> 64
                  OR NEW.canonical_digest_hex GLOB '*[^0-9A-Fa-f]*'
                  OR NEW.receipt_digest_hex IS NULL
                  OR length(NEW.receipt_digest_hex) <> 64
                  OR NEW.receipt_digest_hex GLOB '*[^0-9A-Fa-f]*'
                  OR trim(NEW.canonical_digest_hex) = ''
                  OR trim(NEW.receipt_digest_hex) = ''
             BEGIN
                 SELECT RAISE(ABORT, 'knowledge_snapshot_receipts requires positive generation and 64-character digests');
             END;
             CREATE TRIGGER IF NOT EXISTS trg_knowledge_snapshot_validation_receipts_required_insert
             BEFORE INSERT ON knowledge_snapshot_validation_receipts
             WHEN NEW.validation_event IS NULL
                  OR trim(NEW.validation_event) = ''
                  OR NEW.validation_sequence IS NULL
                  OR NEW.validation_sequence <= 0
                  OR NEW.snapshot_digest_hex IS NULL
                  OR length(NEW.snapshot_digest_hex) <> 64
                  OR NEW.snapshot_digest_hex GLOB '*[^0-9A-Fa-f]*'
                  OR trim(NEW.snapshot_digest_hex) = ''
                  OR NEW.validator_ref IS NULL
                  OR trim(NEW.validator_ref) = ''
                  OR NEW.validator_version IS NULL
                  OR trim(NEW.validator_version) = ''
                  OR NEW.validation_profile IS NULL
                  OR trim(NEW.validation_profile) = ''
                  OR NEW.conforms IS NULL
                  OR NEW.conforms NOT IN (0, 1)
                  OR (NEW.report_digest_hex IS NOT NULL
                      AND (length(NEW.report_digest_hex) <> 64
                           OR NEW.report_digest_hex GLOB '*[^0-9A-Fa-f]*'))
                  OR NEW.receipt_digest_hex IS NULL
                  OR length(NEW.receipt_digest_hex) <> 64
                  OR NEW.receipt_digest_hex GLOB '*[^0-9A-Fa-f]*'
                  OR trim(NEW.receipt_digest_hex) = ''
             BEGIN
                 SELECT RAISE(ABORT, 'knowledge_snapshot_validation_receipts requires valid identity, digest, validator, outcome, and positive sequence');
             END;
             CREATE TRIGGER IF NOT EXISTS trg_knowledge_snapshot_receipts_no_update
             BEFORE UPDATE ON knowledge_snapshot_receipts
             BEGIN
                 SELECT RAISE(ABORT, 'knowledge_snapshot_receipts is append-only: UPDATE prohibited');
             END;
             CREATE TRIGGER IF NOT EXISTS trg_knowledge_snapshot_receipts_no_delete
             BEFORE DELETE ON knowledge_snapshot_receipts
             BEGIN
                 SELECT RAISE(ABORT, 'knowledge_snapshot_receipts is append-only: DELETE prohibited');
             END;
             CREATE TRIGGER IF NOT EXISTS trg_knowledge_snapshot_validation_receipts_no_update
             BEFORE UPDATE ON knowledge_snapshot_validation_receipts
             BEGIN
                 SELECT RAISE(ABORT, 'knowledge_snapshot_validation_receipts is append-only: UPDATE prohibited');
             END;
             CREATE TRIGGER IF NOT EXISTS trg_knowledge_snapshot_validation_receipts_no_delete
             BEFORE DELETE ON knowledge_snapshot_validation_receipts
             BEGIN
                 SELECT RAISE(ABORT, 'knowledge_snapshot_validation_receipts is append-only: DELETE prohibited');
             END;",
        )
        .map_err(|e| format!("Schema receipt immutability triggers: {e}"))?;

        tx.execute_batch("PRAGMA user_version = 1;")
            .map_err(|e| format!("Schema user-version migration: {e}"))?;

        // Validate the fully materialized schema while it is still inside the
        // migration transaction. This prevents a partially upgraded legacy database
        // from becoming visible as "initialized" before the same runtime attestation
        // that protects later fast-path opens has passed.
        verify_initialized_schema_integrity(conn)?;

        tx.commit()
            .map_err(|e| format!("Commit schema migration transaction: {e}"))?;
        self.initialized = true;
        Ok(())
    }
}

fn verify_validation_receipt_foreign_key_contract(
    conn: &rusqlite::Connection,
) -> Result<(), String> {
    let foreign_keys = {
        let mut stmt = conn
            .prepare("PRAGMA foreign_key_list(knowledge_snapshot_validation_receipts)")
            .map_err(|e| format!("Schema integrity foreign-key prepare: {e}"))?;
        stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, String>(7)?,
            ))
        })
        .map_err(|e| format!("Schema integrity foreign-key query: {e}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Schema integrity foreign-key row: {e}"))?
    };

    if foreign_keys.len() != 1
        || !foreign_keys.iter().all(
            |(parent, child_col, parent_col, on_update, on_delete, match_kind)| {
                parent == "knowledge_snapshot_receipts"
                    && child_col == "generation"
                    && parent_col == "generation"
                    && on_update.eq_ignore_ascii_case("NO ACTION")
                    && on_delete.eq_ignore_ascii_case("NO ACTION")
                    && match_kind.eq_ignore_ascii_case("NONE")
            },
        )
    {
        return Err(
            "Schema integrity check failed: validation receipt generation foreign key has the wrong action or match semantics"
                .into(),
        );
    }

    Ok(())
}

fn verify_schema_user_version(conn: &rusqlite::Connection) -> Result<(), String> {
    let user_version: i64 = conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .map_err(|e| format!("Schema user-version verification: {e}"))?;
    if user_version != CURRENT_SCHEMA_USER_VERSION {
        return Err(format!(
            "Schema integrity check failed: unsupported knowledge SQLite user_version {}; expected {}",
            user_version, CURRENT_SCHEMA_USER_VERSION
        ));
    }
    Ok(())
}

fn verify_table_column_contract(
    conn: &rusqlite::Connection,
    table: &str,
    expected: &[(&str, &str, i64)],
) -> Result<(), String> {
    let pragma = format!("PRAGMA table_info({table})");
    let mut stmt = conn
        .prepare(&pragma)
        .map_err(|e| format!("Schema integrity column check for {table}: {e}"))?;
    let actual = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(5)?,
            ))
        })
        .map_err(|e| format!("Schema integrity column query for {table}: {e}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Schema integrity column row for {table}: {e}"))?;

    for (column, expected_type, expected_pk) in expected {
        let (_, actual_type, actual_pk) = actual
            .iter()
            .find(|(name, _, _)| name == column)
            .ok_or_else(|| {
                format!("Schema integrity check failed: missing column {column} on {table}")
            })?;

        if !actual_type.eq_ignore_ascii_case(expected_type) {
            return Err(format!(
                "Schema integrity check failed: column {column} on {table} has declared type {actual_type}, expected {expected_type}"
            ));
        }
        if *actual_pk != *expected_pk {
            return Err(format!(
                "Schema integrity check failed: column {column} on {table} has primary-key position {actual_pk}, expected {expected_pk}"
            ));
        }
    }

    Ok(())
}

fn verify_initialized_schema_integrity(conn: &rusqlite::Connection) -> Result<(), String> {
    const REQUIRED_TABLES: &[&str] = &[
        "knowledge_facts",
        "knowledge_provenance_relations",
        "knowledge_causal_edges",
        "knowledge_ontology",
        "knowledge_snapshot_receipts",
        "knowledge_snapshot_validation_receipts",
    ];
    const REQUIRED_TRIGGERS: &[&str] = &[
        "trg_knowledge_facts_memory_id_required_insert",
        "trg_knowledge_facts_memory_id_required_update",
        "trg_knowledge_provenance_relation_required_insert",
        "trg_knowledge_provenance_relation_required_update",
        "trg_knowledge_snapshot_receipts_required_insert",
        "trg_knowledge_snapshot_validation_receipts_required_insert",
        "trg_knowledge_snapshot_receipts_no_update",
        "trg_knowledge_snapshot_receipts_no_delete",
        "trg_knowledge_snapshot_validation_receipts_no_update",
        "trg_knowledge_snapshot_validation_receipts_no_delete",
    ];

    for name in REQUIRED_TABLES {
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1
                )",
                [*name],
                |row| row.get(0),
            )
            .map_err(|e| format!("Schema integrity table check for {name}: {e}"))?;
        if !exists {
            return Err(format!("Schema integrity check failed: missing table {name}"));
        }
    }

    verify_table_column_contract(
        conn,
        "knowledge_provenance_relations",
        &[
            ("source_memory_id", "TEXT", 1i64),
            ("target_memory_id", "TEXT", 2),
            ("kind", "TEXT", 3),
            ("created_at", "TEXT", 4),
        ],
    )?;
    verify_table_column_contract(
        conn,
        "knowledge_facts",
        &[
            ("id", "INTEGER", 1i64),
            ("memory_id", "TEXT", 0),
            ("canonical_identity", "TEXT", 0),
            ("provenance_family", "TEXT", 0),
            ("vector_blob", "BLOB", 0),
            ("source_text", "TEXT", 0),
            ("confidence", "REAL", 0),
            ("domain", "TEXT", 0),
            ("cycle", "INTEGER", 0),
            ("is_causal", "INTEGER", 0),
        ],
    )?;
    verify_table_column_contract(
        conn,
        "knowledge_causal_edges",
        &[
            ("cause", "TEXT", 1i64),
            ("effect", "TEXT", 2),
            ("strength", "REAL", 0),
            ("is_inhibitory", "INTEGER", 0),
            ("cycle", "INTEGER", 0),
        ],
    )?;
    verify_table_column_contract(
        conn,
        "knowledge_ontology",
        &[
            ("name", "TEXT", 1i64),
            ("vector_blob", "BLOB", 0),
            ("usage_count", "INTEGER", 0),
            ("utility", "REAL", 0),
            ("created_at_cycle", "INTEGER", 0),
            ("last_used_cycle", "INTEGER", 0),
            ("is_a_parent", "TEXT", 0),
        ],
    )?;
    verify_table_column_contract(
        conn,
        "knowledge_snapshot_receipts",
        &[
            ("generation", "INTEGER", 1i64),
            ("canonical_digest_hex", "TEXT", 0),
            ("receipt_digest_hex", "TEXT", 0),
        ],
    )?;
    verify_table_column_contract(
        conn,
        "knowledge_snapshot_validation_receipts",
        &[
            ("validation_event", "TEXT", 1i64),
            ("validation_sequence", "INTEGER", 0),
            ("generation", "INTEGER", 0),
            ("snapshot_digest_hex", "TEXT", 0),
            ("validator_ref", "TEXT", 0),
            ("validator_version", "TEXT", 0),
            ("validation_profile", "TEXT", 0),
            ("conforms", "INTEGER", 0),
            ("report_digest_hex", "TEXT", 0),
            ("receipt_digest_hex", "TEXT", 0),
        ],
    )?;

    for name in REQUIRED_TRIGGERS {
        let sql: Option<String> = conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?1",
                [*name],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| format!("Schema integrity trigger check for {name}: {e}"))?;
        let sql = sql.ok_or_else(|| {
            format!("Schema integrity check failed: missing trigger {name}")
        })?;
        let normalized = sql.split_whitespace().collect::<Vec<_>>().join(" ").to_ascii_lowercase();

        let required_fragments: &[&str] = match *name {
            "trg_knowledge_facts_memory_id_required_insert" => &[
                "before insert on knowledge_facts",
                "new.memory_id is null or trim(new.memory_id) = ''",
                "raise(abort, 'knowledge_facts memory_id must be non-empty')",
            ],
            "trg_knowledge_facts_memory_id_required_update" => &[
                "before update of memory_id on knowledge_facts",
                "new.memory_id is null or trim(new.memory_id) = ''",
                "raise(abort, 'knowledge_facts memory_id must be non-empty')",
            ],
            "trg_knowledge_provenance_relation_required_insert" => &[
                "before insert on knowledge_provenance_relations",
                "new.source_memory_id is null",
                "new.target_memory_id is null",
                "new.source_memory_id = new.target_memory_id",
                "new.created_at is null",
                "new.kind is null",
                "new.kind not in (",
                "raise(abort, 'knowledge_provenance_relations requires valid identities, timestamp, and stable kind')",
            ],
            "trg_knowledge_provenance_relation_required_update" => &[
                "before update of source_memory_id, target_memory_id, kind, created_at on knowledge_provenance_relations",
                "new.source_memory_id is null",
                "new.target_memory_id is null",
                "new.source_memory_id = new.target_memory_id",
                "new.created_at is null",
                "new.kind is null",
                "new.kind not in (",
                "raise(abort, 'knowledge_provenance_relations requires valid identities, timestamp, and stable kind')",
            ],
            "trg_knowledge_snapshot_receipts_required_insert" => &[
                "before insert on knowledge_snapshot_receipts",
                "new.generation is null",
                "new.generation <= 0",
                "new.canonical_digest_hex is null",
                "length(new.canonical_digest_hex) <> 64",
                "new.receipt_digest_hex is null",
                "length(new.receipt_digest_hex) <> 64",
                "raise(abort, 'knowledge_snapshot_receipts requires positive generation and 64-character digests')",
            ],
            "trg_knowledge_snapshot_validation_receipts_required_insert" => &[
                "before insert on knowledge_snapshot_validation_receipts",
                "new.validation_event is null",
                "new.validation_sequence is null",
                "new.validation_sequence <= 0",
                "new.snapshot_digest_hex is null",
                "length(new.snapshot_digest_hex) <> 64",
                "new.validator_ref is null",
                "new.validator_version is null",
                "new.validation_profile is null",
                "new.conforms is null",
                "new.conforms not in (0, 1)",
                "new.report_digest_hex is not null",
                "length(new.report_digest_hex) <> 64",
                "new.report_digest_hex glob '*[^0-9a-fa-f]*'",
                "new.receipt_digest_hex is null",
                "length(new.receipt_digest_hex) <> 64",
                "raise(abort, 'knowledge_snapshot_validation_receipts requires valid identity, digest, validator, outcome, and positive sequence')",
            ],
            "trg_knowledge_snapshot_receipts_no_update" => &[
                "before update on knowledge_snapshot_receipts",
                "raise(abort, 'knowledge_snapshot_receipts is append-only: update prohibited')",
            ],
            "trg_knowledge_snapshot_receipts_no_delete" => &[
                "before delete on knowledge_snapshot_receipts",
                "raise(abort, 'knowledge_snapshot_receipts is append-only: delete prohibited')",
            ],
            "trg_knowledge_snapshot_validation_receipts_no_update" => &[
                "before update on knowledge_snapshot_validation_receipts",
                "raise(abort, 'knowledge_snapshot_validation_receipts is append-only: update prohibited')",
            ],
            "trg_knowledge_snapshot_validation_receipts_no_delete" => &[
                "before delete on knowledge_snapshot_validation_receipts",
                "raise(abort, 'knowledge_snapshot_validation_receipts is append-only: delete prohibited')",
            ],
            _ => &[],
        };

        for fragment in required_fragments {
            if !normalized.contains(fragment) {
                return Err(format!(
                    "Schema integrity check failed: trigger {name} is missing required contract fragment: {fragment}"
                ));
            }
        }
    }

    let foreign_keys_enabled: i64 = conn
        .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
        .map_err(|e| format!("Schema integrity foreign-key enforcement check: {e}"))?;
    if foreign_keys_enabled != 1 {
        return Err(
            "Schema integrity check failed: foreign-key enforcement is disabled on this connection"
                .into(),
        );
    }

    // Verify the declared FK itself, not only current row consistency. foreign_key_check
    // cannot prove a REFERENCES clause exists when the schema has drifted.
    verify_validation_receipt_foreign_key_contract(conn)?;

    let foreign_key_violations = {
        let mut stmt = conn
            .prepare("PRAGMA foreign_key_check(knowledge_snapshot_validation_receipts)")
            .map_err(|e| format!("Schema integrity foreign-key check prepare: {e}"))?;
        stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })
        .map_err(|e| format!("Schema integrity foreign-key check query: {e}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Schema integrity foreign-key check row: {e}"))?
    };
    if let Some((table, rowid, parent, constraint)) = foreign_key_violations.first() {
        return Err(format!(
            "Schema integrity check failed: foreign-key violation in {table} row {rowid} referencing {parent} (constraint {constraint})"
        ));
    }

    // These two indexes enforce identity uniqueness rather than merely improving
    // query performance, so verify both uniqueness and the exact key-column contract.
    for (table, index, expected_columns) in [
        ("knowledge_facts", "idx_facts_memory_id_unique", &["memory_id"][..]),
        (
            "knowledge_snapshot_validation_receipts",
            "idx_snapshot_validation_receipts_sequence_unique",
            &["validation_sequence"][..],
        ),
    ] {
        let pragma = format!("PRAGMA index_list({table})");
        let mut stmt = conn
            .prepare(&pragma)
            .map_err(|e| format!("Schema integrity index check for {table}: {e}"))?;
        let mut rows = stmt
            .query([])
            .map_err(|e| format!("Schema integrity index query for {table}: {e}"))?;
        let mut found_unique = false;
        while let Some(row) = rows
            .next()
            .map_err(|e| format!("Schema integrity index row for {table}: {e}"))?
        {
            let name: String = row
                .get(1)
                .map_err(|e| format!("Schema integrity index name for {table}: {e}"))?;
            let unique: i64 = row
                .get(2)
                .map_err(|e| format!("Schema integrity index uniqueness for {table}: {e}"))?;
            let partial: i64 = row
                .get(4)
                .map_err(|e| format!("Schema integrity index partial flag for {table}: {e}"))?;
            if name == index && unique != 0 && partial == 0 {
                let info_pragma = format!("PRAGMA index_info({index})");
                let mut info_stmt = conn
                    .prepare(&info_pragma)
                    .map_err(|e| format!("Schema integrity index columns for {index}: {e}"))?;
                let index_columns = info_stmt
                    .query_map([], |info_row| info_row.get::<_, Option<String>>(2))
                    .map_err(|e| format!("Schema integrity index column query for {index}: {e}"))?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("Schema integrity index column row for {index}: {e}"))?;
                if index_columns.len() == expected_columns.len()
                    && index_columns
                        .iter()
                        .map(Option::as_deref)
                        .eq(expected_columns.iter().copied().map(Some))
                {
                    let info_pragma = format!("PRAGMA index_xinfo({index})");
                    let mut xinfo_stmt = conn
                        .prepare(&info_pragma)
                        .map_err(|e| {
                            format!("Schema integrity index extended columns for {index}: {e}")
                        })?;
                    let key_columns = xinfo_stmt
                        .query_map([], |info_row| {
                            Ok((
                                info_row.get::<_, Option<String>>(2)?,
                                info_row.get::<_, i64>(3)?,
                                info_row.get::<_, String>(4)?,
                                info_row.get::<_, i64>(5)?,
                            ))
                        })
                        .map_err(|e| {
                            format!("Schema integrity index extended column query for {index}: {e}")
                        })?
                        .collect::<Result<Vec<_>, _>>()
                        .map_err(|e| {
                            format!("Schema integrity index extended column row for {index}: {e}")
                        })?;

                    let semantic_key_columns: Vec<_> = key_columns
                        .into_iter()
                        .filter(|(_, _, _, is_key)| *is_key != 0)
                        .collect();

                    let exact_key_contract = semantic_key_columns.len() == expected_columns.len()
                        && semantic_key_columns.iter().enumerate().all(
                            |(position, (name, desc, collation, is_key))| {
                                *is_key != 0
                                    && *desc == 0
                                    && collation.eq_ignore_ascii_case("BINARY")
                                    && name.as_deref() == Some(expected_columns[position])
                            },
                        );
                    if exact_key_contract {
                        found_unique = true;
                    }
                }
                break;
            }
        }
        if !found_unique {
            return Err(format!(
                "Schema integrity check failed: unique index {index} on {table} has the wrong definition"
            ));
        }
    }

    verify_persistence_trigger_behavior(conn)?;

    Ok(())
}

fn verify_persistence_trigger_behavior(conn: &rusqlite::Connection) -> Result<(), String> {
    conn.execute_batch("SAVEPOINT epf011_trigger_attestation;")
        .map_err(|e| format!("Schema integrity trigger probe savepoint: {e}"))?;

    let result = (|| {
        let fact_probe_rowid: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(rowid), 0) FROM knowledge_facts",
                [],
                |row| row.get(0),
            )
            .map_err(|e| format!("Schema integrity trigger probe fact rowid: {e}"))?
            .checked_add(1)
            .ok_or("Schema integrity trigger probe fact rowid exhausted SQLite INTEGER range")?;
        let fact_memory_id = format!("__epf011_trigger_probe_fact_{fact_probe_rowid}");
        conn.execute(
            "INSERT INTO knowledge_facts
             (memory_id, vector_blob, source_text, confidence, cycle, is_causal)
             VALUES (?1, ?2, 'trigger probe', 0.5, 1, 0)",
            rusqlite::params![fact_memory_id, vec![0x2Au8; BinaryHV::BYTES]],
        )
        .map_err(|e| format!("Schema integrity trigger probe fact insert: {e}"))?;

        if conn
            .execute(
                "INSERT INTO knowledge_facts
                 (memory_id, vector_blob, source_text, confidence, cycle, is_causal)
                 VALUES ('   ', ?1, 'trigger probe malformed', 0.5, 1, 0)",
                [vec![0x2Bu8; BinaryHV::BYTES]],
            )
            .is_ok()
        {
            return Err(
                "Schema integrity check failed: fact identity trigger did not reject blank insert"
                    .into(),
            );
        }
        if conn
            .execute(
                "UPDATE knowledge_facts
                 SET memory_id = ' '
                 WHERE memory_id = ?1",
                [fact_memory_id],
            )
            .is_ok()
        {
            return Err(
                "Schema integrity check failed: fact identity trigger did not reject blank update"
                    .into(),
            );
        }

        let provenance_probe_rowid: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(rowid), 0) FROM knowledge_provenance_relations",
                [],
                |row| row.get(0),
            )
            .map_err(|e| format!("Schema integrity trigger probe provenance rowid: {e}"))?
            .checked_add(1)
            .ok_or("Schema integrity trigger probe provenance rowid exhausted SQLite INTEGER range")?;
        let provenance_source =
            format!("__epf011_trigger_probe_source_{provenance_probe_rowid}");
        let provenance_target =
            format!("__epf011_trigger_probe_target_{provenance_probe_rowid}");
        let provenance_event = format!("event:trigger-probe-{provenance_probe_rowid}");

        conn.execute(
            "INSERT INTO knowledge_provenance_relations
             (source_memory_id, target_memory_id, kind, created_at)
             VALUES (?1, ?2, 'derived_from', ?3)",
            rusqlite::params![provenance_source, provenance_target, provenance_event],
        )
        .map_err(|e| format!("Schema integrity trigger probe provenance insert: {e}"))?;

        if conn
            .execute(
                "INSERT INTO knowledge_provenance_relations
                 (source_memory_id, target_memory_id, kind, created_at)
                 VALUES (?1, ?2, 'unknown_kind', ?3)",
                rusqlite::params![
                    provenance_source,
                    provenance_target,
                    format!("event:trigger-probe-bad-{provenance_probe_rowid}"),
                ],
            )
            .is_ok()
        {
            return Err(
                "Schema integrity check failed: provenance trigger did not reject unknown kind"
                    .into(),
            );
        }
        if conn
            .execute(
                "INSERT INTO knowledge_provenance_relations
                 (source_memory_id, target_memory_id, kind, created_at)
                 VALUES (?1, ?2, NULL, ?3)",
                rusqlite::params![
                    provenance_source,
                    provenance_target,
                    format!("event:trigger-probe-null-{provenance_probe_rowid}"),
                ],
            )
            .is_ok()
        {
            return Err(
                "Schema integrity check failed: provenance trigger did not reject NULL kind"
                    .into(),
            );
        }

        if conn
            .execute(
                "UPDATE knowledge_provenance_relations
                 SET source_memory_id = ' '
                 WHERE source_memory_id = ?1
                   AND target_memory_id = ?2
                   AND kind = 'derived_from'
                   AND created_at = ?3",
                rusqlite::params![provenance_source, provenance_target, provenance_event],
            )
            .is_ok()
        {
            return Err(
                "Schema integrity check failed: provenance trigger did not reject blank update"
                    .into(),
            );
        }

        let snapshot_generation: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(generation), 0) FROM knowledge_snapshot_receipts",
                [],
                |row| row.get(0),
            )
            .map_err(|e| format!("Schema integrity trigger probe generation: {e}"))?
            .checked_add(1)
            .ok_or("Schema integrity trigger probe generation exhausted SQLite INTEGER range")?;
        let snapshot_digest = "0".repeat(64);
        let snapshot = KnowledgeSnapshotReceipt {
            generation: u64::try_from(snapshot_generation)
                .map_err(|_| "Schema integrity trigger probe generation overflow")?,
            canonical_digest_hex: snapshot_digest.clone(),
            receipt_digest_hex: String::new(),
        };

        conn.execute(
            "INSERT INTO knowledge_snapshot_receipts
             (generation, canonical_digest_hex, receipt_digest_hex)
             VALUES (?1, ?2, ?3)",
            rusqlite::params![
                snapshot_generation,
                snapshot_digest,
                snapshot.canonical_receipt_digest_hex(),
            ],
        )
        .map_err(|e| format!("Schema integrity trigger probe snapshot insert: {e}"))?;

        if conn
            .execute(
                "INSERT INTO knowledge_snapshot_receipts
                 (generation, canonical_digest_hex, receipt_digest_hex)
                 VALUES (?1, ?2, ?3)",
                rusqlite::params![
                    snapshot_generation + 1,
                    "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg",
                    "0000000000000000000000000000000000000000000000000000000000000000",
                ],
            )
            .is_ok()
        {
            return Err(
                "Schema integrity check failed: snapshot receipt trigger did not reject malformed insert"
                    .into(),
            );
        }

        let validation_sequence: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(validation_sequence), 0)
                 FROM knowledge_snapshot_validation_receipts",
                [],
                |row| row.get(0),
            )
            .map_err(|e| format!("Schema integrity trigger probe sequence: {e}"))?
            .checked_add(1)
            .ok_or("Schema integrity trigger probe sequence exhausted SQLite INTEGER range")?;
        let validation_event = format!("__epf011_trigger_probe_{snapshot_generation}");
        let validation = KnowledgeSnapshotValidationReceipt {
            validation_event: validation_event.clone(),
            generation: u64::try_from(snapshot_generation)
                .map_err(|_| "Schema integrity trigger probe generation overflow")?,
            snapshot_digest_hex: "0".repeat(64),
            validator_ref: "probe-validator".into(),
            validator_version: "probe-v1".into(),
            validation_profile: "probe".into(),
            conforms: true,
            report_digest_hex: None,
        };

        conn.execute(
            "INSERT INTO knowledge_snapshot_validation_receipts
             (validation_event, validation_sequence, generation, snapshot_digest_hex,
              validator_ref, validator_version, validation_profile, conforms,
              report_digest_hex, receipt_digest_hex)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                validation.validation_event,
                validation_sequence,
                snapshot_generation,
                validation.snapshot_digest_hex,
                validation.validator_ref,
                validation.validator_version,
                validation.validation_profile,
                validation.conforms,
                validation.report_digest_hex,
                validation.canonical_digest_hex_for_sequence(
                    u64::try_from(validation_sequence)
                        .map_err(|_| "Schema integrity trigger probe sequence overflow")?,
                ),
            ],
        )
        .map_err(|e| format!("Schema integrity trigger probe validation insert: {e}"))?;

        if conn
            .execute(
                "INSERT INTO knowledge_snapshot_validation_receipts
                 (validation_event, validation_sequence, generation, snapshot_digest_hex,
                  validator_ref, validator_version, validation_profile, conforms,
                  report_digest_hex, receipt_digest_hex)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, ?9)",
                rusqlite::params![
                    format!("{validation_event}-bad"),
                    validation_sequence + 1,
                    snapshot_generation,
                    "gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg",
                    " ",
                    "probe-v1",
                    " ",
                    1_i64,
                    "0000000000000000000000000000000000000000000000000000000000000000",
                ],
            )
            .is_ok()
        {
            return Err(
                "Schema integrity check failed: validation receipt trigger did not reject malformed insert"
                    .into(),
            );
        }

        for (statement, label) in [
            (
                format!(
                    "UPDATE knowledge_snapshot_receipts
                     SET canonical_digest_hex = '1'
                     WHERE generation = {snapshot_generation}"
                ),
                "snapshot receipt UPDATE",
            ),
            (
                format!(
                    "DELETE FROM knowledge_snapshot_receipts
                     WHERE generation = {snapshot_generation}"
                ),
                "snapshot receipt DELETE",
            ),
            (
                format!(
                    "UPDATE knowledge_snapshot_validation_receipts
                     SET validator_version = 'mutated'
                     WHERE validation_event = '{validation_event}'"
                ),
                "validation receipt UPDATE",
            ),
            (
                format!(
                    "DELETE FROM knowledge_snapshot_validation_receipts
                     WHERE validation_event = '{validation_event}'"
                ),
                "validation receipt DELETE",
            ),
        ] {
            if conn.execute_batch(&statement).is_ok() {
                return Err(format!(
                    "Schema integrity check failed: {label} was not rejected"
                ));
            }
        }

        Ok(())
    })();

    let rollback = conn.execute_batch(
        "ROLLBACK TO epf011_trigger_attestation;
         RELEASE epf011_trigger_attestation;",
    );
    if let Err(e) = rollback {
        return Err(format!(
            "Schema integrity trigger probe rollback failed: {e}"
        ));
    }

    result
}


/// Read the complete knowledge snapshot from one already-open SQLite transaction.
///
/// This helper performs no transaction control of its own, allowing callers to bind
/// the same read view to a write transaction (for post-reconciliation verification)
/// or to a read transaction (for snapshot loading).
fn read_snapshot_from_transaction(
    tx: &rusqlite::Transaction<'_>,
) -> Result<KnowledgePersistenceSnapshot, String> {
        let facts = {
            let mut stmt = tx
                .prepare(
                    "SELECT id, memory_id, canonical_identity, provenance_family, vector_blob, source_text, confidence, domain, cycle, is_causal
                     FROM knowledge_facts ORDER BY cycle DESC, memory_id ASC, id ASC",
                )
                .map_err(|e| format!("Prepare snapshot facts: {e}"))?;
            stmt.query_map([], |row| {
                Ok(FactRecord {
                    memory_id: row.get(1)?,
                    canonical_identity: row.get(2)?,
                    provenance_family: row.get(3)?,
                    vector_bytes: row.get(4)?,
                    source_text: row.get(5)?,
                    confidence: row.get(6)?,
                    domain: row.get(7)?,
                    cycle: u64::try_from(row.get::<_, i64>(8)?)
                        .map_err(|_| rusqlite::Error::InvalidColumnType(8, "cycle".into(), rusqlite::types::Type::Integer))?,
                    is_causal: row.get(9)?,
                })
            })
            .map_err(|e| format!("Query snapshot facts: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Load snapshot fact row: {e}"))?
        };

        let provenance_relations = {
            let mut stmt = tx
                .prepare(
                    "SELECT source_memory_id, target_memory_id, kind, created_at
                     FROM knowledge_provenance_relations
                     ORDER BY created_at, source_memory_id, target_memory_id, kind",
                )
                .map_err(|e| format!("Prepare snapshot provenance: {e}"))?;
            let rows = stmt
                .query_map([], |row| {
                    let kind: String = row.get(2)?;
                    let kind = provenance_kind_from_persisted(&kind).ok_or_else(|| {
                        rusqlite::Error::InvalidColumnType(
                            2,
                            "kind".into(),
                            rusqlite::types::Type::Text,
                        )
                    })?;
                    Ok(ProvenanceRelationRecord {
                        source_memory_id: row.get(0)?,
                        target_memory_id: row.get(1)?,
                        kind,
                        created_at: row.get(3)?,
                    })
                })
                .map_err(|e| format!("Query snapshot provenance: {e}"))?;

            let mut loaded = Vec::new();
            for row in rows {
                let record = row.map_err(|e| format!("Load snapshot provenance row: {e}"))?;
                ProvenanceRelation::from(record.clone())
                    .validate()
                    .map_err(|e| format!("Invalid persisted provenance relation: {e}"))?;
                loaded.push(record);
            }
            loaded
        };

        let causal_edges = {
            let mut stmt = tx
                .prepare(
                    "SELECT cause, effect, strength, is_inhibitory, cycle
                     FROM knowledge_causal_edges
                     ORDER BY cycle DESC, cause ASC, effect ASC",
                )
                .map_err(|e| format!("Prepare snapshot causal edges: {e}"))?;
            let edges = stmt
                .query_map([], |row| {
                    Ok(CausalEdgeRecord {
                        cause: row.get(0)?,
                        effect: row.get(1)?,
                        strength: row.get(2)?,
                        is_inhibitory: row.get(3)?,
                        cycle: u64::try_from(row.get::<_, i64>(4)?)
                            .map_err(|_| rusqlite::Error::InvalidColumnType(4, "cycle".into(), rusqlite::types::Type::Integer))?,
                    })
                })
                .map_err(|e| format!("Query snapshot causal edges: {e}"))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Load snapshot causal edge row: {e}"))?;

            for edge in &edges {
                edge.validate()
                    .map_err(|e| format!("Invalid persisted causal edge snapshot: {e}"))?;
            }
            edges
        };

        let ontology = {
            let mut stmt = tx
                .prepare(
                    "SELECT name, vector_blob, usage_count, utility, created_at_cycle, last_used_cycle, is_a_parent
                     FROM knowledge_ontology
                     ORDER BY utility DESC, name ASC, created_at_cycle ASC, last_used_cycle ASC",
                )
                .map_err(|e| format!("Prepare snapshot ontology: {e}"))?;
            stmt.query_map([], |row| {
                Ok(OntologyRecord {
                    name: row.get(0)?,
                    vector_bytes: row.get(1)?,
                    usage_count: u64::try_from(row.get::<_, i64>(2)?)
                        .map_err(|_| rusqlite::Error::InvalidColumnType(2, "usage_count".into(), rusqlite::types::Type::Integer))?,
                    utility: row.get(3)?,
                    created_at_cycle: u64::try_from(row.get::<_, i64>(4)?)
                        .map_err(|_| rusqlite::Error::InvalidColumnType(4, "created_at_cycle".into(), rusqlite::types::Type::Integer))?,
                    last_used_cycle: u64::try_from(row.get::<_, i64>(5)?)
                        .map_err(|_| rusqlite::Error::InvalidColumnType(5, "last_used_cycle".into(), rusqlite::types::Type::Integer))?,
                    is_a_parent: row.get(6)?,
                })
            })
            .map_err(|e| format!("Query snapshot ontology: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Load snapshot ontology row: {e}"))?
        };

    Ok(KnowledgePersistenceSnapshot {
        facts,
        provenance_relations,
        causal_edges,
        ontology,
    })
}

fn verify_current_snapshot_matches_latest_receipt_in_tx(
    tx: &rusqlite::Transaction<'_>,
) -> Result<(), String> {
    let receipt = tx
        .query_row(
            "SELECT generation, canonical_digest_hex, receipt_digest_hex
             FROM knowledge_snapshot_receipts
             ORDER BY generation DESC
             LIMIT 1",
            [],
            |row| {
                let generation = row.get::<_, i64>(0)?;
                Ok(KnowledgeSnapshotReceipt {
                    generation: u64::try_from(generation).map_err(|_| {
                        rusqlite::Error::IntegralValueOutOfRange(0, generation)
                    })?,
                    canonical_digest_hex: row.get(1)?,
                    receipt_digest_hex: row.get(2)?,
                })
            },
        )
        .optional()
        .map_err(|e| format!("Load latest snapshot receipt for state verification: {e}"))?;

    let Some(receipt) = receipt else {
        return Ok(());
    };

    let snapshot = read_snapshot_from_transaction(tx)?;
    let actual_digest = snapshot.canonical_digest_hex();
    if actual_digest != receipt.canonical_digest_hex {
        return Err(format!(
            "Snapshot receipt digest mismatch: generation {} records {}, observed {}",
            receipt.generation, receipt.canonical_digest_hex, actual_digest
        ));
    }
    Ok(())
}

fn verify_snapshot_receipts_in_tx(
    tx: &rusqlite::Transaction<'_>,
) -> Result<(), String> {
    let mut stmt = tx
        .prepare(
            "SELECT generation, canonical_digest_hex, receipt_digest_hex
             FROM knowledge_snapshot_receipts
             ORDER BY generation ASC",
        )
        .map_err(|e| format!("Prepare snapshot receipt verification: {e}"))?;

    let rows = stmt
        .query_map([], |row| {
            let generation = u64::try_from(row.get::<_, i64>(0)?).map_err(|_| {
                rusqlite::Error::InvalidColumnType(
                    0,
                    "generation".into(),
                    rusqlite::types::Type::Integer,
                )
            })?;
            Ok(KnowledgeSnapshotReceipt {
                generation,
                canonical_digest_hex: row.get(1)?,
                receipt_digest_hex: row.get(2)?,
            })
        })
        .map_err(|e| format!("Query snapshot receipts for verification: {e}"))?;

    let mut expected_generation = 1_u64;
    for row in rows {
        let receipt =
            row.map_err(|e| format!("Load snapshot receipt for verification: {e}"))?;
        if receipt.generation == 0 {
            return Err("Snapshot receipt generation must be positive".into());
        }
        if receipt.generation != expected_generation {
            return Err(format!(
                "Snapshot receipt generation discontinuity: expected {}, observed {}",
                expected_generation, receipt.generation
            ));
        }
        if !is_hex_digest(&receipt.canonical_digest_hex) {
            return Err(format!(
                "Snapshot receipt canonical digest is not a 64-character hexadecimal digest: generation {}",
                receipt.generation
            ));
        }
        if receipt.receipt_digest_hex != receipt.canonical_receipt_digest_hex() {
            return Err(format!(
                "Snapshot receipt self-digest mismatch: generation {}",
                receipt.generation
            ));
        }
        expected_generation = expected_generation
            .checked_add(1)
            .ok_or("Snapshot receipt generation exhausted validation range")?;
    }
    Ok(())
}

fn verify_snapshot_validation_receipts_in_tx(
    tx: &rusqlite::Transaction<'_>,
) -> Result<(), String> {
    let mut stmt = tx
        .prepare(
            "SELECT v.validation_sequence, v.validation_event, v.generation, v.snapshot_digest_hex, v.validator_ref,
                    v.validator_version, v.validation_profile, v.conforms, v.report_digest_hex,
                    v.receipt_digest_hex, r.canonical_digest_hex
             FROM knowledge_snapshot_validation_receipts v
             LEFT JOIN knowledge_snapshot_receipts r ON r.generation = v.generation
             ORDER BY v.validation_sequence ASC",
        )
        .map_err(|e| format!("Prepare validation receipt verification: {e}"))?;

    let rows = stmt
        .query_map([], |row| {
            let validation_sequence = u64::try_from(row.get::<_, i64>(0)?).map_err(|_| {
                rusqlite::Error::InvalidColumnType(
                    0,
                    "validation_sequence".into(),
                    rusqlite::types::Type::Integer,
                )
            })?;
            let generation = u64::try_from(row.get::<_, i64>(2)?).map_err(|_| {
                rusqlite::Error::InvalidColumnType(
                    2,
                    "generation".into(),
                    rusqlite::types::Type::Integer,
                )
            })?;
            Ok((
                KnowledgeSnapshotValidationReceipt {
                    validation_event: row.get(1)?,
                    generation,
                    snapshot_digest_hex: row.get(3)?,
                    validator_ref: row.get(4)?,
                    validator_version: row.get(5)?,
                    validation_profile: row.get(6)?,
                    conforms: row.get(7)?,
                    report_digest_hex: row.get(8)?,
                },
                validation_sequence,
                row.get::<_, Option<String>>(9)?,
                row.get::<_, Option<String>>(10)?,
            ))
        })
        .map_err(|e| format!("Query validation receipts for verification: {e}"))?;

    let mut expected_sequence = 1_u64;
    let mut previous_generation: Option<u64> = None;
    for row in rows {
        let (receipt, validation_sequence, stored_digest, linked_snapshot_digest) =
            row.map_err(|e| format!("Load validation receipt for verification: {e}"))?;

        receipt
            .validate_input()
            .map_err(|e| format!("Invalid persisted snapshot validation receipt: {e}"))?;

        if let Some(previous_generation) = previous_generation {
            if receipt.generation < previous_generation {
                return Err(format!(
                    "Snapshot validation receipt generation regression at sequence {}: previous {}, observed {}",
                    validation_sequence, previous_generation, receipt.generation
                ));
            }
        }
        previous_generation = Some(receipt.generation);

        let Some(linked_snapshot_digest) = linked_snapshot_digest else {
            return Err(format!(
                "Snapshot validation receipt references missing snapshot generation: {}",
                receipt.validation_event
            ));
        };
        if linked_snapshot_digest != receipt.snapshot_digest_hex {
            return Err(format!(
                "Snapshot validation receipt snapshot digest mismatch: {}",
                receipt.validation_event
            ));
        }

        let Some(stored_digest) = stored_digest else {
            return Err(format!(
                "Snapshot validation receipt missing self-digest: {}",
                receipt.validation_event
            ));
        };
        let expected = receipt.canonical_digest_hex_for_sequence(validation_sequence);
        if stored_digest != expected {
            return Err(format!(
                "Snapshot validation receipt self-digest mismatch: {}",
                receipt.validation_event
            ));
        }

        if validation_sequence != expected_sequence {
            return Err(format!(
                "Snapshot validation receipt sequence discontinuity: expected {}, observed {}",
                expected_sequence, validation_sequence
            ));
        }
        expected_sequence = expected_sequence
            .checked_add(1)
            .ok_or("Snapshot validation sequence exhausted validation range")?;
    }
    Ok(())
}

fn delete_absent_keys(
    tx: &rusqlite::Transaction<'_>,
    table: &str,
    key_column: &str,
    retained: Vec<&str>,
) -> Result<(), String> {
    tx.execute_batch(
        "CREATE TEMP TABLE IF NOT EXISTS epf_snapshot_keys_single (
            key TEXT PRIMARY KEY
        );
        DELETE FROM epf_snapshot_keys_single;",
    )
    .map_err(|e| format!("Prepare {table} reconciliation: {e}"))?;

    {
        let mut stmt = tx
            .prepare_cached("INSERT OR IGNORE INTO epf_snapshot_keys_single (key) VALUES (?1)")
            .map_err(|e| format!("Prepare {table} reconciliation keys: {e}"))?;
        for value in retained {
            stmt.execute([value])
                .map_err(|e| format!("Reconcile {table} key: {e}"))?;
        }
    }

    let sql = format!(
        "DELETE FROM {table}
         WHERE {key_column} IS NULL
            OR NOT EXISTS (
                SELECT 1
                FROM epf_snapshot_keys_single retained
                WHERE retained.key = {table}.{key_column}
            )"
    );
    tx.execute(&sql, [])
        .map_err(|e| format!("Reconcile {table}: {e}"))?;
    Ok(())
}

fn delete_absent_composite_keys(
    tx: &rusqlite::Transaction<'_>,
    table: &str,
    left_column: &str,
    right_column: &str,
    retained: Vec<(&str, &str)>,
) -> Result<(), String> {
    tx.execute_batch(
        "CREATE TEMP TABLE IF NOT EXISTS epf_snapshot_keys_pair (
            left_key TEXT NOT NULL,
            right_key TEXT NOT NULL,
            PRIMARY KEY (left_key, right_key)
        );
        DELETE FROM epf_snapshot_keys_pair;",
    )
    .map_err(|e| format!("Prepare {table} reconciliation: {e}"))?;

    {
        let mut stmt = tx
            .prepare_cached(
                "INSERT OR IGNORE INTO epf_snapshot_keys_pair (left_key, right_key)
                 VALUES (?1, ?2)",
            )
            .map_err(|e| format!("Prepare {table} reconciliation keys: {e}"))?;
        for (left, right) in retained {
            stmt.execute([left, right])
                .map_err(|e| format!("Reconcile {table} composite key: {e}"))?;
        }
    }

    let sql = format!(
        "DELETE FROM {table}
         WHERE NOT EXISTS (
             SELECT 1
             FROM epf_snapshot_keys_pair retained
             WHERE retained.left_key = {table}.{left_column}
               AND retained.right_key = {table}.{right_column}
         )"
    );
    tx.execute(&sql, [])
        .map_err(|e| format!("Reconcile {table}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_connection_configures_busy_timeout() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_busy_timeout_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let p = KnowledgePersistence::new(&db_path);
        let conn = p.open_connection().unwrap();
        let timeout: i64 = conn
            .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
            .unwrap();
        assert_eq!(timeout, 5_000);

        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn test_open_connection_enables_foreign_keys() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_foreign_key_enforcement_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let p = KnowledgePersistence::new(&db_path);
        let conn = p.open_connection().unwrap();
        let enabled: i64 = conn
            .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
            .unwrap();
        assert_eq!(enabled, 1);

        p.ensure_schema(&conn).unwrap();
        let err = conn
            .execute(
                "INSERT INTO knowledge_snapshot_validation_receipts
                 (validation_event, validation_sequence, generation, snapshot_digest_hex,
                  validator_ref, validator_version, validation_profile, conforms, receipt_digest_hex)
                 VALUES ('orphan', 1, 999999,
                         '0000000000000000000000000000000000000000000000000000000000000000',
                         'validator', 'v1', 'profile', 1,
                         '0000000000000000000000000000000000000000000000000000000000000000')",
                [],
            )
            .unwrap_err();
        assert!(err.to_string().contains("FOREIGN KEY"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_unconfigured() {
        let mut p = KnowledgePersistence::default();
        assert!(!p.is_configured());
        assert!(p.save_facts(&[]).is_err());
    }

    #[test]
    fn test_provenance_storage_uses_stable_wire_codes() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_provenance_wire_code_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let relation = ProvenanceRelationRecord {
            source_memory_id: "source".into(),
            target_memory_id: "target".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "event:1".into(),
        };
        assert_eq!(p.save_provenance_relations(&[relation]).unwrap(), 1);

        let conn = p.open_connection().unwrap();
        let stored: String = conn
            .query_row(
                "SELECT kind FROM knowledge_provenance_relations
                 WHERE source_memory_id = 'source'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stored, "derived_from");

        let loaded = p.load_provenance_relations().unwrap();
        assert_eq!(loaded[0].kind, ProvenanceRelationKind::DerivedFrom);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_provenance_storage_normalization_preserves_v1_snapshot_digest() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_provenance_digest_compatibility_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let facts = [FactRecord {
            memory_id: "digest-compatible".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x19; BinaryHV::BYTES],
            source_text: "digest compatible".into(),
            confidence: 0.6,
            domain: None,
            cycle: 1,
            is_causal: false,
        }];
        let relations = [ProvenanceRelationRecord {
            source_memory_id: "digest-compatible".into(),
            target_memory_id: "external-source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "event:1".into(),
        }];

        p.save_snapshot(&facts, &relations, &[], &[]).unwrap();
        let committed = p.latest_snapshot_receipt().unwrap().unwrap();
        let before = p.load_snapshot().unwrap().canonical_digest_hex();
        assert_eq!(before, committed.canonical_digest_hex);

        let conn = p.open_connection().unwrap();
        let stored_kind: String = conn
            .query_row(
                "SELECT kind FROM knowledge_provenance_relations
                 WHERE source_memory_id = 'digest-compatible'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stored_kind, "derived_from");

        let after = p.load_snapshot().unwrap().canonical_digest_hex();
        assert_eq!(after, committed.canonical_digest_hex);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schema_migration_normalizes_legacy_provenance_wire_codes() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_legacy_provenance_wire_code_migration_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    canonical_identity TEXT,
                    provenance_family TEXT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE knowledge_provenance_relations (
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source_memory_id, target_memory_id, kind, created_at)
                );
                INSERT INTO knowledge_provenance_relations
                    (source_memory_id, target_memory_id, kind, created_at)
                VALUES ('source', 'target', 'DerivedFrom', 'event:1');",
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let conn = p.open_connection().unwrap();
        p.ensure_schema(&conn).unwrap();

        let stored: String = conn
            .query_row(
                "SELECT kind FROM knowledge_provenance_relations",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stored, "derived_from");

        let loaded = p.load_provenance_relations().unwrap();
        assert_eq!(loaded[0].kind, ProvenanceRelationKind::DerivedFrom);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_legacy_validation_migration_rejects_unrecognized_receipt_digest() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_legacy_validation_digest_drift_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_snapshot_receipts (
                    generation INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_digest_hex TEXT NOT NULL,
                    receipt_digest_hex TEXT NOT NULL
                );
                CREATE TABLE knowledge_snapshot_validation_receipts (
                    validation_event TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL,
                    snapshot_digest_hex TEXT NOT NULL,
                    validator_ref TEXT NOT NULL,
                    validator_version TEXT NOT NULL,
                    validation_profile TEXT NOT NULL,
                    conforms INTEGER NOT NULL,
                    report_digest_hex TEXT,
                    receipt_digest_hex TEXT
                );
                INSERT INTO knowledge_snapshot_receipts
                    (generation, canonical_digest_hex, receipt_digest_hex)
                VALUES (1, 'snapshot-digest', 'placeholder');
                INSERT INTO knowledge_snapshot_validation_receipts
                    (validation_event, generation, snapshot_digest_hex, validator_ref,
                     validator_version, validation_profile, conforms, report_digest_hex,
                     receipt_digest_hex)
                VALUES (
                    'validation:legacy',
                    1,
                    'snapshot-digest',
                    'validator',
                    'v1',
                    'profile',
                    1,
                    NULL,
                    'definitely-not-a-valid-epf-011-digest'
                );
                CREATE TRIGGER trg_knowledge_snapshot_validation_receipts_no_update
                BEFORE UPDATE ON knowledge_snapshot_validation_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'knowledge_snapshot_validation_receipts is append-only: UPDATE prohibited');
                END;",
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let conn = p.open_connection().unwrap();
        let err = p.ensure_schema(&conn).unwrap_err();
        assert!(err.contains("Validation receipt self-digest is not a recognized EPF-011 integrity digest"));

        // The migration is transactional: failure must not leave behind a partially
        // upgraded validation ledger.
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let columns: Vec<String> = conn
            .prepare("PRAGMA table_info(knowledge_snapshot_validation_receipts)")
            .unwrap()
            .query_map([], |row| row.get::<_, String>(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(!columns.iter().any(|column| column == "validation_sequence"));
        assert!(columns.iter().any(|column| column == "receipt_digest_hex"));

        let validation_digest: String = conn
            .query_row(
                "SELECT receipt_digest_hex
                 FROM knowledge_snapshot_validation_receipts
                 WHERE validation_event = 'validation:legacy'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(validation_digest, "definitely-not-a-valid-epf-011-digest");

        let snapshot_digest: String = conn
            .query_row(
                "SELECT receipt_digest_hex
                 FROM knowledge_snapshot_receipts
                 WHERE generation = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(snapshot_digest, "placeholder");

        let trigger_sql: String = conn
            .query_row(
                "SELECT sql
                 FROM sqlite_master
                 WHERE type = 'trigger'
                   AND name = 'trg_knowledge_snapshot_validation_receipts_no_update'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(trigger_sql.contains("UPDATE prohibited"));

        let update_err = conn
            .execute(
                "UPDATE knowledge_snapshot_validation_receipts
                 SET validator_ref = 'should-not-commit'
                 WHERE validation_event = 'validation:legacy'",
                [],
            )
            .unwrap_err();
        assert!(update_err
            .to_string()
            .contains("knowledge_snapshot_validation_receipts is append-only"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_legacy_provenance_normalization_preserves_existing_snapshot_receipt() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_legacy_provenance_receipt_compatibility_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let fact = FactRecord {
            memory_id: "legacy-receipt-fact".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x5A; BinaryHV::BYTES],
            source_text: "legacy receipt".into(),
            confidence: 0.8,
            domain: Some("compatibility".into()),
            cycle: 3,
            is_causal: false,
        };
        let relation = ProvenanceRelationRecord {
            source_memory_id: fact.memory_id.clone(),
            target_memory_id: "external-source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "event:legacy".into(),
        };
        let snapshot = KnowledgePersistenceSnapshot {
            facts: vec![fact.clone()],
            provenance_relations: vec![relation.clone()],
            causal_edges: Vec::new(),
            ontology: Vec::new(),
        };
        let canonical_digest = snapshot.canonical_digest_hex();
        let receipt = KnowledgeSnapshotReceipt {
            generation: 1,
            canonical_digest_hex: canonical_digest.clone(),
            receipt_digest_hex: String::new(),
        };
        let receipt_digest = receipt.canonical_receipt_digest_hex();

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    canonical_identity TEXT,
                    provenance_family TEXT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE knowledge_provenance_relations (
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source_memory_id, target_memory_id, kind, created_at)
                );
                CREATE TABLE knowledge_snapshot_receipts (
                    generation INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_digest_hex TEXT NOT NULL,
                    receipt_digest_hex TEXT NOT NULL
                );
                INSERT INTO knowledge_facts
                    (memory_id, canonical_identity, provenance_family, vector_blob,
                     source_text, confidence, domain, cycle, is_causal)
                VALUES ('legacy-receipt-fact', NULL, NULL, ?1, 'legacy receipt',
                        0.8, 'compatibility', 3, 0);
                INSERT INTO knowledge_provenance_relations
                    (source_memory_id, target_memory_id, kind, created_at)
                VALUES ('legacy-receipt-fact', 'external-source', 'DerivedFrom', 'event:legacy');
                INSERT INTO knowledge_snapshot_receipts
                    (generation, canonical_digest_hex, receipt_digest_hex)
                VALUES (1, ?2, ?3);",
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let conn = p.open_connection().unwrap();
        p.ensure_schema(&conn).unwrap();

        let stored_kind: String = conn
            .query_row(
                "SELECT kind FROM knowledge_provenance_relations
                 WHERE source_memory_id = 'legacy-receipt-fact'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stored_kind, "derived_from");

        let (loaded, verified_receipt) = p.load_snapshot_with_receipt().unwrap().unwrap();
        assert_eq!(loaded, snapshot);
        assert_eq!(verified_receipt.generation, 1);
        assert_eq!(verified_receipt.canonical_digest_hex, canonical_digest);
        assert_eq!(verified_receipt.receipt_digest_hex, receipt_digest);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sqlite_provenance_triggers_reject_out_of_band_invalid_rows() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_provenance_trigger_guard_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence
            .save_provenance_relations(&[])
            .expect("schema initialization should succeed");

        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let invalid_kind = conn.execute(
            "INSERT INTO knowledge_provenance_relations
             (source_memory_id, target_memory_id, kind, created_at)
             VALUES ('derived', 'source', 'DerivedFrom', 'cycle:1')",
            [],
        );
        assert!(invalid_kind.is_err());

        conn.execute(
            "INSERT INTO knowledge_provenance_relations
             (source_memory_id, target_memory_id, kind, created_at)
             VALUES ('derived', 'source', 'derived_from', 'cycle:1')",
            [],
        )
        .unwrap();

        let invalid_update = conn.execute(
            "UPDATE knowledge_provenance_relations
             SET kind = 'NotAProvenanceKind'
             WHERE source_memory_id = 'derived' AND target_memory_id = 'source'",
            [],
        );
        assert!(invalid_update.is_err());

        let invalid_self_reference = conn.execute(
            "UPDATE knowledge_provenance_relations
             SET target_memory_id = source_memory_id
             WHERE source_memory_id = 'derived' AND target_memory_id = 'source'",
            [],
        );
        assert!(invalid_self_reference.is_err());

        let invalid_blank_timestamp = conn.execute(
            "UPDATE knowledge_provenance_relations
             SET created_at = '   '
             WHERE source_memory_id = 'derived' AND target_memory_id = 'source'",
            [],
        );
        assert!(invalid_blank_timestamp.is_err());

        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM knowledge_provenance_relations",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schema_migration_rejects_unknown_provenance_kind() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_unknown_provenance_kind_migration_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_provenance_relations (
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source_memory_id, target_memory_id, kind, created_at)
                );
                INSERT INTO knowledge_provenance_relations
                    (source_memory_id, target_memory_id, kind, created_at)
                VALUES ('source', 'target', 'future_kind', 'event:1');",
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let conn = p.open_connection().unwrap();
        let err = p.ensure_schema(&conn).unwrap_err();
        assert!(err.contains("unknown kind"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_initialized_schema_drift_is_not_masked_by_process_state() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_initialized_schema_drift_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        p.save_facts(&[FactRecord {
            memory_id: "initialized-drift".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x21; BinaryHV::BYTES],
            source_text: "initialized drift".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "DROP TRIGGER trg_knowledge_snapshot_receipts_no_update;",
            )
            .unwrap();
        }

        let conn = p.open_connection().unwrap();
        let err = p.ensure_schema(&conn).unwrap_err();
        assert!(err.contains("Schema integrity check failed"));
        assert!(err.contains("trg_knowledge_snapshot_receipts_no_update"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_provenance_insert_boundary_rejects_null_kind() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_provenance_null_kind_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let conn = p.open_connection().unwrap();
        p.ensure_schema(&conn).unwrap();

        let err = conn
            .execute(
                "INSERT INTO knowledge_provenance_relations
                 (source_memory_id, target_memory_id, kind, created_at)
                 VALUES ('null-kind-source', 'null-kind-target', NULL, 'event:null-kind')",
                [],
            )
            .unwrap_err();
        assert!(err.to_string().contains(
            "knowledge_provenance_relations requires valid identities, timestamp, and stable kind"
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_receipt_insert_boundary_rejects_null_conforms() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_receipt_null_conforms_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        p.save_snapshot(&[], &[], &[], &[]).unwrap();
        let conn = p.open_connection().unwrap();

        let err = conn
            .execute(
                "INSERT INTO knowledge_snapshot_validation_receipts
                 (validation_event, validation_sequence, generation, snapshot_digest_hex,
                  validator_ref, validator_version, validation_profile, conforms,
                  report_digest_hex, receipt_digest_hex)
                 VALUES ('null-conforms', 1, 1,
                         (SELECT canonical_digest_hex
                          FROM knowledge_snapshot_receipts
                          WHERE generation = 1),
                         'validator', 'v1', 'profile', NULL, NULL,
                         '0000000000000000000000000000000000000000000000000000000000000000')",
                [],
            )
            .unwrap_err();
        assert!(err.to_string().contains(
            "knowledge_snapshot_validation_receipts requires valid identity, digest, validator, outcome, and positive sequence"
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schema_column_attestation_rejects_wrong_validation_sequence_type() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_column_contract_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE knowledge_snapshot_validation_receipts (
                validation_event TEXT PRIMARY KEY,
                validation_sequence BLOB,
                generation INTEGER NOT NULL,
                snapshot_digest_hex TEXT NOT NULL,
                validator_ref TEXT NOT NULL,
                validator_version TEXT NOT NULL,
                validation_profile TEXT NOT NULL,
                conforms INTEGER NOT NULL,
                report_digest_hex TEXT,
                receipt_digest_hex TEXT NOT NULL
            );",
        )
        .unwrap();

        let err = verify_table_column_contract(
            &conn,
            "knowledge_snapshot_validation_receipts",
            &[
                ("validation_event", "TEXT", 1),
                ("validation_sequence", "INTEGER", 0),
            ],
        )
        .unwrap_err();
        assert_eq!(
            err,
            "Schema integrity check failed: column validation_sequence on knowledge_snapshot_validation_receipts has declared type BLOB, expected INTEGER"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }


    #[test]
    fn test_initialized_schema_rejects_missing_core_fact_identity_column() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_core_fact_schema_drift_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        p.save_facts(&[FactRecord {
            memory_id: "core-fact-schema".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x42; BinaryHV::BYTES],
            source_text: "core fact schema".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        let conn = p.open_connection().unwrap();
        conn.execute_batch(
            "ALTER TABLE knowledge_facts
             RENAME COLUMN memory_id TO memory_identifier;",
        )
        .unwrap();

        let err = verify_initialized_schema_integrity(&conn).unwrap_err();
        assert_eq!(
            err,
            "Schema integrity check failed: missing column memory_id on knowledge_facts"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_trigger_probe_survives_existing_sentinel_ids() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_trigger_probe_collision_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        p.save_facts(&[FactRecord {
            memory_id: "trigger-probe-bootstrap".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x64; BinaryHV::BYTES],
            source_text: "trigger probe bootstrap".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        let conn = p.open_connection().unwrap();
        conn.execute(
            "INSERT INTO knowledge_facts
             (memory_id, vector_blob, source_text, confidence, cycle, is_causal)
             VALUES ('__epf011_trigger_probe_fact', ?1, 'sentinel', 0.5, 1, 0)",
            [vec![0x65u8; BinaryHV::BYTES]],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO knowledge_provenance_relations
             (source_memory_id, target_memory_id, kind, created_at)
             VALUES ('__epf011_trigger_probe_source', '__epf011_trigger_probe_target',
                     'derived_from', 'event:trigger-probe')",
            [],
        )
        .unwrap();

        verify_initialized_schema_integrity(&conn).unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schema_trigger_attestation_rejects_comment_only_fact_and_provenance_guards() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_comment_core_trigger_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        p.save_facts(&[FactRecord {
            memory_id: "comment-core-trigger".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x62; BinaryHV::BYTES],
            source_text: "comment core trigger".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "DROP TRIGGER trg_knowledge_facts_memory_id_required_insert;
                 CREATE TRIGGER trg_knowledge_facts_memory_id_required_insert
                 BEFORE INSERT ON knowledge_facts
                 WHEN 1
                 BEGIN
                     /* before insert on knowledge_facts
                        new.memory_id is null or trim(new.memory_id) = ''
                        raise(abort, 'knowledge_facts memory_id must be non-empty') */
                     SELECT 1;
                 END;
                 DROP TRIGGER trg_knowledge_provenance_relation_required_insert;
                 CREATE TRIGGER trg_knowledge_provenance_relation_required_insert
                 BEFORE INSERT ON knowledge_provenance_relations
                 WHEN 1
                 BEGIN
                     /* before insert on knowledge_provenance_relations
                        new.source_memory_id is null
                        new.target_memory_id is null
                        new.source_memory_id = new.target_memory_id
                        new.created_at is null
                        new.kind not in (
                            'derived_from', 'revised_from', 'supersedes',
                            'contradicts', 'corroborates', 'representation_of'
                        )
                        raise(abort, 'knowledge_provenance_relations requires valid identities, timestamp, and stable kind') */
                     SELECT 1;
                 END;",
            )
            .unwrap();
        }

        let conn = p.open_connection().unwrap();
        let err = p.ensure_schema(&conn).unwrap_err();
        assert!(err.contains("fact memory"));
        assert!(err.contains("did not reject blank insert") || err.contains("trigger"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schema_trigger_attestation_rejects_comment_only_receipt_guard() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_comment_trigger_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        p.save_facts(&[FactRecord {
            memory_id: "comment-trigger".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x61; BinaryHV::BYTES],
            source_text: "comment trigger".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "DROP TRIGGER trg_knowledge_snapshot_receipts_required_insert;
                 CREATE TRIGGER trg_knowledge_snapshot_receipts_required_insert
                 BEFORE INSERT ON knowledge_snapshot_receipts
                 WHEN 1
                 BEGIN
                     /* before insert on knowledge_snapshot_receipts
                        new.generation is null
                        new.generation <= 0
                        new.canonical_digest_hex is null
                        length(new.canonical_digest_hex) <> 64
                        new.receipt_digest_hex is null
                        length(new.receipt_digest_hex) <> 64
                        raise(abort, 'knowledge_snapshot_receipts requires positive generation and 64-character digests') */
                     SELECT 1;
                 END;",
            )
            .unwrap();
        }

        let conn = p.open_connection().unwrap();
        let err = p.ensure_schema(&conn).unwrap_err();
        assert!(err.contains("snapshot receipt trigger"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_initialized_schema_rejects_weakened_receipt_insert_trigger() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_receipt_trigger_contract_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        p.save_facts(&[FactRecord {
            memory_id: "receipt-trigger".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x51; BinaryHV::BYTES],
            source_text: "receipt trigger".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "DROP TRIGGER trg_knowledge_snapshot_receipts_required_insert;
                 CREATE TRIGGER trg_knowledge_snapshot_receipts_required_insert
                 BEFORE INSERT ON knowledge_snapshot_receipts
                 WHEN NEW.generation IS NULL
                 BEGIN
                     SELECT RAISE(ABORT, 'weakened receipt guard');
                 END;",
            )
            .unwrap();
        }

        let conn = p.open_connection().unwrap();
        let err = p.ensure_schema(&conn).unwrap_err();
        assert!(err.contains("trg_knowledge_snapshot_receipts_required_insert"));
        assert!(err.contains("missing required contract fragment"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_receipt_insert_boundary_rejects_malformed_digests_and_outcomes() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_receipt_insert_boundary_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let conn = p.open_connection().unwrap();
        p.ensure_schema(&conn).unwrap();

        let validation_report_digest_err = conn
            .execute(
                "INSERT INTO knowledge_snapshot_validation_receipts
                 (validation_event, validation_sequence, generation, snapshot_digest_hex,
                  validator_ref, validator_version, validation_profile, conforms,
                  report_digest_hex, receipt_digest_hex)
                 VALUES ('bad-report-digest', 999, 1,
                         '0000000000000000000000000000000000000000000000000000000000000000',
                         'validator', 'v1', 'profile', 1,
                         'gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg',
                         '0000000000000000000000000000000000000000000000000000000000000000')",
                [],
            )
            .unwrap_err();
        assert!(validation_report_digest_err.to_string().contains(
            "knowledge_snapshot_validation_receipts requires valid identity, digest, validator, outcome, and positive sequence"
        ));

        let snapshot_err = conn
            .execute(
                "INSERT INTO knowledge_snapshot_receipts
                 (generation, canonical_digest_hex, receipt_digest_hex)
                 VALUES (999, 'gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg', '0000000000000000000000000000000000000000000000000000000000000000')",
                [],
            )
            .unwrap_err();
        assert!(snapshot_err
            .to_string()
            .contains("knowledge_snapshot_receipts requires positive generation and 64-character digests"));

        let validation_err = conn
            .execute(
                "INSERT INTO knowledge_snapshot_validation_receipts
                 (validation_event, validation_sequence, generation, snapshot_digest_hex,
                  validator_ref, validator_version, validation_profile, conforms,
                  report_digest_hex, receipt_digest_hex)
                 VALUES ('bad', 999, 1,
                         'gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg',
                         'validator', 'v1', 'profile', 1, NULL,
                         '0000000000000000000000000000000000000000000000000000000000000000')",
                [],
            )
            .unwrap_err();
        assert!(validation_err.to_string().contains(
            "knowledge_snapshot_validation_receipts requires valid identity, digest, validator, outcome, and positive sequence"
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_initialized_schema_rejects_disabled_foreign_key_enforcement() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_fk_enforcement_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let conn = p.open_connection().unwrap();
        p.ensure_schema(&conn).unwrap();

        conn.execute_batch("PRAGMA foreign_keys = OFF;").unwrap();
        let err = verify_initialized_schema_integrity(&conn).unwrap_err();
        assert_eq!(
            err,
            "Schema integrity check failed: foreign-key enforcement is disabled on this connection"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_initialized_schema_rejects_existing_foreign_key_violation() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_fk_violation_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let conn = p.open_connection().unwrap();
        p.ensure_schema(&conn).unwrap();

        conn.execute_batch("PRAGMA foreign_keys = OFF;").unwrap();
        conn.execute(
            "INSERT INTO knowledge_snapshot_validation_receipts
             (validation_event, validation_sequence, generation, snapshot_digest_hex,
              validator_ref, validator_version, validation_profile, conforms, receipt_digest_hex)
             VALUES ('orphan', 1, 999, 'snapshot', 'validator', 'v1', 'profile', 1,
                     '0000000000000000000000000000000000000000000000000000000000000000')",
            [],
        )
        .unwrap();
        conn.execute_batch("PRAGMA foreign_keys = ON;").unwrap();

        let err = verify_initialized_schema_integrity(&conn).unwrap_err();
        assert!(err.contains("foreign-key violation in knowledge_snapshot_validation_receipts"));
        assert!(err.contains("referencing knowledge_snapshot_receipts"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_initialized_schema_rejects_unexpected_extra_validation_receipt_foreign_key() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE knowledge_snapshot_receipts (
                generation INTEGER PRIMARY KEY
            );
            CREATE TABLE knowledge_facts (
                memory_id TEXT PRIMARY KEY
            );
            CREATE TABLE knowledge_snapshot_validation_receipts (
                validation_event TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                extra_memory_id TEXT NOT NULL,
                FOREIGN KEY (generation)
                    REFERENCES knowledge_snapshot_receipts(generation),
                FOREIGN KEY (extra_memory_id)
                    REFERENCES knowledge_facts(memory_id)
            );",
        )
        .unwrap();

        let err = verify_validation_receipt_foreign_key_contract(&conn).unwrap_err();
        assert!(err.contains(
            "validation receipt generation foreign key has the wrong action or match semantics"
        ));
    }

    #[test]
    fn test_initialized_schema_rejects_cascading_validation_receipt_foreign_key() {
        let conn = rusqlite::Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE knowledge_snapshot_receipts (
                generation INTEGER PRIMARY KEY
            );
            CREATE TABLE knowledge_snapshot_validation_receipts (
                validation_event TEXT PRIMARY KEY,
                generation INTEGER NOT NULL,
                FOREIGN KEY (generation)
                    REFERENCES knowledge_snapshot_receipts(generation)
                    ON UPDATE CASCADE
                    ON DELETE CASCADE
            );",
        )
        .unwrap();

        let err = verify_validation_receipt_foreign_key_contract(&conn).unwrap_err();
        assert!(err.contains(
            "validation receipt generation foreign key has the wrong action or match semantics"
        ));
    }

    #[test]
    fn test_initialized_schema_rejects_missing_validation_receipt_foreign_key() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_initialized_fk_drift_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        p.save_facts(&[FactRecord {
            memory_id: "fk-drift".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x41; BinaryHV::BYTES],
            source_text: "foreign key drift".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        // Rebuild the validation ledger without its FK, preserving the row shape.
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "PRAGMA foreign_keys = OFF;
                 BEGIN;
                 ALTER TABLE knowledge_snapshot_validation_receipts
                    RENAME TO validation_receipts_with_fk;
                 CREATE TABLE knowledge_snapshot_validation_receipts (
                    validation_event TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL,
                    snapshot_digest_hex TEXT NOT NULL,
                    validator_ref TEXT NOT NULL,
                    validator_version TEXT NOT NULL,
                    validation_profile TEXT NOT NULL,
                    conforms INTEGER NOT NULL,
                    report_digest_hex TEXT,
                    validation_sequence INTEGER NOT NULL DEFAULT 0,
                    receipt_digest_hex TEXT
                 );
                 INSERT INTO knowledge_snapshot_validation_receipts
                    SELECT * FROM validation_receipts_with_fk;
                 DROP TABLE validation_receipts_with_fk;
                 COMMIT;",
            )
            .unwrap();
        }

        let conn = p.open_connection().unwrap();
        let err = p.ensure_schema(&conn).unwrap_err();
        assert!(err.contains("missing validation receipt generation foreign key"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_initialized_schema_rejects_missing_unique_identity_index() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_initialized_unique_index_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        p.save_facts(&[FactRecord {
            memory_id: "unique-index".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x31; BinaryHV::BYTES],
            source_text: "unique index".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch("DROP INDEX idx_facts_memory_id_unique;")
                .unwrap();
        }

        let conn = p.open_connection().unwrap();
        let err = p.ensure_schema(&conn).unwrap_err();
        assert!(err.contains("Schema integrity check failed"));
        assert!(err.contains("idx_facts_memory_id_unique"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schema_rejects_unsupported_user_version_without_migration() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_user_version_drift_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        p.save_facts(&[FactRecord {
            memory_id: "user-version-drift".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x94; BinaryHV::BYTES],
            source_text: "user version drift".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        let conn = p.open_connection().unwrap();
        conn.execute_batch("PRAGMA user_version = 999;").unwrap();

        let mut restarted = KnowledgePersistence::new(&db_path);
        let conn = restarted.open_connection().unwrap();
        let err = restarted.ensure_schema(&conn).unwrap_err();
        assert!(err.contains("Unsupported knowledge SQLite schema user_version 999"));

        let persisted_version: i64 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(persisted_version, 999);

        let fact_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM knowledge_facts", [], |row| row.get(0))
            .unwrap();
        assert_eq!(fact_count, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schema_attestation_rechecks_version_under_lock() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_attestation_version_race_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        p.save_facts(&[FactRecord {
            memory_id: "attestation-version-race".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x96; BinaryHV::BYTES],
            source_text: "attestation version race".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        let conn = p.open_connection().unwrap();
        conn.execute_batch("PRAGMA user_version = 999;").unwrap();

        let mut restarted = KnowledgePersistence::new(&db_path);
        let restarted_conn = restarted.open_connection().unwrap();
        let err = restarted.ensure_schema(&restarted_conn).unwrap_err();
        assert!(err.contains("Unsupported knowledge SQLite schema user_version 999"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_current_schema_attestation_commits_nested_probe_rollback() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_attestation_transaction_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        p.save_facts(&[FactRecord {
            memory_id: "attestation-transaction".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x95; BinaryHV::BYTES],
            source_text: "attestation transaction".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        let conn = p.open_connection().unwrap();
        let sentinel_count_before: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM knowledge_facts WHERE memory_id LIKE '__epf011_trigger_probe_fact_%'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        let mut reopened = KnowledgePersistence::new(&db_path);
        let reopened_conn = reopened.open_connection().unwrap();
        reopened.ensure_schema(&reopened_conn).unwrap();

        let sentinel_count_after: i64 = reopened_conn
            .query_row(
                "SELECT COUNT(*) FROM knowledge_facts WHERE memory_id LIKE '__epf011_trigger_probe_fact_%'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(sentinel_count_after, sentinel_count_before);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_current_schema_does_not_repair_trigger_drift_after_restart() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_restart_drift_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        p.save_facts(&[FactRecord {
            memory_id: "restart-drift".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x93; BinaryHV::BYTES],
            source_text: "restart drift".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            let version: i64 = conn
                .query_row("PRAGMA user_version", [], |row| row.get(0))
                .unwrap();
            assert_eq!(version, CURRENT_SCHEMA_USER_VERSION);
            conn.execute_batch(
                "DROP TRIGGER trg_knowledge_snapshot_receipts_required_insert;",
            )
            .unwrap();
        }

        let mut restarted = KnowledgePersistence::new(&db_path);
        let conn = restarted.open_connection().unwrap();
        let err = restarted.ensure_schema(&conn).unwrap_err();
        assert!(err.contains(
            "trg_knowledge_snapshot_receipts_required_insert"
        ));

        let trigger_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master
                 WHERE type = 'trigger'
                   AND name = 'trg_knowledge_snapshot_receipts_required_insert'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(trigger_count, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schema_initialization_is_idempotent_across_reopens() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_idempotence_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        let fact = FactRecord {
            memory_id: "idempotent-schema".into(),
            canonical_identity: None,
            provenance_family: Some("epf-011".into()),
            vector_bytes: vec![0x73; BinaryHV::BYTES],
            source_text: "schema idempotence".into(),
            confidence: 0.75,
            domain: Some("test".into()),
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();

        let conn = p.open_connection().unwrap();
        p.ensure_schema(&conn).unwrap();
        let before_receipts: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM knowledge_snapshot_receipts",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let before_facts: i64 = conn
            .query_row("SELECT COUNT(*) FROM knowledge_facts", [], |row| row.get(0))
            .unwrap();

        let mut reopened = KnowledgePersistence::new(&db_path);
        let reopened_conn = reopened.open_connection().unwrap();
        reopened.ensure_schema(&reopened_conn).unwrap();

        let after_receipts: i64 = reopened_conn
            .query_row(
                "SELECT COUNT(*) FROM knowledge_snapshot_receipts",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let after_facts: i64 = reopened_conn
            .query_row(
                "SELECT COUNT(*) FROM knowledge_facts",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(after_receipts, before_receipts);
        assert_eq!(after_facts, before_facts);

        let stored_memory_id: String = reopened_conn
            .query_row(
                "SELECT memory_id FROM knowledge_facts WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stored_memory_id, "idempotent-schema");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_initialized_schema_rejects_non_binary_identity_index() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_collated_identity_index_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        p.save_facts(&[FactRecord {
            memory_id: "collated-index".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x74; BinaryHV::BYTES],
            source_text: "collated index".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        let conn = p.open_connection().unwrap();
        conn.execute_batch(
            "DROP INDEX idx_facts_memory_id_unique;
             CREATE UNIQUE INDEX idx_facts_memory_id_unique
             ON knowledge_facts(memory_id COLLATE NOCASE);",
        )
        .unwrap();

        let err = verify_initialized_schema_integrity(&conn).unwrap_err();
        assert!(err.contains(
            "unique index idx_facts_memory_id_unique on knowledge_facts has the wrong definition"
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_initialized_schema_rejects_partial_identity_index() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_partial_identity_index_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        p.save_facts(&[FactRecord {
            memory_id: "partial-index".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x72; BinaryHV::BYTES],
            source_text: "partial index".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        let conn = p.open_connection().unwrap();
        conn.execute_batch(
            "DROP INDEX idx_facts_memory_id_unique;
             CREATE UNIQUE INDEX idx_facts_memory_id_unique
             ON knowledge_facts(memory_id)
             WHERE memory_id IS NOT NULL;",
        )
        .unwrap();

        let err = verify_initialized_schema_integrity(&conn).unwrap_err();
        assert!(err.contains(
            "unique index idx_facts_memory_id_unique on knowledge_facts has the wrong definition"
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_fact_memory_identity_cannot_be_blank_after_schema_migration() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_fact_memory_identity_trigger_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        let conn = p.open_connection().unwrap();
        p.ensure_schema(&conn).unwrap();

        let insert_err = conn
            .execute(
                "INSERT INTO knowledge_facts
                 (memory_id, vector_blob, source_text, confidence, cycle, is_causal)
                 VALUES ('   ', ?1, 'blank', 0.5, 1, 0)",
                [vec![0x11u8; BinaryHV::BYTES]],
            )
            .unwrap_err();
        assert!(insert_err.to_string().contains("memory_id must be non-empty"));

        let null_insert_err = conn
            .execute(
                "INSERT INTO knowledge_facts
                 (memory_id, vector_blob, source_text, confidence, cycle, is_causal)
                 VALUES (NULL, ?1, 'null', 0.5, 1, 0)",
                [vec![0x33u8; BinaryHV::BYTES]],
            )
            .unwrap_err();
        assert!(null_insert_err.to_string().contains("memory_id must be non-empty"));

        p.save_facts(&[FactRecord {
            memory_id: "memory-trigger".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x22u8; BinaryHV::BYTES],
            source_text: "valid".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        }])
        .unwrap();

        let update_err = conn
            .execute(
                "UPDATE knowledge_facts
                 SET memory_id = '\t'
                 WHERE memory_id = 'memory-trigger'",
                [],
            )
            .unwrap_err();
        assert!(update_err.to_string().contains("memory_id must be non-empty"));

        let persisted: String = conn
            .query_row(
                "SELECT memory_id FROM knowledge_facts WHERE memory_id = 'memory-trigger'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(persisted, "memory-trigger");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_facts_surfaces_corrupt_rows_instead_of_dropping_them() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_corrupt_fact_load_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    canonical_identity TEXT,
                    provenance_family TEXT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO knowledge_facts
                 (memory_id, canonical_identity, provenance_family, vector_blob, source_text, confidence, domain, cycle, is_causal)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    "memory-corrupt",
                    Option::<String>::None,
                    Option::<String>::None,
                    vec![1u8; 2048],
                    "corrupt confidence",
                    "not-a-number",
                    Option::<String>::None,
                    1i64,
                    false,
                ],
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let err = p.load_facts().unwrap_err();
        assert!(err.contains("Load fact row"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_facts_rejects_negative_cycle_instead_of_wrapping() {
        let dir = std::env::temp_dir().join(format!("symthaea_negative_fact_cycle_test_{}", std::process::id()));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch("CREATE TABLE knowledge_facts (id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT, canonical_identity TEXT, provenance_family TEXT, vector_blob BLOB NOT NULL, source_text TEXT NOT NULL, confidence REAL NOT NULL, domain TEXT, cycle INTEGER NOT NULL, is_causal INTEGER NOT NULL DEFAULT 0);").unwrap();
            conn.execute("INSERT INTO knowledge_facts (memory_id, vector_blob, source_text, confidence, cycle, is_causal) VALUES (?1, ?2, ?3, ?4, ?5, ?6)", rusqlite::params!["memory-negative", vec![0u8; 2048], "negative", 0.5f32, -1i64, false]).unwrap();
        }
        let mut p=KnowledgePersistence::new(&db_path);
        assert!(p.load_facts().unwrap_err().contains("Load fact row"));
        let _=std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_causal_edges_surfaces_corrupt_rows_instead_of_dropping_them() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_corrupt_causal_load_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_causal_edges (
                    cause TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    strength REAL NOT NULL,
                    is_inhibitory INTEGER NOT NULL DEFAULT 0,
                    cycle INTEGER NOT NULL,
                    PRIMARY KEY (cause, effect)
                );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO knowledge_causal_edges
                 (cause, effect, strength, is_inhibitory, cycle)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params!["cause", "effect", "not-a-number", false, 1i64],
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let err = p.load_causal_edges().unwrap_err();
        assert!(err.contains("Load causal edge row"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_ontology_surfaces_corrupt_rows_instead_of_dropping_them() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_corrupt_ontology_load_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_ontology (
                    name TEXT PRIMARY KEY,
                    vector_blob BLOB NOT NULL,
                    usage_count INTEGER NOT NULL,
                    utility REAL NOT NULL,
                    created_at_cycle INTEGER NOT NULL,
                    last_used_cycle INTEGER NOT NULL,
                    is_a_parent TEXT
                );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO knowledge_ontology
                 (name, vector_blob, usage_count, utility, created_at_cycle, last_used_cycle, is_a_parent)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    "corrupt",
                    vec![1u8; 16],
                    1i64,
                    "not-a-number",
                    1i64,
                    1i64,
                    Option::<String>::None,
                ],
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let err = p.load_ontology().unwrap_err();
        assert!(err.contains("Load ontology row"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_reconciliation_handles_large_retained_sets() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_large_reconcile_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let facts: Vec<FactRecord> = (0..1200)
            .map(|i| FactRecord {
                memory_id: format!("memory-{i:04}"),
                canonical_identity: None,
                provenance_family: None,
                vector_bytes: vec![(i % 251) as u8; BinaryHV::BYTES],
                source_text: format!("fact {i}"),
                confidence: 0.5,
                domain: None,
                cycle: i as u64,
                is_causal: false,
            })
            .collect();
        let edges: Vec<CausalEdgeRecord> = (0..1200)
            .map(|i| CausalEdgeRecord {
                cause: format!("cause-{i:04}"),
                effect: format!("effect-{i:04}"),
                strength: 0.5,
                is_inhibitory: false,
                cycle: i as u64,
            })
            .collect();
        let ontology: Vec<OntologyRecord> = (0..1200)
            .map(|i| OntologyRecord {
                name: format!("primitive-{i:04}"),
                vector_bytes: vec![(i % 251) as u8; BinaryHV::BYTES],
                usage_count: 1,
                utility: 0.5,
                created_at_cycle: i as u64,
                last_used_cycle: i as u64,
                is_a_parent: None,
            })
            .collect();

        p.save_snapshot(&facts, &[], &edges, &ontology).unwrap();
        assert_eq!(p.load_facts().unwrap().len(), 1200);
        assert_eq!(p.load_causal_edges().unwrap().len(), 1200);
        assert_eq!(p.load_ontology().unwrap().len(), 1200);

        let reduced_facts: Vec<_> = facts.iter().take(1000).cloned().collect();
        let reduced_edges: Vec<_> = edges.iter().take(1000).cloned().collect();
        let reduced_ontology: Vec<_> = ontology.iter().take(1000).cloned().collect();
        p.save_snapshot(
            &reduced_facts,
            &[],
            &reduced_edges,
            &reduced_ontology,
        )
        .unwrap();

        assert_eq!(p.load_facts().unwrap().len(), 1000);
        assert_eq!(p.load_causal_edges().unwrap().len(), 1000);
        assert_eq!(p.load_ontology().unwrap().len(), 1000);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_snapshot_reconciles_pruned_projection_rows() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_reconcile_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);

        let old_fact = FactRecord {
            memory_id: "old-fact".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![1u8; BinaryHV::BYTES],
            source_text: "old".into(),
            confidence: 0.4,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        let old_edge = CausalEdgeRecord {
            cause: "old-cause".into(),
            effect: "old-effect".into(),
            strength: 0.5,
            is_inhibitory: false,
            cycle: 1,
        };
        let old_ontology = OntologyRecord {
            name: "old-primitive".into(),
            vector_bytes: vec![2u8; BinaryHV::BYTES],
            usage_count: 1,
            utility: 0.2,
            created_at_cycle: 1,
            last_used_cycle: 1,
            is_a_parent: None,
        };
        let historical_relation = ProvenanceRelationRecord {
            source_memory_id: "old-fact".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:1".into(),
        };
        p.save_snapshot(
            &[old_fact],
            std::slice::from_ref(&historical_relation),
            &[old_edge],
            &[old_ontology],
        )
        .unwrap();

        let new_fact = FactRecord {
            memory_id: "new-fact".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![3u8; BinaryHV::BYTES],
            source_text: "new".into(),
            confidence: 0.9,
            domain: None,
            cycle: 2,
            is_causal: false,
        };
        let new_edge = CausalEdgeRecord {
            cause: "new-cause".into(),
            effect: "new-effect".into(),
            strength: 0.8,
            is_inhibitory: true,
            cycle: 2,
        };
        let new_ontology = OntologyRecord {
            name: "new-primitive".into(),
            vector_bytes: vec![4u8; BinaryHV::BYTES],
            usage_count: 3,
            utility: 0.7,
            created_at_cycle: 2,
            last_used_cycle: 3,
            is_a_parent: Some("concept".into()),
        };
        p.save_snapshot(&[new_fact], &[], &[new_edge], &[new_ontology])
            .unwrap();

        let facts = p.load_facts().unwrap();
        let edges = p.load_causal_edges().unwrap();
        let ontology = p.load_ontology().unwrap();
        let relations = p.load_provenance_relations().unwrap();

        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].memory_id, "new-fact");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].cause, "new-cause");
        assert!(edges[0].is_inhibitory);
        assert_eq!(ontology.len(), 1);
        assert_eq!(ontology[0].name, "new-primitive");
        assert_eq!(relations.len(), 1);
        assert_eq!(relations[0].source_memory_id, "old-fact");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_snapshot_rejects_duplicate_identity_keys() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_duplicate_keys_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "duplicate".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0u8; BinaryHV::BYTES],
            source_text: "fact".into(),
            confidence: 0.8,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        let duplicate_fact = fact.clone();
        let edge = CausalEdgeRecord {
            cause: "cause".into(),
            effect: "effect".into(),
            strength: 0.5,
            is_inhibitory: false,
            cycle: 1,
        };
        let duplicate_edge = edge.clone();
        let ontology = OntologyRecord {
            name: "primitive".into(),
            vector_bytes: vec![0u8; BinaryHV::BYTES],
            usage_count: 1,
            utility: 0.5,
            created_at_cycle: 1,
            last_used_cycle: 1,
            is_a_parent: None,
        };
        let duplicate_ontology = ontology.clone();

        let err = p
            .save_snapshot(
                &[fact, duplicate_fact],
                &[],
                &[edge, duplicate_edge],
                &[ontology, duplicate_ontology],
            )
            .unwrap_err();
        assert!(err.contains("duplicate"));
        assert_eq!(p.total_saved(), 0);
        assert!(!db_path.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_snapshot_rolls_back_prior_writes_on_sql_failure() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_atomic_sql_failure_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT UNIQUE,
                    canonical_identity TEXT,
                    provenance_family TEXT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE knowledge_provenance_relations (
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source_memory_id, target_memory_id, kind, created_at)
                );
                CREATE TABLE knowledge_causal_edges (
                    cause TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    strength REAL NOT NULL,
                    is_inhibitory INTEGER NOT NULL DEFAULT 0,
                    cycle INTEGER NOT NULL,
                    PRIMARY KEY (cause, effect)
                );
                CREATE TABLE knowledge_ontology (
                    name TEXT PRIMARY KEY,
                    vector_blob BLOB NOT NULL,
                    usage_count INTEGER NOT NULL,
                    utility REAL NOT NULL,
                    created_at_cycle INTEGER NOT NULL,
                    last_used_cycle INTEGER NOT NULL,
                    is_a_parent TEXT NOT NULL
                );",
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let fact = FactRecord {
            memory_id: "fact-before-failure".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0u8; BinaryHV::BYTES],
            source_text: "must rollback".into(),
            confidence: 0.8,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        let ontology = OntologyRecord {
            name: "fails-at-constraint".into(),
            vector_bytes: vec![0u8; BinaryHV::BYTES],
            usage_count: 1,
            utility: 0.5,
            created_at_cycle: 1,
            last_used_cycle: 1,
            is_a_parent: None,
        };

        let err = p
            .save_snapshot(&[fact], &[], &[], &[ontology])
            .unwrap_err();
        assert!(err.contains("Snapshot ontology"));

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let fact_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM knowledge_facts", [], |row| row.get(0))
            .unwrap();
        let ontology_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM knowledge_ontology", [], |row| row.get(0))
            .unwrap();
        assert_eq!(fact_count, 0);
        assert_eq!(ontology_count, 0);
        assert_eq!(p.total_saved(), 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_snapshot_is_atomic_on_preflight_failure() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_atomic_preflight_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let valid_fact = FactRecord {
            memory_id: "valid".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0u8; BinaryHV::BYTES],
            source_text: "valid".into(),
            confidence: 0.8,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        let invalid_ontology = OntologyRecord {
            name: "invalid".into(),
            vector_bytes: vec![0u8; BinaryHV::BYTES - 1],
            usage_count: 1,
            utility: 0.2,
            created_at_cycle: 1,
            last_used_cycle: 1,
            is_a_parent: None,
        };

        let err = p
            .save_snapshot(&[valid_fact], &[], &[], &[invalid_ontology])
            .unwrap_err();
        assert!(err.contains("OntologyRecord vector_bytes"));
        assert_eq!(p.total_saved(), 0);
        assert!(!db_path.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_facts_preflight_rejects_malformed_batch() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_fact_save_preflight_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let valid = FactRecord {
            memory_id: "valid".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0u8; BinaryHV::BYTES],
            source_text: "valid".into(),
            confidence: 0.8,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        let invalid = FactRecord {
            memory_id: "invalid".into(),
            vector_bytes: vec![0u8; BinaryHV::BYTES - 1],
            ..valid.clone()
        };

        let err = p.save_facts(&[valid, invalid]).unwrap_err();
        assert!(err.contains("vector_bytes"));
        assert_eq!(p.total_saved(), 0);
        assert!(!db_path.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_ontology_preflight_rejects_malformed_batch() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_ontology_save_preflight_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let valid = OntologyRecord {
            name: "valid".into(),
            vector_bytes: vec![0u8; BinaryHV::BYTES],
            usage_count: 1,
            utility: 0.8,
            created_at_cycle: 1,
            last_used_cycle: 1,
            is_a_parent: None,
        };
        let invalid = OntologyRecord {
            name: "invalid".into(),
            vector_bytes: vec![0u8; BinaryHV::BYTES + 1],
            utility: f64::NAN,
            ..valid.clone()
        };

        let err = p.save_ontology(&[valid, invalid]).unwrap_err();
        assert!(err.contains("vector_bytes"));
        assert_eq!(p.total_saved(), 0);
        assert!(!db_path.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_and_load_provenance_relations_append_only() {
        let dir = std::env::temp_dir().join(format!("symthaea_provenance_relation_test_{}", std::process::id()));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);
        let relation = ProvenanceRelationRecord { source_memory_id: "derived".into(), target_memory_id: "source".into(), kind: ProvenanceRelationKind::DerivedFrom, created_at: "cycle:2".into() };
        assert_eq!(p.save_provenance_relations(std::slice::from_ref(&relation)).unwrap(), 1);
        assert_eq!(p.save_provenance_relations(std::slice::from_ref(&relation)).unwrap(), 0);
        let loaded = p.load_provenance_relations().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].kind, ProvenanceRelationKind::DerivedFrom);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_provenance_relations_rolls_back_batch_on_validation_error() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_provenance_transaction_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        let valid = ProvenanceRelationRecord {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let invalid = ProvenanceRelationRecord {
            source_memory_id: " ".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:3".into(),
        };

        let err = p
            .save_provenance_relations(&[valid, invalid])
            .unwrap_err();
        assert!(err.contains("Invalid provenance relation"));
        assert_eq!(p.total_saved(), 0);
        assert!(p.load_provenance_relations().unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_causal_edges_preflight_rejects_invalid_batch() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_causal_save_preflight_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let valid = CausalEdgeRecord {
            cause: "cause".into(),
            effect: "effect".into(),
            strength: 0.5,
            is_inhibitory: false,
            cycle: 1,
        };
        let invalid = CausalEdgeRecord {
            strength: f32::NAN,
            ..valid.clone()
        };

        let err = p.save_causal_edges(&[valid, invalid]).unwrap_err();
        assert!(err.contains("strength"));
        assert_eq!(p.total_saved(), 0);
        assert!(!db_path.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_provenance_relations_preflight_rejects_invalid_batch() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_provenance_save_preflight_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let valid = ProvenanceRelationRecord {
            source_memory_id: "derived".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:2".into(),
        };
        let invalid = ProvenanceRelationRecord {
            source_memory_id: " ".into(),
            ..valid.clone()
        };

        let err = p
            .save_provenance_relations(&[valid, invalid])
            .unwrap_err();
        assert!(err.contains("Invalid provenance relation"));
        assert_eq!(p.total_saved(), 0);
        assert!(!db_path.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_and_load_facts() {
        let dir =
            std::env::temp_dir().join(format!("symthaea_knowledge_test_{}", std::process::id()));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        assert!(p.is_configured());

        let facts = vec![
            FactRecord {
                memory_id: "memory-1".into(),
                canonical_identity: Some("claim-1".into()),
                provenance_family: Some("family-1".into()),
                vector_bytes: vec![0u8; 2048],
                source_text: "Test fact one".into(),
                confidence: 0.9,
                domain: Some("test".into()),
                cycle: 1,
                is_causal: false,
            },
            FactRecord {
                memory_id: "memory-2".into(),
                canonical_identity: None,
                provenance_family: None,
                vector_bytes: vec![1u8; 2048],
                source_text: "Test fact two".into(),
                confidence: 0.8,
                domain: None,
                cycle: 2,
                is_causal: true,
            },
        ];

        let saved = p.save_facts(&facts).unwrap();
        assert_eq!(saved, 2);
        assert_eq!(p.total_saved(), 2);

        let loaded = p.load_facts().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].source_text, "Test fact two"); // DESC order
        assert_eq!(loaded[0].memory_id, "memory-2");
        assert_eq!(loaded[1].memory_id, "memory-1");
        assert_eq!(loaded[1].canonical_identity.as_deref(), Some("claim-1"));
        assert_eq!(loaded[1].source_text, "Test fact one");

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_facts_total_orders_equal_cycles_by_memory_identity() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_equal_cycle_fact_order_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        let facts = vec![
            FactRecord {
                memory_id: "memory-z".into(),
                canonical_identity: None,
                provenance_family: None,
                vector_bytes: vec![0u8; 2048],
                source_text: "later lexical identity".into(),
                confidence: 0.7,
                domain: None,
                cycle: 9,
                is_causal: false,
            },
            FactRecord {
                memory_id: "memory-a".into(),
                canonical_identity: None,
                provenance_family: None,
                vector_bytes: vec![1u8; 2048],
                source_text: "earlier lexical identity".into(),
                confidence: 0.8,
                domain: None,
                cycle: 9,
                is_causal: false,
            },
        ];

        assert_eq!(p.save_facts(&facts).unwrap(), 2);
        let loaded = p.load_facts().unwrap();
        assert_eq!(
            loaded.iter().map(|f| f.memory_id.as_str()).collect::<Vec<_>>(),
            vec!["memory-a", "memory-z"]
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_facts_rejects_empty_memory_identity() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_empty_memory_id_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        let fact = FactRecord {
            memory_id: "   ".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0u8; 2048],
            source_text: "invalid identity".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };

        let err = p.save_facts(&[fact]).unwrap_err();
        assert!(err.contains("memory_id must be non-empty"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_facts_is_idempotent_by_memory_id() {
        let dir = std::env::temp_dir().join(format!("symthaea_fact_upsert_test_{}", std::process::id()));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        let first = FactRecord {
            memory_id: "memory-stable".into(),
            canonical_identity: Some("claim-1".into()),
            provenance_family: Some("family-1".into()),
            vector_bytes: vec![0u8; 2048],
            source_text: "original".into(),
            confidence: 0.8,
            domain: Some("test".into()),
            cycle: 1,
            is_causal: false,
        };
        assert_eq!(p.save_facts(&[first]).unwrap(), 1);

        let updated = FactRecord {
            memory_id: "memory-stable".into(),
            canonical_identity: Some("claim-1".into()),
            provenance_family: Some("family-1".into()),
            vector_bytes: vec![1u8; 2048],
            source_text: "updated".into(),
            confidence: 0.9,
            domain: Some("test".into()),
            cycle: 2,
            is_causal: true,
        };
        assert_eq!(p.save_facts(&[updated]).unwrap(), 1);

        let loaded = p.load_facts().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].memory_id, "memory-stable");
        assert_eq!(loaded[0].source_text, "updated");
        assert!((loaded[0].confidence - 0.9).abs() < 0.001);
        assert_eq!(loaded[0].cycle, 2);
        assert!(loaded[0].is_causal);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_legacy_rows_receive_stable_memory_identity() {
        let dir = std::env::temp_dir().join(format!("symthaea_legacy_fact_test_{}", std::process::id()));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO knowledge_facts (vector_blob, source_text, confidence, domain, cycle, is_causal)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![vec![7u8; 2048], "legacy", 0.7f32, "test", 3i64, false],
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let loaded = p.load_facts().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].memory_id, "legacy-fact:1");

        // Saving the loaded record must update, not duplicate, the migrated row.
        p.save_facts(&loaded).unwrap();
        let loaded_again = p.load_facts().unwrap();
        assert_eq!(loaded_again.len(), 1);
        assert_eq!(loaded_again[0].memory_id, "legacy-fact:1");

        let _ = std::fs::remove_dir_all(&dir);
    }


    #[test]
    fn test_schema_migration_backfills_legacy_memory_identity_atomically() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_legacy_memory_identity_migration_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO knowledge_facts
                 (vector_blob, source_text, confidence, cycle, is_causal)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![vec![0x42u8; BinaryHV::BYTES], "legacy", 0.5f32, 7i64, false],
            )
            .unwrap();
        }

        let mut p=KnowledgePersistence::new(&db_path);
        let conn=p.open_connection().unwrap();
        p.ensure_schema(&conn).unwrap();

        let memory_id:String=conn
            .query_row(
                "SELECT memory_id FROM knowledge_facts WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(memory_id, "legacy-fact:1");

        // A subsequent read does not need a compatibility write; the migration
        // has already materialized the stable identity.
        let before_changes=conn.changes();
        let facts=p.load_facts().unwrap();
        assert_eq!(facts[0].memory_id, "legacy-fact:1");
        assert_eq!(conn.changes(), before_changes);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schema_migration_rejects_blank_legacy_memory_identity() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_legacy_receipt_migration_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let canonical_digest = "a".repeat(64);
        let report_digest = Some("b".repeat(64));

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE knowledge_provenance_relations (
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source_memory_id, target_memory_id, kind, created_at)
                );
                CREATE TABLE knowledge_causal_edges (
                    cause TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    strength REAL NOT NULL,
                    is_inhibitory INTEGER NOT NULL DEFAULT 0,
                    cycle INTEGER NOT NULL,
                    PRIMARY KEY (cause, effect)
                );
                CREATE TABLE knowledge_ontology (
                    name TEXT PRIMARY KEY,
                    vector_blob BLOB NOT NULL,
                    usage_count INTEGER NOT NULL,
                    utility REAL NOT NULL,
                    created_at_cycle INTEGER NOT NULL,
                    last_used_cycle INTEGER NOT NULL,
                    is_a_parent TEXT
                );
                CREATE TABLE knowledge_snapshot_receipts (
                    generation INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_digest_hex TEXT NOT NULL
                );
                CREATE TABLE knowledge_snapshot_validation_receipts (
                    validation_event TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL,
                    snapshot_digest_hex TEXT NOT NULL,
                    validator_ref TEXT NOT NULL,
                    validator_version TEXT NOT NULL,
                    validation_profile TEXT NOT NULL,
                    conforms INTEGER NOT NULL,
                    report_digest_hex TEXT,
                    FOREIGN KEY (generation) REFERENCES knowledge_snapshot_receipts(generation)
                );",
            )
            .unwrap();

            conn.execute(
                "INSERT INTO knowledge_snapshot_receipts
                 (generation, canonical_digest_hex)
                 VALUES (1, ?1)",
                [canonical_digest.as_str()],
            )
            .unwrap();

            conn.execute(
                "INSERT INTO knowledge_snapshot_validation_receipts
                 (validation_event, generation, snapshot_digest_hex, validator_ref,
                  validator_version, validation_profile, conforms, report_digest_hex)
                 VALUES ('legacy-validation', 1, ?1, 'legacy-validator', 'v1',
                         'legacy-profile', 1, ?2)",
                rusqlite::params![canonical_digest.as_str(), report_digest.as_deref()],
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        {
            let conn = p.open_connection().unwrap();
            p.ensure_schema(&conn).unwrap();

            let snapshot_receipt: (String,) = conn
                .query_row(
                    "SELECT receipt_digest_hex
                     FROM knowledge_snapshot_receipts
                     WHERE generation = 1",
                    [],
                    |row| Ok((row.get(0)?,)),
                )
                .unwrap();
            let expected_snapshot_receipt = KnowledgeSnapshotReceipt {
                generation: 1,
                canonical_digest_hex: canonical_digest.clone(),
                receipt_digest_hex: String::new(),
            }
            .canonical_receipt_digest_hex();
            assert_eq!(snapshot_receipt.0, expected_snapshot_receipt);

            let validation_row: (i64, String) = conn
                .query_row(
                    "SELECT validation_sequence, receipt_digest_hex
                     FROM knowledge_snapshot_validation_receipts
                     WHERE validation_event = 'legacy-validation'",
                    [],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .unwrap();
            assert_eq!(validation_row.0, 1);

            let expected_validation_receipt =
                KnowledgeSnapshotValidationReceipt {
                    validation_event: "legacy-validation".into(),
                    generation: 1,
                    snapshot_digest_hex: canonical_digest.clone(),
                    validator_ref: "legacy-validator".into(),
                    validator_version: "v1".into(),
                    validation_profile: "legacy-profile".into(),
                    conforms: true,
                    report_digest_hex: report_digest.clone(),
                }
                .canonical_digest_hex_for_sequence(1);
            assert_eq!(validation_row.1, expected_validation_receipt);

            let snapshot_receipt_schema: Vec<String> = conn
                .prepare("PRAGMA table_info(knowledge_snapshot_receipts)")
                .unwrap()
                .query_map([], |row| row.get(1))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert!(snapshot_receipt_schema
                .iter()
                .any(|column| column == "receipt_digest_hex"));

            let validation_schema: Vec<String> = conn
                .prepare("PRAGMA table_info(knowledge_snapshot_validation_receipts)")
                .unwrap()
                .query_map([], |row| row.get(1))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            assert!(validation_schema
                .iter()
                .any(|column| column == "validation_sequence"));
            assert!(validation_schema
                .iter()
                .any(|column| column == "receipt_digest_hex"));
        }

        // A second persistence instance must see the already-migrated schema as stable.
        let mut second = KnowledgePersistence::new(&db_path);
        second.verify_snapshot_validation_receipts().unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }


    #[test]
    fn test_schema_migration_rolls_back_on_late_constraint_failure() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_migration_rollback_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE knowledge_provenance_relations (
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source_memory_id, target_memory_id, kind, created_at)
                );
                CREATE TABLE knowledge_causal_edges (
                    cause TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    strength REAL NOT NULL,
                    is_inhibitory INTEGER NOT NULL DEFAULT 0,
                    cycle INTEGER NOT NULL,
                    PRIMARY KEY (cause, effect)
                );
                CREATE TABLE knowledge_ontology (
                    name TEXT PRIMARY KEY,
                    vector_blob BLOB NOT NULL,
                    usage_count INTEGER NOT NULL,
                    utility REAL NOT NULL,
                    created_at_cycle INTEGER NOT NULL,
                    last_used_cycle INTEGER NOT NULL,
                    is_a_parent TEXT
                );
                CREATE TABLE knowledge_snapshot_receipts (
                    generation INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_digest_hex TEXT NOT NULL
                );
                CREATE TABLE knowledge_snapshot_validation_receipts (
                    validation_event TEXT PRIMARY KEY,
                    validation_sequence INTEGER,
                    generation INTEGER NOT NULL,
                    snapshot_digest_hex TEXT NOT NULL,
                    validator_ref TEXT NOT NULL,
                    validator_version TEXT NOT NULL,
                    validation_profile TEXT NOT NULL,
                    conforms INTEGER NOT NULL,
                    report_digest_hex TEXT,
                    FOREIGN KEY (generation) REFERENCES knowledge_snapshot_receipts(generation)
                );",
            )
            .unwrap();

            let canonical_digest = "c".repeat(64);
            conn.execute(
                "INSERT INTO knowledge_snapshot_receipts (generation, canonical_digest_hex)
                 VALUES (1, ?1)",
                [canonical_digest.as_str()],
            )
            .unwrap();
            conn.execute_batch(
                "INSERT INTO knowledge_snapshot_validation_receipts
                    (validation_event, validation_sequence, generation, snapshot_digest_hex,
                     validator_ref, validator_version, validation_profile, conforms)
                 VALUES
                    ('legacy-a', 1, 1, 'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
                     'validator', 'v1', 'profile', 1),
                    ('legacy-b', 1, 1, 'cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc',
                     'validator', 'v1', 'profile', 1);",
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let err = {
            let conn = p.open_connection().unwrap();
            p.ensure_schema(&conn).unwrap_err()
        };
        assert!(err.contains("Schema validation sequence index"));

        let conn = rusqlite::Connection::open(&db_path).unwrap();

        let fact_columns: Vec<String> = conn
            .prepare("PRAGMA table_info(knowledge_facts)")
            .unwrap()
            .query_map([], |row| row.get(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(!fact_columns.iter().any(|column| column == "memory_id"));
        assert!(!fact_columns.iter().any(|column| column == "canonical_identity"));
        assert!(!fact_columns.iter().any(|column| column == "provenance_family"));

        let snapshot_receipt_columns: Vec<String> = conn
            .prepare("PRAGMA table_info(knowledge_snapshot_receipts)")
            .unwrap()
            .query_map([], |row| row.get(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(!snapshot_receipt_columns
            .iter()
            .any(|column| column == "receipt_digest_hex"));

        let validation_columns: Vec<String> = conn
            .prepare("PRAGMA table_info(knowledge_snapshot_validation_receipts)")
            .unwrap()
            .query_map([], |row| row.get(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(validation_columns
            .iter()
            .any(|column| column == "validation_sequence"));
        assert!(!validation_columns
            .iter()
            .any(|column| column == "receipt_digest_hex"));

        let user_version: i64 = conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(user_version, 0);

        let null_receipt_digests: i64 = conn
            .query_row(
                "SELECT COUNT(*)
                 FROM sqlite_schema
                 WHERE type = 'index'
                   AND name = 'idx_snapshot_validation_receipts_sequence_unique'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(null_receipt_digests, 0);

        // The failed transaction must leave the legacy database retryable.
        let mut retry = KnowledgePersistence::new(&db_path);
        {
            let conn = retry.open_connection().unwrap();
            // Repair the deliberately inconsistent partial-migration state.
            conn.execute(
                "DELETE FROM knowledge_snapshot_validation_receipts
                 WHERE validation_event = 'legacy-b'",
                [],
            )
            .unwrap();
        }
        let retry_conn = retry.open_connection().unwrap();
        retry.ensure_schema(&retry_conn).unwrap();
        retry.verify_snapshot_validation_receipts().unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }



    #[test]
    fn test_snapshot_receipt_rejects_malformed_canonical_digest_even_with_valid_self_digest() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_receipt_digest_format_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        let fact = FactRecord {
            memory_id: "digest-format".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x11; BinaryHV::BYTES],
            source_text: "digest format".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();

        {
            let conn = p.open_connection().unwrap();
            let tampered = KnowledgeSnapshotReceipt {
                generation: 1,
                canonical_digest_hex: "not-a-digest".into(),
                receipt_digest_hex: String::new(),
            };
            let self_digest = tampered.canonical_receipt_digest_hex();
            conn.execute(
                "DROP TRIGGER trg_knowledge_snapshot_receipts_no_update;
                 UPDATE knowledge_snapshot_receipts
                 SET canonical_digest_hex = ?1, receipt_digest_hex = ?2
                 WHERE generation = 1;",
                rusqlite::params!["not-a-digest", self_digest],
            )
            .unwrap();
        }

        let err = p.latest_snapshot_receipt().unwrap_err();
        assert!(err.contains("canonical digest is not a 64-character hexadecimal digest"));

        let _ = std::fs::remove_dir_all(&dir);
    }


    #[test]
    fn test_schema_migration_upgrades_sequence_bound_digest_behind_existing_append_only_trigger() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_validation_digest_v1_trigger_migration_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let canonical_digest = "c".repeat(64);
        let legacy_receipt = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:legacy-v1".into(),
            generation: 1,
            snapshot_digest_hex: canonical_digest.clone(),
            validator_ref: "legacy-validator".into(),
            validator_version: "v1".into(),
            validation_profile: "legacy-profile".into(),
            conforms: true,
            report_digest_hex: None,
        };
        let legacy_digest = legacy_receipt.legacy_canonical_digest_hex();

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    canonical_identity TEXT,
                    provenance_family TEXT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE knowledge_provenance_relations (
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source_memory_id, target_memory_id, kind, created_at)
                );
                CREATE TABLE knowledge_causal_edges (
                    cause TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    strength REAL NOT NULL,
                    is_inhibitory INTEGER NOT NULL DEFAULT 0,
                    cycle INTEGER NOT NULL,
                    PRIMARY KEY (cause, effect)
                );
                CREATE TABLE knowledge_ontology (
                    name TEXT PRIMARY KEY,
                    vector_blob BLOB NOT NULL,
                    usage_count INTEGER NOT NULL,
                    utility REAL NOT NULL,
                    created_at_cycle INTEGER NOT NULL,
                    last_used_cycle INTEGER NOT NULL,
                    is_a_parent TEXT
                );
                CREATE TABLE knowledge_snapshot_receipts (
                    generation INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_digest_hex TEXT NOT NULL,
                    receipt_digest_hex TEXT NOT NULL CHECK (length(receipt_digest_hex) = 64)
                );
                CREATE TABLE knowledge_snapshot_validation_receipts (
                    validation_event TEXT PRIMARY KEY,
                    validation_sequence INTEGER NOT NULL UNIQUE CHECK (validation_sequence > 0),
                    generation INTEGER NOT NULL,
                    snapshot_digest_hex TEXT NOT NULL,
                    validator_ref TEXT NOT NULL,
                    validator_version TEXT NOT NULL,
                    validation_profile TEXT NOT NULL,
                    conforms INTEGER NOT NULL CHECK (conforms IN (0, 1)),
                    report_digest_hex TEXT,
                    receipt_digest_hex TEXT NOT NULL CHECK (length(receipt_digest_hex) = 64),
                    FOREIGN KEY (generation) REFERENCES knowledge_snapshot_receipts(generation)
                );
                CREATE INDEX idx_snapshot_validation_receipts_generation
                    ON knowledge_snapshot_validation_receipts(generation, validation_event);
                CREATE INDEX idx_facts_domain ON knowledge_facts(domain);
                CREATE INDEX idx_facts_cycle ON knowledge_facts(cycle);
                CREATE UNIQUE INDEX idx_facts_memory_id_unique
                    ON knowledge_facts(memory_id);
                CREATE UNIQUE INDEX idx_snapshot_validation_receipts_sequence_unique
                    ON knowledge_snapshot_validation_receipts(validation_sequence);
                CREATE TRIGGER trg_knowledge_snapshot_validation_receipts_no_update
                BEFORE UPDATE ON knowledge_snapshot_validation_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'knowledge_snapshot_validation_receipts is append-only: UPDATE prohibited');
                END;
                CREATE TRIGGER trg_knowledge_snapshot_validation_receipts_no_delete
                BEFORE DELETE ON knowledge_snapshot_validation_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'knowledge_snapshot_validation_receipts is append-only: DELETE prohibited');
                END;",
            )
            .unwrap();

            conn.execute(
                "INSERT INTO knowledge_snapshot_receipts
                 (generation, canonical_digest_hex, receipt_digest_hex)
                 VALUES (1, ?1, ?2)",
                rusqlite::params![
                    canonical_digest.as_str(),
                    KnowledgeSnapshotReceipt {
                        generation: 1,
                        canonical_digest_hex: canonical_digest.clone(),
                        receipt_digest_hex: String::new(),
                    }
                    .canonical_receipt_digest_hex(),
                ],
            )
            .unwrap();

            conn.execute(
                "INSERT INTO knowledge_snapshot_validation_receipts
                 (validation_event, validation_sequence, generation, snapshot_digest_hex,
                  validator_ref, validator_version, validation_profile, conforms,
                  report_digest_hex, receipt_digest_hex)
                 VALUES (?1, 1, 1, ?2, ?3, ?4, ?5, 1, NULL, ?6)",
                rusqlite::params![
                    legacy_receipt.validation_event.as_str(),
                    legacy_receipt.snapshot_digest_hex.as_str(),
                    legacy_receipt.validator_ref.as_str(),
                    legacy_receipt.validator_version.as_str(),
                    legacy_receipt.validation_profile.as_str(),
                    legacy_digest.as_str(),
                ],
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let conn = p.open_connection().unwrap();
        p.ensure_schema(&conn).unwrap();

        let stored_digest: String = conn
            .query_row(
                "SELECT receipt_digest_hex
                 FROM knowledge_snapshot_validation_receipts
                 WHERE validation_event = 'validation:legacy-v1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            stored_digest,
            legacy_receipt.canonical_digest_hex_for_sequence(1)
        );

        let blocked = conn
            .execute(
                "UPDATE knowledge_snapshot_validation_receipts
                 SET validator_version = 'blocked'
                 WHERE validation_event = 'validation:legacy-v1'",
                [],
            )
            .unwrap_err();
        assert!(blocked.to_string().contains("UPDATE prohibited"));

        let tx = conn.unchecked_transaction().unwrap();
        verify_snapshot_receipts_in_tx(&tx).unwrap();
        verify_snapshot_validation_receipts_in_tx(&tx).unwrap();
        tx.commit().unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schema_migration_rejects_blank_or_whitespace_legacy_memory_identity() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_blank_legacy_memory_identity_migration_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );",
            )
            .unwrap();
            conn.execute(
                "INSERT INTO knowledge_facts
                 (memory_id, vector_blob, source_text, confidence, cycle, is_causal)
                 VALUES ('   ', ?1, 'blank identity', 0.5, 1, 0)",
                [vec![0x55u8; BinaryHV::BYTES]],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO knowledge_facts
                 (memory_id, vector_blob, source_text, confidence, cycle, is_causal)
                 VALUES (?1, ?2, 'tab identity', 0.5, 2, 0)",
                rusqlite::params!["\t\n", vec![0x66u8; BinaryHV::BYTES]],
            )
            .unwrap();
        }

        let mut p=KnowledgePersistence::new(&db_path);
        let conn=p.open_connection().unwrap();
        let err=p.ensure_schema(&conn).unwrap_err();
        assert!(err.contains("blank memory_id"));

        // The transaction rollback must leave the original legacy row untouched.
        let preserved:String=conn
            .query_row(
                "SELECT memory_id FROM knowledge_facts WHERE id = 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(preserved, "   ");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schema_migration_rejects_malformed_legacy_validation_metadata() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_validation_metadata_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE knowledge_provenance_relations (
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source_memory_id, target_memory_id, kind, created_at)
                );
                CREATE TABLE knowledge_causal_edges (
                    cause TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    strength REAL NOT NULL,
                    is_inhibitory INTEGER NOT NULL DEFAULT 0,
                    cycle INTEGER NOT NULL,
                    PRIMARY KEY (cause, effect)
                );
                CREATE TABLE knowledge_ontology (
                    name TEXT PRIMARY KEY,
                    vector_blob BLOB NOT NULL,
                    usage_count INTEGER NOT NULL,
                    utility REAL NOT NULL,
                    created_at_cycle INTEGER NOT NULL,
                    last_used_cycle INTEGER NOT NULL,
                    is_a_parent TEXT
                );
                CREATE TABLE knowledge_snapshot_receipts (
                    generation INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_digest_hex TEXT NOT NULL
                );
                CREATE TABLE knowledge_snapshot_validation_receipts (
                    validation_event TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL,
                    snapshot_digest_hex TEXT NOT NULL,
                    validator_ref TEXT NOT NULL,
                    validator_version TEXT NOT NULL,
                    validation_profile TEXT NOT NULL,
                    conforms INTEGER NOT NULL,
                    report_digest_hex TEXT,
                    FOREIGN KEY (generation) REFERENCES knowledge_snapshot_receipts(generation)
                );
                INSERT INTO knowledge_snapshot_receipts
                    (generation, canonical_digest_hex)
                VALUES
                    (1, 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee');
                INSERT INTO knowledge_snapshot_validation_receipts
                    (validation_event, generation, snapshot_digest_hex, validator_ref,
                     validator_version, validation_profile, conforms)
                VALUES
                    ('malformed', 1,
                     'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee',
                     '', 'v1', 'legacy-profile', 1);",
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let err = {
            let conn = p.open_connection().unwrap();
            p.ensure_schema(&conn).unwrap_err()
        };
        assert!(err.contains("Invalid persisted snapshot validation receipt"));
        assert!(err.contains("validator reference must be non-empty"));

        // Migration must be atomic: the new receipt self-digest column must not be committed.
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let validation_columns: Vec<String> = conn
            .prepare("PRAGMA table_info(knowledge_snapshot_validation_receipts)")
            .unwrap()
            .query_map([], |row| row.get(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(!validation_columns
            .iter()
            .any(|column| column == "receipt_digest_hex"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_schema_migration_rejects_legacy_foreign_key_violation() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_schema_fk_check_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE knowledge_provenance_relations (
                    source_memory_id TEXT NOT NULL,
                    target_memory_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (source_memory_id, target_memory_id, kind, created_at)
                );
                CREATE TABLE knowledge_causal_edges (
                    cause TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    strength REAL NOT NULL,
                    is_inhibitory INTEGER NOT NULL DEFAULT 0,
                    cycle INTEGER NOT NULL,
                    PRIMARY KEY (cause, effect)
                );
                CREATE TABLE knowledge_ontology (
                    name TEXT PRIMARY KEY,
                    vector_blob BLOB NOT NULL,
                    usage_count INTEGER NOT NULL,
                    utility REAL NOT NULL,
                    created_at_cycle INTEGER NOT NULL,
                    last_used_cycle INTEGER NOT NULL,
                    is_a_parent TEXT
                );
                CREATE TABLE knowledge_snapshot_receipts (
                    generation INTEGER PRIMARY KEY AUTOINCREMENT,
                    canonical_digest_hex TEXT NOT NULL
                );
                CREATE TABLE knowledge_snapshot_validation_receipts (
                    validation_event TEXT PRIMARY KEY,
                    generation INTEGER NOT NULL,
                    snapshot_digest_hex TEXT NOT NULL,
                    validator_ref TEXT NOT NULL,
                    validator_version TEXT NOT NULL,
                    validation_profile TEXT NOT NULL,
                    conforms INTEGER NOT NULL,
                    report_digest_hex TEXT,
                    FOREIGN KEY (generation) REFERENCES knowledge_snapshot_receipts(generation)
                );",
            )
            .unwrap();

            // This orphan is deliberately inserted with FK enforcement disabled,
            // simulating a legacy database written before the per-connection PRAGMA.
            conn.execute(
                "INSERT INTO knowledge_snapshot_validation_receipts
                 (validation_event, generation, snapshot_digest_hex, validator_ref,
                  validator_version, validation_profile, conforms)
                 VALUES ('orphan', 99, ?1, 'legacy-validator', 'v1', 'legacy-profile', 1)",
                ["d".repeat(64).as_str()],
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let err = {
            let conn = p.open_connection().unwrap();
            p.ensure_schema(&conn).unwrap_err()
        };
        assert!(err.contains("Schema foreign-key check failed"));
        assert!(err.contains("1 violation"));

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        let snapshot_receipt_columns: Vec<String> = conn
            .prepare("PRAGMA table_info(knowledge_snapshot_receipts)")
            .unwrap()
            .query_map([], |row| row.get(1))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(!snapshot_receipt_columns
            .iter()
            .any(|column| column == "receipt_digest_hex"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_provenance_relations_total_orders_equal_timestamps_by_kind() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_provenance_equal_timestamp_order_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);
        let relations = [
            ProvenanceRelationRecord {
                source_memory_id: "source".into(),
                target_memory_id: "target".into(),
                kind: ProvenanceRelationKind::Corroborates,
                created_at: "cycle:7".into(),
            },
            ProvenanceRelationRecord {
                source_memory_id: "source".into(),
                target_memory_id: "target".into(),
                kind: ProvenanceRelationKind::Contradicts,
                created_at: "cycle:7".into(),
            },
        ];
        assert_eq!(p.save_provenance_relations(&relations).unwrap(), 2);
        let loaded = p.load_provenance_relations().unwrap();
        assert_eq!(
            loaded.iter().map(|r| r.kind).collect::<Vec<_>>(),
            vec![ProvenanceRelationKind::Contradicts, ProvenanceRelationKind::Corroborates]
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_causal_edges_total_orders_equal_cycles_by_endpoints() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_causal_equal_cycle_order_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);
        let edges = [
            CausalEdgeRecord { cause: "z".into(), effect: "effect".into(), strength: 0.8, is_inhibitory: false, cycle: 4 },
            CausalEdgeRecord { cause: "a".into(), effect: "effect".into(), strength: 0.7, is_inhibitory: false, cycle: 4 },
        ];
        assert_eq!(p.save_causal_edges(&edges).unwrap(), 2);
        let loaded = p.load_causal_edges().unwrap();
        assert_eq!(loaded.iter().map(|e| e.cause.as_str()).collect::<Vec<_>>(), vec!["a", "z"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_rejects_strength_metadata_sign_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_causal_sign_mismatch_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let edge = CausalEdgeRecord {
            cause: "cause".into(),
            effect: "effect".into(),
            strength: 0.8,
            is_inhibitory: true,
            cycle: 1,
        };
        let error = p
            .save_snapshot(&[], &[], &[edge], &[])
            .unwrap_err();
        assert!(error.contains("strength sign must match is_inhibitory"));

        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn test_load_causal_edges_rejects_strength_metadata_sign_mismatch() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_causal_sign_mismatch_load_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_causal_edges (
                    cause TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    strength REAL NOT NULL,
                    is_inhibitory INTEGER NOT NULL DEFAULT 0,
                    cycle INTEGER NOT NULL,
                    PRIMARY KEY (cause, effect)
                );
                INSERT INTO knowledge_causal_edges
                    (cause, effect, strength, is_inhibitory, cycle)
                VALUES ('cause', 'effect', 0.8, 1, 1);",
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let error = p.load_causal_edges().unwrap_err();
        assert!(error.contains("strength sign must match is_inhibitory"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_causal_edges_rejects_out_of_range_strength() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_causal_invalid_strength_load_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        {
            let conn = rusqlite::Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE knowledge_causal_edges (
                    cause TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    strength REAL NOT NULL,
                    is_inhibitory INTEGER NOT NULL DEFAULT 0,
                    cycle INTEGER NOT NULL,
                    PRIMARY KEY (cause, effect)
                );
                INSERT INTO knowledge_causal_edges
                    (cause, effect, strength, is_inhibitory, cycle)
                VALUES ('cause', 'effect', 1.5, 0, 1);",
            )
            .unwrap();
        }

        let mut p = KnowledgePersistence::new(&db_path);
        let error = p.load_causal_edges().unwrap_err();
        assert!(error.contains("strength must be finite and in [-1, 1]"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_ontology_total_orders_equal_utility_by_name() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_ontology_equal_utility_order_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);
        let records = [
            OntologyRecord { name: "zeta".into(), vector_bytes: vec![1u8; 2048], usage_count: 1, utility: 0.5, created_at_cycle: 2, last_used_cycle: 3, is_a_parent: None },
            OntologyRecord { name: "alpha".into(), vector_bytes: vec![0u8; 2048], usage_count: 1, utility: 0.5, created_at_cycle: 1, last_used_cycle: 2, is_a_parent: None },
        ];
        assert_eq!(p.save_ontology(&records).unwrap(), 2);
        let loaded = p.load_ontology().unwrap();
        assert_eq!(loaded.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), vec!["alpha", "zeta"]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_causal_edge_upsert_updates_existing_identity() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_causal_upsert_identity_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let initial = CausalEdgeRecord {
            cause: "cause".into(),
            effect: "effect".into(),
            strength: 0.2,
            is_inhibitory: false,
            cycle: 1,
        };
        let updated = CausalEdgeRecord {
            cause: "cause".into(),
            effect: "effect".into(),
            strength: -0.8,
            is_inhibitory: true,
            cycle: 7,
        };

        assert_eq!(p.save_causal_edges(std::slice::from_ref(&initial)).unwrap(), 1);
        assert_eq!(p.save_causal_edges(std::slice::from_ref(&updated)).unwrap(), 1);

        let loaded = p.load_causal_edges().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].cause, "cause");
        assert_eq!(loaded[0].effect, "effect");
        assert_eq!(loaded[0].strength, -0.8);
        assert!(loaded[0].is_inhibitory);
        assert_eq!(loaded[0].cycle, 7);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_ontology_upsert_updates_existing_identity() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_ontology_upsert_identity_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let initial = OntologyRecord {
            name: "stable".into(),
            vector_bytes: vec![1u8; 2048],
            usage_count: 1,
            utility: 0.2,
            created_at_cycle: 1,
            last_used_cycle: 1,
            is_a_parent: None,
        };
        let updated = OntologyRecord {
            name: "stable".into(),
            vector_bytes: vec![2u8; 2048],
            usage_count: 9,
            utility: 0.8,
            created_at_cycle: 4,
            last_used_cycle: 7,
            is_a_parent: Some("concept".into()),
        };

        assert_eq!(p.save_ontology(std::slice::from_ref(&initial)).unwrap(), 1);
        assert_eq!(p.save_ontology(std::slice::from_ref(&updated)).unwrap(), 1);

        let loaded = p.load_ontology().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].name, "stable");
        assert_eq!(loaded[0].vector_bytes, vec![2u8; 2048]);
        assert_eq!(loaded[0].usage_count, 9);
        assert_eq!(loaded[0].utility, 0.8);
        assert_eq!(loaded[0].created_at_cycle, 4);
        assert_eq!(loaded[0].last_used_cycle, 7);
        assert_eq!(loaded[0].is_a_parent.as_deref(), Some("concept"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_snapshot_rejects_duplicate_fact_identity_before_reconciliation() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_duplicate_fact_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "duplicate".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x11; BinaryHV::BYTES],
            source_text: "duplicate".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };

        let err = p
            .save_snapshot(&[fact.clone(), fact], &[], &[], &[])
            .unwrap_err();
        assert_eq!(err, "Snapshot contains duplicate FactRecord memory_id");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_snapshot_rejects_duplicate_provenance_identity_before_reconciliation() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_duplicate_provenance_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let relation = ProvenanceRelationRecord {
            source_memory_id: "source".into(),
            target_memory_id: "target".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:1".into(),
        };

        let err = p
            .save_snapshot(&[], &[relation.clone(), relation], &[], &[])
            .unwrap_err();
        assert_eq!(
            err,
            "Snapshot contains duplicate ProvenanceRelationRecord key"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }


    #[test]
    fn test_load_snapshot_with_receipt_atomically_binds_digest() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_verified_snapshot_read_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "verified-fact".into(),
            canonical_identity: Some("verified-canonical".into()),
            provenance_family: Some("verified-family".into()),
            vector_bytes: vec![0x33; BinaryHV::BYTES],
            source_text: "verified".into(),
            confidence: 0.75,
            domain: Some("test".into()),
            cycle: 3,
            is_causal: false,
        };

        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();

        let (snapshot, receipt) = p.load_snapshot_with_receipt().unwrap().unwrap();
        assert_eq!(receipt.generation, 1);
        assert_eq!(receipt.canonical_digest_hex, snapshot.canonical_digest_hex());
        assert_eq!(receipt.receipt_digest_hex.len(), 64);
        assert_eq!(
            receipt.receipt_digest_hex,
            receipt.canonical_receipt_digest_hex()
        );

        // Provenance is append-only, so omitting an existing historical relation
        // must not silently admit a narrower snapshot as the current committed state.
        let historical = ProvenanceRelationRecord {
            source_memory_id: "verified-fact".into(),
            target_memory_id: "historical-source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:0".into(),
        };
        p.save_provenance_relations(std::slice::from_ref(&historical))
            .unwrap();

        let err = p
            .save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap_err();
        assert!(err.starts_with(
            "Snapshot reconciliation digest mismatch: requested "
        ));

        let (still_valid, still_receipt) =
            p.load_snapshot_with_receipt().unwrap().unwrap();
        assert_eq!(still_receipt.generation, receipt.generation);
        assert_eq!(
            still_valid.canonical_digest_hex(),
            still_receipt.canonical_digest_hex
        );

        // A complete-snapshot receipt becomes stale if a legacy individual-domain
        // API mutates the projection afterwards. The verified API must detect that
        // rather than pairing the new graph with the old admission event.
        let changed = FactRecord {
            source_text: "mutated after receipt".into(),
            ..fact
        };
        p.save_facts(std::slice::from_ref(&changed)).unwrap();

        let err = p.load_snapshot_with_receipt().unwrap_err();
        assert!(err.starts_with("Snapshot receipt digest mismatch: generation 1"));

        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn test_snapshot_receipt_self_digest_binds_generation_and_append_only_storage() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_receipt_integrity_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "snapshot-integrity".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x33; BinaryHV::BYTES],
            source_text: "snapshot integrity".into(),
            confidence: 0.7,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();

        let receipt = p.latest_snapshot_receipt().unwrap().unwrap();
        assert_eq!(receipt.generation, 1);
        assert_eq!(
            receipt.receipt_digest_hex,
            receipt.canonical_receipt_digest_hex()
        );

        let conn = p.open_connection().unwrap();
        let blocked_update = conn
            .execute(
                "UPDATE knowledge_snapshot_receipts
                 SET generation = generation + 10
                 WHERE generation = 1",
                [],
            )
            .unwrap_err();
        assert!(blocked_update.to_string().contains("UPDATE prohibited"));

        let blocked_delete = conn
            .execute(
                "DELETE FROM knowledge_snapshot_receipts WHERE generation = 1",
                [],
            )
            .unwrap_err();
        assert!(blocked_delete.to_string().contains("DELETE prohibited"));

        // Simulate privileged database tampering by removing the UPDATE guard.
        conn.execute_batch(
            "DROP TRIGGER trg_knowledge_snapshot_receipts_no_update;
             UPDATE knowledge_snapshot_receipts
             SET generation = 11
             WHERE generation = 1;",
        )
        .unwrap();

        let err = p.latest_snapshot_receipt().unwrap_err();
        assert_eq!(
            err,
            "Snapshot receipt self-digest mismatch: generation 11"
        );

        let err = p.load_snapshot_with_receipt().unwrap_err();
        assert_eq!(
            err,
            "Snapshot receipt self-digest mismatch: generation 11"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_receipt_history_exports_all_generations() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_receipt_history_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let first = FactRecord {
            memory_id: "snapshot-history-one".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x91; BinaryHV::BYTES],
            source_text: "snapshot history one".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&first), &[], &[], &[])
            .unwrap();

        let second = FactRecord {
            memory_id: "snapshot-history-two".into(),
            source_text: "snapshot history two".into(),
            cycle: 2,
            ..first
        };
        p.save_snapshot(std::slice::from_ref(&second), &[], &[], &[])
            .unwrap();

        let history = p.snapshot_receipt_history().unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(
            history
                .iter()
                .map(|receipt| receipt.generation)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert!(history.iter().all(KnowledgeSnapshotReceipt::verify_self_digest));
        let history_digest =
            KnowledgeSnapshotReceipt::canonical_history_digest_hex(&history);
        assert_eq!(
            p.snapshot_receipt_history_digest_hex().unwrap(),
            history_digest
        );

        let checkpoint = p.snapshot_receipt_history_checkpoint().unwrap();
        assert_eq!(checkpoint.receipt_count, 2);
        assert_eq!(checkpoint.latest_generation, 2);
        assert!(checkpoint.verify_against_history(&history));
        p.verify_snapshot_receipt_history_checkpoint(&checkpoint)
            .unwrap();

        let prefix_checkpoint =
            KnowledgeSnapshotReceiptHistoryCheckpoint::from_history(&history[..1]);
        assert!(prefix_checkpoint.verify_prefix_against_history(&history));
        p.verify_snapshot_receipt_history_checkpoint_prefix(&prefix_checkpoint)
            .unwrap();

        let mut reordered = history.clone();
        reordered.swap(0, 1);
        assert_ne!(
            KnowledgeSnapshotReceipt::canonical_history_digest_hex(&reordered),
            history_digest
        );
        assert!(!checkpoint.verify_against_history(&reordered));

        let mut truncated = history.clone();
        truncated.pop();
        assert!(!checkpoint.verify_against_history(&truncated));

        let mut substituted = history.clone();
        substituted[1].canonical_digest_hex =
            "e000000000000000000000000000000000000000000000000000000000000000".into();
        let substituted_receipt_digest = substituted[1].canonical_receipt_digest_hex();
        substituted[1].receipt_digest_hex = substituted_receipt_digest;
        assert!(substituted[1].verify_self_digest());
        assert!(substituted[1].verify_integrity());
        // A self-consistent same-height substitution is rejected by the
        // externally retained checkpoint.
        assert!(!checkpoint.verify_against_history(&substituted));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_receipt_self_digest_is_externally_recomputable() {
        let receipt = KnowledgeSnapshotReceipt {
            generation: 7,
            canonical_digest_hex: "a".repeat(64),
            receipt_digest_hex: String::new(),
        };
        let digest = receipt.canonical_receipt_digest_hex();
        let persisted = KnowledgeSnapshotReceipt {
            receipt_digest_hex: digest.clone(),
            ..receipt
        };
        assert_eq!(persisted.recomputed_receipt_digest_hex(), digest);
        assert!(persisted.verify_self_digest());

        let tampered = KnowledgeSnapshotReceipt {
            generation: 8,
            ..persisted
        };
        assert!(!tampered.verify_self_digest());

        let malformed = KnowledgeSnapshotReceipt {
            canonical_digest_hex: "not-a-digest".into(),
            receipt_digest_hex: String::new(),
            ..persisted
        };
        let malformed = KnowledgeSnapshotReceipt {
            receipt_digest_hex: malformed.canonical_receipt_digest_hex(),
            ..malformed
        };
        assert!(malformed.verify_self_digest());
        assert!(!malformed.verify_integrity());
    }

    #[test]
    fn test_snapshot_receipt_history_rejects_generation_gaps() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_receipt_gap_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "gap-1".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x12; BinaryHV::BYTES],
            source_text: "gap".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();
        let second = FactRecord {
            memory_id: "gap-2".into(),
            source_text: "gap two".into(),
            cycle: 2,
            ..fact
        };
        p.save_snapshot(std::slice::from_ref(&second), &[], &[], &[])
            .unwrap();
        assert_eq!(p.latest_snapshot_receipt().unwrap().unwrap().generation, 2);

        let conn = p.open_connection().unwrap();
        conn.execute_batch(
            "DROP TRIGGER trg_knowledge_snapshot_receipts_no_delete;
             DELETE FROM knowledge_snapshot_receipts WHERE generation = 1;",
        )
        .unwrap();

        let err = p.latest_snapshot_receipt().unwrap_err();
        assert_eq!(
            err,
            "Snapshot receipt generation discontinuity: expected 1, observed 2"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_latest_snapshot_receipt_fails_closed_when_live_projection_drifts() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_latest_receipt_live_drift_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "latest-live-drift".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x21; BinaryHV::BYTES],
            source_text: "original".into(),
            confidence: 0.6,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();
        assert!(p.latest_snapshot_receipt().unwrap().is_some());

        let changed = FactRecord {
            source_text: "mutated outside snapshot boundary".into(),
            ..fact
        };
        p.save_facts(std::slice::from_ref(&changed)).unwrap();

        let err = p.latest_snapshot_receipt().unwrap_err();
        assert!(err.starts_with("Snapshot receipt digest mismatch: generation 1"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_record_snapshot_validation_binds_current_generation_and_digest() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_validation_receipt_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "validated-fact".into(),
            canonical_identity: None,
            provenance_family: Some("validation-family".into()),
            vector_bytes: vec![0x44; BinaryHV::BYTES],
            source_text: "validated".into(),
            confidence: 0.8,
            domain: None,
            cycle: 5,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();

        let committed = p.latest_snapshot_receipt().unwrap().unwrap();
        let validation = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:event-1".into(),
            generation: committed.generation,
            snapshot_digest_hex: committed.canonical_digest_hex.clone(),
            validator_ref: "validator:structural-v1".into(),
            validator_version: "structural-validator-v1".into(),
            validation_profile: "epf-011-knowledge-snapshot".into(),
            conforms: true,
            report_digest_hex: Some("b".repeat(64)),
        };
        p.record_snapshot_validation(validation.clone()).unwrap();

        assert_eq!(
            p.latest_snapshot_validation_receipts().unwrap(),
            vec![validation.clone()]
        );
        let audit_records = p.latest_snapshot_validation_receipt_records().unwrap();
        assert_eq!(audit_records.len(), 1);
        assert_eq!(audit_records[0].validation_sequence, 1);
        assert_eq!(audit_records[0].receipt, validation);
        assert_eq!(
            audit_records[0].stored_receipt_digest_hex,
            audit_records[0]
                .receipt
                .canonical_digest_hex_for_sequence(audit_records[0].validation_sequence)
        );
        assert_eq!(
            audit_records[0].stored_receipt_digest_hex,
            audit_records[0].recomputed_receipt_digest_hex()
        );
        assert!(audit_records[0].verify_self_digest());

        let second = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:event-2".into(),
            conforms: false,
            ..validation
        };
        p.record_snapshot_validation(second.clone()).unwrap();

        let receipts = p.latest_snapshot_validation_receipts().unwrap();
        assert_eq!(receipts.len(), 2);
        assert_eq!(receipts[0].validation_event, "validation:event-1");
        assert_eq!(receipts[1].validation_event, "validation:event-2");
        assert!(receipts[0].conforms);
        assert!(!receipts[1].conforms);

        let audit_records = p.latest_snapshot_validation_receipt_records().unwrap();
        assert_eq!(
            audit_records.iter().map(|r| r.validation_sequence).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(audit_records[0].receipt.validation_event, "validation:event-1");
        assert_eq!(audit_records[1].receipt.validation_event, "validation:event-2");
        for record in &audit_records {
            assert_eq!(
                record.stored_receipt_digest_hex,
                record
                    .receipt
                    .canonical_digest_hex_for_sequence(record.validation_sequence)
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_validation_receipt_records_export_full_history() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_validation_history_export_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let first_fact = FactRecord {
            memory_id: "history-export-one".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x71; BinaryHV::BYTES],
            source_text: "history export one".into(),
            confidence: 0.6,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&first_fact), &[], &[], &[])
            .unwrap();

        let first_snapshot = p.latest_snapshot_receipt().unwrap().unwrap();
        p.record_snapshot_validation(KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:history-one".into(),
            generation: first_snapshot.generation,
            snapshot_digest_hex: first_snapshot.canonical_digest_hex.clone(),
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: None,
        })
        .unwrap();

        let second_fact = FactRecord {
            memory_id: "history-export-two".into(),
            source_text: "history export two".into(),
            cycle: 2,
            ..first_fact
        };
        p.save_snapshot(std::slice::from_ref(&second_fact), &[], &[], &[])
            .unwrap();

        let second_snapshot = p.latest_snapshot_receipt().unwrap().unwrap();
        p.record_snapshot_validation(KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:history-two".into(),
            generation: second_snapshot.generation,
            snapshot_digest_hex: second_snapshot.canonical_digest_hex.clone(),
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: false,
            report_digest_hex: Some("c".repeat(64)),
        })
        .unwrap();
        p.record_snapshot_validation(KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:history-three".into(),
            generation: second_snapshot.generation,
            snapshot_digest_hex: second_snapshot.canonical_digest_hex,
            validator_ref: "validator:test-2".into(),
            validator_version: "v2".into(),
            validation_profile: "profile:test-2".into(),
            conforms: true,
            report_digest_hex: None,
        })
        .unwrap();

        let all = p.snapshot_validation_receipt_records().unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(
            all.iter()
                .map(|record| record.validation_sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            all.iter()
                .map(|record| record.receipt.validation_event.as_str())
                .collect::<Vec<_>>(),
            vec![
                "validation:history-one",
                "validation:history-two",
                "validation:history-three"
            ]
        );
        assert_eq!(all[0].receipt.generation, 1);
        assert_eq!(all[1].receipt.generation, 2);
        assert_eq!(all[2].receipt.generation, 2);
        assert!(all.iter().all(KnowledgeSnapshotValidationReceiptRecord::verify_self_digest));
        let history_digest =
            KnowledgeSnapshotValidationReceiptRecord::canonical_history_digest_hex(&all);
        assert_eq!(
            p.snapshot_validation_receipt_history_digest_hex().unwrap(),
            history_digest
        );

        let checkpoint = p.snapshot_validation_receipt_history_checkpoint().unwrap();
        assert_eq!(checkpoint.receipt_count, 3);
        assert_eq!(checkpoint.latest_validation_sequence, 3);
        assert_eq!(checkpoint.latest_generation, 2);
        assert!(checkpoint.verify_against_history(&all));
        p.verify_snapshot_validation_receipt_history_checkpoint(&checkpoint)
            .unwrap();

        let prefix_checkpoint =
            KnowledgeSnapshotValidationReceiptHistoryCheckpoint::from_history(&all[..2]);
        assert!(prefix_checkpoint.verify_prefix_against_history(&all));
        p.verify_snapshot_validation_receipt_history_checkpoint_prefix(&prefix_checkpoint)
            .unwrap();

        let mut reordered = all.clone();
        reordered.swap(0, 1);
        assert_ne!(
            KnowledgeSnapshotValidationReceiptRecord::canonical_history_digest_hex(&reordered),
            history_digest
        );
        assert!(!checkpoint.verify_against_history(&reordered));

        let mut truncated = all.clone();
        truncated.pop();
        assert!(!checkpoint.verify_against_history(&truncated));

        let mut substituted = all.clone();
        substituted[2].receipt.validator_ref = "validator:substituted".into();
        let substituted_receipt_digest = substituted[2].recomputed_receipt_digest_hex();
        substituted[2].stored_receipt_digest_hex = substituted_receipt_digest;
        assert!(substituted[2].verify_self_digest());
        assert!(substituted[2].verify_integrity());
        // The record remains self-consistent, but the externally retained checkpoint
        // detects the substitution.
        assert!(!checkpoint.verify_against_history(&substituted));

        let latest = p.latest_snapshot_validation_receipt_records().unwrap();
        assert_eq!(latest.len(), 2);
        assert_eq!(
            latest
                .iter()
                .map(|record| record.validation_sequence)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert!(latest.iter().all(KnowledgeSnapshotValidationReceiptRecord::verify_self_digest));

        // Historical ledger export remains independently auditable after live projection drift.
        let drifted_fact = FactRecord {
            source_text: "current projection drift".into(),
            ..second_fact
        };
        p.save_facts(std::slice::from_ref(&drifted_fact)).unwrap();

        let historical = p.snapshot_validation_receipt_records().unwrap();
        assert_eq!(
            historical
                .iter()
                .map(|record| record.validation_sequence)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert!(historical
            .iter()
            .all(KnowledgeSnapshotValidationReceiptRecord::verify_self_digest));

        let current_err = p.latest_snapshot_validation_receipt_records().unwrap_err();
        assert!(current_err.starts_with("Snapshot receipt digest mismatch: generation 2"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_validation_receipt_verifier_rejects_generation_regression() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_validation_generation_regression_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let first_fact = FactRecord {
            memory_id: "generation-regression-one".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x81; BinaryHV::BYTES],
            source_text: "generation regression one".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&first_fact), &[], &[], &[])
            .unwrap();
        let first_snapshot = p.latest_snapshot_receipt().unwrap().unwrap();

        p.record_snapshot_validation(KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:generation-one".into(),
            generation: first_snapshot.generation,
            snapshot_digest_hex: first_snapshot.canonical_digest_hex.clone(),
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: None,
        })
        .unwrap();

        let second_fact = FactRecord {
            memory_id: "generation-regression-two".into(),
            source_text: "generation regression two".into(),
            cycle: 2,
            ..first_fact
        };
        p.save_snapshot(std::slice::from_ref(&second_fact), &[], &[], &[])
            .unwrap();
        let second_snapshot = p.latest_snapshot_receipt().unwrap().unwrap();

        p.record_snapshot_validation(KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:generation-two".into(),
            generation: second_snapshot.generation,
            snapshot_digest_hex: second_snapshot.canonical_digest_hex.clone(),
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: None,
        })
        .unwrap();

        // Bypass the application writer and append a sequence-3 row that is internally
        // self-consistent but points backwards to generation 1.
        let late_validation = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:generation-regression".into(),
            generation: first_snapshot.generation,
            snapshot_digest_hex: first_snapshot.canonical_digest_hex,
            validator_ref: "validator:tamper-sim".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: None,
        };
        let sequence = 3_u64;
        let receipt_digest = late_validation.canonical_digest_hex_for_sequence(sequence);
        let conn = p.open_connection().unwrap();
        conn.execute(
            "INSERT INTO knowledge_snapshot_validation_receipts
             (validation_event, validation_sequence, generation, snapshot_digest_hex,
              validator_ref, validator_version, validation_profile, conforms,
              report_digest_hex, receipt_digest_hex)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                late_validation.validation_event,
                i64::try_from(sequence).unwrap(),
                i64::try_from(late_validation.generation).unwrap(),
                late_validation.snapshot_digest_hex,
                late_validation.validator_ref,
                late_validation.validator_version,
                late_validation.validation_profile,
                late_validation.conforms,
                late_validation.report_digest_hex,
                receipt_digest
            ],
        )
        .unwrap();

        let err = p.verify_snapshot_validation_receipts().unwrap_err();
        assert_eq!(
            err,
            "Snapshot validation receipt generation regression at sequence 3: previous 2, observed 1"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_validation_receipt_rejects_invalid_snapshot_digest() {
        let mut validation = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:snapshot-digest".into(),
            generation: 0,
            snapshot_digest_hex: "a".repeat(64),
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: None,
        };

        let err = validation.validate_input().unwrap_err();
        assert_eq!(err, "Snapshot validation generation must be positive");

        validation.generation = 1;
        validation.snapshot_digest_hex = "g".repeat(64);
        let err = validation.validate_input().unwrap_err();
        assert_eq!(
            err,
            "Snapshot validation digest must be a 64-character hexadecimal digest"
        );

        validation.snapshot_digest_hex = "a".repeat(64);
        assert!(validation.validate_input().is_ok());

        let record = KnowledgeSnapshotValidationReceiptRecord {
            validation_sequence: 1,
            receipt: validation,
            stored_receipt_digest_hex: String::new(),
        };
        let record = KnowledgeSnapshotValidationReceiptRecord {
            stored_receipt_digest_hex: record.recomputed_receipt_digest_hex(),
            ..record
        };
        assert!(record.verify_self_digest());
        assert!(record.verify_integrity());

        let malformed_receipt = KnowledgeSnapshotValidationReceipt {
            validator_ref: String::new(),
            ..record.receipt.clone()
        };
        let malformed_record = KnowledgeSnapshotValidationReceiptRecord {
            receipt: malformed_receipt,
            stored_receipt_digest_hex: String::new(),
            ..record
        };
        let malformed_record = KnowledgeSnapshotValidationReceiptRecord {
            stored_receipt_digest_hex: malformed_record.recomputed_receipt_digest_hex(),
            ..malformed_record
        };
        assert!(malformed_record.verify_self_digest());
        assert!(!malformed_record.verify_integrity());
    }

    #[test]
    fn test_snapshot_validation_receipt_rejects_non_hex_report_digest() {
        let mut validation = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:report-digest".into(),
            generation: 1,
            snapshot_digest_hex: "a".repeat(64),
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: Some("g".repeat(64)),
        };

        let err = validation.validate_input().unwrap_err();
        assert_eq!(
            err,
            "Snapshot validation report digest must be a 64-character hexadecimal digest when present"
        );

        validation.report_digest_hex = Some("d".repeat(64));
        assert!(validation.validate_input().is_ok());

        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_validation_report_digest_boundary_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);
        let fact = FactRecord {
            memory_id: "report-digest-boundary".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x88; BinaryHV::BYTES],
            source_text: "report digest boundary".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();
        let snapshot = p.latest_snapshot_receipt().unwrap().unwrap();
        p.record_snapshot_validation(KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:report-digest-boundary".into(),
            generation: snapshot.generation,
            snapshot_digest_hex: snapshot.canonical_digest_hex,
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: Some("e".repeat(64)),
        })
        .unwrap();

        let conn = p.open_connection().unwrap();
        conn.execute_batch(
            "DROP TRIGGER trg_knowledge_snapshot_validation_receipts_no_update;
             UPDATE knowledge_snapshot_validation_receipts
             SET report_digest_hex = 'gggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggggg'
             WHERE validation_event = 'validation:report-digest-boundary';",
        )
        .unwrap();

        // Recompute the v2 receipt digest so the mutation is self-consistent; verification
        // must still reject the semantic digest shape rather than relying on hash mismatch.
        let mutated = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:report-digest-boundary".into(),
            generation: snapshot.generation,
            snapshot_digest_hex: p.latest_snapshot_receipt().unwrap().unwrap().canonical_digest_hex,
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: Some("g".repeat(64)),
        };
        conn.execute(
            "UPDATE knowledge_snapshot_validation_receipts
             SET receipt_digest_hex = ?1
             WHERE validation_event = 'validation:report-digest-boundary'",
            [mutated.canonical_digest_hex_for_sequence(1)],
        )
        .unwrap();

        let err = p.verify_snapshot_validation_receipts().unwrap_err();
        assert!(err.contains(
            "Snapshot validation report digest must be a 64-character hexadecimal digest when present"
        ));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_validation_receipt_self_digest_detects_corruption() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_validation_digest_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "validation-digest".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x77; BinaryHV::BYTES],
            source_text: "digest".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();
        let committed = p.latest_snapshot_receipt().unwrap().unwrap();

        p.record_snapshot_validation(KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:digest".into(),
            generation: committed.generation,
            snapshot_digest_hex: committed.canonical_digest_hex,
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: None,
        })
        .unwrap();

        assert!(p.verify_snapshot_validation_receipts().is_ok());

        let conn = p.open_connection().unwrap();
        let blocked = conn
            .execute(
                "UPDATE knowledge_snapshot_validation_receipts
                 SET validator_version = 'blocked'
                 WHERE validation_event = 'validation:digest'",
                [],
            )
            .unwrap_err();
        assert!(blocked.to_string().contains("UPDATE prohibited"));

        // Simulate a privileged schema-level actor removing the application guard.
        // The receipt self-digest remains the independent corruption detector.
        conn.execute_batch(
            "DROP TRIGGER trg_knowledge_snapshot_validation_receipts_no_update;
             UPDATE knowledge_snapshot_validation_receipts
             SET validator_version = 'tampered'
             WHERE validation_event = 'validation:digest';",
        )
        .unwrap();

        let err = p.verify_snapshot_validation_receipts().unwrap_err();
        assert_eq!(
            err,
            "Snapshot validation receipt self-digest mismatch: validation:digest"
        );

        // Re-digesting a tampered record must not make it acceptable if it now
        // points at a different persisted snapshot generation's digest.
        conn.execute_batch(
            "DROP TRIGGER IF EXISTS trg_knowledge_snapshot_validation_receipts_no_update;
             UPDATE knowledge_snapshot_validation_receipts
             SET validator_version = 'v1',
                 snapshot_digest_hex = 'foreign-snapshot-digest',
                 receipt_digest_hex = NULL
             WHERE validation_event = 'validation:digest';",
        )
        .unwrap();
        let mut tampered = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:digest".into(),
            generation: committed.generation,
            snapshot_digest_hex: "foreign-snapshot-digest".into(),
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: None,
        };
        tampered.validate_input().unwrap();
        conn.execute(
            "UPDATE knowledge_snapshot_validation_receipts
             SET receipt_digest_hex = ?1
             WHERE validation_event = 'validation:digest'",
            rusqlite::params![tampered.canonical_digest_hex_for_sequence(1)],
        )
        .unwrap();

        let err = p.verify_snapshot_validation_receipts().unwrap_err();
        assert_eq!(
            err,
            "Snapshot validation receipt snapshot digest mismatch: validation:digest"
        );

        let second_validation = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:blocked-by-corruption".into(),
            generation: committed.generation,
            snapshot_digest_hex: committed.canonical_digest_hex.clone(),
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: None,
        };
        let err = p
            .record_snapshot_validation(second_validation)
            .unwrap_err();
        assert_eq!(
            err,
            "Snapshot validation receipt snapshot digest mismatch: validation:digest"
        );

        let err = p.latest_snapshot_validation_receipts().unwrap_err();
        assert_eq!(
            err,
            "Snapshot validation receipt snapshot digest mismatch: validation:digest"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_validation_receipt_self_digest_binds_internal_sequence() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_validation_sequence_digest_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "validation-sequence-digest".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0xA4; BinaryHV::BYTES],
            source_text: "sequence digest".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();
        let committed = p.latest_snapshot_receipt().unwrap().unwrap();

        p.record_snapshot_validation(KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:sequence-bound".into(),
            generation: committed.generation,
            snapshot_digest_hex: committed.canonical_digest_hex,
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: None,
        })
        .unwrap();

        let conn = p.open_connection().unwrap();
        conn.execute_batch(
            "DROP TRIGGER trg_knowledge_snapshot_validation_receipts_no_update;
             UPDATE knowledge_snapshot_validation_receipts
             SET validation_sequence = 2
             WHERE validation_event = 'validation:sequence-bound';",
        )
        .unwrap();

        let err = p.verify_snapshot_validation_receipts().unwrap_err();
        assert_eq!(
            err,
            "Snapshot validation receipt self-digest mismatch: validation:sequence-bound"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_validation_verification_uses_sequence_not_rowid() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_validation_sequence_order_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "validation-sequence-order".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x92; BinaryHV::BYTES],
            source_text: "sequence order".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();
        let committed = p.latest_snapshot_receipt().unwrap().unwrap();

        for event in ["validation:seq-order-1", "validation:seq-order-2"] {
            p.record_snapshot_validation(KnowledgeSnapshotValidationReceipt {
                validation_event: event.into(),
                generation: committed.generation,
                snapshot_digest_hex: committed.canonical_digest_hex.clone(),
                validator_ref: "validator:test".into(),
                validator_version: "v1".into(),
                validation_profile: "profile:test".into(),
                conforms: true,
                report_digest_hex: None,
            })
            .unwrap();
        }

        let conn = p.open_connection().unwrap();
        conn.execute_batch(
            "DROP TRIGGER trg_knowledge_snapshot_validation_receipts_no_update;
             UPDATE knowledge_snapshot_validation_receipts
             SET rowid = 200
             WHERE validation_event = 'validation:seq-order-1';
             UPDATE knowledge_snapshot_validation_receipts
             SET rowid = 100
             WHERE validation_event = 'validation:seq-order-2';",
        )
        .unwrap();

        p.verify_snapshot_validation_receipts().unwrap();

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_validation_sequence_rejects_history_gaps() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_validation_sequence_gap_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "validation-sequence-gap".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x91; BinaryHV::BYTES],
            source_text: "sequence gap".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();
        let committed = p.latest_snapshot_receipt().unwrap().unwrap();

        for event in ["validation:seq-1", "validation:seq-2"] {
            p.record_snapshot_validation(KnowledgeSnapshotValidationReceipt {
                validation_event: event.into(),
                generation: committed.generation,
                snapshot_digest_hex: committed.canonical_digest_hex.clone(),
                validator_ref: "validator:test".into(),
                validator_version: "v1".into(),
                validation_profile: "profile:test".into(),
                conforms: true,
                report_digest_hex: None,
            })
            .unwrap();
        }

        let conn = p.open_connection().unwrap();
        conn.execute_batch(
            "DROP TRIGGER trg_knowledge_snapshot_validation_receipts_no_delete;
             DELETE FROM knowledge_snapshot_validation_receipts
             WHERE validation_event = 'validation:seq-1';",
        )
        .unwrap();

        let err = p.verify_snapshot_validation_receipts().unwrap_err();
        assert_eq!(
            err,
            "Snapshot validation receipt sequence discontinuity: expected 1, observed 2"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_record_snapshot_validation_rejects_stale_generation_and_digest() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_stale_snapshot_validation_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "stale-validation".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x55; BinaryHV::BYTES],
            source_text: "stale".into(),
            confidence: 0.6,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();
        let first = p.latest_snapshot_receipt().unwrap().unwrap();

        let new_fact = FactRecord {
            source_text: "new generation".into(),
            cycle: 2,
            ..fact
        };
        p.save_snapshot(std::slice::from_ref(&new_fact), &[], &[], &[])
            .unwrap();
        let second = p.latest_snapshot_receipt().unwrap().unwrap();

        let stale = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:stale".into(),
            generation: first.generation,
            snapshot_digest_hex: first.canonical_digest_hex,
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "test".into(),
            conforms: true,
            report_digest_hex: None,
        };
        let err = p.record_snapshot_validation(stale).unwrap_err();
        assert_eq!(
            err,
            format!(
                "Snapshot validation generation is not current: requested {}, current {}",
                first.generation, second.generation
            )
        );

        let bad_digest = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:bad-digest".into(),
            generation: second.generation,
            snapshot_digest_hex: first.canonical_digest_hex,
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "test".into(),
            conforms: true,
            report_digest_hex: None,
        };
        let err = p.record_snapshot_validation(bad_digest).unwrap_err();
        assert!(err.starts_with(
            "Snapshot validation digest does not match committed receipt"
        ));
        assert!(p.latest_snapshot_validation_receipts().unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_validation_receipt_insert_requires_event_and_positive_sequence() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_validation_receipt_insert_guard_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        let conn = p.open_connection().unwrap();
        p.ensure_schema(&conn).unwrap();

        let err = conn
            .execute(
                "INSERT INTO knowledge_snapshot_validation_receipts
                 (validation_event, validation_sequence, generation, snapshot_digest_hex,
                  validator_ref, validator_version, validation_profile, conforms)
                 VALUES ('', 1, 999, ?1, 'validator', 'v1', 'profile', 1)",
                ["digest"],
            )
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("requires valid identity, digest, validator, outcome, and positive sequence"));

        let err = conn
            .execute(
                "INSERT INTO knowledge_snapshot_validation_receipts
                 (validation_event, validation_sequence, generation, snapshot_digest_hex,
                  validator_ref, validator_version, validation_profile, conforms)
                 VALUES ('validation:bad-seq', 0, 999, ?1, 'validator', 'v1', 'profile', 1)",
                ["digest"],
            )
            .unwrap_err();
        assert!(err
            .to_string()
            .contains("requires valid identity, digest, validator, outcome, and positive sequence"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_validation_receipt_rejects_duplicate_event_and_invalid_input() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_validation_input_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "validation-input".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x66; BinaryHV::BYTES],
            source_text: "input".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();
        let committed = p.latest_snapshot_receipt().unwrap().unwrap();

        let base = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:duplicate-event".into(),
            generation: committed.generation,
            snapshot_digest_hex: committed.canonical_digest_hex.clone(),
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: None,
        };
        p.record_snapshot_validation(base.clone()).unwrap();
        let err = p.record_snapshot_validation(base).unwrap_err();
        assert!(err.contains("UNIQUE") || err.contains("unique"));

        let empty_event = KnowledgeSnapshotValidationReceipt {
            validation_event: " ".into(),
            generation: committed.generation,
            snapshot_digest_hex: committed.canonical_digest_hex.clone(),
            validator_ref: "validator:test".into(),
            validator_version: "v1".into(),
            validation_profile: "profile:test".into(),
            conforms: true,
            report_digest_hex: None,
        };
        assert_eq!(
            p.record_snapshot_validation(empty_event).unwrap_err(),
            "Snapshot validation event must be non-empty"
        );

        let empty_report_digest = KnowledgeSnapshotValidationReceipt {
            validation_event: "validation:empty-report-digest".into(),
            report_digest_hex: Some(" ".into()),
            ..base
        };
        assert_eq!(
            p.record_snapshot_validation(empty_report_digest).unwrap_err(),
            "Snapshot validation report digest must be non-empty when present"
        );

        assert_eq!(p.latest_snapshot_validation_receipts().unwrap().len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_receipt_generation_tracks_only_committed_complete_snapshots() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_receipt_generation_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        assert_eq!(p.latest_snapshot_receipt().unwrap(), None);

        let fact = FactRecord {
            memory_id: "receipt-fact".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x11; BinaryHV::BYTES],
            source_text: "receipt".into(),
            confidence: 0.5,
            domain: None,
            cycle: 1,
            is_causal: false,
        };

        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();
        let first = p.latest_snapshot_receipt().unwrap().unwrap();
        assert_eq!(first.generation, 1);
        let expected = KnowledgePersistenceSnapshot {
            facts: vec![fact.clone()],
            provenance_relations: vec![],
            causal_edges: vec![],
            ontology: vec![],
        }
        .canonical_digest_hex();
        assert_eq!(first.canonical_digest_hex, expected);

        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();
        let second = p.latest_snapshot_receipt().unwrap().unwrap();
        assert_eq!(second.generation, 2);
        assert_eq!(second.canonical_digest_hex, first.canonical_digest_hex);

        let invalid = FactRecord {
            memory_id: "invalid".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x22; BinaryHV::BYTES - 1],
            source_text: "invalid".into(),
            confidence: 0.5,
            domain: None,
            cycle: 2,
            is_causal: false,
        };
        assert!(p.save_snapshot(&[invalid], &[], &[], &[]).is_err());

        let after_failure = p.latest_snapshot_receipt().unwrap().unwrap();
        assert_eq!(after_failure, second);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_digest_is_order_invariant_and_sensitive() {
        let fact_a = FactRecord {
            memory_id: "a".into(),
            canonical_identity: Some("canon-a".into()),
            provenance_family: Some("family".into()),
            vector_bytes: vec![0xAA; BinaryHV::BYTES],
            source_text: "A".into(),
            confidence: 0.25,
            domain: Some("test".into()),
            cycle: 4,
            is_causal: false,
        };
        let fact_b = FactRecord {
            memory_id: "b".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0xBB; BinaryHV::BYTES],
            source_text: "B".into(),
            confidence: 0.75,
            domain: None,
            cycle: 8,
            is_causal: true,
        };
        let relation = ProvenanceRelationRecord {
            source_memory_id: "b".into(),
            target_memory_id: "a".into(),
            kind: ProvenanceRelationKind::Corroborates,
            created_at: "cycle:8".into(),
        };
        let edge = CausalEdgeRecord {
            cause: "b".into(),
            effect: "c".into(),
            strength: -0.5,
            is_inhibitory: true,
            cycle: 8,
        };
        let ontology = OntologyRecord {
            name: "primitive".into(),
            vector_bytes: vec![0xCC; BinaryHV::BYTES],
            usage_count: 2,
            utility: 0.5,
            created_at_cycle: 7,
            last_used_cycle: 8,
            is_a_parent: None,
        };

        let snapshot_a = KnowledgePersistenceSnapshot {
            facts: vec![fact_a.clone(), fact_b.clone()],
            provenance_relations: vec![relation.clone()],
            causal_edges: vec![edge.clone()],
            ontology: vec![ontology.clone()],
        };
        let snapshot_b = KnowledgePersistenceSnapshot {
            facts: vec![fact_b, fact_a],
            provenance_relations: vec![relation],
            causal_edges: vec![edge],
            ontology: vec![ontology],
        };

        assert_eq!(
            snapshot_a.canonical_digest(),
            snapshot_b.canonical_digest()
        );

        let mut changed = snapshot_b.clone();
        changed.facts[0].confidence = 0.5;
        assert_ne!(snapshot_a.canonical_digest(), changed.canonical_digest());
        assert_eq!(snapshot_a.canonical_digest_hex().len(), 64);
    }

    #[test]
    fn test_saved_and_loaded_snapshot_share_canonical_digest() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_digest_round_trip_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "digest-fact".into(),
            canonical_identity: Some("digest-canonical".into()),
            provenance_family: Some("digest-family".into()),
            vector_bytes: vec![0x11; BinaryHV::BYTES],
            source_text: "digest fact".into(),
            confidence: 0.75,
            domain: Some("test".into()),
            cycle: 4,
            is_causal: true,
        };
        let relation = ProvenanceRelationRecord {
            source_memory_id: "digest-fact".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:4".into(),
        };
        let edge = CausalEdgeRecord {
            cause: "digest-fact".into(),
            effect: "digest-effect".into(),
            strength: 0.5,
            is_inhibitory: false,
            cycle: 4,
        };
        let ontology = OntologyRecord {
            name: "digest-primitive".into(),
            vector_bytes: vec![0x22; BinaryHV::BYTES],
            usage_count: 2,
            utility: 0.4,
            created_at_cycle: 4,
            last_used_cycle: 4,
            is_a_parent: None,
        };

        p.save_snapshot(&[fact], &[relation], &[edge], &[ontology])
            .unwrap();
        let loaded = p.load_snapshot().unwrap();
        assert_eq!(loaded.canonical_digest_hex().len(), 64);

        let mut equivalent = loaded.clone();
        equivalent.facts.reverse();
        equivalent.provenance_relations.reverse();
        equivalent.causal_edges.reverse();
        equivalent.ontology.reverse();
        assert_eq!(
            loaded.canonical_digest(),
            equivalent.canonical_digest()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
    #[test]
    fn test_load_snapshot_fails_closed_when_receipted_projection_drifts() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_load_snapshot_receipt_drift_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "load-drift".into(),
            canonical_identity: None,
            provenance_family: None,
            vector_bytes: vec![0x7Eu8; BinaryHV::BYTES],
            source_text: "original".into(),
            confidence: 0.7,
            domain: Some("test".into()),
            cycle: 1,
            is_causal: false,
        };
        p.save_snapshot(std::slice::from_ref(&fact), &[], &[], &[])
            .unwrap();

        {
            let conn = p.open_connection().unwrap();
            conn.execute(
                "UPDATE knowledge_facts
                 SET source_text = 'drifted'
                 WHERE memory_id = 'load-drift'",
                [],
            )
            .unwrap();
        }

        let err = p.load_snapshot().unwrap_err();
        assert!(err.starts_with("Snapshot receipt digest mismatch: generation 1"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_snapshot_returns_all_domains_from_one_read() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_load_snapshot_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            memory_id: "snapshot-fact".into(),
            canonical_identity: Some("snapshot-canonical".into()),
            provenance_family: Some("snapshot-family".into()),
            vector_bytes: vec![0xAB; BinaryHV::BYTES],
            source_text: "snapshot fact".into(),
            confidence: 0.9,
            domain: Some("test".into()),
            cycle: 4,
            is_causal: true,
        };
        let relation = ProvenanceRelationRecord {
            source_memory_id: "snapshot-fact".into(),
            target_memory_id: "source".into(),
            kind: ProvenanceRelationKind::DerivedFrom,
            created_at: "cycle:4".into(),
        };
        let edge = CausalEdgeRecord {
            cause: "cause".into(),
            effect: "effect".into(),
            strength: 0.8,
            is_inhibitory: false,
            cycle: 4,
        };
        let ontology = OntologyRecord {
            name: "snapshot-primitive".into(),
            vector_bytes: vec![0xCD; BinaryHV::BYTES],
            usage_count: 3,
            utility: 0.7,
            created_at_cycle: 4,
            last_used_cycle: 4,
            is_a_parent: None,
        };

        p.save_snapshot(
            std::slice::from_ref(&fact),
            std::slice::from_ref(&relation),
            std::slice::from_ref(&edge),
            std::slice::from_ref(&ontology),
        )
        .unwrap();

        let snapshot = p.load_snapshot().unwrap();
        assert_eq!(snapshot.facts.len(), 1);
        assert_eq!(snapshot.facts[0].memory_id, "snapshot-fact");
        assert_eq!(snapshot.provenance_relations.len(), 1);
        assert_eq!(snapshot.causal_edges.len(), 1);
        assert_eq!(snapshot.ontology.len(), 1);
        assert_eq!(p.total_loaded(), 4);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_persistence_round_trip_is_canonical_across_domains() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_persistence_round_trip_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let mut p = KnowledgePersistence::new(&db_path);

        let facts = [
            FactRecord {
                memory_id: "memory-z".into(),
                canonical_identity: Some("canon-z".into()),
                provenance_family: Some("family-z".into()),
                vector_bytes: vec![0xAA; 2048],
                source_text: "Zeta fact".into(),
                confidence: 0.7,
                domain: Some("test".into()),
                cycle: 9,
                is_causal: true,
            },
            FactRecord {
                memory_id: "memory-a".into(),
                canonical_identity: None,
                provenance_family: Some("family-a".into()),
                vector_bytes: vec![0x11; 2048],
                source_text: "Alpha fact".into(),
                confidence: 0.9,
                domain: None,
                cycle: 3,
                is_causal: false,
            },
        ];
        let relations = [
            ProvenanceRelationRecord {
                source_memory_id: "memory-a".into(),
                target_memory_id: "memory-z".into(),
                kind: ProvenanceRelationKind::DerivedFrom,
                created_at: "cycle:3".into(),
            },
            ProvenanceRelationRecord {
                source_memory_id: "memory-z".into(),
                target_memory_id: "memory-a".into(),
                kind: ProvenanceRelationKind::Corroborates,
                created_at: "cycle:9".into(),
            },
        ];
        let edges = [
            CausalEdgeRecord {
                cause: "z".into(),
                effect: "effect-z".into(),
                strength: 0.8,
                is_inhibitory: false,
                cycle: 9,
            },
            CausalEdgeRecord {
                cause: "a".into(),
                effect: "effect-a".into(),
                strength: 0.6,
                is_inhibitory: true,
                cycle: 3,
            },
        ];
        let ontology = [
            OntologyRecord {
                name: "zeta".into(),
                vector_bytes: vec![0x22; 2048],
                usage_count: 4,
                utility: 0.8,
                created_at_cycle: 9,
                last_used_cycle: 10,
                is_a_parent: Some("animal".into()),
            },
            OntologyRecord {
                name: "alpha".into(),
                vector_bytes: vec![0x33; 2048],
                usage_count: 2,
                utility: 0.2,
                created_at_cycle: 3,
                last_used_cycle: 4,
                is_a_parent: None,
            },
        ];

        assert_eq!(p.save_facts(&facts).unwrap(), 2);
        assert_eq!(p.save_provenance_relations(&relations).unwrap(), 2);
        assert_eq!(p.save_causal_edges(&edges).unwrap(), 2);
        assert_eq!(p.save_ontology(&ontology).unwrap(), 2);

        let first_facts = p.load_facts().unwrap();
        let first_relations = p.load_provenance_relations().unwrap();
        let first_edges = p.load_causal_edges().unwrap();
        let first_ontology = p.load_ontology().unwrap();

        assert_eq!(p.save_facts(&first_facts).unwrap(), first_facts.len());
        assert_eq!(p.save_provenance_relations(&first_relations).unwrap(), 0);
        assert_eq!(p.save_causal_edges(&first_edges).unwrap(), first_edges.len());
        assert_eq!(p.save_ontology(&first_ontology).unwrap(), first_ontology.len());

        let second_facts = p.load_facts().unwrap();
        let second_relations = p.load_provenance_relations().unwrap();
        let second_edges = p.load_causal_edges().unwrap();
        let second_ontology = p.load_ontology().unwrap();

        assert_eq!(
            second_facts
                .iter()
                .map(|f| (
                    f.memory_id.as_str(),
                    f.canonical_identity.as_deref(),
                    f.provenance_family.as_deref(),
                    f.vector_bytes.as_slice(),
                    f.source_text.as_str(),
                    f.confidence.to_bits(),
                    f.domain.as_deref(),
                    f.cycle,
                    f.is_causal
                ))
                .collect::<Vec<_>>(),
            first_facts
                .iter()
                .map(|f| (
                    f.memory_id.as_str(),
                    f.canonical_identity.as_deref(),
                    f.provenance_family.as_deref(),
                    f.vector_bytes.as_slice(),
                    f.source_text.as_str(),
                    f.confidence.to_bits(),
                    f.domain.as_deref(),
                    f.cycle,
                    f.is_causal
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            second_relations
                .iter()
                .map(|r| (
                    r.source_memory_id.as_str(),
                    r.target_memory_id.as_str(),
                    r.kind,
                    r.created_at.as_str()
                ))
                .collect::<Vec<_>>(),
            first_relations
                .iter()
                .map(|r| (
                    r.source_memory_id.as_str(),
                    r.target_memory_id.as_str(),
                    r.kind,
                    r.created_at.as_str()
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            second_edges
                .iter()
                .map(|e| (
                    e.cause.as_str(),
                    e.effect.as_str(),
                    e.strength.to_bits(),
                    e.is_inhibitory,
                    e.cycle
                ))
                .collect::<Vec<_>>(),
            first_edges
                .iter()
                .map(|e| (
                    e.cause.as_str(),
                    e.effect.as_str(),
                    e.strength.to_bits(),
                    e.is_inhibitory,
                    e.cycle
                ))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            second_ontology
                .iter()
                .map(|o| (
                    o.name.as_str(),
                    o.vector_bytes.as_slice(),
                    o.usage_count,
                    o.utility.to_bits(),
                    o.created_at_cycle,
                    o.last_used_cycle,
                    o.is_a_parent.as_deref()
                ))
                .collect::<Vec<_>>(),
            first_ontology
                .iter()
                .map(|o| (
                    o.name.as_str(),
                    o.vector_bytes.as_slice(),
                    o.usage_count,
                    o.utility.to_bits(),
                    o.created_at_cycle,
                    o.last_used_cycle,
                    o.is_a_parent.as_deref()
                ))
                .collect::<Vec<_>>()
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_and_load_causal_edges() {
        let dir = std::env::temp_dir().join(format!("symthaea_causal_test_{}", std::process::id()));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);

        let edges = vec![
            CausalEdgeRecord {
                cause: "sanctions".into(),
                effect: "oil shortage".into(),
                strength: 0.8,
                is_inhibitory: false,
                cycle: 1,
            },
            CausalEdgeRecord {
                cause: "diplomacy".into(),
                effect: "war".into(),
                strength: -0.6,
                is_inhibitory: true,
                cycle: 2,
            },
        ];

        let saved = p.save_causal_edges(&edges).unwrap();
        assert_eq!(saved, 2);
