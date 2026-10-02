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
//! - knowledge_snapshot_receipts: generation, canonical_digest_hex
//! - knowledge_snapshot_validation_receipts: validation_event, generation, validator/profile metadata, outcome
//!
//! Science: Ebbinghaus (1885) memory consolidation across sessions

use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;
use rusqlite::OptionalExtension;
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
    fn canonical_digest_hex(&self) -> String {
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
        if self.snapshot_digest_hex.trim().is_empty() {
            return Err("Snapshot validation digest must be non-empty".into());
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
        if self.report_digest_hex.as_deref().is_some_and(|d| d.trim().is_empty()) {
            return Err("Snapshot validation report digest must be non-empty when present".into());
        }
        Ok(())
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

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin snapshot transaction: {e}"))?;

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
                        format!("{:?}", relation.kind),
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

        tx.execute(
            "INSERT INTO knowledge_snapshot_receipts (canonical_digest_hex) VALUES (?1)",
            rusqlite::params![actual_digest],
        )
        .map_err(|e| format!("Snapshot receipt: {e}"))?;

        tx.commit()
            .map_err(|e| format!("Commit snapshot transaction: {e}"))?;
        self.total_saved += saved_count as u64;
        Ok(())
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
        conn.query_row(
            "SELECT generation, canonical_digest_hex
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
                })
            },
        )
        .optional()
        .map_err(|e| format!("Load latest snapshot receipt: {e}"))
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

        // Materialize deterministic identities for pre-EPF-011 rows before
        // beginning the read transaction.
        conn.execute(
            "UPDATE knowledge_facts
             SET memory_id = 'legacy-fact:' || id
             WHERE memory_id IS NULL",
            [],
        )
        .map_err(|e| e.to_string())?;

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin verified persistence snapshot read: {e}"))?;

        let receipt = tx
            .query_row(
                "SELECT generation, canonical_digest_hex
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

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin snapshot validation transaction: {e}"))?;

        verify_snapshot_validation_receipts_in_tx(&tx)?;

        let latest = tx
            .query_row(
                "SELECT generation, canonical_digest_hex
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

        tx.execute(
            "INSERT INTO knowledge_snapshot_validation_receipts
             (validation_event, generation, snapshot_digest_hex, validator_ref,
              validator_version, validation_profile, conforms, report_digest_hex, receipt_digest_hex)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            rusqlite::params![
                validation.validation_event,
                i64::try_from(validation.generation)
                    .map_err(|_| "Snapshot validation generation exceeds SQLite INTEGER range")?,
                validation.snapshot_digest_hex,
                validation.validator_ref,
                validation.validator_version,
                validation.validation_profile,
                validation.conforms,
                validation.report_digest_hex,
                validation.canonical_digest_hex(),
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
        verify_snapshot_validation_receipts_in_tx(&tx)?;
        tx.commit()
            .map_err(|e| format!("Commit validation receipt verification: {e}"))?;
        Ok(())
    }

    /// Load validation receipts for the latest committed snapshot generation.
    pub fn latest_snapshot_validation_receipts(
        &mut self,
    ) -> Result<Vec<KnowledgeSnapshotValidationReceipt>, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let verify_tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin latest validation verification: {e}"))?;
        verify_snapshot_validation_receipts_in_tx(&verify_tx)?;
        verify_tx
            .commit()
            .map_err(|e| format!("Commit latest validation verification: {e}"))?;

        let mut stmt = conn
            .prepare(
                "SELECT validation_event, generation, snapshot_digest_hex, validator_ref,
                        validator_version, validation_profile, conforms, report_digest_hex
                 FROM knowledge_snapshot_validation_receipts
                 WHERE generation = (
                     SELECT generation
                     FROM knowledge_snapshot_receipts
                     ORDER BY generation DESC
                     LIMIT 1
                 )
                 ORDER BY validation_event ASC",
            )
            .map_err(|e| format!("Prepare latest snapshot validations: {e}"))?;

        stmt.query_map([], |row| {
            let generation = row.get::<_, i64>(1)?;
            Ok(KnowledgeSnapshotValidationReceipt {
                validation_event: row.get(0)?,
                generation: u64::try_from(generation).map_err(|_| {
                    rusqlite::Error::IntegralValueOutOfRange(1, generation)
                })?,
                snapshot_digest_hex: row.get(2)?,
                validator_ref: row.get(3)?,
                validator_version: row.get(4)?,
                validation_profile: row.get(5)?,
                conforms: row.get(6)?,
                report_digest_hex: row.get(7)?,
            })
        })
        .map_err(|e| format!("Query latest snapshot validations: {e}"))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Load latest snapshot validation row: {e}"))
    }

    /// Load all persistence domains from one SQLite read transaction.
    ///
    /// The returned records are all observed from a single database snapshot.
    /// This prevents startup restore from combining facts/provenance/causal/ontology
    /// rows committed by different snapshot generations.
    pub fn load_snapshot(&mut self) -> Result<KnowledgePersistenceSnapshot, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        // Materialize deterministic identities for pre-EPF-011 rows before
        // beginning the read transaction.
        conn.execute(
            "UPDATE knowledge_facts
             SET memory_id = 'legacy-fact:' || id
             WHERE memory_id IS NULL",
            [],
        )
        .map_err(|e| e.to_string())?;

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin persistence snapshot read: {e}"))?;

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

        // Materialize deterministic identities for pre-EPF-011 rows so a subsequent
        // save updates the same row rather than creating a second representation.
        conn.execute(
            "UPDATE knowledge_facts
             SET memory_id = 'legacy-fact:' || id
             WHERE memory_id IS NULL",
            [],
        )
        .map_err(|e| e.to_string())?;

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
                        format!("{:?}", relation.kind),
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
            let kind = match kind.as_str() {
                "DerivedFrom" => ProvenanceRelationKind::DerivedFrom,
                "RevisedFrom" => ProvenanceRelationKind::RevisedFrom,
                "Supersedes" => ProvenanceRelationKind::Supersedes,
                "Contradicts" => ProvenanceRelationKind::Contradicts,
                "Corroborates" => ProvenanceRelationKind::Corroborates,
                "RepresentationOf" => ProvenanceRelationKind::RepresentationOf,
                _ => return Err(rusqlite::Error::InvalidColumnType(2, "kind".into(), rusqlite::types::Type::Text)),
            };
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
            return Ok(());
        }

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS knowledge_facts (
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
                canonical_digest_hex TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS knowledge_snapshot_validation_receipts (
                validation_event TEXT PRIMARY KEY,
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
        let columns: Vec<String> = conn
            .prepare("PRAGMA table_info(knowledge_facts)")
            .map_err(|e| format!("Schema inspect: {e}"))?
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|e| format!("Schema inspect query: {e}"))?
            .filter_map(Result::ok)
            .collect();
        for (name, ty) in [("memory_id", "TEXT"), ("canonical_identity", "TEXT"), ("provenance_family", "TEXT")] {
            if !columns.iter().any(|c| c == name) {
                conn.execute(&format!("ALTER TABLE knowledge_facts ADD COLUMN {name} {ty}"), [])
                    .map_err(|e| format!("Schema migration {name}: {e}"))?;
            }
        }

        // Add validation-receipt self-digest support to databases created by the
        // earlier validation-ledger tranche, then deterministically backfill legacy rows.
        let validation_columns: Vec<String> = conn
            .prepare("PRAGMA table_info(knowledge_snapshot_validation_receipts)")
            .map_err(|e| format!("Validation schema inspect: {e}"))?
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|e| format!("Validation schema inspect query: {e}"))?
            .filter_map(Result::ok)
            .collect();
        if !validation_columns.iter().any(|c| c == "receipt_digest_hex") {
            conn.execute(
                "ALTER TABLE knowledge_snapshot_validation_receipts
                 ADD COLUMN receipt_digest_hex TEXT",
                [],
            )
            .map_err(|e| format!("Validation schema migration receipt_digest_hex: {e}"))?;
        }

        let legacy_validation_rows = {
            let mut stmt = conn
                .prepare(
                    "SELECT rowid, validation_event, generation, snapshot_digest_hex,
                            validator_ref, validator_version, validation_profile,
                            conforms, report_digest_hex
                     FROM knowledge_snapshot_validation_receipts
                     WHERE receipt_digest_hex IS NULL
                     ORDER BY rowid ASC",
                )
                .map_err(|e| format!("Validation receipt backfill prepare: {e}"))?;
            stmt.query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    KnowledgeSnapshotValidationReceipt {
                        validation_event: row.get(1)?,
                        generation: u64::try_from(row.get::<_, i64>(2)?).map_err(|_| {
                            rusqlite::Error::InvalidColumnType(
                                2,
                                "generation".into(),
                                rusqlite::types::Type::Integer,
                            )
                        })?,
                        snapshot_digest_hex: row.get(3)?,
                        validator_ref: row.get(4)?,
                        validator_version: row.get(5)?,
                        validation_profile: row.get(6)?,
                        conforms: row.get(7)?,
                        report_digest_hex: row.get(8)?,
                    },
                ))
            })
            .map_err(|e| format!("Validation receipt backfill query: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Validation receipt backfill row: {e}"))?
        };

        for (rowid, receipt) in legacy_validation_rows {
            conn.execute(
                "UPDATE knowledge_snapshot_validation_receipts
                 SET receipt_digest_hex = ?1
                 WHERE rowid = ?2",
                rusqlite::params![receipt.canonical_digest_hex(), rowid],
            )
            .map_err(|e| format!("Validation receipt backfill update: {e}"))?;
        }

        // Create the identity index only after the additive columns exist on legacy databases.
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_memory_id_unique ON knowledge_facts(memory_id)",
            [],
        )
        .map_err(|e| format!("Schema identity index: {e}"))?;

        self.initialized = true;
        Ok(())
    }
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
                    let kind = match kind.as_str() {
                        "DerivedFrom" => ProvenanceRelationKind::DerivedFrom,
                        "RevisedFrom" => ProvenanceRelationKind::RevisedFrom,
                        "Supersedes" => ProvenanceRelationKind::Supersedes,
                        "Contradicts" => ProvenanceRelationKind::Contradicts,
                        "Corroborates" => ProvenanceRelationKind::Corroborates,
                        "RepresentationOf" => ProvenanceRelationKind::RepresentationOf,
                        _ => return Err(rusqlite::Error::InvalidColumnType(
                            2,
                            "kind".into(),
                            rusqlite::types::Type::Text,
                        )),
                    };
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

fn verify_snapshot_validation_receipts_in_tx(
    tx: &rusqlite::Transaction<'_>,
) -> Result<(), String> {
    let mut stmt = tx
        .prepare(
            "SELECT validation_event, generation, snapshot_digest_hex, validator_ref,
                    validator_version, validation_profile, conforms, report_digest_hex,
                    receipt_digest_hex
             FROM knowledge_snapshot_validation_receipts
             ORDER BY rowid ASC",
        )
        .map_err(|e| format!("Prepare validation receipt verification: {e}"))?;

    let rows = stmt
        .query_map([], |row| {
            let generation = u64::try_from(row.get::<_, i64>(1)?).map_err(|_| {
                rusqlite::Error::InvalidColumnType(
                    1,
                    "generation".into(),
                    rusqlite::types::Type::Integer,
                )
            })?;
            Ok((
                KnowledgeSnapshotValidationReceipt {
                    validation_event: row.get(0)?,
                    generation,
                    snapshot_digest_hex: row.get(2)?,
                    validator_ref: row.get(3)?,
                    validator_version: row.get(4)?,
                    validation_profile: row.get(5)?,
                    conforms: row.get(6)?,
                    report_digest_hex: row.get(7)?,
                },
                row.get::<_, Option<String>>(8)?,
            ))
        })
        .map_err(|e| format!("Query validation receipts for verification: {e}"))?;

    for row in rows {
        let (receipt, stored_digest) =
            row.map_err(|e| format!("Load validation receipt for verification: {e}"))?;
        let Some(stored_digest) = stored_digest else {
            return Err(format!(
                "Snapshot validation receipt missing self-digest: {}",
                receipt.validation_event
            ));
        };
        let expected = receipt.canonical_digest_hex();
        if stored_digest != expected {
            return Err(format!(
                "Snapshot validation receipt self-digest mismatch: {}",
                receipt.validation_event
            ));
        }
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
                 (validation_event, generation, snapshot_digest_hex, validator_ref,
                  validator_version, validation_profile, conforms)
                 VALUES ('orphan', 999999, 'digest', 'validator', 'v1', 'profile', 1)",
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
            report_digest_hex: Some("report-digest-1".into()),
        };
        p.record_snapshot_validation(validation.clone()).unwrap();

        assert_eq!(
            p.latest_snapshot_validation_receipts().unwrap(),
            vec![validation]
        );

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
        conn.execute(
            "UPDATE knowledge_snapshot_validation_receipts
             SET validator_version = 'tampered'
             WHERE validation_event = 'validation:digest'",
            [],
        )
        .unwrap();

        let err = p.verify_snapshot_validation_receipts().unwrap_err();
        assert_eq!(
            err,
            "Snapshot validation receipt self-digest mismatch: validation:digest"
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
            "Snapshot validation receipt self-digest mismatch: validation:digest"
        );

        let err = p.latest_snapshot_validation_receipts().unwrap_err();
        assert_eq!(
            err,
            "Snapshot validation receipt self-digest mismatch: validation:digest"
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
