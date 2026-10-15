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
//! - knowledge_facts: row id, graph_fact_id, vector_blob, source_text, confidence, domain, cycle, is_causal
//! - knowledge_fact_relations: per-fact ordered subject/predicate/object records, with semantic roles,
//!   causal/negation flags, and relation-level confidence
//! - knowledge_causal_edges: edge_id, cause, effect, strength, inhibition/negation flags,
//!   source text, and discovery cycle; each row is one edge observation
//! - knowledge_ontology: name, vector_blob, usage_count, utility, cycle
//!
//! Science: Ebbinghaus (1885) memory consolidation across sessions

use std::collections::HashMap;
use std::path::Path;
use rusqlite::OptionalExtension;
use super::extraction::{ExtractedRelation, SemanticRole};
use symthaea_core::hdc::unified_hv::BinaryHV;

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

/// A fact record for persistence, including the source relations that were
/// otherwise lost when the structured extraction was bundled into an HDC vector.
#[derive(Debug, Clone)]
pub struct FactRecord {
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
    /// Structured extracted relations, retained separately from the lossy HDC bundle.
    pub relations: Vec<ExtractedRelation>,
    /// Stable graph-level identity; absent for rows written before this field existed.
    pub graph_fact_id: Option<u64>,
    /// Confidence assigned at insertion; absent for rows written by older schemas.
    pub initial_confidence: Option<f32>,
    /// Last access cycle; absent for rows written by older schemas.
    pub last_accessed_cycle: Option<u64>,
    /// Number of corroborations tracked by the graph.
    pub corroboration_count: u32,
    /// Number of contradictions tracked by the graph.
    pub contradiction_count: u32,
}

/// A serializable causal edge record
#[derive(Debug, Clone)]
pub struct CausalEdgeRecord {
    pub cause: String,
    pub effect: String,
    /// Non-negative causal strength; inhibition is represented separately.
    pub strength: f32,
    pub is_inhibitory: bool,
    /// Whether the source explicitly negated this relation.
    pub is_negated: bool,
    /// Text from the source that produced this edge observation.
    pub source_text: String,
    /// Cycle when this observation was discovered.
    pub cycle: u64,
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

/// Convert invalid but SQLite-decodable values into row conversion errors so loaders
/// fail closed instead of importing semantically invalid persisted state.
fn invalid_column_data(
    column_index: usize,
    column_type: rusqlite::types::Type,
    message: impl Into<String>,
) -> rusqlite::Error {
    rusqlite::Error::FromSqlConversionFailure(
        column_index,
        column_type,
        Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            message.into(),
        )),
    )
}

fn validate_hypervector_bytes(bytes: &[u8], column_index: usize) -> rusqlite::Result<()> {
    if bytes.len() == BinaryHV::BYTES {
        Ok(())
    } else {
        Err(invalid_column_data(
            column_index,
            rusqlite::types::Type::Blob,
            format!(
                "invalid hypervector length: expected {} bytes, got {}",
                BinaryHV::BYTES,
                bytes.len()
            ),
        ))
    }
}

/// Persist a monotonic allocator high-water mark without lowering any value
/// already committed by an earlier snapshot. The caller should use a transaction
/// when this value must commit atomically with fact rows.
fn persist_graph_fact_id_high_water(
    conn: &rusqlite::Connection,
    candidate_next_id: Option<u64>,
) -> Result<(), String> {
    let existing: Option<i64> = conn
        .query_row(
            "SELECT value FROM knowledge_metadata WHERE key = 'next_graph_fact_id'",
            [],
            |row| row.get(0),
        )
        .optional()
        .map_err(|e| format!("Read prior graph fact ID high-water mark: {e}"))?;
    let existing = existing
        .map(|value| {
            u64::try_from(value)
                .ok()
                .filter(|id| *id > 0)
                .ok_or_else(|| format!("Invalid persisted next graph fact ID: {value}"))
        })
        .transpose()?;
    let next_id = match (existing, candidate_next_id) {
        (Some(old), Some(candidate)) => Some(old.max(candidate)),
        (Some(old), None) => Some(old),
        (None, Some(candidate)) => Some(candidate),
        (None, None) => None,
    };
    if let Some(next_id) = next_id {
        if next_id == 0 {
            return Err("Next graph fact ID must be positive".into());
        }
        let stored_id = i64::try_from(next_id)
            .map_err(|e| format!("Next graph fact ID exceeds SQLite INTEGER range: {e}"))?;
        conn.execute(
            "INSERT OR REPLACE INTO knowledge_metadata (key, value)
             VALUES ('next_graph_fact_id', ?1)",
            rusqlite::params![stored_id],
        )
        .map_err(|e| format!("Write next graph fact ID: {e}"))?;
    }
    Ok(())
}

fn next_graph_fact_id_after_records(
    facts: &[FactRecord],
) -> Result<Option<u64>, String> {
    facts
        .iter()
        .filter_map(|fact| fact.graph_fact_id)
        .max()
        .map(|max_id| {
            max_id
                .checked_add(1)
                .ok_or_else(|| "Cannot advance allocator past maximum graph fact ID".to_string())
        })
        .transpose()
}

fn decode_semantic_role(value: String, column_index: usize) -> rusqlite::Result<SemanticRole> {
    SemanticRole::from_storage_key(&value).ok_or_else(|| {
        invalid_column_data(
            column_index,
            rusqlite::types::Type::Text,
            format!("unknown persisted semantic role key: {value}"),
        )
    })
}

fn validate_fact_relations(relations: &[ExtractedRelation]) -> Result<(), String> {
    for relation in relations {
        if !relation.confidence.is_finite() || !(0.0..=1.0).contains(&relation.confidence) {
            return Err(format!(
                "Invalid extracted-relation confidence outside [0, 1]: {}",
                relation.confidence
            ));
        }
    }
    Ok(())
}

fn insert_fact_relations(
    conn: &rusqlite::Connection,
    fact_row_id: i64,
    relations: &[ExtractedRelation],
) -> Result<(), String> {
    validate_fact_relations(relations)?;
    for (ordinal, relation) in relations.iter().enumerate() {
        let ordinal = i64::try_from(ordinal)
            .map_err(|e| format!("Relation ordinal exceeds SQLite INTEGER range: {e}"))?;
        conn.execute(
            "INSERT INTO knowledge_fact_relations
             (fact_row_id, ordinal, subject, predicate, object, subject_role, object_role,
              is_causal, is_negated, confidence)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                fact_row_id,
                ordinal,
                relation.subject,
                relation.predicate,
                relation.object,
                relation.subject_role.storage_key(),
                relation.object_role.storage_key(),
                relation.is_causal,
                relation.is_negated,
                relation.confidence,
            ],
        )
        .map_err(|e| format!("Insert structured fact relation: {e}"))?;
    }
    Ok(())
}

fn load_fact_relations(
    conn: &rusqlite::Connection,
) -> Result<HashMap<i64, Vec<ExtractedRelation>>, String> {
    let mut stmt = conn
        .prepare(
            "SELECT fact_row_id, ordinal, subject, predicate, object, subject_role, object_role,
                    is_causal, is_negated, confidence
             FROM knowledge_fact_relations
             ORDER BY fact_row_id, ordinal",
        )
        .map_err(|e| format!("Prepare structured fact relations: {e}"))?;
    let rows = stmt
        .query_map([], |row| {
            let confidence = decode_confidence(row.get(9)?, 9)?;
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                ExtractedRelation {
                    subject: row.get(2)?,
                    predicate: row.get(3)?,
                    object: row.get(4)?,
                    subject_role: decode_semantic_role(row.get(5)?, 5)?,
                    object_role: decode_semantic_role(row.get(6)?, 6)?,
                    is_causal: row.get(7)?,
                    is_negated: row.get(8)?,
                    confidence,
                },
            ))
        })
        .map_err(|e| format!("Query structured fact relations: {e}"))?;
    let decoded = rows
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("Decode structured fact relations: {e}"))?;
    let mut by_fact: HashMap<i64, Vec<ExtractedRelation>> = HashMap::new();
    let mut previous: HashMap<i64, i64> = HashMap::new();
    for (fact_row_id, ordinal, relation) in decoded {
        if fact_row_id <= 0 || ordinal < 0 {
            return Err(format!(
                "Invalid structured-relation key: fact row {fact_row_id}, ordinal {ordinal}"
            ));
        }
        if let Some(previous_ordinal) = previous.insert(fact_row_id, ordinal) {
            if previous_ordinal.checked_add(1) != Some(ordinal) {
                return Err(format!(
                    "Non-contiguous relation ordinal for fact row {fact_row_id}: expected {}, got {ordinal}",
                    previous_ordinal.saturating_add(1)
                ));
            }
        } else if ordinal != 0 {
            return Err(format!(
                "Structured relations for fact row {fact_row_id} begin at ordinal {ordinal}, expected 0"
            ));
        }
        by_fact.entry(fact_row_id).or_default().push(relation);
    }
    Ok(by_fact)
}

fn decode_optional_graph_fact_id(
    value: Option<i64>,
    column_index: usize,
) -> rusqlite::Result<Option<u64>> {
    value
        .map(|id| {
            if id > 0 {
                Ok(id as u64)
            } else {
                Err(invalid_column_data(
                    column_index,
                    rusqlite::types::Type::Integer,
                    format!("invalid persisted graph fact ID (must be positive): {id}"),
                ))
            }
        })
        .transpose()
}

fn encode_optional_graph_fact_id(value: Option<u64>) -> Result<Option<i64>, String> {
    value
        .map(|id| {
            if id == 0 {
                return Err("Graph fact ID must be positive".to_string());
            }
            i64::try_from(id)
                .map_err(|e| format!("Graph fact ID exceeds SQLite INTEGER range: {e}"))
        })
        .transpose()
}

fn decode_nonnegative_counter(
    value: i64,
    column_index: usize,
    field: &str,
) -> rusqlite::Result<u64> {
    u64::try_from(value).map_err(|_| {
        invalid_column_data(
            column_index,
            rusqlite::types::Type::Integer,
            format!("invalid negative persisted {field}: {value}"),
        )
    })
}

fn decode_finite_f32(value: f32, column_index: usize, field: &str) -> rusqlite::Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_column_data(
            column_index,
            rusqlite::types::Type::Real,
            format!("non-finite persisted {field}: {value}"),
        ))
    }
}

fn decode_nonnegative_u32_counter(
    value: i64,
    column_index: usize,
    field: &str,
) -> rusqlite::Result<u32> {
    u32::try_from(value).map_err(|_| {
        invalid_column_data(
            column_index,
            rusqlite::types::Type::Integer,
            format!("invalid persisted {field} outside u32 range: {value}"),
        )
    })
}

fn decode_confidence(value: f32, column_index: usize) -> rusqlite::Result<f32> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(invalid_column_data(
            column_index,
            rusqlite::types::Type::Real,
            format!("invalid persisted confidence outside [0, 1]: {value}"),
        ))
    }
}

fn decode_finite_f64(value: f64, column_index: usize, field: &str) -> rusqlite::Result<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_column_data(
            column_index,
            rusqlite::types::Type::Real,
            format!("non-finite persisted {field}: {value}"),
        ))
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
        for fact in facts {
            validate_fact_relations(&fact.relations)?;
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        conn.execute_batch("BEGIN TRANSACTION")
            .map_err(|e| e.to_string())?;

        let mut count = 0;
        for fact in facts {
            let cycle = i64::try_from(fact.cycle)
                .map_err(|e| format!("Fact insertion cycle exceeds SQLite INTEGER range: {e}"))?;
            let last_accessed_cycle = fact
                .last_accessed_cycle
                .map(i64::try_from)
                .transpose()
                .map_err(|e| format!("Fact last-access cycle exceeds SQLite INTEGER range: {e}"))?;
            let graph_fact_id = encode_optional_graph_fact_id(fact.graph_fact_id)?;
            conn.execute(
                "INSERT INTO knowledge_facts
                 (vector_blob, source_text, confidence, domain, cycle, is_causal,
                  initial_confidence, last_accessed_cycle, corroboration_count, contradiction_count, graph_fact_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                rusqlite::params![
                    fact.vector_bytes,
                    fact.source_text,
                    fact.confidence,
                    fact.domain,
                    cycle,
                    fact.is_causal,
                    fact.initial_confidence,
                    last_accessed_cycle,
                    fact.corroboration_count as i64,
                    fact.contradiction_count as i64,
                    graph_fact_id,
                ],
            )
            .map_err(|e| e.to_string())?;
            let fact_row_id = conn.last_insert_rowid();
            insert_fact_relations(&conn, fact_row_id, &fact.relations)?;
            count += 1;
        }

        let candidate_next_id = next_graph_fact_id_after_records(facts)?;
        persist_graph_fact_id_high_water(&conn, candidate_next_id)?;
        conn.execute_batch("COMMIT").map_err(|e| e.to_string())?;

        self.total_saved += count as u64;
        Ok(count)
    }

    /// Atomically replace the persisted fact set with a complete graph snapshot.
    ///
    /// Unlike `save_facts`, repeated snapshots do not append duplicate copies.
    /// The delete and inserts commit together, so any failed snapshot preserves
    /// the previous committed fact set.
    pub fn replace_facts_snapshot(&mut self, facts: &[FactRecord]) -> Result<usize, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }
        for fact in facts {
            validate_fact_relations(&fact.relations)?;
        }

        let mut conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|e| format!("Begin fact snapshot transaction: {e}"))?;

        tx.execute("DELETE FROM knowledge_fact_relations", [])
            .map_err(|e| format!("Clear previous fact relation snapshot: {e}"))?;
        tx.execute("DELETE FROM knowledge_facts", [])
            .map_err(|e| format!("Clear previous fact snapshot: {e}"))?;

        let mut count = 0;
        for fact in facts {
            let cycle = i64::try_from(fact.cycle)
                .map_err(|e| format!("Fact insertion cycle exceeds SQLite INTEGER range: {e}"))?;
            let last_accessed_cycle = fact
                .last_accessed_cycle
                .map(i64::try_from)
                .transpose()
                .map_err(|e| format!("Fact last-access cycle exceeds SQLite INTEGER range: {e}"))?;
            let graph_fact_id = encode_optional_graph_fact_id(fact.graph_fact_id)?;

            tx.execute(
                "INSERT INTO knowledge_facts
                 (vector_blob, source_text, confidence, domain, cycle, is_causal,
                  initial_confidence, last_accessed_cycle, corroboration_count, contradiction_count, graph_fact_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                rusqlite::params![
                    fact.vector_bytes,
                    fact.source_text,
                    fact.confidence,
                    fact.domain,
                    cycle,
                    fact.is_causal,
                    fact.initial_confidence,
                    last_accessed_cycle,
                    fact.corroboration_count as i64,
                    fact.contradiction_count as i64,
                    graph_fact_id,
                ],
            )
            .map_err(|e| format!("Insert fact snapshot row: {e}"))?;
            let fact_row_id = tx.last_insert_rowid();
            insert_fact_relations(&tx, fact_row_id, &fact.relations)?;
            count += 1;
        }

        let candidate_next_id = next_graph_fact_id_after_records(facts)?;
        persist_graph_fact_id_high_water(&tx, candidate_next_id)?;
        tx.commit()
            .map_err(|e| format!("Commit fact snapshot: {e}"))?;
        self.total_saved += count as u64;
        Ok(count)
    }

    /// Atomically replace the complete persisted knowledge state.
    ///
    /// This operation owns snapshot semantics for facts (including their
    /// structured relations), causal edges, and ontology primitives. Unlike the
    /// legacy append/upsert methods, it removes records that are no longer present
    /// in the live state. Inputs are validated before mutation, and all related
    /// tables are replaced in one SQLite transaction so a failed write preserves
    /// the previous committed snapshot.
    pub fn replace_knowledge_snapshot(
        &mut self,
        facts: &[FactRecord],
        edges: &[CausalEdgeRecord],
        ontology: &[OntologyRecord],
        next_graph_fact_id: u64,
    ) -> Result<(usize, usize, usize), String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        // Validate and convert every lossy SQLite INTEGER boundary before
        // touching the database. A caller must never persist state that the
        // strict readers would reject on the next startup.
        let mut prepared_facts = Vec::with_capacity(facts.len());
        let mut max_graph_fact_id = 0_u64;
        for fact in facts {
            validate_fact_relations(&fact.relations)?;
            validate_hypervector_bytes(&fact.vector_bytes, 0)
                .map_err(|e| format!("Invalid fact snapshot vector: {e}"))?;
            if !fact.confidence.is_finite() || !(0.0..=1.0).contains(&fact.confidence) {
                return Err(format!(
                    "Invalid fact snapshot confidence outside [0, 1]: {}",
                    fact.confidence
                ));
            }
            if let Some(initial_confidence) = fact.initial_confidence {
                if !initial_confidence.is_finite()
                    || !(0.0..=1.0).contains(&initial_confidence)
                {
                    return Err(format!(
                        "Invalid fact initial confidence outside [0, 1]: {initial_confidence}"
                    ));
                }
            }
            let cycle = i64::try_from(fact.cycle)
                .map_err(|e| format!("Fact insertion cycle exceeds SQLite INTEGER range: {e}"))?;
            let last_accessed_cycle = fact
                .last_accessed_cycle
                .map(i64::try_from)
                .transpose()
                .map_err(|e| format!("Fact last-access cycle exceeds SQLite INTEGER range: {e}"))?;
            let graph_fact_id = encode_optional_graph_fact_id(fact.graph_fact_id)?;
            if let Some(id) = fact.graph_fact_id {
                max_graph_fact_id = max_graph_fact_id.max(id);
            }
            prepared_facts.push((fact, cycle, last_accessed_cycle, graph_fact_id));
        }

        if next_graph_fact_id == 0 || next_graph_fact_id <= max_graph_fact_id {
            return Err(format!(
                "Next graph fact ID {next_graph_fact_id} must be positive and greater than the highest persisted fact ID {max_graph_fact_id}"
            ));
        }
        i64::try_from(next_graph_fact_id)
            .map_err(|e| format!("Next graph fact ID exceeds SQLite INTEGER range: {e}"))?;

        let mut prepared_edges = Vec::with_capacity(edges.len());
        for edge in edges {
            if !edge.strength.is_finite() || edge.strength.abs() > 1.0 {
                return Err(format!(
                    "Invalid causal edge strength outside [-1, 1] for {} -> {}: {}",
                    edge.cause, edge.effect, edge.strength
                ));
            }
            let cycle = i64::try_from(edge.cycle)
                .map_err(|e| format!("Causal-edge cycle exceeds SQLite INTEGER range: {e}"))?;
            let strength = edge.strength.abs();
            let is_inhibitory = edge.is_inhibitory || edge.strength < 0.0;
            prepared_edges.push((edge, cycle, strength, is_inhibitory));
        }

        let mut prepared_ontology = Vec::with_capacity(ontology.len());
        for record in ontology {
            validate_hypervector_bytes(&record.vector_bytes, 1)
                .map_err(|e| format!("Invalid ontology snapshot vector: {e}"))?;
            if !record.utility.is_finite() {
                return Err(format!(
                    "Non-finite ontology utility for {}: {}",
                    record.name, record.utility
                ));
            }
            let usage_count = i64::try_from(record.usage_count)
                .map_err(|e| format!("Ontology usage count exceeds SQLite INTEGER range: {e}"))?;
            let created_at_cycle = i64::try_from(record.created_at_cycle)
                .map_err(|e| format!("Ontology creation cycle exceeds SQLite INTEGER range: {e}"))?;
            let last_used_cycle = i64::try_from(record.last_used_cycle)
                .map_err(|e| format!("Ontology last-used cycle exceeds SQLite INTEGER range: {e}"))?;
            prepared_ontology.push((record, usage_count, created_at_cycle, last_used_cycle));
        }

        let mut conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|e| format!("Begin knowledge snapshot transaction: {e}"))?;

        tx.execute("DELETE FROM knowledge_fact_relations", [])
            .map_err(|e| format!("Clear fact relation snapshot: {e}"))?;
        tx.execute("DELETE FROM knowledge_facts", [])
            .map_err(|e| format!("Clear fact snapshot: {e}"))?;
        tx.execute("DELETE FROM knowledge_causal_edges", [])
            .map_err(|e| format!("Clear causal-edge snapshot: {e}"))?;
        tx.execute("DELETE FROM knowledge_ontology", [])
            .map_err(|e| format!("Clear ontology snapshot: {e}"))?;

        for (fact, cycle, last_accessed_cycle, graph_fact_id) in &prepared_facts {
            tx.execute(
                "INSERT INTO knowledge_facts
                 (vector_blob, source_text, confidence, domain, cycle, is_causal,
                  initial_confidence, last_accessed_cycle, corroboration_count, contradiction_count, graph_fact_id)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                rusqlite::params![
                    fact.vector_bytes,
                    fact.source_text,
                    fact.confidence,
                    fact.domain,
                    cycle,
                    fact.is_causal,
                    fact.initial_confidence,
                    last_accessed_cycle,
                    fact.corroboration_count as i64,
                    fact.contradiction_count as i64,
                    graph_fact_id,
                ],
            )
            .map_err(|e| format!("Insert fact snapshot row: {e}"))?;
            let fact_row_id = tx.last_insert_rowid();
            insert_fact_relations(&tx, fact_row_id, &fact.relations)?;
        }

        for (edge, cycle, strength, is_inhibitory) in &prepared_edges {
            // Unlike the legacy save API, a full snapshot preserves every edge
            // observation, including same-pair observations with distinct source
            // text, negation, inhibitory polarity, or discovery cycle.
            tx.execute(
                "INSERT INTO knowledge_causal_edges
                 (cause, effect, strength, is_inhibitory, is_negated, source_text, cycle)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    edge.cause,
                    edge.effect,
                    strength,
                    is_inhibitory,
                    edge.is_negated,
                    edge.source_text,
                    cycle,
                ],
            )
            .map_err(|e| format!("Insert causal-edge snapshot row: {e}"))?;
        }

        for (record, usage_count, created_at_cycle, last_used_cycle) in &prepared_ontology {
            tx.execute(
                "INSERT INTO knowledge_ontology
                 (name, vector_blob, usage_count, utility, created_at_cycle, last_used_cycle, is_a_parent)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    record.name,
                    record.vector_bytes,
                    usage_count,
                    record.utility,
                    created_at_cycle,
                    last_used_cycle,
                    record.is_a_parent,
                ],
            )
            .map_err(|e| format!("Insert ontology snapshot row: {e}"))?;
        }

        // Preserve the highest allocator value ever committed, including IDs of
        // facts removed by this snapshot; never allow stale references to alias a
        // different fact created after restart.
        persist_graph_fact_id_high_water(&tx, Some(next_graph_fact_id))?;
        tx.commit()
            .map_err(|e| format!("Commit knowledge snapshot: {e}"))?;
        self.total_saved += (facts.len() + edges.len() + ontology.len()) as u64;
        Ok((facts.len(), edges.len(), ontology.len()))
    }

    /// Load the next graph fact ID reserved by the most recent complete snapshot.
    ///
    /// Returns None for legacy databases that predate allocator persistence.
    pub fn load_next_graph_fact_id(&mut self) -> Result<Option<u64>, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }
        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let stored: Option<i64> = conn
            .query_row(
                "SELECT value FROM knowledge_metadata WHERE key = 'next_graph_fact_id'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| format!("Read next graph fact ID: {e}"))?;
        stored
            .map(|value| {
                u64::try_from(value)
                    .ok()
                    .filter(|id| *id > 0)
                    .ok_or_else(|| {
                        format!("Invalid persisted next graph fact ID: {value}")
                    })
            })
            .transpose()
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
                "SELECT vector_blob, source_text, confidence, domain, cycle, is_causal,
                        initial_confidence, last_accessed_cycle, corroboration_count, contradiction_count,
                        graph_fact_id, id
                 FROM knowledge_facts ORDER BY cycle DESC",
            )
            .map_err(|e| e.to_string())?;

        let mut facts_with_row_id: Vec<(i64, FactRecord)> = stmt
            .query_map([], |row| {
                let vector_bytes: Vec<u8> = row.get(0)?;
                validate_hypervector_bytes(&vector_bytes, 0)?;
                let confidence = decode_confidence(row.get(2)?, 2)?;
                let cycle = decode_nonnegative_counter(row.get(4)?, 4, "fact cycle")?;
                let initial_confidence = row
                    .get::<_, Option<f32>>(6)?
                    .map(|value| decode_confidence(value, 6))
                    .transpose()?;
                let last_accessed_cycle = row
                    .get::<_, Option<i64>>(7)?
                    .map(|value| decode_nonnegative_counter(value, 7, "fact last-access cycle"))
                    .transpose()?;
                let corroboration_count =
                    decode_nonnegative_u32_counter(row.get(8)?, 8, "fact corroboration count")?;
                let contradiction_count =
                    decode_nonnegative_u32_counter(row.get(9)?, 9, "fact contradiction count")?;
                let graph_fact_id = decode_optional_graph_fact_id(row.get(10)?, 10)?;
                let fact_row_id: i64 = row.get(11)?;
                Ok((
                    fact_row_id,
                    FactRecord {
                        vector_bytes,
                        source_text: row.get(1)?,
                        confidence,
                        domain: row.get(3)?,
                        cycle,
                        is_causal: row.get(5)?,
                        relations: Vec::new(),
                        graph_fact_id,
                        initial_confidence,
                        last_accessed_cycle,
                        corroboration_count,
                        contradiction_count,
                    },
                ))
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Collect: {e}"))?;
        drop(stmt);

        let mut relations_by_fact = load_fact_relations(&conn)?;
        let fact_row_ids: std::collections::HashSet<i64> =
            facts_with_row_id.iter().map(|(id, _)| *id).collect();
        if let Some(orphan_id) = relations_by_fact
            .keys()
            .find(|id| !fact_row_ids.contains(*id))
        {
            return Err(format!(
                "Structured relation rows refer to missing fact row {orphan_id}"
            ));
        }
        for (fact_row_id, fact) in &mut facts_with_row_id {
            fact.relations = relations_by_fact.remove(fact_row_id).unwrap_or_default();
        }

        let facts: Vec<FactRecord> = facts_with_row_id
            .into_iter()
            .map(|(_, fact)| fact)
            .collect();
        self.total_loaded += facts.len() as u64;
        Ok(facts)
    }

    /// Save causal edge records.
    pub fn save_causal_edges(&mut self, edges: &[CausalEdgeRecord]) -> Result<usize, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        conn.execute_batch("BEGIN TRANSACTION")
            .map_err(|e| e.to_string())?;

        let mut count = 0;
        for edge in edges {
            if !edge.strength.is_finite() || edge.strength.abs() > 1.0 {
                return Err(format!(
                    "Invalid causal edge strength outside [-1, 1] for {} -> {}: {}",
                    edge.cause, edge.effect, edge.strength
                ));
            }
            let cycle = i64::try_from(edge.cycle)
                .map_err(|e| format!("Causal-edge cycle exceeds SQLite INTEGER range: {e}"))?;
            // Preserve the historical upsert-by-pair behavior of this legacy API.
            // The complete snapshot API below preserves repeated observations.
            conn.execute(
                "DELETE FROM knowledge_causal_edges WHERE cause = ?1 AND effect = ?2",
                rusqlite::params![edge.cause, edge.effect],
            )
            .map_err(|e| format!("Replace existing causal edge: {e}"))?;
            conn.execute(
                "INSERT INTO knowledge_causal_edges
                 (cause, effect, strength, is_inhibitory, is_negated, source_text, cycle)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                rusqlite::params![
                    edge.cause,
                    edge.effect,
                    edge.strength.abs(),
                    edge.is_inhibitory || edge.strength < 0.0,
                    edge.is_negated,
                    edge.source_text,
                    cycle,
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
                "SELECT cause, effect, strength, is_inhibitory, is_negated, source_text, cycle
                 FROM knowledge_causal_edges ORDER BY edge_id",
            )
            .map_err(|e| e.to_string())?;

        let edges: Vec<CausalEdgeRecord> = stmt
            .query_map([], |row| {
                let strength = decode_finite_f32(row.get(2)?, 2, "causal strength")?;
                if strength < 0.0 || strength > 1.0 {
                    return Err(invalid_column_data(
                        2,
                        rusqlite::types::Type::Real,
                        format!("persisted causal strength outside [0, 1]: {strength}"),
                    ));
                }
                let cycle = decode_nonnegative_counter(row.get(6)?, 6, "causal-edge cycle")?;
                Ok(CausalEdgeRecord {
                    cause: row.get(0)?,
                    effect: row.get(1)?,
                    strength,
                    is_inhibitory: row.get(3)?,
                    is_negated: row.get(4)?,
                    source_text: row.get(5)?,
                    cycle,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Collect: {e}"))?;

        self.total_loaded += edges.len() as u64;
        Ok(edges)
    }

    /// Save ontology primitives to the database.
    ///
    /// Uses INSERT OR REPLACE for upsert behavior.
    /// Returns the number of primitives saved.
    pub fn save_ontology(&mut self, records: &[OntologyRecord]) -> Result<usize, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }
        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Tx: {e}"))?;
        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT OR REPLACE INTO knowledge_ontology
                     (name, vector_blob, usage_count, utility, created_at_cycle, last_used_cycle, is_a_parent)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                )
                .map_err(|e| format!("Prepare: {e}"))?;

            for r in records {
                stmt.execute(rusqlite::params![
                    r.name,
                    r.vector_bytes,
                    r.usage_count as i64,
                    r.utility,
                    r.created_at_cycle as i64,
                    r.last_used_cycle as i64,
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
                 FROM knowledge_ontology ORDER BY utility DESC",
            )
            .map_err(|e| format!("Prepare: {e}"))?;

        let records: Vec<OntologyRecord> = stmt
            .query_map([], |row| {
                let vector_bytes: Vec<u8> = row.get(1)?;
                validate_hypervector_bytes(&vector_bytes, 1)?;
                let usage_count = decode_nonnegative_counter(row.get(2)?, 2, "ontology usage count")?;
                let utility = decode_finite_f64(row.get(3)?, 3, "ontology utility")?;
                let created_at_cycle = decode_nonnegative_counter(row.get(4)?, 4, "ontology creation cycle")?;
                let last_used_cycle = decode_nonnegative_counter(row.get(5)?, 5, "ontology last-used cycle")?;
                Ok(OntologyRecord {
                    name: row.get(0)?,
                    vector_bytes,
                    usage_count,
                    utility,
                    created_at_cycle,
                    last_used_cycle,
                    is_a_parent: row.get(6)?,
                })
            })
            .map_err(|e| format!("Query: {e}"))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Collect: {e}"))?;

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
        rusqlite::Connection::open(&self.db_path).map_err(|e| format!("SQLite open: {e}"))
    }

    fn ensure_schema(&mut self, conn: &rusqlite::Connection) -> Result<(), String> {
        if self.initialized {
            return Ok(());
        }

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS knowledge_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vector_blob BLOB NOT NULL,
                source_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                domain TEXT,
                cycle INTEGER NOT NULL,
                is_causal INTEGER NOT NULL DEFAULT 0,
                initial_confidence REAL,
                last_accessed_cycle INTEGER,
                corroboration_count INTEGER NOT NULL DEFAULT 0,
                contradiction_count INTEGER NOT NULL DEFAULT 0,
                graph_fact_id INTEGER
            );
            CREATE TABLE IF NOT EXISTS knowledge_fact_relations (
                fact_row_id INTEGER NOT NULL,
                ordinal INTEGER NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                subject_role TEXT NOT NULL,
                object_role TEXT NOT NULL,
                is_causal INTEGER NOT NULL,
                is_negated INTEGER NOT NULL,
                confidence REAL NOT NULL,
                PRIMARY KEY (fact_row_id, ordinal),
                FOREIGN KEY (fact_row_id) REFERENCES knowledge_facts(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS knowledge_causal_edges (
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cause TEXT NOT NULL,
                effect TEXT NOT NULL,
                strength REAL NOT NULL,
                is_inhibitory INTEGER NOT NULL DEFAULT 0,
                is_negated INTEGER NOT NULL DEFAULT 0,
                source_text TEXT NOT NULL DEFAULT '',
                cycle INTEGER NOT NULL
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
            CREATE TABLE IF NOT EXISTS knowledge_metadata (
                key TEXT PRIMARY KEY NOT NULL,
                value INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_facts_domain ON knowledge_facts(domain);
            CREATE INDEX IF NOT EXISTS idx_facts_cycle ON knowledge_facts(cycle);",
        )
        .map_err(|e| format!("Schema init: {e}"))?;

        // The legacy causal table has PRIMARY KEY(cause, effect), which collapses
        // repeated observations and cannot retain negation or source text. Migrate it
        // once to an observation table with a surrogate key. Legacy signed strengths
        // encoded inhibition; normalize to non-negative magnitude plus an explicit flag.
        let edge_columns = {
            let mut stmt = conn
                .prepare("PRAGMA table_info(knowledge_causal_edges)")
                .map_err(|e| format!("Inspect knowledge_causal_edges schema: {e}"))?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(1))
                .map_err(|e| format!("Query knowledge_causal_edges schema: {e}"))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Collect knowledge_causal_edges schema: {e}"))?
        };
        if !edge_columns.iter().any(|column| column == "edge_id") {
            conn.execute_batch(
                "BEGIN IMMEDIATE;
                 ALTER TABLE knowledge_causal_edges RENAME TO knowledge_causal_edges_legacy;
                 CREATE TABLE knowledge_causal_edges (
                     edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                     cause TEXT NOT NULL,
                     effect TEXT NOT NULL,
                     strength REAL NOT NULL,
                     is_inhibitory INTEGER NOT NULL DEFAULT 0,
                     is_negated INTEGER NOT NULL DEFAULT 0,
                     source_text TEXT NOT NULL DEFAULT '',
                     cycle INTEGER NOT NULL
                 );
                 INSERT INTO knowledge_causal_edges
                     (cause, effect, strength, is_inhibitory, is_negated, source_text, cycle)
                 SELECT cause, effect, ABS(strength),
                        CASE WHEN is_inhibitory != 0 OR strength < 0 THEN 1 ELSE 0 END,
                        0, '', cycle
                 FROM knowledge_causal_edges_legacy;
                 DROP TABLE knowledge_causal_edges_legacy;
                 COMMIT;",
            )
            .map_err(|e| format!("Migrate causal edge observations: {e}"))?;
        }
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_causal_edges_pair
             ON knowledge_causal_edges(cause, effect);",
        )
        .map_err(|e| format!("Index causal edge observations: {e}"))?;

        // Additive migration preserves pre-existing rows. NULL on initial confidence
        // and last-access cycle means the legacy schema never recorded those values.
        let columns = {
            let mut stmt = conn
                .prepare("PRAGMA table_info(knowledge_facts)")
                .map_err(|e| format!("Inspect knowledge_facts schema: {e}"))?;
            let rows = stmt
                .query_map([], |row| row.get::<_, String>(1))
                .map_err(|e| format!("Query knowledge_facts schema: {e}"))?;
            rows.collect::<Result<Vec<_>, _>>()
                .map_err(|e| format!("Collect knowledge_facts schema: {e}"))?
        };
        for (name, declaration) in [
            ("initial_confidence", "REAL"),
            ("last_accessed_cycle", "INTEGER"),
            ("corroboration_count", "INTEGER NOT NULL DEFAULT 0"),
            ("contradiction_count", "INTEGER NOT NULL DEFAULT 0"),
            ("graph_fact_id", "INTEGER"),
        ] {
            if !columns.iter().any(|column| column == name) {
                conn.execute_batch(&format!(
                    "ALTER TABLE knowledge_facts ADD COLUMN {name} {declaration}"
                ))
                .map_err(|e| format!("Migrate knowledge_facts.{name}: {e}"))?;
            }
        }

        conn.execute_batch(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_graph_fact_id
             ON knowledge_facts(graph_fact_id);",
        )
        .map_err(|e| format!("Create graph fact ID index: {e}"))?;

        self.initialized = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_fact_relation_round_trip_preserves_negation_roles_and_confidence() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_fact_relation_roundtrip_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);

        let record = FactRecord {
            vector_bytes: vec![3u8; BinaryHV::BYTES],
            source_text: "The ceasefire did not prevent escalation.".into(),
            confidence: 0.81,
            domain: Some("geopolitics".into()),
            cycle: 12,
            is_causal: true,
            relations: vec![ExtractedRelation {
                subject: "ceasefire".into(),
                predicate: "prevented".into(),
                object: "escalation".into(),
                subject_role: SemanticRole::Agent,
                object_role: SemanticRole::Patient,
                is_causal: true,
                is_negated: true,
                confidence: 0.73,
            }],
            graph_fact_id: Some(17),
            initial_confidence: Some(0.88),
            last_accessed_cycle: Some(15),
            corroboration_count: 2,
            contradiction_count: 1,
        };

        assert_eq!(persistence.save_facts(&[record]).unwrap(), 1);
        let loaded = persistence.load_facts().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].relations.len(), 1);
        let relation = &loaded[0].relations[0];
        assert_eq!(relation.subject, "ceasefire");
        assert_eq!(relation.predicate, "prevented");
        assert_eq!(relation.object, "escalation");
        assert_eq!(relation.subject_role, SemanticRole::Agent);
        assert_eq!(relation.object_role, SemanticRole::Patient);
        assert!(relation.is_causal);
        assert!(relation.is_negated);
        assert!((relation.confidence - 0.73).abs() < f32::EPSILON);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_full_snapshot_rejects_invalid_relation_without_replacing_prior_state() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_invalid_relation_snapshot_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);

        let old_fact = FactRecord {
            vector_bytes: vec![1u8; BinaryHV::BYTES],
            source_text: "previous committed fact".into(),
            confidence: 0.8,
            domain: None,
            cycle: 1,
            is_causal: false,
            relations: Vec::new(),
            graph_fact_id: Some(1),
            initial_confidence: Some(0.8),
            last_accessed_cycle: Some(1),
            corroboration_count: 0,
            contradiction_count: 0,
        };
        persistence
            .replace_knowledge_snapshot(&[old_fact], &[], &[], 2)
            .unwrap();

        let invalid_fact = FactRecord {
            vector_bytes: vec![2u8; BinaryHV::BYTES],
            source_text: "invalid candidate".into(),
            confidence: 0.7,
            domain: None,
            cycle: 2,
            is_causal: true,
            relations: vec![ExtractedRelation {
                subject: "A".into(),
                predicate: "caused".into(),
                object: "B".into(),
                subject_role: SemanticRole::Agent,
                object_role: SemanticRole::Patient,
                is_causal: true,
                is_negated: false,
                confidence: 1.2,
            }],
            graph_fact_id: Some(2),
            initial_confidence: Some(0.7),
            last_accessed_cycle: Some(2),
            corroboration_count: 0,
            contradiction_count: 0,
        };
        let error = persistence
            .replace_knowledge_snapshot(&[invalid_fact], &[], &[], 3)
            .unwrap_err();
        assert!(error.contains("Invalid extracted-relation confidence"), "{error}");

        let restored = persistence.load_facts().unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored[0].source_text, "previous committed fact");
        assert_eq!(restored[0].graph_fact_id, Some(1));
        assert_eq!(restored[0].relations.len(), 0);
        assert_eq!(persistence.load_next_graph_fact_id().unwrap(), Some(2));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_facts_rejects_unknown_persisted_relation_role() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_unknown_relation_role_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence
            .save_facts(&[FactRecord {
                vector_bytes: vec![5u8; BinaryHV::BYTES],
                source_text: "source fact".into(),
                confidence: 0.9,
                domain: None,
                cycle: 1,
                is_causal: false,
                relations: vec![ExtractedRelation {
                    subject: "A".into(),
                    predicate: "blocks".into(),
                    object: "B".into(),
                    subject_role: SemanticRole::Agent,
                    object_role: SemanticRole::Patient,
                    is_causal: false,
                    is_negated: false,
                    confidence: 0.7,
                }],
                graph_fact_id: None,
                initial_confidence: None,
                last_accessed_cycle: None,
                corroboration_count: 0,
                contradiction_count: 0,
            }])
            .unwrap();

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute(
            "UPDATE knowledge_fact_relations SET subject_role = 'unknown_role'",
            [],
        )
        .unwrap();
        drop(conn);

        let error = persistence.load_facts().unwrap_err();
        assert!(error.contains("unknown persisted semantic role key"), "{error}");
        assert_eq!(persistence.total_loaded(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_facts_propagates_row_decode_errors() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_corrupt_facts_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");

        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence
            .save_facts(&[FactRecord {
                vector_bytes: vec![0u8; 2048],
                source_text: "fixture fact".into(),
                confidence: 0.9,
                domain: Some("test".into()),
                cycle: 1,
                is_causal: false,
                relations: Vec::new(),
                graph_fact_id: None,
                initial_confidence: None,
                last_accessed_cycle: None,
                corroboration_count: 0,
                contradiction_count: 0,
            }])
            .unwrap();

        // SQLite's dynamic typing permits this value in the REAL-affinity column.
        // Decoding it as f32 must fail closed rather than silently omit the fact.
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute(
            "UPDATE knowledge_facts SET confidence = ?1",
            rusqlite::params!["not-a-number"],
        )
        .unwrap();
        drop(conn);

        let err = persistence.load_facts().unwrap_err();
        assert!(err.contains("Collect:"), "unexpected error context: {err}");
        assert_eq!(persistence.total_loaded(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }


    #[test]
    fn test_load_facts_rejects_wrong_hypervector_length() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_wrong_fact_vector_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence
            .save_facts(&[FactRecord {
                vector_bytes: vec![0u8; BinaryHV::BYTES],
                source_text: "fixture fact".into(),
                confidence: 0.9,
                domain: Some("test".into()),
                cycle: 1,
                is_causal: false,
                relations: Vec::new(),
                graph_fact_id: None,
                initial_confidence: None,
                last_accessed_cycle: None,
                corroboration_count: 0,
                contradiction_count: 0,
            }])
            .unwrap();

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute(
            "UPDATE knowledge_facts SET vector_blob = ?1",
            rusqlite::params![vec![0u8; 1]],
        )
        .unwrap();
        drop(conn);

        let error = persistence.load_facts().unwrap_err();
        assert!(error.contains("invalid hypervector length"), "{error}");
        assert_eq!(persistence.total_loaded(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_facts_rejects_out_of_range_confidence() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_invalid_fact_confidence_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence
            .save_facts(&[FactRecord {
                vector_bytes: vec![0u8; BinaryHV::BYTES],
                source_text: "fixture fact".into(),
                confidence: 0.9,
                domain: Some("test".into()),
                cycle: 1,
                is_causal: false,
                relations: Vec::new(),
                graph_fact_id: None,
                initial_confidence: None,
                last_accessed_cycle: None,
                corroboration_count: 0,
                contradiction_count: 0,
            }])
            .unwrap();

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute("UPDATE knowledge_facts SET confidence = 1.25", [])
            .unwrap();
        drop(conn);

        let error = persistence.load_facts().unwrap_err();
        assert!(error.contains("invalid persisted confidence"), "{error}");
        assert_eq!(persistence.total_loaded(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_facts_rejects_nonpositive_graph_fact_id() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_nonpositive_graph_fact_id_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence
            .save_facts(&[FactRecord {
                vector_bytes: vec![0u8; BinaryHV::BYTES],
                source_text: "fixture fact".into(),
                confidence: 0.9,
                domain: None,
                cycle: 1,
                is_causal: false,
                relations: Vec::new(),
                graph_fact_id: Some(1),
                initial_confidence: Some(0.9),
                last_accessed_cycle: Some(1),
                corroboration_count: 0,
                contradiction_count: 0,
            }])
            .unwrap();

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute("UPDATE knowledge_facts SET graph_fact_id = 0", [])
            .unwrap();
        drop(conn);

        let error = persistence.load_facts().unwrap_err();
        assert!(error.contains("invalid persisted graph fact ID"), "{error}");
        assert_eq!(persistence.total_loaded(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_facts_rejects_negative_cycle() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_negative_fact_cycle_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence
            .save_facts(&[FactRecord {
                vector_bytes: vec![0u8; BinaryHV::BYTES],
                source_text: "fixture fact".into(),
                confidence: 0.9,
                domain: Some("test".into()),
                cycle: 1,
                is_causal: false,
                relations: Vec::new(),
                graph_fact_id: None,
                initial_confidence: None,
                last_accessed_cycle: None,
                corroboration_count: 0,
                contradiction_count: 0,
            }])
            .unwrap();

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute("UPDATE knowledge_facts SET cycle = -1", []).unwrap();
        drop(conn);

        let error = persistence.load_facts().unwrap_err();
        assert!(error.contains("invalid negative persisted fact cycle"), "{error}");
        assert_eq!(persistence.total_loaded(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_facts_rejects_negative_last_access_cycle() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_negative_fact_access_cycle_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence
            .save_facts(&[FactRecord {
                vector_bytes: vec![0u8; BinaryHV::BYTES],
                source_text: "fixture fact".into(),
                confidence: 0.9,
                domain: None,
                cycle: 3,
                is_causal: false,
                relations: Vec::new(),
                graph_fact_id: None,
                initial_confidence: Some(0.9),
                last_accessed_cycle: Some(3),
                corroboration_count: 0,
                contradiction_count: 0,
            }])
            .unwrap();

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute("UPDATE knowledge_facts SET last_accessed_cycle = -1", [])
            .unwrap();
        drop(conn);

        let error = persistence.load_facts().unwrap_err();
        assert!(error.contains("invalid negative persisted fact last-access cycle"), "{error}");
        assert_eq!(persistence.total_loaded(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_ontology_rejects_wrong_hypervector_length() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_wrong_ontology_vector_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence
            .save_ontology(&[OntologyRecord {
                name: "fixture primitive".into(),
                vector_bytes: vec![0u8; BinaryHV::BYTES],
                usage_count: 1,
                utility: 0.5,
                created_at_cycle: 1,
                last_used_cycle: 1,
                is_a_parent: None,
            }])
            .unwrap();

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute(
            "UPDATE knowledge_ontology SET vector_blob = ?1",
            rusqlite::params![vec![0u8; BinaryHV::BYTES + 1]],
        )
        .unwrap();
        drop(conn);

        let error = persistence.load_ontology().unwrap_err();
        assert!(error.contains("invalid hypervector length"), "{error}");
        assert_eq!(persistence.total_loaded(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_causal_edges_rejects_negative_cycle() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_negative_edge_cycle_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence
            .save_causal_edges(&[CausalEdgeRecord {
                cause: "fixture cause".into(),
                effect: "fixture effect".into(),
                strength: -0.4,
                is_inhibitory: true,
                is_negated: false,
                source_text: String::new(),
                cycle: 1,
            }])
            .unwrap();

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute("UPDATE knowledge_causal_edges SET cycle = -1", [])
            .unwrap();
        drop(conn);

        let error = persistence.load_causal_edges().unwrap_err();
        assert!(error.contains("invalid negative persisted causal-edge cycle"), "{error}");
        assert_eq!(persistence.total_loaded(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_causal_edges_propagates_row_decode_errors() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_corrupt_edges_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");

        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence
            .save_causal_edges(&[CausalEdgeRecord {
                cause: "fixture cause".into(),
                effect: "fixture effect".into(),
                strength: 0.8,
                is_inhibitory: false,
                is_negated: false,
                source_text: String::new(),
                cycle: 1,
            }])
            .unwrap();

        // A corrupt numeric value must be surfaced, not treated as an absent edge.
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute(
            "UPDATE knowledge_causal_edges SET strength = ?1",
            rusqlite::params!["not-a-number"],
        )
        .unwrap();
        drop(conn);

        let err = persistence.load_causal_edges().unwrap_err();
        assert!(err.contains("Collect:"), "unexpected error context: {err}");
        assert_eq!(persistence.total_loaded(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_old_knowledge_facts_schema_migrates_without_losing_legacy_rows() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_legacy_knowledge_schema_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");

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
             (vector_blob, source_text, confidence, domain, cycle, is_causal)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                vec![0u8; BinaryHV::BYTES],
                "legacy fact",
                0.4_f32,
                "legacy",
                7_i64,
                false
            ],
        )
        .unwrap();
        drop(conn);

        let mut persistence = KnowledgePersistence::new(&db_path);
        let records = persistence.load_facts().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].source_text, "legacy fact");
        assert_eq!(records[0].graph_fact_id, None);
        assert_eq!(records[0].initial_confidence, None);
        assert_eq!(records[0].last_accessed_cycle, None);
        assert_eq!(records[0].corroboration_count, 0);
        assert_eq!(records[0].contradiction_count, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_complete_snapshot_removes_stale_edges_and_ontology_records() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_complete_snapshot_replace_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);

        let fact = FactRecord {
            vector_bytes: vec![7u8; BinaryHV::BYTES],
            source_text: "live fact".into(),
            confidence: 0.8,
            domain: Some("test".into()),
            cycle: 7,
            is_causal: true,
            relations: Vec::new(),
            graph_fact_id: None,
            initial_confidence: Some(0.9),
            last_accessed_cycle: Some(8),
            corroboration_count: 2,
            contradiction_count: 1,
        };
        let stale_edge = CausalEdgeRecord {
            cause: "stale-sentinel-cause".into(),
            effect: "stale-sentinel-effect".into(),
            strength: 0.7,
            is_inhibitory: false,
            is_negated: false,
            source_text: "affirmative source observation".into(),
            cycle: 1,
        };
        let stale_primitive = OntologyRecord {
            name: "stale-sentinel-primitive".into(),
            vector_bytes: vec![9u8; BinaryHV::BYTES],
            usage_count: 2,
            utility: 0.4,
            created_at_cycle: 1,
            last_used_cycle: 2,
            is_a_parent: None,
        };

        let duplicate_edge = CausalEdgeRecord {
            strength: 0.8,
            is_inhibitory: true,
            is_negated: true,
            source_text: "negated inhibitory source observation".into(),
            cycle: 2,
            ..stale_edge.clone()
        };
        assert_eq!(
            persistence
                .replace_knowledge_snapshot(
                    &[fact.clone()],
                    &[stale_edge, duplicate_edge],
                    &[stale_primitive],
                    2,
                )
                .unwrap(),
            (1, 2, 1)
        );
        let persisted_observations = persistence.load_causal_edges().unwrap();
        assert_eq!(
            persisted_observations.len(),
            2,
            "complete snapshots must retain repeated same-pair observations"
        );
        assert_eq!(
            persisted_observations[0].source_text,
            "affirmative source observation"
        );
        assert!(!persisted_observations[0].is_negated);
        assert!(!persisted_observations[0].is_inhibitory);
        assert_eq!(persisted_observations[1].strength, 0.8);
        assert!(persisted_observations[1].is_inhibitory);
        assert!(persisted_observations[1].is_negated);
        assert_eq!(
            persisted_observations[1].source_text,
            "negated inhibitory source observation"
        );
        assert_eq!(persisted_observations[1].cycle, 2);

        let replacement_fact = FactRecord {
            source_text: "replacement live fact".into(),
            ..fact
        };
        assert_eq!(
            persistence
                .replace_knowledge_snapshot(&[replacement_fact], &[], &[], 2)
                .unwrap(),
            (1, 0, 0)
        );

        let facts = persistence.load_facts().unwrap();
        let edges = persistence.load_causal_edges().unwrap();
        let ontology = persistence.load_ontology().unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].source_text, "replacement live fact");
        assert!(edges.is_empty(), "removed causal edges must not reappear after restart");
        assert!(
            ontology.is_empty(),
            "pruned ontology primitives must not reappear after restart"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_fact_only_snapshot_does_not_lower_allocator_high_water() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_fact_only_allocator_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);

        let prior = FactRecord {
            vector_bytes: vec![1u8; BinaryHV::BYTES],
            source_text: "prior fact".into(),
            confidence: 0.8,
            domain: None,
            cycle: 1,
            is_causal: false,
            relations: Vec::new(),
            graph_fact_id: Some(5),
            initial_confidence: Some(0.8),
            last_accessed_cycle: Some(1),
            corroboration_count: 0,
            contradiction_count: 0,
        };
        persistence
            .replace_knowledge_snapshot(&[prior], &[], &[], 12)
            .unwrap();

        let replacement = FactRecord {
            vector_bytes: vec![2u8; BinaryHV::BYTES],
            source_text: "fact-only replacement".into(),
            confidence: 0.7,
            domain: None,
            cycle: 2,
            is_causal: false,
            relations: Vec::new(),
            graph_fact_id: Some(7),
            initial_confidence: Some(0.7),
            last_accessed_cycle: Some(2),
            corroboration_count: 0,
            contradiction_count: 0,
        };
        persistence.replace_facts_snapshot(&[replacement]).unwrap();

        assert_eq!(
            persistence.load_next_graph_fact_id().unwrap(),
            Some(12),
            "fact-only snapshots must not lower the globally committed ID high-water mark"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_next_graph_fact_id_rejects_corrupt_allocator_metadata() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_corrupt_fact_allocator_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence.save_facts(&[]).unwrap();

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO knowledge_metadata (key, value)
             VALUES ('next_graph_fact_id', 0)",
            [],
        )
        .unwrap();
        drop(conn);

        let error = persistence.load_next_graph_fact_id().unwrap_err();
        assert!(error.contains("Invalid persisted next graph fact ID"), "{error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_allocator_high_water_survives_removal_of_highest_fact() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_fact_allocator_high_water_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);

        let retained = FactRecord {
            vector_bytes: vec![1u8; BinaryHV::BYTES],
            source_text: "retained fact".into(),
            confidence: 0.8,
            domain: None,
            cycle: 1,
            is_causal: false,
            relations: Vec::new(),
            graph_fact_id: Some(1),
            initial_confidence: Some(0.8),
            last_accessed_cycle: Some(1),
            corroboration_count: 0,
            contradiction_count: 0,
        };
        let soon_removed = FactRecord {
            vector_bytes: vec![2u8; BinaryHV::BYTES],
            source_text: "fact retired before next restart".into(),
            confidence: 0.4,
            domain: None,
            cycle: 2,
            is_causal: false,
            relations: Vec::new(),
            graph_fact_id: Some(2),
            initial_confidence: Some(0.4),
            last_accessed_cycle: Some(2),
            corroboration_count: 0,
            contradiction_count: 0,
        };

        persistence
            .replace_knowledge_snapshot(
                &[retained.clone(), soon_removed],
                &[],
                &[],
                3,
            )
            .unwrap();
        // Even a stale-but-valid lower candidate must not decrease a committed
        // allocator high-water mark.
        persistence
            .replace_knowledge_snapshot(&[retained], &[], &[], 2)
            .unwrap();

        let records = persistence.load_facts().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].graph_fact_id, Some(1));
        assert_eq!(
            persistence.load_next_graph_fact_id().unwrap(),
            Some(3),
            "deleting the highest-ID fact must not lower the persisted allocator"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_complete_snapshot_rolls_back_all_tables_on_insert_failure() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_complete_snapshot_rollback_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);

        let old_fact = FactRecord {
            vector_bytes: vec![1u8; BinaryHV::BYTES],
            source_text: "old fact".into(),
            confidence: 0.8,
            domain: None,
            cycle: 3,
            is_causal: false,
            relations: Vec::new(),
            graph_fact_id: None,
            initial_confidence: Some(0.9),
            last_accessed_cycle: Some(4),
            corroboration_count: 0,
            contradiction_count: 0,
        };
        let old_edge = CausalEdgeRecord {
            cause: "old cause".into(),
            effect: "old effect".into(),
            strength: 0.5,
            is_inhibitory: false,
            is_negated: false,
            source_text: String::new(),
            cycle: 3,
        };
        let old_primitive = OntologyRecord {
            name: "old primitive".into(),
            vector_bytes: vec![2u8; BinaryHV::BYTES],
            usage_count: 1,
            utility: 0.5,
            created_at_cycle: 1,
            last_used_cycle: 2,
            is_a_parent: None,
        };
        persistence
            .replace_knowledge_snapshot(&[old_fact], &[old_edge], &[old_primitive], 2)
            .unwrap();

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TRIGGER reject_snapshot_primitive
             BEFORE INSERT ON knowledge_ontology
             WHEN NEW.name = 'reject-snapshot'
             BEGIN
                 SELECT RAISE(ABORT, 'injected snapshot insert failure');
             END;",
        )
        .unwrap();
        drop(conn);

        let new_fact = FactRecord {
            vector_bytes: vec![3u8; BinaryHV::BYTES],
            source_text: "new fact that must roll back".into(),
            confidence: 0.7,
            domain: None,
            cycle: 10,
            is_causal: false,
            relations: Vec::new(),
            graph_fact_id: None,
            initial_confidence: Some(0.7),
            last_accessed_cycle: Some(10),
            corroboration_count: 0,
            contradiction_count: 0,
        };
        let new_edge = CausalEdgeRecord {
            cause: "new cause".into(),
            effect: "new effect".into(),
            strength: 0.6,
            is_inhibitory: false,
            is_negated: false,
            source_text: String::new(),
            cycle: 10,
        };
        let rejected_primitive = OntologyRecord {
            name: "reject-snapshot".into(),
            vector_bytes: vec![4u8; BinaryHV::BYTES],
            usage_count: 1,
            utility: 0.2,
            created_at_cycle: 10,
            last_used_cycle: 10,
            is_a_parent: None,
        };

        let error = persistence
            .replace_knowledge_snapshot(&[new_fact], &[new_edge], &[rejected_primitive], 4)
            .unwrap_err();
        assert!(error.contains("injected snapshot insert failure"), "{error}");

        let facts = persistence.load_facts().unwrap();
        let edges = persistence.load_causal_edges().unwrap();
        let ontology = persistence.load_ontology().unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].source_text, "old fact");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].cause, "old cause");
        assert_eq!(ontology.len(), 1);
        assert_eq!(ontology[0].name, "old primitive");
        assert_eq!(
            persistence.load_next_graph_fact_id().unwrap(),
            Some(2),
            "failed replacement must preserve the prior committed allocator state"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_fact_snapshot_replacement_does_not_accumulate_rows() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_fact_snapshot_repeat_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);
        let fact = FactRecord {
            vector_bytes: vec![7u8; BinaryHV::BYTES],
            source_text: "snapshot fixture".into(),
            confidence: 0.8,
            domain: Some("test".into()),
            cycle: 7,
            is_causal: false,
            relations: Vec::new(),
            graph_fact_id: None,
            initial_confidence: Some(0.9),
            last_accessed_cycle: Some(8),
            corroboration_count: 2,
            contradiction_count: 1,
        };

        assert_eq!(persistence.replace_facts_snapshot(&[fact.clone()]).unwrap(), 1);
        assert_eq!(persistence.replace_facts_snapshot(&[fact]).unwrap(), 1);
        let loaded = persistence.load_facts().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].source_text, "snapshot fixture");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_transaction_rollback() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_snapshot_rollback_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);
        let existing = FactRecord {
            vector_bytes: vec![1u8; BinaryHV::BYTES],
            source_text: "existing fixture".into(),
            confidence: 0.8,
            domain: None,
            cycle: 3,
            is_causal: false,
            relations: Vec::new(),
            graph_fact_id: None,
            initial_confidence: Some(0.9),
            last_accessed_cycle: Some(4),
            corroboration_count: 0,
            contradiction_count: 0,
        };
        persistence.replace_facts_snapshot(&[existing]).unwrap();

        let invalid = FactRecord {
            vector_bytes: vec![2u8; BinaryHV::BYTES],
            source_text: "range fixture".into(),
            confidence: 0.8,
            domain: None,
            cycle: u64::MAX,
            is_causal: false,
            relations: Vec::new(),
            graph_fact_id: None,
            initial_confidence: Some(0.9),
            last_accessed_cycle: Some(4),
            corroboration_count: 0,
            contradiction_count: 0,
        };
        assert!(persistence.replace_facts_snapshot(&[invalid]).is_err());

        let loaded = persistence.load_facts().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].source_text, "existing fixture");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_unconfigured() {
        let mut p = KnowledgePersistence::default();
        assert!(!p.is_configured());
        assert!(p.save_facts(&[]).is_err());
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
                vector_bytes: vec![0u8; 2048],
                source_text: "Test fact one".into(),
                confidence: 0.9,
                domain: Some("test".into()),
                cycle: 1,
                is_causal: false,
                relations: Vec::new(),
                graph_fact_id: None,
                initial_confidence: Some(0.95),
                last_accessed_cycle: Some(42),
                corroboration_count: 3,
                contradiction_count: 2,
            },
            FactRecord {
                vector_bytes: vec![1u8; 2048],
                source_text: "Test fact two".into(),
                confidence: 0.8,
                domain: None,
                cycle: 2,
                is_causal: true,
                relations: Vec::new(),
                graph_fact_id: None,
                initial_confidence: Some(0.85),
                last_accessed_cycle: Some(44),
                corroboration_count: 4,
                contradiction_count: 1,
            },
        ];

        let saved = p.save_facts(&facts).unwrap();
        assert_eq!(saved, 2);
        assert_eq!(p.total_saved(), 2);

        let loaded = p.load_facts().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].source_text, "Test fact two"); // DESC order
        assert_eq!(loaded[1].source_text, "Test fact one");
        assert_eq!(loaded[0].initial_confidence, Some(0.85));
        assert_eq!(loaded[0].last_accessed_cycle, Some(44));
        assert_eq!(loaded[0].corroboration_count, 4);
        assert_eq!(loaded[0].contradiction_count, 1);
        assert_eq!(loaded[1].initial_confidence, Some(0.95));
        assert_eq!(loaded[1].last_accessed_cycle, Some(42));
        assert_eq!(loaded[1].corroboration_count, 3);
        assert_eq!(loaded[1].contradiction_count, 2);

        // Cleanup
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
                is_negated: false,
                source_text: String::new(),
                cycle: 1,
            },
            CausalEdgeRecord {
                cause: "diplomacy".into(),
                effect: "war".into(),
                strength: 0.6,
                is_inhibitory: true,
                is_negated: false,
                source_text: "diplomacy prevented war".into(),
                cycle: 2,
            },
        ];

        let saved = p.save_causal_edges(&edges).unwrap();
        assert_eq!(saved, 2);

        let loaded = p.load_causal_edges().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].cause, "sanctions");
        assert_eq!(loaded[0].strength, 0.8);
        assert!(!loaded[0].is_inhibitory);
        assert_eq!(loaded[0].cycle, 1);
        assert_eq!(loaded[1].cause, "diplomacy");
        assert_eq!(loaded[1].strength, 0.6);
        assert!(loaded[1].is_inhibitory);
        assert_eq!(loaded[1].source_text, "diplomacy prevented war");
        assert_eq!(loaded[1].cycle, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_legacy_causal_edge_schema_migrates_without_losing_rows() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_legacy_causal_edge_migration_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");

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
            VALUES ('ceasefire', 'escalation', -0.7, 0, 9);",
        )
        .unwrap();
        drop(conn);

        let mut persistence = KnowledgePersistence::new(&db_path);
        let edges = persistence.load_causal_edges().unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].cause, "ceasefire");
        assert_eq!(edges[0].effect, "escalation");
        assert_eq!(edges[0].strength, 0.7);
        assert!(edges[0].is_inhibitory);
        assert!(!edges[0].is_negated);
        assert!(edges[0].source_text.is_empty());
        assert_eq!(edges[0].cycle, 9);

        // The migrated schema accepts distinct observations for the same pair.
        persistence
            .save_causal_edges(&[
                CausalEdgeRecord {
                    cause: "ceasefire".into(),
                    effect: "escalation".into(),
                    strength: 0.6,
                    is_inhibitory: false,
                    is_negated: false,
                    source_text: "later observation one".into(),
                    cycle: 10,
                },
                CausalEdgeRecord {
                    cause: "ceasefire".into(),
                    effect: "escalation".into(),
                    strength: 0.8,
                    is_inhibitory: true,
                    is_negated: true,
                    source_text: "later observation two".into(),
                    cycle: 11,
                },
            ])
            .unwrap();

        let after_legacy_upserts = persistence.load_causal_edges().unwrap();
        assert_eq!(after_legacy_upserts.len(), 1);
        assert_eq!(after_legacy_upserts[0].source_text, "later observation two");
        assert!(after_legacy_upserts[0].is_inhibitory);
        assert!(after_legacy_upserts[0].is_negated);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_upsert_causal_edges() {
        let dir = std::env::temp_dir().join(format!("symthaea_upsert_test_{}", std::process::id()));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);

        let edge = CausalEdgeRecord {
            cause: "a".into(),
            effect: "b".into(),
            strength: 0.5,
            is_inhibitory: false,
            is_negated: false,
            source_text: String::new(),
            cycle: 1,
        };

        p.save_causal_edges(&[edge.clone()]).unwrap();

        // Save again with updated strength
        let edge2 = CausalEdgeRecord {
            strength: 0.9,
            cycle: 2,
            ..edge
        };
        p.save_causal_edges(&[edge2]).unwrap();

        let loaded = p.load_causal_edges().unwrap();
        assert_eq!(loaded.len(), 1); // Upserted, not duplicated
        assert!((loaded[0].strength - 0.9).abs() < 0.01);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
