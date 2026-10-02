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
//!
//! Science: Ebbinghaus (1885) memory consolidation across sessions

use std::path::Path;
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

        if facts.iter().any(|fact| fact.memory_id.trim().is_empty()) {
            return Err("FactRecord memory_id must be non-empty".into());
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
        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Begin provenance transaction: {e}"))?;
        let mut count = 0;
        for relation in relations {
            let relation_for_validation = ProvenanceRelation::from(relation.clone());
            relation_for_validation
                .validate()
                .map_err(|e| format!("Invalid provenance relation: {e}"))?;
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
                    i64::try_from(edge.cycle).map_err(|_| rusqlite::Error::ToSqlConversionFailure(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidInput, "causal edge cycle exceeds SQLite INTEGER range"))))?,
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
        rusqlite::Connection::open(&self.db_path).map_err(|e| format!("SQLite open: {e}"))
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

#[cfg(test)]
mod tests {
    use super::*;

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
