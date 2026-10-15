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
            conn.execute(
                "INSERT INTO knowledge_facts
                 (vector_blob, source_text, confidence, domain, cycle, is_causal,
                  initial_confidence, last_accessed_cycle, corroboration_count, contradiction_count)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
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
                ],
            )
            .map_err(|e| e.to_string())?;
            count += 1;
        }

        conn.execute_batch("COMMIT").map_err(|e| e.to_string())?;

        self.total_saved += count as u64;
        Ok(count)
    }

    /// Atomically replace the persisted fact set with a complete graph snapshot.
    ///
    /// Unlike `save_facts`, this operation is idempotent for repeated snapshots:
    /// it does not append another copy of each existing graph fact. The delete
    /// and all inserts share one transaction, so a failed snapshot preserves the
    /// previous committed fact set.
    pub fn replace_facts_snapshot(&mut self, facts: &[FactRecord]) -> Result<usize, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let mut conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let tx = conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(|e| format!("Begin fact snapshot transaction: {e}"))?;

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

            tx.execute(
                "INSERT INTO knowledge_facts
                 (vector_blob, source_text, confidence, domain, cycle, is_causal,
                  initial_confidence, last_accessed_cycle, corroboration_count, contradiction_count)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
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
                ],
            )
            .map_err(|e| format!("Insert fact snapshot row: {e}"))?;
            count += 1;
        }

        tx.commit()
            .map_err(|e| format!("Commit fact snapshot: {e}"))?;
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

        let mut stmt = conn
            .prepare(
                "SELECT vector_blob, source_text, confidence, domain, cycle, is_causal,
                        initial_confidence, last_accessed_cycle, corroboration_count, contradiction_count
                 FROM knowledge_facts ORDER BY cycle DESC",
            )
            .map_err(|e| e.to_string())?;

        let facts: Vec<FactRecord> = stmt
            .query_map([], |row| {
                let cycle_raw = row.get::<_, i64>(4)?;
                let cycle = u64::try_from(cycle_raw).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        4,
                        rusqlite::types::Type::Integer,
                        Box::new(e),
                    )
                })?;
                let last_accessed_raw: Option<i64> = row.get(7)?;
                let last_accessed_cycle = last_accessed_raw
                    .map(|value| {
                        u64::try_from(value).map_err(|e| {
                            rusqlite::Error::FromSqlConversionFailure(
                                7,
                                rusqlite::types::Type::Integer,
                                Box::new(e),
                            )
                        })
                    })
                    .transpose()?;
                let corroboration_count = u32::try_from(row.get::<_, i64>(8)?).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        8,
                        rusqlite::types::Type::Integer,
                        Box::new(e),
                    )
                })?;
                let contradiction_count = u32::try_from(row.get::<_, i64>(9)?).map_err(|e| {
                    rusqlite::Error::FromSqlConversionFailure(
                        9,
                        rusqlite::types::Type::Integer,
                        Box::new(e),
                    )
                })?;
                Ok(FactRecord {
                    vector_bytes: row.get(0)?,
                    source_text: row.get(1)?,
                    confidence: row.get(2)?,
                    domain: row.get(3)?,
                    cycle,
                    is_causal: row.get(5)?,
                    initial_confidence: row.get(6)?,
                    last_accessed_cycle,
                    corroboration_count,
                    contradiction_count,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("Collect: {e}"))?;

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
            conn.execute(
                "INSERT OR REPLACE INTO knowledge_causal_edges (cause, effect, strength, is_inhibitory, cycle)
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    edge.cause,
                    edge.effect,
                    edge.strength,
                    edge.is_inhibitory,
                    edge.cycle as i64,
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
                "SELECT cause, effect, strength, is_inhibitory, cycle FROM knowledge_causal_edges",
            )
            .map_err(|e| e.to_string())?;

        let edges: Vec<CausalEdgeRecord> = stmt
            .query_map([], |row| {
                Ok(CausalEdgeRecord {
                    cause: row.get(0)?,
                    effect: row.get(1)?,
                    strength: row.get(2)?,
                    is_inhibitory: row.get(3)?,
                    cycle: row.get::<_, i64>(4)? as u64,
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
                Ok(OntologyRecord {
                    name: row.get(0)?,
                    vector_bytes: row.get(1)?,
                    usage_count: row.get::<_, i64>(2)? as u64,
                    utility: row.get(3)?,
                    created_at_cycle: row.get::<_, i64>(4)? as u64,
                    last_used_cycle: row.get::<_, i64>(5)? as u64,
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
                contradiction_count INTEGER NOT NULL DEFAULT 0
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

        // Additive migration for databases created before temporal graph state
        // was persisted. NULL distinguishes unknown legacy values from known zero.
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
        ] {
            if !columns.iter().any(|column| column == name) {
                conn.execute_batch(&format!(
                    "ALTER TABLE knowledge_facts ADD COLUMN {name} {declaration}"
                ))
                .map_err(|e| format!("Migrate knowledge_facts.{name}: {e}"))?;
            }
        }

        self.initialized = true;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;


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
                initial_confidence: Some(0.9),
                last_accessed_cycle: Some(3),
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
    fn test_load_facts_rejects_negative_temporal_cycles() {
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
                vector_bytes: vec![0u8; 2048],
                source_text: "fixture fact".into(),
                confidence: 0.9,
                domain: None,
                cycle: 3,
                is_causal: false,
                initial_confidence: Some(0.9),
                last_accessed_cycle: Some(3),
                corroboration_count: 0,
                contradiction_count: 0,
            }])
            .unwrap();

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute("UPDATE knowledge_facts SET cycle = -1", [])
            .unwrap();
        drop(conn);

        let insertion_cycle_err = persistence.load_facts().unwrap_err();
        assert!(
            insertion_cycle_err.contains("Collect:"),
            "unexpected insertion-cycle error context: {insertion_cycle_err}"
        );

        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute(
            "UPDATE knowledge_facts SET cycle = 3, last_accessed_cycle = -1",
            [],
        )
        .unwrap();
        drop(conn);

        let last_access_err = persistence.load_facts().unwrap_err();
        assert!(
            last_access_err.contains("Collect:"),
            "unexpected last-access-cycle error context: {last_access_err}"
        );
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

        // Model the pre-migration schema exactly, then seed one pre-existing row.
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
            rusqlite::params![vec![0u8; 2048], "legacy fact", 0.4_f32, "legacy", 7_i64, false],
        )
        .unwrap();
        drop(conn);

        let mut persistence = KnowledgePersistence::new(&db_path);
        let records = persistence.load_facts().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].source_text, "legacy fact");
        assert_eq!(records[0].initial_confidence, None);
        assert_eq!(records[0].last_accessed_cycle, None);
        assert_eq!(records[0].corroboration_count, 0);
        assert_eq!(records[0].contradiction_count, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    fn snapshot_test_record(source_text: &str, cycle: u64) -> FactRecord {
        FactRecord {
            vector_bytes: vec![cycle as u8; 2048],
            source_text: source_text.to_string(),
            confidence: 0.8,
            domain: Some("test".into()),
            cycle,
            is_causal: false,
            initial_confidence: Some(0.9),
            last_accessed_cycle: Some(cycle + 1),
            corroboration_count: 2,
            contradiction_count: 1,
        }
    }

    #[test]
    fn test_replace_facts_snapshot_is_idempotent_and_replaces_rows() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_replace_fact_snapshot_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);

        persistence
            .save_facts(&[snapshot_test_record("old fact", 1)])
            .unwrap();
        let snapshot = [snapshot_test_record("current fact", 7)];
        assert_eq!(persistence.replace_facts_snapshot(&snapshot).unwrap(), 1);
        assert_eq!(persistence.replace_facts_snapshot(&snapshot).unwrap(), 1);

        let loaded = persistence.load_facts().unwrap();
        assert_eq!(loaded.len(), 1, "repeated snapshots must not duplicate facts");
        assert_eq!(loaded[0].source_text, "current fact");
        assert_eq!(loaded[0].cycle, 7);

        assert_eq!(persistence.replace_facts_snapshot(&[]).unwrap(), 0);
        assert!(
            persistence.load_facts().unwrap().is_empty(),
            "an empty graph snapshot must clear the persisted fact set"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_failed_fact_snapshot_preserves_previous_committed_rows() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_failed_fact_snapshot_test_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let db_path = dir.join("knowledge.db");
        let mut persistence = KnowledgePersistence::new(&db_path);

        persistence
            .replace_facts_snapshot(&[snapshot_test_record("committed fact", 3)])
            .unwrap();

        let invalid_snapshot = [
            snapshot_test_record("would be partial", 4),
            snapshot_test_record("invalid cycle", u64::MAX),
        ];
        let err = persistence
            .replace_facts_snapshot(&invalid_snapshot)
            .unwrap_err();
        assert!(err.contains("SQLite INTEGER range"), "unexpected error: {err}");

        let loaded = persistence.load_facts().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(
            loaded[0].source_text, "committed fact",
            "a failed snapshot must not erase or partially replace committed data"
        );
        assert_eq!(loaded[0].cycle, 3);

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

        let loaded = p.load_causal_edges().unwrap();
        assert_eq!(loaded.len(), 2);

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
