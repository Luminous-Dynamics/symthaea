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
//! - knowledge_facts: stable fact identity + temporal/confidence metadata
//! - knowledge_fact_roles: role-tagged HDC sub-vectors for compositional retrieval
//! - knowledge_causal_edges: cause, effect, strength, cycle
//! - knowledge_ontology: name, vector_blob, usage_count, utility, cycle
//!
//! Science: Ebbinghaus (1885) memory consolidation across sessions

use std::collections::HashMap;
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

/// A serializable fact record for persistence.
///
/// `role_vectors` use stable numeric semantic-role tags owned by the graph
/// persistence boundary so the storage layer does not depend on feature-specific
/// enum serialization.
#[derive(Debug, Clone)]
pub struct FactRecord {
    /// Stable graph fact identifier.
    pub id: u64,
    /// BinaryHV encoded as raw bytes (2048 bytes for 16,384 bits)
    pub vector_bytes: Vec<u8>,
    /// Role-specific HDC vectors as (stable role tag, raw BinaryHV bytes).
    pub role_vectors: Vec<(u8, Vec<u8>)>,
    /// Source text of the fact
    pub source_text: String,
    /// Original encoding/extraction confidence retained by `FactEncoding`.
    pub encoding_confidence: f32,
    /// Current confidence score after decay/revision.
    pub confidence: f32,
    /// Initial graph confidence at insertion.
    pub initial_confidence: f32,
    /// Domain tag (optional)
    pub domain: Option<String>,
    /// Cycle when fact was inserted
    pub cycle: u64,
    /// Cycle when fact was last accessed/refreshed.
    pub last_accessed_cycle: u64,
    /// Number of independent corroboration updates recorded by the graph.
    pub corroboration_count: u32,
    /// Number of contradiction updates recorded by the graph.
    pub contradiction_count: u32,
    /// Whether the fact contains causal relations
    pub is_causal: bool,
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

    /// Save the complete fact snapshot to the database.
    ///
    /// This is a snapshot API, not an append log: rows not present in `facts`
    /// are removed, stable fact IDs are preserved, and role vectors are replaced
    /// atomically with their owning facts. This prevents repeated snapshots from
    /// multiplying semantically identical facts across restarts.
    pub fn save_facts(&mut self, facts: &[FactRecord]) -> Result<usize, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;

        let tx = conn
            .unchecked_transaction()
            .map_err(|e| format!("Fact snapshot transaction: {e}"))?;

        tx.execute("DELETE FROM knowledge_fact_roles", [])
            .map_err(|e| format!("Clear fact roles: {e}"))?;
        tx.execute("DELETE FROM knowledge_facts", [])
            .map_err(|e| format!("Clear fact snapshot: {e}"))?;

        {
            let mut fact_stmt = tx
                .prepare_cached(
                    "INSERT INTO knowledge_facts
                     (id, vector_blob, source_text, encoding_confidence, confidence,
                      initial_confidence, domain, cycle, last_accessed_cycle,
                      corroboration_count, contradiction_count, is_causal)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                )
                .map_err(|e| format!("Prepare fact snapshot: {e}"))?;

            let mut role_stmt = tx
                .prepare_cached(
                    "INSERT INTO knowledge_fact_roles (fact_id, role_tag, vector_blob)
                     VALUES (?1, ?2, ?3)",
                )
                .map_err(|e| format!("Prepare fact role snapshot: {e}"))?;

            for fact in facts {
                if fact.vector_bytes.len() != 2048 {
                    return Err(format!(
                        "Fact {} has invalid vector length {}; expected 2048",
                        fact.id,
                        fact.vector_bytes.len()
                    ));
                }

                fact_stmt
                    .execute(rusqlite::params![
                        fact.id as i64,
                        &fact.vector_bytes,
                        &fact.source_text,
                        fact.encoding_confidence,
                        fact.confidence,
                        fact.initial_confidence,
                        fact.domain.as_deref(),
                        fact.cycle as i64,
                        fact.last_accessed_cycle as i64,
                        fact.corroboration_count as i64,
                        fact.contradiction_count as i64,
                        fact.is_causal,
                    ])
                    .map_err(|e| format!("Insert fact {}: {e}", fact.id))?;

                for (role_tag, vector_bytes) in &fact.role_vectors {
                    if vector_bytes.len() != 2048 {
                        return Err(format!(
                            "Fact {} role {} has invalid vector length {}; expected 2048",
                            fact.id,
                            role_tag,
                            vector_bytes.len()
                        ));
                    }
                    role_stmt
                        .execute(rusqlite::params![
                            fact.id as i64,
                            *role_tag as i64,
                            vector_bytes,
                        ])
                        .map_err(|e| {
                            format!("Insert fact {} role {}: {e}", fact.id, role_tag)
                        })?;
                }
            }
        }

        tx.commit()
            .map_err(|e| format!("Commit fact snapshot: {e}"))?;

        self.total_saved += facts.len() as u64;
        Ok(facts.len())
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
                "SELECT id, vector_blob, source_text,
                        CASE WHEN encoding_confidence = 0.0 THEN confidence ELSE encoding_confidence END,
                        confidence,
                        CASE WHEN initial_confidence = 0.0 THEN confidence ELSE initial_confidence END,
                        domain, cycle,
                        CASE WHEN last_accessed_cycle = 0 THEN cycle ELSE last_accessed_cycle END,
                        corroboration_count, contradiction_count, is_causal
                 FROM knowledge_facts ORDER BY cycle DESC, id ASC",
            )
            .map_err(|e| e.to_string())?;

        let mut facts: Vec<FactRecord> = stmt
            .query_map([], |row| {
                Ok(FactRecord {
                    id: row.get::<_, i64>(0)? as u64,
                    vector_bytes: row.get(1)?,
                    role_vectors: Vec::new(),
                    source_text: row.get(2)?,
                    encoding_confidence: row.get(3)?,
                    confidence: row.get(4)?,
                    initial_confidence: row.get(5)?,
                    domain: row.get(6)?,
                    cycle: row.get::<_, i64>(7)? as u64,
                    last_accessed_cycle: row.get::<_, i64>(8)? as u64,
                    corroboration_count: row.get::<_, i64>(9)? as u32,
                    contradiction_count: row.get::<_, i64>(10)? as u32,
                    is_causal: row.get(11)?,
                })
            })
            .map_err(|e| e.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| e.to_string())?;

        let mut roles_by_fact: HashMap<u64, Vec<(u8, Vec<u8>)>> = HashMap::new();
        let mut role_stmt = conn
            .prepare(
                "SELECT fact_id, role_tag, vector_blob
                 FROM knowledge_fact_roles ORDER BY fact_id ASC, role_tag ASC",
            )
            .map_err(|e| e.to_string())?;
        let role_rows = role_stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)? as u64,
                    row.get::<_, i64>(1)? as u8,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })
            .map_err(|e| e.to_string())?;

        for row in role_rows {
            let (fact_id, role_tag, vector_bytes) = row.map_err(|e| e.to_string())?;
            if vector_bytes.len() == 2048 {
                roles_by_fact
                    .entry(fact_id)
                    .or_default()
                    .push((role_tag, vector_bytes));
            }
        }

        for fact in &mut facts {
            fact.role_vectors = roles_by_fact.remove(&fact.id).unwrap_or_default();
        }

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
            .filter_map(|r| r.ok())
            .collect();

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
                encoding_confidence REAL NOT NULL DEFAULT 0.0,
                confidence REAL NOT NULL,
                initial_confidence REAL NOT NULL DEFAULT 0.0,
                domain TEXT,
                cycle INTEGER NOT NULL,
                last_accessed_cycle INTEGER NOT NULL DEFAULT 0,
                corroboration_count INTEGER NOT NULL DEFAULT 0,
                contradiction_count INTEGER NOT NULL DEFAULT 0,
                is_causal INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS knowledge_fact_roles (
                fact_id INTEGER NOT NULL,
                role_tag INTEGER NOT NULL,
                vector_blob BLOB NOT NULL,
                PRIMARY KEY (fact_id, role_tag),
                FOREIGN KEY (fact_id) REFERENCES knowledge_facts(id) ON DELETE CASCADE
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
            CREATE INDEX IF NOT EXISTS idx_facts_cycle ON knowledge_facts(cycle);
            CREATE INDEX IF NOT EXISTS idx_fact_roles_fact_id ON knowledge_fact_roles(fact_id);",
        )
        .map_err(|e| format!("Schema init: {e}"))?;

        // Forward-compatible migration for databases created by persistence v1.
        // SQLite's `CREATE TABLE IF NOT EXISTS` does not add columns to an
        // existing table, so add the v2 metadata columns explicitly when absent.
        Self::ensure_column(
            conn,
            "knowledge_facts",
            "encoding_confidence",
            "REAL NOT NULL DEFAULT 0.0",
        )?;
        Self::ensure_column(
            conn,
            "knowledge_facts",
            "initial_confidence",
            "REAL NOT NULL DEFAULT 0.0",
        )?;
        Self::ensure_column(
            conn,
            "knowledge_facts",
            "last_accessed_cycle",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        Self::ensure_column(
            conn,
            "knowledge_facts",
            "corroboration_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;
        Self::ensure_column(
            conn,
            "knowledge_facts",
            "contradiction_count",
            "INTEGER NOT NULL DEFAULT 0",
        )?;

        self.initialized = true;
        Ok(())
    }

    fn ensure_column(
        conn: &rusqlite::Connection,
        table: &str,
        column: &str,
        definition: &str,
    ) -> Result<(), String> {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .map_err(|e| format!("Inspect {table} schema: {e}"))?;
        let names = stmt
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|e| format!("Read {table} schema: {e}"))?;

        for name in names {
            if name.map_err(|e| e.to_string())? == column {
                return Ok(());
            }
        }

        conn.execute(
            &format!("ALTER TABLE {table} ADD COLUMN {column} {definition}"),
            [],
        )
        .map_err(|e| format!("Add {table}.{column}: {e}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fact_record(id: u64, byte: u8, text: &str, confidence: f32, cycle: u64) -> FactRecord {
        FactRecord {
            id,
            vector_bytes: vec![byte; 2048],
            role_vectors: Vec::new(),
            source_text: text.into(),
            encoding_confidence: confidence,
            confidence,
            initial_confidence: confidence,
            domain: None,
            cycle,
            last_accessed_cycle: cycle,
            corroboration_count: 0,
            contradiction_count: 0,
            is_causal: false,
        }
    }

    #[test]
    fn test_unconfigured() {
        let mut p = KnowledgePersistence::default();
        assert!(!p.is_configured());
        assert!(p.save_facts(&[]).is_err());
    }

    #[test]
    fn test_save_and_load_facts_preserves_v2_metadata() {
        let dir =
            std::env::temp_dir().join(format!("symthaea_knowledge_test_{}", std::process::id()));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        assert!(p.is_configured());

        let facts = vec![
            FactRecord {
                id: 41,
                vector_bytes: vec![0u8; 2048],
                role_vectors: vec![(0, vec![2u8; 2048])],
                source_text: "Test fact one".into(),
                encoding_confidence: 0.95,
                confidence: 0.72,
                initial_confidence: 0.9,
                domain: Some("test".into()),
                cycle: 1,
                last_accessed_cycle: 7,
                corroboration_count: 3,
                contradiction_count: 1,
                is_causal: false,
            },
            FactRecord {
                id: 99,
                vector_bytes: vec![1u8; 2048],
                role_vectors: vec![(10, vec![3u8; 2048])],
                source_text: "Test fact two".into(),
                encoding_confidence: 0.88,
                confidence: 0.61,
                initial_confidence: 0.8,
                domain: None,
                cycle: 2,
                last_accessed_cycle: 9,
                corroboration_count: 2,
                contradiction_count: 4,
                is_causal: true,
            },
        ];

        let saved = p.save_facts(&facts).unwrap();
        assert_eq!(saved, 2);
        assert_eq!(p.total_saved(), 2);

        let loaded = p.load_facts().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, 99);
        assert_eq!(loaded[0].source_text, "Test fact two"); // DESC cycle order
        assert_eq!(loaded[0].role_vectors.len(), 1);
        assert_eq!(loaded[0].role_vectors[0].0, 10);
        assert!((loaded[0].encoding_confidence - 0.88).abs() < 0.001);
        assert!((loaded[0].initial_confidence - 0.8).abs() < 0.001);
        assert_eq!(loaded[0].last_accessed_cycle, 9);
        assert_eq!(loaded[0].corroboration_count, 2);
        assert_eq!(loaded[0].contradiction_count, 4);
        assert_eq!(loaded[1].id, 41);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_fact_save_is_snapshot_not_append_log() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_fact_snapshot_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);

        let mut p = KnowledgePersistence::new(&db_path);
        p.save_facts(&[
            fact_record(1, 1, "old one", 0.8, 1),
            fact_record(2, 2, "old two", 0.8, 2),
        ])
        .unwrap();

        p.save_facts(&[fact_record(2, 3, "updated two", 0.9, 3)])
            .unwrap();

        let loaded = p.load_facts().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].id, 2);
        assert_eq!(loaded[0].source_text, "updated two");
        assert_eq!(loaded[0].vector_bytes[0], 3);

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
