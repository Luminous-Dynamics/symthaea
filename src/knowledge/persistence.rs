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

use super::claim_provenance::{
    ClaimContext, EvidenceLink, EvidenceLinkStatus, EvidenceRelation,
};
use std::collections::{HashMap, HashSet};
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
    /// Cycle when the fact was inserted
    pub cycle: u64,
    /// Whether the fact contains causal relations
    pub is_causal: bool,
}

/// Graph snapshot record that retains stable graph identity and optional claim provenance.
#[derive(Debug, Clone)]
pub struct GraphFactRecord {
    /// Stable graph-level fact ID; absent only for pre-migration SQLite rows.
    pub fact_id: Option<u64>,
    /// Legacy persisted fact payload.
    pub fact: FactRecord,
    /// Optional source-witness-grounded claim envelope.
    pub claim_context: Option<ClaimContext>,
}

fn normalize_graph_fact_ids(facts: &[GraphFactRecord]) -> Result<Vec<GraphFactRecord>, String> {
    let mut used_ids = HashSet::new();
    for record in facts {
        if let Some(id) = record.fact_id {
            if !used_ids.insert(id) {
                return Err(format!("duplicate graph snapshot fact ID {id}"));
            }
        }
    }

    let mut next_id = 1u64;
    let mut normalized = Vec::with_capacity(facts.len());
    for record in facts {
        let fact_id = match record.fact_id {
            Some(id) => Some(id),
            None => {
                while used_ids.contains(&next_id) {
                    next_id = next_id
                        .checked_add(1)
                        .ok_or_else(|| "no stable fact IDs available for graph snapshot".to_string())?;
                }
                let assigned = next_id;
                used_ids.insert(assigned);
                next_id = next_id
                    .checked_add(1)
                    .ok_or_else(|| "no stable fact IDs available for graph snapshot".to_string())?;
                Some(assigned)
            }
        };
        normalized.push(GraphFactRecord {
            fact_id,
            fact: record.fact.clone(),
            claim_context: record.claim_context.clone(),
        });
    }
    Ok(normalized)
}

fn validate_graph_snapshot(
    facts: &[GraphFactRecord],
    links: &[EvidenceLink],
) -> Result<(), String> {
    let mut fact_ids = HashSet::new();
    let mut contexts: HashMap<u64, &ClaimContext> = HashMap::new();
    for record in facts {
        let fact_id = record
            .fact_id
            .ok_or_else(|| "atomic graph snapshots require stable fact IDs".to_string())?;
        if !fact_ids.insert(fact_id) {
            return Err(format!("duplicate graph snapshot fact ID {fact_id}"));
        }
        if let Some(context) = record.claim_context.as_ref() {
            context.validate().map_err(|error| {
                format!("graph fact {fact_id} has invalid claim context: {error}")
            })?;
            contexts.insert(fact_id, context);
        }
    }

    let mut link_keys = HashSet::new();
    for link in links {
        if link.from_fact_id == link.to_fact_id {
            return Err("evidence links must connect two distinct facts".into());
        }
        for endpoint in [link.from_fact_id, link.to_fact_id] {
            if !fact_ids.contains(&endpoint) {
                return Err(format!(
                    "evidence link endpoint {endpoint} is absent from graph snapshot"
                ));
            }
        }
        let mut evidence_ids = link.evidence_ids.clone();
        evidence_ids.sort();
        if !link_keys.insert((
            link.from_fact_id,
            link.to_fact_id,
            link.relation,
            link.status,
            evidence_ids,
        )) {
            return Err("graph snapshot repeats an identical evidence link".into());
        }

        if link.status == EvidenceLinkStatus::Rejected
            && link.rationale.as_deref().is_none_or(|value| value.trim().is_empty())
        {
            return Err("rejected evidence links require a rationale".into());
        }
        if link.status != EvidenceLinkStatus::Adjudicated {
            continue;
        }
        let evidence: HashSet<&str> = link.evidence_ids.iter().map(String::as_str).collect();
        if evidence.is_empty()
            || link.evidence_ids.iter().any(|value| value.trim().is_empty())
            || evidence.len() != link.evidence_ids.len()
            || link.rationale.as_deref().is_none_or(|value| value.trim().is_empty())
        {
            return Err("adjudicated evidence links require a rationale and unique evidence IDs".into());
        }
        let from = contexts.get(&link.from_fact_id).ok_or_else(|| {
            format!("adjudicated link source {} has no claim context", link.from_fact_id)
        })?;
        let to = contexts.get(&link.to_fact_id).ok_or_else(|| {
            format!("adjudicated link target {} has no claim context", link.to_fact_id)
        })?;
        if !from.evidence_ids.iter().any(|id| evidence.contains(id.as_str()))
            || !to.evidence_ids.iter().any(|id| evidence.contains(id.as_str()))
        {
            return Err("adjudicated link evidence must reference both claim contexts".into());
        }
        let context_evidence: HashSet<&str> = from
            .evidence_ids
            .iter()
            .chain(to.evidence_ids.iter())
            .map(String::as_str)
            .collect();
        if !evidence.is_subset(&context_evidence) {
            return Err(
                "adjudicated link contains evidence IDs not declared by either endpoint context"
                    .into(),
            );
        }
        match link.relation {
            EvidenceRelation::SemanticNeighbor | EvidenceRelation::SameWitnessVariant => {
                return Err("candidate-only relations cannot be persisted as adjudicated".into());
            }
            EvidenceRelation::Corroborates => {
                if !from.same_claim(to) || !from.admissibly_independent_from(to) {
                    return Err("adjudicated corroboration violates claim identity or source independence".into());
                }
            }
            EvidenceRelation::Contradicts => {
                if !from.same_conflict_scope(to) || from.same_claim(to) {
                    return Err("adjudicated contradiction violates scoped-claim invariants".into());
                }
            }
            EvidenceRelation::Precedes => {
                if !matches!((from.event_time, to.event_time), (Some(left), Some(right)) if left < right) {
                    return Err("adjudicated precedes relation requires explicit increasing event times".into());
                }
            }
            EvidenceRelation::Causes => {
                if !matches!((from.event_time, to.event_time), (Some(left), Some(right)) if left < right)
                    || link.causal_argument.as_deref().is_none_or(|value| value.trim().is_empty())
                {
                    return Err("adjudicated causes relation requires time order and a causal argument".into());
                }
            }
            EvidenceRelation::HistoricallyInfluences => {
                if !matches!((from.event_time, to.event_time), (Some(left), Some(right)) if left < right)
                    || link.transmission_chain.as_deref().is_none_or(|value| value.trim().is_empty())
                {
                    return Err("adjudicated historical influence requires time order and a transmission chain".into());
                }
            }
            EvidenceRelation::Resembles => {}
        }
    }
    Ok(())
}

fn read_graph_fact_records(
    conn: &rusqlite::Connection,
) -> Result<Vec<GraphFactRecord>, String> {
    let mut statement = conn
        .prepare(
            "SELECT fact_id, vector_blob, source_text, confidence, domain, cycle, is_causal, claim_context_json
             FROM knowledge_facts ORDER BY cycle DESC, id DESC",
        )
        .map_err(|error| error.to_string())?;
    let rows = statement
        .query_map([], |row| {
            let claim_context = row
                .get::<_, Option<String>>(7)?
                .map(|raw| {
                    serde_json::from_str::<ClaimContext>(&raw).map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            7,
                            rusqlite::types::Type::Text,
                            Box::new(error),
                        )
                    })
                })
                .transpose()?;
            Ok(GraphFactRecord {
                fact_id: row.get::<_, Option<i64>>(0)?.map(|id| id as u64),
                fact: FactRecord {
                    vector_bytes: row.get(1)?,
                    source_text: row.get(2)?,
                    confidence: row.get(3)?,
                    domain: row.get(4)?,
                    cycle: row.get::<_, i64>(5)? as u64,
                    is_causal: row.get(6)?,
                },
                claim_context,
            })
        })
        .map_err(|error| error.to_string())?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("Graph fact record decoding failed: {error}"))
}

fn read_evidence_link_records(
    conn: &rusqlite::Connection,
) -> Result<Vec<EvidenceLink>, String> {
    let mut statement = conn
        .prepare("SELECT link_json FROM knowledge_evidence_links ORDER BY id")
        .map_err(|error| error.to_string())?;
    let encoded = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|error| error.to_string())?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("Evidence link row decoding failed: {error}"))?;

    encoded
        .iter()
        .enumerate()
        .map(|(index, raw)| {
            serde_json::from_str::<EvidenceLink>(raw).map_err(|error| {
                format!("Evidence link JSON decoding failed at row {}: {error}", index + 1)
            })
        })
        .collect()
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

    /// Save a batch of legacy fact records to the database.
    ///
    /// This retains the original append-style API for existing callers.
    pub fn save_facts(&mut self, facts: &[FactRecord]) -> Result<usize, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        conn.execute_batch("BEGIN TRANSACTION")
            .map_err(|error| error.to_string())?;

        for fact in facts {
            conn.execute(
                "INSERT INTO knowledge_facts (vector_blob, source_text, confidence, domain, cycle, is_causal)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    fact.vector_bytes,
                    fact.source_text,
                    fact.confidence,
                    fact.domain,
                    fact.cycle as i64,
                    fact.is_causal,
                ],
            )
            .map_err(|error| error.to_string())?;
        }

        conn.execute_batch("COMMIT")
            .map_err(|error| error.to_string())?;
        self.total_saved += facts.len() as u64;
        Ok(facts.len())
    }

    /// Load legacy fact records without requiring the newer graph-provenance fields.
    pub fn load_facts(&mut self) -> Result<Vec<FactRecord>, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }

        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let mut statement = conn
            .prepare(
                "SELECT vector_blob, source_text, confidence, domain, cycle, is_causal
                 FROM knowledge_facts ORDER BY cycle DESC, id DESC",
            )
            .map_err(|error| error.to_string())?;
        let rows = statement
            .query_map([], |row| {
                Ok(FactRecord {
                    vector_bytes: row.get(0)?,
                    source_text: row.get(1)?,
                    confidence: row.get(2)?,
                    domain: row.get(3)?,
                    cycle: row.get::<_, i64>(4)? as u64,
                    is_causal: row.get(5)?,
                })
            })
            .map_err(|error| error.to_string())?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("Fact record decoding failed: {error}"))?;

        self.total_loaded += rows.len() as u64;
        Ok(rows)
    }

    /// Replace the graph-fact snapshot when there are no links to persist.
    ///
    /// Any prior typed links are cleared atomically so they cannot survive against
    /// a replaced fact set. Missing legacy IDs are assigned deterministically.
    pub fn save_graph_facts(&mut self, facts: &[GraphFactRecord]) -> Result<usize, String> {
        let normalized = normalize_graph_fact_ids(facts)?;
        self.save_graph_snapshot(&normalized, &[])
            .map(|(saved, _)| saved)
    }

    /// Atomically replace the graph-fact and evidence-link snapshots.
    ///
    /// This prevents a crash between separate fact/link writes from leaving links
    /// pointed at a newer set of facts with recycled identifiers.
    pub fn save_graph_snapshot(
        &mut self,
        facts: &[GraphFactRecord],
        links: &[EvidenceLink],
    ) -> Result<(usize, usize), String> {
        validate_graph_snapshot(facts, links)?;
        if !self.is_configured() {
            return Err("No database path configured".into());
        }
        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let tx = conn.unchecked_transaction().map_err(|error| error.to_string())?;

        tx.execute("DELETE FROM knowledge_facts", [])
            .map_err(|error| error.to_string())?;
        for item in facts {
            let claim_context_json = item
                .claim_context
                .as_ref()
                .map(serde_json::to_string)
                .transpose()
                .map_err(|error| error.to_string())?;
            tx.execute(
                "INSERT INTO knowledge_facts
                 (fact_id, vector_blob, source_text, confidence, domain, cycle, is_causal, claim_context_json)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![
                    item.fact_id.map(|id| id as i64),
                    item.fact.vector_bytes,
                    item.fact.source_text,
                    item.fact.confidence,
                    item.fact.domain,
                    item.fact.cycle as i64,
                    item.fact.is_causal,
                    claim_context_json,
                ],
            )
            .map_err(|error| error.to_string())?;
        }

        tx.execute("DELETE FROM knowledge_evidence_links", [])
            .map_err(|error| error.to_string())?;
        for link in links {
            let encoded = serde_json::to_string(link).map_err(|error| error.to_string())?;
            tx.execute(
                "INSERT INTO knowledge_evidence_links (link_json) VALUES (?1)",
                rusqlite::params![encoded],
            )
            .map_err(|error| error.to_string())?;
        }

        tx.commit().map_err(|error| error.to_string())?;
        self.total_saved += facts.len() as u64 + links.len() as u64;
        Ok((facts.len(), links.len()))
    }

    /// Load the graph snapshot, preserving IDs used by typed evidence-link endpoints.
    pub fn load_graph_facts(&mut self) -> Result<Vec<GraphFactRecord>, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }
        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let rows = read_graph_fact_records(&conn)?;
        self.total_loaded += rows.len() as u64;
        Ok(rows)
    }

    /// Load a fact/link snapshot from one SQLite read transaction and validate
    /// all endpoints, claim contexts, and relation-specific evidence invariants.
    ///
    /// Legacy rows without stable graph IDs are assigned deterministic in-memory
    /// IDs only when no persisted evidence links depend on those missing IDs.
    pub fn load_graph_snapshot(
        &mut self,
    ) -> Result<(Vec<GraphFactRecord>, Vec<EvidenceLink>), String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }
        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let tx = conn.unchecked_transaction().map_err(|error| error.to_string())?;

        let raw_facts = read_graph_fact_records(&tx)?;
        let links = read_evidence_link_records(&tx)?;
        if !links.is_empty() && raw_facts.iter().any(|record| record.fact_id.is_none()) {
            return Err(
                "cannot validate persisted evidence links while graph facts lack stable IDs".into(),
            );
        }
        let facts = normalize_graph_fact_ids(&raw_facts)?;
        validate_graph_snapshot(&facts, &links)?;
        tx.commit().map_err(|error| error.to_string())?;

        self.total_loaded += facts.len() as u64 + links.len() as u64;
        Ok((facts, links))
    }

    /// Compatibility API that returns links only after validating the entire
    /// persisted graph snapshot that owns their endpoints.
    pub fn load_evidence_links(&mut self) -> Result<Vec<EvidenceLink>, String> {
        self.load_graph_snapshot().map(|(_, links)| links)
    }

    /// Replace typed links only when the currently stored fact snapshot gives
    /// every endpoint a stable identity and the entire resulting snapshot passes
    /// validation. Facts and links are checked and written within one transaction.
    pub fn save_evidence_links(&mut self, links: &[EvidenceLink]) -> Result<usize, String> {
        if !self.is_configured() {
            return Err("No database path configured".into());
        }
        let conn = self.open_connection()?;
        self.ensure_schema(&conn)?;
        let tx = conn.unchecked_transaction().map_err(|error| error.to_string())?;

        let facts = read_graph_fact_records(&tx)?;
        if facts.iter().any(|record| record.fact_id.is_none()) {
            return Err(
                "cannot save evidence links while graph facts lack stable IDs; save a graph snapshot first".into(),
            );
        }
        validate_graph_snapshot(&facts, links)?;
        tx.execute("DELETE FROM knowledge_evidence_links", [])
            .map_err(|error| error.to_string())?;
        for link in links {
            let encoded = serde_json::to_string(link).map_err(|error| error.to_string())?;
            tx.execute(
                "INSERT INTO knowledge_evidence_links (link_json) VALUES (?1)",
                rusqlite::params![encoded],
            )
            .map_err(|error| error.to_string())?;
        }
        tx.commit().map_err(|error| error.to_string())?;
        self.total_saved += links.len() as u64;
        Ok(links.len())
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
                fact_id INTEGER,
                vector_blob BLOB NOT NULL,
                source_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                domain TEXT,
                cycle INTEGER NOT NULL,
                is_causal INTEGER NOT NULL DEFAULT 0,
                claim_context_json TEXT
            );
            CREATE TABLE IF NOT EXISTS knowledge_evidence_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                link_json TEXT NOT NULL
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

        // CREATE TABLE IF NOT EXISTS does not migrate databases created by older
        // versions. Add a nullable provenance column without rewriting old facts.
        let columns: Vec<String> = {
            let mut statement = conn
                .prepare("PRAGMA table_info(knowledge_facts)")
                .map_err(|error| format!("Schema inspection: {error}"))?;
            let rows = statement
                .query_map([], |row| row.get::<_, String>(1))
                .map_err(|error| format!("Schema inspection: {error}"))?;
            let collected = rows
                .collect::<Result<Vec<_>, _>>()
                .map_err(|error| format!("Schema inspection: {error}"))?;
            collected
        };
        if !columns.iter().any(|column| column == "fact_id") {
            conn.execute("ALTER TABLE knowledge_facts ADD COLUMN fact_id INTEGER", [])
                .map_err(|error| format!("Schema migration: {error}"))?;
        }
        if !columns.iter().any(|column| column == "claim_context_json") {
            conn.execute(
                "ALTER TABLE knowledge_facts ADD COLUMN claim_context_json TEXT",
                [],
            )
            .map_err(|error| format!("Schema migration: {error}"))?;
        }

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
            },
            FactRecord {
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
        assert_eq!(loaded[1].source_text, "Test fact one");

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_load_preserves_stable_fact_ids_and_claim_context() {
        let dir = std::env::temp_dir().join(format!("symthaea_claim_context_test_{}", std::process::id()));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let context = ClaimContext {
            subject: "Osiris".into(),
            predicate: "associated_with".into(),
            object: "renewal".into(),
            witness_id: "witness-a".into(),
            source_family_id: "family-a".into(),
            context_id: "Dendera:late-period".into(),
            temporal_scope: Some("late_period".into()),
            event_time: Some(-300),
            claim_kind: super::super::claim_provenance::ClaimKind::SourceAttested,
            polarity: super::super::claim_provenance::ClaimPolarity::Positive,
            evidence_ids: vec!["evidence-a".into()],
        };
        let record = GraphFactRecord {
            fact_id: Some(42),
            fact: FactRecord {
                vector_bytes: vec![7u8; 2048],
                source_text: "Osiris is associated with renewal".into(),
                confidence: 0.7,
                domain: Some("mythology".into()),
                cycle: 9,
                is_causal: false,
            },
            claim_context: Some(context.clone()),
        };
        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence.save_graph_facts(&[record]).unwrap();
        let loaded = persistence.load_graph_facts().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].fact_id, Some(42));
        assert_eq!(loaded[0].claim_context, Some(context));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_fact_schema_migrates_additive_provenance_columns() {
        let dir = std::env::temp_dir().join(format!("symthaea_legacy_schema_test_{}", std::process::id()));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        {
            let connection = rusqlite::Connection::open(&db_path).unwrap();
            connection.execute_batch(
                "CREATE TABLE knowledge_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_blob BLOB NOT NULL,
                    source_text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    domain TEXT,
                    cycle INTEGER NOT NULL,
                    is_causal INTEGER NOT NULL DEFAULT 0
                );
                INSERT INTO knowledge_facts (vector_blob, source_text, confidence, domain, cycle, is_causal)
                VALUES (zeroblob(2048), 'legacy fact', 0.6, NULL, 1, 0);"
            ).unwrap();
        }
        let mut persistence = KnowledgePersistence::new(&db_path);
        let loaded = persistence.load_graph_facts().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].fact.source_text, "legacy fact");
        assert_eq!(loaded[0].fact_id, None);
        assert_eq!(loaded[0].claim_context, None);
        persistence.save_graph_facts(&loaded).unwrap();
        let loaded_after_snapshot = persistence.load_graph_facts().unwrap();
        assert_eq!(loaded_after_snapshot.len(), 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn graph_snapshot_rejects_dangling_evidence_link_endpoints() {
        let fact = GraphFactRecord {
            fact_id: Some(1),
            fact: FactRecord {
                vector_bytes: vec![0; 2048],
                source_text: "fact".into(),
                confidence: 0.5,
                domain: None,
                cycle: 0,
                is_causal: false,
            },
            claim_context: None,
        };
        let link = EvidenceLink {
            from_fact_id: 1,
            to_fact_id: 2,
            relation: EvidenceRelation::SemanticNeighbor,
            status: EvidenceLinkStatus::Candidate,
            evidence_ids: Vec::new(),
            rationale: None,
            causal_argument: None,
            transmission_chain: None,
            detected_at_cycle: 0,
        };
        assert!(validate_graph_snapshot(&[fact], &[link])
            .unwrap_err()
            .contains("absent from graph snapshot"));
    }

    #[test]
    fn load_graph_snapshot_rejects_tampered_persisted_link() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_tampered_link_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let context = |witness: &str, family: &str, evidence: &str| ClaimContext {
            subject: "subject".into(),
            predicate: "predicate".into(),
            object: "object".into(),
            witness_id: witness.into(),
            source_family_id: family.into(),
            context_id: "shared-context".into(),
            temporal_scope: Some("period-a".into()),
            event_time: Some(1),
            claim_kind: super::super::claim_provenance::ClaimKind::SourceAttested,
            polarity: super::super::claim_provenance::ClaimPolarity::Positive,
            evidence_ids: vec![evidence.into()],
        };
        let facts = vec![
            GraphFactRecord {
                fact_id: Some(10),
                fact: FactRecord {
                    vector_bytes: vec![10; 2048],
                    source_text: "claim a".into(),
                    confidence: 0.5,
                    domain: None,
                    cycle: 1,
                    is_causal: false,
                },
                claim_context: Some(context("witness-a", "family-a", "evidence-a")),
            },
            GraphFactRecord {
                fact_id: Some(20),
                fact: FactRecord {
                    vector_bytes: vec![20; 2048],
                    source_text: "claim b".into(),
                    confidence: 0.5,
                    domain: None,
                    cycle: 2,
                    is_causal: false,
                },
                claim_context: Some(context("witness-b", "family-b", "evidence-b")),
            },
        ];
        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence.save_graph_facts(&facts).unwrap();

        let tampered = EvidenceLink {
            from_fact_id: 10,
            to_fact_id: 20,
            relation: EvidenceRelation::Corroborates,
            status: EvidenceLinkStatus::Adjudicated,
            evidence_ids: vec![
                "evidence-a".into(),
                "evidence-b".into(),
                "attacker-added-unbound-id".into(),
            ],
            rationale: Some("tampered provenance payload".into()),
            causal_argument: None,
            transmission_chain: None,
            detected_at_cycle: 3,
        };
        let connection = rusqlite::Connection::open(&db_path).unwrap();
        connection
            .execute(
                "INSERT INTO knowledge_evidence_links (link_json) VALUES (?1)",
                rusqlite::params![serde_json::to_string(&tampered).unwrap()],
            )
            .unwrap();
        drop(connection);

        let error = persistence.load_graph_snapshot().unwrap_err();
        assert!(error.contains("not declared by either endpoint context"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn graph_snapshot_rejects_unbound_adjudication_evidence_ids() {
        let context = |witness: &str, family: &str, evidence: &str| ClaimContext {
            subject: "symbol-x".into(),
            predicate: "associated_with".into(),
            object: "meaning-y".into(),
            witness_id: witness.into(),
            source_family_id: family.into(),
            context_id: "synthetic-context".into(),
            temporal_scope: None,
            event_time: None,
            claim_kind: super::super::claim_provenance::ClaimKind::SourceAttested,
            polarity: super::super::claim_provenance::ClaimPolarity::Positive,
            evidence_ids: vec![evidence.into()],
        };
        let fact = |id: u64, witness: &str, family: &str, evidence: &str| GraphFactRecord {
            fact_id: Some(id),
            fact: FactRecord {
                vector_bytes: vec![id as u8; 2048],
                source_text: format!("fact {id}"),
                confidence: 0.5,
                domain: None,
                cycle: id,
                is_causal: false,
            },
            claim_context: Some(context(witness, family, evidence)),
        };
        let link = EvidenceLink {
            from_fact_id: 1,
            to_fact_id: 2,
            relation: EvidenceRelation::Resembles,
            status: EvidenceLinkStatus::Adjudicated,
            evidence_ids: vec![
                "evidence-a".into(),
                "evidence-b".into(),
                "not-attached-to-either-context".into(),
            ],
            rationale: Some("attempted to attach an unbound evidence ID".into()),
            causal_argument: None,
            transmission_chain: None,
            detected_at_cycle: 3,
        };

        let error = validate_graph_snapshot(
            &[
                fact(1, "witness-a", "family-a", "evidence-a"),
                fact(2, "witness-b", "family-b", "evidence-b"),
            ],
            &[link],
        )
        .unwrap_err();
        assert!(error.contains("not declared by either endpoint context"));
    }

    #[test]
    fn graph_snapshot_rejects_adjudicated_link_without_provenance() {
        let fact = |id: u64| GraphFactRecord {
            fact_id: Some(id),
            fact: FactRecord {
                vector_bytes: vec![id as u8; 2048],
                source_text: format!("fact {id}"),
                confidence: 0.5,
                domain: None,
                cycle: id,
                is_causal: false,
            },
            claim_context: None,
        };
        let link = EvidenceLink {
            from_fact_id: 1,
            to_fact_id: 2,
            relation: EvidenceRelation::Corroborates,
            status: EvidenceLinkStatus::Adjudicated,
            evidence_ids: vec!["evidence-1".into()],
            rationale: Some("claimed corroboration".into()),
            causal_argument: None,
            transmission_chain: None,
            detected_at_cycle: 2,
        };
        assert!(validate_graph_snapshot(&[fact(1), fact(2)], &[link])
            .unwrap_err()
            .contains("has no claim context"));
    }

    #[test]
    fn atomic_graph_snapshot_replaces_facts_and_links_together() {
        let dir = std::env::temp_dir().join(format!("symthaea_atomic_snapshot_test_{}", std::process::id()));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let fact = |id: u64, source: &str| GraphFactRecord {
            fact_id: Some(id),
            fact: FactRecord {
                vector_bytes: vec![id as u8; 2048],
                source_text: source.into(),
                confidence: 0.7,
                domain: Some("test".into()),
                cycle: id,
                is_causal: false,
            },
            claim_context: None,
        };
        let link = EvidenceLink {
            from_fact_id: 1,
            to_fact_id: 2,
            relation: super::super::claim_provenance::EvidenceRelation::SemanticNeighbor,
            status: super::super::claim_provenance::EvidenceLinkStatus::Candidate,
            evidence_ids: Vec::new(),
            rationale: Some("similarity candidate only".into()),
            causal_argument: None,
            transmission_chain: None,
            detected_at_cycle: 2,
        };
        let mut persistence = KnowledgePersistence::new(&db_path);
        assert_eq!(
            persistence.save_graph_snapshot(&[fact(1, "old-a"), fact(2, "old-b")], &[link.clone()]).unwrap(),
            (2, 1)
        );
        assert_eq!(
            persistence.save_graph_snapshot(&[fact(3, "new-a")], &[]).unwrap(),
            (1, 0)
        );
        let facts = persistence.load_graph_facts().unwrap();
        let links = persistence.load_evidence_links().unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].fact_id, Some(3));
        assert!(links.is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_load_round_trips_typed_evidence_links() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea_evidence_links_test_{}",
            std::process::id()
        ));
        let db_path = dir.join("knowledge.db");
        let _ = std::fs::create_dir_all(&dir);
        let context = |witness: &str, family: &str, evidence: &str| ClaimContext {
            subject: "subject".into(),
            predicate: "predicate".into(),
            object: "object".into(),
            witness_id: witness.into(),
            source_family_id: family.into(),
            context_id: "shared-context".into(),
            temporal_scope: Some("period-a".into()),
            event_time: Some(1),
            claim_kind: super::super::claim_provenance::ClaimKind::SourceAttested,
            polarity: super::super::claim_provenance::ClaimPolarity::Positive,
            evidence_ids: vec![evidence.into()],
        };
        let make_fact = |id, source_text: &str, witness: &str, family: &str, evidence: &str| {
            GraphFactRecord {
                fact_id: Some(id),
                fact: FactRecord {
                    vector_bytes: vec![id as u8; 2048],
                    source_text: source_text.into(),
                    confidence: 0.5,
                    domain: None,
                    cycle: id,
                    is_causal: false,
                },
                claim_context: Some(context(witness, family, evidence)),
            }
        };
        let facts = vec![
            make_fact(10, "claim from witness A", "witness-a", "family-a", "evidence-a"),
            make_fact(20, "claim from witness B", "witness-b", "family-b", "evidence-b"),
        ];
        let link = EvidenceLink {
            from_fact_id: 10,
            to_fact_id: 20,
            relation: EvidenceRelation::Corroborates,
            status: EvidenceLinkStatus::Adjudicated,
            evidence_ids: vec!["evidence-a".into(), "evidence-b".into()],
            rationale: Some("Independent source witnesses support the same scoped claim".into()),
            causal_argument: None,
            transmission_chain: None,
            detected_at_cycle: 12,
        };
        let mut persistence = KnowledgePersistence::new(&db_path);
        persistence.save_graph_facts(&facts).unwrap();
        persistence.save_evidence_links(std::slice::from_ref(&link)).unwrap();
        let (loaded_facts, loaded_links) = persistence.load_graph_snapshot().unwrap();
        assert_eq!(loaded_facts.len(), 2);
        assert_eq!(loaded_links, vec![link.clone()]);
        assert_eq!(persistence.load_evidence_links().unwrap(), vec![link]);

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
