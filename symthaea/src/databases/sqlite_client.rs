//! SQLite-Backed Persistent Memory Storage
//!
//! Production-ready persistent storage for Symthaea's consciousness system.
//! Memories survive process restarts, enabling true continuity of experience.
//!
//! # Features
//!
//! - **Zero dependencies**: Uses rusqlite with bundled SQLite
//! - **ACID transactions**: Reliable storage with crash recovery
//! - **HDC-native**: Binary hypervector encodings stored efficiently as BLOBs
//! - **Indexed**: Timestamps, memory types, and phi values for fast filtering
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use symthaea::databases::{SqliteMemory, ConsciousnessDatabase, MemoryRecord, MemoryType};
//! use symthaea::hdc::BinaryHV;
//!
//! // Create persistent database
//! let db = SqliteMemory::new("./data/memories.db")?;
//!
//! // Or use in-memory for testing
//! let test_db = SqliteMemory::in_memory()?;
//!
//! // Store a memory
//! let record = MemoryRecord {
//!     id: "unique-id".to_string(),
//!     memory_type: MemoryType::Episodic,
//!     encoding: BinaryHV::random(42),
//!     content: "First conversation with user".to_string(),
//!     timestamp_ms: 1704067200000,
//!     valence: 0.7,
//!     arousal: 0.5,
//!     phi: 0.65,
//!     topics: vec!["greeting".to_string()],
//!     metadata: "{}".to_string(),
//!     consolidation_strength: 0.0,
//!     retrieval_count: 0,
//! };
//! db.store(record).await?;
//!
//! // Search for similar memories
//! let results = db.search_similar(&BinaryHV::random(42), 10).await?;
//! println!("Found {} similar memories", results.len());
//! ```
//!
//! # Schema
//!
//! The database uses a single `memories` table:
//!
//! | Column | Type | Description |
//! |--------|------|-------------|
//! | id | TEXT PRIMARY KEY | Unique identifier |
//! | encoding | BLOB | 2048-byte BinaryHV hypervector |
//! | timestamp_ms | INTEGER | Unix timestamp in milliseconds |
//! | memory_type | TEXT | "episodic", "semantic", "procedural", "working" |
//! | content | TEXT | Human-readable content |
//! | valence | REAL | Emotional valence (-1.0 to 1.0) |
//! | arousal | REAL | Emotional arousal (0.0 to 1.0) |
//! | phi | REAL | Integrated information value |
//! | topics | TEXT | JSON array of topic strings |
//! | metadata | TEXT | JSON object for extensibility |
//!
//! # Thread Safety
//!
//! The database uses `Mutex<Connection>` for thread-safe access. While this
//! serializes database operations, SQLite in WAL mode can handle concurrent
//! reads efficiently. Calls are wrapped in `spawn_blocking` to avoid stalling
//! async runtimes.

use super::{
    ConsciousnessDatabase, DatabaseError, DatabaseStats, DbResult, MemoryRecord, MemoryType,
    SearchResult,
};
use crate::infrastructure::lock_guard::ResilientMutex;
use async_trait::async_trait;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::{Arc, Mutex};
use symthaea_core::hdc::binary_hv::BinaryHV;

// ═══════════════════════════════════════════════════════════════════════════════
// LSH CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Number of LSH bands. Each band produces one hash bucket.
/// More bands = higher recall (fewer false negatives) but more candidates.
const LSH_NUM_BANDS: usize = 6;

/// Number of bit positions sampled per band (rows per band).
/// Higher = more selective per band (fewer false positives per band).
const LSH_ROWS_PER_BAND: usize = 32;

/// Minimum record count before LSH filtering kicks in.
/// Below this threshold, brute-force is fast enough.
const LSH_MIN_RECORDS: usize = 500;

/// SQLite-backed persistent memory database.
///
/// Provides durable storage for consciousness memories using an embedded SQLite
/// database. Implements [`ConsciousnessDatabase`] for the standard memory API.
///
/// # Thread Safety
///
/// Uses a `Mutex<Connection>` internally, making it safe to share across threads
/// via `Arc<SqliteMemory>`. Operations are serialized at the database level.
///
/// # Example
///
/// ```rust,ignore
/// use symthaea::databases::SqliteMemory;
/// use std::sync::Arc;
///
/// // Create shared database for multi-threaded access
/// let db = Arc::new(SqliteMemory::new("./memories.db")?);
///
/// // Clone Arc for different threads
/// let db_clone = Arc::clone(&db);
/// tokio::spawn(async move {
///     let count = db_clone.count().await?;
///     println!("Memory count: {}", count);
/// });
/// ```
pub struct SqliteMemory {
    /// Database connection protected by mutex for thread-safe access.
    conn: Arc<Mutex<Connection>>,

    /// Path to the database file (":memory:" for in-memory databases).
    path: String,
}

impl SqliteMemory {
    /// Create a new SQLite memory database at the given file path.
    ///
    /// Creates the database file and parent directories if they don't exist.
    /// If the database already exists, it opens and validates the schema.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the SQLite database file
    ///
    /// # Errors
    ///
    /// Returns [`DatabaseError::ConnectionFailed`] if:
    /// - Parent directory cannot be created
    /// - Database file cannot be opened/created
    /// - Schema initialization fails
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use symthaea::databases::SqliteMemory;
    ///
    /// // Create in a specific directory
    /// let db = SqliteMemory::new("./data/consciousness/memories.db")?;
    ///
    /// // Create in temp directory
    /// let temp_db = SqliteMemory::new(std::env::temp_dir().join("test.db"))?;
    /// ```
    pub fn new<P: AsRef<Path>>(path: P) -> DbResult<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();

        // Ensure parent directory exists
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                DatabaseError::ConnectionFailed(format!("Failed to create directory: {e}"))
            })?;
        }

        let conn = Connection::open(&path)
            .map_err(|e| DatabaseError::ConnectionFailed(format!("SQLite open failed: {e}")))?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            path: path_str,
        };

        db.initialize_schema()?;

        eprintln!("[SqliteMemory] Initialized at: {}", db.path);
        Ok(db)
    }

    /// Create an in-memory database for testing or ephemeral storage.
    ///
    /// The database exists only for the lifetime of the `SqliteMemory` instance.
    /// All data is lost when the instance is dropped. Useful for unit tests
    /// and temporary sessions.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use symthaea::databases::{SqliteMemory, ConsciousnessDatabase};
    ///
    /// #[tokio::test]
    /// async fn test_memory_operations() {
    ///     let db = SqliteMemory::in_memory().unwrap();
    ///
    ///     // Test operations...
    ///     assert_eq!(db.count().await.unwrap(), 0);
    ///
    ///     // Database automatically cleaned up when `db` goes out of scope
    /// }
    /// ```
    pub fn in_memory() -> DbResult<Self> {
        let conn = Connection::open_in_memory().map_err(|e| {
            DatabaseError::ConnectionFailed(format!("SQLite in-memory failed: {e}"))
        })?;

        let db = Self {
            conn: Arc::new(Mutex::new(conn)),
            path: ":memory:".to_string(),
        };

        db.initialize_schema()?;
        Ok(db)
    }

    /// Initialize the database schema.
    ///
    /// Creates the `memories` table and indexes if they don't exist.
    fn initialize_schema(&self) -> DbResult<()> {
        let conn = self.conn.lock_resilient("sqlite");

        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                encoding BLOB NOT NULL,
                timestamp_ms INTEGER NOT NULL,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                valence REAL NOT NULL,
                arousal REAL NOT NULL,
                phi REAL NOT NULL,
                topics TEXT NOT NULL,
                metadata TEXT NOT NULL,
                consolidation_strength REAL NOT NULL DEFAULT 0.0,
                retrieval_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp_ms);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_phi ON memories(phi);
        "#,
        )
        .map_err(|e| DatabaseError::QueryFailed(format!("Schema creation failed: {e}")))?;

        // Migration: add reconsolidation columns to existing databases
        let _ = conn.execute_batch(
            r#"
            ALTER TABLE memories ADD COLUMN consolidation_strength REAL NOT NULL DEFAULT 0.0;
            ALTER TABLE memories ADD COLUMN retrieval_count INTEGER NOT NULL DEFAULT 0;
        "#,
        );

        // Migration: create LSH index table for approximate nearest-neighbor search
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS vector_lsh (
                memory_id TEXT NOT NULL,
                band_idx INTEGER NOT NULL,
                band_hash INTEGER NOT NULL,
                PRIMARY KEY (memory_id, band_idx),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_lsh_band_hash ON vector_lsh(band_idx, band_hash);
        "#,
        )
        .map_err(|e| DatabaseError::QueryFailed(format!("LSH schema creation failed: {e}")))?;

        // Backfill LSH index for any existing records
        let record_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .unwrap_or(0);

        if record_count > 0 {
            let lsh_count: i64 = conn
                .query_row(
                    "SELECT COUNT(DISTINCT memory_id) FROM vector_lsh",
                    [],
                    |row| row.get(0),
                )
                .unwrap_or(0);

            if lsh_count < record_count {
                Self::backfill_lsh_index(&conn)?;
            }
        }

        Ok(())
    }

    async fn with_connection<T, F>(&self, f: F) -> DbResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&Connection) -> DbResult<T> + Send + 'static,
    {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock_resilient("sqlite");
            f(&conn)
        })
        .await
        .map_err(|e| DatabaseError::Other(format!("Blocking task failed: {e}")))?
    }

    /// Serialize BinaryHV to bytes (2048 bytes = 16,384 bits).
    #[doc(hidden)]
    fn hv_to_bytes(hv: &BinaryHV) -> Vec<u8> {
        hv.0.to_vec()
    }

    /// Deserialize bytes to BinaryHV.
    #[doc(hidden)]
    fn bytes_to_hv(bytes: &[u8]) -> BinaryHV {
        if bytes.len() >= BinaryHV::BYTES {
            let mut arr = [0u8; BinaryHV::BYTES];
            arr.copy_from_slice(&bytes[..BinaryHV::BYTES]);
            BinaryHV(arr)
        } else {
            BinaryHV::zero()
        }
    }

    /// Convert MemoryType to database string representation.
    #[doc(hidden)]
    fn memory_type_to_str(mt: MemoryType) -> &'static str {
        match mt {
            MemoryType::Episodic => "episodic",
            MemoryType::Semantic => "semantic",
            MemoryType::Procedural => "procedural",
            MemoryType::Working => "working",
        }
    }

    /// Convert database string to MemoryType.
    #[doc(hidden)]
    fn str_to_memory_type(s: &str) -> MemoryType {
        match s {
            "episodic" => MemoryType::Episodic,
            "semantic" => MemoryType::Semantic,
            "procedural" => MemoryType::Procedural,
            "working" => MemoryType::Working,
            _ => MemoryType::Episodic,
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // LSH (Locality-Sensitive Hashing) for approximate nearest-neighbor
    // ═════════════════════════════════════════════════════════════════════════

    /// Compute LSH band hashes for a BinaryHV.
    ///
    /// Uses deterministic bit sampling: band `b` samples bit positions
    /// derived from a seed that mixes the band index. Each band produces
    /// a 32-bit hash from `LSH_ROWS_PER_BAND` sampled bits.
    fn lsh_band_hashes(hv: &BinaryHV) -> [i64; LSH_NUM_BANDS] {
        let mut hashes = [0i64; LSH_NUM_BANDS];
        for band in 0..LSH_NUM_BANDS {
            let mut hash: u32 = 0;
            for row in 0..LSH_ROWS_PER_BAND {
                // Deterministic bit position from band + row
                // Use golden-ratio hashing for good distribution across 16384 bits
                let bit_pos = ((band * LSH_ROWS_PER_BAND + row).wrapping_mul(0x9E3779B9))
                    % (BinaryHV::BYTES * 8);
                let byte_idx = bit_pos / 8;
                let bit_idx = bit_pos % 8;
                if (hv.0[byte_idx] >> bit_idx) & 1 == 1 {
                    hash |= 1u32 << (row % 32);
                }
            }
            hashes[band] = hash as i64;
        }
        hashes
    }

    /// Insert LSH index entries for a memory record.
    fn insert_lsh_entries(conn: &Connection, id: &str, hv: &BinaryHV) -> DbResult<()> {
        let hashes = Self::lsh_band_hashes(hv);
        let mut stmt = conn.prepare_cached(
            "INSERT OR REPLACE INTO vector_lsh (memory_id, band_idx, band_hash) VALUES (?1, ?2, ?3)"
        ).map_err(|e| DatabaseError::InsertFailed(format!("LSH prepare failed: {e}")))?;

        for (band, &hash) in hashes.iter().enumerate() {
            stmt.execute(params![id, band as i64, hash])
                .map_err(|e| DatabaseError::InsertFailed(format!("LSH insert failed: {e}")))?;
        }
        Ok(())
    }

    /// Delete LSH index entries for a memory record.
    fn delete_lsh_entries(conn: &Connection, id: &str) -> DbResult<()> {
        conn.execute("DELETE FROM vector_lsh WHERE memory_id = ?1", [id])
            .map_err(|e| DatabaseError::QueryFailed(format!("LSH delete failed: {e}")))?;
        Ok(())
    }

    /// Query candidate memory IDs that share at least one LSH band hash with the query.
    fn lsh_candidates(conn: &Connection, query: &BinaryHV) -> DbResult<Vec<String>> {
        let hashes = Self::lsh_band_hashes(query);

        // Build query: match any band where band_idx = ? AND band_hash = ?
        // Using UNION for each band is cleaner than OR chains
        let mut candidates = std::collections::HashSet::new();
        let mut stmt = conn
            .prepare_cached(
                "SELECT DISTINCT memory_id FROM vector_lsh WHERE band_idx = ?1 AND band_hash = ?2",
            )
            .map_err(|e| DatabaseError::QueryFailed(format!("LSH query prepare failed: {e}")))?;

        for (band, &hash) in hashes.iter().enumerate() {
            let rows = stmt
                .query_map(params![band as i64, hash], |row| row.get::<_, String>(0))
                .map_err(|e| DatabaseError::QueryFailed(format!("LSH query failed: {e}")))?;

            for id in rows.flatten() {
                candidates.insert(id);
            }
        }

        Ok(candidates.into_iter().collect())
    }

    /// Parse a row into a MemoryRecord.
    fn row_to_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryRecord> {
        let encoding_bytes: Vec<u8> = row.get(1)?;
        let topics_json: String = row.get(8)?;
        let topics: Vec<String> = match serde_json::from_str(&topics_json) {
            Ok(t) => t,
            Err(e) => {
                tracing::warn!("Failed to deserialize topics: {}. Using empty array.", e);
                Vec::new()
            }
        };

        Ok(MemoryRecord {
            id: row.get(0)?,
            encoding: Self::bytes_to_hv(&encoding_bytes),
            timestamp_ms: {
                let ts = row.get::<_, i64>(2)?;
                if ts < 0 {
                    tracing::warn!("Negative timestamp {} found, using 0", ts);
                    0u64
                } else {
                    ts as u64
                }
            },
            memory_type: Self::str_to_memory_type(&row.get::<_, String>(3)?),
            content: row.get(4)?,
            valence: row.get::<_, f64>(5)? as f32,
            arousal: row.get::<_, f64>(6)? as f32,
            phi: row.get::<_, f64>(7)?,
            topics,
            metadata: row.get(9)?,
            consolidation_strength: row.get::<_, f64>(10).unwrap_or(0.0),
            retrieval_count: row.get::<_, i64>(11).unwrap_or(0) as u32,
        })
    }

    /// Fetch all records (brute-force path), limited to `limit` most recent.
    fn fetch_all_records(conn: &Connection, limit: usize) -> DbResult<Vec<MemoryRecord>> {
        let mut stmt = conn.prepare(
            "SELECT id, encoding, timestamp_ms, memory_type, content, valence, arousal, phi, topics, metadata, consolidation_strength, retrieval_count
             FROM memories ORDER BY timestamp_ms DESC LIMIT ?1"
        ).map_err(|e| DatabaseError::QueryFailed(format!("Prepare failed: {e}")))?;

        let rows = stmt
            .query_map([limit as i64], Self::row_to_record)
            .map_err(|e| DatabaseError::QueryFailed(format!("Query failed: {e}")))?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Fetch records by a set of IDs (LSH-filtered path).
    fn fetch_records_by_ids(conn: &Connection, ids: &[String]) -> DbResult<Vec<MemoryRecord>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }

        // Use batched queries to avoid SQLite variable limit (999)
        let mut records = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(500) {
            let placeholders: String = chunk
                .iter()
                .enumerate()
                .map(|(i, _)| format!("?{}", i + 1))
                .collect::<Vec<_>>()
                .join(",");

            let sql = format!(
                "SELECT id, encoding, timestamp_ms, memory_type, content, valence, arousal, phi, topics, metadata, consolidation_strength, retrieval_count
                 FROM memories WHERE id IN ({placeholders})"
            );

            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| DatabaseError::QueryFailed(format!("Prepare failed: {e}")))?;

            let params: Vec<&dyn rusqlite::types::ToSql> = chunk
                .iter()
                .map(|id| id as &dyn rusqlite::types::ToSql)
                .collect();

            let rows = stmt
                .query_map(params.as_slice(), Self::row_to_record)
                .map_err(|e| DatabaseError::QueryFailed(format!("Query failed: {e}")))?;

            records.extend(rows.filter_map(|r| r.ok()));
        }

        Ok(records)
    }

    /// Backfill LSH index for all existing records that aren't yet indexed.
    fn backfill_lsh_index(conn: &Connection) -> DbResult<usize> {
        // Find records not yet in the LSH index
        let mut stmt = conn.prepare(
            "SELECT id, encoding FROM memories WHERE id NOT IN (SELECT DISTINCT memory_id FROM vector_lsh)"
        ).map_err(|e| DatabaseError::QueryFailed(format!("LSH backfill query failed: {e}")))?;

        let rows: Vec<(String, Vec<u8>)> = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
            })
            .map_err(|e| DatabaseError::QueryFailed(format!("LSH backfill failed: {e}")))?
            .filter_map(|r| r.ok())
            .collect();

        let count = rows.len();
        for (id, encoding_bytes) in rows {
            let hv = Self::bytes_to_hv(&encoding_bytes);
            Self::insert_lsh_entries(conn, &id, &hv)?;
        }

        if count > 0 {
            tracing::info!(records = count, "LSH index backfilled");
        }
        Ok(count)
    }
}

#[async_trait]
impl ConsciousnessDatabase for SqliteMemory {
    async fn store(&self, record: MemoryRecord) -> DbResult<()> {
        self.with_connection(move |conn| {
            let encoding_bytes = Self::hv_to_bytes(&record.encoding);
            let memory_type_str = Self::memory_type_to_str(record.memory_type);
            let topics_json = match serde_json::to_string(&record.topics) {
                Ok(json) => json,
                Err(e) => {
                    tracing::warn!("Failed to serialize topics: {}. Using empty array.", e);
                    "[]".to_string()
                }
            };

            conn.execute(
                r#"INSERT OR REPLACE INTO memories
                   (id, encoding, timestamp_ms, memory_type, content, valence, arousal, phi, topics, metadata, consolidation_strength, retrieval_count)
                   VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)"#,
                params![
                    record.id,
                    encoding_bytes,
                    record.timestamp_ms as i64,
                    memory_type_str,
                    record.content,
                    record.valence as f64,
                    record.arousal as f64,
                    { record.phi },
                    topics_json,
                    record.metadata,
                    record.consolidation_strength,
                    record.retrieval_count,
                ],
            ).map_err(|e| DatabaseError::InsertFailed(format!("Insert failed: {e}")))?;

            // Update LSH index for this record
            Self::insert_lsh_entries(conn, &record.id, &record.encoding)?;

            Ok(())
        })
        .await
    }

    async fn search_similar(&self, query: &BinaryHV, top_k: usize) -> DbResult<Vec<SearchResult>> {
        let query = *query;
        self.with_connection(move |conn| {
            // Check total record count to decide LSH vs brute-force
            let total: i64 = conn
                .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
                .unwrap_or(0);

            let records: Vec<MemoryRecord> = if total as usize >= LSH_MIN_RECORDS {
                // LSH-accelerated path: get candidates first, then fetch only those
                let candidates = Self::lsh_candidates(conn, &query)?;

                if candidates.is_empty() {
                    // No LSH matches — fall back to brute-force on recent records
                    Self::fetch_all_records(conn, 1000)?
                } else {
                    Self::fetch_records_by_ids(conn, &candidates)?
                }
            } else {
                // Small dataset — brute-force is fine
                Self::fetch_all_records(conn, 1000)?
            };

            // Compute similarities and sort
            let mut results: Vec<SearchResult> = records
                .into_iter()
                .map(|record| {
                    let similarity = query.similarity(&record.encoding);
                    SearchResult { record, similarity }
                })
                .collect();

            // Sort by similarity descending
            results.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
            results.truncate(top_k);

            // Reconsolidate: bump retrieval stats on returned results
            if !results.is_empty() {
                let ids: Vec<&str> = results.iter().map(|r| r.record.id.as_str()).collect();
                for chunk in ids.chunks(500) {
                    let placeholders: String = chunk
                        .iter()
                        .enumerate()
                        .map(|(i, _)| format!("?{}", i + 1))
                        .collect::<Vec<_>>()
                        .join(",");
                    let sql = format!(
                        "UPDATE memories SET retrieval_count = retrieval_count + 1, \
                         consolidation_strength = MIN(consolidation_strength + 0.05, 1.0) \
                         WHERE id IN ({placeholders})"
                    );
                    let params: Vec<&dyn rusqlite::types::ToSql> = chunk
                        .iter()
                        .map(|id| id as &dyn rusqlite::types::ToSql)
                        .collect();
                    let _ = conn.execute(&sql, params.as_slice());
                }
            }

            Ok(results)
        })
        .await
    }

    async fn get(&self, id: &str) -> DbResult<Option<MemoryRecord>> {
        let id = id.to_string();
        self.with_connection(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT id, encoding, timestamp_ms, memory_type, content, valence, arousal, phi, topics, metadata, consolidation_strength, retrieval_count
                 FROM memories WHERE id = ?1"
            ).map_err(|e| DatabaseError::QueryFailed(format!("Prepare failed: {e}")))?;

            let result = stmt.query_row([id], Self::row_to_record);

            match result {
                Ok(record) => Ok(Some(record)),
                Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                Err(e) => Err(DatabaseError::QueryFailed(format!("Get failed: {e}"))),
            }
        })
        .await
    }

    async fn delete(&self, id: &str) -> DbResult<bool> {
        let id = id.to_string();
        self.with_connection(move |conn| {
            // Clean up LSH index entries (CASCADE may not be enabled in all SQLite builds)
            Self::delete_lsh_entries(conn, &id)?;

            let affected = conn
                .execute("DELETE FROM memories WHERE id = ?1", [&id])
                .map_err(|e| DatabaseError::QueryFailed(format!("Delete failed: {e}")))?;

            Ok(affected > 0)
        })
        .await
    }

    async fn count(&self) -> DbResult<usize> {
        self.with_connection(|conn| {
            let count: i64 = conn
                .query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
                .map_err(|e| DatabaseError::QueryFailed(format!("Count failed: {e}")))?;

            Ok(count as usize)
        })
        .await
    }

    async fn health_check(&self) -> DbResult<bool> {
        self.with_connection(|conn| {
            conn.execute_batch("SELECT 1")
                .map_err(|e| DatabaseError::QueryFailed(format!("Health check failed: {e}")))?;

            Ok(true)
        })
        .await
    }

    async fn list_all(&self) -> DbResult<Vec<MemoryRecord>> {
        self.with_connection(|conn| {
            let mut stmt = conn.prepare(
                "SELECT id, encoding, timestamp_ms, memory_type, content, valence, arousal, phi, topics, metadata, consolidation_strength, retrieval_count
                 FROM memories ORDER BY timestamp_ms ASC"
            ).map_err(|e| DatabaseError::QueryFailed(format!("list_all prepare failed: {e}")))?;

            let rows = stmt.query_map([], Self::row_to_record)
                .map_err(|e| DatabaseError::QueryFailed(format!("list_all query failed: {e}")))?;

            Ok(rows.filter_map(|r| r.ok()).collect())
        })
        .await
    }

    async fn stats(&self) -> DbResult<DatabaseStats> {
        let path = self.path.clone();
        self.with_connection(move |conn| {
            // Get total record count
            let total_records: i64 = conn.query_row(
                "SELECT COUNT(*) FROM memories",
                [],
                |row| row.get(0)
            ).map_err(|e| DatabaseError::QueryFailed(format!("Count query failed: {e}")))?;

            // Get SQLite pragma values for database metrics
            let page_size: i64 = conn.query_row(
                "PRAGMA page_size",
                [],
                |row| row.get(0)
            ).unwrap_or(4096);

            let page_count: i64 = conn.query_row(
                "PRAGMA page_count",
                [],
                |row| row.get(0)
            ).unwrap_or(0);

            let freelist_count: i64 = conn.query_row(
                "PRAGMA freelist_count",
                [],
                |row| row.get(0)
            ).unwrap_or(0);

            // Get cache statistics
            let cache_hits: i64 = conn.query_row(
                "SELECT cache_hit FROM pragma_database_list() LIMIT 1",
                [],
                |row| row.get(0)
            ).unwrap_or(0);

            let cache_misses: i64 = conn.query_row(
                "SELECT cache_miss FROM pragma_database_list() LIMIT 1",
                [],
                |row| row.get(0)
            ).unwrap_or(0);

            // Calculate cache hit ratio
            let total_cache_ops = cache_hits + cache_misses;
            let cache_hit_ratio = if total_cache_ops > 0 {
                cache_hits as f64 / total_cache_ops as f64
            } else {
                0.0
            };

            // Get memory type distribution
            let mut type_counts_stmt = conn.prepare(
                "SELECT memory_type, COUNT(*) as cnt FROM memories GROUP BY memory_type ORDER BY cnt DESC"
            ).map_err(|e| DatabaseError::QueryFailed(format!("Type counts query failed: {e}")))?;

            let memory_type_counts: Vec<(String, usize)> = type_counts_stmt
                .query_map([], |row| {
                    let mt: String = row.get(0)?;
                    let cnt: i64 = row.get(1)?;
                    Ok((mt, cnt as usize))
                })
                .map_err(|e| DatabaseError::QueryFailed(format!("Type counts failed: {e}")))?
                .filter_map(|r| r.ok())
                .collect();

            // Get average phi
            let avg_phi: f64 = conn.query_row(
                "SELECT COALESCE(AVG(phi), 0.0) FROM memories",
                [],
                |row| row.get(0)
            ).unwrap_or(0.0);

            // Get timestamp range
            let oldest_timestamp_ms: i64 = conn.query_row(
                "SELECT COALESCE(MIN(timestamp_ms), 0) FROM memories",
                [],
                |row| row.get(0)
            ).unwrap_or(0);

            let newest_timestamp_ms: i64 = conn.query_row(
                "SELECT COALESCE(MAX(timestamp_ms), 0) FROM memories",
                [],
                |row| row.get(0)
            ).unwrap_or(0);

            // Get journal mode for backend status
            let journal_mode: String = conn.query_row(
                "PRAGMA journal_mode",
                [],
                |row| row.get(0)
            ).unwrap_or_else(|_| "unknown".to_string());

            // Calculate database size
            let database_size_bytes = (page_count as u64) * (page_size as u64);

            // Determine backend status
            let backend_status = if path == ":memory:" {
                "in_memory".to_string()
            } else {
                format!("file:{journal_mode}")
            };

            Ok(DatabaseStats {
                total_records: total_records as usize,
                database_size_bytes,
                page_count: page_count as u64,
                page_size: page_size as u64,
                freelist_count: freelist_count as u64,
                cache_hit_ratio,
                cache_hits: cache_hits as u64,
                cache_misses: cache_misses as u64,
                avg_query_latency_us: 0, // Not tracked yet
                total_queries: 0,        // Not tracked yet
                memory_type_counts,
                avg_phi,
                oldest_timestamp_ms: oldest_timestamp_ms as u64,
                newest_timestamp_ms: newest_timestamp_ms as u64,
                backend_status,
            })
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sqlite_memory_basic() {
        let db = SqliteMemory::in_memory().unwrap();

        // Store a memory
        let record = MemoryRecord {
            id: "test-1".to_string(),
            encoding: BinaryHV::random(42),
            timestamp_ms: 1234567890,
            memory_type: MemoryType::Episodic,
            content: "Hello, I am Symthaea".to_string(),
            valence: 0.8,
            arousal: 0.5,
            phi: 0.75,
            topics: vec!["greeting".to_string()],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };

        db.store(record.clone()).await.unwrap();

        // Count
        assert_eq!(db.count().await.unwrap(), 1);

        // Get by ID
        let retrieved = db.get("test-1").await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Hello, I am Symthaea");

        // Search similar
        let results = db.search_similar(&BinaryHV::random(42), 5).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].similarity > 0.99); // Same seed = identical

        // Delete
        assert!(db.delete("test-1").await.unwrap());
        assert_eq!(db.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_sqlite_persistence() {
        let temp_dir = std::env::temp_dir();
        let db_path = temp_dir.join("symthaea_test.db");

        // Clean up from previous runs
        let _ = std::fs::remove_file(&db_path);

        // Create and store
        {
            let db = SqliteMemory::new(&db_path).unwrap();
            let record = MemoryRecord {
                id: "persist-test".to_string(),
                encoding: BinaryHV::random(123),
                timestamp_ms: 1234567890,
                memory_type: MemoryType::Semantic,
                content: "I remember this".to_string(),
                valence: 0.5,
                arousal: 0.3,
                phi: 0.6,
                topics: vec!["test".to_string()],
                metadata: "{}".to_string(),
                consolidation_strength: 0.0,
                retrieval_count: 0,
            };
            db.store(record).await.unwrap();
        }

        // Reopen and verify
        {
            let db = SqliteMemory::new(&db_path).unwrap();
            let record = db.get("persist-test").await.unwrap().unwrap();
            assert_eq!(record.content, "I remember this");
        }

        // Clean up
        let _ = std::fs::remove_file(&db_path);
    }

    #[test]
    fn test_lsh_band_hashes_deterministic() {
        let hv = BinaryHV::random(42);
        let h1 = SqliteMemory::lsh_band_hashes(&hv);
        let h2 = SqliteMemory::lsh_band_hashes(&hv);
        assert_eq!(h1, h2, "Same vector should produce same hashes");
    }

    #[test]
    fn test_lsh_band_hashes_different_vectors() {
        let hv1 = BinaryHV::random(1);
        let hv2 = BinaryHV::random(2);
        let h1 = SqliteMemory::lsh_band_hashes(&hv1);
        let h2 = SqliteMemory::lsh_band_hashes(&hv2);
        // Random BinaryHVs should differ in at least some bands
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_lsh_similar_vectors_share_bands() {
        // Create a vector and a close neighbor (flip only a few bits)
        let hv1 = BinaryHV::random(42);
        let mut hv2 = hv1;
        // Flip 50 out of 16384 bits (99.7% similar)
        for i in 0..50 {
            let byte_idx = i * 3 % BinaryHV::BYTES;
            hv2.0[byte_idx] ^= 1;
        }
        let h1 = SqliteMemory::lsh_band_hashes(&hv1);
        let h2 = SqliteMemory::lsh_band_hashes(&hv2);

        // Very similar vectors should share most LSH bands
        let matching_bands = h1.iter().zip(h2.iter()).filter(|(a, b)| a == b).count();
        assert!(
            matching_bands >= 3,
            "Very similar vectors should share at least 3 of 6 bands, got {}",
            matching_bands
        );
    }

    #[tokio::test]
    async fn test_lsh_index_populated_on_store() {
        let db = SqliteMemory::in_memory().unwrap();
        let record = MemoryRecord {
            id: "lsh-test-1".to_string(),
            encoding: BinaryHV::random(42),
            timestamp_ms: 1234567890,
            memory_type: MemoryType::Episodic,
            content: "LSH test".to_string(),
            valence: 0.5,
            arousal: 0.5,
            phi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(record).await.unwrap();

        // Check that LSH entries were created
        let conn = db.conn.lock().unwrap();
        let lsh_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM vector_lsh WHERE memory_id = 'lsh-test-1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(lsh_count, LSH_NUM_BANDS as i64);
    }

    #[tokio::test]
    async fn test_lsh_entries_deleted_on_delete() {
        let db = SqliteMemory::in_memory().unwrap();
        let record = MemoryRecord {
            id: "lsh-del-1".to_string(),
            encoding: BinaryHV::random(42),
            timestamp_ms: 1234567890,
            memory_type: MemoryType::Episodic,
            content: "Will be deleted".to_string(),
            valence: 0.5,
            arousal: 0.5,
            phi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(record).await.unwrap();
        db.delete("lsh-del-1").await.unwrap();

        let conn = db.conn.lock().unwrap();
        let lsh_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM vector_lsh WHERE memory_id = 'lsh-del-1'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(lsh_count, 0);
    }

    #[tokio::test]
    async fn test_search_similar_finds_exact_match_via_lsh() {
        let db = SqliteMemory::in_memory().unwrap();
        let target_hv = BinaryHV::random(42);

        // Store 600 records to trigger LSH path (>= LSH_MIN_RECORDS)
        for i in 0..600 {
            let record = MemoryRecord {
                id: format!("lsh-search-{}", i),
                encoding: BinaryHV::random(i + 100),
                timestamp_ms: 1000000 + i,
                memory_type: MemoryType::Episodic,
                content: format!("Record {}", i),
                valence: 0.5,
                arousal: 0.5,
                phi: 0.5,
                topics: vec![],
                metadata: "{}".to_string(),
                consolidation_strength: 0.0,
                retrieval_count: 0,
            };
            db.store(record).await.unwrap();
        }

        // Store the target
        let target_record = MemoryRecord {
            id: "target".to_string(),
            encoding: target_hv,
            timestamp_ms: 2000000,
            memory_type: MemoryType::Episodic,
            content: "Target record".to_string(),
            valence: 0.5,
            arousal: 0.5,
            phi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(target_record).await.unwrap();

        // Search for the target — should find it via LSH
        let results = db.search_similar(&target_hv, 5).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].record.id, "target");
        assert!(results[0].similarity > 0.99);
    }

    #[tokio::test]
    async fn test_sqlite_list_all() {
        let db = SqliteMemory::in_memory().unwrap();

        for i in 0..5u64 {
            let record = MemoryRecord {
                id: format!("list-all-{}", i),
                encoding: BinaryHV::random(i),
                timestamp_ms: 1000000 + i,
                memory_type: MemoryType::Episodic,
                content: format!("Record {}", i),
                valence: 0.5,
                arousal: 0.5,
                phi: 0.5,
                topics: vec![],
                metadata: "{}".to_string(),
                consolidation_strength: 0.0,
                retrieval_count: 0,
            };
            db.store(record).await.unwrap();
        }

        let all = db.list_all().await.unwrap();
        assert_eq!(all.len(), 5);
        // Should be ordered by timestamp ascending
        assert_eq!(all[0].id, "list-all-0");
        assert_eq!(all[4].id, "list-all-4");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // DATABASE EVICTION PIPELINE INTEGRATION TESTS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Helper: create a MemoryRecord with minimal boilerplate
    fn make_record(id: &str, seed: u64, memory_type: MemoryType) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            encoding: BinaryHV::random(seed),
            timestamp_ms: 1_700_000_000_000 + seed,
            memory_type,
            content: format!("Record {}", id),
            valence: 0.5,
            arousal: 0.5,
            phi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        }
    }

    #[tokio::test]
    async fn test_database_store_and_retrieve_by_key() {
        let db = SqliteMemory::in_memory().unwrap();

        let hv = BinaryHV::random(77);
        let record = MemoryRecord {
            id: "retrieve-key-1".to_string(),
            encoding: hv,
            timestamp_ms: 1_700_000_000_000,
            memory_type: MemoryType::Episodic,
            content: "Test store-and-retrieve".to_string(),
            valence: 0.9,
            arousal: 0.3,
            phi: 0.85,
            topics: vec!["test".to_string(), "retrieval".to_string()],
            metadata: r#"{"session":"abc"}"#.to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };

        db.store(record).await.unwrap();

        // Retrieve by ID and verify all fields round-trip correctly
        let retrieved = db.get("retrieve-key-1").await.unwrap().unwrap();
        assert_eq!(retrieved.id, "retrieve-key-1");
        assert_eq!(retrieved.content, "Test store-and-retrieve");
        assert_eq!(retrieved.memory_type, MemoryType::Episodic);
        assert!((retrieved.valence - 0.9).abs() < 0.01);
        assert!((retrieved.arousal - 0.3).abs() < 0.01);
        assert!((retrieved.phi - 0.85).abs() < 0.001);
        assert_eq!(
            retrieved.topics,
            vec!["test".to_string(), "retrieval".to_string()]
        );
        assert_eq!(retrieved.metadata, r#"{"session":"abc"}"#);
        assert_eq!(retrieved.consolidation_strength, 0.0);
        assert_eq!(retrieved.retrieval_count, 0);

        // Verify the encoding bytes survived the round-trip
        assert!(
            hv.similarity(&retrieved.encoding) > 0.99,
            "Encoding should survive store/retrieve round-trip"
        );
    }

    #[tokio::test]
    async fn test_database_get_nonexistent_returns_none() {
        let db = SqliteMemory::in_memory().unwrap();
        let result = db.get("does-not-exist").await.unwrap();
        assert!(result.is_none(), "Getting a missing key should return None");
    }

    #[tokio::test]
    async fn test_database_store_overwrites_existing() {
        let db = SqliteMemory::in_memory().unwrap();

        let record_v1 = MemoryRecord {
            id: "overwrite-1".to_string(),
            encoding: BinaryHV::random(10),
            timestamp_ms: 1_000,
            memory_type: MemoryType::Working,
            content: "version 1".to_string(),
            valence: 0.1,
            arousal: 0.1,
            phi: 0.1,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(record_v1).await.unwrap();

        // Store again with same ID but different content
        let record_v2 = MemoryRecord {
            id: "overwrite-1".to_string(),
            encoding: BinaryHV::random(20),
            timestamp_ms: 2_000,
            memory_type: MemoryType::Semantic,
            content: "version 2".to_string(),
            valence: 0.9,
            arousal: 0.9,
            phi: 0.9,
            topics: vec!["updated".to_string()],
            metadata: "{}".to_string(),
            consolidation_strength: 0.5,
            retrieval_count: 3,
        };
        db.store(record_v2).await.unwrap();

        // Should still be one record, with v2 content
        assert_eq!(db.count().await.unwrap(), 1);
        let retrieved = db.get("overwrite-1").await.unwrap().unwrap();
        assert_eq!(retrieved.content, "version 2");
        assert_eq!(retrieved.memory_type, MemoryType::Semantic);
        assert_eq!(retrieved.retrieval_count, 3);
    }

    #[tokio::test]
    async fn test_database_search_similar_ordering() {
        let db = SqliteMemory::in_memory().unwrap();

        // Create a base vector and a near-neighbor
        let base = BinaryHV::random(42);
        let mut near = base;
        // Flip 100 bits out of 16384 (~99.4% similar)
        for i in 0..100 {
            let byte_idx = (i * 7) % BinaryHV::BYTES;
            near.0[byte_idx] ^= 0x01;
        }

        // Store a far vector, the near vector, and the exact match
        db.store(make_record("far", 999, MemoryType::Episodic))
            .await
            .unwrap();

        let near_record = MemoryRecord {
            id: "near".to_string(),
            encoding: near,
            timestamp_ms: 1_700_000_000_002,
            memory_type: MemoryType::Episodic,
            content: "Near neighbor".to_string(),
            valence: 0.5,
            arousal: 0.5,
            phi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(near_record).await.unwrap();

        let exact_record = MemoryRecord {
            id: "exact".to_string(),
            encoding: base,
            timestamp_ms: 1_700_000_000_003,
            memory_type: MemoryType::Episodic,
            content: "Exact match".to_string(),
            valence: 0.5,
            arousal: 0.5,
            phi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(exact_record).await.unwrap();

        // Search with the base vector
        let results = db.search_similar(&base, 3).await.unwrap();
        assert_eq!(results.len(), 3);

        // First result should be the exact match (similarity ~1.0)
        assert_eq!(results[0].record.id, "exact");
        assert!(results[0].similarity > 0.99);

        // Second result should be the near neighbor
        assert_eq!(results[1].record.id, "near");
        assert!(results[1].similarity > 0.90);

        // Results should be in descending similarity order
        for pair in results.windows(2) {
            assert!(
                pair[0].similarity >= pair[1].similarity,
                "Results must be sorted by similarity descending"
            );
        }
    }

    #[tokio::test]
    async fn test_database_search_similar_top_k_limit() {
        let db = SqliteMemory::in_memory().unwrap();

        // Store 10 records
        for i in 0..10u64 {
            db.store(make_record(
                &format!("topk-{}", i),
                i + 100,
                MemoryType::Episodic,
            ))
            .await
            .unwrap();
        }

        // Request top_k=3 — should return at most 3
        let results = db.search_similar(&BinaryHV::random(100), 3).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_database_lsh_cache_kicks_in_at_threshold() {
        // Verify that with < LSH_MIN_RECORDS the brute-force path works,
        // and with >= LSH_MIN_RECORDS the LSH path is used (both produce correct results).
        let db = SqliteMemory::in_memory().unwrap();
        let target_hv = BinaryHV::random(42);

        // Store the target record first
        let target_record = MemoryRecord {
            id: "lsh-threshold-target".to_string(),
            encoding: target_hv,
            timestamp_ms: 9_999_999,
            memory_type: MemoryType::Episodic,
            content: "Target for LSH threshold test".to_string(),
            valence: 0.5,
            arousal: 0.5,
            phi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(target_record).await.unwrap();

        // Below threshold: brute-force should find the target
        let results_below = db.search_similar(&target_hv, 1).await.unwrap();
        assert_eq!(results_below.len(), 1);
        assert_eq!(results_below[0].record.id, "lsh-threshold-target");
        assert!(results_below[0].similarity > 0.99);

        // Fill to LSH_MIN_RECORDS (500) total. We already have 1.
        for i in 0..(LSH_MIN_RECORDS - 1) {
            db.store(make_record(
                &format!("lsh-fill-{}", i),
                i as u64 + 1000,
                MemoryType::Working,
            ))
            .await
            .unwrap();
        }
        assert_eq!(db.count().await.unwrap(), LSH_MIN_RECORDS);

        // Above threshold: LSH-accelerated path should still find the target
        let results_above = db.search_similar(&target_hv, 1).await.unwrap();
        assert!(
            !results_above.is_empty(),
            "LSH path should find at least one result"
        );
        assert_eq!(
            results_above[0].record.id, "lsh-threshold-target",
            "LSH path should find the exact-match target as top result"
        );
        assert!(results_above[0].similarity > 0.99);
    }

    #[tokio::test]
    async fn test_database_lsh_invalidated_on_write() {
        // Verify that storing a new record updates the LSH index,
        // allowing subsequent searches to find the newly-stored record.
        let db = SqliteMemory::in_memory().unwrap();

        // Fill to LSH threshold so LSH path is active
        for i in 0..LSH_MIN_RECORDS {
            db.store(make_record(
                &format!("lsh-inv-{}", i),
                i as u64 + 2000,
                MemoryType::Episodic,
            ))
            .await
            .unwrap();
        }

        // Now store a new record with a distinct vector AFTER the index was populated
        let new_hv = BinaryHV::random(7777);
        let new_record = MemoryRecord {
            id: "lsh-inv-new".to_string(),
            encoding: new_hv,
            timestamp_ms: 99_999_999,
            memory_type: MemoryType::Episodic,
            content: "Newly written after LSH populated".to_string(),
            valence: 0.5,
            arousal: 0.5,
            phi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(new_record).await.unwrap();

        // Verify the new record's LSH entries exist in the index
        let conn = db.conn.lock().unwrap();
        let lsh_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM vector_lsh WHERE memory_id = 'lsh-inv-new'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            lsh_count, LSH_NUM_BANDS as i64,
            "New record should have LSH entries for all bands"
        );
        drop(conn);

        // Search should find it
        let results = db.search_similar(&new_hv, 1).await.unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].record.id, "lsh-inv-new");
        assert!(results[0].similarity > 0.99);
    }

    #[tokio::test]
    async fn test_database_reconsolidation_bumps_retrieval_count() {
        let db = SqliteMemory::in_memory().unwrap();

        let hv = BinaryHV::random(42);
        let record = MemoryRecord {
            id: "recon-1".to_string(),
            encoding: hv,
            timestamp_ms: 1_000_000,
            memory_type: MemoryType::Episodic,
            content: "Reconsolidation test".to_string(),
            valence: 0.5,
            arousal: 0.5,
            phi: 0.7,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        };
        db.store(record).await.unwrap();

        // Verify initial state
        let before = db.get("recon-1").await.unwrap().unwrap();
        assert_eq!(before.retrieval_count, 0);
        assert!((before.consolidation_strength - 0.0).abs() < 1e-6);

        // First search triggers reconsolidation
        let results = db.search_similar(&hv, 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.id, "recon-1");

        // Read the record again to see updated retrieval stats
        let after_first = db.get("recon-1").await.unwrap().unwrap();
        assert_eq!(
            after_first.retrieval_count, 1,
            "retrieval_count should increment by 1 after search"
        );
        assert!(
            after_first.consolidation_strength > 0.0,
            "consolidation_strength should increase after search"
        );
        assert!(
            (after_first.consolidation_strength - 0.05).abs() < 0.001,
            "First retrieval should add 0.05 consolidation strength, got {}",
            after_first.consolidation_strength
        );

        // Second search should bump again
        let _ = db.search_similar(&hv, 1).await.unwrap();
        let after_second = db.get("recon-1").await.unwrap().unwrap();
        assert_eq!(after_second.retrieval_count, 2);
        assert!(
            (after_second.consolidation_strength - 0.10).abs() < 0.001,
            "Second retrieval should bring consolidation_strength to 0.10, got {}",
            after_second.consolidation_strength
        );
    }

    #[tokio::test]
    async fn test_database_reconsolidation_capped_at_one() {
        let db = SqliteMemory::in_memory().unwrap();

        let hv = BinaryHV::random(99);
        let record = MemoryRecord {
            id: "recon-cap".to_string(),
            encoding: hv,
            timestamp_ms: 1_000_000,
            memory_type: MemoryType::Semantic,
            content: "Cap test".to_string(),
            valence: 0.5,
            arousal: 0.5,
            phi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.95, // Start near cap
            retrieval_count: 19,
        };
        db.store(record).await.unwrap();

        // Search should bump consolidation_strength but cap at 1.0
        let _ = db.search_similar(&hv, 1).await.unwrap();
        let after = db.get("recon-cap").await.unwrap().unwrap();
        assert_eq!(after.retrieval_count, 20);
        assert!(
            after.consolidation_strength <= 1.0,
            "consolidation_strength should be capped at 1.0, got {}",
            after.consolidation_strength
        );
        assert!(
            (after.consolidation_strength - 1.0).abs() < 0.001,
            "0.95 + 0.05 should clamp to 1.0, got {}",
            after.consolidation_strength
        );
    }

    #[tokio::test]
    async fn test_database_reconsolidation_only_affects_returned_results() {
        let db = SqliteMemory::in_memory().unwrap();

        let hv_target = BinaryHV::random(42);
        let hv_other = BinaryHV::random(9999); // Unrelated vector

        db.store(MemoryRecord {
            id: "recon-target".to_string(),
            encoding: hv_target,
            timestamp_ms: 1_000_000,
            memory_type: MemoryType::Episodic,
            content: "Target".to_string(),
            valence: 0.5,
            arousal: 0.5,
            phi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        })
        .await
        .unwrap();

        db.store(MemoryRecord {
            id: "recon-bystander".to_string(),
            encoding: hv_other,
            timestamp_ms: 1_000_001,
            memory_type: MemoryType::Episodic,
            content: "Bystander".to_string(),
            valence: 0.5,
            arousal: 0.5,
            phi: 0.5,
            topics: vec![],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        })
        .await
        .unwrap();

        // Search for top_k=1 with target vector — only target should be reconsolidated
        let results = db.search_similar(&hv_target, 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].record.id, "recon-target");

        // Target should be bumped
        let target_after = db.get("recon-target").await.unwrap().unwrap();
        assert_eq!(target_after.retrieval_count, 1);

        // Bystander should NOT be bumped (it was not in the top_k results)
        let bystander_after = db.get("recon-bystander").await.unwrap().unwrap();
        assert_eq!(
            bystander_after.retrieval_count, 0,
            "Bystander record should not be reconsolidated"
        );
    }

    #[tokio::test]
    async fn test_database_eviction_pipeline_end_to_end() {
        // Simulate the eviction pipeline from symthaea.rs:
        // 1. Working memory items are "evicted" as ContinuousHV vectors
        // 2. They get queued as GraduationEvents in MemoryCoordinator
        // 3. MemoryCoordinator processes graduations into EpisodicMemory
        // 4. The evicted items are also persisted to the database as Working type records
        // 5. Subsequent searches find the persisted items and trigger reconsolidation

        use crate::memory::episodic_replay::{EpisodicMemory, EpisodicReplayConfig};
        use crate::memory::memory_coordinator::{
            CoordinatorConfig, GraduationEvent, MemoryCoordinator,
        };
        use symthaea_core::hdc::unified_hv::ContinuousHV;

        let db = SqliteMemory::in_memory().unwrap();

        // Step 1-2: Simulate evicted items becoming graduation events
        let mut coordinator = MemoryCoordinator::new(CoordinatorConfig {
            min_wm_steps_for_graduation: 3,
            ..CoordinatorConfig::default()
        });
        coordinator.update_signals(0.7, 0.8);

        let evicted_hvs: Vec<ContinuousHV> =
            (0..4).map(|i| ContinuousHV::random(1024, i + 10)).collect();

        for (i, hv) in evicted_hvs.iter().enumerate() {
            coordinator.queue_graduation(GraduationEvent {
                content: hv.clone(),
                label: format!("wm_eviction_step_{}", i),
                steps_survived: if i < 2 { 1 } else { 5 }, // First two are too short
                final_activation: 0.4,
                phi_at_graduation: 0.7,
                coherence_at_graduation: 0.8,
            });
        }

        // Step 3: Process graduations into episodic memory
        let episodic_config = EpisodicReplayConfig {
            phi_threshold: 0.1,
            ..Default::default()
        };
        let mut episodic = EpisodicMemory::new(episodic_config);
        let graduated = coordinator.process_graduations(&mut episodic);

        assert_eq!(
            coordinator.stats.graduations_rejected, 2,
            "Two items should be rejected (too few steps)"
        );
        assert_eq!(
            coordinator.stats.graduations_processed, 2,
            "Two items should be processed"
        );
        assert_eq!(
            graduated, 2,
            "Two items should be stored in episodic memory"
        );

        // Step 4: Persist all evicted items to the database (mimicking symthaea.rs tick())
        let timestamp_ms = 1_700_000_000_000u64;
        for (i, hv) in evicted_hvs.iter().enumerate() {
            let record = MemoryRecord {
                id: format!("wm-{}-{}", timestamp_ms, i),
                memory_type: MemoryType::Working,
                encoding: hv.to_binary(0.0),
                content: format!("Working memory eviction at step {}", i),
                timestamp_ms,
                valence: 0.0,
                arousal: 0.0,
                phi: 0.7,
                topics: vec![],
                metadata: "{}".to_string(),
                consolidation_strength: 0.0,
                retrieval_count: 0,
            };
            db.store(record).await.unwrap();
        }
        assert_eq!(db.count().await.unwrap(), 4);

        // Step 5: Search for one of the evicted items by its encoding
        let query_hv = evicted_hvs[2].to_binary(0.0);
        let results = db.search_similar(&query_hv, 2).await.unwrap();
        assert!(!results.is_empty(), "Should find persisted evicted items");

        // The matching record should have its retrieval_count bumped
        let top_id = &results[0].record.id;
        let bumped = db.get(top_id).await.unwrap().unwrap();
        assert_eq!(
            bumped.retrieval_count, 1,
            "Retrieval count should be bumped after search"
        );
        assert!(
            bumped.consolidation_strength > 0.0,
            "Consolidation strength should increase"
        );
    }

    #[tokio::test]
    async fn test_database_all_memory_types_stored_and_filtered() {
        let db = SqliteMemory::in_memory().unwrap();

        let types = [
            ("ep-1", MemoryType::Episodic),
            ("sem-1", MemoryType::Semantic),
            ("proc-1", MemoryType::Procedural),
            ("wk-1", MemoryType::Working),
        ];

        for (id, mt) in &types {
            db.store(make_record(id, 100, *mt)).await.unwrap();
        }

        assert_eq!(db.count().await.unwrap(), 4);

        // Verify each type round-trips correctly
        for (id, expected_type) in &types {
            let record = db.get(id).await.unwrap().unwrap();
            assert_eq!(
                record.memory_type, *expected_type,
                "Memory type should round-trip for {}",
                id
            );
        }

        // Stats should show the distribution
        let stats = db.stats().await.unwrap();
        assert_eq!(stats.total_records, 4);
        assert_eq!(stats.memory_type_counts.len(), 4);
    }

    #[tokio::test]
    async fn test_database_health_check() {
        let db = SqliteMemory::in_memory().unwrap();
        assert!(db.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn test_database_delete_nonexistent_returns_false() {
        let db = SqliteMemory::in_memory().unwrap();
        let deleted = db.delete("ghost-id").await.unwrap();
        assert!(
            !deleted,
            "Deleting a nonexistent record should return false"
        );
    }

    #[tokio::test]
    async fn test_sqlite_stats() {
        let db = SqliteMemory::in_memory().unwrap();

        // Initial stats should show empty database
        let stats = db.stats().await.unwrap();
        assert_eq!(stats.total_records, 0);
        assert_eq!(stats.backend_status, "in_memory");
        assert!(stats.page_size > 0);

        // Store some memories
        for i in 0..5 {
            let record = MemoryRecord {
                id: format!("stats-test-{}", i),
                encoding: BinaryHV::random(i as u64),
                timestamp_ms: 1000000000 + i as u64 * 1000,
                memory_type: if i % 2 == 0 {
                    MemoryType::Episodic
                } else {
                    MemoryType::Semantic
                },
                content: format!("Test memory {}", i),
                valence: 0.5,
                arousal: 0.5,
                phi: 0.5 + i as f64 * 0.1,
                topics: vec!["test".to_string()],
                metadata: "{}".to_string(),
                consolidation_strength: 0.0,
                retrieval_count: 0,
            };
            db.store(record).await.unwrap();
        }

        // Stats should reflect stored data
        let stats = db.stats().await.unwrap();
        assert_eq!(stats.total_records, 5);
        assert!(stats.database_size_bytes > 0);

        // Check memory type distribution
        assert!(!stats.memory_type_counts.is_empty());
        let total_type_count: usize = stats.memory_type_counts.iter().map(|(_, c)| c).sum();
        assert_eq!(total_type_count, 5);

        // Check phi average (0.5 + 0.6 + 0.7 + 0.8 + 0.9) / 5 = 0.7
        assert!((stats.avg_phi - 0.7).abs() < 0.01);

        // Check timestamp range
        assert_eq!(stats.oldest_timestamp_ms, 1000000000);
        assert_eq!(stats.newest_timestamp_ms, 1000004000);
    }
}
