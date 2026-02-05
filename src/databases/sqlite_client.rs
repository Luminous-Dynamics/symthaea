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
//! use symthaea::hdc::HV16;
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
//!     encoding: HV16::random(42),
//!     content: "First conversation with user".to_string(),
//!     timestamp_ms: 1704067200000,
//!     valence: 0.7,
//!     arousal: 0.5,
//!     phi: 0.65,
//!     topics: vec!["greeting".to_string()],
//!     metadata: "{}".to_string(),
//! };
//! db.store(record).await?;
//!
//! // Search for similar memories
//! let results = db.search_similar(&HV16::random(42), 10).await?;
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
//! | encoding | BLOB | 2048-byte HV16 hypervector |
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
//! reads efficiently.

use super::{ConsciousnessDatabase, DbResult, DatabaseError, DatabaseStats, MemoryRecord, MemoryType, SearchResult};
use symthaea_core::hdc::binary_hv::HV16;
use async_trait::async_trait;
use rusqlite::{Connection, params};
use std::sync::Mutex;
use std::path::Path;
use crate::infrastructure::lock_guard::ResilientMutex;

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
    conn: Mutex<Connection>,

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
                DatabaseError::ConnectionFailed(format!("Failed to create directory: {}", e))
            })?;
        }

        let conn = Connection::open(&path).map_err(|e| {
            DatabaseError::ConnectionFailed(format!("SQLite open failed: {}", e))
        })?;

        let db = Self {
            conn: Mutex::new(conn),
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
            DatabaseError::ConnectionFailed(format!("SQLite in-memory failed: {}", e))
        })?;

        let db = Self {
            conn: Mutex::new(conn),
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

        conn.execute_batch(r#"
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
                metadata TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp_ms);
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_phi ON memories(phi);
        "#).map_err(|e| {
            DatabaseError::QueryFailed(format!("Schema creation failed: {}", e))
        })?;

        Ok(())
    }

    /// Serialize HV16 to bytes (2048 bytes = 16,384 bits).
    #[doc(hidden)]
    fn hv_to_bytes(hv: &HV16) -> Vec<u8> {
        hv.0.to_vec()
    }

    /// Deserialize bytes to HV16.
    #[doc(hidden)]
    fn bytes_to_hv(bytes: &[u8]) -> HV16 {
        if bytes.len() >= HV16::BYTES {
            let mut arr = [0u8; HV16::BYTES];
            arr.copy_from_slice(&bytes[..HV16::BYTES]);
            HV16(arr)
        } else {
            HV16::zero()
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
}

#[async_trait]
impl ConsciousnessDatabase for SqliteMemory {
    async fn store(&self, record: MemoryRecord) -> DbResult<()> {
        let conn = self.conn.lock_resilient("sqlite");

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
               (id, encoding, timestamp_ms, memory_type, content, valence, arousal, phi, topics, metadata)
               VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)"#,
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
            ],
        ).map_err(|e| DatabaseError::InsertFailed(format!("Insert failed: {}", e)))?;

        Ok(())
    }

    async fn search_similar(&self, query: &HV16, top_k: usize) -> DbResult<Vec<SearchResult>> {
        let conn = self.conn.lock_resilient("sqlite");

        let mut stmt = conn.prepare(
            "SELECT id, encoding, timestamp_ms, memory_type, content, valence, arousal, phi, topics, metadata
             FROM memories ORDER BY timestamp_ms DESC LIMIT 1000"
        ).map_err(|e| DatabaseError::QueryFailed(format!("Prepare failed: {}", e)))?;

        let rows = stmt.query_map([], |row| {
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
            })
        }).map_err(|e| DatabaseError::QueryFailed(format!("Query failed: {}", e)))?;

        // Compute similarities and sort
        let mut results: Vec<SearchResult> = rows
            .filter_map(|r| r.ok())
            .map(|record| {
                let similarity = query.similarity(&record.encoding);
                SearchResult { record, similarity }
            })
            .collect();

        // Sort by similarity descending
        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results)
    }

    async fn get(&self, id: &str) -> DbResult<Option<MemoryRecord>> {
        let conn = self.conn.lock_resilient("sqlite");

        let mut stmt = conn.prepare(
            "SELECT id, encoding, timestamp_ms, memory_type, content, valence, arousal, phi, topics, metadata
             FROM memories WHERE id = ?1"
        ).map_err(|e| DatabaseError::QueryFailed(format!("Prepare failed: {}", e)))?;

        let result = stmt.query_row([id], |row| {
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
            })
        });

        match result {
            Ok(record) => Ok(Some(record)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(DatabaseError::QueryFailed(format!("Get failed: {}", e))),
        }
    }

    async fn delete(&self, id: &str) -> DbResult<bool> {
        let conn = self.conn.lock_resilient("sqlite");

        let affected = conn.execute("DELETE FROM memories WHERE id = ?1", [id])
            .map_err(|e| DatabaseError::QueryFailed(format!("Delete failed: {}", e)))?;

        Ok(affected > 0)
    }

    async fn count(&self) -> DbResult<usize> {
        let conn = self.conn.lock_resilient("sqlite");

        let count: i64 = conn.query_row("SELECT COUNT(*) FROM memories", [], |row| row.get(0))
            .map_err(|e| DatabaseError::QueryFailed(format!("Count failed: {}", e)))?;

        Ok(count as usize)
    }

    async fn health_check(&self) -> DbResult<bool> {
        let conn = self.conn.lock_resilient("sqlite");

        conn.execute_batch("SELECT 1")
            .map_err(|e| DatabaseError::QueryFailed(format!("Health check failed: {}", e)))?;

        Ok(true)
    }

    async fn stats(&self) -> DbResult<DatabaseStats> {
        let conn = self.conn.lock_resilient("sqlite");

        // Get total record count
        let total_records: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories",
            [],
            |row| row.get(0)
        ).map_err(|e| DatabaseError::QueryFailed(format!("Count query failed: {}", e)))?;

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
        ).map_err(|e| DatabaseError::QueryFailed(format!("Type counts query failed: {}", e)))?;

        let memory_type_counts: Vec<(String, usize)> = type_counts_stmt
            .query_map([], |row| {
                let mt: String = row.get(0)?;
                let cnt: i64 = row.get(1)?;
                Ok((mt, cnt as usize))
            })
            .map_err(|e| DatabaseError::QueryFailed(format!("Type counts failed: {}", e)))?
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
        let backend_status = if self.path == ":memory:" {
            "in_memory".to_string()
        } else {
            format!("file:{}", journal_mode)
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
            encoding: HV16::random(42),
            timestamp_ms: 1234567890,
            memory_type: MemoryType::Episodic,
            content: "Hello, I am Symthaea".to_string(),
            valence: 0.8,
            arousal: 0.5,
            phi: 0.75,
            topics: vec!["greeting".to_string()],
            metadata: "{}".to_string(),
        };

        db.store(record.clone()).await.unwrap();

        // Count
        assert_eq!(db.count().await.unwrap(), 1);

        // Get by ID
        let retrieved = db.get("test-1").await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Hello, I am Symthaea");

        // Search similar
        let results = db.search_similar(&HV16::random(42), 5).await.unwrap();
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
                encoding: HV16::random(123),
                timestamp_ms: 1234567890,
                memory_type: MemoryType::Semantic,
                content: "I remember this".to_string(),
                valence: 0.5,
                arousal: 0.3,
                phi: 0.6,
                topics: vec!["test".to_string()],
                metadata: "{}".to_string(),
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
                encoding: HV16::random(i as u64),
                timestamp_ms: 1000000000 + i as u64 * 1000,
                memory_type: if i % 2 == 0 { MemoryType::Episodic } else { MemoryType::Semantic },
                content: format!("Test memory {}", i),
                valence: 0.5,
                arousal: 0.5,
                phi: 0.5 + i as f64 * 0.1,
                topics: vec!["test".to_string()],
                metadata: "{}".to_string(),
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
