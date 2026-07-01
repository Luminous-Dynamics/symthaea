// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Database Module - Persistent Memory Storage for Consciousness Systems
//!
//! This module provides the persistence layer for Symthaea's consciousness systems,
//! enabling memories to survive across restarts and supporting similarity-based
//! retrieval using HDC encodings.
//!
//! # Overview
//!
//! The database layer abstracts storage backends behind the [`ConsciousnessDatabase`]
//! trait, allowing different implementations (SQLite, in-memory, etc.) while providing
//! a unified API for consciousness-aware memory operations.
//!
//! Key features:
//! - **HDC-native storage**: Memories stored with their hypervector encodings
//! - **Similarity search**: Find related memories using HDC similarity
//! - **Emotional tagging**: Valence/arousal dimensions for affective memory
//! - **Psi tracking**: Store consciousness estimates with memories
//!
//! # Quick Start
//!
//! ```rust,ignore
//! use symthaea::databases::{SqliteMemory, ConsciousnessDatabase, MemoryRecord, MemoryType};
//! use symthaea::hdc::BinaryHV;
//!
//! // Create an in-memory database (for testing)
//! let db = SqliteMemory::in_memory()?;
//!
//! // Store a memory
//! let record = MemoryRecord {
//!     id: "greeting-1".to_string(),
//!     memory_type: MemoryType::Episodic,
//!     encoding: BinaryHV::random(42),
//!     content: "User said hello for the first time".to_string(),
//!     timestamp_ms: 1704067200000,
//!     valence: 0.8,   // Positive emotion
//!     arousal: 0.5,   // Moderate arousal
//!     psi: 0.65,      // Consciousness estimate at encoding
//!     topics: vec!["greeting".to_string(), "first-contact".to_string()],
//!     metadata: "{}".to_string(),
//!     consolidation_strength: 0.0,
//!     retrieval_count: 0,
//! };
//! db.store(record).await?;
//!
//! // Search for similar memories
//! let query = BinaryHV::random(42);  // Same encoding = identical match
//! let results = db.search_similar(&query, 5).await?;
//! assert!(results[0].similarity > 0.99);
//! ```
//!
//! # Memory Types
//!
//! The system supports four memory types, inspired by cognitive science:
//!
//! | Type | Description | Example |
//! |------|-------------|---------|
//! | [`MemoryType::Episodic`] | Autobiographical events | "User asked about weather on 2024-01-01" |
//! | [`MemoryType::Semantic`] | General knowledge/facts | "Paris is the capital of France" |
//! | [`MemoryType::Procedural`] | Skills and how-to | "Steps to generate a report" |
//! | [`MemoryType::Working`] | Short-term active info | "Current conversation context" |
//!
//! # Available Backends
//!
//! - [`SqliteMemory`] - SQLite-backed persistent storage (recommended for production)
//!
//! # See Also
//!
//! - [`crate::memory`] - Higher-level memory systems (hippocampal, working memory)
//! - [`crate::hdc::BinaryHV`] - Binary hypervector type used for encodings

use async_trait::async_trait;
use symthaea_core::hdc::binary_hv::BinaryHV;

pub mod sqlite_client;
pub mod storage_runtime;

#[cfg(feature = "lancedb-backend")]
pub mod lance_client;

#[cfg(feature = "hdc-store")]
pub mod hdc_store_client;

// ============================================================================
// Core Types
// ============================================================================

/// Result type for database operations.
///
/// All database methods return this type, wrapping either the success value
/// or a [`DatabaseError`] describing what went wrong.
pub type DbResult<T> = Result<T, DatabaseError>;

/// Error types for database operations.
///
/// Provides structured error information for debugging and error handling.
/// All variants include a descriptive message string.
///
/// # Example
///
/// ```rust,ignore
/// use symthaea::databases::{SqliteMemory, DatabaseError};
///
/// let result = SqliteMemory::new("/nonexistent/path/db.sqlite");
/// match result {
///     Err(DatabaseError::ConnectionFailed(msg)) => {
///         eprintln!("Could not connect: {}", msg);
///     }
///     _ => {}
/// }
/// ```
#[derive(Debug, Clone)]
pub enum DatabaseError {
    /// Failed to establish database connection.
    ///
    /// Common causes: file permissions, missing directories, invalid path.
    ConnectionFailed(String),

    /// Query execution failed.
    ///
    /// Common causes: SQL syntax error, schema mismatch, constraint violation.
    QueryFailed(String),

    /// Insert/update operation failed.
    ///
    /// Common causes: unique constraint violation, foreign key error.
    InsertFailed(String),

    /// Requested record was not found.
    ///
    /// This is not necessarily an error - use `Option<T>` returns for expected misses.
    NotFound(String),

    /// Catch-all for other database errors.
    Other(String),
}

impl std::fmt::Display for DatabaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed(msg) => write!(f, "Connection failed: {msg}"),
            Self::QueryFailed(msg) => write!(f, "Query failed: {msg}"),
            Self::InsertFailed(msg) => write!(f, "Insert failed: {msg}"),
            Self::NotFound(msg) => write!(f, "Not found: {msg}"),
            Self::Other(msg) => write!(f, "Database error: {msg}"),
        }
    }
}

impl std::error::Error for DatabaseError {}

/// Types of memory in the consciousness system.
///
/// Based on cognitive science models of human memory, this enum categorizes
/// memories by their functional role and temporal characteristics.
///
/// # Memory Type Characteristics
///
/// | Type | Duration | Content | Example |
/// |------|----------|---------|---------|
/// | Episodic | Long-term | Events with context | "Met user on January 1st" |
/// | Semantic | Long-term | Facts without context | "Python is a programming language" |
/// | Procedural | Long-term | How-to knowledge | "Steps to debug code" |
/// | Working | Short-term | Active processing | "Current conversation topic" |
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MemoryType {
    /// Autobiographical memories of specific events.
    ///
    /// Episodic memories encode "what happened, when, and where" with rich
    /// contextual detail. They are personally experienced and time-stamped.
    Episodic,

    /// General knowledge and facts.
    ///
    /// Semantic memories represent factual knowledge without reference to
    /// when or how the information was learned. Context-free truths.
    Semantic,

    /// Skills and how-to knowledge.
    ///
    /// Procedural memories encode sequences of actions and motor skills.
    /// Often implicit - knowing how to do something without explicit recall.
    Procedural,

    /// Short-term active information.
    ///
    /// Working memory holds information currently being processed or needed
    /// for ongoing tasks. Limited capacity, temporary storage.
    Working,
}

/// A memory record stored in the database.
///
/// This is the primary data structure for persistent memory storage in Symthaea.
/// Each memory combines:
/// - **Semantic content**: The actual information being remembered
/// - **HDC encoding**: A hypervector for similarity-based retrieval
/// - **Emotional tags**: Valence and arousal for affective context
/// - **Metadata**: Timestamps, topics, and extensible JSON metadata
///
/// # Creating Memories
///
/// ```rust,ignore
/// use symthaea::databases::{MemoryRecord, MemoryType};
/// use symthaea::hdc::BinaryHV;
///
/// let memory = MemoryRecord {
///     id: uuid::Uuid::new_v4().to_string(),
///     memory_type: MemoryType::Episodic,
///     encoding: BinaryHV::random(42),  // Should be semantic encoding of content
///     content: "First successful phi measurement reached 0.85".to_string(),
///     timestamp_ms: std::time::SystemTime::now()
///         .duration_since(std::time::UNIX_EPOCH)
///         .unwrap()
///         .as_millis() as u64,
///     valence: 0.9,   // Very positive
///     arousal: 0.7,   // Excited
///     psi: 0.85,      // Consciousness estimate at the moment
///     topics: vec!["milestone".to_string(), "phi".to_string()],
///     metadata: r#"{"session_id": "abc123"}"#.to_string(),
///     consolidation_strength: 0.0,
///     retrieval_count: 0,
/// };
/// ```
#[derive(Debug, Clone)]
pub struct MemoryRecord {
    /// Unique identifier for this memory (typically a UUID).
    pub id: String,

    /// Cognitive type of this memory (episodic, semantic, etc.).
    pub memory_type: MemoryType,

    /// HDC hypervector encoding of the memory content.
    ///
    /// This 16,384-bit binary hypervector enables fast similarity search.
    /// Should be generated by encoding the semantic content of the memory.
    pub encoding: BinaryHV,

    /// Human-readable textual content or description.
    pub content: String,

    /// When the memory was created (Unix timestamp in milliseconds).
    pub timestamp_ms: u64,

    /// Emotional valence: -1.0 (very negative) to 1.0 (very positive).
    ///
    /// Represents the pleasure/displeasure dimension of emotion.
    pub valence: f32,

    /// Emotional arousal: 0.0 (calm) to 1.0 (excited/activated).
    ///
    /// Represents the activation/deactivation dimension of emotion.
    pub arousal: f32,

    /// Psi — Consciousness estimate at encoding time.
    ///
    /// Higher psi indicates the memory was formed during a state of
    /// high consciousness/integration.
    pub psi: f64,

    /// Topic tags for categorical organization.
    pub topics: Vec<String>,

    /// Additional metadata as a JSON string.
    pub metadata: String,

    /// Consolidation strength from reconsolidation (0.0 = fragile, 1.0+ = strong).
    ///
    /// Increases each time the memory is retrieved and reconsolidated,
    /// modelling the biological process of memory strengthening through recall.
    pub consolidation_strength: f64,

    /// Number of times this memory has been retrieved/replayed.
    ///
    /// Each retrieval triggers reconsolidation, strengthening the memory trace.
    pub retrieval_count: u32,
}

/// Search result containing a memory record and its similarity score.
///
/// Returned by [`ConsciousnessDatabase::search_similar`] in descending
/// order of similarity (most similar first).
///
/// # Interpreting Similarity
///
/// The similarity score uses HDC Hamming similarity:
/// - **1.0**: Identical encodings (exact match)
/// - **0.7-0.9**: Highly related content
/// - **0.5-0.7**: Moderately related
/// - **~0.5**: Random/unrelated (expected for unrelated content)
/// - **<0.5**: Negatively correlated (rare)
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// The matching memory record.
    pub record: MemoryRecord,

    /// Similarity score between query and this record's encoding (0.0 to 1.0).
    pub similarity: f32,
}

/// Statistics about the database state and health.
///
/// Provides observability into the database for monitoring and debugging.
/// Values are backend-specific - some fields may be zero if not applicable.
///
/// # Example
///
/// ```rust,ignore
/// use symthaea::databases::{SqliteMemory, ConsciousnessDatabase};
///
/// let db = SqliteMemory::in_memory()?;
/// let stats = db.stats().await?;
///
/// println!("Total records: {}", stats.total_records);
/// println!("Database size: {} bytes", stats.database_size_bytes);
/// println!("Cache hit ratio: {:.1}%", stats.cache_hit_ratio * 100.0);
/// ```
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DatabaseStats {
    /// Total number of memory records stored.
    pub total_records: usize,

    /// Size of the database on disk (bytes). 0 for in-memory databases.
    pub database_size_bytes: u64,

    /// Number of database pages (for page-based backends like SQLite).
    pub page_count: u64,

    /// Size of each database page (bytes).
    pub page_size: u64,

    /// Amount of free space in the database file (bytes).
    pub freelist_count: u64,

    /// Cache hit ratio (0.0 to 1.0). Backend-specific.
    pub cache_hit_ratio: f64,

    /// Number of cache hits since startup.
    pub cache_hits: u64,

    /// Number of cache misses since startup.
    pub cache_misses: u64,

    /// Average query latency in microseconds. 0 if not measured.
    pub avg_query_latency_us: u64,

    /// Total number of queries executed.
    pub total_queries: u64,

    /// Memory type distribution: (type_name, count).
    pub memory_type_counts: Vec<(String, usize)>,

    /// Average psi (consciousness estimate) across all memories.
    pub avg_psi: f64,

    /// Oldest memory timestamp (milliseconds since epoch).
    pub oldest_timestamp_ms: u64,

    /// Newest memory timestamp (milliseconds since epoch).
    pub newest_timestamp_ms: u64,

    /// Backend-specific status string (e.g., "wal", "journal", "in_memory").
    pub backend_status: String,
}

// ============================================================================
// Storage Traits
// ============================================================================

#[cfg(feature = "school_learning")]
use crate::school::curriculum::Curriculum;
use symthaea_dream::CausalLink;

/// Durable memory-record operations.
///
/// Use this trait when code needs ordinary record persistence but does not need
/// vector search, causal links, curricula, or backend-specific health details.
#[async_trait]
pub trait MemoryRecordStore: Send + Sync {
    /// Store a memory record, replacing any existing record with the same ID.
    async fn store_memory(&self, record: MemoryRecord) -> DbResult<()>;

    /// Store multiple memory records, replacing any existing records with the same IDs.
    async fn store_memory_batch(&self, records: Vec<MemoryRecord>) -> DbResult<usize>;

    /// Retrieve a specific memory by its unique ID.
    async fn get_memory(&self, id: &str) -> DbResult<Option<MemoryRecord>>;

    /// Delete a memory by its unique ID.
    async fn delete_memory(&self, id: &str) -> DbResult<bool>;

    /// Count total number of memories in the store.
    async fn memory_count(&self) -> DbResult<usize>;

    /// List all records in the store for migration/export.
    async fn list_memories(&self) -> DbResult<Vec<MemoryRecord>>;
}

/// Binary hypervector similarity search.
///
/// Use this for HDC/vector retrieval without depending on metadata ownership or
/// causal-event storage. Backends may implement this via brute force, LSH,
/// LanceDB, mmap indexes, or a remote vector service.
#[async_trait]
pub trait BinaryHvSearchBackend: Send + Sync {
    /// Search for memories similar to the query hypervector.
    async fn search_binary_hv(&self, query: &BinaryHV, top_k: usize)
    -> DbResult<Vec<SearchResult>>;

    /// Search with an optional backend-specific predicate.
    async fn search_binary_hv_filtered(
        &self,
        query: &BinaryHV,
        top_k: usize,
        filter: Option<&str>,
    ) -> DbResult<Vec<SearchResult>>;
}

/// Storage health and observability.
#[async_trait]
pub trait StorageHealthBackend: Send + Sync {
    /// Check whether the backend is healthy.
    async fn storage_health_check(&self) -> DbResult<bool>;

    /// Return backend statistics for monitoring and debugging.
    async fn storage_stats(&self) -> DbResult<DatabaseStats>;
}

/// Durable causal-event storage.
///
/// Kept separate from memory search so vector stores are not forced to pretend
/// they are event logs.
#[async_trait]
pub trait CausalEventStore: Send + Sync {
    /// Store causal links into the backend.
    async fn store_causal_events(&self, links: &[CausalLink]) -> DbResult<()>;

    /// Retrieve all causal links from the backend.
    async fn get_causal_events(&self) -> DbResult<Vec<CausalLink>>;
}

#[async_trait]
impl<T> MemoryRecordStore for T
where
    T: ConsciousnessDatabase + ?Sized,
{
    async fn store_memory(&self, record: MemoryRecord) -> DbResult<()> {
        ConsciousnessDatabase::store(self, record).await
    }

    async fn store_memory_batch(&self, records: Vec<MemoryRecord>) -> DbResult<usize> {
        ConsciousnessDatabase::store_batch(self, records).await
    }

    async fn get_memory(&self, id: &str) -> DbResult<Option<MemoryRecord>> {
        ConsciousnessDatabase::get(self, id).await
    }

    async fn delete_memory(&self, id: &str) -> DbResult<bool> {
        ConsciousnessDatabase::delete(self, id).await
    }

    async fn memory_count(&self) -> DbResult<usize> {
        ConsciousnessDatabase::count(self).await
    }

    async fn list_memories(&self) -> DbResult<Vec<MemoryRecord>> {
        ConsciousnessDatabase::list_all(self).await
    }
}

#[async_trait]
impl<T> BinaryHvSearchBackend for T
where
    T: ConsciousnessDatabase + ?Sized,
{
    async fn search_binary_hv(
        &self,
        query: &BinaryHV,
        top_k: usize,
    ) -> DbResult<Vec<SearchResult>> {
        ConsciousnessDatabase::search_similar(self, query, top_k).await
    }

    async fn search_binary_hv_filtered(
        &self,
        query: &BinaryHV,
        top_k: usize,
        filter: Option<&str>,
    ) -> DbResult<Vec<SearchResult>> {
        ConsciousnessDatabase::search_similar_filtered(self, query, top_k, filter).await
    }
}

#[async_trait]
impl<T> StorageHealthBackend for T
where
    T: ConsciousnessDatabase + ?Sized,
{
    async fn storage_health_check(&self) -> DbResult<bool> {
        ConsciousnessDatabase::health_check(self).await
    }

    async fn storage_stats(&self) -> DbResult<DatabaseStats> {
        ConsciousnessDatabase::stats(self).await
    }
}

#[async_trait]
impl<T> CausalEventStore for T
where
    T: ConsciousnessDatabase + ?Sized,
{
    async fn store_causal_events(&self, links: &[CausalLink]) -> DbResult<()> {
        ConsciousnessDatabase::store_causal_links(self, links).await
    }

    async fn get_causal_events(&self) -> DbResult<Vec<CausalLink>> {
        ConsciousnessDatabase::get_causal_links(self).await
    }
}

// ============================================================================
// Compatibility Database Facade
// ============================================================================

/// Trait for consciousness database backends.
///
/// Defines the interface that all database implementations must provide.
/// This enables swapping backends (SQLite, PostgreSQL, in-memory, etc.)
/// without changing application code.
///
/// All methods are async to support both synchronous and asynchronous backends.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to support concurrent access.
#[async_trait]
pub trait ConsciousnessDatabase: Send + Sync {
    /// Store a memory record, replacing any existing record with the same ID.
    async fn store(&self, record: MemoryRecord) -> DbResult<()>;

    /// Store multiple memory records, replacing any existing records with the same IDs.
    async fn store_batch(&self, records: Vec<MemoryRecord>) -> DbResult<usize> {
        let count = records.len();
        for record in records {
            self.store(record).await?;
        }
        Ok(count)
    }

    /// Search for memories similar to the query hypervector.
    ///
    /// Returns up to `top_k` results sorted by similarity (descending).
    async fn search_similar(&self, query: &BinaryHV, top_k: usize) -> DbResult<Vec<SearchResult>>;

    /// Retrieve a specific memory by its unique ID.
    ///
    /// Returns `Ok(None)` if no memory with this ID exists.
    async fn get(&self, id: &str) -> DbResult<Option<MemoryRecord>>;

    /// Delete a memory by its unique ID.
    ///
    /// Returns `Ok(true)` if deleted, `Ok(false)` if not found.
    async fn delete(&self, id: &str) -> DbResult<bool>;

    /// Count total number of memories in the database.
    async fn count(&self) -> DbResult<usize>;

    /// Check if the database connection is healthy.
    async fn health_check(&self) -> DbResult<bool>;

    /// Get statistics about the database for observability.
    ///
    /// Returns detailed metrics about database size, performance, and content
    /// distribution. Useful for monitoring and debugging.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let stats = db.stats().await?;
    /// println!("Records: {}, Size: {} bytes", stats.total_records, stats.database_size_bytes);
    /// ```
    async fn stats(&self) -> DbResult<DatabaseStats>;

    /// Search with optional filter predicate (e.g., memory_type, timestamp range).
    ///
    /// The `filter` string is backend-specific:
    /// - **LanceDB**: Lance SQL expression applied before loading (predicate pushdown)
    /// - **SQLite**: Ignored (falls back to unfiltered search)
    ///
    /// Default implementation ignores the filter and delegates to [`search_similar`].
    async fn search_similar_filtered(
        &self,
        query: &BinaryHV,
        top_k: usize,
        _filter: Option<&str>,
    ) -> DbResult<Vec<SearchResult>> {
        self.search_similar(query, top_k).await
    }

    /// List all records in the database (for migration/export).
    ///
    /// Default implementation returns an empty vec. Backends should override
    /// this to provide a full scan without similarity computation.
    async fn list_all(&self) -> DbResult<Vec<MemoryRecord>> {
        Ok(Vec::new())
    }

    /// Store a curriculum into the database.
    #[cfg(feature = "school_learning")]
    async fn store_curriculum(&self, _curriculum: &Curriculum) -> DbResult<()> {
        Ok(())
    }

    /// Retrieve all curricula from the database.
    #[cfg(feature = "school_learning")]
    async fn get_curricula(&self) -> DbResult<Vec<Curriculum>> {
        Ok(Vec::new())
    }

    /// Store causal links into the database.
    async fn store_causal_links(&self, _links: &[CausalLink]) -> DbResult<()> {
        Ok(())
    }

    /// Retrieve all causal links from the database.
    async fn get_causal_links(&self) -> DbResult<Vec<CausalLink>> {
        Ok(Vec::new())
    }
}

// ============================================================================
// Runtime Backend Selection
// ============================================================================

/// Which database backend to use for persistent consciousness memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub enum DatabaseBackend {
    /// SQLite — battle-tested, single-file, always available.
    #[default]
    Sqlite,
    /// LanceDB — columnar Arrow storage, async-native.
    /// Requires the `lancedb-backend` feature flag.
    #[cfg(feature = "lancedb-backend")]
    Lance,
    /// HdcStore — zero-copy mmap'd BinaryHV storage with LSH indexing.
    /// Requires the `hdc-store` feature flag.
    #[cfg(feature = "hdc-store")]
    HdcStore,
}

/// Configuration for the consciousness database layer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct DatabaseConfig {
    /// Which backend to use.
    pub backend: DatabaseBackend,
    /// Path for persistent storage. If None, uses in-memory.
    pub path: Option<String>,
}

impl DatabaseConfig {
    /// Validate database configuration parameters.
    ///
    /// Checks that:
    /// - If a path is specified, it is not empty or whitespace-only
    pub fn validate(&self) -> Result<(), String> {
        if let Some(ref path) = self.path {
            if path.trim().is_empty() {
                return Err("DatabaseConfig: path must not be empty or whitespace-only".into());
            }
        }
        Ok(())
    }
}

/// Create a database instance from configuration.
pub async fn create_database(config: &DatabaseConfig) -> DbResult<Box<dyn ConsciousnessDatabase>> {
    match &config.backend {
        DatabaseBackend::Sqlite => {
            let db = match &config.path {
                Some(p) => SqliteMemory::new(p)?,
                None => SqliteMemory::in_memory()?,
            };
            Ok(Box::new(db))
        }
        #[cfg(feature = "lancedb-backend")]
        DatabaseBackend::Lance => {
            let db = match &config.path {
                Some(p) => LanceMemory::new(p).await?,
                None => LanceMemory::in_memory().await?,
            };
            Ok(Box::new(db))
        }
        #[cfg(feature = "hdc-store")]
        DatabaseBackend::HdcStore => {
            let path = config
                .path
                .clone()
                .unwrap_or_else(|| "consciousness.hdc".to_string());
            let db = hdc_store_client::HdcStoreDatabase::new(&path)
                .map_err(|e| DatabaseError::ConnectionFailed(e.to_string()))?;
            Ok(Box::new(db))
        }
    }
}

// Re-exports
pub use sqlite_client::SqliteMemory;

#[cfg(feature = "lancedb-backend")]
pub use lance_client::LanceMemory;

#[cfg(feature = "hdc-store")]
pub use hdc_store_client::HdcStoreDatabase;

#[cfg(test)]
mod tests {
    use super::*;

    fn contract_record(id: &str, seed: u64) -> MemoryRecord {
        MemoryRecord {
            id: id.to_string(),
            memory_type: MemoryType::Episodic,
            encoding: BinaryHV::random(seed),
            content: format!("contract memory {id}"),
            timestamp_ms: 1_700_000_000_000 + seed,
            valence: 0.25,
            arousal: 0.5,
            psi: 0.75,
            topics: vec!["contract".to_string()],
            metadata: "{}".to_string(),
            consolidation_strength: 0.0,
            retrieval_count: 0,
        }
    }

    async fn assert_memory_backend_contract(db: &dyn ConsciousnessDatabase) {
        assert_eq!(db.count().await.unwrap(), 0);
        assert!(db.health_check().await.unwrap());

        db.store(contract_record("exact", 42)).await.unwrap();
        db.store(contract_record("other", 777)).await.unwrap();
        assert_eq!(db.count().await.unwrap(), 2);

        let exact = db.get("exact").await.unwrap().unwrap();
        assert_eq!(exact.content, "contract memory exact");

        let results = db.search_similar(&BinaryHV::random(42), 2).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].record.id, "exact");
        assert!(results[0].similarity >= results[1].similarity);

        let mut replacement = contract_record("exact", 99);
        replacement.content = "replacement".to_string();
        db.store(replacement).await.unwrap();
        assert_eq!(db.count().await.unwrap(), 2);
        let replaced = db.get("exact").await.unwrap().unwrap();
        assert_eq!(replaced.content, "replacement");
        assert_eq!(replaced.encoding.similarity(&BinaryHV::random(99)), 1.0);

        assert!(db.delete("other").await.unwrap());
        assert!(!db.delete("missing").await.unwrap());
        assert_eq!(db.count().await.unwrap(), 1);

        let all = db.list_all().await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, "exact");
    }

    #[test]
    fn test_default_database_config_is_valid() {
        let config = DatabaseConfig::default();
        assert!(
            config.validate().is_ok(),
            "default DatabaseConfig must pass validation"
        );
    }

    #[test]
    fn test_database_config_with_valid_path() {
        let config = DatabaseConfig {
            backend: DatabaseBackend::Sqlite,
            path: Some("/tmp/test.db".into()),
        };
        assert!(config.validate().is_ok(), "valid path should pass");
    }

    #[test]
    fn test_database_config_empty_path_rejected() {
        let config = DatabaseConfig {
            backend: DatabaseBackend::Sqlite,
            path: Some("".into()),
        };
        let err = config.validate().unwrap_err();
        assert!(err.contains("path"), "error should mention path: {err}");
    }

    #[test]
    fn test_database_config_whitespace_path_rejected() {
        let config = DatabaseConfig {
            backend: DatabaseBackend::Sqlite,
            path: Some("   ".into()),
        };
        assert!(
            config.validate().is_err(),
            "whitespace-only path should fail"
        );
    }

    #[test]
    fn test_database_config_none_path_valid() {
        let config = DatabaseConfig {
            backend: DatabaseBackend::Sqlite,
            path: None,
        };
        assert!(
            config.validate().is_ok(),
            "None path (in-memory) should be valid"
        );
    }

    #[tokio::test]
    async fn sqlite_memory_backend_contract() {
        let db = SqliteMemory::in_memory().unwrap();
        assert_memory_backend_contract(&db).await;
    }

    #[tokio::test]
    async fn sqlite_supports_narrow_storage_traits() {
        let db = SqliteMemory::in_memory().unwrap();
        let record = contract_record("narrow", 123);

        MemoryRecordStore::store_memory(&db, record).await.unwrap();
        assert_eq!(MemoryRecordStore::memory_count(&db).await.unwrap(), 1);

        let found = MemoryRecordStore::get_memory(&db, "narrow")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(found.content, "contract memory narrow");

        let results = BinaryHvSearchBackend::search_binary_hv(&db, &BinaryHV::random(123), 1)
            .await
            .unwrap();
        assert_eq!(results[0].record.id, "narrow");

        assert!(
            StorageHealthBackend::storage_health_check(&db)
                .await
                .unwrap()
        );
        assert!(
            MemoryRecordStore::delete_memory(&db, "narrow")
                .await
                .unwrap()
        );
    }

    #[cfg(feature = "hdc-store")]
    #[tokio::test]
    async fn hdc_store_memory_backend_contract() {
        let db = HdcStoreDatabase::in_memory().unwrap();
        assert_memory_backend_contract(&db).await;
    }
}
