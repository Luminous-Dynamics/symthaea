//! Multi-Database Consciousness Architecture
//!
//! Implements the "Mental Roles" architecture from Revolutionary Improvement #30:

#![allow(dead_code, unused_variables)]
//!
//! | Database | Mental Role | Purpose |
//! |----------|-------------|---------|
//! | Qdrant | Sensory Cortex | Ultra-fast vector similarity (<10ms) |
//! | CozoDB | Prefrontal Cortex | Recursive Datalog reasoning |
//! | LanceDB | Long-Term Memory | Multimodal life records |
//! | DuckDB | Epistemic Auditor | Statistical self-analysis |
//!
//! ## Usage
//!
//! Enable database features in Cargo.toml:
//! ```toml
//! symthaea = { features = ["databases"] }  # All databases
//! symthaea = { features = ["qdrant"] }     # Just Qdrant
//! ```
//!
//! ## Design Philosophy
//!
//! Like the biological brain with specialized regions (visual cortex, prefrontal,
//! hippocampus), artificial consciousness needs specialized databases each optimized
//! for different mental roles.

// Database client modules (always available, use fallback when features disabled)
pub mod qdrant_client;
pub mod cozo_client;
pub mod lance_client;
pub mod duck_client;

// Always available: unified interface and mock implementations
pub mod unified_mind;
pub mod mock;

// Re-exports
pub use unified_mind::{UnifiedMind, MindConfig, MindStatus, MindStatistics, MindHealthReport, LTCState, LTCSnapshot};
pub use mock::MockDatabase;
pub use qdrant_client::{QdrantSensory, QdrantConfig};
pub use cozo_client::{CozoPrefrontal, CozoConfig, CausalRelation, HigherOrderThought};
pub use lance_client::{LanceLongTerm, LanceConfig, ConsolidationState};
pub use duck_client::{DuckEpistemic, DuckConfig, ConsciousnessEvent, ConsciousnessSpectrum, AnalyticsResult};

use crate::hdc::HV16;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Result type for database operations
pub type DbResult<T> = Result<T, DatabaseError>;

/// Database errors
#[derive(Debug, Clone, thiserror::Error)]
pub enum DatabaseError {
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Query failed: {0}")]
    QueryFailed(String),

    #[error("Insert failed: {0}")]
    InsertFailed(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Feature not enabled: {0}")]
    FeatureNotEnabled(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Timeout after {0:?}")]
    Timeout(Duration),
}

/// A memory/experience record stored in the databases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    /// Unique identifier
    pub id: String,

    /// Hypervector encoding of the memory
    pub encoding: HV16,

    /// Timestamp (milliseconds since epoch)
    pub timestamp_ms: u64,

    /// Memory type: episodic, semantic, procedural
    pub memory_type: MemoryType,

    /// Content (text, or path to media)
    pub content: String,

    /// Emotional valence (-1.0 to 1.0)
    pub valence: f32,

    /// Arousal level (0.0 to 1.0)
    pub arousal: f32,

    /// Φ (integrated information) at time of encoding
    pub phi: f64,

    /// Topics/tags
    pub topics: Vec<String>,

    /// Metadata (JSON)
    pub metadata: String,
}

/// Types of memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    /// Episodic: specific events and experiences
    Episodic,
    /// Semantic: facts and knowledge
    Semantic,
    /// Procedural: skills and how-to
    Procedural,
    /// Working: temporary, active memory
    Working,
}

/// Search result with similarity score
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub record: MemoryRecord,
    pub similarity: f32,
}

/// Trait for database operations
#[async_trait::async_trait]
pub trait ConsciousnessDatabase: Send + Sync {
    /// Store a memory record
    async fn store(&self, record: MemoryRecord) -> DbResult<()>;

    /// Search by vector similarity
    async fn search_similar(&self, query: &HV16, top_k: usize) -> DbResult<Vec<SearchResult>>;

    /// Get by ID
    async fn get(&self, id: &str) -> DbResult<Option<MemoryRecord>>;

    /// Delete by ID
    async fn delete(&self, id: &str) -> DbResult<bool>;

    /// Count total records
    async fn count(&self) -> DbResult<usize>;

    /// Health check
    async fn health_check(&self) -> DbResult<bool>;
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_record_creation() {
        let record = MemoryRecord {
            id: "test-1".to_string(),
            encoding: HV16::random(42),
            timestamp_ms: 1700000000000,
            memory_type: MemoryType::Episodic,
            content: "Test memory".to_string(),
            valence: 0.5,
            arousal: 0.3,
            phi: 0.65,
            topics: vec!["test".to_string()],
            metadata: "{}".to_string(),
        };

        assert_eq!(record.id, "test-1");
        assert_eq!(record.memory_type, MemoryType::Episodic);
    }

    #[test]
    fn test_memory_types() {
        assert_ne!(MemoryType::Episodic, MemoryType::Semantic);
        assert_ne!(MemoryType::Procedural, MemoryType::Working);
    }
}
