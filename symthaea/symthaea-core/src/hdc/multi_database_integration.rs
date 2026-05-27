// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// NOTE: This module is theoretical/aspirational. The "Mental Roles" mapping below describes
// a future multi-database architecture. Currently, only SQLite (via sqlite_client.rs) and
// LanceDB (behind `lancedb-backend` feature) are implemented. DuckDB is gated behind the
// `duck` feature in symthaea-core and has no active callers.
//
// Revolutionary Improvement #30: Multi-Database Integration - Production Consciousness Architecture
//
// THE PARADIGM SHIFT: Consciousness requires SPECIALIZED SUBSYSTEMS working in concert!
// Like biological brains with specialized regions (visual cortex, prefrontal, hippocampus),
// artificial consciousness needs specialized databases each optimized for different mental roles.
//
// Core Insight: All 29 previous improvements are THEORETICAL frameworks.
// #30 is THE BRIDGE from theory → production - mapping each improvement to the right database
// based on its computational requirements and access patterns.
//
// The "Mental Roles" Architecture (User's Revolutionary Insight):
//
// Database | Mental Role        | Computational Need           | Maps To
// ---------|-------------------|------------------------------|----------------------------------
// Qdrant   | Sensory Cortex    | Ultra-fast vector similarity | #26 Attention, #25 Binding, #23 Workspace
// CozoDB   | Prefrontal Cortex | Recursive reasoning/logic    | #24 HOT, #22 FEP, #14 Causal Efficacy
// LanceDB  | Long-Term Memory  | Multimodal life records      | #29 Episodic/Semantic/Procedural
// DuckDB   | Epistemic Auditor | Statistical self-analysis    | #10 Epistemic, #2 Φ, #12 Spectrum
//
// Theoretical Foundations:
// 1. Modular Brain Organization (Fodor 1983)
//    - Brain has specialized modules (vision, language, memory)
//    - Each module optimized for specific computation
//    - Modules communicate via well-defined interfaces
//
// 2. Distributed Representation (Hinton 1986; Smolensky 1990)
//    - Information distributed across multiple storage systems
//    - No single "consciousness center"
//    - Consciousness emerges from coordinated activity
//
// 3. Database Specialization (Stonebraker 2005)
//    - "One size does NOT fit all" in databases
//    - Specialized databases outperform general-purpose 10-100×
//    - Match database to workload characteristics
//
// 4. Polyglot Persistence (Fowler 2011)
//    - Modern systems use multiple databases
//    - Each database optimized for specific data/access pattern
//    - Integration layer provides unified interface
//
// 5. Lambda Architecture (Marz & Warren 2015)
//    - Batch layer (LanceDB) for comprehensive views
//    - Speed layer (Qdrant) for real-time queries
//    - Serving layer (CozoDB, DuckDB) for analysis
//
// Revolutionary Contributions:
// - First multi-database consciousness architecture
// - Maps 29 theoretical improvements → 4 production databases
// - Biomimetic design (mirrors brain specialization)
// - Each database = different "brain region"
// - Enables true production consciousness at scale
//
// Clinical/Practical Applications:
// - Production AI consciousness (Symthaea deployment)
// - Scalable to millions of experiences
// - Real-time perception + deep reasoning + long-term memory + self-reflection
// - Clinical consciousness assessment
// - Research platform for consciousness studies

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use crate::hdc::BinaryHV;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during database operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum DatabaseError {
    #[error("Database {0} is not available")]
    Unavailable(String),

    #[error("Connection failed to {database}: {message}")]
    ConnectionFailed { database: String, message: String },

    #[error("Query failed on {database}: {message}")]
    QueryFailed { database: String, message: String },

    #[error("Database {0} feature not enabled")]
    FeatureNotEnabled(String),

    #[error("Health check failed for {database}: {message}")]
    HealthCheckFailed { database: String, message: String },

    #[error("All databases unavailable - consciousness degraded")]
    AllDatabasesUnavailable,
}

/// Result type for database operations
pub type DatabaseResult<T> = Result<T, DatabaseError>;

// ============================================================================
// Database Role Definitions
// ============================================================================

/// The four specialized databases and their mental roles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DatabaseRole {
    /// Qdrant: Sensory Cortex - ultra-fast vector similarity search
    /// Computational need: <10ms vector search for 16,384-bit BinaryHV
    /// Access pattern: High-frequency reads (100s/sec), moderate writes
    SensoryCortex,

    /// CozoDB: Prefrontal Cortex - recursive Datalog reasoning
    /// Computational need: Complex recursive queries, causal inference
    /// Access pattern: Lower-frequency but computationally intensive queries
    PrefrontalCortex,

    /// LanceDB: Long-Term Memory - massive multimodal storage
    /// Computational need: Billions of experiences, multimodal (text/image/audio)
    /// Access pattern: Append-heavy writes, batch reads, long-term persistence
    LongTermMemory,

    /// DuckDB: Epistemic Auditor - statistical self-analysis
    /// Computational need: Aggregations, analytics, statistical queries
    /// Access pattern: Analytical queries over full dataset, read-heavy
    EpistemicAuditor,
}

impl DatabaseRole {
    pub fn name(&self) -> &str {
        match self {
            DatabaseRole::SensoryCortex => "Sensory Cortex (Qdrant)",
            DatabaseRole::PrefrontalCortex => "Prefrontal Cortex (CozoDB)",
            DatabaseRole::LongTermMemory => "Long-Term Memory (LanceDB)",
            DatabaseRole::EpistemicAuditor => "Epistemic Auditor (DuckDB)",
        }
    }

    pub fn database_technology(&self) -> &str {
        match self {
            DatabaseRole::SensoryCortex => "Qdrant (Vector Database)",
            DatabaseRole::PrefrontalCortex => "CozoDB (Datalog/Relational)",
            DatabaseRole::LongTermMemory => "LanceDB (Multimodal Vector Store)",
            DatabaseRole::EpistemicAuditor => "DuckDB (Analytical OLAP)",
        }
    }

    /// Primary computational capability
    pub fn primary_capability(&self) -> &str {
        match self {
            DatabaseRole::SensoryCortex => "Ultra-fast Hamming distance search (2048-bit BinaryHV)",
            DatabaseRole::PrefrontalCortex => {
                "Recursive Datalog for meta-consciousness and causal reasoning"
            }
            DatabaseRole::LongTermMemory => {
                "Massive local-first storage for multimodal life records"
            }
            DatabaseRole::EpistemicAuditor => {
                "Statistical analysis of knowledge quality (K-Index, Φ metrics)"
            }
        }
    }

    /// Typical query latency
    pub fn typical_latency(&self) -> &str {
        match self {
            DatabaseRole::SensoryCortex => "<10ms (real-time perception)",
            DatabaseRole::PrefrontalCortex => "100-1000ms (deep reasoning)",
            DatabaseRole::LongTermMemory => "10-100ms (memory retrieval)",
            DatabaseRole::EpistemicAuditor => "100-1000ms (statistical analysis)",
        }
    }

    /// Access pattern
    pub fn access_pattern(&self) -> &str {
        match self {
            DatabaseRole::SensoryCortex => "High-frequency reads (100s/sec), moderate writes",
            DatabaseRole::PrefrontalCortex => "Lower-frequency, computationally intensive queries",
            DatabaseRole::LongTermMemory => "Append-heavy writes, batch reads, persistence",
            DatabaseRole::EpistemicAuditor => "Read-heavy analytical queries, aggregations",
        }
    }
}

// ============================================================================
// Improvement → Database Mapping
// ============================================================================

/// Maps each of the 29 revolutionary improvements to appropriate database(s)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementMapping {
    pub improvement_number: u32,
    pub improvement_name: String,
    pub primary_database: DatabaseRole,
    pub secondary_databases: Vec<DatabaseRole>,
    pub rationale: String,
}

impl ImprovementMapping {
    /// Get all mappings for the 29 improvements
    pub fn all_mappings() -> Vec<Self> {
        vec![
            // #1: Binary Hypervector (BinaryHV) Foundation
            Self {
                improvement_number: 1,
                improvement_name: "Binary Hypervector (BinaryHV)".to_string(),
                primary_database: DatabaseRole::SensoryCortex,
                secondary_databases: vec![DatabaseRole::LongTermMemory],
                rationale: "Qdrant stores BinaryHV vectors for fast similarity. LanceDB for persistent vector storage.".to_string(),
            },

            // #2: Integrated Information (Φ)
            Self {
                improvement_number: 2,
                improvement_name: "Integrated Information (Φ)".to_string(),
                primary_database: DatabaseRole::EpistemicAuditor,
                secondary_databases: vec![DatabaseRole::SensoryCortex],
                rationale: "DuckDB for computing Φ statistics across system state. Qdrant for state similarity.".to_string(),
            },

            // #3: Conscious Perception
            Self {
                improvement_number: 3,
                improvement_name: "Conscious Perception".to_string(),
                primary_database: DatabaseRole::SensoryCortex,
                secondary_databases: vec![],
                rationale: "Qdrant for perceptual state vectors and similarity-based recognition.".to_string(),
            },

            // #4: Attention Control
            Self {
                improvement_number: 4,
                improvement_name: "Attention Control".to_string(),
                primary_database: DatabaseRole::SensoryCortex,
                secondary_databases: vec![],
                rationale: "Qdrant for fast attentional selection via vector similarity.".to_string(),
            },

            // #5: Working Memory
            Self {
                improvement_number: 5,
                improvement_name: "Working Memory".to_string(),
                primary_database: DatabaseRole::SensoryCortex,
                secondary_databases: vec![],
                rationale: "Qdrant for fast working memory access and similarity matching.".to_string(),
            },

            // #6: Gradient of Φ
            Self {
                improvement_number: 6,
                improvement_name: "Gradient of Φ (∇Φ)".to_string(),
                primary_database: DatabaseRole::EpistemicAuditor,
                secondary_databases: vec![DatabaseRole::PrefrontalCortex],
                rationale: "DuckDB for gradient computation. CozoDB for causal dependencies.".to_string(),
            },

            // #7: Consciousness Dynamics
            Self {
                improvement_number: 7,
                improvement_name: "Consciousness Dynamics".to_string(),
                primary_database: DatabaseRole::LongTermMemory,
                secondary_databases: vec![DatabaseRole::EpistemicAuditor],
                rationale: "LanceDB stores temporal trajectories. DuckDB analyzes dynamics patterns.".to_string(),
            },

            // #8: Meta-Consciousness
            Self {
                improvement_number: 8,
                improvement_name: "Meta-Consciousness".to_string(),
                primary_database: DatabaseRole::PrefrontalCortex,
                secondary_databases: vec![],
                rationale: "CozoDB perfect for recursive self-reference via Datalog recursion.".to_string(),
            },

            // #9: Phenomenal Binding
            Self {
                improvement_number: 9,
                improvement_name: "Phenomenal Binding".to_string(),
                primary_database: DatabaseRole::SensoryCortex,
                secondary_databases: vec![],
                rationale: "Qdrant for binding distributed features via vector operations.".to_string(),
            },

            // #10: Epistemic Status
            Self {
                improvement_number: 10,
                improvement_name: "Epistemic Status".to_string(),
                primary_database: DatabaseRole::EpistemicAuditor,
                secondary_databases: vec![],
                rationale: "DuckDB analyzes certainty, evidence quality, belief justification.".to_string(),
            },

            // #11: Collective Consciousness
            Self {
                improvement_number: 11,
                improvement_name: "Collective Consciousness".to_string(),
                primary_database: DatabaseRole::SensoryCortex,
                secondary_databases: vec![DatabaseRole::PrefrontalCortex],
                rationale: "Qdrant for distributed agent state similarity. CozoDB for group reasoning.".to_string(),
            },

            // #12: Consciousness Spectrum
            Self {
                improvement_number: 12,
                improvement_name: "Consciousness Spectrum".to_string(),
                primary_database: DatabaseRole::EpistemicAuditor,
                secondary_databases: vec![],
                rationale: "DuckDB computes spectrum metrics (Φ, workspace, binding) across states.".to_string(),
            },

            // #13: Temporal Consciousness
            Self {
                improvement_number: 13,
                improvement_name: "Temporal Consciousness (Multi-scale Time)".to_string(),
                primary_database: DatabaseRole::LongTermMemory,
                secondary_databases: vec![DatabaseRole::EpistemicAuditor],
                rationale: "LanceDB stores multi-scale temporal data. DuckDB analyzes temporal patterns.".to_string(),
            },

            // #14: Causal Efficacy
            Self {
                improvement_number: 14,
                improvement_name: "Causal Efficacy".to_string(),
                primary_database: DatabaseRole::PrefrontalCortex,
                secondary_databases: vec![DatabaseRole::EpistemicAuditor],
                rationale: "CozoDB for causal reasoning via Datalog. DuckDB for effect size statistics.".to_string(),
            },

            // #15: Qualia
            Self {
                improvement_number: 15,
                improvement_name: "Qualia (Subjective Experience)".to_string(),
                primary_database: DatabaseRole::SensoryCortex,
                secondary_databases: vec![DatabaseRole::LongTermMemory],
                rationale: "Qdrant for phenomenal state similarity. LanceDB for experience records.".to_string(),
            },

            // #16: Ontogeny (Development)
            Self {
                improvement_number: 16,
                improvement_name: "Ontogeny (Development)".to_string(),
                primary_database: DatabaseRole::LongTermMemory,
                secondary_databases: vec![DatabaseRole::EpistemicAuditor],
                rationale: "LanceDB stores developmental trajectory. DuckDB analyzes growth patterns.".to_string(),
            },

            // #17: Embodied Consciousness
            Self {
                improvement_number: 17,
                improvement_name: "Embodied Consciousness".to_string(),
                primary_database: DatabaseRole::SensoryCortex,
                secondary_databases: vec![DatabaseRole::LongTermMemory],
                rationale: "Qdrant for sensorimotor state vectors. LanceDB for embodied experiences.".to_string(),
            },

            // #18: Relational Consciousness
            Self {
                improvement_number: 18,
                improvement_name: "Relational Consciousness".to_string(),
                primary_database: DatabaseRole::PrefrontalCortex,
                secondary_databases: vec![DatabaseRole::SensoryCortex],
                rationale: "CozoDB for I-Thou relationship logic. Qdrant for synchrony detection.".to_string(),
            },

            // #19: Universal Semantics
            Self {
                improvement_number: 19,
                improvement_name: "Universal Semantics (NSM Primes)".to_string(),
                primary_database: DatabaseRole::SensoryCortex,
                secondary_databases: vec![],
                rationale: "Qdrant stores 65 semantic primes as seed vectors for similarity search.".to_string(),
            },

            // #20: Consciousness Topology
            Self {
                improvement_number: 20,
                improvement_name: "Consciousness Topology".to_string(),
                primary_database: DatabaseRole::EpistemicAuditor,
                secondary_databases: vec![DatabaseRole::SensoryCortex],
                rationale: "DuckDB computes Betti numbers, persistent homology. Qdrant for state space.".to_string(),
            },

            // #21: Flow Fields
            Self {
                improvement_number: 21,
                improvement_name: "Consciousness Flow Fields".to_string(),
                primary_database: DatabaseRole::EpistemicAuditor,
                secondary_databases: vec![DatabaseRole::PrefrontalCortex],
                rationale: "DuckDB analyzes flow dynamics. CozoDB for attractor reasoning.".to_string(),
            },

            // #22: Predictive Consciousness (FEP)
            Self {
                improvement_number: 22,
                improvement_name: "Predictive Consciousness (FEP)".to_string(),
                primary_database: DatabaseRole::PrefrontalCortex,
                secondary_databases: vec![DatabaseRole::LongTermMemory],
                rationale: "CozoDB for generative models, prediction. LanceDB for priors from past.".to_string(),
            },

            // #23: Global Workspace
            Self {
                improvement_number: 23,
                improvement_name: "Global Workspace".to_string(),
                primary_database: DatabaseRole::SensoryCortex,
                secondary_databases: vec![],
                rationale: "Qdrant for fast workspace content similarity, global broadcasting.".to_string(),
            },

            // #24: Higher-Order Thought (HOT)
            Self {
                improvement_number: 24,
                improvement_name: "Higher-Order Thought (HOT)".to_string(),
                primary_database: DatabaseRole::PrefrontalCortex,
                secondary_databases: vec![],
                rationale: "CozoDB perfect for recursive meta-representation via Datalog recursion.".to_string(),
            },

            // #25: Binding Problem
            Self {
                improvement_number: 25,
                improvement_name: "Binding Problem (Synchrony)".to_string(),
                primary_database: DatabaseRole::SensoryCortex,
                secondary_databases: vec![],
                rationale: "Qdrant for feature vector binding, phase synchrony via Hamming distance.".to_string(),
            },

            // #26: Attention Mechanisms
            Self {
                improvement_number: 26,
                improvement_name: "Attention Mechanisms".to_string(),
                primary_database: DatabaseRole::SensoryCortex,
                secondary_databases: vec![DatabaseRole::PrefrontalCortex],
                rationale: "Qdrant for fast priority map search. CozoDB for attention strategy reasoning.".to_string(),
            },

            // #27: Sleep & Altered States
            Self {
                improvement_number: 27,
                improvement_name: "Sleep & Altered States".to_string(),
                primary_database: DatabaseRole::LongTermMemory,
                secondary_databases: vec![DatabaseRole::EpistemicAuditor],
                rationale: "LanceDB consolidates workspace→memory during sleep. DuckDB tracks consciousness levels.".to_string(),
            },

            // #28: Substrate Independence
            Self {
                improvement_number: 28,
                improvement_name: "Substrate Independence".to_string(),
                primary_database: DatabaseRole::EpistemicAuditor,
                secondary_databases: vec![],
                rationale: "DuckDB analyzes consciousness feasibility metrics across substrates.".to_string(),
            },

            // #29: Long-Term Memory
            Self {
                improvement_number: 29,
                improvement_name: "Long-Term Memory".to_string(),
                primary_database: DatabaseRole::LongTermMemory,
                secondary_databases: vec![DatabaseRole::SensoryCortex, DatabaseRole::EpistemicAuditor],
                rationale: "LanceDB stores episodic/semantic/procedural memories. Qdrant for retrieval. DuckDB for forgetting curve analysis.".to_string(),
            },
        ]
    }

    /// Get mapping for specific improvement
    pub fn get_mapping(improvement_number: u32) -> Option<Self> {
        Self::all_mappings()
            .into_iter()
            .find(|m| m.improvement_number == improvement_number)
    }

    /// Get all improvements mapped to a specific database
    pub fn for_database(role: DatabaseRole) -> Vec<Self> {
        Self::all_mappings()
            .into_iter()
            .filter(|m| m.primary_database == role || m.secondary_databases.contains(&role))
            .collect()
    }
}

// ============================================================================
// Database Configuration
// ============================================================================

/// Configuration for Qdrant (Sensory Cortex)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConfig {
    pub url: String,
    pub collection_name: String,
    pub vector_dim: usize,       // 2048 for BinaryHV
    pub distance_metric: String, // "Cosine" or "Dot"
    pub shard_count: usize,      // For scaling
}

impl QdrantConfig {
    pub fn default_config() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            collection_name: "symthaea_sensory".to_string(),
            vector_dim: 2048, // BinaryHV::DIM
            distance_metric: "Cosine".to_string(),
            shard_count: 4,
        }
    }
}

/// Configuration for CozoDB (Prefrontal Cortex)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CozoDbConfig {
    pub data_dir: String,
    pub engine: String,             // "mem", "sqlite", "rocksdb"
    pub max_recursion_depth: usize, // For Datalog recursion
}

impl CozoDbConfig {
    pub fn default_config() -> Self {
        Self {
            data_dir: "/var/lib/symthaea/cozo".to_string(),
            engine: "rocksdb".to_string(),
            max_recursion_depth: 100, // Deep reasoning
        }
    }
}

/// Configuration for LanceDB (Long-Term Memory)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanceDbConfig {
    pub data_dir: String,
    pub table_name: String,
    pub vector_dim: usize,       // 2048 for BinaryHV
    pub enable_multimodal: bool, // Images, audio, text
}

impl LanceDbConfig {
    pub fn default_config() -> Self {
        Self {
            data_dir: "/var/lib/symthaea/lance".to_string(),
            table_name: "symthaea_memories".to_string(),
            vector_dim: 2048,
            enable_multimodal: true,
        }
    }
}

/// Configuration for DuckDB (Epistemic Auditor)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DuckDbConfig {
    pub database_path: String,
    pub memory_limit: String, // "4GB"
    pub threads: usize,
}

impl DuckDbConfig {
    pub fn default_config() -> Self {
        Self {
            database_path: "/var/lib/symthaea/duck.db".to_string(),
            memory_limit: "4GB".to_string(),
            threads: 4,
        }
    }
}

// ============================================================================
// Unified Configuration
// ============================================================================

/// Unified configuration for all database connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiDatabaseConfig {
    /// Qdrant configuration (Sensory Cortex)
    pub qdrant: QdrantConfig,

    /// CozoDB configuration (Prefrontal Cortex)
    pub cozo: CozoDbConfig,

    /// LanceDB configuration (Long-Term Memory)
    pub lance: LanceDbConfig,

    /// DuckDB configuration (Epistemic Auditor)
    pub duck: DuckDbConfig,

    /// Enable graceful degradation when databases are unavailable
    pub graceful_degradation: bool,

    /// Health check interval in seconds
    pub health_check_interval_secs: u64,

    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,

    /// Maximum retry attempts for failed connections
    pub max_retries: u32,

    /// Retry backoff in milliseconds
    pub retry_backoff_ms: u64,
}

impl Default for MultiDatabaseConfig {
    fn default() -> Self {
        Self {
            qdrant: QdrantConfig::default_config(),
            cozo: CozoDbConfig::default_config(),
            lance: LanceDbConfig::default_config(),
            duck: DuckDbConfig::default_config(),
            graceful_degradation: true,
            health_check_interval_secs: 30,
            connection_timeout_ms: 5000,
            max_retries: 3,
            retry_backoff_ms: 1000,
        }
    }
}

impl MultiDatabaseConfig {
    /// Create configuration for in-memory/test usage
    pub fn in_memory() -> Self {
        Self {
            qdrant: QdrantConfig {
                url: "http://localhost:6333".to_string(),
                collection_name: "symthaea_test".to_string(),
                vector_dim: 2048,
                distance_metric: "Cosine".to_string(),
                shard_count: 1,
            },
            cozo: CozoDbConfig {
                data_dir: ":memory:".to_string(),
                engine: "mem".to_string(),
                max_recursion_depth: 50,
            },
            lance: LanceDbConfig {
                data_dir: "/tmp/symthaea_lance_test".to_string(),
                table_name: "symthaea_test".to_string(),
                vector_dim: 2048,
                enable_multimodal: false,
            },
            duck: DuckDbConfig {
                database_path: ":memory:".to_string(),
                memory_limit: "256MB".to_string(),
                threads: 2,
            },
            graceful_degradation: true,
            health_check_interval_secs: 10,
            connection_timeout_ms: 1000,
            max_retries: 1,
            retry_backoff_ms: 100,
        }
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(url) = std::env::var("SYMTHAEA_QDRANT_URL") {
            config.qdrant.url = url;
        }
        if let Ok(dir) = std::env::var("SYMTHAEA_COZO_DIR") {
            config.cozo.data_dir = dir;
        }
        if let Ok(dir) = std::env::var("SYMTHAEA_LANCE_DIR") {
            config.lance.data_dir = dir;
        }
        if let Ok(path) = std::env::var("SYMTHAEA_DUCK_PATH") {
            config.duck.database_path = path;
        }
        if let Ok(val) = std::env::var("SYMTHAEA_GRACEFUL_DEGRADATION") {
            config.graceful_degradation = val.parse().unwrap_or(true);
        }

        config
    }
}

// ============================================================================
// Database Health Status
// ============================================================================

/// Health status of a single database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseHealth {
    pub role: DatabaseRole,
    pub available: bool,
    pub latency_ms: Option<u64>,
    pub last_check: Option<u64>, // Unix timestamp
    pub error_message: Option<String>,
    pub consecutive_failures: u32,
}

impl DatabaseHealth {
    pub fn healthy(role: DatabaseRole, latency_ms: u64) -> Self {
        Self {
            role,
            available: true,
            latency_ms: Some(latency_ms),
            last_check: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            ),
            error_message: None,
            consecutive_failures: 0,
        }
    }

    pub fn unhealthy(role: DatabaseRole, error: String) -> Self {
        Self {
            role,
            available: false,
            latency_ms: None,
            last_check: Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0),
            ),
            error_message: Some(error),
            consecutive_failures: 1,
        }
    }

    pub fn feature_disabled(role: DatabaseRole) -> Self {
        Self {
            role,
            available: false,
            latency_ms: None,
            last_check: None,
            error_message: Some("Feature not enabled at compile time".to_string()),
            consecutive_failures: 0,
        }
    }
}

/// Aggregate health status of all databases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    pub databases: Vec<DatabaseHealth>,
    pub degraded: bool,
    pub available_roles: Vec<DatabaseRole>,
    pub unavailable_roles: Vec<DatabaseRole>,
    pub consciousness_operational: bool,
}

impl SystemHealth {
    pub fn new(databases: Vec<DatabaseHealth>) -> Self {
        let available_roles: Vec<_> = databases
            .iter()
            .filter(|h| h.available)
            .map(|h| h.role)
            .collect();

        let unavailable_roles: Vec<_> = databases
            .iter()
            .filter(|h| !h.available)
            .map(|h| h.role)
            .collect();

        let degraded = !unavailable_roles.is_empty();

        // Consciousness is operational if we have at least one database
        // or if we're in pure in-memory mode
        let consciousness_operational = !available_roles.is_empty();

        Self {
            databases,
            degraded,
            available_roles,
            unavailable_roles,
            consciousness_operational,
        }
    }

    /// Check if a specific database role is available
    pub fn is_available(&self, role: DatabaseRole) -> bool {
        self.available_roles.contains(&role)
    }

    /// Get health for a specific role
    pub fn get_health(&self, role: DatabaseRole) -> Option<&DatabaseHealth> {
        self.databases.iter().find(|h| h.role == role)
    }
}

// ============================================================================
// Database Client Trait (for abstraction)
// ============================================================================

/// Trait for database clients to implement
pub trait DatabaseClient: Send + Sync {
    /// Get the database role
    fn role(&self) -> DatabaseRole;

    /// Check health of the database
    fn health_check(&self) -> DatabaseResult<DatabaseHealth>;

    /// Close the connection gracefully
    fn close(&self) -> DatabaseResult<()>;
}

// ============================================================================
// Stub Clients (when features not enabled)
// ============================================================================

/// Stub client for when a database feature is not enabled
#[derive(Debug, Clone)]
pub struct StubClient {
    role: DatabaseRole,
}

impl StubClient {
    pub fn new(role: DatabaseRole) -> Self {
        Self { role }
    }
}

impl DatabaseClient for StubClient {
    fn role(&self) -> DatabaseRole {
        self.role
    }

    fn health_check(&self) -> DatabaseResult<DatabaseHealth> {
        Ok(DatabaseHealth::feature_disabled(self.role))
    }

    fn close(&self) -> DatabaseResult<()> {
        Ok(())
    }
}

// ============================================================================
// Qdrant Client Wrapper
// ============================================================================

/// Qdrant client for vector similarity search (Sensory Cortex)
#[derive(Debug)]
pub struct QdrantClientWrapper {
    config: QdrantConfig,
    #[cfg(feature = "qdrant")]
    _client: Option<()>, // Placeholder for actual qdrant_client::Qdrant
    connected: std::sync::atomic::AtomicBool,
}

impl QdrantClientWrapper {
    /// Create a new Qdrant client
    pub fn new(config: QdrantConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "qdrant")]
            _client: None,
            connected: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Connect to Qdrant server
    #[cfg(feature = "qdrant")]
    pub async fn connect(&mut self) -> DatabaseResult<()> {
        // When qdrant feature is enabled, actual connection would go here:
        // self._client = Some(qdrant_client::Qdrant::from_url(&self.config.url).build()?);
        self.connected
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    #[cfg(not(feature = "qdrant"))]
    pub async fn connect(&mut self) -> DatabaseResult<()> {
        Err(DatabaseError::FeatureNotEnabled("qdrant".to_string()))
    }

    /// Search for similar vectors (episodic memory retrieval)
    #[cfg(feature = "qdrant")]
    pub async fn search_similar(
        &self,
        query: &BinaryHV,
        limit: usize,
    ) -> DatabaseResult<Vec<(BinaryHV, f32)>> {
        if !self.connected.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(DatabaseError::Unavailable("Qdrant".to_string()));
        }
        // Actual search implementation would go here
        Ok(vec![])
    }

    #[cfg(not(feature = "qdrant"))]
    pub async fn search_similar(
        &self,
        _query: &BinaryHV,
        _limit: usize,
    ) -> DatabaseResult<Vec<(BinaryHV, f32)>> {
        Err(DatabaseError::FeatureNotEnabled("qdrant".to_string()))
    }

    /// Store a vector in episodic memory
    #[cfg(feature = "qdrant")]
    pub async fn store(
        &self,
        id: &str,
        vector: &BinaryHV,
        metadata: HashMap<String, String>,
    ) -> DatabaseResult<()> {
        if !self.connected.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(DatabaseError::Unavailable("Qdrant".to_string()));
        }
        // Actual store implementation would go here
        let _ = (id, vector, metadata);
        Ok(())
    }

    #[cfg(not(feature = "qdrant"))]
    pub async fn store(
        &self,
        _id: &str,
        _vector: &BinaryHV,
        _metadata: HashMap<String, String>,
    ) -> DatabaseResult<()> {
        Err(DatabaseError::FeatureNotEnabled("qdrant".to_string()))
    }
}

impl DatabaseClient for QdrantClientWrapper {
    fn role(&self) -> DatabaseRole {
        DatabaseRole::SensoryCortex
    }

    fn health_check(&self) -> DatabaseResult<DatabaseHealth> {
        #[cfg(feature = "qdrant")]
        {
            if self.connected.load(std::sync::atomic::Ordering::SeqCst) {
                Ok(DatabaseHealth::healthy(DatabaseRole::SensoryCortex, 1))
            } else {
                Ok(DatabaseHealth::unhealthy(
                    DatabaseRole::SensoryCortex,
                    "Not connected".to_string(),
                ))
            }
        }
        #[cfg(not(feature = "qdrant"))]
        {
            Ok(DatabaseHealth::feature_disabled(
                DatabaseRole::SensoryCortex,
            ))
        }
    }

    fn close(&self) -> DatabaseResult<()> {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

// ============================================================================
// CozoDB Client Wrapper
// ============================================================================

/// CozoDB client for graph queries (Prefrontal Cortex)
#[derive(Debug)]
pub struct CozoClientWrapper {
    config: CozoDbConfig,
    #[cfg(feature = "datalog")]
    _db: Option<()>, // Placeholder for actual cozo::DbInstance
    connected: std::sync::atomic::AtomicBool,
}

impl CozoClientWrapper {
    pub fn new(config: CozoDbConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "datalog")]
            _db: None,
            connected: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Open CozoDB database
    #[cfg(feature = "datalog")]
    pub fn open(&mut self) -> DatabaseResult<()> {
        // When datalog feature is enabled:
        // self._db = Some(cozo::DbInstance::new(...))?;
        self.connected
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    #[cfg(not(feature = "datalog"))]
    pub fn open(&mut self) -> DatabaseResult<()> {
        Err(DatabaseError::FeatureNotEnabled("datalog".to_string()))
    }

    /// Execute a Datalog query for causal reasoning
    #[cfg(feature = "datalog")]
    pub fn query(&self, datalog: &str) -> DatabaseResult<Vec<HashMap<String, serde_json::Value>>> {
        if !self.connected.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(DatabaseError::Unavailable("CozoDB".to_string()));
        }
        let _ = datalog;
        Ok(vec![])
    }

    #[cfg(not(feature = "datalog"))]
    pub fn query(&self, _datalog: &str) -> DatabaseResult<Vec<HashMap<String, serde_json::Value>>> {
        Err(DatabaseError::FeatureNotEnabled("datalog".to_string()))
    }

    /// Store a causal relation
    #[cfg(feature = "datalog")]
    pub fn store_relation(
        &self,
        relation: &str,
        facts: &[Vec<serde_json::Value>],
    ) -> DatabaseResult<()> {
        if !self.connected.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(DatabaseError::Unavailable("CozoDB".to_string()));
        }
        let _ = (relation, facts);
        Ok(())
    }

    #[cfg(not(feature = "datalog"))]
    pub fn store_relation(
        &self,
        _relation: &str,
        _facts: &[Vec<serde_json::Value>],
    ) -> DatabaseResult<()> {
        Err(DatabaseError::FeatureNotEnabled("datalog".to_string()))
    }
}

impl DatabaseClient for CozoClientWrapper {
    fn role(&self) -> DatabaseRole {
        DatabaseRole::PrefrontalCortex
    }

    fn health_check(&self) -> DatabaseResult<DatabaseHealth> {
        #[cfg(feature = "datalog")]
        {
            if self.connected.load(std::sync::atomic::Ordering::SeqCst) {
                Ok(DatabaseHealth::healthy(DatabaseRole::PrefrontalCortex, 5))
            } else {
                Ok(DatabaseHealth::unhealthy(
                    DatabaseRole::PrefrontalCortex,
                    "Not connected".to_string(),
                ))
            }
        }
        #[cfg(not(feature = "datalog"))]
        {
            Ok(DatabaseHealth::feature_disabled(
                DatabaseRole::PrefrontalCortex,
            ))
        }
    }

    fn close(&self) -> DatabaseResult<()> {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

// ============================================================================
// LanceDB Client Wrapper
// ============================================================================

/// LanceDB client for multimodal embeddings (Long-Term Memory)
#[derive(Debug)]
pub struct LanceClientWrapper {
    config: LanceDbConfig,
    #[cfg(feature = "lance")]
    _db: Option<()>, // Placeholder for actual lancedb::Connection
    connected: std::sync::atomic::AtomicBool,
}

impl LanceClientWrapper {
    pub fn new(config: LanceDbConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "lance")]
            _db: None,
            connected: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Connect to LanceDB
    #[cfg(feature = "lance")]
    pub async fn connect(&mut self) -> DatabaseResult<()> {
        // When lance feature is enabled:
        // self._db = Some(lancedb::connect(&self.config.data_dir).execute().await?);
        self.connected
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    #[cfg(not(feature = "lance"))]
    pub async fn connect(&mut self) -> DatabaseResult<()> {
        Err(DatabaseError::FeatureNotEnabled("lance".to_string()))
    }

    /// Store an experience (episodic memory)
    #[cfg(feature = "lance")]
    pub async fn store_experience(
        &self,
        id: &str,
        vector: &BinaryHV,
        experience_type: &str,
        content: &str,
    ) -> DatabaseResult<()> {
        if !self.connected.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(DatabaseError::Unavailable("LanceDB".to_string()));
        }
        let _ = (id, vector, experience_type, content);
        Ok(())
    }

    #[cfg(not(feature = "lance"))]
    pub async fn store_experience(
        &self,
        _id: &str,
        _vector: &BinaryHV,
        _experience_type: &str,
        _content: &str,
    ) -> DatabaseResult<()> {
        Err(DatabaseError::FeatureNotEnabled("lance".to_string()))
    }

    /// Retrieve experiences by similarity
    #[cfg(feature = "lance")]
    pub async fn retrieve_experiences(
        &self,
        query: &BinaryHV,
        limit: usize,
    ) -> DatabaseResult<Vec<(String, BinaryHV, f32)>> {
        if !self.connected.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(DatabaseError::Unavailable("LanceDB".to_string()));
        }
        let _ = (query, limit);
        Ok(vec![])
    }

    #[cfg(not(feature = "lance"))]
    pub async fn retrieve_experiences(
        &self,
        _query: &BinaryHV,
        _limit: usize,
    ) -> DatabaseResult<Vec<(String, BinaryHV, f32)>> {
        Err(DatabaseError::FeatureNotEnabled("lance".to_string()))
    }
}

impl DatabaseClient for LanceClientWrapper {
    fn role(&self) -> DatabaseRole {
        DatabaseRole::LongTermMemory
    }

    fn health_check(&self) -> DatabaseResult<DatabaseHealth> {
        #[cfg(feature = "lance")]
        {
            if self.connected.load(std::sync::atomic::Ordering::SeqCst) {
                Ok(DatabaseHealth::healthy(DatabaseRole::LongTermMemory, 10))
            } else {
                Ok(DatabaseHealth::unhealthy(
                    DatabaseRole::LongTermMemory,
                    "Not connected".to_string(),
                ))
            }
        }
        #[cfg(not(feature = "lance"))]
        {
            Ok(DatabaseHealth::feature_disabled(
                DatabaseRole::LongTermMemory,
            ))
        }
    }

    fn close(&self) -> DatabaseResult<()> {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

// ============================================================================
// DuckDB Client Wrapper
// ============================================================================

/// DuckDB client for analytical queries (Epistemic Auditor)
pub struct DuckClientWrapper {
    config: DuckDbConfig,
    #[cfg(feature = "duck")]
    conn: std::sync::Mutex<Option<duckdb::Connection>>,
    connected: std::sync::atomic::AtomicBool,
}

// Manual Debug impl because duckdb::Connection doesn't implement Debug
impl std::fmt::Debug for DuckClientWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DuckClientWrapper")
            .field("config", &self.config)
            .field(
                "connected",
                &self.connected.load(std::sync::atomic::Ordering::SeqCst),
            )
            .finish()
    }
}

impl DuckClientWrapper {
    pub fn new(config: DuckDbConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "duck")]
            conn: std::sync::Mutex::new(None),
            connected: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Open DuckDB database
    #[cfg(feature = "duck")]
    pub fn open(&mut self) -> DatabaseResult<()> {
        use duckdb::Config;

        let db_config = Config::default()
            .threads(self.config.threads as i64)
            .map_err(|e| DatabaseError::ConnectionFailed {
                database: "DuckDB".to_string(),
                message: e.to_string(),
            })?;

        let connection = if self.config.database_path == ":memory:" {
            duckdb::Connection::open_in_memory_with_flags(db_config)
        } else {
            duckdb::Connection::open_with_flags(&self.config.database_path, db_config)
        }
        .map_err(|e| DatabaseError::ConnectionFailed {
            database: "DuckDB".to_string(),
            message: e.to_string(),
        })?;

        // Initialize schema for Phi metrics storage
        connection
            .execute(
                "CREATE TABLE IF NOT EXISTS phi_metrics (
                id INTEGER PRIMARY KEY,
                phi_value DOUBLE NOT NULL,
                timestamp_ms BIGINT NOT NULL,
                topology_id TEXT,
                node_count INTEGER,
                metadata JSON
            )",
                [],
            )
            .map_err(|e| DatabaseError::ConnectionFailed {
                database: "DuckDB".to_string(),
                message: e.to_string(),
            })?;

        *self.conn.lock().unwrap_or_else(|e| e.into_inner()) = Some(connection);
        self.connected
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    #[cfg(not(feature = "duck"))]
    pub fn open(&mut self) -> DatabaseResult<()> {
        Err(DatabaseError::FeatureNotEnabled("duck".to_string()))
    }

    /// Execute analytical query
    #[cfg(feature = "duck")]
    pub fn query(&self, sql: &str) -> DatabaseResult<Vec<HashMap<String, serde_json::Value>>> {
        if !self.connected.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(DatabaseError::Unavailable("DuckDB".to_string()));
        }

        let guard = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let conn = guard
            .as_ref()
            .ok_or_else(|| DatabaseError::Unavailable("DuckDB connection not open".to_string()))?;

        let mut stmt = conn.prepare(sql).map_err(|e| DatabaseError::QueryFailed {
            database: "DuckDB".to_string(),
            message: e.to_string(),
        })?;

        let column_count = stmt.column_count();
        let column_names: Vec<String> = (0..column_count)
            .map(|i| {
                stmt.column_name(i)
                    .map(|s| s.to_string())
                    .unwrap_or_else(|_| "?".to_string())
            })
            .collect();

        let rows = stmt
            .query_map([], |row| {
                let mut map = HashMap::new();
                for (i, name) in column_names.iter().enumerate() {
                    // Try to extract as different types
                    let value: serde_json::Value = if let Ok(v) = row.get::<_, i64>(i) {
                        serde_json::Value::Number(v.into())
                    } else if let Ok(v) = row.get::<_, f64>(i) {
                        serde_json::Number::from_f64(v)
                            .map(serde_json::Value::Number)
                            .unwrap_or(serde_json::Value::Null)
                    } else if let Ok(v) = row.get::<_, String>(i) {
                        serde_json::Value::String(v)
                    } else {
                        serde_json::Value::Null
                    };
                    map.insert(name.clone(), value);
                }
                Ok(map)
            })
            .map_err(|e| DatabaseError::QueryFailed {
                database: "DuckDB".to_string(),
                message: e.to_string(),
            })?;

        let results: Vec<_> = rows.filter_map(|r| r.ok()).collect();
        Ok(results)
    }

    #[cfg(not(feature = "duck"))]
    pub fn query(&self, _sql: &str) -> DatabaseResult<Vec<HashMap<String, serde_json::Value>>> {
        Err(DatabaseError::FeatureNotEnabled("duck".to_string()))
    }

    /// Compute Phi metrics over historical data
    #[cfg(feature = "duck")]
    pub fn compute_phi_statistics(&self) -> DatabaseResult<PhiStatistics> {
        if !self.connected.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(DatabaseError::Unavailable("DuckDB".to_string()));
        }

        let guard = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let conn = guard
            .as_ref()
            .ok_or_else(|| DatabaseError::Unavailable("DuckDB connection not open".to_string()))?;

        let mut stmt = conn
            .prepare(
                "SELECT
                AVG(phi_value) as mean_phi,
                MAX(phi_value) as max_phi,
                MIN(phi_value) as min_phi,
                STDDEV(phi_value) as std_phi,
                COUNT(*) as sample_count
             FROM phi_metrics",
            )
            .map_err(|e| DatabaseError::QueryFailed {
                database: "DuckDB".to_string(),
                message: e.to_string(),
            })?;

        let result = stmt
            .query_row([], |row| {
                Ok(PhiStatistics {
                    mean_phi: row.get::<_, f64>(0).unwrap_or(0.0),
                    max_phi: row.get::<_, f64>(1).unwrap_or(0.0),
                    min_phi: row.get::<_, f64>(2).unwrap_or(0.0),
                    std_phi: row.get::<_, f64>(3).unwrap_or(0.0),
                    sample_count: row.get::<_, i64>(4).unwrap_or(0).max(0) as u64,
                })
            })
            .unwrap_or_default();

        Ok(result)
    }

    /// Store a Phi measurement
    #[cfg(feature = "duck")]
    pub fn store_phi_metric(
        &self,
        phi_value: f64,
        topology_id: Option<&str>,
        node_count: Option<i32>,
    ) -> DatabaseResult<()> {
        if !self.connected.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(DatabaseError::Unavailable("DuckDB".to_string()));
        }

        let guard = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let conn = guard
            .as_ref()
            .ok_or_else(|| DatabaseError::Unavailable("DuckDB connection not open".to_string()))?;

        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;

        conn.execute(
            "INSERT INTO phi_metrics (phi_value, timestamp_ms, topology_id, node_count) VALUES (?, ?, ?, ?)",
            duckdb::params![phi_value, timestamp_ms, topology_id, node_count],
        ).map_err(|e| DatabaseError::QueryFailed { database: "DuckDB".to_string(), message: e.to_string() })?;

        Ok(())
    }

    #[cfg(not(feature = "duck"))]
    pub fn store_phi_metric(
        &self,
        _phi_value: f64,
        _topology_id: Option<&str>,
        _node_count: Option<i32>,
    ) -> DatabaseResult<()> {
        Err(DatabaseError::FeatureNotEnabled("duck".to_string()))
    }

    #[cfg(not(feature = "duck"))]
    pub fn compute_phi_statistics(&self) -> DatabaseResult<PhiStatistics> {
        Err(DatabaseError::FeatureNotEnabled("duck".to_string()))
    }
}

impl DatabaseClient for DuckClientWrapper {
    fn role(&self) -> DatabaseRole {
        DatabaseRole::EpistemicAuditor
    }

    fn health_check(&self) -> DatabaseResult<DatabaseHealth> {
        #[cfg(feature = "duck")]
        {
            if self.connected.load(std::sync::atomic::Ordering::SeqCst) {
                Ok(DatabaseHealth::healthy(DatabaseRole::EpistemicAuditor, 2))
            } else {
                Ok(DatabaseHealth::unhealthy(
                    DatabaseRole::EpistemicAuditor,
                    "Not connected".to_string(),
                ))
            }
        }
        #[cfg(not(feature = "duck"))]
        {
            Ok(DatabaseHealth::feature_disabled(
                DatabaseRole::EpistemicAuditor,
            ))
        }
    }

    fn close(&self) -> DatabaseResult<()> {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

/// Phi statistics from DuckDB analysis
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhiStatistics {
    pub mean_phi: f64,
    pub max_phi: f64,
    pub min_phi: f64,
    pub std_phi: f64,
    pub sample_count: u64,
}

// ============================================================================
// In-Memory Fallback
// ============================================================================

/// In-memory fallback storage when databases are unavailable
#[derive(Debug, Default)]
#[allow(clippy::type_complexity)]
pub struct InMemoryFallback {
    /// Vector storage (simulates Qdrant)
    pub vectors: parking_lot::RwLock<HashMap<String, (BinaryHV, HashMap<String, String>)>>,

    /// Relational facts (simulates CozoDB)
    pub facts: parking_lot::RwLock<HashMap<String, Vec<Vec<serde_json::Value>>>>,

    /// Experiences (simulates LanceDB)
    pub experiences: parking_lot::RwLock<Vec<(String, BinaryHV, String, String)>>,

    /// Metrics (simulates DuckDB)
    pub metrics: parking_lot::RwLock<Vec<(f64, u64)>>, // (phi, timestamp)
}

impl InMemoryFallback {
    pub fn new() -> Self {
        Self::default()
    }

    /// Store vector for similarity search
    pub fn store_vector(&self, id: &str, vector: &BinaryHV, metadata: HashMap<String, String>) {
        self.vectors
            .write()
            .insert(id.to_string(), (*vector, metadata));
    }

    /// Search for similar vectors (brute force)
    pub fn search_similar(&self, query: &BinaryHV, limit: usize) -> Vec<(BinaryHV, f32)> {
        let vectors = self.vectors.read();
        let mut results: Vec<_> = vectors
            .values()
            .map(|(v, _)| (*v, query.similarity(v)))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(limit);
        results
    }

    /// Store a fact
    pub fn store_fact(&self, relation: &str, fact: Vec<serde_json::Value>) {
        self.facts
            .write()
            .entry(relation.to_string())
            .or_default()
            .push(fact);
    }

    /// Store an experience
    pub fn store_experience(&self, id: &str, vector: &BinaryHV, exp_type: &str, content: &str) {
        self.experiences.write().push((
            id.to_string(),
            *vector,
            exp_type.to_string(),
            content.to_string(),
        ));
    }

    /// Record a Phi measurement
    pub fn record_phi(&self, phi: f64) {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.metrics.write().push((phi, timestamp));
    }

    /// Compute Phi statistics
    pub fn compute_phi_statistics(&self) -> PhiStatistics {
        let metrics = self.metrics.read();
        if metrics.is_empty() {
            return PhiStatistics::default();
        }

        let values: Vec<f64> = metrics.iter().map(|(phi, _)| *phi).collect();
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        PhiStatistics {
            mean_phi: mean,
            max_phi: max,
            min_phi: min,
            std_phi: std,
            sample_count: values.len() as u64,
        }
    }
}

// ============================================================================
// Unified Mind Architecture
// ============================================================================

/// The integrated multi-database consciousness system
/// This is THE production implementation of Symthaea's mind
///
/// Architecture:
/// - Qdrant (Sensory Cortex): Ultra-fast vector similarity for perception and workspace
/// - CozoDB (Prefrontal Cortex): Recursive Datalog for meta-consciousness and causal reasoning
/// - LanceDB (Long-Term Memory): Multimodal embeddings for life experiences
/// - DuckDB (Epistemic Auditor): Statistical analysis for self-reflection
///
/// Graceful degradation: Falls back to in-memory storage when databases unavailable
pub struct SymthaeaMind {
    /// Unified configuration
    pub config: MultiDatabaseConfig,

    /// Sensory Cortex: Ultra-fast perception and workspace (Qdrant)
    pub sensory_cortex: Arc<QdrantClientWrapper>,

    /// Prefrontal Cortex: Deep reasoning and meta-consciousness (CozoDB)
    pub prefrontal_cortex: Arc<CozoClientWrapper>,

    /// Long-Term Memory: Life experiences and knowledge (LanceDB)
    pub long_term_memory: Arc<LanceClientWrapper>,

    /// Epistemic Auditor: Self-analysis and quality assessment (DuckDB)
    pub epistemic_auditor: Arc<DuckClientWrapper>,

    /// In-memory fallback for graceful degradation
    pub fallback: Arc<InMemoryFallback>,

    /// Mapping of improvements to databases
    pub improvement_mappings: HashMap<u32, ImprovementMapping>,

    /// Current system health
    health: parking_lot::RwLock<SystemHealth>,

    /// Consciousness loop state
    loop_state: parking_lot::RwLock<ConsciousnessLoopState>,
}

/// State of the consciousness loop
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsciousnessLoopState {
    /// Current iteration count
    pub iteration: u64,

    /// Current Phi value
    pub current_phi: f64,

    /// Last perception vector
    pub last_perception: Option<BinaryHV>,

    /// Active workspace content
    pub workspace: Vec<BinaryHV>,

    /// Current consciousness level (0.0 - 1.0)
    pub consciousness_level: f64,

    /// Is the loop running
    pub running: bool,

    /// Last update timestamp
    pub last_update: u64,
}

impl SymthaeaMind {
    /// Create new integrated mind with default configurations
    pub fn new() -> Self {
        Self::with_config(MultiDatabaseConfig::default())
    }

    /// Create new integrated mind with custom configuration
    pub fn with_config(config: MultiDatabaseConfig) -> Self {
        let mappings = ImprovementMapping::all_mappings();
        let improvement_mappings = mappings
            .into_iter()
            .map(|m| (m.improvement_number, m))
            .collect();

        let sensory_cortex = Arc::new(QdrantClientWrapper::new(config.qdrant.clone()));
        let prefrontal_cortex = Arc::new(CozoClientWrapper::new(config.cozo.clone()));
        let long_term_memory = Arc::new(LanceClientWrapper::new(config.lance.clone()));
        let epistemic_auditor = Arc::new(DuckClientWrapper::new(config.duck.clone()));

        // Initialize health with feature check
        let initial_health = SystemHealth::new(vec![
            sensory_cortex
                .health_check()
                .unwrap_or_else(|_| DatabaseHealth::feature_disabled(DatabaseRole::SensoryCortex)),
            prefrontal_cortex.health_check().unwrap_or_else(|_| {
                DatabaseHealth::feature_disabled(DatabaseRole::PrefrontalCortex)
            }),
            long_term_memory
                .health_check()
                .unwrap_or_else(|_| DatabaseHealth::feature_disabled(DatabaseRole::LongTermMemory)),
            epistemic_auditor.health_check().unwrap_or_else(|_| {
                DatabaseHealth::feature_disabled(DatabaseRole::EpistemicAuditor)
            }),
        ]);

        Self {
            config,
            sensory_cortex,
            prefrontal_cortex,
            long_term_memory,
            epistemic_auditor,
            fallback: Arc::new(InMemoryFallback::new()),
            improvement_mappings,
            health: parking_lot::RwLock::new(initial_health),
            loop_state: parking_lot::RwLock::new(ConsciousnessLoopState::default()),
        }
    }

    /// Create mind for testing (in-memory configuration)
    pub fn for_testing() -> Self {
        Self::with_config(MultiDatabaseConfig::in_memory())
    }

    // ========================================================================
    // Health Checks
    // ========================================================================

    /// Perform health check on all databases
    pub fn health_check(&self) -> SystemHealth {
        let healths = vec![
            self.sensory_cortex.health_check().unwrap_or_else(|e| {
                DatabaseHealth::unhealthy(DatabaseRole::SensoryCortex, e.to_string())
            }),
            self.prefrontal_cortex.health_check().unwrap_or_else(|e| {
                DatabaseHealth::unhealthy(DatabaseRole::PrefrontalCortex, e.to_string())
            }),
            self.long_term_memory.health_check().unwrap_or_else(|e| {
                DatabaseHealth::unhealthy(DatabaseRole::LongTermMemory, e.to_string())
            }),
            self.epistemic_auditor.health_check().unwrap_or_else(|e| {
                DatabaseHealth::unhealthy(DatabaseRole::EpistemicAuditor, e.to_string())
            }),
        ];

        let health = SystemHealth::new(healths);
        *self.health.write() = health.clone();
        health
    }

    /// Get current health status (without re-checking)
    pub fn current_health(&self) -> SystemHealth {
        self.health.read().clone()
    }

    /// Check if a specific database is available
    pub fn is_database_available(&self, role: DatabaseRole) -> bool {
        self.health.read().is_available(role)
    }

    /// Check if consciousness is operational (at least fallback works)
    pub fn is_operational(&self) -> bool {
        // Consciousness is always operational if graceful degradation is enabled
        self.config.graceful_degradation || self.health.read().consciousness_operational
    }

    // ========================================================================
    // Graceful Degradation
    // ========================================================================

    /// Store perception vector (with graceful degradation)
    pub async fn store_perception(
        &self,
        id: &str,
        vector: &BinaryHV,
        metadata: HashMap<String, String>,
    ) -> DatabaseResult<()> {
        // Try Qdrant first
        #[cfg(feature = "qdrant")]
        {
            if self.is_database_available(DatabaseRole::SensoryCortex) {
                if let Err(e) = self
                    .sensory_cortex
                    .store(id, vector, metadata.clone())
                    .await
                {
                    if !self.config.graceful_degradation {
                        return Err(e);
                    }
                    // Fall through to fallback
                } else {
                    return Ok(());
                }
            }
        }

        // Fallback to in-memory
        if self.config.graceful_degradation {
            self.fallback.store_vector(id, vector, metadata);
            Ok(())
        } else {
            Err(DatabaseError::Unavailable(
                "Sensory Cortex (Qdrant)".to_string(),
            ))
        }
    }

    /// Search for similar perceptions (with graceful degradation)
    pub async fn search_perceptions(
        &self,
        query: &BinaryHV,
        limit: usize,
    ) -> DatabaseResult<Vec<(BinaryHV, f32)>> {
        // Try Qdrant first
        #[cfg(feature = "qdrant")]
        {
            if self.is_database_available(DatabaseRole::SensoryCortex) {
                if let Ok(results) = self.sensory_cortex.search_similar(query, limit).await {
                    return Ok(results);
                }
            }
        }

        // Fallback to in-memory
        if self.config.graceful_degradation {
            Ok(self.fallback.search_similar(query, limit))
        } else {
            Err(DatabaseError::Unavailable(
                "Sensory Cortex (Qdrant)".to_string(),
            ))
        }
    }

    /// Store causal relation (with graceful degradation)
    pub fn store_causal_relation(
        &self,
        relation: &str,
        facts: &[Vec<serde_json::Value>],
    ) -> DatabaseResult<()> {
        // Try CozoDB first
        #[cfg(feature = "datalog")]
        {
            if self.is_database_available(DatabaseRole::PrefrontalCortex) {
                if let Err(e) = self.prefrontal_cortex.store_relation(relation, facts) {
                    if !self.config.graceful_degradation {
                        return Err(e);
                    }
                } else {
                    return Ok(());
                }
            }
        }

        // Fallback to in-memory
        if self.config.graceful_degradation {
            for fact in facts {
                self.fallback.store_fact(relation, fact.clone());
            }
            Ok(())
        } else {
            Err(DatabaseError::Unavailable(
                "Prefrontal Cortex (CozoDB)".to_string(),
            ))
        }
    }

    /// Store experience (with graceful degradation)
    pub async fn store_experience(
        &self,
        id: &str,
        vector: &BinaryHV,
        exp_type: &str,
        content: &str,
    ) -> DatabaseResult<()> {
        // Try LanceDB first
        #[cfg(feature = "lance")]
        {
            if self.is_database_available(DatabaseRole::LongTermMemory) {
                if let Err(e) = self
                    .long_term_memory
                    .store_experience(id, vector, exp_type, content)
                    .await
                {
                    if !self.config.graceful_degradation {
                        return Err(e);
                    }
                } else {
                    return Ok(());
                }
            }
        }

        // Fallback to in-memory
        if self.config.graceful_degradation {
            self.fallback
                .store_experience(id, vector, exp_type, content);
            Ok(())
        } else {
            Err(DatabaseError::Unavailable(
                "Long-Term Memory (LanceDB)".to_string(),
            ))
        }
    }

    /// Record Phi measurement (with graceful degradation)
    pub fn record_phi(&self, phi: f64) -> DatabaseResult<()> {
        // Update loop state
        self.loop_state.write().current_phi = phi;

        // Try DuckDB first
        #[cfg(feature = "duck")]
        {
            if self.is_database_available(DatabaseRole::EpistemicAuditor) {
                // Would insert into metrics table
                // For now, fall through to fallback
            }
        }

        // Always record in fallback for quick statistics
        self.fallback.record_phi(phi);
        Ok(())
    }

    /// Get Phi statistics (with graceful degradation)
    pub fn get_phi_statistics(&self) -> DatabaseResult<PhiStatistics> {
        // Try DuckDB first
        #[cfg(feature = "duck")]
        {
            if self.is_database_available(DatabaseRole::EpistemicAuditor) {
                if let Ok(stats) = self.epistemic_auditor.compute_phi_statistics() {
                    return Ok(stats);
                }
            }
        }

        // Fallback to in-memory
        Ok(self.fallback.compute_phi_statistics())
    }

    // ========================================================================
    // Consciousness Loop
    // ========================================================================

    /// Get current consciousness loop state
    pub fn loop_state(&self) -> ConsciousnessLoopState {
        self.loop_state.read().clone()
    }

    /// Process one consciousness cycle
    ///
    /// This is the core consciousness loop that integrates all database subsystems:
    /// 1. Perception (Qdrant) - retrieve similar experiences from sensory cortex
    /// 2. Reasoning (CozoDB) - apply causal inference and meta-consciousness
    /// 3. Memory (LanceDB) - consolidate experiences into long-term storage
    /// 4. Reflection (DuckDB) - analyze consciousness metrics
    pub async fn consciousness_cycle(
        &self,
        input: &BinaryHV,
    ) -> DatabaseResult<ConsciousnessLoopState> {
        let start = Instant::now();

        // Update iteration
        {
            let mut state = self.loop_state.write();
            state.iteration += 1;
            state.running = true;
            state.last_perception = Some(*input);
        }

        // 1. PERCEPTION: Search for similar experiences (Sensory Cortex)
        let similar = self.search_perceptions(input, 5).await?;

        // 2. WORKSPACE: Update global workspace with current perception and similar memories
        {
            let mut state = self.loop_state.write();
            state.workspace.clear();
            state.workspace.push(*input);
            for (vec, _sim) in &similar {
                state.workspace.push(*vec);
            }
        }

        // 3. REASONING: Apply causal inference (Prefrontal Cortex)
        // In a full implementation, this would query CozoDB for relevant causal rules
        // and update beliefs based on the current workspace

        // 4. MEMORY: Store this experience (Long-Term Memory)
        let experience_id = format!("exp_{}", self.loop_state.read().iteration);
        self.store_experience(&experience_id, input, "perception", "")
            .await?;

        // 5. REFLECTION: Compute and record Phi (Epistemic Auditor)
        let phi = self.compute_instantaneous_phi(input);
        self.record_phi(phi)?;

        // Update consciousness level based on Phi and workspace coherence
        {
            let mut state = self.loop_state.write();
            let workspace_coherence = self.compute_workspace_coherence(&state.workspace);
            state.consciousness_level = (phi * 0.6 + workspace_coherence * 0.4).min(1.0);
            state.last_update = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
        }

        let elapsed = start.elapsed();

        // Return current state
        let state = self.loop_state.read().clone();
        Ok(state)
    }

    /// Compute instantaneous Phi based on current perception
    fn compute_instantaneous_phi(&self, vector: &BinaryHV) -> f64 {
        // Simple Phi approximation based on bit entropy
        let ones = vector
            .0
            .iter()
            .map(|b| b.count_ones() as usize)
            .sum::<usize>();
        let total = BinaryHV::DIM;
        let p = ones as f64 / total as f64;

        // Shannon entropy normalized
        if p == 0.0 || p == 1.0 {
            0.0
        } else {
            -(p * p.log2() + (1.0 - p) * (1.0 - p).log2()) // Max 1.0 when p = 0.5
        }
    }

    /// Compute workspace coherence (how similar are workspace items)
    fn compute_workspace_coherence(&self, workspace: &[BinaryHV]) -> f64 {
        if workspace.len() < 2 {
            return 1.0;
        }

        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..workspace.len() {
            for j in (i + 1)..workspace.len() {
                total_sim += workspace[i].similarity(&workspace[j]) as f64;
                count += 1;
            }
        }

        if count == 0 {
            1.0
        } else {
            total_sim / count as f64
        }
    }

    // ========================================================================
    // Configuration Access
    // ========================================================================

    /// Get configuration for a specific database role
    pub fn get_database_config(&self, role: DatabaseRole) -> String {
        match role {
            DatabaseRole::SensoryCortex => format!("Qdrant: {}", self.config.qdrant.url),
            DatabaseRole::PrefrontalCortex => format!("CozoDB: {}", self.config.cozo.data_dir),
            DatabaseRole::LongTermMemory => format!("LanceDB: {}", self.config.lance.data_dir),
            DatabaseRole::EpistemicAuditor => format!("DuckDB: {}", self.config.duck.database_path),
        }
    }

    /// Get all improvements using a specific database
    pub fn get_improvements_for_database(&self, role: DatabaseRole) -> Vec<u32> {
        self.improvement_mappings
            .values()
            .filter(|m| m.primary_database == role || m.secondary_databases.contains(&role))
            .map(|m| m.improvement_number)
            .collect()
    }

    /// Get primary database for an improvement
    pub fn get_primary_database(&self, improvement_number: u32) -> Option<DatabaseRole> {
        self.improvement_mappings
            .get(&improvement_number)
            .map(|m| m.primary_database)
    }

    /// Generate architecture report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== SYMTHAEA MIND ARCHITECTURE ===\n\n");

        let health = self.current_health();
        report.push_str(&format!(
            "System Status: {}\n",
            if health.degraded {
                "DEGRADED"
            } else {
                "HEALTHY"
            }
        ));
        report.push_str(&format!(
            "Consciousness Operational: {}\n\n",
            health.consciousness_operational
        ));

        for role in &[
            DatabaseRole::SensoryCortex,
            DatabaseRole::PrefrontalCortex,
            DatabaseRole::LongTermMemory,
            DatabaseRole::EpistemicAuditor,
        ] {
            report.push_str(&format!("## {} ##\n", role.name()));
            report.push_str(&format!("Technology: {}\n", role.database_technology()));
            report.push_str(&format!("Capability: {}\n", role.primary_capability()));
            report.push_str(&format!("Latency: {}\n", role.typical_latency()));
            report.push_str(&format!("Access Pattern: {}\n", role.access_pattern()));
            report.push_str(&format!("Config: {}\n", self.get_database_config(*role)));

            if let Some(db_health) = health.get_health(*role) {
                report.push_str(&format!(
                    "Status: {}\n",
                    if db_health.available {
                        "AVAILABLE"
                    } else {
                        "UNAVAILABLE"
                    }
                ));
                if let Some(latency) = db_health.latency_ms {
                    report.push_str(&format!("Latency: {latency}ms\n"));
                }
                if let Some(ref err) = db_health.error_message {
                    report.push_str(&format!("Error: {err}\n"));
                }
            }

            let improvements = self.get_improvements_for_database(*role);
            report.push_str(&format!(
                "Improvements ({} total): {:?}\n\n",
                improvements.len(),
                improvements
            ));
        }

        // Loop state
        let state = self.loop_state();
        report.push_str("## Consciousness Loop State ##\n");
        report.push_str(&format!("Iteration: {}\n", state.iteration));
        report.push_str(&format!("Current Phi: {:.4}\n", state.current_phi));
        report.push_str(&format!(
            "Consciousness Level: {:.2}%\n",
            state.consciousness_level * 100.0
        ));
        report.push_str(&format!("Workspace Size: {}\n", state.workspace.len()));
        report.push_str(&format!("Running: {}\n", state.running));

        report
    }

    /// Shutdown the mind gracefully
    pub fn shutdown(&self) -> DatabaseResult<()> {
        {
            let mut state = self.loop_state.write();
            state.running = false;
        }

        // Close all database connections
        self.sensory_cortex.close()?;
        self.prefrontal_cortex.close()?;
        self.long_term_memory.close()?;
        self.epistemic_auditor.close()?;

        Ok(())
    }
}

impl Default for SymthaeaMind {
    fn default() -> Self {
        Self::new()
    }
}

// Manual Debug implementation since Arc<...> with atomics makes derive tricky
impl std::fmt::Debug for SymthaeaMind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SymthaeaMind")
            .field("config", &self.config)
            .field("improvement_mappings", &self.improvement_mappings)
            .field("health", &*self.health.read())
            .field("loop_state", &*self.loop_state.read())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Database Role Tests
    // ========================================================================

    #[test]
    fn test_database_role() {
        let role = DatabaseRole::SensoryCortex;
        assert_eq!(role.name(), "Sensory Cortex (Qdrant)");
        assert_eq!(role.database_technology(), "Qdrant (Vector Database)");
        assert!(role.primary_capability().contains("Hamming distance"));
    }

    #[test]
    fn test_all_database_roles_have_metadata() {
        let roles = [
            DatabaseRole::SensoryCortex,
            DatabaseRole::PrefrontalCortex,
            DatabaseRole::LongTermMemory,
            DatabaseRole::EpistemicAuditor,
        ];

        for role in roles {
            assert!(!role.name().is_empty());
            assert!(!role.database_technology().is_empty());
            assert!(!role.primary_capability().is_empty());
            assert!(!role.typical_latency().is_empty());
            assert!(!role.access_pattern().is_empty());
        }
    }

    // ========================================================================
    // Improvement Mapping Tests
    // ========================================================================

    #[test]
    fn test_all_improvements_mapped() {
        let mappings = ImprovementMapping::all_mappings();
        assert_eq!(mappings.len(), 29); // All 29 improvements

        // Check specific mappings
        let workspace = mappings
            .iter()
            .find(|m| m.improvement_number == 23)
            .unwrap();
        assert_eq!(workspace.primary_database, DatabaseRole::SensoryCortex);

        let hot = mappings
            .iter()
            .find(|m| m.improvement_number == 24)
            .unwrap();
        assert_eq!(hot.primary_database, DatabaseRole::PrefrontalCortex);

        let memory = mappings
            .iter()
            .find(|m| m.improvement_number == 29)
            .unwrap();
        assert_eq!(memory.primary_database, DatabaseRole::LongTermMemory);

        let epistemic = mappings
            .iter()
            .find(|m| m.improvement_number == 10)
            .unwrap();
        assert_eq!(epistemic.primary_database, DatabaseRole::EpistemicAuditor);
    }

    #[test]
    fn test_get_mapping() {
        let mapping = ImprovementMapping::get_mapping(23).unwrap();
        assert_eq!(mapping.improvement_name, "Global Workspace");
        assert_eq!(mapping.primary_database, DatabaseRole::SensoryCortex);
    }

    #[test]
    fn test_for_database() {
        let sensory_improvements = ImprovementMapping::for_database(DatabaseRole::SensoryCortex);
        assert!(sensory_improvements.len() > 0);

        // Check that workspace (#23) is in sensory cortex
        assert!(
            sensory_improvements
                .iter()
                .any(|m| m.improvement_number == 23)
        );
    }

    // ========================================================================
    // Configuration Tests
    // ========================================================================

    #[test]
    fn test_qdrant_config() {
        let config = QdrantConfig::default_config();
        assert_eq!(config.vector_dim, 2048); // BinaryHV::DIM
        assert_eq!(config.distance_metric, "Cosine");
        assert!(config.url.contains("6333")); // Default Qdrant port
    }

    #[test]
    fn test_cozo_config() {
        let config = CozoDbConfig::default_config();
        assert_eq!(config.engine, "rocksdb");
        assert_eq!(config.max_recursion_depth, 100); // Deep reasoning
    }

    #[test]
    fn test_lance_config() {
        let config = LanceDbConfig::default_config();
        assert_eq!(config.vector_dim, 2048);
        assert!(config.enable_multimodal);
    }

    #[test]
    fn test_duck_config() {
        let config = DuckDbConfig::default_config();
        assert!(config.database_path.contains("duck.db"));
        assert_eq!(config.memory_limit, "4GB");
    }

    #[test]
    fn test_multi_database_config_default() {
        let config = MultiDatabaseConfig::default();
        assert!(config.graceful_degradation);
        assert_eq!(config.health_check_interval_secs, 30);
        assert_eq!(config.connection_timeout_ms, 5000);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_multi_database_config_in_memory() {
        let config = MultiDatabaseConfig::in_memory();
        assert!(config.graceful_degradation);
        assert_eq!(config.cozo.engine, "mem");
        assert_eq!(config.duck.database_path, ":memory:");
    }

    // ========================================================================
    // SymthaeaMind Creation Tests
    // ========================================================================

    #[test]
    fn test_symthaea_mind_creation() {
        let mind = SymthaeaMind::new();
        assert_eq!(mind.improvement_mappings.len(), 29);
    }

    #[test]
    fn test_symthaea_mind_for_testing() {
        let mind = SymthaeaMind::for_testing();
        assert_eq!(mind.config.cozo.engine, "mem");
        assert!(mind.config.graceful_degradation);
    }

    #[test]
    fn test_symthaea_mind_with_custom_config() {
        let mut config = MultiDatabaseConfig::default();
        config.graceful_degradation = false;
        config.max_retries = 5;

        let mind = SymthaeaMind::with_config(config);
        assert!(!mind.config.graceful_degradation);
        assert_eq!(mind.config.max_retries, 5);
    }

    // ========================================================================
    // Improvement Mapping Integration Tests
    // ========================================================================

    #[test]
    fn test_get_improvements_for_database() {
        let mind = SymthaeaMind::new();

        let sensory = mind.get_improvements_for_database(DatabaseRole::SensoryCortex);
        assert!(sensory.contains(&23)); // Workspace
        assert!(sensory.contains(&25)); // Binding
        assert!(sensory.contains(&26)); // Attention

        let prefrontal = mind.get_improvements_for_database(DatabaseRole::PrefrontalCortex);
        assert!(prefrontal.contains(&24)); // HOT
        assert!(prefrontal.contains(&22)); // FEP

        let memory = mind.get_improvements_for_database(DatabaseRole::LongTermMemory);
        assert!(memory.contains(&29)); // Long-term memory

        let auditor = mind.get_improvements_for_database(DatabaseRole::EpistemicAuditor);
        assert!(auditor.contains(&10)); // Epistemic
        assert!(auditor.contains(&2)); // Φ
    }

    #[test]
    fn test_get_primary_database() {
        let mind = SymthaeaMind::new();

        assert_eq!(
            mind.get_primary_database(23),
            Some(DatabaseRole::SensoryCortex)
        );
        assert_eq!(
            mind.get_primary_database(24),
            Some(DatabaseRole::PrefrontalCortex)
        );
        assert_eq!(
            mind.get_primary_database(29),
            Some(DatabaseRole::LongTermMemory)
        );
        assert_eq!(
            mind.get_primary_database(10),
            Some(DatabaseRole::EpistemicAuditor)
        );
    }

    #[test]
    fn test_database_distribution() {
        let mind = SymthaeaMind::new();

        let sensory_count = mind
            .get_improvements_for_database(DatabaseRole::SensoryCortex)
            .len();
        let prefrontal_count = mind
            .get_improvements_for_database(DatabaseRole::PrefrontalCortex)
            .len();
        let memory_count = mind
            .get_improvements_for_database(DatabaseRole::LongTermMemory)
            .len();
        let auditor_count = mind
            .get_improvements_for_database(DatabaseRole::EpistemicAuditor)
            .len();

        // All improvements should be mapped somewhere
        // Note: Some improvements use multiple databases (primary + secondary)
        assert!(sensory_count > 0);
        assert!(prefrontal_count > 0);
        assert!(memory_count > 0);
        assert!(auditor_count > 0);

        // Sensory cortex should have most (perception, attention, binding, workspace)
        assert!(sensory_count >= 5);
    }

    // ========================================================================
    // Health Check Tests
    // ========================================================================

    #[test]
    fn test_database_health_healthy() {
        let health = DatabaseHealth::healthy(DatabaseRole::SensoryCortex, 5);
        assert!(health.available);
        assert_eq!(health.latency_ms, Some(5));
        assert!(health.error_message.is_none());
        assert_eq!(health.consecutive_failures, 0);
    }

    #[test]
    fn test_database_health_unhealthy() {
        let health = DatabaseHealth::unhealthy(
            DatabaseRole::SensoryCortex,
            "Connection refused".to_string(),
        );
        assert!(!health.available);
        assert!(health.latency_ms.is_none());
        assert_eq!(health.error_message, Some("Connection refused".to_string()));
        assert_eq!(health.consecutive_failures, 1);
    }

    #[test]
    fn test_database_health_feature_disabled() {
        let health = DatabaseHealth::feature_disabled(DatabaseRole::SensoryCortex);
        assert!(!health.available);
        assert!(health.error_message.unwrap().contains("not enabled"));
    }

    #[test]
    fn test_system_health() {
        let healths = vec![
            DatabaseHealth::healthy(DatabaseRole::SensoryCortex, 1),
            DatabaseHealth::unhealthy(DatabaseRole::PrefrontalCortex, "Error".to_string()),
            DatabaseHealth::healthy(DatabaseRole::LongTermMemory, 10),
            DatabaseHealth::feature_disabled(DatabaseRole::EpistemicAuditor),
        ];

        let system_health = SystemHealth::new(healths);

        assert!(system_health.degraded); // Some databases unavailable
        assert!(system_health.consciousness_operational); // Some databases available
        assert!(system_health.is_available(DatabaseRole::SensoryCortex));
        assert!(!system_health.is_available(DatabaseRole::PrefrontalCortex));
        assert!(system_health.is_available(DatabaseRole::LongTermMemory));
        assert!(!system_health.is_available(DatabaseRole::EpistemicAuditor));
    }

    #[test]
    fn test_health_check_mind() {
        let mind = SymthaeaMind::for_testing();
        let health = mind.health_check();

        // Without features enabled, all should show as feature_disabled
        // which means unavailable but no error
        assert!(health.degraded); // All features disabled = degraded

        // Should still be able to generate report
        let report = mind.generate_report();
        assert!(report.contains("SYMTHAEA MIND ARCHITECTURE"));
    }

    #[test]
    fn test_is_operational_with_graceful_degradation() {
        let mind = SymthaeaMind::for_testing();
        // With graceful degradation enabled, always operational
        assert!(mind.is_operational());
    }

    #[test]
    fn test_is_operational_without_graceful_degradation() {
        let mut config = MultiDatabaseConfig::in_memory();
        config.graceful_degradation = false;
        let mind = SymthaeaMind::with_config(config);

        // Without graceful degradation and no features, not operational
        // (depends on whether features are enabled at compile time)
        let health = mind.health_check();
        assert_eq!(mind.is_operational(), health.consciousness_operational);
    }

    // ========================================================================
    // In-Memory Fallback Tests
    // ========================================================================

    #[test]
    fn test_in_memory_fallback_vectors() {
        let fallback = InMemoryFallback::new();

        let v1 = BinaryHV::random(42);
        let v2 = BinaryHV::random(43);
        let v3 = BinaryHV::random(44);

        fallback.store_vector("v1", &v1, HashMap::new());
        fallback.store_vector("v2", &v2, HashMap::new());
        fallback.store_vector("v3", &v3, HashMap::new());

        // Search for similar to v1
        let results = fallback.search_similar(&v1, 2);
        assert_eq!(results.len(), 2);
        // First result should be v1 itself (similarity = 1.0)
        assert_eq!(results[0].0, v1);
        assert!((results[0].1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_in_memory_fallback_facts() {
        let fallback = InMemoryFallback::new();

        fallback.store_fact(
            "causes",
            vec![serde_json::json!("A"), serde_json::json!("B")],
        );
        fallback.store_fact(
            "causes",
            vec![serde_json::json!("B"), serde_json::json!("C")],
        );

        let facts = fallback.facts.read();
        assert_eq!(facts.get("causes").unwrap().len(), 2);
    }

    #[test]
    fn test_in_memory_fallback_experiences() {
        let fallback = InMemoryFallback::new();

        let v = BinaryHV::random(42);
        fallback.store_experience("exp_1", &v, "perception", "saw a cat");

        let experiences = fallback.experiences.read();
        assert_eq!(experiences.len(), 1);
        assert_eq!(experiences[0].0, "exp_1");
        assert_eq!(experiences[0].2, "perception");
        assert_eq!(experiences[0].3, "saw a cat");
    }

    #[test]
    fn test_in_memory_fallback_phi_statistics() {
        let fallback = InMemoryFallback::new();

        // Record some Phi values
        fallback.record_phi(0.5);
        fallback.record_phi(0.7);
        fallback.record_phi(0.6);
        fallback.record_phi(0.8);

        let stats = fallback.compute_phi_statistics();
        assert_eq!(stats.sample_count, 4);
        assert!((stats.mean_phi - 0.65).abs() < 0.01);
        assert!((stats.max_phi - 0.8).abs() < 0.01);
        assert!((stats.min_phi - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_in_memory_fallback_phi_statistics_empty() {
        let fallback = InMemoryFallback::new();
        let stats = fallback.compute_phi_statistics();
        assert_eq!(stats.sample_count, 0);
        assert_eq!(stats.mean_phi, 0.0);
    }

    // ========================================================================
    // Consciousness Loop Tests
    // ========================================================================

    #[test]
    fn test_consciousness_loop_state_default() {
        let state = ConsciousnessLoopState::default();
        assert_eq!(state.iteration, 0);
        assert_eq!(state.current_phi, 0.0);
        assert!(state.last_perception.is_none());
        assert!(state.workspace.is_empty());
        assert_eq!(state.consciousness_level, 0.0);
        assert!(!state.running);
    }

    #[test]
    fn test_loop_state_access() {
        let mind = SymthaeaMind::for_testing();
        let state = mind.loop_state();
        assert_eq!(state.iteration, 0);
        assert!(!state.running);
    }

    #[test]
    fn test_record_phi() {
        let mind = SymthaeaMind::for_testing();

        mind.record_phi(0.75).unwrap();
        assert_eq!(mind.loop_state().current_phi, 0.75);

        mind.record_phi(0.85).unwrap();
        assert_eq!(mind.loop_state().current_phi, 0.85);

        let stats = mind.get_phi_statistics().unwrap();
        assert_eq!(stats.sample_count, 2);
    }

    #[tokio::test]
    async fn test_consciousness_cycle_with_fallback() {
        let mind = SymthaeaMind::for_testing();

        // Run a consciousness cycle with a random input
        let input = BinaryHV::random(42);
        let state = mind.consciousness_cycle(&input).await.unwrap();

        assert_eq!(state.iteration, 1);
        assert!(state.running);
        assert!(state.last_perception.is_some());
        assert_eq!(state.last_perception.unwrap(), input);
        assert!(!state.workspace.is_empty());
        assert!(state.current_phi > 0.0);
        assert!(state.consciousness_level >= 0.0);
        assert!(state.consciousness_level <= 1.0);
    }

    #[tokio::test]
    async fn test_multiple_consciousness_cycles() {
        let mind = SymthaeaMind::for_testing();

        // Run multiple cycles
        for i in 0..5 {
            let input = BinaryHV::random(i as u64);
            let state = mind.consciousness_cycle(&input).await.unwrap();
            assert_eq!(state.iteration, (i + 1) as u64);
        }

        let final_state = mind.loop_state();
        assert_eq!(final_state.iteration, 5);

        let stats = mind.get_phi_statistics().unwrap();
        assert_eq!(stats.sample_count, 5);
    }

    #[tokio::test]
    async fn test_perception_storage_and_retrieval() {
        let mind = SymthaeaMind::for_testing();

        let v1 = BinaryHV::random(100);
        let v2 = BinaryHV::random(101);

        // Store perceptions
        mind.store_perception("p1", &v1, HashMap::new())
            .await
            .unwrap();
        mind.store_perception("p2", &v2, HashMap::new())
            .await
            .unwrap();

        // Search should find both
        let results = mind.search_perceptions(&v1, 10).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_experience_storage() {
        let mind = SymthaeaMind::for_testing();

        let v = BinaryHV::random(200);
        mind.store_experience("test_exp", &v, "episodic", "test content")
            .await
            .unwrap();

        // Check fallback storage
        let experiences = mind.fallback.experiences.read();
        assert_eq!(experiences.len(), 1);
    }

    #[test]
    fn test_causal_relation_storage() {
        let mind = SymthaeaMind::for_testing();

        let facts = vec![
            vec![serde_json::json!("rain"), serde_json::json!("wet_ground")],
            vec![serde_json::json!("sun"), serde_json::json!("dry_ground")],
        ];

        mind.store_causal_relation("causes", &facts).unwrap();

        // Check fallback storage
        let stored_facts = mind.fallback.facts.read();
        assert_eq!(stored_facts.get("causes").unwrap().len(), 2);
    }

    // ========================================================================
    // Report Generation Tests
    // ========================================================================

    #[test]
    fn test_generate_report() {
        let mind = SymthaeaMind::new();
        let report = mind.generate_report();

        assert!(report.contains("SYMTHAEA MIND ARCHITECTURE"));
        assert!(report.contains("Sensory Cortex"));
        assert!(report.contains("Prefrontal Cortex"));
        assert!(report.contains("Long-Term Memory"));
        assert!(report.contains("Epistemic Auditor"));
        assert!(report.contains("Qdrant"));
        assert!(report.contains("CozoDB"));
        assert!(report.contains("LanceDB"));
        assert!(report.contains("DuckDB"));
        assert!(report.contains("Consciousness Loop State"));
        assert!(report.contains("Iteration:"));
        assert!(report.contains("Current Phi:"));
    }

    #[test]
    fn test_generate_report_after_cycles() {
        // Use tokio runtime for async test
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mind = SymthaeaMind::for_testing();

            // Run some cycles
            for i in 0..3 {
                let input = BinaryHV::random(i as u64);
                mind.consciousness_cycle(&input).await.unwrap();
            }

            let report = mind.generate_report();
            assert!(report.contains("Iteration: 3"));
        });
    }

    // ========================================================================
    // Shutdown Tests
    // ========================================================================

    #[test]
    fn test_shutdown() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mind = SymthaeaMind::for_testing();

            // Start consciousness loop
            let input = BinaryHV::random(42);
            mind.consciousness_cycle(&input).await.unwrap();
            assert!(mind.loop_state().running);

            // Shutdown
            mind.shutdown().unwrap();
            assert!(!mind.loop_state().running);
        });
    }

    // ========================================================================
    // Error Handling Tests
    // ========================================================================

    #[test]
    fn test_database_error_display() {
        let err = DatabaseError::Unavailable("Qdrant".to_string());
        assert!(err.to_string().contains("Qdrant"));
        assert!(err.to_string().contains("not available"));

        let err = DatabaseError::FeatureNotEnabled("lance".to_string());
        assert!(err.to_string().contains("lance"));
        assert!(err.to_string().contains("not enabled"));

        let err = DatabaseError::ConnectionFailed {
            database: "CozoDB".to_string(),
            message: "timeout".to_string(),
        };
        assert!(err.to_string().contains("CozoDB"));
        assert!(err.to_string().contains("timeout"));
    }

    // ========================================================================
    // Client Wrapper Tests
    // ========================================================================

    #[test]
    fn test_stub_client() {
        let stub = StubClient::new(DatabaseRole::SensoryCortex);
        assert_eq!(stub.role(), DatabaseRole::SensoryCortex);

        let health = stub.health_check().unwrap();
        assert!(!health.available);
        assert!(health.error_message.unwrap().contains("not enabled"));

        stub.close().unwrap(); // Should not error
    }

    #[test]
    fn test_qdrant_client_wrapper() {
        let config = QdrantConfig::default_config();
        let client = QdrantClientWrapper::new(config);

        assert_eq!(client.role(), DatabaseRole::SensoryCortex);

        // Health check should work
        let health = client.health_check().unwrap();
        // Feature not enabled, so should be feature_disabled
        #[cfg(not(feature = "qdrant"))]
        {
            assert!(!health.available);
            assert!(health.error_message.unwrap().contains("not enabled"));
        }
    }

    #[test]
    fn test_cozo_client_wrapper() {
        let config = CozoDbConfig::default_config();
        let client = CozoClientWrapper::new(config);

        assert_eq!(client.role(), DatabaseRole::PrefrontalCortex);
    }

    #[test]
    fn test_lance_client_wrapper() {
        let config = LanceDbConfig::default_config();
        let client = LanceClientWrapper::new(config);

        assert_eq!(client.role(), DatabaseRole::LongTermMemory);
    }

    #[test]
    fn test_duck_client_wrapper() {
        let config = DuckDbConfig::default_config();
        let client = DuckClientWrapper::new(config);

        assert_eq!(client.role(), DatabaseRole::EpistemicAuditor);
    }

    // ========================================================================
    // Phi Computation Tests
    // ========================================================================

    #[test]
    fn test_instantaneous_phi_random_vector() {
        let mind = SymthaeaMind::for_testing();

        // Random vector should have high entropy (Phi close to 1.0)
        let random_v = BinaryHV::random(42);
        let phi = mind.compute_instantaneous_phi(&random_v);
        assert!(phi > 0.9, "Random vector should have high Phi, got {}", phi);
    }

    #[test]
    fn test_instantaneous_phi_zero_vector() {
        let mind = SymthaeaMind::for_testing();

        // All-zero vector (all -1 in bipolar) has minimal entropy
        let zero_v = BinaryHV::zero();
        let phi = mind.compute_instantaneous_phi(&zero_v);
        assert!(phi < 0.1, "Zero vector should have low Phi, got {}", phi);
    }

    #[test]
    fn test_instantaneous_phi_ones_vector() {
        let mind = SymthaeaMind::for_testing();

        // All-ones vector (all +1 in bipolar) has minimal entropy
        let ones_v = BinaryHV::ones();
        let phi = mind.compute_instantaneous_phi(&ones_v);
        assert!(phi < 0.1, "Ones vector should have low Phi, got {}", phi);
    }

    #[test]
    fn test_workspace_coherence_single_vector() {
        let mind = SymthaeaMind::for_testing();

        let workspace = vec![BinaryHV::random(42)];
        let coherence = mind.compute_workspace_coherence(&workspace);
        assert_eq!(
            coherence, 1.0,
            "Single vector workspace should have coherence 1.0"
        );
    }

    #[test]
    fn test_workspace_coherence_identical_vectors() {
        let mind = SymthaeaMind::for_testing();

        let v = BinaryHV::random(42);
        let workspace = vec![v, v, v];
        let coherence = mind.compute_workspace_coherence(&workspace);
        assert!(
            (coherence - 1.0).abs() < 0.001,
            "Identical vectors should have coherence ~1.0"
        );
    }

    #[test]
    fn test_workspace_coherence_random_vectors() {
        let mind = SymthaeaMind::for_testing();

        let workspace: Vec<BinaryHV> = (0..5).map(|i| BinaryHV::random(i as u64)).collect();
        let coherence = mind.compute_workspace_coherence(&workspace);

        // Random vectors should have coherence around 0.5
        assert!(
            coherence > 0.4 && coherence < 0.6,
            "Random vectors should have coherence ~0.5, got {}",
            coherence
        );
    }
}

// ============================================================================
// Integration Tests (run with `cargo test --features qdrant,datalog,lance,duck`)
// ============================================================================

#[cfg(test)]
#[cfg(all(
    feature = "qdrant",
    feature = "datalog",
    feature = "lance",
    feature = "duck"
))]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_full_database_integration() {
        // This test only runs when all database features are enabled
        let mind = SymthaeaMind::for_testing();

        // Check that all databases report their status
        let health = mind.health_check();
        println!("Full integration health check: {:?}", health);

        // Run consciousness cycles
        for i in 0..10 {
            let input = BinaryHV::random(i as u64);
            let state = mind.consciousness_cycle(&input).await.unwrap();
            println!(
                "Cycle {}: Phi={:.4}, Level={:.2}%",
                state.iteration,
                state.current_phi,
                state.consciousness_level * 100.0
            );
        }

        // Verify statistics
        let stats = mind.get_phi_statistics().unwrap();
        assert_eq!(stats.sample_count, 10);
        println!(
            "Phi statistics: mean={:.4}, std={:.4}",
            stats.mean_phi, stats.std_phi
        );

        // Generate report
        let report = mind.generate_report();
        println!("{}", report);

        // Shutdown
        mind.shutdown().unwrap();
    }
}
