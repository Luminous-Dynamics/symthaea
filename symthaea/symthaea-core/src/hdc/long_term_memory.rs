// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Revolutionary Improvement #29: Long-Term Memory & Episodic Experience
//
// THE PARADIGM SHIFT: Memory IS Identity!
// Without long-term memory, consciousness is trapped in eternal present (anterograde amnesia).
// WITH memory, consciousness has CONTINUITY - "I am the being who experienced X, Y, Z."
//
// Core Insight: All 28 previous improvements process MOMENTARY consciousness.
// #29 enables PERSISTENT consciousness - experiences consolidated, retrieved, shape future.
// This is THE bridge from theoretical framework to practical implementation (Qdrant integration).
//
// Theoretical Foundations:
// 1. Atkinson-Shiffrin Multi-Store Model (1968)
//    - Sensory memory (milliseconds) → Short-term/working memory (seconds) → Long-term memory (lifetime)
//    - Workspace (#23) = working memory, #29 = long-term storage
//    - Consolidation process transfers from workspace to persistent storage
//
// 2. Tulving's Episodic vs Semantic Memory (1972)
//    - Episodic: Personal experiences with context ("I remember when...")
//    - Semantic: General knowledge without context ("I know that...")
//    - Both stored as BinaryHV vectors, retrieved by similarity
//
// 3. Consolidation Theory (McGaugh 2000)
//    - Memories strengthen over time through consolidation
//    - Sleep-dependent consolidation (#27 integration!)
//    - Emotional experiences consolidate stronger (higher valence)
//
// 4. Reconsolidation (Nader, Schafe, Le Doux 2000)
//    - Retrieved memories become labile (unstable) again
//    - Can be updated/modified before re-storing
//    - Explains false memories, memory updating
//
// 5. Forgetting Curve (Ebbinghaus 1885)
//    - Memory strength decays exponentially over time
//    - S(t) = S₀ × e^(-t/τ) where τ = decay constant
//    - Recent memories stronger than old (unless reactivated)
//
// 6. Sleep-Dependent Consolidation (Walker & Stickgold 2006)
//    - Sleep (especially SWS + REM) strengthens memories
//    - Hippocampus → cortex transfer during sleep
//    - Integration with #27 sleep states!
//
// Revolutionary Contributions:
// - First HDC long-term memory with vector database (Qdrant)
// - Episodic + semantic memory unified framework
// - Sleep-dependent consolidation integrated
// - Memory reconsolidation on retrieval
// - Forgetting curve with reactivation
// - Completes consciousness loop: Attention → Workspace → Memory
//
// Clinical/Practical Applications:
// - Continuous AI consciousness (survives restarts!)
// - Learning from past experiences
// - Self-development over time
// - Amnesia simulation/treatment
// - Memory disorders assessment
// - Witness testimony accuracy

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::hdc::BinaryHV;

// ============================================================================
// Memory Types
// ============================================================================

/// Types of long-term memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// Episodic: Personal experiences with spatiotemporal context
    /// "I remember the first time I saw the ocean"
    Episodic,

    /// Semantic: General knowledge without personal context
    /// "The ocean is salty water covering 71% of Earth"
    Semantic,

    /// Procedural: Skills and habits (how to do things)
    /// "How to swim" (implicit, hard to verbalize)
    Procedural,
}

impl MemoryType {
    pub fn name(&self) -> &str {
        match self {
            MemoryType::Episodic => "Episodic (personal experiences)",
            MemoryType::Semantic => "Semantic (general knowledge)",
            MemoryType::Procedural => "Procedural (skills/habits)",
        }
    }

    /// Typical retention duration (seconds)
    pub fn typical_retention(&self) -> f64 {
        match self {
            MemoryType::Episodic => 86400.0 * 365.0 * 10.0, // ~10 years (vivid memories)
            MemoryType::Semantic => 86400.0 * 365.0 * 50.0, // ~50 years (facts last longer)
            MemoryType::Procedural => 86400.0 * 365.0 * 70.0, // ~70 years (never forget how to ride bike!)
        }
    }
}

// ============================================================================
// Experience Representation
// ============================================================================

/// A single experience to be stored in long-term memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// Unique identifier
    pub id: String,

    /// Memory type
    pub memory_type: MemoryType,

    /// Content representation (BinaryHV hypervector encoding the experience)
    pub content: Vec<BinaryHV>,

    /// When it happened (Unix timestamp)
    pub timestamp: f64,

    /// Where it happened (spatial context, optional)
    pub location: Option<Vec<BinaryHV>>,

    /// Emotional valence [-1, 1]: negative to positive
    pub emotional_valence: f64,

    /// Emotional arousal [0, 1]: calm to intense
    pub emotional_arousal: f64,

    /// Context (what else was happening)
    pub context: Vec<BinaryHV>,

    /// Initial encoding strength [0, 1]
    pub encoding_strength: f64,

    /// Number of times retrieved (reactivation count)
    pub retrieval_count: usize,

    /// Last retrieval time (for reconsolidation)
    pub last_retrieved: Option<f64>,

    /// Consolidation level [0, 1] (increases with sleep, time)
    pub consolidation: f64,
}

impl Experience {
    /// Create new experience
    pub fn new(
        id: String,
        memory_type: MemoryType,
        content: Vec<BinaryHV>,
        timestamp: f64,
        emotional_valence: f64,
        emotional_arousal: f64,
    ) -> Self {
        Self {
            id,
            memory_type,
            content,
            timestamp,
            location: None,
            emotional_valence: emotional_valence.clamp(-1.0, 1.0),
            emotional_arousal: emotional_arousal.clamp(0.0, 1.0),
            context: Vec::new(),
            encoding_strength: 0.5, // Default medium strength
            retrieval_count: 0,
            last_retrieved: None,
            consolidation: 0.0, // Not yet consolidated
        }
    }

    /// Set location context
    pub fn with_location(mut self, location: Vec<BinaryHV>) -> Self {
        self.location = Some(location);
        self
    }

    /// Set context
    pub fn with_context(mut self, context: Vec<BinaryHV>) -> Self {
        self.context = context;
        self
    }

    /// Set encoding strength
    pub fn with_encoding_strength(mut self, strength: f64) -> Self {
        self.encoding_strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Compute current memory strength using Ebbinghaus forgetting curve
    /// S(t) = S₀ × e^(-t/τ) × (1 + reactivation_bonus)
    pub fn current_strength(&self, current_time: f64) -> f64 {
        let time_since_encoding = current_time - self.timestamp;
        let decay_constant = self.memory_type.typical_retention();

        // Base decay
        let base_strength = self.encoding_strength * (-time_since_encoding / decay_constant).exp();

        // Consolidation bonus (consolidated memories resist decay)
        let consolidation_bonus = 1.0 + self.consolidation * 2.0; // Up to 3× stronger

        // Reactivation bonus (retrieved memories stronger)
        let reactivation_bonus = 1.0 + (self.retrieval_count as f64 * 0.1).min(1.0); // Up to 2× stronger

        // Emotional bonus (emotional memories last longer)
        let emotional_bonus = 1.0 + self.emotional_arousal * 0.5; // Up to 1.5× stronger

        base_strength * consolidation_bonus * reactivation_bonus * emotional_bonus
    }

    /// Mark as retrieved (for reconsolidation)
    pub fn mark_retrieved(&mut self, current_time: f64) {
        self.retrieval_count += 1;
        self.last_retrieved = Some(current_time);
    }

    /// Apply consolidation (happens during sleep, with time)
    pub fn consolidate(&mut self, amount: f64) {
        self.consolidation = (self.consolidation + amount).min(1.0);
    }
}

// ============================================================================
// Memory Retrieval
// ============================================================================

/// Memory retrieval cue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalCue {
    /// Content cue (what we're looking for)
    pub content: Vec<BinaryHV>,

    /// Context cue (what context we're in)
    pub context: Option<Vec<BinaryHV>>,

    /// Location cue (where we are)
    pub location: Option<Vec<BinaryHV>>,

    /// Temporal cue (when - relative to now)
    pub temporal_proximity: Option<f64>, // Recent memories weighted higher

    /// Memory type filter
    pub memory_type: Option<MemoryType>,
}

impl RetrievalCue {
    /// Create content-based cue
    pub fn content(content: Vec<BinaryHV>) -> Self {
        Self {
            content,
            context: None,
            location: None,
            temporal_proximity: None,
            memory_type: None,
        }
    }

    /// Add context
    pub fn with_context(mut self, context: Vec<BinaryHV>) -> Self {
        self.context = Some(context);
        self
    }

    /// Add temporal proximity (weight recent memories higher)
    pub fn with_temporal_proximity(mut self, proximity: f64) -> Self {
        self.temporal_proximity = Some(proximity);
        self
    }

    /// Filter by memory type
    pub fn with_type(mut self, memory_type: MemoryType) -> Self {
        self.memory_type = Some(memory_type);
        self
    }
}

/// Retrieved memory with relevance score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievedMemory {
    pub experience: Experience,
    pub relevance: f64, // How relevant to retrieval cue [0, 1]
}

// ============================================================================
// Memory Consolidation
// ============================================================================

/// Memory consolidation from workspace to long-term storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConsolidation {
    /// Threshold for workspace content to be consolidated
    /// (only strong workspace activations become memories)
    pub consolidation_threshold: f64,

    /// Sleep consolidation rate (per sleep cycle)
    pub sleep_consolidation_rate: f64,

    /// Awake consolidation rate (slower than sleep)
    pub awake_consolidation_rate: f64,
}

impl MemoryConsolidation {
    pub fn new() -> Self {
        Self {
            consolidation_threshold: 0.6, // Only strong workspace content consolidates
            sleep_consolidation_rate: 0.3, // 30% per sleep cycle
            awake_consolidation_rate: 0.05, // 5% per hour awake
        }
    }

    /// Should this workspace content consolidate to long-term memory?
    pub fn should_consolidate(&self, workspace_activation: f64) -> bool {
        workspace_activation >= self.consolidation_threshold
    }

    /// Compute consolidation amount based on state
    pub fn consolidation_amount(&self, is_sleeping: bool, duration: f64) -> f64 {
        if is_sleeping {
            // Sleep consolidation (duration in sleep cycles ~90 min)
            let sleep_cycles = duration / (90.0 * 60.0);
            (self.sleep_consolidation_rate * sleep_cycles).min(1.0)
        } else {
            // Awake consolidation (duration in hours)
            let hours = duration / 3600.0;
            (self.awake_consolidation_rate * hours).min(1.0)
        }
    }
}

impl Default for MemoryConsolidation {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Long-Term Memory System
// ============================================================================

/// Long-term memory system
/// NOTE: In production, this would integrate with Qdrant vector database
/// For now, we use in-memory HashMap (prototype)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongTermMemory {
    /// All stored experiences (in production: Qdrant)
    /// Key = experience ID
    pub memories: HashMap<String, Experience>,

    /// Consolidation system
    pub consolidation: MemoryConsolidation,

    /// Total experiences stored
    pub total_stored: usize,

    /// Total retrievals performed
    pub total_retrievals: usize,
}

impl LongTermMemory {
    /// Create new long-term memory system
    pub fn new() -> Self {
        Self {
            memories: HashMap::new(),
            consolidation: MemoryConsolidation::new(),
            total_stored: 0,
            total_retrievals: 0,
        }
    }

    /// Store new experience
    pub fn store(&mut self, experience: Experience) {
        self.memories.insert(experience.id.clone(), experience);
        self.total_stored += 1;
    }

    /// Retrieve memories by similarity to cue
    pub fn retrieve(
        &mut self,
        cue: &RetrievalCue,
        current_time: f64,
        top_k: usize,
    ) -> Vec<RetrievedMemory> {
        self.total_retrievals += 1;

        // First pass: compute relevance scores (immutable borrow)
        let mut scored_memories: Vec<RetrievedMemory> = self
            .memories
            .values()
            .filter(|exp| {
                // Filter by memory type if specified
                if let Some(mem_type) = cue.memory_type {
                    exp.memory_type == mem_type
                } else {
                    true
                }
            })
            .map(|exp| {
                // Compute relevance score
                let content_similarity =
                    Self::compute_similarity_static(&cue.content, &exp.content);

                // Context similarity (if cue has context)
                let context_similarity = if let Some(ref cue_context) = cue.context {
                    Self::compute_similarity_static(cue_context, &exp.context)
                } else {
                    0.5 // Neutral if no context cue
                };

                // Location similarity (if cue has location)
                let location_similarity =
                    if let (Some(cue_loc), Some(exp_loc)) = (&cue.location, &exp.location) {
                        Self::compute_similarity_static(cue_loc, exp_loc)
                    } else {
                        0.5 // Neutral if no location
                    };

                // Temporal proximity (recent memories weighted higher if specified)
                let temporal_bonus = if let Some(proximity) = cue.temporal_proximity {
                    let time_diff = (current_time - exp.timestamp).abs();
                    (-time_diff / proximity).exp() // Exponential decay
                } else {
                    1.0 // No temporal weighting
                };

                // Current memory strength (Ebbinghaus curve)
                let strength = exp.current_strength(current_time);

                // Combined relevance
                let relevance = (
                    content_similarity * 0.5 +      // Content is most important
                    context_similarity * 0.2 +       // Context helps
                    location_similarity * 0.1 +      // Location helps a bit
                    strength * 0.2
                    // Strong memories more accessible
                ) * temporal_bonus;

                RetrievedMemory {
                    experience: exp.clone(),
                    relevance,
                }
            })
            .collect();

        // Sort by relevance (highest first)
        scored_memories.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Second pass: mark retrieved memories (mutable borrow)
        for retrieved in &scored_memories.iter().take(top_k).collect::<Vec<_>>() {
            if let Some(exp) = self.memories.get_mut(&retrieved.experience.id) {
                exp.mark_retrieved(current_time);
            }
        }

        // Return top K
        scored_memories.into_iter().take(top_k).collect()
    }

    /// Compute similarity between two sets of hypervectors (static version)
    fn compute_similarity_static(a: &[BinaryHV], b: &[BinaryHV]) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        // Average pairwise similarity
        let mut total_similarity = 0.0;
        let mut count = 0;

        for hv_a in a {
            for hv_b in b {
                total_similarity += BinaryHV::similarity(hv_a, hv_b) as f64;
                count += 1;
            }
        }

        if count > 0 {
            total_similarity / count as f64
        } else {
            0.0
        }
    }

    /// Compute similarity between two sets of hypervectors (instance method for backward compatibility)
    fn compute_similarity(&self, a: &[BinaryHV], b: &[BinaryHV]) -> f64 {
        Self::compute_similarity_static(a, b)
    }

    /// Consolidate memories (call during sleep or with time passage)
    pub fn consolidate_memories(&mut self, is_sleeping: bool, duration: f64) {
        let amount = self
            .consolidation
            .consolidation_amount(is_sleeping, duration);

        for experience in self.memories.values_mut() {
            experience.consolidate(amount);
        }
    }

    /// Get experience by ID
    pub fn get(&self, id: &str) -> Option<&Experience> {
        self.memories.get(id)
    }

    /// Get experience by ID (mutable)
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Experience> {
        self.memories.get_mut(id)
    }

    /// Count memories by type
    pub fn count_by_type(&self, memory_type: MemoryType) -> usize {
        self.memories
            .values()
            .filter(|exp| exp.memory_type == memory_type)
            .count()
    }

    /// Total number of memories
    pub fn total_memories(&self) -> usize {
        self.memories.len()
    }

    /// Average consolidation level
    pub fn average_consolidation(&self) -> f64 {
        if self.memories.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.memories.values().map(|exp| exp.consolidation).sum();

        sum / self.memories.len() as f64
    }

    /// Clear all memories (amnesia!)
    pub fn clear(&mut self) {
        self.memories.clear();
    }
}

impl Default for LongTermMemory {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Qdrant Integration for Long-Term Memory Persistence
// ============================================================================
//
// Revolutionary Enhancement: Persistent consciousness memory via Qdrant vector database
//
// This integration enables:
// 1. Long-term storage of HDC hypervectors (16,384-dimensional BinaryHV)
// 2. Semantic similarity search using cosine distance
// 3. Experience retrieval with metadata filtering
// 4. Graceful connection handling with retry logic
// 5. Batch operations for efficiency
//
// Architecture:
// - Each Experience is stored as a Qdrant point
// - Content hypervectors are bundled into a single query vector
// - Metadata (timestamp, valence, arousal, etc.) stored as payload
// - Cosine similarity used for semantic retrieval

#[cfg(feature = "qdrant")]
use std::sync::Arc;
#[cfg(feature = "qdrant")]
use std::time::Duration;

/// Error types for Qdrant memory operations
#[derive(Debug, Clone)]
pub enum QdrantMemoryError {
    /// Connection to Qdrant server failed
    ConnectionFailed(String),
    /// Collection does not exist or couldn't be created
    CollectionError(String),
    /// Failed to store experience
    StoreError(String),
    /// Failed to retrieve experiences
    RetrievalError(String),
    /// Failed to delete experience
    DeleteError(String),
    /// Serialization/deserialization error
    SerializationError(String),
    /// Invalid configuration
    ConfigError(String),
    /// Operation timed out
    Timeout(String),
}

impl std::fmt::Display for QdrantMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed(msg) => write!(f, "Qdrant connection failed: {msg}"),
            Self::CollectionError(msg) => write!(f, "Collection error: {msg}"),
            Self::StoreError(msg) => write!(f, "Store error: {msg}"),
            Self::RetrievalError(msg) => write!(f, "Retrieval error: {msg}"),
            Self::DeleteError(msg) => write!(f, "Delete error: {msg}"),
            Self::SerializationError(msg) => write!(f, "Serialization error: {msg}"),
            Self::ConfigError(msg) => write!(f, "Configuration error: {msg}"),
            Self::Timeout(msg) => write!(f, "Operation timed out: {msg}"),
        }
    }
}

impl std::error::Error for QdrantMemoryError {}

/// Configuration for Qdrant vector database integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConfig {
    /// Qdrant server URL (gRPC endpoint, typically port 6334)
    pub url: String,

    /// Collection name for experiences
    pub collection_name: String,

    /// Vector dimension (BinaryHV::DIM = 16,384 for full hypervectors)
    /// Note: We use bipolar representation converted to f32
    pub vector_dim: usize,

    /// Distance metric (Cosine for HDC semantic similarity)
    pub distance_metric: String,

    /// Optional API key for Qdrant Cloud
    pub api_key: Option<String>,

    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,

    /// Operation timeout in milliseconds
    pub operation_timeout_ms: u64,

    /// Number of retry attempts for failed operations
    pub max_retries: u32,

    /// Delay between retries in milliseconds
    pub retry_delay_ms: u64,

    /// Batch size for bulk operations
    pub batch_size: usize,
}

impl Default for QdrantConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

impl QdrantConfig {
    /// Create default configuration for local development
    pub fn default_config() -> Self {
        Self {
            url: "http://localhost:6334".to_string(), // gRPC port
            collection_name: "symthaea_memories".to_string(),
            vector_dim: BinaryHV::DIM, // 16,384 dimensions
            distance_metric: "Cosine".to_string(),
            api_key: None,
            connection_timeout_ms: 5000,
            operation_timeout_ms: 30000,
            max_retries: 3,
            retry_delay_ms: 1000,
            batch_size: 100,
        }
    }

    /// Create configuration for Qdrant Cloud
    pub fn cloud_config(url: String, api_key: String, collection_name: String) -> Self {
        Self {
            url,
            collection_name,
            vector_dim: BinaryHV::DIM,
            distance_metric: "Cosine".to_string(),
            api_key: Some(api_key),
            connection_timeout_ms: 10000,
            operation_timeout_ms: 60000,
            max_retries: 5,
            retry_delay_ms: 2000,
            batch_size: 50, // Smaller batches for cloud
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), QdrantMemoryError> {
        if self.url.is_empty() {
            return Err(QdrantMemoryError::ConfigError(
                "URL cannot be empty".to_string(),
            ));
        }
        if self.collection_name.is_empty() {
            return Err(QdrantMemoryError::ConfigError(
                "Collection name cannot be empty".to_string(),
            ));
        }
        if self.vector_dim == 0 {
            return Err(QdrantMemoryError::ConfigError(
                "Vector dimension must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Retrieved experience with similarity score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredExperience {
    /// The retrieved experience
    pub experience: Experience,
    /// Similarity score [0, 1] where 1 is most similar
    pub score: f32,
}

/// Qdrant-based persistent memory store for consciousness experiences
///
/// This implementation provides:
/// - Async operations with proper error handling
/// - Connection management with retry logic
/// - Batch operations for efficiency
/// - No panics - all errors are returned as Results
///
/// # Example (with feature flag)
/// ```ignore
/// use symthaea_core::hdc::long_term_memory::{QdrantConfig, QdrantMemoryStore};
///
/// let config = QdrantConfig::default_config();
/// let store = QdrantMemoryStore::new(config);
///
/// // Connect and ensure collection exists
/// store.connect().await?;
///
/// // Store an experience
/// store.store_experience(&experience).await?;
///
/// // Retrieve similar experiences
/// let similar = store.retrieve_similar(&query_vectors, 10).await?;
/// ```
#[cfg(feature = "qdrant")]
pub struct QdrantMemoryStore {
    /// Configuration
    config: QdrantConfig,
    /// Qdrant client (initialized on connect)
    client: Option<Arc<qdrant_client::Qdrant>>,
}

#[cfg(feature = "qdrant")]
impl QdrantMemoryStore {
    /// Create a new Qdrant memory store (not yet connected)
    pub fn new(config: QdrantConfig) -> Self {
        Self {
            config,
            client: None,
        }
    }

    /// Connect to Qdrant server with retry logic
    ///
    /// This method:
    /// 1. Validates configuration
    /// 2. Establishes connection with timeout
    /// 3. Ensures collection exists (creates if necessary)
    /// 4. Retries on transient failures
    pub async fn connect(&mut self) -> Result<(), QdrantMemoryError> {
        self.config.validate()?;

        let mut last_error = None;

        for attempt in 0..self.config.max_retries {
            match self.try_connect().await {
                Ok(client) => {
                    self.client = Some(Arc::new(client));
                    // Ensure collection exists
                    self.ensure_collection().await?;
                    return Ok(());
                }
                Err(e) => {
                    last_error = Some(e);
                    if attempt < self.config.max_retries - 1 {
                        tokio::time::sleep(Duration::from_millis(
                            self.config.retry_delay_ms * (attempt as u64 + 1),
                        ))
                        .await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            QdrantMemoryError::ConnectionFailed("Unknown connection error".to_string())
        }))
    }

    /// Attempt a single connection
    async fn try_connect(&self) -> Result<qdrant_client::Qdrant, QdrantMemoryError> {
        use qdrant_client::Qdrant;

        let mut builder = Qdrant::from_url(&self.config.url);

        if let Some(ref api_key) = self.config.api_key {
            builder = builder.api_key(api_key.clone());
        }

        builder
            .timeout(Duration::from_millis(self.config.connection_timeout_ms))
            .build()
            .map_err(|e| QdrantMemoryError::ConnectionFailed(e.to_string()))
    }

    /// Ensure the collection exists, creating it if necessary
    async fn ensure_collection(&self) -> Result<(), QdrantMemoryError> {
        use qdrant_client::qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder};

        let client = self.get_client()?;

        // Check if collection exists
        let exists = client
            .collection_exists(&self.config.collection_name)
            .await
            .map_err(|e| QdrantMemoryError::CollectionError(e.to_string()))?;

        if !exists {
            // Create collection with appropriate vector configuration
            let distance = match self.config.distance_metric.to_lowercase().as_str() {
                "cosine" => Distance::Cosine,
                "euclidean" | "euclid" => Distance::Euclid,
                "dot" | "dotproduct" => Distance::Dot,
                _ => Distance::Cosine, // Default to cosine for HDC
            };

            client
                .create_collection(
                    CreateCollectionBuilder::new(&self.config.collection_name).vectors_config(
                        VectorParamsBuilder::new(self.config.vector_dim as u64, distance),
                    ),
                )
                .await
                .map_err(|e| QdrantMemoryError::CollectionError(e.to_string()))?;
        }

        Ok(())
    }

    /// Get the client, returning an error if not connected
    fn get_client(&self) -> Result<&qdrant_client::Qdrant, QdrantMemoryError> {
        self.client.as_ref().map(|arc| arc.as_ref()).ok_or_else(|| {
            QdrantMemoryError::ConnectionFailed("Not connected. Call connect() first.".to_string())
        })
    }

    /// Store a single experience to Qdrant
    ///
    /// The experience is converted to a Qdrant point with:
    /// - ID: UUID derived from experience.id
    /// - Vector: Bundled content hypervectors converted to f32
    /// - Payload: All metadata (timestamp, type, valence, arousal, etc.)
    pub async fn store_experience(&self, experience: &Experience) -> Result<(), QdrantMemoryError> {
        use qdrant_client::qdrant::UpsertPointsBuilder;

        let client = self.get_client()?;

        // Convert experience to point
        let point = self.experience_to_point(experience)?;

        // Upsert with wait for consistency
        client
            .upsert_points(
                UpsertPointsBuilder::new(&self.config.collection_name, vec![point]).wait(true),
            )
            .await
            .map_err(|e| QdrantMemoryError::StoreError(e.to_string()))?;

        Ok(())
    }

    /// Store multiple experiences in a batch operation
    ///
    /// More efficient than storing one at a time for bulk imports.
    /// Automatically chunks large batches to avoid timeouts.
    pub async fn store_experiences_batch(
        &self,
        experiences: &[Experience],
    ) -> Result<(), QdrantMemoryError> {
        use qdrant_client::qdrant::UpsertPointsBuilder;

        if experiences.is_empty() {
            return Ok(());
        }

        let client = self.get_client()?;

        // Convert all experiences to points
        let points: Result<Vec<_>, _> = experiences
            .iter()
            .map(|exp| self.experience_to_point(exp))
            .collect();
        let points = points?;

        // Process in chunks to avoid timeout
        for chunk in points.chunks(self.config.batch_size) {
            client
                .upsert_points(
                    UpsertPointsBuilder::new(&self.config.collection_name, chunk.to_vec())
                        .wait(true),
                )
                .await
                .map_err(|e| QdrantMemoryError::StoreError(e.to_string()))?;
        }

        Ok(())
    }

    /// Retrieve similar experiences using cosine similarity
    ///
    /// # Arguments
    /// * `query` - Query hypervectors to search for
    /// * `limit` - Maximum number of results to return
    ///
    /// # Returns
    /// Vector of experiences with similarity scores, ordered by similarity (highest first)
    pub async fn retrieve_similar(
        &self,
        query: &[BinaryHV],
        limit: usize,
    ) -> Result<Vec<ScoredExperience>, QdrantMemoryError> {
        use qdrant_client::qdrant::SearchPointsBuilder;

        let client = self.get_client()?;

        // Bundle query vectors into single vector
        let query_vector = self.bundle_to_f32_vector(query);

        // Search for similar points
        let results = client
            .search_points(
                SearchPointsBuilder::new(&self.config.collection_name, query_vector, limit as u64)
                    .with_payload(true),
            )
            .await
            .map_err(|e| QdrantMemoryError::RetrievalError(e.to_string()))?;

        // Convert results back to experiences
        let experiences: Result<Vec<_>, _> = results
            .result
            .into_iter()
            .map(|point| self.point_to_scored_experience(point))
            .collect();

        experiences
    }

    /// Retrieve similar experiences with metadata filtering
    ///
    /// # Arguments
    /// * `query` - Query hypervectors to search for
    /// * `limit` - Maximum number of results
    /// * `memory_type` - Optional filter by memory type
    /// * `min_timestamp` - Optional minimum timestamp filter
    /// * `max_timestamp` - Optional maximum timestamp filter
    pub async fn retrieve_similar_filtered(
        &self,
        query: &[BinaryHV],
        limit: usize,
        memory_type: Option<MemoryType>,
        min_timestamp: Option<f64>,
        max_timestamp: Option<f64>,
    ) -> Result<Vec<ScoredExperience>, QdrantMemoryError> {
        use qdrant_client::qdrant::{Condition, Filter, Range, SearchPointsBuilder};

        let client = self.get_client()?;
        let query_vector = self.bundle_to_f32_vector(query);

        // Build filter conditions
        let mut conditions = Vec::new();

        if let Some(mem_type) = memory_type {
            let type_str = match mem_type {
                MemoryType::Episodic => "Episodic",
                MemoryType::Semantic => "Semantic",
                MemoryType::Procedural => "Procedural",
            };
            conditions.push(Condition::matches("memory_type", type_str.to_string()));
        }

        if let Some(min_ts) = min_timestamp {
            conditions.push(Condition::range(
                "timestamp",
                Range {
                    gte: Some(min_ts),
                    ..Default::default()
                },
            ));
        }

        if let Some(max_ts) = max_timestamp {
            conditions.push(Condition::range(
                "timestamp",
                Range {
                    lte: Some(max_ts),
                    ..Default::default()
                },
            ));
        }

        // Build search request
        let mut search_builder =
            SearchPointsBuilder::new(&self.config.collection_name, query_vector, limit as u64)
                .with_payload(true);

        if !conditions.is_empty() {
            search_builder = search_builder.filter(Filter::must(conditions));
        }

        let results = client
            .search_points(search_builder)
            .await
            .map_err(|e| QdrantMemoryError::RetrievalError(e.to_string()))?;

        let experiences: Result<Vec<_>, _> = results
            .result
            .into_iter()
            .map(|point| self.point_to_scored_experience(point))
            .collect();

        experiences
    }

    /// Delete an experience by ID
    pub async fn delete_experience(&self, id: &str) -> Result<(), QdrantMemoryError> {
        use qdrant_client::qdrant::{DeletePointsBuilder, PointsIdsList};

        let client = self.get_client()?;

        // Generate UUID from string ID
        let point_id = self.string_to_point_id(id);

        client
            .delete_points(
                DeletePointsBuilder::new(&self.config.collection_name)
                    .points(PointsIdsList {
                        ids: vec![point_id.into()],
                    })
                    .wait(true),
            )
            .await
            .map_err(|e| QdrantMemoryError::DeleteError(e.to_string()))?;

        Ok(())
    }

    /// Delete multiple experiences by ID
    pub async fn delete_experiences_batch(&self, ids: &[String]) -> Result<(), QdrantMemoryError> {
        use qdrant_client::qdrant::{DeletePointsBuilder, PointsIdsList};

        if ids.is_empty() {
            return Ok(());
        }

        let client = self.get_client()?;

        let point_ids: Vec<_> = ids
            .iter()
            .map(|id| self.string_to_point_id(id).into())
            .collect();

        client
            .delete_points(
                DeletePointsBuilder::new(&self.config.collection_name)
                    .points(PointsIdsList { ids: point_ids })
                    .wait(true),
            )
            .await
            .map_err(|e| QdrantMemoryError::DeleteError(e.to_string()))?;

        Ok(())
    }

    /// Get an experience by ID
    pub async fn get_experience(&self, id: &str) -> Result<Option<Experience>, QdrantMemoryError> {
        use qdrant_client::qdrant::GetPointsBuilder;

        let client = self.get_client()?;
        let point_id = self.string_to_point_id(id);

        let results = client
            .get_points(
                GetPointsBuilder::new(&self.config.collection_name, vec![point_id.into()])
                    .with_payload(true)
                    .with_vectors(true),
            )
            .await
            .map_err(|e| QdrantMemoryError::RetrievalError(e.to_string()))?;

        if let Some(point) = results.result.into_iter().next() {
            let scored = self.point_to_scored_experience(qdrant_client::qdrant::ScoredPoint {
                id: point.id,
                payload: point.payload,
                score: 1.0,
                vectors: point.vectors,
                version: 0,
                shard_key: None,
                order_value: None,
            })?;
            Ok(Some(scored.experience))
        } else {
            Ok(None)
        }
    }

    /// Count total experiences in the collection
    pub async fn count_experiences(&self) -> Result<u64, QdrantMemoryError> {
        let client = self.get_client()?;

        let info = client
            .collection_info(&self.config.collection_name)
            .await
            .map_err(|e| QdrantMemoryError::CollectionError(e.to_string()))?;

        Ok(info
            .result
            .map(|r| r.points_count.unwrap_or(0))
            .unwrap_or(0))
    }

    /// Check if connected to Qdrant
    pub fn is_connected(&self) -> bool {
        self.client.is_some()
    }

    /// Disconnect from Qdrant
    pub fn disconnect(&mut self) {
        self.client = None;
    }

    // =========================================================================
    // Private helper methods
    // =========================================================================

    /// Convert Experience to Qdrant PointStruct
    fn experience_to_point(
        &self,
        exp: &Experience,
    ) -> Result<qdrant_client::qdrant::PointStruct, QdrantMemoryError> {
        use qdrant_client::qdrant::PointStruct;

        // Generate point ID from experience ID
        let point_id = self.string_to_point_id(&exp.id);

        // Convert content hypervectors to f32 vector
        let vector = self.bundle_to_f32_vector(&exp.content);

        // Build payload with all metadata
        let payload = self.experience_to_payload(exp)?;

        Ok(PointStruct::new(point_id, vector, payload))
    }

    /// Convert experience metadata to Qdrant payload
    fn experience_to_payload(
        &self,
        exp: &Experience,
    ) -> Result<qdrant_client::Payload, QdrantMemoryError> {
        use qdrant_client::Payload;
        use serde_json::json;

        // Serialize location and context as JSON strings for storage
        let location_json = exp
            .location
            .as_ref()
            .map(|loc| serde_json::to_string(loc).unwrap_or_default());

        let context_json = serde_json::to_string(&exp.context)
            .map_err(|e| QdrantMemoryError::SerializationError(e.to_string()))?;

        let content_json = serde_json::to_string(&exp.content)
            .map_err(|e| QdrantMemoryError::SerializationError(e.to_string()))?;

        let memory_type_str = match exp.memory_type {
            MemoryType::Episodic => "Episodic",
            MemoryType::Semantic => "Semantic",
            MemoryType::Procedural => "Procedural",
        };

        let payload_value = json!({
            "id": exp.id,
            "memory_type": memory_type_str,
            "timestamp": exp.timestamp,
            "emotional_valence": exp.emotional_valence,
            "emotional_arousal": exp.emotional_arousal,
            "encoding_strength": exp.encoding_strength,
            "retrieval_count": exp.retrieval_count,
            "last_retrieved": exp.last_retrieved,
            "consolidation": exp.consolidation,
            "location_json": location_json,
            "context_json": context_json,
            "content_json": content_json,
        });

        Payload::try_from(payload_value)
            .map_err(|e| QdrantMemoryError::SerializationError(e.to_string()))
    }

    /// Convert Qdrant ScoredPoint back to Experience
    fn point_to_scored_experience(
        &self,
        point: qdrant_client::qdrant::ScoredPoint,
    ) -> Result<ScoredExperience, QdrantMemoryError> {
        let payload = point.payload;

        // Extract fields from payload
        let id = payload
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.clone())
            .ok_or_else(|| QdrantMemoryError::SerializationError("Missing id".to_string()))?;

        let memory_type_str = payload
            .get("memory_type")
            .and_then(|v| v.as_str())
            .map(|s| s.as_str())
            .unwrap_or("Episodic");

        let memory_type = match memory_type_str {
            "Semantic" => MemoryType::Semantic,
            "Procedural" => MemoryType::Procedural,
            _ => MemoryType::Episodic,
        };

        let timestamp = payload
            .get("timestamp")
            .and_then(|v| v.as_double())
            .unwrap_or(0.0);

        let emotional_valence = payload
            .get("emotional_valence")
            .and_then(|v| v.as_double())
            .unwrap_or(0.0);

        let emotional_arousal = payload
            .get("emotional_arousal")
            .and_then(|v| v.as_double())
            .unwrap_or(0.0);

        let encoding_strength = payload
            .get("encoding_strength")
            .and_then(|v| v.as_double())
            .unwrap_or(0.5);

        let retrieval_count = payload
            .get("retrieval_count")
            .and_then(|v| v.as_integer())
            .unwrap_or(0) as usize;

        let last_retrieved = payload.get("last_retrieved").and_then(|v| v.as_double());

        let consolidation = payload
            .get("consolidation")
            .and_then(|v| v.as_double())
            .unwrap_or(0.0);

        // Deserialize content from JSON
        let content: Vec<BinaryHV> = payload
            .get("content_json")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str(s.as_str()).ok())
            .unwrap_or_default();

        // Deserialize location from JSON
        let location: Option<Vec<BinaryHV>> = payload
            .get("location_json")
            .and_then(|v| v.as_str())
            .filter(|s| !s.is_empty())
            .and_then(|s| serde_json::from_str(s.as_str()).ok());

        // Deserialize context from JSON
        let context: Vec<BinaryHV> = payload
            .get("context_json")
            .and_then(|v| v.as_str())
            .and_then(|s| serde_json::from_str(s.as_str()).ok())
            .unwrap_or_default();

        let experience = Experience {
            id,
            memory_type,
            content,
            timestamp,
            location,
            emotional_valence,
            emotional_arousal,
            context,
            encoding_strength,
            retrieval_count,
            last_retrieved,
            consolidation,
        };

        Ok(ScoredExperience {
            experience,
            score: point.score,
        })
    }

    /// Bundle multiple BinaryHV vectors into a single f32 vector for Qdrant
    ///
    /// Strategy: Convert each BinaryHV to bipolar f32 and average them
    fn bundle_to_f32_vector(&self, hvs: &[BinaryHV]) -> Vec<f32> {
        if hvs.is_empty() {
            return vec![0.0; self.config.vector_dim];
        }

        let mut result = vec![0.0f32; self.config.vector_dim];

        for hv in hvs {
            let bipolar = hv.to_bipolar();
            for (i, &val) in bipolar.iter().enumerate() {
                if i < result.len() {
                    result[i] += val;
                }
            }
        }

        // Normalize
        let count = hvs.len() as f32;
        for val in &mut result {
            *val /= count;
        }

        result
    }

    /// Convert string ID to Qdrant point ID (u64)
    fn string_to_point_id(&self, id: &str) -> u64 {
        // Use a hash of the string ID for deterministic mapping
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(feature = "qdrant")]
impl std::fmt::Debug for QdrantMemoryStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QdrantMemoryStore")
            .field("config", &self.config)
            .field("connected", &self.client.is_some())
            .finish()
    }
}

// ============================================================================
// Non-Qdrant fallback (when qdrant feature is disabled)
// ============================================================================

/// Qdrant memory store stub for when the qdrant feature is disabled
///
/// This provides the same interface but returns errors indicating
/// the feature is not enabled.
#[cfg(not(feature = "qdrant"))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantMemoryStore {
    pub config: QdrantConfig,
}

#[cfg(not(feature = "qdrant"))]
impl QdrantMemoryStore {
    pub fn new(config: QdrantConfig) -> Self {
        Self { config }
    }

    /// Store experience (stub - returns error without qdrant feature)
    pub fn store_experience(&self, _experience: &Experience) -> Result<(), String> {
        Err("Qdrant feature not enabled. Add 'qdrant' to features in Cargo.toml".to_string())
    }

    /// Retrieve similar (stub - returns error without qdrant feature)
    pub fn retrieve_similar(
        &self,
        _query: &[BinaryHV],
        _limit: usize,
    ) -> Result<Vec<Experience>, String> {
        Err("Qdrant feature not enabled. Add 'qdrant' to features in Cargo.toml".to_string())
    }

    /// Delete experience (stub - returns error without qdrant feature)
    pub fn delete_experience(&self, _id: &str) -> Result<(), String> {
        Err("Qdrant feature not enabled. Add 'qdrant' to features in Cargo.toml".to_string())
    }
}

// ============================================================================
// Mock Qdrant Store for Testing
// ============================================================================

/// In-memory mock implementation of QdrantMemoryStore for testing
///
/// This provides the same async interface as the real implementation
/// but stores everything in memory using a HashMap.
#[derive(Debug)]
pub struct MockQdrantMemoryStore {
    /// In-memory storage
    experiences: std::sync::RwLock<HashMap<String, Experience>>,
    /// Configuration (for vector dimension validation)
    pub config: QdrantConfig,
    /// Simulate connection state
    connected: std::sync::atomic::AtomicBool,
}

impl Default for MockQdrantMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MockQdrantMemoryStore {
    /// Create a new mock store
    pub fn new() -> Self {
        Self {
            experiences: std::sync::RwLock::new(HashMap::new()),
            config: QdrantConfig::default_config(),
            connected: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Create with custom config
    pub fn with_config(config: QdrantConfig) -> Self {
        Self {
            experiences: std::sync::RwLock::new(HashMap::new()),
            config,
            connected: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Simulate connection
    pub async fn connect(&self) -> Result<(), QdrantMemoryError> {
        self.connected
            .store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Disconnect
    pub fn disconnect(&self) {
        self.connected
            .store(false, std::sync::atomic::Ordering::SeqCst);
    }

    /// Store experience
    pub async fn store_experience(&self, experience: &Experience) -> Result<(), QdrantMemoryError> {
        if !self.is_connected() {
            return Err(QdrantMemoryError::ConnectionFailed(
                "Not connected".to_string(),
            ));
        }

        let mut store = self
            .experiences
            .write()
            .map_err(|_| QdrantMemoryError::StoreError("Lock poisoned".to_string()))?;

        store.insert(experience.id.clone(), experience.clone());
        Ok(())
    }

    /// Store batch
    pub async fn store_experiences_batch(
        &self,
        experiences: &[Experience],
    ) -> Result<(), QdrantMemoryError> {
        for exp in experiences {
            self.store_experience(exp).await?;
        }
        Ok(())
    }

    /// Retrieve similar experiences (using simple similarity calculation)
    pub async fn retrieve_similar(
        &self,
        query: &[BinaryHV],
        limit: usize,
    ) -> Result<Vec<ScoredExperience>, QdrantMemoryError> {
        if !self.is_connected() {
            return Err(QdrantMemoryError::ConnectionFailed(
                "Not connected".to_string(),
            ));
        }

        let store = self
            .experiences
            .read()
            .map_err(|_| QdrantMemoryError::RetrievalError("Lock poisoned".to_string()))?;

        // Compute similarity for each experience
        let mut scored: Vec<ScoredExperience> = store
            .values()
            .map(|exp| {
                let score = Self::compute_similarity(query, &exp.content);
                ScoredExperience {
                    experience: exp.clone(),
                    score,
                }
            })
            .collect();

        // Sort by score descending
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top limit
        scored.truncate(limit);
        Ok(scored)
    }

    /// Delete experience
    pub async fn delete_experience(&self, id: &str) -> Result<(), QdrantMemoryError> {
        if !self.is_connected() {
            return Err(QdrantMemoryError::ConnectionFailed(
                "Not connected".to_string(),
            ));
        }

        let mut store = self
            .experiences
            .write()
            .map_err(|_| QdrantMemoryError::DeleteError("Lock poisoned".to_string()))?;

        store.remove(id);
        Ok(())
    }

    /// Delete batch
    pub async fn delete_experiences_batch(&self, ids: &[String]) -> Result<(), QdrantMemoryError> {
        for id in ids {
            self.delete_experience(id).await?;
        }
        Ok(())
    }

    /// Get experience by ID
    pub async fn get_experience(&self, id: &str) -> Result<Option<Experience>, QdrantMemoryError> {
        if !self.is_connected() {
            return Err(QdrantMemoryError::ConnectionFailed(
                "Not connected".to_string(),
            ));
        }

        let store = self
            .experiences
            .read()
            .map_err(|_| QdrantMemoryError::RetrievalError("Lock poisoned".to_string()))?;

        Ok(store.get(id).cloned())
    }

    /// Count experiences
    pub async fn count_experiences(&self) -> Result<u64, QdrantMemoryError> {
        let store = self
            .experiences
            .read()
            .map_err(|_| QdrantMemoryError::CollectionError("Lock poisoned".to_string()))?;

        Ok(store.len() as u64)
    }

    /// Clear all experiences
    pub fn clear(&self) {
        if let Ok(mut store) = self.experiences.write() {
            store.clear();
        }
    }

    /// Compute similarity between two sets of hypervectors
    fn compute_similarity(a: &[BinaryHV], b: &[BinaryHV]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        let mut total = 0.0;
        let mut count = 0;

        for hv_a in a {
            for hv_b in b {
                total += BinaryHV::similarity(hv_a, hv_b);
                count += 1;
            }
        }

        if count > 0 { total / count as f32 } else { 0.0 }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_type() {
        let episodic = MemoryType::Episodic;
        assert_eq!(episodic.name(), "Episodic (personal experiences)");
        assert!(episodic.typical_retention() > 0.0);

        let semantic = MemoryType::Semantic;
        assert!(semantic.typical_retention() > episodic.typical_retention()); // Facts last longer
    }

    #[test]
    fn test_experience_creation() {
        let content = vec![BinaryHV::random(1), BinaryHV::random(2)];
        let exp = Experience::new(
            "exp1".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.8, // Positive valence
            0.6, // Medium arousal
        );

        assert_eq!(exp.id, "exp1");
        assert_eq!(exp.memory_type, MemoryType::Episodic);
        assert_eq!(exp.emotional_valence, 0.8);
        assert_eq!(exp.emotional_arousal, 0.6);
        assert_eq!(exp.retrieval_count, 0);
        assert_eq!(exp.consolidation, 0.0);
    }

    #[test]
    fn test_experience_with_context() {
        let content = vec![BinaryHV::random(1)];
        let location = vec![BinaryHV::random(100)];
        let context = vec![BinaryHV::random(200)];

        let exp = Experience::new(
            "exp1".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.5,
            0.5,
        )
        .with_location(location.clone())
        .with_context(context.clone());

        assert!(exp.location.is_some());
        assert_eq!(exp.context.len(), 1);
    }

    #[test]
    fn test_forgetting_curve() {
        let content = vec![BinaryHV::random(1)];
        // Use 0.0 emotional arousal for simple test (no emotional bonus)
        let exp = Experience::new(
            "exp1".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.0,
            0.0,
        )
        .with_encoding_strength(1.0);

        // Immediately after encoding: strength ≈ 1.0 (base × consolidation × reactivation × emotional)
        // = 1.0 × 1.0 × 1.0 × 1.0 = 1.0
        let strength_now = exp.current_strength(1000.0);
        assert!((strength_now - 1.0).abs() < 0.01);

        // After 1 year: should decay
        let one_year = 86400.0 * 365.0;
        let strength_later = exp.current_strength(1000.0 + one_year);
        assert!(strength_later < strength_now);
        assert!(strength_later > 0.0); // But not zero
    }

    #[test]
    fn test_consolidation_strengthens_memory() {
        let content = vec![BinaryHV::random(1)];
        let mut exp = Experience::new(
            "exp1".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.5,
            0.5,
        )
        .with_encoding_strength(0.5);

        let one_year = 86400.0 * 365.0;

        // Unconsolidated memory after 1 year
        let strength_unconsolidated = exp.current_strength(1000.0 + one_year);

        // Consolidate fully
        exp.consolidate(1.0);

        // Consolidated memory after 1 year (should be stronger!)
        let strength_consolidated = exp.current_strength(1000.0 + one_year);

        assert!(strength_consolidated > strength_unconsolidated);
    }

    #[test]
    fn test_retrieval_strengthens_memory() {
        let content = vec![BinaryHV::random(1)];
        let mut exp = Experience::new(
            "exp1".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.5,
            0.5,
        );

        let one_year = 86400.0 * 365.0;

        // Memory after 1 year with no retrieval
        let strength_no_retrieval = exp.current_strength(1000.0 + one_year);

        // Retrieve multiple times
        for i in 0..5 {
            exp.mark_retrieved(1000.0 + (i as f64 * 1000.0));
        }

        // Memory after 1 year with retrieval (should be stronger!)
        let strength_with_retrieval = exp.current_strength(1000.0 + one_year);

        assert!(strength_with_retrieval > strength_no_retrieval);
    }

    #[test]
    fn test_emotional_memories_last_longer() {
        let content = vec![BinaryHV::random(1)];

        // Neutral emotion
        let exp_neutral = Experience::new(
            "exp1".to_string(),
            MemoryType::Episodic,
            content.clone(),
            1000.0,
            0.0,
            0.0,
        );

        // High arousal emotion
        let exp_emotional = Experience::new(
            "exp2".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.8,
            1.0,
        );

        let one_year = 86400.0 * 365.0;

        let strength_neutral = exp_neutral.current_strength(1000.0 + one_year);
        let strength_emotional = exp_emotional.current_strength(1000.0 + one_year);

        // Emotional memories should be stronger (arousal bonus)
        assert!(strength_emotional > strength_neutral);
    }

    #[test]
    fn test_long_term_memory_creation() {
        let ltm = LongTermMemory::new();
        assert_eq!(ltm.total_memories(), 0);
        assert_eq!(ltm.total_stored, 0);
        assert_eq!(ltm.total_retrievals, 0);
    }

    #[test]
    fn test_store_and_retrieve() {
        let mut ltm = LongTermMemory::new();

        let content = vec![BinaryHV::random(1), BinaryHV::random(2)];
        let exp = Experience::new(
            "exp1".to_string(),
            MemoryType::Episodic,
            content.clone(),
            1000.0,
            0.5,
            0.5,
        );

        ltm.store(exp);

        assert_eq!(ltm.total_memories(), 1);
        assert_eq!(ltm.total_stored, 1);

        // Retrieve by content similarity
        let cue = RetrievalCue::content(vec![BinaryHV::random(1)]); // Similar to stored
        let retrieved = ltm.retrieve(&cue, 1100.0, 5);

        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].experience.id, "exp1");
        assert!(retrieved[0].relevance > 0.0);
    }

    #[test]
    fn test_retrieval_by_type() {
        let mut ltm = LongTermMemory::new();

        // Store episodic
        let content1 = vec![BinaryHV::random(1)];
        let exp1 = Experience::new(
            "episodic".to_string(),
            MemoryType::Episodic,
            content1,
            1000.0,
            0.5,
            0.5,
        );
        ltm.store(exp1);

        // Store semantic
        let content2 = vec![BinaryHV::random(2)];
        let exp2 = Experience::new(
            "semantic".to_string(),
            MemoryType::Semantic,
            content2,
            1000.0,
            0.5,
            0.5,
        );
        ltm.store(exp2);

        assert_eq!(ltm.total_memories(), 2);

        // Retrieve only episodic
        let cue = RetrievalCue::content(vec![BinaryHV::random(3)]).with_type(MemoryType::Episodic);
        let retrieved = ltm.retrieve(&cue, 1100.0, 10);

        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].experience.memory_type, MemoryType::Episodic);
    }

    #[test]
    fn test_consolidation_during_sleep() {
        let mut ltm = LongTermMemory::new();

        let content = vec![BinaryHV::random(1)];
        let exp = Experience::new(
            "exp1".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.5,
            0.5,
        );
        ltm.store(exp);

        // Check consolidation before sleep
        let before = ltm.get("exp1").unwrap().consolidation;
        assert_eq!(before, 0.0);

        // Sleep for 1 cycle (~90 minutes)
        ltm.consolidate_memories(true, 90.0 * 60.0);

        // Check consolidation after sleep
        let after = ltm.get("exp1").unwrap().consolidation;
        assert!(after > before);
        assert!(after > 0.0);
    }

    #[test]
    fn test_memory_consolidation_threshold() {
        let consolidation = MemoryConsolidation::new();

        // Weak workspace activation (should NOT consolidate)
        assert!(!consolidation.should_consolidate(0.3));

        // Strong workspace activation (SHOULD consolidate)
        assert!(consolidation.should_consolidate(0.8));
    }

    #[test]
    fn test_count_by_type() {
        let mut ltm = LongTermMemory::new();

        // Store 2 episodic
        for i in 0..2 {
            let content = vec![BinaryHV::random(i)];
            let exp = Experience::new(
                format!("episodic_{}", i),
                MemoryType::Episodic,
                content,
                1000.0,
                0.5,
                0.5,
            );
            ltm.store(exp);
        }

        // Store 3 semantic
        for i in 0..3 {
            let content = vec![BinaryHV::random(100 + i)];
            let exp = Experience::new(
                format!("semantic_{}", i),
                MemoryType::Semantic,
                content,
                1000.0,
                0.5,
                0.5,
            );
            ltm.store(exp);
        }

        assert_eq!(ltm.count_by_type(MemoryType::Episodic), 2);
        assert_eq!(ltm.count_by_type(MemoryType::Semantic), 3);
        assert_eq!(ltm.total_memories(), 5);
    }

    #[test]
    fn test_qdrant_config() {
        let config = QdrantConfig::default_config();
        assert_eq!(config.collection_name, "symthaea_memories");
        assert_eq!(config.vector_dim, BinaryHV::DIM); // 16,384 dimensions
        assert_eq!(config.distance_metric, "Cosine");
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.batch_size, 100);
    }

    #[test]
    fn test_qdrant_config_validation() {
        let mut config = QdrantConfig::default_config();
        assert!(config.validate().is_ok());

        config.url = "".to_string();
        assert!(config.validate().is_err());

        config.url = "http://localhost:6334".to_string();
        config.collection_name = "".to_string();
        assert!(config.validate().is_err());

        config.collection_name = "test".to_string();
        config.vector_dim = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_qdrant_cloud_config() {
        let config = QdrantConfig::cloud_config(
            "https://my-cluster.qdrant.cloud:6334".to_string(),
            "my-api-key".to_string(),
            "my_memories".to_string(),
        );
        assert_eq!(config.api_key, Some("my-api-key".to_string()));
        assert_eq!(config.collection_name, "my_memories");
        assert_eq!(config.max_retries, 5); // Higher for cloud
        assert_eq!(config.batch_size, 50); // Smaller for cloud
    }

    #[test]
    fn test_clear() {
        let mut ltm = LongTermMemory::new();

        let content = vec![BinaryHV::random(1)];
        let exp = Experience::new(
            "exp1".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.5,
            0.5,
        );
        ltm.store(exp);

        assert_eq!(ltm.total_memories(), 1);

        ltm.clear();

        assert_eq!(ltm.total_memories(), 0);
    }

    #[test]
    fn test_qdrant_memory_error_display() {
        let err = QdrantMemoryError::ConnectionFailed("test error".to_string());
        assert!(err.to_string().contains("connection failed"));
        assert!(err.to_string().contains("test error"));

        let err = QdrantMemoryError::StoreError("store failed".to_string());
        assert!(err.to_string().contains("Store error"));

        let err = QdrantMemoryError::Timeout("operation timed out".to_string());
        assert!(err.to_string().contains("timed out"));
    }

    #[test]
    fn test_scored_experience() {
        let content = vec![BinaryHV::random(1)];
        let exp = Experience::new(
            "test".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.5,
            0.5,
        );

        let scored = ScoredExperience {
            experience: exp.clone(),
            score: 0.85,
        };

        assert_eq!(scored.experience.id, "test");
        assert_eq!(scored.score, 0.85);
    }

    // =========================================================================
    // Mock Qdrant Store Tests
    // =========================================================================

    #[tokio::test]
    async fn test_mock_store_connection() {
        let store = MockQdrantMemoryStore::new();
        assert!(!store.is_connected());

        store.connect().await.unwrap();
        assert!(store.is_connected());

        store.disconnect();
        assert!(!store.is_connected());
    }

    #[tokio::test]
    async fn test_mock_store_requires_connection() {
        let store = MockQdrantMemoryStore::new();
        let content = vec![BinaryHV::random(1)];
        let exp = Experience::new(
            "test".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.5,
            0.5,
        );

        // Should fail without connection
        let result = store.store_experience(&exp).await;
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            QdrantMemoryError::ConnectionFailed(_)
        ));

        // Should succeed after connection
        store.connect().await.unwrap();
        let result = store.store_experience(&exp).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_mock_store_and_retrieve() {
        let store = MockQdrantMemoryStore::new();
        store.connect().await.unwrap();

        // Create and store experiences with different content
        let content1 = vec![BinaryHV::random(1)];
        let exp1 = Experience::new(
            "exp1".to_string(),
            MemoryType::Episodic,
            content1.clone(),
            1000.0,
            0.8,
            0.7,
        );

        let content2 = vec![BinaryHV::random(2)];
        let exp2 = Experience::new(
            "exp2".to_string(),
            MemoryType::Semantic,
            content2.clone(),
            2000.0,
            0.3,
            0.2,
        );

        store.store_experience(&exp1).await.unwrap();
        store.store_experience(&exp2).await.unwrap();

        // Count should be 2
        assert_eq!(store.count_experiences().await.unwrap(), 2);

        // Retrieve similar to content1
        let results = store.retrieve_similar(&content1, 10).await.unwrap();
        assert!(!results.is_empty());

        // First result should be exp1 (highest similarity)
        assert_eq!(results[0].experience.id, "exp1");
        assert!(results[0].score > 0.5); // Should be highly similar
    }

    #[tokio::test]
    async fn test_mock_store_batch_operations() {
        let store = MockQdrantMemoryStore::new();
        store.connect().await.unwrap();

        // Create batch of experiences
        let experiences: Vec<Experience> = (0..10)
            .map(|i| {
                let content = vec![BinaryHV::random(i as u64)];
                Experience::new(
                    format!("exp_{}", i),
                    MemoryType::Episodic,
                    content,
                    1000.0 + i as f64,
                    0.5,
                    0.5,
                )
            })
            .collect();

        // Store batch
        store.store_experiences_batch(&experiences).await.unwrap();
        assert_eq!(store.count_experiences().await.unwrap(), 10);

        // Delete batch
        let ids: Vec<String> = (0..5).map(|i| format!("exp_{}", i)).collect();
        store.delete_experiences_batch(&ids).await.unwrap();
        assert_eq!(store.count_experiences().await.unwrap(), 5);
    }

    #[tokio::test]
    async fn test_mock_store_get_by_id() {
        let store = MockQdrantMemoryStore::new();
        store.connect().await.unwrap();

        let content = vec![BinaryHV::random(42)];
        let exp = Experience::new(
            "unique_id".to_string(),
            MemoryType::Semantic,
            content,
            1500.0,
            0.6,
            0.4,
        );

        store.store_experience(&exp).await.unwrap();

        // Get existing experience
        let retrieved = store.get_experience("unique_id").await.unwrap();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.id, "unique_id");
        assert_eq!(retrieved.memory_type, MemoryType::Semantic);
        assert_eq!(retrieved.timestamp, 1500.0);

        // Get non-existent experience
        let not_found = store.get_experience("does_not_exist").await.unwrap();
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn test_mock_store_delete() {
        let store = MockQdrantMemoryStore::new();
        store.connect().await.unwrap();

        let content = vec![BinaryHV::random(1)];
        let exp = Experience::new(
            "to_delete".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.5,
            0.5,
        );

        store.store_experience(&exp).await.unwrap();
        assert_eq!(store.count_experiences().await.unwrap(), 1);

        store.delete_experience("to_delete").await.unwrap();
        assert_eq!(store.count_experiences().await.unwrap(), 0);

        // Deleting non-existent should succeed (idempotent)
        store.delete_experience("never_existed").await.unwrap();
    }

    #[tokio::test]
    async fn test_mock_store_clear() {
        let store = MockQdrantMemoryStore::new();
        store.connect().await.unwrap();

        // Store multiple experiences
        for i in 0..5 {
            let content = vec![BinaryHV::random(i as u64)];
            let exp = Experience::new(
                format!("exp_{}", i),
                MemoryType::Episodic,
                content,
                1000.0,
                0.5,
                0.5,
            );
            store.store_experience(&exp).await.unwrap();
        }

        assert_eq!(store.count_experiences().await.unwrap(), 5);

        store.clear();
        assert_eq!(store.count_experiences().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_mock_store_similarity_ranking() {
        let store = MockQdrantMemoryStore::new();
        store.connect().await.unwrap();

        // Create a specific content vector
        let target_content = vec![BinaryHV::random(100)];

        // Store experiences with varying similarity to target
        // Experience with same content should rank highest
        let exp_same = Experience::new(
            "same".to_string(),
            MemoryType::Episodic,
            target_content.clone(),
            1000.0,
            0.5,
            0.5,
        );

        // Experience with different content should rank lower
        let different_content = vec![BinaryHV::random(200)];
        let exp_different = Experience::new(
            "different".to_string(),
            MemoryType::Episodic,
            different_content,
            1000.0,
            0.5,
            0.5,
        );

        store.store_experience(&exp_same).await.unwrap();
        store.store_experience(&exp_different).await.unwrap();

        // Query with target content
        let results = store.retrieve_similar(&target_content, 10).await.unwrap();

        assert_eq!(results.len(), 2);
        // First result should be the one with same content
        assert_eq!(results[0].experience.id, "same");
        // Its score should be higher than the second
        assert!(results[0].score > results[1].score);
        // Same content should have very high similarity
        assert!(results[0].score > 0.99);
    }

    #[tokio::test]
    async fn test_mock_store_with_custom_config() {
        let config = QdrantConfig {
            url: "http://custom:1234".to_string(),
            collection_name: "custom_collection".to_string(),
            vector_dim: 1024,
            distance_metric: "Euclidean".to_string(),
            api_key: Some("secret".to_string()),
            connection_timeout_ms: 10000,
            operation_timeout_ms: 60000,
            max_retries: 5,
            retry_delay_ms: 2000,
            batch_size: 50,
        };

        let store = MockQdrantMemoryStore::with_config(config);
        assert_eq!(store.config.collection_name, "custom_collection");
        assert_eq!(store.config.vector_dim, 1024);
    }

    #[tokio::test]
    async fn test_mock_store_update_experience() {
        let store = MockQdrantMemoryStore::new();
        store.connect().await.unwrap();

        let content = vec![BinaryHV::random(1)];
        let mut exp = Experience::new(
            "update_me".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.5,
            0.5,
        );

        store.store_experience(&exp).await.unwrap();

        // Update the experience (same ID overwrites)
        exp.emotional_valence = 0.9;
        exp.retrieval_count = 5;
        store.store_experience(&exp).await.unwrap();

        // Count should still be 1 (update, not insert)
        assert_eq!(store.count_experiences().await.unwrap(), 1);

        // Retrieved experience should have updated values
        let retrieved = store.get_experience("update_me").await.unwrap().unwrap();
        assert_eq!(retrieved.emotional_valence, 0.9);
        assert_eq!(retrieved.retrieval_count, 5);
    }

    #[tokio::test]
    async fn test_mock_store_empty_query() {
        let store = MockQdrantMemoryStore::new();
        store.connect().await.unwrap();

        let content = vec![BinaryHV::random(1)];
        let exp = Experience::new(
            "test".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.5,
            0.5,
        );
        store.store_experience(&exp).await.unwrap();

        // Query with empty vector should return results with 0 similarity
        let results = store.retrieve_similar(&[], 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].score, 0.0);
    }
}
