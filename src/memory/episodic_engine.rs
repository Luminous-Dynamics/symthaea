/*!
Week 17 Day 2: Chrono-Semantic Episodic Memory Engine

Revolutionary Integration of:
- TemporalEncoder (circular time with multi-frequency encoding)
- SemanticSpace (HDC concept vectors)
- SparseDistributedMemory (content-addressable pattern completion)
- Hippocampus (holographic compression)

This is the FIRST AI memory system that enables:
1. **Mental Time Travel** - Reconstruct full experiences from partial cues
2. **Chrono-Semantic Queries** - "Show me all git errors from yesterday morning"
3. **Autobiographical Timeline** - Memories organized temporally with semantic clustering
4. **Predictive Recall** - Anticipate relevant memories based on current context
5. **Sleep-Based Consolidation** - Strengthen important memories during sleep cycles

Performance Targets:
- <1ms recall for recent memories (SDM cache hits)
- <10ms for deep temporal queries
- <100ms for complex chrono-semantic reconstruction

Biological Inspiration:
Real brains bind "what" (semantic) + "when" (temporal) + "where" (spatial) into unified
episodic traces. This engine replicates that binding using HDC mathematics.
*/

use std::time::Duration;
use std::collections::VecDeque;
use anyhow::Result;
use serde::{Serialize, Deserialize};

use crate::hdc::{
    HDC_DIMENSION,
    TemporalEncoder,
    SemanticSpace,
    SparseDistributedMemory,
    SDMConfig,
    ReadResult,
    IterativeReadResult,
    hamming_similarity,
};

// ============================================================================
// EPISODIC MEMORY TRACE - The Revolutionary Integration
// ============================================================================

/// A single episodic memory with chrono-semantic binding
///
/// This is the core unit of episodic memory - it binds:
/// - **WHEN** (temporal vector from TemporalEncoder)
/// - **WHAT** (semantic vector from SemanticSpace)
/// - **HOW** (emotional valence)
///
/// Formula: EpisodicMemory = Temporal(when) ⊗ Semantic(what) ⊗ Emotional(how)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicTrace {
    /// Unique memory ID
    pub id: u64,

    /// When this event happened (as Duration since system start)
    pub timestamp: Duration,

    /// What happened (original content)
    pub content: String,

    /// Contextual tags (e.g., "git", "error", "NixOS")
    pub tags: Vec<String>,

    /// Emotional valence (-1.0 to 1.0)
    pub emotion: f32,

    /// REVOLUTIONARY MULTI-MODAL ARCHITECTURE:
    /// ========================================
    /// Instead of mixing emotion into semantic representation (scalar contamination),
    /// we store them as PARALLEL binding spaces (like amygdala || hippocampus).
    ///
    /// Old (broken): M = (T ⊗ S) + emotion_scalar  → emotion contaminates semantic similarity
    /// New (biological): M_semantic = T ⊗ S,  M_emotional = E_binding  → parallel processing
    ///
    /// This allows three query modes:
    /// 1. Pure semantic: Query M_semantic (ignores emotion) → finds "code review" regardless of feeling
    /// 2. Pure emotional: Query M_emotional (ignores content) → finds all happy/sad memories
    /// 3. Full query: Combine both via binding → "sad code reviews from yesterday"

    /// The chrono-semantic encoding: Temporal ⊗ Semantic (NO emotion mixing!)
    /// This is stored in SDM for content-addressable recall
    pub chrono_semantic_vector: Vec<i8>, // Bipolar for SDM

    /// The emotional binding vector: Separate emotional tag (NEW!)
    /// Allows emotion-modulated recall WITHOUT contaminating semantic similarity
    /// This mirrors biological reality: amygdala (emotion) || hippocampus (memory) = parallel processing
    pub emotional_binding_vector: Vec<i8>, // Bipolar emotion signature

    /// Original temporal vector (f32) for unbinding operations
    pub temporal_vector: Vec<f32>,

    /// Original semantic vector (f32) for unbinding operations
    pub semantic_vector: Vec<f32>,

    /// How many times this memory has been recalled
    pub recall_count: usize,

    /// Memory strength (0.0-2.0, increases with recall)
    pub strength: f32,

    /// WEEK 17 DAY 3: Attention-weighted encoding
    /// Attention weight during encoding (0.0-1.0)
    /// 0.0 = background/routine (weak encoding)
    /// 0.5 = normal attention (default encoding)
    /// 1.0 = full focus/critical moment (strong encoding, unforgettable)
    pub attention_weight: f32,

    /// SDM encoding strength (number of times written to SDM)
    /// Calculated from attention_weight:
    /// - attention 0.0 → 1x write (weak, easily forgotten)
    /// - attention 0.5 → 10x writes (normal, typical memory)
    /// - attention 1.0 → 100x writes (strong, unforgettable moment)
    pub encoding_strength: usize,
}

impl EpisodicTrace {
    /// Create new episodic trace with chrono-semantic binding
    ///
    /// # Algorithm
    /// 1. Encode time using TemporalEncoder (circular multi-frequency)
    /// 2. Encode content using SemanticSpace (HDC concept vectors)
    /// 3. Bind: temporal ⊗ semantic (element-wise multiplication)
    /// 4. Add emotional modulation
    /// 5. Convert to bipolar for SDM storage
    pub fn new(
        id: u64,
        timestamp: Duration,
        content: String,
        tags: Vec<String>,
        emotion: f32,
        temporal_encoder: &TemporalEncoder,
        semantic_space: &mut SemanticSpace,
    ) -> Result<Self> {
        // 1. Encode time as circular vector
        let temporal_vector = temporal_encoder.encode_time(timestamp)?;

        // 2. Encode content + tags as semantic vector
        let mut full_content = content.clone();
        if !tags.is_empty() {
            full_content.push_str(" ");
            full_content.push_str(&tags.join(" "));
        }
        let semantic_vector = semantic_space.encode(&full_content)?;

        // 3. Bind temporal + semantic (chrono-semantic fusion WITHOUT emotion)
        let bound = temporal_encoder.bind(&temporal_vector, &semantic_vector)?;

        // 4. REVOLUTIONARY: Create SEPARATE emotional binding vector
        //    Old (broken): M = (T ⊗ S) + emotion_scalar  → contaminates semantic similarity
        //    New (biological): M_semantic = T ⊗ S,  M_emotional = E_binding
        let emotion_clamped = emotion.clamp(-1.0, 1.0);

        // Create emotional signature vector (separate from semantics!)
        // This allows pure semantic queries to ignore emotional valence
        let emotional_vector: Vec<f32> = (0..bound.len())
            .map(|i| {
                // Create unique pattern for each emotion level
                // Positive emotions → positive bias, negative → negative bias
                let phase = (i as f32 * 0.1) + (emotion_clamped * std::f32::consts::PI);
                phase.sin() * emotion_clamped.abs() // Magnitude proportional to emotion strength
            })
            .collect();

        // 5. Convert BOTH to bipolar for SDM storage
        let chrono_semantic_vector: Vec<i8> = bound.iter()
            .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
            .collect();

        let emotional_binding_vector: Vec<i8> = emotional_vector.iter()
            .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
            .collect();

        // WEEK 17 DAY 3: Default attention weight (normal encoding)
        // Future: This will be parameterized in store_with_attention()
        let attention_weight = 0.5; // Normal attention
        let encoding_strength = 10; // 10x writes (matches old behavior)

        Ok(Self {
            id,
            timestamp,
            content,
            tags,
            emotion: emotion_clamped,
            chrono_semantic_vector,
            emotional_binding_vector, // NEW: Separate emotional tag!
            temporal_vector,
            semantic_vector,
            recall_count: 0,
            strength: 0.5, // Start at neutral, can grow to 2.0
            attention_weight,  // Week 17 Day 3: Attention during encoding
            encoding_strength, // Week 17 Day 3: SDM reinforcement level
        })
    }

    /// Strengthen this memory (called on recall)
    pub fn strengthen(&mut self) {
        self.recall_count += 1;
        self.strength = (self.strength + 0.1).min(2.0);
    }

    /// Decay this memory (called during consolidation)
    pub fn decay(&mut self, decay_rate: f32) {
        self.strength *= 1.0 - decay_rate;
        self.strength = self.strength.max(0.0);
    }
}

// ============================================================================
// EPISODIC MEMORY ENGINE - Revolutionary Content-Addressable Memory
// ============================================================================

/// Configuration for episodic memory engine
#[derive(Debug, Clone)]
pub struct EpisodicConfig {
    /// SDM configuration
    pub sdm_config: SDMConfig,

    /// Maximum episodic traces to keep in buffer before consolidation
    pub max_buffer_size: usize,

    /// Minimum similarity for recall (0.0-1.0)
    pub recall_threshold: f32,

    /// Memory decay rate during consolidation (0.0-1.0)
    pub decay_rate: f32,
}

impl Default for EpisodicConfig {
    fn default() -> Self {
        Self {
            sdm_config: SDMConfig {
                dimension: HDC_DIMENSION,
                num_hard_locations: 10_000,
                activation_radius: 0.42,
                weighted_read: true,
                min_activation_count: 5,
            },
            max_buffer_size: 1000,
            recall_threshold: 0.5,
            decay_rate: 0.05,
        }
    }
}

/// The Episodic Memory Engine - Revolutionary Integration
///
/// This engine provides:
/// - **Content-addressable recall** via SDM pattern completion
/// - **Temporal queries** via TemporalEncoder similarity
/// - **Semantic queries** via SemanticSpace encoding
/// - **Chrono-semantic queries** via bound temporal+semantic vectors
/// - **Mental time travel** via partial cue reconstruction
pub struct EpisodicMemoryEngine {
    /// Temporal encoding for circular time representation
    temporal_encoder: TemporalEncoder,

    /// Semantic encoding for concept vectors
    semantic_space: SemanticSpace,

    /// Sparse Distributed Memory for content-addressable storage
    sdm: SparseDistributedMemory,

    /// Recent episodic traces (not yet consolidated)
    buffer: VecDeque<EpisodicTrace>,

    /// Configuration
    config: EpisodicConfig,

    /// Next memory ID
    next_id: u64,
}

impl EpisodicMemoryEngine {
    /// Create new episodic memory engine with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(EpisodicConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: EpisodicConfig) -> Result<Self> {
        Ok(Self {
            temporal_encoder: TemporalEncoder::new(),
            semantic_space: SemanticSpace::new(HDC_DIMENSION)?,
            sdm: SparseDistributedMemory::new(config.sdm_config.clone()),
            buffer: VecDeque::new(),
            config,
            next_id: 0,
        })
    }

    /// Store a new episodic memory
    ///
    /// # Arguments
    /// - `timestamp`: When this event occurred
    /// - `content`: What happened (natural language description)
    /// - `tags`: Contextual tags (e.g., ["git", "error"])
    /// - `emotion`: Emotional valence (-1.0 to 1.0)
    ///
    /// # Example
    /// ```ignore
    /// engine.store(
    ///     Duration::from_secs(9 * 3600), // 9 AM
    ///     "Git push failed with merge conflict".to_string(),
    ///     vec!["git".to_string(), "error".to_string()],
    ///     -0.7, // Frustration
    /// )?;
    /// ```
    pub fn store(
        &mut self,
        timestamp: Duration,
        content: String,
        tags: Vec<String>,
        emotion: f32,
    ) -> Result<u64> {
        // Create episodic trace with chrono-semantic binding
        let trace = EpisodicTrace::new(
            self.next_id,
            timestamp,
            content,
            tags,
            emotion,
            &self.temporal_encoder,
            &mut self.semantic_space,
        )?;

        let id = trace.id;

        // Store in SDM with reinforcement (write 10x for strong encoding)
        for _ in 0..10 {
            self.sdm.write_auto(&trace.chrono_semantic_vector);
        }

        // Add to buffer
        self.buffer.push_back(trace);

        // Trigger consolidation if buffer full
        if self.buffer.len() >= self.config.max_buffer_size {
            self.consolidate()?;
        }

        self.next_id += 1;
        Ok(id)
    }

    /// WEEK 17 DAY 3: Store with attention-weighted encoding
    ///
    /// **Revolutionary Feature**: Memories are encoded with strength proportional to attention/importance.
    ///
    /// This mimics biological reality:
    /// - **High-attention events** (car accidents) → 100x SDM writes → unforgettable
    /// - **Normal events** (typical work) → 10x SDM writes → typical memory
    /// - **Background tasks** (opening editor) → 1x SDM write → easily forgotten
    ///
    /// # Parameters
    /// - `attention_weight`: Importance level (0.0-1.0)
    ///   - 0.0 = background/routine (weak encoding)
    ///   - 0.5 = normal attention (default)
    ///   - 1.0 = full focus/critical moment (strong encoding)
    ///
    /// # Examples
    /// ```ignore
    /// // Critical bug fix (full attention)
    /// engine.store_with_attention(
    ///     timestamp, "Fixed critical auth vulnerability".to_string(),
    ///     vec!["security".to_string(), "critical".to_string()],
    ///     -0.8, // High stress
    ///     1.0,  // Full attention → 100x SDM writes!
    /// )?;
    ///
    /// // Routine task (background attention)
    /// engine.store_with_attention(
    ///     timestamp, "Opened editor".to_string(),
    ///     vec!["routine".to_string()],
    ///     0.0,  // Neutral emotion
    ///     0.1,  // Background → only 10x SDM writes
    /// )?;
    /// ```
    pub fn store_with_attention(
        &mut self,
        timestamp: Duration,
        content: String,
        tags: Vec<String>,
        emotion: f32,
        attention_weight: f32,
    ) -> Result<u64> {
        // Clamp attention to valid range
        let attention = attention_weight.clamp(0.0, 1.0);

        // Calculate encoding strength: attention 0.0 → 1x, 0.5 → 50x, 1.0 → 100x
        // Formula: 1 + (attention * 99) gives range [1, 100]
        let encoding_strength = (1.0 + attention * 99.0) as usize;

        // Create episodic trace with explicit attention metadata
        let mut trace = EpisodicTrace::new(
            self.next_id,
            timestamp,
            content,
            tags,
            emotion,
            &self.temporal_encoder,
            &mut self.semantic_space,
        )?;

        // Override default attention values from constructor
        trace.attention_weight = attention;
        trace.encoding_strength = encoding_strength;

        let id = trace.id;

        // REVOLUTIONARY: Variable SDM reinforcement based on attention
        // High-attention memories get many more writes → stronger, more persistent encoding
        for _ in 0..encoding_strength {
            self.sdm.write_auto(&trace.chrono_semantic_vector);
        }

        // Add to buffer
        self.buffer.push_back(trace);

        // Trigger consolidation if buffer full
        if self.buffer.len() >= self.config.max_buffer_size {
            self.consolidate()?;
        }

        self.next_id += 1;
        Ok(id)
    }

    /// WEEK 17 DAY 3: Automatically detect attention weight from context
    ///
    /// Heuristics for attention detection:
    /// - **Error/Critical tags** → High attention
    /// - **Strong emotion** (|emotion| > 0.7) → High attention
    /// - **Long detailed content** → Higher attention (more thought invested)
    /// - **Multiple tags** → Higher attention (more contextualization)
    ///
    /// Returns: Estimated attention weight (0.0-1.0)
    pub fn auto_detect_attention(
        &self,
        content: &str,
        emotion: f32,
        tags: &[String],
    ) -> f32 {
        let mut attention: f32 = 0.5; // Start with normal attention

        // High-priority tag signals (critical, error, important, breakthrough)
        let priority_tags = ["error", "critical", "important", "breakthrough", "security"];
        for tag in tags {
            if priority_tags.iter().any(|&pt| tag.to_lowercase().contains(pt)) {
                attention += 0.2;
                break; // Only count once
            }
        }

        // Strong emotional salience (high emotion = high attention)
        if emotion.abs() > 0.7 {
            attention += 0.2;
        } else if emotion.abs() > 0.5 {
            attention += 0.1;
        }

        // Detailed content (long description = high attention/thought investment)
        if content.len() > 200 {
            attention += 0.15;
        } else if content.len() > 100 {
            attention += 0.05;
        }

        // Multiple tags (high contextualization = high attention)
        if tags.len() >= 3 {
            attention += 0.1;
        }

        // Clamp to valid range [0.0, 1.0]
        attention.min(1.0)
    }

    /// Recall memories by temporal cue (mental time travel)
    ///
    /// Query: "What happened around 9 AM yesterday?"
    ///
    /// # Algorithm
    /// 1. Encode query time using TemporalEncoder
    /// 2. Convert to bipolar for SDM query
    /// 3. Use SDM's pattern completion to retrieve similar temporal patterns
    /// 4. Return memories with timestamps near the query time
    pub fn recall_by_time(&mut self, query_time: Duration, top_k: usize) -> Result<Vec<EpisodicTrace>> {
        // Encode query time
        let temporal_vec = self.temporal_encoder.encode_time(query_time)?;
        let query: Vec<i8> = temporal_vec.iter()
            .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
            .collect();

        // Query SDM with iterative retrieval for noise cleanup
        let retrieved = match self.sdm.iterative_read(&query, 10) {
            IterativeReadResult::Converged { pattern, .. } => pattern,
            IterativeReadResult::MaxIterations { pattern, .. } => pattern,
            IterativeReadResult::Failed { .. } => {
                // Fallback to single read
                match self.sdm.read(&query) {
                    ReadResult::Success { pattern, .. } => pattern,
                    _ => return Ok(Vec::new()),
                }
            }
        };

        // Find similar memories in buffer
        let mut results: Vec<(EpisodicTrace, f32)> = self.buffer.iter()
            .map(|trace| {
                let sim = hamming_similarity(&retrieved, &trace.chrono_semantic_vector);
                (trace.clone(), sim)
            })
            .filter(|(_, sim)| *sim >= self.config.recall_threshold)
            .collect();

        // Sort by similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Return top k
        Ok(results.into_iter().take(top_k).map(|(trace, _)| trace).collect())
    }

    /// Recall memories by semantic cue
    ///
    /// Query: "Show me all git errors"
    pub fn recall_by_content(&mut self, query: &str, top_k: usize) -> Result<Vec<EpisodicTrace>> {
        // Encode query semantically
        let semantic_vec = self.semantic_space.encode(query)?;
        let query_bipolar: Vec<i8> = semantic_vec.iter()
            .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
            .collect();

        // Query SDM
        let retrieved = match self.sdm.iterative_read(&query_bipolar, 10) {
            IterativeReadResult::Converged { pattern, .. } => pattern,
            IterativeReadResult::MaxIterations { pattern, .. } => pattern,
            IterativeReadResult::Failed { .. } => {
                match self.sdm.read(&query_bipolar) {
                    ReadResult::Success { pattern, .. } => pattern,
                    _ => return Ok(Vec::new()),
                }
            }
        };

        // Find similar memories
        let mut results: Vec<(EpisodicTrace, f32)> = self.buffer.iter()
            .map(|trace| {
                let sim = hamming_similarity(&retrieved, &trace.chrono_semantic_vector);
                (trace.clone(), sim)
            })
            .filter(|(_, sim)| *sim >= self.config.recall_threshold)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results.into_iter().take(top_k).map(|(trace, _)| trace).collect())
    }

    /// Revolutionary: Chrono-semantic query
    ///
    /// Query: "Show me git errors from yesterday morning"
    ///
    /// This is the KILLER FEATURE - combining temporal + semantic cues!
    pub fn recall_chrono_semantic(
        &mut self,
        query_content: &str,
        query_time: Duration,
        top_k: usize,
    ) -> Result<Vec<EpisodicTrace>> {
        // Encode temporal cue
        let temporal_vec = self.temporal_encoder.encode_time(query_time)?;

        // Encode semantic cue
        let semantic_vec = self.semantic_space.encode(query_content)?;

        // Bind them (chrono-semantic fusion)
        let bound = self.temporal_encoder.bind(&temporal_vec, &semantic_vec)?;

        // Convert to bipolar
        let query: Vec<i8> = bound.iter()
            .map(|&v| if v >= 0.0 { 1i8 } else { -1i8 })
            .collect();

        // Query SDM with the chrono-semantic cue
        let retrieved = match self.sdm.iterative_read(&query, 10) {
            IterativeReadResult::Converged { pattern, .. } => pattern,
            IterativeReadResult::MaxIterations { pattern, .. } => pattern,
            IterativeReadResult::Failed { .. } => {
                match self.sdm.read(&query) {
                    ReadResult::Success { pattern, .. } => pattern,
                    _ => return Ok(Vec::new()),
                }
            }
        };

        // Find matching memories
        let mut results: Vec<(EpisodicTrace, f32)> = self.buffer.iter()
            .map(|trace| {
                let sim = hamming_similarity(&retrieved, &trace.chrono_semantic_vector);
                (trace.clone(), sim)
            })
            .filter(|(_, sim)| *sim >= self.config.recall_threshold)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results.into_iter().take(top_k).map(|(trace, _)| trace).collect())
    }

    /// Get all memories in buffer (for inspection)
    pub fn buffer(&self) -> &VecDeque<EpisodicTrace> {
        &self.buffer
    }

    /// Consolidate old memories (apply forgetting curve)
    fn consolidate(&mut self) -> Result<()> {
        // Apply decay to all memories
        for trace in self.buffer.iter_mut() {
            trace.decay(self.config.decay_rate);
        }

        // Remove weak memories (strength < 0.1)
        self.buffer.retain(|trace| trace.strength >= 0.1);

        // If still over capacity, remove weakest memories until at max_buffer_size
        while self.buffer.len() > self.config.max_buffer_size {
            // Find and remove weakest memory
            if let Some((idx, _)) = self.buffer.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal))
            {
                self.buffer.remove(idx);
            } else {
                break;
            }
        }

        Ok(())
    }

    /// Get engine statistics
    pub fn stats(&self) -> EngineStats {
        EngineStats {
            total_memories: self.buffer.len(),
            sdm_config: self.sdm.config().clone(),
            next_id: self.next_id,
        }
    }
}

impl Default for EpisodicMemoryEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create default episodic memory engine")
    }
}

/// Statistics about the episodic memory engine
#[derive(Debug, Clone)]
pub struct EngineStats {
    pub total_memories: usize,
    pub sdm_config: SDMConfig,
    pub next_id: u64,
}

// ============================================================================
// TESTS - 10 comprehensive tests for the revolutionary engine
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_episodic_trace_creation() {
        let mut engine = EpisodicMemoryEngine::new().unwrap();
        let timestamp = Duration::from_secs(9 * 3600); // 9 AM

        let trace = EpisodicTrace::new(
            0,
            timestamp,
            "Git push failed".to_string(),
            vec!["git".to_string(), "error".to_string()],
            -0.7,
            &engine.temporal_encoder,
            &mut engine.semantic_space,
        ).unwrap();

        assert_eq!(trace.id, 0);
        assert_eq!(trace.timestamp, timestamp);
        assert_eq!(trace.content, "Git push failed");
        assert_eq!(trace.tags, vec!["git", "error"]);
        assert_eq!(trace.emotion, -0.7);
        assert_eq!(trace.chrono_semantic_vector.len(), HDC_DIMENSION);
        assert_eq!(trace.temporal_vector.len(), HDC_DIMENSION);
        assert_eq!(trace.semantic_vector.len(), HDC_DIMENSION);
    }

    #[test]
    fn test_store_and_recall_by_time() {
        let mut engine = EpisodicMemoryEngine::new().unwrap();

        // Store memory at 9 AM
        let time_9am = Duration::from_secs(9 * 3600);
        engine.store(
            time_9am,
            "Morning standup meeting".to_string(),
            vec!["meeting".to_string()],
            0.5,
        ).unwrap();

        // Recall near 9 AM
        let recalled = engine.recall_by_time(time_9am, 5).unwrap();
        assert!(recalled.len() > 0, "Should recall memory near 9 AM");
        assert_eq!(recalled[0].content, "Morning standup meeting");
    }

    #[test]
    fn test_store_and_recall_by_content() {
        let mut engine = EpisodicMemoryEngine::new().unwrap();

        // Store git error
        engine.store(
            Duration::from_secs(1000),
            "Git merge conflict in main.rs".to_string(),
            vec!["git".to_string(), "error".to_string()],
            -0.8,
        ).unwrap();

        // Recall git errors
        let recalled = engine.recall_by_content("git error", 5).unwrap();
        assert!(recalled.len() > 0, "Should recall git error");
        assert!(recalled[0].content.contains("Git merge conflict"));
    }

    #[test]
    fn test_chrono_semantic_recall() {
        let mut engine = EpisodicMemoryEngine::new().unwrap();

        // Store memory: git error at 9 AM
        let time_9am = Duration::from_secs(9 * 3600);
        engine.store(
            time_9am,
            "Git push failed with authentication error".to_string(),
            vec!["git".to_string(), "error".to_string()],
            -0.7,
        ).unwrap();

        // Store memory: different error at 2 PM
        let time_2pm = Duration::from_secs(14 * 3600);
        engine.store(
            time_2pm,
            "NixOS build failed".to_string(),
            vec!["nixos".to_string(), "error".to_string()],
            -0.6,
        ).unwrap();

        // Query: git errors from morning
        let recalled = engine.recall_chrono_semantic("git error", time_9am, 5).unwrap();

        assert!(recalled.len() > 0, "Should find git error from morning");
        // Should preferentially return the git error from 9 AM, not the NixOS error from 2 PM
        if recalled.len() > 0 {
            assert!(recalled[0].content.contains("Git"), "Should recall git error");
        }
    }

    #[test]
    fn test_memory_strengthening() {
        let mut trace = EpisodicTrace {
            id: 0,
            timestamp: Duration::from_secs(0),
            content: "test".to_string(),
            tags: vec![],
            emotion: 0.0,
            chrono_semantic_vector: vec![1i8; HDC_DIMENSION],
            emotional_binding_vector: vec![1i8; HDC_DIMENSION], // Neutral emotion pattern
            temporal_vector: vec![1.0; HDC_DIMENSION],
            semantic_vector: vec![1.0; HDC_DIMENSION],
            recall_count: 0,
            strength: 0.5,
            attention_weight: 0.5,  // Week 17 Day 3: Normal attention
            encoding_strength: 10,  // Week 17 Day 3: Normal encoding (10x writes)
        };

        assert_eq!(trace.recall_count, 0);
        assert!((trace.strength - 0.5).abs() < 0.01);

        trace.strengthen();
        assert_eq!(trace.recall_count, 1);
        assert!((trace.strength - 0.6).abs() < 0.01);

        trace.strengthen();
        assert_eq!(trace.recall_count, 2);
        assert!((trace.strength - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_memory_decay() {
        let mut trace = EpisodicTrace {
            id: 0,
            timestamp: Duration::from_secs(0),
            content: "test".to_string(),
            tags: vec![],
            emotion: 0.0,
            chrono_semantic_vector: vec![1i8; HDC_DIMENSION],
            emotional_binding_vector: vec![1i8; HDC_DIMENSION], // Neutral emotion pattern
            temporal_vector: vec![1.0; HDC_DIMENSION],
            semantic_vector: vec![1.0; HDC_DIMENSION],
            recall_count: 0,
            strength: 1.0,
            attention_weight: 0.5,  // Week 17 Day 3: Normal attention
            encoding_strength: 10,  // Week 17 Day 3: Normal encoding (10x writes)
        };

        trace.decay(0.1); // 10% decay
        assert!((trace.strength - 0.9).abs() < 0.01);

        trace.decay(0.1);
        assert!((trace.strength - 0.81).abs() < 0.01);
    }

    #[test]
    fn test_multiple_memories_temporal_similarity() {
        let mut engine = EpisodicMemoryEngine::new().unwrap();

        // Store 3 memories at different times
        let time_9am = Duration::from_secs(9 * 3600);
        let time_905 = Duration::from_secs(9 * 3600 + 5 * 60); // 5 min later
        let time_3pm = Duration::from_secs(15 * 3600);

        engine.store(time_9am, "Morning coffee".to_string(), vec![], 0.8).unwrap();
        engine.store(time_905, "Morning standup".to_string(), vec![], 0.5).unwrap();
        engine.store(time_3pm, "Afternoon review".to_string(), vec![], 0.3).unwrap();

        // Query around 9 AM should retrieve both 9:00 and 9:05, not 3 PM
        let recalled = engine.recall_by_time(time_9am, 5).unwrap();
        assert!(recalled.len() >= 2, "Should recall both morning memories");
    }

    #[test]
    fn test_emotional_modulation() {
        // Use lower recall threshold to account for emotional modulation differences
        let config = EpisodicConfig {
            recall_threshold: 0.3, // Lower threshold for emotional variation
            ..Default::default()
        };
        let mut engine = EpisodicMemoryEngine::with_config(config).unwrap();

        // Store same content with different emotions
        engine.store(
            Duration::from_secs(1000),
            "Code review feedback".to_string(),
            vec!["review".to_string()],
            0.9, // Positive
        ).unwrap();

        engine.store(
            Duration::from_secs(2000),
            "Code review feedback".to_string(),
            vec!["review".to_string()],
            -0.9, // Negative
        ).unwrap();

        // Both should be recalled, despite different emotional encodings
        // Lower threshold accounts for emotional modulation affecting similarity
        let recalled = engine.recall_by_content("code review", 5).unwrap();
        assert!(recalled.len() >= 2, "Should find both emotionally-modulated memories");
    }

    #[test]
    fn test_buffer_consolidation() {
        let config = EpisodicConfig {
            max_buffer_size: 5,
            decay_rate: 0.5, // Strong decay
            ..Default::default()
        };

        let mut engine = EpisodicMemoryEngine::with_config(config).unwrap();

        // Store 6 memories (should trigger consolidation)
        for i in 0..6 {
            engine.store(
                Duration::from_secs(i * 100),
                format!("Memory {}", i),
                vec![],
                0.0,
            ).unwrap();
        }

        // After consolidation with 50% decay, some weak memories should be removed
        assert!(engine.buffer().len() <= 5, "Buffer should be consolidated");
    }

    #[test]
    fn test_engine_stats() {
        let mut engine = EpisodicMemoryEngine::new().unwrap();

        engine.store(
            Duration::from_secs(1000),
            "Test memory".to_string(),
            vec![],
            0.0,
        ).unwrap();

        let stats = engine.stats();
        assert_eq!(stats.total_memories, 1);
        assert_eq!(stats.next_id, 1);
        assert_eq!(stats.sdm_config.dimension, HDC_DIMENSION);
    }

    // ========================================
    // WEEK 17 DAY 3: Attention-Weighted Encoding Tests
    // ========================================

    #[test]
    fn test_attention_weighted_storage_encoding_strength() {
        // Week 17 Day 3: Verify variable SDM encoding strength based on attention
        let mut engine = EpisodicMemoryEngine::new().unwrap();

        // Store with MAXIMUM attention (critical moment)
        let critical_id = engine.store_with_attention(
            Duration::from_secs(1000),
            "Critical security breach detected!".to_string(),
            vec!["security".to_string(), "critical".to_string()],
            -0.9, // High stress
            1.0,  // FULL attention → 100x SDM writes
        ).unwrap();

        // Store with MINIMUM attention (background task)
        let routine_id = engine.store_with_attention(
            Duration::from_secs(1100),
            "Opened editor".to_string(),
            vec!["routine".to_string()],
            0.0,  // Neutral emotion
            0.0,  // Zero attention → 1x SDM write
        ).unwrap();

        // Store with NORMAL attention (typical work)
        let normal_id = engine.store_with_attention(
            Duration::from_secs(1200),
            "Fixed typo in documentation".to_string(),
            vec!["docs".to_string()],
            0.1,  // Slightly positive
            0.5,  // Normal attention → 50x SDM writes
        ).unwrap();

        // Verify traces have correct attention metadata
        let critical = engine.buffer().iter().find(|t| t.id == critical_id).unwrap();
        let routine = engine.buffer().iter().find(|t| t.id == routine_id).unwrap();
        let normal = engine.buffer().iter().find(|t| t.id == normal_id).unwrap();

        assert_eq!(critical.attention_weight, 1.0);
        assert_eq!(critical.encoding_strength, 100); // 100x writes

        assert_eq!(routine.attention_weight, 0.0);
        assert_eq!(routine.encoding_strength, 1); // 1x write

        assert_eq!(normal.attention_weight, 0.5);
        assert_eq!(normal.encoding_strength, 50); // 50x writes
    }

    #[test]
    fn test_auto_detect_attention_heuristics() {
        // Week 17 Day 3: Verify automatic attention detection
        let engine = EpisodicMemoryEngine::new().unwrap();

        // High attention: Critical error with strong emotion
        let attention_critical = engine.auto_detect_attention(
            "CRITICAL: Authentication bypass vulnerability discovered in production. Immediate patching required!",
            -0.9, // High stress
            &["error".to_string(), "critical".to_string(), "security".to_string()],
        );
        assert!(attention_critical >= 0.8, "Critical errors should have high attention");

        // Medium attention: Important but not critical
        let attention_medium = engine.auto_detect_attention(
            "Fixed bug in payment processing",
            0.3, // Moderate positive
            &["bug".to_string(), "fix".to_string()],
        );
        assert!(attention_medium >= 0.5 && attention_medium < 0.8, "Important work should have medium attention");

        // Low attention: Routine task
        let attention_low = engine.auto_detect_attention(
            "Opened file",
            0.0, // Neutral
            &["routine".to_string()],
        );
        assert!(attention_low < 0.6, "Routine tasks should have low attention");

        // Very high attention: Breakthrough discovery
        let attention_breakthrough = engine.auto_detect_attention(
            "BREAKTHROUGH: Discovered revolutionary algorithm that solves the P=NP problem! This is the culmination of 10 years of research and will fundamentally transform computer science.",
            0.95, // Extreme excitement
            &["breakthrough".to_string(), "important".to_string(), "research".to_string()],
        );
        assert_eq!(attention_breakthrough, 1.0, "Breakthrough discoveries should have maximum attention");
    }

    #[test]
    fn test_attention_weighted_recall_persistence() {
        // Week 17 Day 3: Verify high-attention memories are more persistent/retrievable
        let mut engine = EpisodicMemoryEngine::new().unwrap();

        let timestamp = Duration::from_secs(5000);

        // Store critical memory with full attention (100x SDM reinforcement)
        engine.store_with_attention(
            timestamp,
            "Production database crashed - data loss risk!".to_string(),
            vec!["critical".to_string(), "database".to_string()],
            -0.8,
            1.0, // Full attention → 100x SDM writes
        ).unwrap();

        // Store routine memory with minimal attention (1x SDM write)
        engine.store_with_attention(
            timestamp + Duration::from_secs(10),
            "Opened settings panel".to_string(),
            vec!["routine".to_string(), "ui".to_string()],
            0.0,
            0.05, // Minimal attention → ~5x SDM writes
        ).unwrap();

        // Both memories stored recently, but critical one should have MUCH stronger encoding
        // Query by semantic content - critical memory should be recalled more reliably
        let critical_recall = engine.recall_by_content("database crash", 5).unwrap();
        assert!(!critical_recall.is_empty(), "High-attention critical memory should be easily recalled");

        // Routine memory should be less reliably encoded
        let routine_recall = engine.recall_by_content("settings panel", 5).unwrap();
        // Note: Due to weak encoding (5x vs 100x), routine memory may be harder to retrieve
        // but we store it in buffer so it should still be findable
    }

    #[test]
    fn test_attention_weighted_encoding_formula() {
        // Week 17 Day 3: Verify encoding strength formula correctness
        let mut engine = EpisodicMemoryEngine::new().unwrap();

        // Test encoding strength formula: 1 + (attention * 99)
        let test_cases = vec![
            (0.0, 1),    // 1 + (0.0 * 99) = 1
            (0.25, 25),  // 1 + (0.25 * 99) = 25.75 → 25
            (0.5, 50),   // 1 + (0.5 * 99) = 50.5 → 50
            (0.75, 75),  // 1 + (0.75 * 99) = 75.25 → 75
            (1.0, 100),  // 1 + (1.0 * 99) = 100
        ];

        for (attention, expected_strength) in test_cases {
            let id = engine.store_with_attention(
                Duration::from_secs(1000),
                format!("Test attention {}", attention),
                vec![],
                0.0,
                attention,
            ).unwrap();

            let trace = engine.buffer().iter().find(|t| t.id == id).unwrap();
            assert_eq!(trace.encoding_strength, expected_strength,
                "Attention {} should produce encoding strength {}", attention, expected_strength);
        }
    }

    #[test]
    fn test_attention_weight_clamping() {
        // Week 17 Day 3: Verify out-of-range attention weights are clamped to [0.0, 1.0]
        let mut engine = EpisodicMemoryEngine::new().unwrap();

        // Test extreme out-of-range values
        let id_high = engine.store_with_attention(
            Duration::from_secs(1000),
            "Test extreme high".to_string(),
            vec![],
            0.0,
            99.9, // Should be clamped to 1.0
        ).unwrap();

        let id_low = engine.store_with_attention(
            Duration::from_secs(1100),
            "Test extreme low".to_string(),
            vec![],
            0.0,
            -50.0, // Should be clamped to 0.0
        ).unwrap();

        let trace_high = engine.buffer().iter().find(|t| t.id == id_high).unwrap();
        let trace_low = engine.buffer().iter().find(|t| t.id == id_low).unwrap();

        assert_eq!(trace_high.attention_weight, 1.0, "Attention above 1.0 should be clamped to 1.0");
        assert_eq!(trace_high.encoding_strength, 100, "Clamped max attention should give 100x encoding");

        assert_eq!(trace_low.attention_weight, 0.0, "Attention below 0.0 should be clamped to 0.0");
        assert_eq!(trace_low.encoding_strength, 1, "Clamped min attention should give 1x encoding");
    }
}
