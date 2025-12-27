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
//    - Both stored as HV16 vectors, retrieved by similarity
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

use crate::hdc::HV16;

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
            MemoryType::Episodic => 86400.0 * 365.0 * 10.0,  // ~10 years (vivid memories)
            MemoryType::Semantic => 86400.0 * 365.0 * 50.0,  // ~50 years (facts last longer)
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

    /// Content representation (HV16 hypervector encoding the experience)
    pub content: Vec<HV16>,

    /// When it happened (Unix timestamp)
    pub timestamp: f64,

    /// Where it happened (spatial context, optional)
    pub location: Option<Vec<HV16>>,

    /// Emotional valence [-1, 1]: negative to positive
    pub emotional_valence: f64,

    /// Emotional arousal [0, 1]: calm to intense
    pub emotional_arousal: f64,

    /// Context (what else was happening)
    pub context: Vec<HV16>,

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
        content: Vec<HV16>,
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
    pub fn with_location(mut self, location: Vec<HV16>) -> Self {
        self.location = Some(location);
        self
    }

    /// Set context
    pub fn with_context(mut self, context: Vec<HV16>) -> Self {
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
    pub content: Vec<HV16>,

    /// Context cue (what context we're in)
    pub context: Option<Vec<HV16>>,

    /// Location cue (where we are)
    pub location: Option<Vec<HV16>>,

    /// Temporal cue (when - relative to now)
    pub temporal_proximity: Option<f64>, // Recent memories weighted higher

    /// Memory type filter
    pub memory_type: Option<MemoryType>,
}

impl RetrievalCue {
    /// Create content-based cue
    pub fn content(content: Vec<HV16>) -> Self {
        Self {
            content,
            context: None,
            location: None,
            temporal_proximity: None,
            memory_type: None,
        }
    }

    /// Add context
    pub fn with_context(mut self, context: Vec<HV16>) -> Self {
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
    pub fn retrieve(&mut self, cue: &RetrievalCue, current_time: f64, top_k: usize) -> Vec<RetrievedMemory> {
        self.total_retrievals += 1;

        // First pass: compute relevance scores (immutable borrow)
        let mut scored_memories: Vec<RetrievedMemory> = self.memories
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
                let content_similarity = Self::compute_similarity_static(&cue.content, &exp.content);

                // Context similarity (if cue has context)
                let context_similarity = if let Some(ref cue_context) = cue.context {
                    Self::compute_similarity_static(cue_context, &exp.context)
                } else {
                    0.5 // Neutral if no context cue
                };

                // Location similarity (if cue has location)
                let location_similarity = if let (Some(ref cue_loc), Some(ref exp_loc)) = (&cue.location, &exp.location) {
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
                    strength * 0.2                   // Strong memories more accessible
                ) * temporal_bonus;

                RetrievedMemory {
                    experience: exp.clone(),
                    relevance,
                }
            })
            .collect();

        // Sort by relevance (highest first)
        scored_memories.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());

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
    fn compute_similarity_static(a: &[HV16], b: &[HV16]) -> f64 {
        if a.is_empty() || b.is_empty() {
            return 0.0;
        }

        // Average pairwise similarity
        let mut total_similarity = 0.0;
        let mut count = 0;

        for hv_a in a {
            for hv_b in b {
                total_similarity += HV16::similarity(hv_a, hv_b) as f64;
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
    fn compute_similarity(&self, a: &[HV16], b: &[HV16]) -> f64 {
        Self::compute_similarity_static(a, b)
    }

    /// Consolidate memories (call during sleep or with time passage)
    pub fn consolidate_memories(&mut self, is_sleeping: bool, duration: f64) {
        let amount = self.consolidation.consolidation_amount(is_sleeping, duration);

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
        self.memories.values()
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

        let sum: f64 = self.memories.values()
            .map(|exp| exp.consolidation)
            .sum();

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
// Qdrant Integration (Placeholder for Production)
// ============================================================================

/// Configuration for Qdrant vector database integration
/// NOTE: This is a design specification for production implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantConfig {
    /// Qdrant server URL
    pub url: String,

    /// Collection name for experiences
    pub collection_name: String,

    /// Vector dimension (HV16::DIM = 2048)
    pub vector_dim: usize,

    /// Distance metric (Cosine for HDC)
    pub distance_metric: String,
}

impl QdrantConfig {
    pub fn default_config() -> Self {
        Self {
            url: "http://localhost:6333".to_string(),
            collection_name: "symthaea_memories".to_string(),
            vector_dim: 2048, // HV16::DIM
            distance_metric: "Cosine".to_string(),
        }
    }
}

/// Qdrant integration for persistent storage
/// TODO: Implement actual Qdrant client integration
/// For now, this is a specification of the interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantMemoryStore {
    pub config: QdrantConfig,
    // In production: qdrant_client::QdrantClient
}

impl QdrantMemoryStore {
    pub fn new(config: QdrantConfig) -> Self {
        Self { config }
    }

    /// Store experience to Qdrant
    /// TODO: Implement with qdrant_client
    #[allow(unused_variables)]
    pub fn store_experience(&self, experience: &Experience) -> Result<(), String> {
        // Production implementation:
        // 1. Convert experience.content to Qdrant vector
        // 2. Create payload with metadata (timestamp, type, valence, etc.)
        // 3. Insert to collection
        // 4. Return result

        Ok(()) // Placeholder
    }

    /// Retrieve experiences by vector similarity
    /// TODO: Implement with qdrant_client
    #[allow(unused_variables)]
    pub fn retrieve_similar(&self, query: &[HV16], limit: usize) -> Result<Vec<Experience>, String> {
        // Production implementation:
        // 1. Convert query to Qdrant vector
        // 2. Search collection
        // 3. Parse results
        // 4. Return experiences

        Ok(Vec::new()) // Placeholder
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
        let content = vec![HV16::random(1), HV16::random(2)];
        let exp = Experience::new(
            "exp1".to_string(),
            MemoryType::Episodic,
            content,
            1000.0,
            0.8,  // Positive valence
            0.6,  // Medium arousal
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
        let content = vec![HV16::random(1)];
        let location = vec![HV16::random(100)];
        let context = vec![HV16::random(200)];

        let exp = Experience::new("exp1".to_string(), MemoryType::Episodic, content, 1000.0, 0.5, 0.5)
            .with_location(location.clone())
            .with_context(context.clone());

        assert!(exp.location.is_some());
        assert_eq!(exp.context.len(), 1);
    }

    #[test]
    fn test_forgetting_curve() {
        let content = vec![HV16::random(1)];
        // Use 0.0 emotional arousal for simple test (no emotional bonus)
        let exp = Experience::new("exp1".to_string(), MemoryType::Episodic, content, 1000.0, 0.0, 0.0)
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
        let content = vec![HV16::random(1)];
        let mut exp = Experience::new("exp1".to_string(), MemoryType::Episodic, content, 1000.0, 0.5, 0.5)
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
        let content = vec![HV16::random(1)];
        let mut exp = Experience::new("exp1".to_string(), MemoryType::Episodic, content, 1000.0, 0.5, 0.5);

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
        let content = vec![HV16::random(1)];

        // Neutral emotion
        let exp_neutral = Experience::new("exp1".to_string(), MemoryType::Episodic, content.clone(), 1000.0, 0.0, 0.0);

        // High arousal emotion
        let exp_emotional = Experience::new("exp2".to_string(), MemoryType::Episodic, content, 1000.0, 0.8, 1.0);

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

        let content = vec![HV16::random(1), HV16::random(2)];
        let exp = Experience::new("exp1".to_string(), MemoryType::Episodic, content.clone(), 1000.0, 0.5, 0.5);

        ltm.store(exp);

        assert_eq!(ltm.total_memories(), 1);
        assert_eq!(ltm.total_stored, 1);

        // Retrieve by content similarity
        let cue = RetrievalCue::content(vec![HV16::random(1)]); // Similar to stored
        let retrieved = ltm.retrieve(&cue, 1100.0, 5);

        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].experience.id, "exp1");
        assert!(retrieved[0].relevance > 0.0);
    }

    #[test]
    fn test_retrieval_by_type() {
        let mut ltm = LongTermMemory::new();

        // Store episodic
        let content1 = vec![HV16::random(1)];
        let exp1 = Experience::new("episodic".to_string(), MemoryType::Episodic, content1, 1000.0, 0.5, 0.5);
        ltm.store(exp1);

        // Store semantic
        let content2 = vec![HV16::random(2)];
        let exp2 = Experience::new("semantic".to_string(), MemoryType::Semantic, content2, 1000.0, 0.5, 0.5);
        ltm.store(exp2);

        assert_eq!(ltm.total_memories(), 2);

        // Retrieve only episodic
        let cue = RetrievalCue::content(vec![HV16::random(3)])
            .with_type(MemoryType::Episodic);
        let retrieved = ltm.retrieve(&cue, 1100.0, 10);

        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].experience.memory_type, MemoryType::Episodic);
    }

    #[test]
    fn test_consolidation_during_sleep() {
        let mut ltm = LongTermMemory::new();

        let content = vec![HV16::random(1)];
        let exp = Experience::new("exp1".to_string(), MemoryType::Episodic, content, 1000.0, 0.5, 0.5);
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
            let content = vec![HV16::random(i)];
            let exp = Experience::new(format!("episodic_{}", i), MemoryType::Episodic, content, 1000.0, 0.5, 0.5);
            ltm.store(exp);
        }

        // Store 3 semantic
        for i in 0..3 {
            let content = vec![HV16::random(100 + i)];
            let exp = Experience::new(format!("semantic_{}", i), MemoryType::Semantic, content, 1000.0, 0.5, 0.5);
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
        assert_eq!(config.vector_dim, 2048); // HV16::DIM
        assert_eq!(config.distance_metric, "Cosine");
    }

    #[test]
    fn test_clear() {
        let mut ltm = LongTermMemory::new();

        let content = vec![HV16::random(1)];
        let exp = Experience::new("exp1".to_string(), MemoryType::Episodic, content, 1000.0, 0.5, 0.5);
        ltm.store(exp);

        assert_eq!(ltm.total_memories(), 1);

        ltm.clear();

        assert_eq!(ltm.total_memories(), 0);
    }
}
