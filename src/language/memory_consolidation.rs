//! Memory Consolidation Module (Phase B2)
//!
//! Smart memory retrieval with clustering, importance weighting,
//! forgetting curves, and sleep-like consolidation. Based on
//! Ebbinghaus forgetting curve and memory consolidation research.

use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use crate::hdc::binary_hv::HV16;

// ============================================================================
// MEMORY ENTRY
// ============================================================================

/// A memory entry with metadata for consolidation
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: u64,
    /// Content of the memory
    pub content: String,
    /// Topic/category
    pub topic: String,
    /// Emotional valence (-1.0 to 1.0)
    pub emotional_valence: f32,
    /// Emotional arousal (0.0 to 1.0)
    pub emotional_arousal: f32,
    /// Creation timestamp (ms since epoch)
    pub created_at: u64,
    /// Last accessed timestamp
    pub last_accessed: u64,
    /// Number of times accessed/rehearsed
    pub access_count: u32,
    /// Current memory strength (0.0 to 1.0)
    pub strength: f32,
    /// HDC encoding for similarity matching
    pub encoding: HV16,
    /// Cluster assignment (if clustered)
    pub cluster_id: Option<usize>,
    /// Associated memories (linked by ID)
    pub associations: Vec<u64>,
}

impl MemoryEntry {
    pub fn new(content: String, topic: String, valence: f32, arousal: f32) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Generate encoding from content
        let seed = content.bytes().fold(12345u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });

        Self {
            id: now,
            content,
            topic,
            emotional_valence: valence,
            emotional_arousal: arousal,
            created_at: now,
            last_accessed: now,
            access_count: 1,
            strength: 1.0,
            encoding: HV16::random(seed),
            cluster_id: None,
            associations: Vec::new(),
        }
    }

    /// Access the memory (updates stats, rehearsal effect)
    pub fn access(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(self.last_accessed);

        self.last_accessed = now;
        self.access_count += 1;

        // Rehearsal boosts strength (spacing effect)
        let time_since_last = now.saturating_sub(self.last_accessed);
        let spacing_bonus = (time_since_last as f32 / 86400000.0).min(1.0) * 0.1;
        self.strength = (self.strength + 0.1 + spacing_bonus).min(1.0);
    }

    /// Age in hours since creation
    pub fn age_hours(&self) -> f32 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(self.created_at);

        (now - self.created_at) as f32 / 3600000.0
    }
}

// ============================================================================
// MEMORY CLUSTERER
// ============================================================================

/// A cluster of related memories
#[derive(Debug, Clone)]
pub struct MemoryCluster {
    /// Cluster ID
    pub id: usize,
    /// Cluster topic/theme
    pub theme: String,
    /// Member memory IDs
    pub members: Vec<u64>,
    /// Centroid encoding
    pub centroid: HV16,
    /// Average emotional valence
    pub avg_valence: f32,
    /// Coherence score (how tight the cluster is)
    pub coherence: f32,
}

/// Groups related memories by topic and semantic similarity
#[derive(Debug)]
pub struct MemoryClusterer {
    /// Clusters by ID
    clusters: HashMap<usize, MemoryCluster>,
    /// Next cluster ID
    next_id: usize,
    /// Minimum similarity for clustering
    similarity_threshold: f32,
    /// Maximum cluster size
    max_cluster_size: usize,
}

impl MemoryClusterer {
    pub fn new() -> Self {
        Self {
            clusters: HashMap::new(),
            next_id: 0,
            similarity_threshold: 0.4,
            max_cluster_size: 50,
        }
    }

    /// Assign a memory to a cluster (or create new)
    pub fn assign(&mut self, memory: &mut MemoryEntry) -> usize {
        // Find best matching cluster
        let mut best_cluster: Option<usize> = None;
        let mut best_similarity = self.similarity_threshold;

        for (id, cluster) in &self.clusters {
            if cluster.members.len() >= self.max_cluster_size {
                continue;
            }

            let similarity = memory.encoding.similarity(&cluster.centroid);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_cluster = Some(*id);
            }
        }

        match best_cluster {
            Some(cluster_id) => {
                // Add to existing cluster
                if let Some(cluster) = self.clusters.get_mut(&cluster_id) {
                    cluster.members.push(memory.id);
                    // Update centroid (simplified: keep as-is for now)
                    cluster.avg_valence = (cluster.avg_valence * (cluster.members.len() - 1) as f32
                        + memory.emotional_valence) / cluster.members.len() as f32;
                }
                memory.cluster_id = Some(cluster_id);
                cluster_id
            }
            None => {
                // Create new cluster
                let new_id = self.next_id;
                self.next_id += 1;

                let cluster = MemoryCluster {
                    id: new_id,
                    theme: memory.topic.clone(),
                    members: vec![memory.id],
                    centroid: memory.encoding.clone(),
                    avg_valence: memory.emotional_valence,
                    coherence: 1.0,
                };

                self.clusters.insert(new_id, cluster);
                memory.cluster_id = Some(new_id);
                new_id
            }
        }
    }

    /// Get cluster by ID
    pub fn get_cluster(&self, id: usize) -> Option<&MemoryCluster> {
        self.clusters.get(&id)
    }

    /// Get all clusters
    pub fn all_clusters(&self) -> Vec<&MemoryCluster> {
        self.clusters.values().collect()
    }

    /// Find clusters related to a query
    pub fn find_related_clusters(&self, query_encoding: &HV16) -> Vec<(usize, f32)> {
        let mut results: Vec<_> = self.clusters.iter()
            .map(|(id, cluster)| (*id, query_encoding.similarity(&cluster.centroid)))
            .filter(|(_, sim)| *sim > 0.2)
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

impl Default for MemoryClusterer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// IMPORTANCE SCORER
// ============================================================================

/// Scoring weights for memory importance
#[derive(Debug, Clone)]
pub struct ImportanceWeights {
    /// Weight for recency
    pub recency: f32,
    /// Weight for emotional intensity
    pub emotional: f32,
    /// Weight for access frequency
    pub frequency: f32,
    /// Weight for semantic relevance
    pub relevance: f32,
    /// Weight for association count
    pub associations: f32,
}

impl Default for ImportanceWeights {
    fn default() -> Self {
        Self {
            recency: 0.25,
            emotional: 0.2,
            frequency: 0.2,
            relevance: 0.25,
            associations: 0.1,
        }
    }
}

/// Scores memory importance for retrieval prioritization
#[derive(Debug)]
pub struct ImportanceScorer {
    weights: ImportanceWeights,
}

impl ImportanceScorer {
    pub fn new() -> Self {
        Self {
            weights: ImportanceWeights::default(),
        }
    }

    pub fn with_weights(weights: ImportanceWeights) -> Self {
        Self { weights }
    }

    /// Score a memory's importance
    pub fn score(&self, memory: &MemoryEntry, query_encoding: Option<&HV16>) -> f32 {
        // Recency score (decay over time)
        let age_hours = memory.age_hours();
        let recency_score = 1.0 / (1.0 + age_hours / 24.0); // Half-life of 1 day

        // Emotional intensity score
        let emotional_score = memory.emotional_arousal.abs()
            + (memory.emotional_valence.abs() * 0.5);

        // Frequency score (log scale)
        let frequency_score = (memory.access_count as f32).ln() / 10.0;

        // Relevance score (if query provided)
        let relevance_score = query_encoding
            .map(|q| memory.encoding.similarity(q).max(0.0))
            .unwrap_or(0.5);

        // Association score
        let association_score = (memory.associations.len() as f32 / 10.0).min(1.0);

        // Weighted sum
        self.weights.recency * recency_score
            + self.weights.emotional * emotional_score.min(1.0)
            + self.weights.frequency * frequency_score.min(1.0)
            + self.weights.relevance * relevance_score
            + self.weights.associations * association_score
    }

    /// Rank memories by importance
    pub fn rank<'a>(&self, memories: &'a [MemoryEntry], query: Option<&HV16>) -> Vec<(&'a MemoryEntry, f32)> {
        let mut scored: Vec<_> = memories.iter()
            .map(|m| (m, self.score(m, query)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

impl Default for ImportanceScorer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// FORGETTING CURVE
// ============================================================================

/// Ebbinghaus forgetting curve parameters
#[derive(Debug, Clone)]
pub struct ForgettingParams {
    /// Initial retention rate
    pub initial_retention: f32,
    /// Decay constant (higher = faster forgetting)
    pub decay_constant: f32,
    /// Minimum retention (floor)
    pub minimum_retention: f32,
    /// Rehearsal boost factor
    pub rehearsal_boost: f32,
}

impl Default for ForgettingParams {
    fn default() -> Self {
        Self {
            initial_retention: 1.0,
            decay_constant: 0.3,
            minimum_retention: 0.1,
            rehearsal_boost: 0.2,
        }
    }
}

/// Implements Ebbinghaus forgetting curve with rehearsal
#[derive(Debug)]
pub struct ForgettingCurve {
    params: ForgettingParams,
}

impl ForgettingCurve {
    pub fn new() -> Self {
        Self {
            params: ForgettingParams::default(),
        }
    }

    pub fn with_params(params: ForgettingParams) -> Self {
        Self { params }
    }

    /// Calculate retention based on time elapsed
    /// Formula: R = e^(-t/S) where S is stability
    pub fn retention(&self, memory: &MemoryEntry) -> f32 {
        let age_hours = memory.age_hours();

        // Stability increases with rehearsals
        let stability = 1.0 + (memory.access_count as f32 * self.params.rehearsal_boost);

        // Ebbinghaus curve: R = e^(-t/S)
        let raw_retention = (-age_hours / (24.0 * stability)).exp();

        // Apply initial retention and floor
        let retention = self.params.initial_retention * raw_retention;
        retention.max(self.params.minimum_retention)
    }

    /// Predict when memory will decay below threshold
    pub fn decay_time(&self, memory: &MemoryEntry, threshold: f32) -> f32 {
        let stability = 1.0 + (memory.access_count as f32 * self.params.rehearsal_boost);

        // Solve: threshold = e^(-t/S) for t
        // t = -S * ln(threshold)
        -stability * 24.0 * (threshold / self.params.initial_retention).ln()
    }

    /// Update memory strength based on forgetting
    pub fn apply_decay(&self, memory: &mut MemoryEntry) {
        memory.strength = self.retention(memory);
    }

    /// Calculate optimal review time (spaced repetition)
    pub fn optimal_review_time(&self, memory: &MemoryEntry) -> f32 {
        // Review when retention drops to ~70%
        let target_retention = 0.7;
        self.decay_time(memory, target_retention)
    }
}

impl Default for ForgettingCurve {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CONSOLIDATION ENGINE
// ============================================================================

/// Memory consolidation state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConsolidationState {
    /// Normal operation
    Active,
    /// Consolidation in progress (like sleep)
    Consolidating,
    /// Post-consolidation review
    Reviewing,
}

/// Sleep-like memory consolidation engine
#[derive(Debug)]
pub struct ConsolidationEngine {
    /// Current state
    state: ConsolidationState,
    /// Memories pending consolidation
    pending: VecDeque<u64>,
    /// Consolidated memory associations (from -> to)
    new_associations: Vec<(u64, u64)>,
    /// Consolidation threshold (memories above this are kept)
    importance_threshold: f32,
    /// Last consolidation timestamp
    last_consolidation: u64,
    /// Consolidation interval (ms)
    consolidation_interval: u64,
}

impl ConsolidationEngine {
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self {
            state: ConsolidationState::Active,
            pending: VecDeque::new(),
            new_associations: Vec::new(),
            importance_threshold: 0.3,
            last_consolidation: now,
            consolidation_interval: 3600000, // 1 hour
        }
    }

    /// Queue a memory for consolidation
    pub fn queue(&mut self, memory_id: u64) {
        if !self.pending.contains(&memory_id) {
            self.pending.push_back(memory_id);
        }
    }

    /// Check if consolidation should occur
    pub fn should_consolidate(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(self.last_consolidation);

        now - self.last_consolidation > self.consolidation_interval
            || self.pending.len() > 50
    }

    /// Begin consolidation process
    pub fn begin_consolidation(&mut self) {
        self.state = ConsolidationState::Consolidating;
        self.new_associations.clear();
    }

    /// Consolidate memories: strengthen important, forget weak, create associations
    pub fn consolidate(&mut self, memories: &mut [MemoryEntry], scorer: &ImportanceScorer) -> ConsolidationResult {
        self.begin_consolidation();

        let mut strengthened = 0;
        let mut weakened = 0;
        let mut forgotten = Vec::new();

        // Score all memories
        let mut scored: Vec<_> = memories.iter_mut()
            .map(|m| {
                let score = scorer.score(m, None);
                (m, score)
            })
            .collect();

        // Sort by score
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Top 20% get strengthened
        let top_count = (scored.len() as f32 * 0.2).ceil() as usize;
        for (memory, _) in scored.iter_mut().take(top_count) {
            memory.strength = (memory.strength + 0.1).min(1.0);
            strengthened += 1;
        }

        // Find associations among top memories
        for i in 0..top_count.min(scored.len()) {
            for j in (i+1)..top_count.min(scored.len()) {
                let sim = scored[i].0.encoding.similarity(&scored[j].0.encoding);
                if sim > 0.5 {
                    self.new_associations.push((scored[i].0.id, scored[j].0.id));
                }
            }
        }

        // Bottom 20% get weakened or forgotten
        let bottom_start = (scored.len() as f32 * 0.8).floor() as usize;
        for (memory, score) in scored.iter_mut().skip(bottom_start) {
            if *score < self.importance_threshold && memory.strength < 0.3 {
                forgotten.push(memory.id);
            } else {
                memory.strength = (memory.strength - 0.05).max(0.0);
                weakened += 1;
            }
        }

        // Apply new associations
        for (from, to) in &self.new_associations {
            for (memory, _) in &mut scored {
                if memory.id == *from && !memory.associations.contains(to) {
                    memory.associations.push(*to);
                }
                if memory.id == *to && !memory.associations.contains(from) {
                    memory.associations.push(*from);
                }
            }
        }

        // Update state
        self.state = ConsolidationState::Reviewing;
        self.pending.clear();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(self.last_consolidation);
        self.last_consolidation = now;

        ConsolidationResult {
            strengthened,
            weakened,
            forgotten,
            new_associations: self.new_associations.len(),
        }
    }

    /// Complete consolidation and return to active state
    pub fn complete(&mut self) {
        self.state = ConsolidationState::Active;
    }

    /// Current consolidation state
    pub fn state(&self) -> ConsolidationState {
        self.state
    }

    /// Get new associations from last consolidation
    pub fn associations(&self) -> &[(u64, u64)] {
        &self.new_associations
    }
}

impl Default for ConsolidationEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of consolidation process
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    /// Number of memories strengthened
    pub strengthened: usize,
    /// Number of memories weakened
    pub weakened: usize,
    /// IDs of memories to forget
    pub forgotten: Vec<u64>,
    /// Number of new associations created
    pub new_associations: usize,
}

// ============================================================================
// MEMORY CONSOLIDATOR (MAIN)
// ============================================================================

/// Complete memory consolidation system
#[derive(Debug)]
pub struct MemoryConsolidator {
    /// Memory clustering
    pub clusterer: MemoryClusterer,
    /// Importance scoring
    pub scorer: ImportanceScorer,
    /// Forgetting curve
    pub forgetting: ForgettingCurve,
    /// Consolidation engine
    pub engine: ConsolidationEngine,
    /// All memories
    memories: Vec<MemoryEntry>,
    /// Memory index by ID
    memory_index: HashMap<u64, usize>,
}

impl MemoryConsolidator {
    pub fn new() -> Self {
        Self {
            clusterer: MemoryClusterer::new(),
            scorer: ImportanceScorer::new(),
            forgetting: ForgettingCurve::new(),
            engine: ConsolidationEngine::new(),
            memories: Vec::new(),
            memory_index: HashMap::new(),
        }
    }

    /// Add a new memory
    pub fn add_memory(&mut self, content: String, topic: String, valence: f32, arousal: f32) -> u64 {
        let mut memory = MemoryEntry::new(content, topic, valence, arousal);
        let id = memory.id;

        // Assign to cluster
        self.clusterer.assign(&mut memory);

        // Queue for consolidation
        self.engine.queue(id);

        // Store
        let idx = self.memories.len();
        self.memories.push(memory);
        self.memory_index.insert(id, idx);

        id
    }

    /// Retrieve relevant memories for a query
    pub fn retrieve(&mut self, query: &str, limit: usize) -> Vec<u64> {
        // Generate query encoding
        let seed = query.bytes().fold(54321u64, |acc, b| {
            acc.wrapping_mul(31).wrapping_add(b as u64)
        });
        let query_encoding = HV16::random(seed);

        // Apply forgetting decay
        for memory in &mut self.memories {
            self.forgetting.apply_decay(memory);
        }

        // Score and rank - collect IDs and scores
        let mut scored: Vec<_> = self.memories.iter()
            .map(|m| (m.id, self.scorer.score(m, Some(&query_encoding))))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Get top IDs
        let top_ids: Vec<u64> = scored.into_iter()
            .take(limit)
            .map(|(id, _)| id)
            .collect();

        // Access top results (rehearsal effect)
        for id in &top_ids {
            if let Some(idx) = self.memory_index.get(id) {
                self.memories[*idx].access();
            }
        }

        top_ids
    }

    /// Get memory by ID
    pub fn get_memory(&self, id: u64) -> Option<&MemoryEntry> {
        self.memory_index.get(&id).map(|idx| &self.memories[*idx])
    }

    /// Run consolidation if needed
    pub fn maybe_consolidate(&mut self) -> Option<ConsolidationResult> {
        if self.engine.should_consolidate() {
            let result = self.engine.consolidate(&mut self.memories, &self.scorer);

            // Remove forgotten memories
            for id in &result.forgotten {
                if let Some(idx) = self.memory_index.remove(id) {
                    self.memories.swap_remove(idx);
                    // Update index for swapped element
                    if idx < self.memories.len() {
                        let swapped_id = self.memories[idx].id;
                        self.memory_index.insert(swapped_id, idx);
                    }
                }
            }

            self.engine.complete();
            Some(result)
        } else {
            None
        }
    }

    /// Get memories in a cluster
    pub fn cluster_memories(&self, cluster_id: usize) -> Vec<&MemoryEntry> {
        self.memories.iter()
            .filter(|m| m.cluster_id == Some(cluster_id))
            .collect()
    }

    /// Get total memory count
    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }

    /// Get cluster count
    pub fn cluster_count(&self) -> usize {
        self.clusterer.clusters.len()
    }

    /// Get average memory strength
    pub fn average_strength(&self) -> f32 {
        if self.memories.is_empty() {
            return 0.0;
        }
        self.memories.iter().map(|m| m.strength).sum::<f32>() / self.memories.len() as f32
    }
}

impl Default for MemoryConsolidator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // MemoryEntry Tests
    #[test]
    fn test_memory_entry_creation() {
        let entry = MemoryEntry::new(
            "Test memory".to_string(),
            "test".to_string(),
            0.5,
            0.3,
        );
        assert_eq!(entry.content, "Test memory");
        assert_eq!(entry.strength, 1.0);
        assert_eq!(entry.access_count, 1);
    }

    #[test]
    fn test_memory_access() {
        let mut entry = MemoryEntry::new(
            "Test".to_string(),
            "test".to_string(),
            0.0,
            0.0,
        );
        let initial_count = entry.access_count;
        entry.access();
        assert_eq!(entry.access_count, initial_count + 1);
    }

    // MemoryClusterer Tests
    #[test]
    fn test_clusterer_creation() {
        let clusterer = MemoryClusterer::new();
        assert!(clusterer.clusters.is_empty());
    }

    #[test]
    fn test_cluster_assignment() {
        let mut clusterer = MemoryClusterer::new();
        let mut memory = MemoryEntry::new(
            "Test".to_string(),
            "test".to_string(),
            0.0,
            0.0,
        );

        let cluster_id = clusterer.assign(&mut memory);
        assert!(memory.cluster_id.is_some());
        assert_eq!(memory.cluster_id.unwrap(), cluster_id);
    }

    // ImportanceScorer Tests
    #[test]
    fn test_scorer_creation() {
        let scorer = ImportanceScorer::new();
        assert_eq!(scorer.weights.recency, 0.25);
    }

    #[test]
    fn test_scoring() {
        let scorer = ImportanceScorer::new();
        let memory = MemoryEntry::new(
            "Important".to_string(),
            "test".to_string(),
            0.8,
            0.9,
        );

        let score = scorer.score(&memory, None);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    // ForgettingCurve Tests
    #[test]
    fn test_forgetting_curve_creation() {
        let curve = ForgettingCurve::new();
        assert_eq!(curve.params.initial_retention, 1.0);
    }

    #[test]
    fn test_retention_fresh_memory() {
        let curve = ForgettingCurve::new();
        let memory = MemoryEntry::new(
            "Fresh".to_string(),
            "test".to_string(),
            0.0,
            0.0,
        );

        let retention = curve.retention(&memory);
        assert!(retention > 0.9); // Fresh memory should have high retention
    }

    // ConsolidationEngine Tests
    #[test]
    fn test_engine_creation() {
        let engine = ConsolidationEngine::new();
        assert_eq!(engine.state(), ConsolidationState::Active);
    }

    #[test]
    fn test_engine_queue() {
        let mut engine = ConsolidationEngine::new();
        engine.queue(123);
        engine.queue(456);
        assert_eq!(engine.pending.len(), 2);
    }

    // MemoryConsolidator Tests
    #[test]
    fn test_consolidator_creation() {
        let consolidator = MemoryConsolidator::new();
        assert_eq!(consolidator.memory_count(), 0);
    }

    #[test]
    fn test_add_memory() {
        let mut consolidator = MemoryConsolidator::new();
        let id = consolidator.add_memory(
            "Test memory".to_string(),
            "test".to_string(),
            0.5,
            0.3,
        );

        assert_eq!(consolidator.memory_count(), 1);
        assert!(consolidator.get_memory(id).is_some());
    }

    #[test]
    fn test_retrieve() {
        let mut consolidator = MemoryConsolidator::new();

        consolidator.add_memory("Love is important".to_string(), "love".to_string(), 0.8, 0.5);
        consolidator.add_memory("Consciousness matters".to_string(), "consciousness".to_string(), 0.6, 0.4);

        let results = consolidator.retrieve("love", 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_cluster_count() {
        let mut consolidator = MemoryConsolidator::new();

        consolidator.add_memory("Topic A content".to_string(), "A".to_string(), 0.0, 0.0);
        consolidator.add_memory("Topic B content".to_string(), "B".to_string(), 0.0, 0.0);

        assert!(consolidator.cluster_count() >= 1);
    }
}
