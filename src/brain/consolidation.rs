//! Memory Consolidation Core - Week 16 Day 2
//!
//! HDC-based coalition compression for long-term semantic storage.
//! Implements biologically authentic memory consolidation during sleep.
//!
//! Architecture inspired by hippocampal memory consolidation:
//! - **Bundling**: Similar coalitions compressed via HDC superposition
//! - **Importance Scoring**: Emotional + salience + repetition weighting
//! - **Semantic Traces**: Compressed long-term memory representations
//! - **Forgetting Curve**: Exponential decay based on importance and access
//!
//! Revolutionary feature: Sleep-based consolidation with HDC compression

use std::time::{Duration, Instant};
use std::sync::Arc;
use crate::brain::prefrontal::Coalition;

/// Type alias for HDC vectors (shared, immutable)
pub type SharedHdcVector = Arc<Vec<i8>>;

/// Compressed semantic memory trace from consolidated coalitions
#[derive(Debug, Clone)]
pub struct SemanticMemoryTrace {
    /// HDC-compressed coalition bundle
    pub compressed_pattern: SharedHdcVector,

    /// Importance score (0.0-1.0) - higher = remember longer
    pub importance: f32,

    /// How many times this trace has been recalled
    pub access_count: u32,

    /// Last time this trace was accessed (for forgetting curve)
    pub last_accessed: Instant,

    /// Emotional valence from amygdala (-1.0 to 1.0)
    pub emotional_valence: f32,

    /// When this trace was created
    pub creation_time: Instant,
}

impl SemanticMemoryTrace {
    /// Create a new semantic memory trace
    pub fn new(
        compressed_pattern: SharedHdcVector,
        importance: f32,
        emotional_valence: f32,
    ) -> Self {
        let now = Instant::now();
        Self {
            compressed_pattern,
            importance: importance.clamp(0.0, 1.0),
            access_count: 0,
            last_accessed: now,
            emotional_valence: emotional_valence.clamp(-1.0, 1.0),
            creation_time: now,
        }
    }

    /// Record that this trace was accessed (for spaced repetition effect)
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = Instant::now();
    }

    /// Get time since last access
    pub fn time_since_access(&self) -> Duration {
        self.last_accessed.elapsed()
    }

    /// Get age of this trace
    pub fn age(&self) -> Duration {
        self.creation_time.elapsed()
    }
}

/// Configuration for memory consolidation
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Minimum HDC similarity to bundle coalitions together (0.0-1.0)
    pub similarity_threshold: f32,

    /// Minimum importance to retain a trace (0.0-1.0)
    pub importance_threshold: f32,

    /// Weight for salience in importance calculation
    pub salience_weight: f32,

    /// Weight for emotional valence in importance calculation
    pub emotion_weight: f32,

    /// Weight for repetition (coalition size) in importance calculation
    pub repetition_weight: f32,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.85,
            importance_threshold: 0.3,
            salience_weight: 0.4,
            emotion_weight: 0.3,
            repetition_weight: 0.3,
        }
    }
}

/// Memory consolidator - compresses coalitions into semantic traces
pub struct MemoryConsolidator {
    /// Configuration parameters
    config: ConsolidationConfig,

    /// Total number of consolidations performed
    total_consolidated: u64,

    /// Total number of traces forgotten
    total_forgotten: u64,
}

impl MemoryConsolidator {
    /// Create new consolidator with default configuration
    pub fn new() -> Self {
        Self::with_config(ConsolidationConfig::default())
    }

    /// Create new consolidator with custom configuration
    pub fn with_config(config: ConsolidationConfig) -> Self {
        Self {
            config,
            total_consolidated: 0,
            total_forgotten: 0,
        }
    }

    /// Get total consolidations performed
    pub fn total_consolidated(&self) -> u64 {
        self.total_consolidated
    }

    /// Get total traces forgotten
    pub fn total_forgotten(&self) -> u64 {
        self.total_forgotten
    }

    /// Consolidate coalitions into semantic memory traces
    ///
    /// Groups similar coalitions, compresses them via HDC bundling,
    /// and creates long-term semantic traces.
    pub fn consolidate_coalitions(
        &mut self,
        coalitions: Vec<Coalition>,
    ) -> Vec<SemanticMemoryTrace> {
        if coalitions.is_empty() {
            return Vec::new();
        }

        let mut traces = Vec::new();
        let mut processed = vec![false; coalitions.len()];

        // Group similar coalitions and create compressed traces
        for i in 0..coalitions.len() {
            if processed[i] {
                continue;
            }

            // Find all similar coalitions
            let mut similar_group = vec![i];
            for j in (i + 1)..coalitions.len() {
                if processed[j] {
                    continue;
                }

                if self.are_similar(&coalitions[i], &coalitions[j]) {
                    similar_group.push(j);
                    processed[j] = true;
                }
            }
            processed[i] = true;

            // Bundle similar coalitions into compressed trace
            let group_coalitions: Vec<&Coalition> = similar_group
                .iter()
                .map(|&idx| &coalitions[idx])
                .collect();

            if let Some(trace) = self.create_trace(&group_coalitions) {
                traces.push(trace);
                self.total_consolidated += 1;
            }
        }

        traces
    }

    /// Check if two coalitions are similar enough to bundle
    fn are_similar(&self, c1: &Coalition, c2: &Coalition) -> bool {
        // Get HDC encodings from leaders
        let hdc1 = match &c1.leader.hdc_semantic {
            Some(hdc) => hdc,
            None => return false,
        };

        let hdc2 = match &c2.leader.hdc_semantic {
            Some(hdc) => hdc,
            None => return false,
        };

        // Calculate HDC similarity (cosine-like for bipolar vectors)
        let similarity = self.hdc_similarity(hdc1, hdc2);
        similarity >= self.config.similarity_threshold
    }

    /// Calculate HDC similarity between two vectors
    fn hdc_similarity(&self, v1: &[i8], v2: &[i8]) -> f32 {
        if v1.len() != v2.len() {
            return 0.0;
        }

        let matches: i32 = v1
            .iter()
            .zip(v2.iter())
            .map(|(&a, &b)| if a == b { 1 } else { 0 })
            .sum();

        matches as f32 / v1.len() as f32
    }

    /// Create semantic trace from a group of similar coalitions
    fn create_trace(&self, coalitions: &[&Coalition]) -> Option<SemanticMemoryTrace> {
        if coalitions.is_empty() {
            return None;
        }

        // Bundle HDC patterns via superposition
        let compressed = self.bundle_hdc_patterns(coalitions)?;

        // Calculate importance score
        let importance = self.calculate_importance(coalitions);

        // Extract emotional valence (average from coalitions)
        let emotional_valence = coalitions
            .iter()
            .map(|c| {
                // Use leader's salience as proxy for emotional intensity
                // In full implementation, would query Amygdala
                c.leader.salience * if c.strength > 0.7 { 1.0 } else { 0.0 }
            })
            .sum::<f32>()
            / coalitions.len() as f32;

        Some(SemanticMemoryTrace::new(
            compressed,
            importance,
            emotional_valence,
        ))
    }

    /// Bundle multiple HDC patterns via superposition
    ///
    /// HDC bundling: element-wise addition, then majority vote
    fn bundle_hdc_patterns(&self, coalitions: &[&Coalition]) -> Option<SharedHdcVector> {
        if coalitions.is_empty() {
            return None;
        }

        // Get all HDC vectors
        let hdc_vectors: Vec<&Vec<i8>> = coalitions
            .iter()
            .filter_map(|c| c.leader.hdc_semantic.as_ref().map(|arc| arc.as_ref()))
            .collect();

        if hdc_vectors.is_empty() {
            return None;
        }

        // All vectors must have same length
        let dim = hdc_vectors[0].len();
        if !hdc_vectors.iter().all(|v| v.len() == dim) {
            return None;
        }

        // Superposition: element-wise sum, then majority vote
        let mut bundled = vec![0i32; dim];
        for hdc_vec in &hdc_vectors {
            for (i, &val) in hdc_vec.iter().enumerate() {
                bundled[i] += val as i32;
            }
        }

        // Majority vote: positive sum → +1, negative → -1, zero → +1 (bias)
        let compressed: Vec<i8> = bundled
            .iter()
            .map(|&sum| if sum >= 0 { 1 } else { -1 })
            .collect();

        Some(Arc::new(compressed))
    }

    /// Calculate importance score for a group of coalitions
    ///
    /// Weighted combination of:
    /// - Salience (how attention-grabbing)
    /// - Emotional valence (how emotionally significant)
    /// - Repetition (how many coalitions in group)
    fn calculate_importance(&self, coalitions: &[&Coalition]) -> f32 {
        if coalitions.is_empty() {
            return 0.0;
        }

        // Average salience
        let avg_salience = coalitions
            .iter()
            .map(|c| c.leader.salience)
            .sum::<f32>()
            / coalitions.len() as f32;

        // Average strength (proxy for emotional significance)
        let avg_strength = coalitions
            .iter()
            .map(|c| c.strength)
            .sum::<f32>()
            / coalitions.len() as f32;

        // Repetition bonus (more coalitions = more important)
        // Logarithmic scaling: log2(count + 1)
        let repetition_score = ((coalitions.len() + 1) as f32).log2() / 4.0; // Max ~0.5 for 16 coalitions

        // Weighted combination
        let importance = avg_salience * self.config.salience_weight
            + avg_strength * self.config.emotion_weight
            + repetition_score * self.config.repetition_weight;

        importance.clamp(0.0, 1.0)
    }

    /// Apply forgetting curve to semantic traces
    ///
    /// Implements Ebbinghaus forgetting curve:
    /// retention = e^(-t / τ)
    ///
    /// where τ (tau) = time constant, scaled by importance
    pub fn apply_forgetting(&mut self, traces: &mut Vec<SemanticMemoryTrace>) {
        let now = Instant::now();
        let mut to_remove = Vec::new();

        for (idx, trace) in traces.iter_mut().enumerate() {
            let time_since_access = now.duration_since(trace.last_accessed).as_secs_f32();

            // Time constant: important memories last longer
            // Base: 1 day (86400s), scaled by importance
            let tau = 86400.0 * trace.importance;

            // Exponential decay
            let retention = (-time_since_access / tau).exp();

            // Access boost: each access increases retention
            // +10% per access, up to +50% max
            let access_boost = (trace.access_count as f32 * 0.1).min(0.5);

            // Update importance
            trace.importance *= retention * (1.0 + access_boost);

            // Mark for deletion if below threshold
            if trace.importance < self.config.importance_threshold {
                to_remove.push(idx);
            }
        }

        // Remove forgotten traces (reverse order to preserve indices)
        for &idx in to_remove.iter().rev() {
            traces.remove(idx);
            self.total_forgotten += 1;
        }
    }

    /// Apply interference-based forgetting
    ///
    /// **Week 16 Day 4**: Competing similar memories interfere with each other.
    /// When multiple traces are similar (high HDC overlap), they compete for retention.
    ///
    /// Biologically inspired by retroactive and proactive interference:
    /// - New memories interfere with old (retroactive)
    /// - Old memories interfere with new (proactive)
    ///
    /// The winner is determined by importance and recency.
    pub fn apply_interference_forgetting(&mut self, traces: &mut Vec<SemanticMemoryTrace>) {
        if traces.len() < 2 {
            return; // No interference with fewer than 2 traces
        }

        let similarity_threshold = 0.80; // Similar memories compete (more inclusive)
        let mut interference_penalties: Vec<f32> = vec![0.0; traces.len()];
        let mut remove_indices: Vec<usize> = Vec::new();

        // Calculate interference between all pairs
        for i in 0..traces.len() {
            for j in (i + 1)..traces.len() {
                let similarity = self.hdc_similarity(
                    &traces[i].compressed_pattern,
                    &traces[j].compressed_pattern,
                );

                if similarity >= similarity_threshold {
                    // Memories interfere - penalize or remove the weaker/older one
                    let trace_i_strength = traces[i].importance *
                        (1.0 + traces[i].access_count as f32 * 0.1);
                    let trace_j_strength = traces[j].importance *
                        (1.0 + traces[j].access_count as f32 * 0.1);

                    // If strengths differ significantly, drop the weaker trace outright
                    let (stronger, weaker) = if trace_i_strength >= trace_j_strength {
                        (i, j)
                    } else {
                        (j, i)
                    };

                    // If the stronger trace is at least 25% stronger, remove the weaker
                    if trace_i_strength.max(trace_j_strength)
                        >= trace_i_strength.min(trace_j_strength) * 1.25
                    {
                        remove_indices.push(weaker);
                    } else {
                        // Otherwise apply a proportional penalty
                        let interference = (similarity - similarity_threshold) / (1.0 - similarity_threshold);
                        interference_penalties[weaker] += interference * 0.6;
                        // Slight penalty to stronger as well
                        interference_penalties[stronger] += interference * 0.1;
                    }
                }
            }
        }

        // Apply interference penalties
        let mut to_remove = Vec::new();
        for (idx, trace) in traces.iter_mut().enumerate() {
            trace.importance *= (1.0 - interference_penalties[idx]).max(0.0);

            if trace.importance < self.config.importance_threshold {
                to_remove.push(idx);
            }
        }

        // Remove interfered traces
        for idx in remove_indices {
            if idx < traces.len() {
                to_remove.push(idx);
            }
        }

        for &idx in to_remove.iter().rev() {
            traces.remove(idx);
            self.total_forgotten += 1;
        }
    }

    /// Apply context-dependent forgetting
    ///
    /// **Week 16 Day 4**: Memories without matching retrieval context fade faster.
    ///
    /// Biologically inspired by encoding specificity principle:
    /// Memories are better recalled when retrieval context matches encoding context.
    ///
    /// Without context match, memories become harder to access and fade faster.
    ///
    /// # Arguments
    /// * `traces` - Memory traces to process
    /// * `current_context` - Optional HDC vector representing current cognitive context
    pub fn apply_context_dependent_forgetting(
        &mut self,
        traces: &mut Vec<SemanticMemoryTrace>,
        current_context: Option<&SharedHdcVector>,
    ) {
        if current_context.is_none() {
            return; // No context mismatch penalty if no context provided
        }

        let context = current_context.unwrap();
        let mut to_remove = Vec::new();

        for (idx, trace) in traces.iter_mut().enumerate() {
            // Calculate context similarity
            let context_match = self.hdc_similarity(&trace.compressed_pattern, context);

            // Context mismatch accelerates forgetting
            // Strong match (>0.8): no penalty
            // Weak match (<0.5): 30% importance reduction
            let context_penalty = if context_match > 0.8 {
                0.0
            } else if context_match < 0.5 {
                0.3
            } else {
                (0.8 - context_match) / (0.8 - 0.5) * 0.3 // Linear interpolation
            };

            trace.importance *= 1.0 - context_penalty;

            if trace.importance < self.config.importance_threshold {
                to_remove.push(idx);
            }
        }

        // Remove context-mismatched traces
        for &idx in to_remove.iter().rev() {
            traces.remove(idx);
            self.total_forgotten += 1;
        }
    }

    /// Apply emotion-modulated forgetting
    ///
    /// **Week 16 Day 4**: Traumatic memories can be suppressed through active forgetting.
    ///
    /// Biologically inspired by motivated forgetting and memory suppression:
    /// - Positive memories (joy, love) are preserved
    /// - Neutral memories decay normally
    /// - Negative memories (trauma, fear) can be actively suppressed
    ///
    /// Suppression is strongest for extreme negative valence and reduces over time.
    pub fn apply_emotion_modulated_forgetting(&mut self, traces: &mut Vec<SemanticMemoryTrace>) {
        let mut to_remove = Vec::new();

        for (idx, trace) in traces.iter_mut().enumerate() {
            let valence = trace.emotional_valence;

            // Emotional modulation of forgetting
            let emotion_factor = if valence > 0.5 {
                // Strong positive emotions enhance retention
                1.0 + (valence - 0.5) * 0.3 // Up to +15% for maximum joy
            } else if valence < -0.6 {
                // Strong negative emotions trigger suppression
                // BUT: repeated access overrides suppression (can't suppress forever)
                let suppression_strength = (-valence - 0.6) / 0.4; // 0.0 to 1.0
                let access_resistance = (trace.access_count as f32 * 0.15).min(0.9);
                1.0 - suppression_strength * (1.0 - access_resistance) * 0.4 // Up to -40% reduction
            } else {
                // Mild emotions: neutral effect
                1.0
            };

            trace.importance *= emotion_factor;

            if trace.importance < self.config.importance_threshold {
                to_remove.push(idx);
            }
        }

        // Remove suppressed traces
        for &idx in to_remove.iter().rev() {
            traces.remove(idx);
            self.total_forgotten += 1;
        }
    }
}

impl Default for MemoryConsolidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain::prefrontal::AttentionBid;

    #[test]
    fn test_semantic_trace_creation() {
        let hdc_vec = Arc::new(vec![1i8, -1, 1, -1, 1, -1, 1, -1]);
        let trace = SemanticMemoryTrace::new(hdc_vec.clone(), 0.8, 0.5);

        assert_eq!(trace.importance, 0.8);
        assert_eq!(trace.emotional_valence, 0.5);
        assert_eq!(trace.access_count, 0);
        assert_eq!(trace.compressed_pattern.len(), 8);
    }

    #[test]
    fn test_trace_access_recording() {
        let hdc_vec = Arc::new(vec![1i8, -1, 1, -1]);
        let mut trace = SemanticMemoryTrace::new(hdc_vec, 0.7, 0.0);

        assert_eq!(trace.access_count, 0);

        trace.record_access();
        assert_eq!(trace.access_count, 1);

        trace.record_access();
        assert_eq!(trace.access_count, 2);
    }

    #[test]
    fn test_consolidator_creation() {
        let consolidator = MemoryConsolidator::new();

        assert_eq!(consolidator.total_consolidated(), 0);
        assert_eq!(consolidator.total_forgotten(), 0);
    }

    #[test]
    fn test_hdc_similarity_identical() {
        let consolidator = MemoryConsolidator::new();
        let v1 = vec![1i8, -1, 1, -1, 1, -1, 1, -1];
        let v2 = vec![1i8, -1, 1, -1, 1, -1, 1, -1];

        let similarity = consolidator.hdc_similarity(&v1, &v2);
        assert!((similarity - 1.0).abs() < 0.001, "Identical vectors should have similarity 1.0");
    }

    #[test]
    fn test_hdc_similarity_opposite() {
        let consolidator = MemoryConsolidator::new();
        let v1 = vec![1i8, 1, 1, 1, 1, 1, 1, 1];
        let v2 = vec![-1i8, -1, -1, -1, -1, -1, -1, -1];

        let similarity = consolidator.hdc_similarity(&v1, &v2);
        assert!(similarity < 0.001, "Opposite vectors should have similarity ~0.0");
    }

    #[test]
    fn test_hdc_similarity_half_match() {
        let consolidator = MemoryConsolidator::new();
        let v1 = vec![1i8, 1, 1, 1, -1, -1, -1, -1];
        let v2 = vec![1i8, 1, 1, 1, 1, 1, 1, 1];

        let similarity = consolidator.hdc_similarity(&v1, &v2);
        assert!((similarity - 0.5).abs() < 0.001, "Half-matching vectors should have similarity 0.5");
    }

    #[test]
    fn test_bundle_single_pattern() {
        let consolidator = MemoryConsolidator::new();

        let hdc_vec = Arc::new(vec![1i8, -1, 1, -1]);
        let bid = AttentionBid::new("Test", "content")
            .with_salience(0.8)
            .with_hdc_semantic(Some(hdc_vec.clone()));

        let leader = bid.clone();
        let coalition = Coalition {
            members: vec![bid],
            strength: 0.8,
            coherence: 1.0,
            leader,
        };

        let bundled = consolidator.bundle_hdc_patterns(&[&coalition]);
        assert!(bundled.is_some());

        let bundled_vec = bundled.unwrap();
        assert_eq!(bundled_vec.len(), 4);
        assert_eq!(&*bundled_vec, &*hdc_vec); // Should be identical for single pattern
    }

    #[test]
    fn test_bundle_multiple_patterns() {
        let consolidator = MemoryConsolidator::new();

        // Create two similar patterns
        let hdc1 = Arc::new(vec![1i8, 1, 1, 1, -1, -1, -1, -1]);
        let hdc2 = Arc::new(vec![1i8, 1, 1, -1, -1, -1, -1, -1]);

        let bid1 = AttentionBid::new("Test1", "content1")
            .with_salience(0.8)
            .with_hdc_semantic(Some(hdc1));
        let bid2 = AttentionBid::new("Test2", "content2")
            .with_salience(0.7)
            .with_hdc_semantic(Some(hdc2));

        let leader1 = bid1.clone();
        let leader2 = bid2.clone();

        let c1 = Coalition {
            members: vec![bid1],
            strength: 0.8,
            coherence: 1.0,
            leader: leader1,
        };

        let c2 = Coalition {
            members: vec![bid2],
            strength: 0.7,
            coherence: 1.0,
            leader: leader2,
        };

        let bundled = consolidator.bundle_hdc_patterns(&[&c1, &c2]);
        assert!(bundled.is_some());

        let bundled_vec = bundled.unwrap();
        // Bundling should create a pattern that's similar to both inputs
        // First 3 elements: both have [1,1,1] → bundled should be [1,1,1]
        // Element 4: [1] + [-1] = 0 → bias to 1
        // Last 4: both have [-1,-1,-1,-1] → bundled should be [-1,-1,-1,-1]
        assert_eq!(bundled_vec[0], 1);
        assert_eq!(bundled_vec[1], 1);
        assert_eq!(bundled_vec[2], 1);
        // Element 3: sum is 0, so bias to 1
        assert_eq!(bundled_vec[3], 1);
        assert_eq!(bundled_vec[4], -1);
        assert_eq!(bundled_vec[5], -1);
        assert_eq!(bundled_vec[6], -1);
        assert_eq!(bundled_vec[7], -1);
    }

    #[test]
    fn test_importance_calculation_single() {
        let consolidator = MemoryConsolidator::new();

        let hdc_vec = Arc::new(vec![1i8, -1, 1, -1]);
        let bid = AttentionBid::new("Test", "content")
            .with_salience(0.8)
            .with_hdc_semantic(Some(hdc_vec));

        let leader = bid.clone();
        let coalition = Coalition {
            members: vec![bid],
            strength: 0.7,
            coherence: 1.0,
            leader,
        };

        let importance = consolidator.calculate_importance(&[&coalition]);

        // Should be weighted combination:
        // salience (0.8) * 0.4 + strength (0.7) * 0.3 + repetition (~0.25) * 0.3
        // = 0.32 + 0.21 + 0.075 = ~0.605
        assert!(importance > 0.5 && importance < 0.7,
                "Importance should be ~0.6, got {}", importance);
    }

    #[test]
    fn test_importance_increases_with_repetition() {
        let consolidator = MemoryConsolidator::new();

        let hdc_vec = Arc::new(vec![1i8, -1, 1, -1]);
        let bid = AttentionBid::new("Test", "content")
            .with_salience(0.8)
            .with_hdc_semantic(Some(hdc_vec.clone()));

        let leader = bid.clone();
        let coalition = Coalition {
            members: vec![bid.clone()],
            strength: 0.7,
            coherence: 1.0,
            leader: leader.clone(),
        };

        // Single coalition
        let importance_single = consolidator.calculate_importance(&[&coalition]);

        // Four coalitions (more repetition)
        let importance_multiple = consolidator.calculate_importance(&[
            &coalition, &coalition, &coalition, &coalition
        ]);

        assert!(importance_multiple > importance_single,
                "More repetitions should increase importance: {} vs {}",
                importance_multiple, importance_single);
    }

    #[test]
    fn test_consolidate_single_coalition() {
        let mut consolidator = MemoryConsolidator::new();

        let hdc_vec = Arc::new(vec![1i8, -1, 1, -1, 1, -1, 1, -1]);
        let bid = AttentionBid::new("Test", "content")
            .with_salience(0.8)
            .with_hdc_semantic(Some(hdc_vec));

        let leader = bid.clone();
        let coalition = Coalition {
            members: vec![bid],
            strength: 0.8,
            coherence: 1.0,
            leader,
        };

        let traces = consolidator.consolidate_coalitions(vec![coalition]);

        assert_eq!(traces.len(), 1);
        assert_eq!(consolidator.total_consolidated(), 1);
        assert!(traces[0].importance > 0.0);
    }

    #[test]
    fn test_consolidate_similar_coalitions_bundled() {
        let mut consolidator = MemoryConsolidator::new();

        // Create two very similar coalitions (should be bundled)
        let hdc1 = Arc::new(vec![1i8, 1, 1, 1, 1, 1, 1, 1]);
        let hdc2 = Arc::new(vec![1i8, 1, 1, 1, 1, 1, 1, -1]); // 7/8 match = 0.875 similarity

        let bid1 = AttentionBid::new("Test1", "content1")
            .with_salience(0.8)
            .with_hdc_semantic(Some(hdc1));
        let bid2 = AttentionBid::new("Test2", "content2")
            .with_salience(0.7)
            .with_hdc_semantic(Some(hdc2));

        let leader1 = bid1.clone();
        let leader2 = bid2.clone();

        let c1 = Coalition {
            members: vec![bid1],
            strength: 0.8,
            coherence: 1.0,
            leader: leader1,
        };

        let c2 = Coalition {
            members: vec![bid2],
            strength: 0.7,
            coherence: 1.0,
            leader: leader2,
        };

        let traces = consolidator.consolidate_coalitions(vec![c1, c2]);

        // Should create 1 trace (bundled together due to high similarity)
        assert_eq!(traces.len(), 1, "Similar coalitions should be bundled into 1 trace");
        assert_eq!(consolidator.total_consolidated(), 1);
    }

    #[test]
    fn test_consolidate_dissimilar_coalitions_separate() {
        let mut consolidator = MemoryConsolidator::new();

        // Create two dissimilar coalitions (should NOT be bundled)
        let hdc1 = Arc::new(vec![1i8, 1, 1, 1, 1, 1, 1, 1]);
        let hdc2 = Arc::new(vec![-1i8, -1, -1, -1, -1, -1, -1, -1]); // 0/8 match = 0.0 similarity

        let bid1 = AttentionBid::new("Test1", "content1")
            .with_salience(0.8)
            .with_hdc_semantic(Some(hdc1));
        let bid2 = AttentionBid::new("Test2", "content2")
            .with_salience(0.7)
            .with_hdc_semantic(Some(hdc2));

        let leader1 = bid1.clone();
        let leader2 = bid2.clone();

        let c1 = Coalition {
            members: vec![bid1],
            strength: 0.8,
            coherence: 1.0,
            leader: leader1,
        };

        let c2 = Coalition {
            members: vec![bid2],
            strength: 0.7,
            coherence: 1.0,
            leader: leader2,
        };

        let traces = consolidator.consolidate_coalitions(vec![c1, c2]);

        // Should create 2 traces (too dissimilar to bundle)
        assert_eq!(traces.len(), 2, "Dissimilar coalitions should create separate traces");
        assert_eq!(consolidator.total_consolidated(), 2);
    }

    #[test]
    fn test_forgetting_curve_reduces_importance() {
        use std::thread;
        use std::time::Duration as StdDuration;

        let mut consolidator = MemoryConsolidator::new();
        let hdc_vec = Arc::new(vec![1i8, -1, 1, -1]);

        let mut trace = SemanticMemoryTrace::new(hdc_vec, 0.8, 0.0);
        let initial_importance = trace.importance;

        // Wait a tiny bit (can't wait days in test!)
        thread::sleep(StdDuration::from_millis(10));

        // Manually set last_accessed to simulate time passing
        trace.last_accessed = Instant::now() - StdDuration::from_secs(10);

        let mut traces = vec![trace];
        consolidator.apply_forgetting(&mut traces);

        // Importance should decrease (though very slightly due to short time)
        if traces.len() > 0 {
            assert!(traces[0].importance <= initial_importance,
                    "Importance should decrease or stay same: {} vs {}",
                    traces[0].importance, initial_importance);
        }
    }

    #[test]
    fn test_forgetting_removes_low_importance() {
        let mut consolidator = MemoryConsolidator::new();
        let hdc_vec = Arc::new(vec![1i8, -1, 1, -1]);

        // Create trace with very low importance
        let mut trace = SemanticMemoryTrace::new(hdc_vec, 0.1, 0.0);

        // Simulate a lot of time passing
        trace.last_accessed = Instant::now() - Duration::from_secs(86400 * 30); // 30 days

        let mut traces = vec![trace];
        consolidator.apply_forgetting(&mut traces);

        // Low importance trace should be forgotten after 30 days
        assert_eq!(traces.len(), 0, "Low importance trace should be forgotten");
        assert_eq!(consolidator.total_forgotten(), 1);
    }

    #[test]
    fn test_access_boost_prevents_forgetting() {
        let mut consolidator = MemoryConsolidator::new();
        let hdc_vec = Arc::new(vec![1i8, -1, 1, -1]);

        // Use higher importance so access boost can make a difference
        let mut trace = SemanticMemoryTrace::new(hdc_vec, 0.8, 0.0);

        // Record many accesses
        for _ in 0..10 {
            trace.record_access();
        }

        // Simulate moderate time passing (1 day instead of 5)
        // With importance 0.8: tau = 86400 * 0.8 = 69120 seconds
        // After 1 day (86400 sec): retention = exp(-86400/69120) = exp(-1.25) ≈ 0.287
        // With access boost: importance *= 0.287 * 1.5 = 0.43 (above 0.3 threshold)
        trace.last_accessed = Instant::now() - Duration::from_secs(86400); // 1 day

        let initial_importance = trace.importance;
        let mut traces = vec![trace];
        consolidator.apply_forgetting(&mut traces);

        // Should still exist (access boost helps retention)
        assert_eq!(traces.len(), 1, "Frequently accessed trace should be retained");

        // But importance should still decrease
        assert!(traces[0].importance < initial_importance,
                "Even with access boost, some decay should occur: {} vs {}",
                traces[0].importance, initial_importance);

        // Should be above threshold due to access boost
        assert!(traces[0].importance > 0.3,
                "Access boost should keep importance above threshold: {}",
                traces[0].importance);
    }

    // ========================================
    // Week 16 Day 4: Advanced Forgetting Mechanisms Tests
    // ========================================

    #[test]
    fn test_interference_forgetting_with_similar_patterns() {
        let mut consolidator = MemoryConsolidator::new();

        // Create very similar patterns (>90% similarity)
        let similar1 = Arc::new(vec![1i8, 1, 1, 1, 1, 1, 1, 1]);
        let similar2 = Arc::new(vec![1i8, 1, 1, 1, 1, 1, 1, -1]); // Only last bit different

        // First trace is stronger (higher importance + accesses)
        let mut trace1 = SemanticMemoryTrace::new(similar1, 0.9, 0.0);
        trace1.access_count = 10;

        // Second trace is weaker
        let mut trace2 = SemanticMemoryTrace::new(similar2, 0.7, 0.0);
        trace2.access_count = 2;

        let mut traces = vec![trace1, trace2];
        consolidator.apply_interference_forgetting(&mut traces);

        // Weaker similar trace should suffer interference and be removed
        // Stronger trace should remain
        assert_eq!(traces.len(), 1, "Should have only one trace after interference");
        assert!(traces[0].access_count == 10,
               "Remaining trace should be the stronger one");
    }

    #[test]
    fn test_interference_forgetting_no_effect_on_dissimilar() {
        let mut consolidator = MemoryConsolidator::new();

        // Create dissimilar patterns (<90% similarity)
        let pattern1 = Arc::new(vec![1i8, 1, 1, 1]);
        let pattern2 = Arc::new(vec![-1i8, -1, -1, -1]);

        let trace1 = SemanticMemoryTrace::new(pattern1, 0.8, 0.0);
        let trace2 = SemanticMemoryTrace::new(pattern2, 0.8, 0.0);

        let mut traces = vec![trace1, trace2];
        consolidator.apply_interference_forgetting(&mut traces);

        // Both should remain (not similar enough to interfere)
        assert_eq!(traces.len(), 2, "Dissimilar traces should not interfere");
    }

    #[test]
    fn test_interference_strength_proportional_to_similarity() {
        let mut consolidator = MemoryConsolidator::new();

        // Create patterns with varying similarity
        let base = Arc::new(vec![1i8, 1, 1, 1, 1, 1, 1, 1]);
        let very_similar = Arc::new(vec![1i8, 1, 1, 1, 1, 1, 1, -1]); // 7/8 = 87.5% similar

        // Equal strength traces
        let trace1 = SemanticMemoryTrace::new(base, 0.8, 0.0);
        let trace2 = SemanticMemoryTrace::new(very_similar, 0.8, 0.0);

        let initial_importance = trace2.importance;
        let mut traces = vec![trace1, trace2];

        consolidator.apply_interference_forgetting(&mut traces);

        // One should be interfered with (reduced importance or removed)
        if traces.len() == 2 {
            // If both remain, at least one should have reduced importance
            assert!(traces[0].importance < initial_importance ||
                   traces[1].importance < initial_importance,
                   "Interference should reduce importance");
        } else {
            // One was removed
            assert_eq!(traces.len(), 1, "Should remove one interfered trace");
        }
    }

    #[test]
    fn test_context_dependent_forgetting_with_context_match() {
        let mut consolidator = MemoryConsolidator::new();

        // Create trace and matching context
        let pattern = Arc::new(vec![1i8, 1, 1, 1]);
        let context = Arc::new(vec![1i8, 1, 1, 1]); // Perfect match

        let trace = SemanticMemoryTrace::new(pattern, 0.8, 0.0);
        let initial_importance = trace.importance;

        let mut traces = vec![trace];
        consolidator.apply_context_dependent_forgetting(&mut traces, Some(&context));

        // Should remain with minimal penalty (>80% match = no penalty)
        assert_eq!(traces.len(), 1, "Matching context should preserve trace");
        assert!((traces[0].importance - initial_importance).abs() < 0.01,
               "Strong context match should have no penalty");
    }

    #[test]
    fn test_context_dependent_forgetting_with_context_mismatch() {
        let mut consolidator = MemoryConsolidator::new();

        // Create trace and mismatched context
        let pattern = Arc::new(vec![1i8, 1, 1, 1]);
        let context = Arc::new(vec![-1i8, -1, -1, -1]); // Complete mismatch

        let mut trace = SemanticMemoryTrace::new(pattern, 0.5, 0.0); // Lower importance
        trace.last_accessed = Instant::now();

        let initial_importance = trace.importance;
        let mut traces = vec![trace];

        consolidator.apply_context_dependent_forgetting(&mut traces, Some(&context));

        // Should be removed or heavily penalized (<50% match = 30% penalty)
        if traces.len() == 1 {
            assert!(traces[0].importance < initial_importance * 0.8,
                   "Context mismatch should significantly reduce importance");
        } else {
            assert_eq!(traces.len(), 0, "Weak trace with context mismatch should be removed");
        }
    }

    #[test]
    fn test_context_dependent_forgetting_no_context_no_penalty() {
        let mut consolidator = MemoryConsolidator::new();

        let pattern = Arc::new(vec![1i8, -1, 1, -1]);
        let trace = SemanticMemoryTrace::new(pattern, 0.8, 0.0);
        let initial_importance = trace.importance;

        let mut traces = vec![trace];
        consolidator.apply_context_dependent_forgetting(&mut traces, None);

        // Should remain unchanged (no context = no penalty)
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].importance, initial_importance,
                  "No context should mean no penalty");
    }

    #[test]
    fn test_emotion_modulated_forgetting_positive_preservation() {
        let mut consolidator = MemoryConsolidator::new();

        // Create positive emotional memory (valence > 0.5)
        let pattern = Arc::new(vec![1i8, -1, 1, -1]);
        let trace = SemanticMemoryTrace::new(pattern, 0.6, 0.8); // Strong positive emotion

        let initial_importance = trace.importance;
        let mut traces = vec![trace];

        consolidator.apply_emotion_modulated_forgetting(&mut traces);

        // Positive emotion should enhance retention (increase importance)
        assert_eq!(traces.len(), 1, "Positive memory should be preserved");
        assert!(traces[0].importance > initial_importance,
               "Positive emotion should enhance retention: {} vs {}",
               traces[0].importance, initial_importance);
    }

    #[test]
    fn test_emotion_modulated_forgetting_negative_suppression() {
        let mut consolidator = MemoryConsolidator::new();

        // Create negative emotional memory (valence < -0.6)
        let pattern = Arc::new(vec![1i8, -1, 1, -1]);
        let mut trace = SemanticMemoryTrace::new(pattern, 0.6, -0.9); // Strong negative emotion
        trace.access_count = 0; // No repeated access to override suppression

        let initial_importance = trace.importance;
        let mut traces = vec![trace];

        consolidator.apply_emotion_modulated_forgetting(&mut traces);

        // Negative emotion should trigger suppression (reduce importance)
        assert_eq!(traces.len(), 1);
        assert!(traces[0].importance < initial_importance,
               "Negative emotion should reduce importance: {} vs {}",
               traces[0].importance, initial_importance);
    }

    #[test]
    fn test_emotion_modulated_forgetting_repeated_access_overrides_suppression() {
        let mut consolidator = MemoryConsolidator::new();

        // Create negative memory with many accesses
        let pattern = Arc::new(vec![1i8, -1, 1, -1]);
        let mut trace = SemanticMemoryTrace::new(pattern, 0.7, -0.9); // Strong negative
        trace.access_count = 10; // Many accesses = can't suppress

        let initial_importance = trace.importance;
        let mut traces = vec![trace];

        consolidator.apply_emotion_modulated_forgetting(&mut traces);

        // Access count should resist suppression
        assert_eq!(traces.len(), 1, "Repeatedly accessed memory resists suppression");

        // Should have minimal reduction due to access resistance
        let importance_ratio = traces[0].importance / initial_importance;
        assert!(importance_ratio > 0.9,
               "High access count should strongly resist suppression: ratio {}",
               importance_ratio);
    }

    #[test]
    fn test_combined_forgetting_mechanisms() {
        let mut consolidator = MemoryConsolidator::new();

        // Create a scenario combining multiple forgetting mechanisms:
        // 1. Interference (similar patterns)
        // 2. Context mismatch
        // 3. Negative emotion

        let pattern1 = Arc::new(vec![1i8, 1, 1, 1]);
        let pattern2 = Arc::new(vec![1i8, 1, 1, -1]); // Similar to pattern1
        let context = Arc::new(vec![-1i8, -1, -1, -1]); // Mismatches both

        // Trace 1: Strong, positive emotion, many accesses
        let mut trace1 = SemanticMemoryTrace::new(pattern1.clone(), 0.9, 0.7);
        trace1.access_count = 15;

        // Trace 2: Weak, negative emotion, no accesses
        let trace2 = SemanticMemoryTrace::new(pattern2, 0.5, -0.8);

        let mut traces = vec![trace1, trace2];

        // Apply all forgetting mechanisms
        consolidator.apply_interference_forgetting(&mut traces);
        consolidator.apply_context_dependent_forgetting(&mut traces, Some(&context));
        consolidator.apply_emotion_modulated_forgetting(&mut traces);

        // Weak negative trace should be removed by combination of factors
        // Strong positive trace should survive despite context mismatch
        assert!(traces.len() <= 1, "Combined forgetting should remove weak traces");

        if traces.len() == 1 {
            assert!(traces[0].access_count == 15,
                   "Remaining trace should be the strong, frequently accessed one");
        }
    }

    #[test]
    fn test_forgetting_preserves_important_memories() {
        let mut consolidator = MemoryConsolidator::new();

        // Create highly important memory despite negative factors
        let pattern = Arc::new(vec![1i8, -1, 1, -1]);
        let mut trace = SemanticMemoryTrace::new(pattern, 0.95, -0.5); // High importance, mild negative
        trace.access_count = 20; // Frequently accessed

        let initial_importance = trace.importance;
        let mut traces = vec![trace];

        // Apply all forgetting mechanisms
        let context = Arc::new(vec![-1i8, 1, -1, 1]); // Somewhat mismatched
        consolidator.apply_interference_forgetting(&mut traces);
        consolidator.apply_context_dependent_forgetting(&mut traces, Some(&context));
        consolidator.apply_emotion_modulated_forgetting(&mut traces);

        // Should survive due to high importance and access count
        assert_eq!(traces.len(), 1, "Important, frequently accessed memory should survive");
        assert!(traces[0].importance > consolidator.config.importance_threshold,
               "Should remain above threshold: {}",
               traces[0].importance);
    }
}
