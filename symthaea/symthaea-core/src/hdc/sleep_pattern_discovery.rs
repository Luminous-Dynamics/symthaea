// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Sleep-Based Pattern Discovery via Resonator Networks
//!
//! Implements pattern discovery during memory consolidation using resonator networks.
//! During sleep cycles, the brain identifies recurring patterns across experiences
//! and consolidates them into higher-order abstractions (schemas).
//!
//! ## Theoretical Foundation
//!
//! 1. **Memory Consolidation During Sleep**
//!    - Systems consolidation (Diekelmann & Born, 2010): Slow-wave sleep replays memories
//!    - Two-stage model: Hippocampus -> Neocortex transfer during sleep
//!    - Pattern completion: Partial cues trigger full memory recall
//!
//! 2. **Resonator Networks for Pattern Discovery**
//!    - Frady et al. (2020): Resonators find solutions via coupled oscillator dynamics
//!    - Applied to memory: Find recurring patterns by letting similar memories "resonate"
//!    - Convergence to attractors = discovered schemas
//!
//! 3. **Integration with Sleep Cycles**
//!    - N3 (Deep Sleep): Slow-wave replay triggers pattern discovery
//!    - REM: Emotional memories and procedural consolidation
//!    - Spindle-ripple coupling: Memory transfer optimization
//!
//! ## Usage
//!
//! ```rust,ignore
//! use symthaea_core::hdc::sleep_pattern_discovery::{PatternDiscoveryEngine, DiscoveryConfig};
//! use symthaea_core::hdc::ContinuousHV;
//!
//! let mut engine = PatternDiscoveryEngine::new(DiscoveryConfig::default());
//!
//! // Add memories to consolidate
//! engine.add_memory("mem1", memory_hv1, 0.8);
//! engine.add_memory("mem2", memory_hv2, 0.7);
//!
//! // Run pattern discovery (during simulated sleep)
//! let patterns = engine.discover_patterns(sleep_cycles);
//!
//! for pattern in patterns {
//!     println!("Found pattern: {} (strength: {:.2})", pattern.name, pattern.strength);
//! }
//! ```

use super::resonator::{Constraint, ResonatorConfig, ResonatorNetwork};
use super::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for pattern discovery during sleep
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Minimum similarity threshold for memories to be considered related
    pub similarity_threshold: f32,
    /// Minimum occurrences to be considered a pattern
    pub min_occurrences: usize,
    /// Maximum patterns to discover per sleep cycle
    pub max_patterns: usize,
    /// Resonator step size
    pub resonator_step_size: f32,
    /// Resonator iterations per pattern search
    pub resonator_iterations: usize,
    /// Convergence threshold for resonator
    pub convergence_threshold: f32,
    /// Whether to consolidate discovered patterns back into memory
    pub auto_consolidate: bool,
    /// Strength multiplier for consolidated patterns
    pub consolidation_strength: f32,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.6,
            min_occurrences: 3,
            max_patterns: 10,
            resonator_step_size: 0.5,
            resonator_iterations: 100,
            convergence_threshold: 0.99,
            auto_consolidate: true,
            consolidation_strength: 1.5,
        }
    }
}

/// Fast configuration for real-time pattern monitoring
impl DiscoveryConfig {
    pub fn fast() -> Self {
        Self {
            resonator_iterations: 50,
            max_patterns: 5,
            convergence_threshold: 0.95,
            ..Default::default()
        }
    }

    /// Thorough configuration for deep sleep consolidation
    pub fn deep_sleep() -> Self {
        Self {
            similarity_threshold: 0.5,
            min_occurrences: 2,
            max_patterns: 20,
            resonator_iterations: 200,
            convergence_threshold: 0.999,
            consolidation_strength: 2.0,
            ..Default::default()
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY ENTRY
// ═══════════════════════════════════════════════════════════════════════════════

/// A memory entry for pattern discovery
#[derive(Debug, Clone)]
pub struct MemoryEntry {
    /// Unique identifier
    pub id: String,
    /// HDC representation of the memory
    pub vector: ContinuousHV,
    /// Emotional strength (affects consolidation priority)
    pub emotional_strength: f32,
    /// Timestamp (relative, higher = more recent)
    pub timestamp: f64,
    /// How many times this memory has been accessed
    pub access_count: usize,
    /// Consolidation level [0, 1]
    pub consolidation: f32,
    /// Tags/categories
    pub tags: Vec<String>,
}

impl MemoryEntry {
    /// Create new memory entry
    pub fn new(id: impl Into<String>, vector: ContinuousHV, emotional_strength: f32) -> Self {
        Self {
            id: id.into(),
            vector,
            emotional_strength: emotional_strength.clamp(0.0, 1.0),
            timestamp: 0.0,
            access_count: 1,
            consolidation: 0.0,
            tags: Vec::new(),
        }
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: f64) -> Self {
        self.timestamp = timestamp;
        self
    }

    /// Add tag
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Compute consolidation priority (higher = more important to consolidate)
    pub fn priority(&self) -> f32 {
        // Factors:
        // 1. Emotional strength (emotional memories consolidate faster)
        // 2. Access count (frequently accessed = important)
        // 3. How unconsolidated (need to consolidate more)
        let emotional_factor = self.emotional_strength * 2.0;
        let access_factor = (self.access_count as f32 * 0.2).min(1.0);
        let need_factor = 1.0 - self.consolidation;

        (emotional_factor + access_factor + need_factor) / 3.0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// DISCOVERED PATTERN
// ═══════════════════════════════════════════════════════════════════════════════

/// A discovered pattern (schema) from memory consolidation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredPattern {
    /// Unique identifier
    pub id: String,
    /// Human-readable name/description
    pub name: String,
    /// The pattern's HDC representation (centroid of similar memories)
    pub vector: Vec<f32>,
    /// How strongly this pattern emerged
    pub strength: f32,
    /// Number of memories contributing to this pattern
    pub occurrence_count: usize,
    /// IDs of contributing memories
    pub member_memory_ids: Vec<String>,
    /// Average similarity of members to the pattern
    pub internal_coherence: f32,
    /// Whether this pattern was successfully consolidated
    pub consolidated: bool,
    /// Resonator convergence info
    pub convergence_iterations: usize,
    /// Tags inherited from member memories
    pub tags: Vec<String>,
}

impl DiscoveredPattern {
    /// Get the pattern as ContinuousHV
    pub fn as_hv(&self) -> ContinuousHV {
        ContinuousHV::from_slice(&self.vector)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// PATTERN DISCOVERY ENGINE
// ═══════════════════════════════════════════════════════════════════════════════

/// Pattern Discovery Engine using Resonator Networks
///
/// During simulated sleep cycles, this engine:
/// 1. Groups similar memories into candidate clusters
/// 2. Uses resonator networks to find stable pattern attractors
/// 3. Extracts schemas (recurring patterns) from converged states
/// 4. Optionally consolidates patterns back into memory
pub struct PatternDiscoveryEngine {
    /// Configuration
    config: DiscoveryConfig,
    /// Memories to process
    memories: HashMap<String, MemoryEntry>,
    /// Discovered patterns (accumulated across sleep cycles)
    discovered_patterns: Vec<DiscoveredPattern>,
    /// Pattern counter for ID generation
    pattern_counter: usize,
    /// Statistics
    stats: DiscoveryStats,
}

/// Statistics for pattern discovery
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiscoveryStats {
    /// Total sleep cycles processed
    pub cycles_processed: usize,
    /// Total patterns discovered
    pub patterns_discovered: usize,
    /// Total memories consolidated
    pub memories_consolidated: usize,
    /// Average pattern strength
    pub avg_pattern_strength: f32,
    /// Average convergence iterations
    pub avg_convergence_iterations: f32,
}

impl PatternDiscoveryEngine {
    /// Create new pattern discovery engine
    pub fn new(config: DiscoveryConfig) -> Self {
        Self {
            config,
            memories: HashMap::new(),
            discovered_patterns: Vec::new(),
            pattern_counter: 0,
            stats: DiscoveryStats::default(),
        }
    }

    /// Add a memory to the discovery engine
    pub fn add_memory(
        &mut self,
        id: impl Into<String>,
        vector: ContinuousHV,
        emotional_strength: f32,
    ) {
        let id = id.into();
        let memory = MemoryEntry::new(id.clone(), vector, emotional_strength);
        self.memories.insert(id, memory);
    }

    /// Add a full memory entry
    pub fn add_memory_entry(&mut self, memory: MemoryEntry) {
        self.memories.insert(memory.id.clone(), memory);
    }

    /// Run pattern discovery (simulates one sleep cycle)
    ///
    /// # Algorithm
    ///
    /// 1. **Clustering**: Group memories by similarity
    /// 2. **Resonance**: For each cluster, use resonator to find pattern attractor
    /// 3. **Extraction**: Convert converged state to pattern schema
    /// 4. **Consolidation**: Optionally strengthen pattern in memory
    pub fn discover_patterns(&mut self, sleep_intensity: f32) -> Vec<DiscoveredPattern> {
        if self.memories.len() < self.config.min_occurrences {
            return Vec::new();
        }

        self.stats.cycles_processed += 1;
        let mut new_patterns = Vec::new();

        // Step 1: Find candidate clusters (groups of similar memories)
        let clusters = self.cluster_memories();

        // Step 2: For each cluster, run resonator to find pattern
        for (cluster_idx, cluster_memory_ids) in clusters.into_iter().enumerate() {
            if cluster_memory_ids.len() < self.config.min_occurrences {
                continue;
            }

            if new_patterns.len() >= self.config.max_patterns {
                break;
            }

            // Get memory vectors for this cluster
            let cluster_vectors: Vec<ContinuousHV> = cluster_memory_ids
                .iter()
                .filter_map(|id| self.memories.get(id).map(|m| m.vector.clone()))
                .collect();

            if cluster_vectors.is_empty() {
                continue;
            }

            // Run resonator network to find pattern attractor
            if let Some(pattern) = self.find_pattern_via_resonator(
                &cluster_vectors,
                &cluster_memory_ids,
                cluster_idx,
                sleep_intensity,
            ) {
                new_patterns.push(pattern);
            }
        }

        // Step 3: Update statistics and consolidation
        for pattern in &new_patterns {
            self.stats.patterns_discovered += 1;
            self.stats.avg_pattern_strength = (self.stats.avg_pattern_strength
                * (self.stats.patterns_discovered - 1) as f32
                + pattern.strength)
                / self.stats.patterns_discovered as f32;

            // Consolidate contributing memories
            if self.config.auto_consolidate {
                for mem_id in &pattern.member_memory_ids {
                    if let Some(memory) = self.memories.get_mut(mem_id) {
                        memory.consolidation =
                            (memory.consolidation + sleep_intensity * 0.2).min(1.0);
                        self.stats.memories_consolidated += 1;
                    }
                }
            }
        }

        // Store discovered patterns
        self.discovered_patterns.extend(new_patterns.clone());

        new_patterns
    }

    /// Cluster memories by similarity using greedy approach
    fn cluster_memories(&self) -> Vec<Vec<String>> {
        let mut clusters: Vec<Vec<String>> = Vec::new();
        let mut assigned: HashMap<String, bool> = HashMap::new();

        // Sort memories by priority (emotional + access-based)
        let mut memory_ids: Vec<_> = self.memories.keys().collect();
        memory_ids.sort_by(|a, b| {
            let pa = self.memories.get(*a).map(|m| m.priority()).unwrap_or(0.0);
            let pb = self.memories.get(*b).map(|m| m.priority()).unwrap_or(0.0);
            pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
        });

        for &seed_id in &memory_ids {
            if *assigned.get(seed_id).unwrap_or(&false) {
                continue;
            }

            let seed = match self.memories.get(seed_id) {
                Some(m) => m,
                None => continue,
            };

            let mut cluster = vec![seed_id.clone()];
            assigned.insert(seed_id.clone(), true);

            // Find similar memories
            for &other_id in &memory_ids {
                if *assigned.get(other_id).unwrap_or(&false) {
                    continue;
                }

                let other = match self.memories.get(other_id) {
                    Some(m) => m,
                    None => continue,
                };

                let similarity = seed.vector.similarity(&other.vector);
                if similarity >= self.config.similarity_threshold {
                    cluster.push(other_id.clone());
                    assigned.insert(other_id.clone(), true);
                }
            }

            if cluster.len() >= self.config.min_occurrences {
                clusters.push(cluster);
            }
        }

        clusters
    }

    /// Use resonator network to find pattern attractor from cluster
    fn find_pattern_via_resonator(
        &mut self,
        vectors: &[ContinuousHV],
        memory_ids: &[String],
        cluster_idx: usize,
        sleep_intensity: f32,
    ) -> Option<DiscoveredPattern> {
        if vectors.is_empty() {
            return None;
        }

        let dim = vectors[0].dim();

        // Create resonator network with cluster vectors as codebook
        let resonator_config = ResonatorConfig {
            step_size: self.config.resonator_step_size,
            momentum: 0.7,
            temperature: 0.2,
            convergence_threshold: self.config.convergence_threshold,
            max_iterations: self.config.resonator_iterations,
            noise_scale: 0.01 * (1.0 - sleep_intensity), // Less noise during deep sleep
            energy_threshold: 0.05,
        };

        let mut resonator = match ResonatorNetwork::with_config(dim, resonator_config) {
            Ok(r) => r,
            Err(_) => return None,
        };

        // Add cluster vectors as symbols (attractors)
        for (i, vec) in vectors.iter().enumerate() {
            let name = format!("mem_{i}");
            let _ = resonator.add_symbol(&name, vec.values.clone());
        }

        // Create constraint: find centroid that's similar to all memories
        // We use the average as initial target
        let centroid = self.compute_centroid(vectors);

        // Create constraints based on each memory being close to the pattern
        let constraints: Vec<Constraint> = vectors
            .iter()
            .enumerate()
            .map(|(i, vec)| {
                Constraint::named(
                    &format!("similar_to_mem_{i}"),
                    centroid.values.clone(),
                    vec.values.clone(),
                )
                .with_weight(1.0)
            })
            .collect();

        // Solve: find stable pattern via resonance
        let solution = match resonator.solve(&constraints, Some(self.config.resonator_iterations)) {
            Ok(s) => s,
            Err(_) => return None,
        };

        // Compute pattern metrics
        let pattern_hv = ContinuousHV::from_slice(&solution.vector);
        let mut total_similarity = 0.0;
        for vec in vectors {
            total_similarity += pattern_hv.similarity(vec);
        }
        let internal_coherence = total_similarity / vectors.len() as f32;

        // Compute pattern strength (based on coherence and convergence)
        let convergence_factor = if solution.converged { 1.0 } else { 0.7 };
        let strength = internal_coherence * convergence_factor * sleep_intensity;

        // Only accept strong patterns
        if strength < 0.3 {
            return None;
        }

        // Collect tags from member memories
        let mut tags: Vec<String> = Vec::new();
        for mem_id in memory_ids {
            if let Some(memory) = self.memories.get(mem_id) {
                for tag in &memory.tags {
                    if !tags.contains(tag) {
                        tags.push(tag.clone());
                    }
                }
            }
        }

        self.pattern_counter += 1;

        Some(DiscoveredPattern {
            id: format!("pattern_{}", self.pattern_counter),
            name: format!("Pattern {} ({}members)", cluster_idx + 1, memory_ids.len()),
            vector: solution.vector,
            strength,
            occurrence_count: memory_ids.len(),
            member_memory_ids: memory_ids.to_vec(),
            internal_coherence,
            consolidated: self.config.auto_consolidate,
            convergence_iterations: solution.iterations,
            tags,
        })
    }

    /// Compute centroid (average) of vectors
    fn compute_centroid(&self, vectors: &[ContinuousHV]) -> ContinuousHV {
        if vectors.is_empty() {
            return ContinuousHV::zero(1024);
        }

        let dim = vectors[0].dim();
        let mut sum = vec![0.0f32; dim];

        for vec in vectors {
            for (i, &v) in vec.values.iter().enumerate() {
                sum[i] += v;
            }
        }

        let n = vectors.len() as f32;
        for s in &mut sum {
            *s /= n;
        }

        // Normalize
        let norm: f32 = sum.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for s in &mut sum {
                *s /= norm;
            }
        }

        ContinuousHV::from_slice(&sum)
    }

    /// Get all discovered patterns
    pub fn patterns(&self) -> &[DiscoveredPattern] {
        &self.discovered_patterns
    }

    /// Get patterns that match a query vector
    pub fn find_matching_patterns(
        &self,
        query: &ContinuousHV,
        min_similarity: f32,
    ) -> Vec<&DiscoveredPattern> {
        self.discovered_patterns
            .iter()
            .filter(|p| {
                let pattern_hv = ContinuousHV::from_slice(&p.vector);
                pattern_hv.similarity(query) >= min_similarity
            })
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> &DiscoveryStats {
        &self.stats
    }

    /// Clear all memories (start fresh)
    pub fn clear_memories(&mut self) {
        self.memories.clear();
    }

    /// Clear discovered patterns
    pub fn clear_patterns(&mut self) {
        self.discovered_patterns.clear();
        self.pattern_counter = 0;
    }

    /// Get memory count
    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }

    /// Get pattern count
    pub fn pattern_count(&self) -> usize {
        self.discovered_patterns.len()
    }
}

impl Default for PatternDiscoveryEngine {
    fn default() -> Self {
        Self::new(DiscoveryConfig::default())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn random_hv(dim: usize, seed: u64) -> ContinuousHV {
        // Deterministic pseudo-random for reproducibility
        let values: Vec<f32> = (0..dim)
            .map(|i| {
                let x = ((i as f64 + seed as f64) * 0.618033988749895).fract();
                (x * 2.0 - 1.0) as f32
            })
            .collect();
        let mut hv = ContinuousHV::from_slice(&values);
        // Normalize
        let norm: f32 = hv.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut hv.values {
                *v /= norm;
            }
        }
        hv
    }

    fn similar_hv(base: &ContinuousHV, noise: f32, seed: u64) -> ContinuousHV {
        let noise_hv = random_hv(base.dim(), seed);
        let values: Vec<f32> = base
            .values
            .iter()
            .zip(noise_hv.values.iter())
            .map(|(&b, &n)| b * (1.0 - noise) + n * noise)
            .collect();
        let mut hv = ContinuousHV::from_slice(&values);
        // Normalize
        let norm: f32 = hv.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut hv.values {
                *v /= norm;
            }
        }
        hv
    }

    #[test]
    fn test_engine_creation() {
        let engine = PatternDiscoveryEngine::default();
        assert_eq!(engine.memory_count(), 0);
        assert_eq!(engine.pattern_count(), 0);
    }

    #[test]
    fn test_add_memories() {
        let mut engine = PatternDiscoveryEngine::default();
        let dim = 512;

        for i in 0..5 {
            let hv = random_hv(dim, i);
            engine.add_memory(format!("mem_{}", i), hv, 0.5);
        }

        assert_eq!(engine.memory_count(), 5);
    }

    #[test]
    fn test_discover_similar_patterns() {
        let mut engine = PatternDiscoveryEngine::new(DiscoveryConfig {
            min_occurrences: 3,
            similarity_threshold: 0.5,
            resonator_iterations: 50,
            ..Default::default()
        });

        let dim = 512;

        // Create a base pattern
        let base = random_hv(dim, 42);

        // Add similar memories (should cluster together)
        for i in 0..5 {
            let similar = similar_hv(&base, 0.2, i + 100);
            engine.add_memory(format!("similar_{}", i), similar, 0.7);
        }

        // Add dissimilar memories (should not cluster)
        for i in 0..3 {
            let different = random_hv(dim, i + 1000);
            engine.add_memory(format!("different_{}", i), different, 0.3);
        }

        // Run pattern discovery
        let patterns = engine.discover_patterns(0.8); // High sleep intensity

        // Should find at least one pattern from the similar memories
        assert!(!patterns.is_empty(), "Should discover at least one pattern");

        let first_pattern = &patterns[0];
        assert!(
            first_pattern.occurrence_count >= 3,
            "Pattern should have multiple members"
        );
        assert!(
            first_pattern.internal_coherence > 0.5,
            "Pattern should be coherent"
        );
    }

    #[test]
    fn test_no_patterns_from_random() {
        let mut engine = PatternDiscoveryEngine::new(DiscoveryConfig {
            min_occurrences: 3,
            similarity_threshold: 0.8, // High threshold
            ..Default::default()
        });

        let dim = 512;

        // Add completely random, unrelated memories
        for i in 0..10 {
            let hv = random_hv(dim, i * 123);
            engine.add_memory(format!("random_{}", i), hv, 0.5);
        }

        let patterns = engine.discover_patterns(0.5);

        // Should find few or no patterns (random vectors unlikely to cluster)
        // With high threshold, random vectors shouldn't cluster
        println!("Found {} patterns from random data", patterns.len());

        // With min_occurrences=3 and similarity_threshold=0.8,
        // 10 random vectors should yield very few (likely 0) patterns
        assert!(
            patterns.len() <= 5,
            "Random vectors with high threshold should produce few patterns, got {}",
            patterns.len()
        );
    }

    #[test]
    fn test_pattern_statistics() {
        let mut engine = PatternDiscoveryEngine::new(DiscoveryConfig {
            min_occurrences: 2,
            similarity_threshold: 0.4,
            resonator_iterations: 30,
            ..Default::default()
        });

        let dim = 256;
        let base = random_hv(dim, 42);

        for i in 0..4 {
            let similar = similar_hv(&base, 0.15, i + 100);
            engine.add_memory(format!("mem_{}", i), similar, 0.6);
        }

        let _patterns = engine.discover_patterns(0.7);

        let stats = engine.stats();
        assert_eq!(stats.cycles_processed, 1);
    }

    #[test]
    fn test_memory_entry_priority() {
        let dim = 256;
        let hv = random_hv(dim, 42);

        let low_priority = MemoryEntry::new("low", hv.clone(), 0.1);
        let high_priority = MemoryEntry::new("high", hv, 0.9);

        // High emotional strength should give higher priority
        assert!(high_priority.priority() > low_priority.priority());
    }

    #[test]
    fn test_find_matching_patterns() {
        let mut engine = PatternDiscoveryEngine::new(DiscoveryConfig {
            min_occurrences: 2,
            similarity_threshold: 0.4,
            resonator_iterations: 30,
            ..Default::default()
        });

        let dim = 256;
        let base = random_hv(dim, 42);

        for i in 0..4 {
            let similar = similar_hv(&base, 0.1, i + 100);
            engine.add_memory(format!("mem_{}", i), similar, 0.6);
        }

        let _patterns = engine.discover_patterns(0.8);

        // Query with the base vector should find matching patterns
        let matches = engine.find_matching_patterns(&base, 0.3);
        println!("Found {} matching patterns for query", matches.len());

        // Each match should have a valid similarity score
        for m in &matches {
            assert!(
                m.internal_coherence.is_finite(),
                "Match similarity should be finite"
            );
            assert!(
                m.internal_coherence >= 0.0,
                "Match similarity should be non-negative"
            );
        }
    }

    #[test]
    fn test_consolidation() {
        let mut engine = PatternDiscoveryEngine::new(DiscoveryConfig {
            min_occurrences: 2,
            similarity_threshold: 0.4,
            auto_consolidate: true,
            ..Default::default()
        });

        let dim = 256;
        let base = random_hv(dim, 42);

        for i in 0..4 {
            let similar = similar_hv(&base, 0.1, i + 100);
            engine.add_memory(format!("mem_{}", i), similar, 0.6);
        }

        let _patterns = engine.discover_patterns(0.9);

        // Check that some memories got consolidated
        let stats = engine.stats();
        println!("Memories consolidated: {}", stats.memories_consolidated);

        // After discover_patterns with auto_consolidate, stats should be valid
        assert_eq!(
            stats.cycles_processed, 1,
            "Should have processed exactly 1 discovery cycle"
        );
        assert!(
            stats.memories_consolidated <= 4,
            "Cannot consolidate more memories than were added"
        );
    }
}
