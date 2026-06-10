// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Lightweight memory system for the Spore WASM consciousness kernel.
//!
//! WASM-compatible: no std::time, no filesystem, no tokio.
//! Two memory subsystems orchestrated by `MemoryCoordinator`:
//! - **SemanticMemory**: HDC-based content-addressable ring buffer
//! - **EpisodicMemory**: Phi-prioritized episode buffer

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Inline cosine similarity (no external imports)
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = (norm_a * norm_b).sqrt();
    if denom < 1e-12 { 0.0 } else { dot / denom }
}

// ---------------------------------------------------------------------------
// SemanticMemory
// ---------------------------------------------------------------------------

/// A single entry in semantic memory: an HDC vector snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticEntry {
    /// HDC hypervector for content-addressable retrieval.
    pub hdc_vector: Vec<f32>,
    /// Cognitive cycle at which this entry was stored.
    pub cycle: u64,
    /// Prediction error at storage time (salience signal).
    pub prediction_error: f32,
}

/// Aggregate statistics for semantic memory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticStats {
    pub total_stores: u64,
    pub total_queries: u64,
}

/// HDC-based content-addressable ring buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticMemory {
    entries: Vec<Option<SemanticEntry>>,
    max_entries: usize,
    write_pos: usize,
    count: usize,
    stats: SemanticStats,
}

impl SemanticMemory {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            entries: (0..capacity).map(|_| None).collect(),
            max_entries: capacity,
            write_pos: 0,
            count: 0,
            stats: SemanticStats::default(),
        }
    }

    /// Store an HDC vector in the ring buffer, overwriting the oldest entry
    /// when full.
    pub fn store(&mut self, hdc: Vec<f32>, prediction_error: f32, cycle: u64) {
        self.entries[self.write_pos] = Some(SemanticEntry {
            hdc_vector: hdc,
            cycle,
            prediction_error,
        });
        self.write_pos = (self.write_pos + 1) % self.max_entries;
        if self.count < self.max_entries {
            self.count += 1;
        }
        self.stats.total_stores += 1;
    }

    /// Find the `top_k` most similar entries to `query` by cosine similarity.
    /// Returns `(index, similarity)` pairs sorted descending by similarity.
    pub fn find_similar(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| {
                slot.as_ref()
                    .map(|entry| (i, cosine_similarity(query, &entry.hdc_vector)))
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Retrieve an entry by index.
    pub fn get(&self, index: usize) -> Option<&SemanticEntry> {
        self.entries.get(index).and_then(|s| s.as_ref())
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn capacity(&self) -> usize {
        self.max_entries
    }

    pub fn stats(&self) -> &SemanticStats {
        &self.stats
    }

    /// Iterate over all stored entries with their slot indices (for checkpoint serialization).
    pub fn iter_entries(&self) -> impl Iterator<Item = (usize, &SemanticEntry)> {
        self.entries
            .iter()
            .enumerate()
            .filter_map(|(i, slot)| slot.as_ref().map(|e| (i, e)))
    }

    /// Restore entries from a checkpoint. Overwrites existing entries at the given slot indices.
    pub fn restore_entries(&mut self, entries: Vec<(usize, SemanticEntry)>) {
        for (idx, entry) in entries {
            if idx < self.entries.len() {
                self.entries[idx] = Some(entry);
                if idx >= self.count {
                    self.count = idx + 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Episode / EpisodicMemory
// ---------------------------------------------------------------------------

/// A single consciousness experience snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// Input HDC vector.
    pub input: Vec<f32>,
    /// Output HDC vector.
    pub output: Vec<f32>,
    /// Integrated information (Phi) at this cycle.
    pub phi: f32,
    /// Cognitive cycle number.
    pub cycle: u64,
    /// Prediction error (surprise signal).
    pub prediction_error: f32,
    /// Affective valence (−1..+1).
    pub valence: f32,
    /// Number of times this episode has been replayed.
    pub replay_count: u32,
    /// Consolidation strength (increases with replay).
    pub consolidation_strength: f32,
}

impl Episode {
    /// Phi-weighted priority with recency bonus.
    ///
    /// `priority = phi * consolidation * (1 + recency_bonus)`
    /// where `recency_bonus = 1 / (1 + age)`.
    pub fn priority_score(&self, current_cycle: u64) -> f32 {
        let age = current_cycle.saturating_sub(self.cycle) as f32;
        let recency_bonus = 1.0 / (1.0 + age);
        self.phi * self.consolidation_strength * (1.0 + recency_bonus)
    }
}

/// Phi-prioritized episode buffer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    episodes: Vec<Episode>,
    pub(crate) max_episodes: usize,
    phi_threshold: f32,
}

impl EpisodicMemory {
    pub fn new(capacity: usize, phi_threshold: f32) -> Self {
        Self {
            episodes: Vec::with_capacity(capacity.min(1024)),
            max_episodes: capacity.max(1),
            phi_threshold,
        }
    }

    /// Store an episode only if its Phi exceeds the threshold.
    /// If the buffer is full, evict the lowest-priority episode.
    /// Returns `true` if the episode was stored.
    pub fn store_if_significant(&mut self, episode: Episode) -> bool {
        if episode.phi < self.phi_threshold {
            return false;
        }

        if self.episodes.len() >= self.max_episodes {
            // Find lowest-priority episode and evict it.
            let current_cycle = episode.cycle;
            if let Some((worst_idx, _)) =
                self.episodes.iter().enumerate().min_by(|(_, a), (_, b)| {
                    a.priority_score(current_cycle)
                        .partial_cmp(&b.priority_score(current_cycle))
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
            {
                self.episodes.swap_remove(worst_idx);
            }
        }

        self.episodes.push(episode);
        true
    }

    /// Sample up to `batch_size` episodes weighted by Phi-priority.
    ///
    /// Uses a simple deterministic top-priority selection (no RNG needed in
    /// WASM). For stochastic replay, shuffle externally.
    pub fn sample_batch(&self, batch_size: usize, current_cycle: u64) -> Vec<&Episode> {
        let mut indices: Vec<(usize, f32)> = self
            .episodes
            .iter()
            .enumerate()
            .map(|(i, ep)| (i, ep.priority_score(current_cycle)))
            .collect();
        indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        indices
            .iter()
            .take(batch_size)
            .map(|(i, _)| &self.episodes[*i])
            .collect()
    }

    /// Get the top `n` episodes by priority score.
    pub fn get_top(&self, n: usize, current_cycle: u64) -> Vec<&Episode> {
        self.sample_batch(n, current_cycle)
    }

    pub fn len(&self) -> usize {
        self.episodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.episodes.is_empty()
    }

    pub fn clear(&mut self) {
        self.episodes.clear();
    }

    /// Iterate over all stored episodes (for checkpoint serialization).
    pub fn iter_episodes(&self) -> impl Iterator<Item = &Episode> {
        self.episodes.iter()
    }

    /// Restore episodes from a checkpoint, replacing all existing episodes.
    pub fn restore_episodes(&mut self, episodes: Vec<Episode>) {
        self.episodes = episodes;
        if self.episodes.len() > self.max_episodes {
            self.episodes.truncate(self.max_episodes);
        }
    }
}

// ---------------------------------------------------------------------------
// MemoryStats (for WASM export)
// ---------------------------------------------------------------------------

/// Aggregate statistics for the full memory system, suitable for WASM export.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    pub semantic_count: usize,
    pub semantic_capacity: usize,
    pub episodic_count: usize,
    pub episodic_capacity: usize,
    pub current_phi: f32,
    pub current_coherence: f32,
    pub total_cycles: u64,
}

// ---------------------------------------------------------------------------
// MemoryCoordinator
// ---------------------------------------------------------------------------

/// Orchestrates semantic and episodic memory subsystems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCoordinator {
    pub semantic: SemanticMemory,
    pub episodic: EpisodicMemory,
    pub current_phi: f32,
    pub current_coherence: f32,
    pub step: u64,
}

impl MemoryCoordinator {
    pub fn new(semantic_capacity: usize, episodic_capacity: usize) -> Self {
        Self {
            semantic: SemanticMemory::new(semantic_capacity),
            episodic: EpisodicMemory::new(episodic_capacity, 0.3),
            current_phi: 0.0,
            current_coherence: 0.0,
            step: 0,
        }
    }

    /// Update tracked consciousness signals.
    pub fn update_signals(&mut self, phi: f32, coherence: f32) {
        self.current_phi = phi;
        self.current_coherence = coherence;
    }

    /// Record one cognitive cycle.
    ///
    /// The input HDC vector is always stored in semantic memory.
    /// An episode is stored in episodic memory only if Phi exceeds the
    /// threshold.
    pub fn record_cycle(
        &mut self,
        input: &[f32],
        output: &[f32],
        prediction_error: f32,
        phi: f32,
        cycle: u64,
    ) {
        // Always store in semantic memory.
        self.semantic.store(input.to_vec(), prediction_error, cycle);

        // Conditionally store in episodic memory.
        let episode = Episode {
            input: input.to_vec(),
            output: output.to_vec(),
            phi,
            cycle,
            prediction_error,
            valence: 0.0,
            replay_count: 0,
            consolidation_strength: 1.0,
        };
        self.episodic.store_if_significant(episode);

        self.step = cycle;
        self.current_phi = phi;
    }

    /// Query semantic memory for the most similar entries.
    pub fn query_semantic(&self, query: &[f32], top_k: usize) -> Vec<(usize, f32)> {
        self.semantic.find_similar(query, top_k)
    }

    pub fn episodic_count(&self) -> usize {
        self.episodic.len()
    }

    pub fn semantic_count(&self) -> usize {
        self.semantic.len()
    }

    /// Produce aggregate stats for WASM export.
    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            semantic_count: self.semantic.len(),
            semantic_capacity: self.semantic.capacity(),
            episodic_count: self.episodic.len(),
            episodic_capacity: self.episodic.max_episodes,
            current_phi: self.current_phi,
            current_coherence: self.current_coherence,
            total_cycles: self.step,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(seed: f32, dim: usize) -> Vec<f32> {
        (0..dim).map(|i| (seed + i as f32).sin()).collect()
    }

    #[test]
    fn test_semantic_store_and_retrieve() {
        let mut mem = SemanticMemory::new(8);
        assert!(mem.is_empty());

        let v = make_vec(1.0, 64);
        mem.store(v.clone(), 0.5, 1);

        assert_eq!(mem.len(), 1);
        let entry = mem.get(0).unwrap();
        assert_eq!(entry.cycle, 1);
        assert!((entry.prediction_error - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ring_buffer_eviction() {
        let mut mem = SemanticMemory::new(3);
        for i in 0..5u64 {
            mem.store(make_vec(i as f32, 16), 0.1, i);
        }
        // Capacity is 3, so only 3 entries remain.
        assert_eq!(mem.len(), 3);
        assert_eq!(mem.capacity(), 3);

        // Oldest entries (cycles 0, 1) were overwritten.
        // write_pos wrapped: entries at [0]=cycle3, [1]=cycle4, [2]=cycle2
        let entry2 = mem.get(2).unwrap();
        assert_eq!(entry2.cycle, 2);
        let entry0 = mem.get(0).unwrap();
        assert_eq!(entry0.cycle, 3);
    }

    #[test]
    fn test_cosine_similarity_search() {
        let mut mem = SemanticMemory::new(16);
        let target = make_vec(42.0, 32);
        let distant = make_vec(999.0, 32);

        mem.store(distant, 0.1, 0);
        mem.store(target.clone(), 0.2, 1);
        mem.store(make_vec(43.0, 32), 0.3, 2); // close to target

        let results = mem.find_similar(&target, 2);
        assert_eq!(results.len(), 2);
        // The exact match (index 1) should be first with similarity ~1.0.
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 > 0.99);
    }

    #[test]
    fn test_phi_threshold_filtering() {
        let mut episodic = EpisodicMemory::new(10, 0.5);

        // Low-Phi episode should be rejected.
        let low = Episode {
            input: vec![1.0],
            output: vec![2.0],
            phi: 0.2,
            cycle: 1,
            prediction_error: 0.1,
            valence: 0.0,
            replay_count: 0,
            consolidation_strength: 1.0,
        };
        assert!(!episodic.store_if_significant(low));
        assert!(episodic.is_empty());

        // High-Phi episode should be accepted.
        let high = Episode {
            input: vec![1.0],
            output: vec![2.0],
            phi: 0.8,
            cycle: 2,
            prediction_error: 0.3,
            valence: 0.5,
            replay_count: 0,
            consolidation_strength: 1.0,
        };
        assert!(episodic.store_if_significant(high));
        assert_eq!(episodic.len(), 1);
    }

    #[test]
    fn test_episodic_eviction_by_priority() {
        let mut episodic = EpisodicMemory::new(2, 0.1);

        let ep1 = Episode {
            input: vec![],
            output: vec![],
            phi: 0.5,
            cycle: 1,
            prediction_error: 0.1,
            valence: 0.0,
            replay_count: 0,
            consolidation_strength: 1.0,
        };
        let ep2 = Episode {
            input: vec![],
            output: vec![],
            phi: 0.9,
            cycle: 2,
            prediction_error: 0.1,
            valence: 0.0,
            replay_count: 0,
            consolidation_strength: 1.0,
        };
        let ep3 = Episode {
            input: vec![],
            output: vec![],
            phi: 0.7,
            cycle: 10,
            prediction_error: 0.1,
            valence: 0.0,
            replay_count: 0,
            consolidation_strength: 1.0,
        };

        episodic.store_if_significant(ep1);
        episodic.store_if_significant(ep2);
        assert_eq!(episodic.len(), 2);

        // ep3 arrives at cycle 10 — ep1 (lower phi, older) should be evicted.
        episodic.store_if_significant(ep3);
        assert_eq!(episodic.len(), 2);

        let top = episodic.get_top(2, 10);
        // Both remaining episodes should have phi >= 0.7.
        for ep in &top {
            assert!(ep.phi >= 0.7);
        }
    }

    #[test]
    fn test_coordinator_routes_correctly() {
        let mut coord = MemoryCoordinator::new(16, 8);

        // Low-phi cycle: semantic only.
        coord.record_cycle(&[1.0, 2.0], &[3.0, 4.0], 0.1, 0.1, 1);
        assert_eq!(coord.semantic_count(), 1);
        assert_eq!(coord.episodic_count(), 0);

        // High-phi cycle: both.
        coord.record_cycle(&[5.0, 6.0], &[7.0, 8.0], 0.2, 0.8, 2);
        assert_eq!(coord.semantic_count(), 2);
        assert_eq!(coord.episodic_count(), 1);

        // Query semantic: the high-phi input should match itself.
        let results = coord.query_semantic(&[5.0, 6.0], 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].1 > 0.99);

        let stats = coord.stats();
        assert_eq!(stats.total_cycles, 2);
    }
}
