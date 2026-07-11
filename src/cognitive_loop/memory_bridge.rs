// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Episodic memory bridge for the cognitive loop.
//!
//! Provides memory encoding, recall, consolidation, and decay for cognitive cycles.
//! Can be connected to the full HippocampusActor for persistence.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

/// Episodic memory trace for the cognitive loop
///
/// Lightweight representation of a memory that can be queried during cycles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodicMemory {
    /// Memory ID
    pub id: u64,
    /// Timestamp when encoded (cycle count)
    pub encoded_at_cycle: usize,
    /// Content summary
    pub content: String,
    /// Embedding (compressed for efficiency)
    pub embedding: Vec<f32>,
    /// Emotional valence (-1.0 to 1.0)
    pub valence: f32,
    /// Φ at encoding time
    pub phi_at_encoding: f32,
    /// Access count
    pub access_count: u32,
    /// Strength (0.0 to 1.0, decays over time)
    pub strength: f32,
    /// Full HDC vector compressed via PolarQuant (when turbo-quant feature is enabled).
    /// Stores the complete 16,384D vector at ~8x compression for high-fidelity recall.
    #[cfg(feature = "turbo-quant")]
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub compressed_full_hv: Option<crate::hdc::hv_compression::CompressedHv>,
}

impl EpisodicMemory {
    /// Compute similarity with query embedding
    pub fn similarity(&self, query: &[f32]) -> f32 {
        if self.embedding.len() != query.len() {
            return 0.0;
        }
        let dot: f32 = self
            .embedding
            .iter()
            .zip(query.iter())
            .map(|(a, b)| a * b)
            .sum();
        let mag_self: f32 = self.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_query: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if mag_self > 0.0 && mag_query > 0.0 {
            dot / (mag_self * mag_query)
        } else {
            0.0
        }
    }
}

/// Episodic Memory Bridge for the cognitive loop
///
/// Provides memory encoding and recall during cognitive cycles.
/// Can be connected to the full HippocampusActor for persistence.
#[derive(Debug, Clone)]
pub struct EpisodicMemoryBridge {
    /// Short-term memory buffer (recent cycles)
    short_term: VecDeque<EpisodicMemory>,
    /// Long-term memory store
    long_term: Vec<EpisodicMemory>,
    /// Maximum short-term memories
    max_short_term: usize,
    /// Maximum long-term memories
    max_long_term: usize,
    /// Next memory ID
    next_id: u64,
    /// Consolidation threshold (strength needed to move to long-term)
    consolidation_threshold: f32,
    /// Statistics
    pub stats: MemoryBridgeStats,
}

/// Statistics for the memory bridge
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryBridgeStats {
    pub total_encoded: u64,
    pub total_recalled: u64,
    pub consolidations: u64,
    pub avg_recall_similarity: f32,
}

impl Default for EpisodicMemoryBridge {
    fn default() -> Self {
        Self {
            short_term: VecDeque::with_capacity(100),
            long_term: Vec::with_capacity(1000),
            max_short_term: 100,
            max_long_term: 1000,
            next_id: 0,
            consolidation_threshold: 0.5,
            stats: MemoryBridgeStats::default(),
        }
    }
}

impl EpisodicMemoryBridge {
    /// Encode a new memory
    pub fn encode(
        &mut self,
        content: impl Into<String>,
        embedding: Vec<f32>,
        valence: f32,
        phi: f32,
        cycle: usize,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;

        let memory = EpisodicMemory {
            id,
            encoded_at_cycle: cycle,
            content: content.into(),
            embedding,
            valence,
            phi_at_encoding: phi,
            access_count: 0,
            strength: 1.0,
            #[cfg(feature = "turbo-quant")]
            compressed_full_hv: None,
        };

        // Add to short-term
        if self.short_term.len() >= self.max_short_term {
            // Consolidate oldest to long-term if strong enough
            if let Some(oldest) = self.short_term.front() {
                if oldest.strength >= self.consolidation_threshold {
                    self.long_term.push(oldest.clone());
                    self.stats.consolidations += 1;
                    // Trim long-term if needed
                    if self.long_term.len() > self.max_long_term {
                        // Remove weakest memory
                        if let Some(min_idx) = self
                            .long_term
                            .iter()
                            .enumerate()
                            .min_by(|a, b| {
                                a.1.strength
                                    .partial_cmp(&b.1.strength)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                            .map(|(i, _)| i)
                        {
                            self.long_term.remove(min_idx);
                        }
                    }
                }
            }
            self.short_term.pop_front();
        }
        self.short_term.push_back(memory);
        self.stats.total_encoded += 1;

        id
    }

    /// Recall memories similar to query embedding
    pub fn recall(
        &mut self,
        query: &[f32],
        top_k: usize,
        min_similarity: f32,
    ) -> Vec<(EpisodicMemory, f32)> {
        let mut results: Vec<(EpisodicMemory, f32)> = Vec::new();

        // Search both short-term and long-term
        for memory in self.short_term.iter().chain(self.long_term.iter()) {
            let sim = memory.similarity(query);
            if sim >= min_similarity {
                results.push((memory.clone(), sim));
            }
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        // Update access counts for recalled memories
        for (recalled, _) in &results {
            // Update in short-term
            if let Some(mem) = self.short_term.iter_mut().find(|m| m.id == recalled.id) {
                mem.access_count += 1;
                mem.strength = (mem.strength + 0.1).min(1.0);
            }
            // Update in long-term
            if let Some(mem) = self.long_term.iter_mut().find(|m| m.id == recalled.id) {
                mem.access_count += 1;
                mem.strength = (mem.strength + 0.05).min(1.0);
            }
        }

        if !results.is_empty() {
            self.stats.total_recalled += 1;
            self.stats.avg_recall_similarity =
                results.iter().map(|(_, s)| s).sum::<f32>() / results.len() as f32;
        }

        results
    }

    /// Decay unused memories
    pub fn decay(&mut self, decay_rate: f32) {
        for memory in self.short_term.iter_mut().chain(self.long_term.iter_mut()) {
            memory.strength = (memory.strength - decay_rate).max(0.0);
        }
        // Remove memories with zero strength from long-term
        self.long_term.retain(|m| m.strength > 0.01);
    }

    /// Get memory count
    pub fn memory_count(&self) -> (usize, usize) {
        (self.short_term.len(), self.long_term.len())
    }

    /// Reset the memory bridge
    pub fn reset(&mut self) {
        self.short_term.clear();
        self.long_term.clear();
        self.next_id = 0;
        self.stats = MemoryBridgeStats::default();
    }

    /// Consolidate recent memories to long-term storage
    ///
    /// Triggered by motor commands to strengthen recent experiences.
    /// This forces consolidation of high-strength short-term memories.
    pub fn consolidate_recent(&mut self, level: f64) {
        if level < 0.3 {
            // C < 0.3: no consolidation (matches clinical anesthesia)
            return;
        }

        // Find strong short-term memories and move to long-term
        let mut strong_memories: Vec<EpisodicMemory> = self
            .short_term
            .iter()
            .filter(|m| m.strength >= self.consolidation_threshold * 0.8) // Slightly lower threshold
            .cloned()
            .collect();

        // Apply graded depth based on consciousness level C
        for memory in &mut strong_memories {
            if level > 0.7 {
                // C > 0.7: deep encoding (episodic + semantic + emotional tags)
                let mut tags = Vec::new();
                if !memory.content.contains("[EPISODIC]") {
                    tags.push("[EPISODIC]");
                }
                if !memory.content.contains("[SEMANTIC]") {
                    tags.push("[SEMANTIC]");
                }
                if !memory.content.contains("[EMOTIONAL]") {
                    tags.push("[EMOTIONAL]");
                }
                if !tags.is_empty() {
                    memory.content = format!("{} {}", memory.content, tags.join(" "));
                }
                memory.strength = 1.0;
            } else {
                // C 0.3-0.7: standard episodic encoding
                if !memory.content.contains("[EPISODIC]") {
                    memory.content = format!("{} [EPISODIC]", memory.content);
                }
                memory.strength = (memory.strength + 0.1).min(1.0);
            }
        }

        for memory in strong_memories {
            // Check if not already in long-term
            if !self.long_term.iter().any(|m| m.id == memory.id) {
                self.long_term.push(memory);
                self.stats.consolidations += 1;
            }
        }

        // Trim long-term if needed
        while self.long_term.len() > self.max_long_term {
            if let Some(min_idx) = self
                .long_term
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.strength
                        .partial_cmp(&b.1.strength)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.long_term.remove(min_idx);
            } else {
                break;
            }
        }
    }

    /// Attach a compressed full HDC vector to an episodic memory.
    ///
    /// When the `turbo-quant` feature is enabled, this stores the complete 16,384D
    /// ContinuousHV in compressed form (~8x smaller) alongside the 64-element sample.
    #[cfg(feature = "turbo-quant")]
    pub fn attach_compressed_hv(
        &mut self,
        memory_id: u64,
        compressed: crate::hdc::hv_compression::CompressedHv,
    ) {
        if let Some(mem) = self.short_term.iter_mut().rev().find(|m| m.id == memory_id) {
            mem.compressed_full_hv = Some(compressed);
            return;
        }
        if let Some(mem) = self.long_term.iter_mut().find(|m| m.id == memory_id) {
            mem.compressed_full_hv = Some(compressed);
        }
    }
}
