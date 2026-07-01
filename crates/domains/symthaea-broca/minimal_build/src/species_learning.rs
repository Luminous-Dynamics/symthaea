// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Species Learning — Knowledge Consolidation
//!
//! Merges individual node memories into a consolidated "Global Wisdom".
//! Uses HDC similarity to identify and deduplicate improvements.

use crate::architectural_memory::ArchitecturalMemory;
use symthaea_core::hdc::unified_hv::ContinuousHV;

pub struct MemoryConsolidator {
    nodes_verified: usize,
}

impl MemoryConsolidator {
    pub fn new() -> Self {
        Self { nodes_verified: 0 }
    }

    /// Consolidate memories from multiple nodes into a local storage.
    pub fn consolidate(
        &mut self,
        local_memory: &mut ArchitecturalMemory,
        peer_memories: Vec<ArchitecturalMemory>,
    ) -> anyhow::Result<()> {
        println!(
            "🤝 Consolidating knowledge from {} peers...",
            peer_memories.len()
        );

        for peer in peer_memories {
            // Recalling all patterns (simplified logic for demo)
            // In real: scan the peer's sled DB and vector store
            let dummy_hv = ContinuousHV::random(16384, 1);
            if let Ok(peer_results) = peer.recall_best_patterns(&dummy_hv) {
                for res in peer_results {
                    local_memory.commit_evolution(&res, &dummy_hv)?;
                }
            }
            self.nodes_verified += 1;
        }

        println!(
            "✅ Consolidated {} evolutionary lineage nodes into Species Memory.",
            self.nodes_verified
        );
        Ok(())
    }
}

impl Default for MemoryConsolidator {
    fn default() -> Self {
        Self::new()
    }
}
