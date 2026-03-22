// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Memory & Consolidation Manager — groups semantic memory, memory coordinator,
//! resonator memory, stability regime, and discovery service into a single CLS field.

use crate::consciousness::primitive_discovery::PrimitiveDiscoveryService;
use crate::consciousness::stability_regime::StabilityRegimeProcessor;
use crate::memory::memory_coordinator::MemoryCoordinator;
use crate::memory::semantic_memory::SemanticMemory;

/// Consolidated manager for memory and consolidation subsystems.
///
/// Replaces 5 top-level CLS fields:
/// - `semantic_memory` — HDC-based similarity lookup for contextual learning
/// - `memory_coordinator` — cross-tier signal broadcaster + graduation
/// - `resonator_memory` — factorized episodic recall via bound hypervectors
/// - `stability_regime` — CfC dynamics: crystallize frequent, keep rare fluid
/// - `discovery_service` — find new primitives seeded by crystallization
pub struct MemoryAndConsolidationManager {
    /// Semantic Memory: HDC-based similarity lookup for CfC contextual learning.
    pub semantic_memory: SemanticMemory,

    /// Memory Coordinator: cross-tier signal broadcaster + graduation.
    pub memory_coordinator: MemoryCoordinator,

    /// Resonator Memory for factorized episodic recall.
    pub resonator_memory: Option<crate::dynamics::resonator::ResonatorMemory>,

    /// Stability regime processor: CfC dynamics for primitives.
    pub stability_regime: StabilityRegimeProcessor,

    /// Discovery service for finding new primitives seeded by crystallization.
    pub discovery_service: PrimitiveDiscoveryService,
}

impl std::fmt::Debug for MemoryAndConsolidationManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryAndConsolidationManager")
            .field("resonator_memory", &self.resonator_memory.is_some())
            .field("total_stored", &self.semantic_memory.stats().total_stored)
            .finish_non_exhaustive()
    }
}

impl MemoryAndConsolidationManager {
    /// Reset memory state for a new episode.
    pub fn reset(&mut self) {
        // SemanticMemory, MemoryCoordinator, and StabilityRegime retain learned state.
        // ResonatorMemory retains codebook. DiscoveryService retains findings.
        // Nothing to reset — all memory is persistent across episodes.
    }
}
