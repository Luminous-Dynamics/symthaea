// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Memory Execution — extracted from CognitiveLoopService
//!
//! Groups the 4 memory, consolidation, and knowledge subsystems that were
//! previously individual fields on `CognitiveLoopService`:
//!
//! | Field | Type | Purpose |
//! |-------|------|---------|
//! | `memory_consol` | `MemoryAndConsolidationManager` | Semantic memory, coordinator, stability regime, discovery, resonator |
//! | `episodic_persistence` | `EpisodicPersistenceManager` | Episodic replay, DB persistence, flush guard, reasoning context |
//! | `causal_enhancer` | `Option<CausalLoopEnhancer>` | Causal structure discovery in (input, output) pairs |
//! | `knowledge_manager` | `Option<KnowledgeManager>` | Knowledge graph, causal DAG, adaptive ontology |
//!
//! ## Design
//!
//! - **Zero behavior change**: Pure structural refactor, same field paths via delegation.
//! - **Public API preserved**: All accessor methods on `CognitiveLoopService` still work.
//! - **Field access**: Internal code accesses via `self.memory.field_name`.

use crate::causal::CausalLoopEnhancer;

use super::episodic_persistence_manager;
use super::memory_consolidation_manager;

/// Groups 4 memory, consolidation, and knowledge subsystems.
///
/// Extracted from `CognitiveLoopService` to reduce its field count
/// and provide a cohesive memory execution boundary.
pub(crate) struct MemoryExecution {
    /// Memory & consolidation: semantic memory, coordinator, resonator, stability, discovery.
    pub memory_consol: memory_consolidation_manager::MemoryAndConsolidationManager,

    /// Episodic persistence: replay, database, flush guard, reasoning context.
    pub episodic_persistence: episodic_persistence_manager::EpisodicPersistenceManager,

    /// Causal loop enhancer for discovering causal structure in (input, output) pairs.
    /// When enabled via `config.causal_enhancement`, this:
    /// - Tracks recent (input, output) pairs
    /// - Periodically runs causal discovery
    /// - Weights attention based on discovered causal parents
    /// - Suggests interventions for exploration
    pub causal_enhancer: Option<CausalLoopEnhancer>,

    /// Knowledge engine: general-purpose reasoning infrastructure.
    /// Extracts structured facts from input, encodes as HDC vectors, stores in a
    /// temporal knowledge graph, builds causal DAG edges, and grows adaptive ontology.
    /// Science: Kanerva (2009) HDC, Pearl (2009) Causality, Carey (2009) conceptual change.
    pub knowledge_manager: Option<crate::knowledge::KnowledgeManager>,
}

impl MemoryExecution {
    /// Construct from individually built components.
    ///
    /// Called from `CognitiveLoopService::new()` after each subsystem is created.
    pub fn new(
        memory_consol: memory_consolidation_manager::MemoryAndConsolidationManager,
        episodic_persistence: episodic_persistence_manager::EpisodicPersistenceManager,
        causal_enhancer: Option<CausalLoopEnhancer>,
        knowledge_manager: Option<crate::knowledge::KnowledgeManager>,
    ) -> Self {
        Self {
            memory_consol,
            episodic_persistence,
            causal_enhancer,
            knowledge_manager,
        }
    }
}
