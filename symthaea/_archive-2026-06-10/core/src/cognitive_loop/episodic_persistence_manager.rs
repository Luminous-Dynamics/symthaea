// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Episodic Persistence Manager — episodic replay + memory database.
//!
//! Groups episodic memory replay, persistent SQLite database,
//! flush guard, and reasoning context into a single data-holder manager.
//!
//! Science: Tulving (1972) — episodic memory; Stickgold (2005) — replay consolidation.

/// Consolidated episodic persistence subsystem.
///
/// Holds episodic replay, memory database, flush guard, and reasoning
/// context fields that were previously scattered across `CognitiveLoopService`.
pub struct EpisodicPersistenceManager {
    /// Episodic memory replay for high-Phi moment consolidation.
    /// When enabled via `config.memory_graduation`, stores high-consciousness episodes
    /// and periodically replays them to reinforce important patterns.
    pub(crate) replay: Option<crate::memory::episodic_replay::EpisodicMemory>,

    /// Persistent memory database for cross-session episode storage.
    /// Created when `config.memory_db_path` is `Some`. Episodes are periodically
    /// flushed (every 199 cycles) via a background thread.
    pub(crate) db: Option<std::sync::Arc<crate::databases::SqliteMemory>>,

    /// Guard to prevent overlapping memory flushes. When true, a flush is in progress.
    pub(crate) flush_in_progress: std::sync::Arc<std::sync::atomic::AtomicBool>,

    /// Last assembled reasoning context from the knowledge engine.
    /// Contains grounded facts, causal chains, and epistemic state.
    /// Populated after knowledge engine processes each cycle's input.
    pub(crate) last_reasoning_context: Option<crate::knowledge::ReasoningContext>,
}

impl EpisodicPersistenceManager {
    /// Create a new EpisodicPersistenceManager.
    ///
    /// `replay` is the optional episodic memory (built from config).
    pub fn new(replay: Option<crate::memory::episodic_replay::EpisodicMemory>) -> Self {
        Self {
            replay,
            db: None,
            flush_in_progress: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            last_reasoning_context: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_episodic_persistence_manager_new() {
        let mgr = EpisodicPersistenceManager::new(None);
        assert!(mgr.replay.is_none());
        assert!(mgr.db.is_none());
        assert!(
            !mgr.flush_in_progress
                .load(std::sync::atomic::Ordering::Relaxed)
        );
        assert!(mgr.last_reasoning_context.is_none());
    }
}
