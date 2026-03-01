//! # Memory Module: Consciousness Persistence
//!
//! This module provides memory systems for consciousness persistence,
//! including episodic memory (hippocampus), conversation memory,
//! semantic memory (HDC-based similarity lookup), episodic replay for
//! high-Phi moment consolidation, and memory consolidation systems.

/// Re-exported from `symthaea-memory` crate (Phase A extraction).
pub mod coherence_tracker {
    pub use symthaea_memory::coherence::*;
}
/// Re-exported from `symthaea-memory` crate (Phase B extraction).
pub mod conversation_memory {
    pub use symthaea_memory::conversation::*;
}
/// Re-exported from `symthaea-memory` crate (Phase C extraction).
pub mod episodic_replay {
    pub use symthaea_memory::episodic_replay::*;
}
/// Re-exported from `symthaea-memory` crate (Phase B extraction).
pub mod hippocampus {
    pub use symthaea_memory::hippocampus::*;
}
/// Re-exported from `symthaea-memory` crate (Phase C extraction).
pub mod memory_coordinator {
    pub use symthaea_memory::coordinator::*;
}
/// Re-exported from `symthaea-memory` crate (Phase A extraction).
pub mod semantic_memory {
    pub use symthaea_memory::semantic::*;
}

// Re-export key types
pub use episodic_replay::{
    Episode, EpisodicMemory, EpisodicMemoryStats, EpisodicReplayConfig, ReplaySessionResult,
};
pub use hippocampus::{
    EmotionalValence, HippocampusActor, HippocampusStats, MemoryTrace, RecallQuery, RecallResult,
};
pub use memory_coordinator::{
    content_hash, CoordinatorConfig, CoordinatorStats, GraduationEvent, MemoryCoordinator,
    MemorySignals, MemorySource,
};
pub use semantic_memory::{SemanticEntry, SemanticMemory, SemanticMemoryStats};
