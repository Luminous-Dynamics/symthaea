//! # Memory Module: Consciousness Persistence
//!
//! This module provides memory systems for consciousness persistence,
//! including episodic memory (hippocampus), conversation memory,
//! semantic memory (HDC-based similarity lookup), episodic replay for
//! high-Phi moment consolidation, and memory consolidation systems.

pub mod coherence_tracker;
pub mod conversation_memory;
pub mod episodic_replay;
pub mod hippocampus;
pub mod memory_coordinator;
pub mod semantic_memory;

// Re-export key types
pub use episodic_replay::{
    Episode, EpisodicMemory, EpisodicMemoryStats, EpisodicReplayConfig, ReplaySessionResult,
};
pub use hippocampus::{
    EmotionalValence, HippocampusActor, HippocampusStats, MemoryTrace, RecallQuery, RecallResult,
};
pub use memory_coordinator::{
    content_hash, CoordinatorConfig, CoordinatorStats, GraduationEvent, MemoryCoordinator,
    MemorySignals,
};
pub use semantic_memory::{SemanticEntry, SemanticMemory, SemanticMemoryStats};
