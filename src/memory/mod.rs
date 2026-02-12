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
pub use hippocampus::{
    HippocampusActor, MemoryTrace, RecallQuery, RecallResult,
    EmotionalValence, HippocampusStats,
};
pub use semantic_memory::{SemanticMemory, SemanticEntry, SemanticMemoryStats};
pub use episodic_replay::{
    EpisodicMemory, EpisodicReplayConfig, Episode,
    EpisodicMemoryStats, ReplaySessionResult,
};
pub use memory_coordinator::{
    MemoryCoordinator, CoordinatorConfig, MemorySignals,
    GraduationEvent, CoordinatorStats, content_hash,
};
