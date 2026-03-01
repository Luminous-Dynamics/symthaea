//! Memory subsystems for Symthaea.
//!
//! Extracted modules:
//! - Phase A: semantic memory (HDC-based similarity lookup), coherence tracking
//! - Phase B: hippocampus (episodic memory), conversation memory (SQLite)

pub mod coherence;
pub mod conversation;
pub mod hippocampus;
pub mod semantic;

pub use coherence::{CoherenceTrajectoryStats, ConversationCoherenceTracker};
pub use conversation::{
    CausalLearning, ConversationMemory, ConversationMemoryStats, ConversationSummary,
    ConversationTurn,
};
pub use hippocampus::{
    EmotionalValence, HippocampusActor, HippocampusStats, MemoryTrace, RecallQuery, RecallResult,
};
pub use semantic::{SemanticEntry, SemanticMemory, SemanticMemoryStats};
