//! Memory subsystems for Symthaea.
//!
//! Phase A extraction: semantic memory (HDC-based similarity lookup) and
//! conversation coherence tracking.

pub mod coherence;
pub mod semantic;

pub use coherence::{CoherenceTrajectoryStats, ConversationCoherenceTracker};
pub use semantic::{SemanticEntry, SemanticMemory, SemanticMemoryStats};
