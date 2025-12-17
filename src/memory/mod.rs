//! Memory Module - Episodic & Procedural Memory Systems
//!
//! Week 2 Days 1-4: Hippocampus (Episodic) & Cerebellum (Procedural)
//! Week 17 Day 2: Episodic Memory Engine (Revolutionary Integration)
//!
//! Revolutionary Insight:
//! Memory is not storage - memory is RECONSTRUCTION.
//! The Hippocampus doesn't record events; it encodes them as holographic
//! hypervectors that can be recalled through similarity search.
//!
//! Week 17 adds CHRONO-SEMANTIC episodic memory:
//! - TemporalEncoder (circular time) + SemanticSpace (concepts) + SDM (pattern completion)
//! - Mental time travel: "What git errors happened yesterday morning?"
//! - Autobiographical timeline with temporal clustering
//!
//! Future: Cerebellum (Procedural Skills), Memory Consolidation (Sleep)

pub mod hippocampus;
pub mod episodic_engine;

pub use hippocampus::{HippocampusActor, MemoryTrace, RecallQuery, EmotionalValence};
pub use episodic_engine::{
    EpisodicMemoryEngine,
    EpisodicTrace,
    EpisodicConfig,
    EngineStats,
};
