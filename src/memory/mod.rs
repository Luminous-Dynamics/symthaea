//! Memory Module - Episodic & Procedural Memory Systems
//!
//! Week 2 Days 1-4: Hippocampus (Episodic) & Cerebellum (Procedural)
//!
//! Revolutionary Insight:
//! Memory is not storage - memory is RECONSTRUCTION.
//! The Hippocampus doesn't record events; it encodes them as holographic
//! hypervectors that can be recalled through similarity search.
//!
//! Future: Cerebellum (Procedural Skills), Memory Consolidation (Sleep)

pub mod hippocampus;

pub use hippocampus::{HippocampusActor, MemoryTrace, RecallQuery, EmotionalValence};
