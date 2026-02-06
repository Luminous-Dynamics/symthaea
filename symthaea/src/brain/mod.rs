//! Brain module - neural architecture components
//!
//! This module provides brain-inspired components including:
//! - Prefrontal cortex (executive functions, working memory)
//! - Actor model (concurrent neural processing)
//! - Social coherence (theory of mind, cooperation)
//! - Various bridge modules for integration

// Core brain modules (self-contained, verified working)
pub mod actor_model;
pub mod prefrontal;
pub mod social_coherence;

// TODO: Bridge modules disabled - they reference non-existent fields/methods:
//   - pending_concepts on ConsciousnessWorldModel (10 errors)
//   - uid on CrystalizedConcept (5 errors)
//   - missing methods on CoreAffect (decay, intensity, to_emotion_category)
//   - missing methods on HippocampusActor (remember)
//   - RecallQuery fields (query, threshold, emotion_filter, context_tags) don't exist
//   - String treated as Option (is_some/is_none/as_deref)
//   - missing imports from language::emotional_core and brain::prefrontal
// These modules were written against an earlier API version and need refactoring.
// #[cfg(feature = "full_consciousness")]
// pub mod affective_bridge;
// #[cfg(feature = "full_consciousness")]
// pub mod consciousness_bridge;
// #[cfg(feature = "full_consciousness")]
// pub mod hippocampus_bridge;

// Dark spot actor needs mycelix DHT
#[cfg(feature = "mycelix_module")]
pub mod dark_spot_actor;

// Re-export key types
pub use prefrontal::{PrefrontalCortex, PrefrontalConfig, WorkingMemoryItem, PlannedAction, ExecutiveDecision};
pub use actor_model::{ActorSystem, Actor, ActorId, ActorRole, ActorMessage, MessageType};
pub use social_coherence::{SocialCoherence, SocialCoherenceConfig, MentalModel, Relationship, RelationshipType};
