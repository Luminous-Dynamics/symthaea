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

// Bridge modules (affective_bridge, consciousness_bridge, hippocampus_bridge)
// were removed — they targeted an earlier API version with incompatible types.

// Dark spot actor needs mycelix DHT
#[cfg(feature = "mycelix_module")]
pub mod dark_spot_actor;

// Re-export key types
pub use actor_model::{Actor, ActorId, ActorMessage, ActorRole, ActorSystem, MessageType};
pub use prefrontal::{
    ExecutiveDecision, PlannedAction, PrefrontalConfig, PrefrontalCortex, WorkingMemoryItem,
};
pub use social_coherence::{
    MentalModel, Relationship, RelationshipType, SocialCoherence, SocialCoherenceConfig,
};
