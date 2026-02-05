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

// Bridge modules need recursive_improvement which needs internal API alignment
#[cfg(feature = "full_consciousness")]
pub mod affective_bridge;
#[cfg(feature = "full_consciousness")]
pub mod consciousness_bridge;
#[cfg(feature = "full_consciousness")]
pub mod hippocampus_bridge;

// Dark spot actor needs mycelix DHT
#[cfg(feature = "mycelix_module")]
pub mod dark_spot_actor;

// Re-export key types
pub use prefrontal::{PrefrontalCortex, PrefrontalConfig, WorkingMemoryItem, PlannedAction, ExecutiveDecision};
pub use actor_model::{ActorSystem, Actor, ActorId, ActorRole, ActorMessage, MessageType};
pub use social_coherence::{SocialCoherence, SocialCoherenceConfig, MentalModel, Relationship, RelationshipType};
