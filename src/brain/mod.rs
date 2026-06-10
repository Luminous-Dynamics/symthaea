// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

pub mod affective_bridge;

/// Dark Spot DHT Actor — bridges Mycelix GIS Dark Spot DHT with the brain's actor model.
/// Gated behind `mycelix` because it requires FL/hashing dependencies at runtime.
#[cfg(feature = "mycelix")]
pub mod dark_spot_actor;

// Re-export key types
pub use actor_model::{Actor, ActorId, ActorMessage, ActorRole, ActorSystem, MessageType};
pub use affective_bridge::{AffectiveBridge, AffectiveBridgeConfig};
pub use prefrontal::{
    ExecutiveDecision, PlannedAction, PrefrontalConfig, PrefrontalCortex, WorkingMemoryItem,
};
pub use social_coherence::{
    MentalModel, Relationship, RelationshipType, SocialCoherence, SocialCoherenceConfig,
};
