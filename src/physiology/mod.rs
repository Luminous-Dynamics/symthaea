// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Physiology module - biological/hormonal systems
//!
//! Models the physiological aspects of consciousness:
//! - Coherence field (synchronization/integration)
//! - Endocrine system (hormonal influences)
//! - Social coherence (collective consciousness dynamics)

pub mod coherence;
pub mod endocrine;
pub mod social_coherence;

// Re-exports for convenience
pub use coherence::{CoherenceConfig, CoherenceField, CoherenceState};
pub use endocrine::HormoneState;
pub use social_coherence::{
    CoherenceBeacon, CoherenceLendingProtocol, CoherenceLoan, CollectiveLearning,
    SocialCoherenceField,
};
