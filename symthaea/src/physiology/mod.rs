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
pub use coherence::{CoherenceField, CoherenceConfig, CoherenceState};
pub use endocrine::HormoneState;
pub use social_coherence::{
    SocialCoherenceField, CoherenceLendingProtocol, CollectiveLearning,
    CoherenceBeacon, CoherenceLoan,
};
