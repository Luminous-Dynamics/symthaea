//! mycelix-sdk stub
//!
//! Minimal stub providing the types that symthaea imports under the
//! `mycelix_sdk` feature flag. Only the modules and types actually
//! referenced by symthaea are included.

pub mod bridge;
pub mod epistemic;
pub mod hyperfeel;
pub mod matl;

// Re-exports matching the real crate
pub use bridge::{
    BridgeEvent, BridgeMessage, CrossHappReputation, HappReputationScore, LocalBridge,
};
pub use epistemic::{EmpiricalLevel, EpistemicClaim, MaterialityLevel, NormativeLevel};
pub use hyperfeel::{EncodingConfig, HyperFeelEncoder, HyperGradient};
pub use matl::ProofOfGradientQuality;
