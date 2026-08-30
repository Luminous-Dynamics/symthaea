#![deny(unsafe_code)]

//! Native interoceptive regulation primitives for Symthaea.
//!
//! This crate models artificial viability state directly. It intentionally
//! contains no semantic category layer and no dependency on the cognitive loop,
//! allowing deterministic regulation experiments to remain mechanically isolated.

mod allostasis;
mod dynamics;
mod homeostasis;
mod snapshot;
mod state;

pub use allostasis::{assess_allostasis, AllostaticConfig, AllostaticReport};
pub use dynamics::{
    InteroceptiveDrive, InteroceptiveDynamicsConfig, NativeInteroceptiveModel,
};
pub use homeostasis::{assess_homeostasis, HomeostaticReport};
pub use snapshot::InteroceptiveSnapshot;
pub use state::{
    NativeInteroceptiveState, ViabilityChannel, ViabilityVariable, CHANNEL_COUNT,
};
