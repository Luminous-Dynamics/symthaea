#![deny(unsafe_code)]

//! Native interoceptive regulation primitives for Symthaea.
//!
//! This crate models artificial viability state directly. It intentionally
//! contains no semantic category layer and no dependency on the cognitive loop,
//! allowing deterministic regulation experiments to remain mechanically isolated.

mod state;

pub use state::{
    NativeInteroceptiveState, ViabilityChannel, ViabilityVariable, CHANNEL_COUNT,
};
