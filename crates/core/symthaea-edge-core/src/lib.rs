#![no_std]
#![forbid(unsafe_code)]
//! Bounded, allocation-free primitives for compact MCU-oriented Symthaea profiles.
//!
//! This is not the full Symthaea runtime and does not claim to implement the complete CfC
//! time-constant policy. The liquid-state kernel consumes an externally computed Q15
//! interpolation coefficient; packed HDC memory is statically capacity-bounded.

#[cfg(test)]
extern crate std;

pub mod liquid_state;
pub mod packed_hdc;
pub mod substrate;

pub use liquid_state::{FixedPointLiquidState, LiquidStateError, Q15_ONE};
pub use packed_hdc::{
    Classification, InsertOutcome, MemoryError, PackedHdcMemory, PackedHv,
};
pub use substrate::BoundedEdgeCore;
