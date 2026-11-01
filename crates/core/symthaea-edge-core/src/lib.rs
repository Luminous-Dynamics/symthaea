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
pub mod time_constant;

pub use liquid_state::{FixedPointLiquidState, LiquidStateError, Q15_ONE};
pub use packed_hdc::{
    Classification, InsertOutcome, MemoryError, PackedHdcMemory, PackedHv,
};
pub use substrate::{BoundedEdgeCore, EdgeCoreUpdateError};
pub use time_constant::{
    alpha_from_dt_tau_us, alpha_from_ratio_q16, TimeConstantError,
    ALPHA_MAX_ERROR_Q15_LSB, MAX_RATIO_Q16, RATIO_Q16_STEP,
};
