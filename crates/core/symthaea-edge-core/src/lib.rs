#![no_std]
#![forbid(unsafe_code)]
//! Bounded, allocation-free primitives for compact MCU-oriented Symthaea profiles.
//!
//! This is not the full Symthaea runtime or a complete CfC implementation. The liquid-state
//! kernel consumes an explicit Q15 coefficient; an optional bounded Q16.16 dt/tau lookup
//! supplies a scalar base-tau coefficient. Packed HDC memory is statically capacity-bounded.

#[cfg(test)]
extern crate std;

pub mod liquid_state;
pub mod packed_hdc;
pub mod scalar_encoder;
pub mod substrate;
pub mod time_constant;

pub use liquid_state::{FixedPointLiquidState, LiquidStateError, Q15_ONE};
pub use packed_hdc::{
    Classification, InsertOutcome, MemoryError, PackedHdcMemory, PackedHv,
};
pub use scalar_encoder::{ScalarEncoderError, ScalarLevelEncoder};
pub use substrate::{BoundedEdgeCore, EdgeCoreUpdateError};
pub use time_constant::{
    alpha_from_dt_tau_us, alpha_from_ratio_q16, TimeConstantError,
    ALPHA_MAX_ERROR_Q15_LSB, MAX_RATIO_Q16, RATIO_Q16_STEP,
};
