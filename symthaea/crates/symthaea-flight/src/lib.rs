//! # symthaea-flight
//!
//! Unified HDC-LTC + FEP Active Inference quadrotor flight control.
//!
//! Uses the full 16,384D `HdcLtcUnifiedNetwork` from `symthaea-core` as the temporal
//! dynamics engine, with `symthaea-fep` providing precision-weighted Active Inference
//! modulation. Multi-rate architecture: 500Hz motor reflex + 25Hz cognitive tick.
//!
//! ## Architecture
//!
//! ```text
//! Sensors → QuadrotorHdcEncoder → ContinuousHV(16384D)
//!                                       ↓
//!                              HdcLtcUnifiedNetwork (evolve_closed_form)
//!                                       ↓
//!                              FlightController (output projection 16384→4)
//!                                       ↓
//!                              QuadrotorCommand [thrust, roll, pitch, yaw]
//!                                       ↓
//!                              MuJoCo Physics Step
//!
//! Every 20th motor step (25Hz):
//!   ActiveInferenceFlightAgent modulates τ, learning rate, prior precision
//! ```

#![allow(clippy::needless_range_loop)]

pub mod types;
pub mod encoder;
pub mod controller;
pub mod fep_agent;
pub mod training;

pub use types::*;
pub use encoder::QuadrotorHdcEncoder;
pub use controller::FlightController;
pub use fep_agent::ActiveInferenceFlightAgent;
pub use training::FlightTrainer;
