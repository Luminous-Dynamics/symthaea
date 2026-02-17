//! VSV-STARK Core Library
//!
//! Fixed-point arithmetic and CanaryCNN implementation for
//! Verifiable Self-Validation using ZK-STARKs.
//!
//! This library is designed to work in both standard Rust and
//! RISC Zero zkVM environments.

pub mod calibration;
pub mod canary_cnn;
pub mod fixedpoint;
pub mod loss;
pub mod weights;

// Re-export primary types
pub use canary_cnn::CanaryCNN;
pub use fixedpoint::Fixed;
pub use loss::{loss_delta_mse, mse_batch, mse_single};
