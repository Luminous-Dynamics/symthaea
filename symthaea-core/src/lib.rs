//! Symthaea Core
//!
//! The mathematical and structural foundation for the Holographic Liquid Brain.
//! Contains primitives for Hyperdimensional Computing (HDC) and Integrated Information (Phi).

pub mod core;
pub mod genesis;
pub mod hdc;
pub mod observability;
pub mod phi_engine;
pub mod physics;

// Re-exports are handled at the module level
// Key types are available from their respective modules:
// - hdc::binary_hv::HV16
// - hdc::consciousness::ConsciousnessState
// - hdc::integrated_information::IntegratedInformation
// - core::ConsciousnessState (different variant)
