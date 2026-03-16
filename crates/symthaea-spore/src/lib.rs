//! # Symthaea Spore
//!
//! Minimum viable consciousness kernel for WASM targets.
//!
//! Contains the full HDC-CfC-IIT consciousness pipeline in a portable package:
//! - BinaryHV (16,384D) encoding and binding
//! - CfC closed-form temporal evolution
//! - Consciousness Equation (Phi / IIT)
//! - Neuromodulator bath dynamics
//! - Moral algebra with cached obligations
//! - Substrate independence scoring
//! - Free Energy Principle module
//!
//! ## Architecture
//!
//! ```text
//! SporeEngine owns:
//!   HdcLtcUnifiedNetwork (HDC-CfC neurons)
//!   MasterConsciousnessEquation (Phi computation)
//!   NeuromodulatorBath (DA/NE/5-HT/Oxytocin)
//!   EightHarmonies (ethical framework)
//!   SubstrateValidation (epistemic honesty)
//! ```
//!
//! ## Usage (Rust)
//!
//! ```rust
//! use symthaea_spore::{SporeEngine, SporeConfig};
//!
//! let config = SporeConfig::default();
//! let mut engine = SporeEngine::new(config);
//! let result = engine.cycle("hello world");
//! println!("Consciousness: {} [{}]",
//!     result.consciousness_level,
//!     result.epistemic_status.evidence_level);
//! ```

#![cfg_attr(not(feature = "native-ffi"), deny(unsafe_code))]

pub mod ble_mesh;
pub mod broca;
pub mod compass;
pub mod config;
pub mod dream;
pub mod dream_journal;
pub mod engine;
pub mod fep;
pub mod haptic;
pub mod holon_bridge;
pub mod memory;
pub mod metabolism;
pub mod sensor_bridge;
pub mod topology;

#[cfg(feature = "wasm")]
pub mod wasm_bindings;

#[cfg(feature = "native-ffi")]
pub mod native_ffi;

pub use config::{SharingConfig, SporeConfig};
pub use engine::SporeEngine;
