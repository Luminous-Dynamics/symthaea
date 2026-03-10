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

#![deny(unsafe_code)]

pub mod broca;
pub mod config;
pub mod dream;
pub mod engine;
pub mod fep;
pub mod memory;
pub mod topology;

#[cfg(feature = "wasm")]
pub mod wasm_bindings;

pub use config::SporeConfig;
pub use engine::SporeEngine;
