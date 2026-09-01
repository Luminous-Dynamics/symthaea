// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

pub mod app_migration;
pub mod boot_consciousness;
pub mod boot_ecology;
pub mod broca;
#[cfg(feature = "broca-full")]
pub mod broca_full;
#[cfg(feature = "broca-pipeline")]
pub mod broca_pipeline;
pub mod compass;
pub mod config;
pub mod daily_ritual;
pub mod dream;
pub mod dream_journal;
pub mod engine;
pub mod fep;
pub mod fractal;
pub mod memory;
pub mod persistence;
pub mod secure_boot;
pub mod security;
pub mod sovereign;
pub mod topology;
pub mod wellbeing_profiles;

pub mod hardware_probe;
pub mod neuroevolution;

pub mod causal;
pub mod global_workspace;
pub mod immune;
pub mod knowledge;
pub mod memory_consolidation;
pub mod reasoning;
pub mod social_tom;

#[cfg(feature = "wasm")]
pub mod quickening;

#[cfg(feature = "wasm")]
pub mod wasm_bindings;

pub use config::{SharingConfig, SporeConfig};
pub use engine::SporeEngine;
