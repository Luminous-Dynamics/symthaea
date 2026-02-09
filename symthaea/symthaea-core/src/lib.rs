//! # Symthaea Core
//!
//! The mathematical and structural foundation for the Holographic Liquid Brain.
//! Provides Hyperdimensional Computing (HDC) primitives, Integrated Information Theory (IIT/Φ),
//! and physics-grounded consciousness modeling.
//!
//! ## Hypervector Type System
//!
//! All semantic content is encoded in high-dimensional vectors. Two canonical types:
//!
//! | Type | Representation | Use Case |
//! |------|---------------|----------|
//! | [`BinaryHV`](hdc::BinaryHV) | `[u8; 2048]` (16,384 bits), `Copy`, SIMD-accelerated | Fast binding, memory, STT encoding |
//! | [`ContinuousHV`](hdc::ContinuousHV) | `Vec<f32>`, configurable dimension | Gradients, phi computation, learning |
//! | [`HV`](hdc::HV) | Enum wrapping both | Unified API across representations |
//!
//! Backward-compatible aliases `HV16` and `RealHV` are available but new code should
//! use `BinaryHV` and `ContinuousHV` directly.
//!
//! ## Modules
//!
//! - **[`hdc`]** — Hyperdimensional computing: vector types, encoding, binding, bundling,
//!   similarity search, attention, memory, and consciousness topology
//! - **[`physics`]** — Physics-grounded modeling: periodic table, emergence chains,
//!   chemical kinetics, IIT/Φ computation, and thermodynamic consciousness
//! - **[`phi_engine`]** — Integrated Information (Φ) calculation engine
//! - **[`core`]** — Core consciousness state types and configuration
//! - **[`genesis`]** — System bootstrap and initialization
//! - **[`observability`]** — Metrics, tracing, and introspection

#![allow(clippy::needless_range_loop)]
// Suppress test-harness-generated deprecated warnings for phi_real module tests.
// phi_real is #![deprecated] (renamed to spectral_connectivity), but the linter
// hook restores its tests; the test harness references deprecated test constants
// at crate level which cannot be suppressed locally.
#![cfg_attr(test, allow(deprecated))]

pub mod core;
pub mod genesis;
pub mod hdc;
pub mod observability;
pub mod phi_engine;
pub mod physics;
