// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
//! | `BinaryHV` | `[u8; 2048]` (16,384 bits), `Copy`, SIMD-accelerated | Fast binding, memory, STT encoding |
//! | `ContinuousHV` | `Vec<f32>`, configurable dimension | Gradients, phi computation, learning |
//! | `HV` | Enum wrapping both | Unified API across representations |
//!
//! Backward-compatible alias `RealHV` is available but new code should
//! use `BinaryHV` and `ContinuousHV` directly.
//!
//! ## Modules
//!
//! - **hdc** — Hyperdimensional computing: vector types, encoding, binding, bundling,
//!   similarity search, attention, memory, and consciousness topology
//! - **consciousness_metrics** — True IIT consciousness metrics: entropy estimation,
//!   MIP search, Phi* computation, temporal/causal analysis
//! - **physics** — Physics-grounded modeling: periodic table, emergence chains,
//!   chemical kinetics, and thermodynamic consciousness
//! - **phi_engine** — Integrated Information (Φ) calculation engine
//! - **core** — Core consciousness state types and configuration
//! - **genesis** — System bootstrap and initialization
//! - **observability** — Metrics, tracing, and introspection

#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::new_without_default)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::redundant_guards)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::manual_is_multiple_of)]
#![cfg_attr(test, allow(deprecated))]

#[allow(missing_docs)]
pub mod consciousness_metrics;
#[allow(missing_docs)]
pub mod core;
#[allow(missing_docs)]
pub mod embodiment;
#[allow(missing_docs)]
pub mod genesis;
#[allow(missing_docs)]
pub mod hdc;
pub mod math;
#[allow(missing_docs)]
pub mod observability;
#[allow(missing_docs)]
pub mod observation;
/// Validated, modality-neutral observation envelope for sensor and evidence-fabric consumers.
#[allow(missing_docs)]
pub mod observation_fabric;
#[allow(missing_docs)]
pub mod phi_engine;
#[allow(missing_docs)]
pub mod physics;
#[cfg(feature = "synthesis-trait")]
#[allow(missing_docs)]
pub mod synthesis_trait;
pub mod temporal;
pub mod rt;
