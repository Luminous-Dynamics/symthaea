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
//! | [`BinaryHV`](hdc::BinaryHV) | `[u8; 2048]` (16,384 bits), `Copy`, SIMD-accelerated | Fast binding, memory, STT encoding |
//! | [`ContinuousHV`](hdc::ContinuousHV) | `Vec<f32>`, configurable dimension | Gradients, phi computation, learning |
//! | [`HV`](hdc::HV) | Enum wrapping both | Unified API across representations |
//!
//! Backward-compatible alias `RealHV` is available but new code should
//! use `BinaryHV` and `ContinuousHV` directly.
//!
//! ## Modules
//!
//! - **[`hdc`]** — Hyperdimensional computing: vector types, encoding, binding, bundling,
//!   similarity search, attention, memory, and consciousness topology
//! - **[`consciousness_metrics`]** — True IIT consciousness metrics: entropy estimation,
//!   MIP search, Phi* computation, temporal/causal analysis
//! - **[`physics`]** — Physics-grounded modeling: periodic table, emergence chains,
//!   chemical kinetics, and thermodynamic consciousness
//! - **[`phi_engine`]** — Integrated Information (Φ) calculation engine
//! - **[`core`]** — Core consciousness state types and configuration
//! - **[`genesis`]** — System bootstrap and initialization
//! - **[`observability`]** — Metrics, tracing, and introspection

#![warn(missing_docs)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::new_without_default)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::if_same_then_else)]
// Known style debt, not fixed here (2026-07-02): 33 occurrences across the
// HDC/math modules, each needing real judgment rather than a mechanical
// rewrite -- type_complexity wants type aliases for research-code tuple
// types, too_many_arguments wants call sites regrouped into param structs,
// should_implement_trait wants actual trait impls (FromStr/Iterator/etc.)
// checked against their exact contract, not just renamed methods. Tracked
// as follow-up cleanup; CI now enforces -D warnings on everything else.
#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::redundant_guards)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::manual_is_multiple_of)]
// Suppress test-harness-generated deprecated warnings for phi_real module tests.
// phi_real is #![deprecated] (renamed to spectral_connectivity), but the linter
// hook restores its tests; the test harness references deprecated test constants
// at crate level which cannot be suppressed locally.
#![cfg_attr(test, allow(deprecated))]

/// Bounded winding/path-class witnesses for explicitly supported S1 and T2 spaces.
pub mod bounded_topology;
/// True IIT consciousness metrics: entropy estimation, MIP search, Phi* computation.
#[allow(missing_docs)]
pub mod consciousness_metrics;
/// Lie-group and articulated configuration-space geometry for planning substrates.
pub mod configuration_space;
/// Tri-state continuous validity, analytic synthetic oracles, and path replay.
pub mod continuous_reachability;
/// Core consciousness state types and configuration.
#[allow(missing_docs)]
pub mod core;
/// Shared embodiment types: MotorSafetyLevel, EmbodimentBridge trait, telemetry.
#[allow(missing_docs)]
pub mod embodiment;
/// System bootstrap and initialization.
#[allow(missing_docs)]
pub mod genesis;
/// Hyperdimensional computing: vector types, encoding, binding, bundling, similarity, and consciousness topology.
#[allow(missing_docs)]
pub mod hdc;
/// Analytic R3/R4 finite-w shell evaluator for HYPERSPACE-001.
pub mod hyperspace_benchmark;
/// Deterministic HYPERSPACE-001 H0-H6 intervention campaign.
pub mod hyperspace_campaign;
/// Fail-closed implicit equality manifolds, local projection, and tangent bases.
pub mod implicit_manifold;
/// Qualified local charts, overlap witnesses, and partial finite-atlas evidence.
pub mod manifold_atlas;
/// Fallible coordinate-dependent Riemannian metric fields with validated samples.
pub mod metric_field;
/// Shared mathematical utilities: softmax, numerical helpers.
pub mod math;
/// Planar SO(2)/SE(2) configuration geometry built on the shared circle factor.
pub mod planar_configuration;
/// Proof-strength reachability results and complete finite reference planning.
pub mod reachability;
/// Deterministic seeded bounded RRT with no infeasibility authority.
pub mod sampling_reachability;
/// Optional smooth-manifold refinement over planner-neutral metric spaces.
pub mod smooth_manifold;
/// Planner-neutral state-space, metric-space, product-space, and trajectory primitives.
pub mod state_space;
/// Metrics, tracing, and introspection.
#[allow(missing_docs)]
pub mod observability;
/// Narrow Tier-1 observation contract for evidence-gathering consumers (CognitiveObservation).
#[allow(missing_docs)]
pub mod observation;
/// Integrated Information (Phi) calculation engine.
#[allow(missing_docs)]
pub mod phi_engine;
/// Physics-grounded modeling: periodic table, emergence, kinetics, and thermodynamics.
#[allow(missing_docs)]
pub mod physics;
/// Unified code synthesis trait for cross-backend code generation.
#[cfg(feature = "synthesis-trait")]
#[allow(missing_docs)]
pub mod synthesis_trait;
/// Shared temporal prediction trait for O(1) CfC-based forecasting.
pub mod temporal;

/// Runtime configuration and workspace execution primitives for the conscious engine.
pub mod rt;