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

/// True IIT consciousness metrics: entropy estimation, MIP search, Phi* computation.
#[allow(missing_docs)]
pub mod consciousness_metrics;
/// Core consciousness state types and configuration.
#[allow(missing_docs)]
pub mod core;
/// System bootstrap and initialization.
#[allow(missing_docs)]
pub mod genesis;
/// Hyperdimensional computing: vector types, encoding, binding, bundling, similarity, and consciousness topology.
#[allow(missing_docs)]
pub mod hdc;
/// Shared mathematical utilities: softmax, numerical helpers.
pub mod math;
/// Metrics, tracing, and introspection.
#[allow(missing_docs)]
pub mod observability;
/// Integrated Information (Phi) calculation engine.
#[allow(missing_docs)]
pub mod phi_engine;
/// Physics-grounded modeling: periodic table, emergence, kinetics, and thermodynamics.
#[allow(missing_docs)]
pub mod physics;
/// Shared temporal prediction trait for O(1) CfC-based forecasting.
pub mod temporal;

// ── Cross-variant consciousness context ─────────────────────────────────────

/// Consciousness state passed from Spore/Holon to embodied FEP agents.
///
/// Allows physical controllers (flight, humanoid, vehicle) to modulate behavior
/// based on the consciousness pipeline's current state. For example:
/// - `SafetyLevel::Red` → emergency maneuver
/// - Low `phi` → conservative, reduce exploration
/// - High `harmony_alignment` → smoother, less aggressive control
///
/// This struct is intentionally small and `Copy` so it can be passed every tick
/// without allocation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsciousnessContext {
    /// Integrated information (0.0–1.0). Higher = more unified experience.
    pub phi: f32,
    /// NRC safety level (0=Green, 1=Yellow, 2=Orange, 3=Red).
    pub safety_level: u8,
    /// Eight Harmonies alignment (0.0–1.0).
    pub harmony_alignment: f32,
    /// Current prediction error from the consciousness pipeline.
    pub prediction_error: f32,
    /// Consciousness level from master equation (0.0–1.0).
    pub consciousness_level: f32,
}

impl ConsciousnessContext {
    /// Default context representing a healthy, moderately-conscious state.
    pub const DEFAULT: Self = Self {
        phi: 0.3,
        safety_level: 0, // Green
        harmony_alignment: 0.5,
        prediction_error: 0.1,
        consciousness_level: 0.5,
    };

    /// Whether the system is in emergency mode (SafetyLevel::Red).
    pub fn is_emergency(&self) -> bool {
        self.safety_level >= 3
    }

    /// Whether exploration should be suppressed (low consciousness or elevated safety).
    pub fn should_suppress_exploration(&self) -> bool {
        self.safety_level >= 2 || self.consciousness_level < 0.15
    }

    /// Exploration gain multiplier: reduces exploration under threat, boosts under flow.
    /// Returns 0.0 (no exploration) to 2.0 (maximum exploration).
    pub fn exploration_gain(&self) -> f32 {
        if self.is_emergency() {
            return 0.0;
        }
        if self.safety_level >= 2 {
            return 0.2;
        }
        // Flow state: high phi, low PE → boost exploration
        let flow = (self.phi * 2.0).min(1.0) * (1.0 - self.prediction_error.min(1.0));
        (0.5 + flow).min(2.0)
    }
}

impl Default for ConsciousnessContext {
    fn default() -> Self {
        Self::DEFAULT
    }
}
