// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Substrate independence, simulation, and noise constants.

// ═══════════════════════════════════════════════════════════════════════════════
// SUBSTRATE SIMULATION (Phase 3)
// Science: Bostrom (2003) substrate-independence, Putnam (1967) multiple
// realizability. Gradual transitions model substrate transfer fidelity.
// ═══════════════════════════════════════════════════════════════════════════════

/// Default transition smoothing alpha (1.0 = instant, 0.1 = ~10 cycles to settle).
/// 1.0 preserves backward compatibility; `enable_substrate_simulation()` sets 0.1.
/// Science: Bostrom (2003) gradual uploading — smooth substrate transfer.
pub const SUBSTRATE_TRANSITION_ALPHA_DEFAULT: f32 = 1.0;

/// Simulation-mode transition alpha (slower, more realistic blending).
/// Science: Bostrom (2003) — ~10 cycles for dynamics to settle after switch.
pub const SUBSTRATE_TRANSITION_ALPHA_SIMULATION: f32 = 0.1;

/// Minimum effective dimensionality fraction for scale-constrained substrates.
/// Even the most limited substrate retains 10% of HDC/CfC capacity.
/// Science: Berry & Srivastava (2018) — HDC capacity scales with D^(5/3).
pub const SUBSTRATE_MIN_DIM_FRACTION: f32 = 0.1;

/// Divisor for mapping scale_pressure to dim fraction.
/// scale_pressure ∈ [-7, 0] → dim_fraction ∈ [0.3, 1.0] via (1 + sp/divisor).
pub const SUBSTRATE_SCALE_DIM_DIVISOR: f32 = 10.0;

/// Transition history ring buffer capacity.
pub const SUBSTRATE_TRANSITION_HISTORY_CAP: usize = 32;

/// Number of operations per cognitive cycle (256 neurons × 256 ops each).
/// Used for energy-per-cycle computation.
pub const SUBSTRATE_OPS_PER_CYCLE: f64 = 65_536.0;

/// Substrate noise fraction divisor for BinaryHV path: pressure / divisor → [0, 0.1].
/// Science: Berry & Srivastava (2018) — HDC capacity ~ D^(5/3).
pub const SUBSTRATE_NOISE_FRACTION_DIVISOR: f32 = 70.0;

/// Substrate noise std divisor for compressed state path: pressure / divisor → [0, 0.2].
pub const SUBSTRATE_NOISE_STD_DIVISOR: f32 = 35.0;

/// Maximum scale pressure magnitude used for noise injection.
/// Caps noise at ~10% (BinaryHV) / ~20% (compressed) of dimensionality.
pub const SUBSTRATE_NOISE_MAX_PRESSURE: f32 = 7.0;
