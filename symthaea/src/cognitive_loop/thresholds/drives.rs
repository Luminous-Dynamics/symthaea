// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Drive system constants: frustration, engagement, exploration.

// ═══════════════════════════════════════════════════════════════════════════════
// USER STATE INFERENCE — FRUSTRATION & ENGAGEMENT
// ═══════════════════════════════════════════════════════════════════════════════

/// Frustration level above which exploration is dampened.
/// Basis: Yerkes-Dodson (1908) — high arousal impairs complex task performance.
pub const FRUSTRATION_DAMPEN_THRESHOLD: f64 = 0.5;

/// Frustration dampening gain — exploration reduction per unit above threshold.
pub const FRUSTRATION_DAMPEN_GAIN: f32 = 0.3;

/// Frustration level triggering NE baseline nudge (locus coeruleus activation).
/// Basis: Sapolsky (2004) — stress-arousal axis engages at moderate frustration.
pub const FRUSTRATION_NE_NUDGE_THRESHOLD: f32 = 0.4;

/// NE baseline nudge scale per unit frustration above threshold.
/// Basis: Schultz (1997) — gentle baseline shifts, naturally decays via bath.
pub const FRUSTRATION_NE_NUDGE_SCALE: f32 = 0.03;

/// Engagement level below which DA baseline is reduced (anhedonia pathway).
pub const ENGAGEMENT_LOW_THRESHOLD: f64 = 0.3;

/// DA baseline boost per flow-cycle.
pub const FLOW_DA_NUDGE: f32 = 0.02;

/// DA baseline reduction per disengaged cycle.
pub const DISENGAGEMENT_DA_NUDGE: f32 = 0.01;

/// Exploration bonus increment per flow-cycle.
/// Basis: Csikszentmihalyi (1990) — flow enables safe exploration.
pub const FLOW_EXPLORATION_INCREMENT: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// NEUROMOD BASELINE BOUNDS
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum neuromodulator baseline — floor for adjust_baseline() clamping.
/// Basis: Doya (2002) — biological transmitter levels have a physiological floor.
pub const NEUROMOD_BASELINE_MIN: f32 = 0.2;

/// Maximum neuromodulator baseline — ceiling for adjust_baseline() clamping.
pub const NEUROMOD_BASELINE_MAX: f32 = 0.8;

// ═══════════════════════════════════════════════════════════════════════════════
// EMA SMOOTHING ALPHAS
// ═══════════════════════════════════════════════════════════════════════════════

/// EMA alpha for flow state tracking (error, coherence averages).
/// Window ≈ 5 cycles. Balances responsiveness with noise rejection.
pub const EMA_ALPHA_FLOW: f32 = 0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// COHERENCE THRESHOLDS
// ═══════════════════════════════════════════════════════════════════════════════

/// Very low coherence — triggers urgency boost and learning rate dampening.
pub const COHERENCE_VERY_LOW: f32 = 0.2;

/// Low coherence — signals quality concerns, moderate LR dampening.
pub const COHERENCE_LOW: f32 = 0.3;

/// Moderate coherence — gate for quality + coherence compound checks.
pub const COHERENCE_MODERATE: f32 = 0.5;

/// High coherence — indicates stable, mature predictions.
pub const COHERENCE_HIGH: f32 = 0.7;
