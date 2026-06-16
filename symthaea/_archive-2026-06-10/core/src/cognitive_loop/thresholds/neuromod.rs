// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neuromodulator bath, calibration, and stillness constants.

// ═══════════════════════════════════════════════════════════════════════════════
// NEUROMOD PHASE — BATH MODULATION PARAMETERS
// ═══════════════════════════════════════════════════════════════════════════════

/// D2 flexibility baseline for exploration responsiveness scaling.
/// Basis: Frank (2005) — D2 receptor modulation of behavioral flexibility.
pub const NEUROMOD_D2_FLEXIBILITY_BASELINE: f64 = 0.5;

/// Attention sensitivity floor — minimum multiplicative sensitivity.
/// Basis: Sarter et al. (2005) — cholinergic modulation bounds.
pub const NEUROMOD_ATTENTION_SENSITIVITY_MIN: f32 = 0.5;

/// Attention sensitivity ceiling — maximum multiplicative sensitivity.
pub const NEUROMOD_ATTENTION_SENSITIVITY_MAX: f32 = 2.0;

/// NE phasic burst threshold for attentional reorienting.
/// Basis: Corbetta & Shulman (2002) — stimulus-driven attention shift.
pub const NEUROMOD_NE_PHASIC_THRESHOLD: f32 = 0.3;

/// NE phasic attention gain — sensitivity boost per unit above threshold.
pub const NEUROMOD_NE_PHASIC_ATTENTION_GAIN: f32 = 0.5;

/// NE phasic exploration scale — exploration delta per unit above threshold.
pub const NEUROMOD_NE_PHASIC_EXPLORATION_SCALE: f32 = 0.15;

/// Arousal EMA decay weight (prior cycle contribution).
/// Basis: Berridge & Waterhouse (2003) — NE-arousal bidirectional coupling.
pub const NEUROMOD_AROUSAL_EMA_DECAY: f32 = 0.9;

/// Arousal EMA input weight (current NE contribution).
pub const NEUROMOD_AROUSAL_EMA_INPUT: f32 = 0.1;

/// NE phasic threshold for transient arousal spike.
pub const NEUROMOD_AROUSAL_PHASIC_THRESHOLD: f32 = 0.2;

/// Arousal phasic spike scale — arousal boost per unit of phasic NE.
pub const NEUROMOD_AROUSAL_PHASIC_SPIKE: f32 = 0.05;

/// Confidence crash velocity threshold — triggers 5-HT emergency dip.
/// Basis: Cools et al. (2008) — serotonergic response to prediction failure.
pub const NEUROMOD_CONFIDENCE_CRASH_VELOCITY: f64 = -0.15;

/// Serotonin emergency production during confidence crash.
pub const NEUROMOD_SEROTONIN_CRASH_PRODUCTION: f32 = -0.1;

/// Exploration baseline above which 5-HT drain occurs.
/// Basis: Tops et al. (2009) — serotonin depletion from sustained exploration.
pub const NEUROMOD_EXPLORATION_DRAIN_BASELINE: f64 = 0.5;

/// Exploration 5-HT drain factor per unit above baseline.
pub const NEUROMOD_EXPLORATION_DRAIN_FACTOR: f64 = 0.03;

/// GABA inhibition threshold — below this, learning/exploration are suppressed.
/// Basis: Olsen & Sieghart (2009) — GABAergic global inhibition.
pub const NEUROMOD_GABA_INHIBITION_THRESHOLD: f32 = 0.95;

/// Seizure protection: exploration freeze factor during E/I recovery.
/// Basis: Turrigiano (2012) — homeostatic plasticity during E/I imbalance.
pub const NEUROMOD_SEIZURE_EXPLORATION_FREEZE: f32 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// HOMEOSTASIS → NEUROMODULATOR CALIBRATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Homeostasis efficiency overshoot threshold → recalibrate neuromods.
/// Science: Turrigiano (2008) — sustained homeostatic failure triggers
/// recalibration of baseline transmitter levels.
pub const HOMEOSTASIS_RECALIBRATE_HIGH: f32 = 1.15;

/// Homeostasis efficiency undershoot threshold → boost neuromods.
pub const HOMEOSTASIS_RECALIBRATE_LOW: f32 = 0.85;

/// Neuromodulator baseline adjustment step per cycle of mistuning.
pub const HOMEOSTASIS_NEUROMOD_STEP: f32 = 0.01;

/// EMA smoothing alpha for end-of-cycle neuromodulator averages.
/// Science: Doya (2002) — slow EMA tracks tonic neuromodulator levels.
pub const NEUROMOD_EMA_ALPHA: f32 = 0.05;
