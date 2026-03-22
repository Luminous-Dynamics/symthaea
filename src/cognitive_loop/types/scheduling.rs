// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Scheduling types — urgency and cycle state.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// CYCLE URGENCY — adaptive subsystem scheduling
// ═══════════════════════════════════════════════════════════════════════════════

/// Urgency level controlling how many subsystems run each cycle.
///
/// Instead of fixed "every Nth cycle" throttling, urgency adapts to the
/// system's current needs:
/// - **Critical**: High error or surprise — run everything for maximum adaptation
/// - **Normal**: Standard processing — run most subsystems
/// - **Cruise**: Low error, stable state — skip expensive subsystems to save compute
///
/// Subsystems decide per-urgency whether to run:
/// - Core pipeline (HDC→CfC→predict→learn): always runs
/// - Moral evaluation: Critical+Normal (skip in Cruise unless new input)
/// - Enhanced FEP: Critical always, Normal every 4th, Cruise every 8th
/// - Stability regime: Critical+Normal, Cruise every 4th
/// - Consciousness monitors (resonance, quantum, temporal): Normal+Critical only
/// - Master equation: Critical every 5th, Normal every 10th, Cruise every 20th
/// - Body awareness (virtual body, affective, embodied): Normal+Critical, Cruise every 2nd
/// - Self models (meta-cognition, narrative, predictive mind/self): C=1, N=2, Cr=4
/// - Workspace (attention schema, GWT, cross-modal, narrative-GWT): C=1, N=2, Cr=4
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CycleUrgency {
    /// High prediction error or surprise — run all subsystems
    Critical,
    /// Standard processing
    #[default]
    Normal,
    /// Low error, stable state — minimal subsystem overhead
    Cruise,
}

impl CycleUrgency {
    /// Derive urgency from raw arousal level (used by Mind auto-emit).
    ///
    /// Maps the biorhythm arousal value to a CycleUrgency level:
    /// - `> 0.7` → Critical (high arousal, blast wisdom immediately)
    /// - `> 0.3` → Normal  (standard processing)
    /// - `≤ 0.3` → Cruise  (low arousal, conserve bandwidth)
    #[allow(dead_code)] // Called from mind/tick.rs — unused in default feature set
    pub(crate) fn from_arousal(arousal: f32) -> Self {
        if arousal > 0.7 {
            Self::Critical
        } else if arousal > 0.3 {
            Self::Normal
        } else {
            Self::Cruise
        }
    }

    /// Compute urgency from current cycle state.
    ///
    /// - `prediction_error`: current cycle's prediction error
    /// - `learning_threshold`: config threshold for "significant" error
    /// - `surprise_triggered`: whether the surprise bridge triggered this cycle
    /// - `consecutive_low_error`: how many consecutive cycles have had error < threshold
    pub(crate) fn from_state(
        prediction_error: f32,
        learning_threshold: f32,
        surprise_triggered: bool,
        consecutive_low_error: u32,
    ) -> Self {
        if surprise_triggered || prediction_error > learning_threshold * 3.0 {
            CycleUrgency::Critical
        } else if prediction_error > learning_threshold || consecutive_low_error < 10 {
            CycleUrgency::Normal
        } else {
            CycleUrgency::Cruise
        }
    }

    /// Whether this urgency level should run a subsystem at the given cycle interval.
    /// Returns true if the subsystem should run this cycle.
    #[inline]
    pub(crate) fn should_run(
        &self,
        cycle: usize,
        critical_interval: usize,
        normal_interval: usize,
        cruise_interval: usize,
    ) -> bool {
        let interval = match self {
            CycleUrgency::Critical => critical_interval,
            CycleUrgency::Normal => normal_interval,
            CycleUrgency::Cruise => cruise_interval,
        };
        interval == 0 || cycle % interval == 0
    }

    /// Whether to run expensive consciousness monitors (resonance, quantum, temporal).
    #[inline]
    pub(crate) fn run_consciousness_monitors(&self) -> bool {
        matches!(self, CycleUrgency::Critical | CycleUrgency::Normal)
    }
}

/// Read-only snapshot of shared cycle state, passed to extracted phase functions
/// to replace loose multi-parameter signatures.
#[derive(Debug, Clone)]
pub(crate) struct CycleState<'a> {
    pub compressed_state: &'a [f32],
    pub output: &'a [f32],
    pub prediction_error: f32,
    pub coherence: f32,
    pub unified_psi: f64,
    pub phi_attention_weight: f32,
    pub hv16_cached: &'a symthaea_core::hdc::BinaryHV,
    pub input: &'a str,
    pub urgency: CycleUrgency,
    pub attention_budget_exceeded: bool,
    pub predictive_budget_gated: bool,
    /// Whether a visual scene was recognized this cycle (for dream salience boost).
    #[cfg(feature = "vision-manifold")]
    pub scene_recognized: bool,
    /// Semantic embedding from neural encoder (for episodic memory similarity).
    #[cfg(feature = "semantic-encoder")]
    pub semantic_embedding: Option<Vec<f32>>,
}
