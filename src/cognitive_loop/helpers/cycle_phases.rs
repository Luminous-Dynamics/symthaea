// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Result structs for extracted cycle phase helpers.
//!
//! Phase implementations live in sibling modules:
//! - `cycle_phases_memory` — resonator codebook + episodic replay
//! - `cycle_phases_dream` — dream engine
//! - `cycle_phases_urgency` — urgency computation + parameter optimization
//! - `cycle_phases_init_stats` — cycle init + end-of-cycle stats

// Re-export CognitiveLoopService so tests can use `super::*`
#[cfg(test)]
use super::super::CognitiveLoopService;

// ═══════════════════════════════════════════════════════════════════════════════
// Result structs for extracted cycle phases
// ═══════════════════════════════════════════════════════════════════════════════

/// Result from the resonator codebook growth + high-phi promotion + diversity phase.
pub(in crate::cognitive_loop) struct ResonatorCodebookResult {
    pub resonator_promotions: usize,
    pub codebook_evictions: usize,
    pub codebook_diversity: f32,
    pub codebook_utilization_rate: f32,
}

/// Result from the dream engine phase (recording, dreaming, wisdom application).
pub(in crate::cognitive_loop) struct DreamPhaseResult {
    pub dream_insights: usize,
    pub dream_phi_improvement: f32,
    pub dream_wisdom_count: usize,
}

/// Result from the episodic replay and memory coordinator phase.
pub(in crate::cognitive_loop) struct EpisodicReplayResult {
    pub surprise_replay_batch_size: usize,
    /// Phasic DA burst replay boost (number of extra episodes, 0 if DA < threshold).
    pub phasic_da_replay_boost: usize,
    /// Whether the memory database was flushed to disk this cycle.
    pub memory_db_flushed: bool,
}

/// Result from the hyper-parameter optimization phase.
pub(in crate::cognitive_loop) struct ParameterOptimizationResult {
    pub best_tau_scale: f32,
    pub phi_gain: f64,
    pub swap_occurred: bool,
}

/// Result from the urgency computation and error pattern analysis phase.
pub(in crate::cognitive_loop) struct UrgencyResult {
    pub urgency: super::super::CycleUrgency,
    pub error_pattern: &'static str,
    pub predicted_urgency: &'static str,
    pub prediction_coherence_urgency_bias: f32,
    pub error_slope: f32,
    pub oscillation_ratio: f32,
}

/// Result from the cycle init and preprocessing phase.
pub(in crate::cognitive_loop) struct CycleInitResult {
    pub exploration_urge_start: f32,
    pub startup_suppressed: bool,
    pub startup_warmup_progress: f32,
}

#[cfg(test)]
#[path = "cycle_phases_tests.rs"]
mod tests;
