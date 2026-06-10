// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use serde::{Deserialize, Serialize};

/// Complete neurochemical state snapshot for telemetry/visualization.
///
/// Consolidates all bath state into a single struct, sampled periodically
/// for dashboard display, logging, and offline analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromodSnapshot {
    // ── Effective levels ──
    pub da_effective: f32,
    pub ne_effective: f32,
    pub sht_effective: f32,
    pub ach_effective: f32,
    // ── Phasic bursts ──
    pub da_phasic: f32,
    pub ne_phasic: f32,
    // ── Receptor sensitivities (overall) ──
    pub da_sensitivity: f32,
    pub ne_sensitivity: f32,
    pub sht_sensitivity: f32,
    pub ach_sensitivity: f32,
    // ── Receptor subtypes ──
    pub da_d1: f32,
    pub da_d2: f32,
    pub ne_alpha: f32,
    pub ne_beta: f32,
    // ── Cross-modulation weights (flattened 4×4) ──
    pub cross_mod_weights: [f32; 16],
    // ── Derived control signals ──
    pub consciousness_mod: f32,
    pub plasticity_gate: f32,
    pub attention_allocation: f32,
    pub mcts_exploration_mod: f32,
    pub sleep_consolidation_boost: f32,
    pub behavioral_flexibility: f32,
    pub gradient_scale: f32,
    pub threshold_gate: f32,
    // ── Phase 4: New transmitters ──
    pub gaba_effective: f32,
    pub oxytocin_effective: f32,
    pub glutamate_effective: f32,
    pub global_inhibition: f32,
    pub social_coherence: f32,
    pub trust_factor: f32,
    pub learning_fatigue: f32,
    pub excitotoxicity_risk: f32,
    // ── Phase 5: Tachyphylaxis ──
    pub tolerant_count: u8,
    pub withdrawal_count: u8,
    // ── Phase 5: Advanced neuroendocrine ──
    pub adenosine_effective: f32,
    pub sleep_pressure: f32,
    pub allostatic_load: f32,
    pub ei_ratio: f32,
    pub ei_seizure_events: u32,
    pub active_injection_count: u8,
    // ── Phase 6: Endocannabinoid + receptor subtypes ──
    pub endocannabinoid_effective: f32,
    pub sht_1a_signal: f32,
    pub sht_2a_signal: f32,
    pub gaba_a_signal: f32,
    pub gaba_b_signal: f32,
    // ── Round 3: Per-transmitter tolerance/withdrawal observability ──
    pub da_high_exposure: u32,
    pub da_withdrawal: u32,
    pub ne_high_exposure: u32,
    pub ne_withdrawal: u32,
    pub sht_high_exposure: u32,
    pub sht_withdrawal: u32,
    pub ach_high_exposure: u32,
    pub ach_withdrawal: u32,
    pub gaba_high_exposure: u32,
    pub gaba_withdrawal: u32,
    pub oxytocin_high_exposure: u32,
    pub oxytocin_withdrawal: u32,
    pub glutamate_high_exposure: u32,
    pub glutamate_withdrawal: u32,
    pub adenosine_high_exposure: u32,
    pub adenosine_withdrawal: u32,
    pub endocannabinoid_high_exposure: u32,
    pub endocannabinoid_withdrawal: u32,
}

/// Persistent neurochemistry state (receptor sensitivities + cross-modulation weights).
///
/// Checkpointed across sessions so personality adapts over time.
/// Science: Volkow et al. (2004) — receptor density changes persist for weeks/months.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeurochemistryCheckpoint {
    /// DA receptor sensitivity
    pub da_sensitivity: f32,
    /// NE receptor sensitivity
    pub ne_sensitivity: f32,
    /// 5-HT receptor sensitivity
    pub sht_sensitivity: f32,
    /// ACh receptor sensitivity
    pub ach_sensitivity: f32,
    /// Cross-modulation weights (Hebbian-learned)
    pub cross_mod_weights: [[f32; 4]; 4],
    /// DA D1 (excitatory/Go) subtype sensitivity
    #[serde(default = "default_one")]
    pub da_d1_sensitivity: f32,
    /// DA D2 (inhibitory/NoGo) subtype sensitivity
    #[serde(default = "default_one")]
    pub da_d2_sensitivity: f32,
    /// NE Alpha (tonic precision) subtype sensitivity
    #[serde(default = "default_one")]
    pub ne_alpha_sensitivity: f32,
    /// NE Beta (phasic reactivity) subtype sensitivity
    #[serde(default = "default_one")]
    pub ne_beta_sensitivity: f32,
    /// GABA receptor sensitivity
    #[serde(default = "default_one")]
    pub gaba_sensitivity: f32,
    /// Oxytocin receptor sensitivity
    #[serde(default = "default_one")]
    pub oxytocin_sensitivity: f32,
    /// Glutamate receptor sensitivity
    #[serde(default = "default_one")]
    pub glutamate_sensitivity: f32,
    /// Sustained glutamate high cycles
    #[serde(default)]
    pub glutamate_high_cycles: u32,
    // ── Phase 5: Tachyphylaxis state (Gainetdinov 2004) ──────────────
    #[serde(default)]
    pub da_high_exposure: u32,
    #[serde(default)]
    pub da_withdrawal: u32,
    #[serde(default)]
    pub ne_high_exposure: u32,
    #[serde(default)]
    pub ne_withdrawal: u32,
    #[serde(default)]
    pub sht_high_exposure: u32,
    #[serde(default)]
    pub sht_withdrawal: u32,
    #[serde(default)]
    pub ach_high_exposure: u32,
    #[serde(default)]
    pub ach_withdrawal: u32,
    #[serde(default)]
    pub gaba_high_exposure: u32,
    #[serde(default)]
    pub gaba_withdrawal: u32,
    #[serde(default)]
    pub oxytocin_high_exposure: u32,
    #[serde(default)]
    pub oxytocin_withdrawal: u32,
    #[serde(default)]
    pub glutamate_high_exposure: u32,
    #[serde(default)]
    pub glutamate_withdrawal: u32,
    // ── Phase 5: Adenosine checkpoint ──
    #[serde(default = "default_one")]
    pub adenosine_sensitivity: f32,
    #[serde(default)]
    pub adenosine_high_exposure: u32,
    #[serde(default)]
    pub adenosine_withdrawal: u32,
    // ── Phase 5: Allostatic load checkpoint ──
    #[serde(default)]
    pub allostatic_load: f32,
    #[serde(default)]
    pub allostatic_recovery_cycles: u32,
    // ── Phase 6: Endocannabinoid checkpoint ──
    #[serde(default = "default_one")]
    pub endocannabinoid_sensitivity: f32,
    #[serde(default)]
    pub endocannabinoid_high_exposure: u32,
    #[serde(default)]
    pub endocannabinoid_withdrawal: u32,
    // ── Phase 6: 5-HT + GABA subtype sensitivities ──
    #[serde(default = "default_one")]
    pub sht_1a_sensitivity: f32,
    #[serde(default = "default_one")]
    pub sht_2a_sensitivity: f32,
    #[serde(default = "default_one")]
    pub gaba_a_sensitivity: f32,
    #[serde(default = "default_one")]
    pub gaba_b_sensitivity: f32,
}

fn default_one() -> f32 {
    1.0
}
