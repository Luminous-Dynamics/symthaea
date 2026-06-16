// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Telemetry types — CycleMetadata and sub-structs.
//!
//! # Field Naming Convention
//!
//! All telemetry fields should follow `domain_component_aspect` naming:
//!
//! | Domain prefix | Example | Scope |
//! |---------------|---------|-------|
//! | `neuromod_*` | `neuromod_da_d1` | Neuromodulator bath signals |
//! | `consciousness_*` | `consciousness_level` | Unified consciousness metrics |
//! | `reasoning_*` | `reasoning_gate_blocked` | Reasoning engine state |
//! | `substrate_*` | `substrate_feasibility` | Substrate independence |
//! | `circadian_*` | `circadian_hour` | Chronobiology |
//! | `structural_*` | `structural_micro_phi` | Hierarchical Phi |
//! | `self_assessment_*` | `self_assessment_accuracy` | Meta-cognitive self-model |
//! | `temporal_*` | `temporal_coherence_score` | Temporal dynamics |
//! | `embodied_*` | `embodied_phi_modulation` | Embodied cognition |
//!
//! **Legacy exceptions** (pre-convention, kept for dashboard compatibility):
//! - `dopamine_effective` (should be `neuromod_dopamine_effective`)
//! - `noradrenaline_effective`, `serotonin_effective`, `acetylcholine_effective`
//! - `body_*` fields (should be `embodied_*`)
//!
//! New fields MUST use the domain prefix. Do not add unprefixed fields.

use serde::{Deserialize, Serialize};
use symthaea_types::N_HARMONIES;

use super::scheduling::CycleUrgency;

// ═══════════════════════════════════════════════════════════════════════════════
// NEUROMODULATOR TELEMETRY — collected per-cycle from NeuromodulatorBath
// ═══════════════════════════════════════════════════════════════════════════════

/// Neuromodulator telemetry snapshot for CycleMetadata construction.
///
/// Extracted as a standalone struct so cycle.rs can build neuromod telemetry
/// in a focused helper rather than inlining 30+ fields in the metadata literal.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NeuromodTelemetry {
    /// Whether the neuromodulator bath suggests querying the exocortex (swarm).
    pub exocortex_query_suggested: bool,
    /// Neurochemical personality description (e.g., "novelty-seeking, cautious").
    pub neuromod_personality: String,
    /// Effective dopamine signal (reward/learning drive, 0.0–2.0).
    pub dopamine_effective: f32,
    /// Effective noradrenaline signal (arousal/exploration, 0.0–2.0).
    pub noradrenaline_effective: f32,
    /// Effective serotonin signal (satisfaction/confidence, 0.0–2.0).
    pub serotonin_effective: f32,
    /// Effective acetylcholine signal (attention/precision, 0.0–2.0).
    pub acetylcholine_effective: f32,
    /// Personality drift rate (max trait delta per snapshot, 0.0 = stable).
    pub neuromod_personality_drift: f32,
    /// Whether personality drift exceeds the anomaly threshold (>0.005/snapshot).
    pub neuromod_personality_drift_anomalous: bool,
    /// DA gradient scaling factor applied to training LR (0.5–2.0).
    pub neuromod_gradient_scale: f32,
    /// ACh threshold gate factor applied to learning threshold (0.5–1.5).
    pub neuromod_threshold_gate: f32,
    /// Cumulative exocortex trigger count since startup.
    pub exocortex_trigger_count: u64,
    /// DA phasic burst magnitude (fast-decaying RPE signal, 0.0–1.0).
    pub neuromod_da_phasic: f32,
    /// NE phasic burst magnitude (fast-decaying surprise signal, 0.0–1.0).
    pub neuromod_ne_phasic: f32,
    /// Neurochemical consciousness modulation factor (0.6–1.2).
    pub neuromod_consciousness_mod: f32,
    /// Sleep consolidation boost (1.0–3.0).
    pub neuromod_sleep_consolidation_boost: f32,
    /// Neuromod-driven attention budget multiplier (0.8–1.5).
    pub neuromod_attention_allocation: f32,
    /// ACh plasticity gate (0.2–1.0).
    pub neuromod_plasticity_gate: f32,
    /// 5-HT/NE MCTS exploration modulation (0.6–1.8).
    pub neuromod_mcts_exploration_mod: f32,
    /// Average DA tag on replayed episodes this cycle (0.0–1.0).
    pub replay_da_tag_avg: f32,
    /// Circadian hour used for continuous waveform modulation (0.0–24.0).
    pub circadian_hour: f32,
    /// DA D1 (Go pathway) effective signal (0.0–2.0).
    pub neuromod_da_d1: f32,
    /// DA D2 (NoGo pathway) effective signal (0.0–2.0).
    pub neuromod_da_d2: f32,
    /// NE alpha (tonic precision) effective signal (0.0–2.0).
    pub neuromod_ne_alpha: f32,
    /// NE beta (phasic reactivity) effective signal (0.0–2.0).
    pub neuromod_ne_beta: f32,
    /// D2-mediated behavioral flexibility factor (0.7–1.5).
    pub neuromod_behavioral_flexibility: f32,
    /// Full neurochemical state snapshot (sampled every 10 cycles, None otherwise).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub neuromod_snapshot: Option<crate::cognitive_loop::neuromodulators::NeuromodSnapshot>,

    // ── Phase 4: Neuroendocrine control telemetry ──────────────────────────
    /// Derived cortisol level from NE/5-HT balance (0.0–1.0).
    pub neuromod_derived_cortisol: f32,
    /// NE phasic burst → ACh suppression magnitude (0.0–0.15).
    pub ne_ach_suppression: f32,
    /// High ACh → NE suppression magnitude (0.0–0.04).
    pub ach_ne_suppression: f32,
    /// Effective GABA signal (tonic inhibition, 0.0–2.0).
    pub neuromod_gaba_effective: f32,
    /// GABA-derived global inhibition factor (0.7–1.0).
    pub neuromod_global_inhibition: f32,
    /// Effective oxytocin signal (social bonding, 0.0–2.0).
    pub neuromod_oxytocin_effective: f32,
    /// Oxytocin-derived social coherence factor (0.8–1.3).
    pub neuromod_social_coherence: f32,
    /// Oxytocin-derived trust factor (0.8–1.2).
    pub neuromod_trust_factor: f32,
    /// Effective glutamate signal (excitatory learning, 0.0–2.0).
    pub neuromod_glutamate_effective: f32,
    /// Excitotoxicity risk from sustained high glutamate (0.0–1.0).
    pub neuromod_excitotoxicity_risk: f32,
    /// Glutamate-derived learning fatigue factor (0.5–1.0).
    pub neuromod_learning_fatigue: f32,
    /// Circadian phase offset in hours (-12.0–12.0).
    pub circadian_phase_offset: f32,
    /// Effective circadian hour after phase offset + timezone (0.0–24.0).
    pub circadian_effective_hour: f32,
    /// Timezone offset in hours from UTC (e.g., -5.0 for CDT).
    pub circadian_timezone_offset: f32,

    // ── Phase 5: Advanced Neuroendocrine Telemetry ────────────────────────
    /// Adenosine effective level (sleep pressure signal, 0.0–2.0).
    pub neuromod_adenosine_effective: f32,
    /// Sleep pressure from adenosine accumulation (0.0–2.0).
    pub neuromod_sleep_pressure: f32,
    /// Allostatic load (cumulative stress, 0.0–1.0).
    pub neuromod_allostatic_load: f32,
    /// Glutamate/GABA excitatory/inhibitory ratio.
    pub neuromod_ei_ratio: f32,
    /// Cumulative seizure-like E/I imbalance events.
    pub neuromod_ei_seizure_events: u32,
    /// Shannon entropy of bath phase space (averaged across 8 dimensions).
    pub neuromod_bath_entropy: f32,
    /// Whether an attractor has been detected in the bath phase space.
    pub neuromod_attractor_detected: bool,
    /// Number of active pharmacological injections (0–4).
    pub active_injection_count: u8,

    // ── Phase 6: Endocannabinoid + Receptor Subtype Telemetry ───────────
    /// Endocannabinoid effective level (0.0–2.0).
    pub neuromod_endocannabinoid_effective: f32,
    /// 5-HT1A signal: anxiolytic/inhibitory (serotonin × 1A sensitivity).
    pub neuromod_sht_1a_signal: f32,
    /// 5-HT2A signal: perceptual richness/consciousness (serotonin × 2A sensitivity).
    pub neuromod_sht_2a_signal: f32,
    /// GABA-A signal: fast ionotropic/sedation (GABA × A sensitivity).
    pub neuromod_gaba_a_signal: f32,
    /// GABA-B signal: slow metabotropic (GABA × B sensitivity).
    pub neuromod_gaba_b_signal: f32,

    // ── Phase 7: Self-Assessment Telemetry ─────────────────────────────
    /// Self-assessment prediction error EMA (0.0–1.0).
    pub self_assessment_pe_ema: f32,
    /// Self-assessment coherence EMA (0.0–1.0).
    pub self_assessment_coherence_ema: f32,
    /// Self-assessment confidence calibration error EMA (0.0–1.0).
    pub self_assessment_confidence_error_ema: f32,
    /// Self-assessment attention utilization EMA (0.0–1.0).
    pub self_assessment_attention_ema: f32,
    /// Self-assessment inhibition error rate EMA (0.0–1.0).
    pub self_assessment_inhibition_error_ema: f32,
    /// Observations since last calibration reset.
    pub self_assessment_observations: u32,
    /// Remaining cooldown cycles before trigger eligibility.
    pub self_assessment_cooldown: u32,
    /// Whether self-assessment triggered auto-calibration this cycle.
    pub self_assessment_calibration_fired: bool,
    /// Whether a pending calibration is waiting to be applied (e.g., during sleep).
    pub pending_calibration_waiting: bool,
    /// Inhibition error count this cycle (sum of prefrontal_veto + gate_blocked + safety).
    pub inhibition_errors_this_cycle: u8,
}

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS-LEVEL METRICS — master consciousness equation + weights
// ═══════════════════════════════════════════════════════════════════════════════

/// Consciousness equation outputs and dynamic weight convergence telemetry.
///
/// Groups the master consciousness level, profile composite, synergy scores,
/// gradient analysis, and weight convergence tracking that were previously
/// flat fields on `CycleMetadata`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConsciousnessLevelMetrics {
    /// Master Consciousness Equation level (0.0 to 1.0).
    /// Comprehensive consciousness metric combining Phi, broadcast, working memory,
    /// attention, recurrence, embodiment, knowledge, narrative, and social factors.
    /// Updated every 10th cycle; 0.0 when not yet computed.
    pub consciousness_level: f64,
    /// Multi-dimensional consciousness composite score (0.0 when off).
    pub consciousness_profile_composite: f64,
    /// Synergy-enhanced composite (non-linear dimension interactions, 0.0 when off).
    pub synergy_enhanced_composite: f64,
    /// Number of emergent consciousness properties detected (0 when off).
    pub emergent_properties_count: usize,
    /// Detected consciousness state label (e.g., "Awake", "Alert", "" when off).
    pub consciousness_state_label: String,
    /// Consciousness state level (0.0–1.0, from NSM grounding, 0.0 when off).
    pub consciousness_state_level: f64,
    /// Dynamic consciousness weights [spectral, equation, pipeline, multimodal].
    pub consciousness_weights: [f64; 4],
    /// Weight stability variance (0.0 = stable, >0.01 = oscillating).
    pub consciousness_weight_variance: f64,
    /// Layer disagreement score (0.0 = agreement, higher = divergence).
    #[serde(default)]
    pub consciousness_layer_disagreement: f64,
    /// Weakest consciousness layer label (empty when not computed).
    #[serde(default)]
    pub consciousness_weakest_layer: String,
    /// Differentiable consciousness gradient magnitude (0.0 when off).
    pub consciousness_gradient_magnitude: f64,
    /// Limiting component identified by gradient analysis ("" when off).
    pub consciousness_limiting_component: String,
    /// Weight convergence state label (Initializing/Converging/Converged/Oscillating).
    pub weight_convergence_state: String,
    /// Cycle at which weights converged (0 if not yet).
    pub convergence_cycle: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMBODIED-AFFECT METRICS — body, affect, mood, somatic
// ═══════════════════════════════════════════════════════════════════════════════

/// Embodied cognition and affective state telemetry.
///
/// Groups body model, affective bridge, mood temperature, and somatic stress
/// that were previously flat fields on `CycleMetadata`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbodiedAffectMetrics {
    /// Virtual body phi modulation (1.0 = neutral, >1 = body boosts consciousness).
    pub body_phi_modulation: f64,
    /// Virtual body affect valence (-1 to 1).
    pub body_valence: f32,
    /// Virtual body affect arousal (0 to 1).
    pub body_arousal: f32,
    /// Embodied cognition phi modulation (1.0 = neutral).
    /// 1.0 when embodied cognition is not enabled.
    pub embodied_phi_modulation: f64,
    /// Embodied cognition agency score (0.0 to 1.0).
    /// 0.0 when embodied cognition is not enabled.
    pub embodied_agency: f64,
    /// Affective bridge valence (-1 to 1, 0.0 when off).
    pub affective_valence: f32,
    /// Affective bridge arousal (0 to 1, 0.5 when off — neutral).
    pub affective_arousal: f32,
    /// Affective consciousness valence (-1.0 to 1.0, 0.0 when off).
    pub affect_consciousness_valence: f32,
    /// Affective consciousness arousal (0.0–1.0, 0.0 when off).
    pub affect_consciousness_arousal: f32,
    /// Affective bias: cognitive temperature (0.0 to 2.0).
    pub mood_temperature: f32,
    /// Somatic stress from infrastructure errors (0.0 = healthy, 1.0 = critical).
    /// Fed by the SomaticErrorBridge: lock poisoning, task panics, DB failures.
    pub somatic_stress: f64,
}

impl Default for EmbodiedAffectMetrics {
    fn default() -> Self {
        Self {
            body_phi_modulation: 1.0,
            embodied_phi_modulation: 1.0,
            affective_arousal: 0.5,
            mood_temperature: 1.0,
            body_valence: 0.0,
            body_arousal: 0.0,
            embodied_agency: 0.0,
            affective_valence: 0.0,
            affect_consciousness_valence: 0.0,
            affect_consciousness_arousal: 0.0,
            somatic_stress: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRUCTURAL PHI METRICS — hierarchical decomposition + spectral MIP
// ═══════════════════════════════════════════════════════════════════════════════

/// Structural Phi decomposition and spectral MIP telemetry.
///
/// Groups micro/meso/macro Phi, sigma, spectral and hierarchical MIP that
/// were previously flat fields on `CycleMetadata`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StructuralPhiMetrics {
    /// Micro-Phi: within-cluster integration (0 when not computed).
    pub structural_micro_phi: f64,
    /// Meso-Phi: inter-cluster integration (0 when not computed).
    pub structural_meso_phi: f64,
    /// Macro-Phi: global spectral MIP (0 when not computed).
    pub structural_macro_phi: f64,
    /// Bottleneck score: gap between macro and meso Phi.
    pub structural_bottleneck: f64,
    /// Emergence ratio: macro / (micro + meso); > 1.0 = emergent.
    pub structural_emergence_ratio: f64,
    /// Number of detected clusters in hierarchical decomposition.
    pub structural_num_clusters: usize,
    /// Σ (Sigma) — Synergistic integration via covariance-based Phi* (Layer 2).
    /// `None` when not computed this cycle (only computed every N cycles).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sigma: Option<f64>,
    /// Spectral MIP Phi — O(n³) Fiedler-ordered MIP approximation (Layer 2+).
    /// `None` when not computed this cycle (only computed every 50 cycles).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spectral_mip_phi: Option<f64>,
    /// Hierarchical spectral MIP Phi (multi-scale: 32→64→128 components).
    /// Uses coarser scales to focus finer scales on the MIP boundary region.
    /// `None` when not computed this cycle (only computed every 100 cycles).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hierarchical_mip_phi: Option<f64>,
    /// Number of scales used in hierarchical MIP (0 when not computed).
    pub hierarchical_mip_scales: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPORAL-PHENOMENAL METRICS — temporal coherence, binding, thermodynamics
// ═══════════════════════════════════════════════════════════════════════════════

/// Temporal consciousness, phenomenal binding, and thermodynamic telemetry.
///
/// Groups temporal coherence/continuity, phenomenal binding, holographic unity,
/// cross-modal binding, and thermodynamic fields that were previously flat
/// fields on `CycleMetadata`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemporalPhenomenalMetrics {
    /// Temporal consciousness coherence (0.0 to 1.0).
    /// 0.0 when temporal consciousness is not enabled.
    pub temporal_coherence_score: f64,
    /// Whether temporal consciousness analysis detected a discontinuity.
    pub temporal_discontinuity: bool,
    /// Number of causal chains detected by temporal analyzer (0 when off).
    pub temporal_causal_chains: usize,
    /// Temporal continuity ratio (0.0–1.0, 0.0 when off).
    pub temporal_continuity: f64,
    /// Longest causal chain length (0 when off).
    pub temporal_max_chain_length: usize,
    /// Phenomenal binding strength Ψ (0.0 when off).
    pub phenomenal_binding_strength: f64,
    /// Whether phenomenal binding detected fragmentation.
    pub phenomenal_fragmented: bool,
    /// Holographic consciousness unity score (0.0–1.0, 0.0 when off).
    pub holographic_unity: f64,
    /// Holographic binding strength (0.0 when off).
    pub holographic_binding: f64,
    /// Cross-modal binding strength (0.0 when off).
    pub cross_modal_binding_strength: f32,
    /// Cross-modal integration Phi (0.0 when off).
    pub cross_modal_psi: f64,
    /// Consciousness thermodynamic entropy (0.0 when off).
    pub thermodynamic_entropy: f64,
    /// Consciousness thermodynamic free energy (0.0 when off).
    pub thermodynamic_free_energy: f64,
    /// Thermodynamic load (0.0 to 1.0, where 1.0 = 6W limit reached).
    pub thermodynamic_load: f32,

    /// Epistemic free energy: F = E[loss] - T*H[beliefs].
    /// Science: arXiv 2601.17607 — Thermodynamic Theory of Learning.
    #[cfg(feature = "epistemic")]
    #[serde(default)]
    pub epistemic_free_energy: f64,

    /// Epistemic speed limit violation: W2^2 / (T * sigma). >1.0 = suspicious.
    /// Science: arXiv 2601.17607 — T * sigma >= W_2(q_0, q_1)^2.
    #[cfg(feature = "epistemic")]
    #[serde(default)]
    pub esl_violation: f64,

    /// Wasserstein-2 proxy for belief displacement this cycle.
    #[cfg(feature = "epistemic")]
    #[serde(default)]
    pub belief_distance: f64,
}

// Metadata about internal decision-making during a cycle.
//
// Provides observability into which subsystems influenced the cycle's output,
// enabling debugging of "why did the agent do that?" questions.
//
// # Domain Groups
//
// Fields are organized by domain (see section comments). Neuromod fields
// are nested via `#[serde(flatten)] pub neuromod: NeuromodTelemetry`; assign
// the snapshot directly to `metadata.neuromod`.
//
// # Diagnostic-only fields (serialized for dashboards, not read internally)
//
// These fields are populated in `cycle_phase_output.rs` and serialized via
// `#[derive(Serialize)]` for API/dashboard consumers, but no internal code
// reads them after population:
//
// `broca`, `calibration_improvements`, `calibration_regressions`,
// `convergence_cycle`, `eq_v2_limiting_component`, `feedback_signals_fired`,
// `liquid_mamba_effective_rank`, `liquid_mamba_semantic_pe`,
// `phi_validation_cached`, `social_strategy_bias_applied`,
// `subsystem_integration_contributors`

// ═══════════════════════════════════════════════════════════════════════════════
// FEEDBACK MODULATION FLAGS — Sessions 9–16 observability booleans
// ═══════════════════════════════════════════════════════════════════════════════

/// Boolean flags tracking whether specific feedback modulation pathways fired
/// during this cycle. Write-only for telemetry/dashboards — not read by
/// internal cognitive loop logic.
///
/// Flattened into `CycleMetadata` via `#[serde(flatten)]` so the JSON schema
/// remains backward-compatible.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeedbackModulationFlags {
    /// Whether feedback integration was frozen this cycle (dampening streak >= 3).
    /// Science: Turrigiano (2008) — homeostatic synaptic silencing.
    #[serde(default)]
    pub feedback_frozen: bool,
    /// Whether compound instability was detected (agreement drop + rising errors).
    /// Science: Friston (2010) — cascading precision failures.
    #[serde(default)]
    pub compound_instability: bool,
    /// Whether flow-state feedback relaxation is active (wider dampening threshold).
    /// Science: Csikszentmihalyi (1990) — reduced self-monitoring during flow.
    #[serde(default)]
    pub flow_feedback_relaxed: bool,
    /// Whether a confidence crash was detected this cycle (>30% drop).
    /// Science: Cools et al. (2008) — serotonergic dip from confidence collapse.
    #[serde(default)]
    pub confidence_crash_detected: bool,
    /// Whether low proposal diversity triggered exploration boost.
    #[serde(default)]
    pub low_diversity_boost: bool,
    /// Whether agreement-confidence velocity coupling fired this cycle.
    #[serde(default)]
    pub agreement_confidence_coupling: bool,
    /// Whether LR was frozen this cycle by crash freeze (Set proposal pinning).
    #[serde(default)]
    pub lr_frozen: bool,
    /// Whether high conflict triggered epistemic exploration boost.
    #[serde(default)]
    pub conflict_exploration_boost: bool,
    /// Whether epistemic conflict drove exploration boost (>2 conflicts).
    #[serde(default)]
    pub epistemic_conflict_exploration: bool,
    /// Whether phenomenal fragmentation triggered confidence dampening.
    #[serde(default)]
    pub phenomenal_fragmentation_recovery: bool,
    /// Whether temporal discontinuity triggered LR dampening.
    #[serde(default)]
    pub temporal_discontinuity_recovery: bool,
    /// Whether cross-modal binding modulated attention sensitivity.
    #[serde(default)]
    pub binding_attention_modulated: bool,
    /// Whether resonator similarity modulated semantic LR.
    #[serde(default)]
    pub resonator_semantic_lr_mod: bool,
    /// Whether FEP TD error convergence dampened exploration.
    #[serde(default)]
    pub fep_td_converged: bool,
    /// Whether rising confidence dampened exploration.
    #[serde(default)]
    pub confidence_rising_dampen: bool,
    /// Whether flow state boosted subsystem LR.
    #[serde(default)]
    pub flow_lr_boost: bool,
    /// Whether FEP efficiency boosted confidence via proposal system.
    #[serde(default)]
    pub fep_efficiency_boost: bool,
    /// Whether attention overload raised threshold.
    #[serde(default)]
    pub attention_overload_threshold: bool,
    /// Whether sustained high quality maintained exploration floor.
    #[serde(default)]
    pub quality_exploration_floor: bool,
    /// Whether MCE bottleneck LR boost was applied this cycle.
    #[serde(default)]
    pub mce_bottleneck_lr_applied: bool,
    /// Whether homeostasis recalibration adjusted LR (overcorrect or sluggish).
    #[serde(default)]
    pub homeostasis_recalibrated: bool,
    /// Whether falling confidence boosted LR.
    #[serde(default)]
    pub confidence_falling_lr_boost: bool,
    /// Whether coherence velocity scaled attention budget.
    #[serde(default)]
    pub coherence_velocity_budget_scaled: bool,
    /// Whether temporal chain depth modulated LR (deep=dampen, shallow=boost).
    #[serde(default)]
    pub temporal_chain_depth_lr_mod: bool,
    /// Whether EqV2 bottleneck response fired a targeted boost.
    #[serde(default)]
    pub eq_v2_bottleneck_response: bool,
    /// Whether affective consciousness modulated LR or exploration.
    #[serde(default)]
    pub affect_consciousness_modulated: bool,
    /// Whether narrative self-phi modulated confidence or exploration.
    #[serde(default)]
    pub narrative_self_phi_modulated: bool,
    /// Whether epistemic Phi (phi_eff) modulated LR or exploration.
    #[serde(default)]
    pub epistemic_phi_modulated: bool,
    /// Whether phenomenal binding strength modulated confidence or threshold.
    #[serde(default)]
    pub phenomenal_binding_modulated: bool,
    /// Whether temporal coherence modulated LR or confidence.
    #[serde(default)]
    pub temporal_coherence_modulated: bool,
    /// Whether holographic unity modulated exploration or LR.
    #[serde(default)]
    pub holographic_unity_modulated: bool,
    /// Whether harmonies alignment modulated confidence or exploration.
    #[serde(default)]
    pub harmonies_alignment_modulated: bool,
    /// Whether consciousness gradient magnitude modulated LR.
    #[serde(default)]
    pub consciousness_gradient_lr_modulated: bool,
    /// Whether value cache confidence modulated exploration or LR.
    #[serde(default)]
    pub value_cache_confidence_modulated: bool,
    /// Whether consciousness state level (high/low extremes) triggered modulation.
    #[serde(default)]
    pub consciousness_state_modulated: bool,
    /// Whether living mind vitality modulated confidence or LR.
    #[serde(default)]
    pub living_mind_vitality_modulated: bool,
    /// Whether living mind coherence modulated confidence or LR.
    #[serde(default)]
    pub living_mind_coherence_modulated: bool,
    /// Whether MCTS plan effectiveness triggered modulation (high or low extremes).
    #[serde(default)]
    pub mcts_effectiveness_modulated: bool,
    /// Whether living mind vitality modulated confidence.
    #[serde(default)]
    pub living_mind_vitality_feedback: bool,
    /// Whether low meta-cognitive accuracy dampened subsystem LR.
    #[serde(default)]
    pub metacog_low_accuracy_dampen: bool,
    /// Whether predictive self-safety boosted LR.
    #[serde(default)]
    pub self_safety_lr_boost: bool,
    /// Whether embodied agency stable range boosted confidence.
    #[serde(default)]
    pub embodied_agency_stable: bool,
    /// Whether pipeline consciousness modulated epistemic threshold.
    #[serde(default)]
    pub pipeline_consciousness_gated: bool,
    /// Whether early low-coherence warning fired (5-10 cycles).
    #[serde(default)]
    pub low_coherence_early_warning: bool,
    /// Whether sustained mode stability dampened exploration.
    #[serde(default)]
    pub mode_stable_exploration_dampen: bool,
    /// Whether confidence crash relaxed binding threshold.
    #[serde(default)]
    pub crash_binding_relaxed: bool,
    /// Whether attention fatigue widened Broca cadence spacing.
    #[serde(default)]
    pub attention_fatigue_broca_gated: bool,
    /// Whether sustained low resonator error gave extra confidence boost.
    #[serde(default)]
    pub resonator_sustained_low_boost: bool,
    /// Whether anomaly recovery was accelerated by improving Phi.
    #[serde(default)]
    pub anomaly_recovery_phi_accelerated: bool,
    /// Whether temporal binding strength modulated exploration/LR.
    #[serde(default)]
    pub temporal_binding_feedback: bool,
    /// Whether consciousness gradient magnitude triggered caution or recovery.
    #[serde(default)]
    pub consciousness_gradient_active: bool,
    /// Whether startup exploration ramp was active (warmup phase).
    #[serde(default)]
    pub startup_exploration_ramped: bool,
    /// Whether epistemic rejection streak triggered recalibration.
    #[serde(default)]
    pub epistemic_rejection_streak_recal: bool,
    /// Whether consecutive full-dampen triggered protective threshold freeze.
    #[serde(default)]
    pub full_dampen_threshold_freeze: bool,
    /// Whether consciousness EMA biased learning rate initialization.
    #[serde(default)]
    pub consciousness_ema_lr_bias: bool,
    /// Whether multi-objective frontier size modulated exploration.
    #[serde(default)]
    pub multi_obj_frontier_gated: bool,
    /// Whether error oscillation bifurcation response fired.
    #[serde(default)]
    pub error_bifurcation_response: bool,
}

/// Metadata about internal decision-making during a cycle.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CycleMetadata {
    /// Whether the surprise exploration bridge triggered exploration this cycle
    pub surprise_triggered: bool,

    /// Whether the prefrontal cortex vetoed or modified the response
    pub prefrontal_veto: bool,

    /// Confidence score from the reasoning engine (0.0 = unused/off, >0 = active)
    pub reasoning_confidence: f32,

    /// Description of the exploration action taken (if any)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exploration_action: Option<String>,

    /// Whether the reasoning engine's tool gate blocked an action this cycle.
    /// When true, the system used a fallback strategy instead.
    pub reasoning_gate_blocked: bool,

    /// Fallback strategy selected when gating blocked an action (if any)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_fallback: Option<String>,

    /// Best action from MCTS planning (Tier 1+), if planning ran
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_plan_action: Option<usize>,

    /// MCTS plan confidence (0.0 = no plan, >0 = plan confidence)
    pub reasoning_plan_confidence: f32,

    /// Human-readable reasoning narrative (Tier 2, best-effort)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_narrative: Option<String>,

    /// Quality diagnostics telemetry (meta-cognitive, dissipative, coherence, anomaly).
    #[serde(flatten)]
    pub quality: QualityDiagnostics,

    /// Narrative self-model's integrated information (0.0 = off/no self, >0 = active self-Φ)
    pub narrative_self_psi: f64,

    /// Consciousness equation outputs, weights, and convergence telemetry.
    #[serde(flatten, default)]
    pub consciousness: ConsciousnessLevelMetrics,

    /// Embodied cognition, affective state, mood, and somatic stress telemetry.
    #[serde(flatten, default)]
    pub embodied: EmbodiedAffectMetrics,

    /// Predictive self-model safety score (1.0 = safe, 0.0 = unsafe).
    /// 0.0 when predictive self is not enabled.
    pub predictive_self_safety: f32,

    /// Behavioral prediction error (average across moral_score, exploration_urge,
    /// behavioral_coherence). 0.0 when predictive self is not enabled.
    pub predictive_behavioral_error: f32,

    /// Attention subsystem telemetry (schema, GWT, budget, memoization).
    #[serde(flatten)]
    pub attention: AttentionMetrics,

    /// Consciousness resonance dominant frequency (Hz).
    /// 0.0 when resonance is not enabled or no history.
    pub resonance_frequency: f64,

    /// Quantum coherence level (0.0 to 1.0).
    /// 0.0 when quantum coherence is not enabled.
    pub quantum_coherence_level: f64,

    /// Temporal consciousness, phenomenal binding, and thermodynamic telemetry.
    #[serde(flatten, default)]
    pub temporal: TemporalPhenomenalMetrics,

    /// Structural Phi decomposition and spectral MIP telemetry.
    #[serde(flatten, default)]
    pub structural: StructuralPhiMetrics,

    /// Whether the narrative-GWT integration vetoed this cycle's action.
    pub narrative_gwt_veto: bool,

    /// Self-Phi from the narrative-GWT integration (0.0 = off/not enabled).
    pub narrative_gwt_self_psi: f64,

    /// Unified Living Mind vitality (0.0 to 1.0).
    /// Measures overall "aliveness" of the system via life-mind continuity.
    /// 0.0 when full_consciousness feature is not enabled.
    pub living_mind_vitality: f64,

    /// Unified Living Mind coherence (0.0 to 1.0).
    /// Measures integration quality of autopoietic, enactive, and predictive subsystems.
    /// 0.0 when full_consciousness feature is not enabled.
    pub living_mind_coherence: f64,

    /// Cycle urgency level (Critical/Normal/Cruise).
    /// Determines how many subsystems ran this cycle.
    pub urgency: CycleUrgency,

    /// Memory-resonator subsystem telemetry (dreams, codebook, replay).
    #[serde(flatten)]
    pub memory: MemoryResonatorMetrics,

    /// Free energy principle (FEP) and predictive processing telemetry.
    #[serde(flatten)]
    pub fep: FepTelemetry,

    /// Hierarchical total free energy (0.0 when off).
    pub hierarchical_total_free_energy: f64,

    /// Phi estimate from primitive consciousness decomposition (0.0 when off).
    pub primitive_psi: f64,

    /// Primitive lattice height (depth of cognitive integration, 0 when off).
    pub lattice_height: usize,
    /// Primitive lattice width (max parallelism at any level, 0 when off).
    pub lattice_width: usize,
    /// Integrating concept from lattice join of active primitives (empty when off).
    pub lattice_join_concept: String,
    /// Number of causal patterns added to resonator codebook this cycle.
    pub causal_codebook_entries: usize,

    // ── Session 1: Compositionality + Value Evaluator ──────────────────────
    /// Total compositions registered in the compositionality engine (0 when off).
    pub compositionality_total: usize,

    /// Composition rule engine: name of the rule applied this cycle (empty when off).
    pub composition_rule_applied: String,

    /// Harmonic and moral geometry telemetry (harmonies, guiding question, moral FEP).
    #[serde(flatten)]
    pub harmonics: HarmonicMetrics,

    /// Ethical and moral topology telemetry (moral score, value evaluator, topology).
    #[serde(flatten)]
    pub ethics: EthicalTelemetry,
    /// Multi-objective evolution: Pareto frontier size (0 when not run this cycle).
    pub multi_obj_frontier_size: usize,

    // ── Session 2: Consciousness Profile + Synergies + Context ─────────────
    /// Current reasoning context detected from input (empty when off).
    pub reasoning_context: String,
    /// Context-aware Phi weight for current context (0.0 when off).
    pub context_phi_weight: f64,

    // ── Session 3: Primitive Reasoning + Adaptive Reasoning ────────────────
    /// Primitive reasoner confidence (0.0–1.0, 0.0 when off).
    pub reasoning_chain_confidence: f32,
    /// Primitive reasoner chain depth (0 when off).
    pub reasoning_chain_depth: usize,
    /// Adaptive reasoner total Phi from RL-guided chain (0.0 when off).
    pub adaptive_reasoning_phi: f64,

    /// Causal self-explanation: total learned causal relations (0 when off).
    pub causal_relations_count: usize,
    /// Causal self-explanation: average confidence in causal model (0.0 when off).
    pub causal_avg_confidence: f64,

    /// Evolution coordinator generation count (0 when off).
    pub evolution_generation: usize,
    /// Evolution coordinator Phi delta from last evolution step (0.0 when off).
    pub evolution_phi_delta: f64,

    /// Neuroevolution generation count (0 when off or feature disabled).
    pub neuroevo_generation: u32,
    /// Neuroevolution best fitness (0.0 when off).
    pub neuroevo_best_fitness: f64,
    /// Neuroevolution population diversity (0.0 when off).
    pub neuroevo_diversity: f64,
    /// Neuroevolution species count (0 when off).
    pub neuroevo_species_count: usize,

    /// Total value-aligned embeddings created by semantic value embedder (0 when off).
    pub value_embeddings_created: u64,
    /// Semantic value embedder cache hit rate (0.0 when off).
    pub value_cache_hit_rate: f32,

    // ── Session 4: Epistemic Tiers + Phi Validation ────────────────────────
    /// Epistemic quality score from 3-axis classification (0.0–1.0, 0.0 when off).
    pub epistemic_quality: f64,
    /// Phi validation Pearson correlation (0.0 when not yet computed).
    pub phi_validation_correlation: f64,

    // ── Session 5: Conflict + Evolution ──
    /// Number of inter-theory conflicts detected (0 when off).
    pub epistemic_conflict_count: usize,

    // ── Session 6: Holographic + Differentiable + Affective + Pipeline + MultiModal ──
    /// Equation V2 limiting component from ConsciousnessEngine ("" when off).
    pub eq_v2_limiting_component: String,
    /// Unified pipeline consciousness score (0.0 when off).
    pub pipeline_consciousness: f64,
    /// Multi-modal integrated phi (0.0 when off).
    pub multimodal_integrated_phi: f64,

    // ── Session 7: Synthetic States + Epistemic Gate ────────────────────
    /// Epistemic gate confidence (0.0–1.0, 0.5 when off).
    pub epistemic_gate_confidence: f32,
    /// Whether epistemic gate approved the current cycle's action.
    pub epistemic_gate_approved: bool,

    /// Primitive validation: mean Φ gain from standard experiment (0.0 until validated).
    pub primitive_validation_phi_gain: f64,
    /// Primitive validation: statistical p-value (1.0 until validated).
    pub primitive_validation_p_value: f64,

    /// Meta-cognitive reasoning confidence (0.0–1.0, 0.5 when off or not evaluated).
    pub meta_reasoning_confidence: f64,
    /// Number of meta-learning insights discovered this cycle.
    pub meta_reasoning_insights: usize,
    /// Number of code-tier primitives selected (0 when input is non-code).
    pub code_primitives_selected: usize,

    /// Whether metacognitive monitoring detected a Phi trajectory anomaly.
    pub metacognitive_anomaly: bool,

    /// Whether the safety gateway blocked the input before processing.
    pub safety_blocked: bool,

    /// Which forbidden category the safety gateway detected (if any).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safety_category: Option<String>,

    /// Negation polarity detected in input text (0.0 = no negation, >0.5 = negated).
    pub negation_polarity: f32,

    /// Selected response strategy for this cycle (e.g., "Exploratory", "Supportive").
    pub selected_strategy: String,

    /// Actual effective learning rate used for training this cycle (after all modulations).
    /// 0.0 when no learning occurred.
    pub actual_effective_lr: f32,

    /// Cognitive group geometric mean (flow x semantic x reasoning x curiosity). 1.0 = neutral.
    pub lr_cognitive_mod: f32,

    /// Meta-learning group geometric mean (FEP x MCE x subsystem). 1.0 = neutral.
    pub lr_meta_mod: f32,

    /// Total feedback proposals this cycle across all 4 channels.
    pub feedback_proposal_count: u32,

    /// Average conflict ratio across feedback channels (0.0 = unanimous, 0.5 = max conflict).
    pub feedback_conflict_ratio: f32,

    /// Feedback proposal counts per priority: [Aesthetic, Cognitive, Homeostatic, Safety].
    pub feedback_priority_counts: [u32; 4],

    /// Feedback signal diversity (unique sources / total proposals).
    pub feedback_diversity: f32,

    /// Cycle reward signal (internal + external blend, -1.0 to 1.0).
    pub cycle_reward: f32,

    /// Number of support triage classifications this cycle (0 when support disabled).
    pub support_triage_count: u32,

    /// Whether a support predictive alert fired this cycle.
    pub support_alert_fired: bool,

    /// Number of knowledge articles graduated via federation this cycle.
    pub support_federation_graduated: usize,

    /// Support subsystem expected free energy (0.0 when not computed).
    pub support_efe: f64,

    // ── Substrate & Convergence Telemetry ──────────────────────────────
    /// Substrate telemetry snapshot (from SubstrateManager::telemetry()).
    /// Groups all substrate-related fields for batch assignment.
    #[serde(default)]
    pub substrate: super::SubstrateTelemetry,

    // ── JEPA telemetry ──────────────────────────────────────────────────
    /// JEPA latent prediction error (cosine loss, 0.0 = perfect). Feature: `jepa`.
    pub jepa_latent_pe: f32,
    /// Total energy spent by JEPA engine (joules). Feature: `jepa`.
    pub jepa_total_energy: f64,
    /// Whether JEPA representation collapse has been detected. Feature: `jepa`.
    pub jepa_collapse_detected: bool,

    /// Muse telemetry: streaming consciousness-driven music synthesis.
    #[cfg(feature = "muse")]
    #[serde(default)]
    pub muse: crate::cognitive_loop::managers::muse_manager::MuseTelemetry,

    /// Thermal telemetry snapshot (from ThermalBridge).
    /// Reports platform thermal state and CfC tau modulation.
    /// Science: Angilletta (2009) thermal performance curves.
    #[serde(default)]
    pub thermal: super::ThermalTelemetry,

    /// Integrity telemetry snapshot (from IntegrityManager).
    /// Reports tamper detection: attestation, temporal, canaries.
    #[serde(default)]
    pub integrity: super::IntegrityTelemetry,

    /// Per-module timing (microseconds). 0 = module disabled or not run this cycle.
    pub module_timings_us: ModuleTimings,

    /// Circadian phase (Dawn/Day/Dusk/Night) from chronobiology module.
    pub circadian_phase: String,

    /// Circadian plasticity modifier (0.0–1.0) applied to learning rate.
    pub circadian_plasticity: f32,

    // ── Phase 13: Cross-Module Coherence ──────
    /// Cross-module agreement score (0.0–1.0): alignment between FEP, MCTS,
    /// resonator confidence, and moral judgment.
    pub cross_module_agreement: f32,
    /// Thalamic depth score used for storage salience modulation.
    pub thalamic_depth_score: f32,

    // ── Phase 14: Subsystem Feedback Closure ──────────────────────────
    /// Whether epistemic gate gated codebook growth this cycle.
    pub epistemic_gate_gated: bool,
    /// Number of causal edges used for attention weighting this cycle.
    pub causal_attention_edges: usize,
    /// MCTS plan effectiveness score from post-hoc evaluation (0.0 when not evaluated).
    pub mcts_plan_effectiveness: f32,

    // ── Phase 15: Adaptive Architecture + Emotional Homeostasis ──────
    /// Multi-horizon prediction coherence (0.0 = divergent, 1.0 = identical).
    pub prediction_coherence: f32,
    /// Emotional valence homeostasis pull (amount returned toward baseline).
    pub valence_homeostasis_pull: f32,
    /// Emotional arousal homeostasis pull (amount returned toward baseline).
    pub arousal_homeostasis_pull: f32,
    /// Whether arousal recovery mode is active (tau slowdown engaged).
    pub arousal_recovery_active: bool,
    /// CfC tau factor from arousal recovery (1.0 = no change, <1.0 = slowdown).
    pub arousal_recovery_tau_factor: f32,
    /// Total cycle wall-clock time in microseconds (same as CycleResult.cycle_time_us
    /// but included in metadata for unified telemetry access).
    pub cycle_duration_us: u64,
    /// Predicted Phi gain from school curriculum recommendation (0.0 when school disabled).
    pub school_predicted_phi_gain: f32,

    // ── Phase 16: Quality-Aware Adaptive Processing ─────────────────
    /// Whether epistemic gate confidence gated expensive modules this cycle.
    pub epistemic_coherence_gated: bool,
    /// Phi validation correlation from most recent validation run (0.0 when not yet run).
    pub phi_validation_cached: f64,
    /// Adjusted spectral weight used in unified Psi computation.
    pub phi_spectral_weight: f32,

    // ── Phase 17: Predictive Self-Tuning ──────────────────────────────
    /// Detected error pattern (Rising/Falling/Oscillating/Spike/Stable).
    pub error_pattern: String,
    /// Whether startup transient suppression is active (cycles 0-50).
    pub startup_suppressed: bool,
    /// Startup warmup progress (0.0–1.0, 1.0 = fully warmed up).
    pub startup_warmup_progress: f32,
    /// Self-model prediction accuracy EMA (0.0–1.0).
    pub self_model_accuracy: f32,
    /// Mode transition confidence (0.0 = just switched, 1.0 = fully settled).
    pub mode_confidence: f32,
    /// Cycles since last urgency mode change.
    pub mode_stability_counter: u32,
    /// Predicted urgency for next 5 cycles (from error pattern analysis).
    pub predicted_urgency: String,

    // ── Phase 18: Closing Feedback Loops ─────────────────────────────
    /// Whether context_phi_weight was applied to modulate unified Psi this cycle.
    pub context_phi_applied: bool,
    /// Evolution phi delta feedback: confidence change applied this cycle.
    pub evolution_confidence_delta: f32,
    /// Homeostasis pull strength multiplier (urgency-adaptive, 1.0 = baseline).
    pub homeostasis_pull_strength: f32,
    /// Prediction coherence bias applied to urgency threshold (-1.0 to 1.0).
    pub prediction_coherence_urgency_bias: f32,

    /// Multimodal director telemetry (generation gating, MCE gating).
    #[serde(default)]
    pub multimodal: crate::cognitive_loop::managers::MultimodalTelemetry,

    /// Vision manifold telemetry (surprise, FEP, dreaming, dilation).
    #[cfg(feature = "vision-manifold")]
    #[serde(default)]
    pub vision: Option<symthaea_vision_manifold::VisionTelemetry>,

    // ── Phase 19: Activating Dormant Pathways ────────────────────────
    /// Consciousness limiting component that was boosted (empty when none).
    pub limiting_component_boosted: String,
    /// Love resonance confidence boost applied this cycle.
    pub love_resonance_boost: f32,
    /// Whether reasoning chain boosted prediction confidence this cycle.
    pub reasoning_chain_boosted: bool,

    // ── Phase 20: Signal-to-Control Synthesis ────────────────────────
    /// LR modulation from harmonic interferences (>0.5 dampens, <0.2 boosts).
    pub harmonic_interference_lr_mod: f32,
    /// Exploration modulation from resonator prediction error (cosine distance).
    pub resonator_error_exploration_mod: f32,
    /// Threshold modulation from phenomenal binding strength (±scale).
    pub binding_threshold_mod: f32,
    /// Whether causal density gated urgency this cycle.
    pub causal_urgency_gated: bool,
    /// Semantic LR modulation from epistemic gate confidence (previous cycle).
    pub epistemic_semantic_lr_mod: f32,
    /// Whether predictive budget gating was active (>80% budget at midpoint).
    pub predictive_budget_gated: bool,

    // ── Phase 21: Consciousness-Grounded Control ────────────────────
    /// Prediction confidence modulation from phenomenal binding strength.
    pub binding_confidence_mod: f32,
    /// Consecutive temporal discontinuity cycles (recovery cascade tracker).
    pub discontinuity_streak: u32,
    /// Whether epistemic conflicts accelerated adaptive reasoning this cycle.
    pub epistemic_reasoning_accelerated: bool,
    /// Whether low agency overrode exploratory strategy to supportive.
    pub agency_strategy_override: bool,
    /// Surprise amplitude modulation from predictive free energy.
    pub pfe_surprise_mod: f32,
    /// Adaptive memoization threshold from codebook diversity.
    pub adaptive_memo_threshold: f32,

    // ── Spatial Reasoning (GridEncoder) ─────────────────────────────────
    /// L2 norm of the grid-encoded current input (0.0 when encoder disabled).
    pub grid_encoding_norm: f32,
    /// Spatial complexity estimate from grid encoding (0.0–1.0, 0.0 when disabled).
    pub grid_spatial_complexity: f32,

    // ── Neuromodulator Bath ────────────────────────────────────────────
    /// Complete neuromodulator telemetry snapshot.
    /// Populated via `collect_neuromod_telemetry()` and assigned to `metadata.neuromod`.
    #[serde(flatten)]
    pub neuromod: NeuromodTelemetry,

    // ── Phase 4: Neuroendocrine Control (CycleMetadata-specific) ─────
    /// Phasic DA replay amplification boost (extra episodes, 0 when below threshold).
    pub neuromod_phasic_replay_boost: usize,
    /// NE phasic reorienting boost applied to attention_sensitivity.
    pub neuromod_ne_reorienting_boost: f32,
    /// Remaining cycles of anomaly drift recovery (0 = not active).
    pub neuromod_drift_recovery_remaining: u32,
    /// NE→arousal EMA feedback applied this cycle.
    pub ne_arousal_feedback: f32,
    /// Confidence velocity (rate of change for crash detection).
    pub confidence_velocity: f32,
    /// Whether a confidence crash triggered 5-HT emergency dip.
    pub sht_crash_dip: bool,
    /// Exploration-driven 5-HT drain this cycle.
    pub exploration_sht_drain: f32,

    // ── Liquid-Mamba Fusion Telemetry ────────────────────────────────
    /// Semantic prediction error from Liquid-Mamba round-trip (0.0–1.0, 0.0 when off).
    /// Measures `1 - cosine(thought_hv, bundled_output_hvs)`.
    #[cfg(feature = "liquid-mamba")]
    pub liquid_mamba_semantic_pe: f32,
    /// Bottleneck effective rank (1.0–256.0). Low = projection collapse.
    #[cfg(feature = "liquid-mamba")]
    pub liquid_mamba_effective_rank: f32,
    /// Current projection learning rate (after warmup + cosine annealing + FEP modulation).
    #[cfg(feature = "liquid-mamba")]
    pub liquid_mamba_lr: f32,
    /// Total distillation steps performed since startup.
    #[cfg(feature = "liquid-mamba")]
    pub liquid_mamba_generation_count: u32,

    /// Mesh network telemetry (health, peers, bandwidth, encryption).
    #[serde(flatten)]
    pub mesh: MeshTelemetry,

    // ── Feedback State Telemetry (Phase 2.2) ────────────────────────────
    /// Feedback proposal system telemetry: counts, consensus outcomes, traces.
    #[serde(flatten)]
    pub feedback: FeedbackTelemetry,

    // ── Staged Computation Model Telemetry (Phase 2.3) ────────────────
    /// Number of subsystems that contributed non-neutral SubsystemOutputs.
    /// 0 = no subsystems using the new CognitiveSubsystem trait yet.
    pub subsystem_integration_contributors: u32,
    /// OR'd output_flags from all subsystem managers (0 = no flags set).
    /// Bits: 0=REQUEST_EXPLORATION, 1=REQUEST_CONSOLIDATION, 2=ANOMALY_DETECTED,
    /// 3=VETO_ACTION, 4=REQUEST_REST, 7=ESCALATE_URGENCY.
    pub subsystem_flags: u32,
    /// Whether any subsystem set the VETO_ACTION flag this cycle.
    pub subsystem_veto_active: bool,

    // ── Nurture Attachment Telemetry ─────────────────────────────────────
    /// Current attachment style (e.g., "Forming", "Secure"). Empty when nurture disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attachment_style: Option<String>,
    /// Current attachment security score (0.0–1.0). None when nurture disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attachment_security: Option<f64>,

    // ── Partnership / Phi-Dyad Telemetry ──────────────────────────────
    /// Relational Phi (Φ_dyad) — consciousness OF the relationship.
    /// Computed from recent AI + input HVs via PhiDyadCalculator.
    /// 0.0 when partnership module inactive.
    pub relational_psi: f64,

    // ── Resonant Speech Output ────────────────────────────────────────
    /// Response profile label derived from cognitive load + consciousness level.
    /// One of "technical", "balanced", "simplified", "empathic".
    #[serde(default = "default_response_profile")]
    pub response_profile: String,

    /// Describes a substrate transition that occurred during this cycle (if any).
    /// Format: "SiliconDigital -> BiologicalNeurons (0.710 -> 0.920)"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub substrate_transition: Option<String>,

    /// Raw substrate feasibility before validation overlay (0.0–1.0).
    #[serde(default)]
    pub substrate_feasibility_raw: f64,
    /// Honest evidence confidence for current substrate (0.0–0.95).
    /// From SubstrateValidationFramework: biological=0.95, silicon=0.10, etc.
    #[serde(default)]
    pub substrate_honest_confidence: f64,
    /// Effective feasibility after validation overlay blending (0.0–1.0).
    /// Equals raw when validation overlay is disabled.
    #[serde(default)]
    pub substrate_effective_feasibility: f64,

    /// CfC tau factor from substrate speed modulation [0.5, 2.0].
    /// 1.0 when speed modulation is disabled.
    #[serde(default = "default_one_f32")]
    pub substrate_tau_factor: f32,
    /// Scale pressure: log10(substrate_max_scale / bio_max_scale).
    /// Telemetry-only. 0.0 when speed modulation is disabled.
    #[serde(default)]
    pub substrate_scale_pressure: f32,

    /// Whether the prediction model is currently in consolidation mode.
    /// Set in prediction.rs when replay-driven consolidation is active.
    #[serde(default)]
    pub is_consolidating: bool,

    // ── Voice Telemetry ─────────────────────────────────────────────────
    /// Smoothed articulation quality from voice feedback bridge (0.0–1.0).
    #[serde(default)]
    pub voice_articulation_quality: f32,
    /// Speech rate stability from voice feedback bridge (0.0–1.0).
    #[serde(default)]
    pub voice_rate_stability: f32,
    /// Overall voice confidence: articulation × 0.6 + stability × 0.4.
    #[serde(default)]
    pub voice_confidence: f32,
    /// Phi adjustment from voice quality (positive = understanding boost).
    #[serde(default)]
    pub voice_phi_adjustment: f32,

    // ── GWT Handler Telemetry ───────────────────────────────────────────
    /// Whether a GWT memory consolidation handler fired this cycle.
    #[serde(default)]
    pub gwt_memory_consolidation_requested: bool,
    /// Number of GWT perception broadcasts consumed by handlers this cycle.
    #[serde(default)]
    pub gwt_perception_broadcasts: u32,

    // ── Cantor Fractal Dream Telemetry ─────────────────────────────────
    /// Cantor CRHV broadcast buffer, dream consolidation, and codebook metrics.
    #[serde(default)]
    pub cantor: CantorTelemetry,

    // ── Social Coherence Telemetry ──────────────────────────────────────
    /// Current social trust level (0.0–1.0) from Mind module's SocialCoherence.
    #[serde(default)]
    pub social_trust_current: f32,
    /// Current social cooperation rate (0.0–1.0).
    #[serde(default)]
    pub social_cooperation_current: f32,
    /// Whether social trust biased strategy selection this cycle.
    #[serde(default)]
    pub social_strategy_bias_applied: bool,
    /// Social learning rate factor applied this cycle (0.8–1.2).
    #[serde(default = "default_one_f32")]
    pub social_learning_rate_factor: f32,
    /// Social prediction accuracy (ToM) — rolling mean of prediction vs outcome (0.0–1.0).
    #[serde(default = "default_half_f32")]
    pub social_prediction_accuracy: f32,
    /// Number of active mental models being tracked.
    #[serde(default)]
    pub social_models_count: usize,
    /// Mean trust across all tracked relationships (0.0–1.0).
    #[serde(default = "default_half_f32")]
    pub social_mean_trust: f32,
    /// ToM prediction mismatch EMA (1 - accuracy, smoothed).
    #[serde(default)]
    pub tom_prediction_mismatch: f32,
    /// Whether ToM mismatch triggered exploration this cycle.
    #[serde(default)]
    pub tom_exploration_triggered: bool,

    // ── Foveation Bridge Telemetry ──────────────────────────────────────
    /// Foveation bridge telemetry (None when foveation feature disabled or not active).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub foveation: Option<FoveationBridgeTelemetry>,

    // ── Physics Bridge Telemetry ────────────────────────────────────────
    /// Physics bridge telemetry (None when physics-bridge feature disabled or not active).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physics_bridge: Option<PhysicsBridgeTelemetry>,

    // ── Broca SSM Language Telemetry ────────────────────────────────────
    /// Broca SSM language generation telemetry (None when ssm_language feature disabled or not active).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub broca: Option<BrocaGenerationTelemetry>,

    // ── Broca Factcheck Telemetry (Mycelix knowledge graph verification) ─
    /// Factcheck bridge telemetry: accuracy EMA, claims verified/suppressed.
    /// None when `mycelix` feature disabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factcheck: Option<FactcheckTelemetry>,

    // ── Adaptive Dynamics Telemetry (Sessions 2-4) ───────────────────────
    /// Epistemic uncertainty: prediction disagreement across horizons (0.0–1.0).
    /// High = model uncertain (reducible by exploration).
    #[serde(default)]
    pub epistemic_uncertainty: f32,
    /// Aleatoric uncertainty: per-dimension prediction variance (0.0–1.0).
    /// High = inherent data noise (not reducible).
    #[serde(default)]
    pub aleatoric_uncertainty: f32,
    /// Theta oscillation phase (0–2π). Gates temporal binding strength.
    #[serde(default)]
    pub theta_phase: f32,
    /// Temporal binding strength after theta + salience gating (0.0–1.0).
    #[serde(default)]
    pub temporal_binding_strength: f32,
    /// PE-adaptive prediction horizon scale (0.58–1.30).
    /// <1.0 = contracted (high PE), >1.0 = expanded (low PE).
    #[serde(default = "default_one_f32")]
    pub prediction_horizon_scale: f32,
    /// FEP surprise → CfC tau modulation factor (0.8–1.0).
    #[serde(default = "default_one_f32")]
    pub fep_tau_factor: f32,
    /// SpectralMIP Phi → CfC tau modulation factor [PHI_TAU_FLOOR, PHI_TAU_CEILING].
    /// 1.0 when phi_tau_feedback is disabled or no Phi available yet.
    #[serde(default = "default_one_f32")]
    pub phi_tau_factor: f32,
    /// CalibrationValidator: total completed validations.
    #[serde(default)]
    pub calibration_validations_total: u32,
    /// CalibrationValidator: improvement count.
    #[serde(default)]
    pub calibration_improvements: u32,
    /// CalibrationValidator: regression count.
    #[serde(default)]
    pub calibration_regressions: u32,
    /// CalibrationValidator: adjustment damping multiplier (1.0 = no damping).
    #[serde(default = "default_one_f32")]
    pub calibration_adjustment_multiplier: f32,
    /// Causal edges incorporated into world model this cycle.
    #[serde(default)]
    pub causal_world_model_edges: usize,
    /// Epistemic uncertainty → attention budget scale factor (0.9–1.3).
    #[serde(default = "default_one_f32")]
    pub epistemic_budget_scale: f32,
    /// Feedback signals fired this cycle (sum of all proposal collectors).
    #[serde(default)]
    pub feedback_signals_fired: u32,
    /// Error trend slope over last 4 cycles (positive = worsening).
    #[serde(default)]
    pub error_slope: f32,
    /// Oscillation ratio of recent error history (0.0–1.0, high = unstable).
    #[serde(default)]
    pub oscillation_ratio: f32,
    /// Cumulative urgency mode transitions since start.
    #[serde(default)]
    pub mode_transitions: u32,
    /// Smoothed epistemic uncertainty (EMA, 0.0–1.0).
    #[serde(default)]
    pub smoothed_epistemic_uncertainty: f32,
    /// High-water mark of feedback signals in any single cycle.
    #[serde(default)]
    pub feedback_signals_high_water: u32,
    /// Number of feedback channels dampened this cycle (0–4).
    #[serde(default)]
    pub feedback_dampened_count: u32,
    /// Feedback signal diversity: unique sources / total proposals (0.0–1.0).
    #[serde(default = "default_one_f32")]
    pub feedback_signal_diversity: f32,
    /// Average PE cost of mode transitions (EMA, 0.0–1.0).
    #[serde(default)]
    pub avg_transition_cost: f32,
    /// Subsystem that contributed the most feedback proposals this cycle.
    #[serde(default)]
    pub feedback_dominant_source: String,
    /// Self-assessment adaptive cooldown duration (cycles).
    #[serde(default)]
    pub calibration_cooldown_duration: u32,

    // ── MCE Factor Telemetry ──────────────────────────────────────────
    /// MCE bottleneck: which factor is limiting consciousness (e.g., "Φ (Integration)").
    #[serde(default)]
    pub mce_bottleneck: String,
    /// MCE softmin: the bottleneck score before sigmoid (0.0–1.0).
    #[serde(default)]
    pub mce_softmin: f64,
    /// MCE weighted sum of all 10 components (0.0–1.0).
    #[serde(default)]
    pub mce_weighted_sum: f64,
    /// MCE narrative coherence factor N (0.0–1.0).
    #[serde(default)]
    pub mce_narrative: f64,
    /// MCE social embedding factor Soc (0.0–1.0).
    #[serde(default)]
    pub mce_social: f64,

    // ── Session 9: Advanced Feedback Intelligence ────────────────────────
    /// PE variance (E[X²]-E[X]²) used for confidence modulation.
    /// High variance = unstable errors = extra dampening.
    /// Science: Yu & Dayan (2005) — expected vs unexpected uncertainty.
    #[serde(default)]
    pub pe_variance: f32,
    /// Dominant source proposal concentration (fraction of total, 0.0–1.0).
    #[serde(default)]
    pub dominant_source_concentration: f32,
    /// Homeostasis pull efficiency: ratio of post/pre distance to target.
    /// <1.0 = pulls working, >1.0 = overcorrecting.
    /// Science: Cannon (1929) — homeostatic regulation monitoring.
    #[serde(default = "default_one_f32")]
    pub homeostasis_efficiency: f32,

    // ── Session 10: Adaptive Feedback Intelligence ────────────────────────
    /// Remaining LR freeze cycles from crash recovery (0 = not frozen).
    #[serde(default)]
    pub crash_freeze_remaining: u32,
    /// Number of distinct proposal sources this cycle (diversity metric).
    /// Science: Dehaene (2014) — healthy cognition requires multi-source consensus.
    #[serde(default)]
    pub proposal_source_count: u32,
    /// Current hysteresis relaxation factor (1.0 = full, decays with stability).
    /// Science: Kelso (1995) — mode stability permits relaxed boundaries.
    #[serde(default = "default_one_f32")]
    pub hysteresis_factor: f32,

    // ── Session 11: Fixes + Adaptive Intelligence ────────────────────────
    /// Proposal conflict ratio (0.0 = unanimous, 0.5 = maximally conflicted).
    /// Science: Dayan & Daw (2008) — model disagreement signals meta-uncertainty.
    #[serde(default)]
    pub proposal_conflict_ratio: f32,

    // ── Session 15: Feedback Loop Observability ─────────────────────────
    /// Moral salience easing applied to consolidation threshold (0.0 when inactive).
    #[serde(default)]
    pub moral_consolidation_ease: f32,
    /// Effective consolidation threshold (consciousness EMA - margin - moral ease).
    #[serde(default)]
    pub consolidation_threshold: f32,

    /// Feedback modulation observability flags (Sessions 9–16).
    /// Each boolean tracks whether a specific modulation pathway fired this cycle.
    /// Write-only for telemetry/dashboards — not read by internal logic.
    #[serde(flatten, default)]
    pub modulation: FeedbackModulationFlags,

    // ── Therapeutic Telemetry ─────────────────────────────────────────────
    /// Therapeutic subsystem telemetry (client state, alliance, crisis, regulation).
    #[cfg(feature = "therapeutic")]
    #[serde(flatten, default)]
    pub therapeutic: TherapeuticTelemetry,

    // ── Perception Manager Telemetry ────────────────────────────────────────
    /// Perception attention sensitivity [0.5, 2.0]. Modulates perceptual thresholds.
    /// Science: Yerkes & Dodson (1908) — optimal arousal modulates sensitivity.
    #[serde(default = "default_one_f32")]
    pub perception_attention_sensitivity: f32,
    /// Perception budget utilization (EMA) [0, 1]. Lavie (2005) perceptual load.
    #[serde(default)]
    pub perception_budget_utilization: f32,
    /// Whether perception is in vigilant mode (high attention, low coherence + high PE).
    #[serde(default)]
    pub perception_vigilant: bool,
    /// Mean perceptual coherence from rolling 8-cycle history [0, 1].
    /// Science: Damasio (1994) — cross-modal binding strength.
    #[serde(default = "default_half_f32")]
    pub perception_mean_coherence: f32,

    // ── Drive Manager Telemetry ───────────────────────────────────────────
    /// Drive boredom level [0, 0.8]. Sustained low PE → exploration urge.
    /// Science: Eastwood et al. (2012) — boredom as failed engagement.
    #[serde(default)]
    pub drive_boredom: f32,
    /// Drive flow intensity [0, 1]. Optimal challenge-skill balance.
    /// Science: Csikszentmihalyi (1990) — flow state intensity.
    #[serde(default)]
    pub drive_flow_intensity: f32,
    /// Whether currently in flow state (sustained low error + high coherence).
    #[serde(default)]
    pub drive_in_flow: bool,
    /// Adaptive exploration threshold [0.05, 0.8]. Surprise must exceed this.
    /// Science: Friston (2010) — precision-weighted surprise modulates exploration.
    #[serde(default)]
    pub drive_exploration_threshold: f32,

    // ── Learning Manager Telemetry ────────────────────────────────────────
    /// Learning plasticity level [0.1, 0.95]. How open the system is to learning.
    /// Science: Abraham & Bear (1996) — metaplasticity BCM rule.
    #[serde(default = "default_half_f32")]
    pub learning_plasticity: f32,
    /// Whether in dream consolidation phase (low arousal sustained ≥15 cycles).
    /// Science: Walker (2017) — NREM sleep facilitates memory integration.
    #[serde(default)]
    pub learning_in_dream_phase: bool,
    /// Learning error trend: positive = errors increasing, negative = improving.
    #[serde(default)]
    pub learning_error_trend: f32,

    // ── Memory Manager Telemetry ──────────────────────────────────────────
    /// Memory consolidation pressure [0, 1]. High = memory overload, needs dreaming.
    /// Science: Frankland & Bontempi (2005) — systems consolidation under pressure.
    #[serde(default)]
    pub memory_consolidation_pressure: f32,
    /// Mean retrieval quality [0, 1]. Signal reliability of memory recall.
    /// Science: Tulving (2002) — episodic retrieval quality.
    #[serde(default = "default_half_f32")]
    pub memory_recall_quality: f32,

    // ── Swarm Manager Telemetry ───────────────────────────────────────────
    /// Number of connected swarm peers.
    #[serde(default)]
    pub swarm_connected_peers: usize,
    /// Connectivity EMA [0, 1] — ratio of connected/expected peers.
    #[serde(default)]
    pub swarm_connectivity_ema: f32,
    /// Mean peer Φ across connected peers.
    #[serde(default)]
    pub swarm_mean_peer_phi: f32,
    /// Affective contagion strength this cycle.
    /// Science: Hatfield et al. (1993) — emotional contagion.
    #[serde(default)]
    pub swarm_affective_contagion: f32,
    /// Federated learning trust confidence.
    #[serde(default)]
    pub swarm_federated_confidence: f32,
    /// Number of network anomaly events (mass disconnects).
    #[serde(default)]
    pub swarm_anomaly_count: u32,

    // ── Governance Manager Telemetry ──────────────────────────────────────
    /// Governance reward EMA from outcome alignment (-1.0 to 1.0).
    /// Science: Woolley et al. (2010) — collective intelligence reward signal.
    #[serde(default)]
    pub governance_reward_ema: f32,
    /// Number of pending governance events queued for processing.
    #[serde(default)]
    pub governance_pending_events: usize,
    /// Number of pending governance outcomes awaiting learning.
    #[serde(default)]
    pub governance_pending_outcomes: usize,
    /// Collective Phi from latest governance tally.
    #[serde(default)]
    pub governance_collective_phi: f32,
    /// Accumulated governance confidence nudge this cycle.
    #[serde(default)]
    pub governance_confidence_delta: f32,
    /// Community mode from collective identity (e.g., "Deliberative").
    #[serde(default)]
    pub governance_community_mode: String,
    /// Number of epistemic blind spots detected.
    #[serde(default)]
    pub governance_blind_spot_count: usize,
    /// Severity of worst blind spot [0, 1].
    #[serde(default)]
    pub governance_max_blind_spot_severity: f32,
    /// Number of epistemic agents tracked.
    #[serde(default)]
    pub governance_epistemic_agents: usize,
    /// Maximum absolute harmonic delta from governance feedback.
    #[serde(default)]
    pub governance_harmonic_delta_max: f32,
    /// Learning rate boost from governance prediction error.
    #[serde(default)]
    pub governance_lr_boost: f32,

    // ── Fabrication Manager Telemetry ─────────────────────────────────────
    /// Manufacturing free energy from ManufacturingTwin (0.0 = equilibrium).
    #[serde(default)]
    pub fabrication_manufacturing_fe: f64,
    /// Design loop free energy from DesignLoopTwin.
    #[serde(default)]
    pub fabrication_design_loop_fe: f64,
    /// Manufacturing safety level as string ("Green"/"Yellow"/"Orange"/"Red").
    #[serde(default)]
    pub fabrication_safety_level: String,
    /// Cincinnati anomaly count this cycle.
    #[serde(default)]
    pub fabrication_anomaly_count: u32,
    /// EMA of anomaly severity [0, 1].
    #[serde(default)]
    pub fabrication_anomaly_ema: f32,
    /// PoGF score EMA [0, 1].
    #[serde(default)]
    pub fabrication_pog_score_ema: f32,
    /// Active print jobs.
    #[serde(default)]
    pub fabrication_active_jobs: u32,
    /// Fabrication reward EMA.
    #[serde(default)]
    pub fabrication_reward_ema: f32,
    /// Mean prediction coherence across manufacturing horizons.
    #[serde(default)]
    pub fabrication_prediction_coherence: f32,
    /// MRP planned orders count.
    #[serde(default)]
    pub fabrication_mrp_planned_orders: u32,
    /// MRP feasibility (true = no shortages).
    #[serde(default)]
    pub fabrication_mrp_feasible: bool,
    /// MRP material shortage count.
    #[serde(default)]
    pub fabrication_mrp_shortages: u32,
    /// MRP work orders in scope.
    #[serde(default)]
    pub fabrication_mrp_work_orders: u32,
    /// Defect prediction quality score [0, 1].
    #[serde(default)]
    pub fabrication_defect_prediction: f32,
    /// Defect prediction confidence [0, 1].
    #[serde(default)]
    pub fabrication_defect_confidence: f32,

    // ── CPG Manager Telemetry ─────────────────────────────────────────────
    /// Kuramoto synchronization index [0, 1]. 0 = incoherent, 1 = perfect sync.
    #[serde(default)]
    pub cpg_sync_index: f32,
    /// Mean oscillator frequency (Hz).
    #[serde(default)]
    pub cpg_mean_freq: f32,
    /// Whether motor output is active.
    #[serde(default)]
    pub cpg_motor_active: bool,
    /// Whether desynchronization alert was triggered.
    #[serde(default)]
    pub cpg_desync_alert: bool,

    // ── Embodiment Bridge Telemetry ──────────────────────────────────────
    /// Total embodiment steps executed.
    #[serde(default)]
    pub embodiment_total_steps: u64,
    /// Control effort from the most recent embodiment step.
    #[serde(default)]
    pub embodiment_control_effort: f32,
    /// Prediction error from the most recent embodiment step.
    #[serde(default)]
    pub embodiment_prediction_error: f32,
    /// Active embodiment platform name (empty if disembodied).
    #[serde(default)]
    pub embodiment_platform: String,
    /// Number of actuators in the active embodiment platform.
    #[serde(default)]
    pub embodiment_num_actuators: u32,

    // ── Radio/Spectrum Manager Telemetry ───────────────────────────────────
    /// Network health: 0=AllUp, 1=LocalDown, 2=MetroOnly, 3=Blackout.
    #[serde(default)]
    pub spectrum_network_health: u8,
    /// Per-tier availability [Local, Metro, Regional] as bitmask.
    #[serde(default)]
    pub spectrum_tier_available: u8,
    /// Consecutive jamming cycles.
    #[serde(default)]
    pub spectrum_jamming_streak: u32,
    /// Spectrum prediction error [0, 1].
    #[serde(default)]
    pub spectrum_prediction_error: f32,
    /// Epistemic discount from network degradation.
    #[serde(default)]
    pub spectrum_epistemic_discount: f32,
    /// Number of degradation streak cycles.
    #[serde(default)]
    pub spectrum_degradation_streak: u32,
    /// Known mesh route table size.
    #[serde(default)]
    pub spectrum_known_peers: usize,
    /// Active encryption sessions.
    #[serde(default)]
    pub spectrum_encryption_sessions: usize,

    // ── Finance Health Telemetry ─────────────────────────────────────────
    /// Number of active collateral positions in the connected Mycelix network.
    #[serde(default)]
    pub finance_active_positions: u32,
    /// Number of positions in Warning or worse LTV status.
    #[serde(default)]
    pub finance_stressed_positions: u32,
    /// Number of positions in MarginCall or Liquidation status.
    #[serde(default)]
    pub finance_critical_positions: u32,
    /// Average LTV ratio across all active positions (0.0-1.0+).
    #[serde(default)]
    pub finance_avg_ltv: f32,
    /// Total SAP in circulation (micro-SAP).
    #[serde(default)]
    pub finance_sap_circulation: u64,
    /// Total compost collected this period (micro-SAP, demurrage redistributed).
    #[serde(default)]
    pub finance_compost_collected: u64,
    /// Number of active covenants.
    #[serde(default)]
    pub finance_active_covenants: u32,
    /// Circuit breaker status: number of open breakers.
    #[serde(default)]
    pub finance_open_breakers: u32,
    /// Oracle consensus confidence (0.0-1.0, signal_integrity from price oracle).
    #[serde(default)]
    pub finance_oracle_confidence: f32,
    /// Financial stress index (0.0-1.0, computed from stressed/total positions ratio).
    /// Science: Borio (2014) — financial stress as systemic risk indicator.
    #[serde(default)]
    pub finance_stress_index: f32,

    // ── Causal Explanation Narrative Telemetry ─────────────────────────────
    /// Causal self-explanation narrative summary (generated every 47 cycles).
    /// Science: Wierzbicka (1996) — NSM-grounded causal transparency.
    #[serde(default)]
    pub consciousness_causal_narrative: String,

    // ── Knowledge Engine Telemetry ─────────────────────────────────────────
    /// Total facts stored in the knowledge graph (0 when knowledge engine disabled).
    #[serde(default)]
    pub knowledge_graph_size: u32,
    /// Best cosine similarity from the most recent knowledge search (0.0 when disabled).
    #[serde(default)]
    pub knowledge_best_similarity: f32,
    /// Total causal edges in the knowledge causal bridge (0 when disabled).
    #[serde(default)]
    pub knowledge_causal_edges: u32,
    /// Epistemic surprise signal: novelty + contradiction (0.0 when disabled).
    /// Science: Friston (2010) — epistemic surprise drives active inference.
    #[serde(default)]
    pub knowledge_epistemic_surprise: f64,
    /// Expected Calibration Error (0.0 = perfect, 1.0 = worst, 0.0 when disabled).
    /// Science: Guo et al. (2017) — calibration of confidence scores.
    #[serde(default)]
    pub knowledge_calibration_ece: f64,
    /// Number of contradictions detected this cycle (0 when disabled).
    #[serde(default)]
    pub knowledge_contradictions: u32,

    // ── Glyph Codex Telemetry ─────────────────────────────────────────────
    /// Dominant glyph field modality name (e.g., "Resonant", "Threshold").
    /// Empty when glyph_codex feature is disabled.
    #[serde(default)]
    pub glyph_dominant_modality: String,
    /// Glyph coherence score (0.0–0.95). Measures symbolic integration
    /// across all 11 Field Modalities. 0.0 when glyph_codex is disabled.
    #[serde(default)]
    pub glyph_coherence: f32,
    /// Name of the nearest resonant glyph (e.g., "Ethical Emergence").
    /// Empty when below quiet threshold or glyph_codex is disabled.
    #[serde(default)]
    pub glyph_resonant_name: String,
    /// Spiral position (0.0–56.0) tracking developmental progression.
    /// 0.0 when glyph_codex is disabled.
    #[serde(default)]
    pub glyph_spiral_position: f32,

    // ── Feature Availability Flags ──────────────────────────────────────
    // These booleans tell dashboards whether 0.0 means "feature disabled"
    // vs "measured as zero". Populated at metadata construction time from
    // compile-time feature flags and runtime config.
    /// Whether the `reasoning_engine` feature is compiled in and active.
    #[serde(default)]
    pub reasoning_engine_enabled: bool,
    /// Whether the `mesh` feature is compiled in and active.
    #[serde(default)]
    pub mesh_enabled: bool,
    /// Whether the `ssm_language` (Broca) feature is compiled in and active.
    #[serde(default)]
    pub ssm_language_enabled: bool,
    /// Whether the `vision-manifold` feature is compiled in.
    #[serde(default)]
    pub vision_manifold_enabled: bool,

    // ── Immune System / Defense ──────────────────────────────────────────
    /// Current safety level label ("GREEN", "YELLOW", "ORANGE", "RED").
    #[serde(default)]
    pub immune_safety_level: String,
    /// Guardian posture label ("Normal", "Cautious", "Defensive", "Emergency", "Hold").
    #[serde(default)]
    pub immune_guardian_posture: String,
    /// Whether guardian patrol is active.
    #[serde(default)]
    pub immune_patrol_active: bool,
    /// Number of active sentinel threat signals.
    #[serde(default)]
    pub immune_active_threats: u32,
    /// Highest threat severity this cycle (0.0–1.0).
    #[serde(default)]
    pub immune_max_severity: f32,
    /// Aggregate threat level (0.0–1.0).
    #[serde(default)]
    pub immune_threat_level: f32,
    /// Number of quarantined peers.
    #[serde(default)]
    pub immune_quarantined_peers: u32,
    /// Stored threat patterns in immune memory.
    #[serde(default)]
    pub immune_threat_patterns: u32,
    /// LR multiplier from safety enforcement (1.0 = no gate).
    #[serde(default = "default_one_f32")]
    pub immune_lr_multiplier: f32,
    /// Exploration multiplier from safety enforcement (1.0 = no gate).
    #[serde(default = "default_one_f32")]
    pub immune_exploration_multiplier: f32,
    /// Whether motor output is halted by safety.
    #[serde(default)]
    pub immune_motor_halt: bool,
    /// Whether collective immune response is active across swarm.
    #[serde(default)]
    pub immune_response_active: bool,
    /// Cumulative emergency posture cycles.
    #[serde(default)]
    pub immune_emergency_cycles: u64,

    // ── Defense Cascade ──────────────────────────────────────────────────
    /// Number of defense actions proposed this cycle.
    #[serde(default)]
    pub defense_actions_proposed: u32,
    /// Number of defense actions that passed the moral filter.
    #[serde(default)]
    pub defense_actions_approved: u32,

    // ── Sovereign Inoculation Telemetry ───────────────────────────────
    // Clock
    /// Mesh-time consensus offset from local clock (µs).
    #[serde(default)]
    pub sovereign_time_offset_us: i64,
    /// Mesh-time stratum level (0=GPS, 15=unsync).
    #[serde(default)]
    pub sovereign_time_stratum: u8,
    /// Mesh-time drift estimation (ppm).
    #[serde(default)]
    pub sovereign_time_drift_ppm: f32,
    /// Mesh-time peer count contributing to consensus.
    #[serde(default)]
    pub sovereign_time_peer_count: usize,
    /// Mesh-time quality (0=Authoritative, 1=Consensus, 2=Degraded, 3=FreeRunning).
    #[serde(default)]
    pub sovereign_time_quality: u8,

    // Trust
    /// Average trust across web-of-trust graph edges.
    #[serde(default)]
    pub sovereign_trust_avg: f32,
    /// Trust graph density (edges / max possible).
    #[serde(default)]
    pub sovereign_trust_density: f32,
    /// Sybil anomaly count detected.
    #[serde(default)]
    pub sovereign_trust_anomalies: u32,
    /// Fraction of trust edges with post-quantum verification.
    #[serde(default)]
    pub sovereign_trust_pq_fraction: f32,

    // Social Fabric
    /// Mean resonance across tracked peers.
    #[serde(default)]
    pub sovereign_social_resonance_mean: f32,
    /// Content diversity metric (0=echo chamber, 1=maximally diverse).
    #[serde(default)]
    pub sovereign_social_diversity: f32,
    /// Echo chamber risk score (0–1, >0.85 = warning).
    #[serde(default)]
    pub sovereign_social_echo_risk: f32,
    /// Number of unique content peers.
    #[serde(default)]
    pub sovereign_social_peer_reach: usize,

    // Survival
    /// Water availability fraction (0.0–1.0).
    #[serde(default)]
    pub sovereign_survival_water_pct: f32,
    /// Current power consumption (kW).
    #[serde(default)]
    pub sovereign_survival_power_kw: f32,
    /// Whether a survival emergency is active.
    #[serde(default)]
    pub sovereign_survival_emergency: bool,
    /// Number of active IoT sensors.
    #[serde(default)]
    pub sovereign_survival_sensor_count: usize,
    /// Number of active resource alerts.
    #[serde(default)]
    pub sovereign_survival_alert_count: usize,

    // ── Math Service Telemetry ──────────────────────────────────────────
    /// Total math problems solved by the math service since startup.
    #[serde(default)]
    pub math_problems_solved: usize,
    /// Multi-path verification rate (0.0–1.0): fraction of solved problems
    /// that were independently verified by multiple solver paths.
    #[serde(default)]
    pub math_verification_rate: f64,
    /// Average confidence across all math solutions this session (0.0–1.0).
    #[serde(default)]
    pub math_avg_confidence: f64,

    // ── FHE Collective Wisdom ─────────────────────────────────────────
    /// Total encrypted contributions made to the collective wisdom pool.
    #[serde(default)]
    pub fhe_contributions_total: usize,
    /// Total aggregations completed this session.
    #[serde(default)]
    pub fhe_aggregations_total: usize,
    /// Current pool size (pending contributions).
    #[serde(default)]
    pub fhe_pool_count: usize,
    /// Cycles since last aggregation.
    #[serde(default)]
    pub fhe_cycles_since_aggregation: usize,

    // ── Scientific Method Telemetry ──────────────────────────────────────
    /// Number of active hypotheses in the scientific method engine (0 when disabled).
    #[cfg(feature = "scientific_method")]
    #[serde(default)]
    pub scientific_hypotheses_active: usize,
    /// Total experiments run by the scientific method engine (0 when disabled).
    #[cfg(feature = "scientific_method")]
    #[serde(default)]
    pub scientific_experiments_run: usize,
    /// Average prediction accuracy (match_score) across experiments (0.0 when disabled).
    #[cfg(feature = "scientific_method")]
    #[serde(default)]
    pub scientific_avg_prediction_accuracy: f64,

    // ── Mathematics Phase 7 Telemetry ──────────────────────────────────
    /// Count of multi-method Phi-ranked solutions produced this cycle (Phase 7a).
    #[cfg(feature = "mathematics")]
    #[serde(default)]
    pub math_phi_ranked_solutions: usize,
    /// Count of uncertain epistemic results this cycle (Phase 7b).
    #[cfg(feature = "mathematics")]
    #[serde(default)]
    pub math_epistemic_uncertain_count: usize,
    /// Count of successful memory recalls this cycle (Phase 7c).
    #[cfg(feature = "mathematics")]
    #[serde(default)]
    pub math_memory_hits: usize,

    // ── Vision Manager Telemetry ──────────────────────────────────────────
    /// Visual prediction error EMA (0.0-1.0). Smoothed visual surprise signal.
    /// Science: Itti & Koch (2001) — saliency-driven attention.
    #[serde(default)]
    pub vision_pe_ema: f32,
    /// Adaptive visual surprise threshold (0.05-0.8). Habituates upward.
    /// Science: Rankin et al. (2009) — habituation dynamics.
    #[serde(default)]
    pub vision_surprise_threshold: f32,
    /// Consecutive low-surprise cycles (habituation streak).
    #[serde(default)]
    pub vision_low_surprise_streak: u32,

    // ── Language Manager Telemetry ────────────────────────────────────────
    /// Broca generation quality EMA (0.0-1.0). Smoothed epistemic confidence.
    /// Science: Clark (2013) — predictive processing quality signal.
    #[serde(default)]
    pub language_quality_ema: f32,
    /// Language coherence EMA (0.0-1.0). Smoothed conversation coherence.
    /// Science: Hagoort (2005) — unification model of language.
    #[serde(default)]
    pub language_coherence_ema: f32,
    /// Consecutive low-coherence cycles (fluency degradation indicator).
    #[serde(default)]
    pub language_low_coherence_streak: u32,

    // ── Reasoning Manager Telemetry ──────────────────────────────────────
    /// Reasoning reliability EMA (0.0-1.0). Smoothed prediction confidence.
    /// Science: Stanovich (2011) — individual differences in rational thinking.
    #[serde(default)]
    pub reasoning_reliability_ema: f64,
    /// Cumulative reasoning quality signal (decayed).
    /// Science: Koriat (2007) — metacognitive monitoring.
    #[serde(default)]
    pub reasoning_cumulative_quality: f64,
    /// Consecutive rising confidence cycles.
    #[serde(default)]
    pub reasoning_rising_streak: u32,
    /// Consecutive falling confidence cycles.
    #[serde(default)]
    pub reasoning_falling_streak: u32,

    // ── Neural Validation Telemetry ─────────────────────────────────────
    /// Per-region cortical activation map (12 regions, 0.0–1.0).
    /// Populated from live subsystem states when `neural_validation` feature enabled.
    /// Used for comparison against TRIBE v2 fMRI predictions.
    #[cfg(feature = "neural_validation")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cortical_activation: Option<symthaea_core::hdc::cortical_activation::CorticalActivationMap>,
}

/// Therapeutic subsystem telemetry for CycleMetadata.
///
/// Tracks client state, alliance, crisis detection, regulation strategy,
/// narrative coherence, and case formulation — all zero/false when the
/// `therapeutic` feature is disabled.
#[cfg(feature = "therapeutic")]
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TherapeuticTelemetry {
    /// Client distress level (0.0–1.0, RDoC-derived).
    #[serde(default)]
    pub therapeutic_client_distress: f32,
    /// Therapeutic alliance composite (0.0–1.0, Bordin bond/goals/tasks).
    #[serde(default)]
    pub therapeutic_alliance: f32,
    /// Whether a crisis was detected this cycle.
    #[serde(default)]
    pub therapeutic_crisis_active: bool,
    /// Crisis type name (empty if no crisis).
    #[serde(default)]
    pub therapeutic_crisis_type: String,
    /// Active regulation strategy name (empty if none).
    #[serde(default)]
    pub therapeutic_strategy: String,
    /// Narrative coherence (0.0–1.0, fragment integration quality).
    #[serde(default)]
    pub therapeutic_narrative_coherence: f32,
    /// Case formulation factor count (predisposing + precipitating + perpetuating + protective).
    #[serde(default)]
    pub therapeutic_formulation_factors: usize,
    /// Case formulation resilience ratio (protective / total, 0.0–1.0).
    #[serde(default)]
    pub therapeutic_resilience_ratio: f32,
    /// Alliance rupture count (cumulative).
    #[serde(default)]
    pub therapeutic_rupture_count: u32,
    /// Alliance repair count (cumulative).
    #[serde(default)]
    pub therapeutic_repair_count: u32,
    /// RDoC clinical severity composite (0.0–1.0).
    #[serde(default)]
    pub therapeutic_clinical_severity: f32,
    /// Number of narrative fragments recorded.
    #[serde(default)]
    pub therapeutic_narrative_fragments: usize,
    /// Serotonin debt from sustained negative valence (0.0–1.0, Jans et al. 2007).
    #[serde(default)]
    pub therapeutic_serotonin_debt: f32,
    /// Dopamine debt from sustained low positive valence (0.0–1.0).
    #[serde(default)]
    pub therapeutic_dopamine_debt: f32,
    /// Dream prediction accuracy (lower = better, 1.0 = no data).
    #[serde(default)]
    pub therapeutic_dream_accuracy: f32,
    /// Scope violation detected in Broca output this cycle (empty = none).
    #[serde(default)]
    pub therapeutic_scope_violation: String,
    /// Last rupture type ("Withdrawal", "Confrontation", or empty).
    #[serde(default)]
    pub therapeutic_last_rupture_type: String,
    /// Repair success rate (0.0–1.0, 1.0 = no ruptures or all repaired).
    #[serde(default = "default_one_f32")]
    pub therapeutic_repair_rate: f32,
    /// Withdrawal rupture count (cumulative).
    #[serde(default)]
    pub therapeutic_withdrawal_count: u32,
    /// Confrontation rupture count (cumulative).
    #[serde(default)]
    pub therapeutic_confrontation_count: u32,
    /// RDoC 6-domain profile [NegVal, PosVal, Cognitive, Social, Arousal, Sensorimotor].
    #[serde(default)]
    pub therapeutic_rdoc_profile: [f32; 6],
    /// Perpetuating factor descriptions from case formulation.
    #[serde(default)]
    pub therapeutic_perpetuating_factors: Vec<String>,
    /// Protective factor descriptions from case formulation.
    #[serde(default)]
    pub therapeutic_protective_factors: Vec<String>,
    /// Strategy effectiveness: Vec of (name, mean_effectiveness, use_count).
    #[serde(default)]
    pub therapeutic_strategy_effectiveness: Vec<(String, f32, u32)>,
    /// Narrative temporal coherence (Adler 2012) [0.0–1.0].
    #[serde(default)]
    pub therapeutic_temporal_coherence: f32,

    // ── Shadow work telemetry (Observability Mode — Jung → Friston) ──
    /// Total shadow pressure across all fragments (cumulative PE × recurrence × valence).
    #[serde(default)]
    pub shadow_total_pressure: f32,
    /// Number of active shadow fragments.
    #[serde(default)]
    pub shadow_fragment_count: u32,
    /// Highest individual fragment pressure (the "loudest" shadow).
    #[serde(default)]
    pub shadow_peak_pressure: f32,
    /// Mean prediction error for shadow-related content.
    #[serde(default)]
    pub shadow_mean_prediction_error: f32,
    /// Number of possible projection events detected this cycle.
    #[serde(default)]
    pub shadow_projection_detections: u32,
    /// Whether shadow pressure exceeds the surfacing threshold (diagnostic only).
    #[serde(default)]
    pub shadow_surfacing_indicated: bool,
    /// Number of shadow fragments queued for dream processing.
    #[serde(default)]
    pub shadow_dream_queue_depth: u32,
    /// Best Phi improvement from dream-processed shadow content.
    #[serde(default)]
    pub shadow_best_dream_phi: f32,
    /// Shadow pressure trend: positive = accumulating, negative = integrating/decaying.
    #[serde(default)]
    pub shadow_pressure_trend: f32,
    /// Ratio of shadow content to total narrative content.
    #[serde(default)]
    pub shadow_to_narrative_ratio: f32,
}

fn default_response_profile() -> String {
    "balanced".to_string()
}

fn default_one_f32() -> f32 {
    1.0
}

fn default_half_f32() -> f32 {
    0.5
}

/// Foveation bridge telemetry snapshot for CycleMetadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FoveationBridgeTelemetry {
    /// Number of pending foveation tasks.
    pub pending_count: usize,
    /// Number of in-flight foveation tasks.
    pub in_flight_count: usize,
    /// Number of ready (completed) foveation results.
    pub ready_count: usize,
    /// Total tasks dispatched since startup.
    pub total_dispatched: u64,
    /// Total tasks completed since startup.
    pub total_completed: u64,
    /// Average processing time in microseconds.
    pub avg_processing_time_us: u64,
    /// Confidence of the most recent recognition result.
    pub last_confidence: f32,
    /// Effective surprise threshold (after NE modulation).
    pub effective_surprise_threshold: f32,
    /// Effective max concurrent tasks (after DA modulation).
    pub effective_max_concurrent: usize,
    /// Number of recognition results this cycle.
    pub recognition_count: usize,
    /// Highest recognition confidence this cycle (0.0 if none).
    pub top_recognition_confidence: f32,
    /// Whether HV binding was applied (foveation results blended into cognitive HV).
    pub hv_binding_applied: bool,
    /// Whether dynamics coupling was triggered (exploration/confidence/LR modulation).
    pub dynamics_coupling_triggered: bool,
}

/// Physics bridge telemetry snapshot for CycleMetadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhysicsBridgeTelemetry {
    /// Number of physics entries in catalog.
    pub catalog_size: usize,
    /// Number of results returned from last query.
    pub results_returned: usize,
    /// Top matching entry name (empty if no match).
    pub top_match: String,
    /// Top match score (0.0 if no match).
    pub top_score: f32,
    /// Total queries performed since startup.
    pub query_count: u64,
    /// Whether a query was performed this cycle.
    pub queried_this_cycle: bool,
    /// Effective query interval after substrate tau modulation.
    pub effective_interval: usize,
    /// Effective blend weight after substrate tau + scale pressure modulation.
    pub effective_blend_weight: f32,
    /// Top domain from last query results (e.g. "Thermodynamics").
    pub top_domain: String,
    /// Pareto frontier context, if injected by GuidedDesignExplorer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pareto_frontier_size: Option<usize>,
    /// Best analogy score from Pareto frontier (0.0–1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pareto_best_analogy: Option<f32>,
}

/// Broca SSM language generation telemetry snapshot for CycleMetadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BrocaGenerationTelemetry {
    /// Whether generation was attempted this cycle.
    pub generated: bool,
    /// Number of tokens produced.
    pub token_count: usize,
    /// Final short-window coherence score from the generator.
    pub final_coherence: f32,
    /// Whether a semantic veto was triggered during generation.
    pub veto_triggered: bool,
    /// Generation wall-clock time in microseconds.
    pub generation_time_us: u64,
    /// Whether generation was skipped due to low consciousness.
    pub consciousness_gated: bool,
    /// Whether generation was skipped due to `EthicalVerdict::Blocked`.
    #[serde(default)]
    pub ethics_gated: bool,
    /// Composite generation quality (0.0–1.0).
    #[serde(default)]
    pub quality: f32,
    /// Long-window coherence from Mamba temporal context.
    #[serde(default)]
    pub long_coherence: f32,
    /// Semantic prediction error (reconstruction accuracy).
    #[serde(default)]
    pub semantic_pe: f32,
    /// Type-token ratio: unique_tokens / total_tokens (0.0–1.0, higher = more diverse).
    #[serde(default)]
    pub type_token_ratio: f32,
    /// Maximum consecutive repetitions of a single token (lower = better).
    #[serde(default)]
    pub max_repetition: usize,
    /// Number of NSM semantic primitives detected in the input for this generation.
    /// Science: Wierzbicka (1996) — universal semantic primes ground language production.
    #[serde(default)]
    pub nsm_primitive_count: usize,
    /// NSM primitive grounding score (0.0–1.0): fraction of input words that
    /// mapped to recognized NSM semantic primes. Higher = better semantic decomposition.
    #[serde(default)]
    pub nsm_grounding: f32,
    /// NSM prime coverage: fraction of active primes expressed by generated tokens (0.0–1.0).
    /// Higher = the generated text semantically covered the intended meaning.
    /// Science: Grice (1975) — cooperative principle; semantic coverage = communicative success.
    #[serde(default)]
    pub nsm_prime_coverage: f32,
}

/// Broca→Mycelix factcheck bridge telemetry snapshot.
///
/// Tracks verification accuracy, claims submitted/verified/suppressed,
/// and current modulation state. Populated when `mycelix` feature enabled.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FactcheckTelemetry {
    /// Running accuracy EMA of Broca's verifiable claims (0.0–1.0).
    pub accuracy_ema: f32,
    /// Total claims submitted for verification (lifetime).
    pub total_claims_submitted: u64,
    /// Total claims verified as True (lifetime).
    pub total_claims_verified: u64,
    /// Total claims suppressed due to False verdict (lifetime).
    pub total_claims_suppressed: u64,
    /// Claims checked this cycle.
    pub claims_this_cycle: u32,
    /// Whether output was suppressed this cycle.
    pub suppressed_this_cycle: bool,
    /// Current cadence penalty being applied.
    pub cadence_penalty: f32,
    /// Pending claims awaiting verification.
    pub pending_count: usize,
}

/// Memory-resonator subsystem telemetry: dreams, codebook, replay.
///
/// Grouped from CycleMetadata flat fields. `#[serde(flatten)]` preserves
/// the original JSON format for backwards compatibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryResonatorMetrics {
    /// Number of insights gained from dream replay this cycle (0 = no dreaming).
    pub dream_insights: usize,
    /// Best Phi improvement discovered by dream counterfactuals (0.0 = no improvement).
    pub dream_phi_improvement: f32,
    /// Total accumulated wisdom entries from dreaming.
    pub dream_wisdom_count: usize,
    /// Whether a continuity gap triggered demand replay this cycle.
    pub continuity_replay_triggered: bool,
    /// Number of symbols in the resonator semantic codebook (0 when disabled).
    pub resonator_codebook_size: usize,
    /// Number of episodes stored in resonator memory (0 when disabled).
    pub resonator_episodes: usize,
    /// Number of iterations used in last resonator factorization (0 when not run).
    pub resonator_factorization_iters: usize,
    /// Whether resonator recall primed working memory confidence this cycle.
    pub resonator_wm_primed: bool,
    /// Number of episodes reconsolidated via resonator recall this cycle.
    pub resonator_reconsolidated: usize,
    /// Number of high-Phi episodes promoted to resonator codebook this cycle.
    pub resonator_promotions: usize,
    /// Best cosine similarity from resonator recall (0.0 when no matches).
    pub resonator_best_sim: f32,
    /// Number of codebook entries evicted this cycle (redundancy pruning).
    pub codebook_evictions: usize,
    /// Codebook diversity: average pairwise cosine distance (0.0–1.0).
    pub codebook_diversity: f32,
    /// Resonator prediction error: cosine distance between last cycle's best match
    /// and this cycle's compressed state (0.0 = perfect prediction, 1.0 = orthogonal).
    pub resonator_prediction_error: f32,
    /// Codebook utilization rate (fraction of symbols retrieved recently, 0.0–1.0).
    pub codebook_utilization_rate: f32,
    /// FEP surprise-modulated replay batch size (0 when replay not triggered).
    pub surprise_replay_batch_size: usize,
}

/// Quality diagnostics telemetry: meta-cognitive, dissipative, coherence, anomaly recovery.
///
/// Grouped from CycleMetadata flat fields. `#[serde(flatten)]` preserves
/// the original JSON format for backwards compatibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityDiagnostics {
    /// Meta-cognitive self-model accuracy (0.0 = uncertain, 1.0 = perfect self-knowledge).
    pub meta_cognitive_accuracy: f32,
    /// Meta-cognitive recursion depth (0 = off, 1 = basic, 2+ = recursive self-modeling).
    pub meta_cognitive_depth: u8,
    /// Dissipative consciousness health score (0.0–1.0, 0.0 when off).
    pub dissipative_health: f64,
    /// Current thermodynamic regime (e.g., "Subcritical", "Critical", "Supercritical").
    pub dissipative_regime: String,
    /// Dissipative entropy production rate (0.0 when off).
    pub dissipative_entropy_rate: f64,
    /// Φ_eff = Φ × R^γ from epistemic conflict reliability weighting (0.0 when off).
    pub epistemic_phi_eff: f64,
    /// Consciousness Equation v2 C(t) result (0.0 when off).
    pub equation_v2_consciousness: f64,
    /// Hierarchical LTC estimated Phi (0.0 when off).
    pub hierarchical_ltc_phi: f32,
    /// Unified quality score (fusion of prediction coherence + agreement + anomaly).
    pub unified_quality_score: f32,
    /// Whether dissipative health gate dampened learning this cycle.
    pub dissipative_health_gated: bool,
    /// Dissipative health gate dampening factor applied (1.0 = no dampening).
    pub dissipative_lr_factor: f32,
    /// Coherence velocity (rate of change, negative = dropping).
    pub coherence_velocity: f32,
    /// Whether temporal discontinuity triggered coherence gating this cycle.
    pub coherence_velocity_gated: bool,
    /// Metacognitive anomaly recovery progress (0.0–1.0, 1.0 = fully recovered).
    pub anomaly_recovery_progress: f32,
    /// Whether anomaly recovery is actively in progress.
    pub anomaly_recovering: bool,
    /// HFE learning rate boost applied when hierarchical free energy exceeds threshold (1.0 = no boost).
    /// Friston (2008) — poor model → learn harder, capped at +10%.
    #[serde(default = "default_one_f32")]
    pub hierarchical_free_energy_lr_boost: f32,
    /// Predictive phi modulation learning rate delta (±1.5% max, coherence-weighted).
    /// Friston (2010) — precision-weighted plasticity gating.
    #[serde(default)]
    pub predictive_phi_lr_delta: f32,
    /// Confidence delta from body valence somatic marker feedback.
    /// Damasio (1999) — positive somatic state boosts coherence, negative dampens.
    #[serde(default)]
    pub body_valence_confidence_delta: f32,
    /// Confidence scale factor from narrative self-Phi (1.02 strong, 0.95 weak, 1.0 neutral).
    /// Gallagher (2000) — strong narrative identity stabilizes learning.
    #[serde(default = "default_one_f32")]
    pub narrative_self_confidence_factor: f32,
}

/// Attention subsystem telemetry: schema focus, GWT, budget, memoization.
///
/// Grouped from CycleMetadata flat fields. `#[serde(flatten)]` preserves
/// the original JSON format for backwards compatibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttentionMetrics {
    /// Attention schema focus intensity (0.0 to 1.0, 0.0 when not enabled).
    pub attention_schema_focus: f32,
    /// Vigilance fatigue level (0.0 = fresh, 1.0 = fully fatigued after ~60 cycles).
    /// Mackworth (1948) vigilance decrement. Resets on attention shift.
    pub attention_fatigue: f32,
    /// Attention schema prediction accuracy (0.0 to 1.0).
    /// Tracks how often the schema correctly predicted shift vs. maintenance.
    pub attention_prediction_accuracy: f32,
    /// Whether a GWT broadcast occurred this cycle.
    pub gwt_broadcast: bool,
    /// GWT winning coalition size (0 if no broadcast).
    pub gwt_coalition_size: u32,
    /// Rolling average of Phi observations from phi_attention (0.0 when off).
    pub psi_attention_avg: f32,
    /// Phi attention gate weight applied to perception (1.0 = neutral).
    pub phi_attention_weight: f32,
    /// Whether attention budget was exceeded this cycle (subsystems skipped).
    pub attention_budget_exceeded: bool,
    /// Total cycle elapsed at budget check point (microseconds).
    pub attention_budget_elapsed_us: u64,
    /// Input similarity to previous cycle (cosine, 0.0–1.0).
    pub input_similarity: f32,
    /// Whether input memoization was used (skipped re-encoding).
    pub input_memoized: bool,
    /// Whether attention budget gating skipped expensive subsystems this cycle.
    pub attention_budget_gated: bool,
    /// AttentionShift motor command intensity applied to attention_sensitivity.
    pub attention_shift_applied: f32,
}

/// Harmonic and moral geometry telemetry.
///
/// Grouped from CycleMetadata flat fields. `#[serde(flatten)]` preserves
/// the original JSON format for backwards compatibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarmonicMetrics {
    /// Harmonies integrator: overall value alignment score (0.0–1.0, 0.0 when off).
    pub harmonies_alignment: f32,
    /// Harmonies integrator: whether the current action was approved.
    pub harmonies_approved: bool,
    /// Harmonic field coherence — geometric mean of all 8 harmonics (0.0 when off).
    pub harmonic_field_coherence: f64,
    /// Infinite Love resonance — emergent unity measure (0.0 when off).
    pub harmonic_love_resonance: f64,
    /// Number of harmonic interference patterns detected (0 when off).
    pub harmonic_interferences: usize,
    /// 8D harmony coordinates: cosine similarity to each Harmony basis.
    pub harmony_coordinates: [f64; N_HARMONIES],
    /// Softmax distribution over harmonies for the current scenario.
    pub moral_scenario_distribution: [f64; N_HARMONIES],
    /// Softmax distribution over harmonies for the EMA prior.
    pub moral_prior_distribution: [f64; N_HARMONIES],
    /// KL divergence from moral prior to observed.
    pub moral_kl_divergence: f64,
    /// Entropy of observed harmony distribution.
    pub moral_entropy: f64,
    /// Moral surprise: -log p(dominant harmony).
    pub moral_surprise: f64,
    /// Dominant harmonic mode (e.g., "Wisdom", "Play", "Coherence").
    pub dominant_harmonic: String,
    /// Current guiding question from wisdom system (e.g., "What don't I know?").
    pub guiding_question: String,
    /// Guiding question subsystem priority category (empty when no question active).
    pub guiding_priority_category: String,
}

/// Ethical and moral topology telemetry.
///
/// Grouped from CycleMetadata flat fields. `#[serde(flatten)]` preserves
/// the original JSON format for backwards compatibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EthicalTelemetry {
    /// Moral judgment score for this cycle (-1.0 to 1.0). 0.0 when skipped.
    pub moral_score: f32,
    /// Moral violation category that triggered specific steering (empty when none).
    pub moral_steering_category: String,
    /// Unified value evaluator overall score (0.0–1.0, 0.0 when off).
    pub value_evaluator_score: f64,
    /// Value evaluator decision this cycle ("" when off).
    pub value_evaluator_decision: String,
    /// Learned value feedback trend (moving avg of recent moral assessments, -1.0 to 1.0).
    pub value_feedback_trend: f32,
    /// Value evaluator gate factor applied to learning rate (1.0 = no change).
    pub value_gate_factor: f32,
    /// Soul value alignment score (-1.0 to 1.0). 0.0 when soul disabled.
    pub soul_alignment: f32,
    /// Empathic unification: compassion level for current input (0.0–1.0, 0.0 when off).
    pub empathic_compassion: f64,
    /// Empathic unification: patience adjustment (-1.0 to 1.0, 0.0 = neutral).
    pub empathic_tone_adj: f64,
    /// Empathic tone adjustment applied to speech rate (-1.0 to 1.0, 0.0 = no change).
    pub empathic_speech_rate_mod: f32,
    /// β₀: connected components in moral space (1 = unified).
    pub moral_topo_beta_0: usize,
    /// β₁: 1-cycles (circular reasoning patterns).
    pub moral_topo_beta_1: usize,
    /// β₂: 2-voids in moral space.
    pub moral_topo_beta_2: usize,
    /// Unity score (1.0 = fully connected, < 1.0 = fragmented).
    pub moral_topo_unity: f64,
    /// Completeness: fraction of harmonies with non-trivial variance.
    pub moral_topo_completeness: f64,
    /// Circularity: proportion of cycles among persistent features.
    pub moral_topo_circularity: f64,
    /// Moral free energy (FEP surprise on harmony manifold, 0.0 when not computed).
    pub moral_topo_free_energy: f64,
    /// Dominant harmony axis index (0–7, maps to Harmony::all()).
    pub moral_topo_dominant_harmony: u8,
    /// Number of scenarios in the moral topology sliding window.
    pub moral_topo_scenario_count: usize,
    /// Composite anomaly score (0.0–1.0). 0.0 = nominal.
    pub moral_anomaly_score: f64,
    /// True when the dominant harmony axis flipped since last evaluation.
    pub moral_value_inversion: bool,
    /// True when free energy exceeds configured σ multiplier of rolling trajectory mean.
    pub moral_free_energy_spike: bool,
    /// True when moral_drift(20) exceeds configured threshold.
    pub moral_drift_alert: bool,
    /// True when β₀ increased since last topology evaluation.
    pub moral_fragmentation_increase: bool,
    /// True when anomaly response modulations were applied this cycle.
    pub moral_anomaly_response_applied: bool,
    /// True when trajectory convergence (compartmentalized adversarial pattern) detected.
    /// This is the most dangerous class of anomaly: individually benign requests forming
    /// an emergent hazardous cluster.
    #[serde(default)]
    pub moral_trajectory_convergence: bool,
    /// Trajectory convergence severity in \[0.0, 1.0\].
    #[serde(default)]
    pub moral_convergence_severity: f64,
    /// Name of matched hazard signature template (e.g. "weaponization"), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moral_matched_hazard: Option<String>,
    /// Harmony entropy (moral breadth): Shannon entropy of harmony variance distribution.
    /// Range: [0, ln(8)] ≈ [0, 2.08]. Higher = broader moral engagement.
    #[serde(default)]
    pub harmony_entropy: f64,
    /// Whether a moral attractor basin was detected (low free energy + low drift).
    #[serde(default)]
    pub moral_attractor_detected: bool,

    // ── Hodge Decomposition (Persistence-Weighted Cross-Scale Symmetry) ─
    /// Harmonic fraction: topologically-protected global resonance (0.0–1.0).
    /// Persistence-weighted integral across all Rips scales.
    #[serde(default)]
    pub hodge_harmonic_fraction: f64,
    /// Gradient fraction: hierarchical, directed information flow (0.0–1.0).
    #[serde(default)]
    pub hodge_gradient_fraction: f64,
    /// Curl fraction: recurrent, rotational information cycling (0.0–1.0).
    #[serde(default)]
    pub hodge_curl_fraction: f64,
    /// Critical Rips scale where harmonic fraction exceeds 0.5 (moral coherence
    /// phase transition). -1.0 if no transition detected. Lower = more fragile.
    /// Note: must not be NaN — NaN serializes as null through flatten → Value
    /// intermediary, breaking serde roundtrip.
    #[serde(default = "default_no_transition")]
    pub hodge_critical_scale: f64,
    /// Whether the system is at criticality (harmonic ∈ [0.2, 0.8]).
    /// The "Goldilocks zone" between echo chamber and isolation.
    #[serde(default)]
    pub hodge_at_criticality: bool,

    /// Whether the system is in Sacred Stillness active rest mode.
    #[serde(default)]
    pub in_active_rest: bool,
    /// Consecutive cycles of Sacred Stillness dominance.
    #[serde(default)]
    pub stillness_dominance_streak: u16,
    /// Unified ethical verdict for this cycle ("Safe", "Caution", or "Blocked").
    #[serde(default)]
    pub unified_verdict: String,
    /// Consequence tracker prediction accuracy (EMA, 0.0–1.0).
    /// Tracks whether ethical verdicts (Safe/Caution/Blocked) correctly predicted outcomes.
    /// 0.5 = uninformative prior, 1.0 = perfect calibration.
    /// Science: Friston (2010) — active inference; Cushman (2013) — dual-process moral cognition.
    #[serde(default = "default_consequence_accuracy")]
    pub ethics_consequence_accuracy: f64,

    // ── Spinozist Affect Fingerprint ─────────────────────────────
    /// 18D Spinozist affect coordinates (cosine similarity to each affect HV).
    /// Order: Harm, Care, Consent, Deception, Joy, Sadness, Fairness, Obligation,
    /// Vulnerability, Autonomy, Desire, Sacred, Authority, Loyalty, Purity,
    /// Liberty, Proportionality, Reciprocity.
    #[serde(default)]
    pub moral_affect_coords: [f32; 18],
    /// Maximum FluctuatioAnimi tension across opposing affect pairs.
    /// High tension (>0.5) indicates moral ambiguity.
    #[serde(default)]
    pub moral_fluctuatio_tension: f32,
    /// Whether the scenario is morally ambiguous (fluctuatio > threshold).
    #[serde(default)]
    pub moral_is_ambiguous: bool,
    /// Epistemic confidence from affect adequacy scores (mean of active affects).
    #[serde(default)]
    pub moral_epistemic_confidence: f32,
}

fn default_consequence_accuracy() -> f64 {
    0.5
}

fn default_no_transition() -> f64 {
    -1.0
}

/// Free energy principle (FEP) and predictive processing telemetry.
///
/// Grouped from CycleMetadata flat fields. `#[serde(flatten)]` preserves
/// the original JSON format for backwards compatibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FepTelemetry {
    /// FEP action index selected this cycle (0=exploit, 1=consolidate, 2=explore, 3=tighten).
    pub fep_action: usize,
    /// FEP pragmatic value for selected action (0.0 when not computed).
    pub fep_pragmatic_value: f64,
    /// FEP accuracy component (expected log likelihood, 0.0 when not computed).
    pub fep_accuracy: f64,
    /// FEP complexity component (KL from prior, 0.0 when not computed).
    pub fep_complexity: f64,
    /// FEP surprise component (negative log evidence, 0.0 when not computed).
    pub fep_surprise: f64,
    /// FEP TD error magnitude (0.0 when not computed).
    pub fep_td_error: f64,
    /// Predictive processing free energy (0.0 when off).
    pub predictive_free_energy: f64,
    /// Predictive processing phi modulation (1.0 when off — neutral).
    pub predictive_phi_modulation: f64,
    /// ODE trajectory expected free energy (best action, 0.0 when planning disabled).
    pub trajectory_efe: f64,
    /// Action selected by trajectory planning (0 when disabled).
    pub trajectory_best_action: usize,
    /// Integrated surprise along best trajectory (0.0 when disabled).
    pub trajectory_surprise: f64,
    /// Total ODE steps across all action rollouts (0 when disabled).
    pub trajectory_ode_steps: usize,
    // ── Markov Blanket Telemetry (Friston 2013) ──────────────────────
    /// Sensory permeability of the Markov blanket (0.0–1.0).
    pub blanket_sensory_permeability: f64,
    /// Active permeability (0.0–1.0).
    pub blanket_active_permeability: f64,
    /// Effective permeability (geometric mean, 0.0–1.0).
    pub blanket_effective_permeability: f64,
    /// Permeability trend (positive = opening, negative = closing).
    pub blanket_trend: f64,
    /// Whether the system is ready for blanket coalescence.
    pub blanket_coalescence_ready: bool,
    /// Number of active peer coalitions.
    pub blanket_coalition_count: usize,
}

/// Mesh network telemetry.
///
/// Grouped from CycleMetadata flat fields. `#[serde(flatten)]` preserves
/// the original JSON format for backwards compatibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeshTelemetry {
    /// Number of connected Iroh P2P swarm peers (from NetworkService).
    pub swarm_peer_count: u32,
    /// Mean Phi across all connected swarm peers (0.0 when no peers).
    pub network_mean_phi: f64,
    /// Network consciousness coherence — how aligned peer phi values are (0.0–1.0).
    pub network_coherence: f64,
    /// Mesh network composite health score (0.0–1.0, 0.0 when mesh disabled).
    pub mesh_health_score: f32,
    /// Number of active mesh peers (0 when mesh disabled).
    pub mesh_peer_count: u32,
    /// Total bytes sent over mesh since startup.
    pub mesh_bytes_sent: u64,
    /// Total bytes received from mesh since startup.
    pub mesh_bytes_received: u64,
    /// Mesh compression ratio (0.0–1.0, lower is better).
    pub mesh_compression_ratio: f64,
    /// Current AIMD bandwidth budget in bytes.
    pub mesh_bandwidth_budget: u64,
    /// Cumulative bandwidth throttle events.
    pub mesh_packets_throttled: u64,
    /// Cumulative packets that failed decryption (wrong key or corrupted).
    pub mesh_packets_decrypt_failed: u64,
}

/// Feedback proposal system telemetry — proposal counts, integration stats, consensus outcomes.
///
/// Consolidates per-cycle feedback proposal data and the consensus integration results.
/// Traces (per-proposal attribution) are populated when `config.trace_feedback = true`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FeedbackTelemetry {
    /// Number of confidence proposals collected this cycle.
    pub feedback_confidence_proposals: u32,
    /// Number of learning rate proposals collected this cycle.
    pub feedback_lr_proposals: u32,
    /// Number of exploration proposals collected this cycle.
    pub feedback_exploration_proposals: u32,
    /// Number of threshold proposals collected this cycle.
    pub feedback_threshold_proposals: u32,
    /// Consensus confidence value after integration (0.0–1.0).
    pub consensus_confidence: f64,
    /// Consensus learning rate value after integration (1.0–3.0).
    pub consensus_lr: f64,
    /// Consensus exploration value after integration (0.0–1.0).
    pub consensus_exploration: f64,
    /// Consensus threshold value after integration (0.5–2.0).
    pub consensus_threshold: f64,
    /// Per-proposal trace for confidence (populated when `trace_feedback = true`).
    /// Each entry: `(source_label, "Add(+0.0300)")`.
    pub feedback_trace_confidence: Vec<(String, String)>,
    /// Per-proposal trace for learning rate (populated when `trace_feedback = true`).
    pub feedback_trace_lr: Vec<(String, String)>,
    /// Per-proposal trace for exploration (populated when `trace_feedback = true`).
    pub feedback_trace_exploration: Vec<(String, String)>,
    /// Per-proposal trace for threshold (populated when `trace_feedback = true`).
    pub feedback_trace_threshold: Vec<(String, String)>,
}

/// Cantor fractal dream subsystem telemetry.
///
/// Reports CRHV broadcast buffer state, dream consolidation metrics,
/// and resonator codebook statistics from the CantorDreamManager.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CantorTelemetry {
    /// Number of CRHVs currently in the GWT broadcast buffer (cap 32).
    pub cantor_buffer_occupancy: u32,
    /// Smoothed metacognitive depth from dream surprise (0.0-1.0).
    pub cantor_metacognitive_depth: f64,
    /// Resonance boost from coherent CRHV pairs (0.0-1.0).
    pub cantor_resonance_boost: f64,
    /// Depth histogram: counts of CRHVs at each recursive depth level (6 bins).
    pub cantor_depth_histogram: [u32; 6],
    /// EMA of dream consolidation surprise (0.0-1.0).
    pub cantor_dream_surprise: f64,
    /// Number of entries in the cleanup engine codebook.
    pub cantor_codebook_size: u32,
}

/// Per-module execution timings in microseconds for overhead profiling.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModuleTimings {
    pub affective_bridge: u64,
    pub predictive_processing: u64,
    pub cross_modal_binding: u64,
    pub surprise_exploration: u64,
    pub prefrontal: u64,
    pub meta_cognition: u64,
    pub narrative_self: u64,
    pub gwt: u64,
    pub virtual_body: u64,
    pub embodied_cognition: u64,
    pub dream_replay: u64,
    pub moral_algebra: u64,
    pub consciousness_resonance: u64,
    pub temporal_consciousness: u64,
    pub attention_schema: u64,
    pub narrative_gwt: u64,
    pub consciousness_thermodynamics: u64,
    pub phenomenal_binding: u64,
    pub hierarchical_free_energy: u64,
    pub resonator_recall: u64,
    pub support_intelligence: u64,
    pub temporal_analyzer: u64,
    pub primitive_lattice: u64,
    pub compositionality: u64,
    pub value_evaluator: u64,
    pub consciousness_profile: u64,
    pub harmonics: u64,
    pub primitive_reasoning: u64,
    pub causal_explanation: u64,
    pub adaptive_reasoning: u64,
    pub epistemic_tiers: u64,
    pub phi_validation: u64,
    pub dissipative_consciousness: u64,
    pub epistemic_conflict: u64,
    pub consciousness_equation_v2: u64,
    pub hierarchical_ltc: u64,
    pub primitive_evolution: u64,
    pub consciousness_holography: u64,
    pub differentiable_consciousness: u64,
    pub affective_consciousness: u64,
    pub unified_consciousness_pipeline: u64,
    pub multi_modal_integration: u64,
    pub synthetic_grounding: u64,
    pub epistemic_gate: u64,
    pub semantic_value_embedder: u64,
    pub composition_rules: u64,
    pub harmonies_integration: u64,
    pub meta_cognitive_reasoning: u64,
    pub code_primitive_routing: u64,
    pub empathic_unification: u64,
    pub multi_objective_evolution: u64,
    /// Grid encoder: spatial reasoning via HDC grid encoding
    pub grid_encoder: u64,
    /// Stability regime: BinaryHV-based regime detection + crystallization
    pub stability_regime: u64,
    // ── Core Pipeline Phases (previously un-instrumented) ──
    /// HDC encoding: text → 16,384-bit hypervector
    pub core_hdc_encode: u64,
    /// BinaryHV conversion + compression for LTC input
    pub core_compress: u64,
    /// Semantic memory lookup (HDC projection + LR factor computation)
    pub core_semantic_lookup: u64,
    /// CfC temporal network step (closed-form ODE solve)
    pub core_cfc_step: u64,
    /// Multi-scale prediction + state readout
    pub core_predict: u64,
    /// Training step (BPTT or SPSA) — 0 when no learning occurs
    pub core_training: u64,
    /// Parallel post-processing (rayon: stability + memory + episodic)
    pub core_parallel_postprocess: u64,

    // ── End-of-cycle sections (previously un-instrumented) ──
    /// Unified living mind integration (cfg-gated `full_consciousness`)
    pub living_mind: u64,
    /// Master consciousness equation (every Nth cycle)
    pub master_consciousness_equation: u64,
    /// End-of-cycle homeostasis (clamps, drift correction)
    pub homeostasis: u64,
    /// Spectral MIP finder (O(n³) Fiedler + Cholesky, every 47 cycles)
    pub spectral_mip: u64,
    /// Soul experience integration (value learning feedback)
    pub soul_experience: u64,
    /// CycleMetadata struct construction + format!() serialization
    pub metadata_assembly: u64,
    /// Unified consciousness engine (SpectralMIP + MultiModal + EqV2 + Pipeline)
    pub consciousness_engine: u64,
    /// Consciousness engine sub-component: Equation V2 (7-theory C(t))
    pub consciousness_engine_equation_v2: u64,
    /// Consciousness engine sub-component: Unified pipeline (end-to-end)
    pub consciousness_engine_pipeline: u64,
    /// Consciousness engine sub-component: Multi-modal integration
    pub consciousness_engine_multimodal: u64,
    /// Unified ethics engine (MoralParser + MoralAlgebra + ValueEvaluator + Harmonies)
    pub ethics_engine: u64,
    /// Ethics engine sub-component: Moral parser + algebra
    pub ethics_engine_moral: u64,
    /// Ethics engine sub-component: Value evaluator
    pub ethics_engine_value: u64,
    /// Ethics engine sub-component: Harmonies integrator
    pub ethics_engine_harmonies: u64,
    // ── Mid-cycle sections (Session 10 instrumentation) ──
    /// World model: sensory update + stiffness/level-error feedback
    pub world_model: u64,
    /// Resonator codebook: novelty check + symbol addition + causal chain entries
    pub resonator_codebook: u64,
    /// High-phi episode → resonator codebook promotion (every 97 cycles)
    pub high_phi_promotion: u64,
    /// Demand-driven consolidation trigger (error spike or semantic miss)
    pub demand_consolidation: u64,
    /// Episodic replay CfC retraining + memory coordinator graduations
    pub episodic_replay: u64,
    /// Hyper-parameter optimization (Meta-Forge)
    pub parameter_optimization: u64,
    /// Counterfactual dream cycle
    pub dream_cycle: u64,
    /// Moral topology: persistent homology analysis on moral scenarios
    pub moral_topology: u64,
    /// Math service: dynamic math problem solving (statistics, linear algebra, FFT, etc.)
    pub math_service: u64,
}
