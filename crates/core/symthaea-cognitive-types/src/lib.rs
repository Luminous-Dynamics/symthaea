// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pure data types for `CognitiveLoopService` cycle telemetry.
//!
//! Extracted from `symthaea`'s `cognitive_loop::types::telemetry` so that
//! adding/editing a telemetry field doesn't force a full `symthaea` rebuild.
//! `CycleMetadata` itself (the ~250-field aggregator struct) and
//! `NeuromodTelemetry` stay in the main crate because they reference
//! manager-owned types (`MuseTelemetry`, `MultimodalTelemetry`,
//! `neuromodulators::NeuromodSnapshot`) that live there — everything else
//! CycleMetadata flattens in is a pure, standalone data type and lives here.
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
//! New fields MUST use the domain prefix. Do not add unprefixed fields.

use serde::{Deserialize, Serialize};
use symthaea_types::N_HARMONIES;

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
// These fields are populated in `cycle_phase_output/` and serialized via
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

fn default_one_f32() -> f32 {
    1.0
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

// ═══════════════════════════════════════════════════════════════════════════════
// SUBSTRATE / THERMAL / INTEGRITY TELEMETRY — moved from cognitive_loop/types/mod.rs
// ═══════════════════════════════════════════════════════════════════════════════

// ── Substrate Telemetry ─────────────────────────────────────────────────────

/// Substrate telemetry snapshot returned by `SubstrateManager::telemetry()`.
///
/// Groups all substrate-related fields into a single struct for assignment
/// to `CycleMetadata.substrate` via `metadata.substrate = self.substrate_manager.telemetry()`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SubstrateTelemetry {
    /// Effective substrate feasibility [0,1] used in consciousness equation.
    /// Legacy field — identical to `substrate_effective_feasibility`.
    pub substrate_feasibility: f64,
    /// Describes a substrate transition that occurred during this cycle (if any).
    pub substrate_transition: Option<String>,
    /// Raw substrate feasibility before validation overlay (0.0-1.0).
    pub substrate_feasibility_raw: f64,
    /// Honest evidence confidence for current substrate (0.0-0.95).
    pub substrate_honest_confidence: f64,
    /// Effective feasibility after validation overlay blending (0.0-1.0).
    pub substrate_effective_feasibility: f64,
    /// CfC tau factor from substrate speed modulation [0.5, 2.0].
    #[serde(default = "default_one_f32_substrate")]
    pub substrate_tau_factor: f32,
    /// Scale pressure: log10(substrate_max_scale / bio_max_scale).
    #[serde(default)]
    pub substrate_scale_pressure: f32,
    /// Per-region feasibility breakdown (empty when per-region not configured).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub per_region_feasibility: Vec<(String, f32)>,
    /// HDC encoding noise fraction applied this cycle [0.0, 0.1].
    /// Non-zero when substrate encoding noise is enabled and scale_pressure < 0.
    #[serde(default)]
    pub substrate_encoding_noise: f32,
    /// Total energy spent so far (joules). Monotonically increasing.
    #[serde(default)]
    pub total_energy_spent: f64,
    /// Energy spent this cycle (joules, speed-adjusted via tau_factor).
    #[serde(default)]
    pub energy_this_cycle: f64,
    /// Energy throughput multiplier (ratio of bio energy to substrate energy).
    #[serde(default = "default_one_f32_substrate")]
    pub energy_throughput_multiplier: f32,
    /// Effective HDC/CfC dimensionality fraction [0.1, 1.0].
    /// 1.0 for substrates at or above biological scale.
    #[serde(default = "default_one_f32_substrate")]
    pub effective_dim_fraction: f32,
    /// Number of substrate transitions recorded so far.
    #[serde(default)]
    pub transition_count: usize,
}

fn default_one_f32_substrate() -> f32 {
    1.0
}

// ── Thermal Telemetry ─────────────────────────────────────────────────────

/// Thermal telemetry snapshot from ThermalBridge.
///
/// Reports platform thermal state and its effect on CfC tau modulation.
/// Populated each cycle from `thermal_bridge.signals()`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ThermalTelemetry {
    /// Current platform thermal level (0=Nominal, 1=Fair, 2=Serious, 3=Critical, 4=Emergency).
    #[serde(default)]
    pub thermal_level: u8,
    /// EMA-smoothed CfC tau factor from thermal state [1.0, 2.5].
    /// Multiplied into delta_t as the 10th modulation factor.
    #[serde(default = "default_one_f64_thermal")]
    pub thermal_tau_factor: f64,
    /// Whether the thermal bridge recommends a consciousness profile downgrade.
    #[serde(default)]
    pub should_reduce_profile: bool,
    /// Recommended frequency cap from thermal state (None = no override).
    #[serde(default)]
    pub target_frequency_override: Option<f32>,
}

fn default_one_f64_thermal() -> f64 {
    1.0
}

// ── Integrity Telemetry ───────────────────────────────────────────────────

/// Integrity telemetry snapshot from IntegrityManager.
///
/// Reports tamper detection status: attestation, temporal consistency,
/// behavioral canaries. Feature-gated behind `integrity`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IntegrityTelemetry {
    /// Whether all BLAKE3 attestation hashes matched.
    pub attestation_passed: bool,
    /// Whether temporal consistency (wall clock vs CfC delta_t) passed.
    pub temporal_passed: bool,
    /// Whether all behavioral canaries returned expected results.
    pub canaries_passed: bool,
    /// Number of anomalies detected this cycle.
    pub anomaly_count: usize,
    /// Whether any anomaly has Critical severity.
    pub has_critical: bool,
    /// Cycle number of the last integrity check.
    pub last_check_cycle: usize,
    /// Consciousness confidence multiplier (1.0 = trusted, 0.5 = drift, 0.1 = critical).
    #[serde(default = "default_integrity_confidence")]
    pub integrity_confidence: f32,
    /// Per-attestation detail: (name, passed, consecutive_failures).
    /// Empty when no attestation check ran this cycle.
    #[serde(default)]
    pub attestation_details: Vec<AttestationDetail>,
    /// Unified cross-source failure streak (attestation + canary).
    /// 1-2 = Warning, 3+ = Critical. Resets on clean tick.
    #[serde(default)]
    pub global_failure_streak: usize,
    /// Rolling 60-cycle history of integrity_confidence values for sparkline display.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub confidence_history: Vec<f32>,
}

/// Per-attestation telemetry entry for dashboard visibility.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct AttestationDetail {
    /// Attestation name (e.g., "safety_thresholds").
    pub name: String,
    /// Whether this attestation passed on last check.
    pub passed: bool,
    /// Number of consecutive failures (0 = healthy).
    pub consecutive_failures: usize,
}

/// When the `integrity` feature is off, IntegrityTelemetry defaults to "all pass, fully trusted".
/// This prevents downstream consumers (Pulse, SafetyAgent) from seeing uninitialized false-alarm data.
impl Default for IntegrityTelemetry {
    fn default() -> Self {
        Self {
            attestation_passed: true,
            temporal_passed: true,
            canaries_passed: true,
            anomaly_count: 0,
            has_critical: false,
            last_check_cycle: 0,
            integrity_confidence: 1.0,
            attestation_details: Vec::new(),
            global_failure_streak: 0,
            confidence_history: Vec::new(),
        }
    }
}

fn default_integrity_confidence() -> f32 {
    1.0
}

// ═══════════════════════════════════════════════════════════════════════════════
// CYCLE URGENCY — moved from cognitive_loop/types/scheduling.rs
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
    #[allow(dead_code)] // Called from symthaea's mind/tick.rs — unused in default feature set
    pub fn from_arousal(arousal: f32) -> Self {
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
    pub fn from_state(
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
    pub fn should_run(
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
    pub fn run_consciousness_monitors(&self) -> bool {
        matches!(self, CycleUrgency::Critical | CycleUrgency::Normal)
    }
}
