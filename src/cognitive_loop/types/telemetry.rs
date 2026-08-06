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

#[cfg(feature = "therapeutic")]
pub use symthaea_cognitive_types::TherapeuticTelemetry;
pub use symthaea_cognitive_types::{
    AttentionMetrics, BrocaGenerationTelemetry, CantorTelemetry, ConsciousnessLevelMetrics,
    EmbodiedAffectMetrics, EthicalTelemetry, FactcheckTelemetry, FeedbackModulationFlags,
    FeedbackTelemetry, FepTelemetry, FoveationBridgeTelemetry, HarmonicMetrics,
    MemoryResonatorMetrics, MeshTelemetry, ModuleTimings, PhysicsBridgeTelemetry,
    QualityDiagnostics, StructuralPhiMetrics, TemporalPhenomenalMetrics,
};

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

fn default_response_profile() -> String {
    "balanced".to_string()
}

fn default_one_f32() -> f32 {
    1.0
}

fn default_half_f32() -> f32 {
    0.5
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

    /// Whether the reasoning engine's tool gate actually ran this cycle (i.e. a
    /// `ToolDescriptor` was present and `classifier::gate()` was invoked at least
    /// once). Distinguishes "gate ran and passed" from "no gate ran" —
    /// `reasoning_gate_blocked` alone cannot: it defaults to `false` both when the
    /// gate passed AND when no gate was evaluated at all. Consumers that learn from
    /// gate outcomes (e.g. posthoc calibration) must check this before treating
    /// `!reasoning_gate_blocked` as "the gate passed."
    #[serde(default)]
    pub reasoning_gate_evaluated: bool,

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
