//! Telemetry types — CycleMetadata and sub-structs.

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
    #[serde(skip_serializing_if = "Option::is_none")]
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

/// Metadata about internal decision-making during a cycle.
///
/// Provides observability into which subsystems influenced the cycle's output,
/// enabling debugging of "why did the agent do that?" questions.
///
/// # Domain Groups
///
/// Fields are organized by domain (see section comments). Neuromod fields
/// are nested via `#[serde(flatten)] pub neuromod: NeuromodTelemetry`; assign
/// the snapshot directly to `metadata.neuromod`.
///
/// # Diagnostic-only fields (serialized for dashboards, not read internally)
///
/// These fields are populated in `cycle_phase_output.rs` and serialized via
/// `#[derive(Serialize)]` for API/dashboard consumers, but no internal code
/// reads them after population:
///
/// `broca`, `calibration_improvements`, `calibration_regressions`,
/// `convergence_cycle`, `eq_v2_limiting_component`, `feedback_signals_fired`,
/// `liquid_mamba_effective_rank`, `liquid_mamba_semantic_pe`,
/// `phi_validation_cached`, `social_strategy_bias_applied`,
/// `subsystem_integration_contributors`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CycleMetadata {
    /// Whether the surprise exploration bridge triggered exploration this cycle
    pub surprise_triggered: bool,

    /// Whether the prefrontal cortex vetoed or modified the response
    pub prefrontal_veto: bool,

    /// Confidence score from the reasoning engine (0.0 = unused/off, >0 = active)
    pub reasoning_confidence: f32,

    /// Description of the exploration action taken (if any)
    pub exploration_action: Option<String>,

    /// Whether the reasoning engine's tool gate blocked an action this cycle.
    /// When true, the system used a fallback strategy instead.
    pub reasoning_gate_blocked: bool,

    /// Fallback strategy selected when gating blocked an action (if any)
    pub reasoning_fallback: Option<String>,

    /// Best action from MCTS planning (Tier 1+), if planning ran
    pub reasoning_plan_action: Option<usize>,

    /// MCTS plan confidence (0.0 = no plan, >0 = plan confidence)
    pub reasoning_plan_confidence: f32,

    /// Human-readable reasoning narrative (Tier 2, best-effort)
    pub reasoning_narrative: Option<String>,

    /// Quality diagnostics telemetry (meta-cognitive, dissipative, coherence, anomaly).
    #[serde(flatten)]
    pub quality: QualityDiagnostics,

    /// Narrative self-model's integrated information (0.0 = off/no self, >0 = active self-Φ)
    pub narrative_self_psi: f64,

    /// Virtual body phi modulation (1.0 = neutral, >1 = body boosts consciousness)
    pub body_phi_modulation: f64,

    /// Virtual body affect valence (-1 to 1)
    pub body_valence: f32,

    /// Virtual body affect arousal (0 to 1)
    pub body_arousal: f32,

    /// Master Consciousness Equation level (0.0 to 1.0).
    /// Comprehensive consciousness metric combining Phi, broadcast, working memory,
    /// attention, recurrence, embodiment, knowledge, narrative, and social factors.
    /// Updated every 10th cycle; 0.0 when not yet computed.
    pub consciousness_level: f64,

    /// Predictive self-model safety score (1.0 = safe, 0.0 = unsafe).
    /// 0.0 when predictive self is not enabled.
    pub predictive_self_safety: f32,

    /// Attention subsystem telemetry (schema, GWT, budget, memoization).
    #[serde(flatten)]
    pub attention: AttentionMetrics,

    /// Consciousness resonance dominant frequency (Hz).
    /// 0.0 when resonance is not enabled or no history.
    pub resonance_frequency: f64,

    /// Quantum coherence level (0.0 to 1.0).
    /// 0.0 when quantum coherence is not enabled.
    pub quantum_coherence_level: f64,

    /// Temporal consciousness coherence (0.0 to 1.0).
    /// 0.0 when temporal consciousness is not enabled.
    pub temporal_coherence_score: f64,

    /// Whether temporal consciousness analysis detected a discontinuity.
    pub temporal_discontinuity: bool,

    /// Embodied cognition phi modulation (1.0 = neutral).
    /// 1.0 when embodied cognition is not enabled.
    pub embodied_phi_modulation: f64,

    /// Embodied cognition agency score (0.0 to 1.0).
    /// 0.0 when embodied cognition is not enabled.
    pub embodied_agency: f64,

    /// Whether the narrative-GWT integration vetoed this cycle's action.
    pub narrative_gwt_veto: bool,

    /// Self-Phi from the narrative-GWT integration (0.0 = off/not enabled).
    pub narrative_gwt_self_psi: f64,

    /// Unified Living Mind vitality (0.0 to 1.0).
    /// Measures overall "aliveness" of the system via life-mind continuity.
    /// 0.0 when full_consciousness feature is not enabled.
    pub living_mind_vitality: f64,

    /// Thermodynamic load (0.0 to 1.0, where 1.0 = 6W limit reached).
    pub thermodynamic_load: f32,

    /// Somatic stress from infrastructure errors (0.0 = healthy, 1.0 = critical).
    /// Fed by the SomaticErrorBridge: lock poisoning, task panics, DB failures.
    pub somatic_stress: f64,

    /// Affective bias: cognitive temperature (0.0 to 2.0).
    pub mood_temperature: f32,

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

    /// Cross-modal binding strength (0.0 when off).
    pub cross_modal_binding_strength: f32,

    /// Cross-modal integration Phi (0.0 when off).
    pub cross_modal_psi: f64,

    /// Affective bridge valence (-1 to 1, 0.0 when off).
    pub affective_valence: f32,

    /// Affective bridge arousal (0 to 1, 0.5 when off — neutral).
    pub affective_arousal: f32,

    /// Consciousness thermodynamic entropy (0.0 when off).
    pub thermodynamic_entropy: f64,

    /// Consciousness thermodynamic free energy (0.0 when off).
    pub thermodynamic_free_energy: f64,

    /// Phenomenal binding strength Ψ (0.0 when off).
    pub phenomenal_binding_strength: f64,

    /// Whether phenomenal binding detected fragmentation.
    pub phenomenal_fragmented: bool,

    /// Hierarchical total free energy (0.0 when off).
    pub hierarchical_total_free_energy: f64,

    /// Phi estimate from primitive consciousness decomposition (0.0 when off).
    pub primitive_psi: f64,

    /// Number of causal chains detected by temporal analyzer (0 when off).
    pub temporal_causal_chains: usize,
    /// Temporal continuity ratio (0.0–1.0, 0.0 when off).
    pub temporal_continuity: f64,
    /// Longest causal chain length (0 when off).
    pub temporal_max_chain_length: usize,
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
    /// Multi-dimensional consciousness composite score (0.0 when off).
    pub consciousness_profile_composite: f64,
    /// Synergy-enhanced composite (non-linear dimension interactions, 0.0 when off).
    pub synergy_enhanced_composite: f64,
    /// Number of emergent consciousness properties detected (0 when off).
    pub emergent_properties_count: usize,
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
    /// Holographic consciousness unity score (0.0–1.0, 0.0 when off).
    pub holographic_unity: f64,
    /// Holographic binding strength (0.0 when off).
    pub holographic_binding: f64,
    /// Differentiable consciousness gradient magnitude (0.0 when off).
    pub consciousness_gradient_magnitude: f64,
    /// Limiting component identified by gradient analysis ("" when off).
    pub consciousness_limiting_component: String,
    /// Equation V2 limiting component from ConsciousnessEngine ("" when off).
    pub eq_v2_limiting_component: String,
    /// Affective consciousness valence (-1.0 to 1.0, 0.0 when off).
    pub affect_consciousness_valence: f32,
    /// Affective consciousness arousal (0.0–1.0, 0.0 when off).
    pub affect_consciousness_arousal: f32,
    /// Unified pipeline consciousness score (0.0 when off).
    pub pipeline_consciousness: f64,
    /// Multi-modal integrated phi (0.0 when off).
    pub multimodal_integrated_phi: f64,

    // ── Session 7: Synthetic States + Epistemic Gate ────────────────────
    /// Detected consciousness state label (e.g., "Awake", "Alert", "" when off).
    pub consciousness_state_label: String,
    /// Consciousness state level (0.0–1.0, from NSM grounding, 0.0 when off).
    pub consciousness_state_level: f64,
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
    pub safety_category: Option<String>,

    /// Negation polarity detected in input text (0.0 = no negation, >0.5 = negated).
    pub negation_polarity: f32,

    /// Selected response strategy for this cycle (e.g., "Exploratory", "Supportive").
    pub selected_strategy: String,

    /// Actual effective learning rate used for training this cycle (after all modulations).
    /// 0.0 when no learning occurred.
    pub actual_effective_lr: f32,

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

    /// Σ (Sigma) — Synergistic integration via covariance-based Phi* (Layer 2).
    /// `None` when not computed this cycle (only computed every N cycles).
    pub sigma: Option<f64>,

    /// Spectral MIP Phi — O(n³) Fiedler-ordered MIP approximation (Layer 2+).
    /// `None` when not computed this cycle (only computed every 50 cycles).
    pub spectral_mip_phi: Option<f64>,

    /// Hierarchical spectral MIP Phi (multi-scale: 32→64→128 components).
    /// Uses coarser scales to focus finer scales on the MIP boundary region.
    /// `None` when not computed this cycle (only computed every 100 cycles).
    pub hierarchical_mip_phi: Option<f64>,

    /// Number of scales used in hierarchical MIP (0 when not computed).
    pub hierarchical_mip_scales: usize,

    // ── Structural Phi decomposition ────────────────────────────────
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

    // ── Dynamic consciousness weights ───────────────────────────────
    /// Dynamic consciousness weights [spectral, equation, pipeline, multimodal].
    pub consciousness_weights: [f64; 4],
    /// Weight stability variance (0.0 = stable, >0.01 = oscillating).
    pub consciousness_weight_variance: f64,

    // ── Substrate & Convergence Telemetry ──────────────────────────────
    /// Effective substrate feasibility [0,1] used in consciousness equation.
    /// Legacy field — identical to `substrate_effective_feasibility`.
    /// Use `substrate_feasibility_raw` for the pre-overlay value.
    pub substrate_feasibility: f64,

    /// Substrate telemetry snapshot (from SubstrateManager::telemetry()).
    /// Groups all substrate-related fields for batch assignment.
    #[serde(default)]
    pub substrate: super::SubstrateTelemetry,

    /// Integrity telemetry snapshot (from IntegrityManager).
    /// Reports tamper detection: attestation, temporal, canaries.
    #[serde(default)]
    pub integrity: super::IntegrityTelemetry,

    /// Weight convergence state label (Initializing/Converging/Converged/Oscillating).
    pub weight_convergence_state: String,
    /// Cycle at which weights converged (0 if not yet).
    pub convergence_cycle: usize,

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

    // ── Nurture Attachment Telemetry ─────────────────────────────────────
    /// Current attachment style (e.g., "Forming", "Secure"). Empty when nurture disabled.
    pub attachment_style: Option<String>,
    /// Current attachment security score (0.0–1.0). None when nurture disabled.
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
    #[serde(default)]
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

    // ── Vision Manifold Telemetry ───────────────────────────────────────
    /// Vision manifold telemetry (None when vision-manifold feature disabled or not active).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vision: Option<VisionManifoldTelemetry>,

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
    /// Whether feedback integration was frozen this cycle (dampening streak ≥3).
    /// Science: Turrigiano (2008) — homeostatic synaptic silencing.
    #[serde(default)]
    pub feedback_frozen: bool,
    /// Dominant source proposal concentration (fraction of total, 0.0–1.0).
    #[serde(default)]
    pub dominant_source_concentration: f32,
    /// Whether compound instability was detected (agreement drop + rising errors).
    /// Science: Friston (2010) — cascading precision failures.
    #[serde(default)]
    pub compound_instability: bool,
    /// Whether flow-state feedback relaxation is active (wider dampening threshold).
    /// Science: Csikszentmihalyi (1990) — reduced self-monitoring during flow.
    #[serde(default)]
    pub flow_feedback_relaxed: bool,
    /// Homeostasis pull efficiency: ratio of post/pre distance to target.
    /// <1.0 = pulls working, >1.0 = overcorrecting.
    /// Science: Cannon (1929) — homeostatic regulation monitoring.
    #[serde(default = "default_one_f32")]
    pub homeostasis_efficiency: f32,

    // ── Session 10: Adaptive Feedback Intelligence ────────────────────────
    /// Whether a confidence crash was detected this cycle (>30% drop).
    /// Science: Cools et al. (2008) — serotonergic dip from confidence collapse.
    #[serde(default)]
    pub confidence_crash_detected: bool,
    /// Remaining LR freeze cycles from crash recovery (0 = not frozen).
    #[serde(default)]
    pub crash_freeze_remaining: u32,
    /// Number of distinct proposal sources this cycle (diversity metric).
    /// Science: Dehaene (2014) — healthy cognition requires multi-source consensus.
    #[serde(default)]
    pub proposal_source_count: u32,
    /// Whether low proposal diversity triggered exploration boost.
    #[serde(default)]
    pub low_diversity_boost: bool,
    /// Current hysteresis relaxation factor (1.0 = full, decays with stability).
    /// Science: Kelso (1995) — mode stability permits relaxed boundaries.
    #[serde(default = "default_one_f32")]
    pub hysteresis_factor: f32,
    /// Whether agreement-confidence velocity coupling fired this cycle.
    #[serde(default)]
    pub agreement_confidence_coupling: bool,

    // ── Session 11: Fixes + Adaptive Intelligence ────────────────────────
    /// Whether LR was frozen this cycle by crash freeze (Set proposal pinning).
    #[serde(default)]
    pub lr_frozen: bool,
    /// Proposal conflict ratio (0.0 = unanimous, 0.5 = maximally conflicted).
    /// Science: Dayan & Daw (2008) — model disagreement signals meta-uncertainty.
    #[serde(default)]
    pub proposal_conflict_ratio: f32,
    /// Whether high conflict triggered epistemic exploration boost.
    #[serde(default)]
    pub conflict_exploration_boost: bool,

    // ── Session 12: Wiring + Binding Intelligence ───────────────────────
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

    // ── Session 13: Convergence + Flow Intelligence ─────────────────────
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

/// Vision manifold telemetry snapshot for CycleMetadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VisionManifoldTelemetry {
    /// Whether vision processing was active this cycle.
    pub vision_active: bool,
    /// Per-frame prediction error from the vision manifold CfC.
    pub prediction_error: f32,
    /// Manifold coherence (cosine similarity between state and frame encoding).
    pub manifold_coherence: f32,
    /// Shannon entropy of the attention/surprise map.
    pub attention_entropy: f32,
    /// Number of patches exceeding the surprise threshold.
    pub num_salient_patches: usize,
    /// Frame sequence number.
    pub frame_sequence: u64,
    /// Whether a training step was triggered this cycle.
    pub training_triggered: bool,
    /// Cosine similarity of the scene recognition match (0.0 if no match).
    pub scene_recognition_similarity: f32,
    /// Cross-manifold prediction error (vision→cognitive, 0.0 if predictor disabled).
    pub cross_manifold_prediction_error: f32,
    /// Time spent encoding the frame into HDC (microseconds).
    pub encode_time_us: u64,
    /// Time spent evolving the CfC manifold state (microseconds).
    pub evolve_time_us: u64,
    /// Mean surprise across the visual field (Friston free energy).
    pub vision_mean_surprise: f32,
    /// Multi-timescale horizon prediction errors [short, mid, long].
    pub vision_horizon_errors: Vec<f32>,
    /// Whether a scene was recognized this cycle (Conway episodic encoding).
    pub scene_recognized: bool,
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
    #[serde(default)]
    pub moral_matched_hazard: Option<String>,
    /// Harmony entropy (moral breadth): Shannon entropy of harmony variance distribution.
    /// Range: [0, ln(8)] ≈ [0, 2.08]. Higher = broader moral engagement.
    #[serde(default)]
    pub harmony_entropy: f64,
    /// Whether a moral attractor basin was detected (low free energy + low drift).
    #[serde(default)]
    pub moral_attractor_detected: bool,
    /// Whether the system is in Sacred Stillness active rest mode.
    #[serde(default)]
    pub in_active_rest: bool,
    /// Consecutive cycles of Sacred Stillness dominance.
    #[serde(default)]
    pub stillness_dominance_streak: u16,
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
}

/// Mesh network telemetry.
///
/// Grouped from CycleMetadata flat fields. `#[serde(flatten)]` preserves
/// the original JSON format for backwards compatibility.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MeshTelemetry {
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
}
