//! Public types returned by the cognitive loop.
//!
//! Extracted from `mod.rs` to reduce file size while preserving all public APIs.

use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════════
// CYCLE CARRYOVER — state that crosses cycle boundaries
// ═══════════════════════════════════════════════════════════════════════════════

/// Cached consciousness integration scores carried between cycles.
#[derive(Debug, Clone)]
pub(crate) struct ConsciousnessCache {
    /// Predictive processing phi modulation (1.0 = neutral)
    pub(crate) predictive_phi_modulation: f64,
    /// Cross-modal Phi (fed back into confidence)
    pub(crate) cross_modal_psi: f64,
    /// Body phi modulation (fed back into unified_psi)
    pub(crate) body_phi_modulation: f64,
    /// Embodied cognition phi modulation (fed back into unified_psi)
    pub(crate) embodied_phi_modulation: f64,
    /// Quantum coherence level (fed back into exploration boost)
    pub(crate) quantum_coherence: f64,
    /// Last computed Σ (Sigma) — cached for inter-cycle use.
    pub(crate) last_sigma: Option<f64>,
    /// Last spectral MIP Phi — cached for inter-cycle use.
    pub(crate) last_spectral_mip_phi: Option<f64>,
    /// Last harmonic field coherence (cached, updated every 10 cycles).
    pub(crate) last_harmonic_coherence: f64,
    /// Last holographic unity score (0.0–1.0, cached from last analyze).
    pub(crate) last_holographic_unity: f64,
    /// Last multi-modal integrated phi (cached).
    pub(crate) last_multimodal_phi: f64,
    /// Last consciousness equation v2 result (cached, updated every 25 cycles).
    pub(crate) last_equation_v2_consciousness: f64,
    /// Last hierarchical spectral MIP Phi (cached, updated every 100 cycles).
    pub(crate) last_hierarchical_mip_phi: Option<f64>,
    // ── Phase 21: Consciousness-Grounded Control ─────────────────────
    /// Last embodied agency score (cached for strategy modulation).
    pub(crate) last_embodied_agency: f64,
    /// Last predictive free energy (cached for surprise amplitude scaling).
    pub(crate) last_predictive_free_energy: f64,
}

impl Default for ConsciousnessCache {
    fn default() -> Self {
        Self {
            predictive_phi_modulation: 1.0,
            cross_modal_psi: 0.0,
            body_phi_modulation: 1.0,
            embodied_phi_modulation: 1.0,
            quantum_coherence: 0.0,
            last_sigma: None,
            last_spectral_mip_phi: None,
            last_harmonic_coherence: 0.0,
            last_holographic_unity: 0.0,
            last_multimodal_phi: 0.0,
            last_equation_v2_consciousness: 0.0,
            last_hierarchical_mip_phi: None,
            last_embodied_agency: 0.5,
            last_predictive_free_energy: 0.0,
        }
    }
}

/// Urgency state for adaptive subsystem scheduling.
#[derive(Debug, Clone)]
pub(crate) struct UrgencyState {
    /// Urgency level (hysteresis — prevents jitter)
    pub(crate) urgency: CycleUrgency,
    /// Consecutive cycles with error below threshold (Cruise mode trigger)
    pub(crate) consecutive_low_error: u32,
    /// Consecutive high-arousal cycles (Yerkes-Dodson trap detection)
    pub(crate) arousal_trap_counter: u32,
    /// Consecutive cycles since last metacognitive anomaly (for recovery ramp)
    pub(crate) anomaly_recovery_counter: u32,
    /// Whether an anomaly was active in the previous cycle
    pub(crate) anomaly_was_active: bool,
    // ── Phase 17: Predictive Self-Tuning ──────────────────────────────
    /// Mode transition confidence (0.0 = just switched, 1.0 = fully settled)
    pub(crate) mode_confidence: f32,
    /// Previous urgency for transition smoothing
    pub(crate) prev_urgency: super::CycleUrgency,
    /// Cycles since last urgency mode change
    pub(crate) mode_stability_counter: u32,
    /// Consecutive temporal discontinuity cycles (Phase 21 recovery cascade).
    pub(crate) discontinuity_streak: u32,
    /// Remaining cycles of anomaly drift recovery (0 = not active).
    /// Science: Turrigiano (2008) — homeostatic plasticity engages for fixed duration.
    pub(crate) anomaly_drift_recovery: u32,
}

impl Default for UrgencyState {
    fn default() -> Self {
        Self {
            urgency: CycleUrgency::Normal,
            consecutive_low_error: 0,
            arousal_trap_counter: 0,
            anomaly_recovery_counter: 0,
            anomaly_was_active: false,
            mode_confidence: 1.0,
            prev_urgency: CycleUrgency::Normal,
            mode_stability_counter: 0,
            discontinuity_streak: 0,
            anomaly_drift_recovery: 0,
        }
    }
}

/// Learning rate modulation state.
#[derive(Debug, Clone)]
pub(crate) struct LearningState {
    /// Prediction confidence snapshot at cycle start (drift clamping)
    pub(crate) prediction_confidence: f32,
    /// MCE consciousness-level LR boost (decays 10%/cycle between MCE firings)
    pub(crate) mce_lr_boost: f32,
    /// Adaptive learning threshold multiplier (1.0 = config value as-is)
    pub(crate) adaptive_threshold_scale: f32,
    /// Subsystem LR modulation factor (accumulated post-training, consumed next cycle).
    pub(crate) subsystem_lr_factor: f32,
    // ── Phase 17: Predictive Self-Tuning ──────────────────────────────
    /// Self-model accuracy EMA (how well past predictions matched outcomes)
    pub(crate) self_model_accuracy: f32,
}

impl Default for LearningState {
    fn default() -> Self {
        Self {
            prediction_confidence: 0.5,
            mce_lr_boost: 0.0,
            adaptive_threshold_scale: 1.0,
            subsystem_lr_factor: 1.0,
            self_model_accuracy: 0.5,
        }
    }
}

/// Cached quality and diagnostic metrics.
#[derive(Debug, Clone)]
pub(crate) struct QualityMetrics {
    /// Number of detected causal chains (cached from last analysis, every 50 cycles).
    pub(crate) causal_chain_count: usize,
    /// Temporal continuity ratio (0.0–1.0, cached from last analysis, every 100 cycles).
    pub(crate) temporal_continuity: f64,
    /// Last value evaluator overall score (cached, updated every 20 cycles).
    pub(crate) last_value_score: f64,
    /// Last epistemic quality score (cached, updated every 50 cycles).
    pub(crate) last_epistemic_quality: f64,
    /// Last dissipative consciousness health score (0.0–1.0, cached from last update).
    pub(crate) last_dissipative_health: f64,
    /// Last Φ_eff from epistemic conflict (cached, updated every 50 cycles).
    pub(crate) last_phi_eff: f64,
    /// Last differentiable consciousness gradient magnitude (cached).
    pub(crate) last_gradient_magnitude: f64,
    /// Last affective valence (cached from last process_stimulus).
    pub(crate) last_affective_valence: f32,
    /// Last detected consciousness state type (cached, updated every 100 cycles).
    pub(crate) last_consciousness_state: String,
    /// Last epistemic confidence (cached from gate evaluation).
    pub(crate) last_epistemic_confidence: f32,
    /// Last unified pipeline consciousness score (cached).
    pub(crate) last_pipeline_consciousness: f64,
    /// Whether narrative-GWT vetoed the previous cycle (suppresses learning)
    pub(crate) narrative_veto_active: bool,
    /// Cached prefrontal veto (reused on skip cycles when amortized)
    pub(crate) cached_prefrontal_veto: bool,
    /// Smoothed coherence from previous cycle (for velocity computation)
    pub(crate) last_coherence: f32,
    /// Coherence velocity (rate of change, updated every 5 cycles)
    pub(crate) coherence_velocity: f32,
    /// Cached phi validation correlation (updated every 500 cycles)
    pub(crate) phi_validation_correlation: f64,
    /// Adjusted spectral weight for unified Psi (from phi validation)
    pub(crate) phi_spectral_weight: f32,
    /// Cached phenomenal binding strength (from late consciousness)
    pub(crate) last_phenomenal_binding: f64,
    /// Last epistemic conflict count (cached for reasoning override).
    pub(crate) last_epistemic_conflict_count: usize,
    /// Whether epistemic conflicts should force adaptive reasoning on next cycle.
    pub(crate) epistemic_reasoning_override: bool,
    /// Last grid encoding norm (cached between amortization cycles).
    pub(crate) last_grid_norm: f32,
    /// Last grid spatial complexity (cached between amortization cycles).
    pub(crate) last_grid_complexity: f32,
    /// Previous cycle's prediction confidence (for crash detection).
    /// Science: Cools et al. (2008) — rapid confidence drop triggers serotonergic dip.
    pub(crate) prev_confidence_for_crash: f32,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            causal_chain_count: 0,
            temporal_continuity: 0.0,
            last_value_score: 0.0,
            last_epistemic_quality: 0.0,
            last_dissipative_health: 0.0,
            last_phi_eff: 0.0,
            last_gradient_magnitude: 0.0,
            last_affective_valence: 0.0,
            last_consciousness_state: String::new(),
            last_epistemic_confidence: 0.5,
            last_pipeline_consciousness: 0.0,
            narrative_veto_active: false,
            cached_prefrontal_veto: false,
            last_coherence: 0.5,
            coherence_velocity: 0.0,
            phi_validation_correlation: 0.0,
            phi_spectral_weight: 0.6,
            last_phenomenal_binding: 0.5,
            last_epistemic_conflict_count: 0,
            epistemic_reasoning_override: false,
            last_grid_norm: 0.0,
            last_grid_complexity: 0.0,
            prev_confidence_for_crash: 0.5,
        }
    }
}

/// Historical state for cycle-to-cycle continuity.
#[derive(Debug, Clone)]
pub(crate) struct CycleHistory {
    /// MCTS plan action (action_idx, confidence) for next cycle
    pub(crate) mcts_plan: Option<(usize, f32)>,
    /// Body arousal (fed back into CfC tau modulation)
    pub(crate) body_arousal: f32,
    /// Resonance frequency (fed back into delta_t modulation)
    pub(crate) resonance_frequency: f64,
    /// Last MCE consciousness level for learning gating
    pub(crate) consciousness_level: f64,
    /// Recent BinaryHV ring buffer for multi-component consciousness profile.
    /// Capacity bound: 4 elements (evict-before-push via pop_front).
    pub(crate) recent_hvs: std::collections::VecDeque<crate::hdc::BinaryHV>,
    /// Cached causal relations count (avoids calling summarize_understanding every cycle).
    pub(crate) last_causal_relations: usize,
    /// Cached causal average confidence (avoids calling summarize_understanding every cycle).
    pub(crate) last_causal_confidence: f64,
    /// Cached consciousness profile composite (avoids computing every cycle).
    pub(crate) last_profile_composite: f64,
    /// Cached synergy-enhanced composite.
    pub(crate) last_synergy_composite: f64,
    /// Cached emergent properties count.
    pub(crate) last_emergent_count: usize,
    /// Whether an MCTS plan was applied this cycle (for post-hoc evaluation next cycle).
    pub(crate) mcts_plan_applied: Option<(usize, f32, f32)>, // (action, confidence, prediction_error_at_time)
    /// Previous cycle's compressed_state for input similarity memoization.
    pub(crate) last_compressed_state: Option<Vec<f32>>,
    /// Previous cycle's emotion_contagion valence (for homeostasis return-to-baseline).
    pub(crate) last_emotion_valence: f32,
    /// Previous cycle's emotion_contagion arousal (for homeostasis return-to-baseline).
    pub(crate) last_emotion_arousal: f32,
    // ── Phase 17: Predictive Self-Tuning ──────────────────────────────
    /// Rolling window of recent prediction errors (last 16 cycles).
    pub(crate) error_history: std::collections::VecDeque<f32>,
    /// Self-model prediction: (cycle_made, predicted_confidence, predicted_urgency)
    pub(crate) self_model_prediction: Option<(usize, f32, super::CycleUrgency)>,
    /// Cached coherence value for this cycle (computed once, reused everywhere).
    pub(crate) cached_coherence: Option<f32>,
}

impl Default for CycleHistory {
    fn default() -> Self {
        Self {
            mcts_plan: None,
            body_arousal: 0.5,
            resonance_frequency: 0.0,
            consciousness_level: 0.0,
            recent_hvs: std::collections::VecDeque::with_capacity(4),
            last_causal_relations: 0,
            last_causal_confidence: 0.0,
            last_profile_composite: 0.0,
            last_synergy_composite: 0.0,
            last_emergent_count: 0,
            mcts_plan_applied: None,
            last_compressed_state: None,
            last_emotion_valence: 0.0,
            last_emotion_arousal: 0.0,
            error_history: std::collections::VecDeque::with_capacity(16),
            self_model_prediction: None,
            cached_coherence: None,
        }
    }
}

/// State carried over between consecutive cognitive cycles.
///
/// These fields represent the "memory" of the previous cycle that influences
/// the next cycle's processing. All fields are reset to defaults by
/// `CognitiveLoopService::reset()`.
#[derive(Debug, Clone, Default)]
pub(crate) struct CycleCarryover {
    /// Cached consciousness integration scores
    pub(crate) consciousness: ConsciousnessCache,
    /// Urgency scheduling state
    pub(crate) urgency: UrgencyState,
    /// Learning rate modulation
    pub(crate) learning: LearningState,
    /// Cached quality/diagnostic metrics
    pub(crate) quality: QualityMetrics,
    /// Historical state for continuity
    pub(crate) history: CycleHistory,
}

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
}

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
    pub neuromod_snapshot: Option<super::neuromodulators::NeuromodSnapshot>,

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
    /// Effective circadian hour after phase offset (0.0–24.0).
    pub circadian_effective_hour: f32,

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

    /// Meta-cognitive self-model accuracy (0.0 = uncertain, 1.0 = perfect self-knowledge)
    pub meta_cognitive_accuracy: f32,

    /// Meta-cognitive recursion depth (0 = off, 1 = basic, 2+ = recursive self-modeling)
    pub meta_cognitive_depth: u8,

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

    /// Attention schema focus intensity (0.0 to 1.0).
    /// 0.0 when attention schema is not enabled.
    pub attention_schema_focus: f32,

    /// Whether a GWT broadcast occurred this cycle.
    pub gwt_broadcast: bool,

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

    /// Number of insights gained from dream replay this cycle (0 = no dreaming).
    pub dream_insights: usize,

    /// Best Phi improvement discovered by dream counterfactuals (0.0 = no improvement).
    pub dream_phi_improvement: f32,

    /// Total accumulated wisdom entries from dreaming.
    pub dream_wisdom_count: usize,

    /// Predictive processing free energy (0.0 when off).
    pub predictive_free_energy: f64,

    /// Predictive processing phi modulation (1.0 when off — neutral).
    pub predictive_phi_modulation: f64,

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

    /// Rolling average of Phi observations from phi_attention (0.0 when off).
    pub psi_attention_avg: f32,

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
    /// Whether a continuity gap triggered demand replay this cycle.
    pub continuity_replay_triggered: bool,

    // ── Session 1: Compositionality + Value Evaluator ──────────────────────
    /// Total compositions registered in the compositionality engine (0 when off).
    pub compositionality_total: usize,
    /// Unified value evaluator overall score (0.0–1.0, 0.0 when off).
    pub value_evaluator_score: f64,
    /// Value evaluator decision this cycle ("" when off).
    pub value_evaluator_decision: String,

    /// Composition rule engine: name of the rule applied this cycle (empty when off).
    pub composition_rule_applied: String,
    /// Harmonies integrator: overall value alignment score (0.0–1.0, 0.0 when off).
    pub harmonies_alignment: f32,
    /// Harmonies integrator: whether the current action was approved.
    pub harmonies_approved: bool,

    /// Empathic unification: compassion level for current input (0.0–1.0, 0.0 when off).
    pub empathic_compassion: f64,
    /// Empathic unification: patience adjustment (-1.0 to 1.0, 0.0 = neutral).
    pub empathic_tone_adj: f64,
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
    /// Harmonic field coherence — geometric mean of all 7 harmonics (0.0 when off).
    pub harmonic_field_coherence: f64,
    /// Infinite Love resonance — emergent unity measure (0.0 when off).
    pub harmonic_love_resonance: f64,
    /// Number of harmonic interference patterns detected (0 when off).
    pub harmonic_interferences: usize,

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

    // ── Session 5: Dissipative + Conflict + Equation + Hierarchical + Evolution ──
    /// Dissipative consciousness health score (0.0–1.0, 0.0 when off).
    pub dissipative_health: f64,
    /// Current thermodynamic regime (e.g., "Subcritical", "Critical", "Supercritical").
    pub dissipative_regime: String,
    /// Dissipative entropy production rate (0.0 when off).
    pub dissipative_entropy_rate: f64,
    /// Φ_eff = Φ × R^γ from epistemic conflict reliability weighting (0.0 when off).
    pub epistemic_phi_eff: f64,
    /// Number of inter-theory conflicts detected (0 when off).
    pub epistemic_conflict_count: usize,
    /// Consciousness Equation v2 C(t) result (0.0 when off).
    pub equation_v2_consciousness: f64,
    /// Hierarchical LTC estimated Phi (0.0 when off).
    pub hierarchical_ltc_phi: f32,

    // ── Session 6: Holographic + Differentiable + Affective + Pipeline + MultiModal ──
    /// Holographic consciousness unity score (0.0–1.0, 0.0 when off).
    pub holographic_unity: f64,
    /// Holographic binding strength (0.0 when off).
    pub holographic_binding: f64,
    /// Differentiable consciousness gradient magnitude (0.0 when off).
    pub consciousness_gradient_magnitude: f64,
    /// Limiting component identified by gradient analysis ("" when off).
    pub consciousness_limiting_component: String,
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

    /// Moral judgment score for this cycle (-1.0 to 1.0). 0.0 when moral evaluation was skipped.
    pub moral_score: f32,

    /// Selected response strategy for this cycle (e.g., "Exploratory", "Supportive").
    pub selected_strategy: String,

    /// Actual effective learning rate used for training this cycle (after all modulations).
    /// 0.0 when no learning occurred.
    pub actual_effective_lr: f32,

    /// Cycle reward signal (internal + external blend, -1.0 to 1.0).
    pub cycle_reward: f32,

    /// FEP action index selected this cycle (0=exploit, 1=consolidate, 2=explore, 3=tighten).
    pub fep_action: usize,

    /// Learned value feedback trend (moving avg of recent moral assessments, -1.0 to 1.0).
    pub value_feedback_trend: f32,

    /// Number of support triage classifications this cycle (0 when support disabled).
    pub support_triage_count: u32,

    /// Whether a support predictive alert fired this cycle.
    pub support_alert_fired: bool,

    /// Number of knowledge articles graduated via federation this cycle.
    pub support_federation_graduated: usize,

    /// Support subsystem expected free energy (0.0 when not computed).
    pub support_efe: f64,

    /// Soul value alignment score (-1.0 to 1.0). Weighted cosine similarity
    /// of current encoding against Seven Harmonies core values. Modulates
    /// learning rate: >0.3 boosts, <-0.3 dampens. 0.0 when soul disabled.
    pub soul_alignment: f32,

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

    /// Number of symbols in the resonator semantic codebook (0 when disabled).
    pub resonator_codebook_size: usize,

    /// Number of episodes stored in resonator memory (0 when disabled).
    pub resonator_episodes: usize,

    /// Number of iterations used in last resonator factorization (0 when not run).
    pub resonator_factorization_iters: usize,

    /// Per-module timing (microseconds). 0 = module disabled or not run this cycle.
    pub module_timings_us: ModuleTimings,

    /// Circadian phase (Dawn/Day/Dusk/Night) from chronobiology module.
    pub circadian_phase: String,

    /// Circadian plasticity modifier (0.0–1.0) applied to learning rate.
    pub circadian_plasticity: f32,

    /// Phi attention gate weight applied to perception (1.0 = neutral).
    pub phi_attention_weight: f32,

    /// Current guiding question from wisdom system (e.g., "What don't I know?").
    pub guiding_question: String,

    /// Dominant harmonic mode (e.g., "Wisdom", "Play", "Coherence").
    pub dominant_harmonic: String,

    // ── Phase 9: Resonator-Memory Loop + FEP Deepening ──────────────
    /// Whether resonator recall primed working memory confidence this cycle.
    pub resonator_wm_primed: bool,
    /// Number of episodes reconsolidated via resonator recall this cycle.
    pub resonator_reconsolidated: usize,
    /// Number of high-Phi episodes promoted to resonator codebook this cycle.
    pub resonator_promotions: usize,
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

    // ── Phase 10: Self-Regulating Resonator + FEP-Behavior Coupling ─────
    /// Best cosine similarity from resonator recall (0.0 when no matches).
    pub resonator_best_sim: f32,
    /// Number of codebook entries evicted this cycle (redundancy pruning).
    pub codebook_evictions: usize,
    /// Codebook diversity: average pairwise cosine distance (0.0–1.0).
    pub codebook_diversity: f32,

    // ── Phase 13: Predictive Resonator + Cross-Module Coherence ──────
    /// Resonator prediction error: cosine distance between last cycle's best match
    /// and this cycle's compressed state (0.0 = perfect prediction, 1.0 = orthogonal).
    pub resonator_prediction_error: f32,
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
    /// Moral violation category that triggered specific steering (empty when none).
    pub moral_steering_category: String,
    /// Codebook utilization rate (fraction of symbols retrieved recently, 0.0–1.0).
    pub codebook_utilization_rate: f32,
    /// FEP surprise-modulated replay batch size (0 when replay not triggered).
    pub surprise_replay_batch_size: usize,

    // ── Phase 15: Adaptive Architecture + Emotional Homeostasis ──────
    /// Whether attention budget was exceeded this cycle (subsystems skipped).
    pub attention_budget_exceeded: bool,
    /// Total cycle elapsed at budget check point (microseconds).
    pub attention_budget_elapsed_us: u64,
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
    /// Input similarity to previous cycle (cosine, 0.0–1.0).
    pub input_similarity: f32,
    /// Whether input memoization was used (skipped re-encoding).
    pub input_memoized: bool,
    /// Guiding question subsystem priority category (empty when no question active).
    pub guiding_priority_category: String,
    /// Total cycle wall-clock time in microseconds (same as CycleResult.cycle_time_us
    /// but included in metadata for unified telemetry access).
    pub cycle_duration_us: u64,
    /// Predicted Phi gain from school curriculum recommendation (0.0 when school disabled).
    pub school_predicted_phi_gain: f32,

    // ── Phase 16: Quality-Aware Adaptive Processing ─────────────────
    /// Whether epistemic gate confidence gated expensive modules this cycle.
    pub epistemic_coherence_gated: bool,
    /// Unified quality score (fusion of prediction coherence + agreement + anomaly).
    pub unified_quality_score: f32,
    /// Whether dissipative health gate dampened learning this cycle.
    pub dissipative_health_gated: bool,
    /// Dissipative health gate dampening factor applied (1.0 = no dampening).
    pub dissipative_lr_factor: f32,
    /// Phi validation correlation from most recent validation run (0.0 when not yet run).
    pub phi_validation_cached: f64,
    /// Adjusted spectral weight used in unified Psi computation.
    pub phi_spectral_weight: f32,
    /// Coherence velocity (rate of change, negative = dropping).
    pub coherence_velocity: f32,
    /// Whether temporal discontinuity triggered coherence gating this cycle.
    pub coherence_velocity_gated: bool,
    /// Metacognitive anomaly recovery progress (0.0–1.0, 1.0 = fully recovered).
    pub anomaly_recovery_progress: f32,
    /// Whether anomaly recovery is actively in progress.
    pub anomaly_recovering: bool,

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
    /// Empathic tone adjustment applied to speech rate (-1.0 to 1.0, 0.0 = no change).
    pub empathic_speech_rate_mod: f32,
    /// Value evaluator gate factor applied to learning rate (1.0 = no change).
    pub value_gate_factor: f32,
    /// Evolution phi delta feedback: confidence change applied this cycle.
    pub evolution_confidence_delta: f32,
    /// Homeostasis pull strength multiplier (urgency-adaptive, 1.0 = baseline).
    pub homeostasis_pull_strength: f32,
    /// Prediction coherence bias applied to urgency threshold (-1.0 to 1.0).
    pub prediction_coherence_urgency_bias: f32,

    // ── Phase 19: Activating Dormant Pathways ────────────────────────
    /// Whether attention budget gating skipped expensive subsystems this cycle.
    pub attention_budget_gated: bool,
    /// Consciousness limiting component that was boosted (empty when none).
    pub limiting_component_boosted: String,
    /// Love resonance confidence boost applied this cycle.
    pub love_resonance_boost: f32,
    /// Whether reasoning chain boosted prediction confidence this cycle.
    pub reasoning_chain_boosted: bool,
    /// AttentionShift motor command intensity applied to attention_sensitivity.
    pub attention_shift_applied: f32,

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

    // ── Mesh Network Telemetry ────────────────────────────────────────
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

    // ── Feedback State Telemetry (Phase 2.2) ────────────────────────────
    /// Number of confidence proposals collected this cycle.
    pub feedback_confidence_proposals: u32,
    /// Number of learning rate proposals collected this cycle.
    pub feedback_lr_proposals: u32,
    /// Per-proposal trace for confidence (populated when `trace_feedback = true`).
    /// Each entry: `(source_label, "Add(+0.0300)")`.
    pub feedback_trace_confidence: Vec<(String, String)>,
    /// Per-proposal trace for learning rate (populated when `trace_feedback = true`).
    pub feedback_trace_lr: Vec<(String, String)>,
    /// Number of exploration proposals collected this cycle.
    pub feedback_exploration_proposals: u32,
    /// Number of threshold proposals collected this cycle.
    pub feedback_threshold_proposals: u32,
    /// Per-proposal trace for exploration (populated when `trace_feedback = true`).
    pub feedback_trace_exploration: Vec<(String, String)>,
    /// Per-proposal trace for threshold (populated when `trace_feedback = true`).
    pub feedback_trace_threshold: Vec<(String, String)>,

    // ── Staged Computation Model Telemetry (Phase 2.3) ────────────────
    /// Number of subsystems that contributed non-neutral SubsystemOutputs.
    /// 0 = no subsystems using the new CognitiveSubsystem trait yet.
    pub subsystem_integration_contributors: u32,

    // ── Nurture Attachment Telemetry ─────────────────────────────────────
    /// Current attachment style (e.g., "Forming", "Secure"). Empty when nurture disabled.
    pub attachment_style: Option<String>,
    /// Current attachment security score (0.0–1.0). None when nurture disabled.
    pub attachment_security: Option<f64>,

    // ── Moral Topology (Persistent Homology) ────────────────────────────
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
    /// Dominant harmony axis index (0–6, maps to Harmony::all()).
    pub moral_topo_dominant_harmony: u8,
    /// Number of scenarios in the moral topology sliding window.
    pub moral_topo_scenario_count: usize,
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

// ═══════════════════════════════════════════════════════════════════════════════
// PHI ATTESTATION RECORD — for governance bridge consumption
// ═══════════════════════════════════════════════════════════════════════════════

/// Record of a Phi measurement from a cognitive cycle, ready for attestation.
///
/// The cognitive loop produces these after each cycle where Phi is computed.
/// The personal cluster (or symthaea-mycelix-bridge) can consume these records,
/// sign them with the agent's cryptographic key, and submit to governance as
/// authenticated `PsiAttestation` entries.
///
/// This is a lightweight struct with no external dependencies — the bridge crate
/// converts it to the Holochain-compatible `PsiAttestationData` format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsiAttestationRecord {
    /// Ψ — Consciousness estimate from this cycle (clamped to [0.0, 1.0])
    pub psi: f64,

    /// Cognitive cycle number that produced this Ψ
    pub cycle_id: u64,

    /// Timestamp in microseconds since UNIX epoch
    pub captured_at_us: u64,

    /// Prediction error at time of measurement (context for Phi quality)
    pub prediction_error: f32,

    /// Urgency level during measurement (Critical/Normal/Cruise)
    pub urgency: CycleUrgency,
}

/// Result of a single cognitive cycle
#[derive(Debug, Clone)]
pub struct CycleResult {
    /// LTC output (interpretation of current state)
    pub output: Vec<f32>,

    /// Prediction error for this cycle
    pub prediction_error: f32,

    /// Peak attention value from encoder
    pub peak_attention: f32,

    /// Detected primitives in input
    pub detected_primitives: Vec<String>,

    /// Whether learning occurred this cycle
    pub learning_occurred: bool,

    /// Training loss (if learning occurred)
    pub training_loss: Option<f32>,

    /// Cycle timing (microseconds)
    pub cycle_time_us: u64,

    /// Internal decision-making metadata for observability
    pub metadata: CycleMetadata,

    /// 32-dimensional projection of the thought hypervector for visualization
    pub thought_vector: Vec<f32>,

    /// The cycle's BinaryHV (16,384D) — the cognitive state distilled to a wisdom
    /// vector, suitable for mesh broadcast to peer nodes.
    pub wisdom_hv: symthaea_core::hdc::BinaryHV,

    /// Signed output for identity verification (when identity feature enabled)
    /// Contains Ed25519 signature over output hash and agent metadata
    #[cfg(feature = "identity")]
    pub signed_output: Option<crate::identity::SignedOutput>,

    /// Agent assurance level at time of processing (when identity feature enabled)
    #[cfg(feature = "identity")]
    pub assurance_level: crate::identity::AssuranceLevel,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MORAL JUDGMENT SUMMARY
// ═══════════════════════════════════════════════════════════════════════════════

/// Summary of moral evaluation for an action or input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoralJudgmentSummary {
    /// The input text that was evaluated
    pub input: String,

    /// Overall moral verdict (Good, Bad, Neutral, ConsentViolation)
    pub verdict: String,

    /// Deontological verdict (Permissible, Impermissible, Obligatory, Supererogatory)
    pub deontological_verdict: String,

    /// List of detected moral violations (e.g., "dishonesty", "theft")
    pub violations: Vec<String>,

    /// List of moral satisfactions (e.g., "honesty", "beneficence")
    pub satisfactions: Vec<String>,

    /// Whether consent was violated
    pub consent_violation: bool,

    /// Moral score (-1.0 to 1.0, negative = bad, positive = good)
    pub moral_score: f32,

    /// Confidence in the moral judgment (0.0 to 1.0)
    pub confidence: f32,
}

impl Default for MoralJudgmentSummary {
    fn default() -> Self {
        Self {
            input: String::new(),
            verdict: "Neutral".to_string(),
            deontological_verdict: "Permissible".to_string(),
            violations: Vec::new(),
            satisfactions: Vec::new(),
            consent_violation: false,
            moral_score: 0.0,
            confidence: 0.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ADAPTIVE BEHAVIOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Adaptive behavior parameters derived from consciousness state
///
/// These parameters allow the system to self-regulate based on its
/// current consciousness pattern, creating a truly self-aware loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct AdaptiveBehavior {
    /// Learning rate multiplier (0.1 to 2.0)
    /// Higher when focused/confident, lower when uncertain
    pub learning_rate_multiplier: f32,

    /// Speech rate multiplier (0.7 to 1.3)
    /// Slower when contemplative/uncertain, faster when excited/focused
    pub speech_rate_multiplier: f32,

    /// Pause duration multiplier (0.5 to 2.0)
    /// Longer pauses when contemplative, shorter when excited
    pub pause_multiplier: f32,

    /// Attention sensitivity (0.5 to 1.5)
    /// Higher sensitivity when exploratory, lower when focused
    pub attention_sensitivity: f32,

    /// Exploration factor (0.0 to 1.0)
    /// Higher when exploratory/uncertain, lower when focused
    pub exploration_factor: f32,

    /// Confidence level (0.0 to 1.0)
    /// Derived from pattern + coherence + voice quality
    pub confidence: f32,

    /// Should pause learning (e.g., during transitions)
    pub pause_learning: bool,

    /// Recommended action hint
    pub action_hint: ActionHint,
}

/// Recommended action based on consciousness state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionHint {
    /// Continue normally
    Continue,
    /// Slow down, be more deliberate
    SlowDown,
    /// Speed up, more confident
    SpeedUp,
    /// Pause and stabilize
    Stabilize,
    /// Explore alternatives
    Explore,
    /// Seek clarification or more input
    SeekInput,
}

impl Default for AdaptiveBehavior {
    fn default() -> Self {
        Self {
            learning_rate_multiplier: 1.0,
            speech_rate_multiplier: 1.0,
            pause_multiplier: 1.0,
            attention_sensitivity: 1.0,
            exploration_factor: 0.3,
            confidence: 0.5,
            pause_learning: false,
            action_hint: ActionHint::Continue,
        }
    }
}

impl AdaptiveBehavior {
    /// Compute adaptive behavior from consciousness pattern and metrics
    pub(crate) fn from_consciousness_state(
        pattern: crate::dynamics::temporal_signatures::ConsciousnessPattern,
        pattern_confidence: f32,
        coherence: f32,
        voice_confidence: f32,
    ) -> Self {
        use crate::dynamics::temporal_signatures::ConsciousnessPattern;

        // Base confidence from all sources
        let confidence =
            (pattern_confidence * 0.4 + coherence * 0.3 + voice_confidence * 0.3).clamp(0.0, 1.0);

        match pattern {
            ConsciousnessPattern::Focused => Self {
                learning_rate_multiplier: 1.3 + confidence * 0.4, // 1.3 to 1.7
                speech_rate_multiplier: 1.05 + confidence * 0.15, // 1.05 to 1.2
                pause_multiplier: 0.7,
                attention_sensitivity: 0.7, // Less distracted
                exploration_factor: 0.1,    // Stay on track
                confidence,
                pause_learning: false,
                action_hint: ActionHint::SpeedUp,
            },

            ConsciousnessPattern::Contemplative => Self {
                learning_rate_multiplier: 0.8,
                speech_rate_multiplier: 0.85,
                pause_multiplier: 1.5, // Longer pauses for reflection
                attention_sensitivity: 1.0,
                exploration_factor: 0.2,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::SlowDown,
            },

            ConsciousnessPattern::Excited => Self {
                learning_rate_multiplier: 1.1,
                speech_rate_multiplier: 1.15,
                pause_multiplier: 0.6,      // Quick transitions
                attention_sensitivity: 1.3, // More reactive
                exploration_factor: 0.4,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::Continue,
            },

            ConsciousnessPattern::Exploratory => Self {
                learning_rate_multiplier: 1.0,
                speech_rate_multiplier: 0.95,
                pause_multiplier: 1.0,
                attention_sensitivity: 1.4, // High sensitivity to new info
                exploration_factor: 0.7,    // Actively explore
                confidence: confidence * 0.8,
                pause_learning: false,
                action_hint: ActionHint::Explore,
            },

            ConsciousnessPattern::Resting => Self {
                learning_rate_multiplier: 0.6,
                speech_rate_multiplier: 0.9,
                pause_multiplier: 1.2,
                attention_sensitivity: 0.8,
                exploration_factor: 0.2,
                confidence,
                pause_learning: false,
                action_hint: ActionHint::Continue,
            },

            ConsciousnessPattern::Transitioning => Self {
                learning_rate_multiplier: 0.3, // Minimal learning during transition
                speech_rate_multiplier: 0.8,
                pause_multiplier: 1.8, // Pause to stabilize
                attention_sensitivity: 1.0,
                exploration_factor: 0.3,
                confidence: confidence * 0.5,
                pause_learning: true, // Pause learning
                action_hint: ActionHint::Stabilize,
            },

            ConsciousnessPattern::Uncertain => Self {
                learning_rate_multiplier: 0.4, // Careful learning
                speech_rate_multiplier: 0.75,  // Slow down
                pause_multiplier: 2.0,         // Long pauses
                attention_sensitivity: 1.2,
                exploration_factor: 0.5,
                confidence: confidence * 0.3,
                pause_learning: false,
                action_hint: ActionHint::SeekInput,
            },
        }
    }

    /// Get effective learning rate with all modulations
    pub(crate) fn effective_learning_rate(&self, base_rate: f32) -> f32 {
        if self.pause_learning {
            0.0
        } else {
            base_rate * self.learning_rate_multiplier
        }
    }

    /// Check if the system should seek more input/clarification
    pub(crate) fn should_seek_input(&self) -> bool {
        self.action_hint == ActionHint::SeekInput || self.confidence < 0.3
    }

    /// Check if the system is in a confident state
    pub(crate) fn is_confident(&self) -> bool {
        self.confidence > 0.6 && !self.pause_learning
    }

    /// Get a human-readable description of current state
    pub(crate) fn description(&self) -> &'static str {
        match self.action_hint {
            ActionHint::Continue => "Operating normally",
            ActionHint::SlowDown => "Deliberating carefully",
            ActionHint::SpeedUp => "Confident and focused",
            ActionHint::Stabilize => "Stabilizing state",
            ActionHint::Explore => "Exploring possibilities",
            ActionHint::SeekInput => "Seeking clarification",
        }
    }
}

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    // ═══════════════════════════════════════════════════════════════════════
    // CycleUrgency
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn urgency_default_is_normal() {
        assert_eq!(CycleUrgency::default(), CycleUrgency::Normal);
    }

    #[test]
    fn urgency_from_arousal_high() {
        assert_eq!(CycleUrgency::from_arousal(0.9), CycleUrgency::Critical);
    }

    #[test]
    fn urgency_from_arousal_mid() {
        assert_eq!(CycleUrgency::from_arousal(0.5), CycleUrgency::Normal);
    }

    #[test]
    fn urgency_from_arousal_low() {
        assert_eq!(CycleUrgency::from_arousal(0.1), CycleUrgency::Cruise);
    }

    #[test]
    fn urgency_from_state_surprise_triggers_critical() {
        let u = CycleUrgency::from_state(0.01, 0.05, true, 100);
        assert_eq!(u, CycleUrgency::Critical);
    }

    #[test]
    fn urgency_from_state_high_error_triggers_critical() {
        let u = CycleUrgency::from_state(0.2, 0.05, false, 100);
        assert_eq!(u, CycleUrgency::Critical);
    }

    #[test]
    fn urgency_from_state_moderate_error_is_normal() {
        let u = CycleUrgency::from_state(0.06, 0.05, false, 5);
        assert_eq!(u, CycleUrgency::Normal);
    }

    #[test]
    fn urgency_from_state_low_error_long_streak_is_cruise() {
        let u = CycleUrgency::from_state(0.01, 0.05, false, 20);
        assert_eq!(u, CycleUrgency::Cruise);
    }

    #[test]
    fn urgency_should_run_cycle_zero() {
        // All urgencies should run at cycle 0
        for urgency in [
            CycleUrgency::Critical,
            CycleUrgency::Normal,
            CycleUrgency::Cruise,
        ] {
            assert!(
                urgency.should_run(0, 3, 5, 20),
                "{:?} should run at cycle 0",
                urgency
            );
        }
    }

    #[test]
    fn urgency_should_run_interval_modulo() {
        let u = CycleUrgency::Normal;
        assert!(u.should_run(10, 3, 5, 20)); // cycle 10 % 5 == 0
        assert!(!u.should_run(11, 3, 5, 20)); // cycle 11 % 5 != 0
    }

    #[test]
    fn urgency_should_run_zero_interval_always_runs() {
        let u = CycleUrgency::Critical;
        assert!(u.should_run(42, 0, 5, 20)); // critical_interval=0 → always
    }

    #[test]
    fn urgency_consciousness_monitors() {
        assert!(CycleUrgency::Critical.run_consciousness_monitors());
        assert!(CycleUrgency::Normal.run_consciousness_monitors());
        assert!(!CycleUrgency::Cruise.run_consciousness_monitors());
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Carryover struct defaults
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn consciousness_cache_default_modulations_neutral() {
        let cc = ConsciousnessCache::default();
        assert_eq!(cc.predictive_phi_modulation, 1.0);
        assert_eq!(cc.body_phi_modulation, 1.0);
        assert_eq!(cc.embodied_phi_modulation, 1.0);
        assert_eq!(cc.cross_modal_psi, 0.0);
        assert!(cc.last_sigma.is_none());
    }

    #[test]
    fn urgency_state_default() {
        let us = UrgencyState::default();
        assert_eq!(us.urgency, CycleUrgency::Normal);
        assert_eq!(us.consecutive_low_error, 0);
        assert_eq!(us.arousal_trap_counter, 0);
        assert!(!us.anomaly_was_active);
        assert_eq!(us.mode_confidence, 1.0);
    }

    #[test]
    fn learning_state_default() {
        let ls = LearningState::default();
        assert_eq!(ls.prediction_confidence, 0.5);
        assert_eq!(ls.mce_lr_boost, 0.0);
        assert_eq!(ls.adaptive_threshold_scale, 1.0);
        assert_eq!(ls.subsystem_lr_factor, 1.0);
    }

    #[test]
    fn quality_metrics_default() {
        let qm = QualityMetrics::default();
        assert_eq!(qm.causal_chain_count, 0);
        assert_eq!(qm.temporal_continuity, 0.0);
        assert_eq!(qm.last_epistemic_confidence, 0.5);
        assert_eq!(qm.last_coherence, 0.5);
        assert!(!qm.narrative_veto_active);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CycleMetadata
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn cycle_metadata_default_sensible() {
        let md = CycleMetadata::default();
        assert!(!md.surprise_triggered);
        assert!(md.consciousness_level.is_finite());
    }

    #[test]
    fn cycle_metadata_serde_roundtrip() {
        let md = CycleMetadata::default();
        let json = serde_json::to_string(&md).expect("serialize CycleMetadata");
        let restored: CycleMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(restored.surprise_triggered, md.surprise_triggered);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MoralJudgmentSummary
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn moral_summary_default() {
        let mj = MoralJudgmentSummary::default();
        assert_eq!(mj.verdict, "Neutral");
        assert_eq!(mj.deontological_verdict, "Permissible");
        assert!(!mj.consent_violation);
        assert_eq!(mj.moral_score, 0.0);
        assert_eq!(mj.confidence, 0.0);
        assert!(mj.violations.is_empty());
        assert!(mj.satisfactions.is_empty());
    }

    #[test]
    fn moral_summary_serde_roundtrip() {
        let mj = MoralJudgmentSummary {
            input: "test action".into(),
            verdict: "Good".into(),
            deontological_verdict: "Permissible".into(),
            violations: vec![],
            satisfactions: vec!["honesty".into()],
            consent_violation: false,
            moral_score: 0.7,
            confidence: 0.9,
        };
        let json = serde_json::to_string(&mj).unwrap();
        let restored: MoralJudgmentSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.verdict, "Good");
        assert_eq!(restored.moral_score, 0.7);
        assert_eq!(restored.satisfactions.len(), 1);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // AdaptiveBehavior
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn adaptive_behavior_default_neutral() {
        let ab = AdaptiveBehavior::default();
        assert_eq!(ab.learning_rate_multiplier, 1.0);
        assert_eq!(ab.speech_rate_multiplier, 1.0);
        assert_eq!(ab.pause_multiplier, 1.0);
        assert_eq!(ab.action_hint, ActionHint::Continue);
        assert!(!ab.pause_learning);
    }

    #[test]
    fn adaptive_behavior_from_focused_pattern() {
        use crate::dynamics::temporal_signatures::ConsciousnessPattern;
        let ab = AdaptiveBehavior::from_consciousness_state(
            ConsciousnessPattern::Focused,
            0.8,
            0.7,
            0.6,
        );
        assert!(ab.learning_rate_multiplier > 1.0, "focused should boost LR");
        assert!(
            ab.speech_rate_multiplier > 1.0,
            "focused should speed up speech"
        );
        assert_eq!(ab.action_hint, ActionHint::SpeedUp);
    }

    #[test]
    fn adaptive_behavior_from_uncertain_pattern() {
        use crate::dynamics::temporal_signatures::ConsciousnessPattern;
        let ab = AdaptiveBehavior::from_consciousness_state(
            ConsciousnessPattern::Uncertain,
            0.3,
            0.3,
            0.2,
        );
        assert!(
            ab.learning_rate_multiplier < 1.0,
            "uncertain should reduce LR"
        );
        assert!(ab.confidence < 0.3, "uncertain should have low confidence");
        assert_eq!(ab.action_hint, ActionHint::SeekInput);
    }

    #[test]
    fn adaptive_behavior_transitioning_pauses_learning() {
        use crate::dynamics::temporal_signatures::ConsciousnessPattern;
        let ab = AdaptiveBehavior::from_consciousness_state(
            ConsciousnessPattern::Transitioning,
            0.5,
            0.5,
            0.5,
        );
        assert!(ab.pause_learning);
        assert_eq!(ab.action_hint, ActionHint::Stabilize);
    }

    #[test]
    fn adaptive_behavior_effective_lr_paused() {
        let mut ab = AdaptiveBehavior::default();
        ab.pause_learning = true;
        assert_eq!(ab.effective_learning_rate(0.01), 0.0);
    }

    #[test]
    fn adaptive_behavior_effective_lr_normal() {
        let ab = AdaptiveBehavior::default();
        let lr = ab.effective_learning_rate(0.01);
        assert!((lr - 0.01).abs() < 1e-6, "default multiplier is 1.0");
    }

    #[test]
    fn adaptive_behavior_should_seek_input() {
        let mut ab = AdaptiveBehavior::default();
        ab.confidence = 0.1; // Very low
        assert!(ab.should_seek_input());
    }

    #[test]
    fn adaptive_behavior_is_confident() {
        let mut ab = AdaptiveBehavior::default();
        ab.confidence = 0.8;
        assert!(ab.is_confident());
        ab.pause_learning = true;
        assert!(!ab.is_confident()); // paused learning = not confident
    }

    #[test]
    fn adaptive_behavior_description_coverage() {
        // Ensure all variants return non-empty strings
        for hint in [
            ActionHint::Continue,
            ActionHint::SlowDown,
            ActionHint::SpeedUp,
            ActionHint::Stabilize,
            ActionHint::Explore,
            ActionHint::SeekInput,
        ] {
            let mut ab = AdaptiveBehavior::default();
            ab.action_hint = hint;
            assert!(!ab.description().is_empty());
        }
    }

    #[test]
    fn adaptive_behavior_confidence_bounded() {
        use crate::dynamics::temporal_signatures::ConsciousnessPattern;
        // All patterns should produce confidence in [0, 1]
        for pattern in [
            ConsciousnessPattern::Focused,
            ConsciousnessPattern::Contemplative,
            ConsciousnessPattern::Excited,
            ConsciousnessPattern::Exploratory,
            ConsciousnessPattern::Resting,
            ConsciousnessPattern::Transitioning,
            ConsciousnessPattern::Uncertain,
        ] {
            let ab = AdaptiveBehavior::from_consciousness_state(pattern, 1.0, 1.0, 1.0);
            assert!(
                ab.confidence >= 0.0 && ab.confidence <= 1.0,
                "{:?}: confidence {} out of bounds",
                pattern,
                ab.confidence
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ActionHint serde
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn action_hint_serde_roundtrip() {
        for hint in [
            ActionHint::Continue,
            ActionHint::SlowDown,
            ActionHint::SpeedUp,
            ActionHint::Stabilize,
            ActionHint::Explore,
            ActionHint::SeekInput,
        ] {
            let json = serde_json::to_string(&hint).unwrap();
            let restored: ActionHint = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, hint);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ModuleTimings
    // ═══════════════════════════════════════════════════════════════════════

    #[test]
    fn module_timings_default_zeros() {
        let mt = ModuleTimings::default();
        assert_eq!(mt.core_hdc_encode, 0);
        assert_eq!(mt.core_cfc_step, 0);
        assert_eq!(mt.core_predict, 0);
        assert_eq!(mt.core_training, 0);
    }

    #[test]
    fn module_timings_serde_roundtrip() {
        let mt = ModuleTimings {
            core_hdc_encode: 100,
            core_cfc_step: 200,
            core_predict: 50,
            core_training: 1000,
            ..ModuleTimings::default()
        };
        let json = serde_json::to_string(&mt).unwrap();
        let restored: ModuleTimings = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.core_hdc_encode, 100);
        assert_eq!(restored.core_training, 1000);
    }
}
