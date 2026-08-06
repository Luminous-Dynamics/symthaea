// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Carryover types — state that crosses cycle boundaries.

use super::scheduling::CycleUrgency;

// ═══════════════════════════════════════════════════════════════════════════════
// CYCLE CARRYOVER — state that crosses cycle boundaries
// ═══════════════════════════════════════════════════════════════════════════════

/// Cached consciousness integration scores carried between cycles.
#[derive(Debug, Clone)]
pub struct ConsciousnessCache {
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
    /// Whether `SpectralMIPFinder::adapt()` has run at least once (added 2026-07-05,
    /// see `ConsciousnessEngineCache::last_spectral_mip_adapted`).
    pub(crate) last_spectral_mip_adapted: bool,
    /// Count of currently-tracked dimensions after the most recent `adapt()`, if any.
    pub(crate) last_spectral_mip_active_dim_count: Option<usize>,
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
    /// Last FluctuatioAnimi tension (cached for FEP exploration coupling).
    pub(crate) last_moral_fluctuatio_tension: f32,
    // ── Structural Phi persistence ──────────────────────────────────
    /// Last structural Phi result (updated every 194 cycles by consciousness engine).
    pub(crate) last_structural_phi:
        Option<symthaea_core::consciousness_metrics::StructuralPhiResult>,
    // ── MCE factor telemetry (updated each MCE firing) ──────────────
    /// MCE bottleneck factor name
    pub(crate) mce_bottleneck_name: String,
    /// MCE softmin value
    pub(crate) mce_softmin: f64,
    /// MCE weighted sum
    pub(crate) mce_weighted_sum: f64,
    /// MCE narrative coherence factor
    pub(crate) mce_narrative: f64,
    /// MCE social embedding factor
    pub(crate) mce_social: f64,
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
            last_spectral_mip_adapted: false,
            last_spectral_mip_active_dim_count: None,
            last_harmonic_coherence: 0.0,
            last_holographic_unity: 0.0,
            last_multimodal_phi: 0.0,
            last_equation_v2_consciousness: 0.0,
            last_hierarchical_mip_phi: None,
            last_embodied_agency: 0.5,
            last_predictive_free_energy: 0.0,
            last_moral_fluctuatio_tension: 0.0,
            last_structural_phi: None,
            mce_bottleneck_name: String::new(),
            mce_softmin: 0.0,
            mce_weighted_sum: 0.0,
            mce_narrative: 0.0,
            mce_social: 0.0,
        }
    }
}

/// Urgency state for adaptive subsystem scheduling.
#[derive(Debug, Clone)]
pub struct UrgencyState {
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
    pub(crate) prev_urgency: CycleUrgency,
    /// Cycles since last urgency mode change
    pub(crate) mode_stability_counter: u32,
    /// Consecutive temporal discontinuity cycles (Phase 21 recovery cascade).
    pub(crate) discontinuity_streak: u32,
    /// Remaining cycles of anomaly drift recovery (0 = not active).
    /// Science: Turrigiano (2008) — homeostatic plasticity engages for fixed duration.
    pub(crate) anomaly_drift_recovery: u32,
    /// Consecutive cycles with prediction coherence < 0.2.
    /// Science: Bar (2009) — sustained incoherence signals model failure, demands reallocation.
    pub(crate) consecutive_low_coherence: u32,
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
            consecutive_low_coherence: 0,
        }
    }
}

/// Learning rate modulation state.
#[derive(Debug, Clone)]
pub struct LearningState {
    /// Prediction confidence snapshot at cycle start (drift clamping)
    pub prediction_confidence: f64,
    /// MCE consciousness-level LR boost (decays 10%/cycle between MCE firings)
    pub mce_lr_boost: f32,
    /// Adaptive learning threshold multiplier (1.0 = config value as-is)
    pub adaptive_threshold_scale: f64,
    /// Subsystem LR modulation factor (accumulated post-training, consumed next cycle).
    pub subsystem_lr_factor: f32,
    // ── Phase 17: Predictive Self-Tuning ──────────────────────────────
    /// Self-model accuracy EMA (how well past predictions matched outcomes)
    pub self_model_accuracy: f32,
    /// Cognitive group geometric mean (flow x semantic x reasoning x curiosity)
    pub lr_cognitive_mod: f32,
    /// Meta-learning group geometric mean (FEP x MCE x subsystem)
    pub lr_meta_mod: f32,
}

impl Default for LearningState {
    fn default() -> Self {
        Self {
            prediction_confidence: 0.5_f64,
            mce_lr_boost: 0.0,
            adaptive_threshold_scale: 1.0,
            subsystem_lr_factor: 1.0,
            self_model_accuracy: 0.5,
            lr_cognitive_mod: 1.0,
            lr_meta_mod: 1.0,
        }
    }
}

/// Cached quality and diagnostic metrics.
#[derive(Debug, Clone)]
pub struct QualityMetrics {
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
    pub(crate) prev_confidence_for_crash: f64,
    /// Last moral score from perception phase (cached for neuromod feedback).
    /// Science: Zak (2012) — moral sentiment drives oxytocin/DA pathways.
    pub(crate) last_moral_score: f32,
    /// EMA-smoothed epistemic uncertainty (alpha=0.2, damps single-cycle noise).
    pub(crate) smoothed_epistemic_uncertainty: f32,
    /// Previous cycle's cross-module agreement (for velocity computation).
    pub(crate) prev_cross_module_agreement: f32,
    /// Consecutive cycles where all 4 feedback channels were dampened.
    /// Science: Turrigiano (2008) — sustained dampening triggers synaptic silencing.
    pub(crate) consecutive_full_dampen: u32,
    /// Homeostasis pull efficiency EMA (ratio of post/pre distance, alpha=0.2).
    /// <1.0 = pulls working, >1.0 = overcorrecting.
    pub(crate) homeostasis_efficiency: f32,
    /// Remaining cycles of LR freeze after a confidence crash (0 = not active).
    /// Science: Cools et al. (2008) — rapid confidence collapse triggers protective freeze.
    pub(crate) crash_freeze_remaining: u32,
    /// Hysteresis relaxation factor (1.0 = full, decays toward HYSTERESIS_RELAXATION_FLOOR).
    /// Science: Kelso (1995) — sustained stability permits relaxed mode boundaries.
    pub(crate) hysteresis_factor: f32,
    /// Last exploration bonus applied to learning rate.
    /// Modulated by user frustration (dampen) and flow state (boost).
    pub(crate) last_exploration_bonus: f32,
    /// Consecutive cycles with zero harmonic interferences (for recovery grace period).
    pub(crate) interference_free_cycles: u32,
    /// Consecutive cycles with low FEP TD error (for convergence detection).
    pub(crate) consecutive_low_td_error: u32,
    /// Consecutive cycles with high unified quality (for exploration floor).
    pub(crate) consecutive_high_quality: u32,
    /// Consecutive cycles where epistemic gate rejected (Session 16).
    /// Science: Berlyne (1960) — sustained rejection = model failure.
    pub(crate) consecutive_epistemic_rejections: u32,
    /// Consecutive cycles with near-zero consciousness gradient (Session 16).
    /// Science: Tononi (2004) — stable consciousness = reliable integration.
    pub(crate) consecutive_stable_gradient: u32,
    /// Last KosmicSong coherence score (0.0-1.0, cached from last synthesis).
    /// Synthesis of Phi × HarmonicAlignment × MoralClarity.
    pub(crate) last_kosmic_coherence: f32,
    // ── Session 17: Adaptive Homeostasis ──────────────────────────────────
    /// Allostatic load (0.0–1.0): cumulative stress burden.
    pub(crate) allostatic_load: f32,
    /// Previous consciousness gradient magnitude (for 2nd derivative).
    pub(crate) prev_gradient_magnitude: f64,
    /// Whether adaptive warmup has exited (stability-based).
    pub(crate) adaptive_warmup_exited: bool,
    // ── Session 18: Predictive Coding & Metacognitive Refinement ────────
    /// EMA of prediction error squared (variance tracking).
    pub(crate) pe_variance_ema: f32,
    /// Confidence calibration: running sum of (predicted - actual) over window.
    pub(crate) confidence_calibration_bias: f32,
    /// Confidence calibration: count of samples in current window.
    pub(crate) confidence_calibration_count: u32,
    /// Whether a subsystem VETO_ACTION flag was set this cycle.
    /// Reset at cycle start. When true, motor output should be suppressed.
    pub(crate) subsystem_veto: bool,
    /// Safety enforcement: motor halt (Red level). Set in Phase 3.5.
    /// When true, ALL motor output must be blocked — both embodiment and file I/O.
    pub(crate) safety_motor_halt: bool,
    /// Safety enforcement: motor read-only (Orange level). Set in Phase 3.5.
    /// When true, only read-only motor output is permitted.
    pub(crate) safety_motor_readonly: bool,
    /// LR momentum: EMA of recent effective LR.
    pub(crate) lr_momentum_ema: f32,
    /// Previous metacognitive prediction (expected consciousness level).
    pub(crate) prev_metacognitive_prediction: f64,
    /// Sleep pressure (0.0–1.0): accumulated synaptic load.
    pub(crate) sleep_pressure: f32,
    /// Whether currently in consolidation mode.
    pub(crate) in_consolidation: bool,
    // ── Session 19: Embodied Cognition & Environmental Coupling ─────────
    /// Last computed readiness score (0.3–1.0), for telemetry.
    pub(crate) last_readiness_score: f32,
    /// Novelty EMA (0.0–1.0): how novel recent inputs are.
    pub(crate) novelty_ema: f32,
    /// Fatigue (0.0–1.0): cognitive resource depletion.
    pub(crate) fatigue: f32,
    /// Consecutive low-effort stable cycles (for recovery detection).
    pub(crate) consecutive_recovery_cycles: u32,
    /// Consecutive cycles with high cross-module agreement (for flow detection).
    pub(crate) consecutive_high_agreement: u32,
    /// Whether currently in flow/resonance state.
    pub(crate) in_flow_state: bool,
    // ── Knowledge Engine: Working Memory Injection ─────────────────────
    /// Grounding quality of knowledge facts injected into working memory (Baddeley 2000).
    pub(crate) wm_knowledge_grounding: f64,
    /// Number of knowledge facts injected into working memory this cycle.
    pub(crate) wm_knowledge_injection_count: u8,
    // ── Epistemic Cube Tiers ─────────────────────────────────────────
    /// Last computed empirical tier (0–4).
    pub(crate) last_cube_e_tier: Option<u8>,
    /// Last computed normative tier (0–3).
    pub(crate) last_cube_n_tier: Option<u8>,
    /// Last computed materiality tier (0–3).
    pub(crate) last_cube_m_tier: Option<u8>,
    /// Last computed Harmony-value (0.0–1.0).
    pub(crate) last_cube_h_value: f32,
    /// Last computed cube quality composite.
    pub(crate) last_cube_quality: f32,
    /// Last vision manifold HDC dimension (16384 or 65536).
    pub(crate) last_vision_hdc_dim: u32,
    /// Last visual Variational Free Energy.
    pub(crate) last_vision_free_energy: f32,
    /// Last visual model complexity.
    pub(crate) last_vision_complexity: f32,
    /// Last visual prediction accuracy.
    pub(crate) last_vision_accuracy: f32,
    /// Whether a geodesic mental simulation was requested last cycle.
    pub(crate) last_request_geodesic: bool,
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
            last_moral_score: 0.0,
            smoothed_epistemic_uncertainty: 0.0,
            prev_cross_module_agreement: 0.5,
            consecutive_full_dampen: 0,
            homeostasis_efficiency: 1.0,
            crash_freeze_remaining: 0,
            hysteresis_factor: 1.0,
            last_exploration_bonus: 1.0,
            interference_free_cycles: 0,
            consecutive_low_td_error: 0,
            consecutive_high_quality: 0,
            consecutive_epistemic_rejections: 0,
            consecutive_stable_gradient: 0,
            last_kosmic_coherence: 0.5,
            allostatic_load: 0.0,
            prev_gradient_magnitude: 0.0,
            adaptive_warmup_exited: false,
            pe_variance_ema: 0.0,
            confidence_calibration_bias: 0.0,
            confidence_calibration_count: 0,
            subsystem_veto: false,
            safety_motor_halt: false,
            safety_motor_readonly: false,
            lr_momentum_ema: 1.0,
            prev_metacognitive_prediction: 0.0,
            sleep_pressure: 0.0,
            in_consolidation: false,
            last_readiness_score: 1.0,
            novelty_ema: 0.5,
            fatigue: 0.0,
            consecutive_recovery_cycles: 0,
            consecutive_high_agreement: 0,
            in_flow_state: false,
            wm_knowledge_grounding: 0.0,
            wm_knowledge_injection_count: 0,
            last_cube_e_tier: None,
            last_cube_n_tier: None,
            last_cube_m_tier: None,
            last_cube_h_value: 0.0,
            last_cube_quality: 0.0,
            last_vision_hdc_dim: 16384,
            last_vision_free_energy: 0.0,
            last_vision_complexity: 0.0,
            last_vision_accuracy: 1.0,
            last_request_geodesic: false,
        }
    }
}

/// Which code path last wrote [`CycleHistory::consciousness_level`], the scalar the
/// robotics motor-safety gate consumes.
///
/// Deliberately **not** named `ConsciousnessLevelKind` or modelled as variants of a
/// single `ConsciousnessLevel` type: the two writers share no inputs, and naming them
/// variants of one thing would assert exactly the coherence that is currently in
/// question. This enum names *writers*, not *kinds of one quantity*.
///
/// See `SYMTHAEA_COGNITIVE_CORE_RECONCILIATION_PLAN_2026-07-28.md`, Phase 4 correction
/// box (2026-07-29).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GateWriter {
    /// Never written this run — still the cold-start floor (0.05), which is a prior,
    /// not a measurement.
    #[default]
    ColdStartFloor,
    /// `cycle_late_consciousness/integration.rs` on the `cycle(&str)` path:
    /// `max(MCE(..max(unified_psi, spectral_boost, structural_boost)..), v2_cached * 0.8)`.
    TextComposite,
    /// `helpers/mod.rs` on the `cycle_with_hv()` path:
    /// `compute_consciousness_level(prediction_confidence, coherence, flow_intensity,
    /// pattern_confidence)` — contains no Phi, no Psi, and no spectral MIP.
    HvConfidenceCoherence,
}

impl GateWriter {
    /// Short label for telemetry/experiment output.
    pub fn label(self) -> &'static str {
        match self {
            GateWriter::ColdStartFloor => "cold-start-floor",
            GateWriter::TextComposite => "text-composite",
            GateWriter::HvConfidenceCoherence => "hv-confidence-coherence",
        }
    }
}

/// What the robotics safety path **actually consumed**, recorded at the point the tier
/// is selected — as distinct from what a writer *produced*.
///
/// Required by Amendment 1 (A1.1) of
/// `SYMTHAEA_PHASE4_CHARACTERIZATION_PROTOCOL_2026-07-29.md`, which made
/// consumption-boundary instrumentation a blocking precondition for any characterization
/// run. Write-side provenance alone cannot establish what the safety path received: a
/// value can be stale, re-clamped at an intermediate layer, misaligned by a cycle,
/// cached, per-platform overridden, or defaulted.
///
/// Join with the write side on [`Self::cycle_index`]. **A produced/consumed mismatch is a
/// primary finding**, not a measurement error — it would mean the characterized quantity
/// is not the governing quantity.
#[derive(Debug, Clone, Copy)]
pub struct GateConsumption {
    /// `stats.total_cycles` at the moment of consumption. Join key.
    pub cycle_index: usize,
    /// The scalar as actually received by the embodiment boundary.
    pub consumed_value: f64,
    /// Tier selected from `consumed_value` by the real production classifier
    /// (`MotorSafetyLevel::from_phi`), never a reimplemented ladder.
    pub resulting_tier: symthaea_core::embodiment::MotorSafetyLevel,
    /// Which platform consumed it.
    pub platform: symthaea_core::embodiment::EmbodimentPlatform,
    /// Provenance of the consumed value: which formula produced it.
    pub writer: GateWriter,
    /// `stats.total_cycles` when that value was written.
    pub written_at: usize,
}

impl GateConsumption {
    /// Cycles between production and consumption. **Expected to be ≥1**: the embodiment
    /// block runs in PHASE 2, before the feedback phase that rewrites the gate field, so
    /// the gate consumes the *previous* cycle's value. A lag exceeding the 67-cycle
    /// refresh interval means the tier was selected from a value no recent measurement
    /// produced.
    pub fn lag_cycles(&self) -> usize {
        self.cycle_index.saturating_sub(self.written_at)
    }
}

/// Historical state for cycle-to-cycle continuity.
#[derive(Debug, Clone)]
pub struct CycleHistory {
    /// MCTS plan action (action_idx, confidence) for next cycle
    pub(crate) mcts_plan: Option<(usize, f32)>,
    /// Body arousal (fed back into CfC tau modulation)
    pub(crate) body_arousal: f32,
    /// Resonance frequency (fed back into delta_t modulation)
    pub(crate) resonance_frequency: f64,
    /// The scalar consumed as the robotics motor-safety gate (`cycle.rs` ->
    /// `EmbodimentBridge::step(.., phi)`), and by robotics telemetry dispatch,
    /// `motor_phi`, prediction confidence, `phi_trend`, and the Psi autoregression.
    ///
    /// **This field has two writers using unrelated formulas** — see
    /// [`GateWriter`]. Its previous doc ("Last MCE consciousness level") was
    /// accurate only for the text path. Read
    /// `SYMTHAEA_COGNITIVE_CORE_RECONCILIATION_PLAN_2026-07-28.md`'s Phase 4
    /// correction box before treating this as a single coherent quantity.
    pub(crate) consciousness_level: f64,
    /// Which formula produced the current [`Self::consciousness_level`].
    ///
    /// Instrumentation only — nothing gates on it. Added 2026-07-29 to make the
    /// two-writer ambiguity *observable* before deciding how to resolve it,
    /// rather than picking a formula by fiat.
    pub(crate) consciousness_level_source: GateWriter,
    /// `stats.total_cycles` at the moment [`Self::consciousness_level`] was last
    /// written. Lets a consumer see staleness, which matters because the two
    /// writers fire on different entry points and at different rates.
    pub(crate) consciousness_level_written_at: usize,
    /// Recent BinaryHV ring buffer for multi-component consciousness profile.
    /// Bounded: capacity 4, evict-before-push via pop_front (cycle_consciousness.rs).
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
    /// Rolling window of recent prediction errors.
    /// Bounded: capacity 16, evict via while-loop pop_front (cycle_phases_urgency.rs).
    pub(crate) error_history: std::collections::VecDeque<f32>,
    /// Self-model prediction: (cycle_made, predicted_confidence, predicted_urgency)
    pub(crate) self_model_prediction: Option<(usize, f64, CycleUrgency)>,
    /// Cached coherence value for this cycle (computed once, reused everywhere).
    pub(crate) cached_coherence: Option<f32>,
    /// EMA of consciousness level for adaptive consolidation gating.
    /// Tracks rolling average so consolidation threshold adapts to the system's
    /// typical consciousness range rather than using a hardcoded 0.5.
    pub(crate) consciousness_ema: f64,
}

impl Default for CycleHistory {
    fn default() -> Self {
        Self {
            mcts_plan: None,
            body_arousal: 0.5,
            resonance_frequency: 0.0,
            consciousness_level: 0.05, // Floor: prevents fully unconscious cold-start
            consciousness_level_source: GateWriter::ColdStartFloor,
            consciousness_level_written_at: 0,
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
            consciousness_ema: 0.0,
        }
    }
}

/// State carried over between consecutive cognitive cycles.
///
/// These fields represent the "memory" of the previous cycle that influences
/// the next cycle's processing. All fields are reset to defaults by
/// `CognitiveLoopService::reset()`.
#[derive(Debug, Clone, Default)]
pub struct CycleCarryover {
    /// Cached consciousness integration scores
    pub consciousness: ConsciousnessCache,
    /// Urgency scheduling state
    pub urgency: UrgencyState,
    /// Learning rate modulation
    pub learning: LearningState,
    /// Cached quality/diagnostic metrics
    pub quality: QualityMetrics,
    /// Historical state for continuity
    pub history: CycleHistory,
    /// Whether GWT broadcast occurred in the previous cycle
    pub gwt_broadcast_occurred: bool,
    /// GWT winning coalition size from previous cycle (0 if no broadcast)
    pub gwt_coalition_size: u32,
    /// Injected code reasoning context from CodingAgent (one-shot: consumed by cycle).
    #[cfg(feature = "reasoning_engine")]
    pub injected_code_context: Option<crate::consciousness::reasoning_engine::CodeReasoningContext>,
}
