//! Feedback phase, homeostasis, arousal, emotional regulation, and adaptive intelligence constants.

// ═══════════════════════════════════════════════════════════════════════════════
// DOMINANCE / EMOTIONAL REGULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Base dominance when in flow state.
/// Basis: Csikszentmihalyi (1990) — flow produces a sense of mastery.
pub const DOMINANCE_FLOW_BASE: f64 = 0.6;

/// Dominance scaling from flow state (additive on top of base).
pub const DOMINANCE_FLOW_SCALE: f64 = 0.2;

/// Dominance when prediction confidence is high.
pub const DOMINANCE_CONFIDENT: f64 = 0.4;

/// Default dominance (no strong signal).
pub const DOMINANCE_DEFAULT: f64 = 0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// STRATEGY MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Exploratory strategy LR dampening factor (multiplicative).
/// Basis: Hills (2015) — exploration reduces learning commitment to current path.
pub const STRATEGY_EXPLORATORY_FACTOR: f64 = 0.8;

/// Detailed strategy attention sensitivity boost (multiplicative).
pub const STRATEGY_DETAILED_SENSITIVITY: f64 = 1.2;

/// Concise strategy speech rate boost (multiplicative).
pub const STRATEGY_CONCISE_SPEECH_RATE: f64 = 1.2;

/// Clarifying strategy factor (multiplicative on confidence threshold).
pub const STRATEGY_CLARIFYING_FACTOR: f64 = 0.5;

/// Supportive strategy pause boost (multiplicative on speech rate).
pub const STRATEGY_SUPPORTIVE_PAUSE: f64 = 1.3;

// ═══════════════════════════════════════════════════════════════════════════════
// VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

// Validate all threshold ordering and non-overlap constraints.
//
// Panics with a descriptive message if any invariant is violated.
// Called once at startup (via `CognitiveLoopService::new()`) to catch
// configuration errors early.
//
// # Invariants
//
// 1. `MORAL_CONCERN_THRESHOLD < 0 < MORAL_BENEFIT_THRESHOLD`
// 2. `FEP_LR_DECAY ∈ (0, 1)` (must actually decay)
// 3. `POLICY_SOFT_THRESHOLD ∈ (0, 1)` (valid probability)
// ═══════════════════════════════════════════════════════════════════════════════
// FEEDBACK PHASE — CONSCIOUSNESS LIMITING & PIPELINE GATES
// ═══════════════════════════════════════════════════════════════════════════════

/// Consciousness gradient magnitude threshold for limiting-component boost.
/// Below this, gradient is too small to warrant targeted intervention.
/// Basis: Dehaene (2014) — minimal gradient for workspace ignition.
pub const LIMITING_COMPONENT_GRADIENT_THRESHOLD: f64 = 0.01;

/// Confidence delta applied when binding is the limiting consciousness component.
pub const LIMITING_BINDING_CONFIDENCE_DELTA: f64 = 0.01;

/// Social learning rate change threshold — below this, skip modulation.
/// Prevents noise from sub-epsilon trust fluctuations.
pub const SOCIAL_LR_CHANGE_THRESHOLD: f64 = 0.01;

/// Minimum consecutive interference-free cycles before harmonic all-clear boost.
/// Basis: Kelso (1995) — stability requires sustained absence of perturbation.
pub const HARMONIC_INTERFERENCE_FREE_CYCLES: u32 = 3;

/// Pipeline consciousness high threshold — above this, relax epistemic caution.
/// Basis: Dehaene (2014) — global workspace ignition requires integrated processing.
pub const PIPELINE_CONSCIOUSNESS_RELAX: f64 = 0.7;

/// Pipeline consciousness low threshold — below this, tighten caution.
pub const PIPELINE_CONSCIOUSNESS_CAUTION: f64 = 0.3;

/// Threshold scale factor when pipeline consciousness is high (relaxation).
pub const PIPELINE_CONSCIOUSNESS_RELAX_SCALE: f64 = 0.97;

/// Threshold scale factor when pipeline consciousness is low (tightening).
pub const PIPELINE_CONSCIOUSNESS_CAUTION_SCALE: f64 = 1.03;

/// Support predictive telemetry interval (cycles). Co-prime with subsystem intervals.
pub const SUPPORT_TELEMETRY_INTERVAL: u64 = 47;

/// Support graduation check interval (cycles). Co-prime with subsystem intervals.
pub const SUPPORT_GRADUATION_INTERVAL: u64 = 97;

/// Sacred Stillness harmony index — must match `Harmony::SacredStillness` enum order.
pub const HARMONY_INDEX_SACRED_STILLNESS: usize = 7;

// ═══════════════════════════════════════════════════════════════════════════════
// AROUSAL TRAP STATE MACHINE
// Science: Yerkes-Dodson (1908) — inverted-U performance curve; sustained
// high arousal degrades performance. Porges (2011) — polyvagal recovery.
// ═══════════════════════════════════════════════════════════════════════════════

/// Arousal threshold for LR suppression (moderate arousal dampens learning).
pub const AROUSAL_TRAP_SUPPRESS_THRESHOLD: f64 = 0.7;

/// Scale factor for arousal LR suppression. At arousal=1.0: (1.0-0.7)*0.25 = 0.075.
pub const AROUSAL_TRAP_SUPPRESS_SCALE: f64 = 0.25;

/// Maximum arousal LR suppression (cap).
pub const AROUSAL_TRAP_SUPPRESS_MAX: f64 = 0.08;

/// Arousal threshold for trap detection (high arousal increments counter).
pub const AROUSAL_TRAP_DETECT_THRESHOLD: f64 = 0.8;

/// Counter threshold for entering Phase 2 (active recovery).
pub const AROUSAL_TRAP_RECOVERY_ENTER: u32 = 5;

/// Counter threshold for entering Phase 3 (forced escape).
pub const AROUSAL_TRAP_ESCAPE_ENTER: u32 = 10;

/// LR dampening scale during recovery (Phase 2).
pub const AROUSAL_TRAP_RECOVERY_LR_SCALE: f64 = 0.1;

/// Exploration boost scale during recovery (Phase 2).
pub const AROUSAL_TRAP_RECOVERY_EXPLORE_SCALE: f64 = 0.025;

/// Confidence scale during forced escape (Phase 3).
pub const AROUSAL_TRAP_ESCAPE_CONFIDENCE_SCALE: f64 = 0.9;

/// Arousal threshold below which consolidation LR boost activates.
/// Steriade (1996): low arousal enhances memory consolidation.
pub const AROUSAL_LOW_CONSOLIDATION_THRESHOLD: f64 = 0.3;

/// Scale factor for low-arousal consolidation boost.
pub const AROUSAL_LOW_CONSOLIDATION_SCALE: f64 = 0.3;

/// Maximum consolidation boost from low arousal.
pub const AROUSAL_LOW_CONSOLIDATION_MAX: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// ACTIVE REST / SACRED STILLNESS
// ═══════════════════════════════════════════════════════════════════════════════

/// Consecutive cycles of Sacred Stillness dominance before active rest mode engages.
/// Science: Tononi & Cirelli (2006) — sustained rest duration determines consolidation depth.
pub const ACTIVE_REST_THRESHOLD: u16 = 10;

/// Cycles a pending calibration can wait without sleep before forced application.
/// Science: McEwen (1998) — allostatic load accumulates when corrective actions are
/// deferred indefinitely. Systems that never sleep still need calibration maintenance.
/// At 50Hz, 2000 cycles ≈ 40 seconds of continuous wakefulness.
pub const ALWAYS_AWAKE_STALE_CYCLES: u64 = 2000;

// ═══════════════════════════════════════════════════════════════════════════════
// EMOTIONAL HOMEOSTASIS
// ═══════════════════════════════════════════════════════════════════════════════

/// Homeostasis pull multiplier during Cruise urgency (relaxed → stronger pull).
/// Basis: Russell (2003) — circumplex model neutral return.
pub const HOMEOSTASIS_PULL_CRUISE: f64 = 1.5;

/// Homeostasis pull multiplier during Normal urgency.
pub const HOMEOSTASIS_PULL_NORMAL: f64 = 1.0;

/// Homeostasis pull multiplier during Critical urgency (weaker → allow reactive emotion).
pub const HOMEOSTASIS_PULL_CRITICAL: f64 = 0.6;

/// Arousal target for homeostasis (slightly above neutral).
pub const HOMEOSTASIS_AROUSAL_TARGET: f64 = 0.3;

/// Emotional inertia strength: dampens rapid valence/arousal swings by
/// blending previous-cycle momentum into the homeostasis pull.
/// Sokolov (1963): habituation creates resistance to rapid stimulus-response shifts.
/// Range: 0.0 (no inertia, instant snap) to 0.5 (strong momentum preservation).
pub const HOMEOSTASIS_EMOTIONAL_INERTIA: f64 = 0.15;

/// Consecutive all-dampen cycles before feedback consensus freezes to base values.
/// Turrigiano (2008): homeostatic plasticity includes brief synaptic silencing.
pub const CONSENSUS_FREEZE_STREAK_THRESHOLD: u32 = 3;

/// Affective valence momentum: blends previous-cycle valence into current stimulus
/// processing, creating smooth affective trajectories instead of discontinuous jumps.
/// Damasio (1999): somatic markers persist across temporal gaps; emotional state has inertia.
/// Range: 0.0 (no momentum) to 0.5 (strong persistence). Default 0.15 = 15% prior.
pub const AFFECTIVE_VALENCE_MOMENTUM: f64 = 0.15;

/// Cross-modal binding temporal smoothing: blends previous-cycle cross-modal Psi
/// into current binding result, preventing rapid binding/unbinding oscillations.
/// Engel et al. (2001): temporal coherence in cross-modal binding requires sustained
/// synchronization across perceptual cycles.
/// Range: 0.0 (no smoothing) to 0.5 (strong hysteresis). Default 0.2 = 20% prior.
pub const CROSS_MODAL_PSI_TEMPORAL_SMOOTHING: f64 = 0.2;

/// Predictive phi modulation smoothing: blends previous-cycle predictive modulation
/// into current result, ensuring precision estimates evolve gradually.
/// Friston (2010): precision-weighted prediction errors should update smoothly
/// to maintain hierarchical model stability.
/// Range: 0.0 (no smoothing) to 0.5 (strong persistence). Default 0.15 = 15% prior.
pub const PREDICTIVE_PHI_MODULATION_SMOOTHING: f64 = 0.15;

// ═══════════════════════════════════════════════════════════════════════════════
// USER STATE INFERENCE (USI) FEEDBACK
// Science: Picard (2000) — affect-aware systems must modulate behavior based on
// inferred user state to maintain productive human-AI collaboration.
// ═══════════════════════════════════════════════════════════════════════════════

/// User frustration threshold above which exploration is dampened.
/// Lazarus (1991): frustration signals goal-blockage; reducing exploration prevents
/// compounding the user's sense of uncontrollability.
pub const USI_FRUSTRATION_EXPLORATION_THRESHOLD: f64 = 0.6;

/// Scale factor for frustration-driven exploration dampening.
/// At max frustration (1.0), exploration is scaled to 1.0 - (1.0 - 0.6) * 0.25 = 0.9.
pub const USI_FRUSTRATION_EXPLORATION_SCALE: f64 = 0.25;

/// User cognitive load threshold above which learning rate is reduced.
/// Sweller (1988): high cognitive load impairs integration; reducing LR
/// avoids overwhelming the user with rapid adaptation.
pub const USI_COGNITIVE_LOAD_LR_THRESHOLD: f64 = 0.7;

/// Scale factor for load-driven LR reduction.
/// At max load (1.0), LR is scaled to 1.0 - (1.0 - 0.7) * 0.3 = 0.91.
pub const USI_COGNITIVE_LOAD_LR_SCALE: f64 = 0.3;

/// User engagement threshold above which confidence is boosted.
/// Csikszentmihalyi (1990): high engagement signals flow-compatible state;
/// boosting confidence reinforces successful resonance.
pub const USI_ENGAGEMENT_CONFIDENCE_THRESHOLD: f64 = 0.8;

/// Scale factor for engagement-driven confidence boost.
/// At max engagement (1.0), confidence boost = (1.0 - 0.8) * 0.05 = 0.01.
pub const USI_ENGAGEMENT_CONFIDENCE_SCALE: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// HOMEOSTASIS EFFICIENCY
// ═══════════════════════════════════════════════════════════════════════════════

/// EMA alpha for homeostasis efficiency tracking.
/// Basis: Cannon (1929)/Ashby (1960) — homeostatic regulation monitoring.
pub const HOMEOSTASIS_EFFICIENCY_EMA: f64 = 0.2;

/// Transition cost threshold above which homeostasis is strengthened.
/// Basis: Kelso (1995) — costly transitions increase attractor persistence.
pub const TRANSITION_COST_THRESHOLD: f64 = 0.1;

/// Maximum transition cost effect on homeostasis strength.
pub const TRANSITION_COST_MAX_EFFECT: f64 = 0.2;

/// Transition cost → homeostasis strength scaling.
pub const TRANSITION_COST_STRENGTH_SCALE: f64 = 1.5;

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 10: ADAPTIVE FEEDBACK INTELLIGENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Confidence crash threshold: >30% drop in 1 cycle triggers emergency stabilization.
/// Science: Cools et al. (2008) — rapid serotonergic dip from confidence collapse.
pub const CONFIDENCE_CRASH_THRESHOLD: f64 = 0.30;

/// Number of cycles to freeze LR after a confidence crash.
/// Science: Turrigiano (2008) — transient plasticity shutdown for homeostatic recovery.
pub const CONFIDENCE_CRASH_FREEZE_CYCLES: u32 = 3;

/// Exploration boost applied during crash recovery.
/// Science: Daw et al. (2006) — uncertainty-triggered shift to model-free exploration.
pub const CONFIDENCE_CRASH_EXPLORATION_BOOST: f64 = 0.04;

/// Self-model accuracy threshold above which self-model gets +5% proposal weight.
/// Science: Friston (2010) — high self-model accuracy justifies increased self-trust.
pub const SELF_MODEL_WEIGHT_HIGH_THRESHOLD: f64 = 0.7;

/// Self-model accuracy threshold below which self-model gets -10% proposal weight.
pub const SELF_MODEL_WEIGHT_LOW_THRESHOLD: f64 = 0.3;

/// Self-model high accuracy → confidence weight bonus.
pub const SELF_MODEL_WEIGHT_BONUS: f64 = 0.05;

/// Self-model low accuracy → confidence weight penalty (multiplicative).
pub const SELF_MODEL_WEIGHT_PENALTY: f64 = 0.90;

/// Coherence velocity → CfC tau modulation: rising coherence factor.
/// Science: Buzsáki (2006) — coherent oscillations slow integration (stability).
pub const COHERENCE_VELOCITY_TAU_BOOST: f64 = 1.05;

/// Coherence velocity → CfC tau modulation: falling coherence factor.
/// Falling coherence → faster integration to explore corrections.
pub const COHERENCE_VELOCITY_TAU_DAMPEN: f64 = 0.95;

/// Minimum coherence velocity to trigger tau modulation.
pub const COHERENCE_VELOCITY_TAU_THRESHOLD: f64 = 0.02;

/// Homeostasis efficiency high threshold → reduce pull strength 15%.
/// Science: Ashby (1960) — efficient regulation should self-attenuate.
pub const HOMEOSTASIS_EFFICIENCY_HIGH: f64 = 1.2;

/// Homeostasis efficiency low threshold → increase pull strength 10%.
pub const HOMEOSTASIS_EFFICIENCY_LOW: f64 = 0.8;

/// Pull reduction factor when efficiency is high (multiplicative).
pub const HOMEOSTASIS_PULL_REDUCTION: f64 = 0.85;

/// Pull increase factor when efficiency is low (multiplicative).
pub const HOMEOSTASIS_PULL_INCREASE: f64 = 1.10;

/// Error pattern → predictive LR scale for Rising pattern.
/// Science: Schultz (2016) — increasing errors demand faster adaptation.
pub const ERROR_PATTERN_RISING_LR: f64 = 1.05;

/// Error pattern → predictive LR scale for Falling pattern.
pub const ERROR_PATTERN_FALLING_LR: f64 = 0.97;

/// Error pattern → predictive LR scale for Oscillating pattern.
/// Science: Doya (2002) — oscillating errors indicate meta-uncertainty.
pub const ERROR_PATTERN_OSCILLATING_LR: f64 = 0.90;

/// Minimum distinct proposal sources after warmup (for diversity metric).
/// Science: Dehaene (2014) — GWT requires multi-source consensus.
pub const PROPOSAL_DIVERSITY_MIN_SOURCES: usize = 3;

/// Warmup cycles before diversity check applies.
pub const PROPOSAL_DIVERSITY_WARMUP: usize = 30;

/// Exploration boost when proposal diversity is too low.
pub const PROPOSAL_DIVERSITY_EXPLORATION_BOOST: f64 = 0.02;

/// Mode stability counter threshold for hysteresis relaxation.
/// Science: Kelso (1995) — sustained stability permits relaxed mode boundaries.
pub const HYSTERESIS_RELAXATION_THRESHOLD: u32 = 20;

/// Rate at which hysteresis relaxes toward baseline per cycle (multiplicative).
pub const HYSTERESIS_RELAXATION_RATE: f64 = 0.98;

/// Minimum hysteresis factor after relaxation (prevents collapse).
pub const HYSTERESIS_RELAXATION_FLOOR: f64 = 0.5;

/// Cross-module agreement velocity threshold for confidence velocity coupling.
/// Science: Tononi (2004) — agreement rise with confidence fall indicates
/// subsystems converging but output not reflecting it → needs gentle correction.
pub const AGREEMENT_CONFIDENCE_COUPLING_THRESHOLD: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// REST / DREAM MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Coherence weight during active rest (>1.0 = emphasize coherence).
/// Science: Tononi & Cirelli (2006) — rest-state consciousness emphasizes integration.
pub const REST_COHERENCE_WEIGHT: f64 = 1.2;

/// Binding intensity dampen during active rest (<1.0 = reduce binding weight).
pub const REST_BINDING_DAMPEN: f64 = 0.8;

/// Fraction of rest modulation attributed to coherence component.
pub const REST_MODULATION_COHERENCE_FRAC: f64 = 0.6;

/// Fraction of rest modulation attributed to binding component.
pub const REST_MODULATION_BINDING_FRAC: f64 = 0.4;

/// Dream consolidation reliability threshold for LR boost.
/// Diekelmann & Born (2010): effective consolidation enhances subsequent encoding.
pub const DREAM_CONSOLIDATION_LR_THRESHOLD: f64 = 0.6;

/// Maximum LR boost from high dream consolidation reliability.
/// Walker (2017): post-sleep learning enhancement factor.
pub const DREAM_CONSOLIDATION_LR_BOOST: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 16: BIDIRECTIONAL FEEDBACK DEEPENING
// ═══════════════════════════════════════════════════════════════════════════════

/// Temporal binding strength below which exploration is boosted.
/// Basis: Buzsáki (2002) — weak theta binding → poor temporal integration → explore.
pub const TEMPORAL_BINDING_EXPLORE_THRESHOLD: f64 = 0.2;

/// Temporal binding strength above which exploration is dampened.
/// Basis: Buzsáki (2002) — strong theta binding → stable temporal model → exploit.
pub const TEMPORAL_BINDING_DAMPEN_THRESHOLD: f64 = 0.6;

/// Exploration boost scale for low temporal binding.
pub const TEMPORAL_BINDING_EXPLORE_SCALE: f64 = 0.03;

/// LR dampening scale for low temporal binding.
pub const TEMPORAL_BINDING_LOW_LR_SCALE: f64 = 0.97;

/// Consciousness gradient magnitude above which stability recovery triggers.
/// Basis: Oizumi et al. (2014) — large gradient = rapid consciousness change → caution.
pub const CONSCIOUSNESS_GRADIENT_CAUTION_THRESHOLD: f64 = 0.1;

/// LR dampening for high consciousness gradient.
pub const CONSCIOUSNESS_GRADIENT_LR_SCALE: f64 = 0.97;

/// Confidence recovery for near-zero gradient + stable consciousness.
/// Basis: Tononi (2004) — stable integration = reliable metrics → trust more.
pub const CONSCIOUSNESS_GRADIENT_RECOVERY_BOOST: f64 = 0.005;

/// Minimum cycles of near-zero gradient before confidence recovery.
pub const CONSCIOUSNESS_GRADIENT_STABLE_CYCLES: u32 = 20;

/// Startup exploration ramp: initial exploration fraction during early warmup.
/// Basis: Hopfield (1982) — settling time requires constrained exploration.
pub const STARTUP_EXPLORATION_INITIAL: f64 = 0.3;

/// Consecutive epistemic gate rejections before recalibration triggers.
/// Basis: Berlyne (1960) — sustained rejection = systematic model failure → recalibrate.
pub const EPISTEMIC_REJECTION_STREAK_THRESHOLD: u32 = 5;

/// Exploration boost when epistemic rejection streak triggers recalibration.
pub const EPISTEMIC_REJECTION_STREAK_EXPLORE: f64 = 0.05;

/// Threshold relaxation when epistemic rejection streak triggers recalibration.
pub const EPISTEMIC_REJECTION_STREAK_THRESHOLD_RELAX: f64 = 0.95;

/// Consecutive full-dampen cycles before protective threshold freeze.
/// Basis: Turrigiano (2008) — sustained dampening = synaptic silencing → protect.
pub const FULL_DAMPEN_FREEZE_THRESHOLD: u32 = 5;

/// Consciousness EMA threshold above which LR gets a startup bias boost.
/// Basis: Dehaene (2014) — integrated processing supports faster learning.
pub const CONSCIOUSNESS_EMA_HIGH_THRESHOLD: f64 = 0.5;

/// LR scale for high consciousness EMA.
pub const CONSCIOUSNESS_EMA_LR_BOOST: f64 = 1.03;

/// Consciousness EMA threshold below which LR gets a startup bias dampening.
pub const CONSCIOUSNESS_EMA_LOW_THRESHOLD: f64 = 0.2;

/// LR scale for low consciousness EMA.
pub const CONSCIOUSNESS_EMA_LR_DAMPEN: f64 = 0.97;

/// Multi-objective frontier size above which exploration is boosted.
/// Basis: Deb (2002) — large Pareto frontier = many competing objectives → explore.
pub const MULTI_OBJ_FRONTIER_LARGE: usize = 5;

/// Multi-objective frontier exploration boost scale.
pub const MULTI_OBJ_FRONTIER_EXPLORE_SCALE: f64 = 0.02;

/// Multi-objective frontier size at which exploration is dampened (converged).
pub const MULTI_OBJ_FRONTIER_SMALL: usize = 1;

/// Multi-objective frontier convergence exploration dampening.
pub const MULTI_OBJ_FRONTIER_DAMPEN: f64 = 0.98;

/// Error oscillation ratio above which bifurcation response triggers.
/// Basis: Kelso (1995) — high oscillation at phase transition → freeze and observe.
pub const ERROR_OSCILLATION_BIFURCATION: f64 = 0.7;

/// LR freeze during bifurcation (multiplicative).
pub const ERROR_OSCILLATION_BIFURCATION_LR: f64 = 0.9;

/// Exploration boost during bifurcation.
pub const ERROR_OSCILLATION_BIFURCATION_EXPLORE: f64 = 0.03;

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 17: ADAPTIVE HOMEOSTASIS & EMERGENT DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Allostatic load decay per cycle.
/// Basis: McEwen (1998) — allostatic load accumulates under stress, decays during recovery.
pub const ALLOSTATIC_LOAD_DECAY: f64 = 0.995;
/// Allostatic load increment per dampening event.
pub const ALLOSTATIC_LOAD_INCREMENT: f64 = 0.02;
/// Allostatic load threshold above which cognitive budget is reduced.
/// Basis: McEwen (2007) — allostatic overload impairs cognition.
pub const ALLOSTATIC_OVERLOAD_THRESHOLD: f64 = 0.6;
/// LR dampening during allostatic overload.
pub const ALLOSTATIC_OVERLOAD_LR_SCALE: f64 = 0.95;
/// Exploration boost during allostatic overload (seek recovery).
pub const ALLOSTATIC_OVERLOAD_EXPLORE_BOOST: f64 = 0.02;
/// Exploration decay factor per cycle.
/// Basis: Sutton & Barto (2018) — epsilon decay prevents exploration drift.
pub const EXPLORATION_DECAY_FACTOR: f64 = 0.998;
/// Exploration baseline that decay converges toward.
pub const EXPLORATION_DECAY_BASELINE: f64 = 0.5;
/// Consciousness gradient acceleration threshold (2nd derivative).
/// Basis: Friston (2010) — generalized coordinates track higher-order dynamics.
pub const CONSCIOUSNESS_ACCEL_THRESHOLD: f64 = 0.05;
/// LR dampening for high consciousness acceleration.
pub const CONSCIOUSNESS_ACCEL_LR_SCALE: f64 = 0.95;
/// Confidence dampening for high consciousness acceleration.
pub const CONSCIOUSNESS_ACCEL_CONFIDENCE_SCALE: f64 = 0.98;
/// Adaptive warmup exit: consciousness gradient below this = stable.
/// Basis: Smith (2018) — super-convergence via stability-based warmup exit.
pub const ADAPTIVE_WARMUP_GRADIENT_THRESHOLD: f64 = 0.02;
/// Adaptive warmup exit: consciousness EMA must exceed this.
pub const ADAPTIVE_WARMUP_EMA_THRESHOLD: f64 = 0.1;
/// Minimum warmup cycles before adaptive exit can trigger.
pub const ADAPTIVE_WARMUP_MIN_CYCLES: u32 = 15;
/// Maximum proposal sources per channel before saturation triggers.
/// Basis: Simon (1956) — bounded rationality under information overload.
pub const PROPOSAL_SATURATION_THRESHOLD: u32 = 8;
/// LR dampening during proposal saturation (wait-and-see).
pub const PROPOSAL_SATURATION_LR_SCALE: f64 = 0.97;
/// Phi threshold below which LR floor is raised.
/// Basis: Tononi (2004) — low Phi = fragmented → cautious learning.
pub const PHI_GATED_LR_FLOOR_THRESHOLD: f64 = 0.1;
/// LR dampening when Phi is below floor threshold.
pub const PHI_GATED_LR_FLOOR_SCALE: f64 = 0.93;
/// Rhythmic exploration oscillation period in cycles.
/// Basis: Lisman & Jensen (2013) — theta-gamma coupling alternates explore/exploit.
pub const RHYTHMIC_EXPLORATION_PERIOD: u32 = 100;
/// Rhythmic exploration oscillation amplitude.
pub const RHYTHMIC_EXPLORATION_AMPLITUDE: f64 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 18: PREDICTIVE CODING & METACOGNITIVE REFINEMENT
// ═══════════════════════════════════════════════════════════════════════════════

/// EMA decay for prediction error variance tracking.
/// Basis: Clark (2013) — prediction error precision weighting.
pub const PE_VARIANCE_EMA_DECAY: f64 = 0.95;
/// PE variance threshold above which LR is dampened.
/// High variance = noisy environment → cautious learning.
pub const PE_VARIANCE_DAMPING_THRESHOLD: f64 = 0.3;
/// LR scale when PE variance exceeds threshold.
pub const PE_VARIANCE_LR_SCALE: f64 = 0.95;

/// Confidence calibration window (cycles).
/// Basis: Keren & Teigen (2004) — metacognitive calibration.
pub const CONFIDENCE_CALIBRATION_WINDOW: usize = 50;
/// Confidence calibration drift threshold (overconfident or underconfident).
pub const CONFIDENCE_CALIBRATION_DRIFT_THRESHOLD: f64 = 0.15;
/// Confidence correction scale per cycle when drift detected.
pub const CONFIDENCE_CALIBRATION_CORRECTION: f64 = 0.01;

/// EMA decay for learning rate momentum.
/// Basis: Kingma & Ba (2015) — Adam-style momentum smoothing.
pub const LR_MOMENTUM_EMA_DECAY: f64 = 0.9;
/// Maximum LR change per cycle (dampens abrupt swings).
pub const LR_MOMENTUM_MAX_DELTA: f64 = 0.15;

/// Metacognitive surprise threshold (predicted vs actual outcome divergence).
/// Basis: Fleming & Dolan (2012) — metacognitive prediction errors.
pub const METACOGNITIVE_SURPRISE_THRESHOLD: f64 = 0.2;
/// Exploration boost on metacognitive surprise.
pub const METACOGNITIVE_SURPRISE_EXPLORE_BOOST: f64 = 0.03;

/// Sleep pressure increment per active cycle.
/// Basis: Tononi & Cirelli (2006) — synaptic homeostasis hypothesis.
pub const SLEEP_PRESSURE_INCREMENT: f64 = 0.001;
/// Sleep pressure threshold that triggers consolidation mode.
pub const SLEEP_PRESSURE_THRESHOLD: f64 = 0.7;
/// Sleep pressure decay rate during consolidation.
pub const SLEEP_PRESSURE_CONSOLIDATION_DECAY: f64 = 0.05;
/// Passive sleep pressure decay during low-stress periods (micro-rest).
/// Basis: Lim & Dinges (2010) — brief rest epochs partially clear adenosine.
/// Applied when readiness > 0.9 (system not under significant load).
pub const SLEEP_PRESSURE_PASSIVE_DECAY: f64 = 0.999;
/// LR dampening during consolidation mode.
pub const SLEEP_PRESSURE_LR_SCALE: f64 = 0.9;

/// Window for gradient sign consistency tracking.
/// Basis: Schaul et al. (2013) — sign-based learning rate adaptation.
pub const GRADIENT_SIGN_WINDOW: usize = 10;
/// Fraction of sign flips above which = oscillating.
pub const GRADIENT_SIGN_FLIP_THRESHOLD: f64 = 0.6;
/// LR dampening when gradient is oscillating.
pub const GRADIENT_SIGN_FLIP_LR_SCALE: f64 = 0.95;

/// Exploration-exploitation balance window.
/// Basis: Cohen et al. (2007) — tonic dopamine sets explore/exploit balance.
pub const EXPLORE_EXPLOIT_WINDOW: usize = 50;
/// Acceptable imbalance range (0.3–0.7 = balanced).
pub const EXPLORE_EXPLOIT_LOW_BOUND: f64 = 0.3;
/// Upper bound of acceptable balance.
pub const EXPLORE_EXPLOIT_HIGH_BOUND: f64 = 0.7;
/// Homeostatic correction strength.
pub const EXPLORE_EXPLOIT_CORRECTION: f64 = 0.01;

/// Proposal conflict: max conflict_ratio before cancellation.
/// Basis: Botvinick et al. (2001) — ACC detects response conflict → pause.
pub const PROPOSAL_CONFLICT_THRESHOLD: f64 = 0.4;

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 19: EMBODIED COGNITION & ENVIRONMENTAL COUPLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Arousal threshold for LR boost (high arousal = salient moment).
/// Basis: Yerkes-Dodson (1908) — inverted-U arousal-performance curve.
pub const AROUSAL_LR_BOOST_THRESHOLD: f64 = 0.6;
/// LR scale when arousal exceeds threshold.
pub const AROUSAL_LR_BOOST_SCALE: f64 = 1.05;
/// Arousal ceiling above which LR is dampened (over-arousal).
pub const AROUSAL_OVERAROUSAL_THRESHOLD: f64 = 0.85;
/// LR scale during over-arousal.
pub const AROUSAL_OVERAROUSAL_LR_SCALE: f64 = 0.93;

/// Novelty habituation EMA decay.
/// Basis: Sokolov (1963) — orienting response habituates to repeated stimuli.
pub const NOVELTY_EMA_DECAY: f64 = 0.95;
/// Novelty threshold below which exploration is reduced.
pub const NOVELTY_LOW_THRESHOLD: f64 = 0.1;
/// Exploration dampening when novelty is low.
pub const NOVELTY_LOW_EXPLORE_SCALE: f64 = 0.98;

/// Fatigue increment per high-effort cycle (many proposals or high LR delta).
/// Basis: Hockey (1997) — cognitive resource depletion theory.
pub const FATIGUE_INCREMENT: f64 = 0.005;
/// Fatigue threshold for effort detection (total proposals > this).
pub const FATIGUE_EFFORT_THRESHOLD: usize = 12;
/// LR dampening under fatigue.
pub const FATIGUE_LR_SCALE: f64 = 0.97;
/// Fatigue threshold above which dampening fires.
pub const FATIGUE_THRESHOLD: f64 = 0.5;

/// Recovery: consecutive low-effort stable cycles needed.
/// Basis: Meijman & Mulder (1998) — effort-recovery model.
/// S21: Reduced from 8 → 5 — old value meant recovery almost never triggered.
pub const RECOVERY_CYCLES_NEEDED: u32 = 5;
/// Fatigue decay per recovery cycle.
/// S21: Increased from 0.03 → 0.08 — halves fatigue in ~9 recovery cycles instead of ~23.
pub const RECOVERY_FATIGUE_DECAY: f64 = 0.08;
/// Confidence boost on full recovery.
pub const RECOVERY_CONFIDENCE_BOOST: f64 = 0.01;

/// Environmental predictability window.
/// Basis: Gottlieb et al. (2013) — information sampling as exploration control.
pub const ENV_PREDICTABILITY_WINDOW: usize = 30;
/// Threshold scale tightening when environment is predictable.
pub const ENV_PREDICTABLE_THRESHOLD_SCALE: f64 = 0.98;
/// Threshold scale loosening when environment is unpredictable.
pub const ENV_UNPREDICTABLE_THRESHOLD_SCALE: f64 = 1.02;
/// Predictability boundary for "predictable" (>= this).
pub const ENV_PREDICTABILITY_HIGH: f64 = 0.7;
/// Predictability boundary for "unpredictable" (<= this).
pub const ENV_PREDICTABILITY_LOW: f64 = 0.3;

/// Maximum attention budget (total proposals per cycle).
/// Basis: Kahneman (1973) — capacity model of attention.
pub const ATTENTION_BUDGET_MAX: usize = 20;

/// Readiness components: weight of PE variance in readiness.
/// Basis: Boksem & Tops (2008) — mental fatigue and cognitive control.
pub const READINESS_PE_WEIGHT: f64 = 0.3;
/// Readiness weight of sleep pressure.
pub const READINESS_SLEEP_WEIGHT: f64 = 0.3;
/// Readiness weight of fatigue.
pub const READINESS_FATIGUE_WEIGHT: f64 = 0.4;

/// Resonance detection: consecutive high-agreement cycles needed.
/// Basis: Csikszentmihalyi (1990) — flow state via absorbed engagement.
pub const RESONANCE_FLOW_CYCLES: u32 = 10;
/// Cross-module agreement threshold for resonance.
pub const RESONANCE_AGREEMENT_THRESHOLD: f64 = 0.8;
/// Confidence boost during flow/resonance state.
pub const RESONANCE_CONFIDENCE_BOOST: f64 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 22+: MISSING CONSTANTS (added to fix compilation)
// ═══════════════════════════════════════════════════════════════════════════════

/// Multiplicative dampener for allostatic load increment when consecutive full-dampen
/// cycles are detected. Values < 1.0 reduce the penalty (Science: adaptive resilience).
pub const ALLOSTATIC_LOAD_DAMPEN_INCREMENT_SCALE: f64 = 0.5;

/// Weight for attention schema (AST) encoding contribution to compressed state.
/// Science: Graziano (2013) — top-down attention schema modulates perception.
pub const AST_ENCODING_WEIGHT: f64 = 0.1;

/// LR boost multiplier when coherence is degraded (coherence < stability floor).
/// Science: Rescorla-Wagner (1972) — prediction error drives learning.
pub const COHERENCE_DEGRADED_LR_BOOST: f64 = 1.05;

/// Exploration adjustment when conceptual confusion is detected.
/// Science: Berlyne (1960) — incongruity increases exploratory drive.
pub const CONCEPTUAL_CONFUSION_EXPLORATION: f64 = 0.03;

/// Number of consecutive high-quality cycles before floor exploration boost.
/// Science: Csikszentmihalyi (1990) — sustained quality enables exploration.
pub const CONSECUTIVE_HIGH_QUALITY_CYCLES: u32 = 10;

/// Confidence penalty when escalation is throttled (mild constraint violation).
pub const ESCALATION_THROTTLE_CONFIDENCE: f64 = 0.02;

/// Exploration penalty when escalation is throttled.
pub const ESCALATION_THROTTLE_EXPLORATION: f64 = 0.02;

/// Confidence penalty when escalation is hard-blocked (severe constraint violation).
pub const ESCALATION_BLOCK_CONFIDENCE: f64 = 0.05;

/// Exploration penalty when escalation is hard-blocked.
pub const ESCALATION_BLOCK_EXPLORATION: f64 = 0.05;

/// LR scale when escalation is hard-blocked (slow down learning during conflict).
pub const ESCALATION_BLOCK_LR_SCALE: f64 = 0.95;

/// Midpoint for exploration bias detection (0.5 = neutral).
/// Science: explore/exploit balance theory.
pub const EXPLORE_BIAS_MIDPOINT: f64 = 0.5;

/// Fatigue level below which the system is considered recovered.
/// Science: recovery threshold after exertion (Hess 1944 sleep-wake theory).
pub const FATIGUE_RECOVERED_THRESHOLD: f64 = 0.1;

/// High FEP accuracy threshold for confidence boost.
/// Science: Friston (2010) — accurate prediction reduces uncertainty.
pub const FEP_ACCURACY_HIGH_CONFIDENCE: f64 = 0.03;

/// Confidence boost when FEP is operating efficiently.
pub const FEP_EFFICIENT_CONFIDENCE: f64 = 0.02;

/// Exploration scale when FEP TD learning has converged.
/// Science: Sutton & Barto (1998) — TD convergence signals exploitation.
pub const FEP_TD_CONVERGE_EXPLORE_SCALE: f64 = 0.98;

/// Flow intensity threshold above which LR is boosted.
/// Science: Csikszentmihalyi (1990) — flow state enhances learning efficiency.
pub const FLOW_INTENSITY_LR_THRESHOLD: f64 = 0.7;

/// LR boost multiplier when in flow state.
pub const FLOW_SUBSYSTEM_LR_BOOST: f64 = 1.05;

/// Scale factor for goal priority → exploration adjustment.
/// Science: Botvinick et al. (2009) — goal-directed attention modulates exploration.
pub const GOAL_PURSUIT_EXPLORATION_SCALE: f64 = 0.05;

/// Gradient magnitude below which prediction is considered OK (stable).
pub const GRADIENT_PREDICTION_OK_THRESHOLD: f64 = 0.05;

/// Gradient magnitude below which the system is considered in a stable attractor.
pub const GRADIENT_STABLE_DETECT_THRESHOLD: f64 = 0.03;

/// Quality score above which a cycle is considered high quality.
pub const HIGH_QUALITY_SCORE_THRESHOLD: f64 = 0.75;

/// Causal depth above which exploitation bias activates (depth > threshold → exploit).
pub const KNOWLEDGE_CAUSAL_DEPTH_EXPLOIT_THRESHOLD: f64 = 3.0;

/// Dampener for exploration urge when causal depth exceeds exploit threshold.
pub const KNOWLEDGE_CAUSAL_DEPTH_EXPLORE_DAMPEN: f64 = 0.1;

/// Exploration boost when knowledge contradiction detected with negation evidence.
pub const KNOWLEDGE_CONTRADICTION_NE_BOOST: f64 = 0.02;

/// Exploration boost when knowledge contradiction detected with serotonin-linked evidence.
pub const KNOWLEDGE_CONTRADICTION_SHT_BOOST: f64 = 0.01;

/// Basis: Ebbinghaus (1885) — periodic consolidation preserves long-term knowledge.
/// How often (in cycles) the knowledge manager persists to SQLite.
pub const KNOWLEDGE_SAVE_INTERVAL: u64 = 500;

/// Basis: Barsalou (2008), Clark (2013) — grounded cognition modulates consciousness.
/// Knowledge grounding ±modulation factor on unified consciousness score.
pub const KNOWLEDGE_CONSCIOUSNESS_MODULATION: f64 = 0.05;

/// Basis: Anderson & Schooler (1991) — power law of forgetting; low-confidence items pruned.
/// Confidence threshold below which non-causal facts are pruned during dream consolidation.
pub const KNOWLEDGE_FORGET_CONFIDENCE_THRESHOLD: f64 = 0.1;

/// Basis: Stickgold & Walker (2013) — sleep strengthens causal memory traces.
/// Confidence boost applied to causal facts during dream consolidation.
pub const KNOWLEDGE_CONSOLIDATION_BOOST: f64 = 0.05;

/// NE baseline boost per unit of knowledge uncertainty above 0.5 (Bouret & Sara 2005).
pub const KNOWLEDGE_UNCERTAINTY_NE_SCALE: f64 = 0.015;
/// DA baseline nudge when causal chain depth exceeds threshold (Schultz 1997).
pub const KNOWLEDGE_CAUSAL_DEPTH_DA_NUDGE: f64 = 0.01;
/// 5-HT baseline nudge for grounded, low-uncertainty knowledge (Cools et al. 2008).
pub const KNOWLEDGE_GROUNDING_SHT_NUDGE: f64 = 0.005;
/// Exploration boost scale for novel knowledge signals (novelty > 0.5).
pub const KNOWLEDGE_NOVELTY_EXPLORE_SCALE: f64 = 0.08;
/// Salience boost for knowledge facts promoted to episodic memory during dreams.
pub const KNOWLEDGE_EPISODIC_SALIENCE_BOOST: f64 = 0.15;
/// Max facts to promote per dream consolidation cycle.
pub const KNOWLEDGE_EPISODIC_MAX_PER_DREAM: usize = 5;
/// Minimum similarity for AGM-style contradiction resolution (demote weaker fact).
pub const KNOWLEDGE_CONTRADICTION_RESOLUTION_THRESHOLD: f64 = 0.8;

/// Blend weight for knowledge grounding in epistemic quality computation.
/// Knowledge grounding contributes this fraction to effective_epistemic in ConsciousnessEquationV2.
/// Science: Mercier & Sperber (2017) — argumentative theory of reasoning; grounded knowledge
/// strengthens epistemic claims by anchoring them in verified factual content.
pub const KNOWLEDGE_GROUNDING_EPISTEMIC_BLEND: f64 = 0.3;

/// Weight of knowledge coherence in the consciousness Knowledge core component.
/// Knowledge coherence (graph size + calibration + contradiction-free) nudges the
/// effective epistemic quality by this fraction of the coherence score.
/// Small weight keeps it subordinate to grounding and epistemic quality.
/// Science: Stanovich (2009) — epistemic rationality correlates with reflective judgment
/// quality; well-calibrated, contradiction-free knowledge bases support deeper reflection.
pub const KNOWLEDGE_COHERENCE_CONSCIOUSNESS_WEIGHT: f64 = 0.1;

/// Log-scale divisor for normalising graph size in the knowledge coherence formula.
/// log2(graph_size + 1) / LOG_SCALE: at ~1000 facts the contribution saturates near 1.0
/// (log2(1024) = 10.0). Larger knowledge bases beyond this get diminishing returns.
/// Science: Miller (1956) — working memory capacity limits; beyond ~1000 chunks, marginal
/// value of additional facts for consciousness integration decreases.
pub const KNOWLEDGE_COHERENCE_LOG_SCALE: f64 = 10.0;

/// Dream consolidation weight boost when knowledge contradictions are active.
/// Contradictions signal unresolved cognitive dissonance requiring offline integration.
/// Science: Festinger (1957) — cognitive dissonance drives memory consolidation to resolve conflict.
pub const DREAM_KNOWLEDGE_CONTRADICTION_BOOST: f64 = 0.2;

/// Dream consolidation weight boost when causal chain depth exceeds threshold.
/// Deep causal reasoning chains are structurally important and worth preserving.
/// Science: Hobson & Friston (2012) — active inference in dreams consolidates causal models.
pub const DREAM_KNOWLEDGE_CAUSAL_DEPTH_BOOST: f64 = 0.1;

/// Minimum causal depth to trigger dream consolidation boost.
/// Chains shorter than this are too shallow to warrant preferential consolidation.
/// Science: Hobson & Friston (2012) — only multi-step causal chains benefit from dream replay.
pub const DREAM_KNOWLEDGE_CAUSAL_DEPTH_THRESHOLD: f64 = 3.0;

/// Dream→Knowledge consolidation weight for strengthening replayed edges.
/// Rasch & Born (2013): sleep replay strengthens memory traces proportional to replay intensity.
pub const DREAM_KNOWLEDGE_REPLAY_WEIGHT: f64 = 0.1;

/// Minimum dream consolidation quality to trigger knowledge update.
/// Tononi & Cirelli (2006): only effective consolidation modifies synaptic weights.
pub const DREAM_KNOWLEDGE_MIN_QUALITY: f64 = 0.4;

/// Attention boost when knowledge contradictions exceed salience threshold.
/// Contradictions signal prediction errors requiring focused examination.
/// Science: Clark (2013) — predictive processing allocates attention to prediction error sources.
pub const KNOWLEDGE_ATTENTION_CONTRADICTION_BOOST: f64 = 0.3;

/// Minimum contradiction signal to trigger attention reallocation.
/// Below this threshold, contradictions are not salient enough to warrant focused attention.
pub const KNOWLEDGE_ATTENTION_CONTRADICTION_THRESHOLD: f64 = 0.5;

/// Minimum causal edge strength to flag as a confounding variable.
/// Science: Confounders with weak links are noise; strong links warrant attention (Pearl 2009).
pub const KNOWLEDGE_CONFOUNDER_STRENGTH_THRESHOLD: f64 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// FEEDBACK PHASE — CONSCIOUSNESS GATING & EPISTEMIC MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Consciousness gradient stability threshold (below = stable, skip modulation).
/// Basis: Dehaene (2014) — ignition requires meaningful gradient.
pub const CONSCIOUSNESS_GRADIENT_STABILITY_THRESHOLD: f64 = 0.01;

/// Attention sensitivity boost for high efficacy.
/// Basis: Bandura (1997) — self-efficacy boosts attentional engagement.
pub const EFFICACY_ATTENTION_BOOST: f64 = 1.05;

/// LR boost for high efficacy.
pub const EFFICACY_LR_BOOST: f64 = 1.05;

/// HOT hubris dampening factor for moral confidence.
/// Basis: Kruger & Dunning (1999) — overconfidence invites moral miscalibration.
pub const HOT_HUBRIS_CONFIDENCE_DAMPEN: f64 = 0.7;

/// Knowledge reasoning log scale divisor for contradiction factor.
pub const KNOWLEDGE_REASONING_LOG_SCALE: f64 = 10.0;

/// Knowledge contradiction factor denominator for LR modulation.
pub const KNOWLEDGE_CONTRADICTION_FACTOR_DENOM: f64 = 0.1;

/// Scale boost sigmoid amplitude.
/// Basis: Tononi (2004) — multi-scale integration follows sigmoid enrichment.
pub const SCALE_BOOST_SIGMOID_AMPLITUDE: f64 = 0.15;

/// Scale boost sigmoid exponent.
pub const SCALE_BOOST_SIGMOID_EXPONENT: f64 = 2.0;

/// Reasoning reliability center (confidence offset for reasoning gate).
pub const REASONING_RELIABILITY_CENTER: f64 = 0.5;

/// Reasoning gate success LR scale.
pub const REASONING_GATE_SUCCESS_LR_SCALE: f64 = 1.02;

/// Social model turn-taking quality default.
pub const SOCIAL_MODEL_TURN_TAKING_DEFAULT: f64 = 0.7;

/// Social model trust threshold for mode selection.
pub const SOCIAL_MODEL_TRUST_MODE_THRESHOLD: f64 = 0.3;

/// Coherence-trust center threshold.
pub const COHERENCE_TRUST_CENTER: f64 = 0.5;

/// Moral score normalization offset.
pub const MORAL_SCORE_NORMALIZE_OFFSET: f64 = 1.0;

/// Moral score normalization scale (maps [-1,1] → [0,1]).
pub const MORAL_SCORE_NORMALIZE_SCALE: f64 = 0.5;

/// Cross-module agreement neutral point (no adjustment).
pub const CROSS_MODULE_AGREEMENT_NEUTRAL: f64 = 0.5;

/// Cross-module agreement adjustment center.
pub const CROSS_MODULE_AGREEMENT_ADJUSTMENT_CENTER: f64 = 0.5;

/// Flow state LR boost multiplier.
/// Basis: Csikszentmihalyi (1990) — flow enhances learning.
pub const FLOW_STATE_LR_BOOST_MULTIPLIER: f64 = 1.05;

/// Unified quality score high threshold for LR scaling.
pub const UNIFIED_QUALITY_HIGH_THRESHOLD: f64 = 0.7;

/// Quality floor exploration adjustment magnitude.
pub const QUALITY_FLOOR_EXPLORATION_ADJUSTMENT: f64 = 0.01;

/// Epistemic conflict exploration scale.
pub const EPISTEMIC_CONFLICT_EXPLORATION_SCALE: f64 = 0.015;

/// Phenomenal fragmentation confidence dampening.
pub const PHENOMENAL_FRAGMENTED_CONFIDENCE_DAMPEN: f64 = 0.95;

/// Phenomenal fragmentation exploration boost.
pub const PHENOMENAL_FRAGMENTED_EXPLORATION_BOOST: f64 = 0.02;

/// Temporal discontinuity LR dampening.
pub const TEMPORAL_DISCONTINUITY_LR_DAMPEN: f64 = 0.95;

/// Temporal discontinuity exploration boost.
pub const TEMPORAL_DISCONTINUITY_EXPLORATION_BOOST: f64 = 0.02;

/// Temporal binding high exploration scale (near-1.0 = slight dampening).
pub const TEMPORAL_BINDING_HIGH_EXPLORATION_SCALE: f64 = 0.98;

/// Cross-modal binding EMA momentum.
pub const CROSS_MODAL_BINDING_MOMENTUM: f64 = 0.9;

/// Cross-modal binding EMA alpha.
pub const CROSS_MODAL_BINDING_ALPHA: f64 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// OUTPUT PHASE — TELEMETRY OBSERVABILITY THRESHOLDS
// Science: Dehaene (2014) — global workspace observability requires threshold
// gating to distinguish meaningful state changes from noise.
// ═══════════════════════════════════════════════════════════════════════════════

/// Conflict exploration boost increment (per-cycle delta).
/// Basis: Botvinick et al. (2001) — ACC conflict → exploration.
pub const CONFLICT_EXPLORATION_INCREMENT: f64 = 0.02;

/// Confidence velocity threshold for rising-confidence dampening detection.
/// Basis: Schultz (2016) — rising confidence at this rate warrants caution.
pub const CONFIDENCE_VELOCITY_RISING_THRESHOLD: f64 = 0.02;


/// FEP accuracy threshold for efficiency detection.
/// Basis: Friston (2010) — accurate prediction signals good model fit.
pub const FEP_ACCURACY_EFFICIENCY_THRESHOLD: f64 = 0.5;

/// FEP complexity threshold for efficiency detection (below = efficient).
/// Basis: Friston (2010) — low complexity = parsimonious model.
pub const FEP_COMPLEXITY_EFFICIENCY_THRESHOLD: f64 = 0.5;

/// Living mind vitality high threshold (above = feedback modulation).
/// Basis: Thompson (2007) — autopoietic vitality above threshold sustains cognition.
pub const LIVING_MIND_VITALITY_HIGH: f64 = 0.6;

/// Living mind vitality low threshold (below = dampened cognition).
/// Basis: Thompson (2007) — autopoietic vitality below threshold indicates depleted self-maintenance.
pub const LIVING_MIND_VITALITY_LOW: f64 = 0.3;

/// Meta-cognitive accuracy low threshold for dampening.
/// Basis: Fleming & Dolan (2012) — poor metacognition warrants caution.
pub const META_COGNITIVE_ACCURACY_LOW: f64 = 0.3;

/// Predictive self-safety high threshold for LR boost.
/// Basis: Seth (2013) — strong interoceptive self-model permits faster learning.
pub const PREDICTIVE_SELF_SAFETY_HIGH: f64 = 0.7;

/// Embodied agency stable range: lower bound.
/// Basis: Gallagher (2005) — sense of agency requires moderate range.
pub const EMBODIED_AGENCY_STABLE_MIN: f64 = 0.4;

/// Embodied agency stable range: upper bound.
/// Basis: Gallagher (2005) — excessive agency signal suggests over-attribution.
pub const EMBODIED_AGENCY_STABLE_MAX: f64 = 0.6;

/// Attention schema control signal fatigue threshold (below = fatigued).
/// Basis: Graziano (2013) — low control signal = attentional depletion.
pub const ATTENTION_SCHEMA_FATIGUE_THRESHOLD: f64 = 0.4;

/// Anomaly recovery PSI acceleration multiplier.
/// Basis: Carver & Scheier (1998) — post-anomaly recovery acceleration.
pub const ANOMALY_RECOVERY_PSI_MULTIPLIER: f64 = 1.05;

/// Consolidation consciousness offset (subtracted from EMA).
/// Basis: Dehaene & Changeux (2011) — consolidation threshold sits below conscious access level.
pub const CONSOLIDATION_CONSCIOUSNESS_OFFSET: f64 = 0.1;

/// Consolidation threshold minimum clamp.
/// Basis: Dehaene & Changeux (2011) — minimum ignition threshold for conscious access.
pub const CONSOLIDATION_THRESHOLD_MIN: f64 = 0.2;

/// Confidence velocity falling threshold for crash detection.
/// Basis: Yu & Dayan (2005) — rapid confidence drops signal unexpected uncertainty.
pub const CONFIDENCE_VELOCITY_FALLING_THRESHOLD: f64 = -0.05;

/// Error slope threshold for memory consolidation trigger.
/// Basis: Rao & Ballard (1999) — rising PE slope signals model surprise.
pub const ERROR_SLOPE_CONSOLIDATION_THRESHOLD: f64 = 0.03;

// ── Modulation observability thresholds (high/low pairs) ─────────────────
// These gate telemetry boolean flags that report which feedback loops fired.
// Science: Tononi (2004) — IIT requires transparency about integration state.

/// Epistemic Phi effectiveness high threshold.
pub const EPISTEMIC_PHI_HIGH: f64 = 0.6;
/// Epistemic Phi effectiveness low threshold.
pub const EPISTEMIC_PHI_LOW: f64 = 0.2;
/// Phenomenal binding strength high threshold.
pub const PHENOMENAL_BINDING_HIGH: f64 = 0.7;
/// Phenomenal binding strength low threshold.
pub const PHENOMENAL_BINDING_LOW: f64 = 0.15;
/// Temporal coherence score high threshold.
pub const TEMPORAL_COHERENCE_HIGH: f64 = 0.75;
/// Temporal coherence score low threshold.
pub const TEMPORAL_COHERENCE_LOW: f64 = 0.15;
/// Holographic unity high threshold.
pub const HOLOGRAPHIC_UNITY_HIGH: f64 = 0.7;
/// Holographic unity low threshold.
pub const HOLOGRAPHIC_UNITY_LOW: f64 = 0.15;
/// Harmonies alignment high threshold.
pub const HARMONIES_ALIGNMENT_HIGH: f64 = 0.8;
/// Harmonies alignment low threshold.
pub const HARMONIES_ALIGNMENT_LOW: f64 = 0.2;
/// Consciousness gradient LR modulation threshold.
pub const CONSCIOUSNESS_GRADIENT_LR_MOD_THRESHOLD: f64 = 0.05;
/// Value cache hit rate low threshold (below = modulated).
pub const VALUE_CACHE_HIT_RATE_LOW: f64 = 0.3;
/// Value cache hit rate high threshold (above = modulated).
pub const VALUE_CACHE_HIT_RATE_HIGH: f64 = 0.9;
/// Consciousness state level high threshold.
pub const CONSCIOUSNESS_STATE_LEVEL_HIGH: f64 = 0.7;
/// Consciousness state level low threshold.
pub const CONSCIOUSNESS_STATE_LEVEL_LOW: f64 = 0.2;
/// Living mind vitality modulation high threshold.
pub const LIVING_MIND_VITALITY_MOD_HIGH: f64 = 0.7;
/// Living mind vitality modulation low threshold.
pub const LIVING_MIND_VITALITY_MOD_LOW: f64 = 0.3;
/// Living mind coherence modulation high threshold.
pub const LIVING_MIND_COHERENCE_MOD_HIGH: f64 = 0.7;
/// Living mind coherence modulation low threshold.
pub const LIVING_MIND_COHERENCE_MOD_LOW: f64 = 0.3;
/// MCTS effectiveness high threshold.
pub const MCTS_EFFECTIVENESS_MOD_HIGH: f64 = 0.6;
/// MCTS effectiveness low threshold.
pub const MCTS_EFFECTIVENESS_MOD_LOW: f64 = 0.2;
/// Resonator semantic similarity high threshold for telemetry.
pub const RESONATOR_SIMILARITY_HIGH: f64 = 0.8;
/// Resonator semantic similarity low threshold for telemetry.
pub const RESONATOR_SIMILARITY_LOW: f64 = 0.3;
/// Binding strength high threshold for telemetry.
pub const BINDING_STRENGTH_TELEMETRY_HIGH: f64 = 0.7;
/// Binding strength low threshold for telemetry.
pub const BINDING_STRENGTH_TELEMETRY_LOW: f64 = 0.3;
/// Pipeline consciousness high threshold for telemetry gating.
/// Basis: Dehaene (2014) — global workspace ignition threshold.
pub const PIPELINE_CONSCIOUSNESS_HIGH_THRESHOLD: f64 = 0.7;
/// Pipeline consciousness low threshold for telemetry gating.
pub const PIPELINE_CONSCIOUSNESS_LOW_THRESHOLD: f64 = 0.3;
