//! # Threshold Registry — Centralized Cognitive Tuning Constants
//!
//! All magic numbers used in the cognitive loop are collected here with:
//! - Scientific citations for each value's biological/theoretical basis
//! - Validation logic for ordering and non-overlap constraints
//! - Clear grouping by subsystem domain
//!
//! ## Why Centralize?
//!
//! Before this registry, ~50 constants were scattered across `cycle.rs`,
//! `cycle_extracted.rs`, and `helpers/`. This made it impossible to:
//! - Audit for contradictions (two thresholds at the same value)
//! - Sweep parameters systematically
//! - Verify ordering invariants (e.g., concern < neutral < benefit)
//!
//! ## Adding New Constants
//!
//! 1. Add the constant to the appropriate group below
//! 2. Add a doc comment citing the scientific basis
//! 3. Add any ordering constraints to `validate()`
//! 4. Update any existing `const` in cycle.rs to reference this module

// ═══════════════════════════════════════════════════════════════════════════════
// MORAL / ETHICAL EVALUATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Moral score below this triggers concern and biases toward Supportive strategy.
/// Basis: Haidt (2001) — moral intuition triggers fast conservative override.
pub const MORAL_CONCERN_THRESHOLD: f32 = -0.3;

/// Moral score above this boosts prediction confidence.
/// Basis: Damasio (1994) — positive somatic markers reinforce decision confidence.
pub const MORAL_BENEFIT_THRESHOLD: f32 = 0.5;

/// Exploration dampening when moral concern detected (multiplicative).
/// Basis: De Martino (2006) — amygdala activation reduces exploratory behavior.
pub const MORAL_CONCERN_EXPLORATION_DAMPEN: f32 = 0.5;

/// Speech rate boost on moral concern (multiplicative, >1 = slower/more cautious).
/// Basis: Forgas (2011) — negative affect promotes systematic processing.
pub const MORAL_CONCERN_PAUSE_BOOST: f32 = 1.5;

/// Confidence nudge for positive moral alignment (multiplicative).
/// Basis: Schwartz (2012) — value-aligned actions strengthen self-efficacy.
pub const MORAL_BENEFIT_CONFIDENCE_BOOST: f32 = 1.05;

/// Moral evaluation amortization interval (cycles). Co-prime.
/// Basis: Kahneman (2011) — System 2 moral evaluation is expensive.
pub const MORAL_EVAL_INTERVAL: usize = 7;

/// Negation polarity threshold for moral input preprocessing.
/// Basis: Horn (1989) — pragmatic negation inverts semantic content.
pub const NEGATION_POLARITY_THRESHOLD: f32 = 0.5;

/// Negation dampening factor applied to moral evaluation.
pub const NEGATION_DAMPENING: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS / EXPLORATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Quantum coherence level above which exploration gets a boost.
/// Basis: Lambert (2013) — quantum coherence enhances biological search.
pub const QUANTUM_COHERENCE_THRESHOLD: f64 = 0.5;

/// Strength of coherence → exploration boost (multiplicative).
pub const QUANTUM_COHERENCE_BOOST_SCALE: f32 = 0.2;

/// GWT broadcast confidence boost (additive).
/// Basis: Baars (2005) — global workspace broadcast increases confidence.
pub const GWT_BROADCAST_CONFIDENCE_BOOST: f32 = 0.03;

// ═══════════════════════════════════════════════════════════════════════════════
// FEP / ACTIVE INFERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Free-energy divisor for surprise-driven LR boost.
/// Basis: Friston (2010) — surprise magnitude scales learning urgency.
pub const FEP_SURPRISE_SCALE: f32 = 3.0;

/// LR boost decay rate when not surprised (per cycle, multiplicative).
/// Basis: Dayan & Balleine (2002) — learning rate returns to baseline without novelty.
pub const FEP_LR_DECAY: f32 = 0.95;

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
// RESONANCE / TEMPORAL DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Neutral resonance frequency (center of tau modulation range).
/// Basis: Buzsáki (2006) — neural oscillations modulate processing speed.
pub const RESONANCE_TAU_CENTER: f64 = 0.5;

/// Maximum CfC time-step modulation from resonance (±%).
pub const RESONANCE_TAU_SCALE: f32 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// POLICY AGREEMENT (FEP↔MCTS)
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum FEP probability to accept MCTS action choice.
/// Basis: Friston & Parr (2020) — policy alignment requires precision threshold.
pub const POLICY_SOFT_THRESHOLD: f64 = 0.2;

/// Confidence boost when FEP and MCTS fully agree.
pub const POLICY_FULL_AGREEMENT_BOOST: f32 = 1.2;

/// Sliding window size for policy agreement tracking.
pub const POLICY_WINDOW_SIZE: usize = 20;

/// Minimum samples in agreement window before adaptive temperature kicks in.
pub const POLICY_MIN_WINDOW: usize = 5;

/// Minimum softmax temperature for action selection.
pub const POLICY_TEMP_BASE: f64 = 0.5;

/// Temperature range [BASE, BASE+RANGE].
pub const POLICY_TEMP_RANGE: f64 = 1.5;

// ═══════════════════════════════════════════════════════════════════════════════
// MCE / CONSCIOUSNESS MEASUREMENT
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum LR boost from consciousness level (MCE) — up to +20%.
/// Basis: Dehaene (2014) — conscious access facilitates learning.
/// Doubled from 0.1 to strengthen consciousness→learning coupling.
pub const MCE_LR_BOOST_SCALE: f32 = 0.2;

/// MCE LR boost decay per cycle (multiplicative).
pub const MCE_BOOST_DECAY: f32 = 0.9;

// ═══════════════════════════════════════════════════════════════════════════════
// ATTENTION BUDGET
// ═══════════════════════════════════════════════════════════════════════════════

/// Attention budget in microseconds per cycle (~20Hz target = 50ms).
/// Basis: Posner (1980) — attention is a limited-capacity resource.
pub const ATTENTION_BUDGET_US: u64 = 50_000;

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY & PERCEPTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Top-K memory recall results to consider per cycle.
/// Basis: Cowan (2001) — working memory capacity is ~4±1 items.
pub const MEMORY_RECALL_TOP_K: usize = 3;

/// Memory context boost scale (for LR modulation from recalled episodes).
/// Basis: Tulving (2002) — episodic memory provides contextual priming.
pub const MEMORY_CONTEXT_BOOST_SCALE: f32 = 0.1;

/// Surprise-driven boredom dampening (multiplicative, <1 reduces boredom threshold).
/// Basis: Berlyne (1960) — collative variables drive curiosity; surprise resets boredom.
pub const SURPRISE_BOREDOM_DAMPEN: f32 = 0.7;

/// Minimum cosine similarity for memory recall to be considered relevant.
/// Basis: Shiffrin & Steyvers (1997) — retrieval threshold in memory models.
pub const MEMORY_RECALL_SIM_THRESHOLD: f32 = 0.3;

/// Default input similarity memoization threshold (cosine).
/// Basis: Tulving & Schacter (1990) — repetition priming allows processing shortcuts.
pub const INPUT_MEMO_THRESHOLD: f32 = 0.95;

// ═══════════════════════════════════════════════════════════════════════════════
// PSI SYNTHESIS WEIGHTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Flow state contribution to unified Psi (additive weight).
/// Basis: Csikszentmihalyi (1990) — flow is a marker of integrated consciousness.
pub const FLOW_PSI_WEIGHT: f32 = 0.2;

/// Relational (dyadic) contribution to unified Psi (additive weight).
/// Basis: Gallagher (2005) — intersubjective consciousness contributes to integration.
pub const RELATIONAL_PSI_WEIGHT: f32 = 0.15;

/// Virtual body contribution to unified Psi (additive weight).
/// Basis: Damasio (1994) — somatic markers modulate consciousness.
pub const BODY_PSI_WEIGHT: f64 = 0.1;

/// Embodied cognition contribution to unified Psi (additive weight).
/// Basis: Thompson (2007) — enactive cognition extends consciousness.
pub const EMBODIED_PSI_WEIGHT: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// STRATEGY MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Exploratory strategy LR dampening factor (multiplicative).
/// Basis: Hills (2015) — exploration reduces learning commitment to current path.
pub const STRATEGY_EXPLORATORY_FACTOR: f32 = 0.8;

/// Detailed strategy attention sensitivity boost (multiplicative).
pub const STRATEGY_DETAILED_SENSITIVITY: f32 = 1.2;

/// Concise strategy speech rate boost (multiplicative).
pub const STRATEGY_CONCISE_SPEECH_RATE: f32 = 1.2;

/// Clarifying strategy factor (multiplicative on confidence threshold).
pub const STRATEGY_CLARIFYING_FACTOR: f32 = 0.5;

/// Supportive strategy pause boost (multiplicative on speech rate).
pub const STRATEGY_SUPPORTIVE_PAUSE: f32 = 1.3;

// ═══════════════════════════════════════════════════════════════════════════════
// REWARD COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Base reward for good outcomes (low prediction error + high confidence).
/// Basis: Schultz (1997) — dopaminergic reward prediction error signal.
pub const REWARD_GOOD_BASE: f32 = 0.5;

/// Confidence scaling for good reward.
pub const REWARD_GOOD_CONFIDENCE_SCALE: f32 = 0.5;

/// Base reward for bad outcomes (high prediction error).
pub const REWARD_BAD_BASE: f32 = -0.3;

/// Prediction error scaling for bad reward.
pub const REWARD_BAD_SCALE: f32 = -0.2;

/// Base reward for mid outcomes (moderate error).
pub const REWARD_MID_BASE: f32 = 0.2;

/// Prediction error scaling for mid reward.
pub const REWARD_MID_SCALE: f32 = -0.5;

/// External reward blending weight (0.5 = equal internal/external).
/// Basis: Deci & Ryan (2000) — intrinsic/extrinsic motivation balance.
pub const REWARD_EXTERNAL_BLEND: f32 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// AMORTIZATION INTERVALS (co-prime for phase-avoidance)
// ═══════════════════════════════════════════════════════════════════════════════

/// Co-prime interval reference table:
/// Old → New mapping: 5→7, 10→11, 15→13, 20→19, 25→23, 50→47, 100→97
///
/// Using primes ensures at most 2 subsystems coincide on any given cycle.
/// LCM of first 8 primes (2,3,5,7,11,13,17,19) = 9,699,690 —
/// effectively never all aligning in practical cycle counts.
/// Chronobiology refresh interval (circadian + personality drift).
pub const BIORHYTHM_INTERVAL: usize = 97;

/// Startup transient suppression duration (cycles).
/// Basis: Hopfield (1982) — recurrent networks need settling time.
pub const STARTUP_WARMUP_CYCLES: usize = 50;

// ═══════════════════════════════════════════════════════════════════════════════
// SELF-MODEL
// ═══════════════════════════════════════════════════════════════════════════════

/// Self-model accuracy EMA decay.
/// Basis: Friston (2010) — self-model updates exponentially weighted.
pub const SELF_MODEL_ACCURACY_EMA: f32 = 0.9;

/// High self-model accuracy → trust boost multiplier.
pub const SELF_MODEL_HIGH_TRUST_BOOST: f32 = 0.03;

/// Low self-model accuracy → confidence scaling.
pub const SELF_MODEL_LOW_CONFIDENCE_SCALE: f32 = 0.98;

/// Self-model accuracy threshold for "high" classification.
pub const SELF_MODEL_HIGH_THRESHOLD: f32 = 0.7;

/// Self-model accuracy threshold for "low" classification.
pub const SELF_MODEL_LOW_THRESHOLD: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// RESONATOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Resonator prediction error threshold for exploration boost.
/// Basis: Schmidhuber (2010) — curiosity from prediction error.
pub const RESONATOR_ERROR_EXPLORATION_THRESHOLD: f32 = 0.5;

/// Resonator error → exploration scale factor.
pub const RESONATOR_ERROR_EXPLORATION_SCALE: f32 = 0.08;

/// Resonator error → confidence dampen ratio (multiply on exploration).
pub const RESONATOR_ERROR_CONFIDENCE_DAMPEN: f32 = 0.5;

/// Resonator low error threshold for confidence boost.
pub const RESONATOR_LOW_ERROR_THRESHOLD: f32 = 0.2;

/// Resonator low error → confidence boost scale.
pub const RESONATOR_LOW_ERROR_CONFIDENCE_SCALE: f32 = 0.03;

/// Resonator consolidation threshold: high similarity → familiar pattern.
pub const RESONATOR_CONSOLIDATION_THRESHOLD: f32 = 0.7;

/// LR scale when resonator detects familiar (consolidated) pattern.
pub const RESONATOR_FAMILIAR_LR_SCALE: f32 = 0.85;

/// Resonator novelty threshold: low similarity → novel input.
pub const RESONATOR_NOVEL_THRESHOLD: f32 = 0.3;

/// LR scale when resonator detects novel (unfamiliar) pattern.
pub const RESONATOR_NOVEL_LR_SCALE: f32 = 1.15;

/// LR scale when attention budget is persistently exceeded (cognitive overload).
pub const ATTENTION_BUDGET_GATED_LR_SCALE: f32 = 0.7;

/// Structural Phi bottleneck threshold — poor integration above this value.
pub const STRUCTURAL_BOTTLENECK_THRESHOLD: f32 = 0.6;

/// LR scale when structural bottleneck is high (protect against fragmentation).
pub const STRUCTURAL_BOTTLENECK_LR_SCALE: f32 = 0.8;

/// Structural emergence confidence threshold — synergistic self-organization.
pub const STRUCTURAL_EMERGENCE_CONFIDENCE_THRESHOLD: f32 = 0.4;

/// Confidence boost when structural emergence is high.
pub const STRUCTURAL_EMERGENCE_CONFIDENCE_BOOST: f32 = 0.05;

/// Causal attention strength threshold for boost activation.
pub const CAUSAL_ATTENTION_STRENGTH_THRESHOLD: f32 = 0.3;

/// Confidence scale for causal attention boost.
pub const CAUSAL_ATTENTION_CONFIDENCE_SCALE: f32 = 0.04;

/// LR scale when world-model sensory mismatch is detected.
pub const WM_MISMATCH_LR_SCALE: f32 = 0.75;

/// Confidence scale (dampen) when world-model sensory mismatch is detected.
pub const WM_MISMATCH_CONFIDENCE_SCALE: f32 = 0.9;

// ═══════════════════════════════════════════════════════════════════════════════
// PHENOMENAL BINDING
// ═══════════════════════════════════════════════════════════════════════════════

/// Binding strength threshold for confidence boost and threshold relief.
pub const BINDING_CONFIDENCE_THRESHOLD: f32 = 0.7;

/// Binding strength low threshold for caution/penalty.
pub const BINDING_LOW_THRESHOLD: f32 = 0.3;

/// Strong binding → threshold relief scale.
pub const BINDING_STRONG_RELIEF_SCALE: f32 = 0.3;

/// Weak binding → threshold caution scale.
pub const BINDING_WEAK_CAUTION_SCALE: f32 = 0.2;

/// Strong binding → confidence boost scale.
pub const BINDING_STRONG_CONFIDENCE_SCALE: f32 = 0.1;

/// Weak binding → confidence dampen scale.
pub const BINDING_WEAK_CONFIDENCE_SCALE: f32 = 0.15;

// ═══════════════════════════════════════════════════════════════════════════════
// PREDICTION COHERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Coherence EMA decay for prediction quality tracking.
pub const COHERENCE_PREDICTION_EMA: f32 = 0.9;

/// Low coherence threshold for confidence dampening.
pub const COHERENCE_LOW_THRESHOLD: f32 = 0.5;

/// Low coherence → confidence dampen scale.
pub const COHERENCE_LOW_DAMPEN_SCALE: f32 = 0.04;

/// High coherence threshold for confidence boost.
pub const COHERENCE_HIGH_THRESHOLD: f32 = 0.8;

/// Coherence → confidence boost factor.
pub const COHERENCE_CONFIDENCE_BOOST: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// THETA OSCILLATION & PREDICTION HORIZONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Theta phase advance per cycle (radians).
/// Basis: Buzsáki (2002) — 6Hz theta at 50Hz loop rate: 6 × 2π / 50 ≈ 0.754 rad.
pub const THETA_PHASE_ADVANCE: f64 = 0.754;

/// Theta → Phi modulation amplitude (±fraction of Phi).
/// Basis: Buzsáki (2006) — theta oscillations gate information integration.
pub const THETA_PHI_MODULATION_AMPLITUDE: f64 = 0.10;

/// EMA alpha for smoothing theta-modulated Phi (prevents 6Hz artifacts).
/// Basis: Buzsáki (2006) — downstream consumers need stable consciousness metrics.
pub const THETA_PHI_SMOOTH_ALPHA: f64 = 0.3;

/// Prediction horizon minimum scale (floor).
/// Prevents extremely short horizons under high PE + slow substrate.
pub const PREDICTION_HORIZON_MIN_SCALE: f32 = 0.3;

/// Prediction horizon maximum scale (ceiling).
/// Prevents extremely long horizons under low PE + fast substrate.
pub const PREDICTION_HORIZON_MAX_SCALE: f32 = 2.0;

/// PE threshold above which prediction horizon contracts (focus near-term).
/// Science: Clark (2013) — high prediction error narrows temporal scope.
pub const HORIZON_PE_CONTRACT_THRESHOLD: f32 = 0.3;

/// Contraction rate: how much horizons shrink per unit PE above threshold.
/// At PE=1.0, scale = 1.0 - (0.7 × 0.6) = 0.58 (42% contraction).
pub const HORIZON_PE_CONTRACT_RATE: f32 = 0.6;

/// PE threshold below which prediction horizon expands (exploit stability).
/// Science: Buzsáki (2006) — stable states permit longer integration windows.
pub const HORIZON_PE_EXPAND_THRESHOLD: f32 = 0.05;

/// Expansion rate: how much horizons expand per unit PE below threshold.
/// At PE=0.0, scale = 1.0 + (0.05 × 6.0) = 1.30 (30% expansion).
pub const HORIZON_PE_EXPAND_RATE: f32 = 6.0;

/// Error slope threshold for horizon contraction (rising errors → shorter horizons).
pub const HORIZON_SLOPE_THRESHOLD: f32 = 0.02;

/// Max error slope effect for contraction (caps at 20% reduction).
pub const HORIZON_SLOPE_CONTRACT_CAP: f32 = 0.1;

/// Contraction multiplier for rising error slopes.
pub const HORIZON_SLOPE_CONTRACT_RATE: f32 = 2.0;

/// Max error slope effect for expansion (caps at 15% increase).
pub const HORIZON_SLOPE_EXPAND_CAP: f32 = 0.1;

/// Expansion multiplier for falling error slopes.
pub const HORIZON_SLOPE_EXPAND_RATE: f32 = 1.5;

/// Sustained low-coherence cycle threshold for exploration boost.
/// Basis: Schmidhuber (2010) — curiosity from persistent model confusion.
pub const LOW_COHERENCE_EXPLORATION_THRESHOLD: u32 = 10;

/// Exploration boost per cycle during sustained low coherence.
pub const LOW_COHERENCE_EXPLORATION_BOOST: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// WORLD MODEL
// ═══════════════════════════════════════════════════════════════════════════════

/// World model stiffness threshold → learning rate boost.
/// Basis: Friston (2005) — high precision priors resist updating.
pub const WORLD_MODEL_STIFFNESS_THRESHOLD: f32 = 0.5;

/// World model stiffness → LR nudge scale.
pub const WORLD_MODEL_STIFFNESS_LR_SCALE: f32 = 0.05;

/// World model sponginess threshold → learning rate reduction.
pub const WORLD_MODEL_SPONGINESS_THRESHOLD: f32 = 0.2;

/// Sponginess → LR dampen scale.
pub const WORLD_MODEL_SPONGY_LR_SCALE: f32 = 0.15;

// ═══════════════════════════════════════════════════════════════════════════════
// EPISTEMIC GATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Epistemic gate: rejection → LR dampening scale.
/// Basis: Friston (2017) — epistemic foraging gated by expected info gain.
pub const EPISTEMIC_REJECTION_LR_SCALE: f32 = 0.3;

/// Epistemic gate: rejection → confidence scaling factor.
pub const EPISTEMIC_REJECTION_CONFIDENCE_SCALE: f32 = 0.15;

/// Epistemic gate: approval threshold for LR boost.
pub const EPISTEMIC_APPROVAL_THRESHOLD: f32 = 0.6;

/// Epistemic gate: approval → LR boost scale.
pub const EPISTEMIC_APPROVAL_LR_SCALE: f32 = 0.08;

/// Caution threshold for epistemic gate (hedging behavior).
pub const EPISTEMIC_CAUTION_THRESHOLD: f32 = 0.4;

/// Caution → threshold scaling factor.
pub const EPISTEMIC_CAUTION_SCALE: f32 = 0.3;

/// Trust threshold for epistemic gate (full endorsement).
pub const EPISTEMIC_TRUST_THRESHOLD: f32 = 0.8;

/// Trust → threshold relief scale.
pub const EPISTEMIC_TRUST_SCALE: f32 = 0.15;

// ═══════════════════════════════════════════════════════════════════════════════
// EVOLUTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Evolution Phi threshold triggering confidence feedback.
/// Basis: Tononi (2012) — Phi changes signal consciousness transitions.
pub const EVOLUTION_PHI_THRESHOLD: f64 = 0.01;

/// Positive evolution → confidence scale.
pub const EVOLUTION_POSITIVE_CONFIDENCE_SCALE: f64 = 0.05;

/// Positive evolution → confidence clamp.
pub const EVOLUTION_POSITIVE_CONFIDENCE_MAX: f64 = 0.03;

/// Negative evolution → exploration scale.
pub const EVOLUTION_NEGATIVE_EXPLORATION_SCALE: f64 = 0.08;

/// Negative evolution → exploration clamp.
pub const EVOLUTION_NEGATIVE_EXPLORATION_MAX: f64 = 0.04;

// ═══════════════════════════════════════════════════════════════════════════════
// HARMONIC INTERFERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Max harmonic interference count before LR dampening.
pub const HARMONIC_INTERFERENCE_MAX_COUNT: usize = 3;

/// Per-interference LR dampening factor.
pub const HARMONIC_INTERFERENCE_DAMPEN: f32 = 0.02;

/// Max cumulative LR dampening from interference.
pub const HARMONIC_INTERFERENCE_MAX_DAMPEN: f32 = 0.1;

/// Harmony boost when all clear (0 interferences).
pub const HARMONIC_ALL_CLEAR_BOOST: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// THALAMIC ROUTING
// ═══════════════════════════════════════════════════════════════════════════════

/// Thalamic salience boost for DeepThought routing.
pub const THALAMIC_DEEP_SALIENCE: f32 = 0.2;

/// Thalamic salience penalty for Reflex routing.
pub const THALAMIC_REFLEX_SALIENCE: f32 = -0.1;

/// DeepThought NE tonic production.
/// Basis: Aston-Jones & Cohen (2005) — sustained alerting.
pub const THALAMIC_DEEP_NE_TONIC: f32 = 0.05;

/// DeepThought ACh tonic production.
/// Basis: Sarter et al. (2005) — sustained attention via basal forebrain.
pub const THALAMIC_DEEP_ACH_TONIC: f32 = 0.08;

/// Reflex GABA inhibition.
/// Basis: Buzsáki (2006) — GABAergic inhibition enables fast gating.
pub const THALAMIC_REFLEX_GABA: f32 = 0.04;

/// DeepThought learning rate multiplier.
pub const THALAMIC_DEEP_LR_FACTOR: f32 = 1.3;

/// Reflex learning rate multiplier.
pub const THALAMIC_REFLEX_LR_FACTOR: f32 = 0.5;

/// DeepThought attention budget scale.
pub const THALAMIC_DEEP_BUDGET_SCALE: f64 = 2.0;

/// Reflex attention budget scale.
pub const THALAMIC_REFLEX_BUDGET_SCALE: f64 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// REASONING CONFIDENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Reasoning chain confidence threshold for prediction confidence boost.
/// Basis: Stanovich (2011) — analytic processing confidence reinforces rationality.
pub const REASONING_CONFIDENCE_BOOST_THRESHOLD: f32 = 0.7;

/// Reasoning chain confidence boost factor (multiplicative on delta).
pub const REASONING_CONFIDENCE_BOOST_FACTOR: f32 = 0.03;

// ═══════════════════════════════════════════════════════════════════════════════
// PHI VALIDATION / SPECTRAL WEIGHT
// ═══════════════════════════════════════════════════════════════════════════════

/// High Phi validation correlation threshold (trust spectral MIP more).
/// Basis: Casali et al. (2013) — Phi estimation reliability.
pub const PHI_VALIDATION_HIGH_THRESHOLD: f64 = 0.7;

/// Low Phi validation correlation threshold (reduce spectral weight).
pub const PHI_VALIDATION_LOW_THRESHOLD: f64 = 0.3;

/// Base spectral weight (neutral correlation).
pub const SPECTRAL_WEIGHT_BASE: f32 = 0.6;

/// Spectral weight adjustment scale per unit correlation delta.
pub const SPECTRAL_WEIGHT_SCALE: f32 = 0.67;

// ═══════════════════════════════════════════════════════════════════════════════
// CAUSAL BINDING
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum causal strength for codebook entry inclusion.
/// Basis: Granger (1969) — causal strength filtering threshold.
pub const CAUSAL_BINDING_THRESHOLD: f32 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPORAL DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Temporal continuity threshold below which replay is triggered.
/// Basis: Tulving (2002) — continuity gaps signal memory consolidation need.
pub const TEMPORAL_REPLAY_TRIGGER: f32 = 0.3;

/// Temporal continuity threshold for prediction confidence boost.
/// Basis: Buzsáki (2006) — temporal coherence enables reliable predictions.
pub const TEMPORAL_CONTINUITY_BOOST_THRESHOLD: f64 = 0.7;

/// Temporal continuity → confidence boost factor.
pub const TEMPORAL_CONTINUITY_BOOST_FACTOR: f64 = 0.05;

/// Per-chain confidence boost factor (causal chain detection).
/// Basis: Pearl (2009) — causal chain detection → structural certainty.
pub const TEMPORAL_CHAIN_BOOST_FACTOR: f32 = 0.005;

// ═══════════════════════════════════════════════════════════════════════════════
// HARMONIC FIELD
// ═══════════════════════════════════════════════════════════════════════════════

/// Harmonic field coherence threshold for LR stability boost.
/// Basis: Schwartz (2012) — value coherence enables stable learning.
pub const HARMONIC_FIELD_BOOST_THRESHOLD: f32 = 0.6;

/// Harmonic field coherence → LR boost factor.
pub const HARMONIC_FIELD_BOOST_FACTOR: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// Q-LEARNING / STRATEGY SELECTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Initial Q-value for all strategies (neutral prior).
/// Basis: Watkins (1989) — unbiased Q-value initialization.
pub const Q_VALUE_INITIAL: f32 = 0.5;

/// Q-learning rate (α) for strategy updates.
/// Basis: Watkins & Dayan (1992) — convergence requires α ∈ (0, 1).
pub const Q_LEARNING_RATE: f32 = 0.1;

/// Initial exploration rate (ε) for epsilon-greedy strategy.
/// Basis: Sutton & Barto (2018) — moderate initial exploration.
pub const EXPLORATION_RATE_INITIAL: f32 = 0.2;

/// Phi threshold for integrative mode (favor Exploratory/Detailed).
/// Basis: Tononi (2004) — high integration enables complex strategies.
pub const PHI_INTEGRATIVE_THRESHOLD: f64 = 0.6;

/// Phi threshold for reactive mode (favor Supportive/Concise).
pub const PHI_REACTIVE_THRESHOLD: f64 = 0.3;

/// Exploration rate decay per interaction (multiplicative).
/// Basis: Tokic (2010) — gradual exploitation shift.
pub const EXPLORATION_DECAY_RATE: f32 = 0.999;

/// Minimum exploration rate floor.
pub const EXPLORATION_RATE_MIN: f32 = 0.05;

/// Reward threshold for "strong positive" (stick with strategy).
/// Basis: Schultz (1997) — dopaminergic reward signal thresholds.
pub const REWARD_POSITIVE_THRESHOLD: f32 = 0.5;

/// Reward threshold for "negative" (try opposite strategy).
pub const REWARD_NEGATIVE_THRESHOLD: f32 = -0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// FOVEATION → DYNAMICS COUPLING (vision-manifold feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Foveation recognition count threshold for "familiar scene" dampening.
/// Basis: Corbetta & Shulman (2002) — recognized objects reduce attentional vigilance.
#[cfg(feature = "vision-manifold")]
pub const FOVEATION_FAMILIAR_RECOGNITION_COUNT: usize = 2;

/// Foveation confidence threshold for high-confidence dampening.
/// Basis: Bar (2003) — confident recognition facilitates predictive processing.
#[cfg(feature = "vision-manifold")]
pub const FOVEATION_HIGH_CONFIDENCE_THRESHOLD: f32 = 0.6;

/// Exploration dampening when scene is familiar (multiplicative).
#[cfg(feature = "vision-manifold")]
pub const FOVEATION_FAMILIAR_EXPLORATION_DAMPEN: f32 = 0.9;

/// Confidence boost for high-confidence foveation (multiplicative).
#[cfg(feature = "vision-manifold")]
pub const FOVEATION_CONFIDENCE_BOOST: f32 = 1.03;

/// LR boost when novel objects detected (low confidence, many recognitions).
#[cfg(feature = "vision-manifold")]
pub const FOVEATION_NOVEL_LR_BOOST: f32 = 1.05;

/// Maximum weight a single foveation result contributes to multimodal HV binding.
/// Basis: Treisman (1980) — feature integration theory; visual binding is secondary to attentional binding.
#[cfg(feature = "vision-manifold")]
pub const FOVEATION_HV_BINDING_WEIGHT: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-MANIFOLD PREDICTION ERROR (vision-manifold feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Cross-manifold prediction error threshold for attention reallocation.
/// Basis: Rao & Ballard (1999) — prediction error drives top-down attention shifts.
#[cfg(feature = "vision-manifold")]
pub const CROSS_MANIFOLD_ERROR_THRESHOLD: f32 = 0.3;

/// Exploration boost per unit of cross-manifold prediction error above threshold.
#[cfg(feature = "vision-manifold")]
pub const CROSS_MANIFOLD_EXPLORATION_SCALE: f32 = 0.15;

/// Confidence dampening when vision doesn't match cognition.
#[cfg(feature = "vision-manifold")]
pub const CROSS_MANIFOLD_CONFIDENCE_DAMPEN: f32 = 0.97;

/// LR boost when cross-manifold error is high (need to update world model).
#[cfg(feature = "vision-manifold")]
pub const CROSS_MANIFOLD_LR_BOOST: f32 = 1.03;

// ═══════════════════════════════════════════════════════════════════════════════
// VISION SURPRISE → EXPLORATION (vision-manifold feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Vision mean surprise threshold for exploration boost.
/// Basis: Friston (2010) — free energy (surprise) is the fundamental drive for exploration.
#[cfg(feature = "vision-manifold")]
pub const VISION_SURPRISE_EXPLORATION_THRESHOLD: f32 = 0.25;

/// Scale factor for vision surprise → exploration boost.
#[cfg(feature = "vision-manifold")]
pub const VISION_SURPRISE_EXPLORATION_SCALE: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// VISION TEMPORAL HORIZONS → FEP (vision-manifold feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Short-horizon (33ms) visual prediction error threshold for FEP surprise boost.
/// Basis: Adams et al. (2013) — precision-weighted prediction errors at multiple timescales.
#[cfg(feature = "vision-manifold")]
pub const VISION_SHORT_HORIZON_ERROR_THRESHOLD: f32 = 0.3;

/// Long-horizon (500ms+) visual prediction error threshold for confidence dampening.
#[cfg(feature = "vision-manifold")]
pub const VISION_LONG_HORIZON_CONFIDENCE_THRESHOLD: f32 = 0.5;

/// FEP exploration boost from short-horizon visual surprise.
#[cfg(feature = "vision-manifold")]
pub const VISION_HORIZON_EXPLORATION_SCALE: f32 = 0.15;

/// Confidence dampening from long-horizon visual uncertainty.
#[cfg(feature = "vision-manifold")]
pub const VISION_HORIZON_CONFIDENCE_DAMPEN: f32 = 0.97;

// ═══════════════════════════════════════════════════════════════════════════════
// SCENE RECOGNITION → DREAM (vision-manifold feature)
// ═══════════════════════════════════════════════════════════════════════════════

/// Dream recording salience boost when a visual scene is recognized.
/// Basis: Conway (2005) — self-relevant and context-rich memories encode preferentially.
#[cfg(feature = "vision-manifold")]
pub const SCENE_RECOGNITION_DREAM_BOOST: f32 = 1.2;

// ═══════════════════════════════════════════════════════════════════════════════
// VISION → TRAINING IMPORTANCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Base importance weight for training samples.
pub const TRAINING_BASE_IMPORTANCE: f32 = 1.0;

/// Vision cross-manifold error scale for training importance.
/// Basis: Niv et al. (2009) — prediction error modulates learning rate.
#[cfg(feature = "vision-manifold")]
pub const VISION_TRAINING_IMPORTANCE_SCALE: f32 = 0.5;

/// Vision mean-surprise scale for training importance.
/// Complementary to cross-manifold error (0.5): surprise is a rawer signal.
/// Basis: Pearce & Hall (1980) — stimulus surprise increases associability.
#[cfg(feature = "vision-manifold")]
pub const VISION_SURPRISE_TRAINING_IMPORTANCE_SCALE: f32 = 0.3;

/// Maximum training importance weight.
#[cfg(feature = "vision-manifold")]
pub const TRAINING_MAX_IMPORTANCE: f32 = 2.0;

// ═══════════════════════════════════════════════════════════════════════════════
// VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Validate all threshold ordering and non-overlap constraints.
///
/// Panics with a descriptive message if any invariant is violated.
/// Called once at startup (via `CognitiveLoopService::new()`) to catch
/// configuration errors early.
///
/// # Invariants
///
/// 1. `MORAL_CONCERN_THRESHOLD < 0 < MORAL_BENEFIT_THRESHOLD`
/// 2. `FEP_LR_DECAY ∈ (0, 1)` (must actually decay)
/// 3. `POLICY_SOFT_THRESHOLD ∈ (0, 1)` (valid probability)
/// 4. `ATTENTION_BUDGET_US > 0` (nonzero budget)
/// 5. `POLICY_MIN_WINDOW < POLICY_WINDOW_SIZE`
/// 6. `DOMINANCE_DEFAULT < DOMINANCE_CONFIDENT < DOMINANCE_FLOW_BASE`
/// 7. All weights are non-negative
#[allow(clippy::assertions_on_constants)]
pub fn validate() {
    // 1. Moral ordering: concern < 0 < benefit
    assert!(
        MORAL_CONCERN_THRESHOLD < 0.0,
        "MORAL_CONCERN_THRESHOLD must be negative: {}",
        MORAL_CONCERN_THRESHOLD
    );
    assert!(
        MORAL_BENEFIT_THRESHOLD > 0.0,
        "MORAL_BENEFIT_THRESHOLD must be positive: {}",
        MORAL_BENEFIT_THRESHOLD
    );
    assert!(
        MORAL_CONCERN_THRESHOLD < MORAL_BENEFIT_THRESHOLD,
        "MORAL_CONCERN_THRESHOLD ({}) must be < MORAL_BENEFIT_THRESHOLD ({})",
        MORAL_CONCERN_THRESHOLD,
        MORAL_BENEFIT_THRESHOLD
    );

    // 2. FEP decay is valid
    assert!(
        (0.0..1.0).contains(&FEP_LR_DECAY),
        "FEP_LR_DECAY must be in (0, 1): {}",
        FEP_LR_DECAY
    );

    // 3. Policy threshold is a valid probability
    assert!(
        (0.0..=1.0).contains(&POLICY_SOFT_THRESHOLD),
        "POLICY_SOFT_THRESHOLD must be in [0, 1]: {}",
        POLICY_SOFT_THRESHOLD
    );

    // 4. Nonzero attention budget
    assert!(ATTENTION_BUDGET_US > 0, "ATTENTION_BUDGET_US must be > 0");

    // 5. Window ordering
    assert!(
        POLICY_MIN_WINDOW < POLICY_WINDOW_SIZE,
        "POLICY_MIN_WINDOW ({}) must be < POLICY_WINDOW_SIZE ({})",
        POLICY_MIN_WINDOW,
        POLICY_WINDOW_SIZE
    );

    // 6. Dominance ordering
    assert!(
        DOMINANCE_DEFAULT < DOMINANCE_CONFIDENT,
        "DOMINANCE_DEFAULT ({}) must be < DOMINANCE_CONFIDENT ({})",
        DOMINANCE_DEFAULT,
        DOMINANCE_CONFIDENT
    );
    assert!(
        DOMINANCE_CONFIDENT < DOMINANCE_FLOW_BASE,
        "DOMINANCE_CONFIDENT ({}) must be < DOMINANCE_FLOW_BASE ({})",
        DOMINANCE_CONFIDENT,
        DOMINANCE_FLOW_BASE
    );

    // 7. Non-negative weights
    assert!(FLOW_PSI_WEIGHT >= 0.0, "FLOW_PSI_WEIGHT must be >= 0");
    assert!(
        RELATIONAL_PSI_WEIGHT >= 0.0,
        "RELATIONAL_PSI_WEIGHT must be >= 0"
    );
    assert!(BODY_PSI_WEIGHT >= 0.0, "BODY_PSI_WEIGHT must be >= 0");
    assert!(
        EMBODIED_PSI_WEIGHT >= 0.0,
        "EMBODIED_PSI_WEIGHT must be >= 0"
    );

    // 8. Self-model ordering
    assert!(
        SELF_MODEL_LOW_THRESHOLD < SELF_MODEL_HIGH_THRESHOLD,
        "SELF_MODEL_LOW_THRESHOLD ({}) must be < SELF_MODEL_HIGH_THRESHOLD ({})",
        SELF_MODEL_LOW_THRESHOLD,
        SELF_MODEL_HIGH_THRESHOLD
    );

    // 9. Epistemic gate ordering
    assert!(
        EPISTEMIC_CAUTION_THRESHOLD < EPISTEMIC_APPROVAL_THRESHOLD,
        "EPISTEMIC_CAUTION_THRESHOLD ({}) must be < EPISTEMIC_APPROVAL_THRESHOLD ({})",
        EPISTEMIC_CAUTION_THRESHOLD,
        EPISTEMIC_APPROVAL_THRESHOLD
    );
    assert!(
        EPISTEMIC_APPROVAL_THRESHOLD < EPISTEMIC_TRUST_THRESHOLD,
        "EPISTEMIC_APPROVAL_THRESHOLD ({}) must be < EPISTEMIC_TRUST_THRESHOLD ({})",
        EPISTEMIC_APPROVAL_THRESHOLD,
        EPISTEMIC_TRUST_THRESHOLD
    );

    // 10. Binding threshold ordering
    assert!(
        BINDING_LOW_THRESHOLD < BINDING_CONFIDENCE_THRESHOLD,
        "BINDING_LOW_THRESHOLD ({}) must be < BINDING_CONFIDENCE_THRESHOLD ({})",
        BINDING_LOW_THRESHOLD,
        BINDING_CONFIDENCE_THRESHOLD
    );

    // 11. Coherence ordering
    assert!(
        COHERENCE_LOW_THRESHOLD < COHERENCE_HIGH_THRESHOLD,
        "COHERENCE_LOW_THRESHOLD ({}) must be < COHERENCE_HIGH_THRESHOLD ({})",
        COHERENCE_LOW_THRESHOLD,
        COHERENCE_HIGH_THRESHOLD
    );

    // 12. Thalamic budget scales
    assert!(
        THALAMIC_REFLEX_BUDGET_SCALE < THALAMIC_DEEP_BUDGET_SCALE,
        "Reflex budget scale ({}) must be < DeepThought budget scale ({})",
        THALAMIC_REFLEX_BUDGET_SCALE,
        THALAMIC_DEEP_BUDGET_SCALE
    );

    // 13. Psi weights don't exceed 1.0 total
    let psi_total = FLOW_PSI_WEIGHT as f64
        + RELATIONAL_PSI_WEIGHT as f64
        + BODY_PSI_WEIGHT
        + EMBODIED_PSI_WEIGHT;
    assert!(
        psi_total <= 1.0,
        "Psi weights sum ({}) must be <= 1.0",
        psi_total
    );

    // 14. Phi validation ordering
    assert!(
        PHI_VALIDATION_LOW_THRESHOLD < PHI_VALIDATION_HIGH_THRESHOLD,
        "PHI_VALIDATION_LOW_THRESHOLD ({}) must be < PHI_VALIDATION_HIGH_THRESHOLD ({})",
        PHI_VALIDATION_LOW_THRESHOLD,
        PHI_VALIDATION_HIGH_THRESHOLD
    );

    // 15. Phi gating ordering
    assert!(
        PHI_REACTIVE_THRESHOLD < PHI_INTEGRATIVE_THRESHOLD,
        "PHI_REACTIVE_THRESHOLD ({}) must be < PHI_INTEGRATIVE_THRESHOLD ({})",
        PHI_REACTIVE_THRESHOLD,
        PHI_INTEGRATIVE_THRESHOLD
    );

    // 16. Exploration rate bounds
    assert!(
        EXPLORATION_RATE_MIN < EXPLORATION_RATE_INITIAL,
        "EXPLORATION_RATE_MIN ({}) must be < EXPLORATION_RATE_INITIAL ({})",
        EXPLORATION_RATE_MIN,
        EXPLORATION_RATE_INITIAL
    );
    assert!(
        (0.0..1.0).contains(&EXPLORATION_DECAY_RATE),
        "EXPLORATION_DECAY_RATE must be in (0, 1): {}",
        EXPLORATION_DECAY_RATE
    );

    // 17. Q-learning rate valid
    assert!(
        (0.0..=1.0).contains(&Q_LEARNING_RATE),
        "Q_LEARNING_RATE must be in [0, 1]: {}",
        Q_LEARNING_RATE
    );

    // 18. Reward threshold ordering
    assert!(
        REWARD_NEGATIVE_THRESHOLD < REWARD_POSITIVE_THRESHOLD,
        "REWARD_NEGATIVE_THRESHOLD ({}) must be < REWARD_POSITIVE_THRESHOLD ({})",
        REWARD_NEGATIVE_THRESHOLD,
        REWARD_POSITIVE_THRESHOLD
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUBSYSTEM FEEDBACK CLAMPING
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum subsystem LR factor (shared across all subsystem feedback loops).
/// Basis: prevents any single loop from completely suppressing learning.
pub const SUBSYSTEM_LR_FACTOR_MIN: f32 = 0.7;

/// Maximum subsystem LR factor (shared across all subsystem feedback loops).
/// Basis: bounds amplification to prevent runaway learning.
pub const SUBSYSTEM_LR_FACTOR_MAX: f32 = 1.3;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-MODULE QUALITY METRICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Cross-module agreement above this signals high convergence → confidence boost.
#[allow(dead_code)]
pub const CROSS_MODULE_AGREEMENT_HIGH: f32 = 0.8;

/// Cross-module agreement below this signals divergence → exploration boost.
#[allow(dead_code)]
pub const CROSS_MODULE_AGREEMENT_LOW: f32 = 0.3;

/// Unified quality score composition: prediction accuracy weight.
#[allow(dead_code)]
pub const UNIFIED_QUALITY_PREDICTION_WEIGHT: f32 = 0.5;

/// Unified quality score composition: cross-module agreement weight.
#[allow(dead_code)]
pub const UNIFIED_QUALITY_AGREEMENT_WEIGHT: f32 = 0.3;

/// Unified quality score composition: anomaly (inverse) weight.
#[allow(dead_code)]
pub const UNIFIED_QUALITY_ANOMALY_WEIGHT: f32 = 0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// FEP / ACTIVE INFERENCE DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════

/// FEP accuracy above this triggers confidence boost.
/// Basis: Friston (2010) — accurate prediction reduces free energy.
#[allow(dead_code)]
pub const FEP_ACCURACY_CONFIDENCE_THRESHOLD: f64 = 0.5;

/// FEP complexity above this dampens learning rate.
/// Basis: Bayesian model complexity penalty (BIC/MDL principle).
#[allow(dead_code)]
pub const FEP_COMPLEXITY_THRESHOLD: f64 = 1.0;

/// FEP pragmatic value above this triggers exploitation strategy.
/// Basis: Friston et al. (2015) — expected free energy pragmatic term.
#[allow(dead_code)]
pub const FEP_PRAGMATIC_EXPLOIT_THRESHOLD: f64 = 0.7;

/// FEP pragmatic value below this triggers exploration.
#[allow(dead_code)]
pub const FEP_PRAGMATIC_EXPLORE_THRESHOLD: f64 = 0.3;

/// FEP temporal difference error above this triggers causal discovery.
#[allow(dead_code)]
pub const FEP_TD_ERROR_DISCOVERY_THRESHOLD: f64 = 0.5;

/// FEP learning signal above this enables world model plasticity increase.
#[allow(dead_code)]
pub const FEP_LEARNING_PLASTICITY_THRESHOLD: f32 = 0.5;

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
pub const HOMEOSTASIS_PULL_CRUISE: f32 = 1.5;

/// Homeostasis pull multiplier during Normal urgency.
pub const HOMEOSTASIS_PULL_NORMAL: f32 = 1.0;

/// Homeostasis pull multiplier during Critical urgency (weaker → allow reactive emotion).
pub const HOMEOSTASIS_PULL_CRITICAL: f32 = 0.6;

/// Arousal target for homeostasis (slightly above neutral).
pub const HOMEOSTASIS_AROUSAL_TARGET: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// PREDICTIVE BUDGET GATING
// ═══════════════════════════════════════════════════════════════════════════════

/// Fraction of attention budget at which predictive gating activates (80%).
pub const PREDICTIVE_BUDGET_GATING_RATIO: f64 = 0.8;

// ═══════════════════════════════════════════════════════════════════════════════
// SELF-MODEL ACCURACY COMPOSITION
// ═══════════════════════════════════════════════════════════════════════════════

/// Confidence error weight in self-model accuracy composite.
/// Basis: Friston (2010) — precision estimation dominates self-model.
pub const SELF_MODEL_CONFIDENCE_WEIGHT: f32 = 0.7;

/// Urgency match weight in self-model accuracy composite.
pub const SELF_MODEL_URGENCY_WEIGHT: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// PE VARIANCE → CONFIDENCE MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// PE variance threshold below which no dampening occurs.
/// Basis: Yu & Dayan (2005) — expected uncertainty is tolerated.
pub const PE_VARIANCE_THRESHOLD: f32 = 0.01;

/// Maximum PE variance effect (above threshold + this = full dampen).
pub const PE_VARIANCE_MAX_EFFECT: f32 = 0.05;

/// PE variance → confidence dampen multiplier.
pub const PE_VARIANCE_DAMPEN_SCALE: f32 = 2.0;

// ═══════════════════════════════════════════════════════════════════════════════
// CFC TAU FACTOR MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Arousal deadzone around neutral (0.5) — no tau modulation within this band.
/// Basis: Yerkes-Dodson (1908) — moderate arousal has no effect on processing speed.
pub const AROUSAL_TAU_DEADZONE: f32 = 0.1;

/// Arousal → tau sensitivity (per-unit deviation from 0.5).
/// Basis: Aston-Jones & Cohen (2005) — arousal modulates LC-NE → processing tempo.
pub const AROUSAL_TAU_SENSITIVITY: f32 = 0.1;

/// Codebook (resonator) similarity threshold for "familiar" → tau speedup.
/// Basis: Buzsáki (2006) — familiar patterns processed faster.
pub const CODEBOOK_FAMILIAR_THRESHOLD: f32 = 0.5;

/// Codebook familiar → tau scale (negative = faster processing).
pub const CODEBOOK_FAMILIAR_TAU_SCALE: f32 = 0.1;

/// Codebook similarity threshold for "novel" → tau slowdown.
pub const CODEBOOK_NOVEL_THRESHOLD: f32 = 0.2;

/// Codebook novel → tau scale (positive = slower processing).
pub const CODEBOOK_NOVEL_TAU_SCALE: f32 = 0.15;

/// Arousal recovery → tau scale (slows processing to allow recovery).
/// Basis: Lövdén (2010) — cognitive recovery requires reduced processing demands.
pub const AROUSAL_RECOVERY_TAU_SCALE: f32 = 0.2;

/// FEP surprise → tau scale (high surprise = faster inference).
/// Basis: Friston (2010) — surprise accelerates inference dynamics.
pub const FEP_SURPRISE_TAU_SCALE: f32 = 0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// HOMEOSTASIS EFFICIENCY
// ═══════════════════════════════════════════════════════════════════════════════

/// EMA alpha for homeostasis efficiency tracking.
/// Basis: Cannon (1929)/Ashby (1960) — homeostatic regulation monitoring.
pub const HOMEOSTASIS_EFFICIENCY_EMA: f32 = 0.2;

/// Transition cost threshold above which homeostasis is strengthened.
/// Basis: Kelso (1995) — costly transitions increase attractor persistence.
pub const TRANSITION_COST_THRESHOLD: f32 = 0.1;

/// Maximum transition cost effect on homeostasis strength.
pub const TRANSITION_COST_MAX_EFFECT: f32 = 0.2;

/// Transition cost → homeostasis strength scaling.
pub const TRANSITION_COST_STRENGTH_SCALE: f32 = 1.5;

// ═══════════════════════════════════════════════════════════════════════════════
// EPISTEMIC → SEMANTIC LR MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Epistemic confidence threshold for semantic LR caution.
/// Basis: Friston (2017) — low epistemic confidence → cautious learning.
pub const EPISTEMIC_SEMANTIC_CAUTION_THRESHOLD: f32 = 0.4;

/// Epistemic confidence threshold for semantic LR boost.
pub const EPISTEMIC_SEMANTIC_BOOST_THRESHOLD: f32 = 0.8;

/// Base caution factor applied when epistemic confidence is low.
pub const EPISTEMIC_SEMANTIC_CAUTION_BASE: f32 = 0.8;

/// Caution scaling from epistemic confidence (additive).
pub const EPISTEMIC_SEMANTIC_CAUTION_SCALE: f32 = 0.5;

/// Boost scaling from epistemic confidence (multiplicative on excess).
pub const EPISTEMIC_SEMANTIC_BOOST_SCALE: f32 = 1.0;

// ═══════════════════════════════════════════════════════════════════════════════
// EPISTEMIC UNCERTAINTY → EXPLORATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Epistemic uncertainty threshold for exploration boost.
/// Basis: Depeweg et al. (2018) — model uncertainty drives active learning.
pub const EPISTEMIC_EXPLORE_THRESHOLD: f32 = 0.4;

/// Epistemic uncertainty → exploration scale.
pub const EPISTEMIC_EXPLORE_SCALE: f32 = 0.1;

/// Low epistemic uncertainty threshold for exploration dampening.
pub const EPISTEMIC_LOW_THRESHOLD: f32 = 0.15;

/// Low epistemic → exploration dampen amount.
pub const EPISTEMIC_LOW_DAMPEN: f32 = 0.02;

/// Oscillation ratio threshold for compound uncertainty multiplier.
/// Basis: Doya (2002) — compound uncertainty warrants aggressive search.
pub const EPISTEMIC_OSCILLATION_THRESHOLD: f32 = 0.5;

/// Oscillation × uncertainty exploration multiplier.
pub const EPISTEMIC_OSCILLATION_MULTIPLIER: f32 = 1.5;

// ═══════════════════════════════════════════════════════════════════════════════
// MCTS PLAN EFFECTIVENESS
// ═══════════════════════════════════════════════════════════════════════════════

/// MCTS effectiveness threshold for confidence boost.
/// Basis: Silver et al. (2016) — effective planning reinforces confidence.
pub const MCTS_EFFECTIVENESS_HIGH: f32 = 0.6;

/// MCTS effectiveness threshold for exploration trigger.
pub const MCTS_EFFECTIVENESS_LOW: f32 = 0.3;

/// MCTS effective → confidence scale.
pub const MCTS_EFFECTIVENESS_CONFIDENCE_SCALE: f32 = 0.03;

/// MCTS poor plan → exploration scale.
pub const MCTS_EFFECTIVENESS_EXPLORE_SCALE: f32 = 0.02;

/// MCTS effectiveness EMA decay.
pub const MCTS_EFFECTIVENESS_EMA: f32 = 0.9;

/// MCTS plan confidence threshold for application.
pub const MCTS_PLAN_CONFIDENCE_THRESHOLD: f32 = 0.7;

/// MCTS plan → action weight scale.
pub const MCTS_PLAN_WEIGHT_SCALE: f32 = 0.4;

/// MCTS exploit → LR scale per unit plan weight.
pub const MCTS_EXPLOIT_LR_SCALE: f32 = 0.1;

/// MCTS consolidate → confidence scale per unit plan weight.
pub const MCTS_CONSOLIDATE_CONFIDENCE_SCALE: f32 = 0.05;

/// MCTS explore → exploration scale per unit plan weight.
pub const MCTS_EXPLORE_SCALE: f32 = 0.08;

/// Dominance confidence threshold (prediction confidence → dominant feeling).
pub const DOMINANCE_CONFIDENCE_THRESHOLD: f64 = 0.6;

// ═══════════════════════════════════════════════════════════════════════════════
// UNCERTAINTY DEFAULTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Default epistemic uncertainty when insufficient prediction data.
/// Basis: Maximum entropy principle — default to moderate uncertainty.
pub const EPISTEMIC_UNCERTAINTY_DEFAULT: f32 = 0.5;

/// Default aleatoric uncertainty when insufficient prediction data.
pub const ALEATORIC_UNCERTAINTY_DEFAULT: f32 = 0.1;

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
pub const CONFIDENCE_CRASH_EXPLORATION_BOOST: f32 = 0.04;

/// Self-model accuracy threshold above which self-model gets +5% proposal weight.
/// Science: Friston (2010) — high self-model accuracy justifies increased self-trust.
pub const SELF_MODEL_WEIGHT_HIGH_THRESHOLD: f32 = 0.7;

/// Self-model accuracy threshold below which self-model gets -10% proposal weight.
pub const SELF_MODEL_WEIGHT_LOW_THRESHOLD: f32 = 0.3;

/// Self-model high accuracy → confidence weight bonus.
pub const SELF_MODEL_WEIGHT_BONUS: f64 = 0.05;

/// Self-model low accuracy → confidence weight penalty (multiplicative).
pub const SELF_MODEL_WEIGHT_PENALTY: f64 = 0.90;

/// Coherence velocity → CfC tau modulation: rising coherence factor.
/// Science: Buzsáki (2006) — coherent oscillations slow integration (stability).
pub const COHERENCE_VELOCITY_TAU_BOOST: f32 = 1.05;

/// Coherence velocity → CfC tau modulation: falling coherence factor.
/// Falling coherence → faster integration to explore corrections.
pub const COHERENCE_VELOCITY_TAU_DAMPEN: f32 = 0.95;

/// Minimum coherence velocity to trigger tau modulation.
pub const COHERENCE_VELOCITY_TAU_THRESHOLD: f32 = 0.02;

/// Homeostasis efficiency high threshold → reduce pull strength 15%.
/// Science: Ashby (1960) — efficient regulation should self-attenuate.
pub const HOMEOSTASIS_EFFICIENCY_HIGH: f32 = 1.2;

/// Homeostasis efficiency low threshold → increase pull strength 10%.
pub const HOMEOSTASIS_EFFICIENCY_LOW: f32 = 0.8;

/// Pull reduction factor when efficiency is high (multiplicative).
pub const HOMEOSTASIS_PULL_REDUCTION: f32 = 0.85;

/// Pull increase factor when efficiency is low (multiplicative).
pub const HOMEOSTASIS_PULL_INCREASE: f32 = 1.10;

/// Error pattern → predictive LR scale for Rising pattern.
/// Science: Schultz (2016) — increasing errors demand faster adaptation.
pub const ERROR_PATTERN_RISING_LR: f32 = 1.05;

/// Error pattern → predictive LR scale for Falling pattern.
pub const ERROR_PATTERN_FALLING_LR: f32 = 0.97;

/// Error pattern → predictive LR scale for Oscillating pattern.
/// Science: Doya (2002) — oscillating errors indicate meta-uncertainty.
pub const ERROR_PATTERN_OSCILLATING_LR: f32 = 0.90;

/// Minimum distinct proposal sources after warmup (for diversity metric).
/// Science: Dehaene (2014) — GWT requires multi-source consensus.
pub const PROPOSAL_DIVERSITY_MIN_SOURCES: usize = 3;

/// Warmup cycles before diversity check applies.
pub const PROPOSAL_DIVERSITY_WARMUP: usize = 30;

/// Exploration boost when proposal diversity is too low.
pub const PROPOSAL_DIVERSITY_EXPLORATION_BOOST: f32 = 0.02;

/// Mode stability counter threshold for hysteresis relaxation.
/// Science: Kelso (1995) — sustained stability permits relaxed mode boundaries.
pub const HYSTERESIS_RELAXATION_THRESHOLD: u32 = 20;

/// Rate at which hysteresis relaxes toward baseline per cycle (multiplicative).
pub const HYSTERESIS_RELAXATION_RATE: f32 = 0.98;

/// Minimum hysteresis factor after relaxation (prevents collapse).
pub const HYSTERESIS_RELAXATION_FLOOR: f32 = 0.5;

/// Cross-module agreement velocity threshold for confidence velocity coupling.
/// Science: Tononi (2004) — agreement rise with confidence fall indicates
/// subsystems converging but output not reflecting it → needs gentle correction.
pub const AGREEMENT_CONFIDENCE_COUPLING_THRESHOLD: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// CONTEXT PHI MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Base scale factor for context-phi modulation of unified Psi.
/// Science: Baars (2002) — global workspace context shapes conscious access weighting.
pub const CONTEXT_PHI_SCALE_BASE: f32 = 0.8;

/// Range added to base scale proportional to context_phi_weight.
/// Full range: [0.8, 1.2] maps no-context → full-context.
pub const CONTEXT_PHI_SCALE_RANGE: f32 = 0.4;

// ═══════════════════════════════════════════════════════════════════════════════
// LOVE RESONANCE COUPLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum harmonic love resonance to trigger confidence/soul amplification.
/// Science: Fredrickson (2013) — positivity resonance requires threshold mutual engagement.
pub const LOVE_RESONANCE_THRESHOLD: f32 = 0.6;

/// Scale factor for love resonance → confidence boost (per unit above threshold).
/// Boost = (resonance - threshold) * scale.
pub const LOVE_RESONANCE_CONFIDENCE_SCALE: f32 = 0.04;

/// Fraction of love resonance boost applied to learning rate.
/// LR *= 1.0 + boost * fraction.
pub const LOVE_RESONANCE_LR_FRACTION: f32 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// REASONING CHAIN CONFIDENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum reasoning chain confidence for chain-depth boost to apply.
/// Science: Stanovich (2011) — Type 2 reasoning only boosts confidence when reliable.
pub const REASONING_CHAIN_CONFIDENCE_THRESHOLD: f32 = 0.7;

/// Scale factor for reasoning chain confidence boost (per unit above threshold).
pub const REASONING_CHAIN_BOOST_SCALE: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// SOCIAL LEARNING MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Base learning rate factor for social trust modulation.
/// Science: Decety & Chaminade (2003) — social trust modulates learning engagement.
pub const SOCIAL_LR_BASE: f32 = 0.8;

/// Range of social trust modulation (LR factor = BASE + RANGE * trust).
/// Full range: [0.8, 1.2] maps distrust → full trust.
pub const SOCIAL_LR_RANGE: f32 = 0.4;

/// Theory-of-Mind accuracy above which prediction confidence is boosted.
/// Science: Frith & Frith (2006) — accurate mentalizing reinforces social predictions.
pub const TOM_ACCURACY_HIGH: f32 = 0.7;

/// Theory-of-Mind accuracy below which prediction confidence is dampened.
pub const TOM_ACCURACY_LOW: f32 = 0.3;

/// Scale factor for ToM accuracy → confidence adjustment (per unit from threshold).
pub const TOM_ACCURACY_SCALE: f32 = 0.05;

/// Minimum causal average confidence to trigger urgency gating.
/// Science: Pearl (2009) — causal reasoning requires sufficient confidence in causal links.
pub const CAUSAL_URGENCY_CONFIDENCE: f32 = 0.6;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-MODULE AGREEMENT ACTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Confidence boost scale when cross-module agreement is high.
pub const AGREEMENT_HIGH_CONFIDENCE_SCALE: f32 = 0.05;

/// Confidence dampen scale when cross-module agreement is low.
pub const AGREEMENT_LOW_CONFIDENCE_SCALE: f32 = 0.1;

/// Exploration boost scale when cross-module agreement is low.
/// Science: Dehaene (2011) — disagreement across modules drives exploratory behavior.
pub const AGREEMENT_LOW_EXPLORATION_SCALE: f32 = 0.15;

/// Very low agreement threshold triggering cautious interpretation.
pub const AGREEMENT_CRITICAL_THRESHOLD: f32 = 0.2;

/// Threshold scale-up factor when agreement is critically low.
pub const AGREEMENT_CRITICAL_CAUTION_SCALE: f32 = 1.2;

/// EMA decay factor for tracking average cross-module agreement.
pub const AGREEMENT_EMA_DECAY: f32 = 0.95;

/// Amplification factor for variance in cross-module agreement computation.
pub const CROSS_MODULE_VARIANCE_AMPLIFICATION: f32 = 4.0;

// ═══════════════════════════════════════════════════════════════════════════════
// COMPOUND INSTABILITY DETECTION
// ═══════════════════════════════════════════════════════════════════════════════

/// Agreement velocity threshold for compound instability (negative = dropping).
/// Science: Friston (2010) — cascading precision failures require active recovery.
pub const COMPOUND_INSTABILITY_VELOCITY: f32 = -0.10;

/// Error slope threshold for compound instability detection.
pub const COMPOUND_INSTABILITY_ERROR_SLOPE: f32 = 0.02;

/// Learning rate scale during compound instability (protective dampening).
pub const COMPOUND_INSTABILITY_LR_SCALE: f32 = 0.93;

/// Exploration boost during compound instability.
pub const COMPOUND_INSTABILITY_EXPLORATION: f32 = 0.025;

/// Agreement velocity threshold for rapid-drop preemptive response.
pub const AGREEMENT_VELOCITY_DROP_THRESHOLD: f32 = -0.15;

/// Learning rate scale during rapid agreement drop.
pub const AGREEMENT_VELOCITY_DROP_LR: f32 = 0.97;

/// Exploration boost during rapid agreement drop.
pub const AGREEMENT_VELOCITY_DROP_EXPLORATION: f32 = 0.015;

/// Confidence coupling scale when agreement rises but confidence falls.
/// Science: Tononi (2004) — integration bottleneck requires gentle correction.
pub const AGREEMENT_CONFIDENCE_COUPLING_SCALE: f32 = 0.98;

/// Coherence velocity threshold for agreement-confidence coupling detection.
pub const AGREEMENT_COHERENCE_VELOCITY_THRESHOLD: f32 = -0.01;

// ═══════════════════════════════════════════════════════════════════════════════
// QUALITY GATING
// ═══════════════════════════════════════════════════════════════════════════════

/// EMA decay factor for unified quality score tracking.
pub const QUALITY_EMA_DECAY: f32 = 0.9;

/// Learning rate boost scale when quality is high (per unit above threshold).
pub const QUALITY_HIGH_LR_SCALE: f32 = 0.25;

/// Minimum learning rate factor after quality-based clamping.
pub const QUALITY_LR_CLAMP_MIN: f32 = 0.7;

/// Maximum learning rate factor after quality-based clamping.
pub const QUALITY_LR_CLAMP_MAX: f32 = 1.5;

/// Exploration dampening factor when quality is low (multiplicative).
pub const LOW_QUALITY_EXPLORATION_DAMPEN: f32 = 0.9;

// ═══════════════════════════════════════════════════════════════════════════════
// ENTROPY LR MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum learning rate modifier from harmony entropy (at zero entropy).
/// Science: broader value-space exploration → richer training signal.
pub const ENTROPY_LR_MIN: f64 = 0.95;

/// Range of entropy-based LR modulation. Full range: [MIN, MIN+RANGE].
pub const ENTROPY_LR_RANGE: f64 = 0.10;

// ═══════════════════════════════════════════════════════════════════════════════
// SOCIAL RELATIONAL DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Phi divergence threshold for triggering exploration boost.
/// Science: Friston (2010) — high divergence = high epistemic value.
pub const PHI_DIVERGENCE_THRESHOLD: f64 = 0.1;

/// Maximum phi divergence considered for exploration scaling.
pub const PHI_DIVERGENCE_MAX: f64 = 0.2;

/// Exploration scale factor for phi divergence boost.
pub const PHI_DIVERGENCE_SCALE: f64 = 0.15;

/// Phi relational threshold for oxytocin production.
/// Science: Feldman (2012) — relational coherence drives oxytocin release.
pub const PHI_RELATIONAL_OXY_THRESHOLD: f64 = 0.3;

/// Oxytocin production scale per unit of relational phi above threshold.
pub const PHI_RELATIONAL_OXY_SCALE: f64 = 0.05;

/// Midpoint for trust evolution from cycle coherence.
/// Science: Bowlby (1969) — secure attachment requires consistent responsiveness.
pub const TRUST_SIGNAL_MIDPOINT: f64 = 0.5;

/// Rate of trust change per coherence unit above/below midpoint.
pub const TRUST_SIGNAL_RATE: f64 = 0.01;

/// Decay factor preventing runaway trust accumulation.
pub const TRUST_DECAY_FACTOR: f64 = 0.999;

// ═══════════════════════════════════════════════════════════════════════════════
// REST / DREAM MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Coherence weight during active rest (>1.0 = emphasize coherence).
/// Science: Tononi & Cirelli (2006) — rest-state consciousness emphasizes integration.
pub const REST_COHERENCE_WEIGHT: f32 = 1.2;

/// Binding intensity dampen during active rest (<1.0 = reduce binding weight).
pub const REST_BINDING_DAMPEN: f32 = 0.8;

/// Fraction of rest modulation attributed to coherence component.
pub const REST_MODULATION_COHERENCE_FRAC: f32 = 0.6;

/// Fraction of rest modulation attributed to binding component.
pub const REST_MODULATION_BINDING_FRAC: f32 = 0.4;

// ═══════════════════════════════════════════════════════════════════════════════
// EMPATHIC SPEECH MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum empathic tone adjustment magnitude to trigger speech rate modulation.
/// Science: Porges (2011) — polyvagal theory links prosody to social engagement.
pub const EMPATHIC_TONE_THRESHOLD: f32 = 0.1;

/// Rate at which empathic tone modulates speech rate (multiplicative).
pub const EMPATHIC_TONE_RATE_SCALE: f32 = 0.1;

/// Minimum speech rate multiplier after empathic modulation.
pub const SPEECH_RATE_CLAMP_MIN: f32 = 0.6;

/// Maximum speech rate multiplier after empathic modulation.
pub const SPEECH_RATE_CLAMP_MAX: f32 = 1.5;

// ═══════════════════════════════════════════════════════════════════════════════
// SOCIAL TRUST STRATEGY BIAS
// Science: Decety & Chaminade (2003) — social trust modulates cooperative strategy.
// ═══════════════════════════════════════════════════════════════════════════════

/// Trust midpoint — deviation from this triggers social strategy bias.
pub const SOCIAL_TRUST_MIDPOINT: f32 = 0.5;
/// Trust deviation deadzone — below this, no strategy bias applied.
pub const SOCIAL_TRUST_DEADZONE: f32 = 0.1;
/// Minimum cooperation rate for trust-based strategy upgrade.
pub const SOCIAL_COOPERATION_THRESHOLD: f32 = 0.3;
/// Scaling factor for trust deviation → bias strength mapping.
/// Maps deviation [0.1, 0.5] → strength [0, 1].
pub const SOCIAL_TRUST_STRENGTH_SCALE: f32 = 2.5;
/// Minimum strength for strategy override (Concise→Supportive or Exploratory→Detailed).
pub const SOCIAL_TRUST_OVERRIDE_THRESHOLD: f32 = 0.5;
/// Minimum strength for exploration adjustment (no strategy override).
pub const SOCIAL_TRUST_EXPLORE_THRESHOLD: f32 = 0.3;
/// Exploration adjustment scale from social trust.
pub const SOCIAL_TRUST_EXPLORE_SCALE: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// TOM PREDICTION MISMATCH
// Science: Frith & Frith (2006) — ToM mismatch drives epistemic exploration.
// ═══════════════════════════════════════════════════════════════════════════════

/// EMA decay for ToM prediction mismatch tracking.
pub const TOM_MISMATCH_EMA_DECAY: f32 = 0.9;
/// ToM mismatch threshold for exploration trigger.
pub const TOM_MISMATCH_THRESHOLD: f32 = 0.4;
/// Exploration boost scale from ToM mismatch.
pub const TOM_MISMATCH_EXPLORE_SCALE: f32 = 0.1;

// ═══════════════════════════════════════════════════════════════════════════════
// SOUL ALIGNMENT LR MODULATION
// Science: Haidt (2001) — moral alignment reinforces learning.
// ═══════════════════════════════════════════════════════════════════════════════

/// Soul alignment threshold for LR boost (positive alignment).
pub const SOUL_ALIGNMENT_BOOST_THRESHOLD: f32 = 0.3;
/// LR boost scale from positive soul alignment.
pub const SOUL_ALIGNMENT_BOOST_SCALE: f32 = 0.1;
/// LR clamp minimum for positive soul alignment.
pub const SOUL_ALIGNMENT_BOOST_LR_MIN: f32 = 0.8;
/// LR clamp maximum for positive soul alignment.
pub const SOUL_ALIGNMENT_BOOST_LR_MAX: f32 = 1.3;
/// Soul alignment threshold for LR dampening (negative alignment).
pub const SOUL_ALIGNMENT_DAMPEN_THRESHOLD: f32 = -0.3;
/// LR dampening scale from negative soul alignment.
pub const SOUL_ALIGNMENT_DAMPEN_SCALE: f32 = 0.15;
/// LR clamp minimum for negative soul alignment.
pub const SOUL_ALIGNMENT_DAMPEN_LR_MIN: f32 = 0.7;
/// LR clamp maximum for negative soul alignment.
pub const SOUL_ALIGNMENT_DAMPEN_LR_MAX: f32 = 1.2;

// ═══════════════════════════════════════════════════════════════════════════════
// SURPRISE-DRIVEN EXPLORATION
// Science: Friston (2005) — prediction error drives epistemic action.
// ═══════════════════════════════════════════════════════════════════════════════

/// Prediction error threshold for surprise-driven exploration.
pub const SURPRISE_PE_THRESHOLD: f32 = 0.2;
/// Maximum PE excess for scale computation.
pub const SURPRISE_PE_EXCESS_CAP: f32 = 0.5;
/// PE scale factor for exploration intensity.
pub const SURPRISE_PE_SCALE_FACTOR: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// MEMO THRESHOLD DIVERSITY ADAPTATION
// Science: Baddeley (2012) — working memory adapts to stimulus diversity.
// ═══════════════════════════════════════════════════════════════════════════════

/// Low codebook diversity threshold → lower memo threshold (more memoization).
pub const MEMO_DIVERSITY_LOW: f32 = 0.4;
/// Adjustment scale for low diversity → threshold reduction.
pub const MEMO_DIVERSITY_LOW_SCALE: f32 = 0.1;
/// Minimum memo threshold floor.
pub const MEMO_THRESHOLD_FLOOR: f32 = 0.88;
/// High codebook diversity threshold → raise memo threshold (less memoization).
pub const MEMO_DIVERSITY_HIGH: f32 = 0.8;
/// Adjustment scale for high diversity → threshold increase.
pub const MEMO_DIVERSITY_HIGH_SCALE: f32 = 0.05;
/// Maximum memo threshold ceiling.
pub const MEMO_THRESHOLD_CEILING: f32 = 0.98;

// ═══════════════════════════════════════════════════════════════════════════════
// THETA-PHASE BINDING
// Science: Lisman & Jensen (2013) — theta-gamma coupling for feature binding.
// ═══════════════════════════════════════════════════════════════════════════════

/// Default salience when Phi attention returns no weights.
pub const THETA_DEFAULT_SALIENCE: f32 = 0.1;
/// Minimum salience clamp for theta-phase binding.
pub const THETA_SALIENCE_CLAMP_MIN: f32 = 0.05;
/// Minimum binding strength clamp.
pub const THETA_BINDING_CLAMP_MIN: f32 = 0.1;
/// Maximum binding strength clamp.
pub const THETA_BINDING_CLAMP_MAX: f32 = 0.9;
/// Binding strength threshold for temporal binding boost.
pub const THETA_BINDING_BOOST_THRESHOLD: f32 = 0.25;

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIDENCE / EXPLORATION ADAPTIVE SCALE
// Science: Daw (2006) — exploitation-exploration tradeoff.
// ═══════════════════════════════════════════════════════════════════════════════

/// Confidence midpoint for adaptive threshold scaling.
pub const CONFIDENCE_SCALE_MIDPOINT: f32 = 0.5;
/// Confidence sensitivity for threshold scaling.
pub const CONFIDENCE_SCALE_SENSITIVITY: f32 = 0.4;
/// Exploration midpoint for adaptive threshold scaling.
pub const EXPLORATION_SCALE_MIDPOINT: f32 = 0.5;
/// Exploration sensitivity for threshold scaling (negative = more exploration → lower threshold).
pub const EXPLORATION_SCALE_SENSITIVITY: f32 = 0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_passes() {
        validate(); // Should not panic
    }

    #[test]
    fn test_moral_ordering() {
        assert!(MORAL_CONCERN_THRESHOLD < 0.0);
        assert!(MORAL_BENEFIT_THRESHOLD > 0.0);
        assert!(MORAL_CONCERN_THRESHOLD < MORAL_BENEFIT_THRESHOLD);
    }

    #[test]
    fn test_dominance_ordering() {
        assert!(DOMINANCE_DEFAULT < DOMINANCE_CONFIDENT);
        assert!(DOMINANCE_CONFIDENT < DOMINANCE_FLOW_BASE);
        assert!(DOMINANCE_FLOW_BASE + DOMINANCE_FLOW_SCALE <= 1.0);
    }

    #[test]
    fn test_decay_rates_valid() {
        assert!((0.0..1.0).contains(&FEP_LR_DECAY));
        assert!((0.0..1.0).contains(&MCE_BOOST_DECAY));
    }

    #[test]
    fn test_psi_weights_sum() {
        let total = FLOW_PSI_WEIGHT as f64
            + RELATIONAL_PSI_WEIGHT as f64
            + BODY_PSI_WEIGHT
            + EMBODIED_PSI_WEIGHT;
        assert!(total <= 1.0, "Psi weights sum to {}", total);
    }

    #[test]
    fn test_attention_budget_reasonable() {
        // Should be between 10ms and 500ms
        assert!(ATTENTION_BUDGET_US >= 10_000);
        assert!(ATTENTION_BUDGET_US <= 500_000);
    }

    #[test]
    fn test_policy_window_ordering() {
        assert!(POLICY_MIN_WINDOW < POLICY_WINDOW_SIZE);
        assert!(POLICY_WINDOW_SIZE > 0);
    }

    #[test]
    fn test_reward_scaling_sensible() {
        assert!(REWARD_GOOD_BASE > 0.0);
        assert!(REWARD_BAD_BASE < 0.0);
        assert!((0.0..=1.0).contains(&REWARD_EXTERNAL_BLEND));
    }

    #[test]
    fn test_self_model_ordering() {
        assert!(SELF_MODEL_LOW_THRESHOLD < SELF_MODEL_HIGH_THRESHOLD);
        assert!(SELF_MODEL_ACCURACY_EMA > 0.0 && SELF_MODEL_ACCURACY_EMA < 1.0);
    }

    #[test]
    fn test_epistemic_ordering() {
        assert!(EPISTEMIC_CAUTION_THRESHOLD < EPISTEMIC_APPROVAL_THRESHOLD);
        assert!(EPISTEMIC_APPROVAL_THRESHOLD < EPISTEMIC_TRUST_THRESHOLD);
    }

    #[test]
    fn test_binding_ordering() {
        assert!(BINDING_LOW_THRESHOLD < BINDING_CONFIDENCE_THRESHOLD);
    }

    #[test]
    fn test_thalamic_budget_ordering() {
        assert!(THALAMIC_REFLEX_BUDGET_SCALE < 1.0);
        assert!(THALAMIC_DEEP_BUDGET_SCALE > 1.0);
        assert!(THALAMIC_REFLEX_LR_FACTOR < 1.0);
        assert!(THALAMIC_DEEP_LR_FACTOR > 1.0);
    }

    #[test]
    fn test_phi_validation_ordering() {
        assert!(PHI_VALIDATION_LOW_THRESHOLD < PHI_VALIDATION_HIGH_THRESHOLD);
        assert!(PHI_VALIDATION_LOW_THRESHOLD > 0.0);
        assert!(PHI_VALIDATION_HIGH_THRESHOLD < 1.0);
    }

    #[test]
    fn test_phi_gating_ordering() {
        assert!(PHI_REACTIVE_THRESHOLD < PHI_INTEGRATIVE_THRESHOLD);
        assert!(PHI_REACTIVE_THRESHOLD > 0.0);
        assert!(PHI_INTEGRATIVE_THRESHOLD < 1.0);
    }

    #[test]
    fn test_q_learning_params() {
        assert!((0.0..=1.0).contains(&Q_VALUE_INITIAL));
        assert!((0.0..=1.0).contains(&Q_LEARNING_RATE));
        assert!(EXPLORATION_RATE_MIN < EXPLORATION_RATE_INITIAL);
        assert!((0.0..1.0).contains(&EXPLORATION_DECAY_RATE));
    }

    #[test]
    fn test_reward_threshold_ordering() {
        assert!(REWARD_NEGATIVE_THRESHOLD < 0.0);
        assert!(REWARD_POSITIVE_THRESHOLD > 0.0);
        assert!(REWARD_NEGATIVE_THRESHOLD < REWARD_POSITIVE_THRESHOLD);
    }

    #[test]
    fn test_temporal_dynamics_params() {
        assert!(TEMPORAL_REPLAY_TRIGGER > 0.0);
        assert!(TEMPORAL_CONTINUITY_BOOST_THRESHOLD > 0.0);
        assert!(TEMPORAL_CHAIN_BOOST_FACTOR > 0.0);
    }

    #[test]
    fn test_harmonic_field_params() {
        assert!(HARMONIC_FIELD_BOOST_THRESHOLD > 0.0);
        assert!(HARMONIC_FIELD_BOOST_THRESHOLD < 1.0);
        assert!(HARMONIC_FIELD_BOOST_FACTOR > 0.0);
    }

    #[test]
    fn test_cfc_tau_modulation_params() {
        assert!(AROUSAL_TAU_DEADZONE > 0.0);
        assert!(AROUSAL_TAU_SENSITIVITY > 0.0 && AROUSAL_TAU_SENSITIVITY <= 0.5);
        assert!(CODEBOOK_FAMILIAR_THRESHOLD > CODEBOOK_NOVEL_THRESHOLD);
        assert!(CODEBOOK_FAMILIAR_TAU_SCALE > 0.0);
        assert!(CODEBOOK_NOVEL_TAU_SCALE > 0.0);
        assert!(AROUSAL_RECOVERY_TAU_SCALE > 0.0 && AROUSAL_RECOVERY_TAU_SCALE <= 0.5);
        assert!(FEP_SURPRISE_TAU_SCALE > 0.0 && FEP_SURPRISE_TAU_SCALE <= 0.5);
    }

    #[test]
    fn test_pe_variance_params() {
        assert!(PE_VARIANCE_THRESHOLD > 0.0);
        assert!(PE_VARIANCE_MAX_EFFECT > PE_VARIANCE_THRESHOLD);
        assert!(PE_VARIANCE_DAMPEN_SCALE > 0.0);
    }

    #[test]
    fn test_homeostasis_efficiency_params() {
        assert!(HOMEOSTASIS_EFFICIENCY_EMA > 0.0 && HOMEOSTASIS_EFFICIENCY_EMA < 1.0);
        assert!(TRANSITION_COST_THRESHOLD > 0.0);
        assert!(TRANSITION_COST_MAX_EFFECT > 0.0);
        assert!(TRANSITION_COST_STRENGTH_SCALE > 0.0);
    }

    #[test]
    fn test_epistemic_semantic_params() {
        assert!(EPISTEMIC_SEMANTIC_CAUTION_THRESHOLD > 0.0);
        assert!(EPISTEMIC_SEMANTIC_BOOST_THRESHOLD > EPISTEMIC_SEMANTIC_CAUTION_THRESHOLD);
        assert!(EPISTEMIC_SEMANTIC_CAUTION_BASE > 0.0);
        assert!(EPISTEMIC_SEMANTIC_CAUTION_SCALE > 0.0);
        assert!(EPISTEMIC_SEMANTIC_BOOST_SCALE > 0.0);
    }

    #[test]
    fn test_epistemic_exploration_params() {
        assert!(EPISTEMIC_EXPLORE_THRESHOLD > 0.0);
        assert!(EPISTEMIC_EXPLORE_SCALE > 0.0);
        assert!(
            EPISTEMIC_LOW_THRESHOLD > 0.0 && EPISTEMIC_LOW_THRESHOLD < EPISTEMIC_EXPLORE_THRESHOLD
        );
        assert!(EPISTEMIC_LOW_DAMPEN > 0.0);
        assert!(EPISTEMIC_OSCILLATION_THRESHOLD > 0.0);
        assert!(EPISTEMIC_OSCILLATION_MULTIPLIER > 1.0);
    }

    #[test]
    fn test_mcts_effectiveness_params() {
        assert!(MCTS_EFFECTIVENESS_HIGH > MCTS_EFFECTIVENESS_LOW);
        assert!(MCTS_EFFECTIVENESS_CONFIDENCE_SCALE > 0.0);
        assert!(MCTS_EFFECTIVENESS_EXPLORE_SCALE > 0.0);
        assert!(MCTS_EFFECTIVENESS_EMA > 0.0 && MCTS_EFFECTIVENESS_EMA < 1.0);
        assert!(MCTS_PLAN_CONFIDENCE_THRESHOLD > 0.0);
        assert!(MCTS_PLAN_WEIGHT_SCALE > 0.0);
    }

    #[test]
    fn test_self_model_accuracy_weights() {
        assert!((SELF_MODEL_CONFIDENCE_WEIGHT + SELF_MODEL_URGENCY_WEIGHT - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_theta_and_horizon_params() {
        assert!(THETA_PHASE_ADVANCE > 0.0);
        assert!(THETA_PHASE_ADVANCE < std::f64::consts::PI); // less than half-cycle per step
        assert!(THETA_PHI_MODULATION_AMPLITUDE > 0.0);
        assert!(THETA_PHI_MODULATION_AMPLITUDE <= 0.2); // don't modulate >20%
        assert!(THETA_PHI_SMOOTH_ALPHA > 0.0);
        assert!(THETA_PHI_SMOOTH_ALPHA < 1.0);
        assert!(PREDICTION_HORIZON_MIN_SCALE > 0.0);
        assert!(PREDICTION_HORIZON_MIN_SCALE < 1.0);
        assert!(PREDICTION_HORIZON_MAX_SCALE > 1.0);
        assert!(PREDICTION_HORIZON_MIN_SCALE < PREDICTION_HORIZON_MAX_SCALE);
        assert!(LOW_COHERENCE_EXPLORATION_THRESHOLD > 0);
        assert!(LOW_COHERENCE_EXPLORATION_BOOST > 0.0);
        assert!(LOW_COHERENCE_EXPLORATION_BOOST < 0.1); // don't over-explore
                                                        // Prediction horizon PE-adaptive scaling
        assert!(HORIZON_PE_CONTRACT_THRESHOLD > 0.0 && HORIZON_PE_CONTRACT_THRESHOLD < 1.0);
        assert!(HORIZON_PE_CONTRACT_RATE > 0.0 && HORIZON_PE_CONTRACT_RATE <= 1.0);
        assert!(
            HORIZON_PE_EXPAND_THRESHOLD > 0.0
                && HORIZON_PE_EXPAND_THRESHOLD < HORIZON_PE_CONTRACT_THRESHOLD
        );
        assert!(HORIZON_PE_EXPAND_RATE > 0.0);
        assert!(HORIZON_SLOPE_THRESHOLD > 0.0);
        assert!(HORIZON_SLOPE_CONTRACT_CAP > 0.0);
        assert!(HORIZON_SLOPE_CONTRACT_RATE > 0.0);
        assert!(HORIZON_SLOPE_EXPAND_CAP > 0.0);
        assert!(HORIZON_SLOPE_EXPAND_RATE > 0.0);
        // At max PE, contraction stays above floor
        let worst_pe_scale = 1.0 - (1.0 - HORIZON_PE_CONTRACT_THRESHOLD) * HORIZON_PE_CONTRACT_RATE;
        assert!(
            worst_pe_scale > PREDICTION_HORIZON_MIN_SCALE,
            "PE contraction ({}) would breach floor ({})",
            worst_pe_scale,
            PREDICTION_HORIZON_MIN_SCALE
        );
    }

    #[test]
    fn test_session10_params() {
        // Confidence crash
        assert!(CONFIDENCE_CRASH_THRESHOLD > 0.0 && CONFIDENCE_CRASH_THRESHOLD < 1.0);
        assert!(CONFIDENCE_CRASH_FREEZE_CYCLES > 0 && CONFIDENCE_CRASH_FREEZE_CYCLES <= 10);
        assert!(CONFIDENCE_CRASH_EXPLORATION_BOOST > 0.0);
        // Self-model weighting
        assert!(SELF_MODEL_WEIGHT_HIGH_THRESHOLD > SELF_MODEL_WEIGHT_LOW_THRESHOLD);
        assert!(SELF_MODEL_WEIGHT_BONUS > 0.0 && SELF_MODEL_WEIGHT_BONUS < 0.5);
        assert!(SELF_MODEL_WEIGHT_PENALTY > 0.5 && SELF_MODEL_WEIGHT_PENALTY < 1.0);
        // Coherence velocity tau
        assert!(COHERENCE_VELOCITY_TAU_BOOST > 1.0);
        assert!(COHERENCE_VELOCITY_TAU_DAMPEN < 1.0 && COHERENCE_VELOCITY_TAU_DAMPEN > 0.0);
        assert!(COHERENCE_VELOCITY_TAU_THRESHOLD > 0.0);
        // Homeostasis efficiency adaptation
        assert!(HOMEOSTASIS_EFFICIENCY_HIGH > 1.0);
        assert!(HOMEOSTASIS_EFFICIENCY_LOW < 1.0 && HOMEOSTASIS_EFFICIENCY_LOW > 0.0);
        assert!(HOMEOSTASIS_PULL_REDUCTION < 1.0);
        assert!(HOMEOSTASIS_PULL_INCREASE > 1.0);
        // Error pattern LR
        assert!(ERROR_PATTERN_RISING_LR > 1.0);
        assert!(ERROR_PATTERN_FALLING_LR < 1.0);
        assert!(ERROR_PATTERN_OSCILLATING_LR < ERROR_PATTERN_FALLING_LR);
        // Proposal diversity
        assert!(PROPOSAL_DIVERSITY_MIN_SOURCES >= 2);
        assert!(PROPOSAL_DIVERSITY_WARMUP > 0);
        assert!(PROPOSAL_DIVERSITY_EXPLORATION_BOOST > 0.0);
        // Hysteresis relaxation
        assert!(HYSTERESIS_RELAXATION_THRESHOLD > 0);
        assert!(HYSTERESIS_RELAXATION_RATE > 0.0 && HYSTERESIS_RELAXATION_RATE < 1.0);
        assert!(HYSTERESIS_RELAXATION_FLOOR > 0.0 && HYSTERESIS_RELAXATION_FLOOR < 1.0);
        // Agreement-confidence coupling
        assert!(AGREEMENT_CONFIDENCE_COUPLING_THRESHOLD > 0.0);
    }

    #[test]
    fn test_context_phi_modulation() {
        assert!(CONTEXT_PHI_SCALE_BASE > 0.0 && CONTEXT_PHI_SCALE_BASE < 1.0);
        assert!(CONTEXT_PHI_SCALE_RANGE > 0.0);
        assert!(CONTEXT_PHI_SCALE_BASE + CONTEXT_PHI_SCALE_RANGE <= 1.5);
    }

    #[test]
    fn test_love_resonance_params() {
        assert!(LOVE_RESONANCE_THRESHOLD > 0.0 && LOVE_RESONANCE_THRESHOLD < 1.0);
        assert!(LOVE_RESONANCE_CONFIDENCE_SCALE > 0.0);
        assert!(LOVE_RESONANCE_LR_FRACTION > 0.0 && LOVE_RESONANCE_LR_FRACTION <= 1.0);
    }

    #[test]
    fn test_reasoning_chain_params() {
        assert!(REASONING_CHAIN_CONFIDENCE_THRESHOLD > 0.5);
        assert!(REASONING_CHAIN_BOOST_SCALE > 0.0 && REASONING_CHAIN_BOOST_SCALE < 0.5);
    }

    #[test]
    fn test_social_learning_params() {
        assert!(SOCIAL_LR_BASE > 0.0 && SOCIAL_LR_BASE < 1.0);
        assert!(SOCIAL_LR_BASE + SOCIAL_LR_RANGE > 1.0);
        assert!(TOM_ACCURACY_HIGH > TOM_ACCURACY_LOW);
        assert!(TOM_ACCURACY_SCALE > 0.0 && TOM_ACCURACY_SCALE < 0.5);
    }

    #[test]
    fn test_agreement_action_params() {
        assert!(AGREEMENT_HIGH_CONFIDENCE_SCALE > 0.0);
        assert!(AGREEMENT_LOW_CONFIDENCE_SCALE > 0.0);
        assert!(AGREEMENT_LOW_EXPLORATION_SCALE > 0.0);
        assert!(AGREEMENT_CRITICAL_THRESHOLD < CROSS_MODULE_AGREEMENT_LOW);
        assert!(AGREEMENT_CRITICAL_CAUTION_SCALE > 1.0);
        assert!(AGREEMENT_EMA_DECAY > 0.0 && AGREEMENT_EMA_DECAY < 1.0);
        assert!(CROSS_MODULE_VARIANCE_AMPLIFICATION > 1.0);
    }

    #[test]
    fn test_compound_instability_params() {
        assert!(COMPOUND_INSTABILITY_VELOCITY < 0.0);
        assert!(COMPOUND_INSTABILITY_ERROR_SLOPE > 0.0);
        assert!(COMPOUND_INSTABILITY_LR_SCALE > 0.5 && COMPOUND_INSTABILITY_LR_SCALE < 1.0);
        assert!(AGREEMENT_VELOCITY_DROP_THRESHOLD < COMPOUND_INSTABILITY_VELOCITY);
        assert!(AGREEMENT_VELOCITY_DROP_LR > COMPOUND_INSTABILITY_LR_SCALE);
    }

    #[test]
    fn test_quality_gating_params() {
        assert!(QUALITY_EMA_DECAY > 0.0 && QUALITY_EMA_DECAY < 1.0);
        assert!(QUALITY_HIGH_LR_SCALE > 0.0);
        assert!(QUALITY_LR_CLAMP_MIN < 1.0);
        assert!(QUALITY_LR_CLAMP_MAX > 1.0);
    }

    #[test]
    fn test_entropy_lr_params() {
        assert!(ENTROPY_LR_MIN > 0.0 && ENTROPY_LR_MIN < 1.0);
        assert!(ENTROPY_LR_MIN + ENTROPY_LR_RANGE > 1.0);
    }

    #[test]
    fn test_social_relational_params() {
        assert!(PHI_DIVERGENCE_THRESHOLD > 0.0 && PHI_DIVERGENCE_THRESHOLD < 1.0);
        assert!(PHI_DIVERGENCE_MAX > PHI_DIVERGENCE_THRESHOLD);
        assert!(PHI_DIVERGENCE_SCALE > 0.0);
        assert!(PHI_RELATIONAL_OXY_THRESHOLD > 0.0 && PHI_RELATIONAL_OXY_THRESHOLD < 1.0);
        assert!(PHI_RELATIONAL_OXY_SCALE > 0.0);
        assert!(TRUST_SIGNAL_MIDPOINT > 0.0 && TRUST_SIGNAL_MIDPOINT < 1.0);
        assert!(TRUST_SIGNAL_RATE > 0.0 && TRUST_SIGNAL_RATE < 0.1);
        assert!(TRUST_DECAY_FACTOR > 0.9 && TRUST_DECAY_FACTOR < 1.0);
    }

    #[test]
    fn test_rest_modulation_params() {
        assert!(REST_COHERENCE_WEIGHT > 1.0);
        assert!(REST_BINDING_DAMPEN < 1.0);
        assert!((REST_MODULATION_COHERENCE_FRAC + REST_MODULATION_BINDING_FRAC - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empathic_speech_params() {
        assert!(EMPATHIC_TONE_THRESHOLD > 0.0);
        assert!(EMPATHIC_TONE_RATE_SCALE > 0.0);
        assert!(SPEECH_RATE_CLAMP_MIN < 1.0);
        assert!(SPEECH_RATE_CLAMP_MAX > 1.0);
    }

    #[test]
    fn test_social_trust_strategy_params() {
        assert!(SOCIAL_TRUST_MIDPOINT > 0.0 && SOCIAL_TRUST_MIDPOINT < 1.0);
        assert!(SOCIAL_TRUST_DEADZONE > 0.0);
        assert!(SOCIAL_TRUST_STRENGTH_SCALE > 0.0);
        assert!(
            SOCIAL_TRUST_OVERRIDE_THRESHOLD > SOCIAL_TRUST_EXPLORE_THRESHOLD,
            "Override ({}) must exceed explore threshold ({})",
            SOCIAL_TRUST_OVERRIDE_THRESHOLD,
            SOCIAL_TRUST_EXPLORE_THRESHOLD
        );
        assert!(SOCIAL_COOPERATION_THRESHOLD > 0.0 && SOCIAL_COOPERATION_THRESHOLD < 1.0);
        assert!(SOCIAL_TRUST_EXPLORE_SCALE > 0.0);
    }

    #[test]
    fn test_tom_mismatch_params() {
        assert!(
            TOM_MISMATCH_EMA_DECAY > 0.0 && TOM_MISMATCH_EMA_DECAY < 1.0,
            "EMA decay must be in (0,1): {}",
            TOM_MISMATCH_EMA_DECAY
        );
        assert!(
            TOM_MISMATCH_THRESHOLD > 0.0 && TOM_MISMATCH_THRESHOLD < 1.0,
            "Threshold must be in (0,1): {}",
            TOM_MISMATCH_THRESHOLD
        );
        assert!(TOM_MISMATCH_EXPLORE_SCALE > 0.0);
    }

    #[test]
    fn test_soul_alignment_params() {
        assert!(SOUL_ALIGNMENT_BOOST_THRESHOLD > 0.0);
        assert!(SOUL_ALIGNMENT_DAMPEN_THRESHOLD < 0.0);
        // Symmetry check: |boost| == |dampen| threshold
        assert!(
            (SOUL_ALIGNMENT_BOOST_THRESHOLD - SOUL_ALIGNMENT_DAMPEN_THRESHOLD.abs()).abs() < 1e-6,
            "Boost ({}) and dampen ({}) thresholds should be symmetric",
            SOUL_ALIGNMENT_BOOST_THRESHOLD,
            SOUL_ALIGNMENT_DAMPEN_THRESHOLD
        );
        // LR clamp ranges are valid
        assert!(SOUL_ALIGNMENT_BOOST_LR_MIN < SOUL_ALIGNMENT_BOOST_LR_MAX);
        assert!(SOUL_ALIGNMENT_DAMPEN_LR_MIN < SOUL_ALIGNMENT_DAMPEN_LR_MAX);
        assert!(SOUL_ALIGNMENT_BOOST_SCALE > 0.0);
        assert!(SOUL_ALIGNMENT_DAMPEN_SCALE > 0.0);
    }

    #[test]
    fn test_surprise_pe_params() {
        assert!(SURPRISE_PE_THRESHOLD > 0.0);
        assert!(SURPRISE_PE_EXCESS_CAP > 0.0);
        assert!(SURPRISE_PE_SCALE_FACTOR > 0.0);
    }

    #[test]
    fn test_memo_diversity_params() {
        assert!(
            MEMO_DIVERSITY_LOW < MEMO_DIVERSITY_HIGH,
            "Low diversity ({}) must be < high diversity ({})",
            MEMO_DIVERSITY_LOW,
            MEMO_DIVERSITY_HIGH
        );
        assert!(
            MEMO_THRESHOLD_FLOOR < MEMO_THRESHOLD_CEILING,
            "Floor ({}) must be < ceiling ({})",
            MEMO_THRESHOLD_FLOOR,
            MEMO_THRESHOLD_CEILING
        );
        assert!(MEMO_DIVERSITY_LOW_SCALE > 0.0);
        assert!(MEMO_DIVERSITY_HIGH_SCALE > 0.0);
    }

    #[test]
    fn test_theta_binding_params() {
        assert!(
            THETA_BINDING_CLAMP_MIN < THETA_BINDING_CLAMP_MAX,
            "Clamp min ({}) must be < clamp max ({})",
            THETA_BINDING_CLAMP_MIN,
            THETA_BINDING_CLAMP_MAX
        );
        assert!(
            THETA_BINDING_BOOST_THRESHOLD >= THETA_BINDING_CLAMP_MIN
                && THETA_BINDING_BOOST_THRESHOLD <= THETA_BINDING_CLAMP_MAX,
            "Boost threshold ({}) must be within clamp range [{}, {}]",
            THETA_BINDING_BOOST_THRESHOLD,
            THETA_BINDING_CLAMP_MIN,
            THETA_BINDING_CLAMP_MAX
        );
        assert!(THETA_DEFAULT_SALIENCE > 0.0);
        assert!(THETA_SALIENCE_CLAMP_MIN > 0.0);
        assert!(THETA_SALIENCE_CLAMP_MIN < THETA_DEFAULT_SALIENCE);
    }

    #[test]
    fn test_confidence_exploration_scale() {
        assert!(
            CONFIDENCE_SCALE_MIDPOINT > 0.0 && CONFIDENCE_SCALE_MIDPOINT < 1.0,
            "Confidence midpoint must be in (0,1): {}",
            CONFIDENCE_SCALE_MIDPOINT
        );
        assert!(CONFIDENCE_SCALE_SENSITIVITY > 0.0);
        assert!(
            EXPLORATION_SCALE_MIDPOINT > 0.0 && EXPLORATION_SCALE_MIDPOINT < 1.0,
            "Exploration midpoint must be in (0,1): {}",
            EXPLORATION_SCALE_MIDPOINT
        );
        assert!(EXPLORATION_SCALE_SENSITIVITY > 0.0);
    }
}
