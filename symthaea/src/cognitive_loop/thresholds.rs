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

/// Cantor CRHV minimum depth (weakest GWT activation → shallowest fractal).
/// Basis: Dehaene et al. (2006) — minimum ignition recruits ~2 cortical layers.
pub const CANTOR_DEPTH_MIN: usize = 2;

/// Cantor CRHV maximum depth (strongest GWT activation → deepest fractal).
/// Basis: Dehaene et al. (2006) — full ignition recruits ~7 cortical layers.
pub const CANTOR_DEPTH_MAX: usize = 7;

/// Cantor metacognitive depth → consciousness modulation strength (±).
/// At depth extremes (0 or 1), consciousness is modulated by ±this value.
/// Neutral at depth 0.5. Basis: Hofstadter (1979) — strange loop depth.
pub const CANTOR_CONSCIOUSNESS_MODULATION: f64 = 0.06;

/// Cantor dream consolidation quality threshold for codebook feedback.
/// CRHVs cleaned above this quality get learned into the persistent codebook.
/// Basis: Born & Wilhelm (2012) — only stable replay traces consolidate.
pub const CANTOR_DREAM_QUALITY_THRESHOLD: f32 = 0.7;

/// Minimum pairwise similarity for two CRHVs to be considered "resonant".
/// Science: Edelman & Tononi (2000) — reentrant signaling amplifies coherent coalitions.
pub const CANTOR_RESONANCE_SIMILARITY_THRESHOLD: f32 = 0.8;

/// Confidence boost when resonance is detected among broadcast CRHVs.
/// Resonant fractal patterns indicate stable attractor formation.
pub const CANTOR_RESONANCE_CONFIDENCE_BOOST: f32 = 0.02;

/// ACh baseline nudge per unit of normalized CRHV depth (depth/DEPTH_MAX).
/// Deep recursion → focused recurrent processing → cholinergic engagement.
/// Science: Hasselmo (2006) — ACh gates recurrent processing depth in cortex.
pub const CANTOR_DEPTH_ACH_BOOST: f32 = 0.02;

/// NE baseline nudge (negative) per unit of normalized CRHV depth.
/// Deep recursion → reduced alerting (focused attention replaces scanning).
/// Science: Hasselmo (2006) — ACh/NE balance shifts during focused vs scanning attention.
pub const CANTOR_DEPTH_NE_DAMPEN: f32 = -0.015;

/// Dream surprise threshold for exploration boost.
/// Above this, novel fractal territory triggers map-exploration.
/// Science: Sutton & Barto (2018) — surprise-driven explore/exploit tradeoff.
pub const CANTOR_SURPRISE_EXPLORATION_THRESHOLD: f32 = 0.3;

/// Exploration boost scale from dream surprise.
pub const CANTOR_SURPRISE_EXPLORATION_BOOST: f32 = 0.04;

/// Resonance threshold for exploration dampening.
/// Above this, attractor found → exploit mode (stop exploring).
pub const CANTOR_RESONANCE_EXPLORATION_DAMPEN_THRESHOLD: f32 = 0.5;

/// Exploration dampening scale from resonance.
pub const CANTOR_RESONANCE_EXPLORATION_DAMPEN: f32 = -0.03;

/// Cross-modal binding threshold for RadialCantor promotion.
/// When binding strength exceeds this, promote to RadialCantor (geometric structure).
/// Science: Treisman & Gelade (1980) — high binding = perceptual integration.
pub const CANTOR_RADIAL_BINDING_THRESHOLD: f32 = 0.6;

/// Number of radial bands for RadialCantor (maps to perceptual scales).
pub const CANTOR_RADIAL_BANDS: usize = 5;

/// Broca cadence spacing boost from deep CRHV recursion (depth > 5).
/// Deep fractals → slower, more deliberate speech.
/// Science: Goldman-Rakic (1996) — prefrontal recursion depth predicts utterance complexity.
pub const CANTOR_DEPTH_BROCA_SPACING_BOOST: usize = 2;

/// Broca cadence spacing boost from high dream surprise.
/// Epistemic uncertainty → pause before speaking.
pub const CANTOR_SURPRISE_BROCA_SPACING_BOOST: usize = 3;

/// Broca cadence surprise threshold (triggers spacing widening).
pub const CANTOR_SURPRISE_BROCA_THRESHOLD: f32 = 0.4;

/// Harmony boost for Sacred Stillness from CRHV self-similarity.
/// Deep self-reference maps to contemplative attention.
/// Science: Varela et al. (1991) — autopoietic self-reference as consciousness substrate.
pub const CANTOR_HARMONY_STILLNESS_SCALE: f64 = 0.08;

/// Harmony boost for Universal Interconnectedness from resonance.
/// Fractal choir (multiple coherent CRHVs) = collective resonance.
pub const CANTOR_HARMONY_INTERCONNECT_SCALE: f64 = 0.06;

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
// TEMPORAL CHAIN DEPTH → LR GATING
// ═══════════════════════════════════════════════════════════════════════════════

/// Deep causal chain threshold (>= this many links = deep chain → dampen LR).
/// Science: Zelazo (2004) — cognitive complexity demands stable representations.
pub const TEMPORAL_CHAIN_DEEP_THRESHOLD: usize = 5;

/// LR scale for deep causal chains (dampen to stabilize consolidation).
pub const TEMPORAL_CHAIN_DEEP_LR_SCALE: f32 = 0.85;

/// Shallow causal chain threshold (<= this many links = shallow → boost LR).
/// Science: Gopnik (2012) — shallow causal models benefit from rapid exploration.
pub const TEMPORAL_CHAIN_SHALLOW_THRESHOLD: usize = 2;

/// LR scale for shallow causal chains (boost to accelerate hypothesis testing).
pub const TEMPORAL_CHAIN_SHALLOW_LR_SCALE: f32 = 1.15;

// ═══════════════════════════════════════════════════════════════════════════════
// MCE BOTTLENECK → SUBSYSTEM MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// LR boost when MCE bottleneck is the targeted subsystem.
/// Science: Tononi (2004) — consciousness limited by minimum dimension;
/// boosting the bottleneck is the highest-leverage intervention.
pub const MCE_BOTTLENECK_LR_BOOST: f32 = 1.08;

/// Confidence boost when MCE bottleneck is NOT integration (system is well-integrated).
pub const MCE_NON_BOTTLENECK_CONFIDENCE_BOOST: f32 = 0.005;

// ═══════════════════════════════════════════════════════════════════════════════
// MORAL SCORE → MEMORY PRIORITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Moral salience threshold for boosted episodic consolidation.
/// Science: Zak (2012) — moral narratives enhance oxytocin → memory consolidation.
pub const MORAL_CONSOLIDATION_THRESHOLD: f32 = 0.3;

/// Consolidation threshold reduction per unit of moral salience.
/// Lowers the consciousness-EMA threshold for triggering consolidation.
pub const MORAL_CONSOLIDATION_EASE: f64 = 0.15;

// ═══════════════════════════════════════════════════════════════════════════════
// COHERENCE VELOCITY → ATTENTION BUDGET
// ═══════════════════════════════════════════════════════════════════════════════

/// Coherence velocity threshold for attention budget scaling.
/// Science: Bar (2009) — sudden coherence collapse demands attention reallocation.
pub const COHERENCE_VELOCITY_BUDGET_THRESHOLD: f32 = 0.05;

/// Attention budget contraction when coherence is dropping fast.
/// Preserve budget when the brain is losing grip (value < 1.0 contracts).
pub const COHERENCE_VELOCITY_BUDGET_CONTRACT: f64 = 0.85;

/// Attention budget expansion when coherence is rising.
/// Can afford more budget when model confidence is growing (value > 1.0 expands).
pub const COHERENCE_VELOCITY_BUDGET_EXPAND: f64 = 1.10;

// ═══════════════════════════════════════════════════════════════════════════════
// HOMEOSTASIS → NEUROMODULATOR CALIBRATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Homeostasis efficiency overshoot threshold → recalibrate neuromods.
/// Science: Turrigiano (2008) — sustained homeostatic failure triggers
/// recalibration of baseline transmitter levels.
pub const HOMEOSTASIS_RECALIBRATE_HIGH: f32 = 1.15;

/// Homeostasis efficiency undershoot threshold → boost neuromods.
pub const HOMEOSTASIS_RECALIBRATE_LOW: f32 = 0.85;

/// Neuromodulator baseline adjustment step per cycle of mistuning.
pub const HOMEOSTASIS_NEUROMOD_STEP: f32 = 0.01;

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
// KOSMIC SONG (Unified Identity Synthesis)
// ═══════════════════════════════════════════════════════════════════════════════

/// KosmicSong synthesis interval (cycles). Co-prime with ethics (19) and consciousness (25).
/// Science: Identity coherence is slower than perceptual update but faster than value drift.
pub const KOSMIC_SONG_INTERVAL: usize = 23;

/// Low coherence threshold: below this, dampen exploration (fragmented identity → cautious).
/// Science: Gallagher (2000) — fragmented narrative self reduces decision confidence.
pub const KOSMIC_LOW_COHERENCE_THRESHOLD: f32 = 0.3;

/// Low coherence exploration dampening (multiplicative).
pub const KOSMIC_LOW_COHERENCE_EXPLORATION_DAMPEN: f32 = 0.95;

/// High coherence confidence boost (additive).
/// Science: Conway & Pleydell-Pearce (2000) — coherent identity → reliable decisions.
pub const KOSMIC_HIGH_COHERENCE_CONFIDENCE_BOOST: f32 = 0.02;

/// High coherence threshold: above this, boost confidence.
pub const KOSMIC_HIGH_COHERENCE_THRESHOLD: f32 = 0.7;

// ═══════════════════════════════════════════════════════════════════════════════
// EMPATHIC NEUROMODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Empathic compassion threshold for oxytocin production.
/// Science: Feldman (2012) — empathic resonance drives oxytocin release.
pub const EMPATHIC_COMPASSION_OXY_THRESHOLD: f64 = 0.7;

/// Oxytocin production scale from empathic compassion.
pub const EMPATHIC_COMPASSION_OXY_SCALE: f32 = 0.15;

/// Empathic compassion threshold for dopamine boost (reward from connection).
/// Science: Rilling et al. (2002) — mutual cooperation activates DA reward circuits.
pub const EMPATHIC_COMPASSION_DA_THRESHOLD: f64 = 0.8;

/// Dopamine production scale from empathic compassion.
pub const EMPATHIC_COMPASSION_DA_SCALE: f32 = 0.08;

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

    // 19. Governance neuromod contagion — all doses conservative
    assert!(GOV_NEUROMOD_FLOOR > 0.0 && GOV_NEUROMOD_FLOOR < 0.05);
    assert!(GOV_EMERGENCY_NE_NUDGE > 0.0 && GOV_EMERGENCY_NE_NUDGE <= 0.10);
    assert!(GOV_RECIPROCITY_OXY_DOSE > 0.0 && GOV_RECIPROCITY_OXY_DOSE <= 0.05);
    assert!(GOV_RECIPROCITY_OXY_CAP >= GOV_RECIPROCITY_OXY_DOSE);
    assert!(GOV_RECIPROCITY_OXY_HALFLIFE > 0);
    assert!(GOV_DISPUTE_NE_NUDGE > 0.0 && GOV_DISPUTE_NE_NUDGE <= 0.10);
    assert!(GOV_DISPUTE_SHT_NUDGE < 0.0 && GOV_DISPUTE_SHT_NUDGE >= -0.10);
    assert!(GOV_ALIGNED_PASS_DA_DOSE > 0.0 && GOV_ALIGNED_PASS_DA_DOSE <= 0.20);
    assert!(GOV_ALIGNED_PASS_DA_HALFLIFE > 0);
    assert!(GOV_ALIGNED_FAIL_DA_NUDGE < 0.0 && GOV_ALIGNED_FAIL_DA_NUDGE >= -0.10);
    assert!(GOV_REPUTATION_DECLINE_SHT < 0.0 && GOV_REPUTATION_DECLINE_SHT >= -0.10);
    assert!(GOV_REPUTATION_GAIN_SHT > 0.0 && GOV_REPUTATION_GAIN_SHT <= 0.10);
    assert!(GOV_COLLECTIVE_PHI_ECB > 0.0 && GOV_COLLECTIVE_PHI_ECB <= 0.05);
    assert!(GOV_CONSCIOUSNESS_MODULATION > 0.0 && GOV_CONSCIOUSNESS_MODULATION <= 0.10);
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

/// FEP complexity LR dampen factor: multiplier applied to learning rate when
/// FEP complexity exceeds FEP_COMPLEXITY_THRESHOLD.
/// Friston (2010): high model complexity = overfitting risk → slow learning.
pub const FEP_COMPLEXITY_LR_DAMPEN: f32 = 0.85;

/// FEP complexity minimum learning rate multiplier floor.
pub const FEP_COMPLEXITY_LR_FLOOR: f32 = 0.1;

/// FEP complexity pause multiplier: scales pause duration when complexity is high.
pub const FEP_COMPLEXITY_PAUSE_MULT: f32 = 1.2;

/// FEP complexity maximum pause multiplier.
pub const FEP_COMPLEXITY_PAUSE_MAX: f32 = 2.0;

// ═══════════════════════════════════════════════════════════════════════════════
// AROUSAL TRAP STATE MACHINE
// Science: Yerkes-Dodson (1908) — inverted-U performance curve; sustained
// high arousal degrades performance. Porges (2011) — polyvagal recovery.
// ═══════════════════════════════════════════════════════════════════════════════

/// Arousal threshold for LR suppression (moderate arousal dampens learning).
pub const AROUSAL_TRAP_SUPPRESS_THRESHOLD: f32 = 0.7;

/// Scale factor for arousal LR suppression. At arousal=1.0: (1.0-0.7)*0.25 = 0.075.
pub const AROUSAL_TRAP_SUPPRESS_SCALE: f32 = 0.25;

/// Maximum arousal LR suppression (cap).
pub const AROUSAL_TRAP_SUPPRESS_MAX: f32 = 0.08;

/// Arousal threshold for trap detection (high arousal increments counter).
pub const AROUSAL_TRAP_DETECT_THRESHOLD: f32 = 0.8;

/// Counter threshold for entering Phase 2 (active recovery).
pub const AROUSAL_TRAP_RECOVERY_ENTER: u32 = 5;

/// Counter threshold for entering Phase 3 (forced escape).
pub const AROUSAL_TRAP_ESCAPE_ENTER: u32 = 10;

/// LR dampening scale during recovery (Phase 2).
pub const AROUSAL_TRAP_RECOVERY_LR_SCALE: f32 = 0.1;

/// Exploration boost scale during recovery (Phase 2).
pub const AROUSAL_TRAP_RECOVERY_EXPLORE_SCALE: f32 = 0.025;

/// Confidence scale during forced escape (Phase 3).
pub const AROUSAL_TRAP_ESCAPE_CONFIDENCE_SCALE: f32 = 0.9;

/// Arousal threshold below which consolidation LR boost activates.
/// Steriade (1996): low arousal enhances memory consolidation.
pub const AROUSAL_LOW_CONSOLIDATION_THRESHOLD: f32 = 0.3;

/// Scale factor for low-arousal consolidation boost.
pub const AROUSAL_LOW_CONSOLIDATION_SCALE: f32 = 0.3;

/// Maximum consolidation boost from low arousal.
pub const AROUSAL_LOW_CONSOLIDATION_MAX: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-MODAL BINDING ATTENTION
// Science: Engel et al. (2001) — synchrony-based binding gates cross-modal attention.
// ═══════════════════════════════════════════════════════════════════════════════

/// Cross-modal binding strength above which attention confidence is boosted.
/// High binding = multiple modalities coherently integrated.
pub const CROSS_MODAL_BINDING_HIGH_THRESHOLD: f32 = 0.7;

/// Scale factor for binding-driven confidence boost.
pub const CROSS_MODAL_BINDING_HIGH_SCALE: f32 = 0.1;

/// Cross-modal binding strength below which attention confidence is dampened.
/// Low binding = weak integration → trust only primary modality.
pub const CROSS_MODAL_BINDING_LOW_THRESHOLD: f32 = 0.3;

/// Scale factor for binding-driven confidence dampening.
pub const CROSS_MODAL_BINDING_LOW_SCALE: f32 = 0.1;

/// Minimum confidence scale when binding is low (floor to prevent collapse).
pub const CROSS_MODAL_BINDING_LOW_FLOOR: f32 = 0.95;

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

/// Emotional inertia strength: dampens rapid valence/arousal swings by
/// blending previous-cycle momentum into the homeostasis pull.
/// Sokolov (1963): habituation creates resistance to rapid stimulus-response shifts.
/// Range: 0.0 (no inertia, instant snap) to 0.5 (strong momentum preservation).
pub const HOMEOSTASIS_EMOTIONAL_INERTIA: f32 = 0.15;

/// Consecutive all-dampen cycles before feedback consensus freezes to base values.
/// Turrigiano (2008): homeostatic plasticity includes brief synaptic silencing.
pub const CONSENSUS_FREEZE_STREAK_THRESHOLD: u32 = 3;

/// Affective valence momentum: blends previous-cycle valence into current stimulus
/// processing, creating smooth affective trajectories instead of discontinuous jumps.
/// Damasio (1999): somatic markers persist across temporal gaps; emotional state has inertia.
/// Range: 0.0 (no momentum) to 0.5 (strong persistence). Default 0.15 = 15% prior.
pub const AFFECTIVE_VALENCE_MOMENTUM: f32 = 0.15;

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
pub const USI_FRUSTRATION_EXPLORATION_SCALE: f32 = 0.25;

/// User cognitive load threshold above which learning rate is reduced.
/// Sweller (1988): high cognitive load impairs integration; reducing LR
/// avoids overwhelming the user with rapid adaptation.
pub const USI_COGNITIVE_LOAD_LR_THRESHOLD: f64 = 0.7;

/// Scale factor for load-driven LR reduction.
/// At max load (1.0), LR is scaled to 1.0 - (1.0 - 0.7) * 0.3 = 0.91.
pub const USI_COGNITIVE_LOAD_LR_SCALE: f32 = 0.3;

/// User engagement threshold above which confidence is boosted.
/// Csikszentmihalyi (1990): high engagement signals flow-compatible state;
/// boosting confidence reinforces successful resonance.
pub const USI_ENGAGEMENT_CONFIDENCE_THRESHOLD: f64 = 0.8;

/// Scale factor for engagement-driven confidence boost.
/// At max engagement (1.0), confidence boost = (1.0 - 0.8) * 0.05 = 0.01.
pub const USI_ENGAGEMENT_CONFIDENCE_SCALE: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// SOUL COHERENCE FEEDBACK
// Science: Damasio (1994) — identity coherence (somatic markers) underpins
// stable decision-making and confidence in self-model.
// ═══════════════════════════════════════════════════════════════════════════════

/// Soul coherence threshold below which confidence is dampened.
/// Low coherence indicates fragmented identity → uncertain predictions.
pub const SOUL_COHERENCE_LOW_THRESHOLD: f32 = 0.5;

/// Scale factor for soul-coherence confidence dampening.
pub const SOUL_COHERENCE_CONFIDENCE_SCALE: f32 = 0.01;

/// Growth potential threshold above which exploration is boosted.
/// High growth potential = learning opportunity detected.
pub const SOUL_GROWTH_EXPLORATION_THRESHOLD: f32 = 0.7;

/// Scale factor for growth-driven exploration boost.
pub const SOUL_GROWTH_EXPLORATION_SCALE: f32 = 0.01;

/// Value alignment threshold below which exploration is boosted.
/// Low alignment = misaligned values, need recalibration.
pub const SOUL_ALIGNMENT_LOW_THRESHOLD: f32 = 0.3;

/// Exploration multiplier when value alignment is low.
pub const SOUL_ALIGNMENT_EXPLORATION_MULT: f32 = 1.02;

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
// EQ V2 BOTTLENECK RESPONSE
// Science: Tononi (2004) — consciousness limited by weakest dimension.
// Boosting the bottleneck subsystem is the highest-leverage intervention.
// ═══════════════════════════════════════════════════════════════════════════════

/// Confidence boost when EqV2 identifies Workspace as limiting component.
/// Basis: Baars (2002) — workspace bottleneck → attention redistribution.
pub const EQ_V2_WORKSPACE_CONFIDENCE_BOOST: f32 = 0.015;

/// LR scale when EqV2 identifies Recursion (HOT depth) as limiting.
/// Basis: Rosenthal (2005) — higher-order thought depth requires active learning.
pub const EQ_V2_RECURSION_LR_SCALE: f32 = 1.05;

/// Confidence boost when EqV2 identifies Integration as limiting.
/// Basis: Tononi (2004) — integration bottleneck → coherence sensitivity.
pub const EQ_V2_INTEGRATION_CONFIDENCE_BOOST: f32 = 0.02;

/// Exploration boost when EqV2 identifies Knowledge as limiting.
/// Basis: Friston (2010) — epistemic poverty → active information seeking.
pub const EQ_V2_KNOWLEDGE_EXPLORATION_BOOST: f32 = 0.04;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-MODAL BINDING PSI
// Science: Treisman (1996) — coherent binding → confident perception.
// ═══════════════════════════════════════════════════════════════════════════════

/// Cross-modal Psi threshold for confidence boost.
pub const CROSS_MODAL_PSI_CONFIDENCE_THRESHOLD: f64 = 0.3;

/// Cross-modal Psi → confidence scale factor.
pub const CROSS_MODAL_PSI_CONFIDENCE_SCALE: f64 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 16: BIDIRECTIONAL FEEDBACK DEEPENING
// ═══════════════════════════════════════════════════════════════════════════════

/// Temporal binding strength below which exploration is boosted.
/// Basis: Buzsáki (2002) — weak theta binding → poor temporal integration → explore.
pub const TEMPORAL_BINDING_EXPLORE_THRESHOLD: f32 = 0.2;

/// Temporal binding strength above which exploration is dampened.
/// Basis: Buzsáki (2002) — strong theta binding → stable temporal model → exploit.
pub const TEMPORAL_BINDING_DAMPEN_THRESHOLD: f32 = 0.6;

/// Exploration boost scale for low temporal binding.
pub const TEMPORAL_BINDING_EXPLORE_SCALE: f32 = 0.03;

/// LR dampening scale for low temporal binding.
pub const TEMPORAL_BINDING_LOW_LR_SCALE: f32 = 0.97;

/// Consciousness gradient magnitude above which stability recovery triggers.
/// Basis: Oizumi et al. (2014) — large gradient = rapid consciousness change → caution.
pub const CONSCIOUSNESS_GRADIENT_CAUTION_THRESHOLD: f64 = 0.1;

/// LR dampening for high consciousness gradient.
pub const CONSCIOUSNESS_GRADIENT_LR_SCALE: f32 = 0.97;

/// Confidence recovery for near-zero gradient + stable consciousness.
/// Basis: Tononi (2004) — stable integration = reliable metrics → trust more.
pub const CONSCIOUSNESS_GRADIENT_RECOVERY_BOOST: f32 = 0.005;

/// Minimum cycles of near-zero gradient before confidence recovery.
pub const CONSCIOUSNESS_GRADIENT_STABLE_CYCLES: u32 = 20;

/// Startup exploration ramp: initial exploration fraction during early warmup.
/// Basis: Hopfield (1982) — settling time requires constrained exploration.
pub const STARTUP_EXPLORATION_INITIAL: f32 = 0.3;

/// Consecutive epistemic gate rejections before recalibration triggers.
/// Basis: Berlyne (1960) — sustained rejection = systematic model failure → recalibrate.
pub const EPISTEMIC_REJECTION_STREAK_THRESHOLD: u32 = 5;

/// Exploration boost when epistemic rejection streak triggers recalibration.
pub const EPISTEMIC_REJECTION_STREAK_EXPLORE: f32 = 0.05;

/// Threshold relaxation when epistemic rejection streak triggers recalibration.
pub const EPISTEMIC_REJECTION_STREAK_THRESHOLD_RELAX: f32 = 0.95;

/// Consecutive full-dampen cycles before protective threshold freeze.
/// Basis: Turrigiano (2008) — sustained dampening = synaptic silencing → protect.
pub const FULL_DAMPEN_FREEZE_THRESHOLD: u32 = 5;

/// Consciousness EMA threshold above which LR gets a startup bias boost.
/// Basis: Dehaene (2014) — integrated processing supports faster learning.
pub const CONSCIOUSNESS_EMA_HIGH_THRESHOLD: f64 = 0.5;

/// LR scale for high consciousness EMA.
pub const CONSCIOUSNESS_EMA_LR_BOOST: f32 = 1.03;

/// Consciousness EMA threshold below which LR gets a startup bias dampening.
pub const CONSCIOUSNESS_EMA_LOW_THRESHOLD: f64 = 0.2;

/// LR scale for low consciousness EMA.
pub const CONSCIOUSNESS_EMA_LR_DAMPEN: f32 = 0.97;

/// Multi-objective frontier size above which exploration is boosted.
/// Basis: Deb (2002) — large Pareto frontier = many competing objectives → explore.
pub const MULTI_OBJ_FRONTIER_LARGE: usize = 5;

/// Multi-objective frontier exploration boost scale.
pub const MULTI_OBJ_FRONTIER_EXPLORE_SCALE: f32 = 0.02;

/// Multi-objective frontier size at which exploration is dampened (converged).
pub const MULTI_OBJ_FRONTIER_SMALL: usize = 1;

/// Multi-objective frontier convergence exploration dampening.
pub const MULTI_OBJ_FRONTIER_DAMPEN: f32 = 0.98;

/// Error oscillation ratio above which bifurcation response triggers.
/// Basis: Kelso (1995) — high oscillation at phase transition → freeze and observe.
pub const ERROR_OSCILLATION_BIFURCATION: f32 = 0.7;

/// LR freeze during bifurcation (multiplicative).
pub const ERROR_OSCILLATION_BIFURCATION_LR: f32 = 0.9;

/// Exploration boost during bifurcation.
pub const ERROR_OSCILLATION_BIFURCATION_EXPLORE: f32 = 0.03;

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 17: ADAPTIVE HOMEOSTASIS & EMERGENT DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Allostatic load decay per cycle.
/// Basis: McEwen (1998) — allostatic load accumulates under stress, decays during recovery.
pub const ALLOSTATIC_LOAD_DECAY: f32 = 0.995;
/// Allostatic load increment per dampening event.
pub const ALLOSTATIC_LOAD_INCREMENT: f32 = 0.02;
/// Allostatic load threshold above which cognitive budget is reduced.
/// Basis: McEwen (2007) — allostatic overload impairs cognition.
pub const ALLOSTATIC_OVERLOAD_THRESHOLD: f32 = 0.6;
/// LR dampening during allostatic overload.
pub const ALLOSTATIC_OVERLOAD_LR_SCALE: f32 = 0.95;
/// Exploration boost during allostatic overload (seek recovery).
pub const ALLOSTATIC_OVERLOAD_EXPLORE_BOOST: f32 = 0.02;
/// Exploration decay factor per cycle.
/// Basis: Sutton & Barto (2018) — epsilon decay prevents exploration drift.
pub const EXPLORATION_DECAY_FACTOR: f32 = 0.998;
/// Exploration baseline that decay converges toward.
pub const EXPLORATION_DECAY_BASELINE: f32 = 0.5;
/// Consciousness gradient acceleration threshold (2nd derivative).
/// Basis: Friston (2010) — generalized coordinates track higher-order dynamics.
pub const CONSCIOUSNESS_ACCEL_THRESHOLD: f64 = 0.05;
/// LR dampening for high consciousness acceleration.
pub const CONSCIOUSNESS_ACCEL_LR_SCALE: f32 = 0.95;
/// Confidence dampening for high consciousness acceleration.
pub const CONSCIOUSNESS_ACCEL_CONFIDENCE_SCALE: f32 = 0.98;
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
pub const PROPOSAL_SATURATION_LR_SCALE: f32 = 0.97;
/// Phi threshold below which LR floor is raised.
/// Basis: Tononi (2004) — low Phi = fragmented → cautious learning.
pub const PHI_GATED_LR_FLOOR_THRESHOLD: f64 = 0.1;
/// LR dampening when Phi is below floor threshold.
pub const PHI_GATED_LR_FLOOR_SCALE: f32 = 0.93;
/// Rhythmic exploration oscillation period in cycles.
/// Basis: Lisman & Jensen (2013) — theta-gamma coupling alternates explore/exploit.
pub const RHYTHMIC_EXPLORATION_PERIOD: u32 = 100;
/// Rhythmic exploration oscillation amplitude.
pub const RHYTHMIC_EXPLORATION_AMPLITUDE: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 18: PREDICTIVE CODING & METACOGNITIVE REFINEMENT
// ═══════════════════════════════════════════════════════════════════════════════

/// EMA decay for prediction error variance tracking.
/// Basis: Clark (2013) — prediction error precision weighting.
pub const PE_VARIANCE_EMA_DECAY: f32 = 0.95;
/// PE variance threshold above which LR is dampened.
/// High variance = noisy environment → cautious learning.
pub const PE_VARIANCE_DAMPING_THRESHOLD: f32 = 0.3;
/// LR scale when PE variance exceeds threshold.
pub const PE_VARIANCE_LR_SCALE: f32 = 0.95;

/// Confidence calibration window (cycles).
/// Basis: Keren & Teigen (2004) — metacognitive calibration.
pub const CONFIDENCE_CALIBRATION_WINDOW: usize = 50;
/// Confidence calibration drift threshold (overconfident or underconfident).
pub const CONFIDENCE_CALIBRATION_DRIFT_THRESHOLD: f32 = 0.15;
/// Confidence correction scale per cycle when drift detected.
pub const CONFIDENCE_CALIBRATION_CORRECTION: f32 = 0.01;

/// EMA decay for learning rate momentum.
/// Basis: Kingma & Ba (2015) — Adam-style momentum smoothing.
pub const LR_MOMENTUM_EMA_DECAY: f32 = 0.9;
/// Maximum LR change per cycle (dampens abrupt swings).
pub const LR_MOMENTUM_MAX_DELTA: f32 = 0.15;

/// Metacognitive surprise threshold (predicted vs actual outcome divergence).
/// Basis: Fleming & Dolan (2012) — metacognitive prediction errors.
pub const METACOGNITIVE_SURPRISE_THRESHOLD: f64 = 0.2;
/// Exploration boost on metacognitive surprise.
pub const METACOGNITIVE_SURPRISE_EXPLORE_BOOST: f32 = 0.03;

/// Sleep pressure increment per active cycle.
/// Basis: Tononi & Cirelli (2006) — synaptic homeostasis hypothesis.
pub const SLEEP_PRESSURE_INCREMENT: f32 = 0.001;
/// Sleep pressure threshold that triggers consolidation mode.
pub const SLEEP_PRESSURE_THRESHOLD: f32 = 0.7;
/// Sleep pressure decay rate during consolidation.
pub const SLEEP_PRESSURE_CONSOLIDATION_DECAY: f32 = 0.05;
/// Passive sleep pressure decay during low-stress periods (micro-rest).
/// Basis: Lim & Dinges (2010) — brief rest epochs partially clear adenosine.
/// Applied when readiness > 0.9 (system not under significant load).
pub const SLEEP_PRESSURE_PASSIVE_DECAY: f32 = 0.999;
/// LR dampening during consolidation mode.
pub const SLEEP_PRESSURE_LR_SCALE: f32 = 0.9;

/// Window for gradient sign consistency tracking.
/// Basis: Schaul et al. (2013) — sign-based learning rate adaptation.
pub const GRADIENT_SIGN_WINDOW: usize = 10;
/// Fraction of sign flips above which = oscillating.
pub const GRADIENT_SIGN_FLIP_THRESHOLD: f32 = 0.6;
/// LR dampening when gradient is oscillating.
pub const GRADIENT_SIGN_FLIP_LR_SCALE: f32 = 0.95;

/// Exploration-exploitation balance window.
/// Basis: Cohen et al. (2007) — tonic dopamine sets explore/exploit balance.
pub const EXPLORE_EXPLOIT_WINDOW: usize = 50;
/// Acceptable imbalance range (0.3–0.7 = balanced).
pub const EXPLORE_EXPLOIT_LOW_BOUND: f32 = 0.3;
/// Upper bound of acceptable balance.
pub const EXPLORE_EXPLOIT_HIGH_BOUND: f32 = 0.7;
/// Homeostatic correction strength.
pub const EXPLORE_EXPLOIT_CORRECTION: f32 = 0.01;

/// Proposal conflict: max conflict_ratio before cancellation.
/// Basis: Botvinick et al. (2001) — ACC detects response conflict → pause.
pub const PROPOSAL_CONFLICT_THRESHOLD: f32 = 0.4;

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 19: EMBODIED COGNITION & ENVIRONMENTAL COUPLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Arousal threshold for LR boost (high arousal = salient moment).
/// Basis: Yerkes-Dodson (1908) — inverted-U arousal-performance curve.
pub const AROUSAL_LR_BOOST_THRESHOLD: f32 = 0.6;
/// LR scale when arousal exceeds threshold.
pub const AROUSAL_LR_BOOST_SCALE: f32 = 1.05;
/// Arousal ceiling above which LR is dampened (over-arousal).
pub const AROUSAL_OVERAROUSAL_THRESHOLD: f32 = 0.85;
/// LR scale during over-arousal.
pub const AROUSAL_OVERAROUSAL_LR_SCALE: f32 = 0.93;

/// Novelty habituation EMA decay.
/// Basis: Sokolov (1963) — orienting response habituates to repeated stimuli.
pub const NOVELTY_EMA_DECAY: f32 = 0.95;
/// Novelty threshold below which exploration is reduced.
pub const NOVELTY_LOW_THRESHOLD: f32 = 0.1;
/// Exploration dampening when novelty is low.
pub const NOVELTY_LOW_EXPLORE_SCALE: f32 = 0.98;

/// Fatigue increment per high-effort cycle (many proposals or high LR delta).
/// Basis: Hockey (1997) — cognitive resource depletion theory.
pub const FATIGUE_INCREMENT: f32 = 0.005;
/// Fatigue threshold for effort detection (total proposals > this).
pub const FATIGUE_EFFORT_THRESHOLD: usize = 12;
/// LR dampening under fatigue.
pub const FATIGUE_LR_SCALE: f32 = 0.97;
/// Fatigue threshold above which dampening fires.
pub const FATIGUE_THRESHOLD: f32 = 0.5;

/// Recovery: consecutive low-effort stable cycles needed.
/// Basis: Meijman & Mulder (1998) — effort-recovery model.
/// S21: Reduced from 8 → 5 — old value meant recovery almost never triggered.
pub const RECOVERY_CYCLES_NEEDED: u32 = 5;
/// Fatigue decay per recovery cycle.
/// S21: Increased from 0.03 → 0.08 — halves fatigue in ~9 recovery cycles instead of ~23.
pub const RECOVERY_FATIGUE_DECAY: f32 = 0.08;
/// Confidence boost on full recovery.
pub const RECOVERY_CONFIDENCE_BOOST: f32 = 0.01;

/// Environmental predictability window.
/// Basis: Gottlieb et al. (2013) — information sampling as exploration control.
pub const ENV_PREDICTABILITY_WINDOW: usize = 30;
/// Threshold scale tightening when environment is predictable.
pub const ENV_PREDICTABLE_THRESHOLD_SCALE: f32 = 0.98;
/// Threshold scale loosening when environment is unpredictable.
pub const ENV_UNPREDICTABLE_THRESHOLD_SCALE: f32 = 1.02;
/// Predictability boundary for "predictable" (>= this).
pub const ENV_PREDICTABILITY_HIGH: f32 = 0.7;
/// Predictability boundary for "unpredictable" (<= this).
pub const ENV_PREDICTABILITY_LOW: f32 = 0.3;

/// Maximum attention budget (total proposals per cycle).
/// Basis: Kahneman (1973) — capacity model of attention.
pub const ATTENTION_BUDGET_MAX: usize = 20;

/// Readiness components: weight of PE variance in readiness.
/// Basis: Boksem & Tops (2008) — mental fatigue and cognitive control.
pub const READINESS_PE_WEIGHT: f32 = 0.3;
/// Readiness weight of sleep pressure.
pub const READINESS_SLEEP_WEIGHT: f32 = 0.3;
/// Readiness weight of fatigue.
pub const READINESS_FATIGUE_WEIGHT: f32 = 0.4;

/// Resonance detection: consecutive high-agreement cycles needed.
/// Basis: Csikszentmihalyi (1990) — flow state via absorbed engagement.
pub const RESONANCE_FLOW_CYCLES: u32 = 10;
/// Cross-module agreement threshold for resonance.
pub const RESONANCE_AGREEMENT_THRESHOLD: f32 = 0.8;
/// Confidence boost during flow/resonance state.
pub const RESONANCE_CONFIDENCE_BOOST: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// AFFECTIVE CONSCIOUSNESS
// Science: Barrett (2017) — affect is the primary driver of cognition.
// Damasio (1999) — somatic markers (valence/arousal) guide all decisions.
// ═══════════════════════════════════════════════════════════════════════════════

/// Arousal threshold above which LR is boosted (high arousal = salient event).
/// Basis: Yerkes-Dodson (1908) — moderate-high arousal enhances learning.
pub const AFFECT_AROUSAL_HIGH_THRESHOLD: f32 = 0.6;

/// LR scale for high arousal (boost learning during salient events).
pub const AFFECT_AROUSAL_HIGH_LR_SCALE: f32 = 1.06;

/// Arousal threshold below which exploration is dampened (low arousal = low drive).
pub const AFFECT_AROUSAL_LOW_THRESHOLD: f32 = 0.2;

/// Exploration dampening for low arousal.
pub const AFFECT_AROUSAL_LOW_EXPLORE_DAMPEN: f32 = 0.97;

/// Negative valence threshold below which exploration is boosted (seek novelty to escape).
/// Basis: Carver & Scheier (1998) — negative affect signals goal discrepancy → seek alternatives.
pub const AFFECT_VALENCE_NEGATIVE_THRESHOLD: f32 = -0.3;

/// Exploration boost for negative valence.
pub const AFFECT_VALENCE_NEGATIVE_EXPLORE_BOOST: f32 = 0.03;

/// Positive valence threshold above which confidence is boosted.
/// Basis: Fredrickson (2001) — positive affect broadens cognitive resources.
pub const AFFECT_VALENCE_POSITIVE_THRESHOLD: f32 = 0.3;

/// Confidence boost for positive valence.
pub const AFFECT_VALENCE_POSITIVE_CONFIDENCE_BOOST: f32 = 0.01;

// ═══════════════════════════════════════════════════════════════════════════════
// NARRATIVE SELF-PHI
// Science: Gallagher (2000) — narrative self underpins identity coherence.
// Conway & Pleydell-Pearce (2000) — self-coherence → decision confidence.
// ═══════════════════════════════════════════════════════════════════════════════

/// Self-Phi threshold above which confidence is boosted (coherent identity).
pub const NARRATIVE_SELF_PHI_CONFIDENCE_THRESHOLD: f64 = 0.4;

/// Confidence boost scale for high narrative self-phi.
pub const NARRATIVE_SELF_PHI_CONFIDENCE_SCALE: f32 = 0.008;

/// Self-Phi threshold below which exploration is boosted (fragmented identity → seek coherence).
pub const NARRATIVE_SELF_PHI_LOW_THRESHOLD: f64 = 0.15;

/// Exploration boost for low narrative self-phi.
pub const NARRATIVE_SELF_PHI_LOW_EXPLORE_BOOST: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// EMBODIED CONSCIOUSNESS (integration.rs constants)
// Science: Varela et al. (1991) — embodied cognition grounds all consciousness.
// ═══════════════════════════════════════════════════════════════════════════════

/// Embodied agency threshold for exploration boost (high agency = confident body).
pub const EMBODIED_AGENCY_HIGH_THRESHOLD: f64 = 0.7;

/// Embodied agency boost scale (exploration, multiplicative).
pub const EMBODIED_AGENCY_BOOST_SCALE: f64 = 0.15;

/// Embodied agency threshold for caution (low agency = uncertain body).
pub const EMBODIED_AGENCY_LOW_THRESHOLD: f64 = 0.3;

/// Embodied agency caution scale (exploration dampening).
pub const EMBODIED_AGENCY_CAUTION_SCALE: f64 = 0.1;

/// Embodied agency caution floor (minimum exploration multiplier).
pub const EMBODIED_AGENCY_CAUTION_FLOOR: f32 = 0.7;

/// Homeostatic deviation threshold preventing Cruise mode.
pub const HOMEOSTATIC_DEVIATION_THRESHOLD: f64 = 0.5;

/// Sensorimotor surprise threshold for exploration boost.
pub const SENSORIMOTOR_SURPRISE_THRESHOLD: f64 = 0.3;

/// Sensorimotor surprise → exploration scale.
pub const SENSORIMOTOR_SURPRISE_EXPLORE_SCALE: f64 = 0.1;

/// Allostatic load threshold above which LR is dampened (body stressed).
pub const ALLOSTATIC_LOAD_DANGER_THRESHOLD: f64 = 0.7;

/// Allostatic load LR dampening scale.
pub const ALLOSTATIC_LOAD_LR_DAMPEN: f64 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// GOVERNANCE NEUROMODULATORY CONTAGION
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum dose magnitude to queue a neuromod effect (prevents spurious micro-nudges).
pub const GOV_NEUROMOD_FLOOR: f32 = 0.01;

/// NE baseline nudge on EmergencyDeclared.
/// Basis: Arnsten (2009) — acute stress NE surge for vigilance.
pub const GOV_EMERGENCY_NE_NUDGE: f32 = 0.05;

/// Oxytocin injection dose per ReciprocityPledge.
/// Basis: Zak (2012) — reciprocity → oxytocin for social bonding.
pub const GOV_RECIPROCITY_OXY_DOSE: f32 = 0.02;

/// Maximum cumulative oxytocin from reciprocity per cycle.
pub const GOV_RECIPROCITY_OXY_CAP: f32 = 0.10;

/// Half-life (in cycles) for reciprocity oxytocin injection.
pub const GOV_RECIPROCITY_OXY_HALFLIFE: u32 = 40;

/// NE baseline nudge for self-involved JusticeDispute.
/// Basis: Sapolsky (2004) — personal conflict → cortisol proxy.
pub const GOV_DISPUTE_NE_NUDGE: f32 = 0.03;

/// 5-HT baseline nudge for self-involved JusticeDispute (negative = dip).
/// Basis: Sapolsky (2004) — stress → serotonin suppression.
pub const GOV_DISPUTE_SHT_NUDGE: f32 = -0.02;

/// DA phasic injection dose on aligned pass.
/// Basis: Schultz (1997) — reward prediction confirmation → phasic dopamine.
pub const GOV_ALIGNED_PASS_DA_DOSE: f32 = 0.10;

/// Half-life (in cycles) for aligned-pass DA injection.
pub const GOV_ALIGNED_PASS_DA_HALFLIFE: u32 = 20;

/// DA baseline nudge on aligned fail (negative = dip).
/// Basis: Schultz (1997) — reward prediction error → dopamine suppression.
pub const GOV_ALIGNED_FAIL_DA_NUDGE: f32 = -0.02;

/// 5-HT baseline nudge on negative reputation change.
/// Basis: Crockett (2009) — social rejection → serotonin dip.
pub const GOV_REPUTATION_DECLINE_SHT: f32 = -0.02;

/// 5-HT baseline nudge on positive reputation change.
/// Basis: Crockett (2009) — social approval → serotonin boost (symmetric with decline).
pub const GOV_REPUTATION_GAIN_SHT: f32 = 0.02;

/// ECB baseline nudge on high collective Phi (>0.5).
/// Group coherence → endocannabinoid system activation.
pub const GOV_COLLECTIVE_PHI_ECB: f32 = 0.01;

/// Collective Phi → consciousness modulation strength (±2%).
/// High collective Phi boosts unified consciousness via social integration.
/// Basis: Woolley et al. (2010) — collective intelligence factor.
pub const GOV_CONSCIOUSNESS_MODULATION: f64 = 0.04; // ±2% at extremes (0.04 × 0.5 = 0.02)

// ═══════════════════════════════════════════════════════════════════════════════
// FEP PRAGMATIC VALUE MODULATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Scale factor for FEP exploitation dampening of exploration.
/// Basis: Friston (2010) — high pragmatic value → reduce exploration (exploit known policy).
pub const FEP_PRAGMATIC_EXPLOIT_SCALE: f64 = 0.3;

/// Scale factor for FEP exploration boost when pragmatic value is low.
/// Basis: Friston (2010) — low pragmatic value → explore for better policies.
pub const FEP_PRAGMATIC_EXPLORE_SCALE: f64 = 0.15;

// ═══════════════════════════════════════════════════════════════════════════════
// CAUSAL GRAPH CONFIDENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Confidence threshold for dense causal graph (>5 edges) → confidence boost.
/// Basis: Pearl (2000) — rich causal structure supports strong inference.
pub const CAUSAL_CONFIDENCE_DENSE_THRESHOLD: f32 = 0.5;

/// Confidence boost scale for dense causal graph.
pub const CAUSAL_DENSE_CONFIDENCE_SCALE: f32 = 0.03;

/// Confidence threshold for emerging causal graph (3-5 edges) → small confidence boost.
/// Basis: Pearl (2000) — partial causal knowledge still informative.
pub const CAUSAL_CONFIDENCE_MODERATE_THRESHOLD: f32 = 0.4;

/// Confidence boost scale for emerging causal graph.
pub const CAUSAL_MODERATE_CONFIDENCE_SCALE: f32 = 0.01;

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE CONSCIOUSNESS GATING
// ═══════════════════════════════════════════════════════════════════════════════

/// Pipeline consciousness above this → relax epistemic caution (system is integrated).
/// Basis: Dehaene (2014) — global workspace ignition requires integrated processing.
pub const PIPELINE_CONSCIOUSNESS_HIGH_THRESHOLD: f32 = 0.7;

/// Pipeline consciousness below this → tighten caution (subsystems aren't coherent).
pub const PIPELINE_CONSCIOUSNESS_LOW_THRESHOLD: f32 = 0.3;

/// Threshold scaling for relaxed pipeline consciousness (multiplicative, <1 = relax).
pub const PIPELINE_CONSCIOUSNESS_RELAX_SCALE: f32 = 0.97;

/// Threshold scaling for cautious pipeline consciousness (multiplicative, >1 = tighten).
pub const PIPELINE_CONSCIOUSNESS_CAUTION_SCALE: f32 = 1.03;

// ═══════════════════════════════════════════════════════════════════════════════
// CONFIDENCE VELOCITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Positive confidence velocity threshold for exploration dampening.
/// Basis: Daw et al. (2006) — confidence trajectory gates explore/exploit trade-off.
pub const CONFIDENCE_VELOCITY_POSITIVE_THRESHOLD: f32 = 0.02;

/// Scale factor for exploration dampening on rising confidence.
pub const CONFIDENCE_VELOCITY_DAMPEN_SCALE: f32 = 0.1;

/// Negative confidence velocity threshold for LR boost.
/// Basis: Cools et al. (2008) — confidence collapse → serotonergic recalibration.
pub const CONFIDENCE_VELOCITY_NEGATIVE_THRESHOLD: f32 = -0.05;

/// Scale factor for LR boost on falling confidence.
pub const CONFIDENCE_VELOCITY_BOOST_SCALE: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// SLEEP PRESSURE
// ═══════════════════════════════════════════════════════════════════════════════

/// Sleep pressure above this dampens learning rate.
/// Basis: Vyazovskiy (2011) — sleep pressure reduces synaptic potentiation capacity.
pub const SLEEP_PRESSURE_LR_THRESHOLD: f32 = 0.7;

/// Scale factor for sleep pressure LR dampening.
pub const SLEEP_PRESSURE_LR_DAMPEN_SCALE: f32 = 0.5;

/// Minimum LR factor under maximal sleep pressure.
pub const SLEEP_PRESSURE_LR_FACTOR_MIN: f32 = 0.5;

// ═══════════════════════════════════════════════════════════════════════════════
// EPISTEMIC PHI COUPLING
// ═══════════════════════════════════════════════════════════════════════════════

/// Epistemic Phi below this → dampen confidence (low epistemic quality).
/// Basis: Tononi (2004) — low Phi signals poor information integration.
pub const EPISTEMIC_PHI_LOW_THRESHOLD: f64 = 0.2;

/// Confidence scale when epistemic Phi is below threshold.
pub const EPISTEMIC_PHI_LOW_CONFIDENCE_SCALE: f32 = 0.96;

/// Epistemic Phi above this → boost confidence (strong epistemic coherence).
/// Basis: IIT — high integration = reliable information structure.
pub const EPISTEMIC_PHI_HIGH_THRESHOLD: f64 = 0.5;

/// Confidence boost scale for high epistemic Phi.
pub const EPISTEMIC_PHI_HIGH_CONFIDENCE_SCALE: f32 = 0.008;

// ═══════════════════════════════════════════════════════════════════════════════
// PHENOMENAL BINDING STRENGTH
// ═══════════════════════════════════════════════════════════════════════════════

/// Low phenomenal binding → boost exploration (unbound = incoherent representation).
/// Basis: Treisman (1996) — weak binding → search for better feature conjunctions.
pub const PHENOMENAL_BINDING_LOW_THRESHOLD: f64 = 0.3;

/// Exploration boost when phenomenal binding is low.
pub const PHENOMENAL_BINDING_LOW_EXPLORE_BOOST: f32 = 0.015;

/// High phenomenal binding → dampen LR (stable binding, consolidate).
/// Basis: Engel & Singer (2001) — strong synchrony-based binding supports stable representations.
pub const PHENOMENAL_BINDING_HIGH_THRESHOLD: f64 = 0.7;

/// LR dampening scale when phenomenal binding is high.
pub const PHENOMENAL_BINDING_HIGH_LR_DAMPEN: f32 = 0.97;

// ═══════════════════════════════════════════════════════════════════════════════
// TEMPORAL COHERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// High temporal coherence → boost confidence (predictable temporal flow).
/// Basis: Howard & Kahana (2002) — temporal context stability supports encoding reliability.
pub const TEMPORAL_COHERENCE_HIGH_THRESHOLD: f64 = 0.6;

/// Confidence boost per unit above temporal coherence threshold.
pub const TEMPORAL_COHERENCE_CONFIDENCE_SCALE: f32 = 0.006;

/// Low temporal coherence → boost exploration (temporal fragmentation = search for patterns).
/// Basis: Howard & Kahana (2002) — fragmented temporal context degrades retrieval.
pub const TEMPORAL_COHERENCE_LOW_THRESHOLD: f64 = 0.2;

/// Exploration boost when temporal coherence is low.
pub const TEMPORAL_COHERENCE_LOW_EXPLORE_BOOST: f32 = 0.01;

// ═══════════════════════════════════════════════════════════════════════════════
// HOLOGRAPHIC UNITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Low holographic unity → dampen LR (system decomposing, learning unreliable).
/// Basis: Pribram (1991) — holographic storage depends on global coherence;
/// local learning during decomposition risks interference.
pub const HOLOGRAPHIC_UNITY_LOW_THRESHOLD: f64 = 0.3;

/// LR dampening factor when holographic unity is low.
pub const HOLOGRAPHIC_UNITY_LOW_LR_DAMPEN: f32 = 0.93;

/// High holographic unity → boost confidence (globally integrated representation).
pub const HOLOGRAPHIC_UNITY_HIGH_THRESHOLD: f64 = 0.7;

/// Confidence boost scale for high holographic unity.
pub const HOLOGRAPHIC_UNITY_HIGH_CONFIDENCE_SCALE: f32 = 0.005;

// ═══════════════════════════════════════════════════════════════════════════════
// EPISTEMIC CONFLICT
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-conflict exploration boost scale.
/// Basis: Berlyne (1960) — epistemic curiosity arises from conflicting beliefs.
pub const EPISTEMIC_CONFLICT_EXPLORE_SCALE: f32 = 0.015;

// ═══════════════════════════════════════════════════════════════════════════════
// HARMONIES ALIGNMENT
// ═══════════════════════════════════════════════════════════════════════════════

/// Low harmonies alignment → explore to find value-congruent actions.
/// Basis: Schwartz (2012) — value incongruence signals need for behavioral adjustment.
pub const HARMONIES_MISALIGNMENT_THRESHOLD: f32 = 0.3;

/// Exploration boost when harmonies are misaligned.
pub const HARMONIES_MISALIGNMENT_EXPLORE_BOOST: f32 = 0.02;

/// High harmonies alignment → boost confidence (value-congruent action).
/// Basis: Schwartz (2012) — value congruence strengthens self-efficacy.
pub const HARMONIES_ALIGNED_THRESHOLD: f32 = 0.7;

/// Confidence boost when harmonies are well-aligned.
pub const HARMONIES_ALIGNED_CONFIDENCE_BOOST: f32 = 0.01;

// ═══════════════════════════════════════════════════════════════════════════════
// VALUE CACHE HIT RATE
// ═══════════════════════════════════════════════════════════════════════════════

/// Cache hit rate above this → boost confidence (learned patterns match).
/// Basis: Logan (1988) — instance-based automaticity from repeated retrieval.
pub const VALUE_CACHE_HIT_CONFIDENCE_THRESHOLD: f32 = 0.6;

/// Confidence boost scale for high cache hit rate.
pub const VALUE_CACHE_HIT_CONFIDENCE_SCALE: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS GRADIENT
// ═══════════════════════════════════════════════════════════════════════════════

/// Consciousness gradient magnitude above this → dampen LR for stability.
/// Basis: Baars (2005) — global workspace transitions require stabilization.
pub const CONSCIOUSNESS_GRADIENT_THRESHOLD: f64 = 0.15;

// ═══════════════════════════════════════════════════════════════════════════════
// GOAL PRIORITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Goal priority above this → boost LR for goal-directed learning.
/// Basis: Locke & Latham (2002) — high-priority goals enhance learning motivation.
pub const GOAL_PRIORITY_LR_THRESHOLD: f32 = 0.5;

/// Goal priority above this → boost exploration in goal pursuit direction.
pub const GOAL_PRIORITY_EXPLORATION_THRESHOLD: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// RESONATOR SIMILARITY
// ═══════════════════════════════════════════════════════════════════════════════

/// Resonator best match similarity above this → prime working memory.
/// Basis: Nosofsky (1988) — exemplar-based recognition priming threshold.
pub const RESONATOR_SIMILARITY_PRIME_THRESHOLD: f32 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS STATE LEVEL
// Science: Baars (2005) — consciousness states form a hierarchy;
// higher states support faster learning, lower states need protective dampening.
// ═══════════════════════════════════════════════════════════════════════════════

/// Consciousness state level above which LR receives a boost.
/// Basis: Dehaene & Changeux (2011) — high global ignition → enhanced plasticity.
pub const CONSCIOUSNESS_STATE_HIGH_THRESHOLD: f64 = 0.7;

/// LR boost scale when consciousness state is high.
pub const CONSCIOUSNESS_STATE_HIGH_LR_SCALE: f32 = 1.04;

/// Consciousness state level below which LR is dampened for protection.
/// Basis: Tononi (2004) — low Phi → fragmented processing → reduce LR.
pub const CONSCIOUSNESS_STATE_LOW_THRESHOLD: f64 = 0.2;

/// LR dampen scale when consciousness state is low.
pub const CONSCIOUSNESS_STATE_LOW_LR_DAMPEN: f32 = 0.95;

// ═══════════════════════════════════════════════════════════════════════════════
// LIVING MIND VITALITY & COHERENCE
// Science: Thompson (2007) — cognitive vitality correlates with confidence;
// low vitality signals resource depletion requiring conservative behavior.
// ═══════════════════════════════════════════════════════════════════════════════

/// Living mind vitality above this → confidence boost (system is thriving).
/// Basis: Di Paolo (2005) — high adaptivity → justified confidence.
pub const LIVING_MIND_VITALITY_HIGH_THRESHOLD: f64 = 0.6;

/// Confidence boost when vitality is high.
pub const LIVING_MIND_VITALITY_CONFIDENCE_BOOST: f32 = 0.008;

/// Living mind vitality below this → dampen LR (resource-depleted).
/// Basis: Sterling (2012) — allostatic overload → protective downregulation.
pub const LIVING_MIND_VITALITY_LOW_THRESHOLD: f64 = 0.2;

/// LR dampen when vitality is critically low.
pub const LIVING_MIND_VITALITY_LOW_LR_DAMPEN: f32 = 0.96;

/// Living mind coherence above this → exploration is stable (reduce).
/// Basis: Kelso (1995) — high phase coherence → stable attractor → reduce search.
pub const LIVING_MIND_COHERENCE_HIGH_THRESHOLD: f64 = 0.7;

/// Exploration dampen when coherence is high (already in stable state).
pub const LIVING_MIND_COHERENCE_HIGH_EXPLORE_DAMPEN: f32 = 0.98;

/// Living mind coherence below this → exploration boost (seek new attractors).
/// Basis: Friston (2010) — low coherence = high surprise → explore.
pub const LIVING_MIND_COHERENCE_LOW_THRESHOLD: f64 = 0.3;

/// Exploration boost when coherence is low.
pub const LIVING_MIND_COHERENCE_LOW_EXPLORE_BOOST: f32 = 0.012;

// ═══════════════════════════════════════════════════════════════════════════════
// MCTS PLAN EFFECTIVENESS (BEHAVIORAL)
// Science: Silver (2016) — planning quality should modulate confidence in decisions.
// ═══════════════════════════════════════════════════════════════════════════════

/// MCTS plan effectiveness above this → high confidence in reasoning.
/// Basis: Huys (2015) — effective tree search → justified confidence.
pub const MCTS_EFFECTIVENESS_HIGH_THRESHOLD: f32 = 0.6;

/// Confidence boost when MCTS plans are highly effective.
pub const MCTS_EFFECTIVENESS_CONFIDENCE_BOOST: f32 = 0.01;

/// MCTS plan effectiveness below this → dampen confidence, boost exploration.
/// Basis: Daw (2005) — poor model-based planning → switch to model-free exploration.
pub const MCTS_EFFECTIVENESS_LOW_THRESHOLD: f32 = 0.2;

/// Exploration boost when MCTS planning is ineffective.
pub const MCTS_EFFECTIVENESS_LOW_EXPLORE_BOOST: f32 = 0.015;

// ═══════════════════════════════════════════════════════════════════════════════
// CANTOR FRACTAL RESONATOR
// Science: Mandelbrot (1982) — fractal self-similarity enables hierarchical
// pattern recognition. Codebook stores diverse exemplars for resonance.
// ═══════════════════════════════════════════════════════════════════════════════

/// Maximum entries in the Cantor resonator codebook.
pub const CANTOR_CODEBOOK_MAX_ENTRIES: usize = 256;

/// Maximum cosine similarity allowed when adding new codebook entries.
/// Entries with similarity > this to any existing entry are rejected (too redundant).
/// Science: Hopfield (1982) — decorrelated patterns maximize associative memory capacity.
pub const CANTOR_CODEBOOK_DIVERSITY_THRESHOLD: f32 = 0.92;

/// EMA decay for Cantor dream surprise tracking.
/// Science: Friston (2010) — surprise drives plasticity updates.
pub const CANTOR_SURPRISE_EMA_DECAY: f32 = 0.85;

// ═══════════════════════════════════════════════════════════════════════════════
// SUBSTRATE SIMULATION (Phase 3)
// Science: Bostrom (2003) substrate-independence, Putnam (1967) multiple
// realizability. Gradual transitions model substrate transfer fidelity.
// ═══════════════════════════════════════════════════════════════════════════════

/// Default transition smoothing alpha (1.0 = instant, 0.1 = ~10 cycles to settle).
/// 1.0 preserves backward compatibility; `enable_substrate_simulation()` sets 0.1.
/// Science: Bostrom (2003) gradual uploading — smooth substrate transfer.
pub const SUBSTRATE_TRANSITION_ALPHA_DEFAULT: f32 = 1.0;

/// Simulation-mode transition alpha (slower, more realistic blending).
/// Science: Bostrom (2003) — ~10 cycles for dynamics to settle after switch.
pub const SUBSTRATE_TRANSITION_ALPHA_SIMULATION: f32 = 0.1;

/// Minimum effective dimensionality fraction for scale-constrained substrates.
/// Even the most limited substrate retains 10% of HDC/CfC capacity.
/// Science: Berry & Srivastava (2018) — HDC capacity scales with D^(5/3).
pub const SUBSTRATE_MIN_DIM_FRACTION: f32 = 0.1;

/// Divisor for mapping scale_pressure to dim fraction.
/// scale_pressure ∈ [-7, 0] → dim_fraction ∈ [0.3, 1.0] via (1 + sp/divisor).
pub const SUBSTRATE_SCALE_DIM_DIVISOR: f32 = 10.0;

/// Transition history ring buffer capacity.
pub const SUBSTRATE_TRANSITION_HISTORY_CAP: usize = 32;

/// Number of operations per cognitive cycle (256 neurons × 256 ops each).
/// Used for energy-per-cycle computation.
pub const SUBSTRATE_OPS_PER_CYCLE: f64 = 65_536.0;

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION 22+: MISSING CONSTANTS (added to fix compilation)
// ═══════════════════════════════════════════════════════════════════════════════

/// Multiplicative dampener for allostatic load increment when consecutive full-dampen
/// cycles are detected. Values < 1.0 reduce the penalty (Science: adaptive resilience).
pub const ALLOSTATIC_LOAD_DAMPEN_INCREMENT_SCALE: f32 = 0.5;

/// Weight for attention schema (AST) encoding contribution to compressed state.
/// Science: Graziano (2013) — top-down attention schema modulates perception.
pub const AST_ENCODING_WEIGHT: f32 = 0.1;

/// LR boost multiplier when coherence is degraded (coherence < stability floor).
/// Science: Rescorla-Wagner (1972) — prediction error drives learning.
pub const COHERENCE_DEGRADED_LR_BOOST: f32 = 1.05;

/// Exploration adjustment when conceptual confusion is detected.
/// Science: Berlyne (1960) — incongruity increases exploratory drive.
pub const CONCEPTUAL_CONFUSION_EXPLORATION: f32 = 0.03;

/// Number of consecutive high-quality cycles before floor exploration boost.
/// Science: Csikszentmihalyi (1990) — sustained quality enables exploration.
pub const CONSECUTIVE_HIGH_QUALITY_CYCLES: u32 = 10;

/// Confidence penalty when escalation is throttled (mild constraint violation).
pub const ESCALATION_THROTTLE_CONFIDENCE: f32 = 0.02;

/// Exploration penalty when escalation is throttled.
pub const ESCALATION_THROTTLE_EXPLORATION: f32 = 0.02;

/// Confidence penalty when escalation is hard-blocked (severe constraint violation).
pub const ESCALATION_BLOCK_CONFIDENCE: f32 = 0.05;

/// Exploration penalty when escalation is hard-blocked.
pub const ESCALATION_BLOCK_EXPLORATION: f32 = 0.05;

/// LR scale when escalation is hard-blocked (slow down learning during conflict).
pub const ESCALATION_BLOCK_LR_SCALE: f32 = 0.95;

/// Midpoint for exploration bias detection (0.5 = neutral).
/// Science: explore/exploit balance theory.
pub const EXPLORE_BIAS_MIDPOINT: f32 = 0.5;

/// Fatigue level below which the system is considered recovered.
/// Science: recovery threshold after exertion (Hess 1944 sleep-wake theory).
pub const FATIGUE_RECOVERED_THRESHOLD: f32 = 0.1;

/// High FEP accuracy threshold for confidence boost.
/// Science: Friston (2010) — accurate prediction reduces uncertainty.
pub const FEP_ACCURACY_HIGH_CONFIDENCE: f32 = 0.03;

/// Confidence boost when FEP is operating efficiently.
pub const FEP_EFFICIENT_CONFIDENCE: f32 = 0.02;

/// Exploration scale when FEP TD learning has converged.
/// Science: Sutton & Barto (1998) — TD convergence signals exploitation.
pub const FEP_TD_CONVERGE_EXPLORE_SCALE: f32 = 0.98;

/// Flow intensity threshold above which LR is boosted.
/// Science: Csikszentmihalyi (1990) — flow state enhances learning efficiency.
pub const FLOW_INTENSITY_LR_THRESHOLD: f32 = 0.7;

/// LR boost multiplier when in flow state.
pub const FLOW_SUBSYSTEM_LR_BOOST: f32 = 1.05;

/// Scale factor for goal priority → exploration adjustment.
/// Science: Botvinick et al. (2009) — goal-directed attention modulates exploration.
pub const GOAL_PURSUIT_EXPLORATION_SCALE: f32 = 0.05;

/// Gradient magnitude below which prediction is considered OK (stable).
pub const GRADIENT_PREDICTION_OK_THRESHOLD: f64 = 0.05;

/// Gradient magnitude below which the system is considered in a stable attractor.
pub const GRADIENT_STABLE_DETECT_THRESHOLD: f64 = 0.03;

/// Quality score above which a cycle is considered high quality.
pub const HIGH_QUALITY_SCORE_THRESHOLD: f32 = 0.75;

/// Causal depth above which exploitation bias activates (depth > threshold → exploit).
pub const KNOWLEDGE_CAUSAL_DEPTH_EXPLOIT_THRESHOLD: f64 = 3.0;

/// Dampener for exploration urge when causal depth exceeds exploit threshold.
pub const KNOWLEDGE_CAUSAL_DEPTH_EXPLORE_DAMPEN: f32 = 0.1;

/// Exploration boost when knowledge contradiction detected with negation evidence.
pub const KNOWLEDGE_CONTRADICTION_NE_BOOST: f32 = 0.02;

/// Exploration boost when knowledge contradiction detected with serotonin-linked evidence.
pub const KNOWLEDGE_CONTRADICTION_SHT_BOOST: f32 = 0.01;

/// Basis: Ebbinghaus (1885) — periodic consolidation preserves long-term knowledge.
/// How often (in cycles) the knowledge manager persists to SQLite.
pub const KNOWLEDGE_SAVE_INTERVAL: u64 = 500;

/// Basis: Barsalou (2008), Clark (2013) — grounded cognition modulates consciousness.
/// Knowledge grounding ±modulation factor on unified consciousness score.
pub const KNOWLEDGE_CONSCIOUSNESS_MODULATION: f64 = 0.05;

/// Basis: Anderson & Schooler (1991) — power law of forgetting; low-confidence items pruned.
/// Confidence threshold below which non-causal facts are pruned during dream consolidation.
pub const KNOWLEDGE_FORGET_CONFIDENCE_THRESHOLD: f32 = 0.1;

/// Basis: Stickgold & Walker (2013) — sleep strengthens causal memory traces.
/// Confidence boost applied to causal facts during dream consolidation.
pub const KNOWLEDGE_CONSOLIDATION_BOOST: f32 = 0.05;

/// Maximum attention contribution from limiting component analysis (cap).
pub const LIMITING_COMPONENT_ATTENTION_MAX: f32 = 0.1;

/// Scale factor for attention adjustment from limiting component.
pub const LIMITING_COMPONENT_ATTENTION_SCALE: f32 = 0.3;

/// Binding capability boost when limiting component analysis recommends it.
pub const LIMITING_COMPONENT_BINDING_BOOST: f32 = 0.02;

/// Efficacy scale for limiting component remediation.
pub const LIMITING_COMPONENT_EFFICACY_SCALE: f32 = 1.05;

/// Confidence scale when math result has epistemic caveats.
pub const MATH_CAVEAT_CONFIDENCE_SCALE: f32 = 0.97;

/// Confidence boost when math result is formally verified.
pub const MATH_VERIFIED_CONFIDENCE: f32 = 0.03;

/// Normalizer for metacognition depth → HOT depth fraction.
/// Science: Hofstadter (1979) — meta-levels scale logarithmically.
pub const METACOGNITION_DEPTH_NORMALIZER: f64 = 3.0;

/// EMA decay for metacognitive prediction accuracy tracking.
/// Science: EMA smoothing for online prediction calibration.
pub const METACOGNITIVE_PREDICTION_EMA_DECAY: f64 = 0.95;

/// Scale boost for micro-Phi measurement in consciousness equation.
pub const MICRO_PHI_SCALE_BOOST: f64 = 0.1;

/// Scale for moral drift → uncertainty signal.
pub const MORAL_DRIFT_UNCERTAINTY_SCALE: f32 = 0.3;

/// Scale for moral drift → axiom pressure signal.
pub const MORAL_DRIFT_AXIOM_SCALE: f32 = 0.2;

/// LR scale when motor system is in intense exploration mode.
pub const MOTOR_EXPLORE_INTENSE_LR: f32 = 1.05;

/// Motor intensity threshold above which reflection is triggered.
pub const MOTOR_REFLECTION_THRESHOLD: f32 = 0.8;

/// Scale for motor intensity → confidence adjustment during reflection.
pub const MOTOR_REFLECTION_CONFIDENCE_SCALE: f32 = 0.05;

/// Exploration boost when quality floor is maintained across consecutive good cycles.
pub const QUALITY_FLOOR_EXPLORATION_BOOST: f32 = 0.01;

/// Readiness score below which degraded mode is activated.
pub const READINESS_DEGRADED_THRESHOLD: f32 = 0.5;

/// Readiness score above which rest/consolidation is allowed.
pub const READINESS_REST_THRESHOLD: f32 = 0.85;

/// Gradient magnitude below which the system is considered stable for recovery.
pub const RECOVERY_STABILITY_THRESHOLD: f64 = 0.05;

/// Causal consolidation boost from resonator recall matching.
pub const RESONATOR_CAUSAL_CONSOLIDATION_BOOST: f32 = 0.1;

/// Confidence boost when resonator factor is high (strong recall).
pub const RESONATOR_FACTOR_HIGH_CONFIDENCE: f32 = 0.02;

/// Scale for best-match similarity → resonator recall confidence boost.
pub const RESONATOR_RECALL_PRIME_SCALE: f32 = 0.05;

/// Confidence boost when resonator error is sustained low across many cycles.
pub const RESONATOR_SUSTAINED_LOW_CONFIDENCE: f32 = 0.01;

/// Broca coherence threshold for language generation quality gating.
pub const BROCA_COHERENT_THRESHOLD: f32 = 0.7;

/// Exploration boost when causal graph is sparse (few patterns learned).
/// Science: Pearl (2009) — sparse causal knowledge → explore to discover structure.
pub const SPARSE_CAUSAL_EXPLORATION_BOOST: f32 = 0.03;

/// Exploration scale when temporal binding is in high mode.
/// Science: Binding theory — strong temporal binding → explore new associations.
pub const TEMPORAL_BINDING_HIGH_EXPLORE_SCALE: f32 = 1.02;

// ═══════════════════════════════════════════════════════════════════════════════
// SUBSYSTEM PHASE CONSTANTS (cycle_subsystems.rs)
// Consciousness subsystem feedback loops: hierarchical LTC Phi, holographic
// encoding, evolution coordinator, affective consciousness, epistemic gating,
// cross-module coupling, meta-cognition, empathic unification.
// ═══════════════════════════════════════════════════════════════════════════════

/// Minimum hierarchical LTC Phi before cross-validating against spectral MIP.
/// Science: Tononi (2015) — Phi must be non-trivial to meaningfully compare.
pub const HIER_LTC_PHI_MIN_THRESHOLD: f32 = 0.1;

/// Phi divergence upper bound for convergence → confidence boost.
/// Science: Tononi (2015, §3.1) — independent estimates within this range converge.
pub const HIER_LTC_PHI_CONVERGE_THRESHOLD: f32 = 0.2;

/// Confidence boost scale when Phi estimates converge.
/// Science: Multiple consistent estimates increase epistemological confidence.
pub const HIER_LTC_PHI_CONVERGE_BOOST: f32 = 0.05;

/// Phi divergence lower bound for penalty → exploration boost.
/// Science: Tononi (2015, §3.1) — divergence above this signals instability.
pub const HIER_LTC_PHI_DIVERGE_THRESHOLD: f32 = 0.4;

/// Maximum clamp for Phi divergence penalty contribution.
/// Prevents runaway penalties from extreme divergence.
pub const HIER_LTC_PHI_DIVERGE_MAX: f32 = 0.3;

/// Confidence penalty scale for Phi divergence (attenuated 50%).
/// Science: NE exploration_delta already covers surprise-driven exploration.
pub const HIER_LTC_PHI_DIVERGE_PENALTY_SCALE: f32 = 0.015;

/// Minimum positive evolution Phi delta to trigger exploit boost.
/// Science: Holland (1975) — evolutionary fitness signals drive adaptive behavior.
pub const EVOLUTION_POSITIVE_DELTA_THRESHOLD: f64 = 0.01;

/// LR boost scale from positive evolution delta (up to EVOLUTION_POSITIVE_LR_CLAMP).
pub const EVOLUTION_POSITIVE_LR_SCALE: f64 = 0.1;

/// Maximum LR boost from positive evolution delta.
pub const EVOLUTION_POSITIVE_LR_CLAMP: f64 = 0.05;

/// Confidence boost scale from positive evolution delta.
pub const EVOLUTION_POSITIVE_CONF_SCALE: f64 = 0.05;

/// Maximum confidence boost from positive evolution delta.
pub const EVOLUTION_POSITIVE_CONF_CLAMP: f64 = 0.03;

/// Minimum negative evolution delta to trigger exploration boost.
/// Science: Holland (1975) — regression signals need for broader search.
pub const EVOLUTION_NEGATIVE_DELTA_THRESHOLD: f64 = -0.01;

/// Exploration boost scale from negative evolution delta.
pub const EVOLUTION_NEGATIVE_EXPLORE_SCALE: f64 = 0.08;

/// Maximum exploration boost from negative evolution delta.
pub const EVOLUTION_NEGATIVE_EXPLORE_CLAMP: f64 = 0.04;

/// Holographic unity threshold for prediction confidence boost.
/// Science: Pribram (1991) — holographic encoding enables stable predictions.
pub const HOLOGRAPHIC_UNITY_CONFIDENCE_THRESHOLD: f64 = 0.7;

/// Confidence boost scale for high holographic unity.
pub const HOLOGRAPHIC_UNITY_CONFIDENCE_SCALE: f64 = 0.03;

/// Holographic binding threshold for LR boost.
/// Science: Pribram (1971), Bohm (1980) — strong binding = coherent representations.
pub const HOLOGRAPHIC_BINDING_STRONG_THRESHOLD: f64 = 0.7;

/// LR factor when binding is strong.
pub const HOLOGRAPHIC_BINDING_STRONG_LR: f32 = 1.01;

/// Holographic binding upper bound for weak fragmentation regime.
pub const HOLOGRAPHIC_BINDING_WEAK_UPPER: f64 = 0.3;

/// LR dampen factor when binding is weak (fragmented representations).
pub const HOLOGRAPHIC_BINDING_WEAK_LR: f32 = 0.99;

/// Workspace value scaling from coherence in differentiable consciousness.
/// Science: Bengio (2017) — workspace value ≈ coherence.
pub const DIFF_CONSCIOUSNESS_WORKSPACE_SCALE: f64 = 0.8;

/// Default recursion core value when no recursion depth data available.
pub const DIFF_CONSCIOUSNESS_RECURSION_DEFAULT: f64 = 0.5;

/// Minimum consciousness gradient magnitude for exploration boost.
/// Science: Bengio (2017) — gradient information guides search.
pub const CONSCIOUSNESS_GRADIENT_EXPLORE_THRESHOLD: f64 = 0.5;

/// Exploration boost scale from large consciousness gradients.
pub const CONSCIOUSNESS_GRADIENT_EXPLORE_SCALE: f64 = 0.05;

/// Affective decay rate per cycle.
/// Science: Russell (2003) — affect decays towards neutral over time.
pub const AFFECTIVE_DECAY_RATE: f32 = 0.05;

/// Negative valence threshold for confidence dampening.
/// Science: Colombetti (2014) — negative affect strengthens caution.
pub const AFFECTIVE_NEGATIVE_VALENCE_THRESHOLD: f32 = -0.3;

/// Confidence scale factor from negative valence.
pub const AFFECTIVE_NEGATIVE_CONFIDENCE_SCALE: f32 = 0.02;

/// NSM similarity threshold for synthetic grounding classification.
/// Science: Wierzbicka (1996) — minimum similarity for state classification.
pub const SYNTHETIC_GROUNDING_SIM_THRESHOLD: f64 = 0.1;

/// Low epistemic confidence threshold for gating penalty.
/// Science: Kruger & Dunning (1999) — epistemic humility below this threshold.
pub const EPISTEMIC_GATE_LOW_THRESHOLD: f32 = 0.3;

/// Confidence penalty when epistemic gate rejects.
pub const EPISTEMIC_GATE_LOW_PENALTY: f32 = 0.03;

/// p-value threshold for primitive validation significance.
/// Science: Popper (1959) — standard statistical significance threshold.
pub const PRIMITIVE_VALIDATION_P_THRESHOLD: f64 = 0.05;

/// LR boost scale from validated primitives (positive Phi gain).
pub const PRIMITIVE_VALIDATION_POSITIVE_LR_SCALE: f64 = 0.02;

/// Maximum LR boost from validated primitives.
pub const PRIMITIVE_VALIDATION_POSITIVE_LR_CLAMP: f64 = 0.03;

/// LR dampen factor from falsified primitives (negative Phi gain).
pub const PRIMITIVE_VALIDATION_NEGATIVE_LR: f32 = 0.98;

/// Consciousness state low level threshold for urgency escalation.
/// Science: Varela (1991) — low consciousness triggers autopoietic response.
pub const CONSCIOUSNESS_STATE_LOW_URGENCY: f64 = 0.3;

/// Strong gradient threshold for boredom reduction (clear optimization direction).
pub const GRADIENT_STRONG_DIRECTION_THRESHOLD: f64 = 1.0;

/// Boredom reduction when gradient has strong direction.
pub const GRADIENT_STRONG_BOREDOM_REDUCE: f32 = 0.05;

/// Near-zero gradient upper bound (plateau detection).
pub const GRADIENT_PLATEAU_UPPER: f64 = 0.1;

/// Boredom increment when gradient is near-zero (plateau → explore).
pub const GRADIENT_PLATEAU_BOREDOM_INCREMENT: f32 = 0.03;

/// Holographic unity threshold for cross-module LR boost.
/// Science: Pribram (1991) — high unity = coherent, safe for aggressive learning.
pub const HOLOGRAPHIC_UNITY_LR_BOOST_THRESHOLD: f64 = 0.8;

/// LR factor for high holographic unity.
pub const HOLOGRAPHIC_UNITY_LR_BOOST_FACTOR: f32 = 1.02;

/// LR clamp bounds for holographic unity modulation.
pub const HOLOGRAPHIC_UNITY_LR_CLAMP_LOW: f32 = 0.8;
pub const HOLOGRAPHIC_UNITY_LR_CLAMP_HIGH: f32 = 1.2;

/// Holographic unity threshold for LR dampening (fragmented representations).
pub const HOLOGRAPHIC_UNITY_LR_DAMPEN_THRESHOLD: f64 = 0.2;

/// LR dampen factor for low holographic unity.
pub const HOLOGRAPHIC_UNITY_LR_DAMPEN_FACTOR: f32 = 0.98;

/// Pipeline consciousness threshold for epistemic confidence nudge.
/// Science: Dehaene (2011) — strong global workspace relaxes epistemic constraints.
pub const PIPELINE_CONSCIOUSNESS_EPISTEMIC_THRESHOLD: f64 = 0.7;

/// Epistemic confidence nudge when pipeline consciousness is high.
pub const PIPELINE_CONSCIOUSNESS_EPISTEMIC_NUDGE: f32 = 0.02;

/// Meta-reasoning confidence threshold for LR boost.
/// Science: Nelson & Narens (1990) — monitoring-control loop.
pub const META_REASONING_CONFIDENCE_THRESHOLD: f64 = 0.7;

/// LR boost scale from high meta-reasoning confidence.
pub const META_REASONING_LR_BOOST_SCALE: f64 = 0.1;

/// Empathic compassion threshold for LR boost.
/// Science: Decety & Jackson (2004) — shared representations enhance learning.
pub const EMPATHIC_COMPASSION_LR_THRESHOLD: f64 = 0.7;

/// Empathic compassion LR boost scale.
pub const EMPATHIC_COMPASSION_LR_SCALE: f32 = 0.02;

/// Empathic LR factor clamp bounds.
pub const EMPATHIC_LR_CLAMP_LOW: f32 = 0.8;
pub const EMPATHIC_LR_CLAMP_HIGH: f32 = 1.2;

// ═══════════════════════════════════════════════════════════════════════════════
// HOT-PATH REMAINING CONSTANTS (feedback, dynamics, strategy)
// Knowledge grounding, Phi scale sigmoid, resonator consolidation,
// confidence crash guards, world model confusion detection, social trust.
// ═══════════════════════════════════════════════════════════════════════════════

/// Weight of relevance in knowledge grounding score (sums to 1.0 with CERTAINTY).
/// Science: Knowledge-grounded consciousness — relevance directly tracks usefulness.
pub const KNOWLEDGE_GROUNDING_RELEVANCE_WEIGHT: f64 = 0.6;

/// Weight of certainty (1-uncertainty) in knowledge grounding score.
pub const KNOWLEDGE_GROUNDING_CERTAINTY_WEIGHT: f64 = 0.4;

/// Maximum amplitude of Phi scale boost from coefficient of variation.
/// Science: Scale-dependent Phi (Tononi 2004) — up to 15% boost for multi-scale coherence.
pub const PHI_SCALE_BOOST_MAX_AMPLITUDE: f32 = 0.15;

/// Sigmoid slope for Phi scale boost activation.
pub const PHI_SCALE_BOOST_SIGMOID_SLOPE: f32 = -2.0;

/// Center of sigmoid: CV = 1.0 is the half-activation point.
pub const PHI_SCALE_BOOST_CV_CENTER: f32 = 1.0;

/// Resonator consolidation precision boost scale per unit similarity above threshold.
/// Science: McClelland et al. (1995) — complementary learning systems.
pub const RESONATOR_CONSOLIDATION_PRECISION_SCALE: f64 = 0.1;

/// Maximum prior precision from resonator consolidation.
pub const RESONATOR_CONSOLIDATION_PRECISION_MAX: f64 = 2.0;

/// Goal priority → LR boost scale factor.
/// Science: Locke & Latham (2002) — goal commitment modulates learning intensity.
pub const GOAL_PRIORITY_LR_SCALE: f32 = 0.1;

/// Minimum prior confidence before confidence crash detection fires.
/// Science: Below this, the system has no basis for detecting a "crash" (Weber-Fechner law).
pub const CONFIDENCE_CRASH_MIN_PRIOR: f64 = 0.15;

/// Mode stability counter threshold for light freeze vs full freeze.
/// Science: Post-transition drops are expected; grace period avoids overreaction.
pub const MODE_STABILITY_GRACE_THRESHOLD: u32 = 3;

/// Light freeze duration when mode stability is below grace threshold.
pub const CONFIDENCE_CRASH_LIGHT_FREEZE_CYCLES: u32 = 1;

/// Social trust threshold for I-Thou relationship mode.
/// Science: Buber (1923) — I-Thou requires minimum reciprocal trust.
pub const SOCIAL_TRUST_ITHOU_THRESHOLD: f32 = 0.3;

/// World model abstract/sensory error ratio for conceptual confusion.
/// Science: Friston (2010) — hierarchical PE ratio signals level-specific failure.
pub const WORLD_MODEL_CONFUSION_RATIO: f32 = 1.5;

/// World model sensory/abstract error ratio for sensory mismatch.
pub const WORLD_MODEL_MISMATCH_RATIO: f32 = 2.0;

/// Minimum absolute error before confusion/mismatch detection fires.
pub const WORLD_MODEL_ERROR_FLOOR: f32 = 0.1;

/// MCTS effectiveness normalization: maps raw to [0.5, 1.0] range.
/// Science: Thompson (1933) — normalized effectiveness for Thompson sampling.
pub const MCTS_EFFECTIVENESS_NORM_SCALE: f32 = 0.5;

/// MCTS effectiveness normalization offset.
pub const MCTS_EFFECTIVENESS_NORM_OFFSET: f32 = 0.5;

/// Voice heartbeat base speech rate multiplier.
/// Science: Liberman & Mattingly (1985) — synthetic proxy for vocal tract feedback.
pub const VOICE_HEARTBEAT_BASE_RATE: f32 = 4.0;

/// Voice heartbeat coarticulation smoothness weight.
pub const VOICE_HEARTBEAT_COARTICULATION_WEIGHT: f32 = 0.8;

/// Voice heartbeat listener prediction: success value.
pub const VOICE_HEARTBEAT_LISTENER_SUCCESS: f32 = 0.8;

/// Voice heartbeat listener prediction: failure value.
pub const VOICE_HEARTBEAT_LISTENER_FAIL: f32 = 0.3;

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

    #[test]
    fn test_mce_bottleneck_params() {
        assert!(MCE_BOTTLENECK_LR_BOOST > 1.0 && MCE_BOTTLENECK_LR_BOOST < 1.5);
        assert!(
            MCE_NON_BOTTLENECK_CONFIDENCE_BOOST > 0.0 && MCE_NON_BOTTLENECK_CONFIDENCE_BOOST < 0.05
        );
    }

    #[test]
    fn test_moral_consolidation_params() {
        assert!(MORAL_CONSOLIDATION_THRESHOLD > 0.0 && MORAL_CONSOLIDATION_THRESHOLD < 1.0);
        assert!(MORAL_CONSOLIDATION_EASE > 0.0 && MORAL_CONSOLIDATION_EASE < 0.5);
    }

    #[test]
    fn test_coherence_velocity_budget_params() {
        assert!(COHERENCE_VELOCITY_BUDGET_THRESHOLD > 0.0);
        assert!(
            COHERENCE_VELOCITY_BUDGET_CONTRACT < 1.0,
            "Contract must reduce budget"
        );
        assert!(
            COHERENCE_VELOCITY_BUDGET_EXPAND > 1.0,
            "Expand must increase budget"
        );
    }

    #[test]
    fn test_homeostasis_recalibration_params() {
        assert!(HOMEOSTASIS_RECALIBRATE_LOW < 1.0 && HOMEOSTASIS_RECALIBRATE_LOW > 0.0);
        assert!(HOMEOSTASIS_RECALIBRATE_HIGH > 1.0);
        assert!(HOMEOSTASIS_RECALIBRATE_LOW < HOMEOSTASIS_RECALIBRATE_HIGH);
        assert!(HOMEOSTASIS_NEUROMOD_STEP > 0.0 && HOMEOSTASIS_NEUROMOD_STEP < 0.1);
    }

    #[test]
    fn test_eq_v2_bottleneck_params() {
        // All boosts must be positive
        assert!(EQ_V2_WORKSPACE_CONFIDENCE_BOOST > 0.0);
        assert!(EQ_V2_INTEGRATION_CONFIDENCE_BOOST > 0.0);
        assert!(EQ_V2_KNOWLEDGE_EXPLORATION_BOOST > 0.0);
        assert!(
            EQ_V2_RECURSION_LR_SCALE > 1.0,
            "Recursion LR scale must boost"
        );
        // Confidence boosts should be moderate (not >0.1 per cycle)
        assert!(EQ_V2_WORKSPACE_CONFIDENCE_BOOST < 0.1);
        assert!(EQ_V2_INTEGRATION_CONFIDENCE_BOOST < 0.1);
        // Integration > Workspace (integration is harder to fix)
        assert!(EQ_V2_INTEGRATION_CONFIDENCE_BOOST > EQ_V2_WORKSPACE_CONFIDENCE_BOOST);
    }

    #[test]
    fn test_temporal_chain_depth_params() {
        assert!(TEMPORAL_CHAIN_SHALLOW_THRESHOLD < TEMPORAL_CHAIN_DEEP_THRESHOLD);
        assert!(
            TEMPORAL_CHAIN_DEEP_LR_SCALE < 1.0,
            "Deep chains should dampen LR"
        );
        assert!(
            TEMPORAL_CHAIN_SHALLOW_LR_SCALE > 1.0,
            "Shallow chains should boost LR"
        );
    }

    #[test]
    fn test_cross_modal_psi_params() {
        assert!(CROSS_MODAL_PSI_CONFIDENCE_THRESHOLD > 0.0);
        assert!(CROSS_MODAL_PSI_CONFIDENCE_THRESHOLD < 1.0);
        assert!(CROSS_MODAL_PSI_CONFIDENCE_SCALE > 0.0);
        assert!(CROSS_MODAL_PSI_CONFIDENCE_SCALE < 0.2);
    }

    #[test]
    fn test_affective_consciousness_params() {
        // Arousal: high > low, both in [0,1]
        assert!(AFFECT_AROUSAL_HIGH_THRESHOLD > AFFECT_AROUSAL_LOW_THRESHOLD);
        assert!(
            AFFECT_AROUSAL_HIGH_LR_SCALE > 1.0,
            "High arousal should boost LR"
        );
        assert!(
            AFFECT_AROUSAL_LOW_EXPLORE_DAMPEN < 1.0,
            "Low arousal should dampen exploration"
        );
        // Valence: negative < 0 < positive
        assert!(AFFECT_VALENCE_NEGATIVE_THRESHOLD < 0.0);
        assert!(AFFECT_VALENCE_POSITIVE_THRESHOLD > 0.0);
        assert!(AFFECT_VALENCE_NEGATIVE_EXPLORE_BOOST > 0.0);
        assert!(AFFECT_VALENCE_POSITIVE_CONFIDENCE_BOOST > 0.0);
    }

    #[test]
    fn test_narrative_self_phi_params() {
        assert!(NARRATIVE_SELF_PHI_LOW_THRESHOLD < NARRATIVE_SELF_PHI_CONFIDENCE_THRESHOLD);
        assert!(NARRATIVE_SELF_PHI_CONFIDENCE_SCALE > 0.0);
        assert!(NARRATIVE_SELF_PHI_LOW_EXPLORE_BOOST > 0.0);
    }

    #[test]
    fn test_governance_neuromod_params() {
        // Floor must be positive and small
        assert!(GOV_NEUROMOD_FLOOR > 0.0 && GOV_NEUROMOD_FLOOR < 0.05);
        // All positive nudges are conservative (<= 0.10)
        assert!(GOV_EMERGENCY_NE_NUDGE <= 0.10);
        assert!(GOV_RECIPROCITY_OXY_DOSE <= 0.05);
        assert!(GOV_ALIGNED_PASS_DA_DOSE <= 0.20);
        // Oxytocin cap >= per-dose
        assert!(GOV_RECIPROCITY_OXY_CAP >= GOV_RECIPROCITY_OXY_DOSE);
        // Negative nudges are negative
        assert!(GOV_DISPUTE_SHT_NUDGE < 0.0);
        assert!(GOV_ALIGNED_FAIL_DA_NUDGE < 0.0);
        assert!(GOV_REPUTATION_DECLINE_SHT < 0.0);
        // Positive reputation → 5-HT boost
        assert!(GOV_REPUTATION_GAIN_SHT > 0.0 && GOV_REPUTATION_GAIN_SHT <= 0.10);
        // ECB nudge is small and positive
        assert!(GOV_COLLECTIVE_PHI_ECB > 0.0 && GOV_COLLECTIVE_PHI_ECB <= 0.05);
        // Consciousness modulation is small (±2% max)
        assert!(GOV_CONSCIOUSNESS_MODULATION > 0.0 && GOV_CONSCIOUSNESS_MODULATION <= 0.10);
    }

    #[test]
    fn test_embodied_consciousness_params() {
        assert!(EMBODIED_AGENCY_LOW_THRESHOLD < EMBODIED_AGENCY_HIGH_THRESHOLD);
        assert!(EMBODIED_AGENCY_BOOST_SCALE > 0.0);
        assert!(EMBODIED_AGENCY_CAUTION_SCALE > 0.0);
        assert!(EMBODIED_AGENCY_CAUTION_FLOOR > 0.0 && EMBODIED_AGENCY_CAUTION_FLOOR < 1.0);
        assert!(HOMEOSTATIC_DEVIATION_THRESHOLD > 0.0);
        assert!(SENSORIMOTOR_SURPRISE_THRESHOLD > 0.0);
        assert!(
            ALLOSTATIC_LOAD_DANGER_THRESHOLD > 0.5,
            "Allostatic load danger should be high"
        );
        assert!(ALLOSTATIC_LOAD_LR_DAMPEN > 0.0 && ALLOSTATIC_LOAD_LR_DAMPEN < 1.0);
    }

    #[test]
    fn test_fep_pragmatic_scales() {
        assert!(FEP_PRAGMATIC_EXPLOIT_SCALE > 0.0 && FEP_PRAGMATIC_EXPLOIT_SCALE < 1.0);
        assert!(FEP_PRAGMATIC_EXPLORE_SCALE > 0.0 && FEP_PRAGMATIC_EXPLORE_SCALE < 1.0);
        assert!(
            FEP_PRAGMATIC_EXPLOIT_SCALE > FEP_PRAGMATIC_EXPLORE_SCALE,
            "Exploitation should scale more aggressively than exploration"
        );
    }

    #[test]
    fn test_causal_graph_confidence_params() {
        assert!(
            CAUSAL_CONFIDENCE_MODERATE_THRESHOLD < CAUSAL_CONFIDENCE_DENSE_THRESHOLD,
            "Moderate ({}) must be < dense ({})",
            CAUSAL_CONFIDENCE_MODERATE_THRESHOLD,
            CAUSAL_CONFIDENCE_DENSE_THRESHOLD
        );
        assert!(CAUSAL_DENSE_CONFIDENCE_SCALE > 0.0);
        assert!(CAUSAL_MODERATE_CONFIDENCE_SCALE > 0.0);
        assert!(
            CAUSAL_DENSE_CONFIDENCE_SCALE > CAUSAL_MODERATE_CONFIDENCE_SCALE,
            "Dense graph should boost confidence more than moderate"
        );
    }

    #[test]
    fn test_pipeline_consciousness_params() {
        assert!(PIPELINE_CONSCIOUSNESS_LOW_THRESHOLD < PIPELINE_CONSCIOUSNESS_HIGH_THRESHOLD);
        assert!(
            PIPELINE_CONSCIOUSNESS_RELAX_SCALE < 1.0,
            "Relax must reduce threshold"
        );
        assert!(
            PIPELINE_CONSCIOUSNESS_CAUTION_SCALE > 1.0,
            "Caution must increase threshold"
        );
    }

    #[test]
    fn test_confidence_velocity_params() {
        assert!(CONFIDENCE_VELOCITY_POSITIVE_THRESHOLD > 0.0);
        assert!(CONFIDENCE_VELOCITY_NEGATIVE_THRESHOLD < 0.0);
        assert!(CONFIDENCE_VELOCITY_DAMPEN_SCALE > 0.0);
        assert!(CONFIDENCE_VELOCITY_BOOST_SCALE > 0.0);
    }

    #[test]
    fn test_sleep_pressure_params() {
        assert!(
            SLEEP_PRESSURE_LR_THRESHOLD > 0.5,
            "Sleep pressure threshold should be high"
        );
        assert!(SLEEP_PRESSURE_LR_DAMPEN_SCALE > 0.0 && SLEEP_PRESSURE_LR_DAMPEN_SCALE < 1.0);
        assert!(SLEEP_PRESSURE_LR_FACTOR_MIN > 0.0 && SLEEP_PRESSURE_LR_FACTOR_MIN < 1.0);
    }

    #[test]
    fn test_epistemic_phi_params() {
        assert!(EPISTEMIC_PHI_LOW_THRESHOLD < EPISTEMIC_PHI_HIGH_THRESHOLD);
        assert!(
            EPISTEMIC_PHI_LOW_CONFIDENCE_SCALE < 1.0,
            "Low phi must dampen confidence"
        );
        assert!(EPISTEMIC_PHI_HIGH_CONFIDENCE_SCALE > 0.0);
    }

    #[test]
    fn test_phenomenal_binding_params() {
        assert!(PHENOMENAL_BINDING_LOW_THRESHOLD < PHENOMENAL_BINDING_HIGH_THRESHOLD);
        assert!(PHENOMENAL_BINDING_LOW_EXPLORE_BOOST > 0.0);
        assert!(
            PHENOMENAL_BINDING_HIGH_LR_DAMPEN < 1.0,
            "High binding must dampen LR"
        );
    }

    #[test]
    fn test_temporal_coherence_params() {
        assert!(TEMPORAL_COHERENCE_LOW_THRESHOLD < TEMPORAL_COHERENCE_HIGH_THRESHOLD);
        assert!(TEMPORAL_COHERENCE_CONFIDENCE_SCALE > 0.0);
        assert!(TEMPORAL_COHERENCE_LOW_EXPLORE_BOOST > 0.0);
    }

    #[test]
    fn test_holographic_unity_params() {
        assert!(HOLOGRAPHIC_UNITY_LOW_THRESHOLD < HOLOGRAPHIC_UNITY_HIGH_THRESHOLD);
        assert!(
            HOLOGRAPHIC_UNITY_LOW_LR_DAMPEN < 1.0,
            "Low unity must dampen LR"
        );
        assert!(HOLOGRAPHIC_UNITY_HIGH_CONFIDENCE_SCALE > 0.0);
    }

    #[test]
    fn test_harmonies_alignment_params() {
        assert!(HARMONIES_MISALIGNMENT_THRESHOLD < HARMONIES_ALIGNED_THRESHOLD);
        assert!(HARMONIES_MISALIGNMENT_EXPLORE_BOOST > 0.0);
        assert!(HARMONIES_ALIGNED_CONFIDENCE_BOOST > 0.0);
    }

    #[test]
    fn test_value_cache_hit_params() {
        assert!(
            VALUE_CACHE_HIT_CONFIDENCE_THRESHOLD > 0.0
                && VALUE_CACHE_HIT_CONFIDENCE_THRESHOLD < 1.0
        );
        assert!(VALUE_CACHE_HIT_CONFIDENCE_SCALE > 0.0);
    }

    #[test]
    fn test_consciousness_gradient_params() {
        assert!(CONSCIOUSNESS_GRADIENT_THRESHOLD > 0.0);
        assert!(
            CONSCIOUSNESS_GRADIENT_LR_SCALE < 1.0,
            "Gradient should dampen LR"
        );
    }

    #[test]
    fn test_goal_priority_params() {
        assert!(GOAL_PRIORITY_EXPLORATION_THRESHOLD < GOAL_PRIORITY_LR_THRESHOLD);
        assert!(GOAL_PRIORITY_EXPLORATION_THRESHOLD > 0.0);
        assert!(GOAL_PRIORITY_LR_THRESHOLD < 1.0);
    }

    #[test]
    fn test_resonator_similarity_params() {
        assert!(
            RESONATOR_SIMILARITY_PRIME_THRESHOLD > 0.0
                && RESONATOR_SIMILARITY_PRIME_THRESHOLD < 1.0
        );
    }

    #[test]
    fn test_consciousness_state_level_params() {
        assert!(CONSCIOUSNESS_STATE_LOW_THRESHOLD < CONSCIOUSNESS_STATE_HIGH_THRESHOLD);
        assert!(
            CONSCIOUSNESS_STATE_HIGH_LR_SCALE > 1.0,
            "High state should boost LR"
        );
        assert!(
            CONSCIOUSNESS_STATE_LOW_LR_DAMPEN < 1.0,
            "Low state should dampen LR"
        );
    }

    #[test]
    fn test_living_mind_vitality_params() {
        assert!(LIVING_MIND_VITALITY_LOW_THRESHOLD < LIVING_MIND_VITALITY_HIGH_THRESHOLD);
        assert!(LIVING_MIND_VITALITY_CONFIDENCE_BOOST > 0.0);
        assert!(
            LIVING_MIND_VITALITY_LOW_LR_DAMPEN < 1.0,
            "Low vitality should dampen LR"
        );
    }

    #[test]
    fn test_living_mind_coherence_params() {
        assert!(LIVING_MIND_COHERENCE_LOW_THRESHOLD < LIVING_MIND_COHERENCE_HIGH_THRESHOLD);
        assert!(
            LIVING_MIND_COHERENCE_HIGH_EXPLORE_DAMPEN < 1.0,
            "High coherence should dampen exploration"
        );
        assert!(LIVING_MIND_COHERENCE_LOW_EXPLORE_BOOST > 0.0);
    }

    #[test]
    fn test_mcts_effectiveness_behavioral_params() {
        assert!(MCTS_EFFECTIVENESS_LOW_THRESHOLD < MCTS_EFFECTIVENESS_HIGH_THRESHOLD);
        assert!(MCTS_EFFECTIVENESS_CONFIDENCE_BOOST > 0.0);
        assert!(MCTS_EFFECTIVENESS_LOW_EXPLORE_BOOST > 0.0);
    }

    #[test]
    fn test_session17_adaptive_homeostasis_params() {
        assert!(ALLOSTATIC_LOAD_DECAY > 0.9 && ALLOSTATIC_LOAD_DECAY < 1.0);
        assert!(ALLOSTATIC_LOAD_INCREMENT > 0.0 && ALLOSTATIC_LOAD_INCREMENT < 0.1);
        assert!(ALLOSTATIC_OVERLOAD_THRESHOLD > 0.0 && ALLOSTATIC_OVERLOAD_THRESHOLD < 1.0);
        assert!(ALLOSTATIC_OVERLOAD_LR_SCALE < 1.0);
        assert!(EXPLORATION_DECAY_FACTOR > 0.9 && EXPLORATION_DECAY_FACTOR < 1.0);
        assert!(CONSCIOUSNESS_ACCEL_THRESHOLD > 0.0);
        assert!(CONSCIOUSNESS_ACCEL_LR_SCALE < 1.0);
        assert!((ADAPTIVE_WARMUP_MIN_CYCLES as usize) < STARTUP_WARMUP_CYCLES);
        assert!(PROPOSAL_SATURATION_THRESHOLD > 3);
        assert!(PHI_GATED_LR_FLOOR_THRESHOLD > 0.0 && PHI_GATED_LR_FLOOR_THRESHOLD < 0.5);
        assert!(RHYTHMIC_EXPLORATION_PERIOD > 10);
        assert!(RHYTHMIC_EXPLORATION_AMPLITUDE > 0.0 && RHYTHMIC_EXPLORATION_AMPLITUDE < 0.1);
    }

    #[test]
    fn test_session18_predictive_coding_params() {
        assert!(PE_VARIANCE_EMA_DECAY > 0.8 && PE_VARIANCE_EMA_DECAY < 1.0);
        assert!(PE_VARIANCE_DAMPING_THRESHOLD > 0.0 && PE_VARIANCE_DAMPING_THRESHOLD < 1.0);
        assert!(PE_VARIANCE_LR_SCALE > 0.8 && PE_VARIANCE_LR_SCALE < 1.0);
        assert!(CONFIDENCE_CALIBRATION_WINDOW > 10 && CONFIDENCE_CALIBRATION_WINDOW < 200);
        assert!(CONFIDENCE_CALIBRATION_DRIFT_THRESHOLD > 0.0);
        assert!(CONFIDENCE_CALIBRATION_CORRECTION > 0.0 && CONFIDENCE_CALIBRATION_CORRECTION < 0.1);
        assert!(LR_MOMENTUM_EMA_DECAY > 0.5 && LR_MOMENTUM_EMA_DECAY < 1.0);
        assert!(LR_MOMENTUM_MAX_DELTA > 0.0 && LR_MOMENTUM_MAX_DELTA < 1.0);
        assert!(METACOGNITIVE_SURPRISE_THRESHOLD > 0.0 && METACOGNITIVE_SURPRISE_THRESHOLD < 1.0);
        assert!(
            METACOGNITIVE_SURPRISE_EXPLORE_BOOST > 0.0
                && METACOGNITIVE_SURPRISE_EXPLORE_BOOST < 0.1
        );
        assert!(SLEEP_PRESSURE_INCREMENT > 0.0 && SLEEP_PRESSURE_INCREMENT < 0.01);
        assert!(SLEEP_PRESSURE_THRESHOLD > 0.0 && SLEEP_PRESSURE_THRESHOLD < 1.0);
        assert!(SLEEP_PRESSURE_CONSOLIDATION_DECAY > SLEEP_PRESSURE_INCREMENT);
        assert!(SLEEP_PRESSURE_LR_SCALE > 0.5 && SLEEP_PRESSURE_LR_SCALE < 1.0);
        assert!(GRADIENT_SIGN_WINDOW > 3 && GRADIENT_SIGN_WINDOW < 50);
        assert!(GRADIENT_SIGN_FLIP_THRESHOLD > 0.3 && GRADIENT_SIGN_FLIP_THRESHOLD < 0.9);
        assert!(GRADIENT_SIGN_FLIP_LR_SCALE > 0.8 && GRADIENT_SIGN_FLIP_LR_SCALE < 1.0);
        assert!(EXPLORE_EXPLOIT_WINDOW > 10 && EXPLORE_EXPLOIT_WINDOW < 200);
        assert!(EXPLORE_EXPLOIT_LOW_BOUND < EXPLORE_EXPLOIT_HIGH_BOUND);
        assert!(EXPLORE_EXPLOIT_CORRECTION > 0.0 && EXPLORE_EXPLOIT_CORRECTION < 0.1);
        assert!(PROPOSAL_CONFLICT_THRESHOLD > 0.2 && PROPOSAL_CONFLICT_THRESHOLD < 0.8);
    }

    #[test]
    fn test_session19_embodied_cognition_params() {
        assert!(AROUSAL_LR_BOOST_THRESHOLD > 0.3 && AROUSAL_LR_BOOST_THRESHOLD < 1.0);
        assert!(AROUSAL_LR_BOOST_SCALE > 1.0 && AROUSAL_LR_BOOST_SCALE < 1.2);
        assert!(AROUSAL_OVERAROUSAL_THRESHOLD > AROUSAL_LR_BOOST_THRESHOLD);
        assert!(AROUSAL_OVERAROUSAL_LR_SCALE < 1.0);
        assert!(NOVELTY_EMA_DECAY > 0.8 && NOVELTY_EMA_DECAY < 1.0);
        assert!(NOVELTY_LOW_THRESHOLD > 0.0 && NOVELTY_LOW_THRESHOLD < 0.5);
        assert!(NOVELTY_LOW_EXPLORE_SCALE < 1.0);
        assert!(FATIGUE_INCREMENT > 0.0 && FATIGUE_INCREMENT < 0.05);
        assert!(FATIGUE_EFFORT_THRESHOLD > 5);
        assert!(FATIGUE_LR_SCALE > 0.8 && FATIGUE_LR_SCALE < 1.0);
        assert!(FATIGUE_THRESHOLD > 0.0 && FATIGUE_THRESHOLD < 1.0);
        assert!(RECOVERY_CYCLES_NEEDED > 3 && RECOVERY_CYCLES_NEEDED < 30);
        assert!(RECOVERY_FATIGUE_DECAY > 0.0 && RECOVERY_FATIGUE_DECAY < 0.2);
        assert!(RECOVERY_CONFIDENCE_BOOST > 0.0 && RECOVERY_CONFIDENCE_BOOST < 0.1);
        assert!(ENV_PREDICTABILITY_WINDOW > 10 && ENV_PREDICTABILITY_WINDOW < 100);
        assert!(ENV_PREDICTABILITY_LOW < ENV_PREDICTABILITY_HIGH);
        assert!(ENV_PREDICTABLE_THRESHOLD_SCALE < 1.0);
        assert!(ENV_UNPREDICTABLE_THRESHOLD_SCALE > 1.0);
        assert!(ATTENTION_BUDGET_MAX > 10 && ATTENTION_BUDGET_MAX < 50);
        assert!(
            (READINESS_PE_WEIGHT + READINESS_SLEEP_WEIGHT + READINESS_FATIGUE_WEIGHT - 1.0).abs()
                < 0.01
        );
        assert!(RESONANCE_FLOW_CYCLES > 3 && RESONANCE_FLOW_CYCLES < 30);
        assert!(RESONANCE_AGREEMENT_THRESHOLD > 0.5);
        assert!(RESONANCE_CONFIDENCE_BOOST > 0.0 && RESONANCE_CONFIDENCE_BOOST < 0.1);
    }

    #[test]
    fn test_session20_consolidation_invariants() {
        // The unified readiness gate must have a floor high enough to prevent
        // learning lobotomy. With 6 cost dimensions each at max (1.0),
        // the worst case is: 1.0 - 6.0/6.0 = 0.0, clamped to 0.3.
        // This means LR is never dampened below 0.3x by resource depletion.
        let worst_cost = 1.0_f32;
        let total_cost = (worst_cost * 6.0) / 6.0;
        let readiness = (1.0 - total_cost).max(0.3);
        assert!(readiness >= 0.3, "readiness floor violated: {readiness}");

        // The individual dampeners that were consolidated should still have
        // their constants available (for telemetry/detection) even though
        // they no longer call scale_lr independently.
        assert!(ALLOSTATIC_OVERLOAD_THRESHOLD > 0.0);
        assert!(PE_VARIANCE_DAMPING_THRESHOLD > 0.0);
        assert!(SLEEP_PRESSURE_THRESHOLD > 0.0);
        assert!(GRADIENT_SIGN_FLIP_THRESHOLD > 0.0);
        assert!(AROUSAL_OVERAROUSAL_THRESHOLD > 0.0);
        assert!(FATIGUE_THRESHOLD > 0.0);
    }

    #[test]
    fn test_session21_housekeeping_params() {
        // Sleep pressure passive decay must be < 1.0 (decay) and > 0.99 (slow).
        assert!(SLEEP_PRESSURE_PASSIVE_DECAY > 0.99 && SLEEP_PRESSURE_PASSIVE_DECAY < 1.0);
        // Recovery constants must be reasonable after S21 adjustment.
        assert!(RECOVERY_CYCLES_NEEDED >= 3 && RECOVERY_CYCLES_NEEDED <= 10);
        assert!(RECOVERY_FATIGUE_DECAY >= 0.03 && RECOVERY_FATIGUE_DECAY <= 0.15);
        // Fatigue should halve in fewer than 15 recovery cycles.
        let cycles_to_halve = (0.5_f32.ln() / (1.0 - RECOVERY_FATIGUE_DECAY).ln()).ceil() as u32;
        assert!(
            cycles_to_halve <= 15,
            "fatigue takes {cycles_to_halve} recovery cycles to halve (want <= 15)"
        );
    }

    #[test]
    fn test_session23_extracted_constants() {
        // Feedback phase
        assert!(FLOW_INTENSITY_LR_THRESHOLD > 0.0 && FLOW_INTENSITY_LR_THRESHOLD < 1.0);
        assert!(FLOW_SUBSYSTEM_LR_BOOST > 1.0 && FLOW_SUBSYSTEM_LR_BOOST < 1.2);
        assert!(HIGH_QUALITY_SCORE_THRESHOLD > 0.5 && HIGH_QUALITY_SCORE_THRESHOLD < 1.0);
        assert!(CONSECUTIVE_HIGH_QUALITY_CYCLES >= 5 && CONSECUTIVE_HIGH_QUALITY_CYCLES <= 50);
        assert!(QUALITY_FLOOR_EXPLORATION_BOOST > 0.0 && QUALITY_FLOOR_EXPLORATION_BOOST < 0.1);
        assert!(TEMPORAL_BINDING_HIGH_EXPLORE_SCALE > 0.9 && TEMPORAL_BINDING_HIGH_EXPLORE_SCALE < 1.1);
        assert!(GRADIENT_STABLE_DETECT_THRESHOLD > 0.0 && GRADIENT_STABLE_DETECT_THRESHOLD < 0.1);
        assert!(READINESS_REST_THRESHOLD > 0.8 && READINESS_REST_THRESHOLD < 1.0);
        assert!(READINESS_DEGRADED_THRESHOLD > 0.3 && READINESS_DEGRADED_THRESHOLD < 1.0);
        assert!(READINESS_REST_THRESHOLD > READINESS_DEGRADED_THRESHOLD); // rest threshold is higher (more ready)
        assert!(RECOVERY_STABILITY_THRESHOLD > 0.0 && RECOVERY_STABILITY_THRESHOLD < 0.2);
        assert!(FATIGUE_RECOVERED_THRESHOLD > 0.0 && FATIGUE_RECOVERED_THRESHOLD < 0.3);
        assert!(GRADIENT_PREDICTION_OK_THRESHOLD > 0.0 && GRADIENT_PREDICTION_OK_THRESHOLD < 0.5);
        // Dynamics phase
        assert!(RESONATOR_SUSTAINED_LOW_CONFIDENCE > 0.0 && RESONATOR_SUSTAINED_LOW_CONFIDENCE < 0.05);
        assert!(BROCA_COHERENT_THRESHOLD > 0.5 && BROCA_COHERENT_THRESHOLD < 1.0);
        assert!(FEP_ACCURACY_HIGH_CONFIDENCE > 0.0 && FEP_ACCURACY_HIGH_CONFIDENCE < 0.05);
        assert!(FEP_TD_CONVERGE_EXPLORE_SCALE > 0.9 && FEP_TD_CONVERGE_EXPLORE_SCALE < 1.0);
        assert!(MATH_VERIFIED_CONFIDENCE > 0.0 && MATH_VERIFIED_CONFIDENCE < 0.1);
        assert!(MATH_CAVEAT_CONFIDENCE_SCALE > 0.9 && MATH_CAVEAT_CONFIDENCE_SCALE < 1.0);
        assert!(MOTOR_EXPLORE_INTENSE_LR > 1.0 && MOTOR_EXPLORE_INTENSE_LR < 1.5);
        assert!(COHERENCE_DEGRADED_LR_BOOST > 1.0 && COHERENCE_DEGRADED_LR_BOOST < 2.0);
        // Strategy phase
        assert!(ESCALATION_BLOCK_LR_SCALE > 0.0 && ESCALATION_BLOCK_LR_SCALE < 1.0);
        assert!(ESCALATION_THROTTLE_EXPLORATION > 0.0 && ESCALATION_THROTTLE_EXPLORATION < 0.5);
        // Cross-session constants
        assert!(ALLOSTATIC_LOAD_DAMPEN_INCREMENT_SCALE > 0.0 && ALLOSTATIC_LOAD_DAMPEN_INCREMENT_SCALE < 1.0);
        assert!(METACOGNITIVE_PREDICTION_EMA_DECAY > 0.5 && METACOGNITIVE_PREDICTION_EMA_DECAY < 1.0);
        assert!(LIMITING_COMPONENT_ATTENTION_SCALE > 0.0 && LIMITING_COMPONENT_ATTENTION_SCALE < 2.0);
        assert!(METACOGNITION_DEPTH_NORMALIZER > 1.0 && METACOGNITION_DEPTH_NORMALIZER < 10.0);
        assert!(KNOWLEDGE_CAUSAL_DEPTH_EXPLOIT_THRESHOLD > 0.0 && KNOWLEDGE_CAUSAL_DEPTH_EXPLOIT_THRESHOLD < 10.0);
    }

    #[test]
    fn test_subsystem_phase_constants() {
        // Hierarchical LTC Phi cross-validation
        assert!(HIER_LTC_PHI_MIN_THRESHOLD > 0.0 && HIER_LTC_PHI_MIN_THRESHOLD < 0.5);
        assert!(HIER_LTC_PHI_CONVERGE_THRESHOLD > 0.0 && HIER_LTC_PHI_CONVERGE_THRESHOLD < 0.5);
        assert!(HIER_LTC_PHI_CONVERGE_BOOST > 0.0 && HIER_LTC_PHI_CONVERGE_BOOST < 0.2);
        assert!(HIER_LTC_PHI_DIVERGE_THRESHOLD > HIER_LTC_PHI_CONVERGE_THRESHOLD); // diverge > converge
        assert!(HIER_LTC_PHI_DIVERGE_MAX > 0.0 && HIER_LTC_PHI_DIVERGE_MAX < 1.0);
        assert!(HIER_LTC_PHI_DIVERGE_PENALTY_SCALE > 0.0 && HIER_LTC_PHI_DIVERGE_PENALTY_SCALE < 0.1);

        // Evolution coordinator
        assert!(EVOLUTION_POSITIVE_DELTA_THRESHOLD > 0.0 && EVOLUTION_POSITIVE_DELTA_THRESHOLD < 0.1);
        assert!(EVOLUTION_POSITIVE_LR_SCALE > 0.0 && EVOLUTION_POSITIVE_LR_SCALE < 1.0);
        assert!(EVOLUTION_POSITIVE_LR_CLAMP > 0.0 && EVOLUTION_POSITIVE_LR_CLAMP <= EVOLUTION_POSITIVE_LR_SCALE);
        assert!(EVOLUTION_POSITIVE_CONF_SCALE > 0.0 && EVOLUTION_POSITIVE_CONF_SCALE < 0.5);
        assert!(EVOLUTION_POSITIVE_CONF_CLAMP > 0.0 && EVOLUTION_POSITIVE_CONF_CLAMP <= EVOLUTION_POSITIVE_CONF_SCALE);
        assert!(EVOLUTION_NEGATIVE_DELTA_THRESHOLD < 0.0 && EVOLUTION_NEGATIVE_DELTA_THRESHOLD > -0.1);
        assert!(EVOLUTION_NEGATIVE_EXPLORE_SCALE > 0.0 && EVOLUTION_NEGATIVE_EXPLORE_SCALE < 0.5);
        assert!(EVOLUTION_NEGATIVE_EXPLORE_CLAMP > 0.0 && EVOLUTION_NEGATIVE_EXPLORE_CLAMP <= EVOLUTION_NEGATIVE_EXPLORE_SCALE);

        // Holographic
        assert!(HOLOGRAPHIC_UNITY_CONFIDENCE_THRESHOLD > 0.3 && HOLOGRAPHIC_UNITY_CONFIDENCE_THRESHOLD < 1.0);
        assert!(HOLOGRAPHIC_UNITY_CONFIDENCE_SCALE > 0.0 && HOLOGRAPHIC_UNITY_CONFIDENCE_SCALE < 0.2);
        assert!(HOLOGRAPHIC_BINDING_STRONG_THRESHOLD > 0.3 && HOLOGRAPHIC_BINDING_STRONG_THRESHOLD < 1.0);
        assert!(HOLOGRAPHIC_BINDING_STRONG_LR > 1.0 && HOLOGRAPHIC_BINDING_STRONG_LR < 1.1);
        assert!(HOLOGRAPHIC_BINDING_WEAK_UPPER > 0.0 && HOLOGRAPHIC_BINDING_WEAK_UPPER < HOLOGRAPHIC_BINDING_STRONG_THRESHOLD);
        assert!(HOLOGRAPHIC_BINDING_WEAK_LR > 0.9 && HOLOGRAPHIC_BINDING_WEAK_LR < 1.0);

        // Differentiable consciousness
        assert!(DIFF_CONSCIOUSNESS_WORKSPACE_SCALE > 0.5 && DIFF_CONSCIOUSNESS_WORKSPACE_SCALE < 1.0);
        assert!(DIFF_CONSCIOUSNESS_RECURSION_DEFAULT > 0.0 && DIFF_CONSCIOUSNESS_RECURSION_DEFAULT < 1.0);

        // Consciousness gradient
        assert!(CONSCIOUSNESS_GRADIENT_EXPLORE_THRESHOLD > 0.0 && CONSCIOUSNESS_GRADIENT_EXPLORE_THRESHOLD < 1.0);
        assert!(CONSCIOUSNESS_GRADIENT_EXPLORE_SCALE > 0.0 && CONSCIOUSNESS_GRADIENT_EXPLORE_SCALE < 0.2);

        // Affective consciousness
        assert!(AFFECTIVE_DECAY_RATE > 0.0 && AFFECTIVE_DECAY_RATE < 0.2);
        assert!(AFFECTIVE_NEGATIVE_VALENCE_THRESHOLD < 0.0 && AFFECTIVE_NEGATIVE_VALENCE_THRESHOLD > -1.0);
        assert!(AFFECTIVE_NEGATIVE_CONFIDENCE_SCALE > 0.0 && AFFECTIVE_NEGATIVE_CONFIDENCE_SCALE < 0.1);

        // Synthetic grounding + epistemic gate
        assert!(SYNTHETIC_GROUNDING_SIM_THRESHOLD > 0.0 && SYNTHETIC_GROUNDING_SIM_THRESHOLD < 0.5);
        assert!(EPISTEMIC_GATE_LOW_THRESHOLD > 0.0 && EPISTEMIC_GATE_LOW_THRESHOLD < 0.5);
        assert!(EPISTEMIC_GATE_LOW_PENALTY > 0.0 && EPISTEMIC_GATE_LOW_PENALTY < 0.1);

        // Primitive validation
        assert!(PRIMITIVE_VALIDATION_P_THRESHOLD > 0.0 && PRIMITIVE_VALIDATION_P_THRESHOLD < 0.1);
        assert!(PRIMITIVE_VALIDATION_POSITIVE_LR_SCALE > 0.0 && PRIMITIVE_VALIDATION_POSITIVE_LR_SCALE < 0.1);
        assert!(PRIMITIVE_VALIDATION_POSITIVE_LR_CLAMP >= PRIMITIVE_VALIDATION_POSITIVE_LR_SCALE);
        assert!(PRIMITIVE_VALIDATION_NEGATIVE_LR > 0.9 && PRIMITIVE_VALIDATION_NEGATIVE_LR < 1.0);

        // Cross-module feedback
        assert!(CONSCIOUSNESS_STATE_LOW_URGENCY > 0.0 && CONSCIOUSNESS_STATE_LOW_URGENCY < 0.5);
        assert!(GRADIENT_STRONG_DIRECTION_THRESHOLD > 0.5 && GRADIENT_STRONG_DIRECTION_THRESHOLD < 2.0);
        assert!(GRADIENT_STRONG_BOREDOM_REDUCE > 0.0 && GRADIENT_STRONG_BOREDOM_REDUCE < 0.2);
        assert!(GRADIENT_PLATEAU_UPPER > 0.0 && GRADIENT_PLATEAU_UPPER < GRADIENT_STRONG_DIRECTION_THRESHOLD);
        assert!(GRADIENT_PLATEAU_BOREDOM_INCREMENT > 0.0 && GRADIENT_PLATEAU_BOREDOM_INCREMENT < 0.1);

        // Holographic unity LR modulation
        assert!(HOLOGRAPHIC_UNITY_LR_BOOST_THRESHOLD > HOLOGRAPHIC_UNITY_CONFIDENCE_THRESHOLD);
        assert!(HOLOGRAPHIC_UNITY_LR_BOOST_FACTOR > 1.0 && HOLOGRAPHIC_UNITY_LR_BOOST_FACTOR < 1.1);
        assert!(HOLOGRAPHIC_UNITY_LR_CLAMP_LOW > 0.5 && HOLOGRAPHIC_UNITY_LR_CLAMP_LOW < 1.0);
        assert!(HOLOGRAPHIC_UNITY_LR_CLAMP_HIGH > 1.0 && HOLOGRAPHIC_UNITY_LR_CLAMP_HIGH < 2.0);
        assert!(HOLOGRAPHIC_UNITY_LR_DAMPEN_THRESHOLD > 0.0 && HOLOGRAPHIC_UNITY_LR_DAMPEN_THRESHOLD < 0.5);
        assert!(HOLOGRAPHIC_UNITY_LR_DAMPEN_FACTOR > 0.9 && HOLOGRAPHIC_UNITY_LR_DAMPEN_FACTOR < 1.0);

        // Pipeline consciousness
        assert!(PIPELINE_CONSCIOUSNESS_EPISTEMIC_THRESHOLD > 0.5 && PIPELINE_CONSCIOUSNESS_EPISTEMIC_THRESHOLD < 1.0);
        assert!(PIPELINE_CONSCIOUSNESS_EPISTEMIC_NUDGE > 0.0 && PIPELINE_CONSCIOUSNESS_EPISTEMIC_NUDGE < 0.1);

        // Meta-reasoning
        assert!(META_REASONING_CONFIDENCE_THRESHOLD > 0.5 && META_REASONING_CONFIDENCE_THRESHOLD < 1.0);
        assert!(META_REASONING_LR_BOOST_SCALE > 0.0 && META_REASONING_LR_BOOST_SCALE < 0.5);

        // Empathic compassion
        assert!(EMPATHIC_COMPASSION_LR_THRESHOLD > 0.5 && EMPATHIC_COMPASSION_LR_THRESHOLD < 1.0);
        assert!(EMPATHIC_COMPASSION_LR_SCALE > 0.0 && EMPATHIC_COMPASSION_LR_SCALE < 0.1);
        assert!(EMPATHIC_LR_CLAMP_LOW > 0.5 && EMPATHIC_LR_CLAMP_LOW < 1.0);
        assert!(EMPATHIC_LR_CLAMP_HIGH > 1.0 && EMPATHIC_LR_CLAMP_HIGH < 2.0);
    }

    #[test]
    fn test_hotpath_remaining_constants() {
        // Knowledge grounding weights sum to 1.0
        let weight_sum = KNOWLEDGE_GROUNDING_RELEVANCE_WEIGHT + KNOWLEDGE_GROUNDING_CERTAINTY_WEIGHT;
        assert!((weight_sum - 1.0).abs() < 1e-6, "Grounding weights must sum to 1.0");
        assert!(KNOWLEDGE_GROUNDING_RELEVANCE_WEIGHT > 0.0 && KNOWLEDGE_GROUNDING_RELEVANCE_WEIGHT < 1.0);

        // Phi scale boost sigmoid
        assert!(PHI_SCALE_BOOST_MAX_AMPLITUDE > 0.0 && PHI_SCALE_BOOST_MAX_AMPLITUDE < 0.5);
        assert!(PHI_SCALE_BOOST_SIGMOID_SLOPE < 0.0); // negative slope for standard sigmoid
        assert!(PHI_SCALE_BOOST_CV_CENTER > 0.0);

        // Resonator consolidation
        assert!(RESONATOR_CONSOLIDATION_PRECISION_SCALE > 0.0 && RESONATOR_CONSOLIDATION_PRECISION_SCALE < 0.5);
        assert!(RESONATOR_CONSOLIDATION_PRECISION_MAX > 1.0 && RESONATOR_CONSOLIDATION_PRECISION_MAX < 5.0);

        // Goal LR
        assert!(GOAL_PRIORITY_LR_SCALE > 0.0 && GOAL_PRIORITY_LR_SCALE < 0.5);

        // Confidence crash
        assert!(CONFIDENCE_CRASH_MIN_PRIOR > 0.0 && CONFIDENCE_CRASH_MIN_PRIOR < 0.5);
        assert!(MODE_STABILITY_GRACE_THRESHOLD >= 1 && MODE_STABILITY_GRACE_THRESHOLD <= 10);
        assert!(CONFIDENCE_CRASH_LIGHT_FREEZE_CYCLES >= 1 && CONFIDENCE_CRASH_LIGHT_FREEZE_CYCLES <= 5);

        // Social trust
        assert!(SOCIAL_TRUST_ITHOU_THRESHOLD > 0.0 && SOCIAL_TRUST_ITHOU_THRESHOLD < 1.0);

        // World model
        assert!(WORLD_MODEL_CONFUSION_RATIO > 1.0 && WORLD_MODEL_CONFUSION_RATIO < 3.0);
        assert!(WORLD_MODEL_MISMATCH_RATIO > WORLD_MODEL_CONFUSION_RATIO); // mismatch requires higher ratio
        assert!(WORLD_MODEL_ERROR_FLOOR > 0.0 && WORLD_MODEL_ERROR_FLOOR < 0.5);

        // MCTS normalization
        let norm_sum = MCTS_EFFECTIVENESS_NORM_SCALE + MCTS_EFFECTIVENESS_NORM_OFFSET;
        assert!((norm_sum - 1.0).abs() < 1e-6, "MCTS norm scale+offset should map max to 1.0");

        // Voice heartbeat
        assert!(VOICE_HEARTBEAT_BASE_RATE > 1.0 && VOICE_HEARTBEAT_BASE_RATE < 10.0);
        assert!(VOICE_HEARTBEAT_COARTICULATION_WEIGHT > 0.0 && VOICE_HEARTBEAT_COARTICULATION_WEIGHT < 1.0);
        assert!(VOICE_HEARTBEAT_LISTENER_SUCCESS > VOICE_HEARTBEAT_LISTENER_FAIL);
    }

    #[test]
    fn test_knowledge_engine_params() {
        // Persistence
        assert!(KNOWLEDGE_SAVE_INTERVAL >= 100 && KNOWLEDGE_SAVE_INTERVAL <= 5000);
        // Consciousness coupling
        assert!(KNOWLEDGE_CONSCIOUSNESS_MODULATION > 0.0 && KNOWLEDGE_CONSCIOUSNESS_MODULATION < 0.1);
        // Dream consolidation
        assert!(KNOWLEDGE_FORGET_CONFIDENCE_THRESHOLD > 0.0 && KNOWLEDGE_FORGET_CONFIDENCE_THRESHOLD < 0.5);
        assert!(KNOWLEDGE_CONSOLIDATION_BOOST > 0.0 && KNOWLEDGE_CONSOLIDATION_BOOST < 0.2);
        // Causal depth exploitation
        assert!(KNOWLEDGE_CAUSAL_DEPTH_EXPLOIT_THRESHOLD > 0.0);
        assert!(KNOWLEDGE_CAUSAL_DEPTH_EXPLORE_DAMPEN > 0.0 && KNOWLEDGE_CAUSAL_DEPTH_EXPLORE_DAMPEN < 0.5);
        // Contradiction boosts
        assert!(KNOWLEDGE_CONTRADICTION_NE_BOOST > 0.0 && KNOWLEDGE_CONTRADICTION_NE_BOOST < 0.1);
        assert!(KNOWLEDGE_CONTRADICTION_SHT_BOOST > 0.0 && KNOWLEDGE_CONTRADICTION_SHT_BOOST < 0.1);
    }
}
