// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Learning, plasticity, FEP, memory, resonator, strategy selection, and epistemic constants.

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

/// Minimum full-dimension (16,384-D) input-HDC similarity before an episodic
/// recall is blended into the CfC prediction — Predictive Compression
/// Program C3 (docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md §7).
/// Distinct from `MEMORY_RECALL_SIM_THRESHOLD` above (a different recall
/// path, `EpisodicMemoryBridge`, scored on a 64-float embedding sample, not
/// the full vector) — registered value, chosen as the midpoint between
/// "uncorrelated" (~0.0 for random full-dimension HDC vectors) and
/// "near-duplicate" (~0.9+); revisit if C3's manipulation check fails.
pub const RECALL_BLEND_SIM_THRESHOLD: f32 = 0.5;

/// Default input similarity memoization threshold (cosine).
/// Basis: Tulving & Schacter (1990) — repetition priming allows processing shortcuts.
pub const INPUT_MEMO_THRESHOLD: f32 = 0.95;

// ── Perception Manager ──────────────────────────────────────────────────────

/// Perceptual coherence above which confidence is boosted.
/// Basis: Dehaene & Changeux (2011) — high coherence indicates global workspace access.
pub const PERCEPTION_COHERENCE_HIGH: f32 = 0.7;

/// Perceptual coherence below which exploration is encouraged.
/// Basis: Friston (2010) — low coherence signals high free-energy, triggering exploration.
pub const PERCEPTION_COHERENCE_LOW: f32 = 0.3;

/// Coherence threshold for entering vigilant mode.
/// Basis: Posner & Petersen (1990) — alerting network activates on degraded percepts.
pub const PERCEPTION_VIGILANCE_COHERENCE: f32 = 0.4;

/// Prediction error threshold for entering vigilant mode.
/// Basis: Rao & Ballard (1999) — high PE signals model-world mismatch.
pub const PERCEPTION_VIGILANCE_PE: f32 = 0.4;

/// Phenomenal binding above which confidence is boosted.
/// Basis: Treisman & Gelade (1980) — feature integration theory, bound percepts are reliable.
pub const PERCEPTION_BINDING_HIGH: f32 = 0.7;

/// Phenomenal binding below which confidence is penalized.
/// Basis: Treisman & Gelade (1980) — unbound features are perceptually unreliable.
pub const PERCEPTION_BINDING_LOW: f32 = 0.3;

/// Minimum attention sensitivity floor.
/// Basis: Broadbent (1958) — filter theory, sensitivity cannot drop below baseline.
pub const PERCEPTION_SENSITIVITY_MIN: f32 = 0.5;

/// Maximum attention sensitivity ceiling.
/// Basis: Kahneman (1973) — attention as limited resource, upper bound on allocation.
pub const PERCEPTION_SENSITIVITY_MAX: f32 = 2.0;

// ═══════════════════════════════════════════════════════════════════════════════
// OUTPUT PHASE — EPISTEMIC & STABILITY THRESHOLDS
// ═══════════════════════════════════════════════════════════════════════════════

/// EMA prior weight for epistemic uncertainty smoothing (1 - alpha).
/// Basis: Standard EMA with alpha=0.2 — balances responsiveness with stability.
pub const EPISTEMIC_UNCERTAINTY_EMA_PRIOR: f32 = 0.8;

/// EMA current weight for epistemic uncertainty smoothing (alpha).
pub const EPISTEMIC_UNCERTAINTY_EMA_CURRENT: f32 = 0.2;

/// Cross-module agreement threshold below which compound instability is flagged.
/// Basis: Friston (2010) — low agreement indicates prediction model fragmentation.
pub const COMPOUND_INSTABILITY_AGREEMENT: f32 = 0.5;

/// Proposal conflict ratio above which epistemic exploration is boosted.
/// Basis: Berlyne (1960) — conceptual conflict drives curiosity/exploration.
pub const PROPOSAL_CONFLICT_EXPLORATION: f32 = 0.3;

/// Flow intensity threshold above which feedback relaxation is applied.
/// Basis: Csikszentmihalyi (1990) — strong flow states should not be disrupted.
pub const FLOW_INTENSITY_FEEDBACK: f32 = 0.5;

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
// EPISTEMIC AUDITOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Epistemic Auditor DuckDB flush cadence (cycles between flushes).
/// 1009 is prime and co-prime to all existing cadences (7, 11, 13, 41, 47, 53, 97, 199, 499, 997).
/// At 31Hz, this is ~32.5 seconds between flushes.
/// Science: intermittent self-reflection avoids metacognitive overhead (Flavell 1979).
pub const EPISTEMIC_AUDITOR_FLUSH_CADENCE: u64 = 1009;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-MODULE QUALITY METRICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Cross-module agreement above this signals high convergence → confidence boost.
pub const CROSS_MODULE_AGREEMENT_HIGH: f32 = 0.8;

/// Cross-module agreement below this signals divergence → exploration boost.
pub const CROSS_MODULE_AGREEMENT_LOW: f32 = 0.3;

/// Unified quality score composition: prediction accuracy weight.
pub const UNIFIED_QUALITY_PREDICTION_WEIGHT: f32 = 0.5;

/// Unified quality score composition: cross-module agreement weight.
pub const UNIFIED_QUALITY_AGREEMENT_WEIGHT: f32 = 0.3;

/// Unified quality score composition: anomaly (inverse) weight.
pub const UNIFIED_QUALITY_ANOMALY_WEIGHT: f32 = 0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// FEP / ACTIVE INFERENCE DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════

/// FEP accuracy above this triggers confidence boost.
/// Basis: Friston (2010) — accurate prediction reduces free energy.
pub const FEP_ACCURACY_CONFIDENCE_THRESHOLD: f64 = 0.5;

/// FEP complexity above this dampens learning rate.
/// Basis: Bayesian model complexity penalty (BIC/MDL principle).
pub const FEP_COMPLEXITY_THRESHOLD: f64 = 1.0;

/// FEP pragmatic value above this triggers exploitation strategy.
/// Basis: Friston et al. (2015) — expected free energy pragmatic term.
pub const FEP_PRAGMATIC_EXPLOIT_THRESHOLD: f64 = 0.7;

/// FEP pragmatic value below this triggers exploration.
pub const FEP_PRAGMATIC_EXPLORE_THRESHOLD: f64 = 0.3;

/// FEP temporal difference error above this triggers causal discovery.
pub const FEP_TD_ERROR_DISCOVERY_THRESHOLD: f64 = 0.5;

/// FEP learning signal above this enables world model plasticity increase.
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
// MOTOR OUTPUT BRIDGE — PHI-GATED ACTION SAFETY
// ═══════════════════════════════════════════════════════════════════════════════

/// Default prediction error for skipped motor actions.
pub const MOTOR_SKIP_PREDICTION_ERROR: f64 = 0.5;

/// Minimum motor confidence for action execution.
/// Below this, FEP isn't certain enough to commit.
pub const MOTOR_CONFIDENCE_MIN: f64 = 0.3;

/// Phi bonus for reversible actions (added to min_phi).
/// Basis: graduated safety — reversible actions need slightly more consciousness.
pub const MOTOR_PHI_REVERSIBLE_BONUS: f64 = 0.1;

/// Phi bonus for actions needing confirmation.
pub const MOTOR_PHI_CONFIRMATION_BONUS: f64 = 0.2;

/// Phi bonus for destructive actions (highest consciousness requirement).
pub const MOTOR_PHI_DESTRUCTIVE_BONUS: f64 = 0.3;

// ═══════════════════════════════════════════════════════════════════════════════
// MEMORY PHASE — CODEBOOK & RESONATOR DYNAMICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Co-prime cadence for high-Phi episode→codebook promotion.
/// Basis: Dehaene (2014) — conscious access creates durable representations.
pub const MEMORY_HIGH_PHI_PROMOTION_CADENCE: usize = 97;

/// Codebook diversity computation amortization interval (cycles).
/// Basis: Kohonen (1982) — periodic reorganization of self-organizing maps.
pub const MEMORY_CODEBOOK_DIVERSITY_INTERVAL: usize = 50;

/// Codebook utilization rate update interval (cycles).
pub const MEMORY_CODEBOOK_UTILIZATION_INTERVAL: usize = 50;

/// Codebook utilization EMA decay.
pub const MEMORY_CODEBOOK_UTIL_EMA_DECAY: f32 = 0.8;

/// Codebook utilization EMA new-data weight.
pub const MEMORY_CODEBOOK_UTIL_EMA_NEW: f32 = 0.2;

/// Novelty threshold increase rate when codebook utilization is low.
pub const MEMORY_NOVELTY_THRESHOLD_INCREASE: f32 = 0.02;

/// Novelty threshold decrease rate when codebook utilization is high.
pub const MEMORY_NOVELTY_THRESHOLD_DECREASE: f32 = 0.01;

/// Low utilization trigger (below this, increase novelty threshold).
pub const MEMORY_CODEBOOK_LOW_UTILIZATION: f32 = 0.2;

/// High utilization trigger (above this, decrease novelty threshold).
pub const MEMORY_CODEBOOK_HIGH_UTILIZATION: f32 = 0.6;

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
// EPISTEMIC CONFLICT
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-conflict exploration boost scale.
/// Basis: Berlyne (1960) — epistemic curiosity arises from conflicting beliefs.
pub const EPISTEMIC_CONFLICT_EXPLORE_SCALE: f32 = 0.015;

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
// CROSS-COUPLING: DRIVE → LEARNING
// ═══════════════════════════════════════════════════════════════════════════════

/// Boredom level above which plasticity boost kicks in.
/// Science: Berlyne (1960) — boredom triggers exploratory information-seeking.
pub const DRIVE_BOREDOM_PLASTICITY_THRESHOLD: f32 = 0.5;

/// Plasticity gain per unit of boredom above threshold.
/// Conservative: 10% max boost at full boredom (0.8 - 0.5) * 0.33 ≈ 0.1.
/// Science: Berlyne (1960) — exploration intensity scales with deprivation.
pub const DRIVE_BOREDOM_PLASTICITY_GAIN: f32 = 0.33;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-COUPLING: KNOWLEDGE → ETHICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Normalized causal depth above which moral confidence nudge activates.
/// causal_depth is normalized to [0,1] (raw depth / 5). 0.6 ≈ 3 raw steps.
/// Science: Pearl (2009) — multi-step causal reasoning grounds moral judgment.
pub const KNOWLEDGE_ETHICS_CAUSAL_DEPTH_THRESHOLD: f64 = 0.6;

/// Prediction confidence gain per unit of excess causal depth.
/// Capped at 0.03 total. Small effect to avoid overconfident moral verdicts.
/// Science: Pearl (2009) — causal understanding raises, but doesn't guarantee, reasoning quality.
pub const KNOWLEDGE_ETHICS_CONFIDENCE_GAIN: f64 = 0.025;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-COUPLING: MEMORY → LEARNING
// ═══════════════════════════════════════════════════════════════════════════════

/// Consolidation pressure above which learning plasticity gets boosted.
/// Science: Born & Wilhelm (2012) — memory consolidation pressure primes subsequent encoding.
pub const MEMORY_CONSOLIDATION_PLASTICITY_THRESHOLD: f32 = 0.6;

/// LR factor boost per unit of consolidation pressure above threshold.
/// Conservative: max boost ≈ (1.0 - 0.6) * 0.15 = 0.06 (6%).
/// Science: Diekelmann & Born (2010) — consolidation enhances subsequent learning.
pub const MEMORY_CONSOLIDATION_PLASTICITY_GAIN: f32 = 0.15;

/// Recall quality below which learning dampening kicks in.
/// Low recall quality signals unreliable memory retrieval → reduce learning rate.
/// Science: Tulving (2002) — retrieval failure indicates encoding problems.
pub const MEMORY_RECALL_QUALITY_DAMPEN_THRESHOLD: f32 = 0.3;

/// LR factor dampening scale when recall quality is poor.
/// Applied as: LR *= (1.0 - (threshold - quality) * scale)
/// Science: Roediger & Karpicke (2006) — testing effect requires successful retrieval.
pub const MEMORY_RECALL_QUALITY_DAMPEN_SCALE: f32 = 0.2;

// ═══════════════════════════════════════════════════════════════════════════════
// CROSS-COUPLING: PERCEPTION → DRIVE
// ═══════════════════════════════════════════════════════════════════════════════

/// Coherence below which perception signals drive system to seek novelty.
/// Science: Damasio (1994) — low perceptual coherence triggers orienting response.
pub const PERCEPTION_LOW_COHERENCE_THRESHOLD: f32 = 0.3;

/// Exploration boost scale when perception coherence is low.
/// Applied per unit of deficit below threshold.
/// Science: Sokolov (1963) — orienting reflex scales with stimulus novelty.
pub const PERCEPTION_LOW_COHERENCE_EXPLORE_GAIN: f32 = 0.08;

/// Budget utilization above which drive exploration is suppressed.
/// Science: Lavie (2005) — high perceptual load reduces capacity for exploration.
pub const PERCEPTION_HIGH_LOAD_SUPPRESS_THRESHOLD: f32 = 0.8;

/// Exploration suppression factor when perceptual load is high.
/// Applied as: exploration *= (1.0 - (utilization - threshold) * factor)
pub const PERCEPTION_HIGH_LOAD_SUPPRESS_FACTOR: f32 = 0.5;

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

/// Exploration boost when causal graph is sparse (few patterns learned).
/// Science: Pearl (2009) — sparse causal knowledge → explore to discover structure.
pub const SPARSE_CAUSAL_EXPLORATION_BOOST: f32 = 0.03;

/// Exploration scale when temporal binding is in high mode.
/// Science: Binding theory — strong temporal binding → explore new associations.
pub const TEMPORAL_BINDING_HIGH_EXPLORE_SCALE: f32 = 1.02;

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

// ═══════════════════════════════════════════════════════════════════════════════
// CYCLE EXTRACTED: Moral feedback, memory, FEP, arousal
// ═══════════════════════════════════════════════════════════════════════════════

/// Scale for value trend moral feedback: `1.0 + (trend * scale)`.
/// Basis: Self-correcting moral alignment via TD-learning.
pub const MORAL_VALUE_FEEDBACK_SCALE: f32 = 0.1;

/// Memory valence threshold for emotional re-experiencing.
/// Basis: Damasio (1999) — emotional re-experiencing from recalled episodes.
pub const MEMORY_VALENCE_THRESHOLD: f32 = 0.1;

/// Scale for memory valence nudge to emotional state.
pub const MEMORY_VALENCE_NUDGE_SCALE: f32 = 0.15;

/// Memory Phi threshold for consciousness priming.
pub const MEMORY_PHI_PRIME_THRESHOLD: f32 = 0.4;

/// Scale for memory Phi confidence priming: `(phi - threshold) * scale`.
pub const MEMORY_PHI_PRIME_SCALE: f32 = 0.05;

/// Scale for surprise bridge exploration adjustment.
pub const SURPRISE_EXPLORATION_FACTOR_SCALE: f32 = 0.3;

/// Baseline integration scale from smoothed coherence.
/// Basis: Tononi (2004) — Φ reflects degree of information integration.
pub const BASELINE_INTEGRATION_SCALE: f32 = 0.3;

/// FEP free energy reduction weight in reward signal.
/// Basis: Friston (2010) — free energy minimization as objective.
pub const FEP_REWARD_WEIGHT: f32 = 0.15;

/// Consent violation confidence scale on the moral confidence pathway.
pub const MORAL_CONSENT_CONFIDENCE_SCALE: f32 = 0.7;

/// LR factor for consent violations (halve learning rate).
pub const MORAL_CONSENT_LR_FACTOR: f32 = 0.5;

/// Exploration scale for harm detection.
pub const MORAL_HARM_EXPLORATION_SCALE: f32 = 0.4;

/// Confidence scale for harm detection.
pub const MORAL_HARM_CONFIDENCE_SCALE: f32 = 0.85;

/// LR factor for duty violations.
pub const MORAL_DUTY_LR_FACTOR: f32 = 0.8;

/// LR factor for other (unclassified) moral violations.
pub const MORAL_OTHER_LR_FACTOR: f32 = 0.9;

/// PFE high threshold for surprise amplification.
pub const PFE_SURPRISE_HIGH_THRESHOLD: f64 = 0.5;

/// PFE surprise amplification scale.
pub const PFE_SURPRISE_AMPLIFY_SCALE: f64 = 0.2;

/// PFE surprise amplification maximum.
pub const PFE_SURPRISE_AMPLIFY_MAX: f64 = 0.1;

/// PFE low threshold for surprise dampening.
pub const PFE_SURPRISE_LOW_THRESHOLD: f64 = 0.2;

/// PFE surprise dampening scale.
pub const PFE_SURPRISE_DAMPEN_SCALE: f64 = 0.15;

/// PFE surprise dampening maximum.
pub const PFE_SURPRISE_DAMPEN_MAX: f64 = 0.05;

/// FEP action 0: free energy LR boost divisor.
pub const FEP_ACTION_FE_DIVISOR: f32 = 2.0;

/// FEP action 0: free energy LR boost scale.
pub const FEP_ACTION_FE_BOOST_SCALE: f32 = 0.5;

/// FEP action 1: sensory precision reset blend (new=0.3, old=0.7).
pub const FEP_SENSORY_PRECISION_BLEND: f32 = 0.3;

/// FEP action 2: exploration nudge when surprised.
pub const FEP_EXPLORATION_NUDGE_SURPRISED: f32 = 0.15;

/// FEP action 2: exploration nudge when calm.
pub const FEP_EXPLORATION_NUDGE_CALM: f32 = 0.05;

/// Arousal suppression threshold.
/// Basis: Yerkes-Dodson (1908) — inverted-U performance curve.
pub const AROUSAL_SUPPRESS_THRESHOLD: f32 = 0.7;

/// Arousal suppression scale: `(arousal - threshold) * scale`.
pub const AROUSAL_SUPPRESS_SCALE: f32 = 0.25;

/// Arousal suppression maximum per cycle.
pub const AROUSAL_SUPPRESS_MAX: f32 = 0.08;

/// Arousal trap detection threshold.
pub const AROUSAL_TRAP_THRESHOLD: f32 = 0.8;

/// Low arousal consolidation threshold.
/// Basis: Steriade (1996) — low arousal enhances consolidation.
pub const LOW_AROUSAL_CONSOLIDATION_THRESHOLD: f32 = 0.3;

/// Low arousal consolidation boost scale.
pub const LOW_AROUSAL_CONSOLIDATION_SCALE: f32 = 0.3;

/// Low arousal consolidation boost maximum.
pub const LOW_AROUSAL_CONSOLIDATION_MAX: f32 = 0.05;

// ═══════════════════════════════════════════════════════════════════════════════
// CYCLE EXTRACTED: Cross-modal binding, reflection, physics
// ═══════════════════════════════════════════════════════════════════════════════

/// Linguistic modality confidence (used in cross-modal binding).
pub const CROSS_MODAL_LINGUISTIC_CONFIDENCE: f32 = 0.8;

/// Cross-modal binding precision boost threshold.
/// Basis: Talsma (2015) — predictive-binding bidirectional coupling.
pub const CROSS_MODAL_PRECISION_BOOST_THRESHOLD: f32 = 0.5;

/// Cross-modal binding precision boost scale.
pub const CROSS_MODAL_PRECISION_BOOST_SCALE: f64 = 0.1;

/// Predictive free energy threshold for cross-modal dampen.
pub const CROSS_MODAL_FE_DAMPEN_THRESHOLD: f64 = 0.6;

/// Scale for cross-modal attention weight dampening from free energy.
pub const CROSS_MODAL_FE_DAMPEN_SCALE: f64 = 0.3;

/// Minimum cross-modal attention weight after dampening.
pub const CROSS_MODAL_FE_DAMPEN_MIN: f32 = 0.5;

/// Self-reflection confidence threshold for applying recommendations.
pub const REFLECTION_CONFIDENCE_THRESHOLD: f32 = 0.5;

/// Self-reflection LR decrease factor.
pub const REFLECTION_LR_DECREASE: f32 = 0.9;

/// Self-reflection LR increase factor.
pub const REFLECTION_LR_INCREASE: f32 = 1.1;

/// Self-reflection exploration increase delta.
pub const REFLECTION_EXPLORATION_INCREASE: f32 = 0.12;

/// Self-reflection exploration decrease scale.
pub const REFLECTION_EXPLORATION_DECREASE: f32 = 0.75;
