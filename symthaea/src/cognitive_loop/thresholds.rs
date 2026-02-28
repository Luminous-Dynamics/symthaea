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

/// Maximum LR boost from consciousness level (MCE) — up to +10%.
/// Basis: Dehaene (2014) — conscious access facilitates learning.
pub const MCE_LR_BOOST_SCALE: f32 = 0.1;

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

    // 8. Psi weights don't exceed 1.0 total
    let psi_total = FLOW_PSI_WEIGHT as f64
        + RELATIONAL_PSI_WEIGHT as f64
        + BODY_PSI_WEIGHT
        + EMBODIED_PSI_WEIGHT;
    assert!(
        psi_total <= 1.0,
        "Psi weights sum ({}) must be <= 1.0",
        psi_total
    );
}

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
}
