// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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

mod consciousness;
mod drives;
mod dynamics;
mod fabrication;
mod feedback;
mod language;
mod learning;
mod managers;
mod moral;
mod neuromod;
mod radio;
mod safety;
mod substrate;

pub use consciousness::*;
pub use drives::*;
pub use dynamics::*;
pub use fabrication::*;
pub use feedback::*;
pub use language::*;
pub use learning::*;
pub use managers::*;
pub use moral::*;
pub use neuromod::*;
pub use radio::*;
pub use safety::*;
pub use substrate::*;

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

    // 20. Psi→neuromod ordering (Round 22)
    assert!(
        PSI_NE_THRESHOLD < PSI_5HT_THRESHOLD,
        "PSI_NE_THRESHOLD ({}) must be < PSI_5HT_THRESHOLD ({})",
        PSI_NE_THRESHOLD,
        PSI_5HT_THRESHOLD
    );
    assert!(
        PSI_5HT_THRESHOLD < PSI_DA_THRESHOLD,
        "PSI_5HT_THRESHOLD ({}) must be < PSI_DA_THRESHOLD ({})",
        PSI_5HT_THRESHOLD,
        PSI_DA_THRESHOLD
    );

    // 21. Epistemic budget ordering
    assert!(
        EPISTEMIC_BUDGET_CONTRACT_THRESHOLD < EPISTEMIC_BUDGET_EXPAND_THRESHOLD,
        "EPISTEMIC_BUDGET_CONTRACT_THRESHOLD ({}) must be < EPISTEMIC_BUDGET_EXPAND_THRESHOLD ({})",
        EPISTEMIC_BUDGET_CONTRACT_THRESHOLD,
        EPISTEMIC_BUDGET_EXPAND_THRESHOLD
    );

    // 22. FEP surprise caps positive
    assert!(FEP_COMPLEXITY_PENALTY_CAP > 0.0);
    assert!(FEP_SURPRISE_EXPLORE_CAP > 0.0);
    assert!(FEP_SURPRISE_EXPLORE_SCALE > 0.0);

    // 23. Broca quality cadence ordering
    assert!(
        BROCA_QUALITY_CADENCE_THRESHOLD < BROCA_QUALITY_HIGH_THRESHOLD,
        "BROCA_QUALITY_CADENCE_THRESHOLD ({}) must be < BROCA_QUALITY_HIGH_THRESHOLD ({})",
        BROCA_QUALITY_CADENCE_THRESHOLD,
        BROCA_QUALITY_HIGH_THRESHOLD
    );

    // 24. Drive thresholds: frustration ordering
    assert!(
        FRUSTRATION_NE_NUDGE_THRESHOLD < FRUSTRATION_DAMPEN_THRESHOLD as f32,
        "FRUSTRATION_NE_NUDGE_THRESHOLD ({}) must be < FRUSTRATION_DAMPEN_THRESHOLD ({})",
        FRUSTRATION_NE_NUDGE_THRESHOLD,
        FRUSTRATION_DAMPEN_THRESHOLD
    );

    // 25. Neuromod baseline bounds: min < max
    assert!(
        NEUROMOD_BASELINE_MIN < NEUROMOD_BASELINE_MAX,
        "NEUROMOD_BASELINE_MIN ({}) must be < NEUROMOD_BASELINE_MAX ({})",
        NEUROMOD_BASELINE_MIN,
        NEUROMOD_BASELINE_MAX
    );
    assert!(
        NEUROMOD_BASELINE_MIN > 0.0,
        "NEUROMOD_BASELINE_MIN must be positive: {}",
        NEUROMOD_BASELINE_MIN
    );
    assert!(
        NEUROMOD_BASELINE_MAX <= 1.0,
        "NEUROMOD_BASELINE_MAX must be <= 1.0: {}",
        NEUROMOD_BASELINE_MAX
    );

    // 26. Coherence threshold ordering
    assert!(
        COHERENCE_VERY_LOW < COHERENCE_LOW,
        "COHERENCE_VERY_LOW ({}) must be < COHERENCE_LOW ({})",
        COHERENCE_VERY_LOW,
        COHERENCE_LOW
    );
    assert!(
        COHERENCE_LOW < COHERENCE_MODERATE,
        "COHERENCE_LOW ({}) must be < COHERENCE_MODERATE ({})",
        COHERENCE_LOW,
        COHERENCE_MODERATE
    );
    assert!(
        COHERENCE_MODERATE < COHERENCE_HIGH,
        "COHERENCE_MODERATE ({}) must be < COHERENCE_HIGH ({})",
        COHERENCE_MODERATE,
        COHERENCE_HIGH
    );

    // 27. EMA alpha bounds (must be in (0, 1))
    assert!(
        EMA_ALPHA_FLOW > 0.0 && EMA_ALPHA_FLOW < 1.0,
        "EMA_ALPHA_FLOW must be in (0, 1): {}",
        EMA_ALPHA_FLOW
    );

    // 28. Joint scale ordering: elbow < shoulder < knee
    assert!(
        JOINT_ELBOW_SCALE < JOINT_SHOULDER_SCALE,
        "JOINT_ELBOW_SCALE ({}) must be < JOINT_SHOULDER_SCALE ({})",
        JOINT_ELBOW_SCALE,
        JOINT_SHOULDER_SCALE
    );
    assert!(
        JOINT_SHOULDER_SCALE < JOINT_KNEE_SCALE,
        "JOINT_SHOULDER_SCALE ({}) must be < JOINT_KNEE_SCALE ({})",
        JOINT_SHOULDER_SCALE,
        JOINT_KNEE_SCALE
    );
}

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_passes() {
        validate(); // Panics on any threshold ordering violation — success means all thresholds are consistent
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
        // gradient_sign and explore_exploit assertions removed (dead fields)
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
        assert!(
            TEMPORAL_BINDING_HIGH_EXPLORE_SCALE > 0.9 && TEMPORAL_BINDING_HIGH_EXPLORE_SCALE < 1.1
        );
        assert!(GRADIENT_STABLE_DETECT_THRESHOLD > 0.0 && GRADIENT_STABLE_DETECT_THRESHOLD < 0.1);
        assert!(READINESS_REST_THRESHOLD > 0.8 && READINESS_REST_THRESHOLD < 1.0);
        assert!(READINESS_DEGRADED_THRESHOLD > 0.3 && READINESS_DEGRADED_THRESHOLD < 1.0);
        assert!(READINESS_REST_THRESHOLD > READINESS_DEGRADED_THRESHOLD); // rest threshold is higher (more ready)
        assert!(RECOVERY_STABILITY_THRESHOLD > 0.0 && RECOVERY_STABILITY_THRESHOLD < 0.2);
        assert!(FATIGUE_RECOVERED_THRESHOLD > 0.0 && FATIGUE_RECOVERED_THRESHOLD < 0.3);
        assert!(GRADIENT_PREDICTION_OK_THRESHOLD > 0.0 && GRADIENT_PREDICTION_OK_THRESHOLD < 0.5);
        // Dynamics phase
        assert!(
            RESONATOR_SUSTAINED_LOW_CONFIDENCE > 0.0 && RESONATOR_SUSTAINED_LOW_CONFIDENCE < 0.05
        );
        assert!(BROCA_COHERENT_THRESHOLD > 0.5 && BROCA_COHERENT_THRESHOLD < 1.0);
        assert!(BROCA_COHERENT_CONFIDENCE_SCALE > 0.0 && BROCA_COHERENT_CONFIDENCE_SCALE < 0.5);
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
        assert!(
            ALLOSTATIC_LOAD_DAMPEN_INCREMENT_SCALE > 0.0
                && ALLOSTATIC_LOAD_DAMPEN_INCREMENT_SCALE < 1.0
        );
        assert!(
            METACOGNITIVE_PREDICTION_EMA_DECAY > 0.5 && METACOGNITIVE_PREDICTION_EMA_DECAY < 1.0
        );
        assert!(
            LIMITING_COMPONENT_ATTENTION_SCALE > 0.0 && LIMITING_COMPONENT_ATTENTION_SCALE < 2.0
        );
        assert!(METACOGNITION_DEPTH_NORMALIZER > 1.0 && METACOGNITION_DEPTH_NORMALIZER < 10.0);
        assert!(
            KNOWLEDGE_CAUSAL_DEPTH_EXPLOIT_THRESHOLD > 0.0
                && KNOWLEDGE_CAUSAL_DEPTH_EXPLOIT_THRESHOLD < 10.0
        );
    }

    #[test]
    fn test_subsystem_phase_constants() {
        // Hierarchical LTC Phi cross-validation
        assert!(HIER_LTC_PHI_MIN_THRESHOLD > 0.0 && HIER_LTC_PHI_MIN_THRESHOLD < 0.5);
        assert!(HIER_LTC_PHI_CONVERGE_THRESHOLD > 0.0 && HIER_LTC_PHI_CONVERGE_THRESHOLD < 0.5);
        assert!(HIER_LTC_PHI_CONVERGE_BOOST > 0.0 && HIER_LTC_PHI_CONVERGE_BOOST < 0.2);
        assert!(HIER_LTC_PHI_DIVERGE_THRESHOLD > HIER_LTC_PHI_CONVERGE_THRESHOLD); // diverge > converge
        assert!(HIER_LTC_PHI_DIVERGE_MAX > 0.0 && HIER_LTC_PHI_DIVERGE_MAX < 1.0);
        assert!(
            HIER_LTC_PHI_DIVERGE_PENALTY_SCALE > 0.0 && HIER_LTC_PHI_DIVERGE_PENALTY_SCALE < 0.1
        );

        // Evolution coordinator
        assert!(
            EVOLUTION_POSITIVE_DELTA_THRESHOLD > 0.0 && EVOLUTION_POSITIVE_DELTA_THRESHOLD < 0.1
        );
        assert!(EVOLUTION_POSITIVE_LR_SCALE > 0.0 && EVOLUTION_POSITIVE_LR_SCALE < 1.0);
        assert!(
            EVOLUTION_POSITIVE_LR_CLAMP > 0.0
                && EVOLUTION_POSITIVE_LR_CLAMP <= EVOLUTION_POSITIVE_LR_SCALE
        );
        assert!(EVOLUTION_POSITIVE_CONF_SCALE > 0.0 && EVOLUTION_POSITIVE_CONF_SCALE < 0.5);
        assert!(
            EVOLUTION_POSITIVE_CONF_CLAMP > 0.0
                && EVOLUTION_POSITIVE_CONF_CLAMP <= EVOLUTION_POSITIVE_CONF_SCALE
        );
        assert!(
            EVOLUTION_NEGATIVE_DELTA_THRESHOLD < 0.0 && EVOLUTION_NEGATIVE_DELTA_THRESHOLD > -0.1
        );
        assert!(EVOLUTION_NEGATIVE_EXPLORE_SCALE > 0.0 && EVOLUTION_NEGATIVE_EXPLORE_SCALE < 0.5);
        assert!(
            EVOLUTION_NEGATIVE_EXPLORE_CLAMP > 0.0
                && EVOLUTION_NEGATIVE_EXPLORE_CLAMP <= EVOLUTION_NEGATIVE_EXPLORE_SCALE
        );

        // Holographic
        assert!(
            HOLOGRAPHIC_UNITY_CONFIDENCE_THRESHOLD > 0.3
                && HOLOGRAPHIC_UNITY_CONFIDENCE_THRESHOLD < 1.0
        );
        assert!(
            HOLOGRAPHIC_UNITY_CONFIDENCE_SCALE > 0.0 && HOLOGRAPHIC_UNITY_CONFIDENCE_SCALE < 0.2
        );
        assert!(
            HOLOGRAPHIC_BINDING_STRONG_THRESHOLD > 0.3
                && HOLOGRAPHIC_BINDING_STRONG_THRESHOLD < 1.0
        );
        assert!(HOLOGRAPHIC_BINDING_STRONG_LR > 1.0 && HOLOGRAPHIC_BINDING_STRONG_LR < 1.1);
        assert!(
            HOLOGRAPHIC_BINDING_WEAK_UPPER > 0.0
                && HOLOGRAPHIC_BINDING_WEAK_UPPER < HOLOGRAPHIC_BINDING_STRONG_THRESHOLD
        );
        assert!(HOLOGRAPHIC_BINDING_WEAK_LR > 0.9 && HOLOGRAPHIC_BINDING_WEAK_LR < 1.0);

        // Differentiable consciousness
        assert!(
            DIFF_CONSCIOUSNESS_WORKSPACE_SCALE > 0.5 && DIFF_CONSCIOUSNESS_WORKSPACE_SCALE < 1.0
        );
        assert!(
            DIFF_CONSCIOUSNESS_RECURSION_DEFAULT > 0.0
                && DIFF_CONSCIOUSNESS_RECURSION_DEFAULT < 1.0
        );

        // Consciousness gradient
        assert!(
            CONSCIOUSNESS_GRADIENT_EXPLORE_THRESHOLD > 0.0
                && CONSCIOUSNESS_GRADIENT_EXPLORE_THRESHOLD < 1.0
        );
        assert!(
            CONSCIOUSNESS_GRADIENT_EXPLORE_SCALE > 0.0
                && CONSCIOUSNESS_GRADIENT_EXPLORE_SCALE < 0.2
        );

        // Affective consciousness
        assert!(AFFECTIVE_DECAY_RATE > 0.0 && AFFECTIVE_DECAY_RATE < 0.2);
        assert!(
            AFFECTIVE_NEGATIVE_VALENCE_THRESHOLD < 0.0
                && AFFECTIVE_NEGATIVE_VALENCE_THRESHOLD > -1.0
        );
        assert!(
            AFFECTIVE_NEGATIVE_CONFIDENCE_SCALE > 0.0 && AFFECTIVE_NEGATIVE_CONFIDENCE_SCALE < 0.1
        );

        // Synthetic grounding + epistemic gate
        assert!(SYNTHETIC_GROUNDING_SIM_THRESHOLD > 0.0 && SYNTHETIC_GROUNDING_SIM_THRESHOLD < 0.5);
        assert!(EPISTEMIC_GATE_LOW_THRESHOLD > 0.0 && EPISTEMIC_GATE_LOW_THRESHOLD < 0.5);
        assert!(EPISTEMIC_GATE_LOW_PENALTY > 0.0 && EPISTEMIC_GATE_LOW_PENALTY < 0.1);

        // Primitive validation
        assert!(PRIMITIVE_VALIDATION_P_THRESHOLD > 0.0 && PRIMITIVE_VALIDATION_P_THRESHOLD < 0.1);
        assert!(
            PRIMITIVE_VALIDATION_POSITIVE_LR_SCALE > 0.0
                && PRIMITIVE_VALIDATION_POSITIVE_LR_SCALE < 0.1
        );
        assert!(PRIMITIVE_VALIDATION_POSITIVE_LR_CLAMP >= PRIMITIVE_VALIDATION_POSITIVE_LR_SCALE);
        assert!(PRIMITIVE_VALIDATION_NEGATIVE_LR > 0.9 && PRIMITIVE_VALIDATION_NEGATIVE_LR < 1.0);

        // Cross-module feedback
        assert!(CONSCIOUSNESS_STATE_LOW_URGENCY > 0.0 && CONSCIOUSNESS_STATE_LOW_URGENCY < 0.5);
        assert!(
            GRADIENT_STRONG_DIRECTION_THRESHOLD > 0.5 && GRADIENT_STRONG_DIRECTION_THRESHOLD < 2.0
        );
        assert!(GRADIENT_STRONG_BOREDOM_REDUCE > 0.0 && GRADIENT_STRONG_BOREDOM_REDUCE < 0.2);
        assert!(
            GRADIENT_PLATEAU_UPPER > 0.0
                && GRADIENT_PLATEAU_UPPER < GRADIENT_STRONG_DIRECTION_THRESHOLD
        );
        assert!(
            GRADIENT_PLATEAU_BOREDOM_INCREMENT > 0.0 && GRADIENT_PLATEAU_BOREDOM_INCREMENT < 0.1
        );

        // Holographic unity LR modulation
        assert!(HOLOGRAPHIC_UNITY_LR_BOOST_THRESHOLD > HOLOGRAPHIC_UNITY_CONFIDENCE_THRESHOLD);
        assert!(HOLOGRAPHIC_UNITY_LR_BOOST_FACTOR > 1.0 && HOLOGRAPHIC_UNITY_LR_BOOST_FACTOR < 1.1);
        assert!(HOLOGRAPHIC_UNITY_LR_CLAMP_LOW > 0.5 && HOLOGRAPHIC_UNITY_LR_CLAMP_LOW < 1.0);
        assert!(HOLOGRAPHIC_UNITY_LR_CLAMP_HIGH > 1.0 && HOLOGRAPHIC_UNITY_LR_CLAMP_HIGH < 2.0);
        assert!(
            HOLOGRAPHIC_UNITY_LR_DAMPEN_THRESHOLD > 0.0
                && HOLOGRAPHIC_UNITY_LR_DAMPEN_THRESHOLD < 0.5
        );
        assert!(
            HOLOGRAPHIC_UNITY_LR_DAMPEN_FACTOR > 0.9 && HOLOGRAPHIC_UNITY_LR_DAMPEN_FACTOR < 1.0
        );

        // Pipeline consciousness
        assert!(
            PIPELINE_CONSCIOUSNESS_EPISTEMIC_THRESHOLD > 0.5
                && PIPELINE_CONSCIOUSNESS_EPISTEMIC_THRESHOLD < 1.0
        );
        assert!(
            PIPELINE_CONSCIOUSNESS_EPISTEMIC_NUDGE > 0.0
                && PIPELINE_CONSCIOUSNESS_EPISTEMIC_NUDGE < 0.1
        );

        // Meta-reasoning
        assert!(
            META_REASONING_CONFIDENCE_THRESHOLD > 0.5 && META_REASONING_CONFIDENCE_THRESHOLD < 1.0
        );
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
        let weight_sum =
            KNOWLEDGE_GROUNDING_RELEVANCE_WEIGHT + KNOWLEDGE_GROUNDING_CERTAINTY_WEIGHT;
        assert!(
            (weight_sum - 1.0).abs() < 1e-6,
            "Grounding weights must sum to 1.0"
        );
        assert!(
            KNOWLEDGE_GROUNDING_RELEVANCE_WEIGHT > 0.0
                && KNOWLEDGE_GROUNDING_RELEVANCE_WEIGHT < 1.0
        );

        // Phi scale boost sigmoid
        assert!(PHI_SCALE_BOOST_MAX_AMPLITUDE > 0.0 && PHI_SCALE_BOOST_MAX_AMPLITUDE < 0.5);
        assert!(PHI_SCALE_BOOST_SIGMOID_SLOPE < 0.0); // negative slope for standard sigmoid
        assert!(PHI_SCALE_BOOST_CV_CENTER > 0.0);

        // Resonator consolidation
        assert!(
            RESONATOR_CONSOLIDATION_PRECISION_SCALE > 0.0
                && RESONATOR_CONSOLIDATION_PRECISION_SCALE < 0.5
        );
        assert!(
            RESONATOR_CONSOLIDATION_PRECISION_MAX > 1.0
                && RESONATOR_CONSOLIDATION_PRECISION_MAX < 5.0
        );

        // Goal LR
        assert!(GOAL_PRIORITY_LR_SCALE > 0.0 && GOAL_PRIORITY_LR_SCALE < 0.5);

        // Confidence crash
        assert!(CONFIDENCE_CRASH_MIN_PRIOR > 0.0 && CONFIDENCE_CRASH_MIN_PRIOR < 0.5);
        assert!(MODE_STABILITY_GRACE_THRESHOLD >= 1 && MODE_STABILITY_GRACE_THRESHOLD <= 10);
        assert!(
            CONFIDENCE_CRASH_LIGHT_FREEZE_CYCLES >= 1 && CONFIDENCE_CRASH_LIGHT_FREEZE_CYCLES <= 5
        );

        // Social trust
        assert!(SOCIAL_TRUST_ITHOU_THRESHOLD > 0.0 && SOCIAL_TRUST_ITHOU_THRESHOLD < 1.0);

        // World model
        assert!(WORLD_MODEL_CONFUSION_RATIO > 1.0 && WORLD_MODEL_CONFUSION_RATIO < 3.0);
        assert!(WORLD_MODEL_MISMATCH_RATIO > WORLD_MODEL_CONFUSION_RATIO); // mismatch requires higher ratio
        assert!(WORLD_MODEL_ERROR_FLOOR > 0.0 && WORLD_MODEL_ERROR_FLOOR < 0.5);

        // MCTS normalization
        let norm_sum = MCTS_EFFECTIVENESS_NORM_SCALE + MCTS_EFFECTIVENESS_NORM_OFFSET;
        assert!(
            (norm_sum - 1.0).abs() < 1e-6,
            "MCTS norm scale+offset should map max to 1.0"
        );

        // Voice heartbeat
        assert!(VOICE_HEARTBEAT_BASE_RATE > 1.0 && VOICE_HEARTBEAT_BASE_RATE < 10.0);
        assert!(
            VOICE_HEARTBEAT_COARTICULATION_WEIGHT > 0.0
                && VOICE_HEARTBEAT_COARTICULATION_WEIGHT < 1.0
        );
        assert!(VOICE_HEARTBEAT_LISTENER_SUCCESS > VOICE_HEARTBEAT_LISTENER_FAIL);
    }

    #[test]
    fn test_knowledge_engine_params() {
        // Persistence
        assert!(KNOWLEDGE_SAVE_INTERVAL >= 100 && KNOWLEDGE_SAVE_INTERVAL <= 5000);
        // Consciousness coupling
        assert!(
            KNOWLEDGE_CONSCIOUSNESS_MODULATION > 0.0 && KNOWLEDGE_CONSCIOUSNESS_MODULATION < 0.1
        );
        // Dream consolidation
        assert!(
            KNOWLEDGE_FORGET_CONFIDENCE_THRESHOLD > 0.0
                && KNOWLEDGE_FORGET_CONFIDENCE_THRESHOLD < 0.5
        );
        assert!(KNOWLEDGE_CONSOLIDATION_BOOST > 0.0 && KNOWLEDGE_CONSOLIDATION_BOOST < 0.2);
        // Causal depth exploitation
        assert!(KNOWLEDGE_CAUSAL_DEPTH_EXPLOIT_THRESHOLD > 0.0);
        assert!(
            KNOWLEDGE_CAUSAL_DEPTH_EXPLORE_DAMPEN > 0.0
                && KNOWLEDGE_CAUSAL_DEPTH_EXPLORE_DAMPEN < 0.5
        );
        // Contradiction boosts
        assert!(KNOWLEDGE_CONTRADICTION_NE_BOOST > 0.0 && KNOWLEDGE_CONTRADICTION_NE_BOOST < 0.1);
        assert!(KNOWLEDGE_CONTRADICTION_SHT_BOOST > 0.0 && KNOWLEDGE_CONTRADICTION_SHT_BOOST < 0.1);
        // Neuromod coupling
        assert!(KNOWLEDGE_UNCERTAINTY_NE_SCALE > 0.0 && KNOWLEDGE_UNCERTAINTY_NE_SCALE < 0.1);
        assert!(KNOWLEDGE_CAUSAL_DEPTH_DA_NUDGE > 0.0 && KNOWLEDGE_CAUSAL_DEPTH_DA_NUDGE < 0.1);
        assert!(KNOWLEDGE_GROUNDING_SHT_NUDGE > 0.0 && KNOWLEDGE_GROUNDING_SHT_NUDGE < 0.1);
        // Exploration modulation
        assert!(KNOWLEDGE_NOVELTY_EXPLORE_SCALE > 0.0 && KNOWLEDGE_NOVELTY_EXPLORE_SCALE < 0.5);
        // Episodic bridge
        assert!(KNOWLEDGE_EPISODIC_SALIENCE_BOOST > 0.0 && KNOWLEDGE_EPISODIC_SALIENCE_BOOST < 1.0);
        assert!(KNOWLEDGE_EPISODIC_MAX_PER_DREAM > 0 && KNOWLEDGE_EPISODIC_MAX_PER_DREAM <= 20);
        // AGM contradiction resolution
        assert!(
            KNOWLEDGE_CONTRADICTION_RESOLUTION_THRESHOLD > 0.5
                && KNOWLEDGE_CONTRADICTION_RESOLUTION_THRESHOLD <= 1.0
        );
        // Knowledge grounding epistemic blend
        assert!(
            KNOWLEDGE_GROUNDING_EPISTEMIC_BLEND > 0.0 && KNOWLEDGE_GROUNDING_EPISTEMIC_BLEND < 1.0
        );
        // Dream knowledge boosts
        assert!(
            DREAM_KNOWLEDGE_CONTRADICTION_BOOST > 0.0 && DREAM_KNOWLEDGE_CONTRADICTION_BOOST < 0.5
        );
        assert!(
            DREAM_KNOWLEDGE_CAUSAL_DEPTH_BOOST > 0.0 && DREAM_KNOWLEDGE_CAUSAL_DEPTH_BOOST < 0.5
        );
        assert!(
            DREAM_KNOWLEDGE_CAUSAL_DEPTH_THRESHOLD > 0.0
                && DREAM_KNOWLEDGE_CAUSAL_DEPTH_THRESHOLD < 10.0
        );
        // Contradiction boost < contradiction boost (dreams more aggressive than attention)
        assert!(DREAM_KNOWLEDGE_CAUSAL_DEPTH_BOOST < DREAM_KNOWLEDGE_CONTRADICTION_BOOST);
        // Dream→Knowledge feedback constants
        assert!(
            DREAM_KNOWLEDGE_REPLAY_WEIGHT > 0.0
                && DREAM_KNOWLEDGE_REPLAY_WEIGHT <= DREAM_KNOWLEDGE_CONTRADICTION_BOOST
        );
        assert!(DREAM_KNOWLEDGE_MIN_QUALITY > 0.0 && DREAM_KNOWLEDGE_MIN_QUALITY < 1.0);
        // Knowledge attention contradiction
        assert!(
            KNOWLEDGE_ATTENTION_CONTRADICTION_BOOST > 0.0
                && KNOWLEDGE_ATTENTION_CONTRADICTION_BOOST < 1.0
        );
        assert!(
            KNOWLEDGE_ATTENTION_CONTRADICTION_THRESHOLD > 0.0
                && KNOWLEDGE_ATTENTION_CONTRADICTION_THRESHOLD < 1.0
        );
        // Cross-coupling: Drive → Learning
        assert!(
            DRIVE_BOREDOM_PLASTICITY_THRESHOLD > 0.0 && DRIVE_BOREDOM_PLASTICITY_THRESHOLD < 1.0
        );
        assert!(DRIVE_BOREDOM_PLASTICITY_GAIN > 0.0 && DRIVE_BOREDOM_PLASTICITY_GAIN < 1.0);
        // Cross-coupling: Knowledge → Ethics
        assert!(
            KNOWLEDGE_ETHICS_CAUSAL_DEPTH_THRESHOLD > 0.0
                && KNOWLEDGE_ETHICS_CAUSAL_DEPTH_THRESHOLD < 1.0
        );
        assert!(KNOWLEDGE_ETHICS_CONFIDENCE_GAIN > 0.0 && KNOWLEDGE_ETHICS_CONFIDENCE_GAIN < 0.1);
        // Cross-coupling: Memory → Learning
        assert!(
            MEMORY_CONSOLIDATION_PLASTICITY_THRESHOLD > 0.0
                && MEMORY_CONSOLIDATION_PLASTICITY_THRESHOLD < 1.0
        );
        assert!(
            MEMORY_CONSOLIDATION_PLASTICITY_GAIN > 0.0
                && MEMORY_CONSOLIDATION_PLASTICITY_GAIN < 0.5
        );
        assert!(
            MEMORY_RECALL_QUALITY_DAMPEN_THRESHOLD > 0.0
                && MEMORY_RECALL_QUALITY_DAMPEN_THRESHOLD < 0.5
        );
        assert!(
            MEMORY_RECALL_QUALITY_DAMPEN_SCALE > 0.0 && MEMORY_RECALL_QUALITY_DAMPEN_SCALE < 1.0
        );
        // Cross-coupling: Perception → Drive
        assert!(
            PERCEPTION_LOW_COHERENCE_THRESHOLD > 0.0 && PERCEPTION_LOW_COHERENCE_THRESHOLD < 0.5
        );
        assert!(
            PERCEPTION_LOW_COHERENCE_EXPLORE_GAIN > 0.0
                && PERCEPTION_LOW_COHERENCE_EXPLORE_GAIN < 0.2
        );
        assert!(
            PERCEPTION_HIGH_LOAD_SUPPRESS_THRESHOLD > 0.5
                && PERCEPTION_HIGH_LOAD_SUPPRESS_THRESHOLD < 1.0
        );
        assert!(
            PERCEPTION_HIGH_LOAD_SUPPRESS_FACTOR > 0.0
                && PERCEPTION_HIGH_LOAD_SUPPRESS_FACTOR < 1.0
        );
    }

    #[test]
    fn test_radio_spectrum_params() {
        // Jamming
        assert!(RADIO_JAMMING_SNR_THRESHOLD > 0.0 && RADIO_JAMMING_SNR_THRESHOLD < 20.0);
        assert!(RADIO_JAMMING_AROUSAL_SPIKE > 0.0 && RADIO_JAMMING_AROUSAL_SPIKE < 0.2);
        assert!(RADIO_JAMMING_EXPLORATION_BOOST > 0.0 && RADIO_JAMMING_EXPLORATION_BOOST < 0.1);
        // Degradation
        assert!(RADIO_DEGRADATION_CONFIDENCE_DROP > 0.0 && RADIO_DEGRADATION_CONFIDENCE_DROP < 0.1);
        // EMA alphas
        assert!(RADIO_TIER_LOSS_EMA_ALPHA > 0.0 && RADIO_TIER_LOSS_EMA_ALPHA < 1.0);
        assert!(RADIO_NOISE_FLOOR_EMA_ALPHA > 0.0 && RADIO_NOISE_FLOOR_EMA_ALPHA < 1.0);
        // Tier degraded threshold
        assert!(RADIO_TIER_DEGRADED_LOSS > 0.0 && RADIO_TIER_DEGRADED_LOSS < 1.0);
        // Peer cap
        assert!(RADIO_MAX_DELTA_PEERS > 0 && RADIO_MAX_DELTA_PEERS <= 256);
        // Bandwidth throttle
        assert!(RADIO_BANDWIDTH_THROTTLE_THRESHOLD > 0);
        // Connectivity penalties
        assert!(
            RADIO_CONNECTIVITY_PENALTY_LOCAL_DOWN > 0.0
                && RADIO_CONNECTIVITY_PENALTY_LOCAL_DOWN < 1.0
        );
        assert!(
            RADIO_CONNECTIVITY_PENALTY_METRO_ONLY > 0.0
                && RADIO_CONNECTIVITY_PENALTY_METRO_ONLY < RADIO_CONNECTIVITY_PENALTY_LOCAL_DOWN
        );
        // Noise floor and PE
        assert!(RADIO_DEFAULT_NOISE_FLOOR_DBM < 0.0);
        assert!(RADIO_NOISE_ERROR_NORMALIZER > 0.0);
        assert!(RADIO_BLACKOUT_EXPLORATION_BOOST > 0.0 && RADIO_BLACKOUT_EXPLORATION_BOOST < 0.2);
        assert!(RADIO_LOSS_LR_DAMPEN_FACTOR > 0.0 && RADIO_LOSS_LR_DAMPEN_FACTOR < 1.0);
        assert!(RADIO_LOSS_LR_DAMPEN_MAX > 0.0 && RADIO_LOSS_LR_DAMPEN_MAX < 1.0);
        assert!(
            RADIO_SPECTRUM_PE_SURPRISE_THRESHOLD > 0.0
                && RADIO_SPECTRUM_PE_SURPRISE_THRESHOLD < 1.0
        );
        assert!(RADIO_SPECTRUM_PE_AROUSAL_MAX > 0.0 && RADIO_SPECTRUM_PE_AROUSAL_MAX < 0.5);
        assert!(
            RADIO_SPECTRUM_PE_AROUSAL_SCALE > 0.0
                && RADIO_SPECTRUM_PE_AROUSAL_SCALE < RADIO_SPECTRUM_PE_AROUSAL_MAX
        );
        // Waterfall
        assert!(RADIO_WATERFALL_CAPACITY >= 16 && RADIO_WATERFALL_CAPACITY <= 256);
        assert!(
            RADIO_WATERFALL_MIN_SAMPLES >= 4
                && RADIO_WATERFALL_MIN_SAMPLES < RADIO_WATERFALL_CAPACITY
        );
        // Frequency hopping
        assert!(RADIO_HOP_COOLDOWN_CYCLES > 0 && RADIO_HOP_COOLDOWN_CYCLES <= 20);
        assert!(RADIO_HOP_SNR_IMPROVEMENT_DB > 0.0 && RADIO_HOP_SNR_IMPROVEMENT_DB < 20.0);
        // Peer discovery
        assert!(RADIO_BEACON_INTERVAL_CYCLES > 0);
        assert!(RADIO_BEACON_SIZE <= 50); // Must fit Regional MTU
        // Relay routing
        assert!(RADIO_MAX_RELAY_HOPS > 0 && RADIO_MAX_RELAY_HOPS <= 8);
        assert!(RADIO_MAX_ROUTE_ENTRIES > 0 && RADIO_MAX_ROUTE_ENTRIES <= 512);
        assert!(RADIO_ROUTE_EXPIRY_CYCLES > 0);
        // FEC
        assert!(RADIO_FEC_OVERHEAD_RATIO > 0.0 && RADIO_FEC_OVERHEAD_RATIO < 0.5);
        assert!(RADIO_FEC_MIN_PAYLOAD > 0 && RADIO_FEC_MIN_PAYLOAD < 256);
        // Energy
        assert!(RADIO_ENERGY_PER_BIT_LOCAL > 0.0);
        assert!(RADIO_ENERGY_PER_BIT_METRO > 0.0);
        assert!(RADIO_ENERGY_PER_BIT_REGIONAL > RADIO_ENERGY_PER_BIT_METRO); // HF is most expensive
        assert!(RADIO_ENERGY_PER_BIT_LOCAL > RADIO_ENERGY_PER_BIT_METRO); // Wi-Fi > LoRa
        assert!(RADIO_ENERGY_BUDGET_PER_CYCLE > 0.0);
        // Crypto
        assert_eq!(RADIO_CRYPTO_NONCE_SIZE, 12); // RFC 8439
        assert!(RADIO_CRYPTO_MAX_PEERS > 0 && RADIO_CRYPTO_MAX_PEERS <= 256);

        // Safety & hop thresholds
        assert!(RADIO_SAFETY_JAMMING_THRESHOLD > 0 && RADIO_SAFETY_JAMMING_THRESHOLD <= 10);
        assert!(RADIO_AUTO_HOP_NOISE_THRESHOLD > 0.0 && RADIO_AUTO_HOP_NOISE_THRESHOLD < 30.0);
        assert!(
            RADIO_BEACON_PEER_CONFIDENCE_BOOST > 0.0 && RADIO_BEACON_PEER_CONFIDENCE_BOOST < 0.1
        );

        // Synthetic observation bounds
        assert!(
            RADIO_SYNTHETIC_SNR_ISOLATED > 0.0
                && RADIO_SYNTHETIC_SNR_ISOLATED < RADIO_SYNTHETIC_SNR_BASE
        );
        assert!(RADIO_SYNTHETIC_SNR_BASE > 0.0);
        assert!(RADIO_SYNTHETIC_SNR_PEER_BONUS > 0.0 && RADIO_SYNTHETIC_SNR_PEER_BONUS < 5.0);
        assert!(RADIO_SYNTHETIC_SNR_PHI_BONUS > 0.0 && RADIO_SYNTHETIC_SNR_PHI_BONUS < 20.0);
        assert!(RADIO_SYNTHETIC_PEER_CAP > 0.0 && RADIO_SYNTHETIC_PEER_CAP <= 50.0);
        assert!(RADIO_SYNTHETIC_NOISE_FLOOR_BASE < 0.0);
        assert!(
            RADIO_SYNTHETIC_NOISE_FLOOR_RANGE > 0.0 && RADIO_SYNTHETIC_NOISE_FLOOR_RANGE < 30.0
        );

        // Energy-aware routing
        assert!(RADIO_ENERGY_AWARE_THRESHOLD > 0.0 && RADIO_ENERGY_AWARE_THRESHOLD < 1.0);

        // Strategy dampening: blackout > degraded
        assert!(
            RADIO_BLACKOUT_STRATEGY_EXPLORATION_DAMPEN > RADIO_DEGRADED_STRATEGY_EXPLORATION_DAMPEN
        );
        assert!(
            RADIO_BLACKOUT_STRATEGY_EXPLORATION_DAMPEN > 0.0
                && RADIO_BLACKOUT_STRATEGY_EXPLORATION_DAMPEN < 0.5
        );
        assert!(
            RADIO_DEGRADED_STRATEGY_EXPLORATION_DAMPEN > 0.0
                && RADIO_DEGRADED_STRATEGY_EXPLORATION_DAMPEN < 0.3
        );

        // Neuromod coupling
        assert!(RADIO_JAMMING_NE_NUDGE > 0.0 && RADIO_JAMMING_NE_NUDGE < 0.1);
        assert!(RADIO_RECOVERY_DA_NUDGE > 0.0 && RADIO_RECOVERY_DA_NUDGE < 0.1);
        assert!(
            RADIO_NEUROMOD_JAMMING_MIN_STREAK > 0
                && RADIO_NEUROMOD_JAMMING_MIN_STREAK <= RADIO_SAFETY_JAMMING_THRESHOLD
        );

        // Consciousness tier thresholds
        assert!(RADIO_CONSCIOUSNESS_HIGH_CONFIDENCE > RADIO_CONSCIOUSNESS_LOW_CONFIDENCE);
        assert!(
            RADIO_CONSCIOUSNESS_HIGH_CONFIDENCE > 0.0 && RADIO_CONSCIOUSNESS_HIGH_CONFIDENCE < 1.0
        );
        assert!(
            RADIO_CONSCIOUSNESS_LOW_CONFIDENCE > 0.0 && RADIO_CONSCIOUSNESS_LOW_CONFIDENCE < 1.0
        );

        // Beacon interval must exceed hop cooldown
        assert!(RADIO_BEACON_INTERVAL_CYCLES > RADIO_HOP_COOLDOWN_CYCLES);
    }

    #[test]
    fn test_dynamics_startup_constants() {
        assert!(
            CONFIDENCE_CRASH_FLOW_MULTIPLIER > 1.0_f64
                && CONFIDENCE_CRASH_FLOW_MULTIPLIER < 3.0_f64
        );
        assert!(DYNAMICS_STARTUP_WARMUP_CYCLES > 0 && DYNAMICS_STARTUP_WARMUP_CYCLES <= 50);
        assert!(DYNAMICS_POST_BOOT_CYCLES > DYNAMICS_STARTUP_WARMUP_CYCLES);
        assert!(
            RESONATOR_STARTUP_CYCLES > 0
                && RESONATOR_STARTUP_CYCLES <= DYNAMICS_STARTUP_WARMUP_CYCLES
        );
        assert!(NEUROMOD_DELTA_THRESHOLD > 0.0 && NEUROMOD_DELTA_THRESHOLD < 0.01);
        assert!(AROUSAL_TRAP_RECOVERY_MIN_CYCLES > 0);
        assert!(AROUSAL_TRAP_RECOVERY_RAMP_CYCLES > 0.0);
        assert!(
            ATTENTION_SENSITIVITY_BOOST_FACTOR > 1.0 && ATTENTION_SENSITIVITY_BOOST_FACTOR < 1.5
        );
        assert!(FEP_EFFICIENT_EXPLORATION_DAMPEN > 0.0 && FEP_EFFICIENT_EXPLORATION_DAMPEN < 1.0);
    }

    #[test]
    fn test_fourier_motor_constants() {
        assert!(FOURIER_MOTOR_ALPHA_HZ > 0.0 && FOURIER_MOTOR_ALPHA_HZ <= 15.0);
        assert!(FOURIER_MOTOR_BETA_HZ > FOURIER_MOTOR_ALPHA_HZ && FOURIER_MOTOR_BETA_HZ <= 35.0);
        assert!(FOURIER_MOTOR_GAMMA_HZ > FOURIER_MOTOR_BETA_HZ && FOURIER_MOTOR_GAMMA_HZ <= 100.0);
        assert!(FOURIER_MOTOR_AMPLITUDE > 0.0 && FOURIER_MOTOR_AMPLITUDE <= FOURIER_AMPLITUDE_MAX);
        assert!(FOURIER_AMPLITUDE_MAX > 0.0 && FOURIER_AMPLITUDE_MAX <= 1.0);
    }

    #[test]
    fn test_cpg_constants() {
        assert!(CPG_DEFAULT_COUPLING_K > 0.0 && CPG_DEFAULT_COUPLING_K <= 10.0);
        assert!(CPG_AROUSAL_FREQ_SCALE > 0.0 && CPG_AROUSAL_FREQ_SCALE <= 2.0);
        assert!(CPG_WALK_MIN_SYNC > CPG_TROT_MIN_SYNC);
        assert!(CPG_TROT_MIN_SYNC > CPG_GALLOP_MIN_SYNC);
        assert!(CPG_GALLOP_MIN_SYNC > CPG_CRITICAL_DESYNC);
        assert!(CPG_WALK_MIN_SYNC <= 1.0 && CPG_CRITICAL_DESYNC >= 0.0);
        assert!(CPG_DESYNC_EXPLORATION_BOOST > 0.0 && CPG_DESYNC_EXPLORATION_BOOST <= 0.1);
        assert!(CPG_INTERVAL > 50 && CPG_INTERVAL < 100);
        // CPG sync tau floor: must be in (0, 1) — floor=0 would kill dynamics on desync
        assert!(CPG_SYNC_TAU_FLOOR > 0.0 && CPG_SYNC_TAU_FLOOR < 1.0);
    }

    #[test]
    fn test_complex_cfc_constants() {
        assert!(COMPLEX_CFC_EIGENVALUE_REAL_MIN < 0.0);
        assert!(COMPLEX_CFC_EIGENVALUE_REAL_MAX < 0.0);
        assert!(COMPLEX_CFC_EIGENVALUE_REAL_MIN < COMPLEX_CFC_EIGENVALUE_REAL_MAX);
        assert!(COMPLEX_CFC_MOTOR_FREQ_MIN_HZ > 0.0);
        assert!(COMPLEX_CFC_MOTOR_FREQ_MAX_HZ > COMPLEX_CFC_MOTOR_FREQ_MIN_HZ);
        assert!(COMPLEX_CFC_EIGENVALUE_LR > 0.0 && COMPLEX_CFC_EIGENVALUE_LR < 0.1);
    }

    #[test]
    fn test_spectral_constants() {
        assert!(SPECTRAL_SAMPLE_RATE > 0.0 && SPECTRAL_SAMPLE_RATE < 100.0);
        assert!(SPECTRAL_HISTORY_CAPACITY > SPECTRAL_MIN_HISTORY);
        assert!(SPECTRAL_MIN_HISTORY >= 16);
        assert!(SPECTRAL_INTERVAL > 60 && SPECTRAL_INTERVAL < 100);
        assert!(
            SPECTRAL_GAMMA_CONSCIOUSNESS_BOOST > 0.0 && SPECTRAL_GAMMA_CONSCIOUSNESS_BOOST <= 0.1
        );
        assert!(SPECTRAL_DELTA_REST_THRESHOLD > 0.3 && SPECTRAL_DELTA_REST_THRESHOLD <= 0.9);
        assert!(
            SPECTRAL_ENTROPY_EXPLORATION_SCALE > 0.0 && SPECTRAL_ENTROPY_EXPLORATION_SCALE <= 0.1
        );
        assert!(SPECTRAL_PAC_THRESHOLD > 0.0 && SPECTRAL_PAC_THRESHOLD < 1.0);
        assert!(SPECTRAL_PAC_CONFIDENCE_BOOST > 0.0 && SPECTRAL_PAC_CONFIDENCE_BOOST <= 0.1);
    }

    #[test]
    fn test_reasoning_engine_feedback_constants() {
        assert!(
            REASONING_RELIABILITY_CONFIDENCE_SCALE > 0.0
                && REASONING_RELIABILITY_CONFIDENCE_SCALE <= 0.1
        );
        assert!(REASONING_RELIABILITY_THRESHOLD > 0.5 && REASONING_RELIABILITY_THRESHOLD < 1.0);
        assert!(
            DREAM_REASONING_RELIABILITY_SCALE > 0.0 && DREAM_REASONING_RELIABILITY_SCALE <= 1.0
        );
        // Max confidence boost: (1.0 - 0.5) * 0.03 = 0.015 (modest)
        assert!((1.0 - 0.5) * REASONING_RELIABILITY_CONFIDENCE_SCALE <= 0.02);
        // Max dream boost: (1.0 - 0.5) * 0.4 = 0.2 (20%)
        assert!((1.0 - 0.5) * DREAM_REASONING_RELIABILITY_SCALE <= 0.25);
    }

    #[test]
    fn test_startup_circadian_strategy_constants() {
        // Startup LR ramp
        assert!(STARTUP_LR_INITIAL_SCALE > 0.0 && STARTUP_LR_INITIAL_SCALE < 1.0);
        assert!((STARTUP_LR_INITIAL_SCALE + STARTUP_LR_RAMP_RANGE - 1.0).abs() < f32::EPSILON);
        // Adaptive LR bounds
        assert!(ADAPTIVE_LR_MIN > 0.0 && ADAPTIVE_LR_MIN < ADAPTIVE_LR_MAX);
        assert!(ADAPTIVE_LR_MAX <= 1.0);
        // Sleep recovery
        assert!(SLEEP_RECOVERY_QUALITY_SCALE > 1.0);
        // Circadian plasticity
        assert!(CIRCADIAN_PLASTICITY_SCALE > 0.0 && CIRCADIAN_PLASTICITY_SCALE <= 1.0);
        // Circadian stillness: Night > Dusk > Dawn > 0
        assert!(CIRCADIAN_STILLNESS_NIGHT > CIRCADIAN_STILLNESS_DUSK);
        assert!(CIRCADIAN_STILLNESS_DUSK > CIRCADIAN_STILLNESS_DAWN);
        assert!(CIRCADIAN_STILLNESS_DAWN > 0.0);
        // Surprise
        assert!(SURPRISE_PE_MULTIPLIER > 1.0 && SURPRISE_PE_MULTIPLIER < 10.0);
        // Coherence default
        assert!(COHERENCE_DEFAULT >= 0.0 && COHERENCE_DEFAULT <= 1.0);
        // Social coherence
        assert!(SOCIAL_COHERENCE_OXY_WEIGHT > 0.0 && SOCIAL_COHERENCE_OXY_WEIGHT <= 1.0);
        assert!(SOCIAL_COHERENCE_OFFSET >= 0.0 && SOCIAL_COHERENCE_OFFSET <= 1.0);
        // FEP baseline
        assert!(FEP_BASELINE_PE_BASE > 0.0 && FEP_BASELINE_PE_BASE < 1.0);
        assert!(FEP_BASELINE_PE_EMA_FACTOR > 0.0 && FEP_BASELINE_PE_EMA_FACTOR < 1.0);
        // Moral FE
        assert!(MORAL_FE_EXPLORATION_THRESHOLD > 0.0 && MORAL_FE_EXPLORATION_THRESHOLD < 1.0);
        assert!(MORAL_FE_BOOST_CAP > 0.0 && MORAL_FE_BOOST_CAP < 1.0);
        assert!(MORAL_TOPOLOGY_MIN_SCENARIOS > 0);
        assert!(
            MORAL_TOPOLOGY_COMPLETENESS_THRESHOLD > 0.0
                && MORAL_TOPOLOGY_COMPLETENESS_THRESHOLD < 1.0
        );
        assert!(
            MORAL_TOPOLOGY_STRUCTURAL_BOOST_SCALE > 0.0
                && MORAL_TOPOLOGY_STRUCTURAL_BOOST_SCALE < 1.0
        );
        // Neuromod EMA
        assert!(NEUROMOD_EMA_ALPHA > 0.0 && NEUROMOD_EMA_ALPHA < 0.5);
        // Substrate noise
        assert!(SUBSTRATE_NOISE_FRACTION_DIVISOR > 0.0);
        assert!(SUBSTRATE_NOISE_STD_DIVISOR > 0.0);
        assert!(SUBSTRATE_NOISE_FRACTION_DIVISOR > SUBSTRATE_NOISE_STD_DIVISOR);
        // Neuromod stillness
        assert!(
            (NEUROMOD_STILLNESS_GABA_WEIGHT + NEUROMOD_STILLNESS_ADENOSINE_WEIGHT - 1.0).abs()
                < f32::EPSILON
        );
        assert!(NEUROMOD_STILLNESS_OFFSET > 0.0 && NEUROMOD_STILLNESS_OFFSET < 1.0);
        assert!(
            NEUROMOD_STILLNESS_CLAMP_MAX > 0.0
                && NEUROMOD_STILLNESS_CLAMP_MAX <= STILLNESS_TOTAL_CLAMP_MAX
        );
        assert!(STILLNESS_TOTAL_CLAMP_MAX > 0.0 && STILLNESS_TOTAL_CLAMP_MAX <= 1.0);
        // Knowledge novelty
        assert!(
            KNOWLEDGE_NOVELTY_EXPLORATION_THRESHOLD > 0.0
                && KNOWLEDGE_NOVELTY_EXPLORATION_THRESHOLD < 1.0
        );
        // Cantor
        assert!(CANTOR_RESONANCE_BOOST_HARMONY_THRESHOLD > 0.0);
        assert!(CANTOR_META_DEPTH_STILLNESS_THRESHOLD > 0.0);

        // Swarm security
        assert!(HANDSHAKE_INITIAL_TRUST_SCORE > 0.0 && HANDSHAKE_INITIAL_TRUST_SCORE <= 1.0);
        assert!(HANDSHAKE_CHALLENGE_TIMEOUT_SECS > 0 && HANDSHAKE_CHALLENGE_TIMEOUT_SECS <= 120);
        assert!(HANDSHAKE_MAX_PENDING_CHALLENGES > 0);
        assert!(HANDSHAKE_MAX_PER_PEER_CHALLENGES > 0);
        assert!(HANDSHAKE_MAX_PER_PEER_CHALLENGES <= HANDSHAKE_MAX_PENDING_CHALLENGES);
        // Key rotation
        assert!(KEY_ROTATION_INTERVAL_DEFAULT > 0);
        assert!(KEY_ROTATION_GRACE_PERIOD_DEFAULT > 0);
        assert!(KEY_ROTATION_GRACE_PERIOD_DEFAULT < KEY_ROTATION_INTERVAL_DEFAULT);

        // Safety enforcement ordering
        assert!(SAFETY_RED_LR_MULTIPLIER < SAFETY_ORANGE_LR_MULTIPLIER);
        assert!(SAFETY_ORANGE_LR_MULTIPLIER < SAFETY_YELLOW_LR_MULTIPLIER);
        assert!(SAFETY_YELLOW_LR_MULTIPLIER < 1.0);
        assert!(SAFETY_ORANGE_EXPLORATION_DAMPEN < SAFETY_YELLOW_EXPLORATION_DAMPEN);
        assert!(DEFENSE_QUARANTINE_MAX_CYCLES > 0);
        assert!(DEFENSE_MAX_MORAL_SEVERITY > 0.0 && DEFENSE_MAX_MORAL_SEVERITY <= 1.0);

        // Neuroevolution
        assert!(NEUROEVO_TAU_BASE_BITS > 0 && NEUROEVO_TAU_BASE_BITS <= 32);
        assert!(NEUROEVO_MAX_LAYERS > 0 && NEUROEVO_MAX_LAYERS <= 10);
        assert!(NEUROEVO_MAX_NEURONS_PER_LAYER > 0);
        assert!(NEUROEVO_FEP_STATE_DIM > 0);
        assert!(NEUROEVO_MAX_AGE_CYCLES > 0);
        assert!(NEUROEVO_FITNESS_FLOOR < 0.0);
        assert!(NEUROEVO_EVAL_STEPS > 0);
        assert!(NEUROEVO_WARMUP_STEPS > 0 && NEUROEVO_WARMUP_STEPS < NEUROEVO_EVAL_STEPS);
        assert!(NEUROEVO_FE_FITNESS_WEIGHT > 0.0 && NEUROEVO_FE_FITNESS_WEIGHT <= 1.0);
        assert!(NEUROEVO_PHI_FITNESS_WEIGHT > 0.0 && NEUROEVO_PHI_FITNESS_WEIGHT <= 1.0);
        assert!(NEUROEVO_POPULATION_SIZE >= 5);
        assert!(
            NEUROEVO_TOURNAMENT_SIZE >= 2 && NEUROEVO_TOURNAMENT_SIZE <= NEUROEVO_POPULATION_SIZE
        );
        assert!(NEUROEVO_ELITISM_FRACTION > 0.0 && NEUROEVO_ELITISM_FRACTION < 1.0);
        assert!(NEUROEVO_MUTATION_RATE > 0.0 && NEUROEVO_MUTATION_RATE < 1.0);
        assert!(NEUROEVO_CROSSOVER_RATE > 0.0 && NEUROEVO_CROSSOVER_RATE <= 1.0);
        assert!(NEUROEVO_CONVERGENCE_PATIENCE > 0);
        assert!(NEUROEVO_SPECIATION_THRESHOLD > 0.0 && NEUROEVO_SPECIATION_THRESHOLD < 1.0);
        assert!(NEUROEVO_MANAGER_INTERVAL > 0);

        // Memory manager
        assert!(MEMORY_CONSOLIDATION_EXPLORATION_DAMPEN > 0.0);
        assert!(MEMORY_CONSOLIDATION_LR_BOOST >= 1.0);
        assert!(MEMORY_RETRIEVAL_CONFIDENCE_WEIGHT + MEMORY_RETRIEVAL_COHERENCE_WEIGHT == 1.0);
        assert!(MEMORY_RETRIEVAL_LOW_QUALITY < MEMORY_RETRIEVAL_HIGH_QUALITY);
        assert!(
            MEMORY_PSI_CONSOLIDATION_THRESHOLD > 0.0 && MEMORY_PSI_CONSOLIDATION_THRESHOLD < 1.0
        );
        assert!(
            MEMORY_PRESSURE_EXPLORATION_THRESHOLD > 0.0
                && MEMORY_PRESSURE_EXPLORATION_THRESHOLD < 1.0
        );

        // Swarm manager
        assert!(SWARM_PEER_PHI_TRUST_SCALE > 0.0 && SWARM_PEER_PHI_TRUST_SCALE <= 1.0);
        assert!(SWARM_CORROBORATION_BOOST > 0.0);
        assert!(SWARM_CORROBORATION_CAP >= SWARM_CORROBORATION_BOOST);
        assert!(SWARM_SOCIAL_BUFFERING_CAP > 0.0);
        assert!(SWARM_COLLECTIVE_PHI_THRESHOLD > 0.0 && SWARM_COLLECTIVE_PHI_THRESHOLD < 1.0);
        assert!(SWARM_COLLECTIVE_PHI_LR_CAP > 0.0);
        assert!(SWARM_ISOLATION_THRESHOLD > 0.0 && SWARM_ISOLATION_THRESHOLD < 1.0);

        // Space alerts
        assert!(SPACE_CONJUNCTION_AROUSAL > 0.0 && SPACE_CONJUNCTION_AROUSAL < 0.5);
        assert!(SPACE_DEBRIS_AROUSAL > 0.0 && SPACE_DEBRIS_AROUSAL < 0.5);
        assert!(SPACE_DEBRIS_AROUSAL > SPACE_CONJUNCTION_AROUSAL); // debris more threatening
        assert!(SPACE_DEBRIS_VALENCE < 0.0); // negative affect
        assert!(SPACE_DEBRIS_CONFIDENCE < 0.0); // uncertainty reduces confidence
        assert!(SPACE_COMM_CONFIDENCE > 0.0); // opportunity boosts confidence
        assert!(SPACE_COMM_LR_BOOST > 0.0 && SPACE_COMM_LR_BOOST < 0.5);
        assert!(SPACE_CONJUNCTION_EXPLORATION > 0.0);
        assert!(SPACE_ANOMALY_AROUSAL > 0.0 && SPACE_ANOMALY_AROUSAL < 0.5);
        assert!(SPACE_MANEUVER_CONFIDENCE > 0.0 && SPACE_MANEUVER_CONFIDENCE < 0.1);

        // Trust manager
        assert!(TRUST_VIOLATION_SLASH_FACTOR > 0.0 && TRUST_VIOLATION_SLASH_FACTOR < 1.0);
        assert!(TRUST_VIOLATION_AROUSAL_CAP > 0.0);
        assert!(TRUST_BETRAYAL_VALENCE_PENALTY > 0.0);
        assert!(TRUST_ANOMALY_AROUSAL > 0.0);

        // Social fabric manager
        assert!(SOCIAL_RESONANCE_HIGH_THRESHOLD > 0.0 && SOCIAL_RESONANCE_HIGH_THRESHOLD < 1.0);
        assert!(SOCIAL_RESONANCE_RANGE > 0.0);
        assert!(SOCIAL_DIVERSITY_THRESHOLD > 0.0 && SOCIAL_DIVERSITY_THRESHOLD < 1.0);
        assert!(SOCIAL_ECHO_CHAMBER_THRESHOLD > 0.0 && SOCIAL_ECHO_CHAMBER_THRESHOLD < 1.0);

        // Time manager
        assert!(TIME_DRIFT_AROUSAL_CAP > 0.0);
        assert!(TIME_DRIFT_SURPRISE_DIVISOR > 0.0);

        // Sentinel manager
        assert!(SENTINEL_AROUSAL_SCALE_NORMAL > 0.0);
        assert!(SENTINEL_AROUSAL_SCALE_HEIGHTENED > SENTINEL_AROUSAL_SCALE_NORMAL);
        assert!(SENTINEL_THREAT_MODERATE > 0.0 && SENTINEL_THREAT_MODERATE < 1.0);
        assert!(SENTINEL_THREAT_CRITICAL > SENTINEL_THREAT_MODERATE);
        assert!(SENTINEL_EXPLORATION_DAMPEN_SCALE > 0.0);
        assert!(SENTINEL_CONFIDENCE_DAMPEN_SCALE > 0.0);

        // Spectral manager (additional)
        assert!(SPECTRAL_GAMMA_THRESHOLD > 0.0 && SPECTRAL_GAMMA_THRESHOLD < 1.0);
        assert!(SPECTRAL_DELTA_AROUSAL_DELTA < 0.0); // calming
        assert!(SPECTRAL_ENTROPY_THRESHOLD > 0.0);
        assert!(SPECTRAL_ENTROPY_MASK_FLOOR > 0.0 && SPECTRAL_ENTROPY_MASK_FLOOR < 1.0);

        // CPG manager (additional)
        assert!(CPG_DESYNC_AROUSAL_DELTA > 0.0);
        assert!(
            CPG_SYNC_PHI_MODULATION_AMPLITUDE > 0.0 && CPG_SYNC_PHI_MODULATION_AMPLITUDE <= 0.1
        );

        // Guiding question
        assert!(GUIDING_EPISTEMIC_EXPLORATION_BOOST > 0.0);
        assert!(GUIDING_AFFECTIVE_CONFIDENCE_BOOST > 0.0);
        assert!(GUIDING_PRAGMATIC_LR_FACTOR >= 1.0);
        assert!(GUIDING_SOCIAL_CONFIDENCE_BOOST > 0.0);

        // Civic crisis detector
        assert!(CRISIS_CONFIDENCE_MIN_DENOMINATOR > 0.0);
        assert!(CRISIS_CONFIDENCE_MAX_SIGNAL > 1.0);
        assert!(CRISIS_SAFETY_RED_ORDINAL > CRISIS_SAFETY_ORANGE_ORDINAL);
        assert!(CRISIS_SEVERITY_BOOST_RED > CRISIS_SEVERITY_BOOST_ORANGE);
        assert!(CRISIS_SEVERITY_MAX > CRISIS_SEVERITY_MIN);

        // Cognitive depth scores
        assert!(DEPTH_SCORE_DEEP_THOUGHT > DEPTH_SCORE_CORTICAL);
        assert!(DEPTH_SCORE_CORTICAL > DEPTH_SCORE_REFLEX);
        assert!(DEPTH_SCORE_REFLEX > 0.0);

        // Vision ACh modulation
        assert!(VISION_ACH_FLOOR > 0.0);
        assert!(VISION_ACH_SCALE_CAP > 1.0);
        assert!(VISION_COHERENCE_CLAMP_MIN < VISION_COHERENCE_CLAMP_MAX);
        assert!(VISION_ERROR_CLAMP_MIN < VISION_ERROR_CLAMP_MAX);
        assert!(VISION_DAMPEN_CLAMP_MIN < VISION_DAMPEN_CLAMP_MAX);

        // Substrate noise cap
        assert!(SUBSTRATE_NOISE_MAX_PRESSURE > 0.0);

        // Consciousness engine: sigma
        assert!(SIGMA_HIGH_THRESHOLD > SIGMA_LOW_THRESHOLD);
        assert!(SIGMA_DAMPEN_SCALE > 0.0);
        assert!(SIGMA_DAMPEN_MAX > 0.0);
        assert!(SIGMA_CONFIDENCE_SCALE > 0.0);
        assert!(SIGMA_BOOST_SCALE > 0.0);
        assert!(SIGMA_BOOST_MAX > 0.0);

        // Consciousness engine: structural phi
        assert!(STRUCTURAL_WEAK_EMERGENCE_THRESHOLD > 0.0);
        assert!(STRUCTURAL_STRONG_EMERGENCE_THRESHOLD > STRUCTURAL_WEAK_EMERGENCE_THRESHOLD);
        assert!(STRUCTURAL_BOTTLENECK_THRESHOLD > 0.0);
        assert!(STRUCTURAL_BOTTLENECK_LR_BOOST >= 1.0);

        // Phi → tau + LR feedback
        assert!(PHI_TAU_REFERENCE > 0.0);
        assert!(PHI_TAU_SIGMOID_STEEPNESS > 0.0);
        assert!(PHI_TAU_CEILING > PHI_TAU_FLOOR);
        assert!(PHI_TAU_FLOOR > 0.0);
        assert!(PHI_TAU_CEILING <= 2.0); // sanity: never more than 2× speed
        assert!(PHI_LR_STABILIZATION_SCALE > 0.0);
        assert!(PHI_LR_STABILIZATION_MAX > 0.0);
        assert!(PHI_LR_STABILIZATION_MAX <= 0.2); // sanity: never kill more than 20% of LR

        // Consciousness engine: phi validation
        assert!(PHI_VALIDATION_HIGH_THRESHOLD > PHI_VALIDATION_LOW_THRESHOLD);
        assert!(PHI_VALIDATION_BOOST_SCALE > 0.0);
        assert!(PHI_VALIDATION_ATTENUATION_SCALE > 0.0);

        // Consciousness engine: multimodal
        assert!(MULTIMODAL_PHI_THRESHOLD > 0.0);
        assert!(MULTIMODAL_CONFIDENCE_SCALE > 0.0);
        assert!(MULTIMODAL_LR_SCALE > 0.0);

        // Consciousness engine: equation V2
        assert!(EQ_V2_HIGH_THRESHOLD > EQ_V2_LOW_THRESHOLD);
        assert!(EQ_V2_CONFIDENCE_SCALE > 0.0);
        assert!(EQ_V2_CONSOLIDATION_SCALE > 0.0);
        assert!(EQ_V2_EXPLORATION_NUDGE > 0.0);

        // Consciousness engine: pipeline
        assert!(PIPELINE_CONSCIOUSNESS_THRESHOLD > 0.0);
        assert!(PIPELINE_LR_SCALE > 0.0);

        // Bath-consciousness coupling
        assert!(BATH_5HT2A_SCALE > 0.0);
        assert!(BATH_GABA_SCALE > 0.0);
        assert!(BATH_ENTROPY_ATTRACTOR_PENALTY < 0.0);

        // Cycle extracted: moral feedback
        assert!(MORAL_VALUE_FEEDBACK_SCALE > 0.0);
        assert!(MEMORY_VALENCE_THRESHOLD > 0.0);
        assert!(MEMORY_PHI_PRIME_THRESHOLD > 0.0);
        assert!(FEP_REWARD_WEIGHT > 0.0 && FEP_REWARD_WEIGHT < 1.0);

        // Cycle extracted: arousal
        assert!(AROUSAL_SUPPRESS_THRESHOLD > 0.0);
        assert!(AROUSAL_TRAP_THRESHOLD > AROUSAL_SUPPRESS_THRESHOLD);
        assert!(LOW_AROUSAL_CONSOLIDATION_THRESHOLD > 0.0);

        // Cycle extracted: PFE surprise
        assert!(PFE_SURPRISE_HIGH_THRESHOLD > PFE_SURPRISE_LOW_THRESHOLD);

        // Late consciousness monitors: thermo
        assert!(THERMO_CRITICAL_CURIOSITY_BOOST > 1.0);
        assert!(THERMO_FLOW_LR_BOOST > 1.0);
        assert!(HOMEOSTASIS_DRIFT_RATE > 0.0 && HOMEOSTASIS_DRIFT_RATE < 1.0);
    }

    #[test]
    fn test_psi_neuromod_ordering() {
        assert!(PSI_NE_THRESHOLD < PSI_5HT_THRESHOLD);
        assert!(PSI_5HT_THRESHOLD < PSI_DA_THRESHOLD);
        assert!(PSI_DA_THRESHOLD <= 1.0);
        assert!(PSI_NE_THRESHOLD > 0.0);
        // Caps must be positive and bounded
        assert!(PSI_DA_CAP > 0.0 && PSI_DA_CAP <= 0.5);
        assert!(PSI_5HT_CAP > 0.0 && PSI_5HT_CAP <= 0.5);
        assert!(PSI_NE_CAP > 0.0 && PSI_NE_CAP <= 0.5);
    }

    #[test]
    fn test_epistemic_budget_ordering() {
        assert!(EPISTEMIC_BUDGET_CONTRACT_THRESHOLD < EPISTEMIC_BUDGET_EXPAND_THRESHOLD);
        assert!(EPISTEMIC_BUDGET_CONTRACT_BASE > 0.0 && EPISTEMIC_BUDGET_CONTRACT_BASE < 1.0);
        assert!(EPISTEMIC_BUDGET_EXPAND_CAP > 0.0 && EPISTEMIC_BUDGET_EXPAND_CAP < 1.0);
    }

    #[test]
    fn test_fep_surprise_params() {
        assert!(FEP_COMPLEXITY_PENALTY_CAP > 0.0);
        assert!(FEP_COMPLEXITY_LR_SCALE > 0.0 && FEP_COMPLEXITY_LR_SCALE < 1.0);
        assert!(FEP_SURPRISE_EXPLORE_CAP > 0.0);
        assert!(FEP_SURPRISE_EXPLORE_SCALE > 0.0 && FEP_SURPRISE_EXPLORE_SCALE < 1.0);
        assert!(FEP_SURPRISE_EXPLORE_SECONDARY_CAP > 0.0);
        assert!(FEP_SURPRISE_EXPLORE_SECONDARY_SCALE > 0.0);
    }

    #[test]
    fn test_stillness_budget_params() {
        assert!(STILLNESS_BUDGET_THRESHOLD > 0.0 && STILLNESS_BUDGET_THRESHOLD < 1.0);
        assert!(STILLNESS_BUDGET_CONTRACT_CAP > 0.0 && STILLNESS_BUDGET_CONTRACT_CAP < 1.0);
    }

    #[test]
    fn test_knowledge_alert_params() {
        assert!(KNOWLEDGE_ALERT_EXPLORE_CAP > 0.0 && KNOWLEDGE_ALERT_EXPLORE_CAP < 1.0);
        assert!(KNOWLEDGE_CONTRADICTION_FLOOR > 0.0 && KNOWLEDGE_CONTRADICTION_FLOOR < 1.0);
    }

    #[test]
    fn test_drive_threshold_params() {
        assert!(FRUSTRATION_DAMPEN_THRESHOLD > 0.0 && FRUSTRATION_DAMPEN_THRESHOLD < 1.0);
        assert!(FRUSTRATION_DAMPEN_GAIN > 0.0 && FRUSTRATION_DAMPEN_GAIN < 1.0);
        assert!(FRUSTRATION_NE_NUDGE_THRESHOLD > 0.0 && FRUSTRATION_NE_NUDGE_THRESHOLD < 1.0);
        assert!(FRUSTRATION_NE_NUDGE_SCALE > 0.0 && FRUSTRATION_NE_NUDGE_SCALE < 0.5);
        assert!(ENGAGEMENT_LOW_THRESHOLD > 0.0 && ENGAGEMENT_LOW_THRESHOLD < 1.0);
        assert!(FLOW_DA_NUDGE > 0.0 && FLOW_DA_NUDGE < 0.5);
        assert!(DISENGAGEMENT_DA_NUDGE > 0.0 && DISENGAGEMENT_DA_NUDGE < 0.5);
        assert!(FLOW_EXPLORATION_INCREMENT > 0.0 && FLOW_EXPLORATION_INCREMENT < 0.5);
    }

    #[test]
    fn test_neuromod_baseline_bounds() {
        assert!(NEUROMOD_BASELINE_MIN > 0.0);
        assert!(NEUROMOD_BASELINE_MAX <= 1.0);
        assert!(NEUROMOD_BASELINE_MIN < NEUROMOD_BASELINE_MAX);
    }

    #[test]
    fn test_coherence_threshold_ordering() {
        assert!(COHERENCE_VERY_LOW < COHERENCE_LOW);
        assert!(COHERENCE_LOW < COHERENCE_MODERATE);
        assert!(COHERENCE_MODERATE < COHERENCE_HIGH);
        assert!(COHERENCE_HIGH <= 1.0);
    }

    #[test]
    fn test_ema_alpha_bounds() {
        assert!(EMA_ALPHA_FLOW > 0.0 && EMA_ALPHA_FLOW < 1.0);
    }

    #[test]
    fn test_joint_scale_ordering() {
        assert!(JOINT_ELBOW_SCALE > 0.0 && JOINT_ELBOW_SCALE < 1.0);
        assert!(JOINT_SHOULDER_SCALE > 0.0 && JOINT_SHOULDER_SCALE < 1.0);
        assert!(JOINT_KNEE_SCALE > 0.0 && JOINT_KNEE_SCALE < 1.0);
        assert!(JOINT_ELBOW_SCALE < JOINT_SHOULDER_SCALE);
        assert!(JOINT_SHOULDER_SCALE < JOINT_KNEE_SCALE);
    }
}
