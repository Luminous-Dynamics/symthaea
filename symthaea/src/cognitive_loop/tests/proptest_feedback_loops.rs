// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property tests for feedback loop wiring.
//!
//! Validates invariants of the MCE narrative/social/softmin, resonance flow,
//! and temporal smoothing feedback loops wired in the March 12-13 sessions.

use proptest::prelude::*;

use crate::cognitive_loop::thresholds::{
    MCE_NARRATIVE_HIGH_CONFIDENCE_BOOST, MCE_NARRATIVE_HIGH_THRESHOLD,
    MCE_NARRATIVE_LOW_EXPLORATION_BOOST, MCE_NARRATIVE_LOW_THRESHOLD, MCE_SOCIAL_LOW_LR_BOOST,
    MCE_SOCIAL_LOW_THRESHOLD, MCE_SOFTMIN_EXPLORATION_THRESHOLD, RESONANCE_FLOW_EXPLORATION_DAMPEN,
    RESONANCE_FLOW_LR_BOOST,
};

proptest! {
    // Property 1: MCE narrative feedback is mutually exclusive (low vs high).
    // The thresholds must not overlap: LOW_THRESHOLD < HIGH_THRESHOLD.
    #[test]
    fn prop_mce_narrative_mutual_exclusion(narrative in 0.0f64..=1.0) {
        let is_low = narrative > 0.0 && narrative < MCE_NARRATIVE_LOW_THRESHOLD as f64;
        let is_high = narrative > MCE_NARRATIVE_HIGH_THRESHOLD as f64;
        prop_assert!(
            !(is_low && is_high),
            "narrative {} triggered both low and high branches",
            narrative
        );
    }

    // Property 2: MCE narrative exploration boost is always non-negative and bounded.
    #[test]
    fn prop_mce_narrative_exploration_bounded(narrative in 0.0f64..=1.0) {
        if narrative > 0.0 && narrative < MCE_NARRATIVE_LOW_THRESHOLD as f64 {
            let boost = MCE_NARRATIVE_LOW_EXPLORATION_BOOST;
            prop_assert!(boost >= 0.0 && boost <= 0.1,
                "narrative low exploration boost out of [0, 0.1]: {}", boost);
        }
    }

    // Property 3: MCE narrative confidence boost is always non-negative and small.
    #[test]
    fn prop_mce_narrative_confidence_bounded(narrative in 0.0f64..=1.0) {
        if narrative > MCE_NARRATIVE_HIGH_THRESHOLD as f64 {
            let boost = MCE_NARRATIVE_HIGH_CONFIDENCE_BOOST;
            prop_assert!(boost >= 0.0 && boost <= 0.05,
                "narrative high confidence boost out of [0, 0.05]: {}", boost);
        }
    }

    // Property 4: MCE social LR boost is > 1.0 (it's a scale factor, not additive).
    #[test]
    fn prop_mce_social_lr_is_scale_factor(social in 0.0f64..=1.0) {
        if social > 0.0 && social < MCE_SOCIAL_LOW_THRESHOLD as f64 {
            prop_assert!(MCE_SOCIAL_LOW_LR_BOOST > 1.0,
                "social LR boost should be > 1.0: {}", MCE_SOCIAL_LOW_LR_BOOST);
            prop_assert!(MCE_SOCIAL_LOW_LR_BOOST < 1.2,
                "social LR boost too aggressive: {}", MCE_SOCIAL_LOW_LR_BOOST);
        }
    }

    // Property 5: MCE softmin exploration boost is proportional and bounded.
    // deficit = threshold - softmin; boost = deficit * 0.05
    // Max deficit = threshold (when softmin=0+ε), max boost = threshold * 0.05
    #[test]
    fn prop_mce_softmin_exploration_proportional(softmin in 0.001f64..=1.0) {
        if softmin < MCE_SOFTMIN_EXPLORATION_THRESHOLD as f64 {
            let deficit = MCE_SOFTMIN_EXPLORATION_THRESHOLD as f64 - softmin;
            let boost = (deficit * 0.05) as f32;
            prop_assert!(boost >= 0.0,
                "softmin exploration boost negative: {}", boost);
            let max_boost = MCE_SOFTMIN_EXPLORATION_THRESHOLD * 0.05;
            prop_assert!(boost <= max_boost + 1e-6,
                "softmin exploration boost {} > max {}", boost, max_boost);
        }
    }

    // Property 6: Resonance flow dampening keeps exploration in [0, 1].
    // scale_exploration multiplies by DAMPEN factor (< 1.0).
    #[test]
    fn prop_resonance_flow_exploration_dampen_bounded(exploration in 0.0f32..=1.0) {
        let dampened = exploration * RESONANCE_FLOW_EXPLORATION_DAMPEN;
        prop_assert!(dampened >= 0.0,
            "dampened exploration negative: {}", dampened);
        prop_assert!(dampened <= exploration + 1e-6,
            "dampened exploration {} > original {}", dampened, exploration);
    }

    // Property 7: Resonance flow LR boost is moderate (< 10% increase).
    #[test]
    fn prop_resonance_flow_lr_boost_moderate(lr in 0.01f32..=5.0) {
        let boosted = lr * RESONANCE_FLOW_LR_BOOST;
        let increase_pct = (boosted - lr) / lr;
        prop_assert!(increase_pct >= 0.0,
            "LR boost decreased LR: {} -> {}", lr, boosted);
        prop_assert!(increase_pct < 0.1,
            "LR boost too aggressive: {:.1}% increase", increase_pct * 100.0);
    }

    // Property 8: Temporal smoothing preserves boundedness.
    // smoothed = raw * (1 - alpha) + prev * alpha
    // If both raw and prev are in [0, 1], smoothed must be in [0, 1].
    #[test]
    fn prop_temporal_smoothing_preserves_bounds(
        raw in 0.0f64..=1.0,
        prev in 0.0f64..=1.0,
        alpha in 0.05f64..=0.5,
    ) {
        let smoothed = raw * (1.0 - alpha) + prev * alpha;
        prop_assert!(smoothed >= -1e-10,
            "smoothed below 0: {} (raw={}, prev={}, alpha={})", smoothed, raw, prev, alpha);
        prop_assert!(smoothed <= 1.0 + 1e-10,
            "smoothed above 1: {} (raw={}, prev={}, alpha={})", smoothed, raw, prev, alpha);
    }

    // Property 9: Full-cycle stability with random cycle counts.
    // The system must remain numerically stable regardless of how many cycles run.
    #[test]
    fn prop_full_cycle_finite_outputs(cycle_count in 5u32..=50) {
        let mut svc = crate::cognitive_loop::CognitiveLoopService::new(
            crate::cognitive_loop::CognitiveLoopConfig::default(),
        ).unwrap();
        for i in 0..cycle_count {
            let result = svc.cycle("proptest stability");
            prop_assert!(
                result.prediction_error.is_finite(),
                "PE non-finite at cycle {}", i
            );
            prop_assert!(
                result.metadata.actual_effective_lr.is_finite()
                    && result.metadata.actual_effective_lr >= 0.0,
                "LR invalid at cycle {}: {}", i, result.metadata.actual_effective_lr
            );
            prop_assert!(
                result.metadata.consciousness.consciousness_level.is_finite(),
                "consciousness non-finite at cycle {}", i
            );
        }
    }
}