// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Property tests for feedback consensus edge cases.
//!
//! Validates that the FeedbackState / ProposalCollector correctly handles
//! empty channels, uniform proposals, and oscillating inputs.

use proptest::prelude::*;

proptest! {
    // Property 1: Empty feedback channel produces neutral consensus.
    // When no proposals are submitted, effective values equal cycle-start values.
    #[test]
    fn prop_empty_feedback_returns_cycle_start(
        start_conf in 0.01f64..=0.99,
        start_lr in 1.0f64..=3.0,
        start_explore in 0.0f64..=1.0,
    ) {
        use crate::cognitive_loop::feedback_state::FeedbackState;
        let mut fs = FeedbackState::default();
        fs.begin_cycle();
        fs.snapshot_cycle_start(start_conf, start_lr, start_explore, 1.0);
        // No proposals — effective values must equal cycle-start values.
        let eff_conf = fs.effective_confidence();
        let eff_lr = fs.effective_lr_boost();
        let eff_explore = fs.effective_exploration();
        prop_assert!(
            (eff_conf - start_conf).abs() < 1e-10,
            "empty confidence {} != start {}", eff_conf, start_conf
        );
        prop_assert!(
            (eff_lr - start_lr).abs() < 1e-10,
            "empty LR {} != start {}", eff_lr, start_lr
        );
        prop_assert!(
            (eff_explore - start_explore).abs() < 1e-10,
            "empty exploration {} != start {}", eff_explore, start_explore
        );
    }

    // Property 2: Uniform proposals produce stable consensus.
    // ProposalCollector uses weighted-average of additive deltas, so N identical
    // proposals with equal weight collapse to a single delta (idempotent).
    #[test]
    fn prop_uniform_proposals_stable(
        start in 0.3f64..=0.7,
        delta in -0.1f64..=0.1,
        count in 1usize..=10,
    ) {
        use crate::cognitive_loop::feedback_state::{FeedbackState, FeedbackProposal};
        let mut fs = FeedbackState::default();
        fs.begin_cycle();
        fs.snapshot_cycle_start(start, 1.5, 0.5, 1.0);
        for _ in 0..count {
            fs.confidence.propose("test", FeedbackProposal::Add(delta));
        }
        let eff = fs.effective_confidence();
        // Weighted average of N identical deltas = delta (idempotent).
        let expected = (start + delta).clamp(0.01, 0.99);
        prop_assert!(
            (eff - expected).abs() < 1e-6,
            "uniform confidence {} != expected {} (start={}, delta={}, count={})",
            eff, expected, start, delta, count
        );
    }

    // Property 3: Oscillating feedback remains bounded after many cycles.
    // Alternating +delta/-delta proposals over 100 iterations must stay in [0, 1].
    #[test]
    fn prop_oscillating_feedback_bounded(
        start in 0.2f64..=0.8,
        amplitude in 0.01f64..=0.2,
    ) {
        use crate::cognitive_loop::feedback_state::{FeedbackState, FeedbackProposal};
        let mut fs = FeedbackState::default();
        let mut current = start;
        for i in 0..100u32 {
            fs.begin_cycle();
            fs.snapshot_cycle_start(current, 1.5, 0.5, 1.0);
            let delta = if i % 2 == 0 { amplitude } else { -amplitude };
            fs.confidence.propose("oscillator", FeedbackProposal::Add(delta));
            current = fs.effective_confidence();
            prop_assert!(
                current >= 0.0 && current <= 1.0,
                "confidence out of bounds {} at iteration {} (start={}, amp={})",
                current, i, start, amplitude
            );
            prop_assert!(
                current.is_finite(),
                "confidence NaN/Inf at iteration {}", i
            );
        }
    }
}
