// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//
// Kani formal verification proofs for consciousness gating invariants.
// Run with: cargo kani --harness <name> -p mycelix-bridge-common

#[cfg(kani)]
mod kani_proofs {
    use crate::consciousness_profile::{
        continuous_vote_weight, ConsciousnessProfile, ConsciousnessTier,
    };

    // ========================================================================
    // Proof 1: combined_score is always in [0.0, 1.0]
    // ========================================================================
    #[kani::proof]
    fn proof_combined_score_bounded() {
        let profile = ConsciousnessProfile {
            identity: kani::any(),
            reputation: kani::any(),
            community: kani::any(),
            engagement: kani::any(),
        };
        let score = profile.combined_score();
        assert!(score >= 0.0, "combined_score must be >= 0.0");
        assert!(score <= 1.0, "combined_score must be <= 1.0");
    }

    // ========================================================================
    // Proof 2: from_score is monotonic — higher scores never produce lower tiers
    // ========================================================================
    #[kani::proof]
    fn proof_tier_monotonicity() {
        let a: f64 = kani::any();
        let b: f64 = kani::any();
        // Restrict to valid range to keep proof tractable
        kani::assume(a >= 0.0 && a <= 1.0);
        kani::assume(b >= 0.0 && b <= 1.0);
        kani::assume(a <= b);
        let tier_a = ConsciousnessTier::from_score(a);
        let tier_b = ConsciousnessTier::from_score(b);
        assert!(tier_a <= tier_b, "from_score must be monotonic");
    }

    // ========================================================================
    // Proof 3: vote_weight_bp is non-decreasing with tier ordering
    // ========================================================================
    #[kani::proof]
    fn proof_vote_weight_monotonic() {
        let tiers = [
            ConsciousnessTier::Observer,
            ConsciousnessTier::Participant,
            ConsciousnessTier::Citizen,
            ConsciousnessTier::Steward,
            ConsciousnessTier::Guardian,
        ];
        let i: usize = kani::any();
        let j: usize = kani::any();
        kani::assume(i < 5 && j < 5 && i <= j);
        assert!(
            tiers[i].vote_weight_bp() <= tiers[j].vote_weight_bp(),
            "vote weight must be non-decreasing with tier"
        );
    }

    // ========================================================================
    // Proof 4: degrade never increases tier
    // ========================================================================
    #[kani::proof]
    fn proof_degrade_never_increases() {
        let tiers = [
            ConsciousnessTier::Observer,
            ConsciousnessTier::Participant,
            ConsciousnessTier::Citizen,
            ConsciousnessTier::Steward,
            ConsciousnessTier::Guardian,
        ];
        let tier_idx: usize = kani::any();
        let levels: u32 = kani::any();
        kani::assume(tier_idx < 5);
        kani::assume(levels <= 10); // Bound for tractability
        let original = tiers[tier_idx];
        let degraded = original.degrade(levels);
        assert!(degraded <= original, "degrade must never increase tier");
    }

    // ========================================================================
    // Proof 5: continuous_vote_weight is always non-negative and finite
    // ========================================================================
    #[kani::proof]
    fn proof_vote_weight_nonnegative() {
        let score: f64 = kani::any();
        let threshold: f64 = kani::any();
        let temperature: f64 = kani::any();
        let max_weight: f64 = kani::any();
        let result = continuous_vote_weight(score, threshold, temperature, max_weight);
        assert!(result >= 0.0, "vote weight must be >= 0.0");
        assert!(result.is_finite(), "vote weight must be finite");
    }

    // ========================================================================
    // Proof 6: clamped() produces dimensions in [0.0, 1.0]
    // ========================================================================
    #[kani::proof]
    fn proof_clamped_bounded() {
        let profile = ConsciousnessProfile {
            identity: kani::any(),
            reputation: kani::any(),
            community: kani::any(),
            engagement: kani::any(),
        };
        let c = profile.clamped();
        assert!(c.identity >= 0.0 && c.identity <= 1.0);
        assert!(c.reputation >= 0.0 && c.reputation <= 1.0);
        assert!(c.community >= 0.0 && c.community <= 1.0);
        assert!(c.engagement >= 0.0 && c.engagement <= 1.0);
    }

    // ========================================================================
    // Proof 7: Observer has zero vote weight (no governance bypass)
    // ========================================================================
    #[kani::proof]
    fn proof_observer_zero_weight() {
        assert_eq!(
            ConsciousnessTier::Observer.vote_weight_bp(),
            0,
            "Observer must have zero vote weight"
        );
    }

    // ========================================================================
    // Proof 8: from_score → min_score roundtrip — tier's min_score
    // is always <= the score that produced it
    // ========================================================================
    #[kani::proof]
    fn proof_tier_min_score_consistent() {
        let score: f64 = kani::any();
        kani::assume(score >= 0.0 && score <= 1.0);
        let tier = ConsciousnessTier::from_score(score);
        assert!(
            tier.min_score() <= score,
            "tier min_score must be <= the score that produced it"
        );
    }
}
