// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration test: Phase 2a — 8D Sovereign Profile gating.
//!
//! Verifies that the simulator's eligible-voter fraction under 8D gating
//! responds to profile asymmetry in the way production Mycelix does:
//! a population that is uniformly-high on all 8 axes has more eligible
//! voters than one that is high on 7 axes but low on `EpistemicIntegrity`,
//! because the voting requirement imposes a per-dimension minimum on EI.

use mycelix_multiworld_sim::sovereign_profile::{
    civic_requirement_voting, CivicTier, DimensionWeights, SovereignProfile,
};

#[test]
fn asymmetric_profile_reduces_voting_eligibility() {
    let w = DimensionWeights::governance();
    let voting = civic_requirement_voting();

    // Population A: uniform 0.5 on all axes → Citizen tier, EI=0.5 ≥ 0.25 → eligible.
    let uniform = SovereignProfile::from_array([0.5; 8]);
    assert!(uniform.meets_requirement(&voting, &w));
    assert_eq!(uniform.tier(&w), CivicTier::Citizen);

    // Population B: 0.7 on everything except EpistemicIntegrity (0.15).
    // Tier from combined score is still Steward-level, but the 0.25 EI floor
    // in civic_requirement_voting disqualifies them from voting.
    let asymmetric = SovereignProfile {
        epistemic_integrity: 0.15,
        thermodynamic_yield: 0.70,
        network_resilience: 0.70,
        economic_velocity: 0.70,
        civic_participation: 0.70,
        stewardship_care: 0.70,
        semantic_resonance: 0.70,
        domain_competence: 0.70,
    };
    assert!(asymmetric.tier(&w) >= CivicTier::Citizen);
    assert!(!asymmetric.meets_requirement(&voting, &w));
}

#[test]
fn weights_preset_shifts_tier_for_same_profile() {
    // Solar-engineer profile: high thermo + network, low civic.
    let engineer = SovereignProfile {
        epistemic_integrity: 0.35,
        thermodynamic_yield: 0.85,
        network_resilience: 0.85,
        economic_velocity: 0.40,
        civic_participation: 0.15,
        stewardship_care: 0.30,
        semantic_resonance: 0.25,
        domain_competence: 0.55,
    };

    let energy = DimensionWeights::energy_cooperative();
    let default = DimensionWeights::governance();

    // Under energy cooperative weights, the engineer scores higher
    // than under the default governance weights.
    assert!(
        engineer.combined_score(&energy) > engineer.combined_score(&default),
        "engineer score under energy weights {:.3} should exceed governance weights {:.3}",
        engineer.combined_score(&energy),
        engineer.combined_score(&default),
    );
}

#[test]
fn monte_carlo_populations_produce_reasonable_tier_distributions() {
    use mycelix_multiworld_sim::stochastic::StochasticEngine;

    let w = DimensionWeights::governance();
    let mut rng = StochasticEngine::new(1234);
    let mut tier_counts = [0usize; 5];
    let n = 1000;
    for _ in 0..n {
        let p = SovereignProfile::sample(0.5, &mut rng);
        tier_counts[p.tier(&w).index()] += 1;
    }
    // With Normal(μ≈0.28, σ=0.18) per-dim and governance weights, expect a
    // skew toward Observer/Participant — but at least a handful of Citizens
    // should emerge. (Guardians are rare at this baseline.)
    let total: usize = tier_counts.iter().sum();
    assert_eq!(total, n);
    assert!(tier_counts[0] > 0, "no Observers generated");
    // Participant + Citizen + Steward cumulative should be at least 10%.
    let active: usize = tier_counts[1] + tier_counts[2] + tier_counts[3];
    assert!(
        active >= n / 10,
        "too few active tiers in MC sample: {:?}",
        tier_counts,
    );
}
