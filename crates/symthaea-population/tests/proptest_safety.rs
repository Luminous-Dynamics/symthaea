// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Safety-invariant property tests for symthaea-population.
//!
//! Tests that governance and ethics safety bounds hold:
//! - GovernanceTier thresholds are strictly monotonically increasing
//! - Ethics score evaluate_decision_ethics is always in [0, 1]
//! - ethical_balance is always in [0, 1]
//! - Kinship symmetry: kinship(a, b) == kinship(b, a)

use proptest::prelude::*;
use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};
use symthaea_population::{
    BiologicalSex, BreedingStrategy, GeneticRescuePlan, GovernanceTier, Individual, MatingPair,
    Population, PopulationDecision, ethical_balance, evaluate_decision_ethics,
};

fn make_population(n: usize) -> Population {
    let individuals: Vec<Individual> = (0..n)
        .map(|i| Individual {
            id: i as u64,
            sex: if i % 2 == 0 {
                BiologicalSex::Female
            } else {
                BiologicalSex::Male
            },
            genotypes: vec![],
            genome_hv: ContinuousHV::random(HDC_DIMENSION, i as u64 * 137),
            generation: 0,
            parent_ids: (None, None),
            fitness: 1.0,
        })
        .collect();

    Population {
        individuals,
        generation: 0,
        founding_size: n,
        diversity_hv: ContinuousHV::zero(HDC_DIMENSION),
    }
}

fn arb_decision() -> impl Strategy<Value = PopulationDecision> {
    prop_oneof![
        (0_u64..1000, 0_u64..1000).prop_map(|(a, b)| {
            PopulationDecision::ApprovePairing(MatingPair {
                parent_a: a,
                parent_b: b,
                kinship: 0.05,
                hla_complementarity: 0.8,
            })
        }),
        (1_usize..1000, 1_u32..100).prop_map(|(size, r#gen)| {
            PopulationDecision::SetGrowthTarget {
                size,
                generation: r#gen,
            }
        }),
        (1_usize..20, 1_u32..50).prop_map(|(migrants, r#gen)| {
            PopulationDecision::ApproveGeneticRescue(GeneticRescuePlan {
                source_population: "ext".to_string(),
                num_migrants: migrants,
                target_generation: r#gen,
                expected_heterozygosity_gain: 0.05,
            })
        }),
        Just(PopulationDecision::ModifyStrategy(
            BreedingStrategy::MinimumKinship
        )),
        Just(PopulationDecision::OverrideAiRecommendation(
            "safety".to_string()
        )),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Safety: GovernanceTier thresholds must be strictly monotonically increasing.
    /// If thresholds are not monotonic, a higher-stakes decision could pass with
    /// lower approval than a lower-stakes one, breaking the escalation model.
    #[test]
    fn prop_governance_tier_threshold_monotonicity(_dummy in 0..1_u8) {
        let tiers = GovernanceTier::all();
        for i in 1..tiers.len() {
            let prev = tiers[i - 1].threshold();
            let curr = tiers[i].threshold();
            prop_assert!(curr > prev,
                "tier {:?} threshold {} must be > tier {:?} threshold {}",
                tiers[i], curr, tiers[i - 1], prev);
        }
        // All thresholds must be in (0, 1]
        for tier in &tiers {
            let t = tier.threshold();
            prop_assert!(t > 0.0 && t <= 1.0,
                "threshold {} out of (0,1] for {:?}", t, tier);
        }
    }

    /// Safety: evaluate_decision_ethics must always return a value in [0, 1].
    /// Out-of-bounds ethics scores would break threshold comparisons and could
    /// allow unethical decisions to pass or block all decisions.
    #[test]
    fn prop_ethics_score_bounded(
        decision in arb_decision(),
        pop_size in 1_usize..500,
    ) {
        let pop = make_population(pop_size);
        let score = evaluate_decision_ethics(&decision, &pop);

        prop_assert!(score >= 0.0 && score <= 1.0,
            "ethics score {} out of [0,1] for pop_size={}", score, pop_size);
    }

    /// Safety: ethical_balance must always return a value in [0, 1].
    /// This function measures how evenly distributed ethical dimension weights are.
    /// Values outside [0, 1] would corrupt decision-making.
    #[test]
    fn prop_ethical_balance_bounded(
        autonomy in 0.0_f64..10.0,
        survival in 0.0_f64..10.0,
        dignity in 0.0_f64..10.0,
        justice in 0.0_f64..10.0,
    ) {
        let balance = ethical_balance(autonomy, survival, dignity, justice);

        prop_assert!(balance >= 0.0 && balance <= 1.0,
            "ethical_balance({}, {}, {}, {}) = {} out of [0,1]",
            autonomy, survival, dignity, justice, balance);
    }

    /// Safety: HDC kinship estimate must be symmetric -- kinship(a, b) == kinship(b, a).
    /// Asymmetric kinship would produce inconsistent breeding pair evaluations,
    /// potentially allowing inbreeding in one direction but not the other.
    #[test]
    fn prop_hdc_kinship_symmetric(
        seed_a in 0_u64..10000,
        seed_b in 0_u64..10000,
    ) {
        let a = Individual {
            id: 0,
            sex: BiologicalSex::Female,
            genotypes: vec![],
            genome_hv: ContinuousHV::random(HDC_DIMENSION, seed_a),
            generation: 0,
            parent_ids: (None, None),
            fitness: 1.0,
        };
        let b = Individual {
            id: 1,
            sex: BiologicalSex::Male,
            genotypes: vec![],
            genome_hv: ContinuousHV::random(HDC_DIMENSION, seed_b),
            generation: 0,
            parent_ids: (None, None),
            fitness: 1.0,
        };

        let k_ab = symthaea_population::hdc_kinship_estimate(&a, &b);
        let k_ba = symthaea_population::hdc_kinship_estimate(&b, &a);

        prop_assert!((k_ab - k_ba).abs() < 1e-10,
            "kinship must be symmetric: k(a,b)={} != k(b,a)={}", k_ab, k_ba);

        // Also verify bounds
        prop_assert!(k_ab >= 0.0 && k_ab <= 1.0,
            "kinship {} out of [0,1]", k_ab);
    }
}
