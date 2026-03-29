// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Safety-invariant property tests for symthaea-genomics.
//!
//! Tests that critical safety bounds hold across arbitrary inputs:
//! - Base complement is an involution (complement(complement(x)) == x)
//! - DamageEvent severity is always in [0, 1]
//! - Damage monotonically increases with age (expected fraction)
//! - RepairPlan confidence is always in [0, 1]

use proptest::prelude::*;
use rand::SeedableRng;
use symthaea_genomics::{Base, DamageModel, DamageType, DnaSequence, RepairPlanner};
use symthaea_genomics::{DamageEvent, DegradedSample, GenomeQuality};

fn arb_base() -> impl Strategy<Value = Base> {
    prop_oneof![
        Just(Base::A),
        Just(Base::T),
        Just(Base::G),
        Just(Base::C),
        Just(Base::N),
    ]
}

fn arb_damage_type() -> impl Strategy<Value = DamageType> {
    prop_oneof![
        Just(DamageType::Deamination),
        Just(DamageType::Depurination),
        Just(DamageType::StrandBreak),
        Just(DamageType::Crosslink),
        Just(DamageType::Oxidation),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// Safety: complement is an involution -- applying it twice returns the original base.
    /// This ensures Watson-Crick pairing is self-consistent and no base is "lost"
    /// through complementation, which would corrupt sequence reconstruction.
    #[test]
    fn prop_complement_involution(base in arb_base()) {
        prop_assert_eq!(base.complement().complement(), base,
            "complement must be an involution: complement(complement({:?})) != {:?}", base, base);
    }

    /// Safety: DamageModel severity values for predicted damage events are always
    /// clamped to [0, 1]. Out-of-bounds severity would corrupt repair planning
    /// confidence calculations and quality assessments.
    #[test]
    fn prop_damage_severity_bounded(
        temp in -20.0_f64..60.0,
        age in 0.0_f64..1_000_000.0,
        seed in any::<u64>(),
    ) {
        let model = DamageModel::new(temp);
        let seq = DnaSequence::from_str("ATGCATGCATGCATGC");
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let events = model.predict_damage(&seq, age, &mut rng);

        for event in &events {
            prop_assert!(event.severity >= 0.0 && event.severity <= 1.0,
                "severity {} out of [0,1] at temp={}, age={}", event.severity, temp, age);
        }
    }

    /// Safety: expected deamination fraction is monotonically non-decreasing with age.
    /// A violation would mean older samples are predicted to have less damage,
    /// which breaks the fundamental Arrhenius kinetics model.
    #[test]
    fn prop_deamination_fraction_monotonic_with_age(
        temp in -10.0_f64..50.0,
        age_young in 0.0_f64..50_000.0,
        age_delta in 0.0_f64..500_000.0,
    ) {
        let model = DamageModel::new(temp);
        let age_old = age_young + age_delta;

        let frac_young = model.expected_deamination_fraction(age_young);
        let frac_old = model.expected_deamination_fraction(age_old);

        prop_assert!(frac_young >= 0.0 && frac_young <= 1.0,
            "young fraction {} out of [0,1]", frac_young);
        prop_assert!(frac_old >= 0.0 && frac_old <= 1.0,
            "old fraction {} out of [0,1]", frac_old);
        prop_assert!(frac_old >= frac_young - 1e-12,
            "deamination fraction must be monotonic: young({})={} > old({})={}",
            age_young, frac_young, age_old, frac_old);
    }

    /// Safety: RepairPlan confidence is always in [0, 1] regardless of damage
    /// configuration, severity, or quality metrics. Out-of-bounds confidence
    /// would break downstream decision thresholds.
    #[test]
    fn prop_repair_plan_confidence_bounded(
        severity in 0.0_f64..1.0,
        coverage in 0.0_f64..100.0,
        damage_type in arb_damage_type(),
        min_confidence in 0.0_f64..1.0,
    ) {
        let planner = RepairPlanner::new(100, min_confidence);
        let quality = GenomeQuality {
            n50: 500,
            total_length: 1000,
            num_contigs: 2,
            mean_coverage: coverage,
            completeness: 0.9,
            error_rate: 0.01,
        };
        let sample = DegradedSample {
            sequence: DnaSequence::from_str(&"A".repeat(100)),
            estimated_age_years: 5000.0,
            storage_temp_celsius: 15.0,
            contamination_estimate: 0.05,
            damage_map: vec![DamageEvent {
                position: 10,
                damage_type,
                severity,
            }],
        };

        let plan = planner.plan_repairs(&sample, &quality);

        prop_assert!(plan.confidence >= 0.0 && plan.confidence <= 1.0,
            "repair plan confidence {} out of [0,1]", plan.confidence);

        // Also check per-region confidence bounds
        for region in &plan.repairable_regions {
            prop_assert!(region.confidence >= 0.0 && region.confidence <= 1.0,
                "region confidence {} out of [0,1]", region.confidence);
        }
    }

    /// Safety: reverse complement is an involution on sequences --
    /// rev_comp(rev_comp(seq)) == seq. Violation would corrupt assembly
    /// algorithms that rely on strand symmetry.
    #[test]
    fn prop_reverse_complement_involution(bases in prop::collection::vec(arb_base(), 0..64)) {
        let seq = DnaSequence::new(bases);
        let double_rc = seq.reverse_complement().reverse_complement();

        prop_assert_eq!(seq.bases, double_rc.bases,
            "reverse_complement must be an involution");
    }
}
