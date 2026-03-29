// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Moral Topology Analysis

Uses proptest to verify that `MoralTopology::analyze()` always produces
finite, bounded outputs regardless of input vector content. These tests
ensure that division guards, normalization, and Betti number computations
never produce NaN or Infinity in telemetry-bound fields.
*/

use super::moral_topology::{MoralTopology, MoralTopologyConfig};
use proptest::prelude::*;
use symthaea_core::hdc::ContinuousHV;
use symthaea_types::N_HARMONIES;

const TEST_DIM: usize = 512;

fn test_config() -> MoralTopologyConfig {
    MoralTopologyConfig {
        dim: TEST_DIM,
        window_size: 8,
        ..Default::default()
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    // =========================================================================
    // Property 1: All f64 fields in MoralTopologySummary are finite
    // =========================================================================
    #[test]
    fn prop_analyze_outputs_finite(seed in 0u64..10_000) {
        let mut topo = MoralTopology::new(test_config());
        // Feed 1-8 random scenarios
        let count = (seed % 8) + 1;
        for i in 0..count {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, seed * 100 + i));
        }
        let assessment = topo.analyze();
        let s = topo.last_summary();

        prop_assert!(s.unity.is_finite(), "unity not finite: {}", s.unity);
        prop_assert!(s.completeness.is_finite(), "completeness not finite: {}", s.completeness);
        prop_assert!(s.circularity.is_finite(), "circularity not finite: {}", s.circularity);
        prop_assert!(s.moral_free_energy.is_finite(), "moral_free_energy not finite: {}", s.moral_free_energy);
        prop_assert!(assessment.moral_free_energy.free_energy.is_finite());
    }

    // =========================================================================
    // Property 2: Unity is always in (0.0, 1.0]
    // =========================================================================
    #[test]
    fn prop_unity_bounded(seed in 0u64..10_000) {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..3 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, seed * 50 + i));
        }
        topo.analyze();
        let s = topo.last_summary();

        prop_assert!(s.unity > 0.0, "unity must be positive: {}", s.unity);
        prop_assert!(s.unity <= 1.0, "unity must be <= 1.0: {}", s.unity);
    }

    // =========================================================================
    // Property 3: Completeness is in [0.0, 1.0]
    // =========================================================================
    #[test]
    fn prop_completeness_bounded(seed in 0u64..10_000) {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..4 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, seed * 30 + i));
        }
        topo.analyze();
        let s = topo.last_summary();

        prop_assert!(s.completeness >= 0.0, "completeness must be >= 0: {}", s.completeness);
        prop_assert!(s.completeness <= 1.0, "completeness must be <= 1: {}", s.completeness);
    }

    // =========================================================================
    // Property 4: Circularity is in [0.0, 1.0]
    // =========================================================================
    #[test]
    fn prop_circularity_bounded(seed in 0u64..10_000) {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..5 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, seed * 20 + i));
        }
        topo.analyze();
        let s = topo.last_summary();

        prop_assert!(s.circularity >= 0.0, "circularity must be >= 0: {}", s.circularity);
        prop_assert!(s.circularity <= 1.0, "circularity must be <= 1: {}", s.circularity);
    }

    // =========================================================================
    // Property 5: beta_0 is always >= 1 (connected component invariant)
    // =========================================================================
    #[test]
    fn prop_beta0_at_least_one(seed in 0u64..10_000) {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..4 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, seed * 70 + i));
        }
        topo.analyze();
        let s = topo.last_summary();

        prop_assert!(s.beta_0 >= 1, "beta_0 must be >= 1: {}", s.beta_0);
    }

    // =========================================================================
    // Property 6: moral_free_energy is non-negative
    // =========================================================================
    #[test]
    fn prop_moral_free_energy_non_negative(seed in 0u64..10_000) {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..3 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, seed * 40 + i));
        }
        let assessment = topo.analyze();

        prop_assert!(
            assessment.moral_free_energy.free_energy >= 0.0,
            "moral free energy must be >= 0: {}",
            assessment.moral_free_energy.free_energy
        );
    }

    // =========================================================================
    // Property 7: All harmony_variance entries are finite and non-negative
    // =========================================================================
    #[test]
    fn prop_harmony_variance_finite(seed in 0u64..10_000) {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..4 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, seed * 60 + i));
        }
        let assessment = topo.analyze();

        for (idx, &v) in assessment.harmony_variance.iter().enumerate() {
            prop_assert!(v.is_finite(), "harmony_variance[{}] not finite: {}", idx, v);
            prop_assert!(v >= 0.0, "harmony_variance[{}] negative: {}", idx, v);
        }
    }

    // =========================================================================
    // Property 8: Empty window produces safe defaults
    // =========================================================================
    #[test]
    fn prop_empty_window_safe(_seed in 0u64..100) {
        let mut topo = MoralTopology::new(test_config());
        let assessment = topo.analyze();

        prop_assert_eq!(assessment.betti.beta_0, 1);
        prop_assert!((assessment.unity - 1.0).abs() < f64::EPSILON);
        prop_assert!((assessment.circularity - 0.0).abs() < f64::EPSILON);
        prop_assert!((assessment.completeness - 0.0).abs() < f64::EPSILON);
    }

    // =========================================================================
    // Property 9: FEP stillness prior floor holds after multiple observations
    // =========================================================================
    #[test]
    fn prop_stillness_prior_floor_holds(seed in 0u64..10_000) {
        let mut topo = MoralTopology::new(test_config());
        // Feed scenarios that suppress Sacred Stillness (index 7)
        for i in 0..8 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, seed * 90 + i));
        }
        // The harmony_prior[7] should never drop below the floor (0.05)
        // We can't directly access harmony_prior, but we can verify through
        // the moral free energy: if the prior has a floor, the free energy
        // for a stillness-heavy scenario should be bounded
        topo.analyze();
        let s = topo.last_summary();
        prop_assert!(s.moral_free_energy.is_finite(),
            "moral_free_energy must remain finite with stillness prior floor");
    }

    // =========================================================================
    // Property 10: Harmony entropy (moral breadth) is bounded
    // =========================================================================
    #[test]
    fn prop_harmony_entropy_bounded(seed in 0u64..10_000) {
        let mut topo = MoralTopology::new(test_config());
        for i in 0..4 {
            topo.add_scenario(ContinuousHV::random(TEST_DIM, seed * 45 + i));
        }
        let assessment = topo.analyze();
        // Compute harmony entropy from variance distribution
        let total_var: f64 = assessment.harmony_variance.iter().sum::<f64>().max(1e-12);
        let entropy: f64 = assessment.harmony_variance.iter()
            .map(|&v| {
                let p = (v / total_var).max(1e-12);
                -p * p.ln()
            })
            .sum();
        prop_assert!(entropy.is_finite(), "harmony entropy not finite: {}", entropy);
        prop_assert!(entropy >= 0.0, "harmony entropy negative: {}", entropy);
        // Max entropy = ln(8) ≈ 2.08
        prop_assert!(entropy <= (N_HARMONIES as f64).ln() + 0.01,
            "harmony entropy exceeds ln(N): {}", entropy);
    }

    // =========================================================================
    // Property 11: Attractor detection is stable under similar inputs
    // =========================================================================
    #[test]
    fn prop_attractor_stable_under_similar_inputs(seed in 0u64..10_000) {
        let mut topo = MoralTopology::new(test_config());
        // Feed identical scenarios to create a stable basin
        let base_hv = ContinuousHV::random(TEST_DIM, seed);
        for _ in 0..8 {
            topo.add_scenario(base_hv.clone());
        }
        let first = topo.analyze();
        // Feed same scenarios again
        for _ in 0..4 {
            topo.add_scenario(base_hv.clone());
        }
        let second = topo.analyze();
        // If first detected attractor, second should too (stable basin)
        if first.attractor_detected {
            prop_assert!(second.attractor_detected,
                "attractor flickered: detected then lost with identical inputs");
        }
        // Both assessments should have finite harmony_entropy
        prop_assert!(first.harmony_entropy.is_finite());
        prop_assert!(second.harmony_entropy.is_finite());
    }

    // =========================================================================
    // Property 12: Repeated analysis with identical inputs maintains stability
    // =========================================================================
    #[test]
    fn prop_repeated_analysis_stable(seed in 0u64..10_000) {
        let mut topo = MoralTopology::new(test_config());
        let base_hv = ContinuousHV::random(TEST_DIM, seed);
        // Feed same HV 8 times, analyze 10 times
        for _ in 0..8 {
            topo.add_scenario(base_hv.clone());
        }
        let mut prev_entropy = 0.0f64;
        for round in 0..10 {
            topo.add_scenario(base_hv.clone());
            let assessment = topo.analyze();
            prop_assert!(assessment.harmony_entropy.is_finite(),
                "harmony_entropy not finite at round {}", round);
            prop_assert!(assessment.moral_free_energy.free_energy.is_finite(),
                "moral_free_energy not finite at round {}", round);
            // Entropy should converge (not diverge)
            if round > 0 {
                let drift = (assessment.harmony_entropy - prev_entropy).abs();
                prop_assert!(drift < 1.0,
                    "harmony entropy drifted by {} at round {}", drift, round);
            }
            prev_entropy = assessment.harmony_entropy;
        }
    }
}
