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
}
