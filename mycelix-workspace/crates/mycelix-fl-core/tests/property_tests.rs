//! Property-based tests for mycelix-fl-core using proptest.
//!
//! Covers aggregation invariants, algorithm-specific properties,
//! Shapley axioms, replay detection, and ensemble defense.

use proptest::prelude::*;

use mycelix_fl_core::aggregation::{
    coordinate_median, fedavg, geometric_median, krum, multi_krum, trimmed_mean,
};
use mycelix_fl_core::types::GradientUpdate;

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// Generate `count` GradientUpdates with `dim`-dimensional gradients in [-100, 100].
fn gradient_update_strategy(
    dim: usize,
    count: usize,
) -> impl Strategy<Value = Vec<GradientUpdate>> {
    prop::collection::vec(
        (
            prop::collection::vec(-100.0f32..100.0f32, dim..=dim),
            1u32..1000u32,
            0.0f32..10.0f32,
        ),
        count..=count,
    )
    .prop_map(|entries| {
        entries
            .into_iter()
            .enumerate()
            .map(|(i, (gradients, batch_size, loss))| {
                GradientUpdate::new(format!("node-{}", i), 1, gradients, batch_size, loss)
            })
            .collect()
    })
}

/// Generate identical GradientUpdates: every participant has the same gradient.
fn identical_updates_strategy(
    dim: usize,
    count: usize,
) -> impl Strategy<Value = Vec<GradientUpdate>> {
    (
        prop::collection::vec(-100.0f32..100.0f32, dim..=dim),
        1u32..1000u32,
        0.0f32..10.0f32,
    )
        .prop_map(move |(gradients, batch_size, loss)| {
            (0..count)
                .map(|i| {
                    GradientUpdate::new(
                        format!("node-{}", i), 1, gradients.clone(), batch_size, loss,
                    )
                })
                .collect()
        })
}

/// Helper: assert every element is finite.
fn assert_all_finite(name: &str, vals: &[f32]) -> Result<(), TestCaseError> {
    for (i, v) in vals.iter().enumerate() {
        prop_assert!(v.is_finite(), "{} non-finite at [{}]: {}", name, i, v);
    }
    Ok(())
}

/// Helper: assert output matches expected within tolerance.
fn assert_close(name: &str, result: &[f32], expected: &[f32], tol: f32) -> Result<(), TestCaseError> {
    for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
        prop_assert!((r - e).abs() < tol, "{} mismatch at [{}]: {} vs {}", name, i, r, e);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Aggregation invariants (all algorithms)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Output dimension must match input dimension for all algorithms.
    #[test]
    fn aggregation_output_dimension_matches_input(updates in gradient_update_strategy(8, 5)) {
        let dim = updates[0].gradients.len();
        prop_assert_eq!(fedavg(&updates).unwrap().len(), dim);
        prop_assert_eq!(trimmed_mean(&updates, 0.1).unwrap().len(), dim);
        prop_assert_eq!(coordinate_median(&updates).unwrap().len(), dim);
        prop_assert_eq!(krum(&updates, 1).unwrap().len(), dim);
        prop_assert_eq!(multi_krum(&updates, 1, 2).unwrap().len(), dim);
        prop_assert_eq!(geometric_median(&updates, 50, 1e-5).unwrap().len(), dim);
    }

    /// All output values must be finite (no NaN/Inf).
    #[test]
    fn aggregation_output_all_finite(updates in gradient_update_strategy(6, 5)) {
        assert_all_finite("fedavg", &fedavg(&updates).unwrap())?;
        assert_all_finite("trimmed_mean", &trimmed_mean(&updates, 0.2).unwrap())?;
        assert_all_finite("coordinate_median", &coordinate_median(&updates).unwrap())?;
        assert_all_finite("krum", &krum(&updates, 1).unwrap())?;
        assert_all_finite("multi_krum", &multi_krum(&updates, 1, 2).unwrap())?;
        assert_all_finite("geometric_median", &geometric_median(&updates, 50, 1e-5).unwrap())?;
    }

    /// Single input returns that input (identity) for algorithms without
    /// minimum participant requirements.
    #[test]
    fn single_input_returns_identity(updates in gradient_update_strategy(4, 1)) {
        let expected = &updates[0].gradients;
        assert_close("fedavg", &fedavg(&updates).unwrap(), expected, 1e-5)?;
        assert_close("trimmed_mean", &trimmed_mean(&updates, 0.0).unwrap(), expected, 1e-5)?;
        assert_close("coordinate_median", &coordinate_median(&updates).unwrap(), expected, 1e-5)?;
        assert_close("geometric_median", &geometric_median(&updates, 50, 1e-6).unwrap(), expected, 1e-4)?;
    }
}

// ---------------------------------------------------------------------------
// FedAvg properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// FedAvg output lies within the convex hull of inputs per dimension.
    #[test]
    fn fedavg_in_convex_hull(updates in gradient_update_strategy(6, 5)) {
        let result = fedavg(&updates).unwrap();
        for d in 0..updates[0].gradients.len() {
            let min_v = updates.iter().map(|u| u.gradients[d]).fold(f32::INFINITY, f32::min);
            let max_v = updates.iter().map(|u| u.gradients[d]).fold(f32::NEG_INFINITY, f32::max);
            prop_assert!(
                result[d] >= min_v - 1e-5 && result[d] <= max_v + 1e-5,
                "fedavg[{}]={} not in [{},{}]", d, result[d], min_v, max_v
            );
        }
    }

    /// Identical inputs yield that identical value.
    #[test]
    fn fedavg_identical_inputs(updates in identical_updates_strategy(6, 4)) {
        let result = fedavg(&updates).unwrap();
        assert_close("fedavg_ident", &result, &updates[0].gradients, 1e-4)?;
    }
}

// ---------------------------------------------------------------------------
// Krum properties
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Krum(1) with equal batch sizes returns exactly one of the inputs.
    #[test]
    fn krum_selects_one_input(
        gradients in prop::collection::vec(
            prop::collection::vec(-50.0f32..50.0f32, 4..=4), 5..=5,
        )
    ) {
        let updates: Vec<GradientUpdate> = gradients.iter().enumerate()
            .map(|(i, g)| GradientUpdate::new(format!("node-{}", i), 1, g.clone(), 100, 0.5))
            .collect();
        let result = krum(&updates, 1).unwrap();
        let matched = updates.iter().any(|u| {
            u.gradients.iter().zip(result.iter()).all(|(a, b)| (a - b).abs() < 1e-6)
        });
        prop_assert!(matched, "krum(1) result {:?} not among inputs", result);
    }
}

// ---------------------------------------------------------------------------
// Trimmed mean within range
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Trimmed mean output lies within per-dimension min/max of all inputs.
    #[test]
    fn trimmed_mean_within_range(updates in gradient_update_strategy(6, 7)) {
        let result = trimmed_mean(&updates, 0.15).unwrap();
        for d in 0..updates[0].gradients.len() {
            let min_v = updates.iter().map(|u| u.gradients[d]).fold(f32::INFINITY, f32::min);
            let max_v = updates.iter().map(|u| u.gradients[d]).fold(f32::NEG_INFINITY, f32::max);
            prop_assert!(
                result[d] >= min_v - 1e-5 && result[d] <= max_v + 1e-5,
                "trimmed_mean[{}]={} not in [{},{}]", d, result[d], min_v, max_v
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Geometric median bounded
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Geometric median lies within the convex hull of inputs.
    #[test]
    fn geometric_median_within_range(updates in gradient_update_strategy(4, 5)) {
        let result = geometric_median(&updates, 100, 1e-6).unwrap();
        for d in 0..updates[0].gradients.len() {
            let min_v = updates.iter().map(|u| u.gradients[d]).fold(f32::INFINITY, f32::min);
            let max_v = updates.iter().map(|u| u.gradients[d]).fold(f32::NEG_INFINITY, f32::max);
            prop_assert!(
                result[d] >= min_v - 0.1 && result[d] <= max_v + 0.1,
                "geometric_median[{}]={} not in [{},{}]", d, result[d], min_v, max_v
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Shapley properties (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "shapley")]
mod shapley_tests {
    use super::*;
    use std::collections::HashMap;
    use mycelix_fl_core::shapley::{ShapleyCalculator, ShapleyConfig};

    fn mean_gradient(gradients: &HashMap<String, Vec<f32>>, dim: usize) -> Vec<f32> {
        let n = gradients.len() as f32;
        let mut avg = vec![0.0f32; dim];
        for g in gradients.values() {
            for (j, v) in g.iter().enumerate() { avg[j] += v / n; }
        }
        avg
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Efficiency: values sum to total coalition value within tolerance.
        #[test]
        fn shapley_efficiency(
            raw in prop::collection::vec(
                prop::collection::vec(-10.0f32..10.0f32, 3..=3), 3..=5usize,
            )
        ) {
            let mut grads: HashMap<String, Vec<f32>> = HashMap::new();
            for (i, g) in raw.iter().enumerate() { grads.insert(format!("node-{}", i), g.clone()); }
            let agg = mean_gradient(&grads, 3);
            let mut calc = ShapleyCalculator::new(ShapleyConfig::exact().with_seed(42));
            let result = calc.calculate_values(&grads, &agg);
            prop_assert!(result.efficiency_error < 0.05,
                "Efficiency error {} too large", result.efficiency_error);
        }

        /// Symmetry: identical contributors get the same value.
        #[test]
        fn shapley_symmetry(
            gradient in prop::collection::vec(-10.0f32..10.0f32, 3..=3),
            extra in prop::collection::vec(-10.0f32..10.0f32, 3..=3),
        ) {
            let mut grads: HashMap<String, Vec<f32>> = HashMap::new();
            grads.insert("sym-a".into(), gradient.clone());
            grads.insert("sym-b".into(), gradient);
            grads.insert("other".into(), extra);
            let agg = mean_gradient(&grads, 3);
            let mut calc = ShapleyCalculator::new(ShapleyConfig::exact().with_seed(42));
            let result = calc.calculate_values(&grads, &agg);
            let va = result.values.get("sym-a").copied().unwrap_or(0.0);
            let vb = result.values.get("sym-b").copied().unwrap_or(0.0);
            prop_assert!((va - vb).abs() < 0.02,
                "Symmetry violated: sym-a={}, sym-b={}", va, vb);
        }

        /// Null player: zero-gradient contributor gets smaller value than honest.
        #[test]
        fn shapley_null_player(
            gradient in prop::collection::vec(1.0f32..10.0f32, 3..=3),
        ) {
            let mut grads: HashMap<String, Vec<f32>> = HashMap::new();
            grads.insert("honest-1".into(), gradient.clone());
            grads.insert("honest-2".into(), gradient.iter().map(|v| v * 0.9).collect());
            grads.insert("null-node".into(), vec![0.0; 3]);
            let agg = gradient;
            let mut calc = ShapleyCalculator::new(ShapleyConfig::exact().with_seed(42));
            let result = calc.calculate_values(&grads, &agg);
            let null_v = result.values.get("null-node").copied().unwrap_or(0.0);
            let honest_v = result.values.get("honest-1").copied().unwrap_or(0.0);
            prop_assert!(null_v.abs() < honest_v.abs() + 0.1,
                "Null player {} should be smaller than honest {}", null_v, honest_v);
        }
    }
}

// ---------------------------------------------------------------------------
// Replay detection (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "replay")]
mod replay_tests {
    use super::*;
    use mycelix_fl_core::replay_detection::{ReplayDetector, ReplayDetectorConfig};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Exact duplicate gradient is flagged as replay.
        #[test]
        fn replay_exact_duplicate_flagged(
            gradient in prop::collection::vec(-100.0f32..100.0f32, 10..=10)
        ) {
            let mut det = ReplayDetector::new(ReplayDetectorConfig::default());
            det.record_submission("node-a", &gradient, 1);
            let r = det.check_replay("node-a", &gradient, 2);
            prop_assert!(r.is_replay, "Exact duplicate should be flagged");
        }

        /// Sufficiently different gradient is NOT flagged.
        #[test]
        fn replay_different_gradient_not_flagged(
            grad1 in prop::collection::vec(-100.0f32..100.0f32, 20..=20),
            offset in prop::collection::vec(50.0f32..200.0f32, 20..=20),
        ) {
            let mut det = ReplayDetector::new(
                ReplayDetectorConfig::default().with_similarity_threshold(0.99)
            );
            det.record_submission("node-a", &grad1, 1);
            let grad2: Vec<f32> = grad1.iter().zip(offset.iter()).map(|(a, b)| a + b).collect();
            let r = det.check_replay("node-a", &grad2, 2);
            prop_assert!(!r.is_replay, "Different gradient flagged (sim={})", r.similarity);
        }

        /// Deterministic: same inputs produce same result.
        #[test]
        fn replay_deterministic(
            gradient in prop::collection::vec(-100.0f32..100.0f32, 10..=10),
            check in prop::collection::vec(-100.0f32..100.0f32, 10..=10),
        ) {
            let mut d1 = ReplayDetector::new(ReplayDetectorConfig::default());
            d1.record_submission("node-a", &gradient, 1);
            let r1 = d1.check_replay("node-a", &check, 2);

            let mut d2 = ReplayDetector::new(ReplayDetectorConfig::default());
            d2.record_submission("node-a", &gradient, 1);
            let r2 = d2.check_replay("node-a", &check, 2);

            prop_assert_eq!(r1.is_replay, r2.is_replay);
            prop_assert!((r1.similarity - r2.similarity).abs() < 1e-6);
        }
    }
}

// ---------------------------------------------------------------------------
// Ensemble defense (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "ensemble")]
mod ensemble_tests {
    use super::*;
    use mycelix_fl_core::ensemble_defense::{EnsembleConfig, EnsembleDefense};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(32))]

        /// Byzantine set is a subset of input participant IDs.
        #[test]
        fn ensemble_byzantine_subset_of_inputs(updates in gradient_update_strategy(4, 6)) {
            let config = EnsembleConfig {
                byzantine_consensus_threshold: 0.3,
                ..EnsembleConfig::robust()
            };
            let ensemble = EnsembleDefense::new(config);
            let result = ensemble.aggregate(&updates).unwrap();
            let ids: Vec<String> = updates.iter().map(|u| u.participant_id.clone()).collect();
            for flagged in &result.consensus_byzantine {
                prop_assert!(ids.contains(flagged), "Flagged {:?} not a participant", flagged);
            }
            for (nid, _) in &result.flagged_nodes {
                prop_assert!(ids.contains(nid), "Flagged {:?} not a participant", nid);
            }
        }

        /// Ensemble output dimension matches input gradient dimension.
        #[test]
        fn ensemble_output_dimension(updates in gradient_update_strategy(5, 6)) {
            let ensemble = EnsembleDefense::new(EnsembleConfig::robust());
            let result = ensemble.aggregate(&updates).unwrap();
            prop_assert_eq!(result.aggregated.len(), updates[0].gradients.len());
        }

        /// Agreement score is in [0, 1].
        #[test]
        fn ensemble_agreement_in_range(updates in gradient_update_strategy(4, 6)) {
            let ensemble = EnsembleDefense::new(EnsembleConfig::robust());
            let result = ensemble.aggregate(&updates).unwrap();
            prop_assert!(result.agreement_score >= 0.0 && result.agreement_score <= 1.0 + 1e-5,
                "Agreement {} out of [0,1]", result.agreement_score);
        }
    }
}
