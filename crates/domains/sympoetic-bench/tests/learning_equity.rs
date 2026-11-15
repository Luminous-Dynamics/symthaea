use sympoetic_bench::{
    metrics::{StratifiedCapabilityObservation, summarize_capability_amplification_by_stratum},
};

#[test]
fn checked_in_distributional_capability_fixture_produces_two_stratum_estimates() {
    let fixture = include_str!(
        "../../../../docs/benchmarks/sympoetic/fixtures/distributional-capability.example.json"
    );
    let observations: Vec<StratifiedCapabilityObservation> =
        serde_json::from_str(fixture).unwrap();
    let summary =
        summarize_capability_amplification_by_stratum(&observations, 2_000, 42).unwrap();

    assert_eq!(summary.strata.len(), 2);
    assert_eq!(summary.strata[0].baseline_stratum, "high_baseline");
    assert_eq!(summary.strata[1].baseline_stratum, "low_baseline");

    let high = &summary.strata[0];
    assert_eq!(high.collaborative_participants, 3);
    assert_eq!(high.control_participants, 3);
    assert!((high.mean_gain_difference + 0.05).abs() < 1e-12);
    assert!((high.collaborative_negative_gain_rate - 1.0 / 3.0).abs() < 1e-12);
    assert_eq!(high.control_negative_gain_rate, 0.0);
    assert!(
        high.negative_gain_rate_difference_ci_95_lower
            <= high.negative_gain_rate_difference
    );
    assert!(
        high.negative_gain_rate_difference_ci_95_upper
            >= high.negative_gain_rate_difference
    );
}
