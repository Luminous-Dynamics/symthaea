// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{
    ValidityCapacityPlan, measure_validity_capacity_score_moments, run_validity_capacity_sweep,
};

#[test]
fn public_score_oracle_matches_primary_smoke_capacity_metrics() {
    let plan = ValidityCapacityPlan::smoke();
    let primary = run_validity_capacity_sweep(&plan).unwrap();
    let diagnostic = measure_validity_capacity_score_moments(&plan).unwrap();
    assert_eq!(primary.observations.len(), diagnostic.observations.len());

    for (primary, diagnostic) in primary.observations.iter().zip(&diagnostic.observations) {
        assert_eq!(primary.case, diagnostic.case);
        assert_eq!(primary.seed, diagnostic.seed);
        assert_eq!(primary.correct, diagnostic.correct);
        assert_eq!(primary.total_queries, diagnostic.total_queries);
        assert_eq!(primary.accuracy.to_bits(), diagnostic.accuracy.to_bits());
        assert_eq!(primary.mean_margin.to_bits(), diagnostic.mean_winner_margin.to_bits());
        assert_eq!(
            primary.smallest_margin.to_bits(),
            diagnostic.smallest_winner_margin.to_bits()
        );
        assert_eq!(primary.spans_written, diagnostic.spans_written);
        assert_eq!(
            primary.represented_key_checkpoint_facts,
            diagnostic.represented_key_checkpoint_facts
        );
    }
}

#[test]
fn public_score_oracle_exposes_null_test_observables_without_success_assumption() {
    let result = measure_validity_capacity_score_moments(&ValidityCapacityPlan::smoke()).unwrap();
    for observation in result.observations {
        assert_eq!(observation.target_scores.count, observation.total_queries);
        assert_eq!(observation.probe_distractor_scores.count, observation.total_queries);
        assert!(observation.target_scores.mean.is_finite());
        assert!(observation.target_scores.variance.is_finite());
        assert!(observation.probe_distractor_scores.mean.is_finite());
        assert!(observation.probe_distractor_scores.variance.is_finite());
        assert!(observation.mean_true_margin.is_finite());
        assert!(observation.smallest_true_margin.is_finite());
        assert!(observation.null_target_noise_variance >= 0.0);
        assert!(observation.null_distractor_noise_variance >= 0.0);

        // No favorable accuracy/variance outcome is required. This only checks
        // the logical relationship between correctness and signed true margin.
        if observation.correct == observation.total_queries {
            assert!(observation.smallest_true_margin >= 0.0);
        }
        if observation.smallest_true_margin < 0.0 {
            assert!(observation.correct < observation.total_queries);
        }
    }
}

#[test]
fn research_v0_score_oracle_reuses_exact_frozen_capacity_plan() {
    let plan = ValidityCapacityPlan::research_v0();
    assert_eq!(plan.cases.len(), 27);
    assert_eq!(plan.replicate_seeds, vec![31_001, 31_002, 31_003, 31_004, 31_005]);
    // The diagnostic defines no independent case/seed selector. It accepts the
    // same public plan object, preventing a second post-hoc result-selection path.
}
