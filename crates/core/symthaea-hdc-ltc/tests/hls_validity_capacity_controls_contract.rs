// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{
    ValidityCapacityControlAxis, ValidityCapacityControlPlan, ValidityCapacityNullModel,
    run_validity_capacity_controls,
};

#[test]
fn public_smoke_controls_are_deterministic_and_share_null_prediction() {
    let plan = ValidityCapacityControlPlan::smoke();
    let first = run_validity_capacity_controls(&plan).unwrap();
    let second = run_validity_capacity_controls(&plan).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.observations.len(), 4);

    let null = ValidityCapacityNullModel::new(256, 2, 4, 16).unwrap();
    for observation in &first.observations {
        assert_eq!(observation.null_target_noise_variance, null.target_noise_variance);
        assert_eq!(
            observation.null_distractor_noise_variance,
            null.distractor_noise_variance
        );
        assert_eq!(
            observation.represented_key_checkpoint_facts,
            observation.case.key_count as u64 * observation.case.horizon
        );
        assert!(observation.accuracy.is_finite());
        assert!(observation.mean_margin.is_finite());
        assert!(observation.smallest_margin.is_finite());
    }
}

#[test]
fn equivalent_smoke_histories_are_segmentation_invariant_within_roundoff() {
    let result = run_validity_capacity_controls(&ValidityCapacityControlPlan::smoke()).unwrap();

    for seed in [33_001u64, 33_002] {
        let unit_writes = result
            .observations
            .iter()
            .find(|observation| {
                observation.seed == seed
                    && observation.case.axis == ValidityCapacityControlAxis::SemanticRunLength
            })
            .unwrap();
        let two_checkpoint_writes = result
            .observations
            .iter()
            .find(|observation| {
                observation.seed == seed
                    && observation.case.axis == ValidityCapacityControlAxis::WriteSegmentation
            })
            .unwrap();

        // Same seed, D/K/C/H, semantic-run length, and forced semantic schedule.
        // Only the write partition differs: 1 checkpoint vs 2 checkpoints/write.
        assert_eq!(unit_writes.semantic_changes, two_checkpoint_writes.semantic_changes);
        assert_ne!(unit_writes.spans_written, two_checkpoint_writes.spans_written);
        assert_eq!(unit_writes.correct, two_checkpoint_writes.correct);
        assert_eq!(unit_writes.total_queries, two_checkpoint_writes.total_queries);
        assert!((unit_writes.mean_margin - two_checkpoint_writes.mean_margin).abs() < 1e-9);
        assert!((unit_writes.smallest_margin - two_checkpoint_writes.smallest_margin).abs() < 1e-9);
    }
}

#[test]
fn research_v0_freezes_orthogonal_controls_on_untouched_seeds() {
    let plan = ValidityCapacityControlPlan::research_v0();
    assert_eq!(plan.cases.len(), 13);
    assert_eq!(
        plan.replicate_seeds,
        vec![32_001, 32_002, 32_003, 32_004, 32_005, 32_006, 32_007, 32_008]
    );

    let semantic = plan
        .cases
        .iter()
        .filter(|case| case.axis == ValidityCapacityControlAxis::SemanticRunLength)
        .collect::<Vec<_>>();
    let segmentation = plan
        .cases
        .iter()
        .filter(|case| case.axis == ValidityCapacityControlAxis::WriteSegmentation)
        .collect::<Vec<_>>();
    assert_eq!(semantic.len(), 7);
    assert_eq!(segmentation.len(), 6);

    assert_eq!(
        semantic.iter().map(|case| case.semantic_run_length).collect::<Vec<_>>(),
        vec![1, 2, 4, 8, 16, 32, 64]
    );
    assert!(semantic.iter().all(|case| case.write_segment_length == 1));

    assert!(segmentation.iter().all(|case| case.semantic_run_length == 32));
    assert_eq!(
        segmentation.iter().map(|case| case.write_segment_length).collect::<Vec<_>>(),
        vec![1, 2, 4, 8, 16, 32]
    );

    // D/K/C/H are fixed everywhere, so every case has exactly the same null
    // prediction. Any systematic case effect is therefore a model departure.
    let reference = ValidityCapacityNullModel::new(4096, 8, 8, 256).unwrap();
    for case in plan.cases {
        assert_eq!(case.dim, 4096);
        assert_eq!(case.key_count, 8);
        assert_eq!(case.candidate_count, 8);
        assert_eq!(case.horizon, 256);
        assert_eq!(
            ValidityCapacityNullModel::new(
                case.dim,
                case.key_count,
                case.candidate_count,
                case.horizon,
            )
            .unwrap(),
            reference
        );
    }
}
