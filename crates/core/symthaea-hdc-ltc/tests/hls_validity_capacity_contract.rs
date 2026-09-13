// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_hdc_ltc::{
    ValidityCapacityAxis, ValidityCapacityPlan, run_validity_capacity_sweep,
};

#[test]
fn public_smoke_capacity_sweep_is_deterministic_and_accounted() {
    let plan = ValidityCapacityPlan::smoke();
    let first = run_validity_capacity_sweep(&plan).unwrap();
    let second = run_validity_capacity_sweep(&plan).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.observations.len(), 4);

    for observation in &first.observations {
        assert_eq!(
            observation.total_queries,
            observation.case.key_count as u64 * observation.case.horizon
        );
        assert_eq!(
            observation.candidate_score_evaluations,
            observation.total_queries * observation.case.candidate_count as u64
        );
        assert!(observation.accuracy.is_finite());
        assert!(observation.mean_margin.is_finite());
        assert!(observation.smallest_margin.is_finite());
    }
}

#[test]
fn research_v0_grid_is_frozen_before_capacity_results() {
    let plan = ValidityCapacityPlan::research_v0();
    assert_eq!(plan.cases.len(), 27);
    assert_eq!(plan.replicate_seeds, vec![31_001, 31_002, 31_003, 31_004, 31_005]);

    let count = |axis| plan.cases.iter().filter(|case| case.axis == axis).count();
    assert_eq!(count(ValidityCapacityAxis::Dimension), 5);
    assert_eq!(count(ValidityCapacityAxis::KeyCount), 5);
    assert_eq!(count(ValidityCapacityAxis::CandidateCount), 5);
    assert_eq!(count(ValidityCapacityAxis::Horizon), 5);
    assert_eq!(count(ValidityCapacityAxis::SpanLength), 7);

    let dimension_values = plan
        .cases
        .iter()
        .filter(|case| case.axis == ValidityCapacityAxis::Dimension)
        .map(|case| case.dim)
        .collect::<Vec<_>>();
    assert_eq!(dimension_values, vec![512, 1_024, 2_048, 4_096, 8_192]);

    let horizon_values = plan
        .cases
        .iter()
        .filter(|case| case.axis == ValidityCapacityAxis::Horizon)
        .map(|case| case.horizon)
        .collect::<Vec<_>>();
    assert_eq!(horizon_values, vec![32, 64, 128, 256, 512]);

    let span_values = plan
        .cases
        .iter()
        .filter(|case| case.axis == ValidityCapacityAxis::SpanLength)
        .map(|case| case.span_length)
        .collect::<Vec<_>>();
    assert_eq!(span_values, vec![1, 2, 4, 8, 16, 32, 64]);
}
