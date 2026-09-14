#[path = "support/validity_direct_sum_oracle.rs"]
mod direct_sum_oracle;

use direct_sum_oracle::{
    ORACLE_EXACT_CAUSAL_CHECKPOINT_LIMIT, OracleError, OracleSpan, explicit_candidate_scores,
};
use symthaea_hdc_ltc::{
    EXACT_CAUSAL_CHECKPOINT_LIMIT, TemporalAxis, UnitaryRole, ValidityIntervalMemory,
    ValidityMemoryError,
};

const SCORE_TOL: f64 = 1.0e-10;
const LARGE_OFFSET_SCORE_TOL: f64 = 1.0e-6;
const ORACLE_SOURCE: &str = include_str!("support/validity_direct_sum_oracle.rs");

fn roles(count: usize, dim: usize, seed: u64) -> Vec<UnitaryRole> {
    (0..count)
        .map(|index| UnitaryRole::new(dim, seed + index as u64))
        .collect()
}

fn build_production_archive(
    axis: &TemporalAxis,
    keys: &[UnitaryRole],
    values: &[UnitaryRole],
    spans: &[OracleSpan],
) -> ValidityIntervalMemory {
    let mut memory = ValidityIntervalMemory::new(axis.dim()).unwrap();
    for span in spans {
        memory
            .write_span(
                axis,
                &keys[span.key_index],
                &values[span.value_index],
                span.start,
                span.end_exclusive,
            )
            .unwrap();
    }
    memory
}

fn expected_ranking(scores: &[f64]) -> (usize, f64, f64) {
    let mut ranked = scores.iter().copied().enumerate().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1));
    (ranked[0].0, ranked[0].1, ranked[1].1)
}

#[test]
fn oracle_support_source_excludes_production_memory_api_dependencies() {
    for forbidden in [
        "ValidityIntervalMemory",
        "write_span(",
        "cleanup(",
        "score_candidate(",
    ] {
        assert!(
            !ORACLE_SOURCE.contains(forbidden),
            "independent oracle support acquired forbidden production dependency: {forbidden}"
        );
    }
}

#[test]
fn hand_authored_axis_and_roles_match_without_rng_fixture_dependence() {
    let axis = TemporalAxis::try_from_frequencies(vec![
        0.0,
        1.0e-13,
        0.125,
        -0.5,
        1.0,
        -1.5,
        2.0,
        -3.0,
    ])
    .unwrap();
    let keys = vec![
        UnitaryRole::try_from_values(vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
            .unwrap(),
        UnitaryRole::try_from_values(vec![-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
            .unwrap(),
    ];
    let values = vec![
        UnitaryRole::try_from_values(vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
            .unwrap(),
        UnitaryRole::try_from_values(vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0])
            .unwrap(),
        UnitaryRole::try_from_values(vec![1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0])
            .unwrap(),
    ];
    let spans = [
        OracleSpan {
            key_index: 0,
            value_index: 0,
            start: 0,
            end_exclusive: 4,
        },
        OracleSpan {
            key_index: 0,
            value_index: 2,
            start: 4,
            end_exclusive: 7,
        },
        OracleSpan {
            key_index: 1,
            value_index: 1,
            start: 0,
            end_exclusive: 2,
        },
        OracleSpan {
            key_index: 1,
            value_index: 0,
            start: 2,
            end_exclusive: 7,
        },
    ];
    let memory = build_production_archive(&axis, &keys, &values, &spans);

    for key_index in 0..keys.len() {
        for checkpoint in 0..7_u64 {
            let oracle =
                explicit_candidate_scores(&axis, &keys, &values, &spans, key_index, checkpoint)
                    .unwrap();
            for (candidate_index, expected) in oracle.iter().copied().enumerate() {
                let actual = memory
                    .score_candidate(
                        &axis,
                        &keys[key_index],
                        &values[candidate_index],
                        checkpoint,
                    )
                    .unwrap();
                assert!(
                    (actual - expected).abs() < SCORE_TOL,
                    "hand-authored fixture mismatch key={key_index}, checkpoint={checkpoint}, candidate={candidate_index}: actual={actual:.17e}, oracle={expected:.17e}"
                );
            }
        }
    }
}

#[test]
fn independent_direct_sum_matches_production_scores_and_cleanup() {
    let dim = 512;
    let axis = TemporalAxis::new(dim, 0xD1CE_7001).unwrap();
    let keys = roles(3, dim, 10_000);
    let values = roles(4, dim, 20_000);
    let spans = [
        OracleSpan {
            key_index: 0,
            value_index: 0,
            start: 0,
            end_exclusive: 3,
        },
        OracleSpan {
            key_index: 0,
            value_index: 1,
            start: 3,
            end_exclusive: 8,
        },
        OracleSpan {
            key_index: 0,
            value_index: 3,
            start: 8,
            end_exclusive: 12,
        },
        OracleSpan {
            key_index: 1,
            value_index: 2,
            start: 0,
            end_exclusive: 5,
        },
        OracleSpan {
            key_index: 1,
            value_index: 0,
            start: 5,
            end_exclusive: 12,
        },
        OracleSpan {
            key_index: 2,
            value_index: 3,
            start: 0,
            end_exclusive: 4,
        },
        OracleSpan {
            key_index: 2,
            value_index: 1,
            start: 4,
            end_exclusive: 9,
        },
        OracleSpan {
            key_index: 2,
            value_index: 2,
            start: 9,
            end_exclusive: 12,
        },
    ];
    let memory = build_production_archive(&axis, &keys, &values, &spans);

    let mut maximum_score_error = 0.0_f64;
    for key_index in 0..keys.len() {
        for checkpoint in 0..12_u64 {
            let oracle =
                explicit_candidate_scores(&axis, &keys, &values, &spans, key_index, checkpoint)
                    .unwrap();
            for (candidate_index, expected) in oracle.iter().copied().enumerate() {
                let actual = memory
                    .score_candidate(
                        &axis,
                        &keys[key_index],
                        &values[candidate_index],
                        checkpoint,
                    )
                    .unwrap();
                let error = (actual - expected).abs();
                maximum_score_error = maximum_score_error.max(error);
                assert!(
                    error < SCORE_TOL,
                    "score mismatch key={key_index}, checkpoint={checkpoint}, candidate={candidate_index}: actual={actual:.17e}, oracle={expected:.17e}, error={error:.17e}"
                );
            }

            let (expected_best, expected_best_score, expected_second_score) =
                expected_ranking(&oracle);
            let cleanup = memory
                .cleanup(&axis, &keys[key_index], &values, checkpoint)
                .unwrap();
            assert_eq!(cleanup.best_index, expected_best);
            assert!((cleanup.best_score - expected_best_score).abs() < SCORE_TOL);
            assert!((cleanup.second_score - expected_second_score).abs() < SCORE_TOL);
            assert!(
                (cleanup.margin - (expected_best_score - expected_second_score)).abs() < SCORE_TOL
            );
        }
    }

    eprintln!("maximum direct-sum/production score error={maximum_score_error:.17e}");
}

#[test]
fn independent_direct_sum_matches_large_offset_scores_without_tolerance_drift() {
    let dim = 256;
    let axis = TemporalAxis::new(dim, 0xD1CE_7002).unwrap();
    let keys = roles(2, dim, 30_000);
    let values = roles(3, dim, 40_000);
    let start = 1_000_000_u64;
    let spans = [
        OracleSpan {
            key_index: 0,
            value_index: 0,
            start,
            end_exclusive: start + 31,
        },
        OracleSpan {
            key_index: 1,
            value_index: 2,
            start,
            end_exclusive: start + 17,
        },
    ];
    let memory = build_production_archive(&axis, &keys, &values, &spans);
    let mut maximum_score_error = 0.0_f64;

    for key_index in 0..keys.len() {
        for checkpoint in [start, start + 7, start + 16, start + 30] {
            let oracle =
                explicit_candidate_scores(&axis, &keys, &values, &spans, key_index, checkpoint)
                    .unwrap();
            for (candidate_index, expected) in oracle.iter().copied().enumerate() {
                let actual = memory
                    .score_candidate(
                        &axis,
                        &keys[key_index],
                        &values[candidate_index],
                        checkpoint,
                    )
                    .unwrap();
                let error = (actual - expected).abs();
                maximum_score_error = maximum_score_error.max(error);
                assert!(
                    error < LARGE_OFFSET_SCORE_TOL,
                    "large-offset score mismatch key={key_index}, checkpoint={checkpoint}, candidate={candidate_index}: error={error:.17e}"
                );
            }
        }
    }

    eprintln!("maximum large-offset direct-sum/production score error={maximum_score_error:.17e}");
}

#[test]
fn oracle_and_public_api_pin_the_same_exact_coordinate_boundary_independently() {
    assert_eq!(
        ORACLE_EXACT_CAUSAL_CHECKPOINT_LIMIT,
        EXACT_CAUSAL_CHECKPOINT_LIMIT
    );

    let dim = 64;
    let axis = TemporalAxis::new(dim, 0xD1CE_7003).unwrap();
    let keys = roles(1, dim, 50_000);
    let values = roles(2, dim, 60_000);
    let spans = [OracleSpan {
        key_index: 0,
        value_index: 0,
        start: EXACT_CAUSAL_CHECKPOINT_LIMIT - 2,
        end_exclusive: EXACT_CAUSAL_CHECKPOINT_LIMIT,
    }];
    let memory = build_production_archive(&axis, &keys, &values, &spans);

    assert!(matches!(
        explicit_candidate_scores(
            &axis,
            &keys,
            &values,
            &spans,
            0,
            ORACLE_EXACT_CAUSAL_CHECKPOINT_LIMIT,
        ),
        Err(OracleError::CheckpointOutOfRange { .. })
    ));
    assert!(matches!(
        memory.score_candidate(
            &axis,
            &keys[0],
            &values[0],
            EXACT_CAUSAL_CHECKPOINT_LIMIT,
        ),
        Err(ValidityMemoryError::CheckpointOutOfExactRange { .. })
    ));
}
