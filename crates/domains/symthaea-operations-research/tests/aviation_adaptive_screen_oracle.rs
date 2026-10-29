// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Independent truth-table oracle for the adaptive recovery screen.
//!
//! The oracle below computes the optimum directly from a separate list of
//! ground-truth costs and feasibility values. It does not use the production
//! bound or stopping logic.

use std::collections::BTreeMap;
use symthaea_operations_research::aviation_adaptive_screen::{
    screen_recovery_candidates, AdaptiveScreenOptions, AdaptiveScreenStatus,
    RecoveryCandidateEvaluation, RecoveryScreenCandidate, RecoveryScreenContext,
};

fn context() -> RecoveryScreenContext {
    RecoveryScreenContext {
        scenario_id: "synthetic-recovery-001".into(),
        scenario_sha256: "a".repeat(64),
        candidate_universe_sha256: "d".repeat(64),
        screening_policy_sha256: "b".repeat(64),
        ranking_artifact_sha256: "c".repeat(64),
        screening_source_commit_sha: "1".repeat(40),
        verifier_source_commit_sha: "2".repeat(40),
    }
}

fn candidate(id: &str, ranking_score: i64, lower_bound: u64) -> RecoveryScreenCandidate {
    RecoveryScreenCandidate {
        candidate_id: id.into(),
        ranking_score,
        admissible_lower_bound_cost: lower_bound,
    }
}

fn options(initial: usize, batch: usize, max: usize, gap: u64) -> AdaptiveScreenOptions {
    AdaptiveScreenOptions {
        initial_candidate_count: initial,
        expansion_batch_size: batch,
        max_evaluations: max,
        max_absolute_optimality_gap: gap,
    }
}

/// None means the independent feasibility oracle proved this candidate infeasible.
fn from_truth(
    candidate_id: &str,
    truth: &[(&str, Option<u64>)],
) -> RecoveryCandidateEvaluation {
    match truth.iter().find(|(id, _)| *id == candidate_id) {
        Some((_, Some(cost))) => RecoveryCandidateEvaluation::VerifiedFeasible {
            objective_cost: *cost,
        },
        Some((_, None)) => RecoveryCandidateEvaluation::VerifiedInfeasible {
            reason: "independent tiny-instance oracle: hard constraint violated".into(),
        },
        None => RecoveryCandidateEvaluation::Indeterminate {
            reason: "independent oracle has no fact for this candidate".into(),
        },
    }
}

fn oracle_best(truth: &[(&str, Option<u64>)]) -> Option<(String, u64)> {
    truth
        .iter()
        .filter_map(|(id, cost)| cost.map(|cost| ((*id).to_string(), cost)))
        .min_by(|left, right| left.1.cmp(&right.1).then_with(|| left.0.cmp(&right.0)))
}

#[test]
fn a_lower_bound_can_certify_without_evaluating_every_candidate() {
    let candidates = vec![candidate("candidate-a", 100, 10), candidate("candidate-b", 20, 40)];
    let truth = [("candidate-a", Some(12)), ("candidate-b", Some(50))];
    let oracle = oracle_best(&truth).unwrap();

    let report = screen_recovery_candidates(
        &context(),
        &candidates,
        options(1, 1, 2, 0),
        |candidate| from_truth(&candidate.candidate_id, &truth),
    )
    .unwrap();

    assert_eq!(report.status, AdaptiveScreenStatus::OptimalWithinTolerance);
    assert_eq!(report.best_candidate_id.as_deref(), Some(oracle.0.as_str()));
    assert_eq!(report.best_objective_cost, Some(oracle.1));
    assert_eq!(report.global_lower_bound_cost, Some(12));
    assert_eq!(report.absolute_optimality_gap, Some(0));
    assert_eq!(report.evaluated_candidate_count, 1);
    assert_eq!(report.remaining_candidate_count, 1);
}

#[test]
fn expansion_finds_a_hidden_better_candidate_and_matches_full_oracle() {
    let candidates = vec![
        candidate("first-ranked", 100, 0),
        candidate("better-hidden", 80, 6),
        candidate("expensive", 10, 50),
    ];
    let truth = [
        ("first-ranked", Some(50)),
        ("better-hidden", Some(7)),
        ("expensive", Some(80)),
    ];
    let oracle = oracle_best(&truth).unwrap();

    let report = screen_recovery_candidates(
        &context(),
        &candidates,
        options(1, 1, 3, 0),
        |candidate| from_truth(&candidate.candidate_id, &truth),
    )
    .unwrap();

    assert_eq!(report.status, AdaptiveScreenStatus::OptimalWithinTolerance);
    assert_eq!(report.best_candidate_id.as_deref(), Some(oracle.0.as_str()));
    assert_eq!(report.best_objective_cost, Some(oracle.1));
    assert_eq!(report.evaluated_candidate_count, 2);
    assert_eq!(report.remaining_candidate_count, 1);
}

#[test]
fn budget_exhaustion_never_claims_optimality_from_a_screened_subset() {
    let candidates = vec![candidate("greedy-winner", 100, 0), candidate("hidden-better", 50, 2)];
    let truth = [("greedy-winner", Some(50)), ("hidden-better", Some(7))];

    let report = screen_recovery_candidates(
        &context(),
        &candidates,
        options(1, 1, 1, 0),
        |candidate| from_truth(&candidate.candidate_id, &truth),
    )
    .unwrap();

    assert_eq!(report.status, AdaptiveScreenStatus::FeasibleUncertified);
    assert_eq!(report.best_candidate_id.as_deref(), Some("greedy-winner"));
    assert_eq!(report.global_lower_bound_cost, Some(2));
    assert_eq!(report.absolute_optimality_gap, Some(48));
    assert_eq!(report.remaining_candidate_count, 1);
}

#[test]
fn indeterminate_evidence_prevents_an_infeasibility_claim() {
    let candidates = vec![candidate("unknown", 100, 0), candidate("infeasible", 50, 10)];
    let truth = [("infeasible", None)];

    let report = screen_recovery_candidates(
        &context(),
        &candidates,
        options(2, 1, 2, 0),
        |candidate| {
            if candidate.candidate_id == "unknown" {
                RecoveryCandidateEvaluation::Indeterminate {
                    reason: "independent checker unavailable".into(),
                }
            } else {
                from_truth(&candidate.candidate_id, &truth)
            }
        },
    )
    .unwrap();

    assert_eq!(report.status, AdaptiveScreenStatus::IndeterminateEvidence);
    assert_eq!(report.best_candidate_id, None);
    assert_eq!(report.remaining_candidate_count, 1);
    assert_eq!(report.remaining_candidates[0].candidate_id, "unknown");
}

#[test]
fn exhaustive_conclusive_infeasibility_is_reported_without_a_fake_solution() {
    let candidates = vec![candidate("a", 100, 0), candidate("b", 50, 0)];
    let truth = [("a", None), ("b", None)];
    let report = screen_recovery_candidates(
        &context(),
        &candidates,
        options(1, 1, 2, 0),
        |candidate| from_truth(&candidate.candidate_id, &truth),
    )
    .unwrap();

    assert_eq!(report.status, AdaptiveScreenStatus::NoFeasiblePlan);
    assert_eq!(report.best_candidate_id, None);
    assert_eq!(report.best_objective_cost, None);
    assert_eq!(report.evaluated_candidate_count, 2);
}

#[test]
fn impossible_lower_bound_witness_is_rejected() {
    let candidates = vec![candidate("contradiction", 100, 20)];
    let error = screen_recovery_candidates(
        &context(),
        &candidates,
        options(1, 1, 1, 0),
        |_| RecoveryCandidateEvaluation::VerifiedFeasible { objective_cost: 10 },
    )
    .unwrap_err();

    assert_eq!(
        error,
        symthaea_operations_research::aviation_adaptive_screen::AdaptiveScreenError::LowerBoundExceedsVerifiedCost {
            candidate_id: "contradiction".into(),
            lower_bound: 20,
            objective_cost: 10,
        }
    );
}

#[test]
fn deterministic_order_and_stable_id_ties_do_not_depend_on_input_order() {
    let forward = vec![candidate("z", 10, 1), candidate("b", 10, 1), candidate("a", 10, 1)];
    let reverse = vec![candidate("a", 10, 1), candidate("b", 10, 1), candidate("z", 10, 1)];
    let truth = [("z", Some(8)), ("b", Some(8)), ("a", Some(8))];
    let run = |rows: &[RecoveryScreenCandidate]| {
        screen_recovery_candidates(&context(), rows, options(3, 1, 3, 0), |candidate| {
            from_truth(&candidate.candidate_id, &truth)
        })
        .unwrap()
    };

    let first = run(&forward);
    let second = run(&reverse);
    assert_eq!(first.best_candidate_id.as_deref(), Some("a"));
    assert_eq!(first, second);
}

#[test]
fn duplicate_ids_and_malformed_provenance_fail_closed() {
    let duplicate = vec![candidate("same", 10, 0), candidate("same", 5, 0)];
    assert!(matches!(
        screen_recovery_candidates(&context(), &duplicate, options(1, 1, 2, 0), |_| {
            RecoveryCandidateEvaluation::Indeterminate { reason: "unknown".into() }
        }),
        Err(symthaea_operations_research::aviation_adaptive_screen::AdaptiveScreenError::DuplicateCandidateId { .. })
    ));

    let mut bad_context = context();
    bad_context.ranking_artifact_sha256 = "not-a-sha".into();
    assert!(matches!(
        screen_recovery_candidates(&bad_context, &[candidate("a", 1, 0)], options(1, 1, 1, 0), |_| {
            RecoveryCandidateEvaluation::Indeterminate { reason: "unknown".into() }
        }),
        Err(symthaea_operations_research::aviation_adaptive_screen::AdaptiveScreenError::InvalidDigest { field: "ranking_artifact_sha256" })
    ));
}

#[test]
fn oracle_fixture_is_explicit_and_independent_from_candidate_ranking() {
    // Deliberately rank the truly optimal plan last; oracle truth is not inferred
    // from the screening rank or any production bound helper.
    let truth = BTreeMap::from([
        ("ranked-first".to_string(), Some(30u64)),
        ("ranked-last".to_string(), Some(4u64)),
        ("infeasible".to_string(), None),
    ]);
    let rows = vec![
        candidate("ranked-first", 100, 0),
        candidate("infeasible", 60, 0),
        candidate("ranked-last", 1, 4),
    ];
    let truth_rows: Vec<(&str, Option<u64>)> = truth
        .iter()
        .map(|(id, value)| (id.as_str(), *value))
        .collect();

    let report = screen_recovery_candidates(
        &context(),
        &rows,
        options(1, 2, 3, 0),
        |candidate| from_truth(&candidate.candidate_id, &truth_rows),
    )
    .unwrap();

    assert_eq!(report.best_candidate_id.as_deref(), Some("ranked-last"));
    assert_eq!(report.best_objective_cost, Some(4));
    assert_eq!(report.status, AdaptiveScreenStatus::OptimalWithinTolerance);
}
