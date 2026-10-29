// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Certified adaptive screening for finite recovery candidate sets.
//!
//! The caller supplies a frozen finite candidate set, deterministic ranking
//! priorities, and a lower bound for every candidate. The evaluation callback
//! must return an independently checked exact feasible cost, an independently
//! checked infeasibility result, or an explicit indeterminate result.
//!
//! A candidate being best among the evaluated subset is NOT an optimality
//! certificate. The report may say `OptimalWithinTolerance` only when the best
//! verified feasible objective is within the declared absolute gap of the
//! minimum admissible lower bound over every unevaluated or indeterminate
//! candidate. These lower bounds are supplied by the caller; this module does
//! not prove their admissibility and is not an aircraft dispatch authority.

/// Frozen run identity. Digests are caller-computed SHA-256 values; this module
/// validates their encoding but does not hash external files or authenticate
/// the source of the values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryScreenContext {
    pub scenario_id: String,
    pub scenario_sha256: String,
    pub screening_policy_sha256: String,
    pub ranking_artifact_sha256: String,
    pub screening_source_commit_sha: String,
    pub verifier_source_commit_sha: String,
}

/// One member of the frozen finite candidate universe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryScreenCandidate {
    /// Stable ID, independent of input vector position.
    pub candidate_id: String,
    /// Larger scores are evaluated first. The score is a ranking hint only.
    pub ranking_score: i64,
    /// Admissible lower bound on the best feasible objective represented by
    /// this candidate. The caller must justify this bound independently.
    pub admissible_lower_bound_cost: u64,
}

/// Result of evaluating one candidate. The "verified" variants are contracts:
/// the callback must independently check feasibility or infeasibility and must
/// recompute exact objective cost from the frozen scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryCandidateEvaluation {
    /// The candidate passed the independent feasibility check and has this
    /// recomputed non-negative integer objective cost.
    VerifiedFeasible { objective_cost: u64 },
    /// The candidate was conclusively shown infeasible under the frozen model.
    VerifiedInfeasible { reason: String },
    /// Available evidence could not establish feasibility or infeasibility.
    Indeterminate { reason: String },
}

/// Deterministic adaptive expansion policy. Costs and the accepted gap use the
/// same integer units as the declared objective; no percentage is inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveScreenOptions {
    pub initial_candidate_count: usize,
    pub expansion_batch_size: usize,
    pub max_evaluations: usize,
    pub max_absolute_optimality_gap: u64,
}

/// Honest terminal state of a screening run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveScreenStatus {
    /// A verified feasible candidate is within the configured gap of the
    /// global lower bound over the entire frozen candidate universe.
    OptimalWithinTolerance,
    /// A feasible candidate exists, but the declared bound does not certify the
    /// configured gap (or indeterminate evidence remains consequential).
    FeasibleUncertified,
    /// All candidates were conclusively evaluated and none is feasible.
    NoFeasiblePlan,
    /// The evaluation budget ended before feasibility could be established.
    NoFeasiblePlanWithinBudget,
    /// No feasible plan was established and at least one candidate evaluation
    /// is indeterminate, so infeasibility cannot be claimed.
    IndeterminateEvidence,
}

/// One evaluated candidate, retained in the audit report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryCandidateAudit {
    pub candidate_id: String,
    pub ranking_score: i64,
    pub admissible_lower_bound_cost: u64,
    pub evaluation: RecoveryCandidateEvaluation,
}

/// Candidate that may still contain a better feasible plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecoveryCandidateRemaining {
    pub candidate_id: String,
    pub ranking_score: i64,
    pub admissible_lower_bound_cost: u64,
    pub reason: RemainingCandidateReason,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RemainingCandidateReason {
    NotEvaluated,
    Indeterminate { reason: String },
}

/// Complete report. `global_lower_bound_cost` is absent when no feasible
/// candidate is known and no meaningful feasible objective bound can be
/// reported. The report records all candidates evaluated and all candidates
/// still capable of improving the best known feasible objective.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdaptiveScreenReport {
    pub scenario_id: String,
    pub scenario_sha256: String,
    pub screening_policy_sha256: String,
    pub ranking_artifact_sha256: String,
    pub screening_source_commit_sha: String,
    pub verifier_source_commit_sha: String,
    pub status: AdaptiveScreenStatus,
    pub best_candidate_id: Option<String>,
    pub best_objective_cost: Option<u64>,
    pub global_lower_bound_cost: Option<u64>,
    pub absolute_optimality_gap: Option<u64>,
    pub accepted_absolute_gap: u64,
    pub full_candidate_count: usize,
    pub evaluated_candidate_count: usize,
    pub remaining_candidate_count: usize,
    pub evaluated_candidates: Vec<RecoveryCandidateAudit>,
    pub remaining_candidates: Vec<RecoveryCandidateRemaining>,
}

/// Fail-closed validation errors. A bad bound witness is rejected when it
/// contradicts an independently recomputed feasible objective.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdaptiveScreenError {
    EmptyCandidateSet,
    InvalidIdentity { field: &'static str },
    InvalidDigest { field: &'static str },
    DuplicateCandidateId { candidate_id: String },
    InvalidOptions,
    EmptyEvaluationReason { candidate_id: String },
    LowerBoundExceedsVerifiedCost {
        candidate_id: String,
        lower_bound: u64,
        objective_cost: u64,
    },
}

/// Rank candidates, evaluate them in deterministic batches, and stop as soon as
/// the verified feasible solution is certified within the declared absolute
/// gap. Ties in the screen priority use the stable candidate ID; ties in
/// objective cost use the stable candidate ID as well.
///
/// The callback is invoked at most once for each candidate. Candidate IDs and
/// lower bounds for not-yet-evaluated candidates are retained in the report.
/// An `Indeterminate` result is not retried automatically; its lower bound
/// remains in the global bound until a future run supplies better evidence.
pub fn screen_recovery_candidates<F>(
    context: &RecoveryScreenContext,
    candidates: &[RecoveryScreenCandidate],
    options: AdaptiveScreenOptions,
    mut evaluate: F,
) -> Result<AdaptiveScreenReport, AdaptiveScreenError>
where
    F: FnMut(&RecoveryScreenCandidate) -> RecoveryCandidateEvaluation,
{
    validate_context(context)?;
    if candidates.is_empty() {
        return Err(AdaptiveScreenError::EmptyCandidateSet);
    }
    if options.initial_candidate_count == 0
        || options.expansion_batch_size == 0
        || options.max_evaluations == 0
    {
        return Err(AdaptiveScreenError::InvalidOptions);
    }

    let mut ordered = candidates.to_vec();
    for candidate in &ordered {
        if candidate.candidate_id.trim().is_empty() {
            return Err(AdaptiveScreenError::InvalidIdentity {
                field: "candidate_id",
            });
        }
    }
    ordered.sort_by(|left, right| {
        right
            .ranking_score
            .cmp(&left.ranking_score)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });
    for index in 0..ordered.len() {
        if ordered[..index]
            .iter()
            .any(|prior| prior.candidate_id == ordered[index].candidate_id)
        {
            return Err(AdaptiveScreenError::DuplicateCandidateId {
                candidate_id: ordered[index].candidate_id.clone(),
            });
        }
    }

    let mut evaluations: Vec<Option<RecoveryCandidateEvaluation>> =
        vec![None; ordered.len()];
    let limit = options.max_evaluations.min(ordered.len());
    let mut next_target = options.initial_candidate_count.min(limit);
    let mut next_index = 0usize;

    loop {
        while next_index < next_target {
            let candidate = &ordered[next_index];
            let result = evaluate(candidate);
            match &result {
                RecoveryCandidateEvaluation::VerifiedFeasible { objective_cost }
                    if *objective_cost < candidate.admissible_lower_bound_cost =>
                {
                    return Err(AdaptiveScreenError::LowerBoundExceedsVerifiedCost {
                        candidate_id: candidate.candidate_id.clone(),
                        lower_bound: candidate.admissible_lower_bound_cost,
                        objective_cost: *objective_cost,
                    });
                }
                RecoveryCandidateEvaluation::VerifiedInfeasible { reason }
                | RecoveryCandidateEvaluation::Indeterminate { reason }
                    if reason.trim().is_empty() =>
                {
                    return Err(AdaptiveScreenError::EmptyEvaluationReason {
                        candidate_id: candidate.candidate_id.clone(),
                    });
                }
                _ => {}
            }
            evaluations[next_index] = Some(result);
            next_index += 1;
        }

        let (best, lower_bound, gap) = summarize_bounds(&ordered, &evaluations);
        if best.is_some() && gap.is_some_and(|value| value <= options.max_absolute_optimality_gap)
        {
            break;
        }
        if next_index >= ordered.len() || next_index >= limit {
            break;
        }
        next_target = next_index
            .saturating_add(options.expansion_batch_size)
            .min(limit)
            .min(ordered.len());
    }

    let (best, lower_bound, gap) = summarize_bounds(&ordered, &evaluations);
    let remaining_candidates: Vec<RecoveryCandidateRemaining> = ordered
        .iter()
        .zip(evaluations.iter())
        .filter_map(|(candidate, evaluation)| match evaluation {
            None => Some(RecoveryCandidateRemaining {
                candidate_id: candidate.candidate_id.clone(),
                ranking_score: candidate.ranking_score,
                admissible_lower_bound_cost: candidate.admissible_lower_bound_cost,
                reason: RemainingCandidateReason::NotEvaluated,
            }),
            Some(RecoveryCandidateEvaluation::Indeterminate { reason }) => {
                Some(RecoveryCandidateRemaining {
                    candidate_id: candidate.candidate_id.clone(),
                    ranking_score: candidate.ranking_score,
                    admissible_lower_bound_cost: candidate.admissible_lower_bound_cost,
                    reason: RemainingCandidateReason::Indeterminate {
                        reason: reason.clone(),
                    },
                })
            }
            Some(RecoveryCandidateEvaluation::VerifiedFeasible { .. })
            | Some(RecoveryCandidateEvaluation::VerifiedInfeasible { .. }) => None,
        })
        .collect();

    let indeterminate_remains = remaining_candidates
        .iter()
        .any(|candidate| matches!(candidate.reason, RemainingCandidateReason::Indeterminate { .. }));
    let unevaluated_remains = remaining_candidates
        .iter()
        .any(|candidate| matches!(candidate.reason, RemainingCandidateReason::NotEvaluated));

    let status = match (best.as_ref(), gap) {
        (Some(_), Some(value)) if value <= options.max_absolute_optimality_gap => {
            AdaptiveScreenStatus::OptimalWithinTolerance
        }
        (Some(_), _) => AdaptiveScreenStatus::FeasibleUncertified,
        (None, _) if indeterminate_remains => AdaptiveScreenStatus::IndeterminateEvidence,
        (None, _) if unevaluated_remains => AdaptiveScreenStatus::NoFeasiblePlanWithinBudget,
        (None, _) => AdaptiveScreenStatus::NoFeasiblePlan,
    };

    let evaluated_candidates: Vec<RecoveryCandidateAudit> = ordered
        .iter()
        .zip(evaluations.iter())
        .filter_map(|(candidate, evaluation)| {
            evaluation.as_ref().map(|evaluation| RecoveryCandidateAudit {
                candidate_id: candidate.candidate_id.clone(),
                ranking_score: candidate.ranking_score,
                admissible_lower_bound_cost: candidate.admissible_lower_bound_cost,
                evaluation: evaluation.clone(),
            })
        })
        .collect();

    Ok(AdaptiveScreenReport {
        scenario_id: context.scenario_id.clone(),
        scenario_sha256: context.scenario_sha256.clone(),
        screening_policy_sha256: context.screening_policy_sha256.clone(),
        ranking_artifact_sha256: context.ranking_artifact_sha256.clone(),
        screening_source_commit_sha: context.screening_source_commit_sha.clone(),
        verifier_source_commit_sha: context.verifier_source_commit_sha.clone(),
        status,
        best_candidate_id: best.as_ref().map(|(candidate_id, _)| candidate_id.clone()),
        best_objective_cost: best.as_ref().map(|(_, cost)| *cost),
        global_lower_bound_cost: lower_bound,
        absolute_optimality_gap: gap,
        accepted_absolute_gap: options.max_absolute_optimality_gap,
        full_candidate_count: ordered.len(),
        evaluated_candidate_count: evaluated_candidates.len(),
        remaining_candidate_count: remaining_candidates.len(),
        evaluated_candidates,
        remaining_candidates,
    })
}

fn summarize_bounds(
    candidates: &[RecoveryScreenCandidate],
    evaluations: &[Option<RecoveryCandidateEvaluation>],
) -> (Option<(String, u64)>, Option<u64>, Option<u64>) {
    let mut best: Option<(String, u64)> = None;
    let mut unresolved_lower_bound: Option<u64> = None;
    for (candidate, evaluation) in candidates.iter().zip(evaluations.iter()) {
        match evaluation {
            Some(RecoveryCandidateEvaluation::VerifiedFeasible { objective_cost }) => {
                let replacement = (candidate.candidate_id.clone(), *objective_cost);
                if best.as_ref().is_none_or(|current| {
                    replacement.1 < current.1
                        || (replacement.1 == current.1 && replacement.0 < current.0)
                }) {
                    best = Some(replacement);
                }
            }
            Some(RecoveryCandidateEvaluation::VerifiedInfeasible { .. }) => {}
            Some(RecoveryCandidateEvaluation::Indeterminate { .. }) | None => {
                unresolved_lower_bound = Some(
                    unresolved_lower_bound
                        .map_or(candidate.admissible_lower_bound_cost, |current| {
                            current.min(candidate.admissible_lower_bound_cost)
                        }),
                );
            }
        }
    }

    match best.as_ref() {
        Some((_, best_cost)) => {
            let lower_bound = unresolved_lower_bound.map_or(*best_cost, |bound| bound.min(*best_cost));
            (best, Some(lower_bound), Some(best_cost - lower_bound))
        }
        None => (best, unresolved_lower_bound, None),
    }
}

fn validate_context(context: &RecoveryScreenContext) -> Result<(), AdaptiveScreenError> {
    if context.scenario_id.trim().is_empty() {
        return Err(AdaptiveScreenError::InvalidIdentity { field: "scenario_id" });
    }
    for (field, value) in [
        ("scenario_sha256", context.scenario_sha256.as_str()),
        ("screening_policy_sha256", context.screening_policy_sha256.as_str()),
        ("ranking_artifact_sha256", context.ranking_artifact_sha256.as_str()),
    ] {
        if !is_lower_hex_digest(value, 64) {
            return Err(AdaptiveScreenError::InvalidDigest { field });
        }
    }
    for (field, value) in [
        ("screening_source_commit_sha", context.screening_source_commit_sha.as_str()),
        ("verifier_source_commit_sha", context.verifier_source_commit_sha.as_str()),
    ] {
        if !(is_lower_hex_digest(value, 40) || is_lower_hex_digest(value, 64)) {
            return Err(AdaptiveScreenError::InvalidDigest { field });
        }
    }
    Ok(())
}

fn is_lower_hex_digest(value: &str, required_len: usize) -> bool {
    value.len() == required_len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
