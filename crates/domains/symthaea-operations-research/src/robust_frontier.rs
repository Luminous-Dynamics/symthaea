// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Evidence-aware robust multi-objective frontier analysis.
//!
//! This is an analysis primitive, not an optimizer or decision authority. It
//! consumes externally evaluated objective intervals and an externally supplied
//! feasibility disposition. It does not generate uncertainty bounds, validate
//! source artifacts, model objective correlations, or establish engineering
//! safety/qualification.

use std::collections::{BTreeMap, BTreeSet};

/// Pairwise comparison is quadratic in the number of candidates. Keep the
/// primitive bounded; larger studies need a separately verified algorithm.
pub const MAX_FRONTIER_CANDIDATES: usize = 2_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ObjectiveDirection {
    Minimize,
    Maximize,
}

/// What the supplied interval is intended to mean. The enum records semantics;
/// it does not prove that the interval-generating method is calibrated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntervalSemantics {
    EngineeringBound,
    MeasurementUncertainty,
    ConfidenceInterval,
    CredibleInterval,
    ScenarioEnvelope,
    SensitivityEnvelope,
}

/// A versioned objective contract shared by every candidate in a study.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectiveSpec {
    pub objective_id: String,
    pub direction: ObjectiveDirection,
    pub unit: String,
    pub reference_conditions_id: String,
    pub interval_semantics: IntervalSemantics,
    pub interval_method_id: String,
}

/// Closed finite interval in native metric units. The caller supplies a
/// source/evidence reference; this utility does not resolve or authenticate it.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectiveInterval {
    pub objective_id: String,
    pub lower: f64,
    pub upper: f64,
    pub unit: String,
    pub reference_conditions_id: String,
    pub interval_semantics: IntervalSemantics,
    pub interval_method_id: String,
    pub evidence_ref: String,
}

/// A missing value is explicit and reasoned. It is never imputed to zero,
/// a midpoint, a favorable bound, or another candidate's value.
#[derive(Clone, Debug, PartialEq)]
pub enum ObjectiveObservation {
    Interval(ObjectiveInterval),
    Missing {
        objective_id: String,
        reason: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum CandidateStatus {
    /// Caller asserts that the declared hard constraints pass with the
    /// referenced evidence. The status is not independently verified here.
    Feasible,
    Infeasible,
    Indeterminate,
    EvaluationFailed,
}

/// Candidate-level record. Non-feasible statuses remain in the report but do
/// not enter robust or midpoint frontiers.
#[derive(Clone, Debug, PartialEq)]
pub struct CandidateAssessment {
    pub candidate_id: String,
    pub status: CandidateStatus,
    pub status_evidence_ref: String,
    /// Exactly one observation per objective, in the declared objective order.
    pub objectives: Vec<ObjectiveObservation>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FrontierValidationError {
    EmptyStudyId,
    EmptyObjectiveSet,
    DuplicateObjectiveId(String),
    EmptyObjectiveField { objective_id: String, field: &'static str },
    EmptyCandidateSet,
    TooManyCandidates { requested: usize, maximum: usize },
    EmptyCandidateId,
    DuplicateCandidateId(String),
    EmptyStatusEvidenceRef { candidate_id: String },
    ObjectiveCountMismatch {
        candidate_id: String,
        expected: usize,
        actual: usize,
    },
    ObjectiveOrderMismatch {
        candidate_id: String,
        expected: String,
        actual: String,
    },
    MissingObjectiveReason { candidate_id: String, objective_id: String },
    MissingObjectiveForFeasibleCandidate { candidate_id: String, objective_id: String },
    EmptyMetricField {
        candidate_id: String,
        objective_id: String,
        field: &'static str,
    },
    NonFiniteBound { candidate_id: String, objective_id: String },
    ReversedInterval { candidate_id: String, objective_id: String },
    IntervalWidthOverflow { candidate_id: String, objective_id: String },
    UnitMismatch { candidate_id: String, objective_id: String },
    ReferenceConditionsMismatch { candidate_id: String, objective_id: String },
    IntervalSemanticsMismatch { candidate_id: String, objective_id: String },
    IntervalMethodMismatch { candidate_id: String, objective_id: String },
}

/// A directional relation established by interval arithmetic. Possible-only
/// means some values inside the declared intervals permit dominance; it does
/// not imply that the values are likely or statistically independent.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct DominancePair {
    pub dominator_id: String,
    pub dominated_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateFrontierResult {
    pub candidate_id: String,
    pub status: CandidateStatus,
    pub missing_objective_ids: Vec<String>,
    pub robustly_dominated_by: Vec<String>,
    pub possible_only_dominators: Vec<String>,
    pub on_robust_frontier: bool,
    pub on_unambiguous_frontier: bool,
    pub on_ambiguous_frontier: bool,
    pub on_midpoint_reference_frontier: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ObjectiveCoverage {
    pub objective_id: String,
    pub candidate_count: usize,
    pub observed_interval_count: usize,
    pub missing_interval_count: usize,
    pub mean_interval_width: Option<f64>,
    pub maximum_interval_width: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FrontierCounts {
    pub candidate_count: usize,
    pub feasible_count: usize,
    pub infeasible_count: usize,
    pub indeterminate_count: usize,
    pub evaluation_failed_count: usize,
    pub missing_objective_observation_count: usize,
    pub robust_frontier_count: usize,
    pub unambiguous_frontier_count: usize,
    pub ambiguous_frontier_count: usize,
    pub robust_dominance_pair_count: usize,
    pub possible_only_dominance_pair_count: usize,
    pub midpoint_frontier_disagreement_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RobustFrontierReport {
    pub study_id: String,
    pub candidates: Vec<CandidateFrontierResult>,
    /// Feasible candidates not demonstrably dominated under the declared bounds.
    pub robust_frontier_ids: Vec<String>,
    /// Robust-frontier candidates with no possible-only incoming dominance.
    /// This is a stricter interval-stability diagnostic, not a chosen design.
    pub unambiguous_frontier_ids: Vec<String>,
    /// Robust-frontier candidates that at least one other candidate could
    /// dominate for some interval values, though dominance is not established.
    pub ambiguous_frontier_ids: Vec<String>,
    /// Ordinary Pareto frontier computed from interval midpoints. Diagnostic
    /// only: a midpoint is not asserted to be a mean or most-likely value.
    pub midpoint_reference_frontier_ids: Vec<String>,
    pub midpoint_frontier_disagreements: Vec<String>,
    pub robust_dominance_pairs: Vec<DominancePair>,
    pub possible_only_dominance_pairs: Vec<DominancePair>,
    pub objective_coverage: Vec<ObjectiveCoverage>,
    pub counts: FrontierCounts,
}

fn nonempty(value: &str) -> bool {
    !value.trim().is_empty()
}

fn validate_plan(study_id: &str, objectives: &[ObjectiveSpec]) -> Vec<FrontierValidationError> {
    let mut errors = Vec::new();
    if !nonempty(study_id) {
        errors.push(FrontierValidationError::EmptyStudyId);
    }
    if objectives.is_empty() {
        errors.push(FrontierValidationError::EmptyObjectiveSet);
    }

    let mut ids = BTreeSet::new();
    for objective in objectives {
        if !nonempty(&objective.objective_id) {
            errors.push(FrontierValidationError::EmptyObjectiveField {
                objective_id: objective.objective_id.clone(),
                field: "objective_id",
            });
        }
        if !ids.insert(objective.objective_id.clone()) {
            errors.push(FrontierValidationError::DuplicateObjectiveId(
                objective.objective_id.clone(),
            ));
        }
        for (field, value) in [
            ("unit", objective.unit.as_str()),
            ("reference_conditions_id", objective.reference_conditions_id.as_str()),
            ("interval_method_id", objective.interval_method_id.as_str()),
        ] {
            if !nonempty(value) {
                errors.push(FrontierValidationError::EmptyObjectiveField {
                    objective_id: objective.objective_id.clone(),
                    field,
                });
            }
        }
    }
    errors
}

fn validate_interval(
    candidate_id: &str,
    spec: &ObjectiveSpec,
    interval: &ObjectiveInterval,
    errors: &mut Vec<FrontierValidationError>,
) {
    if interval.objective_id != spec.objective_id {
        errors.push(FrontierValidationError::ObjectiveOrderMismatch {
            candidate_id: candidate_id.to_owned(),
            expected: spec.objective_id.clone(),
            actual: interval.objective_id.clone(),
        });
    }
    for (field, value) in [
        ("unit", interval.unit.as_str()),
        ("reference_conditions_id", interval.reference_conditions_id.as_str()),
        ("interval_method_id", interval.interval_method_id.as_str()),
        ("evidence_ref", interval.evidence_ref.as_str()),
    ] {
        if !nonempty(value) {
            errors.push(FrontierValidationError::EmptyMetricField {
                candidate_id: candidate_id.to_owned(),
                objective_id: spec.objective_id.clone(),
                field,
            });
        }
    }
    if !interval.lower.is_finite() || !interval.upper.is_finite() {
        errors.push(FrontierValidationError::NonFiniteBound {
            candidate_id: candidate_id.to_owned(),
            objective_id: spec.objective_id.clone(),
        });
    } else if interval.lower > interval.upper {
        errors.push(FrontierValidationError::ReversedInterval {
            candidate_id: candidate_id.to_owned(),
            objective_id: spec.objective_id.clone(),
        });
    } else if !(interval.upper - interval.lower).is_finite() {
        errors.push(FrontierValidationError::IntervalWidthOverflow {
            candidate_id: candidate_id.to_owned(),
            objective_id: spec.objective_id.clone(),
        });
    }

    if interval.unit != spec.unit {
        errors.push(FrontierValidationError::UnitMismatch {
            candidate_id: candidate_id.to_owned(),
            objective_id: spec.objective_id.clone(),
        });
    }
    if interval.reference_conditions_id != spec.reference_conditions_id {
        errors.push(FrontierValidationError::ReferenceConditionsMismatch {
            candidate_id: candidate_id.to_owned(),
            objective_id: spec.objective_id.clone(),
        });
    }
    if interval.interval_semantics != spec.interval_semantics {
        errors.push(FrontierValidationError::IntervalSemanticsMismatch {
            candidate_id: candidate_id.to_owned(),
            objective_id: spec.objective_id.clone(),
        });
    }
    if interval.interval_method_id != spec.interval_method_id {
        errors.push(FrontierValidationError::IntervalMethodMismatch {
            candidate_id: candidate_id.to_owned(),
            objective_id: spec.objective_id.clone(),
        });
    }
}

/// Validate all records before any pairwise relation is computed. Feasible
/// candidates require complete objective intervals. Non-feasible rows may have
/// explicitly reasoned missing objectives, and remain visible in output.
fn validate_candidates(
    objectives: &[ObjectiveSpec],
    candidates: &[CandidateAssessment],
) -> Vec<FrontierValidationError> {
    let mut errors = Vec::new();
    if candidates.is_empty() {
        errors.push(FrontierValidationError::EmptyCandidateSet);
        return errors;
    }
    if candidates.len() > MAX_FRONTIER_CANDIDATES {
        errors.push(FrontierValidationError::TooManyCandidates {
            requested: candidates.len(),
            maximum: MAX_FRONTIER_CANDIDATES,
        });
        return errors;
    }

    let mut candidate_ids = BTreeSet::new();
    for candidate in candidates {
        if !nonempty(&candidate.candidate_id) {
            errors.push(FrontierValidationError::EmptyCandidateId);
        }
        if !candidate_ids.insert(candidate.candidate_id.clone()) {
            errors.push(FrontierValidationError::DuplicateCandidateId(
                candidate.candidate_id.clone(),
            ));
        }
        if !nonempty(&candidate.status_evidence_ref) {
            errors.push(FrontierValidationError::EmptyStatusEvidenceRef {
                candidate_id: candidate.candidate_id.clone(),
            });
        }
        if candidate.objectives.len() != objectives.len() {
            errors.push(FrontierValidationError::ObjectiveCountMismatch {
                candidate_id: candidate.candidate_id.clone(),
                expected: objectives.len(),
                actual: candidate.objectives.len(),
            });
            // Do not index an incomplete vector after reporting it.
            continue;
        }

        for (spec, observation) in objectives.iter().zip(&candidate.objectives) {
            match observation {
                ObjectiveObservation::Interval(interval) => {
                    validate_interval(&candidate.candidate_id, spec, interval, &mut errors);
                }
                ObjectiveObservation::Missing { objective_id, reason } => {
                    if objective_id != &spec.objective_id {
                        errors.push(FrontierValidationError::ObjectiveOrderMismatch {
                            candidate_id: candidate.candidate_id.clone(),
                            expected: spec.objective_id.clone(),
                            actual: objective_id.clone(),
                        });
                    }
                    if !nonempty(reason) {
                        errors.push(FrontierValidationError::MissingObjectiveReason {
                            candidate_id: candidate.candidate_id.clone(),
                            objective_id: spec.objective_id.clone(),
                        });
                    }
                    if candidate.status == CandidateStatus::Feasible {
                        errors.push(FrontierValidationError::MissingObjectiveForFeasibleCandidate {
                            candidate_id: candidate.candidate_id.clone(),
                            objective_id: spec.objective_id.clone(),
                        });
                    }
                }
            }
        }
    }
    errors
}

/// Convert the native objective interval into a loss interval. All relation
/// functions can then use one minimization convention without mixing units or
/// collapsing the objective vector into a scalar.
fn loss_bounds(spec: &ObjectiveSpec, interval: &ObjectiveInterval) -> (f64, f64) {
    match spec.direction {
        ObjectiveDirection::Minimize => (interval.lower, interval.upper),
        ObjectiveDirection::Maximize => (-interval.upper, -interval.lower),
    }
}

fn robustly_dominates(
    objectives: &[ObjectiveSpec],
    a: &CandidateAssessment,
    b: &CandidateAssessment,
) -> bool {
    let mut strictly_better = false;
    for ((spec, a_observation), b_observation) in objectives
        .iter()
        .zip(&a.objectives)
        .zip(&b.objectives)
    {
        let (ObjectiveObservation::Interval(a_interval), ObjectiveObservation::Interval(b_interval)) =
            (a_observation, b_observation)
        else {
            return false;
        };
        let (a_lower, a_upper) = loss_bounds(spec, a_interval);
        let (b_lower, b_upper) = loss_bounds(spec, b_interval);
        if a_upper > b_lower {
            return false;
        }
        strictly_better |= a_upper < b_lower;
    }
    strictly_better
}

/// True when some independent selections inside each objective's intervals
/// could make a no-worse/strictly-better relation. Correlated uncertainty may
/// make such a combination impossible; this relation is a conservative
/// possibility flag, not a probability or joint-scenario calculation.
fn possibly_dominates(
    objectives: &[ObjectiveSpec],
    a: &CandidateAssessment,
    b: &CandidateAssessment,
) -> bool {
    let mut potentially_strict = false;
    for ((spec, a_observation), b_observation) in objectives
        .iter()
        .zip(&a.objectives)
        .zip(&b.objectives)
    {
        let (ObjectiveObservation::Interval(a_interval), ObjectiveObservation::Interval(b_interval)) =
            (a_observation, b_observation)
        else {
            return false;
        };
        let (a_lower, _) = loss_bounds(spec, a_interval);
        let (_, b_upper) = loss_bounds(spec, b_interval);
        if a_lower > b_upper {
            return false;
        }
        potentially_strict |= a_lower < b_upper;
    }
    potentially_strict
}

fn midpoint_value(interval: &ObjectiveInterval) -> f64 {
    interval.lower + (interval.upper - interval.lower) / 2.0
}

fn midpoint_dominates(
    objectives: &[ObjectiveSpec],
    a: &CandidateAssessment,
    b: &CandidateAssessment,
) -> bool {
    let mut strictly_better = false;
    for ((spec, a_observation), b_observation) in objectives
        .iter()
        .zip(&a.objectives)
        .zip(&b.objectives)
    {
        let (ObjectiveObservation::Interval(a_interval), ObjectiveObservation::Interval(b_interval)) =
            (a_observation, b_observation)
        else {
            return false;
        };
        let a_midpoint = midpoint_value(a_interval);
        let b_midpoint = midpoint_value(b_interval);
        let (a_loss, b_loss) = match spec.direction {
            ObjectiveDirection::Minimize => (a_midpoint, b_midpoint),
            ObjectiveDirection::Maximize => (-a_midpoint, -b_midpoint),
        };
        if a_loss > b_loss {
            return false;
        }
        strictly_better |= a_loss < b_loss;
    }
    strictly_better
}

fn coverage_for_objective(
    objective: &ObjectiveSpec,
    candidates: &[CandidateAssessment],
) -> ObjectiveCoverage {
    let mut widths = Vec::new();
    for candidate in candidates {
        if let Some(ObjectiveObservation::Interval(interval)) =
            candidate.objectives.iter().find(|observation| match observation {
                ObjectiveObservation::Interval(interval) => interval.objective_id == objective.objective_id,
                ObjectiveObservation::Missing { objective_id, .. } => objective_id == &objective.objective_id,
            })
        {
            widths.push(interval.upper - interval.lower);
        }
    }
    let mean_interval_width = if widths.is_empty() {
        None
    } else {
        Some(widths.iter().sum::<f64>() / widths.len() as f64)
    };
    let maximum_interval_width = widths.iter().copied().reduce(f64::max);
    ObjectiveCoverage {
        objective_id: objective.objective_id.clone(),
        candidate_count: candidates.len(),
        observed_interval_count: widths.len(),
        missing_interval_count: candidates.len() - widths.len(),
        mean_interval_width,
        maximum_interval_width,
    }
}

/// Compute an interval-aware robust frontier from pre-evaluated candidates.
///
/// Only status Feasible enters either frontier. All input candidates remain in
/// the report. Robust dominance requires A's worst loss bound to be no worse
/// than B's best loss bound for every objective and strictly better on at least
/// one. Possible-only dominance is reported separately and never removes a
/// candidate from the robust frontier.
pub fn analyze_robust_frontier(
    study_id: &str,
    objectives: &[ObjectiveSpec],
    candidates: &[CandidateAssessment],
) -> Result<RobustFrontierReport, Vec<FrontierValidationError>> {
    let mut errors = validate_plan(study_id, objectives);
    if errors.is_empty() {
        errors.extend(validate_candidates(objectives, candidates));
    } else {
        // Still report an empty candidate-set error when the plan is bad too.
        if candidates.is_empty() {
            errors.push(FrontierValidationError::EmptyCandidateSet);
        }
    }
    if !errors.is_empty() {
        return Err(errors);
    }

    // Canonicalize candidate order before calculations for deterministic output.
    let mut canonical_candidates = candidates.to_vec();
    canonical_candidates.sort_by(|a, b| a.candidate_id.cmp(&b.candidate_id));
    let feasible: Vec<&CandidateAssessment> = canonical_candidates
        .iter()
        .filter(|candidate| candidate.status == CandidateStatus::Feasible)
        .collect();

    let mut robust_dominance_pairs = Vec::new();
    let mut possible_only_dominance_pairs = Vec::new();
    for a in &feasible {
        for b in &feasible {
            if a.candidate_id == b.candidate_id {
                continue;
            }
            if robustly_dominates(objectives, a, b) {
                robust_dominance_pairs.push(DominancePair {
                    dominator_id: a.candidate_id.clone(),
                    dominated_id: b.candidate_id.clone(),
                });
            } else if possibly_dominates(objectives, a, b) {
                possible_only_dominance_pairs.push(DominancePair {
                    dominator_id: a.candidate_id.clone(),
                    dominated_id: b.candidate_id.clone(),
                });
            }
        }
    }
    robust_dominance_pairs.sort();
    possible_only_dominance_pairs.sort();

    let robustly_dominated: BTreeSet<String> = robust_dominance_pairs
        .iter()
        .map(|pair| pair.dominated_id.clone())
        .collect();
    let possible_incoming: BTreeSet<String> = possible_only_dominance_pairs
        .iter()
        .map(|pair| pair.dominated_id.clone())
        .collect();

    let robust_frontier_ids: Vec<String> = feasible
        .iter()
        .filter(|candidate| !robustly_dominated.contains(&candidate.candidate_id))
        .map(|candidate| candidate.candidate_id.clone())
        .collect();
    let unambiguous_frontier_ids: Vec<String> = robust_frontier_ids
        .iter()
        .filter(|candidate_id| !possible_incoming.contains(*candidate_id))
        .cloned()
        .collect();
    let ambiguous_frontier_ids: Vec<String> = robust_frontier_ids
        .iter()
        .filter(|candidate_id| possible_incoming.contains(*candidate_id))
        .cloned()
        .collect();

    let midpoint_frontier_ids: Vec<String> = feasible
        .iter()
        .filter(|candidate| {
            !feasible.iter().any(|other| {
                other.candidate_id != candidate.candidate_id
                    && midpoint_dominates(objectives, other, candidate)
            })
        })
        .map(|candidate| candidate.candidate_id.clone())
        .collect();
    let robust_set: BTreeSet<&str> = robust_frontier_ids.iter().map(String::as_str).collect();
    let midpoint_set: BTreeSet<&str> = midpoint_frontier_ids.iter().map(String::as_str).collect();
    let midpoint_frontier_disagreements: Vec<String> = robust_set
        .symmetric_difference(&midpoint_set)
        .map(|id| (*id).to_owned())
        .collect();

    let mut candidate_results = Vec::with_capacity(canonical_candidates.len());
    for candidate in &canonical_candidates {
        let missing_objective_ids: Vec<String> = candidate.objectives.iter().filter_map(|observation| {
            match observation {
                ObjectiveObservation::Missing { objective_id, .. } => Some(objective_id.clone()),
                ObjectiveObservation::Interval(_) => None,
            }
        }).collect();
        let robustly_dominated_by: Vec<String> = robust_dominance_pairs
            .iter()
            .filter(|pair| pair.dominated_id == candidate.candidate_id)
            .map(|pair| pair.dominator_id.clone())
            .collect();
        let possible_only_dominators: Vec<String> = possible_only_dominance_pairs
            .iter()
            .filter(|pair| pair.dominated_id == candidate.candidate_id)
            .map(|pair| pair.dominator_id.clone())
            .collect();
        let on_robust_frontier = robust_frontier_ids.contains(&candidate.candidate_id);
        let on_unambiguous_frontier = unambiguous_frontier_ids.contains(&candidate.candidate_id);
        let on_ambiguous_frontier = ambiguous_frontier_ids.contains(&candidate.candidate_id);
        let on_midpoint_reference_frontier = midpoint_frontier_ids.contains(&candidate.candidate_id);
        candidate_results.push(CandidateFrontierResult {
            candidate_id: candidate.candidate_id.clone(),
            status: candidate.status,
            missing_objective_ids,
            robustly_dominated_by,
            possible_only_dominators,
            on_robust_frontier,
            on_unambiguous_frontier,
            on_ambiguous_frontier,
            on_midpoint_reference_frontier,
        });
    }

    let mut objective_coverage: Vec<ObjectiveCoverage> = objectives
        .iter()
        .map(|objective| coverage_for_objective(objective, &canonical_candidates))
        .collect();
    // Objective order is part of the plan contract and is preserved here.

    let feasible_count = canonical_candidates.iter().filter(|c| c.status == CandidateStatus::Feasible).count();
    let infeasible_count = canonical_candidates.iter().filter(|c| c.status == CandidateStatus::Infeasible).count();
    let indeterminate_count = canonical_candidates.iter().filter(|c| c.status == CandidateStatus::Indeterminate).count();
    let evaluation_failed_count = canonical_candidates.iter().filter(|c| c.status == CandidateStatus::EvaluationFailed).count();
    let missing_objective_observation_count = canonical_candidates
        .iter()
        .flat_map(|candidate| &candidate.objectives)
        .filter(|observation| matches!(observation, ObjectiveObservation::Missing { .. }))
        .count();

    let counts = FrontierCounts {
        candidate_count: canonical_candidates.len(),
        feasible_count,
        infeasible_count,
        indeterminate_count,
        evaluation_failed_count,
        missing_objective_observation_count,
        robust_frontier_count: robust_frontier_ids.len(),
        unambiguous_frontier_count: unambiguous_frontier_ids.len(),
        ambiguous_frontier_count: ambiguous_frontier_ids.len(),
        robust_dominance_pair_count: robust_dominance_pairs.len(),
        possible_only_dominance_pair_count: possible_only_dominance_pairs.len(),
        midpoint_frontier_disagreement_count: midpoint_frontier_disagreements.len(),
    };

    // Retain deterministic order; this binding also ensures future edits don't
    // accidentally replace a missing-interval row with an imputed value.
    objective_coverage.shrink_to_fit();

    Ok(RobustFrontierReport {
        study_id: study_id.to_owned(),
        candidates: candidate_results,
        robust_frontier_ids,
        unambiguous_frontier_ids,
        ambiguous_frontier_ids,
        midpoint_reference_frontier_ids: midpoint_frontier_ids,
        midpoint_frontier_disagreements,
        robust_dominance_pairs,
        possible_only_dominance_pairs,
        objective_coverage,
        counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn objective(objective_id: &str, direction: ObjectiveDirection) -> ObjectiveSpec {
        ObjectiveSpec {
            objective_id: objective_id.into(),
            direction,
            unit: match objective_id {
                "cost" => "USD".into(),
                "payload" => "kg".into(),
                "reliability" => "percent".into(),
                _ => "unit".into(),
            },
            reference_conditions_id: "condition-set-v1".into(),
            interval_semantics: IntervalSemantics::EngineeringBound,
            interval_method_id: "synthetic-known-answer-v1".into(),
        }
    }

    fn plan() -> Vec<ObjectiveSpec> {
        vec![
            objective("cost", ObjectiveDirection::Minimize),
            objective("payload", ObjectiveDirection::Maximize),
        ]
    }

    fn interval(objective_id: &str, lower: f64, upper: f64) -> ObjectiveObservation {
        ObjectiveObservation::Interval(ObjectiveInterval {
            objective_id: objective_id.into(),
            lower,
            upper,
            unit: match objective_id {
                "cost" => "USD".into(),
                "payload" => "kg".into(),
                "reliability" => "percent".into(),
                _ => "unit".into(),
            },
            reference_conditions_id: "condition-set-v1".into(),
            interval_semantics: IntervalSemantics::EngineeringBound,
            interval_method_id: "synthetic-known-answer-v1".into(),
            evidence_ref: format!("synthetic-evidence-{objective_id}"),
        })
    }

    fn missing(objective_id: &str, reason: &str) -> ObjectiveObservation {
        ObjectiveObservation::Missing {
            objective_id: objective_id.into(),
            reason: reason.into(),
        }
    }

    fn candidate(id: &str, cost: (f64, f64), payload: (f64, f64)) -> CandidateAssessment {
        CandidateAssessment {
            candidate_id: id.into(),
            status: CandidateStatus::Feasible,
            status_evidence_ref: format!("synthetic-status-{id}"),
            objectives: vec![
                interval("cost", cost.0, cost.1),
                interval("payload", payload.0, payload.1),
            ],
        }
    }

    #[test]
    fn point_intervals_reproduce_classical_pareto_frontier() {
        let candidates = vec![
            candidate("a", (1.0, 1.0), (5.0, 5.0)),
            candidate("b", (2.0, 2.0), (4.0, 4.0)),
            candidate("c", (1.5, 1.5), (6.0, 6.0)),
            candidate("d", (4.0, 4.0), (1.0, 1.0)),
        ];
        let report = analyze_robust_frontier("test", &plan(), &candidates).unwrap();
        assert_eq!(report.robust_frontier_ids, vec!["a", "c"]);
        assert_eq!(report.midpoint_reference_frontier_ids, vec!["a", "c"]);
        assert_eq!(report.midpoint_frontier_disagreements, Vec::<String>::new());
        assert_eq!(report.robust_dominance_pairs.len(), 4);
    }

    #[test]
    fn separated_intervals_can_establish_robust_dominance() {
        let objectives = vec![
            objective("cost", ObjectiveDirection::Minimize),
            objective("reliability", ObjectiveDirection::Maximize),
        ];
        let a = CandidateAssessment {
            candidate_id: "a".into(),
            status: CandidateStatus::Feasible,
            status_evidence_ref: "status-a".into(),
            objectives: vec![interval("cost", 5.0, 6.0), interval("reliability", 90.0, 95.0)],
        };
        let b = CandidateAssessment {
            candidate_id: "b".into(),
            status: CandidateStatus::Feasible,
            status_evidence_ref: "status-b".into(),
            objectives: vec![interval("cost", 8.0, 9.0), interval("reliability", 80.0, 85.0)],
        };
        let report = analyze_robust_frontier("test", &objectives, &[a, b]).unwrap();
        assert_eq!(report.robust_frontier_ids, vec!["a"]);
        assert_eq!(report.robust_dominance_pairs, vec![DominancePair {
            dominator_id: "a".into(), dominated_id: "b".into()
        }]);
    }

    #[test]
    fn overlapping_intervals_are_possible_only_not_robust_dominance() {
        let objectives = vec![
            objective("cost", ObjectiveDirection::Minimize),
            objective("reliability", ObjectiveDirection::Maximize),
        ];
        let a = CandidateAssessment {
            candidate_id: "a".into(), status: CandidateStatus::Feasible,
            status_evidence_ref: "status-a".into(),
            objectives: vec![interval("cost", 5.0, 8.0), interval("reliability", 85.0, 95.0)],
        };
        let b = CandidateAssessment {
            candidate_id: "b".into(), status: CandidateStatus::Feasible,
            status_evidence_ref: "status-b".into(),
            objectives: vec![interval("cost", 7.0, 10.0), interval("reliability", 80.0, 90.0)],
        };
        let report = analyze_robust_frontier("test", &objectives, &[a, b]).unwrap();
        assert_eq!(report.robust_dominance_pairs.len(), 0);
        assert_eq!(report.possible_only_dominance_pairs.len(), 2);
        assert_eq!(report.robust_frontier_ids, vec!["a", "b"]);
        assert_eq!(report.ambiguous_frontier_ids, vec!["a", "b"]);
    }

    #[test]
    fn genuine_tradeoff_remains_on_frontier() {
        let candidates = vec![
            candidate("low-cost", (1.0, 1.2), (4.0, 4.5)),
            candidate("high-payload", (2.0, 2.2), (6.0, 6.5)),
        ];
        let report = analyze_robust_frontier("test", &plan(), &candidates).unwrap();
        assert_eq!(report.robust_frontier_ids, vec!["high-payload", "low-cost"]);
        assert_eq!(report.robust_dominance_pairs.len(), 0);
    }

    #[test]
    fn infeasible_indeterminate_and_failed_candidates_are_visible_but_excluded() {
        let mut infeasible = candidate("infeasible", (0.5, 0.7), (100.0, 110.0));
        infeasible.status = CandidateStatus::Infeasible;
        infeasible.status_evidence_ref = "hard-constraint-failure".into();
        let mut unknown = candidate("unknown", (0.1, 0.2), (200.0, 210.0));
        unknown.status = CandidateStatus::Indeterminate;
        unknown.status_evidence_ref = "feasibility-unresolved".into();
        let mut failed = candidate("failed", (0.1, 0.2), (200.0, 210.0));
        failed.status = CandidateStatus::EvaluationFailed;
        failed.status_evidence_ref = "run-failed".into();
        let feasible = candidate("feasible", (3.0, 4.0), (2.0, 3.0));
        let report = analyze_robust_frontier("test", &plan(), &[infeasible, unknown, failed, feasible]).unwrap();
        assert_eq!(report.robust_frontier_ids, vec!["feasible"]);
        assert_eq!(report.counts.feasible_count, 1);
        assert_eq!(report.counts.infeasible_count, 1);
        assert_eq!(report.counts.indeterminate_count, 1);
        assert_eq!(report.counts.evaluation_failed_count, 1);
        assert_eq!(report.candidates.len(), 4);
    }

    #[test]
    fn non_feasible_candidate_can_keep_explicit_missing_metric_reasons() {
        let mut unknown = candidate("unknown", (1.0, 2.0), (2.0, 3.0));
        unknown.status = CandidateStatus::Indeterminate;
        unknown.objectives[0] = missing("cost", "upstream model did not converge");
        let report = analyze_robust_frontier("test", &plan(), &[unknown]).unwrap();
        assert!(report.robust_frontier_ids.is_empty());
        assert_eq!(report.counts.missing_objective_observation_count, 1);
        assert_eq!(report.objective_coverage[0].missing_interval_count, 1);
    }

    #[test]
    fn missing_metric_for_claimed_feasible_candidate_fails_closed() {
        let mut invalid = candidate("incomplete", (1.0, 2.0), (2.0, 3.0));
        invalid.objectives[0] = missing("cost", "not available");
        let result = analyze_robust_frontier("test", &plan(), &[invalid]);
        assert!(matches!(result, Err(ref errors) if errors.iter().any(|error|
            matches!(error, FrontierValidationError::MissingObjectiveForFeasibleCandidate { .. }))));
    }

    #[test]
    fn maximize_direction_is_handled_without_changing_native_units() {
        let objectives = vec![objective("payload", ObjectiveDirection::Maximize)];
        let a = CandidateAssessment {
            candidate_id: "a".into(), status: CandidateStatus::Feasible,
            status_evidence_ref: "status-a".into(), objectives: vec![interval("payload", 10.0, 12.0)],
        };
        let b = CandidateAssessment {
            candidate_id: "b".into(), status: CandidateStatus::Feasible,
            status_evidence_ref: "status-b".into(), objectives: vec![interval("payload", 6.0, 8.0)],
        };
        let report = analyze_robust_frontier("test", &objectives, &[a, b]).unwrap();
        assert_eq!(report.robust_frontier_ids, vec!["a"]);
        assert_eq!(report.robust_dominance_pairs[0].dominator_id, "a");
    }

    #[test]
    fn midpoint_frontier_disagreements_are_reported_not_hidden() {
        let objectives = vec![
            objective("cost", ObjectiveDirection::Minimize),
            ObjectiveSpec {
                objective_id: "quality".into(), direction: ObjectiveDirection::Minimize,
                unit: "unit".into(), reference_conditions_id: "condition-set-v1".into(),
                interval_semantics: IntervalSemantics::EngineeringBound,
                interval_method_id: "synthetic-known-answer-v1".into(),
            },
        ];
        let custom = |id: &str, cost: (f64, f64), quality: (f64, f64)| CandidateAssessment {
            candidate_id: id.into(), status: CandidateStatus::Feasible,
            status_evidence_ref: format!("status-{id}"),
            objectives: vec![
                interval("cost", cost.0, cost.1),
                ObjectiveObservation::Interval(ObjectiveInterval {
                    objective_id: "quality".into(), lower: quality.0, upper: quality.1,
                    unit: "unit".into(), reference_conditions_id: "condition-set-v1".into(),
                    interval_semantics: IntervalSemantics::EngineeringBound,
                    interval_method_id: "synthetic-known-answer-v1".into(),
                    evidence_ref: format!("evidence-quality-{id}"),
                }),
            ],
        };
        let candidates = vec![
            custom("wide", (1.0, 9.0), (1.0, 9.0)),
            custom("narrow", (5.0, 6.0), (5.0, 6.0)),
        ];
        let report = analyze_robust_frontier("test", &objectives, &candidates).unwrap();
        assert_eq!(report.midpoint_reference_frontier_ids, vec!["wide"]);
        assert_eq!(report.robust_frontier_ids, vec!["narrow", "wide"]);
        assert_eq!(report.midpoint_frontier_disagreements, vec!["narrow"]);
    }

    #[test]
    fn empty_eligible_frontier_is_not_a_pass_or_error() {
        let mut failed = candidate("failed", (1.0, 2.0), (1.0, 2.0));
        failed.status = CandidateStatus::EvaluationFailed;
        let report = analyze_robust_frontier("test", &plan(), &[failed]).unwrap();
        assert!(report.robust_frontier_ids.is_empty());
        assert_eq!(report.counts.feasible_count, 0);
        assert_eq!(report.counts.candidate_count, 1);
    }

    #[test]
    fn duplicate_candidate_ids_are_rejected() {
        let a = candidate("same", (1.0, 2.0), (1.0, 2.0));
        let b = candidate("same", (2.0, 3.0), (3.0, 4.0));
        let result = analyze_robust_frontier("test", &plan(), &[a, b]);
        assert!(matches!(result, Err(ref errors) if errors.iter().any(|e|
            matches!(e, FrontierValidationError::DuplicateCandidateId(_)))));
    }

    #[test]
    fn reversed_nonfinite_and_overflowing_intervals_fail_closed() {
        let mut reversed = candidate("reversed", (4.0, 1.0), (2.0, 3.0));
        let result = analyze_robust_frontier("test", &plan(), &[reversed.clone()]);
        assert!(matches!(result, Err(ref errors) if errors.iter().any(|e|
            matches!(e, FrontierValidationError::ReversedInterval { .. }))));
        reversed.objectives[0] = interval("cost", f64::NAN, 5.0);
        let result = analyze_robust_frontier("test", &plan(), &[reversed.clone()]);
        assert!(matches!(result, Err(ref errors) if errors.iter().any(|e|
            matches!(e, FrontierValidationError::NonFiniteBound { .. }))));
        reversed.objectives[0] = interval("cost", -f64::MAX, f64::MAX);
        let result = analyze_robust_frontier("test", &plan(), &[reversed]);
        assert!(matches!(result, Err(ref errors) if errors.iter().any(|e|
            matches!(e, FrontierValidationError::IntervalWidthOverflow { .. }))));
    }

    #[test]
    fn condition_unit_and_method_mismatch_fail_closed() {
        let mut invalid = candidate("invalid", (1.0, 2.0), (3.0, 4.0));
        if let ObjectiveObservation::Interval(interval) = &mut invalid.objectives[0] {
            interval.unit = "kg".into();
        }
        let result = analyze_robust_frontier("test", &plan(), &[invalid]);
        assert!(matches!(result, Err(ref errors) if errors.iter().any(|e|
            matches!(e, FrontierValidationError::UnitMismatch { .. }))));
    }

    #[test]
    fn candidate_input_permutation_has_identical_report() {
        let a = candidate("a", (1.0, 2.0), (4.0, 5.0));
        let b = candidate("b", (3.0, 4.0), (6.0, 7.0));
        let c = candidate("c", (2.0, 3.0), (3.0, 4.0));
        let first = analyze_robust_frontier("test", &plan(), &[a.clone(), b.clone(), c.clone()]).unwrap();
        let second = analyze_robust_frontier("test", &plan(), &[c, a, b]).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn candidate_limit_is_checked_before_pairwise_work() {
        let candidates = (0..=MAX_FRONTIER_CANDIDATES).map(|index| {
            candidate(&format!("candidate-{index:04}"), (1.0, 2.0), (3.0, 4.0))
        }).collect::<Vec<_>>();
        let result = analyze_robust_frontier("test", &plan(), &candidates);
        assert!(matches!(result, Err(ref errors) if errors.iter().any(|error|
            matches!(error, FrontierValidationError::TooManyCandidates { .. }))));
    }

    #[test]
    fn empty_plan_or_missing_provenance_is_rejected() {
        assert!(matches!(
            analyze_robust_frontier("test", &[], &[candidate("a", (1.0, 1.0), (1.0, 1.0))]),
            Err(ref errors) if errors.iter().any(|error| matches!(error, FrontierValidationError::EmptyObjectiveSet))
        ));
        let mut invalid = candidate("a", (1.0, 1.0), (1.0, 1.0));
        invalid.status_evidence_ref.clear();
        let result = analyze_robust_frontier("test", &plan(), &[invalid]);
        assert!(matches!(result, Err(ref errors) if errors.iter().any(|error|
            matches!(error, FrontierValidationError::EmptyStatusEvidenceRef { .. }))));
    }
}
