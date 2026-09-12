// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Strict adapter from discovery evaluations to the shared Pareto kernel.

#![forbid(unsafe_code)]

use std::collections::{BTreeMap, BTreeSet};
use symthaea_discovery::{
    CandidateId, Evaluation, Feasibility, FidelityLevel, Objective, ObjectiveDirection, Prediction,
};
use symthaea_pareto::{rank, Direction, ObjectiveSpec, Point};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateRank {
    pub candidate_id: CandidateId,
    pub pareto_rank: usize,
    pub crowding_distance: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnrankedReason {
    EmptyObjectives,
    Infeasible,
    UnknownFeasibility,
    MissingObjectivePrediction {
        metric: String,
        unit: String,
    },
    AmbiguousObjectivePrediction {
        metric: String,
        unit: String,
        fidelity: FidelityLevel,
        count: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnrankedCandidate {
    pub candidate_id: CandidateId,
    pub reason: UnrankedReason,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParetoReport {
    pub ranked: Vec<CandidateRank>,
    pub fronts: Vec<Vec<CandidateId>>,
    pub unranked: Vec<UnrankedCandidate>,
}

/// Compute Pareto ranks without mutating the supplied discovery records.
pub fn rank_evaluations(evaluations: &[Evaluation]) -> Result<ParetoReport, BridgeError> {
    validate_cohort(evaluations)?;

    let canonical_schema = cohort_schema(evaluations)?;
    let Some(schema) = canonical_schema else {
        return Ok(ParetoReport {
            ranked: Vec::new(),
            fronts: Vec::new(),
            unranked: evaluations
                .iter()
                .map(|evaluation| UnrankedCandidate {
                    candidate_id: evaluation.candidate_id.clone(),
                    reason: UnrankedReason::EmptyObjectives,
                })
                .collect(),
        });
    };

    let kernel_specs = kernel_specs(&schema)?;
    let mut points = Vec::new();
    let mut point_candidates = Vec::new();
    let mut unranked = Vec::new();

    for evaluation in evaluations {
        if evaluation.objectives.is_empty() {
            unranked.push(UnrankedCandidate {
                candidate_id: evaluation.candidate_id.clone(),
                reason: UnrankedReason::EmptyObjectives,
            });
            continue;
        }

        match evaluation.feasibility()? {
            Feasibility::Infeasible => {
                unranked.push(UnrankedCandidate {
                    candidate_id: evaluation.candidate_id.clone(),
                    reason: UnrankedReason::Infeasible,
                });
                continue;
            }
            Feasibility::Unknown => {
                unranked.push(UnrankedCandidate {
                    candidate_id: evaluation.candidate_id.clone(),
                    reason: UnrankedReason::UnknownFeasibility,
                });
                continue;
            }
            Feasibility::Feasible => {}
        }

        let mut values = Vec::with_capacity(schema.len());
        let mut incomplete_reason = None;
        for canonical in &schema {
            match select_prediction(evaluation, &canonical.objective)? {
                PredictionSelection::Selected(prediction) => {
                    values.push(transform_value(&canonical.objective, prediction.value));
                }
                PredictionSelection::Missing => {
                    incomplete_reason = Some(UnrankedReason::MissingObjectivePrediction {
                        metric: canonical.objective.metric.clone(),
                        unit: canonical.objective.unit.clone(),
                    });
                    break;
                }
                PredictionSelection::Ambiguous { fidelity, count } => {
                    incomplete_reason = Some(UnrankedReason::AmbiguousObjectivePrediction {
                        metric: canonical.objective.metric.clone(),
                        unit: canonical.objective.unit.clone(),
                        fidelity,
                        count,
                    });
                    break;
                }
            }
        }

        if let Some(reason) = incomplete_reason {
            unranked.push(UnrankedCandidate {
                candidate_id: evaluation.candidate_id.clone(),
                reason,
            });
            continue;
        }

        points.push(Point::new(values));
        point_candidates.push(evaluation.candidate_id.clone());
    }

    let kernel_ranking = rank(&kernel_specs, &points)?;
    let ranked: Vec<CandidateRank> = point_candidates
        .iter()
        .enumerate()
        .map(|(index, candidate_id)| CandidateRank {
            candidate_id: candidate_id.clone(),
            pareto_rank: kernel_ranking.ranks[index],
            crowding_distance: kernel_ranking.crowding_distance[index],
        })
        .collect();

    let fronts = kernel_ranking
        .fronts
        .iter()
        .map(|front| {
            front
                .iter()
                .map(|&index| point_candidates[index].clone())
                .collect()
        })
        .collect();

    Ok(ParetoReport {
        ranked,
        fronts,
        unranked,
    })
}

/// Compute a complete report first, then atomically replace `pareto_rank`
/// metadata in the supplied evaluation cohort.
pub fn rank_and_assign(evaluations: &mut [Evaluation]) -> Result<ParetoReport, BridgeError> {
    let report = rank_evaluations(evaluations)?;
    let rank_by_candidate: BTreeMap<CandidateId, usize> = report
        .ranked
        .iter()
        .map(|rank| (rank.candidate_id.clone(), rank.pareto_rank))
        .collect();

    for evaluation in evaluations {
        evaluation.pareto_rank = rank_by_candidate.get(&evaluation.candidate_id).copied();
    }
    Ok(report)
}

fn validate_cohort(evaluations: &[Evaluation]) -> Result<(), BridgeError> {
    let mut candidates = BTreeSet::new();
    for evaluation in evaluations {
        CandidateId::new(evaluation.candidate_id.0.clone())?;
        evaluation.validate()?;
        if !candidates.insert(evaluation.candidate_id.clone()) {
            return Err(BridgeError::DuplicateCandidate(
                evaluation.candidate_id.0.clone(),
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct CanonicalObjective {
    signature: ObjectiveSignature,
    objective: Objective,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ObjectiveSignature {
    metric: String,
    unit: String,
    direction_tag: u8,
    target_bits: u64,
    tolerance_bits: u64,
}

fn cohort_schema(evaluations: &[Evaluation]) -> Result<Option<Vec<CanonicalObjective>>, BridgeError> {
    let mut expected: Option<Vec<CanonicalObjective>> = None;

    for evaluation in evaluations {
        if evaluation.objectives.is_empty() {
            continue;
        }
        let current = canonical_objectives(evaluation)?;
        if let Some(reference) = &expected {
            let reference_signatures: Vec<_> =
                reference.iter().map(|entry| &entry.signature).collect();
            let current_signatures: Vec<_> =
                current.iter().map(|entry| &entry.signature).collect();
            if reference_signatures != current_signatures {
                return Err(BridgeError::ObjectiveSchemaMismatch {
                    candidate_id: evaluation.candidate_id.0.clone(),
                });
            }
        } else {
            expected = Some(current);
        }
    }

    Ok(expected)
}

fn canonical_objectives(evaluation: &Evaluation) -> Result<Vec<CanonicalObjective>, BridgeError> {
    let mut seen_metric_units = BTreeSet::new();
    let mut canonical = Vec::with_capacity(evaluation.objectives.len());

    for objective in &evaluation.objectives {
        let key = (objective.metric.clone(), objective.unit.clone());
        if !seen_metric_units.insert(key.clone()) {
            return Err(BridgeError::DuplicateObjective {
                candidate_id: evaluation.candidate_id.0.clone(),
                metric: key.0,
                unit: key.1,
            });
        }
        canonical.push(CanonicalObjective {
            signature: objective_signature(objective),
            objective: objective.clone(),
        });
    }

    canonical.sort_by(|left, right| left.signature.cmp(&right.signature));
    Ok(canonical)
}

fn objective_signature(objective: &Objective) -> ObjectiveSignature {
    let (direction_tag, target_bits, tolerance_bits) = match objective.direction {
        ObjectiveDirection::Minimize => (0, 0, 0),
        ObjectiveDirection::Maximize => (1, 0, 0),
        ObjectiveDirection::Target { value, tolerance } => {
            (2, canonical_bits(value), canonical_bits(tolerance))
        }
    };
    ObjectiveSignature {
        metric: objective.metric.clone(),
        unit: objective.unit.clone(),
        direction_tag,
        target_bits,
        tolerance_bits,
    }
}

fn canonical_bits(value: f64) -> u64 {
    if value == 0.0 {
        0
    } else {
        value.to_bits()
    }
}

fn kernel_specs(schema: &[CanonicalObjective]) -> Result<Vec<ObjectiveSpec>, BridgeError> {
    schema
        .iter()
        .map(|canonical| {
            let direction = match canonical.objective.direction {
                ObjectiveDirection::Minimize | ObjectiveDirection::Target { .. } => {
                    Direction::Minimize
                }
                ObjectiveDirection::Maximize => Direction::Maximize,
            };
            ObjectiveSpec::new(
                format!(
                    "{} [{}] #{}",
                    canonical.objective.metric,
                    canonical.objective.unit,
                    canonical.signature.direction_tag
                ),
                direction,
            )
            .map_err(BridgeError::from)
        })
        .collect()
}

enum PredictionSelection<'a> {
    Selected(&'a Prediction),
    Missing,
    Ambiguous {
        fidelity: FidelityLevel,
        count: usize,
    },
}

fn select_prediction<'a>(
    evaluation: &'a Evaluation,
    objective: &Objective,
) -> Result<PredictionSelection<'a>, BridgeError> {
    let same_metric: Vec<&Prediction> = evaluation
        .predictions
        .iter()
        .filter(|prediction| prediction.metric == objective.metric)
        .collect();
    if same_metric.is_empty() {
        return Ok(PredictionSelection::Missing);
    }

    let matching_unit: Vec<&Prediction> = same_metric
        .iter()
        .copied()
        .filter(|prediction| prediction.unit == objective.unit)
        .collect();
    if matching_unit.is_empty() {
        let found: BTreeSet<String> = same_metric
            .iter()
            .map(|prediction| prediction.unit.clone())
            .collect();
        return Err(BridgeError::UnitMismatch {
            candidate_id: evaluation.candidate_id.0.clone(),
            metric: objective.metric.clone(),
            expected: objective.unit.clone(),
            found: found.into_iter().collect(),
        });
    }

    let max_rank = matching_unit
        .iter()
        .map(|prediction| prediction.fidelity.rank())
        .max()
        .expect("matching_unit is non-empty");
    let highest: Vec<&Prediction> = matching_unit
        .into_iter()
        .filter(|prediction| prediction.fidelity.rank() == max_rank)
        .collect();

    if highest.len() != 1 {
        return Ok(PredictionSelection::Ambiguous {
            fidelity: highest[0].fidelity,
            count: highest.len(),
        });
    }
    Ok(PredictionSelection::Selected(highest[0]))
}

fn transform_value(objective: &Objective, prediction: f64) -> f64 {
    match objective.direction {
        ObjectiveDirection::Minimize | ObjectiveDirection::Maximize => prediction,
        ObjectiveDirection::Target { value, tolerance } => {
            ((prediction - value).abs() - tolerance).max(0.0)
        }
    }
}

#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("invalid discovery evaluation: {0}")]
    Discovery(#[from] symthaea_discovery::DiscoveryError),
    #[error("Pareto kernel rejected cohort: {0}")]
    Pareto(#[from] symthaea_pareto::ParetoError),
    #[error("duplicate candidate evaluation {0:?}")]
    DuplicateCandidate(String),
    #[error(
        "candidate {candidate_id:?} declares duplicate objective {metric:?} in unit {unit:?}"
    )]
    DuplicateObjective {
        candidate_id: String,
        metric: String,
        unit: String,
    },
    #[error("candidate {candidate_id:?} does not match the cohort objective schema")]
    ObjectiveSchemaMismatch { candidate_id: String },
    #[error(
        "candidate {candidate_id:?} has unit mismatch for objective {metric:?}: expected {expected:?}, found {found:?}"
    )]
    UnitMismatch {
        candidate_id: String,
        metric: String,
        expected: String,
        found: Vec<String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_discovery::{
        Constraint, ConstraintBound, ModelProvenance, ObjectiveDirection, Prediction,
        UncertaintyEstimate,
    };

    fn prediction(metric: &str, value: f64, unit: &str, fidelity: FidelityLevel) -> Prediction {
        Prediction {
            metric: metric.into(),
            value,
            unit: unit.into(),
            uncertainty: UncertaintyEstimate::certain(),
            fidelity,
            model: ModelProvenance::named("test-model").unwrap(),
            assumptions: vec![],
            evidence: vec![],
        }
    }

    fn objective(metric: &str, unit: &str, direction: ObjectiveDirection) -> Objective {
        Objective {
            metric: metric.into(),
            unit: unit.into(),
            direction,
        }
    }

    fn evaluation(id: &str, cost: f64, efficiency: f64) -> Evaluation {
        Evaluation {
            candidate_id: CandidateId::new(id).unwrap(),
            objectives: vec![
                objective("cost", "USD", ObjectiveDirection::Minimize),
                objective("efficiency", "fraction", ObjectiveDirection::Maximize),
            ],
            constraints: vec![],
            predictions: vec![
                prediction("cost", cost, "USD", FidelityLevel::Surrogate),
                prediction(
                    "efficiency",
                    efficiency,
                    "fraction",
                    FidelityLevel::Surrogate,
                ),
            ],
            pareto_rank: None,
        }
    }

    #[test]
    fn ranks_complete_feasible_cohort_with_mixed_directions() {
        let cohort = vec![
            evaluation("A", 100.0, 0.90),
            evaluation("B", 80.0, 0.80),
            evaluation("C", 120.0, 0.70),
        ];
        let report = rank_evaluations(&cohort).unwrap();
        let ranks: BTreeMap<_, _> = report
            .ranked
            .iter()
            .map(|entry| (entry.candidate_id.0.as_str(), entry.pareto_rank))
            .collect();
        assert_eq!(ranks["A"], 0);
        assert_eq!(ranks["B"], 0);
        assert_eq!(ranks["C"], 1);
        assert_eq!(report.fronts[0].len(), 2);
    }

    #[test]
    fn objective_order_does_not_change_schema_identity() {
        let first = evaluation("A", 100.0, 0.9);
        let mut second = evaluation("B", 90.0, 0.85);
        second.objectives.reverse();
        let report = rank_evaluations(&[first, second]).unwrap();
        assert_eq!(report.ranked.len(), 2);
    }

    #[test]
    fn mixed_application_objective_schemas_fail_closed() {
        let first = evaluation("A", 100.0, 0.9);
        let mut second = evaluation("B", 90.0, 0.85);
        second.objectives[1] = objective(
            "discharge_duration",
            "h",
            ObjectiveDirection::Maximize,
        );
        assert!(matches!(
            rank_evaluations(&[first, second]),
            Err(BridgeError::ObjectiveSchemaMismatch { .. })
        ));
    }

    #[test]
    fn missing_objective_prediction_remains_unranked() {
        let mut incomplete = evaluation("A", 100.0, 0.9);
        incomplete
            .predictions
            .retain(|prediction| prediction.metric != "efficiency");
        let report = rank_evaluations(&[incomplete]).unwrap();
        assert!(report.ranked.is_empty());
        assert!(matches!(
            report.unranked[0].reason,
            UnrankedReason::MissingObjectivePrediction { .. }
        ));
    }

    #[test]
    fn ambiguous_highest_fidelity_objective_remains_unranked() {
        let mut ambiguous = evaluation("A", 100.0, 0.9);
        ambiguous.predictions.push(prediction(
            "efficiency",
            0.91,
            "fraction",
            FidelityLevel::Surrogate,
        ));
        let report = rank_evaluations(&[ambiguous]).unwrap();
        assert!(report.ranked.is_empty());
        assert!(matches!(
            report.unranked[0].reason,
            UnrankedReason::AmbiguousObjectivePrediction { .. }
        ));
    }

    #[test]
    fn unit_mismatch_is_integrity_error_not_missing_evidence() {
        let mut mismatch = evaluation("A", 100.0, 0.9);
        mismatch.predictions[0].unit = "EUR".into();
        assert!(matches!(
            rank_evaluations(&[mismatch]),
            Err(BridgeError::UnitMismatch { .. })
        ));
    }

    #[test]
    fn failed_hard_constraint_remains_unranked() {
        let mut infeasible = evaluation("A", 100.0, 0.9);
        infeasible.constraints.push(Constraint {
            metric: "efficiency".into(),
            unit: "fraction".into(),
            bound: ConstraintBound::AtLeast(0.95),
        });
        let report = rank_evaluations(&[infeasible]).unwrap();
        assert!(report.ranked.is_empty());
        assert_eq!(report.unranked[0].reason, UnrankedReason::Infeasible);
    }

    #[test]
    fn unknown_hard_constraint_remains_unranked() {
        let mut unknown = evaluation("A", 100.0, 0.9);
        unknown.constraints.push(Constraint {
            metric: "cycle_life".into(),
            unit: "cycles".into(),
            bound: ConstraintBound::AtLeast(1000.0),
        });
        let report = rank_evaluations(&[unknown]).unwrap();
        assert!(report.ranked.is_empty());
        assert_eq!(
            report.unranked[0].reason,
            UnrankedReason::UnknownFeasibility
        );
    }

    #[test]
    fn target_tolerance_creates_equal_optimum_inside_band() {
        let build = |id: &str, gap: f64| Evaluation {
            candidate_id: CandidateId::new(id).unwrap(),
            objectives: vec![objective(
                "band_gap",
                "eV",
                ObjectiveDirection::Target {
                    value: 1.40,
                    tolerance: 0.10,
                },
            )],
            constraints: vec![],
            predictions: vec![prediction(
                "band_gap",
                gap,
                "eV",
                FidelityLevel::Surrogate,
            )],
            pareto_rank: None,
        };
        let report = rank_evaluations(&[build("A", 1.35), build("B", 1.45)]).unwrap();
        assert_eq!(report.ranked[0].pareto_rank, 0);
        assert_eq!(report.ranked[1].pareto_rank, 0);
    }

    #[test]
    fn rank_and_assign_does_not_partially_mutate_on_schema_error() {
        let mut first = evaluation("A", 100.0, 0.9);
        let mut second = evaluation("B", 90.0, 0.85);
        first.pareto_rank = Some(41);
        second.pareto_rank = Some(42);
        second.objectives[1] = objective("duration", "h", ObjectiveDirection::Maximize);
        let mut cohort = vec![first, second];

        assert!(rank_and_assign(&mut cohort).is_err());
        assert_eq!(cohort[0].pareto_rank, Some(41));
        assert_eq!(cohort[1].pareto_rank, Some(42));
    }

    #[test]
    fn successful_assign_clears_stale_rank_for_unranked_candidate() {
        let mut complete = evaluation("A", 100.0, 0.9);
        let mut incomplete = evaluation("B", 90.0, 0.85);
        incomplete
            .predictions
            .retain(|prediction| prediction.metric != "efficiency");
        complete.pareto_rank = Some(99);
        incomplete.pareto_rank = Some(99);
        let mut cohort = vec![complete, incomplete];

        let report = rank_and_assign(&mut cohort).unwrap();
        assert_eq!(cohort[0].pareto_rank, Some(0));
        assert_eq!(cohort[1].pareto_rank, None);
        assert_eq!(report.unranked.len(), 1);
    }

    #[test]
    fn duplicate_candidate_evaluations_fail_closed() {
        let cohort = vec![evaluation("A", 100.0, 0.9), evaluation("A", 90.0, 0.8)];
        assert!(matches!(
            rank_evaluations(&cohort),
            Err(BridgeError::DuplicateCandidate(_))
        ));
    }
}
