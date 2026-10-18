// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prior-free, robust ranking of experiments by how strongly candidate models
//! disagree relative to explicitly declared predictive uncertainty.
//!
//! This is intentionally not called expected information gain (EIG): it uses
//! no hypothesis priors, outcome likelihoods, or posterior update. Its
//! maximin standardized-separation score is a transparent screening heuristic
//! for choosing discriminating simulations or physical measurements.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt::{self, Display, Formatter};

pub const SCHEMA_VERSION: &str = "physics-experiment-design-v0.1";

const METHOD_NOTE: &str = "Prior-free maximin standardized separation: for each hypothesis pair, combine squared mean differences divided by the sum of their predictive variances across declared conditionally independent observables; take the minimum pair separation, then divide by the caller-defined relative cost units.";
const LIMITATIONS: &[&str] = &[
    "This is a screening heuristic, not Bayesian expected information gain or a posterior probability.",
    "Predictive standard uncertainties must be calibrated independently; the planner does not estimate them.",
    "Observables are treated as conditionally independent; correlated measurements require a covariance-aware method.",
    "Cost units are caller-defined relative resource weights, not universal currency or wall-clock guarantees.",
    "Predictions are model inputs, not observations or validation evidence. A selected design still needs a declared protocol and independent scoring.",
];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentDesignInput {
    pub schema_version: String,
    pub target: String,
    pub hypothesis_ids: Vec<String>,
    pub experiments: Vec<CandidateExperiment>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateExperiment {
    pub experiment_id: String,
    /// Caller-defined positive relative resource cost. It is not assumed to be USD.
    pub cost_units: f64,
    /// Planned duration in hours; zero is permitted for purely computational designs.
    pub duration_hours: f64,
    pub observables: Vec<ObservablePrediction>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObservablePrediction {
    pub observable_id: String,
    pub predictions: Vec<HypothesisPrediction>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HypothesisPrediction {
    pub hypothesis_id: String,
    pub mean: f64,
    /// One-standard-deviation predictive uncertainty for this observable under the hypothesis.
    pub standard_uncertainty: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DesignReport {
    pub schema_version: String,
    pub target: String,
    pub criterion: String,
    pub hypothesis_count: usize,
    pub ranked_experiments: Vec<RankedExperiment>,
    pub limitations: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RankedExperiment {
    pub rank: usize,
    pub experiment_id: String,
    pub cost_units: f64,
    pub duration_hours: f64,
    /// Worst (least separated) pair of hypotheses, in combined standard-uncertainty units.
    pub worst_case_separation_sigma: f64,
    /// Worst-case separation divided by the caller-defined relative resource cost.
    pub separation_per_cost_unit: f64,
    pub closest_hypothesis_pair: [String; 2],
    pub observable_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    UnsupportedSchema(String),
    EmptyTarget,
    TooFewHypotheses,
    InvalidIdentifier { kind: String },
    DuplicateIdentifier { kind: String, id: String },
    NoExperiments,
    InvalidCost { experiment_id: String },
    InvalidDuration { experiment_id: String },
    NoObservables { experiment_id: String },
    DuplicateObservable {
        experiment_id: String,
        observable_id: String,
    },
    PredictionCountMismatch {
        experiment_id: String,
        observable_id: String,
        expected: usize,
        actual: usize,
    },
    UnknownHypothesis {
        experiment_id: String,
        observable_id: String,
        hypothesis_id: String,
    },
    DuplicatePrediction {
        experiment_id: String,
        observable_id: String,
        hypothesis_id: String,
    },
    MissingPrediction {
        experiment_id: String,
        observable_id: String,
        hypothesis_id: String,
    },
    InvalidPrediction {
        experiment_id: String,
        observable_id: String,
        hypothesis_id: String,
        reason: String,
    },
    NonFiniteScore { experiment_id: String },
}

impl Display for PlanError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchema(value) => write!(f, "unsupported schema version: {value}"),
            Self::EmptyTarget => write!(f, "target description must not be empty"),
            Self::TooFewHypotheses => write!(f, "at least two hypotheses are required"),
            Self::InvalidIdentifier { kind } => write!(f, "{kind} identifier must not be empty"),
            Self::DuplicateIdentifier { kind, id } => {
                write!(f, "duplicate {kind} identifier: {id}")
            }
            Self::NoExperiments => write!(f, "at least one candidate experiment is required"),
            Self::InvalidCost { experiment_id } => {
                write!(f, "experiment {experiment_id} needs finite positive cost_units")
            }
            Self::InvalidDuration { experiment_id } => {
                write!(f, "experiment {experiment_id} needs finite non-negative duration_hours")
            }
            Self::NoObservables { experiment_id } => {
                write!(f, "experiment {experiment_id} has no predicted observables")
            }
            Self::DuplicateObservable { experiment_id, observable_id } => {
                write!(f, "experiment {experiment_id} repeats observable {observable_id}")
            }
            Self::PredictionCountMismatch {
                experiment_id, observable_id, expected, actual,
            } => write!(
                f,
                "experiment {experiment_id}, observable {observable_id}: expected {expected} predictions, found {actual}"
            ),
            Self::UnknownHypothesis { experiment_id, observable_id, hypothesis_id } => write!(
                f,
                "experiment {experiment_id}, observable {observable_id}: unknown hypothesis {hypothesis_id}"
            ),
            Self::DuplicatePrediction { experiment_id, observable_id, hypothesis_id } => write!(
                f,
                "experiment {experiment_id}, observable {observable_id}: duplicate prediction for {hypothesis_id}"
            ),
            Self::MissingPrediction { experiment_id, observable_id, hypothesis_id } => write!(
                f,
                "experiment {experiment_id}, observable {observable_id}: missing prediction for {hypothesis_id}"
            ),
            Self::InvalidPrediction {
                experiment_id, observable_id, hypothesis_id, reason,
            } => write!(
                f,
                "experiment {experiment_id}, observable {observable_id}, hypothesis {hypothesis_id}: invalid prediction ({reason})"
            ),
            Self::NonFiniteScore { experiment_id } => {
                write!(f, "experiment {experiment_id} produced a non-finite separation score")
            }
        }
    }
}

impl Error for PlanError {}

/// Validate, score and deterministically rank candidate interventions.
///
/// For hypothesis pair (a,b) and observable j, the normalized distance is
/// abs(mu_a - mu_b) / sqrt(sigma_a^2 + sigma_b^2). Distances across observables
/// are combined in quadrature under an explicit conditional-independence
/// assumption. The candidate score is the minimum pair distance divided by
/// cost_units, so a design that leaves even one rival pair indistinguishable
/// is not rescued by separating all the other pairs.
pub fn rank_experiments(input: &ExperimentDesignInput) -> Result<DesignReport, PlanError> {
    validate_input(input)?;

    let mut hypotheses = input.hypothesis_ids.clone();
    hypotheses.sort();
    let mut ranked = Vec::with_capacity(input.experiments.len());

    for experiment in &input.experiments {
        let mut worst_distance = f64::INFINITY;
        let mut closest_pair: Option<[String; 2]> = None;

        for first_index in 0..hypotheses.len() {
            for second_index in (first_index + 1)..hypotheses.len() {
                let first_id = &hypotheses[first_index];
                let second_id = &hypotheses[second_index];
                let mut distance_squared = 0.0;

                for observable in &experiment.observables {
                    let first = observable
                        .predictions
                        .iter()
                        .find(|prediction| &prediction.hypothesis_id == first_id)
                        .expect("validation guarantees every hypothesis prediction");
                    let second = observable
                        .predictions
                        .iter()
                        .find(|prediction| &prediction.hypothesis_id == second_id)
                        .expect("validation guarantees every hypothesis prediction");

                    let combined_uncertainty = first
                        .standard_uncertainty
                        .hypot(second.standard_uncertainty);
                    let normalized_delta = (first.mean - second.mean) / combined_uncertainty;
                    distance_squared += normalized_delta * normalized_delta;

                    if !distance_squared.is_finite() {
                        return Err(PlanError::NonFiniteScore {
                            experiment_id: experiment.experiment_id.clone(),
                        });
                    }
                }

                let distance = distance_squared.sqrt();
                if !distance.is_finite() {
                    return Err(PlanError::NonFiniteScore {
                        experiment_id: experiment.experiment_id.clone(),
                    });
                }

                // The hypothesis list is sorted, so exact ties keep the same
                // lexicographically ordered pair regardless of input order.
                if distance < worst_distance {
                    worst_distance = distance;
                    closest_pair = Some([first_id.clone(), second_id.clone()]);
                }
            }
        }

        let separation_per_cost_unit = worst_distance / experiment.cost_units;
        if !separation_per_cost_unit.is_finite() {
            return Err(PlanError::NonFiniteScore {
                experiment_id: experiment.experiment_id.clone(),
            });
        }

        ranked.push(RankedExperiment {
            rank: 0,
            experiment_id: experiment.experiment_id.clone(),
            cost_units: experiment.cost_units,
            duration_hours: experiment.duration_hours,
            worst_case_separation_sigma: worst_distance,
            separation_per_cost_unit,
            closest_hypothesis_pair: closest_pair.expect("at least two hypotheses are validated"),
            observable_count: experiment.observables.len(),
        });
    }

    ranked.sort_by(|a, b| {
        b.separation_per_cost_unit
            .total_cmp(&a.separation_per_cost_unit)
            .then(
                b.worst_case_separation_sigma
                    .total_cmp(&a.worst_case_separation_sigma),
            )
            .then(a.cost_units.total_cmp(&b.cost_units))
            .then(a.experiment_id.cmp(&b.experiment_id))
    });
    for (index, experiment) in ranked.iter_mut().enumerate() {
        experiment.rank = index + 1;
    }

    Ok(DesignReport {
        schema_version: SCHEMA_VERSION.to_owned(),
        target: input.target.clone(),
        criterion: METHOD_NOTE.to_owned(),
        hypothesis_count: hypotheses.len(),
        ranked_experiments: ranked,
        limitations: LIMITATIONS.iter().map(|note| (*note).to_owned()).collect(),
    })
}

fn validate_input(input: &ExperimentDesignInput) -> Result<(), PlanError> {
    if input.schema_version != SCHEMA_VERSION {
        return Err(PlanError::UnsupportedSchema(input.schema_version.clone()));
    }
    if input.target.trim().is_empty() {
        return Err(PlanError::EmptyTarget);
    }
    if input.hypothesis_ids.len() < 2 {
        return Err(PlanError::TooFewHypotheses);
    }

    let mut hypotheses = BTreeSet::new();
    for id in &input.hypothesis_ids {
        if id.trim().is_empty() {
            return Err(PlanError::InvalidIdentifier {
                kind: "hypothesis".to_owned(),
            });
        }
        if !hypotheses.insert(id.clone()) {
            return Err(PlanError::DuplicateIdentifier {
                kind: "hypothesis".to_owned(),
                id: id.clone(),
            });
        }
    }
    if input.experiments.is_empty() {
        return Err(PlanError::NoExperiments);
    }

    let mut experiment_ids = BTreeSet::new();
    for experiment in &input.experiments {
        if experiment.experiment_id.trim().is_empty() {
            return Err(PlanError::InvalidIdentifier {
                kind: "experiment".to_owned(),
            });
        }
        if !experiment_ids.insert(experiment.experiment_id.clone()) {
            return Err(PlanError::DuplicateIdentifier {
                kind: "experiment".to_owned(),
                id: experiment.experiment_id.clone(),
            });
        }
        if !experiment.cost_units.is_finite() || experiment.cost_units <= 0.0 {
            return Err(PlanError::InvalidCost {
                experiment_id: experiment.experiment_id.clone(),
            });
        }
        if !experiment.duration_hours.is_finite() || experiment.duration_hours < 0.0 {
            return Err(PlanError::InvalidDuration {
                experiment_id: experiment.experiment_id.clone(),
            });
        }
        if experiment.observables.is_empty() {
            return Err(PlanError::NoObservables {
                experiment_id: experiment.experiment_id.clone(),
            });
        }

        let mut observable_ids = BTreeSet::new();
        for observable in &experiment.observables {
            if observable.observable_id.trim().is_empty() {
                return Err(PlanError::InvalidIdentifier {
                    kind: "observable".to_owned(),
                });
            }
            if !observable_ids.insert(observable.observable_id.clone()) {
                return Err(PlanError::DuplicateObservable {
                    experiment_id: experiment.experiment_id.clone(),
                    observable_id: observable.observable_id.clone(),
                });
            }
            if observable.predictions.len() != hypotheses.len() {
                return Err(PlanError::PredictionCountMismatch {
                    experiment_id: experiment.experiment_id.clone(),
                    observable_id: observable.observable_id.clone(),
                    expected: hypotheses.len(),
                    actual: observable.predictions.len(),
                });
            }

            let mut prediction_ids = BTreeSet::new();
            for prediction in &observable.predictions {
                if !hypotheses.contains(&prediction.hypothesis_id) {
                    return Err(PlanError::UnknownHypothesis {
                        experiment_id: experiment.experiment_id.clone(),
                        observable_id: observable.observable_id.clone(),
                        hypothesis_id: prediction.hypothesis_id.clone(),
                    });
                }
                if !prediction_ids.insert(prediction.hypothesis_id.clone()) {
                    return Err(PlanError::DuplicatePrediction {
                        experiment_id: experiment.experiment_id.clone(),
                        observable_id: observable.observable_id.clone(),
                        hypothesis_id: prediction.hypothesis_id.clone(),
                    });
                }
                if !prediction.mean.is_finite() {
                    return Err(PlanError::InvalidPrediction {
                        experiment_id: experiment.experiment_id.clone(),
                        observable_id: observable.observable_id.clone(),
                        hypothesis_id: prediction.hypothesis_id.clone(),
                        reason: "mean must be finite".to_owned(),
                    });
                }
                if !prediction.standard_uncertainty.is_finite()
                    || prediction.standard_uncertainty <= 0.0
                {
                    return Err(PlanError::InvalidPrediction {
                        experiment_id: experiment.experiment_id.clone(),
                        observable_id: observable.observable_id.clone(),
                        hypothesis_id: prediction.hypothesis_id.clone(),
                        reason: "standard_uncertainty must be finite and positive".to_owned(),
                    });
                }
            }

            for hypothesis_id in &hypotheses {
                if !prediction_ids.contains(hypothesis_id) {
                    return Err(PlanError::MissingPrediction {
                        experiment_id: experiment.experiment_id.clone(),
                        observable_id: observable.observable_id.clone(),
                        hypothesis_id: hypothesis_id.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// Return the worked Galileo fixture included in this crate.
pub fn galileo_demo_input() -> ExperimentDesignInput {
    serde_json::from_str(include_str!("../fixtures/galileo-discrimination-v0.1.json"))
        .expect("bundled Galileo fixture must remain valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn galileo_design_prefers_the_intervention_that_separates_the_models() {
        let report = rank_experiments(&galileo_demo_input()).unwrap();
        assert_eq!(report.ranked_experiments.len(), 2);
        assert_eq!(
            report.ranked_experiments[0].experiment_id,
            "observe_at_four_seconds"
        );
        assert_eq!(
            report.ranked_experiments[1].experiment_id,
            "matched_at_two_seconds"
        );
        assert!(report.ranked_experiments[0].worst_case_separation_sigma > 60.0);
        assert!(
            report.ranked_experiments[1].worst_case_separation_sigma.abs() < 1.0e-12
        );
    }

    #[test]
    fn ranked_result_is_invariant_to_input_hypothesis_and_candidate_order() {
        let original = galileo_demo_input();
        let mut permuted = original.clone();
        permuted.hypothesis_ids.reverse();
        permuted.experiments.reverse();
        for experiment in &mut permuted.experiments {
            for observable in &mut experiment.observables {
                observable.predictions.reverse();
            }
        }

        let first = rank_experiments(&original).unwrap();
        let second = rank_experiments(&permuted).unwrap();
        assert_eq!(
            first.ranked_experiments,
            second.ranked_experiments,
            "equivalent reordered input should not change the ranking or score"
        );
    }

    #[test]
    fn missing_hypothesis_prediction_fails_closed() {
        let mut input = galileo_demo_input();
        input.experiments[0].observables[0].predictions.pop();
        assert!(matches!(
            rank_experiments(&input),
            Err(PlanError::PredictionCountMismatch { .. })
        ));
    }

    #[test]
    fn zero_or_non_finite_uncertainty_fails_closed() {
        let mut input = galileo_demo_input();
        input.experiments[0].observables[0].predictions[0].standard_uncertainty = 0.0;
        assert!(matches!(
            rank_experiments(&input),
            Err(PlanError::InvalidPrediction { .. })
        ));

        let mut input = galileo_demo_input();
        input.experiments[0].observables[0].predictions[0].standard_uncertainty = f64::INFINITY;
        assert!(matches!(
            rank_experiments(&input),
            Err(PlanError::InvalidPrediction { .. })
        ));
    }

    #[test]
    fn identical_predictions_produce_zero_discriminating_power() {
        let mut input = galileo_demo_input();
        for experiment in &mut input.experiments {
            for prediction in &mut experiment.observables[0].predictions {
                prediction.mean = 1.0;
            }
        }
        let report = rank_experiments(&input).unwrap();
        assert!(report
            .ranked_experiments
            .iter()
            .all(|experiment| experiment.worst_case_separation_sigma.abs() < 1.0e-12));
    }

    #[test]
    fn non_finite_predictions_and_invalid_costs_are_rejected() {
        let mut input = galileo_demo_input();
        input.experiments[0].observables[0].predictions[0].mean = f64::NAN;
        assert!(matches!(
            rank_experiments(&input),
            Err(PlanError::InvalidPrediction { .. })
        ));

        let mut input = galileo_demo_input();
        input.experiments[0].cost_units = 0.0;
        assert!(matches!(
            rank_experiments(&input),
            Err(PlanError::InvalidCost { .. })
        ));
    }

    #[test]
    fn increasing_predictive_uncertainty_reduces_discrimination_score() {
        let mut input = galileo_demo_input();
        let baseline = rank_experiments(&input).unwrap();
        let baseline_score = baseline.ranked_experiments[0].worst_case_separation_sigma;

        for experiment in &mut input.experiments {
            for observable in &mut experiment.observables {
                for prediction in &mut observable.predictions {
                    prediction.standard_uncertainty *= 10.0;
                }
            }
        }
        let degraded = rank_experiments(&input).unwrap();
        let degraded_score = degraded.ranked_experiments[0].worst_case_separation_sigma;
        assert!((degraded_score * 10.0 - baseline_score).abs() < 1.0e-10);
    }

    #[test]
    fn unsupported_schema_and_duplicate_hypotheses_are_rejected() {
        let mut input = galileo_demo_input();
        input.schema_version = "physics-experiment-design-v9".to_owned();
        assert!(matches!(
            rank_experiments(&input),
            Err(PlanError::UnsupportedSchema(_))
        ));

        let mut input = galileo_demo_input();
        input.hypothesis_ids[1] = input.hypothesis_ids[0].clone();
        assert!(matches!(
            rank_experiments(&input),
            Err(PlanError::DuplicateIdentifier { .. })
        ));
    }

    #[test]
    fn serialized_report_discloses_method_and_limitations() {
        let report = rank_experiments(&galileo_demo_input()).unwrap();
        let serialized = serde_json::to_string(&report).unwrap();
        assert!(serialized.contains("Prior-free maximin standardized separation"));
        assert!(serialized.contains("conditionally independent"));
        assert!(serialized.contains("not Bayesian expected information gain"));
    }
}
