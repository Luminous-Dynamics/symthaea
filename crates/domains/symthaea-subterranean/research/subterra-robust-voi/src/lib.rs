//! Model-ensemble robustness evaluation for Subterra one-step EVSI.
//!
//! Evaluates the same candidate measurement set across a finite declared set
//! of plausible prior / likelihood models. An action is selected only if it
//! remains eligible in every model and its worst-case net EVSI exceeds the
//! caller's minimum threshold.
//!
//! "Worst case" means worst case over this finite supplied model set, not over
//! every possible real-world sensor model. The crate does not construct,
//! calibrate, or validate the model set.

use std::collections::{BTreeMap, BTreeSet};

use subterra_voi_reference::{
    ActionStatus as SingleModelActionStatus, CostPrices, PlanningError, PlanningProblem,
    ResourceBudget, plan_measurements,
};

const COMPARISON_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, PartialEq)]
pub struct ModelScenario {
    pub model_id: String,
    /// Same hypothesis / decision / action schema across models. Priors and
    /// conditional observation likelihoods may differ.
    pub problem: PlanningProblem,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustActionStatus {
    Selected,
    RobustlyEligibleNotSelected,
    ExcludedByAtLeastOneModel,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PerModelActionValue {
    pub model_id: String,
    pub status: SingleModelActionStatus,
    pub net_value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustActionAssessment {
    pub action_id: String,
    pub status: RobustActionStatus,
    /// Minimum net EVSI across the finite declared model set.
    pub worst_case_net_value: Option<f64>,
    /// Descriptive unweighted arithmetic mean across declared models.
    /// This is not a probabilistic expectation unless model weights are added.
    pub mean_model_net_value: Option<f64>,
    /// Maximum net EVSI across the finite declared model set.
    pub best_case_net_value: Option<f64>,
    pub per_model: Vec<PerModelActionValue>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustSelectionReport {
    pub selected_action_id: Option<String>,
    pub model_count: usize,
    /// Sorted by descending worst-case value, then action ID.
    pub assessments: Vec<RobustActionAssessment>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RobustError {
    EmptyModelSet,
    EmptyModelId,
    DuplicateModelId(String),
    ModelSchemaMismatch(String),
    InvalidMinimumWorstCaseValue,
    UnderlyingModelError {
        model_id: String,
        error: PlanningError,
    },
    NumericalOverflow(&'static str),
}

impl std::fmt::Display for RobustError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for RobustError {}

fn validate_model_schema(models: &[ModelScenario]) -> Result<(), RobustError> {
    let first = &models[0].problem;

    let first_hypotheses = first
        .hypotheses
        .iter()
        .map(|hypothesis| hypothesis.id.as_str())
        .collect::<Vec<_>>();
    let first_decisions = first
        .decisions
        .iter()
        .map(|decision| (decision.id.as_str(), decision.loss_by_hypothesis.as_slice()))
        .collect::<Vec<_>>();

    let mut first_actions = BTreeMap::new();
    for action in &first.actions {
        let outcomes = action
            .outcomes
            .iter()
            .map(|outcome| outcome.id.as_str())
            .collect::<Vec<_>>();
        first_actions.insert(
            action.id.as_str(),
            (
                action.modality.as_str(),
                action.cost,
                action.external_gate_passed,
                outcomes,
            ),
        );
    }

    for model in models.iter().skip(1) {
        let problem = &model.problem;
        let hypothesis_ids = problem
            .hypotheses
            .iter()
            .map(|hypothesis| hypothesis.id.as_str())
            .collect::<Vec<_>>();
        if hypothesis_ids != first_hypotheses {
            return Err(RobustError::ModelSchemaMismatch(format!(
                "model {} changes hypothesis identity/order",
                model.model_id
            )));
        }

        let decisions = problem
            .decisions
            .iter()
            .map(|decision| (decision.id.as_str(), decision.loss_by_hypothesis.as_slice()))
            .collect::<Vec<_>>();
        if decisions != first_decisions {
            return Err(RobustError::ModelSchemaMismatch(format!(
                "model {} changes mission decision IDs, order, or loss matrix",
                model.model_id
            )));
        }

        if problem.actions.len() != first_actions.len() {
            return Err(RobustError::ModelSchemaMismatch(format!(
                "model {} changes candidate action count",
                model.model_id
            )));
        }

        for action in &problem.actions {
            let Some((modality, cost, gate, outcome_ids)) =
                first_actions.get(action.id.as_str())
            else {
                return Err(RobustError::ModelSchemaMismatch(format!(
                    "model {} adds or removes action {}",
                    model.model_id, action.id
                )));
            };
            let model_outcome_ids = action
                .outcomes
                .iter()
                .map(|outcome| outcome.id.as_str())
                .collect::<Vec<_>>();
            if action.modality != *modality
                || action.cost != *cost
                || action.external_gate_passed != *gate
                || model_outcome_ids != *outcome_ids
            {
                return Err(RobustError::ModelSchemaMismatch(format!(
                    "model {} changes action contract for {}",
                    model.model_id, action.id
                )));
            }
        }
    }

    Ok(())
}

/// Select a measurement only when it is valuable across every supplied model.
///
/// The model family is finite, explicit and caller supplied. Each model must
/// share the same hypothesis IDs/order, mission decisions/losses, action IDs,
/// modalities, resource costs, external-gate results and outcome IDs/order.
/// Priors and outcome likelihood values may vary between models.
///
/// An action denied by an external gate or over budget in any model is
/// excluded from robust selection. Hard constraints are not averaged.
pub fn select_robust_measurement(
    models: &[ModelScenario],
    budget: ResourceBudget,
    prices: CostPrices,
    minimum_worst_case_net_value: f64,
) -> Result<RobustSelectionReport, RobustError> {
    if models.is_empty() {
        return Err(RobustError::EmptyModelSet);
    }
    if !minimum_worst_case_net_value.is_finite() || minimum_worst_case_net_value < 0.0 {
        return Err(RobustError::InvalidMinimumWorstCaseValue);
    }

    let mut model_ids = BTreeSet::new();
    for model in models {
        if model.model_id.trim().is_empty() {
            return Err(RobustError::EmptyModelId);
        }
        if !model_ids.insert(model.model_id.as_str()) {
            return Err(RobustError::DuplicateModelId(model.model_id.clone()));
        }
    }

    validate_model_schema(models)?;

    let mut model_reports = Vec::with_capacity(models.len());
    for model in models {
        let report = plan_measurements(&model.problem, budget, prices, 0.0).map_err(|error| {
            RobustError::UnderlyingModelError {
                model_id: model.model_id.clone(),
                error,
            }
        })?;
        model_reports.push((model.model_id.as_str(), report));
    }

    let mut action_ids = models[0]
        .problem
        .actions
        .iter()
        .map(|action| action.id.clone())
        .collect::<Vec<_>>();
    action_ids.sort();

    let mut assessments = Vec::with_capacity(action_ids.len());
    for action_id in action_ids {
        let mut per_model = Vec::with_capacity(models.len());
        let mut all_eligible = true;
        let mut net_values = Vec::with_capacity(models.len());

        for (model_id, report) in &model_reports {
            let action = report
                .assessments
                .iter()
                .find(|assessment| assessment.action_id == action_id)
                .expect("validated model schema preserves action IDs");

            if action.status != SingleModelActionStatus::Eligible {
                all_eligible = false;
            }
            if let Some(value) = action.net_value {
                if !value.is_finite() {
                    return Err(RobustError::NumericalOverflow("model net EVSI"));
                }
                net_values.push(value);
            }
            per_model.push(PerModelActionValue {
                model_id: (*model_id).to_string(),
                status: action.status,
                net_value: action.net_value,
            });
        }

        if !all_eligible || net_values.len() != models.len() {
            assessments.push(RobustActionAssessment {
                action_id,
                status: RobustActionStatus::ExcludedByAtLeastOneModel,
                worst_case_net_value: None,
                mean_model_net_value: None,
                best_case_net_value: None,
                per_model,
            });
            continue;
        }

        let worst_case = net_values.iter().copied().fold(f64::INFINITY, f64::min);
        let best_case = net_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean = net_values.iter().sum::<f64>() / net_values.len() as f64;

        if !worst_case.is_finite() || !best_case.is_finite() || !mean.is_finite() {
            return Err(RobustError::NumericalOverflow("robust model aggregation"));
        }

        assessments.push(RobustActionAssessment {
            action_id,
            status: RobustActionStatus::RobustlyEligibleNotSelected,
            worst_case_net_value: Some(worst_case),
            mean_model_net_value: Some(mean),
            best_case_net_value: Some(best_case),
            per_model,
        });
    }

    assessments.sort_by(|left, right| {
        match (left.worst_case_net_value, right.worst_case_net_value) {
            (Some(left_value), Some(right_value)) => right_value
                .total_cmp(&left_value)
                .then_with(|| left.action_id.cmp(&right.action_id)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => left.action_id.cmp(&right.action_id),
        }
    });

    let selected_action_id = assessments
        .iter()
        .find(|assessment| {
            assessment.status == RobustActionStatus::RobustlyEligibleNotSelected
                && assessment
                    .worst_case_net_value
                    .is_some_and(|value| {
                        value > minimum_worst_case_net_value + COMPARISON_TOLERANCE
                    })
        })
        .map(|assessment| assessment.action_id.clone());

    if let Some(selected_id) = selected_action_id.as_deref() {
        if let Some(assessment) = assessments
            .iter_mut()
            .find(|assessment| assessment.action_id == selected_id)
        {
            assessment.status = RobustActionStatus::Selected;
        }
    }

    Ok(RobustSelectionReport {
        selected_action_id,
        model_count: models.len(),
        assessments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use subterra_voi_reference::{
        Hypothesis, MeasurementAction, MissionDecision, ObservationOutcome, PlanningProblem,
        ResourceCost,
    };

    fn hypotheses(prior: [f64; 3]) -> Vec<Hypothesis> {
        vec![
            Hypothesis {
                id: "open_continuation".into(),
                prior_probability: prior[0],
            },
            Hypothesis {
                id: "blocked_dead_end".into(),
                prior_probability: prior[1],
            },
            Hypothesis {
                id: "unknown".into(),
                prior_probability: prior[2],
            },
        ]
    }

    fn decisions() -> Vec<MissionDecision> {
        vec![
            MissionDecision {
                id: "enter".into(),
                loss_by_hypothesis: vec![0.0, 100.0, 50.0],
            },
            MissionDecision {
                id: "hold".into(),
                loss_by_hypothesis: vec![30.0, 0.0, 10.0],
            },
        ]
    }

    fn action(id: &str, signal: [f64; 3], cost: f64) -> MeasurementAction {
        MeasurementAction {
            id: id.into(),
            modality: "geophysical_scan".into(),
            outcomes: vec![
                ObservationOutcome {
                    id: "signal".into(),
                    likelihood_by_hypothesis: signal.to_vec(),
                },
                ObservationOutcome {
                    id: "no_signal".into(),
                    likelihood_by_hypothesis: signal.map(|value| 1.0 - value).to_vec(),
                },
            ],
            cost: ResourceCost {
                energy_j: cost,
                duration_s: 1.0,
                data_bytes: 10.0,
                exposure_units: 0.0,
            },
            external_gate_passed: true,
        }
    }

    fn model(model_id: &str, signal: [f64; 3]) -> ModelScenario {
        ModelScenario {
            model_id: model_id.into(),
            problem: PlanningProblem {
                hypotheses: hypotheses([0.45, 0.45, 0.10]),
                decisions: decisions(),
                actions: vec![action("gpr", signal, 100.0)],
            },
        }
    }

    fn budget() -> ResourceBudget {
        ResourceBudget {
            energy_j: 1_000.0,
            duration_s: 20.0,
            data_bytes: 100.0,
            exposure_units: 1.0,
        }
    }

    fn prices() -> CostPrices {
        CostPrices {
            energy_per_j: 0.001,
            time_per_s: 0.01,
            data_per_byte: 0.0001,
            exposure_per_unit: 0.0,
        }
    }

    #[test]
    fn selects_only_when_worst_case_value_is_positive() {
        let models = vec![
            model("nominal", [0.9, 0.1, 0.5]),
            model("degraded", [0.8, 0.2, 0.5]),
        ];
        let report =
            select_robust_measurement(&models, budget(), prices(), 0.0).unwrap();
        assert_eq!(report.selected_action_id.as_deref(), Some("gpr"));
        assert_eq!(report.model_count, 2);
        assert_eq!(
            report.assessments[0].status,
            RobustActionStatus::Selected
        );
    }

    #[test]
    fn model_disagreement_can_force_abstention() {
        let models = vec![
            model("informative", [0.9, 0.1, 0.5]),
            model("uninformative", [0.5, 0.5, 0.5]),
        ];
        let report =
            select_robust_measurement(&models, budget(), prices(), 0.0).unwrap();
        assert!(report.selected_action_id.is_none());
        assert!(report.assessments[0]
            .worst_case_net_value
            .is_some_and(|value| value <= 0.0));
    }

    #[test]
    fn one_model_gate_denial_excludes_the_action() {
        let mut denied = model("denied-model", [0.9, 0.1, 0.5]);
        denied.problem.actions[0].external_gate_passed = false;
        let models = vec![model("nominal", [0.9, 0.1, 0.5]), denied];
        let report =
            select_robust_measurement(&models, budget(), prices(), 0.0).unwrap();
        assert!(report.selected_action_id.is_none());
        assert_eq!(
            report.assessments[0].status,
            RobustActionStatus::ExcludedByAtLeastOneModel
        );
    }

    #[test]
    fn action_order_does_not_change_robust_selection() {
        let mut first = model("first", [0.9, 0.1, 0.5]);
        first.problem.actions.push(action("spectral", [0.8, 0.2, 0.5], 50.0));
        let mut second = model("second", [0.85, 0.15, 0.5]);
        second.problem.actions.push(action("spectral", [0.7, 0.3, 0.5], 50.0));

        let left = select_robust_measurement(
            &[first.clone(), second.clone()],
            budget(),
            prices(),
            0.0,
        )
        .unwrap();

        for model in [&mut first, &mut second] {
            model.problem.actions.reverse();
        }
        let right =
            select_robust_measurement(&[first, second], budget(), prices(), 0.0).unwrap();

        assert_eq!(left.selected_action_id, right.selected_action_id);
        assert_eq!(left.assessments, right.assessments);
    }

    #[test]
    fn hypothesis_order_mismatch_is_rejected() {
        let mut mismatch = model("mismatch", [0.8, 0.2, 0.5]);
        mismatch.problem.hypotheses.swap(0, 1);
        assert!(matches!(
            select_robust_measurement(
                &[model("nominal", [0.9, 0.1, 0.5]), mismatch],
                budget(),
                prices(),
                0.0
            ),
            Err(RobustError::ModelSchemaMismatch(_))
        ));
    }

    #[test]
    fn mission_loss_mismatch_is_rejected() {
        let mut mismatch = model("mismatch", [0.8, 0.2, 0.5]);
        mismatch.problem.decisions[0].loss_by_hypothesis[0] = 1.0;
        assert!(matches!(
            select_robust_measurement(
                &[model("nominal", [0.9, 0.1, 0.5]), mismatch],
                budget(),
                prices(),
                0.0
            ),
            Err(RobustError::ModelSchemaMismatch(_))
        ));
    }

    #[test]
    fn duplicate_model_id_is_rejected() {
        let one = model("same", [0.9, 0.1, 0.5]);
        let two = model("same", [0.8, 0.2, 0.5]);
        assert_eq!(
            select_robust_measurement(&[one, two], budget(), prices(), 0.0),
            Err(RobustError::DuplicateModelId("same".into()))
        );
    }

    #[test]
    fn empty_model_set_is_rejected() {
        assert_eq!(
            select_robust_measurement(&[], budget(), prices(), 0.0),
            Err(RobustError::EmptyModelSet)
        );
    }

    #[test]
    fn threshold_can_force_abstention() {
        let report = select_robust_measurement(
            &[model("nominal", [0.9, 0.1, 0.5]), model("degraded", [0.8, 0.2, 0.5])],
            budget(),
            prices(),
            100.0,
        )
        .unwrap();
        assert!(report.selected_action_id.is_none());
    }

    #[test]
    fn model_weights_are_not_silently_invented() {
        let report = select_robust_measurement(
            &[model("nominal", [0.9, 0.1, 0.5]), model("degraded", [0.8, 0.2, 0.5])],
            budget(),
            prices(),
            0.0,
        )
        .unwrap();
        // The report includes an arithmetic mean for description, but does not
        // accept model weights or treat this unweighted mean as a probability.
        assert!(report.assessments[0].mean_model_net_value.is_some());
    }
}
