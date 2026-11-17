//! Task-to-EVSI adapter for Subterra's existing survival-conditioned planner.
//!
//! This crate does not implement a second optimizer. It:
//! - supplies a typed task boundary around existing planner inputs;
//! - computes no-test, entropy-only and one-step EVSI baselines;
//! - runs the existing bounded contingent planner;
//! - asks the independent robust referee to score the same policy across models.
//!
//! All fixtures in this crate are synthetic. Likelihoods and losses supplied
//! by applications must be separately calibrated and independently governed.

use std::collections::BTreeSet;

use subterra_robust_survival_referee::{
    ModelScenario, PolicyDisposition, RefereeConfig, RefereeError, RobustPolicyReport,
    evaluate_policy_robustly,
};
use subterra_survival_reference::{
    CostPrices, MeasurementAction, MeasurementOutcome, MissionDecision, OutcomeKind, PlannerConfig,
    PlanningError, PlanningProblem, PlanningReport, ResourceBudget, ResourceCost, RiskLimits,
    plan_survival_conditioned,
};

const EPSILON: f64 = 1.0e-10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectionTask {
    PipeCable,
    DrawpointVolume,
    SupportVerification,
}

impl InspectionTask {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::PipeCable => "pipe_cable",
            Self::DrawpointVolume => "drawpoint_volume",
            Self::SupportVerification => "support_verification",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct InspectionCase {
    pub case_id: String,
    pub task: InspectionTask,
    pub primary_model: PlanningProblem,
    /// Includes the primary model and any competing prior/likelihood models.
    pub evaluation_models: Vec<ModelScenario>,
    pub budget: ResourceBudget,
    pub prices: CostPrices,
    pub planner_config: PlannerConfig,
    pub referee_config: RefereeConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NoTestBaseline {
    pub decision_id: String,
    pub expected_decision_loss: f64,
    pub prior_entropy_nats: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionBaseline {
    pub action_id: String,
    pub expected_information_gain_nats: f64,
    pub expected_posterior_decision_loss: f64,
    /// Gross EVSI. Resource costs remain separate and are not silently
    /// subtracted from decision-loss units.
    pub gross_evsi: f64,
    pub resource_cost: ResourceCost,
    pub mission_interruption_probability: f64,
    pub robot_loss_probability: f64,
    pub eligible: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InspectionCaseReport {
    pub case_id: String,
    pub task: InspectionTask,
    pub no_test: NoTestBaseline,
    pub entropy_greedy_action_id: Option<String>,
    pub myopic_evsi_action_id: Option<String>,
    pub myopic_action_baselines: Vec<ActionBaseline>,
    pub planner: PlanningReport,
    pub planner_gross_evsi: f64,
    pub robust: RobustPolicyReport,
    pub robust_policy_admissible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdapterError {
    EmptyCaseId,
    MissingEvaluationModels,
    MissingPrimaryModel,
    DuplicateModelId,
    InvalidPriorOrLikelihood,
    InvalidDecisionLoss,
    InvalidAction,
    NumericalFailure(&'static str),
    Planner(String),
    Referee(String),
}

impl std::fmt::Display for AdapterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for AdapterError {}

fn expected_loss(belief: &[f64], losses: &[f64]) -> Result<f64, AdapterError> {
    if belief.len() != losses.len() {
        return Err(AdapterError::InvalidDecisionLoss);
    }
    let value = belief.iter().zip(losses).map(|(p, l)| p * l).sum::<f64>();
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AdapterError::NumericalFailure("expected loss"))
    }
}

fn best_decision(problem: &PlanningProblem, belief: &[f64]) -> Result<(String, f64), AdapterError> {
    let mut decisions = problem.decisions.iter().collect::<Vec<_>>();
    decisions.sort_by(|a, b| a.id.cmp(&b.id));
    let mut best: Option<(String, f64)> = None;
    for decision in decisions {
        let risk = expected_loss(belief, &decision.loss_by_hypothesis)?;
        if best
            .as_ref()
            .is_none_or(|(_, current)| risk < *current - EPSILON)
        {
            best = Some((decision.id.clone(), risk));
        }
    }
    best.ok_or(AdapterError::InvalidDecisionLoss)
}

fn entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.ln())
        .sum()
}

fn posterior_and_probability(
    belief: &[f64],
    likelihood: &[f64],
) -> Result<(f64, Vec<f64>), AdapterError> {
    if belief.len() != likelihood.len() {
        return Err(AdapterError::InvalidPriorOrLikelihood);
    }
    let probability = belief
        .iter()
        .zip(likelihood)
        .map(|(p, l)| p * l)
        .sum::<f64>();
    if !probability.is_finite() || probability < 0.0 {
        return Err(AdapterError::NumericalFailure("outcome probability"));
    }
    if probability <= 0.0 {
        return Ok((0.0, vec![0.0; belief.len()]));
    }
    let posterior = belief
        .iter()
        .zip(likelihood)
        .map(|(p, l)| p * l / probability)
        .collect::<Vec<_>>();
    let sum = posterior.iter().sum::<f64>();
    if !sum.is_finite() || (sum - 1.0).abs() > 1.0e-8 {
        return Err(AdapterError::NumericalFailure("posterior normalization"));
    }
    Ok((probability, posterior))
}

fn validate_adapter_inputs(problem: &PlanningProblem) -> Result<Vec<f64>, AdapterError> {
    if problem.hypotheses.is_empty() || problem.decisions.is_empty() {
        return Err(AdapterError::InvalidPriorOrLikelihood);
    }
    let mut sum = 0.0;
    let mut ids = BTreeSet::new();
    for h in &problem.hypotheses {
        if h.id.trim().is_empty()
            || !ids.insert(h.id.as_str())
            || !h.prior_probability.is_finite()
            || !(0.0..=1.0).contains(&h.prior_probability)
        {
            return Err(AdapterError::InvalidPriorOrLikelihood);
        }
        sum += h.prior_probability;
    }
    if !sum.is_finite()
        || (sum - 1.0).abs() > 1.0e-8
        || !ids.contains("unknown")
        || problem
            .hypotheses
            .iter()
            .find(|h| h.id == "unknown")
            .is_some_and(|h| h.prior_probability <= 0.0)
    {
        return Err(AdapterError::InvalidPriorOrLikelihood);
    }
    let belief = problem
        .hypotheses
        .iter()
        .map(|h| h.prior_probability)
        .collect::<Vec<_>>();
    for decision in &problem.decisions {
        if decision.loss_by_hypothesis.len() != belief.len()
            || decision
                .loss_by_hypothesis
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(AdapterError::InvalidDecisionLoss);
        }
    }
    for action in &problem.actions {
        if action.id.trim().is_empty()
            || action.modality.trim().is_empty()
            || action.outcomes.is_empty()
        {
            return Err(AdapterError::InvalidAction);
        }
        let mut totals = vec![0.0; belief.len()];
        for outcome in &action.outcomes {
            if outcome.id.trim().is_empty()
                || outcome.likelihood_by_hypothesis.len() != belief.len()
            {
                return Err(AdapterError::InvalidPriorOrLikelihood);
            }
            for (index, probability) in outcome.likelihood_by_hypothesis.iter().enumerate() {
                if !probability.is_finite() || !(0.0..=1.0).contains(probability) {
                    return Err(AdapterError::InvalidPriorOrLikelihood);
                }
                totals[index] += probability;
            }
        }
        if totals
            .iter()
            .any(|total| !total.is_finite() || (*total - 1.0).abs() > 1.0e-8)
        {
            return Err(AdapterError::InvalidPriorOrLikelihood);
        }
    }
    Ok(belief)
}

fn assess_action(
    problem: &PlanningProblem,
    belief: &[f64],
    action: &MeasurementAction,
    no_test_risk: f64,
    prior_entropy: f64,
    budget: ResourceBudget,
    risk_limits: RiskLimits,
) -> Result<ActionBaseline, AdapterError> {
    let within_resource_budget = action.cost.energy_j <= budget.energy_j
        && action.cost.duration_s <= budget.duration_s
        && action.cost.data_bytes <= budget.data_bytes
        && action.cost.exposure_units <= budget.exposure_units;
    let mut expected_loss_sum = 0.0;
    let mut expected_entropy = 0.0;
    let mut mission_interruption_probability = 0.0;
    let mut robot_loss_probability = 0.0;
    for outcome in &action.outcomes {
        let (probability, posterior) =
            posterior_and_probability(belief, &outcome.likelihood_by_hypothesis)?;
        if probability <= 0.0 {
            continue;
        }
        expected_entropy += probability * entropy(&posterior);
        if matches!(
            outcome.kind,
            OutcomeKind::MissionInterrupted | OutcomeKind::RobotLost
        ) {
            mission_interruption_probability += probability;
        }
        if outcome.kind == OutcomeKind::RobotLost {
            robot_loss_probability += probability;
        }
        let branch_risk = if outcome.kind == OutcomeKind::Observation {
            best_decision(problem, &posterior)?.1
        } else {
            let terminal_losses = outcome
                .terminal_loss_by_hypothesis
                .as_deref()
                .ok_or(AdapterError::InvalidAction)?;
            expected_loss(&posterior, terminal_losses)?
        };
        expected_loss_sum += probability * branch_risk;
    }
    let information_gain = prior_entropy - expected_entropy;
    if !information_gain.is_finite() || !expected_loss_sum.is_finite() {
        return Err(AdapterError::NumericalFailure("action baseline"));
    }
    let eligible = action.external_gate_passed
        && within_resource_budget
        && mission_interruption_probability
            <= risk_limits.maximum_mission_interruption_probability + EPSILON
        && robot_loss_probability <= risk_limits.maximum_robot_loss_probability + EPSILON;
    Ok(ActionBaseline {
        action_id: action.id.clone(),
        expected_information_gain_nats: information_gain.max(0.0),
        expected_posterior_decision_loss: expected_loss_sum,
        gross_evsi: no_test_risk - expected_loss_sum,
        resource_cost: action.cost,
        mission_interruption_probability,
        robot_loss_probability,
        eligible,
    })
}

fn planner_config_to_referee(case: &InspectionCase) -> RefereeConfig {
    RefereeConfig {
        resource_budget: case.budget,
        cost_prices: case.prices,
        risk_limits: case.planner_config.risk_limits,
        maximum_models: case.referee_config.maximum_models,
        maximum_policy_nodes: case.referee_config.maximum_policy_nodes,
        maximum_policy_depth: case.referee_config.maximum_policy_depth,
    }
}

pub fn evaluate_inspection_case(
    case: &InspectionCase,
) -> Result<InspectionCaseReport, AdapterError> {
    if case.case_id.trim().is_empty() {
        return Err(AdapterError::EmptyCaseId);
    }
    if case.evaluation_models.is_empty() {
        return Err(AdapterError::MissingEvaluationModels);
    }
    if !case
        .evaluation_models
        .iter()
        .any(|model| model.problem == case.primary_model)
    {
        return Err(AdapterError::MissingPrimaryModel);
    }
    let belief = validate_adapter_inputs(&case.primary_model)?;
    let (no_test_decision, no_test_risk) = best_decision(&case.primary_model, &belief)?;
    let prior_entropy = entropy(&belief);
    let no_test = NoTestBaseline {
        decision_id: no_test_decision,
        expected_decision_loss: no_test_risk,
        prior_entropy_nats: prior_entropy,
    };

    let mut action_baselines = case
        .primary_model
        .actions
        .iter()
        .map(|action| {
            assess_action(
                &case.primary_model,
                &belief,
                action,
                no_test_risk,
                prior_entropy,
                case.budget,
                case.planner_config.risk_limits,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    action_baselines.sort_by(|a, b| a.action_id.cmp(&b.action_id));

    let entropy_greedy_action_id = action_baselines
        .iter()
        .filter(|a| a.eligible)
        .max_by(|a, b| {
            a.expected_information_gain_nats
                .total_cmp(&b.expected_information_gain_nats)
                .then_with(|| b.action_id.cmp(&a.action_id))
        })
        .map(|a| a.action_id.clone());
    let myopic_evsi_action_id = action_baselines
        .iter()
        .filter(|a| a.eligible && a.gross_evsi > EPSILON)
        .max_by(|a, b| {
            a.gross_evsi
                .total_cmp(&b.gross_evsi)
                .then_with(|| b.action_id.cmp(&a.action_id))
        })
        .map(|a| a.action_id.clone());

    let planner: PlanningReport = plan_survival_conditioned(
        &case.primary_model,
        case.budget,
        case.prices,
        case.planner_config,
    )
    .map_err(|e: PlanningError| AdapterError::Planner(e.to_string()))?;

    let planner_gross_evsi = no_test_risk - planner.metrics.expected_decision_loss;
    let robust: RobustPolicyReport = evaluate_policy_robustly(
        &case.evaluation_models,
        &planner.policy,
        planner_config_to_referee(case),
    )
    .map_err(|e: RefereeError| AdapterError::Referee(e.to_string()))?;
    let robust_policy_admissible = robust.disposition
        == PolicyDisposition::AdmissibleAcrossSuppliedModels
        && robust.all_external_gates_passed
        && robust.all_supported_paths_within_budget
        && robust.all_models_within_risk_limits;

    Ok(InspectionCaseReport {
        case_id: case.case_id.clone(),
        task: case.task,
        no_test,
        entropy_greedy_action_id,
        myopic_evsi_action_id,
        myopic_action_baselines: action_baselines,
        planner,
        planner_gross_evsi,
        robust,
        robust_policy_admissible,
    })
}

fn action(
    id: &str,
    modality: &str,
    good_a: [f64; 2],
    good_b: [f64; 2],
    defect: [f64; 2],
    unknown: [f64; 2],
    energy: f64,
) -> MeasurementAction {
    MeasurementAction {
        id: id.into(),
        modality: modality.into(),
        outcomes: vec![
            MeasurementOutcome {
                id: "outcome_a".into(),
                kind: OutcomeKind::Observation,
                likelihood_by_hypothesis: vec![good_a[0], good_b[0], defect[0], unknown[0]],
                terminal_loss_by_hypothesis: None,
            },
            MeasurementOutcome {
                id: "outcome_b".into(),
                kind: OutcomeKind::Observation,
                likelihood_by_hypothesis: vec![good_a[1], good_b[1], defect[1], unknown[1]],
                terminal_loss_by_hypothesis: None,
            },
        ],
        cost: ResourceCost {
            energy_j: energy,
            duration_s: 10.0,
            data_bytes: 100.0,
            exposure_units: 0.01,
        },
        external_gate_passed: true,
    }
}

/// Deterministic synthetic example. The first action sharply identifies two
/// harmless subtypes (high EIG, zero EVSI); the second weakly detects a
/// consequential defect (lower EIG, positive EVSI). It is a math/control
/// fixture, not a real sensor-performance model.
pub fn synthetic_decision_boundary_case(task: InspectionTask) -> InspectionCase {
    let hypotheses = vec![
        subterra_survival_reference::Hypothesis {
            id: "good_variant_a".into(),
            prior_probability: 0.45,
        },
        subterra_survival_reference::Hypothesis {
            id: "good_variant_b".into(),
            prior_probability: 0.45,
        },
        subterra_survival_reference::Hypothesis {
            id: "consequential_defect".into(),
            prior_probability: 0.08,
        },
        subterra_survival_reference::Hypothesis {
            id: "unknown".into(),
            prior_probability: 0.02,
        },
    ];
    let decisions = vec![
        MissionDecision {
            id: "continue_monitoring".into(),
            loss_by_hypothesis: vec![0.0, 0.0, 100.0, 50.0],
        },
        MissionDecision {
            id: "hold_for_specialist_review".into(),
            loss_by_hypothesis: vec![10.0, 10.0, 10.0, 0.0],
        },
    ];
    let (task_prefix, high_modality, low_modality) = match task {
        InspectionTask::PipeCable => (
            "pipe_cable",
            "high_resolution_rgb",
            "thermal_or_acoustic_followup",
        ),
        InspectionTask::DrawpointVolume => {
            ("drawpoint", "dense_depth_cloud", "alternate_stand_off_view")
        }
        InspectionTask::SupportVerification => {
            ("support", "surface_rgb_scan", "independent_metrology")
        }
    };
    let high_information = action(
        &format!("{task_prefix}_high_eig"),
        high_modality,
        [0.999, 0.001],
        [0.001, 0.999],
        [0.5, 0.5],
        [0.5, 0.5],
        120.0,
    );
    let low_information_high_value = action(
        &format!("{task_prefix}_decision_evidence"),
        low_modality,
        [0.95, 0.05],
        [0.95, 0.05],
        [0.80, 0.20],
        [0.90, 0.10],
        35.0,
    );
    let problem = PlanningProblem {
        hypotheses,
        decisions,
        actions: vec![high_information, low_information_high_value],
    };
    let model = ModelScenario {
        model_id: "nominal_synthetic".into(),
        problem: problem.clone(),
    };
    let mut shifted = problem.clone();
    shifted.hypotheses[2].prior_probability = 0.10;
    shifted.hypotheses[0].prior_probability = 0.44;
    shifted.hypotheses[1].prior_probability = 0.44;
    let robust_models = vec![
        model,
        ModelScenario {
            model_id: "shifted_prior_synthetic".into(),
            problem: shifted,
        },
    ];
    let budget = ResourceBudget {
        energy_j: 500.0,
        duration_s: 120.0,
        data_bytes: 2_000.0,
        exposure_units: 0.2,
    };
    let prices = CostPrices {
        energy_per_j: 0.0,
        time_per_s: 0.0,
        data_per_byte: 0.0,
        exposure_per_unit: 0.0,
    };
    let risk_limits = RiskLimits {
        maximum_mission_interruption_probability: 0.10,
        maximum_robot_loss_probability: 0.05,
    };
    let planner_config = PlannerConfig {
        horizon_steps: 1,
        maximum_search_nodes: 10_000,
        maximum_actions: 8,
        maximum_outcomes_per_action: 8,
        risk_limits,
    };
    let referee_config = RefereeConfig {
        resource_budget: budget,
        cost_prices: prices,
        risk_limits,
        maximum_models: 8,
        maximum_policy_nodes: 10_000,
        maximum_policy_depth: 8,
    };
    InspectionCase {
        case_id: format!(
            "{}_high_eig_zero_evsi_vs_low_eig_positive_evsi",
            task.as_str()
        ),
        task,
        primary_model: problem,
        evaluation_models: robust_models,
        budget,
        prices,
        planner_config,
        referee_config,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_entropy_gain_can_have_zero_decision_value() {
        let case = synthetic_decision_boundary_case(InspectionTask::PipeCable);
        let report = evaluate_inspection_case(&case).unwrap();
        let high = report
            .myopic_action_baselines
            .iter()
            .find(|a| a.action_id.ends_with("high_eig"))
            .unwrap();
        assert!((high.expected_information_gain_nats - 0.6167159329029419).abs() < 1.0e-10);
        assert!(high.gross_evsi.abs() < 1.0e-10);
        assert_eq!(
            report.entropy_greedy_action_id.as_deref(),
            Some(high.action_id.as_str())
        );
        assert_ne!(
            report.myopic_evsi_action_id.as_deref(),
            Some(high.action_id.as_str())
        );
    }

    #[test]
    fn modest_information_can_have_positive_decision_value() {
        let case = synthetic_decision_boundary_case(InspectionTask::PipeCable);
        let report = evaluate_inspection_case(&case).unwrap();
        let high = report
            .myopic_action_baselines
            .iter()
            .find(|a| a.action_id.ends_with("high_eig"))
            .unwrap();
        let decision = report
            .myopic_action_baselines
            .iter()
            .find(|a| a.action_id.ends_with("decision_evidence"))
            .unwrap();
        assert!(decision.expected_information_gain_nats < high.expected_information_gain_nats);
        assert!((decision.expected_information_gain_nats - 0.009945983399899605).abs() < 1.0e-10);
        assert!((decision.gross_evsi - 1.09).abs() < 1.0e-10);
        assert_eq!(
            report.myopic_evsi_action_id.as_deref(),
            Some(decision.action_id.as_str())
        );
    }

    #[test]
    fn bounded_planner_selects_decision_relevant_evidence_not_entropy_only() {
        let case = synthetic_decision_boundary_case(InspectionTask::PipeCable);
        let report = evaluate_inspection_case(&case).unwrap();
        assert_eq!(
            report.planner.policy.measurement_action_id.as_deref(),
            report.myopic_evsi_action_id.as_deref()
        );
        assert!(report.planner_gross_evsi > 0.0);
        assert!(report.robust_policy_admissible);
        assert_eq!(report.robust.model_count, 2);
    }

    #[test]
    fn independent_referee_detects_cached_branches_under_prior_shift() {
        let report =
            evaluate_inspection_case(&synthetic_decision_boundary_case(InspectionTask::PipeCable))
                .unwrap();
        let shifted = report
            .robust
            .per_model
            .iter()
            .find(|model| model.model_id == "shifted_prior_synthetic")
            .unwrap();
        assert!(shifted.cached_branch_mismatches > 0);
        assert!(shifted.external_gates_passed);
        assert!(shifted.all_supported_paths_within_budget);
    }

    #[test]
    fn primary_policy_model_must_be_in_the_referee_model_set() {
        let mut case = synthetic_decision_boundary_case(InspectionTask::PipeCable);
        case.evaluation_models.clear();
        assert_eq!(
            evaluate_inspection_case(&case),
            Err(AdapterError::MissingEvaluationModels)
        );
        let mut case = synthetic_decision_boundary_case(InspectionTask::PipeCable);
        case.evaluation_models
            .retain(|model| model.model_id != "nominal_synthetic");
        assert_eq!(
            evaluate_inspection_case(&case),
            Err(AdapterError::MissingPrimaryModel)
        );
    }

    #[test]
    fn same_policy_is_refereed_across_pipe_drawpoint_and_support_adapters() {
        for task in [
            InspectionTask::PipeCable,
            InspectionTask::DrawpointVolume,
            InspectionTask::SupportVerification,
        ] {
            let report = evaluate_inspection_case(&synthetic_decision_boundary_case(task)).unwrap();
            assert_eq!(report.task, task);
            assert!(report.robust_policy_admissible);
        }
    }

    #[test]
    fn ineligible_high_value_action_does_not_win() {
        let mut case = synthetic_decision_boundary_case(InspectionTask::PipeCable);
        case.primary_model.actions[1].external_gate_passed = false;
        case.evaluation_models[0].problem.actions[1].external_gate_passed = false;
        case.evaluation_models[1].problem.actions[1].external_gate_passed = false;
        let report = evaluate_inspection_case(&case).unwrap();
        assert_ne!(
            report.planner.policy.measurement_action_id.as_deref(),
            Some("pipe_cable_decision_evidence")
        );
        assert!(report.planner.policy.measurement_action_id.is_none());
    }

    #[test]
    fn entropy_and_evsi_baselines_respect_robot_loss_limits() {
        let mut case = synthetic_decision_boundary_case(InspectionTask::PipeCable);
        for model in std::iter::once(&mut case.primary_model).chain(
            case.evaluation_models
                .iter_mut()
                .map(|model| &mut model.problem),
        ) {
            let high = model
                .actions
                .iter_mut()
                .find(|action| action.id == "pipe_cable_high_eig")
                .unwrap();
            for outcome in &mut high.outcomes {
                for probability in &mut outcome.likelihood_by_hypothesis {
                    *probability *= 0.94;
                }
            }
            high.outcomes.push(MeasurementOutcome {
                id: "robot_lost".into(),
                kind: OutcomeKind::RobotLost,
                likelihood_by_hypothesis: vec![0.06; 4],
                terminal_loss_by_hypothesis: Some(vec![100.0; 4]),
            });
        }
        let report = evaluate_inspection_case(&case).unwrap();
        let high = report
            .myopic_action_baselines
            .iter()
            .find(|action| action.action_id == "pipe_cable_high_eig")
            .unwrap();
        assert!((high.robot_loss_probability - 0.06).abs() < EPSILON);
        assert!(!high.eligible);
        assert_ne!(
            report.planner.policy.measurement_action_id.as_deref(),
            Some("pipe_cable_high_eig")
        );
    }

    #[test]
    fn missing_or_zero_prior_unknown_hypothesis_is_rejected() {
        let mut case = synthetic_decision_boundary_case(InspectionTask::PipeCable);
        case.primary_model.hypotheses[3].prior_probability = 0.0;
        assert_eq!(
            validate_adapter_inputs(&case.primary_model),
            Err(AdapterError::InvalidPriorOrLikelihood)
        );
    }
}
