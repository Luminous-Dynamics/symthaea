//! Decision-directed, cost-aware control for optical ISRU campaigns.
//!
//! This is a standalone decision-theory reference, not a live instrument controller.
//! It selects among measurements and interventions by expected reduction in
//! Bayes decision risk per campaign cost. Safety/evidence gates are evaluated
//! before optimization and cannot be traded against information gain.
//!
//! All probabilities, losses, costs and gate states are caller-supplied. A real
//! campaign must bind them to versioned, calibrated evidence and a separately
//! authorized safety system.

use std::collections::BTreeSet;

const PROBABILITY_TOLERANCE: f64 = 1.0e-9;
const NUMERIC_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CampaignActionKind {
    MaterialExperiment,
    OpticalReference,
    QcmWitnessInspection,
    Calibration,
    CleaningRequalification,
    IndependentAssay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Gate {
    SampleIdentityValid,
    InstrumentCalibrated,
    OpticalPathQualified,
    ThermalEnvelopeSafe,
    ReferenceChannelValid,
    QcmWitnessChainValid,
    CleaningProtocolQualified,
    CalibrationStandardValid,
    IndependentAssayChainValid,
    IndependentAssayPassed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SafetyState {
    pub sample_identity_valid: bool,
    pub instrument_calibrated: bool,
    pub optical_path_qualified: bool,
    pub thermal_envelope_safe: bool,
    pub reference_channel_valid: bool,
    pub qcm_witness_chain_valid: bool,
    pub cleaning_protocol_qualified: bool,
    pub calibration_standard_valid: bool,
    pub independent_assay_chain_valid: bool,
    pub independent_assay_passed: bool,
}

impl SafetyState {
    pub fn satisfies(&self, gate: Gate) -> bool {
        match gate {
            Gate::SampleIdentityValid => self.sample_identity_valid,
            Gate::InstrumentCalibrated => self.instrument_calibrated,
            Gate::OpticalPathQualified => self.optical_path_qualified,
            Gate::ThermalEnvelopeSafe => self.thermal_envelope_safe,
            Gate::ReferenceChannelValid => self.reference_channel_valid,
            Gate::QcmWitnessChainValid => self.qcm_witness_chain_valid,
            Gate::CleaningProtocolQualified => self.cleaning_protocol_qualified,
            Gate::CalibrationStandardValid => self.calibration_standard_valid,
            Gate::IndependentAssayChainValid => self.independent_assay_chain_valid,
            Gate::IndependentAssayPassed => self.independent_assay_passed,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hypothesis {
    pub id: String,
    pub prior_probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionOption {
    pub id: String,
    /// Loss for each hypothesis, in the same order as CampaignProblem.hypotheses.
    pub loss_by_hypothesis: Vec<f64>,
    /// A decision that needs unavailable evidence is excluded, not penalized.
    pub required_gates: Vec<Gate>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationOutcome {
    pub id: String,
    /// P(this outcome | each hypothesis), in hypothesis order.
    pub likelihood_by_hypothesis: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResourceCost {
    pub energy_wh: f64,
    pub elapsed_seconds: f64,
    pub consumables_g: f64,
    pub experiment_equivalents: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostWeights {
    /// Explicit conversion to a common campaign-cost unit. These weights are
    /// benchmark configuration, not universal physical/economic constants.
    pub energy_per_unit: f64,
    pub seconds_per_unit: f64,
    pub consumables_per_unit: f64,
    pub experiment_equivalents_per_unit: f64,
}

impl ResourceCost {
    pub fn weighted_units(&self, weights: CostWeights) -> f64 {
        self.energy_wh * weights.energy_per_unit
            + self.elapsed_seconds * weights.seconds_per_unit
            + self.consumables_g * weights.consumables_per_unit
            + self.experiment_equivalents * weights.experiment_equivalents_per_unit
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionOption {
    pub id: String,
    pub kind: CampaignActionKind,
    pub cost: ResourceCost,
    pub required_gates: Vec<Gate>,
    pub outcomes: Vec<ObservationOutcome>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CampaignProblem {
    pub hypotheses: Vec<Hypothesis>,
    pub decisions: Vec<DecisionOption>,
    pub actions: Vec<ActionOption>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyConfig {
    /// Converts campaign-cost units into decision-loss units. This is an
    /// explicit value judgement that must be preregistered for comparisons.
    pub cost_loss_weight: f64,
    /// Ignore actions whose net expected value is below this threshold.
    pub minimum_net_value: f64,
    /// Minimum net expected value per weighted resource-cost unit.
    pub minimum_net_value_per_cost: f64,
    pub cost_weights: CostWeights,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PolicyError {
    EmptyHypotheses,
    EmptyDecisions,
    EmptyIdentifier,
    DuplicateIdentifier,
    InvalidPrior,
    InvalidLoss,
    InvalidLikelihood,
    OutcomeLikelihoodsDoNotNormalize,
    InvalidCost,
    InvalidCostWeights,
    InvalidPolicyConfig,
    NoEligibleTerminalDecision,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionEvaluation {
    pub action_id: String,
    pub kind: CampaignActionKind,
    pub eligible: bool,
    pub blockers: Vec<Gate>,
    pub weighted_cost_units: Option<f64>,
    pub expected_value_of_sample_information: Option<f64>,
    pub gross_value_per_cost: Option<f64>,
    pub net_value: Option<f64>,
    pub net_value_per_cost: Option<f64>,
    pub selected: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolicyResult {
    pub current_best_decision_id: String,
    pub current_bayes_risk: f64,
    /// None means no candidate action cleared the preregistered value threshold;
    /// the caller should hold/pause rather than inventing an experiment.
    pub selected_action_id: Option<String>,
    pub evaluations: Vec<ActionEvaluation>,
    pub simulation_only: bool,
}

pub struct DecisionDirectedPolicy {
    pub config: PolicyConfig,
}

impl DecisionDirectedPolicy {
    pub fn evaluate(
        &self,
        problem: &CampaignProblem,
        safety: &SafetyState,
    ) -> Result<PolicyResult, PolicyError> {
        validate_config(self.config)?;
        validate_problem(problem, self.config.cost_weights)?;

        let prior: Vec<f64> = problem
            .hypotheses
            .iter()
            .map(|hypothesis| hypothesis.prior_probability)
            .collect();

        let eligible_decisions: Vec<usize> = problem
            .decisions
            .iter()
            .enumerate()
            .filter_map(|(index, decision)| {
                decision
                    .required_gates
                    .iter()
                    .all(|gate| safety.satisfies(*gate))
                    .then_some(index)
            })
            .collect();

        let (baseline_decision_index, baseline_risk) =
            best_decision(&prior, &problem.decisions, &eligible_decisions)
                .ok_or(PolicyError::NoEligibleTerminalDecision)?;

        let mut evaluations = Vec::with_capacity(problem.actions.len());
        let mut best_action: Option<(usize, f64, String)> = None;

        for (index, action) in problem.actions.iter().enumerate() {
            let blockers: Vec<Gate> = action
                .required_gates
                .iter()
                .copied()
                .filter(|gate| !safety.satisfies(*gate))
                .collect();

            if !blockers.is_empty() {
                evaluations.push(ActionEvaluation {
                    action_id: action.id.clone(),
                    kind: action.kind,
                    eligible: false,
                    blockers,
                    weighted_cost_units: None,
                    expected_value_of_sample_information: None,
                    gross_value_per_cost: None,
                    net_value: None,
                    net_value_per_cost: None,
                    selected: false,
                });
                continue;
            }

            let cost_units = action.cost.weighted_units(self.config.cost_weights);
            let evsi = expected_value_of_sample_information(
                &prior,
                baseline_risk,
                &problem.decisions,
                &eligible_decisions,
                &action.outcomes,
            );
            let gross_value_per_cost = evsi / cost_units;
            let net_value = evsi - self.config.cost_loss_weight * cost_units;
            let net_value_per_cost = net_value / cost_units;

            if net_value > self.config.minimum_net_value
                && net_value_per_cost >= self.config.minimum_net_value_per_cost
            {
                let replace = match &best_action {
                    None => true,
                    Some((_, best_score, best_id)) => {
                        net_value_per_cost > *best_score + NUMERIC_TOLERANCE
                            || ((net_value_per_cost - *best_score).abs() <= NUMERIC_TOLERANCE
                                && action.id < *best_id)
                    }
                };
                if replace {
                    best_action = Some((index, net_value_per_cost, action.id.clone()));
                }
            }

            evaluations.push(ActionEvaluation {
                action_id: action.id.clone(),
                kind: action.kind,
                eligible: true,
                blockers: Vec::new(),
                weighted_cost_units: Some(cost_units),
                expected_value_of_sample_information: Some(evsi),
                gross_value_per_cost: Some(gross_value_per_cost),
                net_value: Some(net_value),
                net_value_per_cost: Some(net_value_per_cost),
                selected: false,
            });
        }

        let selected_action_id = best_action.map(|(index, _, _)| {
            evaluations[index].selected = true;
            problem.actions[index].id.clone()
        });

        // Stable output ordering makes receipts and replay comparisons deterministic.
        evaluations.sort_by(|left, right| left.action_id.cmp(&right.action_id));

        Ok(PolicyResult {
            current_best_decision_id: problem.decisions[baseline_decision_index].id.clone(),
            current_bayes_risk: baseline_risk,
            selected_action_id,
            evaluations,
            simulation_only: true,
        })
    }
}

fn validate_config(config: PolicyConfig) -> Result<(), PolicyError> {
    let weights = config.cost_weights;
    let weight_values = [
        weights.energy_per_unit,
        weights.seconds_per_unit,
        weights.consumables_per_unit,
        weights.experiment_equivalents_per_unit,
    ];
    if weight_values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || !weight_values.iter().any(|value| *value > 0.0)
    {
        return Err(PolicyError::InvalidCostWeights);
    }
    if !config.cost_loss_weight.is_finite()
        || config.cost_loss_weight < 0.0
        || !config.minimum_net_value.is_finite()
        || config.minimum_net_value < 0.0
        || !config.minimum_net_value_per_cost.is_finite()
        || config.minimum_net_value_per_cost < 0.0
    {
        return Err(PolicyError::InvalidPolicyConfig);
    }
    Ok(())
}

fn validate_problem(problem: &CampaignProblem, weights: CostWeights) -> Result<(), PolicyError> {
    if problem.hypotheses.is_empty() {
        return Err(PolicyError::EmptyHypotheses);
    }
    if problem.decisions.is_empty() {
        return Err(PolicyError::EmptyDecisions);
    }

    let mut hypothesis_ids = BTreeSet::new();
    let mut prior_sum = 0.0;
    for hypothesis in &problem.hypotheses {
        if hypothesis.id.trim().is_empty() {
            return Err(PolicyError::EmptyIdentifier);
        }
        if !hypothesis_ids.insert(hypothesis.id.as_str()) {
            return Err(PolicyError::DuplicateIdentifier);
        }
        let prior = hypothesis.prior_probability;
        if !prior.is_finite() || prior < 0.0 || prior > 1.0 {
            return Err(PolicyError::InvalidPrior);
        }
        prior_sum += prior;
    }
    if (prior_sum - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(PolicyError::InvalidPrior);
    }

    let hypothesis_count = problem.hypotheses.len();
    let mut decision_ids = BTreeSet::new();
    for decision in &problem.decisions {
        if decision.id.trim().is_empty() {
            return Err(PolicyError::EmptyIdentifier);
        }
        if !decision_ids.insert(decision.id.as_str()) {
            return Err(PolicyError::DuplicateIdentifier);
        }
        if decision.loss_by_hypothesis.len() != hypothesis_count
            || decision
                .loss_by_hypothesis
                .iter()
                .any(|loss| !loss.is_finite() || *loss < 0.0)
        {
            return Err(PolicyError::InvalidLoss);
        }
    }

    let mut action_ids = BTreeSet::new();
    for action in &problem.actions {
        if action.id.trim().is_empty() {
            return Err(PolicyError::EmptyIdentifier);
        }
        if !action_ids.insert(action.id.as_str()) {
            return Err(PolicyError::DuplicateIdentifier);
        }
        let cost_values = [
            action.cost.energy_wh,
            action.cost.elapsed_seconds,
            action.cost.consumables_g,
            action.cost.experiment_equivalents,
        ];
        if cost_values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
            || !action.cost.weighted_units(weights).is_finite()
            || action.cost.weighted_units(weights) <= 0.0
        {
            return Err(PolicyError::InvalidCost);
        }
        if action.outcomes.is_empty() {
            return Err(PolicyError::InvalidLikelihood);
        }

        let mut outcome_ids = BTreeSet::new();
        let mut likelihood_sums = vec![0.0; hypothesis_count];
        for outcome in &action.outcomes {
            if outcome.id.trim().is_empty() {
                return Err(PolicyError::EmptyIdentifier);
            }
            if !outcome_ids.insert(outcome.id.as_str()) {
                return Err(PolicyError::DuplicateIdentifier);
            }
            if outcome.likelihood_by_hypothesis.len() != hypothesis_count {
                return Err(PolicyError::InvalidLikelihood);
            }
            for (hypothesis_index, likelihood) in
                outcome.likelihood_by_hypothesis.iter().enumerate()
            {
                if !likelihood.is_finite() || *likelihood < 0.0 || *likelihood > 1.0 {
                    return Err(PolicyError::InvalidLikelihood);
                }
                likelihood_sums[hypothesis_index] += likelihood;
            }
        }
        if likelihood_sums
            .iter()
            .any(|sum| !sum.is_finite() || (*sum - 1.0).abs() > PROBABILITY_TOLERANCE)
        {
            return Err(PolicyError::OutcomeLikelihoodsDoNotNormalize);
        }
    }
    Ok(())
}

fn best_decision(
    probabilities: &[f64],
    decisions: &[DecisionOption],
    eligible_decisions: &[usize],
) -> Option<(usize, f64)> {
    eligible_decisions
        .iter()
        .copied()
        .fold(None, |best, index| {
            let risk = expected_loss(probabilities, &decisions[index].loss_by_hypothesis);
            match best {
                None => Some((index, risk)),
                Some((best_index, best_risk))
                    if index != best_index && risk < best_risk - NUMERIC_TOLERANCE =>
                {
                    Some((index, risk))
                }
                Some((best_index, best_risk))
                    if (risk - best_risk).abs() <= NUMERIC_TOLERANCE
                        && decisions[index].id < decisions[best_index].id =>
                {
                    Some((index, risk))
                }
                _ => best,
            }
        })
}

fn expected_loss(probabilities: &[f64], losses: &[f64]) -> f64 {
    probabilities
        .iter()
        .zip(losses)
        .map(|(probability, loss)| probability * loss)
        .sum()
}

fn expected_value_of_sample_information(
    prior: &[f64],
    baseline_risk: f64,
    decisions: &[DecisionOption],
    eligible_decisions: &[usize],
    outcomes: &[ObservationOutcome],
) -> f64 {
    let mut expected_posterior_risk = 0.0;

    for outcome in outcomes {
        let outcome_probability = prior
            .iter()
            .zip(&outcome.likelihood_by_hypothesis)
            .map(|(probability, likelihood)| probability * likelihood)
            .sum::<f64>();

        if outcome_probability <= NUMERIC_TOLERANCE {
            continue;
        }

        let posterior: Vec<f64> = prior
            .iter()
            .zip(&outcome.likelihood_by_hypothesis)
            .map(|(probability, likelihood)| probability * likelihood / outcome_probability)
            .collect();

        let (_, posterior_risk) = best_decision(&posterior, decisions, eligible_decisions)
            .expect("baseline validation guarantees an eligible decision");
        expected_posterior_risk += outcome_probability * posterior_risk;
    }

    // Information cannot worsen Bayes risk when the decision-maker may ignore it.
    // Clamp floating-point residue, but surface any material violation in tests.
    (baseline_risk - expected_posterior_risk).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights() -> CostWeights {
        CostWeights {
            energy_per_unit: 1.0,
            seconds_per_unit: 0.1,
            consumables_per_unit: 0.5,
            experiment_equivalents_per_unit: 2.0,
        }
    }

    fn safety() -> SafetyState {
        SafetyState {
            sample_identity_valid: true,
            instrument_calibrated: true,
            optical_path_qualified: true,
            thermal_envelope_safe: true,
            reference_channel_valid: true,
            qcm_witness_chain_valid: true,
            cleaning_protocol_qualified: true,
            calibration_standard_valid: true,
            independent_assay_chain_valid: true,
            independent_assay_passed: false,
        }
    }

    fn config() -> PolicyConfig {
        PolicyConfig {
            cost_loss_weight: 0.1,
            minimum_net_value: 1.0e-9,
            minimum_net_value_per_cost: 0.0,
            cost_weights: weights(),
        }
    }

    fn problem() -> CampaignProblem {
        CampaignProblem {
            hypotheses: vec![
                Hypothesis {
                    id: "clean".to_string(),
                    prior_probability: 0.7,
                },
                Hypothesis {
                    id: "fouled".to_string(),
                    prior_probability: 0.3,
                },
            ],
            decisions: vec![
                DecisionOption {
                    id: "accept-product".to_string(),
                    loss_by_hypothesis: vec![0.0, 12.0],
                    required_gates: vec![
                        Gate::SampleIdentityValid,
                        Gate::IndependentAssayChainValid,
                        Gate::IndependentAssayPassed,
                    ],
                },
                DecisionOption {
                    id: "reject-product".to_string(),
                    loss_by_hypothesis: vec![4.0, 0.0],
                    required_gates: Vec::new(),
                },
            ],
            actions: vec![
                ActionOption {
                    id: "material-run".to_string(),
                    kind: CampaignActionKind::MaterialExperiment,
                    cost: ResourceCost {
                        energy_wh: 2.0,
                        elapsed_seconds: 20.0,
                        consumables_g: 1.0,
                        experiment_equivalents: 1.0,
                    },
                    required_gates: vec![
                        Gate::SampleIdentityValid,
                        Gate::InstrumentCalibrated,
                        Gate::OpticalPathQualified,
                        Gate::ThermalEnvelopeSafe,
                        Gate::ReferenceChannelValid,
                    ],
                    outcomes: vec![
                        ObservationOutcome {
                            id: "high-proxy".to_string(),
                            likelihood_by_hypothesis: vec![0.75, 0.25],
                        },
                        ObservationOutcome {
                            id: "low-proxy".to_string(),
                            likelihood_by_hypothesis: vec![0.25, 0.75],
                        },
                    ],
                },
                ActionOption {
                    id: "qcm-witness".to_string(),
                    kind: CampaignActionKind::QcmWitnessInspection,
                    cost: ResourceCost {
                        energy_wh: 0.1,
                        elapsed_seconds: 1.0,
                        consumables_g: 0.0,
                        experiment_equivalents: 0.0,
                    },
                    required_gates: vec![Gate::InstrumentCalibrated, Gate::QcmWitnessChainValid],
                    outcomes: vec![
                        ObservationOutcome {
                            id: "clean-reading".to_string(),
                            likelihood_by_hypothesis: vec![0.95, 0.05],
                        },
                        ObservationOutcome {
                            id: "fouled-reading".to_string(),
                            likelihood_by_hypothesis: vec![0.05, 0.95],
                        },
                    ],
                },
            ],
        }
    }

    #[test]
    fn selects_low_cost_diagnostic_over_lower_information_material_run() {
        let result = DecisionDirectedPolicy { config: config() }
            .evaluate(&problem(), &safety())
            .unwrap();
        assert_eq!(result.selected_action_id.as_deref(), Some("qcm-witness"));
        assert_eq!(result.current_best_decision_id, "reject-product");
        assert!(result.simulation_only);
    }

    #[test]
    fn hard_gate_blocks_material_action_without_blocking_diagnostics() {
        let mut state = safety();
        state.thermal_envelope_safe = false;
        let result = DecisionDirectedPolicy { config: config() }
            .evaluate(&problem(), &state)
            .unwrap();
        assert_eq!(result.selected_action_id.as_deref(), Some("qcm-witness"));
        let material = result
            .evaluations
            .iter()
            .find(|evaluation| evaluation.action_id == "material-run")
            .unwrap();
        assert!(!material.eligible);
        assert_eq!(material.blockers, vec![Gate::ThermalEnvelopeSafe]);
    }

    #[test]
    fn product_acceptance_is_not_an_available_decision_without_assay_pass() {
        let result = DecisionDirectedPolicy { config: config() }
            .evaluate(&problem(), &safety())
            .unwrap();
        assert_ne!(result.current_best_decision_id, "accept-product");
    }

    #[test]
    fn product_acceptance_becomes_eligible_only_after_gate_is_satisfied() {
        let mut state = safety();
        state.independent_assay_passed = true;
        let result = DecisionDirectedPolicy { config: config() }
            .evaluate(&problem(), &state)
            .unwrap();
        assert_eq!(result.current_best_decision_id, "reject-product");
        assert!(result.current_bayes_risk.is_finite());
    }

    #[test]
    fn uninformative_action_does_not_trigger_experiment() {
        let mut candidate = problem();
        candidate.actions = vec![ActionOption {
            id: "uninformative".to_string(),
            kind: CampaignActionKind::OpticalReference,
            cost: ResourceCost {
                energy_wh: 1.0,
                elapsed_seconds: 1.0,
                consumables_g: 0.1,
                experiment_equivalents: 0.0,
            },
            required_gates: Vec::new(),
            outcomes: vec![ObservationOutcome {
                id: "same".to_string(),
                likelihood_by_hypothesis: vec![1.0, 1.0],
            }],
        }];
        let result = DecisionDirectedPolicy { config: config() }
            .evaluate(&candidate, &safety())
            .unwrap();
        assert_eq!(result.selected_action_id, None);
        assert_eq!(
            result.evaluations[0].expected_value_of_sample_information,
            Some(0.0)
        );
    }

    #[test]
    fn invalid_outcome_model_is_rejected_instead_of_renormalized() {
        let mut candidate = problem();
        candidate.actions[0].outcomes[0].likelihood_by_hypothesis = vec![0.9, 0.2];
        candidate.actions[0].outcomes[1].likelihood_by_hypothesis = vec![0.2, 0.8];
        let result = DecisionDirectedPolicy { config: config() }.evaluate(&candidate, &safety());
        assert_eq!(result, Err(PolicyError::OutcomeLikelihoodsDoNotNormalize));
    }

    #[test]
    fn impossible_terminal_decision_set_fails_closed() {
        let mut candidate = problem();
        candidate.decisions[1].required_gates = vec![Gate::IndependentAssayPassed];
        let result = DecisionDirectedPolicy { config: config() }.evaluate(&candidate, &safety());
        assert_eq!(result, Err(PolicyError::NoEligibleTerminalDecision));
    }

    #[test]
    fn evaluation_order_is_deterministic() {
        let result = DecisionDirectedPolicy { config: config() }
            .evaluate(&problem(), &safety())
            .unwrap();
        let ids: Vec<&str> = result
            .evaluations
            .iter()
            .map(|evaluation| evaluation.action_id.as_str())
            .collect();
        assert_eq!(ids, vec!["material-run", "qcm-witness"]);
    }
}
