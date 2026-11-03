//! Robust value-of-information planning under a shared set of likelihood models.
//!
//! The point-estimate planner remains the baseline. This module evaluates each
//! candidate under the same weighted likelihood scenarios and reports expected,
//! worst-case and lower-tail information gain. Scenario weights and likelihoods
//! are caller-supplied and are not calibrated or authenticated here.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::{
    Authorization, EvidenceProvenance, Hypothesis, LikelihoodRow, PlannerConfig, PlannerError,
    PlannerErrorKind, PredictedOutcome, RejectionReason, TargetImpact, evaluate_experiment,
    validate_config, validate_evidence_catalog, validate_experiment, validate_hypotheses,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustRankingPolicy {
    ExpectedInformationGain,
    WorstCaseInformationGain,
    LowerTailCvar,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RobustPlannerConfig {
    pub base: PlannerConfig,
    pub ranking_policy: RobustRankingPolicy,
    /// Probability mass included in the lower-tail conditional mean, in (0, 1].
    pub lower_tail_probability: f64,
    /// Minimum fraction of likelihood scenarios in which a candidate must rank first.
    pub minimum_top1_stability: f64,
}

impl Default for RobustPlannerConfig {
    fn default() -> Self {
        Self {
            base: PlannerConfig::default(),
            ranking_policy: RobustRankingPolicy::LowerTailCvar,
            lower_tail_probability: 0.25,
            minimum_top1_stability: 0.5,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustExperimentProposal {
    pub id: String,
    pub outcomes: Vec<String>,
    pub authorization: Authorization,
    pub reversible: bool,
    pub destructive: bool,
    pub target_impact: TargetImpact,
    pub estimated_time_seconds: f64,
    pub estimated_energy_joules: f64,
    pub estimated_risk: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioExperimentLikelihood {
    pub experiment_id: String,
    pub likelihoods: Vec<LikelihoodRow>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LikelihoodModelScenario {
    pub scenario_id: String,
    pub scenario_probability: f64,
    /// Exactly one likelihood matrix for every proposed experiment.
    pub experiments: Vec<ScenarioExperimentLikelihood>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ScenarioInformationGain {
    pub scenario_id: String,
    pub scenario_probability: f64,
    pub expected_information_gain_bits: f64,
    pub scenario_utility: f64,
    pub predicted_outcomes: Vec<PredictedOutcome>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustRankedExperiment {
    pub experiment_id: String,
    pub expected_information_gain_bits: f64,
    pub worst_case_information_gain_bits: f64,
    pub lower_tail_cvar_information_gain_bits: f64,
    pub information_gain_variance_bits_squared: f64,
    pub selected_information_gain_bits: f64,
    pub normalized_cost: f64,
    pub utility: f64,
    pub top1_rank_stability: f64,
    pub top3_rank_stability: f64,
    pub scenario_results: Vec<ScenarioInformationGain>,
    pub support_evidence_ids: Vec<String>,
    pub independent_root_evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RobustRejectionReason {
    AuthorizationUnknown,
    NotAuthorized,
    Irreversible,
    Destructive,
    TargetImpactExceeded,
    TimeBudgetExceeded,
    EnergyBudgetExceeded,
    RiskBudgetExceeded,
    InvalidOutcomeSpace,
    NoRobustInformationGain,
    RankingUnstable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RobustRejectedExperiment {
    pub experiment_id: String,
    pub reason: RobustRejectionReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustAbstentionReason {
    NoCompetingHypotheses,
    NoIndependentlyGroundedHypothesis,
    NoEligibleExperiment,
    NoDiscriminatingExperiment,
    UnstableRanking,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RobustClaimCeiling {
    pub evidence_producer_authenticated: bool,
    pub likelihood_scenarios_calibrated: bool,
    pub scenario_weights_authenticated: bool,
    pub causal_truth_proven: bool,
    pub experiment_executed: bool,
    pub target_mutated: bool,
}

impl Default for RobustClaimCeiling {
    fn default() -> Self {
        Self {
            evidence_producer_authenticated: false,
            likelihood_scenarios_calibrated: false,
            scenario_weights_authenticated: false,
            causal_truth_proven: false,
            experiment_executed: false,
            target_mutated: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustInvestigationPlan {
    pub selected: Option<RobustRankedExperiment>,
    pub ranked_experiments: Vec<RobustRankedExperiment>,
    pub rejected_experiments: Vec<RobustRejectedExperiment>,
    pub abstention: Option<RobustAbstentionReason>,
    pub prior_entropy_bits: f64,
    pub grounded_hypothesis_ids: Vec<String>,
    pub scenario_ids: Vec<String>,
    pub claim_ceiling: RobustClaimCeiling,
}

struct CandidateScenario {
    result: ScenarioInformationGain,
    normalized_cost: f64,
}

impl RobustExperimentProposal {
    fn with_likelihoods(&self, likelihoods: Vec<LikelihoodRow>) -> super::ExperimentProposal {
        super::ExperimentProposal {
            id: self.id.clone(),
            outcomes: self.outcomes.clone(),
            likelihoods,
            authorization: self.authorization,
            reversible: self.reversible,
            destructive: self.destructive,
            target_impact: self.target_impact,
            estimated_time_seconds: self.estimated_time_seconds,
            estimated_energy_joules: self.estimated_energy_joules,
            estimated_risk: self.estimated_risk,
        }
    }
}

fn robust_config_valid(config: RobustPlannerConfig) -> bool {
    config.lower_tail_probability.is_finite()
        && config.lower_tail_probability > 0.0
        && config.lower_tail_probability <= 1.0
        && config.minimum_top1_stability.is_finite()
        && (0.0..=1.0).contains(&config.minimum_top1_stability)
}

fn validate_scenarios(
    scenarios: &[LikelihoodModelScenario],
    experiment_ids: &[&str],
) -> Result<(), PlannerError> {
    if scenarios.is_empty() {
        return Err(PlannerError::new(
            PlannerErrorKind::InvalidLikelihoods,
            "robust planning requires at least one likelihood scenario",
        ));
    }
    let mut scenario_ids = HashSet::new();
    let mut total_probability = 0.0;
    for scenario in scenarios {
        if scenario.scenario_id.trim().is_empty()
            || !scenario_ids.insert(scenario.scenario_id.as_str())
            || !scenario.scenario_probability.is_finite()
            || scenario.scenario_probability <= 0.0
        {
            return Err(PlannerError::new(
                PlannerErrorKind::InvalidLikelihoods,
                "likelihood scenarios require unique non-empty IDs and positive finite probabilities",
            ));
        }
        total_probability += scenario.scenario_probability;
        if scenario.experiments.len() != experiment_ids.len() {
            return Err(PlannerError::new(
                PlannerErrorKind::InvalidLikelihoods,
                format!(
                    "scenario {} must define exactly one likelihood matrix per experiment",
                    scenario.scenario_id
                ),
            ));
        }
        let mut seen = HashSet::new();
        for experiment in &scenario.experiments {
            if !experiment_ids.contains(&experiment.experiment_id.as_str())
                || !seen.insert(experiment.experiment_id.as_str())
            {
                return Err(PlannerError::new(
                    PlannerErrorKind::InvalidLikelihoods,
                    format!(
                        "scenario {} contains an unknown or duplicate experiment likelihood",
                        scenario.scenario_id
                    ),
                ));
            }
        }
        if experiment_ids.iter().any(|id| !seen.contains(id)) {
            return Err(PlannerError::new(
                PlannerErrorKind::InvalidLikelihoods,
                format!(
                    "scenario {} omits a proposed experiment likelihood",
                    scenario.scenario_id
                ),
            ));
        }
    }
    if (total_probability - 1.0).abs() > 1.0e-9 {
        return Err(PlannerError::new(
            PlannerErrorKind::InvalidLikelihoods,
            format!(
                "likelihood scenario probabilities must sum to 1.0 (received {total_probability:.12})"
            ),
        ));
    }
    Ok(())
}

fn lower_tail_cvar(samples: &[(f64, f64)], alpha: f64) -> f64 {
    let mut ordered = samples.to_vec();
    ordered.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut remaining = alpha;
    let mut weighted_value = 0.0;
    for (value, probability) in ordered {
        if remaining <= 0.0 {
            break;
        }
        let included = probability.min(remaining);
        weighted_value += value * included;
        remaining -= included;
    }
    weighted_value / alpha
}

fn variance(samples: &[(f64, f64)], mean: f64) -> f64 {
    samples
        .iter()
        .map(|(value, probability)| probability * (value - mean).powi(2))
        .sum()
}

/// Plan the next experiment using one shared, weighted set of plausible
/// likelihood models. The base point-EIG API is unchanged.
///
/// Every scenario must cover every candidate experiment and use the same
/// hypothesis/outcome contracts. Scenario weights and likelihoods are caller
/// inputs: they are not calibration receipts, authenticated observations, or
/// guarantees that the scenario set contains the true data-generating process.
pub fn plan_next_experiment_robust(
    evidence: &[EvidenceProvenance],
    hypotheses: &[Hypothesis],
    experiments: &[RobustExperimentProposal],
    scenarios: &[LikelihoodModelScenario],
    config: RobustPlannerConfig,
) -> Result<RobustInvestigationPlan, PlannerError> {
    validate_config(config.base)?;
    if !robust_config_valid(config) {
        return Err(PlannerError::new(
            PlannerErrorKind::InvalidPlannerConfig,
            "robust EIG tail probability must be in (0, 1] and minimum top-1 stability in [0, 1]",
        ));
    }

    let catalog = validate_evidence_catalog(evidence)?;
    let (priors, grounded_ids, support_ids, independent_root_ids) =
        match validate_hypotheses(hypotheses, &catalog) {
            Ok(result) => result,
            Err(error) if error.kind == PlannerErrorKind::UngroundedHypothesis => {
                return Ok(RobustInvestigationPlan {
                    selected: None,
                    ranked_experiments: Vec::new(),
                    rejected_experiments: Vec::new(),
                    abstention: Some(RobustAbstentionReason::NoIndependentlyGroundedHypothesis),
                    prior_entropy_bits: 0.0,
                    grounded_hypothesis_ids: Vec::new(),
                    scenario_ids: Vec::new(),
                    claim_ceiling: RobustClaimCeiling::default(),
                });
            }
            Err(error) => return Err(error),
        };

    let prior_entropy_bits =
        super::entropy_bits(priors.values().copied().collect::<Vec<_>>().into_iter());
    let hypothesis_ids: Vec<&str> = priors.keys().copied().collect();

    let mut unique_experiment_ids = HashSet::new();
    for experiment in experiments {
        if experiment.id.trim().is_empty() || !unique_experiment_ids.insert(experiment.id.as_str())
        {
            return Err(PlannerError::new(
                PlannerErrorKind::DuplicateExperimentId,
                format!("empty or duplicate robust experiment ID: {}", experiment.id),
            ));
        }
    }
    let experiment_ids: Vec<&str> = experiments.iter().map(|item| item.id.as_str()).collect();
    validate_scenarios(scenarios, &experiment_ids)?;

    let mut rejected_experiments = Vec::new();
    let mut scenario_results: BTreeMap<String, Vec<CandidateScenario>> = BTreeMap::new();

    if hypotheses.len() < 2 {
        return Ok(RobustInvestigationPlan {
            selected: None,
            ranked_experiments: Vec::new(),
            rejected_experiments,
            abstention: Some(RobustAbstentionReason::NoCompetingHypotheses),
            prior_entropy_bits,
            grounded_hypothesis_ids: grounded_ids,
            scenario_ids: {
                let mut ids: Vec<String> = scenarios
                    .iter()
                    .map(|scenario| scenario.scenario_id.clone())
                    .collect();
                ids.sort();
                ids
            },
            claim_ceiling: RobustClaimCeiling::default(),
        });
    }

    for experiment in experiments {
        let mut candidate_scenarios = Vec::with_capacity(scenarios.len());
        let mut rejection = None;
        for scenario in scenarios {
            let scenario_experiment = scenario
                .experiments
                .iter()
                .find(|item| item.experiment_id == experiment.id)
                .expect("scenario coverage checked before planning");
            let point = experiment.with_likelihoods(scenario_experiment.likelihoods.clone());
            match validate_experiment(&point, &hypothesis_ids, config.base) {
                Ok(()) => {}
                Err(reason) => {
                    if reason == RejectionReason::InvalidLikelihoods {
                        return Err(PlannerError::new(
                            PlannerErrorKind::InvalidLikelihoods,
                            format!(
                                "scenario {} has invalid likelihoods for experiment {}",
                                scenario.scenario_id, experiment.id
                            ),
                        ));
                    }
                    rejection = Some(reason);
                    break;
                }
            }
            let evaluated = evaluate_experiment(
                &point,
                &priors,
                config.base,
                &support_ids,
                &independent_root_ids,
            );
            candidate_scenarios.push(CandidateScenario {
                result: ScenarioInformationGain {
                    scenario_id: scenario.scenario_id.clone(),
                    scenario_probability: scenario.scenario_probability,
                    expected_information_gain_bits: evaluated.expected_information_gain_bits,
                    scenario_utility: evaluated.utility,
                    predicted_outcomes: evaluated.predicted_outcomes,
                },
                normalized_cost: evaluated.normalized_cost,
            });
        }
        if let Some(reason) = rejection {
            // A metadata/budget policy rejection applies to every scenario.
            let mapped = match reason {
                RejectionReason::AuthorizationUnknown => {
                    RobustRejectionReason::AuthorizationUnknown
                }
                RejectionReason::NotAuthorized => RobustRejectionReason::NotAuthorized,
                RejectionReason::Irreversible => RobustRejectionReason::Irreversible,
                RejectionReason::Destructive => RobustRejectionReason::Destructive,
                RejectionReason::TargetImpactExceeded => {
                    RobustRejectionReason::TargetImpactExceeded
                }
                RejectionReason::TimeBudgetExceeded => RobustRejectionReason::TimeBudgetExceeded,
                RejectionReason::EnergyBudgetExceeded => {
                    RobustRejectionReason::EnergyBudgetExceeded
                }
                RejectionReason::RiskBudgetExceeded => RobustRejectionReason::RiskBudgetExceeded,
                RejectionReason::InvalidOutcomeSpace => RobustRejectionReason::InvalidOutcomeSpace,
                RejectionReason::InvalidLikelihoods
                | RejectionReason::NoPredictedInformationGain => {
                    RobustRejectionReason::NoRobustInformationGain
                }
            };
            rejected_experiments.push(RobustRejectedExperiment {
                experiment_id: experiment.id.clone(),
                reason: mapped,
            });
            continue;
        }
        scenario_results.insert(experiment.id.clone(), candidate_scenarios);
    }

    // For rank stability, each shared likelihood scenario ranks the same
    // candidate set using the base planner's scenario-specific utility.
    let eligible_ids: Vec<String> = scenario_results.keys().cloned().collect();
    let mut top1_weights: HashMap<String, f64> = HashMap::new();
    let mut top3_weights: HashMap<String, f64> = HashMap::new();

    for scenario in scenarios {
        let mut ranked: Vec<(String, f64, f64)> = eligible_ids
            .iter()
            .filter_map(|id| {
                scenario_results.get(id).and_then(|items| {
                    items
                        .iter()
                        .find(|item| item.result.scenario_id == scenario.scenario_id)
                        .map(|item| {
                            (
                                id.clone(),
                                item.result.scenario_utility,
                                item.result.expected_information_gain_bits,
                            )
                        })
                })
            })
            .collect();
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.2.total_cmp(&left.2))
                .then_with(|| left.0.cmp(&right.0))
        });
        if let Some((id, _, _)) = ranked.first() {
            *top1_weights.entry(id.clone()).or_default() += scenario.scenario_probability;
        }
        for (id, _, _) in ranked.iter().take(3) {
            *top3_weights.entry(id.clone()).or_default() += scenario.scenario_probability;
        }
    }

    let mut ranked_experiments = Vec::new();
    for (experiment_id, items) in scenario_results {
        let samples: Vec<(f64, f64)> = items
            .iter()
            .map(|item| {
                (
                    item.result.expected_information_gain_bits,
                    item.result.scenario_probability,
                )
            })
            .collect();
        let expected_information_gain_bits = samples
            .iter()
            .map(|(value, probability)| value * probability)
            .sum::<f64>();
        let worst_case_information_gain_bits = samples
            .iter()
            .map(|(value, _)| *value)
            .fold(f64::INFINITY, f64::min);
        let lower_tail_cvar_information_gain_bits =
            lower_tail_cvar(&samples, config.lower_tail_probability);
        let information_gain_variance_bits_squared =
            variance(&samples, expected_information_gain_bits);
        let selected_information_gain_bits = match config.ranking_policy {
            RobustRankingPolicy::ExpectedInformationGain => expected_information_gain_bits,
            RobustRankingPolicy::WorstCaseInformationGain => worst_case_information_gain_bits,
            RobustRankingPolicy::LowerTailCvar => lower_tail_cvar_information_gain_bits,
        };
        let normalized_cost = items[0].normalized_cost;
        let top1_rank_stability = top1_weights.get(&experiment_id).copied().unwrap_or(0.0);
        let top3_rank_stability = top3_weights.get(&experiment_id).copied().unwrap_or(0.0);
        if selected_information_gain_bits <= config.base.minimum_information_gain_bits {
            rejected_experiments.push(RobustRejectedExperiment {
                experiment_id,
                reason: RobustRejectionReason::NoRobustInformationGain,
            });
            continue;
        }
        ranked_experiments.push(RobustRankedExperiment {
            experiment_id,
            expected_information_gain_bits,
            worst_case_information_gain_bits,
            lower_tail_cvar_information_gain_bits,
            information_gain_variance_bits_squared,
            selected_information_gain_bits,
            normalized_cost,
            utility: selected_information_gain_bits / normalized_cost.max(1.0e-12),
            top1_rank_stability,
            top3_rank_stability,
            scenario_results: items.into_iter().map(|item| item.result).collect(),
            support_evidence_ids: support_ids.clone(),
            independent_root_evidence_ids: independent_root_ids.clone(),
        });
    }

    ranked_experiments.sort_by(|left, right| {
        right
            .utility
            .total_cmp(&left.utility)
            .then_with(|| {
                right
                    .selected_information_gain_bits
                    .total_cmp(&left.selected_information_gain_bits)
            })
            .then_with(|| left.experiment_id.cmp(&right.experiment_id))
    });
    rejected_experiments.sort_by(|left, right| left.experiment_id.cmp(&right.experiment_id));

    for candidate in &mut ranked_experiments {
        candidate
            .scenario_results
            .sort_by(|left, right| left.scenario_id.cmp(&right.scenario_id));
    }
    let selected = ranked_experiments
        .iter()
        .find(|candidate| candidate.top1_rank_stability >= config.minimum_top1_stability)
        .cloned();
    for candidate in &ranked_experiments {
        if candidate.top1_rank_stability < config.minimum_top1_stability {
            rejected_experiments.push(RobustRejectedExperiment {
                experiment_id: candidate.experiment_id.clone(),
                reason: RobustRejectionReason::RankingUnstable,
            });
        }
    }
    let abstention = match &selected {
        Some(_) => None,
        None if ranked_experiments.iter().any(|candidate| {
            candidate.selected_information_gain_bits > config.base.minimum_information_gain_bits
        }) =>
        {
            Some(RobustAbstentionReason::UnstableRanking)
        }
        None if experiments.is_empty() => Some(RobustAbstentionReason::NoEligibleExperiment),
        None if rejected_experiments
            .iter()
            .all(|item| item.reason == RobustRejectionReason::NoRobustInformationGain) =>
        {
            Some(RobustAbstentionReason::NoDiscriminatingExperiment)
        }
        None => Some(RobustAbstentionReason::NoEligibleExperiment),
    };

    rejected_experiments.sort_by(|left, right| left.experiment_id.cmp(&right.experiment_id));

    let mut scenario_ids: Vec<String> = scenarios
        .iter()
        .map(|scenario| scenario.scenario_id.clone())
        .collect();
    scenario_ids.sort();

    Ok(RobustInvestigationPlan {
        selected,
        ranked_experiments,
        rejected_experiments,
        abstention,
        prior_entropy_bits,
        grounded_hypothesis_ids: grounded_ids,
        scenario_ids,
        claim_ceiling: RobustClaimCeiling::default(),
    })
}

#[cfg(test)]
mod tests {
    use super::super::{
        EvidenceOrigin, EvidenceProvenance, ExperimentProposal, Hypothesis, LikelihoodRow,
        PlannerConfig, TargetImpact,
    };
    use super::*;

    fn evidence_id(ch: char) -> String {
        format!("ev_{}", ch.to_string().repeat(64))
    }

    fn evidence() -> Vec<EvidenceProvenance> {
        vec![EvidenceProvenance {
            evidence_id: evidence_id('a'),
            origin: EvidenceOrigin::IndependentObservation,
            parent_evidence_ids: Vec::new(),
        }]
    }

    fn hypotheses() -> Vec<Hypothesis> {
        vec![
            Hypothesis {
                id: "cause-a".to_owned(),
                prior_probability: 0.5,
                supporting_evidence_ids: vec![evidence_id('a')],
                contradicting_evidence_ids: Vec::new(),
            },
            Hypothesis {
                id: "cause-b".to_owned(),
                prior_probability: 0.5,
                supporting_evidence_ids: vec![evidence_id('a')],
                contradicting_evidence_ids: Vec::new(),
            },
        ]
    }

    fn proposal(id: &str) -> RobustExperimentProposal {
        RobustExperimentProposal {
            id: id.to_owned(),
            outcomes: vec!["yes".to_owned(), "no".to_owned()],
            authorization: Authorization::Authorized,
            reversible: true,
            destructive: false,
            target_impact: TargetImpact::ReadOnlyObservation,
            estimated_time_seconds: 10.0,
            estimated_energy_joules: 10.0,
            estimated_risk: 0.01,
        }
    }

    fn matrix(experiment_id: &str, a_yes: f64, b_yes: f64) -> ScenarioExperimentLikelihood {
        ScenarioExperimentLikelihood {
            experiment_id: experiment_id.to_owned(),
            likelihoods: vec![
                LikelihoodRow {
                    hypothesis_id: "cause-a".to_owned(),
                    outcome_probabilities: vec![a_yes, 1.0 - a_yes],
                },
                LikelihoodRow {
                    hypothesis_id: "cause-b".to_owned(),
                    outcome_probabilities: vec![b_yes, 1.0 - b_yes],
                },
            ],
        }
    }

    fn scenarios_for(experiments: Vec<ScenarioExperimentLikelihood>) -> LikelihoodModelScenario {
        LikelihoodModelScenario {
            scenario_id: "nominal".to_owned(),
            scenario_probability: 1.0,
            experiments,
        }
    }

    #[test]
    fn one_hypothesis_abstains_with_empty_ranked_results_and_valid_mass() {
        let only = vec![Hypothesis {
            id: "cause-a".to_owned(),
            prior_probability: 1.0,
            supporting_evidence_ids: vec![evidence_id('a')],
            contradicting_evidence_ids: Vec::new(),
        }];
        let proposal = proposal("probe");
        let scenario = LikelihoodModelScenario {
            scenario_id: "nominal".to_owned(),
            scenario_probability: 1.0,
            experiments: vec![ScenarioExperimentLikelihood {
                experiment_id: "probe".to_owned(),
                likelihoods: vec![LikelihoodRow {
                    hypothesis_id: "cause-a".to_owned(),
                    outcome_probabilities: vec![0.5, 0.5],
                }],
            }],
        };

        let plan = plan_next_experiment_robust(
            &evidence(),
            &only,
            &[proposal],
            &[scenario],
            RobustPlannerConfig::default(),
        )
        .unwrap();

        assert!(plan.selected.is_none());
        assert!(plan.ranked_experiments.is_empty());
        assert!(plan.rejected_experiments.is_empty());
        assert_eq!(
            plan.abstention,
            Some(RobustAbstentionReason::NoCompetingHypotheses)
        );
        assert_eq!(plan.scenario_ids, vec!["nominal"]);
    }

    #[test]
    fn robust_policy_reports_expected_worst_case_and_lower_tail_eig() {
        let scenarios = vec![
            LikelihoodModelScenario {
                scenario_id: "optimistic".to_owned(),
                scenario_probability: 0.5,
                experiments: vec![matrix("probe", 0.9, 0.1)],
            },
            LikelihoodModelScenario {
                scenario_id: "uninformative".to_owned(),
                scenario_probability: 0.5,
                experiments: vec![matrix("probe", 0.6, 0.4)],
            },
        ];
        let plan = plan_next_experiment_robust(
            &evidence(),
            &hypotheses(),
            &[proposal("probe")],
            &scenarios,
            RobustPlannerConfig {
                base: PlannerConfig::default(),
                ranking_policy: RobustRankingPolicy::LowerTailCvar,
                lower_tail_probability: 0.25,
                minimum_top1_stability: 0.0,
            },
        )
        .unwrap();
        let selected = plan.selected.unwrap();
        assert!((selected.expected_information_gain_bits - 0.2800269059780251).abs() < 1.0e-12);
        assert!((selected.worst_case_information_gain_bits - 0.02904940554533142).abs() < 1.0e-12);
        assert!(
            (selected.lower_tail_cvar_information_gain_bits - 0.02904940554533142).abs() < 1.0e-12
        );
        assert!(selected.information_gain_variance_bits_squared > 0.0);
        assert_eq!(selected.scenario_results.len(), 2);
    }

    #[test]
    fn lower_tail_policy_prefers_a_consistently_useful_probe_over_a_fragile_one() {
        let scenarios = vec![
            LikelihoodModelScenario {
                scenario_id: "nominal".to_owned(),
                scenario_probability: 0.5,
                experiments: vec![matrix("fragile", 0.9, 0.1), matrix("steady", 0.75, 0.25)],
            },
            LikelihoodModelScenario {
                scenario_id: "shifted".to_owned(),
                scenario_probability: 0.5,
                experiments: vec![matrix("fragile", 0.55, 0.45), matrix("steady", 0.75, 0.25)],
            },
        ];
        let plan = plan_next_experiment_robust(
            &evidence(),
            &hypotheses(),
            &[proposal("fragile"), proposal("steady")],
            &scenarios,
            RobustPlannerConfig {
                minimum_top1_stability: 0.0,
                ..RobustPlannerConfig::default()
            },
        )
        .unwrap();
        assert_eq!(plan.selected.unwrap().experiment_id, "steady");
        let fragile = plan
            .ranked_experiments
            .iter()
            .find(|item| item.experiment_id == "fragile")
            .unwrap();
        assert!(fragile.lower_tail_cvar_information_gain_bits > 1.0e-6);
        assert!((fragile.top1_rank_stability - 0.5).abs() < 1.0e-12);
        assert!(fragile.top1_rank_stability < 1.0);
    }

    #[test]
    fn unstable_top_candidate_abstains_when_minimum_stability_is_not_met() {
        let scenarios = vec![
            LikelihoodModelScenario {
                scenario_id: "nominal".to_owned(),
                scenario_probability: 0.5,
                experiments: vec![matrix("probe-a", 0.9, 0.1), matrix("probe-b", 0.75, 0.25)],
            },
            LikelihoodModelScenario {
                scenario_id: "shifted".to_owned(),
                scenario_probability: 0.5,
                experiments: vec![matrix("probe-a", 0.75, 0.25), matrix("probe-b", 0.9, 0.1)],
            },
        ];
        let plan = plan_next_experiment_robust(
            &evidence(),
            &hypotheses(),
            &[proposal("probe-a"), proposal("probe-b")],
            &scenarios,
            RobustPlannerConfig {
                minimum_top1_stability: 0.75,
                ..RobustPlannerConfig::default()
            },
        )
        .unwrap();
        assert!(plan.selected.is_none());
        assert_eq!(
            plan.abstention,
            Some(RobustAbstentionReason::UnstableRanking)
        );
    }

    #[test]
    fn rejects_incomplete_scenario_coverage_and_bad_probability_mass() {
        let incomplete = vec![scenarios_for(Vec::new())];
        assert_eq!(
            plan_next_experiment_robust(
                &evidence(),
                &hypotheses(),
                &[proposal("probe")],
                &incomplete,
                RobustPlannerConfig::default(),
            )
            .unwrap_err()
            .kind,
            PlannerErrorKind::InvalidLikelihoods
        );
        let bad_weight = vec![LikelihoodModelScenario {
            scenario_id: "nominal".to_owned(),
            scenario_probability: 0.8,
            experiments: vec![matrix("probe", 0.9, 0.1)],
        }];
        assert_eq!(
            plan_next_experiment_robust(
                &evidence(),
                &hypotheses(),
                &[proposal("probe")],
                &bad_weight,
                RobustPlannerConfig::default(),
            )
            .unwrap_err()
            .kind,
            PlannerErrorKind::InvalidLikelihoods
        );
    }

    #[test]
    fn scenario_and_proposal_permutations_preserve_semantic_ranking() {
        let scenarios = vec![
            LikelihoodModelScenario {
                scenario_id: "nominal".to_owned(),
                scenario_probability: 0.7,
                experiments: vec![matrix("probe-a", 0.9, 0.1), matrix("probe-b", 0.75, 0.25)],
            },
            LikelihoodModelScenario {
                scenario_id: "shift".to_owned(),
                scenario_probability: 0.3,
                experiments: vec![matrix("probe-a", 0.8, 0.2), matrix("probe-b", 0.6, 0.4)],
            },
        ];
        let proposals = vec![proposal("probe-a"), proposal("probe-b")];
        let first = plan_next_experiment_robust(
            &evidence(),
            &hypotheses(),
            &proposals,
            &scenarios,
            RobustPlannerConfig::default(),
        )
        .unwrap();
        let mut reversed_scenarios = scenarios.clone();
        reversed_scenarios.reverse();
        let mut reversed_proposals = proposals.clone();
        reversed_proposals.reverse();
        let second = plan_next_experiment_robust(
            &evidence(),
            &hypotheses(),
            &reversed_proposals,
            &reversed_scenarios,
            RobustPlannerConfig::default(),
        )
        .unwrap();
        assert_eq!(
            first.selected.as_ref().map(|item| &item.experiment_id),
            second.selected.as_ref().map(|item| &item.experiment_id)
        );
        assert_eq!(
            first
                .ranked_experiments
                .iter()
                .map(|item| &item.experiment_id)
                .collect::<Vec<_>>(),
            second
                .ranked_experiments
                .iter()
                .map(|item| &item.experiment_id)
                .collect::<Vec<_>>()
        );
    }
    #[test]
    fn authorization_and_budget_rejections_preserve_their_specific_reasons() {
        let scenarios = vec![scenarios_for(vec![matrix("probe", 0.9, 0.1)])];
        let mut unauthorized = proposal("probe");
        unauthorized.authorization = Authorization::Denied;
        let plan = plan_next_experiment_robust(
            &evidence(),
            &hypotheses(),
            &[unauthorized],
            &scenarios,
            RobustPlannerConfig::default(),
        )
        .unwrap();
        assert_eq!(
            plan.rejected_experiments[0].reason,
            RobustRejectionReason::NotAuthorized
        );

        let mut over_budget = proposal("probe");
        over_budget.estimated_time_seconds = 9999.0;
        let plan = plan_next_experiment_robust(
            &evidence(),
            &hypotheses(),
            &[over_budget],
            &scenarios,
            RobustPlannerConfig::default(),
        )
        .unwrap();
        assert_eq!(
            plan.rejected_experiments[0].reason,
            RobustRejectionReason::TimeBudgetExceeded
        );
    }

    #[test]
    fn invalid_robust_configuration_is_rejected() {
        let config = RobustPlannerConfig {
            lower_tail_probability: 0.0,
            ..RobustPlannerConfig::default()
        };
        assert_eq!(
            plan_next_experiment_robust(
                &evidence(),
                &hypotheses(),
                &[proposal("probe")],
                &[scenarios_for(vec![matrix("probe", 0.9, 0.1)])],
                config,
            )
            .unwrap_err()
            .kind,
            PlannerErrorKind::InvalidPlannerConfig
        );
    }

    #[test]
    fn expected_and_worst_case_policies_select_from_their_declared_objective() {
        let scenarios = vec![
            LikelihoodModelScenario {
                scenario_id: "fragile-high".to_owned(),
                scenario_probability: 0.5,
                experiments: vec![matrix("probe-a", 0.95, 0.05), matrix("probe-b", 0.75, 0.25)],
            },
            LikelihoodModelScenario {
                scenario_id: "fragile-zero".to_owned(),
                scenario_probability: 0.5,
                experiments: vec![matrix("probe-a", 0.5, 0.5), matrix("probe-b", 0.75, 0.25)],
            },
        ];
        let expected = plan_next_experiment_robust(
            &evidence(),
            &hypotheses(),
            &[proposal("probe-a"), proposal("probe-b")],
            &scenarios,
            RobustPlannerConfig {
                ranking_policy: RobustRankingPolicy::ExpectedInformationGain,
                minimum_top1_stability: 0.0,
                ..RobustPlannerConfig::default()
            },
        )
        .unwrap();
        assert_eq!(expected.selected.unwrap().experiment_id, "probe-a");

        let worst = plan_next_experiment_robust(
            &evidence(),
            &hypotheses(),
            &[proposal("probe-a"), proposal("probe-b")],
            &scenarios,
            RobustPlannerConfig {
                ranking_policy: RobustRankingPolicy::WorstCaseInformationGain,
                minimum_top1_stability: 0.0,
                ..RobustPlannerConfig::default()
            },
        )
        .unwrap();
        assert_eq!(worst.selected.unwrap().experiment_id, "probe-b");
    }

    #[test]
    fn uninformative_across_all_scenarios_abstains() {
        let scenarios = vec![
            LikelihoodModelScenario {
                scenario_id: "a".to_owned(),
                scenario_probability: 0.5,
                experiments: vec![matrix("probe", 0.5, 0.5)],
            },
            LikelihoodModelScenario {
                scenario_id: "b".to_owned(),
                scenario_probability: 0.5,
                experiments: vec![matrix("probe", 0.5, 0.5)],
            },
        ];
        let plan = plan_next_experiment_robust(
            &evidence(),
            &hypotheses(),
            &[proposal("probe")],
            &scenarios,
            RobustPlannerConfig::default(),
        )
        .unwrap();
        assert!(plan.selected.is_none());
        assert_eq!(
            plan.abstention,
            Some(RobustAbstentionReason::NoDiscriminatingExperiment)
        );
        assert_eq!(
            plan.rejected_experiments[0].reason,
            RobustRejectionReason::NoRobustInformationGain
        );
    }

    #[test]
    fn invalid_outcome_space_preserves_a_specific_rejection_reason() {
        let scenarios = vec![scenarios_for(vec![matrix("probe", 0.9, 0.1)])];
        let mut invalid = proposal("probe");
        invalid.outcomes = vec!["yes".to_owned()];
        let plan = plan_next_experiment_robust(
            &evidence(),
            &hypotheses(),
            &[invalid],
            &scenarios,
            RobustPlannerConfig::default(),
        )
        .unwrap();
        assert_eq!(
            plan.rejected_experiments[0].reason,
            RobustRejectionReason::InvalidOutcomeSpace
        );
    }

    #[test]
    fn scenario_ids_and_scenario_results_are_stably_sorted() {
        let scenarios = vec![
            LikelihoodModelScenario {
                scenario_id: "z-shift".to_owned(),
                scenario_probability: 0.5,
                experiments: vec![matrix("probe", 0.8, 0.2)],
            },
            LikelihoodModelScenario {
                scenario_id: "a-nominal".to_owned(),
                scenario_probability: 0.5,
                experiments: vec![matrix("probe", 0.9, 0.1)],
            },
        ];
        let plan = plan_next_experiment_robust(
            &evidence(),
            &hypotheses(),
            &[proposal("probe")],
            &scenarios,
            RobustPlannerConfig {
                minimum_top1_stability: 0.0,
                ..RobustPlannerConfig::default()
            },
        )
        .unwrap();
        assert_eq!(plan.scenario_ids, vec!["a-nominal", "z-shift"]);
        let selected = plan.selected.unwrap();
        assert_eq!(
            selected
                .scenario_results
                .iter()
                .map(|item| item.scenario_id.as_str())
                .collect::<Vec<_>>(),
            vec!["a-nominal", "z-shift"]
        );
    }
}
