//! Exhaustive paired policy campaign over the declared finite reference grammars.
//!
//! This runner evaluates transparent finite-inference baselines, not a learned
//! research agent. It enumerates every declared generating model and pairs A0
//! and A2 on identical truth and action budgets. Each arm may adaptively choose
//! its own post-start evidence trace (AdaptiveBudget design).

use crate::interaction;
use crate::numeric;
use crate::policy::{PolicyError, StrategyEnumeratorBaseline};
use crate::protocol::{
    AgentAction, BlindEpisode, ClaimDisposition, EvaluatorScore, StructureAccess, WorldFamily,
};
use crate::strategies::ProbeStrategy;
use crate::{Hypothesis, MeasurementMap, Mechanism, candidate_hypotheses};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy)]
enum GeneratingModel {
    Boolean(Hypothesis),
    Numeric(numeric::ModelHypothesis),
    Interaction(interaction::ModelHypothesis),
}

impl GeneratingModel {
    fn family(self) -> WorldFamily {
        match self {
            Self::Boolean(_) => WorldFamily::BooleanMechanism,
            Self::Numeric(_) => WorldFamily::NumericMechanism,
            Self::Interaction(_) => WorldFamily::TwoInputResponse,
        }
    }

    fn public_alias(self) -> &'static str {
        match self {
            Self::Boolean(_) => "boolean_mechanism",
            Self::Numeric(_) => "numeric_mechanism",
            Self::Interaction(_) => "two_input_response",
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PolicySpec {
    name: &'static str,
    strategy: ProbeStrategy,
}

fn policy_specs() -> [PolicySpec; 7] {
    [
        PolicySpec {
            name: "maximum_partition",
            strategy: ProbeStrategy::MaximumPartition,
        },
        PolicySpec {
            name: "minimax",
            strategy: ProbeStrategy::Minimax,
        },
        PolicySpec {
            name: "maximum_entropy",
            strategy: ProbeStrategy::MaximumEntropy,
        },
        PolicySpec {
            name: "minimum_expected_remaining",
            strategy: ProbeStrategy::MinimumExpectedRemaining,
        },
        PolicySpec {
            name: "maximum_structure_information_gain",
            strategy: ProbeStrategy::MaximumStructureInformationGain,
        },
        PolicySpec {
            name: "fixed_grid",
            strategy: ProbeStrategy::FixedGrid,
        },
        PolicySpec {
            name: "random_seed_23",
            strategy: ProbeStrategy::Random { seed: 23 },
        },
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CampaignArm {
    A0,
    A2,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmAggregate {
    pub episodes: usize,
    pub exact_version_space_matches: usize,
    pub exact_structure_set_matches: usize,
    pub generating_model_retained: usize,
    pub decision_correct: usize,
    pub oracle_singleton_model_cases: usize,
    pub oracle_identified_structure_cases: usize,
    pub abstentions: usize,
    pub safe_abstentions: usize,
    pub unnecessary_abstentions: usize,
    pub total_probe_actions: usize,
    pub total_probe_cost_units: u64,
    pub maximum_probe_actions_in_episode: usize,
}

impl ArmAggregate {
    fn new() -> Self {
        Self {
            episodes: 0,
            exact_version_space_matches: 0,
            exact_structure_set_matches: 0,
            generating_model_retained: 0,
            decision_correct: 0,
            oracle_singleton_model_cases: 0,
            oracle_identified_structure_cases: 0,
            abstentions: 0,
            safe_abstentions: 0,
            unnecessary_abstentions: 0,
            total_probe_actions: 0,
            total_probe_cost_units: 0,
            maximum_probe_actions_in_episode: 0,
        }
    }

    fn record(&mut self, score: &EvaluatorScore) {
        self.episodes += 1;
        self.exact_version_space_matches += usize::from(score.set_score.exact_version_space_match);
        self.exact_structure_set_matches +=
            usize::from(score.structure_score.exact_structure_set_match);
        self.generating_model_retained += usize::from(score.generating_model_in_version_space);
        self.decision_correct += usize::from(score.decision_correct);
        self.oracle_singleton_model_cases +=
            usize::from(score.set_score.oracle_model_count == 1);
        self.oracle_identified_structure_cases +=
            usize::from(score.structure_score.oracle_structure_identified);
        self.abstentions += usize::from(score.abstained);
        self.safe_abstentions += usize::from(score.safe_abstention);
        self.unnecessary_abstentions += usize::from(score.unnecessary_abstention);
        self.total_probe_actions += score.probe_count;
        self.total_probe_cost_units += score.probe_count as u64;
        self.maximum_probe_actions_in_episode =
            self.maximum_probe_actions_in_episode.max(score.probe_count);
    }

    pub fn exact_version_space_match_rate(&self) -> f64 {
        rate(self.exact_version_space_matches, self.episodes)
    }

    pub fn exact_structure_set_match_rate(&self) -> f64 {
        rate(self.exact_structure_set_matches, self.episodes)
    }

    pub fn mean_probe_actions(&self) -> f64 {
        if self.episodes == 0 {
            0.0
        } else {
            self.total_probe_actions as f64 / self.episodes as f64
        }
    }
}

fn rate(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairedPolicyResult {
    pub world_family: WorldFamily,
    pub generating_model_count: usize,
    pub action_budget: usize,
    pub policy: String,
    pub comparison_design: String,
    pub a0: ArmAggregate,
    pub a2: ArmAggregate,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FinitePolicyCampaignReport {
    pub report_version: String,
    pub campaign_kind: String,
    pub policy_kind: String,
    pub initial_observation_count: usize,
    pub candidate_truth_count: usize,
    pub policy_count: usize,
    pub pair_count: usize,
    pub results: Vec<PairedPolicyResult>,
    pub limitations: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FiniteCampaignError {
    InvalidBudget,
    Policy(String),
    Protocol(String),
    NotSubmitted,
}

impl fmt::Display for FiniteCampaignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for FiniteCampaignError {}

fn candidate_truths() -> Vec<GeneratingModel> {
    let mut truths = Vec::new();
    truths.extend(
        candidate_hypotheses(crate::StructureAccess::Withheld)
            .into_iter()
            .map(GeneratingModel::Boolean),
    );
    truths.extend(
        numeric::candidate_models(numeric::StructureAccess::Withheld)
            .into_iter()
            .map(GeneratingModel::Numeric),
    );
    truths.extend(
        interaction::candidate_models(interaction::StructureAccess::Withheld)
            .into_iter()
            .map(GeneratingModel::Interaction),
    );
    truths
}

fn default_budget(family: WorldFamily) -> usize {
    match family {
        WorldFamily::BooleanMechanism => 4,
        WorldFamily::NumericMechanism => 6,
        WorldFamily::TwoInputResponse => 9,
    }
}

fn run_arm(
    truth: GeneratingModel,
    arm: CampaignArm,
    policy: PolicySpec,
    budget: usize,
    episode_id: &str,
) -> Result<EvaluatorScore, FiniteCampaignError> {
    if budget == 0 {
        return Err(FiniteCampaignError::InvalidBudget);
    }

    let structure_access = match (truth, arm) {
        (GeneratingModel::Boolean(model), CampaignArm::A0) => {
            StructureAccess::BooleanSupplied(model.mechanism)
        }
        (GeneratingModel::Boolean(_), CampaignArm::A2) => StructureAccess::BooleanWithheld,
        (GeneratingModel::Numeric(model), CampaignArm::A0) => {
            StructureAccess::NumericSupplied(model.mechanism)
        }
        (GeneratingModel::Numeric(_), CampaignArm::A2) => StructureAccess::NumericWithheld,
        (GeneratingModel::Interaction(model), CampaignArm::A0) => {
            StructureAccess::InteractionSupplied(model.structure)
        }
        (GeneratingModel::Interaction(_), CampaignArm::A2) => {
            StructureAccess::InteractionWithheld
        }
    };

    let mut episode = match truth {
        GeneratingModel::Boolean(model) => BlindEpisode::new_boolean(
            episode_id,
            model.mechanism,
            model.measurement_map,
            model.initial_hidden_state,
            structure_access,
            budget,
        ),
        GeneratingModel::Numeric(model) => {
            BlindEpisode::new_numeric(episode_id, model, structure_access, budget)
        }
        GeneratingModel::Interaction(model) => {
            BlindEpisode::new_interaction(episode_id, model, structure_access, budget)
        }
    }
    .map_err(|error| FiniteCampaignError::Protocol(error.to_string()))?;

    // The strategy receives only the public view. Truth is retained by the
    // trusted episode and is not passed into StrategyEnumeratorBaseline.
    for _ in 0..=budget {
        if episode.status() == crate::protocol::EpisodeStatus::Submitted {
            break;
        }
        let view = episode
            .view()
            .map_err(|error| FiniteCampaignError::Protocol(error.to_string()))?;
        let action = StrategyEnumeratorBaseline::next_action(&view, policy.strategy)
            .map_err(|error: PolicyError| FiniteCampaignError::Policy(error.to_string()))?;
        let response = episode
            .apply(action)
            .map_err(|error| FiniteCampaignError::Protocol(error.to_string()))?;
        if response.status == crate::protocol::EpisodeStatus::Submitted {
            break;
        }
    }

    if episode.status() != crate::protocol::EpisodeStatus::Submitted {
        return Err(FiniteCampaignError::NotSubmitted);
    }
    episode
        .evaluate_after_submission()
        .map_err(|error| FiniteCampaignError::Protocol(error.to_string()))
}

/// Run an exhaustive census over every declared generating model and seven
/// transparent one-step policies. A0/A2 share world truth, empty starting
/// evidence and per-family action budgets; their adaptive trajectories may
/// diverge. The function never runs a learned model or executes external code.
pub fn run_exhaustive_finite_policy_campaign(
    budget_override: Option<usize>,
) -> Result<FinitePolicyCampaignReport, FiniteCampaignError> {
    if budget_override == Some(0) {
        return Err(FiniteCampaignError::InvalidBudget);
    }

    let truths = candidate_truths();
    let policies = policy_specs();
    let mut results = Vec::new();

    for family in [
        WorldFamily::BooleanMechanism,
        WorldFamily::NumericMechanism,
        WorldFamily::TwoInputResponse,
    ] {
        let family_truths = truths
            .iter()
            .copied()
            .filter(|truth| truth.family() == family)
            .collect::<Vec<_>>();
        let budget = budget_override.unwrap_or_else(|| default_budget(family));

        for policy in policies {
            let mut a0 = ArmAggregate::new();
            let mut a2 = ArmAggregate::new();

            for (index, truth) in family_truths.iter().copied().enumerate() {
                let pair = format!("{}-{}-{:04}", truth.public_alias(), policy.name, index);

                let score_a0 = run_arm(
                    truth,
                    CampaignArm::A0,
                    policy,
                    budget,
                    &format!("{pair}-a0"),
                )?;
                let score_a2 = run_arm(
                    truth,
                    CampaignArm::A2,
                    policy,
                    budget,
                    &format!("{pair}-a2"),
                )?;

                a0.record(&score_a0);
                a2.record(&score_a2);
            }

            results.push(PairedPolicyResult {
                world_family: family,
                generating_model_count: family_truths.len(),
                action_budget: budget,
                policy: policy.name.to_owned(),
                comparison_design: "adaptive_budget".to_owned(),
                a0,
                a2,
            });
        }
    }

    Ok(FinitePolicyCampaignReport {
        report_version: "symthaea-open-world-finite-policy-campaign-v0".to_owned(),
        campaign_kind: "exhaustive_generating-model_census".to_owned(),
        policy_kind: "StrategyEnumeratorBaseline (exact finite inference; not learned)".to_owned(),
        initial_observation_count: 0,
        candidate_truth_count: truths.len(),
        policy_count: policies.len(),
        pair_count: truths.len() * policies.len(),
        results,
        limitations: vec![
            "This is a deterministic census over the three declared finite reference grammars, not a sample of natural-science systems.".to_owned(),
            "The evaluated policy is the transparent exact finite inference baseline, not the learned Symthaea research orchestrator.".to_owned(),
            "A0 and A2 share world truth, empty starting evidence and maximum action budgets; their adaptive post-start traces may differ.".to_owned(),
            "Probe cost units are the reference worlds' declared unit cost; model inference, runtime and hardware costs are not included.".to_owned(),
            "No external benchmark, physical law or SOTA performance claim follows from this report.".to_owned(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_truth_census_covers_all_declared_models() {
        let truths = candidate_truths();
        assert_eq!(truths.len(), 393);
        assert_eq!(
            truths.iter().filter(|truth| truth.family() == WorldFamily::BooleanMechanism).count(),
            9
        );
        assert_eq!(
            truths.iter().filter(|truth| truth.family() == WorldFamily::NumericMechanism).count(),
            120
        );
        assert_eq!(
            truths.iter().filter(|truth| truth.family() == WorldFamily::TwoInputResponse).count(),
            264
        );
    }

    #[test]
    fn one_paired_numeric_case_preserves_truth_and_budget_under_a0_a2() {
        let truth = GeneratingModel::Numeric(numeric::ModelHypothesis {
            mechanism: numeric::Mechanism::DirectResponse,
            gain: 2,
            decay: None,
            initial_state: None,
            sensor: numeric::SensorMap::Identity,
        });
        let policy = policy_specs()[4];
        let a0 = run_arm(truth, CampaignArm::A0, policy, 6, "paired-smoke-a0").unwrap();
        let a2 = run_arm(truth, CampaignArm::A2, policy, 6, "paired-smoke-a2").unwrap();

        assert!(a0.generating_model_in_version_space);
        assert!(a2.generating_model_in_version_space);
        assert!(a0.set_score.exact_version_space_match);
        assert!(a2.set_score.exact_version_space_match);
        assert_eq!(a0.observed_evidence_count, a0.probe_count);
        assert_eq!(a2.observed_evidence_count, a2.probe_count);
        assert!(a0.probe_count <= 6);
        assert!(a2.probe_count <= 6);
    }

    #[test]
    fn rejects_zero_budget_before_campaign_work() {
        assert_eq!(
            run_exhaustive_finite_policy_campaign(Some(0)),
            Err(FiniteCampaignError::InvalidBudget)
        );
    }
}
