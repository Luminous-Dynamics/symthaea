//! Exhaustive paired policy campaign over the declared finite reference grammars.
//!
//! The runner evaluates transparent finite-inference baselines, not a learned
//! research agent. It enumerates every declared generating model and pairs A0
//! and A2 on identical truth, starting evidence and action budgets. Each arm
//! may adaptively choose its own post-start evidence trace.

use crate::interaction;
use crate::numeric;
use crate::policy::{PolicyError, StrategyEnumeratorBaseline};
use crate::protocol::{BlindEpisode, EvaluatorScore, StructureAccess, WorldFamily};
use crate::strategies::ProbeStrategy;
use crate::{Hypothesis, candidate_hypotheses};
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

    fn family_alias(self) -> &'static str {
        match self {
            Self::Boolean(_) => "boolean_mechanism",
            Self::Numeric(_) => "numeric_mechanism",
            Self::Interaction(_) => "two_input_response",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PolicySpec {
    name: &'static str,
    strategy: ProbeStrategy,
}

fn policies() -> [PolicySpec; 7] {
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
    pub generating_truth_behaviorally_represented: usize,
    pub decision_correct: usize,
    pub evidence_references_valid: usize,
    pub oracle_singleton_model_cases: usize,
    pub claimed_singleton_model_cases: usize,
    pub oracle_identified_structure_cases: usize,
    pub claimed_identified_structure_cases: usize,
    pub correct_identified_structure_cases: usize,
    pub missed_identifiable_structure_cases: usize,
    pub false_unique_structure_claims: usize,
    pub abstentions: usize,
    pub safe_abstentions: usize,
    pub unnecessary_abstentions: usize,
    pub total_probe_actions: usize,
    pub total_probe_cost_units: u64,
    pub total_legal_probe_actions_scored: u64,
    pub total_model_prediction_evaluations: u64,
    pub maximum_probe_actions_in_episode: usize,
}

fn bool_count(value: bool) -> usize {
    if value { 1 } else { 0 }
}

impl ArmAggregate {
    fn new() -> Self {
        Self {
            episodes: 0,
            exact_version_space_matches: 0,
            exact_structure_set_matches: 0,
            generating_model_retained: 0,
            generating_truth_behaviorally_represented: 0,
            decision_correct: 0,
            evidence_references_valid: 0,
            oracle_singleton_model_cases: 0,
            claimed_singleton_model_cases: 0,
            oracle_identified_structure_cases: 0,
            claimed_identified_structure_cases: 0,
            correct_identified_structure_cases: 0,
            missed_identifiable_structure_cases: 0,
            false_unique_structure_claims: 0,
            abstentions: 0,
            safe_abstentions: 0,
            unnecessary_abstentions: 0,
            total_probe_actions: 0,
            total_probe_cost_units: 0,
            total_legal_probe_actions_scored: 0,
            total_model_prediction_evaluations: 0,
            maximum_probe_actions_in_episode: 0,
        }
    }

    fn record(
        &mut self,
        score: &EvaluatorScore,
        work: crate::strategies::ProbeScoringWork,
    ) {
        self.episodes += 1;
        self.exact_version_space_matches += bool_count(score.set_score.exact_version_space_match);
        self.exact_structure_set_matches +=
            bool_count(score.structure_score.exact_structure_set_match);
        self.generating_model_retained += bool_count(score.generating_model_in_version_space);
        self.generating_truth_behaviorally_represented +=
            bool_count(score.generating_truth_behaviorally_represented);
        self.decision_correct += bool_count(score.decision_correct);
        self.evidence_references_valid += bool_count(score.evidence_references_valid);
        self.oracle_singleton_model_cases += bool_count(score.set_score.oracle_model_count == 1);
        self.claimed_singleton_model_cases += bool_count(score.set_score.claimed_model_count == 1);
        self.oracle_identified_structure_cases +=
            bool_count(score.structure_score.oracle_structure_identified);
        self.claimed_identified_structure_cases +=
            bool_count(score.structure_score.claimed_structure_identified);
        self.correct_identified_structure_cases += bool_count(
            score.structure_score.oracle_structure_identified
                && score.structure_score.claimed_structure_identified
                && score.structure_score.exact_structure_set_match,
        );
        self.missed_identifiable_structure_cases += bool_count(
            score.structure_score.oracle_structure_identified
                && !score.structure_score.claimed_structure_identified,
        );
        self.false_unique_structure_claims += bool_count(
            !score.structure_score.oracle_structure_identified
                && score.structure_score.claimed_structure_identified,
        );
        self.abstentions += bool_count(score.abstained);
        self.safe_abstentions += bool_count(score.safe_abstention);
        self.unnecessary_abstentions += bool_count(score.unnecessary_abstention);
        self.total_probe_actions += score.probe_count;
        self.total_probe_cost_units += score.probe_count as u64;
        self.total_legal_probe_actions_scored += work.legal_actions_scored;
        self.total_model_prediction_evaluations += work.model_prediction_evaluations;
        self.maximum_probe_actions_in_episode =
            self.maximum_probe_actions_in_episode.max(score.probe_count);
    }

    pub fn mean_probe_actions(&self) -> f64 {
        if self.episodes == 0 {
            0.0
        } else {
            self.total_probe_actions as f64 / self.episodes as f64
        }
    }

    pub fn exact_model_set_match_rate(&self) -> f64 {
        rate(self.exact_version_space_matches, self.episodes)
    }

    pub fn exact_structure_set_match_rate(&self) -> f64 {
        rate(self.exact_structure_set_matches, self.episodes)
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
pub struct PairedEpisodeRecord {
    /// Opaque deterministic pair identifier; does not encode generating truth.
    pub pair_id: String,
    pub world_family: WorldFamily,
    /// Stable ordinal within this declared finite family, not the true model parameters.
    pub generating_model_ordinal: usize,
    pub action_budget: usize,
    pub policy: String,
    pub comparison_design: String,
    pub a0: EvaluatorScore,
    pub a2: EvaluatorScore,
    pub a0_probe_work: crate::strategies::ProbeScoringWork,
    pub a2_probe_work: crate::strategies::ProbeScoringWork,
}

/// Paired binary outcome counts. "A0 only" and "A2 only" are discordant
/// pairs; these are descriptive finite-census counts, not inferential p-values.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairedBooleanContrast {
    pub a0_only: usize,
    pub a2_only: usize,
    pub both_true: usize,
    pub neither: usize,
}

impl PairedBooleanContrast {
    fn record(&mut self, a0: bool, a2: bool) {
        match (a0, a2) {
            (true, false) => self.a0_only += 1,
            (false, true) => self.a2_only += 1,
            (true, true) => self.both_true += 1,
            (false, false) => self.neither += 1,
        }
    }

    pub fn pair_count(&self) -> usize {
        self.a0_only + self.a2_only + self.both_true + self.neither
    }

    /// Positive values mean more A2-only positives than A0-only positives.
    pub fn net_a2_positive_pairs(&self) -> i64 {
        self.a2_only as i64 - self.a0_only as i64
    }
}

/// Family × policy paired contrast. Lower false-unique counts are preferable;
/// higher exact-set, truth-retention and evidence-validity counts are preferable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairedArmContrast {
    pub world_family: WorldFamily,
    pub policy: String,
    pub comparison_design: String,
    pub pair_count: usize,
    pub exact_model_set_match: PairedBooleanContrast,
    pub exact_structure_set_match: PairedBooleanContrast,
    pub generating_truth_retained: PairedBooleanContrast,
    pub generating_truth_behaviorally_represented: PairedBooleanContrast,
    pub decision_correct: PairedBooleanContrast,
    pub evidence_references_valid: PairedBooleanContrast,
    pub false_unique_structure_claim: PairedBooleanContrast,
    pub false_unique_model_claim: PairedBooleanContrast,
    pub correct_identified_structure: PairedBooleanContrast,
    pub missed_identifiable_structure: PairedBooleanContrast,
    pub safe_abstention: PairedBooleanContrast,
    pub unnecessary_abstention: PairedBooleanContrast,
    pub a0_probe_actions: usize,
    pub a2_probe_actions: usize,
    /// Signed difference in total probe actions (A2 minus A0).
    pub a2_minus_a0_probe_actions: i64,
    pub a0_legal_probe_actions_scored: u64,
    pub a2_legal_probe_actions_scored: u64,
    pub a2_minus_a0_legal_probe_actions_scored: i64,
    pub a0_model_prediction_evaluations: u64,
    pub a2_model_prediction_evaluations: u64,
    /// Signed difference in probe model-prediction evaluations (A2 minus A0).
    pub a2_minus_a0_model_prediction_evaluations: i64,
}

impl PairedArmContrast {
    fn new(world_family: WorldFamily, policy: &str) -> Self {
        Self {
            world_family,
            policy: policy.to_owned(),
            comparison_design: "adaptive_budget".to_owned(),
            pair_count: 0,
            exact_model_set_match: PairedBooleanContrast::default(),
            exact_structure_set_match: PairedBooleanContrast::default(),
            generating_truth_retained: PairedBooleanContrast::default(),
            generating_truth_behaviorally_represented: PairedBooleanContrast::default(),
            decision_correct: PairedBooleanContrast::default(),
            evidence_references_valid: PairedBooleanContrast::default(),
            false_unique_structure_claim: PairedBooleanContrast::default(),
            false_unique_model_claim: PairedBooleanContrast::default(),
            correct_identified_structure: PairedBooleanContrast::default(),
            missed_identifiable_structure: PairedBooleanContrast::default(),
            safe_abstention: PairedBooleanContrast::default(),
            unnecessary_abstention: PairedBooleanContrast::default(),
            a0_probe_actions: 0,
            a2_probe_actions: 0,
            a2_minus_a0_probe_actions: 0,
            a0_legal_probe_actions_scored: 0,
            a2_legal_probe_actions_scored: 0,
            a2_minus_a0_legal_probe_actions_scored: 0,
            a0_model_prediction_evaluations: 0,
            a2_model_prediction_evaluations: 0,
            a2_minus_a0_model_prediction_evaluations: 0,
        }
    }

    fn record(&mut self, pair: &PairedEpisodeRecord) {
        let a0 = &pair.a0;
        let a2 = &pair.a2;
        self.pair_count += 1;
        self.exact_model_set_match.record(
            a0.set_score.exact_version_space_match,
            a2.set_score.exact_version_space_match,
        );
        self.exact_structure_set_match.record(
            a0.structure_score.exact_structure_set_match,
            a2.structure_score.exact_structure_set_match,
        );
        self.generating_truth_retained.record(
            a0.generating_model_in_version_space,
            a2.generating_model_in_version_space,
        );
        self.generating_truth_behaviorally_represented.record(
            a0.generating_truth_behaviorally_represented,
            a2.generating_truth_behaviorally_represented,
        );
        self.decision_correct
            .record(a0.decision_correct, a2.decision_correct);
        self.evidence_references_valid
            .record(a0.evidence_references_valid, a2.evidence_references_valid);
        self.false_unique_structure_claim.record(
            !a0.structure_score.oracle_structure_identified
                && a0.structure_score.claimed_structure_identified,
            !a2.structure_score.oracle_structure_identified
                && a2.structure_score.claimed_structure_identified,
        );
        self.false_unique_model_claim.record(
            a0.set_score.oracle_model_count != 1 && a0.set_score.claimed_model_count == 1,
            a2.set_score.oracle_model_count != 1 && a2.set_score.claimed_model_count == 1,
        );
        self.correct_identified_structure.record(
            a0.structure_score.oracle_structure_identified
                && a0.structure_score.claimed_structure_identified
                && a0.structure_score.exact_structure_set_match,
            a2.structure_score.oracle_structure_identified
                && a2.structure_score.claimed_structure_identified
                && a2.structure_score.exact_structure_set_match,
        );
        self.missed_identifiable_structure.record(
            a0.structure_score.oracle_structure_identified
                && !a0.structure_score.claimed_structure_identified,
            a2.structure_score.oracle_structure_identified
                && !a2.structure_score.claimed_structure_identified,
        );
        self.safe_abstention
            .record(a0.safe_abstention, a2.safe_abstention);
        self.unnecessary_abstention
            .record(a0.unnecessary_abstention, a2.unnecessary_abstention);
        self.a0_probe_actions += a0.probe_count;
        self.a2_probe_actions += a2.probe_count;
        self.a2_minus_a0_probe_actions += a2.probe_count as i64 - a0.probe_count as i64;
        self.a0_legal_probe_actions_scored += pair.a0_probe_work.legal_actions_scored;
        self.a2_legal_probe_actions_scored += pair.a2_probe_work.legal_actions_scored;
        self.a2_minus_a0_legal_probe_actions_scored += pair.a2_probe_work.legal_actions_scored as i64
            - pair.a0_probe_work.legal_actions_scored as i64;
        self.a0_model_prediction_evaluations +=
            pair.a0_probe_work.model_prediction_evaluations;
        self.a2_model_prediction_evaluations +=
            pair.a2_probe_work.model_prediction_evaluations;
        self.a2_minus_a0_model_prediction_evaluations +=
            pair.a2_probe_work.model_prediction_evaluations as i64
                - pair.a0_probe_work.model_prediction_evaluations as i64;
    }
}

fn paired_arm_contrasts(pair_records: &[PairedEpisodeRecord]) -> Vec<PairedArmContrast> {
    let families = [
        WorldFamily::BooleanMechanism,
        WorldFamily::NumericMechanism,
        WorldFamily::TwoInputResponse,
    ];
    let mut contrasts = Vec::new();
    for family in families {
        for policy in policies() {
            let mut contrast = PairedArmContrast::new(family, policy.name);
            for pair in pair_records
                .iter()
                .filter(|pair| pair.world_family == family && pair.policy == policy.name)
            {
                contrast.record(pair);
            }
            if contrast.pair_count > 0 {
                contrasts.push(contrast);
            }
        }
    }
    contrasts
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
    pub episode_count: usize,
    pub results: Vec<PairedPolicyResult>,
    /// Raw post-submission evaluator scores for paired world-model comparisons.
    pub pair_records: Vec<PairedEpisodeRecord>,
    /// Descriptive paired A0/A2 contrasts, one row per family × policy.
    pub paired_contrasts: Vec<PairedArmContrast>,
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

fn strategy_for_pair(policy: PolicySpec, family: WorldFamily, pair_index: usize) -> PolicySpec {
    let family_salt = match family {
        WorldFamily::BooleanMechanism => 0xB001_EA01_u64,
        WorldFamily::NumericMechanism => 0xA11C_0C02_u64,
        WorldFamily::TwoInputResponse => 0x1A2C_7E03_u64,
    };
    let strategy = match policy.strategy {
        ProbeStrategy::Random { seed } => ProbeStrategy::Random {
            seed: seed ^ family_salt ^ (pair_index as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93),
        },
        other => other,
    };
    PolicySpec {
        name: policy.name,
        strategy,
    }
}

fn run_arm(
    truth: GeneratingModel,
    arm: CampaignArm,
    policy: PolicySpec,
    budget: usize,
    episode_id: &str,
) -> Result<(EvaluatorScore, crate::strategies::ProbeScoringWork), FiniteCampaignError> {
    if budget == 0 {
        return Err(FiniteCampaignError::InvalidBudget);
    }

    let access = match (truth, arm) {
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
        (GeneratingModel::Interaction(_), CampaignArm::A2) => StructureAccess::InteractionWithheld,
    };

    let mut episode = match truth {
        GeneratingModel::Boolean(model) => BlindEpisode::new_boolean(
            episode_id,
            model.mechanism,
            model.measurement_map,
            model.initial_hidden_state,
            access,
            budget,
        ),
        GeneratingModel::Numeric(model) => {
            BlindEpisode::new_numeric(episode_id, model, access, budget)
        }
        GeneratingModel::Interaction(model) => {
            BlindEpisode::new_interaction(episode_id, model, access, budget)
        }
    }
    .map_err(|error| FiniteCampaignError::Protocol(error.to_string()))?;

    // Policy receives only AgentView. Generating truth remains in the episode.
    let mut probe_work = crate::strategies::ProbeScoringWork::default();
    for _ in 0..=budget {
        if episode.status() == crate::protocol::EpisodeStatus::Submitted {
            break;
        }
        let view = episode
            .view()
            .map_err(|error| FiniteCampaignError::Protocol(error.to_string()))?;
        let (action, step_work) =
            StrategyEnumeratorBaseline::next_action_with_work(&view, policy.strategy)
                .map_err(|error: PolicyError| FiniteCampaignError::Policy(error.to_string()))?;
        probe_work.legal_actions_scored += step_work.legal_actions_scored;
        probe_work.model_prediction_evaluations += step_work.model_prediction_evaluations;
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
    let score = episode
        .evaluate_after_submission()
        .map_err(|error| FiniteCampaignError::Protocol(error.to_string()))?;
    Ok((score, probe_work))
}

/// Exhaustive generating-model census. A0/A2 share each exact hidden model,
/// empty initial evidence and per-family action cap. Their subsequent adaptive
/// trajectories may diverge, which is why this is an AdaptiveBudget comparison.
pub fn run_exhaustive_finite_policy_campaign(
    budget_override: Option<usize>,
) -> Result<FinitePolicyCampaignReport, FiniteCampaignError> {
    if budget_override == Some(0) {
        return Err(FiniteCampaignError::InvalidBudget);
    }

    let truths = candidate_truths();
    let policy_specs = policies();
    let mut results = Vec::new();
    let mut pair_records = Vec::with_capacity(truths.len() * policy_specs.len());

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

        for policy in policy_specs {
            let mut a0 = ArmAggregate::new();
            let mut a2 = ArmAggregate::new();

            for (index, truth) in family_truths.iter().copied().enumerate() {
                let pair = format!("{}-{}-{:04}", truth.family_alias(), policy.name, index);
                let paired_policy = strategy_for_pair(policy, family, index);
                let (score_a0, work_a0) = run_arm(
                    truth,
                    CampaignArm::A0,
                    paired_policy,
                    budget,
                    &format!("{pair}-a0"),
                )?;
                let (score_a2, work_a2) = run_arm(
                    truth,
                    CampaignArm::A2,
                    paired_policy,
                    budget,
                    &format!("{pair}-a2"),
                )?;
                a0.record(&score_a0, work_a0);
                a2.record(&score_a2, work_a2);
                pair_records.push(PairedEpisodeRecord {
                    pair_id: pair.clone(),
                    world_family: family,
                    generating_model_ordinal: index,
                    action_budget: budget,
                    policy: policy.name.to_owned(),
                    comparison_design: "adaptive_budget".to_owned(),
                    a0: score_a0,
                    a2: score_a2,
                    a0_probe_work: work_a0,
                    a2_probe_work: work_a2,
                });
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

    let paired_contrasts = paired_arm_contrasts(&pair_records);

    Ok(FinitePolicyCampaignReport {
        report_version: "symthaea-open-world-finite-policy-campaign-v2".to_owned(),
        campaign_kind: "exhaustive_generating_model_census".to_owned(),
        policy_kind: "StrategyEnumeratorBaseline (exact finite inference; not learned)".to_owned(),
        initial_observation_count: 0,
        candidate_truth_count: truths.len(),
        policy_count: policy_specs.len(),
        pair_count: truths.len() * policy_specs.len(),
        episode_count: truths.len() * policy_specs.len() * 2,
        results,
        pair_records,
        paired_contrasts,
        limitations: vec![
            "This is a deterministic census over the three declared finite reference grammars, not a sample of natural-science systems.".to_owned(),
            "The evaluated policy is the transparent exact finite inference baseline, not the learned Symthaea research orchestrator.".to_owned(),
            "A0 and A2 share world truth, empty starting evidence and maximum action budgets; post-start adaptive traces may differ.".to_owned(),
            "Random arm uses base seed 23, deterministically salted by world family and pair index; each next action uses a step-specific seed. The same derived seed schedule is paired across A0 and A2.".to_owned(),
            "Probe cost units use the reference worlds' declared unit cost. Model-prediction evaluation counts are an algorithmic search-work proxy for the one-step probe scorer; they exclude discovery filtering, serialization, hardware timing, CPU/GPU, tokens, energy and human-review costs.".to_owned(),
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
            truths
                .iter()
                .filter(|truth| truth.family() == WorldFamily::BooleanMechanism)
                .count(),
            9
        );
        assert_eq!(
            truths
                .iter()
                .filter(|truth| truth.family() == WorldFamily::NumericMechanism)
                .count(),
            120
        );
        assert_eq!(
            truths
                .iter()
                .filter(|truth| truth.family() == WorldFamily::TwoInputResponse)
                .count(),
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
        let policy = policies()[4];
        let (a0, work_a0) =
            run_arm(truth, CampaignArm::A0, policy, 6, "paired-smoke-a0").unwrap();
        let (a2, work_a2) =
            run_arm(truth, CampaignArm::A2, policy, 6, "paired-smoke-a2").unwrap();

        assert!(a0.generating_model_in_version_space);
        assert!(a2.generating_model_in_version_space);
        assert!(a0.set_score.exact_version_space_match);
        assert!(a2.set_score.exact_version_space_match);
        assert!(a0.probe_count <= 6);
        assert!(a2.probe_count <= 6);
        assert!(work_a0.legal_actions_scored > 0);
        assert!(work_a2.legal_actions_scored > 0);
        assert!(work_a0.model_prediction_evaluations > 0);
        assert!(work_a2.model_prediction_evaluations > 0);
    }

    #[test]
    fn random_policy_pair_seeds_are_stable_and_vary_across_worlds() {
        let policy = policies()[6];
        let first = strategy_for_pair(policy, WorldFamily::NumericMechanism, 0);
        assert_eq!(
            first,
            strategy_for_pair(policy, WorldFamily::NumericMechanism, 0)
        );
        assert_ne!(
            first.strategy,
            strategy_for_pair(policy, WorldFamily::NumericMechanism, 1).strategy
        );
        assert_ne!(
            first.strategy,
            strategy_for_pair(policy, WorldFamily::TwoInputResponse, 0).strategy
        );
    }

    #[test]
    fn arm_aggregate_separates_identified_missed_and_false_unique_structures() {
        use crate::protocol::{SetScore, StructureScore};

        let mut aggregate = ArmAggregate::new();
        let false_unique = EvaluatorScore {
            evidence_prefix_sha256: "sha256:".to_owned(),
            generating_model_in_version_space: true,
            generating_truth_behaviorally_represented: true,
            set_score: SetScore {
                claimed_model_count: 2,
                oracle_model_count: 2,
                correct_model_count: 2,
                exact_version_space_match: true,
            },
            structure_score: StructureScore {
                claimed_structure_count: 1,
                oracle_structure_count: 2,
                correct_structure_count: 1,
                exact_structure_set_match: false,
                claimed_structure_identified: true,
                oracle_structure_identified: false,
            },
            safe_abstention: false,
            unnecessary_abstention: false,
            decision_correct: false,
            abstained: false,
            evidence_references_valid: true,
            cited_evidence_count: 1,
            observed_evidence_count: 1,
            probe_count: 1,
        };
        aggregate.record(&false_unique, crate::strategies::ProbeScoringWork::default());

        let missed = EvaluatorScore {
            structure_score: StructureScore {
                claimed_structure_count: 2,
                oracle_structure_count: 1,
                correct_structure_count: 1,
                exact_structure_set_match: false,
                claimed_structure_identified: false,
                oracle_structure_identified: true,
            },
            ..false_unique.clone()
        };
        aggregate.record(&missed, crate::strategies::ProbeScoringWork::default());

        let correct = EvaluatorScore {
            structure_score: StructureScore {
                claimed_structure_count: 1,
                oracle_structure_count: 1,
                correct_structure_count: 1,
                exact_structure_set_match: true,
                claimed_structure_identified: true,
                oracle_structure_identified: true,
            },
            ..false_unique
        };
        aggregate.record(&correct, crate::strategies::ProbeScoringWork::default());

        // The claimed parameterized-model set has two members in all
        // three fixtures. Structural singleton claims are scored separately.
        assert_eq!(aggregate.claimed_singleton_model_cases, 0);
        assert_eq!(aggregate.oracle_singleton_model_cases, 0);
        assert_eq!(aggregate.oracle_identified_structure_cases, 2);
        assert_eq!(aggregate.claimed_identified_structure_cases, 2);
        assert_eq!(aggregate.correct_identified_structure_cases, 1);
        assert_eq!(aggregate.missed_identifiable_structure_cases, 1);
        assert_eq!(aggregate.false_unique_structure_claims, 1);
    }

    #[test]
    fn paired_boolean_contrast_distinguishes_wins_ties_and_losses() {
        let mut contrast = PairedBooleanContrast::default();
        contrast.record(true, false);
        contrast.record(false, true);
        contrast.record(true, true);
        contrast.record(false, false);

        assert_eq!(contrast.a0_only, 1);
        assert_eq!(contrast.a2_only, 1);
        assert_eq!(contrast.both_true, 1);
        assert_eq!(contrast.neither, 1);
        assert_eq!(contrast.pair_count(), 4);
        assert_eq!(contrast.net_a2_positive_pairs(), 0);
    }

    #[test]
    fn rejects_zero_budget_before_campaign_work() {
        assert_eq!(
            run_exhaustive_finite_policy_campaign(Some(0)),
            Err(FiniteCampaignError::InvalidBudget)
        );
    }
}
