//! Matched-budget synthetic benchmark for verifier-grounded direction allocation.
//!
//! This is an executable policy-discrimination instrument, not a neural architecture
//! benchmark. Every policy sees the same deterministic payoff schedule for a given seed.
//! The synthetic verifier owns the schedule and records the same receipt contract for every
//! policy. The oracle policy is an intentionally non-deployable upper bound.
//!
//! The budget counts unit-cost actions: one direction evaluation or one NEW proposal.
//! v0.1 does not claim to solve variable-cost scheduling. Costs are recorded explicitly so a
//! later version can replace action-count budgeting with a strict heterogeneous-cost budget.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::event_tape::{
    ActionEvent, ActionIntent, ActionOutcome, BudgetModel, CandidateDisposition, ComparisonMode,
    EventTape, EventTapeHeader, ExecutionContract, ExecutionMode, MeterValue,
    PolicySnapshot, ProposalDisposition, ResourceReceipt,
};

use super::direction_board::{
    AllocationArm, AttemptDisposition, DirectionAttempt, DirectionBoard, DirectionBoardConfig,
    DirectionDescriptor, NewDirectionDisposition, NewDirectionProposal,
};

const PROFILE_SHA256: &str =
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const METRIC_ID: &str = "held_out_capability";
const TARGET_SCORE: f64 = 0.90;
const BOARD_EXPLORATION_COEFFICIENT: f64 = 0.55;
const NEW_DIRECTION_CREDIT: f64 = 0.20;
const GLOBAL_UCB_EXPLORATION: f64 = 0.55;
// Frozen recency parameter for the nonstationary discounted-UCB control.
const DISCOUNT_FACTOR: f64 = 0.85;

/// Independent payoff families. The no-new-direction family is a null control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectionLandscapeFamily {
    /// Strong early leader saturates while another direction has a delayed breakthrough.
    DelayedBreakthrough,
    /// Existing directions contain the best payoff; NEW proposals should be rejected.
    NoNewDirection,
    /// The best mechanism lies outside the initial grammar and requires NEW admission.
    NewOperatorBreakthrough,
    /// The early score ranking reverses after a delayed intervention.
    RankReversal,
    /// A high-scoring proxy collapses while a slower direction yields better final transfer.
    DeceptiveProxy,
    /// One direction has high payoff variance while another is stable but less flashy.
    HeteroscedasticRewards,
    /// Two initial directions share a strongly correlated payoff profile.
    CorrelatedDirections,
    /// Explicit no-breakthrough/no-new-opportunity null control.
    NullNoBreakthrough,
}

/// Search policies compared under identical action budgets and payoff schedules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DirectionCampaignPolicy {
    /// Always evaluate the direction with the highest observed score; never opens NEW.
    ScoreOnlyWinnerTakesAll,
    /// Rotate through every registered direction and the NEW arm.
    FixedRoundRobin,
    /// Uniform random choice over registered directions and NEW.
    Random,
    /// Standard UCB over global mean score, with an explicit NEW-arm prior.
    GlobalUcb,
    /// The implementation under test: verifier-grounded, local-record UCB board.
    VerifierGroundedDirectionBoard,
    /// Discounted-UCB control for deliberately nonstationary payoff landscapes.
    DiscountedUcb,
    /// Beta-Bernoulli Thompson sampling for target-hit probability.
    ThompsonSampling,
    /// Sees the complete payoff schedule. This is an upper bound, never deployable.
    OracleUpperBound,
}

/// One action in a campaign. A NEW proposal consumes one budget slot but has no task score.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectionCampaignAction {
    pub slot: usize,
    pub arm_id: String,
    pub is_new_proposal: bool,
    pub observed_score: Option<f64>,
    pub proposal_disposition: Option<NewDirectionDisposition>,
    pub cost_units: u64,
}

/// Frozen parameter snapshot for policy interpretation and replay.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectionCampaignPolicyConfig {
    pub target_score: f64,
    pub board_exploration_coefficient: f64,
    pub global_ucb_exploration_coefficient: f64,
    pub new_direction_prior_credit: f64,
    pub common_bootstrap_evaluations: u64,
    pub board_minimum_initial_evaluations: u64,
    pub discounted_ucb_factor: f64,
    pub thompson_prior_alpha: f64,
    pub thompson_prior_beta: f64,
    pub action_cost_units: u64,
    pub oracle_uses_hidden_payoff_schedule: bool,
}

/// Full audit summary for one policy, landscape and seed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectionCampaignRun {
    pub schema: String,
    pub version: String,
    pub landscape: DirectionLandscapeFamily,
    pub policy: DirectionCampaignPolicy,
    pub policy_config: DirectionCampaignPolicyConfig,
    pub policy_config_sha256: String,
    pub seed: u64,
    pub action_budget: usize,
    pub action_count: usize,
    pub total_cost_units: u64,
    pub target_score: f64,
    pub actions_to_target: Option<usize>,
    pub final_best_score: f64,
    pub final_best_direction: String,
    pub oracle_best_score: f64,
    pub simple_regret: f64,
    pub admitted_new_directions: usize,
    pub duplicate_new_proposals: usize,
    pub rejected_new_proposals: usize,
    pub inconclusive_new_proposals: usize,
    pub final_board_sha256: String,
    pub actions: Vec<DirectionCampaignAction>,
}

/// Aggregate across independent seeds for one fixed policy, landscape and budget.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectionCampaignAggregate {
    pub landscape: DirectionLandscapeFamily,
    pub policy: DirectionCampaignPolicy,
    pub action_budget: usize,
    pub seed_count: usize,
    pub mean_final_best_score: f64,
    pub final_best_score_sample_sd: f64,
    pub mean_simple_regret: f64,
    pub target_hit_rate: f64,
    /// Mean actions-to-target among target-hitting runs only.
    pub mean_actions_to_target: Option<f64>,
    pub censored_target_runs: usize,
    pub mean_new_direction_admissions: f64,
    pub mean_total_cost_units: f64,
}

/// Paired comparison of one candidate policy against one baseline on identical seeds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectionCampaignPairedComparison {
    pub landscape: DirectionLandscapeFamily,
    pub candidate_policy: DirectionCampaignPolicy,
    pub baseline_policy: DirectionCampaignPolicy,
    pub action_budget: usize,
    pub pair_count: usize,
    /// Positive values favor the candidate policy.
    pub mean_final_score_delta: f64,
    pub final_score_delta_sample_sd: f64,
    /// Positive values mean the candidate reduced simple regret.
    pub mean_simple_regret_reduction: f64,
    pub candidate_win_rate: f64,
    pub tie_rate: f64,
}

/// Run one policy on one evaluator-owned landscape with a fixed action budget.
///
/// The first evaluation of A, B and C is common bootstrap cost for every policy. All
/// subsequent choices are policy-controlled. Each action costs one synthetic unit.
pub fn run_direction_campaign(
    landscape: DirectionLandscapeFamily,
    policy: DirectionCampaignPolicy,
    action_budget: usize,
    seed: u64,
) -> Result<DirectionCampaignRun, String> {
    if action_budget < 4 {
        return Err("action_budget must be at least 4 (three common bootstrap evaluations plus one policy action)".to_owned());
    }

    let config = DirectionBoardConfig {
        comparison_profile_sha256: PROFILE_SHA256.to_owned(),
        metric_id: METRIC_ID.to_owned(),
        exploration_coefficient: BOARD_EXPLORATION_COEFFICIENT,
        new_direction_prior_yield: NEW_DIRECTION_CREDIT,
        new_direction_admission_credit: NEW_DIRECTION_CREDIT,
        minimum_initial_evaluations: 1,
    };
    let mut board = DirectionBoard::new(config).map_err(|error| error.to_string())?;
    let mut direction_digests = BTreeMap::new();
    let mut observations: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    let mut actions = Vec::with_capacity(action_budget);
    let mut proposal_count = 0usize;
    let mut new_proposal_successes: Vec<bool> = Vec::new();
    let mut admitted_new_directions = 0usize;
    let mut duplicate_new_proposals = 0usize;
    let mut rejected_new_proposals = 0usize;
    let mut inconclusive_new_proposals = 0usize;
    let mut round_robin_cursor = 0usize;
    let mut random_state = seed ^ 0x6a09_e667_f3bc_c909;
    let mut actions_to_target = None;

    for direction_id in ["A", "B", "C"] {
        let descriptor = initial_descriptor(direction_id);
        let verifier_receipt = digest(&format!("initial-verifier-receipt:{direction_id}"));
        let direction_sha = board
            .register_verified_direction(descriptor, &verifier_receipt)
            .map_err(|error| error.to_string())?;
        direction_digests.insert(direction_id.to_owned(), direction_sha);
        observations.insert(direction_id.to_owned(), Vec::new());
        evaluate_direction(
            landscape,
            seed,
            direction_id,
            &mut board,
            &direction_digests,
            &mut observations,
            &mut actions,
            &mut actions_to_target,
            TARGET_SCORE,
        )?;
    }

    while actions.len() < action_budget {
        let available_directions: Vec<String> = observations.keys().cloned().collect();
        let selected_arm = match policy {
            DirectionCampaignPolicy::ScoreOnlyWinnerTakesAll => {
                select_score_only(&observations)?
            }
            DirectionCampaignPolicy::FixedRoundRobin => {
                select_round_robin(&available_directions, &mut round_robin_cursor)
            }
            DirectionCampaignPolicy::Random => {
                select_random(&available_directions, &mut random_state)
            }
            DirectionCampaignPolicy::GlobalUcb => select_global_ucb(
                &observations,
                proposal_count,
                admitted_new_directions,
                actions.len(),
            ),
            DirectionCampaignPolicy::DiscountedUcb => select_discounted_ucb(
                &observations,
                &new_proposal_successes,
                actions.len(),
            ),
            DirectionCampaignPolicy::ThompsonSampling => select_thompson_sampling(
                &observations,
                &new_proposal_successes,
                &mut random_state,
            ),
            DirectionCampaignPolicy::VerifierGroundedDirectionBoard => {
                select_direction_board(&board, &direction_digests)?
            }
            DirectionCampaignPolicy::OracleUpperBound => select_oracle(
                landscape,
                &observations,
                admitted_new_directions > 0,
            ),
        };

        if selected_arm == "NEW" {
            let descriptor = new_direction_descriptor(proposal_count > 0);
            let disposition = if landscape_has_new_operator(landscape) {
                if admitted_new_directions == 0 {
                    NewDirectionDisposition::Admitted
                } else {
                    NewDirectionDisposition::Duplicate
                }
            } else {
                NewDirectionDisposition::Rejected
            };
            let proposal_id = format!("proposal-{seed}-{:04}", proposal_count + 1);
            let proposal = NewDirectionProposal {
                proposal_id: proposal_id.clone(),
                descriptor,
                proposal_sha256: digest(&format!("proposal-payload:{seed}:{proposal_id}")),
                comparison_profile_sha256: PROFILE_SHA256.to_owned(),
                metric_id: METRIC_ID.to_owned(),
                evaluation_cost_units: 1,
                disposition,
                verifier_receipt_sha256: digest(&format!("proposal-receipt:{seed}:{proposal_id}")),
            };
            let admitted_id = board
                .record_new_direction_proposal(proposal)
                .map_err(|error| error.to_string())?;
            new_proposal_successes.push(disposition == NewDirectionDisposition::Admitted);
            match disposition {
                NewDirectionDisposition::Admitted => {
                    let direction_sha = admitted_id.ok_or_else(|| {
                        "synthetic verifier admitted NEW but the board did not register it".to_owned()
                    })?;
                    direction_digests.insert("D".to_owned(), direction_sha);
                    observations.insert("D".to_owned(), Vec::new());
                    admitted_new_directions += 1;
                }
                NewDirectionDisposition::Duplicate => duplicate_new_proposals += 1,
                NewDirectionDisposition::Rejected => rejected_new_proposals += 1,
                NewDirectionDisposition::Inconclusive => inconclusive_new_proposals += 1,
            }
            proposal_count += 1;
            actions.push(DirectionCampaignAction {
                slot: actions.len() + 1,
                arm_id: "NEW".to_owned(),
                is_new_proposal: true,
                observed_score: None,
                proposal_disposition: Some(disposition),
                cost_units: 1,
            });
        } else {
            evaluate_direction(
                landscape,
                seed,
                &selected_arm,
                &mut board,
                &direction_digests,
                &mut observations,
                &mut actions,
                &mut actions_to_target,
                TARGET_SCORE,
            )?;
        }
    }

    board.validate_integrity().map_err(|error| error.to_string())?;
    let final_board_sha256 = board.digest().map_err(|error| error.to_string())?;
    let (final_best_direction, final_best_score) =
        best_observed_direction(&observations)?;
    let oracle_best_score = oracle_best_score(landscape);
    let policy_config = DirectionCampaignPolicyConfig {
        target_score: TARGET_SCORE,
        board_exploration_coefficient: BOARD_EXPLORATION_COEFFICIENT,
        global_ucb_exploration_coefficient: GLOBAL_UCB_EXPLORATION,
        new_direction_prior_credit: NEW_DIRECTION_CREDIT,
        common_bootstrap_evaluations: 3,
        board_minimum_initial_evaluations: 1,
        discounted_ucb_factor: DISCOUNT_FACTOR,
        thompson_prior_alpha: 1.0,
        thompson_prior_beta: 1.0,
        action_cost_units: 1,
        oracle_uses_hidden_payoff_schedule: policy == DirectionCampaignPolicy::OracleUpperBound,
    };
    let policy_config_value = serde_json::to_value(&policy_config)
        .map_err(|error| format!("policy config serialization failed: {error}"))?;
    let policy_config_bytes = serde_json::to_vec(&policy_config_value)
        .map_err(|error| format!("policy config encoding failed: {error}"))?;
    let policy_config_sha256 = digest_bytes(&policy_config_bytes);
    Ok(DirectionCampaignRun {
        schema: "symthaea.direction-campaign.run".to_owned(),
        version: "0.1.0".to_owned(),
        landscape,
        policy,
        policy_config,
        policy_config_sha256,
        seed,
        action_budget,
        action_count: actions.len(),
        total_cost_units: actions.iter().map(|action| action.cost_units).sum(),
        target_score: TARGET_SCORE,
        actions_to_target,
        final_best_score,
        final_best_direction,
        oracle_best_score,
        simple_regret: (oracle_best_score - final_best_score).max(0.0),
        admitted_new_directions,
        duplicate_new_proposals,
        rejected_new_proposals,
        inconclusive_new_proposals,
        final_board_sha256,
        actions,
    })
}

/// Run the full policy × landscape matrix over a caller-supplied seed set.
pub fn run_direction_campaign_matrix(
    landscapes: &[DirectionLandscapeFamily],
    policies: &[DirectionCampaignPolicy],
    action_budget: usize,
    seeds: &[u64],
) -> Result<Vec<DirectionCampaignRun>, String> {
    if landscapes.is_empty() || policies.is_empty() || seeds.is_empty() {
        return Err("landscapes, policies and seeds must all be non-empty".to_owned());
    }
    let mut runs = Vec::with_capacity(landscapes.len() * policies.len() * seeds.len());
    for landscape in landscapes {
        for policy in policies {
            for seed in seeds {
                runs.push(run_direction_campaign(
                    *landscape,
                    *policy,
                    action_budget,
                    *seed,
                )?);
            }
        }
    }
    Ok(runs)
}

/// Aggregate independent seeds. Compare policies by matching landscape, budget and seed;
/// bootstrap uncertainty at the generated landscape/payoff-template level.
pub fn aggregate_direction_campaign_runs(
    runs: &[DirectionCampaignRun],
) -> Result<Vec<DirectionCampaignAggregate>, String> {
    let mut groups: BTreeMap<
        (DirectionLandscapeFamily, DirectionCampaignPolicy, usize),
        Vec<&DirectionCampaignRun>,
    > = BTreeMap::new();
    for run in runs {
        if run.action_count != run.action_budget
            || run.total_cost_units != run.action_budget as u64
        {
            return Err("run action/cost accounting does not reconcile with its budget".to_owned());
        }
        groups
            .entry((run.landscape, run.policy, run.action_budget))
            .or_default()
            .push(run);
    }

    let mut aggregates = Vec::with_capacity(groups.len());
    for ((landscape, policy, action_budget), group) in groups {
        let mut seeds = BTreeSet::new();
        if group.iter().any(|run| !seeds.insert(run.seed)) {
            return Err("aggregate group contains duplicate seed values".to_owned());
        }
        let count = group.len() as f64;
        let mean = group.iter().map(|run| run.final_best_score).sum::<f64>() / count;
        let sample_variance = if group.len() < 2 {
            0.0
        } else {
            group
                .iter()
                .map(|run| (run.final_best_score - mean).powi(2))
                .sum::<f64>()
                / (group.len() - 1) as f64
        };
        let hit_runs: Vec<f64> = group
            .iter()
            .filter_map(|run| run.actions_to_target.map(|value| value as f64))
            .collect();
        let mean_actions_to_target = if hit_runs.is_empty() {
            None
        } else {
            Some(hit_runs.iter().sum::<f64>() / hit_runs.len() as f64)
        };
        aggregates.push(DirectionCampaignAggregate {
            landscape,
            policy,
            action_budget,
            seed_count: group.len(),
            mean_final_best_score: mean,
            final_best_score_sample_sd: sample_variance.sqrt(),
            mean_simple_regret: group.iter().map(|run| run.simple_regret).sum::<f64>() / count,
            target_hit_rate: hit_runs.len() as f64 / count,
            mean_actions_to_target,
            censored_target_runs: group.len() - hit_runs.len(),
            mean_new_direction_admissions: group
                .iter()
                .map(|run| run.admitted_new_directions as f64)
                .sum::<f64>()
                / count,
            mean_total_cost_units: group
                .iter()
                .map(|run| run.total_cost_units as f64)
                .sum::<f64>()
                / count,
        });
    }
    Ok(aggregates)
}

/// Compare the selected policy with every other policy using strictly paired seeds.
/// A missing policy/seed cell is an error; silently comparing different seeds is forbidden.
pub fn compare_paired_direction_campaign_runs(
    runs: &[DirectionCampaignRun],
    candidate_policy: DirectionCampaignPolicy,
) -> Result<Vec<DirectionCampaignPairedComparison>, String> {
    let mut cells: BTreeMap<
        (DirectionLandscapeFamily, usize, u64),
        BTreeMap<DirectionCampaignPolicy, &DirectionCampaignRun>,
    > = BTreeMap::new();
    for run in runs {
        let cell = cells
            .entry((run.landscape, run.action_budget, run.seed))
            .or_default();
        if cell.insert(run.policy, run).is_some() {
            return Err("duplicate policy/landscape/budget/seed cell".to_owned());
        }
    }

    let mut seed_sets: BTreeMap<
        (DirectionLandscapeFamily, usize, DirectionCampaignPolicy),
        BTreeSet<u64>,
    > = BTreeMap::new();
    for ((landscape, budget, seed), policies) in &cells {
        if !policies.contains_key(&candidate_policy) {
            return Err(format!(
                "missing candidate policy {candidate_policy:?} for {landscape:?}, budget {budget}, seed {seed}"
            ));
        }
        for policy in policies.keys() {
            seed_sets
                .entry((*landscape, *budget, *policy))
                .or_default()
                .insert(*seed);
        }
    }

    let mut reference_sets: BTreeMap<(DirectionLandscapeFamily, usize), BTreeSet<u64>> =
        BTreeMap::new();
    for ((landscape, budget, _policy), seeds) in &seed_sets {
        match reference_sets.get(&(*landscape, *budget)) {
            Some(reference) if reference != seeds => {
                return Err(format!(
                    "policy seed sets differ for {landscape:?} at action budget {budget}"
                ));
            }
            Some(_) => {}
            None => {
                reference_sets.insert((*landscape, *budget), seeds.clone());
            }
        }
    }

    let mut paired: BTreeMap<
        (DirectionLandscapeFamily, DirectionCampaignPolicy, usize),
        Vec<(f64, f64)>,
    > = BTreeMap::new();
    for ((landscape, budget, _seed), policies) in &cells {
        let candidate = policies
            .get(&candidate_policy)
            .expect("candidate presence was checked above");
        for (baseline_policy, baseline) in policies {
            if *baseline_policy == candidate_policy {
                continue;
            }
            paired
                .entry((*landscape, *baseline_policy, *budget))
                .or_default()
                .push((
                    candidate.final_best_score - baseline.final_best_score,
                    baseline.simple_regret - candidate.simple_regret,
                ));
        }
    }

    let mut comparisons = Vec::with_capacity(paired.len());
    for ((landscape, baseline_policy, action_budget), deltas) in paired {
        if deltas.is_empty() {
            continue;
        }
        let count = deltas.len() as f64;
        let mean_delta = deltas.iter().map(|(score, _)| *score).sum::<f64>() / count;
        let variance = if deltas.len() < 2 {
            0.0
        } else {
            deltas
                .iter()
                .map(|(score, _)| (score - mean_delta).powi(2))
                .sum::<f64>()
                / (deltas.len() - 1) as f64
        };
        let wins = deltas.iter().filter(|(score, _)| *score > 1e-12).count();
        let ties = deltas.iter().filter(|(score, _)| score.abs() <= 1e-12).count();
        comparisons.push(DirectionCampaignPairedComparison {
            landscape,
            candidate_policy,
            baseline_policy,
            action_budget,
            pair_count: deltas.len(),
            mean_final_score_delta: mean_delta,
            final_score_delta_sample_sd: variance.sqrt(),
            mean_simple_regret_reduction: deltas
                .iter()
                .map(|(_, regret_reduction)| *regret_reduction)
                .sum::<f64>()
                / count,
            candidate_win_rate: wins as f64 / count,
            tie_rate: ties as f64 / count,
        });
    }
    Ok(comparisons)
}

fn evaluate_direction(
    landscape: DirectionLandscapeFamily,
    seed: u64,
    direction_id: &str,
    board: &mut DirectionBoard,
    direction_digests: &BTreeMap<String, String>,
    observations: &mut BTreeMap<String, Vec<f64>>,
    actions: &mut Vec<DirectionCampaignAction>,
    actions_to_target: &mut Option<usize>,
    target_score: f64,
) -> Result<(), String> {
    let values = observations
        .get_mut(direction_id)
        .ok_or_else(|| format!("selected unknown direction {direction_id:?}"))?;
    let evaluation_index = values.len();
    let score = payoff(landscape, direction_id, evaluation_index);
    let direction_sha = direction_digests
        .get(direction_id)
        .ok_or_else(|| format!("missing canonical digest for direction {direction_id:?}"))?
        .clone();
    let slot = actions.len() + 1;
    let attempt_id = format!("attempt-{seed}-{slot:04}-{direction_id}");
    let attempt = DirectionAttempt {
        attempt_id: attempt_id.clone(),
        direction_sha256: direction_sha,
        method_id: format!("method-{direction_id}-{}", evaluation_index + 1),
        parent_candidate_sha256: digest(&format!("parent:{attempt_id}")),
        candidate_sha256: digest(&format!("candidate:{attempt_id}")),
        semantic_delta_sha256: digest(&format!("semantic-delta:{attempt_id}")),
        comparison_profile_sha256: PROFILE_SHA256.to_owned(),
        metric_id: METRIC_ID.to_owned(),
        evaluation_cost_units: 1,
        disposition: AttemptDisposition::Verified,
        score: Some(score),
        verifier_receipt_sha256: digest(&format!("verified-receipt:{attempt_id}:{score:.8}")),
    };
    board
        .record_attempt(attempt)
        .map_err(|error| error.to_string())?;
    values.push(score);
    if actions_to_target.is_none() && score >= target_score {
        *actions_to_target = Some(slot);
    }
    actions.push(DirectionCampaignAction {
        slot,
        arm_id: direction_id.to_owned(),
        is_new_proposal: false,
        observed_score: Some(score),
        proposal_disposition: None,
        cost_units: 1,
    });
    Ok(())
}

fn select_direction_board(
    board: &DirectionBoard,
    direction_digests: &BTreeMap<String, String>,
) -> Result<String, String> {
    let plan = board.recommend(1).map_err(|error| error.to_string())?;
    let selected: Vec<_> = plan
        .allocations
        .iter()
        .filter(|allocation| allocation.recommended_evaluations > 0)
        .collect();
    if selected.len() != 1 || selected[0].recommended_evaluations != 1 {
        return Err("one-slot board recommendation did not select exactly one arm".to_owned());
    }
    match &selected[0].arm {
        AllocationArm::Direction(direction_sha) => direction_digests
            .iter()
            .find(|(_, sha)| sha.as_str() == direction_sha.as_str())
            .map(|(direction_id, _)| direction_id.clone())
            .ok_or_else(|| "board recommended an unknown direction digest".to_owned()),
        AllocationArm::NewDirection => Ok("NEW".to_owned()),
    }
}

fn select_score_only(
    observations: &BTreeMap<String, Vec<f64>>,
) -> Result<String, String> {
    observations
        .iter()
        .filter_map(|(id, values)| values.iter().copied().max_by(f64::total_cmp).map(|best| (id, best)))
        .max_by(|(left_id, left_score), (right_id, right_score)| {
            left_score
                .total_cmp(right_score)
                .then_with(|| right_id.cmp(left_id))
        })
        .map(|(id, _)| id.clone())
        .ok_or_else(|| "score-only policy has no evaluated direction".to_owned())
}

fn select_round_robin(directions: &[String], cursor: &mut usize) -> String {
    let mut arms = directions.to_vec();
    arms.push("NEW".to_owned());
    let selected = arms[*cursor % arms.len()].clone();
    *cursor += 1;
    selected
}

fn select_random(directions: &[String], state: &mut u64) -> String {
    let mut arms = directions.to_vec();
    arms.push("NEW".to_owned());
    let selected = (next_random(state) as usize) % arms.len();
    arms[selected].clone()
}

fn select_global_ucb(
    observations: &BTreeMap<String, Vec<f64>>,
    proposal_count: usize,
    admitted_count: usize,
    total_actions: usize,
) -> String {
    let log_total = ((total_actions as f64) + 2.0).ln();
    let mut best_arm = "NEW".to_owned();
    let new_mean = if proposal_count == 0 {
        NEW_DIRECTION_CREDIT
    } else {
        NEW_DIRECTION_CREDIT * admitted_count as f64 / proposal_count as f64
    };
    let new_bonus = GLOBAL_UCB_EXPLORATION
        * (2.0 * log_total / (proposal_count as f64 + 1.0)).sqrt();
    let mut best_potential = new_mean + new_bonus;

    for (direction_id, values) in observations {
        // An admitted but untried direction is still an arm. Omitting it would
        // make the global-UCB control unable to discover/evaluate new operators.
        let mean = if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        };
        let bonus = GLOBAL_UCB_EXPLORATION
            * (2.0 * log_total / (values.len() as f64 + 1.0)).sqrt();
        let potential = mean + bonus;
        if potential > best_potential
            || ((potential - best_potential).abs() <= f64::EPSILON
                && direction_id.as_str() < best_arm.as_str())
        {
            best_arm.clone_from(direction_id);
            best_potential = potential;
        }
    }
    best_arm
}

/// Select by recency-weighted reward and effective-sample UCB bonus.
///
/// This is a discounted-UCB control, not a claim to reproduce one unique literature
/// configuration. The discount factor is frozen in the benchmark source and recorded
/// by the campaign version. NEW receives one fixed pseudo-observation at the declared
/// discovery prior; every proposal disposition is retained as a Bernoulli reward.
fn select_discounted_ucb(
    observations: &BTreeMap<String, Vec<f64>>,
    new_proposal_successes: &[bool],
    total_actions: usize,
) -> String {
    let log_total = ((total_actions as f64) + 2.0).ln();
    let (new_sum, new_weight) = discounted_binary_history(new_proposal_successes);
    let new_mean = (new_sum + NEW_DIRECTION_CREDIT) / (new_weight + 1.0);
    let new_bonus = (2.0 * log_total / (new_weight + 2.0)).sqrt();
    let mut best_arm = "NEW".to_owned();
    let mut best_potential = new_mean + new_bonus;

    for (direction_id, values) in observations {
        let mut weight = 1.0;
        let mut weighted_sum = 0.0;
        let mut effective_count = 0.0;
        for value in values.iter().rev() {
            weighted_sum += weight * value;
            effective_count += weight;
            weight *= DISCOUNT_FACTOR;
        }
        let mean = if effective_count > 0.0 {
            weighted_sum / effective_count
        } else {
            0.0
        };
        let bonus = (2.0 * log_total / (effective_count + 1.0)).sqrt();
        let potential = mean + bonus;
        if potential > best_potential
            || ((potential - best_potential).abs() <= f64::EPSILON
                && direction_id.as_str() < best_arm.as_str())
        {
            best_arm.clone_from(direction_id);
            best_potential = potential;
        }
    }
    best_arm
}

fn discounted_binary_history(outcomes: &[bool]) -> (f64, f64) {
    let mut weight = 1.0;
    let mut weighted_successes = 0.0;
    let mut effective_count = 0.0;
    for success in outcomes.iter().rev() {
        if *success {
            weighted_successes += weight;
        }
        effective_count += weight;
        weight *= DISCOUNT_FACTOR;
    }
    (weighted_successes, effective_count)
}

/// Beta-Bernoulli Thompson sampling over target-hit probability.
///
/// Each verified score is converted to a Bernoulli outcome using the frozen campaign
/// target. NEW proposals are successful only when the independent synthetic verifier
/// admits a genuinely novel direction; duplicates/rejections are failures. Gamma
/// variates use the Marsaglia-Tsang sampler and a seeded SplitMix stream.
fn select_thompson_sampling(
    observations: &BTreeMap<String, Vec<f64>>,
    new_proposal_successes: &[bool],
    random_state: &mut u64,
) -> String {
    let mut best_arm = "NEW".to_owned();
    let new_successes = new_proposal_successes.iter().filter(|success| **success).count();
    let new_failures = new_proposal_successes.len() - new_successes;
    let mut best_sample = sample_beta(
        1.0 + new_successes as f64,
        1.0 + new_failures as f64,
        random_state,
    );

    for (direction_id, values) in observations {
        let successes = values.iter().filter(|score| **score >= TARGET_SCORE).count();
        let failures = values.len() - successes;
        let sample = sample_beta(
            1.0 + successes as f64,
            1.0 + failures as f64,
            random_state,
        );
        if sample > best_sample
            || ((sample - best_sample).abs() <= f64::EPSILON
                && direction_id.as_str() < best_arm.as_str())
        {
            best_arm.clone_from(direction_id);
            best_sample = sample;
        }
    }
    best_arm
}

fn sample_beta(alpha: f64, beta: f64, state: &mut u64) -> f64 {
    let left = sample_gamma(alpha, state);
    let right = sample_gamma(beta, state);
    let total = left + right;
    if total.is_finite() && total > 0.0 {
        (left / total).clamp(0.0, 1.0)
    } else {
        alpha / (alpha + beta)
    }
}

fn sample_gamma(shape: f64, state: &mut u64) -> f64 {
    if !shape.is_finite() || shape <= 0.0 {
        return 0.0;
    }
    if shape < 1.0 {
        let uniform = unit_random(state).max(1e-12);
        return sample_gamma(shape + 1.0, state) * uniform.powf(1.0 / shape);
    }

    let d = shape - (1.0 / 3.0);
    let c = (9.0 * d).sqrt().recip();
    for _ in 0..1024 {
        let normal = standard_normal(state);
        let base = 1.0 + c * normal;
        if base <= 0.0 {
            continue;
        }
        let v = base * base * base;
        let uniform = unit_random(state).max(1e-12);
        if uniform < 1.0 - 0.0331 * normal.powi(4)
            || uniform.ln() < 0.5 * normal * normal + d * (1.0 - v + v.ln())
        {
            return d * v;
        }
    }
    // Deterministic finite fallback for an extraordinarily unlikely rejection tail.
    shape
}

fn standard_normal(state: &mut u64) -> f64 {
    let u1 = unit_random(state).max(1e-12);
    let u2 = unit_random(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn unit_random(state: &mut u64) -> f64 {
    ((next_random(state) >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn select_oracle(
    landscape: DirectionLandscapeFamily,
    observations: &BTreeMap<String, Vec<f64>>,
    new_admitted: bool,
) -> String {
    if landscape_has_new_operator(landscape) && !new_admitted {
        let best_existing = observations
            .keys()
            .map(|id| maximum_future_payoff(landscape, id, observations[id].len()))
            .fold(0.0, f64::max);
        if maximum_future_payoff(landscape, "D", 0) > best_existing {
            return "NEW".to_owned();
        }
    }

    observations
        .iter()
        .map(|(id, values)| {
            (id, maximum_future_payoff(landscape, id, values.len()))
        })
        .max_by(|(left_id, left_score), (right_id, right_score)| {
            left_score
                .total_cmp(right_score)
                .then_with(|| right_id.cmp(left_id))
        })
        .map(|(id, _)| id.clone())
        .unwrap_or_else(|| "NEW".to_owned())
}

fn best_observed_direction(
    observations: &BTreeMap<String, Vec<f64>>,
) -> Result<(String, f64), String> {
    observations
        .iter()
        .filter_map(|(id, values)| values.iter().copied().max_by(f64::total_cmp).map(|score| (id, score)))
        .max_by(|(left_id, left_score), (right_id, right_score)| {
            left_score
                .total_cmp(right_score)
                .then_with(|| right_id.cmp(left_id))
        })
        .map(|(id, score)| (id.clone(), score))
        .ok_or_else(|| "campaign has no verified direction evaluations".to_owned())
}

fn initial_descriptor(direction_id: &str) -> DirectionDescriptor {
    DirectionDescriptor {
        target_capability: "held-out task transfer".to_owned(),
        solution_object: direction_id.to_owned(),
        construction_principle: format!("known construction principle {direction_id}"),
        scope: "synthetic architecture-search landscape".to_owned(),
    }
}

fn new_direction_descriptor(alias: bool) -> DirectionDescriptor {
    if alias {
        DirectionDescriptor {
            target_capability: "  NOVEL HELD-OUT TRANSFER  ".to_owned(),
            solution_object: " OPERATOR-FAMILY EXTENSION ".to_owned(),
            construction_principle: " COMPOSITIONAL RELATIONAL OPERATOR ".to_owned(),
            scope: " SYNTHETIC GRAMMAR EXTENSION ".to_owned(),
        }
    } else {
        DirectionDescriptor {
            target_capability: "novel held-out transfer".to_owned(),
            solution_object: "operator-family extension".to_owned(),
            construction_principle: "compositional relational operator".to_owned(),
            scope: "synthetic grammar extension".to_owned(),
        }
    }
}

fn landscape_has_new_operator(landscape: DirectionLandscapeFamily) -> bool {
    landscape == DirectionLandscapeFamily::NewOperatorBreakthrough
}

fn oracle_best_score(landscape: DirectionLandscapeFamily) -> f64 {
    let mut best = ["A", "B", "C"]
        .iter()
        .map(|id| maximum_future_payoff(landscape, id, 0))
        .fold(0.0, f64::max);
    if landscape_has_new_operator(landscape) {
        best = best.max(maximum_future_payoff(landscape, "D", 0));
    }
    best
}

fn maximum_future_payoff(
    landscape: DirectionLandscapeFamily,
    direction_id: &str,
    evaluation_index: usize,
) -> f64 {
    let schedule = schedule(landscape, direction_id);
    if schedule.is_empty() {
        return 0.0;
    }
    schedule
        .iter()
        .skip(evaluation_index.min(schedule.len()))
        .copied()
        .fold(
            if evaluation_index >= schedule.len() {
                *schedule.last().unwrap_or(&0.0)
            } else {
                0.0
            },
            f64::max,
        )
}

fn payoff(
    landscape: DirectionLandscapeFamily,
    direction_id: &str,
    evaluation_index: usize,
) -> f64 {
    let schedule = schedule(landscape, direction_id);
    if schedule.is_empty() {
        return 0.0;
    }
    // Exact deterministic schedules make the oracle truly exact. Policy randomness is
    // introduced only through arm selection, never by perturbing evaluator ground truth.
    schedule[evaluation_index.min(schedule.len() - 1)]
}

fn schedule(
    landscape: DirectionLandscapeFamily,
    direction_id: &str,
) -> &'static [f64] {
    match (landscape, direction_id) {
        (DirectionLandscapeFamily::DelayedBreakthrough, "A") => &[0.82, 0.84, 0.84],
        (DirectionLandscapeFamily::DelayedBreakthrough, "B") => &[0.10, 0.14, 0.18, 0.22, 0.96],
        (DirectionLandscapeFamily::DelayedBreakthrough, "C") => &[0.55, 0.57, 0.58],
        (DirectionLandscapeFamily::NoNewDirection, "A") => &[0.86, 0.88],
        (DirectionLandscapeFamily::NoNewDirection, "B") => &[0.63, 0.65],
        (DirectionLandscapeFamily::NoNewDirection, "C") => &[0.72, 0.74],
        (DirectionLandscapeFamily::NewOperatorBreakthrough, "A") => &[0.82, 0.84],
        (DirectionLandscapeFamily::NewOperatorBreakthrough, "B") => &[0.40, 0.42, 0.42],
        (DirectionLandscapeFamily::NewOperatorBreakthrough, "C") => &[0.56, 0.58],
        (DirectionLandscapeFamily::NewOperatorBreakthrough, "D") => &[0.06, 0.08, 0.97],
        (DirectionLandscapeFamily::RankReversal, "A") => &[0.92, 0.42, 0.25],
        (DirectionLandscapeFamily::RankReversal, "B") => &[0.18, 0.35, 0.95],
        (DirectionLandscapeFamily::RankReversal, "C") => &[0.58, 0.60, 0.61],
        (DirectionLandscapeFamily::DeceptiveProxy, "A") => &[0.93, 0.92, 0.35, 0.25],
        (DirectionLandscapeFamily::DeceptiveProxy, "B") => &[0.45, 0.55, 0.75, 0.96],
        (DirectionLandscapeFamily::DeceptiveProxy, "C") => &[0.62, 0.63, 0.64, 0.65],
        (DirectionLandscapeFamily::HeteroscedasticRewards, "A") => {
            &[0.50, 0.92, 0.32, 0.88, 0.28, 0.95]
        }
        (DirectionLandscapeFamily::HeteroscedasticRewards, "B") => {
            &[0.65, 0.66, 0.64, 0.66, 0.65, 0.66]
        }
        (DirectionLandscapeFamily::HeteroscedasticRewards, "C") => {
            &[0.55, 0.57, 0.56, 0.59, 0.58, 0.60]
        }
        (DirectionLandscapeFamily::CorrelatedDirections, "A") => {
            &[0.60, 0.66, 0.70, 0.73, 0.75]
        }
        (DirectionLandscapeFamily::CorrelatedDirections, "B") => {
            &[0.61, 0.65, 0.71, 0.72, 0.74]
        }
        (DirectionLandscapeFamily::CorrelatedDirections, "C") => {
            &[0.50, 0.52, 0.55, 0.57, 0.58]
        }
        (DirectionLandscapeFamily::NullNoBreakthrough, "A") => {
            &[0.60, 0.60, 0.60, 0.60, 0.60]
        }
        (DirectionLandscapeFamily::NullNoBreakthrough, "B") => {
            &[0.60, 0.60, 0.60, 0.60, 0.60]
        }
        (DirectionLandscapeFamily::NullNoBreakthrough, "C") => {
            &[0.60, 0.60, 0.60, 0.60, 0.60]
        }
        _ => &[],
    }
}

fn digest(value: &str) -> String {
    digest_bytes(value.as_bytes())
}

fn digest_bytes(value: &[u8]) -> String {
    let bytes = Sha256::digest(value);
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn next_random(state: &mut u64) -> u64 {
    *state = splitmix64(*state);
    *state
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}


/// Convert this synthetic campaign into the common AR-5B4 event tape.
///
/// This adapter intentionally marks unmeasured host/accelerator/energy/token dimensions as
/// Unknown. The resulting tape is integrity-checkable but cannot pass the qualified
/// matched-budget gate until those resource dimensions are instrumented.
impl DirectionCampaignRun {
    pub fn to_event_tape(
        &self,
        source_commit: &str,
        execution_environment_sha256: &str,
    ) -> Result<EventTape, String> {
        if self.action_count != self.actions.len() {
            return Err("action_count does not match the recorded action list".to_owned());
        }
        if self.action_count > self.action_budget {
            return Err("recorded actions exceed the declared action budget".to_owned());
        }
        let summed_cost = self.actions.iter().try_fold(0u64, |sum, action| {
            sum.checked_add(action.cost_units)
                .ok_or_else(|| "synthetic action cost overflow".to_owned())
        })?;
        if summed_cost != self.total_cost_units {
            return Err("recorded action costs do not reconcile to total_cost_units".to_owned());
        }

        let landscape_id = serde_json::to_value(self.landscape)
            .map_err(|error| format!("landscape serialization failed: {error}"))?
            .as_str()
            .ok_or_else(|| "landscape serialization was not a string".to_owned())?
            .to_owned();
        let policy_id = serde_json::to_value(self.policy)
            .map_err(|error| format!("policy serialization failed: {error}"))?
            .as_str()
            .ok_or_else(|| "policy serialization was not a string".to_owned())?
            .to_owned();
        let config = serde_json::to_value(&self.policy_config)
            .map_err(|error| format!("policy config serialization failed: {error}"))?;
        let config_sha256 = digest_bytes(
            &serde_json::to_vec(&config)
                .map_err(|error| format!("policy config encoding failed: {error}"))?,
        );
        if config_sha256 != self.policy_config_sha256 {
            return Err("policy configuration digest does not match the run snapshot".to_owned());
        }

        let campaign_id = format!(
            "ar5b4-synthetic-{landscape_id}-seed-{}-budget-{}",
            self.seed, self.action_budget
        );
        let task_recipe_sha256 = digest("symthaea.synthetic-direction.recipe.v0.1");
        let evaluator_sha256 = digest("symthaea.synthetic-direction.evaluator.v0.1");
        let verifier_sha256 = digest("symthaea.synthetic-direction.verifier.v0.1");
        let proposal_stream_sha256 =
            digest(&format!("synthetic-payoff-stream:{landscape_id}:{}", self.seed));
        let action_cost_microunits = self
            .policy_config
            .action_cost_units
            .checked_mul(1_000_000)
            .ok_or_else(|| "normalized action-unit price overflow".to_owned())?;
        let budget_limit_microunits = u64::try_from(self.action_budget)
            .map_err(|_| "action budget does not fit u64".to_owned())?
            .checked_mul(action_cost_microunits)
            .ok_or_else(|| "normalized action budget overflow".to_owned())?;
        let budget_model = BudgetModel::ActionUnit {
            charge_per_action_microunits: action_cost_microunits,
        };
        let budget_model_sha256 = digest_bytes(
            &serde_json::to_vec(&budget_model)
                .map_err(|error| format!("budget model encoding failed: {error}"))?,
        );

        let mut tape = EventTape::new(EventTapeHeader {
            schema: "symthaea.ar5b4.event-tape".to_owned(),
            version: "0.1.0".to_owned(),
            campaign_id,
            source_commit: source_commit.to_owned(),
            execution_environment_sha256: execution_environment_sha256.to_owned(),
            comparison_mode: ComparisonMode::MechanismMatched,
            proposal_source_id: "synthetic-direction-campaign-v0.1".to_owned(),
            proposal_stream_sha256: Some(proposal_stream_sha256),
            task_recipe_sha256: task_recipe_sha256.clone(),
            comparison_profile_sha256: PROFILE_SHA256.to_owned(),
            evaluator_sha256: evaluator_sha256.clone(),
            verifier_sha256: verifier_sha256.clone(),
            budget_model_id: "synthetic-action-unit-v0.1".to_owned(),
            budget_model,
            budget_model_sha256,
            budget_limit_microunits,
            initial_direction_ids: vec!["A".to_owned(), "B".to_owned(), "C".to_owned()],
            execution: ExecutionContract {
                mode: ExecutionMode::Sequential,
                worker_count: 1,
                seats_per_round: 1,
                first_round_claim_cap: None,
                steady_state_claim_cap: None,
            },
            policy: PolicySnapshot {
                policy_id: policy_id.clone(),
                algorithm_id: format!("symthaea-direction-policy-{policy_id}-v0.1"),
                config_sha256,
                config,
                has_hidden_oracle_access: self.policy
                    == DirectionCampaignPolicy::OracleUpperBound,
            },
        })
        .map_err(|error| error.to_string())?;

        let mut evaluations_per_direction = BTreeMap::<String, u64>::new();
        let mut proposal_index = 0u64;
        for action in &self.actions {
            if action.cost_units != self.policy_config.action_cost_units {
                return Err(format!(
                    "action slot {} has cost {}, but the frozen synthetic action-unit model requires {}",
                    action.slot, action.cost_units, self.policy_config.action_cost_units
                ));
            }
            let slot = u64::try_from(action.slot)
                .map_err(|_| "action slot does not fit u64".to_owned())?;
            if slot == 0 || slot != tape.events().len() as u64 + 1 {
                return Err("action slots must be contiguous and start at one".to_owned());
            }
            let event_id = format!("{}:event:{slot}", tape.header().campaign_id);
            let (intent, outcome) = if action.is_new_proposal {
                proposal_index = proposal_index
                    .checked_add(1)
                    .ok_or_else(|| "proposal index overflow".to_owned())?;
                let disposition = match action.proposal_disposition {
                    Some(NewDirectionDisposition::Admitted) => {
                        ProposalDisposition::Admitted {
                            canonical_direction_id: "D".to_owned(),
                        }
                    }
                    Some(NewDirectionDisposition::Duplicate) => {
                        ProposalDisposition::Duplicate {
                            canonical_direction_id: "D".to_owned(),
                        }
                    }
                    Some(NewDirectionDisposition::Rejected) => ProposalDisposition::Rejected,
                    Some(NewDirectionDisposition::Inconclusive) => {
                        ProposalDisposition::Inconclusive
                    }
                    None => {
                        return Err(format!(
                            "NEW proposal at action slot {} lacks verifier disposition",
                            action.slot
                        ));
                    }
                };
                let descriptor_sha256 = digest(&format!(
                    "synthetic-direction-descriptor:{}:{}",
                    landscape_id, proposal_index
                ));
                (
                    ActionIntent::ProposeDirection {
                        descriptor_sha256: descriptor_sha256.clone(),
                        parent_direction_id: None,
                        semantic_delta_sha256: descriptor_sha256,
                    },
                    ActionOutcome::DirectionVerification {
                        task_recipe_sha256: task_recipe_sha256.clone(),
                        comparison_profile_sha256: PROFILE_SHA256.to_owned(),
                        verifier_sha256: verifier_sha256.clone(),
                        verifier_receipt_sha256: digest(&format!(
                            "synthetic-verifier-receipt:{}:{}",
                            landscape_id, proposal_index
                        )),
                        disposition,
                    },
                )
            } else {
                if action.proposal_disposition.is_some() {
                    return Err(format!(
                        "candidate evaluation at action slot {} has a proposal disposition",
                        action.slot
                    ));
                }
                let score = action.observed_score.ok_or_else(|| {
                    format!(
                        "candidate evaluation at action slot {} lacks a verified score",
                        action.slot
                    )
                })?;
                let evaluation_index = {
                    let count = evaluations_per_direction
                        .entry(action.arm_id.clone())
                        .or_default();
                    *count = count
                        .checked_add(1)
                        .ok_or_else(|| "direction evaluation index overflow".to_owned())?;
                    *count
                };
                let candidate_sha256 = digest(&format!(
                    "synthetic-candidate:{}:{}:{}",
                    landscape_id, action.arm_id, evaluation_index
                ));
                let semantic_delta_sha256 =
                    digest(&format!("synthetic-delta:{}:{}", landscape_id, action.arm_id));
                (
                    ActionIntent::EvaluateCandidate {
                        candidate_sha256: candidate_sha256.clone(),
                        parent_candidate_sha256: None,
                        semantic_delta_sha256,
                        direction_id: action.arm_id.clone(),
                    },
                    ActionOutcome::CandidateEvaluation {
                        task_recipe_sha256: task_recipe_sha256.clone(),
                        comparison_profile_sha256: PROFILE_SHA256.to_owned(),
                        evaluator_sha256: evaluator_sha256.clone(),
                        verifier_sha256: verifier_sha256.clone(),
                        verifier_receipt_sha256: digest(&format!(
                            "synthetic-candidate-verifier-receipt:{}:{}",
                            candidate_sha256, evaluation_index
                        )),
                        disposition: CandidateDisposition::Verified,
                        score: Some(score),
                        uncertainty: None,
                        evidence_sha256: Some(digest(&format!(
                            "synthetic-evidence:{}:{}:{score:.17}",
                            candidate_sha256, evaluation_index
                        ))),
                    },
                )
            };
            let charge = action
                .cost_units
                .checked_mul(1_000_000)
                .ok_or_else(|| "normalized action charge overflow".to_owned())?;
            let unknown = |reason: &str| MeterValue::Unknown {
                reason: reason.to_owned(),
            };
            let resources = ResourceReceipt {
                host_cpu_millis: unknown("synthetic campaign does not meter host CPU"),
                wall_millis: unknown("synthetic campaign does not meter per-action wall time"),
                accelerator_millis: unknown("synthetic campaign does not meter accelerator time"),
                peak_memory_bytes: unknown("synthetic campaign does not meter host peak memory"),
                accelerator_peak_memory_bytes: unknown(
                    "synthetic campaign does not meter accelerator peak memory"
                ),
                energy_millijoules: unknown("synthetic campaign does not meter energy"),
                input_tokens: unknown("synthetic campaign does not meter model tokens"),
                output_tokens: unknown("synthetic campaign does not meter model tokens"),
                tool_calls: unknown("synthetic campaign does not meter external tool calls"),
                evaluator_calls: meter_value(if action.is_new_proposal { 0 } else { 1 }),
                verifier_calls: meter_value(1),
                charged_budget_microunits: meter_value(charge),
            };
            tape.append(ActionEvent {
                event_id,
                policy_id: policy_id.clone(),
                worker_id: 0,
                round_id: slot,
                batch_id: None,
                observed_through_sequence: slot - 1,
                visible_history_sha256: String::new(),
                retry_of_event_id: None,
                intent,
                outcome,
                resources,
            })
            .map_err(|error| error.to_string())?;
        }

        let summary = tape.validate_integrity().map_err(|error| error.to_string())?;
        let expected_total_charge = self
            .total_cost_units
            .checked_mul(1_000_000)
            .ok_or_else(|| "synthetic campaign normalized cost overflow".to_owned())?;
        if summary.total_charged_budget_microunits != Some(expected_total_charge) {
            return Err("event-tape charge does not reconcile to the campaign cost".to_owned());
        }
        Ok(tape)
    }
}

fn meter_value(value: u64) -> MeterValue {
    MeterValue::Measured { value }
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICIES: [DirectionCampaignPolicy; 8] = [
        DirectionCampaignPolicy::ScoreOnlyWinnerTakesAll,
        DirectionCampaignPolicy::FixedRoundRobin,
        DirectionCampaignPolicy::Random,
        DirectionCampaignPolicy::GlobalUcb,
        DirectionCampaignPolicy::DiscountedUcb,
        DirectionCampaignPolicy::ThompsonSampling,
        DirectionCampaignPolicy::VerifierGroundedDirectionBoard,
        DirectionCampaignPolicy::OracleUpperBound,
    ];

    const LANDSCAPES: [DirectionLandscapeFamily; 8] = [
        DirectionLandscapeFamily::DelayedBreakthrough,
        DirectionLandscapeFamily::NoNewDirection,
        DirectionLandscapeFamily::NewOperatorBreakthrough,
        DirectionLandscapeFamily::RankReversal,
        DirectionLandscapeFamily::DeceptiveProxy,
        DirectionLandscapeFamily::HeteroscedasticRewards,
        DirectionLandscapeFamily::CorrelatedDirections,
        DirectionLandscapeFamily::NullNoBreakthrough,
    ];

    #[test]
    fn every_policy_obeys_exact_unit_action_budget() {
        for landscape in LANDSCAPES {
            for policy in POLICIES {
                let run = run_direction_campaign(landscape, policy, 48, 17)
                    .expect("synthetic campaign should run");
                assert_eq!(run.action_count, 48);
                assert_eq!(run.actions.len(), 48);
                assert_eq!(run.total_cost_units, 48);
                assert!(run.simple_regret >= 0.0);
                assert_eq!(run.final_board_sha256.len(), 64);
            }
        }
    }

    #[test]
    fn random_and_board_policies_are_reproducible_for_the_same_seed() {
        for policy in [
            DirectionCampaignPolicy::Random,
            DirectionCampaignPolicy::DiscountedUcb,
            DirectionCampaignPolicy::ThompsonSampling,
            DirectionCampaignPolicy::VerifierGroundedDirectionBoard,
        ] {
            let first = run_direction_campaign(
                DirectionLandscapeFamily::NewOperatorBreakthrough,
                policy,
                64,
                0x1234,
            )
            .unwrap();
            let second = run_direction_campaign(
                DirectionLandscapeFamily::NewOperatorBreakthrough,
                policy,
                64,
                0x1234,
            )
            .unwrap();
            assert_eq!(first, second);
        }
    }

    #[test]
    fn policy_configuration_snapshot_is_complete_and_digest_bound() {
        let run = run_direction_campaign(
            DirectionLandscapeFamily::DelayedBreakthrough,
            DirectionCampaignPolicy::ThompsonSampling,
            32,
            0xabc,
        )
        .expect("campaign should run");
        let value = serde_json::to_value(&run.policy_config).expect("config serializes");
        let bytes = serde_json::to_vec(&value).expect("config value encodes");
        assert_eq!(run.policy_config_sha256, digest_bytes(&bytes));
        assert_eq!(run.policy_config.target_score, TARGET_SCORE);
        assert_eq!(run.policy_config.common_bootstrap_evaluations, 3);
        assert_eq!(run.policy_config.thompson_prior_alpha, 1.0);
        assert_eq!(run.policy_config.thompson_prior_beta, 1.0);
        assert!(!run.policy_config.oracle_uses_hidden_payoff_schedule);
    }

    #[test]
    fn discounted_ucb_prefers_recent_payoff_after_rank_reversal() {
        let mut observations = BTreeMap::new();
        observations.insert("A".to_owned(), vec![0.95, 0.20, 0.10]);
        observations.insert("B".to_owned(), vec![0.20, 0.30, 0.95]);
        let stale_failed_proposals = vec![false; 100];
        let selected = select_discounted_ucb(&observations, &stale_failed_proposals, 106);
        assert_eq!(selected, "B");
    }

    #[test]
    fn thompson_sampling_obeys_budget_and_is_seed_reproducible() {
        let first = run_direction_campaign(
            DirectionLandscapeFamily::NewOperatorBreakthrough,
            DirectionCampaignPolicy::ThompsonSampling,
            64,
            0x5eed,
        )
        .expect("Thompson campaign should run");
        let second = run_direction_campaign(
            DirectionLandscapeFamily::NewOperatorBreakthrough,
            DirectionCampaignPolicy::ThompsonSampling,
            64,
            0x5eed,
        )
        .expect("Thompson replay should run");
        assert_eq!(first, second);
        assert_eq!(first.action_count, 64);
        assert_eq!(first.total_cost_units, 64);
    }

    #[test]
    fn discounted_ucb_campaign_obeys_budget_and_preserves_nonstationary_policy_id() {
        let run = run_direction_campaign(
            DirectionLandscapeFamily::RankReversal,
            DirectionCampaignPolicy::DiscountedUcb,
            64,
            0xdecaf,
        )
        .expect("discounted-UCB campaign should run");
        assert_eq!(run.action_count, 64);
        assert_eq!(run.total_cost_units, 64);
        assert_eq!(run.policy, DirectionCampaignPolicy::DiscountedUcb);
    }

    #[test]
    fn score_only_policy_does_not_open_a_new_direction() {
        let run = run_direction_campaign(
            DirectionLandscapeFamily::NewOperatorBreakthrough,
            DirectionCampaignPolicy::ScoreOnlyWinnerTakesAll,
            32,
            23,
        )
        .unwrap();
        assert_eq!(run.admitted_new_directions, 0);
        assert_eq!(run.final_best_direction, "A");
        assert!(run.final_best_score < TARGET_SCORE);
    }

    #[test]
    fn board_policy_uses_verifier_to_admit_a_new_operator() {
        let run = run_direction_campaign(
            DirectionLandscapeFamily::NewOperatorBreakthrough,
            DirectionCampaignPolicy::VerifierGroundedDirectionBoard,
            48,
            91,
        )
        .unwrap();
        assert_eq!(run.admitted_new_directions, 1);
        assert!(run.actions.iter().any(|action| action.arm_id == "NEW"
            && action.proposal_disposition == Some(NewDirectionDisposition::Admitted)));
        assert!(run.actions.iter().any(|action| action.arm_id == "D"
            && action.observed_score.is_some()));
    }

    #[test]
    fn global_ucb_can_evaluate_a_newly_admitted_direction() {
        let run = run_direction_campaign(
            DirectionLandscapeFamily::NewOperatorBreakthrough,
            DirectionCampaignPolicy::GlobalUcb,
            64,
            71,
        )
        .unwrap();
        assert_eq!(run.admitted_new_directions, 1);
        assert!(run.actions.iter().any(|action| action.arm_id == "D"
            && action.observed_score.is_some()));
    }

    #[test]
    fn no_new_direction_is_a_real_null_not_an_automatic_success() {
        let run = run_direction_campaign(
            DirectionLandscapeFamily::NoNewDirection,
            DirectionCampaignPolicy::VerifierGroundedDirectionBoard,
            48,
            51,
        )
        .unwrap();
        assert_eq!(run.admitted_new_directions, 0);
        assert!(run.rejected_new_proposals > 0);
        assert!(run.final_best_score < TARGET_SCORE);
    }

    #[test]
    fn paired_comparisons_require_identical_seed_sets() {
        let seeds = [11, 22, 33];
        let runs = run_direction_campaign_matrix(
            &[DirectionLandscapeFamily::DelayedBreakthrough],
            &POLICIES,
            24,
            &seeds,
        )
        .unwrap();
        let comparisons = compare_paired_direction_campaign_runs(
            &runs,
            DirectionCampaignPolicy::VerifierGroundedDirectionBoard,
        )
        .unwrap();
        assert_eq!(comparisons.len(), POLICIES.len() - 1);
        assert!(comparisons.iter().all(|comparison| comparison.pair_count == 3));

        assert!(compare_paired_direction_campaign_runs(
            &runs[..runs.len() - 1],
            DirectionCampaignPolicy::VerifierGroundedDirectionBoard,
        )
        .is_err());
    }

    #[test]
    fn aggregate_rejects_duplicate_seeds_and_conserves_cost() {
        let run = run_direction_campaign(
            DirectionLandscapeFamily::DelayedBreakthrough,
            DirectionCampaignPolicy::VerifierGroundedDirectionBoard,
            32,
            7,
        )
        .unwrap();
        let aggregate = aggregate_direction_campaign_runs(&[run.clone()]).unwrap();
        assert_eq!(aggregate.len(), 1);
        assert_eq!(aggregate[0].seed_count, 1);
        assert_eq!(aggregate[0].mean_total_cost_units, 32.0);
        assert!(aggregate_direction_campaign_runs(&[run.clone(), run]).is_err());
    }

    #[test]
    fn deceptive_proxy_reverses_final_best_direction_after_an_early_proxy_lead() {
        let early_proxy = payoff(DirectionLandscapeFamily::DeceptiveProxy, "A", 0);
        let delayed_transfer = payoff(DirectionLandscapeFamily::DeceptiveProxy, "B", 3);
        assert!(early_proxy > delayed_transfer * 0.9);
        assert!(delayed_transfer > early_proxy);
        assert_eq!(oracle_best_score(DirectionLandscapeFamily::DeceptiveProxy), 0.96);
        assert!(!landscape_has_new_operator(DirectionLandscapeFamily::DeceptiveProxy));
    }

    #[test]
    fn heteroscedastic_reward_control_has_distinct_arm_variances() {
        let high_variance = schedule(
            DirectionLandscapeFamily::HeteroscedasticRewards,
            "A",
        );
        let low_variance = schedule(
            DirectionLandscapeFamily::HeteroscedasticRewards,
            "B",
        );
        let variance = |values: &[f64]| {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            values
                .iter()
                .map(|value| (value - mean).powi(2))
                .sum::<f64>()
                / values.len() as f64
        };
        assert!(variance(high_variance) > variance(low_variance) * 20.0);
    }

    #[test]
    fn correlated_direction_control_uses_near_parallel_reward_profiles() {
        let left = schedule(DirectionLandscapeFamily::CorrelatedDirections, "A");
        let right = schedule(DirectionLandscapeFamily::CorrelatedDirections, "B");
        assert_eq!(left.len(), right.len());
        assert!(left.iter().zip(right).all(|(a, b)| (a - b).abs() <= 0.02));
        assert!(!landscape_has_new_operator(DirectionLandscapeFamily::CorrelatedDirections));
    }

    #[test]
    fn null_no_breakthrough_control_has_no_directional_or_new_operator_advantage() {
        let a = schedule(DirectionLandscapeFamily::NullNoBreakthrough, "A");
        let b = schedule(DirectionLandscapeFamily::NullNoBreakthrough, "B");
        let c = schedule(DirectionLandscapeFamily::NullNoBreakthrough, "C");
        assert!(a.iter().all(|value| *value == 0.60));
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert!(!landscape_has_new_operator(DirectionLandscapeFamily::NullNoBreakthrough));
        assert_eq!(oracle_best_score(DirectionLandscapeFamily::NullNoBreakthrough), 0.60);
    }

    #[test]
    fn synthetic_campaign_emits_integrity_checkable_but_unmetered_event_tape() {
        let run = run_direction_campaign(
            DirectionLandscapeFamily::NewOperatorBreakthrough,
            DirectionCampaignPolicy::ThompsonSampling,
            48,
            0x5eed,
        )
        .expect("synthetic campaign should run");
        let tape = run
            .to_event_tape(
                "0123456789abcdef0123456789abcdef01234567",
                &digest("synthetic-test-environment-v0.1"),
            )
            .expect("campaign should convert to the common event tape");
        let summary = tape
            .validate_integrity()
            .expect("synthetic tape structure and action charges should validate");
        assert_eq!(summary.event_count, run.action_count);
        assert_eq!(
            summary.total_charged_budget_microunits,
            Some(run.total_cost_units * 1_000_000)
        );
        assert!(!summary.fully_metered);
        assert!(
            tape.validate().is_err(),
            "synthetic action-count receipts must not masquerade as complete host/energy metering"
        );
    }

}
