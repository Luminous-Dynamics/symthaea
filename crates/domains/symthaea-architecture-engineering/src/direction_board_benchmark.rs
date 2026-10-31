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
const DISCOUNTED_UCB_GAMMA: f64 = 0.90;
const GAUSSIAN_THOMPSON_PRIOR_VARIANCE: f64 = 0.25;
const GAUSSIAN_THOMPSON_OBSERVATION_VARIANCE: f64 = 0.25;

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
    /// Exponential-discount UCB for changing or rank-reversing payoff landscapes.
    DiscountedUcb,
    /// Gaussian Thompson sampling with a frozen Normal-Normal reward model.
    GaussianThompsonSampling,
    /// The implementation under test: verifier-grounded, local-record UCB board.
    VerifierGroundedDirectionBoard,
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

/// Full audit summary for one policy, landscape and seed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DirectionCampaignRun {
    pub schema: String,
    pub version: String,
    pub landscape: DirectionLandscapeFamily,
    pub policy: DirectionCampaignPolicy,
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
            DirectionCampaignPolicy::DiscountedUcb => {
                select_discounted_ucb(&observations, &actions, DISCOUNTED_UCB_GAMMA)
            }
            DirectionCampaignPolicy::GaussianThompsonSampling => {
                select_gaussian_thompson(&observations, &actions, &mut random_state)
            }
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
    Ok(DirectionCampaignRun {
        schema: "symthaea.direction-campaign.run".to_owned(),
        version: "0.1.0".to_owned(),
        landscape,
        policy,
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

fn action_reward(action: &DirectionCampaignAction) -> f64 {
    if action.is_new_proposal {
        if action.proposal_disposition == Some(NewDirectionDisposition::Admitted) {
            NEW_DIRECTION_CREDIT
        } else {
            0.0
        }
    } else {
        action.observed_score.unwrap_or(0.0).clamp(0.0, 1.0)
    }
}

/// Weighted reward sum and effective count for an arm under exponential recency discount.
fn discounted_arm_statistics(
    arm_id: &str,
    actions: &[DirectionCampaignAction],
    discount: f64,
) -> (f64, f64) {
    let mut weighted_reward = 0.0;
    let mut weighted_count = 0.0;
    for (index, action) in actions.iter().enumerate() {
        if action.arm_id != arm_id {
            continue;
        }
        let age = actions.len().saturating_sub(index + 1) as f64;
        let weight = discount.powf(age);
        weighted_reward += weight * action_reward(action);
        weighted_count += weight;
    }
    (weighted_reward, weighted_count)
}

/// Discounted UCB uses only past receipts and exponentially forgets older rewards.
/// A single frozen pseudo-count prior prevents zero-denominator startup behavior.
fn select_discounted_ucb(
    observations: &BTreeMap<String, Vec<f64>>,
    actions: &[DirectionCampaignAction],
    discount: f64,
) -> String {
    let mut arms: Vec<String> = observations.keys().cloned().collect();
    arms.push("NEW".to_owned());
    let log_total = ((actions.len() as f64) + 2.0).ln();
    let mut best: Option<(String, f64)> = None;
    for arm_id in arms {
        let (weighted_reward, weighted_count) =
            discounted_arm_statistics(&arm_id, actions, discount);
        let prior_reward = if arm_id == "NEW" { NEW_DIRECTION_CREDIT } else { 0.5 };
        let mean = (weighted_reward + prior_reward) / (weighted_count + 1.0);
        let bonus = GLOBAL_UCB_EXPLORATION
            * (2.0 * log_total / (weighted_count + 1.0)).sqrt();
        let potential = mean + bonus;
        let replace = match &best {
            None => true,
            Some((best_id, best_value)) => {
                potential.total_cmp(best_value).is_gt()
                    || (potential.total_cmp(best_value).is_eq() && arm_id.as_str() < best_id.as_str())
            }
        };
        if replace {
            best = Some((arm_id, potential));
        }
    }
    best.map(|(arm_id, _)| arm_id).unwrap_or_else(|| "NEW".to_owned())
}

/// Normal-Normal posterior for normalized continuous rewards. The variance assumptions
/// are fixed constants in this baseline, not fitted on the hidden landscape.
fn gaussian_thompson_posterior(
    arm_id: &str,
    actions: &[DirectionCampaignAction],
) -> (f64, f64) {
    let prior_mean = if arm_id == "NEW" { NEW_DIRECTION_CREDIT } else { 0.5 };
    let prior_precision = 1.0 / GAUSSIAN_THOMPSON_PRIOR_VARIANCE;
    let observation_precision = 1.0 / GAUSSIAN_THOMPSON_OBSERVATION_VARIANCE;
    let mut reward_sum = 0.0;
    let mut observation_count = 0.0;
    for action in actions.iter().filter(|action| action.arm_id == arm_id) {
        reward_sum += action_reward(action);
        observation_count += 1.0;
    }
    let posterior_precision = prior_precision + observation_count * observation_precision;
    let posterior_mean =
        (prior_mean * prior_precision + reward_sum * observation_precision) / posterior_precision;
    (posterior_mean, 1.0 / posterior_precision)
}

fn open_unit_interval(state: &mut u64) -> f64 {
    let mantissa = next_random(state) >> 11;
    (mantissa as f64 + 0.5) / ((1_u64 << 53) as f64)
}

fn sample_standard_normal(state: &mut u64) -> f64 {
    let u1 = open_unit_interval(state);
    let u2 = open_unit_interval(state);
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Seeded Gaussian Thompson sampling over the same normalized reward history.
fn select_gaussian_thompson(
    observations: &BTreeMap<String, Vec<f64>>,
    actions: &[DirectionCampaignAction],
    random_state: &mut u64,
) -> String {
    let mut arms: Vec<String> = observations.keys().cloned().collect();
    arms.push("NEW".to_owned());
    let mut best: Option<(String, f64)> = None;
    for arm_id in arms {
        let (posterior_mean, posterior_variance) =
            gaussian_thompson_posterior(&arm_id, actions);
        let sample =
            posterior_mean + posterior_variance.sqrt() * sample_standard_normal(random_state);
        let replace = match &best {
            None => true,
            Some((best_id, best_value)) => {
                sample.total_cmp(best_value).is_gt()
                    || (sample.total_cmp(best_value).is_eq() && arm_id.as_str() < best_id.as_str())
            }
        };
        if replace {
            best = Some((arm_id, sample));
        }
    }
    best.map(|(arm_id, _)| arm_id).unwrap_or_else(|| "NEW".to_owned())
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
        _ => &[],
    }
}

fn digest(value: &str) -> String {
    let bytes = Sha256::digest(value.as_bytes());
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

#[cfg(test)]
mod tests {
    use super::*;

    const POLICIES: [DirectionCampaignPolicy; 8] = [
        DirectionCampaignPolicy::ScoreOnlyWinnerTakesAll,
        DirectionCampaignPolicy::FixedRoundRobin,
        DirectionCampaignPolicy::Random,
        DirectionCampaignPolicy::GlobalUcb,
        DirectionCampaignPolicy::DiscountedUcb,
        DirectionCampaignPolicy::GaussianThompsonSampling,
        DirectionCampaignPolicy::VerifierGroundedDirectionBoard,
        DirectionCampaignPolicy::OracleUpperBound,
    ];

    const LANDSCAPES: [DirectionLandscapeFamily; 4] = [
        DirectionLandscapeFamily::DelayedBreakthrough,
        DirectionLandscapeFamily::NoNewDirection,
        DirectionLandscapeFamily::NewOperatorBreakthrough,
        DirectionLandscapeFamily::RankReversal,
    ];

    fn synthetic_action(slot: usize, arm_id: &str, score: f64) -> DirectionCampaignAction {
        DirectionCampaignAction {
            slot,
            arm_id: arm_id.to_owned(),
            is_new_proposal: false,
            observed_score: Some(score),
            proposal_disposition: None,
            cost_units: 1,
        }
    }

    #[test]
    fn discounted_ucb_statistics_recover_after_a_rank_reversal() {
        // A led early, but its recent performance collapsed. B has only a recent
        // observation. The exponential recency weighting must reflect that change.
        let actions = vec![
            synthetic_action(1, "A", 0.95),
            synthetic_action(2, "B", 0.80),
            synthetic_action(3, "A", 0.10),
            synthetic_action(4, "A", 0.10),
            synthetic_action(5, "A", 0.10),
        ];
        let (a_reward, a_count) = discounted_arm_statistics("A", &actions, DISCOUNTED_UCB_GAMMA);
        let (b_reward, b_count) = discounted_arm_statistics("B", &actions, DISCOUNTED_UCB_GAMMA);
        let a_mean = (a_reward + 0.5) / (a_count + 1.0);
        let b_mean = (b_reward + 0.5) / (b_count + 1.0);
        assert!(b_mean > a_mean, "recent arm should overtake the stale early leader");
    }

    #[test]
    fn gaussian_thompson_posterior_moves_toward_evidence_and_contracts_variance() {
        let prior = gaussian_thompson_posterior("A", &[]);
        let observed = gaussian_thompson_posterior(
            "A",
            &[synthetic_action(1, "A", 0.9), synthetic_action(2, "A", 0.8)],
        );
        assert!((prior.0 - 0.5).abs() < 1e-12);
        assert!((prior.1 - GAUSSIAN_THOMPSON_PRIOR_VARIANCE).abs() < 1e-12);
        assert!(observed.0 > prior.0);
        assert!(observed.1 < prior.1);
    }

    #[test]
    fn new_arm_reward_uses_the_same_frozen_admission_credit_as_global_ucb() {
        let admitted = DirectionCampaignAction {
            slot: 1,
            arm_id: "NEW".to_owned(),
            is_new_proposal: true,
            observed_score: None,
            proposal_disposition: Some(NewDirectionDisposition::Admitted),
            cost_units: 1,
        };
        let duplicate = DirectionCampaignAction {
            proposal_disposition: Some(NewDirectionDisposition::Duplicate),
            ..admitted.clone()
        };
        assert!((action_reward(&admitted) - NEW_DIRECTION_CREDIT).abs() < f64::EPSILON);
        assert_eq!(action_reward(&duplicate), 0.0);
    }

    #[test]
    fn adaptive_bandit_baselines_do_not_create_novelty_on_the_no_new_null() {
        for policy in [
            DirectionCampaignPolicy::DiscountedUcb,
            DirectionCampaignPolicy::GaussianThompsonSampling,
        ] {
            let run = run_direction_campaign(
                DirectionLandscapeFamily::NoNewDirection,
                policy,
                96,
                0x5eed,
            )
            .unwrap();
            assert_eq!(run.admitted_new_directions, 0);
            assert!(run.final_best_score < TARGET_SCORE);
            assert!(run.actions.iter().filter(|action| action.is_new_proposal).all(
                |action| action.proposal_disposition == Some(NewDirectionDisposition::Rejected)
            ));
        }
    }

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
            DirectionCampaignPolicy::GaussianThompsonSampling,
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
}
