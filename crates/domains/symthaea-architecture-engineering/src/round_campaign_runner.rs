//! Four-seat fixed-direction campaign driver using the AR-5B4 round contract.
//!
//! Track A fixes the direction set before search begins. It compares allocation mechanisms,
//! not novelty discovery. DirectionBoard and Coral are explicitly directions-only ablations;
//! their NEW arms are not silently mixed into this mechanism-matched track.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::coral_baseline::{CoralBaselineConfig, CoralDirectionBoard};
use super::direction_board::{
    AttemptDisposition, DirectionAttempt, DirectionBoard, DirectionBoardConfig,
    DirectionDescriptor,
};
use super::direction_board_benchmark::{
    self, DirectionCampaignAction, DirectionLandscapeFamily,
};
use super::round_campaign::{
    ActionCostReceipt, ActionOutcome, ActionResultReceipt, ClaimCapProfile, CostDimension,
    EvaluationDisposition, RoundAction, RoundCloseReceipt, RoundLedger, RoundPlan, RoundSeat,
};

const RUN_SCHEMA: &str = "symthaea.architecture-engineering.four-seat-campaign";
const RUN_VERSION: &str = "0.1.0";
const SEATS_PER_ROUND: usize = 4;
const CLAIM_CAP_PER_DIRECTION: usize = 2;
const PROFILE_SHA256: &str =
    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const METRIC_ID: &str = "held_out_capability";
const EXPLORATION_COEFFICIENT: f64 = 0.55;
const COST_PROFILE_DOMAIN: &[u8] = b"symthaea.ar5b6.synthetic-cost-profile.v0.1\0";

const DIRECTIONS: [&str; 4] = ["A", "B", "C", "D"];
const REQUIRED_COSTS: [CostDimension; 3] = [
    CostDimension::WallTimeMillis,
    CostDimension::CpuTimeMillis,
    CostDimension::CandidateExecutions,
];

/// Policies compared in the common four-seat fixed-direction execution model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoundSchedulerPolicy {
    /// One evaluation per pre-registered direction in every round.
    FixedRoundRobin,
    /// Seeded choice of two directions, two seats each.
    Random,
    /// Allocate to the two directions with the best observed scores.
    ScoreOnly,
    /// Global mean reward plus UCB exploration.
    GlobalUcb,
    /// Exponential-discount UCB using the same reward history as the other policies.
    DiscountedUcb,
    /// Gaussian Thompson sampling using a frozen Normal-Normal reward model.
    GaussianThompsonSampling,
    /// DirectionBoard local-record yield and UCB exploration, with NEW excluded.
    DirectionBoardFixedSet,
    /// Coral's published direction yield/potential equations, with NEW excluded.
    CoralDirectionOnly,
    /// Synthetic upper bound with access to the future payoff schedule.
    OracleUpperBound,
}

impl RoundSchedulerPolicy {
    fn id(self) -> &'static str {
        match self {
            Self::FixedRoundRobin => "fixed_round_robin",
            Self::Random => "random",
            Self::ScoreOnly => "score_only",
            Self::GlobalUcb => "global_ucb",
            Self::DiscountedUcb => "discounted_ucb",
            Self::GaussianThompsonSampling => "gaussian_thompson_sampling",
            Self::DirectionBoardFixedSet => "direction_board_fixed_set",
            Self::CoralDirectionOnly => "coral_direction_only",
            Self::OracleUpperBound => "oracle_upper_bound",
        }
    }
}

/// A plan and close receipt pair retained as the executable evidence for one round.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoundExecutionRecord {
    pub plan: RoundPlan,
    pub close: RoundCloseReceipt,
}

/// Compact result summary plus the complete plan/receipt history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FourSeatCampaignRun {
    pub schema: String,
    pub version: String,
    pub landscape: DirectionLandscapeFamily,
    pub policy: RoundSchedulerPolicy,
    pub seed: u64,
    pub round_count: usize,
    pub seats_per_round: usize,
    pub action_count: usize,
    pub total_charged_budget_units: u64,
    pub target_score: f64,
    pub actions_to_target: Option<usize>,
    pub final_best_score: f64,
    pub final_best_direction: String,
    pub oracle_best_score: f64,
    pub simple_regret: f64,
    pub ledger: RoundLedger,
    pub executions: Vec<RoundExecutionRecord>,
}

/// Run a fixed-set scheduler comparison under four synchronous seats per round.
///
/// Every policy receives the same initial bootstrap round (A, B, C, D). For later rounds,
/// assignments are computed once from the pre-round snapshot. Each selected direction gets
/// at most two seats. Results are generated independently from the frozen payoff schedule
/// and committed only after the whole round has a valid close receipt.
pub fn run_four_seat_campaign(
    landscape: DirectionLandscapeFamily,
    policy: RoundSchedulerPolicy,
    round_count: usize,
    seed: u64,
) -> Result<FourSeatCampaignRun, String> {
    if round_count == 0 {
        return Err("round_count must be positive".to_owned());
    }

    let campaign_id = format!("ar5b6-{}-{seed:016x}", policy.id());
    let cost_profile_sha256 = digest_bytes(
        COST_PROFILE_DOMAIN,
        b"unit-evaluation;wall=100ms;cpu=80ms;candidate-executions=1",
    );
    let required_cost_dimensions: BTreeSet<CostDimension> =
        REQUIRED_COSTS.into_iter().collect();

    let mut ledger =
        RoundLedger::new(&campaign_id).map_err(|error| error.to_string())?;
    let mut history: BTreeMap<String, Vec<f64>> = DIRECTIONS
        .iter()
        .map(|direction| ((*direction).to_owned(), Vec::new()))
        .collect();
    let mut action_history: Vec<DirectionCampaignAction> = Vec::new();
    let mut random_state = seed ^ 0x510e_527f_ade6_82d1;
    let mut actions_to_target = None;
    let target_score = 0.90;

    let board_config = DirectionBoardConfig {
        comparison_profile_sha256: PROFILE_SHA256.to_owned(),
        metric_id: METRIC_ID.to_owned(),
        exploration_coefficient: EXPLORATION_COEFFICIENT,
        new_direction_prior_yield: 0.20,
        new_direction_admission_credit: 0.20,
        minimum_initial_evaluations: 1,
    };
    let mut direction_board =
        DirectionBoard::new(board_config).map_err(|error| error.to_string())?;
    let mut direction_digests = BTreeMap::new();

    let mut coral_config = CoralBaselineConfig::default();
    coral_config.agent_count = SEATS_PER_ROUND;
    // Track A fixes the same cap for all policies. The paper-native first-round cap
    // is a separate Coral-system ablation, not applied only to this matched comparison.
    coral_config.max_agents_per_direction = CLAIM_CAP_PER_DIRECTION;
    coral_config.first_round_agents_per_direction = CLAIM_CAP_PER_DIRECTION;
    let mut coral_board =
        CoralDirectionBoard::new(coral_config).map_err(|error| error.to_string())?;

    for direction in DIRECTIONS {
        let descriptor = fixed_direction_descriptor(direction);
        let receipt = digest(&format!("fixed-direction-admission:{campaign_id}:{direction}"));
        let direction_digest = direction_board
            .register_verified_direction(descriptor, &receipt)
            .map_err(|error| error.to_string())?;
        direction_digests.insert(direction.to_owned(), direction_digest);
        coral_board
            .register_direction(direction)
            .map_err(|error| error.to_string())?;
    }

    let mut executions = Vec::with_capacity(round_count);

    for round_index in 0..round_count {
        // All policies make the first four assignments identically. Later rounds use
        // only evidence committed before this round began.
        let pre_round_snapshot_sha256 = snapshot_digest(&history)?;
        let assignments = if round_index == 0 {
            DIRECTIONS.iter().map(|value| (*value).to_owned()).collect()
        } else {
            allocate_round(
                landscape,
                policy,
                round_index,
                &history,
                &action_history,
                &direction_board,
                &coral_board,
                &mut random_state,
            )?
        };
        if assignments.len() != SEATS_PER_ROUND {
            return Err(format!(
                "policy {} generated {} seats, expected {}",
                policy.id(),
                assignments.len(),
                SEATS_PER_ROUND
            ));
        }

        let seats: Vec<RoundSeat> = assignments
            .iter()
            .enumerate()
            .map(|(seat_index, direction_id)| {
                let action_id = format!("r{round_index:04}-s{seat_index:02}");
                RoundSeat {
                    work_item_sha256: digest(&format!(
                        "candidate:{campaign_id}:{round_index}:{seat_index}:{direction_id}"
                    )),
                    action_id,
                    action: RoundAction::EvaluateDirection(direction_id.clone()),
                }
            })
            .collect();

        let plan = RoundPlan::new(
            &campaign_id,
            format!("round-{round_index:04}"),
            round_index as u64,
            policy.id(),
            seed,
            ledger.last_close_sha256.clone(),
            pre_round_snapshot_sha256.clone(),
            cost_profile_sha256.clone(),
            required_cost_dimensions.clone(),
            ClaimCapProfile {
                max_evaluations_per_direction_per_round: CLAIM_CAP_PER_DIRECTION,
                max_new_proposals_per_round: 0,
            },
            seats,
        )
        .map_err(|error| error.to_string())?;

        let pre_round_parent_scores: BTreeMap<String, Option<f64>> = history
            .iter()
            .map(|(direction, values)| {
                (
                    direction.clone(),
                    values.iter().copied().max_by(f64::total_cmp),
                )
            })
            .collect();
        let mut within_round_offsets: BTreeMap<String, usize> = BTreeMap::new();
        let mut results = Vec::with_capacity(SEATS_PER_ROUND);
        let mut costs = Vec::with_capacity(SEATS_PER_ROUND);

        for seat in &plan.seats {
            let RoundAction::EvaluateDirection(direction_id) = &seat.action else {
                return Err("Track A cannot contain direction-proposal actions".to_owned());
            };
            let offset = within_round_offsets.entry(direction_id.clone()).or_default();
            let evaluation_index = history[direction_id].len() + *offset;
            *offset += 1;
            let score = direction_board_benchmark::payoff(
                landscape,
                direction_id,
                evaluation_index,
            );
            let result_digest = digest(&format!(
                "result:{}:{direction_id}:{evaluation_index}:{:016x}",
                seat.action_id,
                score.to_bits()
            ));
            let verifier_receipt = digest(&format!(
                "synthetic-independent-verifier-receipt:{}:{result_digest}",
                seat.action_id
            ));
            results.push(ActionResultReceipt {
                round_id: plan.round_id.clone(),
                action_id: seat.action_id.clone(),
                observed_snapshot_sha256: pre_round_snapshot_sha256.clone(),
                result_digest_sha256: result_digest,
                verifier_receipt_sha256: verifier_receipt,
                outcome: ActionOutcome::Evaluation {
                    disposition: EvaluationDisposition::Verified,
                    score: Some(score),
                    resolved_direction_id: direction_id.clone(),
                },
            });
            costs.push(ActionCostReceipt {
                round_id: plan.round_id.clone(),
                action_id: seat.action_id.clone(),
                cost_profile_sha256: cost_profile_sha256.clone(),
                charged_budget_units: 1,
                resource_costs: [
                    (CostDimension::WallTimeMillis, 100),
                    (CostDimension::CpuTimeMillis, 80),
                    (CostDimension::CandidateExecutions, 1),
                ]
                .into_iter()
                .collect(),
            });
        }

        let close = RoundCloseReceipt {
            round_id: plan.round_id.clone(),
            plan_sha256: plan.plan_sha256.clone(),
            results,
            costs,
        };
        let summary = close
            .validate_against(&plan)
            .map_err(|error| error.to_string())?;
        if summary.action_count != SEATS_PER_ROUND
            || summary.charged_budget_units != SEATS_PER_ROUND as u64
        {
            return Err("round did not conserve the four-seat unit budget".to_owned());
        }
        ledger
            .append(&plan, &close)
            .map_err(|error| error.to_string())?;

        // Commit the complete result batch only after the round close validates.
        for result in &close.results {
            let ActionOutcome::Evaluation {
                disposition: EvaluationDisposition::Verified,
                score: Some(score),
                resolved_direction_id,
            } = &result.outcome
            else {
                return Err("synthetic Track A expected verified evaluation outcomes".to_owned());
            };
            let parent_score = pre_round_parent_scores[resolved_direction_id];
            let direction_digest = direction_digests
                .get(resolved_direction_id)
                .ok_or_else(|| format!("missing direction digest for {resolved_direction_id}"))?;
            let seat = plan
                .seats
                .iter()
                .find(|seat| seat.action_id == result.action_id)
                .ok_or_else(|| "result receipt has no corresponding planned seat".to_owned())?;
            let attempt = DirectionAttempt {
                attempt_id: result.action_id.clone(),
                direction_sha256: direction_digest.clone(),
                method_id: format!("round-{round_index}-{}", result.action_id),
                parent_candidate_sha256: digest(&format!(
                    "parent:{campaign_id}:{round_index}:{resolved_direction_id}"
                )),
                candidate_sha256: seat.work_item_sha256.clone(),
                semantic_delta_sha256: digest(&format!(
                    "semantic-delta:{campaign_id}:{round_index}:{}",
                    result.action_id
                )),
                comparison_profile_sha256: PROFILE_SHA256.to_owned(),
                metric_id: METRIC_ID.to_owned(),
                evaluation_cost_units: 1,
                disposition: AttemptDisposition::Verified,
                score: Some(*score),
                verifier_receipt_sha256: result.verifier_receipt_sha256.clone(),
            };
            direction_board
                .record_attempt(attempt)
                .map_err(|error| error.to_string())?;
            coral_board
                .record_evaluation(
                    &result.action_id,
                    resolved_direction_id,
                    Some(*score),
                    parent_score,
                    Vec::new(),
                    1,
                )
                .map_err(|error| error.to_string())?;
            history
                .get_mut(resolved_direction_id)
                .ok_or_else(|| format!("unknown direction {resolved_direction_id}"))?
                .push(*score);
            action_history.push(DirectionCampaignAction {
                slot: action_history.len() + 1,
                arm_id: resolved_direction_id.clone(),
                is_new_proposal: false,
                observed_score: Some(*score),
                proposal_disposition: None,
                cost_units: 1,
            });
            if actions_to_target.is_none() && *score >= target_score {
                actions_to_target = Some(action_history.len());
            }
        }

        direction_board
            .validate_integrity()
            .map_err(|error| error.to_string())?;
        coral_board
            .validate_integrity()
            .map_err(|error| error.to_string())?;
        ledger
            .validate_integrity()
            .map_err(|error| error.to_string())?;
        executions.push(RoundExecutionRecord { plan, close });
    }

    let (final_best_direction, final_best_score) = history
        .iter()
        .filter_map(|(direction, values)| {
            values
                .iter()
                .copied()
                .max_by(f64::total_cmp)
                .map(|score| (direction.clone(), score))
        })
        .max_by(|(left_id, left_score), (right_id, right_score)| {
            left_score
                .total_cmp(right_score)
                .then_with(|| right_id.cmp(left_id))
        })
        .ok_or_else(|| "campaign produced no evaluated scores".to_owned())?;
    let oracle_best_score = direction_board_benchmark::oracle_best_score(landscape);
    let action_count = round_count * SEATS_PER_ROUND;

    Ok(FourSeatCampaignRun {
        schema: RUN_SCHEMA.to_owned(),
        version: RUN_VERSION.to_owned(),
        landscape,
        policy,
        seed,
        round_count,
        seats_per_round: SEATS_PER_ROUND,
        action_count,
        total_charged_budget_units: ledger.total_charged_budget_units,
        target_score,
        actions_to_target,
        final_best_score,
        final_best_direction,
        oracle_best_score,
        simple_regret: (oracle_best_score - final_best_score).max(0.0),
        ledger,
        executions,
    })
}

fn allocate_round(
    landscape: DirectionLandscapeFamily,
    policy: RoundSchedulerPolicy,
    round_index: usize,
    history: &BTreeMap<String, Vec<f64>>,
    action_history: &[DirectionCampaignAction],
    direction_board: &DirectionBoard,
    coral_board: &CoralDirectionBoard,
    random_state: &mut u64,
) -> Result<Vec<String>, String> {
    let mut ranked: Vec<(String, f64)> = match policy {
        RoundSchedulerPolicy::FixedRoundRobin => {
            return Ok(DIRECTIONS.iter().map(|id| (*id).to_owned()).collect());
        }
        RoundSchedulerPolicy::Random => {
            let mut arms: Vec<String> = DIRECTIONS.iter().map(|id| (*id).to_owned()).collect();
            for index in (1..arms.len()).rev() {
                let selected = (direction_board_benchmark::next_random(random_state)
                    % (index as u64 + 1)) as usize;
                arms.swap(index, selected);
            }
            return Ok(vec![
                arms[0].clone(),
                arms[0].clone(),
                arms[1].clone(),
                arms[1].clone(),
            ]);
        }
        RoundSchedulerPolicy::ScoreOnly => DIRECTIONS
            .iter()
            .map(|id| {
                let best = history[*id]
                    .iter()
                    .copied()
                    .max_by(f64::total_cmp)
                    .unwrap_or(0.0);
                ((*id).to_owned(), best)
            })
            .collect(),
        RoundSchedulerPolicy::GlobalUcb => {
            let total_count: usize = history.values().map(Vec::len).sum();
            let log_total = (total_count as f64 + 2.0).ln();
            DIRECTIONS
                .iter()
                .map(|id| {
                    let values = &history[*id];
                    let mean = if values.is_empty() {
                        0.5
                    } else {
                        values.iter().sum::<f64>() / values.len() as f64
                    };
                    let bonus = direction_board_benchmark::GLOBAL_UCB_EXPLORATION
                        * (2.0 * log_total / (values.len() as f64 + 1.0)).sqrt();
                    ((*id).to_owned(), mean + bonus)
                })
                .collect()
        }
        RoundSchedulerPolicy::DiscountedUcb => DIRECTIONS
            .iter()
            .map(|id| {
                let (reward, count) = direction_board_benchmark::discounted_arm_statistics(
                    id,
                    action_history,
                    direction_board_benchmark::DISCOUNTED_UCB_GAMMA,
                );
                let mean = (reward + 0.5) / (count + 1.0);
                let bonus = direction_board_benchmark::GLOBAL_UCB_EXPLORATION
                    * (2.0 * ((action_history.len() as f64) + 2.0).ln() / (count + 1.0))
                        .sqrt();
                ((*id).to_owned(), mean + bonus)
            })
            .collect(),
        RoundSchedulerPolicy::GaussianThompsonSampling => DIRECTIONS
            .iter()
            .map(|id| {
                let (mean, variance) =
                    direction_board_benchmark::gaussian_thompson_posterior(id, action_history);
                let sample = mean
                    + variance.sqrt()
                        * direction_board_benchmark::sample_standard_normal(random_state);
                ((*id).to_owned(), sample)
            })
            .collect(),
        RoundSchedulerPolicy::DirectionBoardFixedSet => {
            let total_count: usize = direction_board
                .direction_stats()
                .iter()
                .map(|stats| stats.attempted_evaluations as usize)
                .sum();
            let log_total = (total_count as f64 + 2.0).ln();
            direction_board
                .direction_stats()
                .into_iter()
                .map(|stats| {
                    let potential = stats.improvement_yield_per_cost
                        + direction_board.config().exploration_coefficient
                            * (log_total / (stats.attempted_evaluations as f64 + 1.0)).sqrt();
                    (
                        stats
                            .descriptor
                            .scope
                            .split_once(':')
                            .map(|(_, direction_id)| direction_id)
                            .unwrap_or("")
                            .to_owned(),
                        potential,
                    )
                })
                .collect(),
        }
        RoundSchedulerPolicy::CoralDirectionOnly => coral_board
            .direction_metrics()
            .map_err(|error| error.to_string())?
            .into_iter()
            .map(|metric| (metric.direction_id, metric.potential))
            .collect(),
        RoundSchedulerPolicy::OracleUpperBound => DIRECTIONS
            .iter()
            .map(|id| {
                (
                    (*id).to_owned(),
                    direction_board_benchmark::maximum_future_payoff(
                        landscape,
                        id,
                        history[*id].len(),
                    ),
                )
            })
            .collect(),
    };

    ranked.sort_by(|(left_id, left_score), (right_id, right_score)| {
        right_score
            .total_cmp(left_score)
            .then_with(|| left_id.cmp(right_id))
    });
    if ranked.len() < 2 {
        return Err("fixed-set allocator requires at least two directions".to_owned());
    }
    Ok(vec![
        ranked[0].0.clone(),
        ranked[0].0.clone(),
        ranked[1].0.clone(),
        ranked[1].0.clone(),
    ])
}

fn fixed_direction_descriptor(direction_id: &str) -> DirectionDescriptor {
    DirectionDescriptor {
        target_capability: "held-out capability".to_owned(),
        solution_object: format!("fixed direction {direction_id}"),
        construction_principle: "pre-registered mechanism family".to_owned(),
        scope: format!("fixed:{direction_id}"),
    }
}

fn snapshot_digest(history: &BTreeMap<String, Vec<f64>>) -> Result<String, String> {
    let bytes = serde_json::to_vec(history).map_err(|error| error.to_string())?;
    Ok(digest_bytes(b"symthaea.ar5b6.pre-round-snapshot.v0.1\0", &bytes))
}

fn digest(value: &str) -> String {
    digest_bytes(b"symthaea.ar5b6.evidence-reference.v0.1\0", value.as_bytes())
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICIES: [RoundSchedulerPolicy; 9] = [
        RoundSchedulerPolicy::FixedRoundRobin,
        RoundSchedulerPolicy::Random,
        RoundSchedulerPolicy::ScoreOnly,
        RoundSchedulerPolicy::GlobalUcb,
        RoundSchedulerPolicy::DiscountedUcb,
        RoundSchedulerPolicy::GaussianThompsonSampling,
        RoundSchedulerPolicy::DirectionBoardFixedSet,
        RoundSchedulerPolicy::CoralDirectionOnly,
        RoundSchedulerPolicy::OracleUpperBound,
    ];

    const LANDSCAPES: [DirectionLandscapeFamily; 4] = [
        DirectionLandscapeFamily::DelayedBreakthrough,
        DirectionLandscapeFamily::NoNewDirection,
        DirectionLandscapeFamily::NewOperatorBreakthrough,
        DirectionLandscapeFamily::RankReversal,
    ];

    #[test]
    fn every_policy_conserves_four_seats_and_uses_the_same_cap() {
        for landscape in LANDSCAPES {
            for policy in POLICIES {
                let run = run_four_seat_campaign(landscape, policy, 8, 17).unwrap();
                assert_eq!(run.action_count, 32);
                assert_eq!(run.total_charged_budget_units, 32);
                assert_eq!(run.executions.len(), 8);
                for execution in &run.executions {
                    assert_eq!(execution.plan.seats.len(), 4);
                    execution.plan.validate_integrity().unwrap();
                    execution.close.validate_against(&execution.plan).unwrap();
                    let mut counts = BTreeMap::<String, usize>::new();
                    for seat in &execution.plan.seats {
                        let RoundAction::EvaluateDirection(direction_id) = &seat.action else {
                            panic!("fixed-set campaign cannot propose a new direction");
                        };
                        *counts.entry(direction_id.clone()).or_default() += 1;
                    }
                    assert!(counts.values().all(|count| *count <= CLAIM_CAP_PER_DIRECTION));
                }
                run.ledger.validate_integrity().unwrap();
            }
        }
    }

    #[test]
    fn all_policies_receive_the_same_bootstrap_directions() {
        let runs: Vec<_> = POLICIES
            .iter()
            .map(|policy| {
                run_four_seat_campaign(
                    DirectionLandscapeFamily::DelayedBreakthrough,
                    *policy,
                    4,
                    91,
                )
                .unwrap()
            })
            .collect();
        let bootstrap: Vec<_> = runs[0].executions[0]
            .plan
            .seats
            .iter()
            .map(|seat| seat.action.clone())
            .collect();
        for run in &runs[1..] {
            let actions: Vec<_> = run.executions[0]
                .plan
                .seats
                .iter()
                .map(|seat| seat.action.clone())
                .collect();
            assert_eq!(actions, bootstrap);
        }
    }

    #[test]
    fn identical_seed_replays_the_complete_round_receipt_chain() {
        for policy in POLICIES {
            let first = run_four_seat_campaign(
                DirectionLandscapeFamily::RankReversal,
                policy,
                12,
                0x1234,
            )
            .unwrap();
            let second = run_four_seat_campaign(
                DirectionLandscapeFamily::RankReversal,
                policy,
                12,
                0x1234,
            )
            .unwrap();
            assert_eq!(first, second);
        }
    }

    #[test]
    fn every_seat_observes_the_frozen_pre_round_snapshot() {
        let run = run_four_seat_campaign(
            DirectionLandscapeFamily::NewOperatorBreakthrough,
            RoundSchedulerPolicy::CoralDirectionOnly,
            6,
            63,
        )
        .unwrap();
        for execution in &run.executions {
            assert!(execution.close.results.iter().all(|receipt| {
                receipt.observed_snapshot_sha256 == execution.plan.pre_round_snapshot_sha256
            }));
        }
    }

    #[test]
    fn fixed_set_track_never_turns_a_high_scoring_registered_operator_into_a_free_new_arm() {
        let run = run_four_seat_campaign(
            DirectionLandscapeFamily::NewOperatorBreakthrough,
            RoundSchedulerPolicy::DirectionBoardFixedSet,
            16,
            29,
        )
        .unwrap();
        assert!(run.executions.iter().all(|execution| {
            execution.plan.seats.iter().all(|seat| {
                matches!(seat.action, RoundAction::EvaluateDirection(_))
            })
        }));
        assert_eq!(run.oracle_best_score, 0.97);
        assert!(run.final_best_score <= run.oracle_best_score);
    }

    #[test]
    fn no_new_direction_null_remains_a_null_for_both_direction_ablations() {
        for policy in [
            RoundSchedulerPolicy::DirectionBoardFixedSet,
            RoundSchedulerPolicy::CoralDirectionOnly,
        ] {
            let run = run_four_seat_campaign(
                DirectionLandscapeFamily::NoNewDirection,
                policy,
                12,
                101,
            )
            .unwrap();
            assert!(run.final_best_score < 0.90);
            assert!(run.executions.iter().all(|execution| {
                execution.plan.seats.iter().all(|seat| {
                    matches!(seat.action, RoundAction::EvaluateDirection(_))
                })
            }));
        }
    }
}
