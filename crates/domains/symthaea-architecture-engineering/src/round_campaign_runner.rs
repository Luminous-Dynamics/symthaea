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

/// Paired comparison against one baseline on an identical landscape/seed/round matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FourSeatPairedComparison {
    pub landscape: DirectionLandscapeFamily,
    pub candidate_policy: RoundSchedulerPolicy,
    pub baseline_policy: RoundSchedulerPolicy,
    pub round_count: usize,
    pub pair_count: usize,
    /// Positive means the candidate achieved a higher final held-out score.
    pub mean_final_score_delta: f64,
    pub final_score_delta_sample_sd: f64,
    /// Positive means the candidate reduced simple regret.
    pub mean_simple_regret_reduction: f64,
    pub candidate_win_rate: f64,
    pub tie_rate: f64,
}

/// Versioned manifest and complete run set for one paired benchmark matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FourSeatCampaignMatrix {
    pub schema: String,
    pub version: String,
    pub landscapes: Vec<DirectionLandscapeFamily>,
    pub policies: Vec<RoundSchedulerPolicy>,
    pub round_count: usize,
    pub seeds: Vec<u64>,
    pub runs: Vec<FourSeatCampaignRun>,
    pub matrix_sha256: String,
}

impl FourSeatCampaignMatrix {
    fn compute_digest(&self) -> Result<String, String> {
        let mut unsigned = self.clone();
        unsigned.matrix_sha256.clear();
        let bytes = serde_json::to_vec(&unsigned).map_err(|error| error.to_string())?;
        Ok(digest_bytes(
            b"symthaea.ar5b6.campaign-matrix.v0.1\0",
            &bytes,
        ))
    }

    /// Require the exact Cartesian product recorded by the manifest; no missing run is optional.
    pub fn validate_integrity(&self) -> Result<(), String> {
        if self.schema != RUN_SCHEMA || self.version != RUN_VERSION {
            return Err("campaign matrix schema/version mismatch".to_owned());
        }
        if self.round_count == 0
            || self.landscapes.is_empty()
            || self.policies.is_empty()
            || self.seeds.is_empty()
        {
            return Err("campaign matrix dimensions must be non-empty and positive".to_owned());
        }
        if self.landscapes.iter().copied().collect::<BTreeSet<_>>().len()
            != self.landscapes.len()
            || self.policies.iter().copied().collect::<BTreeSet<_>>().len()
                != self.policies.len()
            || self.seeds.iter().copied().collect::<BTreeSet<_>>().len() != self.seeds.len()
        {
            return Err("campaign matrix manifest contains duplicate dimensions".to_owned());
        }
        let expected_count = self
            .landscapes
            .len()
            .checked_mul(self.policies.len())
            .and_then(|value| value.checked_mul(self.seeds.len()))
            .ok_or_else(|| "campaign matrix expected-run count overflow".to_owned())?;
        if self.runs.len() != expected_count {
            return Err(format!(
                "campaign matrix has {} runs, manifest requires {expected_count}",
                self.runs.len()
            ));
        }

        let mut expected = BTreeSet::new();
        for landscape in &self.landscapes {
            for policy in &self.policies {
                for seed in &self.seeds {
                    expected.insert((*landscape, *policy, *seed));
                }
            }
        }
        let mut observed = BTreeSet::new();
        for run in &self.runs {
            if run.round_count != self.round_count
                || run.action_count != self.round_count * SEATS_PER_ROUND
            {
                return Err("run budget differs from campaign matrix manifest".to_owned());
            }
            let key = (run.landscape, run.policy, run.seed);
            if !expected.contains(&key) || !observed.insert(key) {
                return Err("unexpected or duplicate campaign run in matrix".to_owned());
            }
        }
        if observed != expected {
            return Err("campaign matrix is missing one or more required landscape/policy/seed runs".to_owned());
        }
        if self.compute_digest()? != self.matrix_sha256 {
            return Err("campaign matrix digest does not match its manifest and runs".to_owned());
        }
        Ok(())
    }
}

/// Run every requested policy on the same deterministic landscape/seed matrix.
pub fn run_four_seat_campaign_matrix(
    landscapes: &[DirectionLandscapeFamily],
    policies: &[RoundSchedulerPolicy],
    round_count: usize,
    seeds: &[u64],
) -> Result<FourSeatCampaignMatrix, String> {
    if landscapes.is_empty() || policies.is_empty() || seeds.is_empty() {
        return Err("landscapes, policies and seeds must all be non-empty".to_owned());
    }
    if landscapes.iter().copied().collect::<BTreeSet<_>>().len() != landscapes.len()
        || policies.iter().copied().collect::<BTreeSet<_>>().len() != policies.len()
        || seeds.iter().copied().collect::<BTreeSet<_>>().len() != seeds.len()
    {
        return Err("landscape, policy and seed inputs must not contain duplicates".to_owned());
    }
    if round_count == 0 {
        return Err("round_count must be positive".to_owned());
    }
    let capacity = landscapes
        .len()
        .checked_mul(policies.len())
        .and_then(|value| value.checked_mul(seeds.len()))
        .ok_or_else(|| "campaign matrix size overflow".to_owned())?;
    let mut runs = Vec::with_capacity(capacity);
    for landscape in landscapes {
        for policy in policies {
            for seed in seeds {
                runs.push(run_four_seat_campaign(
                    *landscape,
                    *policy,
                    round_count,
                    *seed,
                )?);
            }
        }
    }
    let mut matrix = FourSeatCampaignMatrix {
        schema: RUN_SCHEMA.to_owned(),
        version: RUN_VERSION.to_owned(),
        landscapes: landscapes.to_vec(),
        policies: policies.to_vec(),
        round_count,
        seeds: seeds.to_vec(),
        runs,
        matrix_sha256: String::new(),
    };
    matrix.matrix_sha256 = matrix.compute_digest()?;
    matrix.validate_integrity()?;
    Ok(matrix)
}

/// Compare one candidate policy against every manifest-declared baseline using exact paired
/// landscape/seed/round sets. The manifest is checked before comparisons are calculated.
pub fn compare_four_seat_campaign_runs(
    matrix: &FourSeatCampaignMatrix,
    candidate_policy: RoundSchedulerPolicy,
) -> Result<Vec<FourSeatPairedComparison>, String> {
    matrix.validate_integrity()?;
    if !matrix.policies.contains(&candidate_policy) {
        return Err("candidate policy is absent from the run matrix".to_owned());
    }

    let mut comparisons = Vec::new();
    for landscape in &matrix.landscapes {
        let candidate_runs: BTreeMap<u64, &FourSeatCampaignRun> = matrix
            .runs
            .iter()
            .filter(|run| run.landscape == *landscape && run.policy == candidate_policy)
            .map(|run| (run.seed, run))
            .collect();
        if candidate_runs.len() != matrix.seeds.len() {
            return Err(format!("candidate policy has incomplete runs for {landscape:?}"));
        }
        for baseline_policy in matrix.policies.iter().copied() {
            if baseline_policy == candidate_policy {
                continue;
            }
            let baseline_runs: BTreeMap<u64, &FourSeatCampaignRun> = matrix
                .runs
                .iter()
                .filter(|run| run.landscape == *landscape && run.policy == baseline_policy)
                .map(|run| (run.seed, run))
                .collect();
            if baseline_runs.len() != matrix.seeds.len()
                || baseline_runs.keys().copied().collect::<BTreeSet<_>>()
                    != candidate_runs.keys().copied().collect::<BTreeSet<_>>()
            {
                return Err(format!(
                    "unpaired seed sets for {landscape:?}: {candidate_policy:?} vs {baseline_policy:?}"
                ));
            }

            let mut score_deltas = Vec::with_capacity(matrix.seeds.len());
            let mut regret_reductions = Vec::with_capacity(matrix.seeds.len());
            let mut wins = 0usize;
            let mut ties = 0usize;
            for seed in &matrix.seeds {
                let candidate = candidate_runs
                    .get(seed)
                    .ok_or_else(|| format!("missing candidate seed {seed}"))?;
                let baseline = baseline_runs
                    .get(seed)
                    .ok_or_else(|| format!("missing baseline seed {seed}"))?;
                let delta = candidate.final_best_score - baseline.final_best_score;
                score_deltas.push(delta);
                regret_reductions.push(baseline.simple_regret - candidate.simple_regret);
                if delta > f64::EPSILON {
                    wins += 1;
                } else if delta.abs() <= f64::EPSILON {
                    ties += 1;
                }
            }
            let pair_count = matrix.seeds.len();
            let mean_final_score_delta =
                score_deltas.iter().sum::<f64>() / pair_count as f64;
            let final_score_delta_sample_sd = if pair_count > 1 {
                (score_deltas
                    .iter()
                    .map(|delta| (delta - mean_final_score_delta).powi(2))
                    .sum::<f64>()
                    / (pair_count - 1) as f64)
                    .sqrt()
            } else {
                0.0
            };
            comparisons.push(FourSeatPairedComparison {
                landscape: *landscape,
                candidate_policy,
                baseline_policy,
                round_count: matrix.round_count,
                pair_count,
                mean_final_score_delta,
                final_score_delta_sample_sd,
                mean_simple_regret_reduction: regret_reductions.iter().sum::<f64>()
                    / pair_count as f64,
                candidate_win_rate: wins as f64 / pair_count as f64,
                tie_rate: ties as f64 / pair_count as f64,
            });
        }
    }
    Ok(comparisons)
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
            return Ok((0..DIRECTIONS.len())
                .map(|offset| DIRECTIONS[(round_index + offset) % DIRECTIONS.len()].to_owned())
                .collect());
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
                            .map(|(_, direction_id)| direction_id.to_ascii_uppercase())
                            .unwrap_or_default(),
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
    fn paired_matrix_uses_identical_seed_sets_and_rejects_missing_pairs() {
        let seeds = [11, 22, 33];
        let runs = run_four_seat_campaign_matrix(
            &[DirectionLandscapeFamily::DelayedBreakthrough],
            &POLICIES,
            4,
            &seeds,
        )
        .unwrap();
        let comparisons = compare_four_seat_campaign_runs(
            &runs,
            RoundSchedulerPolicy::DirectionBoardFixedSet,
        )
        .unwrap();
        assert_eq!(comparisons.len(), POLICIES.len() - 1);
        assert!(comparisons.iter().all(|item| item.pair_count == seeds.len()));

        assert!(compare_four_seat_campaign_runs(
            &runs[..runs.len() - 1],
            RoundSchedulerPolicy::DirectionBoardFixedSet,
        )
        .is_err());
    }

    #[test]
    fn bootstrap_work_item_digests_are_policy_independent() {
        let board = run_four_seat_campaign(
            DirectionLandscapeFamily::DelayedBreakthrough,
            RoundSchedulerPolicy::DirectionBoardFixedSet,
            2,
            0xabc,
        )
        .unwrap();
        let coral = run_four_seat_campaign(
            DirectionLandscapeFamily::DelayedBreakthrough,
            RoundSchedulerPolicy::CoralDirectionOnly,
            2,
            0xabc,
        )
        .unwrap();
        let board_seats = &board.executions[0].plan.seats;
        let coral_seats = &coral.executions[0].plan.seats;
        assert_eq!(board_seats.len(), coral_seats.len());
        for (left, right) in board_seats.iter().zip(coral_seats) {
            assert_eq!(left.action, right.action);
            assert_eq!(left.action_id, right.action_id);
            assert_eq!(left.work_item_sha256, right.work_item_sha256);
        }
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
