//! VIS-DVPC-050: sequential logging and target-value drift falsification harness.
//!
//! This is a benchmark substrate, NOT an implementation of SeqLS / adjusted SeqLS.
//! Each arm runs an adaptive deployment loop. The incumbent chosen from prior logs
//! determines the next logger; every row retains the exact logger propensity.
//!
//! Compile and test without Cargo:
//!   rustc --edition=2024 -O --test tools/vision-dvpc-050-sequential-logging-ope.rs -o /tmp/vis-dvpc-050-tests
//!   /tmp/vis-dvpc-050-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-050-sequential-logging-ope.rs -o /tmp/vis-dvpc-050
//!   /tmp/vis-dvpc-050 --smoke
//!
//! The cumulative and recent-window estimates use event-specific propensities.
//! The stale-propensity arm is an intentional bug negative control, never a
//! competitive method. Structural support gaps fail closed before selection.
#![deny(unsafe_code)]

use std::env;

const ROUNDS: usize = 8;
const RECENT_WINDOW: usize = 2;
const POLICY_COUNTS: [usize; 3] = [4, 8, 32];
const FULL_REPLICATES: usize = 128;
const FULL_BATCH_SIZE: usize = 128;
const SMOKE_REPLICATES: usize = 4;
const SMOKE_BATCH_SIZE: usize = 8;
const EPS: f64 = 1e-12;

#[derive(Clone, Copy, Debug)]
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    Stationary,
    RewardDrift,
    WeakOverlap,
    ZeroSupport,
}
impl Scenario {
    const ALL: [Self; 4] = [
        Self::Stationary,
        Self::RewardDrift,
        Self::WeakOverlap,
        Self::ZeroSupport,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Stationary => "stationary_changing_logger",
            Self::RewardDrift => "reward_drift_reversal",
            Self::WeakOverlap => "weak_overlap",
            Self::ZeroSupport => "structural_zero_support",
        }
    }
    fn exploration(self) -> f64 {
        match self {
            Self::Stationary | Self::RewardDrift => 0.10,
            Self::WeakOverlap => 0.02,
            Self::ZeroSupport => 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Policy {
    id: usize,
    /// Probability of action 1 in each of the two contexts.
    p1: [f64; 2],
}

fn policies(m: usize) -> Vec<Policy> {
    assert!(m > 1);
    (0..m)
        .map(|j| {
            let first = 0.01 + 0.98 * j as f64 / (m - 1) as f64;
            // 7 is coprime to each registered portfolio size (4, 8, 32).
            let permuted = (7 * j) % m;
            let second = 0.01 + 0.98 * permuted as f64 / (m - 1) as f64;
            Policy {
                id: j,
                p1: [first, second],
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct Observation {
    context: usize,
    action: usize,
    reward: f64,
    /// Actual logging probability for action 1 in each context at this event.
    behavior_p1: [f64; 2],
    round: usize,
    behavior_policy_id: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EstimatorMethod {
    CumulativeRowWise,
    RecentWindowRowWise,
    StalePropensityNegativeControl,
}
impl EstimatorMethod {
    const ALL: [Self; 3] = [
        Self::CumulativeRowWise,
        Self::RecentWindowRowWise,
        Self::StalePropensityNegativeControl,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::CumulativeRowWise => "cumulative_rowwise_propensity",
            Self::RecentWindowRowWise => "recent2_rowwise_propensity",
            Self::StalePropensityNegativeControl => "stale_propensity_negative_control",
        }
    }
    fn start_round(self, current_round: usize) -> usize {
        match self {
            Self::RecentWindowRowWise => current_round.saturating_sub(RECENT_WINDOW - 1),
            Self::CumulativeRowWise | Self::StalePropensityNegativeControl => 0,
        }
    }
    fn is_negative_control(self) -> bool {
        matches!(self, Self::StalePropensityNegativeControl)
    }
}

/// The logger follows the previous round's selected policy, except for the
/// deliberate zero-support control. Positive exploration mixes the incumbent
/// with uniform random actions and is recorded exactly per event.
fn logger_probabilities(scenario: Scenario, incumbent: Policy) -> [f64; 2] {
    if scenario == Scenario::ZeroSupport {
        // Context 0 never logs action 1; context 1 never logs action 0.
        return [0.0, 1.0];
    }
    let epsilon = scenario.exploration();
    [
        epsilon * 0.5 + (1.0 - epsilon) * incumbent.p1[0],
        epsilon * 0.5 + (1.0 - epsilon) * incumbent.p1[1],
    ]
}

fn initial_reward_mean(context: usize, action: usize) -> f64 {
    match (context, action) {
        (0, 0) => 0.10,
        (0, 1) => 0.90,
        (1, 0) => 0.95,
        (1, 1) => 0.15,
        _ => unreachable!(),
    }
}
fn final_reward_mean(context: usize, action: usize) -> f64 {
    match (context, action) {
        (0, 0) => 0.85,
        (0, 1) => 0.20,
        (1, 0) => 0.20,
        (1, 1) => 0.85,
        _ => unreachable!(),
    }
}
fn reward_mean(scenario: Scenario, context: usize, action: usize, round: usize) -> f64 {
    let initial = initial_reward_mean(context, action);
    if scenario != Scenario::RewardDrift {
        return initial;
    }
    let fraction = round as f64 / (ROUNDS - 1) as f64;
    initial + fraction * (final_reward_mean(context, action) - initial)
}

/// Contexts have a known 1/2, 1/2 distribution.
fn true_policy_value(policy: Policy, scenario: Scenario, round: usize) -> f64 {
    let p0 = policy.p1[0];
    let p1 = policy.p1[1];
    let r00 = reward_mean(scenario, 0, 0, round);
    let r01 = reward_mean(scenario, 0, 1, round);
    let r10 = reward_mean(scenario, 1, 0, round);
    let r11 = reward_mean(scenario, 1, 1, round);
    0.5 * ((1.0 - p0) * r00 + p0 * r01 + (1.0 - p1) * r10 + p1 * r11)
}
fn pooled_truth(policy: Policy, scenario: Scenario, through_round: usize) -> f64 {
    (0..=through_round)
        .map(|r| true_policy_value(policy, scenario, r))
        .sum::<f64>()
        / (through_round + 1) as f64
}
fn recent_truth(policy: Policy, scenario: Scenario, through_round: usize) -> f64 {
    let start = through_round.saturating_sub(RECENT_WINDOW - 1);
    (start..=through_round)
        .map(|r| true_policy_value(policy, scenario, r))
        .sum::<f64>()
        / (through_round - start + 1) as f64
}
fn best_policy_index(policy_set: &[Policy], scenario: Scenario, round: usize) -> usize {
    let mut best = 0;
    for j in 1..policy_set.len() {
        if true_policy_value(policy_set[j], scenario, round)
            > true_policy_value(policy_set[best], scenario, round)
        {
            best = j;
        }
    }
    best
}
fn seed_for(scenario_id: usize, m: usize, replicate: usize) -> u64 {
    0x8b8b_8b8b_0500_0050
        ^ (scenario_id as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
        ^ (m as u64).wrapping_mul(0xA076_1D64_78BD_642F)
        ^ (replicate as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB)
}

/// Returns P_logger(action | context); the propensity in each row is data,
/// not inferred later from the current logger or the incumbent policy.
fn behavior_action_probability(obs: &Observation, action: usize, mu1: [f64; 2]) -> f64 {
    if action == 1 {
        mu1[obs.context]
    } else {
        1.0 - mu1[obs.context]
    }
}
fn target_action_probability(policy: Policy, obs: &Observation) -> f64 {
    if obs.action == 1 {
        policy.p1[obs.context]
    } else {
        1.0 - policy.p1[obs.context]
    }
}
fn importance_weight(policy: Policy, obs: &Observation, mu1: [f64; 2]) -> Option<f64> {
    let numerator = target_action_probability(policy, obs);
    let denominator = behavior_action_probability(obs, obs.action, mu1);
    if !denominator.is_finite() || denominator < 0.0 {
        return None;
    }
    if denominator <= EPS {
        return if numerator <= EPS { Some(0.0) } else { None };
    }
    let weight = numerator / denominator;
    if weight.is_finite() {
        Some(weight)
    } else {
        None
    }
}
fn score_contribution(policy: Policy, obs: &Observation, mu1: [f64; 2]) -> Option<f64> {
    Some(obs.reward * importance_weight(policy, obs, mu1)?)
}

/// A target policy is identifiable only if every action to which it assigns
/// positive mass has positive logging support in each context in the selected
/// data window. This is a support audit, not a claim that finite data is enough.
fn policy_identified(policy: Policy, logger_history: &[[f64; 2]], start_round: usize) -> bool {
    if logger_history.is_empty() || start_round >= logger_history.len() {
        return false;
    }
    let selected = &logger_history[start_round..];
    for context in 0..2 {
        let target_p1 = policy.p1[context];
        let support_action1 = selected.iter().any(|mu| mu[context] > EPS);
        let support_action0 = selected.iter().any(|mu| 1.0 - mu[context] > EPS);
        if target_p1 > EPS && !support_action1 {
            return false;
        }
        if 1.0 - target_p1 > EPS && !support_action0 {
            return false;
        }
    }
    true
}
fn portfolio_identified(
    policy_set: &[Policy],
    logger_history: &[[f64; 2]],
    method: EstimatorMethod,
    current_round: usize,
) -> bool {
    let start = method.start_round(current_round);
    policy_set
        .iter()
        .all(|policy| policy_identified(*policy, logger_history, start))
}

fn estimate_policy_value(
    policy: Policy,
    logs: &[Observation],
    first_mu1: [f64; 2],
    method: EstimatorMethod,
    current_round: usize,
) -> Option<f64> {
    let start_round = method.start_round(current_round);
    let mut total = 0.0;
    let mut count = 0usize;
    for obs in logs.iter().filter(|obs| obs.round >= start_round) {
        let mu1 = if method.is_negative_control() {
            first_mu1
        } else {
            obs.behavior_p1
        };
        total += score_contribution(policy, obs, mu1)?;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        let estimate = total / count as f64;
        if estimate.is_finite() {
            Some(estimate)
        } else {
            None
        }
    }
}
fn actual_weight_diagnostics(
    policy_set: &[Policy],
    logs: &[Observation],
    method: EstimatorMethod,
    current_round: usize,
    selected_policy: Policy,
) -> Option<(f64, f64)> {
    let start_round = method.start_round(current_round);
    let relevant: Vec<&Observation> = logs
        .iter()
        .filter(|obs| obs.round >= start_round)
        .collect();
    if relevant.is_empty() {
        return None;
    }
    let mut maximum = 0.0_f64;
    for policy in policy_set {
        for obs in &relevant {
            maximum = maximum.max(importance_weight(*policy, obs, obs.behavior_p1)?);
        }
    }
    let mut sum_w = 0.0;
    let mut sum_w2 = 0.0;
    for obs in &relevant {
        let weight = importance_weight(selected_policy, obs, obs.behavior_p1)?;
        sum_w += weight;
        sum_w2 += weight * weight;
    }
    if sum_w2 <= EPS {
        return Some((maximum, 0.0));
    }
    let ess = sum_w * sum_w / sum_w2;
    Some((maximum, (ess / relevant.len() as f64).clamp(0.0, 1.0)))
}

#[derive(Clone, Debug, Default)]
struct Aggregate {
    attempted: usize,
    selected: usize,
    top1: usize,
    abstained: usize,
    regret_sum: f64,
    current_abs_error_sum: f64,
    pooled_abs_error_sum: f64,
    recent_abs_error_sum: f64,
    max_weight_sum: f64,
    ess_fraction_sum: f64,
}
impl Aggregate {
    fn record_abstention(&mut self) {
        self.attempted += 1;
        self.abstained += 1;
    }
    fn record_selection(
        &mut self,
        top1: bool,
        regret: f64,
        current_abs_error: f64,
        pooled_abs_error: f64,
        recent_abs_error: f64,
        max_weight: f64,
        ess_fraction: f64,
    ) {
        self.attempted += 1;
        self.selected += 1;
        self.top1 += usize::from(top1);
        self.regret_sum += regret;
        self.current_abs_error_sum += current_abs_error;
        self.pooled_abs_error_sum += pooled_abs_error;
        self.recent_abs_error_sum += recent_abs_error;
        self.max_weight_sum += max_weight;
        self.ess_fraction_sum += ess_fraction;
    }
    fn field_mean(&self, sum: f64) -> String {
        if self.selected == 0 {
            "NA".into()
        } else {
            format!("{:.6}", sum / self.selected as f64)
        }
    }
    fn write_row(
        &self,
        scenario: Scenario,
        m: usize,
        method: EstimatorMethod,
        round: usize,
        reps: usize,
    ) {
        let selection_rate = self.selected as f64 / reps as f64;
        let abstention_rate = self.abstained as f64 / reps as f64;
        let top1 = if self.selected == 0 {
            "NA".into()
        } else {
            format!("{:.6}", self.top1 as f64 / self.selected as f64)
        };
        println!(
            "METRIC\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            scenario.name(),
            m,
            method.name(),
            round,
            reps,
            selection_rate,
            abstention_rate,
            top1,
            self.field_mean(self.regret_sum),
            self.field_mean(self.current_abs_error_sum),
            self.field_mean(self.pooled_abs_error_sum),
            self.field_mean(self.recent_abs_error_sum),
            self.field_mean(self.max_weight_sum),
            self.field_mean(self.ess_fraction_sum)
        );
    }
}

fn run_path(
    scenario: Scenario,
    scenario_id: usize,
    policy_set: &[Policy],
    method: EstimatorMethod,
    replicate: usize,
    batch_size: usize,
    aggregates: &mut [Aggregate],
) {
    let mut rng = SplitMix64::new(seed_for(scenario_id, policy_set.len(), replicate));
    let mut incumbent_id = 0usize;
    let mut logs: Vec<Observation> = Vec::with_capacity(ROUNDS * batch_size);
    let mut logger_history: Vec<[f64; 2]> = Vec::with_capacity(ROUNDS);
    let mut first_mu1: Option<[f64; 2]> = None;

    for round in 0..ROUNDS {
        let incumbent = policy_set[incumbent_id];
        let mu1 = logger_probabilities(scenario, incumbent);
        if first_mu1.is_none() {
            first_mu1 = Some(mu1);
        }
        logger_history.push(mu1);
        for _ in 0..batch_size {
            let context = usize::from(rng.unit() >= 0.5);
            let action = usize::from(rng.unit() < mu1[context]);
            let reward = if rng.unit() < reward_mean(scenario, context, action, round) {
                1.0
            } else {
                0.0
            };
            logs.push(Observation {
                context,
                action,
                reward,
                behavior_p1: mu1,
                round,
                behavior_policy_id: incumbent_id,
            });
        }
        let initial_mu1 = first_mu1.expect("first logger was recorded");
        let aggregate = &mut aggregates[round];
        if !portfolio_identified(policy_set, &logger_history, method, round) {
            aggregate.record_abstention();
            // Do not turn missing support into a numeric estimate or a new policy.
            continue;
        }
        let estimates: Option<Vec<f64>> = policy_set
            .iter()
            .map(|p| estimate_policy_value(*p, &logs, initial_mu1, method, round))
            .collect();
        let Some(estimates) = estimates else {
            aggregate.record_abstention();
            continue;
        };
        let mut selected_id = 0usize;
        for j in 1..estimates.len() {
            if estimates[j] > estimates[selected_id] {
                selected_id = j;
            }
        }
        let selected = policy_set[selected_id];
        let estimated_selected = estimates[selected_id];
        let current_value = true_policy_value(selected, scenario, round);
        let historical_value = pooled_truth(selected, scenario, round);
        let recent_value = recent_truth(selected, scenario, round);
        let oracle = policy_set
            .iter()
            .map(|p| true_policy_value(*p, scenario, round))
            .fold(f64::NEG_INFINITY, f64::max);
        let regret = (oracle - current_value).max(0.0);
        let top1 = regret <= 1e-12;
        let Some((max_weight, ess_fraction)) =
            actual_weight_diagnostics(policy_set, &logs, method, round, selected)
        else {
            aggregate.record_abstention();
            continue;
        };
        aggregate.record_selection(
            top1,
            regret,
            (estimated_selected - current_value).abs(),
            (estimated_selected - historical_value).abs(),
            (estimated_selected - recent_value).abs(),
            max_weight,
            ess_fraction,
        );
        incumbent_id = selected_id;
    }
}

fn run_benchmark(replicates: usize, batch_size: usize) {
    println!("BENCHMARK_CONFIG\tid\tVIS-DVPC-050");
    println!("BENCHMARK_CONFIG\treplicates\t{replicates}");
    println!("BENCHMARK_CONFIG\tbatch_size\t{batch_size}");
    println!("BENCHMARK_CONFIG\trounds\t{ROUNDS}");
    println!("BENCHMARK_CONFIG\trecent_window_rounds\t{RECENT_WINDOW}");
    println!("BENCHMARK_CONFIG\tpolicy_counts\t4,8,32");
    println!("BENCHMARK_CONFIG\tmethods\tcumulative_rowwise,recent2_rowwise,stale_propensity_negative_control");
    println!("BENCHMARK_CONFIG\tstale_propensity_method_role\tnegative_control_only");
    println!("BENCHMARK_CONFIG\tconfidence_claims\tnone; estimation/regret harness only");
    println!(
        "METRIC_HEADER\tscenario\tM\tmethod\tround\treplicates\tselection_rate\tabstention_rate\ttop1_rate_given_selection\tmean_regret\tmean_abs_error_current\tmean_abs_error_pooled\tmean_abs_error_recent\tmean_max_importance_weight\tmean_ess_fraction"
    );
    for (scenario_id, scenario) in Scenario::ALL.iter().copied().enumerate() {
        for m in POLICY_COUNTS {
            let policy_set = policies(m);
            for method in EstimatorMethod::ALL {
                let mut aggregates = vec![Aggregate::default(); ROUNDS];
                for replicate in 0..replicates {
                    run_path(
                        scenario,
                        scenario_id,
                        &policy_set,
                        method,
                        replicate,
                        batch_size,
                        &mut aggregates,
                    );
                }
                for (round, aggregate) in aggregates.iter().enumerate() {
                    aggregate.write_row(scenario, m, method, round, replicates);
                }
            }
        }
    }
}

fn main() {
    let smoke = env::args().any(|arg| arg == "--smoke");
    let (replicates, batch_size) = if smoke {
        (SMOKE_REPLICATES, SMOKE_BATCH_SIZE)
    } else {
        (FULL_REPLICATES, FULL_BATCH_SIZE)
    };
    run_benchmark(replicates, batch_size);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_weight_uses_its_actual_logger_not_the_initial_logger() {
        let policy = Policy {
            id: 0,
            p1: [0.8, 0.4],
        };
        let obs = Observation {
            context: 0,
            action: 1,
            reward: 1.0,
            behavior_p1: [0.2, 0.7],
            round: 1,
            behavior_policy_id: 2,
        };
        let rowwise = importance_weight(policy, &obs, obs.behavior_p1).unwrap();
        let stale = importance_weight(policy, &obs, [0.8, 0.8]).unwrap();
        assert!((rowwise - 4.0).abs() < EPS);
        assert!((stale - 1.0).abs() < EPS);
        assert_ne!(rowwise, stale);
    }

    #[test]
    fn stationary_expected_ips_matches_known_policy_value() {
        let policy = Policy {
            id: 1,
            p1: [0.8, 0.3],
        };
        let mu1 = [0.2, 0.7];
        let rewards = [[0.1, 0.9], [0.95, 0.15]];
        for context in 0..2 {
            let mu = mu1[context];
            let expected_ips = mu * (policy.p1[context] / mu) * rewards[context][1]
                + (1.0 - mu)
                    * ((1.0 - policy.p1[context]) / (1.0 - mu))
                    * rewards[context][0];
            let target_value =
                policy.p1[context] * rewards[context][1]
                    + (1.0 - policy.p1[context]) * rewards[context][0];
            assert!((expected_ips - target_value).abs() < EPS);
        }
    }

    #[test]
    fn structural_zero_support_fails_closed() {
        let policy_set = policies(4);
        let logger_history = vec![[0.0, 1.0]; ROUNDS];
        for method in EstimatorMethod::ALL {
            assert!(!portfolio_identified(
                &policy_set,
                &logger_history,
                method,
                ROUNDS - 1
            ));
        }
    }

    #[test]
    fn reward_drift_reverses_the_oracle_policy() {
        let policy_set = policies(4);
        let first = best_policy_index(&policy_set, Scenario::RewardDrift, 0);
        let last = best_policy_index(&policy_set, Scenario::RewardDrift, ROUNDS - 1);
        assert_ne!(first, last);
    }

    #[test]
    fn positive_exploration_preserves_all_action_support() {
        for scenario in [Scenario::Stationary, Scenario::RewardDrift, Scenario::WeakOverlap] {
            let mu = logger_probabilities(scenario, policies(4)[0]);
            assert!(mu.iter().all(|p| *p > 0.0 && *p < 1.0));
        }
        let weak_mu = logger_probabilities(Scenario::WeakOverlap, policies(4)[0]);
        assert!(weak_mu.iter().all(|p| *p < 0.03));
    }

    #[test]
    fn changing_incumbent_changes_the_next_logger() {
        let policy_set = policies(8);
        let a = logger_probabilities(Scenario::Stationary, policy_set[0]);
        let b = logger_probabilities(Scenario::Stationary, policy_set[7]);
        assert_ne!(a, b);
    }

    #[test]
    fn random_stream_is_reproducible() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn cumulative_estimator_uses_each_rows_recorded_propensity() {
        let policy = Policy {
            id: 3,
            p1: [0.8, 0.4],
        };
        let logs = vec![
            Observation {
                context: 0,
                action: 1,
                reward: 1.0,
                behavior_p1: [0.2, 0.7],
                round: 0,
                behavior_policy_id: 0,
            },
            Observation {
                context: 0,
                action: 0,
                reward: 1.0,
                behavior_p1: [0.8, 0.7],
                round: 1,
                behavior_policy_id: 2,
            },
        ];
        let rowwise = estimate_policy_value(
            policy,
            &logs,
            logs[0].behavior_p1,
            EstimatorMethod::CumulativeRowWise,
            1,
        )
        .unwrap();
        let stale = estimate_policy_value(
            policy,
            &logs,
            logs[0].behavior_p1,
            EstimatorMethod::StalePropensityNegativeControl,
            1,
        )
        .unwrap();
        assert!((rowwise - 2.5).abs() < EPS, "rowwise={rowwise}");
        assert!((stale - 1.0).abs() < EPS, "stale={stale}");
        assert!((rowwise - stale).abs() > 1.0);
    }

    #[test]
    fn pooled_and_current_estimands_are_distinct_under_reward_drift() {
        let policy = Policy {
            id: 0,
            p1: [0.1, 0.9],
        };
        let current = true_policy_value(policy, Scenario::RewardDrift, ROUNDS - 1);
        let pooled = pooled_truth(policy, Scenario::RewardDrift, ROUNDS - 1);
        let recent = recent_truth(policy, Scenario::RewardDrift, ROUNDS - 1);
        assert!((current - pooled).abs() > 0.05);
        assert!((current - recent).abs() > EPS);
    }

    #[test]
    fn row_records_keep_version_and_logger_probability() {
        let obs = Observation {
            context: 1,
            action: 0,
            reward: 1.0,
            behavior_p1: [0.2, 0.7],
            round: 3,
            behavior_policy_id: 5,
        };
        assert_eq!(obs.behavior_policy_id, 5);
        assert_eq!(obs.behavior_p1[obs.context], 0.7);
        assert_eq!(obs.round, 3);
    }
}