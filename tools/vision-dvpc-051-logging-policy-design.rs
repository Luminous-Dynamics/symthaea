//! VIS-DVPC-051: constrained logging-policy design for portfolio OPE.
//!
//! A known-answer, dependency-free benchmark for the reward/coverage/variance
//! tradeoff. The "oracle reward-aware" design uses synthetic reward means and is
//! a diagnostic upper-information arm, not deployable policy advice.
//!
//! Compile and run:
//!   rustc --edition=2024 -O --test tools/vision-dvpc-051-logging-policy-design.rs -o /tmp/vis-dvpc-051-tests
//!   /tmp/vis-dvpc-051-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-051-logging-policy-design.rs -o /tmp/vis-dvpc-051
//!   /tmp/vis-dvpc-051 --smoke
//!   /tmp/vis-dvpc-051
//!
//! References motivating the problem (not reproduced by this implementation):
//! Douglas, Persson & Provost (2026), "Logging Policy Design for Off-Policy Evaluation";
//! Haddouche & Sakhi (AISTATS 2026), sequential off-policy learning with LS/PAC-Bayes.
#![deny(unsafe_code)]

use std::env;

const ACTION_PROPENSITY_FLOOR: f64 = 0.02;
const GREEDY_EXPLORATION_MIX: f64 = 0.04;
const POLICY_COUNTS: [usize; 3] = [4, 8, 32];
const DEFAULT_GRID_INTERVALS: usize = 200;
const SMOKE_GRID_INTERVALS: usize = 20;
const EPS: f64 = 1e-12;
const REWARD_UPPER_ENVELOPE: [[f64; 2]; 2] = [[0.20, 0.95], [0.98, 0.20]];
const REWARD_ENVELOPE_SCALES: [f64; 5] = [1.0, 1.1, 1.25, 1.5, 2.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PortfolioRegime {
    Balanced,
    Action1Heavy,
    Action0Heavy,
}
impl PortfolioRegime {
    const ALL: [Self; 3] = [Self::Balanced, Self::Action1Heavy, Self::Action0Heavy];
    fn name(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Action1Heavy => "action1_heavy",
            Self::Action0Heavy => "action0_heavy",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoggerKind {
    Uniform,
    IncumbentMixture,
    PortfolioAverage,
    SecondMomentPseudoTargetProp6,
    MinimaxWeightSecondMoment,
    RewardEnvelopeMinimaxSecondMoment,
    OracleRewardAwareMinimaxVariance,
    RewardGreedyWithFloor,
}
impl LoggerKind {
    const ALL: [Self; 8] = [
        Self::Uniform,
        Self::IncumbentMixture,
        Self::PortfolioAverage,
        Self::SecondMomentPseudoTargetProp6,
        Self::MinimaxWeightSecondMoment,
        Self::RewardEnvelopeMinimaxSecondMoment,
        Self::OracleRewardAwareMinimaxVariance,
        Self::RewardGreedyWithFloor,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::IncumbentMixture => "incumbent_mixture",
            Self::PortfolioAverage => "portfolio_average",
            Self::SecondMomentPseudoTargetProp6 => "prop6_second_moment_pseudotarget_known_reward",
            Self::MinimaxWeightSecondMoment => "minimax_weight_second_moment_grid",
            Self::RewardEnvelopeMinimaxSecondMoment => "reward_envelope_minimax_ips_second_moment_grid",
            Self::OracleRewardAwareMinimaxVariance => "oracle_reward_aware_minimax_ope_variance_grid",
            Self::RewardGreedyWithFloor => "reward_greedy_with_floor",
        }
    }
    fn oracle_only(self) -> bool {
        matches!(
            self,
            Self::SecondMomentPseudoTargetProp6
                | Self::OracleRewardAwareMinimaxVariance
                | Self::RewardGreedyWithFloor
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Policy {
    id: usize,
    /// Probability of action 1 under this target policy in each context.
    p1: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Logger {
    /// Probability of action 1 under the logging policy in each context.
    p1: [f64; 2],
}

#[derive(Clone, Copy, Debug)]
struct Metrics {
    logger_reward: f64,
    reward_gap_vs_floor_greedy: f64,
    worst_weight_second_moment: f64,
    implied_ess_fraction: f64,
    worst_ips_reward_variance: f64,
    mean_ips_reward_variance: f64,
    worst_reward_envelope_second_moment: f64,
    min_action_propensity: f64,
    max_action_propensity: f64,
    worst_policy_id: usize,
}

fn validate_floor(floor: f64) -> Result<(), String> {
    if !floor.is_finite() || floor <= 0.0 || floor >= 0.5 {
        return Err("action propensity floor must be finite and in (0, 0.5)".into());
    }
    Ok(())
}
fn validate_logger(logger: Logger, floor: f64) -> Result<(), String> {
    validate_floor(floor)?;
    for (context, p) in logger.p1.iter().copied().enumerate() {
        if !p.is_finite() || p < floor - EPS || p > 1.0 - floor + EPS {
            return Err(format!(
                "logger action-1 probability for context {context} violates [floor, 1-floor]"
            ));
        }
    }
    Ok(())
}

fn policies(regime: PortfolioRegime, m: usize) -> Vec<Policy> {
    assert!(m > 1, "registered portfolios have at least two policies");
    (0..m)
        .map(|j| {
            let x = j as f64 / (m - 1) as f64;
            let permuted = (7 * j) % m;
            let y = permuted as f64 / (m - 1) as f64;
            let p1 = match regime {
                PortfolioRegime::Balanced => [0.05 + 0.90 * x, 0.95 - 0.90 * y],
                PortfolioRegime::Action1Heavy => [0.55 + 0.40 * x, 0.55 + 0.40 * y],
                PortfolioRegime::Action0Heavy => [0.05 + 0.40 * x, 0.05 + 0.40 * y],
            };
            Policy { id: j, p1 }
        })
        .collect()
}

/// Bernoulli reward means from the frozen Symthaea contextual-bandit fixture.
fn reward_mean(context: usize, action: usize) -> f64 {
    match (context, action) {
        (0, 0) => 0.10,
        (0, 1) => 0.90,
        (1, 0) => 0.95,
        (1, 1) => 0.15,
        _ => unreachable!(),
    }
}

fn target_value(policy: Policy) -> f64 {
    let p0 = policy.p1[0];
    let p1 = policy.p1[1];
    0.5 * ((1.0 - p0) * reward_mean(0, 0)
        + p0 * reward_mean(0, 1)
        + (1.0 - p1) * reward_mean(1, 0)
        + p1 * reward_mean(1, 1))
}
fn logger_reward(logger: Logger) -> f64 {
    let p0 = logger.p1[0];
    let p1 = logger.p1[1];
    0.5 * ((1.0 - p0) * reward_mean(0, 0)
        + p0 * reward_mean(0, 1)
        + (1.0 - p1) * reward_mean(1, 0)
        + p1 * reward_mean(1, 1))
}

/// Exact E_mu[w_pi(X,A)^2] with known context probabilities P(X=x)=1/2.
/// E_mu[w]=1 when the logger has support on the target policy.
fn weight_second_moment(policy: Policy, logger: Logger) -> f64 {
    let mut result = 0.0;
    for context in 0..2 {
        let q = policy.p1[context];
        let p = logger.p1[context];
        result += 0.5 * (q * q / p + (1.0 - q) * (1.0 - q) / (1.0 - p));
    }
    result
}

/// Registered conservative upper envelope on Bernoulli reward means. The envelope
/// is an explicit information assumption; it is not estimated from hidden test truth.
fn reward_upper_mean(context: usize, action: usize) -> f64 {
    REWARD_UPPER_ENVELOPE[context][action]
}

fn validate_reward_envelope(
    envelope: [[f64; 2]; 2],
    true_means: [[f64; 2]; 2],
) -> Result<(), String> {
    for context in 0..2 {
        for action in 0..2 {
            let upper = envelope[context][action];
            let truth = true_means[context][action];
            if !upper.is_finite() || !(0.0..=1.0).contains(&upper) {
                return Err(format!(
                    "reward envelope is not a finite Bernoulli mean at context={context}, action={action}"
                ));
            }
            if !truth.is_finite() || !(0.0..=1.0).contains(&truth) {
                return Err(format!(
                    "true reward mean is not a finite Bernoulli mean at context={context}, action={action}"
                ));
            }
            if upper + EPS < truth {
                return Err(format!(
                    "reward envelope undercovers context={context}, action={action}: upper={upper}, truth={truth}"
                ));
            }
        }
    }
    Ok(())
}

fn frozen_reward_mean_table() -> [[f64; 2]; 2] {
    [
        [reward_mean(0, 0), reward_mean(0, 1)],
        [reward_mean(1, 0), reward_mean(1, 1)],
    ]
}

/// Upper bound on E_mu[(w_pi R)^2] using a registered reward envelope.
/// For Bernoulli rewards, R^2=R, so it upper-bounds the exact second moment if
/// every envelope entry is at least the true reward mean. It is not variance itself.
fn envelope_ips_reward_second_moment_with(
    policy: Policy,
    logger: Logger,
    envelope: [[f64; 2]; 2],
) -> f64 {
    let mut second_moment_upper = 0.0;
    for context in 0..2 {
        let q = policy.p1[context];
        let p = logger.p1[context];
        second_moment_upper += 0.5
            * (q * q * envelope[context][1] / p
                + (1.0 - q) * (1.0 - q) * envelope[context][0] / (1.0 - p));
    }
    second_moment_upper
}
fn envelope_ips_reward_second_moment(policy: Policy, logger: Logger) -> f64 {
    envelope_ips_reward_second_moment_with(policy, logger, REWARD_UPPER_ENVELOPE)
}
fn worst_reward_envelope_second_moment_with(
    policy_set: &[Policy],
    logger: Logger,
    envelope: [[f64; 2]; 2],
) -> (usize, f64) {
    policy_set
        .iter()
        .map(|p| (p.id, envelope_ips_reward_second_moment_with(*p, logger, envelope)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, f64::INFINITY))
}
fn worst_reward_envelope_second_moment(policy_set: &[Policy], logger: Logger) -> (usize, f64) {
    worst_reward_envelope_second_moment_with(policy_set, logger, REWARD_UPPER_ENVELOPE)
}
fn scaled_reward_envelope(scale: f64) -> Result<[[f64; 2]; 2], String> {
    if !scale.is_finite() || scale < 1.0 {
        return Err("reward-envelope scale must be finite and at least 1".into());
    }
    Ok([
        [
            (REWARD_UPPER_ENVELOPE[0][0] * scale).min(1.0),
            (REWARD_UPPER_ENVELOPE[0][1] * scale).min(1.0),
        ],
        [
            (REWARD_UPPER_ENVELOPE[1][0] * scale).min(1.0),
            (REWARD_UPPER_ENVELOPE[1][1] * scale).min(1.0),
        ],
    ])
}

/// Exact E_mu[(w_pi R)^2] - V(pi)^2 for Bernoulli rewards, because R^2=R.
/// This is an oracle-information objective: it uses the known synthetic reward means.
fn ips_reward_variance(policy: Policy, logger: Logger) -> f64 {
    let mut second_moment = 0.0;
    for context in 0..2 {
        let q = policy.p1[context];
        let p = logger.p1[context];
        second_moment += 0.5
            * (q * q * reward_mean(context, 1) / p
                + (1.0 - q) * (1.0 - q) * reward_mean(context, 0) / (1.0 - p));
    }
    (second_moment - target_value(policy).powi(2)).max(0.0)
}
fn worst_weight_second_moment(policy_set: &[Policy], logger: Logger) -> (usize, f64) {
    policy_set
        .iter()
        .map(|p| (p.id, weight_second_moment(*p, logger)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, f64::INFINITY))
}
fn worst_ips_variance(policy_set: &[Policy], logger: Logger) -> (usize, f64) {
    policy_set
        .iter()
        .map(|p| (p.id, ips_reward_variance(*p, logger)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .unwrap_or((0, f64::INFINITY))
}
fn mean_ips_variance(policy_set: &[Policy], logger: Logger) -> f64 {
    if policy_set.is_empty() {
        return f64::INFINITY;
    }
    policy_set
        .iter()
        .map(|policy| ips_reward_variance(*policy, logger))
        .sum::<f64>()
        / policy_set.len() as f64
}

fn clamp_floor(p: f64, floor: f64) -> f64 {
    p.clamp(floor, 1.0 - floor)
}
fn portfolio_average_logger(policy_set: &[Policy], floor: f64) -> Logger {
    let mut p1 = [0.0; 2];
    for policy in policy_set {
        for context in 0..2 {
            p1[context] += policy.p1[context] / policy_set.len() as f64;
        }
    }
    Logger {
        p1: [clamp_floor(p1[0], floor), clamp_floor(p1[1], floor)],
    }
}

/// Proposition 6 (Douglas, Persson & Provost, 2026): for a known distribution
/// over target policies and known reward means, the expected-MSE-optimal logger
/// is proportional to sqrt(E[pi_t(a|x)^2]) * sqrt(mu(a,x)) for each action.
/// This finite portfolio uses the uniform empirical prior over its policies.
fn second_moment_pseudotarget_logger(
    policy_set: &[Policy],
    floor: f64,
) -> Result<Logger, String> {
    if policy_set.is_empty() {
        return Err("target policy distribution must be nonempty".into());
    }
    validate_floor(floor)?;
    let mut p1 = [0.0; 2];
    for context in 0..2 {
        let pseudo_target_action1 = (policy_set
            .iter()
            .map(|policy| policy.p1[context].powi(2))
            .sum::<f64>()
            / policy_set.len() as f64)
            .sqrt();
        let pseudo_target_action0 = (policy_set
            .iter()
            .map(|policy| (1.0 - policy.p1[context]).powi(2))
            .sum::<f64>()
            / policy_set.len() as f64)
            .sqrt();
        let weighted_action1 =
            pseudo_target_action1 * reward_mean(context, 1).sqrt();
        let weighted_action0 =
            pseudo_target_action0 * reward_mean(context, 0).sqrt();
        let normalizer = weighted_action1 + weighted_action0;
        if !normalizer.is_finite() || normalizer <= EPS {
            return Err("Proposition 6 pseudo-target has a zero or invalid normalizer".into());
        }
        p1[context] = clamp_floor(weighted_action1 / normalizer, floor);
    }
    let logger = Logger { p1 };
    validate_logger(logger, floor)?;
    Ok(logger)
}
fn incumbent_mixture_logger(policy_set: &[Policy], floor: f64) -> Logger {
    let incumbent = policy_set[0];
    let p1 = [
        0.5 * GREEDY_EXPLORATION_MIX + (1.0 - GREEDY_EXPLORATION_MIX) * incumbent.p1[0],
        0.5 * GREEDY_EXPLORATION_MIX + (1.0 - GREEDY_EXPLORATION_MIX) * incumbent.p1[1],
    ];
    Logger {
        p1: [clamp_floor(p1[0], floor), clamp_floor(p1[1], floor)],
    }
}
fn reward_greedy_logger(floor: f64) -> Logger {
    let mut p1 = [0.5; 2];
    for (context, p) in p1.iter_mut().enumerate() {
        *p = if reward_mean(context, 1) >= reward_mean(context, 0) {
            1.0 - floor
        } else {
            floor
        };
    }
    Logger { p1 }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum GridObjective {
    WeightSecondMoment,
    RewardEnvelopeSecondMoment([[f64; 2]; 2]),
    RewardVariance,
}
fn objective_score(policy_set: &[Policy], logger: Logger, objective: GridObjective) -> f64 {
    match objective {
        GridObjective::WeightSecondMoment => worst_weight_second_moment(policy_set, logger).1,
        GridObjective::RewardEnvelopeSecondMoment(envelope) => {
            worst_reward_envelope_second_moment_with(policy_set, logger, envelope).1
        }
        GridObjective::RewardVariance => worst_ips_variance(policy_set, logger).1,
    }
}

/// Deterministic two-context finite-grid minimax search. This is a reproducible
/// approximate optimizer, not a claim of exact global optimality off-grid.
fn minimax_logger(
    policy_set: &[Policy],
    floor: f64,
    grid_intervals: usize,
    objective: GridObjective,
) -> Result<Logger, String> {
    validate_floor(floor)?;
    if policy_set.is_empty() || grid_intervals == 0 {
        return Err("nonempty policy portfolio and positive grid intervals are required".into());
    }
    let span = 1.0 - 2.0 * floor;
    let mut best = Logger { p1: [0.5, 0.5] };
    let mut best_score = f64::INFINITY;
    let mut best_uniform_distance = f64::INFINITY;
    for i in 0..=grid_intervals {
        let p0 = floor + span * i as f64 / grid_intervals as f64;
        for j in 0..=grid_intervals {
            let p1 = floor + span * j as f64 / grid_intervals as f64;
            let candidate = Logger { p1: [p0, p1] };
            let score = objective_score(policy_set, candidate, objective);
            let uniform_distance = (p0 - 0.5).powi(2) + (p1 - 0.5).powi(2);
            if score < best_score - EPS
                || ((score - best_score).abs() <= EPS
                    && uniform_distance < best_uniform_distance - EPS)
            {
                best = candidate;
                best_score = score;
                best_uniform_distance = uniform_distance;
            }
        }
    }
    validate_logger(best, floor)?;
    Ok(best)
}

fn logger_for(
    kind: LoggerKind,
    policy_set: &[Policy],
    floor: f64,
    grid_intervals: usize,
) -> Result<Logger, String> {
    let logger = match kind {
        LoggerKind::Uniform => Logger { p1: [0.5, 0.5] },
        LoggerKind::IncumbentMixture => incumbent_mixture_logger(policy_set, floor),
        LoggerKind::PortfolioAverage => portfolio_average_logger(policy_set, floor),
        LoggerKind::SecondMomentPseudoTargetProp6 => {
            second_moment_pseudotarget_logger(policy_set, floor)?
        }
        LoggerKind::MinimaxWeightSecondMoment => minimax_logger(
            policy_set,
            floor,
            grid_intervals,
            GridObjective::WeightSecondMoment,
        )?,
        LoggerKind::RewardEnvelopeMinimaxSecondMoment => minimax_logger(
            policy_set,
            floor,
            grid_intervals,
            GridObjective::RewardEnvelopeSecondMoment(REWARD_UPPER_ENVELOPE),
        )?,
        LoggerKind::OracleRewardAwareMinimaxVariance => minimax_logger(
            policy_set,
            floor,
            grid_intervals,
            GridObjective::RewardVariance,
        )?,
        LoggerKind::RewardGreedyWithFloor => reward_greedy_logger(floor),
    };
    validate_logger(logger, floor)?;
    Ok(logger)
}

fn metrics(policy_set: &[Policy], logger: Logger, floor: f64) -> Result<Metrics, String> {
    validate_logger(logger, floor)?;
    if policy_set.is_empty() {
        return Err("target policy portfolio must be nonempty".into());
    }
    let (worst_policy_id, worst_weight_m2) = worst_weight_second_moment(policy_set, logger);
    let (_, worst_variance) = worst_ips_variance(policy_set, logger);
    let mean_variance = mean_ips_variance(policy_set, logger);
    let (_, worst_reward_envelope_m2) = worst_reward_envelope_second_moment(policy_set, logger);
    let greedy_reward = logger_reward(reward_greedy_logger(floor));
    let min_action_propensity = logger
        .p1
        .iter()
        .map(|p| p.min(1.0 - p))
        .fold(f64::INFINITY, f64::min);
    let max_action_propensity = logger
        .p1
        .iter()
        .map(|p| p.max(1.0 - p))
        .fold(0.0_f64, f64::max);
    Ok(Metrics {
        logger_reward: logger_reward(logger),
        reward_gap_vs_floor_greedy: (greedy_reward - logger_reward(logger)).max(0.0),
        worst_weight_second_moment: worst_weight_m2,
        implied_ess_fraction: 1.0 / worst_weight_m2.max(1.0),
        worst_ips_reward_variance: worst_variance,
        mean_ips_reward_variance: mean_variance,
        worst_reward_envelope_second_moment: worst_reward_envelope_m2,
        min_action_propensity,
        max_action_propensity,
        worst_policy_id,
    })
}

fn benchmark(grid_intervals: usize) -> Result<String, String> {
    if grid_intervals == 0 {
        return Err("grid intervals must be positive".into());
    }
    validate_reward_envelope(REWARD_UPPER_ENVELOPE, frozen_reward_mean_table())?;
    let mut out = String::from(
        "record_type\tportfolio\tM\tlogger\tp1_context0\tp1_context1\tmin_action_propensity\tmax_action_propensity\texpected_logger_reward\treward_gap_vs_floor_greedy\tworst_policy_id\tworst_weight_second_moment\tworst_reward_envelope_ips_second_moment\timplied_min_ess_fraction\tworst_ips_reward_variance\tmean_ips_reward_variance\toracle_only\tgrid_intervals\n",
    );
    out.push_str(&format!(
        "BENCHMARK_CONFIG\tid\tVIS-DVPC-051\tfloor\t{ACTION_PROPENSITY_FLOOR:.6}\tgrid_intervals\t{grid_intervals}\n"
    ));
    out.push_str("BENCHMARK_CONFIG\tinterpretation\tanalytic_expected_values_no_monte_carlo_sampling\n");
    out.push_str("BENCHMARK_CONFIG\treward_envelope\tregistered upper means: 0.20,0.95;0.98,0.20\n");
    out.push_str("BENCHMARK_CONFIG\treward_envelope_sensitivity_scales\t1.0,1.1,1.25,1.5,2.0\n");
    out.push_str("BENCHMARK_CONFIG\toracle_only\toracle_reward_aware_variance and reward_greedy_with_floor use known synthetic reward means\n");
    out.push_str("BENCHMARK_CONFIG\tfrontier\timmediate_reward_vs_worst_policy_IPS_variance_weight_moment_and_envelope_bound\n");
    for regime in PortfolioRegime::ALL {
        for m in POLICY_COUNTS {
            let policy_set = policies(regime, m);
            for scale in REWARD_ENVELOPE_SCALES {
                let envelope = scaled_reward_envelope(scale)?;
                validate_reward_envelope(envelope, frozen_reward_mean_table())?;
                let logger = minimax_logger(
                    &policy_set,
                    ACTION_PROPENSITY_FLOOR,
                    grid_intervals,
                    GridObjective::RewardEnvelopeSecondMoment(envelope),
                )?;
                let (_, envelope_m2) =
                    worst_reward_envelope_second_moment_with(&policy_set, logger, envelope);
                let (_, worst_weight_m2) = worst_weight_second_moment(&policy_set, logger);
                let (_, worst_variance) = worst_ips_variance(&policy_set, logger);
                out.push_str(&format!(
                    "ENVELOPE_SENSITIVITY\t{}\t{}\t{:.2}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\n",
                    regime.name(),
                    m,
                    scale,
                    logger.p1[0],
                    logger.p1[1],
                    logger_reward(logger),
                    worst_weight_m2,
                    1.0 / worst_weight_m2.max(1.0),
                    envelope_m2,
                    worst_variance,
                    worst_reward_envelope_second_moment_with(&policy_set, logger, envelope).0,
                    grid_intervals,
                ));
            }
            for kind in LoggerKind::ALL {
                let logger = logger_for(kind, &policy_set, ACTION_PROPENSITY_FLOOR, grid_intervals)?;
                let met = metrics(&policy_set, logger, ACTION_PROPENSITY_FLOOR)?;
                out.push_str(&format!(
                    "RESULT\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{:.8}\t{}\t{}\n",
                    regime.name(),
                    m,
                    kind.name(),
                    logger.p1[0],
                    logger.p1[1],
                    met.min_action_propensity,
                    met.max_action_propensity,
                    met.logger_reward,
                    met.reward_gap_vs_floor_greedy,
                    met.worst_policy_id,
                    met.worst_weight_second_moment,
                    met.worst_reward_envelope_second_moment,
                    met.implied_ess_fraction,
                    met.worst_ips_reward_variance,
                    met.mean_ips_reward_variance,
                    kind.oracle_only(),
                    grid_intervals,
                ));
            }
        }
    }
    Ok(out)
}

fn main() {
    let smoke = env::args().any(|arg| arg == "--smoke");
    let grid_intervals = if smoke {
        SMOKE_GRID_INTERVALS
    } else {
        DEFAULT_GRID_INTERVALS
    };
    match benchmark(grid_intervals) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_051_ERROR\t{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_second_moment_matches_independent_enumeration() {
        let policy = Policy {
            id: 7,
            p1: [0.8, 0.3],
        };
        let logger = Logger { p1: [0.4, 0.6] };
        let independently_enumerated = 0.5
            * (0.8_f64.powi(2) / 0.4
                + 0.2_f64.powi(2) / 0.6
                + 0.3_f64.powi(2) / 0.6
                + 0.7_f64.powi(2) / 0.4);
        assert!((weight_second_moment(policy, logger) - independently_enumerated).abs() < EPS);
    }

    #[test]
    fn registered_reward_envelope_validates_against_frozen_world() {
        validate_reward_envelope(REWARD_UPPER_ENVELOPE, frozen_reward_mean_table()).unwrap();
    }

    #[test]
    fn reward_envelope_fails_closed_on_undercoverage_and_invalid_means() {
        let mut shifted_world = frozen_reward_mean_table();
        shifted_world[0][1] = 0.97;
        assert!(validate_reward_envelope(REWARD_UPPER_ENVELOPE, shifted_world).is_err());

        let mut invalid_envelope = REWARD_UPPER_ENVELOPE;
        invalid_envelope[1][0] = f64::NAN;
        assert!(validate_reward_envelope(invalid_envelope, frozen_reward_mean_table()).is_err());

        let mut invalid_truth = frozen_reward_mean_table();
        invalid_truth[1][1] = 1.01;
        assert!(validate_reward_envelope(REWARD_UPPER_ENVELOPE, invalid_truth).is_err());
    }

    #[test]
    fn reward_envelope_is_conservative_for_the_frozen_reward_table() {
        for context in 0..2 {
            for action in 0..2 {
                assert!(reward_upper_mean(context, action) + EPS >= reward_mean(context, action));
            }
        }
        let logger = Logger { p1: [0.3, 0.7] };
        for policy in policies(PortfolioRegime::Balanced, 8) {
            let exact_second_moment = ips_reward_variance(policy, logger)
                + target_value(policy).powi(2);
            assert!(
                envelope_ips_reward_second_moment(policy, logger) + EPS >= exact_second_moment
            );
        }
    }

    #[test]
    fn envelope_aware_design_improves_balanced_portfolio_worst_variance_over_uniform() {
        let policy_set = policies(PortfolioRegime::Balanced, 8);
        let uniform = Logger { p1: [0.5, 0.5] };
        let envelope_logger = minimax_logger(
            &policy_set,
            ACTION_PROPENSITY_FLOOR,
            100,
            GridObjective::RewardEnvelopeSecondMoment(REWARD_UPPER_ENVELOPE),
        )
        .unwrap();
        let uniform_variance = worst_ips_variance(&policy_set, uniform).1;
        let envelope_variance = worst_ips_variance(&policy_set, envelope_logger).1;
        assert!(
            envelope_variance + 0.05 < uniform_variance,
            "uniform={uniform_variance}, envelope={envelope_variance}, logger={envelope_logger:?}"
        );
    }

    #[test]
    fn exact_ips_reward_variance_matches_enumerated_bernoulli_moments() {
        let policy = Policy {
            id: 0,
            p1: [0.6, 0.4],
        };
        let logger = Logger { p1: [0.3, 0.7] };
        let mut second_moment = 0.0;
        for context in 0..2 {
            let pi1 = policy.p1[context];
            let mu1 = logger.p1[context];
            let r0 = reward_mean(context, 0);
            let r1 = reward_mean(context, 1);
            second_moment += 0.5
                * (pi1.powi(2) * r1 / mu1 + (1.0 - pi1).powi(2) * r0 / (1.0 - mu1));
        }
        let expected_variance = (second_moment - target_value(policy).powi(2)).max(0.0);
        assert!((ips_reward_variance(policy, logger) - expected_variance).abs() < EPS);
    }

    #[test]
    fn prop6_second_moment_pseudotarget_matches_independent_formula() {
        let policy_set = policies(PortfolioRegime::Action1Heavy, 8);
        let logger = second_moment_pseudotarget_logger(&policy_set, ACTION_PROPENSITY_FLOOR)
            .unwrap();
        for context in 0..2 {
            let pseudo1 = (policy_set
                .iter()
                .map(|policy| policy.p1[context].powi(2))
                .sum::<f64>()
                / policy_set.len() as f64)
                .sqrt();
            let pseudo0 = (policy_set
                .iter()
                .map(|policy| (1.0 - policy.p1[context]).powi(2))
                .sum::<f64>()
                / policy_set.len() as f64)
                .sqrt();
            let expected = (pseudo1 * reward_mean(context, 1).sqrt())
                / (pseudo1 * reward_mean(context, 1).sqrt()
                    + pseudo0 * reward_mean(context, 0).sqrt());
            assert!((logger.p1[context] - expected).abs() < EPS);
        }
        validate_logger(logger, ACTION_PROPENSITY_FLOOR).unwrap();
    }

    #[test]
    fn prop6_pseudotarget_minimizes_average_variance_over_policy_distribution() {
        let policy_set = policies(PortfolioRegime::Balanced, 8);
        let logger = second_moment_pseudotarget_logger(&policy_set, ACTION_PROPENSITY_FLOOR)
            .unwrap();
        let optimal_average = mean_ips_variance(&policy_set, logger);
        let mut grid_minimum = f64::INFINITY;
        let intervals = 400;
        for i in 0..=intervals {
            let p0 = ACTION_PROPENSITY_FLOOR
                + (1.0 - 2.0 * ACTION_PROPENSITY_FLOOR) * i as f64 / intervals as f64;
            for j in 0..=intervals {
                let p1 = ACTION_PROPENSITY_FLOOR
                    + (1.0 - 2.0 * ACTION_PROPENSITY_FLOOR) * j as f64 / intervals as f64;
                grid_minimum = grid_minimum.min(mean_ips_variance(
                    &policy_set,
                    Logger { p1: [p0, p1] },
                ));
            }
        }
        assert!(
            optimal_average <= grid_minimum + 1e-6,
            "Prop 6 logger average variance={optimal_average}, grid minimum={grid_minimum}"
        );
    }

    #[test]
    fn skewed_portfolio_minimax_design_is_not_forced_to_uniform() {
        let policy_set = policies(PortfolioRegime::Action1Heavy, 8);
        let logger = minimax_logger(
            &policy_set,
            ACTION_PROPENSITY_FLOOR,
            100,
            GridObjective::WeightSecondMoment,
        )
        .unwrap();
        assert!(logger.p1.iter().all(|p| (*p - 0.5).abs() > 0.05), "{logger:?}");
        validate_logger(logger, ACTION_PROPENSITY_FLOOR).unwrap();
    }

    #[test]
    fn wider_reward_envelopes_never_reduce_the_fixed_logger_second_moment_bound() {
        let policy_set = policies(PortfolioRegime::Balanced, 8);
        let logger = Logger { p1: [0.43, 0.61] };
        let mut previous = 0.0_f64;
        for scale in REWARD_ENVELOPE_SCALES {
            let envelope = scaled_reward_envelope(scale).unwrap();
            validate_reward_envelope(envelope, frozen_reward_mean_table()).unwrap();
            let current = worst_reward_envelope_second_moment_with(
                &policy_set,
                logger,
                envelope,
            ).1;
            assert!(current + EPS >= previous, "scale={scale}, current={current}, previous={previous}");
            previous = current;
        }
        assert!(scaled_reward_envelope(0.99).is_err());
        assert!(scaled_reward_envelope(f64::NAN).is_err());
    }

    #[test]
    fn weight_only_minimax_can_worsen_reward_weighted_variance() {
        let policy_set = policies(PortfolioRegime::Balanced, 8);
        let uniform = Logger { p1: [0.5, 0.5] };
        let weight_minimax = minimax_logger(
            &policy_set,
            ACTION_PROPENSITY_FLOOR,
            100,
            GridObjective::WeightSecondMoment,
        )
        .unwrap();
        let uniform_m2 = worst_weight_second_moment(&policy_set, uniform).1;
        let minimax_m2 = worst_weight_second_moment(&policy_set, weight_minimax).1;
        let uniform_variance = worst_ips_variance(&policy_set, uniform).1;
        let minimax_variance = worst_ips_variance(&policy_set, weight_minimax).1;
        assert!(minimax_m2 < uniform_m2 - 0.05);
        assert!(
            minimax_variance > uniform_variance + 0.02,
            "variance proxy counterexample expected: logger={weight_minimax:?}, uniform={uniform_variance}, minimax={minimax_variance}"
        );
    }

    #[test]
    fn minimax_grid_matches_independent_exhaustive_grid_oracle() {
        let policy_set = policies(PortfolioRegime::Action0Heavy, 4);
        let intervals = 12;
        let result = minimax_logger(
            &policy_set,
            ACTION_PROPENSITY_FLOOR,
            intervals,
            GridObjective::WeightSecondMoment,
        )
        .unwrap();
        let result_score = objective_score(&policy_set, result, GridObjective::WeightSecondMoment);
        let mut exhaustive = f64::INFINITY;
        for i in 0..=intervals {
            for j in 0..=intervals {
                let logger = Logger {
                    p1: [
                        ACTION_PROPENSITY_FLOOR
                            + (1.0 - 2.0 * ACTION_PROPENSITY_FLOOR) * i as f64 / intervals as f64,
                        ACTION_PROPENSITY_FLOOR
                            + (1.0 - 2.0 * ACTION_PROPENSITY_FLOOR) * j as f64 / intervals as f64,
                    ],
                };
                exhaustive = exhaustive
                    .min(objective_score(&policy_set, logger, GridObjective::WeightSecondMoment));
            }
        }
        assert!((result_score - exhaustive).abs() < EPS);
    }

    #[test]
    fn all_deployable_loggers_preserve_the_registered_support_floor() {
        for regime in PortfolioRegime::ALL {
            for m in POLICY_COUNTS {
                let policy_set = policies(regime, m);
                for kind in LoggerKind::ALL {
                    let logger =
                        logger_for(kind, &policy_set, ACTION_PROPENSITY_FLOOR, 20).unwrap();
                    validate_logger(logger, ACTION_PROPENSITY_FLOOR).unwrap();
                    assert!(logger.p1.iter().all(|p| *p >= ACTION_PROPENSITY_FLOOR - EPS));
                    assert!(logger
                        .p1
                        .iter()
                        .all(|p| *p <= 1.0 - ACTION_PROPENSITY_FLOOR + EPS));
                }
            }
        }
    }

    #[test]
    fn reward_greedy_with_floor_matches_known_reward_ordering() {
        let logger = reward_greedy_logger(ACTION_PROPENSITY_FLOOR);
        assert_eq!(logger.p1, [0.98, 0.02]);
        let met = metrics(&policies(PortfolioRegime::Balanced, 4), logger, ACTION_PROPENSITY_FLOOR)
            .unwrap();
        assert!((met.logger_reward - 0.909).abs() < EPS);
        assert!(met.reward_gap_vs_floor_greedy.abs() < EPS);
    }

    #[test]
    fn minimax_refuses_invalid_floor_and_empty_portfolio() {
        assert!(minimax_logger(
            &policies(PortfolioRegime::Balanced, 4),
            0.0,
            10,
            GridObjective::WeightSecondMoment
        )
        .is_err());
        assert!(minimax_logger(
            &[],
            ACTION_PROPENSITY_FLOOR,
            10,
            GridObjective::WeightSecondMoment
        )
        .is_err());
    }

    #[test]
    fn objective_output_records_oracle_boundary_and_all_rows_are_stable() {
        let a = benchmark(10).unwrap();
        let b = benchmark(10).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.lines().filter(|line| line.starts_with("RESULT\t")).count(), 72);
        assert_eq!(
            a.lines()
                .filter(|line| line.starts_with("ENVELOPE_SENSITIVITY\t"))
                .count(),
            45
        );
        assert!(a.contains("oracle_reward_aware_variance and reward_greedy_with_floor use known synthetic reward means"));
        assert!(a.contains("reward_envelope_minimax_ips_second_moment_grid"));
        assert!(a.contains("minimax_weight_second_moment_grid"));
        assert!(a.contains("oracle_reward_aware_minimax_ope_variance_grid"));
    }
}