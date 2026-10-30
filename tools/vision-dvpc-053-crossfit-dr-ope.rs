//! VIS-DVPC-053: estimator-aware logging and cross-fitted doubly robust OPE.
//!
//! Run:
//!   rustc --edition=2024 -O --test tools/vision-dvpc-053-crossfit-dr-ope.rs -o /tmp/vis-dvpc-053-tests
//!   /tmp/vis-dvpc-053-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-053-crossfit-dr-ope.rs -o /tmp/vis-dvpc-053
//!   /tmp/vis-dvpc-053 --smoke
//!   /tmp/vis-dvpc-053
//!
//! Pilot, shrinkage-calibration, and evaluation datasets use disjoint seeds.
//! Logging propensities are generated and retained exactly. The in-sample
//! nuisance arm is deliberately non-cross-fitted; it is a diagnostic, not a
//! recommended estimator. All reported OPE arms require full target support.
#![deny(unsafe_code)]

use std::env;

const SAMPLE_SIZES: [usize; 2] = [64, 256];
const FULL_REPLICATES: usize = 128;
const SMOKE_REPLICATES: usize = 12;
const CALIBRATION_N: usize = 128;
const ACTION_FLOOR: f64 = 0.02;
const EPS: f64 = 1e-12;

const SOURCE_REWARD: [[f64; 2]; 2] = [[0.12, 0.82], [0.88, 0.18]];
const TARGET_POLICIES: [Policy; 4] = [
    Policy { id: 0, p1: [0.85, 0.15] },
    Policy { id: 1, p1: [0.15, 0.85] },
    Policy { id: 2, p1: [0.50, 0.50] },
    Policy { id: 3, p1: [0.90, 0.90] },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Scenario {
    Stationary,
    NoisyPilot,
    PredictorBias,
    RewardDrift,
    ContextShift,
}
impl Scenario {
    const ALL: [Self; 5] = [
        Self::Stationary,
        Self::NoisyPilot,
        Self::PredictorBias,
        Self::RewardDrift,
        Self::ContextShift,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Stationary => "stationary",
            Self::NoisyPilot => "noisy_pilot",
            Self::PredictorBias => "predictor_bias",
            Self::RewardDrift => "reward_drift",
            Self::ContextShift => "context_shift",
        }
    }
    fn index(self) -> u64 {
        match self {
            Self::Stationary => 0,
            Self::NoisyPilot => 1,
            Self::PredictorBias => 2,
            Self::RewardDrift => 3,
            Self::ContextShift => 4,
        }
    }
    fn pilot_n(self) -> usize {
        if self == Self::NoisyPilot { 24 } else { 256 }
    }
    fn eval_context(self) -> [f64; 2] {
        if self == Self::ContextShift { [0.80, 0.20] } else { [0.50, 0.50] }
    }
    fn eval_rewards(self) -> [[f64; 2]; 2] {
        let mut rewards = SOURCE_REWARD;
        if self == Self::RewardDrift {
            rewards[0][1] = (rewards[0][1] + 0.14).min(0.99);
            rewards[1][0] = (rewards[1][0] - 0.13).max(0.01);
        }
        rewards
    }
    fn predictor_bias(self) -> [[f64; 2]; 2] {
        if self == Self::PredictorBias {
            [[0.22, -0.22], [0.18, -0.18]]
        } else {
            [[0.0; 2]; 2]
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Policy {
    id: usize,
    /// Probability of action 1 in each context.
    p1: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Logger {
    /// Probability of action 1 in each context.
    p1: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoggerKind {
    Uniform,
    RawPilot,
    HeldoutShrinkage,
    LeakyShrinkage,
    OracleMseShrinkage,
}
impl LoggerKind {
    const ALL: [Self; 5] = [
        Self::Uniform,
        Self::RawPilot,
        Self::HeldoutShrinkage,
        Self::LeakyShrinkage,
        Self::OracleMseShrinkage,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Uniform => "uniform",
            Self::RawPilot => "raw_pilot",
            Self::HeldoutShrinkage => "heldout_shrinkage",
            Self::LeakyShrinkage => "leaky_shrinkage",
            Self::OracleMseShrinkage => "oracle_mse_shrinkage",
        }
    }
    fn as_u64_index(self) -> u64 {
        match self {
            Self::Uniform => 0,
            Self::RawPilot => 1,
            Self::HeldoutShrinkage => 2,
            Self::LeakyShrinkage => 3,
            Self::OracleMseShrinkage => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Estimator {
    Ips,
    DmPilot,
    DrPilot,
    DmInSample,
    DrInSample,
    DrCrossfit2,
}
impl Estimator {
    const ALL: [Self; 6] = [
        Self::Ips,
        Self::DmPilot,
        Self::DrPilot,
        Self::DmInSample,
        Self::DrInSample,
        Self::DrCrossfit2,
    ];
    fn name(self) -> &'static str {
        match self {
            Self::Ips => "ips",
            Self::DmPilot => "dm_pilot",
            Self::DrPilot => "dr_pilot",
            Self::DmInSample => "dm_in_sample",
            Self::DrInSample => "dr_in_sample",
            Self::DrCrossfit2 => "dr_crossfit_2fold",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Observation {
    context: usize,
    action: usize,
    reward: f64,
    /// Exact probability assigned to the action by the logger at collection.
    propensity: f64,
}

#[derive(Clone, Copy, Debug)]
struct Estimate {
    value: f64,
    standard_error: f64,
    /// Same-sample fitted nuisance models do not receive score-based CIs.
    interval_valid: bool,
}

#[derive(Default)]
struct Aggregate {
    count: usize,
    sum_error: f64,
    sum_squared_error: f64,
    sum_ci_width: f64,
    interval_count: usize,
    covered: usize,
}
impl Aggregate {
    fn add(&mut self, estimate: Estimate, truth: f64) {
        let error = estimate.value - truth;
        self.count += 1;
        self.sum_error += error;
        self.sum_squared_error += error * error;
        if estimate.interval_valid {
            self.interval_count += 1;
            self.sum_ci_width += 2.0 * 1.96 * estimate.standard_error;
            if error.abs() <= 1.96 * estimate.standard_error + EPS {
                self.covered += 1;
            }
        }
    }
    fn metrics(&self) -> Result<(f64, f64, f64, Option<f64>, Option<f64>), String> {
        if self.count == 0 {
            return Err("aggregate must contain at least one replicate".into());
        }
        let n = self.count as f64;
        let bias = self.sum_error / n;
        let mse = self.sum_squared_error / n;
        let variance = (mse - bias * bias).max(0.0);
        let (coverage, mean_width) = if self.interval_count == 0 {
            (None, None)
        } else {
            let interval_n = self.interval_count as f64;
            (Some(self.covered as f64 / interval_n), Some(self.sum_ci_width / interval_n))
        };
        if [bias, mse, variance].iter().any(|v| !v.is_finite())
            || coverage.is_some_and(|v| !v.is_finite())
            || mean_width.is_some_and(|v| !v.is_finite())
        {
            return Err("non-finite aggregate metric".into());
        }
        Ok((bias, variance, mse, coverage, mean_width))
    }
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1_u64 << 53) as f64)
    }
}

fn mixed_seed(scenario: Scenario, replicate: usize, stream: u64) -> u64 {
    0x6a09_e667_f3bc_c909_u64
        ^ scenario.index().wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (replicate as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)
        ^ stream.wrapping_mul(0x94d0_49bb_1331_11eb)
}

fn sample_dataset(
    n: usize,
    logger: Logger,
    context_probabilities: [f64; 2],
    rewards: [[f64; 2]; 2],
    seed: u64,
) -> Result<Vec<Observation>, String> {
    if n == 0
        || context_probabilities.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        || (context_probabilities[0] + context_probabilities[1] - 1.0).abs() > EPS
        || logger.p1.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        || rewards.iter().flatten().any(|r| !r.is_finite() || *r < 0.0 || *r > 1.0)
    {
        return Err("invalid sample size, context distribution, logger or Bernoulli means".into());
    }
    let mut rng = Rng::new(seed);
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        let context = if rng.uniform() < context_probabilities[0] { 0 } else { 1 };
        let p1 = logger.p1[context];
        let action = if rng.uniform() < p1 { 1 } else { 0 };
        let propensity = if action == 1 { p1 } else { 1.0 - p1 };
        let reward = if rng.uniform() < rewards[context][action] { 1.0 } else { 0.0 };
        rows.push(Observation { context, action, reward, propensity });
    }
    Ok(rows)
}

/// Beta(1,1)-smoothed empirical conditional reward means.
fn fit_reward_model(rows: &[Observation]) -> Result<[[f64; 2]; 2], String> {
    if rows.is_empty() {
        return Err("reward-model fitting requires nonempty data".into());
    }
    let mut successes = [[0.0_f64; 2]; 2];
    let mut counts = [[0.0_f64; 2]; 2];
    for row in rows {
        if row.context > 1 || row.action > 1
            || !row.reward.is_finite() || !(0.0..=1.0).contains(&row.reward)
        {
            return Err("invalid reward-model training observation".into());
        }
        successes[row.context][row.action] += row.reward;
        counts[row.context][row.action] += 1.0;
    }
    let mut model = [[0.5_f64; 2]; 2];
    for x in 0..2 {
        for a in 0..2 {
            model[x][a] = (successes[x][a] + 1.0) / (counts[x][a] + 2.0);
        }
    }
    Ok(model)
}

fn perturb_model(model: [[f64; 2]; 2], bias: [[f64; 2]; 2]) -> [[f64; 2]; 2] {
    let mut changed = model;
    for x in 0..2 {
        for a in 0..2 {
            changed[x][a] = (changed[x][a] + bias[x][a]).clamp(0.01, 0.99);
        }
    }
    changed
}

fn shrink_model(model: [[f64; 2]; 2], weight: f64) -> Result<[[f64; 2]; 2], String> {
    if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
        return Err("shrinkage weight must lie in [0,1]".into());
    }
    let mut shrunk = model;
    for x in 0..2 {
        let context_mean = 0.5 * (model[x][0] + model[x][1]);
        for a in 0..2 {
            shrunk[x][a] = ((1.0 - weight) * model[x][a] + weight * context_mean).clamp(0.0, 1.0);
        }
    }
    Ok(shrunk)
}

/// Paper-inspired empirical weight: w = 1 - Cov(q_hat,R)/Var(q_hat).
/// The caller is responsible for supplying genuinely held-out calibration data.
fn calibration_shrinkage_weight(
    model: [[f64; 2]; 2],
    calibration_rows: &[Observation],
) -> Result<f64, String> {
    if calibration_rows.len() < 2 {
        return Err("shrinkage calibration requires at least two held-out rows".into());
    }
    let predictions: Vec<f64> = calibration_rows.iter()
        .map(|r| model[r.context][r.action]).collect();
    let rewards: Vec<f64> = calibration_rows.iter().map(|r| r.reward).collect();
    let n = predictions.len() as f64;
    let mean_p = predictions.iter().sum::<f64>() / n;
    let mean_r = rewards.iter().sum::<f64>() / n;
    let variance = predictions.iter().map(|p| (p - mean_p).powi(2)).sum::<f64>() / n;
    if variance <= EPS {
        return Ok(0.0);
    }
    let covariance = predictions.iter().zip(rewards.iter())
        .map(|(p, r)| (p - mean_p) * (r - mean_r)).sum::<f64>() / n;
    let weight = (1.0 - covariance / variance).clamp(0.0, 1.0);
    if weight.is_finite() { Ok(weight) } else { Err("non-finite shrinkage weight".into()) }
}

/// Oracle reference: global shrinkage minimizing source-domain squared error
/// after pulling each context's two action means toward their common mean.
fn oracle_mse_shrinkage_weight(model: [[f64; 2]; 2], truth: [[f64; 2]; 2]) -> f64 {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for x in 0..2 {
        let context_mean = 0.5 * (model[x][0] + model[x][1]);
        for a in 0..2 {
            let centered = model[x][a] - context_mean;
            numerator += centered * (truth[x][a] - context_mean);
            denominator += centered * centered;
        }
    }
    if denominator <= EPS { 0.0 } else { (1.0 - numerator / denominator).clamp(0.0, 1.0) }
}

/// Target-aware Neyman logger: mu(a|x) is proportional to
/// pi_target(a|x) * sqrt(q_hat(a|x)), with a positive propensity floor.
fn logger_from_model(model: [[f64; 2]; 2], policy: Policy, floor: f64) -> Result<Logger, String> {
    if !floor.is_finite() || floor <= 0.0 || floor >= 0.5 {
        return Err("logger floor must lie in (0,0.5)".into());
    }
    let mut logger = Logger { p1: [0.5; 2] };
    for x in 0..2 {
        if model[x].iter().any(|q| !q.is_finite() || *q < 0.0 || *q > 1.0) {
            return Err("reward model must contain finite probabilities".into());
        }
        let pi1 = policy.p1[x];
        if !pi1.is_finite() || !(0.0..=1.0).contains(&pi1) {
            return Err("target policy must contain finite probabilities".into());
        }
        let action0_mass = (1.0 - pi1) * model[x][0].sqrt();
        let action1_mass = pi1 * model[x][1].sqrt();
        let total = action0_mass + action1_mass;
        if total <= EPS {
            // No reward information: retain target-action proportions before
            // applying the required exploration floor.
            logger.p1[x] = pi1.clamp(floor, 1.0 - floor);
        } else {
            logger.p1[x] = (action1_mass / total).clamp(floor, 1.0 - floor);
        }
    }
    Ok(logger)
}

fn target_value(policy: Policy, context_probabilities: [f64; 2], rewards: [[f64; 2]; 2]) -> f64 {
    let mut value = 0.0;
    for x in 0..2 {
        let p1 = policy.p1[x];
        value += context_probabilities[x]
            * ((1.0 - p1) * rewards[x][0] + p1 * rewards[x][1]);
    }
    value
}

fn logger_reward(logger: Logger, context_probabilities: [f64; 2], rewards: [[f64; 2]; 2]) -> f64 {
    let mut value = 0.0;
    for x in 0..2 {
        let p1 = logger.p1[x];
        value += context_probabilities[x]
            * ((1.0 - p1) * rewards[x][0] + p1 * rewards[x][1]);
    }
    value
}

fn validate_identification(policy: Policy, logger: Logger) -> Result<(), String> {
    for x in 0..2 {
        let pi1 = policy.p1[x];
        let mu1 = logger.p1[x];
        if !pi1.is_finite() || !(0.0..=1.0).contains(&pi1)
            || !mu1.is_finite() || !(0.0..=1.0).contains(&mu1)
        {
            return Err("invalid target or logging probability".into());
        }
        if (pi1 > EPS && mu1 <= EPS) || (pi1 < 1.0 - EPS && mu1 >= 1.0 - EPS) {
            return Err("unsupported target action: policy value is not identified".into());
        }
    }
    Ok(())
}

fn mean_and_se(scores: &[f64], interval_valid: bool) -> Result<Estimate, String> {
    if scores.is_empty() || scores.iter().any(|s| !s.is_finite()) {
        return Err("estimator requires nonempty finite scores".into());
    }
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    let variance = if scores.len() > 1 {
        scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0)
    } else {
        0.0
    };
    let standard_error = (variance.max(0.0) / n).sqrt();
    if mean.is_finite() && standard_error.is_finite() {
        Ok(Estimate { value: mean, standard_error, interval_valid })
    } else {
        Err("non-finite estimate".into())
    }
}

fn estimate(
    rows: &[Observation],
    policy: Policy,
    logger: Logger,
    estimator: Estimator,
    pilot_model: [[f64; 2]; 2],
) -> Result<Estimate, String> {
    if rows.is_empty() {
        return Err("evaluation data must be nonempty".into());
    }
    validate_identification(policy, logger)?;
    for row in rows {
        if row.context > 1 || row.action > 1 {
            return Err("invalid evaluation context or action".into());
        }
        let expected = if row.action == 1 { logger.p1[row.context] } else { 1.0 - logger.p1[row.context] };
        if !row.propensity.is_finite() || row.propensity <= 0.0
            || (row.propensity - expected).abs() > 1e-10
        {
            return Err("logged propensity does not match the registered logger".into());
        }
    }

    let full_model = match estimator {
        Estimator::DmInSample | Estimator::DrInSample => Some(fit_reward_model(rows)?),
        _ => None,
    };
    let crossfit_models = if estimator == Estimator::DrCrossfit2 {
        let fold0: Vec<Observation> = rows.iter().enumerate()
            .filter_map(|(i, row)| if i % 2 == 0 { Some(*row) } else { None }).collect();
        let fold1: Vec<Observation> = rows.iter().enumerate()
            .filter_map(|(i, row)| if i % 2 == 1 { Some(*row) } else { None }).collect();
        if fold0.is_empty() || fold1.is_empty() {
            return Err("two-fold cross-fitting requires both folds".into());
        }
        Some([fit_reward_model(&fold1)?, fit_reward_model(&fold0)?])
    } else {
        None
    };

    let mut scores = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let pi1 = policy.p1[row.context];
        let pi_action = if row.action == 1 { pi1 } else { 1.0 - pi1 };
        let score = match estimator {
            Estimator::Ips => pi_action * row.reward / row.propensity,
            Estimator::DmPilot => {
                (1.0 - pi1) * pilot_model[row.context][0] + pi1 * pilot_model[row.context][1]
            }
            Estimator::DrPilot => {
                let q = pilot_model[row.context][row.action];
                (1.0 - pi1) * pilot_model[row.context][0]
                    + pi1 * pilot_model[row.context][1]
                    + pi_action / row.propensity * (row.reward - q)
            }
            Estimator::DmInSample => {
                let q = full_model.expect("full model requested")[row.context];
                (1.0 - pi1) * q[0] + pi1 * q[1]
            }
            Estimator::DrInSample => {
                let q = full_model.expect("full model requested");
                let dm = (1.0 - pi1) * q[row.context][0] + pi1 * q[row.context][1];
                dm + pi_action / row.propensity * (row.reward - q[row.context][row.action])
            }
            Estimator::DrCrossfit2 => {
                let q = crossfit_models.expect("cross-fit models requested")[i % 2];
                let dm = (1.0 - pi1) * q[row.context][0] + pi1 * q[row.context][1];
                dm + pi_action / row.propensity * (row.reward - q[row.context][row.action])
            }
        };
        scores.push(score);
    }
    let interval_valid = !matches!(estimator, Estimator::DmPilot | Estimator::DmInSample | Estimator::DrInSample);
    mean_and_se(&scores, interval_valid)
}

fn benchmark(replicates: usize) -> Result<String, String> {
    if replicates == 0 {
        return Err("replicate count must be positive".into());
    }
    let mut output = String::from(
        "record_type\tscenario\tN\tlogger\tpolicy\testimator\tshrinkage_weight\tlogger_reward\tmean_min_propensity\treplicates\tbias\tvariance\tmse\tcoverage_95\tmean_ci_width\tidentification_status\tinference_status\n",
    );
    output.push_str("BENCHMARK_CONFIG\tVIS-DVPC-053\tversion\t0\n");
    output.push_str(&format!("BENCHMARK_CONFIG\treplicates\t{replicates}\n"));
    output.push_str(&format!("BENCHMARK_CONFIG\tcalibration_n\t{CALIBRATION_N}\n"));
    output.push_str(&format!("BENCHMARK_CONFIG\tpropensity_floor\t{ACTION_FLOOR:.6}\n"));
    output.push_str("BENCHMARK_CONFIG\tshrinkage\tpaper-inspired covariance weight; held-out and deliberately leaky variants\n");
    output.push_str("BENCHMARK_CONFIG\testimators\tIPS,DM,DR; known logging propensities\n");
    output.push_str("BENCHMARK_CONFIG\tcrossfit\t2-fold; nuisance fit excludes the scored fold\n");
    output.push_str("BENCHMARK_CONFIG\tdgp\t2 contexts, 2 actions, Bernoulli rewards; exact source and target means\n");

    for scenario in Scenario::ALL {
        let context_probabilities = scenario.eval_context();
        let eval_rewards = scenario.eval_rewards();
        let pilot_n = scenario.pilot_n();

        for sample_size in SAMPLE_SIZES {
            for logger_kind in LoggerKind::ALL {
                for policy in TARGET_POLICIES {
                    let mut aggregates: [Aggregate; 6] = std::array::from_fn(|_| Aggregate::default());
                    let mut mean_w = 0.0;
                    let mut mean_reward = 0.0;
                    let mut mean_min_propensity = 0.0;

                    for replicate in 0..replicates {
                        let pilot_seed = mixed_seed(scenario, replicate, 1);
                        let pilot = sample_dataset(
                            pilot_n,
                            Logger { p1: [0.5, 0.5] },
                            [0.5, 0.5],
                            SOURCE_REWARD,
                            pilot_seed,
                        )?;
                        let fitted = perturb_model(fit_reward_model(&pilot)?, scenario.predictor_bias());
                        let calibration_seed = mixed_seed(scenario, replicate, 2);
                        let calibration = sample_dataset(
                            CALIBRATION_N,
                            Logger { p1: [0.5, 0.5] },
                            [0.5, 0.5],
                            SOURCE_REWARD,
                            calibration_seed,
                        )?;
                        let heldout_w = calibration_shrinkage_weight(fitted, &calibration)?;
                        let leaky_w = calibration_shrinkage_weight(fitted, &pilot)?;
                        let oracle_w = oracle_mse_shrinkage_weight(fitted, SOURCE_REWARD);
                        let weight = match logger_kind {
                            LoggerKind::Uniform | LoggerKind::RawPilot => 0.0,
                            LoggerKind::HeldoutShrinkage => heldout_w,
                            LoggerKind::LeakyShrinkage => leaky_w,
                            LoggerKind::OracleMseShrinkage => oracle_w,
                        };
                        let q_model = shrink_model(fitted, weight)?;
                        let logger = if logger_kind == LoggerKind::Uniform {
                            Logger { p1: [0.5, 0.5] }
                        } else {
                            logger_from_model(q_model, policy, ACTION_FLOOR)?
                        };
                        let evaluation_seed = mixed_seed(
                            scenario,
                            replicate,
                            100 + sample_size as u64 + logger_kind.as_u64_index(),
                        );
                        let rows = sample_dataset(
                            sample_size,
                            logger,
                            context_probabilities,
                            eval_rewards,
                            evaluation_seed,
                        )?;
                        let truth = target_value(policy, context_probabilities, eval_rewards);
                        let realized_logger_reward = logger_reward(logger, context_probabilities, eval_rewards);
                        let min_propensity = logger.p1.iter()
                            .map(|p| p.min(1.0 - p))
                            .fold(1.0_f64, f64::min);

                        for (index, estimator) in Estimator::ALL.iter().enumerate() {
                            let result = estimate(&rows, policy, logger, *estimator, q_model)?;
                            aggregates[index].add(result, truth);
                            let error = result.value - truth;
                            let (ci_lower, ci_upper, inference_status) = if result.interval_valid {
                                (
                                    format!("{:.10}", result.value - 1.96 * result.standard_error),
                                    format!("{:.10}", result.value + 1.96 * result.standard_error),
                                    "score_based_normal_interval_empirical_only",
                                )
                            } else if *estimator == Estimator::DmPilot {
                                ("NA".to_string(), "NA".to_string(), "model_uncertainty_not_in_score_ci")
                            } else {
                                ("NA".to_string(), "NA".to_string(), "same_sample_nuisance_ci_unavailable")
                            };
                            output.push_str(&format!(
                                "REPLICATE\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{:.10}\t{}\t{}\tidentified_full_support\t{}\n",
                                scenario.name(),
                                sample_size,
                                logger_kind.name(),
                                policy.id,
                                estimator.name(),
                                replicate,
                                pilot_seed,
                                calibration_seed,
                                evaluation_seed,
                                result.value,
                                truth,
                                error,
                                error * error,
                                result.standard_error,
                                ci_lower,
                                ci_upper,
                                inference_status,
                            ));
                        }
                        mean_w += weight;
                        mean_reward += realized_logger_reward;
                        mean_min_propensity += min_propensity;
                    }

                    let nrep = replicates as f64;
                    for (index, estimator) in Estimator::ALL.iter().enumerate() {
                        let (bias, variance, mse, coverage, width) = aggregates[index].metrics()?;
                        let coverage_text = coverage.map(|v| format!("{v:.6}")).unwrap_or_else(|| "NA".to_string());
                        let width_text = width.map(|v| format!("{v:.8}")).unwrap_or_else(|| "NA".to_string());
                        let inference_status = match estimator {
                            Estimator::DmPilot => "model_uncertainty_not_in_score_ci",
                            Estimator::DmInSample | Estimator::DrInSample => "same_sample_nuisance_ci_unavailable",
                            _ => "score_based_normal_interval_empirical_only",
                        };
                        output.push_str(&format!(
                            "RESULT\t{}\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{}\t{:.10}\t{:.10}\t{:.10}\t{}\t{}\tidentified_full_support\t{}\n",
                            scenario.name(),
                            sample_size,
                            logger_kind.name(),
                            policy.id,
                            estimator.name(),
                            mean_w / nrep,
                            mean_reward / nrep,
                            mean_min_propensity / nrep,
                            replicates,
                            bias,
                            variance,
                            mse,
                            coverage_text,
                            width_text,
                            inference_status,
                        ));
                    }
                }
            }
        }
    }
    Ok(output)
}

fn main() {
    let smoke = env::args().any(|arg| arg == "--smoke");
    let replicates = if smoke { SMOKE_REPLICATES } else { FULL_REPLICATES };
    match benchmark(replicates) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_053_ERROR\t{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_model_dr_expectation_equals_policy_value() {
        let policy = Policy { id: 99, p1: [0.83, 0.17] };
        let logger = Logger { p1: [0.31, 0.72] };
        let q = [[0.95, 0.05], [0.10, 0.90]];
        for context in 0..2 {
            let pi1 = policy.p1[context];
            let mut expectation = 0.0;
            for action in 0..2 {
                let mu = if action == 1 { logger.p1[context] } else { 1.0 - logger.p1[context] };
                let pi = if action == 1 { pi1 } else { 1.0 - pi1 };
                let dm = (1.0 - pi1) * q[context][0] + pi1 * q[context][1];
                let dr_score = dm + pi / mu * (SOURCE_REWARD[context][action] - q[context][action]);
                expectation += mu * dr_score;
            }
            let truth = (1.0 - pi1) * SOURCE_REWARD[context][0] + pi1 * SOURCE_REWARD[context][1];
            assert!((expectation - truth).abs() < 1e-12);
        }
    }

    #[test]
    fn model_logger_uses_target_weighted_neyman_allocation() {
        let model = [[0.25, 1.0], [1.0, 0.25]];
        let policy = Policy { id: 17, p1: [0.8, 0.2] };
        let logger = logger_from_model(model, policy, ACTION_FLOOR).unwrap();
        assert!((logger.p1[0] - (0.8 / 0.9)).abs() < 1e-12);
        assert!((logger.p1[1] - (0.1 / 0.9)).abs() < 1e-12);
    }

    #[test]
    fn support_absence_fails_closed_instead_of_returning_a_value() {
        let policy = Policy { id: 0, p1: [0.8, 0.5] };
        let logger = Logger { p1: [0.0, 0.5] };
        assert!(validate_identification(policy, logger).is_err());
    }

    #[test]
    fn empirical_shrinkage_weights_are_bounded_and_shrinkage_preserves_probability_range() {
        let rows = sample_dataset(64, Logger { p1: [0.5, 0.5] }, [0.5, 0.5], SOURCE_REWARD, 17).unwrap();
        let calibration = sample_dataset(64, Logger { p1: [0.5, 0.5] }, [0.5, 0.5], SOURCE_REWARD, 18).unwrap();
        let model = fit_reward_model(&rows).unwrap();
        let w = calibration_shrinkage_weight(model, &calibration).unwrap();
        assert!((0.0..=1.0).contains(&w));
        let shrunk = shrink_model(model, w).unwrap();
        assert!(shrunk.iter().flatten().all(|q| q.is_finite() && (0.0..=1.0).contains(q)));
    }

    #[test]
    fn crossfit_runs_with_two_disjoint_training_folds() {
        let rows = sample_dataset(64, Logger { p1: [0.5, 0.5] }, [0.5, 0.5], SOURCE_REWARD, 19).unwrap();
        let policy = TARGET_POLICIES[0];
        let result = estimate(&rows, policy, Logger { p1: [0.5, 0.5] }, Estimator::DrCrossfit2, [[0.5; 2]; 2]).unwrap();
        assert!(result.value.is_finite());
        assert!(result.standard_error.is_finite());
        assert!(result.interval_valid);

        let pilot_dm = estimate(
            &rows,
            policy,
            Logger { p1: [0.5, 0.5] },
            Estimator::DmPilot,
            [[0.5; 2]; 2],
        ).unwrap();
        assert!(!pilot_dm.interval_valid);

        let in_sample_dm = estimate(
            &rows,
            policy,
            Logger { p1: [0.5, 0.5] },
            Estimator::DmInSample,
            [[0.5; 2]; 2],
        ).unwrap();
        let in_sample_dr = estimate(
            &rows,
            policy,
            Logger { p1: [0.5, 0.5] },
            Estimator::DrInSample,
            [[0.5; 2]; 2],
        ).unwrap();
        assert!(!in_sample_dm.interval_valid);
        assert!(!in_sample_dr.interval_valid);

        let pilot_dr = estimate(
            &rows,
            policy,
            Logger { p1: [0.5, 0.5] },
            Estimator::DrPilot,
            [[0.5; 2]; 2],
        ).unwrap();
        assert!(pilot_dr.interval_valid);
    }

    #[test]
    fn deterministic_output_has_exact_registered_row_count_and_schema() {
        let a = benchmark(3).unwrap();
        let b = benchmark(3).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.lines().filter(|line| line.starts_with("RESULT\t")).count(), 5 * 2 * 5 * 4 * 6);
        assert_eq!(a.lines().filter(|line| line.starts_with("REPLICATE\t")).count(), 3 * 5 * 2 * 5 * 4 * 6);
        for line in a.lines().filter(|line| line.starts_with("RESULT\t")) {
            assert_eq!(line.split('\t').count(), 17);
            assert!(line.contains("identified_full_support"));
        }
        for line in a.lines().filter(|line| line.starts_with("REPLICATE\t")) {
            assert_eq!(line.split('\t').count(), 19);
            assert!(line.contains("identified_full_support"));
        }
        assert_eq!(a.lines().filter(|line| line.starts_with("BENCHMARK_CONFIG\t")).count(), 8);
    }

    #[test]
    fn invalid_shrinkage_inputs_and_model_inputs_fail_closed() {
        assert!(shrink_model([[0.5; 2]; 2], f64::NAN).is_err());
        assert!(logger_from_model([[f64::NAN, 0.5], [0.2, 0.8]], TARGET_POLICIES[0], ACTION_FLOOR).is_err());
        assert!(calibration_shrinkage_weight([[0.5; 2]; 2], &[]).is_err());
        assert!(sample_dataset(0, Logger { p1: [0.5, 0.5] }, [0.5, 0.5], SOURCE_REWARD, 1).is_err());
    }
}