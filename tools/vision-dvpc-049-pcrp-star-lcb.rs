//! VIS-DVPC-049: penalized best-CRP lower confidence bound for nonnegative IPS rewards.
//!
//! Implements the pCRP-star LCB defined in Section 3.1 of Ryu et al. (COLT 2025).
//! The bound uses a conservative feasible-wealth witness for the one-dimensional log-wealth
//! maximization. It is a standalone comparator, not the complete COLT experiment suite.
//!
//! Evidence contract:
//! - fixed behavior policy and stationary IID action/reward draws in the synthetic DGP;
//! - known propensities and complete target-policy support;
//! - nonnegative importance-weighted reward scores;
//! - familywise allocation delta/M across the finite candidate portfolio;
//! - structural zero support suppresses all numeric policy-value outputs.
#![deny(unsafe_code)]

use std::env;
use std::fmt::Write as _;
use std::time::Instant;

const ALPHA: f64 = 0.05;
const POLICY_P1: [f64; 3] = [0.10, 0.50, 0.90];
const MU0: f64 = 0.25;
const MU1: f64 = 0.75;
const GOLDEN_SEED: u64 = 0x49_50_43_57_2A_01_10_29;
const ROOT_ITERS: usize = 36;
const BET_GRID_STEPS: usize = 128;
const EPS: f64 = 1.0e-10;
const METHODS: [&str; 2] = ["pCRP_star", "Hoeffding_fixed_horizon"];

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
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    name: &'static str,
    logger_p1: f64,
}

const SCENARIOS: [Scenario; 3] = [
    Scenario { name: "healthy_logger", logger_p1: 0.50 },
    Scenario { name: "weak_overlap_logger", logger_p1: 0.99 },
    Scenario { name: "zero_support_logger", logger_p1: 1.00 },
];

#[derive(Clone, Copy, Debug)]
struct Observation {
    action: usize,
    reward: f64,
}

fn seed_for(rep: usize, scenario_index: usize) -> u64 {
    GOLDEN_SEED
        ^ (rep as u64).wrapping_mul(0xA076_1D64_78BD_642F)
        ^ (scenario_index as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB)
}

fn true_value(target_p1: f64) -> f64 {
    (1.0 - target_p1) * MU0 + target_p1 * MU1
}

fn support_complete(logger_p1: f64, target_p1: f64) -> bool {
    let logger_p0 = 1.0 - logger_p1;
    let target_p0 = 1.0 - target_p1;
    !(logger_p0 == 0.0 && target_p0 > 0.0)
        && !(logger_p1 == 0.0 && target_p1 > 0.0)
}

fn max_importance_weight(logger_p1: f64, target_p1: f64) -> Result<f64, String> {
    if !logger_p1.is_finite()
        || !target_p1.is_finite()
        || !(0.0..=1.0).contains(&logger_p1)
        || !(0.0..=1.0).contains(&target_p1)
    {
        return Err("probabilities must be finite and in [0,1]".into());
    }
    if !support_complete(logger_p1, target_p1) {
        return Err("NOT_IDENTIFIED_ZERO_SUPPORT".into());
    }

    let mut max_weight: f64 = 0.0;
    let logger_p0 = 1.0 - logger_p1;
    let target_p0 = 1.0 - target_p1;
    if target_p0 > 0.0 {
        max_weight = max_weight.max(target_p0 / logger_p0);
    }
    if target_p1 > 0.0 {
        max_weight = max_weight.max(target_p1 / logger_p1);
    }
    if !max_weight.is_finite() {
        return Err("non-finite importance-weight bound".into());
    }
    Ok(max_weight)
}

fn draw_stream(scenario: Scenario, n: usize, rng: &mut SplitMix64) -> Vec<Observation> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let action = if rng.unit() < scenario.logger_p1 { 1 } else { 0 };
        let mean = if action == 1 { MU1 } else { MU0 };
        let reward = if rng.unit() < mean { 1.0 } else { 0.0 };
        out.push(Observation { action, reward });
    }
    out
}

fn policy_score_stream(
    observations: &[Observation],
    logger_p1: f64,
    target_p1: f64,
) -> Result<Vec<f64>, String> {
    if !support_complete(logger_p1, target_p1) {
        return Err("NOT_IDENTIFIED_ZERO_SUPPORT".into());
    }
    let logger = [1.0 - logger_p1, logger_p1];
    let target = [1.0 - target_p1, target_p1];
    let mut scores = Vec::with_capacity(observations.len());
    for obs in observations {
        let h = logger[obs.action];
        let pi = target[obs.action];
        if h <= 0.0 && pi > 0.0 {
            return Err("NOT_IDENTIFIED_ZERO_SUPPORT".into());
        }
        let score = if pi == 0.0 { 0.0 } else { (pi / h) * obs.reward };
        if !score.is_finite() || score < 0.0 {
            return Err("importance-weighted score must be finite and nonnegative".into());
        }
        scores.push(score);
    }
    Ok(scores)
}

fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let hi = a.max(b);
    let lo = a.min(b);
    if hi == f64::INFINITY {
        return hi;
    }
    hi + (lo - hi).exp().ln_1p()
}

fn log_crp_wealth(values: &[f64], candidate_mean: f64, bet: f64) -> f64 {
    if bet <= 0.0 {
        return 0.0;
    }
    if bet >= 1.0 {
        if values.iter().any(|&y| y == 0.0) {
            return f64::NEG_INFINITY;
        }
        return values.iter().map(|y| y.ln() - candidate_mean.ln()).sum();
    }

    let log_cash = (-bet).ln_1p();
    let log_bet = bet.ln();
    let log_mean = candidate_mean.ln();
    let mut total = 0.0;
    for &y in values {
        let log_risky = if y == 0.0 {
            f64::NEG_INFINITY
        } else {
            log_bet + y.ln() - log_mean
        };
        let term = log_add_exp(log_cash, log_risky);
        if !term.is_finite() {
            return term;
        }
        total += term;
    }
    total
}

/// Monotone lower witness for sup_{b in [0,1]} log CRP wealth.
///
/// The candidate bet grid is fixed independently of candidate_mean. Each CRP
/// log wealth is nonincreasing in candidate_mean, and the maximum over this fixed
/// grid is therefore also nonincreasing. The grid maximum is no larger than the
/// continuous supremum, so inversion remains conservative though possibly looser.
fn log_sup_crp_wealth_lower(values: &[f64], candidate_mean: f64) -> Result<f64, String> {
    if values.is_empty() || !candidate_mean.is_finite() || candidate_mean <= 0.0 {
        return Err("need a nonempty stream and a positive finite candidate mean".into());
    }
    if values.iter().any(|y| !y.is_finite() || *y < 0.0) {
        return Err("scores must be finite and nonnegative".into());
    }
    let sum: f64 = values.iter().sum();
    if !sum.is_finite() {
        return Err("score sum overflowed".into());
    }
    let mean = sum / values.len() as f64;
    if candidate_mean >= mean {
        // For a nonnegative process, b=0 attains the exact supremum of log wealth.
        return Ok(0.0);
    }

    // b=0 is included, so best is nonnegative; the grid is fixed for every candidate mean.
    let mut best = 0.0_f64;
    for i in 1..=BET_GRID_STEPS {
        let bet = i as f64 / BET_GRID_STEPS as f64;
        best = best.max(log_crp_wealth(values, candidate_mean, bet));
    }
    let roundoff_pad =
        64.0 * f64::EPSILON * (1.0 + best.abs() + values.len() as f64);
    // Returning zero when the grid maximum is numerically indistinguishable from zero
    // preserves monotonicity at the empirical-mean boundary and remains a valid witness.
    if best <= roundoff_pad {
        Ok(0.0)
    } else {
        Ok(best - roundoff_pad)
    }
}

/// pCRP-star lower confidence bound from Ryu et al. (COLT 2025), Eq. (3.3)-(3.5).
///
/// The pCRP-star wealth is (pi*(n+1))^(-1/2) times the best constant-bet CRP wealth.
/// Using an achieved feasible wealth as a lower bound on the supremum makes the inversion conservative.
fn pcrp_star_lcb(values: &[f64], delta: f64) -> Result<f64, String> {
    if values.is_empty() {
        return Err("confidence bound requires at least one observation".into());
    }
    if !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
        return Err("delta must lie strictly between zero and one".into());
    }
    if values.iter().any(|y| !y.is_finite() || *y < 0.0) {
        return Err("scores must be finite and nonnegative".into());
    }
    let sum: f64 = values.iter().sum();
    if !sum.is_finite() {
        return Err("score sum overflowed".into());
    }
    let mean = sum / values.len() as f64;
    if mean == 0.0 {
        return Ok(0.0);
    }
    let log_threshold = -delta.ln()
        + 0.5 * (std::f64::consts::PI * (values.len() as f64 + 1.0)).ln();

    // mean is always in the accepted region: the derivative at b=0 is nonpositive.
    let mut hi = mean;
    let mut probe = mean * 0.5;
    let mut lo: Option<f64> = None;
    for _ in 0..1075 {
        if probe <= 0.0 {
            break;
        }
        let log_wealth = log_sup_crp_wealth_lower(values, probe)?;
        if log_wealth > log_threshold {
            lo = Some(probe);
            break;
        }
        hi = probe;
        probe *= 0.5;
    }
    let Some(mut low) = lo else {
        // If the crossing falls below the representable positive range, zero is conservative.
        return Ok(0.0);
    };

    for _ in 0..ROOT_ITERS {
        let mid = low + (hi - low) * 0.5;
        if mid == low || mid == hi {
            break;
        }
        let log_wealth = log_sup_crp_wealth_lower(values, mid)?;
        if log_wealth > log_threshold {
            low = mid;
        } else {
            hi = mid;
        }
    }
    Ok(hi.max(0.0).min(mean))
}

fn hoeffding_lcb(values: &[f64], delta: f64, known_upper_bound: f64) -> Result<f64, String> {
    if values.is_empty() || !delta.is_finite() || delta <= 0.0 || delta >= 1.0 {
        return Err("invalid Hoeffding-bound input".into());
    }
    if !known_upper_bound.is_finite() || known_upper_bound < 0.0 {
        return Err("known score bound must be finite and nonnegative".into());
    }
    if values.iter().any(|y| !y.is_finite() || *y < 0.0 || *y > known_upper_bound + EPS) {
        return Err("score violates the declared Hoeffding range".into());
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let radius = known_upper_bound * ((1.0 / delta).ln() / (2.0 * values.len() as f64)).sqrt();
    Ok((mean - radius).max(0.0))
}

#[derive(Clone, Copy, Debug, Default)]
struct RunningMean {
    n: usize,
    mean: f64,
    m2: f64,
}

impl RunningMean {
    fn push(&mut self, x: f64) {
        self.n += 1;
        let delta = x - self.mean;
        self.mean += delta / self.n as f64;
        self.m2 += delta * (x - self.mean);
    }

    fn standard_error(self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            ((self.m2 / (self.n - 1) as f64) / self.n as f64).max(0.0).sqrt()
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Aggregate {
    n: usize,
    familywise_covered: usize,
    top1: usize,
    regret: RunningMean,
    selected_lcb: RunningMean,
    selected_lcb_gap: RunningMean,
}

fn rate_se(successes: usize, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let p = successes as f64 / n as f64;
    (p * (1.0 - p) / n as f64).sqrt()
}

fn metric_line(
    out: &mut String,
    scenario: &str,
    horizon: usize,
    method: &str,
    metric: &str,
    value: f64,
    se: f64,
) {
    let _ = writeln!(
        out,
        "RESULT_METRIC\t{scenario}\t{horizon}\t{method}\t{metric}\t{value:.12}\t{se:.12}"
    );
}

fn run(replicates: usize, horizons: &[usize]) -> Result<String, String> {
    if replicates == 0 || horizons.is_empty() || horizons.iter().any(|n| *n == 0) {
        return Err("replicates and horizons must be positive".into());
    }
    if !horizons.windows(2).all(|w| w[0] < w[1]) {
        return Err("horizons must be strictly increasing".into());
    }

    let max_horizon = *horizons.last().unwrap();
    let mut out = String::from(
        "record_type\tscenario\thorizon_or_key\tmethod_or_status\tmetric_or_value\tvalue\tstandard_error\n",
    );
    let _ = writeln!(out, "CONFIG\treplicates\t{replicates}");
    let _ = writeln!(out, "CONFIG\talpha\t{ALPHA:.8}");
    let _ = writeln!(out, "CONFIG\tpolicies\t{}", POLICY_P1.len());
    let _ = writeln!(out, "CONFIG\tseed\t{GOLDEN_SEED}");
    let _ = writeln!(out, "CONFIG\thorizons\t{}", horizons.iter().map(usize::to_string).collect::<Vec<_>>().join(","));
    let _ = writeln!(out, "CONFIG\tbet_optimizer\tfixed_bet_grid_129_point_lower_witness");
    let _ = writeln!(out, "CONFIG\treward_means\ta0={MU0:.4},a1={MU1:.4}");

    for (scenario_index, scenario) in SCENARIOS.iter().enumerate() {
        let identified = POLICY_P1
            .iter()
            .all(|&p| support_complete(scenario.logger_p1, p));
        let status = if identified { "IDENTIFIED" } else { "NOT_IDENTIFIED_ZERO_SUPPORT" };
        let _ = writeln!(
            out,
            "FIXTURE\t{}\tstatus\t{}\tlogger_p1={:.6}\t-\t-",
            scenario.name, status, scenario.logger_p1
        );
        if !identified {
            let _ = writeln!(
                out,
                "SUPPRESSED\t{}\tall_horizons\tall_methods\tunsupported_target_action\t-\t-",
                scenario.name
            );
            continue;
        }

        let mut aggregates = vec![vec![Aggregate::default(); METHODS.len()]; horizons.len()];
        let mut elapsed_seconds = vec![vec![0.0_f64; METHODS.len()]; horizons.len()];
        let delta_per_policy = ALPHA / POLICY_P1.len() as f64;
        let true_values: Vec<f64> = POLICY_P1.iter().map(|&p| true_value(p)).collect();
        let oracle_value = true_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let oracle_index = true_values
            .iter()
            .position(|v| (*v - oracle_value).abs() <= EPS)
            .unwrap();

        for rep in 0..replicates {
            let mut rng = SplitMix64::new(seed_for(rep, scenario_index));
            let observations = draw_stream(*scenario, max_horizon, &mut rng);
            let streams: Vec<Vec<f64>> = POLICY_P1
                .iter()
                .map(|&p| policy_score_stream(&observations, scenario.logger_p1, p))
                .collect::<Result<Vec<_>, _>>()?;

            for (horizon_index, &horizon) in horizons.iter().enumerate() {
                for (method_index, method) in METHODS.iter().enumerate() {
                    let started = Instant::now();
                    let mut lower_bounds = Vec::with_capacity(POLICY_P1.len());
                    for (policy_index, &target_p1) in POLICY_P1.iter().enumerate() {
                        let scores = &streams[policy_index][..horizon];
                        let lower = match *method {
                            "pCRP_star" => pcrp_star_lcb(scores, delta_per_policy)?,
                            "Hoeffding_fixed_horizon" => {
                                let upper = max_importance_weight(scenario.logger_p1, target_p1)?;
                                hoeffding_lcb(scores, delta_per_policy, upper)?
                            }
                            _ => return Err("unknown method".into()),
                        };
                        lower_bounds.push(lower);
                    }
                    elapsed_seconds[horizon_index][method_index] += started.elapsed().as_secs_f64();

                    let chosen = (1..lower_bounds.len()).fold(0, |best, i| {
                        if lower_bounds[i] > lower_bounds[best] { i } else { best }
                    });
                    let familywise_covered = lower_bounds
                        .iter()
                        .zip(true_values.iter())
                        .all(|(lower, truth)| *lower <= *truth + EPS);
                    let ag = &mut aggregates[horizon_index][method_index];
                    ag.n += 1;
                    if familywise_covered {
                        ag.familywise_covered += 1;
                    }
                    if chosen == oracle_index {
                        ag.top1 += 1;
                    }
                    let regret = oracle_value - true_values[chosen];
                    ag.regret.push(regret);
                    ag.selected_lcb.push(lower_bounds[chosen]);
                    ag.selected_lcb_gap.push(true_values[chosen] - lower_bounds[chosen]);
                }
            }
        }

        for (horizon_index, &horizon) in horizons.iter().enumerate() {
            for (method_index, method) in METHODS.iter().enumerate() {
                let ag = aggregates[horizon_index][method_index];
                let count = ag.n as f64;
                if ag.n == 0 {
                    return Err("empty aggregate".into());
                }
                let coverage = ag.familywise_covered as f64 / count;
                let top1 = ag.top1 as f64 / count;
                metric_line(
                    &mut out, scenario.name, horizon, method, "familywise_lcb_coverage",
                    coverage, rate_se(ag.familywise_covered, ag.n),
                );
                metric_line(
                    &mut out, scenario.name, horizon, method, "top1_policy_selection_accuracy",
                    top1, rate_se(ag.top1, ag.n),
                );
                metric_line(
                    &mut out, scenario.name, horizon, method, "mean_simple_regret",
                    ag.regret.mean, ag.regret.standard_error(),
                );
                metric_line(
                    &mut out, scenario.name, horizon, method, "mean_selected_lcb",
                    ag.selected_lcb.mean, ag.selected_lcb.standard_error(),
                );
                metric_line(
                    &mut out, scenario.name, horizon, method, "mean_selected_truth_minus_lcb",
                    ag.selected_lcb_gap.mean, ag.selected_lcb_gap.standard_error(),
                );
                let _ = writeln!(
                    out,
                    "SYSTEM\t{}\t{}\t{}\telapsed_seconds\t{:.6}\t-",
                    scenario.name, horizon, method, elapsed_seconds[horizon_index][method_index]
                );
            }
        }
    }
    Ok(out)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("--smoke") => run(4, &[8, 16]),
        Some("--benchmark") => run(32, &[64, 256, 512]),
        _ => Err("usage: vision-dvpc-049-pcrp-star-lcb --smoke|--benchmark".into()),
    };
    match result {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_049_ERROR\t{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tolerance: f64) {
        assert!(
            (a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs())),
            "actual={a:.15}, expected={b:.15}, tolerance={tolerance}"
        );
    }

    #[test]
    fn constant_positive_stream_matches_analytic_pcrp_star_root() {
        let n = 12;
        let y = 0.75;
        let delta = 0.05;
        let values = vec![y; n];
        let expected = y * (delta / (std::f64::consts::PI * (n as f64 + 1.0)).sqrt())
            .powf(1.0 / n as f64);
        let actual = pcrp_star_lcb(&values, delta).unwrap();
        close(actual, expected, 2.0e-6);
    }

    #[test]
    fn zero_rewards_return_zero_lcb() {
        assert_eq!(pcrp_star_lcb(&[0.0, 0.0, 0.0, 0.0], 0.05).unwrap(), 0.0);
    }

    #[test]
    fn lcb_is_nonnegative_and_no_larger_than_empirical_mean() {
        let values = [0.0, 0.0, 0.5, 1.0, 1.5, 0.0, 0.2];
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let lcb = pcrp_star_lcb(&values, 0.02).unwrap();
        assert!(lcb >= 0.0);
        assert!(lcb <= mean + 1.0e-9);
    }

    #[test]
    fn feasible_wealth_witness_does_not_exceed_dense_grid_supremum() {
        let values = [0.0, 0.2, 0.8, 1.5, 0.7];
        let nu = 0.4;
        let witness = log_sup_crp_wealth_lower(&values, nu).unwrap();
        let mut grid_best = f64::NEG_INFINITY;
        for i in 0..=10_000 {
            let b = i as f64 / 10_000.0;
            grid_best = grid_best.max(log_crp_wealth(&values, nu, b));
        }
        assert!(
            witness <= grid_best + 1.0e-8,
            "feasible witness {witness} exceeded dense-grid supremum {grid_best}"
        );
    }

    #[test]
    fn fixed_grid_witness_is_monotone_in_candidate_mean() {
        let values = [0.0, 0.2, 0.8, 1.5, 0.7];
        let mut previous = f64::INFINITY;
        for i in 1..=1000 {
            let candidate_mean = i as f64 / 1000.0;
            let current = log_sup_crp_wealth_lower(&values, candidate_mean).unwrap();
            assert!(
                current <= previous + 1.0e-9,
                "witness increased: previous={previous}, current={current}, nu={candidate_mean}"
            );
            previous = current;
        }
    }

    #[test]
    fn endpoint_witness_is_below_the_best_endpoint_wealth() {
        let values = [0.5, 0.7, 0.9];
        let nu = 0.1;
        let witness = log_sup_crp_wealth_lower(&values, nu).unwrap();
        let endpoint = values.iter().map(|y| y.ln() - nu.ln()).sum::<f64>();
        assert!(witness <= endpoint + 1.0e-9);
    }

    #[test]
    fn support_gate_suppresses_zero_support_policy_values() {
        assert!(!support_complete(1.0, 0.9));
        assert!(!support_complete(0.0, 0.1));
        assert!(support_complete(0.99, 0.9));
        assert_eq!(
            max_importance_weight(1.0, 0.9).unwrap_err(),
            "NOT_IDENTIFIED_ZERO_SUPPORT"
        );
    }

    #[test]
    fn bounded_baseline_rejects_out_of_range_scores() {
        assert!(hoeffding_lcb(&[0.1, 1.1], 0.05, 1.0).is_err());
        assert!(hoeffding_lcb(&[0.1, 0.2], 0.0, 1.0).is_err());
    }

    #[test]
    fn invalid_scores_fail_closed() {
        assert!(pcrp_star_lcb(&[], 0.05).is_err());
        assert!(pcrp_star_lcb(&[-0.1, 0.2], 0.05).is_err());
        assert!(pcrp_star_lcb(&[f64::NAN, 0.2], 0.05).is_err());
        assert!(pcrp_star_lcb(&[0.1, 0.2], 1.0).is_err());
    }

    #[test]
    fn policy_value_identity_is_exact_under_the_frozen_logger() {
        for &logger in &[0.5_f64, 0.99] {
            for &target in &POLICY_P1 {
                let h = [1.0 - logger, logger];
                let pi = [1.0 - target, target];
                let expectation = h[0] * (pi[0] / h[0]) * MU0
                    + h[1] * (pi[1] / h[1]) * MU1;
                close(expectation, true_value(target), 1.0e-12);
            }
        }
    }

    #[test]
    fn smoke_run_is_deterministic_after_timing_rows_are_removed() {
        let a = run(3, &[8, 16]).unwrap();
        let b = run(3, &[8, 16]).unwrap();
        let stable = |s: &str| {
            s.lines()
                .filter(|line| !line.starts_with("SYSTEM\t"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(stable(&a), stable(&b));
    }
}
