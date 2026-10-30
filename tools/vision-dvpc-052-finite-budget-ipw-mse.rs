//! VIS-DVPC-052: finite-budget IPW MSE with explicit support truncation.
//!
//! This exact synthetic benchmark compares full-support logging against a
//! support-truncating design. It reports the finite-N squared bias and variance
//! separately and never treats a support-truncated policy as identified.
//!
//! Compile and run:
//!   rustc --edition=2024 -O --test tools/vision-dvpc-052-finite-budget-ipw-mse.rs -o /tmp/vis-dvpc-052-tests
//!   /tmp/vis-dvpc-052-tests
//!   rustc --edition=2024 -O tools/vision-dvpc-052-finite-budget-ipw-mse.rs -o /tmp/vis-dvpc-052
//!   /tmp/vis-dvpc-052 --smoke
//!   /tmp/vis-dvpc-052
//!
//! The support-truncating arm is an oracle diagnostic using known synthetic
//! reward means. A lower finite-sample MSE is NOT an identification guarantee
//! and is never permission to emit a value claim or deployment certificate.
#![deny(unsafe_code)]

use std::env;

const ACTION_PROPENSITY_FLOOR: f64 = 0.02;
const POLICY_COUNTS: [usize; 3] = [4, 8, 32];
const SAMPLE_SIZES: [usize; 6] = [1, 4, 16, 64, 256, 1024];
const FULL_GRID_INTERVALS: usize = 100;
const SMOKE_GRID_INTERVALS: usize = 12;
const EPS: f64 = 1e-12;

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

#[derive(Clone, Copy, Debug, PartialEq)]
struct Policy {
    id: usize,
    /// Probability of action 1 under the target policy in each context.
    p1: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Logger {
    /// Probability of action 1 under the logging policy in each context.
    /// Boundary values 0 or 1 intentionally encode support truncation.
    p1: [f64; 2],
}

#[derive(Clone, Copy, Debug)]
struct PolicyRisk {
    expected_ipw_score: f64,
    true_value: f64,
    bias_squared: f64,
    variance_over_n: f64,
    mse: f64,
    omitted_target_mass: f64,
    identified: bool,
}

#[derive(Clone, Copy, Debug)]
struct DesignResult {
    logger: Logger,
    logger_reward: f64,
    worst_policy_id: usize,
    worst_policy_mse: f64,
    worst_policy_bias_squared: f64,
    worst_policy_variance_over_n: f64,
    worst_policy_omitted_target_mass: f64,
    identified_policy_count: usize,
    portfolio_fully_identified: bool,
}

fn reward_mean(context: usize, action: usize) -> f64 {
    match (context, action) {
        (0, 0) => 0.10,
        (0, 1) => 0.90,
        (1, 0) => 0.95,
        (1, 1) => 0.15,
        _ => unreachable!(),
    }
}

fn policies(regime: PortfolioRegime, m: usize) -> Vec<Policy> {
    assert!(m > 1);
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

fn target_value(policy: Policy) -> f64 {
    let mut value = 0.0;
    for context in 0..2 {
        let q = policy.p1[context];
        value += 0.5 * ((1.0 - q) * reward_mean(context, 0) + q * reward_mean(context, 1));
    }
    value
}

fn logger_reward(logger: Logger) -> f64 {
    let mut value = 0.0;
    for context in 0..2 {
        let p = logger.p1[context];
        value += 0.5 * ((1.0 - p) * reward_mean(context, 0) + p * reward_mean(context, 1));
    }
    value
}

fn logger_probabilities_valid(logger: Logger) -> bool {
    logger
        .p1
        .iter()
        .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
}

/// Exact finite-N IPW MSE under the frozen Bernoulli DGP.
///
/// When mu(a|x)=0, the corresponding target-policy mass contributes zero to
/// the IPW score. That creates the exact support-induced bias. The variance
/// below is the variance of one logged score divided by N; the mean of N IID
/// scores has MSE = bias^2 + variance/N.
fn ipw_policy_risk(policy: Policy, logger: Logger, sample_size: usize) -> Result<PolicyRisk, String> {
    if sample_size == 0 || !logger_probabilities_valid(logger) {
        return Err("positive sample size and valid logger probabilities are required".into());
    }
    let mut expected_score = 0.0;
    let mut score_second_moment = 0.0;
    let mut omitted_target_mass = 0.0;

    for context in 0..2 {
        let q = policy.p1[context];
        let p = logger.p1[context];
        let reward0 = reward_mean(context, 0);
        let reward1 = reward_mean(context, 1);

        if p > EPS {
            expected_score += 0.5 * q * reward1;
            score_second_moment += 0.5 * q * q * reward1 / p;
        } else {
            omitted_target_mass += 0.5 * q;
        }

        if 1.0 - p > EPS {
            expected_score += 0.5 * (1.0 - q) * reward0;
            score_second_moment +=
                0.5 * (1.0 - q) * (1.0 - q) * reward0 / (1.0 - p);
        } else {
            omitted_target_mass += 0.5 * (1.0 - q);
        }
    }

    let truth = target_value(policy);
    let bias = expected_score - truth;
    let variance = (score_second_moment - expected_score * expected_score).max(0.0);
    let variance_over_n = variance / sample_size as f64;
    let mse = bias * bias + variance_over_n;
    if !expected_score.is_finite()
        || !bias.is_finite()
        || !variance_over_n.is_finite()
        || !mse.is_finite()
        || !omitted_target_mass.is_finite()
    {
        return Err("non-finite IPW risk".into());
    }

    let identified = omitted_target_mass <= EPS;
    Ok(PolicyRisk {
        expected_ipw_score: expected_score,
        true_value: truth,
        bias_squared: bias * bias,
        variance_over_n,
        mse,
        omitted_target_mass,
        identified,
    })
}

fn full_support_grid(floor: f64, intervals: usize) -> Result<Vec<f64>, String> {
    if !floor.is_finite() || floor <= 0.0 || floor >= 0.5 || intervals == 0 {
        return Err("full-support grid requires floor in (0,0.5) and positive intervals".into());
    }
    let span = 1.0 - 2.0 * floor;
    Ok((0..=intervals)
        .map(|i| floor + span * i as f64 / intervals as f64)
        .collect())
}

fn support_truncating_grid(floor: f64, intervals: usize) -> Result<Vec<f64>, String> {
    let mut values = full_support_grid(floor, intervals)?;
    values.push(0.0);
    values.push(1.0);
    values.sort_by(f64::total_cmp);
    values.dedup_by(|a, b| (*a - *b).abs() <= EPS);
    Ok(values)
}

fn evaluate_design(policy_set: &[Policy], logger: Logger, sample_size: usize) -> Result<DesignResult, String> {
    if policy_set.is_empty() || !logger_probabilities_valid(logger) || sample_size == 0 {
        return Err("nonempty portfolio, valid logger and positive sample size required".into());
    }
    let mut worst: Option<(usize, PolicyRisk)> = None;
    let mut identified_policy_count = 0usize;
    for policy in policy_set {
        let risk = ipw_policy_risk(*policy, logger, sample_size)?;
        if risk.identified {
            identified_policy_count += 1;
        }
        match worst {
            None => worst = Some((policy.id, risk)),
            Some((_, current)) if risk.mse > current.mse + EPS => {
                worst = Some((policy.id, risk));
            }
            _ => {}
        }
    }
    let (worst_policy_id, worst_risk) = worst.expect("nonempty policy set checked above");
    Ok(DesignResult {
        logger,
        logger_reward: logger_reward(logger),
        worst_policy_id,
        worst_policy_mse: worst_risk.mse,
        worst_policy_bias_squared: worst_risk.bias_squared,
        worst_policy_variance_over_n: worst_risk.variance_over_n,
        worst_policy_omitted_target_mass: worst_risk.omitted_target_mass,
        identified_policy_count,
        portfolio_fully_identified: identified_policy_count == policy_set.len(),
    })
}

fn uniform_distance(logger: Logger) -> f64 {
    (logger.p1[0] - 0.5).powi(2) + (logger.p1[1] - 0.5).powi(2)
}

/// Minimize worst-policy exact finite-N IPW MSE over the selected logger grid.
/// The unrestricted grid includes logger probabilities exactly 0 and 1.
fn minimax_mse_design(
    policy_set: &[Policy],
    sample_size: usize,
    floor: f64,
    intervals: usize,
    allow_support_truncation: bool,
) -> Result<DesignResult, String> {
    if policy_set.is_empty() || sample_size == 0 || intervals == 0 {
        return Err("nonempty portfolio, positive N and positive grid intervals required".into());
    }
    let values = if allow_support_truncation {
        support_truncating_grid(floor, intervals)?
    } else {
        full_support_grid(floor, intervals)?
    };
    let mut best: Option<DesignResult> = None;
    for p0 in &values {
        for p1 in &values {
            let logger = Logger { p1: [*p0, *p1] };
            let candidate = evaluate_design(policy_set, logger, sample_size)?;
            let replace = match best {
                None => true,
                Some(current) => {
                    candidate.worst_policy_mse < current.worst_policy_mse - EPS
                        || ((candidate.worst_policy_mse - current.worst_policy_mse).abs() <= EPS
                            && uniform_distance(candidate.logger) < uniform_distance(current.logger) - EPS)
                }
            };
            if replace {
                best = Some(candidate);
            }
        }
    }
    best.ok_or_else(|| "logger-design grid produced no feasible point".into())
}

fn benchmark(intervals: usize) -> Result<String, String> {
    if intervals == 0 {
        return Err("grid intervals must be positive".into());
    }
    let mut out = String::from(
        "record_type\tportfolio\tM\tN\tdesign\tp1_context0\tp1_context1\texpected_logger_reward\tworst_policy_id\tworst_policy_mse\tworst_policy_bias_squared\tworst_policy_variance_over_n\tworst_policy_omitted_target_mass\tidentified_policy_count\tportfolio_fully_identified\toracle_only\tgrid_intervals\n",
    );
    out.push_str("BENCHMARK_CONFIG\tid\tVIS-DVPC-052\n");
    out.push_str(&format!("BENCHMARK_CONFIG\tpropensity_floor\t{ACTION_PROPENSITY_FLOOR:.6}\n"));
    out.push_str(&format!("BENCHMARK_CONFIG\tgrid_intervals\t{intervals}\n"));
    out.push_str("BENCHMARK_CONFIG\tsample_sizes\t1,4,16,64,256,1024\n");
    out.push_str("BENCHMARK_CONFIG\testimand\tfinite-N IPW MSE against the full target-policy value\n");
    out.push_str("BENCHMARK_CONFIG\tinformation\toracle synthetic reward means; truncated design is diagnostic only\n");

    for regime in PortfolioRegime::ALL {
        for m in POLICY_COUNTS {
            let policy_set = policies(regime, m);
            for n in SAMPLE_SIZES {
                for (name, allow_truncation) in [
                    ("full_support", false),
                    ("support_truncating_oracle", true),
                ] {
                    let result = minimax_mse_design(
                        &policy_set,
                        n,
                        ACTION_PROPENSITY_FLOOR,
                        intervals,
                        allow_truncation,
                    )?;
                    out.push_str(&format!(
                        "RESULT\t{}\t{}\t{}\t{}\t{:.8}\t{:.8}\t{:.8}\t{}\t{:.10}\t{:.10}\t{:.10}\t{:.8}\t{}\t{}\ttrue\t{}\n",
                        regime.name(),
                        m,
                        n,
                        name,
                        result.logger.p1[0],
                        result.logger.p1[1],
                        result.logger_reward,
                        result.worst_policy_id,
                        result.worst_policy_mse,
                        result.worst_policy_bias_squared,
                        result.worst_policy_variance_over_n,
                        result.worst_policy_omitted_target_mass,
                        result.identified_policy_count,
                        result.portfolio_fully_identified,
                        intervals,
                    ));
                }
            }
        }
    }
    Ok(out)
}

fn main() {
    let smoke = env::args().any(|arg| arg == "--smoke");
    let intervals = if smoke {
        SMOKE_GRID_INTERVALS
    } else {
        FULL_GRID_INTERVALS
    };
    match benchmark(intervals) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_052_ERROR\t{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_n_mse_matches_independent_bernoulli_enumeration() {
        let policy = Policy {
            id: 0,
            p1: [0.8, 0.2],
        };
        let logger = Logger { p1: [1.0, 0.0] };
        let risk = ipw_policy_risk(policy, logger, 4).unwrap();
        assert!((risk.expected_ipw_score - 0.74).abs() < EPS);
        assert!((risk.true_value - 0.765).abs() < EPS);
        assert!((risk.bias_squared - 0.000625).abs() < EPS);
        assert!((risk.variance_over_n - 0.0111).abs() < EPS);
        assert!((risk.mse - 0.011725).abs() < EPS);
        assert!((risk.omitted_target_mass - 0.5).abs() < EPS);
        assert!(!risk.identified);
    }

    #[test]
    fn full_support_ipw_is_unbiased_for_every_registered_target_policy() {
        let logger = Logger { p1: [0.31, 0.77] };
        for regime in PortfolioRegime::ALL {
            for policy in policies(regime, 8) {
                let risk = ipw_policy_risk(policy, logger, 64).unwrap();
                assert!(risk.identified);
                assert!(risk.bias_squared < 1e-20);
                assert!((risk.expected_ipw_score - risk.true_value).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn support_truncation_can_lower_small_sample_mse_but_is_not_identified() {
        let policy_set = policies(PortfolioRegime::Balanced, 4);
        let full =
            minimax_mse_design(&policy_set, 1, ACTION_PROPENSITY_FLOOR, 40, false).unwrap();
        let truncated =
            minimax_mse_design(&policy_set, 1, ACTION_PROPENSITY_FLOOR, 40, true).unwrap();
        assert!(truncated.worst_policy_mse + 0.1 < full.worst_policy_mse);
        assert!(!truncated.portfolio_fully_identified);
        assert!(truncated.identified_policy_count < policy_set.len());
        assert!(truncated.worst_policy_bias_squared > 0.0);
        assert!(truncated.worst_policy_omitted_target_mass > 0.0);
    }

    #[test]
    fn large_sample_optimum_returns_to_full_support_on_balanced_fixture() {
        let policy_set = policies(PortfolioRegime::Balanced, 4);
        let full =
            minimax_mse_design(&policy_set, 1024, ACTION_PROPENSITY_FLOOR, 40, false).unwrap();
        let unrestricted =
            minimax_mse_design(&policy_set, 1024, ACTION_PROPENSITY_FLOOR, 40, true).unwrap();
        assert!(unrestricted.worst_policy_mse <= full.worst_policy_mse + EPS);
        assert!(unrestricted.portfolio_fully_identified);
        assert!(unrestricted.worst_policy_bias_squared < 1e-20);
    }

    #[test]
    fn output_is_deterministic_and_counts_all_regime_portfolio_sample_pairs() {
        let a = benchmark(8).unwrap();
        let b = benchmark(8).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.lines().filter(|line| line.starts_with("RESULT\t")).count(), 108);
        assert_eq!(
            a.lines().filter(|line| line.starts_with("BENCHMARK_CONFIG\t")).count(),
            6
        );
    }

    #[test]
    fn invalid_sample_sizes_and_probability_ranges_fail_closed() {
        let p = policies(PortfolioRegime::Balanced, 4)[0];
        assert!(ipw_policy_risk(p, Logger { p1: [0.5, 0.5] }, 0).is_err());
        assert!(ipw_policy_risk(p, Logger { p1: [f64::NAN, 0.5] }, 10).is_err());
        assert!(full_support_grid(0.0, 10).is_err());
        assert!(support_truncating_grid(0.0, 10).is_err());
    }
}