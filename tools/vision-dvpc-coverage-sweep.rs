use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;

const GRID: &str = include_str!("../tests/fixtures/vision-dvpc-023-grid.tsv");
const RISK_ALPHA: f64 = 0.025;
const VALUE_ALPHA: f64 = 0.025;
const JOINT_ALPHA: f64 = 0.05;
const TARGET_JOINT_COVERAGE: f64 = 1.0 - JOINT_ALPHA;
const EPSILON: f64 = 1e-12;
const PPM: f64 = 1_000_000.0;

#[derive(Clone, Copy, Debug)]
struct GridPoint {
    n: usize,
    p_ppm: u64,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn parse_grid(text: &str) -> Result<Vec<GridPoint>, String> {
    let mut points = Vec::new();
    let mut seen = BTreeSet::new();
    for (line_index, line) in text.lines().enumerate() {
        if line_index == 0 {
            if line != "sample_n\tp_ppm" {
                return Err(invalid("grid header must be sample_n<TAB>p_ppm"));
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 2 {
            return Err(invalid(format!(
                "grid line {}: expected exactly two TSV fields",
                line_index + 1
            )));
        }
        let n = fields[0]
            .parse::<usize>()
            .map_err(|_| invalid(format!("grid line {}: invalid sample_n", line_index + 1)))?;
        let p_ppm = fields[1]
            .parse::<u64>()
            .map_err(|_| invalid(format!("grid line {}: invalid p_ppm", line_index + 1)))?;
        if n == 0 || p_ppm > 1_000_000 {
            return Err(invalid(format!(
                "grid line {}: sample_n must be positive and p_ppm <= 1000000",
                line_index + 1
            )));
        }
        if !seen.insert((n, p_ppm)) {
            return Err(invalid(format!(
                "grid line {}: duplicate (sample_n, p_ppm) pair",
                line_index + 1
            )));
        }
        points.push(GridPoint { n, p_ppm });
    }
    if points.is_empty() {
        return Err(invalid("grid contains no data rows"));
    }
    let ns: BTreeSet<usize> = points.iter().map(|point| point.n).collect();
    let ps: BTreeSet<u64> = points.iter().map(|point| point.p_ppm).collect();
    if points.len() != ns.len() * ps.len() {
        return Err(invalid(
            "grid must be a complete Cartesian product of sample sizes and probabilities",
        ));
    }
    Ok(points)
}

fn unique_ns(points: &[GridPoint]) -> Vec<usize> {
    points.iter().map(|point| point.n).collect::<BTreeSet<_>>().into_iter().collect()
}

fn unique_ps(points: &[GridPoint]) -> Vec<u64> {
    points.iter().map(|point| point.p_ppm).collect::<BTreeSet<_>>().into_iter().collect()
}

fn log_factorials(n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n + 1];
    for i in 1..=n {
        result[i] = result[i - 1] + (i as f64).ln();
    }
    result
}

fn log_binomial_pmf(n: usize, k: usize, p: f64, log_fact: &[f64]) -> f64 {
    if p <= 0.0 {
        return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
    }
    if p >= 1.0 {
        return if k == n { 0.0 } else { f64::NEG_INFINITY };
    }
    log_fact[n] - log_fact[k] - log_fact[n - k]
        + (k as f64) * p.ln()
        + ((n - k) as f64) * (-p).ln_1p()
}

fn binomial_pmf(n: usize, p: f64, log_fact: &[f64]) -> Vec<f64> {
    if p <= 0.0 {
        let mut result = vec![0.0; n + 1];
        result[0] = 1.0;
        return result;
    }
    if p >= 1.0 {
        let mut result = vec![0.0; n + 1];
        result[n] = 1.0;
        return result;
    }
    (0..=n)
        .map(|k| log_binomial_pmf(n, k, p, log_fact).exp())
        .collect()
}

fn binomial_cdf(n: usize, k: usize, p: f64, log_fact: &[f64]) -> f64 {
    if k >= n || p <= 0.0 {
        return 1.0;
    }
    if p >= 1.0 {
        return 0.0;
    }
    let logs: Vec<f64> = (0..=k)
        .map(|i| log_binomial_pmf(n, i, p, log_fact))
        .collect();
    let max_log = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scaled_sum: f64 = logs.iter().map(|value| (value - max_log).exp()).sum();
    (max_log.exp() * scaled_sum).clamp(0.0, 1.0)
}

/// Exact one-sided Clopper-Pearson upper confidence bound for a binomial rate.
fn exact_binomial_upper(n: usize, failures: usize, alpha: f64, log_fact: &[f64]) -> Result<f64, String> {
    if n == 0 || failures > n || !(0.0 < alpha && alpha < 1.0) {
        return Err(invalid("require n > 0, failures <= n, and 0 < alpha < 1"));
    }
    if failures == n {
        return Ok(1.0);
    }
    if failures == 0 {
        return Ok(1.0 - alpha.powf(1.0 / n as f64));
    }
    let mut low = 0.0;
    let mut high = 1.0;
    for _ in 0..80 {
        let midpoint = (low + high) / 2.0;
        if binomial_cdf(n, failures, midpoint, log_fact) > alpha {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }
    Ok(high)
}

fn required_zero_failure_trials(alpha: f64, risk_cap: f64) -> Result<usize, String> {
    if !(0.0 < alpha && alpha < 1.0 && 0.0 < risk_cap && risk_cap < 1.0) {
        return Err(invalid("require 0 < alpha < 1 and 0 < risk_cap < 1"));
    }
    let required = (alpha.ln() / (1.0 - risk_cap).ln()).ceil();
    if !required.is_finite() || required < 1.0 || required > usize::MAX as f64 {
        return Err(invalid("required sample size is out of range"));
    }
    Ok(required as usize)
}

fn risk_coverage(n: usize, truth_p: f64, upper_bounds: &[f64], log_fact: &[f64]) -> f64 {
    let distribution = binomial_pmf(n, truth_p, log_fact);
    distribution
        .iter()
        .zip(upper_bounds)
        .filter(|(_, upper)| **upper + EPSILON >= truth_p)
        .map(|(probability, _)| probability)
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

/// Coverage of the one-sided Hoeffding lower bound for iid Bernoulli outcomes in [0, 1].
fn value_coverage(n: usize, truth_mean: f64, alpha: f64, log_fact: &[f64]) -> f64 {
    let distribution = binomial_pmf(n, truth_mean, log_fact);
    let penalty = ((1.0 / alpha).ln() / (2.0 * n as f64)).sqrt();
    distribution
        .iter()
        .enumerate()
        .filter(|(successes, _)| (*successes as f64 / n as f64) - penalty <= truth_mean + EPSILON)
        .map(|(_, probability)| probability)
        .sum::<f64>()
        .clamp(0.0, 1.0)
}

fn exact_coverage_sweep() -> Result<String, String> {
    if (RISK_ALPHA + VALUE_ALPHA) > JOINT_ALPHA + EPSILON {
        return Err(invalid("marginal alpha allocations exceed the joint error budget"));
    }
    let points = parse_grid(GRID)?;
    let ns = unique_ns(&points);
    let ps_ppm = unique_ps(&points);
    let ps: Vec<f64> = ps_ppm.iter().map(|p| *p as f64 / PPM).collect();
    let mut output = String::from(
        "COVERAGE\tsample_n\trisk_p_ppm\tvalue_p_ppm\trisk_coverage\tvalue_coverage\tjoint_union_lower_bound\tjoint_target\n",
    );
    let mut min_risk = 1.0_f64;
    let mut min_value = 1.0_f64;
    let mut min_joint = 1.0_f64;
    let mut coverage_rows = 0usize;

    for n in &ns {
        let log_fact = log_factorials(*n);
        let upper_bounds: Vec<f64> = (0..=*n)
            .map(|failures| exact_binomial_upper(*n, failures, RISK_ALPHA, &log_fact))
            .collect::<Result<_, _>>()?;
        let risk_coverages: Vec<f64> = ps
            .iter()
            .map(|truth_p| risk_coverage(*n, *truth_p, &upper_bounds, &log_fact))
            .collect();
        let value_coverages: Vec<f64> = ps
            .iter()
            .map(|truth_mean| value_coverage(*n, *truth_mean, VALUE_ALPHA, &log_fact))
            .collect();

        for (risk_index, risk_p_ppm) in ps_ppm.iter().enumerate() {
            for (value_index, value_p_ppm) in ps_ppm.iter().enumerate() {
                let risk_cov = risk_coverages[risk_index];
                let value_cov = value_coverages[value_index];
                let joint_lower = (risk_cov + value_cov - 1.0).max(0.0);
                min_risk = min_risk.min(risk_cov);
                min_value = min_value.min(value_cov);
                min_joint = min_joint.min(joint_lower);
                let _ = writeln!(
                    output,
                    "COVERAGE\t{}\t{}\t{}\t{:.12}\t{:.12}\t{:.12}\t{:.6}",
                    n,
                    risk_p_ppm,
                    value_p_ppm,
                    risk_cov,
                    value_cov,
                    joint_lower,
                    TARGET_JOINT_COVERAGE
                );
                coverage_rows += 1;
            }
        }
    }

    // Deliberate transportability counterexample: a bound trained on source p is
    // evaluated against target truth p. The result is diagnostic, never target calibration.
    let shift_n = 72usize;
    let source_p = 0.01_f64;
    let target_p = 0.20_f64;
    let shift_log_fact = log_factorials(shift_n);
    let source_upper_bounds: Vec<f64> = (0..=shift_n)
        .map(|failures| exact_binomial_upper(shift_n, failures, RISK_ALPHA, &shift_log_fact))
        .collect::<Result<_, _>>()?;
    let source_distribution = binomial_pmf(shift_n, source_p, &shift_log_fact);
    let source_to_target_coverage: f64 = source_distribution
        .iter()
        .zip(&source_upper_bounds)
        .filter(|(_, upper)| **upper + EPSILON >= target_p)
        .map(|(probability, _)| probability)
        .sum();
    let _ = writeln!(
        output,
        "SHIFT_CONTROL\t{}\t{}\t{}\t{:.12}\tTARGET_CALIBRATED_FALSE",
        shift_n,
        (source_p * PPM).round() as u64,
        (target_p * PPM).round() as u64,
        source_to_target_coverage
    );
    let _ = writeln!(
        output,
        "SUMMARY\tn_grid={}\tp_grid={}\tcoverage_rows={}\tmin_risk_coverage={:.12}\tmin_value_coverage={:.12}\tmin_joint_union_lower={:.12}\tjoint_target={:.6}",
        ns.len(),
        ps.len(),
        coverage_rows,
        min_risk,
        min_value,
        min_joint,
        TARGET_JOINT_COVERAGE
    );
    Ok(output)
}

fn run_cli(args: &[String]) -> Result<String, String> {
    if args.len() == 2 && args[1] == "--sweep" {
        return exact_coverage_sweep();
    }
    if args.len() == 4 && args[1] == "--zero-failure-required" {
        let alpha = args[2].parse::<f64>().map_err(|_| invalid("invalid alpha"))?;
        let risk_cap = args[3].parse::<f64>().map_err(|_| invalid("invalid risk cap"))?;
        return Ok(format!(
            "required_zero_failure_trials\t{}\n",
            required_zero_failure_trials(alpha, risk_cap)?
        ));
    }
    Err("usage: vision-dvpc-coverage-sweep --sweep | --zero-failure-required ALPHA RISK_CAP".into())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run_cli(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS-DVPC-023 error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_frozen_complete_and_contains_probability_edges() {
        let points = parse_grid(GRID).unwrap();
        assert_eq!(points.len(), 117);
        assert_eq!(unique_ns(&points), vec![20, 40, 58, 59, 71, 72, 100, 120, 250]);
        assert_eq!(unique_ps(&points).first(), Some(&0));
        assert_eq!(unique_ps(&points).last(), Some(&1_000_000));
    }

    #[test]
    fn joint_alpha_spending_is_within_budget() {
        assert!((RISK_ALPHA + VALUE_ALPHA - JOINT_ALPHA) <= EPSILON);
        assert!((1.0 - JOINT_ALPHA - TARGET_JOINT_COVERAGE).abs() <= EPSILON);
    }

    #[test]
    fn exact_zero_failure_boundary_is_71_vs_72_trials() {
        assert!(exact_binomial_upper(71, 0, RISK_ALPHA, &log_factorials(71)).unwrap() > 0.05);
        assert!(exact_binomial_upper(72, 0, RISK_ALPHA, &log_factorials(72)).unwrap() < 0.05);
        assert_eq!(required_zero_failure_trials(RISK_ALPHA, 0.05).unwrap(), 72);
    }

    #[test]
    fn exact_binomial_probabilities_sum_to_one_including_edges() {
        for p in [0.0, 0.001, 0.05, 0.5, 0.999, 1.0] {
            let mass = binomial_pmf(120, p, &log_factorials(120)).iter().sum::<f64>();
            assert!((mass - 1.0).abs() < 1e-10, "p={p}, mass={mass}");
        }
    }

    #[test]
    fn every_grid_cell_meets_marginal_and_joint_coverage_targets() {
        let points = parse_grid(GRID).unwrap();
        let ns = unique_ns(&points);
        let ps: Vec<f64> = unique_ps(&points).iter().map(|p| *p as f64 / PPM).collect();
        for n in ns {
            let log_fact = log_factorials(n);
            let upper_bounds: Vec<f64> = (0..=n)
                .map(|k| exact_binomial_upper(n, k, RISK_ALPHA, &log_fact).unwrap())
                .collect();
            let risk_covs: Vec<f64> = ps.iter().map(|p| risk_coverage(n, *p, &upper_bounds, &log_fact)).collect();
            let value_covs: Vec<f64> = ps.iter().map(|p| value_coverage(n, *p, VALUE_ALPHA, &log_fact)).collect();
            for risk_cov in &risk_covs {
                assert!(*risk_cov + 1e-9 >= 1.0 - RISK_ALPHA, "risk coverage {risk_cov}");
            }
            for value_cov in &value_covs {
                assert!(*value_cov + 1e-9 >= 1.0 - VALUE_ALPHA, "value coverage {value_cov}");
            }
            for risk_cov in &risk_covs {
                for value_cov in &value_covs {
                    assert!(
                        (risk_cov + value_cov - 1.0) + 1e-9 >= TARGET_JOINT_COVERAGE,
                        "joint coverage lower bound: risk={risk_cov}, value={value_cov}"
                    );
                }
            }
        }
    }

    #[test]
    fn source_bound_does_not_transfer_across_material_target_shift() {
        let n = 72usize;
        let log_fact = log_factorials(n);
        let upper_bounds: Vec<f64> = (0..=n)
            .map(|k| exact_binomial_upper(n, k, RISK_ALPHA, &log_fact).unwrap())
            .collect();
        let source_distribution = binomial_pmf(n, 0.01, &log_fact);
        let coverage: f64 = source_distribution
            .iter()
            .zip(&upper_bounds)
            .filter(|(_, upper)| **upper + EPSILON >= 0.20)
            .map(|(probability, _)| probability)
            .sum();
        assert!(coverage < 0.01, "unexpected source-to-target coverage: {coverage}");
    }

    #[test]
    fn invalid_grid_and_invalid_alpha_fail_closed() {
        assert!(parse_grid("sample_n\tp_ppm\n1\t1000001\n").is_err());
        assert!(parse_grid("sample_n\tp_ppm\n1\t0\n1\t0\n").is_err());
        assert!(required_zero_failure_trials(0.0, 0.05).is_err());
        assert!(exact_binomial_upper(3, 4, RISK_ALPHA, &log_factorials(3)).is_err());
    }

    #[test]
    fn output_is_deterministic_and_has_all_expected_rows() {
        let first = exact_coverage_sweep().unwrap();
        let second = exact_coverage_sweep().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.lines().filter(|line| line.starts_with("COVERAGE\t")).count(), 1521);
        assert!(first.contains("SHIFT_CONTROL\t72\t10000\t200000"));
        assert!(first.contains("TARGET_CALIBRATED_FALSE"));
    }
}
