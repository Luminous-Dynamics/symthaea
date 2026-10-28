use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;

const GRID: &str = include_str!("../tests/fixtures/vision-dvpc-024-sequential-grid.tsv");
const RISK_ALPHA: f64 = 0.025;
const VALUE_ALPHA: f64 = 0.025;
const JOINT_ALPHA: f64 = 0.05;
const RISK_CAP: f64 = 0.05;
const VALUE_FLOOR: f64 = 0.5;
const RISK_ALTERNATIVES: [f64; 4] = [0.001, 0.01, 0.025, 0.04];
const VALUE_LAMBDAS: [f64; 2] = [0.5, 1.0];
const CLUSTER_RISK_LAMBDAS: [f64; 2] = [0.5, 1.0];
const EPSILON: f64 = 1e-12;
const PPM: f64 = 1_000_000.0;
const PEEKING_LOOKS: [usize; 2] = [72, 120];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum StreamKind {
    Risk,
    Value,
}

impl StreamKind {
    fn parse(text: &str, line: usize) -> Result<Self, String> {
        match text {
            "risk" => Ok(Self::Risk),
            "value" => Ok(Self::Value),
            _ => Err(format!("line {line}: stream_kind must be risk or value")),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Risk => "RISK",
            Self::Value => "VALUE",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GateVerdict {
    Authorized,
    Denied,
    InsufficientEvidence,
}

impl GateVerdict {
    fn as_str(self) -> &'static str {
        match self {
            Self::Authorized => "AUTHORIZED",
            Self::Denied => "DENIED",
            Self::InsufficientEvidence => "INSUFFICIENT_EVIDENCE",
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    kind: StreamKind,
    horizon: usize,
    truth_ppm: u64,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn parse_grid(text: &str) -> Result<Vec<Scenario>, String> {
    let mut scenarios = Vec::new();
    let mut seen = BTreeSet::new();
    for (index, line) in text.lines().enumerate() {
        let line_no = index + 1;
        if index == 0 {
            if line != "stream_kind\thorizon\ttruth_ppm" {
                return Err(invalid("grid header must be stream_kind<TAB>horizon<TAB>truth_ppm"));
            }
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 3 {
            return Err(invalid(format!("line {line_no}: expected three TSV fields")));
        }
        let kind = StreamKind::parse(fields[0], line_no)?;
        let horizon = fields[1]
            .parse::<usize>()
            .map_err(|_| invalid(format!("line {line_no}: invalid horizon")))?;
        let truth_ppm = fields[2]
            .parse::<u64>()
            .map_err(|_| invalid(format!("line {line_no}: invalid truth_ppm")))?;
        if horizon == 0 || truth_ppm > 1_000_000 {
            return Err(invalid(format!(
                "line {line_no}: horizon must be positive and truth_ppm <= 1000000"
            )));
        }
        if !seen.insert((kind, horizon, truth_ppm)) {
            return Err(invalid(format!("line {line_no}: duplicate scenario")));
        }
        scenarios.push(Scenario {
            kind,
            horizon,
            truth_ppm,
        });
    }
    if scenarios.is_empty() {
        return Err(invalid("grid contains no scenarios"));
    }
    Ok(scenarios)
}

fn log_factorials(n: usize) -> Vec<f64> {
    let mut values = vec![0.0; n + 1];
    for i in 1..=n {
        values[i] = values[i - 1] + (i as f64).ln();
    }
    values
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
        let mid = (low + high) / 2.0;
        if binomial_cdf(n, failures, mid, log_fact) > alpha {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(high)
}

fn log_mean_exp(log_values: &[f64]) -> f64 {
    let max_log = log_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    max_log + log_values.iter().map(|value| (value - max_log).exp()).sum::<f64>().ln()
        - (log_values.len() as f64).ln()
}

/// Mixture likelihood-ratio e-process for H0: failure probability >= risk_cap.
/// Alternatives q are fixed in advance and strictly below risk_cap.
fn risk_e_value(n: usize, failures: usize) -> f64 {
    let logs: Vec<f64> = RISK_ALTERNATIVES
        .iter()
        .map(|q| {
            (failures as f64) * (q / RISK_CAP).ln()
                + ((n - failures) as f64) * ((1.0 - q) / (1.0 - RISK_CAP)).ln()
        })
        .collect();
    log_mean_exp(&logs).exp()
}

/// Mixture of nonnegative bounded betting-factor e-processes for H0: mean <= value_floor.
/// Each lambda is fixed before data collection and is within [0, 1].
fn value_e_value(n: usize, successes: usize) -> f64 {
    let logs: Vec<f64> = VALUE_LAMBDAS
        .iter()
        .map(|lambda| {
            (successes as f64) * (1.0 + lambda * (1.0 - VALUE_FLOOR)).ln()
                + ((n - successes) as f64) * (1.0 - lambda * VALUE_FLOOR).ln()
        })
        .collect();
    log_mean_exp(&logs).exp()
}

fn validate_alpha_spending(alpha_risk: f64, alpha_value: f64, alpha_joint: f64) -> Result<(), String> {
    if !(0.0 < alpha_risk && alpha_risk < 1.0
        && 0.0 < alpha_value && alpha_value < 1.0
        && 0.0 < alpha_joint && alpha_joint < 1.0
        && alpha_risk + alpha_value <= alpha_joint + EPSILON)
    {
        return Err(invalid("risk/value alpha spending exceeds or invalidates joint error budget"));
    }
    Ok(())
}

fn validate_risk_alternatives(risk_cap: f64, alternatives: &[f64]) -> Result<(), String> {
    if !(0.0 < risk_cap && risk_cap < 1.0)
        || alternatives.is_empty()
        || alternatives.iter().any(|q| !(*q > 0.0 && *q < risk_cap))
    {
        return Err(invalid("risk alternatives must be nonempty, positive and strictly below risk cap"));
    }
    Ok(())
}

fn authorize(
    risk_crossed: bool,
    value_crossed: bool,
    evidence_complete: bool,
    target_calibrated: bool,
    cost_within_budget: bool,
    provenance_valid: bool,
) -> GateVerdict {
    if !evidence_complete || !target_calibrated || !cost_within_budget || !provenance_valid {
        return GateVerdict::Denied;
    }
    if !risk_crossed || !value_crossed {
        return GateVerdict::InsufficientEvidence;
    }
    GateVerdict::Authorized
}

fn validate_contract() -> Result<(), String> {
    validate_alpha_spending(RISK_ALPHA, VALUE_ALPHA, JOINT_ALPHA)?;
    validate_risk_alternatives(RISK_CAP, &RISK_ALTERNATIVES)?;
    if VALUE_LAMBDAS.is_empty() || VALUE_LAMBDAS.iter().any(|lambda| !(*lambda >= 0.0 && *lambda <= 1.0)) {
        return Err(invalid("value betting lambdas must be nonempty and in [0, 1]"));
    }
    if !(0.0 < VALUE_FLOOR && VALUE_FLOOR < 1.0) {
        return Err(invalid("value floor must be strictly between 0 and 1"));
    }
    Ok(())
}

/// Exact finite-horizon probability of ever crossing the risk e-process threshold
/// under an iid Bernoulli stream with the supplied true failure probability.
fn risk_crossing_probability(horizon: usize, true_p: f64) -> Result<f64, String> {
    if horizon == 0 || !(0.0 <= true_p && true_p <= 1.0) {
        return Err(invalid("risk horizon must be positive and probability in [0, 1]"));
    }
    let threshold = 1.0 / RISK_ALPHA;
    let mut surviving = vec![0.0; horizon + 1];
    surviving[0] = 1.0;
    for n in 1..=horizon {
        let mut next = vec![0.0; horizon + 1];
        for failures in 0..n {
            let mass = surviving[failures];
            if mass == 0.0 {
                continue;
            }
            let no_failure_count = failures;
            if risk_e_value(n, no_failure_count) + EPSILON < threshold {
                next[no_failure_count] += mass * (1.0 - true_p);
            }
            let failure_count = failures + 1;
            if risk_e_value(n, failure_count) + EPSILON < threshold {
                next[failure_count] += mass * true_p;
            }
        }
        surviving = next;
    }
    Ok((1.0 - surviving.iter().sum::<f64>()).clamp(0.0, 1.0))
}

/// Advance the row-wise process through one cluster at a fixed latent failure probability.
fn advance_risk_survival_for_cluster(
    starting_survival: &[f64],
    start_n: usize,
    cluster_size: usize,
    true_p: f64,
) -> Vec<f64> {
    let horizon = starting_survival.len() - 1;
    let threshold = 1.0 / RISK_ALPHA;
    let mut surviving = starting_survival.to_vec();
    for n in (start_n + 1)..=(start_n + cluster_size) {
        let mut next = vec![0.0; horizon + 1];
        for failures in 0..n {
            let mass = surviving[failures];
            if mass == 0.0 {
                continue;
            }
            if risk_e_value(n, failures) + EPSILON < threshold {
                next[failures] += mass * (1.0 - true_p);
            }
            if risk_e_value(n, failures + 1) + EPSILON < threshold {
                next[failures + 1] += mass * true_p;
            }
        }
        surviving = next;
    }
    surviving
}

/// Exact row-wise false-certification probability under independent clusters with
/// a latent per-cluster p. Marginal row risk can equal the cap while conditional
/// within-cluster probabilities violate the row-wise e-process null.
fn clustered_rowwise_crossing_probability(
    cluster_count: usize,
    cluster_size: usize,
    low_p: f64,
    high_p: f64,
    low_weight: f64,
) -> Result<f64, String> {
    if cluster_count == 0
        || cluster_size == 0
        || !(0.0 <= low_p && low_p <= 1.0)
        || !(0.0 <= high_p && high_p <= 1.0)
        || !(0.0 <= low_weight && low_weight <= 1.0)
    {
        return Err(invalid("invalid clustered risk-mixture parameters"));
    }
    let horizon = cluster_count * cluster_size;
    let mut surviving = vec![0.0; horizon + 1];
    surviving[0] = 1.0;
    for cluster in 0..cluster_count {
        let start_n = cluster * cluster_size;
        let low = advance_risk_survival_for_cluster(&surviving, start_n, cluster_size, low_p);
        let high = advance_risk_survival_for_cluster(&surviving, start_n, cluster_size, high_p);
        surviving = low.iter().zip(high.iter())
            .map(|(a, b)| low_weight * a + (1.0 - low_weight) * b)
            .collect();
    }
    Ok((1.0 - surviving.iter().sum::<f64>()).clamp(0.0, 1.0))
}

/// Cluster-level bounded e-process for H0: mean cluster failure fraction >= risk cap.
/// Each independent cluster contributes one observed fraction in [0, 1].
fn cluster_risk_e_value(cluster_rates: &[f64]) -> Result<f64, String> {
    if cluster_rates.is_empty()
        || cluster_rates.iter().any(|rate| !rate.is_finite() || !(0.0..=1.0).contains(rate))
    {
        return Err(invalid("cluster risk rates must be nonempty, finite and in [0, 1]"));
    }
    let logs: Vec<f64> = CLUSTER_RISK_LAMBDAS
        .iter()
        .map(|lambda| {
            cluster_rates
                .iter()
                .map(|rate| (1.0 + lambda * (RISK_CAP - rate)).ln())
                .sum::<f64>()
        })
        .collect();
    Ok(log_mean_exp(&logs).exp())
}

/// Exact finite-horizon crossing probability for independent clusters with a
/// two-point cluster-risk distribution. The marginal mean equals risk_cap.
fn cluster_risk_mixture_crossing_probability(
    cluster_count: usize,
    low_rate: f64,
    high_rate: f64,
    low_weight: f64,
) -> Result<f64, String> {
    if cluster_count == 0
        || !(0.0..=1.0).contains(&low_rate)
        || !(0.0..=1.0).contains(&high_rate)
        || !(0.0..=1.0).contains(&low_weight)
    {
        return Err(invalid("invalid cluster mixture parameters"));
    }
    let marginal = low_weight * low_rate + (1.0 - low_weight) * high_rate;
    if (marginal - RISK_CAP).abs() > EPSILON {
        return Err(invalid("cluster mixture mean must equal the risk null boundary"));
    }
    let threshold = 1.0 / RISK_ALPHA;
    let mut surviving = vec![0.0; cluster_count + 1];
    surviving[0] = 1.0;
    for n in 1..=cluster_count {
        let mut next = vec![0.0; cluster_count + 1];
        for high_count in 0..n {
            let mass = surviving[high_count];
            if mass == 0.0 {
                continue;
            }
            let low_path = {
                let rates = std::iter::repeat_n(low_rate, n - high_count)
                    .chain(std::iter::repeat_n(high_rate, high_count))
                    .collect::<Vec<_>>();
                cluster_risk_e_value(&rates)? + EPSILON < threshold
            };
            if low_path {
                next[high_count] += mass * low_weight;
            }
            let high_path = {
                let rates = std::iter::repeat_n(low_rate, n - (high_count + 1))
                    .chain(std::iter::repeat_n(high_rate, high_count + 1))
                    .collect::<Vec<_>>();
                cluster_risk_e_value(&rates)? + EPSILON < threshold
            };
            if high_path {
                next[high_count + 1] += mass * (1.0 - low_weight);
            }
        }
        surviving = next;
    }
    Ok((1.0 - surviving.iter().sum::<f64>()).clamp(0.0, 1.0))
}

/// Exact finite-horizon probability of ever crossing the value e-process threshold
/// under an iid Bernoulli reward stream whose mean is true_mean in [0, 1].
fn value_crossing_probability(horizon: usize, true_mean: f64) -> Result<f64, String> {
    if horizon == 0 || !(0.0 <= true_mean && true_mean <= 1.0) {
        return Err(invalid("value horizon must be positive and mean in [0, 1]"));
    }
    let threshold = 1.0 / VALUE_ALPHA;
    let mut surviving = vec![0.0; horizon + 1];
    surviving[0] = 1.0;
    for n in 1..=horizon {
        let mut next = vec![0.0; horizon + 1];
        for successes in 0..n {
            let mass = surviving[successes];
            if mass == 0.0 {
                continue;
            }
            if value_e_value(n, successes) + EPSILON < threshold {
                next[successes] += mass * (1.0 - true_mean);
            }
            let success_count = successes + 1;
            if value_e_value(n, success_count) + EPSILON < threshold {
                next[success_count] += mass * true_mean;
            }
        }
        surviving = next;
    }
    Ok((1.0 - surviving.iter().sum::<f64>()).clamp(0.0, 1.0))
}

/// Deliberate peeking baseline: reuse fixed-sample upper bounds at several looks,
/// declaring safe if any look's one-sided upper bound is <= the cap.
fn naive_fixed_look_crossing_probability(
    horizon: usize,
    true_p: f64,
    looks: &[usize],
) -> Result<f64, String> {
    if horizon == 0 || !(0.0 <= true_p && true_p <= 1.0) || looks.is_empty() {
        return Err(invalid("invalid horizon, probability or empty look schedule"));
    }
    let mut look_bounds = BTreeMap::new();
    for look in looks {
        if *look == 0 || *look > horizon {
            return Err(invalid("each fixed look must be in 1..=horizon"));
        }
        let log_fact = log_factorials(*look);
        let bounds: Vec<f64> = (0..=*look)
            .map(|k| exact_binomial_upper(*look, k, RISK_ALPHA, &log_fact))
            .collect::<Result<_, _>>()?;
        look_bounds.insert(*look, bounds);
    }
    let mut surviving = vec![0.0; horizon + 1];
    surviving[0] = 1.0;
    for n in 1..=horizon {
        let mut next = vec![0.0; horizon + 1];
        for failures in 0..n {
            let mass = surviving[failures];
            if mass == 0.0 {
                continue;
            }
            let bounds = look_bounds.get(&n);
            let no_failure_count = failures;
            let no_failure_certifies = bounds
                .map(|values| values[no_failure_count] <= RISK_CAP + EPSILON)
                .unwrap_or(false);
            if !no_failure_certifies {
                next[no_failure_count] += mass * (1.0 - true_p);
            }
            let failure_count = failures + 1;
            let failure_certifies = bounds
                .map(|values| values[failure_count] <= RISK_CAP + EPSILON)
                .unwrap_or(false);
            if !failure_certifies {
                next[failure_count] += mass * true_p;
            }
        }
        surviving = next;
    }
    Ok((1.0 - surviving.iter().sum::<f64>()).clamp(0.0, 1.0))
}

fn exact_sweep() -> Result<String, String> {
    validate_contract()?;
    let scenarios = parse_grid(GRID)?;
    let mut output = String::from("kind\thorizon\ttruth_ppm\talpha_ppm\tprob_ever_cross\tinterpretation\n");
    let _ = writeln!(
        output,
        "CONTRACT\talpha_risk={:.6}\talpha_value={:.6}\talpha_joint={:.6}\trisk_cap={:.6}\tvalue_floor={:.6}\trisk_alternatives=0.001,0.01,0.025,0.04\tvalue_lambdas=0.5,1.0\tcluster_risk_lambdas=0.5,1.0\tpeeking_looks=72,120",
        RISK_ALPHA, VALUE_ALPHA, JOINT_ALPHA, RISK_CAP, VALUE_FLOOR
    );
    let mut risk_count = 0usize;
    let mut value_count = 0usize;
    for scenario in &scenarios {
        let truth = scenario.truth_ppm as f64 / PPM;
        match scenario.kind {
            StreamKind::Risk => {
                let probability = risk_crossing_probability(scenario.horizon, truth)?;
                let _ = writeln!(
                    output,
                    "RISK\t{}\t{}\t{}\t{:.12}\t{}",
                    scenario.horizon,
                    scenario.truth_ppm,
                    (RISK_ALPHA * PPM).round() as u64,
                    probability,
                    if truth + EPSILON >= RISK_CAP { "NULL_OR_UNSAFE" } else { "SAFE_ALTERNATIVE" }
                );
                risk_count += 1;
            }
            StreamKind::Value => {
                let probability = value_crossing_probability(scenario.horizon, truth)?;
                let _ = writeln!(
                    output,
                    "VALUE\t{}\t{}\t{}\t{:.12}\t{}",
                    scenario.horizon,
                    scenario.truth_ppm,
                    (VALUE_ALPHA * PPM).round() as u64,
                    probability,
                    if truth <= VALUE_FLOOR + EPSILON { "NULL_OR_LOW_VALUE" } else { "POSITIVE_VALUE_ALTERNATIVE" }
                );
                value_count += 1;
            }
        }
    }
    let peeking = naive_fixed_look_crossing_probability(120, RISK_CAP, &PEEKING_LOOKS)?;
    let sequential = risk_crossing_probability(120, RISK_CAP)?;
    let _ = writeln!(
        output,
        "PEEKING_CONTROL\t120\t50000\t72,120\t{:.12}\t{:.12}\t{:.6}",
        peeking,
        sequential,
        RISK_ALPHA
    );
    let clustered = clustered_rowwise_crossing_probability(5, 20, 0.0, 0.10, 0.5)?;
    let cluster_gate_cross = cluster_risk_mixture_crossing_probability(200, 0.0, 0.10, 0.5)?;
    let five_zero_clusters = cluster_risk_e_value(&[0.0; 5])?;
    let _ = writeln!(
        output,
        "CLUSTER_DEPENDENCE_CONTROL\tclusters=5\tcluster_size=20\tmarginal_risk_ppm=50000\trowwise_false_cross={:.12}\tcluster_gate_false_cross_by_200={:.12}\tcluster_gate_all_zero_crossed_5={}\tcluster_gate_all_zero_e_value={:.12}\talpha={:.6}",
        clustered,
        cluster_gate_cross,
        if five_zero_clusters >= 1.0 / RISK_ALPHA { "YES" } else { "NO" },
        five_zero_clusters,
        RISK_ALPHA
    );
    for (name, risk_crossed, value_crossed, evidence_complete, target_calibrated, cost_ok, provenance_ok) in [
        ("all_clear", true, true, true, true, true, true),
        ("missing_evidence", true, true, false, true, true, true),
        ("target_shift_or_uncalibrated", true, true, true, false, true, true),
        ("budget_overrun", true, true, true, true, false, true),
        ("provenance_mismatch", true, true, true, true, true, false),
        ("always_deny_control", false, false, true, true, true, true),
    ] {
        let verdict = authorize(
            risk_crossed,
            value_crossed,
            evidence_complete,
            target_calibrated,
            cost_ok,
            provenance_ok,
        );
        let _ = writeln!(output, "GATE_CONTROL\t{name}\t{}", verdict.as_str());
    }
    let _ = writeln!(
        output,
        "SUMMARY\trisk_scenarios={}\tvalue_scenarios={}\tmax_horizon=200\talpha_risk={:.6}\talpha_value={:.6}\tjoint_alpha={:.6}",
        risk_count,
        value_count,
        RISK_ALPHA,
        VALUE_ALPHA,
        JOINT_ALPHA
    );
    Ok(output)
}

fn run_cli(args: &[String]) -> Result<String, String> {
    if args.len() == 2 && args[1] == "--sweep" {
        return exact_sweep();
    }
    Err("usage: vision-dvpc-anytime-gate --sweep".into())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    match run_cli(&args) {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS-DVPC-024 error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_spending_is_valid_and_predeclared() {
        assert!(validate_contract().is_ok());
        assert!(validate_alpha_spending(0.025, 0.025, 0.05).is_ok());
        assert!(validate_alpha_spending(0.03, 0.03, 0.05).is_err());
        assert!(RISK_ALTERNATIVES.iter().all(|q| *q < RISK_CAP));
        assert!(validate_risk_alternatives(RISK_CAP, &[0.01, 0.04]).is_ok());
        assert!(validate_risk_alternatives(RISK_CAP, &[0.01, 0.06]).is_err());
        assert!(VALUE_LAMBDAS.iter().all(|lambda| (0.0..=1.0).contains(lambda)));
    }

    #[test]
    fn authorization_gate_fails_closed_on_missing_shifted_or_over_budget_evidence() {
        assert_eq!(authorize(true, true, true, true, true, true), GateVerdict::Authorized);
        assert_eq!(authorize(true, true, false, true, true, true), GateVerdict::Denied);
        assert_eq!(authorize(true, true, true, false, true, true), GateVerdict::Denied);
        assert_eq!(authorize(true, true, true, true, false, true), GateVerdict::Denied);
        assert_eq!(authorize(true, true, true, true, true, false), GateVerdict::Denied);
        assert_eq!(authorize(false, false, true, true, true, true), GateVerdict::InsufficientEvidence);
        assert_ne!(authorize(false, false, true, true, true, true), GateVerdict::Authorized);
    }

    #[test]
    fn anytime_risk_crossing_stays_below_alpha_under_null_and_unsafe_rates() {
        for horizon in [20, 40, 80, 120, 200] {
            for true_p in [RISK_CAP, 0.10, 0.20] {
                let crossing = risk_crossing_probability(horizon, true_p).unwrap();
                assert!(
                    crossing <= RISK_ALPHA + 1e-9,
                    "risk crossing={crossing}, p={true_p}, horizon={horizon}"
                );
            }
        }
    }

    #[test]
    fn anytime_value_crossing_stays_below_alpha_under_null() {
        for horizon in [20, 40, 80, 120, 200] {
            for true_mean in [0.40, VALUE_FLOOR] {
                let crossing = value_crossing_probability(horizon, true_mean).unwrap();
                assert!(
                    crossing <= VALUE_ALPHA + 1e-9,
                    "value crossing={crossing}, mean={true_mean}, horizon={horizon}"
                );
            }
        }
    }

    #[test]
    fn sequential_gate_has_nontrivial_power() {
        assert!(risk_crossing_probability(120, 0.0).unwrap() > 0.99);
        assert!(value_crossing_probability(120, 0.70).unwrap() > 0.90);
    }

    #[test]
    fn naive_fixed_sample_peeking_exceeds_nominal_false_authorization_rate() {
        let naive = naive_fixed_look_crossing_probability(120, RISK_CAP, &PEEKING_LOOKS).unwrap();
        let sequential = risk_crossing_probability(120, RISK_CAP).unwrap();
        assert!(naive > RISK_ALPHA + 0.005, "naive crossing={naive}");
        assert!(sequential <= RISK_ALPHA, "e-process crossing={sequential}");
        assert!(naive > sequential);
    }

    #[test]
    fn exact_peeking_control_matches_independent_reference() {
        let naive = naive_fixed_look_crossing_probability(120, RISK_CAP, &PEEKING_LOOKS).unwrap();
        let sequential = risk_crossing_probability(120, RISK_CAP).unwrap();
        assert!((naive - 0.0329371595279454).abs() < 1e-10);
        assert!((sequential - 0.0089242491504725).abs() < 1e-10);
    }

    #[test]
    fn row_level_process_fails_under_clustered_random_effects() {
        let crossing = clustered_rowwise_crossing_probability(5, 20, 0.0, 0.10, 0.5).unwrap();
        assert!((crossing - 0.063416077728).abs() < 1e-9, "crossing={crossing}");
        assert!(crossing > RISK_ALPHA);
    }

    #[test]
    fn independent_cluster_e_process_uses_cluster_means_and_fails_closed_early() {
        let expected_cluster_risk = 0.5 * 0.0 + 0.5 * 0.10;
        assert!((expected_cluster_risk - RISK_CAP).abs() < EPSILON);
        let e5 = cluster_risk_e_value(&[0.0; 5]).unwrap();
        assert!(e5 < 1.0 / RISK_ALPHA);
        let e100 = cluster_risk_e_value(&[0.0; 100]).unwrap();
        assert!(e100 >= 1.0 / RISK_ALPHA);
        let p_cross_200 = cluster_risk_mixture_crossing_probability(200, 0.0, 0.10, 0.5).unwrap();
        assert!(p_cross_200 <= RISK_ALPHA + 1e-9, "cluster-level false crossing={p_cross_200}");
        assert!(p_cross_200 < 1e-8, "expected conservative cluster gate, got {p_cross_200}");
        assert!(cluster_risk_mixture_crossing_probability(5, 0.0, 0.10, 0.4).is_err());
        assert!(cluster_risk_e_value(&[]).is_err());
        assert!(cluster_risk_e_value(&[0.0, 1.01]).is_err());
        for lambda in CLUSTER_RISK_LAMBDAS {
            let low = 1.0 + lambda * (RISK_CAP - 0.0);
            let high = 1.0 + lambda * (RISK_CAP - 0.10);
            let expectation = 0.5 * low + 0.5 * high;
            assert!((expectation - 1.0).abs() < EPSILON);
        }
    }

    #[test]
    fn grid_is_frozen_well_formed_and_complete() {
        let scenarios = parse_grid(GRID).unwrap();
        assert_eq!(scenarios.len(), 40);
        assert_eq!(scenarios.iter().filter(|s| s.kind == StreamKind::Risk).count(), 20);
        assert_eq!(scenarios.iter().filter(|s| s.kind == StreamKind::Value).count(), 20);
        assert!(parse_grid("stream_kind\thorizon\ttruth_ppm\nrisk\t0\t50000\n").is_err());
        assert!(parse_grid("stream_kind\thorizon\ttruth_ppm\nother\t10\t50000\n").is_err());
        assert!(risk_crossing_probability(10, -0.01).is_err());
        assert!(risk_crossing_probability(10, 1.01).is_err());
        assert!(value_crossing_probability(10, f64::NAN).is_err());
    }

    #[test]
    fn output_is_deterministic_and_has_expected_scenarios() {
        let first = exact_sweep().unwrap();
        let second = exact_sweep().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.lines().filter(|line| line.starts_with("RISK\t")).count(), 20);
        assert_eq!(first.lines().filter(|line| line.starts_with("VALUE\t")).count(), 20);
        assert!(first.contains("CONTRACT\talpha_risk=0.025000\talpha_value=0.025000\talpha_joint=0.050000\trisk_cap=0.050000\tvalue_floor=0.500000\trisk_alternatives=0.001,0.01,0.025,0.04\tvalue_lambdas=0.5,1.0\tcluster_risk_lambdas=0.5,1.0\tpeeking_looks=72,120"));
        assert!(first.contains("PEEKING_CONTROL\t120\t50000\t72,120\t0.032937159528\t0.008924249150\t0.025000"));
        assert!(first.contains("CLUSTER_DEPENDENCE_CONTROL\tclusters=5\tcluster_size=20\tmarginal_risk_ppm=50000\trowwise_false_cross=0.063416077728\tcluster_gate_false_cross_by_200=0.000000000063\tcluster_gate_all_zero_crossed_5=NO"));
        assert!(first.contains("GATE_CONTROL\tall_clear\tAUTHORIZED"));
        assert!(first.contains("GATE_CONTROL\tmissing_evidence\tDENIED"));
        assert!(first.contains("GATE_CONTROL\ttarget_shift_or_uncalibrated\tDENIED"));
        assert!(first.contains("GATE_CONTROL\tbudget_overrun\tDENIED"));
        assert!(first.contains("GATE_CONTROL\tprovenance_mismatch\tDENIED"));
        assert!(first.contains("GATE_CONTROL\talways_deny_control\tINSUFFICIENT_EVIDENCE"));
        assert!(first.contains("SUMMARY\trisk_scenarios=20\tvalue_scenarios=20\tmax_horizon=200\talpha_risk=0.025000\talpha_value=0.025000\tjoint_alpha=0.050000"));
    }
}
