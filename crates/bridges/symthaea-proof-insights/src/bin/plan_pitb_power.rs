// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic simulation-based power planning for PITB's paired clustered
//! exact-target success comparison.
//!
//! This is a *planning aid*, not a proof checker or an empirical benchmark
//! result. It models shared family difficulty with a beta-distributed cluster
//! success probability, applies the treatment gain on the probability scale,
//! simulates paired control/treatment outcomes, and uses a cluster-level
//! t-interval as the preregistered gate approximation. The report records the
//! model and its limitations so it cannot be mistaken for measured utility.

use std::{env, fs, path::Path};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

const MIN_CLUSTERS: usize = 20;
const MIN_SIMULATIONS: usize = 1_000;
const Z_975: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone)]
struct Config {
    baseline_rate: f64,
    expected_gain: f64,
    minimum_observed_gain: f64,
    icc: f64,
    cluster_sizes: Vec<usize>,
    simulations: usize,
    seed: u64,
}

#[derive(Debug, Clone)]
struct Trial {
    estimate: f64,
    lower_95: Option<f64>,
    upper_95: Option<f64>,
    passed_gate: bool,
    generated_gain: f64,
}

#[derive(Debug, Clone)]
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    // Open interval (0, 1), avoiding log(0) in the samplers below.
    fn uniform(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64 + 0.5) / ((1_u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform();
        radius * angle.cos()
    }

    fn gamma(&mut self, shape: f64) -> f64 {
        debug_assert!(shape.is_finite() && shape > 0.0);
        if shape < 1.0 {
            // Gamma(k) = Gamma(k+1) * U^(1/k).
            let u = self.uniform().max(f64::MIN_POSITIVE);
            return self.gamma(shape + 1.0) * u.powf(1.0 / shape);
        }

        // Marsaglia-Tsang rejection sampler for shape >= 1.
        let d = shape - 1.0 / 3.0;
        let c = (1.0 / (9.0 * d)).sqrt();
        loop {
            let x = self.normal();
            let base = 1.0 + c * x;
            if base <= 0.0 {
                continue;
            }
            let v = base * base * base;
            let u = self.uniform();
            if u < 1.0 - 0.0331 * x.powi(4)
                || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln())
            {
                return d * v;
            }
        }
    }

    fn beta(&mut self, alpha: f64, beta: f64) -> f64 {
        let x = self.gamma(alpha);
        let y = self.gamma(beta);
        let total = x + y;
        if total.is_finite() && total > 0.0 {
            (x / total).clamp(0.0, 1.0)
        } else {
            // The only expected route here is extreme floating-point
            // underflow for a deliberately pathological configuration.
            alpha / (alpha + beta)
        }
    }
}

fn parse_config(args: &[String]) -> Result<Config> {
    if args.len() != 9 || args.get(1).is_some_and(|x| x == "--help" || x == "-h") {
        bail!(
            "usage: {} <baseline_rate> <expected_gain> <minimum_observed_gain> <icc_0_to_lt_1> <cluster_sizes_csv> <simulations>=1000 <seed> <output.json>",
            args.first().map(String::as_str).unwrap_or("plan_pitb_power")
        );
    }

    let baseline_rate = parse_f64(&args[1], "baseline_rate")?;
    let expected_gain = parse_f64(&args[2], "expected_gain")?;
    let minimum_observed_gain = parse_f64(&args[3], "minimum_observed_gain")?;
    let icc = parse_f64(&args[4], "icc")?;
    let cluster_sizes = args[5]
        .split(',')
        .map(|item| {
            let size = item.trim().parse::<usize>().context("parse cluster size")?;
            if size == 0 {
                bail!("cluster sizes must all be positive integers");
            }
            Ok(size)
        })
        .collect::<Result<Vec<_>>>()?;
    let simulations = args[6].parse::<usize>().context("parse simulations")?;
    let seed = args[7].parse::<u64>().context("parse seed")?;

    if !(0.0..1.0).contains(&baseline_rate) {
        bail!("baseline_rate must be strictly between 0 and 1");
    }
    if !(0.0..1.0).contains(&expected_gain) || expected_gain == 0.0 {
        bail!("expected_gain must be strictly between 0 and 1");
    }
    if !(0.0..1.0).contains(&minimum_observed_gain) || minimum_observed_gain == 0.0 {
        bail!("minimum_observed_gain must be strictly between 0 and 1");
    }
    if baseline_rate + expected_gain <= baseline_rate {
        bail!("expected_gain is below representable precision for baseline_rate");
    }
    if !(0.0..1.0).contains(&icc) {
        bail!("icc must be in [0, 1)");
    }
    if icc > 0.0 {
        let concentration = (1.0 - icc) / icc;
        if !concentration.is_finite()
            || !(baseline_rate * concentration).is_finite()
            || !((1.0 - baseline_rate) * concentration).is_finite()
        {
            bail!("icc is too close to zero for a finite beta-binomial concentration");
        }
    }
    if cluster_sizes.len() < MIN_CLUSTERS {
        bail!(
            "at least {MIN_CLUSTERS} independent component clusters are required for this planning gate"
        );
    }
    if simulations < MIN_SIMULATIONS {
        bail!("simulations must be at least {MIN_SIMULATIONS}");
    }

    Ok(Config {
        baseline_rate,
        expected_gain,
        minimum_observed_gain,
        icc,
        cluster_sizes,
        simulations,
        seed,
    })
}

fn parse_f64(raw: &str, name: &str) -> Result<f64> {
    let value = raw.parse::<f64>().with_context(|| format!("parse {name}"))?;
    if !value.is_finite() {
        bail!("{name} must be finite");
    }
    Ok(value)
}

fn t_critical_975(df: usize) -> f64 {
    // Cornish-Fisher expansion around the two-sided normal 97.5th percentile.
    // The planner requires >=20 clusters, where this approximation is adequate
    // for a design sensitivity analysis; final analysis still uses the frozen
    // benchmark analysis procedure.
    let v = df as f64;
    let z = Z_975;
    z + (z.powi(3) + z) / (4.0 * v)
        + (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * v.powi(2))
        + (3.0 * z.powi(7) + 19.0 * z.powi(5) + 17.0 * z.powi(3) - 15.0 * z)
            / (384.0 * v.powi(3))
}

fn simulate_trial(config: &Config, rng: &mut Rng) -> Trial {
    let clusters = config.cluster_sizes.len();
    let total_n: usize = config.cluster_sizes.iter().sum();
    let mut differences = Vec::with_capacity(clusters);
    let mut generated_treatment_probability = 0.0;
    let mut generated_baseline_probability = 0.0;

    let concentration = if config.icc > 0.0 {
        (1.0 - config.icc) / config.icc
    } else {
        f64::INFINITY
    };
    let (alpha, beta) = if config.icc > 0.0 {
        (
            config.baseline_rate * concentration,
            (1.0 - config.baseline_rate) * concentration,
        )
    } else {
        (config.baseline_rate, 1.0 - config.baseline_rate)
    };

    let mut treatment_total = 0usize;
    let mut control_total = 0usize;
    for &n in &config.cluster_sizes {
        let p_control = if config.icc > 0.0 {
            rng.beta(alpha, beta)
        } else {
            config.baseline_rate
        };
        let p_treatment = (p_control + config.expected_gain).min(1.0);
        generated_baseline_probability += p_control * n as f64;
        generated_treatment_probability += p_treatment * n as f64;

        let mut control_successes = 0usize;
        let mut treatment_successes = 0usize;
        for _ in 0..n {
            if rng.uniform() < p_control {
                control_successes += 1;
            }
            if rng.uniform() < p_treatment {
                treatment_successes += 1;
            }
        }
        differences.push((n, treatment_successes as i64 - control_successes as i64));
        control_total += control_successes;
        treatment_total += treatment_successes;
    }

    let estimate = (treatment_total as f64 - control_total as f64) / total_n as f64;
    let mut sum_sq = 0.0;
    for (n, cluster_difference) in &differences {
        let residual = *cluster_difference as f64 - *n as f64 * estimate;
        sum_sq += residual * residual;
    }
    let variance = clusters as f64 / (clusters - 1) as f64
        * sum_sq
        / ((total_n * total_n) as f64);
    let se = variance.max(0.0).sqrt();

    // Degenerate samples are deliberately not granted a zero-width interval.
    let (lower_95, upper_95, passed_gate) = if se.is_finite() && se > f64::EPSILON {
        let margin = t_critical_975(clusters - 1) * se;
        let lower = estimate - margin;
        let upper = estimate + margin;
        (
            Some(lower),
            Some(upper),
            estimate >= config.minimum_observed_gain && lower > 0.0,
        )
    } else {
        (None, None, false)
    };

    Trial {
        estimate,
        lower_95,
        upper_95,
        passed_gate,
        generated_gain: (generated_treatment_probability - generated_baseline_probability)
            / total_n as f64,
    }
}

fn wilson_interval(successes: usize, trials: usize) -> (f64, f64) {
    let n = trials as f64;
    let p = successes as f64 / n;
    let z2 = Z_975 * Z_975;
    let denominator = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denominator;
    let half = Z_975 * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
    ((center - half).max(0.0), (center + half).min(1.0))
}

fn build_report(config: &Config) -> Value {
    let mut rng = Rng::new(config.seed);
    let mut gate_successes = 0usize;
    let mut valid_intervals = 0usize;
    let mut gains = Vec::with_capacity(config.simulations);
    let mut generated_gains = Vec::with_capacity(config.simulations);
    let mut lower_bounds = Vec::with_capacity(config.simulations);
    let mut upper_bounds = Vec::with_capacity(config.simulations);

    for _ in 0..config.simulations {
        let trial = simulate_trial(config, &mut rng);
        if trial.passed_gate {
            gate_successes += 1;
        }
        if let (Some(lower), Some(upper)) = (trial.lower_95, trial.upper_95) {
            valid_intervals += 1;
            lower_bounds.push(lower);
            upper_bounds.push(upper);
        }
        gains.push(trial.estimate);
        generated_gains.push(trial.generated_gain);
    }

    let power = gate_successes as f64 / config.simulations as f64;
    let (power_lower, power_upper) = wilson_interval(gate_successes, config.simulations);
    let average_gain = gains.iter().sum::<f64>() / gains.len() as f64;
    let average_generated_gain =
        generated_gains.iter().sum::<f64>() / generated_gains.len() as f64;
    let average_lower = (!lower_bounds.is_empty())
        .then(|| lower_bounds.iter().sum::<f64>() / lower_bounds.len() as f64);
    let average_upper = (!upper_bounds.is_empty())
        .then(|| upper_bounds.iter().sum::<f64>() / upper_bounds.len() as f64);

    json!({
        "schema_version": "pitb.power-planning-report.v0",
        "report_kind": "simulation_based_power_planning_not_model_evaluation",
        "configuration": {
            "baseline_rate": config.baseline_rate,
            "expected_gain_input": config.expected_gain,
            "minimum_observed_gain_gate": config.minimum_observed_gain,
            "icc": config.icc,
            "cluster_count": config.cluster_sizes.len(),
            "cluster_sizes": config.cluster_sizes,
            "total_targets_per_simulated_study": config.cluster_sizes.iter().sum::<usize>(),
            "simulations": config.simulations,
            "seed": config.seed,
            "confidence_level": 0.95
        },
        "generative_model": {
            "cluster_difficulty": if config.icc == 0.0 { "fixed_baseline_probability" } else { "beta_binomial_shared_family_difficulty" },
            "treatment_probability": "min(cluster_control_probability + expected_gain_input, 1)",
            "within_cluster_outcomes": "independent Bernoulli conditional on cluster probabilities",
            "pairing": "same simulated target clusters receive both control and treatment",
            "interval": "cluster-level paired difference t interval with Cornish-Fisher critical-value approximation",
            "gate": "estimated gain >= minimum_observed_gain_gate AND 95% lower bound > 0; degenerate intervals do not pass"
        },
        "results": {
            "estimated_gate_power": power,
            "estimated_gate_power_wilson_95": { "lower": power_lower, "upper": power_upper },
            "gate_passes": gate_successes,
            "valid_interval_trials": valid_intervals,
            "degenerate_interval_trials": config.simulations - valid_intervals,
            "mean_observed_gain": average_gain,
            "mean_generated_marginal_gain": average_generated_gain,
            "mean_interval_bounds_over_valid_trials": {
                "lower": average_lower,
                "upper": average_upper
            }
        },
        "limitations": [
            "Planning model only; these values are not PITB results or proof of real benchmark power.",
            "The beta-binomial family effect is a sensitivity model, not an estimate of PITB's unknown intracluster correlation.",
            "Probability clipping at 1 can make the generated marginal gain smaller than the configured expected_gain_input; inspect mean_generated_marginal_gain.",
            "The interval is approximate and is not a substitute for the preregistered final analysis or an independently reviewed power plan.",
            "Use actual eligible family/component sizes and a range of plausible baseline rates, gains, and ICC values; do not tune the assumptions after inspecting test outcomes."
        ]
    })
}

fn run(config: &Config, output: &Path) -> Result<Value> {
    let report = build_report(config);
    let encoded = serde_json::to_vec_pretty(&report).context("serialize power report")?;
    fs::write(output, encoded).with_context(|| format!("write report {}", output.display()))?;
    Ok(report)
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() == 2 && (args[1] == "--help" || args[1] == "-h") {
        println!(
            "usage: {} <baseline_rate> <expected_gain> <minimum_observed_gain> <icc_0_to_lt_1> <cluster_sizes_csv> <simulations>=1000 <seed> <output.json>\nExample: 0.35 0.15 0.10 0.05 2,3,1,4,... 10000 12345 power.json",
            args.first().map(String::as_str).unwrap_or("plan_pitb_power")
        );
        return Ok(());
    }

    let config = parse_config(&args)?;
    let output = Path::new(&args[8]);
    let report = run(&config, output)?;
    println!(
        "PITB_POWER_PLAN_WRITTEN: clusters={} targets={} simulations={} gate_power={:.4} gate_passes={}/{}",
        report["configuration"]["cluster_count"].as_u64().unwrap_or(0),
        report["configuration"]["total_targets_per_simulated_study"].as_u64().unwrap_or(0),
        report["configuration"]["simulations"].as_u64().unwrap_or(0),
        report["results"]["estimated_gate_power"].as_f64().unwrap_or(0.0),
        report["results"]["gate_passes"].as_u64().unwrap_or(0),
        report["configuration"]["simulations"].as_u64().unwrap_or(0)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(seed: u64, simulations: usize) -> Config {
        Config {
            baseline_rate: 0.35,
            expected_gain: 0.15,
            minimum_observed_gain: 0.10,
            icc: 0.05,
            cluster_sizes: vec![2, 3, 1, 4, 2, 2, 3, 1, 4, 2, 3, 1, 2, 4, 1, 3, 2, 2, 4, 1],
            simulations,
            seed,
        }
    }

    #[test]
    fn rng_is_reproducible() {
        let mut a = Rng::new(123);
        let mut b = Rng::new(123);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn beta_draws_stay_in_probability_range() {
        let mut rng = Rng::new(99);
        for _ in 0..1_000 {
            let sample = rng.beta(2.0, 3.0);
            assert!(sample.is_finite() && (0.0..=1.0).contains(&sample));
        }
    }

    #[test]
    fn critical_value_is_finite_and_decreases_with_more_clusters() {
        assert!(t_critical_975(19).is_finite());
        assert!(t_critical_975(99) < t_critical_975(19));
        assert!(t_critical_975(19) > Z_975);
    }

    #[test]
    fn report_is_reproducible_for_fixed_config() {
        let c = config(12345, MIN_SIMULATIONS);
        assert_eq!(build_report(&c), build_report(&c));
    }

    #[test]
    fn report_labels_itself_as_planning_not_empirical_performance() {
        let report = build_report(&config(7, MIN_SIMULATIONS));
        assert_eq!(
            report["report_kind"],
            "simulation_based_power_planning_not_model_evaluation"
        );
        assert!(report["limitations"].as_array().is_some());
        assert!(report["results"]["estimated_gate_power"].as_f64().is_some());
    }

    #[test]
    fn degenerate_seed_case_does_not_pass_with_zero_width_interval() {
        let c = Config {
            baseline_rate: 0.5,
            expected_gain: 0.5,
            minimum_observed_gain: 0.1,
            icc: 0.0,
            cluster_sizes: vec![1; MIN_CLUSTERS],
            simulations: MIN_SIMULATIONS,
            seed: 1,
        };
        let mut rng = Rng::new(c.seed);
        let trial = simulate_trial(&c, &mut rng);
        if trial.lower_95.is_none() {
            assert!(!trial.passed_gate);
        }
    }
}
