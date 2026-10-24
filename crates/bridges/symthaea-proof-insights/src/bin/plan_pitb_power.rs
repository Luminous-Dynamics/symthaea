// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic simulation-based power planning for PITB's paired clustered
//! exact-target success comparison.
//!
//! This is a *planning aid*, not a proof checker or an empirical benchmark
//! result. It models shared family difficulty, heterogeneous family-level
//! treatment effects, and paired target potential outcomes. It applies the paired
//! component-percentile-bootstrap interval rule used by PITB's frontier analysis.
//! It remains a planning sensitivity model, not a benchmark result.

use std::{env, fs, path::Path};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

const MIN_CLUSTERS: usize = 20;
const MIN_SIMULATIONS: usize = 1_000;
const MIN_BOOTSTRAP_REPLICATES: usize = 1_000;
const Z_975: f64 = 1.959_963_984_540_054;

#[derive(Debug, Clone)]
struct Config {
    baseline_rate: f64,
    expected_gain: f64,
    minimum_observed_gain: f64,
    icc: f64,
    cluster_effect_sd: f64,
    pair_association: f64,
    cluster_sizes: Vec<usize>,
    simulations: usize,
    bootstrap_replicates: usize,
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
    if args.len() != 12 || args.get(1).is_some_and(|x| x == "--help" || x == "-h") {
        bail!(
            "usage: {} <baseline_rate> <expected_gain> <minimum_observed_gain> <icc_0_to_lt_1> <cluster_effect_sd_0_to_1> <pair_association_0_to_1> <cluster_sizes_csv> <simulations>=1000 <bootstrap_replicates>=1000 <seed> <output.json>",
            args.first().map(String::as_str).unwrap_or("plan_pitb_power")
        );
    }

    let baseline_rate = parse_f64(&args[1], "baseline_rate")?;
    let expected_gain = parse_f64(&args[2], "expected_gain")?;
    let minimum_observed_gain = parse_f64(&args[3], "minimum_observed_gain")?;
    let icc = parse_f64(&args[4], "icc")?;
    let cluster_effect_sd = parse_f64(&args[5], "cluster_effect_sd")?;
    let pair_association = parse_f64(&args[6], "pair_association")?;
    let cluster_sizes = args[7]
        .split(',')
        .map(|item| {
            let size = item.trim().parse::<usize>().context("parse cluster size")?;
            if size == 0 {
                bail!("cluster sizes must all be positive integers");
            }
            Ok(size)
        })
        .collect::<Result<Vec<_>>>()?;
    let simulations = args[8].parse::<usize>().context("parse simulations")?;
    let bootstrap_replicates = args[9].parse::<usize>().context("parse bootstrap replicates")?;
    let seed = args[10].parse::<u64>().context("parse seed")?;

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
    if !(0.0..=1.0).contains(&cluster_effect_sd) {
        bail!("cluster_effect_sd must be in [0, 1]");
    }
    if !(0.0..=1.0).contains(&pair_association) {
        bail!("pair_association must be in [0, 1]");
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
    let total_targets = cluster_sizes.iter().try_fold(0usize, |sum, size| {
        sum.checked_add(*size).context("sum of cluster sizes overflows usize")
    })?;
    if total_targets == 0 {
        bail!("cluster-size vector must contain at least one target");
    }
    if icc > 0.0 {
        let concentration = (1.0 - icc) / icc;
        let alpha = baseline_rate * concentration;
        let beta = (1.0 - baseline_rate) * concentration;
        if !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
            bail!("icc and baseline_rate produce invalid beta-binomial shape parameters");
        }
    }
    if simulations < MIN_SIMULATIONS {
        bail!("simulations must be at least {MIN_SIMULATIONS}");
    }
    if bootstrap_replicates < MIN_BOOTSTRAP_REPLICATES {
        bail!("bootstrap replicates must be at least {MIN_BOOTSTRAP_REPLICATES}");
    }

    Ok(Config {
        baseline_rate,
        expected_gain,
        minimum_observed_gain,
        icc,
        cluster_effect_sd,
        pair_association,
        cluster_sizes,
        simulations,
        bootstrap_replicates,
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

fn percentile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&probability) {
        return None;
    }
    let position = probability * (sorted.len() - 1) as f64;
    let lower_index = position.floor() as usize;
    let upper_index = position.ceil() as usize;
    let weight = position - lower_index as f64;
    let lower = *sorted.get(lower_index)?;
    let upper = *sorted.get(upper_index)?;
    Some(lower * (1.0 - weight) + upper * weight)
}

fn bootstrap_paired_interval(
    differences: &[(usize, i64)],
    replicates: usize,
    rng: &mut Rng,
) -> Option<[f64; 2]> {
    if differences.len() < MIN_CLUSTERS || replicates < MIN_BOOTSTRAP_REPLICATES {
        return None;
    }
    let mut estimates = Vec::with_capacity(replicates);
    for _ in 0..replicates {
        let mut delta_sum = 0i64;
        let mut target_count = 0usize;
        for _ in 0..differences.len() {
            let index = (rng.next_u64() as usize) % differences.len();
            let (cluster_targets, cluster_delta) = differences[index];
            delta_sum += cluster_delta;
            target_count += cluster_targets;
        }
        if target_count > 0 {
            estimates.push(delta_sum as f64 / target_count as f64);
        }
    }
    if estimates.len() != replicates {
        return None;
    }
    estimates.sort_by(f64::total_cmp);
    Some([percentile(&estimates, 0.025)?, percentile(&estimates, 0.975)?])
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
        // Each family receives its own treatment-effect deviation, modelling
        // cross-family heterogeneity rather than assuming an identical gain.
        let family_gain = config.expected_gain + config.cluster_effect_sd * rng.normal();
        let p_treatment = (p_control + family_gain).clamp(0.0, 1.0);
        generated_baseline_probability += p_control * n as f64;
        generated_treatment_probability += p_treatment * n as f64;

        let mut control_successes = 0usize;
        let mut treatment_successes = 0usize;
        for _ in 0..n {
            let (control_success, treatment_success) = paired_bernoulli(
                rng,
                p_control,
                p_treatment,
                config.pair_association,
            );
            control_successes += if control_success { 1 } else { 0 };
            treatment_successes += if treatment_success { 1 } else { 0 };
        }
        differences.push((n, treatment_successes as i64 - control_successes as i64));
        control_total += control_successes;
        treatment_total += treatment_successes;
    }

    let estimate = (treatment_total as f64 - control_total as f64) / total_n as f64;
    // Resample whole independent components, matching PITB's paired-delta
    // percentile-bootstrap design. A degenerate interval does not pass.
    let interval = bootstrap_paired_interval(
        &differences,
        config.bootstrap_replicates,
        rng,
    );
    let (lower_95, upper_95, passed_gate) = match interval {
        Some([lower, upper]) if upper - lower > f64::EPSILON => (
            Some(lower),
            Some(upper),
            estimate >= config.minimum_observed_gain && lower > 0.0,
        ),
        _ => (None, None, false),
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
            "cluster_effect_sd": config.cluster_effect_sd,
            "pair_association": config.pair_association,
            "cluster_count": config.cluster_sizes.len(),
            "cluster_sizes": config.cluster_sizes,
            "total_targets_per_simulated_study": config.cluster_sizes.iter().sum::<usize>(),
            "simulations": config.simulations,
            "bootstrap_replicates": config.bootstrap_replicates,
            "seed": config.seed,
            "confidence_level": 0.95
        },
        "generative_model": {
            "cluster_difficulty": if config.icc == 0.0 { "fixed_baseline_probability" } else { "beta_binomial_shared_family_difficulty" },
            "treatment_probability": "clamp(cluster_control_probability + expected_gain_input + family_specific_normal_deviation, 0, 1)",
            "family_treatment_effect_heterogeneity": "normal family-level effect deviation with configured probability-scale standard deviation; outcomes are clipped to [0,1]",
            "paired_target_outcomes": "bivariate Bernoulli potential outcomes using configured fraction of maximum feasible positive joint-success association",
            "pair_association_semantics": "0 means independent potential outcomes conditional on family probabilities; 1 means the Frechet upper bound on joint successes; this parameter is not a Pearson correlation",
            "pairing": "control and treatment potential outcomes are simulated on the same target inside each shared family",
            "interval": "paired component-percentile bootstrap, resampling whole components with replacement and recomputing the target-weighted treatment-minus-control difference",
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
            "The beta-binomial family effect and family-specific treatment-effect variation are sensitivity assumptions, not estimates from PITB.",
            "Family treatment-effect deviations can be negative or positive; the resulting target probabilities are clipped to [0,1], so inspect mean_generated_marginal_gain.",
            "Pair association is a sensitivity parameter that scales feasible positive joint-success association, not an empirically estimated Pearson correlation.",
            "The interval uses a finite number of paired component-bootstrap replicates; increase bootstrap_replicates for a stability analysis and match the final preregistered analysis setting.",
            "Degenerate bootstrap intervals are suppressed and never counted as passing the gate.",
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
            "usage: {} <baseline_rate> <expected_gain> <minimum_observed_gain> <icc_0_to_lt_1> <cluster_effect_sd_0_to_1> <pair_association_0_to_1> <cluster_sizes_csv> <simulations>=1000 <bootstrap_replicates>=1000 <seed> <output.json>\nExample: 0.35 0.15 0.10 0.05 0.05 0.5 2,3,1,4,... 10000 1000 12345 power.json",
            args.first().map(String::as_str).unwrap_or("plan_pitb_power")
        );
        return Ok(());
    }

    let config = parse_config(&args)?;
    let output = Path::new(&args[11]);
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
            cluster_effect_sd: 0.05,
            pair_association: 0.5,
            cluster_sizes: vec![2, 3, 1, 4, 2, 2, 3, 1, 4, 2, 3, 1, 2, 4, 1, 3, 2, 2, 4, 1],
            simulations,
            bootstrap_replicates: MIN_BOOTSTRAP_REPLICATES,
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
    fn percentile_uses_linear_interpolation() {
        let values = [0.0, 10.0, 20.0, 30.0];
        assert_eq!(percentile(&values, 0.25), Some(7.5));
        assert_eq!(percentile(&values, 0.975), Some(29.25));
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
    fn parser_accepts_all_config_arguments() {
        let args = vec![
            "plan_pitb_power".to_owned(),
            "0.35".to_owned(),
            "0.15".to_owned(),
            "0.10".to_owned(),
            "0.05".to_owned(),
            "0.05".to_owned(),
            "0.5".to_owned(),
            vec!["2"; MIN_CLUSTERS].join(","),
            "1000".to_owned(),
            "1000".to_owned(),
            "123".to_owned(),
            "power.json".to_owned(),
        ];
        assert!(parse_config(&args).is_ok());
    }

    #[test]
    fn parser_rejects_fewer_than_twenty_independent_clusters() {
        let args = vec![
            "plan_pitb_power".to_owned(),
            "0.35".to_owned(),
            "0.15".to_owned(),
            "0.10".to_owned(),
            "0.05".to_owned(),
            "0.05".to_owned(),
            "0.5".to_owned(),
            "2,3,1".to_owned(),
            "1000".to_owned(),
            "1000".to_owned(),
            "123".to_owned(),
            "power.json".to_owned(),
        ];
        assert!(parse_config(&args).is_err());
    }

    #[test]
    fn paired_sampler_respects_frechet_upper_bound_at_full_association() {
        let mut rng = Rng::new(901);
        for _ in 0..10_000 {
            let (control, treatment) = paired_bernoulli(&mut rng, 0.3, 0.6, 1.0);
            assert!(!control || treatment);
        }
    }

    #[test]
    fn paired_sampler_matches_marginals_at_extreme_associations() {
        let mut rng = Rng::new(77);
        let mut control_successes = 0usize;
        let mut treatment_successes = 0usize;
        let n = 50_000;
        for _ in 0..n {
            let (control, treatment) = paired_bernoulli(&mut rng, 0.25, 0.4, 0.5);
            control_successes += if control { 1 } else { 0 };
            treatment_successes += if treatment { 1 } else { 0 };
        }
        assert!((control_successes as f64 / n as f64 - 0.25).abs() < 0.015);
        assert!((treatment_successes as f64 / n as f64 - 0.4).abs() < 0.015);
    }

    #[test]
    fn parser_rejects_out_of_range_pair_association() {
        let args = vec![
            "plan_pitb_power".to_owned(), "0.35".to_owned(), "0.15".to_owned(),
            "0.10".to_owned(), "0.05".to_owned(), "0.05".to_owned(),
            "1.01".to_owned(), vec!["2"; MIN_CLUSTERS].join(","),
            "1000".to_owned(), "1000".to_owned(), "123".to_owned(), "power.json".to_owned(),
        ];
        assert!(parse_config(&args).is_err());
    }

    #[test]
    fn degenerate_seed_case_does_not_pass_with_zero_width_interval() {
        let c = Config {
            baseline_rate: 0.5,
            expected_gain: 0.5,
            minimum_observed_gain: 0.1,
            icc: 0.0,
            cluster_effect_sd: 0.0,
            pair_association: 0.0,
            cluster_sizes: vec![1; MIN_CLUSTERS],
            simulations: MIN_SIMULATIONS,
            bootstrap_replicates: MIN_BOOTSTRAP_REPLICATES,
            seed: 1,
        };
        let mut rng = Rng::new(c.seed);
        let trial = simulate_trial(&c, &mut rng);
        if trial.lower_95.is_none() {
            assert!(!trial.passed_gate);
        }
    }
}
