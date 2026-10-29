//! VIS-DVPC-041: exact-path audit of MOPE running lower/upper endpoints.
//!
//! This is an exact, finite-horizon synthetic coverage audit, not a deployment
//! claim. It reuses the streaming sufficient-statistic and optimizer modules.
//! For a 95% two-tail sequence, alpha_side = 0.025 is used on each side so the
//! union bound allocates at most 0.05 total noncoverage probability.
#![deny(unsafe_code)]

#[path = "vision-dvpc-040-mope-sufficient-stats.rs"]
mod stream;

use std::env;
use std::fmt::Write as _;

const ALPHA_SIDE: f64 = 0.025;
const HORIZON: usize = 8;
const PATH_COUNT: usize = 65_536;
const TRUE_VALUE: f64 = 0.5;
const POLICY_ACTION_ONE: f64 = 0.5;
const PSI: f64 = 2.0 - 4.0 * std::f64::consts::LN_2;
const ROOT_TOL: f64 = 2e-7;
const WEALTH_TOL: f64 = 2e-8;
const ENDPOINT_TOL: f64 = 1e-10;
const VALUE_GRID: [f64; 21] = [
    0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
];

#[derive(Clone, Copy, Debug)]
struct Scenario {
    name: &'static str,
    behavior_action_one: f64,
    reward_zero_action_zero: f64,
    reward_one_action_one: f64,
    max_weight: f64,
}

impl Scenario {
    fn healthy() -> Self {
        Self {
            name: "healthy",
            behavior_action_one: 0.5,
            reward_zero_action_zero: 0.2,
            reward_one_action_one: 0.8,
            max_weight: 1.0,
        }
    }

    fn weak_overlap() -> Self {
        Self {
            name: "weak_overlap",
            behavior_action_one: 0.99,
            reward_zero_action_zero: 0.2,
            reward_one_action_one: 0.8,
            max_weight: 50.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Outcome {
    action: u8,
    reward: f64,
    weight: f64,
    probability: f64,
}

fn outcome(scenario: Scenario, digit: usize) -> Outcome {
    let p = scenario.behavior_action_one;
    match digit {
        0 => Outcome {
            action: 0,
            reward: 0.0,
            weight: POLICY_ACTION_ONE / (1.0 - p),
            probability: (1.0 - p) * (1.0 - scenario.reward_zero_action_zero),
        },
        1 => Outcome {
            action: 0,
            reward: 1.0,
            weight: POLICY_ACTION_ONE / (1.0 - p),
            probability: (1.0 - p) * scenario.reward_zero_action_zero,
        },
        2 => Outcome {
            action: 1,
            reward: 0.0,
            weight: POLICY_ACTION_ONE / p,
            probability: p * (1.0 - scenario.reward_one_action_one),
        },
        3 => Outcome {
            action: 1,
            reward: 1.0,
            weight: POLICY_ACTION_ONE / p,
            probability: p * scenario.reward_one_action_one,
        },
        _ => unreachable!("base-4 outcome digit"),
    }
}

#[derive(Clone, Copy, Debug)]
struct BetRecord {
    w: f64,
    r: f64,
    lambda1: f64,
    lambda2: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct LookTotals {
    probability_mass: f64,
    pointwise_coverage_mass: f64,
    weighted_lower_endpoint: f64,
    weighted_upper_endpoint: f64,
    empty_interval_mass: f64,
    lower_monotonicity_violation_mass: f64,
    upper_monotonicity_violation_mass: f64,
    root_certificate_checks: u64,
    root_certificate_failures: u64,
}

#[derive(Clone, Debug)]
struct ScenarioResult {
    name: &'static str,
    horizon: usize,
    path_count: usize,
    looks: Vec<LookTotals>,
    sequence_coverage: f64,
    sequence_false_exclusion: f64,
    root_certificate_checks: u64,
    root_certificate_failures: u64,
    probability_mass_error: f64,
}

fn close(a: f64, b: f64, tolerance: f64) -> bool {
    (a - b).abs() <= tolerance * (1.0 + a.abs().max(b.abs()))
}

fn direct_polynomial(history: &[BetRecord], value: f64) -> f64 {
    history.iter().map(|record| {
        let z = record.w - 1.0;
        let d = record.w * record.r - value;
        let linear = record.lambda1 * z + record.lambda2 * d;
        let quadratic = PSI * (
            record.lambda1 * record.lambda1 * z * z
                + 2.0 * record.lambda1 * record.lambda2 * z * d
                + record.lambda2 * record.lambda2 * d * d
        );
        linear + quadratic
    }).sum()
}

fn direct_log_wealth(history: &[BetRecord], value: f64) -> Result<f64, String> {
    let mut total = 0.0;
    for record in history {
        let factor = 1.0 + record.lambda1 * (record.w - 1.0)
            + record.lambda2 * (record.w * record.r - value);
        if !factor.is_finite() || factor < 0.5 - WEALTH_TOL {
            return Err(format!("bet factor below 1/2 at value {value}: {factor}"));
        }
        total += factor.ln();
    }
    Ok(total)
}

/// Certify a candidate endpoint from the direct sufficient-statistic expression,
/// the direct product-wealth process, and the monotonicity condition lambda_2 >= 0.
/// The grid check is a numerical control; nonincreasing wealth for all v follows
/// analytically because each positive factor is nonincreasing in v.
fn certify_root(history: &[BetRecord], root: f64, level: f64) -> bool {
    if !root.is_finite() || !(0.0..=1.0).contains(&root) {
        return false;
    }
    if history.iter().any(|record| record.lambda2 < -WEALTH_TOL) {
        return false;
    }
    if !close(direct_polynomial(history, root), level, ROOT_TOL) {
        return false;
    }
    let at_root = match direct_log_wealth(history, root) {
        Ok(value) => value,
        Err(_) => return false,
    };
    if at_root + WEALTH_TOL < level {
        return false;
    }
    for &candidate in &VALUE_GRID {
        if candidate <= root + ENDPOINT_TOL {
            match direct_log_wealth(history, candidate) {
                Ok(value) if value + WEALTH_TOL >= level => {}
                _ => return false,
            }
        }
    }
    true
}

fn positive_root_control() -> Result<f64, String> {
    // A deliberately strong fixed-logger control creates a root without
    // relying on the horizon-8 DGPs being informative enough to cross.
    let mut fixed_stream = stream::StreamState::new(1.0, ALPHA_SIDE)?;
    let mut history = Vec::new();
    let level = (2.0 / ALPHA_SIDE).ln();
    for _ in 0..24 {
        let lambda = fixed_stream.current_lambda();
        let report = fixed_stream.observe(1.0, 1.0)?;
        history.push(BetRecord { w: 1.0, r: 1.0, lambda1: lambda.x, lambda2: lambda.y });
        if let Some(root) = report.selected_safe_root {
            if root > 0.0 && certify_root(&history, root, level) {
                return Ok(root);
            }
        }
        fixed_stream.choose_next_bet(0.0)?;
    }
    Err("positive root-certificate control failed to find a certified root".to_owned())
}

fn run_scenario(scenario: Scenario, horizon: usize) -> Result<ScenarioResult, String> {
    if horizon == 0 || horizon > 8 {
        return Err("exact path audit supports horizons 1 through 8".to_owned());
    }
    let path_count = 4_usize.pow(horizon as u32);
    let level = (2.0 / ALPHA_SIDE).ln();
    let mut looks = vec![LookTotals::default(); horizon];
    let mut sequence_coverage_mass = 0.0;
    let mut sequence_false_exclusion_mass = 0.0;
    let mut root_certificate_checks = 0u64;
    let mut root_certificate_failures = 0u64;
    let mut probability_mass_error: f64 = 0.0;

    for encoded_path in 0..path_count {
        // Compute the probability of the complete ordered path before taking
        // any per-look metric; each prefix is then weighted by its full path
        // probability, which correctly sums over all suffixes.
        let mut path_code = encoded_path;
        let mut path_probability = 1.0;
        for _ in 0..horizon {
            let digit = path_code & 3;
            path_code >>= 2;
            path_probability *= outcome(scenario, digit).probability;
        }
        if path_probability == 0.0 {
            continue;
        }

        let mut lower_stream = stream::StreamState::new(scenario.max_weight, ALPHA_SIDE)?;
        let mut complement_stream = stream::StreamState::new(scenario.max_weight, ALPHA_SIDE)?;
        let mut lower_history: Vec<BetRecord> = Vec::with_capacity(horizon);
        let mut complement_history: Vec<BetRecord> = Vec::with_capacity(horizon);
        let mut lower_endpoint = 0.0f64;
        let mut complement_endpoint = 0.0f64;
        let mut ever_excluded_true_value = false;
        let mut code = encoded_path;

        for look_index in 0..horizon {
            let digit = code & 3;
            code >>= 2;
            let event = outcome(scenario, digit);
            if event.weight > scenario.max_weight + ENDPOINT_TOL {
                return Err(format!("{} emitted an importance ratio above W", scenario.name));
            }

            let previous_lower = lower_endpoint;
            let previous_upper = 1.0 - complement_endpoint;

            // The bets used on the observation are selected before this
            // outcome is observed; only afterward are endpoints and next bets updated.
            let lower_lambda = lower_stream.current_lambda();
            let complement_lambda = complement_stream.current_lambda();
            let lower_report = lower_stream.observe(event.weight, event.reward)?;
            let complement_reward = 1.0 - event.reward;
            let complement_report = complement_stream.observe(event.weight, complement_reward)?;

            lower_history.push(BetRecord {
                w: event.weight,
                r: event.reward,
                lambda1: lower_lambda.x,
                lambda2: lower_lambda.y,
            });
            complement_history.push(BetRecord {
                w: event.weight,
                r: complement_reward,
                lambda1: complement_lambda.x,
                lambda2: complement_lambda.y,
            });

            if let Some(root) = lower_report.selected_safe_root {
                if root > lower_endpoint + ENDPOINT_TOL {
                    root_certificate_checks += 1;
                    looks[look_index].root_certificate_checks += 1;
                    if certify_root(&lower_history, root, level) {
                        lower_endpoint = root;
                    } else {
                        root_certificate_failures += 1;
                        looks[look_index].root_certificate_failures += 1;
                    }
                }
            }
            if let Some(root) = complement_report.selected_safe_root {
                if root > complement_endpoint + ENDPOINT_TOL {
                    root_certificate_checks += 1;
                    looks[look_index].root_certificate_checks += 1;
                    if certify_root(&complement_history, root, level) {
                        complement_endpoint = root;
                    } else {
                        root_certificate_failures += 1;
                        looks[look_index].root_certificate_failures += 1;
                    }
                }
            }

            let lower_endpoint = lower_endpoint.max(previous_lower);
            let upper_endpoint = (1.0 - complement_endpoint).min(previous_upper);

            if lower_endpoint + ENDPOINT_TOL < previous_lower {
                looks[look_index].lower_monotonicity_violation_mass += path_probability;
            }
            if upper_endpoint > previous_upper + ENDPOINT_TOL {
                looks[look_index].upper_monotonicity_violation_mass += path_probability;
            }

            let interval_is_ordered = lower_endpoint <= upper_endpoint + ENDPOINT_TOL;
            if !interval_is_ordered {
                looks[look_index].empty_interval_mass += path_probability;
            }
            let covers_true = interval_is_ordered
                && lower_endpoint <= TRUE_VALUE + ENDPOINT_TOL
                && upper_endpoint >= TRUE_VALUE - ENDPOINT_TOL;
            if covers_true {
                looks[look_index].pointwise_coverage_mass += path_probability;
            } else {
                ever_excluded_true_value = true;
            }
            looks[look_index].probability_mass += path_probability;
            looks[look_index].weighted_lower_endpoint += path_probability * lower_endpoint;
            looks[look_index].weighted_upper_endpoint += path_probability * upper_endpoint;

            lower_stream.choose_next_bet(lower_endpoint)?;
            complement_stream.choose_next_bet(complement_endpoint)?;
        }
        if ever_excluded_true_value {
            sequence_false_exclusion_mass += path_probability;
        } else {
            sequence_coverage_mass += path_probability;
        }
    }

    for look in &looks {
        probability_mass_error = probability_mass_error.max((look.probability_mass - 1.0).abs());
    }
    Ok(ScenarioResult {
        name: scenario.name,
        horizon,
        path_count,
        looks,
        sequence_coverage: sequence_coverage_mass,
        sequence_false_exclusion: sequence_false_exclusion_mass,
        root_certificate_checks,
        root_certificate_failures,
        probability_mass_error,
    })
}

fn audit() -> Result<String, String> {
    let scenarios = [Scenario::healthy(), Scenario::weak_overlap()];
    let mut output = String::from(
        "record_type\tscenario\tlook_or_horizon\tmetric\tvalue\n"
    );
    let control_root = positive_root_control()?;
    let _ = writeln!(output, "ROOT_CERTIFICATE_CONTROL\tstatus\tOK\tcertified_root\t{control_root:.12}");
    for scenario in scenarios {
        let result = run_scenario(scenario, HORIZON)?;
        if result.path_count != PATH_COUNT { return Err(format!("wrong exact path count: {}", result.path_count)); }
        for (index, stats) in result.looks.iter().enumerate() {
            let look = index + 1;
            let mass = stats.probability_mass;
            let coverage = if mass > 0.0 { stats.pointwise_coverage_mass / mass } else { 0.0 };
            let mean_lower = if mass > 0.0 { stats.weighted_lower_endpoint / mass } else { 0.0 };
            let mean_upper = if mass > 0.0 { stats.weighted_upper_endpoint / mass } else { 0.0 };
            let reversed = if mass > 0.0 { stats.empty_interval_mass / mass } else { 0.0 };
            let lower_mono = if mass > 0.0 { stats.lower_monotonicity_violation_mass / mass } else { 0.0 };
            let upper_mono = if mass > 0.0 { stats.upper_monotonicity_violation_mass / mass } else { 0.0 };
            println_metric(&mut output, "LOOK_METRIC", result.name, look, "probability_mass", mass);
            println_metric(&mut output, "LOOK_DETAIL", result.name, look, "pointwise_coverage", coverage);
            println_metric(&mut output, "LOOK_DETAIL", result.name, look, "mean_lower_endpoint", mean_lower);
            println_metric(&mut output, "LOOK_DETAIL", result.name, look, "mean_upper_endpoint", mean_upper);
            println_metric(&mut output, "LOOK_DETAIL", result.name, look, "mean_interval_width", (mean_upper - mean_lower).max(0.0));
            println_metric(&mut output, "LOOK_DETAIL", result.name, look, "empty_or_reversed_interval_probability", reversed);
            println_metric(&mut output, "LOOK_DETAIL", result.name, look, "lower_monotonicity_violation_probability", lower_mono);
            println_metric(&mut output, "LOOK_DETAIL", result.name, look, "upper_monotonicity_violation_probability", upper_mono);
            let _ = writeln!(output, "LOOK_DETAIL\t{}\t{}\troot_certificate_checks\t{}", result.name, look, stats.root_certificate_checks);
            let _ = writeln!(output, "LOOK_DETAIL\t{}\t{}\troot_certificate_failures\t{}", result.name, look, stats.root_certificate_failures);
        }
        println_metric(&mut output, "SCENARIO", result.name, result.horizon, "sequence_coverage", result.sequence_coverage);
        println_metric(&mut output, "SCENARIO_METRIC", result.name, result.horizon, "sequence_false_exclusion_probability", result.sequence_false_exclusion);
        println_metric(&mut output, "SCENARIO_METRIC", result.name, result.horizon, "root_certificate_checks", result.root_certificate_checks as f64);
        println_metric(&mut output, "SCENARIO_METRIC", result.name, result.horizon, "root_certificate_failures", result.root_certificate_failures as f64);
        println_metric(&mut output, "SCENARIO_METRIC", result.name, result.horizon, "probability_mass_max_abs_error", result.probability_mass_error);
        let _ = writeln!(
            output,
            "ROOT_CERTIFICATE\t{}\tchecked\t{}\tfailures\t{}",
            result.name, result.root_certificate_checks, result.root_certificate_failures
        );
        let _ = writeln!(
            output,
            "PATHS\t{}\tcount\t{}\ttrue_value\t{:.12}\tmax_weight\t{:.12}",
            result.name, result.path_count, TRUE_VALUE, scenario.max_weight
        );
    }
    Ok(output)
}

fn println_metric(output: &mut String, record_type: &str, scenario: &str, index: usize, metric: &str, value: f64) {
    use std::fmt::Write as _;
    let _ = writeln!(output, "{record_type}\t{scenario}\t{index}\t{metric}\t{value:.12}");
}

fn main() {
    let result = match env::args().nth(1).as_deref() {
        Some("--audit") => audit(),
        _ => Err("usage: vision-dvpc-041-mope-endpoint-audit --audit".to_owned()),
    };
    match result {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_041_ERROR\t{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_outcome_probabilities_sum_to_one_and_support_matches_w() {
        for scenario in [Scenario::healthy(), Scenario::weak_overlap()] {
            let sum: f64 = (0..4).map(|d| outcome(scenario, d).probability).sum();
            assert!(close(sum, 1.0, 1e-12));
            for d in 0..4 {
                let event = outcome(scenario, d);
                assert!(event.weight <= scenario.max_weight + ENDPOINT_TOL);
                assert!((0.0..=1.0).contains(&event.reward));
            }
        }
    }

    #[test]
    fn endpoint_stream_and_path_mass_are_valid_on_short_exact_path_space() {
        for scenario in [Scenario::healthy(), Scenario::weak_overlap()] {
            let result = run_scenario(scenario, 4).unwrap();
            assert_eq!(result.path_count, 256);
            assert!(result.probability_mass_error < 1e-10);
            assert!(result.root_certificate_failures == 0);
            assert!(result.sequence_coverage >= 0.95);
            for look in result.looks {
                assert!(close(look.probability_mass, 1.0, 1e-10));
                assert!(look.lower_monotonicity_violation_mass <= 1e-10);
                assert!(look.upper_monotonicity_violation_mass <= 1e-10);
            }
        }
    }

    #[test]
    fn positive_root_control_exercises_the_certificate_path() {
        let root = positive_root_control().unwrap();
        assert!((0.0..=1.0).contains(&root));
    }

    #[test]
    fn direct_root_certificate_accepts_real_crossing_and_rejects_invalid_bets() {
        let mut fixed_stream = stream::StreamState::new(1.0, ALPHA_SIDE).unwrap();
        let mut history = Vec::new();
        let level = (2.0 / ALPHA_SIDE).ln();
        let mut certified_root = None;
        for _ in 0..24 {
            let lambda = fixed_stream.current_lambda();
            let report = fixed_stream.observe(1.0, 1.0).unwrap();
            history.push(BetRecord { w: 1.0, r: 1.0, lambda1: lambda.x, lambda2: lambda.y });
            if let Some(root) = report.selected_safe_root {
                if root > 0.0 && certify_root(&history, root, level) {
                    certified_root = Some(root);
                }
            }
            fixed_stream.choose_next_bet(0.0).unwrap();
        }
        assert!(certified_root.is_some(), "fixed predictable-bet control must yield a certified root");
        let mut invalid = history.clone();
        invalid[0].lambda2 = -0.1;
        assert!(!certify_root(&invalid, certified_root.unwrap(), level));
    }
}
