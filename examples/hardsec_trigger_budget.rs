//! HARDSEC-007 random-trigger coverage budget calculator.
//!
//! Computes the miss probability for N independent random stimuli when the
//! per-stimulus trigger activation probability is p:
//!
//!     P(no activation in N trials) = (1 - p)^N
//!
//! This is a sampling-budget calculator, not a trigger-probability estimator,
//! formal reachability checker, or proof of Trojan absence. The IID assumption
//! must be stated and justified by the experiment.
//!
//! Example:
//!   cargo run --example hardsec_trigger_budget -- \
//!     --activation-probability 0.0001 --trials 1000 \
//!     --target-miss-probability 0.05

use std::env;
use std::process::ExitCode;

const DEFAULT_TRIALS: u64 = 1_000;
const DEFAULT_TARGET_MISS: f64 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Config {
    activation_probability: f64,
    trials: u64,
    target_miss_probability: f64,
}

fn usage() -> &'static str {
    concat!(
        "usage: hardsec_trigger_budget --activation-probability P ",
        "[--trials N] [--target-miss-probability A]\n",
        "P is the assumed per-trial activation probability in [0, 1].\n",
        "N is the number of independent trials (default: 1000).\n",
        "A is the maximum tolerated probability of zero activations (default: 0.05)."
    )
}

fn parse_probability(name: &str, value: &str, allow_one: bool) -> Result<f64, String> {
    let probability: f64 = value
        .parse()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))?;
    if !probability.is_finite()
        || probability < 0.0
        || if allow_one {
            probability > 1.0
        } else {
            probability >= 1.0
        }
    {
        let range = if allow_one { "[0, 1]" } else { "(0, 1)" };
        return Err(format!("{name} must be in {range}, got {value:?}"));
    }
    Ok(probability)
}

fn parse_args() -> Result<Option<Config>, String> {
    let mut activation_probability = None;
    let mut trials = DEFAULT_TRIALS;
    let mut target_miss_probability = DEFAULT_TARGET_MISS;
    let mut args = env::args().skip(1);

    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            return Ok(None);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--activation-probability" => {
                activation_probability = Some(parse_probability(
                    "activation probability",
                    &value,
                    true,
                )?);
            }
            "--trials" => {
                trials = value
                    .parse::<u64>()
                    .map_err(|error| format!("invalid trial count {value:?}: {error}"))?;
                if trials == 0 {
                    return Err("--trials must be greater than zero".to_owned());
                }
            }
            "--target-miss-probability" => {
                target_miss_probability =
                    parse_probability("target miss probability", &value, false)?;
            }
            _ => return Err(format!("unknown argument: {flag}")),
        }
    }

    Ok(Some(Config {
        activation_probability: activation_probability
            .ok_or("--activation-probability is required")?,
        trials,
        target_miss_probability,
    }))
}

fn miss_probability(activation_probability: f64, trials: u64) -> f64 {
    if activation_probability == 0.0 {
        return 1.0;
    }
    if activation_probability == 1.0 {
        return 0.0;
    }
    ((trials as f64) * (-activation_probability).ln_1p()).exp()
}

fn activation_probability_at_least_once(
    activation_probability: f64,
    trials: u64,
) -> f64 {
    if activation_probability == 0.0 {
        return 0.0;
    }
    if activation_probability == 1.0 {
        return 1.0;
    }
    -(((trials as f64) * (-activation_probability).ln_1p()).exp_m1())
}

fn minimum_trials(
    activation_probability: f64,
    target_miss_probability: f64,
) -> Option<u64> {
    if activation_probability == 0.0 {
        return None;
    }
    if activation_probability == 1.0 {
        return Some(1);
    }

    let estimate = (target_miss_probability.ln()
        / (-activation_probability).ln_1p())
    .ceil();

    if !estimate.is_finite() || estimate > u64::MAX as f64 {
        return None;
    }

    let mut trials = (estimate as u64).max(1);
    while miss_probability(activation_probability, trials) > target_miss_probability {
        trials = trials.checked_add(1)?;
    }
    while trials > 1
        && miss_probability(activation_probability, trials - 1) <= target_miss_probability
    {
        trials -= 1;
    }
    Some(trials)
}

fn scientific(value: f64) -> String {
    format!("{value:.17e}")
}

fn run(config: Config) {
    let miss = miss_probability(config.activation_probability, config.trials);
    let activation_at_least_once =
        activation_probability_at_least_once(config.activation_probability, config.trials);
    let required_trials = minimum_trials(
        config.activation_probability,
        config.target_miss_probability,
    );

    println!("{{");
    println!("  \"schema_version\": \"hardsec-trigger-budget-v1\",");
    println!(
        "  \"assumption\": \"iid_independent_stimuli_with_constant_activation_probability\" ,"
    );
    println!(
        "  \"per_trial_activation_probability\": {},",
        scientific(config.activation_probability)
    );
    println!("  \"trials\": {},", config.trials);
    println!("  \"probability_of_zero_activations\": {},", scientific(miss));
    println!(
        "  \"probability_of_at_least_one_activation\": {},",
        scientific(activation_at_least_once)
    );
    println!(
        "  \"target_miss_probability\": {},",
        scientific(config.target_miss_probability)
    );
    match required_trials {
        Some(value) => println!("  \"minimum_trials_for_target\": {value},"),
        None => println!("  \"minimum_trials_for_target\": null,"),
    }
    println!(
        concat!(
            "  \"interpretation\": ",
            "\"sampling_budget_only_not_trigger_probability_estimation_or_formal_reachability\""
        )
    );
    println!("}}");
}

fn main() -> ExitCode {
    match parse_args() {
        Ok(Some(config)) => {
            run(config);
            ExitCode::SUCCESS
        }
        Ok(None) => {
            println!("{}", usage());
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("{error}\n{}", usage());
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_in_a_thousand_activation_has_about_368_percent_miss_rate_at_1000_trials() {
        let miss = miss_probability(1e-3, 1_000);
        assert!((miss - 0.3677).abs() < 0.0002, "actual miss={miss}");
    }

    #[test]
    fn one_in_ten_thousand_activation_is_missed_about_905_percent_of_the_time() {
        let miss = miss_probability(1e-4, 1_000);
        assert!((miss - 0.9048).abs() < 0.0002, "actual miss={miss}");
    }

    #[test]
    fn one_in_a_million_activation_is_almost_always_missed_at_1000_trials() {
        let miss = miss_probability(1e-6, 1_000);
        assert!((miss - 0.9990).abs() < 0.0001, "actual miss={miss}");
    }

    #[test]
    fn at_least_once_probability_uses_stable_small_probability_arithmetic() {
        let value = activation_probability_at_least_once(1e-16, 1_000);
        assert!((value - 1e-13).abs() < 1e-26, "actual activation={value}");
    }

    #[test]
    fn computes_minimum_trials_for_five_percent_miss_target() {
        let trials = minimum_trials(1e-3, 0.05).unwrap();
        assert_eq!(trials, 2_995);
        assert!(miss_probability(1e-3, trials) <= 0.05);
        assert!(miss_probability(1e-3, trials - 1) > 0.05);
    }

    #[test]
    fn activation_probability_zero_has_no_finite_sampling_budget() {
        assert_eq!(minimum_trials(0.0, 0.05), None);
        assert_eq!(miss_probability(0.0, 1_000), 1.0);
    }

    #[test]
    fn certain_activation_requires_one_trial() {
        assert_eq!(minimum_trials(1.0, 0.05), Some(1));
        assert_eq!(miss_probability(1.0, 1), 0.0);
    }

    #[test]
    fn rejects_nan_and_invalid_probabilities() {
        assert!(parse_probability("p", "NaN", true).is_err());
        assert!(parse_probability("p", "-0.1", true).is_err());
        assert!(parse_probability("alpha", "1.0", false).is_err());
    }

    #[test]
    fn reports_unrepresentable_trial_budget_instead_of_wrapping() {
        assert_eq!(minimum_trials(1e-300, 0.05), None);
    }
}
