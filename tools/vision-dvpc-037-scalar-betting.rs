//! VIS-DVPC-037: exact-path comparison of OPE threshold decision rules.
//!
//! Compares repeated fixed-time Wald intervals, an alpha-spending Hoeffding
//! confidence sequence, and a one-sided scalar-betting e-process at a fixed
//! threshold. This is NOT the full two-sided MOPE/OFCS confidence sequence.
//!
//! The scalar-betting rule is based on the predictable betting strategy in
//! Section 5.2 of Karampatziakis, Mineiro & Ramdas (ICML 2021): for
//! xi_t = w_t r_t - threshold, set lambda_t = max(0, S_{t-1}/(S_{t-1}+Q_{t-1}))
//! when the denominator is positive, where S and Q are the cumulative sum of
//! xi and xi^2. The e-process multiplies (1 + lambda_t * xi_t). Under the null
//! V(pi) <= threshold, the increments have non-positive conditional mean;
//! predictable bets in [0,1) keep the product a nonnegative supermartingale.
//!
//! The frozen DGP is IID, two-action, fixed-logger, known-propensity, bounded
//! Bernoulli reward. Every one of 4^8 ordered action/reward paths is enumerated.
//! No Monte Carlo or claim to production/SOTA qualification is intended.
#![deny(unsafe_code)]

use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;
use std::fs;

const FIXTURE: &str = include_str!("../tests/fixtures/vision-dvpc-037-scalar.tsv");
const HEADER: &str = "scenario_id\thorizon\ttarget_probability_a\tlogging_probability_a\treward_mean_a\treward_mean_b\tthreshold";
const HORIZON: usize = 8;
const PATHS: usize = 65_536;
const ALPHA: f64 = 0.05;
const Z_975: f64 = 1.959_963_984_540_054;
const EPS: f64 = 1e-10;
const MASS_TOLERANCE: f64 = 1e-8;

#[derive(Clone, Debug)]
struct Scenario {
    id: String,
    horizon: usize,
    target_a: f64,
    logging_a: f64,
    reward_a: f64,
    reward_b: f64,
    threshold: f64,
}

#[derive(Clone, Debug)]
struct IntervalLook {
    coverage: bool,
    width: f64,
    crosses_threshold: bool,
}

#[derive(Clone, Debug)]
struct BetLook {
    rejects: bool,
    wealth: f64,
    log_wealth: f64,
}

#[derive(Clone, Debug)]
struct IntervalAccumulator {
    pointwise_coverage: Vec<f64>,
    mean_width: Vec<f64>,
    pointwise_crossing: Vec<f64>,
    first_crossing: Vec<f64>,
    ever_crossing: Vec<f64>,
    survival: Vec<f64>,
    sequence_coverage: Vec<f64>,
}

impl IntervalAccumulator {
    fn new(horizon: usize) -> Self {
        let n = horizon + 1;
        Self {
            pointwise_coverage: vec![0.0; n],
            mean_width: vec![0.0; n],
            pointwise_crossing: vec![0.0; n],
            first_crossing: vec![0.0; n],
            ever_crossing: vec![0.0; n],
            survival: vec![0.0; n],
            sequence_coverage: vec![0.0; n],
        }
    }
}

#[derive(Clone, Debug)]
struct BetAccumulator {
    pointwise_rejection: Vec<f64>,
    first_rejection: Vec<f64>,
    ever_rejection: Vec<f64>,
    survival: Vec<f64>,
    mean_wealth: Vec<f64>,
    mean_log_wealth: Vec<f64>,
}

impl BetAccumulator {
    fn new(horizon: usize) -> Self {
        let n = horizon + 1;
        Self {
            pointwise_rejection: vec![0.0; n],
            first_rejection: vec![0.0; n],
            ever_rejection: vec![0.0; n],
            survival: vec![0.0; n],
            mean_wealth: vec![0.0; n],
            mean_log_wealth: vec![0.0; n],
        }
    }
}

#[derive(Clone, Debug)]
struct Evaluation {
    scenario: Scenario,
    truth: f64,
    path_count: usize,
    total_probability_mass: f64,
    probability_mass_by_look: Vec<f64>,
    wald: IntervalAccumulator,
    hoeffding: IntervalAccumulator,
    betting: BetAccumulator,
    wald_sequence_coverage: f64,
    hoeffding_sequence_coverage: f64,
    wald_ever_crossing: f64,
    hoeffding_ever_crossing: f64,
    betting_ever_rejection: f64,
    conditional_wald_stop: Option<f64>,
    conditional_hoeffding_stop: Option<f64>,
    conditional_betting_stop: Option<f64>,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn parse_fixture(input: &str) -> Result<Vec<Scenario>, String> {
    let mut lines = input.lines();
    if lines.next() != Some(HEADER) {
        return Err(invalid("unexpected scalar-betting fixture header"));
    }
    let mut scenarios = Vec::new();
    let mut ids = BTreeSet::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let line_number = line_index + 2;
        if fields.len() != 7 {
            return Err(invalid(format!("line {line_number}: expected seven fields")));
        }
        if fields[0].is_empty() || !ids.insert(fields[0].to_owned()) {
            return Err(invalid(format!("line {line_number}: empty or duplicate scenario_id")));
        }
        let parse = |index: usize, name: &str| -> Result<f64, String> {
            let value = fields[index].parse::<f64>()
                .map_err(|_| invalid(format!("line {line_number}: invalid {name}")))?;
            if !value.is_finite() {
                return Err(invalid(format!("line {line_number}: {name} must be finite")));
            }
            Ok(value)
        };
        let horizon = fields[1].parse::<usize>()
            .map_err(|_| invalid(format!("line {line_number}: invalid horizon")))?;
        if horizon != HORIZON {
            return Err(invalid(format!("line {line_number}: horizon must be frozen at {HORIZON}")));
        }
        let target_a = parse(2, "target_probability_a")?;
        let logging_a = parse(3, "logging_probability_a")?;
        let reward_a = parse(4, "reward_mean_a")?;
        let reward_b = parse(5, "reward_mean_b")?;
        let threshold = parse(6, "threshold")?;
        if !(0.0..1.0).contains(&target_a) || !(0.0..1.0).contains(&logging_a) {
            return Err(invalid(format!("line {line_number}: policy probabilities must be in (0,1)")));
        }
        for (name, value) in [("reward_mean_a", reward_a), ("reward_mean_b", reward_b), ("threshold", threshold)] {
            if !(0.0..=1.0).contains(&value) {
                return Err(invalid(format!("line {line_number}: {name} must be in [0,1]")));
            }
        }
        scenarios.push(Scenario {
            id: fields[0].to_owned(),
            horizon,
            target_a,
            logging_a,
            reward_a,
            reward_b,
            threshold,
        });
    }
    if scenarios.is_empty() {
        return Err(invalid("scalar-betting fixture is empty"));
    }
    Ok(scenarios)
}

fn target_value(s: &Scenario) -> f64 {
    s.target_a * s.reward_a + (1.0 - s.target_a) * s.reward_b
}

fn clipped_interval(center: f64, radius: f64) -> (f64, f64) {
    ((center - radius).clamp(0.0, 1.0), (center + radius).clamp(0.0, 1.0))
}

fn covers(lower: f64, upper: f64, truth: f64) -> bool {
    lower <= upper && lower - EPS <= truth && truth <= upper + EPS
}

fn event_probabilities(s: &Scenario) -> [f64; 4] {
    [
        s.logging_a * (1.0 - s.reward_a),       // action A, reward 0
        s.logging_a * s.reward_a,               // action A, reward 1
        (1.0 - s.logging_a) * (1.0 - s.reward_b), // action B, reward 0
        (1.0 - s.logging_a) * s.reward_b,        // action B, reward 1
    ]
}

fn conditional_mean(first: &[f64], event_mass: f64) -> Option<f64> {
    if event_mass <= EPS {
        None
    } else {
        Some(first.iter().enumerate().map(|(t, mass)| t as f64 * mass).sum::<f64>() / event_mass)
    }
}

fn evaluate(s: &Scenario) -> Result<Evaluation, String> {
    let truth = target_value(s);
    let target_b = 1.0 - s.target_a;
    let weight_a = s.target_a / s.logging_a;
    let weight_b = target_b / (1.0 - s.logging_a);
    let max_weight = weight_a.max(weight_b);
    let probabilities = event_probabilities(s);
    if probabilities.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        || (probabilities.iter().sum::<f64>() - 1.0).abs() > MASS_TOLERANCE
    {
        return Err(invalid(format!("{}: invalid outcome probabilities", s.id)));
    }

    let mut mass_by_look = vec![0.0_f64; s.horizon + 1];
    let mut wald = IntervalAccumulator::new(s.horizon);
    let mut hoeffding = IntervalAccumulator::new(s.horizon);
    let mut betting = BetAccumulator::new(s.horizon);
    let mut total_probability_mass = 0.0;
    let ln_reject = (1.0 / ALPHA).ln();

    for path in 0..PATHS {
        let mut code = path;
        let mut path_probability = 1.0;
        let mut wald_looks: Vec<Option<IntervalLook>> = vec![None; s.horizon + 1];
        let mut hoeffding_looks: Vec<Option<IntervalLook>> = vec![None; s.horizon + 1];
        let mut bet_looks: Vec<Option<BetLook>> = vec![None; s.horizon + 1];
        let mut mean = 0.0;
        let mut m2 = 0.0;
        let mut bet_sum = 0.0;
        let mut bet_sum_squares = 0.0;
        let mut wealth = 1.0;
        let mut log_wealth = 0.0;

        for look in 1..=s.horizon {
            let event = code % 4;
            code /= 4;
            path_probability *= probabilities[event];
            let sample = match event {
                0 | 2 => 0.0,
                1 => weight_a,
                3 => weight_b,
                _ => unreachable!(),
            };

            let delta = sample - mean;
            mean += delta / look as f64;
            m2 += delta * (sample - mean);

            let alpha_t = ALPHA / (look as f64 * (look as f64 + 1.0));
            let hoeffding_radius = max_weight * ((2.0 / alpha_t).ln() / (2.0 * look as f64)).sqrt();
            let (cs_lower, cs_upper) = clipped_interval(mean, hoeffding_radius);
            hoeffding_looks[look] = Some(IntervalLook {
                coverage: covers(cs_lower, cs_upper, truth),
                width: (cs_upper - cs_lower).max(0.0),
                crosses_threshold: cs_lower > s.threshold,
            });

            if look >= 2 {
                let sample_variance = (m2 / (look - 1) as f64).max(0.0);
                let radius = Z_975 * (sample_variance / look as f64).sqrt();
                let (lower, upper) = clipped_interval(mean, radius);
                wald_looks[look] = Some(IntervalLook {
                    coverage: covers(lower, upper, truth),
                    width: (upper - lower).max(0.0),
                    crosses_threshold: lower > s.threshold,
                });
            }

            let xi = sample - s.threshold;
            let denominator = bet_sum + bet_sum_squares;
            let lambda = if bet_sum > 0.0 && denominator > 0.0 {
                (bet_sum / denominator).min(1.0 - f64::EPSILON)
            } else {
                0.0
            };
            let factor = 1.0 + lambda * xi;
            if !lambda.is_finite() || !(0.0..1.0).contains(&lambda) || !factor.is_finite() || factor <= 0.0 {
                return Err(invalid(format!("{}: invalid betting factor on path {path} look {look}", s.id)));
            }
            wealth *= factor;
            log_wealth += factor.ln();
            if !wealth.is_finite() || !log_wealth.is_finite() {
                return Err(invalid(format!("{}: non-finite betting wealth on path {path} look {look}", s.id)));
            }
            bet_looks[look] = Some(BetLook {
                rejects: log_wealth > ln_reject,
                wealth,
                log_wealth,
            });
            bet_sum += xi;
            bet_sum_squares += xi * xi;
        }

        total_probability_mass += path_probability;
        for look in 1..=s.horizon {
            mass_by_look[look] += path_probability;

            let cs = hoeffding_looks[look].as_ref().unwrap();
            hoeffding.pointwise_coverage[look] += path_probability * if cs.coverage { 1.0 } else { 0.0 };
            hoeffding.mean_width[look] += path_probability * cs.width;
            hoeffding.pointwise_crossing[look] += path_probability * if cs.crosses_threshold { 1.0 } else { 0.0 };

            let bet = bet_looks[look].as_ref().unwrap();
            betting.pointwise_rejection[look] += path_probability * if bet.rejects { 1.0 } else { 0.0 };
            betting.mean_wealth[look] += path_probability * bet.wealth;
            betting.mean_log_wealth[look] += path_probability * bet.log_wealth;

            if look >= 2 {
                let w = wald_looks[look].as_ref().unwrap();
                wald.pointwise_coverage[look] += path_probability * if w.coverage { 1.0 } else { 0.0 };
                wald.mean_width[look] += path_probability * w.width;
                wald.pointwise_crossing[look] += path_probability * if w.crosses_threshold { 1.0 } else { 0.0 };
            }
        }

        let mut wald_ever = false;
        let mut wald_all_covered = true;
        let mut hoeffding_ever = false;
        let mut hoeffding_all_covered = true;
        let mut betting_ever = false;

        for look in 1..=s.horizon {
            let cs = hoeffding_looks[look].as_ref().unwrap();
            if !cs.coverage {
                hoeffding_all_covered = false;
            }
            if !hoeffding_ever && cs.crosses_threshold {
                hoeffding_ever = true;
                hoeffding.first_crossing[look] += path_probability;
            }
            if hoeffding_ever {
                hoeffding.ever_crossing[look] += path_probability;
            } else {
                hoeffding.survival[look] += path_probability;
            }
            if hoeffding_all_covered {
                hoeffding.sequence_coverage[look] += path_probability;
            }

            if look >= 2 {
                let w = wald_looks[look].as_ref().unwrap();
                if !w.coverage {
                    wald_all_covered = false;
                }
                if !wald_ever && w.crosses_threshold {
                    wald_ever = true;
                    wald.first_crossing[look] += path_probability;
                }
                if wald_ever {
                    wald.ever_crossing[look] += path_probability;
                } else {
                    wald.survival[look] += path_probability;
                }
                if wald_all_covered {
                    wald.sequence_coverage[look] += path_probability;
                }
            }

            let bet = bet_looks[look].as_ref().unwrap();
            if !betting_ever && bet.rejects {
                betting_ever = true;
                betting.first_rejection[look] += path_probability;
            }
            if betting_ever {
                betting.ever_rejection[look] += path_probability;
            } else {
                betting.survival[look] += path_probability;
            }
        }
    }

    if (total_probability_mass - 1.0).abs() > MASS_TOLERANCE {
        return Err(invalid(format!("{}: enumerated path probability mass {total_probability_mass:.12} != 1", s.id)));
    }
    for look in 1..=s.horizon {
        if (mass_by_look[look] - 1.0).abs() > MASS_TOLERANCE {
            return Err(invalid(format!("{} look {look}: probability mass {:.12} != 1", s.id, mass_by_look[look])));
        }
    }

    let final_wald = wald.ever_crossing[s.horizon];
    let final_hoeffding = hoeffding.ever_crossing[s.horizon];
    let final_betting = betting.ever_rejection[s.horizon];
    Ok(Evaluation {
        scenario: s.clone(),
        truth,
        path_count: PATHS,
        total_probability_mass,
        probability_mass_by_look: mass_by_look,
        wald_sequence_coverage: wald.sequence_coverage[s.horizon],
        hoeffding_sequence_coverage: hoeffding.sequence_coverage[s.horizon],
        wald_ever_crossing: final_wald,
        hoeffding_ever_crossing: final_hoeffding,
        betting_ever_rejection: final_betting,
        conditional_wald_stop: conditional_mean(&wald.first_crossing, final_wald),
        conditional_hoeffding_stop: conditional_mean(&hoeffding.first_crossing, final_hoeffding),
        conditional_betting_stop: conditional_mean(&betting.first_rejection, final_betting),
        wald,
        hoeffding,
        betting,
    })
}

fn write_record(
    out: &mut String,
    record_type: &str,
    scenario_id: &str,
    look: usize,
    method: &str,
    verdict: &str,
    metric: &str,
    value: &str,
) {
    let _ = writeln!(out, "{record_type}\t{scenario_id}\t{look}\t{method}\t{verdict}\t{metric}\t{value}");
}

fn write_number(
    out: &mut String,
    record_type: &str,
    scenario_id: &str,
    look: usize,
    method: &str,
    verdict: &str,
    metric: &str,
    value: f64,
) {
    write_record(out, record_type, scenario_id, look, method, verdict, metric, &format!("{value:.12}"));
}

fn sweep(input: &str) -> Result<String, String> {
    let scenarios = parse_fixture(input)?;
    let mut out = String::from("record_type\tscenario_id\tlook\tmethod\tverdict\tmetric\tvalue\n");
    for scenario in scenarios {
        let result = evaluate(&scenario)?;
        let interpretation = if result.truth <= scenario.threshold + EPS {
            "FALSE_PROMOTION_NULL"
        } else {
            "DECISION_POWER_ALTERNATIVE"
        };
        write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-", interpretation,
            "truth", result.truth);
        write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-", interpretation,
            "threshold", scenario.threshold);
        write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-", interpretation,
            "final_probability_mass", result.total_probability_mass);
        write_record(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-", interpretation,
            "enumerated_paths", &result.path_count.to_string());
        write_record(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-", interpretation,
            "path_enumeration", "EXACT_4_POW_8");

        for look in 1..=scenario.horizon {
            write_number(&mut out, "DP_LOOK", &scenario.id, look, "-", interpretation,
                "probability_mass", result.probability_mass_by_look[look]);
            write_record(&mut out, "DP_LOOK", &scenario.id, look, "-", interpretation,
                "path_count", &result.path_count.to_string());

            let cs = &result.hoeffding;
            for (metric, value) in [
                ("pointwise_coverage", cs.pointwise_coverage[look]),
                ("mean_interval_width", cs.mean_width[look]),
                ("pointwise_threshold_crossing_probability", cs.pointwise_crossing[look]),
                ("first_threshold_crossing_probability", cs.first_crossing[look]),
                ("ever_threshold_crossing_probability", cs.ever_crossing[look]),
                ("survival_probability", cs.survival[look]),
                ("sequence_coverage_to_look", cs.sequence_coverage[look]),
            ] {
                write_number(&mut out, "LOOK_METRIC", &scenario.id, look, "ALPHA_SPENDING_HOEFFDING_CS",
                    interpretation, metric, value);
            }

            if look >= 2 {
                let w = &result.wald;
                for (metric, value) in [
                    ("pointwise_coverage", w.pointwise_coverage[look]),
                    ("mean_interval_width", w.mean_width[look]),
                    ("pointwise_threshold_crossing_probability", w.pointwise_crossing[look]),
                    ("first_threshold_crossing_probability", w.first_crossing[look]),
                    ("ever_threshold_crossing_probability", w.ever_crossing[look]),
                    ("survival_probability", w.survival[look]),
                    ("sequence_coverage_to_look", w.sequence_coverage[look]),
                ] {
                    write_number(&mut out, "LOOK_METRIC", &scenario.id, look, "WALD_FIXED_TIME",
                        interpretation, metric, value);
                }
            }

            let bet = &result.betting;
            for (metric, value) in [
                ("pointwise_rejection_probability", bet.pointwise_rejection[look]),
                ("first_rejection_probability", bet.first_rejection[look]),
                ("ever_rejection_probability", bet.ever_rejection[look]),
                ("survival_probability", bet.survival[look]),
                ("mean_wealth", bet.mean_wealth[look]),
                ("mean_log_wealth", bet.mean_log_wealth[look]),
            ] {
                write_number(&mut out, "LOOK_METRIC", &scenario.id, look, "SCALAR_BETTING_ONE_SIDED",
                    interpretation, metric, value);
            }
        }

        for (method, metric, value) in [
            ("WALD_FIXED_TIME", "sequence_coverage", result.wald_sequence_coverage),
            ("ALPHA_SPENDING_HOEFFDING_CS", "sequence_coverage", result.hoeffding_sequence_coverage),
            ("WALD_FIXED_TIME", "ever_threshold_crossing_probability", result.wald_ever_crossing),
            ("ALPHA_SPENDING_HOEFFDING_CS", "ever_threshold_crossing_probability", result.hoeffding_ever_crossing),
            ("SCALAR_BETTING_ONE_SIDED", "ever_rejection_probability", result.betting_ever_rejection),
        ] {
            write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, method, interpretation, metric, value);
        }
        for (method, metric, value) in [
            ("WALD_FIXED_TIME", "conditional_mean_first_crossing_look", result.conditional_wald_stop),
            ("ALPHA_SPENDING_HOEFFDING_CS", "conditional_mean_first_crossing_look", result.conditional_hoeffding_stop),
            ("SCALAR_BETTING_ONE_SIDED", "conditional_mean_first_rejection_look", result.conditional_betting_stop),
        ] {
            match value {
                Some(value) => write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, method,
                    interpretation, metric, value),
                None => write_record(&mut out, "SCENARIO", &scenario.id, scenario.horizon, method,
                    interpretation, metric, "NOT_REACHED"),
            }
        }
        write_record(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "SCALAR_BETTING_ONE_SIDED",
            interpretation, "method_boundary", "ONE_SIDED_E_PROCESS_NOT_FULL_MOPE_CI");
    }
    Ok(out)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("--sweep") => sweep(FIXTURE),
        Some("--input") if args.len() == 3 => {
            fs::read_to_string(&args[2]).map_err(|e| format!("failed to read input: {e}"))
                .and_then(|input| sweep(&input))
        }
        _ => Err("usage: vision-dvpc-037-scalar-betting --sweep | --input <fixture.tsv>".to_owned()),
    };
    match result {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_037_ERROR\t{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn by_id(id: &str) -> Scenario {
        parse_fixture(FIXTURE).unwrap().into_iter().find(|s| s.id == id).unwrap()
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!((actual - expected).abs() <= tolerance,
            "actual={actual:.15}, expected={expected:.15}");
    }

    #[test]
    fn fixture_is_valid_unique_and_horizon_frozen() {
        let scenarios = parse_fixture(FIXTURE).unwrap();
        assert_eq!(scenarios.len(), 6);
        assert!(scenarios.iter().any(|s| s.id == "healthy_null"));
        assert!(scenarios.iter().any(|s| s.id == "weak_clear_alt_p099"));
        assert!(parse_fixture(&format!("{HEADER}\ninvalid\t7\t0.5\t0.5\t0.2\t0.8\t0.5\n")).is_err());
    }

    #[test]
    fn exact_enumeration_reproduces_wald_and_hoeffding_controls() {
        let result = evaluate(&by_id("healthy_null")).unwrap();
        close(result.total_probability_mass, 1.0, MASS_TOLERANCE);
        close(result.wald_sequence_coverage, 0.4375, 1e-9);
        close(result.wald_ever_crossing, 0.28125, 1e-9);
        close(result.hoeffding_sequence_coverage, 1.0, 1e-9);
        close(result.hoeffding_ever_crossing, 0.0, 1e-9);
    }

    #[test]
    fn scalar_betting_null_controls_stay_under_alpha() {
        let healthy = evaluate(&by_id("healthy_null")).unwrap();
        close(healthy.betting_ever_rejection, 0.0, 1e-9);
        let weak = evaluate(&by_id("weak_null_p099")).unwrap();
        close(weak.betting_ever_rejection, 0.001975038788, 1e-9);
        assert!(weak.betting_ever_rejection <= ALPHA + EPS);
        close(weak.wald_ever_crossing, 0.039414394874, 1e-9);
        assert!(weak.hoeffding_ever_crossing < 1e-8);
        for look in 1..=HORIZON {
            close(healthy.betting.mean_wealth[look], 1.0, 1e-8);
            close(weak.betting.mean_wealth[look], 1.0, 1e-8);
        }
    }

    #[test]
    fn betting_rule_has_more_power_than_hoeffding_on_frozen_weak_alternative() {
        let result = evaluate(&by_id("weak_small_alt_p099")).unwrap();
        close(result.betting_ever_rejection, 0.003379111912, 1e-9);
        assert!(result.betting_ever_rejection > result.hoeffding_ever_crossing);
    }

    #[test]
    fn every_look_probability_mass_is_one_for_every_scenario() {
        for scenario in parse_fixture(FIXTURE).unwrap() {
            let result = evaluate(&scenario).unwrap();
            for look in 1..=scenario.horizon {
                close(result.probability_mass_by_look[look], 1.0, MASS_TOLERANCE);
            }
        }
    }

    #[test]
    fn scalar_bet_is_not_mislabeled_as_an_interval() {
        let output = sweep(FIXTURE).unwrap();
        assert!(output.contains("ONE_SIDED_E_PROCESS_NOT_FULL_MOPE_CI"));
        assert!(!output.contains("SCALAR_BETTING_ONE_SIDED\tFALSE_PROMOTION_NULL\tmean_interval_width"));
    }
}
