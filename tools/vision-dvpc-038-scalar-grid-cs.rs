//! VIS-DVPC-038: exact-path audit of a hedged scalar-betting candidate-value set.
//!
//! Implements the scalar-betting component described in the Scalar Betting section of
//! Karampatziakis, Mineiro & Ramdas (ICML 2021), but uses a frozen 0.05 value grid.
//! This is NOT the full two-dimensional, no-grid MOPE algorithm and does not claim
//! continuous-parameter confidence-sequence coverage.
//!
//! For each candidate v, predictable bets are selected from prior statistics for
//! xi+ = w*r - v and xi- = w*(1-r) - (1-v). Candidate v is rejected permanently
//! once 0.5*(K+ + K-) exceeds 1/alpha. The entire 4^8 path distribution is enumerated.
#![deny(unsafe_code)]

use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;
use std::fs;

const FIXTURE: &str = include_str!("../tests/fixtures/vision-dvpc-038-scalar-grid.tsv");
const HEADER: &str = "scenario_id\thorizon\ttarget_probability_a\tlogging_probability_a\treward_mean_a\treward_mean_b";
const HORIZON: usize = 8;
const PATHS: usize = 65_536;
const GRID_STEP: f64 = 0.05;
const GRID_SIZE: usize = 21;
const ALPHA: f64 = 0.05;
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
}

#[derive(Clone, Debug)]
struct Evaluation {
    scenario: Scenario,
    truth: f64,
    truth_grid_index: usize,
    total_probability_mass: f64,
    probability_mass_by_look: Vec<f64>,
    pointwise_rejection: Vec<Vec<f64>>,
    first_rejection: Vec<Vec<f64>>,
    ever_rejection: Vec<Vec<f64>>,
    retained_candidate_count: Vec<f64>,
    grid_hull_width: Vec<f64>,
    empty_set_probability: Vec<f64>,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn candidate_value(index: usize) -> f64 {
    index as f64 * GRID_STEP
}

fn parse_fixture(input: &str) -> Result<Vec<Scenario>, String> {
    let mut lines = input.lines();
    if lines.next() != Some(HEADER) {
        return Err(invalid("unexpected scalar-grid fixture header"));
    }
    let mut scenarios = Vec::new();
    let mut ids = BTreeSet::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let line_number = line_index + 2;
        if fields.len() != 6 {
            return Err(invalid(format!("line {line_number}: expected six fields")));
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
        if !(0.0..1.0).contains(&target_a) || !(0.0..1.0).contains(&logging_a) {
            return Err(invalid(format!("line {line_number}: policy probabilities must be in (0,1)")));
        }
        if !(0.0..=1.0).contains(&reward_a) || !(0.0..=1.0).contains(&reward_b) {
            return Err(invalid(format!("line {line_number}: reward means must be in [0,1]")));
        }
        scenarios.push(Scenario {
            id: fields[0].to_owned(),
            horizon,
            target_a,
            logging_a,
            reward_a,
            reward_b,
        });
    }
    if scenarios.is_empty() {
        return Err(invalid("scalar-grid fixture is empty"));
    }
    Ok(scenarios)
}

fn target_value(s: &Scenario) -> f64 {
    s.target_a * s.reward_a + (1.0 - s.target_a) * s.reward_b
}

fn event_probabilities(s: &Scenario) -> [f64; 4] {
    [
        s.logging_a * (1.0 - s.reward_a),       // A, reward 0
        s.logging_a * s.reward_a,               // A, reward 1
        (1.0 - s.logging_a) * (1.0 - s.reward_b), // B, reward 0
        (1.0 - s.logging_a) * s.reward_b,       // B, reward 1
    ]
}

fn predictable_scalar_bet(sum: f64, sum_squares: f64) -> f64 {
    let denominator = sum + sum_squares;
    if sum > 0.0 && denominator > 0.0 {
        (sum / denominator).clamp(0.0, 1.0 - f64::EPSILON)
    } else {
        0.0
    }
}

fn evaluate(s: &Scenario) -> Result<Evaluation, String> {
    let truth = target_value(s);
    let raw_grid_index = (truth / GRID_STEP).round();
    let truth_grid_index = raw_grid_index as usize;
    if truth_grid_index >= GRID_SIZE || (candidate_value(truth_grid_index) - truth).abs() > EPS {
        return Err(invalid(format!("{}: true value {truth:.12} is not on the frozen 0.05 grid", s.id)));
    }
    let target_b = 1.0 - s.target_a;
    let weight_a = s.target_a / s.logging_a;
    let weight_b = target_b / (1.0 - s.logging_a);
    let event_probs = event_probabilities(s);
    if event_probs.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0)
        || (event_probs.iter().sum::<f64>() - 1.0).abs() > MASS_TOLERANCE
    {
        return Err(invalid(format!("{}: invalid event probabilities", s.id)));
    }

    let mut probability_mass_by_look = vec![0.0_f64; s.horizon + 1];
    let mut pointwise_rejection = vec![vec![0.0_f64; GRID_SIZE]; s.horizon + 1];
    let mut first_rejection = vec![vec![0.0_f64; GRID_SIZE]; s.horizon + 1];
    let mut ever_rejection = vec![vec![0.0_f64; GRID_SIZE]; s.horizon + 1];
    let mut retained_candidate_count = vec![0.0_f64; s.horizon + 1];
    let mut grid_hull_width = vec![0.0_f64; s.horizon + 1];
    let mut empty_set_probability = vec![0.0_f64; s.horizon + 1];
    let mut total_probability_mass = 0.0;
    let rejection_boundary = 1.0 / ALPHA;

    for path in 0..PATHS {
        let mut code = path;
        let mut samples = [0.0_f64; HORIZON];
        let mut complements = [0.0_f64; HORIZON];
        let mut path_probability = 1.0;
        for look in 0..s.horizon {
            let event = (code % 4) as u8;
            code /= 4;
            path_probability *= event_probs[event as usize];
            match event {
                0 => { samples[look] = 0.0; complements[look] = weight_a; }
                1 => { samples[look] = weight_a; complements[look] = 0.0; }
                2 => { samples[look] = 0.0; complements[look] = weight_b; }
                3 => { samples[look] = weight_b; complements[look] = 0.0; }
                _ => unreachable!(),
            }
        }
        if !path_probability.is_finite() || path_probability < 0.0 {
            return Err(invalid(format!("{}: invalid probability for path {path}", s.id)));
        }
        total_probability_mass += path_probability;

        let mut sum_plus = [0.0_f64; GRID_SIZE];
        let mut squares_plus = [0.0_f64; GRID_SIZE];
        let mut sum_minus = [0.0_f64; GRID_SIZE];
        let mut squares_minus = [0.0_f64; GRID_SIZE];
        let mut wealth_plus = [1.0_f64; GRID_SIZE];
        let mut wealth_minus = [1.0_f64; GRID_SIZE];
        let mut ever_rejected = [false; GRID_SIZE];

        for look in 1..=s.horizon {
            let sample = samples[look - 1];
            let complement = complements[look - 1];
            probability_mass_by_look[look] += path_probability;

            for candidate in 0..GRID_SIZE {
                let value = candidate_value(candidate);
                let xi_plus = sample - value;
                let xi_minus = complement - (1.0 - value);
                let lambda_plus = predictable_scalar_bet(sum_plus[candidate], squares_plus[candidate]);
                let lambda_minus = predictable_scalar_bet(sum_minus[candidate], squares_minus[candidate]);
                let factor_plus = 1.0 + lambda_plus * xi_plus;
                let factor_minus = 1.0 + lambda_minus * xi_minus;
                if !factor_plus.is_finite() || factor_plus <= 0.0
                    || !factor_minus.is_finite() || factor_minus <= 0.0
                {
                    return Err(invalid(format!("{}: invalid wealth factor path={path} look={look} candidate={candidate}", s.id)));
                }
                wealth_plus[candidate] *= factor_plus;
                wealth_minus[candidate] *= factor_minus;
                let hedged_wealth = 0.5 * (wealth_plus[candidate] + wealth_minus[candidate]);
                let rejects_now = hedged_wealth > rejection_boundary;
                if rejects_now {
                    pointwise_rejection[look][candidate] += path_probability;
                    if !ever_rejected[candidate] {
                        first_rejection[look][candidate] += path_probability;
                        ever_rejected[candidate] = true;
                    }
                }
                if ever_rejected[candidate] {
                    ever_rejection[look][candidate] += path_probability;
                }
                sum_plus[candidate] += xi_plus;
                squares_plus[candidate] += xi_plus * xi_plus;
                sum_minus[candidate] += xi_minus;
                squares_minus[candidate] += xi_minus * xi_minus;
            }

            let mut accepted = 0usize;
            let mut lo = None;
            let mut hi = None;
            for candidate in 0..GRID_SIZE {
                if !ever_rejected[candidate] {
                    accepted += 1;
                    let value = candidate_value(candidate);
                    lo.get_or_insert(value);
                    hi = Some(value);
                }
            }
            retained_candidate_count[look] += path_probability * accepted as f64;
            match (lo, hi) {
                (Some(low), Some(high)) => grid_hull_width[look] += path_probability * (high - low),
                _ => empty_set_probability[look] += path_probability,
            }
        }
    }

    if (total_probability_mass - 1.0).abs() > MASS_TOLERANCE {
        return Err(invalid(format!("{}: path mass {total_probability_mass:.12} != 1", s.id)));
    }
    for look in 1..=s.horizon {
        if (probability_mass_by_look[look] - 1.0).abs() > MASS_TOLERANCE {
            return Err(invalid(format!("{} look {look}: probability mass {:.12} != 1", s.id, probability_mass_by_look[look])));
        }
    }

    Ok(Evaluation {
        scenario: s.clone(),
        truth,
        truth_grid_index,
        total_probability_mass,
        probability_mass_by_look,
        pointwise_rejection,
        first_rejection,
        ever_rejection,
        retained_candidate_count,
        grid_hull_width,
        empty_set_probability,
    })
}

fn write_record(out: &mut String, record_type: &str, scenario: &str, look: usize,
    candidate: &str, metric: &str, value: &str) {
    let _ = writeln!(out, "{record_type}\t{scenario}\t{look}\t{candidate}\t{metric}\t{value}");
}

fn write_number(out: &mut String, record_type: &str, scenario: &str, look: usize,
    candidate: &str, metric: &str, value: f64) {
    write_record(out, record_type, scenario, look, candidate, metric, &format!("{value:.12}"));
}

fn sweep(input: &str) -> Result<String, String> {
    let scenarios = parse_fixture(input)?;
    let mut out = String::from("record_type\tscenario_id\tlook\tcandidate_value\tmetric\tvalue\n");
    for scenario in scenarios {
        let result = evaluate(&scenario)?;
        write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-",
            "true_value", result.truth);
        write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-",
            "total_probability_mass", result.total_probability_mass);
        write_record(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-",
            "enumerated_paths", &PATHS.to_string());
        write_record(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-",
            "true_value_grid_index", &result.truth_grid_index.to_string());
        write_record(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-",
            "coverage_scope", "TRUE_VALUE_ON_FROZEN_GRID_ONLY");

        for look in 1..=scenario.horizon {
            write_number(&mut out, "DP_LOOK", &scenario.id, look, "-",
                "probability_mass", result.probability_mass_by_look[look]);
            write_record(&mut out, "DP_LOOK", &scenario.id, look, "-",
                "path_count", &PATHS.to_string());
            write_number(&mut out, "GRID_SET", &scenario.id, look, "-",
                "expected_retained_candidate_count", result.retained_candidate_count[look]);
            write_number(&mut out, "GRID_SET", &scenario.id, look, "-",
                "expected_convex_hull_width_diagnostic_only", result.grid_hull_width[look]);
            write_number(&mut out, "GRID_SET", &scenario.id, look, "-",
                "empty_set_probability", result.empty_set_probability[look]);
            for candidate in 0..GRID_SIZE {
                let value = format!("{:.2}", candidate_value(candidate));
                for (metric, probability) in [
                    ("pointwise_rejection_probability", result.pointwise_rejection[look][candidate]),
                    ("first_rejection_probability", result.first_rejection[look][candidate]),
                    ("ever_rejection_probability", result.ever_rejection[look][candidate]),
                ] {
                    write_number(&mut out, "GRID_METRIC", &scenario.id, look, &value, metric, probability);
                }
                if candidate == result.truth_grid_index {
                    write_number(&mut out, "TRUE_VALUE", &scenario.id, look, &value,
                        "coverage_probability", 1.0 - result.ever_rejection[look][candidate]);
                }
            }
        }
        let truth_index = result.truth_grid_index;
        write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, &format!("{:.2}", result.truth),
            "true_candidate_ever_rejection_probability", result.ever_rejection[scenario.horizon][truth_index]);
        write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, &format!("{:.2}", result.truth),
            "true_candidate_coverage_probability", 1.0 - result.ever_rejection[scenario.horizon][truth_index]);
        write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-",
            "expected_retained_candidate_count_final", result.retained_candidate_count[scenario.horizon]);
        write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-",
            "convex_hull_width_diagnostic_only_final", result.grid_hull_width[scenario.horizon]);
    }
    Ok(out)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("--sweep") => sweep(FIXTURE),
        Some("--input") if args.len() == 3 => fs::read_to_string(&args[2])
            .map_err(|e| format!("failed to read input: {e}"))
            .and_then(|input| sweep(&input)),
        _ => Err("usage: vision-dvpc-038-scalar-grid-cs --sweep | --input <fixture.tsv>".to_owned()),
    };
    match result {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_038_ERROR\t{error}");
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
    fn fixture_is_valid_unique_and_true_values_are_on_grid() {
        let scenarios = parse_fixture(FIXTURE).unwrap();
        assert_eq!(scenarios.len(), 6);
        for scenario in &scenarios {
            let value = target_value(scenario);
            let grid = candidate_value((value / GRID_STEP).round() as usize);
            close(value, grid, EPS);
        }
    }

    #[test]
    fn exact_path_mass_and_known_weak_overlap_null_rejection() {
        let result = evaluate(&by_id("weak_null_p099")).unwrap();
        close(result.total_probability_mass, 1.0, MASS_TOLERANCE);
        close(result.ever_rejection[HORIZON][result.truth_grid_index],
            0.008101857333562, 1e-9);
        assert!(result.ever_rejection[HORIZON][result.truth_grid_index] <= ALPHA + EPS);
    }

    #[test]
    fn scalar_hedge_reproduces_registered_candidate_test_values() {
        let null = evaluate(&by_id("healthy_null")).unwrap();
        close(null.ever_rejection[HORIZON][null.truth_grid_index], 0.0, 1e-9);
        let alternative = evaluate(&by_id("weak_small_alt_p099")).unwrap();
        close(alternative.ever_rejection[HORIZON][10], 0.007547226594, 1e-9);
        let clear = evaluate(&by_id("weak_clear_alt_p099")).unwrap();
        close(clear.ever_rejection[HORIZON][clear.truth_grid_index],
            0.002021686820, 1e-9);
    }

    #[test]
    fn true_candidate_grid_coverage_stays_at_least_ninety_five_percent() {
        for id in ["healthy_null", "weak_null_p099", "weak_small_alt_p099", "weak_clear_alt_p099"] {
            let result = evaluate(&by_id(id)).unwrap();
            let rejected = result.ever_rejection[HORIZON][result.truth_grid_index];
            assert!(rejected <= ALPHA + EPS, "{id}: true grid candidate false rejection={rejected}");
        }
    }

    #[test]
    fn every_look_mass_is_one_and_grid_count_is_fixed() {
        let result = evaluate(&by_id("healthy_null")).unwrap();
        for look in 1..=HORIZON {
            close(result.probability_mass_by_look[look], 1.0, MASS_TOLERANCE);
        }
        assert_eq!(GRID_SIZE, 21);
        assert_eq!(PATHS, 65_536);
    }

    #[test]
    fn first_rejection_mass_equals_final_ever_rejection_mass() {
        for scenario in parse_fixture(FIXTURE).unwrap() {
            let result = evaluate(&scenario).unwrap();
            for candidate in 0..GRID_SIZE {
                let summed_first = (1..=HORIZON)
                    .map(|look| result.first_rejection[look][candidate])
                    .sum::<f64>();
                close(summed_first, result.ever_rejection[HORIZON][candidate], MASS_TOLERANCE);
            }
        }
    }

    #[test]
    fn output_does_not_claim_continuous_coverage_or_full_mope() {
        let output = sweep(FIXTURE).unwrap();
        assert!(output.contains("TRUE_VALUE_ON_FROZEN_GRID_ONLY"));
        assert!(output.contains("expected_convex_hull_width_diagnostic_only"));
    }
}
