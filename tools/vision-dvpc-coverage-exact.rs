//! VIS-DVPC-034: exact finite-sample confidence-interval coverage for IPS.
//! All scenarios are synthetic iid Bernoulli contextual-bandit DGPs.
//! Interval coverage is a property of the frozen DGP and assumptions, not real-world certification.
#![deny(unsafe_code)]

use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;
use std::fs;

const FIXTURE: &str = include_str!("../tests/fixtures/vision-dvpc-034-coverage.tsv");
const HEADER: &str = "scenario_id\tn\ttarget_probability_a\ttrue_logging_probability_a\testimated_logging_probability_a\treward_mean_a\treward_mean_b";
const EPS: f64 = 1e-10;
const ALPHA: f64 = 0.05;
const Z_975: f64 = 1.959_963_984_540_054;

#[derive(Clone, Debug)]
struct Scenario {
    id: String,
    n: usize,
    target_a: f64,
    true_logging_a: f64,
    estimated_logging_a: f64,
    reward_a: f64,
    reward_b: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Method {
    Wald,
    Hoeffding,
    EmpiricalBernstein,
}

impl Method {
    const ALL: [Self; 3] = [Self::Wald, Self::Hoeffding, Self::EmpiricalBernstein];

    fn name(self) -> &'static str {
        match self {
            Self::Wald => "WALD_95",
            Self::Hoeffding => "HOEFFDING_95",
            Self::EmpiricalBernstein => "EMP_BERNSTEIN_95",
        }
    }
}

#[derive(Clone, Debug)]
struct CoverageResult {
    method: Method,
    verdict: &'static str,
    coverage: f64,
    undercoverage_probability: f64,
    mean_width: f64,
    worst_width: f64,
    state_count: usize,
    probability_mass: f64,
}

#[derive(Clone, Debug)]
struct ScenarioResult {
    scenario: Scenario,
    truth: f64,
    verdict: &'static str,
    propensity_correct: bool,
    enumerated_states: usize,
    probability_mass: f64,
    intervals: Vec<CoverageResult>,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn parse_fixture(input: &str) -> Result<Vec<Scenario>, String> {
    let mut lines = input.lines();
    if lines.next() != Some(HEADER) {
        return Err(invalid("unexpected coverage fixture header"));
    }
    let mut scenarios = Vec::new();
    let mut ids = BTreeSet::new();
    for (line_index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 7 {
            return Err(invalid(format!("line {}: expected seven fields", line_index + 2)));
        }
        if fields[0].is_empty() || !ids.insert(fields[0].to_owned()) {
            return Err(invalid(format!("line {}: empty or duplicate scenario_id", line_index + 2)));
        }
        let parse = |index: usize, name: &str| -> Result<f64, String> {
            let value = fields[index].parse::<f64>()
                .map_err(|_| invalid(format!("line {}: invalid {name}", line_index + 2)))?;
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(invalid(format!("line {}: {name} must be finite and in [0,1]", line_index + 2)));
            }
            Ok(value)
        };
        let n = fields[1].parse::<usize>()
            .map_err(|_| invalid(format!("line {}: invalid n", line_index + 2)))?;
        if !(2..=8).contains(&n) {
            return Err(invalid(format!("line {}: n must be in 2..=8", line_index + 2)));
        }
        scenarios.push(Scenario {
            id: fields[0].to_owned(),
            n,
            target_a: parse(2, "target_probability_a")?,
            true_logging_a: parse(3, "true_logging_probability_a")?,
            estimated_logging_a: parse(4, "estimated_logging_probability_a")?,
            reward_a: parse(5, "reward_mean_a")?,
            reward_b: parse(6, "reward_mean_b")?,
        });
    }
    if scenarios.is_empty() {
        return Err(invalid("coverage fixture is empty"));
    }
    Ok(scenarios)
}

fn truth(s: &Scenario) -> f64 {
    s.target_a * s.reward_a + (1.0 - s.target_a) * s.reward_b
}

fn unsupported_target_mass(s: &Scenario) -> f64 {
    let target_b = 1.0 - s.target_a;
    let logging_b = 1.0 - s.true_logging_a;
    (if s.target_a > EPS && s.true_logging_a <= 0.0 { s.target_a } else { 0.0 })
        + (if target_b > EPS && logging_b <= 0.0 { target_b } else { 0.0 })
}

fn event_probabilities(s: &Scenario) -> [f64; 4] {
    [
        s.true_logging_a * (1.0 - s.reward_a),
        s.true_logging_a * s.reward_a,
        (1.0 - s.true_logging_a) * (1.0 - s.reward_b),
        (1.0 - s.true_logging_a) * s.reward_b,
    ]
}

fn clamp_interval(center: f64, radius: f64) -> (f64, f64) {
    ((center - radius).max(0.0), (center + radius).min(1.0))
}

fn sample_variance(values: &[f64], mean: f64) -> f64 {
    values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

fn radius(method: Method, variance: f64, max_weight: f64, n: usize) -> f64 {
    match method {
        Method::Wald => Z_975 * (variance / n as f64).sqrt(),
        Method::Hoeffding => max_weight * ((2.0 / ALPHA).ln() / (2.0 * n as f64)).sqrt(),
        // Reference formula documented by SCOPE-RL's HCOPE implementation.
        // For iid contributions in [0, max_weight], this is assessed against exact coverage.
        Method::EmpiricalBernstein => {
            let log = (2.0 / ALPHA).ln();
            7.0 * max_weight * log / (3.0 * (n - 1) as f64)
                + (2.0 * variance * log / (n - 1) as f64).sqrt()
        }
    }
}

fn evaluate(s: &Scenario) -> Result<ScenarioResult, String> {
    let target_b = 1.0 - s.target_a;
    let true_logging_b = 1.0 - s.true_logging_a;
    let estimated_logging_b = 1.0 - s.estimated_logging_a;
    let target_truth = truth(s);
    let propensity_correct = (s.true_logging_a - s.estimated_logging_a).abs() <= EPS;

    if unsupported_target_mass(s) > EPS {
        return Ok(ScenarioResult {
            scenario: s.clone(),
            truth: target_truth,
            verdict: "NOT_IDENTIFIED",
            propensity_correct,
            enumerated_states: 0,
            probability_mass: 0.0,
            intervals: Vec::new(),
        });
    }
    if s.target_a > EPS && s.estimated_logging_a <= 0.0 {
        return Err(invalid(format!("{}: estimated propensity is zero for target-supported action A", s.id)));
    }
    if target_b > EPS && estimated_logging_b <= 0.0 {
        return Err(invalid(format!("{}: estimated propensity is zero for target-supported action B", s.id)));
    }

    let weight_a = if s.target_a > EPS { s.target_a / s.estimated_logging_a } else { 0.0 };
    let weight_b = if target_b > EPS { target_b / estimated_logging_b } else { 0.0 };
    let max_weight = weight_a.max(weight_b);
    if !max_weight.is_finite() || max_weight <= 0.0 {
        return Err(invalid(format!("{}: invalid maximum importance weight", s.id)));
    }

    let probabilities = event_probabilities(s);
    let state_count = 4usize.pow(s.n as u32);
    let mut total_mass = 0.0;
    let mut coverage = [0.0_f64; 3];
    let mut weighted_width = [0.0_f64; 3];
    let mut worst_width = [0.0_f64; 3];

    for state in 0..state_count {
        let mut code = state;
        let mut probability = 1.0;
        let mut values = Vec::with_capacity(s.n);
        for _ in 0..s.n {
            let event = code % 4;
            code /= 4;
            probability *= probabilities[event];
            let action_a = event < 2;
            let reward = if event % 2 == 0 { 0.0 } else { 1.0 };
            values.push((if action_a { weight_a } else { weight_b }) * reward);
        }
        if probability <= 0.0 {
            continue;
        }

        let estimate = values.iter().sum::<f64>() / s.n as f64;
        let variance = sample_variance(&values, estimate);
        total_mass += probability;
        for (index, method) in Method::ALL.into_iter().enumerate() {
            let (lower, upper) = clamp_interval(estimate, radius(method, variance, max_weight, s.n));
            if lower <= upper && lower - EPS <= target_truth && target_truth <= upper + EPS {
                coverage[index] += probability;
            }
            // An interval entirely outside the parameter range has an empty intersection,
            // not a point at the nearest boundary.
            let width = (upper - lower).max(0.0);
            weighted_width[index] += probability * width;
            worst_width[index] = worst_width[index].max(width);
        }
    }

    if (total_mass - 1.0).abs() > 1e-8 {
        return Err(invalid(format!("{}: enumerated probability mass {total_mass} != 1", s.id)));
    }

    let assumption_verdict = if propensity_correct {
        "ASSUMPTIONS_MATCH_SYNTHETIC_DGP"
    } else {
        "ASSUMPTION_VIOLATION_DESCRIPTIVE_ONLY"
    };
    let intervals = Method::ALL.into_iter().enumerate().map(|(i, method)| CoverageResult {
        method,
        verdict: assumption_verdict,
        coverage: coverage[i],
        undercoverage_probability: (1.0 - coverage[i]).max(0.0),
        mean_width: weighted_width[i],
        worst_width: worst_width[i],
        state_count,
        probability_mass: total_mass,
    }).collect();

    Ok(ScenarioResult {
        scenario: s.clone(),
        truth: target_truth,
        verdict: "SUPPORT_OK_SYNTHETIC_ONLY",
        propensity_correct,
        enumerated_states: state_count,
        probability_mass: total_mass,
        intervals,
    })
}

fn sweep(input: &str) -> Result<String, String> {
    let scenarios = parse_fixture(input)?;
    let mut out = String::from("record_type\tscenario_id\tmethod\tverdict\tmetric\tvalue\n");
    let mut minimum_valid_coverage = [1.0_f64; 3];
    let mut worst_valid_undercoverage = [0.0_f64; 3];
    let mut max_valid_mean_width = [0.0_f64; 3];
    let mut valid_scenarios = 0usize;

    for scenario in scenarios {
        let result = evaluate(&scenario)?;
        let _ = writeln!(out, "SCENARIO\t{}\t-\t{}\ttarget_truth_evaluator_only\t{:.12}",
            result.scenario.id, result.verdict, result.truth);
        let _ = writeln!(out, "SCENARIO\t{}\t-\t{}\testimated_logging_matches_true\t{}",
            result.scenario.id, result.verdict, result.propensity_correct);
        if result.verdict == "NOT_IDENTIFIED" {
            let _ = writeln!(out, "SCENARIO\t{}\t-\tNOT_IDENTIFIED\tconfidence_intervals\tSUPPRESSED", result.scenario.id);
            continue;
        }
        let _ = writeln!(out, "SCENARIO\t{}\t-\t{}\tenumerated_states\t{}",
            result.scenario.id, result.verdict, result.enumerated_states);
        let _ = writeln!(out, "SCENARIO\t{}\t-\t{}\tprobability_mass\t{:.12}",
            result.scenario.id, result.verdict, result.probability_mass);

        if result.propensity_correct {
            valid_scenarios += 1;
        }
        for (index, interval) in result.intervals.iter().enumerate() {
            for (field, value) in [
                ("nominal_coverage", 1.0 - ALPHA),
                ("exact_coverage", interval.coverage),
                ("exact_undercoverage_probability", interval.undercoverage_probability),
                ("mean_interval_width", interval.mean_width),
                ("worst_interval_width", interval.worst_width),
                ("enumerated_states", interval.state_count as f64),
                ("probability_mass", interval.probability_mass),
            ] {
                let _ = writeln!(out, "METRIC\t{}\t{}\t{}\t{}\t{:.12}",
                    result.scenario.id, interval.method.name(), interval.verdict, field, value);
            }
            if result.propensity_correct {
                minimum_valid_coverage[index] = minimum_valid_coverage[index].min(interval.coverage);
                worst_valid_undercoverage[index] = worst_valid_undercoverage[index].max(interval.undercoverage_probability);
                max_valid_mean_width[index] = max_valid_mean_width[index].max(interval.mean_width);
            }
        }
    }

    for (index, method) in Method::ALL.into_iter().enumerate() {
        let passes = if method == Method::Wald {
            "MEASURE_ONLY"
        } else if valid_scenarios > 0 && minimum_valid_coverage[index] + 1e-9 >= 1.0 - ALPHA {
            "COVERAGE_FLOOR_MET"
        } else {
            "COVERAGE_FLOOR_FAILED"
        };
        let _ = writeln!(out, "AGGREGATE\tALL_CORRECT_PROPENSITY\t{}\t{}\tminimum_exact_coverage\t{:.12}",
            method.name(), passes, minimum_valid_coverage[index]);
        let _ = writeln!(out, "AGGREGATE\tALL_CORRECT_PROPENSITY\t{}\t{}\tworst_undercoverage_probability\t{:.12}",
            method.name(), passes, worst_valid_undercoverage[index]);
        let _ = writeln!(out, "AGGREGATE\tALL_CORRECT_PROPENSITY\t{}\t{}\tworst_scenario_mean_width\t{:.12}",
            method.name(), passes, max_valid_mean_width[index]);
    }
    out.push_str("LIMIT\t-\t-\tALL_SCENARIOS\tconfidence_sequence\tNOT_IMPLEMENTED\n");
    out.push_str("LIMIT\t-\t-\tALL_SCENARIOS\treal_trace_calibration\tNOT_CLAIMED\n");
    out.push_str("LIMIT\t-\t-\tALL_SCENARIOS\tconfidence_intervals_under_zero_support\tSUPPRESSED\n");
    Ok(out)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let result = match args.as_slice() {
        [_program, flag] if flag == "--sweep" => sweep(FIXTURE),
        [_program, flag, path] if flag == "--sweep-file" => {
            fs::read_to_string(path)
                .map_err(|error| invalid(format!("cannot read {path}: {error}")))
                .and_then(|input| sweep(&input))
        }
        _ => Err("usage: vision-dvpc-coverage-exact --sweep | --sweep-file SCENARIOS.tsv".into()),
    };
    match result {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS-DVPC-034 error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(id: &str) -> ScenarioResult {
        let scenario = parse_fixture(FIXTURE).unwrap().into_iter().find(|s| s.id == id).unwrap();
        evaluate(&scenario).unwrap()
    }

    fn metric(result: &ScenarioResult, method: Method) -> &CoverageResult {
        result.intervals.iter().find(|item| item.method == method).unwrap()
    }

    #[test]
    fn fixture_has_correct_weak_misspecified_and_zero_support_controls() {
        let scenarios = parse_fixture(FIXTURE).unwrap();
        assert_eq!(scenarios.len(), 7);
        assert!(scenarios.iter().any(|s| s.true_logging_a == 0.99 && s.estimated_logging_a == 0.99));
        assert!(scenarios.iter().any(|s| (s.true_logging_a - s.estimated_logging_a).abs() > EPS));
        assert!(scenarios.iter().any(|s| unsupported_target_mass(s) > EPS));
    }

    #[test]
    fn exact_distribution_mass_is_one_and_all_states_are_enumerated() {
        for scenario in parse_fixture(FIXTURE).unwrap().into_iter().filter(|s| unsupported_target_mass(s) <= EPS) {
            let result = evaluate(&scenario).unwrap();
            assert_eq!(result.enumerated_states, 4096);
            assert!((result.probability_mass - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn healthy_wald_coverage_known_answer_exposes_small_sample_undercoverage() {
        let r = result("healthy_correct");
        assert!((metric(&r, Method::Wald).coverage - 0.78125).abs() < 1e-8);
        assert!((metric(&r, Method::Hoeffding).coverage - 1.0).abs() < 1e-8);
        assert!((metric(&r, Method::EmpiricalBernstein).coverage - 1.0).abs() < 1e-8);
    }

    #[test]
    fn weak_overlap_wald_undercoverage_is_pinned() {
        let r = result("weak_overlap_001");
        assert!((metric(&r, Method::Wald).coverage - 0.063024).abs() < 1e-6);
        assert!(metric(&r, Method::Hoeffding).coverage >= 0.95);
        assert!(metric(&r, Method::EmpiricalBernstein).coverage >= 0.95);
    }

    #[test]
    fn bounded_interval_coverage_floor_holds_exactly_under_correct_propensities() {
        for scenario in parse_fixture(FIXTURE).unwrap() {
            let result = evaluate(&scenario).unwrap();
            if result.verdict == "NOT_IDENTIFIED" || !result.propensity_correct {
                continue;
            }
            for method in [Method::Hoeffding, Method::EmpiricalBernstein] {
                assert!(metric(&result, method).coverage + 1e-9 >= 0.95,
                    "{} {} coverage={}", scenario.id, method.name(), metric(&result, method).coverage);
            }
        }
    }

    #[test]
    fn propensity_misspecification_is_labelled_descriptive_not_a_valid_guarantee() {
        let result = result("propensity_misspecified");
        assert!(!result.propensity_correct);
        assert!(result.intervals.iter().all(|item| item.verdict == "ASSUMPTION_VIOLATION_DESCRIPTIVE_ONLY"));
    }

    #[test]
    fn zero_support_suppresses_all_intervals() {
        let result = result("zero_support");
        assert_eq!(result.verdict, "NOT_IDENTIFIED");
        assert!(result.intervals.is_empty());
        assert_eq!(result.enumerated_states, 0);
    }

    #[test]
    fn reward_probability_out_of_range_and_duplicate_ids_fail_closed() {
        let invalid_reward = FIXTURE.replace("\t0.2\t0.8\n", "\t1.2\t0.8\n");
        assert!(parse_fixture(&invalid_reward).is_err());
        let duplicate = format!("{FIXTURE}healthy_correct\t6\t0.5\t0.5\t0.5\t0.2\t0.8\n");
        assert!(parse_fixture(&duplicate).is_err());
    }

    #[test]
    fn interval_width_stays_inside_the_reward_range() {
        for scenario in parse_fixture(FIXTURE).unwrap().into_iter().filter(|s| unsupported_target_mass(s) <= EPS) {
            let result = evaluate(&scenario).unwrap();
            for interval in &result.intervals {
                assert!((0.0..=1.0).contains(&interval.mean_width));
                assert!((0.0..=1.0).contains(&interval.worst_width));
            }
        }
    }
}
