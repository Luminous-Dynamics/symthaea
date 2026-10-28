//! VIS-DVPC-028: exact finite-sample OPE estimator comparison.
//! A small Bernoulli contextual-bandit laboratory; all results are synthetic.
#![deny(unsafe_code)]

use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;

const GRID: &str = include_str!("../tests/fixtures/vision-dvpc-028-scenarios.tsv");
const SCENARIO_HEADER: &str = "scenario_id\tn\ttarget_probability_a\ttrue_logging_probability_a\testimated_logging_probability_a\treward_mean_a\treward_mean_b\tqhat_a\tqhat_b\tclip_cap";
const EPS: f64 = 1e-10;

#[derive(Clone, Debug)]
struct Scenario {
    id: String,
    n: usize,
    target_a: f64,
    logging_a: f64,
    estimated_logging_a: f64,
    reward_a: f64,
    reward_b: f64,
    qhat_a: f64,
    qhat_b: f64,
    clip_cap: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Estimator {
    Direct,
    Ips,
    Snips,
    DoublyRobust,
    ClippedIps,
    Switch,
}

impl Estimator {
    const ALL: [Self; 6] = [
        Self::Direct,
        Self::Ips,
        Self::Snips,
        Self::DoublyRobust,
        Self::ClippedIps,
        Self::Switch,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Direct => "DM",
            Self::Ips => "IPS",
            Self::Snips => "SNIPS",
            Self::DoublyRobust => "DR",
            Self::ClippedIps => "CLIPPED_IPS",
            Self::Switch => "SWITCH",
        }
    }
}

#[derive(Clone, Debug)]
struct Metrics {
    estimator: Estimator,
    expected: f64,
    bias: f64,
    variance: f64,
    rmse: f64,
    max_abs_error: f64,
}

#[derive(Clone, Debug)]
struct SweepResult {
    scenario: Scenario,
    target_value: f64,
    verdict: &'static str,
    unsupported_target_mass: f64,
    supported_target_contribution: f64,
    identified_lower: f64,
    identified_upper: f64,
    enumerated_states: usize,
    probability_mass: f64,
    metrics: Vec<Metrics>,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn parse_grid(input: &str) -> Result<Vec<Scenario>, String> {
    let mut lines = input.lines();
    if lines.next() != Some(SCENARIO_HEADER) {
        return Err(invalid("unexpected scenario header"));
    }
    let mut result = Vec::new();
    let mut ids = BTreeSet::new();
    for (index, line) in lines.enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 10 {
            return Err(invalid(format!("line {}: expected ten fields", index + 2)));
        }
        if fields[0].is_empty() || !ids.insert(fields[0].to_owned()) {
            return Err(invalid(format!("line {}: empty or duplicate scenario_id", index + 2)));
        }
        let parse = |i: usize, name: &str| -> Result<f64, String> {
            let value = fields[i].parse::<f64>()
                .map_err(|_| invalid(format!("line {}: invalid {name}", index + 2)))?;
            if !value.is_finite() {
                return Err(invalid(format!("line {}: {name} must be finite", index + 2)));
            }
            Ok(value)
        };
        let n = fields[1].parse::<usize>()
            .map_err(|_| invalid(format!("line {}: invalid n", index + 2)))?;
        if !(1..=8).contains(&n) {
            return Err(invalid(format!("line {}: n must be in 1..=8 for exact enumeration", index + 2)));
        }
        let s = Scenario {
            id: fields[0].to_owned(),
            n,
            target_a: parse(2, "target_probability_a")?,
            logging_a: parse(3, "true_logging_probability_a")?,
            estimated_logging_a: parse(4, "estimated_logging_probability_a")?,
            reward_a: parse(5, "reward_mean_a")?,
            reward_b: parse(6, "reward_mean_b")?,
            qhat_a: parse(7, "qhat_a")?,
            qhat_b: parse(8, "qhat_b")?,
            clip_cap: parse(9, "clip_cap")?,
        };
        for (name, value) in [
            ("target_probability_a", s.target_a),
            ("true_logging_probability_a", s.logging_a),
            ("estimated_logging_probability_a", s.estimated_logging_a),
            ("reward_mean_a", s.reward_a),
            ("reward_mean_b", s.reward_b),
            ("qhat_a", s.qhat_a),
            ("qhat_b", s.qhat_b),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(invalid(format!("line {}: {name} must be in [0,1]", index + 2)));
            }
        }
        if s.clip_cap <= 0.0 {
            return Err(invalid(format!("line {}: clip_cap must be positive", index + 2)));
        }
        result.push(s);
    }
    if result.is_empty() {
        return Err(invalid("scenario grid is empty"));
    }
    Ok(result)
}

fn target_value(s: &Scenario) -> f64 {
    s.target_a * s.reward_a + (1.0 - s.target_a) * s.reward_b
}

fn unsupported_target_mass(s: &Scenario) -> f64 {
    let target_b = 1.0 - s.target_a;
    let logging_b = 1.0 - s.logging_a;
    let mut mass = 0.0;
    if s.target_a > 0.0 && s.logging_a <= 0.0 {
        mass += s.target_a;
    }
    if target_b > 0.0 && logging_b <= 0.0 {
        mass += target_b;
    }
    mass
}

fn logged_event_probabilities(s: &Scenario) -> [f64; 4] {
    [
        s.logging_a * (1.0 - s.reward_a),
        s.logging_a * s.reward_a,
        (1.0 - s.logging_a) * (1.0 - s.reward_b),
        (1.0 - s.logging_a) * s.reward_b,
    ]
}

fn identification_region(s: &Scenario) -> (f64, f64, f64) {
    let target_b = 1.0 - s.target_a;
    let logging_b = 1.0 - s.logging_a;
    let supported = (if s.logging_a > 0.0 { s.target_a * s.reward_a } else { 0.0 })
        + (if logging_b > 0.0 { target_b * s.reward_b } else { 0.0 });
    let unsupported = unsupported_target_mass(s);
    (supported, supported, (supported + unsupported).min(1.0))
}

fn enumerate(s: &Scenario) -> Result<SweepResult, String> {
    let truth = target_value(s);
    let unsupported_mass = unsupported_target_mass(s);
    let (supported_target_contribution, identified_lower, identified_upper) = identification_region(s);
    if unsupported_mass > EPS {
        return Ok(SweepResult {
            scenario: s.clone(),
            target_value: truth,
            verdict: "NOT_IDENTIFIED",
            unsupported_target_mass: unsupported_mass,
            supported_target_contribution,
            identified_lower,
            identified_upper,
            enumerated_states: 0,
            probability_mass: 0.0,
            metrics: Vec::new(),
        });
    }

    let target_b = 1.0 - s.target_a;
    let logging_b = 1.0 - s.logging_a;
    let estimated_logging_b = 1.0 - s.estimated_logging_a;
    let q_target = s.target_a * s.qhat_a + target_b * s.qhat_b;
    let event_probabilities = logged_event_probabilities(s);
    let state_count = 4usize.pow(s.n as u32);
    let mut distribution: Vec<(f64, [f64; 6])> = Vec::with_capacity(state_count);
    let mut probability_mass = 0.0;

    for state in 0..state_count {
        let mut code = state;
        let mut probability = 1.0;
        let mut weights = [0.0; 8];
        let mut rewards = [0.0; 8];
        let mut q_selected = [0.0; 8];
        for row in 0..s.n {
            let event = code % 4;
            code /= 4;
            probability *= event_probabilities[event];
            let action_a = event < 2;
            let reward = if event % 2 == 0 { 0.0 } else { 1.0 };
            let estimator_logging_p = if action_a { s.estimated_logging_a } else { estimated_logging_b };
            let target_p = if action_a { s.target_a } else { target_b };
            let qhat = if action_a { s.qhat_a } else { s.qhat_b };
            if target_p <= 0.0 {
                weights[row] = 0.0;
            } else {
                if estimator_logging_p <= 0.0 {
                    return Err(invalid("estimated logging propensity is zero for an observed target-supported action"));
                }
                weights[row] = target_p / estimator_logging_p;
            }
            rewards[row] = reward;
            q_selected[row] = qhat;
        }
        if probability <= 0.0 {
            continue;
        }

        let mut ips_sum = 0.0;
        let mut weight_sum = 0.0;
        let mut weighted_reward_sum = 0.0;
        let mut dr_sum = 0.0;
        let mut clipped_ips_sum = 0.0;
        let mut switch_sum = 0.0;
        for row in 0..s.n {
            let weight = weights[row];
            let reward = rewards[row];
            let qhat = q_selected[row];
            ips_sum += weight * reward;
            weight_sum += weight;
            weighted_reward_sum += weight * reward;
            dr_sum += q_target + weight * (reward - qhat);
            clipped_ips_sum += weight.min(s.clip_cap) * reward;
            switch_sum += q_target + if weight <= s.clip_cap {
                weight * (reward - qhat)
            } else {
                0.0
            };
        }
        if weight_sum <= 0.0 {
            return Err(invalid("SNIPS denominator is zero; add an explicit undefined-estimator mass contract"));
        }
        let estimates = [
            q_target,
            ips_sum / s.n as f64,
            weighted_reward_sum / weight_sum,
            dr_sum / s.n as f64,
            clipped_ips_sum / s.n as f64,
            switch_sum / s.n as f64,
        ];
        probability_mass += probability;
        distribution.push((probability, estimates));
    }

    if (probability_mass - 1.0).abs() > 1e-8 {
        return Err(invalid(format!("enumerated probability mass is {probability_mass}, not 1")));
    }

    let mut metrics = Vec::with_capacity(Estimator::ALL.len());
    for (column, estimator) in Estimator::ALL.into_iter().enumerate() {
        let expected = distribution.iter().map(|(p, e)| p * e[column]).sum::<f64>();
        let variance = distribution.iter().map(|(p, e)| p * (e[column] - expected).powi(2)).sum::<f64>();
        let mse = distribution.iter().map(|(p, e)| p * (e[column] - truth).powi(2)).sum::<f64>();
        let max_abs_error = distribution.iter().map(|(_, e)| (e[column] - truth).abs()).fold(0.0, f64::max);
        metrics.push(Metrics {
            estimator,
            expected,
            bias: expected - truth,
            variance,
            rmse: mse.sqrt(),
            max_abs_error,
        });
    }

    Ok(SweepResult {
        scenario: s.clone(),
        target_value: truth,
        verdict: "SUPPORT_OK_SYNTHETIC_ONLY",
        unsupported_target_mass: 0.0,
        supported_target_contribution,
        identified_lower,
        identified_upper,
        enumerated_states: distribution.len(),
        probability_mass,
        metrics,
    })
}

fn run_sweep(input: &str) -> Result<String, String> {
    let scenarios = parse_grid(input)?;
    let mut out = String::from(
        "record_type\tscenario_id\testimator\tverdict\tmetric\tvalue\n",
    );
    for scenario in scenarios {
        let result = enumerate(&scenario)?;
        let _ = writeln!(
            out,
            "SCENARIO\t{}\t-\t{}\toracle_truth_evaluator_only\t{:.12}",
            result.scenario.id, result.verdict, result.target_value
        );
        for (field, value) in [
            ("supported_target_contribution", result.supported_target_contribution),
            ("unsupported_target_mass", result.unsupported_target_mass),
            ("lower_bound", result.identified_lower),
            ("upper_bound", result.identified_upper),
            ("singleton_region", if (result.identified_upper - result.identified_lower).abs() <= EPS { 1.0 } else { 0.0 }),
        ] {
            let _ = writeln!(
                out,
                "IDENTIFICATION_REGION\t{}\t-\t{}\t{}\t{:.12}",
                result.scenario.id,
                if (result.identified_upper - result.identified_lower).abs() <= EPS { "IDENTIFIED_SINGLETON" } else { "PARTIALLY_IDENTIFIED" },
                field,
                value
            );
        }
        if result.verdict == "NOT_IDENTIFIED" {
            let _ = writeln!(
                out,
                "SCENARIO\t{}\t-\tNOT_IDENTIFIED\testimator_metrics\tSUPPRESSED",
                result.scenario.id
            );
            continue;
        }
        let _ = writeln!(
            out,
            "SCENARIO\t{}\t-\t{}\tenumerated_states\t{}",
            result.scenario.id, result.verdict, result.enumerated_states
        );
        let _ = writeln!(
            out,
            "SCENARIO\t{}\t-\t{}\tprobability_mass\t{:.12}",
            result.scenario.id, result.verdict, result.probability_mass
        );
        for metric in result.metrics {
            for (field, value) in [
                ("expected", metric.expected),
                ("bias", metric.bias),
                ("variance", metric.variance),
                ("rmse", metric.rmse),
                ("max_abs_error", metric.max_abs_error),
            ] {
                let _ = writeln!(
                    out,
                    "METRIC\t{}\t{}\t{}\t{}\t{:.12}",
                    result.scenario.id,
                    metric.estimator.name(),
                    result.verdict,
                    field,
                    value
                );
            }
        }
    }
    out.push_str("LIMIT\t-\t-\tALL_SCENARIOS\tconfidence_intervals\tNOT_COMPUTED\n");
    out.push_str("LIMIT\t-\t-\tALL_SCENARIOS\tproduction_authorization\tNOT_CLAIMED\n");
    Ok(out)
}


fn main() {
    let args: Vec<String> = env::args().collect();
    let result = match args.as_slice() {
        [_program, flag] if flag == "--sweep" => run_sweep(GRID),
        [_program, flag, path] if flag == "--sweep-file" => {
            std::fs::read_to_string(path)
                .map_err(|error| invalid(format!("cannot read {path}: {error}")))
                .and_then(|input| run_sweep(&input))
        }
        _ => Err("usage: vision-dvpc-ope-exact --sweep | --sweep-file SCENARIOS.tsv".into()),
    };
    match result {
        Ok(text) => print!("{text}"),
        Err(error) => {
            eprintln!("VIS-DVPC-028 error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(id: &str) -> Scenario {
        parse_grid(GRID).unwrap().into_iter().find(|s| s.id == id).unwrap()
    }

    fn metric<'a>(result: &'a SweepResult, estimator: Estimator) -> &'a Metrics {
        result.metrics.iter().find(|m| m.estimator == estimator).unwrap()
    }

    #[test]
    fn fixture_is_frozen_and_includes_healthy_weak_and_zero_support_cases() {
        let scenarios = parse_grid(GRID).unwrap();
        assert_eq!(scenarios.len(), 8);
        assert_eq!(scenarios[0].id, "healthy_correct");
        assert!(scenarios.iter().any(|s| s.logging_a == 0.9));
        assert!(scenarios.iter().any(|s| s.logging_a == 0.95));
        assert!(scenarios.iter().any(|s| s.logging_a == 0.99));
        assert!(scenarios.iter().any(|s| unsupported_target_mass(s) > EPS));
        assert!(scenarios.iter().any(|s| (s.logging_a - s.estimated_logging_a).abs() > EPS));
    }

    #[test]
    fn exact_distribution_mass_sums_to_one() {
        for s in parse_grid(GRID).unwrap().into_iter().filter(|s| unsupported_target_mass(s) <= EPS) {
            let report = enumerate(&s).unwrap();
            assert!((report.probability_mass - 1.0).abs() < 1e-10, "{}", s.id);
            assert_eq!(report.enumerated_states, 4096);
        }
    }

    #[test]
    fn ips_is_exactly_unbiased_under_true_propensities_and_support() {
        for id in ["healthy_correct", "healthy_misspecified", "weak_overlap_010", "weak_overlap_005", "weak_overlap_001_misspecified"] {
            let report = enumerate(&scenario(id)).unwrap();
            assert!(metric(&report, Estimator::Ips).bias.abs() < 1e-9, "{} bias={}", id, metric(&report, Estimator::Ips).bias);
        }
    }

    #[test]
    fn doubly_robust_mean_is_correct_with_misspecified_q_when_propensities_are_true() {
        for id in ["healthy_misspecified", "weak_overlap_001_misspecified"] {
            let report = enumerate(&scenario(id)).unwrap();
            assert!(metric(&report, Estimator::DoublyRobust).bias.abs() < 1e-9, "{} bias={}", id, metric(&report, Estimator::DoublyRobust).bias);
            assert!(metric(&report, Estimator::Direct).bias.abs() > 0.01, "{} should expose model bias", id);
        }
    }

    #[test]
    fn correct_reward_model_preserves_dr_expectation_under_misspecified_propensity() {
        let report = enumerate(&scenario("propensity_misspecified_correct_q")).unwrap();
        assert!(metric(&report, Estimator::DoublyRobust).bias.abs() < 1e-9);
        assert!(metric(&report, Estimator::Ips).bias.abs() > 0.1);
    }

    #[test]
    fn simultaneous_propensity_and_reward_misspecification_breaks_dr_expectation() {
        let report = enumerate(&scenario("both_misspecified")).unwrap();
        assert!(metric(&report, Estimator::DoublyRobust).bias.abs() > 0.05);
        assert!((metric(&report, Estimator::DoublyRobust).expected - 0.420833333333).abs() < 1e-8);
    }

    #[test]
    fn exact_reference_variance_for_healthy_ips_and_correct_dr() {
        let report = enumerate(&scenario("healthy_correct")).unwrap();
        assert!((metric(&report, Estimator::Ips).variance - 1.0 / 24.0).abs() < 1e-9);
        assert!(metric(&report, Estimator::DoublyRobust).variance < metric(&report, Estimator::Ips).variance);
    }

    #[test]
    fn weak_overlap_increases_ips_variance_and_correct_q_can_reduce_dr_variance() {
        let healthy = enumerate(&scenario("healthy_correct")).unwrap();
        let weak = enumerate(&scenario("weak_overlap_005")).unwrap();
        assert!(metric(&weak, Estimator::Ips).variance > metric(&healthy, Estimator::Ips).variance);
        assert!(metric(&weak, Estimator::DoublyRobust).variance < metric(&weak, Estimator::Ips).variance);
    }

    #[test]
    fn switch_and_clipping_expose_predeclared_bias_variance_tradeoff() {
        let report = enumerate(&scenario("weak_overlap_001_misspecified")).unwrap();
        assert!(metric(&report, Estimator::ClippedIps).bias.abs() > 0.05);
        assert!(metric(&report, Estimator::Switch).bias.abs() > 0.05);
        assert!(metric(&report, Estimator::Switch).variance < metric(&report, Estimator::DoublyRobust).variance);
    }

    #[test]
    fn weak_overlap_misspecification_matches_independent_numeric_known_answers() {
        let report = enumerate(&scenario("weak_overlap_001_misspecified")).unwrap();
        assert!((metric(&report, Estimator::Ips).variance - 3.300084175084).abs() < 1e-8);
        assert!((metric(&report, Estimator::Snips).expected - 0.233448729630).abs() < 1e-8);
        assert!((metric(&report, Estimator::ClippedIps).expected - 0.12).abs() < 1e-8);
        assert!((metric(&report, Estimator::Switch).expected - 0.35).abs() < 1e-8);
        assert!((metric(&report, Estimator::Switch).variance - 0.006743476431).abs() < 1e-8);
    }

    #[test]
    fn full_logging_support_collapses_structural_region_to_true_value() {
        for scenario in parse_grid(GRID).unwrap().into_iter().filter(|scenario| unsupported_target_mass(scenario) <= EPS) {
            let report = enumerate(&scenario).unwrap();
            assert!((report.identified_lower - report.target_value).abs() < EPS, "lower bound: {}", scenario.id);
            assert!((report.identified_upper - report.target_value).abs() < EPS, "upper bound: {}", scenario.id);
            assert!((report.identified_upper - report.identified_lower).abs() < EPS, "region should be singleton: {}", scenario.id);
            assert_eq!(report.verdict, "SUPPORT_OK_SYNTHETIC_ONLY");
        }
    }

    #[test]
    fn zero_support_emits_structural_region_not_a_confidence_interval() {
        let report = enumerate(&scenario("zero_support")).unwrap();
        assert_eq!(report.verdict, "NOT_IDENTIFIED");
        assert!((report.supported_target_contribution - 0.1).abs() < EPS);
        assert!((report.identified_lower - 0.1).abs() < EPS);
        assert!((report.identified_upper - 0.6).abs() < EPS);
        assert!(report.identified_lower <= report.target_value);
        assert!(report.target_value <= report.identified_upper);
        assert!(report.metrics.is_empty());
    }

    #[test]
    fn zero_support_two_worlds_have_identical_logged_law_but_different_target_values() {
        let base = scenario("zero_support");
        let mut low_world = base.clone();
        low_world.reward_b = 0.0;
        let mut high_world = base;
        high_world.reward_b = 1.0;
        assert_eq!(logged_event_probabilities(&low_world), logged_event_probabilities(&high_world));
        assert!((target_value(&low_world) - 0.1).abs() < EPS);
        assert!((target_value(&high_world) - 0.6).abs() < EPS);
        let low = enumerate(&low_world).unwrap();
        let high = enumerate(&high_world).unwrap();
        assert_eq!(low.verdict, "NOT_IDENTIFIED");
        assert_eq!(high.verdict, "NOT_IDENTIFIED");
        assert!((low.identified_lower - high.identified_lower).abs() < EPS);
        assert!((low.identified_upper - high.identified_upper).abs() < EPS);
        assert!(low.identified_lower <= target_value(&low_world));
        assert!(target_value(&low_world) <= low.identified_upper);
        assert!(high.identified_lower <= target_value(&high_world));
        assert!(target_value(&high_world) <= high.identified_upper);
    }

    #[test]
    fn exact_zero_support_is_not_identified_and_metrics_are_suppressed() {
        let report = enumerate(&scenario("zero_support")).unwrap();
        assert_eq!(report.verdict, "NOT_IDENTIFIED");
        assert!((report.unsupported_target_mass - 0.5).abs() < EPS);
        assert!(report.metrics.is_empty());
        assert_eq!(report.enumerated_states, 0);
    }

    #[test]
    fn sweep_is_deterministic_and_reports_all_estimators_without_ci_claims() {
        let first = run_sweep(GRID).unwrap();
        let second = run_sweep(GRID).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.lines().filter(|line| line.starts_with("METRIC\t")).count(), 210);
        assert!(first.contains("SCENARIO\tzero_support\t-\tNOT_IDENTIFIED\testimator_metrics\tSUPPRESSED"));
        assert!(first.contains("METRIC\thealthy_correct\tIPS\tSUPPORT_OK_SYNTHETIC_ONLY\texpected\t0.500000000000"));
        assert!(first.contains("confidence_intervals\tNOT_COMPUTED"));
        assert!(first.starts_with("record_type\tscenario_id\testimator\tverdict\tmetric\tvalue\n"));
        assert!(first.contains("IDENTIFICATION_REGION\tzero_support\t-\tPARTIALLY_IDENTIFIED\tlower_bound\t0.100000000000"));
        assert!(first.contains("IDENTIFICATION_REGION\tzero_support\t-\tPARTIALLY_IDENTIFIED\tupper_bound\t0.600000000000"));
    }

    #[test]
    fn malformed_scenarios_fail_closed() {
        assert!(parse_grid("bad header\n").is_err());
        assert!(parse_grid(&format!("{SCENARIO_HEADER}\nx\t0\t0.5\t0.5\t0.5\t0.2\t0.8\t0.2\t0.8\t2.5\n")).is_err());
        assert!(parse_grid(&format!("{SCENARIO_HEADER}\nx\t6\t1.4\t0.5\t0.5\t0.2\t0.8\t0.2\t0.8\t2.5\n")).is_err());
    }
}
