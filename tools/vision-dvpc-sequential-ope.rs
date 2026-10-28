//! VIS-DVPC-035: exact optional-stopping audit for sequential off-policy evaluation.
//! Fixed-time Wald intervals are compared with an alpha-spending Hoeffding sequence.
//! Assumptions: independent rows, stationary reward means, known row-specific propensities and a bounded IPS contribution.
//! This transparent baseline is not the published OFCS algorithm.
#![deny(unsafe_code)]

use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;
use std::fs;

const FIXTURE: &str = include_str!("../tests/fixtures/vision-dvpc-035-sequential.tsv");
const HEADER: &str = "scenario_id\thorizon\ttarget_probability_a\ttrue_logging_schedule_a\testimated_logging_schedule_a\treward_mean_a\treward_mean_b";
const MAX_HORIZON: usize = 8;
const ALPHA: f64 = 0.05;
const Z_975: f64 = 1.959_963_984_540_054;
const EPS: f64 = 1e-10;

#[derive(Clone, Debug)]
struct Scenario {
    id: String,
    horizon: usize,
    target_a: f64,
    true_logging_schedule: Vec<f64>,
    estimated_logging_schedule: Vec<f64>,
    reward_a: f64,
    reward_b: f64,
}

#[derive(Clone, Debug)]
struct LookMetric {
    look: usize,
    method: &'static str,
    verdict: &'static str,
    coverage: f64,
    mean_width: f64,
    false_promotion_probability: f64,
}

#[derive(Clone, Debug)]
struct ScenarioResult {
    scenario: Scenario,
    truth: f64,
    verdict: &'static str,
    propensities_match: bool,
    enumerated_states: usize,
    probability_mass: f64,
    looks: Vec<LookMetric>,
    wald_simultaneous_coverage: Option<f64>,
    sequence_simultaneous_coverage: Option<f64>,
    wald_ever_false_promotion: Option<f64>,
    sequence_ever_false_promotion: Option<f64>,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn parse_schedule(value: &str, horizon: usize, name: &str, line_number: usize) -> Result<Vec<f64>, String> {
    let values: Vec<f64> = value.split(',')
        .map(|field| field.parse::<f64>()
            .map_err(|_| invalid(format!("line {line_number}: invalid {name} entry"))))
        .collect::<Result<_, _>>()?;
    if values.len() != horizon {
        return Err(invalid(format!("line {line_number}: {name} must contain exactly {horizon} entries")));
    }
    if values.iter().any(|p| !p.is_finite() || !(0.0..=1.0).contains(p)) {
        return Err(invalid(format!("line {line_number}: {name} entries must be finite and in [0,1]")));
    }
    Ok(values)
}

fn parse_fixture(input: &str) -> Result<Vec<Scenario>, String> {
    let mut lines = input.lines();
    if lines.next() != Some(HEADER) {
        return Err(invalid("unexpected sequential fixture header"));
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
        let horizon = fields[1].parse::<usize>()
            .map_err(|_| invalid(format!("line {}: invalid horizon", line_index + 2)))?;
        if horizon != MAX_HORIZON {
            return Err(invalid(format!("line {}: horizon must be frozen at {MAX_HORIZON}", line_index + 2)));
        }
        scenarios.push(Scenario {
            id: fields[0].to_owned(),
            horizon,
            target_a: parse(2, "target_probability_a")?,
            true_logging_schedule: parse_schedule(fields[3], horizon, "true_logging_schedule_a", line_index + 2)?,
            estimated_logging_schedule: parse_schedule(fields[4], horizon, "estimated_logging_schedule_a", line_index + 2)?,
            reward_a: parse(5, "reward_mean_a")?,
            reward_b: parse(6, "reward_mean_b")?,
        });
    }
    if scenarios.is_empty() {
        return Err(invalid("sequential fixture is empty"));
    }
    Ok(scenarios)
}

fn target_value(s: &Scenario) -> f64 {
    s.target_a * s.reward_a + (1.0 - s.target_a) * s.reward_b
}

fn unsupported_target_mass(s: &Scenario) -> f64 {
    let target_b = 1.0 - s.target_a;
    let action_a_ever_logged = s.true_logging_schedule.iter().any(|p| *p > 0.0);
    let action_b_ever_logged = s.true_logging_schedule.iter().any(|p| *p < 1.0);
    (if s.target_a > 0.0 && !action_a_ever_logged { s.target_a } else { 0.0 })
        + (if target_b > 0.0 && !action_b_ever_logged { target_b } else { 0.0 })
}

fn event_probabilities(s: &Scenario, look: usize) -> [f64; 4] {
    let logging_a = s.true_logging_schedule[look];
    [
        logging_a * (1.0 - s.reward_a),
        logging_a * s.reward_a,
        (1.0 - logging_a) * (1.0 - s.reward_b),
        (1.0 - logging_a) * s.reward_b,
    ]
}

fn clipped_interval(center: f64, radius: f64) -> (f64, f64) {
    ((center - radius).max(0.0), (center + radius).min(1.0))
}

fn interval_covers(lower: f64, upper: f64, truth: f64) -> bool {
    lower <= upper && lower - EPS <= truth && truth <= upper + EPS
}

fn alpha_spending_hoeffding_radius(max_importance_weight: f64, look: usize) -> f64 {
    // Sum_{t=1}^infinity alpha_t = alpha, since sum 1/[t(t+1)] = 1.
    // For each look, two-sided Hoeffding has failure probability at most alpha_t.
    let alpha_t = ALPHA / (look as f64 * (look as f64 + 1.0));
    max_importance_weight * ( (2.0 / alpha_t).ln() / (2.0 * look as f64) ).sqrt()
}

fn evaluate(s: &Scenario) -> Result<ScenarioResult, String> {
    let truth = target_value(s);
    let propensities_match = s.true_logging_schedule.iter()
        .zip(&s.estimated_logging_schedule)
        .all(|(truth, estimate)| (truth - estimate).abs() <= EPS);
    if unsupported_target_mass(s) > EPS {
        return Ok(ScenarioResult {
            scenario: s.clone(),
            truth,
            verdict: "NOT_IDENTIFIED",
            propensities_match,
            enumerated_states: 0,
            probability_mass: 0.0,
            looks: Vec::new(),
            wald_simultaneous_coverage: None,
            sequence_simultaneous_coverage: None,
            wald_ever_false_promotion: None,
            sequence_ever_false_promotion: None,
        });
    }

    let target_b = 1.0 - s.target_a;
    if s.estimated_logging_schedule.iter().any(|p| {
        (s.target_a > 0.0 && *p <= 0.0) || (target_b > 0.0 && *p >= 1.0)
    }) {
        return Err(invalid(format!("{}: estimated logging propensity is zero for a target-supported action", s.id)));
    }
    let max_weight = s.estimated_logging_schedule.iter()
        .map(|p| {
            (if s.target_a > 0.0 { s.target_a / *p } else { 0.0 })
                .max(if target_b > 0.0 { target_b / (1.0 - *p) } else { 0.0 })
        })
        .fold(0.0_f64, f64::max);
    if !max_weight.is_finite() || max_weight <= 0.0 {
        return Err(invalid(format!("{}: invalid maximum importance weight", s.id)));
    }

    let state_count = 4usize.pow(s.horizon as u32);
    let mut mass = 0.0;
    let mut wald_coverage = vec![0.0_f64; s.horizon + 1];
    let mut wald_width = vec![0.0_f64; s.horizon + 1];
    let mut wald_false_promotion = vec![0.0_f64; s.horizon + 1];
    let mut seq_coverage = vec![0.0_f64; s.horizon + 1];
    let mut seq_width = vec![0.0_f64; s.horizon + 1];
    let mut seq_false_promotion = vec![0.0_f64; s.horizon + 1];
    let mut wald_simultaneous = 0.0;
    let mut sequence_simultaneous = 0.0;
    let mut wald_ever_promotion = 0.0;
    let mut sequence_ever_promotion = 0.0;

    for state in 0..state_count {
        let mut code = state;
        let mut probability = 1.0;
        let mut samples = Vec::with_capacity(s.horizon);
        for look in 0..s.horizon {
            let event = code % 4;
            code /= 4;
            let probabilities = event_probabilities(s, look);
            probability *= probabilities[event];
            let action_a = event < 2;
            let reward = if event % 2 == 0 { 0.0 } else { 1.0 };
            let estimated_logging_a = s.estimated_logging_schedule[look];
            let estimated_logging_b = 1.0 - estimated_logging_a;
            let weight = if action_a {
                if s.target_a > 0.0 { s.target_a / estimated_logging_a } else { 0.0 }
            } else if target_b > 0.0 {
                target_b / estimated_logging_b
            } else {
                0.0
            };
            samples.push(weight * reward);
        }
        if probability <= 0.0 {
            continue;
        }

        let mut count = 0usize;
        let mut mean = 0.0;
        let mut m2 = 0.0;
        let mut all_wald_cover = true;
        let mut all_sequence_cover = true;
        let mut any_wald_false_promotion = false;
        let mut any_sequence_false_promotion = false;

        for value in samples {
            count += 1;
            let delta = value - mean;
            mean += delta / count as f64;
            m2 += delta * (value - mean);

            let cs_radius = alpha_spending_hoeffding_radius(max_weight, count);
            let (cs_lower, cs_upper) = clipped_interval(mean, cs_radius);
            let cs_covers = interval_covers(cs_lower, cs_upper, truth);
            if !cs_covers {
                all_sequence_cover = false;
            }
            let cs_promote = cs_lower > truth;
            if cs_promote {
                any_sequence_false_promotion = true;
            }
            seq_coverage[count] += probability * if cs_covers { 1.0 } else { 0.0 };
            seq_width[count] += probability * (cs_upper - cs_lower).max(0.0);
            seq_false_promotion[count] += probability * if cs_promote { 1.0 } else { 0.0 };

            if count >= 2 {
                let variance = m2 / (count - 1) as f64;
                let radius = Z_975 * (variance / count as f64).sqrt();
                let (lower, upper) = clipped_interval(mean, radius);
                let covers = interval_covers(lower, upper, truth);
                if !covers {
                    all_wald_cover = false;
                }
                let promote = lower > truth;
                if promote {
                    any_wald_false_promotion = true;
                }
                wald_coverage[count] += probability * if covers { 1.0 } else { 0.0 };
                wald_width[count] += probability * (upper - lower).max(0.0);
                wald_false_promotion[count] += probability * if promote { 1.0 } else { 0.0 };
            }
        }
        mass += probability;
        if all_wald_cover {
            wald_simultaneous += probability;
        }
        if all_sequence_cover {
            sequence_simultaneous += probability;
        }
        if any_wald_false_promotion {
            wald_ever_promotion += probability;
        }
        if any_sequence_false_promotion {
            sequence_ever_promotion += probability;
        }
    }

    if (mass - 1.0).abs() > 1e-8 {
        return Err(invalid(format!("{}: enumerated probability mass {mass} != 1", s.id)));
    }

    let assumption_verdict = if propensities_match {
        "ASSUMPTIONS_MATCH_SYNTHETIC_DGP"
    } else {
        "ASSUMPTION_VIOLATION_DESCRIPTIVE_ONLY"
    };
    let mut looks = Vec::new();
    for look in 1..=s.horizon {
        looks.push(LookMetric {
            look,
            method: "ALPHA_SPENDING_HOEFFDING_CS",
            verdict: assumption_verdict,
            coverage: seq_coverage[look] / mass,
            mean_width: seq_width[look] / mass,
            false_promotion_probability: seq_false_promotion[look] / mass,
        });
        if look >= 2 {
            looks.push(LookMetric {
                look,
                method: "WALD_FIXED_TIME",
                verdict: assumption_verdict,
                coverage: wald_coverage[look] / mass,
                mean_width: wald_width[look] / mass,
                false_promotion_probability: wald_false_promotion[look] / mass,
            });
        }
    }

    Ok(ScenarioResult {
        scenario: s.clone(),
        truth,
        verdict: "SUPPORT_OK_SYNTHETIC_ONLY",
        propensities_match,
        enumerated_states: state_count,
        probability_mass: mass,
        looks,
        wald_simultaneous_coverage: Some(wald_simultaneous / mass),
        sequence_simultaneous_coverage: Some(sequence_simultaneous / mass),
        wald_ever_false_promotion: Some(wald_ever_promotion / mass),
        sequence_ever_false_promotion: Some(sequence_ever_promotion / mass),
    })
}

fn sweep(input: &str) -> Result<String, String> {
    let scenarios = parse_fixture(input)?;
    let mut out = String::from("record_type\tscenario_id\tmethod\tverdict\tmetric\tvalue\n");
    let mut supported_scenarios = 0usize;
    for scenario in scenarios {
        let result = evaluate(&scenario)?;
        let _ = writeln!(out, "SCENARIO\t{}\t-\t{}\ttarget_truth_evaluator_only\t{:.12}",
            result.scenario.id, result.verdict, result.truth);
        let _ = writeln!(out, "SCENARIO\t{}\t-\t{}\tpropensities_match\t{}",
            result.scenario.id, result.verdict, result.propensities_match);
        if result.verdict == "NOT_IDENTIFIED" {
            let _ = writeln!(out, "SCENARIO\t{}\t-\tNOT_IDENTIFIED\tconfidence_intervals\tSUPPRESSED",
                result.scenario.id);
            continue;
        }
        supported_scenarios += 1;
        let _ = writeln!(out, "SCENARIO\t{}\t-\t{}\tenumerated_states\t{}",
            result.scenario.id, result.verdict, result.enumerated_states);
        let _ = writeln!(out, "SCENARIO\t{}\t-\t{}\tprobability_mass\t{:.12}",
            result.scenario.id, result.verdict, result.probability_mass);
        for metric in &result.looks {
            for (name, value) in [
                ("pointwise_coverage", metric.coverage),
                ("mean_interval_width", metric.mean_width),
                ("false_promotion_probability_at_look", metric.false_promotion_probability),
            ] {
                let _ = writeln!(out, "LOOK_METRIC\t{}\t{}_N{}\t{}\t{}\t{:.12}",
                    result.scenario.id, metric.method, metric.look, metric.verdict, name, value);
            }
        }
        let _ = writeln!(out, "AGGREGATE\t{}\tWALD_REPEATED_LOOKS\t{}\tsimultaneous_coverage_n2_to_n8\t{:.12}",
            result.scenario.id,
            if result.propensities_match { "ASSUMPTIONS_MATCH_SYNTHETIC_DGP" } else { "ASSUMPTION_VIOLATION_DESCRIPTIVE_ONLY" },
            result.wald_simultaneous_coverage.unwrap_or(0.0));
        let _ = writeln!(out, "AGGREGATE\t{}\tWALD_REPEATED_LOOKS\t{}\tever_false_promotion_probability_n2_to_n8\t{:.12}",
            result.scenario.id,
            if result.propensities_match { "ASSUMPTIONS_MATCH_SYNTHETIC_DGP" } else { "ASSUMPTION_VIOLATION_DESCRIPTIVE_ONLY" },
            result.wald_ever_false_promotion.unwrap_or(0.0));
        let _ = writeln!(out, "AGGREGATE\t{}\tALPHA_SPENDING_HOEFFDING_CS\t{}\tsimultaneous_coverage_n1_to_n8\t{:.12}",
            result.scenario.id,
            if result.propensities_match { "ASSUMPTIONS_MATCH_SYNTHETIC_DGP" } else { "ASSUMPTION_VIOLATION_DESCRIPTIVE_ONLY" },
            result.sequence_simultaneous_coverage.unwrap_or(0.0));
        let _ = writeln!(out, "AGGREGATE\t{}\tALPHA_SPENDING_HOEFFDING_CS\t{}\tever_false_promotion_probability_n1_to_n8\t{:.12}",
            result.scenario.id,
            if result.propensities_match { "ASSUMPTIONS_MATCH_SYNTHETIC_DGP" } else { "ASSUMPTION_VIOLATION_DESCRIPTIVE_ONLY" },
            result.sequence_ever_false_promotion.unwrap_or(0.0));
    }
    let _ = writeln!(out, "SUMMARY\tALL_SCENARIOS\t-\t-\tsupported_scenarios\t{supported_scenarios}");
    out.push_str("LIMIT\tALL_SCENARIOS\t-\tindependent_rows_stationary_rewards_known_row_level_propensities_bounded_ips_assumptions\tREQUIRED\n");
    out.push_str("LIMIT\tALL_SCENARIOS\t-\tpublished_off_policy_confidence_sequence_algorithm\tNOT_IMPLEMENTED\n");
    out.push_str("LIMIT\tALL_SCENARIOS\t-\treal_trace_calibration\tNOT_CLAIMED\n");
    out.push_str("LIMIT\tALL_SCENARIOS\t-\tzero_support_intervals\tSUPPRESSED\n");
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
        _ => Err("usage: vision-dvpc-sequential-ope --sweep | --sweep-file SCENARIOS.tsv".into()),
    };
    match result {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS-DVPC-035 error: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(id: &str) -> Scenario {
        parse_fixture(FIXTURE).unwrap().into_iter().find(|s| s.id == id).unwrap()
    }

    fn result(id: &str) -> ScenarioResult {
        evaluate(&scenario(id)).unwrap()
    }

    fn aggregate(result: &ScenarioResult, method: &str) -> f64 {
        match method {
            "WALD" => result.wald_simultaneous_coverage.unwrap(),
            "CS" => result.sequence_simultaneous_coverage.unwrap(),
            _ => panic!("unknown method"),
        }
    }

    #[test]
    fn fixture_pins_healthy_weak_misspecified_and_zero_support() {
        let scenarios = parse_fixture(FIXTURE).unwrap();
        assert_eq!(scenarios.len(), 6);
        assert!(scenarios.iter().any(|s| s.true_logging_schedule.iter().any(|p| *p == 0.99)
            && s.true_logging_schedule.iter().zip(&s.estimated_logging_schedule).all(|(a, b)| (a - b).abs() <= EPS)));
        assert!(scenarios.iter().any(|s| s.true_logging_schedule.iter().zip(&s.estimated_logging_schedule)
            .any(|(a, b)| (a - b).abs() > EPS)));
        assert!(scenarios.iter().any(|s| unsupported_target_mass(s) > EPS));
        assert!(scenarios.iter().any(|s| s.id == "rotating_logger"
            && s.true_logging_schedule.iter().any(|p| *p < 0.6)
            && s.true_logging_schedule.iter().any(|p| *p > 0.95)));
    }

    #[test]
    fn all_supported_joint_distributions_enumerate_exactly() {
        for scenario in parse_fixture(FIXTURE).unwrap().into_iter().filter(|s| unsupported_target_mass(s) <= EPS) {
            let result = evaluate(&scenario).unwrap();
            assert_eq!(result.enumerated_states, 4usize.pow(MAX_HORIZON as u32));
            assert!((result.probability_mass - 1.0).abs() < 1e-8);
        }
    }

    #[test]
    fn repeated_wald_looks_inflate_false_promotion_and_destroy_simultaneous_coverage() {
        let r = result("healthy_correct");
        assert!((aggregate(&r, "WALD") - 0.4375).abs() < 1e-8);
        assert!((r.wald_ever_false_promotion.unwrap() - 0.28125).abs() < 1e-8);
        assert!(r.wald_ever_false_promotion.unwrap() > ALPHA);
        assert!(r.wald_simultaneous_coverage.unwrap() < 0.95);
    }

    #[test]
    fn alpha_spending_sequence_preserves_coverage_under_correct_propensities() {
        for id in ["healthy_correct", "weak_overlap_010", "weak_overlap_001", "rotating_logger"] {
            let r = result(id);
            assert!(r.propensities_match);
            assert!(r.sequence_simultaneous_coverage.unwrap() + 1e-9 >= 1.0 - ALPHA,
                "{} simultaneous={}", id, r.sequence_simultaneous_coverage.unwrap());
            assert!(r.sequence_ever_false_promotion.unwrap() <= ALPHA + 1e-9,
                "{} false-promotion={}", id, r.sequence_ever_false_promotion.unwrap());
        }
    }

    #[test]
    fn known_weak_overlap_control_is_conservative_not_misreported_as_precise() {
        let r = result("weak_overlap_001");
        assert!(r.sequence_simultaneous_coverage.unwrap() > 0.999);
        let final_cs = r.looks.iter().find(|m| m.method == "ALPHA_SPENDING_HOEFFDING_CS" && m.look == 8).unwrap();
        assert!(final_cs.mean_width > 0.999);
    }

    #[test]
    fn misspecified_propensities_are_descriptive_only() {
        let r = result("propensity_misspecified");
        assert!(!r.propensities_match);
        assert!(r.looks.iter().all(|m| m.verdict == "ASSUMPTION_VIOLATION_DESCRIPTIVE_ONLY"));
    }

    #[test]
    fn zero_support_suppresses_all_intervals() {
        let r = result("zero_support");
        assert_eq!(r.verdict, "NOT_IDENTIFIED");
        assert!(r.looks.is_empty());
        assert_eq!(r.enumerated_states, 0);
        assert!(r.sequence_simultaneous_coverage.is_none());
    }

    #[test]
    fn malformed_reward_values_and_duplicate_scenario_ids_fail_closed() {
        let invalid_reward = FIXTURE.replace("\t0.2\t0.8\n", "\t1.2\t0.8\n");
        assert!(parse_fixture(&invalid_reward).is_err());
        let duplicate = format!("{FIXTURE}healthy_correct\t8\t0.5\t0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5\t0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5\t0.2\t0.8\n");
        assert!(parse_fixture(&duplicate).is_err());
    }

    #[test]
    fn alpha_spending_sums_to_nominal_error_budget() {
        let total: f64 = (1..=100_000).map(|n| ALPHA / ((n as f64) * (n as f64 + 1.0))).sum();
        assert!((total - ALPHA).abs() < 1e-6);
    }

    #[test]
    fn report_explicitly_limits_claims() {
        let out = sweep(FIXTURE).unwrap();
        assert!(out.contains("published_off_policy_confidence_sequence_algorithm\tNOT_IMPLEMENTED"));
        assert!(out.contains("real_trace_calibration\tNOT_CLAIMED"));
        assert!(out.contains("confidence_intervals\tSUPPRESSED"));
    }
}
