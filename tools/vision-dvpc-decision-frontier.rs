//! VIS-DVPC-036: exact-DP frontier for sequential off-policy decision efficiency.
//!
//! Fixed-logger iid two-action Bernoulli DGPs have three IPS contribution values:
//! zero, target_a/logging_a, or (1-target_a)/(1-logging_a). This program exactly
//! aggregates paths by counts of the two positive contributions rather than
//! enumerating 3^H histories. At look H it visits O(H^2) states; cumulatively O(H^3)
//! transitions and O(H^2) memory.
//!
//! Compared methods:
//! - Reused fixed-time 95% Wald intervals, a deliberately unsafe stopping baseline.
//! - A transparent alpha-spending Hoeffding confidence-sequence baseline.
//!
//! Synthetic iid/fixed-logger only. This is not the published OFCS algorithm.
#![deny(unsafe_code)]

use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;
use std::fs;

const FIXTURE: &str = include_str!("../tests/fixtures/vision-dvpc-036-decision.tsv");
const HEADER: &str = "scenario_id\thorizon\ttarget_probability_a\tlogging_probability_a\treward_mean_a\treward_mean_b\tthreshold";
const MAX_HORIZON: usize = 256;
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
struct LookMetric {
    look: usize,
    method: &'static str,
    pointwise_coverage: f64,
    mean_interval_width: f64,
    pointwise_threshold_crossing_probability: f64,
    first_threshold_crossing_probability: f64,
    ever_threshold_crossing_probability: f64,
    survival_probability: f64,
    sequence_coverage_to_look: f64,
}

#[derive(Clone, Debug)]
struct Evaluation {
    scenario: Scenario,
    truth: f64,
    total_dp_state_evaluations: usize,
    probability_mass_by_look: Vec<f64>,
    reachable_states_by_look: Vec<usize>,
    looks: Vec<LookMetric>,
    sequence_coverage_wald: f64,
    sequence_coverage_cs: f64,
    ever_crossing_wald: f64,
    ever_crossing_cs: f64,
    conditional_mean_first_crossing_wald: Option<f64>,
    conditional_mean_first_crossing_cs: Option<f64>,
}

fn invalid(message: impl Into<String>) -> String {
    message.into()
}

fn parse_fixture(input: &str) -> Result<Vec<Scenario>, String> {
    let mut lines = input.lines();
    if lines.next() != Some(HEADER) {
        return Err(invalid("unexpected decision-frontier fixture header"));
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
        if !(2..=MAX_HORIZON).contains(&horizon) {
            return Err(invalid(format!("line {line_number}: horizon must be in 2..={MAX_HORIZON}")));
        }
        let target_a = parse(2, "target_probability_a")?;
        let logging_a = parse(3, "logging_probability_a")?;
        let reward_a = parse(4, "reward_mean_a")?;
        let reward_b = parse(5, "reward_mean_b")?;
        let threshold = parse(6, "threshold")?;
        if !(0.0..1.0).contains(&target_a) {
            return Err(invalid(format!("line {line_number}: target_probability_a must be in (0,1)")));
        }
        if !(0.0..1.0).contains(&logging_a) {
            return Err(invalid(format!("line {line_number}: logging_probability_a must be in (0,1)")));
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
        return Err(invalid("decision-frontier fixture is empty"));
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

fn alpha_spending_hoeffding_radius(max_weight: f64, look: usize) -> f64 {
    // alpha_t = alpha/[t(t+1)]; sum_t alpha_t = alpha.
    // Hoeffding gives P(|mean - truth| >= r_t) <= alpha_t.
    let t = look as f64;
    let alpha_t = ALPHA / (t * (t + 1.0));
    max_weight * ((2.0 / alpha_t).ln() / (2.0 * t)).sqrt()
}

fn sum_reachable(values: &[f64], stride: usize, look: usize) -> f64 {
    let mut sum = 0.0;
    for a in 0..=look {
        for b in 0..=(look - a) {
            sum += values[a * stride + b];
        }
    }
    sum
}

fn evaluate(s: &Scenario) -> Result<Evaluation, String> {
    let truth = target_value(s);
    let target_b = 1.0 - s.target_a;
    let weight_a = s.target_a / s.logging_a;
    let weight_b = target_b / (1.0 - s.logging_a);
    let max_weight = weight_a.max(weight_b);
    if !max_weight.is_finite() || max_weight <= 0.0 {
        return Err(invalid(format!("{}: invalid importance-weight bound", s.id)));
    }

    // Category X=0 includes every logged action with reward zero. Its estimator
    // contribution is identical, so the category probability is the residual.
    let p_a_success = s.logging_a * s.reward_a;
    let p_b_success = (1.0 - s.logging_a) * s.reward_b;
    let p_zero = 1.0 - p_a_success - p_b_success;
    if [p_zero, p_a_success, p_b_success].iter().any(|p| !p.is_finite() || *p < -EPS) {
        return Err(invalid(format!("{}: invalid outcome-category probabilities", s.id)));
    }

    let stride = s.horizon + 1;
    let cells = stride * stride;
    let mut mass = vec![0.0_f64; cells];
    let mut cover_wald = vec![0.0_f64; cells];
    let mut cover_cs = vec![0.0_f64; cells];
    let mut no_promote_wald = vec![0.0_f64; cells];
    let mut no_promote_cs = vec![0.0_f64; cells];
    mass[0] = 1.0;
    cover_wald[0] = 1.0;
    cover_cs[0] = 1.0;
    no_promote_wald[0] = 1.0;
    no_promote_cs[0] = 1.0;

    let mut probability_mass_by_look = Vec::with_capacity(s.horizon);
    let mut reachable_states_by_look = Vec::with_capacity(s.horizon);
    let mut looks = Vec::with_capacity(2 * s.horizon - 1);
    let mut total_dp_state_evaluations = 0usize;
    let mut sequence_coverage_wald = 1.0;
    let mut sequence_coverage_cs = 1.0;
    let mut survival_wald = 1.0;
    let mut survival_cs = 1.0;
    let mut expected_first_crossing_wald = 0.0;
    let mut expected_first_crossing_cs = 0.0;

    for look in 1..=s.horizon {
        let mut next_mass = vec![0.0_f64; cells];
        let mut next_cover_wald = vec![0.0_f64; cells];
        let mut next_cover_cs = vec![0.0_f64; cells];
        let mut next_no_promote_wald = vec![0.0_f64; cells];
        let mut next_no_promote_cs = vec![0.0_f64; cells];

        for a in 0..look {
            for b in 0..(look - a) {
                let old = a * stride + b;
                let transitions = [
                    (a * stride + b, p_zero),
                    ((a + 1) * stride + b, p_a_success),
                    (a * stride + b + 1, p_b_success),
                ];
                for (new, probability) in transitions {
                    next_mass[new] += mass[old] * probability;
                    next_cover_wald[new] += cover_wald[old] * probability;
                    next_cover_cs[new] += cover_cs[old] * probability;
                    next_no_promote_wald[new] += no_promote_wald[old] * probability;
                    next_no_promote_cs[new] += no_promote_cs[old] * probability;
                }
            }
        }

        let reachable = (look + 1) * (look + 2) / 2;
        total_dp_state_evaluations += reachable;
        reachable_states_by_look.push(reachable);

        let mut mass_at_look = 0.0;
        let mut wald_coverage = 0.0;
        let mut cs_coverage = 0.0;
        let mut wald_width = 0.0;
        let mut cs_width = 0.0;
        let mut wald_point_cross = 0.0;
        let mut cs_point_cross = 0.0;
        let mut wald_no_promote_before = 0.0;
        let mut cs_no_promote_before = 0.0;
        let mut wald_no_promote_after = 0.0;
        let mut cs_no_promote_after = 0.0;

        let cs_radius = alpha_spending_hoeffding_radius(max_weight, look);
        for a in 0..=look {
            for b in 0..=(look - a) {
                let idx = a * stride + b;
                let state_mass = next_mass[idx];
                mass_at_look += state_mass;
                wald_no_promote_before += next_no_promote_wald[idx];
                cs_no_promote_before += next_no_promote_cs[idx];

                let sum = a as f64 * weight_a + b as f64 * weight_b;
                let sum_squares = a as f64 * weight_a * weight_a + b as f64 * weight_b * weight_b;
                let mean = sum / look as f64;

                let (cs_lower, cs_upper) = clipped_interval(mean, cs_radius);
                cs_coverage += state_mass * if covers(cs_lower, cs_upper, truth) { 1.0 } else { 0.0 };
                cs_width += state_mass * (cs_upper - cs_lower).max(0.0);
                if cs_lower > s.threshold {
                    cs_point_cross += state_mass;
                    next_no_promote_cs[idx] = 0.0;
                }
                if !covers(cs_lower, cs_upper, truth) {
                    next_cover_cs[idx] = 0.0;
                }

                if look >= 2 {
                    let m2 = (sum_squares - sum * sum / look as f64).max(0.0);
                    let sample_variance = m2 / (look - 1) as f64;
                    let radius = Z_975 * (sample_variance / look as f64).sqrt();
                    let (wald_lower, wald_upper) = clipped_interval(mean, radius);
                    wald_coverage += state_mass * if covers(wald_lower, wald_upper, truth) { 1.0 } else { 0.0 };
                    wald_width += state_mass * (wald_upper - wald_lower).max(0.0);
                    if wald_lower > s.threshold {
                        wald_point_cross += state_mass;
                        next_no_promote_wald[idx] = 0.0;
                    }
                    if !covers(wald_lower, wald_upper, truth) {
                        next_cover_wald[idx] = 0.0;
                    }
                }
                wald_no_promote_after += next_no_promote_wald[idx];
                cs_no_promote_after += next_no_promote_cs[idx];
            }
        }

        if (mass_at_look - 1.0).abs() > MASS_TOLERANCE {
            return Err(invalid(format!("{} at look {look}: DP probability mass {mass_at_look:.12} != 1", s.id)));
        }
        probability_mass_by_look.push(mass_at_look);

        sequence_coverage_cs = sum_reachable(&next_cover_cs, stride, look);
        survival_cs = cs_no_promote_after.clamp(0.0, 1.0);
        let cs_first_cross = (cs_no_promote_before - cs_no_promote_after).max(0.0);
        expected_first_crossing_cs += look as f64 * cs_first_cross;

        if look >= 2 {
            sequence_coverage_wald = sum_reachable(&next_cover_wald, stride, look);
            survival_wald = wald_no_promote_after.clamp(0.0, 1.0);
            let wald_first_cross = (wald_no_promote_before - wald_no_promote_after).max(0.0);
            expected_first_crossing_wald += look as f64 * wald_first_cross;
            looks.push(LookMetric {
                look,
                method: "WALD_FIXED_TIME",
                pointwise_coverage: (wald_coverage / mass_at_look).clamp(0.0, 1.0),
                mean_interval_width: (wald_width / mass_at_look).clamp(0.0, 1.0),
                pointwise_threshold_crossing_probability: (wald_point_cross / mass_at_look).clamp(0.0, 1.0),
                first_threshold_crossing_probability: wald_first_cross.clamp(0.0, 1.0),
                ever_threshold_crossing_probability: (1.0 - survival_wald).clamp(0.0, 1.0),
                survival_probability: survival_wald,
                sequence_coverage_to_look: sequence_coverage_wald.clamp(0.0, 1.0),
            });
        }

        looks.push(LookMetric {
            look,
            method: "ALPHA_SPENDING_HOEFFDING_CS",
            pointwise_coverage: (cs_coverage / mass_at_look).clamp(0.0, 1.0),
            mean_interval_width: (cs_width / mass_at_look).clamp(0.0, 1.0),
            pointwise_threshold_crossing_probability: (cs_point_cross / mass_at_look).clamp(0.0, 1.0),
            first_threshold_crossing_probability: cs_first_cross.clamp(0.0, 1.0),
            ever_threshold_crossing_probability: (1.0 - survival_cs).clamp(0.0, 1.0),
            survival_probability: survival_cs,
            sequence_coverage_to_look: sequence_coverage_cs.clamp(0.0, 1.0),
        });

        mass = next_mass;
        cover_wald = next_cover_wald;
        cover_cs = next_cover_cs;
        no_promote_wald = next_no_promote_wald;
        no_promote_cs = next_no_promote_cs;
    }

    let ever_crossing_wald = (1.0 - survival_wald).clamp(0.0, 1.0);
    let ever_crossing_cs = (1.0 - survival_cs).clamp(0.0, 1.0);
    Ok(Evaluation {
        scenario: s.clone(),
        truth,
        total_dp_state_evaluations,
        probability_mass_by_look,
        reachable_states_by_look,
        looks,
        sequence_coverage_wald: sequence_coverage_wald.clamp(0.0, 1.0),
        sequence_coverage_cs: sequence_coverage_cs.clamp(0.0, 1.0),
        ever_crossing_wald,
        ever_crossing_cs,
        conditional_mean_first_crossing_wald: if ever_crossing_wald > EPS {
            Some(expected_first_crossing_wald / ever_crossing_wald)
        } else { None },
        conditional_mean_first_crossing_cs: if ever_crossing_cs > EPS {
            Some(expected_first_crossing_cs / ever_crossing_cs)
        } else { None },
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
        let class = if (result.truth - scenario.threshold).abs() <= EPS {
            "NULL_THRESHOLD_EQUALS_TRUTH"
        } else if result.truth > scenario.threshold {
            "POSITIVE_MARGIN_ALTERNATIVE"
        } else {
            "THRESHOLD_ABOVE_TRUTH"
        };
        write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-", class, "truth", result.truth);
        write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-", class, "threshold", scenario.threshold);
        write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-", class, "final_probability_mass",
            *result.probability_mass_by_look.last().unwrap_or(&0.0));
        write_record(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-", class, "total_dp_state_evaluations",
            &result.total_dp_state_evaluations.to_string());
        write_record(&mut out, "SCENARIO", &scenario.id, scenario.horizon, "-", class, "path_enumeration", "NOT_USED");
        for (look, states) in result.reachable_states_by_look.iter().enumerate() {
            write_number(&mut out, "DP_LOOK", &scenario.id, look + 1, "-", class, "probability_mass",
                result.probability_mass_by_look[look]);
            write_record(&mut out, "DP_LOOK", &scenario.id, look + 1, "-", class, "reachable_state_count", &states.to_string());
        }
        for metric in &result.looks {
            for (name, value) in [
                ("pointwise_coverage", metric.pointwise_coverage),
                ("mean_interval_width", metric.mean_interval_width),
                ("pointwise_threshold_crossing_probability", metric.pointwise_threshold_crossing_probability),
                ("first_threshold_crossing_probability", metric.first_threshold_crossing_probability),
                ("ever_threshold_crossing_probability", metric.ever_threshold_crossing_probability),
                ("survival_probability", metric.survival_probability),
                ("sequence_coverage_to_look", metric.sequence_coverage_to_look),
            ] {
                write_number(&mut out, "LOOK_METRIC", &scenario.id, metric.look, metric.method, class, name, value);
            }
        }

        let interpretation = match class {
            "NULL_THRESHOLD_EQUALS_TRUTH" => "FALSE_PROMOTION_NULL",
            "POSITIVE_MARGIN_ALTERNATIVE" => "DECISION_POWER_ALTERNATIVE",
            _ => "THRESHOLD_ABOVE_TRUTH",
        };
        for (method, coverage, crossing, conditional_time) in [
            ("WALD_FIXED_TIME", result.sequence_coverage_wald, result.ever_crossing_wald,
                result.conditional_mean_first_crossing_wald),
            ("ALPHA_SPENDING_HOEFFDING_CS", result.sequence_coverage_cs, result.ever_crossing_cs,
                result.conditional_mean_first_crossing_cs),
        ] {
            write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, method, class,
                "sequence_coverage", coverage);
            write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, method, class,
                "ever_threshold_crossing_probability", crossing);
            write_record(&mut out, "SCENARIO", &scenario.id, scenario.horizon, method, class,
                "crossing_interpretation", interpretation);
            match conditional_time {
                Some(value) => write_number(&mut out, "SCENARIO", &scenario.id, scenario.horizon, method, class,
                    "conditional_mean_first_crossing_look", value),
                None => write_record(&mut out, "SCENARIO", &scenario.id, scenario.horizon, method, class,
                    "conditional_mean_first_crossing_look", "NOT_REACHED"),
            }
        }
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
        _ => Err("usage: vision-dvpc-decision-frontier --sweep | --input <fixture.tsv>".to_owned()),
    };
    match result {
        Ok(output) => print!("{output}"),
        Err(error) => {
            eprintln!("VIS_DVPC_036_ERROR\t{error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario_by_id(id: &str) -> Scenario {
        parse_fixture(FIXTURE).unwrap().into_iter().find(|s| s.id == id).unwrap()
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!((actual - expected).abs() <= tolerance, "actual={actual:.15}, expected={expected:.15}");
    }

    #[test]
    fn fixture_is_unique_valid_and_contains_scaling_grid() {
        let scenarios = parse_fixture(FIXTURE).unwrap();
        assert_eq!(scenarios.len(), 6);
        assert!(scenarios.iter().any(|s| s.id == "healthy_null_h8" && s.horizon == 8));
        assert!(scenarios.iter().any(|s| s.id == "healthy_null_h256" && s.horizon == 256));
        assert!(scenarios.iter().any(|s| s.id == "weak_alt_small_p099_h256" && s.logging_a > 0.98));
        assert!(parse_fixture(&format!("{HEADER}\ninvalid\t999\t0.5\t0.5\t0.2\t0.8\t0.5\n")).is_err());
    }

    #[test]
    fn dp_reproduces_known_horizon_eight_optional_stopping_controls() {
        let result = evaluate(&scenario_by_id("healthy_null_h8")).unwrap();
        close(result.sequence_coverage_wald, 0.4375, 1e-9);
        close(result.ever_crossing_wald, 0.28125, 1e-9);
        close(result.sequence_coverage_cs, 1.0, 1e-9);
        close(result.ever_crossing_cs, 0.0, 1e-9);
    }

    #[test]
    fn probability_mass_sums_to_one_at_every_look() {
        let result = evaluate(&scenario_by_id("healthy_null_h8")).unwrap();
        assert_eq!(result.probability_mass_by_look.len(), 8);
        for (look, mass) in result.probability_mass_by_look.iter().enumerate() {
            close(*mass, 1.0, MASS_TOLERANCE);
            assert_eq!(result.reachable_states_by_look[look], (look + 2) * (look + 3) / 2);
        }
    }

    #[test]
    fn dynamic_program_state_count_is_polynomial_and_bounded() {
        let mut scenario = scenario_by_id("healthy_null_h8");
        scenario.horizon = 32;
        let result = evaluate(&scenario).unwrap();
        let h = scenario.horizon;
        let expected = (h + 1) * (h + 2) * (h + 3) / 6 - 1;
        assert_eq!(result.total_dp_state_evaluations, expected);
        assert!(result.total_dp_state_evaluations < 10_000);
    }

    #[test]
    fn null_and_alternative_interpretations_are_not_conflated() {
        let scenarios = parse_fixture(FIXTURE).unwrap();
        let null = scenarios.iter().find(|s| s.id == "healthy_null_h8").unwrap();
        let alternative = scenarios.iter().find(|s| s.id == "healthy_alt_clear_h256").unwrap();
        assert!((target_value(null) - null.threshold).abs() <= EPS);
        assert!(target_value(alternative) > alternative.threshold);
    }
}
