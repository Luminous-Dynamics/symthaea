//! Exact finite-state benchmark for sequential model-adequacy alarms.
//!
//! This monitor tests whether a known nominal Bernoulli rate p0 remains compatible with an
//! observed calibration stream. It contrasts a valid single fixed-horizon look, a deliberately
//! invalid practice of reusing fixed-time intervals after every observation, and a time-uniform
//! alpha-spending boundary. The step-shift scenario measures power/delay, not confidence coverage
//! under nonstationarity.

use std::{env, process::ExitCode};

const ALPHA: f64 = 0.05;
const CELLS: f64 = 12.0;
const PER_CELL_ALPHA: f64 = ALPHA / CELLS;
const NOMINAL_P: f64 = 0.5;
const SHIFTED_P: f64 = 0.8;
const CHANGE_AFTER: usize = 100;
const HORIZONS: [usize; 4] = [100, 250, 500, 1000];
const EPS: f64 = 1.0e-12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Boundary {
    FixedFinalOnly,
    FixedTimeRepeated,
    TimeUniform,
}

impl Boundary {
    const ALL: [Self; 3] = [
        Self::FixedFinalOnly,
        Self::FixedTimeRepeated,
        Self::TimeUniform,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::FixedFinalOnly => "fixed_time_final_only",
            Self::FixedTimeRepeated => "fixed_time_reused_each_look",
            Self::TimeUniform => "time_uniform_alpha_spending",
        }
    }

    fn should_check(self, n: usize, horizon: usize) -> bool {
        match self {
            Self::FixedFinalOnly => n == horizon,
            Self::FixedTimeRepeated | Self::TimeUniform => true,
        }
    }

    fn radius(self, n: usize) -> f64 {
        let log_term = match self {
            Self::FixedFinalOnly | Self::FixedTimeRepeated => (2.0 * CELLS / ALPHA).ln(),
            Self::TimeUniform => {
                (2.0 * CELLS * n as f64 * (n + 1) as f64 / ALPHA).ln()
            }
        };
        (log_term / (2.0 * n as f64)).sqrt()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct AlarmMetrics {
    false_alarm_probability: f64,
    detection_probability: f64,
    mean_detection_delay: Option<f64>,
    survival_probability: f64,
}

#[derive(Clone, Copy)]
struct Scenario {
    name: &'static str,
    post_shift_p: Option<f64>,
    change_after: usize,
}

const SCENARIOS: [Scenario; 2] = [
    Scenario {
        name: "stationary_null",
        post_shift_p: None,
        change_after: 0,
    },
    Scenario {
        name: "step_shift",
        post_shift_p: Some(SHIFTED_P),
        change_after: CHANGE_AFTER,
    },
];

/// Exact dynamic program over the number of successes among streams that have not alarmed.
/// Each path's probability is integrated exactly; no Monte Carlo sampling is used.
fn evaluate(
    boundary: Boundary,
    p0: f64,
    post_shift_p: Option<f64>,
    change_after: usize,
    horizon: usize,
) -> AlarmMetrics {
    let mut active = vec![0.0; horizon + 1];
    active[0] = 1.0;
    let mut first_alarm = vec![0.0; horizon + 1];

    for n in 1..=horizon {
        let probability = match post_shift_p {
            Some(shifted) if n > change_after => shifted,
            _ => p0,
        };
        let mut next = vec![0.0; horizon + 1];

        for successes in 0..n {
            let mass = active[successes];
            if mass == 0.0 {
                continue;
            }

            for (next_successes, branch_mass) in [
                (successes, mass * (1.0 - probability)),
                (successes + 1, mass * probability),
            ] {
                let should_alarm = if boundary.should_check(n, horizon) {
                    let estimate = next_successes as f64 / n as f64;
                    let radius = boundary.radius(n);
                    let lower = (estimate - radius).max(0.0);
                    let upper = (estimate + radius).min(1.0);
                    p0 < lower || p0 > upper
                } else {
                    false
                };

                if should_alarm {
                    first_alarm[n] += branch_mass;
                } else {
                    next[next_successes] += branch_mass;
                }
            }
        }
        active = next;
    }

    let mut false_alarm_probability = 0.0;
    let mut detection_probability = 0.0;
    let mut delay_mass = 0.0;
    for n in 1..=horizon {
        match post_shift_p {
            None => false_alarm_probability += first_alarm[n],
            Some(_) if n <= change_after => false_alarm_probability += first_alarm[n],
            Some(_) => {
                detection_probability += first_alarm[n];
                delay_mass += (n - change_after) as f64 * first_alarm[n];
            }
        }
    }
    if post_shift_p.is_none() {
        detection_probability = 0.0;
    }

    let mean_detection_delay = if post_shift_p.is_some() && detection_probability > 0.0 {
        Some(delay_mass / detection_probability)
    } else {
        None
    };

    AlarmMetrics {
        false_alarm_probability,
        detection_probability,
        mean_detection_delay,
        survival_probability: active.iter().sum(),
    }
}

fn csv_optional(value: Option<f64>) -> String {
    match value {
        Some(number) if number.is_finite() => format!("{number:.12}"),
        _ => "NA".to_owned(),
    }
}

fn print_csv() {
    println!(
        "boundary,scenario,horizon,change_after,p0,p1,false_alarm_probability,detection_probability,mean_detection_delay,per_cell_alpha,survival_probability"
    );
    for boundary in Boundary::ALL {
        for horizon in HORIZONS {
            for scenario in SCENARIOS {
                let metrics = evaluate(
                    boundary,
                    NOMINAL_P,
                    scenario.post_shift_p,
                    scenario.change_after,
                    horizon,
                );
                let p1 = scenario
                    .post_shift_p
                    .map(|p| format!("{p:.12}"))
                    .unwrap_or_else(|| "NA".to_owned());
                let change_after = if scenario.post_shift_p.is_some() {
                    scenario.change_after.to_string()
                } else {
                    "NA".to_owned()
                };
                println!(
                    "{},{scenario_name},{horizon},{change_after},{NOMINAL_P:.12},{p1},{:.12},{},{},{PER_CELL_ALPHA:.12},{:.12}",
                    boundary.name(),
                    metrics.false_alarm_probability,
                    csv_optional(scenario.post_shift_p.map(|_| metrics.detection_probability)),
                    csv_optional(metrics.mean_detection_delay),
                    metrics.survival_probability,
                    scenario_name = scenario.name
                );
            }
        }
    }
}

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    match args.next() {
        None => {
            print_csv();
            ExitCode::SUCCESS
        }
        Some(argument) if argument == "--help" || argument == "-h" => {
            eprintln!("Usage: cargo run --bin active-evidence-shift-monitor --");
            ExitCode::SUCCESS
        }
        Some(argument) => {
            eprintln!("unexpected argument: {argument}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_fixed_time_looks_inflate_false_alarm_probability() {
        let single_look = evaluate(
            Boundary::FixedFinalOnly,
            NOMINAL_P,
            None,
            0,
            1000,
        );
        let repeated = evaluate(
            Boundary::FixedTimeRepeated,
            NOMINAL_P,
            None,
            0,
            1000,
        );
        let anytime = evaluate(
            Boundary::TimeUniform,
            NOMINAL_P,
            None,
            0,
            1000,
        );
        assert!(single_look.false_alarm_probability <= PER_CELL_ALPHA + EPS);
        assert!(anytime.false_alarm_probability <= PER_CELL_ALPHA + EPS);
        assert!(
            repeated.false_alarm_probability > PER_CELL_ALPHA,
            "repeated fixed-time look false alarm {} did not exceed per-cell alpha {}",
            repeated.false_alarm_probability,
            PER_CELL_ALPHA
        );
    }

    #[test]
    fn anytime_monitor_detects_large_step_shift_with_bounded_null_alarm() {
        let null = evaluate(
            Boundary::TimeUniform,
            NOMINAL_P,
            None,
            0,
            500,
        );
        let shifted = evaluate(
            Boundary::TimeUniform,
            NOMINAL_P,
            Some(SHIFTED_P),
            CHANGE_AFTER,
            500,
        );
        assert!(null.false_alarm_probability <= PER_CELL_ALPHA + EPS);
        assert!(shifted.false_alarm_probability <= PER_CELL_ALPHA + EPS);
        assert!(shifted.detection_probability > 0.95);
        assert!(shifted.mean_detection_delay.is_some());
    }

    #[test]
    fn alarm_probabilities_and_survival_conserve_mass() {
        for boundary in Boundary::ALL {
            for horizon in HORIZONS {
                for scenario in SCENARIOS {
                    let metrics = evaluate(
                        boundary,
                        NOMINAL_P,
                        scenario.post_shift_p,
                        scenario.change_after,
                        horizon,
                    );
                    let total = metrics.false_alarm_probability
                        + metrics.detection_probability
                        + metrics.survival_probability;
                    assert!((total - 1.0).abs() <= 1.0e-9);
                    assert!(metrics.false_alarm_probability >= 0.0);
                    assert!(metrics.detection_probability >= 0.0);
                    assert!(metrics.survival_probability >= 0.0);
                }
            }
        }
    }

    #[test]
    fn time_uniform_radius_dominates_fixed_time_radius() {
        for n in 1..=1000 {
            assert!(Boundary::TimeUniform.radius(n) >= Boundary::FixedTimeRepeated.radius(n));
        }
    }
}
