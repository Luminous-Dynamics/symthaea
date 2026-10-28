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
const BETA_PRIOR_A: f64 = 0.5;
const BETA_PRIOR_B: f64 = 0.5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Boundary {
    FixedFinalOnly,
    FixedTimeRepeated,
    TimeUniform,
    BetaMixtureEProcess,
}

impl Boundary {
    const ALL: [Self; 4] = [
        Self::FixedFinalOnly,
        Self::FixedTimeRepeated,
        Self::TimeUniform,
        Self::BetaMixtureEProcess,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::FixedFinalOnly => "fixed_time_final_only",
            Self::FixedTimeRepeated => "fixed_time_reused_each_look",
            Self::TimeUniform => "time_uniform_alpha_spending",
            Self::BetaMixtureEProcess => "beta_mixture_e_process",
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
            Self::BetaMixtureEProcess => {
                unreachable!("Beta-mixture e-process does not use a Hoeffding radius")
            }
        };
        (log_term / (2.0 * n as f64)).sqrt()
    }

    fn should_alarm(self, n: usize, successes: usize, horizon: usize, p0: f64) -> bool {
        if !self.should_check(n, horizon) {
            return false;
        }
        if self == Self::BetaMixtureEProcess {
            return beta_mixture_log_e_value(n, successes, p0)
                >= (1.0 / PER_CELL_ALPHA).ln();
        }
        let estimate = successes as f64 / n as f64;
        let width = self.radius(n);
        let lower = (estimate - width).max(0.0);
        let upper = (estimate + width).min(1.0);
        p0 < lower || p0 > upper
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

/// Lanczos approximation for log-gamma on positive values.
fn log_gamma(value: f64) -> f64 {
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    let shifted = value - 1.0;
    let mut sum = COEFFICIENTS[0];
    for (index, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        sum += coefficient / (shifted + index as f64);
    }
    let t = shifted + 7.5;
    0.918_938_533_204_672_7 + (shifted + 0.5) * t.ln() - t + sum.ln()
}

fn log_beta(a: f64, b: f64) -> f64 {
    log_gamma(a) + log_gamma(b) - log_gamma(a + b)
}

/// Log Beta-Bernoulli mixture likelihood ratio against the simple null p0.
/// The e-value M_n has expectation one under iid Bernoulli(p0); Ville's inequality
/// therefore bounds P(sup_n M_n >= 1/delta) <= delta under the null. A Beta(1/2,1/2)
/// alternative mixture is fixed before observing data.
fn beta_mixture_log_e_value(n: usize, successes: usize, p0: f64) -> f64 {
    assert!(n > 0 && successes <= n);
    assert!(p0 > 0.0 && p0 < 1.0);
    log_beta(
        successes as f64 + BETA_PRIOR_A,
        (n - successes) as f64 + BETA_PRIOR_B,
    ) - log_beta(BETA_PRIOR_A, BETA_PRIOR_B)
        - successes as f64 * p0.ln()
        - (n - successes) as f64 * (-p0).ln_1p()
}

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

        for (successes, mass) in active.iter().copied().take(n).enumerate() {
            if mass == 0.0 {
                continue;
            }

            for (next_successes, branch_mass) in [
                (successes, mass * (1.0 - probability)),
                (successes + 1, mass * probability),
            ] {
                let should_alarm = boundary.should_alarm(n, next_successes, horizon, p0);

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
                    "{},{},{},{},{:.12},{},{:.12},{},{},{:.12},{:.12}",
                    boundary.name(),
                    scenario.name,
                    horizon,
                    change_after,
                    NOMINAL_P,
                    p1,
                    metrics.false_alarm_probability,
                    csv_optional(scenario.post_shift_p.map(|_| metrics.detection_probability)),
                    csv_optional(metrics.mean_detection_delay),
                    PER_CELL_ALPHA,
                    metrics.survival_probability
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

    #[test]
    fn beta_mixture_e_process_controls_null_alarm_and_detects_step_shift_faster() {
        let beta_null = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            None,
            0,
            1000,
        );
        assert!(beta_null.false_alarm_probability <= PER_CELL_ALPHA + EPS);

        let beta_shift = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            Some(SHIFTED_P),
            CHANGE_AFTER,
            500,
        );
        let hoeffding_shift = evaluate(
            Boundary::TimeUniform,
            NOMINAL_P,
            Some(SHIFTED_P),
            CHANGE_AFTER,
            500,
        );
        assert!(beta_shift.false_alarm_probability <= PER_CELL_ALPHA + EPS);
        assert!(beta_shift.detection_probability > 0.95);
        assert!(
            beta_shift.mean_detection_delay.unwrap()
                < hoeffding_shift.mean_detection_delay.unwrap()
        );
    }

    #[test]
    fn beta_mixture_e_value_matches_beta_binomial_formula_at_known_values() {
        // With n=2 and one success, Beta(1/2,1/2) marginal probability is 1/8;
        // under p0=1/2, the observed-sequence likelihood is 1/4, so the BF is 1/2.
        let value = beta_mixture_log_e_value(2, 1, 0.5).exp();
        assert!((value - 0.5).abs() <= 1.0e-10);
        assert!((log_beta(0.5, 0.5) - std::f64::consts::PI.ln()).abs() <= 1.0e-10);
    }
}
