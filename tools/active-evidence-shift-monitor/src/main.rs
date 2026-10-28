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
const TOTAL_LABEL_BUDGET: usize = 1000;
const SHIFT_AFTER_TOTAL_LABELS: usize = 100;
const SENTINEL_FRACTIONS: [f64; 7] = [0.0, 0.05, 0.10, 0.20, 0.25, 0.50, 1.0];
const BUDGET_SHIFT_PROBABILITIES: [f64; 5] = [0.55, 0.60, 0.70, 0.80, 0.90];
const BETA_PRIOR_A: f64 = 0.5;
const BETA_PRIOR_B: f64 = 0.5;
const FAMILY_HORIZON: usize = 1000;
const PORTFOLIO_WEIGHTS: [f64; 4] = [0.25, 0.25, 0.25, 0.25];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EProcessFamily {
    Jeffreys,
    Uniform,
    BetaFiveFive,
    BetaTwentyTwenty,
    EqualPortfolio,
}

impl EProcessFamily {
    const ALL: [Self; 5] = [
        Self::Jeffreys,
        Self::Uniform,
        Self::BetaFiveFive,
        Self::BetaTwentyTwenty,
        Self::EqualPortfolio,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Jeffreys => "beta_0.5_0.5",
            Self::Uniform => "beta_1_1",
            Self::BetaFiveFive => "beta_5_5",
            Self::BetaTwentyTwenty => "beta_20_20",
            Self::EqualPortfolio => "equal_portfolio_4_priors",
        }
    }
}

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
            Self::FixedTimeRepeated | Self::TimeUniform | Self::BetaMixtureEProcess => true,
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
    /// Bernoulli rate in the observed stream before any target-environment shift.
    /// This can differ from p0 when adaptive acquisition changes the sample mix.
    pre_shift_p: f64,
    post_shift_p: Option<f64>,
    change_after: usize,
}

const SCENARIOS: [Scenario; 12] = [
    Scenario {
        name: "stationary_null",
        pre_shift_p: NOMINAL_P,
        post_shift_p: None,
        change_after: 0,
    },
    Scenario {
        name: "step_down_010",
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.10),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_down_020",
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.20),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_down_030",
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.30),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_down_040",
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.40),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_down_045",
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.45),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_up_055",
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.55),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_up_060",
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.60),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_up_070",
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.70),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_up_080",
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(SHIFTED_P),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_up_090",
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.90),
        change_after: CHANGE_AFTER,
    },
    // Target population remains at p0=0.5, but the adaptive selector samples a stratum
    // with event rate 0.8 nine times as often as a stratum with rate 0.2: p_obs=0.74.
    // This is selection bias, not an environmental change; alarm probability is therefore
    // a spurious environment-shift alarm for this scenario.
    Scenario {
        name: "selection_bias_no_environment_shift",
        pre_shift_p: 0.74,
        post_shift_p: None,
        change_after: 0,
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


/// Log Beta-Bernoulli mixture likelihood ratio for a fixed Beta(a,b) alternative.
/// Each fixed-prior ratio is an e-process under iid Bernoulli(p0).
fn beta_mixture_log_e_value_with_prior(
    n: usize,
    successes: usize,
    p0: f64,
    a: f64,
    b: f64,
) -> f64 {
    assert!(n > 0 && successes <= n);
    assert!(p0 > 0.0 && p0 < 1.0);
    assert!(a > 0.0 && b > 0.0);
    log_beta(successes as f64 + a, (n - successes) as f64 + b)
        - log_beta(a, b)
        - successes as f64 * p0.ln()
        - (n - successes) as f64 * (-p0).ln_1p()
}

/// Log e-value for a fixed prior or a predeclared equal-weight portfolio.
///
/// A convex mixture of nonnegative e-processes is itself an e-process, so
/// the fixed four-prior portfolio remains anytime-valid without choosing a
/// prior after observing the shift.
fn family_log_e_value(
    family: EProcessFamily,
    n: usize,
    successes: usize,
    p0: f64,
) -> f64 {
    match family {
        EProcessFamily::Jeffreys => {
            beta_mixture_log_e_value_with_prior(n, successes, p0, 0.5, 0.5)
        }
        EProcessFamily::Uniform => {
            beta_mixture_log_e_value_with_prior(n, successes, p0, 1.0, 1.0)
        }
        EProcessFamily::BetaFiveFive => {
            beta_mixture_log_e_value_with_prior(n, successes, p0, 5.0, 5.0)
        }
        EProcessFamily::BetaTwentyTwenty => {
            beta_mixture_log_e_value_with_prior(n, successes, p0, 20.0, 20.0)
        }
        EProcessFamily::EqualPortfolio => {
            let component_logs = [
                beta_mixture_log_e_value_with_prior(n, successes, p0, 0.5, 0.5),
                beta_mixture_log_e_value_with_prior(n, successes, p0, 1.0, 1.0),
                beta_mixture_log_e_value_with_prior(n, successes, p0, 5.0, 5.0),
                beta_mixture_log_e_value_with_prior(n, successes, p0, 20.0, 20.0),
            ];
            let maximum = component_logs
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            maximum
                + component_logs
                    .iter()
                    .zip(PORTFOLIO_WEIGHTS)
                    .map(|(value, weight)| weight * (*value - maximum).exp())
                    .sum::<f64>()
                    .ln()
        }
    }
}

fn family_alarm_table(
    family: EProcessFamily,
    p0: f64,
    max_horizon: usize,
) -> Vec<Vec<bool>> {
    let threshold = (1.0 / PER_CELL_ALPHA).ln();
    let mut rows = vec![Vec::new(); max_horizon + 1];
    for n in 1..=max_horizon {
        rows[n] = (0..=n)
            .map(|successes| family_log_e_value(family, n, successes, p0) >= threshold)
            .collect();
    }
    rows
}

/// Exact dynamic program over the number of successes among streams that have not alarmed.
/// Each path's probability is integrated exactly; no Monte Carlo sampling is used.
fn evaluate(
    boundary: Boundary,
    p0: f64,
    pre_shift_p: f64,
    post_shift_p: Option<f64>,
    change_after: usize,
    horizon: usize,
) -> AlarmMetrics {
    evaluate_with_alarm(
        pre_shift_p,
        post_shift_p,
        change_after,
        horizon,
        |n, successes| boundary.should_alarm(n, successes, horizon, p0),
    )
}

fn evaluate_with_alarm<F>(
    pre_shift_p: f64,
    post_shift_p: Option<f64>,
    change_after: usize,
    horizon: usize,
    should_alarm: F,
) -> AlarmMetrics
where
    F: Fn(usize, usize) -> bool,
{
    let mut active = vec![0.0; horizon + 1];
    active[0] = 1.0;
    let mut first_alarm = vec![0.0; horizon + 1];

    for n in 1..=horizon {
        let probability = match post_shift_p {
            Some(shifted) if n > change_after => shifted,
            _ => pre_shift_p,
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
                if should_alarm(n, next_successes) {
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

fn print_eprocess_family_csv() {
    println!(
        "family,scenario,horizon,change_after,p0,stream_pre_p,p1,false_alarm_probability,detection_probability,mean_detection_delay,per_cell_alpha,survival_probability,log_e_threshold"
    );

    for family in EProcessFamily::ALL {
        let alarm_table = family_alarm_table(family, NOMINAL_P, FAMILY_HORIZON);
        for scenario in SCENARIOS {
            if !matches!(
                scenario.name,
                "stationary_null"
                    | "step_up_055"
                    | "step_up_060"
                    | "step_up_070"
                    | "step_up_080"
                    | "step_up_090"
                    | "selection_bias_no_environment_shift"
            ) {
                continue;
            }

            let metrics = evaluate_with_alarm(
                scenario.pre_shift_p,
                scenario.post_shift_p,
                scenario.change_after,
                FAMILY_HORIZON,
                |n, successes| alarm_table[n][successes],
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
                "{},{},{},{},{:.12},{:.12},{},{:.12},{},{},{:.12},{:.12},{:.12}",
                family.name(),
                scenario.name,
                FAMILY_HORIZON,
                change_after,
                NOMINAL_P,
                scenario.pre_shift_p,
                p1,
                metrics.false_alarm_probability,
                csv_optional(scenario.post_shift_p.map(|_| metrics.detection_probability)),
                csv_optional(metrics.mean_detection_delay),
                PER_CELL_ALPHA,
                metrics.survival_probability,
                (1.0 / PER_CELL_ALPHA).ln()
            );
        }
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
        "boundary,scenario,horizon,change_after,p0,stream_pre_p,p1,false_alarm_probability,detection_probability,mean_detection_delay,per_cell_alpha,survival_probability"
    );
    for boundary in Boundary::ALL {
        for horizon in HORIZONS {
            for scenario in SCENARIOS {
                let metrics = evaluate(
                    boundary,
                    NOMINAL_P,
                    scenario.pre_shift_p,
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
                    "{},{},{},{},{:.12},{:.12},{},{:.12},{},{},{:.12},{:.12}",
                    boundary.name(),
                    scenario.name,
                    horizon,
                    change_after,
                    NOMINAL_P,
                    scenario.pre_shift_p,
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

fn print_sentinel_budget_csv() {
    println!(
        "sentinel_fraction,post_shift_p,total_label_budget,sentinel_labels,adaptive_labels,change_after_sentinel_labels,no_target_shift_alarm_probability,pre_change_false_alarm_probability,true_shift_detection_probability,mean_delay_sentinel_labels,mean_delay_total_label_equivalent,per_cell_alpha"
    );

    for fraction in SENTINEL_FRACTIONS {
        let sentinel_labels = (fraction * TOTAL_LABEL_BUDGET as f64).round() as usize;
        let adaptive_labels = TOTAL_LABEL_BUDGET - sentinel_labels;

        if sentinel_labels == 0 {
            // With no randomized audit, the only available stream is the adaptively selected
            // stream. Its 0.74 event rate is a selection effect, not an environment shift.
            let selected_only = evaluate(
                Boundary::BetaMixtureEProcess,
                NOMINAL_P,
                0.74,
                None,
                0,
                TOTAL_LABEL_BUDGET,
            );
            println!(
                "{:.4},NA,{},{},{},NA,{:.12},NA,NA,NA,NA,{:.12}",
                fraction,
                TOTAL_LABEL_BUDGET,
                sentinel_labels,
                adaptive_labels,
                selected_only.false_alarm_probability,
                PER_CELL_ALPHA
            );
            continue;
        }

        // Use a deterministic audit cadence, so the shift-at-total-label-100 point maps to
        // a known sentinel count. The audit labels are random draws from the target population.
        let change_after_sentinel =
            (fraction * SHIFT_AFTER_TOTAL_LABELS as f64).round() as usize;
        let stable = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            NOMINAL_P,
            None,
            0,
            sentinel_labels,
        );

        for post_shift_p in BUDGET_SHIFT_PROBABILITIES {
            let shifted = evaluate(
                Boundary::BetaMixtureEProcess,
                NOMINAL_P,
                NOMINAL_P,
                Some(post_shift_p),
                change_after_sentinel,
                sentinel_labels,
            );
            let delay_total_equivalent =
                shifted.mean_detection_delay.map(|delay| delay / fraction);

            println!(
                "{:.4},{:.2},{},{},{},{},{:.12},{:.12},{},{},{},{:.12}",
                fraction,
                post_shift_p,
                TOTAL_LABEL_BUDGET,
                sentinel_labels,
                adaptive_labels,
                change_after_sentinel,
                stable.false_alarm_probability,
                shifted.false_alarm_probability,
                csv_optional(Some(shifted.detection_probability)),
                csv_optional(shifted.mean_detection_delay),
                csv_optional(delay_total_equivalent),
                PER_CELL_ALPHA
            );
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
        Some(argument) if argument == "--sentinel-budget-csv" => {
            print_sentinel_budget_csv();
            ExitCode::SUCCESS
        }
        Some(argument) if argument == "--eprocess-family-csv" => {
            print_eprocess_family_csv();
            ExitCode::SUCCESS
        }
        Some(argument) if argument == "--help" || argument == "-h" => {
            eprintln!("Usage: cargo run --bin active-evidence-shift-monitor -- [--sentinel-budget-csv|--eprocess-family-csv]");
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
            NOMINAL_P,
            None,
            0,
            1000,
        );
        let repeated = evaluate(
            Boundary::FixedTimeRepeated,
            NOMINAL_P,
            NOMINAL_P,
            None,
            0,
            1000,
        );
        let anytime = evaluate(
            Boundary::TimeUniform,
            NOMINAL_P,
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
            NOMINAL_P,
            None,
            0,
            500,
        );
        let shifted = evaluate(
            Boundary::TimeUniform,
            NOMINAL_P,
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
                        scenario.pre_shift_p,
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
            NOMINAL_P,
            None,
            0,
            1000,
        );
        assert!(beta_null.false_alarm_probability <= PER_CELL_ALPHA + EPS);

        let beta_shift = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            NOMINAL_P,
            Some(SHIFTED_P),
            CHANGE_AFTER,
            500,
        );
        let hoeffding_shift = evaluate(
            Boundary::TimeUniform,
            NOMINAL_P,
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
    fn beta_mixture_improves_power_for_a_moderate_shift_in_this_fixture() {
        let hoeffding = evaluate(
            Boundary::TimeUniform,
            NOMINAL_P,
            NOMINAL_P,
            Some(0.40),
            CHANGE_AFTER,
            1000,
        );
        let beta = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            NOMINAL_P,
            Some(0.40),
            CHANGE_AFTER,
            1000,
        );
        assert!(beta.false_alarm_probability <= PER_CELL_ALPHA + EPS);
        assert!(beta.detection_probability > 0.90);
        assert!(hoeffding.detection_probability < 0.40);
        assert!(beta.detection_probability > hoeffding.detection_probability + 0.50);
    }

    #[test]
    fn sentinel_budget_frontier_makes_audit_cost_detection_power_explicit() {
        let selected_only = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            0.74,
            None,
            0,
            TOTAL_LABEL_BUDGET,
        );
        let five_percent_stable = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            NOMINAL_P,
            None,
            0,
            50,
        );
        let ten_percent_mild_shift = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            NOMINAL_P,
            Some(0.60),
            10,
            100,
        );
        let ten_percent_moderate_shift = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            NOMINAL_P,
            Some(0.70),
            10,
            100,
        );
        let ten_percent_large_shift = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            NOMINAL_P,
            Some(SHIFTED_P),
            10,
            100,
        );
        let twenty_percent_moderate_shift = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            NOMINAL_P,
            Some(0.70),
            20,
            200,
        );
        let twenty_percent_large_shift = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            NOMINAL_P,
            Some(SHIFTED_P),
            20,
            200,
        );

        assert!(selected_only.false_alarm_probability > 0.99);
        assert!(five_percent_stable.false_alarm_probability <= PER_CELL_ALPHA + EPS);
        assert!(ten_percent_mild_shift.detection_probability < 0.05);
        assert!(ten_percent_moderate_shift.detection_probability > 0.40);
        assert!(ten_percent_moderate_shift.detection_probability < 0.50);
        assert!(ten_percent_large_shift.detection_probability > 0.95);
        assert!(twenty_percent_moderate_shift.detection_probability > 0.85);
        assert!(twenty_percent_large_shift.detection_probability > 0.99);
        assert!(ten_percent_large_shift.mean_detection_delay.is_some());
        assert!(twenty_percent_large_shift.mean_detection_delay.is_some());

        // Check every configured audit share and shift direction against the per-cell
        // anytime false-alarm budget, not only the handpicked power witnesses above.
        for fraction in SENTINEL_FRACTIONS.into_iter().filter(|fraction| *fraction > 0.0) {
            let sentinel_labels =
                (fraction * TOTAL_LABEL_BUDGET as f64).round() as usize;
            let change_after_sentinel =
                (fraction * SHIFT_AFTER_TOTAL_LABELS as f64).round() as usize;
            let stable = evaluate(
                Boundary::BetaMixtureEProcess,
                NOMINAL_P,
                NOMINAL_P,
                None,
                0,
                sentinel_labels,
            );
            assert!(
                stable.false_alarm_probability <= PER_CELL_ALPHA + EPS,
                "stable sentinel alarm {} exceeded alpha at fraction {}",
                stable.false_alarm_probability,
                fraction
            );
            for post_shift_p in BUDGET_SHIFT_PROBABILITIES {
                let shifted = evaluate(
                    Boundary::BetaMixtureEProcess,
                    NOMINAL_P,
                    NOMINAL_P,
                    Some(post_shift_p),
                    change_after_sentinel,
                    sentinel_labels,
                );
                assert!(
                    shifted.false_alarm_probability <= PER_CELL_ALPHA + EPS,
                    "pre-change false alarm {} exceeded alpha at fraction {} and p1 {}",
                    shifted.false_alarm_probability,
                    fraction,
                    post_shift_p
                );
            }
        }
    }

    #[test]
    fn eprocess_prior_families_preserve_null_budget_and_portfolio_improves_weak_shift_fixture() {
        let mut detections = Vec::new();
        for family in EProcessFamily::ALL {
            let table = family_alarm_table(family, NOMINAL_P, FAMILY_HORIZON);
            let null = evaluate_with_alarm(
                NOMINAL_P,
                None,
                0,
                FAMILY_HORIZON,
                |n, successes| table[n][successes],
            );
            assert!(
                null.false_alarm_probability <= PER_CELL_ALPHA + EPS,
                "family {} exceeds the anytime null alarm allocation",
                family.name()
            );

            let weak_shift = evaluate_with_alarm(
                NOMINAL_P,
                Some(0.55),
                CHANGE_AFTER,
                FAMILY_HORIZON,
                |n, successes| table[n][successes],
            );
            assert!(weak_shift.false_alarm_probability <= PER_CELL_ALPHA + EPS);
            let total = weak_shift.false_alarm_probability
                + weak_shift.detection_probability
                + weak_shift.survival_probability;
            assert!((total - 1.0).abs() <= 1.0e-9);
            detections.push((family, weak_shift.detection_probability));
        }

        let jeffreys = detections
            .iter()
            .find(|(family, _)| *family == EProcessFamily::Jeffreys)
            .unwrap()
            .1;
        let portfolio = detections
            .iter()
            .find(|(family, _)| *family == EProcessFamily::EqualPortfolio)
            .unwrap()
            .1;
        assert!(
            portfolio > jeffreys + 0.04,
            "fixed portfolio did not improve weak-shift power in fixture: portfolio={portfolio}, Jeffreys={jeffreys}"
        );

        let individual_log_values = [
            beta_mixture_log_e_value_with_prior(2, 1, NOMINAL_P, 0.5, 0.5),
            beta_mixture_log_e_value_with_prior(2, 1, NOMINAL_P, 1.0, 1.0),
            beta_mixture_log_e_value_with_prior(2, 1, NOMINAL_P, 5.0, 5.0),
            beta_mixture_log_e_value_with_prior(2, 1, NOMINAL_P, 20.0, 20.0),
        ];
        let expected = individual_log_values
            .iter()
            .zip(PORTFOLIO_WEIGHTS)
            .map(|(value, weight)| weight * value.exp())
            .sum::<f64>();
        assert!(
            (family_log_e_value(EProcessFamily::EqualPortfolio, 2, 1, NOMINAL_P).exp()
                - expected)
                .abs()
                <= 1.0e-10
        );
    }

    #[test]
    fn adaptive_selection_mix_can_mimic_environment_shift_without_random_sentinel() {
        let sentinel = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            NOMINAL_P,
            None,
            0,
            1000,
        );
        let selected_stream = evaluate(
            Boundary::BetaMixtureEProcess,
            NOMINAL_P,
            0.74,
            None,
            0,
            1000,
        );
        assert!(sentinel.false_alarm_probability <= PER_CELL_ALPHA + EPS);
        assert!(
            selected_stream.false_alarm_probability > 0.50,
            "selection-biased stream did not expose spurious alarms: {}",
            selected_stream.false_alarm_probability
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
