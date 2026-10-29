//! Exact finite-state benchmark for sequential model-adequacy alarms.
//!
//! This monitor tests whether a known nominal Bernoulli rate p0 remains compatible with an
//! observed calibration stream. It contrasts a valid single fixed-horizon look, a deliberately
//! invalid practice of reusing fixed-time intervals after every observation, and a time-uniform
//! alpha-spending boundary. The step-shift scenario measures power/delay, not confidence coverage
//! under nonstationarity.

use std::{env, process::ExitCode};

mod anytime_change_mixture;

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
const COMPOSITE_NULL_LOWER: f64 = 0.48;
const COMPOSITE_NULL_UPPER: f64 = 0.52;
const COMPOSITE_BETTING_FRACTIONS: [f64; 5] = [0.05, 0.1, 0.2, 0.4, 0.8];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EProcessFamily {
    Jeffreys,
    Uniform,
    BetaFiveFive,
    BetaTwentyTwenty,
    EqualPortfolio,
    CompositeBandBetting,
}

impl EProcessFamily {
    const ALL: [Self; 6] = [
        Self::Jeffreys,
        Self::Uniform,
        Self::BetaFiveFive,
        Self::BetaTwentyTwenty,
        Self::EqualPortfolio,
        Self::CompositeBandBetting,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::Jeffreys => "beta_0.5_0.5",
            Self::Uniform => "beta_1_1",
            Self::BetaFiveFive => "beta_5_5",
            Self::BetaTwentyTwenty => "beta_20_20",
            Self::EqualPortfolio => "equal_portfolio_4_priors",
            Self::CompositeBandBetting => "composite_band_betting_0.48_0.52",
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
    class: ScenarioClass,
    /// Bernoulli rate in the observed stream before any target-environment shift.
    /// This can differ from p0 when adaptive acquisition changes the sample mix.
    pre_shift_p: f64,
    post_shift_p: Option<f64>,
    change_after: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ScenarioClass {
    StationaryNull,
    TargetShift,
    SelectionBiasControl,
    BaselineMisspecification,
    WithinBandTransitionControl,
}

impl ScenarioClass {
    fn name(self) -> &'static str {
        match self {
            Self::StationaryNull => "stationary_null",
            Self::TargetShift => "target_shift",
            Self::SelectionBiasControl => "selection_bias_control",
            Self::BaselineMisspecification => "baseline_misspecification",
            Self::WithinBandTransitionControl => "within_band_transition_control",
        }
    }
}

const SCENARIOS: [Scenario; 19] = [
    Scenario {
        name: "stationary_null",
        class: ScenarioClass::StationaryNull,
        pre_shift_p: NOMINAL_P,
        post_shift_p: None,
        change_after: 0,
    },
    Scenario {
        name: "step_down_010",
        class: ScenarioClass::TargetShift,
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.10),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_down_020",
        class: ScenarioClass::TargetShift,
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.20),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_down_030",
        class: ScenarioClass::TargetShift,
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.30),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_down_040",
        class: ScenarioClass::TargetShift,
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.40),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_down_045",
        class: ScenarioClass::TargetShift,
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.45),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_up_055",
        class: ScenarioClass::TargetShift,
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.55),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_up_060",
        class: ScenarioClass::TargetShift,
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.60),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_up_070",
        class: ScenarioClass::TargetShift,
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(0.70),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_up_080",
        class: ScenarioClass::TargetShift,
        pre_shift_p: NOMINAL_P,
        post_shift_p: Some(SHIFTED_P),
        change_after: CHANGE_AFTER,
    },
    Scenario {
        name: "step_up_090",
        class: ScenarioClass::TargetShift,
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
        class: ScenarioClass::SelectionBiasControl,
        pre_shift_p: 0.74,
        post_shift_p: None,
        change_after: 0,
    },
    // Stable streams with a true rate different from the model's fixed nominal p0=0.5.
    // These are assumption-sensitivity controls: an alarm is expected if the baseline
    // rate is misspecified, and does not by itself establish a later environment change.
    Scenario {
        name: "stable_baseline_rate_045",
        class: ScenarioClass::BaselineMisspecification,
        pre_shift_p: 0.45,
        post_shift_p: None,
        change_after: 0,
    },
    Scenario {
        name: "stable_baseline_rate_048",
        class: ScenarioClass::BaselineMisspecification,
        pre_shift_p: 0.48,
        post_shift_p: None,
        change_after: 0,
    },
    Scenario {
        name: "stable_baseline_rate_049",
        class: ScenarioClass::BaselineMisspecification,
        pre_shift_p: 0.49,
        post_shift_p: None,
        change_after: 0,
    },
    Scenario {
        name: "stable_baseline_rate_051",
        class: ScenarioClass::BaselineMisspecification,
        pre_shift_p: 0.51,
        post_shift_p: None,
        change_after: 0,
    },
    Scenario {
        name: "stable_baseline_rate_052",
        class: ScenarioClass::BaselineMisspecification,
        pre_shift_p: 0.52,
        post_shift_p: None,
        change_after: 0,
    },
    Scenario {
        name: "stable_baseline_rate_055",
        class: ScenarioClass::BaselineMisspecification,
        pre_shift_p: 0.55,
        post_shift_p: None,
        change_after: 0,
    },
    // The process changes from p=.49 to p=.51, but both conditional rates remain inside
    // the composite null band [.48,.52]. A valid composite-null monitor should control the
    // total alarm probability across this predictable transition.
    Scenario {
        name: "within_band_transition_049_to_051",
        class: ScenarioClass::WithinBandTransitionControl,
        pre_shift_p: 0.49,
        post_shift_p: Some(0.51),
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


/// Composite-null e-process for Bernoulli conditional probabilities inside [lower, upper].
///
/// For each fixed lambda, the upper factor 1 + lambda * (X - upper) has conditional
/// expectation <= 1 whenever p <= upper; the lower factor 1 - lambda * (X - lower)
/// has conditional expectation <= 1 whenever p >= lower. Every factor is nonnegative
/// for the configured lambda grid. Mixing the five stake sizes in each direction, then
/// mixing directions, preserves the supermartingale guarantee for any conditional p_t
/// in the declared interval. The bounds must be preregistered, not fitted on the test stream.
fn composite_band_log_e_value(
    n: usize,
    successes: usize,
    lower: f64,
    upper: f64,
) -> f64 {
    assert!(n > 0 && successes <= n);
    assert!(0.0 <= lower && lower < upper && upper <= 1.0);

    let failures = (n - successes) as f64;
    let successes = successes as f64;
    let upper_logs: [f64; 5] = std::array::from_fn(|index| {
        let lambda = COMPOSITE_BETTING_FRACTIONS[index];
        successes * (lambda * (1.0 - upper)).ln_1p()
            + failures * (-lambda * upper).ln_1p()
    });
    let lower_logs: [f64; 5] = std::array::from_fn(|index| {
        let lambda = COMPOSITE_BETTING_FRACTIONS[index];
        successes * (-lambda * (1.0 - lower)).ln_1p()
            + failures * (lambda * lower).ln_1p()
    });

    let log_mean_exp = |values: &[f64; 5]| {
        let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        maximum
            + values
                .iter()
                .map(|value| (*value - maximum).exp())
                .sum::<f64>()
                .ln()
            - (values.len() as f64).ln()
    };
    let upper_log_e = log_mean_exp(&upper_logs);
    let lower_log_e = log_mean_exp(&lower_logs);
    let maximum = upper_log_e.max(lower_log_e);
    maximum + (0.5 * (upper_log_e - maximum).exp()
        + 0.5 * (lower_log_e - maximum).exp())
        .ln()
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
        EProcessFamily::CompositeBandBetting => composite_band_log_e_value(
            n,
            successes,
            COMPOSITE_NULL_LOWER,
            COMPOSITE_NULL_UPPER,
        ),
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

/// Exact DP for a stationary-null classification when the conditional event probability
/// is a deterministic/predictable schedule rather than a single IID Bernoulli rate.
fn evaluate_null_schedule<P, A>(
    horizon: usize,
    probability_at: P,
    should_alarm: A,
) -> AlarmMetrics
where
    P: Fn(usize) -> f64,
    A: Fn(usize, usize) -> bool,
{
    let mut active = vec![0.0; horizon + 1];
    active[0] = 1.0;
    let mut first_alarm = vec![0.0; horizon + 1];

    for n in 1..=horizon {
        let probability = probability_at(n);
        assert!((0.0..=1.0).contains(&probability));
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

    AlarmMetrics {
        false_alarm_probability: first_alarm.iter().sum(),
        detection_probability: 0.0,
        mean_detection_delay: None,
        survival_probability: active.iter().sum(),
    }
}


const CUSUM_SCALE: usize = 25;
const CUSUM_UPPER_INCREMENT: usize = 12;
const CUSUM_LOWER_INCREMENT: usize = 13;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CusumDirection {
    Upper,
    Lower,
}

impl CusumDirection {
    const ALL: [Self; 2] = [Self::Upper, Self::Lower];

    fn name(self) -> &'static str {
        match self {
            Self::Upper => "upper_cusum",
            Self::Lower => "lower_cusum",
        }
    }

    /// Probability that this directional statistic sees an evidence event.
    /// For a lower shift, a Bernoulli failure is the event.
    fn event_probability(self, observed_probability: f64) -> f64 {
        match self {
            Self::Upper => observed_probability,
            Self::Lower => 1.0 - observed_probability,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct CusumMetrics {
    scenario_or_pre_shift_alarm_probability: f64,
    post_shift_detection_probability: Option<f64>,
    mean_post_shift_detection_delay: Option<f64>,
    median_post_shift_detection_delay: Option<f64>,
    p90_post_shift_detection_delay: Option<f64>,
    survival_probability: f64,
}

/// Reflected one-sided Bernoulli CUSUM on an exact 1/25 score grid.
/// An evidence event adds 12/25; a non-event subtracts 13/25. The event-rate
/// boundary 0.52 is therefore exactly zero-drift for the calibration process.
fn cusum_next_state(state: usize, event: bool) -> usize {
    if event {
        state + CUSUM_UPPER_INCREMENT
    } else {
        state.saturating_sub(CUSUM_LOWER_INCREMENT)
    }
}

/// Exact finite-horizon alarm probability for a one-sided CUSUM and a constant
/// event probability. This is a finite-horizon calibration, not an infinite-horizon
/// or Ville-style guarantee.
fn cusum_alarm_probability(horizon: usize, threshold: usize, event_probability: f64) -> f64 {
    if threshold == 0 {
        return 1.0;
    }
    let mut active = vec![0.0_f64; threshold];
    active[0] = 1.0;
    let mut alarm_probability = 0.0;

    for n in 1..=horizon {
        let mut next = vec![0.0_f64; threshold];
        let reachable_states = threshold.min((n - 1) * CUSUM_UPPER_INCREMENT + 1);
        for (state, &mass) in active.iter().take(reachable_states).enumerate() {
            if mass == 0.0 {
                continue;
            }

            let event_mass = mass * event_probability;
            let event_state = cusum_next_state(state, true);
            if event_state >= threshold {
                alarm_probability += event_mass;
            } else {
                next[event_state] += event_mass;
            }

            let non_event_mass = mass * (1.0 - event_probability);
            let non_event_state = cusum_next_state(state, false);
            if non_event_state >= threshold {
                alarm_probability += non_event_mass;
            } else {
                next[non_event_state] += non_event_mass;
            }
        }
        active = next;
    }
    alarm_probability
}

/// Find the smallest integer threshold whose exact boundary alarm probability
/// is no greater than the requested false-alarm operating point.
fn calibrate_cusum_threshold(horizon: usize, alpha: f64) -> usize {
    let mut low = 1;
    let mut high = horizon * CUSUM_UPPER_INCREMENT + 1;
    while low < high {
        let middle = low + (high - low) / 2;
        let alarm = cusum_alarm_probability(horizon, middle, COMPOSITE_NULL_UPPER);
        if alarm <= alpha {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    low
}

fn probability_range_sum(values: &[f64], start: usize, end: usize) -> f64 {
    if start > end || start >= values.len() {
        return 0.0;
    }
    values[start..=end.min(values.len() - 1)].iter().sum()
}

fn conditional_detection_delay_quantile(
    first_alarm: &[f64],
    change_after: usize,
    horizon: usize,
    detection_probability: f64,
    quantile: f64,
) -> Option<f64> {
    if detection_probability <= 0.0 {
        return None;
    }
    let target_mass = detection_probability * quantile;
    let mut cumulative_mass = 0.0;
    for n in change_after.saturating_add(1)..=horizon {
        cumulative_mass += first_alarm[n];
        if cumulative_mass >= target_mass {
            return Some((n - change_after) as f64);
        }
    }
    Some(horizon.saturating_sub(change_after) as f64)
}

#[cfg(test)]
fn evaluate_cusum_null_schedule<P>(
    direction: CusumDirection,
    threshold: usize,
    horizon: usize,
    conditional_p_at: P,
) -> CusumMetrics
where
    P: Fn(usize, usize) -> f64,
{
    let mut active = vec![0.0_f64; threshold];
    active[0] = 1.0;
    let mut alarm_probability = 0.0;

    for n in 1..=horizon {
        let mut next = vec![0.0_f64; threshold];
        let reachable_states = threshold.min((n - 1) * CUSUM_UPPER_INCREMENT + 1);
        for (state, &mass) in active.iter().take(reachable_states).enumerate() {
            if mass == 0.0 {
                continue;
            }
            let conditional_p = conditional_p_at(n, state);
            assert!(
                (COMPOSITE_NULL_LOWER..=COMPOSITE_NULL_UPPER).contains(&conditional_p),
                "predictable CUSUM null schedule escaped the accepted rate band: p={conditional_p}"
            );
            let event_probability = direction.event_probability(conditional_p);
            for (event, branch_mass) in [
                (true, mass * event_probability),
                (false, mass * (1.0 - event_probability)),
            ] {
                let next_state = cusum_next_state(state, event);
                if next_state >= threshold {
                    alarm_probability += branch_mass;
                } else {
                    next[next_state] += branch_mass;
                }
            }
        }
        active = next;
    }

    CusumMetrics {
        scenario_or_pre_shift_alarm_probability: alarm_probability,
        post_shift_detection_probability: None,
        mean_post_shift_detection_delay: None,
        median_post_shift_detection_delay: None,
        p90_post_shift_detection_delay: None,
        survival_probability: active.iter().sum(),
    }
}

/// Run the exact CUSUM dynamic program once to the largest requested horizon and
/// return typed metrics at each checkpoint.
fn evaluate_cusum_scenario(
    direction: CusumDirection,
    threshold: usize,
    scenario: Scenario,
) -> Vec<(usize, CusumMetrics)> {
    let max_horizon = *HORIZONS.last().expect("horizons must be nonempty");
    let mut active = vec![0.0_f64; threshold];
    active[0] = 1.0;
    let mut first_alarm = vec![0.0_f64; max_horizon + 1];
    let mut snapshots = Vec::with_capacity(HORIZONS.len());
    let mut checkpoint = 0;

    for n in 1..=max_horizon {
        let observed_probability = match scenario.post_shift_p {
            Some(p1) if n > scenario.change_after => p1,
            _ => scenario.pre_shift_p,
        };
        let event_probability = direction.event_probability(observed_probability);
        let mut next = vec![0.0_f64; threshold];
        let reachable_states = threshold.min((n - 1) * CUSUM_UPPER_INCREMENT + 1);
        for (state, &mass) in active.iter().take(reachable_states).enumerate() {
            if mass == 0.0 {
                continue;
            }
            for (event, branch_mass) in [
                (true, mass * event_probability),
                (false, mass * (1.0 - event_probability)),
            ] {
                let next_state = cusum_next_state(state, event);
                if next_state >= threshold {
                    first_alarm[n] += branch_mass;
                } else {
                    next[next_state] += branch_mass;
                }
            }
        }
        active = next;

        if checkpoint < HORIZONS.len() && n == HORIZONS[checkpoint] {
            let survival = active.iter().sum::<f64>();
            snapshots.push((n, survival));
            checkpoint += 1;
        }
    }

    snapshots
        .into_iter()
        .map(|(horizon, survival)| {
            let is_target_shift = scenario.class == ScenarioClass::TargetShift;
            let alarm_probability = if is_target_shift {
                probability_range_sum(
                    &first_alarm,
                    1,
                    scenario.change_after.min(horizon),
                )
            } else {
                probability_range_sum(&first_alarm, 1, horizon)
            };
            let detection_probability = if is_target_shift {
                Some(probability_range_sum(
                    &first_alarm,
                    scenario.change_after.saturating_add(1),
                    horizon,
                ))
            } else {
                None
            };
            let delay = detection_probability.and_then(|detection| {
                if detection <= 0.0 {
                    None
                } else {
                    let delay_mass = (scenario.change_after.saturating_add(1)..=horizon)
                        .map(|n| (n - scenario.change_after) as f64 * first_alarm[n])
                        .sum::<f64>();
                    Some(delay_mass / detection)
                }
            });
            let median_delay = detection_probability.and_then(|detection| {
                conditional_detection_delay_quantile(
                    &first_alarm,
                    scenario.change_after,
                    horizon,
                    detection,
                    0.50,
                )
            });
            let p90_delay = detection_probability.and_then(|detection| {
                conditional_detection_delay_quantile(
                    &first_alarm,
                    scenario.change_after,
                    horizon,
                    detection,
                    0.90,
                )
            });
            (
                horizon,
                CusumMetrics {
                    scenario_or_pre_shift_alarm_probability: alarm_probability,
                    post_shift_detection_probability: detection_probability,
                    mean_post_shift_detection_delay: delay,
                    median_post_shift_detection_delay: median_delay,
                    p90_post_shift_detection_delay: p90_delay,
                    survival_probability: survival,
                },
            )
        })
        .collect()
}

fn composite_boundary_reference_alarm_probability(horizon: usize) -> f64 {
    let table = family_alarm_table(EProcessFamily::CompositeBandBetting, NOMINAL_P, horizon);
    evaluate_with_alarm(
        COMPOSITE_NULL_UPPER,
        None,
        0,
        horizon,
        |n, successes| table[n][successes],
    )
    .false_alarm_probability
}

fn print_cusum_baseline_csv() {
    let horizon = HORIZONS[HORIZONS.len() - 1];
    let matched_reference_family_alarm_probability = composite_boundary_reference_alarm_probability(horizon);
    let matched_direction_alarm_probability = matched_reference_family_alarm_probability / 2.0;
    let threshold = calibrate_cusum_threshold(horizon, matched_direction_alarm_probability);
    let calibration_boundary_alarm_probability =
        cusum_alarm_probability(horizon, threshold, COMPOSITE_NULL_UPPER);
    let family_worst_case_union_bound = 2.0 * calibration_boundary_alarm_probability;
    println!(
        "direction,scenario,scenario_class,horizon,change_after,p0,stream_pre_p,p1,null_lower,null_upper,threshold_scaled,threshold_score,matched_reference_family_alarm_probability,matched_direction_alarm_probability,calibration_boundary_alarm_probability,family_worst_case_union_bound,scenario_or_pre_shift_alarm_probability,post_shift_detection_probability,mean_post_shift_detection_delay,median_post_shift_detection_delay,p90_post_shift_detection_delay,per_cell_alpha,survival_probability"
    );
    for direction in CusumDirection::ALL {
        for scenario in SCENARIOS {
            for (horizon, metrics) in evaluate_cusum_scenario(direction, threshold, scenario) {
                let change_after = if scenario.post_shift_p.is_some() {
                    scenario.change_after.to_string()
                } else {
                    "NA".to_owned()
                };
                let p1 = scenario
                    .post_shift_p
                    .map(|p| format!("{p:.12}"))
                    .unwrap_or_else(|| "NA".to_owned());
                println!(
                    "{},{},{},{},{},{:.12},{:.12},{},{:.12},{:.12},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{},{},{},{},{:.12},{:.12}",
                    direction.name(),
                    scenario.name,
                    scenario.class.name(),
                    horizon,
                    change_after,
                    NOMINAL_P,
                    scenario.pre_shift_p,
                    p1,
                    COMPOSITE_NULL_LOWER,
                    COMPOSITE_NULL_UPPER,
                    threshold,
                    threshold as f64 / CUSUM_SCALE as f64,
                    matched_reference_family_alarm_probability,
                    matched_direction_alarm_probability,
                    calibration_boundary_alarm_probability,
                    family_worst_case_union_bound,
                    metrics.scenario_or_pre_shift_alarm_probability,
                    csv_optional_precise(metrics.post_shift_detection_probability),
                    csv_optional_precise(metrics.mean_post_shift_detection_delay),
                    csv_optional_precise(metrics.median_post_shift_detection_delay),
                    csv_optional_precise(metrics.p90_post_shift_detection_delay),
                    PER_CELL_ALPHA,
                    metrics.survival_probability
                );
            }
        }
    }
}

fn print_eprocess_family_csv() {
    println!(
        "family,scenario,scenario_class,horizon,change_after,p0,stream_pre_p,p1,scenario_or_pre_shift_alarm_probability,post_shift_detection_probability,mean_post_shift_detection_delay,per_cell_alpha,survival_probability,log_e_threshold"
    );

    for family in EProcessFamily::ALL {
        let alarm_table = family_alarm_table(family, NOMINAL_P, FAMILY_HORIZON);
        for scenario in SCENARIOS {
            if !matches!(
                scenario.name,
                "stationary_null"
                    | "step_down_010"
                    | "step_down_020"
                    | "step_down_030"
                    | "step_down_040"
                    | "step_down_045"
                    | "step_up_055"
                    | "step_up_060"
                    | "step_up_070"
                    | "step_up_080"
                    | "step_up_090"
                    | "selection_bias_no_environment_shift"
                    | "stable_baseline_rate_045"
                    | "stable_baseline_rate_048"
                    | "stable_baseline_rate_049"
                    | "stable_baseline_rate_051"
                    | "stable_baseline_rate_052"
                    | "stable_baseline_rate_055"
                    | "within_band_transition_049_to_051"
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
            let scenario_or_pre_shift_alarm = if scenario.class == ScenarioClass::WithinBandTransitionControl {
                metrics.false_alarm_probability + metrics.detection_probability
            } else {
                metrics.false_alarm_probability
            };
            let target_shift = scenario.class == ScenarioClass::TargetShift;
            let detection = target_shift.then_some(metrics.detection_probability);
            let delay = if target_shift { metrics.mean_detection_delay } else { None };
            println!(
                "{},{},{},{},{},{:.12},{:.12},{},{:.12},{},{},{:.12},{:.12},{:.12}",
                family.name(),
                scenario.name,
                scenario.class.name(),
                FAMILY_HORIZON,
                change_after,
                NOMINAL_P,
                scenario.pre_shift_p,
                p1,
                scenario_or_pre_shift_alarm,
                csv_optional(detection),
                csv_optional(delay),
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

fn csv_optional_precise(value: Option<f64>) -> String {
    match value {
        Some(number) if number.is_finite() => format!("{number:.12e}"),
        _ => "NA".to_owned(),
    }
}

fn print_csv() {
    println!(
        "boundary,scenario,scenario_class,horizon,change_after,p0,stream_pre_p,p1,scenario_or_pre_shift_alarm_probability,post_shift_detection_probability,mean_post_shift_detection_delay,per_cell_alpha,survival_probability"
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
                let scenario_or_pre_shift_alarm = if scenario.class == ScenarioClass::WithinBandTransitionControl {
                    metrics.false_alarm_probability + metrics.detection_probability
                } else {
                    metrics.false_alarm_probability
                };
                let target_shift = scenario.class == ScenarioClass::TargetShift;
                let detection = target_shift.then_some(metrics.detection_probability);
                let delay = if target_shift { metrics.mean_detection_delay } else { None };
                println!(
                    "{},{},{},{},{},{:.12},{:.12},{},{:.12},{},{},{:.12},{:.12}",
                    boundary.name(),
                    scenario.name,
                    scenario.class.name(),
                    horizon,
                    change_after,
                    NOMINAL_P,
                    scenario.pre_shift_p,
                    p1,
                    scenario_or_pre_shift_alarm,
                    csv_optional(detection),
                    csv_optional(delay),
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

#[derive(Clone, Copy)]
struct AnytimeMixtureProbe {
    name: &'static str,
    class: &'static str,
    pre_p: f64,
    post_p: Option<f64>,
    change_after: usize,
}

const ANYTIME_MIXTURE_PROBES: [AnytimeMixtureProbe; 14] = [
    AnytimeMixtureProbe {
        name: "stationary_p050",
        class: "stationary_null",
        pre_p: 0.50,
        post_p: None,
        change_after: 0,
    },
    AnytimeMixtureProbe {
        name: "stable_edge_p048",
        class: "stationary_null",
        pre_p: 0.48,
        post_p: None,
        change_after: 0,
    },
    AnytimeMixtureProbe {
        name: "stable_edge_p052",
        class: "stationary_null",
        pre_p: 0.52,
        post_p: None,
        change_after: 0,
    },
    AnytimeMixtureProbe {
        name: "selection_bias_p074",
        class: "selection_bias_control",
        pre_p: 0.74,
        post_p: None,
        change_after: 0,
    },
    AnytimeMixtureProbe {
        name: "within_band_alternating",
        class: "within_band_transition_control",
        pre_p: 0.50,
        post_p: None,
        change_after: 0,
    },
    AnytimeMixtureProbe {
        name: "within_band_feedback",
        class: "within_band_transition_control",
        pre_p: 0.50,
        post_p: None,
        change_after: 0,
    },
    AnytimeMixtureProbe {
        name: "step_up_055",
        class: "target_shift",
        pre_p: 0.50,
        post_p: Some(0.55),
        change_after: 100,
    },
    AnytimeMixtureProbe {
        name: "step_up_060",
        class: "target_shift",
        pre_p: 0.50,
        post_p: Some(0.60),
        change_after: 100,
    },
    AnytimeMixtureProbe {
        name: "step_up_080",
        class: "target_shift",
        pre_p: 0.50,
        post_p: Some(0.80),
        change_after: 100,
    },
    AnytimeMixtureProbe {
        name: "step_down_045",
        class: "target_shift",
        pre_p: 0.50,
        post_p: Some(0.45),
        change_after: 100,
    },
    AnytimeMixtureProbe {
        name: "step_down_040",
        class: "target_shift",
        pre_p: 0.50,
        post_p: Some(0.40),
        change_after: 100,
    },
    AnytimeMixtureProbe {
        name: "step_down_020",
        class: "target_shift",
        pre_p: 0.50,
        post_p: Some(0.20),
        change_after: 100,
    },
    AnytimeMixtureProbe {
        name: "all_ones_adversarial",
        class: "extreme_path_control",
        pre_p: 1.0,
        post_p: None,
        change_after: 0,
    },
    AnytimeMixtureProbe {
        name: "all_zeros_adversarial",
        class: "extreme_path_control",
        pre_p: 0.0,
        post_p: None,
        change_after: 0,
    },
];

fn print_anytime_mixture_probe_csv() {
    use anytime_change_mixture::{
        probe_bernoulli, AnytimeChangeMixture, CHANGE_HAZARDS, DOWNWARD_ALTERNATIVES,
        UPWARD_ALTERNATIVES,
    };

    let horizon = FAMILY_HORIZON;
    let alpha = PER_CELL_ALPHA;
    let log_threshold = -alpha.ln();
    println!(
        "trace_id,scenario,scenario_class,horizon,change_after,schedule_policy,pre_rate,post_rate,alpha,log_threshold,component_count,first_alarm_step,alarm_phase,final_log_e_value,upward_alternatives,downward_alternatives,change_hazards"
    );

    for (index, scenario) in ANYTIME_MIXTURE_PROBES.iter().enumerate() {
        let seed = 0xA11C_E5EED_u64
            ^ (index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut monitor = AnytimeChangeMixture::new();
        let mut first_alarm = None;

        for step in 1..=horizon {
            let conditional_p = if scenario.name == "all_ones_adversarial" {
                1.0
            } else if scenario.name == "all_zeros_adversarial" {
                0.0
            } else if scenario.name == "within_band_alternating" {
                if step % 2 == 0 {
                    anytime_change_mixture::NULL_UPPER
                } else {
                    anytime_change_mixture::NULL_LOWER
                }
            } else if scenario.name == "within_band_feedback" {
                if monitor.log_e_value() < log_threshold / 2.0 {
                    anytime_change_mixture::NULL_UPPER
                } else {
                    anytime_change_mixture::NULL_LOWER
                }
            } else if scenario.name.starts_with("step_") && step > scenario.change_after {
                scenario.post_p.unwrap()
            } else {
                scenario.pre_p
            };
            let event = probe_bernoulli(seed, step as u64, conditional_p);
            monitor.observe(event);
            if first_alarm.is_none() && monitor.crossed(alpha) {
                first_alarm = Some(step);
            }
        }

        let alarm_step_text = first_alarm
            .map(|step| step.to_string())
            .unwrap_or_else(|| "NA".to_owned());
        let schedule_policy = if scenario.name == "within_band_alternating" {
            "alternating_band_endpoints"
        } else if scenario.name == "within_band_feedback" {
            "previsible_e_value_feedback"
        } else if scenario.name.starts_with("step_") {
            "single_step_after_100"
        } else if scenario.name == "all_ones_adversarial" {
            "all_ones_extreme_path"
        } else if scenario.name == "all_zeros_adversarial" {
            "all_zeros_extreme_path"
        } else {
            "constant_probability"
        };
        let pre_rate_text = if scenario.name == "within_band_alternating"
            || scenario.name == "within_band_feedback"
        {
            "NA".to_owned()
        } else {
            format!("{:.12}", scenario.pre_p)
        };
        let alarm_phase = match (scenario.post_p, first_alarm) {
            (Some(_), Some(step)) if step <= scenario.change_after => "pre_change",
            (Some(_), Some(_)) => "post_change",
            (Some(_), None) => "not_detected",
            (None, Some(_)) => "alarm_on_control",
            (None, None) => "no_alarm",
        };
        let post_p_text = scenario
            .post_p
            .map(|p| format!("{p:.12}"))
            .unwrap_or_else(|| "NA".to_owned());
        let up_text = UPWARD_ALTERNATIVES
            .iter()
            .map(|p| format!("{p:.2}"))
            .collect::<Vec<_>>()
            .join(";");
        let down_text = DOWNWARD_ALTERNATIVES
            .iter()
            .map(|p| format!("{p:.2}"))
            .collect::<Vec<_>>()
            .join(";");
        let hazard_text = CHANGE_HAZARDS
            .iter()
            .map(|h| format!("{h:.4}"))
            .collect::<Vec<_>>()
            .join(";");
        println!(
            "{},{},{},{},{},{},{},{},{:.12},{:.12},{},{},{},{:.12e},{},{},{}",
            index + 1,
            scenario.name,
            scenario.class,
            monitor.steps(),
            if scenario.post_p.is_some() {
                scenario.change_after.to_string()
            } else {
                "NA".to_owned()
            },
            schedule_policy,
            pre_rate_text,
            post_p_text,
            alpha,
            log_threshold,
            monitor.component_count(),
            alarm_step_text,
            alarm_phase,
            monitor.log_e_value(),
            up_text,
            down_text,
            hazard_text
        );
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
        Some(argument) if argument == "--cusum-baseline-csv" => {
            print_cusum_baseline_csv();
            ExitCode::SUCCESS
        }
        Some(argument) if argument == "--anytime-mixture-probes-csv" => {
            print_anytime_mixture_probe_csv();
            ExitCode::SUCCESS
        }
        Some(argument) if argument == "--help" || argument == "-h" => {
            eprintln!("Usage: cargo run --bin active-evidence-shift-monitor -- [--sentinel-budget-csv|--eprocess-family-csv|--cusum-baseline-csv|--anytime-mixture-probes-csv]");
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
            let misspecified_baseline = evaluate_with_alarm(
                0.48,
                None,
                0,
                FAMILY_HORIZON,
                |n, successes| table[n][successes],
            );
            if family == EProcessFamily::CompositeBandBetting {
                assert!(
                    misspecified_baseline.false_alarm_probability <= PER_CELL_ALPHA + EPS,
                    "composite-null family exceeded alpha inside lower null boundary"
                );
                let upper_boundary = evaluate_with_alarm(
                    0.52,
                    None,
                    0,
                    FAMILY_HORIZON,
                    |n, successes| table[n][successes],
                );
                assert!(
                    upper_boundary.false_alarm_probability <= PER_CELL_ALPHA + EPS,
                    "composite-null family exceeded alpha inside upper null boundary"
                );
                let outside_band = evaluate_with_alarm(
                    0.45,
                    None,
                    0,
                    FAMILY_HORIZON,
                    |n, successes| table[n][successes],
                );
                assert!(
                    outside_band.false_alarm_probability > 0.04,
                    "composite-null family failed the outside-band control"
                );
            } else {
                assert!(
                    misspecified_baseline.false_alarm_probability > PER_CELL_ALPHA,
                    "family {} hid the declared-p0 misspecification control",
                    family.name()
                );
            }

            if family == EProcessFamily::CompositeBandBetting {
                let within_band_transition = evaluate_with_alarm(
                    0.49,
                    Some(0.51),
                    CHANGE_AFTER,
                    FAMILY_HORIZON,
                    |n, successes| table[n][successes],
                );
                let transition_alarm_probability =
                    within_band_transition.false_alarm_probability
                        + within_band_transition.detection_probability;
                assert!(
                    transition_alarm_probability <= PER_CELL_ALPHA + EPS,
                    "composite-null family exceeded alpha for a transition within the accepted band"
                );
                assert!(
                    (transition_alarm_probability
                        + within_band_transition.survival_probability
                        - 1.0)
                        .abs()
                        <= 1.0e-9,
                    "composite-null transition probability mass did not conserve"
                );
            }

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

        let boundary_table = family_alarm_table(
            EProcessFamily::CompositeBandBetting,
            NOMINAL_P,
            FAMILY_HORIZON,
        );
        let composite_shift_060 = evaluate_with_alarm(
            NOMINAL_P,
            Some(0.60),
            CHANGE_AFTER,
            FAMILY_HORIZON,
            |n, successes| boundary_table[n][successes],
        );
        let composite_shift_080 = evaluate_with_alarm(
            NOMINAL_P,
            Some(0.80),
            CHANGE_AFTER,
            FAMILY_HORIZON,
            |n, successes| boundary_table[n][successes],
        );
        assert!(composite_shift_060.detection_probability > 0.70);
        assert!(composite_shift_080.detection_probability > 0.99);

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
    fn composite_null_controls_predictable_alternation_between_band_edges() {
        let table = family_alarm_table(
            EProcessFamily::CompositeBandBetting,
            NOMINAL_P,
            FAMILY_HORIZON,
        );
        let metrics = evaluate_null_schedule(
            FAMILY_HORIZON,
            |n| if n % 2 == 0 { 0.52 } else { 0.48 },
            |n, successes| table[n][successes],
        );
        assert!(
            metrics.false_alarm_probability <= PER_CELL_ALPHA + EPS,
            "composite-null e-process exceeded alpha for alternating in-band probabilities"
        );
        assert!(metrics.false_alarm_probability > 0.00015);
        assert!(metrics.false_alarm_probability < 0.00030);
        assert!(
            (metrics.false_alarm_probability + metrics.survival_probability - 1.0).abs()
                <= 1.0e-9
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
    #[test]
    fn cusum_threshold_is_exactly_calibrated_at_the_composite_null_edge() {
        let horizon = HORIZONS[HORIZONS.len() - 1];
        let matched_reference_family_alarm_probability =
            composite_boundary_reference_alarm_probability(horizon);
        assert!(
            (matched_reference_family_alarm_probability - 0.0008915879337992225).abs() <= 5.0e-10,
            "composite reference edge FPR changed; update the preregistered calibration target"
        );
        let matched_direction_alarm_probability = matched_reference_family_alarm_probability / 2.0;
        let threshold = calibrate_cusum_threshold(horizon, matched_direction_alarm_probability);
        assert_eq!(threshold, 1444, "family-matched CUSUM threshold changed; review benchmark protocol");
        let upper_alarm = cusum_alarm_probability(horizon, threshold, COMPOSITE_NULL_UPPER);
        let lower_alarm =
            cusum_alarm_probability(horizon, threshold, 1.0 - COMPOSITE_NULL_LOWER);
        assert!(upper_alarm <= matched_direction_alarm_probability + EPS);
        assert!(lower_alarm <= matched_direction_alarm_probability + EPS);
        assert!((upper_alarm - lower_alarm).abs() <= EPS);
        assert!(
            2.0 * upper_alarm <= matched_reference_family_alarm_probability + EPS,
            "two-direction CUSUM union bound exceeds matched reference family alarm rate"
        );
        assert!(
            2.0 * upper_alarm <= PER_CELL_ALPHA + EPS,
            "two-direction CUSUM union bound exceeds the per-cell alpha allocation"
        );
        if threshold > 1 {
            let previous =
                cusum_alarm_probability(horizon, threshold - 1, COMPOSITE_NULL_UPPER);
            assert!(
                previous > matched_direction_alarm_probability + EPS,
                "calibration did not find the minimum threshold: previous={previous}, target={matched_direction_alarm_probability}"
            );
        }
    }

    #[test]
    fn cusum_scenario_metrics_preserve_semantics_and_probability_mass() {
        let horizon = HORIZONS[HORIZONS.len() - 1];
        let matched_reference_family_alarm_probability =
            composite_boundary_reference_alarm_probability(horizon);
        let matched_direction_alarm_probability = matched_reference_family_alarm_probability / 2.0;
        let threshold = calibrate_cusum_threshold(horizon, matched_direction_alarm_probability);
        for direction in CusumDirection::ALL {
            for scenario in SCENARIOS {
                for (horizon, metrics) in evaluate_cusum_scenario(direction, threshold, scenario) {
                    let total = metrics.scenario_or_pre_shift_alarm_probability
                        + metrics.post_shift_detection_probability.unwrap_or(0.0)
                        + metrics.survival_probability;
                    assert!((total - 1.0).abs() <= 1.0e-9);
                    if scenario.class == ScenarioClass::TargetShift {
                        let detection = metrics.post_shift_detection_probability.unwrap();
                        if detection > 0.0 {
                            let median = metrics.median_post_shift_detection_delay.unwrap();
                            let p90 = metrics.p90_post_shift_detection_delay.unwrap();
                            assert!(metrics.mean_post_shift_detection_delay.is_some());
                            assert!(median >= 1.0 && median <= p90);
                            assert!(p90 <= horizon.saturating_sub(scenario.change_after) as f64);
                        } else {
                            assert!(metrics.mean_post_shift_detection_delay.is_none());
                            assert!(metrics.median_post_shift_detection_delay.is_none());
                            assert!(metrics.p90_post_shift_detection_delay.is_none());
                        }
                    } else {
                        assert!(metrics.post_shift_detection_probability.is_none());
                        assert!(metrics.mean_post_shift_detection_delay.is_none());
                        assert!(metrics.median_post_shift_detection_delay.is_none());
                        assert!(metrics.p90_post_shift_detection_delay.is_none());
                    }
                    let is_accepted_null_control =
                        scenario.class == ScenarioClass::StationaryNull
                            || scenario.class == ScenarioClass::WithinBandTransitionControl
                            || (scenario.class == ScenarioClass::BaselineMisspecification
                                && (COMPOSITE_NULL_LOWER..=COMPOSITE_NULL_UPPER)
                                    .contains(&scenario.pre_shift_p));
                    if is_accepted_null_control {
                        assert!(
                            metrics.scenario_or_pre_shift_alarm_probability <= matched_direction_alarm_probability + EPS,
                            "CUSUM violated matched-reference boundary on an accepted-null scenario: direction={}, scenario={}, horizon={}, alarm={}",
                            direction.name(),
                            scenario.name,
                            horizon,
                            metrics.scenario_or_pre_shift_alarm_probability
                        );
                    }
                }
            }
        }
        let large_up = evaluate_cusum_scenario(
            CusumDirection::Upper,
            threshold,
            SCENARIOS.iter().copied().find(|s| s.name == "step_up_080").unwrap(),
        );
        let final_power = large_up.last().unwrap().1.post_shift_detection_probability.unwrap();
        assert!(final_power > 0.99, "upper CUSUM missed large up-shift: {final_power}");
        let large_down = evaluate_cusum_scenario(
            CusumDirection::Lower,
            threshold,
            SCENARIOS.iter().copied().find(|s| s.name == "step_down_010").unwrap(),
        );
        let down_power = large_down.last().unwrap().1.post_shift_detection_probability.unwrap();
        assert!(down_power > 0.99, "lower CUSUM missed large down-shift: {down_power}");
        let selected = SCENARIOS.iter().copied()
            .find(|s| s.name == "selection_bias_no_environment_shift").unwrap();
        let selected_alarm =
            evaluate_cusum_scenario(CusumDirection::Upper, threshold, selected)
                .last()
                .unwrap()
                .1
                .scenario_or_pre_shift_alarm_probability;
        assert!(selected_alarm > 0.99, "selection-bias control did not alarm: {selected_alarm}");
    }

    #[test]
    fn cusum_improves_weak_and_moderate_step_power_at_matched_edge_fpr() {
        let horizon = HORIZONS[HORIZONS.len() - 1];
        let matched_reference_family_alarm_probability =
            composite_boundary_reference_alarm_probability(horizon);
        let matched_direction_alarm_probability = matched_reference_family_alarm_probability / 2.0;
        let threshold = calibrate_cusum_threshold(horizon, matched_direction_alarm_probability);
        let composite_table = family_alarm_table(
            EProcessFamily::CompositeBandBetting,
            NOMINAL_P,
            horizon,
        );

        let cusum_055 = evaluate_cusum_scenario(
            CusumDirection::Upper,
            threshold,
            SCENARIOS.iter().copied().find(|s| s.name == "step_up_055").unwrap(),
        )
        .last()
        .unwrap()
        .1
        .post_shift_detection_probability
        .unwrap();
        let composite_055 = evaluate_with_alarm(
            NOMINAL_P,
            Some(0.55),
            CHANGE_AFTER,
            horizon,
            |n, successes| composite_table[n][successes],
        )
        .detection_probability;

        let cusum_060 = evaluate_cusum_scenario(
            CusumDirection::Upper,
            threshold,
            SCENARIOS.iter().copied().find(|s| s.name == "step_up_060").unwrap(),
        )
        .last()
        .unwrap()
        .1
        .post_shift_detection_probability
        .unwrap();
        let composite_060 = evaluate_with_alarm(
            NOMINAL_P,
            Some(0.60),
            CHANGE_AFTER,
            horizon,
            |n, successes| composite_table[n][successes],
        )
        .detection_probability;

        assert!(
            cusum_055 > composite_055 + 0.025,
            "matched-edge CUSUM did not improve weak-shift power: CUSUM={cusum_055}, composite={composite_055}"
        );
        assert!(
            cusum_060 > composite_060 + 0.15,
            "matched-edge CUSUM did not improve moderate-shift power: CUSUM={cusum_060}, composite={composite_060}"
        );
    }
    #[test]
    fn cusum_controls_predictable_alternating_and_feedback_null_schedules() {
        let horizon = HORIZONS[HORIZONS.len() - 1];
        let matched_reference_family_alarm_probability =
            composite_boundary_reference_alarm_probability(horizon);
        let matched_direction_alarm_probability = matched_reference_family_alarm_probability / 2.0;
        let threshold = calibrate_cusum_threshold(horizon, matched_direction_alarm_probability);

        for direction in CusumDirection::ALL {
            let alternating = evaluate_cusum_null_schedule(
                direction,
                threshold,
                horizon,
                |n, _state| if n % 2 == 0 { 0.52 } else { 0.48 },
            );
            assert!(
                alternating.scenario_or_pre_shift_alarm_probability
                    <= matched_direction_alarm_probability + EPS,
                "CUSUM exceeded matched edge rate on alternating null schedule: direction={}, alarm={}",
                direction.name(),
                alternating.scenario_or_pre_shift_alarm_probability
            );
            assert!(
                (alternating.scenario_or_pre_shift_alarm_probability
                    + alternating.survival_probability
                    - 1.0)
                    .abs()
                    <= 1.0e-9
            );

            // This rate choice is predictable: it depends only on the CUSUM state
            // before the next observation. It preferentially feeds evidence while the
            // statistic is low and damps evidence after the statistic grows.
            let feedback = evaluate_cusum_null_schedule(
                direction,
                threshold,
                horizon,
                |_n, state| match (direction, state < threshold / 2) {
                    (CusumDirection::Upper, true) | (CusumDirection::Lower, false) => 0.52,
                    (CusumDirection::Upper, false) | (CusumDirection::Lower, true) => 0.48,
                },
            );
            assert!(
                feedback.scenario_or_pre_shift_alarm_probability
                    <= matched_direction_alarm_probability + EPS,
                "CUSUM exceeded matched edge rate on predictable feedback null: direction={}, alarm={}",
                direction.name(),
                feedback.scenario_or_pre_shift_alarm_probability
            );
            assert!(
                (feedback.scenario_or_pre_shift_alarm_probability
                    + feedback.survival_probability
                    - 1.0)
                    .abs()
                    <= 1.0e-9
            );
        }
    }


}
