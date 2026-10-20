// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dependency-free synthetic benchmark for capability-preserving allostatic control.
//!
//! This benchmark tests software-control hypotheses only. The state variables are
//! engineering abstractions; they are not evidence of biological or subjective
//! experience. Oracle and shuffled-forecast policies are diagnostic controls.

pub const HORIZON: usize = 48;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scenario {
    Calm,
    Acute,
    Chronic,
    Repeated,
    RegimeShift,
}

impl Scenario {
    pub const fn all() -> [Self; 5] {
        [Self::Calm, Self::Acute, Self::Chronic, Self::Repeated, Self::RegimeShift]
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Calm => "calm",
            Self::Acute => "acute",
            Self::Chronic => "chronic",
            Self::Repeated => "repeated",
            Self::RegimeShift => "regime_shift",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    TaskOnly,
    StressReactive,
    AllostaticReactive,
    PredictiveViability,
    ShuffledForecastControl,
    OraclePredictive,
}

impl Policy {
    pub const fn all() -> [Self; 6] {
        [
            Self::TaskOnly,
            Self::StressReactive,
            Self::AllostaticReactive,
            Self::PredictiveViability,
            Self::ShuffledForecastControl,
            Self::OraclePredictive,
        ]
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::TaskOnly => "task_only",
            Self::StressReactive => "stress_reactive",
            Self::AllostaticReactive => "allostatic_reactive",
            Self::PredictiveViability => "predictive_viability",
            Self::ShuffledForecastControl => "shuffled_forecast_control",
            Self::OraclePredictive => "oracle_predictive",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Push,
    Maintain,
}

#[derive(Clone, Copy, Debug)]
pub struct Observation {
    pub current_stress: f64,
    pub resource_reserve: f64,
    pub allostatic_load: f64,
    pub forecast_load: f64,
    pub forecast_confidence: f64,
    pub shuffled_forecast_load: f64,
    pub shuffled_forecast_confidence: f64,
    pub oracle_forecast_load: f64,
}

/// A viability burden is a normative severity estimate. Confidence is deliberately
/// absent: epistemic uncertainty is measured separately and must not erase severity.
pub fn viability_burden(deviation: f64, viability_weight: f64) -> f64 {
    deviation.clamp(0.0, 1.0) * viability_weight.max(0.0)
}

/// Epistemic uncertainty is reported independently from the estimated burden.
pub fn epistemic_uncertainty(confidence: f64) -> f64 {
    1.0 - confidence.clamp(0.0, 1.0)
}

pub fn mean_burden(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Discrete area under the burden curve. dt is an abstract model-step duration,
/// not seconds unless a separately qualified mapping says otherwise.
pub fn cumulative_exposure(samples: &[f64], dt: f64) -> f64 {
    if !dt.is_finite() || dt < 0.0 || samples.iter().any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    samples.iter().map(|value| value.max(0.0) * dt).sum()
}

fn unit_noise(seed: u64, tick: usize) -> f64 {
    let mut z = seed ^ (tick as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / ((1u64 << 53) as f64)
}

pub fn environmental_load(scenario: Scenario, tick: usize, seed: u64) -> f64 {
    let base = match scenario {
        Scenario::Calm => 0.06,
        Scenario::Acute => {
            if (8..12).contains(&tick) { 0.94 } else { 0.07 }
        }
        Scenario::Chronic => {
            if (5..35).contains(&tick) { 0.58 } else { 0.09 }
        }
        Scenario::Repeated => {
            if (4..7).contains(&tick)
                || (13..16).contains(&tick)
                || (22..25).contains(&tick)
                || (31..34).contains(&tick)
            {
                0.82
            } else {
                0.10
            }
        }
        Scenario::RegimeShift => {
            if tick < HORIZON / 2 { 0.08 } else { 0.72 }
        }
    };
    let jitter = (unit_noise(seed, tick) - 0.5) * 0.04;
    (base + jitter).clamp(0.0, 1.0)
}

/// Synthetic external warning stream. It is scenario input, not computed by
/// reading the realized future environment. Acute/chronic/repeated scenarios
/// provide controlled lead signals; regime_shift is deliberately unannounced.
fn forecast_signal(scenario: Scenario, tick: usize, seed: u64) -> f64 {
    let base = match scenario {
        Scenario::Calm => 0.06,
        Scenario::Acute => {
            if (5..12).contains(&tick) { 0.92 } else { 0.07 }
        }
        Scenario::Chronic => {
            if (3..35).contains(&tick) { 0.58 } else { 0.09 }
        }
        Scenario::Repeated => {
            if (2..7).contains(&tick)
                || (11..16).contains(&tick)
                || (20..25).contains(&tick)
                || (29..34).contains(&tick)
            {
                0.82
            } else {
                0.10
            }
        }
        Scenario::RegimeShift => {
            if tick < HORIZON / 2 { 0.08 } else { 0.72 }
        }
    };
    let jitter = (unit_noise(seed.wrapping_add(313), tick) - 0.5) * 0.04;
    (base + jitter).clamp(0.0, 1.0)
}

/// Deterministic affine permutation of time indices. Since 13 and the
/// 48-step horizon are coprime, every position occurs exactly once. This
/// destroys temporal alignment without changing the warning stream's marginals.
fn shuffled_forecast_index(tick: usize, seed: u64) -> usize {
    let offset = (unit_noise(seed.wrapping_add(991), 0) * HORIZON as f64) as usize;
    (tick * 13 + offset) % HORIZON
}

/// Perfect short-horizon look-ahead, available only to the diagnostic oracle.
fn oracle_forecast_mean(scenario: Scenario, start: usize, seed: u64) -> f64 {
    let mut values = [0.0; 3];
    for (index, value) in values.iter_mut().enumerate() {
        *value = environmental_load(scenario, start + index, seed);
    }
    mean_burden(&values)
}

fn choose_action(policy: Policy, observation: Observation) -> Action {
    let reserve = observation.resource_reserve;
    let load = observation.allostatic_load;
    let stress = observation.current_stress;
    let maintain = match policy {
        Policy::TaskOnly => reserve < 0.12,
        Policy::StressReactive => stress >= 0.78 || reserve < 0.22,
        Policy::AllostaticReactive => load >= 0.58 || reserve < 0.22,
        Policy::PredictiveViability => {
            let uncertainty = epistemic_uncertainty(observation.forecast_confidence);
            let conservative_forecast =
                (observation.forecast_load + uncertainty * 0.25).clamp(0.0, 1.0);
            // Extend the planning horizon only as forecast risk rises. A fixed
            // three-step horizon over-rests in calm conditions because decisions
            // are re-evaluated after every action.
            let horizon = 1.0 + 2.0 * conservative_forecast;
            let projected_reserve =
                reserve - horizon * (0.055 + 0.025 * conservative_forecast);
            let reserve_floor = 0.15 + 0.10 * conservative_forecast;
            let projected_load =
                load + horizon * (0.026 + 0.032 * conservative_forecast);
            projected_reserve < reserve_floor || projected_load > 0.68 || reserve < 0.22
        }
        Policy::ShuffledForecastControl => {
            let uncertainty = epistemic_uncertainty(observation.shuffled_forecast_confidence);
            let false_forecast =
                (observation.shuffled_forecast_load + uncertainty * 0.25).clamp(0.0, 1.0);
            let horizon = 1.0 + 2.0 * false_forecast;
            let projected_reserve = reserve - horizon * (0.055 + 0.025 * false_forecast);
            let reserve_floor = 0.15 + 0.10 * false_forecast;
            let projected_load = load + horizon * (0.026 + 0.032 * false_forecast);
            projected_reserve < reserve_floor || projected_load > 0.68 || reserve < 0.22
        }
        Policy::OraclePredictive => {
            let forecast = observation.oracle_forecast_load;
            let horizon = 1.0 + 2.0 * forecast;
            let projected_reserve = reserve - horizon * (0.055 + 0.025 * forecast);
            let reserve_floor = 0.15 + 0.10 * forecast;
            let projected_load = load + horizon * (0.026 + 0.032 * forecast);
            projected_reserve < reserve_floor || projected_load > 0.68 || reserve < 0.22
        }
    };

    if maintain {
        Action::Maintain
    } else {
        Action::Push
    }
}

#[derive(Clone, Debug)]
pub struct EpisodeMetrics {
    pub task_reward: f64,
    pub cumulative_exposure: f64,
    pub mean_load: f64,
    pub peak_load: f64,
    pub minimum_reserve: f64,
    pub terminal_reserve: f64,
    pub terminal_load: f64,
    pub unsafe_ticks: usize,
    pub missed_limits: usize,
    pub false_alarms: usize,
    pub push_steps: usize,
    pub maintain_steps: usize,
    pub resource_cost: f64,
    pub failed: bool,
    /// None means recovery was not observed before the fixed horizon ended.
    pub recovery_steps: Option<usize>,
}

pub fn run_episode(policy: Policy, scenario: Scenario, seed: u64) -> EpisodeMetrics {
    let mut reserve: f64 = 0.85;
    let mut minimum_reserve: f64 = reserve;
    let mut allostatic_load: f64 = 0.0;
    let mut task_reward = 0.0;
    let mut cumulative_load = 0.0;
    let mut peak_load: f64 = 0.0;
    let mut unsafe_ticks = 0;
    let mut missed_limits = 0;
    let mut false_alarms = 0;
    let mut push_steps = 0;
    let mut maintain_steps = 0;
    let mut resource_cost = 0.0;
    let mut failed = false;
    let mut last_disturbance = None;
    let mut trace: Vec<(f64, f64)> = Vec::with_capacity(HORIZON);

    for tick in 0..HORIZON {
        let stress = environmental_load(scenario, tick, seed);
        if stress > 0.35 {
            last_disturbance = Some(tick);
        }

        // Candidate policies receive a noisy exogenous warning stream. Confidence
        // controls forecast noise but does not expose the scenario or future state.
        // Only OraclePredictive receives actual future environment values.
        let forecast_confidence =
            0.65 + 0.30 * unit_noise(seed.wrapping_add(777), tick);
        let forecast_noise_scale = 0.10 + 0.35 * (1.0 - forecast_confidence);
        let forecast_noise =
            (unit_noise(seed.wrapping_add(91), tick) - 0.5) * forecast_noise_scale;
        let forecast_load =
            (forecast_signal(scenario, tick, seed) + forecast_noise).clamp(0.0, 1.0);
        let shuffled_start = shuffled_forecast_index(tick, seed);
        let shuffled_forecast_confidence =
            0.65 + 0.30 * unit_noise(seed.wrapping_add(777), shuffled_start);
        let shuffled_noise_scale = 0.10 + 0.35 * (1.0 - shuffled_forecast_confidence);
        let shuffled_noise =
            (unit_noise(seed.wrapping_add(91), shuffled_start) - 0.5) * shuffled_noise_scale;
        let shuffled_forecast_load =
            (forecast_signal(scenario, shuffled_start, seed) + shuffled_noise).clamp(0.0, 1.0);
        let oracle_forecast_load = oracle_forecast_mean(scenario, tick + 1, seed);

        let observation = Observation {
            current_stress: stress,
            resource_reserve: reserve,
            allostatic_load,
            forecast_load,
            forecast_confidence,
            shuffled_forecast_load,
            shuffled_forecast_confidence,
            oracle_forecast_load,
        };
        let action = choose_action(policy, observation);
        let push_cost = 0.055 + 0.025 * stress;

        if action == Action::Push {
            let projected_reserve = reserve - push_cost;
            let projected_load = allostatic_load + 0.008 + stress * 0.032 + 0.018;
            if projected_reserve < 0.15 || projected_load > 0.88 {
                missed_limits += 1;
            }
            reserve -= push_cost;
            resource_cost += push_cost;
            task_reward += 1.0 - 0.12 * stress;
            push_steps += 1;
            allostatic_load += 0.008 + stress * 0.032 + 0.018;
        } else {
            task_reward += 0.12;
            maintain_steps += 1;
            reserve += 0.08 * (1.0 - 0.7 * stress);
            let recovery = if stress < 0.25 { 0.055 } else { 0.020 };
            allostatic_load += 0.008 + stress * 0.032 - recovery;
            if stress < 0.20 && reserve > 0.30 && allostatic_load < 0.25
                && forecast_load < 0.25
            {
                false_alarms += 1;
            }
        }

        reserve = reserve.clamp(0.0, 1.0);
        minimum_reserve = minimum_reserve.min(reserve);
        allostatic_load = allostatic_load.clamp(0.0, 1.0);
        cumulative_load += allostatic_load;
        peak_load = peak_load.max(allostatic_load);
        if reserve < 0.15 || allostatic_load > 0.88 {
            unsafe_ticks += 1;
        }
        if reserve < 0.05 || allostatic_load >= 0.98 {
            failed = true;
        }
        trace.push((reserve, allostatic_load));
    }

    let recovery_steps = last_disturbance.and_then(|last| {
        trace.iter().enumerate().find_map(|(tick, (reserve, load))| {
            if tick > last && *reserve >= 0.75 && *load <= 0.25 {
                Some(tick - last)
            } else {
                None
            }
        })
    });

    EpisodeMetrics {
        task_reward,
        cumulative_exposure: cumulative_load,
        mean_load: cumulative_load / HORIZON as f64,
        peak_load,
        minimum_reserve,
        terminal_reserve: reserve,
        terminal_load: allostatic_load,
        unsafe_ticks,
        missed_limits,
        false_alarms,
        push_steps,
        maintain_steps,
        resource_cost,
        failed,
        recovery_steps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cumulative_exposure_distinguishes_duration_from_mean() {
        let short = vec![0.4; 8];
        let long = vec![0.4; 16];
        assert!((mean_burden(&short) - mean_burden(&long)).abs() < 1e-12);
        assert!((cumulative_exposure(&long, 1.0)
            - 2.0 * cumulative_exposure(&short, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn cumulative_exposure_rejects_nonfinite_evidence() {
        assert!(cumulative_exposure(&[0.2, f64::NAN], 1.0).is_nan());
        assert!(cumulative_exposure(&[0.2, f64::INFINITY], 1.0).is_nan());
        assert!(cumulative_exposure(&[0.2], f64::NAN).is_nan());
        assert!(cumulative_exposure(&[0.2], -1.0).is_nan());
    }

    #[test]
    fn confidence_change_does_not_erase_viability_burden() {
        let high_confidence = (0.8, 1.5, 0.9);
        let low_confidence = (0.8, 1.5, 0.2);
        let burden_high_confidence =
            viability_burden(high_confidence.0, high_confidence.1);
        let burden_low_confidence =
            viability_burden(low_confidence.0, low_confidence.1);
        assert!((burden_high_confidence - burden_low_confidence).abs() < 1e-12);
        assert!(
            epistemic_uncertainty(low_confidence.2) > epistemic_uncertainty(high_confidence.2)
        );
    }

    #[test]
    fn equal_mean_can_hide_different_peaks() {
        let acute = [1.0, 0.0];
        let chronic = [0.5, 0.5];
        assert!((mean_burden(&acute) - mean_burden(&chronic)).abs() < 1e-12);
        assert!(acute.iter().copied().fold(0.0, f64::max)
            > chronic.iter().copied().fold(0.0, f64::max));
        assert!((cumulative_exposure(&acute, 1.0)
            - cumulative_exposure(&chronic, 1.0)).abs() < 1e-12);
    }

    #[test]
    fn shuffled_forecast_preserves_all_indices_but_breaks_time_order() {
        for seed in 0..32 {
            let mut indices: Vec<usize> =
                (0..HORIZON).map(|tick| shuffled_forecast_index(tick, seed)).collect();
            indices.sort_unstable();
            assert_eq!(indices, (0..HORIZON).collect::<Vec<_>>());
        }
        let ordered: Vec<usize> = (0..HORIZON).collect();
        let shuffled: Vec<usize> =
            (0..HORIZON).map(|tick| shuffled_forecast_index(tick, 0)).collect();
        assert_ne!(shuffled, ordered);
    }

    #[test]
    fn synthetic_warning_can_precede_the_actual_disturbance() {
        let warning = forecast_signal(Scenario::Acute, 5, 0);
        let current_stress = environmental_load(Scenario::Acute, 5, 0);
        assert!(warning > current_stress + 0.5);
    }

    #[test]
    fn predictive_policy_can_act_before_current_stress_rises() {
        let observation = Observation {
            current_stress: 0.05,
            resource_reserve: 0.45,
            allostatic_load: 0.40,
            forecast_load: 0.90,
            forecast_confidence: 0.90,
            shuffled_forecast_load: 0.10,
            shuffled_forecast_confidence: 0.90,
            oracle_forecast_load: 0.90,
        };
        assert_eq!(choose_action(Policy::StressReactive, observation), Action::Push);
        assert_eq!(choose_action(Policy::PredictiveViability, observation), Action::Maintain);
        assert_eq!(choose_action(Policy::ShuffledForecastControl, observation), Action::Push);
    }

    #[test]
    fn outputs_are_bounded_and_finite_for_the_declared_suite() {
        for scenario in Scenario::all() {
            for seed in 0..32 {
                for policy in Policy::all() {
                    let result = run_episode(policy, scenario, seed);
                    assert!(result.task_reward.is_finite());
                    assert!(result.cumulative_exposure.is_finite());
                    assert!((0.0..=1.0).contains(&result.mean_load));
                    assert!((0.0..=1.0).contains(&result.peak_load));
                    assert!((0.0..=1.0).contains(&result.minimum_reserve));
                    assert!((0.0..=1.0).contains(&result.terminal_reserve));
                    assert!((0.0..=1.0).contains(&result.terminal_load));
                    assert!(result.unsafe_ticks <= HORIZON);
                    assert!(result.push_steps + result.maintain_steps == HORIZON);
                }
            }
        }
    }
}
