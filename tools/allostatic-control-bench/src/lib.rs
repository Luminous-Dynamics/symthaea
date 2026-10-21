// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dependency-free synthetic benchmark for capability-preserving allostatic control.
//!
//! This benchmark tests software-control hypotheses only. The state variables are
//! engineering abstractions; they are not evidence of biological or subjective
//! experience. Oracle and shuffled-forecast policies are diagnostic controls.

pub const HORIZON: usize = 48;
pub const EPISODE_CSV_HEADER: &str = "scenario,seed,policy,shielded,task_reward,cumulative_exposure,mean_load,peak_load,minimum_reserve,mean_safe_action_options,minimum_safe_action_options,terminal_reserve,terminal_load,unsafe_ticks,unsafe_episode,violation_exposure,max_violation_severity,missed_limits,shield_vetoes,no_safe_action_ticks,false_alarms,false_alarm_opportunities,push_steps,maintain_steps,emergency_recover_steps,resource_cost,failed,recovery_steps";

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
        [
            Self::Calm,
            Self::Acute,
            Self::Chronic,
            Self::Repeated,
            Self::RegimeShift,
        ]
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
    EmergencyRecover,
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
            if (8..12).contains(&tick) {
                0.94
            } else {
                0.07
            }
        }
        Scenario::Chronic => {
            if (5..35).contains(&tick) {
                0.58
            } else {
                0.09
            }
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
            if tick < HORIZON / 2 {
                0.08
            } else {
                0.72
            }
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
            if (5..12).contains(&tick) {
                0.92
            } else {
                0.07
            }
        }
        Scenario::Chronic => {
            if (3..35).contains(&tick) {
                0.58
            } else {
                0.09
            }
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
        // The regime shift is deliberately unannounced: the warning stream
        // stays at baseline even when the true disturbance changes at mid-horizon.
        Scenario::RegimeShift => 0.08,
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
            let projected_reserve = reserve - horizon * (0.055 + 0.025 * conservative_forecast);
            let reserve_floor = 0.15 + 0.10 * conservative_forecast;
            let projected_load = load + horizon * (0.026 + 0.032 * conservative_forecast);
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

/// Predict the one-step state under the benchmark's declared transition model.
fn projected_state(action: Action, reserve: f64, load: f64, stress: f64) -> (f64, f64) {
    match action {
        Action::Push => (
            reserve - (0.055 + 0.025 * stress),
            load + 0.008 + stress * 0.032 + 0.018,
        ),
        Action::Maintain => (
            reserve + 0.08 * (1.0 - 0.7 * stress),
            load + 0.008 + stress * 0.032 - if stress < 0.25 { 0.055 } else { 0.020 },
        ),
        Action::EmergencyRecover => (
            reserve + 0.10 * (1.0 - 0.5 * stress),
            load + 0.008 + stress * 0.032 - 0.10,
        ),
    }
}

fn satisfies_one_step_viability(action: Action, reserve: f64, load: f64, stress: f64) -> bool {
    let (next_reserve, next_load) = projected_state(action, reserve, load, stress);
    next_reserve >= 0.15 && next_load <= 0.88
}

/// Count how many currently available actions stay inside the declared
/// one-step viability region. This measures local option breadth, not long-horizon
/// reachability or the value of those options.
pub fn safe_action_option_count(
    reserve: f64,
    load: f64,
    stress: f64,
    shield_can_recover: bool,
) -> usize {
    let mut count = 0;
    if satisfies_one_step_viability(Action::Push, reserve, load, stress) {
        count += 1;
    }
    if satisfies_one_step_viability(Action::Maintain, reserve, load, stress) {
        count += 1;
    }
    if shield_can_recover
        && satisfies_one_step_viability(Action::EmergencyRecover, reserve, load, stress)
    {
        count += 1;
    }
    count
}

/// Model-perfect one-step shield for the synthetic environment. This is not a
/// formal guarantee outside the declared transition model or when state estimates
/// and action effects are wrong.
fn apply_safety_shield(
    proposed: Action,
    reserve: f64,
    load: f64,
    stress: f64,
) -> (Action, bool, bool) {
    if satisfies_one_step_viability(proposed, reserve, load, stress) {
        return (proposed, false, false);
    }
    if satisfies_one_step_viability(Action::Maintain, reserve, load, stress) {
        return (Action::Maintain, true, false);
    }
    if satisfies_one_step_viability(Action::EmergencyRecover, reserve, load, stress) {
        return (Action::EmergencyRecover, true, false);
    }
    (Action::EmergencyRecover, true, true)
}

#[derive(Clone, Debug)]
pub struct EpisodeMetrics {
    pub shielded: bool,
    pub task_reward: f64,
    pub cumulative_exposure: f64,
    pub mean_load: f64,
    pub peak_load: f64,
    pub minimum_reserve: f64,
    /// Mean number of currently viable one-step actions over the episode, [0, 3].
    pub mean_safe_action_options: f64,
    /// Minimum number of currently viable one-step actions in any step, [0, 3].
    pub minimum_safe_action_options: usize,
    pub terminal_reserve: f64,
    pub terminal_load: f64,
    pub unsafe_ticks: usize,
    pub unsafe_episode: bool,
    /// Sum of normalized constraint-violation magnitudes over abstract time steps.
    pub violation_exposure: f64,
    /// Largest normalized single-step violation, in [0, 1].
    pub max_violation_severity: f64,
    /// Number of raw-policy action proposals whose predicted next state crossed a declared limit.
    pub missed_limits: usize,
    /// Actions vetoed/replaced by the independent one-step viability filter.
    pub shield_vetoes: usize,
    /// Steps for which no available action was predicted to satisfy both one-step limits.
    pub no_safe_action_ticks: usize,
    pub false_alarms: usize,
    pub false_alarm_opportunities: usize,
    pub push_steps: usize,
    pub maintain_steps: usize,
    pub emergency_recover_steps: usize,
    pub resource_cost: f64,
    pub failed: bool,
    /// None means recovery was not observed before the fixed horizon ended.
    pub recovery_steps: Option<usize>,
}

pub fn run_episode(policy: Policy, scenario: Scenario, seed: u64) -> EpisodeMetrics {
    run_episode_with_shield(policy, scenario, seed, false)
}

pub fn run_episode_with_shield(
    policy: Policy,
    scenario: Scenario,
    seed: u64,
    shielded: bool,
) -> EpisodeMetrics {
    let mut reserve: f64 = 0.85;
    let mut minimum_reserve: f64 = reserve;
    let mut safe_action_options_total: usize = 0;
    let mut minimum_safe_action_options: usize = 3;
    let mut allostatic_load: f64 = 0.0;
    let mut task_reward = 0.0;
    let mut cumulative_load = 0.0;
    let mut peak_load: f64 = 0.0;
    let mut unsafe_ticks = 0;
    let mut violation_exposure = 0.0;
    let mut max_violation_severity: f64 = 0.0;
    let mut missed_limits = 0;
    let mut shield_vetoes = 0;
    let mut no_safe_action_ticks = 0;
    let mut false_alarms = 0;
    let mut false_alarm_opportunities = 0;
    let mut push_steps = 0;
    let mut maintain_steps = 0;
    let mut emergency_recover_steps = 0;
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
        let forecast_confidence = 0.65 + 0.30 * unit_noise(seed.wrapping_add(777), tick);
        let forecast_noise_scale = 0.10 + 0.35 * (1.0 - forecast_confidence);
        let forecast_noise = (unit_noise(seed.wrapping_add(91), tick) - 0.5) * forecast_noise_scale;
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
        // Count false-alarm opportunities from the pre-action state; do not
        // let the action itself change the denominator or hide the opportunity.
        let safe_options = safe_action_option_count(reserve, allostatic_load, stress, shielded);
        safe_action_options_total += safe_options;
        minimum_safe_action_options = minimum_safe_action_options.min(safe_options);

        let false_alarm_opportunity =
            stress < 0.20 && reserve > 0.30 && allostatic_load < 0.25 && forecast_load < 0.25;
        if false_alarm_opportunity {
            false_alarm_opportunities += 1;
        }

        let proposed_action = choose_action(policy, observation);
        let projected = projected_state(proposed_action, reserve, allostatic_load, stress);
        if projected.0 < 0.15 || projected.1 > 0.88 {
            // Counterfactual raw-policy proposal, before any shield veto.
            missed_limits += 1;
        }
        let (action, vetoed, no_safe_action) = if shielded {
            apply_safety_shield(proposed_action, reserve, allostatic_load, stress)
        } else {
            (proposed_action, false, false)
        };
        if vetoed {
            shield_vetoes += 1;
        }
        if no_safe_action {
            no_safe_action_ticks += 1;
        }

        // Reward/accounting is action-specific; state evolution uses the same
        // declared transition function used by the shield's one-step prediction.
        match action {
            Action::Push => {
                resource_cost += 0.055 + 0.025 * stress;
                task_reward += 1.0 - 0.12 * stress;
                push_steps += 1;
            }
            Action::Maintain => {
                task_reward += 0.12;
                maintain_steps += 1;
            }
            Action::EmergencyRecover => {
                // A conservative, task-interrupting recovery action reserved for
                // the shield; ordinary policies never request it.
                task_reward += 0.02;
                emergency_recover_steps += 1;
            }
        }
        let (next_reserve, next_load) = projected_state(action, reserve, allostatic_load, stress);
        reserve = next_reserve;
        allostatic_load = next_load;

        if action == Action::Maintain && false_alarm_opportunity {
            false_alarms += 1;
        }

        reserve = reserve.clamp(0.0, 1.0);
        minimum_reserve = minimum_reserve.min(reserve);
        allostatic_load = allostatic_load.clamp(0.0, 1.0);
        cumulative_load += allostatic_load;
        peak_load = peak_load.max(allostatic_load);
        let reserve_severity = ((0.15 - reserve).max(0.0) / 0.15).clamp(0.0, 1.0);
        let load_severity = ((allostatic_load - 0.88).max(0.0) / 0.12).clamp(0.0, 1.0);
        let step_severity = reserve_severity.max(load_severity);
        violation_exposure += reserve_severity + load_severity;
        max_violation_severity = max_violation_severity.max(step_severity);
        if reserve < 0.15 || allostatic_load > 0.88 {
            unsafe_ticks += 1;
        }
        if reserve < 0.05 || allostatic_load >= 0.98 {
            failed = true;
        }
        trace.push((reserve, allostatic_load));
    }

    let recovery_steps = last_disturbance.and_then(|last| {
        trace
            .iter()
            .enumerate()
            .find_map(|(tick, (reserve, load))| {
                if tick > last && *reserve >= 0.75 && *load <= 0.25 {
                    Some(tick - last)
                } else {
                    None
                }
            })
    });

    EpisodeMetrics {
        shielded,
        task_reward,
        cumulative_exposure: cumulative_load,
        mean_load: cumulative_load / HORIZON as f64,
        peak_load,
        minimum_reserve,
        mean_safe_action_options: safe_action_options_total as f64 / HORIZON as f64,
        minimum_safe_action_options,
        terminal_reserve: reserve,
        terminal_load: allostatic_load,
        unsafe_ticks,
        unsafe_episode: unsafe_ticks > 0,
        violation_exposure,
        max_violation_severity,
        missed_limits,
        shield_vetoes,
        no_safe_action_ticks,
        false_alarms,
        false_alarm_opportunities,
        push_steps,
        maintain_steps,
        emergency_recover_steps,
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
        assert!(
            (cumulative_exposure(&long, 1.0) - 2.0 * cumulative_exposure(&short, 1.0)).abs()
                < 1e-12
        );
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
        let burden_high_confidence = viability_burden(high_confidence.0, high_confidence.1);
        let burden_low_confidence = viability_burden(low_confidence.0, low_confidence.1);
        assert!((burden_high_confidence - burden_low_confidence).abs() < 1e-12);
        assert!(epistemic_uncertainty(low_confidence.2) > epistemic_uncertainty(high_confidence.2));
    }

    #[test]
    fn equal_mean_can_hide_different_peaks() {
        let acute = [1.0, 0.0];
        let chronic = [0.5, 0.5];
        assert!((mean_burden(&acute) - mean_burden(&chronic)).abs() < 1e-12);
        assert!(
            acute.iter().copied().fold(0.0, f64::max) > chronic.iter().copied().fold(0.0, f64::max)
        );
        assert!(
            (cumulative_exposure(&acute, 1.0) - cumulative_exposure(&chronic, 1.0)).abs() < 1e-12
        );
    }

    #[test]
    fn shuffled_forecast_preserves_all_indices_but_breaks_time_order() {
        for seed in 0..32 {
            let mut indices: Vec<usize> = (0..HORIZON)
                .map(|tick| shuffled_forecast_index(tick, seed))
                .collect();
            indices.sort_unstable();
            assert_eq!(indices, (0..HORIZON).collect::<Vec<_>>());
        }
        let ordered: Vec<usize> = (0..HORIZON).collect();
        let shuffled: Vec<usize> = (0..HORIZON)
            .map(|tick| shuffled_forecast_index(tick, 0))
            .collect();
        assert_ne!(shuffled, ordered);
    }

    #[test]
    fn synthetic_warning_can_precede_the_actual_disturbance() {
        let warning = forecast_signal(Scenario::Acute, 5, 0);
        let current_stress = environmental_load(Scenario::Acute, 5, 0);
        assert!(warning > current_stress + 0.5);
    }

    #[test]
    fn regime_shift_has_no_advance_warning() {
        for tick in 0..HORIZON {
            assert!(forecast_signal(Scenario::RegimeShift, tick, 7) < 0.11);
        }
        assert!(environmental_load(Scenario::RegimeShift, HORIZON / 2, 7) > 0.65);
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
        assert_eq!(
            choose_action(Policy::StressReactive, observation),
            Action::Push
        );
        assert_eq!(
            choose_action(Policy::PredictiveViability, observation),
            Action::Maintain
        );
        assert_eq!(
            choose_action(Policy::ShuffledForecastControl, observation),
            Action::Push
        );
    }

    #[test]
    fn viable_action_breadth_distinguishes_full_reduced_and_empty_sets() {
        assert_eq!(safe_action_option_count(0.80, 0.20, 0.10, false), 2);
        assert_eq!(safe_action_option_count(0.80, 0.20, 0.10, true), 3);
        assert_eq!(safe_action_option_count(0.16, 0.87, 0.90, false), 0);
        assert_eq!(safe_action_option_count(0.16, 0.87, 0.90, true), 1);
        assert_eq!(safe_action_option_count(0.0, 0.95, 1.0, false), 0);
        assert_eq!(safe_action_option_count(0.0, 0.95, 1.0, true), 0);
    }

    #[test]
    fn maintain_can_still_violate_load_limit_under_high_burden() {
        assert!(!satisfies_one_step_viability(
            Action::Maintain,
            0.50,
            0.87,
            0.90
        ));
    }

    #[test]
    fn safety_shield_vetoes_unsafe_push_and_selects_recovery_if_needed() {
        let (action, vetoed, no_safe_action) = apply_safety_shield(Action::Push, 0.16, 0.87, 0.90);
        assert_eq!(action, Action::EmergencyRecover);
        assert!(vetoed);
        assert!(!no_safe_action);
        assert!(satisfies_one_step_viability(action, 0.16, 0.87, 0.90));
    }

    #[test]
    fn outputs_are_bounded_and_finite_for_the_declared_suite() {
        for scenario in Scenario::all() {
            for seed in 0..32 {
                for policy in Policy::all() {
                    for shielded in [false, true] {
                        let result = run_episode_with_shield(policy, scenario, seed, shielded);
                        assert!(result.task_reward.is_finite());
                        assert!(result.cumulative_exposure.is_finite());
                        assert!((0.0..=1.0).contains(&result.mean_load));
                        assert!((0.0..=1.0).contains(&result.peak_load));
                        assert!((0.0..=1.0).contains(&result.minimum_reserve));
                        assert!((0.0..=3.0).contains(&result.mean_safe_action_options));
                        assert!(result.minimum_safe_action_options <= 3);
                        assert!((0.0..=1.0).contains(&result.terminal_reserve));
                        assert!((0.0..=1.0).contains(&result.terminal_load));
                        assert!(result.violation_exposure.is_finite());
                        assert!((0.0..=1.0).contains(&result.max_violation_severity));
                        assert!(result.unsafe_ticks <= HORIZON);
                        assert_eq!(result.unsafe_episode, result.unsafe_ticks > 0);
                        assert!(
                            result.push_steps
                                + result.maintain_steps
                                + result.emergency_recover_steps
                                == HORIZON
                        );
                        if shielded {
                            assert!(result.minimum_safe_action_options >= 1);
                            assert_eq!(result.no_safe_action_ticks, 0);
                            assert_eq!(
                                result.unsafe_ticks, 0,
                                "model-perfect shield should maintain one-step viability: {scenario:?} {policy:?} seed={seed}"
                            );
                        }
                    }
                }
            }
        }
    }
}
