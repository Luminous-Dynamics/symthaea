// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Availability-aligned extension of the deterministic irregular scalar workload.
//!
//! The companion hdc_ltc_linear_input_reference example reports trajectory
//! error at each observation's source timestamp. This example additionally
//! projects each reconstructed state to the time the observation is actually
//! available. The candidate sees only delivered observations; hidden oracle
//! knots are used exclusively by the evaluator.
//!
//! Source-time error and availability-time error are distinct measurements.
//! The latter includes the unobserved interval between source time and arrival:
//! the candidate projects that interval with a zero-order hold on the newest
//! delivered value, while the evaluator follows the hidden input trajectory.
//!
//! This is an algorithm-track reference with zero assumed processing latency.
//! It does not measure wall-clock service time, hardware energy, or HDC-LTC
//! network performance. Checkpoint/replay cost for retrospective reconstruction
//! is also not measured here and must be charged by a later live implementation.
//!
//! From the repository root:
//!   cargo fmt --check
//!   cargo test --locked -p symthaea-silicon --example temporal_availability_reference
//!   cargo run --locked -p symthaea-silicon --example temporal_availability_reference

use serde::{Deserialize, Serialize};
use std::error::Error;

const ORACLE_RK4_STEPS_PER_KNOT: usize = 4_096;
const FRESHNESS_BUDGET_TICKS: u64 = 6;
const DECISION_DEADLINE_BUDGET_TICKS: u64 = 8;
const INITIAL_STATE: f64 = -0.25;
const TAU: f64 = 140.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum SamplingProfile {
    Regular,
    Jittered,
    Bursty,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OracleKnot {
    time: u64,
    target: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateObservation {
    event_id: u64,
    source_time: u64,
    available_at: u64,
    target: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolverKind {
    ZeroOrderHoldExact,
    LinearInterpolationExact,
    LinearEuler,
    LinearRungeKutta4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MethodSpec {
    kind: SolverKind,
    substeps: usize,
}

impl MethodSpec {
    fn name(self) -> String {
        match self.kind {
            SolverKind::ZeroOrderHoldExact => "zero_order_hold_exact".to_owned(),
            SolverKind::LinearInterpolationExact => {
                "linear_interpolation_exact_scalar".to_owned()
            }
            SolverKind::LinearEuler => format!("linear_euler_nfe_{}", self.substeps),
            SolverKind::LinearRungeKutta4 => {
                format!("linear_rk4_nfe_{}", 4 * self.substeps)
            }
        }
    }

    fn rhs_evaluations_per_interval(self) -> u64 {
        match self.kind {
            SolverKind::ZeroOrderHoldExact | SolverKind::LinearInterpolationExact => 0,
            SolverKind::LinearEuler => self.substeps as u64,
            SolverKind::LinearRungeKutta4 => (4 * self.substeps) as u64,
        }
    }

    fn analytic_updates_per_interval(self) -> u64 {
        match self.kind {
            SolverKind::ZeroOrderHoldExact | SolverKind::LinearInterpolationExact => 1,
            SolverKind::LinearEuler | SolverKind::LinearRungeKutta4 => 0,
        }
    }
}

fn method_specs() -> Vec<MethodSpec> {
    let mut methods = vec![
        MethodSpec {
            kind: SolverKind::ZeroOrderHoldExact,
            substeps: 0,
        },
        MethodSpec {
            kind: SolverKind::LinearInterpolationExact,
            substeps: 0,
        },
    ];
    for substeps in [4, 8, 16, 32] {
        methods.push(MethodSpec {
            kind: SolverKind::LinearEuler,
            substeps,
        });
    }
    // RK4 has four RHS evaluations per substep. These correspond to fixed
    // budgets of 4, 8, 16 and 32 evaluations, respectively.
    for substeps in [1, 2, 4, 8] {
        methods.push(MethodSpec {
            kind: SolverKind::LinearRungeKutta4,
            substeps,
        });
    }
    methods
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MethodResult {
    method: String,
    analytic_interval_updates: u64,
    euler_rhs_evaluations: u64,
    rk4_rhs_evaluations: u64,
    availability_projection_updates: u64,
    max_abs_source_time_error: f64,
    terminal_abs_source_time_error: f64,
    max_abs_availability_time_error: f64,
    terminal_abs_availability_time_error: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TraceResult {
    workload_version: String,
    profile: SamplingProfile,
    time_unit: String,
    fixture_fingerprint: String,
    scheduled_events: u64,
    dropped_events: u64,
    delivered_events: u64,
    freshness_budget_ticks: u64,
    decision_deadline_budget_ticks: u64,
    stale_outputs: u64,
    late_outputs: u64,
    max_source_to_availability_delay: u64,
    outputs_emitted_at_availability: u64,
    methods: Vec<MethodResult>,
}

fn oracle_knots() -> Vec<OracleKnot> {
    vec![
        OracleKnot {
            time: 0,
            target: 0.0,
        },
        OracleKnot {
            time: 115,
            target: 0.85,
        },
        OracleKnot {
            time: 260,
            target: -0.45,
        },
        OracleKnot {
            time: 410,
            target: 1.10,
        },
        OracleKnot {
            time: 590,
            target: -0.80,
        },
        OracleKnot {
            time: 780,
            target: 0.35,
        },
        OracleKnot {
            time: 1_000,
            target: 0.65,
        },
    ]
}

fn target_at(knots: &[OracleKnot], time: u64) -> Result<f64, &'static str> {
    if knots.len() < 2 || time < knots[0].time || time > knots[knots.len() - 1].time {
        return Err("time outside oracle trace");
    }
    for pair in knots.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if time >= left.time && time <= right.time {
            let width = (right.time - left.time) as f64;
            let alpha = (time - left.time) as f64 / width;
            return Ok(left.target + (right.target - left.target) * alpha);
        }
    }
    Err("oracle interval not found")
}

#[inline]
fn one_minus_exp_neg(x: f64) -> f64 {
    -(-x).exp_m1()
}

/// Exact endpoint solution for tau * dz/dt = g(t) - z with a linear g(t).
fn exact_linear_target_update(
    z0: f64,
    g0: f64,
    g1: f64,
    tau: f64,
    h: f64,
) -> Result<f64, &'static str> {
    if !h.is_finite() || h < 0.0 || !tau.is_finite() || tau <= 0.0 {
        return Err("invalid interval or time constant");
    }
    if !z0.is_finite() || !g0.is_finite() || !g1.is_finite() {
        return Err("non-finite state or input");
    }
    if h == 0.0 {
        return Ok(z0);
    }

    let x = h / tau;
    let decay = (-x).exp();
    let one_minus = one_minus_exp_neg(x);
    let correction_over_h = if x < 1.0e-3 {
        let polynomial = 0.5
            + x * (-1.0 / 6.0
                + x * (1.0 / 24.0
                    + x * (-1.0 / 120.0
                        + x * (1.0 / 720.0
                            + x * (-1.0 / 5040.0 + x * (1.0 / 40320.0))))));
        x * polynomial
    } else {
        (h - tau * one_minus) / h
    };

    let result = decay * z0 + one_minus * g0 + (g1 - g0) * correction_over_h;
    if result.is_finite() {
        Ok(result)
    } else {
        Err("non-finite integration result")
    }
}

fn euler_linear_target(
    z0: f64,
    g0: f64,
    g1: f64,
    tau: f64,
    h: f64,
    steps: usize,
) -> Result<f64, &'static str> {
    if steps == 0 || !h.is_finite() || h <= 0.0 || !tau.is_finite() || tau <= 0.0 {
        return Err("invalid Euler integration configuration");
    }
    let step = h / steps as f64;
    let mut state = z0;
    for index in 0..steps {
        let local_time = index as f64 * step;
        let target = g0 + (g1 - g0) * (local_time / h);
        state += step * (target - state) / tau;
    }
    if state.is_finite() {
        Ok(state)
    } else {
        Err("Euler integration produced non-finite state")
    }
}

fn rk4_linear_target(
    z0: f64,
    g0: f64,
    g1: f64,
    tau: f64,
    h: f64,
    steps: usize,
) -> Result<f64, &'static str> {
    if steps == 0 || !h.is_finite() || h <= 0.0 || !tau.is_finite() || tau <= 0.0 {
        return Err("invalid RK4 integration configuration");
    }
    let step = h / steps as f64;
    let derivative = |time: f64, state: f64| {
        let target = g0 + (g1 - g0) * (time / h);
        (target - state) / tau
    };

    let mut state = z0;
    for index in 0..steps {
        let time = index as f64 * step;
        let k1 = derivative(time, state);
        let k2 = derivative(time + step / 2.0, state + step * k1 / 2.0);
        let k3 = derivative(time + step / 2.0, state + step * k2 / 2.0);
        let k4 = derivative(time + step, state + step * k3);
        state += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
    }
    if state.is_finite() {
        Ok(state)
    } else {
        Err("RK4 integration produced non-finite state")
    }
}

fn oracle_state_at(
    knots: &[OracleKnot],
    initial_state: f64,
    tau: f64,
    time: u64,
) -> Result<f64, &'static str> {
    if !initial_state.is_finite() || !tau.is_finite() || tau <= 0.0 {
        return Err("invalid oracle state or time constant");
    }
    if knots.len() < 2 || time < knots[0].time || time > knots[knots.len() - 1].time {
        return Err("time outside oracle trace");
    }

    let mut state = initial_state;
    for pair in knots.windows(2) {
        let left = pair[0];
        if time <= left.time {
            break;
        }
        let right = pair[1];
        let end = time.min(right.time);
        let g0 = target_at(knots, left.time)?;
        let g1 = target_at(knots, end)?;
        let interval = (end - left.time) as f64;
        state = rk4_linear_target(
            state,
            g0,
            g1,
            tau,
            interval,
            ORACLE_RK4_STEPS_PER_KNOT,
        )?;
        if end == time {
            break;
        }
    }
    Ok(state)
}

fn scheduled_times(profile: SamplingProfile) -> Vec<u64> {
    match profile {
        SamplingProfile::Regular => (0..=10).map(|index| index * 100).collect(),
        SamplingProfile::Jittered => {
            vec![0, 77, 181, 296, 394, 501, 627, 744, 851, 931, 1_000]
        }
        SamplingProfile::Bursty => {
            vec![0, 20, 40, 60, 240, 260, 280, 720, 740, 760, 1_000]
        }
    }
}

fn dropped_indices(profile: SamplingProfile) -> &'static [usize] {
    match profile {
        SamplingProfile::Regular => &[],
        SamplingProfile::Jittered => &[5],
        SamplingProfile::Bursty => &[2, 7],
    }
}

fn generate_observations(
    profile: SamplingProfile,
    knots: &[OracleKnot],
) -> Result<(Vec<CandidateObservation>, u64), &'static str> {
    if knots.len() < 2 {
        return Err("oracle requires at least two knots");
    }
    for pair in knots.windows(2) {
        if pair[1].time <= pair[0].time
            || !pair[0].target.is_finite()
            || !pair[1].target.is_finite()
        {
            return Err("oracle knots must be finite and strictly increasing");
        }
    }

    let schedule = scheduled_times(profile);
    let dropped = dropped_indices(profile);
    let mut observations = Vec::new();
    let mut max_delay = 0_u64;

    for (index, source_time) in schedule.iter().copied().enumerate() {
        if dropped.contains(&index) {
            continue;
        }
        let delay = match profile {
            SamplingProfile::Regular => 2 + (index as u64 % 3),
            SamplingProfile::Jittered => 2 + ((index as u64 * 7) % 13),
            SamplingProfile::Bursty => 1 + (index as u64 % 5),
        };
        let available_at = source_time
            .checked_add(delay)
            .ok_or("availability time overflow")?;
        max_delay = max_delay.max(delay);
        observations.push(CandidateObservation {
            event_id: index as u64,
            source_time,
            available_at,
            target: target_at(knots, source_time)?,
        });
    }

    for pair in observations.windows(2) {
        if pair[1].source_time <= pair[0].source_time {
            return Err("candidate source times must be strictly increasing");
        }
        if pair[1].available_at <= pair[0].available_at {
            return Err("candidate availability times must be strictly increasing");
        }
    }
    Ok((observations, max_delay))
}

fn fixture_fingerprint(
    profile: SamplingProfile,
    knots: &[OracleKnot],
    observations: &[CandidateObservation],
) -> String {
    // FNV-1a is a deterministic fixture identifier, not a security digest.
    let mut hash = 0xcbf29ce484222325_u64;
    let profile_tag = match profile {
        SamplingProfile::Regular => 0_u8,
        SamplingProfile::Jittered => 1_u8,
        SamplingProfile::Bursty => 2_u8,
    };
    let mut absorb = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    };
    absorb(&[profile_tag]);
    for knot in knots {
        absorb(&knot.time.to_le_bytes());
        absorb(&knot.target.to_bits().to_le_bytes());
    }
    for observation in observations {
        absorb(&observation.event_id.to_le_bytes());
        absorb(&observation.source_time.to_le_bytes());
        absorb(&observation.available_at.to_le_bytes());
        absorb(&observation.target.to_bits().to_le_bytes());
    }
    format!("fnv1a64:{hash:016x}")
}

fn advance_source_state(
    state: f64,
    previous: CandidateObservation,
    current: CandidateObservation,
    method: MethodSpec,
) -> Result<f64, &'static str> {
    let interval = (current.source_time - previous.source_time) as f64;
    if interval <= 0.0 {
        return Err("source observation interval must be positive");
    }
    match method.kind {
        SolverKind::ZeroOrderHoldExact => exact_linear_target_update(
            state,
            previous.target,
            previous.target,
            TAU,
            interval,
        ),
        SolverKind::LinearInterpolationExact => exact_linear_target_update(
            state,
            previous.target,
            current.target,
            TAU,
            interval,
        ),
        SolverKind::LinearEuler => euler_linear_target(
            state,
            previous.target,
            current.target,
            TAU,
            interval,
            method.substeps,
        ),
        SolverKind::LinearRungeKutta4 => rk4_linear_target(
            state,
            previous.target,
            current.target,
            TAU,
            interval,
            method.substeps,
        ),
    }
}

fn evaluate_method(
    observations: &[CandidateObservation],
    knots: &[OracleKnot],
    method: MethodSpec,
) -> Result<MethodResult, &'static str> {
    if observations.is_empty() {
        return Err("candidate observation stream is empty");
    }

    // This state remains on source/event time. The current observation is only
    // usable after available_at, so its reconstructed state is not emitted
    // until after the event-time update and a causal forward projection.
    let mut source_state = INITIAL_STATE;
    let mut source_max_error = 0.0_f64;
    let mut source_terminal_error = 0.0_f64;
    let mut availability_max_error = 0.0_f64;
    let mut availability_terminal_error = 0.0_f64;

    for (index, current) in observations.iter().copied().enumerate() {
        if index > 0 {
            let previous = observations[index - 1];
            source_state = advance_source_state(source_state, previous, current, method)?;
        }

        let source_reference =
            oracle_state_at(knots, INITIAL_STATE, TAU, current.source_time)?;
        let source_error = (source_state - source_reference).abs();
        source_max_error = source_max_error.max(source_error);
        source_terminal_error = source_error;

        // At arrival, project the reconstructed state from event time to
        // availability time using only the newest delivered input value.
        // The evaluator's oracle continues to follow hidden changes during
        // this interval; that difference is the freshness/latency penalty.
        let delay = (current.available_at - current.source_time) as f64;
        let state_at_availability = exact_linear_target_update(
            source_state,
            current.target,
            current.target,
            TAU,
            delay,
        )?;
        let availability_reference =
            oracle_state_at(knots, INITIAL_STATE, TAU, current.available_at)?;
        let availability_error = (state_at_availability - availability_reference).abs();
        availability_max_error = availability_max_error.max(availability_error);
        availability_terminal_error = availability_error;
    }

    let intervals = observations.len().saturating_sub(1) as u64;
    let rhs = method
        .rhs_evaluations_per_interval()
        .checked_mul(intervals)
        .ok_or("RHS evaluation counter overflow")?;
    let analytic = method
        .analytic_updates_per_interval()
        .checked_mul(intervals)
        .ok_or("analytic update counter overflow")?;

    Ok(MethodResult {
        method: method.name(),
        analytic_interval_updates: analytic,
        euler_rhs_evaluations: if method.kind == SolverKind::LinearEuler {
            rhs
        } else {
            0
        },
        rk4_rhs_evaluations: if method.kind == SolverKind::LinearRungeKutta4 {
            rhs
        } else {
            0
        },
        availability_projection_updates: observations.len() as u64,
        max_abs_source_time_error: source_max_error,
        terminal_abs_source_time_error: source_terminal_error,
        max_abs_availability_time_error: availability_max_error,
        terminal_abs_availability_time_error: availability_terminal_error,
    })
}

fn run_profile(profile: SamplingProfile) -> Result<TraceResult, &'static str> {
    let knots = oracle_knots();
    let (observations, max_delay) = generate_observations(profile, &knots)?;
    let scheduled = scheduled_times(profile).len() as u64;
    let dropped = dropped_indices(profile).len() as u64;
    let stale = observations
        .iter()
        .filter(|item| item.available_at - item.source_time > FRESHNESS_BUDGET_TICKS)
        .count() as u64;
    let late = observations
        .iter()
        .filter(|item| item.available_at - item.source_time > DECISION_DEADLINE_BUDGET_TICKS)
        .count() as u64;
    let methods = method_specs()
        .into_iter()
        .map(|method| evaluate_method(&observations, &knots, method))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(TraceResult {
        workload_version: "piecewise-linear-leaky-availability-v0.1".to_owned(),
        profile,
        time_unit: "synthetic logical ticks".to_owned(),
        fixture_fingerprint: fixture_fingerprint(profile, &knots, &observations),
        scheduled_events: scheduled,
        dropped_events: dropped,
        delivered_events: observations.len() as u64,
        freshness_budget_ticks: FRESHNESS_BUDGET_TICKS,
        decision_deadline_budget_ticks: DECISION_DEADLINE_BUDGET_TICKS,
        stale_outputs: stale,
        late_outputs: late,
        max_source_to_availability_delay: max_delay,
        // This reference assumes zero processing latency and emits exactly
        // when the final observation becomes available. A deployed system
        // must add its measured processing and queueing time separately.
        outputs_emitted_at_availability: observations.len() as u64,
        methods,
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    for profile in [
        SamplingProfile::Regular,
        SamplingProfile::Jittered,
        SamplingProfile::Bursty,
    ] {
        let result = run_profile(profile)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        println!("{}", serde_json::to_string_pretty(&result)?);
    }
    println!(
        "scope: availability-aligned scalar reference; no wall-clock, hardware, energy, or HDC-LTC claim"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedules_preserve_scheduled_dropped_and_delivered_counts() {
        for (profile, expected_scheduled, expected_dropped, expected_delivered) in [
            (SamplingProfile::Regular, 11, 0, 11),
            (SamplingProfile::Jittered, 11, 1, 10),
            (SamplingProfile::Bursty, 11, 2, 9),
        ] {
            let result = run_profile(profile).unwrap();
            assert_eq!(result.scheduled_events, expected_scheduled);
            assert_eq!(result.dropped_events, expected_dropped);
            assert_eq!(result.delivered_events, expected_delivered);
        }
    }

    #[test]
    fn fixed_nfe_sweeps_match_the_declared_rhs_budgets() {
        let knots = oracle_knots();
        let (observations, _) =
            generate_observations(SamplingProfile::Regular, &knots).unwrap();
        let intervals = observations.len() as u64 - 1;

        for (substeps, expected_nfe) in [(4, 4), (8, 8), (16, 16), (32, 32)] {
            let euler = evaluate_method(
                &observations,
                &knots,
                MethodSpec {
                    kind: SolverKind::LinearEuler,
                    substeps,
                },
            )
            .unwrap();
            assert_eq!(euler.euler_rhs_evaluations, intervals * expected_nfe);

            let rk4 = evaluate_method(
                &observations,
                &knots,
                MethodSpec {
                    kind: SolverKind::LinearRungeKutta4,
                    substeps: expected_nfe as usize / 4,
                },
            )
            .unwrap();
            assert_eq!(rk4.rk4_rhs_evaluations, intervals * expected_nfe);
        }
    }

    #[test]
    fn outputs_are_only_emitted_at_or_after_observation_availability() {
        let knots = oracle_knots();
        for profile in [
            SamplingProfile::Regular,
            SamplingProfile::Jittered,
            SamplingProfile::Bursty,
        ] {
            let (observations, _) = generate_observations(profile, &knots).unwrap();
            assert!(observations.iter().all(|item| item.available_at >= item.source_time));
            let result = run_profile(profile).unwrap();
            assert_eq!(
                result.outputs_emitted_at_availability,
                result.delivered_events
            );
        }
    }

    #[test]
    fn availability_error_charges_hidden_input_changes_after_source_time() {
        let knots = vec![
            OracleKnot {
                time: 0,
                target: 0.0,
            },
            OracleKnot {
                time: 10,
                target: 10.0,
            },
            OracleKnot {
                time: 15,
                target: 0.0,
            },
        ];
        let observations = vec![
            CandidateObservation {
                event_id: 0,
                source_time: 0,
                available_at: 0,
                target: 0.0,
            },
            CandidateObservation {
                event_id: 1,
                source_time: 10,
                available_at: 15,
                target: 10.0,
            },
        ];
        let result = evaluate_method(
            &observations,
            &knots,
            MethodSpec {
                kind: SolverKind::LinearInterpolationExact,
                substeps: 0,
            },
        )
        .unwrap();

        assert!(result.max_abs_source_time_error < 1.0e-8);
        assert!(result.max_abs_availability_time_error > 1.0e-3);
        assert!(
            result.max_abs_availability_time_error > result.max_abs_source_time_error
        );
    }

    #[test]
    fn delayed_jittered_outputs_are_separately_classified_as_stale_or_late() {
        let jittered = run_profile(SamplingProfile::Jittered).unwrap();
        let regular = run_profile(SamplingProfile::Regular).unwrap();
        assert!(jittered.stale_outputs > 0);
        assert!(jittered.late_outputs > 0);
        assert_eq!(regular.stale_outputs, 0);
        assert_eq!(regular.late_outputs, 0);
        assert!(jittered.stale_outputs <= jittered.delivered_events);
        assert!(jittered.late_outputs <= jittered.delivered_events);
    }

    #[test]
    fn all_methods_report_source_and_availability_errors_separately() {
        for profile in [
            SamplingProfile::Regular,
            SamplingProfile::Jittered,
            SamplingProfile::Bursty,
        ] {
            let result = run_profile(profile).unwrap();
            assert_eq!(result.methods.len(), 10);
            assert!(result.methods.iter().all(|method| {
                method.max_abs_source_time_error.is_finite()
                    && method.terminal_abs_source_time_error.is_finite()
                    && method.max_abs_availability_time_error.is_finite()
                    && method.terminal_abs_availability_time_error.is_finite()
                    && method.availability_projection_updates == result.delivered_events
            }));
        }
    }

    #[test]
    fn deterministic_fixture_serializes_identically() {
        for profile in [
            SamplingProfile::Regular,
            SamplingProfile::Jittered,
            SamplingProfile::Bursty,
        ] {
            let first = serde_json::to_string(&run_profile(profile).unwrap()).unwrap();
            let second = serde_json::to_string(&run_profile(profile).unwrap()).unwrap();
            assert_eq!(first, second);
        }
    }

    #[test]
    fn rejects_non_monotonic_or_non_finite_oracle_knots() {
        let mut knots = oracle_knots();
        knots[2].time = knots[1].time;
        assert_eq!(
            generate_observations(SamplingProfile::Regular, &knots),
            Err("oracle knots must be finite and strictly increasing")
        );

        let mut knots = oracle_knots();
        knots[1].target = f64::NAN;
        assert_eq!(
            generate_observations(SamplingProfile::Regular, &knots),
            Err("oracle knots must be finite and strictly increasing")
        );
    }

    #[test]
    fn fixed_nfe_method_names_expose_the_actual_function_evaluation_budget() {
        let names: Vec<String> = method_specs().into_iter().map(MethodSpec::name).collect();
        assert!(names.contains(&"linear_euler_nfe_4".to_owned()));
        assert!(names.contains(&"linear_euler_nfe_8".to_owned()));
        assert!(names.contains(&"linear_euler_nfe_16".to_owned()));
        assert!(names.contains(&"linear_euler_nfe_32".to_owned()));
        assert!(names.contains(&"linear_rk4_nfe_4".to_owned()));
        assert!(names.contains(&"linear_rk4_nfe_8".to_owned()));
        assert!(names.contains(&"linear_rk4_nfe_16".to_owned()));
        assert!(names.contains(&"linear_rk4_nfe_32".to_owned()));
    }
}