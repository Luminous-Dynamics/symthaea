// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Research-only analytic reference for a scalar leaky system driven by a
//! piecewise-linear target.
//!
//! This is an isolated numerical control for NEURO-SILICON-006. It is not the
//! coupled HDC-LTC neuron, does not change production evolution, and makes no
//! hardware or performance claim.
//!
//! From the repository root:
//!   cargo test --locked -p symthaea-silicon --example hdc_ltc_linear_input_reference
//!   cargo run --locked -p symthaea-silicon --example hdc_ltc_linear_input_reference

use std::error::Error;
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReferenceError {
    InvalidInterval,
    InvalidTimeConstant,
    InvalidCurvatureBound,
    NonFiniteInput,
    NonFiniteResult,
}

impl Display for ReferenceError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Error for ReferenceError {}

/// Computes 1 - exp(-x) for non-negative x without cancellation near zero.
#[inline]
fn one_minus_exp_neg(x: f64) -> f64 {
    -(-x).exp_m1()
}

/// Computes the exact endpoint state for
///
///     tau * dz/dt = g(t) - z
///
/// when g(t) is linearly interpolated from g0 to g1 over [0, h].
///
/// The result is exact for this isolated scalar linear system and this
/// piecewise-linear target. It is not an exact solution of coupled nonlinear
/// HDC-LTC dynamics.
fn linear_target_update(
    z0: f64,
    g0: f64,
    g1: f64,
    tau: f64,
    h: f64,
) -> Result<f64, ReferenceError> {
    if !h.is_finite() || h < 0.0 {
        return Err(ReferenceError::InvalidInterval);
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(ReferenceError::InvalidTimeConstant);
    }
    if !z0.is_finite() || !g0.is_finite() || !g1.is_finite() {
        return Err(ReferenceError::NonFiniteInput);
    }
    if h == 0.0 {
        return Ok(z0);
    }

    let x = h / tau;
    let decay = (-x).exp();
    let one_minus = one_minus_exp_neg(x);

    // h - tau * (1 - exp(-h/tau)) loses precision for small h/tau.
    // For x < 1e-3, use:
    //   tau * (x - 1 + exp(-x))
    // = h*x*(1/2 - x/6 + x^2/24 - x^3/120 + ...).
    let correction_over_h = if x < 1.0e-3 {
        let polynomial = 0.5
            + x * (-1.0 / 6.0
                + x * (1.0 / 24.0
                    + x * (-1.0 / 120.0
                        + x * (1.0 / 720.0
                            + x * (-1.0 / 5040.0 + x * (1.0 / 40320.0))))));
        x * polynomial
    } else {
        // This branch remains well-behaved for very large x, including
        // h/tau overflowing to +infinity.
        let correction = h - tau * one_minus;
        correction / h
    };

    let result = decay * z0 + one_minus * g0 + (g1 - g0) * correction_over_h;
    if result.is_finite() {
        Ok(result)
    } else {
        Err(ReferenceError::NonFiniteResult)
    }
}

/// Local endpoint-error bound when the true target g is twice differentiable
/// and |g''(t)| <= curvature_bound throughout the interval.
///
/// Linear interpolation has pointwise error <= M*h^2/8. The leaky-system
/// convolution kernel integrates to (1 - exp(-h/tau)), giving the bound below.
///
/// This applies only to the isolated leaky system and a valid hard curvature
/// bound. An empirical curvature estimate is not a formal certificate.
fn linear_interpolation_error_bound(
    curvature_bound: f64,
    tau: f64,
    h: f64,
) -> Result<f64, ReferenceError> {
    if !h.is_finite() || h < 0.0 {
        return Err(ReferenceError::InvalidInterval);
    }
    if !tau.is_finite() || tau <= 0.0 {
        return Err(ReferenceError::InvalidTimeConstant);
    }
    if !curvature_bound.is_finite() || curvature_bound < 0.0 {
        return Err(ReferenceError::InvalidCurvatureBound);
    }
    if h == 0.0 || curvature_bound == 0.0 {
        return Ok(0.0);
    }

    let x = h / tau;
    let bound = (curvature_bound * h * h / 8.0) * one_minus_exp_neg(x);
    if bound.is_finite() {
        Ok(bound)
    } else {
        Err(ReferenceError::NonFiniteResult)
    }
}

fn rk4_linear_target(z0: f64, g0: f64, g1: f64, tau: f64, h: f64, steps: usize) -> f64 {
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
    state
}

fn rk4_quadratic_target(
    z0: f64,
    a: f64,
    b: f64,
    c: f64,
    tau: f64,
    h: f64,
    steps: usize,
) -> f64 {
    let step = h / steps as f64;
    let target = |time: f64| a + b * time + 0.5 * c * time * time;
    let derivative = |time: f64, state: f64| (target(time) - state) / tau;

    let mut state = z0;
    for index in 0..steps {
        let time = index as f64 * step;
        let k1 = derivative(time, state);
        let k2 = derivative(time + step / 2.0, state + step * k1 / 2.0);
        let k3 = derivative(time + step / 2.0, state + step * k2 / 2.0);
        let k4 = derivative(time + step, state + step * k3);
        state += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
    }
    state
}


// ---------------------------------------------------------------------------
// Deterministic irregular-input benchmark fixture.
// ---------------------------------------------------------------------------
//
// The oracle knots are evaluator-only. Candidate methods receive only the
// delivered observations, in availability order. When a new observation
// arrives, a method may retrospectively evolve its latent state over the
// preceding source-time interval; the result is usable only after that
// observation's availability time. This is causal at decision time, but not
// zero-latency. Source-to-availability delay is reported separately.
//
// This fixture measures numerical work counts (RHS evaluations / analytic
// interval updates). It does not measure wall-clock time, DRAM traffic or joules.

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct MethodResult {
    method: String,
    analytic_interval_updates: u64,
    euler_rhs_evaluations: u64,
    rk4_rhs_evaluations: u64,
    max_abs_state_error: f64,
    terminal_abs_state_error: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TraceResult {
    workload_version: String,
    time_unit: String,
    trace_fingerprint: String,
    profile: SamplingProfile,
    scheduled_events: u64,
    dropped_events: u64,
    delivered_events: u64,
    max_source_to_availability_delay: u64,
    methods: Vec<MethodResult>,
}

fn oracle_knots() -> Vec<OracleKnot> {
    vec![
        OracleKnot { time: 0, target: 0.0 },
        OracleKnot { time: 115, target: 0.85 },
        OracleKnot { time: 260, target: -0.45 },
        OracleKnot { time: 410, target: 1.10 },
        OracleKnot { time: 590, target: -0.80 },
        OracleKnot { time: 780, target: 0.35 },
        OracleKnot { time: 1_000, target: 0.65 },
    ]
}

/// Evaluator-only target value on the frozen piecewise-linear oracle signal.
fn oracle_target_at(knots: &[OracleKnot], time: u64) -> Result<f64, &'static str> {
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

/// Independent fine-step RK4 evaluator reference for the isolated leaky
/// scalar system, integrated across every hidden oracle knot up to time.
/// Candidate analytic-update methods do not call this solver, avoiding a
/// same-formula candidate/oracle comparison.
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
        let g0 = oracle_target_at(knots, left.time)?;
        let g1 = oracle_target_at(knots, end)?;
        let h = (end - left.time) as f64;
        state = rk4_linear_target(state, g0, g1, tau, h, 4_096);
        if !state.is_finite() {
            return Err("fine-step RK4 oracle produced non-finite state");
        }
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

/// Build a candidate-visible observation stream. The hidden oracle knots are
/// used only by the evaluator. Delivery delay is deterministic and bounded.
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
            target: oracle_target_at(knots, source_time)?,
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

fn euler_linear_target(
    z0: f64,
    g0: f64,
    g1: f64,
    tau: f64,
    interval: f64,
    steps: usize,
) -> f64 {
    let step = interval / steps as f64;
    let mut state = z0;
    for index in 0..steps {
        let local_time = index as f64 * step;
        let target = g0 + (g1 - g0) * (local_time / interval);
        state += step * (target - state) / tau;
    }
    state
}

fn trace_fingerprint(
    profile: SamplingProfile,
    knots: &[OracleKnot],
    observations: &[CandidateObservation],
) -> String {
    // FNV-1a is used only as a stable fixture fingerprint, not a cryptographic
    // integrity digest or an evidence-authentication mechanism.
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
    for event in observations {
        absorb(&event.event_id.to_le_bytes());
        absorb(&event.source_time.to_le_bytes());
        absorb(&event.available_at.to_le_bytes());
        absorb(&event.target.to_bits().to_le_bytes());
    }
    format!("fnv1a64:{hash:016x}")
}

fn run_trace(profile: SamplingProfile) -> Result<TraceResult, &'static str> {
    const INITIAL_STATE: f64 = -0.25;
    const TAU: f64 = 140.0;
    const EULER_STEPS_PER_INTERVAL: usize = 4;
    const RK4_STEPS_PER_INTERVAL: usize = 4;

    let knots = oracle_knots();
    let (observations, max_delay) = generate_observations(profile, &knots)?;
    if observations.len() < 2 {
        return Err("at least two observations are required");
    }

    let mut z_zoh = INITIAL_STATE;
    let mut z_linear_exact = INITIAL_STATE;
    let mut z_euler = INITIAL_STATE;
    let mut z_rk4 = INITIAL_STATE;
    let mut errors = [0.0_f64; 4];
    let mut euler_evaluations = 0_u64;
    let mut rk4_evaluations = 0_u64;

    for pair in observations.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        let interval = (current.source_time - previous.source_time) as f64;
        if interval <= 0.0 {
            return Err("candidate observation interval must be positive");
        }

        // ZOH control: exact integration of the held previous observation.
        z_zoh = linear_target_update(
            z_zoh,
            previous.target,
            previous.target,
            TAU,
            interval,
        )
        .map_err(|_| "ZOH reference update failed")?;

        // Piecewise-linear interpolation using the two observations that are
        // now both available. This state is publishable only at current.available_at.
        z_linear_exact = linear_target_update(
            z_linear_exact,
            previous.target,
            current.target,
            TAU,
            interval,
        )
        .map_err(|_| "linear exact update failed")?;

        z_euler = euler_linear_target(
            z_euler,
            previous.target,
            current.target,
            TAU,
            interval,
            EULER_STEPS_PER_INTERVAL,
        );
        euler_evaluations = euler_evaluations
            .checked_add(EULER_STEPS_PER_INTERVAL as u64)
            .ok_or("Euler work counter overflow")?;

        z_rk4 = rk4_linear_target(
            z_rk4,
            previous.target,
            current.target,
            TAU,
            interval,
            RK4_STEPS_PER_INTERVAL,
        );
        rk4_evaluations = rk4_evaluations
            .checked_add((4 * RK4_STEPS_PER_INTERVAL) as u64)
            .ok_or("RK4 work counter overflow")?;

        let reference = oracle_state_at(&knots, INITIAL_STATE, TAU, current.source_time)?;
        errors[0] = errors[0].max((z_zoh - reference).abs());
        errors[1] = errors[1].max((z_linear_exact - reference).abs());
        errors[2] = errors[2].max((z_euler - reference).abs());
        errors[3] = errors[3].max((z_rk4 - reference).abs());
    }

    let last = *observations.last().ok_or("observation stream empty")?;
    let final_reference = oracle_state_at(&knots, INITIAL_STATE, TAU, last.source_time)?;
    let terminal_errors = [
        (z_zoh - final_reference).abs(),
        (z_linear_exact - final_reference).abs(),
        (z_euler - final_reference).abs(),
        (z_rk4 - final_reference).abs(),
    ];
    let scheduled = scheduled_times(profile).len() as u64;

    Ok(TraceResult {
        workload_version: "piecewise-linear-leaky-reference-v0.1".to_owned(),
        time_unit: "synthetic logical ticks".to_owned(),
        trace_fingerprint: trace_fingerprint(profile, &knots, &observations),
        profile,
        scheduled_events: scheduled,
        dropped_events: dropped_indices(profile).len() as u64,
        delivered_events: observations.len() as u64,
        max_source_to_availability_delay: max_delay,
        methods: vec![
            MethodResult {
                method: "zero_order_hold_exact".to_owned(),
                analytic_interval_updates: (observations.len() - 1) as u64,
                euler_rhs_evaluations: 0,
                rk4_rhs_evaluations: 0,
                max_abs_state_error: errors[0],
                terminal_abs_state_error: terminal_errors[0],
            },
            MethodResult {
                method: "linear_interpolation_exact_scalar".to_owned(),
                analytic_interval_updates: (observations.len() - 1) as u64,
                euler_rhs_evaluations: 0,
                rk4_rhs_evaluations: 0,
                max_abs_state_error: errors[1],
                terminal_abs_state_error: terminal_errors[1],
            },
            MethodResult {
                method: format!("linear_interpolation_euler_{EULER_STEPS_PER_INTERVAL}"),
                analytic_interval_updates: 0,
                euler_rhs_evaluations: euler_evaluations,
                rk4_rhs_evaluations: 0,
                max_abs_state_error: errors[2],
                terminal_abs_state_error: terminal_errors[2],
            },
            MethodResult {
                method: format!("linear_interpolation_rk4_{RK4_STEPS_PER_INTERVAL}"),
                analytic_interval_updates: 0,
                euler_rhs_evaluations: 0,
                rk4_rhs_evaluations: rk4_evaluations,
                max_abs_state_error: errors[3],
                terminal_abs_state_error: terminal_errors[3],
            },
        ],
    })
}

fn main() -> Result<(), Box<dyn Error>> {
    let z0 = -0.4;
    let g0 = 0.2;
    let g1 = 1.1;
    let tau = 0.7;
    let h = 0.5;
    let result = linear_target_update(z0, g0, g1, tau, h)?;
    let bound = linear_interpolation_error_bound(0.8, tau, h)?;
    println!("isolated leaky linear-target endpoint: {result:.12}");
    println!("illustrative bound for curvature <= 0.8: {bound:.12}");
    println!("deterministic irregular-input reference profiles:");
    for profile in [
        SamplingProfile::Regular,
        SamplingProfile::Jittered,
        SamplingProfile::Bursty,
    ] {
        let result = run_trace(profile)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        println!("{}", serde_json::to_string_pretty(&result)?);
    }
    println!("scope: scalar analytic reference and synthetic input fixture only; not HDC-LTC qualification");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_target_reduces_to_zero_order_hold() {
        let z0 = -1.25;
        let target = 0.75;
        for (tau, h) in [
            (1.0, 1.0e-8),
            (0.1, 1.0e-4),
            (0.7, 0.4),
            (2.0, 9.0),
            (0.01, 3.0),
        ] {
            let actual = linear_target_update(z0, target, target, tau, h).unwrap();
            let expected = (-h / tau).exp() * z0
                + one_minus_exp_neg(h / tau) * target;
            assert!(
                (actual - expected).abs() <= 2.0e-14 * expected.abs().max(1.0),
                "constant-input mismatch for tau={tau}, h={h}: {actual} vs {expected}"
            );
        }
    }

    #[test]
    fn linear_target_formula_matches_fine_rk4_reference() {
        let cases = [
            (0.7, 0.6, -2.0, -1.0, 3.0),
            (2.3, 6.9, 0.5, 2.0, -4.0),
            (0.05, 0.1, 8.0, -3.0, 0.2),
            (1.5, 0.015, -0.7, 1.3, 0.1),
            (3.0, 1.5, 4.0, -5.0, 7.0),
        ];

        for (tau, h, z0, g0, g1) in cases {
            let actual = linear_target_update(z0, g0, g1, tau, h).unwrap();
            let reference = rk4_linear_target(z0, g0, g1, tau, h, 20_000);
            let error = (actual - reference).abs();
            assert!(
                error < 2.0e-10,
                "tau={tau}, h={h}, error={error}, actual={actual}, reference={reference}"
            );
        }
    }

    #[test]
    fn quadratic_target_error_is_bounded_by_curvature_envelope() {
        let cases = [
            (0.7, 0.9, 0.4, -0.2, 1.5),
            (2.0, 1.25, -1.0, 0.8, -0.7),
            (0.15, 0.3, 2.0, -1.2, 0.9),
        ];

        for (tau, h, a, b, c) in cases {
            let z0 = -0.3;
            let g0 = a;
            let g1 = a + b * h + 0.5 * c * h * h;

            let linear = linear_target_update(z0, g0, g1, tau, h).unwrap();
            let reference = rk4_quadratic_target(z0, a, b, c, tau, h, 30_000);
            let observed_error = (linear - reference).abs();
            let bound = linear_interpolation_error_bound(c.abs(), tau, h).unwrap();

            assert!(
                observed_error <= bound + 3.0e-10,
                "observed error {observed_error} exceeds bound {bound} (tau={tau}, h={h}, c={c})"
            );
        }
    }

    #[test]
    fn zero_interval_is_an_exact_noop() {
        assert_eq!(linear_target_update(3.25, -2.0, 4.0, 0.5, 0.0).unwrap(), 3.25);
        assert_eq!(linear_interpolation_error_bound(1.0, 0.5, 0.0).unwrap(), 0.0);
    }

    #[test]
    fn rejects_invalid_or_non_finite_contract_inputs() {
        assert_eq!(
            linear_target_update(0.0, 1.0, 1.0, 0.0, 1.0),
            Err(ReferenceError::InvalidTimeConstant)
        );
        assert_eq!(
            linear_target_update(0.0, 1.0, 1.0, 1.0, -1.0),
            Err(ReferenceError::InvalidInterval)
        );
        assert_eq!(
            linear_target_update(0.0, f64::NAN, 1.0, 1.0, 1.0),
            Err(ReferenceError::NonFiniteInput)
        );
        assert_eq!(
            linear_interpolation_error_bound(-1.0, 1.0, 1.0),
            Err(ReferenceError::InvalidCurvatureBound)
        );
    }

    #[test]
    fn deterministic_schedules_cover_regular_jittered_and_bursty_inputs() {
        let knots = oracle_knots();
        for profile in [
            SamplingProfile::Regular,
            SamplingProfile::Jittered,
            SamplingProfile::Bursty,
        ] {
            let (observations, _) = generate_observations(profile, &knots).unwrap();
            assert!(observations.len() >= 2);
            for pair in observations.windows(2) {
                assert!(pair[1].source_time > pair[0].source_time);
                assert!(pair[1].available_at > pair[0].available_at);
            }
        }
        assert_eq!(generate_observations(SamplingProfile::Regular, &knots).unwrap().0.len(), 11);
        assert_eq!(generate_observations(SamplingProfile::Jittered, &knots).unwrap().0.len(), 10);
        assert_eq!(generate_observations(SamplingProfile::Bursty, &knots).unwrap().0.len(), 9);
    }

    #[test]
    fn candidate_receives_only_delivered_observations_and_missing_events_remain_counted() {
        let knots = oracle_knots();
        let (jittered, _) = generate_observations(SamplingProfile::Jittered, &knots).unwrap();
        let (bursty, _) = generate_observations(SamplingProfile::Bursty, &knots).unwrap();
        assert_eq!(jittered.len() as u64 + 1, scheduled_times(SamplingProfile::Jittered).len() as u64);
        assert_eq!(bursty.len() as u64 + 2, scheduled_times(SamplingProfile::Bursty).len() as u64);
        assert!(jittered.iter().all(|event| event.available_at >= event.source_time));
    }

    #[test]
    fn trace_harness_reports_distinct_solver_work_and_finite_errors() {
        for profile in [
            SamplingProfile::Regular,
            SamplingProfile::Jittered,
            SamplingProfile::Bursty,
        ] {
            let result = run_trace(profile).unwrap();
            let intervals = result.delivered_events - 1;
            assert_eq!(result.methods[0].analytic_interval_updates, intervals);
            assert_eq!(result.methods[1].analytic_interval_updates, intervals);
            assert_eq!(result.methods[2].euler_rhs_evaluations, intervals * 4);
            assert_eq!(result.methods[3].rk4_rhs_evaluations, intervals * 16);
            assert!(result.methods.iter().all(|method| {
                method.max_abs_state_error.is_finite()
                    && method.terminal_abs_state_error.is_finite()
            }));
        }
    }

    #[test]
    fn linear_interpolation_improves_error_over_zoh_at_equal_interval_update_count() {
        for profile in [
            SamplingProfile::Regular,
            SamplingProfile::Jittered,
            SamplingProfile::Bursty,
        ] {
            let result = run_trace(profile).unwrap();
            let zoh = &result.methods[0];
            let linear = &result.methods[1];
            assert_eq!(
                zoh.analytic_interval_updates,
                linear.analytic_interval_updates
            );
            assert_eq!(zoh.analytic_interval_updates, result.delivered_events - 1);
            assert!(
                linear.max_abs_state_error < zoh.max_abs_state_error,
                "profile {profile:?}: linear {} vs ZOH {}",
                linear.max_abs_state_error,
                zoh.max_abs_state_error
            );
        }
    }

    #[test]
    fn deterministic_trace_profiles_serialize_identically() {
        for profile in [
            SamplingProfile::Regular,
            SamplingProfile::Jittered,
            SamplingProfile::Bursty,
        ] {
            let result = run_trace(profile).unwrap();
            let first = serde_json::to_string(&result).unwrap();
            let second = serde_json::to_string(&run_trace(profile).unwrap()).unwrap();
            assert_eq!(first, second);
            assert!(result.trace_fingerprint.starts_with("fnv1a64:"));
            assert_eq!(result.time_unit, "synthetic logical ticks");
        }
    }

    #[test]
    fn rejects_nonmonotonic_or_invalid_oracle_knots() {
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
    fn small_interval_arithmetic_remains_finite_and_continuous() {
        let z0 = 0.25;
        let g0 = -0.5;
        let g1 = 0.75;
        let tau = 1.0;
        let mut previous = z0;

        for h in [1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4] {
            let actual = linear_target_update(z0, g0, g1, tau, h).unwrap();
            assert!(actual.is_finite());
            assert!((actual - z0).abs() < 2.0e-4);
            assert!((actual - previous).abs() < 2.0e-4);
            previous = actual;
        }
    }
}
