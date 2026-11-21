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
    println!("scope: scalar analytic reference only; not HDC-LTC qualification");
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
