// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Initial-value ODE integration with the classic fourth-order Runge-Kutta
//! method — scalar and first-order systems. This consolidates the RK4 that
//! `symthaea-ecology` and the orbital-mechanics code each carried privately.

/// Integrate a scalar IVP `dy/dt = f(t, y)`, `y(t0) = y0`, over `steps` uniform
/// steps to `t1`. Returns `(t, y)` at each step (excluding the initial point).
pub fn rk4(
    f: impl Fn(f64, f64) -> f64,
    y0: f64,
    t0: f64,
    t1: f64,
    steps: usize,
) -> Vec<(f64, f64)> {
    let steps = steps.max(1);
    let h = (t1 - t0) / steps as f64;
    let mut out = Vec::with_capacity(steps);
    let (mut t, mut y) = (t0, y0);
    for _ in 0..steps {
        let k1 = f(t, y);
        let k2 = f(t + 0.5 * h, y + 0.5 * h * k1);
        let k3 = f(t + 0.5 * h, y + 0.5 * h * k2);
        let k4 = f(t + h, y + h * k3);
        y += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += h;
        out.push((t, y));
    }
    out
}

/// Integrate a first-order system `dy/dt = f(t, y)` where `y` is a state
/// vector. Returns `(t, y)` at each step.
pub fn rk4_system(
    f: impl Fn(f64, &[f64]) -> Vec<f64>,
    y0: Vec<f64>,
    t0: f64,
    t1: f64,
    steps: usize,
) -> Vec<(f64, Vec<f64>)> {
    let steps = steps.max(1);
    let h = (t1 - t0) / steps as f64;
    let add = |a: &[f64], b: &[f64], s: f64| -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x + s * y).collect()
    };
    let mut out = Vec::with_capacity(steps);
    let mut t = t0;
    let mut y = y0;
    for _ in 0..steps {
        let k1 = f(t, &y);
        let k2 = f(t + 0.5 * h, &add(&y, &k1, 0.5 * h));
        let k3 = f(t + 0.5 * h, &add(&y, &k2, 0.5 * h));
        let k4 = f(t + h, &add(&y, &k3, h));
        for i in 0..y.len() {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
        out.push((t, y.clone()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::E;

    #[test]
    fn exponential_growth() {
        // dy/dt = y, y(0)=1 → y(1) = e.
        let traj = rk4(|_, y| y, 1.0, 0.0, 1.0, 1000);
        let (_, y1) = *traj.last().unwrap();
        assert!((y1 - E).abs() < 1e-9, "{y1}");
    }

    #[test]
    fn harmonic_oscillator_system() {
        // y'' = -y as a system: [y, v]' = [v, -y], y(0)=1, v(0)=0.
        // At t = π the position should be -1.
        let traj = rk4_system(
            |_, s| vec![s[1], -s[0]],
            vec![1.0, 0.0],
            0.0,
            std::f64::consts::PI,
            10000,
        );
        let (_, last) = traj.last().unwrap();
        assert!((last[0] - (-1.0)).abs() < 1e-6, "{last:?}");
        assert!(last[1].abs() < 1e-6); // velocity ≈ 0
    }
}
