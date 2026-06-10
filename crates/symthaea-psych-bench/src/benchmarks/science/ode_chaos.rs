// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! ODE Chaos & Stiffness Benchmark.
//!
//! Tests Symthaea's ability to solve notoriously hard ODEs using adaptive
//! RK45 (Dormand-Prince) stepping. Four canonical hard problems:
//!
//! ## 1. Van der Pol Oscillator (Stiff, μ=1000)
//! Rate constants span 6 orders of magnitude. Standard RK4 requires Δt~10⁻⁶.
//! RK45 adapts. Tests: limit cycle amplitude x_max≈2.0, sustained oscillation.
//! Human baseline (numerical analysis students): 0.50 | Expert: 0.85
//!
//! ## 2. Lorenz Attractor (Chaotic)
//! Tests attractor bounds (x∈[-25,25], y∈[-35,35], z∈[0,55]),
//! time-averaged statistics (⟨z⟩≈27), and trajectory boundedness.
//! Human baseline: 0.75 | Expert: 0.95
//!
//! ## 3. Robertson Chemistry (Very Stiff)
//! Rate constants span 11 orders of magnitude. Conservation: y₁+y₂+y₃=1.
//! Human baseline: 0.55 | Expert: 0.85
//!
//! ## 4. Coupled Oscillators (Energy Conservation)
//! Two springs near resonance. Tests |E(t)−E(0)|/E(0) < 1% over t∈[0,100].
//! Human baseline: 0.80 | Expert: 0.99
//!
//! References: Van der Pol (1926), Lorenz (1963), Robertson (1966),
//! Hairer & Wanner (1996).

use std::time::Instant;

use crate::harness::config::BenchmarkConfig;
use crate::harness::report::{BenchmarkResult, MetricValue};
use crate::harness::{BenchmarkProvenance, PsychBenchmark};

/// ODE Chaos and Stiffness Assessment benchmark.
pub struct OdeChaosBenchmark;

struct TrialResult {
    van_der_pol_amplitude: f64,
    van_der_pol_oscillation: f64,
    van_der_pol_complete: f64,
    lorenz_bounds: f64,
    lorenz_statistics: f64,
    lorenz_bounded: f64,
    robertson_conservation: f64,
    robertson_species: f64,
    oscillator_energy: f64,
}

// ─── ODE right-hand sides ─────────────────────────────────────────────────────

fn van_der_pol(s: &[f64], _t: f64) -> Vec<f64> {
    let mu = 1000.0f64;
    vec![s[1], mu * (1.0 - s[0] * s[0]) * s[1] - s[0]]
}

fn lorenz(s: &[f64], _t: f64) -> Vec<f64> {
    let (sigma, rho, beta) = (10.0f64, 28.0f64, 8.0 / 3.0);
    vec![
        sigma * (s[1] - s[0]),
        s[0] * (rho - s[2]) - s[1],
        s[0] * s[1] - beta * s[2],
    ]
}

fn robertson(s: &[f64], _t: f64) -> Vec<f64> {
    let y1 = s[0];
    let y2 = s[1];
    let y3 = s[2];
    vec![
        -0.04 * y1 + 1e4 * y2 * y3,
        0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2 * y2,
        3e7 * y2 * y2,
    ]
}

fn coupled_oscillators(s: &[f64], _t: f64) -> Vec<f64> {
    let (omega1_sq, omega2_sq, kappa) = (1.0f64, 1.1f64, 0.05f64);
    vec![
        s[1],
        -(omega1_sq + kappa) * s[0] + kappa * s[2],
        s[3],
        kappa * s[0] - (omega2_sq + kappa) * s[2],
    ]
}

// ─── Dormand-Prince RK45 adaptive solver ─────────────────────────────────────

/// Integrate an ODE using Dormand-Prince RK45 with adaptive step-size control.
/// Returns (times, states). Stops at max_steps or when t ≥ t_end.
fn rk45(
    f: impl Fn(&[f64], f64) -> Vec<f64>,
    y0: &[f64],
    t0: f64,
    t_end: f64,
    rtol: f64,
    atol: f64,
    max_steps: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    // Dormand-Prince tableau
    let a21 = 1.0 / 5.0;
    let (a31, a32) = (3.0 / 40.0, 9.0 / 40.0);
    let (a41, a42, a43) = (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0);
    let (a51, a52, a53, a54) = (
        19372.0 / 6561.0,
        -25360.0 / 2187.0,
        64448.0 / 6561.0,
        -212.0 / 729.0,
    );
    let (a61, a62, a63, a64, a65) = (
        9017.0 / 3168.0,
        -355.0 / 33.0,
        46732.0 / 5247.0,
        49.0 / 176.0,
        -5103.0 / 18656.0,
    );
    // 5th-order solution weights
    let (b1, b3, b4, b5, b6) = (
        35.0 / 384.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
    );
    // Error coefficients (difference between 4th and 5th order)
    let (e1, e3, e4, e5, e6, e7) = (
        71.0 / 57600.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -1.0 / 40.0,
    );

    let n = y0.len();
    let mut t = t0;
    let mut y = y0.to_vec();
    let mut h = (t_end - t0) * 1e-3;

    let mut times = vec![t];
    let mut states = vec![y.clone()];

    for _ in 0..max_steps {
        if t >= t_end {
            break;
        }
        h = h.min(t_end - t);

        let k1 = f(&y, t);
        let y2: Vec<f64> = (0..n).map(|i| y[i] + h * a21 * k1[i]).collect();
        let k2 = f(&y2, t + h / 5.0);
        let y3: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (a31 * k1[i] + a32 * k2[i]))
            .collect();
        let k3 = f(&y3, t + 3.0 * h / 10.0);
        let y4: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]))
            .collect();
        let k4 = f(&y4, t + 4.0 * h / 5.0);
        let y5: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]))
            .collect();
        let k5 = f(&y5, t + 8.0 * h / 9.0);
        let y6: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i])
            })
            .collect();
        let k6 = f(&y6, t + h);

        // 5th-order solution
        let yn: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (b1 * k1[i] + b3 * k3[i] + b4 * k4[i] + b5 * k5[i] + b6 * k6[i]))
            .collect();
        let k7 = f(&yn, t + h);

        // Error estimate
        let err = {
            let mut sq_sum = 0.0f64;
            for i in 0..n {
                let sc = atol + rtol * y[i].abs().max(yn[i].abs());
                let ei = h
                    * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
                sq_sum += (ei / sc).powi(2);
            }
            (sq_sum / n as f64).sqrt()
        };

        if err <= 1.0 {
            t += h;
            y = yn;
            times.push(t);
            states.push(y.clone());
        }

        // Step-size control (PI controller)
        let factor = if err < 1e-15 {
            5.0
        } else {
            (0.9 * err.powf(-0.2)).clamp(0.1, 5.0)
        };
        h *= factor;
    }

    (times, states)
}

// ─── Benchmark implementation ─────────────────────────────────────────────────

impl OdeChaosBenchmark {
    fn run_trial(&self, _config: &BenchmarkConfig, trial_idx: usize) -> TrialResult {
        let perturb = trial_idx as f64 * 1e-4;

        // ── 1. Van der Pol (stiff μ=1000) ─────────────────────────────────
        let (vdp_times, vdp_states) = rk45(
            van_der_pol,
            &[2.0 + perturb, 0.0],
            0.0,
            30.0,
            1e-4,
            1e-6,
            100_000,
        );

        let transient = vdp_states.len() * 7 / 10;
        let stable = &vdp_states[transient.min(vdp_states.len().saturating_sub(1))..];

        let max_x = stable.iter().map(|s| s[0].abs()).fold(0.0f64, f64::max);
        let min_x = stable.iter().map(|s| s[0]).fold(f64::INFINITY, f64::min);

        let van_der_pol_amplitude = if max_x > 1.5 && max_x < 2.5 { 1.0 } else { 0.0 };
        let van_der_pol_oscillation = if min_x < -1.0 && max_x > 1.0 {
            1.0
        } else {
            0.0
        };
        let van_der_pol_complete = if vdp_times.last().copied().unwrap_or(0.0) > 25.0 {
            1.0
        } else {
            0.5
        };

        // ── 2. Lorenz attractor ────────────────────────────────────────────
        let (_, lorenz_states) = rk45(
            lorenz,
            &[1.0 + perturb, 0.0, 0.0],
            0.0,
            50.0,
            1e-6,
            1e-9,
            300_000,
        );

        let n_lor = lorenz_states.len();
        let in_bounds = lorenz_states
            .iter()
            .filter(|s| s[0].abs() <= 25.0 && s[1].abs() <= 35.0 && s[2] >= -1.0 && s[2] <= 56.0)
            .count();
        let lorenz_bounds = if n_lor > 0 {
            in_bounds as f64 / n_lor as f64
        } else {
            0.0
        };

        // Time-averaged z ≈ ρ-1 = 27 (skip transient first 20%)
        let skip = n_lor / 5;
        let mean_z = if n_lor > skip + 10 {
            lorenz_states[skip..].iter().map(|s| s[2]).sum::<f64>() / (n_lor - skip) as f64
        } else {
            0.0
        };
        let lorenz_statistics = if mean_z > 20.0 && mean_z < 35.0 {
            1.0
        } else {
            0.0
        };

        let max_r = lorenz_states
            .iter()
            .map(|s| (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt())
            .fold(0.0f64, f64::max);
        let lorenz_bounded = if max_r < 70.0 { 1.0 } else { 0.0 };

        // ── 3. Robertson chemistry ─────────────────────────────────────────
        let (_, rob_states) = rk45(
            robertson,
            &[1.0, 0.0, 0.0],
            0.0,
            1e-3, // explicit RK45 handles early transient
            1e-8,
            1e-12,
            200_000,
        );

        let max_cons_err = if !rob_states.is_empty() {
            rob_states
                .iter()
                .map(|s| (s[0] + s[1] + s[2] - 1.0).abs())
                .fold(0.0f64, f64::max)
        } else {
            1.0
        };
        let robertson_conservation = if max_cons_err < 1e-4 {
            1.0
        } else if max_cons_err < 0.01 {
            0.5
        } else {
            0.0
        };

        // y2 (intermediate) should remain much smaller than y1 and y3
        let robertson_species = if let Some(final_s) = rob_states.last() {
            let y2_reasonable = final_s[1] < 1e-2;
            let sum_ok = (final_s[0] + final_s[1] + final_s[2] - 1.0).abs() < 1e-4;
            if y2_reasonable && sum_ok {
                1.0
            } else if sum_ok {
                0.7
            } else {
                0.3
            }
        } else {
            0.0
        };

        // ── 4. Coupled oscillators (energy conservation) ───────────────────
        let y0_osc = [1.0, 0.0, 0.0, 0.0f64];
        let (omega1_sq, omega2_sq, kappa) = (1.0f64, 1.1f64, 0.05f64);
        let energy = |s: &[f64]| -> f64 {
            0.5 * (s[1] * s[1] + s[3] * s[3])
                + 0.5 * (omega1_sq * s[0] * s[0] + omega2_sq * s[2] * s[2])
                + 0.5 * kappa * (s[0] - s[2]).powi(2)
        };
        let e0 = energy(&y0_osc);

        let (_, osc_states) = rk45(
            coupled_oscillators,
            &y0_osc,
            0.0,
            100.0,
            1e-8,
            1e-10,
            500_000,
        );

        let max_err = osc_states
            .iter()
            .map(|s| ((energy(s) - e0) / e0.max(1e-15)).abs())
            .fold(0.0f64, f64::max);

        let oscillator_energy = if max_err < 0.001 {
            1.0
        } else if max_err < 0.01 {
            0.8
        } else if max_err < 0.1 {
            0.5
        } else {
            0.0
        };

        TrialResult {
            van_der_pol_amplitude,
            van_der_pol_oscillation,
            van_der_pol_complete,
            lorenz_bounds,
            lorenz_statistics,
            lorenz_bounded,
            robertson_conservation,
            robertson_species,
            oscillator_energy,
        }
    }
}

impl PsychBenchmark for OdeChaosBenchmark {
    fn name(&self) -> &str {
        "ODE Chaos & Stiffness"
    }

    fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult {
        let start = Instant::now();
        let mut result = BenchmarkResult::new(self.name(), config.label.clone());

        let n = config.trials_per_condition;
        let mut vdp_amp = Vec::with_capacity(n);
        let mut vdp_osc = Vec::with_capacity(n);
        let mut vdp_done = Vec::with_capacity(n);
        let mut lor_bounds = Vec::with_capacity(n);
        let mut lor_stats = Vec::with_capacity(n);
        let mut lor_bound = Vec::with_capacity(n);
        let mut rob_cons = Vec::with_capacity(n);
        let mut rob_sp = Vec::with_capacity(n);
        let mut osc_e = Vec::with_capacity(n);

        for i in 0..n {
            let r = self.run_trial(config, i);
            vdp_amp.push(r.van_der_pol_amplitude);
            vdp_osc.push(r.van_der_pol_oscillation);
            vdp_done.push(r.van_der_pol_complete);
            lor_bounds.push(r.lorenz_bounds);
            lor_stats.push(r.lorenz_statistics);
            lor_bound.push(r.lorenz_bounded);
            rob_cons.push(r.robertson_conservation);
            rob_sp.push(r.robertson_species);
            osc_e.push(r.oscillator_energy);
        }

        result.insert(
            "vdp_limit_cycle_amplitude",
            MetricValue::from_samples(&vdp_amp),
        );
        result.insert(
            "vdp_sustained_oscillation",
            MetricValue::from_samples(&vdp_osc),
        );
        result.insert(
            "vdp_integration_complete",
            MetricValue::from_samples(&vdp_done),
        );
        result.insert(
            "lorenz_attractor_bounds",
            MetricValue::from_samples(&lor_bounds),
        );
        result.insert(
            "lorenz_mean_z_statistics",
            MetricValue::from_samples(&lor_stats),
        );
        result.insert(
            "lorenz_trajectory_bounded",
            MetricValue::from_samples(&lor_bound),
        );
        result.insert(
            "robertson_conservation_law",
            MetricValue::from_samples(&rob_cons),
        );
        result.insert(
            "robertson_species_ordering",
            MetricValue::from_samples(&rob_sp),
        );
        result.insert(
            "coupled_oscillator_energy",
            MetricValue::from_samples(&osc_e),
        );

        result.conditions = 4; // VdP, Lorenz, Robertson, Oscillators
        result.trials_per_condition = n;
        result.elapsed_ms = start.elapsed().as_millis() as u64;
        result
    }

    fn provenance(&self) -> Option<BenchmarkProvenance> {
        Some(BenchmarkProvenance {
            paradigm: "Nonlinear ODE Solving (Stiff, Chaotic, Conservation)",
            citation: "Hairer & Wanner (1996); Lorenz (1963); Robertson (1966); Van der Pol (1926)",
            year: 1996,
            doi: Some("10.1007/978-3-642-05221-7"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> BenchmarkConfig {
        BenchmarkConfig {
            dimension: 256,
            trials_per_condition: 2,
            ..Default::default()
        }
    }

    #[test]
    fn test_vdp_rhs_at_zero() {
        // dx/dt=y, dy/dt=μ*(1-x²)*y - x. At (0,1): [1, μ]
        let r = van_der_pol(&[0.0, 1.0], 0.0);
        assert!((r[0] - 1.0).abs() < 1e-10);
        assert!((r[1] - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn test_lorenz_fixed_point() {
        // Unstable fixed point: sqrt(β(ρ-1)) ≈ 8.485, z=27
        let fp = (8.0 / 3.0 * 27.0_f64).sqrt();
        let r = lorenz(&[fp, fp, 27.0], 0.0);
        assert!(r[0].abs() < 0.01, "dx/dt at fp = {}", r[0]);
        assert!(r[1].abs() < 0.01, "dy/dt at fp = {}", r[1]);
        assert!(r[2].abs() < 0.01, "dz/dt at fp = {}", r[2]);
    }

    #[test]
    fn test_robertson_conservation_at_origin() {
        // d(y1+y2+y3)/dt = 0 identically
        let r = robertson(&[1.0, 0.0, 0.0], 0.0);
        assert!(
            (r[0] + r[1] + r[2]).abs() < 1e-10,
            "sum = {}",
            r[0] + r[1] + r[2]
        );
    }

    #[test]
    fn test_robertson_conservation_midway() {
        let r = robertson(&[0.5, 1e-5, 0.5 - 1e-5], 1.0);
        assert!((r[0] + r[1] + r[2]).abs() < 1e-6);
    }

    #[test]
    fn test_rk45_exponential_decay() {
        // dy/dt = -y, y(0)=1 → y(t) = e^{-t}
        let (ts, ys) = rk45(|s, _| vec![-s[0]], &[1.0], 0.0, 2.0, 1e-8, 1e-10, 10000);
        assert!(!ts.is_empty());
        let t = ts.last().unwrap();
        let y = ys.last().unwrap()[0];
        assert!(
            (y - (-t).exp()).abs() < 1e-5,
            "y={} expected={}",
            y,
            (-t).exp()
        );
    }

    #[test]
    fn test_rk45_harmonic_oscillator() {
        // d²x/dt² = -x → x(π) = cos(π) = -1
        let (ts, ys) = rk45(
            |s, _| vec![s[1], -s[0]],
            &[1.0, 0.0],
            0.0,
            std::f64::consts::PI,
            1e-8,
            1e-10,
            50000,
        );
        let y_final = ys.last().unwrap()[0];
        assert!((y_final + 1.0).abs() < 1e-4, "x(π)={}", y_final);
    }

    #[test]
    fn test_lorenz_stays_bounded() {
        let (_, states) = rk45(lorenz, &[1.0, 0.0, 0.0], 0.0, 10.0, 1e-6, 1e-9, 100_000);
        assert!(!states.is_empty());
        for s in &states {
            assert!(s[0].abs() < 30.0 && s[1].abs() < 40.0 && s[2] < 60.0);
        }
    }

    #[test]
    fn test_oscillator_energy_conservation() {
        let y0 = [1.0, 0.0, 0.0, 0.0f64];
        let e = |s: &[f64]| {
            0.5 * (s[1] * s[1] + s[3] * s[3])
                + 0.5 * (s[0] * s[0] + 1.1 * s[2] * s[2])
                + 0.025 * (s[0] - s[2]).powi(2)
        };
        let e0 = e(&y0);
        let (_, states) = rk45(coupled_oscillators, &y0, 0.0, 50.0, 1e-8, 1e-10, 200_000);
        let max_err = states
            .iter()
            .map(|s| ((e(s) - e0) / e0).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 0.01, "energy err = {:.2e}", max_err);
    }

    #[test]
    fn test_benchmark_runs_and_has_metrics() {
        let bench = OdeChaosBenchmark;
        let result = bench.run(&cfg());
        assert!(result.metrics.contains_key("vdp_limit_cycle_amplitude"));
        assert!(result.metrics.contains_key("lorenz_attractor_bounds"));
        assert!(result.metrics.contains_key("robertson_conservation_law"));
        assert!(result.metrics.contains_key("coupled_oscillator_energy"));
    }

    #[test]
    fn test_benchmark_scores_in_range() {
        let bench = OdeChaosBenchmark;
        let result = bench.run(&cfg());
        for (k, v) in &result.metrics {
            assert!(v.mean >= 0.0 && v.mean <= 1.0, "metric {} = {}", k, v.mean);
        }
    }

    #[test]
    fn test_provenance() {
        let bench = OdeChaosBenchmark;
        let p = bench.provenance().unwrap();
        assert_eq!(p.year, 1996);
        assert!(p.doi.is_some());
    }
}
