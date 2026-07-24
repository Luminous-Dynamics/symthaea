// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Discovers local Lyapunov functions for a damped oscillator and a
//! nonlinear (Van der Pol) system, and shows the honest negative result for
//! a marginally-stable (undamped) system where no certificate exists.
//!
//! Run: `cargo run -p symthaea-physics-bridge --example lyapunov_report`

use symthaea_physics_bridge::lyapunov::discover_local_lyapunov;

fn damped_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    vec![s[1], -s[0] - 0.3 * s[1]]
}
fn undamped_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    vec![s[1], -s[0]]
}
fn van_der_pol_stable_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let mu = -0.5;
    vec![s[1], mu * (1.0 - s[0] * s[0]) * s[1] - s[0]]
}
fn van_der_pol_unstable_rhs(s: &[f64], _t: f64) -> Vec<f64> {
    let mu = 0.5; // standard sign: origin is unstable, limit cycle attracts from outside
    vec![s[1], mu * (1.0 - s[0] * s[0]) * s[1] - s[0]]
}

fn report(name: &str, rhs: fn(&[f64], f64) -> Vec<f64>, radius: f64) {
    println!("\n=== {name} (sample radius {radius}) ===");
    match discover_local_lyapunov(rhs, &[0.0, 0.0], &["x", "v"], radius, 500, 42) {
        Some(cert) => {
            println!("  V(x,v) = {}", cert.v_expr);
            println!(
                "  P = [[{:.4}, {:.4}], [{:.4}, {:.4}]]",
                cert.p[(0, 0)],
                cert.p[(0, 1)],
                cert.p[(1, 0)],
                cert.p[(1, 1)]
            );
            println!(
                "  dV/dt <= 0 violations: {}/{} ({:.1}%), max violation = {:.3e}",
                (cert.violation_fraction * cert.samples_checked as f64).round() as usize,
                cert.samples_checked,
                cert.violation_fraction * 100.0,
                cert.max_violation
            );
        }
        None => {
            println!(
                "  No certificate -- equilibrium is not locally asymptotically stable \
                 (linearization is not Hurwitz)."
            );
        }
    }
}

fn main() {
    report(
        "Damped harmonic oscillator (linear, globally valid)",
        damped_rhs,
        3.0,
    );
    report(
        "Undamped harmonic oscillator (marginally stable center)",
        undamped_rhs,
        1.0,
    );
    report(
        "Van der Pol, mu=-0.5 (nonlinear, locally stable)",
        van_der_pol_stable_rhs,
        0.05,
    );
    report(
        "Van der Pol, mu=-0.5 (same system, larger radius -- watch violations appear)",
        van_der_pol_stable_rhs,
        2.0,
    );
    report(
        "Van der Pol, mu=+0.5 (standard sign -- origin unstable)",
        van_der_pol_unstable_rhs,
        0.05,
    );
}
