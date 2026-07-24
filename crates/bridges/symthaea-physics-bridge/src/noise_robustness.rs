// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Noise-robustness benchmark for the autonomous-invariant discovery pipeline.
//!
//! Every Ramanujan Protocol showcase example runs on noiseless,
//! numerically-integrated trajectories. Real experimental or observational
//! data never looks like that. This module measures what actually happens
//! to the discovery pipeline's core scoring mechanism when the input is
//! corrupted by realistic measurement noise.
//!
//! ## What this does and does not test
//!
//! The autonomous discovery pipeline (`discover_invariants_autonomous` in
//! `symthaea_core::hdc::conjecture_engine`) always re-simulates its own
//! trajectory internally from the supplied RHS function — there is no public
//! entry point that accepts externally-generated (e.g. noisy) trajectory
//! data. Rather than invasively refactor that ~450-line, heavily-used
//! function to add one, this module tests the load-bearing piece directly:
//! [`lie_derivative_variance`], the exact fitness function the real GP
//! search uses to rank candidates (now `pub`, not `pub(crate)`, precisely so
//! this benchmark scores against the *real* metric, not a reimplementation
//! that could quietly diverge from it).
//!
//! The question this answers: **does the true invariant still score better
//! than plausible-but-wrong decoys once the trajectory is noisy?** This is a
//! *necessary* condition for the full GP search to succeed — if the fitness
//! function can't tell the true invariant from a decoy under noise, no
//! amount of population/mutation tuning will find it either. It is not
//! sufficient on its own (a full run also depends on the search actually
//! generating the right candidate), so a strong result here is evidence the
//! full pipeline would likely hold up, not a proof that it does. Extending
//! this to a true end-to-end noisy-GP benchmark would need the
//! trajectory-injection refactor described above — a separate, larger piece
//! of work, deliberately not bundled into this one.

use symthaea_core::hdc::conjecture_engine::{Expr, lie_derivative_variance};

fn xorshift_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn uniform01(state: &mut u64) -> f64 {
    (xorshift_next(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Standard-normal sample via Box-Muller. `u1` is floored away from 0 to
/// avoid `ln(0) = -inf` propagating to NaN — the textbook Box-Muller
/// footgun.
fn standard_normal(state: &mut u64) -> f64 {
    let u1 = uniform01(state).max(1e-12);
    let u2 = uniform01(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Add i.i.d. Gaussian measurement noise to each component of each state in
/// `trajectory`, scaled to `relative_sigma` fraction of that component's own
/// magnitude (with a small absolute floor so near-zero crossings don't get a
/// vanishing noise scale). This models sensor/observation noise on an
/// otherwise-exact trajectory — NOT process noise perturbing the underlying
/// dynamics, which is a different (and different-to-interpret) thing.
pub fn add_measurement_noise(
    trajectory: &[Vec<f64>],
    relative_sigma: f64,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut rng = seed | 1;
    trajectory
        .iter()
        .map(|state| {
            state
                .iter()
                .map(|&v| {
                    let scale = v.abs().max(0.1) * relative_sigma;
                    v + standard_normal(&mut rng) * scale
                })
                .collect()
        })
        .collect()
}

/// Centered moving-average smoother, window `2*radius+1` samples wide.
/// Edge points use whatever window fits (shrinking, not wrapping/padding).
pub fn smooth_trajectory(trajectory: &[Vec<f64>], radius: usize) -> Vec<Vec<f64>> {
    if trajectory.is_empty() {
        return Vec::new();
    }
    let dim = trajectory[0].len();
    let n = trajectory.len();
    (0..n)
        .map(|i| {
            let lo = i.saturating_sub(radius);
            let hi = (i + radius + 1).min(n);
            let count = (hi - lo) as f64;
            (0..dim)
                .map(|d| trajectory[lo..hi].iter().map(|s| s[d]).sum::<f64>() / count)
                .collect()
        })
        .collect()
}

/// One scored candidate: its label and its `lie_derivative_variance` against
/// a given (possibly noisy/smoothed) trajectory.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub label: String,
    pub variance: f64,
}

/// Score `truth` and each of `decoys` against `trajectory` via the real GP
/// fitness function, sorted best (lowest variance) first. Returns the 1-based
/// rank of the truth candidate alongside the full sorted table.
pub fn rank_truth_among_decoys(
    truth: (&str, &Expr),
    decoys: &[(&str, Expr)],
    rhs: fn(&[f64], f64) -> Vec<f64>,
    trajectory: &[Vec<f64>],
    var_names: &[&str],
) -> (usize, Vec<ScoredCandidate>) {
    let mut scored: Vec<ScoredCandidate> = Vec::with_capacity(1 + decoys.len());
    scored.push(ScoredCandidate {
        label: truth.0.to_string(),
        variance: lie_derivative_variance(truth.1, rhs, trajectory, var_names),
    });
    for (label, expr) in decoys {
        scored.push(ScoredCandidate {
            label: label.to_string(),
            variance: lie_derivative_variance(expr, rhs, trajectory, var_names),
        });
    }
    scored.sort_by(|a, b| a.variance.total_cmp(&b.variance));
    let rank = scored
        .iter()
        .position(|c| c.label == truth.0)
        .map(|i| i + 1)
        .unwrap_or(usize::MAX);
    (rank, scored)
}

/// One row of a noise-level sweep: at this noise level (and with/without
/// smoothing), where did the truth candidate rank, and how far apart were
/// its variance and the best decoy's?
#[derive(Debug, Clone)]
pub struct SweepRow {
    pub noise_level: f64,
    pub smoothed: bool,
    pub rank: usize,
    pub truth_variance: f64,
    pub best_decoy_variance: f64,
}

/// Run [`rank_truth_among_decoys`] across a range of noise levels, optionally
/// smoothing each noisy trajectory first. `seed` is fixed per call so results
/// are exactly reproducible.
#[allow(clippy::too_many_arguments)]
pub fn noise_sweep(
    truth: (&str, &Expr),
    decoys: &[(&str, Expr)],
    rhs: fn(&[f64], f64) -> Vec<f64>,
    clean_trajectory: &[Vec<f64>],
    var_names: &[&str],
    noise_levels: &[f64],
    smooth_radius: Option<usize>,
    seed: u64,
) -> Vec<SweepRow> {
    noise_levels
        .iter()
        .map(|&level| {
            let observed = if level > 0.0 {
                add_measurement_noise(clean_trajectory, level, seed)
            } else {
                clean_trajectory.to_vec()
            };
            let data = match smooth_radius {
                Some(r) => smooth_trajectory(&observed, r),
                None => observed,
            };
            let (rank, scored) = rank_truth_among_decoys(truth, decoys, rhs, &data, var_names);
            let truth_variance = scored
                .iter()
                .find(|c| c.label == truth.0)
                .map(|c| c.variance)
                .unwrap_or(f64::MAX);
            let best_decoy_variance = scored
                .iter()
                .filter(|c| c.label != truth.0)
                .map(|c| c.variance)
                .fold(f64::MAX, f64::min);
            SweepRow {
                noise_level: level,
                smoothed: smooth_radius.is_some(),
                rank,
                truth_variance,
                best_decoy_variance,
            }
        })
        .collect()
}

/// Small self-contained test systems, shared by this module's tests and the
/// `noise_robustness_report` example.
pub mod systems {
    use super::Expr;
    use symthaea_core::hdc::conjecture_engine::BinOp;

    pub fn harmonic_rhs(s: &[f64], _t: f64) -> Vec<f64> {
        vec![s[1], -s[0]]
    }

    /// RK4-integrate the harmonic oscillator from (x=1, v=0) for `steps` steps of `dt`.
    pub fn harmonic_trajectory(steps: usize, dt: f64) -> Vec<Vec<f64>> {
        rk4_integrate([1.0, 0.0], harmonic_rhs, steps, dt)
    }

    /// Kepler two-body problem (natural units, GM=1): [x, y, vx, vy].
    pub fn kepler_rhs(s: &[f64], _t: f64) -> Vec<f64> {
        let (x, y, vx, vy) = (s[0], s[1], s[2], s[3]);
        let r2 = x * x + y * y;
        let r3 = r2 * r2.sqrt();
        if r3 < 1e-12 {
            return vec![vx, vy, 0.0, 0.0];
        }
        vec![vx, vy, -x / r3, -y / r3]
    }

    /// RK4-integrate an elliptical Kepler orbit for `steps` steps of `dt`.
    pub fn kepler_trajectory(steps: usize, dt: f64) -> Vec<Vec<f64>> {
        rk4_integrate([1.0, 0.0, 0.0, 0.8], kepler_rhs, steps, dt)
    }

    fn rk4_integrate<const N: usize>(
        initial: [f64; N],
        rhs: fn(&[f64], f64) -> Vec<f64>,
        steps: usize,
        dt: f64,
    ) -> Vec<Vec<f64>> {
        let mut state = initial;
        let mut out = Vec::with_capacity(steps);
        let f = |s: &[f64; N]| -> [f64; N] {
            let v = rhs(s, 0.0);
            std::array::from_fn(|i| v[i])
        };
        let add_scaled = |a: &[f64; N], b: &[f64; N], s: f64| -> [f64; N] {
            std::array::from_fn(|i| a[i] + s * b[i])
        };
        for _ in 0..steps {
            out.push(state.to_vec());
            let k1 = f(&state);
            let k2 = f(&add_scaled(&state, &k1, 0.5 * dt));
            let k3 = f(&add_scaled(&state, &k2, 0.5 * dt));
            let k4 = f(&add_scaled(&state, &k3, dt));
            for i in 0..N {
                state[i] += dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
        }
        out
    }

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }
    fn pow2(name: &str) -> Expr {
        Expr::BinOp(BinOp::Pow, Box::new(var(name)), Box::new(Expr::Const(2.0)))
    }
    fn add(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Add, Box::new(l), Box::new(r))
    }
    fn sub(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Sub, Box::new(l), Box::new(r))
    }
    fn scale(c: f64, e: Expr) -> Expr {
        Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(e))
    }
    fn div(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Div, Box::new(l), Box::new(r))
    }
    fn pow(e: Expr, k: f64) -> Expr {
        Expr::BinOp(BinOp::Pow, Box::new(e), Box::new(Expr::Const(k)))
    }

    /// True invariant: x^2 + v^2.
    pub fn harmonic_truth() -> Expr {
        add(pow2("x"), pow2("v"))
    }

    /// Plausible-but-wrong decoys: wrong coefficient, wrong sign, linear.
    pub fn harmonic_decoys() -> Vec<(&'static str, Expr)> {
        vec![
            (
                "x^2 + 2v^2 (wrong coefficient)",
                add(pow2("x"), scale(2.0, pow2("v"))),
            ),
            ("x^2 - v^2 (wrong sign)", sub(pow2("x"), pow2("v"))),
            ("x + v (linear, not conserved)", add(var("x"), var("v"))),
        ]
    }

    /// True invariant: orbital energy E = 1/2(vx^2+vy^2) - 1/r.
    pub fn kepler_truth() -> Expr {
        let r = pow(add(pow2("x"), pow2("y")), 0.5);
        sub(
            scale(0.5, add(pow2("vx"), pow2("vy"))),
            div(Expr::Const(1.0), r),
        )
    }

    /// Plausible-but-wrong decoys: missing potential, wrong potential power,
    /// position magnitude alone.
    pub fn kepler_decoys() -> Vec<(&'static str, Expr)> {
        let r2 = add(pow2("x"), pow2("y"));
        vec![
            (
                "1/2(vx^2+vy^2) (missing potential)",
                scale(0.5, add(pow2("vx"), pow2("vy"))),
            ),
            (
                "1/2(vx^2+vy^2) - 1/r^2 (wrong potential power)",
                sub(
                    scale(0.5, add(pow2("vx"), pow2("vy"))),
                    div(Expr::Const(1.0), r2.clone()),
                ),
            ),
            ("x^2 + y^2 (position magnitude, not conserved)", r2),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::systems::*;
    use super::*;

    #[test]
    fn noise_injection_preserves_trajectory_length_and_dimension() {
        let traj = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let noisy = add_measurement_noise(&traj, 0.1, 42);
        assert_eq!(noisy.len(), traj.len());
        assert!(noisy.iter().all(|s| s.len() == 2));
    }

    #[test]
    fn zero_noise_reproduces_original_within_tolerance() {
        let traj = vec![vec![1.0, -2.0], vec![3.0, 0.5]];
        let noisy = add_measurement_noise(&traj, 0.0, 42);
        for (a, b) in traj.iter().zip(noisy.iter()) {
            for (x, y) in a.iter().zip(b.iter()) {
                assert!((x - y).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn smoothing_reduces_pointwise_variance_on_noisy_constant_signal() {
        let mut rng = 7u64;
        let clean = vec![vec![5.0]; 200];
        let noisy: Vec<Vec<f64>> = clean
            .iter()
            .map(|s| vec![s[0] + standard_normal(&mut rng)])
            .collect();
        let smoothed = smooth_trajectory(&noisy, 5);

        let var_of = |traj: &[Vec<f64>]| -> f64 {
            let mean = traj.iter().map(|s| s[0]).sum::<f64>() / traj.len() as f64;
            traj.iter().map(|s| (s[0] - mean).powi(2)).sum::<f64>() / traj.len() as f64
        };
        assert!(
            var_of(&smoothed) < var_of(&noisy),
            "smoothing should reduce variance on a noisy constant signal"
        );
    }

    #[test]
    fn harmonic_truth_ranks_first_at_zero_noise() {
        let traj = harmonic_trajectory(1000, 0.01);
        let truth = harmonic_truth();
        let (rank, scored) = rank_truth_among_decoys(
            ("x^2+v^2", &truth),
            &harmonic_decoys(),
            harmonic_rhs,
            &traj,
            &["x", "v"],
        );
        assert_eq!(rank, 1, "scored table: {:?}", scored);
    }

    #[test]
    fn harmonic_truth_still_ranks_first_under_moderate_measurement_noise() {
        let traj = harmonic_trajectory(1000, 0.01);
        let noisy = add_measurement_noise(&traj, 0.02, 123);
        let truth = harmonic_truth();
        let (rank, scored) = rank_truth_among_decoys(
            ("x^2+v^2", &truth),
            &harmonic_decoys(),
            harmonic_rhs,
            &noisy,
            &["x", "v"],
        );
        assert_eq!(
            rank, 1,
            "truth should still rank first at 2% measurement noise: {:?}",
            scored
        );
    }

    #[test]
    fn kepler_truth_ranks_first_at_zero_noise() {
        let traj = kepler_trajectory(2000, 0.001);
        let truth = kepler_truth();
        let (rank, scored) = rank_truth_among_decoys(
            ("E = 1/2 v^2 - 1/r", &truth),
            &kepler_decoys(),
            kepler_rhs,
            &traj,
            &["x", "y", "vx", "vy"],
        );
        assert_eq!(rank, 1, "scored table: {:?}", scored);
    }

    #[test]
    fn kepler_truth_still_ranks_first_under_moderate_measurement_noise() {
        let traj = kepler_trajectory(2000, 0.001);
        let noisy = add_measurement_noise(&traj, 0.02, 123);
        let truth = kepler_truth();
        let (rank, scored) = rank_truth_among_decoys(
            ("E = 1/2 v^2 - 1/r", &truth),
            &kepler_decoys(),
            kepler_rhs,
            &noisy,
            &["x", "y", "vx", "vy"],
        );
        assert_eq!(
            rank, 1,
            "truth should still rank first at 2% measurement noise: {:?}",
            scored
        );
    }

    #[test]
    fn noise_sweep_at_zero_noise_always_ranks_truth_first() {
        // The one point on the sweep we can assert unconditionally: zero
        // noise reduces to the exact-data case, which must always succeed
        // regardless of system or smoothing setting.
        let traj = harmonic_trajectory(1000, 0.01);
        let truth = harmonic_truth();
        let rows = noise_sweep(
            ("x^2+v^2", &truth),
            &harmonic_decoys(),
            harmonic_rhs,
            &traj,
            &["x", "v"],
            &[0.0],
            None,
            99,
        );
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].rank, 1, "row: {:?}", rows[0]);
    }

    #[test]
    fn noise_sweep_rows_are_finite_across_levels() {
        // Not a robustness assertion (that's what the example's full sweep
        // table is for) — just a sanity check that the sweep machinery
        // itself doesn't produce NaN/inf at higher noise levels.
        let traj = kepler_trajectory(2000, 0.001);
        let truth = kepler_truth();
        let rows = noise_sweep(
            ("E", &truth),
            &kepler_decoys(),
            kepler_rhs,
            &traj,
            &["x", "y", "vx", "vy"],
            &[0.01, 0.05, 0.1, 0.2],
            Some(5),
            7,
        );
        for row in &rows {
            assert!(row.truth_variance.is_finite(), "row: {:?}", row);
            assert!(row.best_decoy_variance.is_finite(), "row: {:?}", row);
        }
    }
}
