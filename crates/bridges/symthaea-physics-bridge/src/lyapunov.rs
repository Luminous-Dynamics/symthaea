// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Local Lyapunov function discovery for dissipative systems.
//!
//! Every invariant the Ramanujan Protocol showcase discovers is *exact*
//! (dV/dt = 0) — conserved quantities of conservative systems. Real
//! dissipative systems (friction, radiation damping, anything with an
//! attractor) don't have those; they have Lyapunov functions instead,
//! quantities that decrease monotonically toward an equilibrium
//! (dV/dt ≤ 0). The existing `find_lyapunov_candidate` in
//! `conjecture_engine::autonomous` isn't a real search for one — it tests a
//! single hardcoded candidate (Σxᵢ²) for late-trajectory variance stability
//! and, by its own comment, explicitly returns "(bounded attractor, not
//! Lyapunov)" rather than actually verifying monotonic decrease.
//!
//! ## Why this isn't a genetic-programming search
//!
//! For a system with a locally asymptotically stable equilibrium, a local
//! quadratic Lyapunov function is *guaranteed to exist* and is directly
//! computable — no search needed. Linearize the dynamics at the equilibrium
//! (Jacobian `A`); if `A` is Hurwitz (all eigenvalues have negative real
//! part), the continuous Lyapunov equation `AᵀP + PA = -Q` has a unique
//! solution `P` for any positive-definite `Q`, and `V(x) = xᵀPx` is a valid
//! local Lyapunov function for the *linearized* system by a standard
//! control-theory theorem (see e.g. Khalil, *Nonlinear Systems*, §4.3). This
//! is the textbook-correct approach, not a heuristic — a GP search could only
//! ever rediscover what this computes directly and exactly.
//!
//! What a GP search over expression trees *would* still be needed for:
//! non-quadratic or non-local Lyapunov functions (e.g. for a limit cycle, or
//! a global rather than local certificate). Out of scope here — this module
//! is the well-posed local/linearization case, done correctly, not a partial
//! attempt at the harder general case.
//!
//! ## What "valid" means here
//!
//! `V` is only *guaranteed* to satisfy dV/dt ≤ 0 for the **linearized**
//! system, in some neighborhood of the equilibrium — not globally, and not
//! exactly for the true nonlinear dynamics beyond that neighborhood. This
//! module verifies dV/dt ≤ 0 **numerically**, by sampling points at a given
//! radius around the equilibrium and evaluating the true (possibly
//! nonlinear) `rhs`, and reports the sampled violation rate honestly rather
//! than asserting global validity it hasn't checked.

use nalgebra::{Complex, DMatrix};
use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, fd_gradient};

/// Finite-difference Jacobian of `rhs` at `point` (n x n, `J[i][j] = d(rhs_i)/d(x_j)`).
pub fn numerical_jacobian(rhs: fn(&[f64], f64) -> Vec<f64>, point: &[f64]) -> DMatrix<f64> {
    const EPS: f64 = 1e-6;
    let n = point.len();
    let mut j = DMatrix::zeros(n, n);
    for col in 0..n {
        let mut plus = point.to_vec();
        let mut minus = point.to_vec();
        plus[col] += EPS;
        minus[col] -= EPS;
        let f_plus = rhs(&plus, 0.0);
        let f_minus = rhs(&minus, 0.0);
        for row in 0..n {
            j[(row, col)] = (f_plus[row] - f_minus[row]) / (2.0 * EPS);
        }
    }
    j
}

/// Is `a` Hurwitz (every eigenvalue has strictly negative real part)? This is
/// exactly the condition under which the equilibrium is locally
/// asymptotically stable for the linearized system, and the condition under
/// which the continuous Lyapunov equation has a unique (positive-definite,
/// for positive-definite Q) solution.
pub fn is_hurwitz(a: &DMatrix<f64>) -> bool {
    let eigenvalues: Vec<Complex<f64>> = a.complex_eigenvalues().iter().copied().collect();
    !eigenvalues.is_empty() && eigenvalues.iter().all(|e| e.re < -1e-9)
}

/// Solve the continuous Lyapunov equation `AᵀP + PA = -Q` for symmetric `P`,
/// via vectorization: `vec(AᵀP + PA) = (I⊗Aᵀ + Aᵀ⊗I) vec(P) = -vec(Q)`. Returns
/// `None` if `a` is not Hurwitz (no uniqueness/existence guarantee then) or
/// if the resulting `P` is not positive definite (numerical failure, or `a`
/// was marginally accepted by the Hurwitz tolerance).
pub fn solve_lyapunov_equation(a: &DMatrix<f64>, q: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    if !is_hurwitz(a) {
        return None;
    }
    let n = a.nrows();
    let at = a.transpose();
    let identity = DMatrix::<f64>::identity(n, n);
    // M = I ⊗ Aᵀ + Aᵀ ⊗ I  (n² x n²)
    let m = identity.kronecker(&at) + at.kronecker(&identity);
    let rhs_vec = DMatrix::from_iterator(n * n, 1, q.iter().map(|v| -v));
    let lu = m.lu();
    let p_vec = lu.solve(&rhs_vec)?;
    let mut p = DMatrix::zeros(n, n);
    for i in 0..n * n {
        p[(i % n, i / n)] = p_vec[i];
    }
    // Numerical solve may not be exactly symmetric; symmetrize.
    let p_sym = (&p + p.transpose()) * 0.5;
    if is_positive_definite(&p_sym) {
        Some(p_sym)
    } else {
        None
    }
}

/// Is symmetric `p` positive definite? (Cholesky succeeds iff it is.)
pub fn is_positive_definite(p: &DMatrix<f64>) -> bool {
    p.clone().cholesky().is_some()
}

/// Build the symbolic quadratic form `V(x) = xᵀPx = Σᵢⱼ P[i][j] xᵢ xⱼ` as an
/// [`Expr`], so it composes with the rest of the discovery/recognition
/// pipeline (printing, catalog matching, future Z3 work).
pub fn quadratic_form_expr(p: &DMatrix<f64>, var_names: &[&str]) -> Expr {
    let n = var_names.len();
    assert_eq!(p.nrows(), n);
    assert_eq!(p.ncols(), n);
    let mut terms: Vec<Expr> = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let coeff = p[(i, j)];
            if coeff.abs() < 1e-12 {
                continue;
            }
            let term = Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Const(coeff)),
                Box::new(Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Var(var_names[i].to_string())),
                    Box::new(Expr::Var(var_names[j].to_string())),
                )),
            );
            terms.push(term);
        }
    }
    terms
        .into_iter()
        .reduce(|a, b| Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b)))
        .unwrap_or(Expr::Const(0.0))
}

/// A discovered local Lyapunov certificate, with an honest numeric
/// verification report against the true (possibly nonlinear) dynamics.
#[derive(Debug, Clone)]
pub struct LyapunovCertificate {
    pub p: DMatrix<f64>,
    pub v_expr: Expr,
    /// Fraction of sampled points (within `sample_radius` of the
    /// equilibrium) where dV/dt > 0 was observed against the true dynamics.
    pub violation_fraction: f64,
    /// The largest positive dV/dt observed among sampled points (0.0 if none).
    pub max_violation: f64,
    pub samples_checked: usize,
}

/// Deterministic xorshift64, matching the sampler used elsewhere in this crate.
fn xorshift_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Discover a local Lyapunov function for `rhs` at `equilibrium`, and
/// numerically verify it against the true dynamics within `sample_radius`.
/// Uses `Q = I` (the standard default choice absent other information).
/// Returns `None` if the equilibrium isn't locally asymptotically stable
/// (linearization not Hurwitz) — an honest negative result, not a failure to
/// search hard enough.
pub fn discover_local_lyapunov(
    rhs: fn(&[f64], f64) -> Vec<f64>,
    equilibrium: &[f64],
    var_names: &[&str],
    sample_radius: f64,
    num_samples: usize,
    seed: u64,
) -> Option<LyapunovCertificate> {
    let n = equilibrium.len();
    let a = numerical_jacobian(rhs, equilibrium);
    let q = DMatrix::<f64>::identity(n, n);
    let p = solve_lyapunov_equation(&a, &q)?;
    let v_expr = quadratic_form_expr(&p, var_names);

    let mut rng = seed | 1;
    let mut violations = 0usize;
    let mut max_violation = 0.0f64;
    let mut checked = 0usize;
    for _ in 0..num_samples {
        let mut point = equilibrium.to_vec();
        for coord in point.iter_mut() {
            let r = xorshift_next(&mut rng);
            let unit = (r >> 11) as f64 / (1u64 << 53) as f64; // [0,1)
            *coord += (unit * 2.0 - 1.0) * sample_radius;
        }
        let grad = fd_gradient(&v_expr, &point, var_names);
        let flow = rhs(&point, 0.0);
        if grad.len() != flow.len() {
            continue;
        }
        let dv_dt: f64 = grad.iter().zip(flow.iter()).map(|(g, f)| g * f).sum();
        if !dv_dt.is_finite() {
            continue;
        }
        checked += 1;
        if dv_dt > 1e-9 {
            violations += 1;
            max_violation = max_violation.max(dv_dt);
        }
    }

    Some(LyapunovCertificate {
        p,
        v_expr,
        violation_fraction: if checked > 0 {
            violations as f64 / checked as f64
        } else {
            f64::NAN
        },
        max_violation,
        samples_checked: checked,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Damped harmonic oscillator: dx/dt = v, dv/dt = -x - 0.3*v.
    fn damped_rhs_03(s: &[f64], _t: f64) -> Vec<f64> {
        vec![s[1], -s[0] - 0.3 * s[1]]
    }
    fn undamped_rhs(s: &[f64], _t: f64) -> Vec<f64> {
        vec![s[1], -s[0]]
    }
    /// Van der Pol with negative mu: locally stable at the origin (unlike
    /// the standard mu>0 case, which is locally unstable / limit-cycle
    /// attracting from outside).
    fn van_der_pol_stable_rhs(s: &[f64], _t: f64) -> Vec<f64> {
        let mu = -0.5;
        vec![s[1], mu * (1.0 - s[0] * s[0]) * s[1] - s[0]]
    }

    #[test]
    fn jacobian_matches_known_linear_system() {
        let j = numerical_jacobian(damped_rhs_03, &[0.0, 0.0]);
        // A = [[0, 1], [-1, -0.3]]
        assert!((j[(0, 0)] - 0.0).abs() < 1e-4);
        assert!((j[(0, 1)] - 1.0).abs() < 1e-4);
        assert!((j[(1, 0)] - (-1.0)).abs() < 1e-4);
        assert!((j[(1, 1)] - (-0.3)).abs() < 1e-4);
    }

    #[test]
    fn damped_oscillator_is_hurwitz_undamped_is_not() {
        let damped_a = numerical_jacobian(damped_rhs_03, &[0.0, 0.0]);
        assert!(is_hurwitz(&damped_a), "damped oscillator should be Hurwitz");

        let undamped_a = numerical_jacobian(undamped_rhs, &[0.0, 0.0]);
        assert!(
            !is_hurwitz(&undamped_a),
            "undamped oscillator (marginally stable, eigenvalues +-i) must NOT be reported Hurwitz"
        );
    }

    #[test]
    fn undamped_oscillator_yields_no_lyapunov_certificate() {
        // Honest negative result: a center (purely imaginary eigenvalues)
        // has no solution to the continuous Lyapunov equation with the
        // uniqueness guarantee this method relies on.
        let cert = discover_local_lyapunov(undamped_rhs, &[0.0, 0.0], &["x", "v"], 1.0, 50, 7);
        assert!(cert.is_none());
    }

    #[test]
    fn damped_oscillator_lyapunov_matches_hand_solved_p() {
        // Hand-solved via A^T P + P A = -I for A=[[0,1],[-1,-gamma]], gamma=0.3:
        // p12 = 0.5, p22 = 1/gamma, p11 = p22 + gamma/2
        let gamma = 0.3;
        let expected_p12 = 0.5;
        let expected_p22 = 1.0 / gamma;
        let expected_p11 = expected_p22 + gamma / 2.0;

        let cert = discover_local_lyapunov(damped_rhs_03, &[0.0, 0.0], &["x", "v"], 3.0, 200, 42)
            .expect("damped oscillator must yield a certificate");

        assert!(
            (cert.p[(0, 0)] - expected_p11).abs() < 1e-3,
            "p11: {}",
            cert.p[(0, 0)]
        );
        assert!(
            (cert.p[(0, 1)] - expected_p12).abs() < 1e-3,
            "p12: {}",
            cert.p[(0, 1)]
        );
        assert!(
            (cert.p[(1, 1)] - expected_p22).abs() < 1e-3,
            "p22: {}",
            cert.p[(1, 1)]
        );

        // The damped oscillator is LINEAR, so dV/dt should be <=0 essentially
        // everywhere sampled, not just locally near the equilibrium.
        assert_eq!(
            cert.violation_fraction, 0.0,
            "linear system: no violations expected anywhere, got {} / {} samples",
            cert.violation_fraction, cert.samples_checked
        );
    }

    #[test]
    fn stable_van_der_pol_yields_locally_valid_certificate() {
        // Genuinely nonlinear system. Only local validity is guaranteed --
        // check close to the equilibrium, not far from it.
        let cert = discover_local_lyapunov(
            van_der_pol_stable_rhs,
            &[0.0, 0.0],
            &["x", "v"],
            0.05,
            200,
            99,
        )
        .expect("stable Van der Pol (mu<0) should yield a certificate at the origin");
        assert!(is_positive_definite(&cert.p));
        assert_eq!(
            cert.violation_fraction, 0.0,
            "close to equilibrium, the local certificate should hold: {}/{} violated (max {})",
            cert.violation_fraction, cert.samples_checked, cert.max_violation
        );
    }

    #[test]
    fn quadratic_form_expr_matches_matrix_entries() {
        let p = DMatrix::from_row_slice(2, 2, &[2.0, 1.0, 1.0, 3.0]);
        let expr = quadratic_form_expr(&p, &["x", "y"]);
        // V = 2x^2 + 1xy + 1yx + 3y^2 = 2x^2 + 2xy + 3y^2 at (x=1,y=1) -> 2+2+3=7
        let v = expr.eval(&[("x", 1.0), ("y", 1.0)]);
        assert!((v - 7.0).abs() < 1e-9, "V(1,1) = {}", v);
    }
}
