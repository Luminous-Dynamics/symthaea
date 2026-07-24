// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Stage B: local (Noether-current-style) conservation checking.
//!
//! Stage A (`symthaea-physics-bridge/src/pde_wave_stage_a.rs`) finds a single
//! *global* conserved scalar (`Σᵢ ρᵢ`) for a discretized field. A true Noether
//! current is a stronger, more local claim: a density `ρᵢ` and flux `J_{i+1/2}`
//! satisfying the discrete continuity equation `dρᵢ/dt = J_{i+1/2} − J_{i−1/2}`
//! at *every* grid point, not just in aggregate. [`discrete_continuity_residual`]
//! checks that claim directly.
//!
//! ## Stencil convention
//!
//! A density candidate is an [`Expr`] over local point-stencil variable names
//! `"u_l"`, `"u_c"`, `"u_r"`, `"v_c"` (left/center/right position, center
//! velocity — the same functional form is evaluated at every grid point,
//! which is what makes this a field-theoretic statement rather than `n`
//! separate point-particle facts). A flux candidate is an [`Expr`] over
//! bond-stencil variable names `"u_c"`, `"u_r"`, `"v_c"`, `"v_r"` (left and
//! right endpoints of one bond). No new [`Expr`] variant is needed —
//! [`Expr::eval`] already takes an arbitrary named-binding slice.
//!
//! Full-system state stays Stage A's convention: `[u1..un, v1..vn]`.
//! Boundary values (`u0`, `u_{n+1}`, `v0`, `v_{n+1}`) are `0.0`, substituted
//! automatically.

use super::{BinOp, Expr, fd_gradient, simplify};

const STENCIL_NAMES: [&str; 6] = ["u_l", "u_c", "u_r", "v_l", "v_c", "v_r"];
const BOND_NAMES: [&str; 4] = ["u_c", "u_r", "v_c", "v_r"];

/// Gauge-fix a flux candidate: `J' = J - J(0,0,0,0)`.
///
/// `J` only ever enters [`discrete_continuity_residual`] as a *difference*,
/// `J_{i+1/2} - J_{i-1/2}` -- so any constant shift `J + c` scores identically
/// to `J`. With the density `ρ` held fixed (the M2 setting: only `J` is being
/// searched over), a constant shift is the *only* surviving gauge freedom
/// (the general continuity-equation gauge group is `ρ' = ρ + Ġ`, `J' = J - Ġ`
/// for arbitrary `G`; fixing `ρ` collapses that to just `G = c/2`). Without
/// removing it, the search is underdetermined -- `J_truth + 7.3` would be
/// indistinguishable from `J_truth`. Subtracting the value at the all-zero
/// stencil state is a physically motivated normalization (flux should vanish
/// with no field present) that removes exactly this degeneracy. Used both
/// when scoring search candidates and when comparing a result against
/// [`Expr`] truth for cross-validation -- an un-gauge-fixed comparison would
/// produce false negatives.
pub fn gauge_fix_flux(j: &Expr) -> Expr {
    let zero_bindings: Vec<(&str, f64)> = BOND_NAMES.iter().copied().zip([0.0; 4]).collect();
    let c = j.eval(&zero_bindings);
    if !c.is_finite() || c == 0.0 {
        return j.clone();
    }
    simplify(&Expr::BinOp(
        BinOp::Sub,
        Box::new(j.clone()),
        Box::new(Expr::Const(c)),
    ))
}

/// `du_l/dt, du_c/dt, du_r/dt, [unused], dv_c/dt, [unused]` for the discrete
/// wave-chain dynamics (`dv_i/dt = u_{i+1} - 2*u_i + u_{i-1}`). The two
/// "unused" slots (would-be `dv_l/dt`, `dv_r/dt`) are placeholders: any
/// density built only from `{u_l, u_c, u_r, v_c}` (the documented stencil
/// convention) has zero gradient there, so their value never affects the
/// resulting Lie-derivative dot product. If a future density candidate
/// legitimately depended on `v_l`/`v_r` this function would need extending
/// — not a concern for the wave-chain case this module was built for.
fn wave_chain_local_flow(u_l: f64, u_c: f64, u_r: f64, v_l: f64, v_c: f64, v_r: f64) -> [f64; 6] {
    [v_l, v_c, v_r, 0.0, u_r - 2.0 * u_c + u_l, 0.0]
}

fn bond_eval(j: &Expr, u_c: f64, u_r: f64, v_c: f64, v_r: f64) -> f64 {
    let bindings: Vec<(&str, f64)> = BOND_NAMES
        .iter()
        .copied()
        .zip([u_c, u_r, v_c, v_r])
        .collect();
    j.eval(&bindings)
}

/// Max discrete-continuity residual `|dρᵢ/dt − (J_{i+1/2} − J_{i−1/2})|` over
/// every interior grid point `i` (`1..=n`) and every trajectory sample.
/// "Small is good" -- same convention as
/// [`lie_derivative_variance`](super::lie_derivative_variance). Returns
/// `f64::MAX` on any non-finite intermediate value or an empty/malformed
/// trajectory, so a broken candidate can't silently score as "conserved."
///
/// `rho` is evaluated with the point-stencil names, `j` with the
/// bond-stencil names (see module docs). Generic over `local_flow`, the
/// physical system's own `(u_l,u_c,u_r,v_l,v_c,v_r) -> [du_l/dt, du_c/dt,
/// du_r/dt, dv_l/dt, dv_c/dt, dv_r/dt]` dynamics -- this crate stays
/// physics-agnostic; a caller (e.g. `symthaea-physics-bridge`) supplies its
/// own local-flow closure, optionally capturing its own parameters (e.g. a
/// coupling constant), without this crate needing to know about them.
///
/// **Generality caveat, stated explicitly rather than implied**: a 3-point
/// position stencil genuinely cannot compute `dv_l/dt`/`dv_r/dt` for most
/// local dynamics -- those generally need `u_{i-2}`/`u_{i+2}}`, outside this
/// stencil. The two "unused" slots in `local_flow`'s returned `[f64; 6]` are
/// correct placeholders **only** for a density that (like every `rho`
/// candidate this codebase has used so far) depends solely on
/// `{u_l,u_c,u_r,v_c}` and therefore has zero gradient there regardless of
/// their value -- not a fully general local-dynamics interface for an
/// arbitrary future density shape.
pub fn discrete_continuity_residual_with_flow(
    rho: &Expr,
    j: &Expr,
    trajectory: &[Vec<f64>],
    n: usize,
    local_flow: impl Fn(f64, f64, f64, f64, f64, f64) -> [f64; 6],
) -> f64 {
    if n == 0 || trajectory.is_empty() {
        return f64::MAX;
    }

    let mut max_residual: f64 = 0.0;
    let mut any_finite = false;

    for state in trajectory {
        if state.len() != 2 * n {
            return f64::MAX;
        }
        let u = |i: i64| -> f64 {
            if i < 1 || i > n as i64 {
                0.0
            } else {
                state[(i - 1) as usize]
            }
        };
        let v = |i: i64| -> f64 {
            if i < 1 || i > n as i64 {
                0.0
            } else {
                state[n + (i - 1) as usize]
            }
        };

        for i in 1..=n as i64 {
            let (u_l, u_c, u_r) = (u(i - 1), u(i), u(i + 1));
            let (v_l, v_c, v_r) = (v(i - 1), v(i), v(i + 1));

            let s6 = [u_l, u_c, u_r, v_l, v_c, v_r];
            let grad = fd_gradient(rho, &s6, &STENCIL_NAMES);
            if grad.iter().any(|g| !g.is_finite()) {
                return f64::MAX;
            }
            let flow = local_flow(u_l, u_c, u_r, v_l, v_c, v_r);
            let d_rho_dt: f64 = grad.iter().zip(flow.iter()).map(|(g, f)| g * f).sum();

            let j_right = bond_eval(j, u_c, u_r, v_c, v_r);
            let j_left = bond_eval(j, u_l, u_c, v_l, v_c);

            if !d_rho_dt.is_finite() || !j_right.is_finite() || !j_left.is_finite() {
                return f64::MAX;
            }

            let residual = (d_rho_dt - (j_right - j_left)).abs();
            max_residual = max_residual.max(residual);
            any_finite = true;
        }
    }

    if !any_finite { f64::MAX } else { max_residual }
}

/// Thin wrapper preserving the original wave-chain-only behavior exactly --
/// [`discrete_continuity_residual_with_flow`] with [`wave_chain_local_flow`]
/// baked in. Zero behavior change from before this function was generalized
/// (verified by the full existing `conjecture_engine::` test suite staying
/// green), kept for every existing caller in this arc's M0-M3 wave-chain work.
pub fn discrete_continuity_residual(
    rho: &Expr,
    j: &Expr,
    trajectory: &[Vec<f64>],
    n: usize,
) -> f64 {
    discrete_continuity_residual_with_flow(rho, j, trajectory, n, wave_chain_local_flow)
}

/// Below this, `Σ z(J)²` (or `Σ y²`) is treated as numerically degenerate --
/// too close to constant/zero across the sampled `(i,t)` set for `alpha*` in
/// [`shape_calibrated_residual`] to be a meaningful scale estimate. Guards
/// against a near-zero or gauge-fixed-to-zero candidate exploiting the
/// calibration's normalization (dividing by an near-zero `⟨z,z⟩` or `⟨y,y⟩`
/// is either undefined or numerically unstable, not "perfectly calibrated").
const MIN_SHAPE_VARIANCE: f64 = 1e-9;

/// Target-blind shape/scale decomposition of a flux candidate `j` against
/// the discrete continuity equation, given fixed `rho`. "Target-blind"
/// means this never compares `j` to any *specific* known-correct flux
/// expression -- it only uses the physically observed continuity target
/// `y_i(t) = dρᵢ/dt` (from `rho` and the system's own dynamics) and the
/// candidate's own local imbalance `z_i(t) = J_{i+1/2} - J_{i-1/2}`, both
/// sampled at every interior grid point `i` and trajectory sample `t`, then
/// flattened into vectors over that combined `(i,t)` index.
///
/// This exists because [`discrete_continuity_residual`] only rewards a
/// candidate once its *value* is already close to the right scale --
/// M2.1 diagnosed this as offering no partial-credit gradient toward the
/// answer (a partially-correct-shape, wrong-amplitude candidate scores no
/// better than noise). Here, instead:
///
/// - **Shape alignment**: `s(J) = ⟨z(J), y⟩ / (‖z(J)‖·‖y‖)` -- cosine
///   similarity between the candidate's and the target's local-imbalance
///   *pattern*, independent of scale.
/// - **Calibrated residual**: fit the one analytically-optimal scalar
///   `alpha* = ⟨z(J), y⟩ / ⟨z(J), z(J)⟩` (least-squares best multiple of
///   `z(J)` that approximates `y`) and report
///   `L_cal(J) = ‖y - alpha*·z(J)‖² / ‖y‖²`.
///
/// **These are not two independent fitness terms.** Substituting `alpha*`
/// into `L_cal` gives exactly `L_cal(J) = 1 - s(J)²` -- the calibrated
/// residual *is* a monotone transform of shape alignment, not an additional
/// signal. `alignment` is retained on [`ShapeCalibration`] purely as a
/// diagnostic (sign-carrying, easier to plot over generations); the scalar
/// fitness callers should minimize is `calibrated_residual`.
///
/// Returns `None` if the trajectory/candidate is malformed, any
/// intermediate evaluation is non-finite, or either `⟨z,z⟩` or `⟨y,y⟩` is
/// below [`MIN_SHAPE_VARIANCE`] (see that constant's docs).
pub fn shape_calibrated_residual(
    rho: &Expr,
    j: &Expr,
    trajectory: &[Vec<f64>],
    n: usize,
) -> Option<ShapeCalibration> {
    if n == 0 || trajectory.is_empty() {
        return None;
    }

    let (mut sum_zy, mut sum_zz, mut sum_yy) = (0.0_f64, 0.0_f64, 0.0_f64);
    let mut any = false;

    for state in trajectory {
        if state.len() != 2 * n {
            return None;
        }
        let u = |i: i64| -> f64 {
            if i < 1 || i > n as i64 {
                0.0
            } else {
                state[(i - 1) as usize]
            }
        };
        let v = |i: i64| -> f64 {
            if i < 1 || i > n as i64 {
                0.0
            } else {
                state[n + (i - 1) as usize]
            }
        };

        for i in 1..=n as i64 {
            let (u_l, u_c, u_r) = (u(i - 1), u(i), u(i + 1));
            let (v_l, v_c, v_r) = (v(i - 1), v(i), v(i + 1));

            let s6 = [u_l, u_c, u_r, v_l, v_c, v_r];
            let grad = fd_gradient(rho, &s6, &STENCIL_NAMES);
            if grad.iter().any(|g| !g.is_finite()) {
                return None;
            }
            let flow = wave_chain_local_flow(u_l, u_c, u_r, v_l, v_c, v_r);
            let d_rho_dt: f64 = grad.iter().zip(flow.iter()).map(|(g, f)| g * f).sum();
            let y = -d_rho_dt;

            let j_right = bond_eval(j, u_c, u_r, v_c, v_r);
            let j_left = bond_eval(j, u_l, u_c, v_l, v_c);
            if !y.is_finite() || !j_right.is_finite() || !j_left.is_finite() {
                return None;
            }
            let z = j_right - j_left;

            sum_zy += z * y;
            sum_zz += z * z;
            sum_yy += y * y;
            any = true;
        }
    }

    if !any || sum_zz < MIN_SHAPE_VARIANCE || sum_yy < MIN_SHAPE_VARIANCE {
        return None;
    }

    let alignment = sum_zy / (sum_zz.sqrt() * sum_yy.sqrt());
    let alpha = sum_zy / sum_zz;
    let calibrated_residual = (1.0 - alignment * alignment).clamp(0.0, 1.0);

    Some(ShapeCalibration {
        alignment,
        alpha,
        calibrated_residual,
    })
}

/// See [`shape_calibrated_residual`].
#[derive(Debug, Clone, Copy)]
pub struct ShapeCalibration {
    /// `⟨z(J), y⟩ / (‖z(J)‖·‖y‖)` -- cosine similarity of the candidate's
    /// and target's local-imbalance patterns, in `[-1, 1]`. Diagnostic only
    /// (see [`shape_calibrated_residual`]'s docs on why it's not an
    /// independent fitness term).
    pub alignment: f64,
    /// Analytically-optimal scale `⟨z(J), y⟩ / ⟨z(J), z(J)⟩`. Callers that
    /// accept a candidate should absorb this into the reported expression
    /// (`alpha * J`, gauge-fixed) before cross-validating against held-out
    /// trajectories -- the calibrated fitness never sees held-out data, but
    /// the *final reported candidate* must be properly scaled to be
    /// evaluated fairly by [`discrete_continuity_residual`].
    pub alpha: f64,
    /// `‖y - alpha*·z(J)‖² / ‖y‖²`, exactly `1 - alignment²`. "Small is
    /// good", same convention as [`discrete_continuity_residual`]. This is
    /// the scalar fitness to minimize during search.
    pub calibrated_residual: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `rho = v_c`, `n = 1` (single free grid point, zero boundaries): the
    /// resulting target `y = -dρ/dt = -(u_r - 2u_c + u_l)` reduces, at the
    /// lone free point, to `y = 2*u_c` exactly (since `u_l = u_r = 0` at the
    /// boundary). Chosen for hand-computable expected values, not physical
    /// realism -- these tests check the calibration's *algebra*, not the
    /// wave-chain physics (that's `pde_wave_stage_b.rs`'s job).
    fn toy_rho() -> Expr {
        Expr::Var("v_c".to_string())
    }

    /// Three states `[u1, v1]` (n=1, so length 2), independently varying u
    /// and v so downstream candidates aren't accidentally perfectly aligned
    /// or anti-aligned by construction.
    fn toy_trajectory() -> Vec<Vec<f64>> {
        vec![vec![0.3, 0.5], vec![-0.2, 0.4], vec![0.1, -0.3]]
    }

    #[test]
    fn calibrated_residual_equals_one_minus_alignment_squared() {
        // j = u_r: at the lone free point, j_right always binds "u_r" to
        // the (out-of-range, zero) right neighbor, so j_right == 0 always;
        // j_left binds "u_r" to u_c's own value, so j_left == u_c. Hence
        // z = j_right - j_left = -u_c, a clean deterministic multiple of
        // y = 2*u_c (alignment == -1 exactly, hand-verified below).
        let j = Expr::Var("u_r".to_string());
        let traj = toy_trajectory();
        let cal = shape_calibrated_residual(&toy_rho(), &j, &traj, 1)
            .expect("well-posed candidate must score");

        assert!(
            (cal.alignment - (-1.0)).abs() < 1e-9,
            "expected perfect anti-alignment, got {}",
            cal.alignment
        );
        assert!(
            (cal.alpha - (-2.0)).abs() < 1e-9,
            "expected alpha=-2 (y=2*u_c, z=-u_c), got {}",
            cal.alpha
        );
        let expected = 1.0 - cal.alignment * cal.alignment;
        assert!(
            (cal.calibrated_residual - expected).abs() < 1e-12,
            "calibrated_residual must equal 1-alignment^2 exactly: got {} vs {}",
            cal.calibrated_residual,
            expected
        );
        assert!(
            cal.calibrated_residual < 1e-9,
            "a perfectly-shaped candidate should have ~zero calibrated residual, got {}",
            cal.calibrated_residual
        );
    }

    #[test]
    fn calibrated_residual_is_scale_invariant_but_alpha_is_not() {
        let j = Expr::Var("u_r".to_string());
        let j_scaled = Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(3.0)), Box::new(j.clone()));
        let traj = toy_trajectory();

        let cal = shape_calibrated_residual(&toy_rho(), &j, &traj, 1).unwrap();
        let cal_scaled = shape_calibrated_residual(&toy_rho(), &j_scaled, &traj, 1).unwrap();

        assert!(
            (cal.alignment - cal_scaled.alignment).abs() < 1e-9,
            "alignment must be scale-invariant: {} vs {}",
            cal.alignment,
            cal_scaled.alignment
        );
        assert!(
            (cal.calibrated_residual - cal_scaled.calibrated_residual).abs() < 1e-9,
            "calibrated_residual must be scale-invariant: {} vs {}",
            cal.calibrated_residual,
            cal_scaled.calibrated_residual
        );
        // alpha must compensate inversely for the 3x scale-up, so the
        // *effective* scaled flux (alpha * j) is identical either way.
        assert!(
            (cal.alpha - 3.0 * cal_scaled.alpha).abs() < 1e-9,
            "alpha must compensate inversely for scale: {} vs 3*{}",
            cal.alpha,
            cal_scaled.alpha
        );
    }

    #[test]
    fn calibrated_residual_rejects_constant_flux() {
        // A constant flux has j_right == j_left everywhere by construction
        // (no gauge-fixing needed to see it): z = 0 identically, which must
        // trip the degenerate-variance guard rather than produce a
        // divide-by-zero or spuriously "perfect" alpha.
        let j = Expr::Const(1.0);
        let traj = toy_trajectory();
        assert!(shape_calibrated_residual(&toy_rho(), &j, &traj, 1).is_none());
    }

    #[test]
    fn calibrated_residual_ranks_an_unrelated_shape_worse_than_a_perfect_one() {
        // j = v_c: z = v_c-at-i - v_c-at-(i-1)-slot, a pattern with no
        // structural relationship to y = 2*u_c on this trajectory (u and v
        // vary independently in `toy_trajectory`) -- should land strictly
        // between "perfect" (0) and "None" (degenerate), not tied to either.
        let j_perfect = Expr::Var("u_r".to_string());
        let j_unrelated = Expr::Var("v_c".to_string());
        let traj = toy_trajectory();

        let cal_perfect = shape_calibrated_residual(&toy_rho(), &j_perfect, &traj, 1).unwrap();
        let cal_unrelated = shape_calibrated_residual(&toy_rho(), &j_unrelated, &traj, 1).unwrap();

        assert!(cal_perfect.calibrated_residual < 1e-9);
        assert!(
            cal_unrelated.calibrated_residual > 1e-3,
            "an unrelated shape should score meaningfully worse than a perfect one, got {}",
            cal_unrelated.calibrated_residual
        );
        assert!(cal_unrelated.calibrated_residual <= 1.0);
    }

    #[test]
    fn calibrated_residual_none_on_empty_trajectory() {
        assert!(shape_calibrated_residual(&toy_rho(), &Expr::Const(1.0), &[], 1).is_none());
    }
}
