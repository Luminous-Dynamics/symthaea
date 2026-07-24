// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symmetry Inference from Discovered Expressions
//!
//! Walks a ConjectureEngine [`Expr`] and heuristically detects Lie-group
//! symmetries so the recognition query can declare them. Without this, the
//! query hardcodes `SymmetryDescriptor::none()` and any catalog entry
//! claiming its true symmetry drops ~0.20 on the symmetry similarity axis.
//!
//! ## What it detects
//!
//! Two narrow but high-value patterns that cover the Ramanujan Protocol
//! showcase's canonical invariants:
//!
//! 1. **Sum of squared variables** — `x² + y²`, `x² + y² + z²`, `px² + py² + pz²`
//!    → SO(n) where n = number of distinct squared variables.
//!    Matches Harmonic Oscillator, kinetic energy, `r² = x² + y² + z²`.
//!
//! 2. **2D antisymmetric cross product** — `a·b - c·d` where `{a,c}` and
//!    `{b,d}` are disjoint pairs → SO(2).
//!    Matches `x·vy - y·vx` (angular momentum z-component).
//!
//! Everything else returns [`SymmetryDescriptor::none()`]. Notably, `x - ln(x)`,
//! polynomial + log combinations, and anything involving transcendentals is
//! NOT classified — we prefer to undersell symmetry than to claim false
//! rotational invariance on a transcendental like the Lotka-Volterra invariant.
//!
//! ## Beyond the two structural heuristics: numeric Lie-derivative fallback
//!
//! The two heuristics above are exact-shape pattern matches — fast, but blind
//! to anything that isn't literally a monomial sum-of-squares or cross-product
//! (e.g. `sqrt(x²+y²)`, `1/(x²+y²)`, or any other radially-symmetric function
//! of the same invariant). [`infer_symmetry_numeric`] closes that gap using the
//! Lie-theory machinery in `symthaea_core::hdc::lie_theory`: for a candidate
//! group action (an so(n) generator acting on some subset of the expression's
//! variables), the expression is invariant iff its Lie derivative along the
//! generator's flow vanishes everywhere — evaluated numerically (finite-
//! difference gradient, matching the same technique
//! `conjecture_engine::gp_support::lie_derivative_variance` already uses for
//! checking conservation along a *time* flow; here the flow is a *symmetry*
//! generator instead) at several sampled points rather than proven
//! symbolically. This is deliberately bounded, not a full search:
//!
//! - joint SO(n) acting on all of the expression's variables at once (n ≤ 6,
//!   for generator-count cost), and
//! - SO(2) acting on each pair of variables independently,
//!
//! which strictly covers the two structural heuristics' cases (any function of
//! `x²+y²+...`, not just the sum itself) while staying tractable. It does not
//! search arbitrary variable subsets/groupings, and it is a numerical check
//! (some sampled points, fixed tolerance) rather than a symbolic proof — treat
//! a positive result as strong evidence, not a certificate.
//!
//! ## Why the structural heuristics still run first
//!
//! They're exact and essentially free; the numeric fallback only runs when
//! they don't fire, keeping the common cases fast and precise.

use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};
use symthaea_core::hdc::lie_theory::{self, LieAlgebra};

use crate::types::{LieGroup, SymmetryDescriptor};

/// Infer a best-effort symmetry descriptor from a discovered expression.
pub fn infer_symmetry(expr: &Expr) -> SymmetryDescriptor {
    if let Some(n) = detect_sum_of_squares(expr)
        && (2..=10).contains(&n)
    {
        return SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(n)], false);
    }
    if detect_2d_antisymmetric_cross(expr) {
        return SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(2)], false);
    }
    infer_symmetry_numeric(expr)
}

/// The generic numeric fallback described in the module docs. Public so
/// callers/tests can exercise it directly, independent of whether the fast
/// structural heuristics already fired.
pub fn infer_symmetry_numeric(expr: &Expr) -> SymmetryDescriptor {
    let mut var_names = Vec::new();
    collect_var_names(expr, &mut var_names);

    if var_names.len() < 2 {
        return SymmetryDescriptor::none();
    }

    // Joint rotation of all variables (bounded to so(6) for generator-count cost).
    let n = var_names.len().min(6);
    let joint_algebra = lie_theory::so_n(n);
    let joint_indices: Vec<usize> = (0..n).collect();
    if is_lie_invariant(expr, &var_names, &joint_indices, &joint_algebra) {
        return SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(n as u8)], false);
    }

    // Pairwise SO(2) on each distinct pair of variables.
    let so2 = lie_theory::so_n(2);
    for i in 0..var_names.len() {
        for j in (i + 1)..var_names.len() {
            if is_lie_invariant(expr, &var_names, &[i, j], &so2) {
                return SymmetryDescriptor::from_lie_groups(vec![LieGroup::SO(2)], false);
            }
        }
    }

    SymmetryDescriptor::none()
}

/// Collect free variable names referenced by `expr`, first-seen order, deduplicated.
fn collect_var_names(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Var(name) => {
            if !out.contains(name) {
                out.push(name.clone());
            }
        }
        Expr::Const(_) => {}
        Expr::BinOp(_, l, r) => {
            collect_var_names(l, out);
            collect_var_names(r, out);
        }
        Expr::Func(_, arg) => collect_var_names(arg, out),
        Expr::Sum(body, var) => {
            collect_var_names(body, out);
            if !out.contains(var) {
                out.push(var.clone());
            }
        }
    }
}

/// Central-difference gradient of `expr` at `state`, w.r.t. `var_names` (same
/// binding order as `state`). Mirrors
/// `conjecture_engine::gp_support::fd_gradient`, reimplemented locally since
/// that helper is `pub(crate)` to symthaea-core.
fn fd_gradient(expr: &Expr, state: &[f64], var_names: &[String]) -> Vec<f64> {
    const EPS: f64 = 1e-5;
    let mut grad = Vec::with_capacity(var_names.len());
    for i in 0..var_names.len() {
        let mut plus = state.to_vec();
        let mut minus = state.to_vec();
        plus[i] += EPS;
        minus[i] -= EPS;
        let bindings_plus: Vec<(&str, f64)> = var_names
            .iter()
            .map(|s| s.as_str())
            .zip(plus.iter().copied())
            .collect();
        let bindings_minus: Vec<(&str, f64)> = var_names
            .iter()
            .map(|s| s.as_str())
            .zip(minus.iter().copied())
            .collect();
        grad.push((expr.eval(&bindings_plus) - expr.eval(&bindings_minus)) / (2.0 * EPS));
    }
    grad
}

/// Deterministic xorshift64 sampler — no external RNG dependency, reproducible
/// across runs (important for test stability).
fn sample_states(dim: usize, count: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let mut x = state;
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    state = x;
                    let unit = (x >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
                    // [0.5, 3.0) — strictly positive, avoids the origin (where
                    // e.g. 1/r and ln(x) are singular) while still probing a
                    // range wide enough to distinguish real invariance from
                    // a coincidental zero at a single point.
                    0.5 + unit * 2.5
                })
                .collect()
        })
        .collect()
}

/// Numerically test whether `expr` is invariant under the group generated by
/// `algebra`, acting on the variables at `active_indices` (all other
/// variables held fixed). Requires the normalized squared Lie derivative to
/// be below tolerance at every sampled point, for every generator — and
/// requires at least one sample to be "informative" (non-vanishing gradient
/// on the active subspace), so an expression that's merely locally flat at
/// every sampled point doesn't get misread as globally invariant.
fn is_lie_invariant(
    expr: &Expr,
    var_names: &[String],
    active_indices: &[usize],
    algebra: &LieAlgebra,
) -> bool {
    const TOLERANCE: f64 = 1e-6;
    const MIN_GRADIENT_MAG_SQ: f64 = 1e-8;
    let samples = sample_states(var_names.len(), 8, 0x9E3779B97F4A7C15);

    let mut informative = false;
    for generator in &algebra.basis {
        for state in &samples {
            let grad = fd_gradient(expr, state, var_names);
            if grad.iter().any(|g| !g.is_finite()) {
                return false;
            }

            // flow[active_indices[gi]] = sum_gj generator[gi][gj] * state[active_indices[gj]]
            let mut flow = vec![0.0; var_names.len()];
            for (gi, &vi) in active_indices.iter().enumerate() {
                let mut acc = 0.0;
                for (gj, &vj) in active_indices.iter().enumerate() {
                    acc += generator[gi][gj] * state[vj];
                }
                flow[vi] = acc;
            }

            let lie: f64 = grad.iter().zip(flow.iter()).map(|(g, f)| g * f).sum();
            let grad_sq: f64 = active_indices.iter().map(|&vi| grad[vi] * grad[vi]).sum();
            if !lie.is_finite() || !grad_sq.is_finite() {
                return false;
            }
            if grad_sq < MIN_GRADIENT_MAG_SQ {
                continue;
            }
            informative = true;
            if (lie * lie) / grad_sq > TOLERANCE {
                return false;
            }
        }
    }
    informative
}

/// Flatten an `Add` tree into a vector of leaf terms. Unlike the recognize
/// module's `flatten_add`, we stop at all non-Add boundaries *without*
/// descending into Sub — the two have different semantics for symmetry
/// inference: a `Sub` at the top level means the expression is not a simple
/// sum and shouldn't be treated as a sum-of-squares candidate.
fn flatten_add_only<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
    if let Expr::BinOp(BinOp::Add, l, r) = expr {
        flatten_add_only(l, out);
        flatten_add_only(r, out);
    } else {
        out.push(expr);
    }
}

/// If `expr` is a sum of k ≥ 1 distinct squared variables (i.e.
/// `Power(Var(xᵢ), 2)` with unique `xᵢ`), return k. Otherwise return None.
///
/// Accepts extra coefficients via `Mul(Const(c), Power(Var, 2))` but rejects
/// any non-square term — so `x² + y²` matches (returns 2) but
/// `x² + y² + y³` does not.
fn detect_sum_of_squares(expr: &Expr) -> Option<u8> {
    let mut terms = Vec::new();
    flatten_add_only(expr, &mut terms);
    if terms.len() < 2 {
        return None;
    }

    let mut seen = Vec::new();
    for t in terms {
        let var = squared_var(t)?;
        // Reject duplicates — they'd over-count the symmetry group dimension.
        if seen.contains(&var) {
            return None;
        }
        seen.push(var);
    }
    Some(seen.len() as u8)
}

/// If `expr` is `Power(Var(name), 2)` or `Const * Power(Var(name), 2)`,
/// return the variable name; otherwise None.
fn squared_var(expr: &Expr) -> Option<String> {
    match expr {
        Expr::BinOp(BinOp::Pow, base, exp) => {
            if let (Expr::Var(name), Expr::Const(k)) = (base.as_ref(), exp.as_ref())
                && (*k - 2.0).abs() < 1e-12
            {
                return Some(name.clone());
            }
            None
        }
        Expr::BinOp(BinOp::Mul, l, r) => {
            // Allow Const · Power(Var, 2)
            match (l.as_ref(), r.as_ref()) {
                (Expr::Const(_), other) | (other, Expr::Const(_)) => squared_var(other),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Detect `a·b - c·d` where `{a,c}` and `{b,d}` are two distinct variable
/// pairs, modeling an antisymmetric cross product.
///
/// Example matches: `x*vy - y*vx`, `p*q - q*p` → SO(2). The match is
/// structural only; we don't verify that the variables form an actual
/// canonical pair (position/momentum), just that the shape is antisymmetric.
fn detect_2d_antisymmetric_cross(expr: &Expr) -> bool {
    let Expr::BinOp(BinOp::Sub, l, r) = expr else {
        return false;
    };
    let Expr::BinOp(BinOp::Mul, l1, l2) = l.as_ref() else {
        return false;
    };
    let Expr::BinOp(BinOp::Mul, r1, r2) = r.as_ref() else {
        return false;
    };
    let (l1v, l2v, r1v, r2v) = match (l1.as_ref(), l2.as_ref(), r1.as_ref(), r2.as_ref()) {
        (Expr::Var(a), Expr::Var(b), Expr::Var(c), Expr::Var(d)) => (a, b, c, d),
        _ => return false,
    };
    // {l1, r1} and {l2, r2} should each be a pair of distinct variables,
    // and the two pairs together must use 4 distinct names (e.g. x, vy, y, vx).
    let all = [l1v, l2v, r1v, r2v];
    let unique_count = all
        .iter()
        .enumerate()
        .filter(|(i, name)| !all[..*i].contains(name))
        .count();
    if unique_count != 4 {
        return false;
    }
    // The shape `a·b - c·d` is antisymmetric iff swapping `a↔c` and `b↔d`
    // negates the expression, which is automatic when {a,c} and {b,d} are
    // the two pairs of the cross product.
    true
}

// Silence unused-import warning when UnaryFn is referenced only in rustdoc.
#[allow(dead_code)]
fn _unused(_: UnaryFn) {}

#[cfg(test)]
mod tests {
    use super::*;

    fn pow2(name: &str) -> Expr {
        Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var(name.to_string())),
            Box::new(Expr::Const(2.0)),
        )
    }

    fn add(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Add, Box::new(l), Box::new(r))
    }

    fn sub(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Sub, Box::new(l), Box::new(r))
    }

    fn mul(l: Expr, r: Expr) -> Expr {
        Expr::BinOp(BinOp::Mul, Box::new(l), Box::new(r))
    }

    fn var(name: &str) -> Expr {
        Expr::Var(name.to_string())
    }

    #[test]
    fn harmonic_oscillator_infers_so2() {
        let expr = add(pow2("x"), pow2("v"));
        let sym = infer_symmetry(&expr);
        assert_eq!(sym.lie_groups, vec![LieGroup::SO(2)]);
    }

    #[test]
    fn three_d_sum_of_squares_infers_so3() {
        let expr = add(add(pow2("x"), pow2("y")), pow2("z"));
        let sym = infer_symmetry(&expr);
        assert_eq!(sym.lie_groups, vec![LieGroup::SO(3)]);
    }

    #[test]
    fn angular_momentum_cartesian_infers_so2() {
        // x*vy - y*vx
        let expr = sub(mul(var("x"), var("vy")), mul(var("y"), var("vx")));
        let sym = infer_symmetry(&expr);
        assert_eq!(sym.lie_groups, vec![LieGroup::SO(2)]);
    }

    #[test]
    fn lv_invariant_stays_none() {
        // (x - ln(x)) + (y - ln(y)) — transcendental, must NOT claim rotational symmetry.
        let ln = |name: &str| Expr::Func(UnaryFn::Log, Box::new(var(name)));
        let expr = add(sub(var("x"), ln("x")), sub(var("y"), ln("y")));
        let sym = infer_symmetry(&expr);
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn duplicate_squared_var_rejected() {
        // x² + x² should not classify as SO(2) — duplicates indicate a
        // coefficient mismatch, not a rotational symmetry.
        let expr = add(pow2("x"), pow2("x"));
        let sym = infer_symmetry(&expr);
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn mixed_power_rejected() {
        // x² + y³ — not a pure sum of squares, shouldn't trigger.
        let expr = add(
            pow2("x"),
            Expr::BinOp(BinOp::Pow, Box::new(var("y")), Box::new(Expr::Const(3.0))),
        );
        let sym = infer_symmetry(&expr);
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn single_squared_var_not_enough() {
        // Just `x²` is scale-symmetric, not rotational — don't claim SO(1).
        let sym = infer_symmetry(&pow2("x"));
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn same_pair_cross_rejected() {
        // x*y - y*x is zero, structurally degenerate; should not classify.
        let expr = sub(mul(var("x"), var("y")), mul(var("y"), var("x")));
        let sym = infer_symmetry(&expr);
        assert!(sym.lie_groups.is_empty());
    }

    // ─── Numeric fallback: cases the structural heuristics CANNOT catch ───

    fn sqrt_of(e: Expr) -> Expr {
        Expr::Func(UnaryFn::Sqrt, Box::new(e))
    }

    #[test]
    fn radial_magnitude_infers_so2_via_numeric_fallback() {
        // sqrt(x^2 + y^2) is SO(2)-invariant, but is not a bare sum of
        // squares (it's sqrt of one) — the structural heuristic cannot match
        // this shape at all. This is the actual capability gain: any function
        // of the invariant, not just the invariant itself.
        let expr = sqrt_of(add(pow2("x"), pow2("y")));
        assert!(
            detect_sum_of_squares(&expr).is_none(),
            "sanity: structural heuristic must NOT match sqrt(..)"
        );
        let sym = infer_symmetry_numeric(&expr);
        assert_eq!(sym.lie_groups, vec![LieGroup::SO(2)]);
    }

    #[test]
    fn inverse_square_radial_infers_so3_via_numeric_fallback() {
        // 1/(x^2+y^2+z^2) — Kepler-potential-shaped, SO(3)-invariant, and
        // again not a literal sum-of-squares shape.
        let expr = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Const(1.0)),
            Box::new(add(add(pow2("x"), pow2("y")), pow2("z"))),
        );
        let sym = infer_symmetry_numeric(&expr);
        assert_eq!(sym.lie_groups, vec![LieGroup::SO(3)]);
    }

    #[test]
    fn non_invariant_product_stays_none_under_numeric_fallback() {
        // x*y is NOT rotationally invariant (it picks up x^2-y^2 cross terms
        // under a generic rotation) — must not false-positive.
        let expr = mul(var("x"), var("y"));
        let sym = infer_symmetry_numeric(&expr);
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn lv_invariant_stays_none_under_numeric_fallback_too() {
        // Re-run the transcendental Lotka-Volterra case directly against the
        // numeric fallback (not just the combined infer_symmetry), since this
        // is exactly the shape a Lie-derivative search could plausibly
        // false-positive on if the tolerance were too loose.
        let ln = |name: &str| Expr::Func(UnaryFn::Log, Box::new(var(name)));
        let expr = add(sub(var("x"), ln("x")), sub(var("y"), ln("y")));
        let sym = infer_symmetry_numeric(&expr);
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn single_variable_never_attempts_numeric_fallback() {
        // Fewer than 2 variables: no rotation is even well-defined.
        let sym = infer_symmetry_numeric(&pow2("x"));
        assert!(sym.lie_groups.is_empty());
    }

    #[test]
    fn four_var_sum_of_squares_infers_so4_via_numeric_fallback() {
        let expr = add(add(add(pow2("x"), pow2("y")), pow2("z")), pow2("w"));
        // Structural heuristic already covers this one (regression check),
        // but confirm the numeric path independently agrees.
        let sym = infer_symmetry_numeric(&expr);
        assert_eq!(sym.lie_groups, vec![LieGroup::SO(4)]);
    }
}
