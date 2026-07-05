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
//! ## Why a narrow heuristic
//!
//! A full symmetry-detection algorithm would compute Lie derivatives and check
//! whether the expression is annihilated. That's the right long-term path but
//! substantial work. This module instead targets the 90%-of-the-signal subset
//! the showcase produces, as a pragmatic plateau-breaker.

use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};

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
    SymmetryDescriptor::none()
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
}
