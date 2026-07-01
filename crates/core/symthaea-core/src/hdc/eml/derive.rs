// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Paper-aligned EML derivations.
//!
//! This module intentionally exposes constructions over the pure grammar
//! `S -> 1 | Var(name) | eml(S, S)`. It is a named derivation library, not a
//! new evaluator: every constructor returns an [`EmlExpr`] using only the
//! primitive EML node plus the distinguished terminal `1` and variables.

use super::EmlExpr;

pub fn one() -> EmlExpr {
    EmlExpr::terminal_one()
}

pub fn var(name: impl Into<String>) -> EmlExpr {
    EmlExpr::terminal_var(name)
}

pub fn eml(left: EmlExpr, right: EmlExpr) -> EmlExpr {
    EmlExpr::eml(left, right)
}

/// `exp(x) = eml(x, 1)`.
pub fn exp(x: EmlExpr) -> EmlExpr {
    eml(x, one())
}

/// `ln(x) = eml(1, eml(eml(1, x), 1))`.
pub fn ln(x: EmlExpr) -> EmlExpr {
    eml(one(), exp(eml(one(), x)))
}

/// Domain-restricted subtraction: `x - y = eml(ln(x), exp(y))`.
///
/// This identity is useful, but unlike `exp` and `ln` it is not globally
/// total under principal-branch evaluation because the intermediate `ln(x)`
/// is undefined at zero.
pub fn sub(left: EmlExpr, right: EmlExpr) -> EmlExpr {
    eml(ln(left), exp(right))
}

/// Additive inverse placeholder.
///
/// The paper derives negation from the EML basis, but Symthaea has not yet
/// certified a compact derivation that avoids singular intermediate values in
/// the current principal-branch evaluator.
pub fn neg(_x: EmlExpr) -> Option<EmlExpr> {
    None
}

/// Addition placeholder.
pub fn add(_left: EmlExpr, _right: EmlExpr) -> Option<EmlExpr> {
    None
}

/// Multiplication placeholder.
pub fn mul(_left: EmlExpr, _right: EmlExpr) -> Option<EmlExpr> {
    None
}

/// Division placeholder.
pub fn div(_left: EmlExpr, _right: EmlExpr) -> Option<EmlExpr> {
    None
}

/// Squaring placeholder.
pub fn square(_x: EmlExpr) -> Option<EmlExpr> {
    None
}

/// Reciprocal placeholder.
pub fn reciprocal(_x: EmlExpr) -> Option<EmlExpr> {
    None
}

/// `x^n` for integer `n` using the current safe finite derivation subset.
pub fn integer_power(x: EmlExpr, n: i64) -> Option<EmlExpr> {
    match n {
        0 => Some(one()),
        1 => Some(x),
        _ => None,
    }
}

/// Complex-principal `sqrt(x)` placeholder.
///
/// The paper derives radicals from the EML basis, but Symthaea does not yet
/// carry a compact rational-exponent derivation (`1/2`) in the production EML
/// library. Keep this explicit so coverage reports can distinguish "not yet
/// derived" from "not part of the pure grammar".
pub fn sqrt(_x: EmlExpr) -> Option<EmlExpr> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::complex::Complex;
    use crate::hdc::eml::eval_complex;
    use std::collections::HashMap;

    fn assert_complex_close(got: Complex, expected: Complex) {
        assert!(
            got.approx_eq(&expected, 1e-9),
            "got {got:?}, expected {expected:?}"
        );
    }

    #[test]
    fn exp_derivation_matches_complex_exp() {
        let x = Complex::new(0.25, 0.75);
        let expr = exp(var("x"));
        let got = eval_complex(&expr, &HashMap::from([("x", x)])).unwrap();
        assert_complex_close(got, x.exp());
    }

    #[test]
    fn ln_derivation_matches_complex_ln() {
        let x = Complex::new(2.0, 0.5);
        let expr = ln(var("x"));
        let got = eval_complex(&expr, &HashMap::from([("x", x)])).unwrap();
        assert_complex_close(got, x.ln());
    }

    #[test]
    fn subtraction_derivation_matches_complex_subtraction_when_left_is_nonzero() {
        let x = Complex::new(2.0, 0.25);
        let y = Complex::new(1.5, -0.5);
        let vars = HashMap::from([("x", x), ("y", y)]);

        assert_complex_close(
            eval_complex(&sub(var("x"), var("y")), &vars).unwrap(),
            x - y,
        );
    }

    #[test]
    fn uncertified_derivations_stay_explicitly_unavailable() {
        assert!(neg(var("x")).is_none());
        assert!(add(var("x"), var("y")).is_none());
        assert!(mul(var("x"), var("y")).is_none());
        assert!(div(var("x"), var("y")).is_none());
        assert!(square(var("x")).is_none());
        assert!(reciprocal(var("x")).is_none());
        assert!(integer_power(var("x"), -1).is_none());
        assert!(integer_power(var("x"), 2).is_none());
        assert!(integer_power(var("x"), 3).is_none());
        assert!(sqrt(var("x")).is_none());
    }
}
