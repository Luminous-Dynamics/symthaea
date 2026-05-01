// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Dimensional Inference
//!
//! Walks a ConjectureEngine [`Expr`] tree and propagates SI dimensions from
//! per-variable unit annotations. The result populates the
//! [`PhysicsEquation::dimensions`] field at recognition time, unlocking the
//! bridge's dimensional similarity axis (which would otherwise be inert
//! because [`recognize_expr`] hardcoded `DimensionalSignature::DIMENSIONLESS`).
//!
//! ## Why this matters
//!
//! Recognition similarity scoring is a weighted sum:
//! ```text
//! score = 0.4 · structural + 0.3 · symmetry + 0.2 · dimensional + 0.1 · full
//! ```
//!
//! Without dimensional inference, the dimensional component is always zero
//! against any non-dimensionless catalog entry (because the dimensional
//! encoder maps `DIMENSIONLESS` to the zero hypervector). The structural
//! axis caps recognition at ~0.70 within the polynomial-with-negation
//! family, and the dimensional axis can't disambiguate.
//!
//! With dimensional inference: a discovered formula whose units genuinely
//! match a catalog entry's `dimensions` field gains the full +0.20
//! dimensional contribution, while structurally-similar but
//! dimensionally-mismatched entries gain nothing. This breaks the plateau
//! decisively for the correct entry.
//!
//! ## Inference rules
//!
//! - `Var(name)` → look up `name` in `var_units`; if absent, treat as dimensionless
//! - `Const(_)` → dimensionless (numeric literals carry no units)
//! - `Add(a, b)` / `Sub(a, b)` → both sides must match; result is `dim(a)`.
//!   If they don't match, the formula is dimensionally inconsistent and we
//!   return `Inconsistent` (signaling the caller to fall back to dimensionless)
//! - `Mul(a, b)` → `dim(a) + dim(b)`
//! - `Div(a, b)` → `dim(a) - dim(b)`
//! - `Pow(base, Const(k))` → `k · dim(base)` (only integer exponents propagate
//!   cleanly; fractional exponents like `√` go through Sqrt path)
//! - `Pow(base, _other)` → if base is dimensionless, result is dimensionless;
//!   otherwise inconsistent (variable exponents on dimensional bases are nonsense)
//! - `Sqrt(a)` → `½ · dim(a)`; if `dim(a)` exponents are not all even, the
//!   result is fractional which `i8` can't represent → return `Inconsistent`
//! - `Func(Log, _)` / `Func(Exp, _)` / `Func(Sin, _)` / `Func(Cos, _)` →
//!   argument must be dimensionless; result is dimensionless
//! - `Func(Abs, a)` / `Func(Floor, a)` → `dim(a)`
//! - `Sum` (Σ) → `dim(body)`
//!
//! ## Inconsistency handling
//!
//! When a formula is dimensionally inconsistent (mismatched Add operands,
//! odd exponents under Sqrt, etc.), [`infer_dimensions`] returns
//! [`InferenceResult::Inconsistent`]. Callers should fall back to
//! `DimensionalSignature::DIMENSIONLESS` for the recognition query in this
//! case — the formula is still structurally meaningful even if its units
//! don't propagate cleanly.

use std::collections::HashMap;

use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};

use crate::types::DimensionalSignature;

/// A unit annotation map: variable name → SI dimension.
///
/// Convenience type for callers building up unit context.
pub type UnitMap = HashMap<String, DimensionalSignature>;

/// Result of dimensional inference on an expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InferenceResult {
    /// Dimensions inferred successfully.
    Inferred(DimensionalSignature),
    /// Formula is dimensionally inconsistent (e.g., adding length to time).
    /// Callers should fall back to `DIMENSIONLESS` for recognition queries.
    Inconsistent,
}

impl InferenceResult {
    /// Extract the inferred dimensions, falling back to `DIMENSIONLESS`
    /// on inconsistency.
    pub fn or_dimensionless(self) -> DimensionalSignature {
        match self {
            InferenceResult::Inferred(d) => d,
            InferenceResult::Inconsistent => DimensionalSignature::DIMENSIONLESS,
        }
    }

    /// `true` if dimensions were inferred (regardless of value).
    pub fn is_inferred(&self) -> bool {
        matches!(self, InferenceResult::Inferred(_))
    }
}

/// Infer the SI dimensions of a ConjectureEngine `Expr` given per-variable
/// unit annotations.
///
/// See module documentation for the inference rules. Variables without
/// an entry in `var_units` are treated as dimensionless.
///
/// # Example
///
/// ```ignore
/// use std::collections::HashMap;
/// use symthaea_physics_bridge::{infer_dimensions, DimensionalSignature, InferenceResult};
///
/// // ½ m v² (kinetic energy)
/// let expr = /* ... build the AST ... */;
/// let mut units = HashMap::new();
/// units.insert("m".to_string(), DimensionalSignature::MASS);
/// units.insert("v".to_string(), DimensionalSignature::VELOCITY);
///
/// let result = infer_dimensions(&expr, &units);
/// match result {
///     InferenceResult::Inferred(dim) => assert_eq!(dim, DimensionalSignature::ENERGY),
///     InferenceResult::Inconsistent => panic!("expected energy"),
/// }
/// ```
pub fn infer_dimensions(expr: &Expr, var_units: &UnitMap) -> InferenceResult {
    use InferenceResult::*;

    match expr {
        Expr::Var(name) => Inferred(
            var_units
                .get(name)
                .copied()
                .unwrap_or(DimensionalSignature::DIMENSIONLESS),
        ),

        Expr::Const(_) => Inferred(DimensionalSignature::DIMENSIONLESS),

        Expr::BinOp(op, lhs, rhs) => {
            let left = infer_dimensions(lhs, var_units);
            let right = infer_dimensions(rhs, var_units);

            match (left, right) {
                (Inconsistent, _) | (_, Inconsistent) => Inconsistent,
                (Inferred(a), Inferred(b)) => match op {
                    BinOp::Add | BinOp::Sub => {
                        if a == b {
                            Inferred(a)
                        } else {
                            Inconsistent
                        }
                    }
                    BinOp::Mul => Inferred(a.add(&b)),
                    BinOp::Div => Inferred(a.sub(&b)),
                    BinOp::Pow => {
                        // Only constant integer exponents propagate cleanly.
                        if let Expr::Const(k) = rhs.as_ref() {
                            // Round to nearest int for fractional Const exponents
                            // like 0.5 (Sqrt is encoded as Pow(_, 0.5) sometimes).
                            // If k is exactly half-integer, we handle Sqrt-like
                            // cases via a separate path below.
                            if (k - k.round()).abs() < 1e-9 {
                                let int_k = *k as i8;
                                match a.scale(int_k) {
                                    Some(scaled) => Inferred(scaled),
                                    None => Inconsistent,
                                }
                            } else if (k - 0.5).abs() < 1e-9 {
                                // Square root via Pow(x, 0.5)
                                halve_dimensions(&a)
                            } else if (k + 0.5).abs() < 1e-9 {
                                // Inverse square root via Pow(x, -0.5)
                                match halve_dimensions(&a) {
                                    Inferred(d) => match d.scale(-1) {
                                        Some(neg) => Inferred(neg),
                                        None => Inconsistent,
                                    },
                                    Inconsistent => Inconsistent,
                                }
                            } else {
                                // Other fractional exponents: only valid if
                                // the base is dimensionless
                                if a.is_dimensionless() {
                                    Inferred(DimensionalSignature::DIMENSIONLESS)
                                } else {
                                    Inconsistent
                                }
                            }
                        } else {
                            // Variable exponent: only valid if base is dimensionless
                            if a.is_dimensionless() {
                                Inferred(DimensionalSignature::DIMENSIONLESS)
                            } else {
                                Inconsistent
                            }
                        }
                    }
                },
            }
        }

        Expr::Func(f, arg) => {
            let inner = infer_dimensions(arg, var_units);
            match (f, inner) {
                (_, Inconsistent) => Inconsistent,
                (UnaryFn::Sqrt, Inferred(d)) => halve_dimensions(&d),
                (UnaryFn::Log | UnaryFn::Exp | UnaryFn::Sin | UnaryFn::Cos, Inferred(d)) => {
                    // Argument must be dimensionless; result is dimensionless.
                    if d.is_dimensionless() {
                        Inferred(DimensionalSignature::DIMENSIONLESS)
                    } else {
                        Inconsistent
                    }
                }
                (UnaryFn::Abs | UnaryFn::Floor, Inferred(d)) => Inferred(d),
            }
        }

        Expr::Sum(body, _var) => {
            // Σ_k body(k) — preserves dimensions of body
            infer_dimensions(body, var_units)
        }
    }
}

/// Halve all exponents of a dimensional signature.
///
/// Returns `Inconsistent` if any exponent is odd (since fractional
/// dimensions can't be represented in `i8`).
fn halve_dimensions(d: &DimensionalSignature) -> InferenceResult {
    let arr = d.as_array();
    let mut out = [0i8; 7];
    for (i, exp) in arr.iter().enumerate() {
        if exp % 2 != 0 {
            return InferenceResult::Inconsistent;
        }
        out[i] = exp / 2;
    }
    InferenceResult::Inferred(DimensionalSignature::from_array(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};

    /// Build a unit map quickly for tests.
    fn units(pairs: &[(&str, DimensionalSignature)]) -> UnitMap {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn test_constant_is_dimensionless() {
        let expr = Expr::Const(3.14);
        let result = infer_dimensions(&expr, &UnitMap::new());
        assert_eq!(
            result,
            InferenceResult::Inferred(DimensionalSignature::DIMENSIONLESS)
        );
    }

    #[test]
    fn test_var_lookup() {
        let expr = Expr::Var("v".into());
        let u = units(&[("v", DimensionalSignature::VELOCITY)]);
        let result = infer_dimensions(&expr, &u);
        assert_eq!(
            result,
            InferenceResult::Inferred(DimensionalSignature::VELOCITY)
        );
    }

    #[test]
    fn test_unknown_var_is_dimensionless() {
        let expr = Expr::Var("unknown".into());
        let result = infer_dimensions(&expr, &UnitMap::new());
        assert_eq!(
            result,
            InferenceResult::Inferred(DimensionalSignature::DIMENSIONLESS)
        );
    }

    #[test]
    fn test_velocity_squared_is_l2_t_minus_2() {
        // v²
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("v".into())),
            Box::new(Expr::Const(2.0)),
        );
        let u = units(&[("v", DimensionalSignature::VELOCITY)]);
        let result = infer_dimensions(&expr, &u);
        let expected = DimensionalSignature::from_array([0, 2, -2, 0, 0, 0, 0]);
        assert_eq!(result, InferenceResult::Inferred(expected));
    }

    #[test]
    fn test_kinetic_energy_inference() {
        // ½ m v²
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(0.5)),
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("m".into())),
                Box::new(Expr::BinOp(
                    BinOp::Pow,
                    Box::new(Expr::Var("v".into())),
                    Box::new(Expr::Const(2.0)),
                )),
            )),
        );
        let u = units(&[
            ("m", DimensionalSignature::MASS),
            ("v", DimensionalSignature::VELOCITY),
        ]);
        let result = infer_dimensions(&expr, &u);
        assert_eq!(
            result,
            InferenceResult::Inferred(DimensionalSignature::ENERGY)
        );
    }

    #[test]
    fn test_kepler_orbital_energy_inference() {
        // ½(vx² + vy²) - 1/r
        // vx, vy: velocity (LT⁻¹)
        // r: length (L)
        // Expected: energy density without mass = L²T⁻² (consistent)
        // Note: Without mass, this is energy/mass not full energy. The
        // dimensional axis still matches "specific energy" which is what
        // the Kepler formulation uses (E/m = ½v² - GM/r).
        let v_squared = |name: &str| -> Expr {
            Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var(name.into())),
                Box::new(Expr::Const(2.0)),
            )
        };
        let expr = Expr::BinOp(
            BinOp::Sub,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Const(0.5)),
                Box::new(Expr::BinOp(
                    BinOp::Add,
                    Box::new(v_squared("vx")),
                    Box::new(v_squared("vy")),
                )),
            )),
            Box::new(Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::Const(1.0)),
                Box::new(Expr::Var("r".into())),
            )),
        );
        let u = units(&[
            ("vx", DimensionalSignature::VELOCITY),
            ("vy", DimensionalSignature::VELOCITY),
            ("r", DimensionalSignature::LENGTH),
        ]);
        let result = infer_dimensions(&expr, &u);
        // ½(vx² + vy²) → L²T⁻²
        // 1/r → L⁻¹
        // L²T⁻² - L⁻¹ → INCONSISTENT (different dimensions!)
        // This is the right answer — the Kepler orbital energy formula
        // E = ½v² - 1/r is dimensionally inconsistent UNLESS you include
        // GM explicitly. Pure ½v² - 1/r requires the implicit assumption
        // that GM = 1 (natural units).
        assert!(matches!(result, InferenceResult::Inconsistent));
    }

    #[test]
    fn test_kepler_with_explicit_constants_inference() {
        // ½ v² - GM/r where GM has dimensions [L³T⁻²]
        // (same as Newton's gravitational parameter μ = GM)
        let v2 = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("v".into())),
            Box::new(Expr::Const(2.0)),
        );
        let kinetic = Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(0.5)), Box::new(v2));
        let potential = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Var("mu".into())),
            Box::new(Expr::Var("r".into())),
        );
        let expr = Expr::BinOp(BinOp::Sub, Box::new(kinetic), Box::new(potential));
        let u = units(&[
            ("v", DimensionalSignature::VELOCITY),
            ("r", DimensionalSignature::LENGTH),
            // Standard gravitational parameter: GM has dims L³T⁻²
            (
                "mu",
                DimensionalSignature::from_array([0, 3, -2, 0, 0, 0, 0]),
            ),
        ]);
        let result = infer_dimensions(&expr, &u);
        // ½v² → L²T⁻²
        // mu/r → L³T⁻²/L = L²T⁻²
        // L²T⁻² - L²T⁻² → L²T⁻²  ✓
        let expected = DimensionalSignature::from_array([0, 2, -2, 0, 0, 0, 0]);
        assert_eq!(result, InferenceResult::Inferred(expected));
    }

    #[test]
    fn test_sin_of_dimensionless_is_ok() {
        // sin(2π·t/T) where t/T is dimensionless
        let expr = Expr::Func(UnaryFn::Sin, Box::new(Expr::Var("phase".into())));
        let u = units(&[("phase", DimensionalSignature::DIMENSIONLESS)]);
        let result = infer_dimensions(&expr, &u);
        assert_eq!(
            result,
            InferenceResult::Inferred(DimensionalSignature::DIMENSIONLESS)
        );
    }

    #[test]
    fn test_sin_of_dimensional_is_inconsistent() {
        // sin(t) where t has time units — physically nonsense
        let expr = Expr::Func(UnaryFn::Sin, Box::new(Expr::Var("t".into())));
        let u = units(&[("t", DimensionalSignature::TIME)]);
        let result = infer_dimensions(&expr, &u);
        assert_eq!(result, InferenceResult::Inconsistent);
    }

    #[test]
    fn test_log_of_dimensionless_is_ok() {
        let expr = Expr::Func(UnaryFn::Log, Box::new(Expr::Var("ratio".into())));
        let u = units(&[("ratio", DimensionalSignature::DIMENSIONLESS)]);
        let result = infer_dimensions(&expr, &u);
        assert_eq!(
            result,
            InferenceResult::Inferred(DimensionalSignature::DIMENSIONLESS)
        );
    }

    #[test]
    fn test_sqrt_of_area_is_length() {
        // √(area) → length
        let expr = Expr::Func(UnaryFn::Sqrt, Box::new(Expr::Var("a".into())));
        let u = units(&[("a", DimensionalSignature::AREA)]);
        let result = infer_dimensions(&expr, &u);
        assert_eq!(
            result,
            InferenceResult::Inferred(DimensionalSignature::LENGTH)
        );
    }

    #[test]
    fn test_sqrt_of_odd_exponent_is_inconsistent() {
        // √v where v has L¹T⁻¹ — odd L exponent
        let expr = Expr::Func(UnaryFn::Sqrt, Box::new(Expr::Var("v".into())));
        let u = units(&[("v", DimensionalSignature::VELOCITY)]);
        let result = infer_dimensions(&expr, &u);
        assert_eq!(result, InferenceResult::Inconsistent);
    }

    #[test]
    fn test_inconsistent_addition() {
        // length + time — physically nonsense
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("d".into())),
            Box::new(Expr::Var("t".into())),
        );
        let u = units(&[
            ("d", DimensionalSignature::LENGTH),
            ("t", DimensionalSignature::TIME),
        ]);
        let result = infer_dimensions(&expr, &u);
        assert_eq!(result, InferenceResult::Inconsistent);
    }

    #[test]
    fn test_or_dimensionless_fallback() {
        let inconsistent = InferenceResult::Inconsistent;
        assert_eq!(
            inconsistent.or_dimensionless(),
            DimensionalSignature::DIMENSIONLESS
        );

        let inferred = InferenceResult::Inferred(DimensionalSignature::ENERGY);
        assert_eq!(inferred.or_dimensionless(), DimensionalSignature::ENERGY);
    }

    #[test]
    fn test_harmonic_oscillator_energy_inference() {
        // ½(x² + v²) where x: length, v: velocity
        // x² → L², v² → L²T⁻². These differ → Inconsistent.
        // Same as Kepler: pure x² + v² assumes natural units (k=m=1).
        // The honest answer is "inconsistent in raw form."
        let x_sq = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Const(2.0)),
        );
        let v_sq = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("v".into())),
            Box::new(Expr::Const(2.0)),
        );
        let expr = Expr::BinOp(BinOp::Add, Box::new(x_sq), Box::new(v_sq));
        let u = units(&[
            ("x", DimensionalSignature::LENGTH),
            ("v", DimensionalSignature::VELOCITY),
        ]);
        let result = infer_dimensions(&expr, &u);
        assert!(matches!(result, InferenceResult::Inconsistent));
    }

    #[test]
    fn test_force_law_inference() {
        // F = m·a (Newton's second law)
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".into())),
            Box::new(Expr::Var("a".into())),
        );
        let u = units(&[
            ("m", DimensionalSignature::MASS),
            ("a", DimensionalSignature::ACCELERATION),
        ]);
        let result = infer_dimensions(&expr, &u);
        assert_eq!(
            result,
            InferenceResult::Inferred(DimensionalSignature::FORCE)
        );
    }
}
