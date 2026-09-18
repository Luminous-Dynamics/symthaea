// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed dimensional inference for authority-bearing physics validation.
//!
//! The existing `dimensional_inference` module is intentionally permissive for
//! semantic recognition: unknown variables become dimensionless and inconsistent
//! expressions may be mapped to dimensionless by callers. This module preserves
//! that discovery path while providing a separate API with no fallback. Unknown
//! variables and dimensional inconsistencies are explicit rejection states.

use std::collections::BTreeSet;

use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};

use crate::{UnitMap, types::DimensionalSignature};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StrictInferenceFailure {
    unknown_variables: BTreeSet<String>,
    inconsistent: bool,
}

impl StrictInferenceFailure {
    fn unknown(variable: String) -> Self {
        Self {
            unknown_variables: BTreeSet::from([variable]),
            inconsistent: false,
        }
    }

    fn inconsistent() -> Self {
        Self {
            unknown_variables: BTreeSet::new(),
            inconsistent: true,
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.unknown_variables.extend(other.unknown_variables);
        self.inconsistent |= other.inconsistent;
        self
    }

    pub fn unknown_variables(&self) -> &BTreeSet<String> {
        &self.unknown_variables
    }

    pub fn is_inconsistent(&self) -> bool {
        self.inconsistent
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrictInferenceResult {
    Inferred(DimensionalSignature),
    Rejected(StrictInferenceFailure),
}

impl StrictInferenceResult {
    pub fn is_inferred(&self) -> bool {
        matches!(self, Self::Inferred(_))
    }

    pub fn failure(&self) -> Option<&StrictInferenceFailure> {
        match self {
            Self::Inferred(_) => None,
            Self::Rejected(failure) => Some(failure),
        }
    }
}

/// Infer SI dimensions without any dimensionless fallback.
///
/// Unlike `infer_dimensions`, every variable must have an explicit unit
/// annotation. Any unknown variable or inconsistent operation returns
/// `StrictInferenceResult::Rejected` and therefore cannot be mistaken for a
/// valid dimensionless physical expression.
pub fn infer_dimensions_strict(expr: &Expr, var_units: &UnitMap) -> StrictInferenceResult {
    match infer(expr, var_units) {
        Ok(dimensions) => StrictInferenceResult::Inferred(dimensions),
        Err(failure) => StrictInferenceResult::Rejected(failure),
    }
}

fn infer(
    expr: &Expr,
    var_units: &UnitMap,
) -> Result<DimensionalSignature, StrictInferenceFailure> {
    match expr {
        Expr::Var(name) => var_units
            .get(name)
            .copied()
            .ok_or_else(|| StrictInferenceFailure::unknown(name.clone())),
        Expr::Const(_) => Ok(DimensionalSignature::DIMENSIONLESS),
        Expr::BinOp(op, lhs, rhs) => {
            let left = infer(lhs, var_units);
            let right = infer(rhs, var_units);
            let (a, b) = combine_binary_failures(left, right)?;
            match op {
                BinOp::Add | BinOp::Sub => {
                    if a == b {
                        Ok(a)
                    } else {
                        Err(StrictInferenceFailure::inconsistent())
                    }
                }
                BinOp::Mul => Ok(a.add(&b)),
                BinOp::Div => Ok(a.sub(&b)),
                BinOp::Pow => infer_power(a, rhs),
            }
        }
        Expr::Func(function, argument) => {
            let inner = infer(argument, var_units)?;
            match function {
                UnaryFn::Sqrt => halve_dimensions(&inner),
                UnaryFn::Log | UnaryFn::Exp | UnaryFn::Sin | UnaryFn::Cos => {
                    if inner.is_dimensionless() {
                        Ok(DimensionalSignature::DIMENSIONLESS)
                    } else {
                        Err(StrictInferenceFailure::inconsistent())
                    }
                }
                UnaryFn::Abs | UnaryFn::Floor => Ok(inner),
            }
        }
        Expr::Sum(body, _variable) => infer(body, var_units),
    }
}

fn combine_binary_failures(
    left: Result<DimensionalSignature, StrictInferenceFailure>,
    right: Result<DimensionalSignature, StrictInferenceFailure>,
) -> Result<(DimensionalSignature, DimensionalSignature), StrictInferenceFailure> {
    match (left, right) {
        (Ok(left), Ok(right)) => Ok((left, right)),
        (Err(left), Err(right)) => Err(left.merge(right)),
        (Err(failure), Ok(_)) | (Ok(_), Err(failure)) => Err(failure),
    }
}

fn infer_power(
    base_dimensions: DimensionalSignature,
    exponent: &Expr,
) -> Result<DimensionalSignature, StrictInferenceFailure> {
    let Expr::Const(exponent) = exponent else {
        return if base_dimensions.is_dimensionless() {
            Ok(DimensionalSignature::DIMENSIONLESS)
        } else {
            Err(StrictInferenceFailure::inconsistent())
        };
    };

    if (exponent - exponent.round()).abs() < 1e-9 {
        let integer = *exponent as i8;
        base_dimensions
            .scale(integer)
            .ok_or_else(StrictInferenceFailure::inconsistent)
    } else if (exponent - 0.5).abs() < 1e-9 {
        halve_dimensions(&base_dimensions)
    } else if (exponent + 0.5).abs() < 1e-9 {
        let half = halve_dimensions(&base_dimensions)?;
        half.scale(-1)
            .ok_or_else(StrictInferenceFailure::inconsistent)
    } else if base_dimensions.is_dimensionless() {
        Ok(DimensionalSignature::DIMENSIONLESS)
    } else {
        Err(StrictInferenceFailure::inconsistent())
    }
}

fn halve_dimensions(
    dimensions: &DimensionalSignature,
) -> Result<DimensionalSignature, StrictInferenceFailure> {
    let mut output = [0i8; 7];
    for (index, exponent) in dimensions.as_array().iter().enumerate() {
        if exponent % 2 != 0 {
            return Err(StrictInferenceFailure::inconsistent());
        }
        output[index] = exponent / 2;
    }
    Ok(DimensionalSignature::from_array(output))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(pairs: &[(&str, DimensionalSignature)]) -> UnitMap {
        pairs.iter().map(|(name, dim)| ((*name).to_owned(), *dim)).collect()
    }

    #[test]
    fn unknown_variable_is_rejected_not_dimensionless() {
        let result = infer_dimensions_strict(&Expr::Var("mystery".into()), &UnitMap::new());
        let failure = result.failure().expect("strict inference must reject unknown units");
        assert!(failure.unknown_variables().contains("mystery"));
        assert!(!failure.is_inconsistent());
    }

    #[test]
    fn all_unknown_variables_are_preserved() {
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("a".into())),
            Box::new(Expr::Var("b".into())),
        );
        let result = infer_dimensions_strict(&expr, &UnitMap::new());
        let failure = result.failure().unwrap();
        assert_eq!(
            failure.unknown_variables(),
            &BTreeSet::from(["a".to_owned(), "b".to_owned()])
        );
    }

    #[test]
    fn mismatched_addition_is_rejected() {
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("t".into())),
        );
        let result = infer_dimensions_strict(
            &expr,
            &units(&[
                ("x", DimensionalSignature::LENGTH),
                ("t", DimensionalSignature::TIME),
            ]),
        );
        let failure = result.failure().unwrap();
        assert!(failure.is_inconsistent());
        assert!(failure.unknown_variables().is_empty());
    }

    #[test]
    fn known_kinetic_energy_infers_energy() {
        let velocity_squared = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("v".into())),
            Box::new(Expr::Const(2.0)),
        );
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("m".into())),
            Box::new(velocity_squared),
        );
        assert_eq!(
            infer_dimensions_strict(
                &expr,
                &units(&[
                    ("m", DimensionalSignature::MASS),
                    ("v", DimensionalSignature::VELOCITY),
                ]),
            ),
            StrictInferenceResult::Inferred(DimensionalSignature::ENERGY)
        );
    }

    #[test]
    fn dimensional_variable_exponent_is_rejected() {
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("n".into())),
        );
        let result = infer_dimensions_strict(
            &expr,
            &units(&[
                ("x", DimensionalSignature::LENGTH),
                ("n", DimensionalSignature::DIMENSIONLESS),
            ]),
        );
        assert!(result.failure().unwrap().is_inconsistent());
    }

    #[test]
    fn permissive_and_strict_paths_have_intentionally_different_unknown_semantics() {
        let expr = Expr::Var("unknown".into());
        assert_eq!(
            crate::infer_dimensions(&expr, &UnitMap::new()).or_dimensionless(),
            DimensionalSignature::DIMENSIONLESS
        );
        assert!(matches!(
            infer_dimensions_strict(&expr, &UnitMap::new()),
            StrictInferenceResult::Rejected(_)
        ));
    }
}
