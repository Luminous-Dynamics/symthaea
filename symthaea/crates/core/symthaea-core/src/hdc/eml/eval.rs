// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::{EmlExpr, EmlTerminal};
use crate::hdc::complex::Complex;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmlEvalMode {
    RealIeee,
    RealConstructive,
    ComplexPrincipal,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EmlEvalError {
    MissingVariable(String),
    DomainError(&'static str),
}

pub fn eval_real(expr: &EmlExpr, vars: &HashMap<&str, f64>) -> Result<f64, EmlEvalError> {
    eval_real_mode(expr, vars, EmlEvalMode::RealIeee)
}

pub fn eval_real_mode(
    expr: &EmlExpr,
    vars: &HashMap<&str, f64>,
    mode: EmlEvalMode,
) -> Result<f64, EmlEvalError> {
    match expr {
        EmlExpr::Terminal(EmlTerminal::One) => Ok(1.0),
        EmlExpr::Terminal(EmlTerminal::Var(name)) => vars
            .get(name.as_str())
            .copied()
            .ok_or_else(|| EmlEvalError::MissingVariable(name.clone())),
        EmlExpr::Eml(left, right) => {
            let l = eval_real_mode(left, vars, mode)?;
            let r = eval_real_mode(right, vars, mode)?;
            match mode {
                EmlEvalMode::RealIeee => {
                    if r <= 0.0 {
                        return Err(EmlEvalError::DomainError(
                            "real EML requires strictly positive right branch",
                        ));
                    }
                    Ok(l.exp() - r.ln())
                }
                EmlEvalMode::RealConstructive => {
                    let log_r = constructive_ln(r)?;
                    Ok(l.exp() - log_r)
                }
                EmlEvalMode::ComplexPrincipal => Err(EmlEvalError::DomainError(
                    "complex mode requires complex evaluator",
                )),
            }
        }
    }
}

fn constructive_ln(x: f64) -> Result<f64, EmlEvalError> {
    if x > 0.0 {
        Ok(x.ln())
    } else if x == 0.0 {
        Ok(f64::NEG_INFINITY)
    } else {
        Err(EmlEvalError::DomainError(
            "constructive real EML still requires nonnegative right branch",
        ))
    }
}

pub fn eval_complex(
    expr: &EmlExpr,
    vars: &HashMap<&str, Complex>,
) -> Result<Complex, EmlEvalError> {
    match expr {
        EmlExpr::Terminal(EmlTerminal::One) => Ok(Complex::ONE),
        EmlExpr::Terminal(EmlTerminal::Var(name)) => vars
            .get(name.as_str())
            .copied()
            .ok_or_else(|| EmlEvalError::MissingVariable(name.clone())),
        EmlExpr::Eml(left, right) => {
            let l = eval_complex(left, vars)?;
            let r = eval_complex(right, vars)?;
            if r == Complex::ZERO {
                return Err(EmlEvalError::DomainError(
                    "complex EML principal-branch log undefined at zero",
                ));
            }
            Ok(l.exp() - r.ln())
        }
    }
}
