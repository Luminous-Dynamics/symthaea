// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # EML Backend
//!
//! Minimal Exp-Minus-Log IR and compiler for a narrow elementary subset.

mod compile;
pub mod derive;
mod eval;
mod verify;

pub use compile::{
    EmlCompileError, compile_expr, compile_expr_constructive, compile_term,
    compile_term_constructive,
};
pub use eval::{EmlEvalError, EmlEvalMode, eval_complex, eval_real, eval_real_mode};
pub use verify::{
    EmlRealDomainAssumption, EmlVerificationReport, verify_expr_compilation,
    verify_term_compilation,
};

use serde::{Deserialize, Serialize};

/// Terminal symbols allowed in a pure EML expression.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmlTerminal {
    /// Distinguished terminal required by the paper basis.
    One,
    /// External variable terminal, e.g. `x`.
    Var(String),
}

/// Pure EML tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmlExpr {
    Terminal(EmlTerminal),
    Eml(Box<EmlExpr>, Box<EmlExpr>),
}

/// Simple structural metrics for compiled EML trees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmlMetrics {
    pub depth: usize,
    pub nodes: usize,
}

impl EmlExpr {
    pub fn terminal_one() -> Self {
        Self::Terminal(EmlTerminal::One)
    }

    pub fn terminal_var(name: impl Into<String>) -> Self {
        Self::Terminal(EmlTerminal::Var(name.into()))
    }

    pub fn eml(left: EmlExpr, right: EmlExpr) -> Self {
        Self::Eml(Box::new(left), Box::new(right))
    }

    pub fn metrics(&self) -> EmlMetrics {
        match self {
            Self::Terminal(_) => EmlMetrics { depth: 0, nodes: 1 },
            Self::Eml(left, right) => {
                let l = left.metrics();
                let r = right.metrics();
                EmlMetrics {
                    depth: 1 + l.depth.max(r.depth),
                    nodes: 1 + l.nodes + r.nodes,
                }
            }
        }
    }
}

impl std::fmt::Display for EmlExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Terminal(EmlTerminal::One) => write!(f, "1"),
            Self::Terminal(EmlTerminal::Var(name)) => write!(f, "{name}"),
            Self::Eml(left, right) => write!(f, "eml({left},{right})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::complex::Complex;
    use std::collections::HashMap;

    #[test]
    fn test_metrics_and_display() {
        let expr = EmlExpr::eml(EmlExpr::terminal_var("x"), EmlExpr::terminal_one());
        assert_eq!(expr.to_string(), "eml(x,1)");
        assert_eq!(expr.metrics(), EmlMetrics { depth: 1, nodes: 3 });
    }

    #[test]
    fn test_eval_real_exp() {
        let expr = EmlExpr::eml(EmlExpr::terminal_var("x"), EmlExpr::terminal_one());
        let vars = HashMap::from([("x", 1.25)]);
        let y = eval_real(&expr, &vars).unwrap();
        assert!((y - 1.25_f64.exp()).abs() < 1e-12);
    }

    #[test]
    fn test_eval_complex_ln_formula() {
        let x = EmlExpr::terminal_var("x");
        let ln_expr = EmlExpr::eml(
            EmlExpr::terminal_one(),
            EmlExpr::eml(
                EmlExpr::eml(EmlExpr::terminal_one(), x),
                EmlExpr::terminal_one(),
            ),
        );
        let vars = HashMap::from([("x", Complex::new(2.0, 3.0))]);
        let got = eval_complex(&ln_expr, &vars).unwrap();
        let expected = Complex::new(2.0, 3.0).ln();
        assert!((got.re - expected.re).abs() < 1e-10);
        assert!((got.im - expected.im).abs() < 1e-10);
    }
}
