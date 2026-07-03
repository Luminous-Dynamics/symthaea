// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::{EmlEvalMode, EmlExpr, eval_complex, eval_real_mode};
use crate::hdc::arithmetic_engine::{SymbolicOp, TermType};
use crate::hdc::complex::Complex;
use crate::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmlRealDomainAssumption {
    AnyFinite,
    NonZero,
    Positive,
    GreaterThanOne,
}

impl EmlRealDomainAssumption {
    pub fn is_unconstrained(self) -> bool {
        matches!(self, Self::AnyFinite)
    }

    pub fn short_tag(self) -> &'static str {
        match self {
            Self::AnyFinite => "any",
            Self::NonZero => "nonzero",
            Self::Positive => "positive",
            Self::GreaterThanOne => "gt1",
        }
    }

    pub fn display_label(self) -> &'static str {
        match self {
            Self::AnyFinite => "any finite reals",
            Self::NonZero => "nonzero reals",
            Self::Positive => "positive reals",
            Self::GreaterThanOne => "reals > 1",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmlVerificationReport {
    pub passed: bool,
    pub mode: EmlEvalMode,
    pub real_domain_assumption: Option<EmlRealDomainAssumption>,
    pub samples_checked: usize,
    pub max_abs_error: f64,
    pub first_failure: Option<String>,
}

pub fn verify_expr_compilation(
    expr: &Expr,
    compiled: &EmlExpr,
    mode: EmlEvalMode,
) -> EmlVerificationReport {
    let vars = expr_variables(expr);
    match mode {
        EmlEvalMode::RealIeee | EmlEvalMode::RealConstructive => {
            verify_expr_real(expr, compiled, &vars, mode)
        }
        EmlEvalMode::ComplexPrincipal => verify_expr_complex(expr, compiled, &vars),
    }
}

pub fn verify_term_compilation(
    term: &TermType,
    compiled: &EmlExpr,
    mode: EmlEvalMode,
) -> EmlVerificationReport {
    let vars = term_variables(term);
    match mode {
        EmlEvalMode::RealIeee | EmlEvalMode::RealConstructive => {
            verify_term_real(term, compiled, &vars, mode)
        }
        EmlEvalMode::ComplexPrincipal => verify_term_complex(term, compiled, &vars),
    }
}

fn verify_expr_real(
    expr: &Expr,
    compiled: &EmlExpr,
    vars: &[String],
    mode: EmlEvalMode,
) -> EmlVerificationReport {
    let mut last_failure = None;
    for &assumption in real_domain_search_order(mode) {
        let report = verify_expr_real_under_assumption(expr, compiled, vars, mode, assumption);
        if report.passed {
            return report;
        }
        last_failure = Some(report);
    }
    last_failure.unwrap_or_else(|| failure(mode, None, 0, f64::INFINITY, "no samples".into()))
}

fn verify_expr_real_under_assumption(
    expr: &Expr,
    compiled: &EmlExpr,
    vars: &[String],
    mode: EmlEvalMode,
    assumption: EmlRealDomainAssumption,
) -> EmlVerificationReport {
    let samples = real_samples_for_expr(vars, assumption);
    let mut max_abs_error: f64 = 0.0;
    let mut checked = 0;
    for bindings in &samples {
        let src_bindings: Vec<(&str, f64)> = bindings.iter().map(|(k, v)| (*k, *v)).collect();
        let eml_bindings: HashMap<&str, f64> = bindings.iter().copied().collect();
        let expected = expr.eval(&src_bindings);
        let got = match eval_real_mode(compiled, &eml_bindings, mode) {
            Ok(v) => v,
            Err(err) => {
                return failure(
                    mode,
                    Some(assumption),
                    checked,
                    f64::INFINITY,
                    format!("{err:?}"),
                );
            }
        };
        let err = (expected - got).abs();
        max_abs_error = max_abs_error.max(err);
        checked += 1;
        if !expected.is_finite() || !got.is_finite() || err > 1e-9 {
            return failure(
                mode,
                Some(assumption),
                checked,
                max_abs_error,
                format!("expected={expected} got={got}"),
            );
        }
    }
    success(mode, Some(assumption), checked, max_abs_error)
}

fn verify_expr_complex(expr: &Expr, compiled: &EmlExpr, vars: &[String]) -> EmlVerificationReport {
    let Some(term) = expr_to_term(expr) else {
        return failure(
            EmlEvalMode::ComplexPrincipal,
            None,
            0,
            f64::INFINITY,
            "expr not representable as TermType".into(),
        );
    };
    verify_term_complex(&term, compiled, vars)
}

fn verify_term_real(
    term: &TermType,
    compiled: &EmlExpr,
    vars: &[String],
    mode: EmlEvalMode,
) -> EmlVerificationReport {
    let mut last_failure = None;
    for &assumption in real_domain_search_order(mode) {
        let report = verify_term_real_under_assumption(term, compiled, vars, mode, assumption);
        if report.passed {
            return report;
        }
        last_failure = Some(report);
    }
    last_failure.unwrap_or_else(|| failure(mode, None, 0, f64::INFINITY, "no samples".into()))
}

fn verify_term_real_under_assumption(
    term: &TermType,
    compiled: &EmlExpr,
    vars: &[String],
    mode: EmlEvalMode,
    assumption: EmlRealDomainAssumption,
) -> EmlVerificationReport {
    let samples = real_samples_for_term(vars, assumption);
    let mut max_abs_error: f64 = 0.0;
    let mut checked = 0;
    for bindings in &samples {
        let src_bindings: Vec<(&str, f64)> = bindings.iter().map(|(k, v)| (*k, *v)).collect();
        let eml_bindings: HashMap<&str, f64> = bindings.iter().copied().collect();
        let expected = eval_term_real(term, &src_bindings);
        let got = match eval_real_mode(compiled, &eml_bindings, mode) {
            Ok(v) => v,
            Err(err) => {
                return failure(
                    mode,
                    Some(assumption),
                    checked,
                    f64::INFINITY,
                    format!("{err:?}"),
                );
            }
        };
        let err = (expected - got).abs();
        max_abs_error = max_abs_error.max(err);
        checked += 1;
        if !expected.is_finite() || !got.is_finite() || err > 1e-9 {
            return failure(
                mode,
                Some(assumption),
                checked,
                max_abs_error,
                format!("expected={expected} got={got}"),
            );
        }
    }
    success(mode, Some(assumption), checked, max_abs_error)
}

fn verify_term_complex(
    term: &TermType,
    compiled: &EmlExpr,
    vars: &[String],
) -> EmlVerificationReport {
    let samples = complex_samples(vars);
    let mut max_abs_error: f64 = 0.0;
    let mut checked = 0;
    for bindings in &samples {
        let eml_bindings: HashMap<&str, Complex> = bindings.iter().copied().collect();
        let expected = match eval_term_complex(term, bindings) {
            Some(v) => v,
            None => {
                return failure(
                    EmlEvalMode::ComplexPrincipal,
                    None,
                    checked,
                    f64::INFINITY,
                    "unsupported complex term".into(),
                );
            }
        };
        let got = match eval_complex(compiled, &eml_bindings) {
            Ok(v) => v,
            Err(err) => {
                return failure(
                    EmlEvalMode::ComplexPrincipal,
                    None,
                    checked,
                    f64::INFINITY,
                    format!("{err:?}"),
                );
            }
        };
        let err = ((expected.re - got.re).powi(2) + (expected.im - got.im).powi(2)).sqrt();
        max_abs_error = max_abs_error.max(err);
        checked += 1;
        if err > 1e-9 {
            return failure(
                EmlEvalMode::ComplexPrincipal,
                None,
                checked,
                max_abs_error,
                format!("expected={expected:?} got={got:?}"),
            );
        }
    }
    success(EmlEvalMode::ComplexPrincipal, None, checked, max_abs_error)
}

fn success(
    mode: EmlEvalMode,
    real_domain_assumption: Option<EmlRealDomainAssumption>,
    checked: usize,
    max_abs_error: f64,
) -> EmlVerificationReport {
    EmlVerificationReport {
        passed: true,
        mode,
        real_domain_assumption,
        samples_checked: checked,
        max_abs_error,
        first_failure: None,
    }
}

fn failure(
    mode: EmlEvalMode,
    real_domain_assumption: Option<EmlRealDomainAssumption>,
    checked: usize,
    max_abs_error: f64,
    reason: String,
) -> EmlVerificationReport {
    EmlVerificationReport {
        passed: false,
        mode,
        real_domain_assumption,
        samples_checked: checked,
        max_abs_error,
        first_failure: Some(reason),
    }
}

fn real_domain_search_order(mode: EmlEvalMode) -> &'static [EmlRealDomainAssumption] {
    match mode {
        EmlEvalMode::RealIeee => &[
            EmlRealDomainAssumption::AnyFinite,
            EmlRealDomainAssumption::NonZero,
            EmlRealDomainAssumption::Positive,
            EmlRealDomainAssumption::GreaterThanOne,
        ],
        EmlEvalMode::RealConstructive => &[
            EmlRealDomainAssumption::Positive,
            EmlRealDomainAssumption::GreaterThanOne,
        ],
        EmlEvalMode::ComplexPrincipal => &[],
    }
}

fn real_sample_values_for_assumption(assumption: EmlRealDomainAssumption) -> &'static [f64] {
    match assumption {
        EmlRealDomainAssumption::AnyFinite => &[-2.0, -0.5, 0.0, 0.5, 2.0],
        EmlRealDomainAssumption::NonZero => &[-3.0, -1.5, -0.5, 0.5, 1.5, 3.0],
        EmlRealDomainAssumption::Positive => &[0.5, 1.25, 2.5, 4.0],
        EmlRealDomainAssumption::GreaterThanOne => &[1.25, 2.0, 3.5, 5.0],
    }
}

fn real_samples(vars: &[String], assumption: EmlRealDomainAssumption) -> Vec<Vec<(&str, f64)>> {
    if vars.is_empty() {
        return vec![Vec::new()];
    }
    let values = real_sample_values_for_assumption(assumption);
    let mut out = vec![Vec::new()];
    for var in vars {
        let mut next = Vec::new();
        for prefix in &out {
            for value in values {
                let mut with = prefix.clone();
                with.push((var.as_str(), *value));
                next.push(with);
            }
        }
        out = next;
    }
    out
}

fn real_samples_for_expr(
    vars: &[String],
    assumption: EmlRealDomainAssumption,
) -> Vec<Vec<(&str, f64)>> {
    real_samples(vars, assumption)
}

fn real_samples_for_term(
    vars: &[String],
    assumption: EmlRealDomainAssumption,
) -> Vec<Vec<(&str, f64)>> {
    real_samples(vars, assumption)
}

fn complex_samples(vars: &[String]) -> Vec<Vec<(&str, Complex)>> {
    if vars.is_empty() {
        return vec![Vec::new()];
    }
    let values = [Complex::new(0.5, 0.25), Complex::new(2.0, -0.5)];
    let mut out = vec![Vec::new()];
    for var in vars {
        let mut next = Vec::new();
        for prefix in &out {
            for value in values {
                let mut with = prefix.clone();
                with.push((var.as_str(), value));
                next.push(with);
            }
        }
        out = next;
    }
    out
}

fn expr_variables(expr: &Expr) -> Vec<String> {
    let mut vars = HashSet::new();
    collect_expr_variables(expr, &mut vars);
    let mut out: Vec<_> = vars.into_iter().collect();
    out.sort();
    out
}

fn collect_expr_variables(expr: &Expr, vars: &mut HashSet<String>) {
    match expr {
        Expr::Var(name) => {
            vars.insert(name.clone());
        }
        Expr::Const(_) => {}
        Expr::BinOp(_, left, right) => {
            collect_expr_variables(left, vars);
            collect_expr_variables(right, vars);
        }
        Expr::Func(_, arg) => collect_expr_variables(arg, vars),
        Expr::Sum(body, var) => {
            vars.insert(var.clone());
            collect_expr_variables(body, vars);
        }
    }
}

fn term_variables(term: &TermType) -> Vec<String> {
    let mut vars = HashSet::new();
    collect_term_variables(term, &mut vars);
    let mut out: Vec<_> = vars.into_iter().collect();
    out.sort();
    out
}

fn collect_term_variables(term: &TermType, vars: &mut HashSet<String>) {
    match term {
        TermType::Variable(name) => {
            vars.insert(name.clone());
        }
        TermType::Constant(_) => {}
        TermType::BinaryOp { left, right, .. } => {
            collect_term_variables(left, vars);
            collect_term_variables(right, vars);
        }
        TermType::UnaryOp { operand, .. } => collect_term_variables(operand, vars),
        TermType::Function { arg, .. } => collect_term_variables(arg, vars),
    }
}

fn expr_to_term(expr: &Expr) -> Option<TermType> {
    match expr {
        Expr::Var(name) => Some(TermType::Variable(name.clone())),
        Expr::Const(c) if (*c - 1.0).abs() < 1e-12 => Some(TermType::Constant(1)),
        Expr::Const(_) => None,
        Expr::BinOp(BinOp::Sub, left, right) => Some(TermType::BinaryOp {
            op: SymbolicOp::Sub,
            left: Box::new(expr_to_term(left)?),
            right: Box::new(expr_to_term(right)?),
        }),
        Expr::BinOp(BinOp::Div, left, right) => Some(TermType::BinaryOp {
            op: SymbolicOp::Div,
            left: Box::new(expr_to_term(left)?),
            right: Box::new(expr_to_term(right)?),
        }),
        Expr::Func(UnaryFn::Exp, arg) => Some(TermType::Function {
            name: "exp".to_string(),
            arg: Box::new(expr_to_term(arg)?),
        }),
        Expr::Func(UnaryFn::Log, arg) => Some(TermType::Function {
            name: "ln".to_string(),
            arg: Box::new(expr_to_term(arg)?),
        }),
        Expr::BinOp(_, _, _) | Expr::Sum(_, _) | Expr::Func(_, _) => None,
    }
}

fn eval_term_real(term: &TermType, vars: &[(&str, f64)]) -> f64 {
    match term {
        TermType::Constant(c) => *c as f64,
        TermType::Variable(name) => vars
            .iter()
            .find(|(n, _)| *n == name.as_str())
            .map(|(_, v)| *v)
            .unwrap_or(f64::NAN),
        TermType::Function { name, arg } if name == "exp" => eval_term_real(arg, vars).exp(),
        TermType::Function { name, arg } if name == "ln" || name == "log" => {
            eval_term_real(arg, vars).ln()
        }
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Sub => {
            eval_term_real(left, vars) - eval_term_real(right, vars)
        }
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Div => {
            eval_term_real(left, vars) / eval_term_real(right, vars)
        }
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Pow => {
            eval_term_real(left, vars).powf(eval_term_real(right, vars))
        }
        _ => f64::NAN,
    }
}

fn eval_term_complex(term: &TermType, vars: &[(&str, Complex)]) -> Option<Complex> {
    match term {
        TermType::Constant(c) => Some(Complex::new(*c as f64, 0.0)),
        TermType::Variable(name) => vars
            .iter()
            .find(|(n, _)| *n == name.as_str())
            .map(|(_, v)| *v),
        TermType::Function { name, arg } if name == "exp" => {
            Some(eval_term_complex(arg, vars)?.exp())
        }
        TermType::Function { name, arg } if name == "ln" || name == "log" => {
            Some(eval_term_complex(arg, vars)?.ln())
        }
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Sub => {
            Some(eval_term_complex(left, vars)? - eval_term_complex(right, vars)?)
        }
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Div => {
            let right = eval_term_complex(right, vars)?;
            if right == Complex::ZERO {
                return None;
            }
            Some(eval_term_complex(left, vars)? / right)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::eml::{compile_expr, compile_expr_constructive, compile_term};

    #[test]
    fn test_verify_exp_expr_real() {
        let expr = Expr::Func(UnaryFn::Exp, Box::new(Expr::Var("x".into())));
        let compiled = compile_expr(&expr).unwrap();
        let report = verify_expr_compilation(&expr, &compiled, EmlEvalMode::RealIeee);
        assert!(report.passed, "{report:?}");
        assert_eq!(
            report.real_domain_assumption,
            Some(EmlRealDomainAssumption::AnyFinite)
        );
    }

    #[test]
    fn test_verify_ln_expr_complex() {
        let expr = Expr::Func(UnaryFn::Log, Box::new(Expr::Var("x".into())));
        let compiled = compile_expr(&expr).unwrap();
        let report = verify_expr_compilation(&expr, &compiled, EmlEvalMode::ComplexPrincipal);
        assert!(report.passed, "{report:?}");
        assert_eq!(report.real_domain_assumption, None);
    }

    #[test]
    fn test_verify_term_ln_real() {
        let term = TermType::Function {
            name: "ln".into(),
            arg: Box::new(TermType::Variable("x".into())),
        };
        let compiled = compile_term(&term).unwrap();
        let report = verify_term_compilation(&term, &compiled, EmlEvalMode::RealIeee);
        assert!(report.passed, "{report:?}");
        assert_eq!(
            report.real_domain_assumption,
            Some(EmlRealDomainAssumption::Positive)
        );
    }

    #[test]
    fn test_verify_sub_expr_real() {
        let expr = Expr::BinOp(
            BinOp::Sub,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let compiled = compile_expr(&expr).unwrap();
        let report = verify_expr_compilation(&expr, &compiled, EmlEvalMode::RealIeee);
        assert!(report.passed, "{report:?}");
        assert_eq!(
            report.real_domain_assumption,
            Some(EmlRealDomainAssumption::Positive)
        );
    }

    #[test]
    fn test_verify_div_term_complex() {
        let term = TermType::BinaryOp {
            op: SymbolicOp::Div,
            left: Box::new(TermType::Variable("x".into())),
            right: Box::new(TermType::Variable("y".into())),
        };
        let compiled = compile_term(&term).unwrap();
        let report = verify_term_compilation(&term, &compiled, EmlEvalMode::ComplexPrincipal);
        assert!(report.passed, "{report:?}");
    }

    #[test]
    fn test_verify_div_expr_real_requires_gt1_domain() {
        let expr = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let compiled = compile_expr(&expr).unwrap();
        let report = verify_expr_compilation(&expr, &compiled, EmlEvalMode::RealIeee);
        assert!(report.passed, "{report:?}");
        assert_eq!(
            report.real_domain_assumption,
            Some(EmlRealDomainAssumption::GreaterThanOne)
        );
    }

    #[test]
    fn test_verify_div_term_real_requires_gt1_domain() {
        let term = TermType::BinaryOp {
            op: SymbolicOp::Div,
            left: Box::new(TermType::Variable("x".into())),
            right: Box::new(TermType::Variable("y".into())),
        };
        let compiled = compile_term(&term).unwrap();
        let report = verify_term_compilation(&term, &compiled, EmlEvalMode::RealIeee);
        assert!(report.passed, "{report:?}");
        assert_eq!(
            report.real_domain_assumption,
            Some(EmlRealDomainAssumption::GreaterThanOne)
        );
    }

    #[test]
    fn test_verify_constructive_add_expr_real() {
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let compiled = compile_expr_constructive(&expr).unwrap();
        let report = verify_expr_compilation(&expr, &compiled, EmlEvalMode::RealConstructive);
        assert!(report.passed, "{report:?}");
        assert_eq!(
            report.real_domain_assumption,
            Some(EmlRealDomainAssumption::Positive)
        );
    }

    #[test]
    fn test_verify_constructive_mul_expr_real() {
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let compiled = compile_expr_constructive(&expr).unwrap();
        let report = verify_expr_compilation(&expr, &compiled, EmlEvalMode::RealConstructive);
        assert!(report.passed, "{report:?}");
        assert_eq!(
            report.real_domain_assumption,
            Some(EmlRealDomainAssumption::GreaterThanOne)
        );
    }

    #[test]
    fn test_real_sample_profile_for_any_finite_uses_signed_support() {
        let vars = vec!["x".to_string()];
        let samples = real_samples_for_expr(&vars, EmlRealDomainAssumption::AnyFinite);
        assert!(samples.iter().flatten().any(|(_, v)| *v < 0.0));
        assert!(samples.iter().flatten().any(|(_, v)| *v == 0.0));
        assert!(samples.iter().flatten().any(|(_, v)| *v > 0.0));
    }

    #[test]
    fn test_real_sample_profile_for_positive_stays_positive() {
        let vars = vec!["x".to_string()];
        let samples = real_samples_for_expr(&vars, EmlRealDomainAssumption::Positive);
        assert!(samples.iter().flatten().all(|(_, v)| *v > 0.0));
        assert!(samples.iter().flatten().any(|(_, v)| *v < 1.0));
    }

    #[test]
    fn test_real_sample_profile_for_gt1_stays_above_one() {
        let vars = vec!["x".to_string(), "y".to_string()];
        let samples = real_samples_for_expr(&vars, EmlRealDomainAssumption::GreaterThanOne);
        assert!(samples.iter().flatten().all(|(_, v)| *v > 1.0));
    }
}
