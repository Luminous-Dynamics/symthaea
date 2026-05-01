// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use super::EmlExpr;
use crate::hdc::arithmetic_engine::{SymbolicOp, TermType};
use crate::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmlCompileError {
    UnsupportedExpr,
    UnsupportedFunction(String),
    NonElementaryConstant(String),
    NonCompilableStructure,
}

pub fn compile_expr(expr: &Expr) -> Result<EmlExpr, EmlCompileError> {
    match expr {
        Expr::Var(name) => Ok(EmlExpr::terminal_var(name.clone())),
        Expr::Const(c) => {
            if (*c - 1.0).abs() < 1e-12 {
                Ok(EmlExpr::terminal_one())
            } else {
                Err(EmlCompileError::NonElementaryConstant(format!("{c}")))
            }
        }
        Expr::Func(UnaryFn::Exp, arg) => {
            let inner = compile_expr(arg)?;
            Ok(compile_exp(inner))
        }
        Expr::Func(UnaryFn::Log, arg) => {
            let inner = compile_expr(arg)?;
            Ok(compile_ln(inner))
        }
        Expr::Func(other, _) => Err(EmlCompileError::UnsupportedFunction(format!("{other:?}"))),
        Expr::BinOp(BinOp::Sub, left, right) => {
            let left = compile_expr(left)?;
            let right = compile_expr(right)?;
            Ok(compile_sub(left, right))
        }
        Expr::BinOp(BinOp::Add, left, right) => {
            let left = compile_expr(left)?;
            let right = compile_expr(right)?;
            Ok(compile_add(left, right))
        }
        Expr::BinOp(BinOp::Mul, left, right) => {
            let left = compile_expr(left)?;
            let right = compile_expr(right)?;
            Ok(compile_mul(left, right))
        }
        Expr::BinOp(BinOp::Div, left, right) => {
            let left = compile_expr(left)?;
            let right = compile_expr(right)?;
            Ok(compile_div(left, right))
        }
        Expr::BinOp(BinOp::Pow, left, right) => compile_pow_expr(left, right),
        Expr::Sum(_, _) => Err(EmlCompileError::UnsupportedExpr),
    }
}

pub fn compile_expr_constructive(expr: &Expr) -> Result<EmlExpr, EmlCompileError> {
    match expr {
        Expr::BinOp(BinOp::Add, left, right) => {
            let left = compile_expr_constructive(left)?;
            let right = compile_expr_constructive(right)?;
            Ok(compile_add(left, right))
        }
        Expr::BinOp(BinOp::Mul, left, right) => {
            let left = compile_expr_constructive(left)?;
            let right = compile_expr_constructive(right)?;
            Ok(compile_mul(left, right))
        }
        Expr::BinOp(BinOp::Pow, left, right) => compile_pow_expr_constructive(left, right),
        Expr::BinOp(BinOp::Sub, left, right) => {
            let left = compile_expr_constructive(left)?;
            let right = compile_expr_constructive(right)?;
            Ok(compile_sub(left, right))
        }
        Expr::BinOp(BinOp::Div, left, right) => {
            let left = compile_expr_constructive(left)?;
            let right = compile_expr_constructive(right)?;
            Ok(compile_div(left, right))
        }
        Expr::Func(UnaryFn::Exp, arg) => Ok(compile_exp(compile_expr_constructive(arg)?)),
        Expr::Func(UnaryFn::Log, arg) => Ok(compile_ln(compile_expr_constructive(arg)?)),
        Expr::Var(_) | Expr::Const(_) => compile_expr(expr),
        Expr::Func(other, _) => Err(EmlCompileError::UnsupportedFunction(format!("{other:?}"))),
        Expr::Sum(_, _) => Err(EmlCompileError::UnsupportedExpr),
    }
}

pub fn compile_term(term: &TermType) -> Result<EmlExpr, EmlCompileError> {
    match term {
        TermType::Constant(1) => Ok(EmlExpr::terminal_one()),
        TermType::Constant(c) => Err(EmlCompileError::NonElementaryConstant(c.to_string())),
        TermType::Variable(name) => Ok(EmlExpr::terminal_var(name.clone())),
        TermType::Function { name, arg } if name == "exp" => {
            let inner = compile_term(arg)?;
            Ok(compile_exp(inner))
        }
        TermType::Function { name, arg } if name == "ln" || name == "log" => {
            let inner = compile_term(arg)?;
            Ok(compile_ln(inner))
        }
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Sub => {
            let left = compile_term(left)?;
            let right = compile_term(right)?;
            Ok(compile_sub(left, right))
        }
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Add => {
            let left = compile_term(left)?;
            let right = compile_term(right)?;
            Ok(compile_add(left, right))
        }
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Mul => {
            let left = compile_term(left)?;
            let right = compile_term(right)?;
            Ok(compile_mul(left, right))
        }
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Div => {
            let left = compile_term(left)?;
            let right = compile_term(right)?;
            Ok(compile_div(left, right))
        }
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Pow => {
            compile_pow_term(left, right)
        }
        TermType::Function { name, .. } => Err(EmlCompileError::UnsupportedFunction(name.clone())),
        TermType::BinaryOp { .. } | TermType::UnaryOp { .. } => {
            Err(EmlCompileError::NonCompilableStructure)
        }
    }
}

pub fn compile_term_constructive(term: &TermType) -> Result<EmlExpr, EmlCompileError> {
    match term {
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Add => Ok(compile_add(
            compile_term_constructive(left)?,
            compile_term_constructive(right)?,
        )),
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Mul => Ok(compile_mul(
            compile_term_constructive(left)?,
            compile_term_constructive(right)?,
        )),
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Pow => {
            compile_pow_term_constructive(left, right)
        }
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Sub => Ok(compile_sub(
            compile_term_constructive(left)?,
            compile_term_constructive(right)?,
        )),
        TermType::BinaryOp { op, left, right } if *op == SymbolicOp::Div => Ok(compile_div(
            compile_term_constructive(left)?,
            compile_term_constructive(right)?,
        )),
        TermType::Function { name, arg } if name == "exp" => {
            Ok(compile_exp(compile_term_constructive(arg)?))
        }
        TermType::Function { name, arg } if name == "ln" || name == "log" => {
            Ok(compile_ln(compile_term_constructive(arg)?))
        }
        TermType::Constant(_) | TermType::Variable(_) => compile_term(term),
        TermType::Function { name, .. } => Err(EmlCompileError::UnsupportedFunction(name.clone())),
        TermType::BinaryOp { .. } | TermType::UnaryOp { .. } => {
            Err(EmlCompileError::NonCompilableStructure)
        }
    }
}

fn compile_ln(arg: EmlExpr) -> EmlExpr {
    EmlExpr::eml(
        EmlExpr::terminal_one(),
        EmlExpr::eml(
            EmlExpr::eml(EmlExpr::terminal_one(), arg),
            EmlExpr::terminal_one(),
        ),
    )
}

fn compile_exp(arg: EmlExpr) -> EmlExpr {
    EmlExpr::eml(arg, EmlExpr::terminal_one())
}

fn compile_neg(arg: EmlExpr) -> EmlExpr {
    compile_ln(compile_div(EmlExpr::terminal_one(), compile_exp(arg)))
}

fn compile_add(left: EmlExpr, right: EmlExpr) -> EmlExpr {
    compile_sub(left, compile_neg(right))
}

fn compile_sub(left: EmlExpr, right: EmlExpr) -> EmlExpr {
    EmlExpr::eml(compile_ln(left), compile_exp(right))
}

fn compile_mul(left: EmlExpr, right: EmlExpr) -> EmlExpr {
    compile_exp(compile_add(compile_ln(left), compile_ln(right)))
}

fn compile_div(left: EmlExpr, right: EmlExpr) -> EmlExpr {
    compile_exp(compile_sub(compile_ln(left), compile_ln(right)))
}

fn compile_pow_expr(base: &Expr, exponent: &Expr) -> Result<EmlExpr, EmlCompileError> {
    let base = compile_expr(base)?;
    match exponent {
        Expr::Const(c) => match rounded_integer(*c) {
            Some(2) => Ok(compile_mul(base.clone(), base)),
            Some(-1) => Ok(compile_div(EmlExpr::terminal_one(), base)),
            Some(0) => Ok(EmlExpr::terminal_one()),
            Some(1) => Ok(base),
            _ => Err(EmlCompileError::UnsupportedExpr),
        },
        _ => Err(EmlCompileError::UnsupportedExpr),
    }
}

fn compile_pow_expr_constructive(base: &Expr, exponent: &Expr) -> Result<EmlExpr, EmlCompileError> {
    let base = compile_expr_constructive(base)?;
    match exponent {
        Expr::Const(c) => match rounded_integer(*c) {
            Some(2) => Ok(compile_mul(base.clone(), base)),
            Some(-1) => Ok(compile_div(EmlExpr::terminal_one(), base)),
            Some(0) => Ok(EmlExpr::terminal_one()),
            Some(1) => Ok(base),
            _ => Err(EmlCompileError::UnsupportedExpr),
        },
        _ => Err(EmlCompileError::UnsupportedExpr),
    }
}

fn compile_pow_term(base: &TermType, exponent: &TermType) -> Result<EmlExpr, EmlCompileError> {
    let base = compile_term(base)?;
    match exponent {
        TermType::Constant(2) => Ok(compile_mul(base.clone(), base)),
        TermType::Constant(-1) => Ok(compile_div(EmlExpr::terminal_one(), base)),
        TermType::Constant(0) => Ok(EmlExpr::terminal_one()),
        TermType::Constant(1) => Ok(base),
        TermType::Constant(_) => Err(EmlCompileError::UnsupportedExpr),
        _ => Err(EmlCompileError::NonCompilableStructure),
    }
}

fn compile_pow_term_constructive(
    base: &TermType,
    exponent: &TermType,
) -> Result<EmlExpr, EmlCompileError> {
    let base = compile_term_constructive(base)?;
    match exponent {
        TermType::Constant(2) => Ok(compile_mul(base.clone(), base)),
        TermType::Constant(-1) => Ok(compile_div(EmlExpr::terminal_one(), base)),
        TermType::Constant(0) => Ok(EmlExpr::terminal_one()),
        TermType::Constant(1) => Ok(base),
        TermType::Constant(_) => Err(EmlCompileError::UnsupportedExpr),
        _ => Err(EmlCompileError::NonCompilableStructure),
    }
}

fn rounded_integer(value: f64) -> Option<i64> {
    let rounded = value.round();
    if (value - rounded).abs() < 1e-12 {
        Some(rounded as i64)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::conjecture_engine::Expr;

    #[test]
    fn test_compile_exp_expr() {
        let expr = Expr::Func(UnaryFn::Exp, Box::new(Expr::Var("x".into())));
        let compiled = compile_expr(&expr).unwrap();
        assert_eq!(compiled.to_string(), "eml(x,1)");
    }

    #[test]
    fn test_compile_ln_expr() {
        let expr = Expr::Func(UnaryFn::Log, Box::new(Expr::Var("x".into())));
        let compiled = compile_expr(&expr).unwrap();
        assert_eq!(compiled.to_string(), "eml(1,eml(eml(1,x),1))");
    }

    #[test]
    fn test_compile_sub_expr() {
        let expr = Expr::BinOp(
            BinOp::Sub,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let compiled = compile_expr(&expr).unwrap();
        assert!(compiled.to_string().starts_with("eml("));
    }

    #[test]
    fn test_compile_div_expr() {
        let expr = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let compiled = compile_expr(&expr).unwrap();
        assert!(compiled.to_string().starts_with("eml("));
    }

    #[test]
    fn test_compile_add_expr() {
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let compiled = compile_expr(&expr).unwrap();
        assert!(compiled.to_string().starts_with("eml("));
    }

    #[test]
    fn test_compile_mul_expr() {
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let compiled = compile_expr(&expr).unwrap();
        assert!(compiled.to_string().starts_with("eml("));
    }

    #[test]
    fn test_compile_pow_square_expr() {
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Const(2.0)),
        );
        let compiled = compile_expr(&expr).unwrap();
        assert!(compiled.to_string().starts_with("eml("));
    }

    #[test]
    fn test_compile_constructive_add_expr() {
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let compiled = compile_expr_constructive(&expr).unwrap();
        assert!(compiled.to_string().starts_with("eml("));
    }

    #[test]
    fn test_compile_constructive_mul_expr() {
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let compiled = compile_expr_constructive(&expr).unwrap();
        assert!(compiled.to_string().starts_with("eml("));
    }
}
