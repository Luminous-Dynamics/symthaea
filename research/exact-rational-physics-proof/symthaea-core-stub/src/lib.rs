//! Test-only host shim for the production symbolic-expression module.
//!
//! It provides only the small expression AST referenced by symbolic.rs.
//! It does not claim to compile or validate production HDC behavior.

use std::fmt;

#[derive(Debug, Clone)]
pub enum Expr {
    Var(String),
    Const(f64),
    BinOp(BinOp, Box<Expr>, Box<Expr>),
    Func(UnaryFn, Box<Expr>),
    Sum(Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Pow,
    Div,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryFn {
    Sin,
    Cos,
    Log,
    Sqrt,
    Exp,
    Abs,
    Floor,
}

#[path = "../../../../crates/core/symthaea-core/src/hdc/conjecture_engine/symbolic.rs"]
mod symbolic;

pub use symbolic::SymExpr;

pub mod hdc {
    pub mod conjecture_engine {
        pub use crate::SymExpr;
    }
}
