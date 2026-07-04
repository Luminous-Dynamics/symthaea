// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! SMT-LIB2 Serializer for Rust Expressions
//!
//! Converts loop-free Rust arithmetic expressions into SMT-LIB2 format
//! for formal verification via Z3.

use syn::{BinOp, Expr, Lit, UnOp};

/// Convert a syn::Expr into an SMT-LIB2 string.
/// Returns None if the expression contains unsupported elements (calls, loops, etc.)
pub fn expr_to_smtlib2(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Binary(eb) => {
            let left = expr_to_smtlib2(&eb.left)?;
            let right = expr_to_smtlib2(&eb.right)?;
            let op = match eb.op {
                BinOp::Add(_) => "+",
                BinOp::Sub(_) => "-",
                BinOp::Mul(_) => "*",
                BinOp::Div(_) => "/",
                BinOp::Rem(_) => "rem",
                BinOp::Eq(_) => "=",
                BinOp::Ne(_) => "not =", // Needs special handling for SMT
                BinOp::Lt(_) => "<",
                BinOp::Le(_) => "<=",
                BinOp::Gt(_) => ">",
                BinOp::Ge(_) => ">=",
                BinOp::And(_) => "and",
                BinOp::Or(_) => "or",
                _ => return None,
            };

            if op == "not =" {
                Some(format!("(not (= {} {}))", left, right))
            } else {
                Some(format!("({} {} {})", op, left, right))
            }
        }
        Expr::Unary(eu) => {
            let inner = expr_to_smtlib2(&eu.expr)?;
            match eu.op {
                UnOp::Not(_) => Some(format!("(not {})", inner)),
                UnOp::Neg(_) => Some(format!("(- {})", inner)),
                _ => None,
            }
        }
        Expr::Lit(el) => match &el.lit {
            Lit::Int(li) => Some(li.base10_digits().to_string()),
            Lit::Float(lf) => Some(lf.base10_digits().to_string()),
            Lit::Bool(lb) => Some(if lb.value { "true" } else { "false" }.to_string()),
            _ => None,
        },
        Expr::Path(ep) => {
            // Treat paths as variables
            ep.path.get_ident().map(|ident| ident.to_string())
        }
        Expr::Paren(ep) => expr_to_smtlib2(&ep.expr),
        _ => None,
    }
}

/// Generate SMT-LIB2 declarations for all variables found in an expression.
pub fn get_smt_declarations(expr: &Expr, default_type: &str) -> String {
    let mut vars = std::collections::HashSet::new();
    collect_vars(expr, &mut vars);

    vars.into_iter()
        .map(|v| format!("(declare-const {} {})", v, default_type))
        .collect::<Vec<_>>()
        .join("\n")
}

fn collect_vars(expr: &Expr, vars: &mut std::collections::HashSet<String>) {
    match expr {
        Expr::Binary(eb) => {
            collect_vars(&eb.left, vars);
            collect_vars(&eb.right, vars);
        }
        Expr::Unary(eu) => collect_vars(&eu.expr, vars),
        Expr::Path(ep) => {
            if let Some(ident) = ep.path.get_ident() {
                vars.insert(ident.to_string());
            }
        }
        Expr::Paren(ep) => collect_vars(&ep.expr, vars),
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_str;

    #[test]
    fn serializes_simple_arithmetic() {
        let expr: Expr = parse_str("a + b * 2").unwrap();
        let smt = expr_to_smtlib2(&expr).unwrap();
        assert_eq!(smt, "(+ a (* b 2))");
    }

    #[test]
    fn serializes_comparisons() {
        let expr: Expr = parse_str("x >= 10 && y < 5").unwrap();
        let smt = expr_to_smtlib2(&expr).unwrap();
        assert_eq!(smt, "(and (>= x 10) (< y 5))");
    }

    #[test]
    fn generates_declarations() {
        let expr: Expr = parse_str("a + b + c").unwrap();
        let decls = get_smt_declarations(&expr, "Int");
        assert!(decls.contains("(declare-const a Int)"));
        assert!(decls.contains("(declare-const b Int)"));
        assert!(decls.contains("(declare-const c Int)"));
    }
}
