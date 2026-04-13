// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Discovery → Recognition Bridge
//!
//! Converts ConjectureEngine [`Expr`] trees into physics-bridge [`EquationNode`]
//! ASTs, then runs them through the semantic search catalog to identify which
//! known physics equation (if any) matches an autonomous discovery.
//!
//! This closes the loop:
//! ```text
//! ODE → ConjectureEngine → Expr → expr_to_equation_node → EquationNode
//!                                                       ↓
//!                                          PhysicsSearchEngine::search_equation
//!                                                       ↓
//!                                          SearchResult { name, score, breakdown }
//!                                                       ↓
//!                                          RecognitionReport (with LEM classification)
//! ```
//!
//! When Symthaea autonomously discovers `E = ½v² - 1/r`, this module answers:
//! *"I've seen this before — it's Kepler's orbital energy, similarity 97%."*
//!
//! ## Encoding conventions
//!
//! The physics-bridge catalog represents scalar variables as
//! [`EquationNode::Constant`] nodes (not `Field`), and represents trig/log
//! expressions as single-atom constants like `"sin(θ)"` or `"ln(Q)"`. This
//! module mirrors those conventions so that discovered formulas actually match
//! catalog entries in skeleton mode:
//!
//! - `Expr::Var(name)` → `EquationNode::Constant { name }`
//! - `Expr::Func(Sin, arg)` → `EquationNode::Constant { name: "sin(<arg>)" }`
//! - `Expr::Func(Cos, arg)` → `EquationNode::Constant { name: "cos(<arg>)" }`
//! - `Expr::Func(Log, arg)` → `EquationNode::Constant { name: "ln(<arg>)" }`
//!
//! This is lossy for deep structural matching inside trig arguments, but it
//! matches how the catalog actually encodes these functions. Recognition
//! accuracy is substantially higher with this convention than with `Field`
//! or `Product` encodings.
//!
//! For exact symbolic proofs, use the ConjectureEngine's own Z3 bridge.
//! This module is for **fast catalog recognition**, not formal verification.

use symthaea_core::hdc::conjecture_engine::{BinOp, Expr, UnaryFn};

use crate::query::PhysicsSearchEngine;
use crate::types::{
    DiffOperator, DimensionalSignature, EquationNode, PhysicsDomain, PhysicsEquation,
    SearchResult, SymmetryDescriptor,
};

/// Convert a ConjectureEngine [`Expr`] into a physics-bridge [`EquationNode`].
///
/// Returns `None` only when the expression contains structure that cannot be
/// represented at all (currently: never — all variants map to something, though
/// some mappings are lossy — see module docs).
pub fn expr_to_equation_node(expr: &Expr) -> EquationNode {
    match expr {
        // IMPORTANT: ConjectureEngine variables (n, r, x, v, etc.) are named
        // scalars — they are not tensor fields. The catalog's existing entries
        // (Rydberg, Newton's laws, etc.) encode scalar variables as Constant
        // nodes, not Field nodes. Using Constant here is what makes recognition
        // actually match existing catalog entries in skeleton mode.
        Expr::Var(name) => EquationNode::Constant { name: name.clone() },
        Expr::Const(c) => EquationNode::Scalar(*c),
        Expr::BinOp(BinOp::Add, l, r) => {
            // Flatten nested Add into a single Sum(vec![])
            let mut terms = Vec::new();
            flatten_add(l, &mut terms);
            flatten_add(r, &mut terms);
            if terms.len() == 1 {
                terms.into_iter().next().unwrap()
            } else {
                EquationNode::Sum(terms)
            }
        }
        Expr::BinOp(BinOp::Sub, l, r) => EquationNode::Sum(vec![
            expr_to_equation_node(l),
            EquationNode::Negate(Box::new(expr_to_equation_node(r))),
        ]),
        Expr::BinOp(BinOp::Mul, l, r) => {
            // Flatten nested Mul into a single Product(vec![])
            let mut factors = Vec::new();
            flatten_mul(l, &mut factors);
            flatten_mul(r, &mut factors);
            if factors.len() == 1 {
                factors.into_iter().next().unwrap()
            } else {
                EquationNode::Product(factors)
            }
        }
        Expr::BinOp(BinOp::Div, l, r) => {
            // a / b  →  a * b^(-1)
            EquationNode::Product(vec![
                expr_to_equation_node(l),
                EquationNode::Power {
                    base: Box::new(expr_to_equation_node(r)),
                    exponent: Box::new(EquationNode::Scalar(-1.0)),
                },
            ])
        }
        Expr::BinOp(BinOp::Pow, base, exp) => EquationNode::Power {
            base: Box::new(expr_to_equation_node(base)),
            exponent: Box::new(expr_to_equation_node(exp)),
        },
        Expr::Func(UnaryFn::Sqrt, arg) => {
            EquationNode::Sqrt(Box::new(expr_to_equation_node(arg)))
        }
        Expr::Func(UnaryFn::Exp, arg) => {
            EquationNode::Exponential(Box::new(expr_to_equation_node(arg)))
        }
        // Lossy: trig/log/abs/floor are encoded as single-atom Constant nodes
        // matching the catalog's convention (e.g. "sin(θ)", "ln(Q)"). This loses
        // the internal argument structure for semantic matching, but aligns with
        // how the physics-bridge catalog actually encodes these functions.
        Expr::Func(UnaryFn::Sin, arg) => EquationNode::Constant {
            name: format!("sin({})", arg),
        },
        Expr::Func(UnaryFn::Cos, arg) => EquationNode::Constant {
            name: format!("cos({})", arg),
        },
        Expr::Func(UnaryFn::Log, arg) => EquationNode::Constant {
            name: format!("ln({})", arg),
        },
        Expr::Func(UnaryFn::Abs, arg) => EquationNode::Constant {
            name: format!("abs({})", arg),
        },
        Expr::Func(UnaryFn::Floor, arg) => EquationNode::Constant {
            name: format!("floor({})", arg),
        },
        Expr::Sum(body, _var) => {
            // Σ operator as an integral-like differential operator wrapping the body
            EquationNode::DiffOp {
                operator: DiffOperator::Integral,
                operand: Box::new(expr_to_equation_node(body)),
            }
        }
    }
}

fn flatten_add(expr: &Expr, out: &mut Vec<EquationNode>) {
    if let Expr::BinOp(BinOp::Add, l, r) = expr {
        flatten_add(l, out);
        flatten_add(r, out);
    } else {
        out.push(expr_to_equation_node(expr));
    }
}

fn flatten_mul(expr: &Expr, out: &mut Vec<EquationNode>) {
    if let Expr::BinOp(BinOp::Mul, l, r) = expr {
        flatten_mul(l, out);
        flatten_mul(r, out);
    } else {
        out.push(expr_to_equation_node(expr));
    }
}


/// A full recognition report for a discovered formula.
#[derive(Debug, Clone)]
pub struct RecognitionReport {
    /// The name of the top matching equation from the catalog.
    pub best_match: Option<String>,
    /// Physics domain of the best match.
    pub best_domain: Option<PhysicsDomain>,
    /// Similarity score of the best match (0.0 - 1.0).
    pub best_score: f32,
    /// Whether the discovery is considered novel (score below novelty threshold).
    pub is_novel: bool,
    /// All top-k search results for inspection.
    pub all_results: Vec<SearchResult>,
}

impl RecognitionReport {
    /// A single-line summary for printing next to a discovered formula.
    pub fn headline(&self) -> String {
        match (&self.best_match, self.best_score) {
            (Some(name), score) if score > 0.85 => {
                format!("↳ MATCHES '{}' ({:.0}% similarity)", name, score * 100.0)
            }
            (Some(name), score) if score > 0.60 => {
                format!("↳ resembles '{}' ({:.0}% similarity)", name, score * 100.0)
            }
            (Some(name), score) if score > 0.40 => {
                format!(
                    "↳ weakly analogous to '{}' ({:.0}%) — possibly novel",
                    name, score * 100.0
                )
            }
            _ => "↳ NO KNOWN MATCH — possibly novel discovery".to_string(),
        }
    }
}

/// Recognize a discovered [`Expr`] against the physics catalog.
///
/// Builds a minimal [`PhysicsEquation`] wrapper (domain defaulted to
/// [`PhysicsDomain::ClassicalMechanics`], symmetries empty) and runs the full
/// multi-aspect search. Returns the top 5 catalog matches with structural,
/// symmetry, dimensional, and full-similarity breakdowns.
///
/// The novelty threshold is 0.40 — formulas with no catalog neighbor above this
/// score are flagged as `is_novel = true`.
pub fn recognize_expr(
    engine: &PhysicsSearchEngine,
    expr: &Expr,
    name: &str,
) -> RecognitionReport {
    let rhs = expr_to_equation_node(expr);
    // IMPORTANT: Catalog entries are stored as Equals(lhs, rhs) trees because
    // they represent equations (F = ma, E = ½v² − 1/r, etc.). A bare RHS
    // skeleton won't match those at the top level. Wrap the discovered formula
    // in a generic Equals(Constant("result"), rhs) so both sides use the
    // Equals root node. The `result` constant name is irrelevant in skeleton
    // mode (constants are interchangeable), and it carries no bits that would
    // contaminate the structural comparison.
    let ast = EquationNode::Equals(
        Box::new(EquationNode::Constant { name: "result".into() }),
        Box::new(rhs),
    );
    let query = PhysicsEquation {
        name: name.to_string(),
        domain: PhysicsDomain::ClassicalMechanics,
        ast,
        symmetries: SymmetryDescriptor::none(),
        dimensions: DimensionalSignature::DIMENSIONLESS,
        tensor: None,
    };

    let results = engine.search_equation(&query, 5);
    let (best_match, best_domain, best_score) = results
        .first()
        .map(|r| (Some(r.name.clone()), Some(r.domain), r.score))
        .unwrap_or((None, None, 0.0));

    RecognitionReport {
        is_novel: best_score < 0.40,
        best_match,
        best_domain,
        best_score,
        all_results: results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::conjecture_engine::{BinOp, Expr};

    /// A simple polynomial `n² + n` should convert to a Sum of a Power and a Constant.
    #[test]
    fn test_expr_to_equation_node_polynomial() {
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("n".into())),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::Var("n".into())),
        );
        let node = expr_to_equation_node(&expr);
        match node {
            EquationNode::Sum(terms) => {
                assert_eq!(terms.len(), 2, "should flatten to two terms");
                assert!(matches!(terms[0], EquationNode::Power { .. }));
                assert!(matches!(terms[1], EquationNode::Constant { .. }));
            }
            _ => panic!("expected Sum, got {:?}", node),
        }
    }

    /// Division `1/r` should become `1 * r^(-1)`.
    #[test]
    fn test_expr_to_equation_node_division() {
        let expr = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Var("r".into())),
        );
        let node = expr_to_equation_node(&expr);
        match node {
            EquationNode::Product(factors) => {
                assert_eq!(factors.len(), 2);
                assert!(matches!(factors[0], EquationNode::Scalar(_)));
                assert!(matches!(factors[1], EquationNode::Power { .. }));
            }
            _ => panic!("expected Product, got {:?}", node),
        }
    }

    /// The Kepler energy skeleton `½(vx² + vy²) - 1/r` should convert without
    /// panic and yield a Sum of a Product and a negated Product.
    #[test]
    fn test_kepler_energy_conversion() {
        let v_sq = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("vx".into())),
                Box::new(Expr::Const(2.0)),
            )),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("vy".into())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let kinetic = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(0.5)),
            Box::new(v_sq),
        );
        let potential = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Var("r".into())),
        );
        let energy = Expr::BinOp(BinOp::Sub, Box::new(kinetic), Box::new(potential));

        // Should not panic
        let node = expr_to_equation_node(&energy);
        // Top-level is Sub → Sum(kinetic, Negate(potential))
        match node {
            EquationNode::Sum(terms) => {
                assert_eq!(terms.len(), 2);
                // Second term should be the negated potential
                assert!(matches!(terms[1], EquationNode::Negate(_)));
            }
            _ => panic!("expected Sum at top level, got {:?}", node),
        }
    }

    /// End-to-end recognition: convert a Kepler energy skeleton and search the catalog.
    /// Does not assert a specific match — the catalog may or may not contain an
    /// exact Kepler entry — but asserts the search returns results without panicking.
    #[test]
    fn test_recognize_kepler_energy_runs() {
        let engine = PhysicsSearchEngine::new();
        let energy = Expr::BinOp(
            BinOp::Sub,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Const(0.5)),
                Box::new(Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::BinOp(
                        BinOp::Pow,
                        Box::new(Expr::Var("vx".into())),
                        Box::new(Expr::Const(2.0)),
                    )),
                    Box::new(Expr::BinOp(
                        BinOp::Pow,
                        Box::new(Expr::Var("vy".into())),
                        Box::new(Expr::Const(2.0)),
                    )),
                )),
            )),
            Box::new(Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::Const(1.0)),
                Box::new(Expr::Var("r".into())),
            )),
        );

        let report = recognize_expr(&engine, &energy, "kepler_energy_test");
        eprintln!("\n═══ KEPLER ENERGY RECOGNITION ═══");
        eprintln!("  {}", report.headline());
        eprintln!("  Best score: {:.3}", report.best_score);
        eprintln!("  Top 5 catalog matches:");
        for (i, r) in report.all_results.iter().enumerate() {
            eprintln!(
                "    {}. {} ({:?}) score={:.3}",
                i + 1,
                r.name,
                r.domain,
                r.score
            );
        }

        // Contract: function must not panic and must return at most 5 results
        assert!(report.all_results.len() <= 5);
    }

    /// End-to-end recognition: the hydrogen energy formula `-13.6/n²`.
    #[test]
    fn test_recognize_hydrogen_formula_runs() {
        let engine = PhysicsSearchEngine::new();
        let expr = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Const(-13.6)),
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("n".into())),
                Box::new(Expr::Const(2.0)),
            )),
        );

        let report = recognize_expr(&engine, &expr, "hydrogen_E_test");
        eprintln!("\n═══ HYDROGEN FORMULA RECOGNITION ═══");
        eprintln!("  {}", report.headline());
        for r in report.all_results.iter().take(3) {
            eprintln!("    {} ({:?}) score={:.3}", r.name, r.domain, r.score);
        }

        assert!(!report.all_results.is_empty());
    }

    /// Convert trig functions to single-atom catalog-matching Constants.
    #[test]
    fn test_trig_conversion_catalog_matching() {
        let sin_x = Expr::Func(UnaryFn::Sin, Box::new(Expr::Var("x".into())));
        let node = expr_to_equation_node(&sin_x);
        // Encoded as single Constant("sin(x)") matching catalog convention
        match node {
            EquationNode::Constant { name } => {
                assert_eq!(name, "sin(x)", "should be single-atom constant");
            }
            _ => panic!("expected Constant for sin(x), got {:?}", node),
        }

        // Also verify nested argument is rendered via Display
        let sin_n_plus_1 = Expr::Func(
            UnaryFn::Sin,
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Var("n".into())),
                Box::new(Expr::Const(1.0)),
            )),
        );
        let node2 = expr_to_equation_node(&sin_n_plus_1);
        if let EquationNode::Constant { name } = node2 {
            assert!(name.starts_with("sin"), "should be sin(...): {}", name);
        } else {
            panic!("expected Constant");
        }
    }

}
