// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Q0 structural-HDC qualification harness for mathematical ASTs.
//!
//! This is deliberately an example-local encoder first: it gives MATH-EXP-001A
//! an executable, frozen intervention without changing runtime retrieval,
//! mathematical memory, proof authority, or the core `hdc` module surface.
//! A later tranche can migrate the qualified encoder into the library.

use std::collections::BTreeMap;

use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::fol_formula_ext::{ArithOp, FolFormulaExt, NumericType, Term};
use symthaea_core::hdc::logic_engine::Proposition;
use symthaea_core::hdc::primitive_system::seed_from_name;

/// Stable identity for the intervention under test.
///
/// Changing any encoding rule requires a new ID and a new experiment lineage.
pub const ENCODER_ID: &str = "symthaea-math-structural-hdc-v1";

#[derive(Default)]
struct StructuralMathEncoderV1 {
    bound_vars: Vec<String>,
    free_term_vars: BTreeMap<String, usize>,
    proposition_atoms: BTreeMap<String, usize>,
}

impl StructuralMathEncoderV1 {
    fn encode(formula: &FolFormulaExt) -> BinaryHV {
        Self::default().encode_formula(formula)
    }

    fn token(label: &str) -> BinaryHV {
        BinaryHV::random(seed_from_name(&format!("{ENCODER_ID}::{label}")))
    }

    fn ordered_node(tag: &str, left: BinaryHV, right: BinaryHV) -> BinaryHV {
        let node = Self::token(tag);
        let lhs = Self::token("ROLE_LEFT").bind(&left.permute(1));
        let rhs = Self::token("ROLE_RIGHT").bind(&right.permute(7));
        BinaryHV::bundle(&[node, lhs, rhs])
    }

    fn commutative_node(tag: &str, left: BinaryHV, right: BinaryHV) -> BinaryHV {
        let node = Self::token(tag);
        let member_role = Self::token("ROLE_COMMUTATIVE_MEMBER");
        let lhs = member_role.bind(&left.permute(3));
        let rhs = member_role.bind(&right.permute(3));
        BinaryHV::bundle(&[node, lhs, rhs])
    }

    fn unary_node(tag: &str, child: BinaryHV) -> BinaryHV {
        let node = Self::token(tag);
        let body = Self::token("ROLE_BODY").bind(&child.permute(5));
        BinaryHV::bundle(&[node, body, Self::token("ARITY_1")])
    }

    fn quantifier_node(tag: &str, ty: NumericType, body: BinaryHV) -> BinaryHV {
        let node = Self::token(tag).bind(&Self::type_token(ty));
        let body = Self::token("ROLE_QUANTIFIER_BODY").bind(&body.permute(11));
        BinaryHV::bundle(&[node, body, Self::token("QUANTIFIER")])
    }

    fn type_token(ty: NumericType) -> BinaryHV {
        match ty {
            NumericType::Int => Self::token("TYPE_INT"),
            NumericType::Real => Self::token("TYPE_REAL"),
            NumericType::Nat => Self::token("TYPE_NAT"),
        }
    }

    fn encode_formula(&mut self, formula: &FolFormulaExt) -> BinaryHV {
        match formula {
            FolFormulaExt::Base(prop) => self.encode_prop(prop),
            FolFormulaExt::Eq(lhs, rhs) => {
                let lhs = self.encode_term(lhs);
                let rhs = self.encode_term(rhs);
                Self::commutative_node("FORMULA_EQ", lhs, rhs)
            }
            FolFormulaExt::Lt(lhs, rhs) => {
                let lhs = self.encode_term(lhs);
                let rhs = self.encode_term(rhs);
                Self::ordered_node("FORMULA_LT", lhs, rhs)
            }
            FolFormulaExt::Le(lhs, rhs) => {
                let lhs = self.encode_term(lhs);
                let rhs = self.encode_term(rhs);
                Self::ordered_node("FORMULA_LE", lhs, rhs)
            }
            FolFormulaExt::And(lhs, rhs) => {
                let lhs = self.encode_formula(lhs);
                let rhs = self.encode_formula(rhs);
                Self::commutative_node("FORMULA_AND", lhs, rhs)
            }
            FolFormulaExt::Or(lhs, rhs) => {
                let lhs = self.encode_formula(lhs);
                let rhs = self.encode_formula(rhs);
                Self::commutative_node("FORMULA_OR", lhs, rhs)
            }
            FolFormulaExt::Not(body) => {
                let body = self.encode_formula(body);
                Self::unary_node("FORMULA_NOT", body)
            }
            FolFormulaExt::Implies(lhs, rhs) => {
                let lhs = self.encode_formula(lhs);
                let rhs = self.encode_formula(rhs);
                Self::ordered_node("FORMULA_IMPLIES", lhs, rhs)
            }
            FolFormulaExt::Forall(var, ty, body) => {
                self.bound_vars.push(var.clone());
                let body = self.encode_formula(body);
                self.bound_vars.pop();
                Self::quantifier_node("FORMULA_FORALL", *ty, body)
            }
            FolFormulaExt::Exists(var, ty, body) => {
                self.bound_vars.push(var.clone());
                let body = self.encode_formula(body);
                self.bound_vars.pop();
                Self::quantifier_node("FORMULA_EXISTS", *ty, body)
            }
        }
    }

    fn encode_term(&mut self, term: &Term) -> BinaryHV {
        match term {
            Term::Var(name) => self.encode_variable(name),
            Term::IntLit(value) => Self::token(Self::int_class(*value)),
            Term::RealLit(value) => Self::token(Self::real_class(*value)),
            Term::RatLit(num, den) => {
                let sign = if *num == 0 {
                    "ZERO"
                } else if (*num < 0) ^ (*den < 0) {
                    "NEG"
                } else {
                    "POS"
                };
                let denom = if *den == 0 {
                    "DEN_ZERO"
                } else if den.unsigned_abs() == 1 {
                    "DEN_ONE"
                } else {
                    "DEN_OTHER"
                };
                Self::token(&format!("TERM_RATIONAL_{sign}_{denom}"))
            }
            Term::BinOp(op, lhs, rhs) => {
                let lhs = self.encode_term(lhs);
                let rhs = self.encode_term(rhs);
                match op {
                    ArithOp::Add => Self::commutative_node("TERM_ADD", lhs, rhs),
                    ArithOp::Mul => Self::commutative_node("TERM_MUL", lhs, rhs),
                    ArithOp::Sub => Self::ordered_node("TERM_SUB", lhs, rhs),
                    ArithOp::Div => Self::ordered_node("TERM_DIV", lhs, rhs),
                }
            }
            Term::Pow(base, exponent) => {
                let base = self.encode_term(base);
                let exponent = Self::token(Self::pow_class(*exponent));
                Self::ordered_node("TERM_POW", base, exponent)
            }
            Term::Neg(body) => {
                let body = self.encode_term(body);
                Self::unary_node("TERM_NEG", body)
            }
        }
    }

    fn encode_variable(&mut self, name: &str) -> BinaryHV {
        if let Some(distance) = self.bound_vars.iter().rev().position(|bound| bound == name) {
            return Self::token(&format!("BOUND_VAR_DISTANCE_{distance}"));
        }

        let next = self.free_term_vars.len();
        let index = *self.free_term_vars.entry(name.to_string()).or_insert(next);
        Self::token(&format!("FREE_TERM_VAR_{index}"))
    }

    fn encode_prop(&mut self, prop: &Proposition) -> BinaryHV {
        match prop {
            Proposition::Atom(name) => {
                let next = self.proposition_atoms.len();
                let index = *self
                    .proposition_atoms
                    .entry(name.clone())
                    .or_insert(next);
                Self::token(&format!("PROP_ATOM_{index}"))
            }
            Proposition::Not(body) => {
                let body = self.encode_prop(body);
                Self::unary_node("PROP_NOT", body)
            }
            Proposition::And(lhs, rhs) => {
                let lhs = self.encode_prop(lhs);
                let rhs = self.encode_prop(rhs);
                Self::commutative_node("PROP_AND", lhs, rhs)
            }
            Proposition::Or(lhs, rhs) => {
                let lhs = self.encode_prop(lhs);
                let rhs = self.encode_prop(rhs);
                Self::commutative_node("PROP_OR", lhs, rhs)
            }
            Proposition::Implies(lhs, rhs) => {
                let lhs = self.encode_prop(lhs);
                let rhs = self.encode_prop(rhs);
                Self::ordered_node("PROP_IMPLIES", lhs, rhs)
            }
            Proposition::Iff(lhs, rhs) => {
                let lhs = self.encode_prop(lhs);
                let rhs = self.encode_prop(rhs);
                Self::commutative_node("PROP_IFF", lhs, rhs)
            }
            Proposition::True => Self::token("PROP_TRUE"),
            Proposition::False => Self::token("PROP_FALSE"),
        }
    }

    fn int_class(value: i64) -> &'static str {
        match value {
            0 => "TERM_INT_ZERO",
            1 => "TERM_INT_ONE",
            -1 => "TERM_INT_NEG_ONE",
            2..=9 => "TERM_INT_SMALL_POS",
            -9..=-2 => "TERM_INT_SMALL_NEG",
            value if value > 0 => "TERM_INT_POS",
            _ => "TERM_INT_NEG",
        }
    }

    fn real_class(value: f64) -> &'static str {
        if value.is_nan() {
            "TERM_REAL_NAN"
        } else if value.is_infinite() && value.is_sign_positive() {
            "TERM_REAL_POS_INF"
        } else if value.is_infinite() {
            "TERM_REAL_NEG_INF"
        } else if value == 0.0 {
            "TERM_REAL_ZERO"
        } else if value == 1.0 {
            "TERM_REAL_ONE"
        } else if value == -1.0 {
            "TERM_REAL_NEG_ONE"
        } else if value.is_sign_positive() {
            "TERM_REAL_POS"
        } else {
            "TERM_REAL_NEG"
        }
    }

    fn pow_class(exponent: u32) -> &'static str {
        match exponent {
            0 => "POW_EXP_ZERO",
            1 => "POW_EXP_ONE",
            2 => "POW_EXP_TWO",
            3 => "POW_EXP_THREE",
            4..=8 => "POW_EXP_SMALL",
            _ => "POW_EXP_LARGE",
        }
    }
}

fn linear_equation(var: &str, offset: i64, rhs: i64) -> FolFormulaExt {
    FolFormulaExt::forall(
        var,
        NumericType::Real,
        FolFormulaExt::eq(Term::var(var).add(Term::int(offset)), Term::int(rhs)),
    )
}

fn multiplicative_equation(var: &str, factor: i64, rhs: i64) -> FolFormulaExt {
    FolFormulaExt::forall(
        var,
        NumericType::Real,
        FolFormulaExt::eq(Term::var(var).mul(Term::int(factor)), Term::int(rhs)),
    )
}

fn implication_pair(reversed: bool) -> FolFormulaExt {
    let premise = FolFormulaExt::eq(Term::var("x"), Term::int(0));
    let conclusion = FolFormulaExt::le(Term::var("x"), Term::int(1));
    let body = if reversed {
        conclusion.implies(premise)
    } else {
        premise.implies(conclusion)
    };
    FolFormulaExt::forall("x", NumericType::Real, body)
}

fn q0_report() -> Vec<(&'static str, f32)> {
    let query = StructuralMathEncoderV1::encode(&linear_equation("q", 4, 9));
    let structural_neighbor = StructuralMathEncoderV1::encode(&linear_equation("x", 2, 5));
    let operator_distractor =
        StructuralMathEncoderV1::encode(&multiplicative_equation("x", 2, 6));
    let shuffled_control = BinaryHV::random(seed_from_name("MATH_EXP_001_Q0_SHUFFLED_CONTROL"));

    vec![
        ("structural_neighbor", query.similarity(&structural_neighbor)),
        ("operator_distractor", query.similarity(&operator_distractor)),
        ("shuffled_control", query.similarity(&shuffled_control)),
    ]
}

fn main() {
    println!("encoder_id,{ENCODER_ID}");
    println!("fixture,similarity");
    for (name, similarity) in q0_report() {
        println!("{name},{similarity:.6}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_encoding_is_exact() {
        let formula = linear_equation("x", 2, 5);
        assert_eq!(
            StructuralMathEncoderV1::encode(&formula),
            StructuralMathEncoderV1::encode(&formula)
        );
    }

    #[test]
    fn alpha_renaming_and_same_literal_classes_are_invariant() {
        let a = linear_equation("x", 2, 5);
        let b = linear_equation("renamed", 3, 7);
        assert_eq!(
            StructuralMathEncoderV1::encode(&a),
            StructuralMathEncoderV1::encode(&b)
        );
    }

    #[test]
    fn commutative_addition_is_order_invariant() {
        let a = FolFormulaExt::forall(
            "x",
            NumericType::Real,
            FolFormulaExt::forall(
                "y",
                NumericType::Real,
                FolFormulaExt::eq(
                    Term::var("x").add(Term::var("y")),
                    Term::int(0),
                ),
            ),
        );
        let b = FolFormulaExt::forall(
            "x",
            NumericType::Real,
            FolFormulaExt::forall(
                "y",
                NumericType::Real,
                FolFormulaExt::eq(
                    Term::var("y").add(Term::var("x")),
                    Term::int(0),
                ),
            ),
        );
        assert_eq!(
            StructuralMathEncoderV1::encode(&a),
            StructuralMathEncoderV1::encode(&b)
        );
    }

    #[test]
    fn implication_direction_is_preserved() {
        assert_ne!(
            StructuralMathEncoderV1::encode(&implication_pair(false)),
            StructuralMathEncoderV1::encode(&implication_pair(true))
        );
    }

    #[test]
    fn repeated_variable_structure_is_not_erased() {
        let repeated = FolFormulaExt::eq(
            Term::var("x").add(Term::var("x")),
            Term::int(0),
        );
        let distinct = FolFormulaExt::eq(
            Term::var("x").add(Term::var("y")),
            Term::int(0),
        );
        assert_ne!(
            StructuralMathEncoderV1::encode(&repeated),
            StructuralMathEncoderV1::encode(&distinct)
        );
    }

    #[test]
    fn quantifier_type_changes_embedding() {
        let real = FolFormulaExt::forall(
            "x",
            NumericType::Real,
            FolFormulaExt::eq(Term::var("x"), Term::int(0)),
        );
        let integer = FolFormulaExt::forall(
            "x",
            NumericType::Int,
            FolFormulaExt::eq(Term::var("x"), Term::int(0)),
        );
        assert_ne!(
            StructuralMathEncoderV1::encode(&real),
            StructuralMathEncoderV1::encode(&integer)
        );
    }

    #[test]
    fn q0_structural_neighbor_beats_operator_and_shuffled_controls() {
        let report = q0_report();
        let structural = report[0].1;
        let operator = report[1].1;
        let shuffled = report[2].1;

        assert!(structural > 0.99, "structural similarity={structural}");
        assert!(
            structural > operator + 0.10,
            "structural={structural}, operator={operator}"
        );
        assert!(
            structural > shuffled + 0.10,
            "structural={structural}, shuffled={shuffled}"
        );
    }
}
