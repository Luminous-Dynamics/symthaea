// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Q0 comparison harness: structural HDC vs a conventional canonical-AST retriever.
//!
//! This intentionally reuses the exact #4087 encoder implementation by including
//! that example as a private module and exposing only a tiny wrapper. That avoids
//! silently forking the HDC intervention while letting this stacked experiment
//! compare it against a non-HDC structural representation.

use serde::Deserialize;
use std::collections::BTreeMap;
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::fol_formula_ext::{ArithOp, FolFormulaExt, NumericType, Term};
use symthaea_core::hdc::logic_engine::Proposition;
use symthaea_core::hdc::primitive_system::seed_from_name;

mod frozen_hdc {
    include!("math_structural_hdc_q0.rs");

    pub(super) fn encode_formula(
        formula: &symthaea_core::hdc::fol_formula_ext::FolFormulaExt,
    ) -> symthaea_core::hdc::binary_hv::BinaryHV {
        StructuralMathEncoderV1::encode(formula)
    }

    pub(super) fn encoder_id() -> &'static str {
        ENCODER_ID
    }
}

const MANIFEST_JSON: &str = include_str!(
    "../../../../data/benchmarks/math_structural_q0_v1.json"
);
const BASELINE_ID: &str = "canonical-ast-sparse-v1";

#[derive(Debug, Deserialize)]
struct Manifest {
    version: String,
    cases: Vec<ManifestCase>,
}

#[derive(Debug, Deserialize)]
struct ManifestCase {
    id: String,
    positive_id: String,
}

struct Q0Case {
    id: &'static str,
    positive_id: &'static str,
    query: FolFormulaExt,
    candidates: Vec<(&'static str, FolFormulaExt)>,
}

#[derive(Default)]
struct CanonicalAstEncoder {
    features: BTreeMap<String, f64>,
    bound_vars: Vec<String>,
    free_vars: BTreeMap<String, usize>,
    proposition_atoms: BTreeMap<String, usize>,
}

impl CanonicalAstEncoder {
    fn encode(formula: &FolFormulaExt) -> BTreeMap<String, f64> {
        let mut encoder = Self::default();
        encoder.visit_formula(formula, 0, "ROOT");
        encoder.features
    }

    fn bump(&mut self, key: impl Into<String>) {
        *self.features.entry(key.into()).or_insert(0.0) += 1.0;
    }

    fn node(&mut self, tag: &str, depth: usize, role: &str) {
        self.bump(format!("node:{tag}"));
        self.bump(format!("depth:{}:{tag}", depth.min(8)));
        self.bump(format!("role:{role}:{tag}"));
    }

    fn visit_formula(&mut self, formula: &FolFormulaExt, depth: usize, role: &str) {
        match formula {
            FolFormulaExt::Base(prop) => {
                self.node("FORMULA_BASE", depth, role);
                self.visit_prop(prop, depth + 1, "BODY");
            }
            FolFormulaExt::Eq(lhs, rhs) => {
                self.node("FORMULA_EQ", depth, role);
                self.visit_term(lhs, depth + 1, "M");
                self.visit_term(rhs, depth + 1, "M");
            }
            FolFormulaExt::Lt(lhs, rhs) => {
                self.node("FORMULA_LT", depth, role);
                self.visit_term(lhs, depth + 1, "L");
                self.visit_term(rhs, depth + 1, "R");
            }
            FolFormulaExt::Le(lhs, rhs) => {
                self.node("FORMULA_LE", depth, role);
                self.visit_term(lhs, depth + 1, "L");
                self.visit_term(rhs, depth + 1, "R");
            }
            FolFormulaExt::And(lhs, rhs) => {
                self.node("FORMULA_AND", depth, role);
                self.visit_formula(lhs, depth + 1, "M");
                self.visit_formula(rhs, depth + 1, "M");
            }
            FolFormulaExt::Or(lhs, rhs) => {
                self.node("FORMULA_OR", depth, role);
                self.visit_formula(lhs, depth + 1, "M");
                self.visit_formula(rhs, depth + 1, "M");
            }
            FolFormulaExt::Not(body) => {
                self.node("FORMULA_NOT", depth, role);
                self.visit_formula(body, depth + 1, "BODY");
            }
            FolFormulaExt::Implies(lhs, rhs) => {
                self.node("FORMULA_IMPLIES", depth, role);
                self.visit_formula(lhs, depth + 1, "PREMISE");
                self.visit_formula(rhs, depth + 1, "CONCLUSION");
            }
            FolFormulaExt::Forall(var, ty, body) => {
                self.node("FORMULA_FORALL", depth, role);
                self.bump(format!("quantifier_type:{}", type_name(*ty)));
                self.bound_vars.push(var.clone());
                self.visit_formula(body, depth + 1, "QBODY");
                self.bound_vars.pop();
            }
            FolFormulaExt::Exists(var, ty, body) => {
                self.node("FORMULA_EXISTS", depth, role);
                self.bump(format!("quantifier_type:{}", type_name(*ty)));
                self.bound_vars.push(var.clone());
                self.visit_formula(body, depth + 1, "QBODY");
                self.bound_vars.pop();
            }
        }
    }

    fn visit_term(&mut self, term: &Term, depth: usize, role: &str) {
        match term {
            Term::Var(name) => {
                self.node("TERM_VAR", depth, role);
                if let Some(distance) = self.bound_vars.iter().rev().position(|v| v == name) {
                    self.bump(format!("bound_var_distance:{distance}"));
                } else {
                    let next = self.free_vars.len();
                    let index = *self.free_vars.entry(name.clone()).or_insert(next);
                    self.bump(format!("free_var_id:{index}"));
                }
            }
            Term::IntLit(value) => {
                self.node(int_class(*value), depth, role);
            }
            Term::RealLit(value) => {
                self.node(real_class(*value), depth, role);
            }
            Term::RatLit(num, den) => {
                self.node(rational_class(*num, *den), depth, role);
            }
            Term::BinOp(op, lhs, rhs) => {
                let (tag, lhs_role, rhs_role) = match op {
                    ArithOp::Add => ("TERM_ADD", "M", "M"),
                    ArithOp::Mul => ("TERM_MUL", "M", "M"),
                    ArithOp::Sub => ("TERM_SUB", "L", "R"),
                    ArithOp::Div => ("TERM_DIV", "L", "R"),
                };
                self.node(tag, depth, role);
                self.visit_term(lhs, depth + 1, lhs_role);
                self.visit_term(rhs, depth + 1, rhs_role);
            }
            Term::Pow(base, exponent) => {
                self.node("TERM_POW", depth, role);
                self.bump(format!("pow_class:{}", pow_class(*exponent)));
                self.visit_term(base, depth + 1, "BASE");
            }
            Term::Neg(body) => {
                self.node("TERM_NEG", depth, role);
                self.visit_term(body, depth + 1, "BODY");
            }
        }
    }

    fn visit_prop(&mut self, prop: &Proposition, depth: usize, role: &str) {
        match prop {
            Proposition::Atom(name) => {
                self.node("PROP_ATOM", depth, role);
                let next = self.proposition_atoms.len();
                let index = *self
                    .proposition_atoms
                    .entry(name.clone())
                    .or_insert(next);
                self.bump(format!("prop_atom_id:{index}"));
            }
            Proposition::Not(body) => {
                self.node("PROP_NOT", depth, role);
                self.visit_prop(body, depth + 1, "BODY");
            }
            Proposition::And(lhs, rhs) => {
                self.node("PROP_AND", depth, role);
                self.visit_prop(lhs, depth + 1, "M");
                self.visit_prop(rhs, depth + 1, "M");
            }
            Proposition::Or(lhs, rhs) => {
                self.node("PROP_OR", depth, role);
                self.visit_prop(lhs, depth + 1, "M");
                self.visit_prop(rhs, depth + 1, "M");
            }
            Proposition::Implies(lhs, rhs) => {
                self.node("PROP_IMPLIES", depth, role);
                self.visit_prop(lhs, depth + 1, "PREMISE");
                self.visit_prop(rhs, depth + 1, "CONCLUSION");
            }
            Proposition::Iff(lhs, rhs) => {
                self.node("PROP_IFF", depth, role);
                self.visit_prop(lhs, depth + 1, "M");
                self.visit_prop(rhs, depth + 1, "M");
            }
            Proposition::True => self.node("PROP_TRUE", depth, role),
            Proposition::False => self.node("PROP_FALSE", depth, role),
        }
    }
}

fn type_name(ty: NumericType) -> &'static str {
    match ty {
        NumericType::Int => "INT",
        NumericType::Real => "REAL",
        NumericType::Nat => "NAT",
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

fn rational_class(num: i64, den: i64) -> &'static str {
    match (num, den) {
        (0, _) => "TERM_RATIONAL_ZERO",
        (_, 0) => "TERM_RATIONAL_DEN_ZERO",
        (n, d) if (n < 0) ^ (d < 0) => "TERM_RATIONAL_NEG",
        _ => "TERM_RATIONAL_POS",
    }
}

fn pow_class(exponent: u32) -> &'static str {
    match exponent {
        0 => "ZERO",
        1 => "ONE",
        2 => "TWO",
        3 => "THREE",
        4..=8 => "SMALL",
        _ => "LARGE",
    }
}

fn cosine(a: &BTreeMap<String, f64>, b: &BTreeMap<String, f64>) -> f64 {
    let dot: f64 = a
        .iter()
        .map(|(key, value)| value * b.get(key).copied().unwrap_or(0.0))
        .sum();
    let a_norm = a.values().map(|value| value * value).sum::<f64>().sqrt();
    let b_norm = b.values().map(|value| value * value).sum::<f64>().sqrt();
    if a_norm == 0.0 || b_norm == 0.0 {
        0.0
    } else {
        dot / (a_norm * b_norm)
    }
}

fn linear_additive(var: &str, offset: i64, rhs: i64, ty: NumericType) -> FolFormulaExt {
    FolFormulaExt::forall(
        var,
        ty,
        FolFormulaExt::eq(Term::var(var).add(Term::int(offset)), Term::int(rhs)),
    )
}

fn implication(var: &str, reversed: bool) -> FolFormulaExt {
    let premise = FolFormulaExt::eq(Term::var(var), Term::int(0));
    let conclusion = FolFormulaExt::le(Term::var(var), Term::int(1));
    let body = if reversed {
        conclusion.implies(premise)
    } else {
        premise.implies(conclusion)
    };
    FolFormulaExt::forall(var, NumericType::Real, body)
}

fn repeated_pair(left: &str, right: &str, multiply: bool) -> FolFormulaExt {
    let lhs = if multiply {
        Term::var(left).mul(Term::var(right))
    } else {
        Term::var(left).add(Term::var(right))
    };
    FolFormulaExt::eq(lhs, Term::int(0))
}

fn nested_quantifiers(outer_forall: bool, inner_ty: NumericType, a: &str, b: &str) -> FolFormulaExt {
    let body = FolFormulaExt::eq(Term::var(a).add(Term::var(b)), Term::int(0));
    let inner = FolFormulaExt::exists(b, inner_ty, body);
    if outer_forall {
        FolFormulaExt::forall(a, NumericType::Real, inner)
    } else {
        FolFormulaExt::exists(
            a,
            NumericType::Real,
            FolFormulaExt::forall(
                b,
                inner_ty,
                FolFormulaExt::eq(Term::var(a).add(Term::var(b)), Term::int(0)),
            ),
        )
    }
}

fn commutative_pair(a: &str, b: &str, subtract: bool) -> FolFormulaExt {
    let lhs = if subtract {
        Term::var(a).sub(Term::var(b))
    } else {
        Term::var(a).add(Term::var(b))
    };
    FolFormulaExt::eq(lhs, Term::int(0))
}

fn q0_cases() -> Vec<Q0Case> {
    vec![
        Q0Case {
            id: "linear-additive-alpha-literal",
            positive_id: "linear-additive-positive",
            query: linear_additive("q", 4, 9, NumericType::Real),
            candidates: vec![
                ("linear-additive-positive", linear_additive("x", 2, 5, NumericType::Real)),
                (
                    "operator-distractor",
                    FolFormulaExt::forall(
                        "x",
                        NumericType::Real,
                        FolFormulaExt::eq(Term::var("x").mul(Term::int(2)), Term::int(6)),
                    ),
                ),
                ("type-distractor", linear_additive("x", 2, 5, NumericType::Int)),
                (
                    "relation-distractor",
                    FolFormulaExt::forall(
                        "x",
                        NumericType::Real,
                        FolFormulaExt::le(Term::var("x").add(Term::int(2)), Term::int(5)),
                    ),
                ),
            ],
        },
        Q0Case {
            id: "implication-direction",
            positive_id: "implication-positive",
            query: implication("x", false),
            candidates: vec![
                ("implication-positive", implication("renamed", false)),
                ("reversed-implication", implication("x", true)),
                (
                    "conjunction-distractor",
                    FolFormulaExt::forall(
                        "x",
                        NumericType::Real,
                        FolFormulaExt::eq(Term::var("x"), Term::int(0))
                            .and(FolFormulaExt::le(Term::var("x"), Term::int(1))),
                    ),
                ),
            ],
        },
        Q0Case {
            id: "variable-coreference",
            positive_id: "coreference-positive",
            query: repeated_pair("x", "x", false),
            candidates: vec![
                ("coreference-positive", repeated_pair("a", "a", false)),
                ("distinct-variable-distractor", repeated_pair("a", "b", false)),
                ("operator-distractor", repeated_pair("a", "a", true)),
            ],
        },
        Q0Case {
            id: "quantifier-nesting-type",
            positive_id: "quantifier-positive",
            query: nested_quantifiers(true, NumericType::Real, "x", "y"),
            candidates: vec![
                (
                    "quantifier-positive",
                    nested_quantifiers(true, NumericType::Real, "alpha", "beta"),
                ),
                (
                    "quantifier-order-distractor",
                    nested_quantifiers(false, NumericType::Real, "x", "y"),
                ),
                (
                    "inner-type-distractor",
                    nested_quantifiers(true, NumericType::Int, "x", "y"),
                ),
            ],
        },
        Q0Case {
            id: "commutative-child-order",
            positive_id: "commutative-positive",
            query: commutative_pair("x", "y", false),
            candidates: vec![
                ("commutative-positive", commutative_pair("beta", "alpha", false)),
                ("ordered-operator-distractor", commutative_pair("alpha", "beta", true)),
                (
                    "relation-distractor",
                    FolFormulaExt::lt(
                        Term::var("alpha").add(Term::var("beta")),
                        Term::int(0),
                    ),
                ),
            ],
        },
    ]
}

struct CaseScore {
    id: &'static str,
    hdc_positive: f32,
    hdc_best_negative: f32,
    hdc_shuffled: f32,
    ast_positive: f64,
    ast_best_negative: f64,
    hdc_top1: bool,
    ast_top1: bool,
}

fn score_case(case: &Q0Case) -> CaseScore {
    let hdc_query = frozen_hdc::encode_formula(&case.query);
    let ast_query = CanonicalAstEncoder::encode(&case.query);

    let mut hdc_positive = 0.0_f32;
    let mut hdc_best_negative = f32::NEG_INFINITY;
    let mut ast_positive = 0.0_f64;
    let mut ast_best_negative = f64::NEG_INFINITY;

    for (candidate_id, formula) in &case.candidates {
        let hdc_similarity = hdc_query.similarity(&frozen_hdc::encode_formula(formula));
        let ast_similarity = cosine(&ast_query, &CanonicalAstEncoder::encode(formula));
        if *candidate_id == case.positive_id {
            hdc_positive = hdc_similarity;
            ast_positive = ast_similarity;
        } else {
            hdc_best_negative = hdc_best_negative.max(hdc_similarity);
            ast_best_negative = ast_best_negative.max(ast_similarity);
        }
    }

    let shuffled = BinaryHV::random(seed_from_name(&format!(
        "MATH_STRUCTURAL_Q0_V1_SHUFFLED::{}",
        case.id
    )));

    CaseScore {
        id: case.id,
        hdc_positive,
        hdc_best_negative,
        hdc_shuffled: hdc_query.similarity(&shuffled),
        ast_positive,
        ast_best_negative,
        hdc_top1: hdc_positive > hdc_best_negative,
        ast_top1: ast_positive > ast_best_negative,
    }
}

fn manifest() -> Manifest {
    serde_json::from_str(MANIFEST_JSON).expect("Q0 manifest must parse")
}

fn main() {
    println!("manifest_version,{}", manifest().version);
    println!("hdc_encoder_id,{}", frozen_hdc::encoder_id());
    println!("structural_baseline_id,{BASELINE_ID}");
    println!("case,hdc_positive,hdc_best_negative,hdc_shuffled,hdc_margin,hdc_top1,ast_positive,ast_best_negative,ast_margin,ast_top1");
    for case in q0_cases() {
        let score = score_case(&case);
        println!(
            "{},{:.6},{:.6},{:.6},{:.6},{},{:.6},{:.6},{:.6},{}",
            score.id,
            score.hdc_positive,
            score.hdc_best_negative,
            score.hdc_shuffled,
            score.hdc_positive - score.hdc_best_negative,
            score.hdc_top1,
            score.ast_positive,
            score.ast_best_negative,
            score.ast_positive - score.ast_best_negative,
            score.ast_top1,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_and_executable_cases_are_locked_together() {
        let manifest = manifest();
        let cases = q0_cases();
        assert_eq!(manifest.version, "math-structural-q0-v1");
        assert_eq!(manifest.cases.len(), cases.len());
        for (manifest_case, executable_case) in manifest.cases.iter().zip(cases.iter()) {
            assert_eq!(manifest_case.id, executable_case.id);
            assert_eq!(manifest_case.positive_id, executable_case.positive_id);
        }
    }

    #[test]
    fn canonical_ast_is_alpha_and_literal_class_invariant() {
        let a = CanonicalAstEncoder::encode(&linear_additive(
            "x",
            2,
            5,
            NumericType::Real,
        ));
        let b = CanonicalAstEncoder::encode(&linear_additive(
            "renamed",
            3,
            7,
            NumericType::Real,
        ));
        assert_eq!(a, b);
    }

    #[test]
    fn canonical_ast_preserves_implication_direction() {
        let forward = CanonicalAstEncoder::encode(&implication("x", false));
        let reverse = CanonicalAstEncoder::encode(&implication("x", true));
        assert_ne!(forward, reverse);
    }

    #[test]
    fn both_structural_retrievers_find_every_frozen_positive() {
        for case in q0_cases() {
            let score = score_case(&case);
            assert!(score.hdc_top1, "HDC missed {}", case.id);
            assert!(score.ast_top1, "canonical AST missed {}", case.id);
            assert!(
                score.hdc_positive > score.hdc_shuffled,
                "HDC shuffled control unexpectedly matched {}",
                case.id
            );
        }
    }
}
