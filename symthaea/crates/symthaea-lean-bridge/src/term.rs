// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Lean 4 FOL term builder.
//!
//! Target grammar (minimal subset for Phase 1):
//! ```text
//! term ::= ident
//!        | term term                    -- application
//!        | fun x => term                -- lambda
//!        | ∀ x, term                    -- forall
//!        | ∃ x, term                    -- exists
//!        | term → term                  -- implication
//!        | term ∧ term | term ∨ term | ¬ term | ⊤ | ⊥
//! ```
//!
//! Week-2 milestone: round-trip 1 fixture FOL proposition through this builder
//! and `lean4 --check` it.

/// Lean 4 term AST. Minimal Phase 1 surface.
#[derive(Debug, Clone, PartialEq)]
pub enum LeanTerm {
    /// Bare identifier (variable, constant, or defined lemma name).
    Ident(String),
    /// Function application: `f a`.
    App(Box<LeanTerm>, Box<LeanTerm>),
    /// Lambda abstraction: `fun x => body`.
    Lambda(String, Box<LeanTerm>),
    /// Universal quantifier: `∀ x, body`.
    Forall(String, Box<LeanTerm>),
    /// Existential quantifier: `∃ x, body`.
    Exists(String, Box<LeanTerm>),
    /// Implication: `a → b`.
    Implies(Box<LeanTerm>, Box<LeanTerm>),
    /// Conjunction: `a ∧ b`.
    And(Box<LeanTerm>, Box<LeanTerm>),
    /// Disjunction: `a ∨ b`.
    Or(Box<LeanTerm>, Box<LeanTerm>),
    /// Negation: `¬ a`.
    Not(Box<LeanTerm>),
    /// Constant `True`.
    True,
    /// Constant `False`.
    False,
}

impl LeanTerm {
    /// Pretty-print a term as Lean 4 source. Precedence-naive (fully parenthesized).
    pub fn to_lean(&self) -> String {
        match self {
            LeanTerm::Ident(s) => s.clone(),
            LeanTerm::App(f, a) => format!("({} {})", f.to_lean(), a.to_lean()),
            LeanTerm::Lambda(x, body) => format!("(fun {} => {})", x, body.to_lean()),
            LeanTerm::Forall(x, body) => format!("(∀ {}, {})", x, body.to_lean()),
            LeanTerm::Exists(x, body) => format!("(∃ {}, {})", x, body.to_lean()),
            LeanTerm::Implies(a, b) => format!("({} → {})", a.to_lean(), b.to_lean()),
            LeanTerm::And(a, b) => format!("({} ∧ {})", a.to_lean(), b.to_lean()),
            LeanTerm::Or(a, b) => format!("({} ∨ {})", a.to_lean(), b.to_lean()),
            LeanTerm::Not(a) => format!("(¬ {})", a.to_lean()),
            LeanTerm::True => "True".to_string(),
            LeanTerm::False => "False".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ident_prints_bare() {
        assert_eq!(LeanTerm::Ident("h".into()).to_lean(), "h");
    }

    #[test]
    fn forall_wraps_body() {
        let t = LeanTerm::Forall(
            "n".into(),
            Box::new(LeanTerm::Implies(
                Box::new(LeanTerm::Ident("Nat".into())),
                Box::new(LeanTerm::True),
            )),
        );
        assert_eq!(t.to_lean(), "(∀ n, (Nat → True))");
    }

    #[test]
    fn app_is_left_associative_naively() {
        let t = LeanTerm::App(
            Box::new(LeanTerm::App(
                Box::new(LeanTerm::Ident("f".into())),
                Box::new(LeanTerm::Ident("x".into())),
            )),
            Box::new(LeanTerm::Ident("y".into())),
        );
        assert_eq!(t.to_lean(), "((f x) y)");
    }
}
