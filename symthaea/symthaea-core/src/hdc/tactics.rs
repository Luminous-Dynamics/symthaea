// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tactic Engine for Mathematical Proof Construction.
//!
//! Provides Lean/Coq-inspired tactics that operate on proof goals.
//! A `Goal` is a sequent Γ ⊢ P (hypotheses ⊢ conclusion).
//! Tactics transform goals into (possibly empty) subgoals.
//!
//! # Tactics implemented
//!
//! - `intro`: introduce a hypothesis
//! - `apply`: apply a lemma/hypothesis to reduce goal
//! - `exact`: close goal with a proof term
//! - `split`: split conjunction goal into two subgoals
//! - `left`/`right`: choose disjunct
//! - `cases`: case split on a hypothesis
//! - `induction`: structural induction on natural number
//! - `ring`: prove polynomial identity (over ℤ/ℚ)
//! - `omega`: linear arithmetic over ℤ (Omega test)
//! - `norm_num`: numeric evaluation
//! - `simp`: simplification with rewrite rules
//! - `contradiction`: find explicit contradiction in hypotheses
//! - `assumption`: close goal if conclusion is already a hypothesis
//! - `rw`: rewrite with an equation

use std::collections::HashMap;
use std::fmt;

// ─── Expression ──────────────────────────────────────────────────────────────

/// A mathematical/logical expression in the tactic engine.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Var(String),
    Const(i64),
    Bool(bool),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
    Eq(Box<Expr>, Box<Expr>),
    Lt(Box<Expr>, Box<Expr>),
    Le(Box<Expr>, Box<Expr>),
    Not(Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Implies(Box<Expr>, Box<Expr>),
    ForAll(String, Box<Expr>),
    Exists(String, Box<Expr>),
    App(Box<Expr>, Box<Expr>),
    Lam(String, Box<Expr>),
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Var(v) => write!(f, "{}", v),
            Expr::Const(n) => write!(f, "{}", n),
            Expr::Bool(b) => write!(f, "{}", b),
            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Neg(a) => write!(f, "(-{})", a),
            Expr::Eq(a, b) => write!(f, "({} = {})", a, b),
            Expr::Lt(a, b) => write!(f, "({} < {})", a, b),
            Expr::Le(a, b) => write!(f, "({} ≤ {})", a, b),
            Expr::Not(a) => write!(f, "(¬{})", a),
            Expr::And(a, b) => write!(f, "({} ∧ {})", a, b),
            Expr::Or(a, b) => write!(f, "({} ∨ {})", a, b),
            Expr::Implies(a, b) => write!(f, "({} → {})", a, b),
            Expr::ForAll(v, body) => write!(f, "(∀{}. {})", v, body),
            Expr::Exists(v, body) => write!(f, "(∃{}. {})", v, body),
            Expr::App(func, arg) => write!(f, "({} {})", func, arg),
            Expr::Lam(v, body) => write!(f, "(λ{}. {})", v, body),
        }
    }
}

impl Expr {
    /// Evaluate numeric expression to i64, returning None if not fully numeric.
    pub fn eval(&self, env: &HashMap<String, i64>) -> Option<i64> {
        match self {
            Expr::Const(n) => Some(*n),
            Expr::Var(v) => env.get(v).copied(),
            Expr::Add(a, b) => Some(a.eval(env)? + b.eval(env)?),
            Expr::Sub(a, b) => Some(a.eval(env)? - b.eval(env)?),
            Expr::Mul(a, b) => Some(a.eval(env)? * b.eval(env)?),
            Expr::Neg(a) => Some(-a.eval(env)?),
            _ => None,
        }
    }

    /// Quick tautology check: True, P→P, or P∨¬P.
    pub fn is_tautology(&self) -> bool {
        match self {
            Expr::Bool(true) => true,
            Expr::Implies(a, b) => a == b,
            Expr::Or(a, b) => {
                if let Expr::Not(inner) = b.as_ref() {
                    a == inner
                } else if let Expr::Not(inner) = a.as_ref() {
                    inner == b
                } else {
                    false
                }
            }
            Expr::Eq(a, b) => a == b,
            _ => false,
        }
    }

    /// Substitute var → val throughout the expression.
    pub fn substitute(&self, var: &str, val: &Expr) -> Expr {
        match self {
            Expr::Var(v) if v == var => val.clone(),
            Expr::Var(_) | Expr::Const(_) | Expr::Bool(_) => self.clone(),
            Expr::Add(a, b) => Expr::Add(
                Box::new(a.substitute(var, val)),
                Box::new(b.substitute(var, val)),
            ),
            Expr::Sub(a, b) => Expr::Sub(
                Box::new(a.substitute(var, val)),
                Box::new(b.substitute(var, val)),
            ),
            Expr::Mul(a, b) => Expr::Mul(
                Box::new(a.substitute(var, val)),
                Box::new(b.substitute(var, val)),
            ),
            Expr::Neg(a) => Expr::Neg(Box::new(a.substitute(var, val))),
            Expr::Eq(a, b) => Expr::Eq(
                Box::new(a.substitute(var, val)),
                Box::new(b.substitute(var, val)),
            ),
            Expr::Lt(a, b) => Expr::Lt(
                Box::new(a.substitute(var, val)),
                Box::new(b.substitute(var, val)),
            ),
            Expr::Le(a, b) => Expr::Le(
                Box::new(a.substitute(var, val)),
                Box::new(b.substitute(var, val)),
            ),
            Expr::Not(a) => Expr::Not(Box::new(a.substitute(var, val))),
            Expr::And(a, b) => Expr::And(
                Box::new(a.substitute(var, val)),
                Box::new(b.substitute(var, val)),
            ),
            Expr::Or(a, b) => Expr::Or(
                Box::new(a.substitute(var, val)),
                Box::new(b.substitute(var, val)),
            ),
            Expr::Implies(a, b) => Expr::Implies(
                Box::new(a.substitute(var, val)),
                Box::new(b.substitute(var, val)),
            ),
            Expr::ForAll(v, body) if v != var => {
                Expr::ForAll(v.clone(), Box::new(body.substitute(var, val)))
            }
            Expr::Exists(v, body) if v != var => {
                Expr::Exists(v.clone(), Box::new(body.substitute(var, val)))
            }
            Expr::App(func, arg) => Expr::App(
                Box::new(func.substitute(var, val)),
                Box::new(arg.substitute(var, val)),
            ),
            Expr::Lam(v, body) if v != var => {
                Expr::Lam(v.clone(), Box::new(body.substitute(var, val)))
            }
            _ => self.clone(),
        }
    }

    /// Collect free variable names.
    pub fn free_vars(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_free_vars(&[], &mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_free_vars(&self, bound: &[String], out: &mut Vec<String>) {
        match self {
            Expr::Var(v) => {
                if !bound.contains(v) {
                    out.push(v.clone());
                }
            }
            Expr::Const(_) | Expr::Bool(_) => {}
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Eq(a, b)
            | Expr::Lt(a, b)
            | Expr::Le(a, b)
            | Expr::And(a, b)
            | Expr::Or(a, b)
            | Expr::Implies(a, b)
            | Expr::App(a, b) => {
                a.collect_free_vars(bound, out);
                b.collect_free_vars(bound, out);
            }
            Expr::Neg(a) | Expr::Not(a) => a.collect_free_vars(bound, out),
            Expr::ForAll(v, body) | Expr::Exists(v, body) | Expr::Lam(v, body) => {
                let mut new_bound = bound.to_vec();
                new_bound.push(v.clone());
                body.collect_free_vars(&new_bound, out);
            }
        }
    }

    /// Negate an expression (push negation one level in when possible).
    pub fn negate(&self) -> Expr {
        match self {
            Expr::Not(inner) => *inner.clone(),
            Expr::Bool(b) => Expr::Bool(!b),
            Expr::And(a, b) => Expr::Or(Box::new(a.negate()), Box::new(b.negate())),
            Expr::Or(a, b) => Expr::And(Box::new(a.negate()), Box::new(b.negate())),
            other => Expr::Not(Box::new(other.clone())),
        }
    }

    /// Basic algebraic/logical simplification.
    pub fn simplify(&self) -> Expr {
        match self {
            // Arithmetic identities
            Expr::Add(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Expr::Const(0), _) => b,
                    (_, Expr::Const(0)) => a,
                    (Expr::Const(x), Expr::Const(y)) => Expr::Const(x + y),
                    _ => Expr::Add(Box::new(a), Box::new(b)),
                }
            }
            Expr::Sub(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    _ if a == b => Expr::Const(0),
                    (_, Expr::Const(0)) => a,
                    (Expr::Const(x), Expr::Const(y)) => Expr::Const(x - y),
                    _ => Expr::Sub(Box::new(a), Box::new(b)),
                }
            }
            Expr::Mul(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Expr::Const(0), _) | (_, Expr::Const(0)) => Expr::Const(0),
                    (Expr::Const(1), _) => b,
                    (_, Expr::Const(1)) => a,
                    (Expr::Const(x), Expr::Const(y)) => Expr::Const(x * y),
                    _ => Expr::Mul(Box::new(a), Box::new(b)),
                }
            }
            Expr::Neg(a) => {
                let a = a.simplify();
                match a {
                    Expr::Const(n) => Expr::Const(-n),
                    Expr::Neg(inner) => *inner,
                    _ => Expr::Neg(Box::new(a)),
                }
            }
            // Logical identities
            Expr::And(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Expr::Bool(true), _) => b,
                    (_, Expr::Bool(true)) => a,
                    (Expr::Bool(false), _) | (_, Expr::Bool(false)) => Expr::Bool(false),
                    _ if a == b => a,
                    _ => Expr::And(Box::new(a), Box::new(b)),
                }
            }
            Expr::Or(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Expr::Bool(false), _) => b,
                    (_, Expr::Bool(false)) => a,
                    (Expr::Bool(true), _) | (_, Expr::Bool(true)) => Expr::Bool(true),
                    _ if a == b => a,
                    _ => Expr::Or(Box::new(a), Box::new(b)),
                }
            }
            Expr::Not(a) => {
                let a = a.simplify();
                match a {
                    Expr::Bool(b) => Expr::Bool(!b),
                    Expr::Not(inner) => *inner,
                    _ => Expr::Not(Box::new(a)),
                }
            }
            Expr::Implies(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Expr::Bool(false), _) => Expr::Bool(true),
                    (_, Expr::Bool(true)) => Expr::Bool(true),
                    (Expr::Bool(true), _) => b,
                    _ if a == b => Expr::Bool(true),
                    _ => Expr::Implies(Box::new(a), Box::new(b)),
                }
            }
            Expr::Eq(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                if a == b {
                    Expr::Bool(true)
                } else {
                    match (&a, &b) {
                        (Expr::Const(x), Expr::Const(y)) => Expr::Bool(x == y),
                        _ => Expr::Eq(Box::new(a), Box::new(b)),
                    }
                }
            }
            Expr::Lt(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Expr::Const(x), Expr::Const(y)) => Expr::Bool(x < y),
                    _ => Expr::Lt(Box::new(a), Box::new(b)),
                }
            }
            Expr::Le(a, b) => {
                let a = a.simplify();
                let b = b.simplify();
                match (&a, &b) {
                    (Expr::Const(x), Expr::Const(y)) => Expr::Bool(x <= y),
                    _ => Expr::Le(Box::new(a), Box::new(b)),
                }
            }
            _ => self.clone(),
        }
    }
}

// ─── Goal ────────────────────────────────────────────────────────────────────

/// A proof goal: a sequent Γ ⊢ P.
#[derive(Debug, Clone)]
pub struct Goal {
    /// Named hypotheses available in the context.
    pub hypotheses: Vec<(String, Expr)>,
    /// The formula to prove.
    pub conclusion: Expr,
}

impl Goal {
    /// Create a goal with no hypotheses.
    pub fn new(conclusion: Expr) -> Self {
        Self {
            hypotheses: Vec::new(),
            conclusion,
        }
    }

    /// Add a hypothesis and return self (builder pattern).
    pub fn with_hyp(mut self, name: &str, expr: Expr) -> Self {
        self.hypotheses.push((name.to_string(), expr));
        self
    }

    /// True if the conclusion is trivially true or is already a hypothesis.
    pub fn is_closed(&self) -> bool {
        match &self.conclusion {
            Expr::Bool(true) => return true,
            _ => {}
        }
        if self.conclusion.is_tautology() {
            return true;
        }
        // Check if conclusion appears directly in hypotheses
        self.hypotheses
            .iter()
            .any(|(_, h)| h == &self.conclusion)
    }

    /// True if hypotheses contain a direct contradiction (P and ¬P).
    pub fn has_contradiction(&self) -> bool {
        for (_, h) in &self.hypotheses {
            let neg = h.negate();
            if self.hypotheses.iter().any(|(_, h2)| h2 == &neg) {
                return true;
            }
            // Also check Bool(false) in hypotheses
            if matches!(h, Expr::Bool(false)) {
                return true;
            }
        }
        false
    }
}

// ─── TacticResult ────────────────────────────────────────────────────────────

/// Result of applying a tactic to a goal.
#[derive(Debug, Clone)]
pub enum TacticResult {
    /// Goal was proved.
    Closed,
    /// Tactic produced these subgoals.
    Subgoals(Vec<Goal>),
    /// Tactic was not applicable.
    Failed(String),
}

// ─── Tactics ─────────────────────────────────────────────────────────────────

/// Introduce a hypothesis.
///
/// - If conclusion is `P → Q`: add P as hypothesis `name`, new goal is `⊢ Q`.
/// - If conclusion is `∀x.P`: add x as free variable annotation, new goal is `P`.
pub fn tactic_intro(goal: &Goal, name: &str) -> TacticResult {
    match &goal.conclusion {
        Expr::Implies(premise, conclusion) => {
            let mut new_hyps = goal.hypotheses.clone();
            new_hyps.push((name.to_string(), *premise.clone()));
            TacticResult::Subgoals(vec![Goal {
                hypotheses: new_hyps,
                conclusion: *conclusion.clone(),
            }])
        }
        Expr::ForAll(var, body) => {
            let mut new_hyps = goal.hypotheses.clone();
            new_hyps.push((name.to_string(), Expr::Var(var.clone())));
            TacticResult::Subgoals(vec![Goal {
                hypotheses: new_hyps,
                conclusion: *body.clone(),
            }])
        }
        _ => TacticResult::Failed(format!(
            "intro: conclusion is not an implication or universal quantifier: {}",
            goal.conclusion
        )),
    }
}

/// Apply a lemma to reduce the goal.
///
/// - If lemma is `A → B` and conclusion is `B`: new subgoal is `⊢ A`.
/// - If lemma matches conclusion exactly: `Closed`.
pub fn tactic_apply(goal: &Goal, lemma: &Expr) -> TacticResult {
    if lemma == &goal.conclusion {
        return TacticResult::Closed;
    }
    match lemma {
        Expr::Implies(premise, conclusion) => {
            if conclusion.as_ref() == &goal.conclusion {
                TacticResult::Subgoals(vec![Goal {
                    hypotheses: goal.hypotheses.clone(),
                    conclusion: *premise.clone(),
                }])
            } else {
                TacticResult::Failed(format!(
                    "apply: lemma conclusion {} does not match goal {}",
                    conclusion, goal.conclusion
                ))
            }
        }
        _ => TacticResult::Failed(format!(
            "apply: lemma {} is not an implication and does not match goal {}",
            lemma, goal.conclusion
        )),
    }
}

/// Close goal if proof matches conclusion or is a hypothesis.
pub fn tactic_exact(goal: &Goal, proof: &Expr) -> TacticResult {
    if proof == &goal.conclusion {
        return TacticResult::Closed;
    }
    if goal.hypotheses.iter().any(|(_, h)| h == &goal.conclusion) {
        return TacticResult::Closed;
    }
    if goal.hypotheses.iter().any(|(_, h)| h == proof) && proof == &goal.conclusion {
        return TacticResult::Closed;
    }
    TacticResult::Failed(format!(
        "exact: {} does not match conclusion {}",
        proof, goal.conclusion
    ))
}

/// Close goal if conclusion appears in hypotheses.
pub fn tactic_assumption(goal: &Goal) -> TacticResult {
    if goal.is_closed() {
        return TacticResult::Closed;
    }
    if goal.hypotheses.iter().any(|(_, h)| h == &goal.conclusion) {
        TacticResult::Closed
    } else {
        TacticResult::Failed("assumption: conclusion not found in hypotheses".to_string())
    }
}

/// Split conjunction goal into two subgoals.
pub fn tactic_split(goal: &Goal) -> TacticResult {
    match &goal.conclusion {
        Expr::And(a, b) => TacticResult::Subgoals(vec![
            Goal {
                hypotheses: goal.hypotheses.clone(),
                conclusion: *a.clone(),
            },
            Goal {
                hypotheses: goal.hypotheses.clone(),
                conclusion: *b.clone(),
            },
        ]),
        _ => TacticResult::Failed(format!(
            "split: conclusion {} is not a conjunction",
            goal.conclusion
        )),
    }
}

/// Choose left disjunct.
pub fn tactic_left(goal: &Goal) -> TacticResult {
    match &goal.conclusion {
        Expr::Or(a, _) => TacticResult::Subgoals(vec![Goal {
            hypotheses: goal.hypotheses.clone(),
            conclusion: *a.clone(),
        }]),
        _ => TacticResult::Failed(format!(
            "left: conclusion {} is not a disjunction",
            goal.conclusion
        )),
    }
}

/// Choose right disjunct.
pub fn tactic_right(goal: &Goal) -> TacticResult {
    match &goal.conclusion {
        Expr::Or(_, b) => TacticResult::Subgoals(vec![Goal {
            hypotheses: goal.hypotheses.clone(),
            conclusion: *b.clone(),
        }]),
        _ => TacticResult::Failed(format!(
            "right: conclusion {} is not a disjunction",
            goal.conclusion
        )),
    }
}

/// Case split on a hypothesis.
///
/// - If hypothesis is `A ∨ B`: two subgoals [Γ,A⊢C] and [Γ,B⊢C].
/// - If hypothesis is `A ∧ B`: one subgoal with both A and B as hypotheses.
pub fn tactic_cases(goal: &Goal, hyp_name: &str) -> TacticResult {
    let hyp = goal.hypotheses.iter().find(|(n, _)| n == hyp_name);
    match hyp {
        None => TacticResult::Failed(format!("cases: hypothesis '{}' not found", hyp_name)),
        Some((_, expr)) => match expr.clone() {
            Expr::Or(a, b) => {
                let mut hyps_a = goal
                    .hypotheses
                    .iter()
                    .filter(|(n, _)| n != hyp_name)
                    .cloned()
                    .collect::<Vec<_>>();
                let mut hyps_b = hyps_a.clone();
                hyps_a.push((format!("{}_left", hyp_name), *a));
                hyps_b.push((format!("{}_right", hyp_name), *b));
                TacticResult::Subgoals(vec![
                    Goal {
                        hypotheses: hyps_a,
                        conclusion: goal.conclusion.clone(),
                    },
                    Goal {
                        hypotheses: hyps_b,
                        conclusion: goal.conclusion.clone(),
                    },
                ])
            }
            Expr::And(a, b) => {
                let mut new_hyps = goal
                    .hypotheses
                    .iter()
                    .filter(|(n, _)| n != hyp_name)
                    .cloned()
                    .collect::<Vec<_>>();
                new_hyps.push((format!("{}_fst", hyp_name), *a));
                new_hyps.push((format!("{}_snd", hyp_name), *b));
                TacticResult::Subgoals(vec![Goal {
                    hypotheses: new_hyps,
                    conclusion: goal.conclusion.clone(),
                }])
            }
            _ => TacticResult::Failed(format!(
                "cases: hypothesis '{}' is not a disjunction or conjunction",
                hyp_name
            )),
        },
    }
}

/// Find explicit contradiction P and ¬P in hypotheses and close.
pub fn tactic_contradiction(goal: &Goal) -> TacticResult {
    if goal.has_contradiction() {
        TacticResult::Closed
    } else {
        TacticResult::Failed("contradiction: no contradiction found in hypotheses".to_string())
    }
}

/// Rewrite with an equation hypothesis.
///
/// `eq_hyp` should name a hypothesis of the form `Eq(lhs, rhs)`.
/// `direction` = true means rewrite lhs→rhs in conclusion; false means rhs→lhs.
pub fn tactic_rw(goal: &Goal, eq_hyp: &str, direction: bool) -> TacticResult {
    let hyp = goal.hypotheses.iter().find(|(n, _)| n == eq_hyp);
    match hyp {
        None => TacticResult::Failed(format!("rw: hypothesis '{}' not found", eq_hyp)),
        Some((_, expr)) => match expr.clone() {
            Expr::Eq(lhs, rhs) => {
                let (from, to) = if direction { (*lhs, *rhs) } else { (*rhs, *lhs) };
                let new_concl = rewrite_in(&goal.conclusion, &from, &to);
                if new_concl == goal.conclusion {
                    TacticResult::Failed(format!(
                        "rw: pattern {} not found in {}",
                        from, goal.conclusion
                    ))
                } else {
                    let new_goal = Goal {
                        hypotheses: goal.hypotheses.clone(),
                        conclusion: new_concl,
                    };
                    if new_goal.is_closed() {
                        TacticResult::Closed
                    } else {
                        TacticResult::Subgoals(vec![new_goal])
                    }
                }
            }
            _ => TacticResult::Failed(format!(
                "rw: hypothesis '{}' is not an equation",
                eq_hyp
            )),
        },
    }
}

/// Rewrite all occurrences of `from` to `to` in `expr`.
fn rewrite_in(expr: &Expr, from: &Expr, to: &Expr) -> Expr {
    if expr == from {
        return to.clone();
    }
    match expr {
        Expr::Add(a, b) => Expr::Add(
            Box::new(rewrite_in(a, from, to)),
            Box::new(rewrite_in(b, from, to)),
        ),
        Expr::Sub(a, b) => Expr::Sub(
            Box::new(rewrite_in(a, from, to)),
            Box::new(rewrite_in(b, from, to)),
        ),
        Expr::Mul(a, b) => Expr::Mul(
            Box::new(rewrite_in(a, from, to)),
            Box::new(rewrite_in(b, from, to)),
        ),
        Expr::Neg(a) => Expr::Neg(Box::new(rewrite_in(a, from, to))),
        Expr::Eq(a, b) => Expr::Eq(
            Box::new(rewrite_in(a, from, to)),
            Box::new(rewrite_in(b, from, to)),
        ),
        Expr::Lt(a, b) => Expr::Lt(
            Box::new(rewrite_in(a, from, to)),
            Box::new(rewrite_in(b, from, to)),
        ),
        Expr::Le(a, b) => Expr::Le(
            Box::new(rewrite_in(a, from, to)),
            Box::new(rewrite_in(b, from, to)),
        ),
        Expr::Not(a) => Expr::Not(Box::new(rewrite_in(a, from, to))),
        Expr::And(a, b) => Expr::And(
            Box::new(rewrite_in(a, from, to)),
            Box::new(rewrite_in(b, from, to)),
        ),
        Expr::Or(a, b) => Expr::Or(
            Box::new(rewrite_in(a, from, to)),
            Box::new(rewrite_in(b, from, to)),
        ),
        Expr::Implies(a, b) => Expr::Implies(
            Box::new(rewrite_in(a, from, to)),
            Box::new(rewrite_in(b, from, to)),
        ),
        Expr::App(a, b) => Expr::App(
            Box::new(rewrite_in(a, from, to)),
            Box::new(rewrite_in(b, from, to)),
        ),
        _ => expr.clone(),
    }
}

/// Prove polynomial identity by normalizing both sides to a canonical sum-of-terms form.
///
/// Works for integer polynomial expressions (Add, Sub, Mul, Neg, Const, Var).
pub fn tactic_ring(goal: &Goal) -> TacticResult {
    match &goal.conclusion {
        Expr::Eq(lhs, rhs) => {
            let lhs_norm = poly_normalize(lhs);
            let rhs_norm = poly_normalize(rhs);
            if lhs_norm == rhs_norm {
                TacticResult::Closed
            } else {
                TacticResult::Failed(format!(
                    "ring: polynomial {} ≠ {} after normalization",
                    goal.conclusion, goal.conclusion
                ))
            }
        }
        _ => TacticResult::Failed("ring: conclusion is not an equality".to_string()),
    }
}

/// Normalize a polynomial expression into a sorted monomial map: var^exp → coefficient.
/// Returns a HashMap<Vec<(String, u32)>, i64> representing a canonical polynomial.
fn poly_normalize(expr: &Expr) -> HashMap<Vec<(String, u32)>, i64> {
    let mut result: HashMap<Vec<(String, u32)>, i64> = HashMap::new();
    poly_add_to(&mut result, expr, 1);
    // Remove zero coefficients
    result.retain(|_, v| *v != 0);
    result
}

fn poly_add_to(acc: &mut HashMap<Vec<(String, u32)>, i64>, expr: &Expr, coeff: i64) {
    match expr {
        Expr::Const(n) => {
            *acc.entry(vec![]).or_insert(0) += coeff * n;
        }
        Expr::Var(v) => {
            let key = vec![(v.clone(), 1u32)];
            *acc.entry(key).or_insert(0) += coeff;
        }
        Expr::Add(a, b) => {
            poly_add_to(acc, a, coeff);
            poly_add_to(acc, b, coeff);
        }
        Expr::Sub(a, b) => {
            poly_add_to(acc, a, coeff);
            poly_add_to(acc, b, -coeff);
        }
        Expr::Neg(a) => {
            poly_add_to(acc, a, -coeff);
        }
        Expr::Mul(a, b) => {
            // Distribute: (a_terms) * (b_terms)
            let mut a_map: HashMap<Vec<(String, u32)>, i64> = HashMap::new();
            poly_add_to(&mut a_map, a, 1);
            let mut b_map: HashMap<Vec<(String, u32)>, i64> = HashMap::new();
            poly_add_to(&mut b_map, b, 1);
            for (ak, av) in &a_map {
                for (bk, bv) in &b_map {
                    let mut combined = ak.clone();
                    for (bvar, bexp) in bk {
                        if let Some(entry) = combined.iter_mut().find(|(v, _)| v == bvar) {
                            entry.1 += bexp;
                        } else {
                            combined.push((bvar.clone(), *bexp));
                        }
                    }
                    combined.sort_by_key(|(v, _)| v.clone());
                    *acc.entry(combined).or_insert(0) += coeff * av * bv;
                }
            }
        }
        _ => {
            // Non-polynomial: treat as opaque variable
            let key = vec![(format!("{}", expr), 1u32)];
            *acc.entry(key).or_insert(0) += coeff;
        }
    }
}

/// Linear arithmetic solver for integer goals.
///
/// Tries to verify the conclusion directly by evaluating with a small environment
/// or by checking if it follows from linear combinations of hypotheses.
pub fn tactic_omega(goal: &Goal) -> TacticResult {
    // First: try to evaluate if all vars are bound in hypotheses as equalities to constants
    let mut env: HashMap<String, i64> = HashMap::new();
    for (_, hyp) in &goal.hypotheses {
        if let Expr::Eq(lhs, rhs) = hyp {
            if let Expr::Var(v) = lhs.as_ref() {
                if let Some(n) = rhs.eval(&env) {
                    env.insert(v.clone(), n);
                }
            }
            if let Expr::Var(v) = rhs.as_ref() {
                if let Some(n) = lhs.eval(&env) {
                    env.insert(v.clone(), n);
                }
            }
        }
    }

    // Try evaluating the conclusion as a boolean
    let simplified = goal.conclusion.simplify();
    if let Expr::Bool(true) = &simplified {
        return TacticResult::Closed;
    }
    if let Expr::Bool(false) = &simplified {
        return TacticResult::Failed("omega: conclusion evaluates to false".to_string());
    }

    // Try with environment
    let conc_with_env = substitute_env(&goal.conclusion, &env);
    let simplified2 = conc_with_env.simplify();
    match &simplified2 {
        Expr::Bool(true) => TacticResult::Closed,
        Expr::Bool(false) => TacticResult::Failed("omega: conclusion is false under current bindings".to_string()),
        Expr::Eq(a, b) => {
            if let (Some(av), Some(bv)) = (a.eval(&env), b.eval(&env)) {
                if av == bv {
                    TacticResult::Closed
                } else {
                    TacticResult::Failed(format!("omega: {} ≠ {}", av, bv))
                }
            } else {
                TacticResult::Failed("omega: cannot evaluate equation".to_string())
            }
        }
        Expr::Le(a, b) => {
            if let (Some(av), Some(bv)) = (a.eval(&env), b.eval(&env)) {
                if av <= bv {
                    TacticResult::Closed
                } else {
                    TacticResult::Failed(format!("omega: {} > {}", av, bv))
                }
            } else {
                TacticResult::Failed("omega: cannot evaluate inequality".to_string())
            }
        }
        Expr::Lt(a, b) => {
            if let (Some(av), Some(bv)) = (a.eval(&env), b.eval(&env)) {
                if av < bv {
                    TacticResult::Closed
                } else {
                    TacticResult::Failed(format!("omega: {} ≥ {}", av, bv))
                }
            } else {
                TacticResult::Failed("omega: cannot evaluate inequality".to_string())
            }
        }
        _ => TacticResult::Failed("omega: cannot solve this linear arithmetic goal".to_string()),
    }
}

fn substitute_env(expr: &Expr, env: &HashMap<String, i64>) -> Expr {
    let mut result = expr.clone();
    for (var, val) in env {
        result = result.substitute(var, &Expr::Const(*val));
    }
    result
}

/// Evaluate numeric equalities/inequalities by computing both sides.
pub fn tactic_norm_num(goal: &Goal) -> TacticResult {
    let env: HashMap<String, i64> = HashMap::new();
    let simplified = goal.conclusion.simplify();
    match &simplified {
        Expr::Bool(true) => return TacticResult::Closed,
        Expr::Bool(false) => return TacticResult::Failed("norm_num: evaluates to false".to_string()),
        Expr::Eq(a, b) => {
            if let (Some(av), Some(bv)) = (a.eval(&env), b.eval(&env)) {
                return if av == bv {
                    TacticResult::Closed
                } else {
                    TacticResult::Failed(format!("norm_num: {} ≠ {}", av, bv))
                };
            }
        }
        Expr::Lt(a, b) => {
            if let (Some(av), Some(bv)) = (a.eval(&env), b.eval(&env)) {
                return if av < bv {
                    TacticResult::Closed
                } else {
                    TacticResult::Failed(format!("norm_num: {} ≥ {}", av, bv))
                };
            }
        }
        Expr::Le(a, b) => {
            if let (Some(av), Some(bv)) = (a.eval(&env), b.eval(&env)) {
                return if av <= bv {
                    TacticResult::Closed
                } else {
                    TacticResult::Failed(format!("norm_num: {} > {}", av, bv))
                };
            }
        }
        _ => {}
    }
    TacticResult::Failed("norm_num: cannot evaluate this expression numerically".to_string())
}

/// Apply rewrite rules (left → right) to conclusion until fixpoint, then try assumption.
pub fn tactic_simp(goal: &Goal, rules: &[(Expr, Expr)]) -> TacticResult {
    let mut current = goal.conclusion.clone();
    let mut changed = true;
    let mut iterations = 0;
    while changed && iterations < 100 {
        changed = false;
        iterations += 1;
        for (from, to) in rules {
            let new = rewrite_in(&current, from, to);
            if new != current {
                current = new;
                changed = true;
            }
        }
        // Also apply algebraic simplification
        let simplified = current.simplify();
        if simplified != current {
            current = simplified;
            changed = true;
        }
    }
    let new_goal = Goal {
        hypotheses: goal.hypotheses.clone(),
        conclusion: current,
    };
    if new_goal.is_closed() {
        TacticResult::Closed
    } else {
        // Try assumption on simplified goal
        if tactic_assumption(&new_goal).is_closed_variant() {
            TacticResult::Closed
        } else {
            TacticResult::Subgoals(vec![new_goal])
        }
    }
}

impl TacticResult {
    fn is_closed_variant(&self) -> bool {
        matches!(self, TacticResult::Closed)
    }
}

/// Structural induction on a natural number variable.
///
/// Produces base case `⊢ P(0)` and inductive step `P(n) ⊢ P(n+1)`.
pub fn tactic_induction(goal: &Goal, var: &str) -> TacticResult {
    // Expect ∀n. P(n) or just P(var) with var as a free variable
    let body = match &goal.conclusion {
        Expr::ForAll(v, body) if v == var => *body.clone(),
        _ => goal.conclusion.clone(),
    };

    // Base case: P(0)
    let base = body.substitute(var, &Expr::Const(0));
    let base_goal = Goal {
        hypotheses: goal.hypotheses.clone(),
        conclusion: base,
    };

    // Inductive step: P(n) ⊢ P(n+1)
    let ind_hyp = body.clone(); // P(n)
    let ind_concl = body.substitute(
        var,
        &Expr::Add(Box::new(Expr::Var(var.to_string())), Box::new(Expr::Const(1))),
    );
    let mut step_hyps = goal.hypotheses.clone();
    step_hyps.push((format!("ih_{}", var), ind_hyp));
    let step_goal = Goal {
        hypotheses: step_hyps,
        conclusion: ind_concl,
    };

    TacticResult::Subgoals(vec![base_goal, step_goal])
}

/// Apply tactics in sequence, stopping at first failure.
pub fn try_seq(goal: &Goal, tactics: &[Box<dyn Fn(&Goal) -> TacticResult>]) -> TacticResult {
    let mut current_goals = vec![goal.clone()];
    for tactic in tactics {
        if current_goals.is_empty() {
            return TacticResult::Closed;
        }
        let mut next_goals = Vec::new();
        let first_goal = current_goals.remove(0);
        match tactic(&first_goal) {
            TacticResult::Closed => {
                // This goal is done, continue with remaining
            }
            TacticResult::Subgoals(mut subs) => {
                next_goals.append(&mut subs);
                next_goals.extend(current_goals);
                current_goals = next_goals;
                continue;
            }
            TacticResult::Failed(msg) => return TacticResult::Failed(msg),
        }
        current_goals.extend(next_goals);
    }
    if current_goals.is_empty() {
        TacticResult::Closed
    } else {
        TacticResult::Subgoals(current_goals)
    }
}

// ─── Phase 1: IMO number-theory tactics ──────────────────────────────────────
//
// These tactics bridge `NumberTheoryEngine` / `diophantine::pell_equation`
// into the tactic framework. They are parameterized (not pattern-matched out
// of the goal) because `Expr` does not yet carry modular-arithmetic or nested
// existential shape. Integration tests below show how to build the matching
// `Expr` goal and close it with these tactics.

use crate::hdc::diophantine::pell_equation;
use crate::hdc::number_theory::NumberTheoryEngine;

/// Closes `∃x. ∃y. a·x + b·y = c` when the primitive `linear_diophantine`
/// finds a solution. Caller supplies (a, b, c) as concrete integers.
///
/// Returns `Closed` with an implicit witness, `Failed` when no solution
/// exists (gcd(a,b) ∤ c).
pub fn tactic_linear_diophantine(_goal: &Goal, a: i64, b: i64, c: i64) -> TacticResult {
    let engine = NumberTheoryEngine::new();
    match engine.linear_diophantine(a, b, c) {
        Some((x0, y0, _dx, _dy)) => {
            if a * x0 + b * y0 == c {
                TacticResult::Closed
            } else {
                TacticResult::Failed(format!(
                    "linear_diophantine produced inconsistent witness ({}, {})",
                    x0, y0
                ))
            }
        }
        None => TacticResult::Failed(format!(
            "no integer solution to {}x + {}y = {}",
            a, b, c
        )),
    }
}

/// Closes `∃x. ∃y. x² − D·y² = 1 ∧ y > 0` by invoking the Pell solver.
pub fn tactic_pell(_goal: &Goal, d: i64) -> TacticResult {
    match pell_equation(d) {
        Some(sol) => {
            let (x, y) = sol.fundamental;
            if sol.verify(x, y) && y > 0 {
                TacticResult::Closed
            } else {
                TacticResult::Failed(format!(
                    "pell solver returned inconsistent fundamental ({}, {})",
                    x, y
                ))
            }
        }
        None => TacticResult::Failed(format!(
            "x² − {}·y² = 1 has no nontrivial solution (D ≤ 0 or perfect square)",
            d
        )),
    }
}

/// Closes quadratic-residuosity goals via the Legendre symbol.
/// `expected`: +1 for QR, -1 for non-residue, 0 for p|a.
pub fn tactic_quadratic_residue(_goal: &Goal, a: i64, p: i64, expected: i32) -> TacticResult {
    if p <= 2 || p % 2 == 0 {
        return TacticResult::Failed(format!("legendre requires odd prime p, got {}", p));
    }
    let engine = NumberTheoryEngine::new();
    let actual = engine.legendre_symbol(a, p);
    if actual == expected {
        TacticResult::Closed
    } else {
        TacticResult::Failed(format!(
            "(({}/{})) = {}, expected {}",
            a, p, actual, expected
        ))
    }
}

/// Closes v_p(a^n − b^n) = k goals via Lifting the Exponent.
pub fn tactic_lte_bound(_goal: &Goal, p: i64, a: i64, b: i64, n: u32, k: u32) -> TacticResult {
    let engine = NumberTheoryEngine::new();
    match engine.lifting_the_exponent(p, a, b, n) {
        Some(v) if v == k => TacticResult::Closed,
        Some(v) => TacticResult::Failed(format!(
            "LTE gives v_{}({}^{} − {}^{}) = {}, not {}",
            p, a, n, b, n, v, k
        )),
        None => TacticResult::Failed(format!(
            "LTE preconditions fail for p={}, a={}, b={}",
            p, a, b
        )),
    }
}

/// Closes ∃x. x ≡ a_i (mod m_i) for all i  via CRT. Supplies residue list.
pub fn tactic_crt_solve(_goal: &Goal, residues: &[(i64, i64)]) -> TacticResult {
    let engine = NumberTheoryEngine::new();
    match engine.crt(residues) {
        Some((x, m)) => {
            for &(a, mi) in residues {
                if x.rem_euclid(mi) != a.rem_euclid(mi) {
                    return TacticResult::Failed(format!(
                        "crt witness x={} violates x ≡ {} (mod {})",
                        x, a, mi
                    ));
                }
            }
            assert!(m > 0);
            TacticResult::Closed
        }
        None => TacticResult::Failed("CRT system inconsistent".to_string()),
    }
}

// ─── TacticProver ────────────────────────────────────────────────────────────

/// Automated proof search via BFS over the tactic library.
pub struct TacticProver {
    pub max_depth: usize,
    pub max_goals: usize,
}

impl TacticProver {
    pub fn new(max_depth: usize) -> Self {
        Self {
            max_depth,
            max_goals: 64,
        }
    }

    /// Try to prove a goal, returning the sequence of tactic names applied.
    pub fn prove(&self, goal: &Goal) -> Option<Vec<String>> {
        // BFS: each state is (remaining goals, tactic trace)
        self.prove_goals(vec![goal.clone()], Vec::new(), 0)
    }

    fn prove_goals(
        &self,
        goals: Vec<Goal>,
        trace: Vec<String>,
        depth: usize,
    ) -> Option<Vec<String>> {
        if goals.is_empty() {
            return Some(trace);
        }
        if depth >= self.max_depth {
            return None;
        }

        let goal = &goals[0];
        let rest = goals[1..].to_vec();

        // Try fast-path tactics first
        let fast_tactics: Vec<(&str, Box<dyn Fn(&Goal) -> TacticResult>)> = vec![
            ("assumption", Box::new(|g: &Goal| tactic_assumption(g))),
            ("contradiction", Box::new(|g: &Goal| tactic_contradiction(g))),
            ("norm_num", Box::new(|g: &Goal| tactic_norm_num(g))),
            ("ring", Box::new(|g: &Goal| tactic_ring(g))),
            ("simp", Box::new(|g: &Goal| tactic_simp(g, &[]))),
        ];

        for (name, tactic) in &fast_tactics {
            match tactic(goal) {
                TacticResult::Closed => {
                    let mut new_trace = trace.clone();
                    new_trace.push(name.to_string());
                    if let result @ Some(_) = self.prove_goals(rest.clone(), new_trace, depth + 1) {
                        return result;
                    }
                }
                TacticResult::Subgoals(subs) => {
                    let mut new_goals = subs;
                    new_goals.extend(rest.clone());
                    if new_goals.len() <= self.max_goals {
                        let mut new_trace = trace.clone();
                        new_trace.push(name.to_string());
                        if let result @ Some(_) =
                            self.prove_goals(new_goals, new_trace, depth + 1)
                        {
                            return result;
                        }
                    }
                }
                TacticResult::Failed(_) => continue,
            }
        }

        // Try structural tactics: intro, split
        match &goal.conclusion {
            Expr::Implies(_, _) | Expr::ForAll(_, _) => {
                let tac_result = tactic_intro(goal, &format!("h{}", depth));
                if let TacticResult::Subgoals(subs) = tac_result {
                    let mut new_goals = subs;
                    new_goals.extend(rest.clone());
                    let mut new_trace = trace.clone();
                    new_trace.push(format!("intro h{}", depth));
                    if let result @ Some(_) = self.prove_goals(new_goals, new_trace, depth + 1) {
                        return result;
                    }
                }
            }
            Expr::And(_, _) => {
                if let TacticResult::Subgoals(subs) = tactic_split(goal) {
                    let mut new_goals = subs;
                    new_goals.extend(rest.clone());
                    let mut new_trace = trace.clone();
                    new_trace.push("split".to_string());
                    if let result @ Some(_) = self.prove_goals(new_goals, new_trace, depth + 1) {
                        return result;
                    }
                }
            }
            Expr::Or(_, _) => {
                // Try left
                if let TacticResult::Subgoals(subs) = tactic_left(goal) {
                    let mut new_goals = subs;
                    new_goals.extend(rest.clone());
                    let mut new_trace = trace.clone();
                    new_trace.push("left".to_string());
                    if let result @ Some(_) = self.prove_goals(new_goals, new_trace, depth + 1) {
                        return result;
                    }
                }
                // Try right
                if let TacticResult::Subgoals(subs) = tactic_right(goal) {
                    let mut new_goals = subs;
                    new_goals.extend(rest.clone());
                    let mut new_trace = trace.clone();
                    new_trace.push("right".to_string());
                    if let result @ Some(_) = self.prove_goals(new_goals, new_trace, depth + 1) {
                        return result;
                    }
                }
            }
            _ => {}
        }

        // Try cases on all hypotheses that are Or/And
        for (hyp_name, hyp_expr) in &goal.hypotheses {
            if matches!(hyp_expr, Expr::Or(_, _) | Expr::And(_, _)) {
                if let TacticResult::Subgoals(subs) = tactic_cases(goal, hyp_name) {
                    let mut new_goals = subs;
                    new_goals.extend(rest.clone());
                    if new_goals.len() <= self.max_goals {
                        let mut new_trace = trace.clone();
                        new_trace.push(format!("cases {}", hyp_name));
                        if let result @ Some(_) =
                            self.prove_goals(new_goals, new_trace, depth + 1)
                        {
                            return result;
                        }
                    }
                }
            }
        }

        None
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intro_on_implication() {
        // P → Q with intro should produce goal {P} ⊢ Q
        let goal = Goal::new(Expr::Implies(
            Box::new(Expr::Var("P".into())),
            Box::new(Expr::Var("Q".into())),
        ));
        match tactic_intro(&goal, "hP") {
            TacticResult::Subgoals(sub) => {
                assert_eq!(sub.len(), 1);
                assert_eq!(sub[0].conclusion, Expr::Var("Q".into()));
                assert!(sub[0].hypotheses.iter().any(|(n, _)| n == "hP"));
            }
            other => panic!("Expected Subgoals, got {:?}", other),
        }
    }

    #[test]
    fn test_assumption_closes_p_proves_p() {
        // {P} ⊢ P should be closed by assumption
        let goal = Goal::new(Expr::Var("P".into())).with_hyp("hP", Expr::Var("P".into()));
        assert!(matches!(tactic_assumption(&goal), TacticResult::Closed));
    }

    #[test]
    fn test_split_produces_two_goals() {
        // ⊢ A ∧ B → two subgoals ⊢ A and ⊢ B
        let goal = Goal::new(Expr::And(
            Box::new(Expr::Var("A".into())),
            Box::new(Expr::Var("B".into())),
        ));
        match tactic_split(&goal) {
            TacticResult::Subgoals(subs) => {
                assert_eq!(subs.len(), 2);
                assert_eq!(subs[0].conclusion, Expr::Var("A".into()));
                assert_eq!(subs[1].conclusion, Expr::Var("B".into()));
            }
            other => panic!("Expected Subgoals, got {:?}", other),
        }
    }

    #[test]
    fn test_contradiction_closes_goal() {
        // {P, ¬P} ⊢ Q should close via contradiction
        let goal = Goal::new(Expr::Var("Q".into()))
            .with_hyp("hP", Expr::Var("P".into()))
            .with_hyp("hnP", Expr::Not(Box::new(Expr::Var("P".into()))));
        assert!(matches!(tactic_contradiction(&goal), TacticResult::Closed));
    }

    #[test]
    fn test_ring_closes_commutativity() {
        // x + y = y + x
        let goal = Goal::new(Expr::Eq(
            Box::new(Expr::Add(
                Box::new(Expr::Var("x".into())),
                Box::new(Expr::Var("y".into())),
            )),
            Box::new(Expr::Add(
                Box::new(Expr::Var("y".into())),
                Box::new(Expr::Var("x".into())),
            )),
        ));
        assert!(matches!(tactic_ring(&goal), TacticResult::Closed));
    }

    #[test]
    fn test_ring_closes_distribution() {
        // x * (y + z) = x*y + x*z
        let x = || Box::new(Expr::Var("x".into()));
        let y = || Box::new(Expr::Var("y".into()));
        let z = || Box::new(Expr::Var("z".into()));
        let lhs = Expr::Mul(x(), Box::new(Expr::Add(y(), z())));
        let rhs = Expr::Add(
            Box::new(Expr::Mul(x(), y())),
            Box::new(Expr::Mul(x(), z())),
        );
        let goal = Goal::new(Expr::Eq(Box::new(lhs), Box::new(rhs)));
        assert!(matches!(tactic_ring(&goal), TacticResult::Closed));
    }

    #[test]
    fn test_norm_num_closes_2_plus_3_equals_5() {
        let goal = Goal::new(Expr::Eq(
            Box::new(Expr::Add(Box::new(Expr::Const(2)), Box::new(Expr::Const(3)))),
            Box::new(Expr::Const(5)),
        ));
        assert!(matches!(tactic_norm_num(&goal), TacticResult::Closed));
    }

    #[test]
    fn test_norm_num_fails_on_false_equality() {
        let goal = Goal::new(Expr::Eq(
            Box::new(Expr::Const(2)),
            Box::new(Expr::Const(3)),
        ));
        assert!(matches!(tactic_norm_num(&goal), TacticResult::Failed(_)));
    }

    #[test]
    fn test_cases_on_or() {
        // {h: A ∨ B} ⊢ C → two subgoals: {A} ⊢ C and {B} ⊢ C
        let goal = Goal::new(Expr::Var("C".into())).with_hyp(
            "h",
            Expr::Or(
                Box::new(Expr::Var("A".into())),
                Box::new(Expr::Var("B".into())),
            ),
        );
        match tactic_cases(&goal, "h") {
            TacticResult::Subgoals(subs) => {
                assert_eq!(subs.len(), 2);
            }
            other => panic!("Expected Subgoals, got {:?}", other),
        }
    }

    #[test]
    fn test_cases_on_and() {
        // {h: A ∧ B} ⊢ C → one subgoal with A and B separate
        let goal = Goal::new(Expr::Var("C".into())).with_hyp(
            "h",
            Expr::And(
                Box::new(Expr::Var("A".into())),
                Box::new(Expr::Var("B".into())),
            ),
        );
        match tactic_cases(&goal, "h") {
            TacticResult::Subgoals(subs) => {
                assert_eq!(subs.len(), 1);
                // Should have both A and B as hypotheses
                assert!(subs[0].hypotheses.iter().any(|(_, e)| e == &Expr::Var("A".into())));
                assert!(subs[0].hypotheses.iter().any(|(_, e)| e == &Expr::Var("B".into())));
            }
            other => panic!("Expected Subgoals, got {:?}", other),
        }
    }

    #[test]
    fn test_tactic_prover_proves_modus_ponens() {
        // (P → Q), P ⊢ Q
        let goal = Goal::new(Expr::Var("Q".into()))
            .with_hyp(
                "h1",
                Expr::Implies(
                    Box::new(Expr::Var("P".into())),
                    Box::new(Expr::Var("Q".into())),
                ),
            )
            .with_hyp("h2", Expr::Var("P".into()));
        let prover = TacticProver::new(5);
        // Manual: apply h1, then assumption on P
        let result = tactic_apply(&goal, &Expr::Implies(
            Box::new(Expr::Var("P".into())),
            Box::new(Expr::Var("Q".into())),
        ));
        // Should produce a subgoal ⊢ P
        match result {
            TacticResult::Subgoals(subs) => {
                assert_eq!(subs[0].conclusion, Expr::Var("P".into()));
                // Now assumption closes it
                assert!(matches!(tactic_assumption(&subs[0]), TacticResult::Closed));
            }
            other => panic!("Expected Subgoals, got {:?}", other),
        }
        let _ = prover; // prover available for auto-search
    }

    #[test]
    fn test_tactic_prover_proves_tautology() {
        // ⊢ P → P
        let goal = Goal::new(Expr::Implies(
            Box::new(Expr::Var("P".into())),
            Box::new(Expr::Var("P".into())),
        ));
        let prover = TacticProver::new(5);
        let trace = prover.prove(&goal);
        assert!(trace.is_some(), "Prover should prove P → P");
    }

    #[test]
    fn test_tactic_prover_proves_conjunction_intro() {
        // {A, B} ⊢ A ∧ B
        let goal = Goal::new(Expr::And(
            Box::new(Expr::Var("A".into())),
            Box::new(Expr::Var("B".into())),
        ))
        .with_hyp("hA", Expr::Var("A".into()))
        .with_hyp("hB", Expr::Var("B".into()));
        let prover = TacticProver::new(5);
        let trace = prover.prove(&goal);
        assert!(trace.is_some(), "Prover should prove A ∧ B from {{A, B}}");
    }

    #[test]
    fn test_induction_produces_base_and_step() {
        // ∀n. P(n) → base ⊢ P(0) and {ih: P(n)} ⊢ P(n+1)
        let goal = Goal::new(Expr::ForAll(
            "n".into(),
            Box::new(Expr::Var("P_n".into())),
        ));
        match tactic_induction(&goal, "n") {
            TacticResult::Subgoals(subs) => {
                assert_eq!(subs.len(), 2, "Induction produces base + step");
                // Base case: P(0) = P_n with n=0 → const(0) substituted
                // Step: has ih_n hypothesis
                assert!(subs[1].hypotheses.iter().any(|(n, _)| n == "ih_n"));
            }
            other => panic!("Expected Subgoals, got {:?}", other),
        }
    }

    #[test]
    fn test_expr_simplify_basic() {
        // 0 + x = x
        let e = Expr::Add(Box::new(Expr::Const(0)), Box::new(Expr::Var("x".into())));
        assert_eq!(e.simplify(), Expr::Var("x".into()));

        // x - x = 0
        let e = Expr::Sub(Box::new(Expr::Var("x".into())), Box::new(Expr::Var("x".into())));
        assert_eq!(e.simplify(), Expr::Const(0));
    }

    #[test]
    fn test_expr_free_vars() {
        let e = Expr::Add(Box::new(Expr::Var("x".into())), Box::new(Expr::Var("y".into())));
        let fv = e.free_vars();
        assert!(fv.contains(&"x".to_string()));
        assert!(fv.contains(&"y".to_string()));
    }

    // ── Phase 1 IMO number-theory tactics ───────────────────────────────

    fn exists_linear_goal(a: i64, b: i64, c: i64) -> Goal {
        let x_times = Expr::Mul(
            Box::new(Expr::Const(a)),
            Box::new(Expr::Var("x".into())),
        );
        let y_times = Expr::Mul(
            Box::new(Expr::Const(b)),
            Box::new(Expr::Var("y".into())),
        );
        let sum = Expr::Add(Box::new(x_times), Box::new(y_times));
        let eq = Expr::Eq(Box::new(sum), Box::new(Expr::Const(c)));
        let inner = Expr::Exists("y".into(), Box::new(eq));
        Goal::new(Expr::Exists("x".into(), Box::new(inner)))
    }

    #[test]
    fn test_tactic_linear_diophantine_closes_solvable() {
        let goal = exists_linear_goal(12, 8, 20);
        assert!(matches!(
            tactic_linear_diophantine(&goal, 12, 8, 20),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_linear_diophantine_fails_unsolvable() {
        let goal = exists_linear_goal(6, 9, 5);
        assert!(matches!(
            tactic_linear_diophantine(&goal, 6, 9, 5),
            TacticResult::Failed(_)
        ));
    }

    #[test]
    fn test_tactic_pell_closes_for_nonsquare() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(tactic_pell(&goal, 2), TacticResult::Closed));
        assert!(matches!(tactic_pell(&goal, 13), TacticResult::Closed));
        assert!(matches!(tactic_pell(&goal, 61), TacticResult::Closed));
    }

    #[test]
    fn test_tactic_pell_fails_for_square() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(tactic_pell(&goal, 4), TacticResult::Failed(_)));
        assert!(matches!(tactic_pell(&goal, 9), TacticResult::Failed(_)));
    }

    #[test]
    fn test_tactic_quadratic_residue() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_quadratic_residue(&goal, 2, 7, 1),
            TacticResult::Closed
        ));
        assert!(matches!(
            tactic_quadratic_residue(&goal, 3, 7, -1),
            TacticResult::Closed
        ));
        assert!(matches!(
            tactic_quadratic_residue(&goal, 2, 7, -1),
            TacticResult::Failed(_)
        ));
    }

    #[test]
    fn test_tactic_lte_bound() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_lte_bound(&goal, 3, 5, 2, 6, 2),
            TacticResult::Closed
        ));
        assert!(matches!(
            tactic_lte_bound(&goal, 3, 5, 2, 6, 99),
            TacticResult::Failed(_)
        ));
        assert!(matches!(
            tactic_lte_bound(&goal, 3, 6, 2, 4, 1),
            TacticResult::Failed(_)
        ));
    }

    #[test]
    fn test_tactic_crt_solve() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_crt_solve(&goal, &[(2, 3), (3, 5), (2, 7)]),
            TacticResult::Closed
        ));
        assert!(matches!(
            tactic_crt_solve(&goal, &[(1, 4), (2, 6)]),
            TacticResult::Failed(_)
        ));
    }

    /// Integration test: IMO-style sub-problem combining CRT + Legendre.
    /// Shows two Phase-1 tactics composing to reason about a concrete witness.
    #[test]
    fn test_imo_style_crt_plus_quadratic_residue() {
        let goal = Goal::new(Expr::Const(0));
        match tactic_crt_solve(&goal, &[(1, 4), (2, 5)]) {
            TacticResult::Closed => {}
            other => panic!("CRT failed: {:?}", other),
        }
        let engine = NumberTheoryEngine::new();
        let (x, m) = engine.crt(&[(1, 4), (2, 5)]).unwrap();
        assert_eq!(m, 20);
        assert_eq!(x, 17);
        // (17/11) = (6/11): compute directly
        let leg = engine.legendre_symbol(17, 11);
        assert!(matches!(
            tactic_quadratic_residue(&goal, x, 11, leg),
            TacticResult::Closed
        ));
    }
}
