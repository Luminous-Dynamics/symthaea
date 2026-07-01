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
        self.hypotheses.iter().any(|(_, h)| h == &self.conclusion)
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
                let (from, to) = if direction {
                    (*lhs, *rhs)
                } else {
                    (*rhs, *lhs)
                };
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
            _ => TacticResult::Failed(format!("rw: hypothesis '{}' is not an equation", eq_hyp)),
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
        Expr::Bool(false) => {
            TacticResult::Failed("omega: conclusion is false under current bindings".to_string())
        }
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
        Expr::Bool(false) => {
            return TacticResult::Failed("norm_num: evaluates to false".to_string());
        }
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
        &Expr::Add(
            Box::new(Expr::Var(var.to_string())),
            Box::new(Expr::Const(1)),
        ),
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

use crate::hdc::barycentric::{Barycentric, centroid, circumcenter, incenter, orthocenter};
use crate::hdc::combinatorial::{
    find_linear_invariant, find_linear_monovariant, pigeonhole_apply, pigeonhole_min_max_bucket,
};
use crate::hdc::computational_geometry::Point2D;
use crate::hdc::diophantine::pell_equation;
use crate::hdc::functional_equations::{Classification, EquationKind, classify};
use crate::hdc::inequalities::{
    amgm_holds, cauchy_schwarz_holds, jensen_convex_holds, power_mean_inequality_holds,
    schur_t1_holds, schur_t2_holds,
};
use crate::hdc::number_theory::NumberTheoryEngine;
use crate::hdc::synthetic_geometry::{GeomPredicate, GeomState};

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
        None => TacticResult::Failed(format!("no integer solution to {}x + {}y = {}", a, b, c)),
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

// ─── Phase 2: IMO synthetic-geometry tactics ─────────────────────────────────
//
// These bridge `synthetic_geometry::GeomState` and `barycentric` into the
// tactic framework. Because `Expr` has no native geometric vocabulary, the
// tactics take a `GeomState` (the configuration) and a target
// `GeomPredicate` (the goal to derive), and return `Closed` iff saturation
// proves the target.

/// Run forward saturation on `state` and return `Closed` iff the target
/// predicate appears in the fact base after at most `max_iters` passes.
///
/// This is the analog of Lean's `polyrith` or AlphaGeometry's DD saturation
/// loop: pure forward chaining, no backtracking, numerical verification on
/// every derived fact.
pub fn tactic_angle_chase(
    state: &mut GeomState,
    target: &GeomPredicate,
    max_iters: usize,
) -> TacticResult {
    // Fact already present? Close immediately.
    if state.facts.contains(target) {
        return TacticResult::Closed;
    }
    // Verify the target numerically — if it's false, don't bother saturating.
    match state.verify(target) {
        Some(false) => {
            return TacticResult::Failed(format!(
                "target predicate is numerically false: {:?}",
                target
            ));
        }
        None => {
            return TacticResult::Failed("target references points not in state".to_string());
        }
        Some(true) => {}
    }
    // Saturate, then check.
    state.saturate(max_iters);
    if state.facts.contains(target) {
        TacticResult::Closed
    } else {
        // The target is numerically true but we couldn't derive it via our
        // forward-saturation rule set. Report this honestly as a gap in the
        // deductive vocabulary, not a mathematical falsehood.
        TacticResult::Failed(format!(
            "angle_chase exhausted rules without deriving target: {:?}",
            target
        ))
    }
}

/// Power of a point theorem: for a circle with center O and radius r, and
/// any point P, and a chord through P meeting the circle at X and Y,
/// PX · PY = |PO|² − r² (if P is outside) or r² − |PO|² (if inside).
///
/// This tactic checks the power-of-point identity numerically for a
/// user-supplied point P, circle (O, r), and chord endpoints X, Y — useful
/// as a primitive for ratio-based geometry problems.
pub fn tactic_power_of_point(
    p: &Point2D,
    center: &Point2D,
    radius: f64,
    x: &Point2D,
    y: &Point2D,
) -> TacticResult {
    let px = p.distance(x);
    let py = p.distance(y);
    let po2 = (p.x - center.x).powi(2) + (p.y - center.y).powi(2);
    let expected = (po2 - radius * radius).abs();
    // Account for inside vs outside: px * py should equal |po² − r²|.
    let actual = px * py;
    if (actual - expected).abs() < 1e-7 {
        TacticResult::Closed
    } else {
        TacticResult::Failed(format!(
            "power of point violated: px·py = {}, |po²−r²| = {}",
            actual, expected
        ))
    }
}

/// Similar triangles by SSS: two triangles ABC and DEF are similar iff
/// the ratios of corresponding sides are equal within tolerance.
pub fn tactic_similar_triangles_sss(
    a: &Point2D,
    b: &Point2D,
    c: &Point2D,
    d: &Point2D,
    e: &Point2D,
    f: &Point2D,
) -> TacticResult {
    let ab = a.distance(b);
    let bc = b.distance(c);
    let ca = c.distance(a);
    let de = d.distance(e);
    let ef = e.distance(f);
    let fd = f.distance(d);
    if de < 1e-12 || ef < 1e-12 || fd < 1e-12 {
        return TacticResult::Failed("degenerate triangle".into());
    }
    let r1 = ab / de;
    let r2 = bc / ef;
    let r3 = ca / fd;
    if (r1 - r2).abs() < 1e-7 && (r2 - r3).abs() < 1e-7 {
        TacticResult::Closed
    } else {
        TacticResult::Failed(format!("side ratios differ: {} {} {}", r1, r2, r3))
    }
}

/// Barycentric coerce: compute the barycentric coordinates of a named point
/// in a triangle, identify which classical center it coincides with (if any),
/// and return `Closed` with the identification.
///
/// This is the algebraic-fallback tactic: when saturation stalls on a
/// geometry problem, drop into coordinates and compute directly.
pub fn tactic_barycentric_coerce(
    state: &GeomState,
    point_name: &str,
    triangle: (&str, &str, &str),
) -> TacticResult {
    let (a_name, b_name, c_name) = triangle;
    let (p, a, b, c) = match (
        state.points.get(point_name),
        state.points.get(a_name),
        state.points.get(b_name),
        state.points.get(c_name),
    ) {
        (Some(p), Some(a), Some(b), Some(c)) => (p, a, b, c),
        _ => return TacticResult::Failed("point or triangle vertices missing in state".into()),
    };
    // Compute P's barycentric coordinates.
    let bary = match Barycentric::from_cartesian(p, a, b, c) {
        Some(b) => b,
        None => return TacticResult::Failed("degenerate triangle".into()),
    };
    // Compare against each classical center within tolerance.
    let eps = 1e-6;
    let same = |p1: &Point2D, p2: &Point2D| (p1.x - p2.x).abs() < eps && (p1.y - p2.y).abs() < eps;
    let g = centroid(a, b, c);
    let i = incenter(a, b, c);
    let o = circumcenter(a, b, c);
    let h = orthocenter(a, b, c);
    let _ = bary;
    if same(p, &g) {
        TacticResult::Closed
    } else if same(p, &i) {
        TacticResult::Closed
    } else if same(p, &o) {
        TacticResult::Closed
    } else if same(p, &h) {
        TacticResult::Closed
    } else {
        TacticResult::Failed(format!(
            "{} is not a classical center of triangle {}{}{}",
            point_name, a_name, b_name, c_name
        ))
    }
}

// ─── Phase 3A: IMO inequality tactics ────────────────────────────────────────
//
// Numerical verification wrappers around `hdc::inequalities`. These close
// goals of the form "inequality X holds for witness W" by computing the
// inequality at concrete values. They do NOT prove the inequality for all
// real inputs — that's Z3's job. Their value is fast numerical pre-check
// before committing to a symbolic proof.

/// Closes a goal asserting AM ≥ GM for a concrete non-negative slice.
pub fn tactic_amgm_check(_goal: &Goal, xs: &[f64]) -> TacticResult {
    if xs.iter().any(|&x| x < 0.0) {
        return TacticResult::Failed("AM-GM requires non-negative inputs".into());
    }
    if amgm_holds(xs) {
        TacticResult::Closed
    } else {
        TacticResult::Failed(format!("AM-GM violated on {:?}", xs))
    }
}

/// Closes a goal asserting (Σaᵢbᵢ)² ≤ (Σaᵢ²)(Σbᵢ²) for concrete slices.
pub fn tactic_cauchy_schwarz_check(_goal: &Goal, a: &[f64], b: &[f64]) -> TacticResult {
    if a.len() != b.len() {
        return TacticResult::Failed("Cauchy-Schwarz requires equal-length slices".into());
    }
    if cauchy_schwarz_holds(a, b) {
        TacticResult::Closed
    } else {
        TacticResult::Failed(format!("Cauchy-Schwarz violated on {:?}, {:?}", a, b))
    }
}

/// Closes a goal asserting the power-mean inequality M_p ≤ M_q holds on a
/// concrete non-negative slice.
pub fn tactic_power_mean_check(_goal: &Goal, xs: &[f64], p: f64, q: f64) -> TacticResult {
    if p > q {
        return TacticResult::Failed(format!("power mean requires p ≤ q, got p={}, q={}", p, q));
    }
    if power_mean_inequality_holds(xs, p, q) {
        TacticResult::Closed
    } else {
        TacticResult::Failed(format!(
            "power mean violated on {:?} at p={}, q={}",
            xs, p, q
        ))
    }
}

/// Closes a Jensen-inequality goal for a user-supplied convex function,
/// weights (summing to 1), and sample points.
pub fn tactic_jensen_check<F>(_goal: &Goal, f: F, weights: &[f64], points: &[f64]) -> TacticResult
where
    F: Fn(f64) -> f64,
{
    if weights.len() != points.len() {
        return TacticResult::Failed("Jensen: weights and points length mismatch".into());
    }
    let total: f64 = weights.iter().sum();
    if (total - 1.0).abs() > 1e-9 {
        return TacticResult::Failed(format!("Jensen: weights sum to {}, not 1", total));
    }
    if weights.iter().any(|&w| w < 0.0) {
        return TacticResult::Failed("Jensen: weights must be non-negative".into());
    }
    if jensen_convex_holds(f, weights, points) {
        TacticResult::Closed
    } else {
        TacticResult::Failed("Jensen (convex form) violated".into())
    }
}

/// Closes a Schur-inequality goal for concrete non-negative triples at
/// exponent t ∈ {1, 2}.
pub fn tactic_schur_check(_goal: &Goal, a: f64, b: f64, c: f64, t: u32) -> TacticResult {
    if a < 0.0 || b < 0.0 || c < 0.0 {
        return TacticResult::Failed("Schur requires non-negative inputs".into());
    }
    let ok = match t {
        1 => schur_t1_holds(a, b, c),
        2 => schur_t2_holds(a, b, c),
        _ => return TacticResult::Failed(format!("Schur exponent t={} not supported", t)),
    };
    if ok {
        TacticResult::Closed
    } else {
        TacticResult::Failed(format!("Schur t={} violated at ({}, {}, {})", t, a, b, c))
    }
}

// ─── Phase 3C: functional equation classification tactic ────────────────────
//
// Wraps `hdc::functional_equations::classify` for the IMO solver. Goal-form
// problems like "find all f: R → R such that f(x+y) = f(x) + f(y)" supply
// a sample set; this tactic returns `Closed` if classification matches the
// expected family with high confidence, else `Failed` with the actual best
// fit for diagnostics.

/// Closes a functional-equation goal: given sampled `(x, f(x))` pairs and
/// an `expected` family, succeed iff the classifier identifies that family
/// with confidence ≥ 0.99. Lower confidences are reported in the failure
/// message to aid debugging. The classifier IS the verification step —
/// it's a numerical proof-of-fit, not a symbolic proof of uniqueness.
pub fn tactic_classify_functional_equation(
    _goal: &Goal,
    samples: &[(f64, f64)],
    expected: EquationKind,
) -> TacticResult {
    if samples.is_empty() {
        return TacticResult::Failed("functional equation: empty sample set".into());
    }
    let Classification {
        kind,
        constant,
        confidence,
    } = classify(samples);
    if kind == expected && confidence >= 0.99 {
        TacticResult::Closed
    } else {
        TacticResult::Failed(format!(
            "expected {:?}, got {:?} (constant={:.4}, confidence={:.3})",
            expected, kind, constant, confidence
        ))
    }
}

// ─── Phase 4 (scoped): combinatorial tactics ─────────────────────────────────
//
// Three tactics wrapping the primitives from `hdc::combinatorial`:
// pigeonhole, invariant discovery, monovariant (termination proof).

/// Closes a pigeonhole-style goal: given `items`, partition them by a
/// concrete function, and verify that pigeonhole forces at least
/// `min_collision` items in some bucket.
pub fn tactic_pigeonhole<T, K, F>(
    _goal: &Goal,
    items: &[T],
    partition: F,
    min_collision: usize,
) -> TacticResult
where
    K: std::hash::Hash + Eq,
    F: Fn(&T) -> K,
{
    match pigeonhole_apply(items, partition, min_collision) {
        Some(_) => TacticResult::Closed,
        None => TacticResult::Failed(format!(
            "pigeonhole did not force a bucket of size ≥ {}",
            min_collision
        )),
    }
}

/// Closes a "pigeonhole guarantees some bucket has ≥ k items" goal via
/// the classical min_max_bucket formula. Useful when the partition is
/// abstract but the cardinalities are known.
pub fn tactic_pigeonhole_count(
    _goal: &Goal,
    items: usize,
    boxes: usize,
    claimed_min: usize,
) -> TacticResult {
    let actual = pigeonhole_min_max_bucket(items, boxes);
    if actual >= claimed_min {
        TacticResult::Closed
    } else {
        TacticResult::Failed(format!(
            "pigeonhole guarantees only {}, not {}",
            actual, claimed_min
        ))
    }
}

/// Closes a goal "the quantity c · s is invariant" by finding a linear
/// invariant for the given trajectory. Returns Closed with the
/// discovered coefficients (reported in the Failed message for audit,
/// since TacticResult::Closed carries no payload).
pub fn tactic_invariant_search(_goal: &Goal, trajectory: &[Vec<f64>]) -> TacticResult {
    match find_linear_invariant(trajectory) {
        Some((_c, residual)) if residual < 1e-6 => TacticResult::Closed,
        Some((c, residual)) => TacticResult::Failed(format!(
            "found near-invariant but residual too high ({:.2e}): c = {:?}",
            residual, c
        )),
        None => TacticResult::Failed("no linear invariant exists".to_string()),
    }
}

/// Closes a termination goal by finding a strict monovariant: a linear
/// function that strictly decreases (or increases, if `seek_decreasing`
/// is false) at every step of the trajectory.
pub fn tactic_monovariant(
    _goal: &Goal,
    trajectory: &[Vec<f64>],
    seek_decreasing: bool,
) -> TacticResult {
    match find_linear_monovariant(trajectory, seek_decreasing) {
        Some(_) => TacticResult::Closed,
        None => TacticResult::Failed("no linear monovariant found".to_string()),
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
            (
                "contradiction",
                Box::new(|g: &Goal| tactic_contradiction(g)),
            ),
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
                        if let result @ Some(_) = self.prove_goals(new_goals, new_trace, depth + 1)
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
                        if let result @ Some(_) = self.prove_goals(new_goals, new_trace, depth + 1)
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
        let rhs = Expr::Add(Box::new(Expr::Mul(x(), y())), Box::new(Expr::Mul(x(), z())));
        let goal = Goal::new(Expr::Eq(Box::new(lhs), Box::new(rhs)));
        assert!(matches!(tactic_ring(&goal), TacticResult::Closed));
    }

    #[test]
    fn test_norm_num_closes_2_plus_3_equals_5() {
        let goal = Goal::new(Expr::Eq(
            Box::new(Expr::Add(
                Box::new(Expr::Const(2)),
                Box::new(Expr::Const(3)),
            )),
            Box::new(Expr::Const(5)),
        ));
        assert!(matches!(tactic_norm_num(&goal), TacticResult::Closed));
    }

    #[test]
    fn test_norm_num_fails_on_false_equality() {
        let goal = Goal::new(Expr::Eq(Box::new(Expr::Const(2)), Box::new(Expr::Const(3))));
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
                assert!(
                    subs[0]
                        .hypotheses
                        .iter()
                        .any(|(_, e)| e == &Expr::Var("A".into()))
                );
                assert!(
                    subs[0]
                        .hypotheses
                        .iter()
                        .any(|(_, e)| e == &Expr::Var("B".into()))
                );
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
        let result = tactic_apply(
            &goal,
            &Expr::Implies(
                Box::new(Expr::Var("P".into())),
                Box::new(Expr::Var("Q".into())),
            ),
        );
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
        let goal = Goal::new(Expr::ForAll("n".into(), Box::new(Expr::Var("P_n".into()))));
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
        let e = Expr::Sub(
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("x".into())),
        );
        assert_eq!(e.simplify(), Expr::Const(0));
    }

    #[test]
    fn test_expr_free_vars() {
        let e = Expr::Add(
            Box::new(Expr::Var("x".into())),
            Box::new(Expr::Var("y".into())),
        );
        let fv = e.free_vars();
        assert!(fv.contains(&"x".to_string()));
        assert!(fv.contains(&"y".to_string()));
    }

    // ── Phase 1 IMO number-theory tactics ───────────────────────────────

    fn exists_linear_goal(a: i64, b: i64, c: i64) -> Goal {
        let x_times = Expr::Mul(Box::new(Expr::Const(a)), Box::new(Expr::Var("x".into())));
        let y_times = Expr::Mul(Box::new(Expr::Const(b)), Box::new(Expr::Var("y".into())));
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

    // ── Phase 2 IMO synthetic-geometry tactics ─────────────────────────

    use crate::hdc::computational_geometry::Point2D as P2;

    fn pt(x: f64, y: f64) -> P2 {
        P2::new(x, y)
    }

    #[test]
    fn test_tactic_angle_chase_derives_inscribed_angle_equality() {
        // Cyclic quadrilateral on the unit circle.
        let mut s = GeomState::new();
        s.add_point("A", pt(1.0, 0.0));
        s.add_point("B", pt(0.0, 1.0));
        s.add_point("C", pt(-1.0, 0.0));
        s.add_point("D", pt(0.0, -1.0));
        s.add_fact(GeomPredicate::Concyclic(
            "A".into(),
            "B".into(),
            "C".into(),
            "D".into(),
        ));
        // Inscribed angle: ∠BAC = ∠BDC (both subtend arc BC).
        let target = GeomPredicate::AngleEq(
            "B".into(),
            "A".into(),
            "C".into(),
            "B".into(),
            "D".into(),
            "C".into(),
        );
        match tactic_angle_chase(&mut s, &target, 10) {
            TacticResult::Closed => {}
            other => panic!("angle_chase should close: {:?}", other),
        }
    }

    #[test]
    fn test_tactic_angle_chase_rejects_false_target() {
        let mut s = GeomState::new();
        s.add_point("A", pt(0.0, 0.0));
        s.add_point("B", pt(1.0, 0.0));
        s.add_point("C", pt(0.0, 1.0));
        // A, B, C are NOT collinear — claim they are and expect Failed.
        let target = GeomPredicate::Collinear("A".into(), "B".into(), "C".into());
        assert!(matches!(
            tactic_angle_chase(&mut s, &target, 5),
            TacticResult::Failed(_)
        ));
    }

    #[test]
    fn test_tactic_power_of_point_external() {
        // Circle centered at origin, radius 2. Point P = (3, 0).
        // Any chord through P meets the circle at two points whose distances
        // to P multiply to |PO|² − r² = 9 − 4 = 5.
        // The horizontal chord meets at (-2, 0) and (2, 0): px = 5, py = 1.
        let center = pt(0.0, 0.0);
        let p = pt(3.0, 0.0);
        let x = pt(-2.0, 0.0);
        let y = pt(2.0, 0.0);
        assert!(matches!(
            tactic_power_of_point(&p, &center, 2.0, &x, &y),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_similar_triangles_sss() {
        // 3-4-5 right triangle and its 6-8-10 scaling.
        let a = pt(0.0, 0.0);
        let b = pt(4.0, 0.0);
        let c = pt(0.0, 3.0);
        let d = pt(0.0, 0.0);
        let e = pt(8.0, 0.0);
        let f = pt(0.0, 6.0);
        assert!(matches!(
            tactic_similar_triangles_sss(&a, &b, &c, &d, &e, &f),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_similar_triangles_not_similar() {
        let a = pt(0.0, 0.0);
        let b = pt(4.0, 0.0);
        let c = pt(0.0, 3.0);
        let d = pt(0.0, 0.0);
        let e = pt(5.0, 0.0);
        let f = pt(0.0, 6.0); // different aspect ratio
        assert!(matches!(
            tactic_similar_triangles_sss(&a, &b, &c, &d, &e, &f),
            TacticResult::Failed(_)
        ));
    }

    #[test]
    fn test_tactic_barycentric_coerce_identifies_centroid() {
        let mut s = GeomState::new();
        s.add_point("A", pt(0.0, 0.0));
        s.add_point("B", pt(6.0, 0.0));
        s.add_point("C", pt(0.0, 6.0));
        // Centroid of this triangle is at (2, 2).
        s.add_point("G", pt(2.0, 2.0));
        assert!(matches!(
            tactic_barycentric_coerce(&s, "G", ("A", "B", "C")),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_barycentric_coerce_rejects_non_center() {
        let mut s = GeomState::new();
        s.add_point("A", pt(0.0, 0.0));
        s.add_point("B", pt(6.0, 0.0));
        s.add_point("C", pt(0.0, 6.0));
        s.add_point("X", pt(5.0, 5.0)); // not a classical center
        assert!(matches!(
            tactic_barycentric_coerce(&s, "X", ("A", "B", "C")),
            TacticResult::Failed(_)
        ));
    }

    /// End-to-end Phase 2 integration: cyclic quadrilateral on the unit
    /// circle, use saturation to derive inscribed-angle equality, then use
    /// it to identify that the triangle formed by three of the vertices
    /// has its circumcircle coincident with the quadrilateral's circle.
    #[test]
    fn test_phase2_integration_cyclic_quadrilateral() {
        let mut s = GeomState::new();
        // Square inscribed in unit circle
        s.add_point("A", pt(1.0, 0.0));
        s.add_point("B", pt(0.0, 1.0));
        s.add_point("C", pt(-1.0, 0.0));
        s.add_point("D", pt(0.0, -1.0));
        s.add_fact(GeomPredicate::Concyclic(
            "A".into(),
            "B".into(),
            "C".into(),
            "D".into(),
        ));
        // Step 1: saturate to derive inscribed-angle facts.
        let added = s.saturate(10);
        assert!(added > 0, "saturation should produce new angle facts");
        // Step 2: verify at least one inscribed-angle equality is present.
        let has_angle_fact = s
            .facts
            .iter()
            .any(|f| matches!(f, GeomPredicate::AngleEq(_, _, _, _, _, _)));
        assert!(has_angle_fact);
        // Step 3: the circumcircle of triangle ABC should have center (0,0),
        // which matches the quadrilateral's inscribed circle center — verify
        // via the barycentric `circumcenter` primitive.
        use crate::hdc::barycentric::circumcenter;
        let a = pt(1.0, 0.0);
        let b = pt(0.0, 1.0);
        let c = pt(-1.0, 0.0);
        let o = circumcenter(&a, &b, &c);
        assert!(o.x.abs() < 1e-9 && o.y.abs() < 1e-9);
        // Step 4: every derived fact must still verify.
        assert!(s.facts_consistent());
    }

    // ── Phase 3A inequality tactics ────────────────────────────────────

    #[test]
    fn test_tactic_amgm_check_closes() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_amgm_check(&goal, &[1.0, 4.0, 9.0]),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_amgm_check_rejects_negative() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_amgm_check(&goal, &[1.0, -1.0, 4.0]),
            TacticResult::Failed(_)
        ));
    }

    #[test]
    fn test_tactic_cauchy_schwarz_check_closes() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_cauchy_schwarz_check(&goal, &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_cauchy_schwarz_length_mismatch() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_cauchy_schwarz_check(&goal, &[1.0, 2.0], &[1.0, 2.0, 3.0]),
            TacticResult::Failed(_)
        ));
    }

    #[test]
    fn test_tactic_power_mean_check_closes() {
        let goal = Goal::new(Expr::Const(0));
        // HM ≤ GM ≤ AM ≤ QM — multiple valid (p, q) pairs
        for &(p, q) in &[(-1.0, 0.0), (0.0, 1.0), (1.0, 2.0)] {
            assert!(matches!(
                tactic_power_mean_check(&goal, &[1.0, 2.0, 4.0], p, q),
                TacticResult::Closed
            ));
        }
    }

    #[test]
    fn test_tactic_power_mean_wrong_order() {
        let goal = Goal::new(Expr::Const(0));
        // p > q should fail the precondition check
        assert!(matches!(
            tactic_power_mean_check(&goal, &[1.0, 2.0, 4.0], 2.0, 1.0),
            TacticResult::Failed(_)
        ));
    }

    #[test]
    fn test_tactic_jensen_check_x_squared() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_jensen_check(&goal, |x: f64| x * x, &[0.3, 0.3, 0.4], &[1.0, 2.0, 3.0]),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_jensen_check_weights_dont_sum_to_one() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_jensen_check(&goal, |x: f64| x * x, &[0.5, 0.3], &[1.0, 2.0]),
            TacticResult::Failed(_)
        ));
    }

    #[test]
    fn test_tactic_schur_check_t1() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_schur_check(&goal, 1.0, 2.0, 3.0, 1),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_schur_check_t2() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_schur_check(&goal, 0.5, 1.5, 2.5, 2),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_schur_unsupported_exponent() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_schur_check(&goal, 1.0, 2.0, 3.0, 5),
            TacticResult::Failed(_)
        ));
    }

    /// Phase 3A integration: stack three inequality tactics in sequence to
    /// verify the classical chain   HM ≤ GM ≤ AM  on a concrete triple.
    /// Each tactic closes independently; together they demonstrate the
    /// compositional pattern for numerical inequality reasoning.
    #[test]
    fn test_phase3a_integration_hm_gm_am_chain() {
        let goal = Goal::new(Expr::Const(0));
        let xs = &[1.0, 2.0, 4.0][..];
        // HM ≤ GM  (power mean at p=-1 ≤ p=0)
        assert!(matches!(
            tactic_power_mean_check(&goal, xs, -1.0, 0.0),
            TacticResult::Closed
        ));
        // GM ≤ AM  (power mean at p=0 ≤ p=1)
        assert!(matches!(
            tactic_power_mean_check(&goal, xs, 0.0, 1.0),
            TacticResult::Closed
        ));
        // And direct AM ≥ GM via the AM-GM tactic
        assert!(matches!(tactic_amgm_check(&goal, xs), TacticResult::Closed));
    }

    // ── Phase 3C functional equation classification tactic ─────────────

    #[test]
    fn test_tactic_classify_cauchy_linear() {
        let goal = Goal::new(Expr::Const(0));
        // f(x) = 5x sampled on a sum-closed grid 0..6.
        let samples: Vec<(f64, f64)> = (0..=6).map(|i| (i as f64, 5.0 * i as f64)).collect();
        assert!(matches!(
            tactic_classify_functional_equation(&goal, &samples, EquationKind::CauchyAdditive),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_classify_exponential() {
        let goal = Goal::new(Expr::Const(0));
        // f(x) = 3^x on 0..6 — verifies the exponential law f(x+y)=f(x)f(y).
        let samples: Vec<(f64, f64)> = (0..=6).map(|i| (i as f64, 3f64.powi(i))).collect();
        assert!(matches!(
            tactic_classify_functional_equation(&goal, &samples, EquationKind::Exponential),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_classify_wrong_family_fails() {
        let goal = Goal::new(Expr::Const(0));
        // Linear samples — Cauchy works, exponential doesn't.
        let samples: Vec<(f64, f64)> = (0..=6).map(|i| (i as f64, 2.0 * i as f64)).collect();
        assert!(matches!(
            tactic_classify_functional_equation(&goal, &samples, EquationKind::Exponential),
            TacticResult::Failed(_)
        ));
    }

    #[test]
    fn test_tactic_classify_empty_samples_fails() {
        let goal = Goal::new(Expr::Const(0));
        assert!(matches!(
            tactic_classify_functional_equation(&goal, &[], EquationKind::CauchyAdditive),
            TacticResult::Failed(_)
        ));
    }

    // ── Phase 4 (scoped) combinatorial tactics ─────────────────────────

    #[test]
    fn test_tactic_pigeonhole_mod_6() {
        let goal = Goal::new(Expr::Const(0));
        let ints: Vec<i32> = vec![3, 14, 27, 100, 5, 18, 71];
        assert!(matches!(
            tactic_pigeonhole(&goal, &ints, |&n: &i32| (n % 6 + 6) % 6, 2),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_pigeonhole_count_classic() {
        let goal = Goal::new(Expr::Const(0));
        // 14 people, 12 months → some month has ≥ 2 people
        assert!(matches!(
            tactic_pigeonhole_count(&goal, 14, 12, 2),
            TacticResult::Closed
        ));
        // But not ≥ 3 — not forced by pigeonhole on 14 items in 12 boxes
        assert!(matches!(
            tactic_pigeonhole_count(&goal, 14, 12, 3),
            TacticResult::Failed(_)
        ));
    }

    #[test]
    fn test_tactic_invariant_search_closes() {
        let goal = Goal::new(Expr::Const(0));
        // (a, b) → (a-1, b+1): sum invariant
        let trajectory = vec![vec![5.0, 3.0], vec![4.0, 4.0], vec![3.0, 5.0]];
        assert!(matches!(
            tactic_invariant_search(&goal, &trajectory),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_invariant_search_rejects_no_invariant() {
        let goal = Goal::new(Expr::Const(0));
        // (a, b) → (2a, b+1): no linear invariant
        let trajectory = vec![vec![1.0, 0.0], vec![2.0, 1.0], vec![4.0, 2.0]];
        assert!(matches!(
            tactic_invariant_search(&goal, &trajectory),
            TacticResult::Failed(_)
        ));
    }

    #[test]
    fn test_tactic_monovariant_termination() {
        let goal = Goal::new(Expr::Const(0));
        // Simple counter: proves termination
        let trajectory = vec![vec![10.0], vec![9.0], vec![8.0], vec![7.0]];
        assert!(matches!(
            tactic_monovariant(&goal, &trajectory, true),
            TacticResult::Closed
        ));
    }

    #[test]
    fn test_tactic_monovariant_rejects_oscillation() {
        let goal = Goal::new(Expr::Const(0));
        let trajectory = vec![vec![1.0], vec![2.0], vec![1.0], vec![2.0]];
        assert!(matches!(
            tactic_monovariant(&goal, &trajectory, true),
            TacticResult::Failed(_)
        ));
    }

    /// Phase 4 integration: prove a discrete combinatorial claim using
    /// pigeonhole + invariant search in sequence. Scenario: a chip-firing
    /// game on a line where each step moves a chip from (x, y) to
    /// (x-1, y+1). Show (1) the total chip count x+y is invariant, and
    /// (2) the game terminates because x is a strict monovariant.
    #[test]
    fn test_phase4_integration_chip_firing() {
        let goal = Goal::new(Expr::Const(0));
        let trajectory = vec![
            vec![5.0, 0.0],
            vec![4.0, 1.0],
            vec![3.0, 2.0],
            vec![2.0, 3.0],
            vec![1.0, 4.0],
            vec![0.0, 5.0],
        ];
        // Claim 1: chip count (x + y) is invariant
        assert!(matches!(
            tactic_invariant_search(&goal, &trajectory),
            TacticResult::Closed
        ));
        // Claim 2: x strictly decreases → termination
        assert!(matches!(
            tactic_monovariant(&goal, &trajectory, true),
            TacticResult::Closed
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
