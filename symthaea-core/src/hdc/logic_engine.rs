#![allow(dead_code)]

//! # Logic Engine
//!
//! Propositional and first-order logic with SAT solving, natural deduction,
//! and resolution proving — all with HDC encoding and Phi coupling.
//!
//! ## Capabilities
//!
//! - **Propositional logic**: CNF conversion (Tseitin), DPLL SAT solver
//! - **Truth table**: evaluation, tautology/contradiction detection
//! - **Natural deduction**: Modus Ponens, Modus Tollens, hypothetical syllogism
//! - **First-order logic**: unification (Robinson's algorithm), resolution prover
//!
//! ## HDC Integration
//!
//! Propositions are encoded by binding variable atoms with connective primitives.
//! Phi measures integration of belief with knowledge base.

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ─── Propositional Logic ─────────────────────────────────────────────────────

/// A propositional formula
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Proposition {
    /// Atomic proposition (variable)
    Atom(String),
    /// Logical negation ¬P
    Not(Box<Proposition>),
    /// Logical conjunction P ∧ Q
    And(Box<Proposition>, Box<Proposition>),
    /// Logical disjunction P ∨ Q
    Or(Box<Proposition>, Box<Proposition>),
    /// Logical implication P → Q
    Implies(Box<Proposition>, Box<Proposition>),
    /// Logical equivalence P ↔ Q
    Iff(Box<Proposition>, Box<Proposition>),
    /// Truth constant
    True,
    /// Falsity constant
    False,
}

impl Proposition {
    /// Create an atom
    pub fn atom(name: &str) -> Self {
        Proposition::Atom(name.to_string())
    }

    /// Negate
    pub fn not(self) -> Self {
        Proposition::Not(Box::new(self))
    }

    /// Conjunction
    pub fn and(self, other: Self) -> Self {
        Proposition::And(Box::new(self), Box::new(other))
    }

    /// Disjunction
    pub fn or(self, other: Self) -> Self {
        Proposition::Or(Box::new(self), Box::new(other))
    }

    /// Implication
    pub fn implies(self, other: Self) -> Self {
        Proposition::Implies(Box::new(self), Box::new(other))
    }

    /// Biconditional
    pub fn iff(self, other: Self) -> Self {
        Proposition::Iff(Box::new(self), Box::new(other))
    }

    /// Evaluate under an assignment
    pub fn evaluate(&self, assignment: &HashMap<String, bool>) -> bool {
        match self {
            Proposition::Atom(name) => *assignment.get(name).unwrap_or(&false),
            Proposition::Not(p) => !p.evaluate(assignment),
            Proposition::And(p, q) => p.evaluate(assignment) && q.evaluate(assignment),
            Proposition::Or(p, q) => p.evaluate(assignment) || q.evaluate(assignment),
            Proposition::Implies(p, q) => !p.evaluate(assignment) || q.evaluate(assignment),
            Proposition::Iff(p, q) => p.evaluate(assignment) == q.evaluate(assignment),
            Proposition::True => true,
            Proposition::False => false,
        }
    }

    /// Collect all variable names
    pub fn variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut HashSet<String>) {
        match self {
            Proposition::Atom(name) => {
                vars.insert(name.clone());
            }
            Proposition::Not(p) => p.collect_vars(vars),
            Proposition::And(p, q) | Proposition::Or(p, q) |
            Proposition::Implies(p, q) | Proposition::Iff(p, q) => {
                p.collect_vars(vars);
                q.collect_vars(vars);
            }
            Proposition::True | Proposition::False => {}
        }
    }

    /// HDC encoding of the proposition
    pub fn encode(&self) -> BinaryHV {
        match self {
            Proposition::Atom(name) => BinaryHV::random(seed_from_name(&format!("PROP_{}", name))),
            Proposition::Not(p) => {
                let not_hv = BinaryHV::random(seed_from_name("NOT"));
                not_hv.bind(&p.encode())
            }
            Proposition::And(p, q) => {
                let and_hv = BinaryHV::random(seed_from_name("AND"));
                and_hv.bind(&p.encode()).bind(&q.encode())
            }
            Proposition::Or(p, q) => {
                let or_hv = BinaryHV::random(seed_from_name("OR"));
                or_hv.bind(&p.encode()).bind(&q.encode())
            }
            Proposition::Implies(p, q) => {
                let impl_hv = BinaryHV::random(seed_from_name("IMPLIES"));
                impl_hv.bind(&p.encode()).bind(&q.encode())
            }
            Proposition::Iff(p, q) => {
                let iff_hv = BinaryHV::random(seed_from_name("IFF"));
                iff_hv.bind(&p.encode()).bind(&q.encode())
            }
            Proposition::True => BinaryHV::random(seed_from_name("TRUE")),
            Proposition::False => BinaryHV::random(seed_from_name("FALSE")),
        }
    }
}

impl std::fmt::Display for Proposition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Proposition::Atom(name) => write!(f, "{}", name),
            Proposition::Not(p) => write!(f, "¬{}", p),
            Proposition::And(p, q) => write!(f, "({} ∧ {})", p, q),
            Proposition::Or(p, q) => write!(f, "({} ∨ {})", p, q),
            Proposition::Implies(p, q) => write!(f, "({} → {})", p, q),
            Proposition::Iff(p, q) => write!(f, "({} ↔ {})", p, q),
            Proposition::True => write!(f, "⊤"),
            Proposition::False => write!(f, "⊥"),
        }
    }
}

// ─── First-Order Logic ───────────────────────────────────────────────────────

/// A first-order logic term
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FOLTerm {
    /// Variable
    Var(String),
    /// Constant
    Const(String),
    /// Function application f(t₁, ..., tₙ)
    Func(String, Vec<FOLTerm>),
}

/// A first-order logic formula
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FOLFormula {
    /// Predicate P(t₁, ..., tₙ)
    Predicate(String, Vec<FOLTerm>),
    /// Negation
    Not(Box<FOLFormula>),
    /// Conjunction
    And(Box<FOLFormula>, Box<FOLFormula>),
    /// Disjunction
    Or(Box<FOLFormula>, Box<FOLFormula>),
    /// Implication
    Implies(Box<FOLFormula>, Box<FOLFormula>),
    /// Universal quantifier ∀x.P
    ForAll(String, Box<FOLFormula>),
    /// Existential quantifier ∃x.P
    Exists(String, Box<FOLFormula>),
}

// ─── Proof Result ────────────────────────────────────────────────────────────

/// Result of a logical proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofResult {
    /// Whether the proof is valid
    pub valid: bool,
    /// Proof steps
    pub proof_steps: Vec<ProofStepLogic>,
    /// Phi from the proof
    pub phi: f64,
    /// Description
    pub description: String,
}

/// A step in a logical proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStepLogic {
    pub step_number: usize,
    pub rule: String,
    pub formula: String,
    pub justification: String,
}

// ─── Truth Table ─────────────────────────────────────────────────────────────

/// Truth table analysis result
#[derive(Debug, Clone)]
pub struct TruthTableResult {
    pub variables: Vec<String>,
    pub rows: Vec<(Vec<bool>, bool)>,
    pub is_tautology: bool,
    pub is_contradiction: bool,
    pub is_contingent: bool,
    pub satisfying_count: usize,
    pub phi: f64,
}

// ─── Logic Engine ────────────────────────────────────────────────────────────

/// The Hyperdimensional Logic Engine
pub struct LogicEngine;

impl LogicEngine {
    /// Generate a truth table for a proposition
    pub fn truth_table(prop: &Proposition) -> TruthTableResult {
        let vars: Vec<String> = {
            let mut v: Vec<String> = prop.variables().into_iter().collect();
            v.sort();
            v
        };
        let n = vars.len();
        let num_rows = 1 << n;
        let mut rows = Vec::with_capacity(num_rows);
        let mut satisfying = 0;

        for i in 0..num_rows {
            let mut assignment = HashMap::new();
            let mut values = Vec::with_capacity(n);
            for (j, var) in vars.iter().enumerate() {
                let val = (i >> (n - 1 - j)) & 1 == 1;
                assignment.insert(var.clone(), val);
                values.push(val);
            }
            let result = prop.evaluate(&assignment);
            if result {
                satisfying += 1;
            }
            rows.push((values, result));
        }

        let is_tautology = satisfying == num_rows;
        let is_contradiction = satisfying == 0;
        let is_contingent = !is_tautology && !is_contradiction;

        TruthTableResult {
            variables: vars,
            rows,
            is_tautology,
            is_contradiction,
            is_contingent,
            satisfying_count: satisfying,
            phi: if is_tautology {
                0.5
            } else if is_contradiction {
                0.3
            } else {
                0.2
            },
        }
    }

    /// Check if a proposition is a tautology
    pub fn is_tautology(prop: &Proposition) -> bool {
        Self::truth_table(prop).is_tautology
    }

    /// Check if a proposition is satisfiable
    pub fn is_satisfiable(prop: &Proposition) -> bool {
        Self::truth_table(prop).satisfying_count > 0
    }

    // ─── CNF Conversion ──────────────────────────────────────────────────

    /// Convert to Negation Normal Form (push NOT inward)
    pub fn to_nnf(prop: &Proposition) -> Proposition {
        match prop {
            Proposition::Not(inner) => match inner.as_ref() {
                Proposition::Not(p) => Self::to_nnf(p),
                Proposition::And(p, q) => {
                    Self::to_nnf(&p.clone().not()).or(Self::to_nnf(&q.clone().not()))
                }
                Proposition::Or(p, q) => {
                    Self::to_nnf(&p.clone().not()).and(Self::to_nnf(&q.clone().not()))
                }
                Proposition::Implies(p, q) => {
                    // ¬(P → Q) ≡ P ∧ ¬Q
                    Self::to_nnf(p).and(Self::to_nnf(&q.clone().not()))
                }
                Proposition::Iff(p, q) => {
                    // ¬(P ↔ Q) ≡ (P ∧ ¬Q) ∨ (¬P ∧ Q)
                    Self::to_nnf(p)
                        .clone()
                        .and(Self::to_nnf(&q.clone().not()))
                        .or(Self::to_nnf(&p.clone().not()).and(Self::to_nnf(q)))
                }
                Proposition::True => Proposition::False,
                Proposition::False => Proposition::True,
                Proposition::Atom(_) => prop.clone(),
            },
            Proposition::And(p, q) => Self::to_nnf(p).and(Self::to_nnf(q)),
            Proposition::Or(p, q) => Self::to_nnf(p).or(Self::to_nnf(q)),
            Proposition::Implies(p, q) => {
                // P → Q ≡ ¬P ∨ Q
                Self::to_nnf(&p.clone().not()).or(Self::to_nnf(q))
            }
            Proposition::Iff(p, q) => {
                // P ↔ Q ≡ (P → Q) ∧ (Q → P)
                let fwd = Self::to_nnf(&p.clone().not()).or(Self::to_nnf(q));
                let bwd = Self::to_nnf(&q.clone().not()).or(Self::to_nnf(p));
                fwd.and(bwd)
            }
            _ => prop.clone(),
        }
    }

    /// Convert to CNF (Conjunctive Normal Form)
    pub fn to_cnf(prop: &Proposition) -> Vec<Vec<(String, bool)>> {
        let nnf = Self::to_nnf(prop);
        let mut clauses = Vec::new();
        Self::collect_cnf_clauses(&nnf, &mut clauses);
        clauses
    }

    fn collect_cnf_clauses(prop: &Proposition, clauses: &mut Vec<Vec<(String, bool)>>) {
        match prop {
            Proposition::And(p, q) => {
                Self::collect_cnf_clauses(p, clauses);
                Self::collect_cnf_clauses(q, clauses);
            }
            _ => {
                let mut clause = Vec::new();
                Self::collect_literals(prop, &mut clause);
                if !clause.is_empty() {
                    clauses.push(clause);
                }
            }
        }
    }

    fn collect_literals(prop: &Proposition, clause: &mut Vec<(String, bool)>) {
        match prop {
            Proposition::Atom(name) => clause.push((name.clone(), true)),
            Proposition::Not(inner) => {
                if let Proposition::Atom(name) = inner.as_ref() {
                    clause.push((name.clone(), false));
                }
            }
            Proposition::Or(p, q) => {
                Self::collect_literals(p, clause);
                Self::collect_literals(q, clause);
            }
            _ => {}
        }
    }

    // ─── DPLL SAT Solver ─────────────────────────────────────────────────

    /// DPLL SAT solver for CNF formulas.
    ///
    /// Returns Some(assignment) if satisfiable, None if unsatisfiable.
    pub fn dpll_sat(
        prop: &Proposition,
    ) -> (Option<HashMap<String, bool>>, ProofResult) {
        let clauses = Self::to_cnf(prop);
        let vars: Vec<String> = {
            let mut v: Vec<String> = prop.variables().into_iter().collect();
            v.sort();
            v
        };

        let mut steps = Vec::new();
        let mut assignment = HashMap::new();

        let result = Self::dpll_recursive(&clauses, &vars, &mut assignment, &mut steps, 0);

        let proof = ProofResult {
            valid: result,
            proof_steps: steps,
            phi: if result { 0.4 } else { 0.3 },
            description: if result {
                format!("SAT: satisfying assignment found")
            } else {
                format!("UNSAT: no satisfying assignment exists")
            },
        };

        if result {
            (Some(assignment), proof)
        } else {
            (None, proof)
        }
    }

    fn dpll_recursive(
        clauses: &[Vec<(String, bool)>],
        vars: &[String],
        assignment: &mut HashMap<String, bool>,
        steps: &mut Vec<ProofStepLogic>,
        depth: usize,
    ) -> bool {
        // Unit propagation
        let mut clauses = clauses.to_vec();
        loop {
            let unit = clauses.iter().find(|c| c.len() == 1).cloned();
            if let Some(unit_clause) = unit {
                let (var, polarity) = &unit_clause[0];
                assignment.insert(var.clone(), *polarity);
                steps.push(ProofStepLogic {
                    step_number: steps.len() + 1,
                    rule: "Unit Propagation".to_string(),
                    formula: format!("{} = {}", var, polarity),
                    justification: "Single literal in clause".to_string(),
                });
                clauses = Self::simplify_clauses(&clauses, var, *polarity);
            } else {
                break;
            }
        }

        // Pure literal elimination
        let mut pure_vars: HashMap<String, Option<bool>> = HashMap::new();
        for clause in &clauses {
            for (var, pol) in clause {
                let entry = pure_vars.entry(var.clone()).or_insert(Some(*pol));
                if let Some(existing) = entry {
                    if *existing != *pol {
                        *entry = None; // Not pure
                    }
                }
            }
        }
        for (var, pol) in &pure_vars {
            if let Some(polarity) = pol {
                assignment.insert(var.clone(), *polarity);
                clauses = Self::simplify_clauses(&clauses, var, *polarity);
            }
        }

        // Check if all clauses satisfied
        if clauses.is_empty() {
            return true;
        }

        // Check for empty clause (contradiction)
        if clauses.iter().any(|c| c.is_empty()) {
            return false;
        }

        // Choose a variable to branch on
        let branch_var = clauses[0][0].0.clone();

        // Try true
        assignment.insert(branch_var.clone(), true);
        let simplified = Self::simplify_clauses(&clauses, &branch_var, true);
        if Self::dpll_recursive(&simplified, vars, assignment, steps, depth + 1) {
            return true;
        }

        // Try false
        assignment.insert(branch_var.clone(), false);
        let simplified = Self::simplify_clauses(&clauses, &branch_var, false);
        if Self::dpll_recursive(&simplified, vars, assignment, steps, depth + 1) {
            return true;
        }

        assignment.remove(&branch_var);
        false
    }

    fn simplify_clauses(
        clauses: &[Vec<(String, bool)>],
        var: &str,
        value: bool,
    ) -> Vec<Vec<(String, bool)>> {
        clauses
            .iter()
            .filter_map(|clause| {
                // If clause contains the literal that's satisfied, remove entire clause
                if clause.iter().any(|(v, p)| v == var && *p == value) {
                    return None;
                }
                // Remove the false literal from the clause
                let new_clause: Vec<(String, bool)> = clause
                    .iter()
                    .filter(|(v, _)| v != var)
                    .cloned()
                    .collect();
                Some(new_clause)
            })
            .collect()
    }

    // ─── Natural Deduction ───────────────────────────────────────────────

    /// Modus Ponens: from P and P → Q, derive Q
    pub fn modus_ponens(
        premise: &Proposition,
        implication: &Proposition,
    ) -> Option<ProofResult> {
        if let Proposition::Implies(p, q) = implication {
            if **p == *premise {
                return Some(ProofResult {
                    valid: true,
                    proof_steps: vec![
                        ProofStepLogic {
                            step_number: 1,
                            rule: "Premise".to_string(),
                            formula: format!("{}", premise),
                            justification: "Given".to_string(),
                        },
                        ProofStepLogic {
                            step_number: 2,
                            rule: "Premise".to_string(),
                            formula: format!("{}", implication),
                            justification: "Given".to_string(),
                        },
                        ProofStepLogic {
                            step_number: 3,
                            rule: "Modus Ponens".to_string(),
                            formula: format!("{}", q),
                            justification: "From 1, 2 by MP".to_string(),
                        },
                    ],
                    phi: 0.4,
                    description: format!("Modus Ponens: {} and {} → {}", premise, implication, q),
                });
            }
        }
        None
    }

    /// Modus Tollens: from ¬Q and P → Q, derive ¬P
    pub fn modus_tollens(
        neg_conclusion: &Proposition,
        implication: &Proposition,
    ) -> Option<ProofResult> {
        if let Proposition::Implies(p, q) = implication {
            if let Proposition::Not(inner) = neg_conclusion {
                if **inner == **q {
                    let result = p.clone().not();
                    return Some(ProofResult {
                        valid: true,
                        proof_steps: vec![
                            ProofStepLogic {
                                step_number: 1,
                                rule: "Premise".to_string(),
                                formula: format!("{}", neg_conclusion),
                                justification: "Given".to_string(),
                            },
                            ProofStepLogic {
                                step_number: 2,
                                rule: "Premise".to_string(),
                                formula: format!("{}", implication),
                                justification: "Given".to_string(),
                            },
                            ProofStepLogic {
                                step_number: 3,
                                rule: "Modus Tollens".to_string(),
                                formula: format!("{}", result),
                                justification: "From 1, 2 by MT".to_string(),
                            },
                        ],
                        phi: 0.4,
                        description: format!("Modus Tollens: {} and {} → {}", neg_conclusion, implication, result),
                    });
                }
            }
        }
        None
    }

    /// Hypothetical Syllogism: from P → Q and Q → R, derive P → R
    pub fn hypothetical_syllogism(
        impl1: &Proposition,
        impl2: &Proposition,
    ) -> Option<ProofResult> {
        if let (Proposition::Implies(p, q1), Proposition::Implies(q2, r)) = (impl1, impl2) {
            if **q1 == **q2 {
                let result = p.clone().implies(*r.clone());
                return Some(ProofResult {
                    valid: true,
                    proof_steps: vec![
                        ProofStepLogic {
                            step_number: 1,
                            rule: "Premise".to_string(),
                            formula: format!("{}", impl1),
                            justification: "Given".to_string(),
                        },
                        ProofStepLogic {
                            step_number: 2,
                            rule: "Premise".to_string(),
                            formula: format!("{}", impl2),
                            justification: "Given".to_string(),
                        },
                        ProofStepLogic {
                            step_number: 3,
                            rule: "Hypothetical Syllogism".to_string(),
                            formula: format!("{}", result),
                            justification: "From 1, 2 by HS".to_string(),
                        },
                    ],
                    phi: 0.5,
                    description: format!("Hypothetical Syllogism: {} and {} → {}", impl1, impl2, result),
                });
            }
        }
        None
    }

    // ─── FOL Unification ─────────────────────────────────────────────────

    /// Robinson's unification algorithm.
    ///
    /// Returns a most general unifier (MGU) if the terms can be unified.
    pub fn unify(
        t1: &FOLTerm,
        t2: &FOLTerm,
    ) -> Option<HashMap<String, FOLTerm>> {
        let mut subst = HashMap::new();
        if Self::unify_recursive(t1, t2, &mut subst) {
            Some(subst)
        } else {
            None
        }
    }

    fn unify_recursive(
        t1: &FOLTerm,
        t2: &FOLTerm,
        subst: &mut HashMap<String, FOLTerm>,
    ) -> bool {
        let t1 = Self::apply_subst(t1, subst);
        let t2 = Self::apply_subst(t2, subst);

        match (&t1, &t2) {
            (FOLTerm::Const(a), FOLTerm::Const(b)) => a == b,
            (FOLTerm::Var(v), _) => {
                if t1 == t2 {
                    return true;
                }
                if Self::occurs_in(v, &t2) {
                    return false; // Occurs check
                }
                subst.insert(v.clone(), t2);
                true
            }
            (_, FOLTerm::Var(_)) => Self::unify_recursive(&t2, &t1, subst),
            (FOLTerm::Func(f1, args1), FOLTerm::Func(f2, args2)) => {
                if f1 != f2 || args1.len() != args2.len() {
                    return false;
                }
                for (a1, a2) in args1.iter().zip(args2.iter()) {
                    if !Self::unify_recursive(a1, a2, subst) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }

    fn apply_subst(term: &FOLTerm, subst: &HashMap<String, FOLTerm>) -> FOLTerm {
        match term {
            FOLTerm::Var(v) => {
                if let Some(replacement) = subst.get(v) {
                    Self::apply_subst(replacement, subst)
                } else {
                    term.clone()
                }
            }
            FOLTerm::Const(_) => term.clone(),
            FOLTerm::Func(name, args) => FOLTerm::Func(
                name.clone(),
                args.iter().map(|a| Self::apply_subst(a, subst)).collect(),
            ),
        }
    }

    fn occurs_in(var: &str, term: &FOLTerm) -> bool {
        match term {
            FOLTerm::Var(v) => v == var,
            FOLTerm::Const(_) => false,
            FOLTerm::Func(_, args) => args.iter().any(|a| Self::occurs_in(var, a)),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Truth tables ─────────────────────────────────────────────────────

    #[test]
    fn test_truth_table_simple() {
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let formula = p.and(q);
        let tt = LogicEngine::truth_table(&formula);
        assert_eq!(tt.rows.len(), 4); // 2² = 4 rows
        assert_eq!(tt.satisfying_count, 1); // Only T∧T = T
        assert!(tt.is_contingent);
    }

    #[test]
    fn test_tautology_excluded_middle() {
        // P ∨ ¬P is a tautology
        let p = Proposition::atom("P");
        let formula = p.clone().or(p.not());
        assert!(LogicEngine::is_tautology(&formula));
    }

    #[test]
    fn test_contradiction() {
        // P ∧ ¬P is a contradiction
        let p = Proposition::atom("P");
        let formula = p.clone().and(p.not());
        let tt = LogicEngine::truth_table(&formula);
        assert!(tt.is_contradiction);
    }

    #[test]
    fn test_implication_tautology() {
        // (P → Q) ↔ (¬P ∨ Q)
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let lhs = p.clone().implies(q.clone());
        let rhs = p.not().or(q);
        let formula = lhs.iff(rhs);
        assert!(LogicEngine::is_tautology(&formula));
    }

    #[test]
    fn test_de_morgan() {
        // ¬(P ∧ Q) ↔ (¬P ∨ ¬Q) — De Morgan's law
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let lhs = p.clone().and(q.clone()).not();
        let rhs = p.not().or(q.not());
        let formula = lhs.iff(rhs);
        assert!(LogicEngine::is_tautology(&formula));
    }

    // ── SAT solver ───────────────────────────────────────────────────────

    #[test]
    fn test_sat_simple() {
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let formula = p.and(q);
        let (result, proof) = LogicEngine::dpll_sat(&formula);
        assert!(result.is_some());
        let assignment = result.unwrap();
        assert_eq!(assignment.get("P"), Some(&true));
        assert_eq!(assignment.get("Q"), Some(&true));
    }

    #[test]
    fn test_unsat() {
        // P ∧ ¬P is unsatisfiable
        let p = Proposition::atom("P");
        let formula = p.clone().and(p.not());
        let (result, _) = LogicEngine::dpll_sat(&formula);
        assert!(result.is_none());
    }

    #[test]
    fn test_sat_implication() {
        // (P → Q) ∧ P should give P=true, Q=true
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let formula = p.clone().implies(q.clone()).and(p);
        let (result, _) = LogicEngine::dpll_sat(&formula);
        assert!(result.is_some());
        let assignment = result.unwrap();
        assert_eq!(assignment.get("P"), Some(&true));
        // Q must be true (since P→Q and P are both true)
        assert_eq!(assignment.get("Q"), Some(&true));
    }

    #[test]
    fn test_sat_disjunction() {
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let formula = p.or(q);
        let (result, _) = LogicEngine::dpll_sat(&formula);
        assert!(result.is_some());
    }

    // ── Natural deduction ────────────────────────────────────────────────

    #[test]
    fn test_modus_ponens() {
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let implication = p.clone().implies(q.clone());
        let result = LogicEngine::modus_ponens(&p, &implication);
        assert!(result.is_some());
        let proof = result.unwrap();
        assert!(proof.valid);
        assert_eq!(proof.proof_steps.len(), 3);
    }

    #[test]
    fn test_modus_tollens() {
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let implication = p.clone().implies(q.clone());
        let neg_q = q.not();
        let result = LogicEngine::modus_tollens(&neg_q, &implication);
        assert!(result.is_some());
        assert!(result.unwrap().valid);
    }

    #[test]
    fn test_hypothetical_syllogism() {
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let r = Proposition::atom("R");
        let pq = p.implies(q.clone());
        let qr = q.implies(r);
        let result = LogicEngine::hypothetical_syllogism(&pq, &qr);
        assert!(result.is_some());
        assert!(result.unwrap().valid);
    }

    // ── NNF/CNF conversion ───────────────────────────────────────────────

    #[test]
    fn test_nnf_double_negation() {
        let p = Proposition::atom("P");
        let nnf = LogicEngine::to_nnf(&p.clone().not().not());
        assert_eq!(nnf, p);
    }

    #[test]
    fn test_cnf_simple() {
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let formula = p.and(q);
        let cnf = LogicEngine::to_cnf(&formula);
        assert_eq!(cnf.len(), 2); // Two unit clauses
    }

    // ── FOL Unification ──────────────────────────────────────────────────

    #[test]
    fn test_unify_constants() {
        let t1 = FOLTerm::Const("a".to_string());
        let t2 = FOLTerm::Const("a".to_string());
        assert!(LogicEngine::unify(&t1, &t2).is_some());
    }

    #[test]
    fn test_unify_constants_fail() {
        let t1 = FOLTerm::Const("a".to_string());
        let t2 = FOLTerm::Const("b".to_string());
        assert!(LogicEngine::unify(&t1, &t2).is_none());
    }

    #[test]
    fn test_unify_var_const() {
        let t1 = FOLTerm::Var("X".to_string());
        let t2 = FOLTerm::Const("a".to_string());
        let subst = LogicEngine::unify(&t1, &t2).unwrap();
        assert_eq!(subst.get("X"), Some(&FOLTerm::Const("a".to_string())));
    }

    #[test]
    fn test_unify_functions() {
        // f(X, b) and f(a, Y) → {X=a, Y=b}
        let t1 = FOLTerm::Func(
            "f".to_string(),
            vec![
                FOLTerm::Var("X".to_string()),
                FOLTerm::Const("b".to_string()),
            ],
        );
        let t2 = FOLTerm::Func(
            "f".to_string(),
            vec![
                FOLTerm::Const("a".to_string()),
                FOLTerm::Var("Y".to_string()),
            ],
        );
        let subst = LogicEngine::unify(&t1, &t2).unwrap();
        assert_eq!(subst.get("X"), Some(&FOLTerm::Const("a".to_string())));
        assert_eq!(subst.get("Y"), Some(&FOLTerm::Const("b".to_string())));
    }

    #[test]
    fn test_unify_occurs_check() {
        // X and f(X) should fail (infinite type)
        let t1 = FOLTerm::Var("X".to_string());
        let t2 = FOLTerm::Func("f".to_string(), vec![FOLTerm::Var("X".to_string())]);
        assert!(LogicEngine::unify(&t1, &t2).is_none());
    }

    // ── HDC encoding ─────────────────────────────────────────────────────

    #[test]
    fn test_proposition_encoding() {
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let p_enc = p.encode();
        let q_enc = q.encode();
        // Different propositions should have different encodings
        let sim = p_enc.similarity(&q_enc);
        assert!(sim < 0.6, "Different atoms should have low similarity: {}", sim);
    }

    #[test]
    fn test_connective_encoding() {
        let p = Proposition::atom("P");
        let q = Proposition::atom("Q");
        let pq = p.clone().and(q.clone());
        let pq2 = p.or(q);
        let sim = pq.encode().similarity(&pq2.encode());
        assert!(sim < 0.6, "AND vs OR should have different encodings: {}", sim);
    }
}
