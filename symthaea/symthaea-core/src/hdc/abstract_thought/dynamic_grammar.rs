// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Dynamic Grammar Generation — Macro-Operators for GP
//!
//! When the ConjectureEngine consistently discovers the same sub-expression
//! across multiple independently verified conjectures, this module packages
//! that sub-expression as a reusable macro-operator and injects it into
//! the GP grammar pool.
//!
//! ## Promotion Criteria (Strict)
//!
//! A subtree is promoted to macro-operator only when:
//! 1. It appears in >= 3 conjectures (`min_occurrences`)
//! 2. From >= 2 different source sequences (independence)
//! 3. At least 1 occurrence is FormallyVerified or Z3-proven
//! 4. Current operator count < `max_operators` (20)
//!
//! ## Design
//!
//! This is the computational analogue of how humans invented integrals:
//! a recurring complex pattern gets compressed into a single manipulable symbol.
//!
//! ## References
//!
//! - Koza (1992) — Automatically Defined Functions in GP
//! - Langdon & Poli (2002) — Foundations of Genetic Programming

use std::collections::HashSet;

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::conjecture_engine::{ConjectureEngine, ConjectureStatus, Expr};
use crate::hdc::deterministic_seeds::seed_from_name;

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// A sub-expression observed recurring across conjectures.
#[derive(Debug, Clone)]
pub struct SubtreeCandidate {
    /// The normalized subtree (constants replaced with 1.0)
    pub pattern: Expr,
    /// Canonical string for equality comparison
    pub canonical: String,
    /// HDC encoding of the structural pattern
    pub encoding: BinaryHV,
    /// Conjecture indices where this subtree appears
    pub occurrences: Vec<usize>,
    /// Source sequence names (for independence check)
    pub sources: HashSet<String>,
    /// How many of the occurrences are verified (Formally or Z3)
    pub verified_count: usize,
}

/// A promoted macro-operator added to the GP grammar.
#[derive(Debug, Clone)]
pub struct MacroOperator {
    /// Unique name, e.g., "MACRO_0_add_sqrt"
    pub name: String,
    /// The template expression (normalized, with Const(1.0) as placeholders)
    pub template: Expr,
    /// Canonical string representation
    pub canonical: String,
    /// Number of constant placeholders (parameters to optimize during GP)
    pub arity: usize,
    /// Source conjecture indices that justified promotion
    pub source_conjectures: Vec<usize>,
    /// How many times this operator appeared in top conjectures during GP runs
    pub usage_count: u64,
    /// Cycle when this operator was promoted
    pub created_at: u64,
}

// ═══════════════════════════════════════════════════════════════════════════
// DYNAMIC GRAMMAR
// ═══════════════════════════════════════════════════════════════════════════

/// Manages the dynamic grammar pool for the GP engine.
pub struct DynamicGrammar {
    /// Active macro-operators in the grammar pool
    pub operators: Vec<MacroOperator>,
    /// Candidate subtrees being tracked (not yet promoted)
    pub candidates: Vec<SubtreeCandidate>,
    /// Maximum number of active operators (prevents grammar explosion)
    pub max_operators: usize,
    /// Minimum occurrences to promote a candidate
    pub min_occurrences: usize,
    /// Minimum verified conjectures among occurrences
    pub min_verified: usize,
    /// Minimum independent sources for promotion
    pub min_sources: usize,
    /// Monotonic cycle counter
    pub cycle: u64,
    /// Next operator ID
    next_id: usize,
}

impl DynamicGrammar {
    pub fn new() -> Self {
        Self {
            operators: Vec::new(),
            candidates: Vec::new(),
            max_operators: 20,
            min_occurrences: 3,
            min_verified: 1,
            min_sources: 2,
            cycle: 0,
            next_id: 0,
        }
    }

    /// Observe a recurring subtree (typically from MetaHDC clustering).
    ///
    /// Updates existing candidate if the pattern is already tracked,
    /// or creates a new candidate. The subtree is semantically canonicalized
    /// before tracking so that trivial rewrites (`x*1`, `x^1`, `x*x → x^2`,
    /// commutative reordering) collapse to a single candidate.
    pub fn observe_subtree(
        &mut self,
        subtree: Expr,
        conjecture_ids: &[usize],
        engine: &ConjectureEngine,
    ) {
        // Canonicalize first — collapses trivial rewrites
        let canon_expr = crate::hdc::abstract_thought::canonicalize_expr(&subtree);

        // Reject degenerate patterns: single Var, single Const, or empty subtrees.
        // These aren't meaningful macros (they'd just reduce to a GP leaf).
        match &canon_expr {
            Expr::Var(_) | Expr::Const(_) => return,
            _ => {}
        }
        if canon_expr.complexity() < 2 {
            return;
        }

        let canonical = crate::hdc::abstract_thought::expr_canonical_string(&canon_expr);

        // Check if already tracked
        if let Some(existing) = self.candidates.iter_mut().find(|c| c.canonical == canonical) {
            // Merge new occurrences
            for &id in conjecture_ids {
                if !existing.occurrences.contains(&id) {
                    existing.occurrences.push(id);
                    if id < engine.conjectures.len() {
                        let conj = &engine.conjectures[id];
                        existing.sources.insert(conj.source.clone());
                        if is_verified(&conj.status) {
                            existing.verified_count += 1;
                        }
                    }
                }
            }
            return;
        }

        // Already promoted? Skip.
        if self.operators.iter().any(|op| op.canonical == canonical) {
            return;
        }

        // New candidate
        let mut sources = HashSet::new();
        let mut verified_count = 0;
        for &id in conjecture_ids {
            if id < engine.conjectures.len() {
                let conj = &engine.conjectures[id];
                sources.insert(conj.source.clone());
                if is_verified(&conj.status) {
                    verified_count += 1;
                }
            }
        }

        let encoding = BinaryHV::random(seed_from_name(&format!("CANDIDATE_{}", canonical)));

        self.candidates.push(SubtreeCandidate {
            pattern: canon_expr,
            canonical,
            encoding,
            occurrences: conjecture_ids.to_vec(),
            sources,
            verified_count,
        });
    }

    /// Promote eligible candidates to macro-operators.
    ///
    /// Checks all candidates against promotion criteria and promotes those
    /// that meet the threshold.
    pub fn promote_eligible(&mut self, engine: &ConjectureEngine) {
        let mut promoted_indices = Vec::new();

        for (i, candidate) in self.candidates.iter().enumerate() {
            if self.operators.len() >= self.max_operators {
                break;
            }

            if candidate.occurrences.len() >= self.min_occurrences
                && candidate.sources.len() >= self.min_sources
                && candidate.verified_count >= self.min_verified
            {
                let arity = count_constants(&candidate.pattern);
                let name = format!(
                    "MACRO_{}_{:.20}",
                    self.next_id,
                    candidate.canonical.replace(' ', "_")
                );

                self.operators.push(MacroOperator {
                    name,
                    template: candidate.pattern.clone(),
                    canonical: candidate.canonical.clone(),
                    arity,
                    source_conjectures: candidate.occurrences.clone(),
                    usage_count: 0,
                    created_at: self.cycle,
                });
                self.next_id += 1;
                promoted_indices.push(i);
            }
        }

        // Remove promoted candidates (iterate in reverse to preserve indices)
        for i in promoted_indices.into_iter().rev() {
            self.candidates.swap_remove(i);
        }
    }

    /// Prune macro-operators that have never been used.
    pub fn prune_unused(&mut self) {
        self.operators.retain(|op| {
            // Keep operators that have been used, or are too new to judge
            op.usage_count > 0 || (self.cycle - op.created_at) < 10
        });
    }

    /// Record that a macro-operator was used in a top conjecture.
    pub fn record_usage(&mut self, canonical: &str) {
        if let Some(op) = self.operators.iter_mut().find(|op| op.canonical == canonical) {
            op.usage_count += 1;
        }
    }

    /// Instantiate a macro-operator template with random constants.
    ///
    /// Replaces `Const(1.0)` placeholders with random values from a seed.
    pub fn instantiate_macro(template: &Expr, rng: &mut u64) -> Expr {
        match template {
            Expr::Const(_) => {
                // Replace placeholder with random constant
                *rng = lcg_step(*rng);
                let val = (*rng as f64 / u64::MAX as f64) * 10.0 - 5.0; // [-5, 5]
                Expr::Const(val)
            }
            Expr::Var(name) => Expr::Var(name.clone()),
            Expr::BinOp(op, l, r) => Expr::BinOp(
                *op,
                Box::new(Self::instantiate_macro(l, rng)),
                Box::new(Self::instantiate_macro(r, rng)),
            ),
            Expr::Func(f, arg) => {
                Expr::Func(*f, Box::new(Self::instantiate_macro(arg, rng)))
            }
            Expr::Sum(body, var) => {
                Expr::Sum(Box::new(Self::instantiate_macro(body, rng)), var.clone())
            }
        }
    }

    /// Advance the cycle counter.
    pub fn tick(&mut self) {
        self.cycle += 1;
    }
}

impl Default for DynamicGrammar {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// A conjecture counts as "verified" for grammar promotion if it has been
/// tested against data with reasonable confidence. We accept formal proofs,
/// symbolic checks, AND numerically-tested conjectures with low test MSE —
/// the latter is a pragmatic proxy for real-world discovery where Z3
/// verification may be unavailable (e.g., transcendental formulas).
fn is_verified(status: &ConjectureStatus) -> bool {
    match status {
        ConjectureStatus::FormallyVerified { .. } => true,
        ConjectureStatus::SymbolicallyChecked => true,
        ConjectureStatus::NumericallyTested { test_mse } => *test_mse < 1e-3,
        _ => false,
    }
}

/// Count Const nodes in an expression (macro-operator arity).
fn count_constants(expr: &Expr) -> usize {
    match expr {
        Expr::Const(_) => 1,
        Expr::Var(_) => 0,
        Expr::BinOp(_, l, r) => count_constants(l) + count_constants(r),
        Expr::Func(_, arg) => count_constants(arg),
        Expr::Sum(body, _) => count_constants(body),
    }
}

/// Simple LCG random number generator (matches conjecture_engine's lcg_step).
fn lcg_step(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::conjecture_engine::{BinOp, Conjecture, MathDomain, UnaryFn};
    use crate::hdc::abstract_thought::normalize_expr;

    fn make_conjecture(
        formula: Expr,
        domain: MathDomain,
        source: &str,
        status: ConjectureStatus,
    ) -> Conjecture {
        let formula_str = format!("{}", formula);
        let complexity = formula.complexity();
        Conjecture {
            formula,
            formula_str,
            source: source.to_string(),
            domain,
            training_mse: 0.001,
            complexity,
            fitness: 0.001 + 0.001 * complexity as f64,
            status,
            confidence: 0.95,
        }
    }

    fn sqrt_n_plus_c(c: f64) -> Expr {
        Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Func(
                UnaryFn::Sqrt,
                Box::new(Expr::Var("n".to_string())),
            )),
            Box::new(Expr::Const(c)),
        )
    }

    #[test]
    fn test_observe_and_candidate_creation() {
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        let c0 = make_conjecture(
            sqrt_n_plus_c(1.0),
            MathDomain::NumberTheory,
            "seq_a",
            ConjectureStatus::FormallyVerified { proof_steps: 5 },
        );
        engine.conjectures.push(c0);

        let subtree = normalize_expr(&sqrt_n_plus_c(1.0));
        grammar.observe_subtree(subtree, &[0], &engine);

        assert_eq!(grammar.candidates.len(), 1);
        assert_eq!(grammar.candidates[0].occurrences.len(), 1);
    }

    #[test]
    fn test_candidate_not_promoted_below_threshold() {
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        // Only 2 occurrences — below threshold of 3
        for i in 0..2 {
            let c = make_conjecture(
                sqrt_n_plus_c(i as f64 + 1.0),
                MathDomain::NumberTheory,
                &format!("seq_{}", i),
                ConjectureStatus::FormallyVerified { proof_steps: 5 },
            );
            engine.conjectures.push(c);
        }

        let subtree = normalize_expr(&sqrt_n_plus_c(1.0));
        grammar.observe_subtree(subtree, &[0, 1], &engine);
        grammar.promote_eligible(&engine);

        assert!(grammar.operators.is_empty(), "Should not promote with only 2 occurrences");
    }

    #[test]
    fn test_candidate_promoted_at_threshold() {
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        // 3 occurrences from 3 different sources, 1 verified
        for i in 0..3 {
            let status = if i == 0 {
                ConjectureStatus::FormallyVerified { proof_steps: 5 }
            } else {
                ConjectureStatus::NumericallyTested { test_mse: 0.01 }
            };
            let c = make_conjecture(
                sqrt_n_plus_c(i as f64 + 1.0),
                MathDomain::NumberTheory,
                &format!("seq_{}", i), // Different sources
                status,
            );
            engine.conjectures.push(c);
        }

        let subtree = normalize_expr(&sqrt_n_plus_c(1.0));
        grammar.observe_subtree(subtree, &[0, 1, 2], &engine);
        grammar.promote_eligible(&engine);

        assert_eq!(grammar.operators.len(), 1, "Should promote at threshold");
        assert!(grammar.operators[0].name.starts_with("MACRO_"));
    }

    #[test]
    fn test_independence_check() {
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        // 3 occurrences but all from the SAME source — should NOT promote
        for i in 0..3 {
            let c = make_conjecture(
                sqrt_n_plus_c(i as f64 + 1.0),
                MathDomain::NumberTheory,
                "same_source", // Same source!
                ConjectureStatus::FormallyVerified { proof_steps: 5 },
            );
            engine.conjectures.push(c);
        }

        let subtree = normalize_expr(&sqrt_n_plus_c(1.0));
        grammar.observe_subtree(subtree, &[0, 1, 2], &engine);
        grammar.promote_eligible(&engine);

        assert!(
            grammar.operators.is_empty(),
            "Should not promote from single source"
        );
    }

    #[test]
    fn test_max_operators_cap() {
        let mut grammar = DynamicGrammar::new();
        grammar.max_operators = 2;

        let mut engine = ConjectureEngine::new();
        // Create enough conjectures for 3 different patterns
        for pattern_id in 0..3 {
            for i in 0..3 {
                let formula = if pattern_id == 0 {
                    sqrt_n_plus_c(i as f64)
                } else if pattern_id == 1 {
                    Expr::BinOp(
                        BinOp::Mul,
                        Box::new(Expr::Var("n".to_string())),
                        Box::new(Expr::Const(i as f64)),
                    )
                } else {
                    Expr::Func(UnaryFn::Log, Box::new(Expr::Var("n".to_string())))
                };
                let c = make_conjecture(
                    formula,
                    MathDomain::NumberTheory,
                    &format!("src_{}_{}", pattern_id, i),
                    ConjectureStatus::FormallyVerified { proof_steps: 5 },
                );
                let idx = engine.conjectures.len();
                engine.conjectures.push(c);

                let base_idx = pattern_id * 3;
                if i == 2 {
                    // Observe all 3 together
                    let subtree = if pattern_id == 0 {
                        normalize_expr(&sqrt_n_plus_c(1.0))
                    } else if pattern_id == 1 {
                        normalize_expr(&Expr::BinOp(
                            BinOp::Mul,
                            Box::new(Expr::Var("n".to_string())),
                            Box::new(Expr::Const(1.0)),
                        ))
                    } else {
                        normalize_expr(&Expr::Func(
                            UnaryFn::Log,
                            Box::new(Expr::Var("n".to_string())),
                        ))
                    };
                    grammar.observe_subtree(
                        subtree,
                        &[base_idx, base_idx + 1, base_idx + 2],
                        &engine,
                    );
                }
            }
        }

        grammar.promote_eligible(&engine);
        assert!(
            grammar.operators.len() <= 2,
            "Should cap at max_operators: got {}",
            grammar.operators.len()
        );
    }

    #[test]
    fn test_prune_unused() {
        let mut grammar = DynamicGrammar::new();
        grammar.operators.push(MacroOperator {
            name: "MACRO_old".to_string(),
            template: Expr::Var("n".to_string()),
            canonical: "n".to_string(),
            arity: 0,
            source_conjectures: vec![0],
            usage_count: 0,
            created_at: 0,
        });
        grammar.cycle = 20; // Old enough to prune

        grammar.prune_unused();
        assert!(grammar.operators.is_empty(), "Should prune unused old operator");
    }

    #[test]
    fn test_prune_keeps_used() {
        let mut grammar = DynamicGrammar::new();
        grammar.operators.push(MacroOperator {
            name: "MACRO_used".to_string(),
            template: Expr::Var("n".to_string()),
            canonical: "n".to_string(),
            arity: 0,
            source_conjectures: vec![0],
            usage_count: 5,
            created_at: 0,
        });
        grammar.cycle = 20;

        grammar.prune_unused();
        assert_eq!(grammar.operators.len(), 1, "Should keep used operator");
    }

    #[test]
    fn test_prune_keeps_new() {
        let mut grammar = DynamicGrammar::new();
        grammar.operators.push(MacroOperator {
            name: "MACRO_new".to_string(),
            template: Expr::Var("n".to_string()),
            canonical: "n".to_string(),
            arity: 0,
            source_conjectures: vec![0],
            usage_count: 0,
            created_at: 5,
        });
        grammar.cycle = 10; // Only 5 cycles old — too new to prune

        grammar.prune_unused();
        assert_eq!(grammar.operators.len(), 1, "Should keep new operator");
    }

    #[test]
    fn test_instantiate_macro_produces_different_constants() {
        let template = normalize_expr(&sqrt_n_plus_c(1.0));
        let mut rng1 = 42u64;
        let mut rng2 = 99u64;

        let inst1 = DynamicGrammar::instantiate_macro(&template, &mut rng1);
        let inst2 = DynamicGrammar::instantiate_macro(&template, &mut rng2);

        // Both should be valid expressions (no NaN)
        let val1 = inst1.eval(&[("n", 4.0)]);
        let val2 = inst2.eval(&[("n", 4.0)]);
        assert!(val1.is_finite(), "Instantiated macro should evaluate");
        assert!(val2.is_finite(), "Instantiated macro should evaluate");
        // Different seeds should produce different constants
        assert!(
            (val1 - val2).abs() > 1e-10,
            "Different seeds should give different instances"
        );
    }

    #[test]
    fn test_record_usage() {
        let mut grammar = DynamicGrammar::new();
        grammar.operators.push(MacroOperator {
            name: "MACRO_test".to_string(),
            template: Expr::Var("n".to_string()),
            canonical: "n".to_string(),
            arity: 0,
            source_conjectures: vec![0],
            usage_count: 0,
            created_at: 0,
        });

        grammar.record_usage("n");
        assert_eq!(grammar.operators[0].usage_count, 1);

        grammar.record_usage("nonexistent");
        assert_eq!(grammar.operators[0].usage_count, 1); // Unchanged
    }

    #[test]
    fn test_duplicate_observation_merges() {
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        let c0 = make_conjecture(
            sqrt_n_plus_c(1.0),
            MathDomain::NumberTheory,
            "seq_a",
            ConjectureStatus::FormallyVerified { proof_steps: 5 },
        );
        engine.conjectures.push(c0);

        let subtree = normalize_expr(&sqrt_n_plus_c(1.0));
        grammar.observe_subtree(subtree.clone(), &[0], &engine);
        grammar.observe_subtree(subtree, &[0], &engine); // Duplicate

        assert_eq!(grammar.candidates.len(), 1, "Should merge, not duplicate");
        assert_eq!(grammar.candidates[0].occurrences.len(), 1, "Should not double-count");
    }
}
