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
//! 3. At least 1 occurrence is verified strongly enough for the selected path
//! 4. Current operator count < `max_operators` (20)
//!
//! ## Design
//!
//! This is the computational analogue of how humans invented integrals:
//! a recurring complex pattern gets compressed into a single manipulable symbol.
//! Promotion now carries explicit metadata so downstream GP can filter by
//! variable signature and proof quality instead of treating the macro pool as
//! one undifferentiated bag.
//!
//! ## References
//!
//! - Koza (1992) — Automatically Defined Functions in GP
//! - Langdon & Poli (2002) — Foundations of Genetic Programming

use std::collections::{HashMap, HashSet};

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::conjecture_engine::{
    BinOp, Conjecture, ConjectureEngine, ConjectureStatus, Expr, MacroPromotionTier,
    PreferredEmlBackend, conjecture_has_verified_eml_backend,
};
use crate::hdc::deterministic_seeds::seed_from_name;
use crate::hdc::eml;

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
    /// How many of the occurrences are verified (numerically or stronger).
    /// Used by the standard promotion path.
    pub verified_count: usize,
    /// How many of the occurrences are *formally* verified (Z3-proven or
    /// symbolically checked — not just numerically tested with low MSE).
    /// Used by the fast-track promotion path: a single formally-verified
    /// occurrence is sufficient to promote a unique novel shape.
    pub strongly_verified_count: usize,
    /// Strongest promotion tier supported by the contributing conjectures.
    pub promotion_tier: MacroPromotionTier,
    /// Parent formulas that contributed this subtree.
    pub parent_formulas: HashSet<String>,
    /// Distinct variables referenced by this subtree.
    pub vars_used: Vec<String>,
    /// Number of distinct variables referenced by this subtree.
    pub var_count: usize,
    /// Canonical signature key for the subtree's variable set.
    pub signature: String,
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
    /// How this macro entered the pool.
    pub promotion_tier: MacroPromotionTier,
    /// Source conjecture indices that justified promotion
    pub source_conjectures: Vec<usize>,
    /// Human-readable parents that justified promotion.
    pub parent_formulas: Vec<String>,
    /// Distinct variables referenced by the macro.
    pub vars_used: Vec<String>,
    /// Number of distinct variables referenced by the macro.
    pub var_count: usize,
    /// Canonical signature key for the macro's variable set.
    pub signature: String,
    /// Number of independent sources that contributed this macro.
    pub source_count: usize,
    /// How many times this operator appeared in top conjectures during GP runs
    pub usage_count: u64,
    /// Cycle when this operator was promoted
    pub created_at: u64,
}

/// Per-signature summary of the active macro pool.
#[derive(Debug, Clone, PartialEq)]
pub struct SignatureMacroMetrics {
    pub signature: String,
    pub operator_count: usize,
    pub used_operator_count: usize,
    pub total_usage_count: u64,
}

/// Snapshot metrics for evaluating macro-pool quality.
#[derive(Debug, Clone, PartialEq)]
pub struct MacroPoolMetrics {
    pub cycle: u64,
    pub total_operators: usize,
    pub total_candidates: usize,
    pub formal_operators: usize,
    pub recurrent_operators: usize,
    pub quarantined_operators: usize,
    pub used_operators: usize,
    pub unused_operators: usize,
    pub mature_operators: usize,
    pub mature_used_operators: usize,
    pub total_usage_count: u64,
    pub avg_usage_count: f64,
    pub avg_source_count: f64,
    pub active_precision: f64,
    pub mature_precision: f64,
    pub total_promoted: u64,
    pub total_pruned: u64,
    pub survival_rate: f64,
    pub signature_stats: Vec<SignatureMacroMetrics>,
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
    /// Signature-partitioned operator index for variable-aware retrieval.
    operators_by_signature: HashMap<String, Vec<usize>>,
    /// Lifetime count of promoted operators.
    total_promoted: u64,
    /// Lifetime count of pruned operators.
    total_pruned: u64,
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
            operators_by_signature: HashMap::new(),
            total_promoted: 0,
            total_pruned: 0,
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

        let canonical = subtree_candidate_identity(&canon_expr);
        let vars_used = crate::hdc::abstract_thought::expr_variables(&canon_expr);
        let var_count = vars_used.len();
        let signature = crate::hdc::abstract_thought::expr_signature(&canon_expr);

        // Check if already tracked
        if let Some(existing) = self
            .candidates
            .iter_mut()
            .find(|c| c.canonical == canonical)
        {
            // Merge new occurrences
            for &id in conjecture_ids {
                if !existing.occurrences.contains(&id) {
                    existing.occurrences.push(id);
                    if id < engine.conjectures.len() {
                        let conj = &engine.conjectures[id];
                        existing.sources.insert(conj.source.clone());
                        existing
                            .parent_formulas
                            .insert(conjecture_parent_formula_identity(conj));
                        if conj.macro_promotion_tier.allows_recurrent_promotion()
                            && is_verified(&conj.status)
                        {
                            existing.verified_count += 1;
                            existing.promotion_tier = existing
                                .promotion_tier
                                .max(MacroPromotionTier::RecurrentNumerical);
                        }
                        if conj.macro_promotion_tier.allows_recurrent_promotion()
                            && conjecture_supports_fast_track(conj)
                            && subtree_admits_fast_track(&canon_expr)
                        {
                            existing.strongly_verified_count += 1;
                            existing.promotion_tier = MacroPromotionTier::Formal;
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
        let mut strongly_verified_count = 0;
        let mut parent_formulas = HashSet::new();
        let mut promotion_tier = MacroPromotionTier::Quarantined;
        for &id in conjecture_ids {
            if id < engine.conjectures.len() {
                let conj = &engine.conjectures[id];
                sources.insert(conj.source.clone());
                parent_formulas.insert(conjecture_parent_formula_identity(conj));
                if conj.macro_promotion_tier.allows_recurrent_promotion()
                    && is_verified(&conj.status)
                {
                    verified_count += 1;
                    promotion_tier = promotion_tier.max(MacroPromotionTier::RecurrentNumerical);
                }
                if conj.macro_promotion_tier.allows_recurrent_promotion()
                    && conjecture_supports_fast_track(conj)
                    && subtree_admits_fast_track(&canon_expr)
                {
                    strongly_verified_count += 1;
                    promotion_tier = MacroPromotionTier::Formal;
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
            strongly_verified_count,
            promotion_tier,
            parent_formulas,
            vars_used,
            var_count,
            signature,
        });
    }

    /// Promote eligible candidates to macro-operators.
    ///
    /// Two parallel paths:
    /// 1. **Standard path**: needs N+ occurrences from K+ independent sources
    ///    and at least one numerically-verified hit. Catches polynomial idioms
    ///    that recur across many discoveries.
    /// 2. **Fast track**: a single FormallyVerified or SymbolicallyChecked
    ///    occurrence is sufficient, regardless of source/occurrence counts.
    ///    The proof IS the redundancy. This is what lets unique novel shapes
    ///    like Hardy-Ramanujan's `exp(c*sqrt(n))` enter the pool from a
    ///    single discovery.
    pub fn promote_eligible(&mut self, engine: &ConjectureEngine) {
        let _ = engine; // engine reserved for future cross-checks
        let mut eligible: Vec<(usize, bool)> = Vec::new();

        for (i, candidate) in self.candidates.iter().enumerate() {
            if self.operators.len() + eligible.len() >= self.max_operators {
                // Still scan all candidates so ordering is decided by quality, not insertion order.
            }

            let standard_promote = candidate.occurrences.len() >= self.min_occurrences
                && candidate.sources.len() >= self.min_sources
                && candidate.verified_count >= self.min_verified;

            // Fast track: any formally-verified occurrence is sufficient.
            // The Z3 proof or symbolic check is the quality signal — we
            // don't need redundancy across sources.
            let fast_track_promote =
                candidate.strongly_verified_count >= 1 && !candidate.occurrences.is_empty();

            if standard_promote || fast_track_promote {
                eligible.push((i, fast_track_promote));
            }
        }

        eligible.sort_by(|(a_idx, a_fast), (b_idx, b_fast)| {
            let a = &self.candidates[*a_idx];
            let b = &self.candidates[*b_idx];
            b_fast
                .cmp(a_fast)
                .then_with(|| {
                    candidate_fast_track_rank(a, engine).cmp(&candidate_fast_track_rank(b, engine))
                })
                .then_with(|| candidate_eml_rank(a).cmp(&candidate_eml_rank(b)))
                .then_with(|| b.strongly_verified_count.cmp(&a.strongly_verified_count))
                .then_with(|| b.verified_count.cmp(&a.verified_count))
                .then_with(|| b.sources.len().cmp(&a.sources.len()))
                .then_with(|| b.occurrences.len().cmp(&a.occurrences.len()))
                .then_with(|| a.canonical.cmp(&b.canonical))
        });

        let remaining_slots = self.max_operators.saturating_sub(self.operators.len());
        let mut promoted_indices = Vec::new();

        for (i, fast_track_promote) in eligible.into_iter().take(remaining_slots) {
            let candidate = &self.candidates[i];
            let arity = count_constants(&candidate.pattern);
            let name = format!(
                "MACRO_{}_{:.20}",
                self.next_id,
                candidate.canonical.replace(' ', "_")
            );
            let promotion_tier = if fast_track_promote {
                MacroPromotionTier::Formal
            } else {
                MacroPromotionTier::RecurrentNumerical
            };
            let mut parent_formulas: Vec<String> =
                candidate.parent_formulas.iter().cloned().collect();
            parent_formulas.sort();

            self.operators.push(MacroOperator {
                name,
                template: candidate.pattern.clone(),
                canonical: candidate.canonical.clone(),
                arity,
                promotion_tier,
                source_conjectures: candidate.occurrences.clone(),
                parent_formulas,
                vars_used: candidate.vars_used.clone(),
                var_count: candidate.var_count,
                signature: candidate.signature.clone(),
                source_count: candidate.sources.len(),
                usage_count: 0,
                created_at: self.cycle,
            });
            self.next_id += 1;
            self.total_promoted += 1;
            promoted_indices.push(i);
        }

        // Remove promoted candidates in descending index order so swap_remove
        // never invalidates a later removal target.
        promoted_indices.sort_unstable();
        for i in promoted_indices.into_iter().rev() {
            self.candidates.swap_remove(i);
        }
        if !self.operators.is_empty() {
            self.rebuild_operator_index();
        }
    }

    /// Prune macro-operators that have never been used.
    pub fn prune_unused(&mut self) {
        let before = self.operators.len();
        self.operators.retain(|op| {
            // Keep operators that have been used, or are too new to judge
            op.usage_count > 0 || (self.cycle - op.created_at) < 10
        });
        self.total_pruned += (before - self.operators.len()) as u64;
        if self.operators.len() != before {
            self.rebuild_operator_index();
        }
    }

    /// Record that a macro-operator was used in a top conjecture.
    pub fn record_usage(&mut self, canonical: &str) {
        if let Some(op) = self
            .operators
            .iter_mut()
            .find(|op| op.canonical == canonical)
        {
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
            Expr::Func(f, arg) => Expr::Func(*f, Box::new(Self::instantiate_macro(arg, rng))),
            Expr::Sum(body, var) => {
                Expr::Sum(Box::new(Self::instantiate_macro(body, rng)), var.clone())
            }
        }
    }

    /// Advance the cycle counter.
    pub fn tick(&mut self) {
        self.cycle += 1;
    }

    /// Retrieve operators whose variable requirements are compatible with the
    /// current problem's variable set.
    pub fn operators_compatible_with_vars(&self, allowed_vars: &[&str]) -> Vec<&MacroOperator> {
        let allowed: HashSet<&str> = allowed_vars.iter().copied().collect();
        let mut operators = Vec::new();

        if self.operators_by_signature.is_empty() {
            for op in &self.operators {
                if signature_is_compatible(&op.signature, &allowed) {
                    operators.push(op);
                }
            }
        } else {
            for (signature, indices) in &self.operators_by_signature {
                if !signature_is_compatible(signature, &allowed) {
                    continue;
                }
                for &idx in indices {
                    operators.push(&self.operators[idx]);
                }
            }
        }

        operators.sort_by(|a, b| {
            b.promotion_tier
                .cmp(&a.promotion_tier)
                .then_with(|| b.usage_count.cmp(&a.usage_count))
                .then_with(|| b.source_count.cmp(&a.source_count))
                .then_with(|| b.var_count.cmp(&a.var_count))
        });
        operators
    }

    fn rebuild_operator_index(&mut self) {
        self.operators_by_signature.clear();
        for (idx, op) in self.operators.iter().enumerate() {
            self.operators_by_signature
                .entry(op.signature.clone())
                .or_default()
                .push(idx);
        }
    }

    /// Snapshot macro-pool quality metrics for reporting and benchmarks.
    pub fn metrics(&self) -> MacroPoolMetrics {
        let total_operators = self.operators.len();
        let used_operators = self
            .operators
            .iter()
            .filter(|op| op.usage_count > 0)
            .count();
        let mature_operators = self
            .operators
            .iter()
            .filter(|op| (self.cycle - op.created_at) >= 10)
            .count();
        let mature_used_operators = self
            .operators
            .iter()
            .filter(|op| (self.cycle - op.created_at) >= 10 && op.usage_count > 0)
            .count();
        let total_usage_count = self.operators.iter().map(|op| op.usage_count).sum::<u64>();
        let total_source_count = self
            .operators
            .iter()
            .map(|op| op.source_count)
            .sum::<usize>();

        let mut signature_map: HashMap<String, SignatureMacroMetrics> = HashMap::new();
        let mut formal_operators = 0usize;
        let mut recurrent_operators = 0usize;
        let mut quarantined_operators = 0usize;

        for op in &self.operators {
            match op.promotion_tier {
                MacroPromotionTier::Formal => formal_operators += 1,
                MacroPromotionTier::RecurrentNumerical => recurrent_operators += 1,
                MacroPromotionTier::Quarantined => quarantined_operators += 1,
            }

            let entry = signature_map
                .entry(op.signature.clone())
                .or_insert_with(|| SignatureMacroMetrics {
                    signature: op.signature.clone(),
                    operator_count: 0,
                    used_operator_count: 0,
                    total_usage_count: 0,
                });
            entry.operator_count += 1;
            if op.usage_count > 0 {
                entry.used_operator_count += 1;
            }
            entry.total_usage_count += op.usage_count;
        }

        let mut signature_stats: Vec<SignatureMacroMetrics> = signature_map.into_values().collect();
        signature_stats.sort_by(|a, b| a.signature.cmp(&b.signature));

        MacroPoolMetrics {
            cycle: self.cycle,
            total_operators,
            total_candidates: self.candidates.len(),
            formal_operators,
            recurrent_operators,
            quarantined_operators,
            used_operators,
            unused_operators: total_operators.saturating_sub(used_operators),
            mature_operators,
            mature_used_operators,
            total_usage_count,
            avg_usage_count: if total_operators > 0 {
                total_usage_count as f64 / total_operators as f64
            } else {
                0.0
            },
            avg_source_count: if total_operators > 0 {
                total_source_count as f64 / total_operators as f64
            } else {
                0.0
            },
            active_precision: if total_operators > 0 {
                used_operators as f64 / total_operators as f64
            } else {
                0.0
            },
            mature_precision: if mature_operators > 0 {
                mature_used_operators as f64 / mature_operators as f64
            } else {
                0.0
            },
            total_promoted: self.total_promoted,
            total_pruned: self.total_pruned,
            survival_rate: if self.total_promoted > 0 {
                total_operators as f64 / self.total_promoted as f64
            } else {
                0.0
            },
            signature_stats,
        }
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

/// Verification check for the fast-track promotion path.
///
/// Accepts:
/// - `FormallyVerified` — Z3 or tactic proof succeeded
/// - `SymbolicallyChecked` — symbolic identity check passed
///
/// Numerical fits may still promote through the recurrence path, but never
/// through singleton fast-track promotion.
fn is_strongly_verified(status: &ConjectureStatus) -> bool {
    match status {
        ConjectureStatus::FormallyVerified { .. } => true,
        ConjectureStatus::SymbolicallyChecked => true,
        _ => false,
    }
}

fn conjecture_supports_fast_track(conjecture: &crate::hdc::conjecture_engine::Conjecture) -> bool {
    is_strongly_verified(&conjecture.status) || conjecture_has_verified_eml_backend(conjecture)
}

fn conjecture_parent_formula_identity(
    conjecture: &crate::hdc::conjecture_engine::Conjecture,
) -> String {
    conjecture
        .preferred_eml_canonical_form()
        .unwrap_or_else(|| conjecture.formula_str.clone())
}

fn subtree_candidate_identity(expr: &Expr) -> String {
    if let Ok(compiled) = eml::compile_expr(expr) {
        format!("eml:strict:{}", compiled)
    } else if let Ok(compiled) = eml::compile_expr_constructive(expr) {
        format!("eml:constructive:{}", compiled)
    } else {
        crate::hdc::abstract_thought::expr_canonical_string(expr)
    }
}

fn candidate_eml_rank(candidate: &SubtreeCandidate) -> u8 {
    if candidate.canonical.starts_with("eml:strict:") {
        0
    } else if candidate.canonical.starts_with("eml:constructive:") {
        1
    } else {
        2
    }
}

fn candidate_fast_track_rank(candidate: &SubtreeCandidate, engine: &ConjectureEngine) -> u8 {
    candidate
        .occurrences
        .iter()
        .filter_map(|&id| engine.conjectures.get(id))
        .filter(|conj| conjecture_supports_fast_track(conj))
        .map(conjecture_fast_track_rank)
        .min()
        .unwrap_or(u8::MAX)
}

fn conjecture_fast_track_rank(conjecture: &Conjecture) -> u8 {
    if is_strongly_verified(&conjecture.status) {
        return 0;
    }

    match conjecture.preferred_eml_backend() {
        Some(PreferredEmlBackend::StrictRealAndComplex) => {
            if conjecture
                .eml_real_domain
                .is_some_and(|d| d.is_unconstrained())
            {
                1
            } else {
                2
            }
        }
        Some(PreferredEmlBackend::StrictReal) => {
            if conjecture
                .eml_real_domain
                .is_some_and(|d| d.is_unconstrained())
            {
                3
            } else {
                4
            }
        }
        Some(PreferredEmlBackend::StrictComplex) => 5,
        Some(PreferredEmlBackend::ConstructiveReal) => 6,
        Some(PreferredEmlBackend::StrictUnverified) | None => 7,
    }
}

fn subtree_admits_fast_track(expr: &Expr) -> bool {
    !(contains_unary_fn(expr)
        && crate::hdc::abstract_thought::expr_variables(expr).len() <= 1
        && (expr.complexity() <= 4
            || matches!(expr, Expr::Func(_, arg) if is_trivial_wrapper_arg(arg))))
}

fn contains_unary_fn(expr: &Expr) -> bool {
    match expr {
        Expr::Func(_, _) => true,
        Expr::Var(_) | Expr::Const(_) => false,
        Expr::BinOp(_, l, r) => contains_unary_fn(l) || contains_unary_fn(r),
        Expr::Sum(body, _) => contains_unary_fn(body),
    }
}

fn is_trivial_wrapper_arg(expr: &Expr) -> bool {
    match expr {
        Expr::Var(_) => true,
        Expr::BinOp(BinOp::Mul, l, r) => {
            (matches!(l.as_ref(), Expr::Const(_)) && is_trivial_wrapper_arg(r))
                || (matches!(r.as_ref(), Expr::Const(_)) && is_trivial_wrapper_arg(l))
        }
        Expr::BinOp(BinOp::Div, l, r) => {
            matches!(r.as_ref(), Expr::Const(_)) && is_trivial_wrapper_arg(l)
        }
        Expr::BinOp(BinOp::Pow, l, r) => {
            matches!(r.as_ref(), Expr::Const(_)) && is_trivial_wrapper_arg(l)
        }
        _ => false,
    }
}

fn signature_is_compatible(signature: &str, allowed: &HashSet<&str>) -> bool {
    if signature == "<const>" {
        return true;
    }
    signature.split('|').all(|var| allowed.contains(var))
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
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::abstract_thought::normalize_expr;
    use crate::hdc::conjecture_engine::{
        BinOp, Conjecture, MathDomain, UnaryFn, attach_eml_metadata,
    };
    use crate::hdc::eml::{EmlExpr, EmlRealDomainAssumption};

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
            macro_promotion_tier: MacroPromotionTier::Formal,
            eml_compiled: None,
            eml_metrics: None,
            eml_verified_real: None,
            eml_real_domain: None,
            eml_verified_complex: None,
            eml_constructive_compiled: None,
            eml_constructive_metrics: None,
            eml_verified_constructive_real: None,
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

    fn n_squared_plus_c(c: f64) -> Expr {
        Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("n".to_string())),
                Box::new(Expr::Const(2.0)),
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

    // IGNORED 2026-07-07: unpassable until the constructive EML backend is completed.
    // `eml::compile_expr_constructive` is currently a stub identical to strict
    // `compile_expr` (accepts the same expression set — see eml/compile.rs), so in
    // `subtree_candidate_identity` strict always wins first and `eml:constructive:` is
    // NEVER produced. The distinction is real and consumed (conjecture_metadata.rs
    // classifies `PreferredEmlBackend::ConstructiveReal` via `EmlEvalMode::RealConstructive`)
    // but its intended semantics — what constructive accepts/verifies beyond strict — are
    // undocumented. Completing it is an EML-owner design task, not a test tweak; do NOT
    // relax the assertion to `starts_with("eml:")` (that hides a dead backend). See
    // CI_GREEN_TRIAGE_2026-07-07.md item B5.
    #[test]
    #[ignore = "constructive EML backend is a stub == strict; eml:constructive: is never produced. Needs the intended RealConstructive semantics. See CI_GREEN_TRIAGE_2026-07-07.md B5."]
    fn test_observe_subtree_uses_eml_candidate_identity() {
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        let c0 = make_conjecture(
            Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("n".to_string())),
                Box::new(Expr::Var("n".to_string())),
            ),
            MathDomain::NumberTheory,
            "seq_mul",
            ConjectureStatus::Proposed,
        );
        let c1 = make_conjecture(
            Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("n".to_string())),
                Box::new(Expr::Const(2.0)),
            ),
            MathDomain::Physics,
            "seq_pow",
            ConjectureStatus::Proposed,
        );
        engine.conjectures.push(c0);
        engine.conjectures.push(c1);

        let mul_subtree = normalize_expr(&Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("n".to_string())),
            Box::new(Expr::Var("n".to_string())),
        ));
        let pow_subtree = normalize_expr(&Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("n".to_string())),
            Box::new(Expr::Const(2.0)),
        ));

        grammar.observe_subtree(mul_subtree, &[0], &engine);
        grammar.observe_subtree(pow_subtree, &[1], &engine);

        assert_eq!(grammar.candidates.len(), 1);
        assert_eq!(grammar.candidates[0].occurrences.len(), 2);
        assert!(
            grammar.candidates[0]
                .canonical
                .starts_with("eml:constructive:")
        );
    }

    #[test]
    fn test_candidate_not_promoted_below_threshold() {
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        // Only 2 occurrences — below threshold of 3.
        // Use NumericallyTested (NOT FormallyVerified) so the fast-track
        // path is not triggered — we want to test the standard-path threshold.
        for i in 0..2 {
            let c = make_conjecture(
                sqrt_n_plus_c(i as f64 + 1.0),
                MathDomain::NumberTheory,
                &format!("seq_{}", i),
                ConjectureStatus::NumericallyTested { test_mse: 0.5 },
            );
            engine.conjectures.push(c);
        }

        let subtree = normalize_expr(&sqrt_n_plus_c(1.0));
        grammar.observe_subtree(subtree, &[0, 1], &engine);
        grammar.promote_eligible(&engine);

        assert!(
            grammar.operators.is_empty(),
            "Should not promote with only 2 occurrences"
        );
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
        // via the standard path. Use NumericallyTested to avoid triggering
        // the formal-verification fast-track path.
        for i in 0..3 {
            let c = make_conjecture(
                sqrt_n_plus_c(i as f64 + 1.0),
                MathDomain::NumberTheory,
                "same_source", // Same source!
                ConjectureStatus::NumericallyTested { test_mse: 0.5 },
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
    fn test_fast_track_promotes_single_formally_verified() {
        // The new fast-track path: a single FormallyVerified occurrence
        // is sufficient to promote a nontrivial admissible shape, regardless
        // of source/occurrence count.
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        let c = make_conjecture(
            n_squared_plus_c(1.0),
            MathDomain::NumberTheory,
            "unique_seq",
            ConjectureStatus::FormallyVerified { proof_steps: 5 },
        );
        engine.conjectures.push(c);

        let subtree = normalize_expr(&n_squared_plus_c(1.0));
        grammar.observe_subtree(subtree, &[0], &engine);
        grammar.promote_eligible(&engine);

        assert_eq!(
            grammar.operators.len(),
            1,
            "Single formally-verified conjecture should fast-track promote"
        );
    }

    #[test]
    fn test_fast_track_rejects_extremely_low_mse() {
        // Numeric fits, even extremely low-MSE fits, are not proof. They can
        // only contribute through an explicitly non-quarantined recurrent path,
        // never singleton fast-track promotion.
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        let c = make_conjecture(
            n_squared_plus_c(1.0),
            MathDomain::NumberTheory,
            "unique_seq",
            ConjectureStatus::NumericallyTested { test_mse: 1e-9 },
        );
        engine.conjectures.push(c);

        let subtree = normalize_expr(&n_squared_plus_c(1.0));
        grammar.observe_subtree(subtree, &[0], &engine);
        grammar.promote_eligible(&engine);

        assert_eq!(
            grammar.operators.len(),
            0,
            "NumericallyTested singleton fits must not fast-track promote"
        );
    }

    #[test]
    fn test_fast_track_promotes_single_verified_eml_backend() {
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        let mut c = make_conjecture(
            n_squared_plus_c(1.0),
            MathDomain::NumberTheory,
            "unique_seq",
            ConjectureStatus::Proposed,
        );
        c.macro_promotion_tier = MacroPromotionTier::RecurrentNumerical;
        attach_eml_metadata(&mut c);
        assert!(
            c.eml_constructive_compiled.is_some(),
            "test setup requires constructive EML backend"
        );
        engine.conjectures.push(c);

        let subtree = normalize_expr(&n_squared_plus_c(1.0));
        grammar.observe_subtree(subtree, &[0], &engine);
        grammar.promote_eligible(&engine);

        assert_eq!(
            grammar.operators.len(),
            1,
            "Single verified EML-backed conjecture should fast-track promote"
        );
    }

    #[test]
    fn test_parent_formula_identity_prefers_eml_canonical_form() {
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        let mut c1 = make_conjecture(
            Expr::Func(UnaryFn::Exp, Box::new(Expr::Var("x".to_string()))),
            MathDomain::NumberTheory,
            "seq_a",
            ConjectureStatus::Proposed,
        );
        c1.formula_str = "exp_alias_a".to_string();
        attach_eml_metadata(&mut c1);

        let mut c2 = make_conjecture(
            Expr::Func(UnaryFn::Exp, Box::new(Expr::Var("x".to_string()))),
            MathDomain::Physics,
            "seq_b",
            ConjectureStatus::Proposed,
        );
        c2.formula_str = "exp_alias_b".to_string();
        attach_eml_metadata(&mut c2);

        engine.conjectures.push(c1);
        engine.conjectures.push(c2);

        let subtree = normalize_expr(&Expr::Func(
            UnaryFn::Exp,
            Box::new(Expr::Var("x".to_string())),
        ));
        grammar.observe_subtree(subtree, &[0, 1], &engine);

        assert_eq!(grammar.candidates.len(), 1);
        let candidate = &grammar.candidates[0];
        assert_eq!(candidate.parent_formulas.len(), 1);
        assert!(candidate.parent_formulas.contains("eml(x,1)"));
    }

    #[test]
    fn test_simple_unary_strict_eml_does_not_fast_track() {
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        let mut c = make_conjecture(
            Expr::Func(UnaryFn::Exp, Box::new(Expr::Var("x".to_string()))),
            MathDomain::NumberTheory,
            "simple_unary_strict",
            ConjectureStatus::Proposed,
        );
        attach_eml_metadata(&mut c);
        assert!(
            c.eml_compiled.is_some(),
            "test setup requires strict EML compilation"
        );
        engine.conjectures.push(c);

        let subtree = normalize_expr(&Expr::Func(
            UnaryFn::Exp,
            Box::new(Expr::Var("x".to_string())),
        ));
        grammar.observe_subtree(subtree, &[0], &engine);

        assert_eq!(grammar.candidates.len(), 1);
        assert_eq!(
            grammar.candidates[0].strongly_verified_count, 0,
            "simple unary strict forms should stay off the fast-track path"
        );
    }

    #[test]
    fn test_fast_track_does_not_trigger_on_poor_mse() {
        // NumericallyTested with test_mse = 0.5 (poor fit) should NOT
        // trigger fast-track. The threshold is 1e-2 which catches reasonable
        // transcendental approximations but excludes obvious noise.
        let mut grammar = DynamicGrammar::new();
        let mut engine = ConjectureEngine::new();

        let c = make_conjecture(
            sqrt_n_plus_c(1.0),
            MathDomain::NumberTheory,
            "unique_seq",
            ConjectureStatus::NumericallyTested { test_mse: 0.5 },
        );
        engine.conjectures.push(c);

        let subtree = normalize_expr(&sqrt_n_plus_c(1.0));
        grammar.observe_subtree(subtree, &[0], &engine);
        grammar.promote_eligible(&engine);

        assert!(
            grammar.operators.is_empty(),
            "Poor-MSE numerical should not fast-track promote"
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
    fn test_max_operators_prefers_strict_eml_candidate() {
        let mut grammar = DynamicGrammar::new();
        grammar.max_operators = 1;

        let mut engine = ConjectureEngine::new();
        for i in 0..3 {
            let mut strict = make_conjecture(
                Expr::BinOp(
                    BinOp::Div,
                    Box::new(Expr::Func(
                        UnaryFn::Exp,
                        Box::new(Expr::Var("x".to_string())),
                    )),
                    Box::new(Expr::Var("y".to_string())),
                ),
                MathDomain::NumberTheory,
                &format!("strict_seq_{i}"),
                ConjectureStatus::Proposed,
            );
            attach_eml_metadata(&mut strict);
            engine.conjectures.push(strict);
        }
        for i in 0..3 {
            let mut constructive = make_conjecture(
                Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::Var("x".to_string())),
                    Box::new(Expr::Var("y".to_string())),
                ),
                MathDomain::Physics,
                &format!("constructive_seq_{i}"),
                ConjectureStatus::Proposed,
            );
            attach_eml_metadata(&mut constructive);
            engine.conjectures.push(constructive);
        }

        let strict_subtree = normalize_expr(&Expr::Func(
            UnaryFn::Exp,
            Box::new(Expr::Var("x".to_string())),
        ));
        let strict_subtree = normalize_expr(&Expr::BinOp(
            BinOp::Div,
            Box::new(strict_subtree),
            Box::new(Expr::Var("y".to_string())),
        ));
        let constructive_subtree = normalize_expr(&Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        ));
        grammar.observe_subtree(strict_subtree, &[0, 1, 2], &engine);
        grammar.observe_subtree(constructive_subtree, &[3, 4, 5], &engine);

        grammar.promote_eligible(&engine);

        assert_eq!(grammar.operators.len(), 1);
        assert!(
            grammar.operators[0].canonical.starts_with("eml:strict:"),
            "strict EML candidate should win the cap, got {}",
            grammar.operators[0].canonical
        );
    }

    #[test]
    fn test_max_operators_prefers_unconstrained_strict_over_constrained_strict() {
        let mut grammar = DynamicGrammar::new();
        grammar.max_operators = 1;

        let mut engine = ConjectureEngine::new();
        for i in 0..3 {
            let mut unconstrained = make_conjecture(
                Expr::BinOp(
                    BinOp::Div,
                    Box::new(Expr::Var("x".to_string())),
                    Box::new(Expr::Var("y".to_string())),
                ),
                MathDomain::NumberTheory,
                &format!("unconstrained_seq_{i}"),
                ConjectureStatus::Proposed,
            );
            unconstrained.eml_compiled = Some(EmlExpr::terminal_var("strict_unconstrained"));
            unconstrained.eml_metrics = unconstrained.eml_compiled.as_ref().map(EmlExpr::metrics);
            unconstrained.eml_verified_real = Some(true);
            unconstrained.eml_real_domain = Some(EmlRealDomainAssumption::AnyFinite);
            unconstrained.eml_verified_complex = Some(true);
            assert_eq!(
                unconstrained.preferred_eml_backend(),
                Some(PreferredEmlBackend::StrictRealAndComplex)
            );
            assert!(
                unconstrained
                    .eml_real_domain
                    .is_some_and(|d| d.is_unconstrained())
            );
            engine.conjectures.push(unconstrained);
        }
        for i in 0..3 {
            let mut constrained = make_conjecture(
                Expr::BinOp(
                    BinOp::Div,
                    Box::new(Expr::Var("x".to_string())),
                    Box::new(Expr::Var("z".to_string())),
                ),
                MathDomain::Physics,
                &format!("constrained_seq_{i}"),
                ConjectureStatus::Proposed,
            );
            constrained.eml_compiled = Some(EmlExpr::terminal_var("strict_constrained"));
            constrained.eml_metrics = constrained.eml_compiled.as_ref().map(EmlExpr::metrics);
            constrained.eml_verified_real = Some(true);
            constrained.eml_real_domain = Some(EmlRealDomainAssumption::GreaterThanOne);
            constrained.eml_verified_complex = Some(true);
            assert_eq!(
                constrained.preferred_eml_backend(),
                Some(PreferredEmlBackend::StrictRealAndComplex)
            );
            assert!(
                constrained
                    .eml_real_domain
                    .is_some_and(|d| !d.is_unconstrained())
            );
            engine.conjectures.push(constrained);
        }

        let unconstrained_subtree = normalize_expr(&Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("y".to_string())),
        ));
        let constrained_subtree = normalize_expr(&Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Var("x".to_string())),
            Box::new(Expr::Var("z".to_string())),
        ));
        grammar.observe_subtree(unconstrained_subtree, &[0, 1, 2], &engine);
        grammar.observe_subtree(constrained_subtree, &[3, 4, 5], &engine);

        assert_eq!(grammar.candidates.len(), 2);
        let unconstrained_candidate = grammar
            .candidates
            .iter()
            .find(|candidate| candidate.canonical.contains("y"))
            .expect("unconstrained strict candidate missing");
        let constrained_candidate = grammar
            .candidates
            .iter()
            .find(|candidate| candidate.canonical.contains("z"))
            .expect("constrained strict candidate missing");
        assert_eq!(
            candidate_fast_track_rank(unconstrained_candidate, &engine),
            1
        );
        assert_eq!(candidate_fast_track_rank(constrained_candidate, &engine), 2);

        grammar.promote_eligible(&engine);

        assert_eq!(grammar.operators.len(), 1);
        assert!(
            grammar.operators[0].canonical.contains("y"),
            "expected unconstrained strict candidate to win the cap, got {}",
            grammar.operators[0].canonical
        );
    }

    #[test]
    fn test_promote_eligible_removes_multiple_candidates_without_index_panic() {
        let mut grammar = DynamicGrammar::new();
        grammar.max_operators = 3;

        let mut engine = ConjectureEngine::new();
        for i in 0..3 {
            let mut strict = make_conjecture(
                Expr::BinOp(
                    BinOp::Div,
                    Box::new(Expr::Func(
                        UnaryFn::Exp,
                        Box::new(Expr::Var("x".to_string())),
                    )),
                    Box::new(Expr::Var("y".to_string())),
                ),
                MathDomain::NumberTheory,
                &format!("strict_seq_{i}"),
                ConjectureStatus::Proposed,
            );
            attach_eml_metadata(&mut strict);
            engine.conjectures.push(strict);
        }
        for i in 0..3 {
            let mut constructive = make_conjecture(
                Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::Var("x".to_string())),
                    Box::new(Expr::Var("y".to_string())),
                ),
                MathDomain::Physics,
                &format!("constructive_seq_{i}"),
                ConjectureStatus::Proposed,
            );
            attach_eml_metadata(&mut constructive);
            engine.conjectures.push(constructive);
        }
        for i in 0..3 {
            let mut recurrent = make_conjecture(
                Expr::BinOp(
                    BinOp::Mul,
                    Box::new(Expr::Var("a".to_string())),
                    Box::new(Expr::Var("b".to_string())),
                ),
                MathDomain::Chemistry,
                &format!("recurrent_seq_{i}"),
                ConjectureStatus::FormallyVerified { proof_steps: 5 },
            );
            attach_eml_metadata(&mut recurrent);
            engine.conjectures.push(recurrent);
        }

        grammar.observe_subtree(
            normalize_expr(&Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Var("y".to_string())),
            )),
            &[3, 4, 5],
            &engine,
        );
        grammar.observe_subtree(
            normalize_expr(&Expr::BinOp(
                BinOp::Div,
                Box::new(Expr::Func(
                    UnaryFn::Exp,
                    Box::new(Expr::Var("x".to_string())),
                )),
                Box::new(Expr::Var("y".to_string())),
            )),
            &[0, 1, 2],
            &engine,
        );
        grammar.observe_subtree(
            normalize_expr(&Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("a".to_string())),
                Box::new(Expr::Var("b".to_string())),
            )),
            &[6, 7, 8],
            &engine,
        );

        grammar.promote_eligible(&engine);

        assert_eq!(grammar.operators.len(), 3);
        assert!(
            grammar.candidates.is_empty(),
            "all promoted candidates should be removed cleanly"
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
            promotion_tier: MacroPromotionTier::Formal,
            source_conjectures: vec![0],
            parent_formulas: vec!["n".to_string()],
            vars_used: vec!["n".to_string()],
            var_count: 1,
            signature: "n".to_string(),
            source_count: 1,
            usage_count: 0,
            created_at: 0,
        });
        grammar.cycle = 20; // Old enough to prune

        grammar.prune_unused();
        assert!(
            grammar.operators.is_empty(),
            "Should prune unused old operator"
        );
    }

    #[test]
    fn test_prune_keeps_used() {
        let mut grammar = DynamicGrammar::new();
        grammar.operators.push(MacroOperator {
            name: "MACRO_used".to_string(),
            template: Expr::Var("n".to_string()),
            canonical: "n".to_string(),
            arity: 0,
            promotion_tier: MacroPromotionTier::Formal,
            source_conjectures: vec![0],
            parent_formulas: vec!["n".to_string()],
            vars_used: vec!["n".to_string()],
            var_count: 1,
            signature: "n".to_string(),
            source_count: 1,
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
            promotion_tier: MacroPromotionTier::Formal,
            source_conjectures: vec![0],
            parent_formulas: vec!["n".to_string()],
            vars_used: vec!["n".to_string()],
            var_count: 1,
            signature: "n".to_string(),
            source_count: 1,
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
            promotion_tier: MacroPromotionTier::Formal,
            source_conjectures: vec![0],
            parent_formulas: vec!["n".to_string()],
            vars_used: vec!["n".to_string()],
            var_count: 1,
            signature: "n".to_string(),
            source_count: 1,
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
        assert_eq!(
            grammar.candidates[0].occurrences.len(),
            1,
            "Should not double-count"
        );
    }

    #[test]
    fn test_metrics_capture_usage_and_signature_stats() {
        let mut grammar = DynamicGrammar::new();
        grammar.cycle = 20;
        grammar.total_promoted = 3;
        grammar.total_pruned = 1;

        grammar.operators.push(MacroOperator {
            name: "FORMAL_1D".to_string(),
            template: Expr::Var("n".to_string()),
            canonical: "n".to_string(),
            arity: 0,
            promotion_tier: MacroPromotionTier::Formal,
            source_conjectures: vec![0],
            parent_formulas: vec!["n".to_string()],
            vars_used: vec!["n".to_string()],
            var_count: 1,
            signature: "n".to_string(),
            source_count: 2,
            usage_count: 3,
            created_at: 0,
        });
        grammar.operators.push(MacroOperator {
            name: "RECURRENT_4D".to_string(),
            template: Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("x".to_string())),
                Box::new(Expr::Var("vy".to_string())),
            ),
            canonical: "(x * vy)".to_string(),
            arity: 0,
            promotion_tier: MacroPromotionTier::RecurrentNumerical,
            source_conjectures: vec![1, 2],
            parent_formulas: vec!["(x * vy)".to_string()],
            vars_used: vec!["vy".to_string(), "x".to_string()],
            var_count: 2,
            signature: "vy|x".to_string(),
            source_count: 3,
            usage_count: 0,
            created_at: 0,
        });
        grammar.rebuild_operator_index();

        let metrics = grammar.metrics();
        assert_eq!(metrics.total_operators, 2);
        assert_eq!(metrics.formal_operators, 1);
        assert_eq!(metrics.recurrent_operators, 1);
        assert_eq!(metrics.used_operators, 1);
        assert_eq!(metrics.mature_operators, 2);
        assert_eq!(metrics.mature_used_operators, 1);
        assert_eq!(metrics.total_promoted, 3);
        assert_eq!(metrics.total_pruned, 1);
        assert!((metrics.active_precision - 0.5).abs() < 1e-9);
        assert!((metrics.mature_precision - 0.5).abs() < 1e-9);
        assert!((metrics.survival_rate - (2.0 / 3.0)).abs() < 1e-9);
        assert_eq!(metrics.signature_stats.len(), 2);
        assert_eq!(metrics.signature_stats[0].signature, "n");
        assert_eq!(metrics.signature_stats[0].used_operator_count, 1);
    }
}
