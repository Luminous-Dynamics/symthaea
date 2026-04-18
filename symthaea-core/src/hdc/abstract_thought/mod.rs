// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Abstract Thought — Meta-cognition for the ConjectureEngine
//!
//! Three interconnected capabilities that form a self-reinforcing feedback loop:
//!
//! 1. **Meta-HDC** — encode verified conjectures as concept vectors, cluster to find
//!    patterns-between-patterns (meta-isomorphisms)
//! 2. **Dynamic Grammar** — promote recurring sub-expressions to macro-operators
//!    in the GP grammar pool, enabling the engine to build on its own discoveries
//! 3. **Category Discovery** — detect functorial relationships between mathematical
//!    domains, upgrading coincidental cross-domain matches to structural relationships
//!
//! ## Feedback Loop
//!
//! ```text
//! Meta-HDC clusters discoveries → recurring subtrees identified
//!     ↓
//! Dynamic Grammar promotes subtrees → new macro-operators in GP
//!     ↓
//! GP discovers better conjectures → new concept vectors
//!     ↓
//! Category Discovery checks cross-domain structure → functors
//!     ↓
//! Functors become new discoveries → fed back to Meta-HDC
//! ```
//!
//! ## References
//!
//! - Eilenberg & Mac Lane (1945) — General theory of natural equivalences
//! - Kanerva (2009) — Hyperdimensional computing: An introduction
//! - Koza (1992) — Genetic Programming

pub mod category_discovery;
pub mod dynamic_grammar;
pub mod macro_quality;
pub mod meta_hdc;

use std::collections::HashSet;

use crate::hdc::arithmetic_engine::{SymbolicExpr, SymbolicOp, TermType};
use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::conjecture_engine::{BinOp, ConjectureEngine, Expr, MathDomain, UnaryFn};
use crate::hdc::primitive_system::PrimitiveSystem;

// ── Subtree extraction bounds ──────────────────────────────────────────
//
// Min complexity of 2 lets unary function applications like `sqrt(n)`,
// `exp(n)`, `log(n)`, `sin(n)`, `cos(n)` (all complexity 2) become
// extractable as candidate macros. Previously we required min=3 which
// silently filtered out every primitive function call.
//
// Max complexity of 15 allows slightly larger compound shapes through.
// The macro pool is still bounded by `DynamicGrammar::max_operators`
// (default 20), so this can't cause unbounded growth.

/// Minimum subtree complexity (AST node count) for promotion candidates.
pub const SUBTREE_MIN_COMPLEXITY: usize = 2;

/// Maximum subtree complexity (AST node count) for promotion candidates.
pub const SUBTREE_MAX_COMPLEXITY: usize = 15;

use self::category_discovery::CategoryDiscovery;
use self::dynamic_grammar::DynamicGrammar;
use self::meta_hdc::MetaHDC;

// ═══════════════════════════════════════════════════════════════════════════
// ABSTRACT THOUGHT ORCHESTRATOR
// ═══════════════════════════════════════════════════════════════════════════

/// Orchestrates the three abstract thought capabilities.
pub struct AbstractThought {
    pub meta_hdc: MetaHDC,
    pub dynamic_grammar: DynamicGrammar,
    pub category_discovery: CategoryDiscovery,
    /// Track which conjectures have been encoded (by index)
    encoded_up_to: usize,
}

impl AbstractThought {
    pub fn new() -> Self {
        Self {
            meta_hdc: MetaHDC::new(),
            dynamic_grammar: DynamicGrammar::new(),
            category_discovery: CategoryDiscovery::new(),
            encoded_up_to: 0,
        }
    }

    /// Run one cycle of abstract thought after conjecture generation.
    ///
    /// This is the core feedback loop: encode → cluster → extract → promote → categorize.
    pub fn reflect(&mut self, engine: &ConjectureEngine, primitives: &PrimitiveSystem) {
        self.dynamic_grammar.tick();

        // 1. Encode new verified conjectures as concept vectors
        for i in self.encoded_up_to..engine.conjectures.len() {
            let conj = &engine.conjectures[i];
            if conj.confidence >= 0.3 {
                self.meta_hdc.add_conjecture(conj, i, primitives);
            }
        }
        self.encoded_up_to = engine.conjectures.len();

        // 2. Cluster concept vectors
        // Use k = max(2, n/3) to avoid degenerate single-member clusters
        let n = self.meta_hdc.concepts.len();
        if n >= 3 {
            let k = (n / 3).max(2).min(10);
            self.meta_hdc.cluster(k);
        }

        // 3. Extract recurring subtrees — three paths in priority order:
        // (a) from cross-domain clusters (meta-patterns)
        // (b) globally, 2+ occurrences (standard recurrence)
        // (c) verified-singleton path (novel shapes from formally-verified
        //     or extremely-low-MSE conjectures — feeds the fast-track
        //     promotion mechanism)
        let cluster_recurring = self.meta_hdc.recurring_subtrees(engine);
        for (subtree, conj_ids) in cluster_recurring {
            self.dynamic_grammar
                .observe_subtree(subtree, &conj_ids, engine);
        }
        let global_recurring = self.meta_hdc.global_recurring_subtrees(engine, 2);
        for (subtree, conj_ids) in global_recurring {
            self.dynamic_grammar
                .observe_subtree(subtree, &conj_ids, engine);
        }
        // Verified-singleton path: every subtree from a strongly-verified
        // conjecture becomes a candidate, even if it only appears once.
        // The fast-track promotion path in dynamic_grammar decides what to promote.
        let verified_only = self.meta_hdc.verified_subtrees(engine);
        for (subtree, conj_ids) in verified_only {
            self.dynamic_grammar
                .observe_subtree(subtree, &conj_ids, engine);
        }

        // 4. Promote candidates that meet threshold
        self.dynamic_grammar.promote_eligible(engine);

        // 5. Check cross-domain matches for functorial structure
        let matches = engine.discover_cross_domain_formulas(3.0);
        if matches.len() >= 3 {
            self.category_discovery.update_from_matches(&matches);
            self.category_discovery.find_functors();
        }

        // 6. Prune unused operators
        self.dynamic_grammar.prune_unused();
    }

    /// Get active macro-operators for GP injection.
    pub fn macro_operators(&self) -> &[dynamic_grammar::MacroOperator] {
        &self.dynamic_grammar.operators
    }

    pub fn macro_pool_metrics(&self) -> dynamic_grammar::MacroPoolMetrics {
        self.dynamic_grammar.metrics()
    }
}

impl Default for AbstractThought {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPR → SYMBOLICEXPR BRIDGE
// ═══════════════════════════════════════════════════════════════════════════

/// Convert a conjecture engine `Expr` to a `SymbolicExpr` for HDC encoding.
///
/// This bridges the two expression representations:
/// - `Expr` (conjecture_engine): lightweight, no HDC, used for fast GP
/// - `SymbolicExpr` (arithmetic_engine): carries BinaryHV encoding
pub fn expr_to_symbolic(expr: &Expr, primitives: &PrimitiveSystem) -> SymbolicExpr {
    match expr {
        Expr::Var(name) => SymbolicExpr::variable(name, primitives),
        Expr::Const(c) => {
            // SymbolicExpr::constant takes i64; for non-integer f64 we approximate
            let rounded = c.round() as i64;
            if ((*c) - rounded as f64).abs() < 1e-9 {
                SymbolicExpr::constant(rounded, primitives)
            } else {
                // For irrational constants, use the nearest integer encoding
                // and bind with a perturbation to distinguish
                let base = SymbolicExpr::constant(rounded, primitives);
                // Return best approximation — the structural pattern matters more
                // than the exact value for concept vector encoding
                base
            }
        }
        Expr::BinOp(op, left, right) => {
            let l = expr_to_symbolic(left, primitives);
            let r = expr_to_symbolic(right, primitives);
            // Use SymbolicExpr methods when primitives exist, fall back to
            // direct HDC encoding when required primitives are missing.
            // This makes concept vector encoding work with any PrimitiveSystem.
            let op_name = match op {
                BinOp::Add => "ADD",
                BinOp::Sub => "SUB",
                BinOp::Mul => "MUL",
                BinOp::Div => "DIV",
                BinOp::Pow => "POW",
            };
            let op_hv = BinaryHV::random(crate::hdc::deterministic_seeds::seed_from_name(
                &format!("BINOP_{}", op_name),
            ));
            let combined = BinaryHV::bundle(&[l.encoding, r.encoding]);
            let encoding = op_hv.bind(&combined);
            let sym = match op {
                BinOp::Add => "+",
                BinOp::Sub => "-",
                BinOp::Mul => "*",
                BinOp::Div => "/",
                BinOp::Pow => "^",
            };
            SymbolicExpr {
                term_type: TermType::BinaryOp {
                    op: match op {
                        BinOp::Add => SymbolicOp::Add,
                        BinOp::Sub => SymbolicOp::Sub,
                        BinOp::Mul => SymbolicOp::Mul,
                        BinOp::Div => SymbolicOp::Div,
                        BinOp::Pow => SymbolicOp::Pow,
                    },
                    left: Box::new(l.term_type),
                    right: Box::new(r.term_type),
                },
                encoding,
                phi: l.phi + r.phi + 0.1,
                display: format!("({} {} {})", l.display, sym, r.display),
            }
        }
        Expr::Func(func, arg) => {
            let a = expr_to_symbolic(arg, primitives);
            // Encode function application by binding with function-name HV
            let func_name = match func {
                UnaryFn::Sqrt => "sqrt",
                UnaryFn::Log => "ln",
                UnaryFn::Exp => "exp",
                UnaryFn::Sin => "sin",
                UnaryFn::Cos => "cos",
                UnaryFn::Abs => "abs",
                UnaryFn::Floor => "floor",
            };
            let func_hv = BinaryHV::random(crate::hdc::deterministic_seeds::seed_from_name(
                &format!("FUNC_{}", func_name),
            ));
            SymbolicExpr {
                term_type: TermType::Function {
                    name: func_name.to_string(),
                    arg: Box::new(a.term_type.clone()),
                },
                encoding: func_hv.bind(&a.encoding),
                phi: a.phi + 0.1,
                display: format!("{}({})", func_name, a.display),
            }
        }
        Expr::Sum(body, var) => {
            let b = expr_to_symbolic(body, primitives);
            // Encode summation by binding body with SUM role vector
            let sum_hv = BinaryHV::random(crate::hdc::deterministic_seeds::seed_from_name(
                "SUM_OPERATOR",
            ));
            let var_hv = BinaryHV::random(crate::hdc::deterministic_seeds::seed_from_name(
                &format!("SUM_VAR_{}", var),
            ));
            SymbolicExpr {
                term_type: b.term_type.clone(),
                encoding: sum_hv.bind(&var_hv).bind(&b.encoding),
                phi: b.phi + 0.2,
                display: format!("Sum_{}({})", var, b.display),
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBTREE EXTRACTION UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/// Extract all subtrees of an expression with complexity in [min, max].
///
/// Used by Meta-HDC to find recurring structural patterns across conjectures.
pub fn extract_subtrees(expr: &Expr, min_complexity: usize, max_complexity: usize) -> Vec<Expr> {
    let mut results = Vec::new();
    extract_subtrees_inner(expr, min_complexity, max_complexity, &mut results);
    results
}

fn extract_subtrees_inner(expr: &Expr, min: usize, max: usize, results: &mut Vec<Expr>) {
    let c = expr.complexity();
    if c >= min && c <= max {
        results.push(expr.clone());
    }
    // Recurse into children regardless (they may have subtrees in range)
    match expr {
        Expr::Var(_) | Expr::Const(_) => {}
        Expr::BinOp(_, l, r) => {
            extract_subtrees_inner(l, min, max, results);
            extract_subtrees_inner(r, min, max, results);
        }
        Expr::Func(_, arg) => {
            extract_subtrees_inner(arg, min, max, results);
        }
        Expr::Sum(body, _) => {
            extract_subtrees_inner(body, min, max, results);
        }
    }
}

/// Normalize an expression by replacing all constants with 1.0 — **except**
/// exponents of `Pow` nodes, which are structural, not coefficients.
///
/// This captures the *structural shape* of a formula independent of specific
/// constant values, enabling structural pattern matching across conjectures.
///
/// ## Why `Pow` exponents are special
///
/// A literal `Pow(Var, Const(2))` represents a quadratic shape;
/// `Pow(Var, Const(3))` represents a cubic shape. Collapsing both to
/// `Pow(Var, Const(1))` (which evaluates to the plain Var) destroys the
/// fundamental distinction between power-law classes. This was the root
/// cause of the Stage 1 curriculum-transfer failure: the macro extracted
/// from `1/sqrt(n² + 1)` got canonicalized to `1/sqrt((n^1) + 1)` which
/// evaluates to `1/sqrt(n + 1)` — a linear-inside-sqrt shape, completely
/// different from the quadratic distance kernel. When re-seeded into
/// the GP, the mutation loop had to guess the right exponent from
/// scratch, effectively making the macro worse than useless.
///
/// With this fix, `Pow(Var, Const(k))` preserves `k` literally. Additive
/// and multiplicative constants (which ARE tunable parameters) still get
/// normalized to 1.0 as before.
pub fn normalize_expr(expr: &Expr) -> Expr {
    match expr {
        Expr::Var(name) => Expr::Var(name.clone()),
        Expr::Const(_) => Expr::Const(1.0),
        Expr::BinOp(BinOp::Pow, base, exp) => {
            // Preserve a literal constant exponent as-is. The base still
            // normalizes recursively (a base of `2*n` still collapses to
            // `1*n` so the underlying shape matches across queries).
            // Non-constant exponents (e.g. `n^(1+k)`) recurse normally
            // because they're genuinely structural holes, not fixed powers.
            let new_exp: Expr = match exp.as_ref() {
                Expr::Const(c) => Expr::Const(normalize_pow_exponent(*c)),
                _ => normalize_expr(exp),
            };
            Expr::BinOp(
                BinOp::Pow,
                Box::new(normalize_expr(base)),
                Box::new(new_exp),
            )
        }
        Expr::BinOp(op, l, r) => Expr::BinOp(
            *op,
            Box::new(normalize_expr(l)),
            Box::new(normalize_expr(r)),
        ),
        Expr::Func(f, arg) => Expr::Func(*f, Box::new(normalize_expr(arg))),
        Expr::Sum(body, var) => Expr::Sum(Box::new(normalize_expr(body)), var.clone()),
    }
}

fn normalize_pow_exponent(exponent: f64) -> f64 {
    if !exponent.is_finite() {
        return exponent;
    }

    // Tolerance must be wide enough to absorb Nelder-Mead convergence drift
    // (typically 1e-7 to 1e-8 on f64 objectives) but still tight enough to
    // distinguish integer exponents like 2 from deliberate non-integers like
    // 1.5 (Kepler's law). 1e-6 is well inside that gap.
    let rounded = exponent.round();
    if (exponent - rounded).abs() < 1e-6 && rounded.abs() <= 64.0 {
        return rounded;
    }

    let doubled = (exponent * 2.0).round();
    let half_step = doubled / 2.0;
    if (exponent - half_step).abs() < 1e-6 && half_step.abs() <= 64.0 {
        return half_step;
    }

    exponent
}

/// Semantically canonicalize an expression for macro deduplication.
///
/// Applies algebraic identity rewrites, sorts commutative operands, and
/// promotes structural equivalents to a single form. Two expressions that
/// are mathematically equivalent (modulo commutativity + simple identities)
/// produce the same canonical form.
///
/// Rewrites applied:
/// - `x + 0 → x`, `0 + x → x`
/// - `x - 0 → x`
/// - `x * 1 → x`, `1 * x → x`
/// - `x * 0 → 0`, `0 * x → 0`
/// - `x / 1 → x`
/// - `x ^ 0 → 1`, `x ^ 1 → x`
/// - `x * x → x ^ 2`
/// - Commutative sort for `+` and `*`: operands ordered by Display string
///
/// This is intentionally a best-effort simplifier — it catches the trivial
/// rewrites that were polluting the macro pool (`(1 * n)`, `(n ^ 1)`,
/// `((...) / 1)`), but does not attempt full algebraic simplification.
pub fn canonicalize_expr(expr: &Expr) -> Expr {
    let c = canonicalize_inner(expr, false);
    // Run twice — rewrites can cascade (e.g., (x^1 * 1) → (x * 1) → x)
    canonicalize_inner(&c, false)
}

/// Inner canonicalization with context flag.
///
/// `inside_func`: true when we're recursing into a function argument.
/// Inside function arguments, the `x*1 → x` and `1*x → x` simplifications
/// are SUPPRESSED so that templated shapes like `exp(c * Var)` survive
/// normalization. The literal `1` left in `exp(1 * n)` represents a
/// scale parameter that the GP/Nelder-Mead will fit at injection time.
fn canonicalize_inner(expr: &Expr, inside_func: bool) -> Expr {
    match expr {
        Expr::Var(name) => Expr::Var(name.clone()),
        Expr::Const(c) => Expr::Const(*c),
        Expr::BinOp(op, l, r) => {
            let l = canonicalize_inner(l, inside_func);
            let r = canonicalize_inner(r, inside_func);
            match op {
                BinOp::Add => {
                    // x + 0 or 0 + x → x  (always — additive identity is harmless inside funcs)
                    if is_zero(&l) {
                        return r;
                    }
                    if is_zero(&r) {
                        return l;
                    }
                    // Commutative sort
                    let (a, b) = sort_commutative(l, r);
                    Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b))
                }
                BinOp::Sub => {
                    // x - 0 → x
                    if is_zero(&r) {
                        return l;
                    }
                    // x - x → 0 (catches (1-1) fossil and any self-subtraction)
                    if structurally_equal(&l, &r) {
                        return Expr::Const(0.0);
                    }
                    Expr::BinOp(BinOp::Sub, Box::new(l), Box::new(r))
                }
                BinOp::Mul => {
                    // x * 0 → 0  (always — annihilator survives any context)
                    if is_zero(&l) || is_zero(&r) {
                        return Expr::Const(0.0);
                    }
                    // x * 1 → x  (ONLY outside function args — preserves
                    // templated scale params like exp(c*n))
                    if !inside_func {
                        if is_one(&l) {
                            return r;
                        }
                        if is_one(&r) {
                            return l;
                        }
                    }
                    // x * x → x ^ 2  (always — semantic rewrite)
                    if structurally_equal(&l, &r) {
                        return Expr::BinOp(BinOp::Pow, Box::new(l), Box::new(Expr::Const(2.0)));
                    }
                    // Commutative sort
                    let (a, b) = sort_commutative(l, r);
                    Expr::BinOp(BinOp::Mul, Box::new(a), Box::new(b))
                }
                BinOp::Div => {
                    // x / 1 → x  (ONLY outside function args)
                    if !inside_func && is_one(&r) {
                        return l;
                    }
                    Expr::BinOp(BinOp::Div, Box::new(l), Box::new(r))
                }
                BinOp::Pow => {
                    // x ^ 0 → 1
                    if is_zero(&r) {
                        return Expr::Const(1.0);
                    }
                    // x ^ 1 → x  (ONLY outside function args — preserves exp(n^1) shapes)
                    if !inside_func && is_one(&r) {
                        return l;
                    }
                    // 1 ^ x → 1 (eliminates the (1^n) fossil seen in compounding benchmark)
                    if is_one(&l) {
                        return Expr::Const(1.0);
                    }
                    // 0 ^ x → 0 (for nonzero x; we already handled x=0 above)
                    if is_zero(&l) {
                        return Expr::Const(0.0);
                    }
                    Expr::BinOp(BinOp::Pow, Box::new(l), Box::new(r))
                }
            }
        }
        // Recursing INTO a function argument flips the flag on
        Expr::Func(f, arg) => Expr::Func(*f, Box::new(canonicalize_inner(arg, true))),
        // Sum body: treat like a function argument (the iteration variable is an inner context)
        Expr::Sum(body, var) => Expr::Sum(Box::new(canonicalize_inner(body, true)), var.clone()),
    }
}

fn is_zero(expr: &Expr) -> bool {
    matches!(expr, Expr::Const(c) if c.abs() < 1e-12)
}

fn is_one(expr: &Expr) -> bool {
    matches!(expr, Expr::Const(c) if (*c - 1.0).abs() < 1e-12)
}

/// Structural equality check (ignoring constant values' exact magnitude).
/// Used for the `x * x → x^2` rewrite — we want it to fire on `(n * n)`
/// but also on already-normalized subtrees.
fn structurally_equal(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Var(na), Expr::Var(nb)) => na == nb,
        (Expr::Const(ca), Expr::Const(cb)) => (ca - cb).abs() < 1e-12,
        (Expr::BinOp(opa, la, ra), Expr::BinOp(opb, lb, rb)) => {
            opa == opb && structurally_equal(la, lb) && structurally_equal(ra, rb)
        }
        (Expr::Func(fa, aa), Expr::Func(fb, ab)) => fa == fb && structurally_equal(aa, ab),
        (Expr::Sum(ba, va), Expr::Sum(bb, vb)) => va == vb && structurally_equal(ba, bb),
        _ => false,
    }
}

/// Sort two expressions so that the smaller (by Display string) comes first.
/// Used for commutative normalization: `n + 1` and `1 + n` both become `1 + n`.
fn sort_commutative(a: Expr, b: Expr) -> (Expr, Expr) {
    let sa = format!("{}", a);
    let sb = format!("{}", b);
    if sa <= sb {
        (a, b)
    } else {
        (b, a)
    }
}

/// Canonical string representation of an expression.
///
/// Applies normalization (constants → 1.0) *then* canonicalization
/// (identity rewrites + commutative sorting + `x*x → x^2`). Two
/// structurally equivalent expressions (modulo commutativity + identity
/// rewrites) produce the same string.
pub fn expr_canonical_string(expr: &Expr) -> String {
    let normalized = normalize_expr(expr);
    let canonicalized = canonicalize_expr(&normalized);
    format!("{}", canonicalized)
}

/// Collect the distinct variable names referenced by an expression.
pub fn expr_variables(expr: &Expr) -> Vec<String> {
    let mut vars = HashSet::new();
    collect_expr_variables(expr, &mut vars);
    let mut vars: Vec<String> = vars.into_iter().collect();
    vars.sort();
    vars
}

fn collect_expr_variables(expr: &Expr, out: &mut HashSet<String>) {
    match expr {
        Expr::Var(name) => {
            out.insert(name.clone());
        }
        Expr::Const(_) => {}
        Expr::BinOp(_, l, r) => {
            collect_expr_variables(l, out);
            collect_expr_variables(r, out);
        }
        Expr::Func(_, arg) => collect_expr_variables(arg, out),
        Expr::Sum(body, var) => {
            out.insert(var.clone());
            collect_expr_variables(body, out);
        }
    }
}

/// Canonical variable-signature key, e.g. `n` or `vx|vy|x|y`.
pub fn expr_signature(expr: &Expr) -> String {
    let vars = expr_variables(expr);
    if vars.is_empty() {
        "<const>".to_string()
    } else {
        vars.join("|")
    }
}

/// Canonical signature key for an explicit variable set.
pub fn signature_from_vars(var_names: &[&str]) -> String {
    let mut vars: Vec<String> = var_names.iter().map(|name| (*name).to_string()).collect();
    vars.sort();
    vars.dedup();
    if vars.is_empty() {
        "<const>".to_string()
    } else {
        vars.join("|")
    }
}

/// Domain role vector seed — deterministic per MathDomain variant.
pub fn domain_seed(domain: MathDomain) -> u64 {
    crate::hdc::deterministic_seeds::seed_from_name(&format!("DOMAIN_{:?}", domain))
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_primitives() -> PrimitiveSystem {
        PrimitiveSystem::new()
    }

    #[test]
    fn test_expr_to_symbolic_variable() {
        let prims = make_primitives();
        let expr = Expr::Var("n".to_string());
        let sym = expr_to_symbolic(&expr, &prims);
        assert_eq!(sym.display, "n");
    }

    #[test]
    fn test_expr_to_symbolic_constant() {
        let prims = make_primitives();
        let expr = Expr::Const(42.0);
        let sym = expr_to_symbolic(&expr, &prims);
        assert_eq!(sym.display, "42");
    }

    #[test]
    fn test_expr_to_symbolic_binop() {
        let prims = make_primitives();
        // n + 1
        let expr = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("n".to_string())),
            Box::new(Expr::Const(1.0)),
        );
        let sym = expr_to_symbolic(&expr, &prims);
        assert!(sym.display.contains("+") || sym.display.contains("n"));
    }

    #[test]
    fn test_extract_subtrees() {
        // (n + 1) * (n - 2) — complexity 7
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::BinOp(
                BinOp::Add,
                Box::new(Expr::Var("n".to_string())),
                Box::new(Expr::Const(1.0)),
            )),
            Box::new(Expr::BinOp(
                BinOp::Sub,
                Box::new(Expr::Var("n".to_string())),
                Box::new(Expr::Const(2.0)),
            )),
        );
        let subtrees = extract_subtrees(&expr, 3, 5);
        // Should find (n + 1) and (n - 2), both complexity 3
        assert_eq!(subtrees.len(), 2);
        for st in &subtrees {
            assert!(st.complexity() >= 3 && st.complexity() <= 5);
        }
    }

    #[test]
    fn test_normalize_strips_constants() {
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(3.14)),
            Box::new(Expr::Var("n".to_string())),
        );
        let norm = normalize_expr(&expr);
        // Constant should become 1.0
        match &norm {
            Expr::BinOp(BinOp::Mul, l, _) => match l.as_ref() {
                Expr::Const(c) => assert_eq!(*c, 1.0),
                _ => panic!("Expected constant"),
            },
            _ => panic!("Expected BinOp"),
        }
    }

    #[test]
    fn test_canonical_string_structural_equality() {
        // Two formulas with same structure but different constants
        let a = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Const(5.0)),
            Box::new(Expr::Var("n".to_string())),
        );
        let b = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Const(99.0)),
            Box::new(Expr::Var("n".to_string())),
        );
        assert_eq!(expr_canonical_string(&a), expr_canonical_string(&b));
    }

    #[test]
    fn test_canonical_string_structural_inequality() {
        let a = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Var("n".to_string())),
        );
        let b = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Var("n".to_string())),
        );
        assert_ne!(expr_canonical_string(&a), expr_canonical_string(&b));
    }

    #[test]
    fn test_extract_subtrees_empty_for_leaves() {
        let expr = Expr::Var("x".to_string());
        let subtrees = extract_subtrees(&expr, 3, 10);
        assert!(subtrees.is_empty());
    }

    #[test]
    fn test_normalize_preserves_functions() {
        let expr = Expr::Func(UnaryFn::Sqrt, Box::new(Expr::Const(42.0)));
        let norm = normalize_expr(&expr);
        match &norm {
            Expr::Func(UnaryFn::Sqrt, arg) => match arg.as_ref() {
                Expr::Const(c) => assert_eq!(*c, 1.0),
                _ => panic!("Expected constant"),
            },
            _ => panic!("Expected Func"),
        }
    }

    #[test]
    fn test_normalize_preserves_pow_exponents() {
        // Regression: `Pow(Var, Const(2))` must stay as `Pow(Var, Const(2))`,
        // NOT become `Pow(Var, Const(1))`. The exponent is structural — it
        // distinguishes linear from quadratic from cubic shapes — and
        // collapsing it destroys the very generalization we want macros to
        // carry. This test pins the fix from Move 13 in place.
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("n".to_string())),
            Box::new(Expr::Const(2.0)),
        );
        let norm = normalize_expr(&expr);
        match &norm {
            Expr::BinOp(BinOp::Pow, base, exp) => {
                assert!(matches!(base.as_ref(), Expr::Var(_)));
                match exp.as_ref() {
                    Expr::Const(c) => {
                        assert_eq!(*c, 2.0, "Pow exponent must be preserved; got {}", c)
                    }
                    _ => panic!("Expected constant exponent"),
                }
            }
            _ => panic!("Expected Pow BinOp, got {:?}", norm),
        }
    }

    #[test]
    fn test_normalize_snaps_near_integer_pow_exponents() {
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("n".to_string())),
            Box::new(Expr::Const(2.999999999999993)),
        );
        let norm = normalize_expr(&expr);
        match &norm {
            Expr::BinOp(BinOp::Pow, _, exp) => match exp.as_ref() {
                Expr::Const(c) => assert_eq!(
                    *c, 3.0,
                    "near-integer Pow exponents should canonicalize exactly"
                ),
                _ => panic!("Expected constant exponent"),
            },
            _ => panic!("Expected Pow BinOp, got {:?}", norm),
        }
    }

    #[test]
    fn test_normalize_keeps_nonsemantic_pow_exponents_visible() {
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("n".to_string())),
            Box::new(Expr::Const(2.976395)),
        );
        let norm = normalize_expr(&expr);
        match &norm {
            Expr::BinOp(BinOp::Pow, _, exp) => match exp.as_ref() {
                Expr::Const(c) => assert!(
                    (*c - 2.976395).abs() < 1e-12,
                    "arbitrary fitted exponents should remain auditable"
                ),
                _ => panic!("Expected constant exponent"),
            },
            _ => panic!("Expected Pow BinOp, got {:?}", norm),
        }
    }

    #[test]
    fn test_normalize_preserves_distance_kernel_shape() {
        // End-to-end shape preservation: `1 / sqrt(n² + 1)` should normalize
        // to `1 / sqrt(n² + 1)` (Pow exponent preserved) rather than
        // `1 / sqrt(n + 1)` (exponent collapsed to 1 = identity).
        let expr = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Func(
                UnaryFn::Sqrt,
                Box::new(Expr::BinOp(
                    BinOp::Add,
                    Box::new(Expr::BinOp(
                        BinOp::Pow,
                        Box::new(Expr::Var("n".to_string())),
                        Box::new(Expr::Const(2.0)),
                    )),
                    Box::new(Expr::Const(1.0)),
                )),
            )),
        );
        let norm = normalize_expr(&expr);
        // Evaluate at n=3: target is 1/sqrt(10) ≈ 0.316.
        // If the exponent had been collapsed to 1, we'd get 1/sqrt(4) = 0.5.
        let val = norm.eval(&[("n", 3.0)]);
        assert!(
            (val - 0.316227766).abs() < 1e-6,
            "distance-kernel shape should evaluate to ~0.316 at n=3, got {} — \
             indicates Pow exponent was collapsed to 1",
            val
        );
    }

    // ── Canonicalization tests ─────────────────────────────────────

    #[test]
    fn test_canonicalize_identity_multiply() {
        // 1 * n → n
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Var("n".to_string())),
        );
        let canon = canonicalize_expr(&expr);
        matches!(canon, Expr::Var(_));
    }

    #[test]
    fn test_canonicalize_identity_power() {
        // n ^ 1 → n
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Var("n".to_string())),
            Box::new(Expr::Const(1.0)),
        );
        let canon = canonicalize_expr(&expr);
        assert!(matches!(canon, Expr::Var(_)));
    }

    #[test]
    fn test_canonicalize_identity_divide() {
        // (n * n) / 1 → n^2 (after cascading rewrites)
        let expr = Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("n".to_string())),
                Box::new(Expr::Var("n".to_string())),
            )),
            Box::new(Expr::Const(1.0)),
        );
        let canon = canonicalize_expr(&expr);
        // Should become n^2
        match canon {
            Expr::BinOp(BinOp::Pow, base, exp) => {
                assert!(matches!(*base, Expr::Var(_)));
                assert!(matches!(*exp, Expr::Const(c) if (c - 2.0).abs() < 1e-9));
            }
            _ => panic!("Expected n^2, got {:?}", canon),
        }
    }

    #[test]
    fn test_canonicalize_x_times_x_becomes_square() {
        // n * n → n^2
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var("n".to_string())),
            Box::new(Expr::Var("n".to_string())),
        );
        let canon = canonicalize_expr(&expr);
        assert_eq!(format!("{}", canon), "(n ^ 2)");
    }

    #[test]
    fn test_canonicalize_one_pow_x_eliminates_fossil() {
        // 1 ^ n → 1 (the fossil seen in compounding benchmark)
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Var("n".to_string())),
        );
        let canon = canonicalize_expr(&expr);
        match canon {
            Expr::Const(c) => assert!((c - 1.0).abs() < 1e-12),
            _ => panic!("Expected Const(1.0), got {:?}", canon),
        }
    }

    #[test]
    fn test_canonicalize_zero_pow_x() {
        // 0 ^ n → 0 (for nonzero n)
        let expr = Expr::BinOp(
            BinOp::Pow,
            Box::new(Expr::Const(0.0)),
            Box::new(Expr::Var("n".to_string())),
        );
        let canon = canonicalize_expr(&expr);
        match canon {
            Expr::Const(c) => assert!(c.abs() < 1e-12),
            _ => panic!("Expected Const(0.0), got {:?}", canon),
        }
    }

    #[test]
    fn test_canonicalize_preserves_func_arg_scale() {
        // exp(1 * n) should NOT collapse to exp(n) — the (1*n) marker
        // represents a templatized scale parameter
        let expr = Expr::Func(
            UnaryFn::Exp,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Const(1.0)),
                Box::new(Expr::Var("n".to_string())),
            )),
        );
        let canon = canonicalize_expr(&expr);
        // The Mul should still be present
        match &canon {
            Expr::Func(UnaryFn::Exp, arg) => match arg.as_ref() {
                Expr::BinOp(BinOp::Mul, _, _) => {} // good
                other => panic!("Expected (Mul) inside Exp, got {:?}", other),
            },
            other => panic!("Expected Exp, got {:?}", other),
        }
    }

    #[test]
    fn test_canonicalize_func_arg_pow_preserved() {
        // sqrt(n^1) inside a function should preserve the n^1 shape
        // because the 1 represents a templatized exponent
        let expr = Expr::Func(
            UnaryFn::Sqrt,
            Box::new(Expr::BinOp(
                BinOp::Pow,
                Box::new(Expr::Var("n".to_string())),
                Box::new(Expr::Const(1.0)),
            )),
        );
        let canon = canonicalize_expr(&expr);
        match &canon {
            Expr::Func(UnaryFn::Sqrt, arg) => match arg.as_ref() {
                Expr::BinOp(BinOp::Pow, _, _) => {} // preserved
                other => panic!("Expected (Pow) inside Sqrt, got {:?}", other),
            },
            other => panic!("Expected Sqrt, got {:?}", other),
        }
    }

    #[test]
    fn test_canonicalize_outer_mul_one_still_simplifies() {
        // Outside a function, (1 * n) should still collapse to n
        let expr = Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Var("n".to_string())),
        );
        let canon = canonicalize_expr(&expr);
        match canon {
            Expr::Var(name) => assert_eq!(name, "n"),
            other => panic!("Expected Var(n), got {:?}", other),
        }
    }

    #[test]
    fn test_canonicalize_func_arg_zero_still_collapses() {
        // exp(0 * n) → exp(0) — multiplicative annihilator survives any context
        // (we want this — 0 in a function arg is genuinely a constant, not a scale param)
        let expr = Expr::Func(
            UnaryFn::Exp,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Const(0.0)),
                Box::new(Expr::Var("n".to_string())),
            )),
        );
        let canon = canonicalize_expr(&expr);
        match &canon {
            Expr::Func(UnaryFn::Exp, arg) => match arg.as_ref() {
                Expr::Const(c) => assert!(c.abs() < 1e-12),
                other => panic!("Expected Const(0) inside Exp, got {:?}", other),
            },
            other => panic!("Expected Exp, got {:?}", other),
        }
    }

    #[test]
    fn test_canonicalize_commutative_sort() {
        // 1 + n should canonicalize the same way as n + 1
        let a = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Var("n".to_string())),
        );
        let b = Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Var("n".to_string())),
            Box::new(Expr::Const(1.0)),
        );
        assert_eq!(
            format!("{}", canonicalize_expr(&a)),
            format!("{}", canonicalize_expr(&b))
        );
    }

    #[test]
    fn test_canonicalize_preserves_nontrivial() {
        // sqrt(n) - 1 should not be further simplified
        let expr = Expr::BinOp(
            BinOp::Sub,
            Box::new(Expr::Func(
                UnaryFn::Sqrt,
                Box::new(Expr::Var("n".to_string())),
            )),
            Box::new(Expr::Const(1.0)),
        );
        let canon = canonicalize_expr(&expr);
        // Should remain structurally similar (Sub is not commutative, Sub with 1 is not identity)
        assert!(matches!(canon, Expr::BinOp(BinOp::Sub, _, _)));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // INTEGRATION TESTS — Full Pipeline
    // ═══════════════════════════════════════════════════════════════════════

    use crate::hdc::conjecture_engine::{
        Conjecture, ConjectureEngine, ConjectureStatus, MathDomain, ObservedSequence,
    };

    fn make_verified_conjecture(formula: Expr, domain: MathDomain, source: &str) -> Conjecture {
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
            status: ConjectureStatus::FormallyVerified { proof_steps: 5 },
            confidence: 0.95,
            macro_promotion_tier: crate::hdc::conjecture_engine::MacroPromotionTier::Formal,
        }
    }

    /// sqrt(n) + c — a pattern that will recur across domains
    fn sqrt_n_plus(c: f64) -> Expr {
        Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::Func(
                UnaryFn::Sqrt,
                Box::new(Expr::Var("n".to_string())),
            )),
            Box::new(Expr::Const(c)),
        )
    }

    /// n * n + c — a different structural pattern
    fn n_squared_plus(c: f64) -> Expr {
        Expr::BinOp(
            BinOp::Add,
            Box::new(Expr::BinOp(
                BinOp::Mul,
                Box::new(Expr::Var("n".to_string())),
                Box::new(Expr::Var("n".to_string())),
            )),
            Box::new(Expr::Const(c)),
        )
    }

    #[test]
    fn test_full_pipeline_reflect() {
        let prims = make_primitives();
        let mut engine = ConjectureEngine::new();
        engine.enable_abstract_thought();

        // Inject conjectures that share the sqrt(n)+c pattern across 3 domains
        engine.conjectures.push(make_verified_conjecture(
            sqrt_n_plus(1.0),
            MathDomain::NumberTheory,
            "primes_sqrt",
        ));
        engine.conjectures.push(make_verified_conjecture(
            sqrt_n_plus(2.0),
            MathDomain::Physics,
            "energy_sqrt",
        ));
        engine.conjectures.push(make_verified_conjecture(
            sqrt_n_plus(3.0),
            MathDomain::Chemistry,
            "rate_sqrt",
        ));

        // Run reflect
        engine.reflect(&prims);

        // Check that abstract thought state was updated
        let at = engine.abstract_thought.as_ref().unwrap();
        assert_eq!(at.meta_hdc.concepts.len(), 3, "Should encode 3 conjectures");
    }

    #[test]
    fn test_full_pipeline_macro_promotion() {
        let prims = make_primitives();
        let mut engine = ConjectureEngine::new();
        engine.enable_abstract_thought();

        // Inject 5 conjectures sharing sqrt(n)+c from 5 different sources/domains
        let domains = [
            (MathDomain::NumberTheory, "nt_seq"),
            (MathDomain::Physics, "phys_seq"),
            (MathDomain::Chemistry, "chem_seq"),
            (MathDomain::Biology, "bio_seq"),
            (MathDomain::Economics, "econ_seq"),
        ];
        for (i, (domain, source)) in domains.iter().enumerate() {
            engine.conjectures.push(make_verified_conjecture(
                sqrt_n_plus(i as f64 + 1.0),
                *domain,
                source,
            ));
        }

        // Run reflect — should encode, cluster, find recurring subtrees, maybe promote
        engine.reflect(&prims);

        let at = engine.abstract_thought.as_ref().unwrap();

        // 5 conjectures encoded
        assert_eq!(at.meta_hdc.concepts.len(), 5);

        // Should have clustered (5 >= min cluster size of 5)
        assert!(
            !at.meta_hdc.clusters.is_empty(),
            "Should have at least 1 cluster"
        );

        // The cluster should detect cross-domain diversity
        let max_diversity = at
            .meta_hdc
            .clusters
            .iter()
            .map(|c| c.domain_diversity)
            .max()
            .unwrap_or(0);
        assert!(
            max_diversity >= 2,
            "Should detect cross-domain pattern, got diversity {}",
            max_diversity
        );
    }

    #[test]
    fn test_full_pipeline_category_discovery() {
        let prims = make_primitives();
        let mut engine = ConjectureEngine::new();
        engine.enable_abstract_thought();

        // Set up cross-domain matches by adding observations + conjectures
        // that span NumberTheory → Physics → Chemistry
        let nt_data: Vec<(f64, f64)> = (1..=20).map(|n| (n as f64, (n as f64).sqrt())).collect();
        let phys_data: Vec<(f64, f64)> = (1..=20)
            .map(|n| (n as f64, (n as f64).sqrt() * 1.1))
            .collect();

        engine.observe(ObservedSequence::new(
            "nt_sqrt",
            MathDomain::NumberTheory,
            nt_data,
        ));
        engine.observe(ObservedSequence::new(
            "phys_sqrt",
            MathDomain::Physics,
            phys_data,
        ));

        // Add conjectures that match across domains
        engine.conjectures.push(make_verified_conjecture(
            Expr::Func(UnaryFn::Sqrt, Box::new(Expr::Var("n".to_string()))),
            MathDomain::NumberTheory,
            "nt_sqrt",
        ));

        engine.reflect(&prims);

        // Verify the engine is functional and doesn't crash
        // (category discovery needs cross_domain_formulas which requires matching conjectures)
        let at = engine.abstract_thought.as_ref().unwrap();
        assert_eq!(at.meta_hdc.concepts.len(), 1);
    }

    #[test]
    fn test_reflect_idempotent_encoding() {
        let prims = make_primitives();
        let mut engine = ConjectureEngine::new();
        engine.enable_abstract_thought();

        engine.conjectures.push(make_verified_conjecture(
            sqrt_n_plus(1.0),
            MathDomain::NumberTheory,
            "seq_a",
        ));

        // Reflect twice — should only encode the conjecture once
        engine.reflect(&prims);
        engine.reflect(&prims);

        let at = engine.abstract_thought.as_ref().unwrap();
        assert_eq!(
            at.meta_hdc.concepts.len(),
            1,
            "Should not double-encode on repeated reflect"
        );
    }

    #[test]
    fn test_macro_operators_accessor() {
        let mut engine = ConjectureEngine::new();
        engine.enable_abstract_thought();

        // Initially empty
        assert!(engine.macro_operators().is_empty());
    }

    #[test]
    fn test_reflect_without_abstract_thought_is_noop() {
        let prims = make_primitives();
        let mut engine = ConjectureEngine::new();
        // Do NOT enable abstract thought
        engine.conjectures.push(make_verified_conjecture(
            sqrt_n_plus(1.0),
            MathDomain::NumberTheory,
            "seq_a",
        ));
        // Should not crash
        engine.reflect(&prims);
    }

    #[test]
    fn test_end_to_end_feedback_loop() {
        // Full closed-loop test:
        // 1. Pre-populate conjectures sharing a pattern across domains
        // 2. Reflect → cluster → promote macro operator
        // 3. Run generate_conjectures on new data
        // 4. Verify macro operator is exposed via accessor
        let prims = make_primitives();
        let mut engine = ConjectureEngine::new();
        engine.enable_abstract_thought();

        // Seed 5 conjectures sharing n*n pattern across 5 independent sources/domains
        let sources = [
            (MathDomain::NumberTheory, "nt"),
            (MathDomain::Combinatorics, "comb"),
            (MathDomain::Physics, "phys"),
            (MathDomain::Biology, "bio"),
            (MathDomain::Economics, "econ"),
        ];
        for (i, (domain, src)) in sources.iter().enumerate() {
            engine.conjectures.push(make_verified_conjecture(
                n_squared_plus(i as f64 + 1.0),
                *domain,
                src,
            ));
        }

        // Reflect: encode → cluster → recurring subtrees → candidates → promote
        engine.reflect(&prims);

        // The feedback loop produces:
        // - 5 concept vectors encoded
        // - at least 1 cluster with cross-domain diversity
        let at = engine.abstract_thought.as_ref().unwrap();
        assert_eq!(at.meta_hdc.concepts.len(), 5);
        assert!(!at.meta_hdc.clusters.is_empty());

        // Check that recurring subtrees were detected + candidates tracked
        let total_candidates = at.dynamic_grammar.candidates.len();
        let total_operators = at.dynamic_grammar.operators.len();
        assert!(
            total_candidates + total_operators > 0,
            "Should detect recurring patterns (candidates={}, operators={})",
            total_candidates,
            total_operators
        );

        // Verify macro_operators accessor works
        let _macros = engine.macro_operators();
    }
}
