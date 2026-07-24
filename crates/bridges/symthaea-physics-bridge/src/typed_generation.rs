// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Constrained physical reasoning: dimensionally-typed GP candidate generation.
//!
//! The Ramanujan Protocol arc (`symthaea-core::hdc::conjecture_engine`) has, until now,
//! generated candidate expressions with an untyped grammar -- any `Var`/`Const`/`BinOp`/
//! `Func` tree, with no notion that `m` might carry mass units and `v` velocity units. This
//! crate's [`dimensional_inference::infer_dimensions`] already exists and is well-tested,
//! but only as an **analysis** tool: given an expression, infer whether it's dimensionally
//! consistent. It has never been used as a **generative constraint**.
//!
//! [`random_expr_with_dimension`] closes that gap: given a target [`DimensionalSignature`]
//! and a set of typed variables, it constructs (not reject-samples) an expression tree
//! guaranteed -- and verified, via [`infer_dimensions`](dimensional_inference::infer_dimensions)
//! -- to have that dimension. This is the first of the "constrained physical reasoning"
//! capabilities in the longer-horizon sequence agreed for the Ramanujan Protocol (HDC →
//! constrained physical reasoning → FEP active-experiment-selection → CfC → IIT); it is
//! deliberately scoped to the generative primitive plus a cheap validation, not a new
//! multi-hour discovery search -- wiring this into an actual evolutionary search is a
//! separate, later step.
//!
//! ## Why constructive, not reject-and-retry
//!
//! M2.1 (see `pde_wave_stage_b.rs`'s module doc) measured that random untyped trees almost
//! never contain a specific needed structural motif (0.022% reachability in that case).
//! Reject-sampling an untyped generator against a multi-unit dimensional target would
//! compound with that same problem -- most random trees over multi-unit variables are
//! dimensionally *inconsistent* (see `untyped_generator_rarely_hits_target_dimension`
//! below), so rejection could take arbitrarily long to terminate. Instead,
//! [`random_expr_with_dimension`] builds top-down, tracking the dimension it still needs to
//! produce at each node, and only combines subtrees in ways that are dimensionally valid by
//! construction (matching [`infer_dimensions`]'s own propagation rules exactly -- `Mul` adds
//! dimensions, `Div` subtracts, integer `Pow` scales).

use symthaea_core::hdc::conjecture_engine::{BinOp, Expr};

use crate::dimensional_inference::{InferenceResult, UnitMap, infer_dimensions};
use crate::types::DimensionalSignature;

fn xorshift_next(rng: &mut u64) -> u64 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    *rng
}

fn rand_index(rng: &mut u64, n: usize) -> usize {
    (xorshift_next(rng) as usize) % n
}

/// Small palette of dimensionless coefficients, deliberately excluding `0.0`
/// (a zero coefficient degenerates the whole subtree, uninteresting for a
/// diversity/reachability demonstration).
const RANDOM_CONSTANTS: [f64; 6] = [0.5, 1.0, 2.0, 3.0, -1.0, -0.5];

/// Bound on how many constructive attempts [`random_expr_with_dimension`] makes
/// before falling back to a guaranteed-correct (if trivial) result. See the
/// module docs and `raw_constructive_path_rarely_needs_the_safety_net` for how
/// often this retry loop is actually needed in practice (rarely -- the
/// constructive path is correct by construction in all but a narrow edge case,
/// see [`build`]'s docs).
const MAX_VERIFY_RETRIES: usize = 20;

/// Halve every exponent of `d`. Caller must ensure all exponents are even
/// (checked by [`build`] before this is invoked, matching
/// `dimensional_inference::halve_dimensions`'s inverse operation).
fn halve(d: DimensionalSignature) -> DimensionalSignature {
    let arr = d.as_array();
    let mut out = [0i8; 7];
    for (i, exp) in arr.iter().enumerate() {
        out[i] = exp / 2;
    }
    DimensionalSignature::from_array(out)
}

enum Combinator {
    ExactLeaf,
    ConstLeaf,
    /// Carries the pre-filtered candidate variables (see [`build`]'s
    /// distance-guided filtering) so execution doesn't need to re-derive them.
    MulVar(Vec<usize>),
    DivVar(Vec<usize>),
    Square,
    AddSame,
    SubSame,
    ScaleConst,
}

/// Sum of absolute exponents -- a simple "how far from dimensionless"
/// distance used to steer [`build`]'s `MulVar`/`DivVar` choices toward
/// variables that actually reduce the remaining work, instead of a blind
/// random walk over the 7-dimensional exponent space (which rarely
/// converges within a bounded depth -- see this module's history/doc
/// comments for the measured before/after).
fn distance(d: &DimensionalSignature) -> i32 {
    d.as_array().iter().map(|e| i32::from(*e).abs()).sum()
}

/// One-step (no further recursion) search for a `Mul(v1, v2)` or `Div(v1,
/// v2)` of two available variables that lands exactly on `target`. Used as
/// [`build`]'s terminal-case fallback when no combinator with remaining
/// depth budget applies -- a smarter last resort than picking an arbitrary,
/// possibly wrong-dimension variable.
fn single_step_lookahead(
    rng: &mut u64,
    target: DimensionalSignature,
    vars: &[(String, DimensionalSignature)],
) -> Option<Expr> {
    let mut mul_hits = Vec::new();
    let mut div_hits = Vec::new();
    for (n1, d1) in vars {
        for (n2, d2) in vars {
            if d1.add(d2) == target {
                mul_hits.push((n1.clone(), n2.clone()));
            }
            if d1.sub(d2) == target {
                div_hits.push((n1.clone(), n2.clone()));
            }
        }
    }
    let total = mul_hits.len() + div_hits.len();
    if total == 0 {
        return None;
    }
    let idx = rand_index(rng, total);
    if idx < mul_hits.len() {
        let (n1, n2) = &mul_hits[idx];
        Some(Expr::BinOp(
            BinOp::Mul,
            Box::new(Expr::Var(n1.clone())),
            Box::new(Expr::Var(n2.clone())),
        ))
    } else {
        let (n1, n2) = &div_hits[idx - mul_hits.len()];
        Some(Expr::BinOp(
            BinOp::Div,
            Box::new(Expr::Var(n1.clone())),
            Box::new(Expr::Var(n2.clone())),
        ))
    }
}

/// Constructive core: builds an `Expr` targeting `target`'s dimension, given
/// `vars` (name, dimension) pairs. **Guaranteed to terminate** (every
/// recursive branch strictly decreases `max_depth`) but **not unconditionally
/// guaranteed to hit `target` exactly**: at `max_depth == 0`, if no variable's
/// dimension exactly matches `target` and `target` isn't dimensionless, the
/// forced base case falls back to an arbitrary variable regardless of its
/// dimension -- a rare, honest failure mode that only bites when recursion
/// bottoms out mid-decomposition (e.g. `max_depth` too small to reach a
/// multi-step target like `ENERGY` from `{MASS, VELOCITY}`, which needs at
/// least 2 levels: `Mul` then `Square` then an exact `VELOCITY` leaf). Never
/// call this directly for a verified result -- use
/// [`random_expr_with_dimension`], which wraps this with an
/// [`infer_dimensions`]-checked retry loop. Exposed at `pub(crate)` visibility
/// so this module's own tests can measure the raw success rate directly,
/// separate from the wrapper's retry logic.
pub(crate) fn build(
    rng: &mut u64,
    target: DimensionalSignature,
    vars: &[(String, DimensionalSignature)],
    max_depth: usize,
) -> Expr {
    let exact: Vec<&str> = vars
        .iter()
        .filter(|(_, d)| *d == target)
        .map(|(n, _)| n.as_str())
        .collect();

    // Weighted option list, biased toward combinators that make real progress
    // toward `target` (ExactLeaf/MulVar/DivVar/Square) over ones that just
    // wrap the *same* target one level deeper without simplifying anything
    // (AddSame/SubSame/ScaleConst). The wrapping combinators are excluded
    // entirely once `max_depth <= 1` -- reserving the last step(s) of budget
    // for genuine progress. `MulVar`/`DivVar` are further restricted to
    // variables that strictly reduce [`distance`] to `target` -- without
    // this, they degenerate into an undirected random walk over the
    // 7-dimensional exponent space that rarely converges within a bounded
    // depth. Measured raw (no-retry) success rate on this module's demo
    // domain: 4.6% with unrestricted MulVar/DivVar var choice, 15.4% after
    // just excluding wrapping combinators near the depth limit, 86.7% after
    // adding this distance filter -- see
    // `raw_constructive_path_rarely_needs_the_safety_net`.
    let target_distance = distance(&target);
    let mul_candidates: Vec<usize> = vars
        .iter()
        .enumerate()
        .filter(|(_, (_, d))| distance(&target.sub(d)) < target_distance)
        .map(|(i, _)| i)
        .collect();
    let div_candidates: Vec<usize> = vars
        .iter()
        .enumerate()
        .filter(|(_, (_, d))| distance(&target.add(d)) < target_distance)
        .map(|(i, _)| i)
        .collect();

    let mut options: Vec<Combinator> = Vec::new();
    if !exact.is_empty() {
        options.push(Combinator::ExactLeaf);
        options.push(Combinator::ExactLeaf);
        options.push(Combinator::ExactLeaf);
    }
    if target.is_dimensionless() {
        options.push(Combinator::ConstLeaf);
    }
    if max_depth > 0 {
        if !mul_candidates.is_empty() {
            options.push(Combinator::MulVar(mul_candidates.clone()));
            options.push(Combinator::MulVar(mul_candidates));
        }
        if !div_candidates.is_empty() {
            options.push(Combinator::DivVar(div_candidates.clone()));
            options.push(Combinator::DivVar(div_candidates));
        }
        if target.as_array().iter().all(|e| e % 2 == 0) {
            options.push(Combinator::Square);
            options.push(Combinator::Square);
        }
        if max_depth > 1 {
            options.push(Combinator::AddSame);
            options.push(Combinator::SubSame);
            options.push(Combinator::ScaleConst);
        }
    }

    if options.is_empty() {
        // No exact match, not dimensionless, no depth left for a full
        // MulVar/DivVar/Square step. Try a one-step lookahead first: does
        // any single var (or pair of vars, combined via Mul/Div) land
        // exactly on `target`? This resolves the large majority of
        // otherwise-lossy terminal cases correctly instead of guessing.
        if let Some(e) = single_step_lookahead(rng, target, vars) {
            return e;
        }
        // Genuinely unreachable in the given budget/var set -- an honest,
        // rare fallback. `random_expr_with_dimension`'s retry loop (a fresh
        // draw can take a different, successful path) and final fallback
        // catch this; documented as this function's one honest failure mode.
        if vars.is_empty() {
            return Expr::Const(1.0);
        }
        let (name, _) = &vars[rand_index(rng, vars.len())];
        return Expr::Var(name.clone());
    }

    match &options[rand_index(rng, options.len())] {
        Combinator::ExactLeaf => {
            let name = exact[rand_index(rng, exact.len())];
            Expr::Var(name.to_string())
        }
        Combinator::ConstLeaf => {
            Expr::Const(RANDOM_CONSTANTS[rand_index(rng, RANDOM_CONSTANTS.len())])
        }
        Combinator::MulVar(candidates) => {
            let (name, dim) = &vars[candidates[rand_index(rng, candidates.len())]];
            let remainder = target.sub(dim);
            let rhs = build(rng, remainder, vars, max_depth - 1);
            Expr::BinOp(BinOp::Mul, Box::new(Expr::Var(name.clone())), Box::new(rhs))
        }
        Combinator::DivVar(candidates) => {
            let (name, dim) = &vars[candidates[rand_index(rng, candidates.len())]];
            let remainder = target.add(dim);
            let lhs = build(rng, remainder, vars, max_depth - 1);
            Expr::BinOp(BinOp::Div, Box::new(lhs), Box::new(Expr::Var(name.clone())))
        }
        Combinator::Square => {
            let halved = halve(target);
            let inner = build(rng, halved, vars, max_depth - 1);
            Expr::BinOp(BinOp::Pow, Box::new(inner), Box::new(Expr::Const(2.0)))
        }
        Combinator::AddSame => {
            let a = build(rng, target, vars, max_depth - 1);
            let b = build(rng, target, vars, max_depth - 1);
            Expr::BinOp(BinOp::Add, Box::new(a), Box::new(b))
        }
        Combinator::SubSame => {
            let a = build(rng, target, vars, max_depth - 1);
            let b = build(rng, target, vars, max_depth - 1);
            Expr::BinOp(BinOp::Sub, Box::new(a), Box::new(b))
        }
        Combinator::ScaleConst => {
            let inner = build(rng, target, vars, max_depth - 1);
            let c = RANDOM_CONSTANTS[rand_index(rng, RANDOM_CONSTANTS.len())];
            Expr::BinOp(BinOp::Mul, Box::new(Expr::Const(c)), Box::new(inner))
        }
    }
}

/// Generate a random `Expr` **verified** to have dimension `target`, given
/// typed variables `var_units`. Constructs via [`build`] (see its docs for
/// the strategy and its narrow, honest failure mode), then checks the result
/// against [`infer_dimensions`] -- the same ground truth this crate already
/// uses to classify discovered expressions -- retrying up to
/// [`MAX_VERIFY_RETRIES`] times on a miss. Falls back to a guaranteed-correct
/// exact-variable leaf if one exists in `var_units` for `target`, or a bare
/// dimensionless `Const(1.0)` if `target` is dimensionless; if neither
/// applies (no exact-match variable and `target` isn't dimensionless), the
/// last constructive attempt is returned unverified -- this can only happen
/// if `var_units` genuinely cannot reach `target` at all, which callers
/// should avoid by including a decomposable variable set for whatever
/// targets they intend to generate.
pub fn random_expr_with_dimension(
    rng: &mut u64,
    target: DimensionalSignature,
    var_units: &UnitMap,
    max_depth: usize,
) -> Expr {
    let vars: Vec<(String, DimensionalSignature)> =
        var_units.iter().map(|(k, v)| (k.clone(), *v)).collect();

    let mut last = Expr::Const(1.0);
    for _ in 0..MAX_VERIFY_RETRIES {
        let candidate = build(rng, target, &vars, max_depth);
        if infer_dimensions(&candidate, var_units) == InferenceResult::Inferred(target) {
            return candidate;
        }
        last = candidate;
    }

    for (name, dim) in &vars {
        if *dim == target {
            return Expr::Var(name.clone());
        }
    }
    if target.is_dimensionless() {
        return Expr::Const(1.0);
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn demo_units() -> UnitMap {
        HashMap::from([
            ("m".to_string(), DimensionalSignature::MASS),
            ("v".to_string(), DimensionalSignature::VELOCITY),
            ("r".to_string(), DimensionalSignature::LENGTH),
            ("t".to_string(), DimensionalSignature::TIME),
        ])
    }

    /// Small self-contained mirror of `symthaea_core`'s untyped
    /// `random_expr_multivar` grammar (that function is `pub(crate)` to its
    /// own crate, not reachable from here) -- used only as the "existing
    /// untyped generator" baseline for
    /// `untyped_generator_rarely_hits_target_dimension` below. Same shape:
    /// `Var`/`Const` leaves, `BinOp`/unary-`Func` internal nodes.
    fn naive_untyped_expr(rng: &mut u64, max_depth: usize, var_names: &[&str]) -> Expr {
        use symthaea_core::hdc::conjecture_engine::UnaryFn;
        if max_depth == 0 || rand_index(rng, 3) == 0 {
            if rand_index(rng, 2) == 0 {
                Expr::Var(var_names[rand_index(rng, var_names.len())].to_string())
            } else {
                Expr::Const(RANDOM_CONSTANTS[rand_index(rng, RANDOM_CONSTANTS.len())])
            }
        } else if rand_index(rng, 5) == 0 {
            let fns = [UnaryFn::Sqrt, UnaryFn::Log, UnaryFn::Sin, UnaryFn::Cos];
            Expr::Func(
                fns[rand_index(rng, fns.len())],
                Box::new(naive_untyped_expr(rng, max_depth - 1, var_names)),
            )
        } else {
            let ops = [BinOp::Add, BinOp::Sub, BinOp::Mul, BinOp::Div, BinOp::Pow];
            Expr::BinOp(
                ops[rand_index(rng, ops.len())],
                Box::new(naive_untyped_expr(rng, max_depth - 1, var_names)),
                Box::new(naive_untyped_expr(rng, max_depth - 1, var_names)),
            )
        }
    }

    #[test]
    fn typed_generator_always_dimensionally_valid() {
        let units = demo_units();
        let targets = [
            ("ENERGY", DimensionalSignature::ENERGY),
            ("FORCE", DimensionalSignature::FORCE),
            ("MOMENTUM", DimensionalSignature::MOMENTUM),
            ("VELOCITY", DimensionalSignature::VELOCITY),
            ("DIMENSIONLESS", DimensionalSignature::DIMENSIONLESS),
        ];
        let mut total = 0usize;
        let mut valid = 0usize;
        for (label, target) in targets {
            for seed in 1..=5u64 {
                let mut rng = seed;
                for _ in 0..40 {
                    let e = random_expr_with_dimension(&mut rng, target, &units, 5);
                    total += 1;
                    let ok = infer_dimensions(&e, &units) == InferenceResult::Inferred(target);
                    if ok {
                        valid += 1;
                    }
                    assert!(ok, "[{label}] generated expr not dimensionally valid: {e}");
                }
            }
        }
        println!("typed generator: {valid}/{total} verified dimensionally valid (expect 100%)");
    }

    #[test]
    fn raw_constructive_path_rarely_needs_the_safety_net() {
        // Calls `build` directly (no retry, no fallback) to honestly
        // characterize how much work the verification wrapper is actually
        // doing -- per the module docs, the constructive path should be
        // correct on its own in the large majority of cases.
        let units = demo_units();
        let vars: Vec<(String, DimensionalSignature)> =
            units.iter().map(|(k, v)| (k.clone(), *v)).collect();
        let targets = [
            DimensionalSignature::ENERGY,
            DimensionalSignature::FORCE,
            DimensionalSignature::MOMENTUM,
        ];
        let mut total = 0usize;
        let mut valid = 0usize;
        for target in targets {
            let mut rng = 0xC0FF_EE00_u64;
            for _ in 0..500 {
                let e = build(&mut rng, target, &vars, 5);
                total += 1;
                if infer_dimensions(&e, &units) == InferenceResult::Inferred(target) {
                    valid += 1;
                }
            }
        }
        let rate = valid as f64 / total as f64;
        println!(
            "raw constructive path: {valid}/{total} ({:.1}%) correct without retry",
            rate * 100.0
        );
        assert!(
            rate > 0.8,
            "constructive path should be correct without retry the large majority of the \
             time (safety net should be a rare backstop, not doing the heavy lifting), got \
             {:.1}%",
            rate * 100.0
        );
    }

    #[test]
    fn untyped_generator_rarely_hits_target_dimension() {
        let units = demo_units();
        let var_names = ["m", "v", "r", "t"];
        let target = DimensionalSignature::ENERGY;
        let mut rng = 0xBAD_5EED_u64;
        let n = 1000;
        let mut hits = 0usize;
        for _ in 0..n {
            let e = naive_untyped_expr(&mut rng, 5, &var_names);
            if infer_dimensions(&e, &units) == InferenceResult::Inferred(target) {
                hits += 1;
            }
        }
        let rate = hits as f64 / n as f64;
        println!(
            "untyped generator: {hits}/{n} ({:.2}%) hit ENERGY exactly by chance",
            rate * 100.0
        );
        assert!(
            rate < 0.05,
            "expected the untyped generator to rarely hit a specific multi-unit target \
             (an M2.1-shaped finding recurring in this domain), got {:.2}%",
            rate * 100.0
        );
    }

    #[test]
    fn typed_generator_produces_diverse_forms() {
        let units = demo_units();
        let target = DimensionalSignature::ENERGY;
        let mut forms = std::collections::HashSet::new();
        for seed in 1..=100u64 {
            let mut rng = seed;
            let e = random_expr_with_dimension(&mut rng, target, &units, 5);
            forms.insert(format!("{e}"));
        }
        println!(
            "typed generator: {}/100 distinct canonical forms for ENERGY",
            forms.len()
        );
        assert!(
            forms.len() > 20,
            "typed generator should produce meaningfully diverse structures, not collapse \
             to one trivial recipe; got only {} distinct forms/100 draws",
            forms.len()
        );
    }
}
