// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! AST-level mutation operators for a single target function.
//!
//! Unlike text/regex-based mutation (which can accidentally match digits
//! inside identifiers, comments, or string literals), these operators walk
//! a real `syn` AST and only ever touch genuine expression nodes within the
//! chosen function's body. Each call to [`Mutator::mutate_one`] applies
//! exactly one mutation at one randomly-chosen eligible site, keeping every
//! candidate a small, human-reviewable diff.

use rand::Rng;
use syn::visit::Visit;
use syn::visit_mut::VisitMut;
use syn::{BinOp, ExprBinary, ExprLit, ImplItemFn, ItemFn, Lit};

/// A human-readable record of what a mutation changed, for the certificate.
#[derive(Debug, Clone)]
pub struct MutationDescription {
    pub operator: &'static str,
    pub detail: String,
}

/// One independently-applicable mutation strategy.
pub trait MutationOperator: Send + Sync {
    fn name(&self) -> &'static str;

    /// Count how many sites in `body` this operator could apply to.
    fn count_sites(&self, body: &syn::Block) -> usize;

    /// Apply this operator at the `site_index`-th eligible site (0-based,
    /// in AST visitation order). Returns a description of what changed, or
    /// `None` if `site_index` was out of range (caller error / stale count).
    fn apply_at(
        &self,
        body: &mut syn::Block,
        site_index: usize,
        rng: &mut dyn RngCore,
    ) -> Option<MutationDescription>;
}

/// Object-safe RNG wrapper so `MutationOperator` can stay a plain trait
/// object without generic parameters leaking into `Mutator`.
pub trait RngCore {
    fn next_f64(&mut self) -> f64;
    fn gen_range_usize(&mut self, upper_exclusive: usize) -> usize;
}

impl<R: Rng> RngCore for R {
    fn next_f64(&mut self) -> f64 {
        self.r#gen::<f64>()
    }
    fn gen_range_usize(&mut self, upper_exclusive: usize) -> usize {
        if upper_exclusive == 0 {
            0
        } else {
            self.gen_range(0..upper_exclusive)
        }
    }
}

/// Swaps a comparison operator for a semantically-plausible neighbor
/// (`<` <-> `<=`, `>` <-> `>=`, `==` <-> `!=`). These are the mutations most
/// likely to compile *and* to correspond to a genuine off-by-one or
/// boundary-condition question, rather than nonsense.
pub struct ComparisonOperatorSwap;

struct CountComparisons(usize);
impl<'ast> Visit<'ast> for CountComparisons {
    fn visit_expr_binary(&mut self, node: &'ast ExprBinary) {
        if comparison_swap_for(&node.op).is_some() {
            self.0 += 1;
        }
        syn::visit::visit_expr_binary(self, node);
    }
}

struct ApplyComparisonAt {
    target_index: usize,
    seen: usize,
    applied: Option<MutationDescription>,
}
impl VisitMut for ApplyComparisonAt {
    fn visit_expr_binary_mut(&mut self, node: &mut ExprBinary) {
        if let Some(new_op) = comparison_swap_for(&node.op) {
            if self.seen == self.target_index && self.applied.is_none() {
                let before = op_str(&node.op);
                let after = op_str(&new_op);
                node.op = new_op;
                self.applied = Some(MutationDescription {
                    operator: "ComparisonOperatorSwap",
                    detail: format!("{before} -> {after}"),
                });
            }
            self.seen += 1;
        }
        syn::visit_mut::visit_expr_binary_mut(self, node);
    }
}

fn comparison_swap_for(op: &BinOp) -> Option<BinOp> {
    // syn's punctuation constructors accept `impl IntoSpans<[Span; N]>`; a
    // single bare `Span` broadcasts to all N slots, which is all a
    // synthetic mutated token needs (it doesn't need per-character spans
    // preserved). Note `syn::token::Eq` is the single `=` assignment
    // token, NOT `==` -- `BinOp::Eq` wraps `Token![==]`, i.e. `EqEq`.
    match op {
        BinOp::Lt(t) => Some(BinOp::Le(syn::token::Le(t.spans[0]))),
        BinOp::Le(t) => Some(BinOp::Lt(syn::token::Lt(t.spans[0]))),
        BinOp::Gt(t) => Some(BinOp::Ge(syn::token::Ge(t.spans[0]))),
        BinOp::Ge(t) => Some(BinOp::Gt(syn::token::Gt(t.spans[0]))),
        BinOp::Eq(t) => Some(BinOp::Ne(syn::token::Ne(t.spans[0]))),
        BinOp::Ne(t) => Some(BinOp::Eq(syn::token::EqEq(t.spans[0]))),
        _ => None,
    }
}

fn op_str(op: &BinOp) -> &'static str {
    match op {
        BinOp::Lt(_) => "<",
        BinOp::Le(_) => "<=",
        BinOp::Gt(_) => ">",
        BinOp::Ge(_) => ">=",
        BinOp::Eq(_) => "==",
        BinOp::Ne(_) => "!=",
        _ => "?",
    }
}

impl MutationOperator for ComparisonOperatorSwap {
    fn name(&self) -> &'static str {
        "ComparisonOperatorSwap"
    }
    fn count_sites(&self, body: &syn::Block) -> usize {
        let mut counter = CountComparisons(0);
        counter.visit_block(body);
        counter.0
    }
    fn apply_at(
        &self,
        body: &mut syn::Block,
        site_index: usize,
        _rng: &mut dyn RngCore,
    ) -> Option<MutationDescription> {
        let mut applier = ApplyComparisonAt {
            target_index: site_index,
            seen: 0,
            applied: None,
        };
        applier.visit_block_mut(body);
        applier.applied
    }
}

/// Swaps `+`/`-` or `*`/`/` in a binary arithmetic expression. Far more
/// likely than the comparison swap to produce something that still
/// compiles but is numerically wrong -- expected to be rejected by the
/// correctness gate most of the time, which is the point: a search loop
/// should be allowed to try mutations that mostly fail.
pub struct ArithmeticOperatorSwap;

struct CountArithmetic(usize);
impl<'ast> Visit<'ast> for CountArithmetic {
    fn visit_expr_binary(&mut self, node: &'ast ExprBinary) {
        if arithmetic_swap_for(&node.op).is_some() {
            self.0 += 1;
        }
        syn::visit::visit_expr_binary(self, node);
    }
}

struct ApplyArithmeticAt {
    target_index: usize,
    seen: usize,
    applied: Option<MutationDescription>,
}
impl VisitMut for ApplyArithmeticAt {
    fn visit_expr_binary_mut(&mut self, node: &mut ExprBinary) {
        if let Some(new_op) = arithmetic_swap_for(&node.op) {
            if self.seen == self.target_index && self.applied.is_none() {
                let before = arith_op_str(&node.op);
                let after = arith_op_str(&new_op);
                node.op = new_op;
                self.applied = Some(MutationDescription {
                    operator: "ArithmeticOperatorSwap",
                    detail: format!("{before} -> {after}"),
                });
            }
            self.seen += 1;
        }
        syn::visit_mut::visit_expr_binary_mut(self, node);
    }
}

fn arithmetic_swap_for(op: &BinOp) -> Option<BinOp> {
    match op {
        BinOp::Add(t) => Some(BinOp::Sub(syn::token::Minus(t.spans[0]))),
        BinOp::Sub(t) => Some(BinOp::Add(syn::token::Plus(t.spans[0]))),
        BinOp::Mul(t) => Some(BinOp::Div(syn::token::Slash(t.spans[0]))),
        BinOp::Div(t) => Some(BinOp::Mul(syn::token::Star(t.spans[0]))),
        _ => None,
    }
}

fn arith_op_str(op: &BinOp) -> &'static str {
    match op {
        BinOp::Add(_) => "+",
        BinOp::Sub(_) => "-",
        BinOp::Mul(_) => "*",
        BinOp::Div(_) => "/",
        _ => "?",
    }
}

impl MutationOperator for ArithmeticOperatorSwap {
    fn name(&self) -> &'static str {
        "ArithmeticOperatorSwap"
    }
    fn count_sites(&self, body: &syn::Block) -> usize {
        let mut counter = CountArithmetic(0);
        counter.visit_block(body);
        counter.0
    }
    fn apply_at(
        &self,
        body: &mut syn::Block,
        site_index: usize,
        _rng: &mut dyn RngCore,
    ) -> Option<MutationDescription> {
        let mut applier = ApplyArithmeticAt {
            target_index: site_index,
            seen: 0,
            applied: None,
        };
        applier.visit_block_mut(body);
        applier.applied
    }
}

/// Perturbs a numeric literal (float or int) by a small random percentage.
/// The AST-level analogue of `self_optimization.rs`'s regex-based constant
/// mutation, but immune to its main hazard: only genuine `syn::Lit` nodes
/// are touched, never substrings of identifiers or version comments that
/// happen to look like a number.
pub struct NumericLiteralPerturb {
    pub max_fraction: f64,
}

struct CountLiterals(usize);
impl<'ast> Visit<'ast> for CountLiterals {
    fn visit_expr_lit(&mut self, node: &'ast ExprLit) {
        if is_perturbable(&node.lit) {
            self.0 += 1;
        }
        syn::visit::visit_expr_lit(self, node);
    }
}

fn is_perturbable(lit: &Lit) -> bool {
    matches!(lit, Lit::Float(_) | Lit::Int(_))
}

struct ApplyLiteralAt {
    target_index: usize,
    seen: usize,
    fraction: f64,
    // pulled from the RngCore up front since VisitMut can't hold `&mut dyn RngCore`
    // across a recursive visit without lifetime friction; the caller passes
    // a pre-sampled perturbation direction instead.
    direction: f64,
    applied: Option<MutationDescription>,
}
impl VisitMut for ApplyLiteralAt {
    fn visit_expr_lit_mut(&mut self, node: &mut ExprLit) {
        if is_perturbable(&node.lit) && self.seen == self.target_index && self.applied.is_none() {
            let scale = 1.0 + self.direction * self.fraction;
            match &node.lit {
                Lit::Float(f) => {
                    if let Ok(v) = f.base10_parse::<f64>() {
                        let new_v = v * scale;
                        let new_lit = syn::LitFloat::new(&format!("{new_v}"), f.span());
                        let before = v;
                        node.lit = Lit::Float(new_lit);
                        self.applied = Some(MutationDescription {
                            operator: "NumericLiteralPerturb",
                            detail: format!("{before} -> {new_v}"),
                        });
                    }
                }
                Lit::Int(i) => {
                    if let Ok(v) = i.base10_parse::<i64>() {
                        let new_v = ((v as f64) * scale).round() as i64;
                        if new_v != v {
                            let new_lit = syn::LitInt::new(&format!("{new_v}"), i.span());
                            node.lit = Lit::Int(new_lit);
                            self.applied = Some(MutationDescription {
                                operator: "NumericLiteralPerturb",
                                detail: format!("{v} -> {new_v}"),
                            });
                        }
                    }
                }
                _ => {}
            }
        }
        if is_perturbable(&node.lit) {
            self.seen += 1;
        }
        syn::visit_mut::visit_expr_lit_mut(self, node);
    }
}

impl MutationOperator for NumericLiteralPerturb {
    fn name(&self) -> &'static str {
        "NumericLiteralPerturb"
    }
    fn count_sites(&self, body: &syn::Block) -> usize {
        let mut counter = CountLiterals(0);
        counter.visit_block(body);
        counter.0
    }
    fn apply_at(
        &self,
        body: &mut syn::Block,
        site_index: usize,
        rng: &mut dyn RngCore,
    ) -> Option<MutationDescription> {
        let direction = if rng.next_f64() < 0.5 { -1.0 } else { 1.0 };
        let mut applier = ApplyLiteralAt {
            target_index: site_index,
            seen: 0,
            fraction: self.max_fraction,
            direction,
            applied: None,
        };
        applier.visit_block_mut(body);
        applier.applied
    }
}

/// Swaps `&&`/`||` in a boolean expression. Like [`ArithmeticOperatorSwap`],
/// expected to compile far more often than it stays correct -- flipping
/// short-circuit AND to OR (or back) is exactly the kind of boundary-logic
/// question (e.g. "should this guard require both conditions or either
/// one?") a search loop should be free to try and have rejected by the
/// correctness gate.
pub struct BooleanOperatorSwap;

struct CountBoolean(usize);
impl<'ast> Visit<'ast> for CountBoolean {
    fn visit_expr_binary(&mut self, node: &'ast ExprBinary) {
        if boolean_swap_for(&node.op).is_some() {
            self.0 += 1;
        }
        syn::visit::visit_expr_binary(self, node);
    }
}

struct ApplyBooleanAt {
    target_index: usize,
    seen: usize,
    applied: Option<MutationDescription>,
}
impl VisitMut for ApplyBooleanAt {
    fn visit_expr_binary_mut(&mut self, node: &mut ExprBinary) {
        if let Some(new_op) = boolean_swap_for(&node.op) {
            if self.seen == self.target_index && self.applied.is_none() {
                let before = bool_op_str(&node.op);
                let after = bool_op_str(&new_op);
                node.op = new_op;
                self.applied = Some(MutationDescription {
                    operator: "BooleanOperatorSwap",
                    detail: format!("{before} -> {after}"),
                });
            }
            self.seen += 1;
        }
        syn::visit_mut::visit_expr_binary_mut(self, node);
    }
}

fn boolean_swap_for(op: &BinOp) -> Option<BinOp> {
    match op {
        BinOp::And(t) => Some(BinOp::Or(syn::token::OrOr(t.spans[0]))),
        BinOp::Or(t) => Some(BinOp::And(syn::token::AndAnd(t.spans[0]))),
        _ => None,
    }
}

fn bool_op_str(op: &BinOp) -> &'static str {
    match op {
        BinOp::And(_) => "&&",
        BinOp::Or(_) => "||",
        _ => "?",
    }
}

impl MutationOperator for BooleanOperatorSwap {
    fn name(&self) -> &'static str {
        "BooleanOperatorSwap"
    }
    fn count_sites(&self, body: &syn::Block) -> usize {
        let mut counter = CountBoolean(0);
        counter.visit_block(body);
        counter.0
    }
    fn apply_at(
        &self,
        body: &mut syn::Block,
        site_index: usize,
        _rng: &mut dyn RngCore,
    ) -> Option<MutationDescription> {
        let mut applier = ApplyBooleanAt {
            target_index: site_index,
            seen: 0,
            applied: None,
        };
        applier.visit_block_mut(body);
        applier.applied
    }
}

/// Picks one random operator and one random eligible site among all
/// registered operators, applies it, and returns the description -- or
/// `None` if no operator has any eligible site in this function body.
pub struct Mutator {
    operators: Vec<Box<dyn MutationOperator>>,
}

impl Default for Mutator {
    fn default() -> Self {
        Self {
            operators: vec![
                Box::new(ComparisonOperatorSwap),
                Box::new(ArithmeticOperatorSwap),
                Box::new(BooleanOperatorSwap),
                Box::new(NumericLiteralPerturb { max_fraction: 0.1 }),
            ],
        }
    }
}

impl Mutator {
    pub fn new(operators: Vec<Box<dyn MutationOperator>>) -> Self {
        Self { operators }
    }

    /// Apply exactly one mutation, chosen uniformly among all (operator,
    /// site) pairs currently eligible in `body`. Mutates `body` in place.
    pub fn mutate_one(
        &self,
        body: &mut syn::Block,
        rng: &mut impl Rng,
    ) -> Option<MutationDescription> {
        let counts: Vec<usize> = self
            .operators
            .iter()
            .map(|op| op.count_sites(body))
            .collect();
        let total: usize = counts.iter().sum();
        if total == 0 {
            return None;
        }
        let mut pick = rng.gen_range(0..total);
        for (op, &count) in self.operators.iter().zip(counts.iter()) {
            if pick < count {
                return op.apply_at(body, pick, rng);
            }
            pick -= count;
        }
        None
    }
}

/// Locate a free function (`ItemFn`) or an `impl` method (`ImplItemFn`) by
/// name within a parsed file, returning a mutable reference to its body.
pub fn find_function_body_mut<'f>(
    file: &'f mut syn::File,
    fn_name: &str,
) -> Option<&'f mut syn::Block> {
    for item in &mut file.items {
        match item {
            syn::Item::Fn(ItemFn { sig, block, .. }) if sig.ident == fn_name => {
                return Some(block.as_mut());
            }
            syn::Item::Impl(imp) => {
                for impl_item in &mut imp.items {
                    if let syn::ImplItem::Fn(ImplItemFn { sig, block, .. }) = impl_item {
                        if sig.ident == fn_name {
                            return Some(block);
                        }
                    }
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn parse_fn_body(src: &str) -> syn::Block {
        let file: syn::File = syn::parse_str(src).expect("test fixture must parse");
        match &file.items[0] {
            syn::Item::Fn(f) => (*f.block).clone(),
            _ => panic!("expected a free function"),
        }
    }

    #[test]
    fn comparison_swap_counts_and_applies() {
        let mut body = parse_fn_body("fn f(x: i32) -> bool { x < 5 }");
        let op = ComparisonOperatorSwap;
        assert_eq!(op.count_sites(&body), 1);
        let mut rng = StdRng::seed_from_u64(1);
        let desc = op.apply_at(&mut body, 0, &mut rng).unwrap();
        assert_eq!(desc.operator, "ComparisonOperatorSwap");
        assert_eq!(desc.detail, "< -> <=");
        // Re-parse: the mutated body should now contain `<=`.
        let regenerated = quote::quote!(#body).to_string();
        assert!(regenerated.contains("<="));
    }

    #[test]
    fn arithmetic_swap_counts_and_applies() {
        let mut body = parse_fn_body("fn f(x: i32, y: i32) -> i32 { x + y }");
        let op = ArithmeticOperatorSwap;
        assert_eq!(op.count_sites(&body), 1);
        let mut rng = StdRng::seed_from_u64(1);
        let desc = op.apply_at(&mut body, 0, &mut rng).unwrap();
        assert_eq!(desc.detail, "+ -> -");
    }

    #[test]
    fn boolean_swap_counts_and_applies() {
        let mut body = parse_fn_body("fn f(a: bool, b: bool) -> bool { a && b }");
        let op = BooleanOperatorSwap;
        assert_eq!(op.count_sites(&body), 1);
        let mut rng = StdRng::seed_from_u64(1);
        let desc = op.apply_at(&mut body, 0, &mut rng).unwrap();
        assert_eq!(desc.operator, "BooleanOperatorSwap");
        assert_eq!(desc.detail, "&& -> ||");
        let regenerated = quote::quote!(#body).to_string();
        assert!(regenerated.contains("||"));
    }

    #[test]
    fn boolean_swap_ignores_non_boolean_binops() {
        let body = parse_fn_body("fn f(x: i32, y: i32) -> bool { x < y }");
        let op = BooleanOperatorSwap;
        assert_eq!(op.count_sites(&body), 0);
    }

    #[test]
    fn numeric_literal_perturb_changes_the_value() {
        let mut body = parse_fn_body("fn f() -> f64 { 0.2 }");
        let op = NumericLiteralPerturb { max_fraction: 0.1 };
        assert_eq!(op.count_sites(&body), 1);
        let mut rng = StdRng::seed_from_u64(7);
        let desc = op.apply_at(&mut body, 0, &mut rng).unwrap();
        assert_eq!(desc.operator, "NumericLiteralPerturb");
        assert!(desc.detail.contains("0.2 ->"));
    }

    #[test]
    fn mutator_returns_none_when_no_sites_eligible() {
        let mut body =
            parse_fn_body("fn f() -> &'static str { \"no numbers or comparisons here\" }");
        let mutator = Mutator::new(vec![Box::new(ComparisonOperatorSwap)]);
        let mut rng = StdRng::seed_from_u64(1);
        assert!(mutator.mutate_one(&mut body, &mut rng).is_none());
    }

    #[test]
    fn mutator_picks_among_all_registered_operators() {
        let mut body = parse_fn_body("fn f(x: i32) -> bool { x < 5 }");
        let mutator = Mutator::default();
        let mut rng = StdRng::seed_from_u64(3);
        let desc = mutator.mutate_one(&mut body, &mut rng);
        assert!(desc.is_some());
    }

    #[test]
    fn find_function_body_mut_locates_impl_method() {
        let mut file: syn::File =
            syn::parse_str("struct S; impl S { fn target(&self, x: i32) -> i32 { x + 1 } }")
                .unwrap();
        let body = find_function_body_mut(&mut file, "target");
        assert!(body.is_some());
    }

    #[test]
    fn find_function_body_mut_returns_none_for_missing_name() {
        let mut file: syn::File = syn::parse_str("fn present() -> i32 { 1 }").unwrap();
        assert!(find_function_body_mut(&mut file, "absent").is_none());
    }
}
