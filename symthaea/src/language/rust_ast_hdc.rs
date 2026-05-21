// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rust AST → HDC bridge.
//!
//! This is the first concrete foundation for treating Rust as structure rather
//! than text: parse with `syn`, extract stable AST features, and encode those
//! features into compositional hypervectors.

use std::collections::{BTreeMap, BTreeSet};

use symthaea_core::hdc::ContinuousHV;
use syn::visit::{self, Visit};

use crate::hdc::code_encoder::CodeHDEncoder;

#[derive(Debug, Clone)]
pub struct RustAstHdcEncoding {
    pub hv: ContinuousHV,
    pub features: BTreeMap<String, usize>,
}

pub fn encode_rust_ast_hdc(source: &str, dim: usize) -> Result<RustAstHdcEncoding, syn::Error> {
    let file = syn::parse_file(source)?;
    let mut visitor = AstFeatureVisitor::default();
    visitor.visit_file(&file);

    let encoder = CodeHDEncoder::new(dim);
    let mut feature_hvs = Vec::new();
    for (feature, count) in &visitor.features {
        let role = encoder.encode_name(feature);
        let count_hv = ContinuousHV::random(dim, stable_seed(feature) ^ *count as u64);
        feature_hvs.push(role.bind(&count_hv));
    }

    let hv = if feature_hvs.is_empty() {
        ContinuousHV::zero(dim)
    } else {
        ContinuousHV::bundle_owned(&feature_hvs)
    };

    Ok(RustAstHdcEncoding {
        hv,
        features: visitor.features,
    })
}

pub fn ast_feature_count(features: &BTreeMap<String, usize>) -> usize {
    features.values().sum()
}

pub fn ast_feature_cosine_similarity(
    a: &BTreeMap<String, usize>,
    b: &BTreeMap<String, usize>,
) -> Option<f32> {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (feature, count_a) in a {
        let count_a = *count_a as f32;
        norm_a += count_a * count_a;
        if let Some(count_b) = b.get(feature) {
            dot += count_a * *count_b as f32;
        }
    }
    for count_b in b.values() {
        let count_b = *count_b as f32;
        norm_b += count_b * count_b;
    }

    let denominator = norm_a.sqrt() * norm_b.sqrt();
    (denominator > 0.0).then(|| (dot / denominator).clamp(0.0, 1.0))
}

pub fn ast_feature_l1_distance(a: &BTreeMap<String, usize>, b: &BTreeMap<String, usize>) -> usize {
    let mut distance = 0usize;

    for (feature, count_a) in a {
        let count_b = b.get(feature).copied().unwrap_or(0);
        distance += count_a.abs_diff(count_b);
    }
    for (feature, count_b) in b {
        if !a.contains_key(feature) {
            distance += *count_b;
        }
    }

    distance
}

pub fn merge_ast_feature_counts(
    target: &mut BTreeMap<String, usize>,
    source: &BTreeMap<String, usize>,
) {
    for (feature, count) in source {
        *target.entry(feature.clone()).or_insert(0) += *count;
    }
}

pub fn ast_feature_similarity_to_any<'a>(
    features: &BTreeMap<String, usize>,
    prototypes: impl IntoIterator<Item = &'a BTreeMap<String, usize>>,
) -> Option<f32> {
    prototypes
        .into_iter()
        .filter_map(|prototype| ast_feature_cosine_similarity(features, prototype))
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

#[derive(Default)]
struct AstFeatureVisitor {
    features: BTreeMap<String, usize>,
    scopes: Vec<BTreeSet<String>>,
    mutable_bindings: BTreeSet<String>,
    moved_bindings: BTreeSet<String>,
}

impl AstFeatureVisitor {
    fn bump(&mut self, feature: impl Into<String>) {
        *self.features.entry(feature.into()).or_insert(0) += 1;
    }

    fn enter_scope(&mut self) {
        self.scopes.push(BTreeSet::new());
    }

    fn exit_scope(&mut self) {
        self.scopes.pop();
    }

    fn define_binding(&mut self, name: String, mutable: bool, type_name: Option<String>) {
        self.bump("semantic:def");
        self.bump(format!("semantic:def_name:{name}"));
        if mutable {
            self.bump("semantic:def_mutable");
            self.mutable_bindings.insert(name.clone());
        }
        if let Some(type_name) = type_name {
            self.bump(format!(
                "semantic:def_type:{}",
                normalize_token_text(&type_name)
            ));
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name);
        }
    }

    fn is_defined(&self, name: &str) -> bool {
        self.scopes.iter().rev().any(|scope| scope.contains(name))
    }

    fn mark_use(&mut self, name: &str) {
        self.bump("semantic:use");
        self.bump(format!("semantic:use_name:{name}"));
        if self.moved_bindings.contains(name) {
            self.bump("semantic:use_after_move_shape");
        }
        if !self.is_defined(name)
            && !matches!(name, "true" | "false" | "Some" | "None" | "Ok" | "Err")
            && !name.chars().next().is_some_and(|ch| ch.is_uppercase())
        {
            self.bump("semantic:use_unresolved_shape");
        }
    }

    fn mark_assignment_target(&mut self, expr: &syn::Expr) {
        self.bump("semantic:assignment");
        if let Some(name) = expr_ident(expr) {
            self.bump(format!("semantic:assignment_target:{name}"));
            if self.mutable_bindings.contains(&name) {
                self.bump("semantic:assignment_to_mutable");
            } else {
                self.bump("semantic:assignment_to_immutable_shape");
            }
        }
    }

    fn mark_return_expr(&mut self, expr: &syn::Expr) {
        self.bump("semantic:return_value");
        self.bump(format!("semantic:return_shape:{}", expr_shape(expr)));
        if let syn::Expr::Reference(reference) = strip_parens(expr) {
            if let Some(name) = expr_ident(&reference.expr) {
                self.bump("semantic:return_reference");
                self.bump(format!("semantic:return_reference_name:{name}"));
            }
        }
    }
}

impl<'ast> Visit<'ast> for AstFeatureVisitor {
    fn visit_item_fn(&mut self, item_fn: &'ast syn::ItemFn) {
        self.bump("item:function");
        if item_fn.sig.asyncness.is_some() {
            self.bump("modifier:async");
        }
        if item_fn.sig.generics.lt_token.is_some() {
            self.bump("function:generic");
        }
        self.bump(format!("name:{}", item_fn.sig.ident));
        self.enter_scope();
        for input in &item_fn.sig.inputs {
            if let syn::FnArg::Typed(pat_type) = input {
                if let Some((name, mutable, type_name)) = binding_from_pat(&pat_type.pat) {
                    self.bump("semantic:param");
                    self.define_binding(name, mutable, type_name);
                }
                self.bump(format!(
                    "semantic:param_type:{}",
                    normalize_token_text(
                        &quote::ToTokens::to_token_stream(&pat_type.ty).to_string()
                    )
                ));
            }
        }
        visit::visit_item_fn(self, item_fn);
        self.exit_scope();
    }

    fn visit_item_struct(&mut self, item_struct: &'ast syn::ItemStruct) {
        self.bump("item:struct");
        self.bump(format!("name:{}", item_struct.ident));
        visit::visit_item_struct(self, item_struct);
    }

    fn visit_item_enum(&mut self, item_enum: &'ast syn::ItemEnum) {
        self.bump("item:enum");
        self.bump(format!("name:{}", item_enum.ident));
        visit::visit_item_enum(self, item_enum);
    }

    fn visit_expr_for_loop(&mut self, expr: &'ast syn::ExprForLoop) {
        self.bump("control:for_loop");
        self.bump("semantic:iteration");
        self.enter_scope();
        if let Some((name, mutable, type_name)) = binding_from_pat(&expr.pat) {
            self.bump("semantic:loop_binding");
            self.define_binding(name, mutable, type_name);
        }
        visit::visit_expr_for_loop(self, expr);
        self.exit_scope();
    }

    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        self.bump("control:while_loop");
        visit::visit_expr_while(self, expr);
    }

    fn visit_expr_loop(&mut self, expr: &'ast syn::ExprLoop) {
        self.bump("control:loop");
        self.bump("semantic:unbounded_loop");
        visit::visit_expr_loop(self, expr);
    }

    fn visit_expr_if(&mut self, expr: &'ast syn::ExprIf) {
        self.bump("control:if");
        visit::visit_expr_if(self, expr);
    }

    fn visit_expr_match(&mut self, expr: &'ast syn::ExprMatch) {
        self.bump("control:match");
        visit::visit_expr_match(self, expr);
    }

    fn visit_expr_method_call(&mut self, expr: &'ast syn::ExprMethodCall) {
        self.bump("call:method");
        self.bump(format!("method:{}", expr.method));
        match expr.method.to_string().as_str() {
            "iter" => self.bump("semantic:borrowed_iteration"),
            "iter_mut" => self.bump("semantic:mutable_iteration"),
            "into_iter" => self.bump("semantic:consuming_iteration"),
            "copied" => self.bump("semantic:copy_projection"),
            "cloned" => self.bump("semantic:clone_projection"),
            "map" => self.bump("semantic:map_transform"),
            "filter" => self.bump("semantic:filter_predicate"),
            "fold" | "reduce" | "sum" | "product" => self.bump("semantic:aggregation"),
            "unwrap" | "expect" => self.bump("semantic:fallible_unwrap"),
            _ => {}
        }
        visit::visit_expr_method_call(self, expr);
    }

    fn visit_expr_call(&mut self, expr: &'ast syn::ExprCall) {
        self.bump("call:function");
        if let Some(name) = expr_ident(&expr.func) {
            match name.as_str() {
                "Ok" => self.bump("semantic:result_ok"),
                "Err" => self.bump("semantic:result_err"),
                "Some" => self.bump("semantic:option_some"),
                "None" => self.bump("semantic:option_none"),
                _ => {}
            }
        }
        for arg in &expr.args {
            if let Some(name) = expr_ident(arg) {
                self.moved_bindings.insert(name);
                self.bump("semantic:possible_move_into_call");
            }
        }
        visit::visit_expr_call(self, expr);
    }

    fn visit_expr_closure(&mut self, expr: &'ast syn::ExprClosure) {
        self.bump("expr:closure");
        visit::visit_expr_closure(self, expr);
    }

    fn visit_expr_try(&mut self, expr: &'ast syn::ExprTry) {
        self.bump("error:try_operator");
        self.bump("semantic:fallible_propagation");
        visit::visit_expr_try(self, expr);
    }

    fn visit_local(&mut self, local: &'ast syn::Local) {
        if let Some((name, mutable, type_name)) = binding_from_pat(&local.pat) {
            self.define_binding(name, mutable, type_name);
        }
        visit::visit_local(self, local);
    }

    fn visit_expr_assign(&mut self, expr: &'ast syn::ExprAssign) {
        self.mark_assignment_target(&expr.left);
        visit::visit_expr_assign(self, expr);
    }

    fn visit_expr_binary(&mut self, expr: &'ast syn::ExprBinary) {
        self.bump(format!("semantic:binop:{}", binop_name(&expr.op)));
        if binop_is_assignment(&expr.op) {
            self.mark_assignment_target(&expr.left);
        }
        visit::visit_expr_binary(self, expr);
    }

    fn visit_expr_return(&mut self, expr: &'ast syn::ExprReturn) {
        if let Some(value) = &expr.expr {
            self.mark_return_expr(value);
        } else {
            self.bump("semantic:return_unit");
        }
        visit::visit_expr_return(self, expr);
    }

    fn visit_expr_path(&mut self, expr: &'ast syn::ExprPath) {
        if expr.qself.is_none() && expr.path.segments.len() == 1 {
            self.mark_use(&expr.path.segments[0].ident.to_string());
        }
        visit::visit_expr_path(self, expr);
    }

    fn visit_expr_reference(&mut self, expr: &'ast syn::ExprReference) {
        if expr.mutability.is_some() {
            self.bump("semantic:mutable_borrow");
        } else {
            self.bump("semantic:shared_borrow");
        }
        visit::visit_expr_reference(self, expr);
    }

    fn visit_lit(&mut self, lit: &'ast syn::Lit) {
        self.bump(format!("semantic:literal:{}", literal_kind(lit)));
        visit::visit_lit(self, lit);
    }
}

fn binding_from_pat(pat: &syn::Pat) -> Option<(String, bool, Option<String>)> {
    match pat {
        syn::Pat::Ident(pat_ident) => Some((
            pat_ident.ident.to_string(),
            pat_ident.mutability.is_some(),
            None,
        )),
        syn::Pat::Type(pat_type) => binding_from_pat(&pat_type.pat).map(|(name, mutable, _)| {
            (
                name,
                mutable,
                Some(quote::ToTokens::to_token_stream(&pat_type.ty).to_string()),
            )
        }),
        syn::Pat::Reference(reference) => binding_from_pat(&reference.pat),
        _ => None,
    }
}

fn expr_ident(expr: &syn::Expr) -> Option<String> {
    match strip_parens(expr) {
        syn::Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
            Some(path.path.segments[0].ident.to_string())
        }
        syn::Expr::Reference(reference) => expr_ident(&reference.expr),
        _ => None,
    }
}

fn strip_parens(expr: &syn::Expr) -> &syn::Expr {
    match expr {
        syn::Expr::Paren(paren) => strip_parens(&paren.expr),
        _ => expr,
    }
}

fn expr_shape(expr: &syn::Expr) -> &'static str {
    match strip_parens(expr) {
        syn::Expr::Array(_) => "array",
        syn::Expr::Binary(_) => "binary",
        syn::Expr::Block(_) => "block",
        syn::Expr::Call(_) => "call",
        syn::Expr::Closure(_) => "closure",
        syn::Expr::Field(_) => "field",
        syn::Expr::If(_) => "if",
        syn::Expr::Lit(_) => "literal",
        syn::Expr::Match(_) => "match",
        syn::Expr::MethodCall(_) => "method_call",
        syn::Expr::Path(_) => "path",
        syn::Expr::Reference(_) => "reference",
        syn::Expr::Struct(_) => "struct",
        syn::Expr::Try(_) => "try",
        syn::Expr::Tuple(_) => "tuple",
        _ => "other",
    }
}

fn binop_name(op: &syn::BinOp) -> &'static str {
    match op {
        syn::BinOp::Add(_) => "add",
        syn::BinOp::Sub(_) => "sub",
        syn::BinOp::Mul(_) => "mul",
        syn::BinOp::Div(_) => "div",
        syn::BinOp::Rem(_) => "rem",
        syn::BinOp::And(_) => "and",
        syn::BinOp::Or(_) => "or",
        syn::BinOp::BitXor(_) => "bit_xor",
        syn::BinOp::BitAnd(_) => "bit_and",
        syn::BinOp::BitOr(_) => "bit_or",
        syn::BinOp::Shl(_) => "shl",
        syn::BinOp::Shr(_) => "shr",
        syn::BinOp::Eq(_) => "eq",
        syn::BinOp::Lt(_) => "lt",
        syn::BinOp::Le(_) => "le",
        syn::BinOp::Ne(_) => "ne",
        syn::BinOp::Ge(_) => "ge",
        syn::BinOp::Gt(_) => "gt",
        syn::BinOp::AddAssign(_) => "add_assign",
        syn::BinOp::SubAssign(_) => "sub_assign",
        syn::BinOp::MulAssign(_) => "mul_assign",
        syn::BinOp::DivAssign(_) => "div_assign",
        syn::BinOp::RemAssign(_) => "rem_assign",
        syn::BinOp::BitXorAssign(_) => "bit_xor_assign",
        syn::BinOp::BitAndAssign(_) => "bit_and_assign",
        syn::BinOp::BitOrAssign(_) => "bit_or_assign",
        syn::BinOp::ShlAssign(_) => "shl_assign",
        syn::BinOp::ShrAssign(_) => "shr_assign",
        _ => "other",
    }
}

fn binop_is_assignment(op: &syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::AddAssign(_)
            | syn::BinOp::SubAssign(_)
            | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_)
            | syn::BinOp::RemAssign(_)
            | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::BitAndAssign(_)
            | syn::BinOp::BitOrAssign(_)
            | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_)
    )
}

fn literal_kind(lit: &syn::Lit) -> &'static str {
    match lit {
        syn::Lit::Str(_) => "str",
        syn::Lit::ByteStr(_) => "byte_str",
        syn::Lit::Byte(_) => "byte",
        syn::Lit::Char(_) => "char",
        syn::Lit::Int(_) => "int",
        syn::Lit::Float(_) => "float",
        syn::Lit::Bool(_) => "bool",
        syn::Lit::Verbatim(_) => "verbatim",
        _ => "other",
    }
}

fn normalize_token_text(raw: &str) -> String {
    raw.chars().filter(|ch| !ch.is_whitespace()).collect()
}

fn stable_seed(value: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in value.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x1000_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_rust_ast_features() {
        let encoded = encode_rust_ast_hdc(
            "pub fn sum(values: &[i32]) -> i32 { values.iter().copied().sum() }",
            512,
        )
        .unwrap();

        assert_eq!(encoded.hv.dim(), 512);
        assert_eq!(encoded.features.get("item:function"), Some(&1));
        assert_eq!(encoded.features.get("call:method"), Some(&3));
        assert!(encoded.features.contains_key("method:iter"));
        assert!(encoded.features.contains_key("semantic:borrowed_iteration"));
        assert!(encoded.features.contains_key("semantic:copy_projection"));
        assert!(encoded.features.contains_key("semantic:aggregation"));
    }

    #[test]
    fn rejects_invalid_rust() {
        assert!(encode_rust_ast_hdc("fn broken( {", 512).is_err());
    }

    #[test]
    fn compares_ast_feature_trajectories() {
        let a = encode_rust_ast_hdc("pub fn add(a: i32, b: i32) -> i32 { a + b }", 512)
            .unwrap()
            .features;
        let b = encode_rust_ast_hdc(
            "pub fn add(a: i32, b: i32) -> i32 { if a > b { a } else { b } }",
            512,
        )
        .unwrap()
        .features;

        let identical = ast_feature_cosine_similarity(&a, &a).unwrap();
        let changed = ast_feature_cosine_similarity(&a, &b).unwrap();

        assert_eq!(ast_feature_count(&a), a.values().sum::<usize>());
        assert!(identical > 0.99);
        assert!(changed < identical);
        assert!(ast_feature_l1_distance(&a, &b) > 0);
    }

    #[test]
    fn semantic_features_distinguish_same_shape_different_logic() {
        let increment = encode_rust_ast_hdc("pub fn shift(x: i32) -> i32 { x + 1 }", 512)
            .unwrap()
            .features;
        let decrement = encode_rust_ast_hdc("pub fn shift(x: i32) -> i32 { x - 1 }", 512)
            .unwrap()
            .features;

        assert_eq!(increment.get("semantic:binop:add"), Some(&1));
        assert_eq!(decrement.get("semantic:binop:sub"), Some(&1));
        assert!(ast_feature_l1_distance(&increment, &decrement) > 0);
    }

    #[test]
    fn semantic_features_capture_ownership_and_result_flow() {
        let encoded = encode_rust_ast_hdc(
            r#"
            pub fn parse_positive(input: &str) -> Result<i32, String> {
                let value: i32 = input.parse()?;
                if value > 0 {
                    Ok(value)
                } else {
                    Err("not positive".to_string())
                }
            }
            "#,
            512,
        )
        .unwrap();

        assert!(
            encoded
                .features
                .contains_key("semantic:fallible_propagation")
        );
        assert!(encoded.features.contains_key("semantic:result_ok"));
        assert!(encoded.features.contains_key("semantic:result_err"));
        assert!(encoded.features.contains_key("semantic:binop:gt"));
        assert!(encoded.features.contains_key("semantic:def_type:i32"));
    }

    #[test]
    fn merges_and_scores_ast_feature_prototypes() {
        let a = encode_rust_ast_hdc("pub fn add(a: i32, b: i32) -> i32 { a + b }", 512)
            .unwrap()
            .features;
        let b = encode_rust_ast_hdc("pub fn is_positive(value: i32) -> bool { value > 0 }", 512)
            .unwrap()
            .features;

        let mut prototype = BTreeMap::new();
        merge_ast_feature_counts(&mut prototype, &a);
        merge_ast_feature_counts(&mut prototype, &b);

        assert_eq!(
            prototype.get("item:function"),
            Some(&(a["item:function"] + b["item:function"]))
        );
        let score = ast_feature_similarity_to_any(&a, [&prototype]).unwrap();
        assert!(score > 0.0);
    }
}
