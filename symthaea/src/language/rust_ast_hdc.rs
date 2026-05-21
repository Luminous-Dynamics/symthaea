// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rust AST → HDC bridge.
//!
//! This is the first concrete foundation for treating Rust as structure rather
//! than text: parse with `syn`, extract stable AST features, and encode those
//! features into compositional hypervectors.

use std::collections::BTreeMap;

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
}

impl AstFeatureVisitor {
    fn bump(&mut self, feature: impl Into<String>) {
        *self.features.entry(feature.into()).or_insert(0) += 1;
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
        visit::visit_item_fn(self, item_fn);
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
        visit::visit_expr_for_loop(self, expr);
    }

    fn visit_expr_while(&mut self, expr: &'ast syn::ExprWhile) {
        self.bump("control:while_loop");
        visit::visit_expr_while(self, expr);
    }

    fn visit_expr_loop(&mut self, expr: &'ast syn::ExprLoop) {
        self.bump("control:loop");
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
        visit::visit_expr_method_call(self, expr);
    }

    fn visit_expr_call(&mut self, expr: &'ast syn::ExprCall) {
        self.bump("call:function");
        visit::visit_expr_call(self, expr);
    }

    fn visit_expr_closure(&mut self, expr: &'ast syn::ExprClosure) {
        self.bump("expr:closure");
        visit::visit_expr_closure(self, expr);
    }

    fn visit_expr_try(&mut self, expr: &'ast syn::ExprTry) {
        self.bump("error:try_operator");
        visit::visit_expr_try(self, expr);
    }
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
