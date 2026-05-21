// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Structural prototype memory for AST-HDC code generation.
//!
//! This is intentionally lightweight: it stores aggregate AST feature counts
//! for previously successful code shapes and scores new candidates against
//! those prototypes. The goal is to turn AST-HDC from passive telemetry into a
//! usable prior for retry context, benchmark reports, and Broca repair records.

use std::collections::BTreeMap;

use serde::Serialize;

use super::rust_ast_hdc::{
    ast_feature_similarity_to_any, encode_rust_ast_hdc, merge_ast_feature_counts,
};

#[derive(Debug, Clone, Serialize)]
pub struct StructuralPriorScore {
    pub label: String,
    pub score: f32,
}

#[derive(Debug, Clone)]
pub struct StructuralPrototypeLabels {
    pub category: String,
    pub return_shape: String,
    pub backend: String,
}

impl StructuralPrototypeLabels {
    pub fn new(
        category: impl Into<String>,
        return_shape: impl Into<String>,
        backend: impl Into<String>,
    ) -> Self {
        Self {
            category: category.into(),
            return_shape: return_shape.into(),
            backend: backend.into(),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct StructuralPrototypeBank {
    global: BTreeMap<String, usize>,
    success_count: usize,
    by_category: BTreeMap<String, BTreeMap<String, usize>>,
    by_return_shape: BTreeMap<String, BTreeMap<String, usize>>,
    by_backend: BTreeMap<String, BTreeMap<String, usize>>,
    by_repair_category: BTreeMap<String, BTreeMap<String, usize>>,
}

impl StructuralPrototypeBank {
    pub fn score(
        &self,
        features: &BTreeMap<String, usize>,
        labels: &StructuralPrototypeLabels,
    ) -> Option<StructuralPriorScore> {
        let candidates = [
            (
                format!("category:{}", labels.category),
                self.by_category.get(&labels.category),
            ),
            (
                format!("return:{}", labels.return_shape),
                self.by_return_shape.get(&labels.return_shape),
            ),
            (
                format!("backend:{}", labels.backend),
                self.by_backend.get(&labels.backend),
            ),
            (
                "global".to_string(),
                (self.success_count > 0).then_some(&self.global),
            ),
        ];

        candidates
            .into_iter()
            .filter_map(|(label, prototype)| {
                prototype
                    .and_then(|prototype| ast_feature_similarity_to_any(features, [prototype]))
                    .map(|score| StructuralPriorScore { label, score })
            })
            .max_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    pub fn observe_success(
        &mut self,
        features: &BTreeMap<String, usize>,
        labels: &StructuralPrototypeLabels,
    ) {
        self.success_count += 1;
        merge_ast_feature_counts(&mut self.global, features);
        merge_ast_feature_counts(
            self.by_category.entry(labels.category.clone()).or_default(),
            features,
        );
        merge_ast_feature_counts(
            self.by_return_shape
                .entry(labels.return_shape.clone())
                .or_default(),
            features,
        );
        merge_ast_feature_counts(
            self.by_backend.entry(labels.backend.clone()).or_default(),
            features,
        );
    }

    pub fn observe_repair_success(&mut self, features: &BTreeMap<String, usize>, category: &str) {
        merge_ast_feature_counts(
            self.by_repair_category
                .entry(category.to_string())
                .or_default(),
            features,
        );
    }

    pub fn prototype_count(&self) -> usize {
        usize::from(self.success_count > 0)
            + self.by_category.len()
            + self.by_return_shape.len()
            + self.by_backend.len()
            + self.by_repair_category.len()
    }

    pub fn success_count(&self) -> usize {
        self.success_count
    }
}

pub fn ast_features_for_source(source: &str) -> Option<BTreeMap<String, usize>> {
    encode_rust_ast_hdc(source, symthaea_core::hdc::unified_hv::HDC_DIMENSION)
        .ok()
        .map(|encoded| encoded.features)
}

pub fn return_shape_for_signature(signature: &str) -> String {
    let Some((_, raw_return)) = signature.split_once("->") else {
        return "unit".to_string();
    };
    let return_type = raw_return
        .split('{')
        .next()
        .unwrap_or(raw_return)
        .trim()
        .trim_end_matches(';')
        .trim();
    if return_type.is_empty() {
        return "unit".to_string();
    }

    let normalized = return_type
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    if normalized.starts_with("Result<") {
        "Result".to_string()
    } else if normalized.starts_with("Option<") {
        "Option".to_string()
    } else if normalized.starts_with("Vec<") {
        "Vec".to_string()
    } else if normalized.starts_with("&[") {
        "slice_ref".to_string()
    } else if normalized.starts_with('&') {
        "reference".to_string()
    } else {
        normalized
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scores_candidates_against_observed_successes() {
        let mut bank = StructuralPrototypeBank::default();
        let labels = StructuralPrototypeLabels::new("aggregation", "i32", "CodeGenerator");
        let features =
            ast_features_for_source("pub fn sum(values: &[i32]) -> i32 { values.iter().sum() }")
                .unwrap();
        bank.observe_success(&features, &labels);

        let candidate =
            ast_features_for_source("pub fn add(a: i32, b: i32) -> i32 { a + b }").unwrap();
        let score = bank.score(&candidate, &labels).unwrap();

        assert!(score.score > 0.0);
        assert!(score.label.starts_with("category:") || score.label == "global");
    }

    #[test]
    fn normalizes_common_return_shapes() {
        assert_eq!(
            return_shape_for_signature("fn parse(input: &str) -> Result<i32, Error>"),
            "Result"
        );
        assert_eq!(
            return_shape_for_signature("fn collect() -> Vec<String>"),
            "Vec"
        );
        assert_eq!(return_shape_for_signature("fn log()"), "unit");
    }
}
