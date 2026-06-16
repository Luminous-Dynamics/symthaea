// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Structural Scorer for Nix code
//!
//! Uses rnix to flatten Nix attribute sets into dotted paths and compares them.

use rnix::ast::{AttrpathValue, Root};
use rowan::ast::AstNode;
use std::collections::HashSet;

#[derive(Debug, Default, Clone)]
pub struct StructuralVerdict {
    pub missing_required: Vec<String>,
    pub extraneous: Vec<String>,
    pub value_mismatches: Vec<String>,
    pub parse_error: Option<String>,
}

impl StructuralVerdict {
    pub fn pass(&self) -> bool {
        self.missing_required.is_empty()
            && self.value_mismatches.is_empty()
            && self.parse_error.is_none()
    }
}

pub struct NixStructuralScorer {}

impl NixStructuralScorer {
    pub fn new() -> Self {
        Self {}
    }

    pub fn score(&self, generated: &str, golden: &str) -> StructuralVerdict {
        let gen_paths = self.flatten(generated);
        let gold_paths = self.flatten(golden);

        let mut verdict = StructuralVerdict::default();

        if let (Err(e), _) = (&gen_paths, &gold_paths) {
            verdict.parse_error = Some(e.clone());
            return verdict;
        }

        let gen_set = gen_paths.unwrap_or_default();
        let gold_set = gold_paths.unwrap_or_default();

        for p in &gold_set {
            if !gen_set.contains(p) {
                verdict.missing_required.push(p.clone());
            }
        }

        for p in &gen_set {
            if !gold_set.contains(p) {
                verdict.extraneous.push(p.clone());
            }
        }

        verdict
    }

    fn flatten(&self, source: &str) -> Result<HashSet<String>, String> {
        let ast = Root::parse(source);
        if !ast.errors().is_empty() {
            return Err(format!("Parse error: {:?}", ast.errors()[0]));
        }

        let mut paths = HashSet::new();
        let tree = ast.tree();
        let root_node = tree.syntax();

        // Simple walk for top-level attribute assignments
        for node in root_node.descendants() {
            if let Some(entry) = AttrpathValue::cast(node) {
                if let Some(path) = entry.attrpath() {
                    paths.insert(path.to_string().replace(" ", ""));
                }
            }
        }

        Ok(paths)
    }
}
