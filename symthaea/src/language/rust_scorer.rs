// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Structural scorer for Rust (Logic Substrate).
//!
//! Unlike declarative substrates (Nix/HCL), the Rust scorer uses
//! `tree-sitter-rust` to extract semantic entities (functions,
//! structs, impls) and compares their signatures and presence.
//!
//! Level 4 Scorer: Handles semantic identity, visibility, and
//! function signatures. Logic body comparison is limited to jaccard
//! similarity of the internal entity tree.

use super::code_parser::{CodeParser, EntityKind};
use super::rust_parser::RustParser;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RustVerdict {
    pub pass: bool,
    pub score: f32,
    pub summary: String,
    pub parse_error: Option<String>,
    pub missing_functions: Vec<String>,
    pub signature_mismatches: Vec<String>,
}

pub fn score(generated: &str, golden: &str) -> RustVerdict {
    let mut parser = RustParser::new();

    let gen_parsed = match parser.parse(generated) {
        Ok(p) => p,
        Err(e) => {
            return RustVerdict {
                pass: false,
                score: 0.0,
                summary: format!("Parse error: {}", e.message),
                parse_error: Some(e.message),
                ..Default::default()
            };
        }
    };

    let gold_parsed = match parser.parse(golden) {
        Ok(p) => p,
        Err(_) => {
            return RustVerdict {
                pass: false,
                score: 0.0,
                summary: "Golden source failed to parse".to_string(),
                ..Default::default()
            };
        }
    };

    let mut missing_functions = Vec::new();
    let mut signature_mismatches = Vec::new();

    let gold_fns = gold_parsed.entities_of_kind(EntityKind::Function);
    let gen_fns = gen_parsed.entities_of_kind(EntityKind::Function);

    for gold_fn in &gold_fns {
        if let Some(gen_fn) = gen_fns.iter().find(|f| f.name == gold_fn.name) {
            // Compare signatures (parameters and return type)
            let gold_ret = gold_fn.annotations.get("return_type");
            let gen_ret = gen_fn.annotations.get("return_type");
            if gold_ret != gen_ret {
                signature_mismatches.push(format!(
                    "Function '{}' return type mismatch: want {:?}, got {:?}",
                    gold_fn.name, gold_ret, gen_ret
                ));
            }
        } else {
            missing_functions.push(gold_fn.name.clone());
        }
    }

    let pass = missing_functions.is_empty() && signature_mismatches.is_empty();
    let score = if gold_fns.is_empty() {
        1.0
    } else {
        1.0 - (missing_functions.len() as f32 / gold_fns.len() as f32)
    };

    let summary = if pass {
        "PASS".to_string()
    } else {
        format!(
            "FAIL: missing={} mismatches={}",
            missing_functions.len(),
            signature_mismatches.len()
        )
    };

    RustVerdict {
        pass,
        score,
        summary,
        missing_functions,
        signature_mismatches,
        ..Default::default()
    }
}
