// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Structural scorer for Python (Logic Substrate).
//!
//! Uses the tree-sitter Python parser to extract functions and classes,
//! then compares generated code against a golden structural target.

use super::code_parser::{CodeParser, EntityKind};
use super::python_parser::PythonParser;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PythonVerdict {
    pub pass: bool,
    pub score: f32,
    pub summary: String,
    pub parse_error: Option<String>,
    pub missing_functions: Vec<String>,
    pub missing_classes: Vec<String>,
    pub signature_mismatches: Vec<String>,
}

pub fn score(generated: &str, golden: &str) -> PythonVerdict {
    let mut parser = PythonParser::new();

    let gen_parsed = match parser.parse(generated) {
        Ok(parsed) => parsed,
        Err(e) => {
            return PythonVerdict {
                pass: false,
                score: 0.0,
                summary: format!("Parse error: {}", e.message),
                parse_error: Some(e.message),
                ..Default::default()
            };
        }
    };

    let gold_parsed = match parser.parse(golden) {
        Ok(parsed) => parsed,
        Err(e) => {
            return PythonVerdict {
                pass: false,
                score: 0.0,
                summary: format!("Golden parse error: {}", e.message),
                parse_error: Some(e.message),
                ..Default::default()
            };
        }
    };

    let gen_functions = gen_parsed.entities_of_kind(EntityKind::Function);
    let gold_functions = gold_parsed.entities_of_kind(EntityKind::Function);
    let gen_classes = gen_parsed.entities_of_kind(EntityKind::Class);
    let gold_classes = gold_parsed.entities_of_kind(EntityKind::Class);

    let mut missing_functions = Vec::new();
    let mut missing_classes = Vec::new();
    let mut signature_mismatches = Vec::new();

    for gold_fn in &gold_functions {
        if let Some(gen_fn) = gen_functions.iter().find(|f| f.name == gold_fn.name) {
            let gold_return = gold_fn.annotations.get("return_type");
            let gen_return = gen_fn.annotations.get("return_type");
            if gold_return != gen_return {
                signature_mismatches.push(format!(
                    "Function '{}' return type mismatch: want {:?}, got {:?}",
                    gold_fn.name, gold_return, gen_return
                ));
            }
        } else {
            missing_functions.push(gold_fn.name.clone());
        }
    }

    for gold_class in &gold_classes {
        if !gen_classes
            .iter()
            .any(|class| class.name == gold_class.name)
        {
            missing_classes.push(gold_class.name.clone());
        }
    }

    let required_count = gold_functions.len() + gold_classes.len();
    let missing_count = missing_functions.len() + missing_classes.len();
    let score = if required_count == 0 {
        1.0
    } else {
        1.0 - (missing_count as f32 / required_count as f32)
    };
    let pass = missing_functions.is_empty()
        && missing_classes.is_empty()
        && signature_mismatches.is_empty();
    let summary = if pass {
        "PASS".to_string()
    } else {
        format!(
            "FAIL: missing_functions={} missing_classes={} mismatches={}",
            missing_functions.len(),
            missing_classes.len(),
            signature_mismatches.len()
        )
    };

    PythonVerdict {
        pass,
        score,
        summary,
        missing_functions,
        missing_classes,
        signature_mismatches,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matching_python_function_passes() {
        let generated = "def square(x: int) -> int:\n    return x * x\n";
        let golden = "def square(x: int) -> int:\n    return x ** 2\n";

        let verdict = score(generated, golden);

        assert!(verdict.pass, "{:?}", verdict);
        assert_eq!(verdict.score, 1.0);
    }

    #[test]
    fn missing_python_function_fails() {
        let generated = "def cube(x: int) -> int:\n    return x * x * x\n";
        let golden = "def square(x: int) -> int:\n    return x * x\n";

        let verdict = score(generated, golden);

        assert!(!verdict.pass);
        assert_eq!(verdict.missing_functions, vec!["square"]);
    }
}
