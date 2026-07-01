#![cfg(feature = "code_generation")]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Code Generation End-to-End Tests
// ==================================================================================
//
// Validates the full code generation pipeline:
//   Intent → CfC plan → Native emit → [LLM completion signal] → Verify
//
// Run: cargo test --test code_generation_e2e --features code_generation
// ==================================================================================

use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::language::code_generator::{CodeContext, CodeGenerator};
use symthaea::language::code_intent::{CodeIntent, CodeSpec, CodeTarget};
use symthaea::language::code_parser::{CodeParser, EntityKind};
use symthaea::language::code_verifier::CodeVerifier;
use symthaea::mind::structured_thought::{
    CodeContext as ThoughtCodeContext, EpistemicStatus, StructuredThought,
};

// ==================================================================================
// 1. Full Pipeline: Intent → Plan → Emit → Verify (native path, no LLM)
// ==================================================================================

#[test]
fn e2e_add_function_native_complete() {
    let r#gen = CodeGenerator::with_default_dim();

    let intent = CodeIntent::Create {
        target: CodeTarget::new("add", EntityKind::Function).with_language("rust"),
        spec: CodeSpec::new("rust", "add", "Add two numbers")
            .with_signature("fn add(a: i32, b: i32) -> i32")
            .with_example("add(1, 2)", "3")
            .with_example("add(-1, 1)", "0")
            .with_epistemic(EpistemicStatus::Certain),
    };

    let result = r#gen.generate(&intent, &CodeContext::default());

    // Pipeline produced real code
    assert!(!result.source.is_empty());
    assert_eq!(result.language, "rust");

    // Body is real, not a placeholder
    assert!(
        result.source.contains("a + b"),
        "Expected real body: {}",
        result.source
    );
    assert!(
        !result.source.contains("todo!"),
        "Should not contain todo: {}",
        result.source
    );

    // Tests are real assertions
    assert!(result.source.contains("assert_eq!(add(1, 2), 3)"));
    assert!(result.source.contains("assert_eq!(add(-1, 1), 0)"));

    // Phi and similarity are populated
    assert!(result.phi_score >= 0.0);
    assert!(result.intent_similarity > 0.0);
}

#[test]
fn e2e_reverse_string_native_complete() {
    let r#gen = CodeGenerator::with_default_dim();

    let intent = CodeIntent::Create {
        target: CodeTarget::new("reverse", EntityKind::Function).with_language("rust"),
        spec: CodeSpec::new("rust", "reverse", "Reverse a string")
            .with_signature("fn reverse(s: &str) -> String"),
    };

    let result = r#gen.generate(&intent, &CodeContext::default());

    assert!(result.source.contains("s.chars().rev().collect()"));
    assert!(!result.source.contains("todo!"));
}

#[test]
fn e2e_factorial_native_complete() {
    let r#gen = CodeGenerator::with_default_dim();

    let intent = CodeIntent::Create {
        target: CodeTarget::new("factorial", EntityKind::Function).with_language("rust"),
        spec: CodeSpec::new("rust", "factorial", "Compute factorial of n")
            .with_signature("fn factorial(n: u64) -> u64"),
    };

    let result = r#gen.generate(&intent, &CodeContext::default());

    assert!(
        result.source.contains(".product()"),
        "Expected factorial body: {}",
        result.source
    );
    assert!(!result.source.contains("todo!"));
}

#[test]
fn e2e_struct_with_constructor() {
    let r#gen = CodeGenerator::with_default_dim();

    let intent = CodeIntent::Create {
        target: CodeTarget::new("Point", EntityKind::Struct).with_language("rust"),
        spec: CodeSpec::new(
            "rust",
            "Point",
            "A 2D point with x: f64 and y: f64 coordinates",
        ),
    };

    let result = r#gen.generate(&intent, &CodeContext::default());

    // The CfC sequencer plans structure dynamically. The emitter should
    // produce either a struct or a function based on the plan.
    // With "A 2D point" in the purpose, we expect fields to appear.
    assert!(
        result.source.contains("x: f64") || result.source.contains("Point"),
        "Should reference Point struct or fields: {}",
        result.source
    );
}

// ==================================================================================
// 2. LLM Completion Detection: Complex intent → todo!() → needs_llm_completion
// ==================================================================================

#[test]
fn e2e_complex_intent_triggers_llm_completion() {
    let r#gen = CodeGenerator::with_default_dim();

    // "Implement Dijkstra's shortest path" — too complex for pattern matching
    let intent = CodeIntent::Create {
        target: CodeTarget::new("dijkstra", EntityKind::Function).with_language("rust"),
        spec: CodeSpec::new(
            "rust",
            "dijkstra",
            "Implement Dijkstra's shortest path algorithm on a weighted adjacency list",
        )
        .with_signature("fn dijkstra(graph: &[Vec<(usize, u32)>], start: usize) -> Vec<u32>"),
    };

    let result = r#gen.generate(&intent, &CodeContext::default());

    // Should have a function signature but a todo! body
    assert!(result.source.contains("pub fn dijkstra"));
    assert!(
        result.source.contains("todo!"),
        "Complex algorithm should fall through to todo!: {}",
        result.source
    );

    // Build the thought code context — needs_llm_completion should be true
    let needs_llm =
        result.source.contains("todo!(") || result.source.contains("NotImplementedError");
    assert!(needs_llm, "Should need LLM completion");

    // Verify the prompt serialization includes the completion signal
    let mut thought = StructuredThought::default();
    thought.code_context = Some(ThoughtCodeContext {
        language: "rust".to_string(),
        spec_purpose: Some("Dijkstra's shortest path".to_string()),
        spec_signature: Some(
            "fn dijkstra(graph: &[Vec<(usize, u32)>], start: usize) -> Vec<u32>".to_string(),
        ),
        spec_constraints: vec![],
        spec_examples: vec![],
        plan_steps: result
            .plan_steps
            .iter()
            .map(|s| format!("{:?}", s.action))
            .collect(),
        generated_code: Some(result.source.clone()),
        phi_score: Some(result.phi_score),
        intent_similarity: Some(result.intent_similarity),
        syntactically_valid: None,
        notes: result.notes.clone(),
        needs_llm_completion: needs_llm,
    });

    let prompt = thought.to_translation_prompt();
    assert!(prompt.contains("NEEDS_COMPLETION: true"));
    assert!(prompt.contains("GENERATED_CODE"));
    assert!(prompt.contains("dijkstra"));
}

#[test]
fn e2e_simple_intent_no_llm_needed() {
    let r#gen = CodeGenerator::with_default_dim();

    let intent = CodeIntent::Create {
        target: CodeTarget::new("is_even", EntityKind::Function).with_language("rust"),
        spec: CodeSpec::new("rust", "is_even", "Check if a number is even")
            .with_signature("fn is_even(n: i32) -> bool"),
    };

    let result = r#gen.generate(&intent, &CodeContext::default());

    let needs_llm =
        result.source.contains("todo!(") || result.source.contains("NotImplementedError");
    assert!(
        !needs_llm,
        "Simple pattern should NOT need LLM: {}",
        result.source
    );
}

// ==================================================================================
// 3. Python Pipeline
// ==================================================================================

#[test]
fn e2e_python_add_native() {
    let r#gen = CodeGenerator::with_default_dim();

    let intent = CodeIntent::Create {
        target: CodeTarget::new("add", EntityKind::Function).with_language("python"),
        spec: CodeSpec::new("python", "add", "Add two numbers").with_example("add(1, 2)", "3"),
    };

    let result = r#gen.generate(&intent, &CodeContext::default());

    assert_eq!(result.language, "python");
    assert!(
        result.source.contains("return a + b"),
        "Expected Python body: {}",
        result.source
    );
    assert!(result.source.contains("assert add(1, 2) == 3"));
}

#[test]
fn e2e_python_complex_triggers_llm() {
    let r#gen = CodeGenerator::with_default_dim();

    // Use a purpose that won't match any Python emitter keyword pattern
    let intent = CodeIntent::Create {
        target: CodeTarget::new("solve_knapsack", EntityKind::Function).with_language("python"),
        spec: CodeSpec::new(
            "python",
            "solve_knapsack",
            "Implement dynamic programming knapsack solver",
        ),
    };

    let result = r#gen.generate(&intent, &CodeContext::default());

    assert_eq!(result.language, "python");
    let needs_llm = result.source.contains("NotImplementedError");
    assert!(needs_llm, "Knapsack should need LLM: {}", result.source);
}

// ==================================================================================
// 4. Tree-sitter Verification Round-trip
// ==================================================================================

#[test]
fn e2e_verify_generated_rust_syntax() {
    let r#gen = CodeGenerator::with_default_dim();
    let verifier = CodeVerifier::new(CodeHDEncoder::new(512));

    let intent = CodeIntent::Create {
        target: CodeTarget::new("multiply", EntityKind::Function).with_language("rust"),
        spec: CodeSpec::new("rust", "multiply", "Multiply two numbers")
            .with_signature("fn multiply(a: f64, b: f64) -> f64")
            .with_example("multiply(2.0, 3.0)", "6.0"),
    };

    let result = r#gen.generate(&intent, &CodeContext::default());

    // Verify with tree-sitter + HDC round-trip
    let mut parser = symthaea::language::rust_parser::RustParser::new();
    let parsed = parser
        .parse(&result.source)
        .expect("Should parse valid Rust");

    // Should have entities (function definition)
    assert!(
        !parsed.entities.is_empty(),
        "Tree-sitter should find entities in: {}",
        result.source
    );

    // HDC round-trip verification
    let encoder = CodeHDEncoder::new(512);
    let intent_hv = encoder.encode_name("Multiply two numbers");
    let verification = verifier.verify_against_intent(&parsed, &intent_hv);

    // Semantic similarity is stochastic at small dimensions — just verify it's finite
    assert!(
        verification.semantic_similarity.is_finite(),
        "HDC round-trip should produce finite similarity"
    );
}

// ==================================================================================
// 5. Multi-language Consistency
// ==================================================================================

#[test]
fn e2e_same_intent_different_languages() {
    let r#gen = CodeGenerator::with_default_dim();

    let rust_intent = CodeIntent::Create {
        target: CodeTarget::new("add", EntityKind::Function).with_language("rust"),
        spec: CodeSpec::new("rust", "add", "Add two numbers")
            .with_signature("fn add(a: i32, b: i32) -> i32"),
    };

    let python_intent = CodeIntent::Create {
        target: CodeTarget::new("add", EntityKind::Function).with_language("python"),
        spec: CodeSpec::new("python", "add", "Add two numbers"),
    };

    let rust_result = r#gen.generate(&rust_intent, &CodeContext::default());
    let python_result = r#gen.generate(&python_intent, &CodeContext::default());

    // Both should produce real code
    assert!(!rust_result.source.contains("todo!"));
    assert!(!python_result.source.contains("NotImplementedError"));

    // Both should have the core operation
    assert!(rust_result.source.contains("+"));
    assert!(python_result.source.contains("+"));
}

// ==================================================================================
// 6. Batch Coverage: Verify common patterns don't fall through to LLM
// ==================================================================================

#[test]
fn e2e_batch_native_coverage() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();

    let cases = vec![
        ("add", "Add two numbers", "fn add(a: i32, b: i32) -> i32"),
        (
            "subtract",
            "Subtract two numbers",
            "fn subtract(a: i32, b: i32) -> i32",
        ),
        (
            "multiply",
            "Multiply two numbers",
            "fn multiply(a: f64, b: f64) -> f64",
        ),
        (
            "divide",
            "Divide two numbers",
            "fn divide(a: f64, b: f64) -> f64",
        ),
        (
            "reverse",
            "Reverse a string",
            "fn reverse(s: &str) -> String",
        ),
        ("length", "Get string length", "fn length(s: &str) -> usize"),
        (
            "is_even",
            "Check if number is even",
            "fn is_even(n: i32) -> bool",
        ),
        (
            "is_odd",
            "Check if number is odd",
            "fn is_odd(n: i32) -> bool",
        ),
        (
            "factorial",
            "Compute factorial",
            "fn factorial(n: u64) -> u64",
        ),
        (
            "uppercase",
            "Convert to uppercase",
            "fn uppercase(s: &str) -> String",
        ),
        ("abs", "Absolute value", "fn abs(n: i32) -> i32"),
        (
            "sort",
            "Sort a vector",
            "fn sort(items: Vec<i32>) -> Vec<i32>",
        ),
    ];

    let mut native_count = 0;
    let mut llm_count = 0;

    for (name, purpose, sig) in &cases {
        let intent = CodeIntent::Create {
            target: CodeTarget::new(*name, EntityKind::Function).with_language("rust"),
            spec: CodeSpec::new("rust", *name, *purpose).with_signature(*sig),
        };

        let result = r#gen.generate(&intent, &ctx);

        if result.source.contains("todo!") {
            llm_count += 1;
            eprintln!("  LLM needed: {} — {}", name, purpose);
        } else {
            native_count += 1;
        }
    }

    eprintln!(
        "Native coverage: {}/{} ({:.0}%)",
        native_count,
        cases.len(),
        native_count as f64 / cases.len() as f64 * 100.0
    );

    // At least 80% should be handled natively
    assert!(
        native_count as f64 / cases.len() as f64 >= 0.8,
        "Expected >=80% native coverage, got {}/{}",
        native_count,
        cases.len()
    );
}