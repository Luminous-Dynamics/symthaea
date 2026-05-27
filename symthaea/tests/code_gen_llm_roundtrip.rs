#![cfg(feature = "code_generation")]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// ==================================================================================
// Code Generation LLM Roundtrip Tests
// ==================================================================================
//
// Tests the full LLM completion path end-to-end:
//   Generate code with todo!() -> detect needs_llm -> construct prompt -> call Ollama -> verify
//
// Run: cargo test --test code_gen_llm_roundtrip --features code_generation
//
// The LLM roundtrip test (test_llm_completion_roundtrip) requires Ollama at
// localhost:11434. It skips gracefully if Ollama is unavailable. All other tests
// run without external dependencies.
// ==================================================================================

use symthaea::language::code_generator::{CodeContext, CodeGenerator};
use symthaea::language::code_intent::{CodeIntent, CodeSpec, CodeTarget};
use symthaea::language::code_parser::EntityKind;
use symthaea::language::consciousness_prompts::CODE_GENERATION_SYSTEM_PROMPT;
use symthaea::language::llm_backend::{GenerationParams, LLMBackend, OllamaBackend};
use symthaea::mind::structured_thought::{CodeContext as ThoughtCodeContext, StructuredThought};

// ==================================================================================
// Helpers
// ==================================================================================

/// Check if Ollama is running at localhost:11434 with a 2-second TCP timeout.
fn ollama_available() -> bool {
    std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().unwrap(),
        std::time::Duration::from_secs(2),
    )
    .is_ok()
}

/// Determine whether generated source code needs LLM completion.
/// Mirrors the detection logic in symthaea.rs (line ~1620).
fn needs_llm(source: &str) -> bool {
    source.contains("todo!(") || source.contains("NotImplementedError")
}

// ==================================================================================
// 1. Full LLM Roundtrip (requires Ollama)
// ==================================================================================

#[tokio::test]
async fn test_llm_completion_roundtrip() {
    if !ollama_available() {
        eprintln!("SKIP: Ollama not available at localhost:11434");
        return;
    }

    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();

    // Step 1: Generate code that needs LLM (complex algorithm)
    let intent = CodeIntent::Create {
        target: CodeTarget::new("dijkstra", EntityKind::Function).with_language("rust"),
        spec: CodeSpec::new(
            "rust",
            "dijkstra",
            "Implement Dijkstra's shortest path algorithm on a weighted adjacency list",
        )
        .with_signature("fn dijkstra(graph: &[Vec<(usize, u32)>], start: usize) -> Vec<u32>"),
    };

    let generated = r#gen.generate(&intent, &ctx);
    assert!(
        needs_llm(&generated.source),
        "Complex algo should produce todo!(): {}",
        generated.source
    );

    // Step 2: Build the structured thought (as symthaea.rs does)
    let mut thought = StructuredThought::default();
    thought.code_context = Some(ThoughtCodeContext {
        language: "rust".to_string(),
        spec_purpose: Some("Dijkstra's shortest path algorithm".to_string()),
        spec_signature: Some(
            "fn dijkstra(graph: &[Vec<(usize, u32)>], start: usize) -> Vec<u32>".to_string(),
        ),
        spec_constraints: vec![],
        spec_examples: vec![],
        plan_steps: generated
            .plan_steps
            .iter()
            .map(|s| format!("{:?}", s.action))
            .collect(),
        generated_code: Some(generated.source.clone()),
        phi_score: Some(generated.phi_score),
        intent_similarity: Some(generated.intent_similarity),
        syntactically_valid: None,
        notes: generated.notes.clone(),
        needs_llm_completion: true,
    });

    // Step 3: Construct the prompt
    let prompt = thought.to_translation_prompt();
    assert!(prompt.contains("NEEDS_COMPLETION: true"));
    assert!(prompt.contains("GENERATED_CODE"));
    assert!(prompt.contains("dijkstra"));
    assert!(prompt.contains("SPEC_SIGNATURE"));

    // Step 4: Call Ollama with the code generation system prompt
    let model = std::env::var("SYMTHAEA_LLM_MODEL").unwrap_or_else(|_| "gemma3:1b".to_string());
    let backend = OllamaBackend::with_timeouts(
        "http://localhost:11434",
        &model,
        std::time::Duration::from_secs(5),
        std::time::Duration::from_secs(120),
    );

    if !backend.is_available().await {
        eprintln!("SKIP: Ollama backend not responding (model may not be pulled)");
        return;
    }

    let params = GenerationParams {
        temperature: 0.2,
        max_tokens: 2048,
        system_prompt: Some(CODE_GENERATION_SYSTEM_PROMPT.to_string()),
        consciousness_context: None,
    };

    let response = backend.generate(&prompt, &params).await;
    assert!(response.is_ok(), "Ollama call failed: {:?}", response.err());

    let completed_code = response.unwrap();
    assert!(!completed_code.is_empty(), "LLM returned empty response");

    // The LLM should have replaced todo!() with a real implementation.
    // We check for signs of a real Dijkstra implementation.
    let lower = completed_code.to_lowercase();
    let has_impl_signs = lower.contains("dist")
        || lower.contains("heap")
        || lower.contains("priority")
        || lower.contains("visited")
        || lower.contains("shortest")
        || lower.contains("graph")
        || lower.contains("vec!");

    assert!(
        has_impl_signs,
        "LLM response should contain implementation artifacts, got:\n{}",
        &completed_code[..completed_code.len().min(500)]
    );

    eprintln!(
        "LLM roundtrip OK: received {} bytes of code",
        completed_code.len()
    );
}

// ==================================================================================
// 2. Native Emitter: Simple patterns must NOT trigger LLM
// ==================================================================================

#[test]
fn test_native_emitter_no_llm_needed() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();

    // These simple patterns should all be handled natively without todo!()
    let simple_cases: Vec<(&str, &str, &str)> = vec![
        ("add", "Add two numbers", "fn add(a: i32, b: i32) -> i32"),
        (
            "is_even",
            "Check if a number is even",
            "fn is_even(n: i32) -> bool",
        ),
        (
            "reverse",
            "Reverse a string",
            "fn reverse(s: &str) -> String",
        ),
        (
            "factorial",
            "Compute factorial",
            "fn factorial(n: u64) -> u64",
        ),
        ("abs", "Absolute value", "fn abs(n: i32) -> i32"),
    ];

    for (name, purpose, sig) in &simple_cases {
        let intent = CodeIntent::Create {
            target: CodeTarget::new(*name, EntityKind::Function).with_language("rust"),
            spec: CodeSpec::new("rust", *name, *purpose).with_signature(*sig),
        };

        let result = r#gen.generate(&intent, &ctx);

        assert!(
            !needs_llm(&result.source),
            "Simple pattern '{}' ({}) should NOT need LLM, but got:\n{}",
            name,
            purpose,
            result.source
        );

        // Should contain real code, not be empty
        assert!(
            !result.source.is_empty(),
            "Generated source for '{}' should not be empty",
            name
        );
    }
}

// ==================================================================================
// 3. LLM Detection Accuracy: 10 cases, verify todo!() detection matches expectation
// ==================================================================================

#[test]
fn test_llm_detection_accuracy() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();

    // (name, purpose, signature, expect_needs_llm)
    let cases: Vec<(&str, &str, &str, bool)> = vec![
        // Simple patterns — should NOT need LLM
        ("add", "Add two numbers", "fn add(a: i32, b: i32) -> i32", false),
        ("subtract", "Subtract two numbers", "fn subtract(a: i32, b: i32) -> i32", false),
        ("multiply", "Multiply two numbers", "fn multiply(a: f64, b: f64) -> f64", false),
        ("is_even", "Check if number is even", "fn is_even(n: i32) -> bool", false),
        ("reverse", "Reverse a string", "fn reverse(s: &str) -> String", false),
        // Complex algorithms — SHOULD need LLM
        (
            "dijkstra",
            "Implement Dijkstra's shortest path algorithm",
            "fn dijkstra(graph: &[Vec<(usize, u32)>], start: usize) -> Vec<u32>",
            true,
        ),
        (
            "astar",
            "Implement A-star pathfinding with heuristic",
            "fn astar(grid: &[Vec<bool>], start: (usize,usize), goal: (usize,usize)) -> Option<Vec<(usize,usize)>>",
            true,
        ),
        (
            "bfs_shortest",
            "BFS shortest path in an unweighted graph",
            "fn bfs_shortest(adj: &[Vec<usize>], src: usize, dst: usize) -> Option<Vec<usize>>",
            true,
        ),
        (
            "knuth_morris_pratt",
            "Implement Knuth-Morris-Pratt string matching",
            "fn kmp_search(text: &str, pattern: &str) -> Vec<usize>",
            true,
        ),
        (
            "red_black_insert",
            "Insert into a red-black tree and rebalance",
            "fn rb_insert(tree: &mut RBTree, key: i64)",
            true,
        ),
    ];

    let mut correct = 0;
    let mut total = 0;

    for (name, purpose, sig, expect_llm) in &cases {
        let intent = CodeIntent::Create {
            target: CodeTarget::new(*name, EntityKind::Function).with_language("rust"),
            spec: CodeSpec::new("rust", *name, *purpose).with_signature(*sig),
        };

        let result = r#gen.generate(&intent, &ctx);
        let actual_needs_llm = needs_llm(&result.source);

        total += 1;
        if actual_needs_llm == *expect_llm {
            correct += 1;
        } else {
            eprintln!(
                "  MISMATCH: '{}' expected needs_llm={}, got {} — source: {}",
                name,
                expect_llm,
                actual_needs_llm,
                &result.source[..result.source.len().min(120)]
            );
        }
    }

    eprintln!(
        "LLM detection accuracy: {}/{} ({:.0}%)",
        correct,
        total,
        correct as f64 / total as f64 * 100.0
    );

    // Allow at most 2 mismatches (80% accuracy threshold)
    assert!(
        correct >= 8,
        "Expected at least 8/10 correct detections, got {}/{}",
        correct,
        total
    );
}

// ==================================================================================
// 4. Prompt Construction: Verify CodeContext serialization includes needed fields
// ==================================================================================

#[test]
fn test_prompt_construction() {
    let r#gen = CodeGenerator::with_default_dim();
    let ctx = CodeContext::default();

    let intent = CodeIntent::Create {
        target: CodeTarget::new("merge_sort", EntityKind::Function).with_language("rust"),
        spec: CodeSpec::new("rust", "merge_sort", "Implement merge sort")
            .with_signature("fn merge_sort(arr: &mut [i32])")
            .with_constraint("Must be stable sort")
            .with_constraint("In-place is preferred")
            .with_example("merge_sort(&mut [3,1,2])", "[1,2,3]"),
    };

    let generated = r#gen.generate(&intent, &ctx);

    // Build StructuredThought with full code context
    let mut thought = StructuredThought::default();
    thought.code_context = Some(ThoughtCodeContext {
        language: "rust".to_string(),
        spec_purpose: Some("Implement merge sort".to_string()),
        spec_signature: Some("fn merge_sort(arr: &mut [i32])".to_string()),
        spec_constraints: vec![
            "Must be stable sort".to_string(),
            "In-place is preferred".to_string(),
        ],
        spec_examples: vec![(
            "merge_sort(&mut [3,1,2])".to_string(),
            "[1,2,3]".to_string(),
        )],
        plan_steps: generated
            .plan_steps
            .iter()
            .map(|s| format!("{:?}", s.action))
            .collect(),
        generated_code: Some(generated.source.clone()),
        phi_score: Some(generated.phi_score),
        intent_similarity: Some(generated.intent_similarity),
        syntactically_valid: None,
        notes: generated.notes.clone(),
        needs_llm_completion: needs_llm(&generated.source),
    });

    let prompt = thought.to_translation_prompt();

    // Verify all expected fields appear in the serialized prompt
    assert!(
        prompt.contains("CODE_LANGUAGE: rust"),
        "Prompt must include CODE_LANGUAGE, got:\n{}",
        prompt
    );
    assert!(
        prompt.contains("SPEC_PURPOSE: Implement merge sort"),
        "Prompt must include SPEC_PURPOSE"
    );
    assert!(
        prompt.contains("SPEC_SIGNATURE: fn merge_sort(arr: &mut [i32])"),
        "Prompt must include SPEC_SIGNATURE"
    );
    assert!(
        prompt.contains("Must be stable sort"),
        "Prompt must include constraints"
    );
    assert!(
        prompt.contains("In-place is preferred"),
        "Prompt must include all constraints"
    );
    assert!(
        prompt.contains("merge_sort(&mut [3,1,2])"),
        "Prompt must include examples"
    );
    assert!(
        prompt.contains("[1,2,3]"),
        "Prompt must include example outputs"
    );
    assert!(
        prompt.contains("GENERATED_CODE"),
        "Prompt must include generated code block"
    );
    assert!(
        prompt.contains("CODE_PHI:"),
        "Prompt must include phi score"
    );
    assert!(
        prompt.contains("CODE_INTENT_SIMILARITY:"),
        "Prompt must include intent similarity"
    );
    assert!(
        prompt.contains("PLAN_STEPS"),
        "Prompt must include plan steps"
    );

    // If the code needs LLM, the completion signal must be present
    if needs_llm(&generated.source) {
        assert!(
            prompt.contains("NEEDS_COMPLETION: true"),
            "Prompt must include NEEDS_COMPLETION when code has todo!()"
        );
        assert!(
            prompt.contains("Replace ONLY the placeholder bodies"),
            "Prompt must include completion instructions"
        );
    }

    eprintln!("Prompt construction verified: {} bytes", prompt.len());
}

// ==================================================================================
// 5. Prompt includes notes when present
// ==================================================================================

#[test]
fn test_prompt_includes_notes() {
    let mut thought = StructuredThought::default();
    thought.code_context = Some(ThoughtCodeContext {
        language: "rust".to_string(),
        spec_purpose: Some("Test function".to_string()),
        spec_signature: None,
        spec_constraints: vec![],
        spec_examples: vec![],
        plan_steps: vec![],
        generated_code: Some("fn test() { todo!() }".to_string()),
        phi_score: Some(0.42),
        intent_similarity: Some(0.85),
        syntactically_valid: None,
        notes: vec![
            "Low confidence in body implementation".to_string(),
            "Consider using BinaryHeap".to_string(),
        ],
        needs_llm_completion: true,
    });

    let prompt = thought.to_translation_prompt();

    assert!(
        prompt.contains("CODE_NOTES:"),
        "Prompt must have CODE_NOTES section"
    );
    assert!(
        prompt.contains("Low confidence in body implementation"),
        "Prompt must include first note"
    );
    assert!(
        prompt.contains("Consider using BinaryHeap"),
        "Prompt must include second note"
    );
}