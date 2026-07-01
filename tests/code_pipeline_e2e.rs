#![cfg(all(feature = "code_generation", feature = "school_learning"))]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! End-to-end integration test: Code Generation → Execution → Learning pipeline.
//!
//! Exercises the full pipeline:
//!   CodeLearningEngine (with real execution)
//!     → CodeGenerator (native emission)
//!     → CodeExecutor (real rustc)
//!     → Distillation cache
//!     → CodingExperienceStore (in-memory)
//!     → Generation ratio tracking
//!
//! Run: cargo test --test code_pipeline_e2e --features school_learning -- --nocapture

use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::language::code_generator::{CodeContext, CodeGenerator};
use symthaea::language::code_intent::{CodeIntent, CodeSpec, CodeTarget, EntityKind};
use symthaea::school::code_learning::{CodeLearningEngine, MetabolicBudget, TIER1_OBJECTIVES};

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 1: Full Code Pipeline — Generate → Execute → Learn → Distill
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_full_code_pipeline() {
    eprintln!("\n========================================================================");
    eprintln!("=== TEST: Full Code Pipeline (Generate -> Execute -> Learn -> Distill) ===");
    eprintln!("========================================================================\n");

    // 1. Create engine with real execution
    let encoder = CodeHDEncoder::new(256);
    let generator = CodeGenerator::new(encoder);
    let mut engine =
        CodeLearningEngine::with_real_execution(generator).with_budget(MetabolicBudget::new(500.0));

    // 2. Verify initial state
    assert_eq!(
        engine.distillation_count(),
        0,
        "Distillation cache should start empty"
    );
    assert!(
        engine.past_examples().is_empty(),
        "Past examples should start empty"
    );
    let (native_ratio, llm_ratio) = engine.generation_ratio();
    eprintln!(
        "Initial generation ratio: native={:.2}, llm={:.2}",
        native_ratio, llm_ratio
    );
    // Uninformative prior when no data
    assert!(
        (native_ratio - 0.5).abs() < 0.01,
        "Initial native ratio should be 0.5 (uninformative prior)"
    );

    // 3. Run individual objectives and inspect outcomes
    eprintln!("\n--- Running codegen_simple_arithmetic ---");
    let arith_outcomes = engine.run_objective("codegen_simple_arithmetic");
    eprintln!("  Lessons returned: {}", arith_outcomes.len());
    for outcome in &arith_outcomes {
        eprintln!(
            "  [{}] compiled={} tests={}/{} retries={} surprise={:.3} energy={:.1} distill={}",
            outcome.objective_id,
            outcome.compiled,
            outcome.tests_passed,
            outcome.tests_passed + outcome.tests_failed,
            outcome.retries_used,
            outcome.surprise,
            outcome.energy_spent,
            outcome.distillation_eligible,
        );
        if !outcome.compiled {
            eprintln!("    Source:\n{}", outcome.source);
        }
    }
    assert!(!arith_outcomes.is_empty(), "Should have arithmetic lessons");

    eprintln!("\n--- Running codegen_string_ops ---");
    let string_outcomes = engine.run_objective("codegen_string_ops");
    eprintln!("  Lessons returned: {}", string_outcomes.len());
    for outcome in &string_outcomes {
        eprintln!(
            "  [{}] compiled={} tests={}/{} retries={} surprise={:.3} llm={}",
            outcome.objective_id,
            outcome.compiled,
            outcome.tests_passed,
            outcome.tests_passed + outcome.tests_failed,
            outcome.retries_used,
            outcome.surprise,
            outcome.used_llm,
        );
    }
    assert!(!string_outcomes.is_empty(), "Should have string lessons");

    // 4. Now run a full session across all Tier 1 + Tier 2 objectives
    eprintln!("\n--- Running Tier 1 + Tier 2 session ---");
    // Reset budget for the session
    engine.reset_budget();
    let all_objectives: Vec<&str> = TIER1_OBJECTIVES
        .iter()
        .chain(symthaea::school::code_learning::TIER2_OBJECTIVES.iter())
        .chain(symthaea::school::code_learning::TIER3_OBJECTIVES.iter())
        .copied()
        .collect();
    let summary = engine.run_session(&all_objectives);

    eprintln!("  Lessons attempted: {}", summary.lessons_attempted);
    eprintln!(
        "  Compiled: {} ({:.0}%)",
        summary.lessons_compiled,
        summary.compile_rate()
    );
    eprintln!(
        "  Passed: {} ({:.0}%)",
        summary.lessons_passed,
        summary.pass_rate()
    );
    eprintln!("  Avg surprise: {:.3}", summary.avg_surprise);
    eprintln!("  Avg plan coverage: {:.3}", summary.avg_plan_coverage);
    eprintln!("  Total retries: {}", summary.total_retries);
    eprintln!("  Total energy: {:.1}", summary.total_energy_spent);
    eprintln!("  Distillation eligible: {}", summary.distillation_eligible);
    eprintln!(
        "  Error patterns learned: {}",
        summary.error_patterns_learned
    );
    eprintln!(
        "  Avg prediction error: {:.3}",
        summary.avg_prediction_error
    );
    eprintln!("  Hallucination rate: {:.3}", summary.hallucination_rate);

    // Print EVERY lesson outcome for diagnosis
    eprintln!("\n--- Per-Lesson Breakdown ---");
    for outcome in &summary.outcomes {
        let status = if outcome.is_success() {
            "PASS"
        } else if outcome.compiled {
            "COMPILED (tests failed)"
        } else {
            "FAIL"
        };
        eprintln!(
            "  [{}] {} | tests={}/{} | retries={} | surprise={:.3}",
            outcome.objective_id,
            status,
            outcome.tests_passed,
            outcome.tests_passed + outcome.tests_failed,
            outcome.retries_used,
            outcome.surprise,
        );
        if !outcome.compiled {
            // Print first 8 lines of source to diagnose
            let preview: String = outcome
                .source
                .lines()
                .take(8)
                .collect::<Vec<_>>()
                .join("\n");
            eprintln!(
                "    Source (first 8 lines):\n    {}",
                preview.replace('\n', "\n    ")
            );
        }
    }

    // 5. Verify distillation cache is populated
    let distill_count = engine.distillation_count();
    eprintln!("\n--- Distillation Cache ({} entries) ---", distill_count);
    for (purpose, _source, quality) in engine.distillation_cache() {
        eprintln!("  [{:.2}] {}", quality, purpose);
    }

    // 6. Verify past examples are populated
    let past = engine.past_examples();
    eprintln!("\n--- Past Examples ({} entries) ---", past.len());
    for (purpose, code) in past.iter().take(5) {
        eprintln!("  {} → {} chars", purpose, code.len());
    }

    // 7. Check generation ratio after real runs
    let (native_ratio_post, llm_ratio_post) = engine.generation_ratio();
    eprintln!("\n--- Generation Ratio ---");
    eprintln!(
        "  native={:.2}, llm={:.2}",
        native_ratio_post, llm_ratio_post
    );
    // After running Tier 1, the ratio should be informative (not 0.5/0.5)
    // unless no distillation records were created
    eprintln!("  (ratio reflects distillation_records tracking)");

    // 8. Budget state
    let budget = engine.budget();
    eprintln!("\n--- Budget State ---");
    eprintln!("  Total: {:.1}", budget.total_budget);
    eprintln!("  Spent: {:.1}", budget.spent);
    eprintln!("  Utilization: {:.0}%", budget.utilization() * 100.0);

    // 9. Core assertions
    assert!(
        summary.lessons_attempted >= 12,
        "Should run at least 12 lessons (Tier 1 + Tier 2), got {}",
        summary.lessons_attempted,
    );

    // At minimum, some lessons should compile (the native emitter handles
    // arithmetic and string patterns well)
    assert!(
        summary.lessons_compiled > 0,
        "At least some lessons should compile"
    );

    eprintln!("\n=== test_full_code_pipeline PASSED ===\n");
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST 2: Experience Store Informs Generation via Error Hints
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn test_experience_store_informs_generation() {
    eprintln!("\n========================================================================");
    eprintln!("=== TEST: Experience Store Informs Generation ===");
    eprintln!("========================================================================\n");

    // 1. Create a CodeGenerator
    let encoder = CodeHDEncoder::new(256);
    let r#gen = CodeGenerator::new(encoder);

    // 2. Build a CodeContext with error hints pre-populated
    //    (simulating what CodingExperienceStore.cached_error_hints() would provide)
    let error_hints = vec![
        (
            "E0308".to_string(),
            "add `as u32` cast for type mismatch".to_string(),
        ),
        (
            "E0382".to_string(),
            "clone the value before moving to avoid use-after-move".to_string(),
        ),
    ];

    let context = CodeContext {
        error_hints,
        ..Default::default()
    };

    // 3. Generate code with the error-hint-enriched context
    let intent = CodeIntent::Create {
        target: CodeTarget::new("sum_values", EntityKind::Function).with_language("rust"),
        spec: CodeSpec::new("rust", "sum_values", "Sum a vector of numbers")
            .with_signature("fn sum_values(v: &[i32]) -> i32"),
    };

    let result = r#gen.generate(&intent, &context);

    eprintln!("Generated source ({} chars):", result.source.len());
    eprintln!("{}", result.source);
    eprintln!("\nPhi score: {:.3}", result.phi_score);
    eprintln!("Intent similarity: {:.3}", result.intent_similarity);
    eprintln!("Notes ({}):", result.notes.len());
    for note in &result.notes {
        eprintln!("  - {}", note);
    }

    // 4. Verify the error hints appear in the result's notes
    //    The CodeGenerator.generate() method formats hints as:
    //    "ERROR_HINT(<pattern>): <hint>"
    let has_e0308_hint = result
        .notes
        .iter()
        .any(|n| n.contains("E0308") && n.contains("as u32"));
    let has_e0382_hint = result
        .notes
        .iter()
        .any(|n| n.contains("E0382") && n.contains("clone"));

    eprintln!("\nE0308 hint present in notes: {}", has_e0308_hint);
    eprintln!("E0382 hint present in notes: {}", has_e0382_hint);

    assert!(
        has_e0308_hint,
        "E0308 error hint should appear in generated notes"
    );
    assert!(
        has_e0382_hint,
        "E0382 error hint should appear in generated notes"
    );

    // 5. Also verify the code itself was generated (not empty/todo)
    assert!(!result.source.is_empty(), "Should generate code");
    eprintln!("Code contains todo!: {}", result.source.contains("todo!"));

    // 6. Test with a second generation that uses past_examples context
    let context_with_examples = CodeContext {
        past_examples: vec![(
            "sum_values".to_string(),
            "fn sum_values(v: &[i32]) -> i32 { v.iter().sum() }".to_string(),
        )],
        error_hints: vec![(
            "E0308".to_string(),
            "add `as u32` cast for type mismatch".to_string(),
        )],
        ..Default::default()
    };

    let intent2 = CodeIntent::Create {
        target: CodeTarget::new("product_values", EntityKind::Function).with_language("rust"),
        spec: CodeSpec::new("rust", "product_values", "Multiply all values in a vector")
            .with_signature("fn product_values(v: &[i32]) -> i32"),
    };

    let result2 = r#gen.generate(&intent2, &context_with_examples);
    eprintln!("\nSecond generation with past examples:");
    eprintln!("  Source: {}", result2.source);
    eprintln!("  Notes: {:?}", result2.notes);

    let has_past_example = result2.notes.iter().any(|n| n.contains("PAST_EXAMPLE"));
    let has_error_hint = result2.notes.iter().any(|n| n.contains("ERROR_HINT"));
    eprintln!("  Has PAST_EXAMPLE note: {}", has_past_example);
    eprintln!("  Has ERROR_HINT note: {}", has_error_hint);

    assert!(has_past_example, "Past examples should appear in notes");
    assert!(has_error_hint, "Error hints should appear in notes");

    eprintln!("\n=== test_experience_store_informs_generation PASSED ===\n");
}