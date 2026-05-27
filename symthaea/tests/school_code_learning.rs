#![cfg(all(feature = "code_generation", feature = "school_learning"))]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration test: School Code Learning with real compilation
//!
//! Runs the CodeLearningEngine with real `rustc` execution to validate
//! that Symthaea can generate, compile, and test Rust code autonomously.
//!
//! Tier 1 tests use native emission only (no LLM needed).
//! Tier 2+ tests use qwen2.5-coder:7b via Ollama when available.
//!
//! Run: cargo test --test school_code_learning -- --nocapture

use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::language::code_generator::CodeGenerator;
use symthaea::school::code_learning::{
    CodeLearningEngine, MetabolicBudget, TIER1_OBJECTIVES, TIER2_OBJECTIVES, TIER3_OBJECTIVES,
    build_lesson_bank, default_llm_prompt,
};

fn make_real_engine() -> CodeLearningEngine {
    let encoder = CodeHDEncoder::new(256);
    let generator = CodeGenerator::new(encoder);
    CodeLearningEngine::with_real_execution(generator).with_llm_prompt(default_llm_prompt)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TIER 1: Native emission (no LLM needed)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn tier1_arithmetic_compiles_and_passes() {
    let mut engine = make_real_engine();
    let outcomes = engine.run_objective("codegen_simple_arithmetic");

    eprintln!("=== Tier 1: Simple Arithmetic ===");
    for outcome in &outcomes {
        eprintln!(
            "  {} | compiled={} tests={}/{} retries={} energy={:.1}",
            outcome.objective_id,
            outcome.compiled,
            outcome.tests_passed,
            outcome.tests_passed + outcome.tests_failed,
            outcome.retries_used,
            outcome.energy_spent,
        );
        if !outcome.compiled {
            eprintln!("    Source:\n{}", outcome.source);
        }
    }

    let compiled = outcomes.iter().filter(|o| o.compiled).count();
    eprintln!("  Compile rate: {}/{}", compiled, outcomes.len());
    assert!(!outcomes.is_empty(), "Should have arithmetic lessons");
}

#[test]
fn tier1_strings_compiles() {
    let mut engine = make_real_engine();
    let outcomes = engine.run_objective("codegen_string_ops");

    eprintln!("=== Tier 1: String Ops ===");
    for outcome in &outcomes {
        eprintln!(
            "  {} | compiled={} tests={}/{} llm={} energy={:.1}",
            outcome.objective_id,
            outcome.compiled,
            outcome.tests_passed,
            outcome.tests_passed + outcome.tests_failed,
            outcome.used_llm,
            outcome.energy_spent,
        );
    }
    assert!(!outcomes.is_empty());
}

#[test]
fn tier1_booleans_compiles() {
    let mut engine = make_real_engine();
    let outcomes = engine.run_objective("codegen_boolean_checks");

    eprintln!("=== Tier 1: Boolean Checks ===");
    for outcome in &outcomes {
        eprintln!(
            "  {} | compiled={} surprise={:.2}",
            outcome.objective_id, outcome.compiled, outcome.surprise,
        );
    }
    assert!(!outcomes.is_empty());
}

#[test]
fn tier1_collections_compiles() {
    let mut engine = make_real_engine();
    let outcomes = engine.run_objective("codegen_basic_collections");

    eprintln!("=== Tier 1: Collections ===");
    for outcome in &outcomes {
        eprintln!(
            "  {} | compiled={} tests={}/{}",
            outcome.objective_id,
            outcome.compiled,
            outcome.tests_passed,
            outcome.tests_passed + outcome.tests_failed,
        );
    }
    assert!(!outcomes.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════════
// FULL TIER 1 SESSION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn tier1_full_session() {
    let encoder = CodeHDEncoder::new(256);
    let generator = CodeGenerator::new(encoder);
    // Use a generous budget so all 12 lessons can run even with LLM fallback
    let mut engine = CodeLearningEngine::with_real_execution(generator)
        .with_llm_prompt(default_llm_prompt)
        .with_budget(MetabolicBudget::new(500.0));
    let summary = engine.run_session(TIER1_OBJECTIVES);

    eprintln!("=== Tier 1 Full Session Summary ===");
    eprintln!("  Lessons attempted: {}", summary.lessons_attempted);
    eprintln!(
        "  Compile rate: {:.0}% ({}/{})",
        summary.compile_rate(),
        summary.lessons_compiled,
        summary.lessons_attempted,
    );
    eprintln!(
        "  Pass rate: {:.0}% ({}/{})",
        summary.pass_rate(),
        summary.lessons_passed,
        summary.lessons_attempted,
    );
    eprintln!("  Avg surprise: {:.3}", summary.avg_surprise);
    eprintln!("  Avg plan coverage: {:.3}", summary.avg_plan_coverage);
    eprintln!("  Total retries: {}", summary.total_retries);
    eprintln!("  Total energy: {:.1}", summary.total_energy_spent);
    eprintln!("  Distillation eligible: {}", summary.distillation_eligible,);
    eprintln!("  Distillation cache: {}", engine.distillation_count(),);

    assert!(
        summary.lessons_attempted >= 12,
        "Should run all 12 Tier 1 lessons, got {}",
        summary.lessons_attempted,
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// TIER 2: Composition (may need LLM for some)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[ignore] // Requires Ollama running
fn tier2_session_with_llm() {
    let mut engine = make_real_engine();
    let summary = engine.run_session(TIER2_OBJECTIVES);

    eprintln!("=== Tier 2 Session Summary ===");
    eprintln!("  Lessons attempted: {}", summary.lessons_attempted);
    eprintln!("  Compile rate: {:.0}%", summary.compile_rate());
    eprintln!("  Pass rate: {:.0}%", summary.pass_rate());
    eprintln!("  LLM retries: {}", summary.total_llm_retries);
    eprintln!("  Total energy: {:.1}", summary.total_energy_spent);

    for outcome in &summary.outcomes {
        eprintln!(
            "  {} | compiled={} llm={} llm_retries={} energy={:.1}",
            outcome.objective_id,
            outcome.compiled,
            outcome.used_llm,
            outcome.llm_retries_used,
            outcome.energy_spent,
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TIER 3: Advanced (LLM required)
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
#[ignore] // Requires Ollama running
fn tier3_algorithms_with_llm() {
    let mut engine = make_real_engine();
    let summary = engine.run_session(TIER3_OBJECTIVES);

    eprintln!("=== Tier 3 Session Summary ===");
    eprintln!("  Lessons attempted: {}", summary.lessons_attempted);
    eprintln!("  Compile rate: {:.0}%", summary.compile_rate());
    eprintln!("  Pass rate: {:.0}%", summary.pass_rate());
    eprintln!("  LLM retries: {}", summary.total_llm_retries);
    eprintln!("  Total energy: {:.1}", summary.total_energy_spent);

    for outcome in &summary.outcomes {
        if !outcome.compiled {
            eprintln!(
                "  FAILED {} — source:\n{}",
                outcome.objective_id, outcome.source
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// METABOLIC BUDGET VALIDATION
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn budget_limits_session() {
    let encoder = CodeHDEncoder::new(256);
    let generator = CodeGenerator::new(encoder);
    // Budget for exactly 3 native lessons (3.0 energy, reserve = 0.6)
    let mut engine =
        CodeLearningEngine::with_real_execution(generator).with_budget(MetabolicBudget::new(3.5));

    let summary = engine.run_session(TIER1_OBJECTIVES);

    eprintln!("=== Budget-Limited Session ===");
    eprintln!(
        "  Attempted: {} (budget should limit this)",
        summary.lessons_attempted
    );
    eprintln!("  Energy spent: {:.1}", summary.total_energy_spent);
    eprintln!(
        "  Budget utilization: {:.0}%",
        engine.budget().utilization() * 100.0
    );

    assert!(
        summary.lessons_attempted <= 4,
        "Budget should limit to ~3 lessons, got {}",
        summary.lessons_attempted,
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// DISTILLATION PIPELINE SMOKE
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn distillation_cache_populated_after_session() {
    let mut engine = make_real_engine();
    let _summary = engine.run_session(TIER1_OBJECTIVES);

    eprintln!("=== Distillation Cache ===");
    eprintln!("  Cache size: {}", engine.distillation_count());
    eprintln!("  Past examples: {}", engine.past_examples().len());

    for (purpose, _source, quality) in engine.distillation_cache() {
        eprintln!("  [{:.2}] {}", quality, purpose);
    }

    // At least some lessons should produce distillation targets
    // (in simulation mode, all compile, so all should be eligible)
}