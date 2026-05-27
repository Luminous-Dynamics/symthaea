#![cfg(all(feature = "code_generation", feature = "school_learning"))]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Focused honesty regressions for simulated code execution surfaces.

use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::language::code_discovery::CodeDiscovery;
use symthaea::language::code_executor::{CodeExecutor, ExecutionResult};
use symthaea::language::code_generator::CodeGenerator;
use symthaea::school::code_learning::{CodeLearningEngine, TIER1_OBJECTIVES};

fn make_generator() -> CodeGenerator {
    CodeGenerator::new(CodeHDEncoder::new(256))
}

#[test]
fn simulated_execution_result_is_not_success() {
    let result = ExecutionResult {
        compiled: true,
        compile_errors: Vec::new(),
        tests_passed: 1,
        tests_failed: 0,
        test_output: String::new(),
        runtime_error: None,
        elapsed: std::time::Duration::default(),
        simulated: true,
        test_failures: Vec::new(),
    };

    assert!(
        !result.is_success(),
        "Simulated execution must not count as a real success"
    );
}

#[test]
fn code_learning_defaults_to_simulated_execution() {
    let engine = CodeLearningEngine::new(make_generator());
    assert!(!engine.supports_real_execution());
}

#[test]
fn simulated_code_learning_session_reports_simulation() {
    let mut engine = CodeLearningEngine::new(make_generator());
    let summary = engine.run_session(&[TIER1_OBJECTIVES[0]]);

    assert_eq!(summary.lessons_attempted, summary.simulated_lessons);
    assert_eq!(summary.lessons_compiled, 0);
    assert_eq!(summary.compile_rate(), 0.0);
    assert_eq!(summary.pass_rate(), 0.0);
    assert_eq!(summary.simulated_rate(), 100.0);
    assert!(
        summary
            .outcomes
            .iter()
            .all(|outcome| outcome.simulated_execution)
    );
    assert!(summary.outcomes.iter().all(|outcome| !outcome.compiled));
}

#[test]
fn discovery_rejects_simulated_execution() {
    let mut discovery = CodeDiscovery::new(256);
    let mut executor = CodeExecutor::new();

    let result = discovery.discover(
        "increment an integer",
        "pub fn increment(x: i32) -> i32",
        "x",
        "i32",
        "i32",
        "#[test]\nfn test_increment() { assert_eq!(increment(2), 3); }",
        &mut executor,
    );

    assert!(!result.found);
    assert!(result.simulated_execution);
    assert_eq!(result.best_fitness, 0.0);
    assert_eq!(result.generations, 0);
    assert_eq!(result.compiled_count, 0);
}
