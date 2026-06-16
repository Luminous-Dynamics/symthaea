#![cfg(feature = "code_generation")]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Focused verification regressions for compiler-grounded code generation.

use symthaea::hdc::code_encoder::CodeHDEncoder;
use symthaea::language::code_executor::CodeExecutor;
use symthaea::language::code_generator::CodeGenerator;
use symthaea::language::verified_generation::generate_verified_function;

fn make_generator() -> CodeGenerator {
    CodeGenerator::new(CodeHDEncoder::new(512))
}

#[test]
fn verified_generation_rejects_simulated_execution() {
    let generator = make_generator();
    let mut executor = CodeExecutor::new();

    let result = generate_verified_function(
        &generator,
        &mut executor,
        "add",
        "Add two integers",
        "fn add(a: i32, b: i32) -> i32",
        &[("add(2, 3)", "5")],
    );

    assert!(!result.compiled);
    assert!(!result.tests_passed);
    assert!(
        result
            .compile_errors
            .iter()
            .any(|err| err.contains("requires real execution")),
        "expected real-execution failure, got {:?}",
        result.compile_errors
    );
}

#[test]
fn verified_generation_real_execution_fails_gracefully_when_toolchain_is_limited() {
    let generator = make_generator();
    let mut executor = CodeExecutor::with_real_execution();

    let result = generate_verified_function(
        &generator,
        &mut executor,
        "add",
        "Add two integers",
        "fn add(a: i32, b: i32) -> i32",
        &[("add(2, 3)", "5"), ("add(0, 0)", "0")],
    );

    if !result.is_guaranteed() {
        assert!(
            !result.compile_errors.is_empty() || !result.test_failures.is_empty(),
            "real execution must surface explicit failures instead of crashing or silently succeeding: {}",
            result.summary()
        );
        assert!(
            result
                .compile_errors
                .iter()
                .all(|err| !err.contains("requires real execution")),
            "real executor should not be misreported as simulation-only: {:?}",
            result.compile_errors
        );
        if result
            .compile_errors
            .iter()
            .any(|err| err.contains("linking with `cc` failed"))
        {
            assert!(
                result.compile_errors.iter().any(|err| err.contains('\n')),
                "linker failures should preserve context, got {:?}",
                result.compile_errors
            );
        }
    }
}
