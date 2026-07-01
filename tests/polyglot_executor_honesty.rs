#![cfg(feature = "code_generation")]
// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real-execution regressions for the non-Rust coding surfaces.

use std::process::Command;

use symthaea::language::code_executor::CodeExecutor;

#[test]
fn python_real_execution_is_required_for_success() {
    let mut simulated_executor = CodeExecutor::new();
    let simulated = simulated_executor.execute_python("assert 2 + 2 == 4");
    assert!(simulated.simulated);
    assert!(
        !simulated.is_success(),
        "simulated Python execution must not count as success"
    );

    let mut executor = CodeExecutor::with_real_execution();
    assert!(executor.supports_real_execution());

    let result = executor.execute_python("assert 2 + 2 == 4\nprint('python-ok')");
    assert!(
        result.is_success(),
        "real Python execution should pass: {:?} {:?}",
        result.compile_errors,
        result.runtime_error
    );
    assert!(!result.simulated);
    assert!(result.test_output.contains("python-ok"));
}

#[test]
fn python_runtime_failures_are_not_success() {
    let mut executor = CodeExecutor::with_real_execution();

    let result = executor.execute_python("assert 2 + 2 == 5");
    assert!(!result.is_success());
    assert!(!result.simulated);
    assert_eq!(result.tests_failed, 1);
}

#[test]
fn nix_real_evaluation_is_required_for_success() {
    if Command::new("nix-instantiate")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("[skip] nix-instantiate not available");
        return;
    }

    let mut simulated_executor = CodeExecutor::new();
    let simulated = simulated_executor.evaluate_nix("1 + 1");
    assert!(simulated.simulated);
    assert!(
        !simulated.is_success(),
        "simulated Nix evaluation must not count as success"
    );

    let mut executor = CodeExecutor::with_real_execution();
    assert!(executor.supports_real_execution());

    let result = executor.evaluate_nix("1 + 1");
    assert!(
        result.is_success(),
        "real Nix evaluation should pass: {:?}",
        result.compile_errors
    );
    assert!(!result.simulated);
    assert!(result.test_output.contains('2'));
}

#[test]
fn nix_evaluation_failures_are_not_success() {
    if Command::new("nix-instantiate")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("[skip] nix-instantiate not available");
        return;
    }

    let mut executor = CodeExecutor::with_real_execution();
    let result = executor.evaluate_nix("let x = ; in x");

    assert!(!result.is_success());
    assert!(!result.simulated);
    assert!(!result.compile_errors.is_empty());
}
