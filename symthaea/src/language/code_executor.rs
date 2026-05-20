// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Code Executor — Sandboxed compilation and test execution
//!
//! Closes the code generation feedback loop: generate → compile → test → learn.
//! Uses `infrastructure::sandbox::Sandbox` for safe command execution.
//!
//! # Pipeline
//!
//! ```text
//! Generated source code
//!     ↓ write to tempdir
//! rustc/python/nix eval
//!     ↓ capture exit code + stderr
//! ExecutionResult { compiled, errors, test results }
//!     ↓ feed back as FEP surprise
//! CodeGenerator retry (if failed)
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::infrastructure::sandbox::{Sandbox, SandboxError};

static CODE_EXECUTOR_WORKDIR_SEQ: AtomicU64 = AtomicU64::new(0);

/// A parsed test failure with assertion details.
///
/// Enables semantic understanding of *why* a test failed, not just *that* it failed.
/// Used by the coding agent to feed failure constraints back into the next generation.
#[derive(Debug, Clone)]
pub struct TestFailure {
    /// Test name (e.g., "tests::test_add")
    pub test_name: String,
    /// Assertion text if extractable (e.g., "assert_eq!(add(2,3), 5)")
    pub assertion: Option<String>,
    /// Expected value if extractable (e.g., "5")
    pub expected: Option<String>,
    /// Actual value if extractable (e.g., "4")
    pub actual: Option<String>,
    /// Raw failure message
    pub message: String,
}

/// Result of executing generated code
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Whether the code compiled/parsed without errors
    pub compiled: bool,
    /// Compiler/interpreter error messages
    pub compile_errors: Vec<String>,
    /// Number of tests that passed (0 if no tests or not applicable)
    pub tests_passed: usize,
    /// Number of tests that failed
    pub tests_failed: usize,
    /// Raw test output
    pub test_output: String,
    /// Runtime error, if any
    pub runtime_error: Option<String>,
    /// Total execution time
    pub elapsed: Duration,
    /// Whether this was a simulation (no real execution)
    pub simulated: bool,
    /// Parsed test failures with semantic details (assertion, expected, actual).
    /// Populated by `parse_test_failures()` after test execution.
    pub test_failures: Vec<TestFailure>,
}

impl ExecutionResult {
    /// Whether the code is fully successful (compiled + all tests passed)
    pub fn is_success(&self) -> bool {
        !self.simulated && self.compiled && self.tests_failed == 0
    }

    /// Parse test failures from the raw test output, populating `test_failures`.
    ///
    /// Call this after construction to extract semantic details from rustc/cargo test output.
    /// Extracts test names, assertion text, expected/actual values from format:
    /// ```text
    /// ---- tests::test_add stdout ----
    /// thread 'tests::test_add' panicked at 'assertion `left == right` failed
    ///   left: 4
    ///  right: 5'
    /// ```
    pub fn parse_test_failures(&mut self) {
        self.test_failures = parse_test_failure_details(&self.test_output);
    }

    /// Get formatted constraint strings from test failures.
    ///
    /// Returns strings like "test_add: expected 5 but got 4" that can be
    /// injected into the next generation prompt as constraints.
    pub fn failure_constraints(&self) -> Vec<String> {
        self.test_failures
            .iter()
            .map(|f| match (&f.expected, &f.actual) {
                (Some(exp), Some(act)) => {
                    format!(
                        "CONSTRAINT: {} expected {} but got {}",
                        f.test_name, exp, act
                    )
                }
                _ => format!("CONSTRAINT: {} failed: {}", f.test_name, f.message),
            })
            .collect()
    }

    /// Convert to an FEP surprise signal in [0.0, 1.0].
    ///
    /// - Compilation failure: 0.8 + 0.2 * (errors / 10)
    /// - Test failure: 0.3 + 0.5 * (failed / total)
    /// - Success: 0.0
    pub fn to_surprise(&self) -> f32 {
        if !self.compiled {
            let error_factor = (self.compile_errors.len() as f32 / 10.0).min(1.0);
            0.8 + 0.2 * error_factor
        } else if self.tests_failed > 0 {
            let total = (self.tests_passed + self.tests_failed) as f32;
            if total > 0.0 {
                0.3 + 0.5 * (self.tests_failed as f32 / total)
            } else {
                0.3
            }
        } else {
            0.0
        }
    }

    /// Create a result for simulation mode
    fn simulated_success() -> Self {
        Self {
            compiled: true,
            compile_errors: Vec::new(),
            tests_passed: 0,
            tests_failed: 0,
            test_output: "[Simulated] Compilation successful".to_string(),
            runtime_error: None,
            elapsed: Duration::from_millis(50),
            simulated: true,
            test_failures: Vec::new(),
        }
    }
}

/// Sandboxed code executor for compilation and testing
pub struct CodeExecutor {
    sandbox: Sandbox,
    /// Temporary directory for writing source files
    work_dir: PathBuf,
}

impl CodeExecutor {
    /// Create a new code executor with a fresh sandbox.
    ///
    /// The sandbox starts in simulation mode by default for safety.
    /// Call `enable_real_execution()` to actually compile/run code.
    pub fn new() -> Self {
        let sandbox = Sandbox::new()
            .with_timeout(Duration::from_secs(30))
            .simulation_only();
        let work_dir = fresh_work_dir();
        Self { sandbox, work_dir }
    }

    /// Create an executor that actually runs commands.
    pub fn with_real_execution() -> Self {
        let mut sandbox = Sandbox::new()
            .with_timeout(Duration::from_secs(30))
            .enable_real_execution();
        // Add code-related commands to the allowlist
        sandbox.allow_command("rustc");
        sandbox.allow_command("cargo");
        sandbox.allow_command("python3");
        sandbox.allow_command("python");
        sandbox.allow_command("nix-instantiate");
        Self {
            sandbox,
            work_dir: fresh_work_dir(),
        }
    }

    /// Whether this executor can perform real, non-simulated verification work.
    pub fn supports_real_execution(&self) -> bool {
        !self.sandbox.is_simulation_only() && self.sandbox.is_real_execution_enabled()
    }

    /// Access the temporary work directory used by this executor.
    pub fn work_dir(&self) -> &PathBuf {
        &self.work_dir
    }

    /// Apply a Unified Diff (.patch) to a local repository
    pub fn apply_patch(
        &mut self,
        repo_path: &std::path::Path,
        patch_content: &str,
    ) -> Result<(), String> {
        let patch_file = self.work_dir.join("fix.patch");
        std::fs::write(&patch_file, patch_content).map_err(|e| e.to_string())?;

        // Use the sandbox to apply the patch securely
        self.sandbox.allow_command("bash");
        self.sandbox.allow_command("patch");

        let cmd = format!(
            "cd {} && patch -p1 < {}",
            repo_path.display(),
            patch_file.display()
        );
        let result = self
            .sandbox
            .run("bash", &["-c", &cmd])
            .map_err(|e| e.to_string())?;

        if result.success() {
            Ok(())
        } else {
            Err(result.stderr)
        }
    }

    /// Run a repository's full test suite
    pub fn execute_workspace_tests(&mut self, repo_path: &std::path::Path) -> ExecutionResult {
        let start = std::time::Instant::now();
        self.sandbox.allow_command("bash");
        self.sandbox.allow_command("cargo");

        let cmd = format!("cd {} && cargo test", repo_path.display());

        match self.sandbox.run("bash", &["-c", &cmd]) {
            Ok(result) => {
                if !result.success() {
                    let errors = parse_compile_errors(&result.stderr);
                    let (passed, failed) = parse_test_output(&result.stdout);

                    let mut exec_result = ExecutionResult {
                        compiled: failed > 0 || errors.is_empty(), // If tests ran and failed, it compiled
                        compile_errors: errors.clone(),
                        tests_passed: passed,
                        tests_failed: failed,
                        test_output: result.combined_output(),
                        runtime_error: None,
                        elapsed: start.elapsed(),
                        simulated: result.simulated,
                        test_failures: Vec::new(),
                    };
                    exec_result.parse_test_failures();
                    return exec_result;
                }

                let (passed, failed) = parse_test_output(&result.stdout);
                ExecutionResult {
                    compiled: true,
                    compile_errors: Vec::new(),
                    tests_passed: passed,
                    tests_failed: failed,
                    test_output: result.combined_output(),
                    runtime_error: None,
                    elapsed: start.elapsed(),
                    simulated: result.simulated,
                    test_failures: Vec::new(),
                }
            }
            Err(e) => ExecutionResult {
                compiled: false,
                compile_errors: vec![format!("Sandbox error executing workspace: {e}")],
                tests_passed: 0,
                tests_failed: 0,
                test_output: String::new(),
                runtime_error: None,
                elapsed: start.elapsed(),
                simulated: false,
                test_failures: Vec::new(),
            },
        }
    }

    /// Compile Rust source code and optionally run tests.
    ///
    /// Writes source to a temp file, invokes `rustc --edition 2021`,
    /// and captures errors. If `test_source` is provided, appends it
    /// and runs with `--test`.
    pub fn execute_rust(&mut self, source: &str, test_source: Option<&str>) -> ExecutionResult {
        let start = std::time::Instant::now();

        // Ensure work directory exists
        if let Err(e) = std::fs::create_dir_all(&self.work_dir) {
            return ExecutionResult {
                compiled: false,
                compile_errors: vec![format!("Failed to create work dir: {e}")],
                tests_passed: 0,
                tests_failed: 0,
                test_output: String::new(),
                runtime_error: None,
                elapsed: start.elapsed(),
                simulated: false,
                test_failures: Vec::new(),
            };
        }

        // Write source file
        let source_path = self.work_dir.join("generated.rs");
        let full_source = if let Some(tests) = test_source {
            // Strip any emitter-generated test module from the source to avoid
            // duplicate `mod tests` (E0428). The external test_source takes precedence.
            let clean_source = strip_test_module(source);
            format!("{clean_source}\n\n#[cfg(test)]\nmod tests {{\n    use super::*;\n{tests}\n}}")
        } else {
            source.to_string()
        };

        if let Err(e) = std::fs::write(&source_path, &full_source) {
            return ExecutionResult {
                compiled: false,
                compile_errors: vec![format!("Failed to write source: {e}")],
                tests_passed: 0,
                tests_failed: 0,
                test_output: String::new(),
                runtime_error: None,
                elapsed: start.elapsed(),
                simulated: false,
                test_failures: Vec::new(),
            };
        }

        // Compile
        let output_path = self.work_dir.join("generated");
        let compile_args = if test_source.is_some() {
            let mut args = vec![
                "--edition".to_string(),
                "2021".to_string(),
                "--test".to_string(),
                source_path.to_str().unwrap_or("generated.rs").to_string(),
                "-o".to_string(),
                output_path.to_str().unwrap_or("generated").to_string(),
            ];
            args.extend(rustc_linker_args());
            args
        } else {
            let mut args = vec![
                "--edition".to_string(),
                "2021".to_string(),
                source_path.to_str().unwrap_or("generated.rs").to_string(),
                "-o".to_string(),
                output_path.to_str().unwrap_or("generated").to_string(),
            ];
            args.extend(rustc_linker_args());
            args
        };
        let compile_arg_refs: Vec<&str> = compile_args.iter().map(String::as_str).collect();

        match self.sandbox.run("rustc", &compile_arg_refs) {
            Ok(result) => {
                if !result.success() {
                    let errors = parse_compile_errors(&result.stderr);
                    return ExecutionResult {
                        compiled: false,
                        compile_errors: if errors.is_empty() {
                            vec![result.stderr.clone()]
                        } else {
                            errors
                        },
                        tests_passed: 0,
                        tests_failed: 0,
                        test_output: result.stderr,
                        runtime_error: None,
                        elapsed: start.elapsed(),
                        simulated: result.simulated,
                        test_failures: Vec::new(),
                    };
                }

                // If tests, run the compiled test binary
                if test_source.is_some() {
                    // Allow the generated binary in the sandbox
                    if let Some(path_str) = output_path.to_str() {
                        self.sandbox.allow_command(path_str);
                    }
                    match self
                        .sandbox
                        .run(output_path.to_str().unwrap_or("./generated"), &[])
                    {
                        Ok(test_result) => {
                            let (passed, failed) = parse_test_output(&test_result.stdout);
                            ExecutionResult {
                                compiled: true,
                                compile_errors: Vec::new(),
                                tests_passed: passed,
                                tests_failed: failed,
                                test_output: test_result.combined_output(),
                                runtime_error: if test_result.success() {
                                    None
                                } else {
                                    Some(test_result.stderr.clone())
                                },
                                elapsed: start.elapsed(),
                                simulated: test_result.simulated,
                                test_failures: Vec::new(),
                            }
                        }
                        Err(e) => ExecutionResult {
                            compiled: true,
                            compile_errors: Vec::new(),
                            tests_passed: 0,
                            tests_failed: 0,
                            test_output: String::new(),
                            runtime_error: Some(format!("Test execution failed: {e}")),
                            elapsed: start.elapsed(),
                            simulated: false,
                            test_failures: Vec::new(),
                        },
                    }
                } else {
                    ExecutionResult {
                        compiled: true,
                        compile_errors: Vec::new(),
                        tests_passed: 0,
                        tests_failed: 0,
                        test_output: String::new(),
                        runtime_error: None,
                        elapsed: start.elapsed(),
                        simulated: result.simulated,
                        test_failures: Vec::new(),
                    }
                }
            }
            Err(SandboxError::CommandNotAllowed(_)) | Err(SandboxError::RealExecutionDisabled) => {
                // Simulation mode fallback
                ExecutionResult::simulated_success()
            }
            Err(e) => ExecutionResult {
                compiled: false,
                compile_errors: vec![format!("Sandbox error: {e}")],
                tests_passed: 0,
                tests_failed: 0,
                test_output: String::new(),
                runtime_error: None,
                elapsed: start.elapsed(),
                simulated: false,
                test_failures: Vec::new(),
            },
        }
    }

    /// Compile Rust source with `--test` and run inline tests.
    ///
    /// Unlike `execute_rust`, this does NOT append a test module wrapper —
    /// it expects the source to already contain `#[cfg(test)] mod tests { ... }`.
    /// This is used when the emitters have generated inline assertions.
    pub fn execute_rust_with_inline_tests(&mut self, source: &str) -> ExecutionResult {
        let start = std::time::Instant::now();

        if let Err(e) = std::fs::create_dir_all(&self.work_dir) {
            return ExecutionResult {
                compiled: false,
                compile_errors: vec![format!("Failed to create work dir: {e}")],
                tests_passed: 0,
                tests_failed: 0,
                test_output: String::new(),
                runtime_error: None,
                elapsed: start.elapsed(),
                simulated: false,
                test_failures: Vec::new(),
            };
        }

        let source_path = self.work_dir.join("generated_test.rs");
        if let Err(e) = std::fs::write(&source_path, source) {
            return ExecutionResult {
                compiled: false,
                compile_errors: vec![format!("Failed to write source: {e}")],
                tests_passed: 0,
                tests_failed: 0,
                test_output: String::new(),
                runtime_error: None,
                elapsed: start.elapsed(),
                simulated: false,
                test_failures: Vec::new(),
            };
        }

        let output_path = self.work_dir.join("generated_test");
        let mut compile_args = vec![
            "--edition".to_string(),
            "2021".to_string(),
            "--test".to_string(),
            source_path
                .to_str()
                .unwrap_or("generated_test.rs")
                .to_string(),
            "-o".to_string(),
            output_path.to_str().unwrap_or("generated_test").to_string(),
        ];
        compile_args.extend(rustc_linker_args());
        let compile_arg_refs: Vec<&str> = compile_args.iter().map(String::as_str).collect();

        match self.sandbox.run("rustc", &compile_arg_refs) {
            Ok(result) => {
                if !result.success() {
                    let errors = parse_compile_errors(&result.stderr);
                    return ExecutionResult {
                        compiled: false,
                        compile_errors: if errors.is_empty() {
                            vec![result.stderr.clone()]
                        } else {
                            errors
                        },
                        tests_passed: 0,
                        tests_failed: 0,
                        test_output: result.stderr,
                        runtime_error: None,
                        elapsed: start.elapsed(),
                        simulated: result.simulated,
                        test_failures: Vec::new(),
                    };
                }

                // Run the test binary
                match self
                    .sandbox
                    .run(output_path.to_str().unwrap_or("./generated_test"), &[])
                {
                    Ok(test_result) => {
                        let (passed, failed) = parse_test_output(&test_result.stdout);
                        ExecutionResult {
                            compiled: true,
                            compile_errors: Vec::new(),
                            tests_passed: passed,
                            tests_failed: failed,
                            test_output: test_result.combined_output(),
                            runtime_error: if test_result.success() {
                                None
                            } else {
                                Some(test_result.stderr.clone())
                            },
                            elapsed: start.elapsed(),
                            simulated: test_result.simulated,
                            test_failures: Vec::new(),
                        }
                    }
                    Err(e) => ExecutionResult {
                        compiled: true,
                        compile_errors: Vec::new(),
                        tests_passed: 0,
                        tests_failed: 0,
                        test_output: String::new(),
                        runtime_error: Some(format!("Test execution failed: {e}")),
                        elapsed: start.elapsed(),
                        simulated: false,
                        test_failures: Vec::new(),
                    },
                }
            }
            Err(SandboxError::CommandNotAllowed(_)) | Err(SandboxError::RealExecutionDisabled) => {
                ExecutionResult::simulated_success()
            }
            Err(e) => ExecutionResult {
                compiled: false,
                compile_errors: vec![format!("Sandbox error: {e}")],
                tests_passed: 0,
                tests_failed: 0,
                test_output: String::new(),
                runtime_error: None,
                elapsed: start.elapsed(),
                simulated: false,
                test_failures: Vec::new(),
            },
        }
    }

    /// Execute Python source code.
    pub fn execute_python(&mut self, source: &str) -> ExecutionResult {
        let start = std::time::Instant::now();

        if let Err(e) = std::fs::create_dir_all(&self.work_dir) {
            return ExecutionResult {
                compiled: false,
                compile_errors: vec![format!("Failed to create work dir: {e}")],
                tests_passed: 0,
                tests_failed: 0,
                test_output: String::new(),
                runtime_error: None,
                elapsed: start.elapsed(),
                simulated: false,
                test_failures: Vec::new(),
            };
        }

        let source_path = self.work_dir.join("generated.py");
        if let Err(e) = std::fs::write(&source_path, source) {
            return ExecutionResult {
                compiled: false,
                compile_errors: vec![format!("Failed to write source: {e}")],
                tests_passed: 0,
                tests_failed: 0,
                test_output: String::new(),
                runtime_error: None,
                elapsed: start.elapsed(),
                simulated: false,
                test_failures: Vec::new(),
            };
        }

        // Python syntax check first
        let check_arg = format!(
            "import py_compile; py_compile.compile('{}', doraise=True)",
            source_path.display()
        );
        match self.sandbox.run("python3", &["-c", &check_arg]) {
            Ok(result) => {
                if !result.success() {
                    return ExecutionResult {
                        compiled: false,
                        compile_errors: vec![result.stderr.clone()],
                        tests_passed: 0,
                        tests_failed: 0,
                        test_output: result.stderr,
                        runtime_error: None,
                        elapsed: start.elapsed(),
                        simulated: result.simulated,
                        test_failures: Vec::new(),
                    };
                }

                // Run the file
                let src_str = source_path.to_string_lossy();
                match self.sandbox.run("python3", &[&src_str]) {
                    Ok(run_result) => ExecutionResult {
                        compiled: true,
                        compile_errors: Vec::new(),
                        tests_passed: if run_result.success() { 1 } else { 0 },
                        tests_failed: if run_result.success() { 0 } else { 1 },
                        test_output: run_result.combined_output(),
                        runtime_error: if run_result.success() {
                            None
                        } else {
                            Some(run_result.stderr.clone())
                        },
                        elapsed: start.elapsed(),
                        simulated: run_result.simulated,
                        test_failures: Vec::new(),
                    },
                    Err(e) => ExecutionResult {
                        compiled: true,
                        compile_errors: Vec::new(),
                        tests_passed: 0,
                        tests_failed: 1,
                        test_output: String::new(),
                        runtime_error: Some(format!("Execution failed: {e}")),
                        elapsed: start.elapsed(),
                        simulated: false,
                        test_failures: Vec::new(),
                    },
                }
            }
            Err(SandboxError::CommandNotAllowed(_)) | Err(SandboxError::RealExecutionDisabled) => {
                ExecutionResult::simulated_success()
            }
            Err(e) => ExecutionResult {
                compiled: false,
                compile_errors: vec![format!("Sandbox error: {e}")],
                tests_passed: 0,
                tests_failed: 0,
                test_output: String::new(),
                runtime_error: None,
                elapsed: start.elapsed(),
                simulated: false,
                test_failures: Vec::new(),
            },
        }
    }

    /// Evaluate a Nix expression.
    pub fn evaluate_nix(&mut self, expr: &str) -> ExecutionResult {
        let start = std::time::Instant::now();

        match self.sandbox.nix_eval(expr) {
            Ok(result) => ExecutionResult {
                compiled: result.success(),
                compile_errors: if result.success() {
                    Vec::new()
                } else {
                    vec![result.stderr.clone()]
                },
                tests_passed: if result.success() { 1 } else { 0 },
                tests_failed: if result.success() { 0 } else { 1 },
                test_output: result.combined_output(),
                runtime_error: None,
                elapsed: start.elapsed(),
                simulated: result.simulated,
                test_failures: Vec::new(),
            },
            Err(SandboxError::CommandNotAllowed(_)) | Err(SandboxError::RealExecutionDisabled) => {
                ExecutionResult::simulated_success()
            }
            Err(e) => ExecutionResult {
                compiled: false,
                compile_errors: vec![format!("Nix eval error: {e}")],
                tests_passed: 0,
                tests_failed: 0,
                test_output: String::new(),
                runtime_error: None,
                elapsed: start.elapsed(),
                simulated: false,
                test_failures: Vec::new(),
            },
        }
    }

    /// Clean up temporary files
    pub fn cleanup(&self) {
        let _ = std::fs::remove_dir_all(&self.work_dir);
    }
}

fn fresh_work_dir() -> PathBuf {
    let seq = CODE_EXECUTOR_WORKDIR_SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("symthaea-code-exec-{}-{seq}", std::process::id()))
}

impl Drop for CodeExecutor {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// A structured compilation error with optional location info.
#[derive(Debug, Clone)]
pub struct CompileError {
    /// The full error message line.
    pub message: String,
    /// Rustc error code, if present (e.g., "E0308").
    pub code: Option<String>,
    /// Source file path from the error, if parsed.
    pub file: Option<String>,
    /// Line number in source, if parsed (1-indexed).
    pub line: Option<usize>,
    /// Column number in source, if parsed (1-indexed).
    pub column: Option<usize>,
    /// Error category for recovery strategy selection.
    pub category: ErrorCategory,
    /// Compiler-suggested replacement text (from `--message-format=json`).
    /// When available, this is a machine-applicable fix from rustc itself.
    pub suggested_replacement: Option<String>,
}

/// Category of compilation error — determines recovery strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Type mismatch (E0308, E0277) — may be fixable with conversions
    TypeMismatch,
    /// Missing import (E0412, E0433) — fixable by adding `use`
    MissingImport,
    /// Borrow checker (E0382, E0502, E0505, E0596) — may need mut/clone/ref
    BorrowError,
    /// Value moved (E0382)
    MovedValue,
    /// Lifetime error (E0106, E0621) — needs lifetime annotation
    LifetimeError,
    /// Visibility error (E0603)
    VisibilityError,
    /// Unused code (warnings treated as errors)
    UnusedCode,
    /// Missing trait impl (E0277 for Display/Debug/Clone)
    MissingImpl,
    /// Undeclared generic type parameter (E0412 for single-letter types like T, U, V)
    /// Fixable by inserting `<T>` on the enclosing item.
    UndeclaredGeneric,
    /// Unwanted `fn main()` in library crate (E0601)
    /// Fixable by stripping the main wrapper.
    UnwantedMain,
    /// Syntax error — code is malformed
    SyntaxError,
    /// Timeout — execution took too long
    Timeout,
    /// Linker error
    LinkerError,
    /// Sandbox or environment error
    SandboxError,
    /// Other/unknown error
    Other,
}

impl CompileError {
    /// Parse a structured error from a rustc error line and its context.
    fn from_rustc_output(error_line: &str, context_lines: &[&str]) -> Self {
        let code = Self::extract_error_code(error_line);
        let category = Self::categorize(&code, error_line);

        // Try to parse location from context: "--> src/file.rs:123:45"
        let (file, line, column) = context_lines
            .iter()
            .find_map(|l| Self::parse_location(l))
            .unwrap_or((None, None, None));

        CompileError {
            message: error_line.to_string(),
            code,
            file,
            line,
            column,
            category,
            suggested_replacement: None,
        }
    }

    /// Extract the unresolved type name from "cannot find type `T` in this scope".
    fn extract_unresolved_type(message: &str) -> Option<String> {
        // rustc format: "cannot find type `T` in this scope"
        if message.contains("cannot find type") {
            if let Some(start) = message.find('`') {
                if let Some(end) = message[start + 1..].find('`') {
                    return Some(message[start + 1..start + 1 + end].to_string());
                }
            }
        }
        None
    }

    /// Extract error code like "E0308" from "error[E0308]: ..."
    fn extract_error_code(line: &str) -> Option<String> {
        if let Some(start) = line.find("[E") {
            if let Some(end) = line[start..].find(']') {
                return Some(line[start + 1..start + end].to_string());
            }
        }
        None
    }

    /// Parse location from rustc's "--> file:line:col" format.
    fn parse_location(line: &str) -> Option<(Option<String>, Option<usize>, Option<usize>)> {
        let trimmed = line.trim();
        if !trimmed.starts_with("-->") {
            return None;
        }
        let loc = trimmed.trim_start_matches("-->").trim();
        let parts: Vec<&str> = loc.rsplitn(3, ':').collect();
        match parts.len() {
            3 => {
                let col = parts[0].parse().ok();
                let line = parts[1].parse().ok();
                let file = Some(parts[2].to_string());
                Some((file, line, col))
            }
            2 => {
                let line = parts[0].parse().ok();
                let file = Some(parts[1].to_string());
                Some((file, line, None))
            }
            _ => None,
        }
    }

    /// Categorize an error based on its code and message.
    fn categorize(code: &Option<String>, message: &str) -> ErrorCategory {
        if let Some(c) = code {
            match c.as_str() {
                "E0308" => ErrorCategory::TypeMismatch,
                "E0277" if message.contains("expected") => ErrorCategory::TypeMismatch,
                "E0277" => ErrorCategory::MissingImpl,
                "E0412" => {
                    // Single-letter uppercase = generic param (T, U, V, K, E, etc.)
                    let is_generic = Self::extract_unresolved_type(message)
                        .map(|t| {
                            t.len() == 1
                                && t.chars().next().map_or(false, |c| c.is_ascii_uppercase())
                        })
                        .unwrap_or(false);
                    if is_generic {
                        ErrorCategory::UndeclaredGeneric
                    } else {
                        ErrorCategory::MissingImport
                    }
                }
                "E0433" | "E0432" => ErrorCategory::MissingImport,
                "E0601" => ErrorCategory::UnwantedMain,
                "E0382" | "E0502" | "E0505" | "E0596" | "E0507" => ErrorCategory::BorrowError,
                "E0106" | "E0621" => ErrorCategory::LifetimeError,
                _ => ErrorCategory::Other,
            }
        } else {
            let lower = message.to_lowercase();
            if lower.contains("unused") {
                ErrorCategory::UnusedCode
            } else if lower.contains("expected") && lower.contains("found") {
                ErrorCategory::TypeMismatch
            } else if lower.contains("cannot find") || lower.contains("not found") {
                ErrorCategory::MissingImport
            } else if lower.contains("cannot borrow") || lower.contains("move out of") {
                ErrorCategory::BorrowError
            } else if lower.contains("lifetime") {
                ErrorCategory::LifetimeError
            } else {
                ErrorCategory::Other
            }
        }
    }
}

/// Parse rustc error output into individual error messages (flat strings).
fn parse_compile_errors(stderr: &str) -> Vec<String> {
    let lines: Vec<&str> = stderr.lines().collect();
    let mut errors = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        if lines[i].starts_with("error") {
            let mut chunk = vec![lines[i].to_string()];
            let mut j = i + 1;
            while j < lines.len()
                && !lines[j].starts_with("error")
                && !lines[j].starts_with("warning")
            {
                let line = lines[j];
                if !line.trim().is_empty() {
                    chunk.push(line.to_string());
                }
                j += 1;
            }
            errors.push(chunk.join("\n"));
            i = j;
        } else {
            i += 1;
        }
    }

    errors
}

/// Extra rustc linker flags for standalone generated-code checks.
///
/// Rustup toolchains on Nix can point their bundled `gcc-ld` shim at a garbage
/// collected Nix store path. Prefer the system C compiler and GNU ld when
/// available, while allowing callers to override both choices.
fn rustc_linker_args() -> Vec<String> {
    let linker = std::env::var("SYMTHAEA_RUSTC_LINKER").ok().or_else(|| {
        let system_cc = "/run/current-system/sw/bin/cc";
        std::path::Path::new(system_cc)
            .exists()
            .then(|| system_cc.to_string())
    });

    let mut args = Vec::new();
    if let Some(linker) = linker {
        args.push("-C".to_string());
        args.push(format!("linker={linker}"));
    }

    match std::env::var("SYMTHAEA_RUSTC_LINK_ARG") {
        Ok(link_arg) if !link_arg.trim().is_empty() => {
            args.push("-C".to_string());
            args.push(format!("link-arg={link_arg}"));
        }
        Err(_) if cfg!(target_os = "linux") => {
            args.push("-C".to_string());
            args.push("link-arg=-fuse-ld=bfd".to_string());
        }
        _ => {}
    }

    args
}

/// Parse rustc error output into structured errors with location info.
///
/// Groups error lines with their context (location, help suggestions)
/// for line-number-aware auto-fix.
pub fn parse_structured_errors(stderr: &str) -> Vec<CompileError> {
    let lines: Vec<&str> = stderr.lines().collect();
    let mut errors = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        if lines[i].starts_with("error") {
            // Collect context lines (location, help, notes) until next error or blank
            let error_line = lines[i];
            let mut context = Vec::new();
            let mut j = i + 1;
            while j < lines.len()
                && !lines[j].starts_with("error")
                && !lines[j].starts_with("warning")
                && j < i + 10
            {
                context.push(lines[j]);
                j += 1;
            }
            errors.push(CompileError::from_rustc_output(error_line, &context));
            i = j;
        } else {
            i += 1;
        }
    }
    errors
}

/// A rustc JSON diagnostic (subset of fields we use).
///
/// Produced by `cargo check --message-format=json`. Each line of stdout is a JSON
/// object; we only care about `"compiler-message"` entries whose `message.level`
/// is `"error"`.
#[derive(Debug, serde::Deserialize)]
struct RustcJsonEnvelope {
    reason: Option<String>,
    message: Option<RustcDiagnostic>,
}

/// Core diagnostic from rustc's JSON output.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RustcDiagnostic {
    /// Human-readable message text.
    pub message: String,
    /// Error code object, e.g. `{"code": "E0308", "explanation": ...}`.
    pub code: Option<RustcDiagnosticCode>,
    /// Severity level: "error", "warning", etc.
    pub level: String,
    /// Primary source spans.
    pub spans: Vec<RustcSpan>,
    /// Child diagnostics (help, note, suggestion).
    pub children: Vec<RustcDiagnostic>,
}

/// Rustc error code.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RustcDiagnosticCode {
    pub code: String,
}

/// A span in rustc's JSON output.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RustcSpan {
    pub file_name: String,
    pub line_start: usize,
    pub line_end: usize,
    pub column_start: usize,
    pub column_end: usize,
    /// If this span is a suggestion, the replacement text.
    pub suggested_replacement: Option<String>,
    pub is_primary: bool,
}

/// Parse `cargo check --message-format=json` output into structured errors.
///
/// Each line of stdout is a JSON object. We extract `"compiler-message"` entries
/// with level `"error"`, converting them to `CompileError` with full location info
/// and any compiler-suggested replacements.
pub fn parse_json_diagnostics(json_output: &str) -> Vec<CompileError> {
    let mut errors = Vec::new();

    for line in json_output.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let envelope: RustcJsonEnvelope = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(_) => continue,
        };

        if envelope.reason.as_deref() != Some("compiler-message") {
            continue;
        }

        let diag = match envelope.message {
            Some(d) if d.level == "error" => d,
            _ => continue,
        };

        let code_str = diag.code.as_ref().map(|c| c.code.clone());
        let category = CompileError::categorize(&code_str, &diag.message);

        // Find primary span for location
        let primary_span = diag.spans.iter().find(|s| s.is_primary);
        let (file, line_num, column) = match primary_span {
            Some(s) => (
                Some(s.file_name.clone()),
                Some(s.line_start),
                Some(s.column_start),
            ),
            None => (None, None, None),
        };

        // Look for suggested replacement: first in primary span, then in children
        let suggested = primary_span
            .and_then(|s| s.suggested_replacement.clone())
            .or_else(|| {
                diag.children.iter().find_map(|child| {
                    child
                        .spans
                        .iter()
                        .find_map(|s| s.suggested_replacement.clone())
                })
            });

        errors.push(CompileError {
            message: diag.message,
            code: code_str,
            file,
            line: line_num,
            column,
            category,
            suggested_replacement: suggested,
        });
    }

    errors
}

/// Attempt to auto-fix common Rust compilation errors in source code.
///
/// Strip the `#[cfg(test)] mod tests { ... }` block from generated source.
///
/// Used when external test source is provided — avoids duplicate `mod tests` (E0428).
/// Handles both `#[cfg(test)]\nmod tests {` and `mod tests {` patterns.
fn strip_test_module(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut result = Vec::new();
    let mut in_test_module = false;
    let mut brace_depth = 0i32;
    let mut skip_cfg_test = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Detect #[cfg(test)] preceding mod tests
        if trimmed == "#[cfg(test)]" {
            // Check if next line is `mod tests {`
            if i + 1 < lines.len() && lines[i + 1].trim().starts_with("mod tests") {
                skip_cfg_test = true;
                continue;
            }
        }

        if skip_cfg_test {
            skip_cfg_test = false;
            if trimmed.starts_with("mod tests") {
                in_test_module = true;
                brace_depth = trimmed.chars().filter(|&c| c == '{').count() as i32
                    - trimmed.chars().filter(|&c| c == '}').count() as i32;
                if brace_depth <= 0 {
                    in_test_module = false;
                }
                continue;
            }
        }

        if in_test_module {
            brace_depth += trimmed.chars().filter(|&c| c == '{').count() as i32;
            brace_depth -= trimmed.chars().filter(|&c| c == '}').count() as i32;
            if brace_depth <= 0 {
                in_test_module = false;
            }
            continue;
        }

        result.push(*line);
    }

    // Remove trailing empty lines
    while result.last().map_or(false, |l| l.trim().is_empty()) {
        result.pop();
    }

    result.join("\n")
}

/// Applies mechanical fixes for well-known rustc error patterns. When
/// structured errors with line numbers are available (via `try_auto_fix_structured`),
/// fixes are targeted to specific lines for higher accuracy.
///
/// Returns `Some(fixed_source)` if any fix was applied, `None` otherwise.
pub fn try_auto_fix(source: &str, errors: &[String]) -> Option<String> {
    let mut fixed = source.to_string();
    let mut any_fix = false;

    for error in errors {
        let err_lower = error.to_lowercase();

        // Missing mut: "cannot borrow `x` as mutable"
        if err_lower.contains("cannot borrow") && err_lower.contains("as mutable") {
            if let Some(var) = extract_between(error, "`", "`") {
                let var_clean = var.trim_start_matches('*');
                let pattern = format!("let {}", var_clean);
                let replacement = format!("let mut {}", var_clean);
                if fixed.contains(&pattern) {
                    fixed = fixed.replacen(&pattern, &replacement, 1);
                    any_fix = true;
                }
            }
        }

        // Unused variable
        if err_lower.contains("unused variable") {
            if let Some(var) = extract_between(error, "`", "`") {
                if !var.starts_with('_') {
                    let pattern = format!("let {}", var);
                    let replacement = format!("let _{}", var);
                    if fixed.contains(&pattern) {
                        fixed = fixed.replacen(&pattern, &replacement, 1);
                        any_fix = true;
                    }
                    let param_pattern = format!("{}: ", var);
                    let param_replacement = format!("_{}: ", var);
                    if !any_fix && fixed.contains(&param_pattern) {
                        fixed = fixed.replacen(&param_pattern, &param_replacement, 1);
                        any_fix = true;
                    }
                }
            }
        }

        // Missing #[derive(Debug)]
        if (err_lower.contains("doesn't implement") || err_lower.contains("does not implement"))
            && err_lower.contains("debug")
        {
            if let Some(type_name) = extract_between(error, "`", "`") {
                let struct_pattern = format!("struct {}", type_name);
                if let Some(pos) = fixed.find(&struct_pattern) {
                    let before = &fixed[..pos];
                    if !before.ends_with("]\n") && !before.contains("#[derive(Debug") {
                        fixed.insert_str(pos, "#[derive(Debug, Clone)]\n");
                        any_fix = true;
                    }
                }
            }
        }

        // Dead code warning treated as error
        if err_lower.contains("unused") && err_lower.contains("function") {
            if let Some(fn_name) = extract_between(error, "`", "`") {
                let fn_pattern = format!("fn {}", fn_name);
                if let Some(pos) = fixed.find(&fn_pattern) {
                    let before = &fixed[..pos];
                    if !before.ends_with("#[allow(dead_code)]\n") {
                        fixed.insert_str(pos, "#[allow(dead_code)]\n");
                        any_fix = true;
                    }
                }
            }
        }
    }

    // Check for missing common imports and prepend them
    let import_fixes: &[(&str, &str)] = &[
        ("HashMap", "use std::collections::HashMap;\n"),
        ("HashSet", "use std::collections::HashSet;\n"),
        ("BTreeMap", "use std::collections::BTreeMap;\n"),
        ("File", "use std::fs::File;\n"),
        ("Duration", "use std::time::Duration;\n"),
        ("Instant", "use std::time::Instant;\n"),
        ("io::Read", "use std::io::Read;\n"),
        ("io::Write", "use std::io::Write;\n"),
        ("BufReader", "use std::io::BufReader;\n"),
        ("fmt::Display", "use std::fmt;\n"),
        ("Ordering", "use std::cmp::Ordering;\n"),
        ("BinaryHeap", "use std::collections::BinaryHeap;\n"),
        ("VecDeque", "use std::collections::VecDeque;\n"),
        ("Path", "use std::path::Path;\n"),
        ("PathBuf", "use std::path::PathBuf;\n"),
        ("Arc", "use std::sync::Arc;\n"),
        ("Mutex", "use std::sync::Mutex;\n"),
        ("Rc", "use std::rc::Rc;\n"),
    ];

    for error in errors {
        if error.contains("cannot find") || error.contains("not found") {
            for (type_name, import_stmt) in import_fixes {
                if error.contains(type_name) && !fixed.contains(import_stmt.trim()) {
                    fixed = format!("{}{}", import_stmt, fixed);
                    any_fix = true;
                }
            }
        }
    }

    if any_fix { Some(fixed) } else { None }
}

/// Enhanced auto-fix using structured errors with line numbers.
///
/// This is the line-number-aware version of `try_auto_fix`. When rustc provides
/// location info (file:line:col), fixes are applied directly to the error line
/// rather than using blind pattern matching. This enables fixes that the basic
/// version can't do safely (type conversions, clone insertion, lifetime annotations).
///
/// Returns `Some(fixed_source)` if any fix was applied, `None` otherwise.
pub fn try_auto_fix_structured(source: &str, errors: &[CompileError]) -> Option<String> {
    let mut lines: Vec<String> = source.lines().map(|l| l.to_string()).collect();
    let mut any_fix = false;
    // Track line offsets from insertions (derive attributes, imports)
    let mut line_offset: i64 = 0;

    for error in errors {
        let target_line = error.line.map(|l| {
            let adjusted = l as i64 + line_offset;
            if adjusted < 1 {
                0
            } else {
                (adjusted as usize).saturating_sub(1)
            }
        });

        match error.category {
            ErrorCategory::TypeMismatch => {
                if let Some(idx) = target_line {
                    if idx < lines.len() {
                        let line = lines[idx].clone();
                        // "expected String, found &str" → add .to_string()
                        if error.message.contains("expected")
                            && error.message.contains("String")
                            && error.message.contains("&str")
                        {
                            if let Some(col) = error.column {
                                let col_idx = col.saturating_sub(1);
                                let before = &line[..col_idx.min(line.len())];
                                if before.trim_end().ends_with('"')
                                    || before.trim_end().ends_with(')')
                                {
                                    let trimmed = line.trim_end();
                                    if !trimmed.ends_with(".to_string()") {
                                        let new_line = if let Some(pos) = trimmed.rfind(';') {
                                            format!(
                                                "{}.to_string(){}",
                                                &trimmed[..pos],
                                                &trimmed[pos..]
                                            )
                                        } else if let Some(pos) = trimmed.rfind(',') {
                                            format!(
                                                "{}.to_string(){}",
                                                &trimmed[..pos],
                                                &trimmed[pos..]
                                            )
                                        } else {
                                            format!("{}.to_string()", trimmed)
                                        };
                                        lines[idx] = new_line;
                                        any_fix = true;
                                    }
                                }
                            }
                        }
                        // "expected &str, found String" → add .as_str() or &
                        if error.message.contains("expected")
                            && error.message.contains("&str")
                            && error.message.contains("found")
                            && error.message.contains("String")
                            && !error.message.contains("expected `String`")
                        {
                            let trimmed = line.trim_end();
                            if !trimmed.ends_with(".as_str()") {
                                if let Some(pos) = trimmed.rfind(';') {
                                    lines[idx] =
                                        format!("{}.as_str(){}", &trimmed[..pos], &trimmed[pos..]);
                                    any_fix = true;
                                }
                            }
                        }
                    }
                }
            }

            ErrorCategory::BorrowError => {
                // "cannot move out of" → add .clone() at the error location
                if error.message.contains("cannot move out of") {
                    if let Some(var) = extract_between(&error.message, "`", "`") {
                        let var_clean = var.trim_start_matches('*');
                        if let Some(idx) = target_line {
                            if idx < lines.len() {
                                let line = &lines[idx];
                                // Insert .clone() after the variable reference
                                let pattern = var_clean;
                                if let Some(pos) = line.find(pattern) {
                                    let after_var = pos + pattern.len();
                                    // Only add .clone() if not already there and variable is standalone
                                    let after = &line[after_var..];
                                    if !after.starts_with(".clone()") {
                                        let next_char = after.chars().next();
                                        if matches!(next_char, Some(')' | ',' | ';' | ' ' | '.')) {
                                            let new_line = format!(
                                                "{}{}.clone(){}",
                                                &line[..after_var],
                                                "",
                                                &line[after_var..]
                                            );
                                            lines[idx] = new_line;
                                            any_fix = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // "cannot borrow as mutable" — handled by basic try_auto_fix
            }

            ErrorCategory::LifetimeError => {
                // "missing lifetime specifier" on fn return type → add <'a>
                if error.message.contains("missing lifetime specifier") {
                    if let Some(idx) = target_line {
                        if idx < lines.len() {
                            let line = &lines[idx];
                            // Pattern: "fn foo(s: &str) -> &str"
                            // Fix: "fn foo<'a>(s: &'a str) -> &'a str"
                            if line.contains("fn ") && line.contains("-> &") {
                                let mut new_line = line.clone();
                                // Add lifetime parameter to function
                                if !new_line.contains("<'") {
                                    if let Some(paren) = new_line.find('(') {
                                        new_line.insert_str(paren, "<'a>");
                                    }
                                }
                                // Add 'a to all bare &str / & references in return type
                                if let Some(arrow) = new_line.find("-> &") {
                                    let rest = &new_line[arrow..];
                                    if !rest.contains("-> &'") {
                                        new_line = new_line.replacen("-> &", "-> &'a ", 1);
                                    }
                                }
                                // Add 'a to parameter references
                                // Simple case: &str → &'a str, &T → &'a T
                                let params_start = new_line.find('(').unwrap_or(0);
                                let params_end = new_line.find(')').unwrap_or(new_line.len());
                                if params_start < params_end {
                                    let params = new_line[params_start..=params_end].to_string();
                                    let fixed_params = params.replace(": &", ": &'a ");
                                    if fixed_params != params {
                                        new_line = format!(
                                            "{}{}{}",
                                            &new_line[..params_start],
                                            fixed_params,
                                            &new_line[params_end + 1..]
                                        );
                                    }
                                }
                                if new_line != *line {
                                    lines[idx] = new_line;
                                    any_fix = true;
                                }
                            }
                        }
                    }
                }
            }

            ErrorCategory::MissingImpl => {
                // "doesn't implement Clone" → add #[derive(Clone)] above struct
                let trait_name = if error.message.contains("Clone") {
                    Some("Clone")
                } else if error.message.contains("Default") {
                    Some("Default")
                } else if error.message.contains("PartialEq") {
                    Some("PartialEq")
                } else {
                    None
                };

                if let Some(trait_name) = trait_name {
                    if let Some(type_name) = extract_between(&error.message, "`", "`") {
                        let struct_pattern = format!("struct {}", type_name);
                        let insert_idx = lines.iter().enumerate().find_map(|(i, line)| {
                            if line.contains(&struct_pattern) {
                                Some(i)
                            } else {
                                None
                            }
                        });
                        if let Some(i) = insert_idx {
                            let derive = format!("#[derive({})]", trait_name);
                            if i == 0 || !lines[i - 1].contains(&derive) {
                                lines.insert(i, derive);
                                line_offset += 1;
                                any_fix = true;
                            }
                        }
                    }
                }
            }

            // Other categories handled by basic try_auto_fix
            _ => {}
        }
    }

    if any_fix {
        Some(lines.join("\n"))
    } else {
        None
    }
}

/// Extract text between two delimiter strings (first occurrence).
fn extract_between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let start_idx = text.find(start)? + start.len();
    let remaining = &text[start_idx..];
    let end_idx = remaining.find(end)?;
    Some(&remaining[..end_idx])
}

/// Parse Rust test runner output for pass/fail counts
fn parse_test_output(stdout: &str) -> (usize, usize) {
    // Look for: "test result: ok. N passed; M failed; ..."
    for line in stdout.lines().rev() {
        if line.starts_with("test result:") {
            let passed = line
                .split_whitespace()
                .zip(line.split_whitespace().skip(1))
                .find(|(_, next)| *next == "passed;")
                .and_then(|(num, _)| num.parse::<usize>().ok())
                .unwrap_or(0);
            let failed = line
                .split_whitespace()
                .zip(line.split_whitespace().skip(1))
                .find(|(_, next)| *next == "failed;")
                .and_then(|(num, _)| num.parse::<usize>().ok())
                .unwrap_or(0);
            return (passed, failed);
        }
    }
    (0, 0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Test Failure Parsing — Semantic extraction from test output
// ═══════════════════════════════════════════════════════════════════════════════

/// Parse semantic test failure details from rustc/cargo test output.
///
/// Extracts test names, assertion text, and expected/actual values from
/// the standard Rust test runner format.
///
/// # Recognized patterns
///
/// - `---- test_name stdout ----` → test name
/// - `assertion \`left == right\` failed` → assertion type
/// - `left: VALUE` / `right: VALUE` → expected/actual
/// - `panicked at 'MESSAGE'` → raw failure message
/// - `assert_eq!(A, B)` → assertion text
pub fn parse_test_failure_details(test_output: &str) -> Vec<TestFailure> {
    let mut failures: Vec<TestFailure> = Vec::new();
    let lines: Vec<&str> = test_output.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        // Detect test name from "---- test_name stdout ----"
        if line.starts_with("---- ") && line.ends_with(" stdout ----") {
            let test_name = line
                .strip_prefix("---- ")
                .and_then(|s| s.strip_suffix(" stdout ----"))
                .unwrap_or("unknown")
                .to_string();

            // Scan ahead for failure details
            let mut message = String::new();
            let mut assertion = None;
            let mut expected = None;
            let mut actual = None;
            let mut j = i + 1;

            while j < lines.len() {
                let detail = lines[j].trim();

                // Stop at next test section or "failures:" summary
                if detail.starts_with("---- ") || detail == "failures:" {
                    break;
                }

                // Extract "panicked at" message
                if let Some(pos) = detail.find("panicked at") {
                    let msg_start = pos + "panicked at".len();
                    let msg = detail[msg_start..]
                        .trim()
                        .trim_matches('\'')
                        .trim_matches('"');
                    message = msg.to_string();
                }

                // Extract assertion text
                if detail.contains("assert_eq!")
                    || detail.contains("assert_ne!")
                    || detail.contains("assert!")
                {
                    assertion = Some(detail.to_string());
                }

                // Extract left/right values from Rust assertion output.
                // Values may have trailing artifacts from the panic message:
                // - `5', src/lib.rs:5:5` — strip `', <location>` suffix
                // - `"world"'` — strip trailing single quote from `panicked at '...'` wrapper
                let strip_trailing_artifacts = |s: &str| -> String {
                    let mut trimmed = s.trim().to_string();
                    // Strip `', <location>` suffix (e.g., `5', src/lib.rs:5:5`)
                    if let Some(pos) = trimmed.rfind("', ") {
                        let candidate = &trimmed[pos + 3..];
                        if candidate.contains(':') || candidate.contains('/') {
                            trimmed = trimmed[..pos].to_string();
                        }
                    }
                    // Strip trailing `'` from `panicked at '...'` wrapper
                    if trimmed.ends_with('\'') && !trimmed.starts_with('\'') {
                        trimmed.pop();
                    }
                    trimmed
                };
                if detail.starts_with("left:") || detail.starts_with("left =") {
                    let val = detail
                        .split_once(':')
                        .map(|x| x.1)
                        .or_else(|| detail.split_once('=').map(|x| x.1))
                        .map(|s| strip_trailing_artifacts(s));
                    actual = val;
                }
                if detail.starts_with("right:") || detail.starts_with("right =") {
                    let val = detail
                        .split_once(':')
                        .map(|x| x.1)
                        .or_else(|| detail.split_once('=').map(|x| x.1))
                        .map(|s| strip_trailing_artifacts(s));
                    expected = val;
                }

                // Detect "assertion `left == right` failed"
                if detail.contains("left == right") && detail.contains("failed") {
                    if message.is_empty() {
                        message = "assertion `left == right` failed".to_string();
                    }
                }

                j += 1;
            }

            if !message.is_empty() || assertion.is_some() || expected.is_some() {
                let detail_entry = TestFailure {
                    test_name: test_name.clone(),
                    assertion,
                    expected,
                    actual,
                    message: if message.is_empty() {
                        "test failed".to_string()
                    } else {
                        message
                    },
                };
                // Replace any existing skeleton entry (from "test ... FAILED" line)
                // with the detailed version from the stdout section
                if let Some(pos) = failures.iter().position(|f| f.test_name == test_name) {
                    failures[pos] = detail_entry;
                } else {
                    failures.push(detail_entry);
                }
            }

            i = j;
            continue;
        }

        // Also detect "test result: FAILED" summary for test names
        // Format: "test tests::test_name ... FAILED"
        if line.starts_with("test ") && line.ends_with("... FAILED") {
            let test_name = line
                .strip_prefix("test ")
                .and_then(|s| s.strip_suffix("... FAILED"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();

            // Only add if not already captured from stdout section
            if !test_name.is_empty() && !failures.iter().any(|f| f.test_name == test_name) {
                failures.push(TestFailure {
                    test_name,
                    assertion: None,
                    expected: None,
                    actual: None,
                    message: "test failed (details not captured)".to_string(),
                });
            }
        }

        i += 1;
    }

    failures
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_result_surprise_compilation_failure() {
        let result = ExecutionResult {
            compiled: false,
            compile_errors: vec!["error[E0308]: mismatched types".into()],
            tests_passed: 0,
            tests_failed: 0,
            test_output: String::new(),
            runtime_error: None,
            elapsed: Duration::from_millis(100),
            simulated: false,
            test_failures: Vec::new(),
        };
        let surprise = result.to_surprise();
        assert!(
            surprise > 0.8,
            "Compile failure should have high surprise: {surprise}"
        );
        assert!(surprise <= 1.0);
    }

    #[test]
    fn test_execution_result_surprise_test_failure() {
        let result = ExecutionResult {
            compiled: true,
            compile_errors: Vec::new(),
            tests_passed: 3,
            tests_failed: 1,
            test_output: String::new(),
            runtime_error: None,
            elapsed: Duration::from_millis(100),
            simulated: false,
            test_failures: Vec::new(),
        };
        let surprise = result.to_surprise();
        assert!(
            surprise > 0.3 && surprise < 0.8,
            "Test failure moderate surprise: {surprise}"
        );
    }

    #[test]
    fn test_execution_result_surprise_success() {
        let result = ExecutionResult {
            compiled: true,
            compile_errors: Vec::new(),
            tests_passed: 5,
            tests_failed: 0,
            test_output: String::new(),
            runtime_error: None,
            elapsed: Duration::from_millis(100),
            simulated: false,
            test_failures: Vec::new(),
        };
        assert_eq!(result.to_surprise(), 0.0);
    }

    #[test]
    fn test_execution_result_is_success() {
        let success = ExecutionResult {
            compiled: true,
            compile_errors: Vec::new(),
            tests_passed: 1,
            tests_failed: 0,
            test_output: String::new(),
            runtime_error: None,
            elapsed: Duration::from_millis(10),
            simulated: false,
            test_failures: Vec::new(),
        };
        assert!(success.is_success());

        let simulated = ExecutionResult {
            compiled: true,
            compile_errors: Vec::new(),
            tests_passed: 1,
            tests_failed: 0,
            test_output: "[Simulated] Compilation successful".to_string(),
            runtime_error: None,
            elapsed: Duration::from_millis(10),
            simulated: true,
            test_failures: Vec::new(),
        };
        assert!(!simulated.is_success());

        let failure = ExecutionResult {
            compiled: false,
            compile_errors: vec!["error".into()],
            tests_passed: 0,
            tests_failed: 0,
            test_output: String::new(),
            runtime_error: None,
            elapsed: Duration::from_millis(10),
            simulated: false,
            test_failures: Vec::new(),
        };
        assert!(!failure.is_success());
    }

    #[test]
    fn test_parse_compile_errors() {
        let stderr = "warning: unused variable `x`\nerror[E0308]: mismatched types\nerror: aborting due to previous error";
        let errors = parse_compile_errors(stderr);
        assert_eq!(errors.len(), 2);
        assert!(errors[0].contains("E0308"));
    }

    #[test]
    fn test_parse_compile_errors_preserves_linker_context() {
        let stderr = "error: linking with `cc` failed: exit status: 1\n  = note: /nix/store/.../bin/ld: cannot find crt1.o: No such file or directory\n  = note: collect2: error: ld returned 1 exit status\n\nerror: aborting due to 1 previous error";
        let errors = parse_compile_errors(stderr);
        assert_eq!(errors.len(), 2);
        assert!(errors[0].contains("linking with `cc` failed"));
        assert!(errors[0].contains("cannot find crt1.o"));
        assert!(errors[0].contains("ld returned 1 exit status"));
    }

    #[test]
    fn test_parse_test_output() {
        let stdout = "running 3 tests\ntest foo ... ok\ntest bar ... FAILED\ntest baz ... ok\n\ntest result: FAILED. 2 passed; 1 failed; 0 ignored;";
        let (passed, failed) = parse_test_output(stdout);
        assert_eq!(passed, 2);
        assert_eq!(failed, 1);
    }

    #[test]
    fn test_parse_test_output_all_pass() {
        let stdout = "running 3 tests\ntest foo ... ok\ntest bar ... ok\ntest baz ... ok\n\ntest result: ok. 3 passed; 0 failed;";
        let (passed, failed) = parse_test_output(stdout);
        assert_eq!(passed, 3);
        assert_eq!(failed, 0);
    }

    #[test]
    fn test_executor_runs_valid_rust() {
        let mut executor = CodeExecutor::new();
        let result = executor.execute_rust("fn main() {}", None);
        // Valid Rust compiles (real or simulated depending on sandbox config)
        assert!(result.compiled);
    }

    #[test]
    fn test_nix_evaluation() {
        let mut executor = CodeExecutor::new();
        let result = executor.evaluate_nix("1 + 1");
        // Nix eval succeeds (real or simulated depending on env)
        assert!(result.compiled);
    }

    #[test]
    fn test_auto_fix_missing_derive_debug() {
        let source = "struct Worker {\n    id: usize,\n}\nfn main() { let w = Worker { id: 1 }; println!(\"{:?}\", w); }";
        let errors = vec!["`Worker` doesn't implement `Debug`".to_string()];
        let fixed = try_auto_fix(source, &errors);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("#[derive(Debug, Clone)]"));
        assert!(fixed.contains("struct Worker"));
    }

    #[test]
    fn test_auto_fix_missing_mut() {
        let source = "fn main() { let v = vec![1, 2]; v.push(3); }";
        let errors = vec!["cannot borrow `v` as mutable".to_string()];
        let fixed = try_auto_fix(source, &errors);
        assert!(fixed.is_some());
        assert!(fixed.unwrap().contains("let mut v"));
    }

    #[test]
    fn test_auto_fix_missing_import() {
        let source = "fn main() { let m: HashMap<String, i32> = HashMap::new(); }";
        let errors = vec!["cannot find type `HashMap` in this scope".to_string()];
        let fixed = try_auto_fix(source, &errors);
        assert!(fixed.is_some());
        assert!(fixed.unwrap().contains("use std::collections::HashMap;"));
    }

    #[test]
    fn test_auto_fix_unused_function() {
        let source = "fn helper() -> i32 { 42 }\nfn main() {}";
        let errors = vec!["unused function: `helper`".to_string()];
        let fixed = try_auto_fix(source, &errors);
        assert!(fixed.is_some());
        assert!(fixed.unwrap().contains("#[allow(dead_code)]"));
    }

    #[test]
    fn test_auto_fix_no_errors_returns_none() {
        let source = "fn main() {}";
        let errors: Vec<String> = vec![];
        assert!(try_auto_fix(source, &errors).is_none());
    }

    #[test]
    fn test_auto_fix_unknown_error_returns_none() {
        let source = "fn main() {}";
        let errors = vec!["some unknown error we can't fix".to_string()];
        assert!(try_auto_fix(source, &errors).is_none());
    }

    // ── Phase D: Structured error parsing tests ─────────────────────────

    #[test]
    fn test_parse_structured_errors_with_location() {
        let stderr = "error[E0308]: mismatched types\n  --> generated.rs:5:12\n  |\n5 |     let x: String = \"hello\";\n  |            ^^^^^^   ------- expected due to this value\n  = note: expected struct `String`\nerror: aborting due to previous error";
        let errors = parse_structured_errors(stderr);
        assert_eq!(errors.len(), 2); // E0308 + "aborting"
        assert_eq!(errors[0].code, Some("E0308".to_string()));
        assert_eq!(errors[0].file, Some("generated.rs".to_string()));
        assert_eq!(errors[0].line, Some(5));
        assert_eq!(errors[0].column, Some(12));
        assert_eq!(errors[0].category, ErrorCategory::TypeMismatch);
    }

    #[test]
    fn test_parse_structured_errors_no_location() {
        let stderr = "error: aborting due to previous error";
        let errors = parse_structured_errors(stderr);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].line.is_none());
        assert_eq!(errors[0].category, ErrorCategory::Other);
    }

    #[test]
    fn test_error_category_classification() {
        assert_eq!(
            CompileError::categorize(&Some("E0308".into()), "expected `String`, found `&str`"),
            ErrorCategory::TypeMismatch
        );
        assert_eq!(
            CompileError::categorize(&Some("E0412".into()), "cannot find type"),
            ErrorCategory::MissingImport
        );
        assert_eq!(
            CompileError::categorize(&Some("E0596".into()), "cannot borrow"),
            ErrorCategory::BorrowError
        );
        assert_eq!(
            CompileError::categorize(&Some("E0106".into()), "missing lifetime"),
            ErrorCategory::LifetimeError
        );
        assert_eq!(
            CompileError::categorize(&None, "unused variable `x`"),
            ErrorCategory::UnusedCode
        );
    }

    #[test]
    fn test_parse_location_formats() {
        // Standard: --> file:line:col
        let (f, l, c) = CompileError::parse_location("  --> src/main.rs:42:8").unwrap();
        assert_eq!(f, Some("src/main.rs".to_string()));
        assert_eq!(l, Some(42));
        assert_eq!(c, Some(8));

        // No column
        let (f, l, c) = CompileError::parse_location("  --> src/lib.rs:10").unwrap();
        assert_eq!(f, Some("src/lib.rs".to_string()));
        assert_eq!(l, Some(10));
        assert_eq!(c, None);

        // Not a location line
        assert!(CompileError::parse_location("  = note: something").is_none());
    }

    #[test]
    fn test_structured_auto_fix_clone_insertion() {
        let source =
            "fn main() {\n    let s = String::from(\"hello\");\n    let a = s;\n    let b = s;\n}";
        let errors = vec![CompileError {
            message: "cannot move out of `s` because it is borrowed".into(),
            code: Some("E0382".into()),
            file: Some("generated.rs".into()),
            line: Some(4), // "let b = s;" is line 4
            column: Some(13),
            category: ErrorCategory::BorrowError,
            suggested_replacement: None,
        }];
        let fixed = try_auto_fix_structured(source, &errors);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(fixed.contains("s.clone()"), "Should add .clone(): {fixed}");
    }

    #[test]
    fn test_structured_auto_fix_lifetime() {
        let source = "fn first_word(s: &str) -> &str {\n    &s[..1]\n}";
        let errors = vec![CompileError {
            message: "missing lifetime specifier".into(),
            code: Some("E0106".into()),
            file: Some("generated.rs".into()),
            line: Some(1),
            column: Some(27),
            category: ErrorCategory::LifetimeError,
            suggested_replacement: None,
        }];
        let fixed = try_auto_fix_structured(source, &errors);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(
            fixed.contains("<'a>"),
            "Should add lifetime parameter: {fixed}"
        );
        assert!(
            fixed.contains("-> &'a"),
            "Should annotate return type: {fixed}"
        );
    }

    #[test]
    fn test_structured_auto_fix_missing_derive() {
        let source = "struct Point { x: f64, y: f64 }\nfn main() { let p = Point { x: 1.0, y: 2.0 }; let q = p.clone(); }";
        let errors = vec![CompileError {
            message: "`Point` doesn't implement `Clone`".into(),
            code: Some("E0277".into()),
            file: Some("generated.rs".into()),
            line: Some(2),
            column: Some(55),
            category: ErrorCategory::MissingImpl,
            suggested_replacement: None,
        }];
        let fixed = try_auto_fix_structured(source, &errors);
        assert!(fixed.is_some());
        let fixed = fixed.unwrap();
        assert!(
            fixed.contains("#[derive(Clone)]"),
            "Should add derive(Clone): {fixed}"
        );
    }

    #[test]
    fn test_auto_fix_missing_import_path_types() {
        let source = "fn main() { let p = PathBuf::from(\".\"); }";
        let errors = vec!["cannot find type `PathBuf` in this scope".to_string()];
        let fixed = try_auto_fix(source, &errors);
        assert!(fixed.is_some());
        assert!(fixed.unwrap().contains("use std::path::PathBuf;"));
    }

    #[test]
    fn test_auto_fix_missing_import_sync_types() {
        let source = "fn main() { let a = Arc::new(42); }";
        let errors = vec!["cannot find type `Arc` in this scope".to_string()];
        let fixed = try_auto_fix(source, &errors);
        assert!(fixed.is_some());
        assert!(fixed.unwrap().contains("use std::sync::Arc;"));
    }

    #[test]
    fn test_structured_auto_fix_no_errors() {
        let source = "fn main() {}";
        let errors: Vec<CompileError> = vec![];
        assert!(try_auto_fix_structured(source, &errors).is_none());
    }

    #[test]
    fn test_parse_json_diagnostics_basic() {
        let json = r#"{"reason":"compiler-message","package_id":"test","manifest_path":"","target":{"kind":["lib"],"crate_types":["lib"],"name":"test","src_path":"","edition":"2021","doctest":true,"test":true,"doc":true},"message":{"rendered":"","children":[],"code":{"code":"E0308","explanation":null},"level":"error","message":"mismatched types","spans":[{"byte_end":100,"byte_start":90,"column_end":15,"column_start":5,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":null,"line_end":10,"line_start":10,"suggested_replacement":"x as i64","suggestion_applicability":"MachineApplicable","text":[]}]}}"#;
        let errors = parse_json_diagnostics(json);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].code.as_deref(), Some("E0308"));
        assert_eq!(errors[0].file.as_deref(), Some("src/lib.rs"));
        assert_eq!(errors[0].line, Some(10));
        assert_eq!(errors[0].suggested_replacement.as_deref(), Some("x as i64"));
    }

    #[test]
    fn test_parse_json_diagnostics_skips_warnings() {
        let json = r#"{"reason":"compiler-message","message":{"rendered":"","children":[],"code":null,"level":"warning","message":"unused variable","spans":[]}}"#;
        let errors = parse_json_diagnostics(json);
        assert!(errors.is_empty(), "Should skip warnings");
    }

    #[test]
    fn test_parse_json_diagnostics_skips_non_compiler() {
        let json = r#"{"reason":"build-script-executed","package_id":"test","linked_libs":[],"linked_paths":[],"cfgs":[],"env":[],"out_dir":""}"#;
        let errors = parse_json_diagnostics(json);
        assert!(errors.is_empty(), "Should skip non-compiler-message");
    }

    #[test]
    fn test_parse_json_diagnostics_multiline() {
        let json = concat!(
            r#"{"reason":"build-script-executed","package_id":"x","linked_libs":[],"linked_paths":[],"cfgs":[],"env":[],"out_dir":""}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"rendered":"","children":[],"code":{"code":"E0433","explanation":null},"level":"error","message":"failed to resolve: use of undeclared crate or module","spans":[{"byte_end":10,"byte_start":0,"column_end":10,"column_start":1,"expansion":null,"file_name":"src/main.rs","is_primary":true,"label":null,"line_end":1,"line_start":1,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#,
            "\n",
            r#"{"reason":"compiler-message","message":{"rendered":"","children":[],"code":{"code":"E0412","explanation":null},"level":"error","message":"cannot find type","spans":[{"byte_end":50,"byte_start":40,"column_end":20,"column_start":10,"expansion":null,"file_name":"src/main.rs","is_primary":true,"label":null,"line_end":3,"line_start":3,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#,
        );
        let errors = parse_json_diagnostics(json);
        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].code.as_deref(), Some("E0433"));
        assert_eq!(errors[1].code.as_deref(), Some("E0412"));
    }

    #[test]
    fn test_categorize_e0412_generic_t() {
        let cat = CompileError::categorize(
            &Some("E0412".to_string()),
            "cannot find type `T` in this scope",
        );
        assert_eq!(cat, ErrorCategory::UndeclaredGeneric);
    }

    #[test]
    fn test_categorize_e0412_named_type() {
        let cat = CompileError::categorize(
            &Some("E0412".to_string()),
            "cannot find type `HashMap` in this scope",
        );
        assert_eq!(cat, ErrorCategory::MissingImport);
    }

    #[test]
    fn test_categorize_e0601() {
        let cat = CompileError::categorize(
            &Some("E0601".to_string()),
            "`main` function not found in crate `mylib`",
        );
        assert_eq!(cat, ErrorCategory::UnwantedMain);
    }

    #[test]
    fn test_extract_unresolved_type() {
        assert_eq!(
            CompileError::extract_unresolved_type("cannot find type `T` in this scope"),
            Some("T".to_string())
        );
        assert_eq!(
            CompileError::extract_unresolved_type("cannot find type `HashMap` in this scope"),
            Some("HashMap".to_string())
        );
        assert_eq!(
            CompileError::extract_unresolved_type("some other error message"),
            None
        );
    }

    #[test]
    fn test_parse_json_diagnostics_child_suggestion() {
        // Suggestion in children (common for "help: consider importing" messages)
        let json = r#"{"reason":"compiler-message","message":{"rendered":"","children":[{"rendered":"","children":[],"code":null,"level":"help","message":"consider importing","spans":[{"byte_end":0,"byte_start":0,"column_end":1,"column_start":1,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":null,"line_end":1,"line_start":1,"suggested_replacement":"use std::collections::HashMap;\n","suggestion_applicability":"MaybeIncorrect","text":[]}]}],"code":{"code":"E0412","explanation":null},"level":"error","message":"cannot find type `HashMap`","spans":[{"byte_end":30,"byte_start":20,"column_end":15,"column_start":5,"expansion":null,"file_name":"src/lib.rs","is_primary":true,"label":null,"line_end":5,"line_start":5,"suggested_replacement":null,"suggestion_applicability":null,"text":[]}]}}"#;
        let errors = parse_json_diagnostics(json);
        assert_eq!(errors.len(), 1);
        assert_eq!(
            errors[0].suggested_replacement.as_deref(),
            Some("use std::collections::HashMap;\n")
        );
    }

    // =========================================================================
    // Test Failure Parsing Tests
    // =========================================================================

    #[test]
    fn test_parse_test_failures_basic() {
        let output = r#"
running 2 tests
test tests::test_add ... FAILED
test tests::test_sub ... ok

---- tests::test_add stdout ----
thread 'tests::test_add' panicked at 'assertion `left == right` failed
  left: 4
 right: 5', src/lib.rs:5:5

failures:
    tests::test_add
"#;
        let failures = parse_test_failure_details(output);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].test_name, "tests::test_add");
        assert_eq!(failures[0].actual.as_deref(), Some("4"));
        assert_eq!(failures[0].expected.as_deref(), Some("5"));
        assert!(failures[0].message.contains("left == right"));
    }

    #[test]
    fn test_parse_test_failures_multiple() {
        let output = r#"
---- tests::test_a stdout ----
thread 'tests::test_a' panicked at 'assertion failed: x > 0'
---- tests::test_b stdout ----
thread 'tests::test_b' panicked at 'assertion `left == right` failed
  left: "hello"
 right: "world"'
"#;
        let failures = parse_test_failure_details(output);
        assert_eq!(failures.len(), 2);
        assert_eq!(failures[0].test_name, "tests::test_a");
        assert!(failures[0].message.contains("assertion failed"));
        assert_eq!(failures[1].test_name, "tests::test_b");
        assert_eq!(failures[1].actual.as_deref(), Some("\"hello\""));
        assert_eq!(failures[1].expected.as_deref(), Some("\"world\""));
    }

    #[test]
    fn test_parse_test_failures_empty() {
        let output = "running 3 tests\ntest a ... ok\ntest b ... ok\ntest c ... ok\n";
        let failures = parse_test_failure_details(output);
        assert!(failures.is_empty());
    }

    #[test]
    fn test_parse_test_failures_summary_only() {
        let output = "test tests::test_foo ... FAILED\ntest tests::test_bar ... FAILED\n";
        let failures = parse_test_failure_details(output);
        assert_eq!(failures.len(), 2);
        assert_eq!(failures[0].test_name, "tests::test_foo");
        assert_eq!(failures[1].test_name, "tests::test_bar");
    }

    #[test]
    fn test_failure_constraints() {
        let result = ExecutionResult {
            compiled: true,
            compile_errors: vec![],
            tests_passed: 1,
            tests_failed: 1,
            test_output: String::new(),
            runtime_error: None,
            elapsed: Duration::from_millis(100),
            simulated: false,
            test_failures: vec![TestFailure {
                test_name: "test_add".to_string(),
                assertion: Some("assert_eq!(add(2,3), 5)".to_string()),
                expected: Some("5".to_string()),
                actual: Some("4".to_string()),
                message: "assertion failed".to_string(),
            }],
        };
        let constraints = result.failure_constraints();
        assert_eq!(constraints.len(), 1);
        assert!(constraints[0].contains("expected 5 but got 4"));
    }
}
