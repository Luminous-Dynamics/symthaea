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
    /// Path to the compiled binary artifact (if any).
    pub binary_path: Option<PathBuf>,
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
    pub fn parse_test_failures(&mut self) {
        self.test_failures = parse_test_failure_details(&self.test_output);
    }

    /// Get formatted constraint strings from test failures.
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

    /// Create a result for simulation mode (sandbox refused real execution).
    ///
    /// `simulated` MUST be `true` here: nothing was compiled or run, so this
    /// result must never count as real verification. Before 2026-07-06 this
    /// constructor set `compiled: true, simulated: false`, which let
    /// sandbox-rejected executions masquerade as genuine compile passes
    /// downstream (Phase 5.5 episodic-memory storage, `is_success()`,
    /// verified-generation gating).
    fn simulated_success() -> Self {
        Self {
            compiled: true,
            compile_errors: Vec::new(),
            tests_passed: 0,
            tests_failed: 0,
            test_output: "[Simulated] Compilation not actually run (sandbox disallowed execution)"
                .to_string(),
            runtime_error: None,
            elapsed: Duration::from_millis(0),
            simulated: true,
            binary_path: None,
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

    pub fn supports_real_execution(&self) -> bool {
        !self.sandbox.is_simulation_only() && self.sandbox.is_real_execution_enabled()
    }

    pub fn work_dir(&self) -> &PathBuf {
        &self.work_dir
    }

    pub fn apply_patch(
        &mut self,
        repo_path: &std::path::Path,
        patch_content: &str,
    ) -> Result<(), String> {
        let patch_file = self.work_dir.join("fix.patch");
        std::fs::write(&patch_file, patch_content).map_err(|e| e.to_string())?;

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
                        compiled: failed > 0 || errors.is_empty(),
                        compile_errors: errors.clone(),
                        tests_passed: passed,
                        tests_failed: failed,
                        test_output: result.combined_output(),
                        runtime_error: None,
                        elapsed: start.elapsed(),
                        simulated: result.simulated,
                        binary_path: None,
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
                    binary_path: None,
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
                binary_path: None,
                test_failures: Vec::new(),
            },
        }
    }

    pub fn execute_rust(&mut self, source: &str, test_source: Option<&str>) -> ExecutionResult {
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
                binary_path: None,
                test_failures: Vec::new(),
            };
        }

        let source_path = self.work_dir.join("generated.rs");
        let full_source = if let Some(tests) = test_source {
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
                binary_path: None,
                test_failures: Vec::new(),
            };
        }

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
                        binary_path: None,
                        test_failures: Vec::new(),
                    };
                }

                if test_source.is_some() {
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
                                binary_path: Some(output_path),
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
                            binary_path: Some(output_path),
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
                        binary_path: Some(output_path),
                        test_failures: Vec::new(),
                    }
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
                binary_path: None,
                test_failures: Vec::new(),
            },
        }
    }

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
                binary_path: None,
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
                binary_path: None,
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
                        binary_path: None,
                        test_failures: Vec::new(),
                    };
                }

                // The compiled test binary's path is dynamically generated (a
                // temp filename), so it can never be in the sandbox's static
                // command allowlist checked by `is_command_allowed()` — without
                // this, every run below hit `SandboxError::CommandNotAllowed`
                // and silently reported `tests_passed: 0, tests_failed: 0` with
                // the real reason buried in `runtime_error` (a field callers,
                // including this crate's own benchmarks, don't check). This
                // meant `execute_rust_with_inline_tests` — the sole compiler
                // verification path used by both `CodeOrchestrator`'s own
                // acceptance gate and every Rust benchmark run this week —
                // could compile code but never actually execute a single test,
                // silently passing anything that merely compiled. Mirrors the
                // identical, correct fix already present in the sibling
                // `execute_rust()` function just above (line ~360).
                if let Some(path_str) = output_path.to_str() {
                    self.sandbox.allow_command(path_str);
                }
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
                            binary_path: Some(output_path),
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
                        binary_path: Some(output_path),
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
                binary_path: None,
                test_failures: Vec::new(),
            },
        }
    }

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
                binary_path: None,
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
                binary_path: None,
                test_failures: Vec::new(),
            };
        }

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
                        binary_path: None,
                        test_failures: Vec::new(),
                    };
                }

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
                        binary_path: None,
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
                        binary_path: None,
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
                binary_path: None,
                test_failures: Vec::new(),
            },
        }
    }

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
                binary_path: None,
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
                binary_path: None,
                test_failures: Vec::new(),
            },
        }
    }

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

#[derive(Debug, Clone)]
pub struct CompileError {
    pub message: String,
    pub code: Option<String>,
    pub file: Option<String>,
    pub line: Option<usize>,
    pub column: Option<usize>,
    pub category: ErrorCategory,
    pub suggested_replacement: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    TypeMismatch,
    MissingImport,
    BorrowError,
    MovedValue,
    LifetimeError,
    VisibilityError,
    UnusedCode,
    MissingImpl,
    UndeclaredGeneric,
    UnwantedMain,
    SyntaxError,
    Timeout,
    LinkerError,
    SandboxError,
    Other,
}

impl CompileError {
    fn from_rustc_output(error_line: &str, context_lines: &[&str]) -> Self {
        let code = Self::extract_error_code(error_line);
        let category = Self::categorize(&code, error_line);

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

    fn extract_unresolved_type(message: &str) -> Option<String> {
        if message.contains("cannot find type") {
            if let Some(start) = message.find('`') {
                if let Some(end) = message[start + 1..].find('`') {
                    return Some(message[start + 1..start + 1 + end].to_string());
                }
            }
        }
        None
    }

    fn extract_error_code(line: &str) -> Option<String> {
        if let Some(start) = line.find("[E") {
            if let Some(end) = line[start..].find(']') {
                return Some(line[start + 1..start + end].to_string());
            }
        }
        None
    }

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

    fn categorize(code: &Option<String>, message: &str) -> ErrorCategory {
        if let Some(c) = code {
            match c.as_str() {
                "E0308" => ErrorCategory::TypeMismatch,
                "E0277" if message.contains("expected") => ErrorCategory::TypeMismatch,
                "E0277" => ErrorCategory::MissingImpl,
                "E0412" => {
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

pub fn parse_structured_errors(stderr: &str) -> Vec<CompileError> {
    let lines: Vec<&str> = stderr.lines().collect();
    let mut errors = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        if lines[i].starts_with("error") {
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

#[derive(Debug, serde::Deserialize)]
struct RustcJsonEnvelope {
    reason: Option<String>,
    message: Option<RustcDiagnostic>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RustcDiagnostic {
    pub message: String,
    pub code: Option<RustcDiagnosticCode>,
    pub level: String,
    pub spans: Vec<RustcSpan>,
    pub children: Vec<RustcDiagnostic>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RustcDiagnosticCode {
    pub code: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RustcSpan {
    pub file_name: String,
    pub line_start: usize,
    pub line_end: usize,
    pub column_start: usize,
    pub column_end: usize,
    pub suggested_replacement: Option<String>,
    pub is_primary: bool,
}

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

        let primary_span = diag.spans.iter().find(|s| s.is_primary);
        let (file, line_num, column) = match primary_span {
            Some(s) => (
                Some(s.file_name.clone()),
                Some(s.line_start),
                Some(s.column_start),
            ),
            None => (None, None, None),
        };

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

fn strip_test_module(source: &str) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut result = Vec::new();
    let mut in_test_module = false;
    let mut brace_depth = 0i32;
    let mut skip_cfg_test = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        if trimmed == "#[cfg(test)]" {
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

    while result.last().map_or(false, |l| l.trim().is_empty()) {
        result.pop();
    }

    result.join("\n")
}

pub fn try_auto_fix(source: &str, errors: &[String]) -> Option<String> {
    let mut fixed = source.to_string();
    let mut any_fix = false;

    for error in errors {
        let err_lower = error.to_lowercase();

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

pub fn try_auto_fix_structured(source: &str, errors: &[CompileError]) -> Option<String> {
    let mut lines: Vec<String> = source.lines().map(|l| l.to_string()).collect();
    let mut any_fix = false;
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

            ErrorCategory::BorrowError if error.message.contains("cannot move out of") => {
                if let Some(var) = extract_between(&error.message, "`", "`") {
                    let var_clean = var.trim_start_matches('*');
                    if let Some(idx) = target_line {
                        if idx < lines.len() {
                            let line = &lines[idx];
                            let pattern = var_clean;
                            if let Some(pos) = line.find(pattern) {
                                let after_var = pos + pattern.len();
                                let after = &line[after_var..];
                                if !after.starts_with(".clone()") {
                                    let next_char = after.chars().next();
                                    if matches!(next_char, Some(')' | ',' | ';' | ' ' | '.')) {
                                        let new_line = format!(
                                            "{}.clone(){}",
                                            &line[..after_var],
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

            ErrorCategory::LifetimeError
                if error.message.contains("missing lifetime specifier") =>
            {
                if let Some(idx) = target_line {
                    if idx < lines.len() {
                        let line = &lines[idx];
                        if line.contains("fn ") && line.contains("-> &") {
                            let mut new_line = line.clone();
                            if !new_line.contains("<'") {
                                if let Some(paren) = new_line.find('(') {
                                    new_line.insert_str(paren, "<'a>");
                                }
                            }
                            if let Some(arrow) = new_line.find("-> &") {
                                let rest = &new_line[arrow..];
                                if !rest.contains("-> &'") {
                                    new_line = new_line.replacen("-> &", "-> &'a ", 1);
                                }
                            }
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

            ErrorCategory::MissingImpl => {
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
            _ => {}
        }
    }

    if any_fix {
        Some(lines.join("\n"))
    } else {
        None
    }
}

fn extract_between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let start_idx = text.find(start)? + start.len();
    let remaining = &text[start_idx..];
    let end_idx = remaining.find(end)?;
    Some(&remaining[..end_idx])
}

fn parse_test_output(stdout: &str) -> (usize, usize) {
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

pub fn parse_test_failure_details(test_output: &str) -> Vec<TestFailure> {
    let mut failures: Vec<TestFailure> = Vec::new();
    let lines: Vec<&str> = test_output.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i].trim();

        if line.starts_with("---- ") && line.ends_with(" stdout ----") {
            let test_name = line
                .strip_prefix("---- ")
                .and_then(|s| s.strip_suffix(" stdout ----"))
                .unwrap_or("unknown")
                .to_string();

            let mut message = String::new();
            let mut assertion = None;
            let mut expected = None;
            let mut actual = None;
            let mut j = i + 1;

            while j < lines.len() {
                let detail = lines[j].trim();
                if detail.starts_with("---- ") || detail == "failures:" {
                    break;
                }

                if let Some(pos) = detail.find("panicked at") {
                    let msg_start = pos + "panicked at".len();
                    let msg = detail[msg_start..]
                        .trim()
                        .trim_matches('\'')
                        .trim_matches('"');
                    message = msg.to_string();
                }

                if detail.contains("assert_eq!")
                    || detail.contains("assert_ne!")
                    || detail.contains("assert!")
                {
                    assertion = Some(detail.to_string());
                }

                let strip_trailing_artifacts = |s: &str| -> String {
                    let mut trimmed = s.trim().to_string();
                    if let Some(pos) = trimmed.rfind("', ") {
                        let candidate = &trimmed[pos + 3..];
                        if candidate.contains(':') || candidate.contains('/') {
                            trimmed = trimmed[..pos].to_string();
                        }
                    }
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
                if let Some(pos) = failures.iter().position(|f| f.test_name == test_name) {
                    failures[pos] = detail_entry;
                } else {
                    failures.push(detail_entry);
                }
            }
            i = j;
            continue;
        }

        if line.starts_with("test ") && line.ends_with("... FAILED") {
            let test_name = line
                .strip_prefix("test ")
                .and_then(|s| s.strip_suffix("... FAILED"))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();

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
            elapsed: Duration::from_millis(0),
            simulated: false,
            binary_path: None,
            test_failures: Vec::new(),
        };
        let surprise = result.to_surprise();
        assert!(surprise > 0.8);
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
            elapsed: Duration::from_millis(0),
            simulated: false,
            binary_path: None,
            test_failures: Vec::new(),
        };
        assert!(success.is_success());
    }

    #[test]
    fn test_parse_compile_errors() {
        let stderr = "error[E0308]: mismatched types";
        let errors = parse_compile_errors(stderr);
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_simulated_success_is_labeled_simulated_and_not_verified() {
        let result = ExecutionResult::simulated_success();
        assert!(
            result.simulated,
            "sandbox-rejected execution must be labeled simulated"
        );
        assert!(
            !result.is_success(),
            "a simulated result must never count as full verification"
        );
    }

    /// Regression test for a real bug (found 2026-07-07 via the Rust-native
    /// orchestrator benchmark): the compiled test binary's dynamically
    /// generated path was never in the sandbox's static command allowlist,
    /// so `execute_rust_with_inline_tests` could compile code but never
    /// actually execute a single test — every run silently reported
    /// `tests_passed: 0, tests_failed: 0` (the real
    /// `SandboxError::CommandNotAllowed` was buried in `runtime_error`,
    /// which callers didn't check), making every candidate look like it
    /// passed by simply not running anything. This test would have failed
    /// before the fix: a genuinely failing assertion would have reported
    /// `tests_failed: 0` instead of `1`.
    #[test]
    fn test_execute_rust_with_inline_tests_actually_runs_tests() {
        let mut executor = CodeExecutor::with_real_execution();
        if !executor.supports_real_execution() {
            // No rustc available in this environment — nothing to verify.
            return;
        }

        let passing = "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn t() { assert_eq!(add(2, 3), 5); }\n}\n";
        let result = executor.execute_rust_with_inline_tests(passing);
        assert!(
            result.compiled,
            "expected compile success: {:?}",
            result.compile_errors
        );
        assert_eq!(
            result.tests_passed, 1,
            "a genuinely passing test must be detected as passed, not silently skipped (runtime_error: {:?})",
            result.runtime_error
        );
        assert_eq!(result.tests_failed, 0);

        let failing = "pub fn add(a: i32, b: i32) -> i32 { a + b }\n\n\
             #[cfg(test)]\nmod tests {\n    use super::*;\n    #[test]\n    fn t() { assert_eq!(add(2, 3), 999); }\n}\n";
        let result = executor.execute_rust_with_inline_tests(failing);
        assert!(result.compiled);
        assert_eq!(
            result.tests_failed, 1,
            "a genuinely failing test must be detected as failed, not silently reported as 0/0 (runtime_error: {:?})",
            result.runtime_error
        );
    }
}
