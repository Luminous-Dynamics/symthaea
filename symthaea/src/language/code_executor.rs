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
use std::time::Duration;

use crate::infrastructure::sandbox::{Sandbox, SandboxError};

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
}

impl ExecutionResult {
    /// Whether the code is fully successful (compiled + all tests passed)
    pub fn is_success(&self) -> bool {
        self.compiled && self.tests_failed == 0
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
        let work_dir = std::env::temp_dir().join(format!(
            "symthaea-code-exec-{}",
            std::process::id()
        ));
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
            work_dir: std::env::temp_dir().join(format!(
                "symthaea-code-exec-{}",
                std::process::id()
            )),
        }
    }

    /// Compile Rust source code and optionally run tests.
    ///
    /// Writes source to a temp file, invokes `rustc --edition 2021`,
    /// and captures errors. If `test_source` is provided, appends it
    /// and runs with `--test`.
    pub fn execute_rust(
        &mut self,
        source: &str,
        test_source: Option<&str>,
    ) -> ExecutionResult {
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
            };
        }

        // Write source file
        let source_path = self.work_dir.join("generated.rs");
        let full_source = if let Some(tests) = test_source {
            format!("{source}\n\n#[cfg(test)]\nmod tests {{\n    use super::*;\n{tests}\n}}")
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
            };
        }

        // Compile
        let output_path = self.work_dir.join("generated");
        let compile_args = if test_source.is_some() {
            vec![
                "--edition", "2021", "--test",
                source_path.to_str().unwrap_or("generated.rs"),
                "-o", output_path.to_str().unwrap_or("generated"),
            ]
        } else {
            vec![
                "--edition", "2021",
                source_path.to_str().unwrap_or("generated.rs"),
                "-o", output_path.to_str().unwrap_or("generated"),
            ]
        };

        match self.sandbox.run("rustc", &compile_args.iter().map(|s| *s).collect::<Vec<_>>()) {
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
                    };
                }

                // If tests, run the compiled test binary
                if test_source.is_some() {
                    match self.sandbox.run(
                        output_path.to_str().unwrap_or("./generated"),
                        &[],
                    ) {
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
            },
        }
    }

    /// Clean up temporary files
    pub fn cleanup(&self) {
        let _ = std::fs::remove_dir_all(&self.work_dir);
    }
}

impl Drop for CodeExecutor {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// Parse rustc error output into individual error messages
fn parse_compile_errors(stderr: &str) -> Vec<String> {
    stderr
        .lines()
        .filter(|line| line.starts_with("error"))
        .map(|line| line.to_string())
        .collect()
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
        };
        let surprise = result.to_surprise();
        assert!(surprise > 0.8, "Compile failure should have high surprise: {surprise}");
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
        };
        let surprise = result.to_surprise();
        assert!(surprise > 0.3 && surprise < 0.8, "Test failure moderate surprise: {surprise}");
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
        };
        assert!(success.is_success());

        let failure = ExecutionResult {
            compiled: false,
            compile_errors: vec!["error".into()],
            tests_passed: 0,
            tests_failed: 0,
            test_output: String::new(),
            runtime_error: None,
            elapsed: Duration::from_millis(10),
            simulated: false,
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
    fn test_simulated_executor() {
        let mut executor = CodeExecutor::new();
        let result = executor.execute_rust("fn main() {}", None);
        assert!(result.simulated);
        assert!(result.compiled);
    }

    #[test]
    fn test_nix_evaluation_simulated() {
        let mut executor = CodeExecutor::new();
        let result = executor.evaluate_nix("1 + 1");
        assert!(result.simulated);
        assert!(result.compiled);
    }
}
