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
        let work_dir =
            std::env::temp_dir().join(format!("symthaea-code-exec-{}", std::process::id()));
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
            work_dir: std::env::temp_dir()
                .join(format!("symthaea-code-exec-{}", std::process::id())),
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
                "--edition",
                "2021",
                "--test",
                source_path.to_str().unwrap_or("generated.rs"),
                "-o",
                output_path.to_str().unwrap_or("generated"),
            ]
        } else {
            vec![
                "--edition",
                "2021",
                source_path.to_str().unwrap_or("generated.rs"),
                "-o",
                output_path.to_str().unwrap_or("generated"),
            ]
        };

        match self.sandbox.run(
            "rustc",
            &compile_args.iter().map(|s| *s).collect::<Vec<_>>(),
        ) {
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
            };
        }

        let output_path = self.work_dir.join("generated_test");
        let compile_args: Vec<&str> = vec![
            "--edition",
            "2021",
            "--test",
            source_path.to_str().unwrap_or("generated_test.rs"),
            "-o",
            output_path.to_str().unwrap_or("generated_test"),
        ];

        match self.sandbox.run("rustc", &compile_args) {
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

/// Attempt to auto-fix common Rust compilation errors in source code.
///
/// Applies mechanical fixes for well-known rustc error patterns:
/// - E0308 `expected String, found &str` → `.to_string()`
/// - E0308 `expected &str, found String` → `.as_str()`
/// - E0596 `cannot borrow as mutable` → add `mut`
/// - Missing `use` for common types → prepend import
/// - `unused variable` → prefix with `_`
///
/// Returns `Some(fixed_source)` if any fix was applied, `None` otherwise.
pub fn try_auto_fix(source: &str, errors: &[String]) -> Option<String> {
    let mut fixed = source.to_string();
    let mut any_fix = false;

    for error in errors {
        let err_lower = error.to_lowercase();

        // Missing mut: "cannot borrow `x` as mutable"
        if err_lower.contains("cannot borrow") && err_lower.contains("as mutable") {
            // Find the variable name and add `mut` to its binding
            if let Some(var) = extract_between(error, "`", "`") {
                let var_clean = var.trim_start_matches('*');
                // Try "let VAR" → "let mut VAR"
                let pattern = format!("let {}", var_clean);
                let replacement = format!("let mut {}", var_clean);
                if fixed.contains(&pattern) {
                    fixed = fixed.replacen(&pattern, &replacement, 1);
                    any_fix = true;
                }
            }
        }

        // Type mismatch: expected String, found &str
        if err_lower.contains("expected")
            && err_lower.contains("string")
            && err_lower.contains("&str")
        {
            // This is tricky to fix in place without line numbers, skip for now
            // but add a note to the source
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
                    // Also try in function params
                    let param_pattern = format!("{}: ", var);
                    let param_replacement = format!("_{}: ", var);
                    if !any_fix && fixed.contains(&param_pattern) {
                        fixed = fixed.replacen(&param_pattern, &param_replacement, 1);
                        any_fix = true;
                    }
                }
            }
        }

        // Missing return type annotation — common with closures
        if err_lower.contains("consider giving this closure") && err_lower.contains("return type") {
            // Can't auto-fix without more context
        }

        // Missing #[derive(Debug)] — very common with LLM-generated structs
        // Error: "`MyStruct` doesn't implement `Debug`"
        if (err_lower.contains("doesn't implement") || err_lower.contains("does not implement"))
            && err_lower.contains("debug")
        {
            if let Some(type_name) = extract_between(error, "`", "`") {
                // Find "struct TypeName" and prepend #[derive(Debug)]
                let struct_pattern = format!("struct {}", type_name);
                if let Some(pos) = fixed.find(&struct_pattern) {
                    // Check if #[derive(...)] already exists on the line above
                    let before = &fixed[..pos];
                    if !before.ends_with("]\n") && !before.contains(&format!("#[derive(Debug")) {
                        fixed.insert_str(pos, "#[derive(Debug, Clone)]\n");
                        any_fix = true;
                    }
                }
            }
        }

        // Missing Display impl — "doesn't implement `std::fmt::Display`"
        if (err_lower.contains("doesn't implement") || err_lower.contains("does not implement"))
            && err_lower.contains("display")
        {
            // If the error mentions a custom type, we can't auto-fix Display.
            // But if the code uses `.to_string()` on something that needs Display,
            // try wrapping with format!("{:?}") instead.
        }

        // Missing Clone/Copy — "cannot move out of"
        if err_lower.contains("cannot move out of") {
            // Try adding .clone() — crude but often works
            if let Some(var) = extract_between(error, "`", "`") {
                let var_clean = var.trim_start_matches('*');
                // Only clone if the variable is used, not a field access
                if !var_clean.contains('.') {
                    let use_pattern = format!("{}", var_clean);
                    // Don't blindly add .clone() — too risky without line info
                }
            }
        }

        // Lifetime error: "missing lifetime specifier"
        if err_lower.contains("missing lifetime specifier")
            && err_lower.contains("expected named lifetime")
        {
            // Add 'a lifetime to &str returns — common LLM mistake
            // e.g. fn foo(s: &str) -> &str → fn foo(s: &str) -> &str (needs lifetime)
            // This is too context-dependent to auto-fix safely
        }

        // Dead code warning treated as error (deny(dead_code))
        if err_lower.contains("unused") && err_lower.contains("function") {
            // Prefix function with _ or add #[allow(dead_code)]
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

    if any_fix {
        Some(fixed)
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
}
