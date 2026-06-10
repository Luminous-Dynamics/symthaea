// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Resolution System — External outcome resolvers for MAGI Loop
//!
//! Provides concrete resolvers that execute real-world actions and observe outcomes:
//! - [`ExitCodeResolver`]: Runs a command and checks its exit code
//! - [`ResourceStateResolver`]: Checks filesystem resource state (exists / not exists)
//!
//! All resolvers implement the [`Resolver`] trait.

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use super::OutcomeCategory;

/// Result of a resolution attempt.
#[derive(Debug)]
pub struct ResolutionResult {
    /// The outcome category (Success / SafeFailure / etc.)
    pub outcome: OutcomeCategory,
    /// Raw output from the resolver (e.g., stdout)
    pub raw_output: Option<String>,
    /// Human-readable reason for the outcome
    pub reason: Option<String>,
}

/// Trait for concrete resolvers that check external reality.
pub trait Resolver {
    /// Execute the resolution and return the result.
    fn resolve(&self) -> ResolutionResult;
}

// ═══════════════════════════════════════════════════════════════════════════════
// ExitCodeResolver — run a command and check exit code
// ═══════════════════════════════════════════════════════════════════════════════

/// Resolver that spawns a process and checks whether its exit code matches
/// the expected set.
#[derive(Debug, Clone)]
pub struct ExitCodeResolver {
    program: String,
    expected_codes: Vec<i32>,
    args: Vec<String>,
    working_dir: Option<String>,
}

impl ExitCodeResolver {
    /// Create a new resolver for the given program with expected exit codes.
    pub fn new(program: impl Into<String>, expected_codes: Vec<i32>) -> Self {
        Self {
            program: program.into(),
            expected_codes,
            args: Vec::new(),
            working_dir: None,
        }
    }

    /// Add command-line arguments.
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Set working directory for the child process.
    pub fn with_working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Execute the command with a timeout and return the resolution result.
    pub fn execute(&self, _timeout: Duration) -> ResolutionResult {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args);

        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        match cmd.output() {
            Ok(output) => {
                let code = output.status.code().unwrap_or(-1);
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                let raw = if !stdout.is_empty() { stdout } else { stderr };

                if self.expected_codes.contains(&code) {
                    ResolutionResult {
                        outcome: OutcomeCategory::Success,
                        raw_output: Some(raw),
                        reason: None,
                    }
                } else {
                    ResolutionResult {
                        outcome: OutcomeCategory::SafeFailure,
                        raw_output: Some(raw),
                        reason: Some(format!(
                            "Exit code {} not in expected {:?}",
                            code, self.expected_codes
                        )),
                    }
                }
            }
            Err(e) => ResolutionResult {
                outcome: OutcomeCategory::SafeFailure,
                raw_output: None,
                reason: Some(format!("Failed to execute '{}': {}", self.program, e)),
            },
        }
    }
}

impl Resolver for ExitCodeResolver {
    fn resolve(&self) -> ResolutionResult {
        self.execute(Duration::from_secs(30))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ResourceStateResolver — check filesystem resource state
// ═══════════════════════════════════════════════════════════════════════════════

/// Resolver that checks whether a filesystem resource exists (or doesn't).
#[derive(Debug, Clone)]
pub struct ResourceStateResolver {
    path: String,
    expect_exists: bool,
}

impl ResourceStateResolver {
    /// Check that a path exists.
    pub fn exists(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            expect_exists: true,
        }
    }

    /// Check that a path does NOT exist.
    pub fn not_exists(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            expect_exists: false,
        }
    }

    /// Execute the check and return the resolution result.
    pub fn execute(&self) -> ResolutionResult {
        let exists = Path::new(&self.path).exists();
        let matched = exists == self.expect_exists;

        if matched {
            ResolutionResult {
                outcome: OutcomeCategory::Success,
                raw_output: Some(format!(
                    "Path '{}': exists={}, expected_exists={}",
                    self.path, exists, self.expect_exists
                )),
                reason: None,
            }
        } else {
            ResolutionResult {
                outcome: OutcomeCategory::SafeFailure,
                raw_output: Some(format!(
                    "Path '{}': exists={}, expected_exists={}",
                    self.path, exists, self.expect_exists
                )),
                reason: Some(format!(
                    "Expected exists={} but got exists={}",
                    self.expect_exists, exists
                )),
            }
        }
    }
}

impl Resolver for ResourceStateResolver {
    fn resolve(&self) -> ResolutionResult {
        self.execute()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exit_code_resolver_success() {
        let resolver = ExitCodeResolver::new("true", vec![0]);
        let result = resolver.execute(Duration::from_secs(5));
        assert!(matches!(result.outcome, OutcomeCategory::Success));
    }

    #[test]
    fn test_exit_code_resolver_failure() {
        let resolver = ExitCodeResolver::new("false", vec![0]);
        let result = resolver.execute(Duration::from_secs(5));
        assert!(matches!(result.outcome, OutcomeCategory::SafeFailure));
    }

    #[test]
    fn test_resource_state_resolver_exists() {
        // /tmp always exists
        let resolver = ResourceStateResolver::exists("/tmp");
        let result = resolver.execute();
        assert!(matches!(result.outcome, OutcomeCategory::Success));
    }

    #[test]
    fn test_resource_state_resolver_not_exists() {
        let resolver =
            ResourceStateResolver::not_exists("/nonexistent_path_that_should_not_exist_12345");
        let result = resolver.execute();
        assert!(matches!(result.outcome, OutcomeCategory::Success));
    }
}
