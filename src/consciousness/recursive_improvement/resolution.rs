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

use std::io::{Read, Stdio};
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use super::OutcomeCategory;

/// Whether a resolver actually observed an outcome or failed to obtain one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionDisposition {
    /// A command exit/resource state was observed and classified.
    Observed,
    /// The command exceeded its deadline; there is no outcome to score.
    TimedOut,
    /// The resolver could not establish an outcome (for example, spawn/wait failure).
    Unclear,
}

/// Result of a resolution attempt.
///
/// `outcome` is `None` whenever no authoritative outcome was observed. Callers must not
/// turn timeout/spawn/wait failures into calibration successes or ordinary task failures.
#[derive(Debug)]
pub struct ResolutionResult {
    /// Observed outcome. Unresolved results deliberately carry `None`.
    pub outcome: Option<OutcomeCategory>,
    /// How the resolution attempt ended.
    pub disposition: ResolutionDisposition,
    /// Captured output, capped to avoid unbounded memory use.
    pub raw_output: Option<String>,
    /// Human-readable reason for the outcome or unresolved state.
    pub reason: Option<String>,
}

const MAX_CAPTURED_OUTPUT_BYTES: usize = 64 * 1024;
const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Drain a child's pipe while retaining at most `limit` bytes.
fn read_bounded<R: Read>(mut reader: R, limit: usize) -> (Vec<u8>, bool) {
    let mut retained = Vec::with_capacity(limit.min(8 * 1024));
    let mut buffer = [0u8; 4096];
    let mut truncated = false;
    loop {
        match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => {
                let remaining = limit.saturating_sub(retained.len());
                let keep = count.min(remaining);
                retained.extend_from_slice(&buffer[..keep]);
                if keep < count {
                    truncated = true;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => break,
        }
    }
    (retained, truncated)
}

fn join_output(
    handle: thread::JoinHandle<(Vec<u8>, bool)>,
) -> (Vec<u8>, bool) {
    handle.join().unwrap_or_default()
}

fn choose_output(
    stdout: (Vec<u8>, bool),
    stderr: (Vec<u8>, bool),
) -> String {
    let (bytes, truncated) = if !stdout.0.is_empty() {
        stdout
    } else {
        stderr
    };
    let mut text = String::from_utf8_lossy(&bytes).into_owned();
    if truncated {
        text.push_str("\n[output truncated at 65536 bytes]");
    }
    text
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

    /// Execute the command with an enforced wall-clock timeout.
    ///
    /// Stdout/stderr are drained concurrently and each capture retains at most 64 KiB while
    /// continuing to drain excess bytes, preventing ordinary high-output commands from blocking
    /// on a full pipe or consuming unbounded memory. The direct child is killed and reaped on
    /// timeout. Callers must treat `outcome: None` as unresolved, not as a scored task failure.
    pub fn execute(&self, timeout: Duration) -> ResolutionResult {
        let mut cmd = Command::new(&self.program);
        cmd.args(&self.args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(error) => {
                return ResolutionResult {
                    outcome: None,
                    disposition: ResolutionDisposition::Unclear,
                    raw_output: None,
                    reason: Some(format!("Failed to spawn '{}': {}", self.program, error)),
                };
            }
        };

        let Some(stdout) = child.stdout.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return ResolutionResult {
                outcome: None,
                disposition: ResolutionDisposition::Unclear,
                raw_output: None,
                reason: Some("Child stdout pipe was unavailable".to_string()),
            };
        };
        let Some(stderr) = child.stderr.take() else {
            let _ = child.kill();
            let _ = child.wait();
            return ResolutionResult {
                outcome: None,
                disposition: ResolutionDisposition::Unclear,
                raw_output: None,
                reason: Some("Child stderr pipe was unavailable".to_string()),
            };
        };

        let stdout_reader = thread::spawn(move || read_bounded(stdout, MAX_CAPTURED_OUTPUT_BYTES));
        let stderr_reader = thread::spawn(move || read_bounded(stderr, MAX_CAPTURED_OUTPUT_BYTES));
        let started = Instant::now();

        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break status,
                Ok(None) if started.elapsed() >= timeout => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let stdout = join_output(stdout_reader);
                    let stderr = join_output(stderr_reader);
                    return ResolutionResult {
                        outcome: None,
                        disposition: ResolutionDisposition::TimedOut,
                        raw_output: Some(choose_output(stdout, stderr)),
                        reason: Some(format!(
                            "Command '{}' timed out after {:?}",
                            self.program, timeout
                        )),
                    };
                }
                Ok(None) => thread::sleep(POLL_INTERVAL),
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    let stdout = join_output(stdout_reader);
                    let stderr = join_output(stderr_reader);
                    return ResolutionResult {
                        outcome: None,
                        disposition: ResolutionDisposition::Unclear,
                        raw_output: Some(choose_output(stdout, stderr)),
                        reason: Some(format!("Failed while waiting for '{}': {}", self.program, error)),
                    };
                }
            }
        };

        let stdout = join_output(stdout_reader);
        let stderr = join_output(stderr_reader);
        let raw = choose_output(stdout, stderr);
        let code = status.code().unwrap_or(-1);

        if self.expected_codes.contains(&code) {
            ResolutionResult {
                outcome: Some(OutcomeCategory::Success),
                disposition: ResolutionDisposition::Observed,
                raw_output: Some(raw),
                reason: None,
            }
        } else {
            ResolutionResult {
                outcome: Some(OutcomeCategory::SafeFailure),
                disposition: ResolutionDisposition::Observed,
                raw_output: Some(raw),
                reason: Some(format!(
                    "Exit code {} not in expected {:?}",
                    code, self.expected_codes
                )),
            }
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
                outcome: Some(OutcomeCategory::Success),
                disposition: ResolutionDisposition::Observed,
                raw_output: Some(format!(
                    "Path '{}': exists={}, expected_exists={}",
                    self.path, exists, self.expect_exists
                )),
                reason: None,
            }
        } else {
            ResolutionResult {
                outcome: Some(OutcomeCategory::SafeFailure),
                disposition: ResolutionDisposition::Observed,
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
        assert!(matches!(result.outcome, Some(OutcomeCategory::Success)));
    }

    #[test]
    fn timeout_is_unresolved_and_child_is_reaped_promptly() {
        let resolver = ExitCodeResolver::new("sleep", vec![0]).with_args(vec!["5".to_string()]);
        let started = Instant::now();
        let result = resolver.execute(Duration::from_millis(50));
        assert_eq!(result.outcome, None);
        assert_eq!(result.disposition, ResolutionDisposition::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(result.reason.as_deref().is_some_and(|reason| reason.contains("timed out")));
    }

    #[test]
    fn spawn_failure_is_unresolved_not_an_observed_task_failure() {
        let resolver = ExitCodeResolver::new(
            "/definitely/not/a/real/symthaea-command",
            vec![0],
        );
        let result = resolver.execute(Duration::from_secs(1));
        assert_eq!(result.outcome, None);
        assert_eq!(result.disposition, ResolutionDisposition::Unclear);
    }

    #[test]
    fn command_output_capture_is_bounded() {
        let resolver = ExitCodeResolver::new("yes", vec![0]);
        let result = resolver.execute(Duration::from_millis(100));
        assert_eq!(result.disposition, ResolutionDisposition::TimedOut);
        let bytes = result.raw_output.as_deref().unwrap_or_default().len();
        assert!(bytes <= MAX_CAPTURED_OUTPUT_BYTES + 64);
    }

    #[test]
    fn test_exit_code_resolver_failure() {
        let resolver = ExitCodeResolver::new("false", vec![0]);
        let result = resolver.execute(Duration::from_secs(5));
        assert!(matches!(result.outcome, Some(OutcomeCategory::SafeFailure)));
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
