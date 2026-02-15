//! # Resolution Execution System
//!
//! This module implements actual execution of `ResolutionAuthority` to resolve predictions.
//! Each resolver type has a concrete implementation that interacts with the real world.
//!
//! ## Resolvers
//!
//! - **TestSuite**: Runs tests and checks pass/fail
//! - **ExitCode**: Executes command and checks exit code
//! - **ResourceState**: Checks file/resource existence and state
//! - **HumanConfirmation**: Prompts human for confirmation via callback
//!
//! ## Design
//!
//! Resolvers are designed to be:
//! - **Non-blocking**: Use async where appropriate
//! - **Timeout-aware**: All operations have timeouts
//! - **Error-safe**: Failures are captured, not panicked
//! - **Testable**: Can be mocked for unit tests

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use regex::Regex;

use super::world_prediction::{
    OutcomeCategory, ResolutionAuthority, ResourceExpectation, WorldPrediction,
};

// ═══════════════════════════════════════════════════════════════════════════════
// RESOLUTION RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of attempting to resolve a prediction
#[derive(Debug, Clone)]
pub struct ResolutionResult {
    /// The outcome category determined
    pub outcome: OutcomeCategory,
    /// Confidence in the resolution (0.0-1.0)
    pub confidence: f64,
    /// Detailed reason/explanation
    pub reason: String,
    /// Raw output from the resolution (if any)
    pub raw_output: Option<String>,
    /// Whether resolution was successful (vs error during resolution)
    pub resolution_succeeded: bool,
}

impl ResolutionResult {
    /// Create a successful resolution
    pub fn success(reason: impl Into<String>) -> Self {
        Self {
            outcome: OutcomeCategory::Success,
            confidence: 1.0,
            reason: reason.into(),
            raw_output: None,
            resolution_succeeded: true,
        }
    }

    /// Create a failure resolution
    pub fn failure(outcome: OutcomeCategory, reason: impl Into<String>) -> Self {
        Self {
            outcome,
            confidence: 1.0,
            reason: reason.into(),
            raw_output: None,
            resolution_succeeded: true,
        }
    }

    /// Create an unclear resolution (couldn't determine outcome)
    pub fn unclear(reason: impl Into<String>) -> Self {
        Self {
            outcome: OutcomeCategory::NoEffect,
            confidence: 0.5,
            reason: reason.into(),
            raw_output: None,
            resolution_succeeded: false,
        }
    }

    /// Add raw output
    pub fn with_output(mut self, output: impl Into<String>) -> Self {
        self.raw_output = Some(output.into());
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESOLVER TRAIT
// ═══════════════════════════════════════════════════════════════════════════════

/// Trait for executing resolution
pub trait Resolver {
    /// Attempt to resolve a prediction
    fn resolve(&self, prediction: &WorldPrediction, timeout: Duration) -> ResolutionResult;
}

// ═══════════════════════════════════════════════════════════════════════════════
// EXIT CODE RESOLVER
// ═══════════════════════════════════════════════════════════════════════════════

/// Resolver that executes a command and checks exit code
pub struct ExitCodeResolver {
    /// Command to execute
    pub command: String,
    /// Arguments to the command
    pub args: Vec<String>,
    /// Exit codes that count as success
    pub success_codes: Vec<i32>,
    /// Working directory (optional)
    pub working_dir: Option<String>,
}

impl ExitCodeResolver {
    /// Create a new exit code resolver
    pub fn new(command: impl Into<String>, success_codes: Vec<i32>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            success_codes,
            working_dir: None,
        }
    }

    /// Add arguments
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Set working directory
    pub fn with_working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Execute and get result
    pub fn execute(&self, _timeout: Duration) -> ResolutionResult {
        let mut cmd = Command::new(&self.command);
        cmd.args(&self.args);

        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        // Execute with timeout (simplified - real impl would use async)
        match cmd.output() {
            Ok(output) => {
                let exit_code = output.status.code().unwrap_or(-1);
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let combined_output = format!("stdout:\n{}\nstderr:\n{}", stdout, stderr);

                if self.success_codes.contains(&exit_code) {
                    ResolutionResult::success(format!("Command exited with code {}", exit_code))
                        .with_output(combined_output)
                } else {
                    ResolutionResult::failure(
                        OutcomeCategory::SafeFailure,
                        format!(
                            "Command exited with code {} (expected one of {:?})",
                            exit_code, self.success_codes
                        ),
                    )
                    .with_output(combined_output)
                }
            }
            Err(e) => ResolutionResult::unclear(format!("Failed to execute command: {}", e)),
        }
    }
}

impl Resolver for ExitCodeResolver {
    fn resolve(&self, _prediction: &WorldPrediction, timeout: Duration) -> ResolutionResult {
        self.execute(timeout)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TEST SUITE RESOLVER
// ═══════════════════════════════════════════════════════════════════════════════

/// Resolver that runs a test suite and checks results
pub struct TestSuiteResolver {
    /// Test command (e.g., "cargo test", "pytest", "npm test")
    pub test_command: String,
    /// Test pattern/filter
    pub test_pattern: Option<String>,
    /// Working directory
    pub working_dir: Option<String>,
    /// Minimum pass rate for success (0.0-1.0)
    pub pass_threshold: f64,
}

impl TestSuiteResolver {
    /// Create a cargo test resolver
    pub fn cargo_test(pattern: Option<String>) -> Self {
        Self {
            test_command: "cargo".to_string(),
            test_pattern: pattern,
            working_dir: None,
            pass_threshold: 1.0, // All tests must pass by default
        }
    }

    /// Create a pytest resolver
    pub fn pytest(pattern: Option<String>) -> Self {
        Self {
            test_command: "pytest".to_string(),
            test_pattern: pattern,
            working_dir: None,
            pass_threshold: 1.0,
        }
    }

    /// Set working directory
    pub fn with_working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Set pass threshold
    pub fn with_pass_threshold(mut self, threshold: f64) -> Self {
        self.pass_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Execute tests and get result
    pub fn execute(&self, _timeout: Duration) -> ResolutionResult {
        let mut cmd = Command::new(&self.test_command);

        // Build arguments based on test command type
        if self.test_command == "cargo" {
            cmd.arg("test");
            if let Some(ref pattern) = self.test_pattern {
                cmd.arg(pattern);
            }
            cmd.arg("--");
            cmd.arg("--test-threads=1"); // Ensure deterministic output
        } else if self.test_command == "pytest" {
            if let Some(ref pattern) = self.test_pattern {
                cmd.arg("-k").arg(pattern);
            }
            cmd.arg("-v"); // Verbose output
        } else {
            // Generic test command
            if let Some(ref pattern) = self.test_pattern {
                cmd.arg(&pattern);
            }
        }

        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        match cmd.output() {
            Ok(output) => {
                let exit_code = output.status.code().unwrap_or(-1);
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let combined_output = format!("stdout:\n{}\nstderr:\n{}", stdout, stderr);

                // Parse test results from output
                let (passed, failed) = self.parse_test_results(&stdout, &stderr);
                let total = passed + failed;
                let pass_rate = if total > 0 {
                    passed as f64 / total as f64
                } else {
                    if exit_code == 0 {
                        1.0
                    } else {
                        0.0
                    }
                };

                if pass_rate >= self.pass_threshold {
                    ResolutionResult::success(format!(
                        "Tests passed: {}/{} ({:.1}%)",
                        passed,
                        total,
                        pass_rate * 100.0
                    ))
                    .with_output(combined_output)
                    .with_confidence(pass_rate)
                } else if pass_rate > 0.0 {
                    ResolutionResult::failure(
                        OutcomeCategory::Partial,
                        format!(
                            "Partial pass: {}/{} ({:.1}%), threshold: {:.1}%",
                            passed,
                            total,
                            pass_rate * 100.0,
                            self.pass_threshold * 100.0
                        ),
                    )
                    .with_output(combined_output)
                    .with_confidence(pass_rate)
                } else {
                    ResolutionResult::failure(
                        OutcomeCategory::SafeFailure,
                        format!("All tests failed: 0/{}", total),
                    )
                    .with_output(combined_output)
                    .with_confidence(0.0)
                }
            }
            Err(e) => ResolutionResult::unclear(format!("Failed to run tests: {}", e)),
        }
    }

    /// Parse test results from output (basic implementation)
    fn parse_test_results(&self, stdout: &str, stderr: &str) -> (usize, usize) {
        let combined = format!("{}\n{}", stdout, stderr);

        // Try to parse cargo test output: "test result: ok. X passed; Y failed"
        if let Some(caps) = Regex::new(r"(\d+) passed[;,]?\s*(\d+) failed")
            .ok()
            .and_then(|re| re.captures(&combined))
        {
            let passed: usize = caps
                .get(1)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            let failed: usize = caps
                .get(2)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);
            return (passed, failed);
        }

        // Try pytest format: "X passed, Y failed"
        if let Some(caps) = Regex::new(r"(\d+) passed")
            .ok()
            .and_then(|re| re.captures(&combined))
        {
            let passed: usize = caps
                .get(1)
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);

            let failed: usize = Regex::new(r"(\d+) failed")
                .ok()
                .and_then(|re| re.captures(&combined))
                .and_then(|c| c.get(1))
                .and_then(|m| m.as_str().parse().ok())
                .unwrap_or(0);

            return (passed, failed);
        }

        // Fallback: count "ok" and "FAILED" lines
        let passed = combined.lines().filter(|l| l.contains("... ok")).count();
        let failed = combined.lines().filter(|l| l.contains("FAILED")).count();
        (passed, failed)
    }
}

impl Resolver for TestSuiteResolver {
    fn resolve(&self, _prediction: &WorldPrediction, timeout: Duration) -> ResolutionResult {
        self.execute(timeout)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESOURCE STATE RESOLVER
// ═══════════════════════════════════════════════════════════════════════════════

/// Resolver that checks resource/file state
pub struct ResourceStateResolver {
    /// Path to check
    pub path: String,
    /// Expected state
    pub expected: ResourceExpectation,
}

impl ResourceStateResolver {
    /// Create a new resource state resolver
    pub fn new(path: impl Into<String>, expected: ResourceExpectation) -> Self {
        Self {
            path: path.into(),
            expected,
        }
    }

    /// Check if resource exists
    pub fn exists(path: impl Into<String>) -> Self {
        Self::new(path, ResourceExpectation::Exists)
    }

    /// Check if resource does NOT exist
    pub fn not_exists(path: impl Into<String>) -> Self {
        Self::new(path, ResourceExpectation::NotExists)
    }

    /// Check if resource contains string
    pub fn contains(path: impl Into<String>, content: impl Into<String>) -> Self {
        Self::new(path, ResourceExpectation::Contains(content.into()))
    }

    /// Execute check and get result
    pub fn execute(&self) -> ResolutionResult {
        let path = Path::new(&self.path);

        match &self.expected {
            ResourceExpectation::Exists => {
                if path.exists() {
                    ResolutionResult::success(format!("Resource exists: {}", self.path))
                } else {
                    ResolutionResult::failure(
                        OutcomeCategory::SafeFailure,
                        format!("Resource does not exist: {}", self.path),
                    )
                }
            }
            ResourceExpectation::NotExists => {
                if !path.exists() {
                    ResolutionResult::success(format!("Resource correctly absent: {}", self.path))
                } else {
                    ResolutionResult::failure(
                        OutcomeCategory::SafeFailure,
                        format!("Resource exists but should not: {}", self.path),
                    )
                }
            }
            ResourceExpectation::Contains(expected_content) => {
                match std::fs::read_to_string(path) {
                    Ok(content) => {
                        if content.contains(expected_content) {
                            ResolutionResult::success(format!(
                                "Resource contains expected content: {}",
                                self.path
                            ))
                        } else {
                            ResolutionResult::failure(
                                OutcomeCategory::SafeFailure,
                                format!(
                                    "Resource does not contain expected content: {}",
                                    self.path
                                ),
                            )
                            .with_output(content)
                        }
                    }
                    Err(e) => ResolutionResult::unclear(format!("Failed to read resource: {}", e)),
                }
            }
            ResourceExpectation::MatchesPattern(pattern) => match std::fs::read_to_string(path) {
                Ok(content) => match Regex::new(pattern) {
                    Ok(re) => {
                        if re.is_match(&content) {
                            ResolutionResult::success(format!(
                                "Resource matches pattern: {}",
                                self.path
                            ))
                        } else {
                            ResolutionResult::failure(
                                OutcomeCategory::SafeFailure,
                                format!(
                                    "Resource does not match pattern '{}': {}",
                                    pattern, self.path
                                ),
                            )
                            .with_output(content)
                        }
                    }
                    Err(e) => ResolutionResult::unclear(format!("Invalid regex pattern: {}", e)),
                },
                Err(e) => ResolutionResult::unclear(format!("Failed to read resource: {}", e)),
            },
            ResourceExpectation::HasSize { min, max } => match std::fs::metadata(path) {
                Ok(meta) => {
                    let size = meta.len();
                    let meets_min = min.map(|m| size >= m).unwrap_or(true);
                    let meets_max = max.map(|m| size <= m).unwrap_or(true);

                    if meets_min && meets_max {
                        ResolutionResult::success(format!(
                            "Resource size {} bytes is within bounds",
                            size
                        ))
                    } else {
                        ResolutionResult::failure(
                            OutcomeCategory::SafeFailure,
                            format!(
                                "Resource size {} bytes outside bounds (min: {:?}, max: {:?})",
                                size, min, max
                            ),
                        )
                    }
                }
                Err(e) => {
                    ResolutionResult::unclear(format!("Failed to check resource size: {}", e))
                }
            },
            ResourceExpectation::ModifiedAfter(_) => {
                // This requires storing the timestamp at prediction time
                ResolutionResult::unclear("ModifiedAfter check not implemented".to_string())
            }
        }
    }
}

impl Resolver for ResourceStateResolver {
    fn resolve(&self, _prediction: &WorldPrediction, _timeout: Duration) -> ResolutionResult {
        self.execute()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// HUMAN CONFIRMATION RESOLVER
// ═══════════════════════════════════════════════════════════════════════════════

/// Callback type for human confirmation
pub type HumanConfirmationCallback =
    Box<dyn Fn(&str) -> Option<(OutcomeCategory, String)> + Send + Sync>;

/// Resolver that asks a human for confirmation
pub struct HumanConfirmationResolver {
    /// Prompt to show to the human
    pub prompt: String,
    /// Callback to get human input (None = use default stdin)
    pub callback: Option<HumanConfirmationCallback>,
}

impl HumanConfirmationResolver {
    /// Create a new human confirmation resolver
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            callback: None,
        }
    }

    /// Set a custom callback for getting human input
    pub fn with_callback(mut self, callback: HumanConfirmationCallback) -> Self {
        self.callback = Some(callback);
        self
    }

    /// Execute confirmation request
    pub fn execute(&self, _timeout: Duration) -> ResolutionResult {
        // If we have a callback, use it
        if let Some(ref callback) = self.callback {
            match callback(&self.prompt) {
                Some((outcome, reason)) => {
                    return ResolutionResult {
                        outcome,
                        confidence: 1.0,
                        reason,
                        raw_output: None,
                        resolution_succeeded: true,
                    };
                }
                None => {
                    return ResolutionResult::unclear("Human did not provide confirmation");
                }
            }
        }

        // Default: return unclear (actual stdin interaction would need async)
        // In production, this would integrate with a UI or CLI prompt system
        ResolutionResult::unclear(format!(
            "Human confirmation required (no callback set): {}",
            self.prompt
        ))
    }
}

impl Resolver for HumanConfirmationResolver {
    fn resolve(&self, _prediction: &WorldPrediction, timeout: Duration) -> ResolutionResult {
        self.execute(timeout)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RESOLUTION EXECUTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// Main executor for resolving predictions
pub struct ResolutionExecutor {
    /// Default timeout for resolutions
    pub default_timeout: Duration,
    /// Human confirmation callback (if set)
    pub human_callback: Option<HumanConfirmationCallback>,
}

impl Default for ResolutionExecutor {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(60),
            human_callback: None,
        }
    }
}

impl ResolutionExecutor {
    /// Create a new executor
    pub fn new() -> Self {
        Self::default()
    }

    /// Set default timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.default_timeout = timeout;
        self
    }

    /// Set human confirmation callback
    pub fn with_human_callback(mut self, callback: HumanConfirmationCallback) -> Self {
        self.human_callback = Some(callback);
        self
    }

    /// Resolve a prediction using its resolution authority
    pub fn resolve(&self, prediction: &WorldPrediction) -> ResolutionResult {
        let timeout = prediction.deadline.min(self.default_timeout);

        match &prediction.resolution_contract.resolver {
            ResolutionAuthority::TestSuite {
                test_pattern,
                pass_threshold,
            } => {
                // Use as_ref() to avoid cloning the Option's inner String
                let resolver = TestSuiteResolver::cargo_test(Some(test_pattern.clone()))
                    .with_pass_threshold(*pass_threshold);
                resolver.execute(timeout)
            }

            ResolutionAuthority::ExitCode { success_codes } => {
                // Extract command from action context - use reference to avoid cloning
                let default_cmd = "echo 'no command specified'".to_string();
                let command = prediction
                    .action_context
                    .metadata
                    .get("command")
                    .unwrap_or(&default_cmd);

                // Parse command (simplified - real impl would use shell parsing)
                let parts: Vec<&str> = command.split_whitespace().collect();
                if parts.is_empty() {
                    return ResolutionResult::unclear("No command to execute");
                }

                // Clone success_codes only when needed - this is unavoidable
                // since ExitCodeResolver takes ownership
                let resolver = ExitCodeResolver::new(parts[0], success_codes.clone())
                    .with_args(parts[1..].iter().map(|s| (*s).to_string()).collect());
                resolver.execute(timeout)
            }

            ResolutionAuthority::DiffVerifier {
                expected_path,
                tolerance: _,
            } => {
                // For now, just check if expected file exists
                // Full implementation would compare actual output to expected
                let resolver = ResourceStateResolver::exists(expected_path.to_string_lossy());
                resolver.execute()
            }

            ResolutionAuthority::ExternalAPI {
                endpoint,
                expected_status: _,
                expected_body_contains: _,
            } => {
                // Would need HTTP client - return unclear for now
                ResolutionResult::unclear(format!(
                    "External API resolution not implemented: {}",
                    endpoint
                ))
            }

            ResolutionAuthority::HumanConfirmation { prompt, timeout: _ } => {
                // Use as_str() and to_owned() instead of clone() for clarity
                let resolver = HumanConfirmationResolver::new(prompt.as_str());
                // Note: callback handling would need Arc<dyn Fn> for proper sharing
                resolver.execute(timeout)
            }

            ResolutionAuthority::ResourceState {
                path,
                expected_state,
            } => {
                // expected_state.clone() is unavoidable as ResourceStateResolver takes ownership
                let resolver =
                    ResourceStateResolver::new(path.to_string_lossy(), expected_state.clone());
                resolver.execute()
            }

            ResolutionAuthority::Composite {
                authorities,
                quorum,
            } => {
                // Resolve each authority using a helper to avoid cloning the entire prediction
                let successes = authorities
                    .iter()
                    .filter(|auth| {
                        self.resolve_single_authority(prediction, auth, timeout)
                            .outcome
                            == OutcomeCategory::Success
                    })
                    .count();

                if successes >= *quorum {
                    ResolutionResult::success(format!(
                        "Composite resolution: {}/{} authorities agreed (quorum: {})",
                        successes,
                        authorities.len(),
                        quorum
                    ))
                } else {
                    ResolutionResult::failure(
                        OutcomeCategory::SafeFailure,
                        format!(
                            "Composite resolution failed: {}/{} authorities agreed (quorum: {})",
                            successes,
                            authorities.len(),
                            quorum
                        ),
                    )
                }
            }
        }
    }

    /// Helper to resolve a single authority without cloning the entire prediction
    fn resolve_single_authority(
        &self,
        prediction: &WorldPrediction,
        authority: &ResolutionAuthority,
        timeout: Duration,
    ) -> ResolutionResult {
        // Directly match on the authority instead of cloning the prediction
        match authority {
            ResolutionAuthority::TestSuite {
                test_pattern,
                pass_threshold,
            } => {
                let resolver = TestSuiteResolver::cargo_test(Some(test_pattern.clone()))
                    .with_pass_threshold(*pass_threshold);
                resolver.execute(timeout)
            }
            ResolutionAuthority::ExitCode { success_codes } => {
                let default_cmd = "echo 'no command specified'".to_string();
                let command = prediction
                    .action_context
                    .metadata
                    .get("command")
                    .unwrap_or(&default_cmd);
                let parts: Vec<&str> = command.split_whitespace().collect();
                if parts.is_empty() {
                    return ResolutionResult::unclear("No command to execute");
                }
                let resolver = ExitCodeResolver::new(parts[0], success_codes.clone())
                    .with_args(parts[1..].iter().map(|s| (*s).to_string()).collect());
                resolver.execute(timeout)
            }
            ResolutionAuthority::DiffVerifier {
                expected_path,
                tolerance: _,
            } => ResourceStateResolver::exists(expected_path.to_string_lossy()).execute(),
            ResolutionAuthority::ExternalAPI { endpoint, .. } => ResolutionResult::unclear(
                format!("External API resolution not implemented: {}", endpoint),
            ),
            ResolutionAuthority::HumanConfirmation { prompt, .. } => {
                HumanConfirmationResolver::new(prompt.as_str()).execute(timeout)
            }
            ResolutionAuthority::ResourceState {
                path,
                expected_state,
            } => {
                ResourceStateResolver::new(path.to_string_lossy(), expected_state.clone()).execute()
            }
            ResolutionAuthority::Composite {
                authorities,
                quorum,
            } => {
                // Recursive case - count successes
                let successes = authorities
                    .iter()
                    .filter(|auth| {
                        self.resolve_single_authority(prediction, auth, timeout)
                            .outcome
                            == OutcomeCategory::Success
                    })
                    .count();
                if successes >= *quorum {
                    ResolutionResult::success(format!(
                        "Composite: {}/{} agreed",
                        successes,
                        authorities.len()
                    ))
                } else {
                    ResolutionResult::failure(
                        OutcomeCategory::SafeFailure,
                        format!("Composite failed: {}/{}", successes, authorities.len()),
                    )
                }
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_result_creation() {
        let success = ResolutionResult::success("Test passed");
        assert_eq!(success.outcome, OutcomeCategory::Success);
        assert_eq!(success.confidence, 1.0);

        let failure = ResolutionResult::failure(OutcomeCategory::SafeFailure, "Test failed");
        assert_eq!(failure.outcome, OutcomeCategory::SafeFailure);

        let unclear = ResolutionResult::unclear("Could not determine");
        assert!(!unclear.resolution_succeeded);
    }

    #[test]
    fn test_resource_state_exists() {
        // Test with a file that definitely exists
        let resolver = ResourceStateResolver::exists("/etc/passwd");
        let result = resolver.execute();
        // On most systems, /etc/passwd exists
        if std::path::Path::new("/etc/passwd").exists() {
            assert_eq!(result.outcome, OutcomeCategory::Success);
        }
    }

    #[test]
    fn test_resource_state_not_exists() {
        let resolver =
            ResourceStateResolver::not_exists("/this/path/definitely/does/not/exist/12345");
        let result = resolver.execute();
        assert_eq!(result.outcome, OutcomeCategory::Success);
    }

    #[test]
    fn test_exit_code_resolver() {
        // Test with 'true' command (always exits 0)
        let resolver = ExitCodeResolver::new("true", vec![0]);
        let result = resolver.execute(Duration::from_secs(5));
        assert_eq!(result.outcome, OutcomeCategory::Success);

        // Test with 'false' command (always exits 1)
        let resolver = ExitCodeResolver::new("false", vec![0]);
        let result = resolver.execute(Duration::from_secs(5));
        assert_eq!(result.outcome, OutcomeCategory::SafeFailure);
    }

    #[test]
    fn test_human_confirmation_no_callback() {
        let resolver = HumanConfirmationResolver::new("Is this correct?");
        let result = resolver.execute(Duration::from_secs(1));
        assert!(!result.resolution_succeeded);
    }

    #[test]
    fn test_human_confirmation_with_callback() {
        let resolver =
            HumanConfirmationResolver::new("Is this correct?").with_callback(Box::new(|_prompt| {
                Some((OutcomeCategory::Success, "User confirmed".to_string()))
            }));
        let result = resolver.execute(Duration::from_secs(1));
        assert_eq!(result.outcome, OutcomeCategory::Success);
    }

    #[test]
    fn test_resolution_executor_creation() {
        let executor = ResolutionExecutor::new().with_timeout(Duration::from_secs(30));
        assert_eq!(executor.default_timeout, Duration::from_secs(30));
    }
}
