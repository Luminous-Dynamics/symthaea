// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compiler-Grounded Reward Training for Broca
//!
//! Uses the Nix parser/evaluator command as a narrow validation signal. A
//! successful parse/evaluation is not proof of task-level semantic correctness.
//! This module currently validates candidates; it does not generate repairs.

use std::io::Write;
use std::process::Command;

#[derive(Debug, Clone)]
pub enum CompilerVerdict {
    /// The configured nix-instantiate command completed successfully.
    ///
    /// This is a syntax/evaluation-stage signal, not a behavioral correctness
    /// proof for the user's requested task.
    Pass { eval_time_ms: u64 },
    Fail { error: String, stage: &'static str },
}

#[derive(Debug, Clone)]
pub struct TrainingResult {
    /// Reward derived solely from the validation verdict.
    pub reward: f32,
    /// Number of distinct candidates validated (currently one).
    pub steps: usize,
    /// Whether the configured validation command accepted the candidate.
    pub converged: bool,
    pub final_code: String,
}

pub struct CompilerGroundedTrainer;

impl CompilerGroundedTrainer {
    pub fn new() -> Self {
        Self
    }

    /// Run nix-instantiate to check the candidate with the configured Nix
    /// parse/evaluation command. This does not run task-specific tests.
    pub fn verify_nix(&self, code: &str) -> CompilerVerdict {
        // Use an atomically created, uniquely named file rather than a
        // predictable candidate-derived path under a shared temporary directory.
        // Keep the handle alive until the child process has finished; Drop then
        // removes the file on both success and failure paths.
        let candidate_file = match create_nix_temp_file(code) {
            Ok(file) => file,
            Err(e) => {
                return CompilerVerdict::Fail {
                    error: format!("IO error creating candidate file: {e}"),
                    stage: "setup",
                };
            }
        };

        let start = std::time::Instant::now();
        let output = Command::new("nix-instantiate")
            .arg("--eval")
            .arg("--parse")
            .arg(candidate_file.path())
            .output();

        let duration = start.elapsed();

        match output {
            Ok(out) if out.status.success() => CompilerVerdict::Pass {
                eval_time_ms: duration.as_millis() as u64,
            },
            Ok(out) => CompilerVerdict::Fail {
                error: String::from_utf8_lossy(&out.stderr).to_string(),
                stage: "nix-instantiate",
            },
            Err(e) => CompilerVerdict::Fail {
                error: format!("Process error: {e}"),
                stage: "spawn",
            },
        }
    }

    /// Perform one validation step using the external Nix command.
    ///
    /// This API is retained for callers that consume TrainingResult, but no
    /// repair is attempted: the same immutable source is never re-verified
    /// and structural self-similarity is never treated as a success signal.
    pub fn train_step(&self, _prompt: &str, initial_code: &str) -> TrainingResult {
        let code = initial_code.to_string();
        let verdict = self.verify_nix(&code);
        let reward = reward_for_verdict(&verdict);
        let converged = matches!(verdict, CompilerVerdict::Pass { .. });

        TrainingResult {
            reward,
            steps: 1,
            converged,
            final_code: code,
        }
    }
}

/// Create a uniquely named Nix candidate file with tempfile's exclusive
/// creation and restrictive default permissions. The caller owns the handle
/// and must keep it alive until the child process no longer needs the path.
fn create_nix_temp_file(code: &str) -> std::io::Result<tempfile::NamedTempFile> {
    let mut file = tempfile::Builder::new()
        .prefix("symthaea-grounded-broca-")
        .suffix(".nix")
        .tempfile()?;
    file.write_all(code.as_bytes())?;
    file.flush()?;
    Ok(file)
}

/// Convert a verifier verdict into a reward without consulting candidate
/// self-similarity or any other non-independent proxy.
fn reward_for_verdict(verdict: &CompilerVerdict) -> f32 {
    match verdict {
        CompilerVerdict::Pass { .. } => 1.0,
        CompilerVerdict::Fail { .. } => 0.0,
    }
}

impl Default for CompilerGroundedTrainer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{CompilerVerdict, reward_for_verdict};

    #[test]
    fn accepted_validation_receives_reward() {
        let verdict = CompilerVerdict::Pass { eval_time_ms: 2 };
        assert_eq!(reward_for_verdict(&verdict), 1.0);
    }

    #[test]
    fn rejected_candidate_never_receives_positive_reward() {
        let verdict = CompilerVerdict::Fail {
            error: "parse failed".to_string(),
            stage: "nix-instantiate",
        };
        assert_eq!(reward_for_verdict(&verdict), 0.0);
    }

    #[test]
    fn spawn_failure_never_receives_positive_reward() {
        let verdict = CompilerVerdict::Fail {
            error: "nix-instantiate not found".to_string(),
            stage: "spawn",
        };
        assert_eq!(reward_for_verdict(&verdict), 0.0);
    }
}


#[cfg(test)]
mod temporary_file_tests {
    use super::create_nix_temp_file;

    #[test]
    fn candidate_files_are_unique_even_for_identical_source() {
        let first = create_nix_temp_file("{ value = 1; }").expect("first temp file");
        let second = create_nix_temp_file("{ value = 1; }").expect("second temp file");

        assert_ne!(first.path(), second.path());
    }

    #[test]
    fn candidate_contents_are_written_before_verification() {
        let source = "{ value = 42; }";
        let candidate = create_nix_temp_file(source).expect("candidate temp file");

        let actual = std::fs::read_to_string(candidate.path()).expect("read candidate");
        assert_eq!(actual, source);
    }

    #[test]
    fn candidate_file_is_removed_when_handle_is_dropped() {
        let path = {
            let candidate = create_nix_temp_file("{ value = 7; }").expect("candidate temp file");
            let path = candidate.path().to_path_buf();
            assert!(path.exists());
            path
        };

        assert!(!path.exists(), "temporary candidate must be removed on drop");
    }
}
