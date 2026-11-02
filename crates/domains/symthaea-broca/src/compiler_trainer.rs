// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compiler-Grounded Reward Training for Broca
//!
//! Uses the Nix parser/evaluator command as a narrow validation signal. A
//! successful parse/evaluation is not proof of task-level semantic correctness.
//! This module currently validates candidates; it does not generate repairs.

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
        let tmp_dir = std::env::temp_dir().join("symthaea_grounded_broca");
        if let Err(e) = std::fs::create_dir_all(&tmp_dir) {
            return CompilerVerdict::Fail {
                error: format!("IO error creating temporary directory: {e}"),
                stage: "setup",
            };
        }
        let tmp_path = tmp_dir.join(format!("test_{:?}.nix", md5::compute(code.as_bytes())));

        if let Err(e) = std::fs::write(&tmp_path, code) {
            return CompilerVerdict::Fail {
                error: format!("IO error writing candidate: {e}"),
                stage: "setup",
            };
        }

        let start = std::time::Instant::now();
        let output = Command::new("nix-instantiate")
            .arg("--eval")
            .arg("--parse")
            .arg(&tmp_path)
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
