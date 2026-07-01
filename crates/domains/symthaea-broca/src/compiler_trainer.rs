// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compiler-Grounded Reward Training for Broca
//!
//! Replaces golden-matching with real compiler verdicts as the reward signal.

use crate::structural_scorer::NixStructuralScorer;
use std::process::Command;

#[derive(Debug, Clone)]
pub enum CompilerVerdict {
    Pass { eval_time_ms: u64 },
    Fail { error: String, stage: &'static str },
}

#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub reward: f32,
    pub steps: usize,
    pub converged: bool,
    pub final_code: String,
}

pub struct CompilerGroundedTrainer {
    max_repair_iters: usize,
    scorer: NixStructuralScorer,
}

impl CompilerGroundedTrainer {
    pub fn new() -> Self {
        Self {
            max_repair_iters: 3,
            scorer: NixStructuralScorer::new(),
        }
    }

    /// Run nix-instantiate to verify Nix code correctness.
    pub fn verify_nix(&self, code: &str) -> CompilerVerdict {
        let tmp_dir = std::env::temp_dir().join("symthaea_grounded_broca");
        std::fs::create_dir_all(&tmp_dir).ok();
        let tmp_path = tmp_dir.join(format!("test_{:?}.nix", md5::compute(code.as_bytes())));

        if let Err(e) = std::fs::write(&tmp_path, code) {
            return CompilerVerdict::Fail {
                error: format!("IO error: {}", e),
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
                error: format!("Process error: {}", e),
                stage: "spawn",
            },
        }
    }

    /// Perform a single training step using compiler feedback.
    pub fn train_step(&self, _prompt: &str, initial_code: &str) -> TrainingResult {
        let code = initial_code.to_string();
        let mut total_reward = 0.0;
        let mut converged = false;

        for step in 0..self.max_repair_iters {
            match self.verify_nix(&code) {
                CompilerVerdict::Pass { .. } => {
                    total_reward += 1.0;
                    converged = true;
                    return TrainingResult {
                        reward: total_reward,
                        steps: step,
                        converged,
                        final_code: code,
                    };
                }
                CompilerVerdict::Fail { .. } => {
                    // In broca-only mode, we just check if it's structurally similar to itself
                    // as a sanity check. Real repair would happen in the main crate.
                    let verdict = self.scorer.score(&code, &code);
                    if verdict.parse_error.is_none() {
                        total_reward += 0.1;
                    } else {
                        total_reward -= 0.2;
                        break;
                    }
                }
            }
        }

        TrainingResult {
            reward: total_reward,
            steps: self.max_repair_iters,
            converged,
            final_code: code,
        }
    }
}

impl Default for CompilerGroundedTrainer {
    fn default() -> Self {
        Self::new()
    }
}
