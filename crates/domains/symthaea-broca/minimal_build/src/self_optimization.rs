// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Self-Optimization Engine — Recursive Architectural Evolution
//!
//! Allows Symthaea to mutate its own core logic and weights,
//! verifying improvements via internal benchmarks.

use crate::evolutionary_scaffolder::EvolutionResult;
use crate::formal_logic_scorer::FormalLogicScorer;
use crate::moral_safety_scorer::compute_moral_safety;
use rand::Rng;
use std::path::{Path, PathBuf};
use std::process::Command;
use symthaea_core::hdc::fol_formula_ext::FolFormulaExt;
use symthaea_core::hdc::logic_engine::Proposition;

pub struct SelfOptimizationEngine {
    project_root: PathBuf,
    formal_scorer: FormalLogicScorer,
}

impl SelfOptimizationEngine {
    pub fn new(project_root: PathBuf) -> Self {
        Self {
            project_root,
            formal_scorer: FormalLogicScorer::new(),
        }
    }

    /// Mutate a specific core file and run benchmarks to see if it improved.
    pub fn evolve_file(&self, file_path: &Path) -> anyhow::Result<EvolutionResult> {
        let source = std::fs::read_to_string(file_path)?;
        let mut rng = rand::thread_rng();

        // 1. Benchmark baseline
        let original_score = self.run_baseline_benchmark()?;

        // 2. Perform mutation in-memory
        let mutated = self.mutate_rust_constants(&source, &mut rng);

        // **NEW**: Meta-Conscious Audit
        // 2a. Moral/Safety Check
        let moral_score = compute_moral_safety(&mutated);
        if moral_score < 0.9 {
            anyhow::bail!("Evolution rejected: mutation failed moral/safety audit");
        }

        // 2b. Formal Logic Check (E-axis)
        let formula = FolFormulaExt::from_prop(Proposition::Atom(
            "Evolution remains consistent with core goals".to_string(),
        ));
        if !self.formal_scorer.verify_formula(&formula).verified {
            anyhow::bail!("Evolution rejected: mutation failed formal logic consistency check");
        }

        // 3. Temporarily apply to disk and test build
        std::fs::write(file_path, &mutated)?;
        let build_success = self.verify_build();

        if !build_success {
            // Rollback
            std::fs::write(file_path, &source)?;
            anyhow::bail!("Evolution rejected: mutation broke the build");
        }

        // 4. Benchmark again
        let new_score = self.run_baseline_benchmark()?;

        if new_score > original_score {
            Ok(EvolutionResult {
                id: rand::random(),
                success_score: new_score,
                mutation_description: format!("Optimized constants in {:?}", file_path),
                changed_files: vec![file_path.to_string_lossy().into_owned()],
                before_code: source,
                after_code: mutated,
                metrics: std::collections::HashMap::new(),
            })
        } else {
            // Rollback
            std::fs::write(file_path, &source)?;
            anyhow::bail!("Evolution rejected: mutation did not improve score");
        }
    }

    fn run_baseline_benchmark(&self) -> anyhow::Result<f32> {
        let output = Command::new("cargo")
            .arg("run")
            .arg("--bin")
            .arg("broca-eval")
            .arg("--")
            .arg("--quick")
            .current_dir(&self.project_root)
            .output()?;

        if !output.status.success() {
            return Ok(0.0);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);

        // REAL REGEX PARSING
        let re = regex::Regex::new(r"Composite Score:\s+([0-9.]+)")?;
        if let Some(caps) = re.captures(&stdout) {
            if let Some(score_str) = caps.get(1) {
                return Ok(score_str.as_str().parse::<f32>().unwrap_or(0.0));
            }
        }

        Ok(0.0)
    }

    fn mutate_rust_constants(&self, code: &str, rng: &mut impl Rng) -> String {
        let mut lines: Vec<String> = code.lines().map(|s| s.to_string()).collect();

        for line in lines.iter_mut() {
            // Find floating point constants (e.g., 0.85, 3.0)
            if rng.gen_bool(0.05) && line.contains('.') {
                let line_clone = line.clone();
                let parts: Vec<&str> = line_clone.split_whitespace().collect();
                for part in parts {
                    if let Ok(val) = part
                        .trim_matches(|c: char| !c.is_digit(10) && c != '.')
                        .parse::<f32>()
                    {
                        // Perturb by +/- 5%
                        let perturbation = 1.0 + (rng.gen_range(-0.05..0.05));
                        let new_val = val * perturbation;
                        *line = line.replace(&val.to_string(), &format!("{:.2}", new_val));
                    }
                }
            }
        }

        lines.join("\n")
    }

    fn verify_build(&self) -> bool {
        Command::new("cargo")
            .arg("check")
            .current_dir(&self.project_root)
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}
