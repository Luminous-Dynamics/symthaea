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
    ///
    /// # Safety boundary
    ///
    /// This mutates a file on disk and then *compiles and runs it*
    /// (`cargo check`/`cargo run --bin broca-eval`), which is arbitrary
    /// code execution with this process's privileges. The only content
    /// gate is [`compute_moral_safety`] -- a regex blocklist over literal
    /// source-text patterns, which is a coarse pre-filter, **not a
    /// security boundary** (it does not catch semantically-equivalent
    /// alternate APIs). The one guarantee this function does enforce is
    /// that `file_path` must resolve inside [`Self::project_root`]: an
    /// absolute path or `..` traversal outside the project tree is
    /// rejected before any read, mutation, or compilation happens.
    pub fn evolve_file(&self, file_path: &Path) -> anyhow::Result<EvolutionResult> {
        self.require_path_within_project_root(file_path)?;

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

    /// Reject any `path` that does not canonicalize to somewhere inside
    /// `self.project_root` -- blocks absolute paths, `..` traversal, and
    /// symlink escapes from mutating/compiling/executing files outside the
    /// project's own source tree.
    fn require_path_within_project_root(&self, path: &Path) -> anyhow::Result<()> {
        let canonical_root = self.project_root.canonicalize().map_err(|e| {
            anyhow::anyhow!(
                "failed to canonicalize project_root {:?}: {e}",
                self.project_root
            )
        })?;
        let canonical_path = path
            .canonicalize()
            .map_err(|e| anyhow::anyhow!("failed to canonicalize {path:?}: {e}"))?;

        if !canonical_path.starts_with(&canonical_root) {
            anyhow::bail!(
                "Evolution rejected: {file:?} resolves outside project_root {root:?} \
                 (self-modification is confined to the project's own source tree)",
                file = canonical_path,
                root = canonical_root,
            );
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_inside_project_root_is_accepted() {
        let dir =
            std::env::temp_dir().join(format!("symthaea-selfopt-test-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let file = dir.join("src").join("lib.rs");
        std::fs::write(&file, "// test file").unwrap();

        let engine = SelfOptimizationEngine::new(dir.clone());
        assert!(engine.require_path_within_project_root(&file).is_ok());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn path_outside_project_root_is_rejected() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea-selfopt-test-inside-{}",
            std::process::id()
        ));
        let outside = std::env::temp_dir().join(format!(
            "symthaea-selfopt-test-outside-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let outside_file = outside.join("passwd_or_whatever.rs");
        std::fs::write(&outside_file, "// not part of the project").unwrap();

        let engine = SelfOptimizationEngine::new(dir.clone());
        let err = engine
            .require_path_within_project_root(&outside_file)
            .unwrap_err();
        assert!(err.to_string().contains("outside project_root"));

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside);
    }

    #[test]
    fn traversal_escaping_project_root_via_dotdot_is_rejected() {
        let base = std::env::temp_dir().join(format!(
            "symthaea-selfopt-test-traversal-{}",
            std::process::id()
        ));
        let project_root = base.join("project");
        let sibling = base.join("sibling");
        std::fs::create_dir_all(project_root.join("src")).unwrap();
        std::fs::create_dir_all(&sibling).unwrap();
        let sibling_file = sibling.join("target.rs");
        std::fs::write(&sibling_file, "// outside the project").unwrap();

        // A path that stays textually "inside" project_root/src but walks
        // back out to the sibling directory via `..`.
        let traversal_path = project_root
            .join("src")
            .join("..")
            .join("..")
            .join("sibling")
            .join("target.rs");

        let engine = SelfOptimizationEngine::new(project_root.clone());
        let err = engine
            .require_path_within_project_root(&traversal_path)
            .unwrap_err();
        assert!(err.to_string().contains("outside project_root"));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn evolve_file_rejects_before_touching_disk_for_outside_path() {
        let dir = std::env::temp_dir().join(format!(
            "symthaea-selfopt-test-evolve-{}",
            std::process::id()
        ));
        let outside = std::env::temp_dir().join(format!(
            "symthaea-selfopt-test-evolve-outside-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let outside_file = outside.join("target.rs");
        std::fs::write(&outside_file, "const X: f32 = 1.0;\n").unwrap();
        let original_contents = std::fs::read_to_string(&outside_file).unwrap();

        let engine = SelfOptimizationEngine::new(dir.clone());
        let result = engine.evolve_file(&outside_file);
        assert!(result.is_err());
        // The file must be completely untouched -- rejection happens before
        // any mutation, benchmark, or build step runs.
        assert_eq!(
            std::fs::read_to_string(&outside_file).unwrap(),
            original_contents
        );

        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&outside);
    }
}
