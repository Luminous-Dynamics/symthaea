// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Counterfactual branching: "what if" analysis via simulation forking.
//!
//! Run a simulation, save a checkpoint at a specific point, then branch
//! with different parameters to compare alternative histories.
//!
//! Example: "What if the Carrington event at year 45 hadn't happened?"
//! → Fork at year 44, run one branch with disasters, one without.
//! → Compare CVS, population, harmony trajectories.
//!
//! This transforms the simulator from a single-trajectory tool into
//! a counterfactual reasoning engine.

use serde::{Deserialize, Serialize};

/// A simulation checkpoint: serializable state at a specific tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimCheckpoint {
    /// Tick at which this checkpoint was taken.
    pub tick: u32,
    /// Year (tick / 12).
    pub year: f64,
    /// RNG state (seed + tick offset for reproducibility).
    pub rng_seed: u64,
    pub rng_offset: u32,
    /// Key metrics at checkpoint time.
    pub cvs: f64,
    pub population: usize,
    pub collective_phi: f64,
    pub harmony_scores: [f64; 8],
    pub total_disasters: u32,
    /// Description of why this checkpoint was taken.
    pub description: String,
}

/// A counterfactual branch: a fork with modified parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualBranch {
    /// Name of this branch (e.g., "No Carrington Event").
    pub name: String,
    /// What was changed from the baseline.
    pub modification: String,
    /// Final CVS of this branch.
    pub final_cvs: f64,
    /// Final population.
    pub final_population: usize,
    /// Key trajectory points for comparison.
    pub trajectory: Vec<TrajectoryPoint>,
}

/// A trajectory point for comparison plotting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryPoint {
    pub tick: u32,
    pub year: f64,
    pub cvs: f64,
    pub population: usize,
    pub collective_phi: f64,
}

/// Result of a counterfactual comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualComparison {
    /// The checkpoint where the fork happened.
    pub fork_point: SimCheckpoint,
    /// The baseline branch (original parameters).
    pub baseline: CounterfactualBranch,
    /// Alternative branches (modified parameters).
    pub alternatives: Vec<CounterfactualBranch>,
    /// Analysis: which branch performed best?
    pub analysis: ComparisonAnalysis,
}

/// Analysis of counterfactual comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonAnalysis {
    /// Which branch had the highest final CVS.
    pub best_branch: String,
    /// CVS delta between best and worst branches.
    pub cvs_spread: f64,
    /// Population delta between best and worst.
    pub population_spread: i64,
    /// Whether the fork event was material (>5% CVS difference).
    pub fork_is_material: bool,
    /// Human-readable summary.
    pub summary: String,
}

impl CounterfactualComparison {
    /// Analyze the comparison results.
    pub fn analyze(mut self) -> Self {
        let all_branches: Vec<&CounterfactualBranch> = std::iter::once(&self.baseline)
            .chain(self.alternatives.iter())
            .collect();

        let best = all_branches
            .iter()
            .max_by(|a, b| a.final_cvs.partial_cmp(&b.final_cvs).unwrap())
            .unwrap();
        let worst = all_branches
            .iter()
            .min_by(|a, b| a.final_cvs.partial_cmp(&b.final_cvs).unwrap())
            .unwrap();

        let cvs_spread = best.final_cvs - worst.final_cvs;
        let pop_spread = best.final_population as i64 - worst.final_population as i64;

        self.analysis = ComparisonAnalysis {
            best_branch: best.name.clone(),
            cvs_spread,
            population_spread: pop_spread,
            fork_is_material: cvs_spread > 0.05,
            summary: format!(
                "Best: {} (CVS {:.4}), Worst: {} (CVS {:.4}). Delta: {:.4} ({})",
                best.name,
                best.final_cvs,
                worst.name,
                worst.final_cvs,
                cvs_spread,
                if cvs_spread > 0.05 {
                    "MATERIAL"
                } else {
                    "not material"
                }
            ),
        };
        self
    }

    /// Format as Markdown.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str("# Counterfactual Analysis\n\n");
        md.push_str(&format!(
            "**Fork Point**: Year {:.1} (tick {})\n\n",
            self.fork_point.year, self.fork_point.tick
        ));

        md.push_str("## Branches\n\n");
        md.push_str("| Branch | CVS | Population | Modification |\n");
        md.push_str("|--------|-----|------------|-------------|\n");
        md.push_str(&format!(
            "| {} | {:.4} | {} | Baseline |\n",
            self.baseline.name, self.baseline.final_cvs, self.baseline.final_population
        ));
        for alt in &self.alternatives {
            md.push_str(&format!(
                "| {} | {:.4} | {} | {} |\n",
                alt.name, alt.final_cvs, alt.final_population, alt.modification
            ));
        }

        md.push_str(&format!("\n## Analysis\n\n{}\n", self.analysis.summary));
        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_analysis() {
        let comparison = CounterfactualComparison {
            fork_point: SimCheckpoint {
                tick: 540,
                year: 45.0,
                rng_seed: 42,
                rng_offset: 540,
                cvs: 0.7,
                population: 15000,
                collective_phi: 0.5,
                harmony_scores: [0.5; 8],
                total_disasters: 100,
                description: "Before Carrington event".into(),
            },
            baseline: CounterfactualBranch {
                name: "With Carrington".into(),
                modification: "Standard disaster engine".into(),
                final_cvs: 0.65,
                final_population: 18000,
                trajectory: vec![],
            },
            alternatives: vec![CounterfactualBranch {
                name: "Without Carrington".into(),
                modification: "Carrington event disabled".into(),
                final_cvs: 0.72,
                final_population: 20000,
                trajectory: vec![],
            }],
            analysis: ComparisonAnalysis {
                best_branch: String::new(),
                cvs_spread: 0.0,
                population_spread: 0,
                fork_is_material: false,
                summary: String::new(),
            },
        }
        .analyze();

        assert_eq!(comparison.analysis.best_branch, "Without Carrington");
        assert!(comparison.analysis.cvs_spread > 0.05);
        assert!(comparison.analysis.fork_is_material);
        assert!(comparison.analysis.summary.contains("MATERIAL"));
    }

    #[test]
    fn test_markdown_output() {
        let comparison = CounterfactualComparison {
            fork_point: SimCheckpoint {
                tick: 540,
                year: 45.0,
                rng_seed: 42,
                rng_offset: 540,
                cvs: 0.7,
                population: 15000,
                collective_phi: 0.5,
                harmony_scores: [0.5; 8],
                total_disasters: 100,
                description: "Test fork".into(),
            },
            baseline: CounterfactualBranch {
                name: "Baseline".into(),
                modification: "None".into(),
                final_cvs: 0.65,
                final_population: 18000,
                trajectory: vec![],
            },
            alternatives: vec![],
            analysis: ComparisonAnalysis {
                best_branch: "Baseline".into(),
                cvs_spread: 0.0,
                population_spread: 0,
                fork_is_material: false,
                summary: "Single branch".into(),
            },
        };

        let md = comparison.to_markdown();
        assert!(md.contains("Counterfactual Analysis"));
        assert!(md.contains("Year 45.0"));
        assert!(md.contains("Baseline"));
    }
}
