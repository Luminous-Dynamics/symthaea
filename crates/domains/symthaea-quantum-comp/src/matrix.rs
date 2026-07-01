//! Experiment matrix runner for replicated binding comparisons.
//!
//! Alpha.6 adds a small grid runner so researchers can evaluate the same
//! protocol over dimensions and noise levels with consistent seeds.

use crate::comparative::{
    ComparativeBindingConfig, ComparativeBindingReport, ComparativeBindingRunner,
};
use crate::errors::{QuantumCompError, Result};
use crate::probe::BindingProbeConfig;

/// Configuration for a dimension-by-noise experiment matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentMatrixConfig {
    /// Hypervector dimensions to evaluate.
    pub dimensions: Vec<usize>,
    /// Noise levels to evaluate.
    pub noise_levels: Vec<f32>,
    /// Trials per replicate.
    pub trials: usize,
    /// Replicates per matrix cell.
    pub replicates: usize,
    /// Base deterministic seed.
    pub seed: u64,
    /// Seed stride between replicates.
    pub seed_stride: u64,
    /// Similarity threshold used for topology summaries.
    pub topology_threshold: f32,
}

impl Default for ExperimentMatrixConfig {
    fn default() -> Self {
        Self {
            dimensions: vec![256, 512, 1024],
            noise_levels: vec![0.0, 0.05, 0.10, 0.20],
            trials: 8,
            replicates: 4,
            seed: 0xA16A_0006,
            seed_stride: 0x9E37_79B9_7F4A_7C15,
            topology_threshold: 0.55,
        }
    }
}

/// One cell in the experiment matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentMatrixCell {
    /// Hypervector dimension for the cell.
    pub dimension: usize,
    /// Noise level for the cell.
    pub noise: f32,
    /// Replicated comparison report for the cell.
    pub report: ComparativeBindingReport,
}

/// Complete matrix report.
#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentMatrixReport {
    /// Matrix configuration.
    pub config: ExperimentMatrixConfig,
    /// Matrix cells in dimension-major, noise-minor order.
    pub cells: Vec<ExperimentMatrixCell>,
}

impl ExperimentMatrixReport {
    /// Returns a CSV table of noisy similarity means across methods.
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "dimension,noise,replicates,classical_noisy_mean,phase_noisy_mean,correlation_noisy_mean,classical_phase_dz,correlation_classical_dz\n",
        );
        for cell in &self.cells {
            out.push_str(&format!(
                "{},{},{},{},{},{},{:?},{:?}\n",
                cell.dimension,
                cell.noise,
                cell.report.config.replicates,
                cell.report.classical.noisy.mean,
                cell.report.phase.noisy.mean,
                cell.report.correlation.noisy.mean,
                cell.report.classical_minus_phase_noisy_dz,
                cell.report.correlation_minus_classical_noisy_dz,
            ));
        }
        out
    }

    /// Returns a Markdown table of noisy similarity means across methods.
    pub fn to_markdown(&self) -> String {
        let mut out = String::from(
            "# Experiment Matrix Report\n\n| Dimension | Noise | Replicates | Classical noisy | Phase noisy | Correlation noisy | Classical-phase dz | Correlation-classical dz |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        );
        for cell in &self.cells {
            out.push_str(&format!(
                "| {} | {:.6} | {} | {:.6} | {:.6} | {:.6} | {:?} | {:?} |\n",
                cell.dimension,
                cell.noise,
                cell.report.config.replicates,
                cell.report.classical.noisy.mean,
                cell.report.phase.noisy.mean,
                cell.report.correlation.noisy.mean,
                cell.report.classical_minus_phase_noisy_dz,
                cell.report.correlation_minus_classical_noisy_dz,
            ));
        }
        out
    }
}

/// Runs an experiment matrix.
#[derive(Debug, Clone)]
pub struct ExperimentMatrixRunner {
    config: ExperimentMatrixConfig,
}

impl ExperimentMatrixRunner {
    /// Creates a matrix runner after validating configuration.
    pub fn new(config: ExperimentMatrixConfig) -> Result<Self> {
        if config.dimensions.is_empty() {
            return Err(QuantumCompError::InvalidConfig(
                "dimensions must be nonempty",
            ));
        }
        if config.noise_levels.is_empty() {
            return Err(QuantumCompError::InvalidConfig(
                "noise_levels must be nonempty",
            ));
        }
        if config.trials == 0 {
            return Err(QuantumCompError::InvalidConfig("trials must be > 0"));
        }
        if config.replicates == 0 {
            return Err(QuantumCompError::InvalidConfig("replicates must be > 0"));
        }
        if !(0.0..=1.0).contains(&config.topology_threshold) {
            return Err(QuantumCompError::InvalidProbability);
        }
        for &dimension in &config.dimensions {
            if dimension == 0 {
                return Err(QuantumCompError::InvalidDimension);
            }
        }
        for &noise in &config.noise_levels {
            if !(0.0..=1.0).contains(&noise) {
                return Err(QuantumCompError::InvalidProbability);
            }
        }
        Ok(Self { config })
    }

    /// Returns the matrix configuration.
    pub fn config(&self) -> &ExperimentMatrixConfig {
        &self.config
    }

    /// Runs the full matrix.
    pub fn run(&self) -> Result<ExperimentMatrixReport> {
        let mut cells =
            Vec::with_capacity(self.config.dimensions.len() * self.config.noise_levels.len());
        for (dimension_index, &dimension) in self.config.dimensions.iter().enumerate() {
            for (noise_index, &noise) in self.config.noise_levels.iter().enumerate() {
                let seed = self
                    .config
                    .seed
                    .wrapping_add((dimension_index as u64).wrapping_mul(0xD1B5_4A32_D192_ED03))
                    .wrapping_add((noise_index as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
                let base = BindingProbeConfig {
                    dimension,
                    trials: self.config.trials,
                    noise,
                    seed,
                    topology_threshold: self.config.topology_threshold,
                };
                let comparison = ComparativeBindingRunner::new(ComparativeBindingConfig {
                    base,
                    replicates: self.config.replicates,
                    seed_stride: self.config.seed_stride,
                })?
                .run()?;
                cells.push(ExperimentMatrixCell {
                    dimension,
                    noise,
                    report: comparison,
                });
            }
        }
        Ok(ExperimentMatrixReport {
            config: self.config.clone(),
            cells,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_runner_builds_cells() {
        let cfg = ExperimentMatrixConfig {
            dimensions: vec![64, 128],
            noise_levels: vec![0.0, 0.1],
            trials: 2,
            replicates: 2,
            seed: 9,
            seed_stride: 17,
            topology_threshold: 0.55,
        };
        let report = ExperimentMatrixRunner::new(cfg).unwrap().run().unwrap();
        assert_eq!(report.cells.len(), 4);
        assert!(report.to_csv().contains("dimension,noise"));
    }
}
