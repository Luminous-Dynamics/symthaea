//! Replicated comparison probes for binding methods.
//!
//! Alpha.4 adds replicated summaries so researchers can inspect run-to-run
//! stability instead of relying on a single aggregate probe.

use crate::errors::{QuantumCompError, Result};
use crate::probe::{BindingProbeConfig, BindingProbeRunner};
use crate::statistics::{SampleSummary, paired_effect_size};

/// Configuration for replicated comparison runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComparativeBindingConfig {
    /// Base binding probe configuration. Its seed is advanced per replicate.
    pub base: BindingProbeConfig,
    /// Number of independent replicated probe runs.
    pub replicates: usize,
    /// Deterministic stride added to the base seed per replicate.
    pub seed_stride: u64,
}

impl Default for ComparativeBindingConfig {
    fn default() -> Self {
        Self {
            base: BindingProbeConfig::default(),
            replicates: 8,
            seed_stride: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// Summary for one binding method across replicated probes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MethodComparisonSummary {
    /// Recovery similarity summary before noise.
    pub recovery: SampleSummary,
    /// Similarity summary after the configured noise process.
    pub noisy: SampleSummary,
}

/// Replicated comparison report.
#[derive(Debug, Clone, PartialEq)]
pub struct ComparativeBindingReport {
    /// Configuration used to produce the report.
    pub config: ComparativeBindingConfig,
    /// Classical XOR HDC method summary.
    pub classical: MethodComparisonSummary,
    /// Phase-HDC simulation method summary.
    pub phase: MethodComparisonSummary,
    /// Correlation-sketch method summary.
    pub correlation: MethodComparisonSummary,
    /// Paired effect size of classical noisy similarity minus phase noisy similarity.
    pub classical_minus_phase_noisy_dz: Option<f32>,
    /// Paired effect size of correlation noisy similarity minus classical noisy similarity.
    pub correlation_minus_classical_noisy_dz: Option<f32>,
}

impl ComparativeBindingReport {
    /// Returns a line-oriented text report.
    pub fn to_text(&self) -> String {
        fn fmt(label: &str, summary: &MethodComparisonSummary) -> String {
            let (r_lo, r_hi) = summary.recovery.approximate_95_ci();
            let (n_lo, n_hi) = summary.noisy.approximate_95_ci();
            format!(
                "{label}_recovery_mean={:.6} {label}_recovery_ci95=[{:.6},{:.6}] {label}_noisy_mean={:.6} {label}_noisy_ci95=[{:.6},{:.6}]",
                summary.recovery.mean, r_lo, r_hi, summary.noisy.mean, n_lo, n_hi,
            )
        }
        format!(
            "comparative-binding-v0.4\nreplicates={} dimension={} trials_per_replicate={} noise={} seed={}\n{}\n{}\n{}\nclassical_minus_phase_noisy_dz={:?}\ncorrelation_minus_classical_noisy_dz={:?}",
            self.config.replicates,
            self.config.base.dimension,
            self.config.base.trials,
            self.config.base.noise,
            self.config.base.seed,
            fmt("classical", &self.classical),
            fmt("phase", &self.phase),
            fmt("correlation", &self.correlation),
            self.classical_minus_phase_noisy_dz,
            self.correlation_minus_classical_noisy_dz,
        )
    }
}

/// Runs replicated binding comparisons.
#[derive(Debug, Clone)]
pub struct ComparativeBindingRunner {
    config: ComparativeBindingConfig,
}

impl ComparativeBindingRunner {
    /// Creates a replicated comparison runner.
    pub fn new(config: ComparativeBindingConfig) -> Result<Self> {
        if config.replicates == 0 {
            return Err(QuantumCompError::InvalidConfig("replicates must be > 0"));
        }
        BindingProbeRunner::new(config.base)?;
        Ok(Self { config })
    }

    /// Runs replicated binding comparisons.
    pub fn run(&self) -> Result<ComparativeBindingReport> {
        let mut classical_recovery = Vec::with_capacity(self.config.replicates);
        let mut classical_noisy = Vec::with_capacity(self.config.replicates);
        let mut phase_recovery = Vec::with_capacity(self.config.replicates);
        let mut phase_noisy = Vec::with_capacity(self.config.replicates);
        let mut correlation_recovery = Vec::with_capacity(self.config.replicates);
        let mut correlation_noisy = Vec::with_capacity(self.config.replicates);

        for replicate in 0..self.config.replicates {
            let mut cfg = self.config.base;
            cfg.seed = cfg
                .seed
                .wrapping_add((replicate as u64).wrapping_mul(self.config.seed_stride));
            let report = BindingProbeRunner::new(cfg)?.run()?;
            classical_recovery.push(report.result.classical_recovery_similarity);
            classical_noisy.push(report.result.classical_noisy_similarity);
            phase_recovery.push(report.result.phase_recovery_similarity);
            phase_noisy.push(report.result.phase_noisy_similarity);
            correlation_recovery.push(report.result.correlation_recovery_similarity);
            correlation_noisy.push(report.result.correlation_noisy_similarity);
        }

        let classical = MethodComparisonSummary {
            recovery: SampleSummary::from_samples(&classical_recovery).expect("nonempty by config"),
            noisy: SampleSummary::from_samples(&classical_noisy).expect("nonempty by config"),
        };
        let phase = MethodComparisonSummary {
            recovery: SampleSummary::from_samples(&phase_recovery).expect("nonempty by config"),
            noisy: SampleSummary::from_samples(&phase_noisy).expect("nonempty by config"),
        };
        let correlation = MethodComparisonSummary {
            recovery: SampleSummary::from_samples(&correlation_recovery)
                .expect("nonempty by config"),
            noisy: SampleSummary::from_samples(&correlation_noisy).expect("nonempty by config"),
        };

        Ok(ComparativeBindingReport {
            config: self.config,
            classical,
            phase,
            correlation,
            classical_minus_phase_noisy_dz: paired_effect_size(&classical_noisy, &phase_noisy),
            correlation_minus_classical_noisy_dz: paired_effect_size(
                &correlation_noisy,
                &classical_noisy,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparative_runner_reports_replicates() {
        let base = BindingProbeConfig {
            dimension: 128,
            trials: 4,
            noise: 0.05,
            seed: 5,
            topology_threshold: 0.55,
        };
        let report = ComparativeBindingRunner::new(ComparativeBindingConfig {
            base,
            replicates: 3,
            seed_stride: 99,
        })
        .unwrap()
        .run()
        .unwrap();
        assert_eq!(report.classical.noisy.count, 3);
    }
}
