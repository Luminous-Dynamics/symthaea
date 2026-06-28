//! Noise sweep utilities for reproducible binding probes.

use crate::benchmark::BindingProbeReport;
use crate::errors::{QuantumCompError, Result};
use crate::probe::{BindingProbeConfig, BindingProbeRunner};

/// Configuration for running a sequence of binding probes over increasing noise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoiseSweepConfig {
    /// Base binding probe configuration. Its `noise` field is ignored by the sweep.
    pub base: BindingProbeConfig,
    /// Number of noise levels to evaluate.
    pub steps: usize,
    /// Maximum noise value included in the sweep.
    pub max_noise: f32,
}

impl Default for NoiseSweepConfig {
    fn default() -> Self {
        Self {
            base: BindingProbeConfig::default(),
            steps: 6,
            max_noise: 0.25,
        }
    }
}

/// One point in a noise sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseSweepPoint {
    /// Noise value used for the point.
    pub noise: f32,
    /// Binding probe report at this noise value.
    pub report: BindingProbeReport,
}

/// Full sweep report.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseSweepReport {
    /// Sweep points in ascending noise order.
    pub points: Vec<NoiseSweepPoint>,
}

impl NoiseSweepReport {
    /// Returns a compact plain-text table.
    pub fn to_text_table(&self) -> String {
        let mut out = String::from(
            "noise,classical_noisy,phase_noisy,correlation_noisy,beta1_proxy,edge_density\n",
        );
        for point in &self.points {
            let r = &point.report.result;
            out.push_str(&format!(
                "{:.6},{:.6},{:.6},{:.6},{}, {:.6}\n",
                point.noise,
                r.classical_noisy_similarity,
                r.phase_noisy_similarity,
                r.correlation_noisy_similarity,
                r.beta1_proxy,
                r.topology_edge_density,
            ));
        }
        out
    }
}

/// Runs noise sweeps for binding probes.
#[derive(Debug, Clone)]
pub struct NoiseSweepRunner {
    config: NoiseSweepConfig,
}

impl NoiseSweepRunner {
    /// Creates a noise sweep runner.
    pub fn new(config: NoiseSweepConfig) -> Result<Self> {
        if config.steps == 0 {
            return Err(QuantumCompError::InvalidConfig("steps must be > 0"));
        }
        if !(0.0..=1.0).contains(&config.max_noise) {
            return Err(QuantumCompError::InvalidProbability);
        }
        BindingProbeRunner::new(config.base)?;
        Ok(Self { config })
    }

    /// Runs the sweep.
    pub fn run(&self) -> Result<NoiseSweepReport> {
        let mut points = Vec::with_capacity(self.config.steps);
        let denom = self.config.steps.saturating_sub(1).max(1) as f32;
        for i in 0..self.config.steps {
            let noise = if self.config.steps == 1 {
                self.config.max_noise
            } else {
                self.config.max_noise * (i as f32 / denom)
            };
            let mut cfg = self.config.base;
            cfg.noise = noise;
            let report = BindingProbeRunner::new(cfg)?.run()?;
            points.push(NoiseSweepPoint { noise, report });
        }
        Ok(NoiseSweepReport { points })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_sweep_runs() {
        let base = BindingProbeConfig {
            dimension: 128,
            trials: 4,
            noise: 0.0,
            seed: 1,
            topology_threshold: 0.55,
        };
        let report = NoiseSweepRunner::new(NoiseSweepConfig {
            base,
            steps: 3,
            max_noise: 0.2,
        })
        .unwrap()
        .run()
        .unwrap();
        assert_eq!(report.points.len(), 3);
        assert_eq!(report.points[0].noise, 0.0);
    }
}
