//! Binding probe runner comparing classical and quantum-inspired HDC operations.

use crate::benchmark::{BenchmarkManifest, BenchmarkResult, BindingProbeReport};
use crate::classical_hdc::BinaryHypervector;
use crate::correlation_hdc::CorrelationBindingSketch;
use crate::errors::{QuantumCompError, Result};
use crate::phase_hdc::PhaseHypervector;
use crate::substrate::SubstrateProfile;
use crate::topology::threshold_graph_summary;

/// Configuration for the binding probe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BindingProbeConfig {
    /// Hypervector dimension.
    pub dimension: usize,
    /// Number of trials.
    pub trials: usize,
    /// Noise probability or phase sigma, depending on backend.
    pub noise: f32,
    /// Deterministic seed.
    pub seed: u64,
    /// Similarity threshold used for topology graph construction.
    pub topology_threshold: f32,
}

impl Default for BindingProbeConfig {
    fn default() -> Self {
        Self {
            dimension: 1024,
            trials: 16,
            noise: 0.05,
            seed: 0x5159_4D54_4841_4541,
            topology_threshold: 0.55,
        }
    }
}

/// Runs reproducible binding probes.
#[derive(Debug, Clone)]
pub struct BindingProbeRunner {
    config: BindingProbeConfig,
}

impl BindingProbeRunner {
    /// Creates a runner.
    pub fn new(config: BindingProbeConfig) -> Result<Self> {
        if config.dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        if config.trials == 0 {
            return Err(QuantumCompError::InvalidConfig("trials must be > 0"));
        }
        if !(0.0..=1.0).contains(&config.noise) || !(0.0..=1.0).contains(&config.topology_threshold)
        {
            return Err(QuantumCompError::InvalidProbability);
        }
        Ok(Self { config })
    }

    /// Returns the immutable run configuration.
    pub fn config(&self) -> BindingProbeConfig {
        self.config
    }

    /// Runs the comparison benchmark.
    pub fn run(&self) -> Result<BindingProbeReport> {
        let mut classical_recovery = 0.0f32;
        let mut phase_recovery = 0.0f32;
        let mut correlation_recovery = 0.0f32;
        let mut classical_noisy = 0.0f32;
        let mut phase_noisy = 0.0f32;
        let mut correlation_noisy = 0.0f32;
        let mut sim_matrix = vec![vec![0.0f32; self.config.trials]; self.config.trials];
        let mut recovered_classical = Vec::with_capacity(self.config.trials);

        for trial in 0..self.config.trials {
            let seed = self
                .config
                .seed
                .wrapping_add((trial as u64).wrapping_mul(0x9E37_79B9));
            let item = BinaryHypervector::random(self.config.dimension, seed ^ 0xA11CE)?;
            let key = BinaryHypervector::random(self.config.dimension, seed ^ 0xB0B)?;
            let bound = item.bind_xor(&key)?;
            let recovered = bound.unbind_xor(&key)?;
            classical_recovery += item.similarity(&recovered)?;
            let noisy = recovered.with_bitflip_noise(self.config.noise, seed ^ 0xD15EA5E);
            classical_noisy += item.similarity(&noisy)?;
            recovered_classical.push(noisy.clone());

            let phase_item = PhaseHypervector::from_binary(&item);
            let phase_key = PhaseHypervector::from_binary(&key);
            let phase_bound = phase_item.bind_phase(&phase_key)?;
            let phase_recovered = phase_bound.unbind_phase(&phase_key)?;
            phase_recovery += phase_item.circular_similarity(&phase_recovered)?;
            let phase_noisy_hv = phase_recovered
                .with_phase_noise(self.config.noise * core::f32::consts::PI, seed ^ 0xF00D);
            phase_noisy += phase_item.circular_similarity(&phase_noisy_hv)?;

            let sketch = CorrelationBindingSketch::bind(&item, &key)?;
            let correlation_recovered = sketch.recover_item(&key)?;
            correlation_recovery += item.similarity(&correlation_recovered)?;
            let correlation_noisy_hv =
                correlation_recovered.with_bitflip_noise(self.config.noise, seed ^ 0xC0FFEE);
            correlation_noisy += item.similarity(&correlation_noisy_hv)?;
        }

        for i in 0..self.config.trials {
            for j in 0..self.config.trials {
                sim_matrix[i][j] = recovered_classical[i].similarity(&recovered_classical[j])?;
            }
        }
        let topology = threshold_graph_summary(&sim_matrix, self.config.topology_threshold)?;

        let denom = self.config.trials as f32;
        let result = BenchmarkResult {
            classical_recovery_similarity: classical_recovery / denom,
            phase_recovery_similarity: phase_recovery / denom,
            correlation_recovery_similarity: correlation_recovery / denom,
            classical_noisy_similarity: classical_noisy / denom,
            phase_noisy_similarity: phase_noisy / denom,
            correlation_noisy_similarity: correlation_noisy / denom,
            beta1_proxy: topology.beta1_proxy,
            topology_edge_density: topology.edge_density,
            topology_mean_degree: topology.mean_degree,
        };

        let manifest = BenchmarkManifest {
            name: "classical-vs-phase-vs-correlation-binding-probe-v0.2".to_string(),
            dimension: self.config.dimension,
            trials: self.config.trials,
            noise: self.config.noise,
            seed: self.config.seed,
            topology_threshold: self.config.topology_threshold,
            substrate: SubstrateProfile::quantum_inspired(),
        };

        Ok(BindingProbeReport { manifest, result })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_runs() {
        let runner = BindingProbeRunner::new(BindingProbeConfig {
            dimension: 256,
            trials: 4,
            noise: 0.05,
            seed: 42,
            topology_threshold: 0.55,
        })
        .unwrap();
        let report = runner.run().unwrap();
        assert!(report.result.classical_recovery_similarity > 0.99);
        assert!(report.result.phase_recovery_similarity > 0.99);
        assert!(report.result.correlation_recovery_similarity > 0.99);
    }
}
