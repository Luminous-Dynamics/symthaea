//! Benchmark manifests and reports.

use crate::substrate::SubstrateProfile;

/// Reproducible manifest for one benchmark run.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkManifest {
    /// Benchmark name.
    pub name: String,
    /// Bit or phase dimension.
    pub dimension: usize,
    /// Number of independent trials.
    pub trials: usize,
    /// Noise level used by the benchmark.
    pub noise: f32,
    /// Deterministic seed.
    pub seed: u64,
    /// Threshold used for topology graph construction.
    pub topology_threshold: f32,
    /// Substrate profile.
    pub substrate: SubstrateProfile,
}

impl BenchmarkManifest {
    /// Returns a deterministic, non-cryptographic run fingerprint.
    ///
    /// This is only a convenience hash for reports. It is not a security hash and
    /// should not be used as a Mycelix receipt or artifact commitment.
    pub fn reproducibility_fingerprint(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        fn mix(h: &mut u64, bytes: &[u8]) {
            for b in bytes {
                *h ^= *b as u64;
                *h = h.wrapping_mul(0x0000_0100_0000_01B3);
            }
        }
        mix(&mut h, self.name.as_bytes());
        mix(&mut h, &self.dimension.to_le_bytes());
        mix(&mut h, &self.trials.to_le_bytes());
        mix(&mut h, &self.noise.to_bits().to_le_bytes());
        mix(&mut h, &self.seed.to_le_bytes());
        mix(&mut h, &self.topology_threshold.to_bits().to_le_bytes());
        mix(&mut h, self.substrate.backend_name.as_bytes());
        h
    }
}

/// Numeric result for a benchmark run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BenchmarkResult {
    /// Mean classical recovery similarity.
    pub classical_recovery_similarity: f32,
    /// Mean phase recovery similarity.
    pub phase_recovery_similarity: f32,
    /// Mean correlation-sketch recovery similarity.
    pub correlation_recovery_similarity: f32,
    /// Mean classical noisy similarity.
    pub classical_noisy_similarity: f32,
    /// Mean phase noisy similarity.
    pub phase_noisy_similarity: f32,
    /// Mean correlation-sketch noisy similarity.
    pub correlation_noisy_similarity: f32,
    /// Topological beta-1 proxy for the similarity graph.
    pub beta1_proxy: isize,
    /// Threshold graph density in `[0, 1]`.
    pub topology_edge_density: f32,
    /// Threshold graph mean degree.
    pub topology_mean_degree: f32,
}

/// Full binding-probe report.
#[derive(Debug, Clone, PartialEq)]
pub struct BindingProbeReport {
    /// Run manifest.
    pub manifest: BenchmarkManifest,
    /// Results.
    pub result: BenchmarkResult,
}

impl BindingProbeReport {
    /// Returns a plain text summary for CLI examples and logs.
    pub fn to_text(&self) -> String {
        format!(
            "{}\ndimension={} trials={} noise={} seed={} topology_threshold={} fingerprint={:016x}\nclassical_recovery={:.4}\nphase_recovery={:.4}\ncorrelation_recovery={:.4}\nclassical_noisy={:.4}\nphase_noisy={:.4}\ncorrelation_noisy={:.4}\nbeta1_proxy={} edge_density={:.4} mean_degree={:.4}\nsubstrate={:?} confidence={:?}",
            self.manifest.name,
            self.manifest.dimension,
            self.manifest.trials,
            self.manifest.noise,
            self.manifest.seed,
            self.manifest.topology_threshold,
            self.manifest.reproducibility_fingerprint(),
            self.result.classical_recovery_similarity,
            self.result.phase_recovery_similarity,
            self.result.correlation_recovery_similarity,
            self.result.classical_noisy_similarity,
            self.result.phase_noisy_similarity,
            self.result.correlation_noisy_similarity,
            self.result.beta1_proxy,
            self.result.topology_edge_density,
            self.result.topology_mean_degree,
            self.manifest.substrate.backend,
            self.manifest.substrate.confidence,
        )
    }

    /// Returns a small JSON-like report string without requiring `serde`.
    pub fn to_json_like(&self) -> String {
        format!(
            "{{\"name\":\"{}\",\"dimension\":{},\"trials\":{},\"noise\":{},\"seed\":{},\"topology_threshold\":{},\"fingerprint\":\"{:016x}\",\"classical_recovery\":{},\"phase_recovery\":{},\"correlation_recovery\":{},\"classical_noisy\":{},\"phase_noisy\":{},\"correlation_noisy\":{},\"beta1_proxy\":{},\"edge_density\":{},\"mean_degree\":{}}}",
            escape_json(&self.manifest.name),
            self.manifest.dimension,
            self.manifest.trials,
            self.manifest.noise,
            self.manifest.seed,
            self.manifest.topology_threshold,
            self.manifest.reproducibility_fingerprint(),
            self.result.classical_recovery_similarity,
            self.result.phase_recovery_similarity,
            self.result.correlation_recovery_similarity,
            self.result.classical_noisy_similarity,
            self.result.phase_noisy_similarity,
            self.result.correlation_noisy_similarity,
            self.result.beta1_proxy,
            self.result.topology_edge_density,
            self.result.topology_mean_degree,
        )
    }
}

fn escape_json(input: &str) -> String {
    input.replace('\\', "\\\\").replace('"', "\\\"")
}
