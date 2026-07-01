//! Classical proxy for entanglement-mediated binding probes.
//!
//! This module does not simulate a physical quantum computer. It gives Symthaea a
//! stable experimental type for asking whether parity/correlation-style binding,
//! noisy pair coherence, and topology summaries behave differently from ordinary
//! binary HDC baselines under controlled assumptions.

use crate::classical_hdc::BinaryHypervector;
use crate::errors::{QuantumCompError, Result};
use crate::experiment::{ExperimentManifest, ExperimentProtocol};
use crate::rng::XorShift64;
use crate::substrate::SubstrateProfile;

/// Pairwise parity sketch with an explicit classical coherence proxy.
#[derive(Debug, Clone, PartialEq)]
pub struct EntanglementProxySketch {
    dimension: usize,
    parity: BinaryHypervector,
    pair_coherence: Vec<f32>,
}

impl EntanglementProxySketch {
    /// Creates a proxy sketch from an item and key.
    ///
    /// `decoherence` in `[0, 1]` degrades the pair-coherence amplitudes and can
    /// flip parity bits. This is a toy research parameter, not a physical noise model.
    pub fn bind(
        item: &BinaryHypervector,
        key: &BinaryHypervector,
        decoherence: f32,
        seed: u64,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&decoherence) {
            return Err(QuantumCompError::InvalidProbability);
        }
        if item.dimension() != key.dimension() {
            return Err(QuantumCompError::DimensionMismatch {
                expected: item.dimension(),
                actual: key.dimension(),
            });
        }
        let mut rng = XorShift64::new(seed);
        let mut parity = item.bind_xor(key)?;
        let mut pair_coherence = Vec::with_capacity(item.dimension());
        for bit in 0..item.dimension() {
            let true_parity = parity.bit(bit).unwrap_or(false);
            if rng.chance(decoherence * 0.5) {
                parity.set_bit(bit, !true_parity)?;
            }
            let sign = if true_parity { -1.0 } else { 1.0 };
            let jitter = rng.next_centered_f32() * decoherence;
            pair_coherence.push((sign * (1.0 - decoherence) + jitter).clamp(-1.0, 1.0));
        }
        Ok(Self {
            dimension: item.dimension(),
            parity,
            pair_coherence,
        })
    }

    /// Returns the sketch dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the stored parity hypervector.
    pub fn parity(&self) -> &BinaryHypervector {
        &self.parity
    }

    /// Returns pair-coherence proxy values in `[-1, 1]`.
    pub fn pair_coherence(&self) -> &[f32] {
        &self.pair_coherence
    }

    /// Recovers an item using the supplied key and stored parity.
    pub fn recover_item(&self, key: &BinaryHypervector) -> Result<BinaryHypervector> {
        self.parity.unbind_xor(key)
    }

    /// Mean absolute pair coherence in `[0, 1]`.
    pub fn mean_abs_pair_coherence(&self) -> f32 {
        if self.pair_coherence.is_empty() {
            return 0.0;
        }
        self.pair_coherence.iter().map(|v| v.abs()).sum::<f32>() / self.pair_coherence.len() as f32
    }

    /// Fraction of proxy pairs whose coherence sign agrees with the stored parity.
    pub fn sign_agreement(&self) -> f32 {
        if self.pair_coherence.is_empty() {
            return 0.0;
        }
        let mut ok = 0usize;
        for i in 0..self.dimension {
            let parity_bit = self.parity.bit(i).unwrap_or(false);
            let coherence_bit = self.pair_coherence[i] < 0.0;
            if parity_bit == coherence_bit {
                ok += 1;
            }
        }
        ok as f32 / self.dimension as f32
    }
}

/// Configuration for the entanglement-proxy binding probe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntanglementProxyConfig {
    /// Hypervector dimension.
    pub dimension: usize,
    /// Independent trials.
    pub trials: usize,
    /// Toy decoherence parameter in `[0, 1]`.
    pub decoherence: f32,
    /// Deterministic seed.
    pub seed: u64,
}

impl Default for EntanglementProxyConfig {
    fn default() -> Self {
        Self {
            dimension: 1024,
            trials: 16,
            decoherence: 0.05,
            seed: 0x5159_4D54_5142_4954,
        }
    }
}

/// Report produced by [`EntanglementProxyRunner`].
#[derive(Debug, Clone, PartialEq)]
pub struct EntanglementProxyReport {
    /// Experiment manifest and claim boundary.
    pub manifest: ExperimentManifest,
    /// Mean recovery similarity when the correct key is supplied.
    pub mean_recovery_similarity: f32,
    /// Mean recovery similarity when a wrong key is supplied.
    pub wrong_key_similarity: f32,
    /// Gap between correct-key and wrong-key recovery.
    pub recovery_gap: f32,
    /// Mean absolute pair coherence.
    pub mean_abs_pair_coherence: f32,
    /// Mean sign agreement between coherence proxy and stored parity.
    pub mean_sign_agreement: f32,
}

impl EntanglementProxyReport {
    /// Returns a compact text summary.
    pub fn to_text(&self) -> String {
        format!(
            "{}\nmean_recovery={:.6}\nwrong_key={:.6}\nrecovery_gap={:.6}\nmean_abs_pair_coherence={:.6}\nmean_sign_agreement={:.6}",
            self.manifest.to_text(),
            self.mean_recovery_similarity,
            self.wrong_key_similarity,
            self.recovery_gap,
            self.mean_abs_pair_coherence,
            self.mean_sign_agreement,
        )
    }
}

/// Runs the entanglement-proxy binding probe.
#[derive(Debug, Clone)]
pub struct EntanglementProxyRunner {
    config: EntanglementProxyConfig,
}

impl EntanglementProxyRunner {
    /// Creates a new runner.
    pub fn new(config: EntanglementProxyConfig) -> Result<Self> {
        if config.dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        if config.trials == 0 {
            return Err(QuantumCompError::InvalidConfig("trials must be > 0"));
        }
        if !(0.0..=1.0).contains(&config.decoherence) {
            return Err(QuantumCompError::InvalidProbability);
        }
        Ok(Self { config })
    }

    /// Runs the probe.
    pub fn run(&self) -> Result<EntanglementProxyReport> {
        let mut recovery = 0.0f32;
        let mut wrong_key = 0.0f32;
        let mut coherence = 0.0f32;
        let mut sign_agreement = 0.0f32;

        for trial in 0..self.config.trials {
            let seed = self
                .config
                .seed
                .wrapping_add((trial as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let item = BinaryHypervector::random(self.config.dimension, seed ^ 0xA17E)?;
            let key = BinaryHypervector::random(self.config.dimension, seed ^ 0xBEEF)?;
            let wrong = BinaryHypervector::random(self.config.dimension, seed ^ 0xBAD5EED)?;
            let sketch =
                EntanglementProxySketch::bind(&item, &key, self.config.decoherence, seed ^ 0xE17A)?;
            let recovered = sketch.recover_item(&key)?;
            let wrong_recovered = sketch.recover_item(&wrong)?;
            recovery += item.similarity(&recovered)?;
            wrong_key += item.similarity(&wrong_recovered)?;
            coherence += sketch.mean_abs_pair_coherence();
            sign_agreement += sketch.sign_agreement();
        }

        let denom = self.config.trials as f32;
        let mean_recovery_similarity = recovery / denom;
        let wrong_key_similarity = wrong_key / denom;
        let manifest = ExperimentManifest::local_simulation(
            "entanglement-proxy-binding-probe-v0.3",
            ExperimentProtocol::EntanglementProxyBinding,
            self.config.seed,
            self.config.dimension,
            self.config.trials,
            SubstrateProfile::quantum_inspired(),
        );
        Ok(EntanglementProxyReport {
            manifest,
            mean_recovery_similarity,
            wrong_key_similarity,
            recovery_gap: mean_recovery_similarity - wrong_key_similarity,
            mean_abs_pair_coherence: coherence / denom,
            mean_sign_agreement: sign_agreement / denom,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entanglement_proxy_gap_is_large_at_low_decoherence() {
        let report = EntanglementProxyRunner::new(EntanglementProxyConfig {
            dimension: 256,
            trials: 8,
            decoherence: 0.01,
            seed: 7,
        })
        .unwrap()
        .run()
        .unwrap();
        assert!(report.mean_recovery_similarity > 0.95);
        assert!(report.recovery_gap > 0.35);
    }
}
