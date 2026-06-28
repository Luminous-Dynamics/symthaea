//! Negative-control probes for binding experiments.
//!
//! A research crate needs controls. These probes verify that recovery succeeds
//! with the intended key and collapses toward chance with wrong keys or random
//! unrelated items.

use crate::classical_hdc::BinaryHypervector;
use crate::errors::{QuantumCompError, Result};
use crate::experiment::{ExperimentManifest, ExperimentProtocol};
use crate::substrate::SubstrateProfile;

/// Configuration for negative-control binding checks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeControlConfig {
    /// Hypervector dimension.
    pub dimension: usize,
    /// Number of independent trials.
    pub trials: usize,
    /// Bit-flip noise applied after correct recovery.
    pub noise: f32,
    /// Deterministic seed.
    pub seed: u64,
}

impl Default for NegativeControlConfig {
    fn default() -> Self {
        Self {
            dimension: 1024,
            trials: 16,
            noise: 0.05,
            seed: 0x434F_4E54_524F_4C53,
        }
    }
}

/// Report for a negative-control run.
#[derive(Debug, Clone, PartialEq)]
pub struct NegativeControlReport {
    /// Experiment manifest and claim boundary.
    pub manifest: ExperimentManifest,
    /// Mean similarity when recovering with the matched key.
    pub matched_key_similarity: f32,
    /// Mean similarity when recovering with an unrelated key.
    pub wrong_key_similarity: f32,
    /// Mean similarity to an unrelated random item.
    pub random_item_similarity: f32,
    /// Matched minus wrong-key similarity.
    pub control_gap: f32,
}

impl NegativeControlReport {
    /// Returns a compact text report.
    pub fn to_text(&self) -> String {
        format!(
            "{}\nmatched_key={:.6}\nwrong_key={:.6}\nrandom_item={:.6}\ncontrol_gap={:.6}",
            self.manifest.to_text(),
            self.matched_key_similarity,
            self.wrong_key_similarity,
            self.random_item_similarity,
            self.control_gap,
        )
    }
}

/// Runs negative-control checks for classical XOR binding.
#[derive(Debug, Clone)]
pub struct NegativeControlRunner {
    config: NegativeControlConfig,
}

impl NegativeControlRunner {
    /// Creates a new runner.
    pub fn new(config: NegativeControlConfig) -> Result<Self> {
        if config.dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        if config.trials == 0 {
            return Err(QuantumCompError::InvalidConfig("trials must be > 0"));
        }
        if !(0.0..=1.0).contains(&config.noise) {
            return Err(QuantumCompError::InvalidProbability);
        }
        Ok(Self { config })
    }

    /// Runs the control probe.
    pub fn run(&self) -> Result<NegativeControlReport> {
        let mut matched = 0.0f32;
        let mut wrong = 0.0f32;
        let mut random = 0.0f32;
        for trial in 0..self.config.trials {
            let seed = self
                .config
                .seed
                .wrapping_add((trial as u64).wrapping_mul(0x517C_C1B7));
            let item = BinaryHypervector::random(self.config.dimension, seed ^ 0x1A)?;
            let key = BinaryHypervector::random(self.config.dimension, seed ^ 0x2B)?;
            let wrong_key = BinaryHypervector::random(self.config.dimension, seed ^ 0x3C)?;
            let unrelated = BinaryHypervector::random(self.config.dimension, seed ^ 0x4D)?;
            let bound = item.bind_xor(&key)?;
            let recovered = bound
                .unbind_xor(&key)?
                .with_bitflip_noise(self.config.noise, seed ^ 0x55);
            let wrong_recovered = bound.unbind_xor(&wrong_key)?;
            matched += item.similarity(&recovered)?;
            wrong += item.similarity(&wrong_recovered)?;
            random += item.similarity(&unrelated)?;
        }
        let denom = self.config.trials as f32;
        let matched_key_similarity = matched / denom;
        let wrong_key_similarity = wrong / denom;
        let manifest = ExperimentManifest::local_simulation(
            "negative-control-binding-probe-v0.3",
            ExperimentProtocol::NegativeControl,
            self.config.seed,
            self.config.dimension,
            self.config.trials,
            SubstrateProfile::classical_cpu(),
        );
        Ok(NegativeControlReport {
            manifest,
            matched_key_similarity,
            wrong_key_similarity,
            random_item_similarity: random / denom,
            control_gap: matched_key_similarity - wrong_key_similarity,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_gap_is_visible() {
        let report = NegativeControlRunner::new(NegativeControlConfig {
            dimension: 256,
            trials: 8,
            noise: 0.02,
            seed: 99,
        })
        .unwrap()
        .run()
        .unwrap();
        assert!(report.control_gap > 0.35);
        assert!(report.wrong_key_similarity < 0.65);
    }
}
