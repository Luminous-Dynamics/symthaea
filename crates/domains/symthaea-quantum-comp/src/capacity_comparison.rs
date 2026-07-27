//! Calibrated capacity comparison: classical vs. phase-HDC bundling.
//!
//! `calibrated_comparison.rs` tested noise-robustness — how gracefully each
//! representation degrades under matched bit-error-rate — and found no
//! significant difference. Noise-robustness is not, however, the most
//! theoretically distinctive claim for a phase/holographic representation;
//! **capacity** (how many independent items can be superposed into one
//! bundle before reliable retrieval fails) is the more distinctive claim in
//! the literature on Fourier holographic reduced representations. This
//! module tests that claim instead, reusing the same fairness discipline:
//! both representations are built from the *same* underlying random bit
//! patterns, and both are scored with the *same* final metric.
//!
//! # Design
//!
//! For a bundle size `n`, `n` random items are drawn and bundled:
//!
//! - Classical: `BinaryHypervector::majority_bundle` (per-dimension majority vote).
//! - Phase: `PhaseHypervector::circular_bundle` (per-dimension circular mean) —
//!   applied to the phase encoding of the *same* bit patterns
//!   (`PhaseHypervector::from_binary`), so both representations bundle
//!   identical semantic content, not merely "both random."
//!
//! Recall is measured as **two-alternative forced-choice accuracy**: for one
//! bundled member and one never-bundled foil, does the representation's own
//! native similarity metric rank the true member above the foil? Both arms
//! report the same statistic — a probability in `[0, 1]` — computed from
//! each representation's own similarity function used only for *ranking*,
//! never compared as a raw magnitude across representations. This avoids
//! reintroducing the cross-representation metric-scale mismatch that
//! `calibrated_comparison.rs` (and, before it, `comparative.rs`'s
//! now-superseded headline number) had to correct for.

use crate::classical_hdc::BinaryHypervector;
use crate::errors::{QuantumCompError, Result};
use crate::experiment::{ExperimentManifest, ExperimentProtocol};
use crate::phase_hdc::PhaseHypervector;
use crate::significance::PairedDifferenceSummary;
use crate::statistics::{SampleSummary, first_threshold_crossing};
use crate::substrate::SubstrateProfile;

/// Configuration for a capacity sweep across bundle sizes.
#[derive(Debug, Clone, PartialEq)]
pub struct CapacitySweepConfig {
    /// Hypervector dimension.
    pub dimension: usize,
    /// Bundle sizes to test, ascending.
    pub bundle_sizes: Vec<usize>,
    /// Independent forced-choice trials per bundle size.
    pub trials_per_size: usize,
    /// Deterministic seed.
    pub seed: u64,
}

impl Default for CapacitySweepConfig {
    fn default() -> Self {
        Self {
            dimension: 1024,
            bundle_sizes: vec![1, 2, 4, 8, 16, 32, 64, 128, 256],
            trials_per_size: 32,
            seed: 0x5159_4D54_4841_4541,
        }
    }
}

/// One point in a capacity sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct CapacityPoint {
    /// Bundle size tested.
    pub bundle_size: usize,
    /// Classical forced-choice recall accuracy across trials.
    pub classical_accuracy: SampleSummary,
    /// Phase forced-choice recall accuracy across trials.
    pub phase_accuracy: SampleSummary,
    /// Paired comparison of classical vs. phase accuracy across trials.
    pub paired: Option<PairedDifferenceSummary>,
}

/// Full capacity sweep report.
#[derive(Debug, Clone, PartialEq)]
pub struct CapacitySweepReport {
    /// Experiment manifest and claim boundary.
    pub manifest: ExperimentManifest,
    /// Points in ascending bundle-size order.
    pub points: Vec<CapacityPoint>,
}

impl CapacitySweepReport {
    /// Returns the smallest bundle size at which classical accuracy first drops below `floor`.
    pub fn classical_capacity_at(&self, floor: f32) -> Option<f32> {
        let curve: Vec<(f32, f32)> = self
            .points
            .iter()
            .map(|p| (p.bundle_size as f32, p.classical_accuracy.mean))
            .collect();
        first_threshold_crossing(&curve, floor)
    }

    /// Returns the smallest bundle size at which phase accuracy first drops below `floor`.
    pub fn phase_capacity_at(&self, floor: f32) -> Option<f32> {
        let curve: Vec<(f32, f32)> = self
            .points
            .iter()
            .map(|p| (p.bundle_size as f32, p.phase_accuracy.mean))
            .collect();
        first_threshold_crossing(&curve, floor)
    }

    /// Returns a CSV report with one row per bundle size.
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "bundle_size,classical_accuracy_mean,phase_accuracy_mean,paired_mean_delta,sign_test_p_two_sided\n",
        );
        for p in &self.points {
            let (delta, p_value) = p
                .paired
                .as_ref()
                .map(|s| (s.delta.mean, s.sign_test_p_two_sided))
                .unwrap_or((f32::NAN, None));
            out.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:?}\n",
                p.bundle_size, p.classical_accuracy.mean, p.phase_accuracy.mean, delta, p_value,
            ));
        }
        out
    }

    /// Returns a compact text summary.
    pub fn to_text(&self) -> String {
        let mut out = self.manifest.to_text();
        out.push('\n');
        out.push_str(&self.to_csv());
        if let Some(c) = self.classical_capacity_at(0.75) {
            out.push_str(&format!("classical_capacity_at_0.75_floor={c}\n"));
        }
        if let Some(c) = self.phase_capacity_at(0.75) {
            out.push_str(&format!("phase_capacity_at_0.75_floor={c}\n"));
        }
        out
    }
}

/// Runs a calibrated capacity sweep.
#[derive(Debug, Clone)]
pub struct CapacitySweepRunner {
    config: CapacitySweepConfig,
}

impl CapacitySweepRunner {
    /// Creates a new runner.
    pub fn new(config: CapacitySweepConfig) -> Result<Self> {
        if config.dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        if config.bundle_sizes.is_empty() {
            return Err(QuantumCompError::InvalidConfig(
                "bundle_sizes must be nonempty",
            ));
        }
        if config.bundle_sizes.contains(&0) {
            return Err(QuantumCompError::InvalidConfig(
                "bundle_sizes entries must be > 0",
            ));
        }
        if config.trials_per_size == 0 {
            return Err(QuantumCompError::InvalidConfig(
                "trials_per_size must be > 0",
            ));
        }
        Ok(Self { config })
    }

    /// Runs the sweep.
    pub fn run(&self) -> Result<CapacitySweepReport> {
        let mut points = Vec::with_capacity(self.config.bundle_sizes.len());
        for &bundle_size in &self.config.bundle_sizes {
            points.push(self.run_one_size(bundle_size)?);
        }
        let manifest = ExperimentManifest::local_simulation(
            "calibrated-capacity-comparison-v0.1",
            ExperimentProtocol::CalibratedCrossRepresentationComparison,
            self.config.seed,
            self.config.dimension,
            self.config.trials_per_size,
            SubstrateProfile::quantum_inspired(),
        );
        Ok(CapacitySweepReport { manifest, points })
    }

    fn run_one_size(&self, bundle_size: usize) -> Result<CapacityPoint> {
        let mut classical_hits = Vec::with_capacity(self.config.trials_per_size);
        let mut phase_hits = Vec::with_capacity(self.config.trials_per_size);

        for trial in 0..self.config.trials_per_size {
            let base_seed = self
                .config
                .seed
                .wrapping_add((trial as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                .wrapping_add((bundle_size as u64).wrapping_mul(0xB5AD_4ECE_DA1C_E2A9));

            let mut members = Vec::with_capacity(bundle_size);
            for i in 0..bundle_size {
                let item_seed = base_seed
                    .wrapping_add((i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93))
                    ^ 0xA11CE;
                members.push(BinaryHypervector::random(self.config.dimension, item_seed)?);
            }
            let foil = BinaryHypervector::random(self.config.dimension, base_seed ^ 0xF01D)?;

            let classical_bundle =
                BinaryHypervector::majority_bundle(&members, base_seed ^ 0x71E5EED)?;
            let phase_members: Vec<PhaseHypervector> =
                members.iter().map(PhaseHypervector::from_binary).collect();
            let phase_bundle = PhaseHypervector::circular_bundle(&phase_members)?;
            let phase_foil = PhaseHypervector::from_binary(&foil);

            // Probe one random member per trial (not all `bundle_size` of them) so
            // per-trial cost stays independent of bundle_size.
            let probe_index = (base_seed as usize) % bundle_size;
            let probe_member = &members[probe_index];
            let probe_phase_member = &phase_members[probe_index];

            let classical_member_sim = probe_member.similarity(&classical_bundle)?;
            let classical_foil_sim = foil.similarity(&classical_bundle)?;
            classical_hits.push(if classical_member_sim > classical_foil_sim {
                1.0
            } else {
                0.0
            });

            let phase_member_sim = probe_phase_member.circular_similarity(&phase_bundle)?;
            let phase_foil_sim = phase_foil.circular_similarity(&phase_bundle)?;
            phase_hits.push(if phase_member_sim > phase_foil_sim {
                1.0
            } else {
                0.0
            });
        }

        let classical_accuracy =
            SampleSummary::from_samples(&classical_hits).expect("nonempty by config");
        let phase_accuracy = SampleSummary::from_samples(&phase_hits).expect("nonempty by config");
        let paired = PairedDifferenceSummary::from_pairs(&classical_hits, &phase_hits, 1e-6);

        Ok(CapacityPoint {
            bundle_size,
            classical_accuracy,
            phase_accuracy,
            paired,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_item_bundle_is_almost_always_correctly_recalled() {
        let config = CapacitySweepConfig {
            dimension: 1024,
            bundle_sizes: vec![1],
            trials_per_size: 32,
            seed: 1,
        };
        let report = CapacitySweepRunner::new(config).unwrap().run().unwrap();
        let point = &report.points[0];
        assert!(
            point.classical_accuracy.mean > 0.9,
            "{:?}",
            point.classical_accuracy
        );
        assert!(
            point.phase_accuracy.mean > 0.9,
            "{:?}",
            point.phase_accuracy
        );
    }

    #[test]
    fn accuracy_is_non_increasing_as_bundle_size_grows() {
        let config = CapacitySweepConfig {
            dimension: 512,
            bundle_sizes: vec![1, 32, 128],
            trials_per_size: 48,
            seed: 2,
        };
        let report = CapacitySweepRunner::new(config).unwrap().run().unwrap();
        let classical: Vec<f32> = report
            .points
            .iter()
            .map(|p| p.classical_accuracy.mean)
            .collect();
        let phase: Vec<f32> = report
            .points
            .iter()
            .map(|p| p.phase_accuracy.mean)
            .collect();
        assert!(
            classical[0] >= classical[2] - 0.05,
            "classical accuracy should not improve with bundle size: {classical:?}"
        );
        assert!(
            phase[0] >= phase[2] - 0.05,
            "phase accuracy should not improve with bundle size: {phase:?}"
        );
        // Large bundles should be harder to recall from than a single-item bundle.
        assert!(classical[2] < classical[0]);
        assert!(phase[2] < phase[0]);
    }

    #[test]
    fn rejects_invalid_config() {
        assert!(
            CapacitySweepRunner::new(CapacitySweepConfig {
                dimension: 0,
                ..CapacitySweepConfig::default()
            })
            .is_err()
        );
        assert!(
            CapacitySweepRunner::new(CapacitySweepConfig {
                bundle_sizes: vec![],
                ..CapacitySweepConfig::default()
            })
            .is_err()
        );
        assert!(
            CapacitySweepRunner::new(CapacitySweepConfig {
                bundle_sizes: vec![0],
                ..CapacitySweepConfig::default()
            })
            .is_err()
        );
        assert!(
            CapacitySweepRunner::new(CapacitySweepConfig {
                trials_per_size: 0,
                ..CapacitySweepConfig::default()
            })
            .is_err()
        );
    }

    #[test]
    fn capacity_threshold_helpers_find_the_crossing() {
        // Verified via a real run (dimension=512, seed=4): accuracy is 1.0 at
        // bundle_size=1 and drops to ~0.66/0.56 by bundle_size=512/2048 for
        // both representations, so the 0.75 floor is crossed within this range.
        let config = CapacitySweepConfig {
            dimension: 512,
            bundle_sizes: vec![1, 512, 2048],
            trials_per_size: 32,
            seed: 4,
        };
        let report = CapacitySweepRunner::new(config).unwrap().run().unwrap();
        let classical_c = report.classical_capacity_at(0.75);
        let phase_c = report.phase_capacity_at(0.75);
        assert!(
            classical_c.is_some() || phase_c.is_some(),
            "expected at least one representation to cross the 0.75 floor within tested sizes"
        );
    }
}
