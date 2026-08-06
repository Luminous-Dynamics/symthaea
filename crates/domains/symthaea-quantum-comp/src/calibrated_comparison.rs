//! Calibrated cross-representation comparison.
//!
//! `comparative.rs` and `noise_sweep.rs` compare classical binary HDC and
//! phase-HDC by feeding the *same literal* `noise` parameter into two
//! structurally different perturbation models:
//!
//! - Classical: `BinaryHypervector::with_bitflip_noise(p)` flips each bit
//!   independently with probability `p` — a *linear* bit-error-rate.
//! - Phase: `PhaseHypervector::with_phase_noise(sigma)` perturbs each
//!   dimension's angle by noise scaled by `sigma`, scored via
//!   `circular_similarity`'s `cos(Δphase)` — *quadratically* insensitive to
//!   small angles.
//!
//! Comparing them at matched literal parameter values produced a spurious,
//! enormous effect size (`classical_minus_phase_noisy_dz = -33.36` in
//! `comparative_report`'s default configuration — see
//! `docs/RESEARCH_NOTES.md`, "First independent run and a real finding
//! (2026-07-24)"). That number reflects the mismatched noise models, not a
//! property of either representation.
//!
//! This module calibrates both channels to a shared, representation-neutral
//! unit: **bit-error-rate (BER)** under each representation's own natural
//! hard-decision rule.
//!
//! - Classical BER is exact by construction: `BER = p`.
//! - Phase BER is measured empirically (Monte Carlo, deterministic seeds)
//!   using the crate's real `with_phase_noise` + `to_binary_halfplane`
//!   code path — not an idealized Gaussian formula. A bit is encoded as
//!   phase `0` or `π` (`PhaseHypervector::from_binary`); after noise,
//!   `to_binary_halfplane` hard-decodes each phase back to a bit, and BER is
//!   the fraction of dimensions that decode to the wrong bit.
//!
//! [`calibrate_phase_sigma_for_ber`] finds, by bisection over the actually
//! measured BER curve, the `sigma` whose phase channel matches a target
//! classical bitflip probability. Once BER is matched,
//! [`CalibratedComparisonRunner`] runs the full bind → unbind → noise →
//! recover pipeline for both representations and scores them with the
//! **same final metric**: `BinaryHypervector::similarity` between the
//! original item bits and the noisy-recovered result, hard-decoding
//! phase-HDC's result via `to_binary_halfplane` first. Both arms report
//! literally the same statistic, in the same units, at matched input noise
//! strength — eliminating both halves of the original artifact (mismatched
//! noise magnitude *and* mismatched output metric).

use crate::classical_hdc::BinaryHypervector;
use crate::errors::{QuantumCompError, Result};
use crate::experiment::{ExperimentManifest, ExperimentProtocol};
use crate::phase_hdc::PhaseHypervector;
use crate::significance::PairedDifferenceSummary;
use crate::statistics::SampleSummary;
use crate::substrate::SubstrateProfile;

/// Returns the classical channel's bit-error-rate for a given bitflip probability.
///
/// This is the identity function: independent bitflip noise with probability
/// `p` has bit-error-rate exactly `p` by construction. It exists so the
/// classical and phase channels have parallel, equally-named entry points in
/// this module's public API.
pub fn classical_channel_ber(probability: f32) -> f32 {
    probability.clamp(0.0, 1.0)
}

/// Empirically measures the classical channel's realized bit-error-rate.
///
/// Included alongside [`measure_phase_channel_ber`] so both channels get the
/// same audit treatment rather than trusting the classical side "by
/// construction" while only measuring the phase side.
pub fn measure_classical_channel_ber(probability: f32, dimension: usize, seed: u64) -> Result<f32> {
    if dimension == 0 {
        return Err(QuantumCompError::InvalidDimension);
    }
    if !(0.0..=1.0).contains(&probability) {
        return Err(QuantumCompError::InvalidProbability);
    }
    let clean = BinaryHypervector::random(dimension, seed ^ 0x71A5_51C4_0000_0001)?;
    let noisy = clean.with_bitflip_noise(probability, seed ^ 0xB17F_71C4_0000_0002);
    let dist = clean.hamming_distance(&noisy)? as f32;
    Ok(dist / dimension as f32)
}

/// Empirically measures the phase channel's realized bit-error-rate for a given `sigma`.
///
/// Encodes a random bit pattern as phase (`0` or `π` per dimension), applies
/// `with_phase_noise(sigma)`, hard-decodes via `to_binary_halfplane`, and
/// returns the fraction of dimensions that decoded to the wrong bit. Uses a
/// large `dimension` as the Monte Carlo sample size — one call already
/// averages over `dimension` independent per-dimension noise draws.
pub fn measure_phase_channel_ber(sigma: f32, dimension: usize, seed: u64) -> Result<f32> {
    if dimension == 0 {
        return Err(QuantumCompError::InvalidDimension);
    }
    if sigma < 0.0 {
        return Err(QuantumCompError::InvalidConfig("sigma must be >= 0"));
    }
    let clean = BinaryHypervector::random(dimension, seed ^ 0x6841_5EF3_0000_0003)?;
    let phase = PhaseHypervector::from_binary(&clean);
    let noisy = phase.with_phase_noise(sigma, seed ^ 0x0C0D_EC0D_0000_0004);
    let decoded = noisy.to_binary_halfplane()?;
    let dist = clean.hamming_distance(&decoded)? as f32;
    Ok(dist / dimension as f32)
}

/// Finds, by bisection, the phase-noise `sigma` whose measured bit-error-rate
/// matches `target_ber`.
///
/// Both channels saturate at BER = 0.5 (chance) as their respective noise
/// parameter grows without bound, so the search expands its upper bracket
/// geometrically until it brackets the target rather than assuming a fixed
/// range.
pub fn calibrate_phase_sigma_for_ber(
    target_ber: f32,
    calibration_dimension: usize,
    seed: u64,
) -> Result<f32> {
    if !(0.0..0.5).contains(&target_ber) {
        return Err(QuantumCompError::InvalidProbability);
    }
    if target_ber == 0.0 {
        return Ok(0.0);
    }
    let mut lo = 0.0f32;
    let mut hi = core::f32::consts::TAU;
    let mut expansions = 0;
    while measure_phase_channel_ber(hi, calibration_dimension, seed)? < target_ber {
        hi *= 2.0;
        expansions += 1;
        if expansions > 16 {
            return Err(QuantumCompError::InvalidConfig(
                "phase channel BER did not reach target within search bound",
            ));
        }
    }
    for _ in 0..40 {
        let mid = (lo + hi) * 0.5;
        let mid_ber = measure_phase_channel_ber(mid, calibration_dimension, seed)?;
        if mid_ber < target_ber {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((lo + hi) * 0.5)
}

/// Configuration for one calibrated comparison point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CalibratedComparisonConfig {
    /// Hypervector dimension used for the bind/unbind/recovery trials.
    pub dimension: usize,
    /// Independent trials.
    pub trials: usize,
    /// Shared target bit-error-rate both channels are calibrated to.
    pub target_ber: f32,
    /// Deterministic seed.
    pub seed: u64,
    /// Dimension used for the (separate, larger) BER calibration measurement.
    pub calibration_dimension: usize,
}

impl Default for CalibratedComparisonConfig {
    fn default() -> Self {
        Self {
            dimension: 1024,
            trials: 16,
            target_ber: 0.05,
            seed: 0x5159_4D54_4841_4541,
            calibration_dimension: 16_384,
        }
    }
}

/// Report for one calibrated comparison point.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedComparisonReport {
    /// Experiment manifest and claim boundary.
    pub manifest: ExperimentManifest,
    /// Target bit-error-rate both channels were calibrated to.
    pub target_ber: f32,
    /// Phase-noise sigma calibrated to reach `target_ber`.
    pub phase_sigma: f32,
    /// Classical channel's realized BER, measured the same way as the phase channel.
    pub measured_classical_ber: f32,
    /// Phase channel's realized BER at the calibrated `phase_sigma`.
    pub measured_phase_ber: f32,
    /// Classical recovery accuracy (item vs. noisy-recovered bits), matched-key trials.
    pub classical_recovery_accuracy: SampleSummary,
    /// Phase recovery accuracy (item bits vs. hard-decoded noisy-recovered phase), matched-key trials.
    pub phase_recovery_accuracy: SampleSummary,
    /// Classical recovery accuracy when unbinding with the wrong key (sanity check).
    pub classical_wrong_key_accuracy: SampleSummary,
    /// Phase recovery accuracy when unbinding with the wrong key (sanity check).
    pub phase_wrong_key_accuracy: SampleSummary,
    /// Paired comparison of classical vs. phase recovery accuracy across trials.
    pub paired_recovery: Option<PairedDifferenceSummary>,
}

impl CalibratedComparisonReport {
    /// Returns a compact line-oriented text report.
    pub fn to_text(&self) -> String {
        fn fmt(label: &str, s: &SampleSummary) -> String {
            let (lo, hi) = s.approximate_95_ci();
            format!(
                "{label}_mean={:.6} {label}_ci95=[{:.6},{:.6}]",
                s.mean, lo, hi
            )
        }
        format!(
            "{}\ntarget_ber={:.4} phase_sigma={:.6} measured_classical_ber={:.6} measured_phase_ber={:.6}\n{}\n{}\n{}\n{}\n{}",
            self.manifest.to_text(),
            self.target_ber,
            self.phase_sigma,
            self.measured_classical_ber,
            self.measured_phase_ber,
            fmt("classical_recovery", &self.classical_recovery_accuracy),
            fmt("phase_recovery", &self.phase_recovery_accuracy),
            fmt("classical_wrong_key", &self.classical_wrong_key_accuracy),
            fmt("phase_wrong_key", &self.phase_wrong_key_accuracy),
            self.paired_recovery
                .as_ref()
                .map(|p| p.to_text("classical_recovery", "phase_recovery"))
                .unwrap_or_else(|| "paired_recovery=unavailable".to_string()),
        )
    }
}

/// Runs a single calibrated comparison point.
#[derive(Debug, Clone)]
pub struct CalibratedComparisonRunner {
    config: CalibratedComparisonConfig,
}

impl CalibratedComparisonRunner {
    /// Creates a new runner.
    pub fn new(config: CalibratedComparisonConfig) -> Result<Self> {
        if config.dimension == 0 || config.calibration_dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        if config.trials == 0 {
            return Err(QuantumCompError::InvalidConfig("trials must be > 0"));
        }
        if !(0.0..0.5).contains(&config.target_ber) {
            return Err(QuantumCompError::InvalidProbability);
        }
        Ok(Self { config })
    }

    /// Runs the calibrated comparison.
    pub fn run(&self) -> Result<CalibratedComparisonReport> {
        let calibration_seed = self.config.seed ^ 0xCA11_B4A7_E000_0005;
        let phase_sigma = calibrate_phase_sigma_for_ber(
            self.config.target_ber,
            self.config.calibration_dimension,
            calibration_seed,
        )?;
        let measured_classical_ber = measure_classical_channel_ber(
            self.config.target_ber,
            self.config.calibration_dimension,
            calibration_seed,
        )?;
        let measured_phase_ber = measure_phase_channel_ber(
            phase_sigma,
            self.config.calibration_dimension,
            calibration_seed,
        )?;

        let mut classical_recovery = Vec::with_capacity(self.config.trials);
        let mut phase_recovery = Vec::with_capacity(self.config.trials);
        let mut classical_wrong_key = Vec::with_capacity(self.config.trials);
        let mut phase_wrong_key = Vec::with_capacity(self.config.trials);

        for trial in 0..self.config.trials {
            let seed = self
                .config
                .seed
                .wrapping_add((trial as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let item = BinaryHypervector::random(self.config.dimension, seed ^ 0xA11CE)?;
            let key = BinaryHypervector::random(self.config.dimension, seed ^ 0xB0B)?;
            let wrong = BinaryHypervector::random(self.config.dimension, seed ^ 0xBAD5EED)?;

            // Classical arm: bind, unbind with the correct key, add BER-calibrated
            // bitflip noise, compare to the original item bits.
            let bound = item.bind_xor(&key)?;
            let recovered = bound.unbind_xor(&key)?;
            let noisy = recovered.with_bitflip_noise(self.config.target_ber, seed ^ 0xD15EA5E);
            classical_recovery.push(item.similarity(&noisy)?);

            let wrong_recovered = bound.unbind_xor(&wrong)?;
            classical_wrong_key.push(item.similarity(&wrong_recovered)?);

            // Phase arm: same pipeline, but noise is BER-calibrated phase jitter and
            // the recovered phase is hard-decoded to bits before scoring — same
            // final metric, same units, as the classical arm.
            let phase_item = PhaseHypervector::from_binary(&item);
            let phase_key = PhaseHypervector::from_binary(&key);
            let phase_wrong = PhaseHypervector::from_binary(&wrong);
            let phase_bound = phase_item.bind_phase(&phase_key)?;
            let phase_recovered = phase_bound.unbind_phase(&phase_key)?;
            let phase_noisy = phase_recovered.with_phase_noise(phase_sigma, seed ^ 0xF00D);
            let phase_decoded = phase_noisy.to_binary_halfplane()?;
            phase_recovery.push(item.similarity(&phase_decoded)?);

            let phase_wrong_recovered = phase_bound.unbind_phase(&phase_wrong)?;
            let phase_wrong_decoded = phase_wrong_recovered.to_binary_halfplane()?;
            phase_wrong_key.push(item.similarity(&phase_wrong_decoded)?);
        }

        let classical_recovery_accuracy =
            SampleSummary::from_samples(&classical_recovery).expect("nonempty by config");
        let phase_recovery_accuracy =
            SampleSummary::from_samples(&phase_recovery).expect("nonempty by config");
        let classical_wrong_key_accuracy =
            SampleSummary::from_samples(&classical_wrong_key).expect("nonempty by config");
        let phase_wrong_key_accuracy =
            SampleSummary::from_samples(&phase_wrong_key).expect("nonempty by config");
        let paired_recovery =
            PairedDifferenceSummary::from_pairs(&classical_recovery, &phase_recovery, 1e-6);

        let manifest = ExperimentManifest::local_simulation(
            "calibrated-cross-representation-comparison-v0.1",
            ExperimentProtocol::CalibratedCrossRepresentationComparison,
            self.config.seed,
            self.config.dimension,
            self.config.trials,
            SubstrateProfile::quantum_inspired(),
        );

        Ok(CalibratedComparisonReport {
            manifest,
            target_ber: self.config.target_ber,
            phase_sigma,
            measured_classical_ber,
            measured_phase_ber,
            classical_recovery_accuracy,
            phase_recovery_accuracy,
            classical_wrong_key_accuracy,
            phase_wrong_key_accuracy,
            paired_recovery,
        })
    }
}

/// Configuration for a calibrated sweep across several target bit-error-rates.
///
/// This is the calibrated replacement for `noise_sweep`'s cross-representation
/// columns: every row here is comparable across representations because the
/// noise strength (BER) and the output metric (bit-recovery accuracy) are
/// both shared, not merely the literal `noise` parameter value.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedSweepConfig {
    /// Hypervector dimension for the bind/unbind/recovery trials.
    pub dimension: usize,
    /// Independent trials per BER point.
    pub trials: usize,
    /// Target bit-error-rates to sweep, ascending.
    pub target_bers: Vec<f32>,
    /// Deterministic seed.
    pub seed: u64,
    /// Dimension used for BER calibration measurements.
    pub calibration_dimension: usize,
}

impl Default for CalibratedSweepConfig {
    fn default() -> Self {
        Self {
            dimension: 1024,
            trials: 16,
            target_bers: vec![0.01, 0.05, 0.10, 0.20, 0.30, 0.40],
            seed: 0x5159_4D54_4841_4541,
            calibration_dimension: 16_384,
        }
    }
}

/// Full calibrated sweep report.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibratedSweepReport {
    /// Points in ascending target-BER order.
    pub points: Vec<CalibratedComparisonReport>,
}

impl CalibratedSweepReport {
    /// Returns a CSV report with one row per target BER.
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "target_ber,phase_sigma,measured_classical_ber,measured_phase_ber,classical_recovery_mean,phase_recovery_mean,classical_wrong_key_mean,phase_wrong_key_mean,paired_mean_delta,sign_test_p_two_sided\n",
        );
        for p in &self.points {
            let (delta, p_value) = p
                .paired_recovery
                .as_ref()
                .map(|s| (s.delta.mean, s.sign_test_p_two_sided))
                .unwrap_or((f32::NAN, None));
            out.push_str(&format!(
                "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:?}\n",
                p.target_ber,
                p.phase_sigma,
                p.measured_classical_ber,
                p.measured_phase_ber,
                p.classical_recovery_accuracy.mean,
                p.phase_recovery_accuracy.mean,
                p.classical_wrong_key_accuracy.mean,
                p.phase_wrong_key_accuracy.mean,
                delta,
                p_value,
            ));
        }
        out
    }
}

/// Runs a calibrated sweep across several target bit-error-rates.
#[derive(Debug, Clone)]
pub struct CalibratedSweepRunner {
    config: CalibratedSweepConfig,
}

impl CalibratedSweepRunner {
    /// Creates a new sweep runner.
    pub fn new(config: CalibratedSweepConfig) -> Result<Self> {
        if config.target_bers.is_empty() {
            return Err(QuantumCompError::InvalidConfig(
                "target_bers must be nonempty",
            ));
        }
        Ok(Self { config })
    }

    /// Runs the sweep.
    pub fn run(&self) -> Result<CalibratedSweepReport> {
        let mut points = Vec::with_capacity(self.config.target_bers.len());
        for &target_ber in &self.config.target_bers {
            let point_config = CalibratedComparisonConfig {
                dimension: self.config.dimension,
                trials: self.config.trials,
                target_ber,
                seed: self.config.seed,
                calibration_dimension: self.config.calibration_dimension,
            };
            points.push(CalibratedComparisonRunner::new(point_config)?.run()?);
        }
        Ok(CalibratedSweepReport { points })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classical_channel_ber_is_identity() {
        assert_eq!(classical_channel_ber(0.13), 0.13);
        assert_eq!(classical_channel_ber(-1.0), 0.0);
        assert_eq!(classical_channel_ber(2.0), 1.0);
    }

    #[test]
    fn measured_classical_ber_matches_nominal_probability() {
        let ber = measure_classical_channel_ber(0.1, 16_384, 42).unwrap();
        assert!((ber - 0.1).abs() < 0.01, "measured_classical_ber={ber}");
    }

    #[test]
    fn phase_ber_is_monotonic_in_sigma() {
        // With the fixed nearest-symbol decode boundary (see phase_hdc.rs's
        // `to_binary_halfplane`), each symbol has a real ~PI/2 noise margin
        // before it can flip, so small sigma genuinely gives zero BER now
        // (unlike the pre-fix decode, which had ~zero margin).
        let none = measure_phase_channel_ber(0.5, 16_384, 7).unwrap();
        let low = measure_phase_channel_ber(1.5, 16_384, 7).unwrap();
        let mid = measure_phase_channel_ber(3.0, 16_384, 7).unwrap();
        let high = measure_phase_channel_ber(8.0, 16_384, 7).unwrap();
        assert_eq!(none, 0.0, "none={none} should be within the noise margin");
        assert!(low < mid, "low={low} mid={mid}");
        assert!(mid < high, "mid={mid} high={high}");
        assert!(high <= 0.51, "high={high} should approach chance (0.5)");
    }

    #[test]
    fn calibration_finds_matching_ber_out_of_sample() {
        // Calibrate against one seed, then re-measure with a *different* seed to
        // confirm the calibrated sigma generalizes rather than merely fitting the
        // calibration draw.
        for target in [0.02f32, 0.1, 0.25, 0.4] {
            let sigma = calibrate_phase_sigma_for_ber(target, 16_384, 1001).unwrap();
            let out_of_sample_ber = measure_phase_channel_ber(sigma, 16_384, 99_999).unwrap();
            assert!(
                (out_of_sample_ber - target).abs() < 0.01,
                "target={target} sigma={sigma} out_of_sample_ber={out_of_sample_ber}"
            );
        }
    }

    #[test]
    fn zero_target_ber_gives_zero_sigma() {
        assert_eq!(calibrate_phase_sigma_for_ber(0.0, 1024, 1).unwrap(), 0.0);
    }

    #[test]
    fn rejects_out_of_range_target_ber() {
        assert!(calibrate_phase_sigma_for_ber(0.5, 1024, 1).is_err());
        assert!(calibrate_phase_sigma_for_ber(0.6, 1024, 1).is_err());
        assert!(calibrate_phase_sigma_for_ber(-0.1, 1024, 1).is_err());
    }

    #[test]
    fn near_zero_ber_gives_near_perfect_matched_key_recovery() {
        let config = CalibratedComparisonConfig {
            dimension: 2048,
            trials: 8,
            target_ber: 0.0,
            seed: 5,
            calibration_dimension: 16_384,
        };
        let report = CalibratedComparisonRunner::new(config)
            .unwrap()
            .run()
            .unwrap();
        assert!(report.classical_recovery_accuracy.mean > 0.999);
        assert!(report.phase_recovery_accuracy.mean > 0.999);
    }

    #[test]
    fn wrong_key_recovery_is_near_chance_for_both_channels() {
        let config = CalibratedComparisonConfig {
            dimension: 2048,
            trials: 16,
            target_ber: 0.1,
            seed: 5,
            calibration_dimension: 16_384,
        };
        let report = CalibratedComparisonRunner::new(config)
            .unwrap()
            .run()
            .unwrap();
        assert!(
            (0.4..0.6).contains(&report.classical_wrong_key_accuracy.mean),
            "classical_wrong_key_mean={}",
            report.classical_wrong_key_accuracy.mean
        );
        assert!(
            (0.4..0.6).contains(&report.phase_wrong_key_accuracy.mean),
            "phase_wrong_key_mean={}",
            report.phase_wrong_key_accuracy.mean
        );
    }

    #[test]
    fn calibrated_comparison_no_longer_shows_the_uncalibrated_artifact() {
        // Before calibration, comparative_report's default config produced
        // classical_minus_phase_noisy_dz = -33.36 at noise=0.05 (see
        // docs/RESEARCH_NOTES.md). At a matched BER, the two channels should be
        // in the same ballpark, not off by an order of magnitude.
        let config = CalibratedComparisonConfig {
            dimension: 1024,
            trials: 32,
            target_ber: 0.05,
            seed: 11,
            calibration_dimension: 16_384,
        };
        let report = CalibratedComparisonRunner::new(config)
            .unwrap()
            .run()
            .unwrap();
        let classical_mean = report.classical_recovery_accuracy.mean;
        let phase_mean = report.phase_recovery_accuracy.mean;
        assert!(
            (classical_mean - phase_mean).abs() < 0.05,
            "classical_mean={classical_mean} phase_mean={phase_mean} \
             (should be close at matched BER, unlike the uncalibrated comparative_report)"
        );
    }

    #[test]
    fn sweep_runs_all_configured_points_in_order() {
        let config = CalibratedSweepConfig {
            dimension: 512,
            trials: 4,
            target_bers: vec![0.01, 0.1, 0.3],
            seed: 3,
            calibration_dimension: 8192,
        };
        let report = CalibratedSweepRunner::new(config).unwrap().run().unwrap();
        assert_eq!(report.points.len(), 3);
        let bers: Vec<f32> = report.points.iter().map(|p| p.target_ber).collect();
        assert_eq!(bers, vec![0.01, 0.1, 0.3]);
        let csv = report.to_csv();
        assert_eq!(csv.lines().count(), 4); // header + 3 rows
    }
}
