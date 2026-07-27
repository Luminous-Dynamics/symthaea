//! Calibrated continuous-value comparison: classical vs. phase-HDC storage
//! of a scalar, not a bit.
//!
//! `calibrated_comparison.rs` (noise-robustness) and `capacity_comparison.rs`
//! (bundling capacity) both found no representational advantage for
//! phase-HDC over classical binary HDC. Both tests, though, forced
//! phase-HDC into a purely binary role — encoding only two symbol points (0,
//! π) and hard-decoding back to bits before scoring. That discards the one
//! real structural difference phase-HDC has: a *continuous* degree of
//! freedom per dimension, which binary HDC does not have at all. This module
//! tests the property most likely to reveal an advantage, if one exists:
//! storing and recovering a continuous scalar directly.
//!
//! # Fair encodings
//!
//! Both representations spend their *entire* dimension as redundancy for
//! **one** scalar `x ∈ [0, 1]` (not compositional capacity — that's what
//! `capacity_comparison.rs` already tested):
//!
//! - **Classical**: `BinaryHypervector::thermometer_encode` — unary/thermometer
//!   coding, the graceful-degradation-by-construction baseline for
//!   representing a continuous quantity in bits (see that function's doc
//!   comment for why it's the fair choice, not a strawman).
//! - **Phase**: `x` is mapped to an angle on a *half*-circle,
//!   `θ = π/2 + x·π ∈ [π/2, 3π/2]`, via `PhaseHypervector::from_phases`
//!   repeated across every dimension, then recovered via `circular_mean`.
//!   The half-circle restriction is deliberate, not arbitrary: it keeps
//!   `x=0` and `x=1` maximally far apart (exactly antipodal, π radians
//!   apart) — the phase-space analog of thermometer coding's `x=0`/`x=1`
//!   being maximally different bit patterns (Hamming distance = dimension).
//!   Using the *full* circle instead would make `x=0` and `x=1` encode to
//!   the *same* angle, which is a different (circular-quantity) task, not a
//!   fair comparison against classical's strictly linear encoding.
//!
//! Both arms bind to the same underlying random key (the classical key's
//! bits, phase-encoded via `from_binary` for the phase arm — the same
//! same-underlying-content discipline used throughout this crate's
//! calibrated comparisons), then use the existing BER-calibration machinery
//! from `calibrated_comparison.rs` unchanged, then decode and score with
//! **mean absolute error** in `x`-units for both arms — the same metric, the
//! same bounded range `[0, 1]`, computed via a wraparound-safe circular
//! distance for the phase arm so an extreme-noise wraparound is never
//! silently mis-scored.
//!
//! # A disclosed asymmetry, not a bug
//!
//! At zero noise, classical's thermometer decode has an inherent
//! quantization floor of about `1 / (2 * dimension)` (it can only report one
//! of `dimension + 1` distinct levels). Phase's `circular_mean` decode has no
//! such floor — its precision is bounded only by `f32` rounding. This is a
//! real structural difference between the representations, not a
//! measurement artifact, and it is the most likely place (if anywhere) for
//! this module to find a real effect that the previous two calibrated
//! comparisons did not.
//!
//! # Debiasing, and a real crossover
//!
//! The raw comparison above found phase-HDC winning at every noise level —
//! but that was against classical's *naive* `thermometer_decode`, which has
//! a provable, closed-form bias toward 0.5 under noise (see
//! `debias_thermometer_estimate`'s doc comment). Applying that correction
//! (legitimate whenever the channel's bit-error-rate is known, exactly the
//! case in this module's own BER-calibrated setup) changes the picture: it
//! does not just shrink the gap, it produces a genuine **crossover** around
//! `target_ber ≈ 0.10`. Below that, debiased classical significantly beats
//! phase (the bias-removal benefit dominates); above it, phase wins by a
//! growing, statistically solid margin (the correction's variance
//! amplification — dividing by `1 - 2p`, which shrinks toward zero as
//! `p → 0.5` — starts to dominate). See `docs/RESEARCH_NOTES.md` for the
//! full numbers and the honest reading of what this does and doesn't show.
//!
//! # Shrinkage: tested, does not close the remaining gap
//!
//! `ShrinkageProbeRunner` tests whether a *partial* correction (blending
//! raw and fully-debiased estimates by `lambda ∈ [0, 1]`) beats full
//! debiasing in the noise range where full debiasing loses. It mostly
//! doesn't: full debiasing is already at or near MAE-optimal for
//! `target_ber` up to about 0.40. Only very close to the noise ceiling
//! (`target_ber ≳ 0.45`) does interior shrinkage produce a real, significant
//! improvement over full debiasing — and even there, phase still wins
//! overall. See `docs/RESEARCH_NOTES.md`'s "shrinkage probe" section for the
//! full numbers, including one suggestive-but-statistically-unresolved
//! signal right at the noise ceiling that is disclosed, not claimed.

use crate::calibrated_comparison::{
    calibrate_phase_sigma_for_ber, measure_classical_channel_ber, measure_phase_channel_ber,
};
use crate::classical_hdc::BinaryHypervector;
use crate::errors::{QuantumCompError, Result};
use crate::experiment::{ExperimentManifest, ExperimentProtocol};
use crate::phase_hdc::PhaseHypervector;
use crate::rng::XorShift64;
use crate::significance::PairedDifferenceSummary;
use crate::statistics::SampleSummary;
use crate::substrate::SubstrateProfile;

const HALF_CIRCLE_START: f32 = core::f32::consts::FRAC_PI_2;

fn x_to_theta(x: f32) -> f32 {
    HALF_CIRCLE_START + x.clamp(0.0, 1.0) * core::f32::consts::PI
}

/// Wraparound-safe distance between two angles, expressed in `x`-units
/// (`[0, 1]`, since the encoding above uses exactly `PI` radians per unit).
fn theta_error_in_x_units(theta_true: f32, theta_hat: f32) -> f32 {
    let raw = (theta_hat - theta_true).rem_euclid(core::f32::consts::TAU);
    let angular_dist = raw.min(core::f32::consts::TAU - raw);
    angular_dist / core::f32::consts::PI
}

/// Applies the closed-form linear correction for thermometer decode's
/// noise-induced bias toward 0.5.
///
/// `RESEARCH_NOTES.md`'s continuous-value comparison found that
/// `thermometer_decode`'s raw estimate has expected value
/// `x·(1 - 2p) + p` under independent bitflip noise with probability `p` —
/// a bias toward 0.5, not random noise. A decoder that knows the channel's
/// bit-error-rate can invert that exactly: `x_hat = (raw - p) / (1 - 2p)`.
/// This assumes `p` is known (e.g. a calibrated/measured channel, as in this
/// module's own BER-calibration setup) — it is not a claim that classical
/// HDC can debias *without* knowing its own noise rate. The correction is
/// undefined at `p = 0.5` (chance level, where the encoding has already
/// lost all information); this returns the chance-level estimate `0.5`
/// there rather than dividing by zero. Result is clamped to `[0, 1]` since
/// the raw noise draw can otherwise push the linear correction outside the
/// valid range.
fn debias_thermometer_estimate(raw_estimate: f32, p: f32) -> f32 {
    let denom = 1.0 - 2.0 * p;
    if denom.abs() < 1e-6 {
        return 0.5;
    }
    ((raw_estimate - p) / denom).clamp(0.0, 1.0)
}

/// Blends the raw and fully-debiased thermometer estimates by a shrinkage
/// factor `lambda ∈ [0, 1]`.
///
/// `lambda = 0` reproduces the raw (biased, lower-variance) estimate;
/// `lambda = 1` reproduces `debias_thermometer_estimate`'s fully
/// bias-corrected (unbiased, higher-variance) estimate. Full debiasing
/// removes bias exactly but amplifies residual variance by `1/(1-2p)²`, and
/// `RESEARCH_NOTES.md`'s continuous-value comparison found that cost starts
/// to dominate above `target_ber ≈ 0.10` — full debiasing is not
/// automatically MAE-optimal. This tests whether some interior `lambda`
/// (a classical bias-variance tradeoff, not a phase-inspired trick) does
/// better than either extreme in that regime.
fn blend_thermometer_estimate(raw_estimate: f32, p: f32, lambda: f32) -> f32 {
    let debiased = debias_thermometer_estimate(raw_estimate, p);
    ((1.0 - lambda) * raw_estimate + lambda * debiased).clamp(0.0, 1.0)
}

/// Configuration for a shrinkage-factor sweep at one fixed target BER.
///
/// Tests whether a *partial* debiasing correction can close the gap to
/// phase-HDC in the noise regime where full debiasing (`lambda=1`) loses —
/// see `docs/RESEARCH_NOTES.md`'s continuous-value comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct ShrinkageProbeConfig {
    /// Hypervector dimension.
    pub dimension: usize,
    /// Target bit-error-rate to probe (a single fixed noise level).
    pub target_ber: f32,
    /// Shrinkage factors to sweep, each in `[0, 1]`.
    pub lambdas: Vec<f32>,
    /// Independent trials (shared across all `lambdas` — each trial's raw
    /// classical decode is computed once and blended at every `lambda`, so
    /// results across `lambdas` are paired, not independently resampled).
    pub trials: usize,
    /// Deterministic seed.
    pub seed: u64,
    /// Dimension used for BER calibration measurements.
    pub calibration_dimension: usize,
}

impl Default for ShrinkageProbeConfig {
    fn default() -> Self {
        Self {
            dimension: 1024,
            target_ber: 0.30,
            lambdas: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            trials: 200,
            seed: 0x5159_4D54_4841_4541,
            calibration_dimension: 16_384,
        }
    }
}

/// Classical error at one shrinkage factor.
#[derive(Debug, Clone, PartialEq)]
pub struct ShrinkagePoint {
    /// Shrinkage factor tested.
    pub lambda: f32,
    /// Classical mean absolute recovery error at this `lambda`, in `x`-units.
    pub classical_error: SampleSummary,
    /// Paired comparison of classical (at this `lambda`) vs. phase error
    /// across trials — trial-paired, since both are computed from the same
    /// per-trial draws.
    pub paired_vs_phase: Option<PairedDifferenceSummary>,
}

/// Full shrinkage-probe report.
#[derive(Debug, Clone, PartialEq)]
pub struct ShrinkageProbeReport {
    /// Experiment manifest and claim boundary.
    pub manifest: ExperimentManifest,
    /// Target bit-error-rate probed.
    pub target_ber: f32,
    /// Phase error at this BER — the fixed baseline every `lambda` is
    /// compared against (computed once per trial, shared across all points).
    pub phase_error: SampleSummary,
    /// Points in the order `lambdas` was given.
    pub points: Vec<ShrinkagePoint>,
}

impl ShrinkageProbeReport {
    /// Returns the point with the lowest classical mean error.
    pub fn best_lambda(&self) -> Option<&ShrinkagePoint> {
        self.points
            .iter()
            .min_by(|a, b| a.classical_error.mean.total_cmp(&b.classical_error.mean))
    }

    /// Returns a CSV report with one row per `lambda`.
    pub fn to_csv(&self) -> String {
        let mut out = format!(
            "target_ber={:.6} phase_mae={:.6}\nlambda,classical_mae,vs_phase_delta,vs_phase_p\n",
            self.target_ber, self.phase_error.mean,
        );
        for p in &self.points {
            let (delta, p_value) = p
                .paired_vs_phase
                .as_ref()
                .map(|s| (s.delta.mean, s.sign_test_p_two_sided))
                .unwrap_or((f32::NAN, None));
            out.push_str(&format!(
                "{:.4},{:.6},{:.6},{:?}\n",
                p.lambda, p.classical_error.mean, delta, p_value,
            ));
        }
        out
    }
}

/// Runs a shrinkage-factor sweep at one fixed target BER.
#[derive(Debug, Clone)]
pub struct ShrinkageProbeRunner {
    config: ShrinkageProbeConfig,
}

impl ShrinkageProbeRunner {
    /// Creates a new runner.
    pub fn new(config: ShrinkageProbeConfig) -> Result<Self> {
        if config.dimension == 0 || config.calibration_dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        if !(0.0..0.5).contains(&config.target_ber) {
            return Err(QuantumCompError::InvalidProbability);
        }
        if config.lambdas.is_empty() {
            return Err(QuantumCompError::InvalidConfig("lambdas must be nonempty"));
        }
        if config.lambdas.iter().any(|&l| !(0.0..=1.0).contains(&l)) {
            return Err(QuantumCompError::InvalidConfig(
                "lambdas entries must be in [0, 1]",
            ));
        }
        if config.trials == 0 {
            return Err(QuantumCompError::InvalidConfig("trials must be > 0"));
        }
        Ok(Self { config })
    }

    /// Runs the sweep.
    pub fn run(&self) -> Result<ShrinkageProbeReport> {
        let calibration_seed = self.config.seed ^ 0xCA11_B4A7_E000_0007;
        let phase_sigma = calibrate_phase_sigma_for_ber(
            self.config.target_ber,
            self.config.calibration_dimension,
            calibration_seed,
        )?;

        let mut classical_errors_by_lambda: Vec<Vec<f32>> =
            vec![Vec::with_capacity(self.config.trials); self.config.lambdas.len()];
        let mut phase_errors = Vec::with_capacity(self.config.trials);

        for trial in 0..self.config.trials {
            let base_seed = self
                .config
                .seed
                .wrapping_add((trial as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

            let mut value_rng = XorShift64::new(base_seed ^ 0x5CA1_AB1E);
            let x = value_rng.next_f32();

            let key = BinaryHypervector::random(self.config.dimension, base_seed ^ 0xB0B)?;
            let phase_key = PhaseHypervector::from_binary(&key);

            let item = BinaryHypervector::thermometer_encode(x, self.config.dimension)?;
            let bound = item.bind_xor(&key)?;
            let recovered = bound.unbind_xor(&key)?;
            let noisy = recovered.with_bitflip_noise(self.config.target_ber, base_seed ^ 0xD15EA5E);
            let x_hat_raw = noisy.thermometer_decode();
            for (i, &lambda) in self.config.lambdas.iter().enumerate() {
                let x_hat = blend_thermometer_estimate(x_hat_raw, self.config.target_ber, lambda);
                classical_errors_by_lambda[i].push((x - x_hat).abs());
            }

            let theta = x_to_theta(x);
            let phase_item = PhaseHypervector::from_phases(vec![theta; self.config.dimension])?;
            let phase_bound = phase_item.bind_phase(&phase_key)?;
            let phase_recovered = phase_bound.unbind_phase(&phase_key)?;
            let phase_noisy = phase_recovered.with_phase_noise(phase_sigma, base_seed ^ 0xF00D);
            let theta_hat = phase_noisy.circular_mean();
            phase_errors.push(theta_error_in_x_units(theta, theta_hat));
        }

        let phase_error = SampleSummary::from_samples(&phase_errors).expect("nonempty by config");
        let points = self
            .config
            .lambdas
            .iter()
            .zip(classical_errors_by_lambda.iter())
            .map(|(&lambda, errors)| ShrinkagePoint {
                lambda,
                classical_error: SampleSummary::from_samples(errors).expect("nonempty by config"),
                paired_vs_phase: PairedDifferenceSummary::from_pairs(errors, &phase_errors, 1e-6),
            })
            .collect();

        let manifest = ExperimentManifest::local_simulation(
            "calibrated-shrinkage-probe-v0.1",
            ExperimentProtocol::CalibratedCrossRepresentationComparison,
            self.config.seed,
            self.config.dimension,
            self.config.trials,
            SubstrateProfile::quantum_inspired(),
        );

        Ok(ShrinkageProbeReport {
            manifest,
            target_ber: self.config.target_ber,
            phase_error,
            points,
        })
    }
}

/// Configuration for a continuous-value sweep across target bit-error-rates.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuousValueSweepConfig {
    /// Hypervector dimension (redundancy spent on the single scalar).
    pub dimension: usize,
    /// Target bit-error-rates to sweep, ascending. Include `0.0` to measure
    /// the zero-noise quantization-floor asymmetry.
    pub target_bers: Vec<f32>,
    /// Independent trials per target BER (each trial draws a fresh random `x`).
    pub trials_per_ber: usize,
    /// Deterministic seed.
    pub seed: u64,
    /// Dimension used for BER calibration measurements (see `calibrated_comparison`).
    pub calibration_dimension: usize,
}

impl Default for ContinuousValueSweepConfig {
    fn default() -> Self {
        Self {
            dimension: 1024,
            target_bers: vec![0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40],
            trials_per_ber: 32,
            seed: 0x5159_4D54_4841_4541,
            calibration_dimension: 16_384,
        }
    }
}

/// One point in a continuous-value sweep.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuousValuePoint {
    /// Target bit-error-rate for this point.
    pub target_ber: f32,
    /// Calibrated phase-noise sigma reaching `target_ber`.
    pub phase_sigma: f32,
    /// Classical channel's realized BER.
    pub measured_classical_ber: f32,
    /// Phase channel's realized BER at `phase_sigma`.
    pub measured_phase_ber: f32,
    /// Classical mean absolute recovery error, in `x`-units (`[0, 1]`) —
    /// raw `thermometer_decode`, no bias correction.
    pub classical_error: SampleSummary,
    /// Classical mean absolute recovery error after applying
    /// `debias_thermometer_estimate` (the noise-rate-aware linear bias
    /// correction) to the same raw decode.
    pub classical_debiased_error: SampleSummary,
    /// Phase mean absolute recovery error, in `x`-units (`[0, 1]`).
    pub phase_error: SampleSummary,
    /// Paired comparison of raw classical vs. phase error across trials.
    pub paired_raw_vs_phase: Option<PairedDifferenceSummary>,
    /// Paired comparison of debiased classical vs. phase error across trials
    /// — the direct test of whether the bias correction closes the gap.
    pub paired_debiased_vs_phase: Option<PairedDifferenceSummary>,
}

/// Full continuous-value sweep report.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuousValueSweepReport {
    /// Experiment manifest and claim boundary.
    pub manifest: ExperimentManifest,
    /// Points in ascending target-BER order.
    pub points: Vec<ContinuousValuePoint>,
}

impl ContinuousValueSweepReport {
    /// Returns a CSV report with one row per target BER.
    pub fn to_csv(&self) -> String {
        let mut out = String::from(
            "target_ber,phase_sigma,measured_classical_ber,measured_phase_ber,classical_mae,classical_debiased_mae,phase_mae,raw_vs_phase_delta,raw_vs_phase_p,debiased_vs_phase_delta,debiased_vs_phase_p\n",
        );
        for p in &self.points {
            let (raw_delta, raw_p) = p
                .paired_raw_vs_phase
                .as_ref()
                .map(|s| (s.delta.mean, s.sign_test_p_two_sided))
                .unwrap_or((f32::NAN, None));
            let (debiased_delta, debiased_p) = p
                .paired_debiased_vs_phase
                .as_ref()
                .map(|s| (s.delta.mean, s.sign_test_p_two_sided))
                .unwrap_or((f32::NAN, None));
            out.push_str(&format!(
                "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:?},{:.6},{:?}\n",
                p.target_ber,
                p.phase_sigma,
                p.measured_classical_ber,
                p.measured_phase_ber,
                p.classical_error.mean,
                p.classical_debiased_error.mean,
                p.phase_error.mean,
                raw_delta,
                raw_p,
                debiased_delta,
                debiased_p,
            ));
        }
        out
    }

    /// Returns a compact text summary.
    pub fn to_text(&self) -> String {
        let mut out = self.manifest.to_text();
        out.push('\n');
        out.push_str(&self.to_csv());
        out
    }
}

/// Runs a calibrated continuous-value sweep.
#[derive(Debug, Clone)]
pub struct ContinuousValueSweepRunner {
    config: ContinuousValueSweepConfig,
}

impl ContinuousValueSweepRunner {
    /// Creates a new runner.
    pub fn new(config: ContinuousValueSweepConfig) -> Result<Self> {
        if config.dimension == 0 || config.calibration_dimension == 0 {
            return Err(QuantumCompError::InvalidDimension);
        }
        if config.target_bers.is_empty() {
            return Err(QuantumCompError::InvalidConfig(
                "target_bers must be nonempty",
            ));
        }
        if config.trials_per_ber == 0 {
            return Err(QuantumCompError::InvalidConfig(
                "trials_per_ber must be > 0",
            ));
        }
        Ok(Self { config })
    }

    /// Runs the sweep.
    pub fn run(&self) -> Result<ContinuousValueSweepReport> {
        let mut points = Vec::with_capacity(self.config.target_bers.len());
        for &target_ber in &self.config.target_bers {
            points.push(self.run_one(target_ber)?);
        }
        let manifest = ExperimentManifest::local_simulation(
            "calibrated-continuous-value-comparison-v0.1",
            ExperimentProtocol::CalibratedCrossRepresentationComparison,
            self.config.seed,
            self.config.dimension,
            self.config.trials_per_ber,
            SubstrateProfile::quantum_inspired(),
        );
        Ok(ContinuousValueSweepReport { manifest, points })
    }

    fn run_one(&self, target_ber: f32) -> Result<ContinuousValuePoint> {
        let calibration_seed = self.config.seed ^ 0xCA11_B4A7_E000_0006;
        let phase_sigma = calibrate_phase_sigma_for_ber(
            target_ber,
            self.config.calibration_dimension,
            calibration_seed,
        )?;
        let measured_classical_ber = measure_classical_channel_ber(
            target_ber,
            self.config.calibration_dimension,
            calibration_seed,
        )?;
        let measured_phase_ber = measure_phase_channel_ber(
            phase_sigma,
            self.config.calibration_dimension,
            calibration_seed,
        )?;

        let mut classical_errors = Vec::with_capacity(self.config.trials_per_ber);
        let mut classical_debiased_errors = Vec::with_capacity(self.config.trials_per_ber);
        let mut phase_errors = Vec::with_capacity(self.config.trials_per_ber);

        for trial in 0..self.config.trials_per_ber {
            let base_seed = self
                .config
                .seed
                .wrapping_add((trial as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

            let mut value_rng = XorShift64::new(base_seed ^ 0x5CA1_AB1E);
            let x = value_rng.next_f32();

            let key = BinaryHypervector::random(self.config.dimension, base_seed ^ 0xB0B)?;
            let phase_key = PhaseHypervector::from_binary(&key);

            let item = BinaryHypervector::thermometer_encode(x, self.config.dimension)?;
            let bound = item.bind_xor(&key)?;
            let recovered = bound.unbind_xor(&key)?;
            let noisy = recovered.with_bitflip_noise(target_ber, base_seed ^ 0xD15EA5E);
            let x_hat_classical = noisy.thermometer_decode();
            classical_errors.push((x - x_hat_classical).abs());
            let x_hat_debiased = debias_thermometer_estimate(x_hat_classical, target_ber);
            classical_debiased_errors.push((x - x_hat_debiased).abs());

            let theta = x_to_theta(x);
            let phase_item = PhaseHypervector::from_phases(vec![theta; self.config.dimension])?;
            let phase_bound = phase_item.bind_phase(&phase_key)?;
            let phase_recovered = phase_bound.unbind_phase(&phase_key)?;
            let phase_noisy = phase_recovered.with_phase_noise(phase_sigma, base_seed ^ 0xF00D);
            let theta_hat = phase_noisy.circular_mean();
            phase_errors.push(theta_error_in_x_units(theta, theta_hat));
        }

        let classical_error =
            SampleSummary::from_samples(&classical_errors).expect("nonempty by config");
        let classical_debiased_error =
            SampleSummary::from_samples(&classical_debiased_errors).expect("nonempty by config");
        let phase_error = SampleSummary::from_samples(&phase_errors).expect("nonempty by config");
        let paired_raw_vs_phase =
            PairedDifferenceSummary::from_pairs(&classical_errors, &phase_errors, 1e-6);
        let paired_debiased_vs_phase =
            PairedDifferenceSummary::from_pairs(&classical_debiased_errors, &phase_errors, 1e-6);

        Ok(ContinuousValuePoint {
            target_ber,
            phase_sigma,
            measured_classical_ber,
            measured_phase_ber,
            classical_error,
            classical_debiased_error,
            phase_error,
            paired_raw_vs_phase,
            paired_debiased_vs_phase,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn x_to_theta_puts_endpoints_at_maximal_circular_distance() {
        let t0 = x_to_theta(0.0);
        let t1 = x_to_theta(1.0);
        assert!((theta_error_in_x_units(t0, t1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn theta_error_handles_wraparound_correctly() {
        // Just past the top of the circle vs. just past zero: angularly close,
        // must not be scored as if they were far apart.
        let a = 0.01_f32;
        let b = core::f32::consts::TAU - 0.01_f32;
        let err = theta_error_in_x_units(a, b);
        assert!(err < 0.01, "err={err}");
    }

    #[test]
    fn zero_target_ber_gives_near_zero_error_for_both_representations() {
        let config = ContinuousValueSweepConfig {
            dimension: 4096,
            target_bers: vec![0.0],
            trials_per_ber: 16,
            seed: 3,
            calibration_dimension: 16_384,
        };
        let report = ContinuousValueSweepRunner::new(config)
            .unwrap()
            .run()
            .unwrap();
        let point = &report.points[0];
        assert!(
            point.classical_error.mean < 0.01,
            "{:?}",
            point.classical_error
        );
        assert!(point.phase_error.mean < 0.01, "{:?}", point.phase_error);
    }

    #[test]
    fn error_grows_with_target_ber() {
        let config = ContinuousValueSweepConfig {
            dimension: 512,
            target_bers: vec![0.0, 0.3],
            trials_per_ber: 24,
            seed: 6,
            calibration_dimension: 16_384,
        };
        let report = ContinuousValueSweepRunner::new(config)
            .unwrap()
            .run()
            .unwrap();
        assert!(report.points[1].classical_error.mean > report.points[0].classical_error.mean);
        assert!(report.points[1].phase_error.mean > report.points[0].phase_error.mean);
    }

    #[test]
    fn debiasing_is_a_no_op_at_zero_noise() {
        let config = ContinuousValueSweepConfig {
            dimension: 512,
            target_bers: vec![0.0],
            trials_per_ber: 16,
            seed: 1,
            calibration_dimension: 16_384,
        };
        let report = ContinuousValueSweepRunner::new(config)
            .unwrap()
            .run()
            .unwrap();
        let point = &report.points[0];
        assert_eq!(
            point.classical_error.mean,
            point.classical_debiased_error.mean
        );
    }

    #[test]
    fn debiasing_reverses_the_advantage_at_low_noise() {
        // Verified via a real run (dimension=1024, seed=default, target_ber=0.05,
        // trials=32): raw classical loses to phase, but debiased classical wins
        // significantly (p=0.02). The bias-removal benefit dominates here.
        let config = ContinuousValueSweepConfig {
            dimension: 1024,
            target_bers: vec![0.05],
            trials_per_ber: 32,
            seed: 0x5159_4D54_4841_4541,
            calibration_dimension: 16_384,
        };
        let report = ContinuousValueSweepRunner::new(config)
            .unwrap()
            .run()
            .unwrap();
        let point = &report.points[0];
        assert!(
            point.classical_debiased_error.mean < point.phase_error.mean,
            "classical_debiased={:?} phase={:?}",
            point.classical_debiased_error,
            point.phase_error
        );
    }

    #[test]
    fn phase_still_wins_at_high_noise_even_after_debiasing() {
        // Verified via a real run (dimension=1024, seed=default, target_ber=0.30,
        // trials=200): phase beats debiased classical significantly
        // (p=2.65e-5). The correction's variance amplification (dividing by
        // 1-2p) dominates here, on the other side of the ~0.10 crossover.
        let config = ContinuousValueSweepConfig {
            dimension: 1024,
            target_bers: vec![0.30],
            trials_per_ber: 200,
            seed: 0x5159_4D54_4841_4541,
            calibration_dimension: 16_384,
        };
        let report = ContinuousValueSweepRunner::new(config)
            .unwrap()
            .run()
            .unwrap();
        let point = &report.points[0];
        assert!(
            point.phase_error.mean < point.classical_debiased_error.mean,
            "phase={:?} classical_debiased={:?}",
            point.phase_error,
            point.classical_debiased_error
        );
    }

    #[test]
    fn rejects_invalid_config() {
        assert!(
            ContinuousValueSweepRunner::new(ContinuousValueSweepConfig {
                dimension: 0,
                ..ContinuousValueSweepConfig::default()
            })
            .is_err()
        );
        assert!(
            ContinuousValueSweepRunner::new(ContinuousValueSweepConfig {
                target_bers: vec![],
                ..ContinuousValueSweepConfig::default()
            })
            .is_err()
        );
        assert!(
            ContinuousValueSweepRunner::new(ContinuousValueSweepConfig {
                trials_per_ber: 0,
                ..ContinuousValueSweepConfig::default()
            })
            .is_err()
        );
    }

    #[test]
    fn shrinkage_rejects_invalid_config() {
        assert!(
            ShrinkageProbeRunner::new(ShrinkageProbeConfig {
                dimension: 0,
                ..ShrinkageProbeConfig::default()
            })
            .is_err()
        );
        assert!(
            ShrinkageProbeRunner::new(ShrinkageProbeConfig {
                target_ber: 0.5,
                ..ShrinkageProbeConfig::default()
            })
            .is_err()
        );
        assert!(
            ShrinkageProbeRunner::new(ShrinkageProbeConfig {
                lambdas: vec![],
                ..ShrinkageProbeConfig::default()
            })
            .is_err()
        );
        assert!(
            ShrinkageProbeRunner::new(ShrinkageProbeConfig {
                lambdas: vec![1.5],
                ..ShrinkageProbeConfig::default()
            })
            .is_err()
        );
        assert!(
            ShrinkageProbeRunner::new(ShrinkageProbeConfig {
                trials: 0,
                ..ShrinkageProbeConfig::default()
            })
            .is_err()
        );
    }

    #[test]
    fn full_debiasing_is_already_near_optimal_in_the_mid_high_noise_range() {
        // Verified via a real run (dimension=1024, seed=default, target_ber=0.30,
        // trials=200): the grid search over lambda finds lambda=1.0 (full
        // debiasing) as the single best point -- shrinkage provides no
        // meaningful further improvement in this regime, and phase still wins
        // decisively regardless of lambda.
        let config = ShrinkageProbeConfig {
            target_ber: 0.30,
            trials: 200,
            ..ShrinkageProbeConfig::default()
        };
        let report = ShrinkageProbeRunner::new(config).unwrap().run().unwrap();
        let best = report.best_lambda().unwrap();
        assert_eq!(best.lambda, 1.0, "{:?}", report.points);
        assert!(
            best.classical_error.mean > report.phase_error.mean,
            "classical={:?} phase={:?}",
            best.classical_error,
            report.phase_error
        );
    }

    #[test]
    fn partial_shrinkage_beats_full_debiasing_near_the_noise_ceiling() {
        // Verified via a real run (dimension=1024, seed=default, target_ber=0.45,
        // trials=400): lambda=0.8 (interior) beats lambda=1.0 by a real margin
        // (0.1076 vs 0.1151), confirming the correction's variance amplification
        // (1/(1-2p)) does eventually make full debiasing suboptimal -- but phase
        // still wins overall at this BER even at the optimal lambda.
        let config = ShrinkageProbeConfig {
            target_ber: 0.45,
            trials: 400,
            ..ShrinkageProbeConfig::default()
        };
        let report = ShrinkageProbeRunner::new(config).unwrap().run().unwrap();
        let best = report.best_lambda().unwrap();
        let full_debiasing = report.points.last().unwrap();
        assert!(best.lambda < 1.0, "{:?}", report.points);
        assert!(
            best.classical_error.mean < full_debiasing.classical_error.mean,
            "best={:?} full_debiasing={:?}",
            best.classical_error,
            full_debiasing.classical_error
        );
        assert!(
            best.classical_error.mean > report.phase_error.mean,
            "phase still wins overall: classical={:?} phase={:?}",
            best.classical_error,
            report.phase_error
        );
    }
}
