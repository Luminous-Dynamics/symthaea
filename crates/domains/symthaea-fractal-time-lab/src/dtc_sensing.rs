// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Finite closed-system AC-probe sensing diagnostics for kicked-Ising Floquet models.
//!
//! The model is a sampled, piecewise-constant longitudinal probe kick coupled
//! to the total Z magnetization. At the final stroboscopic time, the readout is
//! a projective measurement of total computational-basis magnetization. The
//! classical Fisher information is computed from the exact distribution over
//! Hamming-weight outcomes, not from an independent-spin/binomial assumption.
//!
//! This is a small-system mechanism benchmark. It is not a continuous-time
//! simulation of the dipolar 13C diamond experiment, does not include
//! inhomogeneous couplings, bath dynamics, NV readout, spin preparation cost,
//! dead time, measurement noise or calibration drift. A result is information
//! for the declared ideal measurement model, not a device sensitivity claim.

use crate::exact_floquet::{
    FloquetError, KickedIsingFloquet, SubharmonicResponse, analyze_subharmonic_response,
};
use std::error::Error;
use std::fmt;

/// Maximum number of stroboscopic drive periods accepted by the small-system
/// sensing reference to bound accidental unbounded simulation requests.
pub const MAX_DTC_PROBE_PERIODS: usize = 10_000;

/// Configuration for a finite exact-Floquet AC-probe sensing experiment.
#[derive(Clone, Debug, PartialEq)]
pub struct DtcProbeConfig {
    pub n_spins: usize,
    /// Uniform nearest-neighbour ZZ interaction in hbar=1, period=1 units.
    pub nearest_neighbor_coupling: f64,
    pub longitudinal_fields: Vec<f64>,
    pub pulse_angle_rad: f64,
    pub periods: usize,
    pub analysis_start: usize,
    pub probe_frequency_cycles_per_drive_period: f64,
    pub probe_initial_phase_rad: f64,
    /// Dimensionless longitudinal probe-kick amplitude A multiplying sum_i Z_i.
    pub probe_amplitude: f64,
    /// Finite-difference displacement used to estimate d p(outcome) / d A.
    pub amplitude_derivative_step: f64,
    /// Number of early/rolling samples used to measure subharmonic lifetime.
    pub lifetime_window: usize,
    /// Lifetime is the first window whose mean absolute demodulated order is
    /// below this fraction of the initial reference window.
    pub lifetime_threshold_fraction: f64,
}

impl DtcProbeConfig {
    pub fn validate(&self) -> Result<(), DtcProbeError> {
        let lifetime_end = self
            .analysis_start
            .checked_add(self.lifetime_window)
            .ok_or(DtcProbeError::InvalidConfiguration)?;
        if !(1..=20).contains(&self.n_spins)
            || self.longitudinal_fields.len() != self.n_spins
            || !self.nearest_neighbor_coupling.is_finite()
            || !self.pulse_angle_rad.is_finite()
            || self.longitudinal_fields.iter().any(|value| !value.is_finite())
            || self.periods < 32
            || self.periods > MAX_DTC_PROBE_PERIODS
            || self.analysis_start >= self.periods
            || self.periods - self.analysis_start < 32
            || !self.probe_frequency_cycles_per_drive_period.is_finite()
            || !(0.0..=0.5).contains(&self.probe_frequency_cycles_per_drive_period)
            || !self.probe_initial_phase_rad.is_finite()
            || !self.probe_amplitude.is_finite()
            || !self.amplitude_derivative_step.is_finite()
            || self.amplitude_derivative_step <= 0.0
            || self.lifetime_window < 2
            || lifetime_end >= self.periods
            || !self.lifetime_threshold_fraction.is_finite()
            || !(0.0..1.0).contains(&self.lifetime_threshold_fraction)
        {
            return Err(DtcProbeError::InvalidConfiguration);
        }
        Ok(())
    }
}

/// Errors in the declared finite-Floquet sensing model or its readout analysis.
#[derive(Clone, Debug, PartialEq)]
pub enum DtcProbeError {
    InvalidConfiguration,
    InvalidProbabilityDistribution,
    SingularReadoutSupport {
        outcome_down_spins: usize,
    },
    NonFiniteFisherInformation,
    Floquet(FloquetError),
}

impl fmt::Display for DtcProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration => {
                write!(f, "finite-Floquet probe-sensing configuration is invalid")
            }
            Self::InvalidProbabilityDistribution => {
                write!(f, "computational-basis readout probabilities are not normalized")
            }
            Self::SingularReadoutSupport { outcome_down_spins } => write!(
                f,
                "readout outcome {outcome_down_spins} has zero nominal probability but nonzero local derivative"
            ),
            Self::NonFiniteFisherInformation => {
                write!(f, "classical Fisher information is not finite and non-negative")
            }
            Self::Floquet(error) => write!(f, "Floquet reference failed: {error}"),
        }
    }
}

impl Error for DtcProbeError {}

impl From<FloquetError> for DtcProbeError {
    fn from(value: FloquetError) -> Self {
        Self::Floquet(value)
    }
}

/// Subharmonic lifetime estimate with explicit right-censoring.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SubharmonicLifetime {
    /// The demodulated signal crossed the threshold at this stroboscopic period.
    Observed {
        first_crossing_period: usize,
        reference_order: f64,
        threshold_order: f64,
    },
    /// No crossing occurred during the simulated horizon.
    RightCensored {
        horizon_period: usize,
        reference_order: f64,
        threshold_order: f64,
    },
    /// The initial reference window had no measurable demodulated order.
    NoInitialOrder,
}

/// Estimate the lifetime from a rolling mean-absolute demodulated order.
///
/// The first window after analysis_start sets the reference. The lifetime is
/// the first later window whose mean absolute (-1)^n m_z(n) falls below the
/// declared fraction of that reference. If it never crosses, the result is
/// right-censored at the simulation horizon rather than reported as infinite.
pub fn estimate_subharmonic_lifetime(
    signal: &[f64],
    analysis_start: usize,
    window: usize,
    threshold_fraction: f64,
) -> Option<SubharmonicLifetime> {
    if window < 2
        || analysis_start.checked_add(window)? >= signal.len()
        || !threshold_fraction.is_finite()
        || !(0.0..1.0).contains(&threshold_fraction)
        || signal.iter().any(|value| !value.is_finite())
    {
        return None;
    }

    let demodulated_abs = |start: usize| -> f64 {
        let sum = (start..start + window)
            .map(|period| signal[period].abs())
            .sum::<f64>();
        sum / window as f64
    };
    let reference = demodulated_abs(analysis_start);
    if !reference.is_finite() || reference <= f64::EPSILON {
        return Some(SubharmonicLifetime::NoInitialOrder);
    }
    let threshold = reference * threshold_fraction;

    let last_start = signal.len() - window;
    for start in (analysis_start + 1)..=last_start {
        let order = demodulated_abs(start);
        if order < threshold {
            return Some(SubharmonicLifetime::Observed {
                first_crossing_period: start,
                reference_order: reference,
                threshold_order: threshold,
            });
        }
    }

    Some(SubharmonicLifetime::RightCensored {
        horizon_period: signal.len() - 1,
        reference_order: reference,
        threshold_order: threshold,
    })
}

/// Probability distribution for one total-magnetization measurement.
///
/// probability_by_down_spins[k] is the probability that exactly k spins are
/// measured in Z=-1. This is a collective readout with N+1 outcomes; no
/// independent-spin approximation is applied.
#[derive(Clone, Debug, PartialEq)]
pub struct MagnetizationReadout {
    pub probability_by_down_spins: Vec<f64>,
    pub mean_magnetization: f64,
    pub normalization_error: f64,
}

/// Ideal finite-system sensing result for one declared Floquet experiment.
#[derive(Clone, Debug, PartialEq)]
pub struct DtcProbeSensingReport {
    pub n_spins: usize,
    pub periods: usize,
    pub probe_frequency_cycles_per_drive_period: f64,
    pub probe_amplitude: f64,
    pub probe_initial_phase_rad: f64,
    pub baseline_subharmonic_response: SubharmonicResponse,
    pub probed_subharmonic_response: SubharmonicResponse,
    pub baseline_lifetime: SubharmonicLifetime,
    pub probed_lifetime: SubharmonicLifetime,
    pub probe_readout: MagnetizationReadout,
    /// Local classical Fisher information for the total-Z readout, per shot.
    /// Units are inverse squared dimensionless probe-kick amplitude.
    pub classical_fisher_information_per_shot: f64,
    /// Fisher information divided by simulated Floquet periods. This omits
    /// state preparation, measurement and dead-time costs.
    pub fisher_information_per_drive_period: f64,
    pub norm_error: f64,
}

/// Run baseline, nominal-probe and +/-delta probe simulations, then compute
/// the local classical Fisher information of a final total-Z measurement.
///
/// The derivative is a centered finite difference in probe-kick amplitude.
/// All runs use the same initial state, Hamiltonian, drive, frequency, phase,
/// duration and readout. Only the probe amplitude changes.
pub fn evaluate_dtc_probe_sensing(
    config: &DtcProbeConfig,
) -> Result<DtcProbeSensingReport, DtcProbeError> {
    config.validate()?;

    let mut baseline_model = build_model(config)?;
    let baseline_signal = baseline_model.stroboscopic_magnetization(config.periods);
    let baseline_response =
        analyze_subharmonic_response(&baseline_signal, config.analysis_start)
            .ok_or(DtcProbeError::InvalidConfiguration)?;
    let baseline_lifetime = estimate_subharmonic_lifetime(
        &baseline_signal,
        config.analysis_start,
        config.lifetime_window,
        config.lifetime_threshold_fraction,
    )
    .ok_or(DtcProbeError::InvalidConfiguration)?;

    let mut nominal_model = build_model(config)?;
    let probed_signal = nominal_model.stroboscopic_magnetization_with_uniform_ac_probe(
        config.periods,
        config.probe_amplitude,
        config.probe_frequency_cycles_per_drive_period,
        config.probe_initial_phase_rad,
    )?;
    let probed_response = analyze_subharmonic_response(&probed_signal, config.analysis_start)
        .ok_or(DtcProbeError::InvalidConfiguration)?;
    let probed_lifetime = estimate_subharmonic_lifetime(
        &probed_signal,
        config.analysis_start,
        config.lifetime_window,
        config.lifetime_threshold_fraction,
    )
    .ok_or(DtcProbeError::InvalidConfiguration)?;
    let probe_readout = readout_distribution(&nominal_model);
    validate_readout(&probe_readout)?;

    let delta = config.amplitude_derivative_step;
    let mut plus_model = build_model(config)?;
    plus_model.stroboscopic_magnetization_with_uniform_ac_probe(
        config.periods,
        config.probe_amplitude + delta,
        config.probe_frequency_cycles_per_drive_period,
        config.probe_initial_phase_rad,
    )?;
    let plus_readout = readout_distribution(&plus_model);
    validate_readout(&plus_readout)?;

    let mut minus_model = build_model(config)?;
    minus_model.stroboscopic_magnetization_with_uniform_ac_probe(
        config.periods,
        config.probe_amplitude - delta,
        config.probe_frequency_cycles_per_drive_period,
        config.probe_initial_phase_rad,
    )?;
    let minus_readout = readout_distribution(&minus_model);
    validate_readout(&minus_readout)?;

    let fisher = total_z_classical_fisher_information(
        &probe_readout.probability_by_down_spins,
        &plus_readout.probability_by_down_spins,
        &minus_readout.probability_by_down_spins,
        delta,
    )?;
    let fisher_per_period = fisher / config.periods as f64;
    let norm_error = (nominal_model.norm_squared() - 1.0).abs();

    if !fisher_per_period.is_finite() || !norm_error.is_finite() {
        return Err(DtcProbeError::NonFiniteFisherInformation);
    }

    Ok(DtcProbeSensingReport {
        n_spins: config.n_spins,
        periods: config.periods,
        probe_frequency_cycles_per_drive_period: config.probe_frequency_cycles_per_drive_period,
        probe_amplitude: config.probe_amplitude,
        probe_initial_phase_rad: config.probe_initial_phase_rad,
        baseline_subharmonic_response: baseline_response,
        probed_subharmonic_response: probed_response,
        baseline_lifetime,
        probed_lifetime,
        probe_readout,
        classical_fisher_information_per_shot: fisher,
        fisher_information_per_drive_period: fisher_per_period,
        norm_error,
    })
}

/// Convert exact computational-basis amplitudes into a collective
/// total-magnetization outcome distribution.
pub fn readout_distribution(model: &KickedIsingFloquet) -> MagnetizationReadout {
    let n_spins = model.n_spins();
    let mut probabilities = vec![0.0; n_spins + 1];
    for (basis, amplitude) in model.amplitudes().iter().enumerate() {
        probabilities[basis.count_ones() as usize] += amplitude.norm_sqr();
    }
    let mean_magnetization = probabilities
        .iter()
        .enumerate()
        .map(|(down_spins, probability)| {
            probability * (n_spins as f64 - 2.0 * down_spins as f64) / n_spins as f64
        })
        .sum::<f64>();
    let normalization_error = (probabilities.iter().sum::<f64>() - 1.0).abs();

    MagnetizationReadout {
        probability_by_down_spins: probabilities,
        mean_magnetization,
        normalization_error,
    }
}

fn validate_readout(readout: &MagnetizationReadout) -> Result<(), DtcProbeError> {
    if readout.normalization_error > 1.0e-10
        || readout
            .probability_by_down_spins
            .iter()
            .any(|probability| !probability.is_finite() || *probability < 0.0)
    {
        return Err(DtcProbeError::InvalidProbabilityDistribution);
    }
    Ok(())
}

fn total_z_classical_fisher_information(
    nominal: &[f64],
    plus: &[f64],
    minus: &[f64],
    derivative_step: f64,
) -> Result<f64, DtcProbeError> {
    if nominal.is_empty()
        || nominal.len() != plus.len()
        || nominal.len() != minus.len()
        || !derivative_step.is_finite()
        || derivative_step <= 0.0
    {
        return Err(DtcProbeError::InvalidConfiguration);
    }

    let mut information = 0.0;
    for outcome in 0..nominal.len() {
        let probability = nominal[outcome];
        let derivative = (plus[outcome] - minus[outcome]) / (2.0 * derivative_step);
        if probability <= 1.0e-15 {
            if derivative.abs() > 1.0e-10 {
                return Err(DtcProbeError::SingularReadoutSupport {
                    outcome_down_spins: outcome,
                });
            }
            continue;
        }
        information += derivative * derivative / probability;
    }

    if !information.is_finite() || information < 0.0 {
        return Err(DtcProbeError::NonFiniteFisherInformation);
    }
    Ok(information)
}

/// Ideal single-parameter Cramér–Rao lower bound for independent repeated
/// total-Z measurements under the same model. It excludes nuisance parameters.
pub fn cramer_rao_standard_error(
    fisher_information_per_shot: f64,
    repetitions: usize,
) -> Option<f64> {
    if !fisher_information_per_shot.is_finite()
        || fisher_information_per_shot <= 0.0
        || repetitions == 0
    {
        return None;
    }
    Some((fisher_information_per_shot * repetitions as f64).sqrt().recip())
}

fn build_model(config: &DtcProbeConfig) -> Result<KickedIsingFloquet, DtcProbeError> {
    Ok(KickedIsingFloquet::new(
        config.n_spins,
        vec![config.nearest_neighbor_coupling; config.n_spins - 1],
        config.longitudinal_fields.clone(),
        config.pulse_angle_rad,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn config() -> DtcProbeConfig {
        DtcProbeConfig {
            n_spins: 6,
            nearest_neighbor_coupling: 0.5,
            longitudinal_fields: vec![0.0; 6],
            pulse_angle_rad: PI + 0.08,
            periods: 96,
            analysis_start: 16,
            probe_frequency_cycles_per_drive_period: 0.25,
            probe_initial_phase_rad: 0.0,
            probe_amplitude: 0.04,
            amplitude_derivative_step: 1.0e-3,
            lifetime_window: 8,
            lifetime_threshold_fraction: 0.5,
        }
    }

    #[test]
    fn collective_readout_distribution_is_normalized_and_matches_magnetization() {
        let model = build_model(&config()).unwrap();
        let readout = readout_distribution(&model);
        assert!(readout.normalization_error < 1.0e-12);
        assert_eq!(readout.probability_by_down_spins.len(), 7);
        assert!(readout
            .probability_by_down_spins
            .iter()
            .all(|probability| *probability >= 0.0));
        assert!((readout.mean_magnetization - model.magnetization()).abs() < 1.0e-12);
    }

    #[test]
    fn probe_frequency_sensing_reports_finite_nonnegative_fisher_information() {
        let report = evaluate_dtc_probe_sensing(&config()).unwrap();
        assert!(report.classical_fisher_information_per_shot.is_finite());
        assert!(report.classical_fisher_information_per_shot >= 0.0);
        assert!(report.fisher_information_per_drive_period.is_finite());
        assert!(report.norm_error < 1.0e-10);
    }

    #[test]
    fn zero_probe_reduces_to_baseline_subharmonic_trace() {
        let mut zero_probe = config();
        zero_probe.probe_amplitude = 0.0;
        let report = evaluate_dtc_probe_sensing(&zero_probe).unwrap();
        assert!(
            (report.baseline_subharmonic_response.mean_demodulated_order
                - report.probed_subharmonic_response.mean_demodulated_order)
                .abs()
                < 1.0e-12
        );
        assert!(
            (report.baseline_subharmonic_response.sign_coherence
                - report.probed_subharmonic_response.sign_coherence)
                .abs()
                < 1.0e-12
        );
    }

    #[test]
    fn lifetime_is_right_censored_when_threshold_is_never_crossed() {
        let signal: Vec<f64> = (0..100)
            .map(|period| if period % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        assert!(matches!(
            estimate_subharmonic_lifetime(&signal, 10, 8, 0.5),
            Some(SubharmonicLifetime::RightCensored { .. })
        ));
    }

    #[test]
    fn lifetime_detects_a_declining_demodulated_envelope() {
        let signal: Vec<f64> = (0..100)
            .map(|period| {
                let envelope = (-(period as f64) / 25.0).exp();
                if period % 2 == 0 {
                    envelope
                } else {
                    -envelope
                }
            })
            .collect();
        assert!(matches!(
            estimate_subharmonic_lifetime(&signal, 0, 6, 0.5),
            Some(SubharmonicLifetime::Observed { .. })
        ));
    }

    #[test]
    fn out_of_nyquist_probe_frequency_fails_closed() {
        let mut invalid = config();
        invalid.probe_frequency_cycles_per_drive_period = 0.5001;
        assert_eq!(
            evaluate_dtc_probe_sensing(&invalid),
            Err(DtcProbeError::InvalidConfiguration)
        );
    }

    #[test]
    fn fisher_information_matches_binary_probability_reference() {
        let nominal = [0.4, 0.6];
        let plus = [0.42, 0.58];
        let minus = [0.38, 0.62];
        let derivative = 0.04 / 0.002;
        let expected = derivative * derivative / 0.4 + derivative * derivative / 0.6;
        let observed =
            total_z_classical_fisher_information(&nominal, &plus, &minus, 0.001).unwrap();
        assert!((observed - expected).abs() < 1.0e-9);
    }

    #[test]
    fn one_parameter_cramer_rao_bound_decreases_with_repetitions() {
        let single = cramer_rao_standard_error(2.0, 1).unwrap();
        let hundred = cramer_rao_standard_error(2.0, 100).unwrap();
        assert!((hundred - single / 10.0).abs() < 1.0e-12);
    }
}
