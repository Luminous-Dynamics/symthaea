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
use rustfft::num_complex::Complex;
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
    /// Nominal phase at the first probe midpoint. The report also treats this
    /// as an unknown local nuisance when computing the joint Fisher matrix.
    pub probe_initial_phase_rad: f64,
    /// Dimensionless longitudinal probe-kick amplitude A multiplying sum_i Z_i.
    pub probe_amplitude: f64,
    /// Finite-difference displacement used to estimate d p(outcome) / d A.
    pub amplitude_derivative_step: f64,
    /// Centered finite-difference step for the initial probe phase nuisance,
    /// in radians.
    pub phase_derivative_step_rad: f64,
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
            || self
                .longitudinal_fields
                .iter()
                .any(|value| !value.is_finite())
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
            || !self.phase_derivative_step_rad.is_finite()
            || self.phase_derivative_step_rad <= 0.0
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
    SingularReadoutSupport { outcome_down_spins: usize },
    NonFiniteFisherInformation,
    InvalidStateVector,
    Floquet(FloquetError),
}

impl fmt::Display for DtcProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfiguration => {
                write!(f, "finite-Floquet probe-sensing configuration is invalid")
            }
            Self::InvalidProbabilityDistribution => {
                write!(
                    f,
                    "computational-basis readout probabilities are not normalized"
                )
            }
            Self::SingularReadoutSupport { outcome_down_spins } => write!(
                f,
                "readout outcome {outcome_down_spins} has zero nominal probability but nonzero local derivative"
            ),
            Self::NonFiniteFisherInformation => {
                write!(f, "Fisher information is not finite and non-negative")
            }
            Self::InvalidStateVector => {
                write!(f, "state vectors are invalid for finite-difference QFI")
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
    /// Known-phase conditional classical FI for the total-Z readout, per shot.
    /// Units are inverse squared dimensionless probe-kick amplitude.
    pub classical_fisher_information_per_shot: f64,
    /// Known-phase conditional pure-state QFI from the centered amplitude
    /// derivative. This is an upper bound for measurements on the declared
    /// ideal pure state, not achieved readout.
    pub quantum_fisher_information_per_shot: f64,
    /// Joint classical Fisher matrix for [probe amplitude, initial phase].
    pub classical_fisher_matrix_per_shot: [[f64; 2]; 2],
    /// Joint pure-state QFIM for [probe amplitude, initial phase].
    pub quantum_fisher_matrix_per_shot: [[f64; 2]; 2],
    /// Amplitude information after treating phase as an unknown nuisance.
    /// None means the phase is itself informative but the joint information
    /// matrix is singular; a zero-frequency phase-invariant control can still
    /// return the known-phase amplitude information.
    pub classical_effective_amplitude_fisher_with_phase_nuisance: Option<f64>,
    pub quantum_effective_amplitude_fisher_with_phase_nuisance: Option<f64>,
    /// Conversion from phase uncertainty to equivalent clock offset, in
    /// drive periods per radian: |dT/dphi| = 1 / (2 pi f). Undefined at f=0.
    pub phase_time_offset_drive_periods_per_radian: Option<f64>,
    /// Local joint Cramér-Rao covariance lower bounds per independent shot,
    /// ordered [amplitude, phase]. Singular matrices are reported as None.
    pub classical_joint_crlb_covariance_per_shot: Option<[[f64; 2]; 2]>,
    pub quantum_joint_crlb_covariance_per_shot: Option<[[f64; 2]; 2]>,
    /// Classical total-Z Fisher information divided by the pure-state QFI.
    /// None means that the QFI is numerically zero.
    pub classical_to_quantum_fisher_ratio: Option<f64>,
    /// Allowance includes finite-difference truncation and roundoff.
    pub classical_fisher_within_quantum_bound: bool,
    /// Fisher information divided by simulated Floquet periods. This omits
    /// state preparation, measurement and dead-time costs.
    pub fisher_information_per_drive_period: f64,
    pub quantum_fisher_information_per_drive_period: f64,
    pub norm_error: f64,
}

/// Run baseline, nominal, +/-amplitude and +/-phase probe simulations, then
/// compute joint amplitude/phase Fisher information for a final total-Z readout.
///
/// Both derivatives use centered finite differences. All runs share the same
/// Hamiltonian, drive, frequency, duration, preparation and readout; only the
/// amplitude or nominal phase is perturbed. The known-phase scalar FI/QFI
/// remain available as the amplitude diagonal, while nuisance-adjusted
/// amplitude information is reported separately via a Schur complement.
pub fn evaluate_dtc_probe_sensing(
    config: &DtcProbeConfig,
) -> Result<DtcProbeSensingReport, DtcProbeError> {
    config.validate()?;

    let mut baseline_model = build_model(config)?;
    let baseline_signal = baseline_model.stroboscopic_magnetization(config.periods);
    let baseline_response = analyze_subharmonic_response(&baseline_signal, config.analysis_start)
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

    let phase_delta = config.phase_derivative_step_rad;
    let mut phase_plus_model = build_model(config)?;
    phase_plus_model.stroboscopic_magnetization_with_uniform_ac_probe(
        config.periods,
        config.probe_amplitude,
        config.probe_frequency_cycles_per_drive_period,
        config.probe_initial_phase_rad + phase_delta,
    )?;
    let phase_plus_readout = readout_distribution(&phase_plus_model);
    validate_readout(&phase_plus_readout)?;

    let mut phase_minus_model = build_model(config)?;
    phase_minus_model.stroboscopic_magnetization_with_uniform_ac_probe(
        config.periods,
        config.probe_amplitude,
        config.probe_frequency_cycles_per_drive_period,
        config.probe_initial_phase_rad - phase_delta,
    )?;
    let phase_minus_readout = readout_distribution(&phase_minus_model);
    validate_readout(&phase_minus_readout)?;

    let classical_fisher_matrix = total_z_classical_fisher_matrix(
        &probe_readout.probability_by_down_spins,
        &plus_readout.probability_by_down_spins,
        &minus_readout.probability_by_down_spins,
        &phase_plus_readout.probability_by_down_spins,
        &phase_minus_readout.probability_by_down_spins,
        delta,
        phase_delta,
    )?;
    let quantum_fisher_matrix = pure_state_quantum_fisher_matrix(
        nominal_model.amplitudes(),
        plus_model.amplitudes(),
        minus_model.amplitudes(),
        phase_plus_model.amplitudes(),
        phase_minus_model.amplitudes(),
        delta,
        phase_delta,
    )?;
    let fisher = classical_fisher_matrix[0][0];
    let quantum_fisher = quantum_fisher_matrix[0][0];
    let classical_effective_amplitude_fisher =
        effective_amplitude_information_with_phase_nuisance(classical_fisher_matrix);
    let quantum_effective_amplitude_fisher =
        effective_amplitude_information_with_phase_nuisance(quantum_fisher_matrix);
    let phase_time_offset_conversion = if config.probe_frequency_cycles_per_drive_period > 0.0 {
        Some(
            1.0 / (2.0 * std::f64::consts::PI
                * config.probe_frequency_cycles_per_drive_period),
        )
    } else {
        None
    };
    let classical_joint_covariance = invert_fisher_matrix_2x2(classical_fisher_matrix);
    let quantum_joint_covariance = invert_fisher_matrix_2x2(quantum_fisher_matrix);
    let fisher_per_period = fisher / config.periods as f64;
    let quantum_fisher_per_period = quantum_fisher / config.periods as f64;
    let classical_to_quantum_fisher_ratio = if quantum_fisher > f64::EPSILON {
        Some(fisher / quantum_fisher)
    } else {
        None
    };
    let classical_fisher_within_quantum_bound =
        fisher <= quantum_fisher + quantum_fisher.max(1.0) * 1.0e-3;
    let norm_error = (nominal_model.norm_squared() - 1.0).abs();

    if !fisher_per_period.is_finite()
        || !quantum_fisher.is_finite()
        || !quantum_fisher_per_period.is_finite()
        || !norm_error.is_finite()
    {
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
        quantum_fisher_information_per_shot: quantum_fisher,
        classical_fisher_matrix_per_shot: classical_fisher_matrix,
        quantum_fisher_matrix_per_shot: quantum_fisher_matrix,
        classical_effective_amplitude_fisher_with_phase_nuisance:
            classical_effective_amplitude_fisher,
        quantum_effective_amplitude_fisher_with_phase_nuisance:
            quantum_effective_amplitude_fisher,
        phase_time_offset_drive_periods_per_radian: phase_time_offset_conversion,
        classical_joint_crlb_covariance_per_shot: classical_joint_covariance,
        quantum_joint_crlb_covariance_per_shot: quantum_joint_covariance,
        classical_to_quantum_fisher_ratio,
        classical_fisher_within_quantum_bound,
        fisher_information_per_drive_period: fisher_per_period,
        quantum_fisher_information_per_drive_period: quantum_fisher_per_period,
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

/// Pure-state quantum Fisher information for the probe-amplitude parameter.
///
/// For a normalized pure state, F_Q = 4 ( <dpsi|dpsi> - |<psi|dpsi>|^2 ).
/// The state derivative is evaluated from the same plus/minus amplitude
/// trajectories used by the classical-readout Fisher estimator.
fn pure_state_quantum_fisher_information(
    nominal: &[Complex<f64>],
    plus: &[Complex<f64>],
    minus: &[Complex<f64>],
    derivative_step: f64,
) -> Result<f64, DtcProbeError> {
    if nominal.is_empty()
        || nominal.len() != plus.len()
        || nominal.len() != minus.len()
        || !derivative_step.is_finite()
        || derivative_step <= 0.0
    {
        return Err(DtcProbeError::InvalidStateVector);
    }

    let mut nominal_norm = 0.0;
    let mut derivative_norm = 0.0;
    let mut overlap = Complex::new(0.0, 0.0);
    for index in 0..nominal.len() {
        let psi = nominal[index];
        let derivative = (plus[index] - minus[index]) / (2.0 * derivative_step);
        if !psi.re.is_finite()
            || !psi.im.is_finite()
            || !plus[index].re.is_finite()
            || !plus[index].im.is_finite()
            || !minus[index].re.is_finite()
            || !minus[index].im.is_finite()
        {
            return Err(DtcProbeError::InvalidStateVector);
        }
        nominal_norm += psi.norm_sqr();
        derivative_norm += derivative.norm_sqr();
        overlap += psi.conj() * derivative;
    }

    if (nominal_norm - 1.0).abs() > 1.0e-8 {
        return Err(DtcProbeError::InvalidStateVector);
    }
    let raw = 4.0 * (derivative_norm - overlap.norm_sqr());
    if !raw.is_finite() || raw < -1.0e-8 {
        return Err(DtcProbeError::NonFiniteFisherInformation);
    }
    Ok(raw.max(0.0))
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

/// Joint pure-state quantum Fisher information matrix for amplitude and phase.
fn pure_state_quantum_fisher_matrix(
    nominal: &[Complex<f64>],
    amplitude_plus: &[Complex<f64>],
    amplitude_minus: &[Complex<f64>],
    phase_plus: &[Complex<f64>],
    phase_minus: &[Complex<f64>],
    amplitude_step: f64,
    phase_step: f64,
) -> Result<[[f64; 2]; 2], DtcProbeError> {
    let states = [amplitude_plus, amplitude_minus, phase_plus, phase_minus];
    if nominal.is_empty()
        || states.iter().any(|state| state.len() != nominal.len())
        || !amplitude_step.is_finite()
        || amplitude_step <= 0.0
        || !phase_step.is_finite()
        || phase_step <= 0.0
    {
        return Err(DtcProbeError::InvalidStateVector);
    }

    let nominal_norm = nominal.iter().map(|amplitude| amplitude.norm_sqr()).sum::<f64>();
    if !nominal_norm.is_finite() || (nominal_norm - 1.0).abs() > 1.0e-8 {
        return Err(DtcProbeError::InvalidStateVector);
    }

    let mut derivatives = [
        vec![Complex::new(0.0, 0.0); nominal.len()],
        vec![Complex::new(0.0, 0.0); nominal.len()],
    ];
    for index in 0..nominal.len() {
        if !nominal[index].re.is_finite() || !nominal[index].im.is_finite() {
            return Err(DtcProbeError::InvalidStateVector);
        }
        for state in states {
            if !state[index].re.is_finite() || !state[index].im.is_finite() {
                return Err(DtcProbeError::InvalidStateVector);
            }
        }
        derivatives[0][index] =
            (amplitude_plus[index] - amplitude_minus[index]) / (2.0 * amplitude_step);
        derivatives[1][index] = (phase_plus[index] - phase_minus[index]) / (2.0 * phase_step);
    }

    let mut derivative_norm = [[0.0; 2]; 2];
    let mut overlaps = [Complex::new(0.0, 0.0); 2];
    for left in 0..2 {
        for index in 0..nominal.len() {
            overlaps[left] += nominal[index].conj() * derivatives[left][index];
        }
        for right in 0..2 {
            for index in 0..nominal.len() {
                derivative_norm[left][right] +=
                    (derivatives[left][index].conj() * derivatives[right][index]).re;
            }
        }
    }

    let mut information = [[0.0; 2]; 2];
    for left in 0..2 {
        for right in 0..2 {
            let projected = derivative_norm[left][right]
                - (overlaps[left].conj() * overlaps[right]).re;
            information[left][right] = 4.0 * projected;
            if !information[left][right].is_finite() {
                return Err(DtcProbeError::NonFiniteFisherInformation);
            }
        }
    }

    // Symmetrize only the roundoff-level asymmetry implied by the analytic
    // expression. A material asymmetry would indicate an implementation bug.
    let asymmetry = (information[0][1] - information[1][0]).abs();
    if asymmetry > 1.0e-8 * information[0][0].abs().max(information[1][1].abs()).max(1.0) {
        return Err(DtcProbeError::NonFiniteFisherInformation);
    }
    let cross = 0.5 * (information[0][1] + information[1][0]);
    information[0][1] = cross;
    information[1][0] = cross;
    for diagonal in [information[0][0], information[1][1]] {
        if diagonal < -1.0e-8 {
            return Err(DtcProbeError::NonFiniteFisherInformation);
        }
    }
    information[0][0] = information[0][0].max(0.0);
    information[1][1] = information[1][1].max(0.0);
    Ok(information)
}

/// Joint classical Fisher information matrix for the declared total-Z
/// probability distribution, with parameters [amplitude, phase].
fn total_z_classical_fisher_matrix(
    nominal: &[f64],
    amplitude_plus: &[f64],
    amplitude_minus: &[f64],
    phase_plus: &[f64],
    phase_minus: &[f64],
    amplitude_step: f64,
    phase_step: f64,
) -> Result<[[f64; 2]; 2], DtcProbeError> {
    let distributions = [amplitude_plus, amplitude_minus, phase_plus, phase_minus];
    if nominal.is_empty()
        || distributions
            .iter()
            .any(|distribution| distribution.len() != nominal.len())
        || !amplitude_step.is_finite()
        || amplitude_step <= 0.0
        || !phase_step.is_finite()
        || phase_step <= 0.0
    {
        return Err(DtcProbeError::InvalidConfiguration);
    }

    let mut information = [[0.0; 2]; 2];
    for outcome in 0..nominal.len() {
        let probability = nominal[outcome];
        if !probability.is_finite() || probability < 0.0 {
            return Err(DtcProbeError::InvalidProbabilityDistribution);
        }
        let derivatives = [
            (amplitude_plus[outcome] - amplitude_minus[outcome]) / (2.0 * amplitude_step),
            (phase_plus[outcome] - phase_minus[outcome]) / (2.0 * phase_step),
        ];
        if derivatives.iter().any(|value| !value.is_finite()) {
            return Err(DtcProbeError::NonFiniteFisherInformation);
        }
        if probability <= 1.0e-15 {
            if derivatives.iter().any(|value| value.abs() > 1.0e-10) {
                return Err(DtcProbeError::SingularReadoutSupport {
                    outcome_down_spins: outcome,
                });
            }
            continue;
        }
        for left in 0..2 {
            for right in 0..2 {
                information[left][right] +=
                    derivatives[left] * derivatives[right] / probability;
            }
        }
    }

    for row in &information {
        if row.iter().any(|value| !value.is_finite()) {
            return Err(DtcProbeError::NonFiniteFisherInformation);
        }
    }
    let cross = 0.5 * (information[0][1] + information[1][0]);
    information[0][1] = cross;
    information[1][0] = cross;
    Ok(information)
}

/// Schur-complement information for amplitude after eliminating the phase
/// nuisance. If phase has no information and no amplitude/phase coupling,
/// phase is irrelevant to this amplitude readout and the known-phase value is
/// retained. Otherwise the singular nuisance case returns None.
fn effective_amplitude_information_with_phase_nuisance(
    information: [[f64; 2]; 2],
) -> Option<f64> {
    let amplitude = information[0][0];
    let cross = information[0][1];
    let phase = information[1][1];
    if !amplitude.is_finite() || !cross.is_finite() || !phase.is_finite() {
        return None;
    }
    if phase <= f64::EPSILON {
        return if cross.abs() <= 1.0e-12 {
            Some(amplitude.max(0.0))
        } else {
            None
        };
    }
    let effective = amplitude - cross * cross / phase;
    if !effective.is_finite() || effective < -1.0e-8 * amplitude.abs().max(1.0) {
        return None;
    }
    Some(effective.max(0.0))
}

/// Invert a symmetric 2x2 Fisher matrix when both parameters are jointly
/// identifiable. The returned covariance is a per-shot local CRLB.
fn invert_fisher_matrix_2x2(information: [[f64; 2]; 2]) -> Option<[[f64; 2]; 2]> {
    let a = information[0][0];
    let b = 0.5 * (information[0][1] + information[1][0]);
    let d = information[1][1];
    if !a.is_finite() || !b.is_finite() || !d.is_finite() || a <= 0.0 || d <= 0.0 {
        return None;
    }
    let determinant = a * d - b * b;
    let scale = (a * d).abs().max(b * b).max(f64::MIN_POSITIVE);
    if !determinant.is_finite() || determinant <= 1.0e-12 * scale {
        return None;
    }
    let covariance = [
        [d / determinant, -b / determinant],
        [-b / determinant, a / determinant],
    ];
    if covariance.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }
    Some(covariance)
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
    Some(
        (fisher_information_per_shot * repetitions as f64)
            .sqrt()
            .recip(),
    )
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
            phase_derivative_step_rad: 1.0e-3,
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
        assert!(
            readout
                .probability_by_down_spins
                .iter()
                .all(|probability| *probability >= 0.0)
        );
        assert!((readout.mean_magnetization - model.magnetization()).abs() < 1.0e-12);
    }

    #[test]
    fn probe_frequency_sensing_reports_finite_nonnegative_fisher_information() {
        let report = evaluate_dtc_probe_sensing(&config()).unwrap();
        assert!(report.classical_fisher_information_per_shot.is_finite());
        assert!(report.classical_fisher_information_per_shot >= 0.0);
        assert!(report.fisher_information_per_drive_period.is_finite());
        assert!(report.quantum_fisher_information_per_shot.is_finite());
        assert!(
            report
                .quantum_fisher_information_per_drive_period
                .is_finite()
        );
        assert!(report.norm_error < 1.0e-10);
    }

    #[test]
    fn joint_fisher_matrices_are_symmetric_and_classical_is_bounded_by_quantum() {
        let report = evaluate_dtc_probe_sensing(&config()).unwrap();
        let classical = report.classical_fisher_matrix_per_shot;
        let quantum = report.quantum_fisher_matrix_per_shot;
        assert!((classical[0][1] - classical[1][0]).abs() < 1.0e-10);
        assert!((quantum[0][1] - quantum[1][0]).abs() < 1.0e-10);

        // For a 2x2 symmetric matrix, PSD is equivalent to non-negative
        // diagonal entries and determinant (within finite-difference tolerance).
        for matrix in [classical, quantum] {
            assert!(matrix[0][0] >= -1.0e-10);
            assert!(matrix[1][1] >= -1.0e-10);
            assert!(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0] >= -1.0e-8);
        }

        let difference = [
            [
                quantum[0][0] - classical[0][0],
                quantum[0][1] - classical[0][1],
            ],
            [
                quantum[1][0] - classical[1][0],
                quantum[1][1] - classical[1][1],
            ],
        ];
        assert!(difference[0][0] >= -1.0e-6);
        assert!(difference[1][1] >= -1.0e-6);
        assert!(
            difference[0][0] * difference[1][1]
                - difference[0][1] * difference[1][0]
                >= -1.0e-5
        );
        assert!(
            (report.classical_fisher_information_per_shot - classical[0][0]).abs() < 1.0e-12
        );
        assert!(
            (report.quantum_fisher_information_per_shot - quantum[0][0]).abs() < 1.0e-12
        );
    }

    #[test]
    fn phase_nuisance_cannot_increase_amplitude_information() {
        let report = evaluate_dtc_probe_sensing(&config()).unwrap();
        let known_phase = report.classical_fisher_information_per_shot;
        let unknown_phase = report
            .classical_effective_amplitude_fisher_with_phase_nuisance
            .unwrap();
        assert!(unknown_phase <= known_phase + 1.0e-10);
        assert!(report.phase_time_offset_drive_periods_per_radian.is_some());
    }

    #[test]
    fn zero_frequency_has_no_phase_to_clock_conversion() {
        let mut zero_frequency = config();
        zero_frequency.probe_frequency_cycles_per_drive_period = 0.0;
        let report = evaluate_dtc_probe_sensing(&zero_frequency).unwrap();
        assert_eq!(report.phase_time_offset_drive_periods_per_radian, None);
        assert!(report.classical_effective_amplitude_fisher_with_phase_nuisance.is_some());
    }

    #[test]
    fn invalid_phase_derivative_step_fails_closed() {
        let mut invalid = config();
        invalid.phase_derivative_step_rad = f64::NAN;
        assert_eq!(
            evaluate_dtc_probe_sensing(&invalid),
            Err(DtcProbeError::InvalidConfiguration)
        );
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
                if period % 2 == 0 { envelope } else { -envelope }
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
    fn pure_state_qfi_matches_analytic_single_qubit_rotation() {
        let parameter = 0.3_f64;
        let delta = 1.0e-4_f64;
        let state = [
            Complex::new(parameter.cos(), 0.0),
            Complex::new(parameter.sin(), 0.0),
        ];
        let plus = [
            Complex::new((parameter + delta).cos(), 0.0),
            Complex::new((parameter + delta).sin(), 0.0),
        ];
        let minus = [
            Complex::new((parameter - delta).cos(), 0.0),
            Complex::new((parameter - delta).sin(), 0.0),
        ];
        let qfi = pure_state_quantum_fisher_information(&state, &plus, &minus, delta).unwrap();
        assert!((qfi - 4.0).abs() < 1.0e-7);
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
