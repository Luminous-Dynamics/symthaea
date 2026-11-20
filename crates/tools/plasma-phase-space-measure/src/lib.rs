//! Particle-label-independent diagnostics for weighted kinetic phase-space measures.
//!
//! This crate compares finite characteristic-function signatures of 1D periodic-x,
//! unbounded-v empirical particle measures. It is a benchmark diagnostic, not a
//! production plasma solver or a complete metric on probability measures.
//!
//! For each declared mode (k, eta), the complex coefficient is
//!
//!   mu_hat(k, eta) = sum_i w_i exp(i * (2*pi*k*x_i/L + eta*(v_i-v_ref)/v_scale)).
//!
//! The caller must freeze the mode set and coordinate scales as part of the
//! benchmark identity. Finite mode sets can alias distinct measures.

#![forbid(unsafe_code)]

use std::error::Error;
use std::f64::consts::TAU;
use std::fmt;

/// One weighted macro-particle in one-dimensional periodic position and
/// unbounded velocity phase space.
///
/// weight is the effective signed measure weight. Signed weights are
/// supported for delta-f-style representations; total-variation weight is
/// used for normalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseSpaceParticle1d {
    pub position: f64,
    pub velocity: f64,
    pub weight: f64,
}

/// One characteristic-function feature.
///
/// The phase is 2*pi*position_mode*x/L + velocity_frequency*(v-v_ref)/v_scale.
/// importance must be finite and strictly positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CharacteristicMode {
    pub position_mode: u32,
    pub velocity_frequency: f64,
    pub importance: f64,
}

/// Frozen coordinates and finite characteristic feature set.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseSpaceConfig {
    pub domain_length: f64,
    pub velocity_reference: f64,
    pub velocity_scale: f64,
    pub modes: Vec<CharacteristicMode>,
}

/// A complex Fourier/characteristic coefficient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexCoefficient {
    pub real: f64,
    pub imaginary: f64,
}

impl ComplexCoefficient {
    /// Euclidean magnitude of the complex coefficient.
    pub fn magnitude(self) -> f64 {
        self.real.hypot(self.imaginary)
    }
}

/// Characteristic signature for a weighted particle measure.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseSpaceSignature {
    pub coefficients: Vec<ComplexCoefficient>,
    pub total_variation_weight: f64,
    pub net_weight: f64,
    /// Metadata only; this value is never included in the measure distance.
    pub particle_count: usize,
}

/// Per-feature comparison, with errors normalized by the larger total-variation
/// weight of the candidate and reference measures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeComparison {
    pub absolute_error: f64,
    pub relative_error: f64,
}

/// Summary comparison of a candidate measure against a reference measure.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseSpaceComparison {
    /// Feature-importance-weighted RMS complex-coefficient error, normalized
    /// by max(total_variation(candidate), total_variation(reference)).
    pub weighted_rms_relative_error: f64,
    /// Largest normalized complex-coefficient error across configured modes.
    pub max_mode_relative_error: f64,
    /// Absolute net-weight mismatch divided by the same normalization scale.
    pub net_weight_relative_error: f64,
    /// Absolute total-variation-weight mismatch divided by the same scale.
    pub total_variation_relative_error: f64,
    pub candidate_total_variation_weight: f64,
    pub reference_total_variation_weight: f64,
    /// Particle counts are reported for analysis, never used in the metric.
    pub candidate_particle_count: usize,
    pub reference_particle_count: usize,
    pub per_mode: Vec<ModeComparison>,
}

/// Invalid input or non-finite computation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeasureError {
    InvalidDomainLength,
    InvalidVelocityScale,
    NonFiniteVelocityReference,
    EmptyModeSet,
    MissingDcMode,
    InvalidMode { mode: usize },
    NonFinitePosition { particle: usize },
    NonFiniteVelocity { particle: usize },
    NonFiniteWeight { particle: usize },
    NonFiniteOutput,
}

impl fmt::Display for MeasureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDomainLength => write!(f, "domain length must be finite and positive"),
            Self::InvalidVelocityScale => write!(f, "velocity scale must be finite and positive"),
            Self::NonFiniteVelocityReference => write!(f, "velocity reference must be finite"),
            Self::EmptyModeSet => write!(f, "at least one characteristic mode is required"),
            Self::MissingDcMode => write!(f, "mode set must include the DC mode (k=0, eta=0)"),
            Self::InvalidMode { mode } => {
                write!(f, "characteristic mode {mode} has invalid frequency or importance")
            }
            Self::NonFinitePosition { particle } => {
                write!(f, "particle {particle} has a non-finite position")
            }
            Self::NonFiniteVelocity { particle } => {
                write!(f, "particle {particle} has a non-finite velocity")
            }
            Self::NonFiniteWeight { particle } => {
                write!(f, "particle {particle} has a non-finite weight")
            }
            Self::NonFiniteOutput => write!(f, "phase-space diagnostic produced a non-finite value"),
        }
    }
}

impl Error for MeasureError {}

/// Neumaier compensated summation reduces (but cannot eliminate) dependence on
/// particle ordering and split representation due to floating-point arithmetic.
#[derive(Debug, Default, Clone, Copy)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
}

fn validate_config(config: &PhaseSpaceConfig) -> Result<(), MeasureError> {
    if !config.domain_length.is_finite() || config.domain_length <= 0.0 {
        return Err(MeasureError::InvalidDomainLength);
    }
    if !config.velocity_scale.is_finite() || config.velocity_scale <= 0.0 {
        return Err(MeasureError::InvalidVelocityScale);
    }
    if !config.velocity_reference.is_finite() {
        return Err(MeasureError::NonFiniteVelocityReference);
    }
    if config.modes.is_empty() {
        return Err(MeasureError::EmptyModeSet);
    }

    let mut has_dc = false;
    for (index, mode) in config.modes.iter().enumerate() {
        if !mode.velocity_frequency.is_finite()
            || !mode.importance.is_finite()
            || mode.importance <= 0.0
        {
            return Err(MeasureError::InvalidMode { mode: index });
        }
        if mode.position_mode == 0 && mode.velocity_frequency == 0.0 {
            has_dc = true;
        }
    }
    if !has_dc {
        return Err(MeasureError::MissingDcMode);
    }
    Ok(())
}

/// Evaluate a finite weighted characteristic signature.
///
/// Positions are interpreted modulo domain_length. The returned particle
/// count is informational only. If all particle weights are zero, the
/// total-variation and net weights are both zero and every coefficient is zero.
pub fn evaluate_phase_space_signature(
    particles: &[PhaseSpaceParticle1d],
    config: &PhaseSpaceConfig,
) -> Result<PhaseSpaceSignature, MeasureError> {
    validate_config(config)?;

    let mut total_variation = CompensatedSum::default();
    let mut net_weight = CompensatedSum::default();
    for (index, particle) in particles.iter().enumerate() {
        if !particle.position.is_finite() {
            return Err(MeasureError::NonFinitePosition { particle: index });
        }
        if !particle.velocity.is_finite() {
            return Err(MeasureError::NonFiniteVelocity { particle: index });
        }
        if !particle.weight.is_finite() {
            return Err(MeasureError::NonFiniteWeight { particle: index });
        }
        total_variation.add(particle.weight.abs());
        net_weight.add(particle.weight);
    }

    let mut coefficients = Vec::with_capacity(config.modes.len());
    for mode in &config.modes {
        let mut real = CompensatedSum::default();
        let mut imaginary = CompensatedSum::default();

        for particle in particles {
            let wrapped_position = particle.position.rem_euclid(config.domain_length);
            let normalized_position = wrapped_position / config.domain_length;
            let normalized_velocity =
                (particle.velocity - config.velocity_reference) / config.velocity_scale;
            let phase = TAU * mode.position_mode as f64 * normalized_position
                + mode.velocity_frequency * normalized_velocity;
            if !phase.is_finite() {
                return Err(MeasureError::NonFiniteOutput);
            }
            let (sin_phase, cos_phase) = phase.sin_cos();
            real.add(particle.weight * cos_phase);
            imaginary.add(particle.weight * sin_phase);
        }

        let coefficient = ComplexCoefficient {
            real: real.total(),
            imaginary: imaginary.total(),
        };
        if !coefficient.real.is_finite() || !coefficient.imaginary.is_finite() {
            return Err(MeasureError::NonFiniteOutput);
        }
        coefficients.push(coefficient);
    }

    let total_variation_weight = total_variation.total();
    let net_weight = net_weight.total();
    if !total_variation_weight.is_finite() || !net_weight.is_finite() {
        return Err(MeasureError::NonFiniteOutput);
    }

    Ok(PhaseSpaceSignature {
        coefficients,
        total_variation_weight,
        net_weight,
        particle_count: particles.len(),
    })
}

/// Compare candidate and reference weighted empirical measures using the
/// configured finite characteristic feature set.
///
/// The normalization scale is the larger total-variation weight. If both
/// measures have zero total variation, the scale is defined as one, yielding
/// zero error for two empty/zero-weight measures. This diagnostic is not a
/// complete metric: distinct measures can share the same finite signature.
pub fn compare_phase_space_measures(
    candidate: &[PhaseSpaceParticle1d],
    reference: &[PhaseSpaceParticle1d],
    config: &PhaseSpaceConfig,
) -> Result<PhaseSpaceComparison, MeasureError> {
    validate_config(config)?;
    let candidate_signature = evaluate_phase_space_signature(candidate, config)?;
    let reference_signature = evaluate_phase_space_signature(reference, config)?;

    let normalization = candidate_signature
        .total_variation_weight
        .max(reference_signature.total_variation_weight);
    let normalization = if normalization == 0.0 {
        1.0
    } else {
        normalization
    };

    let mut weighted_squared_error = CompensatedSum::default();
    let mut total_importance = CompensatedSum::default();
    let mut max_mode_relative_error: f64 = 0.0;
    let mut per_mode = Vec::with_capacity(config.modes.len());

    for (index, (candidate_coefficient, reference_coefficient)) in candidate_signature
        .coefficients
        .iter()
        .zip(reference_signature.coefficients.iter())
        .enumerate()
    {
        let real_error = candidate_coefficient.real - reference_coefficient.real;
        let imaginary_error =
            candidate_coefficient.imaginary - reference_coefficient.imaginary;
        let absolute_error = real_error.hypot(imaginary_error);
        let relative_error = absolute_error / normalization;
        if !absolute_error.is_finite() || !relative_error.is_finite() {
            return Err(MeasureError::NonFiniteOutput);
        }

        let importance = config.modes[index].importance;
        weighted_squared_error.add(importance * relative_error * relative_error);
        total_importance.add(importance);
        max_mode_relative_error = max_mode_relative_error.max(relative_error);
        per_mode.push(ModeComparison {
            absolute_error,
            relative_error,
        });
    }

    let importance_total = total_importance.total();
    let weighted_square_total = weighted_squared_error.total();
    if !importance_total.is_finite()
        || importance_total <= 0.0
        || !weighted_square_total.is_finite()
    {
        return Err(MeasureError::NonFiniteOutput);
    }
    let weighted_mean_square = weighted_square_total / importance_total;
    let weighted_rms_relative_error = weighted_mean_square.sqrt();
    let net_weight_relative_error =
        (candidate_signature.net_weight - reference_signature.net_weight).abs() / normalization;
    let total_variation_relative_error = (candidate_signature.total_variation_weight
        - reference_signature.total_variation_weight)
        .abs()
        / normalization;

    if !weighted_rms_relative_error.is_finite()
        || !net_weight_relative_error.is_finite()
        || !total_variation_relative_error.is_finite()
    {
        return Err(MeasureError::NonFiniteOutput);
    }

    Ok(PhaseSpaceComparison {
        weighted_rms_relative_error,
        max_mode_relative_error,
        net_weight_relative_error,
        total_variation_relative_error,
        candidate_total_variation_weight: candidate_signature.total_variation_weight,
        reference_total_variation_weight: reference_signature.total_variation_weight,
        candidate_particle_count: candidate_signature.particle_count,
        reference_particle_count: reference_signature.particle_count,
        per_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> PhaseSpaceConfig {
        PhaseSpaceConfig {
            domain_length: 8.0,
            velocity_reference: 0.25,
            velocity_scale: 2.0,
            modes: vec![
                CharacteristicMode { position_mode: 0, velocity_frequency: 0.0, importance: 2.0 },
                CharacteristicMode { position_mode: 1, velocity_frequency: 0.0, importance: 1.0 },
                CharacteristicMode { position_mode: 2, velocity_frequency: 0.0, importance: 1.0 },
                CharacteristicMode { position_mode: 0, velocity_frequency: 1.0, importance: 1.0 },
                CharacteristicMode { position_mode: 0, velocity_frequency: 2.0, importance: 1.0 },
                CharacteristicMode { position_mode: 1, velocity_frequency: 1.0, importance: 1.0 },
                CharacteristicMode { position_mode: 2, velocity_frequency: -1.0, importance: 1.0 },
            ],
        }
    }

    fn particle(position: f64, velocity: f64, weight: f64) -> PhaseSpaceParticle1d {
        PhaseSpaceParticle1d { position, velocity, weight }
    }

    fn assert_near(left: f64, right: f64, tolerance: f64) {
        assert!(
            (left - right).abs() <= tolerance,
            "left={left:.17e}, right={right:.17e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn particle_permutation_preserves_signature() {
        let a = vec![
            particle(0.5, -1.0, 0.7),
            particle(2.0, 0.25, -0.2),
            particle(7.5, 1.5, 0.4),
            particle(4.0, 2.25, 0.1),
        ];
        let b = vec![a[2], a[0], a[3], a[1]];
        let comparison = compare_phase_space_measures(&a, &b, &config()).unwrap();
        assert!(comparison.weighted_rms_relative_error <= 1.0e-14);
        assert_eq!(comparison.candidate_particle_count, 4);
        assert_eq!(comparison.reference_particle_count, 4);
    }

    #[test]
    fn co_located_weight_split_preserves_measure_not_particle_count() {
        let reference = vec![
            particle(1.25, -0.75, 0.8),
            particle(5.5, 1.25, -0.3),
        ];
        let candidate = vec![
            particle(1.25, -0.75, 0.3),
            particle(1.25, -0.75, 0.5),
            particle(5.5, 1.25, -0.1),
            particle(5.5, 1.25, -0.2),
        ];
        let comparison = compare_phase_space_measures(&candidate, &reference, &config()).unwrap();
        assert!(comparison.weighted_rms_relative_error <= 1.0e-14);
        assert_eq!(comparison.candidate_particle_count, 4);
        assert_eq!(comparison.reference_particle_count, 2);
    }

    #[test]
    fn integer_period_shifts_preserve_position_representative() {
        let reference = vec![
            particle(0.125, -0.5, 0.6),
            particle(6.75, 1.0, 0.4),
        ];
        let candidate = vec![
            particle(16.125, -0.5, 0.6),
            particle(-9.25, 1.0, 0.4),
        ];
        let comparison = compare_phase_space_measures(&candidate, &reference, &config()).unwrap();
        assert!(comparison.weighted_rms_relative_error <= 1.0e-13);
    }

    #[test]
    fn velocity_change_is_detected_even_when_positions_match() {
        let reference = vec![
            particle(1.0, -1.0, 0.5),
            particle(3.0, 1.0, 0.5),
        ];
        let candidate = vec![
            particle(1.0, 1.0, 0.5),
            particle(3.0, -1.0, 0.5),
        ];
        let comparison = compare_phase_space_measures(&candidate, &reference, &config()).unwrap();
        assert!(comparison.weighted_rms_relative_error > 1.0e-3);
    }

    #[test]
    fn net_weight_negative_control_is_visible_in_dc_mode() {
        let reference = vec![particle(1.0, 0.0, 1.0)];
        let candidate = vec![particle(1.0, 0.0, 0.75)];
        let comparison = compare_phase_space_measures(&candidate, &reference, &config()).unwrap();
        assert!(comparison.net_weight_relative_error > 0.0);
        assert!(comparison.per_mode[0].relative_error > 0.0);
    }

    #[test]
    fn signed_weights_use_total_variation_normalization() {
        let particles = vec![
            particle(1.0, -1.0, 1.0),
            particle(2.0, 0.5, -0.4),
        ];
        let signature = evaluate_phase_space_signature(&particles, &config()).unwrap();
        assert_near(signature.total_variation_weight, 1.4, 1.0e-14);
        assert_near(signature.net_weight, 0.6, 1.0e-14);
    }

    #[test]
    fn zero_weight_measures_have_zero_distance() {
        let a = vec![particle(1.0, 2.0, 0.0)];
        let b = vec![particle(7.0, -3.0, 0.0), particle(2.0, 0.0, 0.0)];
        let comparison = compare_phase_space_measures(&a, &b, &config()).unwrap();
        assert_eq!(comparison.weighted_rms_relative_error, 0.0);
        assert_eq!(comparison.max_mode_relative_error, 0.0);
        assert_eq!(comparison.candidate_particle_count, 1);
        assert_eq!(comparison.reference_particle_count, 2);
    }

    #[test]
    fn empty_measure_matches_empty_measure() {
        let comparison = compare_phase_space_measures(&[], &[], &config()).unwrap();
        assert_eq!(comparison.weighted_rms_relative_error, 0.0);
        assert_eq!(comparison.net_weight_relative_error, 0.0);
    }

    #[test]
    fn finite_mode_signature_detects_a_moved_weight() {
        let reference = vec![
            particle(0.5, 0.0, 0.5),
            particle(4.5, 0.0, 0.5),
        ];
        let candidate = vec![
            particle(1.5, 0.0, 0.5),
            particle(4.5, 0.0, 0.5),
        ];
        let comparison = compare_phase_space_measures(&candidate, &reference, &config()).unwrap();
        assert!(comparison.max_mode_relative_error > 1.0e-3);
    }

    #[test]
    fn missing_dc_mode_is_rejected() {
        let mut invalid = config();
        invalid.modes.remove(0);
        assert_eq!(
            evaluate_phase_space_signature(&[], &invalid).unwrap_err(),
            MeasureError::MissingDcMode
        );
    }

    #[test]
    fn invalid_scales_and_mode_importance_are_rejected() {
        let mut invalid = config();
        invalid.domain_length = 0.0;
        assert_eq!(
            evaluate_phase_space_signature(&[], &invalid).unwrap_err(),
            MeasureError::InvalidDomainLength
        );

        let mut invalid = config();
        invalid.velocity_scale = f64::NAN;
        assert_eq!(
            evaluate_phase_space_signature(&[], &invalid).unwrap_err(),
            MeasureError::InvalidVelocityScale
        );

        let mut invalid = config();
        invalid.modes[1].importance = 0.0;
        assert_eq!(
            evaluate_phase_space_signature(&[], &invalid).unwrap_err(),
            MeasureError::InvalidMode { mode: 1 }
        );
    }

    #[test]
    fn non_finite_particle_values_are_rejected() {
        let invalid_position = vec![particle(f64::NAN, 0.0, 1.0)];
        assert_eq!(
            evaluate_phase_space_signature(&invalid_position, &config()).unwrap_err(),
            MeasureError::NonFinitePosition { particle: 0 }
        );

        let invalid_velocity = vec![particle(0.0, f64::INFINITY, 1.0)];
        assert_eq!(
            evaluate_phase_space_signature(&invalid_velocity, &config()).unwrap_err(),
            MeasureError::NonFiniteVelocity { particle: 0 }
        );

        let invalid_weight = vec![particle(0.0, 0.0, f64::NEG_INFINITY)];
        assert_eq!(
            evaluate_phase_space_signature(&invalid_weight, &config()).unwrap_err(),
            MeasureError::NonFiniteWeight { particle: 0 }
        );
    }

    #[test]
    fn finite_mode_signatures_can_alias_distinct_measures() {
        let mut finite_modes = config();
        finite_modes.modes.retain(|mode| mode.position_mode <= 2);
        let reference = vec![
            particle(0.0, 0.0, 1.0),
            particle(2.0, 0.0, 1.0),
            particle(4.0, 0.0, 1.0),
            particle(6.0, 0.0, 1.0),
        ];
        let candidate = vec![
            particle(1.0, 0.0, 1.0),
            particle(3.0, 0.0, 1.0),
            particle(5.0, 0.0, 1.0),
            particle(7.0, 0.0, 1.0),
        ];
        let comparison =
            compare_phase_space_measures(&candidate, &reference, &finite_modes).unwrap();

        // The supports differ, but the configured k=0,1,2 coefficients agree.
        // This test documents why finite-feature error is not a proof of equality.
        assert!(comparison.weighted_rms_relative_error <= 1.0e-14);
    }

    #[test]
    fn mode_set_must_not_be_empty() {
        let mut invalid = config();
        invalid.modes.clear();
        assert_eq!(
            evaluate_phase_space_signature(&[], &invalid).unwrap_err(),
            MeasureError::EmptyModeSet
        );
    }
}
