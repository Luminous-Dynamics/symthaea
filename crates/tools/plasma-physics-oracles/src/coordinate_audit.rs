//! Coordinate-map sensitivity oracles for periodic kinetic-plasma models.
//!
//! These helpers test mathematical properties of normalization schemes. They
//! do not execute or validate a neural operator.

#![forbid(unsafe_code)]

use std::error::Error;
use std::f64::consts::TAU;
use std::fmt;

/// Errors produced by coordinate-map audit helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoordinateAuditError {
    EmptyCoordinates,
    NonFiniteCoordinate { index: usize },
    InvalidDomainLength,
    NonFiniteOrigin,
    ZeroSampleSpan,
    LengthMismatch { before: usize, after: usize },
    ChangedIndexOutOfRange { index: usize, length: usize },
}

impl fmt::Display for CoordinateAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCoordinates => write!(f, "coordinate set must be non-empty"),
            Self::NonFiniteCoordinate { index } => write!(f, "coordinate {index} is non-finite"),
            Self::InvalidDomainLength => write!(f, "domain length must be finite and positive"),
            Self::NonFiniteOrigin => write!(f, "domain origin must be finite"),
            Self::ZeroSampleSpan => write!(f, "sample min-max normalization requires non-zero span"),
            Self::LengthMismatch { before, after } => {
                write!(f, "coordinate lengths differ: {before} vs {after}")
            }
            Self::ChangedIndexOutOfRange { index, length } => {
                write!(f, "changed index {index} is outside coordinate length {length}")
            }
        }
    }
}

impl Error for CoordinateAuditError {}

fn validate_coordinates(coordinates: &[f64]) -> Result<(), CoordinateAuditError> {
    if coordinates.is_empty() {
        return Err(CoordinateAuditError::EmptyCoordinates);
    }
    for (index, value) in coordinates.iter().enumerate() {
        if !value.is_finite() {
            return Err(CoordinateAuditError::NonFiniteCoordinate { index });
        }
    }
    Ok(())
}

/// Normalize a coordinate set independently using its own minimum and maximum.
///
/// Changing one extremal coordinate can therefore change every other
/// normalized coordinate.
pub fn sample_minmax_normalize(
    coordinates: &[f64],
) -> Result<Vec<f64>, CoordinateAuditError> {
    validate_coordinates(coordinates)?;
    let minimum = coordinates.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = coordinates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = maximum - minimum;
    if !span.is_finite() || span <= 0.0 {
        return Err(CoordinateAuditError::ZeroSampleSpan);
    }

    let normalized: Vec<f64> = coordinates
        .iter()
        .map(|value| (value - minimum) / span)
        .collect();
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err(CoordinateAuditError::ZeroSampleSpan);
    }
    Ok(normalized)
}

/// Map coordinates into a fixed periodic domain without consulting the other
/// particles in the sample. This map is appropriate only when periodic
/// boundary conditions and the declared domain are physically correct.
pub fn fixed_periodic_domain_normalize(
    coordinates: &[f64],
    origin: f64,
    domain_length: f64,
) -> Result<Vec<f64>, CoordinateAuditError> {
    validate_coordinates(coordinates)?;
    if !origin.is_finite() {
        return Err(CoordinateAuditError::NonFiniteOrigin);
    }
    if !domain_length.is_finite() || domain_length <= 0.0 {
        return Err(CoordinateAuditError::InvalidDomainLength);
    }

    let normalized: Vec<f64> = coordinates
        .iter()
        .map(|value| (value - origin).rem_euclid(domain_length) / domain_length)
        .collect();
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err(CoordinateAuditError::InvalidDomainLength);
    }
    Ok(normalized)
}

/// Smooth two-component periodic embedding (sin(2πx/L), cos(2πx/L)).
pub fn periodic_coordinate_embedding(
    coordinate: f64,
    domain_length: f64,
) -> Result<[f64; 2], CoordinateAuditError> {
    if !coordinate.is_finite() {
        return Err(CoordinateAuditError::NonFiniteCoordinate { index: 0 });
    }
    if !domain_length.is_finite() || domain_length <= 0.0 {
        return Err(CoordinateAuditError::InvalidDomainLength);
    }

    let phase = TAU * coordinate.rem_euclid(domain_length) / domain_length;
    let embedding = [phase.sin(), phase.cos()];
    if embedding.iter().any(|value| !value.is_finite()) {
        return Err(CoordinateAuditError::InvalidDomainLength);
    }
    Ok(embedding)
}

/// Maximum absolute movement among all particles except the intentionally
/// changed one. Useful for measuring sample-normalization coupling.
pub fn max_unaffected_coordinate_shift(
    before: &[f64],
    after: &[f64],
    changed_index: usize,
) -> Result<f64, CoordinateAuditError> {
    if before.len() != after.len() {
        return Err(CoordinateAuditError::LengthMismatch {
            before: before.len(),
            after: after.len(),
        });
    }
    validate_coordinates(before)?;
    validate_coordinates(after)?;
    if changed_index >= before.len() {
        return Err(CoordinateAuditError::ChangedIndexOutOfRange {
            index: changed_index,
            length: before.len(),
        });
    }

    Ok(before
        .iter()
        .zip(after.iter())
        .enumerate()
        .filter(|(index, _)| *index != changed_index)
        .map(|(_, (left, right))| (left - right).abs())
        .fold(0.0, f64::max))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
        );
    }

    fn embedding_distance(left: [f64; 2], right: [f64; 2]) -> f64 {
        let dx = left[0] - right[0];
        let dy = left[1] - right[1];
        (dx * dx + dy * dy).sqrt()
    }

    #[test]
    fn changing_one_sample_extremum_moves_unaffected_minmax_coordinates() {
        let before = [0.0, 0.25, 0.5, 0.75, 1.0];
        let after = [0.0, 0.25, 0.5, 0.75, 2.0];
        let normalized_before = sample_minmax_normalize(&before).unwrap();
        let normalized_after = sample_minmax_normalize(&after).unwrap();
        let shift =
            max_unaffected_coordinate_shift(&normalized_before, &normalized_after, 4).unwrap();

        close(normalized_before[1], 0.25, 1.0e-15);
        close(normalized_after[1], 0.125, 1.0e-15);
        close(shift, 0.375, 1.0e-15);
    }

    #[test]
    fn minmax_extremal_perturbation_is_not_a_single_particle_count_artifact() {
        for particle_count in [4_usize, 16, 64, 256] {
            let before: Vec<f64> = (0..particle_count)
                .map(|index| index as f64 / (particle_count - 1) as f64)
                .collect();
            let mut after = before.clone();
            after[particle_count - 1] = 2.0;
            let normalized_before = sample_minmax_normalize(&before).unwrap();
            let normalized_after = sample_minmax_normalize(&after).unwrap();
            let shift = max_unaffected_coordinate_shift(
                &normalized_before,
                &normalized_after,
                particle_count - 1,
            )
            .unwrap();
            assert!(shift > 0.49, "N={particle_count}, shift={shift}");
        }
    }

    #[test]
    fn fixed_periodic_domain_map_does_not_couple_other_particles() {
        let before = [0.0, 0.25, 0.5, 0.75, 1.0];
        let after = [0.0, 0.25, 0.5, 0.75, 2.0];
        let normalized_before =
            fixed_periodic_domain_normalize(&before, 0.0, 2.0).unwrap();
        let normalized_after =
            fixed_periodic_domain_normalize(&after, 0.0, 2.0).unwrap();
        let shift =
            max_unaffected_coordinate_shift(&normalized_before, &normalized_after, 4).unwrap();
        close(shift, 0.0, 0.0);
    }

    #[test]
    fn periodic_embedding_is_continuous_across_the_domain_seam() {
        let length = 7.0;
        let left = periodic_coordinate_embedding(length - 1.0e-9, length).unwrap();
        let right = periodic_coordinate_embedding(0.0, length).unwrap();
        assert!(embedding_distance(left, right) < 1.0e-8);
    }

    #[test]
    fn periodic_embedding_is_invariant_under_integer_domain_translations() {
        let length = 5.5;
        let base = periodic_coordinate_embedding(1.25, length).unwrap();
        for turns in [-3.0, -1.0, 1.0, 4.0] {
            let translated =
                periodic_coordinate_embedding(1.25 + turns * length, length).unwrap();
            close(base[0], translated[0], 1.0e-14);
            close(base[1], translated[1], 1.0e-14);
        }
    }

    #[test]
    fn raw_fixed_coordinate_is_not_periodic_at_the_seam() {
        let length = 10.0;
        let left = (length - 1.0e-9) / length;
        let right = 0.0 / length;
        assert!((left - right).abs() > 0.99);
    }

    #[test]
    fn rejects_empty_constant_non_finite_and_invalid_domain_inputs() {
        assert_eq!(
            sample_minmax_normalize(&[]).unwrap_err(),
            CoordinateAuditError::EmptyCoordinates
        );
        assert_eq!(
            sample_minmax_normalize(&[1.0, 1.0]).unwrap_err(),
            CoordinateAuditError::ZeroSampleSpan
        );
        assert_eq!(
            sample_minmax_normalize(&[0.0, f64::NAN]).unwrap_err(),
            CoordinateAuditError::NonFiniteCoordinate { index: 1 }
        );
        assert_eq!(
            fixed_periodic_domain_normalize(&[0.0], 0.0, 0.0).unwrap_err(),
            CoordinateAuditError::InvalidDomainLength
        );
        assert_eq!(
            periodic_coordinate_embedding(0.0, f64::INFINITY).unwrap_err(),
            CoordinateAuditError::InvalidDomainLength
        );
    }

    #[test]
    fn rejects_mismatched_lengths_and_invalid_changed_index() {
        assert_eq!(
            max_unaffected_coordinate_shift(&[0.0, 1.0], &[0.0], 0).unwrap_err(),
            CoordinateAuditError::LengthMismatch {
                before: 2,
                after: 1,
            }
        );
        assert_eq!(
            max_unaffected_coordinate_shift(&[0.0, 1.0], &[0.0, 1.0], 2).unwrap_err(),
            CoordinateAuditError::ChangedIndexOutOfRange {
                index: 2,
                length: 2,
            }
        );
    }
}
