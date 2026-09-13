// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Fallible coordinate-dependent Riemannian metric fields.
//!
//! This module is additive to the existing HDC/Einstein-search `MetricTensor`.
//! It does not change that type's historical constructor or inverse semantics.
//! Claim-bearing manifold code should obtain tensors through [`sample_metric`],
//! which validates dimensions, finiteness, symmetry, positive definiteness and
//! conditioning under an exact identity-bound policy. [`MetricSample::try_inverse`]
//! provides a fail-closed inverse rather than any singular-to-identity fallback.

use blake3::Hasher;
use nalgebra::{DMatrix, SymmetricEigen};
use thiserror::Error;

use crate::hdc::riemannian_geometry::MetricTensor;
use crate::state_space::{EuclideanSpace, StateSpace};

/// Maximum coordinate dimension admitted by the V1 validated metric-field layer.
pub const MAX_METRIC_FIELD_DIMENSION: usize = 64;

/// Exact numerical qualification policy for metric samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricValidationPolicy {
    symmetry_tolerance: f64,
    positive_eigenvalue_floor: f64,
    max_condition_number: f64,
}

impl MetricValidationPolicy {
    /// Construct an exact metric validation policy.
    pub fn new(
        symmetry_tolerance: f64,
        positive_eigenvalue_floor: f64,
        max_condition_number: f64,
    ) -> Result<Self, MetricFieldError> {
        if !symmetry_tolerance.is_finite() || symmetry_tolerance < 0.0 {
            return Err(MetricFieldError::InvalidProfile {
                reason: format!(
                    "symmetry tolerance must be finite and >= 0, got {symmetry_tolerance}"
                ),
            });
        }
        if !positive_eigenvalue_floor.is_finite() || positive_eigenvalue_floor <= 0.0 {
            return Err(MetricFieldError::InvalidProfile {
                reason: format!(
                    "positive eigenvalue floor must be finite and > 0, got {positive_eigenvalue_floor}"
                ),
            });
        }
        if !max_condition_number.is_finite() || max_condition_number < 1.0 {
            return Err(MetricFieldError::InvalidProfile {
                reason: format!(
                    "max condition number must be finite and >= 1, got {max_condition_number}"
                ),
            });
        }
        Ok(Self {
            symmetry_tolerance,
            positive_eigenvalue_floor,
            max_condition_number,
        })
    }

    /// Named deterministic software-reference policy.
    pub fn reference_v1() -> Self {
        Self::new(1e-12, 1e-12, 1e12).expect("reference metric validation policy is valid")
    }

    /// Maximum accepted `|g_ij - g_ji|` before symmetrization is rejected.
    pub fn symmetry_tolerance(&self) -> f64 {
        self.symmetry_tolerance
    }

    /// Strict lower eigenvalue bound for positive-definite qualification.
    pub fn positive_eigenvalue_floor(&self) -> f64 {
        self.positive_eigenvalue_floor
    }

    /// Maximum admitted spectral condition number.
    pub fn max_condition_number(&self) -> f64 {
        self.max_condition_number
    }

    /// Deterministic identity of the exact validation policy.
    pub fn identity(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-metric-validation-policy-v1\0");
        hasher.update(&self.symmetry_tolerance.to_bits().to_le_bytes());
        hasher.update(&self.positive_eigenvalue_floor.to_bits().to_le_bytes());
        hasher.update(&self.max_condition_number.to_bits().to_le_bytes());
        *hasher.finalize().as_bytes()
    }
}

impl Default for MetricValidationPolicy {
    fn default() -> Self {
        Self::reference_v1()
    }
}

/// Exact identity-bearing descriptor for one coordinate-dependent metric field.
#[derive(Clone, Debug, PartialEq)]
pub struct MetricFieldProfile {
    dimension: usize,
    coordinate_space_identity: [u8; 32],
    kind: String,
    parameters: Vec<u8>,
    validation_policy: MetricValidationPolicy,
    identity: [u8; 32],
}

impl MetricFieldProfile {
    /// Construct a profile over an exact coordinate-space identity.
    pub fn new(
        dimension: usize,
        coordinate_space_identity: [u8; 32],
        kind: impl Into<String>,
        parameters: Vec<u8>,
        validation_policy: MetricValidationPolicy,
    ) -> Result<Self, MetricFieldError> {
        if dimension == 0 || dimension > MAX_METRIC_FIELD_DIMENSION {
            return Err(MetricFieldError::InvalidProfile {
                reason: format!(
                    "metric dimension must be in 1..={MAX_METRIC_FIELD_DIMENSION}, got {dimension}"
                ),
            });
        }
        let kind = kind.into();
        if kind.trim().is_empty() {
            return Err(MetricFieldError::InvalidProfile {
                reason: "metric field kind must not be empty".to_string(),
            });
        }

        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-metric-field-profile-v1\0");
        hasher.update(&(dimension as u64).to_le_bytes());
        hasher.update(&coordinate_space_identity);
        update_len_prefixed(&mut hasher, kind.as_bytes());
        update_len_prefixed(&mut hasher, &parameters);
        hasher.update(&validation_policy.identity());
        let identity = *hasher.finalize().as_bytes();

        Ok(Self {
            dimension,
            coordinate_space_identity,
            kind,
            parameters,
            validation_policy,
            identity,
        })
    }

    /// Coordinate dimension of this field.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Exact identity of the coordinate/state-space profile the field is defined on.
    pub fn coordinate_space_identity(&self) -> [u8; 32] {
        self.coordinate_space_identity
    }

    /// Stable field-kind identifier.
    pub fn kind(&self) -> &str {
        &self.kind
    }

    /// Opaque exact parameters bound into field identity.
    pub fn parameters(&self) -> &[u8] {
        &self.parameters
    }

    /// Exact sample qualification policy.
    pub fn validation_policy(&self) -> MetricValidationPolicy {
        self.validation_policy
    }

    /// Deterministic identity of this exact field/profile/policy combination.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

fn update_len_prefixed(hasher: &mut Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Domain classification for a coordinate-dependent metric field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetricDomainStatus {
    /// Point lies inside the declared field domain.
    Inside,
    /// Point lies outside the declared field domain.
    Outside { reason: String },
}

/// Source contract for a coordinate-dependent metric.
///
/// Implementations provide raw row-major components. Consumers should call
/// [`sample_metric`] rather than treating raw components as a qualified tensor.
pub trait MetricField {
    /// Exact field/profile identity.
    fn profile(&self) -> &MetricFieldProfile;

    /// Domain classification at one finite, correctly dimensioned coordinate.
    fn domain_status(&self, _point: &[f64]) -> Result<MetricDomainStatus, MetricFieldError> {
        Ok(MetricDomainStatus::Inside)
    }

    /// Raw row-major metric components at one point.
    fn raw_components(&self, point: &[f64]) -> Result<Vec<f64>, MetricFieldError>;
}

/// Fail-closed errors produced by metric-field qualification.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum MetricFieldError {
    /// Profile construction failed.
    #[error("invalid metric-field profile: {reason}")]
    InvalidProfile { reason: String },
    /// Coordinate vector dimension disagrees with the exact field profile.
    #[error("metric point dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    /// A coordinate is NaN or infinite.
    #[error("non-finite coordinate {coordinate}")]
    NonFiniteCoordinate { coordinate: usize },
    /// Field domain explicitly rejects the point.
    #[error("metric point is outside declared domain: {reason}")]
    OutsideDomain { reason: String },
    /// Raw component count disagrees with `dimension^2`.
    #[error("metric component count mismatch: expected {expected}, got {actual}")]
    ComponentCountMismatch { expected: usize, actual: usize },
    /// A raw metric component is NaN or infinite.
    #[error("non-finite metric component at flat index {index}")]
    NonFiniteComponent { index: usize },
    /// Metric sample exceeds declared symmetry tolerance.
    #[error(
        "metric is asymmetric at ({row}, {column}): |g_ij-g_ji|={difference} exceeds tolerance {tolerance}"
    )]
    Asymmetric {
        row: usize,
        column: usize,
        difference: f64,
        tolerance: f64,
    },
    /// Metric fails the strict positive-definite eigenvalue policy.
    #[error(
        "metric is not qualified positive definite: minimum eigenvalue {minimum_eigenvalue} <= floor {floor}"
    )]
    NotPositiveDefinite {
        minimum_eigenvalue: f64,
        floor: f64,
    },
    /// Spectral condition number exceeds the exact policy.
    #[error(
        "metric condition number {condition_number} exceeds maximum {maximum_condition_number}"
    )]
    IllConditioned {
        condition_number: f64,
        maximum_condition_number: f64,
    },
    /// Eigenvalue qualification produced non-finite arithmetic.
    #[error("non-finite spectral arithmetic while qualifying metric")]
    NonFiniteSpectrum,
    /// A qualified tensor still failed fallible numerical inversion.
    #[error("qualified metric inversion failed")]
    InversionFailed,
    /// Inversion produced a non-finite component.
    #[error("metric inverse produced non-finite component at flat index {index}")]
    NonFiniteInverse { index: usize },
    /// Field-specific evaluation failed without a stronger standard category.
    #[error("metric-field evaluation failed: {reason}")]
    Evaluation { reason: String },
}

/// Qualified metric tensor plus numerical validation receipt.
#[derive(Clone, Debug)]
pub struct MetricSample {
    point: Vec<f64>,
    tensor: MetricTensor,
    minimum_eigenvalue: f64,
    maximum_eigenvalue: f64,
    condition_number: f64,
    maximum_asymmetry: f64,
    field_identity: [u8; 32],
    sample_identity: [u8; 32],
}

impl MetricSample {
    /// Exact coordinate point sampled, with signed zero canonicalized.
    pub fn point(&self) -> &[f64] {
        &self.point
    }

    /// Validated, symmetrized metric tensor.
    pub fn tensor(&self) -> &MetricTensor {
        &self.tensor
    }

    /// Minimum eigenvalue of the validated symmetric sample.
    pub fn minimum_eigenvalue(&self) -> f64 {
        self.minimum_eigenvalue
    }

    /// Maximum eigenvalue of the validated symmetric sample.
    pub fn maximum_eigenvalue(&self) -> f64 {
        self.maximum_eigenvalue
    }

    /// Spectral condition number `lambda_max / lambda_min`.
    pub fn condition_number(&self) -> f64 {
        self.condition_number
    }

    /// Largest raw pre-symmetrization `|g_ij - g_ji|` observed.
    pub fn maximum_asymmetry(&self) -> f64 {
        self.maximum_asymmetry
    }

    /// Exact field profile identity.
    pub fn field_identity(&self) -> [u8; 32] {
        self.field_identity
    }

    /// Identity binding field, canonical point and validated tensor components.
    pub fn identity(&self) -> [u8; 32] {
        self.sample_identity
    }

    /// Fail-closed inverse of this already-qualified positive-definite sample.
    ///
    /// This intentionally does not call the historical `MetricTensor::inverse`,
    /// whose specialized research semantics may return Euclidean identity at a
    /// singular pivot.
    pub fn try_inverse(&self) -> Result<MetricTensor, MetricFieldError> {
        let dimension = self.tensor.dim;
        let matrix = DMatrix::from_row_slice(dimension, dimension, &self.tensor.components);
        let inverse = matrix
            .try_inverse()
            .ok_or(MetricFieldError::InversionFailed)?;

        // nalgebra uses column-major internal storage. Export explicitly by
        // `(row, column)` so MetricTensor receives its documented row-major form.
        // Average symmetric pairs to remove inversion roundoff asymmetry.
        let mut components = vec![0.0_f64; dimension * dimension];
        for row in 0..dimension {
            for column in 0..dimension {
                let value = 0.5 * (inverse[(row, column)] + inverse[(column, row)]);
                if !value.is_finite() {
                    return Err(MetricFieldError::NonFiniteInverse {
                        index: row * dimension + column,
                    });
                }
                components[row * dimension + column] = value + 0.0;
            }
        }
        Ok(MetricTensor::new(components, dimension))
    }
}

/// Evaluate and fully qualify one metric sample.
pub fn sample_metric<F: MetricField>(
    field: &F,
    point: &[f64],
) -> Result<MetricSample, MetricFieldError> {
    let profile = field.profile();
    let dimension = profile.dimension;
    if point.len() != dimension {
        return Err(MetricFieldError::DimensionMismatch {
            expected: dimension,
            actual: point.len(),
        });
    }
    for (coordinate, value) in point.iter().enumerate() {
        if !value.is_finite() {
            return Err(MetricFieldError::NonFiniteCoordinate { coordinate });
        }
    }
    match field.domain_status(point)? {
        MetricDomainStatus::Inside => {}
        MetricDomainStatus::Outside { reason } => {
            return Err(MetricFieldError::OutsideDomain { reason });
        }
    }

    let raw = field.raw_components(point)?;
    let expected = dimension * dimension;
    if raw.len() != expected {
        return Err(MetricFieldError::ComponentCountMismatch {
            expected,
            actual: raw.len(),
        });
    }
    if let Some(index) = raw.iter().position(|value| !value.is_finite()) {
        return Err(MetricFieldError::NonFiniteComponent { index });
    }

    let policy = profile.validation_policy;
    let mut symmetric = raw.clone();
    let mut maximum_asymmetry = 0.0_f64;
    for row in 0..dimension {
        for column in (row + 1)..dimension {
            let left = raw[row * dimension + column];
            let right = raw[column * dimension + row];
            let difference = (left - right).abs();
            maximum_asymmetry = maximum_asymmetry.max(difference);
            if difference > policy.symmetry_tolerance {
                return Err(MetricFieldError::Asymmetric {
                    row,
                    column,
                    difference,
                    tolerance: policy.symmetry_tolerance,
                });
            }
            // Canonicalize accepted floating asymmetry under the declared policy.
            let average = 0.5 * (left + right);
            if !average.is_finite() {
                return Err(MetricFieldError::NonFiniteComponent {
                    index: row * dimension + column,
                });
            }
            symmetric[row * dimension + column] = average;
            symmetric[column * dimension + row] = average;
        }
    }
    for value in &mut symmetric {
        *value += 0.0;
    }

    let matrix = DMatrix::from_row_slice(dimension, dimension, &symmetric);
    let eigen = SymmetricEigen::new(matrix);
    if eigen.eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(MetricFieldError::NonFiniteSpectrum);
    }
    let minimum_eigenvalue = eigen
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let maximum_eigenvalue = eigen
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if minimum_eigenvalue <= policy.positive_eigenvalue_floor {
        return Err(MetricFieldError::NotPositiveDefinite {
            minimum_eigenvalue,
            floor: policy.positive_eigenvalue_floor,
        });
    }
    let condition_number = maximum_eigenvalue / minimum_eigenvalue;
    if !condition_number.is_finite() {
        return Err(MetricFieldError::NonFiniteSpectrum);
    }
    if condition_number > policy.max_condition_number {
        return Err(MetricFieldError::IllConditioned {
            condition_number,
            maximum_condition_number: policy.max_condition_number,
        });
    }

    let tensor = MetricTensor::new(symmetric, dimension);
    let point: Vec<f64> = point.iter().map(|value| *value + 0.0).collect();
    let field_identity = profile.identity;
    let sample_identity = hash_metric_sample(field_identity, &point, &tensor.components);
    Ok(MetricSample {
        point,
        tensor,
        minimum_eigenvalue,
        maximum_eigenvalue,
        condition_number,
        maximum_asymmetry,
        field_identity,
        sample_identity,
    })
}

fn hash_metric_sample(
    field_identity: [u8; 32],
    point: &[f64],
    components: &[f64],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-qualified-metric-sample-v1\0");
    hasher.update(&field_identity);
    hasher.update(&(point.len() as u64).to_le_bytes());
    for value in point {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    hasher.update(&(components.len() as u64).to_le_bytes());
    for value in components {
        hasher.update(&value.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

/// Analytic Poincare disk metric on a bounded subset of the unit disk.
///
/// In Cartesian coordinates `(x, y)`:
/// `g = 4 / (1 - x^2 - y^2)^2 * I`.
#[derive(Clone, Debug)]
pub struct PoincareDiskMetricField {
    maximum_radius: f64,
    profile: MetricFieldProfile,
}

impl PoincareDiskMetricField {
    /// Construct a field restricted to `sqrt(x^2+y^2) <= maximum_radius < 1`.
    pub fn new(
        maximum_radius: f64,
        validation_policy: MetricValidationPolicy,
    ) -> Result<Self, MetricFieldError> {
        if !maximum_radius.is_finite() || maximum_radius <= 0.0 || maximum_radius >= 1.0 {
            return Err(MetricFieldError::InvalidProfile {
                reason: format!(
                    "Poincare maximum radius must be finite and in (0, 1), got {maximum_radius}"
                ),
            });
        }
        let coordinate_space = EuclideanSpace::new(2);
        let profile = MetricFieldProfile::new(
            2,
            coordinate_space.profile().identity(),
            "poincare-disk-cartesian-v1",
            maximum_radius.to_bits().to_le_bytes().to_vec(),
            validation_policy,
        )?;
        Ok(Self {
            maximum_radius,
            profile,
        })
    }

    /// Maximum admitted Euclidean radius of the exact coordinate domain.
    pub fn maximum_radius(&self) -> f64 {
        self.maximum_radius
    }
}

impl MetricField for PoincareDiskMetricField {
    fn profile(&self) -> &MetricFieldProfile {
        &self.profile
    }

    fn domain_status(&self, point: &[f64]) -> Result<MetricDomainStatus, MetricFieldError> {
        let radius = point[0].hypot(point[1]);
        if !radius.is_finite() {
            return Err(MetricFieldError::Evaluation {
                reason: "Poincare radius became non-finite".to_string(),
            });
        }
        if radius <= self.maximum_radius {
            Ok(MetricDomainStatus::Inside)
        } else {
            Ok(MetricDomainStatus::Outside {
                reason: format!(
                    "radius {radius} exceeds declared maximum {}",
                    self.maximum_radius
                ),
            })
        }
    }

    fn raw_components(&self, point: &[f64]) -> Result<Vec<f64>, MetricFieldError> {
        let radius_squared = point[0] * point[0] + point[1] * point[1];
        let denominator = 1.0 - radius_squared;
        let scale = 4.0 / (denominator * denominator);
        if !scale.is_finite() {
            return Err(MetricFieldError::Evaluation {
                reason: "Poincare conformal factor became non-finite".to_string(),
            });
        }
        Ok(vec![scale, 0.0, 0.0, scale])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug)]
    struct FixtureField {
        profile: MetricFieldProfile,
        components: Vec<f64>,
    }

    impl FixtureField {
        fn new(components: Vec<f64>, policy: MetricValidationPolicy) -> Self {
            let coordinate_space = EuclideanSpace::new(2);
            Self {
                profile: MetricFieldProfile::new(
                    2,
                    coordinate_space.profile().identity(),
                    "metric-field-test-fixture-v1",
                    Vec::new(),
                    policy,
                )
                .unwrap(),
                components,
            }
        }
    }

    impl MetricField for FixtureField {
        fn profile(&self) -> &MetricFieldProfile {
            &self.profile
        }

        fn raw_components(&self, _point: &[f64]) -> Result<Vec<f64>, MetricFieldError> {
            Ok(self.components.clone())
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn poincare_metric_matches_independent_analytic_samples() {
        let field = PoincareDiskMetricField::new(0.9, MetricValidationPolicy::reference_v1())
            .unwrap();
        let origin = sample_metric(&field, &[0.0, 0.0]).unwrap();
        close(origin.tensor().g(0, 0), 4.0, 1e-12);
        close(origin.tensor().g(1, 1), 4.0, 1e-12);
        close(origin.condition_number(), 1.0, 1e-12);

        let half = sample_metric(&field, &[0.5, 0.0]).unwrap();
        let expected = 4.0 / 0.75_f64.powi(2);
        close(half.tensor().g(0, 0), expected, 1e-12);
        close(half.tensor().g(1, 1), expected, 1e-12);
    }

    #[test]
    fn metric_domain_is_explicit_and_fail_closed() {
        let field = PoincareDiskMetricField::new(0.8, MetricValidationPolicy::reference_v1())
            .unwrap();
        let error = sample_metric(&field, &[0.9, 0.0]).unwrap_err();
        assert!(matches!(error, MetricFieldError::OutsideDomain { .. }));
    }

    #[test]
    fn asymmetric_metric_is_rejected_not_silently_rewritten() {
        let field = FixtureField::new(
            vec![1.0, 0.1, 0.0, 1.0],
            MetricValidationPolicy::reference_v1(),
        );
        let error = sample_metric(&field, &[0.0, 0.0]).unwrap_err();
        assert!(matches!(error, MetricFieldError::Asymmetric { .. }));
    }

    #[test]
    fn tiny_asymmetry_is_canonicalized_under_exact_policy() {
        let policy = MetricValidationPolicy::new(1e-6, 1e-12, 1e12).unwrap();
        let field = FixtureField::new(vec![2.0, 2e-8, 0.0, 3.0], policy);
        let sample = sample_metric(&field, &[0.0, 0.0]).unwrap();
        close(sample.tensor().g(0, 1), 1e-8, 1e-18);
        close(sample.tensor().g(1, 0), 1e-8, 1e-18);
        close(sample.maximum_asymmetry(), 2e-8, 1e-18);
    }

    #[test]
    fn indefinite_and_singular_metrics_are_rejected() {
        for components in [vec![1.0, 0.0, 0.0, -1.0], vec![1.0, 0.0, 0.0, 0.0]] {
            let field = FixtureField::new(components, MetricValidationPolicy::reference_v1());
            let error = sample_metric(&field, &[0.0, 0.0]).unwrap_err();
            assert!(matches!(error, MetricFieldError::NotPositiveDefinite { .. }));
        }
    }

    #[test]
    fn ill_conditioned_metric_is_policy_rejected() {
        let policy = MetricValidationPolicy::new(0.0, 1e-15, 100.0).unwrap();
        let field = FixtureField::new(vec![1.0, 0.0, 0.0, 1e-4], policy);
        let error = sample_metric(&field, &[0.0, 0.0]).unwrap_err();
        assert!(matches!(error, MetricFieldError::IllConditioned { .. }));
    }

    #[test]
    fn qualified_inverse_is_fallible_and_row_major_correct() {
        let field = FixtureField::new(
            vec![2.0, 1.0, 1.0, 3.0],
            MetricValidationPolicy::reference_v1(),
        );
        let sample = sample_metric(&field, &[0.0, 0.0]).unwrap();
        let inverse = sample.try_inverse().unwrap();
        close(inverse.g(0, 0), 0.6, 1e-12);
        close(inverse.g(0, 1), -0.2, 1e-12);
        close(inverse.g(1, 0), -0.2, 1e-12);
        close(inverse.g(1, 1), 0.4, 1e-12);
    }

    #[test]
    fn signed_zero_does_not_split_metric_sample_identity() {
        let positive = FixtureField::new(
            vec![2.0, 0.0, 0.0, 3.0],
            MetricValidationPolicy::reference_v1(),
        );
        let negative = FixtureField::new(
            vec![2.0, -0.0, -0.0, 3.0],
            MetricValidationPolicy::reference_v1(),
        );
        let a = sample_metric(&positive, &[0.0, 0.0]).unwrap();
        let b = sample_metric(&negative, &[-0.0, 0.0]).unwrap();
        assert_eq!(positive.profile().identity(), negative.profile().identity());
        assert_eq!(a.identity(), b.identity());
    }

    #[test]
    fn validation_policy_and_domain_change_field_identity() {
        let a = PoincareDiskMetricField::new(0.8, MetricValidationPolicy::reference_v1()).unwrap();
        let b = PoincareDiskMetricField::new(0.9, MetricValidationPolicy::reference_v1()).unwrap();
        let strict = MetricValidationPolicy::new(1e-14, 1e-12, 1e12).unwrap();
        let c = PoincareDiskMetricField::new(0.8, strict).unwrap();
        assert_ne!(a.profile().identity(), b.profile().identity());
        assert_ne!(a.profile().identity(), c.profile().identity());
    }

    #[test]
    fn non_finite_components_fail_closed() {
        let field = FixtureField::new(
            vec![1.0, 0.0, 0.0, f64::NAN],
            MetricValidationPolicy::reference_v1(),
        );
        assert_eq!(
            sample_metric(&field, &[0.0, 0.0]).unwrap_err(),
            MetricFieldError::NonFiniteComponent { index: 3 }
        );
    }
}