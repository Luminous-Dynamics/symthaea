// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic support-function geometry for positive-definite ellipsoids.
//!
//! This module consumes an already-qualified ellipsoid. It does not estimate
//! its center or shape, establish confidence coverage, authenticate provenance,
//! or assert that the geometry came from a valid statistical theorem.
//!
//! The shape is represented by a lower-triangular factor L with positive
//! diagonal, so the ellipsoid is:
//!
//!     E = { center + L y : ||y||_2 <= radius }.
//!
//! For any direction c, its support interval is:
//!
//!     c' center ± radius * ||L' c||_2.
//!
//! The single upstream ellipsoid event covers every deterministic direction;
//! no extra multiplicity correction is required for post-processing directions
//! once the ellipsoid itself has valid simultaneous coverage.

use std::collections::BTreeSet;
use std::fmt;

pub const ELLIPSOID_SUPPORT_GEOMETRY_METHOD_ID: &str =
    "spectra-positive-definite-ellipsoid-support-geometry-v1";

#[derive(Debug, Clone, PartialEq)]
pub struct EllipsoidGeometry {
    /// Frozen semantic identity and order of vector coordinates.
    pub coordinate_ids: Vec<String>,
    /// Center of the ellipsoid in the frozen coordinate order.
    pub center: Vec<f64>,
    /// Lower-triangular shape factor L, defining shape matrix L Lᵀ.
    pub lower_shape_factor: Vec<Vec<f64>>,
    /// Euclidean radius in the whitened y-coordinate system.
    pub radius: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinearDirection {
    /// Stable ID for a preregistered downstream functional.
    pub direction_id: String,
    /// Coefficients in exactly the geometry's coordinate order.
    pub coefficients: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalSupportInterval {
    pub center_projection: f64,
    pub support_radius: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DirectionalSupportResult {
    pub method_id: &'static str,
    pub direction_id: String,
    pub coordinate_ids: Vec<String>,
    pub interval: DirectionalSupportInterval,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EllipsoidGeometryError {
    EmptyCoordinates,
    EmptyCoordinateId,
    DuplicateCoordinateId,
    CenterDimensionMismatch,
    ShapeDimensionMismatch,
    ShapeRowDimensionMismatch,
    NonFiniteCenter,
    NonFiniteShapeFactor,
    ShapeFactorNotLowerTriangular,
    ShapeFactorNotPositiveDefinite,
    ShapeMatrixNotSymmetric,
    ShapeMatrixNotPositiveDefinite,
    InvalidRadius,
    EmptyDirectionId,
    InvalidDirectionDimension,
    NonFiniteDirection,
    DuplicateDirectionId,
    EmptyDirectionSet,
    NumericFailure,
    ReplayMismatch,
}

impl fmt::Display for EllipsoidGeometryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::EmptyCoordinates => "ellipsoid coordinate set must be nonempty",
            Self::EmptyCoordinateId => "coordinate IDs must be nonempty",
            Self::DuplicateCoordinateId => "coordinate IDs must be unique",
            Self::CenterDimensionMismatch => "ellipsoid center dimension does not match coordinate IDs",
            Self::ShapeDimensionMismatch => "ellipsoid shape factor row count does not match coordinate dimension",
            Self::ShapeRowDimensionMismatch => "ellipsoid shape factor must be square",
            Self::NonFiniteCenter => "ellipsoid center must contain only finite values",
            Self::NonFiniteShapeFactor => "ellipsoid shape factor must contain only finite values",
            Self::ShapeFactorNotLowerTriangular => "ellipsoid shape factor must be lower triangular",
            Self::ShapeFactorNotPositiveDefinite => "ellipsoid shape factor diagonal must be strictly positive",
            Self::ShapeMatrixNotSymmetric => "ellipsoid shape matrix must be exactly symmetric",
            Self::ShapeMatrixNotPositiveDefinite => "ellipsoid shape matrix must be positive definite",
            Self::InvalidRadius => "ellipsoid radius must be finite and nonnegative",
            Self::EmptyDirectionId => "direction ID must be nonempty",
            Self::InvalidDirectionDimension => "direction dimension does not match ellipsoid coordinates",
            Self::NonFiniteDirection => "direction coefficients must be finite",
            Self::DuplicateDirectionId => "direction IDs must be unique within one support batch",
            Self::EmptyDirectionSet => "support evaluation requires at least one direction",
            Self::NumericFailure => "ellipsoid support calculation is not representable",
            Self::ReplayMismatch => "recorded directional supports differ from deterministic recomputation",
        };
        f.write_str(message)
    }
}
impl std::error::Error for EllipsoidGeometryError {}

impl EllipsoidGeometry {
    /// Build geometry from an explicitly symmetric positive-definite shape matrix.
    ///
    /// This performs a strict Cholesky factorization. It does not add jitter,
    /// clip eigenvalues or regularize singular inputs; those operations change
    /// the inferential set and belong to the upstream statistical method.
    pub fn from_shape_matrix(
        coordinate_ids: Vec<String>,
        center: Vec<f64>,
        shape_matrix: &[Vec<f64>],
        radius: f64,
    ) -> Result<Self, EllipsoidGeometryError> {
        let dimension = coordinate_ids.len();
        if dimension == 0 {
            return Err(EllipsoidGeometryError::EmptyCoordinates);
        }
        if shape_matrix.len() != dimension {
            return Err(EllipsoidGeometryError::ShapeDimensionMismatch);
        }
        if shape_matrix.iter().any(|row| row.len() != dimension) {
            return Err(EllipsoidGeometryError::ShapeRowDimensionMismatch);
        }
        if shape_matrix.iter().flatten().any(|value| !value.is_finite()) {
            return Err(EllipsoidGeometryError::NonFiniteShapeFactor);
        }
        for row in 0..dimension {
            for col in 0..row {
                if shape_matrix[row][col] != shape_matrix[col][row] {
                    return Err(EllipsoidGeometryError::ShapeMatrixNotSymmetric);
                }
            }
        }

        let mut lower = vec![vec![0.0; dimension]; dimension];
        for row in 0..dimension {
            for col in 0..=row {
                let mut value = shape_matrix[row][col];
                for index in 0..col {
                    value -= lower[row][index] * lower[col][index];
                }
                if !value.is_finite() {
                    return Err(EllipsoidGeometryError::NumericFailure);
                }
                if row == col {
                    if value <= 0.0 {
                        return Err(EllipsoidGeometryError::ShapeMatrixNotPositiveDefinite);
                    }
                    lower[row][col] = value.sqrt();
                } else {
                    let factor = value / lower[col][col];
                    if !factor.is_finite() {
                        return Err(EllipsoidGeometryError::NumericFailure);
                    }
                    lower[row][col] = factor;
                }
            }
        }

        let geometry = Self {
            coordinate_ids,
            center,
            lower_shape_factor: lower,
            radius,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    pub fn validate(&self) -> Result<(), EllipsoidGeometryError> {
        let dimension = self.coordinate_ids.len();
        if dimension == 0 {
            return Err(EllipsoidGeometryError::EmptyCoordinates);
        }
        if self.coordinate_ids.iter().any(|id| id.trim().is_empty()) {
            return Err(EllipsoidGeometryError::EmptyCoordinateId);
        }
        let mut ids = BTreeSet::new();
        if self.coordinate_ids.iter().any(|id| !ids.insert(id.as_str())) {
            return Err(EllipsoidGeometryError::DuplicateCoordinateId);
        }
        if self.center.len() != dimension {
            return Err(EllipsoidGeometryError::CenterDimensionMismatch);
        }
        if self.lower_shape_factor.len() != dimension {
            return Err(EllipsoidGeometryError::ShapeDimensionMismatch);
        }
        if self.lower_shape_factor.iter().any(|row| row.len() != dimension) {
            return Err(EllipsoidGeometryError::ShapeRowDimensionMismatch);
        }
        if self.center.iter().any(|value| !value.is_finite()) {
            return Err(EllipsoidGeometryError::NonFiniteCenter);
        }
        if self.lower_shape_factor.iter().flatten().any(|value| !value.is_finite()) {
            return Err(EllipsoidGeometryError::NonFiniteShapeFactor);
        }
        for row in 0..dimension {
            if self.lower_shape_factor[row][row] <= 0.0 {
                return Err(EllipsoidGeometryError::ShapeFactorNotPositiveDefinite);
            }
            for col in (row + 1)..dimension {
                if self.lower_shape_factor[row][col] != 0.0 {
                    return Err(EllipsoidGeometryError::ShapeFactorNotLowerTriangular);
                }
            }
        }
        if !self.radius.is_finite() || self.radius < 0.0 {
            return Err(EllipsoidGeometryError::InvalidRadius);
        }
        Ok(())
    }

    pub fn support(
        &self,
        direction: &LinearDirection,
    ) -> Result<DirectionalSupportResult, EllipsoidGeometryError> {
        self.validate()?;
        validate_direction(direction, self.coordinate_ids.len())?;
        let interval = support_interval(self, &direction.coefficients)?;
        Ok(DirectionalSupportResult {
            method_id: ELLIPSOID_SUPPORT_GEOMETRY_METHOD_ID,
            direction_id: direction.direction_id.clone(),
            coordinate_ids: self.coordinate_ids.clone(),
            interval,
        })
    }

    /// Evaluate a frozen family of directions and preserve canonical ID order.
    ///
    /// Because these are all deterministic support queries on one already-valid
    /// ellipsoid, the family does not consume another statistical alpha budget.
    pub fn support_many(
        &self,
        directions: &[LinearDirection],
    ) -> Result<Vec<DirectionalSupportResult>, EllipsoidGeometryError> {
        self.validate()?;
        if directions.is_empty() {
            return Err(EllipsoidGeometryError::EmptyDirectionSet);
        }
        let mut sorted = directions.to_vec();
        sorted.sort_by(|left, right| left.direction_id.cmp(&right.direction_id));
        let mut ids = BTreeSet::new();
        let mut output = Vec::with_capacity(sorted.len());
        for direction in &sorted {
            if !ids.insert(direction.direction_id.as_str()) {
                return Err(EllipsoidGeometryError::DuplicateDirectionId);
            }
            output.push(self.support(direction)?);
        }
        Ok(output)
    }
}

fn validate_direction(
    direction: &LinearDirection,
    dimension: usize,
) -> Result<(), EllipsoidGeometryError> {
    if direction.direction_id.trim().is_empty() {
        return Err(EllipsoidGeometryError::EmptyDirectionId);
    }
    if direction.coefficients.len() != dimension {
        return Err(EllipsoidGeometryError::InvalidDirectionDimension);
    }
    if direction.coefficients.iter().any(|value| !value.is_finite()) {
        return Err(EllipsoidGeometryError::NonFiniteDirection);
    }
    Ok(())
}

fn support_interval(
    geometry: &EllipsoidGeometry,
    coefficients: &[f64],
) -> Result<DirectionalSupportInterval, EllipsoidGeometryError> {
    let dimension = coefficients.len();
    let mut center_projection = 0.0;
    for index in 0..dimension {
        center_projection += coefficients[index] * geometry.center[index];
    }
    if !center_projection.is_finite() {
        return Err(EllipsoidGeometryError::NumericFailure);
    }

    // For B = L Lᵀ, cᵀ B c = ||Lᵀ c||². Accumulate with hypot to
    // avoid avoidable overflow/underflow from a naive sum of squares.
    let mut norm = 0.0_f64;
    for col in 0..dimension {
        let mut projected = 0.0;
        for row in col..dimension {
            projected += geometry.lower_shape_factor[row][col] * coefficients[row];
        }
        if !projected.is_finite() {
            return Err(EllipsoidGeometryError::NumericFailure);
        }
        norm = norm.hypot(projected);
    }

    let support_radius = geometry.radius * norm;
    let lower_bound = center_projection - support_radius;
    let upper_bound = center_projection + support_radius;
    if !support_radius.is_finite()
        || !lower_bound.is_finite()
        || !upper_bound.is_finite()
        || lower_bound > upper_bound
    {
        return Err(EllipsoidGeometryError::NumericFailure);
    }

    Ok(DirectionalSupportInterval {
        center_projection,
        support_radius,
        lower_bound,
        upper_bound,
    })
}

/// Recompute every recorded direction in canonical order and compare all fields.
/// This verifies geometry only; it does not authenticate the ellipsoid source.
pub fn verify_ellipsoid_support_replay(
    geometry: &EllipsoidGeometry,
    directions: &[LinearDirection],
    recorded: &[DirectionalSupportResult],
) -> Result<(), EllipsoidGeometryError> {
    let expected = geometry.support_many(directions)?;
    if expected == recorded {
        Ok(())
    } else {
        Err(EllipsoidGeometryError::ReplayMismatch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_dimensional() -> EllipsoidGeometry {
        EllipsoidGeometry {
            coordinate_ids: vec!["utility".to_owned()],
            center: vec![2.0],
            lower_shape_factor: vec![vec![3.0]],
            radius: 2.0,
        }
    }

    fn two_dimensional() -> EllipsoidGeometry {
        EllipsoidGeometry {
            coordinate_ids: vec!["latency".to_owned(), "utility".to_owned()],
            center: vec![1.0, -1.0],
            // B = L Lᵀ = [[4, 2], [2, 10]]
            lower_shape_factor: vec![vec![2.0, 0.0], vec![1.0, 3.0]],
            radius: 0.5,
        }
    }

    fn direction(id: &str, coefficients: &[f64]) -> LinearDirection {
        LinearDirection {
            direction_id: id.to_owned(),
            coefficients: coefficients.to_vec(),
        }
    }

    #[test]
    fn one_dimensional_support_matches_analytic_interval() {
        let result = one_dimensional()
            .support(&direction("four-times-utility", &[4.0]))
            .unwrap();
        assert_eq!(result.interval.center_projection, 8.0);
        assert_eq!(result.interval.support_radius, 24.0);
        assert_eq!(result.interval.lower_bound, -16.0);
        assert_eq!(result.interval.upper_bound, 32.0);
    }

    #[test]
    fn shape_matrix_constructor_recovers_correlated_cholesky_factor() {
        let geometry = EllipsoidGeometry::from_shape_matrix(
            vec!["latency".to_owned(), "utility".to_owned()],
            vec![1.0, -1.0],
            &[vec![4.0, 2.0], vec![2.0, 10.0]],
            0.5,
        )
        .unwrap();
        assert_eq!(geometry.lower_shape_factor, vec![vec![2.0, 0.0], vec![1.0, 3.0]]);
        let result = geometry.support(&direction("correlated", &[1.0, 2.0])).unwrap();
        assert!((result.interval.support_radius - 0.5 * 52.0_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn shape_matrix_constructor_rejects_asymmetric_singular_and_indefinite_inputs() {
        let ids = vec!["x".to_owned(), "y".to_owned()];
        assert_eq!(
            EllipsoidGeometry::from_shape_matrix(
                ids.clone(),
                vec![0.0, 0.0],
                &[vec![2.0, 0.1], vec![0.2, 1.0]],
                1.0,
            ),
            Err(EllipsoidGeometryError::ShapeMatrixNotSymmetric)
        );
        assert_eq!(
            EllipsoidGeometry::from_shape_matrix(
                ids.clone(),
                vec![0.0, 0.0],
                &[vec![1.0, 1.0], vec![1.0, 1.0]],
                1.0,
            ),
            Err(EllipsoidGeometryError::ShapeMatrixNotPositiveDefinite)
        );
        assert_eq!(
            EllipsoidGeometry::from_shape_matrix(
                ids,
                vec![0.0, 0.0],
                &[vec![1.0, 2.0], vec![2.0, 1.0]],
                1.0,
            ),
            Err(EllipsoidGeometryError::ShapeMatrixNotPositiveDefinite)
        );
    }

    #[test]
    fn correlated_two_dimensional_support_uses_full_shape_matrix() {
        let result = two_dimensional()
            .support(&direction("latency-plus-two-utility", &[1.0, 2.0]))
            .unwrap();
        assert_eq!(result.interval.center_projection, -1.0);
        let expected_radius = 0.5 * 52.0_f64.sqrt();
        assert!((result.interval.support_radius - expected_radius).abs() < 1.0e-12);
        assert!((result.interval.lower_bound - (-1.0 - expected_radius)).abs() < 1.0e-12);
        assert!((result.interval.upper_bound - (-1.0 + expected_radius)).abs() < 1.0e-12);
    }

    #[test]
    fn multiple_direction_supports_are_canonical_and_replayable() {
        let geometry = two_dimensional();
        let directions = vec![
            direction("z", &[0.0, 1.0]),
            direction("a", &[1.0, 0.0]),
            direction("sum", &[1.0, 1.0]),
        ];
        let result = geometry.support_many(&directions).unwrap();
        assert_eq!(
            result.iter().map(|item| item.direction_id.as_str()).collect::<Vec<_>>(),
            vec!["a", "sum", "z"]
        );
        assert_eq!(verify_ellipsoid_support_replay(&geometry, &directions, &result), Ok(()));

        let mut changed = result.clone();
        changed[0].interval.upper_bound += 0.01;
        assert_eq!(
            verify_ellipsoid_support_replay(&geometry, &directions, &changed),
            Err(EllipsoidGeometryError::ReplayMismatch)
        );
    }

    #[test]
    fn support_is_homogeneous_for_positive_and_negative_directions() {
        let geometry = two_dimensional();
        let base = geometry.support(&direction("base", &[1.0, -0.5])).unwrap().interval;
        let positive = geometry.support(&direction("positive", &[3.0, -1.5])).unwrap().interval;
        let negative = geometry.support(&direction("negative", &[-1.0, 0.5])).unwrap().interval;
        assert!((positive.lower_bound - 3.0 * base.lower_bound).abs() < 1.0e-12);
        assert!((positive.upper_bound - 3.0 * base.upper_bound).abs() < 1.0e-12);
        assert!((negative.lower_bound + base.upper_bound).abs() < 1.0e-12);
        assert!((negative.upper_bound + base.lower_bound).abs() < 1.0e-12);
    }

    #[test]
    fn zero_direction_returns_the_singleton_zero_projection() {
        let result = two_dimensional().support(&direction("zero", &[0.0, 0.0])).unwrap();
        assert_eq!(result.interval.center_projection, 0.0);
        assert_eq!(result.interval.support_radius, 0.0);
        assert_eq!(result.interval.lower_bound, 0.0);
        assert_eq!(result.interval.upper_bound, 0.0);
    }

    #[test]
    fn rejects_non_spd_or_malformed_geometry() {
        let mut geometry = two_dimensional();
        geometry.lower_shape_factor[0][0] = 0.0;
        assert_eq!(geometry.validate(), Err(EllipsoidGeometryError::ShapeFactorNotPositiveDefinite));

        let mut geometry = two_dimensional();
        geometry.lower_shape_factor[0][1] = 0.1;
        assert_eq!(geometry.validate(), Err(EllipsoidGeometryError::ShapeFactorNotLowerTriangular));

        let mut geometry = two_dimensional();
        geometry.lower_shape_factor[1].pop();
        assert_eq!(geometry.validate(), Err(EllipsoidGeometryError::ShapeRowDimensionMismatch));

        let mut geometry = two_dimensional();
        geometry.radius = f64::NAN;
        assert_eq!(geometry.validate(), Err(EllipsoidGeometryError::InvalidRadius));

        let mut geometry = two_dimensional();
        geometry.coordinate_ids[1] = geometry.coordinate_ids[0].clone();
        assert_eq!(geometry.validate(), Err(EllipsoidGeometryError::DuplicateCoordinateId));
    }

    #[test]
    fn rejects_bad_directions_and_duplicate_direction_ids() {
        let geometry = two_dimensional();
        assert_eq!(
            geometry.support(&direction("wrong-dimension", &[1.0])),
            Err(EllipsoidGeometryError::InvalidDirectionDimension)
        );
        assert_eq!(
            geometry.support(&direction("nonfinite", &[f64::NAN, 0.0])),
            Err(EllipsoidGeometryError::NonFiniteDirection)
        );
        assert_eq!(
            geometry.support_many(&[]),
            Err(EllipsoidGeometryError::EmptyDirectionSet)
        );
        assert_eq!(
            geometry.support_many(&[direction("same", &[1.0, 0.0]), direction("same", &[0.0, 1.0])]),
            Err(EllipsoidGeometryError::DuplicateDirectionId)
        );
    }

    #[test]
    fn deterministic_generated_support_matches_direct_matrix_formula() {
        let geometry = EllipsoidGeometry {
            coordinate_ids: vec!["x".into(), "y".into(), "z".into()],
            center: vec![0.2, -0.5, 1.0],
            lower_shape_factor: vec![
                vec![1.2, 0.0, 0.0],
                vec![0.1, 0.8, 0.0],
                vec![-0.2, 0.3, 1.5],
            ],
            radius: 1.7,
        };
        let mut state = 0x5eed_311e_u64;
        for index in 0..512 {
            let mut coefficients = Vec::with_capacity(3);
            for _ in 0..3 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                coefficients.push(2.0 * ((state >> 11) as f64 / ((1_u64 << 53) as f64)) - 1.0);
            }
            let result = geometry.support(&direction(&format!("direction-{index}"), &coefficients)).unwrap();
            let projected = (0..3)
                .map(|col| {
                    (col..3)
                        .map(|row| geometry.lower_shape_factor[row][col] * coefficients[row])
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            let expected_norm = projected.iter().map(|value| value * value).sum::<f64>().sqrt();
            let expected_center = coefficients.iter().zip(&geometry.center).map(|(a,b)| a*b).sum::<f64>();
            assert!((result.interval.support_radius - geometry.radius * expected_norm).abs() < 1.0e-12);
            assert!((result.interval.center_projection - expected_center).abs() < 1.0e-12);
        }
    }
}
