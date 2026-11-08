// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Model-conditional order-statistic calibration for conformal 1+1D FRW.
//!
//! For a(eta)=exp(kappa*eta^2/2) on a diamond of conformal extent T, use the
//! dimensionless parameter q=kappa*T^2. In normalized null coordinates
//! (u,v) in [0,1]^2, the volume weight is exp(q*((u+v-1)/2)^2).
//!
//! The QMC reference below is a deterministic finite quadrature. It is not an
//! exact expectation, a sampling confidence interval, or a continuum-limit
//! result. It assumes volume-faithful sampling over the declared diamond.
//! Order-only data do not identify kappa and T independently.

use std::error::Error;
use std::fmt;

/// Minimum QMC points required to avoid meaningless small-sample calibration.
pub const MIN_CURVED_ORDER_QMC_POINTS: usize = 512;

/// Error in curved order-statistic reference or inversion.
#[derive(Clone, Debug, PartialEq)]
pub enum CurvedOrderStatisticsError {
    InvalidDimensionlessCurvature,
    InvalidQuadratureCount,
    InvalidObservedFraction,
    InvalidSearchRange,
    NonMonotoneCalibration,
    FractionOutsideCalibration,
    NumericalFailure,
}

impl fmt::Display for CurvedOrderStatisticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensionlessCurvature => {
                write!(f, "dimensionless curvature q must be finite and non-negative")
            }
            Self::InvalidQuadratureCount => write!(
                f,
                "QMC quadrature requires at least {MIN_CURVED_ORDER_QMC_POINTS} points"
            ),
            Self::InvalidObservedFraction => {
                write!(f, "observed order fraction must be finite and in [0,1]")
            }
            Self::InvalidSearchRange => write!(f, "dimensionless-curvature search range is invalid"),
            Self::NonMonotoneCalibration => {
                write!(f, "finite-QMC ordering-fraction calibration is not monotone")
            }
            Self::FractionOutsideCalibration => {
                write!(f, "observed order fraction lies outside the calibrated range")
            }
            Self::NumericalFailure => write!(f, "curved order-statistic calculation failed"),
        }
    }
}

impl Error for CurvedOrderStatisticsError {}

/// Deterministic finite-QMC reference for a comparable-pair fraction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvedOrderFractionReference {
    pub dimensionless_curvature: f64,
    pub comparable_pair_fraction: f64,
    pub quadrature_points: usize,
}

/// Finite-QMC inverse result for q = kappa*T^2.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DimensionlessCurvatureEstimate {
    pub dimensionless_curvature: f64,
    pub observed_ordering_fraction: f64,
    pub reconstructed_ordering_fraction: f64,
    pub quadrature_points: usize,
    pub iterations: usize,
}

/// Compute a deterministic QMC estimate of the expected comparable-pair
/// fraction under the normalized volume measure.
///
/// The weighted dominance count uses a Fenwick tree and O(P log P) work.
/// The independent O(P^2) pair-loop reference belongs in tests/examples and
/// must remain separate from this implementation.
pub fn qmc_expected_comparable_pair_fraction(
    dimensionless_curvature: f64,
    quadrature_points: usize,
) -> Result<CurvedOrderFractionReference, CurvedOrderStatisticsError> {
    validate_qmc_inputs(dimensionless_curvature, quadrature_points)?;

    #[derive(Clone, Copy)]
    struct Point {
        u: f64,
        v: f64,
        weight: f64,
    }

    let mut points = Vec::with_capacity(quadrature_points);
    let mut total_weight = 0.0;
    let mut squared_weight_sum = 0.0;
    for index in 1..=quadrature_points {
        let u = halton(index, 2);
        let v = halton(index, 3);
        let normalized_eta = 0.5 * (u + v - 1.0);
        let weight = (dimensionless_curvature * normalized_eta * normalized_eta).exp();
        if !weight.is_finite() || weight <= 0.0 {
            return Err(CurvedOrderStatisticsError::NumericalFailure);
        }
        total_weight += weight;
        squared_weight_sum += weight * weight;
        points.push(Point { u, v, weight });
    }

    let mut sorted_v: Vec<f64> = points.iter().map(|point| point.v).collect();
    sorted_v.sort_by(f64::total_cmp);
    sorted_v.dedup_by(|left, right| *left == *right);
    points.sort_by(|left, right| {
        left.u
            .total_cmp(&right.u)
            .then_with(|| left.v.total_cmp(&right.v))
    });

    let mut tree = FenwickSum::new(sorted_v.len());
    let mut comparable_unordered_weight = 0.0;
    let mut group_start = 0;
    while group_start < points.len() {
        let mut group_end = group_start + 1;
        while group_end < points.len() && points[group_end].u == points[group_start].u {
            group_end += 1;
        }

        // Query before inserting equal-u points: causal order is strict.
        for point in &points[group_start..group_end] {
            let prefix_count = sorted_v.partition_point(|value| *value < point.v);
            comparable_unordered_weight += point.weight * tree.prefix_sum(prefix_count);
        }
        for point in &points[group_start..group_end] {
            let rank = sorted_v
                .binary_search_by(|value| value.total_cmp(&point.v))
                .map_err(|_| CurvedOrderStatisticsError::NumericalFailure)?;
            tree.add(rank, point.weight);
        }
        group_start = group_end;
    }

    let denominator = total_weight * total_weight - squared_weight_sum;
    let fraction = 2.0 * comparable_unordered_weight / denominator;
    if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
        return Err(CurvedOrderStatisticsError::NumericalFailure);
    }

    Ok(CurvedOrderFractionReference {
        dimensionless_curvature,
        comparable_pair_fraction: fraction,
        quadrature_points,
    })
}

/// Estimate q = kappa*T^2 from an empirical comparable-pair fraction.
///
/// This is conditional on the conformal-FRW metric, Alexandrov diamond and
/// volume-faithful sampling assumptions. It returns no finite-sample interval;
/// that uncertainty must be estimated using independent causal-set
/// sprinklings and must be kept separate from QMC-reference convergence.
pub fn estimate_dimensionless_curvature(
    observed_ordering_fraction: f64,
    maximum_dimensionless_curvature: f64,
    quadrature_points: usize,
    iterations: usize,
) -> Result<DimensionlessCurvatureEstimate, CurvedOrderStatisticsError> {
    if !observed_ordering_fraction.is_finite()
        || !(0.0..=1.0).contains(&observed_ordering_fraction)
    {
        return Err(CurvedOrderStatisticsError::InvalidObservedFraction);
    }
    if !maximum_dimensionless_curvature.is_finite()
        || maximum_dimensionless_curvature <= 0.0
        || maximum_dimensionless_curvature * 0.25 > 600.0
        || !(8..=96).contains(&iterations)
    {
        return Err(CurvedOrderStatisticsError::InvalidSearchRange);
    }
    validate_qmc_inputs(0.0, quadrature_points)?;

    let mut previous = qmc_expected_comparable_pair_fraction(0.0, quadrature_points)?
        .comparable_pair_fraction;
    for step in 1..=8 {
        let q = maximum_dimensionless_curvature * step as f64 / 8.0;
        let current = qmc_expected_comparable_pair_fraction(q, quadrature_points)?
            .comparable_pair_fraction;
        if current + 1.0e-12 < previous {
            return Err(CurvedOrderStatisticsError::NonMonotoneCalibration);
        }
        previous = current;
    }

    let lower_fraction = qmc_expected_comparable_pair_fraction(0.0, quadrature_points)?
        .comparable_pair_fraction;
    let upper_fraction = qmc_expected_comparable_pair_fraction(
        maximum_dimensionless_curvature,
        quadrature_points,
    )?
    .comparable_pair_fraction;
    if observed_ordering_fraction < lower_fraction - 1.0e-12
        || observed_ordering_fraction > upper_fraction + 1.0e-12
    {
        return Err(CurvedOrderStatisticsError::FractionOutsideCalibration);
    }

    let mut low = 0.0;
    let mut high = maximum_dimensionless_curvature;
    for _ in 0..iterations {
        let middle = 0.5 * (low + high);
        let reference = qmc_expected_comparable_pair_fraction(middle, quadrature_points)?
            .comparable_pair_fraction;
        if reference < observed_ordering_fraction {
            low = middle;
        } else {
            high = middle;
        }
    }

    let estimate = 0.5 * (low + high);
    let reconstructed = qmc_expected_comparable_pair_fraction(estimate, quadrature_points)?
        .comparable_pair_fraction;
    Ok(DimensionlessCurvatureEstimate {
        dimensionless_curvature: estimate,
        observed_ordering_fraction,
        reconstructed_ordering_fraction: reconstructed,
        quadrature_points,
        iterations,
    })
}

fn validate_qmc_inputs(
    dimensionless_curvature: f64,
    quadrature_points: usize,
) -> Result<(), CurvedOrderStatisticsError> {
    if quadrature_points < MIN_CURVED_ORDER_QMC_POINTS {
        return Err(CurvedOrderStatisticsError::InvalidQuadratureCount);
    }
    if !dimensionless_curvature.is_finite() || dimensionless_curvature < 0.0 {
        return Err(CurvedOrderStatisticsError::InvalidDimensionlessCurvature);
    }
    if dimensionless_curvature * 0.25 > 600.0 {
        return Err(CurvedOrderStatisticsError::InvalidSearchRange);
    }
    Ok(())
}

fn halton(mut index: usize, base: usize) -> f64 {
    let mut factor = 1.0;
    let mut value = 0.0;
    while index > 0 {
        factor /= base as f64;
        value += factor * (index % base) as f64;
        index /= base;
    }
    value
}

struct FenwickSum {
    tree: Vec<f64>,
}

impl FenwickSum {
    fn new(length: usize) -> Self {
        Self {
            tree: vec![0.0; length + 1],
        }
    }

    fn add(&mut self, zero_based_index: usize, value: f64) {
        let mut index = zero_based_index + 1;
        while index < self.tree.len() {
            self.tree[index] += value;
            index += index & (!index + 1);
        }
    }

    /// Sum the first `count` compressed coordinates.
    fn prefix_sum(&self, count: usize) -> f64 {
        let mut index = count;
        let mut total = 0.0;
        while index > 0 {
            total += self.tree[index];
            index &= index - 1;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair_loop_reference(q: f64, points: usize) -> f64 {
        let mut coordinates = Vec::with_capacity(points);
        let mut weights = Vec::with_capacity(points);
        let mut sum = 0.0;
        let mut squared_sum = 0.0;
        for index in 1..=points {
            let u = halton(index, 2);
            let v = halton(index, 3);
            let eta = 0.5 * (u + v - 1.0);
            let weight = (q * eta * eta).exp();
            coordinates.push((u, v));
            weights.push(weight);
            sum += weight;
            squared_sum += weight * weight;
        }
        let mut related = 0.0;
        for left in 0..points {
            for right in (left + 1)..points {
                let (u0, v0) = coordinates[left];
                let (u1, v1) = coordinates[right];
                if (u0 < u1 && v0 < v1) || (u1 < u0 && v1 < v0) {
                    related += weights[left] * weights[right];
                }
            }
        }
        2.0 * related / (sum * sum - squared_sum)
    }

    #[test]
    fn fenwick_qmc_matches_independent_pair_loop_reference() {
        for q in [0.0, 0.5, 2.0, 4.0] {
            let fast = qmc_expected_comparable_pair_fraction(q, 512).unwrap();
            let reference = pair_loop_reference(q, 512);
            assert!(
                (fast.comparable_pair_fraction - reference).abs() < 1.0e-12,
                "q={q}, fast={}, reference={reference}",
                fast.comparable_pair_fraction
            );
        }
    }

    #[test]
    fn flat_limit_recovers_half_ordering_fraction_with_finite_qmc_error() {
        let result = qmc_expected_comparable_pair_fraction(0.0, 8_192).unwrap();
        assert!((result.comparable_pair_fraction - 0.5).abs() < 0.01);
    }

    #[test]
    fn scale_rescaling_preserves_dimensionless_curvature_and_reference() {
        let first_kappa = 0.8;
        let first_extent: f64 = 2.0;
        let scale: f64 = 4.0;
        let second_kappa = first_kappa * scale * scale;
        let second_extent = first_extent / scale;
        let q1 = first_kappa * first_extent * first_extent;
        let q2 = second_kappa * second_extent * second_extent;
        assert!((q1 - q2).abs() < 1.0e-12);

        let first = qmc_expected_comparable_pair_fraction(q1, 4_096).unwrap();
        let second = qmc_expected_comparable_pair_fraction(q2, 4_096).unwrap();
        assert!((first.comparable_pair_fraction - second.comparable_pair_fraction).abs() < 1.0e-12);
    }

    #[test]
    fn dimensionless_curvature_inverse_recovers_q_from_its_reference() {
        let reference = qmc_expected_comparable_pair_fraction(2.0, 4_096).unwrap();
        let estimate =
            estimate_dimensionless_curvature(reference.comparable_pair_fraction, 4.0, 4_096, 40)
                .unwrap();
        assert!((estimate.dimensionless_curvature - 2.0).abs() < 1.0e-8);
        assert!(
            (estimate.reconstructed_ordering_fraction - reference.comparable_pair_fraction).abs()
                < 1.0e-12
        );
    }

    #[test]
    fn flat_reference_and_curved_reference_are_not_conflated() {
        let flat = qmc_expected_comparable_pair_fraction(0.0, 4_096).unwrap();
        let curved = qmc_expected_comparable_pair_fraction(4.0, 4_096).unwrap();
        assert!((flat.comparable_pair_fraction - 0.5).abs() < 0.01);
        assert!(curved.comparable_pair_fraction > flat.comparable_pair_fraction);
    }

    #[test]
    fn invalid_ranges_and_out_of_calibration_fraction_fail_closed() {
        assert_eq!(
            qmc_expected_comparable_pair_fraction(1.0, 128),
            Err(CurvedOrderStatisticsError::InvalidQuadratureCount)
        );
        assert_eq!(
            qmc_expected_comparable_pair_fraction(-1.0, 512),
            Err(CurvedOrderStatisticsError::InvalidDimensionlessCurvature)
        );
        assert_eq!(
            estimate_dimensionless_curvature(0.9, 2.0, 512, 32),
            Err(CurvedOrderStatisticsError::FractionOutsideCalibration)
        );
    }
}
