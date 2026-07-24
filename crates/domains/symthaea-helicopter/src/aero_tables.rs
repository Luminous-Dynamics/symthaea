// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Validated aerodynamic coefficient lookup tables.
//!
//! Reduced-order models often hide hard-coded coefficient curves inside
//! equations. This module makes table axes, validity bounds, interpolation, and
//! extrapolation policy explicit so calibration evidence can bind the exact
//! aerodynamic data used by a scenario.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtrapolationPolicy {
    Reject,
    Clamp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LookupDisposition {
    Interpolated,
    Exact,
    Clamped,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CoefficientLookup {
    pub value: f64,
    pub disposition: LookupDisposition,
    pub x_used: f64,
    pub y_used: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AeroTableError {
    TooFewPoints,
    DimensionMismatch,
    NonFiniteValue,
    NonMonotonicAxis,
    OutOfBounds,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AeroCoefficientTable1D {
    pub axis_name: String,
    pub coefficient_name: String,
    axis: Vec<f64>,
    values: Vec<f64>,
    policy: ExtrapolationPolicy,
}

impl AeroCoefficientTable1D {
    pub fn new(
        axis_name: impl Into<String>,
        coefficient_name: impl Into<String>,
        axis: Vec<f64>,
        values: Vec<f64>,
        policy: ExtrapolationPolicy,
    ) -> Result<Self, AeroTableError> {
        validate_axis(&axis)?;
        if axis.len() != values.len() {
            return Err(AeroTableError::DimensionMismatch);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(AeroTableError::NonFiniteValue);
        }
        Ok(Self {
            axis_name: axis_name.into(),
            coefficient_name: coefficient_name.into(),
            axis,
            values,
            policy,
        })
    }

    pub fn bounds(&self) -> (f64, f64) {
        (self.axis[0], self.axis[self.axis.len() - 1])
    }

    pub fn lookup(&self, x: f64) -> Result<CoefficientLookup, AeroTableError> {
        if !x.is_finite() {
            return Err(AeroTableError::NonFiniteValue);
        }
        let (x_used, clamped) = apply_bounds(x, self.bounds(), self.policy)?;
        if let Some(index) = self.axis.iter().position(|candidate| *candidate == x_used) {
            return Ok(CoefficientLookup {
                value: self.values[index],
                disposition: if clamped {
                    LookupDisposition::Clamped
                } else {
                    LookupDisposition::Exact
                },
                x_used,
                y_used: None,
            });
        }
        let upper = upper_index(&self.axis, x_used);
        let lower = upper - 1;
        let fraction = (x_used - self.axis[lower]) / (self.axis[upper] - self.axis[lower]);
        let value = lerp(self.values[lower], self.values[upper], fraction);
        Ok(CoefficientLookup {
            value,
            disposition: if clamped {
                LookupDisposition::Clamped
            } else {
                LookupDisposition::Interpolated
            },
            x_used,
            y_used: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AeroCoefficientSurface2D {
    pub x_axis_name: String,
    pub y_axis_name: String,
    pub coefficient_name: String,
    x_axis: Vec<f64>,
    y_axis: Vec<f64>,
    /// Row-major values indexed as `y_index * x_count + x_index`.
    values: Vec<f64>,
    policy: ExtrapolationPolicy,
}

impl AeroCoefficientSurface2D {
    pub fn new(
        x_axis_name: impl Into<String>,
        y_axis_name: impl Into<String>,
        coefficient_name: impl Into<String>,
        x_axis: Vec<f64>,
        y_axis: Vec<f64>,
        values: Vec<f64>,
        policy: ExtrapolationPolicy,
    ) -> Result<Self, AeroTableError> {
        validate_axis(&x_axis)?;
        validate_axis(&y_axis)?;
        if values.len() != x_axis.len() * y_axis.len() {
            return Err(AeroTableError::DimensionMismatch);
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(AeroTableError::NonFiniteValue);
        }
        Ok(Self {
            x_axis_name: x_axis_name.into(),
            y_axis_name: y_axis_name.into(),
            coefficient_name: coefficient_name.into(),
            x_axis,
            y_axis,
            values,
            policy,
        })
    }

    pub fn bounds(&self) -> ((f64, f64), (f64, f64)) {
        (
            (self.x_axis[0], self.x_axis[self.x_axis.len() - 1]),
            (self.y_axis[0], self.y_axis[self.y_axis.len() - 1]),
        )
    }

    pub fn lookup(&self, x: f64, y: f64) -> Result<CoefficientLookup, AeroTableError> {
        if !x.is_finite() || !y.is_finite() {
            return Err(AeroTableError::NonFiniteValue);
        }
        let (x_bounds, y_bounds) = self.bounds();
        let (x_used, x_clamped) = apply_bounds(x, x_bounds, self.policy)?;
        let (y_used, y_clamped) = apply_bounds(y, y_bounds, self.policy)?;

        let (x0, x1, tx, x_exact) = bracket(&self.x_axis, x_used);
        let (y0, y1, ty, y_exact) = bracket(&self.y_axis, y_used);
        let q00 = self.value_at(x0, y0);
        let q10 = self.value_at(x1, y0);
        let q01 = self.value_at(x0, y1);
        let q11 = self.value_at(x1, y1);
        let lower = lerp(q00, q10, tx);
        let upper = lerp(q01, q11, tx);
        let value = lerp(lower, upper, ty);
        let disposition = if x_clamped || y_clamped {
            LookupDisposition::Clamped
        } else if x_exact && y_exact {
            LookupDisposition::Exact
        } else {
            LookupDisposition::Interpolated
        };

        Ok(CoefficientLookup {
            value,
            disposition,
            x_used,
            y_used: Some(y_used),
        })
    }

    fn value_at(&self, x_index: usize, y_index: usize) -> f64 {
        self.values[y_index * self.x_axis.len() + x_index]
    }
}

fn validate_axis(axis: &[f64]) -> Result<(), AeroTableError> {
    if axis.len() < 2 {
        return Err(AeroTableError::TooFewPoints);
    }
    if axis.iter().any(|value| !value.is_finite()) {
        return Err(AeroTableError::NonFiniteValue);
    }
    if axis.windows(2).any(|window| window[0] >= window[1]) {
        return Err(AeroTableError::NonMonotonicAxis);
    }
    Ok(())
}

fn apply_bounds(
    value: f64,
    bounds: (f64, f64),
    policy: ExtrapolationPolicy,
) -> Result<(f64, bool), AeroTableError> {
    if (bounds.0..=bounds.1).contains(&value) {
        return Ok((value, false));
    }
    match policy {
        ExtrapolationPolicy::Reject => Err(AeroTableError::OutOfBounds),
        ExtrapolationPolicy::Clamp => Ok((value.clamp(bounds.0, bounds.1), true)),
    }
}

fn upper_index(axis: &[f64], value: f64) -> usize {
    axis.partition_point(|candidate| *candidate < value)
        .clamp(1, axis.len() - 1)
}

fn bracket(axis: &[f64], value: f64) -> (usize, usize, f64, bool) {
    if let Some(index) = axis.iter().position(|candidate| *candidate == value) {
        return (index, index, 0.0, true);
    }
    let upper = upper_index(axis, value);
    let lower = upper - 1;
    let fraction = (value - axis[lower]) / (axis[upper] - axis[lower]);
    (lower, upper, fraction, false)
}

fn lerp(a: f64, b: f64, fraction: f64) -> f64 {
    a + (b - a) * fraction
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_dimensional_lookup_interpolates() {
        let table = AeroCoefficientTable1D::new(
            "advance-ratio",
            "lift-factor",
            vec![0.0, 0.5, 1.0],
            vec![1.0, 1.2, 0.8],
            ExtrapolationPolicy::Reject,
        )
        .unwrap();
        let lookup = table.lookup(0.25).unwrap();
        assert_eq!(lookup.disposition, LookupDisposition::Interpolated);
        assert!((lookup.value - 1.1).abs() < 1.0e-12);
    }

    #[test]
    fn bilinear_surface_interpolates_center() {
        let table = AeroCoefficientSurface2D::new(
            "collective",
            "inflow",
            "thrust-coefficient",
            vec![0.0, 1.0],
            vec![0.0, 1.0],
            vec![0.0, 1.0, 2.0, 3.0],
            ExtrapolationPolicy::Reject,
        )
        .unwrap();
        let lookup = table.lookup(0.5, 0.5).unwrap();
        assert!((lookup.value - 1.5).abs() < 1.0e-12);
    }

    #[test]
    fn clamping_is_visible_in_evidence() {
        let table = AeroCoefficientTable1D::new(
            "angle",
            "coefficient",
            vec![-1.0, 1.0],
            vec![-2.0, 2.0],
            ExtrapolationPolicy::Clamp,
        )
        .unwrap();
        let lookup = table.lookup(3.0).unwrap();
        assert_eq!(lookup.disposition, LookupDisposition::Clamped);
        assert_eq!(lookup.x_used, 1.0);
        assert_eq!(lookup.value, 2.0);
    }

    #[test]
    fn non_monotonic_axes_are_rejected() {
        assert_eq!(
            AeroCoefficientTable1D::new(
                "x",
                "c",
                vec![0.0, 0.0],
                vec![1.0, 2.0],
                ExtrapolationPolicy::Reject,
            ),
            Err(AeroTableError::NonMonotonicAxis)
        );
    }
}
