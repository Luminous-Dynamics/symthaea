// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Continuous temporal roles in the Fourier/phasor domain.
//!
//! This module implements the unitary core needed for fractional-power temporal
//! binding without yet committing the research line to a particular real-space
//! FFT representation. A temporal axis is a deterministic set of angular
//! frequencies `omega_k`. The role for real-valued time `t` is
//!
//! `T(t)_k = exp(i * omega_k * t)`.
//!
//! Therefore, up to floating-point error,
//!
//! `T(a) * T(b) = T(a + b)` and `T(t)^-1 = T(-t)`.
//!
//! Validity regions can be encoded analytically as
//!
//! `I[a,b)_k = integral_a^b exp(i * omega_k * tau) d tau`.
//!
//! That interval primitive is the important distinction from an ad-hoc lag key:
//! historical state is valid over time spans, not merely at mutation instants.

use crate::continuous_hv::UnitaryRole;
use std::f64::consts::PI;
use std::fmt;

const OMEGA_EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq)]
pub enum TemporalAlgebraError {
    ZeroDimension,
    NonFiniteFrequency { index: usize, value: f64 },
    NonFiniteTime(f64),
    InvalidInterval { start: f64, end: f64 },
    DimensionMismatch { expected: usize, actual: usize },
}

impl fmt::Display for TemporalAlgebraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "temporal axis dimension must be non-zero"),
            Self::NonFiniteFrequency { index, value } => write!(
                f,
                "temporal frequency at index {index} must be finite, got {value}"
            ),
            Self::NonFiniteTime(value) => {
                write!(f, "temporal coordinate must be finite, got {value}")
            }
            Self::InvalidInterval { start, end } => write!(
                f,
                "temporal interval must satisfy finite start <= end, got [{start}, {end})"
            ),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "temporal algebra dimension mismatch: expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for TemporalAlgebraError {}

/// Deterministic continuous-time axis.
///
/// Frequencies are angular phase slopes in radians per unit of represented time.
/// `new()` samples them uniformly from `[-pi, pi)`, matching the common unitary
/// fractional-power construction in the Fourier domain.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalAxis {
    frequencies: Vec<f64>,
}

impl TemporalAxis {
    pub fn new(dim: usize, seed: u64) -> Result<Self, TemporalAlgebraError> {
        if dim == 0 {
            return Err(TemporalAlgebraError::ZeroDimension);
        }
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        if state == 0 {
            state = 0xD1B5_4A32_D192_ED03;
        }
        let frequencies = (0..dim)
            .map(|_| {
                let unit = next_unit_f64(&mut state);
                (2.0 * unit - 1.0) * PI
            })
            .collect();
        Ok(Self { frequencies })
    }

    pub fn try_from_frequencies(
        frequencies: Vec<f64>,
    ) -> Result<Self, TemporalAlgebraError> {
        if frequencies.is_empty() {
            return Err(TemporalAlgebraError::ZeroDimension);
        }
        if let Some((index, value)) = frequencies
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(TemporalAlgebraError::NonFiniteFrequency { index, value });
        }
        Ok(Self { frequencies })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.frequencies.len()
    }

    #[inline]
    pub fn frequencies(&self) -> &[f64] {
        &self.frequencies
    }

    /// Continuous unitary temporal role `T(t)`.
    pub fn at(&self, time: f64) -> Result<TemporalPhasor, TemporalAlgebraError> {
        validate_time(time)?;
        let mut real = Vec::with_capacity(self.dim());
        let mut imag = Vec::with_capacity(self.dim());
        for &omega in &self.frequencies {
            let phase = omega * time;
            let (sin, cos) = phase.sin_cos();
            real.push(cos);
            imag.push(sin);
        }
        Ok(TemporalPhasor { real, imag })
    }

    /// Analytic encoding of the half-open temporal interval `[start, end)`.
    ///
    /// Per frequency,
    ///
    /// `integral exp(i*omega*t) dt = (exp(i*omega*end)-exp(i*omega*start))/(i*omega)`.
    ///
    /// The zero-frequency limit is exactly `end - start`.
    pub fn interval(
        &self,
        start: f64,
        end: f64,
    ) -> Result<TemporalInterval, TemporalAlgebraError> {
        if !start.is_finite() || !end.is_finite() || end < start {
            return Err(TemporalAlgebraError::InvalidInterval { start, end });
        }

        let mut real = Vec::with_capacity(self.dim());
        let mut imag = Vec::with_capacity(self.dim());
        for &omega in &self.frequencies {
            if omega.abs() <= OMEGA_EPSILON {
                real.push(end - start);
                imag.push(0.0);
            } else {
                let start_phase = omega * start;
                let end_phase = omega * end;
                real.push((end_phase.sin() - start_phase.sin()) / omega);
                imag.push((start_phase.cos() - end_phase.cos()) / omega);
            }
        }
        Ok(TemporalInterval { real, imag })
    }
}

/// Unit-modulus complex hypervector representing one point on the temporal axis.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalPhasor {
    real: Vec<f64>,
    imag: Vec<f64>,
}

impl TemporalPhasor {
    pub fn identity(dim: usize) -> Result<Self, TemporalAlgebraError> {
        if dim == 0 {
            return Err(TemporalAlgebraError::ZeroDimension);
        }
        Ok(Self {
            real: vec![1.0; dim],
            imag: vec![0.0; dim],
        })
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.real.len()
    }

    #[inline]
    pub fn real(&self) -> &[f64] {
        &self.real
    }

    #[inline]
    pub fn imag(&self) -> &[f64] {
        &self.imag
    }

    /// Complex elementwise binding.
    pub fn bind(&self, other: &Self) -> Result<Self, TemporalAlgebraError> {
        check_dim(self.dim(), other.dim())?;
        let mut real = Vec::with_capacity(self.dim());
        let mut imag = Vec::with_capacity(self.dim());
        for i in 0..self.dim() {
            real.push(self.real[i] * other.real[i] - self.imag[i] * other.imag[i]);
            imag.push(self.real[i] * other.imag[i] + self.imag[i] * other.real[i]);
        }
        Ok(Self { real, imag })
    }

    /// Exact algebraic inverse in the unitary phasor domain: complex conjugation.
    pub fn inverse(&self) -> Self {
        Self {
            real: self.real.clone(),
            imag: self.imag.iter().map(|value| -*value).collect(),
        }
    }

    /// Bind a real bipolar HDC role into the same Fourier-domain association.
    ///
    /// `-1` is simply a phase rotation by pi and `+1` is the identity, so this
    /// operation remains unitary and commutes with temporal phasor binding.
    pub fn bind_role(&self, role: &UnitaryRole) -> Result<Self, TemporalAlgebraError> {
        check_dim(self.dim(), role.dim())?;
        Ok(Self {
            real: self
                .real
                .iter()
                .zip(role.as_slice())
                .map(|(value, role)| *value * *role as f64)
                .collect(),
            imag: self
                .imag
                .iter()
                .zip(role.as_slice())
                .map(|(value, role)| *value * *role as f64)
                .collect(),
        })
    }

    /// Mean real inner product `Re(conj(self) * other)`.
    ///
    /// For phasors from the same axis this depends only on the time difference,
    /// providing the translation-invariant kernel used by fractional binding.
    pub fn similarity(&self, other: &Self) -> Result<f64, TemporalAlgebraError> {
        check_dim(self.dim(), other.dim())?;
        Ok((0..self.dim())
            .map(|i| self.real[i] * other.real[i] + self.imag[i] * other.imag[i])
            .sum::<f64>()
            / self.dim() as f64)
    }

    /// Maximum per-channel deviation from unit modulus.
    pub fn max_unit_modulus_error(&self) -> f64 {
        (0..self.dim())
            .map(|i| (self.real[i] * self.real[i] + self.imag[i] * self.imag[i] - 1.0).abs())
            .fold(0.0, f64::max)
    }

    /// Translate an interval by this phasor through complex binding.
    pub fn bind_interval(
        &self,
        interval: &TemporalInterval,
    ) -> Result<TemporalInterval, TemporalAlgebraError> {
        check_dim(self.dim(), interval.dim())?;
        let mut real = Vec::with_capacity(self.dim());
        let mut imag = Vec::with_capacity(self.dim());
        for i in 0..self.dim() {
            real.push(self.real[i] * interval.real[i] - self.imag[i] * interval.imag[i]);
            imag.push(self.real[i] * interval.imag[i] + self.imag[i] * interval.real[i]);
        }
        Ok(TemporalInterval { real, imag })
    }

    pub fn max_abs_difference(&self, other: &Self) -> Result<f64, TemporalAlgebraError> {
        check_dim(self.dim(), other.dim())?;
        Ok((0..self.dim())
            .flat_map(|i| {
                [
                    (self.real[i] - other.real[i]).abs(),
                    (self.imag[i] - other.imag[i]).abs(),
                ]
            })
            .fold(0.0, f64::max))
    }
}

/// Fourier-domain integral over a continuous time interval.
#[derive(Debug, Clone, PartialEq)]
pub struct TemporalInterval {
    real: Vec<f64>,
    imag: Vec<f64>,
}

impl TemporalInterval {
    #[inline]
    pub fn dim(&self) -> usize {
        self.real.len()
    }

    #[inline]
    pub fn real(&self) -> &[f64] {
        &self.real
    }

    #[inline]
    pub fn imag(&self) -> &[f64] {
        &self.imag
    }

    /// Query an interval with a point phasor.
    ///
    /// This is the mean real inner product `Re(conj(T(t)) * I[a,b))`, i.e. the
    /// time-kernel integrated over the validity region.
    pub fn score_at(&self, point: &TemporalPhasor) -> Result<f64, TemporalAlgebraError> {
        check_dim(self.dim(), point.dim())?;
        Ok((0..self.dim())
            .map(|i| point.real[i] * self.real[i] + point.imag[i] * self.imag[i])
            .sum::<f64>()
            / self.dim() as f64)
    }

    pub fn bind_role(&self, role: &UnitaryRole) -> Result<Self, TemporalAlgebraError> {
        check_dim(self.dim(), role.dim())?;
        Ok(Self {
            real: self
                .real
                .iter()
                .zip(role.as_slice())
                .map(|(value, role)| *value * *role as f64)
                .collect(),
            imag: self
                .imag
                .iter()
                .zip(role.as_slice())
                .map(|(value, role)| *value * *role as f64)
                .collect(),
        })
    }

    pub fn l2_norm(&self) -> f64 {
        self.real
            .iter()
            .zip(&self.imag)
            .map(|(real, imag)| real * real + imag * imag)
            .sum::<f64>()
            .sqrt()
    }

    pub fn max_abs_difference(&self, other: &Self) -> Result<f64, TemporalAlgebraError> {
        check_dim(self.dim(), other.dim())?;
        Ok((0..self.dim())
            .flat_map(|i| {
                [
                    (self.real[i] - other.real[i]).abs(),
                    (self.imag[i] - other.imag[i]).abs(),
                ]
            })
            .fold(0.0, f64::max))
    }
}

fn validate_time(time: f64) -> Result<(), TemporalAlgebraError> {
    if time.is_finite() {
        Ok(())
    } else {
        Err(TemporalAlgebraError::NonFiniteTime(time))
    }
}

fn check_dim(expected: usize, actual: usize) -> Result<(), TemporalAlgebraError> {
    if expected == actual {
        Ok(())
    } else {
        Err(TemporalAlgebraError::DimensionMismatch { expected, actual })
    }
}

fn next_unit_f64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let bits = *state >> 11;
    bits as f64 * (1.0 / (1_u64 << 53) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractional_time_binding_obeys_group_law() {
        let axis = TemporalAxis::new(512, 41).unwrap();
        let a = axis.at(-3.125).unwrap();
        let b = axis.at(7.75).unwrap();
        let composed = a.bind(&b).unwrap();
        let direct = axis.at(4.625).unwrap();
        assert!(composed.max_abs_difference(&direct).unwrap() < 2e-13);
    }

    #[test]
    fn inverse_returns_identity_and_unit_modulus_is_preserved() {
        let axis = TemporalAxis::new(512, 42).unwrap();
        let point = axis.at(123.456).unwrap();
        assert!(point.max_unit_modulus_error() < 3e-16);
        let recovered = point.bind(&point.inverse()).unwrap();
        let identity = TemporalPhasor::identity(axis.dim()).unwrap();
        assert!(recovered.max_abs_difference(&identity).unwrap() < 3e-16);
    }

    #[test]
    fn temporal_similarity_is_translation_invariant() {
        let axis = TemporalAxis::new(2048, 43).unwrap();
        let a = axis.at(2.0).unwrap();
        let b = axis.at(3.5).unwrap();
        let shifted_a = axis.at(102.0).unwrap();
        let shifted_b = axis.at(103.5).unwrap();
        let before = a.similarity(&b).unwrap();
        let after = shifted_a.similarity(&shifted_b).unwrap();
        assert!((before - after).abs() < 2e-14);
    }

    #[test]
    fn bipolar_role_binding_commutes_with_temporal_translation() {
        let axis = TemporalAxis::new(512, 44).unwrap();
        let role = UnitaryRole::new(axis.dim(), 45);
        let t = axis.at(1.25).unwrap();
        let dt = axis.at(4.5).unwrap();

        let left = t.bind_role(&role).unwrap().bind(&dt).unwrap();
        let right = t.bind(&dt).unwrap().bind_role(&role).unwrap();
        assert!(left.max_abs_difference(&right).unwrap() < 2e-15);
    }

    #[test]
    fn analytic_interval_translates_by_group_action() {
        let axis = TemporalAxis::new(512, 46).unwrap();
        let interval = axis.interval(-2.0, 3.25).unwrap();
        let shift = axis.at(7.5).unwrap();
        let translated = shift.bind_interval(&interval).unwrap();
        let direct = axis.interval(5.5, 10.75).unwrap();
        assert!(translated.max_abs_difference(&direct).unwrap() < 2e-13);
    }

    #[test]
    fn analytic_interval_matches_midpoint_quadrature() {
        let axis = TemporalAxis::new(24, 47).unwrap();
        let start = -0.75;
        let end = 1.4;
        let analytic = axis.interval(start, end).unwrap();
        let steps = 20_000usize;
        let dt = (end - start) / steps as f64;
        let mut real = vec![0.0; axis.dim()];
        let mut imag = vec![0.0; axis.dim()];
        for step in 0..steps {
            let time = start + (step as f64 + 0.5) * dt;
            let point = axis.at(time).unwrap();
            for i in 0..axis.dim() {
                real[i] += point.real()[i] * dt;
                imag[i] += point.imag()[i] * dt;
            }
        }
        let numerical = TemporalInterval { real, imag };
        assert!(analytic.max_abs_difference(&numerical).unwrap() < 2e-9);
    }

    #[test]
    fn malformed_temporal_inputs_fail_closed() {
        assert!(matches!(
            TemporalAxis::new(0, 1),
            Err(TemporalAlgebraError::ZeroDimension)
        ));
        let axis = TemporalAxis::new(8, 48).unwrap();
        assert!(matches!(
            axis.at(f64::NAN),
            Err(TemporalAlgebraError::NonFiniteTime(_))
        ));
        assert!(matches!(
            axis.interval(2.0, 1.0),
            Err(TemporalAlgebraError::InvalidInterval { .. })
        ));
    }
}
