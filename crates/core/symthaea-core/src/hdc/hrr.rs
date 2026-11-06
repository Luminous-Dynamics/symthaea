// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Holographic Reduced Representations (HRR) over continuous hypervectors.
//!
//! This module is intentionally separate from ContinuousHV::bind(), whose
//! current algebra is Hadamard multiplication. HRR binding here is circular
//! convolution, accelerated by an in-crate radix-2 FFT. A unitary HRR role has
//! unit-magnitude Fourier coefficients, making circular correlation an exact
//! inverse (up to floating-point error).
//!
//! Binding is commutative. Role/value order is represented through distinct
//! role vectors or explicit permutation/sequence encodings, not by claiming
//! circular convolution itself is non-commutative.

use num_complex::Complex32;

use super::unified_hv::ContinuousHV;

/// Construction/application errors for the HRR backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HdcHrrError {
    ZeroDimension,
    NonPowerOfTwo { dimension: usize },
    DimensionMismatch { expected: usize, actual: usize },
    NonFiniteValue { index: usize },
    ZeroSpectrumBin { index: usize },
}

impl std::fmt::Display for HdcHrrError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroDimension => write!(f, "HRR dimension must be non-zero"),
            Self::NonPowerOfTwo { dimension } => write!(
                f,
                "radix-2 HRR FFT requires a power-of-two dimension, got {dimension}"
            ),
            Self::DimensionMismatch { expected, actual } => write!(
                f,
                "HRR dimension mismatch: expected {expected}, received {actual}"
            ),
            Self::NonFiniteValue { index } => {
                write!(f, "HRR input at index {index} is not finite")
            }
            Self::ZeroSpectrumBin { index } => {
                write!(f, "cannot unitary-normalize near-zero HRR spectrum bin {index}")
            }
        }
    }
}

impl std::error::Error for HdcHrrError {}

/// A deterministic unitary HRR role vector.
///
/// Its Fourier-domain coefficients all have unit magnitude. Thus for any
/// filler x, corr(conv(x, role), role) reconstructs x to floating-point
/// precision. The role remains a real-valued ContinuousHV in the time domain.
#[derive(Debug, Clone)]
pub struct HdcHrrRole {
    vector: ContinuousHV,
}

impl HdcHrrRole {
    /// Build a real, deterministic, unitary HRR role from a seed.
    pub fn from_seed(dimension: usize, seed: u64) -> Result<Self, HdcHrrError> {
        validate_dimension(dimension)?;
        let source = ContinuousHV::random(dimension, seed);
        let mut spectrum = to_complex(&source)?;
        fft_in_place(&mut spectrum, false);

        // Enforce a real-valued inverse transform while making every spectral
        // coefficient have unit magnitude. Positive-frequency phases are
        // mirrored by conjugation to preserve Hermitian symmetry.
        spectrum[0] = Complex32::new(if spectrum[0].re < 0.0 { -1.0 } else { 1.0 }, 0.0);
        if dimension > 1 {
            let nyquist = dimension / 2;
            spectrum[nyquist] =
                Complex32::new(if spectrum[nyquist].re < 0.0 { -1.0 } else { 1.0 }, 0.0);

            for index in 1..nyquist {
                let magnitude = spectrum[index].norm();
                if !magnitude.is_finite() || magnitude < 1e-12 {
                    return Err(HdcHrrError::ZeroSpectrumBin { index });
                }
                let phase = spectrum[index] / magnitude;
                spectrum[index] = phase;
                spectrum[dimension - index] = phase.conj();
            }
        }

        fft_in_place(&mut spectrum, true)?;
        let vector = ContinuousHV::from_values(spectrum.into_iter().map(|z| z.re).collect());
        Ok(Self { vector })
    }

    /// Borrow the real-valued unitary role vector.
    pub fn vector(&self) -> &ContinuousHV {
        &self.vector
    }

    /// Encode a filler using circular convolution: role ⊛ filler.
    pub fn bind(&self, filler: &ContinuousHV) -> Result<ContinuousHV, HdcHrrError> {
        circular_convolution(&self.vector, filler)
    }

    /// Recover a filler using circular correlation with this unitary role.
    pub fn unbind(&self, bound: &ContinuousHV) -> Result<ContinuousHV, HdcHrrError> {
        circular_correlation(bound, &self.vector)
    }
}

/// Circular convolution using FFT: ifft(fft(left) * fft(right)).
///
/// Complexity is O(D log D); both inputs and output have dimension D.
pub fn circular_convolution(
    left: &ContinuousHV,
    right: &ContinuousHV,
) -> Result<ContinuousHV, HdcHrrError> {
    validate_pair(left, right)?;
    let mut left_spectrum = to_complex(left)?;
    let mut right_spectrum = to_complex(right)?;

    fft_in_place(&mut left_spectrum, false)?;
    fft_in_place(&mut right_spectrum, false)?;

    for (left_bin, right_bin) in left_spectrum.iter_mut().zip(right_spectrum.iter()) {
        *left_bin *= *right_bin;
    }

    fft_in_place(&mut left_spectrum, true)?;
    Ok(ContinuousHV::from_values(
        left_spectrum.into_iter().map(|z| z.re).collect(),
    ))
}

/// Circular correlation using FFT: ifft(fft(bound) * conj(fft(role))).
///
/// This is an exact inverse for unitary HRR roles and an approximate cleanup
/// operation for general non-unitary roles.
pub fn circular_correlation(
    bound: &ContinuousHV,
    role: &ContinuousHV,
) -> Result<ContinuousHV, HdcHrrError> {
    validate_pair(bound, role)?;
    let mut bound_spectrum = to_complex(bound)?;
    let mut role_spectrum = to_complex(role)?;

    fft_in_place(&mut bound_spectrum, false)?;
    fft_in_place(&mut role_spectrum, false)?;

    for (bound_bin, role_bin) in bound_spectrum.iter_mut().zip(role_spectrum.iter()) {
        *bound_bin *= role_bin.conj();
    }

    fft_in_place(&mut bound_spectrum, true)?;
    Ok(ContinuousHV::from_values(
        bound_spectrum.into_iter().map(|z| z.re).collect(),
    ))
}

fn validate_dimension(dimension: usize) -> Result<(), HdcHrrError> {
    if dimension == 0 {
        return Err(HdcHrrError::ZeroDimension);
    }
    if !dimension.is_power_of_two() {
        return Err(HdcHrrError::NonPowerOfTwo { dimension });
    }
    Ok(())
}

fn validate_pair(left: &ContinuousHV, right: &ContinuousHV) -> Result<(), HdcHrrError> {
    let dimension = left.dim();
    validate_dimension(dimension)?;
    if right.dim() != dimension {
        return Err(HdcHrrError::DimensionMismatch {
            expected: dimension,
            actual: right.dim(),
        });
    }
    for (index, value) in left.values.iter().chain(right.values.iter()).enumerate() {
        if !value.is_finite() {
            return Err(HdcHrrError::NonFiniteValue { index });
        }
    }
    Ok(())
}

fn to_complex(input: &ContinuousHV) -> Result<Vec<Complex32>, HdcHrrError> {
    let mut values = Vec::with_capacity(input.dim());
    for (index, &value) in input.values.iter().enumerate() {
        if !value.is_finite() {
            return Err(HdcHrrError::NonFiniteValue { index });
        }
        values.push(Complex32::new(value, 0.0));
    }
    Ok(values)
}

/// In-place radix-2 Cooley–Tukey FFT.
///
/// Forward transform is unnormalized. Inverse transform divides by N.
fn fft_in_place(values: &mut [Complex32], inverse: bool) -> Result<(), HdcHrrError> {
    let n = values.len();
    validate_dimension(n)?;

    // Bit-reversal permutation.
    let mut reversed = 0usize;
    for index in 1..n {
        let mut bit = n >> 1;
        while (reversed & bit) != 0 {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if index < reversed {
            values.swap(index, reversed);
        }
    }

    let direction = if inverse { 1.0 } else { -1.0 };
    let mut width = 2usize;
    while width <= n {
        let half = width / 2;
        let angle = direction * std::f32::consts::TAU / width as f32;
        let root = Complex32::from_polar(1.0, angle);

        for block_start in (0..n).step_by(width) {
            let mut twiddle = Complex32::new(1.0, 0.0);
            for offset in 0..half {
                let even = values[block_start + offset];
                let odd = values[block_start + offset + half] * twiddle;
                values[block_start + offset] = even + odd;
                values[block_start + offset + half] = even - odd;
                twiddle *= root;
            }
        }

        width = width.saturating_mul(2);
    }

    if inverse {
        let scale = 1.0 / n as f32;
        values.iter_mut().for_each(|value| *value *= scale);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_difference(left: &ContinuousHV, right: &ContinuousHV) -> f32 {
        left.values
            .iter()
            .zip(right.values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn unitary_role_exactly_unbinds_filler() {
        for dimension in [1, 2, 16, 64, 256, 4096, 16_384] {
            let role = HdcHrrRole::from_seed(dimension, 0x4852_5221).unwrap();
            let filler = ContinuousHV::random(dimension, 0xF111_EA);

            let bound = role.bind(&filler).unwrap();
            let recovered = role.unbind(&bound).unwrap();

            assert!(
                max_abs_difference(&filler, &recovered) < 2e-4,
                "HRR reconstruction error exceeded tolerance at dimension {dimension}"
            );
            assert!(
                (filler.norm() - bound.norm()).abs() < 2e-4,
                "unitary role should preserve L2 norm at dimension {dimension}"
            );
        }
    }

    #[test]
    fn circular_convolution_is_commutative() {
        let left = ContinuousHV::random(128, 101);
        let right = ContinuousHV::random(128, 202);
        let lr = circular_convolution(&left, &right).unwrap();
        let rl = circular_convolution(&right, &left).unwrap();
        assert!(max_abs_difference(&lr, &rl) < 2e-5);
    }

    #[test]
    fn separate_roles_support_independent_slot_retrieval() {
        let dimension = 1024;
        let subject_role = HdcHrrRole::from_seed(dimension, 11).unwrap();
        let relation_role = HdcHrrRole::from_seed(dimension, 12).unwrap();
        let subject = ContinuousHV::random(dimension, 21);
        let relation = ContinuousHV::random(dimension, 22);

        let subject_slot = subject_role.bind(&subject).unwrap();
        let relation_slot = relation_role.bind(&relation).unwrap();
        let record = subject_slot.add(&relation_slot);

        let recovered_subject = subject_role.unbind(&record).unwrap();
        let recovered_relation = relation_role.unbind(&record).unwrap();

        assert!(recovered_subject.similarity(&subject) > 0.70);
        assert!(recovered_relation.similarity(&relation) > 0.70);
    }

    #[test]
    fn bundled_record_retrieves_each_role_filler_at_nontrivial_load() {
        // Deterministic associative-retrieval smoke benchmark. This is not a
        // capacity claim; the load/dimension curve belongs in the benchmark suite.
        let dimension = 2048;
        let load = 32;
        let roles: Vec<HdcHrrRole> = (0..load)
            .map(|index| HdcHrrRole::from_seed(dimension, 10_000 + index as u64).unwrap())
            .collect();
        let fillers: Vec<ContinuousHV> = (0..load)
            .map(|index| ContinuousHV::random(dimension, 20_000 + index as u64))
            .collect();

        let mut record = ContinuousHV::zero(dimension);
        for (role, filler) in roles.iter().zip(fillers.iter()) {
            record = record.add(&role.bind(filler).unwrap());
        }

        let correct_retrievals = roles
            .iter()
            .enumerate()
            .filter(|(index, role)| {
                let recovered = role.unbind(&record).unwrap();
                let best = fillers
                    .iter()
                    .enumerate()
                    .map(|(candidate_index, candidate)| {
                        (candidate_index, recovered.similarity(candidate))
                    })
                    .max_by(|left, right| left.1.total_cmp(&right.1))
                    .map(|(candidate_index, _)| candidate_index)
                    .unwrap_or(usize::MAX);
                best == *index
            })
            .count();

        assert!(
            correct_retrievals >= load * 30 / 32,
            "HRR role-filler top-1 retrieval degraded at dimension={dimension}, load={load}: {correct_retrievals}/{load}"
        );
    }

    #[test]
    fn rejects_invalid_dimensions_and_non_finite_values() {
        assert!(matches!(
            HdcHrrRole::from_seed(0, 1),
            Err(HdcHrrError::ZeroDimension)
        ));
        assert!(matches!(
            HdcHrrRole::from_seed(12, 1),
            Err(HdcHrrError::NonPowerOfTwo { dimension: 12 })
        ));

        let left = ContinuousHV::random(64, 10);
        let right = ContinuousHV::random(32, 11);
        assert!(matches!(
            circular_convolution(&left, &right),
            Err(HdcHrrError::DimensionMismatch {
                expected: 64,
                actual: 32
            })
        ));

        let mut invalid = ContinuousHV::random(64, 12);
        invalid.values[17] = f32::INFINITY;
        assert!(matches!(
            circular_correlation(&invalid, &left),
            Err(HdcHrrError::NonFiniteValue { index: 17 })
        ));
    }
}
