//! Source-faithful inverse adapter for ADEPT's shifted spatial Fourier rows.
//!
//! Pinned upstream: ergodicio/adept@0c85ce0d3d66abab51212191c2644f58a31a49cd.
//!
//! ADEPT forms coefficients with fft(..., norm="forward"), then applies
//! fftshift on the final (spatial Fourier) axis. It reconstructs rows with
//! ifft(..., norm="forward").real. The local radix-2 inverse FFT divides by
//! N, whereas ADEPT's norm="forward" inverse does not; this adapter therefore
//! multiplies the local inverse result by N. The shift reversal is performed
//! only on the spatial Fourier axis.
//!
//! The input and output are row-major (mode, kx) and (mode, x), respectively.
//! Complex phase is retained until after the inverse transform. The imaginary
//! residual is measured before taking the real part and is a fail-closed gate:
//! it is never silently clipped or hidden.

/// Immutable source identity for the Fourier normalization and ordering below.
pub const ADEPT_FOURIER_SOURCE_COMMIT: &str =
    "ergodicio/adept@0c85ce0d3d66abab51212191c2644f58a31a49cd";

/// Human-readable description of the pinned transform convention.
pub const ADEPT_FOURIER_CONVENTION: &str =
    "fft(norm=forward), fftshift(last axis), ifft(norm=forward).real";

/// One complex ADEPT Fourier coefficient, stored without losing signed phase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexFourierCoefficient {
    pub real: f64,
    pub imaginary: f64,
}

impl ComplexFourierCoefficient {
    pub const fn new(real: f64, imaginary: f64) -> Self {
        Self { real, imaginary }
    }
}

/// ADEPT coefficient rows in row-major (mode, shifted_kx) order.
///
/// The shift applies to the spatial Fourier axis only. Hermite and Legendre
/// mode indices are independent rows and are not shifted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShiftedFourierCoefficientFields<'a> {
    pub mode_count: usize,
    pub spatial_count: usize,
    pub coefficients: &'a [ComplexFourierCoefficient],
}

/// Reconstructed real coefficient rows in row-major (mode, x) order.
#[derive(Debug, Clone, PartialEq)]
pub struct RealSpaceCoefficientFields {
    mode_count: usize,
    spatial_count: usize,
    coefficients: Vec<f64>,
    maximum_imaginary_residual: f64,
    maximum_real_magnitude: f64,
}

impl RealSpaceCoefficientFields {
    pub fn mode_count(&self) -> usize {
        self.mode_count
    }

    pub fn spatial_count(&self) -> usize {
        self.spatial_count
    }

    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    pub fn get(&self, mode_index: usize, spatial_index: usize) -> Option<f64> {
        if mode_index >= self.mode_count || spatial_index >= self.spatial_count {
            return None;
        }
        self.coefficients
            .get(mode_index * self.spatial_count + spatial_index)
            .copied()
    }

    /// Largest absolute imaginary sample before the source-mandated real part.
    pub fn maximum_imaginary_residual(&self) -> f64 {
        self.maximum_imaginary_residual
    }

    /// Largest absolute reconstructed real sample.
    pub fn maximum_real_magnitude(&self) -> f64 {
        self.maximum_real_magnitude
    }
}

/// Invalid dimensions, source coefficients, or real-output qualification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FourierAdapterError {
    InvalidSpatialGrid {
        spatial_count: usize,
    },
    NoRepresentedModes,
    ShapeOverflow,
    CoefficientShape {
        expected: usize,
        actual: usize,
    },
    NonFiniteCoefficient {
        index: usize,
    },
    InvalidImaginaryTolerance,
    AllocationFailed {
        elements: usize,
    },
    NonFiniteReconstruction {
        mode_index: usize,
        spatial_index: usize,
    },
    ImaginaryResidualExceeded {
        observed: f64,
        allowed: f64,
    },
}

/// Convert ADEPT's shifted complex Fourier rows to physical-space real rows.
///
/// This reproduces the pinned ADEPT normalization:
///
/// - the forward transform is divided by spatial_count;
/// - the stored rows are fftshift-ordered along the spatial Fourier axis;
/// - the inverse transform is unscaled, matching NumPy/JAX
///   ifft(..., norm="forward");
/// - only after measuring the imaginary residual is the real part retained.
///
/// maximum_imaginary_tolerance is a caller-frozen absolute tolerance. The
/// function returns an error, including the observed and allowed residuals,
/// when the reconstructed rows are not sufficiently real.
///
/// The result can be passed directly to MixedCoefficientFields as either
/// the Hermite or Legendre coefficient slice. It does not decode NetCDF,
/// perform the Hermite-to-Legendre closure, solve Poisson, or evolve time.
pub fn adept_shifted_fourier_to_real_space(
    fields: ShiftedFourierCoefficientFields<'_>,
    maximum_imaginary_tolerance: f64,
) -> Result<RealSpaceCoefficientFields, FourierAdapterError> {
    let mode_count = fields.mode_count;
    let spatial_count = fields.spatial_count;

    if spatial_count < 4 || !spatial_count.is_power_of_two() {
        return Err(FourierAdapterError::InvalidSpatialGrid { spatial_count });
    }
    if mode_count == 0 {
        return Err(FourierAdapterError::NoRepresentedModes);
    }
    if !maximum_imaginary_tolerance.is_finite() || maximum_imaginary_tolerance < 0.0 {
        return Err(FourierAdapterError::InvalidImaginaryTolerance);
    }

    let expected = mode_count
        .checked_mul(spatial_count)
        .ok_or(FourierAdapterError::ShapeOverflow)?;
    if fields.coefficients.len() != expected {
        return Err(FourierAdapterError::CoefficientShape {
            expected,
            actual: fields.coefficients.len(),
        });
    }

    for (index, coefficient) in fields.coefficients.iter().enumerate() {
        if !coefficient.real.is_finite() || !coefficient.imaginary.is_finite() {
            return Err(FourierAdapterError::NonFiniteCoefficient { index });
        }
    }

    let mut output = Vec::new();
    output
        .try_reserve_exact(expected)
        .map_err(|_| FourierAdapterError::AllocationFailed { elements: expected })?;
    output.resize(expected, 0.0);

    let mut row = Vec::new();
    row.try_reserve_exact(spatial_count)
        .map_err(|_| FourierAdapterError::AllocationFailed {
            elements: spatial_count,
        })?;

    let half = spatial_count / 2;
    let mut maximum_imaginary_residual: f64 = 0.0;
    let mut maximum_real_magnitude: f64 = 0.0;

    for mode_index in 0..mode_count {
        row.clear();

        // Undo fftshift (equivalent to ifftshift for the even power-of-two
        // spatial grids admitted by this adapter), without touching mode rows.
        for raw_k_index in 0..spatial_count {
            let shifted_k_index = (raw_k_index + half) % spatial_count;
            let input_index = mode_index * spatial_count + shifted_k_index;
            let coefficient = fields.coefficients[input_index];
            row.push(super::Complex::new(
                coefficient.real,
                coefficient.imaginary,
            ));
        }

        // The local inverse divides by N. ADEPT's norm="forward" inverse is
        // unscaled, so multiplying by N reproduces its exact normalization.
        super::fft(&mut row, true);
        let source_scale = spatial_count as f64;

        for (spatial_index, value) in row.iter().copied().enumerate() {
            let reconstructed = value.scale(source_scale);
            if !reconstructed.re.is_finite() || !reconstructed.im.is_finite() {
                return Err(FourierAdapterError::NonFiniteReconstruction {
                    mode_index,
                    spatial_index,
                });
            }

            maximum_imaginary_residual =
                maximum_imaginary_residual.max(reconstructed.im.abs());
            maximum_real_magnitude =
                maximum_real_magnitude.max(reconstructed.re.abs());

            output[mode_index * spatial_count + spatial_index] = reconstructed.re;
        }
    }

    if maximum_imaginary_residual > maximum_imaginary_tolerance {
        return Err(FourierAdapterError::ImaginaryResidualExceeded {
            observed: maximum_imaginary_residual,
            allowed: maximum_imaginary_tolerance,
        });
    }

    Ok(RealSpaceCoefficientFields {
        mode_count,
        spatial_count,
        coefficients: output,
        maximum_imaginary_residual,
        maximum_real_magnitude,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    fn shifted_index(raw_k_index: usize, spatial_count: usize) -> usize {
        (raw_k_index + spatial_count / 2) % spatial_count
    }

    fn raw_index(signed_mode: isize, spatial_count: usize) -> usize {
        signed_mode.rem_euclid(spatial_count as isize) as usize
    }

    fn set_shifted_mode(
        values: &mut [ComplexFourierCoefficient],
        mode_index: usize,
        spatial_count: usize,
        signed_mode: isize,
        coefficient: ComplexFourierCoefficient,
    ) {
        let shifted = shifted_index(raw_index(signed_mode, spatial_count), spatial_count);
        values[mode_index * spatial_count + shifted] = coefficient;
    }

    fn add_real_cosine(
        values: &mut [ComplexFourierCoefficient],
        mode_index: usize,
        spatial_count: usize,
        signed_mode: isize,
        amplitude: f64,
        phase: f64,
    ) {
        let positive = ComplexFourierCoefficient::new(
            0.5 * amplitude * phase.cos(),
            0.5 * amplitude * phase.sin(),
        );
        let negative =
            ComplexFourierCoefficient::new(positive.real, -positive.imaginary);
        set_shifted_mode(values, mode_index, spatial_count, signed_mode, positive);
        set_shifted_mode(values, mode_index, spatial_count, -signed_mode, negative);
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn constant_row_reproduces_source_forward_normalization() {
        let spatial_count = 8;
        let mut coefficients =
            vec![ComplexFourierCoefficient::new(0.0, 0.0); spatial_count];
        set_shifted_mode(
            &mut coefficients,
            0,
            spatial_count,
            0,
            ComplexFourierCoefficient::new(2.5, 0.0),
        );

        let real = adept_shifted_fourier_to_real_space(
            ShiftedFourierCoefficientFields {
                mode_count: 1,
                spatial_count,
                coefficients: &coefficients,
            },
            1.0e-13,
        )
        .unwrap();

        for x in 0..spatial_count {
            close(real.get(0, x).unwrap(), 2.5, 1.0e-13);
        }
        assert!(real.maximum_imaginary_residual() <= 1.0e-13);
        close(real.maximum_real_magnitude(), 2.5, 1.0e-13);
    }

    #[test]
    fn positive_fourier_mode_pair_preserves_cosine_phase() {
        let spatial_count = 16;
        let phase = 0.37;
        let amplitude = 1.25;
        let mut coefficients =
            vec![ComplexFourierCoefficient::new(0.0, 0.0); spatial_count];
        add_real_cosine(&mut coefficients, 0, spatial_count, 2, amplitude, phase);

        let real = adept_shifted_fourier_to_real_space(
            ShiftedFourierCoefficientFields {
                mode_count: 1,
                spatial_count,
                coefficients: &coefficients,
            },
            2.0e-13,
        )
        .unwrap();

        for x in 0..spatial_count {
            let theta = TAU * 2.0 * x as f64 / spatial_count as f64;
            close(
                real.get(0, x).unwrap(),
                amplitude * (theta + phase).cos(),
                2.0e-13,
            );
        }
    }

    #[test]
    fn negative_fourier_mode_pair_preserves_signed_phase() {
        let spatial_count = 16;
        let phase = -0.29;
        let amplitude = 0.8;
        let mut coefficients =
            vec![ComplexFourierCoefficient::new(0.0, 0.0); spatial_count];
        add_real_cosine(&mut coefficients, 0, spatial_count, -3, amplitude, phase);

        let real = adept_shifted_fourier_to_real_space(
            ShiftedFourierCoefficientFields {
                mode_count: 1,
                spatial_count,
                coefficients: &coefficients,
            },
            2.0e-13,
        )
        .unwrap();

        for x in 0..spatial_count {
            let theta = -TAU * 3.0 * x as f64 / spatial_count as f64;
            close(
                real.get(0, x).unwrap(),
                amplitude * (theta + phase).cos(),
                2.0e-13,
            );
        }
    }

    #[test]
    fn multiple_modes_and_rows_preserve_superposition_and_row_major_order() {
        let spatial_count = 16;
        let mut coefficients = vec![
            ComplexFourierCoefficient::new(0.0, 0.0);
            2 * spatial_count
        ];
        add_real_cosine(&mut coefficients, 0, spatial_count, 1, 1.1, 0.2);
        add_real_cosine(&mut coefficients, 0, spatial_count, 4, 0.3, -0.4);
        add_real_cosine(&mut coefficients, 1, spatial_count, -2, 0.7, 0.6);

        let real = adept_shifted_fourier_to_real_space(
            ShiftedFourierCoefficientFields {
                mode_count: 2,
                spatial_count,
                coefficients: &coefficients,
            },
            3.0e-13,
        )
        .unwrap();

        for x in 0..spatial_count {
            let theta = TAU * x as f64 / spatial_count as f64;
            let expected_first = 1.1 * (theta + 0.2).cos()
                + 0.3 * (4.0 * theta - 0.4).cos();
            let expected_second = 0.7 * (-2.0 * theta + 0.6).cos();
            close(real.get(0, x).unwrap(), expected_first, 3.0e-13);
            close(real.get(1, x).unwrap(), expected_second, 3.0e-13);
        }
        assert_eq!(real.get(0, spatial_count), None);
        assert_eq!(real.get(2, 0), None);
    }

    #[test]
    fn inverse_fftshift_is_applied_only_on_the_spatial_axis() {
        let spatial_count = 8;
        let mut coefficients = vec![
            ComplexFourierCoefficient::new(0.0, 0.0);
            2 * spatial_count
        ];
        set_shifted_mode(
            &mut coefficients,
            0,
            spatial_count,
            0,
            ComplexFourierCoefficient::new(1.0, 0.0),
        );
        set_shifted_mode(
            &mut coefficients,
            1,
            spatial_count,
            0,
            ComplexFourierCoefficient::new(-2.0, 0.0),
        );

        let real = adept_shifted_fourier_to_real_space(
            ShiftedFourierCoefficientFields {
                mode_count: 2,
                spatial_count,
                coefficients: &coefficients,
            },
            1.0e-13,
        )
        .unwrap();

        for x in 0..spatial_count {
            close(real.get(0, x).unwrap(), 1.0, 1.0e-13);
            close(real.get(1, x).unwrap(), -2.0, 1.0e-13);
        }
    }

    #[test]
    fn imaginary_residual_is_reported_and_rejected_without_silent_clipping() {
        let spatial_count = 8;
        let mut coefficients =
            vec![ComplexFourierCoefficient::new(0.0, 0.0); spatial_count];
        set_shifted_mode(
            &mut coefficients,
            0,
            spatial_count,
            0,
            ComplexFourierCoefficient::new(1.0, 1.0e-4),
        );

        let error = adept_shifted_fourier_to_real_space(
            ShiftedFourierCoefficientFields {
                mode_count: 1,
                spatial_count,
                coefficients: &coefficients,
            },
            1.0e-5,
        )
        .unwrap_err();

        match error {
            FourierAdapterError::ImaginaryResidualExceeded { observed, allowed } => {
                close(observed, 1.0e-4, 1.0e-13);
                close(allowed, 1.0e-5, 0.0);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn malformed_spatial_grid_is_rejected() {
        for spatial_count in [0, 3, 6, 12] {
            let error = adept_shifted_fourier_to_real_space(
                ShiftedFourierCoefficientFields {
                    mode_count: 1,
                    spatial_count,
                    coefficients: &[],
                },
                1.0e-12,
            );
            assert_eq!(
                error,
                Err(FourierAdapterError::InvalidSpatialGrid { spatial_count })
            );
        }
    }

    #[test]
    fn zero_modes_and_wrong_shapes_are_rejected() {
        let no_modes = adept_shifted_fourier_to_real_space(
            ShiftedFourierCoefficientFields {
                mode_count: 0,
                spatial_count: 8,
                coefficients: &[],
            },
            1.0e-12,
        );
        assert_eq!(no_modes, Err(FourierAdapterError::NoRepresentedModes));

        let wrong_shape = adept_shifted_fourier_to_real_space(
            ShiftedFourierCoefficientFields {
                mode_count: 2,
                spatial_count: 8,
                coefficients: &[ComplexFourierCoefficient::new(0.0, 0.0); 8],
            },
            1.0e-12,
        );
        assert_eq!(
            wrong_shape,
            Err(FourierAdapterError::CoefficientShape {
                expected: 16,
                actual: 8,
            })
        );
    }

    #[test]
    fn non_finite_inputs_and_invalid_tolerances_are_rejected() {
        let mut coefficients =
            vec![ComplexFourierCoefficient::new(0.0, 0.0); 8];
        coefficients[3] = ComplexFourierCoefficient::new(f64::NAN, 0.0);
        let non_finite = adept_shifted_fourier_to_real_space(
            ShiftedFourierCoefficientFields {
                mode_count: 1,
                spatial_count: 8,
                coefficients: &coefficients,
            },
            1.0e-12,
        );
        assert_eq!(
            non_finite,
            Err(FourierAdapterError::NonFiniteCoefficient { index: 3 })
        );

        let finite = [ComplexFourierCoefficient::new(0.0, 0.0); 8];
        for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(
                adept_shifted_fourier_to_real_space(
                    ShiftedFourierCoefficientFields {
                        mode_count: 1,
                        spatial_count: 8,
                        coefficients: &finite,
                    },
                    tolerance,
                ),
                Err(FourierAdapterError::InvalidImaginaryTolerance)
            );
        }
    }

    #[test]
    fn multiplication_by_spatial_count_is_required_for_adept_inverse_convention() {
        let spatial_count = 32;
        let physical_constant = -0.625;
        let mut coefficients =
            vec![ComplexFourierCoefficient::new(0.0, 0.0); spatial_count];
        set_shifted_mode(
            &mut coefficients,
            0,
            spatial_count,
            0,
            ComplexFourierCoefficient::new(physical_constant, 0.0),
        );

        let real = adept_shifted_fourier_to_real_space(
            ShiftedFourierCoefficientFields {
                mode_count: 1,
                spatial_count,
                coefficients: &coefficients,
            },
            1.0e-13,
        )
        .unwrap();

        for value in real.coefficients() {
            close(*value, physical_constant, 1.0e-13);
        }
    }
}
