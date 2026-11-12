//! Exact 2x2 cross-port return-ratio reference.
//!
//! This narrow model demonstrates why diagonal-only SISO projections cannot
//! replace a full two-port matrix when cross-port terms are present. It is a
//! synthetic loop benchmark, not a general generalized-Nyquist implementation
//! or a physical converter/aircraft model.

#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPortLoopParameters {
    /// Residue in L12(s) = residue_12 / (s - pole_12_per_s).
    pub residue_12: f64,
    /// Pole of the first cross-port channel, per second.
    pub pole_12_per_s: f64,
    /// Residue in L21(s) = residue_21 / (s - pole_21_per_s).
    pub residue_21: f64,
    /// Pole of the second cross-port channel, per second.
    pub pole_21_per_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MimoReferenceError {
    NonFiniteParameter(&'static str),
    NonFiniteDerivedValue(&'static str),
    InvalidAngularFrequency,
    SingularFrequencyResponse,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexValue {
    pub real: f64,
    pub imag: f64,
}

impl ComplexValue {
    const ZERO: Self = Self { real: 0.0, imag: 0.0 };
    const ONE: Self = Self { real: 1.0, imag: 0.0 };

    fn multiply(self, rhs: Self) -> Self {
        Self {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real,
        }
    }

    fn subtract(self, rhs: Self) -> Self {
        Self {
            real: self.real - rhs.real,
            imag: self.imag - rhs.imag,
        }
    }

    fn divide(self, rhs: Self) -> Result<Self, MimoReferenceError> {
        let denominator = rhs.real * rhs.real + rhs.imag * rhs.imag;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(MimoReferenceError::SingularFrequencyResponse);
        }
        let value = Self {
            real: (self.real * rhs.real + self.imag * rhs.imag) / denominator,
            imag: (self.imag * rhs.real - self.real * rhs.imag) / denominator,
        };
        validate_complex(value)?;
        Ok(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPortFrequencyResponse {
    pub angular_frequency_rad_s: f64,
    /// Row-major L(jw) = [[0, L12], [L21, 0]].
    pub return_ratio_matrix: [[ComplexValue; 2]; 2],
    /// det(I + L(jw)) under the declared negative-feedback convention.
    pub determinant_i_plus_l: ComplexValue,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPortMimoReport {
    /// Number of active off-diagonal transfer-channel poles in the open-loop
    /// matrix that lie strictly in the right half-plane.
    pub open_loop_rhp_pole_count: usize,
    /// True if either cross-port term is nonzero. In that case diagonal-only
    /// SISO projections discard information.
    pub diagonal_siso_is_incomplete: bool,
    /// Coefficients [1, a1, a0] of det(I + L(s)) after clearing the declared
    /// two channel denominators. None means at least one channel is absent,
    /// so the determinant is identically one and no coupled polynomial exists.
    pub closed_loop_characteristic: Option<[f64; 3]>,
    /// Positive real root of the cleared feedback characteristic, if proven
    /// by a negative constant term and computed with finite binary64 values.
    pub positive_real_feedback_pole_per_s: Option<f64>,
    /// dr_positive/d(residue_12), when a positive simple real root exists.
    pub positive_pole_sensitivity_to_residue_12: Option<f64>,
    /// dr_positive/d(residue_21), when a positive simple real root exists.
    pub positive_pole_sensitivity_to_residue_21: Option<f64>,
}

/// Analyze the exact two-port cross-coupling structure.
///
/// The feedback convention is det(I + L(s)) = 0 for
///
/// L(s) = [[0, k12/(s-p12)], [k21/(s-p21), 0]].
///
/// For nonzero k12 and k21, clearing the two declared denominators gives
/// (s-p12)(s-p21)-k12*k21 = 0. The declared matrix realization/pole metadata
/// must be retained alongside this polynomial; a sampled determinant trace
/// alone is not a generalized-Nyquist certificate.
pub fn analyze_two_port_loop(
    parameters: TwoPortLoopParameters,
) -> Result<TwoPortMimoReport, MimoReferenceError> {
    validate_parameters(parameters)?;

    let mut open_loop_rhp_pole_count = 0;
    if parameters.residue_12 != 0.0 && parameters.pole_12_per_s > 0.0 {
        open_loop_rhp_pole_count += 1;
    }
    if parameters.residue_21 != 0.0 && parameters.pole_21_per_s > 0.0 {
        open_loop_rhp_pole_count += 1;
    }

    let diagonal_siso_is_incomplete =
        parameters.residue_12 != 0.0 || parameters.residue_21 != 0.0;

    let has_both_cross_channels =
        parameters.residue_12 != 0.0 && parameters.residue_21 != 0.0;

    let closed_loop_characteristic = if has_both_cross_channels {
        let a1 = -(parameters.pole_12_per_s + parameters.pole_21_per_s);
        let a0 = parameters.pole_12_per_s * parameters.pole_21_per_s
            - parameters.residue_12 * parameters.residue_21;
        if !a1.is_finite() || !a0.is_finite() {
            return Err(MimoReferenceError::NonFiniteDerivedValue(
                "closed_loop_characteristic",
            ));
        }
        Some([1.0, a1, a0])
    } else {
        None
    };

    let (positive_root, sensitivity_12, sensitivity_21) =
        if let Some([_, a1, a0]) = closed_loop_characteristic {
            if a0 < 0.0 {
                // For a monic quadratic with a0 < 0, the real roots have
                // opposite signs. The positive root is simple.
                let discriminant = a1 * a1 - 4.0 * a0;
                if !discriminant.is_finite() || discriminant <= 0.0 {
                    return Err(MimoReferenceError::NonFiniteDerivedValue(
                        "closed_loop_discriminant",
                    ));
                }
                let root_gap = discriminant.sqrt();
                let root = (-a1 + root_gap) / 2.0;
                let sensitivity_scale = 1.0 / root_gap;
                let ds_dk12 = parameters.residue_21 * sensitivity_scale;
                let ds_dk21 = parameters.residue_12 * sensitivity_scale;
                if !root.is_finite()
                    || !sensitivity_scale.is_finite()
                    || !ds_dk12.is_finite()
                    || !ds_dk21.is_finite()
                {
                    return Err(MimoReferenceError::NonFiniteDerivedValue(
                        "positive_root_or_sensitivity",
                    ));
                }
                (Some(root), Some(ds_dk12), Some(ds_dk21))
            } else {
                (None, None, None)
            }
        } else {
            (None, None, None)
        };

    Ok(TwoPortMimoReport {
        open_loop_rhp_pole_count,
        diagonal_siso_is_incomplete,
        closed_loop_characteristic,
        positive_real_feedback_pole_per_s: positive_root,
        positive_pole_sensitivity_to_residue_12: sensitivity_12,
        positive_pole_sensitivity_to_residue_21: sensitivity_21,
    })
}

/// Evaluate the full 2x2 return-ratio matrix and det(I + L) at s = jw.
pub fn evaluate_two_port_frequency_response(
    parameters: TwoPortLoopParameters,
    angular_frequency_rad_s: f64,
) -> Result<TwoPortFrequencyResponse, MimoReferenceError> {
    validate_parameters(parameters)?;
    if !angular_frequency_rad_s.is_finite() || angular_frequency_rad_s < 0.0 {
        return Err(MimoReferenceError::InvalidAngularFrequency);
    }

    let jw = ComplexValue {
        real: 0.0,
        imag: angular_frequency_rad_s,
    };
    let l12 = ComplexValue {
        real: parameters.residue_12,
        imag: 0.0,
    }
    .divide(ComplexValue {
        real: -parameters.pole_12_per_s,
        imag: jw.imag,
    })?;
    let l21 = ComplexValue {
        real: parameters.residue_21,
        imag: 0.0,
    }
    .divide(ComplexValue {
        real: -parameters.pole_21_per_s,
        imag: jw.imag,
    })?;

    let determinant = ComplexValue::ONE.subtract(l12.multiply(l21));
    validate_complex(determinant)?;

    Ok(TwoPortFrequencyResponse {
        angular_frequency_rad_s,
        return_ratio_matrix: [
            [ComplexValue::ZERO, l12],
            [l21, ComplexValue::ZERO],
        ],
        determinant_i_plus_l: determinant,
    })
}

fn validate_parameters(
    parameters: TwoPortLoopParameters,
) -> Result<(), MimoReferenceError> {
    for (name, value) in [
        ("residue_12", parameters.residue_12),
        ("pole_12_per_s", parameters.pole_12_per_s),
        ("residue_21", parameters.residue_21),
        ("pole_21_per_s", parameters.pole_21_per_s),
    ] {
        if !value.is_finite() {
            return Err(MimoReferenceError::NonFiniteParameter(name));
        }
    }
    Ok(())
}

fn validate_complex(value: ComplexValue) -> Result<(), MimoReferenceError> {
    if !value.real.is_finite() || !value.imag.is_finite() {
        return Err(MimoReferenceError::NonFiniteDerivedValue(
            "complex_frequency_response",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> TwoPortLoopParameters {
        TwoPortLoopParameters {
            residue_12: 1.0,
            pole_12_per_s: 1.0,
            residue_21: 1.0,
            pole_21_per_s: -2.0,
        }
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.3e}"
        );
    }

    #[test]
    fn diagonal_siso_projections_miss_the_cross_port_positive_pole() {
        let report = analyze_two_port_loop(fixture()).unwrap();
        assert!(report.diagonal_siso_is_incomplete);
        assert_eq!(report.open_loop_rhp_pole_count, 1);
        let [_, a1, a0] = report.closed_loop_characteristic.unwrap();
        close(a1, 1.0, 1e-12);
        close(a0, -3.0, 1e-12);
        close(
            report.positive_real_feedback_pole_per_s.unwrap(),
            (-1.0 + 13.0_f64.sqrt()) / 2.0,
            1e-12,
        );
        close(
            report.positive_pole_sensitivity_to_residue_12.unwrap(),
            1.0 / 13.0_f64.sqrt(),
            1e-12,
        );
        close(
            report.positive_pole_sensitivity_to_residue_21.unwrap(),
            1.0 / 13.0_f64.sqrt(),
            1e-12,
        );
    }

    #[test]
    fn exact_dc_and_one_radian_frequency_responses_match() {
        let p = fixture();
        let dc = evaluate_two_port_frequency_response(p, 0.0).unwrap();
        close(dc.return_ratio_matrix[0][1].real, -1.0, 1e-12);
        close(dc.return_ratio_matrix[1][0].real, 0.5, 1e-12);
        close(dc.determinant_i_plus_l.real, 1.5, 1e-12);
        close(dc.determinant_i_plus_l.imag, 0.0, 1e-12);

        let one = evaluate_two_port_frequency_response(p, 1.0).unwrap();
        close(one.return_ratio_matrix[0][1].real, -0.5, 1e-12);
        close(one.return_ratio_matrix[0][1].imag, -0.5, 1e-12);
        close(one.return_ratio_matrix[1][0].real, 0.4, 1e-12);
        close(one.return_ratio_matrix[1][0].imag, -0.2, 1e-12);
        close(one.determinant_i_plus_l.real, 1.3, 1e-12);
        close(one.determinant_i_plus_l.imag, 0.1, 1e-12);
    }

    #[test]
    fn removing_either_cross_channel_removes_feedback_characteristic() {
        let mut p = fixture();
        p.residue_21 = 0.0;
        let report = analyze_two_port_loop(p).unwrap();
        assert_eq!(report.closed_loop_characteristic, None);
        assert_eq!(report.positive_real_feedback_pole_per_s, None);
        // The active L12 channel still carries its +1 open-loop pole. It must
        // not disappear merely because det(I+L) is identically one.
        assert_eq!(report.open_loop_rhp_pole_count, 1);
    }

    #[test]
    fn invalid_frequency_and_non_finite_parameters_fail_closed() {
        assert_eq!(
            evaluate_two_port_frequency_response(fixture(), f64::NAN),
            Err(MimoReferenceError::InvalidAngularFrequency)
        );
        let mut p = fixture();
        p.residue_12 = f64::INFINITY;
        assert_eq!(
            analyze_two_port_loop(p),
            Err(MimoReferenceError::NonFiniteParameter("residue_12"))
        );
    }
}
