//! Signed-intervention parity decomposition for complex kinetic responses.
//!
//! This primitive separates odd- and even-in-amplitude components from matched
//! negative, zero and positive interventions. It is a diagnostic operation,
//! not a causal-mechanism classifier: higher-order terms and numerical effects
//! still require amplitude sweeps, refinement and control experiments.

use std::fmt;

/// Complex response coefficient for one spatial Fourier mode at one time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ComplexResponse {
    pub real: f64,
    pub imaginary: f64,
}

impl ComplexResponse {
    pub const fn new(real: f64, imaginary: f64) -> Self {
        Self { real, imaginary }
    }

    pub fn amplitude(self) -> f64 {
        self.real.hypot(self.imaginary)
    }

    pub fn is_finite(self) -> bool {
        self.real.is_finite() && self.imaginary.is_finite()
    }

    fn add(self, other: Self) -> Self {
        Self::new(self.real + other.real, self.imaginary + other.imaginary)
    }

    fn subtract(self, other: Self) -> Self {
        Self::new(self.real - other.real, self.imaginary - other.imaginary)
    }

    fn scale(self, factor: f64) -> Self {
        Self::new(self.real * factor, self.imaginary * factor)
    }
}

/// A measured response for a single spatial mode and physical sample time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeResponseSample {
    pub spatial_mode: i32,
    pub time: f64,
    pub response: ComplexResponse,
}

/// Odd/even decomposition of a matched signed-amplitude intervention triplet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignedResponseParity {
    pub spatial_mode: i32,
    pub time: f64,
    /// [R(+A) - R(-A)] / 2.
    pub odd_component: ComplexResponse,
    /// [R(+A) + R(-A) - 2 R(0)] / 2.
    pub even_component: ComplexResponse,
    /// Odd component divided by A. It includes O(A^2) corrections in general.
    pub odd_per_amplitude: ComplexResponse,
    /// Even component divided by A^2. It includes O(A^2) corrections in general.
    pub even_per_amplitude_squared: ComplexResponse,
    pub intervention_amplitude: f64,
}

/// Invalid or unmatchable signed-response inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResponseParityError {
    InvalidAmplitude,
    NonFiniteTime,
    NonFiniteResponse,
    ModeMismatch { expected: i32, actual: i32 },
    TimeMismatch { expected: f64, actual: f64 },
    NonFiniteResult,
}

impl fmt::Display for ResponseParityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAmplitude => write!(
                formatter,
                "positive intervention amplitude must be finite and greater than zero"
            ),
            Self::NonFiniteTime => write!(formatter, "response sample time must be finite"),
            Self::NonFiniteResponse => write!(formatter, "complex response must be finite"),
            Self::ModeMismatch { expected, actual } => write!(
                formatter,
                "response mode {actual} does not match matched-trial mode {expected}"
            ),
            Self::TimeMismatch { expected, actual } => write!(
                formatter,
                "response time {actual} does not match matched-trial time {expected}"
            ),
            Self::NonFiniteResult => {
                write!(formatter, "parity arithmetic produced a non-finite result")
            }
        }
    }
}

impl std::error::Error for ResponseParityError {}

/// Decompose a matched R(-A), R(0), R(+A) triplet into odd and even response.
///
/// All samples must refer to exactly the same signed spatial mode and physical
/// time. Exact time equality is intentional: callers should first align or
/// interpolate traces under a separately declared temporal-alignment policy.
/// The returned normalized components are finite-amplitude estimates, not
/// asymptotic Taylor coefficients unless an amplitude-convergence study shows
/// that higher-order contamination is negligible.
pub fn decompose_signed_response(
    negative: ModeResponseSample,
    zero: ModeResponseSample,
    positive: ModeResponseSample,
    amplitude: f64,
) -> Result<SignedResponseParity, ResponseParityError> {
    if !amplitude.is_finite() || amplitude <= 0.0 {
        return Err(ResponseParityError::InvalidAmplitude);
    }

    validate_sample(negative)?;
    validate_sample(zero)?;
    validate_sample(positive)?;
    validate_match(zero, negative)?;
    validate_match(zero, positive)?;

    let odd_component = positive.response.subtract(negative.response).scale(0.5);
    let even_component = positive
        .response
        .add(negative.response)
        .subtract(zero.response.scale(2.0))
        .scale(0.5);
    let odd_per_amplitude = odd_component.scale(1.0 / amplitude);
    let even_per_amplitude_squared = even_component.scale(1.0 / amplitude.powi(2));

    if !odd_component.is_finite()
        || !even_component.is_finite()
        || !odd_per_amplitude.is_finite()
        || !even_per_amplitude_squared.is_finite()
    {
        return Err(ResponseParityError::NonFiniteResult);
    }

    Ok(SignedResponseParity {
        spatial_mode: zero.spatial_mode,
        time: zero.time,
        odd_component,
        even_component,
        odd_per_amplitude,
        even_per_amplitude_squared,
        intervention_amplitude: amplitude,
    })
}

fn validate_sample(sample: ModeResponseSample) -> Result<(), ResponseParityError> {
    if !sample.time.is_finite() {
        return Err(ResponseParityError::NonFiniteTime);
    }
    if !sample.response.is_finite() {
        return Err(ResponseParityError::NonFiniteResponse);
    }
    Ok(())
}

fn validate_match(
    reference: ModeResponseSample,
    candidate: ModeResponseSample,
) -> Result<(), ResponseParityError> {
    if candidate.spatial_mode != reference.spatial_mode {
        return Err(ResponseParityError::ModeMismatch {
            expected: reference.spatial_mode,
            actual: candidate.spatial_mode,
        });
    }
    if candidate.time != reference.time {
        return Err(ResponseParityError::TimeMismatch {
            expected: reference.time,
            actual: candidate.time,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(mode: i32, time: f64, real: f64, imaginary: f64) -> ModeResponseSample {
        ModeResponseSample {
            spatial_mode: mode,
            time,
            response: ComplexResponse::new(real, imaginary),
        }
    }

    fn close(actual: ComplexResponse, expected: ComplexResponse, tolerance: f64) {
        assert!((actual.real - expected.real).abs() <= tolerance);
        assert!((actual.imaginary - expected.imaginary).abs() <= tolerance);
    }

    #[test]
    fn separates_linear_odd_response_from_matched_zero_baseline() {
        let amplitude = 0.125;
        let baseline = ComplexResponse::new(0.4, -0.2);
        let linear = ComplexResponse::new(1.5, 0.75);
        let negative = baseline.subtract(linear.scale(amplitude));
        let positive = baseline.add(linear.scale(amplitude));
        let result = decompose_signed_response(
            sample(2, 3.5, negative.real, negative.imaginary),
            sample(2, 3.5, baseline.real, baseline.imaginary),
            sample(2, 3.5, positive.real, positive.imaginary),
            amplitude,
        )
        .unwrap();

        close(result.odd_component, linear.scale(amplitude), 1.0e-14);
        close(
            result.even_component,
            ComplexResponse::new(0.0, 0.0),
            1.0e-14,
        );
        close(result.odd_per_amplitude, linear, 1.0e-14);
    }

    #[test]
    fn separates_quadratic_even_component_in_complex_response() {
        let amplitude: f64 = 0.05;
        let baseline = ComplexResponse::new(-0.3, 0.2);
        let first = ComplexResponse::new(0.8, -0.6);
        let second = ComplexResponse::new(0.25, 0.4);
        let make = |signed_amplitude: f64| {
            baseline
                .add(first.scale(signed_amplitude))
                .add(second.scale(signed_amplitude.powi(2)))
        };
        let negative = make(-amplitude);
        let zero = make(0.0);
        let positive = make(amplitude);
        let result = decompose_signed_response(
            sample(1, 7.0, negative.real, negative.imaginary),
            sample(1, 7.0, zero.real, zero.imaginary),
            sample(1, 7.0, positive.real, positive.imaginary),
            amplitude,
        )
        .unwrap();

        close(result.odd_per_amplitude, first, 1.0e-13);
        close(result.even_per_amplitude_squared, second, 1.0e-12);
    }

    #[test]
    fn preserves_complex_phase_information_in_both_parities() {
        let amplitude: f64 = 0.2;
        let odd = ComplexResponse::new(0.0, 0.6);
        let even = ComplexResponse::new(-0.4, 0.0);
        let zero = ComplexResponse::new(0.1, 0.2);
        let positive = zero
            .add(odd.scale(amplitude))
            .add(even.scale(amplitude.powi(2)));
        let negative = zero
            .subtract(odd.scale(amplitude))
            .add(even.scale(amplitude.powi(2)));
        let result = decompose_signed_response(
            sample(3, 1.25, negative.real, negative.imaginary),
            sample(3, 1.25, zero.real, zero.imaginary),
            sample(3, 1.25, positive.real, positive.imaginary),
            amplitude,
        )
        .unwrap();

        close(result.odd_per_amplitude, odd, 1.0e-14);
        close(result.even_per_amplitude_squared, even, 1.0e-14);
        assert!((result.odd_per_amplitude.amplitude() - 0.6).abs() < 1.0e-14);
    }

    #[test]
    fn rejects_mismatched_spatial_modes() {
        let result = decompose_signed_response(
            sample(1, 2.0, 0.0, 0.0),
            sample(1, 2.0, 0.0, 0.0),
            sample(2, 2.0, 0.0, 0.0),
            0.1,
        );
        assert_eq!(
            result,
            Err(ResponseParityError::ModeMismatch {
                expected: 1,
                actual: 2
            })
        );
    }

    #[test]
    fn rejects_mismatched_sample_times_instead_of_silently_aligning() {
        let result = decompose_signed_response(
            sample(1, 2.0, 0.0, 0.0),
            sample(1, 2.0, 0.0, 0.0),
            sample(1, 2.01, 0.0, 0.0),
            0.1,
        );
        assert!(matches!(
            result,
            Err(ResponseParityError::TimeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_nonpositive_or_nonfinite_amplitude() {
        for amplitude in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            assert_eq!(
                decompose_signed_response(
                    sample(1, 0.0, 0.0, 0.0),
                    sample(1, 0.0, 0.0, 0.0),
                    sample(1, 0.0, 0.0, 0.0),
                    amplitude,
                ),
                Err(ResponseParityError::InvalidAmplitude)
            );
        }
    }

    #[test]
    fn rejects_nonfinite_inputs_and_overflowed_results() {
        assert_eq!(
            decompose_signed_response(
                sample(1, 0.0, f64::NAN, 0.0),
                sample(1, 0.0, 0.0, 0.0),
                sample(1, 0.0, 0.0, 0.0),
                0.1,
            ),
            Err(ResponseParityError::NonFiniteResponse)
        );
        assert_eq!(
            decompose_signed_response(
                sample(1, 0.0, -f64::MAX, 0.0),
                sample(1, 0.0, 0.0, 0.0),
                sample(1, 0.0, f64::MAX, 0.0),
                0.1,
            ),
            Err(ResponseParityError::NonFiniteResult)
        );
    }

    #[test]
    fn finite_amplitude_estimates_expose_higher_order_contamination() {
        let amplitude: f64 = 0.1;
        let baseline = ComplexResponse::new(0.2, -0.1);
        let first = ComplexResponse::new(0.7, 0.3);
        let second = ComplexResponse::new(-0.2, 0.4);
        let third = ComplexResponse::new(0.5, -0.6);
        let fourth = ComplexResponse::new(-0.3, 0.2);
        let make = |signed_amplitude: f64| {
            baseline
                .add(first.scale(signed_amplitude))
                .add(second.scale(signed_amplitude.powi(2)))
                .add(third.scale(signed_amplitude.powi(3)))
                .add(fourth.scale(signed_amplitude.powi(4)))
        };
        let negative = make(-amplitude);
        let zero = make(0.0);
        let positive = make(amplitude);
        let result = decompose_signed_response(
            sample(4, 9.0, negative.real, negative.imaginary),
            sample(4, 9.0, zero.real, zero.imaginary),
            sample(4, 9.0, positive.real, positive.imaginary),
            amplitude,
        )
        .unwrap();

        let amplitude_squared = amplitude.powi(2);
        close(
            result.odd_per_amplitude,
            first.add(third.scale(amplitude_squared)),
            1.0e-13,
        );
        close(
            result.even_per_amplitude_squared,
            second.add(fourth.scale(amplitude_squared)),
            1.0e-12,
        );
    }
}
