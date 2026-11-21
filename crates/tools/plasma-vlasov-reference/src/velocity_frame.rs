//! Exact low-order moment transformations for affine velocity coordinates.
//!
//! For a physical distribution f(v), define the computational-coordinate
//! density g(xi) = scale * f(shift + scale * xi), with scale > 0. This
//! convention makes the zeroth moment invariant under the coordinate map.
//!
//! The transformations below are exact algebraic identities for moments; they
//! do not remap sampled distribution values, project Hermite coefficients,
//! preserve positivity, or implement an adaptive Hermite solver.

/// Affine velocity coordinate v = shift + scale * xi.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityFrame {
    /// Physical velocity corresponding to xi = 0.
    pub shift: f64,
    /// Positive physical velocity per unit computational coordinate.
    pub scale: f64,
}

impl VelocityFrame {
    /// Validate the affine coordinate frame.
    pub fn validate(self) -> Result<(), VelocityFrameError> {
        if !self.shift.is_finite() {
            return Err(VelocityFrameError::InvalidShift);
        }
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(VelocityFrameError::InvalidScale);
        }
        Ok(())
    }

    /// Return exact first derivatives of the physical raw moments.
    ///
    /// The Jacobian is evaluated for fixed source computational moments:
    /// - dM/dmu is a 3x3 matrix;
    /// - dM/du is the derivative with respect to the frame shift;
    /// - dM/dalpha is the derivative with respect to the positive frame scale.
    ///
    /// This is an analytic sensitivity for the affine moment map only. It does
    /// not differentiate a sampled-distribution projection, a time integrator,
    /// or a discrete representation-selection decision.
    pub fn moment_jacobian(
        self,
        moments: FrameVelocityMoments,
    ) -> Result<AffineMomentJacobian, VelocityFrameError> {
        self.validate()?;
        validate_moments(moments.zeroth, moments.first, moments.second)?;

        let jacobian = AffineMomentJacobian {
            with_respect_to_frame_moments: [
                [1.0, 0.0, 0.0],
                [self.shift, self.scale, 0.0],
                [
                    self.shift * self.shift,
                    2.0 * self.shift * self.scale,
                    self.scale * self.scale,
                ],
            ],
            with_respect_to_shift: [
                0.0,
                moments.zeroth,
                2.0 * self.shift * moments.zeroth + 2.0 * self.scale * moments.first,
            ],
            with_respect_to_scale: [
                0.0,
                moments.first,
                2.0 * self.shift * moments.first + 2.0 * self.scale * moments.second,
            ],
        };
        jacobian.validate()?;
        Ok(jacobian)
    }

    /// Convert moments of g(xi) = scale * f(shift + scale * xi) into
    /// physical-velocity moments of f(v).
    ///
    /// For mu_j = integral xi^j g(xi) dxi and M_j = integral v^j f(v) dv:
    /// M0 = mu0
    /// M1 = shift * mu0 + scale * mu1
    /// M2 = shift^2 * mu0 + 2 * shift * scale * mu1 + scale^2 * mu2
    ///
    /// The second moment is not multiplied by particle mass or one half; those
    /// factors belong to the physical model's kinetic-energy convention.
    pub fn frame_to_physical(
        self,
        moments: FrameVelocityMoments,
    ) -> Result<PhysicalVelocityMoments, VelocityFrameError> {
        self.validate()?;
        validate_moments(moments.zeroth, moments.first, moments.second)?;

        let physical = PhysicalVelocityMoments {
            zeroth: moments.zeroth,
            first: self.shift * moments.zeroth + self.scale * moments.first,
            second: self.shift * self.shift * moments.zeroth
                + 2.0 * self.shift * self.scale * moments.first
                + self.scale * self.scale * moments.second,
        };
        validate_moments(physical.zeroth, physical.first, physical.second)?;
        Ok(physical)
    }

    /// Convert physical-velocity moments into moments of
    /// g(xi) = scale * f(shift + scale * xi).
    ///
    /// This is the algebraic inverse of frame_to_physical.
    pub fn physical_to_frame(
        self,
        moments: PhysicalVelocityMoments,
    ) -> Result<FrameVelocityMoments, VelocityFrameError> {
        self.validate()?;
        validate_moments(moments.zeroth, moments.first, moments.second)?;

        let scale_squared = self.scale * self.scale;
        let frame = FrameVelocityMoments {
            zeroth: moments.zeroth,
            first: (moments.first - self.shift * moments.zeroth) / self.scale,
            second: (moments.second - 2.0 * self.shift * moments.first
                + self.shift * self.shift * moments.zeroth)
                / scale_squared,
        };
        validate_moments(frame.zeroth, frame.first, frame.second)?;
        Ok(frame)
    }
}

/// First three raw moments in computational coordinate xi.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameVelocityMoments {
    /// Integral of g(xi).
    pub zeroth: f64,
    /// Integral of xi * g(xi).
    pub first: f64,
    /// Integral of xi^2 * g(xi).
    pub second: f64,
}

/// First three raw moments in physical velocity v.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicalVelocityMoments {
    /// Integral of f(v), i.e. number density under the declared normalization.
    pub zeroth: f64,
    /// Integral of v * f(v), i.e. momentum density divided by particle mass.
    pub first: f64,
    /// Integral of v^2 * f(v), before the model's mass and one-half factors.
    pub second: f64,
}

/// Exact Jacobian of physical moments with respect to source moments and frame parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineMomentJacobian {
    /// Rows are physical moments M0..M2; columns are computational moments mu0..mu2.
    pub with_respect_to_frame_moments: [[f64; 3]; 3],
    /// Derivatives of M0..M2 with respect to the frame shift.
    pub with_respect_to_shift: [f64; 3],
    /// Derivatives of M0..M2 with respect to the positive frame scale.
    pub with_respect_to_scale: [f64; 3],
}

impl AffineMomentJacobian {
    fn validate(self) -> Result<(), VelocityFrameError> {
        if self
            .with_respect_to_frame_moments
            .iter()
            .flatten()
            .chain(self.with_respect_to_shift.iter())
            .chain(self.with_respect_to_scale.iter())
            .any(|value| !value.is_finite())
        {
            return Err(VelocityFrameError::NonFiniteJacobian);
        }
        Ok(())
    }
}

/// Invalid affine-frame or moment input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VelocityFrameError {
    InvalidShift,
    InvalidScale,
    NonFiniteZerothMoment,
    NonFiniteFirstMoment,
    NonFiniteSecondMoment,
    NonFiniteJacobian,
}

fn validate_moments(zeroth: f64, first: f64, second: f64) -> Result<(), VelocityFrameError> {
    if !zeroth.is_finite() {
        return Err(VelocityFrameError::NonFiniteZerothMoment);
    }
    if !first.is_finite() {
        return Err(VelocityFrameError::NonFiniteFirstMoment);
    }
    if !second.is_finite() {
        return Err(VelocityFrameError::NonFiniteSecondMoment);
    }
    Ok(())
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

    #[test]
    fn affine_moment_jacobian_matches_analytic_known_answer() {
        let frame = VelocityFrame {
            shift: 2.0,
            scale: 0.5,
        };
        let moments = FrameVelocityMoments {
            zeroth: 3.0,
            first: 4.0,
            second: 5.0,
        };
        let jacobian = frame.moment_jacobian(moments).unwrap();

        assert_eq!(
            jacobian.with_respect_to_frame_moments,
            [[1.0, 0.0, 0.0], [2.0, 0.5, 0.0], [4.0, 2.0, 0.25],]
        );
        assert_eq!(jacobian.with_respect_to_shift, [0.0, 3.0, 16.0]);
        assert_eq!(jacobian.with_respect_to_scale, [0.0, 4.0, 21.0]);
    }

    #[test]
    fn affine_shift_and_scale_jacobians_match_central_finite_differences() {
        let frame = VelocityFrame {
            shift: -0.7,
            scale: 1.4,
        };
        let moments = FrameVelocityMoments {
            zeroth: 1.3,
            first: -0.2,
            second: 0.9,
        };
        let analytic = frame.moment_jacobian(moments).unwrap();
        let epsilon = 1.0e-6;

        let plus_shift = VelocityFrame {
            shift: frame.shift + epsilon,
            ..frame
        }
        .frame_to_physical(moments)
        .unwrap();
        let minus_shift = VelocityFrame {
            shift: frame.shift - epsilon,
            ..frame
        }
        .frame_to_physical(moments)
        .unwrap();
        let shift_fd = [
            (plus_shift.zeroth - minus_shift.zeroth) / (2.0 * epsilon),
            (plus_shift.first - minus_shift.first) / (2.0 * epsilon),
            (plus_shift.second - minus_shift.second) / (2.0 * epsilon),
        ];

        let plus_scale = VelocityFrame {
            scale: frame.scale + epsilon,
            ..frame
        }
        .frame_to_physical(moments)
        .unwrap();
        let minus_scale = VelocityFrame {
            scale: frame.scale - epsilon,
            ..frame
        }
        .frame_to_physical(moments)
        .unwrap();
        let scale_fd = [
            (plus_scale.zeroth - minus_scale.zeroth) / (2.0 * epsilon),
            (plus_scale.first - minus_scale.first) / (2.0 * epsilon),
            (plus_scale.second - minus_scale.second) / (2.0 * epsilon),
        ];

        for index in 0..3 {
            close(
                shift_fd[index],
                analytic.with_respect_to_shift[index],
                2.0e-9,
            );
            close(
                scale_fd[index],
                analytic.with_respect_to_scale[index],
                2.0e-9,
            );
        }
    }

    #[test]
    fn affine_moment_jacobian_rejects_non_finite_derivatives() {
        let frame = VelocityFrame {
            shift: 1.0e200,
            scale: 1.0,
        };
        let moments = FrameVelocityMoments {
            zeroth: 1.0e200,
            first: 0.0,
            second: 1.0,
        };
        assert_eq!(
            frame.moment_jacobian(moments),
            Err(VelocityFrameError::NonFiniteJacobian)
        );
    }

    #[test]
    fn shift_and_scale_map_known_frame_moments_exactly() {
        let frame = VelocityFrame {
            shift: 4.0,
            scale: 2.0,
        };
        let computational = FrameVelocityMoments {
            zeroth: 2.0,
            first: 0.5,
            second: 3.0,
        };

        let physical = frame.frame_to_physical(computational).unwrap();
        close(physical.zeroth, 2.0, 0.0);
        close(physical.first, 9.0, 0.0);
        close(physical.second, 52.0, 0.0);
    }

    #[test]
    fn physical_to_frame_is_the_inverse_of_frame_to_physical() {
        let frame = VelocityFrame {
            shift: -3.25,
            scale: 1.75,
        };
        let computational = FrameVelocityMoments {
            zeroth: 1.2,
            first: -0.3,
            second: 2.4,
        };

        let physical = frame.frame_to_physical(computational).unwrap();
        let recovered = frame.physical_to_frame(physical).unwrap();
        close(recovered.zeroth, computational.zeroth, 1.0e-14);
        close(recovered.first, computational.first, 1.0e-14);
        close(recovered.second, computational.second, 1.0e-13);
    }

    #[test]
    fn changing_frames_preserves_the_physical_moments() {
        let source = VelocityFrame {
            shift: 2.0,
            scale: 0.75,
        };
        let destination = VelocityFrame {
            shift: -1.5,
            scale: 2.25,
        };
        let source_moments = FrameVelocityMoments {
            zeroth: 0.8,
            first: 0.12,
            second: 0.64,
        };

        let physical = source.frame_to_physical(source_moments).unwrap();
        let destination_moments = destination.physical_to_frame(physical).unwrap();
        let reconstructed = destination.frame_to_physical(destination_moments).unwrap();

        close(reconstructed.zeroth, physical.zeroth, 1.0e-14);
        close(reconstructed.first, physical.first, 1.0e-14);
        close(reconstructed.second, physical.second, 1.0e-13);
    }

    #[test]
    fn zeroth_moment_is_invariant_under_affine_velocity_coordinates() {
        let frame = VelocityFrame {
            shift: 17.0,
            scale: 0.125,
        };
        let computational = FrameVelocityMoments {
            zeroth: 3.5,
            first: -0.2,
            second: 0.9,
        };
        let physical = frame.frame_to_physical(computational).unwrap();
        assert_eq!(physical.zeroth, computational.zeroth);
    }

    #[test]
    fn signed_perturbation_moments_are_supported_without_positivity_projection() {
        let frame = VelocityFrame {
            shift: 0.4,
            scale: 1.3,
        };
        let signed = FrameVelocityMoments {
            zeroth: 0.0,
            first: -0.25,
            second: -0.04,
        };
        let physical = frame.frame_to_physical(signed).unwrap();
        let recovered = frame.physical_to_frame(physical).unwrap();

        close(recovered.zeroth, signed.zeroth, 0.0);
        close(recovered.first, signed.first, 1.0e-14);
        close(recovered.second, signed.second, 1.0e-14);
    }

    #[test]
    fn non_positive_or_non_finite_scales_are_rejected() {
        for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let frame = VelocityFrame { shift: 0.0, scale };
            assert_eq!(frame.validate(), Err(VelocityFrameError::InvalidScale));
        }
    }

    #[test]
    fn non_finite_shifts_are_rejected() {
        for shift in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let frame = VelocityFrame { shift, scale: 1.0 };
            assert_eq!(frame.validate(), Err(VelocityFrameError::InvalidShift));
        }
    }

    #[test]
    fn non_finite_moments_are_rejected() {
        let frame = VelocityFrame {
            shift: 0.0,
            scale: 1.0,
        };
        let bad_zeroth = FrameVelocityMoments {
            zeroth: f64::NAN,
            first: 0.0,
            second: 0.0,
        };
        assert_eq!(
            frame.frame_to_physical(bad_zeroth),
            Err(VelocityFrameError::NonFiniteZerothMoment)
        );

        let bad_first = PhysicalVelocityMoments {
            zeroth: 1.0,
            first: f64::INFINITY,
            second: 0.0,
        };
        assert_eq!(
            frame.physical_to_frame(bad_first),
            Err(VelocityFrameError::NonFiniteFirstMoment)
        );
    }

    #[test]
    fn overflow_during_transform_is_rejected() {
        let frame = VelocityFrame {
            shift: 1.0e200,
            scale: 1.0,
        };
        let moments = FrameVelocityMoments {
            zeroth: 1.0e-50,
            first: 0.0,
            second: 1.0,
        };
        assert_eq!(
            frame.frame_to_physical(moments),
            Err(VelocityFrameError::NonFiniteSecondMoment)
        );
    }
}
