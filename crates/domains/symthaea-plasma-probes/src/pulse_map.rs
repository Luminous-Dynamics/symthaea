// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Discrete canonical map and tangent contract for the AKPD sinusoidal velocity kick.
//!
//! This implements the exact v1 single-particle phase-space map from the
//! versioned AKPD pulse-family contract. It supplies state and parameter
//! Jacobians for a future solver tangent/adjoint adapter. It does not remap a
//! sampled distribution, solve Vlasov–Poisson, integrate between interventions,
//! or establish physical response convergence.

use thiserror::Error;

/// Errors from pulse-map evaluation and derivative checking.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PulseMapError {
    #[error("pulse geometry or control parameter is invalid")]
    InvalidPulseParameters,
    #[error("particle state contains a non-finite coordinate")]
    InvalidParticleState,
    #[error("finite-difference step must be finite and strictly positive")]
    InvalidDifferenceStep,
    #[error("pulse-map derivative check produced a non-finite result")]
    NonFiniteDerivative,
}

/// Fixed geometry and controls for one instantaneous sinusoidal velocity kick.
///
/// The displacement is delta_v(x) = A * sin(2*pi*m*(x-x_min)/L + phase).
/// Mode is a positive integer because this is a periodic Fourier mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SinusoidalVelocityKick {
    pub x_min: f64,
    pub domain_length: f64,
    pub mode: u32,
    pub amplitude: f64,
    pub phase_rad: f64,
}

impl SinusoidalVelocityKick {
    pub fn new(
        x_min: f64,
        domain_length: f64,
        mode: u32,
        amplitude: f64,
        phase_rad: f64,
    ) -> Result<Self, PulseMapError> {
        let kick = Self {
            x_min,
            domain_length,
            mode,
            amplitude,
            phase_rad,
        };
        kick.validate()?;
        Ok(kick)
    }

    pub fn validate(&self) -> Result<(), PulseMapError> {
        if !self.x_min.is_finite()
            || !self.domain_length.is_finite()
            || self.domain_length <= 0.0
            || self.mode == 0
            || !self.amplitude.is_finite()
            || self.amplitude < 0.0
            || !self.phase_rad.is_finite()
        {
            return Err(PulseMapError::InvalidPulseParameters);
        }
        Ok(())
    }

    /// Return the periodic spatial angle for a particle position.
    pub fn angle(&self, x: f64) -> Result<f64, PulseMapError> {
        self.validate()?;
        if !x.is_finite() {
            return Err(PulseMapError::InvalidParticleState);
        }
        let cycles = (x - self.x_min) / self.domain_length;
        let angle = std::f64::consts::TAU * f64::from(self.mode) * cycles + self.phase_rad;
        if angle.is_finite() {
            Ok(angle)
        } else {
            Err(PulseMapError::NonFiniteDerivative)
        }
    }

    /// Evaluate the instantaneous velocity increment.
    pub fn delta_v(&self, x: f64) -> Result<f64, PulseMapError> {
        let value = self.amplitude * self.angle(x)?.sin();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(PulseMapError::NonFiniteDerivative)
        }
    }

    /// Apply the canonical kick to one phase-space point.
    ///
    /// Position is unchanged and velocity receives the declared impulse.
    pub fn apply(&self, position: f64, velocity: f64) -> Result<PhasePoint, PulseMapError> {
        if !velocity.is_finite() {
            return Err(PulseMapError::InvalidParticleState);
        }
        let kicked = velocity + self.delta_v(position)?;
        if !kicked.is_finite() {
            return Err(PulseMapError::NonFiniteDerivative);
        }
        Ok(PhasePoint {
            x: position,
            v: kicked,
        })
    }

    /// Exact Jacobian d(x+, v+)/d(x, v) of the discrete kick map.
    ///
    /// A_state = [[1, 0], [A*k*cos(theta), 1]], k = 2*pi*m/L.
    /// Its determinant is one and it preserves the canonical symplectic form.
    pub fn state_jacobian(&self, position: f64) -> Result<Matrix2, PulseMapError> {
        let angle = self.angle(position)?;
        let wave_number =
            std::f64::consts::TAU * f64::from(self.mode) / self.domain_length;
        let shear = self.amplitude * wave_number * angle.cos();
        if !shear.is_finite() {
            return Err(PulseMapError::NonFiniteDerivative);
        }
        Ok(Matrix2 {
            m11: 1.0,
            m12: 0.0,
            m21: shear,
            m22: 1.0,
        })
    }

    /// Exact Jacobian d(x+, v+)/d(A, phase).
    ///
    /// The first row is zero. The second row is [sin(theta), A*cos(theta)].
    pub fn parameter_jacobian(&self, position: f64) -> Result<ParameterJacobian, PulseMapError> {
        let angle = self.angle(position)?;
        let derivative = ParameterJacobian {
            dx_d_amplitude: 0.0,
            dv_d_amplitude: angle.sin(),
            dx_d_phase: 0.0,
            dv_d_phase: self.amplitude * angle.cos(),
        };
        if [
            derivative.dx_d_amplitude,
            derivative.dv_d_amplitude,
            derivative.dx_d_phase,
            derivative.dv_d_phase,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return Err(PulseMapError::NonFiniteDerivative);
        }
        Ok(derivative)
    }

    /// Signed kinetic-energy change per unit particle mass across the kick.
    ///
    /// This is an external intervention. Kinetic energy and momentum are not
    /// invariants across the kick; record this change rather than enforcing
    /// conservation through the pulse.
    pub fn kinetic_energy_delta(
        &self,
        position: f64,
        velocity: f64,
    ) -> Result<f64, PulseMapError> {
        if !velocity.is_finite() {
            return Err(PulseMapError::InvalidParticleState);
        }
        let kicked = self.apply(position, velocity)?.v;
        let delta = 0.5 * (kicked * kicked - velocity * velocity);
        if delta.is_finite() {
            Ok(delta)
        } else {
            Err(PulseMapError::NonFiniteDerivative)
        }
    }
}

/// One canonical phase-space point (x, v).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhasePoint {
    pub x: f64,
    pub v: f64,
}

/// Row-major 2x2 state Jacobian.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matrix2 {
    pub m11: f64,
    pub m12: f64,
    pub m21: f64,
    pub m22: f64,
}

impl Matrix2 {
    pub fn determinant(self) -> f64 {
        self.m11 * self.m22 - self.m12 * self.m21
    }

    /// Canonical 2D symplectic defect: Frobenius norm of A^T J A - J,
    /// with J = [[0,1],[-1,0]]. In 2D this equals sqrt(2)*|det(A)-1|.
    pub fn symplectic_defect(self) -> f64 {
        std::f64::consts::SQRT_2 * (self.determinant() - 1.0).abs()
    }
}

/// Parameter derivatives of the post-kick point with respect to amplitude and phase.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParameterJacobian {
    pub dx_d_amplitude: f64,
    pub dv_d_amplitude: f64,
    pub dx_d_phase: f64,
    pub dv_d_phase: f64,
}

/// Maximum absolute errors in finite-difference derivative cross-checks.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PulseJacobianCheck {
    pub state_jacobian_max_abs_error: f64,
    pub parameter_jacobian_max_abs_error: f64,
    pub symplectic_defect: f64,
}

/// Cross-check analytic state and parameter derivatives using central
/// finite differences (one-sided in amplitude only at the admissible A=0 face).
///
/// This is a manufactured check. A production adapter must additionally
/// compare the derivative against the exact discrete split-integrator map and
/// its independently implemented tangent/adjoint.
pub fn check_pulse_jacobians_central_difference(
    kick: &SinusoidalVelocityKick,
    position: f64,
    velocity: f64,
    difference_step: f64,
) -> Result<PulseJacobianCheck, PulseMapError> {
    kick.validate()?;
    if !position.is_finite() || !velocity.is_finite() {
        return Err(PulseMapError::InvalidParticleState);
    }
    if !difference_step.is_finite() || difference_step <= 0.0 {
        return Err(PulseMapError::InvalidDifferenceStep);
    }

    let analytic_state = kick.state_jacobian(position)?;
    let analytic_parameters = kick.parameter_jacobian(position)?;

    let x_plus = kick.apply(position + difference_step, velocity)?;
    let x_minus = kick.apply(position - difference_step, velocity)?;
    let v_plus = kick.apply(position, velocity + difference_step)?;
    let v_minus = kick.apply(position, velocity - difference_step)?;
    let fd_xx = (x_plus.x - x_minus.x) / (2.0 * difference_step);
    let fd_vx = (x_plus.v - x_minus.v) / (2.0 * difference_step);
    let fd_xv = (v_plus.x - v_minus.x) / (2.0 * difference_step);
    let fd_vv = (v_plus.v - v_minus.v) / (2.0 * difference_step);

    let amp_plus = SinusoidalVelocityKick {
        amplitude: kick.amplitude + difference_step,
        ..*kick
    };
    let amp_minus = SinusoidalVelocityKick {
        amplitude: (kick.amplitude - difference_step).max(0.0),
        ..*kick
    };
    let phase_plus = SinusoidalVelocityKick {
        phase_rad: kick.phase_rad + difference_step,
        ..*kick
    };
    let phase_minus = SinusoidalVelocityKick {
        phase_rad: kick.phase_rad - difference_step,
        ..*kick
    };
    let fd_v_amplitude = if kick.amplitude >= difference_step {
        (amp_plus.apply(position, velocity)?.v - amp_minus.apply(position, velocity)?.v)
            / (2.0 * difference_step)
    } else {
        (amp_plus.apply(position, velocity)?.v - kick.apply(position, velocity)?.v)
            / difference_step
    };
    let fd_v_phase = (phase_plus.apply(position, velocity)?.v
        - phase_minus.apply(position, velocity)?.v)
        / (2.0 * difference_step);

    let state_errors = [
        (fd_xx - analytic_state.m11).abs(),
        (fd_xv - analytic_state.m12).abs(),
        (fd_vx - analytic_state.m21).abs(),
        (fd_vv - analytic_state.m22).abs(),
    ];
    let parameter_errors = [
        analytic_parameters.dx_d_amplitude.abs(),
        (fd_v_amplitude - analytic_parameters.dv_d_amplitude).abs(),
        analytic_parameters.dx_d_phase.abs(),
        (fd_v_phase - analytic_parameters.dv_d_phase).abs(),
    ];
    let state_jacobian_max_abs_error = state_errors.into_iter().fold(0.0_f64, f64::max);
    let parameter_jacobian_max_abs_error =
        parameter_errors.into_iter().fold(0.0_f64, f64::max);
    let symplectic_defect = analytic_state.symplectic_defect();

    if !state_jacobian_max_abs_error.is_finite()
        || !parameter_jacobian_max_abs_error.is_finite()
        || !symplectic_defect.is_finite()
    {
        return Err(PulseMapError::NonFiniteDerivative);
    }
    Ok(PulseJacobianCheck {
        state_jacobian_max_abs_error,
        parameter_jacobian_max_abs_error,
        symplectic_defect,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manufactured_kick() -> SinusoidalVelocityKick {
        SinusoidalVelocityKick::new(
            -std::f64::consts::PI,
            2.0 * std::f64::consts::PI,
            2,
            0.012,
            0.37,
        )
        .unwrap()
    }

    #[test]
    fn canonical_velocity_kick_matches_registered_operator() {
        let kick = manufactured_kick();
        let x = -0.41;
        let v = 0.73;
        let theta = std::f64::consts::TAU * 2.0 * (x + std::f64::consts::PI)
            / (2.0 * std::f64::consts::PI)
            + 0.37;
        let mapped = kick.apply(x, v).unwrap();
        assert_eq!(mapped.x, x);
        assert!((mapped.v - (v + 0.012 * theta.sin())).abs() < 1.0e-14);
    }

    #[test]
    fn exact_state_jacobian_is_symplectic_and_unit_determinant() {
        let kick = manufactured_kick();
        for x in [
            -std::f64::consts::PI,
            -1.2,
            0.0,
            2.1,
            std::f64::consts::PI,
        ] {
            let matrix = kick.state_jacobian(x).unwrap();
            assert!((matrix.determinant() - 1.0).abs() < 1.0e-14);
            assert!(matrix.symplectic_defect() < 1.0e-14);
            assert_eq!(matrix.m12, 0.0);
            assert_eq!(matrix.m11, 1.0);
            assert_eq!(matrix.m22, 1.0);
        }
    }

    #[test]
    fn exact_parameter_jacobian_matches_amplitude_and_phase_derivatives() {
        let kick = manufactured_kick();
        let x = 0.58;
        let jacobian = kick.parameter_jacobian(x).unwrap();
        let theta = kick.angle(x).unwrap();
        assert_eq!(jacobian.dx_d_amplitude, 0.0);
        assert!((jacobian.dv_d_amplitude - theta.sin()).abs() < 1.0e-14);
        assert_eq!(jacobian.dx_d_phase, 0.0);
        assert!((jacobian.dv_d_phase - kick.amplitude * theta.cos()).abs() < 1.0e-14);
    }

    #[test]
    fn central_difference_recovers_state_and_parameter_jacobians() {
        let kick = manufactured_kick();
        let result = check_pulse_jacobians_central_difference(&kick, 0.23, -0.4, 1.0e-5).unwrap();
        assert!(result.state_jacobian_max_abs_error < 1.0e-8);
        assert!(result.parameter_jacobian_max_abs_error < 1.0e-8);
        assert!(result.symplectic_defect < 1.0e-14);
    }

    #[test]
    fn periodic_domain_translation_preserves_kick_and_derivatives() {
        let kick = manufactured_kick();
        let base = kick.apply(-0.2, 0.5).unwrap();
        let translated = kick.apply(-0.2 + kick.domain_length, 0.5).unwrap();
        assert!((base.v - translated.v).abs() < 1.0e-14);
        let first = kick.state_jacobian(-0.2).unwrap();
        let second = kick.state_jacobian(-0.2 + kick.domain_length).unwrap();
        assert!((first.m21 - second.m21).abs() < 1.0e-13);
    }

    #[test]
    fn energy_exchange_is_reported_not_constrained_away() {
        let kick = manufactured_kick();
        let delta = kick.kinetic_energy_delta(0.7, 1.2).unwrap();
        assert!(delta.is_finite());
        let reverse = SinusoidalVelocityKick::new(
            kick.x_min,
            kick.domain_length,
            kick.mode,
            kick.amplitude,
            kick.phase_rad + std::f64::consts::PI,
        )
        .unwrap()
        .kinetic_energy_delta(0.7, 1.2)
        .unwrap();
        assert!(delta * reverse <= 0.0);
    }

    #[test]
    fn amplitude_zero_uses_a_valid_forward_difference_at_the_admissible_boundary() {
        let kick = SinusoidalVelocityKick::new(
            -std::f64::consts::PI,
            2.0 * std::f64::consts::PI,
            2,
            0.0,
            0.37,
        )
        .unwrap();
        let check = check_pulse_jacobians_central_difference(&kick, 0.23, -0.4, 1.0e-6).unwrap();
        assert!(check.state_jacobian_max_abs_error < 1.0e-10);
        assert!(check.parameter_jacobian_max_abs_error < 1.0e-6);
        assert!(check.symplectic_defect < 1.0e-14);
    }

    #[test]
    fn rejects_invalid_geometry_controls_and_non_finite_points() {
        assert_eq!(
            SinusoidalVelocityKick::new(0.0, 0.0, 1, 0.1, 0.0),
            Err(PulseMapError::InvalidPulseParameters)
        );
        assert_eq!(
            SinusoidalVelocityKick::new(0.0, 1.0, 0, 0.1, 0.0),
            Err(PulseMapError::InvalidPulseParameters)
        );
        assert_eq!(
            SinusoidalVelocityKick::new(0.0, 1.0, 1, -0.1, 0.0),
            Err(PulseMapError::InvalidPulseParameters)
        );
        let kick = manufactured_kick();
        assert_eq!(
            kick.apply(f64::NAN, 0.0),
            Err(PulseMapError::InvalidParticleState)
        );
        assert_eq!(
            kick.apply(0.0, f64::INFINITY),
            Err(PulseMapError::InvalidParticleState)
        );
    }
}
