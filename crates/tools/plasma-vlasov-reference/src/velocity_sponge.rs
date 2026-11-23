//! Explicit, accounted high-|s| damping in velocity-Fourier space.
//!
//! This is a controlled spectral sponge experiment, not a characteristic
//! outgoing-wave boundary condition. It deliberately removes fine velocity
//! information and reports the resulting quadratic Casimir / f-strophy loss.
//! It must be compared against the collisionless periodic-velocity baseline
//! as a distinct numerical model.

use std::fmt;

use super::phase_space_cascade::QuadraticCasimirBudget;
use super::{Complex, SolverError, VlasovPoisson1d1v, VlasovPoissonDiagnostics, fft, signed_mode};

/// Invalid parameters or solver failures during a velocity-Fourier sponge.
#[derive(Debug, Clone, PartialEq)]
pub enum VelocityFourierSpongeError {
    InvalidDuration,
    InvalidStartFraction,
    InvalidMaximumDamping,
    InvalidExponent,
    NonFiniteTransform,
    Solver(SolverError),
}

impl fmt::Display for VelocityFourierSpongeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDuration => {
                write!(formatter, "sponge duration must be finite and nonnegative")
            }
            Self::InvalidStartFraction => {
                write!(formatter, "sponge start fraction must be in [0, 1)")
            }
            Self::InvalidMaximumDamping => {
                write!(
                    formatter,
                    "maximum damping rate must be finite and nonnegative"
                )
            }
            Self::InvalidExponent => {
                write!(formatter, "sponge exponent must be in 1..=16")
            }
            Self::NonFiniteTransform => {
                write!(
                    formatter,
                    "velocity-Fourier sponge produced a non-finite transform"
                )
            }
            Self::Solver(error) => write!(formatter, "solver failure during sponge: {error}"),
        }
    }
}

impl std::error::Error for VelocityFourierSpongeError {}

/// Explicit loss accounting for one high-|s| velocity-Fourier sponge step.
///
/// The sponge does not advance the solver clock. Duration is the operator
/// interval used to compute exponential attenuation and is intended to be
/// supplied by an outer split integrator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityFourierSpongeReport {
    pub duration: f64,
    pub start_fraction_of_nyquist: f64,
    pub maximum_damping_rate: f64,
    pub exponent: u32,
    pub nyquist_attenuation: f64,
    pub casimir_before: QuadraticCasimirBudget,
    pub casimir_after: QuadraticCasimirBudget,
    pub diagnostics_before: VlasovPoissonDiagnostics,
    pub diagnostics_after: VlasovPoissonDiagnostics,
    /// Positive is removed C2; small negative roundoff is not clamped.
    pub removed_total_casimir: f64,
    pub removed_spatial_mean_casimir: f64,
    pub removed_fluctuation_f_strophy: f64,
    /// Independent Parseval sum of damped velocity-Fourier modal energy.
    pub spectral_removed_casimir: f64,
    /// Maximum imaginary residual after inverse transforming real input.
    pub max_imaginary_residual: f64,
    /// Maximum absolute real value in the inverse-transformed candidate.
    pub max_real_magnitude: f64,
    /// (C2 before - C2 after) - spectral modal loss.
    pub casimir_accounting_residual: f64,
}

impl VlasovPoisson1d1v {
    /// Apply an explicit exponential high-|s| sponge to every spatial row.
    ///
    /// For signed velocity-Fourier mode s and normalized mode fraction
    /// r = |s| / s_Nyquist, modes with r <= start_fraction are unchanged.
    /// Above the onset, the amplitude multiplier is
    ///
    ///   exp(-maximum_damping_rate * duration
    ///       * ((r-start)/(1-start))^exponent)
    ///
    /// The transform is periodic on the configured finite velocity interval.
    /// This operation is dissipative, does not advance self.time, and is not
    /// part of step(dt). The caller must explicitly schedule and account for
    /// it. The zero-velocity-Fourier mode is untouched, preserving the
    /// discrete spatial density and therefore the Poisson source up to FFT
    /// roundoff.
    pub fn apply_velocity_fourier_sponge(
        &mut self,
        duration: f64,
        start_fraction_of_nyquist: f64,
        maximum_damping_rate: f64,
        exponent: u32,
    ) -> Result<VelocityFourierSpongeReport, VelocityFourierSpongeError> {
        if !duration.is_finite() || duration < 0.0 {
            return Err(VelocityFourierSpongeError::InvalidDuration);
        }
        if !start_fraction_of_nyquist.is_finite()
            || !(0.0..1.0).contains(&start_fraction_of_nyquist)
        {
            return Err(VelocityFourierSpongeError::InvalidStartFraction);
        }
        if !maximum_damping_rate.is_finite() || maximum_damping_rate < 0.0 {
            return Err(VelocityFourierSpongeError::InvalidMaximumDamping);
        }
        if !(1..=16).contains(&exponent) {
            return Err(VelocityFourierSpongeError::InvalidExponent);
        }

        let before = self
            .quadratic_casimir_budget()
            .map_err(VelocityFourierSpongeError::Solver)?;
        let diagnostics_before = self
            .diagnostics()
            .map_err(VelocityFourierSpongeError::Solver)?;

        // Preserve exact identity semantics and avoid an unnecessary FFT
        // round-trip when the requested operator has no effect.
        if duration == 0.0 || maximum_damping_rate == 0.0 {
            return Ok(VelocityFourierSpongeReport {
                duration,
                start_fraction_of_nyquist,
                maximum_damping_rate,
                exponent,
                nyquist_attenuation: 1.0,
                casimir_before: before,
                casimir_after: before,
                diagnostics_before,
                diagnostics_after: diagnostics_before,
                removed_total_casimir: 0.0,
                removed_spatial_mean_casimir: 0.0,
                removed_fluctuation_f_strophy: 0.0,
                spectral_removed_casimir: 0.0,
                max_imaginary_residual: 0.0,
                max_real_magnitude: self
                    .distribution
                    .iter()
                    .map(|value| value.abs())
                    .fold(0.0_f64, f64::max),
                casimir_accounting_residual: 0.0,
            });
        }

        let config = self.config;
        let expected =
            config
                .nx
                .checked_mul(config.nv)
                .ok_or(VelocityFourierSpongeError::Solver(
                    SolverError::GridSizeOverflow,
                ))?;

        let mut candidate = Vec::new();
        candidate.try_reserve_exact(expected).map_err(|_| {
            VelocityFourierSpongeError::Solver(SolverError::AllocationFailed { elements: expected })
        })?;

        let mut line = Vec::new();
        line.try_reserve_exact(config.nv).map_err(|_| {
            VelocityFourierSpongeError::Solver(SolverError::AllocationFailed {
                elements: config.nv,
            })
        })?;
        line.resize(config.nv, Complex::ZERO);

        let nyquist_mode = config.nv / 2;
        let mut spectral_removed_casimir = 0.0;
        let mut max_imaginary_residual = 0.0_f64;
        let mut max_real_magnitude = 0.0_f64;
        let spectral_weight = config.dx() * config.dv() / config.nv as f64;

        for ix in 0..config.nx {
            let row_start = ix * config.nv;
            for (mode, value) in line.iter_mut().enumerate() {
                *value = Complex::new(self.distribution[row_start + mode], 0.0);
            }

            fft(&mut line, false);
            for (mode, coefficient) in line.iter_mut().enumerate() {
                let signed = signed_mode(mode, config.nv);
                let normalized = signed.unsigned_abs() as f64 / nyquist_mode as f64;
                let attenuation = if normalized > start_fraction_of_nyquist {
                    let ramp = (normalized - start_fraction_of_nyquist)
                        / (1.0 - start_fraction_of_nyquist);
                    let damping_rate = maximum_damping_rate * ramp.powi(exponent as i32);
                    (-damping_rate * duration).exp()
                } else {
                    1.0
                };

                let power = coefficient
                    .re
                    .mul_add(coefficient.re, coefficient.im * coefficient.im);
                spectral_removed_casimir +=
                    (1.0 - attenuation * attenuation) * power * spectral_weight;
                *coefficient = coefficient.scale(attenuation);
                if !coefficient.re.is_finite()
                    || !coefficient.im.is_finite()
                    || !spectral_removed_casimir.is_finite()
                {
                    return Err(VelocityFourierSpongeError::NonFiniteTransform);
                }
            }

            fft(&mut line, true);
            for value in &line {
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(VelocityFourierSpongeError::NonFiniteTransform);
                }
                max_imaginary_residual = max_imaginary_residual.max(value.im.abs());
                max_real_magnitude = max_real_magnitude.max(value.re.abs());
                candidate.push(value.re);
            }
        }

        let previous_distribution = std::mem::replace(&mut self.distribution, candidate);
        let previous_field = std::mem::take(&mut self.electric_field);

        if let Err(error) = self
            .refresh_electric_field()
            .and_then(|()| self.ensure_finite_state())
        {
            self.distribution = previous_distribution;
            self.electric_field = previous_field;
            return Err(VelocityFourierSpongeError::Solver(error));
        }

        let imaginary_tolerance = 256.0 * f64::EPSILON * max_real_magnitude.max(f64::MIN_POSITIVE);
        if max_imaginary_residual > imaginary_tolerance {
            self.distribution = previous_distribution;
            self.electric_field = previous_field;
            return Err(VelocityFourierSpongeError::NonFiniteTransform);
        }

        let after = match self.quadratic_casimir_budget() {
            Ok(budget) => budget,
            Err(error) => {
                self.distribution = previous_distribution;
                self.electric_field = previous_field;
                return Err(VelocityFourierSpongeError::Solver(error));
            }
        };
        let diagnostics_after = match self.diagnostics() {
            Ok(diagnostics) => diagnostics,
            Err(error) => {
                self.distribution = previous_distribution;
                self.electric_field = previous_field;
                return Err(VelocityFourierSpongeError::Solver(error));
            }
        };

        let removed_total_casimir = before.total_casimir - after.total_casimir;
        let removed_spatial_mean_casimir = before.spatial_mean_casimir - after.spatial_mean_casimir;
        let removed_fluctuation_f_strophy =
            before.fluctuation_f_strophy - after.fluctuation_f_strophy;
        let casimir_accounting_residual = removed_total_casimir - spectral_removed_casimir;

        if !removed_total_casimir.is_finite()
            || !removed_spatial_mean_casimir.is_finite()
            || !removed_fluctuation_f_strophy.is_finite()
            || !spectral_removed_casimir.is_finite()
            || !casimir_accounting_residual.is_finite()
        {
            self.distribution = previous_distribution;
            self.electric_field = previous_field;
            return Err(VelocityFourierSpongeError::NonFiniteTransform);
        }

        let nyquist_attenuation = (-maximum_damping_rate * duration).exp();

        Ok(VelocityFourierSpongeReport {
            duration,
            start_fraction_of_nyquist,
            maximum_damping_rate,
            exponent,
            nyquist_attenuation,
            casimir_before: before,
            casimir_after: after,
            diagnostics_before,
            diagnostics_after,
            removed_total_casimir,
            removed_spatial_mean_casimir,
            removed_fluctuation_f_strophy,
            spectral_removed_casimir,
            max_imaginary_residual,
            max_real_magnitude,
            casimir_accounting_residual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VlasovPoissonConfig;
    use std::f64::consts::TAU;

    fn config() -> VlasovPoissonConfig {
        VlasovPoissonConfig {
            nx: 64,
            nv: 256,
            domain_length: 4.0 * std::f64::consts::PI,
            velocity_min: -6.0,
            velocity_max: 6.0,
            charge_to_mass: -1.0,
            permittivity: 1.0,
            background_density: 1.0,
        }
    }

    fn distribution(with_x_dependence: bool) -> Vec<f64> {
        let config = config();
        let mut values = vec![0.0; config.nx * config.nv];
        for ix in 0..config.nx {
            let spatial = if with_x_dependence {
                1.0 + 0.05 * (TAU * config.x(ix) / config.domain_length).cos()
            } else {
                1.0
            };
            for iv in 0..config.nv {
                let phase = TAU * 8.0 * iv as f64 / config.nv as f64;
                let high_phase = TAU * 112.0 * iv as f64 / config.nv as f64;
                values[ix * config.nv + iv] =
                    spatial * (1.0 + 0.1 * phase.cos() + 0.2 * high_phase.cos());
            }
        }
        values
    }

    fn cosine_amplitude(distribution: &[f64], mode: usize) -> f64 {
        let config = config();
        let mut projection = 0.0;
        for ix in 0..config.nx {
            for iv in 0..config.nv {
                let phase = TAU * mode as f64 * iv as f64 / config.nv as f64;
                projection += distribution[ix * config.nv + iv] * phase.cos();
            }
        }
        2.0 * projection / (config.nx * config.nv) as f64
    }

    #[test]
    fn sponge_damps_high_s_and_preserves_low_s_density_and_clock() {
        let config = config();
        let initial = distribution(false);
        let mut solver = VlasovPoisson1d1v::new(config, initial.clone()).unwrap();
        let time_before = solver.time();
        let density_mode_before = solver.electric_mode_coefficient(1).unwrap();
        let expected_high = 0.2 * (-0.5625_f64).exp();

        let report = solver
            .apply_velocity_fourier_sponge(0.5, 0.5, 2.0, 2)
            .unwrap();
        let high_after = cosine_amplitude(solver.distribution(), 112);
        let low_after = cosine_amplitude(solver.distribution(), 8);
        let low_before = cosine_amplitude(&initial, 8);
        let density_mode_after = solver.electric_mode_coefficient(1).unwrap();

        assert!((high_after - expected_high).abs() < 1.0e-10);
        assert!((low_after - low_before).abs() < 1.0e-10);
        assert!((density_mode_after.amplitude() - density_mode_before.amplitude()).abs() < 1.0e-12);
        let mass_delta =
            (report.diagnostics_after.total_mass - report.diagnostics_before.total_mass).abs();
        let mass_scale = report.diagnostics_before.total_mass.abs().max(1.0);
        assert!(
            mass_delta < 1.0e-12 * mass_scale,
            "zero-velocity-mode mass drift = {mass_delta:e} (scale {mass_scale:e})"
        );
        assert!(report.diagnostics_after.minimum_distribution > 0.0);
        assert_eq!(
            report.diagnostics_after.time,
            report.diagnostics_before.time
        );
        assert_eq!(solver.time(), time_before);
        assert!((report.nyquist_attenuation - (-1.0_f64).exp()).abs() < 1.0e-14);
        assert!(report.removed_total_casimir > 0.0);
        assert!(report.max_imaginary_residual < 256.0 * f64::EPSILON * report.max_real_magnitude);
        assert!(report.max_real_magnitude > 0.0);
    }

    #[test]
    fn sponge_reports_parseval_casimir_and_fluctuation_loss() {
        let mut solver = VlasovPoisson1d1v::new(config(), distribution(true)).unwrap();
        let report = solver
            .apply_velocity_fourier_sponge(0.5, 0.5, 2.0, 2)
            .unwrap();
        let scale = report.casimir_before.total_casimir.max(1.0);
        assert!(report.removed_total_casimir > 0.0);
        assert!(report.removed_fluctuation_f_strophy > 0.0);
        assert!(
            report.casimir_accounting_residual.abs() < 1.0e-11 * scale,
            "C2 accounting residual = {:e}",
            report.casimir_accounting_residual
        );
        assert!(
            (report.removed_total_casimir
                - report.removed_spatial_mean_casimir
                - report.removed_fluctuation_f_strophy)
                .abs()
                < 1.0e-11 * scale
        );
    }

    #[test]
    fn zero_duration_sponge_is_an_identity() {
        let initial = distribution(true);
        let mut solver = VlasovPoisson1d1v::new(config(), initial.clone()).unwrap();
        let report = solver
            .apply_velocity_fourier_sponge(0.0, 0.5, 2.0, 2)
            .unwrap();
        assert_eq!(solver.distribution(), initial.as_slice());
        assert_eq!(report.removed_total_casimir, 0.0);
    }

    #[test]
    fn invalid_sponge_parameters_fail_without_mutation() {
        let initial = distribution(true);
        let mut solver = VlasovPoisson1d1v::new(config(), initial.clone()).unwrap();
        let invalid = [
            (f64::NAN, 0.5, 2.0, 2),
            (-1.0, 0.5, 2.0, 2),
            (0.5, f64::NAN, 2.0, 2),
            (0.5, 1.0, 2.0, 2),
            (0.5, 0.5, f64::INFINITY, 2),
            (0.5, 0.5, -1.0, 2),
            (0.5, 0.5, 2.0, 0),
            (0.5, 0.5, 2.0, 17),
        ];
        for (duration, start, damping, exponent) in invalid {
            assert!(
                solver
                    .apply_velocity_fourier_sponge(duration, start, damping, exponent)
                    .is_err()
            );
            assert_eq!(solver.distribution(), initial.as_slice());
            assert_eq!(solver.time(), 0.0);
        }
    }
}
