//! Instantaneous semidiscrete f-strophy transfer rates and scale-space fluxes.
//!
//! Unlike the occupancy diagnostic, this module evaluates the Vlasov RHS at
//! the current state, transforms the resulting time derivative, and measures
//! the rate of change of every joint (k,s) f-strophy bin. Positive outward
//! flux means the f-strophy in the declared low-mode set is decreasing.

use std::fmt;

use super::{Complex, VlasovPoisson1d1v, VlasovPoissonConfig, fft, signed_mode};

/// Invalid scale cutoffs for an instantaneous phase-space flux calculation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhaseSpaceFluxError {
    InvalidXModeCutoff { cutoff: usize, nyquist: usize },
    InvalidVelocityModeCutoff { cutoff: usize, nyquist: usize },
    NonFiniteTransfer,
}

impl fmt::Display for PhaseSpaceFluxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidXModeCutoff { cutoff, nyquist } => write!(
                formatter,
                "x-mode cutoff {cutoff} must be below Nyquist mode {nyquist}"
            ),
            Self::InvalidVelocityModeCutoff { cutoff, nyquist } => write!(
                formatter,
                "velocity-mode cutoff {cutoff} must be below Nyquist mode {nyquist}"
            ),
            Self::NonFiniteTransfer => {
                write!(formatter, "phase-space transfer calculation produced a non-finite value")
            }
        }
    }
}

impl std::error::Error for PhaseSpaceFluxError {}

/// Instantaneous change of Parseval-normalized fluctuation f-strophy.
///
/// Joint rates are row-major in standard FFT index order:
/// x-mode-index times nv plus velocity-mode-index. Units are f-strophy per
/// unit simulation time.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseSpaceTransferRates {
    pub nx: usize,
    pub nv: usize,
    pub x_mode_cutoff: usize,
    pub velocity_mode_cutoff: usize,
    /// Real-space total quadratic-Casimir rate; zero for the declared
    /// collisionless periodic semidiscrete operator up to roundoff.
    pub total_casimir_rate: f64,
    /// Rate of the spatial-mean Casimir contribution.
    pub spatial_mean_casimir_rate: f64,
    /// Real-space rate of fluctuation f-strophy.
    pub real_space_f_strophy_rate: f64,
    /// Sum of all joint spectral-bin rates.
    pub spectral_f_strophy_rate: f64,
    /// Difference between spectral sum and real-space f-strophy rate.
    pub parseval_absolute_residual: f64,
    pub parseval_relative_residual: f64,
    /// Difference between total Casimir rate and the mean + fluctuation rates.
    pub casimir_partition_residual: f64,
    /// Sum of rates for signed k*s < 0, excluding Nyquist bins.
    pub phase_mixing_sector_rate: f64,
    /// Sum of rates for signed k*s > 0, excluding Nyquist bins.
    pub phase_unmixing_sector_rate: f64,
    /// Sum of rates on k=0 or s=0, excluding Nyquist bins.
    pub zero_axis_rate: f64,
    /// Sum of rates touching either Nyquist axis.
    pub nyquist_rate: f64,
    /// Positive means transfer out of |k| <= x_mode_cutoff.
    pub flux_out_of_low_x_modes: f64,
    /// Positive means transfer out of |s| <= velocity_mode_cutoff.
    pub flux_out_of_low_velocity_modes: f64,
    pub by_x_mode_rate: Vec<f64>,
    pub by_velocity_mode_rate: Vec<f64>,
    pub joint_mode_rate: Vec<f64>,
}

impl PhaseSpaceTransferRates {
    pub fn mode_rate(
        &self,
        x_mode_index: usize,
        velocity_mode_index: usize,
    ) -> Option<f64> {
        if x_mode_index >= self.nx || velocity_mode_index >= self.nv {
            return None;
        }
        self.joint_mode_rate
            .get(x_mode_index * self.nv + velocity_mode_index)
            .copied()
    }
}

impl VlasovPoisson1d1v {
    /// Calculate the instantaneous semidiscrete phase-space transfer rate.
    ///
    /// The RHS is evaluated from the same spectral derivative convention as
    /// the solver's infinitesimal advection generators:
    ///
    ///     partial_t f = -v * D_x(f) - (q/m) * E(x) * D_v(f)
    ///
    /// D_x and D_v use the collocation Fourier derivative, with self-conjugate
    /// Nyquist derivatives set to zero. The spatial mean is removed from both
    /// f and its RHS before the joint transform.
    ///
    /// Outward flux is defined as the negative rate of f-strophy in the
    /// corresponding low-mode set. Positive values therefore indicate net
    /// loss from the low modes. This is an instantaneous operator diagnostic,
    /// not a finite-dt difference or a time-integrated cascade measurement.
    pub fn phase_space_transfer_rates(
        &self,
        x_mode_cutoff: usize,
        velocity_mode_cutoff: usize,
    ) -> Result<PhaseSpaceTransferRates, PhaseSpaceFluxError> {
        let config = self.config;
        let nx = config.nx;
        let nv = config.nv;

        if x_mode_cutoff >= nx / 2 {
            return Err(PhaseSpaceFluxError::InvalidXModeCutoff {
                cutoff: x_mode_cutoff,
                nyquist: nx / 2,
            });
        }
        if velocity_mode_cutoff >= nv / 2 {
            return Err(PhaseSpaceFluxError::InvalidVelocityModeCutoff {
                cutoff: velocity_mode_cutoff,
                nyquist: nv / 2,
            });
        }

        let derivative_x = spectral_derivative_x(&self.distribution, config);
        let derivative_v = spectral_derivative_v(&self.distribution, config);
        let mut rhs = vec![0.0; nx * nv];

        for ix in 0..nx {
            let acceleration = config.charge_to_mass * self.electric_field[ix];
            for iv in 0..nv {
                let index = ix * nv + iv;
                rhs[index] = -config.velocity(iv) * derivative_x[index]
                    - acceleration * derivative_v[index];
            }
        }

        let mut mean_f = vec![0.0; nv];
        let mut mean_rhs = vec![0.0; nv];
        for ix in 0..nx {
            for iv in 0..nv {
                let index = ix * nv + iv;
                mean_f[iv] += self.distribution[index] / nx as f64;
                mean_rhs[iv] += rhs[index] / nx as f64;
            }
        }

        let dx = config.dx();
        let dv = config.dv();
        let mut total_casimir_rate = 0.0;
        let mut spatial_mean_casimir_rate = 0.0;
        let mut real_space_f_strophy_rate = 0.0;

        for ix in 0..nx {
            for iv in 0..nv {
                let index = ix * nv + iv;
                let value = self.distribution[index];
                let value_rate = rhs[index];
                let fluctuation = value - mean_f[iv];
                let fluctuation_rate = value_rate - mean_rhs[iv];

                total_casimir_rate =
                    (2.0 * value * value_rate).mul_add(dx * dv, total_casimir_rate);
                real_space_f_strophy_rate =
                    (2.0 * fluctuation * fluctuation_rate).mul_add(dx * dv, real_space_f_strophy_rate);
            }
        }
        for iv in 0..nv {
            spatial_mean_casimir_rate = (2.0 * mean_f[iv] * mean_rhs[iv])
                .mul_add(config.domain_length * dv, spatial_mean_casimir_rate);
        }

        let mut x_f_spectrum = vec![Complex::ZERO; nx * nv];
        let mut x_rhs_spectrum = vec![Complex::ZERO; nx * nv];
        let mut x_line_f = vec![Complex::ZERO; nx];
        let mut x_line_rhs = vec![Complex::ZERO; nx];

        for iv in 0..nv {
            for ix in 0..nx {
                let index = ix * nv + iv;
                x_line_f[ix] = Complex::new(self.distribution[index] - mean_f[iv], 0.0);
                x_line_rhs[ix] = Complex::new(rhs[index] - mean_rhs[iv], 0.0);
            }
            fft(&mut x_line_f, false);
            fft(&mut x_line_rhs, false);
            for kx in 0..nx {
                x_f_spectrum[kx * nv + iv] = x_line_f[kx];
                x_rhs_spectrum[kx * nv + iv] = x_line_rhs[kx];
            }
        }

        let normalization = dx * dv / (nx as f64 * nv as f64);
        let mut velocity_line_f = vec![Complex::ZERO; nv];
        let mut velocity_line_rhs = vec![Complex::ZERO; nv];
        let mut by_x_mode_rate = vec![0.0; nx];
        let mut by_velocity_mode_rate = vec![0.0; nv];
        let mut joint_mode_rate = vec![0.0; nx * nv];
        let mut spectral_f_strophy_rate = 0.0;
        let mut phase_mixing_sector_rate = 0.0;
        let mut phase_unmixing_sector_rate = 0.0;
        let mut zero_axis_rate = 0.0;
        let mut nyquist_rate = 0.0;
        let mut low_x_rate = 0.0;
        let mut low_velocity_rate = 0.0;

        for kx in 0..nx {
            for sv in 0..nv {
                velocity_line_f[sv] = x_f_spectrum[kx * nv + sv];
                velocity_line_rhs[sv] = x_rhs_spectrum[kx * nv + sv];
            }
            fft(&mut velocity_line_f, false);
            fft(&mut velocity_line_rhs, false);

            let signed_k = signed_mode(kx, nx);
            for sv in 0..nv {
                let f_mode = velocity_line_f[sv];
                let rhs_mode = velocity_line_rhs[sv];
                let rate = 2.0
                    * normalization
                    * f_mode.re.mul_add(rhs_mode.re, f_mode.im * rhs_mode.im);
                if !f_mode.re.is_finite()
                    || !f_mode.im.is_finite()
                    || !rhs_mode.re.is_finite()
                    || !rhs_mode.im.is_finite()
                    || !rate.is_finite()
                {
                    return Err(PhaseSpaceFluxError::NonFiniteTransfer);
                }

                let signed_s = signed_mode(sv, nv);
                joint_mode_rate[kx * nv + sv] = rate;
                by_x_mode_rate[kx] += rate;
                by_velocity_mode_rate[sv] += rate;
                spectral_f_strophy_rate += rate;

                if kx == nx / 2 || sv == nv / 2 {
                    nyquist_rate += rate;
                } else {
                    match (signed_k * signed_s).cmp(&0) {
                        std::cmp::Ordering::Less => phase_mixing_sector_rate += rate,
                        std::cmp::Ordering::Greater => phase_unmixing_sector_rate += rate,
                        std::cmp::Ordering::Equal => zero_axis_rate += rate,
                    }
                }

                if signed_k.unsigned_abs() <= x_mode_cutoff {
                    low_x_rate += rate;
                }
                if signed_s.unsigned_abs() <= velocity_mode_cutoff {
                    low_velocity_rate += rate;
                }
            }
        }

        let parseval_absolute_residual =
            (spectral_f_strophy_rate - real_space_f_strophy_rate).abs();
        let parseval_scale = spectral_f_strophy_rate
            .abs()
            .max(real_space_f_strophy_rate.abs())
            .max(f64::MIN_POSITIVE);
        let parseval_relative_residual = parseval_absolute_residual / parseval_scale;
        let casimir_partition_residual =
            total_casimir_rate - spatial_mean_casimir_rate - real_space_f_strophy_rate;

        let values = [
            total_casimir_rate,
            spatial_mean_casimir_rate,
            real_space_f_strophy_rate,
            spectral_f_strophy_rate,
            parseval_relative_residual,
            casimir_partition_residual,
            phase_mixing_sector_rate,
            phase_unmixing_sector_rate,
            zero_axis_rate,
            nyquist_rate,
            low_x_rate,
            low_velocity_rate,
        ];
        if values.iter().any(|value| !value.is_finite())
            || by_x_mode_rate.iter().any(|value| !value.is_finite())
            || by_velocity_mode_rate.iter().any(|value| !value.is_finite())
        {
            return Err(PhaseSpaceFluxError::NonFiniteTransfer);
        }

        Ok(PhaseSpaceTransferRates {
            nx,
            nv,
            x_mode_cutoff,
            velocity_mode_cutoff,
            total_casimir_rate,
            spatial_mean_casimir_rate,
            real_space_f_strophy_rate,
            spectral_f_strophy_rate,
            parseval_absolute_residual,
            parseval_relative_residual,
            casimir_partition_residual,
            phase_mixing_sector_rate,
            phase_unmixing_sector_rate,
            zero_axis_rate,
            nyquist_rate,
            flux_out_of_low_x_modes: -low_x_rate,
            flux_out_of_low_velocity_modes: -low_velocity_rate,
            by_x_mode_rate,
            by_velocity_mode_rate,
            joint_mode_rate,
        })
    }
}

fn spectral_derivative_x(
    distribution: &[f64],
    config: VlasovPoissonConfig,
) -> Vec<f64> {
    let mut derivative = vec![0.0; config.nx * config.nv];
    let mut line = vec![Complex::ZERO; config.nx];

    for iv in 0..config.nv {
        for (ix, value) in line.iter_mut().enumerate() {
            *value = Complex::new(distribution[ix * config.nv + iv], 0.0);
        }
        fft(&mut line, false);
        for (mode, coefficient) in line.iter_mut().enumerate() {
            let wave_number = if mode == config.nx / 2 {
                0.0
            } else {
                signed_mode(mode, config.nx) as f64 * std::f64::consts::TAU
                    / config.domain_length
            };
            *coefficient = coefficient.mul_i(wave_number);
        }
        fft(&mut line, true);
        for (ix, value) in line.iter().enumerate() {
            derivative[ix * config.nv + iv] = value.re;
        }
    }

    derivative
}

fn spectral_derivative_v(
    distribution: &[f64],
    config: VlasovPoissonConfig,
) -> Vec<f64> {
    let mut derivative = vec![0.0; config.nx * config.nv];
    let velocity_length = config.velocity_max - config.velocity_min;
    let mut line = vec![Complex::ZERO; config.nv];

    for ix in 0..config.nx {
        for (iv, value) in line.iter_mut().enumerate() {
            *value = Complex::new(distribution[ix * config.nv + iv], 0.0);
        }
        fft(&mut line, false);
        for (mode, coefficient) in line.iter_mut().enumerate() {
            let velocity_wave_number = if mode == config.nv / 2 {
                0.0
            } else {
                signed_mode(mode, config.nv) as f64 * std::f64::consts::TAU
                    / velocity_length
            };
            *coefficient = coefficient.mul_i(velocity_wave_number);
        }
        fft(&mut line, true);
        for (iv, value) in line.iter().enumerate() {
            derivative[ix * config.nv + iv] = value.re;
        }
    }

    derivative
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{VlasovPoissonConfig, maxwellian_perturbation};

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

    fn initial_solver() -> VlasovPoisson1d1v {
        let config = config();
        let distribution = maxwellian_perturbation(config, 1.0e-2, 1, 1.0).unwrap();
        VlasovPoisson1d1v::new(config, distribution).unwrap()
    }

    fn low_velocity_f_strophy(
        solver: &VlasovPoisson1d1v,
        cutoff: usize,
    ) -> f64 {
        let spectrum = solver.joint_phase_space_spectrum().unwrap();
        spectrum
            .by_velocity_mode
            .iter()
            .enumerate()
            .filter(|(index, _)| signed_mode(*index, spectrum.nv).unsigned_abs() <= cutoff)
            .map(|(_, value)| *value)
            .sum()
    }

    #[test]
    fn semidiscrete_transfer_satisfies_parseval_and_casimir_partition() {
        let solver = initial_solver();
        let rates = solver.phase_space_transfer_rates(4, 16).unwrap();
        let scale = rates.total_casimir_rate.abs()
            + rates.real_space_f_strophy_rate.abs()
            + rates.spatial_mean_casimir_rate.abs()
            + 1.0;
        assert!(rates.parseval_relative_residual < 1.0e-11);
        assert!(rates.casimir_partition_residual.abs() < 1.0e-11 * scale);
        assert!(rates.total_casimir_rate.abs() < 1.0e-11 * scale);
        assert_eq!(rates.mode_rate(64, 0), None);
        assert_eq!(rates.mode_rate(0, 256), None);
    }

    #[test]
    fn free_streaming_produces_outward_velocity_f_strophy_flux() {
        let mut solver = initial_solver();
        solver.advance_free_streaming(3.0).unwrap();
        let rates = solver.phase_space_transfer_rates(3, 2).unwrap();
        assert!(
            rates.flux_out_of_low_velocity_modes > 0.0,
            "velocity flux = {:e}",
            rates.flux_out_of_low_velocity_modes
        );
        assert!(rates.phase_mixing_sector_rate >= -1.0e-12);
    }

    #[test]
    fn finite_step_low_velocity_change_matches_instantaneous_flux() {
        let mut solver = initial_solver();
        solver.advance_free_streaming(3.0).unwrap();
        let cutoff = 2usize;
        let rates = solver.phase_space_transfer_rates(3, cutoff).unwrap();
        let before = low_velocity_f_strophy(&solver, cutoff);
        let dt = 1.0e-6;
        let mut advanced = solver.clone();
        advanced.step(dt).unwrap();
        let after = low_velocity_f_strophy(&advanced, cutoff);
        let observed_derivative = (after - before) / dt;
        let expected_derivative = -rates.flux_out_of_low_velocity_modes;
        let scale = expected_derivative.abs().max(1.0e-12);
        assert!(
            (observed_derivative - expected_derivative).abs() / scale < 2.0e-2,
            "observed low-s derivative={observed_derivative:e}, expected={expected_derivative:e}"
        );
    }

    #[test]
    fn invalid_flux_cutoffs_fail_closed() {
        let solver = initial_solver();
        assert_eq!(
            solver.phase_space_transfer_rates(32, 4),
            Err(PhaseSpaceFluxError::InvalidXModeCutoff {
                cutoff: 32,
                nyquist: 32,
            })
        );
        assert_eq!(
            solver.phase_space_transfer_rates(4, 128),
            Err(PhaseSpaceFluxError::InvalidVelocityModeCutoff {
                cutoff: 128,
                nyquist: 128,
            })
        );
    }
}
