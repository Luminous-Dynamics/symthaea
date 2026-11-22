//! Quadratic-Casimir and joint phase-space spectrum diagnostics.
//!
//! The forward transform convention is exp(-i k x) in space and exp(-i s v)
//! in velocity. For free streaming, a positive spatial mode shears toward
//! negative velocity-Fourier mode (s approximately -k t). Therefore k*s < 0
//! is labelled the phase-mixing sector and k*s > 0 the phase-unmixing sector.
//! These are directional spectral occupancies, not transfer rates or fluxes.

use super::{Complex, SolverError, VlasovPoisson1d1v, fft, signed_mode};

/// Discrete quadratic-Casimir partition for a 1D1V distribution.
///
/// All values use the physical-coordinate quadrature dx*dv. The spatial mean
/// is averaged over x, so C2 = C2_mean + f_strophy in exact arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuadraticCasimirBudget {
    /// Total discrete Casimir: dx*dv*sum_{x,v} f(x,v)^2.
    pub total_casimir: f64,
    /// Spatial-mean contribution: Lx*dv*sum_v mean_x(f)^2.
    pub spatial_mean_casimir: f64,
    /// Fluctuation contribution: dx*dv*sum_{x,v} (f-mean_x(f))^2.
    pub fluctuation_f_strophy: f64,
    /// total - mean - fluctuation; should be roundoff-sized.
    pub partition_residual: f64,
}

/// Joint Fourier-space distribution of fluctuation f-strophy.
///
/// joint_strophy is row-major in standard FFT index order:
/// x_mode_index * nv + velocity_mode_index. Convert indices to signed modes
/// with signed_x_mode and signed_velocity_mode.
///
/// phase_mixing_f_strophy sums modes with signed k*s < 0.
/// phase_unmixing_f_strophy sums modes with signed k*s > 0.
/// Modes on either zero axis are reported separately.
#[derive(Debug, Clone, PartialEq)]
pub struct JointPhaseSpaceSpectrum {
    pub nx: usize,
    pub nv: usize,
    /// Real-space fluctuation f-strophy from the Casimir budget.
    pub real_space_f_strophy: f64,
    /// Sum of joint spectral bins after Parseval normalization.
    pub spectral_f_strophy: f64,
    /// Absolute difference between spectral and real-space f-strophy.
    pub parseval_absolute_residual: f64,
    /// Absolute Parseval residual divided by max(real-space, spectral, MIN_POSITIVE).
    pub parseval_relative_residual: f64,
    /// Directional occupancy for k*s < 0 under the documented FFT convention.
    pub phase_mixing_f_strophy: f64,
    /// Directional occupancy for k*s > 0 under the documented FFT convention.
    pub phase_unmixing_f_strophy: f64,
    /// Occupancy for k=0 or s=0 modes (counted once).
    pub zero_axis_f_strophy: f64,
    /// Marginal f-strophy indexed by the standard FFT x-mode index.
    pub by_x_mode: Vec<f64>,
    /// Marginal f-strophy indexed by the standard FFT velocity-mode index.
    pub by_velocity_mode: Vec<f64>,
    /// Joint f-strophy bins, row-major (x mode, velocity mode).
    pub joint_strophy: Vec<f64>,
}

impl JointPhaseSpaceSpectrum {
    /// Convert an FFT-order x index to its signed integer mode.
    pub fn signed_x_mode(&self, mode_index: usize) -> Option<isize> {
        (mode_index < self.nx).then(|| signed_mode(mode_index, self.nx))
    }

    /// Convert an FFT-order velocity index to its signed integer mode.
    pub fn signed_velocity_mode(&self, mode_index: usize) -> Option<isize> {
        (mode_index < self.nv).then(|| signed_mode(mode_index, self.nv))
    }

    /// Return the f-strophy in one joint FFT bin, or None for an invalid index.
    pub fn mode_energy(&self, x_mode_index: usize, velocity_mode_index: usize) -> Option<f64> {
        if x_mode_index >= self.nx || velocity_mode_index >= self.nv {
            return None;
        }
        self.joint_strophy
            .get(x_mode_index * self.nv + velocity_mode_index)
            .copied()
    }
}

impl VlasovPoisson1d1v {
    /// Compute the discrete quadratic-Casimir partition.
    ///
    /// This is a diagnostic of the declared finite grid and quadrature. For
    /// collisionless Vlasov evolution, the continuum quadratic Casimir is
    /// invariant; observed drift is a numerical/resolution diagnostic.
    pub fn quadratic_casimir_budget(&self) -> Result<QuadraticCasimirBudget, SolverError> {
        let config = self.config;
        let dx = config.dx();
        let dv = config.dv();
        let mut spatial_mean = vec![0.0; config.nv];

        for ix in 0..config.nx {
            for (iv, mean) in spatial_mean.iter_mut().enumerate() {
                *mean += self.distribution[ix * config.nv + iv] / config.nx as f64;
            }
        }

        let mut total_casimir = 0.0;
        let mut spatial_mean_casimir = 0.0;
        let mut fluctuation_f_strophy = 0.0;
        for ix in 0..config.nx {
            for (iv, mean) in spatial_mean.iter().copied().enumerate() {
                let value = self.distribution[ix * config.nv + iv];
                total_casimir = (value * value).mul_add(dx * dv, total_casimir);
                let fluctuation = value - mean;
                fluctuation_f_strophy =
                    (fluctuation * fluctuation).mul_add(dx * dv, fluctuation_f_strophy);
            }
        }
        for mean in spatial_mean {
            spatial_mean_casimir =
                (mean * mean).mul_add(config.domain_length * dv, spatial_mean_casimir);
        }

        let partition_residual = total_casimir - spatial_mean_casimir - fluctuation_f_strophy;
        if !total_casimir.is_finite()
            || !spatial_mean_casimir.is_finite()
            || !fluctuation_f_strophy.is_finite()
            || !partition_residual.is_finite()
        {
            return Err(SolverError::NonFiniteState);
        }

        Ok(QuadraticCasimirBudget {
            total_casimir,
            spatial_mean_casimir,
            fluctuation_f_strophy,
            partition_residual,
        })
    }

    /// Compute the joint x/velocity Fourier spectrum of fluctuation f-strophy.
    ///
    /// The spatial mean is removed before the transform. Forward transforms
    /// are unnormalised; each squared coefficient is multiplied by
    /// dx*dv/(nx*nv), making the sum over all joint bins equal the real-space
    /// fluctuation f-strophy by Parseval's identity.
    ///
    /// This computes spectral occupancy, not a transfer flux. Flux requires
    /// time-resolved cross-scale transfer terms or a separately derived budget.
    pub fn joint_phase_space_spectrum(&self) -> Result<JointPhaseSpaceSpectrum, SolverError> {
        let config = self.config;
        let nx = config.nx;
        let nv = config.nv;
        let budget = self.quadratic_casimir_budget()?;
        let mut spatial_mean = vec![0.0; nv];
        for ix in 0..nx {
            for (iv, mean) in spatial_mean.iter_mut().enumerate() {
                *mean += self.distribution[ix * nv + iv] / nx as f64;
            }
        }

        let mut x_line = vec![Complex::ZERO; nx];
        let mut x_spectrum = vec![Complex::ZERO; nx * nv];
        for iv in 0..nv {
            for ix in 0..nx {
                x_line[ix] = Complex::new(self.distribution[ix * nv + iv] - spatial_mean[iv], 0.0);
            }
            fft(&mut x_line, false);
            for (kx, coefficient) in x_line.iter().copied().enumerate() {
                x_spectrum[kx * nv + iv] = coefficient;
            }
        }

        let normalization = config.dx() * config.dv() / (nx as f64 * nv as f64);
        let mut velocity_line = vec![Complex::ZERO; nv];
        let mut by_x_mode = vec![0.0; nx];
        let mut by_velocity_mode = vec![0.0; nv];
        let mut joint_strophy = vec![0.0; nx * nv];
        let mut spectral_f_strophy = 0.0;
        let mut phase_mixing_f_strophy = 0.0;
        let mut phase_unmixing_f_strophy = 0.0;
        let mut zero_axis_f_strophy = 0.0;

        for kx_index in 0..nx {
            for sv_index in 0..nv {
                velocity_line[sv_index] = x_spectrum[kx_index * nv + sv_index];
            }
            fft(&mut velocity_line, false);

            let signed_k = signed_mode(kx_index, nx);
            for (sv_index, coefficient) in velocity_line.iter().copied().enumerate() {
                let power = coefficient
                    .re
                    .mul_add(coefficient.re, coefficient.im * coefficient.im);
                let energy = power * normalization;
                if !power.is_finite() || !energy.is_finite() {
                    return Err(SolverError::NonFiniteState);
                }

                let signed_s = signed_mode(sv_index, nv);
                joint_strophy[kx_index * nv + sv_index] = energy;
                by_x_mode[kx_index] += energy;
                by_velocity_mode[sv_index] += energy;
                spectral_f_strophy += energy;

                match (signed_k * signed_s).cmp(&0) {
                    std::cmp::Ordering::Less => phase_mixing_f_strophy += energy,
                    std::cmp::Ordering::Greater => phase_unmixing_f_strophy += energy,
                    std::cmp::Ordering::Equal => zero_axis_f_strophy += energy,
                }
            }
        }

        let parseval_absolute_residual = (spectral_f_strophy - budget.fluctuation_f_strophy).abs();
        let parseval_scale = spectral_f_strophy
            .max(budget.fluctuation_f_strophy)
            .max(f64::MIN_POSITIVE);
        let parseval_relative_residual = parseval_absolute_residual / parseval_scale;

        if !spectral_f_strophy.is_finite()
            || !phase_mixing_f_strophy.is_finite()
            || !phase_unmixing_f_strophy.is_finite()
            || !zero_axis_f_strophy.is_finite()
            || !parseval_relative_residual.is_finite()
        {
            return Err(SolverError::NonFiniteState);
        }

        Ok(JointPhaseSpaceSpectrum {
            nx,
            nv,
            real_space_f_strophy: budget.fluctuation_f_strophy,
            spectral_f_strophy,
            parseval_absolute_residual,
            parseval_relative_residual,
            phase_mixing_f_strophy,
            phase_unmixing_f_strophy,
            zero_axis_f_strophy,
            by_x_mode,
            by_velocity_mode,
            joint_strophy,
        })
    }
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

    #[test]
    fn quadratic_casimir_partition_closes_and_spectrum_satisfies_parseval() {
        let solver = initial_solver();
        let budget = solver.quadratic_casimir_budget().unwrap();
        let scale = budget.total_casimir.max(1.0);
        assert!(budget.partition_residual.abs() <= 1.0e-12 * scale);

        let spectrum = solver.joint_phase_space_spectrum().unwrap();
        assert!(spectrum.parseval_relative_residual < 1.0e-12);
        assert!(
            (spectrum.spectral_f_strophy - budget.fluctuation_f_strophy).abs() <= 1.0e-12 * scale
        );
        assert!(
            (spectrum.phase_mixing_f_strophy
                + spectrum.phase_unmixing_f_strophy
                + spectrum.zero_axis_f_strophy
                - spectrum.spectral_f_strophy)
                .abs()
                <= 1.0e-12 * scale
        );
        assert_eq!(spectrum.signed_x_mode(1), Some(1));
        assert_eq!(spectrum.signed_x_mode(63), Some(-1));
        assert_eq!(spectrum.signed_velocity_mode(1), Some(1));
        assert_eq!(spectrum.signed_velocity_mode(255), Some(-1));
        assert_eq!(spectrum.mode_energy(64, 0), None);
        assert_eq!(spectrum.mode_energy(0, 256), None);
    }

    #[test]
    fn free_streaming_moves_joint_spectrum_toward_phase_mixing_sector() {
        let mut solver = initial_solver();
        solver.advance_free_streaming(4.0).unwrap();
        let mixed = solver.joint_phase_space_spectrum().unwrap();
        assert!(
            mixed.phase_mixing_f_strophy > 4.0 * mixed.phase_unmixing_f_strophy,
            "mixing={}, unmixing={}",
            mixed.phase_mixing_f_strophy,
            mixed.phase_unmixing_f_strophy
        );
        assert!(mixed.parseval_relative_residual < 1.0e-12);
    }

    #[test]
    fn collisionless_quadratic_casimir_is_preserved_by_unitary_subflows() {
        let mut solver = initial_solver();
        let initial = solver.quadratic_casimir_budget().unwrap();
        solver.advance_free_streaming(0.4).unwrap();
        let kick: Vec<f64> = (0..solver.config.nx)
            .map(|ix| {
                0.02 * (std::f64::consts::TAU * 2.0 * solver.config.x(ix)
                    / solver.config.domain_length)
                    .sin()
            })
            .collect();
        solver.apply_velocity_impulse(&kick).unwrap();
        let after = solver.quadratic_casimir_budget().unwrap();
        let scale = initial.total_casimir.max(1.0);
        assert!(
            (after.total_casimir - initial.total_casimir).abs() < 1.0e-11 * scale,
            "C2 drift = {:e}",
            after.total_casimir - initial.total_casimir
        );
    }
}
