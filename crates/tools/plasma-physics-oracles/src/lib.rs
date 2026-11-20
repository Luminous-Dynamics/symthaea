pub mod coordinate_audit;

//! Small, deterministic reference oracles for electrostatic PIC research.
//!
//! These routines are deliberately small-N verification tools, not production
//! plasma solvers. Their equations and discrete conventions are documented in
//! the package README.

#![forbid(unsafe_code)]

use std::error::Error;
use std::f64::consts::TAU;
use std::fmt;

/// A one-dimensional periodic macro-particle.
///
/// The charge is the effective particle charge under the caller's declared
/// unit convention. Macro-particle weights, if used, must already be included.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Particle1d {
    pub position: f64,
    pub charge: f64,
}

/// Parameters for a truncated periodic Fourier Green's function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicFourierConfig {
    pub domain_length: f64,
    pub permittivity: f64,
    /// Number of positive Fourier modes. The zero mode is omitted.
    pub modes: usize,
}

/// Energy and configuration-space forces for a particle configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct VariationalResult {
    pub potential_energy: f64,
    pub forces: Vec<f64>,
}

/// Periodic, one-dimensional, staggered finite-volume Gauss-law diagnostics.
///
/// The electric field is stored on right cell faces: E[i] = E_(i+1/2).
/// Charge density is stored at cell centres: rho[i] = rho_i.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussLawReport {
    /// Cellwise residual (E_(i+1/2) - E_(i-1/2))/dx - rho_i/epsilon.
    pub cell_residuals: Vec<f64>,
    pub max_abs_residual: f64,
    pub rms_residual: f64,
    /// Integrated charge sum_i rho_i * dx. A periodic 1D divergence has
    /// zero mean, so a periodic Poisson problem requires neutral net charge.
    pub net_charge: f64,
}

impl GaussLawReport {
    /// Apply separately declared residual and periodic-neutrality tolerances.
    ///
    /// Invalid (negative or non-finite) tolerances fail closed.
    pub fn satisfies_tolerances(
        &self,
        residual_tolerance: f64,
        neutrality_tolerance: f64,
    ) -> bool {
        residual_tolerance.is_finite()
            && residual_tolerance >= 0.0
            && neutrality_tolerance.is_finite()
            && neutrality_tolerance >= 0.0
            && self.max_abs_residual <= residual_tolerance
            && self.net_charge.abs() <= neutrality_tolerance
    }
}

/// Errors returned by the reference oracles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OracleError {
    InvalidDomainLength,
    InvalidPermittivity,
    ZeroFourierModes,
    NonFinitePosition { particle: usize },
    NonFiniteParticleCharge { particle: usize },
    NonFiniteOracleOutput,
    TooFewCells,
    FieldChargeLengthMismatch {
        field_cells: usize,
        charge_cells: usize,
    },
    InvalidCellWidth,
    NonFiniteFieldValue { cell: usize },
    NonFiniteChargeDensity { cell: usize },
}

impl fmt::Display for OracleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDomainLength => write!(f, "domain length must be finite and positive"),
            Self::InvalidPermittivity => write!(f, "permittivity must be finite and positive"),
            Self::ZeroFourierModes => write!(f, "at least one positive Fourier mode is required"),
            Self::NonFinitePosition { particle } => {
                write!(f, "particle {particle} has a non-finite position")
            }
            Self::NonFiniteParticleCharge { particle } => {
                write!(f, "particle {particle} has a non-finite charge")
            }
            Self::NonFiniteOracleOutput => write!(f, "oracle output is non-finite"),
            Self::TooFewCells => write!(f, "periodic Gauss verification requires at least two cells"),
            Self::FieldChargeLengthMismatch {
                field_cells,
                charge_cells,
            } => write!(
                f,
                "field has {field_cells} cells but charge density has {charge_cells} cells"
            ),
            Self::InvalidCellWidth => write!(f, "cell width must be finite and positive"),
            Self::NonFiniteFieldValue { cell } => {
                write!(f, "electric field at right face {cell} is non-finite")
            }
            Self::NonFiniteChargeDensity { cell } => {
                write!(f, "charge density in cell {cell} is non-finite")
            }
        }
    }
}

impl Error for OracleError {}

/// Evaluate a spectrally truncated periodic 1D pair energy and its exact
/// configuration-space gradient.
///
/// With k_m = 2*pi*m/L, the declared zero-mode-removed, self-interaction-
/// excluded energy is
///
/// U = 2/(L*epsilon) * sum_m sum_(i<j) q_i*q_j*cos(k_m*(x_i-x_j))/k_m^2.
///
/// The returned force is the analytic negative gradient:
///
/// F_i = 2*q_i/(L*epsilon) * sum_m sum_(j!=i)
///       q_j*sin(k_m*(x_i-x_j))/k_m.
///
/// Positions are interpreted modulo L. The routine is O(M*N^2), where M
/// is the fixed number of positive Fourier modes. This intentionally favours
/// transparent known answers over speed; it is not a production field solver.
pub fn periodic_fourier_energy_and_forces(
    particles: &[Particle1d],
    config: PeriodicFourierConfig,
) -> Result<VariationalResult, OracleError> {
    if !config.domain_length.is_finite() || config.domain_length <= 0.0 {
        return Err(OracleError::InvalidDomainLength);
    }
    if !config.permittivity.is_finite() || config.permittivity <= 0.0 {
        return Err(OracleError::InvalidPermittivity);
    }
    if config.modes == 0 {
        return Err(OracleError::ZeroFourierModes);
    }

    for (index, particle) in particles.iter().enumerate() {
        if !particle.position.is_finite() {
            return Err(OracleError::NonFinitePosition { particle: index });
        }
        if !particle.charge.is_finite() {
            return Err(OracleError::NonFiniteParticleCharge { particle: index });
        }
    }

    let length = config.domain_length;
    let epsilon = config.permittivity;
    let positions: Vec<f64> = particles
        .iter()
        .map(|particle| particle.position.rem_euclid(length))
        .collect();
    let mut potential_energy = 0.0;
    let mut forces = vec![0.0; particles.len()];

    for mode in 1..=config.modes {
        let wave_number = TAU * mode as f64 / length;
        let energy_scale = 2.0 / (length * epsilon * wave_number * wave_number);
        let force_scale = 2.0 / (length * epsilon * wave_number);

        if !wave_number.is_finite()
            || !energy_scale.is_finite()
            || !force_scale.is_finite()
        {
            return Err(OracleError::NonFiniteOracleOutput);
        }

        for (i, (particle_i, position_i)) in particles.iter().zip(&positions).enumerate() {
            for (j, (particle_j, position_j)) in
                particles.iter().zip(&positions).enumerate().skip(i + 1)
            {
                let phase = wave_number * (position_i - position_j);
                let charge_product = particle_i.charge * particle_j.charge;
                let pair_energy = energy_scale * charge_product * phase.cos();
                let pair_force = force_scale * charge_product * phase.sin();

                potential_energy += pair_energy;
                forces[i] += pair_force;
                forces[j] -= pair_force;
            }
        }
    }

    if !potential_energy.is_finite() || forces.iter().any(|force| !force.is_finite()) {
        return Err(OracleError::NonFiniteOracleOutput);
    }

    Ok(VariationalResult {
        potential_energy,
        forces,
    })
}

/// Verify the periodic 1D staggered finite-volume Gauss constraint.
///
/// electric_field_right_faces[i] is the field on the right face of cell i;
/// the periodic left-face value is electric_field_right_faces[(i+N-1)%N].
/// The cell residual is D_h E - rho/epsilon.
///
/// This function checks a supplied field; it does not compute or correct one.
pub fn verify_periodic_1d_gauss_law(
    electric_field_right_faces: &[f64],
    charge_density: &[f64],
    cell_width: f64,
    permittivity: f64,
) -> Result<GaussLawReport, OracleError> {
    if electric_field_right_faces.len() != charge_density.len() {
        return Err(OracleError::FieldChargeLengthMismatch {
            field_cells: electric_field_right_faces.len(),
            charge_cells: charge_density.len(),
        });
    }
    let cell_count = electric_field_right_faces.len();
    if cell_count < 2 {
        return Err(OracleError::TooFewCells);
    }
    if !cell_width.is_finite() || cell_width <= 0.0 {
        return Err(OracleError::InvalidCellWidth);
    }
    if !permittivity.is_finite() || permittivity <= 0.0 {
        return Err(OracleError::InvalidPermittivity);
    }

    for (cell, value) in electric_field_right_faces.iter().enumerate() {
        if !value.is_finite() {
            return Err(OracleError::NonFiniteFieldValue { cell });
        }
    }
    for (cell, value) in charge_density.iter().enumerate() {
        if !value.is_finite() {
            return Err(OracleError::NonFiniteChargeDensity { cell });
        }
    }

    let mut cell_residuals = Vec::with_capacity(cell_count);
    let mut squared_residual_sum = 0.0;
    let mut max_abs_residual: f64 = 0.0;
    let mut net_charge = 0.0;

    for (cell, (&right_face, &density)) in electric_field_right_faces
        .iter()
        .zip(charge_density.iter())
        .enumerate()
    {
        let left_face = electric_field_right_faces[(cell + cell_count - 1) % cell_count];
        let divergence = (right_face - left_face) / cell_width;
        let residual = divergence - density / permittivity;

        cell_residuals.push(residual);
        squared_residual_sum += residual * residual;
        max_abs_residual = max_abs_residual.max(residual.abs());
        net_charge += charge_density[cell] * cell_width;
    }

    let rms_residual = (squared_residual_sum / cell_count as f64).sqrt();
    if !max_abs_residual.is_finite() || !rms_residual.is_finite() || !net_charge.is_finite() {
        return Err(OracleError::NonFiniteOracleOutput);
    }

    Ok(GaussLawReport {
        cell_residuals,
        max_abs_residual,
        rms_residual,
        net_charge,
    })
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

    fn config(modes: usize) -> PeriodicFourierConfig {
        PeriodicFourierConfig {
            domain_length: TAU,
            permittivity: 1.0,
            modes,
        }
    }

    #[test]
    fn empty_and_single_particle_states_have_zero_pair_energy_and_force() {
        let empty = periodic_fourier_energy_and_forces(&[], config(2)).unwrap();
        close(empty.potential_energy, 0.0, 0.0);
        assert!(empty.forces.is_empty());

        let single = periodic_fourier_energy_and_forces(
            &[Particle1d {
                position: 1.7,
                charge: -3.0,
            }],
            config(4),
        )
        .unwrap();
        close(single.potential_energy, 0.0, 0.0);
        assert_eq!(single.forces.len(), 1);
        close(single.forces[0], 0.0, 0.0);
    }

    #[test]
    fn one_mode_two_particle_force_matches_closed_form() {
        let result = periodic_fourier_energy_and_forces(
            &[
                Particle1d {
                    position: TAU / 4.0,
                    charge: 1.0,
                },
                Particle1d {
                    position: 0.0,
                    charge: 1.0,
                },
            ],
            config(1),
        )
        .unwrap();

        close(result.potential_energy, 0.0, 1.0e-14);
        close(result.forces[0], 1.0 / std::f64::consts::PI, 1.0e-14);
        close(result.forces[1], -1.0 / std::f64::consts::PI, 1.0e-14);
    }

    #[test]
    fn internal_forces_obey_action_reaction() {
        let particles = [
            Particle1d {
                position: 0.3,
                charge: 1.0,
            },
            Particle1d {
                position: 2.1,
                charge: -0.5,
            },
            Particle1d {
                position: 5.4,
                charge: 0.25,
            },
        ];
        let result = periodic_fourier_energy_and_forces(&particles, config(5)).unwrap();
        close(result.forces.iter().sum(), 0.0, 1.0e-12);
    }

    #[test]
    fn energy_and_forces_are_permutation_equivariant() {
        let particles = [
            Particle1d {
                position: 0.4,
                charge: 1.0,
            },
            Particle1d {
                position: 2.7,
                charge: -0.7,
            },
            Particle1d {
                position: 4.9,
                charge: 0.2,
            },
        ];
        let permuted = [particles[2], particles[0], particles[1]];
        let first = periodic_fourier_energy_and_forces(&particles, config(3)).unwrap();
        let second = periodic_fourier_energy_and_forces(&permuted, config(3)).unwrap();

        close(first.potential_energy, second.potential_energy, 1.0e-12);
        close(first.forces[0], second.forces[1], 1.0e-12);
        close(first.forces[1], second.forces[2], 1.0e-12);
        close(first.forces[2], second.forces[0], 1.0e-12);
    }

    #[test]
    fn common_periodic_translation_does_not_change_energy_or_force() {
        let particles = [
            Particle1d {
                position: 6.1,
                charge: 1.0,
            },
            Particle1d {
                position: 0.8,
                charge: -0.7,
            },
            Particle1d {
                position: 3.4,
                charge: 0.2,
            },
        ];
        let translated: Vec<_> = particles
            .iter()
            .map(|particle| Particle1d {
                position: particle.position + 1.8,
                charge: particle.charge,
            })
            .collect();
        let first = periodic_fourier_energy_and_forces(&particles, config(4)).unwrap();
        let second = periodic_fourier_energy_and_forces(&translated, config(4)).unwrap();

        close(first.potential_energy, second.potential_energy, 1.0e-11);
        for (left, right) in first.forces.iter().zip(second.forces.iter()) {
            close(*left, *right, 1.0e-11);
        }
    }

    #[test]
    fn force_is_the_negative_finite_difference_energy_gradient() {
        let particles = [
            Particle1d {
                position: 0.6,
                charge: 1.0,
            },
            Particle1d {
                position: 2.2,
                charge: -0.4,
            },
            Particle1d {
                position: 5.1,
                charge: 0.8,
            },
        ];
        let base = periodic_fourier_energy_and_forces(&particles, config(4)).unwrap();
        let step = 1.0e-6;

        for (index, expected_force) in base.forces.iter().enumerate() {
            let mut plus = particles;
            let mut minus = particles;
            plus[index].position += step;
            minus[index].position -= step;
            let energy_plus =
                periodic_fourier_energy_and_forces(&plus, config(4)).unwrap().potential_energy;
            let energy_minus =
                periodic_fourier_energy_and_forces(&minus, config(4)).unwrap().potential_energy;
            let derivative = (energy_plus - energy_minus) / (2.0 * step);
            close(*expected_force, -derivative, 2.0e-8);
        }
    }

    #[test]
    fn invalid_particle_and_domain_inputs_fail_closed() {
        assert_eq!(
            periodic_fourier_energy_and_forces(
                &[Particle1d {
                    position: f64::NAN,
                    charge: 1.0,
                }],
                config(1)
            )
            .unwrap_err(),
            OracleError::NonFinitePosition { particle: 0 }
        );
        assert_eq!(
            periodic_fourier_energy_and_forces(
                &[Particle1d {
                    position: 0.0,
                    charge: f64::INFINITY,
                }],
                config(1)
            )
            .unwrap_err(),
            OracleError::NonFiniteParticleCharge { particle: 0 }
        );
        let mut invalid = config(1);
        invalid.domain_length = 0.0;
        assert_eq!(
            periodic_fourier_energy_and_forces(&[], invalid).unwrap_err(),
            OracleError::InvalidDomainLength
        );
    }

    #[test]
    fn manufactured_staggered_field_satisfies_discrete_gauss_law() {
        let field = [0.0, 1.0, 0.0, -1.0];
        let charge_density = [4.0, 4.0, -4.0, -4.0];
        let report = verify_periodic_1d_gauss_law(&field, &charge_density, 0.5, 2.0).unwrap();

        close(report.max_abs_residual, 0.0, 1.0e-14);
        close(report.rms_residual, 0.0, 1.0e-14);
        close(report.net_charge, 0.0, 1.0e-14);
        assert!(report.satisfies_tolerances(1.0e-12, 1.0e-12));
    }

    #[test]
    fn gauss_verifier_localises_a_perturbed_cell() {
        let field = [0.0, 1.0, 0.0, -1.0];
        let charge_density = [5.0, 4.0, -4.0, -4.0];
        let report = verify_periodic_1d_gauss_law(&field, &charge_density, 0.5, 2.0).unwrap();

        close(report.cell_residuals[0], -0.5, 1.0e-14);
        close(report.max_abs_residual, 0.5, 1.0e-14);
        assert!(!report.satisfies_tolerances(1.0e-3, 1.0e-3));
    }

    #[test]
    fn periodic_gauss_check_reports_non_neutral_charge() {
        let field = [0.0, 0.0, 0.0, 0.0];
        let charge_density = [1.0, 1.0, 1.0, 1.0];
        let report = verify_periodic_1d_gauss_law(&field, &charge_density, 0.5, 2.0).unwrap();

        close(report.net_charge, 2.0, 1.0e-14);
        assert!(!report.satisfies_tolerances(1.0, 1.0e-12));
    }

    #[test]
    fn gauss_verifier_rejects_shape_and_non_finite_inputs() {
        assert!(matches!(
            verify_periodic_1d_gauss_law(&[0.0, 0.0], &[0.0], 1.0, 1.0),
            Err(OracleError::FieldChargeLengthMismatch { .. })
        ));
        assert_eq!(
            verify_periodic_1d_gauss_law(&[0.0], &[0.0], 1.0, 1.0).unwrap_err(),
            OracleError::TooFewCells
        );
        assert_eq!(
            verify_periodic_1d_gauss_law(&[f64::NAN, 0.0], &[0.0, 0.0], 1.0, 1.0)
                .unwrap_err(),
            OracleError::NonFiniteFieldValue { cell: 0 }
        );
    }
}
