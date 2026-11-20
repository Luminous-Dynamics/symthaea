//! Residual-to-error certificates for one pinned periodic 1D Poisson stencil.
//!
//! The certificate applies to the mean-zero subspace of the uniform periodic
//! second-order finite-difference Laplacian. It is not a generic Poisson
//! certificate and does not directly bound pointwise particle forces.

#![forbid(unsafe_code)]

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

/// Errors produced by the periodic Poisson certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoissonCertificateError {
    LengthMismatch { potential: usize, charge: usize },
    TooFewCells,
    InvalidCellWidth,
    InvalidPermittivity,
    InvalidNeutralityTolerance,
    InvalidGaugeTolerance,
    NonFinitePotential { cell: usize },
    NonFiniteChargeDensity { cell: usize },
    NonFiniteIntermediate,
    ProjectionExceedsTolerance { kind: &'static str },
}

impl fmt::Display for PoissonCertificateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { potential, charge } => {
                write!(f, "potential has {potential} cells but charge has {charge} cells")
            }
            Self::TooFewCells => write!(f, "periodic Poisson certificate requires at least 3 cells"),
            Self::InvalidCellWidth => write!(f, "cell width must be finite and positive"),
            Self::InvalidPermittivity => write!(f, "permittivity must be finite and positive"),
            Self::InvalidNeutralityTolerance => {
                write!(f, "neutrality tolerance must be finite and non-negative")
            }
            Self::InvalidGaugeTolerance => {
                write!(f, "gauge tolerance must be finite and non-negative")
            }
            Self::NonFinitePotential { cell } => {
                write!(f, "potential estimate in cell {cell} is non-finite")
            }
            Self::NonFiniteChargeDensity { cell } => {
                write!(f, "charge density in cell {cell} is non-finite")
            }
            Self::NonFiniteIntermediate => write!(f, "Poisson certificate intermediate is non-finite"),
            Self::ProjectionExceedsTolerance { kind } => {
                write!(f, "{kind} zero-mode projection exceeds the declared tolerance")
            }
        }
    }
}

impl Error for PoissonCertificateError {}

/// A conservative residual-based error certificate for a periodic 1D solve.
///
/// Norms are Euclidean vector norms (not dx-weighted norms). The reported
/// bounds apply after the explicit zero-mode and gauge projections in this
/// record.
#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicPoissonCertificate {
    pub cell_count: usize,
    pub cell_width: f64,
    pub lambda_min_nonzero: f64,
    pub residual_l2: f64,
    pub potential_error_l2_bound: f64,
    pub electric_field_error_l2_bound: f64,
    pub removed_charge_density_mean: f64,
    pub removed_potential_mean: f64,
    pub net_charge_before_projection: f64,
}

/// Certify a potential estimate against the periodic second-order Poisson
/// operator
///
///     A phi[i] = (2 phi[i] - phi[i-1] - phi[i+1]) / dx^2.
///
/// The right-hand side is rho/epsilon_0. On the mean-zero subspace the
/// smallest eigenvalue is
///
///     lambda_1 = 4 sin^2(pi/N) / dx^2.
///
/// The residual r = rho_projected/epsilon_0 - A phi_projected gives
///
///     ||phi - phi_hat||_2 <= ||r||_2 / lambda_1
///
/// and, for E[i] = -(phi[i+1]-phi[i])/dx,
///
///     ||E - E_hat||_2 <= ||r||_2 / sqrt(lambda_1).
///
/// The input charge-density mean and potential-estimate mean are projected
/// out only when their magnitudes are within the caller-declared tolerances.
/// Both removed means and the pre-projection net charge are returned. This
/// certificate concerns the projected periodic problem; it does not silently
/// discard a large non-neutral component.
pub fn certify_periodic_1d_poisson(
    potential_estimate: &[f64],
    charge_density: &[f64],
    cell_width: f64,
    permittivity: f64,
    neutrality_tolerance: f64,
    gauge_tolerance: f64,
) -> Result<PeriodicPoissonCertificate, PoissonCertificateError> {
    if potential_estimate.len() != charge_density.len() {
        return Err(PoissonCertificateError::LengthMismatch {
            potential: potential_estimate.len(),
            charge: charge_density.len(),
        });
    }
    let cell_count = potential_estimate.len();
    if cell_count < 3 {
        return Err(PoissonCertificateError::TooFewCells);
    }
    if !cell_width.is_finite() || cell_width <= 0.0 {
        return Err(PoissonCertificateError::InvalidCellWidth);
    }
    if !permittivity.is_finite() || permittivity <= 0.0 {
        return Err(PoissonCertificateError::InvalidPermittivity);
    }
    if !neutrality_tolerance.is_finite() || neutrality_tolerance < 0.0 {
        return Err(PoissonCertificateError::InvalidNeutralityTolerance);
    }
    if !gauge_tolerance.is_finite() || gauge_tolerance < 0.0 {
        return Err(PoissonCertificateError::InvalidGaugeTolerance);
    }

    for (cell, value) in potential_estimate.iter().enumerate() {
        if !value.is_finite() {
            return Err(PoissonCertificateError::NonFinitePotential { cell });
        }
    }
    for (cell, value) in charge_density.iter().enumerate() {
        if !value.is_finite() {
            return Err(PoissonCertificateError::NonFiniteChargeDensity { cell });
        }
    }

    let n = cell_count as f64;
    let potential_mean = potential_estimate.iter().sum::<f64>() / n;
    let charge_mean = charge_density.iter().sum::<f64>() / n;
    if !potential_mean.is_finite() || !charge_mean.is_finite() {
        return Err(PoissonCertificateError::NonFiniteIntermediate);
    }
    if potential_mean.abs() > gauge_tolerance {
        return Err(PoissonCertificateError::ProjectionExceedsTolerance { kind: "potential gauge" });
    }
    if charge_mean.abs() > neutrality_tolerance {
        return Err(PoissonCertificateError::ProjectionExceedsTolerance { kind: "charge neutrality" });
    }

    let potential: Vec<f64> = potential_estimate
        .iter()
        .map(|value| value - potential_mean)
        .collect();
    let charge: Vec<f64> = charge_density
        .iter()
        .map(|value| value - charge_mean)
        .collect();
    let inverse_dx_squared = 1.0 / (cell_width * cell_width);
    if !inverse_dx_squared.is_finite() {
        return Err(PoissonCertificateError::NonFiniteIntermediate);
    }

    let mut residual_squared = 0.0;
    for (cell, (&phi, &rho)) in potential.iter().zip(charge.iter()).enumerate() {
        let previous = potential[(cell + cell_count - 1) % cell_count];
        let next = potential[(cell + 1) % cell_count];
        let discrete_laplacian = (2.0 * phi - previous - next) * inverse_dx_squared;
        let residual = rho / permittivity - discrete_laplacian;
        residual_squared += residual * residual;
    }

    let residual_l2 = residual_squared.sqrt();
    let lambda_min_nonzero =
        4.0 * (PI / n).sin().powi(2) * inverse_dx_squared;
    if !residual_l2.is_finite()
        || !lambda_min_nonzero.is_finite()
        || lambda_min_nonzero <= 0.0
    {
        return Err(PoissonCertificateError::NonFiniteIntermediate);
    }

    let potential_error_l2_bound = residual_l2 / lambda_min_nonzero;
    let electric_field_error_l2_bound = residual_l2 / lambda_min_nonzero.sqrt();
    let net_charge_before_projection = charge_mean * n * cell_width;
    if !potential_error_l2_bound.is_finite()
        || !electric_field_error_l2_bound.is_finite()
        || !net_charge_before_projection.is_finite()
    {
        return Err(PoissonCertificateError::NonFiniteIntermediate);
    }

    Ok(PeriodicPoissonCertificate {
        cell_count,
        cell_width,
        lambda_min_nonzero,
        residual_l2,
        potential_error_l2_bound,
        electric_field_error_l2_bound,
        removed_charge_density_mean: charge_mean,
        removed_potential_mean: potential_mean,
        net_charge_before_projection,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn l2(values: &[f64]) -> f64 {
        values.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    fn mode(count: usize, mode_number: usize, amplitude: f64) -> Vec<f64> {
        (0..count)
            .map(|cell| {
                amplitude
                    * (2.0 * PI * mode_number as f64 * cell as f64 / count as f64).cos()
            })
            .collect()
    }

    fn discrete_lambda(count: usize, mode_number: usize, cell_width: f64) -> f64 {
        4.0 * (PI * mode_number as f64 / count as f64).sin().powi(2)
            / (cell_width * cell_width)
    }

    #[test]
    fn exact_manufactured_mode_has_zero_residual_up_to_roundoff() {
        let count = 32;
        let dx = 0.2;
        let epsilon = 2.0;
        let potential = mode(count, 1, 0.75);
        let lambda = discrete_lambda(count, 1, dx);
        let charge: Vec<f64> = potential.iter().map(|value| epsilon * lambda * value).collect();

        let certificate =
            certify_periodic_1d_poisson(&potential, &charge, dx, epsilon, 1.0e-12, 1.0e-12)
                .unwrap();
        assert!(certificate.residual_l2 < 1.0e-11);
        assert!(certificate.potential_error_l2_bound < 1.0e-9);
        assert!(certificate.electric_field_error_l2_bound < 1.0e-9);
    }

    #[test]
    fn residual_bounds_actual_potential_and_forward_field_errors() {
        let count = 64;
        let dx = 0.125;
        let epsilon = 1.5;
        let exact = mode(count, 2, 1.0);
        let perturbation = mode(count, 1, 0.05);
        let estimate: Vec<f64> = exact
            .iter()
            .zip(perturbation.iter())
            .map(|(truth, error)| truth + error)
            .collect();
        let charge: Vec<f64> = exact
            .iter()
            .map(|value| epsilon * discrete_lambda(count, 2, dx) * value)
            .collect();

        let certificate =
            certify_periodic_1d_poisson(&estimate, &charge, dx, epsilon, 1.0e-12, 1.0e-12)
                .unwrap();

        let potential_error = l2(&perturbation);
        let field_error_values: Vec<f64> = perturbation
            .iter()
            .enumerate()
            .map(|(cell, value)| {
                let next = (cell + 1) % count;
                -(perturbation[next] - value) / dx
            })
            .collect();
        let field_error = l2(&field_error_values);

        assert!(potential_error <= certificate.potential_error_l2_bound + 1.0e-12);
        assert!(field_error <= certificate.electric_field_error_l2_bound + 1.0e-12);
    }

    #[test]
    fn reports_small_neutrality_and_gauge_projections() {
        let count = 16;
        let dx = 0.25;
        let mut potential = mode(count, 1, 1.0);
        for value in &mut potential {
            *value += 1.0e-8;
        }
        let mut charge = mode(count, 1, 1.0);
        for value in &mut charge {
            *value += 1.0e-9;
        }

        let certificate = certify_periodic_1d_poisson(
            &potential,
            &charge,
            dx,
            1.0,
            1.0e-8,
            1.0e-7,
        )
        .unwrap();
        assert!((certificate.removed_potential_mean - 1.0e-8).abs() < 1.0e-14);
        assert!((certificate.removed_charge_density_mean - 1.0e-9).abs() < 1.0e-14);
        assert!(
            (certificate.net_charge_before_projection - 1.0e-9 * count as f64 * dx).abs()
                < 1.0e-14
        );
    }

    #[test]
    fn rejects_non_neutral_source_and_non_zero_gauge_outside_tolerance() {
        let potential = mode(8, 1, 1.0);
        let non_neutral = vec![1.0; 8];
        assert_eq!(
            certify_periodic_1d_poisson(&potential, &non_neutral, 1.0, 1.0, 1.0e-12, 1.0)
                .unwrap_err(),
            PoissonCertificateError::ProjectionExceedsTolerance {
                kind: "charge neutrality"
            }
        );

        let gauged: Vec<f64> = potential.iter().map(|value| value + 0.1).collect();
        let neutral = mode(8, 1, 1.0);
        assert_eq!(
            certify_periodic_1d_poisson(&gauged, &neutral, 1.0, 1.0, 1.0, 1.0e-3)
                .unwrap_err(),
            PoissonCertificateError::ProjectionExceedsTolerance {
                kind: "potential gauge"
            }
        );
    }

    #[test]
    fn rejects_shape_and_invalid_numeric_inputs() {
        assert_eq!(
            certify_periodic_1d_poisson(&[0.0, 1.0], &[0.0], 1.0, 1.0, 0.0, 0.0).unwrap_err(),
            PoissonCertificateError::LengthMismatch {
                potential: 2,
                charge: 1
            }
        );
        assert_eq!(
            certify_periodic_1d_poisson(&[0.0, 1.0], &[0.0, 0.0], 1.0, 1.0, 0.0, 0.0)
                .unwrap_err(),
            PoissonCertificateError::TooFewCells
        );
        assert_eq!(
            certify_periodic_1d_poisson(&[0.0, f64::NAN, 1.0], &[0.0; 3], 1.0, 1.0, 0.0, 0.0)
                .unwrap_err(),
            PoissonCertificateError::NonFinitePotential { cell: 1 }
        );
        assert_eq!(
            certify_periodic_1d_poisson(
                &[0.0; 3],
                &[0.0, f64::INFINITY, 0.0],
                1.0,
                1.0,
                0.0,
                0.0
            )
            .unwrap_err(),
            PoissonCertificateError::NonFiniteChargeDensity { cell: 1 }
        );
    }
}
