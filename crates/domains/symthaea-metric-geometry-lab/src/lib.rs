// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Small, auditable differential-geometry kernel for analytic 4D spacetimes.
//!
//! Implements:
//! - symmetric 4x4 metric tensors and pivoted matrix inversion;
//! - finite-difference metric derivatives;
//! - Christoffel symbols of the Levi-Civita connection;
//! - affine-parameter geodesic integration with RK4;
//! - Minkowski Cartesian and Schwarzschild spherical-coordinate metrics.
//!
//! Epistemic boundary: this is a reference geometry / geodesic kernel. It
//! does not solve the Einstein field equations, ADM constraints, matter
//! evolution, gravitational radiation, or a self-consistent source problem.
//! It is not a numerical-relativity replacement and does not validate
//! GravCraft's proposed metric ansatz.

#![forbid(unsafe_code)]

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

pub type Vector4 = [f64; 4];
pub type Matrix4 = [[f64; 4]; 4];
pub type Christoffel4 = [[[f64; 4]; 4]; 4];
/// R^rho_(sigma mu nu), with the derivative convention documented below.
pub type Riemann4 = [[[[f64; 4]; 4]; 4]; 4];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MetricInertia {
    pub negative: usize,
    pub zero: usize,
    pub positive: usize,
}

impl MetricInertia {
    pub fn is_lorentzian_minus_plus_plus_plus(self) -> bool {
        self.negative == 1 && self.zero == 0 && self.positive == 3
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CurvatureTensors4 {
    pub riemann: Riemann4,
    pub ricci: Matrix4,
    pub ricci_scalar: f64,
    pub kretschmann_scalar: f64,
}

const DERIVATIVE_RELATIVE_STEP: f64 = 1.0e-5;
const MATRIX_PIVOT_TOLERANCE: f64 = 1.0e-14;

#[derive(Clone, Debug, PartialEq)]
pub enum MetricError {
    NonFiniteCoordinate { axis: usize },
    InvalidMass,
    OutsideCoordinateChart(&'static str),
    NonFiniteMetricEntry { row: usize, column: usize },
    MetricNotSymmetric { row: usize, column: usize },
    SingularMetric,
    InvalidStep,
    NonFiniteTangent { axis: usize },
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteCoordinate { axis } => write!(f, "coordinate {axis} is not finite"),
            Self::InvalidMass => write!(f, "Schwarzschild mass must be finite and positive"),
            Self::OutsideCoordinateChart(message) => write!(f, "point lies outside metric chart: {message}"),
            Self::NonFiniteMetricEntry { row, column } => write!(f, "metric entry g[{row}][{column}] is not finite"),
            Self::MetricNotSymmetric { row, column } => write!(f, "metric is not symmetric at [{row}][{column}]"),
            Self::SingularMetric => write!(f, "metric matrix is singular or numerically ill-conditioned"),
            Self::InvalidStep => write!(f, "geodesic integration step must be finite and positive"),
            Self::NonFiniteTangent { axis } => write!(f, "geodesic tangent {axis} is not finite"),
        }
    }
}

impl Error for MetricError {}

/// Symmetric covariant metric tensor g_mu_nu in a declared coordinate chart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetricTensor4 {
    pub components: Matrix4,
}

impl MetricTensor4 {
    pub fn new(components: Matrix4) -> Result<Self, MetricError> {
        for row in 0..4 {
            for column in 0..4 {
                let value = components[row][column];
                if !value.is_finite() {
                    return Err(MetricError::NonFiniteMetricEntry { row, column });
                }
                let transpose = components[column][row];
                let tolerance = 1.0e-12 * value.abs().max(transpose.abs()).max(1.0);
                if (value - transpose).abs() > tolerance {
                    return Err(MetricError::MetricNotSymmetric { row, column });
                }
            }
        }
        Ok(Self { components })
    }

    /// Invert g_mu_nu with partial-pivoted Gauss-Jordan elimination.
    pub fn inverse(&self) -> Result<Matrix4, MetricError> {
        let mut augmented = [[0.0_f64; 8]; 4];
        for row in 0..4 {
            for column in 0..4 {
                augmented[row][column] = self.components[row][column];
            }
            augmented[row][row + 4] = 1.0;
        }

        for pivot_column in 0..4 {
            let mut pivot_row = pivot_column;
            let mut pivot_magnitude = augmented[pivot_row][pivot_column].abs();
            for row in (pivot_column + 1)..4 {
                let candidate = augmented[row][pivot_column].abs();
                if candidate > pivot_magnitude {
                    pivot_magnitude = candidate;
                    pivot_row = row;
                }
            }
            if !pivot_magnitude.is_finite() || pivot_magnitude <= MATRIX_PIVOT_TOLERANCE {
                return Err(MetricError::SingularMetric);
            }
            if pivot_row != pivot_column {
                augmented.swap(pivot_row, pivot_column);
            }

            let pivot = augmented[pivot_column][pivot_column];
            for column in 0..8 {
                augmented[pivot_column][column] /= pivot;
            }

            for row in 0..4 {
                if row == pivot_column {
                    continue;
                }
                let factor = augmented[row][pivot_column];
                for column in 0..8 {
                    augmented[row][column] -= factor * augmented[pivot_column][column];
                }
            }
        }

        let mut inverse = [[0.0_f64; 4]; 4];
        for row in 0..4 {
            for column in 0..4 {
                let value = augmented[row][column + 4];
                if !value.is_finite() {
                    return Err(MetricError::SingularMetric);
                }
                inverse[row][column] = value;
            }
        }
        Ok(inverse)
    }

    /// Count negative, zero and positive eigenvalues using a symmetric
    /// Jacobi diagonalization. The result is invariant under nonsingular
    /// coordinate changes by Sylvester's law of inertia.
    pub fn inertia(&self) -> MetricInertia {
        let mut matrix = self.components;
        let scale = matrix
            .iter()
            .flat_map(|row| row.iter())
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        if scale == 0.0 {
            return MetricInertia {
                negative: 0,
                zero: 4,
                positive: 0,
            };
        }
        let tolerance = 1.0e-13 * scale;

        for _ in 0..128 {
            let mut p = 0;
            let mut q = 1;
            let mut largest = matrix[p][q].abs();
            for row in 0..4 {
                for column in (row + 1)..4 {
                    if matrix[row][column].abs() > largest {
                        largest = matrix[row][column].abs();
                        p = row;
                        q = column;
                    }
                }
            }
            if largest <= tolerance {
                break;
            }

            let off_diagonal = matrix[p][q];
            let tau = (matrix[q][q] - matrix[p][p]) / (2.0 * off_diagonal);
            let denominator = tau.abs() + tau.hypot(1.0);
            let t = if tau >= 0.0 {
                1.0 / denominator
            } else {
                -1.0 / denominator
            };
            let cosine = 1.0 / (1.0 + t * t).sqrt();
            let sine = t * cosine;
            let pp = matrix[p][p];
            let qq = matrix[q][q];

            for index in 0..4 {
                if index == p || index == q {
                    continue;
                }
                let ip = matrix[index][p];
                let iq = matrix[index][q];
                matrix[index][p] = cosine * ip - sine * iq;
                matrix[p][index] = matrix[index][p];
                matrix[index][q] = sine * ip + cosine * iq;
                matrix[q][index] = matrix[index][q];
            }

            matrix[p][p] = pp - t * off_diagonal;
            matrix[q][q] = qq + t * off_diagonal;
            matrix[p][q] = 0.0;
            matrix[q][p] = 0.0;
        }

        let mut inertia = MetricInertia {
            negative: 0,
            zero: 0,
            positive: 0,
        };
        for axis in 0..4 {
            let eigenvalue = matrix[axis][axis];
            if eigenvalue < -tolerance {
                inertia.negative += 1;
            } else if eigenvalue > tolerance {
                inertia.positive += 1;
            } else {
                inertia.zero += 1;
            }
        }
        inertia
    }

    /// True exactly when the numerical inertia is (-,+,+,+), within the
    /// relative eigenvalue tolerance used by the inertia calculation.
    pub fn is_lorentzian_minus_plus_plus_plus(&self) -> bool {
        self.inertia().is_lorentzian_minus_plus_plus_plus()
    }

    /// Contract g_mu_nu u^mu v^nu.
    pub fn inner_product(&self, left: Vector4, right: Vector4) -> f64 {
        let mut result = 0.0;
        for mu in 0..4 {
            for nu in 0..4 {
                result += self.components[mu][nu] * left[mu] * right[nu];
            }
        }
        result
    }
}

/// A coordinate-dependent metric field.
pub trait MetricField {
    /// Evaluate covariant g_mu_nu at coordinates [x0,x1,x2,x3].
    fn metric(&self, coordinates: Vector4) -> Result<MetricTensor4, MetricError>;

    /// Central-difference derivative step. Override for unusual coordinate scales.
    fn derivative_step(&self, axis: usize, coordinate: f64) -> f64 {
        let _ = axis;
        DERIVATIVE_RELATIVE_STEP * coordinate.abs().max(1.0)
    }
}

/// Minkowski spacetime in Cartesian coordinates, signature (-,+,+,+).
#[derive(Clone, Copy, Debug, Default)]
pub struct MinkowskiCartesian;

impl MetricField for MinkowskiCartesian {
    fn metric(&self, coordinates: Vector4) -> Result<MetricTensor4, MetricError> {
        validate_coordinates(coordinates)?;
        MetricTensor4::new([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    fn derivative_step(&self, _axis: usize, _coordinate: f64) -> f64 {
        DERIVATIVE_RELATIVE_STEP
    }
}

/// Schwarzschild exterior in standard Schwarzschild spherical coordinates.
///
/// Coordinates are (t, r, theta, phi), with G=c=1 and signature (-,+,+,+).
/// Valid only for r>2M and 0<theta<pi; horizon and polar coordinate
/// singularities are intentionally excluded from this first reference kernel.
#[derive(Clone, Copy, Debug)]
pub struct SchwarzschildExterior {
    pub mass: f64,
}

impl SchwarzschildExterior {
    pub fn new(mass: f64) -> Result<Self, MetricError> {
        if !mass.is_finite() || mass <= 0.0 {
            return Err(MetricError::InvalidMass);
        }
        Ok(Self { mass })
    }
}

impl MetricField for SchwarzschildExterior {
    fn metric(&self, coordinates: Vector4) -> Result<MetricTensor4, MetricError> {
        validate_coordinates(coordinates)?;
        let radius = coordinates[1];
        let theta = coordinates[2];
        if radius <= 2.0 * self.mass {
            return Err(MetricError::OutsideCoordinateChart("r must be greater than 2M"));
        }
        if theta <= 0.0 || theta >= PI {
            return Err(MetricError::OutsideCoordinateChart("theta must lie strictly between 0 and pi"));
        }

        let f = 1.0 - 2.0 * self.mass / radius;
        let sin_theta = theta.sin();
        let mut metric = [[0.0_f64; 4]; 4];
        metric[0][0] = -f;
        metric[1][1] = 1.0 / f;
        metric[2][2] = radius * radius;
        metric[3][3] = radius * radius * sin_theta * sin_theta;
        MetricTensor4::new(metric)
    }
}

fn validate_coordinates(coordinates: Vector4) -> Result<(), MetricError> {
    for (axis, value) in coordinates.iter().enumerate() {
        if !value.is_finite() {
            return Err(MetricError::NonFiniteCoordinate { axis });
        }
    }
    Ok(())
}

/// Compute partial_alpha g_mu_nu by centered finite differences.
pub fn metric_derivatives<M: MetricField>(
    field: &M,
    coordinates: Vector4,
) -> Result<[Matrix4; 4], MetricError> {
    validate_coordinates(coordinates)?;
    let mut derivatives = [[[0.0_f64; 4]; 4]; 4];
    for axis in 0..4 {
        let step = field.derivative_step(axis, coordinates[axis]);
        if !step.is_finite() || step <= 0.0 {
            return Err(MetricError::InvalidStep);
        }
        let mut plus = coordinates;
        let mut minus = coordinates;
        plus[axis] += step;
        minus[axis] -= step;
        let metric_plus = field.metric(plus)?;
        let metric_minus = field.metric(minus)?;
        for mu in 0..4 {
            for nu in 0..4 {
                derivatives[axis][mu][nu] =
                    (metric_plus.components[mu][nu] - metric_minus.components[mu][nu])
                        / (2.0 * step);
            }
        }
    }
    Ok(derivatives)
}

/// Levi-Civita connection Gamma^mu_(alpha beta), computed from metric derivatives.
pub fn christoffel_symbols<M: MetricField>(
    field: &M,
    coordinates: Vector4,
) -> Result<Christoffel4, MetricError> {
    let metric = field.metric(coordinates)?;
    let inverse = metric.inverse()?;
    let derivatives = metric_derivatives(field, coordinates)?;
    let mut gamma = [[[0.0_f64; 4]; 4]; 4];

    for mu in 0..4 {
        for alpha in 0..4 {
            for beta in 0..4 {
                let mut sum = 0.0;
                for nu in 0..4 {
                    sum += inverse[mu][nu]
                        * (derivatives[alpha][nu][beta]
                            + derivatives[beta][nu][alpha]
                            - derivatives[nu][alpha][beta]);
                }
                gamma[mu][alpha][beta] = 0.5 * sum;
            }
        }
    }
    Ok(gamma)
}

/// Riemann tensor with convention
///
/// R^rho_(sigma mu nu) = partial_mu Gamma^rho_(nu sigma)
///   - partial_nu Gamma^rho_(mu sigma)
///   + Gamma^rho_(mu lambda) Gamma^lambda_(nu sigma)
///   - Gamma^rho_(nu lambda) Gamma^lambda_(mu sigma).
///
/// Metric derivatives are centered finite differences. Curvature therefore
/// has one additional numerical differentiation and must be convergence-tested.
pub fn riemann_tensor<M: MetricField>(
    field: &M,
    coordinates: Vector4,
) -> Result<Riemann4, MetricError> {
    validate_coordinates(coordinates)?;
    let gamma = christoffel_symbols(field, coordinates)?;
    let mut gamma_derivatives = [[[[0.0_f64; 4]; 4]; 4]; 4];

    for axis in 0..4 {
        let step = field.derivative_step(axis, coordinates[axis]);
        if !step.is_finite() || step <= 0.0 {
            return Err(MetricError::InvalidStep);
        }
        let mut plus = coordinates;
        let mut minus = coordinates;
        plus[axis] += step;
        minus[axis] -= step;
        let gamma_plus = christoffel_symbols(field, plus)?;
        let gamma_minus = christoffel_symbols(field, minus)?;
        for rho in 0..4 {
            for sigma in 0..4 {
                for nu in 0..4 {
                    gamma_derivatives[axis][rho][nu][sigma] =
                        (gamma_plus[rho][nu][sigma] - gamma_minus[rho][nu][sigma])
                            / (2.0 * step);
                }
            }
        }
    }

    let mut riemann = [[[[0.0_f64; 4]; 4]; 4]; 4];
    for rho in 0..4 {
        for sigma in 0..4 {
            for mu in 0..4 {
                for nu in 0..4 {
                    let mut value = gamma_derivatives[mu][rho][nu][sigma]
                        - gamma_derivatives[nu][rho][mu][sigma];
                    for lambda in 0..4 {
                        value += gamma[rho][mu][lambda] * gamma[lambda][nu][sigma]
                            - gamma[rho][nu][lambda] * gamma[lambda][mu][sigma];
                    }
                    riemann[rho][sigma][mu][nu] = value;
                }
            }
        }
    }
    Ok(riemann)
}

/// Compute Riemann, Ricci, Ricci scalar and the Kretschmann invariant.
///
/// The Kretschmann scalar is R_abcd R^abcd. In vacuum Schwarzschild,
/// R_mu_nu and R vanish while K = 48 M^2/r^6.
pub fn curvature_tensors<M: MetricField>(
    field: &M,
    coordinates: Vector4,
) -> Result<CurvatureTensors4, MetricError> {
    let metric = field.metric(coordinates)?;
    let inverse = metric.inverse()?;
    let riemann = riemann_tensor(field, coordinates)?;

    let mut ricci = [[0.0_f64; 4]; 4];
    for sigma in 0..4 {
        for nu in 0..4 {
            for rho in 0..4 {
                ricci[sigma][nu] += riemann[rho][sigma][rho][nu];
            }
        }
    }

    let mut ricci_scalar = 0.0;
    for mu in 0..4 {
        for nu in 0..4 {
            ricci_scalar += inverse[mu][nu] * ricci[mu][nu];
        }
    }

    let mut riemann_lower = [[[[0.0_f64; 4]; 4]; 4]; 4];
    for alpha in 0..4 {
        for beta in 0..4 {
            for gamma in 0..4 {
                for delta in 0..4 {
                    for rho in 0..4 {
                        riemann_lower[alpha][beta][gamma][delta] +=
                            metric.components[alpha][rho]
                                * riemann[rho][beta][gamma][delta];
                    }
                }
            }
        }
    }

    let mut kretschmann_scalar = 0.0;
    for alpha in 0..4 {
        for beta in 0..4 {
            for gamma in 0..4 {
                for delta in 0..4 {
                    for a in 0..4 {
                        for b in 0..4 {
                            for c in 0..4 {
                                for d in 0..4 {
                                    kretschmann_scalar += riemann_lower[alpha][beta][gamma][delta]
                                        * inverse[alpha][a]
                                        * inverse[beta][b]
                                        * inverse[gamma][c]
                                        * inverse[delta][d]
                                        * riemann_lower[a][b][c][d];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(CurvatureTensors4 {
        riemann,
        ricci,
        ricci_scalar,
        kretschmann_scalar,
    })
}

/// A point and tangent of a timelike, null or spacelike geodesic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeodesicState {
    pub coordinates: Vector4,
    pub tangent: Vector4,
}

impl GeodesicState {
    pub fn validate(&self) -> Result<(), MetricError> {
        validate_coordinates(self.coordinates)?;
        for (axis, value) in self.tangent.iter().enumerate() {
            if !value.is_finite() {
                return Err(MetricError::NonFiniteTangent { axis });
            }
        }
        Ok(())
    }

    pub fn norm_squared<M: MetricField>(&self, field: &M) -> Result<f64, MetricError> {
        Ok(field.metric(self.coordinates)?.inner_product(self.tangent, self.tangent))
    }
}

#[derive(Clone, Copy)]
struct GeodesicDerivative {
    coordinates: Vector4,
    tangent: Vector4,
}

fn geodesic_derivative<M: MetricField>(
    field: &M,
    state: GeodesicState,
) -> Result<GeodesicDerivative, MetricError> {
    let gamma = christoffel_symbols(field, state.coordinates)?;
    let mut acceleration = [0.0_f64; 4];
    for mu in 0..4 {
        let mut value = 0.0;
        for alpha in 0..4 {
            for beta in 0..4 {
                value -= gamma[mu][alpha][beta]
                    * state.tangent[alpha]
                    * state.tangent[beta];
            }
        }
        acceleration[mu] = value;
    }
    Ok(GeodesicDerivative {
        coordinates: state.tangent,
        tangent: acceleration,
    })
}

fn offset_state(
    state: GeodesicState,
    derivative: GeodesicDerivative,
    scale: f64,
) -> GeodesicState {
    let mut next = state;
    for axis in 0..4 {
        next.coordinates[axis] += scale * derivative.coordinates[axis];
        next.tangent[axis] += scale * derivative.tangent[axis];
    }
    next
}

/// One RK4 step of the affinely parameterized geodesic equation.
///
/// d2 x^mu/dlambda2 + Gamma^mu_(alpha beta) dx^alpha/dlambda dx^beta/dlambda = 0.
pub fn geodesic_rk4_step<M: MetricField>(
    field: &M,
    state: &mut GeodesicState,
    step: f64,
) -> Result<(), MetricError> {
    if !step.is_finite() || step <= 0.0 {
        return Err(MetricError::InvalidStep);
    }
    state.validate()?;

    let k1 = geodesic_derivative(field, *state)?;
    let k2 = geodesic_derivative(field, offset_state(*state, k1, 0.5 * step))?;
    let k3 = geodesic_derivative(field, offset_state(*state, k2, 0.5 * step))?;
    let k4 = geodesic_derivative(field, offset_state(*state, k3, step))?;

    let mut next = *state;
    for axis in 0..4 {
        next.coordinates[axis] += step / 6.0
            * (k1.coordinates[axis]
                + 2.0 * k2.coordinates[axis]
                + 2.0 * k3.coordinates[axis]
                + k4.coordinates[axis]);
        next.tangent[axis] += step / 6.0
            * (k1.tangent[axis]
                + 2.0 * k2.tangent[axis]
                + 2.0 * k3.tangent[axis]
                + k4.tangent[axis]);
    }
    next.validate()?;
    *state = next;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_abs_gamma(gamma: Christoffel4) -> f64 {
        gamma
            .iter()
            .flat_map(|plane| plane.iter())
            .flat_map(|row| row.iter())
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()))
    }

    #[test]
    fn minkowski_metric_inverse_and_connection_are_exact() {
        let field = MinkowskiCartesian;
        let point = [2.0, 3.0, -4.0, 1.5];
        let metric = field.metric(point).unwrap();
        let inverse = metric.inverse().unwrap();
        for row in 0..4 {
            for column in 0..4 {
                let expected = if row == column { 1.0 } else { 0.0 };
                let mut product = 0.0;
                for inner in 0..4 {
                    product += metric.components[row][inner] * inverse[inner][column];
                }
                assert!((product - expected).abs() < 1e-14);
            }
        }
        assert!(max_abs_gamma(christoffel_symbols(&field, point).unwrap()) < 1e-12);
    }

    #[test]
    fn schwarzschild_christoffels_match_analytic_components() {
        let mass = 1.0;
        let radius = 10.0;
        let theta = PI / 2.0;
        let field = SchwarzschildExterior::new(mass).unwrap();
        let gamma = christoffel_symbols(&field, [0.0, radius, theta, 0.0]).unwrap();
        let f = 1.0 - 2.0 * mass / radius;
        let expected_t_tr = mass / (radius * radius * f);
        let expected_r_tt = f * mass / (radius * radius);
        let expected_r_rr = -mass / (radius * radius * f);
        let expected_r_theta_theta = -radius * f;
        let expected_theta_r_theta = 1.0 / radius;

        assert!((gamma[0][0][1] - expected_t_tr).abs() < 1e-9);
        assert!((gamma[0][1][0] - expected_t_tr).abs() < 1e-9);
        assert!((gamma[1][0][0] - expected_r_tt).abs() < 1e-9);
        assert!((gamma[1][1][1] - expected_r_rr).abs() < 1e-9);
        assert!((gamma[1][2][2] - expected_r_theta_theta).abs() < 1e-8);
        assert!((gamma[2][1][2] - expected_theta_r_theta).abs() < 1e-9);
    }

    #[test]
    fn metric_inertia_identifies_lorentzian_signature() {
        let minkowski = MinkowskiCartesian.metric([0.0; 4]).unwrap();
        assert_eq!(
            minkowski.inertia(),
            MetricInertia { negative: 1, zero: 0, positive: 3 }
        );
        assert!(minkowski.is_lorentzian_minus_plus_plus_plus());

        let schwarzschild = SchwarzschildExterior::new(1.0).unwrap();
        let exterior = schwarzschild.metric([0.0, 10.0, PI / 2.0, 0.0]).unwrap();
        assert_eq!(
            exterior.inertia(),
            MetricInertia { negative: 1, zero: 0, positive: 3 }
        );

        let euclidean = MetricTensor4::new([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .unwrap();
        assert!(!euclidean.is_lorentzian_minus_plus_plus_plus());
    }

    #[test]
    fn minkowski_curvature_is_zero() {
        let field = MinkowskiCartesian;
        let curvature = curvature_tensors(&field, [2.0, 3.0, -4.0, 1.5]).unwrap();
        assert!(curvature.ricci_scalar.abs() < 1e-10);
        assert!(curvature.kretschmann_scalar.abs() < 1e-10);
        let max_riemann = curvature
            .riemann
            .iter()
            .flat_map(|a| a.iter())
            .flat_map(|b| b.iter())
            .flat_map(|c| c.iter())
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        assert!(max_riemann < 1e-10);
    }

    #[test]
    fn schwarzschild_exterior_is_vacuum_and_kretschmann_matches_analytic_value() {
        let mass = 1.0;
        let radius: f64 = 10.0;
        let field = SchwarzschildExterior::new(mass).unwrap();
        let curvature =
            curvature_tensors(&field, [0.0, radius, PI / 2.0, 0.0]).unwrap();

        let max_ricci = curvature
            .ricci
            .iter()
            .flat_map(|row| row.iter())
            .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
        assert!(max_ricci < 1e-6, "max Ricci component={max_ricci}");
        assert!(curvature.ricci_scalar.abs() < 1e-7);
        let expected = 48.0 * mass * mass / radius.powi(6);
        assert!(
            (curvature.kretschmann_scalar - expected).abs() < 1e-8,
            "K={}, expected={expected}",
            curvature.kretschmann_scalar
        );
    }

    #[test]
    fn flat_spacetime_geodesic_moves_linearly() {
        let field = MinkowskiCartesian;
        let mut state = GeodesicState {
            coordinates: [0.0, 1.0, 2.0, 3.0],
            tangent: [1.0, 0.2, -0.1, 0.3],
        };
        for _ in 0..100 {
            geodesic_rk4_step(&field, &mut state, 0.01).unwrap();
        }
        assert!((state.coordinates[0] - 1.0).abs() < 1e-11);
        assert!((state.coordinates[1] - 1.2).abs() < 1e-11);
        assert!((state.coordinates[2] - 1.9).abs() < 1e-11);
        assert!((state.coordinates[3] - 3.3).abs() < 1e-11);
        assert!((state.tangent[0] - 1.0).abs() < 1e-11);
    }

    #[test]
    fn schwarzschild_timelike_geodesic_preserves_normalization() {
        let field = SchwarzschildExterior::new(1.0).unwrap();
        let radius = 10.0;
        let radial_tangent = -0.05;
        let f = 1.0 - 2.0 / radius;
        let time_tangent = ((1.0 + radial_tangent * radial_tangent / f) / f).sqrt();
        let mut state = GeodesicState {
            coordinates: [0.0, radius, PI / 2.0, 0.0],
            tangent: [time_tangent, radial_tangent, 0.0, 0.0],
        };
        let initial_norm = state.norm_squared(&field).unwrap();
        assert!((initial_norm + 1.0).abs() < 1e-12);

        for _ in 0..1_000 {
            geodesic_rk4_step(&field, &mut state, 1e-3).unwrap();
        }
        let final_norm = state.norm_squared(&field).unwrap();
        assert!((final_norm + 1.0).abs() < 1e-7, "final norm={final_norm}");
        assert!(state.coordinates[1] < radius);
    }

    #[test]
    fn coordinate_chart_and_singular_metrics_fail_closed() {
        let schwarzschild = SchwarzschildExterior::new(1.0).unwrap();
        assert!(schwarzschild.metric([0.0, 2.0, PI / 2.0, 0.0]).is_err());
        assert!(schwarzschild.metric([0.0, 10.0, 0.0, 0.0]).is_err());
        let singular = MetricTensor4::new([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        .unwrap();
        assert!(matches!(singular.inverse(), Err(MetricError::SingularMetric)));
    }
}
