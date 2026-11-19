//! Exact polynomial oracle for the SCI-SYMGRAM-001 radial-force case.
//!
//! This crate checks identities in a bounded polynomial family. It is a
//! development reference, not a physical simulator or a scientific discovery
//! policy. Integer coefficients keep all checks exact and dependency-free.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;

/// Exponents of (x1, x2, v1, v2), in that order.
type Monomial = [u8; 4];

/// Small sparse polynomial with exact integer coefficients.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Polynomial {
    terms: BTreeMap<Monomial, i128>,
}

impl Polynomial {
    fn zero() -> Self {
        Self::default()
    }

    fn constant(value: i128) -> Self {
        if value == 0 {
            return Self::zero();
        }
        let mut terms = BTreeMap::new();
        terms.insert([0; 4], value);
        Self { terms }
    }

    fn variable(index: usize) -> Self {
        assert!(index < 4, "state variable index must be in 0..4");
        let mut exponent = [0; 4];
        exponent[index] = 1;
        let mut terms = BTreeMap::new();
        terms.insert(exponent, 1);
        Self { terms }
    }

    fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    fn add(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (monomial, coefficient) in &other.terms {
            let sum = result
                .terms
                .get(monomial)
                .copied()
                .unwrap_or(0)
                .checked_add(*coefficient)
                .expect("polynomial coefficient overflow");
            if sum == 0 {
                result.terms.remove(monomial);
            } else {
                result.terms.insert(*monomial, sum);
            }
        }
        result
    }

    fn scale(&self, scalar: i128) -> Self {
        if scalar == 0 {
            return Self::zero();
        }
        let terms = self
            .terms
            .iter()
            .map(|(monomial, coefficient)| {
                (
                    *monomial,
                    coefficient
                        .checked_mul(scalar)
                        .expect("polynomial coefficient overflow"),
                )
            })
            .collect();
        Self { terms }
    }

    fn sub(&self, other: &Self) -> Self {
        self.add(&other.scale(-1))
    }

    fn mul(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for (left_monomial, left_coefficient) in &self.terms {
            for (right_monomial, right_coefficient) in &other.terms {
                let monomial = std::array::from_fn(|index| {
                    left_monomial[index]
                        .checked_add(right_monomial[index])
                        .expect("polynomial exponent overflow")
                });
                let product = left_coefficient
                    .checked_mul(*right_coefficient)
                    .expect("polynomial coefficient overflow");
                let sum = result
                    .terms
                    .get(&monomial)
                    .copied()
                    .unwrap_or(0)
                    .checked_add(product)
                    .expect("polynomial coefficient overflow");
                if sum == 0 {
                    result.terms.remove(&monomial);
                } else {
                    result.terms.insert(monomial, sum);
                }
            }
        }
        result
    }

    fn pow(&self, mut exponent: u32) -> Self {
        let mut result = Self::constant(1);
        let mut factor = self.clone();
        while exponent > 0 {
            if exponent & 1 == 1 {
                result = result.mul(&factor);
            }
            exponent >>= 1;
            if exponent > 0 {
                factor = factor.mul(&factor);
            }
        }
        result
    }

    fn derivative(&self, variable: usize) -> Self {
        assert!(variable < 4, "state variable index must be in 0..4");
        let mut result = Self::zero();
        for (monomial, coefficient) in &self.terms {
            let power = monomial[variable];
            if power == 0 {
                continue;
            }
            let mut derivative_monomial = *monomial;
            derivative_monomial[variable] -= 1;
            let derivative_coefficient = coefficient
                .checked_mul(i128::from(power))
                .expect("polynomial coefficient overflow");
            result.terms.insert(derivative_monomial, derivative_coefficient);
        }
        result
    }

    fn evaluate(&self, state: [i128; 4]) -> i128 {
        let mut total = 0_i128;
        for (monomial, coefficient) in &self.terms {
            let mut term = *coefficient;
            for (index, power) in monomial.iter().enumerate() {
                for _ in 0..*power {
                    term = term
                        .checked_mul(state[index])
                        .expect("polynomial evaluation overflow");
                }
            }
            total = total
                .checked_add(term)
                .expect("polynomial evaluation overflow");
        }
        total
    }

    /// Lie derivative of this scalar polynomial along a vector field.
    fn along(&self, vector_field: &[Self; 4]) -> Self {
        let mut result = Self::zero();
        for (index, component) in vector_field.iter().enumerate() {
            result = result.add(&self.derivative(index).mul(component));
        }
        result
    }
}

/// Failure to construct a supported exact reference case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationError {
    NonPositiveCoupling,
    CouplingOutsideBound { coupling: i128, maximum: i128 },
    ExponentOutsideBound { exponent: u32, maximum: u32 },
    StateCoordinateOutsideBound { index: usize, maximum_magnitude: i128 },
}

impl fmt::Display for QualificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveCoupling => write!(formatter, "coupling must be positive"),
            Self::CouplingOutsideBound { coupling, maximum } => {
                write!(formatter, "coupling {coupling} exceeds the supported maximum {maximum}")
            }
            Self::ExponentOutsideBound { exponent, maximum } => {
                write!(formatter, "exponent {exponent} exceeds the supported maximum {maximum}")
            }
            Self::StateCoordinateOutsideBound { index, maximum_magnitude } => {
                write!(formatter, "state coordinate {index} exceeds the supported magnitude {maximum_magnitude}")
            }
        }
    }
}

impl std::error::Error for QualificationError {}

/// Exact algebraic results for one radial-force development case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QualificationReport {
    pub exponent: u32,
    pub coupling: i128,
    pub vector_field_equivariant: bool,
    pub invariant_relation_holds: bool,
    pub quotient_dynamics_match: bool,
    pub angular_momentum_conserved: bool,
    pub energy_conserved: bool,
}

impl QualificationReport {
    /// True only when every declared exact identity passed.
    pub fn all_exact_checks_pass(&self) -> bool {
        self.vector_field_equivariant
            && self.invariant_relation_holds
            && self.quotient_dynamics_match
            && self.angular_momentum_conserved
            && self.energy_conserved
    }
}

fn radial_vector_field(exponent: u32, coupling: i128) -> [Polynomial; 4] {
    let x1 = Polynomial::variable(0);
    let x2 = Polynomial::variable(1);
    let v1 = Polynomial::variable(2);
    let v2 = Polynomial::variable(3);
    let radius_squared = x1.mul(&x1).add(&x2.mul(&x2));
    let radial_gain = radius_squared.pow(exponent).scale(coupling);
    [
        v1,
        v2,
        radial_gain.mul(&x1).scale(-1),
        radial_gain.mul(&x2).scale(-1),
    ]
}

fn anisotropic_oscillator(k1: i128, k2: i128) -> [Polynomial; 4] {
    let x1 = Polynomial::variable(0);
    let x2 = Polynomial::variable(1);
    let v1 = Polynomial::variable(2);
    let v2 = Polynomial::variable(3);
    [v1, v2, x1.scale(-k1), x2.scale(-k2)]
}

fn rotation_generator_action() -> [Polynomial; 4] {
    let x1 = Polynomial::variable(0);
    let x2 = Polynomial::variable(1);
    let v1 = Polynomial::variable(2);
    let v2 = Polynomial::variable(3);

    // A z for A = diag(J, J), J = [[0, -1], [1, 0]].
    [x2.scale(-1), x1, v2.scale(-1), v1]
}

fn scalar_is_rotation_invariant(expression: &Polynomial) -> bool {
    expression.along(&rotation_generator_action()).is_zero()
}

fn infinitesimal_rotation_residuals(vector_field: &[Polynomial; 4]) -> [Polynomial; 4] {
    let generator_action = rotation_generator_action();

    // A F(z), using the same block generator.
    let generator_on_field = [
        vector_field[1].scale(-1),
        vector_field[0].clone(),
        vector_field[3].scale(-1),
        vector_field[2].clone(),
    ];

    std::array::from_fn(|index| {
        vector_field[index]
            .along(&generator_action)
            .sub(&generator_on_field[index])
    })
}

fn invariant_polynomials() -> (Polynomial, Polynomial, Polynomial, Polynomial) {
    let x1 = Polynomial::variable(0);
    let x2 = Polynomial::variable(1);
    let v1 = Polynomial::variable(2);
    let v2 = Polynomial::variable(3);

    let a = x1.mul(&x1).add(&x2.mul(&x2));
    let b = x1.mul(&v1).add(&x2.mul(&v2));
    let c = v1.mul(&v1).add(&v2.mul(&v2));
    let angular_momentum = x1.mul(&v2).sub(&x2.mul(&v1));
    (a, b, c, angular_momentum)
}

fn determinant3(matrix: [[i128; 3]; 3]) -> i128 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

/// Whether the invariant map (a,b,c,ell) has differential rank three at a
/// state. The origin is the non-free orbit; x=0 with v!=0 is regular.
pub fn quotient_invariant_map_has_rank_three_at(
    state: [i128; 4],
) -> Result<bool, QualificationError> {
    const MAX_STATE_MAGNITUDE: i128 = 1_000_000;
    for (index, coordinate) in state.iter().enumerate() {
        if coordinate.unsigned_abs() > MAX_STATE_MAGNITUDE as u128 {
            return Err(QualificationError::StateCoordinateOutsideBound {
                index,
                maximum_magnitude: MAX_STATE_MAGNITUDE,
            });
        }
    }

    let (a, b, c, angular_momentum) = invariant_polynomials();
    let invariants = [a, b, c, angular_momentum];
    let jacobian: [[i128; 4]; 4] = std::array::from_fn(|row| {
        std::array::from_fn(|column| invariants[row].derivative(column).evaluate(state))
    });

    for omitted_row in 0..4 {
        let rows: Vec<usize> = (0..4).filter(|row| *row != omitted_row).collect();
        for omitted_column in 0..4 {
            let columns: Vec<usize> =
                (0..4).filter(|column| *column != omitted_column).collect();
            let minor: [[i128; 3]; 3] = std::array::from_fn(|row| {
                std::array::from_fn(|column| jacobian[rows[row]][columns[column]])
            });
            if determinant3(minor) != 0 {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

/// Check the exact SO(2)-equivariant radial family
///
///     x_dot = v,  v_dot = -k (x . x)^m x.
///
/// The bounded reference accepts integer m <= 8 and integer 0 < k <= 1,000,000.
/// All returned checks are exact polynomial identities over the integers.
pub fn qualify_radial_case(
    exponent: u32,
    coupling: i128,
) -> Result<QualificationReport, QualificationError> {
    const MAX_EXPONENT: u32 = 8;
    const MAX_COUPLING: i128 = 1_000_000;
    if coupling <= 0 {
        return Err(QualificationError::NonPositiveCoupling);
    }
    if coupling > MAX_COUPLING {
        return Err(QualificationError::CouplingOutsideBound {
            coupling,
            maximum: MAX_COUPLING,
        });
    }
    if exponent > MAX_EXPONENT {
        return Err(QualificationError::ExponentOutsideBound {
            exponent,
            maximum: MAX_EXPONENT,
        });
    }

    let vector_field = radial_vector_field(exponent, coupling);
    let (a, b, c, angular_momentum) = invariant_polynomials();

    let vector_field_equivariant = infinitesimal_rotation_residuals(&vector_field)
        .iter()
        .all(Polynomial::is_zero);

    // The Gram-determinant identity: |x|^2 |v|^2 - (x.v)^2 = det(x,v)^2.
    let invariant_relation_holds = a
        .mul(&c)
        .sub(&b.mul(&b))
        .sub(&angular_momentum.mul(&angular_momentum))
        .is_zero();

    let radial_gain = a.pow(exponent).scale(coupling);
    let expected_a_dot = b.scale(2);
    let expected_b_dot = c.sub(&a.mul(&radial_gain));
    let expected_c_dot = b.mul(&radial_gain).scale(-2);
    let quotient_dynamics_match = a.along(&vector_field).sub(&expected_a_dot).is_zero()
        && b.along(&vector_field).sub(&expected_b_dot).is_zero()
        && c.along(&vector_field).sub(&expected_c_dot).is_zero()
        && angular_momentum.along(&vector_field).is_zero();

    let angular_momentum_conserved = angular_momentum.along(&vector_field).is_zero();

    // 2(m+1)H = (m+1)c + k a^(m+1), so no rational coefficients are needed.
    let energy = c
        .scale(i128::from(exponent + 1))
        .add(&a.pow(exponent + 1).scale(coupling));
    let energy_conserved = energy.along(&vector_field).is_zero();

    Ok(QualificationReport {
        exponent,
        coupling,
        vector_field_equivariant,
        invariant_relation_holds,
        quotient_dynamics_match,
        angular_momentum_conserved,
        energy_conserved,
    })
}

/// Exact invariance status for the outputs used by the radial-force fixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservableInvarianceReport {
    pub radius_squared_invariant: bool,
    pub radial_velocity_invariant: bool,
    pub speed_squared_invariant: bool,
    pub signed_angular_momentum_invariant: bool,
    pub absolute_x1_invariant: bool,
}

/// Check which declared scalar outputs are invariant under the SO(2) action.
///
/// Absolute external-frame coordinates must not be reconstructed from the
/// orbit invariants; the output is deliberately expected to be non-invariant.
pub fn qualify_radial_observables() -> ObservableInvarianceReport {
    let (a, b, c, angular_momentum) = invariant_polynomials();
    let absolute_x1 = Polynomial::variable(0);
    ObservableInvarianceReport {
        radius_squared_invariant: scalar_is_rotation_invariant(&a),
        radial_velocity_invariant: scalar_is_rotation_invariant(&b),
        speed_squared_invariant: scalar_is_rotation_invariant(&c),
        signed_angular_momentum_invariant: scalar_is_rotation_invariant(&angular_momentum),
        absolute_x1_invariant: scalar_is_rotation_invariant(&absolute_x1),
    }
}

/// Returns whether an oscillator with diagonal stiffness (k1, k2) is
/// infinitesimally SO(2)-equivariant. Unequal stiffness must be rejected.
pub fn anisotropic_oscillator_is_rotation_equivariant(k1: i128, k2: i128) -> bool {
    infinitesimal_rotation_residuals(&anisotropic_oscillator(k1, k2))
        .iter()
        .all(Polynomial::is_zero)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radial_family_is_exactly_equivariant_for_multiple_exponents_and_scales() {
        for exponent in 0..=8 {
            for coupling in [1, 2, 7] {
                let report = qualify_radial_case(exponent, coupling).unwrap();
                assert!(report.vector_field_equivariant, "{report:?}");
            }
        }
    }

    #[test]
    fn gram_determinant_relation_holds_exactly() {
        for exponent in 0..=8 {
            let report = qualify_radial_case(exponent, 3).unwrap();
            assert!(report.invariant_relation_holds, "{report:?}");
        }
    }

    #[test]
    fn invariant_coordinate_dynamics_match_the_raw_vector_field() {
        for exponent in 0..=8 {
            for coupling in [1, 3] {
                let report = qualify_radial_case(exponent, coupling).unwrap();
                assert!(report.quotient_dynamics_match, "{report:?}");
                assert!(report.angular_momentum_conserved, "{report:?}");
            }
        }
    }

    #[test]
    fn scaled_energy_is_exactly_conserved() {
        for exponent in 0..=8 {
            for coupling in [1, 2, 5] {
                let report = qualify_radial_case(exponent, coupling).unwrap();
                assert!(report.energy_conserved, "{report:?}");
            }
        }
    }

    #[test]
    fn complete_radial_report_passes_every_exact_check() {
        for exponent in 0..=8 {
            for coupling in [1, 2, 5] {
                let report = qualify_radial_case(exponent, coupling).unwrap();
                assert!(report.all_exact_checks_pass(), "{report:?}");
            }
        }
    }

    #[test]
    fn invariant_outputs_pass_but_absolute_orientation_is_rejected() {
        let report = qualify_radial_observables();
        assert!(report.radius_squared_invariant, "{report:?}");
        assert!(report.radial_velocity_invariant, "{report:?}");
        assert!(report.speed_squared_invariant, "{report:?}");
        assert!(report.signed_angular_momentum_invariant, "{report:?}");
        assert!(!report.absolute_x1_invariant, "{report:?}");
    }

    #[test]
    fn regular_boundary_orbits_are_distinguished_from_the_fixed_origin() {
        assert_eq!(quotient_invariant_map_has_rank_three_at([0, 0, 1, 0]), Ok(true));
        assert_eq!(quotient_invariant_map_has_rank_three_at([0, 0, 0, 1]), Ok(true));
        assert_eq!(quotient_invariant_map_has_rank_three_at([1, 0, 2, 0]), Ok(true));
        assert_eq!(quotient_invariant_map_has_rank_three_at([0, 0, 0, 0]), Ok(false));
        assert_eq!(
            quotient_invariant_map_has_rank_three_at([i128::MIN, 0, 0, 0]),
            Err(QualificationError::StateCoordinateOutsideBound {
                index: 0,
                maximum_magnitude: 1_000_000
            })
        );
    }

    #[test]
    fn isotropic_oscillator_is_rotation_equivariant() {
        assert!(anisotropic_oscillator_is_rotation_equivariant(4, 4));
    }

    #[test]
    fn anisotropic_oscillator_is_not_rotation_equivariant() {
        assert!(!anisotropic_oscillator_is_rotation_equivariant(4, 5));
        assert!(!anisotropic_oscillator_is_rotation_equivariant(1, 3));
    }

    #[test]
    fn non_positive_coupling_is_rejected() {
        assert_eq!(
            qualify_radial_case(0, 0),
            Err(QualificationError::NonPositiveCoupling)
        );
        assert_eq!(
            qualify_radial_case(0, -1),
            Err(QualificationError::NonPositiveCoupling)
        );
    }

    #[test]
    fn coupling_above_bounded_exact_arithmetic_limit_is_rejected() {
        assert_eq!(
            qualify_radial_case(0, i128::MAX),
            Err(QualificationError::CouplingOutsideBound {
                coupling: i128::MAX,
                maximum: 1_000_000
            })
        );
    }

    #[test]
    fn exponent_outside_bounded_polynomial_reference_is_rejected() {
        assert_eq!(
            qualify_radial_case(9, 1),
            Err(QualificationError::ExponentOutsideBound {
                exponent: 9,
                maximum: 8
            })
        );
    }
}
