//! Small-N weighted moment oracle for 1D1V kinetic phase-space measures.
//!
//! The represented state is a finite signed measure on S^1_L x R. Particle
//! records are canonicalized and co-located signed coefficients are aggregated
//! before total variation and moments are evaluated. The oracle is a diagnostic
//! reference; it does not certify a time integrator or a self-consistent field
//! solve.
//!
//! For nonnegative weights, the half second velocity moment is kinetic energy
//! in the selected mass normalization. For signed delta-f coefficients it is
//! a signed moment contribution, not a positive physical energy by itself.

#![forbid(unsafe_code)]

use std::error::Error;
use std::fmt;

/// One weighted particle on periodic position and unbounded velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseSpaceParticle1d {
    pub position: f64,
    pub velocity: f64,
    pub weight: f64,
}

/// Frozen periodic domain and velocity normalization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentConfig {
    /// Period of the spatial coordinate.
    pub domain_length: f64,
    /// Positive scale used only for dimensionless reported moments.
    pub velocity_scale: f64,
}

/// Exact finite-measure moment report.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentReport {
    /// Signed zeroth moment: sum_i w_i.
    pub zeroth_moment: f64,
    /// Signed first velocity moment: sum_i w_i v_i.
    pub first_velocity_moment: f64,
    /// Signed half second velocity moment: 0.5 * sum_i w_i v_i^2.
    ///
    /// This is a physical kinetic-energy moment only for nonnegative particle
    /// measures with the appropriate mass/weight normalization.
    pub half_second_velocity_moment: f64,
    /// The same first moment normalized by velocity_scale.
    pub normalized_first_velocity_moment: f64,
    /// The same half second moment normalized by velocity_scale^2.
    pub normalized_half_second_velocity_moment: f64,
    /// Mean velocity for nonnegative measures with positive total mass.
    ///
    /// Undefined for signed measures and zero-mass measures.
    pub mean_velocity: Option<f64>,
    /// Half second central velocity moment, evaluated directly about mean_velocity.
    ///
    /// Defined only for nonnegative measures with positive total mass. It is
    /// a physical thermal-energy moment only with the appropriate mass/weight
    /// normalization.
    pub half_second_central_velocity_moment: Option<f64>,
    /// The central moment normalized by velocity_scale^2.
    pub normalized_half_second_central_velocity_moment: Option<f64>,
    /// True total variation after co-located signed coefficients are summed.
    pub total_variation: f64,
    /// Whether every nonzero aggregated atom has a nonnegative coefficient.
    pub is_nonnegative_measure: bool,
    /// Input particle count; metadata only.
    pub input_particle_count: usize,
    /// Unique nonzero support count after periodic canonicalization/aggregation.
    pub unique_atom_count: usize,
}

/// Invalid input, overflow, or an invalid normalization contract.
#[derive(Debug, Clone, PartialEq)]
pub enum MomentError {
    InvalidDomainLength,
    InvalidVelocityScale,
    NonFinitePosition { particle: usize },
    NonFiniteVelocity { particle: usize },
    NonFiniteWeight { particle: usize },
    AggregateWeightOverflow,
    TotalVariationOverflow,
    MomentOverflow,
}

impl fmt::Display for MomentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDomainLength => {
                write!(f, "domain length must be finite and positive")
            }
            Self::InvalidVelocityScale => {
                write!(f, "velocity scale must be finite and positive")
            }
            Self::NonFinitePosition { particle } => {
                write!(f, "particle {particle} has a non-finite position")
            }
            Self::NonFiniteVelocity { particle } => {
                write!(f, "particle {particle} has a non-finite velocity")
            }
            Self::NonFiniteWeight { particle } => {
                write!(f, "particle {particle} has a non-finite weight")
            }
            Self::AggregateWeightOverflow => {
                write!(f, "co-located signed particle weights overflowed")
            }
            Self::TotalVariationOverflow => {
                write!(f, "represented-measure total variation overflowed")
            }
            Self::MomentOverflow => {
                write!(f, "one or more weighted velocity moments overflowed")
            }
        }
    }
}

impl Error for MomentError {}

#[derive(Debug, Default, Clone, Copy)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
    }

    fn total(self) -> f64 {
        self.sum + self.correction
    }
}

#[derive(Debug, Clone, Copy)]
struct WeightedAtom {
    position: f64,
    velocity: f64,
    weight: f64,
}

fn canonical_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

fn validate_config(config: MomentConfig) -> Result<(), MomentError> {
    if !config.domain_length.is_finite() || config.domain_length <= 0.0 {
        return Err(MomentError::InvalidDomainLength);
    }
    if !config.velocity_scale.is_finite() || config.velocity_scale <= 0.0 {
        return Err(MomentError::InvalidVelocityScale);
    }
    Ok(())
}

/// Reduce periodic representatives, sort deterministically and aggregate
/// co-located signed coefficients. Total variation is computed only after
/// this reduction, so +a and -b at the same point represent |a-b|, not
/// |a|+|b|.
fn aggregate_measure(
    particles: &[PhaseSpaceParticle1d],
    domain_length: f64,
) -> Result<(Vec<WeightedAtom>, f64, bool), MomentError> {
    let mut atoms = Vec::with_capacity(particles.len());
    for (index, particle) in particles.iter().enumerate() {
        if !particle.position.is_finite() {
            return Err(MomentError::NonFinitePosition { particle: index });
        }
        if !particle.velocity.is_finite() {
            return Err(MomentError::NonFiniteVelocity { particle: index });
        }
        if !particle.weight.is_finite() {
            return Err(MomentError::NonFiniteWeight { particle: index });
        }
        atoms.push(WeightedAtom {
            position: canonical_zero(particle.position.rem_euclid(domain_length)),
            velocity: canonical_zero(particle.velocity),
            weight: particle.weight,
        });
    }

    atoms.sort_by(|left, right| {
        left.position
            .total_cmp(&right.position)
            .then_with(|| left.velocity.total_cmp(&right.velocity))
            .then_with(|| left.weight.total_cmp(&right.weight))
    });

    let mut aggregated = Vec::with_capacity(atoms.len());
    let mut total_variation = CompensatedSum::default();
    let mut is_nonnegative_measure = true;
    let mut index = 0;
    while index < atoms.len() {
        let position = atoms[index].position;
        let velocity = atoms[index].velocity;
        let mut coefficient = CompensatedSum::default();
        while index < atoms.len()
            && atoms[index].position == position
            && atoms[index].velocity == velocity
        {
            coefficient.add(atoms[index].weight);
            index += 1;
        }

        let weight = coefficient.total();
        if !weight.is_finite() {
            return Err(MomentError::AggregateWeightOverflow);
        }
        if weight != 0.0 {
            is_nonnegative_measure &= weight > 0.0;
            total_variation.add(weight.abs());
            aggregated.push(WeightedAtom {
                position,
                velocity,
                weight,
            });
        }
    }

    let total_variation = total_variation.total();
    if !total_variation.is_finite() {
        return Err(MomentError::TotalVariationOverflow);
    }
    Ok((aggregated, total_variation, is_nonnegative_measure))
}

/// Compute exact finite empirical moments with deterministic support
/// canonicalization and compensated accumulation.
///
/// No normalization by total weight is performed: the raw signed moments and
/// their total-variation scale are returned. The configured velocity scale
/// adds dimensionless companion moments without hiding the raw quantities.
pub fn evaluate_phase_space_moments(
    particles: &[PhaseSpaceParticle1d],
    config: MomentConfig,
) -> Result<MomentReport, MomentError> {
    validate_config(config)?;
    let (atoms, total_variation, is_nonnegative_measure) =
        aggregate_measure(particles, config.domain_length)?;

    let mut zeroth = CompensatedSum::default();
    let mut first = CompensatedSum::default();
    let mut second = CompensatedSum::default();
    let mut normalized_first = CompensatedSum::default();
    let mut normalized_second = CompensatedSum::default();

    for atom in &atoms {
        let scaled_velocity = atom.velocity / config.velocity_scale;
        let terms = [
            atom.weight,
            atom.weight * atom.velocity,
            0.5 * atom.weight * atom.velocity * atom.velocity,
            atom.weight * scaled_velocity,
            0.5 * atom.weight * scaled_velocity * scaled_velocity,
        ];
        if terms.iter().any(|term| !term.is_finite()) {
            return Err(MomentError::MomentOverflow);
        }
        zeroth.add(terms[0]);
        first.add(terms[1]);
        second.add(terms[2]);
        normalized_first.add(terms[3]);
        normalized_second.add(terms[4]);
    }

    let zeroth_moment = zeroth.total();
    let first_velocity_moment = first.total();
    let mean_velocity = if is_nonnegative_measure && zeroth_moment > 0.0 {
        let mean = first_velocity_moment / zeroth_moment;
        if !mean.is_finite() {
            return Err(MomentError::MomentOverflow);
        }
        Some(mean)
    } else {
        None
    };

    let (half_second_central_velocity_moment, normalized_half_second_central_velocity_moment) =
        if let Some(mean) = mean_velocity {
            let mut central = CompensatedSum::default();
            let mut normalized_central = CompensatedSum::default();
            for atom in &atoms {
                let centered_velocity = atom.velocity - mean;
                let normalized_centered_velocity = centered_velocity / config.velocity_scale;
                let central_term =
                    0.5 * atom.weight * centered_velocity * centered_velocity;
                let normalized_term = 0.5
                    * atom.weight
                    * normalized_centered_velocity
                    * normalized_centered_velocity;
                if !central_term.is_finite() || !normalized_term.is_finite() {
                    return Err(MomentError::MomentOverflow);
                }
                central.add(central_term);
                normalized_central.add(normalized_term);
            }
            (Some(central.total()), Some(normalized_central.total()))
        } else {
            (None, None)
        };

    let report = MomentReport {
        zeroth_moment,
        first_velocity_moment,
        half_second_velocity_moment: second.total(),
        normalized_first_velocity_moment: normalized_first.total(),
        normalized_half_second_velocity_moment: normalized_second.total(),
        mean_velocity,
        half_second_central_velocity_moment,
        normalized_half_second_central_velocity_moment,
        total_variation,
        is_nonnegative_measure,
        input_particle_count: particles.len(),
        unique_atom_count: atoms.len(),
    };

    if [
        report.zeroth_moment,
        report.first_velocity_moment,
        report.half_second_velocity_moment,
        report.normalized_first_velocity_moment,
        report.normalized_half_second_velocity_moment,
    ]
    .iter()
    .any(|value| !value.is_finite())
        || report.mean_velocity.is_some_and(|value| !value.is_finite())
        || report
            .half_second_central_velocity_moment
            .is_some_and(|value| !value.is_finite())
        || report
            .normalized_half_second_central_velocity_moment
            .is_some_and(|value| !value.is_finite())
    {
        return Err(MomentError::MomentOverflow);
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MomentConfig {
        MomentConfig {
            domain_length: 8.0,
            velocity_scale: 2.0,
        }
    }

    fn particle(position: f64, velocity: f64, weight: f64) -> PhaseSpaceParticle1d {
        PhaseSpaceParticle1d {
            position,
            velocity,
            weight,
        }
    }

    fn assert_near(left: f64, right: f64, tolerance: f64) {
        assert!(
            (left - right).abs() <= tolerance,
            "expected {left} ~= {right} within {tolerance}"
        );
    }

    #[test]
    fn known_positive_measure_moments_are_exact() {
        let measure = [particle(0.0, -2.0, 0.25), particle(4.0, 2.0, 0.75)];
        let report = evaluate_phase_space_moments(&measure, config()).unwrap();
        assert_near(report.zeroth_moment, 1.0, 1.0e-14);
        assert_near(report.first_velocity_moment, 1.0, 1.0e-14);
        assert_near(report.half_second_velocity_moment, 2.0, 1.0e-14);
        assert_near(report.normalized_first_velocity_moment, 0.5, 1.0e-14);
        assert_near(report.normalized_half_second_velocity_moment, 0.5, 1.0e-14);
        assert_near(report.mean_velocity.unwrap(), 1.0, 1.0e-14);
        assert_near(
            report.half_second_central_velocity_moment.unwrap(),
            1.5,
            1.0e-14,
        );
        assert_near(
            report
                .normalized_half_second_central_velocity_moment
                .unwrap(),
            0.375,
            1.0e-14,
        );
        assert!(report.is_nonnegative_measure);
        assert_near(report.total_variation, 1.0, 1.0e-14);
    }

    #[test]
    fn permutation_and_colocated_split_merge_are_invariant() {
        let original = [particle(1.0, 0.5, 0.4), particle(6.0, -1.0, 0.6)];
        let permuted = [original[1], original[0]];
        let split = [
            particle(1.0, 0.5, 0.1),
            particle(1.0, 0.5, 0.3),
            particle(6.0, -1.0, 0.2),
            particle(6.0, -1.0, 0.4),
        ];
        let a = evaluate_phase_space_moments(&original, config()).unwrap();
        let b = evaluate_phase_space_moments(&permuted, config()).unwrap();
        let c = evaluate_phase_space_moments(&split, config()).unwrap();
        assert_eq!(a.zeroth_moment, b.zeroth_moment);
        assert_eq!(a.first_velocity_moment, b.first_velocity_moment);
        assert_eq!(a.half_second_velocity_moment, b.half_second_velocity_moment);
        assert_near(c.zeroth_moment, a.zeroth_moment, 1.0e-14);
        assert_near(c.first_velocity_moment, a.first_velocity_moment, 1.0e-14);
        assert_near(
            c.half_second_velocity_moment,
            a.half_second_velocity_moment,
            1.0e-14,
        );
        assert_eq!(c.unique_atom_count, 2);
    }

    #[test]
    fn periodic_coordinate_representatives_are_equivalent() {
        let a = [particle(0.25, 1.5, 1.0)];
        let b = [particle(8.25, 1.5, 1.0), particle(-7.75, 1.5, 0.0)];
        let left = evaluate_phase_space_moments(&a, config()).unwrap();
        let right = evaluate_phase_space_moments(&b, config()).unwrap();
        assert_eq!(left.zeroth_moment, right.zeroth_moment);
        assert_eq!(left.first_velocity_moment, right.first_velocity_moment);
        assert_eq!(left.total_variation, right.total_variation);
    }

    #[test]
    fn signed_colocated_cancellation_uses_measure_total_variation() {
        let measure = [particle(2.0, -0.5, 0.5), particle(10.0, -0.5, -0.25)];
        let report = evaluate_phase_space_moments(&measure, config()).unwrap();
        assert_near(report.zeroth_moment, 0.25, 1.0e-14);
        assert_near(report.first_velocity_moment, -0.125, 1.0e-14);
        assert_near(report.half_second_velocity_moment, 0.03125, 1.0e-14);
        assert_near(report.total_variation, 0.25, 1.0e-14);
        assert_near(report.mean_velocity.unwrap(), -0.5, 1.0e-14);
        assert_near(
            report.half_second_central_velocity_moment.unwrap(),
            0.0,
            1.0e-14,
        );
        assert!(report.is_nonnegative_measure);
        assert_eq!(report.unique_atom_count, 1);
    }

    #[test]
    fn signed_support_retains_signed_moments_and_is_not_positive_measure() {
        let measure = [particle(0.0, -1.0, 1.0), particle(1.0, 2.0, -0.5)];
        let report = evaluate_phase_space_moments(&measure, config()).unwrap();
        assert_near(report.zeroth_moment, 0.5, 1.0e-14);
        assert_near(report.first_velocity_moment, -2.0, 1.0e-14);
        assert_near(report.half_second_velocity_moment, -0.5, 1.0e-14);
        assert!(report.mean_velocity.is_none());
        assert!(report.half_second_central_velocity_moment.is_none());
        assert!(report
            .normalized_half_second_central_velocity_moment
            .is_none());
        assert!(!report.is_nonnegative_measure);
        assert_near(report.total_variation, 1.5, 1.0e-14);
    }

    #[test]
    fn common_weight_rescaling_scales_every_raw_moment() {
        let measure = [particle(0.0, -0.5, 0.2), particle(4.0, 1.5, 0.8)];
        let scaled = measure.map(|p| PhaseSpaceParticle1d {
            weight: 3.0 * p.weight,
            ..p
        });
        let a = evaluate_phase_space_moments(&measure, config()).unwrap();
        let b = evaluate_phase_space_moments(&scaled, config()).unwrap();
        assert_near(b.zeroth_moment, 3.0 * a.zeroth_moment, 1.0e-14);
        assert_near(
            b.first_velocity_moment,
            3.0 * a.first_velocity_moment,
            1.0e-14,
        );
        assert_near(
            b.half_second_velocity_moment,
            3.0 * a.half_second_velocity_moment,
            1.0e-14,
        );
        assert_near(b.total_variation, 3.0 * a.total_variation, 1.0e-14);
    }

    #[test]
    fn velocity_boost_follows_exact_galilean_moment_transform() {
        let measure = [particle(0.0, -1.0, 0.25), particle(4.0, 2.0, 0.75)];
        let boost = 0.75;
        let boosted = measure.map(|p| PhaseSpaceParticle1d {
            velocity: p.velocity + boost,
            ..p
        });
        let a = evaluate_phase_space_moments(&measure, config()).unwrap();
        let b = evaluate_phase_space_moments(&boosted, config()).unwrap();
        assert_near(
            b.first_velocity_moment,
            a.first_velocity_moment + boost * a.zeroth_moment,
            1.0e-14,
        );
        assert_near(
            b.half_second_velocity_moment,
            a.half_second_velocity_moment
                + boost * a.first_velocity_moment
                + 0.5 * boost * boost * a.zeroth_moment,
            1.0e-14,
        );
    }

    #[test]
    fn zero_weights_and_empty_measures_are_well_defined() {
        let empty = evaluate_phase_space_moments(&[], config()).unwrap();
        assert_eq!(empty.zeroth_moment, 0.0);
        assert_eq!(empty.total_variation, 0.0);
        assert!(empty.mean_velocity.is_none());
        assert!(empty.half_second_central_velocity_moment.is_none());
        assert!(empty.is_nonnegative_measure);
        assert_eq!(empty.unique_atom_count, 0);

        let with_zero = [particle(0.0, 0.0, 1.0), particle(4.0, 1.0, 0.0)];
        let without_zero = [particle(0.0, 0.0, 1.0)];
        let a = evaluate_phase_space_moments(&with_zero, config()).unwrap();
        let b = evaluate_phase_space_moments(&without_zero, config()).unwrap();
        assert_eq!(a.zeroth_moment, b.zeroth_moment);
        assert_eq!(a.unique_atom_count, b.unique_atom_count);
    }

    #[test]
    fn free_streaming_preserves_moments_across_periodic_wraps() {
        let mut measure = [
            particle(7.5, 0.9, 0.4),
            particle(0.1, -1.1, 0.35),
            particle(3.2, 0.2, 0.25),
        ];
        let initial = evaluate_phase_space_moments(&measure, config()).unwrap();

        for _ in 0..32 {
            measure = measure.map(|p| PhaseSpaceParticle1d {
                position: p.position + 3.7 * p.velocity,
                ..p
            });
            let report = evaluate_phase_space_moments(&measure, config()).unwrap();
            assert_near(report.zeroth_moment, initial.zeroth_moment, 1.0e-12);
            assert_near(
                report.first_velocity_moment,
                initial.first_velocity_moment,
                1.0e-12,
            );
            assert_near(
                report.half_second_velocity_moment,
                initial.half_second_velocity_moment,
                1.0e-12,
            );
            assert_near(report.total_variation, initial.total_variation, 1.0e-12);
            assert_near(
                report.half_second_central_velocity_moment.unwrap(),
                initial.half_second_central_velocity_moment.unwrap(),
                1.0e-12,
            );
        }
    }

    #[test]
    fn invalid_config_and_non_finite_inputs_fail_closed() {
        let measure = [particle(0.0, 0.0, 1.0)];
        let mut invalid = config();
        invalid.domain_length = 0.0;
        assert_eq!(
            evaluate_phase_space_moments(&measure, invalid),
            Err(MomentError::InvalidDomainLength)
        );
        invalid = config();
        invalid.velocity_scale = f64::NAN;
        assert_eq!(
            evaluate_phase_space_moments(&measure, invalid),
            Err(MomentError::InvalidVelocityScale)
        );
        assert!(matches!(
            evaluate_phase_space_moments(&[particle(f64::NAN, 0.0, 1.0)], config()),
            Err(MomentError::NonFinitePosition { particle: 0 })
        ));
        assert!(matches!(
            evaluate_phase_space_moments(&[particle(0.0, f64::INFINITY, 1.0)], config()),
            Err(MomentError::NonFiniteVelocity { particle: 0 })
        ));
        assert!(matches!(
            evaluate_phase_space_moments(&[particle(0.0, 0.0, f64::NAN)], config()),
            Err(MomentError::NonFiniteWeight { particle: 0 })
        ));
    }

    #[test]
    fn aggregate_and_total_variation_overflow_fail_closed() {
        let aggregate_overflow = [particle(0.0, 0.0, f64::MAX), particle(8.0, 0.0, f64::MAX)];
        assert_eq!(
            evaluate_phase_space_moments(&aggregate_overflow, config()),
            Err(MomentError::AggregateWeightOverflow)
        );

        let tv_overflow = [particle(0.0, 0.0, f64::MAX), particle(1.0, 0.0, f64::MAX)];
        assert_eq!(
            evaluate_phase_space_moments(&tv_overflow, config()),
            Err(MomentError::TotalVariationOverflow)
        );
    }

    #[test]
    fn moment_overflow_fails_closed() {
        let measure = [particle(0.0, 2.0, f64::MAX)];
        assert_eq!(
            evaluate_phase_space_moments(&measure, config()),
            Err(MomentError::MomentOverflow)
        );
    }
}
