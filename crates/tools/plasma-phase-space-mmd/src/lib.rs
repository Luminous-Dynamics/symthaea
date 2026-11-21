//! Exact small-N MMD reference oracle for weighted 1D1V phase-space measures.
//!
//! The state space is the periodic cylinder S^1_L x R. Particle labels are
//! deliberately absent from the comparison: only the represented weighted
//! empirical measures matter.
//!
//! The spatial kernel is von Mises and the velocity kernel is Gaussian:
//!
//!   k_x(x,x') = exp(-2*kappa*sin(pi*(x-x')/L)^2)
//!   k_v(v,v') = exp(-(v-v')^2/(2*ell_v^2)).
//!
//! Their product is characteristic on S^1_L x R for kappa > 0 and ell_v > 0.
//! This implementation evaluates the finite empirical quadratic form directly
//! in O(N^2 + M^2 + NM); it is an assurance oracle, not a production solver or
//! a statistical two-sample test.

#![forbid(unsafe_code)]

use std::error::Error;
use std::f64::consts::PI;
use std::fmt;

/// One weighted particle on periodic position and unbounded velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhaseSpaceParticle1d {
    pub position: f64,
    pub velocity: f64,
    pub weight: f64,
}

/// Frozen kernel parameters for the 1D1V periodic-x / unbounded-v cohort.
///
/// The dimensionless spatial concentration kappa must be positive. Its
/// equivalent small-separation Gaussian length scale is
/// ell_x = L / (2*pi*sqrt(kappa)). The velocity length scale has velocity units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MmdConfig {
    pub domain_length: f64,
    pub spatial_concentration: f64,
    pub velocity_length_scale: f64,
}

/// Exact-small-N reference result, normalized by the larger total variation
/// of the two represented signed measures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MmdResult {
    /// MMD^2 / max(TV(candidate), TV(reference))^2.
    pub normalized_mmd_squared: f64,
    /// Square root of normalized_mmd_squared.
    pub normalized_mmd: f64,
    /// max(TV(candidate), TV(reference)), before normalization.
    pub total_variation_scale: f64,
    /// Input counts are metadata only and do not enter the distance.
    pub candidate_particle_count: usize,
    pub reference_particle_count: usize,
    /// Number of unique nonzero support atoms after canonical aggregation.
    pub candidate_atom_count: usize,
    pub reference_atom_count: usize,
    /// Number of kernel evaluations: U^2 + V^2 + U*V for U,V unique atoms.
    pub pair_evaluations: u128,
    /// Conservative floating-point tolerance used when checking MMD^2 >= 0.
    pub roundoff_tolerance: f64,
    /// True only when a small negative result within tolerance was clamped.
    pub negative_roundoff_clamped: bool,
}

/// Invalid input, overflow, or a violation of the positive-kernel invariant.
#[derive(Debug, Clone, PartialEq)]
pub enum MmdError {
    InvalidDomainLength,
    InvalidSpatialConcentration,
    InvalidVelocityLengthScale,
    NonFinitePosition {
        measure: &'static str,
        particle: usize,
    },
    NonFiniteVelocity {
        measure: &'static str,
        particle: usize,
    },
    NonFiniteWeight {
        measure: &'static str,
        particle: usize,
    },
    TotalVariationOverflow {
        measure: &'static str,
    },
    PairCountOverflow,
    NonFiniteOutput,
    MateriallyNegativeMmdSquared {
        value: f64,
        tolerance: f64,
    },
}

impl fmt::Display for MmdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDomainLength => write!(f, "domain length must be finite and positive"),
            Self::InvalidSpatialConcentration => {
                write!(f, "spatial concentration must be finite and positive")
            }
            Self::InvalidVelocityLengthScale => {
                write!(f, "velocity length scale must be finite and positive")
            }
            Self::NonFinitePosition { measure, particle } => {
                write!(f, "{measure} particle {particle} has a non-finite position")
            }
            Self::NonFiniteVelocity { measure, particle } => {
                write!(f, "{measure} particle {particle} has a non-finite velocity")
            }
            Self::NonFiniteWeight { measure, particle } => {
                write!(f, "{measure} particle {particle} has a non-finite weight")
            }
            Self::TotalVariationOverflow { measure } => {
                write!(f, "{measure} total-variation weight overflowed")
            }
            Self::PairCountOverflow => write!(f, "kernel pair-evaluation count overflowed"),
            Self::NonFiniteOutput => write!(f, "MMD oracle produced a non-finite result"),
            Self::MateriallyNegativeMmdSquared { value, tolerance } => write!(
                f,
                "MMD^2 is materially negative ({value:e}; roundoff tolerance {tolerance:e})"
            ),
        }
    }
}

impl Error for MmdError {}

/// Neumaier compensated summation.
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

fn validate_config(config: MmdConfig) -> Result<(), MmdError> {
    if !config.domain_length.is_finite() || config.domain_length <= 0.0 {
        return Err(MmdError::InvalidDomainLength);
    }
    if !config.spatial_concentration.is_finite() || config.spatial_concentration <= 0.0 {
        return Err(MmdError::InvalidSpatialConcentration);
    }
    if !config.velocity_length_scale.is_finite() || config.velocity_length_scale <= 0.0 {
        return Err(MmdError::InvalidVelocityLengthScale);
    }
    Ok(())
}

/// Canonicalize periodic coordinates, sort support/weights deterministically,
/// aggregate co-located signed coefficients, then compute true atomic TV.
/// The O(N log N) temporary vector is acceptable for this small-N oracle and
/// avoids representation-dependent signed-weight normalization.
fn aggregate_measure(
    particles: &[PhaseSpaceParticle1d],
    measure: &'static str,
    domain_length: f64,
) -> Result<(Vec<WeightedAtom>, f64), MmdError> {
    let mut atoms = Vec::with_capacity(particles.len());
    for (index, particle) in particles.iter().enumerate() {
        if !particle.position.is_finite() {
            return Err(MmdError::NonFinitePosition {
                measure,
                particle: index,
            });
        }
        if !particle.velocity.is_finite() {
            return Err(MmdError::NonFiniteVelocity {
                measure,
                particle: index,
            });
        }
        if !particle.weight.is_finite() {
            return Err(MmdError::NonFiniteWeight {
                measure,
                particle: index,
            });
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
            return Err(MmdError::TotalVariationOverflow { measure });
        }
        if weight != 0.0 {
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
        return Err(MmdError::TotalVariationOverflow { measure });
    }
    Ok((aggregated, total_variation))
}

/// Evaluate the product kernel. Positions have already been reduced modulo L.
fn kernel(left: WeightedAtom, right: WeightedAtom, config: MmdConfig) -> f64 {
    let delta_x = left.position - right.position;
    let half_angle = PI * (delta_x / config.domain_length);
    let sine = half_angle.sin();
    // Parenthesize before multiplying by kappa: this avoids (-inf)*0 = NaN
    // when a finite but extremely large concentration is supplied at x=x'.
    let spatial_exponent = -(config.spatial_concentration * (2.0 * sine * sine));
    let spatial = spatial_exponent.exp();

    let delta_v = left.velocity - right.velocity;
    let scaled_velocity = delta_v / config.velocity_length_scale;
    let velocity = (-0.5 * scaled_velocity * scaled_velocity).exp();

    spatial * velocity
}

fn pair_count(n: usize, m: usize) -> Result<u128, MmdError> {
    let n = n as u128;
    let m = m as u128;
    n.checked_mul(n)
        .and_then(|nn| m.checked_mul(m).and_then(|mm| nn.checked_add(mm)))
        .and_then(|sum| n.checked_mul(m).and_then(|nm| sum.checked_add(nm)))
        .ok_or(MmdError::PairCountOverflow)
}

fn quadratic_sum(
    left: &[WeightedAtom],
    right: &[WeightedAtom],
    normalization: f64,
    config: MmdConfig,
) -> (f64, f64) {
    let mut sum = CompensatedSum::default();
    let mut absolute_sum = CompensatedSum::default();

    for left_atom in left {
        let left_weight = left_atom.weight / normalization;
        for right_atom in right {
            let right_weight = right_atom.weight / normalization;
            let term = left_weight * right_weight * kernel(*left_atom, *right_atom, config);
            sum.add(term);
            absolute_sum.add(term.abs());
        }
    }

    (sum.total(), absolute_sum.total())
}

/// Compare two weighted empirical phase-space measures using the exact
/// finite-sample MMD quadratic form for the declared product kernel.
///
/// Co-located particles are reduced to one atom before computing both total
/// variation and MMD. This is essential for signed representations: e.g.
/// weights +0.5 and -0.25 at the same support point represent a single +0.25
/// atom, whose TV is 0.25 rather than the particle-weight sum 0.75.
///
/// The direct reference calculation uses O(N log N + M log M + U^2 + V^2 + UV)
/// time and O(N+M) temporary memory, where U,V are the unique nonzero atoms.
/// Signed weights represent finite signed-measure coefficients. This metric
/// is not Wasserstein distance, field error, or a solver-acceptance certificate.
pub fn compare_phase_space_measures_mmd(
    candidate: &[PhaseSpaceParticle1d],
    reference: &[PhaseSpaceParticle1d],
    config: MmdConfig,
) -> Result<MmdResult, MmdError> {
    validate_config(config)?;

    let (candidate_atoms, candidate_tv) =
        aggregate_measure(candidate, "candidate", config.domain_length)?;
    let (reference_atoms, reference_tv) =
        aggregate_measure(reference, "reference", config.domain_length)?;

    let total_variation_scale = candidate_tv.max(reference_tv);
    let normalization = if total_variation_scale == 0.0 {
        1.0
    } else {
        total_variation_scale
    };
    let pair_evaluations = pair_count(candidate_atoms.len(), reference_atoms.len())?;

    let (candidate_self, candidate_abs) =
        quadratic_sum(&candidate_atoms, &candidate_atoms, normalization, config);
    let (reference_self, reference_abs) =
        quadratic_sum(&reference_atoms, &reference_atoms, normalization, config);
    let (cross, cross_abs) =
        quadratic_sum(&candidate_atoms, &reference_atoms, normalization, config);

    let mut normalized_mmd_squared = candidate_self + reference_self - 2.0 * cross;
    let absolute_term_sum = candidate_abs + reference_abs + 2.0 * cross_abs;
    let roundoff_tolerance = 128.0 * f64::EPSILON * absolute_term_sum.max(1.0);

    if !normalized_mmd_squared.is_finite()
        || !absolute_term_sum.is_finite()
        || !roundoff_tolerance.is_finite()
    {
        return Err(MmdError::NonFiniteOutput);
    }

    let mut negative_roundoff_clamped = false;
    if normalized_mmd_squared < -roundoff_tolerance {
        return Err(MmdError::MateriallyNegativeMmdSquared {
            value: normalized_mmd_squared,
            tolerance: roundoff_tolerance,
        });
    }
    if normalized_mmd_squared < 0.0 {
        normalized_mmd_squared = 0.0;
        negative_roundoff_clamped = true;
    }

    let normalized_mmd = normalized_mmd_squared.sqrt();
    if !normalized_mmd.is_finite() {
        return Err(MmdError::NonFiniteOutput);
    }

    Ok(MmdResult {
        normalized_mmd_squared,
        normalized_mmd,
        total_variation_scale,
        candidate_particle_count: candidate.len(),
        reference_particle_count: reference.len(),
        candidate_atom_count: candidate_atoms.len(),
        reference_atom_count: reference_atoms.len(),
        pair_evaluations,
        roundoff_tolerance,
        negative_roundoff_clamped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> MmdConfig {
        MmdConfig {
            domain_length: 8.0,
            spatial_concentration: 2.0,
            velocity_length_scale: 1.0,
        }
    }

    fn particle(position: f64, velocity: f64, weight: f64) -> PhaseSpaceParticle1d {
        PhaseSpaceParticle1d {
            position,
            velocity,
            weight,
        }
    }

    fn assert_mmd_zero(result: MmdResult) {
        assert!(
            result.normalized_mmd <= 1.0e-12,
            "expected MMD approximately zero, got {:?}",
            result
        );
    }

    fn low_spatial_mode_signature(
        particles: &[PhaseSpaceParticle1d],
        modes: &[u32],
        domain_length: f64,
    ) -> Vec<(f64, f64)> {
        modes
            .iter()
            .map(|mode| {
                let mut real = CompensatedSum::default();
                let mut imaginary = CompensatedSum::default();
                for item in particles {
                    let phase =
                        std::f64::consts::TAU * f64::from(*mode) * item.position / domain_length;
                    real.add(item.weight * phase.cos());
                    imaginary.add(item.weight * phase.sin());
                }
                (real.total(), imaginary.total())
            })
            .collect()
    }

    #[test]
    fn identical_measure_has_zero_mmd() {
        let measure = [particle(0.5, -0.25, 0.4), particle(5.0, 1.5, 0.6)];
        let result = compare_phase_space_measures_mmd(&measure, &measure, config()).unwrap();
        assert_mmd_zero(result);
        assert_eq!(result.pair_evaluations, 12);
    }

    #[test]
    fn particle_permutation_does_not_change_the_measure() {
        let candidate = [
            particle(0.5, -0.25, 0.2),
            particle(2.0, 0.75, 0.3),
            particle(5.5, 1.25, 0.5),
        ];
        let reference = [candidate[2], candidate[0], candidate[1]];
        let result = compare_phase_space_measures_mmd(&candidate, &reference, config()).unwrap();
        assert_mmd_zero(result);
        assert_eq!(result.candidate_atom_count, 3);
    }

    #[test]
    fn co_located_weight_split_preserves_the_measure() {
        let candidate = [particle(1.25, -0.5, 0.75), particle(6.0, 2.0, 0.25)];
        let reference = [
            particle(1.25, -0.5, 0.25),
            particle(1.25, -0.5, 0.20),
            particle(1.25, -0.5, 0.30),
            particle(6.0, 2.0, 0.10),
            particle(6.0, 2.0, 0.15),
        ];
        let result = compare_phase_space_measures_mmd(&candidate, &reference, config()).unwrap();
        assert_mmd_zero(result);
        assert_eq!(result.candidate_particle_count, 2);
        assert_eq!(result.reference_particle_count, 5);
        assert_eq!(result.candidate_atom_count, result.reference_atom_count);
    }

    #[test]
    fn signed_co_located_cancellation_uses_measure_total_variation() {
        let candidate = [particle(1.0, 0.0, 0.25)];
        let reference = [particle(1.0, 0.0, 0.50), particle(1.0, 0.0, -0.25)];
        let result = compare_phase_space_measures_mmd(&candidate, &reference, config()).unwrap();
        assert_mmd_zero(result);
        assert_eq!(result.total_variation_scale, 0.25);
        assert_eq!(result.candidate_atom_count, 1);
        assert_eq!(result.reference_atom_count, 1);
    }

    #[test]
    fn periodic_coordinate_representatives_are_equivalent() {
        let candidate = [particle(1.0, 0.25, 1.0)];
        let reference = [particle(1.0 + 3.0 * config().domain_length, 0.25, 1.0)];
        let result = compare_phase_space_measures_mmd(&candidate, &reference, config()).unwrap();
        assert_mmd_zero(result);
    }

    #[test]
    fn characteristic_oracle_separates_a_finite_fourier_alias() {
        let length = config().domain_length;
        let candidate = (0..4)
            .map(|index| particle(f64::from(index) * length / 4.0, 0.0, 0.25))
            .collect::<Vec<_>>();
        let reference = (0..4)
            .map(|index| particle(length / 8.0 + f64::from(index) * length / 4.0, 0.0, 0.25))
            .collect::<Vec<_>>();

        // Both supports have the same DC, first and second spatial harmonics.
        let modes = [0, 1, 2];
        let left_signature = low_spatial_mode_signature(&candidate, &modes, length);
        let right_signature = low_spatial_mode_signature(&reference, &modes, length);
        for (left, right) in left_signature.iter().zip(&right_signature) {
            assert!((left.0 - right.0).abs() <= 1.0e-12);
            assert!((left.1 - right.1).abs() <= 1.0e-12);
        }

        let result = compare_phase_space_measures_mmd(&candidate, &reference, config()).unwrap();
        assert!(
            result.normalized_mmd > 0.1,
            "the characteristic reference should resolve this finite-mode alias: {:?}",
            result
        );
    }

    #[test]
    fn joint_kernel_detects_changed_position_velocity_correlation() {
        let candidate = [particle(0.0, 0.0, 0.5), particle(4.0, 1.0, 0.5)];
        let reference = [particle(0.0, 1.0, 0.5), particle(4.0, 0.0, 0.5)];

        // The position and velocity marginals agree, while the joint measure differs.
        let result = compare_phase_space_measures_mmd(&candidate, &reference, config()).unwrap();
        assert!(
            result.normalized_mmd > 0.1,
            "joint position-velocity dependence should be detected: {:?}",
            result
        );
    }

    #[test]
    fn signed_measure_permutation_and_split_are_invariant() {
        let candidate = [particle(1.0, 0.0, -0.5), particle(5.0, 1.0, 0.5)];
        let reference = [
            particle(5.0, 1.0, 0.2),
            particle(1.0, 0.0, -0.2),
            particle(1.0, 0.0, -0.3),
            particle(5.0, 1.0, 0.3),
        ];
        let result = compare_phase_space_measures_mmd(&candidate, &reference, config()).unwrap();
        assert_mmd_zero(result);
    }

    #[test]
    fn common_weight_rescaling_cancels_under_total_variation_normalization() {
        let candidate = [particle(1.0, 0.0, 0.5), particle(5.0, 1.0, 0.5)];
        let reference = [particle(1.0, 0.0, 0.4), particle(5.0, 1.0, 0.6)];
        let scaled_candidate = [particle(1.0, 0.0, 50.0), particle(5.0, 1.0, 50.0)];
        let scaled_reference = [particle(1.0, 0.0, 40.0), particle(5.0, 1.0, 60.0)];

        let original = compare_phase_space_measures_mmd(&candidate, &reference, config()).unwrap();
        let scaled =
            compare_phase_space_measures_mmd(&scaled_candidate, &scaled_reference, config())
                .unwrap();
        assert!((original.normalized_mmd - scaled.normalized_mmd).abs() <= 1.0e-12);
    }

    #[test]
    fn empty_measures_have_zero_distance() {
        let result = compare_phase_space_measures_mmd(&[], &[], config()).unwrap();
        assert_mmd_zero(result);
        assert_eq!(result.total_variation_scale, 0.0);
        assert_eq!(result.pair_evaluations, 0);
    }

    #[test]
    fn empty_vs_nonempty_measure_is_detected() {
        let reference = [particle(2.0, -1.0, 1.0)];
        let result = compare_phase_space_measures_mmd(&[], &reference, config()).unwrap();
        assert!((result.normalized_mmd - 1.0).abs() <= 1.0e-12);
    }

    #[test]
    fn zero_weight_particles_do_not_change_the_measure() {
        let candidate = [particle(0.5, 0.0, 1.0)];
        let reference = [particle(0.5, 0.0, 1.0), particle(7.0, 3.0, 0.0)];
        let result = compare_phase_space_measures_mmd(&candidate, &reference, config()).unwrap();
        assert_mmd_zero(result);
    }

    #[test]
    fn distinct_single_particle_supports_have_positive_distance() {
        let candidate = [particle(0.0, 0.0, 1.0)];
        let reference = [particle(4.0, 0.0, 1.0)];
        let result = compare_phase_space_measures_mmd(&candidate, &reference, config()).unwrap();
        assert!(result.normalized_mmd > 0.0);
    }

    #[test]
    fn kernel_is_symmetric_and_periodic_on_known_points() {
        let left = particle(0.75, -0.2, 1.0);
        let right = particle(3.25, 1.7, 1.0);
        let left_atom = WeightedAtom {
            position: left.position,
            velocity: left.velocity,
            weight: 1.0,
        };
        let right_atom = WeightedAtom {
            position: right.position,
            velocity: right.velocity,
            weight: 1.0,
        };
        let forward = kernel(left_atom, right_atom, config());
        let reverse = kernel(right_atom, left_atom, config());
        let shifted = kernel(
            WeightedAtom {
                position: (left.position + 2.0 * config().domain_length)
                    .rem_euclid(config().domain_length),
                velocity: left.velocity,
                weight: 1.0,
            },
            right_atom,
            config(),
        );
        assert!((forward - reverse).abs() <= 1.0e-14);
        assert!((forward - shifted).abs() <= 1.0e-14);
        assert!(forward > 0.0 && forward <= 1.0);
    }

    #[test]
    fn invalid_kernel_scales_are_rejected() {
        let mut invalid = config();
        invalid.domain_length = 0.0;
        assert_eq!(
            compare_phase_space_measures_mmd(&[], &[], invalid),
            Err(MmdError::InvalidDomainLength)
        );

        invalid = config();
        invalid.spatial_concentration = f64::NAN;
        assert_eq!(
            compare_phase_space_measures_mmd(&[], &[], invalid),
            Err(MmdError::InvalidSpatialConcentration)
        );

        invalid = config();
        invalid.velocity_length_scale = f64::INFINITY;
        assert_eq!(
            compare_phase_space_measures_mmd(&[], &[], invalid),
            Err(MmdError::InvalidVelocityLengthScale)
        );
    }

    #[test]
    fn non_finite_particle_data_is_rejected() {
        let invalid = [particle(f64::NAN, 0.0, 1.0)];
        assert!(matches!(
            compare_phase_space_measures_mmd(&invalid, &[], config()),
            Err(MmdError::NonFinitePosition {
                measure: "candidate",
                particle: 0
            })
        ));

        let invalid = [particle(0.0, f64::INFINITY, 1.0)];
        assert!(matches!(
            compare_phase_space_measures_mmd(&[], &invalid, config()),
            Err(MmdError::NonFiniteVelocity {
                measure: "reference",
                particle: 0
            })
        ));

        let invalid = [particle(0.0, 0.0, f64::NAN)];
        assert!(matches!(
            compare_phase_space_measures_mmd(&invalid, &[], config()),
            Err(MmdError::NonFiniteWeight {
                measure: "candidate",
                particle: 0
            })
        ));
    }

    #[test]
    fn overflowing_total_variation_fails_closed() {
        let invalid = [particle(0.0, 0.0, f64::MAX), particle(1.0, 0.0, f64::MAX)];
        assert!(matches!(
            compare_phase_space_measures_mmd(&invalid, &[], config()),
            Err(MmdError::TotalVariationOverflow {
                measure: "candidate"
            })
        ));
    }
}
