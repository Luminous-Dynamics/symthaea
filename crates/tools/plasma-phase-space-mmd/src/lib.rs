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
    InvalidFeatureCount,
    InvalidAdaptiveConfig,
    InvalidMixtureConfig,
    UnsupportedSpatialConcentrationForSampling,
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
            Self::InvalidFeatureCount => write!(f, "spectral feature count must be positive"),
            Self::InvalidAdaptiveConfig => {
                write!(f, "adaptive spectral MMD configuration is invalid")
            }
            Self::InvalidMixtureConfig => {
                write!(f, "multi-scale MMD mixture configuration is invalid")
            }
            Self::UnsupportedSpatialConcentrationForSampling => write!(
                f,
                "spatial concentration exceeds the deterministic spectral sampler range"
            ),
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

/// Configuration for the linear-cost spectral MMD approximation.
///
/// The seed and feature count are part of the benchmark identity. A run with
/// the same inputs, kernel configuration, seed and feature count is exactly
/// reproducible on the same floating-point platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpectralMmdConfig {
    pub feature_count: usize,
    pub seed: u64,
}

/// Monte Carlo approximation of normalized squared MMD.
///
/// Each feature contributes one independent nonnegative sample of the exact
/// squared MMD spectral integral. The reported Hoeffding radius is a 95%
/// feature-sampling bound over random feature draws, not a confidence interval
/// for physical solver error or finite-particle sampling uncertainty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralMmdResult {
    pub normalized_mmd_squared: f64,
    pub normalized_mmd: f64,
    pub feature_sample_variance: f64,
    pub feature_standard_error: f64,
    pub hoeffding_95_half_width: f64,
    pub feature_bound_lower: f64,
    pub feature_bound_upper: f64,
    pub feature_count: usize,
    pub seed: u64,
    pub candidate_particle_count: usize,
    pub reference_particle_count: usize,
    pub candidate_atom_count: usize,
    pub reference_atom_count: usize,
    pub particle_feature_evaluations: u128,
}

#[derive(Debug, Clone, Copy)]
struct SpectralFeature {
    spatial_harmonic: i64,
    velocity_frequency: f64,
}

/// Small, explicit deterministic PRNG. SplitMix64 is used only to make
/// benchmark feature draws reproducible; it is not a cryptographic generator.
#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    /// Uniform on the open interval (0, 1), avoiding log(0) and endpoint
    /// singularities in Box-Muller and transformed-rejection sampling.
    fn uniform_open(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 12;
        (mantissa as f64 + 1.0) / 4_503_599_627_370_497.0
    }

    fn standard_normal(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform_open().ln()).sqrt();
        let angle = std::f64::consts::TAU * self.uniform_open();
        radius * angle.cos()
    }
}

/// log(k!) with direct summation for small k and a high-order Stirling
/// expansion for larger k. The expansion error is negligible at k >= 256.
fn log_factorial(k: u64) -> f64 {
    if k < 256 {
        return (2..=k).map(|value| (value as f64).ln()).sum();
    }

    let x = k as f64;
    let inverse = 1.0 / x;
    let inverse_squared = inverse * inverse;
    (x + 0.5) * x.ln() - x + 0.918_938_533_204_672_7 + inverse / 12.0
        - inverse * inverse_squared / 360.0
        + inverse * inverse_squared * inverse_squared / 1260.0
        - inverse * inverse_squared * inverse_squared * inverse_squared / 1680.0
}

/// Draw one Poisson(lambda) value. Knuth inversion is efficient for small
/// lambda; Hörmann PTRS transformed rejection avoids O(lambda) work for large
/// lambda. The spatial von Mises spectrum is sampled as a Skellam difference
/// of two independent Poisson(kappa/2) variates.
fn sample_poisson(lambda: f64, rng: &mut SplitMix64) -> Result<u64, MmdError> {
    if !lambda.is_finite() || !(0.0..=1.0e12).contains(&lambda) {
        return Err(MmdError::UnsupportedSpatialConcentrationForSampling);
    }
    if lambda == 0.0 {
        return Ok(0);
    }

    if lambda < 10.0 {
        let limit = (-lambda).exp();
        let mut product = 1.0;
        let mut count = 0_u64;
        loop {
            count = count.saturating_add(1);
            product *= rng.uniform_open();
            if product <= limit {
                return Ok(count - 1);
            }
        }
    }

    let square_root = lambda.sqrt();
    let b = 0.931 + 2.53 * square_root;
    let a = -0.059 + 0.02483 * b;
    let inverse_alpha = 1.1239 + 1.1328 / (b - 3.4);
    let quick_acceptance = 0.9277 - 3.6224 / (b - 2.0);

    loop {
        let u = rng.uniform_open() - 0.5;
        let v = rng.uniform_open();
        let distance = 0.5 - u.abs();
        let proposal = ((2.0 * a / distance + b) * u + lambda + 0.43).floor();

        if proposal < 0.0 || proposal > (i64::MAX as f64) {
            continue;
        }
        let candidate = proposal as u64;

        if distance >= 0.07 && v <= quick_acceptance {
            return Ok(candidate);
        }
        if distance < 0.013 && v > distance {
            continue;
        }

        let lhs = (v * inverse_alpha / (a / (distance * distance) + b)).ln();
        let rhs = -lambda + proposal * lambda.ln() - log_factorial(candidate);
        if lhs <= rhs {
            return Ok(candidate);
        }
    }
}

/// Draw from the normalized Fourier spectrum of the periodic von Mises
/// factor. Its coefficient at integer n is exp(-kappa) I_|n|(kappa), exactly
/// the Skellam(lambda=kappa/2, lambda=kappa/2) probability mass.
fn sample_spatial_harmonic(concentration: f64, rng: &mut SplitMix64) -> Result<i64, MmdError> {
    let lambda = concentration * 0.5;
    let left = sample_poisson(lambda, rng)?;
    let right = sample_poisson(lambda, rng)?;
    let difference = i128::from(left) - i128::from(right);
    i64::try_from(difference).map_err(|_| MmdError::UnsupportedSpatialConcentrationForSampling)
}

fn sample_spectral_feature(
    config: MmdConfig,
    rng: &mut SplitMix64,
) -> Result<SpectralFeature, MmdError> {
    let spatial_harmonic = sample_spatial_harmonic(config.spatial_concentration, rng)?;
    let velocity_frequency = rng.standard_normal() / config.velocity_length_scale;
    if !velocity_frequency.is_finite() {
        return Err(MmdError::NonFiniteOutput);
    }
    Ok(SpectralFeature {
        spatial_harmonic,
        velocity_frequency,
    })
}

/// Evaluate one complex Fourier feature for a canonical weighted atom.
fn feature_value(atom: WeightedAtom, feature: SpectralFeature, config: MmdConfig) -> (f64, f64) {
    let spatial_phase = std::f64::consts::TAU
        * feature.spatial_harmonic as f64
        * (atom.position / config.domain_length);
    let phase = spatial_phase + feature.velocity_frequency * atom.velocity;
    let (sine, cosine) = phase.sin_cos();
    (cosine, sine)
}

fn spectral_squared_signature_difference(
    candidate_atoms: &[WeightedAtom],
    reference_atoms: &[WeightedAtom],
    normalization: f64,
    feature: SpectralFeature,
    kernel_config: MmdConfig,
) -> Result<f64, MmdError> {
    let mut candidate_real = CompensatedSum::default();
    let mut candidate_imaginary = CompensatedSum::default();
    let mut reference_real = CompensatedSum::default();
    let mut reference_imaginary = CompensatedSum::default();

    for atom in candidate_atoms {
        let (real, imaginary) = feature_value(*atom, feature, kernel_config);
        candidate_real.add((atom.weight / normalization) * real);
        candidate_imaginary.add((atom.weight / normalization) * imaginary);
    }
    for atom in reference_atoms {
        let (real, imaginary) = feature_value(*atom, feature, kernel_config);
        reference_real.add((atom.weight / normalization) * real);
        reference_imaginary.add((atom.weight / normalization) * imaginary);
    }

    let real_difference = candidate_real.total() - reference_real.total();
    let imaginary_difference = candidate_imaginary.total() - reference_imaginary.total();
    let squared_difference =
        real_difference * real_difference + imaginary_difference * imaginary_difference;
    if !squared_difference.is_finite() || !(0.0..=4.0 + 1.0e-10).contains(&squared_difference) {
        return Err(MmdError::NonFiniteOutput);
    }
    Ok(squared_difference)
}

/// Compare two phase-space measures using O((N+M)D) spectral features.
///
/// The kernel spectral measure factorizes exactly:
/// - spatial harmonic n follows Skellam(kappa/2, kappa/2), since the periodic
///   von Mises Fourier coefficients are exp(-kappa) I_|n|(kappa);
/// - velocity frequency follows Normal(0, 1/ell_v^2), the Fourier spectrum of
///   the Gaussian velocity kernel.
///
/// Both measures use the same sampled features (common random numbers). Each
/// squared complex signature difference is an unbiased, nonnegative sample of
/// the normalized squared MMD spectral integral. The input measures are first
/// canonicalized and co-located signed weights aggregated, exactly as in the
/// small-N reference oracle.
pub fn compare_phase_space_measures_spectral(
    candidate: &[PhaseSpaceParticle1d],
    reference: &[PhaseSpaceParticle1d],
    kernel_config: MmdConfig,
    feature_config: SpectralMmdConfig,
) -> Result<SpectralMmdResult, MmdError> {
    validate_config(kernel_config)?;
    if feature_config.feature_count == 0 || feature_config.feature_count > 10_000_000 {
        return Err(MmdError::InvalidFeatureCount);
    }

    let (candidate_atoms, candidate_tv) =
        aggregate_measure(candidate, "candidate", kernel_config.domain_length)?;
    let (reference_atoms, reference_tv) =
        aggregate_measure(reference, "reference", kernel_config.domain_length)?;

    let total_variation_scale = candidate_tv.max(reference_tv);
    let normalization = if total_variation_scale == 0.0 {
        1.0
    } else {
        total_variation_scale
    };

    let mut rng = SplitMix64::new(feature_config.seed);
    let mut mean = 0.0;
    let mut sum_squared_deviation = 0.0;

    for sample_index in 0..feature_config.feature_count {
        let feature = sample_spectral_feature(kernel_config, &mut rng)?;
        let squared_difference = spectral_squared_signature_difference(
            &candidate_atoms,
            &reference_atoms,
            normalization,
            feature,
            kernel_config,
        )?;

        // Welford's online variance update avoids storing D samples.
        let count = (sample_index + 1) as f64;
        let delta = squared_difference - mean;
        mean += delta / count;
        let delta_after_update = squared_difference - mean;
        sum_squared_deviation += delta * delta_after_update;
    }

    if !mean.is_finite() || !sum_squared_deviation.is_finite() {
        return Err(MmdError::NonFiniteOutput);
    }

    let feature_sample_variance = if feature_config.feature_count > 1 {
        (sum_squared_deviation / (feature_config.feature_count - 1) as f64).max(0.0)
    } else {
        0.0
    };
    let feature_standard_error =
        (feature_sample_variance / feature_config.feature_count as f64).sqrt();

    // Each normalized measure has total variation <= 1. The difference of
    // two characteristic coefficients has magnitude <= 2, hence q in [0,4].
    // Hoeffding at alpha=0.05 gives this conservative two-sided feature-sampling
    // radius; it is not a bound on particle-sampling or physical solver error.
    let hoeffding_95_half_width =
        4.0 * ((2.0 / 0.05_f64).ln() / (2.0 * feature_config.feature_count as f64)).sqrt();
    let feature_bound_lower = (mean - hoeffding_95_half_width).max(0.0);
    let feature_bound_upper = (mean + hoeffding_95_half_width).min(4.0);
    let normalized_mmd = mean.sqrt();

    let particle_feature_evaluations = (candidate_atoms.len() as u128)
        .checked_add(reference_atoms.len() as u128)
        .and_then(|count| count.checked_mul(feature_config.feature_count as u128))
        .ok_or(MmdError::PairCountOverflow)?;

    if !normalized_mmd.is_finite()
        || !feature_sample_variance.is_finite()
        || !feature_standard_error.is_finite()
        || !hoeffding_95_half_width.is_finite()
    {
        return Err(MmdError::NonFiniteOutput);
    }

    Ok(SpectralMmdResult {
        normalized_mmd_squared: mean,
        normalized_mmd,
        feature_sample_variance,
        feature_standard_error,
        hoeffding_95_half_width,
        feature_bound_lower,
        feature_bound_upper,
        feature_count: feature_config.feature_count,
        seed: feature_config.seed,
        candidate_particle_count: candidate.len(),
        reference_particle_count: reference.len(),
        candidate_atom_count: candidate_atoms.len(),
        reference_atom_count: reference_atoms.len(),
        particle_feature_evaluations,
    })
}

/// Configuration for checkpointed, anytime-valid spectral MMD evaluation.
///
/// Checkpoints double from initial_feature_count up to max_feature_count. The
/// tolerance is applied to normalized squared MMD, not its square root.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveSpectralMmdConfig {
    pub initial_feature_count: usize,
    pub max_feature_count: usize,
    pub confidence_level: f64,
    pub tolerance: f64,
    pub seed: u64,
}

/// Decision for the distribution-fidelity sub-gate only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveMmdDecision {
    WithinTolerance,
    AboveTolerance,
    UnresolvedBudgetExhausted,
}

/// One inspected checkpoint in the alpha-spending confidence sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveMmdCheckpoint {
    pub feature_count: usize,
    pub alpha_for_checkpoint: f64,
    pub cumulative_alpha_spent: f64,
    pub normalized_mmd_squared_estimate: f64,
    pub interval_radius: f64,
    pub interval_lower: f64,
    pub interval_upper: f64,
}

/// Result of adaptive spectral MMD with simultaneous checkpoint coverage.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveSpectralMmdResult {
    pub decision: AdaptiveMmdDecision,
    pub normalized_mmd_squared: f64,
    pub normalized_mmd: f64,
    pub feature_sample_variance: f64,
    pub feature_standard_error: f64,
    pub feature_bound_lower: f64,
    pub feature_bound_upper: f64,
    pub tolerance: f64,
    pub confidence_level: f64,
    pub alpha_budget: f64,
    pub alpha_spent: f64,
    pub feature_count: usize,
    pub seed: u64,
    pub candidate_particle_count: usize,
    pub reference_particle_count: usize,
    pub candidate_atom_count: usize,
    pub reference_atom_count: usize,
    pub particle_feature_evaluations: u128,
    pub checkpoints: Vec<AdaptiveMmdCheckpoint>,
}

/// Evaluate normalized squared MMD adaptively with a time-uniform guarantee
/// over the preregistered geometric checkpoints.
///
/// At look t, alpha_t = alpha/(t*(t+1)); the sum over all looks is alpha.
/// Hoeffding intervals at the inspected checkpoints therefore have simultaneous
/// coverage at least confidence_level by a union bound. The function only
/// inspects at checkpoint boundaries; it never peeks mid-batch.
///
/// If the interval is wholly below the frozen tolerance, the MMD sub-gate is
/// within tolerance. If wholly above, it fails. If still overlapping at the
/// feature budget limit, the result is unresolved and must not be accepted.
/// This does not authorize a physical trajectory without all other gates.
pub fn compare_phase_space_measures_spectral_adaptive(
    candidate: &[PhaseSpaceParticle1d],
    reference: &[PhaseSpaceParticle1d],
    kernel_config: MmdConfig,
    adaptive_config: AdaptiveSpectralMmdConfig,
) -> Result<AdaptiveSpectralMmdResult, MmdError> {
    validate_config(kernel_config)?;
    if adaptive_config.initial_feature_count == 0
        || adaptive_config.max_feature_count < adaptive_config.initial_feature_count
        || adaptive_config.max_feature_count > 10_000_000
        || adaptive_config.max_feature_count == 0
        || !adaptive_config.confidence_level.is_finite()
        || !(0.0..1.0).contains(&adaptive_config.confidence_level)
        || !adaptive_config.tolerance.is_finite()
        || !(0.0..=4.0).contains(&adaptive_config.tolerance)
    {
        return Err(MmdError::InvalidAdaptiveConfig);
    }

    let (candidate_atoms, candidate_tv) =
        aggregate_measure(candidate, "candidate", kernel_config.domain_length)?;
    let (reference_atoms, reference_tv) =
        aggregate_measure(reference, "reference", kernel_config.domain_length)?;
    let total_variation_scale = candidate_tv.max(reference_tv);
    let normalization = if total_variation_scale == 0.0 {
        1.0
    } else {
        total_variation_scale
    };

    let alpha_budget = 1.0 - adaptive_config.confidence_level;
    let mut cumulative_alpha_spent = 0.0;
    let mut rng = SplitMix64::new(adaptive_config.seed);
    let mut feature_count = 0_usize;
    let mut target_feature_count = adaptive_config.initial_feature_count;
    let mut mean = 0.0;
    let mut sum_squared_deviation = 0.0;
    let mut look_index = 1_usize;
    let mut checkpoints = Vec::new();
    let (decision, final_lower, final_upper) = loop {
        while feature_count < target_feature_count {
            let feature = sample_spectral_feature(kernel_config, &mut rng)?;
            let squared_difference = spectral_squared_signature_difference(
                &candidate_atoms,
                &reference_atoms,
                normalization,
                feature,
                kernel_config,
            )?;
            feature_count += 1;

            // Welford's online variance update; only the running statistics
            // are retained, not the individual D feature samples.
            let count = feature_count as f64;
            let delta = squared_difference - mean;
            mean += delta / count;
            let delta_after_update = squared_difference - mean;
            sum_squared_deviation += delta * delta_after_update;
        }

        let look = look_index as f64;
        let alpha_for_checkpoint = alpha_budget / (look * (look + 1.0));
        cumulative_alpha_spent += alpha_for_checkpoint;
        let interval_radius =
            4.0 * ((2.0 / alpha_for_checkpoint).ln() / (2.0 * feature_count as f64)).sqrt();
        let final_lower = (mean - interval_radius).max(0.0);
        let final_upper = (mean + interval_radius).min(4.0);

        if !mean.is_finite()
            || !sum_squared_deviation.is_finite()
            || !alpha_for_checkpoint.is_finite()
            || !interval_radius.is_finite()
            || !final_lower.is_finite()
            || !final_upper.is_finite()
        {
            return Err(MmdError::NonFiniteOutput);
        }

        checkpoints.push(AdaptiveMmdCheckpoint {
            feature_count,
            alpha_for_checkpoint,
            cumulative_alpha_spent,
            normalized_mmd_squared_estimate: mean,
            interval_radius,
            interval_lower: final_lower,
            interval_upper: final_upper,
        });

        if final_upper <= adaptive_config.tolerance {
            break (
                AdaptiveMmdDecision::WithinTolerance,
                final_lower,
                final_upper,
            );
        }
        if final_lower > adaptive_config.tolerance {
            break (
                AdaptiveMmdDecision::AboveTolerance,
                final_lower,
                final_upper,
            );
        }
        if feature_count >= adaptive_config.max_feature_count {
            break (
                AdaptiveMmdDecision::UnresolvedBudgetExhausted,
                final_lower,
                final_upper,
            );
        }

        target_feature_count = target_feature_count
            .saturating_mul(2)
            .min(adaptive_config.max_feature_count);
        look_index = look_index.saturating_add(1);
    };

    let feature_sample_variance = if feature_count > 1 {
        (sum_squared_deviation / (feature_count - 1) as f64).max(0.0)
    } else {
        0.0
    };
    let feature_standard_error = (feature_sample_variance / feature_count as f64).sqrt();
    let normalized_mmd = mean.sqrt();
    let particle_feature_evaluations = (candidate_atoms.len() as u128)
        .checked_add(reference_atoms.len() as u128)
        .and_then(|count| count.checked_mul(feature_count as u128))
        .ok_or(MmdError::PairCountOverflow)?;

    if !feature_sample_variance.is_finite()
        || !feature_standard_error.is_finite()
        || !normalized_mmd.is_finite()
        || !cumulative_alpha_spent.is_finite()
        || cumulative_alpha_spent > alpha_budget + 1.0e-12
    {
        return Err(MmdError::NonFiniteOutput);
    }

    Ok(AdaptiveSpectralMmdResult {
        decision,
        normalized_mmd_squared: mean,
        normalized_mmd,
        feature_sample_variance,
        feature_standard_error,
        feature_bound_lower: final_lower,
        feature_bound_upper: final_upper,
        tolerance: adaptive_config.tolerance,
        confidence_level: adaptive_config.confidence_level,
        alpha_budget,
        alpha_spent: cumulative_alpha_spent,
        feature_count,
        seed: adaptive_config.seed,
        candidate_particle_count: candidate.len(),
        reference_particle_count: reference.len(),
        candidate_atom_count: candidate_atoms.len(),
        reference_atom_count: reference_atoms.len(),
        particle_feature_evaluations,
        checkpoints,
    })
}


/// One positive component in a frozen characteristic-kernel mixture.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MmdKernelComponent {
    /// Positive mixture mass. Component masses are normalized after a
    /// strict unit-sum validation (absolute tolerance 1e-12).
    pub weight: f64,
    /// Positive von Mises spatial concentration.
    pub spatial_concentration: f64,
    /// Positive Gaussian velocity length scale.
    pub velocity_length_scale: f64,
}

/// A finite positive mixture of characteristic kernels on S^1_L x R.
///
/// Every component is unit diagonal and characteristic when its scales are
/// positive. A positive mixture of such kernels remains characteristic.
#[derive(Debug, Clone, PartialEq)]
pub struct MmdMixtureConfig {
    pub domain_length: f64,
    pub components: Vec<MmdKernelComponent>,
}

/// Spectral mixture estimate together with component allocation provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct SpectralMmdMixtureResult {
    pub estimate: SpectralMmdResult,
    /// Number of spectral features sampled from each configured component.
    pub component_feature_counts: Vec<usize>,
}

fn validate_mixture_config(config: &MmdMixtureConfig) -> Result<f64, MmdError> {
    if !config.domain_length.is_finite()
        || config.domain_length <= 0.0
        || config.components.is_empty()
        || config.components.len() > 32
    {
        return Err(MmdError::InvalidMixtureConfig);
    }

    let mut total_weight = CompensatedSum::default();
    for (index, component) in config.components.iter().enumerate() {
        if !component.weight.is_finite()
            || component.weight <= 0.0
            || !component.spatial_concentration.is_finite()
            || component.spatial_concentration <= 0.0
            || !component.velocity_length_scale.is_finite()
            || component.velocity_length_scale <= 0.0
        {
            return Err(MmdError::InvalidMixtureConfig);
        }
        if config.components[..index].iter().any(|previous| {
            previous.spatial_concentration == component.spatial_concentration
                && previous.velocity_length_scale == component.velocity_length_scale
        }) {
            return Err(MmdError::InvalidMixtureConfig);
        }
        total_weight.add(component.weight);
    }

    let total_weight = total_weight.total();
    if !total_weight.is_finite() || (total_weight - 1.0).abs() > 1.0e-12 {
        return Err(MmdError::InvalidMixtureConfig);
    }
    Ok(total_weight)
}

fn component_config(domain_length: f64, component: MmdKernelComponent) -> MmdConfig {
    MmdConfig {
        domain_length,
        spatial_concentration: component.spatial_concentration,
        velocity_length_scale: component.velocity_length_scale,
    }
}

fn mixture_kernel(
    left: WeightedAtom,
    right: WeightedAtom,
    config: &MmdMixtureConfig,
    total_weight: f64,
) -> f64 {
    let mut sum = CompensatedSum::default();
    for component in &config.components {
        let weight = component.weight / total_weight;
        sum.add(weight * kernel(left, right, component_config(config.domain_length, *component)));
    }
    sum.total()
}

fn quadratic_sum_mixture(
    left: &[WeightedAtom],
    right: &[WeightedAtom],
    normalization: f64,
    config: &MmdMixtureConfig,
    total_weight: f64,
) -> (f64, f64) {
    let mut sum = CompensatedSum::default();
    let mut absolute_sum = CompensatedSum::default();
    for left_atom in left {
        let left_weight = left_atom.weight / normalization;
        for right_atom in right {
            let right_weight = right_atom.weight / normalization;
            let term = left_weight
                * right_weight
                * mixture_kernel(*left_atom, *right_atom, config, total_weight);
            sum.add(term);
            absolute_sum.add(term.abs());
        }
    }
    (sum.total(), absolute_sum.total())
}

/// Exact small-N MMD reference for a frozen positive mixture of characteristic
/// periodic × Gaussian kernels. Co-located signed atoms are canonicalized by
/// the same function and total-variation convention as the single-scale oracle.
pub fn compare_phase_space_measures_mmd_mixture(
    candidate: &[PhaseSpaceParticle1d],
    reference: &[PhaseSpaceParticle1d],
    config: &MmdMixtureConfig,
) -> Result<MmdResult, MmdError> {
    let total_weight = validate_mixture_config(config)?;
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

    let (candidate_self, candidate_abs) = quadratic_sum_mixture(
        &candidate_atoms,
        &candidate_atoms,
        normalization,
        config,
        total_weight,
    );
    let (reference_self, reference_abs) = quadratic_sum_mixture(
        &reference_atoms,
        &reference_atoms,
        normalization,
        config,
        total_weight,
    );
    let (cross, cross_abs) = quadratic_sum_mixture(
        &candidate_atoms,
        &reference_atoms,
        normalization,
        config,
        total_weight,
    );

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

fn sample_mixture_component(
    config: &MmdMixtureConfig,
    total_weight: f64,
    rng: &mut SplitMix64,
) -> usize {
    let target = rng.uniform_open();
    let mut cumulative = 0.0;
    for (index, component) in config.components.iter().enumerate() {
        cumulative += component.weight / total_weight;
        if target < cumulative {
            return index;
        }
    }
    // The last branch absorbs only floating-point CDF roundoff; all component
    // weights were validated positive and the sum normalized above.
    config.components.len() - 1
}

/// Common-random-feature approximation for a frozen positive kernel mixture.
///
/// First sample component r with probability beta_r, then sample a frequency
/// from that component's exact Skellam × Gaussian spectrum. This samples the
/// complete spectral measure of the mixture without truncating any component.
/// The returned [0,4] Hoeffding interval concerns random-feature integration
/// error in normalized squared MMD only.
pub fn compare_phase_space_measures_spectral_mixture(
    candidate: &[PhaseSpaceParticle1d],
    reference: &[PhaseSpaceParticle1d],
    mixture: &MmdMixtureConfig,
    feature_config: SpectralMmdConfig,
) -> Result<SpectralMmdMixtureResult, MmdError> {
    let total_weight = validate_mixture_config(mixture)?;
    if feature_config.feature_count == 0 || feature_config.feature_count > 10_000_000 {
        return Err(MmdError::InvalidFeatureCount);
    }

    let (candidate_atoms, candidate_tv) =
        aggregate_measure(candidate, "candidate", mixture.domain_length)?;
    let (reference_atoms, reference_tv) =
        aggregate_measure(reference, "reference", mixture.domain_length)?;
    let total_variation_scale = candidate_tv.max(reference_tv);
    let normalization = if total_variation_scale == 0.0 {
        1.0
    } else {
        total_variation_scale
    };

    let mut rng = SplitMix64::new(feature_config.seed);
    let mut component_feature_counts = vec![0_usize; mixture.components.len()];
    let mut mean = 0.0;
    let mut sum_squared_deviation = 0.0;

    for sample_index in 0..feature_config.feature_count {
        let component_index = sample_mixture_component(mixture, total_weight, &mut rng);
        component_feature_counts[component_index] += 1;
        let kernel_config = component_config(
            mixture.domain_length,
            mixture.components[component_index],
        );
        let feature = sample_spectral_feature(kernel_config, &mut rng)?;
        let squared_difference = spectral_squared_signature_difference(
            &candidate_atoms,
            &reference_atoms,
            normalization,
            feature,
            kernel_config,
        )?;

        let count = (sample_index + 1) as f64;
        let delta = squared_difference - mean;
        mean += delta / count;
        let delta_after_update = squared_difference - mean;
        sum_squared_deviation += delta * delta_after_update;
    }

    if !mean.is_finite() || !sum_squared_deviation.is_finite() {
        return Err(MmdError::NonFiniteOutput);
    }
    let feature_sample_variance = if feature_config.feature_count > 1 {
        (sum_squared_deviation / (feature_config.feature_count - 1) as f64).max(0.0)
    } else {
        0.0
    };
    let feature_standard_error =
        (feature_sample_variance / feature_config.feature_count as f64).sqrt();
    let hoeffding_95_half_width =
        4.0 * ((2.0 / 0.05_f64).ln() / (2.0 * feature_config.feature_count as f64)).sqrt();
    let feature_bound_lower = (mean - hoeffding_95_half_width).max(0.0);
    let feature_bound_upper = (mean + hoeffding_95_half_width).min(4.0);
    let normalized_mmd = mean.sqrt();

    let particle_feature_evaluations = (candidate_atoms.len() as u128)
        .checked_add(reference_atoms.len() as u128)
        .and_then(|count| count.checked_mul(feature_config.feature_count as u128))
        .ok_or(MmdError::PairCountOverflow)?;

    if !normalized_mmd.is_finite()
        || !feature_sample_variance.is_finite()
        || !feature_standard_error.is_finite()
        || !hoeffding_95_half_width.is_finite()
    {
        return Err(MmdError::NonFiniteOutput);
    }

    Ok(SpectralMmdMixtureResult {
        estimate: SpectralMmdResult {
            normalized_mmd_squared: mean,
            normalized_mmd,
            feature_sample_variance,
            feature_standard_error,
            hoeffding_95_half_width,
            feature_bound_lower,
            feature_bound_upper,
            feature_count: feature_config.feature_count,
            seed: feature_config.seed,
            candidate_particle_count: candidate.len(),
            reference_particle_count: reference.len(),
            candidate_atom_count: candidate_atoms.len(),
            reference_atom_count: reference_atoms.len(),
            particle_feature_evaluations,
        },
        component_feature_counts,
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

    #[test]
    fn spectral_estimator_is_reproducible_for_frozen_seed() {
        let candidate = [particle(0.5, -0.25, 0.4), particle(5.0, 1.5, 0.6)];
        let reference = [particle(0.75, -0.1, 0.4), particle(4.5, 1.25, 0.6)];
        let features = SpectralMmdConfig {
            feature_count: 512,
            seed: 0x5eed_cafe,
        };
        let first =
            compare_phase_space_measures_spectral(&candidate, &reference, config(), features)
                .unwrap();
        let second =
            compare_phase_space_measures_spectral(&candidate, &reference, config(), features)
                .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.feature_count, 512);
        assert_eq!(first.seed, 0x5eed_cafe);
    }

    #[test]
    fn spectral_estimator_uses_linear_particle_feature_work() {
        let candidate = [particle(0.5, 0.0, 0.5), particle(1.5, 1.0, 0.5)];
        let reference = [particle(0.75, 0.0, 0.5), particle(2.5, 1.0, 0.5)];
        let result = compare_phase_space_measures_spectral(
            &candidate,
            &reference,
            config(),
            SpectralMmdConfig {
                feature_count: 100,
                seed: 17,
            },
        )
        .unwrap();
        assert_eq!(result.particle_feature_evaluations, 400);
        assert_eq!(result.candidate_atom_count, 2);
        assert_eq!(result.reference_atom_count, 2);
        assert!(result.feature_standard_error.is_finite());
        assert!(result.hoeffding_95_half_width > 0.0);
        assert!(result.feature_bound_lower <= result.normalized_mmd_squared);
        assert!(result.feature_bound_upper >= result.normalized_mmd_squared);
    }

    #[test]
    fn spectral_estimator_preserves_permutation_and_signed_split_invariance() {
        let candidate = [particle(1.0, 0.25, 0.5), particle(6.0, -0.5, 0.5)];
        let permutation = [particle(6.0, -0.5, 0.5), particle(1.0, 0.25, 0.5)];
        let signed_split = [
            particle(1.0, 0.25, 0.75),
            particle(1.0, 0.25, -0.25),
            particle(6.0, -0.5, 0.2),
            particle(6.0, -0.5, 0.3),
        ];
        let features = SpectralMmdConfig {
            feature_count: 1024,
            seed: 12345,
        };
        let permuted =
            compare_phase_space_measures_spectral(&candidate, &permutation, config(), features)
                .unwrap();
        let split =
            compare_phase_space_measures_spectral(&candidate, &signed_split, config(), features)
                .unwrap();
        assert!(permuted.normalized_mmd <= 1.0e-12);
        assert!(split.normalized_mmd <= 1.0e-12);
        assert_eq!(split.reference_atom_count, 2);
    }

    #[test]
    fn spectral_estimator_tracks_exact_oracle_with_reported_bound() {
        let candidate = [
            particle(0.0, -0.5, 0.2),
            particle(1.5, 0.25, 0.3),
            particle(5.0, 1.25, 0.5),
        ];
        let reference = [
            particle(0.25, -0.4, 0.2),
            particle(2.0, 0.5, 0.3),
            particle(5.5, 1.0, 0.5),
        ];
        let exact = compare_phase_space_measures_mmd(&candidate, &reference, config()).unwrap();
        let approximate = compare_phase_space_measures_spectral(
            &candidate,
            &reference,
            config(),
            SpectralMmdConfig {
                feature_count: 8192,
                seed: 0x2026_1121,
            },
        )
        .unwrap();
        let absolute_error =
            (approximate.normalized_mmd_squared - exact.normalized_mmd_squared).abs();
        assert!(
            absolute_error <= approximate.hoeffding_95_half_width,
            "squared-MMD error {absolute_error:e} exceeded feature-sampling bound {:e}",
            approximate.hoeffding_95_half_width
        );
    }

    #[test]
    fn spectral_estimator_resolves_the_finite_fourier_alias_control() {
        let length = config().domain_length;
        let candidate = (0..4)
            .map(|index| particle(f64::from(index) * length / 4.0, 0.0, 0.25))
            .collect::<Vec<_>>();
        let reference = (0..4)
            .map(|index| particle(length / 8.0 + f64::from(index) * length / 4.0, 0.0, 0.25))
            .collect::<Vec<_>>();
        let result = compare_phase_space_measures_spectral(
            &candidate,
            &reference,
            config(),
            SpectralMmdConfig {
                feature_count: 4096,
                seed: 99,
            },
        )
        .unwrap();
        assert!(
            result.normalized_mmd_squared > 0.02,
            "spectral approximation failed the alias negative control: {:?}",
            result
        );
    }

    #[test]
    fn spectral_estimator_rejects_zero_feature_count() {
        assert_eq!(
            compare_phase_space_measures_spectral(
                &[],
                &[],
                config(),
                SpectralMmdConfig {
                    feature_count: 0,
                    seed: 1,
                },
            ),
            Err(MmdError::InvalidFeatureCount)
        );
    }

    #[test]
    fn spectral_skellam_sampler_has_expected_moments() {
        let concentration = 30.0;
        let mut rng = SplitMix64::new(0x5eed);
        let samples = 50_000_u64;
        let mut sum = 0.0;
        let mut sum_squared = 0.0;
        for _ in 0..samples {
            let harmonic = sample_spatial_harmonic(concentration, &mut rng).unwrap() as f64;
            sum += harmonic;
            sum_squared += harmonic * harmonic;
        }
        let mean = sum / samples as f64;
        let variance = sum_squared / samples as f64 - mean * mean;
        assert!(mean.abs() < 0.12, "Skellam mean was {mean}");
        assert!(
            (variance - concentration).abs() < 0.8,
            "Skellam variance was {variance}, expected {concentration}"
        );
    }

    #[test]
    fn spectral_equal_empty_measures_have_zero_estimate() {
        let result = compare_phase_space_measures_spectral(
            &[],
            &[],
            config(),
            SpectralMmdConfig {
                feature_count: 256,
                seed: 4,
            },
        )
        .unwrap();
        assert_eq!(result.normalized_mmd_squared, 0.0);
        assert_eq!(result.normalized_mmd, 0.0);
        assert_eq!(result.particle_feature_evaluations, 0);
    }

    #[test]
    fn adaptive_mmd_stops_early_when_distribution_is_within_tolerance() {
        let measure = [particle(0.5, -0.25, 0.4), particle(5.0, 1.5, 0.6)];
        let result = compare_phase_space_measures_spectral_adaptive(
            &measure,
            &measure,
            config(),
            AdaptiveSpectralMmdConfig {
                initial_feature_count: 64,
                max_feature_count: 4096,
                confidence_level: 0.95,
                tolerance: 1.0,
                seed: 51,
            },
        )
        .unwrap();
        assert_eq!(result.decision, AdaptiveMmdDecision::WithinTolerance);
        assert_eq!(result.feature_count, 64);
        assert_eq!(result.checkpoints.len(), 1);
        assert_eq!(result.normalized_mmd_squared, 0.0);
    }

    #[test]
    fn adaptive_mmd_stops_early_when_distribution_exceeds_tolerance() {
        let reference = [particle(2.0, -1.0, 1.0)];
        let result = compare_phase_space_measures_spectral_adaptive(
            &[],
            &reference,
            config(),
            AdaptiveSpectralMmdConfig {
                initial_feature_count: 64,
                max_feature_count: 4096,
                confidence_level: 0.95,
                tolerance: 0.1,
                seed: 52,
            },
        )
        .unwrap();
        assert_eq!(result.decision, AdaptiveMmdDecision::AboveTolerance);
        assert_eq!(result.feature_count, 64);
        assert_eq!(result.normalized_mmd_squared, 1.0);
    }

    #[test]
    fn adaptive_mmd_uses_alpha_spending_and_abstains_at_budget() {
        let reference = [particle(2.0, -1.0, 1.0)];
        let adaptive_config = AdaptiveSpectralMmdConfig {
            initial_feature_count: 64,
            max_feature_count: 256,
            confidence_level: 0.95,
            tolerance: 1.0,
            seed: 53,
        };
        let result = compare_phase_space_measures_spectral_adaptive(
            &[],
            &reference,
            config(),
            adaptive_config,
        )
        .unwrap();
        assert_eq!(
            result.decision,
            AdaptiveMmdDecision::UnresolvedBudgetExhausted
        );
        assert_eq!(
            result
                .checkpoints
                .iter()
                .map(|checkpoint| checkpoint.feature_count)
                .collect::<Vec<_>>(),
            vec![64, 128, 256]
        );
        assert!((result.alpha_spent - 0.0375).abs() <= 1.0e-12);
        assert!(result.alpha_spent <= result.alpha_budget);
        assert!(
            result
                .checkpoints
                .windows(2)
                .all(|pair| { pair[0].cumulative_alpha_spent < pair[1].cumulative_alpha_spent })
        );
    }

    #[test]
    fn adaptive_mmd_matches_the_same_frozen_feature_prefix() {
        let reference = [particle(2.0, -1.0, 1.0)];
        let seed = 0x2026_1121;
        let adaptive = compare_phase_space_measures_spectral_adaptive(
            &[],
            &reference,
            config(),
            AdaptiveSpectralMmdConfig {
                initial_feature_count: 64,
                max_feature_count: 256,
                confidence_level: 0.95,
                tolerance: 1.0,
                seed,
            },
        )
        .unwrap();
        let fixed = compare_phase_space_measures_spectral(
            &[],
            &reference,
            config(),
            SpectralMmdConfig {
                feature_count: 256,
                seed,
            },
        )
        .unwrap();
        assert_eq!(adaptive.feature_count, 256);
        assert_eq!(
            adaptive.normalized_mmd_squared,
            fixed.normalized_mmd_squared
        );
    }

    #[test]
    fn adaptive_mmd_rejects_invalid_alpha_threshold_and_budget() {
        let measure = [particle(0.0, 0.0, 1.0)];
        let invalid = [
            AdaptiveSpectralMmdConfig {
                initial_feature_count: 0,
                max_feature_count: 64,
                confidence_level: 0.95,
                tolerance: 0.1,
                seed: 1,
            },
            AdaptiveSpectralMmdConfig {
                initial_feature_count: 128,
                max_feature_count: 64,
                confidence_level: 0.95,
                tolerance: 0.1,
                seed: 1,
            },
            AdaptiveSpectralMmdConfig {
                initial_feature_count: 64,
                max_feature_count: 128,
                confidence_level: 1.0,
                tolerance: 0.1,
                seed: 1,
            },
            AdaptiveSpectralMmdConfig {
                initial_feature_count: 64,
                max_feature_count: 128,
                confidence_level: 0.95,
                tolerance: 4.1,
                seed: 1,
            },
        ];
        for adaptive_config in invalid {
            assert_eq!(
                compare_phase_space_measures_spectral_adaptive(
                    &measure,
                    &measure,
                    config(),
                    adaptive_config
                ),
                Err(MmdError::InvalidAdaptiveConfig)
            );
        }
    }

    fn mixture_config() -> MmdMixtureConfig {
        MmdMixtureConfig {
            domain_length: 8.0,
            components: vec![
                MmdKernelComponent {
                    weight: 0.2,
                    spatial_concentration: 0.5,
                    velocity_length_scale: 2.0,
                },
                MmdKernelComponent {
                    weight: 0.5,
                    spatial_concentration: 2.0,
                    velocity_length_scale: 1.0,
                },
                MmdKernelComponent {
                    weight: 0.3,
                    spatial_concentration: 20.0,
                    velocity_length_scale: 0.25,
                },
            ],
        }
    }

    #[test]
    fn exact_mixture_mmd_squared_is_weighted_sum_of_component_oracles() {
        let candidate = [
            particle(0.0, -0.5, 0.2),
            particle(1.5, 0.25, 0.3),
            particle(5.0, 1.25, 0.5),
        ];
        let reference = [
            particle(0.25, -0.4, 0.2),
            particle(2.0, 0.5, 0.3),
            particle(5.5, 1.0, 0.5),
        ];
        let mixture = mixture_config();
        let mixed =
            compare_phase_space_measures_mmd_mixture(&candidate, &reference, &mixture).unwrap();
        let total_weight = mixture.components.iter().map(|item| item.weight).sum::<f64>();
        let mut weighted_sum = 0.0;
        for component in &mixture.components {
            let exact = compare_phase_space_measures_mmd(
                &candidate,
                &reference,
                component_config(mixture.domain_length, *component),
            )
            .unwrap();
            weighted_sum += component.weight / total_weight * exact.normalized_mmd_squared;
        }
        assert!(
            (mixed.normalized_mmd_squared - weighted_sum).abs() <= 1.0e-12,
            "mixture MMD^2 {} differed from weighted component sum {weighted_sum}",
            mixed.normalized_mmd_squared
        );
    }

    #[test]
    fn exact_mixture_preserves_permutation_and_signed_split_invariance() {
        let candidate = [
            particle(1.0, 0.25, 0.5),
            particle(6.0, -0.5, 0.5),
        ];
        let permutation = [
            particle(6.0, -0.5, 0.5),
            particle(1.0, 0.25, 0.5),
        ];
        let signed_split = [
            particle(1.0, 0.25, 0.75),
            particle(1.0, 0.25, -0.25),
            particle(6.0, -0.5, 0.2),
            particle(6.0, -0.5, 0.3),
        ];
        let permuted =
            compare_phase_space_measures_mmd_mixture(&candidate, &permutation, &mixture_config())
                .unwrap();
        let split =
            compare_phase_space_measures_mmd_mixture(&candidate, &signed_split, &mixture_config())
                .unwrap();
        assert_mmd_zero(permuted);
        assert_mmd_zero(split);
        assert_eq!(split.reference_atom_count, 2);
    }

    #[test]
    fn exact_mixture_is_invariant_to_component_order() {
        let candidate = [particle(0.0, 0.0, 0.5), particle(4.0, 1.0, 0.5)];
        let reference = [particle(0.25, 0.0, 0.5), particle(4.25, 1.0, 0.5)];
        let forward = mixture_config();
        let mut reversed = forward.clone();
        reversed.components.reverse();
        let left =
            compare_phase_space_measures_mmd_mixture(&candidate, &reference, &forward).unwrap();
        let right =
            compare_phase_space_measures_mmd_mixture(&candidate, &reference, &reversed).unwrap();
        assert!(
            (left.normalized_mmd_squared - right.normalized_mmd_squared).abs() <= 1.0e-12
        );
    }

    #[test]
    fn spectral_mixture_is_reproducible_and_accounts_component_draws() {
        let candidate = [particle(0.0, 0.0, 0.5), particle(4.0, 1.0, 0.5)];
        let reference = [particle(0.25, 0.0, 0.5), particle(4.25, 1.0, 0.5)];
        let feature_config = SpectralMmdConfig {
            feature_count: 2048,
            seed: 0x51ca_1e,
        };
        let first = compare_phase_space_measures_spectral_mixture(
            &candidate,
            &reference,
            &mixture_config(),
            feature_config,
        )
        .unwrap();
        let second = compare_phase_space_measures_spectral_mixture(
            &candidate,
            &reference,
            &mixture_config(),
            feature_config,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.component_feature_counts.iter().sum::<usize>(),
            feature_config.feature_count
        );
        assert_eq!(first.estimate.particle_feature_evaluations, 8192);
    }

    #[test]
    fn spectral_mixture_tracks_exact_mixture_inside_feature_bound() {
        let candidate = [
            particle(0.0, -0.5, 0.2),
            particle(1.5, 0.25, 0.3),
            particle(5.0, 1.25, 0.5),
        ];
        let reference = [
            particle(0.25, -0.4, 0.2),
            particle(2.0, 0.5, 0.3),
            particle(5.5, 1.0, 0.5),
        ];
        let exact =
            compare_phase_space_measures_mmd_mixture(&candidate, &reference, &mixture_config())
                .unwrap();
        let approximate = compare_phase_space_measures_spectral_mixture(
            &candidate,
            &reference,
            &mixture_config(),
            SpectralMmdConfig {
                feature_count: 16_384,
                seed: 0x2026_1130,
            },
        )
        .unwrap();
        let error =
            (approximate.estimate.normalized_mmd_squared - exact.normalized_mmd_squared).abs();
        assert!(
            error <= approximate.estimate.hoeffding_95_half_width,
            "mixture spectral error {error:e} exceeds feature bound {:e}",
            approximate.estimate.hoeffding_95_half_width
        );
    }

    #[test]
    fn spectral_mixture_is_zero_for_equivalent_signed_measures() {
        let candidate = [particle(1.0, 0.25, 0.5), particle(6.0, -0.5, 0.5)];
        let reference = [
            particle(6.0, -0.5, 0.5),
            particle(1.0, 0.25, 0.75),
            particle(1.0, 0.25, -0.25),
        ];
        let result = compare_phase_space_measures_spectral_mixture(
            &candidate,
            &reference,
            &mixture_config(),
            SpectralMmdConfig {
                feature_count: 256,
                seed: 88,
            },
        )
        .unwrap();
        assert!(result.estimate.normalized_mmd <= 1.0e-12);
    }

    #[test]
    fn malformed_kernel_mixtures_are_rejected() {
        let mut invalid = mixture_config();
        invalid.components.clear();
        assert_eq!(
            compare_phase_space_measures_mmd_mixture(&[], &[], &invalid),
            Err(MmdError::InvalidMixtureConfig)
        );

        invalid = mixture_config();
        invalid.components[0].weight = 0.0;
        assert_eq!(
            compare_phase_space_measures_mmd_mixture(&[], &[], &invalid),
            Err(MmdError::InvalidMixtureConfig)
        );

        invalid = mixture_config();
        invalid.components[0].spatial_concentration = f64::NAN;
        assert_eq!(
            compare_phase_space_measures_mmd_mixture(&[], &[], &invalid),
            Err(MmdError::InvalidMixtureConfig)
        );

        invalid = mixture_config();
        invalid.components[2] = invalid.components[1];
        assert_eq!(
            compare_phase_space_measures_mmd_mixture(&[], &[], &invalid),
            Err(MmdError::InvalidMixtureConfig)
        );

        invalid = mixture_config();
        invalid.components[0].weight += 0.1;
        assert_eq!(
            compare_phase_space_measures_mmd_mixture(&[], &[], &invalid),
            Err(MmdError::InvalidMixtureConfig)
        );
    }


}
