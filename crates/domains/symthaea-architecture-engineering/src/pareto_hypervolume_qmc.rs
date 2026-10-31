//! Randomized quasi-Monte-Carlo hypervolume estimation for multi-objective NAS.
//!
//! This module is the scalable numerical counterpart to the exact 2-D reference oracle.
//! It estimates dominated hypervolume for 2..=16 normalized maximize-is-better objectives
//! using independently Cranley-Patterson-shifted Halton replicates. Each estimate carries
//! a between-scramble standard error, normal-approximation interval, exact point-set digest
//! and exact sampling-design digest.
//!
//! The uncertainty interval is an empirical randomized-QMC error estimate, not a rigorous
//! finite-sample confidence guarantee. Reproducible evaluation must freeze dimension, reference,
//! sample count, scramble count and seed. High-dimensional discontinuous dominance indicators
//! can converge slowly; callers should compare against exact low-dimensional fixtures and run
//! convergence studies before using the estimate to drive consequential pruning.

use std::error::Error;
use std::fmt;

use sha2::{Digest, Sha256};

const MIN_DIMENSIONS: usize = 2;
const MAX_DIMENSIONS: usize = 16;
const MIN_SAMPLES_PER_REPLICATE: usize = 128;
const MAX_SAMPLES_PER_REPLICATE: usize = 1_000_000;
const MIN_SCRAMBLE_REPLICATES: usize = 8;
const MAX_SCRAMBLE_REPLICATES: usize = 256;
const PRIMES: [u64; MAX_DIMENSIONS] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
];
const DESIGN_DOMAIN: &[u8] = b"symthaea.nas.pareto-hv-rqmc-design.v0.1";
const POINT_SET_DOMAIN: &[u8] = b"symthaea.nas.pareto-hv-point-set.v0.1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HypervolumeQmcConfig {
    pub samples_per_replicate: usize,
    pub scramble_replicates: usize,
    pub seed: u64,
}

impl Default for HypervolumeQmcConfig {
    fn default() -> Self {
        Self {
            samples_per_replicate: 4096,
            scramble_replicates: 16,
            seed: 0x6a09_e667_f3bc_c909,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct HypervolumeQmcEstimate {
    pub estimate: f64,
    pub standard_error: f64,
    /// Normal-approximation 95% interval across independent random shifts, clipped to
    /// the maximum possible hypervolume box.
    pub confidence_interval_95: (f64, f64),
    pub box_volume: f64,
    pub dimension: usize,
    pub sample_count_per_replicate: usize,
    pub scramble_replicates: usize,
    pub hit_counts_per_replicate: Vec<usize>,
    pub point_set_sha256: String,
    pub sampling_design_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HypervolumeQmcError {
    pub field: String,
    pub message: String,
}

impl HypervolumeQmcError {
    fn new(field: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for HypervolumeQmcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl Error for HypervolumeQmcError {}

/// Estimate the union volume dominated by points in the box [reference, 1]^d.
///
/// The function supports 2..=16 objectives and arbitrary candidate counts. All objective
/// coordinates must already have been normalized to [0,1], with larger always better.
/// A sample is dominated if at least one candidate is no worse in every objective.
/// Points that fail to dominate the reference in any coordinate contribute zero volume.
///
/// QMC sample locations are generated from Halton sequences with independent random shifts
/// per replicate. The same config gives common random numbers for comparing nested point sets.
pub fn estimate_hypervolume_qmc(
    points: &[Vec<f64>],
    reference: &[f64],
    config: &HypervolumeQmcConfig,
) -> Result<HypervolumeQmcEstimate, HypervolumeQmcError> {
    validate_config(config)?;
    let dimension = reference.len();
    if !(MIN_DIMENSIONS..=MAX_DIMENSIONS).contains(&dimension) {
        return Err(HypervolumeQmcError::new(
            "reference",
            format!("dimension must be in {MIN_DIMENSIONS}..={MAX_DIMENSIONS}"),
        ));
    }
    for (index, value) in reference.iter().copied().enumerate() {
        if !value.is_finite() || !(0.0..1.0).contains(&value) {
            return Err(HypervolumeQmcError::new(
                format!("reference[{index}]"),
                "reference coordinates must be finite and in [0,1)",
            ));
        }
    }

    let mut eligible = Vec::with_capacity(points.len());
    for (point_index, point) in points.iter().enumerate() {
        if point.len() != dimension {
            return Err(HypervolumeQmcError::new(
                format!("points[{point_index}]"),
                "every point must have exactly the reference dimension",
            ));
        }
        for (coordinate, value) in point.iter().copied().enumerate() {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(HypervolumeQmcError::new(
                    format!("points[{point_index}][{coordinate}]"),
                    "normalized objective values must be finite and in [0,1]",
                ));
            }
        }
        if point
            .iter()
            .zip(reference)
            .all(|(value, lower)| value > lower)
        {
            eligible.push(point.clone());
        }
    }

    let point_set_sha256 = compute_point_set_sha256(points, reference);
    let sampling_design_sha256 = compute_sampling_design_sha256(reference, config);
    let box_volume = reference
        .iter()
        .map(|lower| 1.0 - lower)
        .product::<f64>();

    if eligible.is_empty() {
        return Ok(HypervolumeQmcEstimate {
            estimate: 0.0,
            standard_error: 0.0,
            confidence_interval_95: (0.0, 0.0),
            box_volume,
            dimension,
            sample_count_per_replicate: config.samples_per_replicate,
            scramble_replicates: config.scramble_replicates,
            hit_counts_per_replicate: vec![0; config.scramble_replicates],
            point_set_sha256,
            sampling_design_sha256,
        });
    }

    let mut replicate_estimates = Vec::with_capacity(config.scramble_replicates);
    let mut hit_counts_per_replicate = Vec::with_capacity(config.scramble_replicates);
    for replicate in 0..config.scramble_replicates {
        let mut rng = SplitMix64::new(
            config.seed
                ^ (replicate as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
                ^ 0xbb67_ae85_84ca_a73b,
        );
        let shifts: Vec<f64> = (0..dimension).map(|_| rng.next_f64()).collect();
        let mut hits = 0usize;

        for sample_index in 0..config.samples_per_replicate {
            let sequence_index = sample_index as u64 + 1;
            let mut dominated = false;
            let mut sample_coordinates = [0.0f64; MAX_DIMENSIONS];
            for coordinate in 0..dimension {
                let base_value = radical_inverse(sequence_index, PRIMES[coordinate]);
                let shifted = base_value + shifts[coordinate];
                let unit_value = shifted - shifted.floor();
                sample_coordinates[coordinate] =
                    reference[coordinate] + unit_value * (1.0 - reference[coordinate]);
            }

            for point in &eligible {
                if (0..dimension)
                    .all(|coordinate| point[coordinate] >= sample_coordinates[coordinate])
                {
                    dominated = true;
                    break;
                }
            }
            if dominated {
                hits += 1;
            }
        }

        hit_counts_per_replicate.push(hits);
        replicate_estimates.push(box_volume * hits as f64 / config.samples_per_replicate as f64);
    }

    let estimate =
        replicate_estimates.iter().sum::<f64>() / replicate_estimates.len() as f64;
    let standard_error = sample_standard_deviation(&replicate_estimates)
        / (replicate_estimates.len() as f64).sqrt();
    let margin = 1.96 * standard_error;
    let confidence_interval_95 = (
        (estimate - margin).max(0.0),
        (estimate + margin).min(box_volume),
    );

    Ok(HypervolumeQmcEstimate {
        estimate,
        standard_error,
        confidence_interval_95,
        box_volume,
        dimension,
        sample_count_per_replicate: config.samples_per_replicate,
        scramble_replicates: config.scramble_replicates,
        hit_counts_per_replicate,
        point_set_sha256,
        sampling_design_sha256,
    })
}

fn validate_config(config: &HypervolumeQmcConfig) -> Result<(), HypervolumeQmcError> {
    if !(MIN_SAMPLES_PER_REPLICATE..=MAX_SAMPLES_PER_REPLICATE)
        .contains(&config.samples_per_replicate)
    {
        return Err(HypervolumeQmcError::new(
            "samples_per_replicate",
            format!(
                "must be in {MIN_SAMPLES_PER_REPLICATE}..={MAX_SAMPLES_PER_REPLICATE}"
            ),
        ));
    }
    if !(MIN_SCRAMBLE_REPLICATES..=MAX_SCRAMBLE_REPLICATES)
        .contains(&config.scramble_replicates)
    {
        return Err(HypervolumeQmcError::new(
            "scramble_replicates",
            format!(
                "must be in {MIN_SCRAMBLE_REPLICATES}..={MAX_SCRAMBLE_REPLICATES}"
            ),
        ));
    }
    Ok(())
}

fn radical_inverse(mut index: u64, base: u64) -> f64 {
    let mut factor = 1.0 / base as f64;
    let mut value = 0.0;
    while index > 0 {
        let digit = index % base;
        value += digit as f64 * factor;
        index /= base;
        factor /= base as f64;
    }
    value
}

fn sample_standard_deviation(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let sum_squared = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>();
    (sum_squared / (values.len() - 1) as f64).sqrt()
}

fn compute_point_set_sha256(points: &[Vec<f64>], reference: &[f64]) -> String {
    let mut canonical_points = points.to_vec();
    canonical_points.sort_by(|left, right| {
        for (left_value, right_value) in left.iter().zip(right) {
            let ordering = left_value.total_cmp(right_value);
            if ordering != std::cmp::Ordering::Equal {
                return ordering;
            }
        }
        left.len().cmp(&right.len())
    });

    let mut hasher = Sha256::new();
    hasher.update((POINT_SET_DOMAIN.len() as u64).to_be_bytes());
    hasher.update(POINT_SET_DOMAIN);
    hasher.update((reference.len() as u64).to_be_bytes());
    for value in reference {
        hasher.update(value.to_bits().to_be_bytes());
    }
    hasher.update((canonical_points.len() as u64).to_be_bytes());
    for point in canonical_points {
        hasher.update((point.len() as u64).to_be_bytes());
        for value in point {
            hasher.update(value.to_bits().to_be_bytes());
        }
    }
    encode_sha256(hasher.finalize().as_slice())
}

fn compute_sampling_design_sha256(
    reference: &[f64],
    config: &HypervolumeQmcConfig,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update((DESIGN_DOMAIN.len() as u64).to_be_bytes());
    hasher.update(DESIGN_DOMAIN);
    hasher.update(config.seed.to_be_bytes());
    hasher.update((config.samples_per_replicate as u64).to_be_bytes());
    hasher.update((config.scramble_replicates as u64).to_be_bytes());
    hasher.update((reference.len() as u64).to_be_bytes());
    for value in reference {
        hasher.update(value.to_bits().to_be_bytes());
    }
    encode_sha256(hasher.finalize().as_slice())
}

fn encode_sha256(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut encoded, "{byte:02x}");
    }
    encoded
}

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

    fn next_f64(&mut self) -> f64 {
        let mantissa = self.next_u64() >> 11;
        mantissa as f64 * (1.0 / ((1u64 << 53) as f64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pareto_regret_voc::hypervolume_2d;

    fn config() -> HypervolumeQmcConfig {
        HypervolumeQmcConfig {
            samples_per_replicate: 4096,
            scramble_replicates: 16,
            seed: 42,
        }
    }

    #[test]
    fn estimates_three_dimensional_single_box_volume() {
        let points = vec![vec![0.5, 0.4, 0.3]];
        let estimate = estimate_hypervolume_qmc(&points, &[0.0, 0.0, 0.0], &config()).unwrap();
        assert!((estimate.estimate - 0.06).abs() < 0.001);
        assert!(estimate.standard_error >= 0.0);
        assert!(estimate.confidence_interval_95.0 <= estimate.estimate);
        assert!(estimate.confidence_interval_95.1 >= estimate.estimate);
    }

    #[test]
    fn randomized_qmc_matches_exact_two_dimensional_oracle() {
        let points = vec![vec![0.8, 0.3], vec![0.4, 0.7]];
        let exact = hypervolume_2d(&[[0.8, 0.3], [0.4, 0.7]], [0.0, 0.0]).unwrap();
        let estimate = estimate_hypervolume_qmc(&points, &[0.0, 0.0], &config()).unwrap();
        assert!((exact - 0.4).abs() < 1e-12);
        assert!((estimate.estimate - exact).abs() < 0.005);
    }

    #[test]
    fn point_set_order_does_not_change_estimate_or_digest() {
        let points_a = vec![vec![0.8, 0.3, 0.6], vec![0.4, 0.7, 0.5]];
        let points_b = vec![vec![0.4, 0.7, 0.5], vec![0.8, 0.3, 0.6]];
        let first = estimate_hypervolume_qmc(&points_a, &[0.0, 0.0, 0.0], &config()).unwrap();
        let second = estimate_hypervolume_qmc(&points_b, &[0.0, 0.0, 0.0], &config()).unwrap();
        assert_eq!(first.estimate, second.estimate);
        assert_eq!(first.point_set_sha256, second.point_set_sha256);
        assert_eq!(first.sampling_design_sha256, second.sampling_design_sha256);
    }

    #[test]
    fn adding_a_candidate_cannot_reduce_a_common_design_estimate() {
        let base = vec![vec![0.8, 0.3, 0.6]];
        let extended = vec![vec![0.8, 0.3, 0.6], vec![0.4, 0.7, 0.5]];
        let base_estimate =
            estimate_hypervolume_qmc(&base, &[0.0, 0.0, 0.0], &config()).unwrap();
        let extended_estimate =
            estimate_hypervolume_qmc(&extended, &[0.0, 0.0, 0.0], &config()).unwrap();
        assert!(extended_estimate.estimate + 1e-12 >= base_estimate.estimate);
    }

    #[test]
    fn duplicate_points_do_not_change_qmc_estimate() {
        let point = vec![vec![0.8, 0.3, 0.6], vec![0.4, 0.7, 0.5]];
        let duplicate = vec![
            vec![0.8, 0.3, 0.6],
            vec![0.4, 0.7, 0.5],
            vec![0.8, 0.3, 0.6],
        ];
        let first = estimate_hypervolume_qmc(&point, &[0.0, 0.0, 0.0], &config()).unwrap();
        let second = estimate_hypervolume_qmc(&duplicate, &[0.0, 0.0, 0.0], &config()).unwrap();
        assert_eq!(first.estimate, second.estimate);
    }

    #[test]
    fn points_below_reference_contribute_zero_volume() {
        let points = vec![vec![0.8, 0.2, 0.7], vec![0.2, 0.9, 0.3]];
        let estimate = estimate_hypervolume_qmc(&points, &[0.3, 0.3, 0.3], &config()).unwrap();
        assert!((estimate.estimate - 0.0).abs() < 1e-12);
        assert_eq!(estimate.standard_error, 0.0);
    }

    #[test]
    fn estimates_are_deterministic_for_fixed_sampling_design() {
        let points = vec![vec![0.9, 0.5, 0.8], vec![0.4, 0.9, 0.4]];
        let first = estimate_hypervolume_qmc(&points, &[0.1, 0.1, 0.1], &config()).unwrap();
        let second = estimate_hypervolume_qmc(&points, &[0.1, 0.1, 0.1], &config()).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn seed_changes_sampling_design_receipt() {
        let points = vec![vec![0.9, 0.5, 0.8]];
        let mut second_config = config();
        second_config.seed += 1;
        let first = estimate_hypervolume_qmc(&points, &[0.0, 0.0, 0.0], &config()).unwrap();
        let second =
            estimate_hypervolume_qmc(&points, &[0.0, 0.0, 0.0], &second_config).unwrap();
        assert_ne!(
            first.sampling_design_sha256,
            second.sampling_design_sha256
        );
    }

    #[test]
    fn refuses_invalid_dimension_sample_count_and_reference() {
        assert!(estimate_hypervolume_qmc(&[vec![0.5]], &[0.0], &config()).is_err());
        let mut invalid = config();
        invalid.samples_per_replicate = 16;
        assert!(estimate_hypervolume_qmc(&[vec![0.5, 0.5]], &[0.0, 0.0], &invalid).is_err());
        assert!(estimate_hypervolume_qmc(
            &[vec![0.5, 0.5]],
            &[0.0, 1.0],
            &config()
        )
        .is_err());
    }

    #[test]
    fn rejects_non_finite_or_wrong_dimension_points() {
        assert!(estimate_hypervolume_qmc(
            &[vec![0.5, 0.5], vec![0.5]],
            &[0.0, 0.0],
            &config()
        )
        .is_err());
        assert!(estimate_hypervolume_qmc(
            &[vec![0.5, f64::NAN]],
            &[0.0, 0.0],
            &config()
        )
        .is_err());
    }
}
