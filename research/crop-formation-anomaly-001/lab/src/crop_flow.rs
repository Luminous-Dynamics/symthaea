//! Circular and spatial descriptors for measured crop-flow directions.
//!
//! Samples must come from a declared measurement protocol. These statistics
//! describe observed orientation fields; they do not identify a cause.

use crate::geometry::Point2;
use std::f64::consts::TAU;

const MIN_RESULTANT: f64 = 1.0e-12;
const MIN_DISTANCE_SQUARED: f64 = 1.0e-18;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CropFlowSample {
    pub position: Point2,
    /// Direction of the observed stem/crop flow, measured in radians.
    pub direction_rad: f64,
    /// Non-negative bend/deformation magnitude in a declared external unit.
    pub bend_magnitude: f64,
    /// Non-negative analysis weight (for example, measurement quality weight).
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CropFlowDescriptor {
    pub sample_count: usize,
    /// None when the weighted circular mean is undefined or nearly isotropic.
    pub mean_direction_rad: Option<f64>,
    /// Circular resultant length in [0, 1]. This is concentration, not certainty.
    pub resultant_length: f64,
    pub mean_bend_magnitude: f64,
    pub bend_magnitude_std_dev: f64,
    /// Weighted proportions per equal-width direction bin; sums to one.
    pub direction_histogram: Vec<f64>,
    /// Entropy divided by log2(bin_count); None for a one-bin histogram.
    pub normalized_direction_entropy: Option<f64>,
    /// Mean cosine alignment with each sample's nearest distinct-position neighbour.
    /// +1 means locally aligned, -1 means locally opposed.
    pub nearest_neighbor_alignment: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CropFlowError {
    EmptySamples,
    EmptyHistogram,
    NonFinitePosition { index: usize },
    NonFiniteDirection { index: usize },
    NonFiniteBendMagnitude { index: usize },
    NegativeBendMagnitude { index: usize },
    NonFiniteWeight { index: usize },
    NegativeWeight { index: usize },
    ZeroTotalWeight,
}

impl std::fmt::Display for CropFlowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySamples => write!(f, "at least one crop-flow sample is required"),
            Self::EmptyHistogram => write!(f, "direction histogram bin count must be non-zero"),
            Self::NonFinitePosition { index } => {
                write!(f, "sample {index} has a non-finite position")
            }
            Self::NonFiniteDirection { index } => {
                write!(f, "sample {index} has a non-finite flow direction")
            }
            Self::NonFiniteBendMagnitude { index } => {
                write!(f, "sample {index} has a non-finite bend magnitude")
            }
            Self::NegativeBendMagnitude { index } => {
                write!(f, "sample {index} has a negative bend magnitude")
            }
            Self::NonFiniteWeight { index } => {
                write!(f, "sample {index} has a non-finite weight")
            }
            Self::NegativeWeight { index } => {
                write!(f, "sample {index} has a negative weight")
            }
            Self::ZeroTotalWeight => write!(f, "crop-flow samples have zero total weight"),
        }
    }
}

impl std::error::Error for CropFlowError {}

/// Describe direction and deformation measurements in a calibrated 2D field.
///
/// Weights are scaled by their maximum before summation to reduce avoidable
/// overflow. A sample with zero weight is validated but contributes no mass.
pub fn describe_crop_flow(
    samples: &[CropFlowSample],
    angular_bins: usize,
) -> Result<CropFlowDescriptor, CropFlowError> {
    if samples.is_empty() {
        return Err(CropFlowError::EmptySamples);
    }
    if angular_bins == 0 {
        return Err(CropFlowError::EmptyHistogram);
    }

    let mut max_weight = 0.0_f64;
    for (index, sample) in samples.iter().enumerate() {
        if !sample.position.x.is_finite() || !sample.position.y.is_finite() {
            return Err(CropFlowError::NonFinitePosition { index });
        }
        if !sample.direction_rad.is_finite() {
            return Err(CropFlowError::NonFiniteDirection { index });
        }
        if !sample.bend_magnitude.is_finite() {
            return Err(CropFlowError::NonFiniteBendMagnitude { index });
        }
        if sample.bend_magnitude < 0.0 {
            return Err(CropFlowError::NegativeBendMagnitude { index });
        }
        if !sample.weight.is_finite() {
            return Err(CropFlowError::NonFiniteWeight { index });
        }
        if sample.weight < 0.0 {
            return Err(CropFlowError::NegativeWeight { index });
        }
        max_weight = max_weight.max(sample.weight);
    }

    if max_weight <= 0.0 {
        return Err(CropFlowError::ZeroTotalWeight);
    }

    let normalized_weights: Vec<f64> = samples
        .iter()
        .map(|sample| sample.weight / max_weight)
        .collect();
    let weight_sum: f64 = normalized_weights.iter().sum();

    let mut cosine_sum = 0.0;
    let mut sine_sum = 0.0;
    let mut bend_sum = 0.0;
    let mut direction_histogram = vec![0.0; angular_bins];

    for (sample, normalized_weight) in samples.iter().zip(normalized_weights.iter().copied()) {
        if normalized_weight == 0.0 {
            continue;
        }
        let direction = sample.direction_rad.rem_euclid(TAU);
        cosine_sum += normalized_weight * direction.cos();
        sine_sum += normalized_weight * direction.sin();
        bend_sum += normalized_weight * sample.bend_magnitude;

        let bin_position = (direction / TAU * angular_bins as f64).floor();
        let bin_index = (bin_position as usize).min(angular_bins - 1);
        direction_histogram[bin_index] += normalized_weight;
    }

    let mean_cosine = cosine_sum / weight_sum;
    let mean_sine = sine_sum / weight_sum;
    let resultant_length = (mean_cosine * mean_cosine + mean_sine * mean_sine)
        .sqrt()
        .clamp(0.0, 1.0);

    let mean_direction_rad = if resultant_length <= MIN_RESULTANT {
        None
    } else {
        Some(mean_sine.atan2(mean_cosine).rem_euclid(TAU))
    };

    let mean_bend_magnitude = bend_sum / weight_sum;
    let weighted_variance = samples
        .iter()
        .zip(normalized_weights.iter().copied())
        .map(|(sample, weight)| {
            let difference = sample.bend_magnitude - mean_bend_magnitude;
            weight * difference * difference
        })
        .sum::<f64>()
        / weight_sum;

    for value in &mut direction_histogram {
        *value /= weight_sum;
    }

    let normalized_direction_entropy = if angular_bins <= 1 {
        None
    } else {
        let entropy = direction_histogram
            .iter()
            .filter(|probability| **probability > 0.0)
            .map(|probability| -probability * probability.log2())
            .sum::<f64>();
        Some((entropy / (angular_bins as f64).log2()).clamp(0.0, 1.0))
    };

    let nearest_neighbor_alignment = nearest_neighbor_alignment(samples, &normalized_weights);

    Ok(CropFlowDescriptor {
        sample_count: samples.len(),
        mean_direction_rad,
        resultant_length,
        mean_bend_magnitude,
        bend_magnitude_std_dev: weighted_variance.max(0.0).sqrt(),
        direction_histogram,
        normalized_direction_entropy,
        nearest_neighbor_alignment,
    })
}

fn nearest_neighbor_alignment(
    samples: &[CropFlowSample],
    normalized_weights: &[f64],
) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }

    let mut weighted_alignment = 0.0;
    let mut total_pair_weight = 0.0;

    for (index, sample) in samples.iter().enumerate() {
        let sample_weight = normalized_weights[index];
        if sample_weight == 0.0 {
            continue;
        }

        let mut nearest: Option<(usize, f64)> = None;
        for (candidate_index, candidate) in samples.iter().enumerate() {
            if candidate_index == index || normalized_weights[candidate_index] == 0.0 {
                continue;
            }
            let dx = sample.position.x - candidate.position.x;
            let dy = sample.position.y - candidate.position.y;
            let distance_squared = dx * dx + dy * dy;
            if distance_squared <= MIN_DISTANCE_SQUARED {
                continue;
            }
            if nearest.is_none_or(|(_, best_distance)| distance_squared < best_distance) {
                nearest = Some((candidate_index, distance_squared));
            }
        }

        if let Some((nearest_index, _)) = nearest {
            let pair_weight = sample_weight * normalized_weights[nearest_index];
            let alignment = (sample.direction_rad - samples[nearest_index].direction_rad).cos();
            weighted_alignment += pair_weight * alignment;
            total_pair_weight += pair_weight;
        }
    }

    if total_pair_weight <= 0.0 {
        None
    } else {
        Some((weighted_alignment / total_pair_weight).clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_samples_have_unit_resultant_and_local_alignment() {
        let samples = [
            CropFlowSample {
                position: Point2::new(0.0, 0.0),
                direction_rad: 1.0,
                bend_magnitude: 0.5,
                weight: 1.0,
            },
            CropFlowSample {
                position: Point2::new(1.0, 0.0),
                direction_rad: 1.0,
                bend_magnitude: 0.5,
                weight: 1.0,
            },
            CropFlowSample {
                position: Point2::new(2.0, 0.0),
                direction_rad: 1.0,
                bend_magnitude: 0.5,
                weight: 1.0,
            },
        ];
        let result = describe_crop_flow(&samples, 8).expect("valid flow field");

        assert!((result.resultant_length - 1.0).abs() < 1.0e-12);
        assert!((result.mean_direction_rad.expect("defined mean") - 1.0).abs() < 1.0e-12);
        assert!((result.nearest_neighbor_alignment.expect("neighbour") - 1.0).abs() < 1.0e-12);
        assert!((result.mean_bend_magnitude - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn opposite_directions_have_no_mean_direction() {
        let samples = [
            CropFlowSample {
                position: Point2::new(0.0, 0.0),
                direction_rad: 0.0,
                bend_magnitude: 1.0,
                weight: 1.0,
            },
            CropFlowSample {
                position: Point2::new(1.0, 0.0),
                direction_rad: std::f64::consts::PI,
                bend_magnitude: 1.0,
                weight: 1.0,
            },
        ];
        let result = describe_crop_flow(&samples, 8).expect("valid flow field");

        assert!(result.mean_direction_rad.is_none());
        assert!(result.resultant_length < 1.0e-12);
        assert!((result.nearest_neighbor_alignment.expect("neighbour") + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn circular_mean_handles_angles_across_zero() {
        let samples = [
            CropFlowSample {
                position: Point2::new(0.0, 0.0),
                direction_rad: TAU - 0.01,
                bend_magnitude: 0.2,
                weight: 1.0,
            },
            CropFlowSample {
                position: Point2::new(1.0, 0.0),
                direction_rad: 0.01,
                bend_magnitude: 0.2,
                weight: 1.0,
            },
        ];
        let result = describe_crop_flow(&samples, 12).expect("valid flow field");
        let mean = result.mean_direction_rad.expect("defined circular mean");

        assert!(mean < 0.02 || mean > TAU - 0.02);
        assert!(result.resultant_length > 0.99);
    }

    #[test]
    fn uniform_direction_bins_have_maximum_normalized_entropy() {
        let angles = [
            0.0,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::PI,
            3.0 * std::f64::consts::FRAC_PI_2,
        ];
        let samples: Vec<CropFlowSample> = angles
            .iter()
            .enumerate()
            .map(|(index, direction_rad)| CropFlowSample {
                position: Point2::new(index as f64, 0.0),
                direction_rad: *direction_rad,
                bend_magnitude: 1.0,
                weight: 1.0,
            })
            .collect();
        let result = describe_crop_flow(&samples, 4).expect("valid flow field");

        assert!((result.normalized_direction_entropy.expect("four bins") - 1.0).abs() < 1.0e-12);
        assert!(result.mean_direction_rad.is_none());
    }

    #[test]
    fn invalid_flow_samples_fail_closed() {
        let valid = CropFlowSample {
            position: Point2::new(0.0, 0.0),
            direction_rad: 0.0,
            bend_magnitude: 1.0,
            weight: 1.0,
        };
        assert_eq!(describe_crop_flow(&[], 8), Err(CropFlowError::EmptySamples));
        assert_eq!(
            describe_crop_flow(&[valid], 0),
            Err(CropFlowError::EmptyHistogram)
        );
        let negative_weight = CropFlowSample {
            weight: -1.0,
            ..valid
        };
        assert_eq!(
            describe_crop_flow(&[negative_weight], 8),
            Err(CropFlowError::NegativeWeight { index: 0 })
        );
    }
}
