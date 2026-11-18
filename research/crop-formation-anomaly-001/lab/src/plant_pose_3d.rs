//! Three-dimensional stem-axis orientation statistics for plant posture fields.
//!
//! Samples are already measured 3D positions and base-to-tip axis vectors.
//! Convert tip-to-base source vectors before calling this module, retaining the
//! original convention in the evidence record. This module does not reconstruct
//! stems from photographs or LiDAR point clouds.

use std::f64::consts::TAU;

const MIN_VECTOR_NORM: f64 = 1.0e-12;
const MIN_POSITION_DISTANCE_SQUARED: f64 = 1.0e-18;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector3 {
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    fn norm(self) -> f64 {
        self.x.hypot(self.y).hypot(self.z)
    }

    fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    fn normalized(self) -> Self {
        let norm = self.norm();
        Self::new(self.x / norm, self.y / norm, self.z / norm)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StemPoseSample {
    pub base_position: Point3,
    /// Directed base-to-tip stem axis in a declared field/world coordinate frame.
    /// The caller must normalize any tip-to-base source convention before analysis.
    pub axis: Vector3,
    /// Non-negative bend/deformation magnitude in a declared unit.
    pub bend_magnitude: f64,
    /// Non-negative measurement quality weight with a documented external basis.
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StemPose3DDescriptor {
    pub sample_count: usize,
    /// Weighted mean of normalized base-to-tip vectors.
    pub mean_axis_vector: Vector3,
    /// Norm of the mean unit vector. Zero means cancelling/opposed directions.
    pub axis_resultant_length: f64,
    /// Mean angle from +Z (vertical up) in radians.
    pub mean_inclination_from_vertical_rad: f64,
    pub inclination_std_dev_rad: f64,
    /// Circular mean of horizontal azimuth; undefined for near-vertical/cancelling axes.
    pub mean_azimuth_rad: Option<f64>,
    pub azimuth_resultant_length: f64,
    pub mean_bend_magnitude: f64,
    pub bend_magnitude_std_dev: f64,
    /// Mean dot product to the nearest distinct-position neighbour in 3D.
    pub nearest_neighbor_axis_alignment: Option<f64>,
    /// Optional centre-relative horizontal radial alignment (+ outward, - inward).
    pub radial_alignment: Option<f64>,
    /// Optional centre-relative horizontal tangential alignment (+ CCW, - CW).
    pub tangential_alignment: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StemPose3DError {
    EmptySamples,
    NonFinitePosition { index: usize },
    NonFiniteAxis { index: usize },
    DegenerateAxis { index: usize },
    NonFiniteBendMagnitude { index: usize },
    NegativeBendMagnitude { index: usize },
    NonFiniteWeight { index: usize },
    NegativeWeight { index: usize },
    ZeroTotalWeight,
    NonFiniteReferenceCenter,
}

impl std::fmt::Display for StemPose3DError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySamples => write!(f, "at least one 3D stem-pose sample is required"),
            Self::NonFinitePosition { index } => write!(f, "sample {index} has non-finite position"),
            Self::NonFiniteAxis { index } => write!(f, "sample {index} has a non-finite axis vector"),
            Self::DegenerateAxis { index } => write!(f, "sample {index} has a zero-length axis vector"),
            Self::NonFiniteBendMagnitude { index } => write!(f, "sample {index} has non-finite bend magnitude"),
            Self::NegativeBendMagnitude { index } => write!(f, "sample {index} has negative bend magnitude"),
            Self::NonFiniteWeight { index } => write!(f, "sample {index} has non-finite weight"),
            Self::NegativeWeight { index } => write!(f, "sample {index} has negative weight"),
            Self::ZeroTotalWeight => write!(f, "3D stem-pose samples have zero total weight"),
            Self::NonFiniteReferenceCenter => write!(f, "reference centre must be finite"),
        }
    }
}

impl std::error::Error for StemPose3DError {}

pub fn describe_stem_pose_3d(
    samples: &[StemPoseSample],
) -> Result<StemPose3DDescriptor, StemPose3DError> {
    describe_stem_pose_3d_internal(samples, None)
}

/// Describe stem-axis posture relative to a calibrated horizontal field centre.
///
/// The centre and all sample positions must use the same coordinate frame.
/// The radial/tangential metrics use each axis' horizontal projection and are
/// undefined for vertical stems or positions at the centre.
pub fn describe_stem_pose_3d_about_center(
    samples: &[StemPoseSample],
    center_x: f64,
    center_y: f64,
) -> Result<StemPose3DDescriptor, StemPose3DError> {
    if !center_x.is_finite() || !center_y.is_finite() {
        return Err(StemPose3DError::NonFiniteReferenceCenter);
    }
    describe_stem_pose_3d_internal(samples, Some((center_x, center_y)))
}

fn describe_stem_pose_3d_internal(
    samples: &[StemPoseSample],
    center: Option<(f64, f64)>,
) -> Result<StemPose3DDescriptor, StemPose3DError> {
    if samples.is_empty() {
        return Err(StemPose3DError::EmptySamples);
    }

    let mut max_weight = 0.0_f64;
    let mut normalized_axes = Vec::with_capacity(samples.len());
    let mut inclinations = Vec::with_capacity(samples.len());
    let mut azimuths = Vec::with_capacity(samples.len());

    for (index, sample) in samples.iter().enumerate() {
        if !sample.base_position.x.is_finite()
            || !sample.base_position.y.is_finite()
            || !sample.base_position.z.is_finite()
        {
            return Err(StemPose3DError::NonFinitePosition { index });
        }
        if !sample.axis.x.is_finite()
            || !sample.axis.y.is_finite()
            || !sample.axis.z.is_finite()
        {
            return Err(StemPose3DError::NonFiniteAxis { index });
        }
        let axis_norm = sample.axis.norm();
        if !axis_norm.is_finite() || axis_norm <= MIN_VECTOR_NORM {
            return Err(StemPose3DError::DegenerateAxis { index });
        }
        if !sample.bend_magnitude.is_finite() {
            return Err(StemPose3DError::NonFiniteBendMagnitude { index });
        }
        if sample.bend_magnitude < 0.0 {
            return Err(StemPose3DError::NegativeBendMagnitude { index });
        }
        if !sample.weight.is_finite() {
            return Err(StemPose3DError::NonFiniteWeight { index });
        }
        if sample.weight < 0.0 {
            return Err(StemPose3DError::NegativeWeight { index });
        }
        max_weight = max_weight.max(sample.weight);

        let unit_axis = sample.axis.normalized();
        normalized_axes.push(unit_axis);
        let horizontal_norm = unit_axis.x.hypot(unit_axis.y);
        inclinations.push(horizontal_norm.atan2(unit_axis.z));
        if horizontal_norm <= MIN_VECTOR_NORM {
            azimuths.push(None);
        } else {
            azimuths.push(Some(unit_axis.y.atan2(unit_axis.x).rem_euclid(TAU)));
        }
    }

    if max_weight <= 0.0 {
        return Err(StemPose3DError::ZeroTotalWeight);
    }

    let weights: Vec<f64> = samples
        .iter()
        .map(|sample| sample.weight / max_weight)
        .collect();
    let weight_sum: f64 = weights.iter().sum();

    let mut mean_axis = Vector3::new(0.0, 0.0, 0.0);
    let mut inclination_sum = 0.0;
    let mut bend_sum = 0.0;
    let mut azimuth_cos_sum = 0.0;
    let mut azimuth_sin_sum = 0.0;
    let mut azimuth_weight_sum = 0.0;

    for index in 0..samples.len() {
        let weight = weights[index];
        if weight == 0.0 {
            continue;
        }
        let axis = normalized_axes[index];
        mean_axis.x += weight * axis.x;
        mean_axis.y += weight * axis.y;
        mean_axis.z += weight * axis.z;
        inclination_sum += weight * inclinations[index];
        bend_sum += weight * samples[index].bend_magnitude;

        if let Some(azimuth) = azimuths[index] {
            azimuth_cos_sum += weight * azimuth.cos();
            azimuth_sin_sum += weight * azimuth.sin();
            azimuth_weight_sum += weight;
        }
    }

    mean_axis.x /= weight_sum;
    mean_axis.y /= weight_sum;
    mean_axis.z /= weight_sum;
    let axis_resultant_length = mean_axis.norm().clamp(0.0, 1.0);

    let mean_inclination = inclination_sum / weight_sum;
    let mean_bend_magnitude = bend_sum / weight_sum;
    let inclination_variance = samples
        .iter()
        .zip(weights.iter().copied())
        .enumerate()
        .map(|(index, (_, weight))| {
            let delta = inclinations[index] - mean_inclination;
            weight * delta * delta
        })
        .sum::<f64>()
        / weight_sum;

    let (mean_azimuth_rad, azimuth_resultant_length) =
        if azimuth_weight_sum <= 0.0 {
            (None, 0.0)
        } else {
            let mean_cos = azimuth_cos_sum / azimuth_weight_sum;
            let mean_sin = azimuth_sin_sum / azimuth_weight_sum;
            let resultant = mean_cos.hypot(mean_sin).clamp(0.0, 1.0);
            if resultant <= MIN_VECTOR_NORM {
                (None, resultant)
            } else {
                (Some(mean_sin.atan2(mean_cos).rem_euclid(TAU)), resultant)
            }
        };

    let bend_variance = samples
        .iter()
        .zip(weights.iter().copied())
        .map(|(sample, weight)| {
            let delta = sample.bend_magnitude - mean_bend_magnitude;
            weight * delta * delta
        })
        .sum::<f64>()
        / weight_sum;

    let nearest_neighbor_axis_alignment =
        nearest_neighbor_alignment_3d(samples, &normalized_axes, &weights);
    let (radial_alignment, tangential_alignment) = center
        .map(|(center_x, center_y)| {
            centre_relative_horizontal_alignment(samples, &normalized_axes, &weights, center_x, center_y)
        })
        .unwrap_or((None, None));

    Ok(StemPose3DDescriptor {
        sample_count: samples.len(),
        mean_axis_vector: mean_axis,
        axis_resultant_length,
        mean_inclination_from_vertical_rad: mean_inclination,
        inclination_std_dev_rad: inclination_variance.max(0.0).sqrt(),
        mean_azimuth_rad,
        azimuth_resultant_length,
        mean_bend_magnitude,
        bend_magnitude_std_dev: bend_variance.max(0.0).sqrt(),
        nearest_neighbor_axis_alignment,
        radial_alignment,
        tangential_alignment,
    })
}

fn nearest_neighbor_alignment_3d(
    samples: &[StemPoseSample],
    normalized_axes: &[Vector3],
    weights: &[f64],
) -> Option<f64> {
    if samples.len() < 2 {
        return None;
    }
    let mut alignment_sum = 0.0;
    let mut pair_weight_sum = 0.0;

    for index in 0..samples.len() {
        if weights[index] <= 0.0 {
            continue;
        }
        let position = samples[index].base_position;
        let mut nearest: Option<(usize, f64)> = None;
        for candidate_index in 0..samples.len() {
            if candidate_index == index || weights[candidate_index] <= 0.0 {
                continue;
            }
            let candidate = samples[candidate_index].base_position;
            let dx = position.x - candidate.x;
            let dy = position.y - candidate.y;
            let dz = position.z - candidate.z;
            let distance_squared = dx * dx + dy * dy + dz * dz;
            if distance_squared <= MIN_POSITION_DISTANCE_SQUARED {
                continue;
            }
            if nearest.is_none_or(|(_, best)| distance_squared < best) {
                nearest = Some((candidate_index, distance_squared));
            }
        }
        if let Some((nearest_index, _)) = nearest {
            let pair_weight = weights[index] * weights[nearest_index];
            alignment_sum += pair_weight * normalized_axes[index].dot(normalized_axes[nearest_index]);
            pair_weight_sum += pair_weight;
        }
    }

    if pair_weight_sum <= 0.0 {
        None
    } else {
        Some((alignment_sum / pair_weight_sum).clamp(-1.0, 1.0))
    }
}

fn centre_relative_horizontal_alignment(
    samples: &[StemPoseSample],
    axes: &[Vector3],
    weights: &[f64],
    center_x: f64,
    center_y: f64,
) -> (Option<f64>, Option<f64>) {
    let mut radial_sum = 0.0;
    let mut tangential_sum = 0.0;
    let mut total_weight = 0.0;

    for index in 0..samples.len() {
        if weights[index] <= 0.0 {
            continue;
        }
        let dx = samples[index].base_position.x - center_x;
        let dy = samples[index].base_position.y - center_y;
        let horizontal_position_norm = dx.hypot(dy);
        let horizontal_axis_norm = axes[index].x.hypot(axes[index].y);
        if horizontal_position_norm <= MIN_VECTOR_NORM
            || horizontal_axis_norm <= MIN_VECTOR_NORM
        {
            continue;
        }
        let radial_direction = dy.atan2(dx);
        let axis_direction = axes[index].y.atan2(axes[index].x);
        let relative = axis_direction - radial_direction;
        radial_sum += weights[index] * relative.cos();
        tangential_sum += weights[index] * relative.sin();
        total_weight += weights[index];
    }

    if total_weight <= 0.0 {
        (None, None)
    } else {
        (
            Some((radial_sum / total_weight).clamp(-1.0, 1.0)),
            Some((tangential_sum / total_weight).clamp(-1.0, 1.0)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertical_stems_have_zero_inclination_and_no_azimuth() {
        let samples = [
            StemPoseSample {
                base_position: Point3::new(0.0, 0.0, 0.0),
                axis: Vector3::new(0.0, 0.0, 2.0),
                bend_magnitude: 0.0,
                weight: 1.0,
            },
            StemPoseSample {
                base_position: Point3::new(1.0, 0.0, 0.0),
                axis: Vector3::new(0.0, 0.0, 1.0),
                bend_magnitude: 0.0,
                weight: 1.0,
            },
        ];
        let result = describe_stem_pose_3d(&samples).expect("vertical field");
        assert!(result.mean_inclination_from_vertical_rad.abs() < 1.0e-12);
        assert_eq!(result.mean_azimuth_rad, None);
        assert!((result.axis_resultant_length - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn horizontal_eastward_stems_have_quarter_turn_inclination_and_zero_azimuth() {
        let samples = [
            StemPoseSample {
                base_position: Point3::new(0.0, 0.0, 0.0),
                axis: Vector3::new(1.0, 0.0, 0.0),
                bend_magnitude: 0.5,
                weight: 1.0,
            },
            StemPoseSample {
                base_position: Point3::new(1.0, 0.0, 0.0),
                axis: Vector3::new(2.0, 0.0, 0.0),
                bend_magnitude: 0.5,
                weight: 1.0,
            },
        ];
        let result = describe_stem_pose_3d(&samples).expect("eastward field");
        assert!((result.mean_inclination_from_vertical_rad - std::f64::consts::FRAC_PI_2).abs() < 1.0e-12);
        assert!(result.mean_azimuth_rad.expect("defined azimuth").abs() < 1.0e-12);
        assert!((result.nearest_neighbor_axis_alignment.expect("neighbours") - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn opposing_stem_axes_have_zero_global_resultant() {
        let samples = [
            StemPoseSample { base_position: Point3::new(0.0, 0.0, 0.0), axis: Vector3::new(1.0, 0.0, 0.0), bend_magnitude: 1.0, weight: 1.0 },
            StemPoseSample { base_position: Point3::new(1.0, 0.0, 0.0), axis: Vector3::new(-1.0, 0.0, 0.0), bend_magnitude: 1.0, weight: 1.0 },
        ];
        let result = describe_stem_pose_3d(&samples).expect("opposing field");
        assert!(result.axis_resultant_length < 1.0e-12);
        assert!(result.mean_azimuth_rad.is_none());
        assert!((result.nearest_neighbor_axis_alignment.expect("neighbours") + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn centre_relative_alignment_distinguishes_radial_and_tangential_pose() {
        let radial = [
            StemPoseSample { base_position: Point3::new(1.0, 0.0, 0.0), axis: Vector3::new(1.0, 0.0, 1.0), bend_magnitude: 1.0, weight: 1.0 },
            StemPoseSample { base_position: Point3::new(0.0, 1.0, 0.0), axis: Vector3::new(0.0, 1.0, 1.0), bend_magnitude: 1.0, weight: 1.0 },
        ];
        let radial_result = describe_stem_pose_3d_about_center(&radial, 0.0, 0.0)
            .expect("radial pose field");
        assert!((radial_result.radial_alignment.expect("radial score") - 1.0).abs() < 1.0e-12);
        assert!(radial_result.tangential_alignment.expect("tangential score").abs() < 1.0e-12);

        let tangential = [
            StemPoseSample { base_position: Point3::new(1.0, 0.0, 0.0), axis: Vector3::new(0.0, 1.0, 1.0), bend_magnitude: 1.0, weight: 1.0 },
            StemPoseSample { base_position: Point3::new(0.0, 1.0, 0.0), axis: Vector3::new(-1.0, 0.0, 1.0), bend_magnitude: 1.0, weight: 1.0 },
        ];
        let tangential_result = describe_stem_pose_3d_about_center(&tangential, 0.0, 0.0)
            .expect("tangential pose field");
        assert!(tangential_result.radial_alignment.expect("radial score").abs() < 1.0e-12);
        assert!((tangential_result.tangential_alignment.expect("tangential score") - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn non_uniform_weights_change_the_mean_pose() {
        let samples = [
            StemPoseSample { base_position: Point3::new(0.0, 0.0, 0.0), axis: Vector3::new(0.0, 0.0, 1.0), bend_magnitude: 0.0, weight: 10.0 },
            StemPoseSample { base_position: Point3::new(1.0, 0.0, 0.0), axis: Vector3::new(1.0, 0.0, 0.0), bend_magnitude: 2.0, weight: 1.0 },
        ];
        let result = describe_stem_pose_3d(&samples).expect("weighted pose field");
        assert!(result.mean_inclination_from_vertical_rad < 0.2);
        assert!(result.mean_bend_magnitude < 0.2);
    }

    #[test]
    fn malformed_3d_pose_samples_fail_closed() {
        let valid = StemPoseSample {
            base_position: Point3::new(0.0, 0.0, 0.0),
            axis: Vector3::new(0.0, 0.0, 1.0),
            bend_magnitude: 1.0,
            weight: 1.0,
        };
        assert_eq!(describe_stem_pose_3d(&[]), Err(StemPose3DError::EmptySamples));
        assert_eq!(
            describe_stem_pose_3d(&[StemPoseSample { axis: Vector3::new(0.0, 0.0, 0.0), ..valid }]),
            Err(StemPose3DError::DegenerateAxis { index: 0 })
        );
        assert_eq!(
            describe_stem_pose_3d_about_center(&[valid], f64::NAN, 0.0),
            Err(StemPose3DError::NonFiniteReferenceCenter)
        );
        assert_eq!(
            describe_stem_pose_3d(&[StemPoseSample { weight: -1.0, ..valid }]),
            Err(StemPose3DError::NegativeWeight { index: 0 })
        );
    }
}
