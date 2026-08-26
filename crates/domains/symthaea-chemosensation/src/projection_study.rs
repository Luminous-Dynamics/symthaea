// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dataset-level assessment of the lossy ContinuousHV -> BinaryHV root boundary.
//!
//! A single near/far regression can catch a gross projection failure, but it is
//! too weak to justify a canonical representation boundary. This module measures
//! projection behavior over a comparable set of chemical bridge inputs without
//! declaring an acceptance threshold. Thresholds belong in a preregistered study
//! once calibration and held-out physical data exist.
//!
//! All geometric metrics are fixed full-resolution calculations, independent of
//! Symthaea's adaptive cognitive stride.

use crate::{
    ChemicalBridgeTarget, ChemicalEncodingSpaceId, ChemicalModalBridgeInput,
    ChemicalRootProjectionError, ChemicalRootProjector,
};
use crate::projection_geometry::exact_cosine;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChemicalProjectionStudyError {
    TooFewSamples { actual: usize, minimum: usize },
    Projection(ChemicalRootProjectionError),
    MixedTargets {
        expected: ChemicalBridgeTarget,
        actual: ChemicalBridgeTarget,
    },
    MixedEncodingSpaces {
        expected: ChemicalEncodingSpaceId,
        actual: ChemicalEncodingSpaceId,
    },
}

impl From<ChemicalRootProjectionError> for ChemicalProjectionStudyError {
    fn from(value: ChemicalRootProjectionError) -> Self {
        Self::Projection(value)
    }
}

/// Descriptive statistics for one comparable set of projected chemical inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct ChemicalProjectionDatasetAssessment {
    pub target: ChemicalBridgeTarget,
    pub encoding_space_id: ChemicalEncodingSpaceId,
    pub sample_count: usize,
    pub pair_count: usize,
    /// Mean |continuous cosine - projected bipolar cosine| over all unique pairs.
    pub mean_absolute_similarity_distortion: f32,
    /// Worst absolute pairwise similarity distortion in the set.
    pub max_absolute_similarity_distortion: f32,
    /// Pearson correlation between all pairwise ContinuousHV similarities and
    /// projected bipolar similarities. `None` when either side has zero variance.
    pub pairwise_similarity_correlation: Option<f32>,
    /// Fraction of samples whose strongest ContinuousHV neighbor remains among
    /// the strongest projected neighbors. Projected ties count as preserved.
    pub nearest_neighbor_preservation: f32,
    /// Mean full-resolution cosine between each source ContinuousHV and its own
    /// bipolar expansion.
    pub mean_source_to_bipolar_similarity: f32,
    /// Mean fraction of positive bits in projected BinaryHV vectors.
    pub mean_positive_fraction: f32,
    /// Largest absolute deviation of projected bit density from 0.5.
    pub max_bit_balance_deviation: f32,
}

impl ChemicalRootProjector {
    /// Assess one modality/encoding-space dataset under this projector's fixed
    /// projection policy.
    pub fn assess_dataset(
        &self,
        inputs: &[ChemicalModalBridgeInput],
    ) -> Result<ChemicalProjectionDatasetAssessment, ChemicalProjectionStudyError> {
        if inputs.len() < 2 {
            return Err(ChemicalProjectionStudyError::TooFewSamples {
                actual: inputs.len(),
                minimum: 2,
            });
        }

        let expected_target = inputs[0].target;
        let expected_space = inputs[0].encoding_space_id;
        for input in inputs {
            if input.target != expected_target {
                return Err(ChemicalProjectionStudyError::MixedTargets {
                    expected: expected_target,
                    actual: input.target,
                });
            }
            if input.encoding_space_id != expected_space {
                return Err(ChemicalProjectionStudyError::MixedEncodingSpaces {
                    expected: expected_space,
                    actual: input.encoding_space_id,
                });
            }
        }

        // `project` revalidates the public bridge receipt and all numeric
        // invariants. Keeping those checks here prevents a dataset report from
        // laundering a forged individual input into apparently valid statistics.
        let projections = inputs
            .iter()
            .map(|input| self.project(input))
            .collect::<Result<Vec<_>, _>>()?;
        let bipolar = projections
            .iter()
            .map(|projection| projection.binary_vector.to_continuous())
            .collect::<Vec<_>>();

        let mut continuous_pairs = Vec::new();
        let mut projected_pairs = Vec::new();
        let mut distortions = Vec::new();
        for left in 0..inputs.len() {
            for right in (left + 1)..inputs.len() {
                let continuous = exact_cosine(&inputs[left].vector, &inputs[right].vector)
                    .map_err(|_| {
                        ChemicalProjectionStudyError::Projection(
                            ChemicalRootProjectionError::DegenerateVector,
                        )
                    })?;
                let projected = exact_cosine(&bipolar[left], &bipolar[right]).map_err(|_| {
                    ChemicalProjectionStudyError::Projection(
                        ChemicalRootProjectionError::DegenerateVector,
                    )
                })?;
                continuous_pairs.push(continuous);
                projected_pairs.push(projected);
                distortions.push((continuous - projected).abs());
            }
        }

        let pair_count = distortions.len();
        let mean_absolute_similarity_distortion =
            distortions.iter().sum::<f32>() / pair_count as f32;
        let max_absolute_similarity_distortion =
            distortions.iter().copied().fold(0.0f32, f32::max);
        let pairwise_similarity_correlation = pearson(&continuous_pairs, &projected_pairs);
        let nearest_neighbor_preservation = nearest_neighbor_preservation(inputs, &bipolar)?;
        let mean_source_to_bipolar_similarity = projections
            .iter()
            .map(|projection| projection.quality.source_to_bipolar_similarity)
            .sum::<f32>()
            / projections.len() as f32;
        let mean_positive_fraction = projections
            .iter()
            .map(|projection| projection.quality.positive_fraction)
            .sum::<f32>()
            / projections.len() as f32;
        let max_bit_balance_deviation = projections
            .iter()
            .map(|projection| (projection.quality.positive_fraction - 0.5).abs())
            .fold(0.0f32, f32::max);

        Ok(ChemicalProjectionDatasetAssessment {
            target: expected_target,
            encoding_space_id: expected_space,
            sample_count: inputs.len(),
            pair_count,
            mean_absolute_similarity_distortion,
            max_absolute_similarity_distortion,
            pairwise_similarity_correlation,
            nearest_neighbor_preservation,
            mean_source_to_bipolar_similarity,
            mean_positive_fraction,
            max_bit_balance_deviation,
        })
    }
}

fn nearest_neighbor_preservation(
    inputs: &[ChemicalModalBridgeInput],
    projected: &[symthaea_core::hdc::unified_hv::ContinuousHV],
) -> Result<f32, ChemicalProjectionStudyError> {
    const TIE_EPSILON: f32 = 1e-6;
    let mut preserved = 0usize;

    for anchor in 0..inputs.len() {
        let mut continuous_best_index = None;
        let mut continuous_best_similarity = f32::NEG_INFINITY;
        for candidate in 0..inputs.len() {
            if candidate == anchor {
                continue;
            }
            let similarity = exact_cosine(&inputs[anchor].vector, &inputs[candidate].vector)
                .map_err(|_| {
                    ChemicalProjectionStudyError::Projection(
                        ChemicalRootProjectionError::DegenerateVector,
                    )
                })?;
            if similarity > continuous_best_similarity {
                continuous_best_similarity = similarity;
                continuous_best_index = Some(candidate);
            }
        }
        let continuous_best_index =
            continuous_best_index.expect("dataset size >= 2 validated by caller");

        let mut projected_best_similarity = f32::NEG_INFINITY;
        for candidate in 0..projected.len() {
            if candidate == anchor {
                continue;
            }
            let similarity = exact_cosine(&projected[anchor], &projected[candidate]).map_err(|_| {
                ChemicalProjectionStudyError::Projection(
                    ChemicalRootProjectionError::DegenerateVector,
                )
            })?;
            projected_best_similarity = projected_best_similarity.max(similarity);
        }
        let original_neighbor_projected_similarity =
            exact_cosine(&projected[anchor], &projected[continuous_best_index]).map_err(|_| {
                ChemicalProjectionStudyError::Projection(
                    ChemicalRootProjectionError::DegenerateVector,
                )
            })?;
        if original_neighbor_projected_similarity + TIE_EPSILON >= projected_best_similarity {
            preserved += 1;
        }
    }

    Ok(preserved as f32 / inputs.len() as f32)
}

fn pearson(left: &[f32], right: &[f32]) -> Option<f32> {
    debug_assert_eq!(left.len(), right.len());
    if left.is_empty() {
        return None;
    }

    let n = left.len() as f32;
    let left_mean = left.iter().sum::<f32>() / n;
    let right_mean = right.iter().sum::<f32>() / n;
    let mut covariance = 0.0f32;
    let mut left_variance = 0.0f32;
    let mut right_variance = 0.0f32;
    for (&x, &y) in left.iter().zip(right) {
        let dx = x - left_mean;
        let dy = y - right_mean;
        covariance += dx * dy;
        left_variance += dx * dx;
        right_variance += dy * dy;
    }
    let denominator = (left_variance * right_variance).sqrt();
    if denominator <= f32::EPSILON {
        None
    } else {
        Some((covariance / denominator).clamp(-1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CalibrationState, ChannelEncodingSpec, ChemicalChannel, ChemicalFingerprintEncoder,
        ChemicalModalBridge, ChemicalModality, ChemicalObservation, ChemicalPercept,
        MeasurementUnit, SensorHealth,
    };

    fn inputs(values: &[f32]) -> Vec<ChemicalModalBridgeInput> {
        let encoder = ChemicalFingerprintEncoder::new(vec![ChannelEncodingSpec::new(
            "voc",
            MeasurementUnit::PartsPerMillion,
            0.0,
            100.0,
            11,
            11,
            101,
        )])
        .unwrap();
        let bridge = ChemicalModalBridge::default();
        values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                let observation = ChemicalObservation::new(
                    index as u64 + 1,
                    ChemicalModality::Olfactory,
                    format!("nose-{index}"),
                    vec![ChemicalChannel {
                        name: "voc".into(),
                        raw_value: *value,
                        unit: MeasurementUnit::PartsPerMillion,
                        calibration: CalibrationState::identity("cal-v1"),
                        health: SensorHealth::default(),
                    }],
                );
                let percept = ChemicalPercept {
                    fingerprint: encoder.encode(&observation).unwrap().unwrap(),
                    evidence: observation,
                };
                bridge.aggregate(&[percept]).unwrap()
            })
            .collect()
    }

    #[test]
    fn dataset_assessment_is_deterministic_and_bounded() {
        let samples = inputs(&[10.0, 11.0, 25.0, 50.0, 90.0]);
        let projector = ChemicalRootProjector::default();
        let a = projector.assess_dataset(&samples).unwrap();
        let b = projector.assess_dataset(&samples).unwrap();

        assert_eq!(a, b);
        assert_eq!(a.sample_count, 5);
        assert_eq!(a.pair_count, 10);
        assert!(a.mean_absolute_similarity_distortion >= 0.0);
        assert!(a.max_absolute_similarity_distortion >= a.mean_absolute_similarity_distortion);
        assert!((0.0..=1.0).contains(&a.nearest_neighbor_preservation));
        assert!((-1.0..=1.0).contains(&a.mean_source_to_bipolar_similarity));
        assert!((0.0..=1.0).contains(&a.mean_positive_fraction));
        assert!((0.0..=0.5).contains(&a.max_bit_balance_deviation));
        if let Some(correlation) = a.pairwise_similarity_correlation {
            assert!((-1.0..=1.0).contains(&correlation));
        }
    }

    #[test]
    fn too_small_dataset_is_rejected() {
        let one = inputs(&[50.0]);
        assert!(matches!(
            ChemicalRootProjector::default().assess_dataset(&one),
            Err(ChemicalProjectionStudyError::TooFewSamples {
                actual: 1,
                minimum: 2,
            })
        ));
    }

    #[test]
    fn current_scalar_fixture_reports_neighborhood_metric_without_gate() {
        let samples = inputs(&[10.0, 11.0, 30.0, 31.0, 70.0, 71.0, 90.0]);
        let assessment = ChemicalRootProjector::default()
            .assess_dataset(&samples)
            .unwrap();

        assert!((0.0..=1.0).contains(&assessment.nearest_neighbor_preservation));
    }

    #[test]
    fn forged_input_is_not_laundered_by_dataset_statistics() {
        let mut samples = inputs(&[10.0, 20.0, 30.0]);
        samples[1].confidence = f32::NAN;
        assert!(matches!(
            ChemicalRootProjector::default().assess_dataset(&samples),
            Err(ChemicalProjectionStudyError::Projection(
                ChemicalRootProjectionError::InvalidConfidence
            ))
        ));
    }

    #[test]
    fn pearson_handles_constant_series_without_fake_correlation() {
        assert_eq!(pearson(&[1.0, 1.0], &[0.0, 1.0]), None);
        assert_eq!(pearson(&[0.0, 1.0], &[1.0, 1.0]), None);
    }
}
