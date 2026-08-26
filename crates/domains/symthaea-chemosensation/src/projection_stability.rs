// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Threshold-margin diagnostics for the lossy ContinuousHV -> BinaryHV boundary.
//!
//! Pairwise similarity preservation is necessary but not sufficient for a stable
//! sign projection. A ContinuousHV can project to a plausible BinaryHV while many
//! dimensions sit arbitrarily close to the quantization threshold; tiny upstream
//! perturbations could then flip many bits. This module makes that brittleness
//! observable without inventing an acceptance threshold before held-out physical
//! data exists.

use crate::{
    ChemicalModalBridgeInput, ChemicalProjectionStudyError, ChemicalRootProjectionError,
    ChemicalRootProjector,
};

/// Descriptive distance-to-threshold statistics for one validated projection.
///
/// All values are absolute distances in the source ContinuousHV coordinate
/// system. Smaller values mean less perturbation is required to flip projected
/// bits. These are diagnostics, not pass/fail criteria.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChemicalProjectionMarginAssessment {
    pub threshold: f32,
    pub minimum_absolute_margin: f32,
    pub p01_absolute_margin: f32,
    pub p05_absolute_margin: f32,
    pub median_absolute_margin: f32,
    pub mean_absolute_margin: f32,
}

/// Dataset-level summary of threshold stability for comparable chemical inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChemicalProjectionStabilityDatasetAssessment {
    pub sample_count: usize,
    /// Mean of each sample's 1st-percentile absolute threshold margin.
    pub mean_p01_absolute_margin: f32,
    /// Smallest 1st-percentile margin observed in the dataset.
    pub minimum_p01_absolute_margin: f32,
    /// Mean of each sample's 5th-percentile absolute threshold margin.
    pub mean_p05_absolute_margin: f32,
    /// Smallest 5th-percentile margin observed in the dataset.
    pub minimum_p05_absolute_margin: f32,
    /// Mean median threshold margin across samples.
    pub mean_median_absolute_margin: f32,
    /// Smallest single-dimension threshold margin anywhere in the dataset.
    pub global_minimum_absolute_margin: f32,
}

impl ChemicalRootProjector {
    /// Measure how close a validated ContinuousHV lies to this projector's sign
    /// threshold.
    ///
    /// Calling [`ChemicalRootProjector::project`] first intentionally reuses the
    /// projection boundary's evidence, dimension, finiteness, confidence, and
    /// non-degeneracy validation rather than allowing stability diagnostics to
    /// launder malformed public bridge inputs.
    pub fn assess_threshold_margins(
        &self,
        input: &ChemicalModalBridgeInput,
    ) -> Result<ChemicalProjectionMarginAssessment, ChemicalRootProjectionError> {
        self.project(input)?;
        Ok(summarize_margins(
            &input.vector.values,
            self.config().threshold,
        ))
    }

    /// Aggregate threshold-margin diagnostics over a comparable dataset.
    ///
    /// The existing dataset assessment is invoked first so target/encoding-space
    /// comparability and every individual bridge receipt are validated under one
    /// contract. No stability acceptance threshold is declared here.
    pub fn assess_dataset_stability(
        &self,
        inputs: &[ChemicalModalBridgeInput],
    ) -> Result<ChemicalProjectionStabilityDatasetAssessment, ChemicalProjectionStudyError> {
        self.assess_dataset(inputs)?;

        let margins = inputs
            .iter()
            .map(|input| self.assess_threshold_margins(input))
            .collect::<Result<Vec<_>, _>>()
            .map_err(ChemicalProjectionStudyError::Projection)?;

        let sample_count = margins.len();
        debug_assert!(sample_count >= 2);
        let divisor = sample_count as f32;

        Ok(ChemicalProjectionStabilityDatasetAssessment {
            sample_count,
            mean_p01_absolute_margin: margins
                .iter()
                .map(|assessment| assessment.p01_absolute_margin)
                .sum::<f32>()
                / divisor,
            minimum_p01_absolute_margin: margins
                .iter()
                .map(|assessment| assessment.p01_absolute_margin)
                .fold(f32::INFINITY, f32::min),
            mean_p05_absolute_margin: margins
                .iter()
                .map(|assessment| assessment.p05_absolute_margin)
                .sum::<f32>()
                / divisor,
            minimum_p05_absolute_margin: margins
                .iter()
                .map(|assessment| assessment.p05_absolute_margin)
                .fold(f32::INFINITY, f32::min),
            mean_median_absolute_margin: margins
                .iter()
                .map(|assessment| assessment.median_absolute_margin)
                .sum::<f32>()
                / divisor,
            global_minimum_absolute_margin: margins
                .iter()
                .map(|assessment| assessment.minimum_absolute_margin)
                .fold(f32::INFINITY, f32::min),
        })
    }
}

fn summarize_margins(values: &[f32], threshold: f32) -> ChemicalProjectionMarginAssessment {
    debug_assert!(!values.is_empty());
    debug_assert!(threshold.is_finite());
    debug_assert!(values.iter().all(|value| value.is_finite()));

    let mut margins = values
        .iter()
        .map(|value| (*value - threshold).abs())
        .collect::<Vec<_>>();
    margins.sort_by(|left, right| {
        left.partial_cmp(right)
            .expect("finite margins validated by projection boundary")
    });

    let mean_absolute_margin =
        (margins.iter().map(|value| *value as f64).sum::<f64>() / margins.len() as f64) as f32;

    ChemicalProjectionMarginAssessment {
        threshold,
        minimum_absolute_margin: margins[0],
        p01_absolute_margin: nearest_rank(&margins, 0.01),
        p05_absolute_margin: nearest_rank(&margins, 0.05),
        median_absolute_margin: nearest_rank(&margins, 0.50),
        mean_absolute_margin,
    }
}

/// Deterministic nearest-rank-like quantile over a sorted non-empty slice.
///
/// Indexing uses `floor(q * (n - 1))`, which keeps endpoints exact and avoids
/// interpolation policy becoming another hidden experiment parameter.
fn nearest_rank(sorted: &[f32], quantile: f64) -> f32 {
    debug_assert!(!sorted.is_empty());
    debug_assert!((0.0..=1.0).contains(&quantile));
    let index = (quantile * (sorted.len() - 1) as f64).floor() as usize;
    sorted[index]
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
    fn margin_summary_is_deterministic_and_ordered() {
        let values = [-2.0, -0.1, 0.0, 0.2, 3.0];
        let a = summarize_margins(&values, 0.0);
        let b = summarize_margins(&values, 0.0);
        assert_eq!(a, b);
        assert_eq!(a.minimum_absolute_margin, 0.0);
        assert_eq!(a.p01_absolute_margin, 0.0);
        assert_eq!(a.p05_absolute_margin, 0.0);
        assert_eq!(a.median_absolute_margin, 0.2);
        assert!((a.mean_absolute_margin - 1.06).abs() < 1e-6);
    }

    #[test]
    fn shifting_threshold_changes_margin_receipt() {
        let values = [-1.0, -0.25, 0.25, 1.0];
        let zero = summarize_margins(&values, 0.0);
        let shifted = summarize_margins(&values, 0.24);
        assert_ne!(zero, shifted);
        assert!(shifted.minimum_absolute_margin < zero.minimum_absolute_margin);
    }

    #[test]
    fn real_projection_margin_receipt_is_finite_and_monotone() {
        let sample = inputs(&[50.0]).pop().unwrap();
        let assessment = ChemicalRootProjector::default()
            .assess_threshold_margins(&sample)
            .unwrap();

        assert_eq!(assessment.threshold, 0.0);
        assert!(assessment.minimum_absolute_margin.is_finite());
        assert!(assessment.minimum_absolute_margin >= 0.0);
        assert!(assessment.minimum_absolute_margin <= assessment.p01_absolute_margin);
        assert!(assessment.p01_absolute_margin <= assessment.p05_absolute_margin);
        assert!(assessment.p05_absolute_margin <= assessment.median_absolute_margin);
        assert!(assessment.mean_absolute_margin >= 0.0);
    }

    #[test]
    fn dataset_stability_is_descriptive_and_deterministic() {
        let samples = inputs(&[10.0, 11.0, 30.0, 50.0, 90.0]);
        let projector = ChemicalRootProjector::default();
        let a = projector.assess_dataset_stability(&samples).unwrap();
        let b = projector.assess_dataset_stability(&samples).unwrap();

        assert_eq!(a, b);
        assert_eq!(a.sample_count, samples.len());
        assert!(a.global_minimum_absolute_margin >= 0.0);
        assert!(a.minimum_p01_absolute_margin >= a.global_minimum_absolute_margin);
        assert!(a.minimum_p05_absolute_margin >= a.minimum_p01_absolute_margin);
        assert!(a.mean_p01_absolute_margin >= a.minimum_p01_absolute_margin);
        assert!(a.mean_p05_absolute_margin >= a.minimum_p05_absolute_margin);
        assert!(a.mean_median_absolute_margin >= 0.0);
    }
}
