// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cognitive boundary for chemical evidence.
//!
//! The pipeline derives a percept, temporal context, and novelty assessment in
//! that order. Novelty memory admission remains an explicit operation: merely
//! perceiving an unfamiliar chemical pattern does not teach the system that the
//! pattern is normal.

use crate::{
    ChemicalEncodingSpaceId, ChemicalNoveltyMemory, ChemicalObservation, ChemicalObservationId,
    ChemicalPercept, ChemicalPerceptEncoder, ChemicalTemporalContext, ChemicalTemporalTracker,
    FingerprintError, NoveltyAssessment, TemporalError,
};

#[derive(Debug, Clone, PartialEq)]
pub struct CognitiveChemicalPercept {
    pub percept: ChemicalPercept,
    pub temporal: ChemicalTemporalContext,
    pub novelty: NoveltyAssessment,
}

impl CognitiveChemicalPercept {
    pub fn observation_id(&self) -> ChemicalObservationId {
        self.percept.observation_id()
    }

    pub fn encoding_space_id(&self) -> ChemicalEncodingSpaceId {
        self.percept.encoding_space_id()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChemicalCognitionError {
    Fingerprint(FingerprintError),
    Temporal(TemporalError),
}

impl From<FingerprintError> for ChemicalCognitionError {
    fn from(value: FingerprintError) -> Self {
        Self::Fingerprint(value)
    }
}

impl From<TemporalError> for ChemicalCognitionError {
    fn from(value: TemporalError) -> Self {
        Self::Temporal(value)
    }
}

#[derive(Debug, Clone)]
pub struct ChemicalCognitionPipeline {
    encoder: ChemicalPerceptEncoder,
    temporal: ChemicalTemporalTracker,
    novelty: ChemicalNoveltyMemory,
}

impl ChemicalCognitionPipeline {
    pub fn new(
        encoder: ChemicalPerceptEncoder,
        temporal: ChemicalTemporalTracker,
        novelty: ChemicalNoveltyMemory,
    ) -> Self {
        Self {
            encoder,
            temporal,
            novelty,
        }
    }

    pub fn novelty_memory(&self) -> &ChemicalNoveltyMemory {
        &self.novelty
    }

    pub fn temporal_tracker(&self) -> &ChemicalTemporalTracker {
        &self.temporal
    }

    /// Convert one observation into cognitive chemical evidence.
    ///
    /// `Ok(None)` is genuine absence of trustworthy configured evidence. The
    /// operation never manufactures a zero percept to fill a missing modality.
    /// Missing clock provenance does not erase valid chemistry; the temporal
    /// context simply withholds elapsed-time claims and does not create an anchor.
    pub fn perceive(
        &mut self,
        observation: &ChemicalObservation,
    ) -> Result<Option<CognitiveChemicalPercept>, ChemicalCognitionError> {
        let Some(percept) = self.encoder.encode(observation)? else {
            return Ok(None);
        };

        // The temporal tracker validates declared clock/time ordering before
        // mutating its anchor. Novelty assessment is read-only, so a temporal
        // integrity error leaves cognitive state unchanged.
        let temporal = self.temporal.observe(&percept)?;
        let novelty = self.novelty.assess(&percept);

        Ok(Some(CognitiveChemicalPercept {
            percept,
            temporal,
            novelty,
        }))
    }

    /// Explicitly admit a perceived chemical pattern to novelty memory.
    ///
    /// Returns true only if the memory accepted a new trustworthy reference.
    pub fn remember(&mut self, percept: &CognitiveChemicalPercept) -> bool {
        self.novelty.admit(&percept.percept)
    }

    pub fn clear_state(&mut self) {
        self.temporal.clear();
        self.novelty.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CalibrationState, ChannelEncodingSpec, ChemicalChannel, ChemicalClockDomainId,
        ChemicalFingerprintEncoder, ChemicalModality, ChemicalNoveltyConfig, MeasurementUnit,
        SensorHealth,
    };

    fn pipeline() -> ChemicalCognitionPipeline {
        let fingerprint = ChemicalFingerprintEncoder::new(vec![ChannelEncodingSpec::new(
            "voc",
            MeasurementUnit::PartsPerMillion,
            0.0,
            100.0,
            16,
            11,
            101,
        )])
        .unwrap();
        let encoder = ChemicalPerceptEncoder::new(fingerprint);
        let temporal = ChemicalTemporalTracker::new(0.5).unwrap();
        let novelty = ChemicalNoveltyMemory::new(ChemicalNoveltyConfig::default()).unwrap();
        ChemicalCognitionPipeline::new(encoder, temporal, novelty)
    }

    fn test_clock() -> ChemicalClockDomainId {
        ChemicalClockDomainId::new("test-rig/monotonic").unwrap()
    }

    fn observation(timestamp_us: u64, value: f32, confidence: f32) -> ChemicalObservation {
        ChemicalObservation::new(
            timestamp_us,
            ChemicalModality::Olfactory,
            "nose-a",
            vec![ChemicalChannel {
                name: "voc".into(),
                raw_value: value,
                unit: MeasurementUnit::PartsPerMillion,
                calibration: CalibrationState::identity("cal-a"),
                health: SensorHealth {
                    score: confidence,
                    saturated: false,
                    contaminated: false,
                },
            }],
        )
        .with_clock_domain(test_clock())
    }

    #[test]
    fn perception_does_not_implicitly_learn_novel_pattern() {
        let mut pipeline = pipeline();
        let first = pipeline
            .perceive(&observation(1_000_000, 20.0, 0.9))
            .unwrap()
            .unwrap();
        assert_eq!(first.novelty.novelty, 1.0);
        assert_eq!(pipeline.novelty_memory().len(ChemicalModality::Olfactory), 0);

        let second = pipeline
            .perceive(&observation(2_000_000, 20.0, 0.9))
            .unwrap()
            .unwrap();
        assert_eq!(second.novelty.novelty, 1.0);
    }

    #[test]
    fn unclocked_chemistry_remains_perceivable_without_temporal_claims() {
        let mut pipeline = pipeline();
        let mut raw = observation(1_000_000, 20.0, 0.9);
        raw.clock_domain = None;
        let perceived = pipeline.perceive(&raw).unwrap().unwrap();
        assert_eq!(perceived.temporal.clock_domain, None);
        assert_eq!(perceived.temporal.elapsed_s, None);
        assert_eq!(perceived.temporal.change_rate_per_s, None);
        assert!(!perceived.temporal.anchor_updated);
        assert_eq!(perceived.novelty.novelty, 1.0);
    }

    #[test]
    fn cognitive_output_exposes_evidence_and_representation_identity() {
        let mut pipeline = pipeline();
        let raw = observation(1_000_000, 20.0, 0.9);
        let expected_observation_id = ChemicalObservationId::from_observation(&raw);
        let perceived = pipeline.perceive(&raw).unwrap().unwrap();

        assert_eq!(perceived.observation_id(), expected_observation_id);
        assert_eq!(
            perceived.encoding_space_id(),
            perceived.percept.fingerprint.encoding_space_id
        );
    }

    #[test]
    fn explicit_memory_admission_reduces_repeat_novelty() {
        let mut pipeline = pipeline();
        let first = pipeline
            .perceive(&observation(1_000_000, 20.0, 0.9))
            .unwrap()
            .unwrap();
        assert!(pipeline.remember(&first));

        let repeated = pipeline
            .perceive(&observation(2_000_000, 20.0, 0.9))
            .unwrap()
            .unwrap();
        assert!(repeated.novelty.novelty < 1e-6);
        assert_eq!(repeated.novelty.nearest.unwrap().source, "nose-a");
    }

    #[test]
    fn no_usable_evidence_returns_none() {
        let mut pipeline = pipeline();
        assert!(pipeline
            .perceive(&observation(1_000_000, 20.0, 0.0))
            .unwrap()
            .is_none());
    }

    #[test]
    fn low_confidence_percept_is_not_admitted_to_memory() {
        let mut pipeline = pipeline();
        let weak = pipeline
            .perceive(&observation(1_000_000, 20.0, 0.2))
            .unwrap()
            .unwrap();
        assert!(!pipeline.remember(&weak));
        assert_eq!(pipeline.novelty_memory().len(ChemicalModality::Olfactory), 0);
    }

    #[test]
    fn out_of_order_evidence_is_a_cognitive_integrity_error() {
        let mut pipeline = pipeline();
        pipeline
            .perceive(&observation(2_000_000, 20.0, 0.9))
            .unwrap()
            .unwrap();
        assert!(matches!(
            pipeline.perceive(&observation(1_000_000, 30.0, 0.9)),
            Err(ChemicalCognitionError::Temporal(
                TemporalError::NonMonotonicTimestamp { .. }
            ))
        ));
    }

    #[test]
    fn clock_migration_is_a_cognitive_integrity_boundary() {
        let mut pipeline = pipeline();
        pipeline
            .perceive(&observation(2_000_000, 20.0, 0.9))
            .unwrap()
            .unwrap();
        let mut migrated = observation(3_000_000, 30.0, 0.9);
        migrated.clock_domain =
            Some(ChemicalClockDomainId::new("other-rig/monotonic").unwrap());
        assert!(matches!(
            pipeline.perceive(&migrated),
            Err(ChemicalCognitionError::Temporal(
                TemporalError::ClockDomainMismatch { .. }
            ))
        ));
    }
}
