// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Typed epistemic provenance for visual cognition.
//!
//! The core invariant is intentionally asymmetric:
//!
//! ```text
//! observation evidence may originate only at an observation boundary
//! prediction / memory / simulation / counterfactual state may never upgrade itself to observed
//! ```
//!
//! Observation identity is also explicit about stream restarts and capture-clock semantics.
//! Numeric timestamps are never assumed comparable merely because they share a unit.
//!
//! This module does not grant action authority. It only preserves what kind of visual
//! claim a datum represents and the evidence it is allowed to cite.

use serde::{Deserialize, Deserializer, Serialize};
use std::{cmp::Ordering, fmt};

/// Epistemic origin of a visual claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualOrigin {
    /// Direct sensor evidence from a concrete observation boundary.
    Observed,
    /// Recalled evidence whose original observation lineage is retained.
    Remembered,
    /// A belief inferred from one or more evidence-bearing claims.
    Inferred,
    /// A forecast about a future visual state.
    Predicted,
    /// A model-generated visual state used for internal simulation.
    Simulated,
    /// A hypothetical visual state under an explicit intervention or assumption.
    Counterfactual,
}

impl VisualOrigin {
    /// Returns true only for direct sensor-originated claims.
    pub const fn is_observed(self) -> bool {
        matches!(self, Self::Observed)
    }

    /// Returns true for origins that are generated rather than directly sensed.
    pub const fn is_generative(self) -> bool {
        matches!(self, Self::Predicted | Self::Simulated | Self::Counterfactual)
    }
}

/// Declared semantic domain of an observation timestamp.
///
/// The value describes what the numeric `captured_at_us` means. It does not itself prove
/// synchronization quality, clock authenticity, or a bound on timing error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualCaptureClock {
    /// Unix-epoch-relative microseconds. Values from different streams may be numerically
    /// ordered, but this declaration alone does not prove synchronization accuracy.
    UnixEpoch,
    /// Monotonic time local to one exact stream epoch.
    StreamMonotonic,
    /// A device-local acquisition clock. Values are comparable only inside one stream epoch.
    DeviceLocal,
    /// The producer supplied a number but did not establish its clock semantics.
    Unspecified,
}

/// Stable identity for one live/replayed capture stream instance.
///
/// `stream_epoch` distinguishes restarts/reconnections that may reuse frame counters or local
/// timestamps. Source owners choose both values; the vision layer does not derive them from
/// frame number, wall clock, or model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct VisualStreamRef {
    source_id: u64,
    stream_epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
struct VisualStreamRefWire {
    source_id: u64,
    stream_epoch: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualObservationError {
    MissingSourceIdentity,
    MissingStreamEpoch,
}

impl fmt::Display for VisualObservationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingSourceIdentity => {
                f.write_str("visual stream source identity must be non-zero")
            }
            Self::MissingStreamEpoch => f.write_str("visual stream epoch must be non-zero"),
        }
    }
}

impl std::error::Error for VisualObservationError {}

impl VisualStreamRef {
    pub fn new(source_id: u64, stream_epoch: u64) -> Result<Self, VisualObservationError> {
        if source_id == 0 {
            return Err(VisualObservationError::MissingSourceIdentity);
        }
        if stream_epoch == 0 {
            return Err(VisualObservationError::MissingStreamEpoch);
        }
        Ok(Self {
            source_id,
            stream_epoch,
        })
    }

    pub const fn source_id(self) -> u64 {
        self.source_id
    }

    pub const fn stream_epoch(self) -> u64 {
        self.stream_epoch
    }
}

impl<'de> Deserialize<'de> for VisualStreamRef {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = VisualStreamRefWire::deserialize(deserializer)?;
        Self::new(wire.source_id, wire.stream_epoch).map_err(serde::de::Error::custom)
    }
}

/// Stable identity of one concrete visual observation.
///
/// Fields are private so capture-clock and stream-epoch semantics cannot be bypassed by a
/// struct literal. `frame_id` may start at zero; uniqueness is scoped to the exact stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VisualObservationRef {
    stream: VisualStreamRef,
    frame_id: u64,
    captured_at_us: u64,
    clock_domain: VisualCaptureClock,
}

impl VisualObservationRef {
    pub const fn new(
        stream: VisualStreamRef,
        frame_id: u64,
        captured_at_us: u64,
        clock_domain: VisualCaptureClock,
    ) -> Self {
        Self {
            stream,
            frame_id,
            captured_at_us,
            clock_domain,
        }
    }

    pub const fn stream(self) -> VisualStreamRef {
        self.stream
    }

    pub const fn source_id(self) -> u64 {
        self.stream.source_id()
    }

    pub const fn stream_epoch(self) -> u64 {
        self.stream.stream_epoch()
    }

    pub const fn frame_id(self) -> u64 {
        self.frame_id
    }

    pub const fn captured_at_us(self) -> u64 {
        self.captured_at_us
    }

    pub const fn clock_domain(self) -> VisualCaptureClock {
        self.clock_domain
    }

    /// Frame ordering is defined only inside one exact stream epoch.
    pub fn frame_ordering(&self, other: &Self) -> Option<Ordering> {
        (self.stream == other.stream).then(|| self.frame_id.cmp(&other.frame_id))
    }

    /// Timestamp ordering is returned only when the declared clock semantics make numeric
    /// comparison meaningful.
    ///
    /// - Unix epoch values can be compared across streams (without claiming synchronization
    ///   accuracy).
    /// - stream-monotonic/device-local values can be compared only inside the same stream.
    /// - unspecified clocks are never numerically ordered by this API.
    pub fn timestamp_ordering(&self, other: &Self) -> Option<Ordering> {
        match (self.clock_domain, other.clock_domain) {
            (VisualCaptureClock::UnixEpoch, VisualCaptureClock::UnixEpoch) => {
                Some(self.captured_at_us.cmp(&other.captured_at_us))
            }
            (VisualCaptureClock::StreamMonotonic, VisualCaptureClock::StreamMonotonic)
            | (VisualCaptureClock::DeviceLocal, VisualCaptureClock::DeviceLocal)
                if self.stream == other.stream =>
            {
                Some(self.captured_at_us.cmp(&other.captured_at_us))
            }
            _ => None,
        }
    }
}

/// A compact, typed provenance envelope for visual claims.
///
/// Fields are private so production code cannot construct contradictory provenance directly.
/// Deserialization is also validated before a `VisualEvidence` value can exist.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VisualEvidence {
    origin: VisualOrigin,
    observation: Option<VisualObservationRef>,
    parent_observations: Vec<VisualObservationRef>,
    confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
struct VisualEvidenceWire {
    origin: VisualOrigin,
    observation: Option<VisualObservationRef>,
    parent_observations: Vec<VisualObservationRef>,
    confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisualEvidenceError {
    InvalidConfidence,
    ObservedRequiresObservation,
    ObservedCannotCarryParentLineage,
    NonObservedCannotCarryDirectObservation,
    RememberedRequiresObservationLineage,
    InferredRequiresObservationLineage,
    DuplicateParentObservation,
}

impl fmt::Display for VisualEvidenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidConfidence => "visual confidence must be finite and within [0, 1]",
            Self::ObservedRequiresObservation => {
                "observed visual evidence requires a direct observation reference"
            }
            Self::ObservedCannotCarryParentLineage => {
                "observed visual evidence cannot carry derived parent-observation lineage"
            }
            Self::NonObservedCannotCarryDirectObservation => {
                "non-observed visual evidence cannot carry the direct-observation slot"
            }
            Self::RememberedRequiresObservationLineage => {
                "remembered visual evidence requires original observation lineage"
            }
            Self::InferredRequiresObservationLineage => {
                "inferred visual evidence requires supporting observation lineage"
            }
            Self::DuplicateParentObservation => {
                "visual evidence cannot count the same parent observation more than once"
            }
        };
        f.write_str(message)
    }
}

impl std::error::Error for VisualEvidenceError {}

impl<'de> Deserialize<'de> for VisualEvidence {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = VisualEvidenceWire::deserialize(deserializer)?;
        let evidence = Self {
            origin: wire.origin,
            observation: wire.observation,
            parent_observations: wire.parent_observations,
            confidence: wire.confidence,
        };
        evidence.validate().map_err(serde::de::Error::custom)?;
        Ok(evidence)
    }
}

impl VisualEvidence {
    /// Construct direct observation evidence.
    ///
    /// This is the only constructor that can produce `VisualOrigin::Observed`.
    pub fn observed(
        observation: VisualObservationRef,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::validate_confidence(confidence)?;
        Ok(Self {
            origin: VisualOrigin::Observed,
            observation: Some(observation),
            parent_observations: Vec::new(),
            confidence,
        })
    }

    /// Construct remembered evidence from one or more original observations.
    pub fn remembered(
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::from_derived(VisualOrigin::Remembered, parent_observations, confidence)
    }

    /// Construct an inference grounded in one or more observations.
    pub fn inferred(
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::from_derived(VisualOrigin::Inferred, parent_observations, confidence)
    }

    /// Construct a prediction. Parent observation lineage is optional because some
    /// forecasts may be generated from a purely simulated initial state.
    pub fn predicted(
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::from_generated(VisualOrigin::Predicted, parent_observations, confidence)
    }

    /// Construct internal simulation evidence.
    pub fn simulated(
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::from_generated(VisualOrigin::Simulated, parent_observations, confidence)
    }

    /// Construct an explicit counterfactual visual claim.
    pub fn counterfactual(
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::from_generated(VisualOrigin::Counterfactual, parent_observations, confidence)
    }

    pub const fn origin(&self) -> VisualOrigin {
        self.origin
    }

    pub const fn observation(&self) -> Option<VisualObservationRef> {
        self.observation
    }

    pub fn parent_observations(&self) -> &[VisualObservationRef] {
        &self.parent_observations
    }

    pub const fn confidence(&self) -> f32 {
        self.confidence
    }

    /// Revalidate an envelope after any in-memory trust-boundary crossing.
    ///
    /// Wire deserialization already performs this validation automatically.
    pub fn validate(&self) -> Result<(), VisualEvidenceError> {
        Self::validate_confidence(self.confidence)?;
        Self::validate_unique_parents(&self.parent_observations)?;

        match self.origin {
            VisualOrigin::Observed => {
                if self.observation.is_none() {
                    return Err(VisualEvidenceError::ObservedRequiresObservation);
                }
                if !self.parent_observations.is_empty() {
                    return Err(VisualEvidenceError::ObservedCannotCarryParentLineage);
                }
            }
            VisualOrigin::Remembered => {
                if self.observation.is_some() {
                    return Err(VisualEvidenceError::NonObservedCannotCarryDirectObservation);
                }
                if self.parent_observations.is_empty() {
                    return Err(VisualEvidenceError::RememberedRequiresObservationLineage);
                }
            }
            VisualOrigin::Inferred => {
                if self.observation.is_some() {
                    return Err(VisualEvidenceError::NonObservedCannotCarryDirectObservation);
                }
                if self.parent_observations.is_empty() {
                    return Err(VisualEvidenceError::InferredRequiresObservationLineage);
                }
            }
            VisualOrigin::Predicted | VisualOrigin::Simulated | VisualOrigin::Counterfactual => {
                if self.observation.is_some() {
                    return Err(VisualEvidenceError::NonObservedCannotCarryDirectObservation);
                }
            }
        }

        Ok(())
    }

    fn from_derived(
        origin: VisualOrigin,
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::validate_confidence(confidence)?;
        Self::validate_unique_parents(&parent_observations)?;
        if parent_observations.is_empty() {
            return Err(match origin {
                VisualOrigin::Remembered => VisualEvidenceError::RememberedRequiresObservationLineage,
                VisualOrigin::Inferred => VisualEvidenceError::InferredRequiresObservationLineage,
                _ => unreachable!("derived constructor accepts only remembered/inferred origins"),
            });
        }
        Ok(Self {
            origin,
            observation: None,
            parent_observations,
            confidence,
        })
    }

    fn from_generated(
        origin: VisualOrigin,
        parent_observations: Vec<VisualObservationRef>,
        confidence: f32,
    ) -> Result<Self, VisualEvidenceError> {
        Self::validate_confidence(confidence)?;
        Self::validate_unique_parents(&parent_observations)?;
        debug_assert!(origin.is_generative());
        Ok(Self {
            origin,
            observation: None,
            parent_observations,
            confidence,
        })
    }

    fn validate_confidence(confidence: f32) -> Result<(), VisualEvidenceError> {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(VisualEvidenceError::InvalidConfidence);
        }
        Ok(())
    }

    fn validate_unique_parents(
        parent_observations: &[VisualObservationRef],
    ) -> Result<(), VisualEvidenceError> {
        for (idx, item) in parent_observations.iter().enumerate() {
            if parent_observations[..idx].contains(item) {
                return Err(VisualEvidenceError::DuplicateParentObservation);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(epoch: u64) -> VisualStreamRef {
        VisualStreamRef::new(7, epoch).unwrap()
    }

    fn obs(frame_id: u64) -> VisualObservationRef {
        VisualObservationRef::new(
            stream(11),
            frame_id,
            1_000_000 + frame_id,
            VisualCaptureClock::StreamMonotonic,
        )
    }

    #[test]
    fn stream_identity_rejects_zero_components() {
        assert_eq!(
            VisualStreamRef::new(0, 1),
            Err(VisualObservationError::MissingSourceIdentity)
        );
        assert_eq!(
            VisualStreamRef::new(1, 0),
            Err(VisualObservationError::MissingStreamEpoch)
        );
    }

    #[test]
    fn stream_wire_validation_is_fail_closed() {
        let bad_source = r#"{"source_id":0,"stream_epoch":1}"#;
        let bad_epoch = r#"{"source_id":1,"stream_epoch":0}"#;
        assert!(serde_json::from_str::<VisualStreamRef>(bad_source).is_err());
        assert!(serde_json::from_str::<VisualStreamRef>(bad_epoch).is_err());
    }

    #[test]
    fn restart_epoch_prevents_frame_identity_collision() {
        let first = VisualObservationRef::new(
            stream(1),
            5,
            100,
            VisualCaptureClock::StreamMonotonic,
        );
        let restarted = VisualObservationRef::new(
            stream(2),
            5,
            100,
            VisualCaptureClock::StreamMonotonic,
        );
        assert_ne!(first, restarted);
        assert_eq!(first.frame_ordering(&restarted), None);
        assert_eq!(first.timestamp_ordering(&restarted), None);
    }

    #[test]
    fn monotonic_timestamps_compare_only_within_exact_stream() {
        let a = VisualObservationRef::new(
            stream(3),
            1,
            10,
            VisualCaptureClock::StreamMonotonic,
        );
        let b = VisualObservationRef::new(
            stream(3),
            2,
            20,
            VisualCaptureClock::StreamMonotonic,
        );
        let other_stream = VisualObservationRef::new(
            VisualStreamRef::new(8, 3).unwrap(),
            1,
            15,
            VisualCaptureClock::StreamMonotonic,
        );
        assert_eq!(a.timestamp_ordering(&b), Some(Ordering::Less));
        assert_eq!(a.timestamp_ordering(&other_stream), None);
    }

    #[test]
    fn unix_epoch_timestamps_can_be_numerically_compared_across_streams() {
        let a = VisualObservationRef::new(
            VisualStreamRef::new(1, 1).unwrap(),
            1,
            10,
            VisualCaptureClock::UnixEpoch,
        );
        let b = VisualObservationRef::new(
            VisualStreamRef::new(2, 1).unwrap(),
            1,
            11,
            VisualCaptureClock::UnixEpoch,
        );
        assert_eq!(a.timestamp_ordering(&b), Some(Ordering::Less));
    }

    #[test]
    fn unspecified_clock_is_never_numerically_ordered() {
        let a = VisualObservationRef::new(
            stream(4),
            1,
            10,
            VisualCaptureClock::Unspecified,
        );
        let b = VisualObservationRef::new(
            stream(4),
            2,
            20,
            VisualCaptureClock::Unspecified,
        );
        assert_eq!(a.timestamp_ordering(&b), None);
        assert_eq!(a.frame_ordering(&b), Some(Ordering::Less));
    }

    #[test]
    fn observed_requires_direct_observation_identity() {
        let evidence = VisualEvidence::observed(obs(42), 0.9).unwrap();
        assert_eq!(evidence.origin(), VisualOrigin::Observed);
        assert_eq!(evidence.observation(), Some(obs(42)));
        assert!(evidence.parent_observations().is_empty());
        assert!(evidence.validate().is_ok());
    }

    #[test]
    fn remembered_and_inferred_require_observation_lineage() {
        assert_eq!(
            VisualEvidence::remembered(Vec::new(), 0.8),
            Err(VisualEvidenceError::RememberedRequiresObservationLineage)
        );
        assert_eq!(
            VisualEvidence::inferred(Vec::new(), 0.8),
            Err(VisualEvidenceError::InferredRequiresObservationLineage)
        );
    }

    #[test]
    fn generated_origins_never_claim_direct_observation() {
        for evidence in [
            VisualEvidence::predicted(vec![obs(1)], 0.7).unwrap(),
            VisualEvidence::simulated(vec![obs(1)], 0.7).unwrap(),
            VisualEvidence::counterfactual(vec![obs(1)], 0.7).unwrap(),
        ] {
            assert!(!evidence.origin().is_observed());
            assert!(evidence.observation().is_none());
            assert!(evidence.validate().is_ok());
        }
    }

    #[test]
    fn invalid_confidence_fails_closed() {
        assert_eq!(
            VisualEvidence::observed(obs(1), f32::NAN),
            Err(VisualEvidenceError::InvalidConfidence)
        );
        assert_eq!(
            VisualEvidence::simulated(Vec::new(), 1.1),
            Err(VisualEvidenceError::InvalidConfidence)
        );
    }

    #[test]
    fn duplicate_parent_observations_are_rejected() {
        assert_eq!(
            VisualEvidence::predicted(vec![obs(3), obs(3)], 0.6),
            Err(VisualEvidenceError::DuplicateParentObservation)
        );
    }

    #[test]
    fn serde_roundtrip_preserves_origin_and_lineage() {
        let evidence = VisualEvidence::counterfactual(vec![obs(2), obs(3)], 0.55).unwrap();
        let json = serde_json::to_string(&evidence).unwrap();
        let decoded: VisualEvidence = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, evidence);
        assert!(decoded.validate().is_ok());
    }

    #[test]
    fn tampered_wire_state_cannot_deserialize_as_observed() {
        let json = r#"{
            "origin":"observed",
            "observation":null,
            "parent_observations":[],
            "confidence":0.99
        }"#;
        assert!(serde_json::from_str::<VisualEvidence>(json).is_err());
    }

    #[test]
    fn tampered_simulation_cannot_carry_direct_observation() {
        let json = r#"{
            "origin":"simulated",
            "observation":{
                "stream":{"source_id":1,"stream_epoch":1},
                "frame_id":9,
                "captured_at_us":10,
                "clock_domain":"stream_monotonic"
            },
            "parent_observations":[],
            "confidence":0.5
        }"#;
        assert!(serde_json::from_str::<VisualEvidence>(json).is_err());
    }
}
