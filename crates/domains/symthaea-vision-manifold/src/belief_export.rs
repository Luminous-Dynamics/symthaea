// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only, epistemically conservative export of visual object beliefs.
//!
//! This module deliberately exports only semantics the current vision tracker
//! actually owns. In particular:
//!
//! ```text
//! retained track != predicted object != current observation
//! ```
//!
//! The current object tracker retains a stable local track identity, patch-grid
//! kinematics, and the frame at which a track was last matched. It does not yet
//! retain a stable externally-addressable source-observation identity per match,
//! and it does not produce a qualified probability that an object persists.
//! Those facts are represented as unavailable rather than fabricated.

use serde::{Deserialize, Serialize};

use crate::manifold::{TrackedObject, VisionManifold};

/// Stable semantic identity for [`VisualBeliefExportV1`].
pub const VISUAL_BELIEF_EXPORT_SCHEMA_ID: &str = "symthaea.visual-belief-export.v1";
/// Schema version for [`VisualBeliefExportV1`].
pub const VISUAL_BELIEF_EXPORT_SCHEMA_VERSION: u32 = 1;

/// Closed lifecycle vocabulary supported by visual-belief export v1.
///
/// V1 intentionally does not contain an `OccludedPredicted` variant. The current
/// object-memory layer can retain an unmatched track, but retention alone is not
/// proof that a concrete current-state prediction was produced for that object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualBeliefLifecycleV1 {
    /// The track was refreshed by the current frame's object observation path.
    Observed,
    /// The track remains in bounded object memory but was not observed this frame.
    UnobservedRetained,
}

/// Epistemic origin supported by visual-belief export v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualEvidenceOriginV1 {
    /// Backed by a current-frame tracker observation.
    Observed,
    /// Retained from prior observation without claiming a current prediction.
    Remembered,
}

/// Why source-observation lineage is unavailable in v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationLineageUnavailableReasonV1 {
    /// Object memory currently retains track state but not an externally stable
    /// source-observation identifier for each object match.
    TrackerDoesNotRetainExternalObservationIdentity,
}

/// Source-observation lineage for one visual belief.
///
/// V1 has only an explicit unavailable state. Adding concrete observation
/// references is a schema change, not a reinterpretation of existing bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualObservationLineageV1 {
    Unavailable {
        reason: ObservationLineageUnavailableReasonV1,
    },
}

/// Why a qualified persistence confidence is unavailable in v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConfidenceUnavailableReasonV1 {
    /// The current tracker exposes matching/representation evidence, but no
    /// qualified calibrated probability of physical object persistence.
    NoQualifiedPersistenceConfidence,
}

/// Confidence semantics for visual-belief export v1.
///
/// This is intentionally not a bare `f32`. Tracker similarity, scene coherence,
/// FEP confidence, and calibrated persistence probability are different
/// propositions even when each happens to lie in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualBeliefConfidenceV1 {
    Unavailable {
        reason: ConfidenceUnavailableReasonV1,
    },
}

/// Coordinate frame for exported v1 object kinematics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualCoordinateFrameV1 {
    /// Object-memory coordinates measured in the encoder's patch grid.
    PatchGrid,
}

/// Image/patch-grid kinematics retained by current object memory.
///
/// These values are not metric world position or physical velocity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisualGridKinematicsV1 {
    pub coordinate_frame: VisualCoordinateFrameV1,
    pub centroid_row: usize,
    pub centroid_col: usize,
    /// Smoothed row velocity in patch-grid cells per observed frame.
    pub velocity_row: f32,
    /// Smoothed column velocity in patch-grid cells per observed frame.
    pub velocity_col: f32,
}

/// One externally inspectable visual entity belief.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisualEntityBeliefV1 {
    /// Symthaea-local object-memory identity. This is not a world/canonical ID.
    pub hypothesis_id: u64,
    pub lifecycle: VisualBeliefLifecycleV1,
    /// Frame containing the most recent real tracker observation for this track.
    pub last_observed_frame: u64,
    pub evidence_origin: VisualEvidenceOriginV1,
    pub observation_lineage: VisualObservationLineageV1,
    pub persistence_confidence: VisualBeliefConfidenceV1,
    pub kinematics: VisualGridKinematicsV1,
    pub track_length: u64,
}

/// Deterministic read-only snapshot of the current visual object-belief state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisualBeliefExportV1 {
    /// Stable semantic identity, validated independently from the numeric version.
    pub schema_id: String,
    pub schema_version: u32,
    /// Vision-manifold frame identity at export time.
    pub frame_index: u64,
    /// Beliefs sorted strictly by local hypothesis ID for deterministic export.
    pub beliefs: Vec<VisualEntityBeliefV1>,
}

/// Structural validation failure for visual-belief export v1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisualBeliefExportError {
    UnsupportedSchemaIdentity { found: String },
    UnsupportedSchemaVersion { found: u32 },
    BeliefsNotStrictlyOrdered,
    LastObservationInFuture {
        hypothesis_id: u64,
        last_observed_frame: u64,
        frame_index: u64,
    },
    ObservedWithoutCurrentObservation { hypothesis_id: u64 },
    RetainedWithoutPriorObservation { hypothesis_id: u64 },
    OriginLifecycleMismatch { hypothesis_id: u64 },
    NonFiniteGridVelocity { hypothesis_id: u64 },
    ZeroTrackLength { hypothesis_id: u64 },
}

impl std::fmt::Display for VisualBeliefExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaIdentity { found } => {
                write!(f, "unsupported visual-belief export schema identity {found}")
            }
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported visual-belief export schema version {found}")
            }
            Self::BeliefsNotStrictlyOrdered => {
                write!(f, "visual beliefs must be strictly ordered by hypothesis_id")
            }
            Self::LastObservationInFuture {
                hypothesis_id,
                last_observed_frame,
                frame_index,
            } => write!(
                f,
                "visual belief {hypothesis_id} was last observed at frame {last_observed_frame} beyond export frame {frame_index}"
            ),
            Self::ObservedWithoutCurrentObservation { hypothesis_id } => write!(
                f,
                "observed visual belief {hypothesis_id} is not backed by the export frame"
            ),
            Self::RetainedWithoutPriorObservation { hypothesis_id } => write!(
                f,
                "retained visual belief {hypothesis_id} does not refer to an earlier observation"
            ),
            Self::OriginLifecycleMismatch { hypothesis_id } => write!(
                f,
                "visual belief {hypothesis_id} has incompatible lifecycle and evidence origin"
            ),
            Self::NonFiniteGridVelocity { hypothesis_id } => {
                write!(f, "visual belief {hypothesis_id} has non-finite grid velocity")
            }
            Self::ZeroTrackLength { hypothesis_id } => {
                write!(f, "visual belief {hypothesis_id} has zero track length")
            }
        }
    }
}

impl std::error::Error for VisualBeliefExportError {}

impl VisualBeliefExportV1 {
    /// Derive the strongest honest v1 belief snapshot from the current public
    /// Vision Manifold object-memory state.
    ///
    /// This method is read-only. It does not mutate tracking state, run a
    /// predictor, synthesize observation IDs, or derive benchmark/world IDs.
    pub fn from_manifold(manifold: &VisionManifold) -> Self {
        let frame_index = manifold.frame_count();
        let mut beliefs = manifold
            .object_memory()
            .map(|memory| {
                memory
                    .tracks()
                    .iter()
                    .map(|track| belief_from_track(track, frame_index))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        beliefs.sort_by_key(|belief| belief.hypothesis_id);

        Self {
            schema_id: VISUAL_BELIEF_EXPORT_SCHEMA_ID.to_owned(),
            schema_version: VISUAL_BELIEF_EXPORT_SCHEMA_VERSION,
            frame_index,
            beliefs,
        }
    }

    /// Validate the closed v1 semantic contract.
    pub fn validate(&self) -> Result<(), VisualBeliefExportError> {
        if self.schema_id != VISUAL_BELIEF_EXPORT_SCHEMA_ID {
            return Err(VisualBeliefExportError::UnsupportedSchemaIdentity {
                found: self.schema_id.clone(),
            });
        }
        if self.schema_version != VISUAL_BELIEF_EXPORT_SCHEMA_VERSION {
            return Err(VisualBeliefExportError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }

        if self
            .beliefs
            .windows(2)
            .any(|pair| pair[0].hypothesis_id >= pair[1].hypothesis_id)
        {
            return Err(VisualBeliefExportError::BeliefsNotStrictlyOrdered);
        }

        for belief in &self.beliefs {
            if belief.last_observed_frame > self.frame_index {
                return Err(VisualBeliefExportError::LastObservationInFuture {
                    hypothesis_id: belief.hypothesis_id,
                    last_observed_frame: belief.last_observed_frame,
                    frame_index: self.frame_index,
                });
            }
            if belief.track_length == 0 {
                return Err(VisualBeliefExportError::ZeroTrackLength {
                    hypothesis_id: belief.hypothesis_id,
                });
            }
            if !belief.kinematics.velocity_row.is_finite()
                || !belief.kinematics.velocity_col.is_finite()
            {
                return Err(VisualBeliefExportError::NonFiniteGridVelocity {
                    hypothesis_id: belief.hypothesis_id,
                });
            }

            match belief.lifecycle {
                VisualBeliefLifecycleV1::Observed => {
                    if belief.last_observed_frame != self.frame_index {
                        return Err(VisualBeliefExportError::ObservedWithoutCurrentObservation {
                            hypothesis_id: belief.hypothesis_id,
                        });
                    }
                    if belief.evidence_origin != VisualEvidenceOriginV1::Observed {
                        return Err(VisualBeliefExportError::OriginLifecycleMismatch {
                            hypothesis_id: belief.hypothesis_id,
                        });
                    }
                }
                VisualBeliefLifecycleV1::UnobservedRetained => {
                    if belief.last_observed_frame >= self.frame_index {
                        return Err(VisualBeliefExportError::RetainedWithoutPriorObservation {
                            hypothesis_id: belief.hypothesis_id,
                        });
                    }
                    if belief.evidence_origin != VisualEvidenceOriginV1::Remembered {
                        return Err(VisualBeliefExportError::OriginLifecycleMismatch {
                            hypothesis_id: belief.hypothesis_id,
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

fn belief_from_track(track: &TrackedObject, frame_index: u64) -> VisualEntityBeliefV1 {
    let currently_observed = track.last_seen_frame == frame_index;
    let (lifecycle, evidence_origin) = if currently_observed {
        (
            VisualBeliefLifecycleV1::Observed,
            VisualEvidenceOriginV1::Observed,
        )
    } else {
        (
            VisualBeliefLifecycleV1::UnobservedRetained,
            VisualEvidenceOriginV1::Remembered,
        )
    };

    VisualEntityBeliefV1 {
        hypothesis_id: track.track_id,
        lifecycle,
        last_observed_frame: track.last_seen_frame,
        evidence_origin,
        observation_lineage: VisualObservationLineageV1::Unavailable {
            reason: ObservationLineageUnavailableReasonV1::TrackerDoesNotRetainExternalObservationIdentity,
        },
        persistence_confidence: VisualBeliefConfidenceV1::Unavailable {
            reason: ConfidenceUnavailableReasonV1::NoQualifiedPersistenceConfidence,
        },
        kinematics: VisualGridKinematicsV1 {
            coordinate_frame: VisualCoordinateFrameV1::PatchGrid,
            centroid_row: track.centroid_row,
            centroid_col: track.centroid_col,
            velocity_row: track.velocity_row,
            velocity_col: track.velocity_col,
        },
        track_length: track.track_length,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::hdc::ContinuousHV;

    fn track(track_id: u64, last_seen_frame: u64) -> TrackedObject {
        TrackedObject {
            track_id,
            appearance_hv: ContinuousHV::zero(256),
            identity_hv: ContinuousHV::zero(256),
            centroid_row: 3,
            centroid_col: 7,
            velocity_row: 0.25,
            velocity_col: -0.5,
            last_seen_frame,
            track_length: 4,
        }
    }

    fn export(frame_index: u64, beliefs: Vec<VisualEntityBeliefV1>) -> VisualBeliefExportV1 {
        VisualBeliefExportV1 {
            schema_id: VISUAL_BELIEF_EXPORT_SCHEMA_ID.to_owned(),
            schema_version: VISUAL_BELIEF_EXPORT_SCHEMA_VERSION,
            frame_index,
            beliefs,
        }
    }

    #[test]
    fn current_track_exports_as_observed_without_fabricated_lineage_or_confidence() {
        let belief = belief_from_track(&track(7, 12), 12);
        assert_eq!(belief.lifecycle, VisualBeliefLifecycleV1::Observed);
        assert_eq!(belief.evidence_origin, VisualEvidenceOriginV1::Observed);
        assert_eq!(belief.last_observed_frame, 12);
        assert_eq!(
            belief.observation_lineage,
            VisualObservationLineageV1::Unavailable {
                reason: ObservationLineageUnavailableReasonV1::TrackerDoesNotRetainExternalObservationIdentity,
            }
        );
        assert_eq!(
            belief.persistence_confidence,
            VisualBeliefConfidenceV1::Unavailable {
                reason: ConfidenceUnavailableReasonV1::NoQualifiedPersistenceConfidence,
            }
        );
        assert_eq!(belief.kinematics.coordinate_frame, VisualCoordinateFrameV1::PatchGrid);
    }

    #[test]
    fn retained_track_is_not_promoted_to_prediction_or_observation() {
        let belief = belief_from_track(&track(11, 5), 8);
        assert_eq!(
            belief.lifecycle,
            VisualBeliefLifecycleV1::UnobservedRetained
        );
        assert_eq!(belief.evidence_origin, VisualEvidenceOriginV1::Remembered);
        assert_eq!(belief.last_observed_frame, 5);
    }

    #[test]
    fn validation_rejects_unknown_schema_identity() {
        let mut snapshot = export(3, vec![]);
        snapshot.schema_id = "symthaea.visual-belief-export.future".to_owned();
        assert!(matches!(
            snapshot.validate(),
            Err(VisualBeliefExportError::UnsupportedSchemaIdentity { .. })
        ));
    }

    #[test]
    fn validation_rejects_observed_belief_with_stale_observation_time() {
        let mut belief = belief_from_track(&track(1, 4), 4);
        belief.last_observed_frame = 3;
        let snapshot = export(4, vec![belief]);
        assert!(matches!(
            snapshot.validate(),
            Err(VisualBeliefExportError::ObservedWithoutCurrentObservation { hypothesis_id: 1 })
        ));
    }

    #[test]
    fn validation_rejects_retained_belief_that_claims_current_observation() {
        let mut belief = belief_from_track(&track(2, 4), 5);
        belief.last_observed_frame = 5;
        let snapshot = export(5, vec![belief]);
        assert!(matches!(
            snapshot.validate(),
            Err(VisualBeliefExportError::RetainedWithoutPriorObservation { hypothesis_id: 2 })
        ));
    }

    #[test]
    fn validation_rejects_origin_lifecycle_mismatch() {
        let mut belief = belief_from_track(&track(3, 5), 5);
        belief.evidence_origin = VisualEvidenceOriginV1::Remembered;
        let snapshot = export(5, vec![belief]);
        assert!(matches!(
            snapshot.validate(),
            Err(VisualBeliefExportError::OriginLifecycleMismatch { hypothesis_id: 3 })
        ));
    }

    #[test]
    fn validation_rejects_future_observation_time() {
        let mut belief = belief_from_track(&track(4, 6), 6);
        belief.last_observed_frame = 7;
        let snapshot = export(6, vec![belief]);
        assert!(matches!(
            snapshot.validate(),
            Err(VisualBeliefExportError::LastObservationInFuture {
                hypothesis_id: 4,
                ..
            })
        ));
    }

    #[test]
    fn validation_rejects_non_finite_grid_velocity() {
        let mut belief = belief_from_track(&track(5, 9), 9);
        belief.kinematics.velocity_row = f32::NAN;
        let snapshot = export(9, vec![belief]);
        assert!(matches!(
            snapshot.validate(),
            Err(VisualBeliefExportError::NonFiniteGridVelocity { hypothesis_id: 5 })
        ));
    }

    #[test]
    fn validation_requires_canonical_hypothesis_order() {
        let a = belief_from_track(&track(9, 3), 3);
        let b = belief_from_track(&track(2, 3), 3);
        let snapshot = export(3, vec![a, b]);
        assert_eq!(
            snapshot.validate(),
            Err(VisualBeliefExportError::BeliefsNotStrictlyOrdered)
        );
    }

    #[test]
    fn helper_export_is_deterministic_for_same_track_state() {
        let a = belief_from_track(&track(42, 17), 20);
        let b = belief_from_track(&track(42, 17), 20);
        assert_eq!(a, b);
    }

    #[test]
    fn validation_accepts_honest_observed_and_retained_states() {
        let observed = belief_from_track(&track(1, 10), 10);
        let retained = belief_from_track(&track(2, 8), 10);
        let snapshot = export(10, vec![observed, retained]);
        assert_eq!(snapshot.validate(), Ok(()));
    }

    #[test]
    fn manifold_without_object_memory_exports_empty_valid_snapshot() {
        let mut config = crate::types::VisionConfig::default();
        config.hdc_dim = 256;
        let manifold = VisionManifold::new(config, 16, 16);
        let snapshot = VisualBeliefExportV1::from_manifold(&manifold);
        assert_eq!(snapshot.schema_id, VISUAL_BELIEF_EXPORT_SCHEMA_ID);
        assert!(snapshot.beliefs.is_empty());
        assert_eq!(snapshot.validate(), Ok(()));
    }
}
