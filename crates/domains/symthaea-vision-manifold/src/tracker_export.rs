// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only export of the Vision Manifold's current object-tracker memory.
//!
//! This is intentionally a tracker-state contract, not an epistemic belief
//! contract. It exports only state the current `ObjectMemory` directly owns:
//! local track identity, last-seen frame, bounded-retention status, patch-grid
//! kinematics, and track length.
//!
//! It does **not** infer or export:
//!
//! - semantic/entity identity;
//! - observation-evidence lineage;
//! - prediction provenance;
//! - persistence confidence/probability;
//! - metric camera/world pose;
//! - physical velocity;
//! - simulator/canonical object identity.
//!
//! Those semantics belong to later provenance/belief layers and must not be
//! reconstructed from tracker state by an evaluator adapter.

use serde::{Deserialize, Serialize};

use crate::manifold::{TrackedObject, VisionManifold};

/// Stable semantic identity for [`VisualTrackerStateExportV1`].
pub const VISUAL_TRACKER_STATE_EXPORT_SCHEMA_ID: &str =
    "symthaea.visual-tracker-state-export.v1";
/// Numeric schema version for [`VisualTrackerStateExportV1`].
pub const VISUAL_TRACKER_STATE_EXPORT_SCHEMA_VERSION: u32 = 1;

/// Whether the tracker refreshed a track on the export frame or merely retained it.
///
/// `RetainedUnobserved` is deliberately not called predicted, occluded, remembered,
/// or persistent. Those are stronger propositions than raw `ObjectMemory` retention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualTrackPresenceV1 {
    /// `last_seen_frame == export.frame_index`.
    ObservedCurrentFrame,
    /// The track remains in bounded tracker memory but was not refreshed this frame.
    RetainedUnobserved,
}

/// Coordinate frame of the kinematics carried by tracker export v1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualTrackerCoordinateFrameV1 {
    /// Encoder patch-grid coordinates, not camera/world metric coordinates.
    PatchGrid,
}

/// Kinematics retained by the current object tracker.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisualTrackerKinematicsV1 {
    pub coordinate_frame: VisualTrackerCoordinateFrameV1,
    pub centroid_row: usize,
    pub centroid_col: usize,
    /// Smoothed row velocity in patch-grid cells per observed frame.
    pub velocity_row: f32,
    /// Smoothed column velocity in patch-grid cells per observed frame.
    pub velocity_col: f32,
}

/// One Symthaea-local tracker state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisualTrackStateV1 {
    /// Tracker-local identity only. This is not an entity/world/canonical ID.
    pub track_id: u64,
    pub presence: VisualTrackPresenceV1,
    /// Frame on which this track was last actually refreshed by object matching.
    pub last_seen_frame: u64,
    pub kinematics: VisualTrackerKinematicsV1,
    /// Number of observation refreshes accumulated by this track.
    pub track_length: u64,
}

/// Deterministic read-only snapshot of current object-tracker memory.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VisualTrackerStateExportV1 {
    pub schema_id: String,
    pub schema_version: u32,
    /// Vision-manifold frame identity at export time.
    pub frame_index: u64,
    /// Tracks sorted strictly by local tracker ID.
    pub tracks: Vec<VisualTrackStateV1>,
}

/// Structural validation failure for tracker-state export v1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VisualTrackerStateExportError {
    UnsupportedSchemaIdentity { found: String },
    UnsupportedSchemaVersion { found: u32 },
    TracksNotStrictlyOrdered,
    LastSeenInFuture {
        track_id: u64,
        last_seen_frame: u64,
        frame_index: u64,
    },
    CurrentPresenceWithoutCurrentRefresh { track_id: u64 },
    RetainedPresenceWithoutEarlierRefresh { track_id: u64 },
    NonFiniteGridVelocity { track_id: u64 },
    ZeroTrackLength { track_id: u64 },
}

impl std::fmt::Display for VisualTrackerStateExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedSchemaIdentity { found } => {
                write!(f, "unsupported visual tracker-state schema identity {found}")
            }
            Self::UnsupportedSchemaVersion { found } => {
                write!(f, "unsupported visual tracker-state schema version {found}")
            }
            Self::TracksNotStrictlyOrdered => {
                write!(f, "visual tracks must be strictly ordered by track_id")
            }
            Self::LastSeenInFuture {
                track_id,
                last_seen_frame,
                frame_index,
            } => write!(
                f,
                "visual track {track_id} was last seen at frame {last_seen_frame} beyond export frame {frame_index}"
            ),
            Self::CurrentPresenceWithoutCurrentRefresh { track_id } => write!(
                f,
                "current visual track {track_id} was not refreshed on the export frame"
            ),
            Self::RetainedPresenceWithoutEarlierRefresh { track_id } => write!(
                f,
                "retained visual track {track_id} does not refer to an earlier refresh"
            ),
            Self::NonFiniteGridVelocity { track_id } => {
                write!(f, "visual track {track_id} has non-finite patch-grid velocity")
            }
            Self::ZeroTrackLength { track_id } => {
                write!(f, "visual track {track_id} has zero track length")
            }
        }
    }
}

impl std::error::Error for VisualTrackerStateExportError {}

impl VisualTrackerStateExportV1 {
    /// Derive an exact tracker-memory snapshot through existing read-only public APIs.
    ///
    /// No tracker update, predictor, semantic model, evidence conversion, or external
    /// identity lookup is performed by this function.
    pub fn from_manifold(manifold: &VisionManifold) -> Self {
        let frame_index = manifold.frame_count();
        let mut tracks = manifold
            .object_memory()
            .map(|memory| {
                memory
                    .tracks()
                    .iter()
                    .map(|track| state_from_track(track, frame_index))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        tracks.sort_by_key(|track| track.track_id);

        Self {
            schema_id: VISUAL_TRACKER_STATE_EXPORT_SCHEMA_ID.to_owned(),
            schema_version: VISUAL_TRACKER_STATE_EXPORT_SCHEMA_VERSION,
            frame_index,
            tracks,
        }
    }

    /// Validate the closed v1 tracker-state contract.
    pub fn validate(&self) -> Result<(), VisualTrackerStateExportError> {
        if self.schema_id != VISUAL_TRACKER_STATE_EXPORT_SCHEMA_ID {
            return Err(VisualTrackerStateExportError::UnsupportedSchemaIdentity {
                found: self.schema_id.clone(),
            });
        }
        if self.schema_version != VISUAL_TRACKER_STATE_EXPORT_SCHEMA_VERSION {
            return Err(VisualTrackerStateExportError::UnsupportedSchemaVersion {
                found: self.schema_version,
            });
        }
        if self
            .tracks
            .windows(2)
            .any(|pair| pair[0].track_id >= pair[1].track_id)
        {
            return Err(VisualTrackerStateExportError::TracksNotStrictlyOrdered);
        }

        for track in &self.tracks {
            if track.last_seen_frame > self.frame_index {
                return Err(VisualTrackerStateExportError::LastSeenInFuture {
                    track_id: track.track_id,
                    last_seen_frame: track.last_seen_frame,
                    frame_index: self.frame_index,
                });
            }
            if track.track_length == 0 {
                return Err(VisualTrackerStateExportError::ZeroTrackLength {
                    track_id: track.track_id,
                });
            }
            if !track.kinematics.velocity_row.is_finite()
                || !track.kinematics.velocity_col.is_finite()
            {
                return Err(VisualTrackerStateExportError::NonFiniteGridVelocity {
                    track_id: track.track_id,
                });
            }

            match track.presence {
                VisualTrackPresenceV1::ObservedCurrentFrame => {
                    if track.last_seen_frame != self.frame_index {
                        return Err(
                            VisualTrackerStateExportError::CurrentPresenceWithoutCurrentRefresh {
                                track_id: track.track_id,
                            },
                        );
                    }
                }
                VisualTrackPresenceV1::RetainedUnobserved => {
                    if track.last_seen_frame >= self.frame_index {
                        return Err(
                            VisualTrackerStateExportError::RetainedPresenceWithoutEarlierRefresh {
                                track_id: track.track_id,
                            },
                        );
                    }
                }
            }
        }

        Ok(())
    }
}

fn state_from_track(track: &TrackedObject, frame_index: u64) -> VisualTrackStateV1 {
    let presence = if track.last_seen_frame == frame_index {
        VisualTrackPresenceV1::ObservedCurrentFrame
    } else {
        VisualTrackPresenceV1::RetainedUnobserved
    };

    VisualTrackStateV1 {
        track_id: track.track_id,
        presence,
        last_seen_frame: track.last_seen_frame,
        kinematics: VisualTrackerKinematicsV1 {
            coordinate_frame: VisualTrackerCoordinateFrameV1::PatchGrid,
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

    fn export(frame_index: u64, tracks: Vec<VisualTrackStateV1>) -> VisualTrackerStateExportV1 {
        VisualTrackerStateExportV1 {
            schema_id: VISUAL_TRACKER_STATE_EXPORT_SCHEMA_ID.to_owned(),
            schema_version: VISUAL_TRACKER_STATE_EXPORT_SCHEMA_VERSION,
            frame_index,
            tracks,
        }
    }

    #[test]
    fn current_track_exports_only_current_tracker_presence() {
        let state = state_from_track(&track(7, 12), 12);
        assert_eq!(state.presence, VisualTrackPresenceV1::ObservedCurrentFrame);
        assert_eq!(state.last_seen_frame, 12);
        assert_eq!(
            state.kinematics.coordinate_frame,
            VisualTrackerCoordinateFrameV1::PatchGrid
        );
    }

    #[test]
    fn retained_track_is_not_promoted_to_prediction_or_belief_origin() {
        let state = state_from_track(&track(11, 5), 8);
        assert_eq!(state.presence, VisualTrackPresenceV1::RetainedUnobserved);
        assert_eq!(state.last_seen_frame, 5);
    }

    #[test]
    fn validation_rejects_unknown_schema_identity() {
        let mut snapshot = export(3, vec![]);
        snapshot.schema_id = "symthaea.visual-tracker-state-export.future".to_owned();
        assert!(matches!(
            snapshot.validate(),
            Err(VisualTrackerStateExportError::UnsupportedSchemaIdentity { .. })
        ));
    }

    #[test]
    fn validation_rejects_current_presence_with_stale_refresh() {
        let mut state = state_from_track(&track(1, 4), 4);
        state.last_seen_frame = 3;
        let snapshot = export(4, vec![state]);
        assert!(matches!(
            snapshot.validate(),
            Err(VisualTrackerStateExportError::CurrentPresenceWithoutCurrentRefresh {
                track_id: 1
            })
        ));
    }

    #[test]
    fn validation_rejects_retained_presence_that_claims_current_refresh() {
        let mut state = state_from_track(&track(2, 4), 5);
        state.last_seen_frame = 5;
        let snapshot = export(5, vec![state]);
        assert!(matches!(
            snapshot.validate(),
            Err(VisualTrackerStateExportError::RetainedPresenceWithoutEarlierRefresh {
                track_id: 2
            })
        ));
    }

    #[test]
    fn validation_rejects_future_last_seen_frame() {
        let mut state = state_from_track(&track(4, 6), 6);
        state.last_seen_frame = 7;
        let snapshot = export(6, vec![state]);
        assert!(matches!(
            snapshot.validate(),
            Err(VisualTrackerStateExportError::LastSeenInFuture { track_id: 4, .. })
        ));
    }

    #[test]
    fn validation_rejects_non_finite_grid_velocity() {
        let mut state = state_from_track(&track(5, 9), 9);
        state.kinematics.velocity_row = f32::NAN;
        let snapshot = export(9, vec![state]);
        assert!(matches!(
            snapshot.validate(),
            Err(VisualTrackerStateExportError::NonFiniteGridVelocity { track_id: 5 })
        ));
    }

    #[test]
    fn validation_requires_canonical_track_order() {
        let a = state_from_track(&track(9, 3), 3);
        let b = state_from_track(&track(2, 3), 3);
        let snapshot = export(3, vec![a, b]);
        assert_eq!(
            snapshot.validate(),
            Err(VisualTrackerStateExportError::TracksNotStrictlyOrdered)
        );
    }

    #[test]
    fn tracker_export_is_deterministic_for_same_track_state() {
        let a = state_from_track(&track(42, 17), 20);
        let b = state_from_track(&track(42, 17), 20);
        assert_eq!(a, b);
    }

    #[test]
    fn validation_accepts_current_and_retained_tracker_states() {
        let current = state_from_track(&track(1, 10), 10);
        let retained = state_from_track(&track(2, 8), 10);
        let snapshot = export(10, vec![current, retained]);
        assert_eq!(snapshot.validate(), Ok(()));
    }

    #[test]
    fn manifold_without_object_memory_exports_empty_valid_snapshot() {
        let mut config = crate::types::VisionConfig::default();
        config.hdc_dim = 256;
        let manifold = VisionManifold::new(config, 16, 16);
        let snapshot = VisualTrackerStateExportV1::from_manifold(&manifold);
        assert_eq!(snapshot.schema_id, VISUAL_TRACKER_STATE_EXPORT_SCHEMA_ID);
        assert!(snapshot.tracks.is_empty());
        assert_eq!(snapshot.validate(), Ok(()));
    }
}
