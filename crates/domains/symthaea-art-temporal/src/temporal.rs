// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporal and cinematic contracts for host-neutral artistic worlds.
//!
//! This module adds time without turning artistic choice into a scalar objective.
//! It binds shots, render observations, and counterfactual previews to exact
//! revisions and exact frame coordinates so a real-time host cannot silently
//! compare evidence from different worlds or different moments.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{ArtifactRef, EntityId, IntentId, ProposalId, RevisionId};

pub const ART_TEMPORAL_SCHEMA_V1: &str = "symthaea.art-world.temporal.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameRate {
    pub numerator: u32,
    pub denominator: u32,
}

impl FrameRate {
    pub fn new(numerator: u32, denominator: u32) -> Result<Self, TemporalArtError> {
        if numerator == 0 || denominator == 0 {
            return Err(TemporalArtError::InvalidFrameRate);
        }
        Ok(Self {
            numerator,
            denominator,
        })
    }

    pub fn frames_per_second(self) -> f64 {
        f64::from(self.numerator) / f64::from(self.denominator)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StudioFrame(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameSpan {
    pub start: StudioFrame,
    pub end_exclusive: StudioFrame,
}

impl FrameSpan {
    pub fn new(start: u64, end_exclusive: u64) -> Result<Self, TemporalArtError> {
        if end_exclusive <= start {
            return Err(TemporalArtError::InvalidFrameSpan);
        }
        Ok(Self {
            start: StudioFrame(start),
            end_exclusive: StudioFrame(end_exclusive),
        })
    }

    pub fn contains(self, frame: StudioFrame) -> bool {
        frame >= self.start && frame < self.end_exclusive
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraPose {
    pub translation: [f64; 3],
    /// Quaternion in x, y, z, w order.
    pub rotation_xyzw: [f64; 4],
    pub vertical_fov_radians: f64,
}

impl CameraPose {
    pub fn validate(&self) -> Result<(), TemporalArtError> {
        let finite = self
            .translation
            .iter()
            .chain(self.rotation_xyzw.iter())
            .chain(std::iter::once(&self.vertical_fov_radians))
            .all(|value| value.is_finite());
        if !finite {
            return Err(TemporalArtError::NonFiniteCameraPose);
        }
        if self.vertical_fov_radians <= 0.0 || self.vertical_fov_radians >= std::f64::consts::PI {
            return Err(TemporalArtError::InvalidFieldOfView);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraKeyframe {
    pub frame: StudioFrame,
    pub pose: CameraPose,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ScheduledProposal {
    pub frame: StudioFrame,
    pub proposal_id: ProposalId,
}

/// A cinematic plan is descriptive and revision-bound. It does not itself move
/// a host camera or apply proposals.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShotPlan {
    pub schema: String,
    pub shot_id: String,
    pub base_revision: RevisionId,
    pub span: FrameSpan,
    pub frame_rate: FrameRate,
    pub camera_id: Option<EntityId>,
    pub camera_path: Vec<CameraKeyframe>,
    pub scheduled_proposals: Vec<ScheduledProposal>,
    pub intent_id: Option<IntentId>,
    pub notes: Vec<String>,
}

impl ShotPlan {
    pub fn validate(&self) -> Result<(), TemporalArtError> {
        if self.schema != ART_TEMPORAL_SCHEMA_V1 {
            return Err(TemporalArtError::SchemaMismatch(self.schema.clone()));
        }
        if self.shot_id.trim().is_empty() {
            return Err(TemporalArtError::EmptyShotId);
        }

        let mut previous = None;
        for keyframe in &self.camera_path {
            if !self.span.contains(keyframe.frame) {
                return Err(TemporalArtError::KeyframeOutsideShot(keyframe.frame));
            }
            keyframe.pose.validate()?;
            if previous.is_some_and(|frame| keyframe.frame <= frame) {
                return Err(TemporalArtError::NonMonotonicKeyframes);
            }
            previous = Some(keyframe.frame);
        }

        for scheduled in &self.scheduled_proposals {
            if !self.span.contains(scheduled.frame) {
                return Err(TemporalArtError::ProposalOutsideShot(scheduled.frame));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderPurpose {
    CommittedObservation,
    CounterfactualPreview,
    PortfolioFrame,
    Diagnostic,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderChannel {
    Color,
    Depth,
    Normals,
    ObjectId,
    Motion,
}

/// Request for a bounded render observation of one exact world revision at one
/// exact frame. Hosts may refuse requests that exceed their declared budget.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderObservationRequest {
    pub schema: String,
    pub sample_id: String,
    pub revision: RevisionId,
    pub frame: StudioFrame,
    pub camera_id: Option<EntityId>,
    pub width: u32,
    pub height: u32,
    pub purpose: RenderPurpose,
    pub channels: Vec<RenderChannel>,
}

impl RenderObservationRequest {
    pub fn validate(&self) -> Result<(), TemporalArtError> {
        if self.schema != ART_TEMPORAL_SCHEMA_V1 {
            return Err(TemporalArtError::SchemaMismatch(self.schema.clone()));
        }
        if self.sample_id.trim().is_empty() {
            return Err(TemporalArtError::EmptySampleId);
        }
        if self.width == 0 || self.height == 0 {
            return Err(TemporalArtError::InvalidResolution);
        }
        if self.channels.is_empty() {
            return Err(TemporalArtError::NoRenderChannels);
        }
        Ok(())
    }
}

/// Receipt for pixels/data returned by a host. The revision and frame are
/// repeated intentionally so consumers can verify temporal alignment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderObservationReceipt {
    pub request: RenderObservationRequest,
    pub observed_revision: RevisionId,
    pub observed_frame: StudioFrame,
    pub scene_content_hash: String,
    pub artifact: ArtifactRef,
}

impl RenderObservationReceipt {
    pub fn validate_alignment(&self) -> Result<(), TemporalArtError> {
        self.request.validate()?;
        if self.request.revision != self.observed_revision {
            return Err(TemporalArtError::RenderRevisionMismatch {
                requested: self.request.revision.clone(),
                observed: self.observed_revision.clone(),
            });
        }
        if self.request.frame != self.observed_frame {
            return Err(TemporalArtError::RenderFrameMismatch {
                requested: self.request.frame,
                observed: self.observed_frame,
            });
        }
        if self.scene_content_hash.trim().is_empty() {
            return Err(TemporalArtError::MissingSceneHash);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposalPreview {
    pub proposal_id: ProposalId,
    pub receipt: RenderObservationReceipt,
}

/// Baseline plus alternative proposal renders for the same revision/frame.
/// No aggregate score is present by design.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CounterfactualRenderSet {
    pub base_revision: RevisionId,
    pub frame: StudioFrame,
    /// The do-nothing / abstention observation.
    pub baseline: RenderObservationReceipt,
    pub candidates: Vec<ProposalPreview>,
}

impl CounterfactualRenderSet {
    pub fn validate_alignment(&self) -> Result<(), TemporalArtError> {
        self.baseline.validate_alignment()?;
        if self.baseline.observed_revision != self.base_revision
            || self.baseline.observed_frame != self.frame
        {
            return Err(TemporalArtError::CounterfactualAlignmentMismatch);
        }
        for candidate in &self.candidates {
            candidate.receipt.validate_alignment()?;
            if candidate.receipt.request.purpose != RenderPurpose::CounterfactualPreview
                || candidate.receipt.request.revision != self.base_revision
                || candidate.receipt.request.frame != self.frame
            {
                return Err(TemporalArtError::CounterfactualAlignmentMismatch);
            }
        }
        Ok(())
    }
}

/// Time-indexed observed consequence. Consumers may maintain many dimensions;
/// this type intentionally provides no weighted sum or universal preference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalConsequenceSample {
    pub frame: StudioFrame,
    pub dimension: String,
    pub observed_delta: Option<f64>,
    pub uncertainty: Option<f64>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum TemporalArtError {
    #[error("frame rate numerator and denominator must both be non-zero")]
    InvalidFrameRate,
    #[error("frame span must contain at least one frame")]
    InvalidFrameSpan,
    #[error("unsupported temporal art schema: {0}")]
    SchemaMismatch(String),
    #[error("shot id may not be empty")]
    EmptyShotId,
    #[error("sample id may not be empty")]
    EmptySampleId,
    #[error("camera pose contains non-finite values")]
    NonFiniteCameraPose,
    #[error("camera vertical field of view must be inside (0, pi)")]
    InvalidFieldOfView,
    #[error("camera keyframe {0:?} lies outside the shot")]
    KeyframeOutsideShot(StudioFrame),
    #[error("camera keyframes must be strictly increasing")]
    NonMonotonicKeyframes,
    #[error("scheduled proposal {0:?} lies outside the shot")]
    ProposalOutsideShot(StudioFrame),
    #[error("render resolution must be non-zero")]
    InvalidResolution,
    #[error("at least one render channel is required")]
    NoRenderChannels,
    #[error("render revision mismatch: requested {requested:?}, observed {observed:?}")]
    RenderRevisionMismatch {
        requested: RevisionId,
        observed: RevisionId,
    },
    #[error("render frame mismatch: requested {requested:?}, observed {observed:?}")]
    RenderFrameMismatch {
        requested: StudioFrame,
        observed: StudioFrame,
    },
    #[error("render receipt is missing the scene content hash")]
    MissingSceneHash,
    #[error("counterfactual renders are not aligned to one revision/frame")]
    CounterfactualAlignmentMismatch,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn artifact() -> ArtifactRef {
        ArtifactRef {
            media_type: "image/png".into(),
            locator: "memory://frame".into(),
            digest: Some("abc".into()),
        }
    }

    fn request(revision: &str, frame: u64, purpose: RenderPurpose) -> RenderObservationRequest {
        RenderObservationRequest {
            schema: ART_TEMPORAL_SCHEMA_V1.into(),
            sample_id: format!("sample-{frame}"),
            revision: RevisionId::from(revision),
            frame: StudioFrame(frame),
            camera_id: None,
            width: 320,
            height: 180,
            purpose,
            channels: vec![RenderChannel::Color],
        }
    }

    #[test]
    fn rejects_zero_frame_rate() {
        assert_eq!(FrameRate::new(0, 1), Err(TemporalArtError::InvalidFrameRate));
    }

    #[test]
    fn shot_keyframes_are_revision_bound_and_monotonic() {
        let pose = CameraPose {
            translation: [0.0, 0.0, 3.0],
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            vertical_fov_radians: 1.0,
        };
        let shot = ShotPlan {
            schema: ART_TEMPORAL_SCHEMA_V1.into(),
            shot_id: "shot-1".into(),
            base_revision: RevisionId::from("r1"),
            span: FrameSpan::new(10, 20).unwrap(),
            frame_rate: FrameRate::new(24, 1).unwrap(),
            camera_id: None,
            camera_path: vec![
                CameraKeyframe {
                    frame: StudioFrame(10),
                    pose: pose.clone(),
                },
                CameraKeyframe {
                    frame: StudioFrame(19),
                    pose,
                },
            ],
            scheduled_proposals: Vec::new(),
            intent_id: None,
            notes: Vec::new(),
        };
        assert!(shot.validate().is_ok());
    }

    #[test]
    fn render_receipt_rejects_cross_revision_pixels() {
        let receipt = RenderObservationReceipt {
            request: request("r1", 4, RenderPurpose::CommittedObservation),
            observed_revision: RevisionId::from("r2"),
            observed_frame: StudioFrame(4),
            scene_content_hash: "scene".into(),
            artifact: artifact(),
        };
        assert!(matches!(
            receipt.validate_alignment(),
            Err(TemporalArtError::RenderRevisionMismatch { .. })
        ));
    }

    #[test]
    fn counterfactual_set_preserves_do_nothing_baseline() {
        let baseline = RenderObservationReceipt {
            request: request("r1", 4, RenderPurpose::CommittedObservation),
            observed_revision: RevisionId::from("r1"),
            observed_frame: StudioFrame(4),
            scene_content_hash: "scene".into(),
            artifact: artifact(),
        };
        let candidate = RenderObservationReceipt {
            request: request("r1", 4, RenderPurpose::CounterfactualPreview),
            observed_revision: RevisionId::from("r1"),
            observed_frame: StudioFrame(4),
            scene_content_hash: "preview".into(),
            artifact: artifact(),
        };
        let set = CounterfactualRenderSet {
            base_revision: RevisionId::from("r1"),
            frame: StudioFrame(4),
            baseline,
            candidates: vec![ProposalPreview {
                proposal_id: ProposalId::from("p1"),
                receipt: candidate,
            }],
        };
        assert!(set.validate_alignment().is_ok());
    }
}
