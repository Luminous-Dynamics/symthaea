// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fidelity-aware render evidence shared by all artistic hosts.
//!
//! Render fidelity is provenance. Preview, cognitive-observation, and
//! portfolio-quality renders can all be useful without being treated as the
//! same measurement plane.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use thiserror::Error;

use crate::{
    RenderObservationReceipt, RenderPurpose, StudioFrame, TemporalArtError,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RenderFidelityClass {
    InteractivePreview,
    CognitiveObservation,
    Portfolio,
    Diagnostic,
    Custom(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderFidelity {
    pub class: RenderFidelityClass,
    pub width: u32,
    pub height: u32,
    pub samples_per_pixel: Option<u32>,
    pub profile: String,
}

impl RenderFidelity {
    pub fn validate(&self) -> Result<(), FidelityError> {
        if self.width == 0 || self.height == 0 {
            return Err(FidelityError::InvalidResolution);
        }
        if self.profile.trim().is_empty() {
            return Err(FidelityError::EmptyProfile);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FidelityTaggedRenderReceipt {
    pub receipt: RenderObservationReceipt,
    pub fidelity: RenderFidelity,
    /// Optional stable camera identity when a host supports multiple views.
    pub camera_stable_id: Option<String>,
}

impl FidelityTaggedRenderReceipt {
    pub fn validate(&self) -> Result<(), FidelityError> {
        self.receipt
            .validate_alignment()
            .map_err(FidelityError::Temporal)?;
        self.fidelity.validate()?;
        if self.receipt.request.width != self.fidelity.width
            || self.receipt.request.height != self.fidelity.height
        {
            return Err(FidelityError::ResolutionMismatch);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlignedCounterfactualRenderSet {
    pub baseline: FidelityTaggedRenderReceipt,
    pub candidates: Vec<FidelityTaggedRenderReceipt>,
}

impl AlignedCounterfactualRenderSet {
    pub fn validate(&self) -> Result<(), FidelityError> {
        self.baseline.validate()?;
        if self.baseline.receipt.request.purpose != RenderPurpose::CommittedObservation {
            return Err(FidelityError::BaselinePurposeMismatch);
        }
        for candidate in &self.candidates {
            candidate.validate()?;
            if candidate.receipt.request.purpose != RenderPurpose::CounterfactualPreview {
                return Err(FidelityError::CandidatePurposeMismatch);
            }
            same_plane(&self.baseline, candidate)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SynchronizedRenderSet {
    pub revision_id: String,
    pub frame: StudioFrame,
    pub views: Vec<FidelityTaggedRenderReceipt>,
}

impl SynchronizedRenderSet {
    pub fn validate(&self) -> Result<(), FidelityError> {
        if self.views.is_empty() {
            return Err(FidelityError::EmptyViewSet);
        }
        let fidelity = self.views[0].fidelity.clone();
        let scene_hash = self.views[0].receipt.scene_content_hash.clone();
        let mut cameras = BTreeSet::new();
        for view in &self.views {
            view.validate()?;
            if view.receipt.observed_revision.0 != self.revision_id
                || view.receipt.observed_frame != self.frame
                || view.receipt.scene_content_hash != scene_hash
                || view.fidelity != fidelity
            {
                return Err(FidelityError::CrossPlaneViewSet);
            }
            let camera = view
                .camera_stable_id
                .as_ref()
                .ok_or(FidelityError::MissingCameraIdentity)?;
            if !cameras.insert(camera.as_str()) {
                return Err(FidelityError::DuplicateCameraIdentity(camera.clone()));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalRenderWindow {
    pub camera_stable_id: String,
    pub samples: Vec<FidelityTaggedRenderReceipt>,
}

impl TemporalRenderWindow {
    pub fn validate(&self) -> Result<(), FidelityError> {
        if self.camera_stable_id.trim().is_empty() {
            return Err(FidelityError::MissingCameraIdentity);
        }
        if self.samples.is_empty() {
            return Err(FidelityError::EmptyTemporalWindow);
        }
        let fidelity = self.samples[0].fidelity.clone();
        let mut previous = None;
        for sample in &self.samples {
            sample.validate()?;
            if sample.fidelity != fidelity {
                return Err(FidelityError::MixedFidelityWindow);
            }
            if sample.camera_stable_id.as_deref() != Some(self.camera_stable_id.as_str()) {
                return Err(FidelityError::CrossCameraWindow);
            }
            if previous.is_some_and(|frame| sample.receipt.observed_frame <= frame) {
                return Err(FidelityError::NonMonotonicFrames);
            }
            previous = Some(sample.receipt.observed_frame);
        }
        Ok(())
    }
}

fn same_plane(
    left: &FidelityTaggedRenderReceipt,
    right: &FidelityTaggedRenderReceipt,
) -> Result<(), FidelityError> {
    if left.receipt.observed_revision != right.receipt.observed_revision
        || left.receipt.observed_frame != right.receipt.observed_frame
        || left.receipt.scene_content_hash != right.receipt.scene_content_hash
    {
        return Err(FidelityError::CounterfactualAlignmentMismatch);
    }
    if left.fidelity != right.fidelity {
        return Err(FidelityError::CounterfactualFidelityMismatch);
    }
    Ok(())
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum FidelityError {
    #[error(transparent)]
    Temporal(TemporalArtError),
    #[error("render fidelity resolution must be non-zero")]
    InvalidResolution,
    #[error("render fidelity profile may not be empty")]
    EmptyProfile,
    #[error("render receipt resolution differs from fidelity receipt")]
    ResolutionMismatch,
    #[error("counterfactual baseline must be a committed observation")]
    BaselinePurposeMismatch,
    #[error("counterfactual candidate must be a preview observation")]
    CandidatePurposeMismatch,
    #[error("counterfactual renders differ in revision/frame/scene")]
    CounterfactualAlignmentMismatch,
    #[error("counterfactual renders differ in render fidelity")]
    CounterfactualFidelityMismatch,
    #[error("synchronized render set may not be empty")]
    EmptyViewSet,
    #[error("synchronized renders do not share one observation plane")]
    CrossPlaneViewSet,
    #[error("stable camera identity is required")]
    MissingCameraIdentity,
    #[error("duplicate stable camera identity: {0}")]
    DuplicateCameraIdentity(String),
    #[error("temporal render window may not be empty")]
    EmptyTemporalWindow,
    #[error("temporal render window mixes fidelity")]
    MixedFidelityWindow,
    #[error("temporal render window mixes camera identity")]
    CrossCameraWindow,
    #[error("temporal render frames must be strictly increasing")]
    NonMonotonicFrames,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ART_TEMPORAL_SCHEMA_V1, ArtifactRef, RenderChannel,
        RenderObservationRequest, RevisionId,
    };

    fn fidelity(profile: &str) -> RenderFidelity {
        RenderFidelity {
            class: RenderFidelityClass::CognitiveObservation,
            width: 320,
            height: 180,
            samples_per_pixel: None,
            profile: profile.into(),
        }
    }

    fn tagged(
        id: &str,
        frame: u64,
        camera: &str,
        purpose: RenderPurpose,
        fidelity: RenderFidelity,
    ) -> FidelityTaggedRenderReceipt {
        let request = RenderObservationRequest {
            schema: ART_TEMPORAL_SCHEMA_V1.into(),
            sample_id: id.into(),
            revision: RevisionId::from("r1"),
            frame: StudioFrame(frame),
            camera_id: None,
            width: fidelity.width,
            height: fidelity.height,
            purpose,
            channels: vec![RenderChannel::Color],
        };
        FidelityTaggedRenderReceipt {
            receipt: RenderObservationReceipt {
                request,
                observed_revision: RevisionId::from("r1"),
                observed_frame: StudioFrame(frame),
                scene_content_hash: "scene".into(),
                artifact: ArtifactRef {
                    media_type: "image/png".into(),
                    locator: format!("memory://{id}"),
                    digest: Some(format!("digest-{id}")),
                },
            },
            fidelity,
            camera_stable_id: Some(camera.into()),
        }
    }

    #[test]
    fn counterfactual_render_set_rejects_mixed_fidelity() {
        let set = AlignedCounterfactualRenderSet {
            baseline: tagged(
                "base",
                1,
                "camera",
                RenderPurpose::CommittedObservation,
                fidelity("cognitive"),
            ),
            candidates: vec![tagged(
                "candidate",
                1,
                "camera",
                RenderPurpose::CounterfactualPreview,
                fidelity("portfolio"),
            )],
        };
        assert_eq!(
            set.validate(),
            Err(FidelityError::CounterfactualFidelityMismatch)
        );
    }

    #[test]
    fn synchronized_views_require_unique_camera_ids() {
        let f = fidelity("cognitive");
        let set = SynchronizedRenderSet {
            revision_id: "r1".into(),
            frame: StudioFrame(1),
            views: vec![
                tagged("a", 1, "camera", RenderPurpose::CommittedObservation, f.clone()),
                tagged("b", 1, "camera", RenderPurpose::CommittedObservation, f),
            ],
        };
        assert!(matches!(
            set.validate(),
            Err(FidelityError::DuplicateCameraIdentity(_))
        ));
    }
}
