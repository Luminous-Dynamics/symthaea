// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Renderer-neutral, fail-closed contract for bounded vocal performance edits.
//!
//! This crate validates edit intent, base-artifact identity, dependency-cone
//! declarations and renderer receipts. It does not apply edits, rerender audio,
//! or prove pronunciation/music correctness. The renderer must derive the
//! dependency cone from a validated v3 graph; callers cannot declare an
//! arbitrary small window and thereby make a global edit look local.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const PATCH_SCHEMA_VERSION: u32 = 1;
const RECEIPT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditTargetKind {
    Performance,
    Word,
    Phone,
    Syllable,
    ScoreEvent,
    Phrase,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditTarget {
    pub kind: EditTargetKind,
    /// Stable graph identifier, not a vector index or display label.
    pub stable_id: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum VocalEditOperation {
    SelectPronunciationCandidate { candidate_id: String },
    ReplacePhone { new_ipa: String },
    SetScoreEventPitch { frequency_hz: f64 },
    SetScoreEventVelocity { velocity: f64 },
    SetMelismaNucleus { phone_index: usize },
    SetPhraseBreath { breath_before_s: f64, breath_after_s: f64 },
    ReplaceTempoMap { tempo_map_blake3: String },
    ReplaceVoiceArtifact { voice_artifact_blake3: String },
}

impl VocalEditOperation {
    fn expected_target_kind(&self) -> EditTargetKind {
        match self {
            Self::SelectPronunciationCandidate { .. } => EditTargetKind::Word,
            Self::ReplacePhone { .. } => EditTargetKind::Phone,
            Self::SetScoreEventPitch { .. } | Self::SetScoreEventVelocity { .. } => {
                EditTargetKind::ScoreEvent
            }
            Self::SetMelismaNucleus { .. } => EditTargetKind::Syllable,
            Self::SetPhraseBreath { .. } => EditTargetKind::Phrase,
            Self::ReplaceTempoMap { .. } | Self::ReplaceVoiceArtifact { .. } => {
                EditTargetKind::Performance
            }
        }
    }

    fn requires_whole_stem(&self) -> bool {
        matches!(self, Self::ReplaceTempoMap { .. } | Self::ReplaceVoiceArtifact { .. })
    }

    fn validate(&self) -> Result<(), String> {
        match self {
            Self::SelectPronunciationCandidate { candidate_id } => {
                require_id(candidate_id, "candidate_id")?;
            }
            Self::ReplacePhone { new_ipa } => {
                if new_ipa.trim().is_empty() {
                    return Err("replacement IPA phone cannot be empty".into());
                }
            }
            Self::SetScoreEventPitch { frequency_hz } => {
                if !frequency_hz.is_finite() || *frequency_hz <= 0.0 {
                    return Err("score-event frequency must be finite and positive".into());
                }
            }
            Self::SetScoreEventVelocity { velocity } => {
                if !velocity.is_finite() || !(0.0..=1.0).contains(velocity) {
                    return Err("score-event velocity must be finite and within [0, 1]".into());
                }
            }
            Self::SetMelismaNucleus { .. } => {}
            Self::SetPhraseBreath {
                breath_before_s,
                breath_after_s,
            } => {
                if !breath_before_s.is_finite()
                    || *breath_before_s < 0.0
                    || !breath_after_s.is_finite()
                    || *breath_after_s < 0.0
                {
                    return Err("phrase breath durations must be finite and non-negative".into());
                }
            }
            Self::ReplaceTempoMap { tempo_map_blake3 } => {
                validate_digest(tempo_map_blake3, "tempo_map_blake3")?;
            }
            Self::ReplaceVoiceArtifact {
                voice_artifact_blake3,
            } => {
                validate_digest(voice_artifact_blake3, "voice_artifact_blake3")?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EditAuthority {
    User,
    HumanReviewer,
    CommunityReviewer,
    ApprovedPolicy,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EditAuthorization {
    pub authority: EditAuthority,
    /// Opaque reference to an authorization, review, or user-action record.
    /// Do not store personal reviewer identity in this contract.
    pub evidence_reference: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactDigests {
    pub pronunciation_plan_blake3: String,
    pub score_alignment_blake3: String,
    pub rendered_audio_blake3: String,
    pub renderer_capability_blake3: String,
    pub voice_artifact_blake3: String,
}

impl ArtifactDigests {
    pub fn validate(&self) -> Result<(), String> {
        validate_digest(&self.pronunciation_plan_blake3, "pronunciation_plan_blake3")?;
        validate_digest(&self.score_alignment_blake3, "score_alignment_blake3")?;
        validate_digest(&self.rendered_audio_blake3, "rendered_audio_blake3")?;
        validate_digest(&self.renderer_capability_blake3, "renderer_capability_blake3")?;
        validate_digest(&self.voice_artifact_blake3, "voice_artifact_blake3")?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DependencyScope {
    Local,
    WholeStem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameWindow {
    /// Half-open sample-frame interval.
    pub start_frame: u64,
    pub end_frame: u64,
}

impl FrameWindow {
    pub fn validate(&self, total_frames: u64, label: &str) -> Result<(), String> {
        if self.start_frame >= self.end_frame || self.end_frame > total_frames {
            return Err(format!(
                "{label} must be a non-empty half-open interval within the audio"
            ));
        }
        Ok(())
    }

    pub fn contains(&self, other: &Self) -> bool {
        self.start_frame <= other.start_frame && self.end_frame >= other.end_frame
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DependencyCone {
    pub scope: DependencyScope,
    /// Stable graph nodes which must be invalidated and reevaluated.
    pub affected_nodes: Vec<EditTarget>,
    /// Stable graph nodes explicitly declared outside the dependency cone.
    pub protected_nodes: Vec<EditTarget>,
    /// Exact requested edit interval; absent for a global/whole-stem edit.
    pub target_window: Option<FrameWindow>,
    /// Rerender interval including coarticulation/crossfade context.
    pub render_window: Option<FrameWindow>,
    pub context_before_frames: u64,
    pub context_after_frames: u64,
}

impl DependencyCone {
    fn validate(
        &self,
        target: &EditTarget,
        total_frames: u64,
        requires_whole_stem: bool,
    ) -> Result<(), String> {
        validate_unique_targets(&self.affected_nodes, "affected_nodes")?;
        validate_unique_targets(&self.protected_nodes, "protected_nodes")?;

        if !self.affected_nodes.contains(target) {
            return Err("dependency cone does not include the patch target".into());
        }
        if self.affected_nodes.iter().any(|node| self.protected_nodes.contains(node)) {
            return Err("affected and protected dependency nodes overlap".into());
        }

        match self.scope {
            DependencyScope::Local => {
                if requires_whole_stem {
                    return Err("this operation requires whole-stem invalidation".into());
                }
                if self.protected_nodes.is_empty() {
                    return Err("a local edit must identify at least one protected node".into());
                }
                let target_window = self.target_window.as_ref()
                    .ok_or_else(|| "local edit requires a target window".to_string())?;
                let render_window = self.render_window.as_ref()
                    .ok_or_else(|| "local edit requires a render window".to_string())?;
                target_window.validate(total_frames, "target_window")?;
                render_window.validate(total_frames, "render_window")?;
                if !render_window.contains(target_window) {
                    return Err("render window must contain the target window".into());
                }
                if render_window.start_frame == 0 || render_window.end_frame == total_frames {
                    return Err("a local render window cannot cover the whole audio stem".into());
                }
                if target_window.start_frame - render_window.start_frame != self.context_before_frames
                    || render_window.end_frame - target_window.end_frame != self.context_after_frames
                {
                    return Err("declared context guard does not match target/render windows".into());
                }
            }
            DependencyScope::WholeStem => {
                if !self.protected_nodes.is_empty() {
                    return Err("whole-stem edits cannot claim protected nodes".into());
                }
                if self.target_window.is_some()
                    || self.render_window.is_some()
                    || self.context_before_frames != 0
                    || self.context_after_frames != 0
                {
                    return Err("whole-stem edits cannot claim a bounded render window".into());
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VocalEditPatch {
    pub schema_version: u32,
    pub patch_id: String,
    pub renderer_id: String,
    pub base: ArtifactDigests,
    pub target: EditTarget,
    pub operation: VocalEditOperation,
    pub dependency_cone: DependencyCone,
    pub sample_rate_hz: u32,
    pub total_frames: u64,
    pub authorization: EditAuthorization,
}

impl VocalEditPatch {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != PATCH_SCHEMA_VERSION {
            return Err(format!("unsupported vocal edit patch schema version {}", self.schema_version));
        }
        require_id(&self.patch_id, "patch_id")?;
        require_id(&self.renderer_id, "renderer_id")?;
        require_id(&self.target.stable_id, "target.stable_id")?;
        require_id(&self.authorization.evidence_reference, "authorization.evidence_reference")?;
        self.base.validate()?;
        self.operation.validate()?;
        if self.target.kind != self.operation.expected_target_kind() {
            return Err(format!(
                "operation requires target kind {:?}, got {:?}",
                self.operation.expected_target_kind(),
                self.target.kind
            ));
        }
        if self.sample_rate_hz == 0 || self.total_frames == 0 {
            return Err("sample rate and total frame count must be positive".into());
        }
        self.dependency_cone.validate(
            &self.target,
            self.total_frames,
            self.operation.requires_whole_stem(),
        )?;
        Ok(())
    }

    /// Re-authorize replay only when all source artifacts still match.
    pub fn verify_base(&self, current: &ArtifactDigests) -> Result<(), String> {
        self.validate()?;
        current.validate()?;
        if &self.base != current {
            return Err("stale edit patch: base plan/score/audio/renderer/voice digest mismatch".into());
        }
        Ok(())
    }

    /// Domain-separated digest over the exact, validated serialized patch.
    pub fn patch_blake3(&self) -> Result<String, String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)
            .map_err(|error| format!("failed to serialize vocal edit patch: {error}"))?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"luminous-vocal-edit-patch-v1\0");
        hasher.update(&(bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
        Ok(hasher.finalize().to_hex().to_string())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderScope {
    BoundedLocal,
    WholeStem,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreservedNodeEvidence {
    pub node: EditTarget,
    pub sample_window: FrameWindow,
    pub before_region_blake3: String,
    pub after_region_blake3: String,
    /// Exact f32 bit-pattern match fraction over this preserved region.
    pub exact_sample_fraction: f64,
    /// RMS difference in the input amplitude domain.
    pub rms_delta: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VocalEditReceipt {
    pub schema_version: u32,
    pub patch_blake3: String,
    pub base: ArtifactDigests,
    pub output_pronunciation_plan_blake3: String,
    pub output_score_alignment_blake3: String,
    pub output_audio_blake3: String,
    pub output_voice_artifact_blake3: String,
    pub renderer_id: String,
    pub render_scope: RenderScope,
    pub actual_render_window: Option<FrameWindow>,
    pub preserved_nodes: Vec<PreservedNodeEvidence>,
    pub locality_report_blake3: Option<String>,
    pub seam_report_blake3: Option<String>,
    pub seam_pass: Option<bool>,
}

impl VocalEditReceipt {
    /// Verify receipt linkage and preservation evidence against the original
    /// patch. This does not read waveform data or independently rescore VELB;
    /// it verifies the content-addressed report references and structural
    /// claims that an external verifier must then inspect.
    pub fn verify_against(&self, patch: &VocalEditPatch) -> Result<(), String> {
        patch.validate()?;
        if self.schema_version != RECEIPT_SCHEMA_VERSION {
            return Err(format!("unsupported vocal edit receipt schema version {}", self.schema_version));
        }
        validate_digest(&self.patch_blake3, "patch_blake3")?;
        validate_digest(&self.output_pronunciation_plan_blake3, "output_pronunciation_plan_blake3")?;
        validate_digest(&self.output_score_alignment_blake3, "output_score_alignment_blake3")?;
        validate_digest(&self.output_audio_blake3, "output_audio_blake3")?;
        validate_digest(&self.output_voice_artifact_blake3, "output_voice_artifact_blake3")?;
        require_id(&self.renderer_id, "renderer_id")?;
        if self.renderer_id != patch.renderer_id {
            return Err("receipt renderer identity differs from the requested renderer".into());
        }
        match &patch.operation {
            VocalEditOperation::ReplaceVoiceArtifact { voice_artifact_blake3 } => {
                if &self.output_voice_artifact_blake3 != voice_artifact_blake3 {
                    return Err("receipt output voice artifact does not match the requested replacement".into());
                }
            }
            _ => {
                if self.output_voice_artifact_blake3 != patch.base.voice_artifact_blake3 {
                    return Err("receipt changed voice artifact without a voice-replacement operation".into());
                }
            }
        }

        if self.patch_blake3 != patch.patch_blake3()? || self.base != patch.base {
            return Err("receipt does not bind to this exact patch and base artifact set".into());
        }

        match patch.dependency_cone.scope {
            DependencyScope::Local => {
                if self.render_scope != RenderScope::BoundedLocal {
                    return Err("local patch receipt claims a whole-stem rerender".into());
                }
                if self.actual_render_window != patch.dependency_cone.render_window {
                    return Err("receipt render window differs from declared dependency cone".into());
                }
                validate_unique_preserved(&self.preserved_nodes)?;
                let expected: BTreeSet<_> = patch.dependency_cone.protected_nodes.iter().cloned().collect();
                let actual: BTreeSet<_> = self.preserved_nodes.iter().map(|item| item.node.clone()).collect();
                if expected != actual {
                    return Err("receipt does not cover exactly the declared protected nodes".into());
                }
                let locality = self.locality_report_blake3.as_deref()
                    .ok_or_else(|| "local edit receipt requires a VELB locality report digest".to_string())?;
                let seam = self.seam_report_blake3.as_deref()
                    .ok_or_else(|| "local edit receipt requires a seam report digest".to_string())?;
                validate_digest(locality, "locality_report_blake3")?;
                validate_digest(seam, "seam_report_blake3")?;
                if self.seam_pass != Some(true) {
                    return Err("local edit receipt requires a passing seam report".into());
                }
                for evidence in &self.preserved_nodes {
                    validate_digest(&evidence.before_region_blake3, "before_region_blake3")?;
                    validate_digest(&evidence.after_region_blake3, "after_region_blake3")?;
                    evidence.sample_window.validate(patch.total_frames, "preserved sample window")?;
                    if evidence.sample_window.start_frame < self.actual_render_window.unwrap().end_frame
                        && evidence.sample_window.end_frame > self.actual_render_window.unwrap().start_frame
                    {
                        return Err("preserved sample window overlaps the rerender window".into());
                    }
                    if !evidence.exact_sample_fraction.is_finite()
                        || !(0.0..=1.0).contains(&evidence.exact_sample_fraction)
                        || !evidence.rms_delta.is_finite()
                        || evidence.rms_delta < 0.0
                    {
                        return Err("preserved-region metrics are invalid".into());
                    }
                    if evidence.exact_sample_fraction == 1.0
                        && evidence.before_region_blake3 != evidence.after_region_blake3
                    {
                        return Err("bit-exact preserved samples must have identical region digests".into());
                    }
                    if evidence.before_region_blake3 == evidence.after_region_blake3
                        && evidence.exact_sample_fraction != 1.0
                    {
                        return Err("identical region digests require an exact sample fraction of 1".into());
                    }
                }
            }
            DependencyScope::WholeStem => {
                if self.render_scope != RenderScope::WholeStem
                    || self.actual_render_window.is_some()
                    || !self.preserved_nodes.is_empty()
                {
                    return Err("whole-stem patch receipt makes an invalid locality claim".into());
                }
                if self.locality_report_blake3.is_some()
                    || self.seam_report_blake3.is_some()
                    || self.seam_pass.is_some()
                {
                    return Err("whole-stem receipt cannot claim a bounded-local VELB/seam report".into());
                }
            }
        }
        Ok(())
    }
}

fn validate_unique_targets(targets: &[EditTarget], label: &str) -> Result<(), String> {
    if targets.is_empty() {
        return Err(format!("{label} cannot be empty"));
    }
    let mut seen = BTreeSet::new();
    for target in targets {
        require_id(&target.stable_id, &format!("{label}.stable_id"))?;
        if !seen.insert(target) {
            return Err(format!("{label} contains duplicate graph nodes"));
        }
    }
    Ok(())
}

fn validate_unique_preserved(evidence: &[PreservedNodeEvidence]) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for item in evidence {
        require_id(&item.node.stable_id, "preserved_nodes.node.stable_id")?;
        if !seen.insert(&item.node) {
            return Err("receipt contains duplicate preserved graph nodes".into());
        }
    }
    Ok(())
}

fn require_id(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} cannot be empty"));
    }
    if value.len() > 512 {
        return Err(format!("{label} exceeds 512 bytes"));
    }
    Ok(())
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase 64-character BLAKE3 hex digest"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIGEST_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const DIGEST_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const DIGEST_C: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn base() -> ArtifactDigests {
        ArtifactDigests {
            pronunciation_plan_blake3: DIGEST_A.into(),
            score_alignment_blake3: DIGEST_B.into(),
            rendered_audio_blake3: DIGEST_C.into(),
            renderer_capability_blake3: DIGEST_A.into(),
            voice_artifact_blake3: DIGEST_B.into(),
        }
    }

    fn node(kind: EditTargetKind, id: &str) -> EditTarget {
        EditTarget { kind, stable_id: id.into() }
    }

    fn local_patch() -> VocalEditPatch {
        VocalEditPatch {
            schema_version: 1,
            patch_id: "edit-word-0042".into(),
            renderer_id: "formant-vocoder-resolved-plan-v1".into(),
            base: base(),
            target: node(EditTargetKind::Word, "word-0042"),
            operation: VocalEditOperation::SelectPronunciationCandidate {
                candidate_id: "candidate-en-us-2".into(),
            },
            dependency_cone: DependencyCone {
                scope: DependencyScope::Local,
                affected_nodes: vec![
                    node(EditTargetKind::Word, "word-0042"),
                    node(EditTargetKind::Syllable, "syllable-0042-0"),
                    node(EditTargetKind::Phone, "phone-0042-0-1"),
                ],
                protected_nodes: vec![
                    node(EditTargetKind::Word, "word-0043"),
                    node(EditTargetKind::Phrase, "phrase-0002"),
                ],
                target_window: Some(FrameWindow { start_frame: 100, end_frame: 200 }),
                render_window: Some(FrameWindow { start_frame: 80, end_frame: 220 }),
                context_before_frames: 20,
                context_after_frames: 20,
            },
            sample_rate_hz: 48_000,
            total_frames: 1_000,
            authorization: EditAuthorization {
                authority: EditAuthority::User,
                evidence_reference: "user-action:edit-0042".into(),
            },
        }
    }

    fn local_receipt(patch: &VocalEditPatch) -> VocalEditReceipt {
        VocalEditReceipt {
            schema_version: 1,
            patch_blake3: patch.patch_blake3().unwrap(),
            base: patch.base.clone(),
            output_pronunciation_plan_blake3: DIGEST_C.into(),
            output_score_alignment_blake3: DIGEST_A.into(),
            output_audio_blake3: DIGEST_B.into(),
            output_voice_artifact_blake3: patch.base.voice_artifact_blake3.clone(),
            renderer_id: "formant-vocoder-resolved-plan-v1".into(),
            render_scope: RenderScope::BoundedLocal,
            actual_render_window: patch.dependency_cone.render_window,
            preserved_nodes: patch.dependency_cone.protected_nodes.iter().cloned().enumerate().map(|(index, node)| {
                PreservedNodeEvidence {
                    node,
                    sample_window: FrameWindow {
                        start_frame: 300 + index as u64 * 100,
                        end_frame: 350 + index as u64 * 100,
                    },
                    before_region_blake3: DIGEST_A.into(),
                    after_region_blake3: DIGEST_A.into(),
                    exact_sample_fraction: 1.0,
                    rms_delta: 0.0,
                }
            }).collect(),
            locality_report_blake3: Some(DIGEST_C.into()),
            seam_report_blake3: Some(DIGEST_B.into()),
            seam_pass: Some(true),
        }
    }

    #[test]
    fn valid_local_patch_and_receipt_verify() {
        let patch = local_patch();
        patch.validate().unwrap();
        local_receipt(&patch).verify_against(&patch).unwrap();
    }

    #[test]
    fn stale_base_plan_score_audio_renderer_or_voice_is_rejected() {
        let patch = local_patch();
        for mutation in 0..5 {
            let mut stale = patch.base.clone();
            match mutation {
                0 => stale.pronunciation_plan_blake3 = DIGEST_C.into(),
                1 => stale.score_alignment_blake3 = DIGEST_C.into(),
                2 => stale.rendered_audio_blake3 = DIGEST_A.into(),
                3 => stale.renderer_capability_blake3 = DIGEST_C.into(),
                _ => stale.voice_artifact_blake3 = DIGEST_C.into(),
            }
            assert!(patch.verify_base(&stale).is_err());
        }
        patch.verify_base(&patch.base).unwrap();
    }

    #[test]
    fn local_cone_must_contain_target_and_not_overlap_protected_nodes() {
        let mut patch = local_patch();
        patch.dependency_cone.affected_nodes.clear();
        assert!(patch.validate().is_err());

        let mut patch = local_patch();
        patch.dependency_cone.affected_nodes.push(node(EditTargetKind::Phrase, "phrase-0002"));
        assert!(patch.validate().is_err());
    }

    #[test]
    fn local_window_and_context_guard_must_be_consistent() {
        let mut patch = local_patch();
        patch.dependency_cone.context_before_frames = 19;
        assert!(patch.validate().is_err());

        let mut patch = local_patch();
        patch.dependency_cone.render_window = Some(FrameWindow { start_frame: 110, end_frame: 220 });
        assert!(patch.validate().is_err());
    }

    #[test]
    fn operation_target_kind_and_values_are_checked() {
        let mut wrong_target = local_patch();
        wrong_target.target.kind = EditTargetKind::Phone;
        assert!(wrong_target.validate().is_err());

        let mut bad_pitch = local_patch();
        bad_pitch.target = node(EditTargetKind::ScoreEvent, "note-17");
        bad_pitch.operation = VocalEditOperation::SetScoreEventPitch { frequency_hz: f64::NAN };
        assert!(bad_pitch.validate().is_err());

        let mut bad_ipa = local_patch();
        bad_ipa.target = node(EditTargetKind::Phone, "phone-0042-0-1");
        bad_ipa.operation = VocalEditOperation::ReplacePhone { new_ipa: "  ".into() };
        assert!(bad_ipa.validate().is_err());
    }

    #[test]
    fn tempo_and_voice_identity_changes_require_whole_stem_scope() {
        let mut patch = local_patch();
        patch.target = node(EditTargetKind::Performance, "performance-001");
        patch.operation = VocalEditOperation::ReplaceTempoMap { tempo_map_blake3: DIGEST_C.into() };
        assert!(patch.validate().is_err());

        patch.dependency_cone = DependencyCone {
            scope: DependencyScope::WholeStem,
            affected_nodes: vec![node(EditTargetKind::Performance, "performance-001")],
            protected_nodes: vec![],
            target_window: None,
            render_window: None,
            context_before_frames: 0,
            context_after_frames: 0,
        };
        patch.validate().unwrap();
    }

    #[test]
    fn stale_patch_digest_or_missing_protected_evidence_invalidates_receipt() {
        let patch = local_patch();
        let mut receipt = local_receipt(&patch);
        receipt.patch_blake3 = DIGEST_C.into();
        assert!(receipt.verify_against(&patch).is_err());

        let mut receipt = local_receipt(&patch);
        receipt.preserved_nodes.pop();
        assert!(receipt.verify_against(&patch).is_err());

        let mut receipt = local_receipt(&patch);
        receipt.seam_pass = Some(false);
        assert!(receipt.verify_against(&patch).is_err());
    }

    #[test]
    fn local_receipt_cannot_claim_whole_stem() {
        let patch = local_patch();
        let mut receipt = local_receipt(&patch);
        receipt.render_scope = RenderScope::WholeStem;
        assert!(receipt.verify_against(&patch).is_err());
    }

    #[test]
    fn patch_digest_changes_with_operation_and_target() {
        let patch = local_patch();
        let digest = patch.patch_blake3().unwrap();
        let mut changed = patch.clone();
        changed.operation = VocalEditOperation::SelectPronunciationCandidate {
            candidate_id: "candidate-en-us-3".into(),
        };
        assert_ne!(digest, changed.patch_blake3().unwrap());

        let mut changed_target = patch;
        changed_target.target.stable_id = "word-9999".into();
        assert_ne!(digest, changed_target.patch_blake3().unwrap());
    }

    #[test]
    fn malformed_digests_and_duplicate_nodes_are_rejected() {
        let mut patch = local_patch();
        patch.base.rendered_audio_blake3 = "not-a-digest".into();
        assert!(patch.validate().is_err());

        let mut patch = local_patch();
        let duplicate = patch.dependency_cone.affected_nodes[0].clone();
        patch.dependency_cone.affected_nodes.push(duplicate);
        assert!(patch.validate().is_err());
    }
}
