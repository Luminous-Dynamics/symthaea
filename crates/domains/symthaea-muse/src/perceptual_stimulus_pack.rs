// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1C: deterministic stimulus-pack evidence contract for the first
//! blinded perceptual study.
//!
//! P1C binds the exact audio subjects presented later to human listeners. It
//! does not create a participant schedule, reveal arm identity to listeners,
//! collect responses, or authorize a perceptual claim.
//!
//! C6F commits acoustic feature panels but not full waveform hashes. P1C
//! therefore introduces an explicit render-subject bridge from the exact C6F
//! source/bundle identity to per-seed baseline/intervention waveform SHA-256s.
//! Human stimulus sources must match that bridge exactly.
//!
//! The only allowed signal transformation is constant negative gain applied to
//! the louder arm of a same-item pair. The quieter arm must remain bit-identical
//! to its source. No EQ, compression, limiting, resampling, trimming, fading,
//! channel remixing, or post-hoc excerpt selection is permitted.

use crate::evidence_digest::{
    canonical_json_sha256,
    perceptual_study_protocol::{
        FrozenPerceptualStudyProtocolV1, LoudnessMatchingV1, MEL003_FIXED_SEEDS,
        StimulusExtentV1,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const C6F_RENDER_SUBJECT_BINDING_VERSION: &str = "mel003-c6f-render-subject-binding-v1";
pub const PERCEPTUAL_STIMULUS_PACK_VERSION: &str = "mel003-perceptual-stimulus-pack-v1";
pub const LOUDNESS_MEASUREMENT_PROFILE_V1: &str = "itu-r-bs1770-k-weighted-integrated-v1";
pub const GAIN_TRANSFORM_PROFILE_V1: &str = "constant-gain-attenuation-only-v1";
pub const REQUIRED_SAMPLE_RATE_HZ: u32 = 44_100;
pub const REQUIRED_CHANNEL_COUNT: u8 = 2;
pub const MAX_RESIDUAL_PAIR_DELTA_LU_V1: f64 = 0.2;
pub const NUMERIC_TOLERANCE_DB: f64 = 1.0e-6;
pub const LOUDNESS_REMEASUREMENT_TOLERANCE_LU: f64 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StimulusArmV1 {
    Baseline,
    Intervention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StimulusTransformV1 {
    ConstantGainAttenuationOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StimulusRendererIdentityV1 {
    /// Exact source revision used to reconstruct the C6F baseline/intervention
    /// renders before loudness matching.
    pub source_revision: String,
    pub renderer_version: String,
    pub render_config_sha256: String,
    pub environment_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct C6fRenderSubjectItemV1 {
    pub item_id: String,
    pub seed: u64,
    pub baseline_render_sha256: String,
    pub intervention_render_sha256: String,
    pub sample_rate_hz: u32,
    pub channel_count: u8,
    pub frame_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrozenC6fRenderSubjectBindingV1 {
    pub binding_version: String,
    pub protocol_sha256: String,
    pub c6f_source_commit: String,
    pub c6f_bundle_sha256: String,
    pub c6f_bundle_version: String,
    pub renderer: StimulusRendererIdentityV1,
    pub items: Vec<C6fRenderSubjectItemV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum C6fRenderSubjectBindingIssueV1 {
    InvalidProtocol,
    ProtocolSerializationFailed,
    ProtocolDigestMismatch,
    WrongVersion,
    AcousticSubjectMismatch,
    RendererRevisionMismatch,
    InvalidDigest { field: String },
    InvalidSourceRevision,
    EmptyRendererVersion,
    WrongItemCount { found: usize },
    WrongItemOrder { index: usize },
    EmptyItemId { index: usize },
    DuplicateItemId { item_id: String },
    DuplicateSeed { seed: u64 },
    WrongProtocolItemBinding { seed: u64 },
    WrongSeedPanel,
    InvalidRenderDigest { item_id: String, arm: StimulusArmV1 },
    IdenticalArmRenders { item_id: String },
    WrongSampleRate { item_id: String },
    WrongChannelCount { item_id: String },
    ZeroFrameCount { item_id: String },
}

impl FrozenC6fRenderSubjectBindingV1 {
    pub fn validate(
        &self,
        protocol: &FrozenPerceptualStudyProtocolV1,
    ) -> Vec<C6fRenderSubjectBindingIssueV1> {
        let mut issues = Vec::new();
        if !protocol.validate().is_empty() {
            issues.push(C6fRenderSubjectBindingIssueV1::InvalidProtocol);
        }
        match canonical_json_sha256(protocol) {
            Ok(value) if value == self.protocol_sha256 => {}
            Ok(_) => issues.push(C6fRenderSubjectBindingIssueV1::ProtocolDigestMismatch),
            Err(_) => issues.push(C6fRenderSubjectBindingIssueV1::ProtocolSerializationFailed),
        }
        if self.binding_version != C6F_RENDER_SUBJECT_BINDING_VERSION {
            issues.push(C6fRenderSubjectBindingIssueV1::WrongVersion);
        }
        if self.c6f_source_commit != protocol.acoustic_subject.c6f_source_commit
            || self.c6f_bundle_sha256 != protocol.acoustic_subject.c6f_bundle_sha256
            || self.c6f_bundle_version != protocol.acoustic_subject.c6f_bundle_version
        {
            issues.push(C6fRenderSubjectBindingIssueV1::AcousticSubjectMismatch);
        }
        if self.renderer.source_revision != self.c6f_source_commit {
            issues.push(C6fRenderSubjectBindingIssueV1::RendererRevisionMismatch);
        }
        for (field, digest) in [
            ("protocol_sha256", self.protocol_sha256.as_str()),
            ("c6f_bundle_sha256", self.c6f_bundle_sha256.as_str()),
            (
                "renderer.render_config_sha256",
                self.renderer.render_config_sha256.as_str(),
            ),
            (
                "renderer.environment_sha256",
                self.renderer.environment_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                issues.push(C6fRenderSubjectBindingIssueV1::InvalidDigest {
                    field: field.into(),
                });
            }
        }
        if !is_git_sha1(&self.c6f_source_commit) || !is_git_sha1(&self.renderer.source_revision) {
            issues.push(C6fRenderSubjectBindingIssueV1::InvalidSourceRevision);
        }
        if self.renderer.renderer_version.trim().is_empty() {
            issues.push(C6fRenderSubjectBindingIssueV1::EmptyRendererVersion);
        }

        validate_render_binding_items(self, protocol, &mut issues);
        issues
    }

    pub fn binding_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoudnessMeterIdentityV1 {
    pub measurement_profile: String,
    pub source_revision: String,
    pub implementation_version: String,
    pub environment_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StimulusAudioAssetV1 {
    pub source_sha256: String,
    pub output_sha256: String,
    pub sample_rate_hz: u32,
    pub channel_count: u8,
    pub frame_count: usize,
    pub initial_integrated_lufs: f64,
    pub final_integrated_lufs: f64,
    /// Constant gain applied to the complete waveform. Must be <= 0 dB.
    pub applied_gain_db: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceptualStimulusPairV1 {
    pub item_id: String,
    pub seed: u64,
    pub baseline: StimulusAudioAssetV1,
    pub intervention: StimulusAudioAssetV1,
    /// Which source arm was louder before matching. `None` means equal within
    /// the frozen numerical tolerance, in which case neither arm may change.
    pub attenuated_arm: Option<StimulusArmV1>,
    pub post_match_pair_delta_lu: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrozenPerceptualStimulusPackV1 {
    pub pack_version: String,
    pub protocol_sha256: String,
    pub render_subject_binding_sha256: String,
    pub c6f_source_commit: String,
    pub c6f_bundle_sha256: String,
    pub c6f_bundle_version: String,
    pub extent: StimulusExtentV1,
    pub loudness_matching: LoudnessMatchingV1,
    pub transform: StimulusTransformV1,
    pub gain_transform_profile: String,
    pub max_attenuation_db: f64,
    pub max_residual_pair_delta_lu: f64,
    pub renderer: StimulusRendererIdentityV1,
    pub loudness_meter: LoudnessMeterIdentityV1,
    pub items: Vec<PerceptualStimulusPairV1>,
    /// P1C stops before participant-facing scheduling/blinding material.
    pub participant_labels_bound: bool,
    pub participant_schedule_bound: bool,
    pub responses_present: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerceptualStimulusPackIssueV1 {
    InvalidProtocol,
    ProtocolSerializationFailed,
    ProtocolDigestMismatch,
    InvalidRenderSubjectBinding,
    RenderSubjectBindingSerializationFailed,
    RenderSubjectBindingDigestMismatch,
    WrongPackVersion,
    AcousticSubjectMismatch,
    RendererIdentityMismatch,
    WrongExtent,
    WrongLoudnessPolicy,
    WrongTransform,
    WrongGainTransformProfile,
    InvalidMaximumAttenuation,
    MaximumAttenuationMismatch,
    WrongResidualTolerance,
    InvalidDigest { field: String },
    InvalidSourceRevision { field: String },
    EmptyIdentityField { field: String },
    WrongLoudnessMeasurementProfile,
    WrongItemCount { found: usize },
    WrongItemOrder { index: usize },
    EmptyItemId { index: usize },
    DuplicateItemId { item_id: String },
    DuplicateSeed { seed: u64 },
    WrongSeedPanel,
    SourceDoesNotMatchC6fBinding { item_id: String, arm: StimulusArmV1 },
    InvalidAssetDigest { item_id: String, arm: StimulusArmV1 },
    WrongSampleRate { item_id: String, arm: StimulusArmV1 },
    WrongChannelCount { item_id: String, arm: StimulusArmV1 },
    ZeroFrameCount { item_id: String, arm: StimulusArmV1 },
    PairFrameCountMismatch { item_id: String },
    NonFiniteLoudness { item_id: String, arm: StimulusArmV1 },
    NonFiniteGain { item_id: String, arm: StimulusArmV1 },
    PositiveGain { item_id: String, arm: StimulusArmV1 },
    GainExceedsMaximum { item_id: String, arm: StimulusArmV1 },
    WrongAttenuatedArm { item_id: String },
    QuieterArmChanged { item_id: String, arm: StimulusArmV1 },
    AttenuatedArmUnchanged { item_id: String, arm: StimulusArmV1 },
    GainDoesNotMatchInitialLoudnessDifference { item_id: String },
    LoudnessRemeasurementMismatch { item_id: String, arm: StimulusArmV1 },
    InvalidPostMatchDelta { item_id: String },
    PostMatchDeltaExceedsTolerance { item_id: String },
    RecordedDeltaMismatch { item_id: String },
    PrematureParticipantLabels,
    PrematureParticipantSchedule,
    ResponsesPresent,
}

impl FrozenPerceptualStimulusPackV1 {
    pub fn validate(
        &self,
        protocol: &FrozenPerceptualStudyProtocolV1,
        render_binding: &FrozenC6fRenderSubjectBindingV1,
    ) -> Vec<PerceptualStimulusPackIssueV1> {
        let mut issues = Vec::new();
        if !protocol.validate().is_empty() {
            issues.push(PerceptualStimulusPackIssueV1::InvalidProtocol);
        }
        match canonical_json_sha256(protocol) {
            Ok(value) if value == self.protocol_sha256 => {}
            Ok(_) => issues.push(PerceptualStimulusPackIssueV1::ProtocolDigestMismatch),
            Err(_) => issues.push(PerceptualStimulusPackIssueV1::ProtocolSerializationFailed),
        }
        if !render_binding.validate(protocol).is_empty() {
            issues.push(PerceptualStimulusPackIssueV1::InvalidRenderSubjectBinding);
        }
        match canonical_json_sha256(render_binding) {
            Ok(value) if value == self.render_subject_binding_sha256 => {}
            Ok(_) => issues.push(PerceptualStimulusPackIssueV1::RenderSubjectBindingDigestMismatch),
            Err(_) => {
                issues.push(PerceptualStimulusPackIssueV1::RenderSubjectBindingSerializationFailed)
            }
        }
        if self.pack_version != PERCEPTUAL_STIMULUS_PACK_VERSION {
            issues.push(PerceptualStimulusPackIssueV1::WrongPackVersion);
        }
        if self.c6f_source_commit != protocol.acoustic_subject.c6f_source_commit
            || self.c6f_bundle_sha256 != protocol.acoustic_subject.c6f_bundle_sha256
            || self.c6f_bundle_version != protocol.acoustic_subject.c6f_bundle_version
            || self.c6f_source_commit != render_binding.c6f_source_commit
            || self.c6f_bundle_sha256 != render_binding.c6f_bundle_sha256
            || self.c6f_bundle_version != render_binding.c6f_bundle_version
        {
            issues.push(PerceptualStimulusPackIssueV1::AcousticSubjectMismatch);
        }
        if self.renderer != render_binding.renderer {
            issues.push(PerceptualStimulusPackIssueV1::RendererIdentityMismatch);
        }
        if self.extent != StimulusExtentV1::WholeFourBarSubject
            || self.extent != protocol.stimulus.extent
        {
            issues.push(PerceptualStimulusPackIssueV1::WrongExtent);
        }
        if self.loudness_matching != LoudnessMatchingV1::PairwiseIntegratedLufsAttenuationOnly
            || self.loudness_matching != protocol.stimulus.loudness_matching
        {
            issues.push(PerceptualStimulusPackIssueV1::WrongLoudnessPolicy);
        }
        if self.transform != StimulusTransformV1::ConstantGainAttenuationOnly {
            issues.push(PerceptualStimulusPackIssueV1::WrongTransform);
        }
        if self.gain_transform_profile != GAIN_TRANSFORM_PROFILE_V1 {
            issues.push(PerceptualStimulusPackIssueV1::WrongGainTransformProfile);
        }
        if !self.max_attenuation_db.is_finite() || self.max_attenuation_db <= 0.0 {
            issues.push(PerceptualStimulusPackIssueV1::InvalidMaximumAttenuation);
        }
        if !same_number(
            self.max_attenuation_db,
            protocol.stimulus.maximum_attenuation_db,
        ) {
            issues.push(PerceptualStimulusPackIssueV1::MaximumAttenuationMismatch);
        }
        if !same_number(
            self.max_residual_pair_delta_lu,
            MAX_RESIDUAL_PAIR_DELTA_LU_V1,
        ) {
            issues.push(PerceptualStimulusPackIssueV1::WrongResidualTolerance);
        }

        for (field, digest) in [
            ("protocol_sha256", self.protocol_sha256.as_str()),
            (
                "render_subject_binding_sha256",
                self.render_subject_binding_sha256.as_str(),
            ),
            ("c6f_bundle_sha256", self.c6f_bundle_sha256.as_str()),
            (
                "renderer.render_config_sha256",
                self.renderer.render_config_sha256.as_str(),
            ),
            (
                "renderer.environment_sha256",
                self.renderer.environment_sha256.as_str(),
            ),
            (
                "loudness_meter.environment_sha256",
                self.loudness_meter.environment_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                issues.push(PerceptualStimulusPackIssueV1::InvalidDigest {
                    field: field.into(),
                });
            }
        }
        for (field, revision) in [
            ("c6f_source_commit", self.c6f_source_commit.as_str()),
            ("renderer.source_revision", self.renderer.source_revision.as_str()),
            (
                "loudness_meter.source_revision",
                self.loudness_meter.source_revision.as_str(),
            ),
        ] {
            if !is_git_sha1(revision) {
                issues.push(PerceptualStimulusPackIssueV1::InvalidSourceRevision {
                    field: field.into(),
                });
            }
        }
        for (field, value) in [
            ("renderer.renderer_version", self.renderer.renderer_version.as_str()),
            (
                "loudness_meter.implementation_version",
                self.loudness_meter.implementation_version.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                issues.push(PerceptualStimulusPackIssueV1::EmptyIdentityField {
                    field: field.into(),
                });
            }
        }
        if self.loudness_meter.measurement_profile != LOUDNESS_MEASUREMENT_PROFILE_V1 {
            issues.push(PerceptualStimulusPackIssueV1::WrongLoudnessMeasurementProfile);
        }

        validate_pack_items(self, protocol, render_binding, &mut issues);

        if self.participant_labels_bound {
            issues.push(PerceptualStimulusPackIssueV1::PrematureParticipantLabels);
        }
        if self.participant_schedule_bound {
            issues.push(PerceptualStimulusPackIssueV1::PrematureParticipantSchedule);
        }
        if self.responses_present {
            issues.push(PerceptualStimulusPackIssueV1::ResponsesPresent);
        }
        issues
    }

    pub fn pack_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

fn validate_render_binding_items(
    binding: &FrozenC6fRenderSubjectBindingV1,
    protocol: &FrozenPerceptualStudyProtocolV1,
    issues: &mut Vec<C6fRenderSubjectBindingIssueV1>,
) {
    if binding.items.len() != MEL003_FIXED_SEEDS.len() {
        issues.push(C6fRenderSubjectBindingIssueV1::WrongItemCount {
            found: binding.items.len(),
        });
    }
    let protocol_items: BTreeMap<_, _> = protocol
        .items
        .iter()
        .map(|item| (item.seed, item.item_id.as_str()))
        .collect();
    let mut item_ids = BTreeSet::new();
    let mut seeds = BTreeSet::new();
    for (index, item) in binding.items.iter().enumerate() {
        if MEL003_FIXED_SEEDS.get(index).copied() != Some(item.seed) {
            issues.push(C6fRenderSubjectBindingIssueV1::WrongItemOrder { index });
        }
        if item.item_id.trim().is_empty() {
            issues.push(C6fRenderSubjectBindingIssueV1::EmptyItemId { index });
        } else if !item_ids.insert(item.item_id.clone()) {
            issues.push(C6fRenderSubjectBindingIssueV1::DuplicateItemId {
                item_id: item.item_id.clone(),
            });
        }
        if !seeds.insert(item.seed) {
            issues.push(C6fRenderSubjectBindingIssueV1::DuplicateSeed { seed: item.seed });
        }
        if protocol_items.get(&item.seed).copied() != Some(item.item_id.as_str()) {
            issues.push(C6fRenderSubjectBindingIssueV1::WrongProtocolItemBinding {
                seed: item.seed,
            });
        }
        for (arm, digest) in [
            (StimulusArmV1::Baseline, item.baseline_render_sha256.as_str()),
            (
                StimulusArmV1::Intervention,
                item.intervention_render_sha256.as_str(),
            ),
        ] {
            if !is_sha256(digest) {
                issues.push(C6fRenderSubjectBindingIssueV1::InvalidRenderDigest {
                    item_id: item.item_id.clone(),
                    arm,
                });
            }
        }
        if item.baseline_render_sha256 == item.intervention_render_sha256 {
            issues.push(C6fRenderSubjectBindingIssueV1::IdenticalArmRenders {
                item_id: item.item_id.clone(),
            });
        }
        if item.sample_rate_hz != REQUIRED_SAMPLE_RATE_HZ {
            issues.push(C6fRenderSubjectBindingIssueV1::WrongSampleRate {
                item_id: item.item_id.clone(),
            });
        }
        if item.channel_count != REQUIRED_CHANNEL_COUNT {
            issues.push(C6fRenderSubjectBindingIssueV1::WrongChannelCount {
                item_id: item.item_id.clone(),
            });
        }
        if item.frame_count == 0 {
            issues.push(C6fRenderSubjectBindingIssueV1::ZeroFrameCount {
                item_id: item.item_id.clone(),
            });
        }
    }
    let expected: BTreeSet<_> = MEL003_FIXED_SEEDS.into_iter().collect();
    if seeds != expected {
        issues.push(C6fRenderSubjectBindingIssueV1::WrongSeedPanel);
    }
}

fn validate_pack_items(
    pack: &FrozenPerceptualStimulusPackV1,
    protocol: &FrozenPerceptualStudyProtocolV1,
    render_binding: &FrozenC6fRenderSubjectBindingV1,
    issues: &mut Vec<PerceptualStimulusPackIssueV1>,
) {
    if pack.items.len() != MEL003_FIXED_SEEDS.len() {
        issues.push(PerceptualStimulusPackIssueV1::WrongItemCount {
            found: pack.items.len(),
        });
    }
    let protocol_items: BTreeMap<_, _> = protocol
        .items
        .iter()
        .map(|item| (item.seed, item.item_id.as_str()))
        .collect();
    let binding_items: BTreeMap<_, _> = render_binding
        .items
        .iter()
        .map(|item| (item.seed, item))
        .collect();
    let mut item_ids = BTreeSet::new();
    let mut seeds = BTreeSet::new();

    for (index, pair) in pack.items.iter().enumerate() {
        if MEL003_FIXED_SEEDS.get(index).copied() != Some(pair.seed) {
            issues.push(PerceptualStimulusPackIssueV1::WrongItemOrder { index });
        }
        if pair.item_id.trim().is_empty() {
            issues.push(PerceptualStimulusPackIssueV1::EmptyItemId { index });
        } else if !item_ids.insert(pair.item_id.clone()) {
            issues.push(PerceptualStimulusPackIssueV1::DuplicateItemId {
                item_id: pair.item_id.clone(),
            });
        }
        if !seeds.insert(pair.seed) {
            issues.push(PerceptualStimulusPackIssueV1::DuplicateSeed { seed: pair.seed });
        }
        if protocol_items.get(&pair.seed).copied() != Some(pair.item_id.as_str()) {
            issues.push(PerceptualStimulusPackIssueV1::WrongSeedPanel);
        }

        let binding_item = binding_items.get(&pair.seed).copied();
        validate_asset(
            pair,
            StimulusArmV1::Baseline,
            &pair.baseline,
            binding_item,
            pack,
            issues,
        );
        validate_asset(
            pair,
            StimulusArmV1::Intervention,
            &pair.intervention,
            binding_item,
            pack,
            issues,
        );
        if pair.baseline.frame_count != pair.intervention.frame_count {
            issues.push(PerceptualStimulusPackIssueV1::PairFrameCountMismatch {
                item_id: pair.item_id.clone(),
            });
        }
        validate_gain_semantics(pair, pack, issues);
    }

    let expected: BTreeSet<_> = MEL003_FIXED_SEEDS.into_iter().collect();
    if seeds != expected {
        issues.push(PerceptualStimulusPackIssueV1::WrongSeedPanel);
    }
}

fn validate_asset(
    pair: &PerceptualStimulusPairV1,
    arm: StimulusArmV1,
    asset: &StimulusAudioAssetV1,
    binding_item: Option<&C6fRenderSubjectItemV1>,
    pack: &FrozenPerceptualStimulusPackV1,
    issues: &mut Vec<PerceptualStimulusPackIssueV1>,
) {
    for digest in [&asset.source_sha256, &asset.output_sha256] {
        if !is_sha256(digest) {
            issues.push(PerceptualStimulusPackIssueV1::InvalidAssetDigest {
                item_id: pair.item_id.clone(),
                arm,
            });
            break;
        }
    }
    let source_matches_binding = binding_item.is_some_and(|binding| {
        let expected_digest = match arm {
            StimulusArmV1::Baseline => &binding.baseline_render_sha256,
            StimulusArmV1::Intervention => &binding.intervention_render_sha256,
        };
        asset.source_sha256 == *expected_digest
            && asset.sample_rate_hz == binding.sample_rate_hz
            && asset.channel_count == binding.channel_count
            && asset.frame_count == binding.frame_count
    });
    if !source_matches_binding {
        issues.push(PerceptualStimulusPackIssueV1::SourceDoesNotMatchC6fBinding {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
    if asset.sample_rate_hz != REQUIRED_SAMPLE_RATE_HZ {
        issues.push(PerceptualStimulusPackIssueV1::WrongSampleRate {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
    if asset.channel_count != REQUIRED_CHANNEL_COUNT {
        issues.push(PerceptualStimulusPackIssueV1::WrongChannelCount {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
    if asset.frame_count == 0 {
        issues.push(PerceptualStimulusPackIssueV1::ZeroFrameCount {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
    if !asset.initial_integrated_lufs.is_finite() || !asset.final_integrated_lufs.is_finite() {
        issues.push(PerceptualStimulusPackIssueV1::NonFiniteLoudness {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
    if !asset.applied_gain_db.is_finite() {
        issues.push(PerceptualStimulusPackIssueV1::NonFiniteGain {
            item_id: pair.item_id.clone(),
            arm,
        });
    } else {
        if asset.applied_gain_db > NUMERIC_TOLERANCE_DB {
            issues.push(PerceptualStimulusPackIssueV1::PositiveGain {
                item_id: pair.item_id.clone(),
                arm,
            });
        }
        if -asset.applied_gain_db > pack.max_attenuation_db + NUMERIC_TOLERANCE_DB {
            issues.push(PerceptualStimulusPackIssueV1::GainExceedsMaximum {
                item_id: pair.item_id.clone(),
                arm,
            });
        }
    }
}

fn validate_gain_semantics(
    pair: &PerceptualStimulusPairV1,
    pack: &FrozenPerceptualStimulusPackV1,
    issues: &mut Vec<PerceptualStimulusPackIssueV1>,
) {
    let initial_delta = pair.baseline.initial_integrated_lufs - pair.intervention.initial_integrated_lufs;
    let expected_arm = if initial_delta > NUMERIC_TOLERANCE_DB {
        Some(StimulusArmV1::Baseline)
    } else if initial_delta < -NUMERIC_TOLERANCE_DB {
        Some(StimulusArmV1::Intervention)
    } else {
        None
    };
    if pair.attenuated_arm != expected_arm {
        issues.push(PerceptualStimulusPackIssueV1::WrongAttenuatedArm {
            item_id: pair.item_id.clone(),
        });
    }

    let expected_attenuation = initial_delta.abs();
    match expected_arm {
        Some(StimulusArmV1::Baseline) => {
            require_changed_attenuated(
                pair,
                StimulusArmV1::Baseline,
                &pair.baseline,
                expected_attenuation,
                issues,
            );
            require_unchanged_quieter(
                pair,
                StimulusArmV1::Intervention,
                &pair.intervention,
                issues,
            );
        }
        Some(StimulusArmV1::Intervention) => {
            require_changed_attenuated(
                pair,
                StimulusArmV1::Intervention,
                &pair.intervention,
                expected_attenuation,
                issues,
            );
            require_unchanged_quieter(
                pair,
                StimulusArmV1::Baseline,
                &pair.baseline,
                issues,
            );
        }
        None => {
            require_unchanged_quieter(pair, StimulusArmV1::Baseline, &pair.baseline, issues);
            require_unchanged_quieter(
                pair,
                StimulusArmV1::Intervention,
                &pair.intervention,
                issues,
            );
        }
    }

    if expected_attenuation > pack.max_attenuation_db + NUMERIC_TOLERANCE_DB {
        let arm = expected_arm.unwrap_or(StimulusArmV1::Baseline);
        issues.push(PerceptualStimulusPackIssueV1::GainExceedsMaximum {
            item_id: pair.item_id.clone(),
            arm,
        });
    }

    if !pair.post_match_pair_delta_lu.is_finite() || pair.post_match_pair_delta_lu < 0.0 {
        issues.push(PerceptualStimulusPackIssueV1::InvalidPostMatchDelta {
            item_id: pair.item_id.clone(),
        });
        return;
    }
    let measured_delta =
        (pair.baseline.final_integrated_lufs - pair.intervention.final_integrated_lufs).abs();
    if !same_number(pair.post_match_pair_delta_lu, measured_delta) {
        issues.push(PerceptualStimulusPackIssueV1::RecordedDeltaMismatch {
            item_id: pair.item_id.clone(),
        });
    }
    if pair.post_match_pair_delta_lu > pack.max_residual_pair_delta_lu + NUMERIC_TOLERANCE_DB {
        issues.push(PerceptualStimulusPackIssueV1::PostMatchDeltaExceedsTolerance {
            item_id: pair.item_id.clone(),
        });
    }
}

fn require_unchanged_quieter(
    pair: &PerceptualStimulusPairV1,
    arm: StimulusArmV1,
    asset: &StimulusAudioAssetV1,
    issues: &mut Vec<PerceptualStimulusPackIssueV1>,
) {
    if asset.source_sha256 != asset.output_sha256
        || !same_number(asset.applied_gain_db, 0.0)
        || !same_number(asset.initial_integrated_lufs, asset.final_integrated_lufs)
    {
        issues.push(PerceptualStimulusPackIssueV1::QuieterArmChanged {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
}

fn require_changed_attenuated(
    pair: &PerceptualStimulusPairV1,
    arm: StimulusArmV1,
    asset: &StimulusAudioAssetV1,
    expected_attenuation_db: f64,
    issues: &mut Vec<PerceptualStimulusPackIssueV1>,
) {
    if asset.source_sha256 == asset.output_sha256 {
        issues.push(PerceptualStimulusPackIssueV1::AttenuatedArmUnchanged {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
    if !same_number(asset.applied_gain_db, -expected_attenuation_db) {
        issues.push(
            PerceptualStimulusPackIssueV1::GainDoesNotMatchInitialLoudnessDifference {
                item_id: pair.item_id.clone(),
            },
        );
    }
    let expected_final = asset.initial_integrated_lufs + asset.applied_gain_db;
    if !measurement_close(asset.final_integrated_lufs, expected_final) {
        issues.push(PerceptualStimulusPackIssueV1::LoudnessRemeasurementMismatch {
            item_id: pair.item_id.clone(),
            arm,
        });
    }
}

fn same_number(left: f64, right: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= NUMERIC_TOLERANCE_DB
}

fn measurement_close(left: f64, right: f64) -> bool {
    left.is_finite()
        && right.is_finite()
        && (left - right).abs() <= LOUDNESS_REMEASUREMENT_TOLERANCE_LU
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_git_sha1(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::perceptual_study_protocol::{
        AnalysisPolicyV1, BlindingAndRandomizationPolicyV1, ExternalPerceptualPreregistrationV1,
        ForbiddenPerceptualClaimV1, Mel003AcousticSubjectBindingV1, MissingResponsePolicyV1,
        ParticipantPolicyV1, PerceptualEndpointRoleV1, PerceptualEndpointV1,
        PerceptualStudyItemV1, PerceptualTaskV1, PrimaryAnalysisModelV1, SampleSizePlanV1,
        SecondaryMultiplicityPolicyV1, StimulusPolicyV1, MEL003_C6F_BUNDLE_VERSION,
        PERCEPTUAL_STUDY_PROTOCOL_VERSION,
    };

    const DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const OTHER_DIGEST: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const COMMIT: &str = "cccccccccccccccccccccccccccccccccccccccc";

    fn protocol() -> FrozenPerceptualStudyProtocolV1 {
        FrozenPerceptualStudyProtocolV1 {
            protocol_version: PERCEPTUAL_STUDY_PROTOCOL_VERSION.into(),
            acoustic_subject: Mel003AcousticSubjectBindingV1 {
                c6f_source_commit: COMMIT.into(),
                c6f_bundle_sha256: DIGEST.into(),
                c6f_bundle_version: MEL003_C6F_BUNDLE_VERSION.into(),
            },
            external_preregistration: ExternalPerceptualPreregistrationV1 {
                registry: "OSF".into(),
                record_id: "mel003-p1".into(),
                frozen_at_utc: "2026-09-19T00:00:00Z".into(),
                record_sha256: DIGEST.into(),
            },
            analysis_spec_sha256: DIGEST.into(),
            items: MEL003_FIXED_SEEDS
                .into_iter()
                .map(|seed| PerceptualStudyItemV1 {
                    item_id: format!("sonata-seed-{seed}"),
                    seed,
                })
                .collect(),
            endpoints: vec![
                PerceptualEndpointV1 {
                    task: PerceptualTaskV1::AbxDiscrimination,
                    role: PerceptualEndpointRoleV1::Primary,
                    chance_probability: 0.5,
                    estimand: "ABX correctness probability".into(),
                },
                PerceptualEndpointV1 {
                    task: PerceptualTaskV1::DirectionalRearticulation2Afc,
                    role: PerceptualEndpointRoleV1::KeySecondary,
                    chance_probability: 0.5,
                    estimand: "directional re-articulation choice probability".into(),
                },
            ],
            stimulus: StimulusPolicyV1 {
                extent: StimulusExtentV1::WholeFourBarSubject,
                synchronized_playhead_required: true,
                loudness_matching: LoudnessMatchingV1::PairwiseIntegratedLufsAttenuationOnly,
                maximum_attenuation_db: 6.0,
                preserve_pair_alignment: true,
                neutral_presentation_labels_required: true,
                disjoint_practice_material_required: true,
            },
            blinding: BlindingAndRandomizationPolicyV1 {
                randomization_commitment_sha256: DIGEST.into(),
                schedule_builder_version: "schedule-v1".into(),
                balance_ab_label_assignment: true,
                balance_abx_hidden_identity: true,
                balance_directional_left_right_assignment: true,
                reveal_correct_answers_during_scored_collection: false,
                arm_labelled_monitoring_during_collection: false,
                investigator_can_modify_schedule_after_first_response: false,
            },
            participants: ParticipantPolicyV1 {
                minimum_age_years: 18,
                informed_consent_required: true,
                pseudonymous_participant_tokens_required: true,
                raw_names_or_contact_details_in_study_dataset_allowed: false,
                stereo_playback_check_required: true,
                task_comprehension_practice_required: true,
                practice_feedback_allowed: true,
                scored_trial_feedback_allowed: false,
            },
            sample_size: SampleSizePlanV1 {
                planning_artifact_sha256: DIGEST.into(),
                planned_completed_participants: 48,
                maximum_enrolled_participants: 54,
                outcome_adaptive_stopping_allowed: false,
            },
            analysis: AnalysisPolicyV1 {
                primary_model: PrimaryAnalysisModelV1::CrossedParticipantItemLogisticRandomIntercepts,
                participant_grouping_factor_required: true,
                item_grouping_factor_required: true,
                primary_alternative_is_greater_than_chance: true,
                alpha: 0.05,
                confidence_level: 0.95,
                secondary_multiplicity: SecondaryMultiplicityPolicyV1::HolmWithinRegisteredSecondaryFamily,
                report_item_level_outcomes: true,
                report_participant_level_outcomes: true,
                report_random_effect_variance: true,
                missing_response_policy: MissingResponsePolicyV1::RetainRawExcludeIncompleteSessionNoImputation,
            },
            forbidden_claims: vec![
                ForbiddenPerceptualClaimV1::Preference,
                ForbiddenPerceptualClaimV1::ArtisticQuality,
                ForbiddenPerceptualClaimV1::EmotionalImpact,
                ForbiddenPerceptualClaimV1::StyleIdentity,
                ForbiddenPerceptualClaimV1::CulturalAuthenticity,
                ForbiddenPerceptualClaimV1::HumanLikePerformance,
                ForbiddenPerceptualClaimV1::IndependentAcousticReplication,
                ForbiddenPerceptualClaimV1::CognitionProductAuthority,
                ForbiddenPerceptualClaimV1::GeneralizationBeyondRegisteredItemsAndEligiblePopulation,
            ],
            stimulus_pack_bound: false,
            participant_schedule_bound: false,
            collection_authorized: false,
            responses_present: false,
        }
    }

    fn render_binding(protocol: &FrozenPerceptualStudyProtocolV1) -> FrozenC6fRenderSubjectBindingV1 {
        FrozenC6fRenderSubjectBindingV1 {
            binding_version: C6F_RENDER_SUBJECT_BINDING_VERSION.into(),
            protocol_sha256: protocol.protocol_sha256().unwrap(),
            c6f_source_commit: protocol.acoustic_subject.c6f_source_commit.clone(),
            c6f_bundle_sha256: protocol.acoustic_subject.c6f_bundle_sha256.clone(),
            c6f_bundle_version: protocol.acoustic_subject.c6f_bundle_version.clone(),
            renderer: StimulusRendererIdentityV1 {
                source_revision: protocol.acoustic_subject.c6f_source_commit.clone(),
                renderer_version: "native-sonata-renderer-v1".into(),
                render_config_sha256: DIGEST.into(),
                environment_sha256: DIGEST.into(),
            },
            items: protocol
                .items
                .iter()
                .enumerate()
                .map(|(index, item)| C6fRenderSubjectItemV1 {
                    item_id: item.item_id.clone(),
                    seed: item.seed,
                    baseline_render_sha256: format!("{:064x}", index + 1),
                    intervention_render_sha256: format!("{:064x}", index + 100),
                    sample_rate_hz: REQUIRED_SAMPLE_RATE_HZ,
                    channel_count: REQUIRED_CHANNEL_COUNT,
                    frame_count: 88_200,
                })
                .collect(),
        }
    }

    fn asset(source: &str, output: &str, initial: f64, final_lufs: f64, gain: f64) -> StimulusAudioAssetV1 {
        StimulusAudioAssetV1 {
            source_sha256: source.into(),
            output_sha256: output.into(),
            sample_rate_hz: REQUIRED_SAMPLE_RATE_HZ,
            channel_count: REQUIRED_CHANNEL_COUNT,
            frame_count: 88_200,
            initial_integrated_lufs: initial,
            final_integrated_lufs: final_lufs,
            applied_gain_db: gain,
        }
    }

    fn pack(
        protocol: &FrozenPerceptualStudyProtocolV1,
        binding: &FrozenC6fRenderSubjectBindingV1,
    ) -> FrozenPerceptualStimulusPackV1 {
        let items = binding
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                let baseline_output = format!("{:064x}", index + 300);
                PerceptualStimulusPairV1 {
                    item_id: item.item_id.clone(),
                    seed: item.seed,
                    baseline: asset(
                        &item.baseline_render_sha256,
                        &baseline_output,
                        -17.0,
                        -18.0,
                        -1.0,
                    ),
                    intervention: asset(
                        &item.intervention_render_sha256,
                        &item.intervention_render_sha256,
                        -18.0,
                        -18.0,
                        0.0,
                    ),
                    attenuated_arm: Some(StimulusArmV1::Baseline),
                    post_match_pair_delta_lu: 0.0,
                }
            })
            .collect();
        FrozenPerceptualStimulusPackV1 {
            pack_version: PERCEPTUAL_STIMULUS_PACK_VERSION.into(),
            protocol_sha256: protocol.protocol_sha256().unwrap(),
            render_subject_binding_sha256: binding.binding_sha256().unwrap(),
            c6f_source_commit: protocol.acoustic_subject.c6f_source_commit.clone(),
            c6f_bundle_sha256: protocol.acoustic_subject.c6f_bundle_sha256.clone(),
            c6f_bundle_version: protocol.acoustic_subject.c6f_bundle_version.clone(),
            extent: StimulusExtentV1::WholeFourBarSubject,
            loudness_matching: LoudnessMatchingV1::PairwiseIntegratedLufsAttenuationOnly,
            transform: StimulusTransformV1::ConstantGainAttenuationOnly,
            gain_transform_profile: GAIN_TRANSFORM_PROFILE_V1.into(),
            max_attenuation_db: protocol.stimulus.maximum_attenuation_db,
            max_residual_pair_delta_lu: MAX_RESIDUAL_PAIR_DELTA_LU_V1,
            renderer: binding.renderer.clone(),
            loudness_meter: LoudnessMeterIdentityV1 {
                measurement_profile: LOUDNESS_MEASUREMENT_PROFILE_V1.into(),
                source_revision: COMMIT.into(),
                implementation_version: "symthaea-measure-lufs-v1".into(),
                environment_sha256: DIGEST.into(),
            },
            items,
            participant_labels_bound: false,
            participant_schedule_bound: false,
            responses_present: false,
        }
    }

    #[test]
    fn valid_pack_is_bound_to_exact_c6f_render_subjects() {
        let protocol = protocol();
        assert!(protocol.validate().is_empty());
        let binding = render_binding(&protocol);
        assert!(binding.validate(&protocol).is_empty());
        let pack = pack(&protocol, &binding);
        assert!(pack.validate(&protocol, &binding).is_empty());
        for (pair, source) in pack.items.iter().zip(&binding.items) {
            assert_eq!(pair.baseline.source_sha256, source.baseline_render_sha256);
            assert_eq!(pair.intervention.source_sha256, source.intervention_render_sha256);
            assert_eq!(pair.intervention.source_sha256, pair.intervention.output_sha256);
            assert_ne!(pair.baseline.source_sha256, pair.baseline.output_sha256);
        }
        assert_eq!(pack.pack_sha256().unwrap().len(), 64);
    }

    #[test]
    fn unbound_waveform_substitution_is_rejected() {
        let protocol = protocol();
        let binding = render_binding(&protocol);
        let mut pack = pack(&protocol, &binding);
        pack.items[0].baseline.source_sha256 = OTHER_DIGEST.into();
        assert!(pack.validate(&protocol, &binding).iter().any(|issue| matches!(
            issue,
            PerceptualStimulusPackIssueV1::SourceDoesNotMatchC6fBinding { .. }
        )));
    }

    #[test]
    fn positive_gain_is_rejected() {
        let protocol = protocol();
        let binding = render_binding(&protocol);
        let mut pack = pack(&protocol, &binding);
        pack.items[0].intervention.applied_gain_db = 1.0;
        assert!(pack.validate(&protocol, &binding).iter().any(|issue| matches!(
            issue,
            PerceptualStimulusPackIssueV1::PositiveGain { .. }
        )));
    }

    #[test]
    fn quieter_arm_must_be_bit_identical() {
        let protocol = protocol();
        let binding = render_binding(&protocol);
        let mut pack = pack(&protocol, &binding);
        pack.items[0].intervention.output_sha256 = OTHER_DIGEST.into();
        assert!(pack.validate(&protocol, &binding).iter().any(|issue| matches!(
            issue,
            PerceptualStimulusPackIssueV1::QuieterArmChanged { .. }
        )));
    }

    #[test]
    fn attenuation_and_remeasurement_are_load_bearing() {
        let protocol = protocol();
        let binding = render_binding(&protocol);
        let mut pack = pack(&protocol, &binding);
        pack.items[0].baseline.applied_gain_db = -0.5;
        pack.items[0].baseline.final_integrated_lufs = -17.5;
        pack.items[0].post_match_pair_delta_lu = 0.5;
        let issues = pack.validate(&protocol, &binding);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            PerceptualStimulusPackIssueV1::GainDoesNotMatchInitialLoudnessDifference { .. }
        )));
        assert!(issues.iter().any(|issue| matches!(
            issue,
            PerceptualStimulusPackIssueV1::PostMatchDeltaExceedsTolerance { .. }
        )));
    }

    #[test]
    fn item_order_is_canonical_and_participant_state_is_out_of_scope() {
        let protocol = protocol();
        let binding = render_binding(&protocol);
        let mut pack = pack(&protocol, &binding);
        pack.items.swap(0, 1);
        pack.participant_schedule_bound = true;
        pack.responses_present = true;
        let issues = pack.validate(&protocol, &binding);
        assert!(issues.iter().any(|issue| matches!(
            issue,
            PerceptualStimulusPackIssueV1::WrongItemOrder { .. }
        )));
        assert!(issues.contains(&PerceptualStimulusPackIssueV1::PrematureParticipantSchedule));
        assert!(issues.contains(&PerceptualStimulusPackIssueV1::ResponsesPresent));
    }
}
