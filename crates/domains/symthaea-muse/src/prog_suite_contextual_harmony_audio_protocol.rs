// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pre-outcome native-audio survival protocol for the contextual ProgSuite
//! harmony intervention.
//!
//! This module freezes the first rendered-audio question before any lockbox
//! subject is executed:
//!
//! > when a frozen symbolic progression intervention exists, does any exact
//! > whole-work native StereoF32 waveform difference survive realization?
//!
//! V1 is intentionally narrower than onset, spectral, perceptual, preference,
//! or quality analysis. Renderer erasure is an admissible observed outcome.

use crate::MusicalState;
use serde::{Deserialize, Serialize};
use symthaea_music_theory::{
    PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT,
    ProgSuiteContextualHarmonyLockboxErrorV1, ProgSuiteContextualHarmonyLockboxV1,
    ProgSuiteHarmonyLockboxStatusV1,
    predeclare_prog_suite_contextual_harmony_motif_lockbox_v1,
};

pub const PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_SURVIVAL_PROTOCOL_VERSION: &str =
    "melothaea-prog-suite-contextual-harmony-audio-survival-protocol-v1";
pub const PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_RENDERER_ID: &str =
    "symthaea-muse::theory_realize::realize_with_spec/native";
pub const PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_SAMPLE_RATE: u32 = 44_100;
pub const PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_REPEATED_RENDERS_PER_ARM: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioSurvivalArmV1 {
    Legacy,
    Contextual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioSurvivalOutputV1 {
    NativeStereoF32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioSurvivalHashV1 {
    /// SHA-256 over finite stereo f32 samples serialized frame-major as
    /// little-endian left bits followed by little-endian right bits.
    StereoF32LittleEndianSha256,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioSurvivalPostRenderProcessingV1 {
    /// Preserve the renderer's own canonical internal processing/mastering,
    /// but apply no independent gain, loudness normalization, trimming,
    /// excerpting, denoising, EQ, or corrective mastering after it returns.
    NoneBeyondNativeRenderer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioSurvivalRequiredCheckV1 {
    SourceLockboxValidAndStillPredeclared,
    SymbolicComparisonCanonicallyValidated,
    SameSpecSeedIntentStateAndSampleRateAcrossArms,
    NativeStereoF32Output,
    AllSamplesFinite,
    TwoExactRendersPerArm,
    RepeatedRendersBitEqualWithinArm,
    WholeWorkNativeWaveformHashRecorded,
    WholeWorkScoreHashRecorded,
    AlignedFrameDeltaRecorded,
    UnmatchedTailFramesRecorded,
    NoPostRenderLevelMatching,
    NoExcerptSelection,
    NoOutcomeTunedThreshold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioSurvivalMetricV1 {
    SourceScoreSha256,
    ContextualScoreSha256,
    SourceNativeWaveformSha256,
    ContextualNativeWaveformSha256,
    SourceFrameCount,
    ContextualFrameCount,
    AlignedFrameCount,
    ChangedAlignedFrameCount,
    MeanAbsoluteSampleDelta,
    RmsSampleDelta,
    PeakAbsoluteSampleDelta,
    UnmatchedSourceTailFrames,
    UnmatchedContextualTailFrames,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioSurvivalOutcomeV1 {
    /// The frozen contextual policy produced no actual symbolic intervention
    /// for this subject. Audio equality/difference is therefore not evidence
    /// for renderer survival of the intervention.
    NoSymbolicIntervention,
    /// Symbolic score identity differs under the frozen progression policy,
    /// but native whole-work StereoF32 output is exactly equal.
    SymbolicInterventionRendererErased,
    /// Symbolic score identity differs and native whole-work StereoF32 output
    /// differs under the exact paired render policy.
    SymbolicInterventionSurvivedNativeRender,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioSurvivalPrimaryUnitV1 {
    MotifSeedSubject,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioSurvivalAggregationV1 {
    DescriptiveSubjectCountsAndMetricDistributionsOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgSuiteAudioSurvivalAnalysisPlanV1 {
    pub primary_unit: ProgSuiteAudioSurvivalPrimaryUnitV1,
    pub expected_subject_count: usize,
    pub aggregation: ProgSuiteAudioSurvivalAggregationV1,
    /// Audio frames are repeated measurements inside one musical subject.
    pub frame_level_inference_allowed: bool,
    /// Score notes/events remain dependent measurements inside one subject.
    pub event_level_inference_allowed: bool,
    /// V1 classifies exact equality/non-equality only. No effect-size threshold
    /// may be fitted from these frozen subjects.
    pub outcome_tuned_threshold_allowed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteAudioSurvivalRenderPolicyV1 {
    pub renderer_id: String,
    /// Muse package version only; not a Git/source-revision identity.
    pub muse_package_version: String,
    pub sample_rate: u32,
    pub output: ProgSuiteAudioSurvivalOutputV1,
    pub hash: ProgSuiteAudioSurvivalHashV1,
    pub render_state: MusicalState,
    pub repeated_renders_per_arm: u8,
    pub post_render_processing: ProgSuiteAudioSurvivalPostRenderProcessingV1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProgSuiteAudioSurvivalNonClaimV1 {
    ProtocolDoesNotEstablishExecution,
    PackageVersionDoesNotEstablishSourceRevision,
    NativeWaveformDifferenceDoesNotEstablishAudibility,
    NativeWaveformDifferenceDoesNotEstablishAcousticOnsetDifference,
    NativeWaveformDifferenceDoesNotEstablishSpectralSalience,
    NativeWaveformDifferenceDoesNotEstablishListenerPreference,
    NativeWaveformDifferenceDoesNotEstablishArtisticQuality,
    WholeWorkDifferenceDoesNotLocalizeTheEffect,
    FixedRendererDoesNotEstablishRendererGeneralization,
    LockboxPopulationDoesNotEstablishUniversalMusicGeneralization,
    FrameCountsDoNotEstablishStatisticalIndependence,
    ProtocolDoesNotGrantProductAuthority,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProgSuiteContextualHarmonyAudioSurvivalProtocolV1 {
    pub version: String,
    /// Full frozen symbolic lockbox retained rather than only a loose version
    /// string, so every future audio subject remains bound to exact motif/seed
    /// identities and the pre-outcome analysis boundary.
    pub source_lockbox: ProgSuiteContextualHarmonyLockboxV1,
    pub render_policy: ProgSuiteAudioSurvivalRenderPolicyV1,
    pub arms: Vec<ProgSuiteAudioSurvivalArmV1>,
    pub required_checks: Vec<ProgSuiteAudioSurvivalRequiredCheckV1>,
    pub metrics: Vec<ProgSuiteAudioSurvivalMetricV1>,
    pub admissible_outcomes: Vec<ProgSuiteAudioSurvivalOutcomeV1>,
    pub analysis: ProgSuiteAudioSurvivalAnalysisPlanV1,
    pub nonclaims: Vec<ProgSuiteAudioSurvivalNonClaimV1>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1 {
    SourceLockbox(ProgSuiteContextualHarmonyLockboxErrorV1),
    WrongVersion { found: String },
    SourceLockboxNoLongerPredeclared,
    NonCanonicalProtocol,
}

pub fn predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1(
) -> Result<
    ProgSuiteContextualHarmonyAudioSurvivalProtocolV1,
    ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1,
> {
    let source_lockbox = predeclare_prog_suite_contextual_harmony_motif_lockbox_v1()
        .map_err(ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1::SourceLockbox)?;
    source_lockbox
        .validate()
        .map_err(ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1::SourceLockbox)?;
    if source_lockbox.status != ProgSuiteHarmonyLockboxStatusV1::PredeclaredBeforeExecution {
        return Err(
            ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1::SourceLockboxNoLongerPredeclared,
        );
    }

    Ok(ProgSuiteContextualHarmonyAudioSurvivalProtocolV1 {
        version: PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_SURVIVAL_PROTOCOL_VERSION.into(),
        source_lockbox,
        render_policy: canonical_render_policy(),
        arms: vec![
            ProgSuiteAudioSurvivalArmV1::Legacy,
            ProgSuiteAudioSurvivalArmV1::Contextual,
        ],
        required_checks: required_checks(),
        metrics: required_metrics(),
        admissible_outcomes: admissible_outcomes(),
        analysis: canonical_analysis_plan(),
        nonclaims: required_nonclaims(),
    })
}

impl ProgSuiteContextualHarmonyAudioSurvivalProtocolV1 {
    pub fn validate(
        &self,
    ) -> Result<(), ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1> {
        if self.version != PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_SURVIVAL_PROTOCOL_VERSION {
            return Err(
                ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1::WrongVersion {
                    found: self.version.clone(),
                },
            );
        }
        self.source_lockbox
            .validate()
            .map_err(ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1::SourceLockbox)?;
        if self.source_lockbox.status != ProgSuiteHarmonyLockboxStatusV1::PredeclaredBeforeExecution {
            return Err(
                ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1::SourceLockboxNoLongerPredeclared,
            );
        }
        let canonical = predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1()?;
        if &canonical != self {
            return Err(ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1::NonCanonicalProtocol);
        }
        Ok(())
    }
}

fn canonical_render_policy() -> ProgSuiteAudioSurvivalRenderPolicyV1 {
    ProgSuiteAudioSurvivalRenderPolicyV1 {
        renderer_id: PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_RENDERER_ID.into(),
        muse_package_version: env!("CARGO_PKG_VERSION").into(),
        sample_rate: PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_SAMPLE_RATE,
        output: ProgSuiteAudioSurvivalOutputV1::NativeStereoF32,
        hash: ProgSuiteAudioSurvivalHashV1::StereoF32LittleEndianSha256,
        render_state: MusicalState::default(),
        repeated_renders_per_arm:
            PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_REPEATED_RENDERS_PER_ARM,
        post_render_processing:
            ProgSuiteAudioSurvivalPostRenderProcessingV1::NoneBeyondNativeRenderer,
    }
}

fn required_checks() -> Vec<ProgSuiteAudioSurvivalRequiredCheckV1> {
    vec![
        ProgSuiteAudioSurvivalRequiredCheckV1::SourceLockboxValidAndStillPredeclared,
        ProgSuiteAudioSurvivalRequiredCheckV1::SymbolicComparisonCanonicallyValidated,
        ProgSuiteAudioSurvivalRequiredCheckV1::SameSpecSeedIntentStateAndSampleRateAcrossArms,
        ProgSuiteAudioSurvivalRequiredCheckV1::NativeStereoF32Output,
        ProgSuiteAudioSurvivalRequiredCheckV1::AllSamplesFinite,
        ProgSuiteAudioSurvivalRequiredCheckV1::TwoExactRendersPerArm,
        ProgSuiteAudioSurvivalRequiredCheckV1::RepeatedRendersBitEqualWithinArm,
        ProgSuiteAudioSurvivalRequiredCheckV1::WholeWorkNativeWaveformHashRecorded,
        ProgSuiteAudioSurvivalRequiredCheckV1::WholeWorkScoreHashRecorded,
        ProgSuiteAudioSurvivalRequiredCheckV1::AlignedFrameDeltaRecorded,
        ProgSuiteAudioSurvivalRequiredCheckV1::UnmatchedTailFramesRecorded,
        ProgSuiteAudioSurvivalRequiredCheckV1::NoPostRenderLevelMatching,
        ProgSuiteAudioSurvivalRequiredCheckV1::NoExcerptSelection,
        ProgSuiteAudioSurvivalRequiredCheckV1::NoOutcomeTunedThreshold,
    ]
}

fn required_metrics() -> Vec<ProgSuiteAudioSurvivalMetricV1> {
    vec![
        ProgSuiteAudioSurvivalMetricV1::SourceScoreSha256,
        ProgSuiteAudioSurvivalMetricV1::ContextualScoreSha256,
        ProgSuiteAudioSurvivalMetricV1::SourceNativeWaveformSha256,
        ProgSuiteAudioSurvivalMetricV1::ContextualNativeWaveformSha256,
        ProgSuiteAudioSurvivalMetricV1::SourceFrameCount,
        ProgSuiteAudioSurvivalMetricV1::ContextualFrameCount,
        ProgSuiteAudioSurvivalMetricV1::AlignedFrameCount,
        ProgSuiteAudioSurvivalMetricV1::ChangedAlignedFrameCount,
        ProgSuiteAudioSurvivalMetricV1::MeanAbsoluteSampleDelta,
        ProgSuiteAudioSurvivalMetricV1::RmsSampleDelta,
        ProgSuiteAudioSurvivalMetricV1::PeakAbsoluteSampleDelta,
        ProgSuiteAudioSurvivalMetricV1::UnmatchedSourceTailFrames,
        ProgSuiteAudioSurvivalMetricV1::UnmatchedContextualTailFrames,
    ]
}

fn admissible_outcomes() -> Vec<ProgSuiteAudioSurvivalOutcomeV1> {
    vec![
        ProgSuiteAudioSurvivalOutcomeV1::NoSymbolicIntervention,
        ProgSuiteAudioSurvivalOutcomeV1::SymbolicInterventionRendererErased,
        ProgSuiteAudioSurvivalOutcomeV1::SymbolicInterventionSurvivedNativeRender,
    ]
}

fn canonical_analysis_plan() -> ProgSuiteAudioSurvivalAnalysisPlanV1 {
    ProgSuiteAudioSurvivalAnalysisPlanV1 {
        primary_unit: ProgSuiteAudioSurvivalPrimaryUnitV1::MotifSeedSubject,
        expected_subject_count: PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT,
        aggregation:
            ProgSuiteAudioSurvivalAggregationV1::DescriptiveSubjectCountsAndMetricDistributionsOnly,
        frame_level_inference_allowed: false,
        event_level_inference_allowed: false,
        outcome_tuned_threshold_allowed: false,
    }
}

fn required_nonclaims() -> Vec<ProgSuiteAudioSurvivalNonClaimV1> {
    vec![
        ProgSuiteAudioSurvivalNonClaimV1::ProtocolDoesNotEstablishExecution,
        ProgSuiteAudioSurvivalNonClaimV1::PackageVersionDoesNotEstablishSourceRevision,
        ProgSuiteAudioSurvivalNonClaimV1::NativeWaveformDifferenceDoesNotEstablishAudibility,
        ProgSuiteAudioSurvivalNonClaimV1::NativeWaveformDifferenceDoesNotEstablishAcousticOnsetDifference,
        ProgSuiteAudioSurvivalNonClaimV1::NativeWaveformDifferenceDoesNotEstablishSpectralSalience,
        ProgSuiteAudioSurvivalNonClaimV1::NativeWaveformDifferenceDoesNotEstablishListenerPreference,
        ProgSuiteAudioSurvivalNonClaimV1::NativeWaveformDifferenceDoesNotEstablishArtisticQuality,
        ProgSuiteAudioSurvivalNonClaimV1::WholeWorkDifferenceDoesNotLocalizeTheEffect,
        ProgSuiteAudioSurvivalNonClaimV1::FixedRendererDoesNotEstablishRendererGeneralization,
        ProgSuiteAudioSurvivalNonClaimV1::LockboxPopulationDoesNotEstablishUniversalMusicGeneralization,
        ProgSuiteAudioSurvivalNonClaimV1::FrameCountsDoNotEstablishStatisticalIndependence,
        ProgSuiteAudioSurvivalNonClaimV1::ProtocolDoesNotGrantProductAuthority,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_binds_the_exact_predeclared_64_subject_lockbox() {
        let protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        protocol.validate().unwrap();
        assert_eq!(
            protocol.source_lockbox.status,
            ProgSuiteHarmonyLockboxStatusV1::PredeclaredBeforeExecution
        );
        assert_eq!(
            protocol.source_lockbox.subjects.len(),
            PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT
        );
        assert_eq!(
            protocol.analysis.expected_subject_count,
            PROG_SUITE_CONTEXTUAL_HARMONY_LOCKBOX_SUBJECT_COUNT
        );
    }

    #[test]
    fn native_renderer_policy_is_exact_and_has_no_post_processing() {
        let protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        assert_eq!(
            protocol.render_policy.renderer_id,
            PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_RENDERER_ID
        );
        assert_eq!(
            protocol.render_policy.sample_rate,
            PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_SAMPLE_RATE
        );
        assert_eq!(
            protocol.render_policy.repeated_renders_per_arm,
            PROG_SUITE_CONTEXTUAL_HARMONY_AUDIO_REPEATED_RENDERS_PER_ARM
        );
        assert_eq!(
            protocol.render_policy.post_render_processing,
            ProgSuiteAudioSurvivalPostRenderProcessingV1::NoneBeyondNativeRenderer
        );
        assert_eq!(protocol.render_policy.render_state, MusicalState::default());
    }

    #[test]
    fn renderer_erasure_is_an_admissible_observation_not_a_schema_failure() {
        let protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        assert!(protocol.admissible_outcomes.contains(
            &ProgSuiteAudioSurvivalOutcomeV1::SymbolicInterventionRendererErased
        ));
        assert!(protocol.admissible_outcomes.contains(
            &ProgSuiteAudioSurvivalOutcomeV1::SymbolicInterventionSurvivedNativeRender
        ));
    }

    #[test]
    fn frames_and_events_cannot_be_promoted_to_independent_subjects() {
        let protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        assert_eq!(
            protocol.analysis.primary_unit,
            ProgSuiteAudioSurvivalPrimaryUnitV1::MotifSeedSubject
        );
        assert!(!protocol.analysis.frame_level_inference_allowed);
        assert!(!protocol.analysis.event_level_inference_allowed);
        assert!(!protocol.analysis.outcome_tuned_threshold_allowed);
    }

    #[test]
    fn sample_rate_tampering_fails_canonical_validation() {
        let mut protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        protocol.render_policy.sample_rate = 48_000;
        assert_eq!(
            protocol.validate(),
            Err(ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1::NonCanonicalProtocol)
        );
    }

    #[test]
    fn render_state_tampering_fails_canonical_validation() {
        let mut protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        protocol.render_policy.render_state.arousal = 0.99;
        assert_eq!(
            protocol.validate(),
            Err(ProgSuiteContextualHarmonyAudioSurvivalProtocolErrorV1::NonCanonicalProtocol)
        );
    }

    #[test]
    fn nonclaim_registry_keeps_waveform_and_perception_separate() {
        let protocol =
            predeclare_prog_suite_contextual_harmony_audio_survival_protocol_v1().unwrap();
        assert!(protocol.nonclaims.contains(
            &ProgSuiteAudioSurvivalNonClaimV1::NativeWaveformDifferenceDoesNotEstablishAudibility
        ));
        assert!(protocol.nonclaims.contains(
            &ProgSuiteAudioSurvivalNonClaimV1::NativeWaveformDifferenceDoesNotEstablishArtisticQuality
        ));
        assert!(protocol.nonclaims.contains(
            &ProgSuiteAudioSurvivalNonClaimV1::WholeWorkDifferenceDoesNotLocalizeTheEffect
        ));
    }
}
