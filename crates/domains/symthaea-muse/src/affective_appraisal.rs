// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Typed, evidence-bearing affective appraisal for musical self-listening.
//!
//! This is a functional research core, not a claim that Melothaea has
//! phenomenal feelings. It separates:
//! - an estimate of affect expressed by the heard music;
//! - a human report of affect induced in that listener;
//! - Melothaea's computational appraisal of an independently perceived stream;
//! - a human preference judgement about exact audio artifacts.
//!
//! IndependentPerceptualObservationV1 contains no composer-intended valence
//! or arousal. A requested emotional target must not be echoed back as proof
//! that the audio expressed that emotion. goal_congruence is a post-perception
//! comparison against an explicit current goal; callers must not derive it
//! from the intended affect label itself.
//!
//! The default profile is an inspectable heuristic prior, not a human-trained
//! model. Its weights require blinded listener calibration before claims about
//! human emotion or preference are warranted.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A bounded valence/arousal/dominance-like coordinate.
///
/// This is a computational representation, not evidence of subjective feeling.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AffectVectorV1 {
    /// Unpleasant to pleasant, in [-1, 1].
    pub valence: f32,
    /// Deactivated to activated, in [0, 1].
    pub arousal: f32,
    /// Low perceived agency/control to high, in [-1, 1].
    pub dominance: f32,
}

impl AffectVectorV1 {
    pub const NEUTRAL: Self = Self {
        valence: 0.0,
        arousal: 0.5,
        dominance: 0.0,
    };

    pub fn validate(&self) -> Result<(), AppraisalError> {
        bounded("affect.valence", self.valence, -1.0, 1.0)?;
        bounded("affect.arousal", self.arousal, 0.0, 1.0)?;
        bounded("affect.dominance", self.dominance, -1.0, 1.0)
    }
}

impl Default for AffectVectorV1 {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

/// Content-addressed identity for an independently produced perceptual result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PerceptualEvidenceRefV1 {
    pub observation_id: String,
    /// Stable identity of the independent perception pipeline.
    pub perceptual_pipeline_id: String,
    /// Immutable pipeline/model version used to produce this observation.
    pub perceptual_pipeline_version: String,
    /// SHA-256 of the canonical perceptual-analysis record. The producer must
    /// compute this from the serialized observation; validate() checks format
    /// only and does not have access to the source bytes.
    pub analysis_sha256: String,
    /// SHA-256 of the exact audio bytes that were perceived. The producer must
    /// compute this from those bytes; validate() checks format only.
    pub audio_sha256: String,
    /// Optional SHA-256 of the exact symbolic score used for alignment.
    pub score_sha256: Option<String>,
    /// Strictly increasing within one appraisal stream.
    pub sequence: u64,
}

impl PerceptualEvidenceRefV1 {
    pub fn validate(&self) -> Result<(), AppraisalError> {
        if self.observation_id.trim().is_empty() {
            return Err(AppraisalError::EmptyIdentifier("observation_id"));
        }
        if self.perceptual_pipeline_id.trim().is_empty() {
            return Err(AppraisalError::EmptyIdentifier("perceptual_pipeline_id"));
        }
        if self.perceptual_pipeline_version.trim().is_empty() {
            return Err(AppraisalError::EmptyIdentifier("perceptual_pipeline_version"));
        }
        validate_sha256("analysis_sha256", &self.analysis_sha256)?;
        validate_sha256("audio_sha256", &self.audio_sha256)?;
        if let Some(score_sha256) = &self.score_sha256 {
            validate_sha256("score_sha256", score_sha256)?;
        }
        Ok(())
    }
}

/// Independent evidence about what the audio perceptually contains.
///
/// Values are normalized to [0, 1]. These are observations/predictions from
/// the perception path, not composer intentions or human self-reports.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndependentPerceptualObservationV1 {
    pub evidence: PerceptualEvidenceRefV1,
    pub novelty: f32,
    pub prediction_error: f32,
    pub uncertainty: f32,
    pub tension: f32,
    pub anticipation: f32,
    pub resolution: f32,
    /// Match between the heard content and a declared non-emotional goal.
    pub goal_congruence: f32,
    /// Confidence in the audio/perceptual analysis itself.
    pub perceptual_confidence: f32,
}

impl IndependentPerceptualObservationV1 {
    pub fn validate(&self) -> Result<(), AppraisalError> {
        self.evidence.validate()?;
        for (name, value) in [
            ("novelty", self.novelty),
            ("prediction_error", self.prediction_error),
            ("uncertainty", self.uncertainty),
            ("tension", self.tension),
            ("anticipation", self.anticipation),
            ("resolution", self.resolution),
            ("goal_congruence", self.goal_congruence),
            ("perceptual_confidence", self.perceptual_confidence),
        ] {
            bounded(name, value, 0.0, 1.0)?;
        }
        Ok(())
    }
}

/// Versioned, explicit appraisal parameters.
///
/// Weight vectors are normalized at use time, so their scale does not affect
/// results. Change the version when interpretation or default weights change.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppraisalProfileV1 {
    pub profile_id: String,
    pub version: u32,
    pub smoothing_alpha: f32,
    pub valence_goal_weight: f32,
    pub valence_resolution_weight: f32,
    pub valence_tension_weight: f32,
    pub valence_uncertainty_weight: f32,
    pub arousal_novelty_weight: f32,
    pub arousal_surprise_weight: f32,
    pub arousal_tension_weight: f32,
    pub arousal_anticipation_weight: f32,
}

impl AppraisalProfileV1 {
    /// Transparent initial prior. These weights are not human-calibrated.
    pub fn heuristic_v1() -> Self {
        Self {
            profile_id: "melothaea-affect-heuristic".into(),
            version: 1,
            smoothing_alpha: 0.20,
            valence_goal_weight: 0.30,
            valence_resolution_weight: 0.20,
            valence_tension_weight: 0.30,
            valence_uncertainty_weight: 0.20,
            arousal_novelty_weight: 0.20,
            arousal_surprise_weight: 0.25,
            arousal_tension_weight: 0.25,
            arousal_anticipation_weight: 0.30,
        }
    }

    pub fn validate(&self) -> Result<(), AppraisalError> {
        if self.profile_id.trim().is_empty() {
            return Err(AppraisalError::EmptyIdentifier("profile_id"));
        }
        if self.version == 0 {
            return Err(AppraisalError::InvalidVersion);
        }
        bounded("smoothing_alpha", self.smoothing_alpha, 0.0, 1.0)?;
        let weights = [
            ("valence_goal_weight", self.valence_goal_weight),
            ("valence_resolution_weight", self.valence_resolution_weight),
            ("valence_tension_weight", self.valence_tension_weight),
            ("valence_uncertainty_weight", self.valence_uncertainty_weight),
            ("arousal_novelty_weight", self.arousal_novelty_weight),
            ("arousal_surprise_weight", self.arousal_surprise_weight),
            ("arousal_tension_weight", self.arousal_tension_weight),
            ("arousal_anticipation_weight", self.arousal_anticipation_weight),
        ];
        for (name, weight) in weights {
            bounded(name, weight, 0.0, 1.0)?;
        }
        let valence_sum = self.valence_goal_weight
            + self.valence_resolution_weight
            + self.valence_tension_weight
            + self.valence_uncertainty_weight;
        let arousal_sum = self.arousal_novelty_weight
            + self.arousal_surprise_weight
            + self.arousal_tension_weight
            + self.arousal_anticipation_weight;
        if valence_sum <= f32::EPSILON || arousal_sum <= f32::EPSILON {
            return Err(AppraisalError::InvalidWeights);
        }
        Ok(())
    }
}

impl Default for AppraisalProfileV1 {
    fn default() -> Self {
        Self::heuristic_v1()
    }
}

/// Melothaea's functional appraisal state—not a human report or a claim of
/// subjective experience.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MelothaeaAppraisalV1 {
    pub affect: AffectVectorV1,
    pub curiosity: f32,
    pub uncertainty: f32,
    pub familiarity: f32,
    pub tension: f32,
    pub anticipation: f32,
    pub resolution: f32,
    pub confidence: f32,
    pub last_observation_sequence: Option<u64>,
}

impl Default for MelothaeaAppraisalV1 {
    fn default() -> Self {
        Self {
            affect: AffectVectorV1::NEUTRAL,
            curiosity: 0.0,
            uncertainty: 0.5,
            familiarity: 0.5,
            tension: 0.0,
            anticipation: 0.0,
            resolution: 0.0,
            confidence: 0.0,
            last_observation_sequence: None,
        }
    }
}

impl MelothaeaAppraisalV1 {
    /// Validate a current or deserialized appraisal state before reuse.
    pub fn validate(&self) -> Result<(), AppraisalError> {
        self.affect.validate()?;
        for (name, value) in [
            ("curiosity", self.curiosity),
            ("uncertainty", self.uncertainty),
            ("familiarity", self.familiarity),
            ("tension", self.tension),
            ("anticipation", self.anticipation),
            ("resolution", self.resolution),
            ("confidence", self.confidence),
        ] {
            bounded(name, value, 0.0, 1.0)?;
        }
        Ok(())
    }

    /// Update functional appraisal from audio-derived evidence.
    ///
    /// Composer intent is not an input. The anti-circularity boundary is
    /// therefore enforced by the public API rather than only by convention.
    pub fn update(
        &mut self,
        observation: &IndependentPerceptualObservationV1,
        profile: &AppraisalProfileV1,
    ) -> Result<AppraisalTransitionV1, AppraisalError> {
        self.validate()?;
        observation.validate()?;
        profile.validate()?;
        if self
            .last_observation_sequence
            .is_some_and(|last| observation.evidence.sequence <= last)
        {
            return Err(AppraisalError::NonIncreasingSequence);
        }

        let before = self.clone();
        let valence_weight_sum = profile.valence_goal_weight
            + profile.valence_resolution_weight
            + profile.valence_tension_weight
            + profile.valence_uncertainty_weight;
        let valence_target = (
            profile.valence_goal_weight * observation.goal_congruence
                + profile.valence_resolution_weight * observation.resolution
                - profile.valence_tension_weight * observation.tension
                - profile.valence_uncertainty_weight * observation.uncertainty
        ) / valence_weight_sum;

        let arousal_weight_sum = profile.arousal_novelty_weight
            + profile.arousal_surprise_weight
            + profile.arousal_tension_weight
            + profile.arousal_anticipation_weight;
        let arousal_target = (
            profile.arousal_novelty_weight * observation.novelty
                + profile.arousal_surprise_weight * observation.prediction_error
                + profile.arousal_tension_weight * observation.tension
                + profile.arousal_anticipation_weight * observation.anticipation
        ) / arousal_weight_sum;

        let dominance_target = observation.goal_congruence - observation.uncertainty;
        let alpha = profile.smoothing_alpha;
        self.affect.valence = ema(self.affect.valence, valence_target, alpha);
        self.affect.arousal = ema(self.affect.arousal, arousal_target, alpha);
        self.affect.dominance = ema(self.affect.dominance, dominance_target, alpha);
        self.curiosity = ema(
            self.curiosity,
            observation.novelty * (1.0 - observation.uncertainty),
            alpha,
        );
        self.uncertainty = ema(self.uncertainty, observation.uncertainty, alpha);
        self.familiarity = ema(self.familiarity, 1.0 - observation.novelty, alpha);
        self.tension = ema(self.tension, observation.tension, alpha);
        self.anticipation = ema(self.anticipation, observation.anticipation, alpha);
        self.resolution = ema(self.resolution, observation.resolution, alpha);
        self.confidence = ema(
            self.confidence,
            observation.perceptual_confidence * (1.0 - observation.uncertainty),
            alpha,
        );
        self.last_observation_sequence = Some(observation.evidence.sequence);

        // Clamp only floating-point round-off; input/profile values are
        // validated above and never silently repaired.
        self.affect.valence = self.affect.valence.clamp(-1.0, 1.0);
        self.affect.arousal = self.affect.arousal.clamp(0.0, 1.0);
        self.affect.dominance = self.affect.dominance.clamp(-1.0, 1.0);
        self.curiosity = self.curiosity.clamp(0.0, 1.0);
        self.uncertainty = self.uncertainty.clamp(0.0, 1.0);
        self.familiarity = self.familiarity.clamp(0.0, 1.0);
        self.tension = self.tension.clamp(0.0, 1.0);
        self.anticipation = self.anticipation.clamp(0.0, 1.0);
        self.resolution = self.resolution.clamp(0.0, 1.0);
        self.confidence = self.confidence.clamp(0.0, 1.0);
        self.affect.validate()?;

        Ok(AppraisalTransitionV1 {
            observation: observation.evidence.clone(),
            profile_id: profile.profile_id.clone(),
            profile_version: profile.version,
            before,
            after: self.clone(),
        })
    }
}

/// One auditable appraisal transition, suitable for an append-only journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppraisalTransitionV1 {
    pub observation: PerceptualEvidenceRefV1,
    pub profile_id: String,
    pub profile_version: u32,
    pub before: MelothaeaAppraisalV1,
    pub after: MelothaeaAppraisalV1,
}

/// An independent estimate of affect expressed by the music artifact.
///
/// This describes the perceived expression, not what a listener felt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpressedAffectEstimateV1 {
    pub estimate: AffectVectorV1,
    pub confidence: f32,
    pub evidence: PerceptualEvidenceRefV1,
}

impl ExpressedAffectEstimateV1 {
    pub fn validate(&self) -> Result<(), AppraisalError> {
        self.estimate.validate()?;
        bounded("expressed.confidence", self.confidence, 0.0, 1.0)?;
        self.evidence.validate()
    }
}

/// A human's report of their own affect after hearing a particular artifact.
///
/// This must be collected from the participant; a model prediction must never
/// be relabeled as a human observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanInducedAffectReportV1 {
    pub participant_pseudonym: String,
    pub affect: AffectVectorV1,
    pub confidence: f32,
    pub audio_sha256: String,
    pub report_time_ms: u64,
}

impl HumanInducedAffectReportV1 {
    pub fn validate(&self) -> Result<(), AppraisalError> {
        if self.participant_pseudonym.trim().is_empty() {
            return Err(AppraisalError::EmptyIdentifier("participant_pseudonym"));
        }
        self.affect.validate()?;
        bounded("human_report.confidence", self.confidence, 0.0, 1.0)?;
        validate_sha256("audio_sha256", &self.audio_sha256)
    }
}

/// Human preference is a different target from emotion recognition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanPreferenceDecisionV1 {
    PreferA,
    PreferB,
    Tie,
    NeitherAcceptable,
}

/// A blinded human judgement bound to the exact pair of audio artifacts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanPreferenceJudgmentV1 {
    pub participant_pseudonym: String,
    pub decision: HumanPreferenceDecisionV1,
    pub audio_a_sha256: String,
    pub audio_b_sha256: String,
}

impl HumanPreferenceJudgmentV1 {
    pub fn validate(&self) -> Result<(), AppraisalError> {
        if self.participant_pseudonym.trim().is_empty() {
            return Err(AppraisalError::EmptyIdentifier("participant_pseudonym"));
        }
        validate_sha256("audio_a_sha256", &self.audio_a_sha256)?;
        validate_sha256("audio_b_sha256", &self.audio_b_sha256)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AppraisalError {
    EmptyIdentifier(&'static str),
    InvalidDigest(&'static str),
    NonFinite(&'static str),
    OutOfRange(&'static str),
    InvalidVersion,
    InvalidWeights,
    NonIncreasingSequence,
}

impl fmt::Display for AppraisalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(name) => write!(f, "{name} must not be empty"),
            Self::InvalidDigest(name) => {
                write!(f, "{name} must be a 64-character SHA-256 hex digest")
            }
            Self::NonFinite(name) => write!(f, "{name} must be finite"),
            Self::OutOfRange(name) => write!(f, "{name} is outside its declared range"),
            Self::InvalidVersion => {
                write!(f, "appraisal profile version must be greater than zero")
            }
            Self::InvalidWeights => {
                write!(f, "each appraisal weight group must have a positive sum")
            }
            Self::NonIncreasingSequence => {
                write!(f, "perceptual observation sequence must increase strictly")
            }
        }
    }
}

impl std::error::Error for AppraisalError {}

fn bounded(name: &'static str, value: f32, min: f32, max: f32) -> Result<(), AppraisalError> {
    if !value.is_finite() {
        return Err(AppraisalError::NonFinite(name));
    }
    if !(min..=max).contains(&value) {
        return Err(AppraisalError::OutOfRange(name));
    }
    Ok(())
}

fn validate_sha256(name: &'static str, value: &str) -> Result<(), AppraisalError> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(AppraisalError::InvalidDigest(name));
    }
    Ok(())
}

fn ema(previous: f32, current: f32, alpha: f32) -> f32 {
    previous + alpha * (current - previous)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(sequence: u64) -> IndependentPerceptualObservationV1 {
        IndependentPerceptualObservationV1 {
            evidence: PerceptualEvidenceRefV1 {
                observation_id: format!("listen-{sequence}"),
                perceptual_pipeline_id: "muse-audio-perception".into(),
                perceptual_pipeline_version: "0.1.0".into(),
                analysis_sha256: "c".repeat(64),
                audio_sha256: "a".repeat(64),
                score_sha256: Some("b".repeat(64)),
                sequence,
            },
            novelty: 0.8,
            prediction_error: 0.7,
            uncertainty: 0.2,
            tension: 0.6,
            anticipation: 0.5,
            resolution: 0.8,
            goal_congruence: 0.9,
            perceptual_confidence: 0.95,
        }
    }

    #[test]
    fn appraisal_is_bounded_and_records_exact_evidence() {
        let mut state = MelothaeaAppraisalV1::default();
        let result = state
            .update(&observation(1), &AppraisalProfileV1::default())
            .unwrap();
        assert_eq!(result.observation.sequence, 1);
        assert_eq!(result.profile_version, 1);
        assert_eq!(result.after, state);
        assert!(state.affect.validate().is_ok());
        for value in [
            state.curiosity,
            state.uncertainty,
            state.familiarity,
            state.tension,
            state.anticipation,
            state.resolution,
            state.confidence,
            state.affect.arousal,
        ] {
            assert!((0.0..=1.0).contains(&value));
        }
        assert!((-1.0..=1.0).contains(&state.affect.valence));
        assert!((-1.0..=1.0).contains(&state.affect.dominance));
    }

    #[test]
    fn identical_perceptual_streams_produce_identical_appraisal() {
        // Composer targets are not represented in the observation schema:
        // only audio-derived evidence and explicit non-emotional goal
        // congruence are accepted by the appraisal API.
        let mut first = MelothaeaAppraisalV1::default();
        let mut second = MelothaeaAppraisalV1::default();
        let heard = observation(1);
        let profile = AppraisalProfileV1::default();
        first.update(&heard, &profile).unwrap();
        second.update(&heard, &profile).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn malformed_persisted_state_fails_closed() {
        let mut state = MelothaeaAppraisalV1::default();
        state.curiosity = f32::NAN;
        assert_eq!(
            state.update(&observation(1), &AppraisalProfileV1::default()),
            Err(AppraisalError::NonFinite("curiosity"))
        );
    }

    #[test]
    fn observations_must_be_strictly_ordered() {
        let mut state = MelothaeaAppraisalV1::default();
        let profile = AppraisalProfileV1::default();
        state.update(&observation(4), &profile).unwrap();
        assert_eq!(
            state.update(&observation(4), &profile),
            Err(AppraisalError::NonIncreasingSequence)
        );
        assert_eq!(
            state.update(&observation(3), &profile),
            Err(AppraisalError::NonIncreasingSequence)
        );
    }

    #[test]
    fn non_finite_and_out_of_range_inputs_fail_closed() {
        let mut bad = observation(1);
        bad.tension = f32::NAN;
        assert_eq!(bad.validate(), Err(AppraisalError::NonFinite("tension")));

        let mut bad = observation(1);
        bad.novelty = 1.1;
        assert_eq!(bad.validate(), Err(AppraisalError::OutOfRange("novelty")));
    }

    #[test]
    fn perceptual_pipeline_identity_is_required() {
        let mut bad = observation(1);
        bad.evidence.perceptual_pipeline_version = " ".into();
        assert_eq!(
            bad.validate(),
            Err(AppraisalError::EmptyIdentifier("perceptual_pipeline_version"))
        );
    }

    #[test]
    fn invalid_analysis_identity_fails_closed() {
        let mut bad = observation(1);
        bad.evidence.analysis_sha256 = "missing-digest".into();
        assert_eq!(
            bad.validate(),
            Err(AppraisalError::InvalidDigest("analysis_sha256"))
        );
    }

    #[test]
    fn invalid_audio_identity_fails_closed() {
        let mut bad = observation(1);
        bad.evidence.audio_sha256 = "not-a-digest".into();
        assert_eq!(
            bad.validate(),
            Err(AppraisalError::InvalidDigest("audio_sha256"))
        );
    }

    #[test]
    fn a_zero_weight_profile_is_rejected() {
        let mut profile = AppraisalProfileV1::default();
        profile.valence_goal_weight = 0.0;
        profile.valence_resolution_weight = 0.0;
        profile.valence_tension_weight = 0.0;
        profile.valence_uncertainty_weight = 0.0;
        assert_eq!(profile.validate(), Err(AppraisalError::InvalidWeights));
    }

    #[test]
    fn distinct_human_and_system_targets_have_distinct_wire_types() {
        let estimate = ExpressedAffectEstimateV1 {
            estimate: AffectVectorV1::NEUTRAL,
            confidence: 0.8,
            evidence: observation(1).evidence,
        };
        let report = HumanInducedAffectReportV1 {
            participant_pseudonym: "listener-001".into(),
            affect: AffectVectorV1::NEUTRAL,
            confidence: 0.8,
            audio_sha256: "a".repeat(64),
            report_time_ms: 1,
        };
        let judgment = HumanPreferenceJudgmentV1 {
            participant_pseudonym: "listener-001".into(),
            decision: HumanPreferenceDecisionV1::Tie,
            audio_a_sha256: "a".repeat(64),
            audio_b_sha256: "b".repeat(64),
        };
        // Compile-time type separation: these records cannot be supplied to
        // MelothaeaAppraisalV1::update, which accepts only independent audio
        // perception observations.
        assert_eq!(estimate.confidence, report.confidence);
        assert_eq!(judgment.decision, HumanPreferenceDecisionV1::Tie);
        assert!(estimate.validate().is_ok());
        assert!(report.validate().is_ok());
        assert!(judgment.validate().is_ok());

        let mut invalid_report = report;
        invalid_report.participant_pseudonym = " ".into();
        assert_eq!(
            invalid_report.validate(),
            Err(AppraisalError::EmptyIdentifier("participant_pseudonym"))
        );

        let mut invalid_judgment = judgment;
        invalid_judgment.audio_b_sha256 = "missing-digest".into();
        assert_eq!(
            invalid_judgment.validate(),
            Err(AppraisalError::InvalidDigest("audio_b_sha256"))
        );
    }
}
