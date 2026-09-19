// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Duration-aware qualification for blind A/B judgments.
//!
//! `BlindComparisonController` establishes durable subject binding and admitted
//! playback-start evidence. This wrapper adds a configurable minimum of
//! contiguous media progress per visible side before a blind judgment may be
//! recorded. It still does not claim listener attention or perceptual exposure.

use std::fmt;

use crate::comparison::{ComparisonSession, MusicalComparisonAnchor};
use crate::comparison_blind::{BlindLabel, BlindTrialId, RevealedBlindAssignment};
use crate::comparison_blind_controller::{BlindComparisonController, BlindControllerError};
use crate::comparison_judgment::{
    BlindComparisonChoice, BlindComparisonJudgment, ResolvedBlindComparisonJudgment,
};
use crate::comparison_transport::ComparisonTransportTransaction;
use crate::playback::{PlaybackPhase, PlaybackState};

const POSITION_EPS_SECONDS: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlindExposurePolicy {
    /// Minimum admitted contiguous playback progress required for each visible
    /// side before a judgment is eligible.
    pub minimum_seconds_per_side: f64,
    /// A single observed position jump larger than this is treated as a
    /// discontinuity (for example, a seek or a long unobserved gap) and adds no
    /// exposure. The cursor is reset at the new position.
    pub max_contiguous_step_seconds: f64,
}

impl BlindExposurePolicy {
    pub fn new(
        minimum_seconds_per_side: f64,
        max_contiguous_step_seconds: f64,
    ) -> Result<Self, BlindExposureError> {
        if !minimum_seconds_per_side.is_finite() || minimum_seconds_per_side <= 0.0 {
            return Err(BlindExposureError::InvalidPolicy);
        }
        if !max_contiguous_step_seconds.is_finite() || max_contiguous_step_seconds <= 0.0 {
            return Err(BlindExposureError::InvalidPolicy);
        }
        Ok(Self {
            minimum_seconds_per_side,
            max_contiguous_step_seconds,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BlindDurationEvidence {
    pub visible_a_seconds: f64,
    pub visible_b_seconds: f64,
}

impl BlindDurationEvidence {
    pub fn qualifies(self, policy: BlindExposurePolicy) -> bool {
        self.visible_a_seconds + POSITION_EPS_SECONDS >= policy.minimum_seconds_per_side
            && self.visible_b_seconds + POSITION_EPS_SECONDS >= policy.minimum_seconds_per_side
    }

    fn add_capped(&mut self, label: BlindLabel, delta: f64, cap: f64) {
        let target = match label {
            BlindLabel::A => &mut self.visible_a_seconds,
            BlindLabel::B => &mut self.visible_b_seconds,
        };
        *target = (*target + delta).min(cap);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ProgressCursor {
    label: BlindLabel,
    load_epoch: u64,
    position_seconds: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BlindExposureError {
    InvalidPolicy,
    Controller(BlindControllerError),
    PlaybackStartNotRecorded,
    PlaybackNotPlaying,
    PlaybackEpochMismatch,
    PlaybackSourceMismatch,
    PlaybackPositionMismatch,
    InvalidPlaybackPosition,
    InsufficientProgress {
        visible_a_seconds: f64,
        visible_b_seconds: f64,
        minimum_seconds_per_side: f64,
    },
}

impl fmt::Display for BlindExposureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPolicy => write!(f, "blind exposure policy values must be finite and positive"),
            Self::Controller(error) => error.fmt(f),
            Self::PlaybackStartNotRecorded => write!(
                f,
                "playback progress cannot count before playback-start admission for this side and epoch"
            ),
            Self::PlaybackNotPlaying => {
                write!(f, "blind exposure progress requires playback state Playing")
            }
            Self::PlaybackEpochMismatch => {
                write!(f, "blind exposure progress belongs to a different playback epoch")
            }
            Self::PlaybackSourceMismatch => write!(
                f,
                "blind exposure progress source does not match the active comparison subject"
            ),
            Self::PlaybackPositionMismatch => write!(
                f,
                "reported time-advance position does not match admitted playback state"
            ),
            Self::InvalidPlaybackPosition => {
                write!(f, "blind exposure progress position must be finite and non-negative")
            }
            Self::InsufficientProgress {
                visible_a_seconds,
                visible_b_seconds,
                minimum_seconds_per_side,
            } => write!(
                f,
                "blind judgment requires {minimum_seconds_per_side:.3}s per side; admitted contiguous progress is A={visible_a_seconds:.3}s B={visible_b_seconds:.3}s"
            ),
        }
    }
}

impl std::error::Error for BlindExposureError {}

impl From<BlindControllerError> for BlindExposureError {
    fn from(value: BlindControllerError) -> Self {
        Self::Controller(value)
    }
}

/// Evidence-qualified wrapper around [`BlindComparisonController`].
///
/// The browser adapter should call `record_*_playback_started` only after the
/// corresponding playback-start event has been admitted by the reducer, and
/// `record_time_advanced` only after an admitted `TimeAdvanced` event. That
/// event-origin discipline is an adapter responsibility; this pure type checks
/// the resulting epoch/source/position state before counting progress.
#[derive(Debug, PartialEq)]
pub struct DurationQualifiedBlindController {
    inner: BlindComparisonController,
    policy: BlindExposurePolicy,
    duration: BlindDurationEvidence,
    cursor: Option<ProgressCursor>,
}

impl DurationQualifiedBlindController {
    pub fn new(
        trial_id: BlindTrialId,
        session: &ComparisonSession,
        swapped: bool,
        policy: BlindExposurePolicy,
    ) -> Result<Self, BlindExposureError> {
        Ok(Self {
            inner: BlindComparisonController::new(trial_id, session, swapped)?,
            policy,
            duration: BlindDurationEvidence::default(),
            cursor: None,
        })
    }

    pub const fn trial_id(&self) -> BlindTrialId {
        self.inner.trial_id()
    }

    pub const fn policy(&self) -> BlindExposurePolicy {
        self.policy
    }

    pub const fn duration_evidence(&self) -> BlindDurationEvidence {
        self.duration
    }

    pub fn qualifies_for_judgment(&self) -> bool {
        self.duration.qualifies(self.policy)
    }

    pub fn record_active_playback_started(
        &mut self,
        session: &ComparisonSession,
        playback: &PlaybackState,
    ) -> Result<BlindLabel, BlindExposureError> {
        let label = self
            .inner
            .record_active_playback_started(session, playback)?;
        self.cursor = Some(ProgressCursor {
            label,
            load_epoch: playback.load_epoch,
            position_seconds: playback.position_seconds,
        });
        Ok(label)
    }

    pub fn record_committed_switch_playback(
        &mut self,
        transaction: &ComparisonTransportTransaction,
        session: &ComparisonSession,
        playback: &PlaybackState,
    ) -> Result<BlindLabel, BlindExposureError> {
        let label = self
            .inner
            .record_committed_switch_playback(transaction, session, playback)?;
        self.cursor = Some(ProgressCursor {
            label,
            load_epoch: playback.load_epoch,
            position_seconds: playback.position_seconds,
        });
        Ok(label)
    }

    /// Count one admitted `TimeAdvanced` observation.
    ///
    /// Non-forward samples add zero. A forward jump above the configured
    /// continuity bound also adds zero and resets the cursor to that new
    /// position; subsequent small forward progress may count from there.
    pub fn record_time_advanced(
        &mut self,
        session: &ComparisonSession,
        playback: &PlaybackState,
        load_epoch: u64,
        seconds: f64,
    ) -> Result<f64, BlindExposureError> {
        let visible = self.inner.visible_subjects(session)?;
        if playback.phase != PlaybackPhase::Playing {
            return Err(BlindExposureError::PlaybackNotPlaying);
        }
        if playback.load_epoch != load_epoch {
            return Err(BlindExposureError::PlaybackEpochMismatch);
        }
        if playback.source.as_ref() != Some(&session.active_subject().source) {
            return Err(BlindExposureError::PlaybackSourceMismatch);
        }
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(BlindExposureError::InvalidPlaybackPosition);
        }
        if (playback.position_seconds - seconds).abs() > POSITION_EPS_SECONDS {
            return Err(BlindExposureError::PlaybackPositionMismatch);
        }

        let label = visible
            .into_iter()
            .find(|view| view.active)
            .map(|view| view.label)
            .expect("comparison session always has one active side");
        let Some(mut cursor) = self.cursor else {
            return Err(BlindExposureError::PlaybackStartNotRecorded);
        };
        if cursor.label != label || cursor.load_epoch != load_epoch {
            return Err(BlindExposureError::PlaybackStartNotRecorded);
        }

        let delta = seconds - cursor.position_seconds;
        cursor.position_seconds = seconds;
        self.cursor = Some(cursor);
        if delta <= 0.0 || delta > self.policy.max_contiguous_step_seconds {
            return Ok(0.0);
        }

        self.duration
            .add_capped(label, delta, self.policy.minimum_seconds_per_side);
        Ok(delta)
    }

    pub fn record_judgment(
        &mut self,
        session: &ComparisonSession,
        choice: BlindComparisonChoice,
        anchor: Option<MusicalComparisonAnchor>,
        self_reported_confidence: Option<f32>,
        note: String,
    ) -> Result<BlindComparisonJudgment, BlindExposureError> {
        if !self.qualifies_for_judgment() {
            return Err(BlindExposureError::InsufficientProgress {
                visible_a_seconds: self.duration.visible_a_seconds,
                visible_b_seconds: self.duration.visible_b_seconds,
                minimum_seconds_per_side: self.policy.minimum_seconds_per_side,
            });
        }
        Ok(self.inner.record_judgment(
            session,
            choice,
            anchor,
            self_reported_confidence,
            note,
        )?)
    }

    pub fn reveal(
        &mut self,
        session: &ComparisonSession,
    ) -> Result<RevealedBlindAssignment, BlindExposureError> {
        Ok(self.inner.reveal(session)?)
    }

    pub fn resolve_judgment(
        &self,
        session: &ComparisonSession,
        judgment: &BlindComparisonJudgment,
    ) -> Result<ResolvedBlindComparisonJudgment, BlindExposureError> {
        Ok(self.inner.resolve_judgment(session, judgment)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::{
        ComparisonSide, ComparisonSubject, ComparisonTransportIntent,
    };
    use crate::comparison_switch::ComparisonSwitchPlan;
    use crate::comparison_transport::ComparisonTransportTransaction;
    use crate::playback::{
        PlaybackEvent, PlaybackPresentation, PlaybackSource, PlaybackSubjectKind,
    };
    use symthaea_muse_protocol::{
        ArtifactIdentity, CompositionArtifactId, RenditionArtifactId, ScoreContentArtifactId,
    };

    const TRIAL: BlindTrialId = BlindTrialId(91);

    fn policy() -> BlindExposurePolicy {
        BlindExposurePolicy::new(1.0, 0.6).unwrap()
    }

    fn identity(score: char, composition: char, rendition: char) -> ArtifactIdentity {
        ArtifactIdentity {
            score_content: ScoreContentArtifactId(score.to_string().repeat(64)),
            composition: CompositionArtifactId(composition.to_string().repeat(64)),
            rendition: RenditionArtifactId(rendition.to_string().repeat(64)),
        }
    }

    fn source(name: &str, rendition: char) -> PlaybackSource {
        PlaybackSource {
            rendition_id: Some(RenditionArtifactId(rendition.to_string().repeat(64))),
            audio_url: format!("/audio/{name}"),
            duration_hint_seconds: Some(30.0),
            advance_on_end: false,
            presentation: PlaybackPresentation {
                kind: PlaybackSubjectKind::Review,
                title: name.to_string(),
                subtitle: None,
                style_hint: None,
            },
        }
    }

    fn session() -> ComparisonSession {
        let a = ComparisonSubject::new(source("a", 'c'), Some(identity('a', 'b', 'c'))).unwrap();
        let b = ComparisonSubject::new(source("b", 'f'), Some(identity('d', 'e', 'f'))).unwrap();
        let mut session = ComparisonSession::new(a, b).unwrap();
        session.set_musical_anchor(Some(MusicalComparisonAnchor::new(1, 0.5).unwrap()));
        session.set_transport_intent(ComparisonTransportIntent::Playing);
        session
    }

    fn start_active(session: &ComparisonSession) -> PlaybackState {
        let mut playback = PlaybackState::default();
        let _ = playback.reduce(PlaybackEvent::LoadRequested {
            source: session.active_subject().source.clone(),
            autoplay: false,
        });
        let epoch = playback.load_epoch;
        let _ = playback.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: epoch,
            duration_seconds: 30.0,
        });
        let _ = playback.reduce(PlaybackEvent::PlaybackStarted { load_epoch: epoch });
        playback
    }

    fn advance(
        controller: &mut DurationQualifiedBlindController,
        session: &ComparisonSession,
        playback: &mut PlaybackState,
        seconds: f64,
    ) -> f64 {
        let epoch = playback.load_epoch;
        let _ = playback.reduce(PlaybackEvent::TimeAdvanced {
            load_epoch: epoch,
            seconds,
        });
        controller
            .record_time_advanced(session, playback, epoch, seconds)
            .unwrap()
    }

    fn plan(session: &ComparisonSession) -> ComparisonSwitchPlan {
        ComparisonSwitchPlan {
            from_side: ComparisonSide::A,
            to_side: ComparisonSide::B,
            source: session.b.source.clone(),
            target_seconds: 5.0,
            play_after_seek: true,
        }
    }

    #[test]
    fn policy_rejects_non_positive_or_non_finite_values() {
        for (minimum, step) in [
            (0.0, 0.5),
            (-1.0, 0.5),
            (1.0, 0.0),
            (1.0, f64::NAN),
            (f64::INFINITY, 0.5),
        ] {
            assert_eq!(
                BlindExposurePolicy::new(minimum, step),
                Err(BlindExposureError::InvalidPolicy)
            );
        }
    }

    #[test]
    fn progress_requires_a_recorded_playback_start_for_the_same_epoch_and_label() {
        let session = session();
        let mut playback = start_active(&session);
        let mut controller =
            DurationQualifiedBlindController::new(TRIAL, &session, false, policy()).unwrap();
        let epoch = playback.load_epoch;
        let _ = playback.reduce(PlaybackEvent::TimeAdvanced {
            load_epoch: epoch,
            seconds: 0.25,
        });
        assert_eq!(
            controller.record_time_advanced(&session, &playback, epoch, 0.25),
            Err(BlindExposureError::PlaybackStartNotRecorded)
        );
    }

    #[test]
    fn discontinuous_position_jump_adds_zero_and_resets_the_cursor() {
        let session = session();
        let mut playback = start_active(&session);
        let mut controller =
            DurationQualifiedBlindController::new(TRIAL, &session, false, policy()).unwrap();
        controller
            .record_active_playback_started(&session, &playback)
            .unwrap();

        assert!((advance(&mut controller, &session, &mut playback, 0.5) - 0.5).abs() < 1e-9);
        assert_eq!(advance(&mut controller, &session, &mut playback, 4.0), 0.0);
        assert!((advance(&mut controller, &session, &mut playback, 4.4) - 0.4).abs() < 1e-9);
        assert!((controller.duration_evidence().visible_a_seconds - 0.9).abs() < 1e-9);
    }

    #[test]
    fn judgment_remains_blocked_when_only_playback_start_counts_exist() {
        let session = session();
        let playback = start_active(&session);
        let mut controller =
            DurationQualifiedBlindController::new(TRIAL, &session, false, policy()).unwrap();
        controller
            .record_active_playback_started(&session, &playback)
            .unwrap();

        assert!(matches!(
            controller.record_judgment(
                &session,
                BlindComparisonChoice::NoPreference,
                session.musical_anchor,
                None,
                String::new(),
            ),
            Err(BlindExposureError::InsufficientProgress { .. })
        ));
    }

    #[test]
    fn both_sides_need_contiguous_progress_before_blind_judgment() {
        let mut session = session();
        let mut playback = start_active(&session);
        let mut controller =
            DurationQualifiedBlindController::new(TRIAL, &session, true, policy()).unwrap();
        controller
            .record_active_playback_started(&session, &playback)
            .unwrap();
        advance(&mut controller, &session, &mut playback, 0.5);
        advance(&mut controller, &session, &mut playback, 1.0);
        assert!(!controller.qualifies_for_judgment());

        let (mut transaction, _) =
            ComparisonTransportTransaction::begin(&session, &mut playback, plan(&session)).unwrap();
        transaction.metadata_loaded(&mut playback, 30.0).unwrap();
        transaction.seek_completed(&mut playback, 5.0).unwrap();
        transaction.playback_started(&mut playback).unwrap();
        transaction.commit(&mut session, &playback).unwrap();
        controller
            .record_committed_switch_playback(&transaction, &session, &playback)
            .unwrap();
        advance(&mut controller, &session, &mut playback, 5.5);
        advance(&mut controller, &session, &mut playback, 6.0);
        assert!(controller.qualifies_for_judgment());

        let judgment = controller
            .record_judgment(
                &session,
                BlindComparisonChoice::Prefer(BlindLabel::A),
                session.musical_anchor,
                Some(0.6),
                "duration-qualified blind report".into(),
            )
            .unwrap();
        controller.reveal(&session).unwrap();
        let resolved = controller.resolve_judgment(&session, &judgment).unwrap();
        assert_eq!(
            resolved.choice,
            crate::comparison_judgment::ResolvedComparisonChoice::Prefer(ComparisonSide::B)
        );
    }
}
