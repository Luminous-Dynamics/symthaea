// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-bound controller for blinded A/B comparison.
//!
//! This layer closes two gaps left intentionally open by the lower-level blind
//! primitives:
//! 1. a blind trial is bound to the durable identities of its exact subjects;
//! 2. exposure bookkeeping is advanced only from admitted playback state.
//!
//! URLs are never promoted into durable identity. A blind evidence trial requires
//! at least a verified rendition commitment for each subject. A playback-start
//! event establishes only that playback entered `Playing`; it does not establish
//! a minimum listening duration or attentive human listening.

use std::fmt;

use symthaea_muse_protocol::{ArtifactIdentity, RenditionArtifactId};

use crate::comparison::{ComparisonSession, ComparisonSide, MusicalComparisonAnchor};
use crate::comparison_blind::{
    BlindComparisonState, BlindLabel, BlindSubjectView, BlindTrialId, RevealedBlindAssignment,
};
use crate::comparison_judgment::{
    BlindComparisonChoice, BlindComparisonJudgment, BlindJudgmentError, BlindListeningExposure,
    ResolvedBlindComparisonJudgment, record_blind_judgment, resolve_blind_judgment,
};
use crate::comparison_transport::{ComparisonTransportPhase, ComparisonTransportTransaction};
use crate::playback::{PlaybackPhase, PlaybackState};

#[derive(Clone, Debug, PartialEq, Eq)]
enum BoundSubjectIdentity {
    Full(ArtifactIdentity),
    Rendition(RenditionArtifactId),
}

impl BoundSubjectIdentity {
    fn capture(
        session: &ComparisonSession,
        side: ComparisonSide,
    ) -> Result<Self, BlindControllerError> {
        let subject = session.subject(side);
        if let Some(identity) = &subject.artifact_identity {
            return Ok(Self::Full(identity.clone()));
        }
        if let Some(rendition) = &subject.source.rendition_id {
            return Ok(Self::Rendition(rendition.clone()));
        }
        Err(BlindControllerError::MissingDurableIdentity(side))
    }

    fn matches(&self, session: &ComparisonSession, side: ComparisonSide) -> bool {
        let subject = session.subject(side);
        match self {
            Self::Full(expected) => {
                if subject.artifact_identity.as_ref() != Some(expected) {
                    return false;
                }
                subject
                    .source
                    .rendition_id
                    .as_ref()
                    .is_none_or(|rendition| rendition == &expected.rendition)
            }
            Self::Rendition(expected) => subject
                .artifact_identity
                .as_ref()
                .map(|identity| &identity.rendition)
                .or(subject.source.rendition_id.as_ref())
                == Some(expected),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindTrialBinding {
    pub trial_id: BlindTrialId,
    a: BoundSubjectIdentity,
    b: BoundSubjectIdentity,
}

impl BlindTrialBinding {
    fn capture(
        trial_id: BlindTrialId,
        session: &ComparisonSession,
    ) -> Result<Self, BlindControllerError> {
        Ok(Self {
            trial_id,
            a: BoundSubjectIdentity::capture(session, ComparisonSide::A)?,
            b: BoundSubjectIdentity::capture(session, ComparisonSide::B)?,
        })
    }

    fn verify(&self, session: &ComparisonSession) -> Result<(), BlindControllerError> {
        if !self.a.matches(session, ComparisonSide::A) {
            return Err(BlindControllerError::SessionSubjectChanged(
                ComparisonSide::A,
            ));
        }
        if !self.b.matches(session, ComparisonSide::B) {
            return Err(BlindControllerError::SessionSubjectChanged(
                ComparisonSide::B,
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlindControllerError {
    MissingDurableIdentity(ComparisonSide),
    SessionSubjectChanged(ComparisonSide),
    TrialBindingCorrupted,
    PlaybackNotStarted,
    PlaybackSourceMismatch,
    TransactionNotCommitted,
    TransactionSideMismatch,
    Judgment(BlindJudgmentError),
}

impl fmt::Display for BlindControllerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingDurableIdentity(side) => write!(
                f,
                "blind evidence trial requires durable identity for comparison side {side:?}"
            ),
            Self::SessionSubjectChanged(side) => write!(
                f,
                "comparison side {side:?} no longer matches the subject bound to this blind trial"
            ),
            Self::TrialBindingCorrupted => {
                write!(f, "blind state trial id no longer matches its subject binding")
            }
            Self::PlaybackNotStarted => write!(
                f,
                "blind exposure requires admitted playing state for the active subject"
            ),
            Self::PlaybackSourceMismatch => write!(
                f,
                "audible playback source does not match the trial's active comparison subject"
            ),
            Self::TransactionNotCommitted => write!(
                f,
                "switched-side blind exposure requires a committed comparison transport transaction"
            ),
            Self::TransactionSideMismatch => write!(
                f,
                "committed comparison transaction does not match the session's active side"
            ),
            Self::Judgment(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for BlindControllerError {}

impl From<BlindJudgmentError> for BlindControllerError {
    fn from(value: BlindJudgmentError) -> Self {
        Self::Judgment(value)
    }
}

/// Owns one blind trial's assignment, durable subject binding, and admitted
/// playback-start bookkeeping.
#[derive(Debug, PartialEq)]
pub struct BlindComparisonController {
    blind: BlindComparisonState,
    binding: BlindTrialBinding,
    exposure: BlindListeningExposure,
}

impl BlindComparisonController {
    pub fn new(
        trial_id: BlindTrialId,
        session: &ComparisonSession,
        swapped: bool,
    ) -> Result<Self, BlindControllerError> {
        let binding = BlindTrialBinding::capture(trial_id, session)?;
        Ok(Self {
            blind: BlindComparisonState::new(trial_id, swapped),
            binding,
            exposure: BlindListeningExposure::default(),
        })
    }

    pub const fn trial_id(&self) -> BlindTrialId {
        self.binding.trial_id
    }

    pub const fn exposure(&self) -> BlindListeningExposure {
        self.exposure
    }

    pub fn visible_subjects(
        &self,
        session: &ComparisonSession,
    ) -> Result<[BlindSubjectView; 2], BlindControllerError> {
        self.verify_session(session)?;
        Ok(self.blind.visible_subjects(session))
    }

    pub fn record_active_playback_started(
        &mut self,
        session: &ComparisonSession,
        playback: &PlaybackState,
    ) -> Result<BlindLabel, BlindControllerError> {
        self.verify_session(session)?;
        self.verify_active_playback(session, playback)?;
        let label = self.label_for_side(session.active_side);
        self.exposure.record(label);
        Ok(label)
    }

    pub fn record_committed_switch_playback(
        &mut self,
        transaction: &ComparisonTransportTransaction,
        session: &ComparisonSession,
        playback: &PlaybackState,
    ) -> Result<BlindLabel, BlindControllerError> {
        self.verify_session(session)?;
        if transaction.phase() != ComparisonTransportPhase::Committed {
            return Err(BlindControllerError::TransactionNotCommitted);
        }
        if session.active_side != transaction.to_side() {
            return Err(BlindControllerError::TransactionSideMismatch);
        }
        self.verify_active_playback(session, playback)?;
        let label = self.label_for_side(session.active_side);
        self.exposure.record(label);
        Ok(label)
    }

    pub fn record_judgment(
        &mut self,
        session: &ComparisonSession,
        choice: BlindComparisonChoice,
        anchor: Option<MusicalComparisonAnchor>,
        self_reported_confidence: Option<f32>,
        note: String,
    ) -> Result<BlindComparisonJudgment, BlindControllerError> {
        self.verify_session(session)?;
        Ok(record_blind_judgment(
            &mut self.blind,
            self.exposure,
            choice,
            anchor,
            self_reported_confidence,
            note,
        )?)
    }

    pub fn reveal(
        &mut self,
        session: &ComparisonSession,
    ) -> Result<RevealedBlindAssignment, BlindControllerError> {
        self.verify_session(session)?;
        Ok(self.blind.reveal())
    }

    pub fn resolve_judgment(
        &self,
        session: &ComparisonSession,
        judgment: &BlindComparisonJudgment,
    ) -> Result<ResolvedBlindComparisonJudgment, BlindControllerError> {
        self.verify_session(session)?;
        Ok(resolve_blind_judgment(&self.blind, judgment)?)
    }

    fn verify_session(&self, session: &ComparisonSession) -> Result<(), BlindControllerError> {
        if self.blind.trial_id() != self.binding.trial_id {
            return Err(BlindControllerError::TrialBindingCorrupted);
        }
        self.binding.verify(session)
    }

    fn verify_active_playback(
        &self,
        session: &ComparisonSession,
        playback: &PlaybackState,
    ) -> Result<(), BlindControllerError> {
        if playback.phase != PlaybackPhase::Playing {
            return Err(BlindControllerError::PlaybackNotStarted);
        }
        if playback.source.as_ref() != Some(&session.active_subject().source) {
            return Err(BlindControllerError::PlaybackSourceMismatch);
        }
        Ok(())
    }

    fn label_for_side(&self, side: ComparisonSide) -> BlindLabel {
        if self.blind.side_for_transport(BlindLabel::A) == side {
            BlindLabel::A
        } else {
            BlindLabel::B
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::{ComparisonSubject, ComparisonTransportIntent};
    use crate::comparison_switch::ComparisonSwitchPlan;
    use crate::playback::{
        PlaybackEvent, PlaybackPresentation, PlaybackSource, PlaybackSubjectKind,
    };
    use symthaea_muse_protocol::{
        CompositionArtifactId, RenditionArtifactId, ScoreContentArtifactId,
    };

    const TRIAL: BlindTrialId = BlindTrialId(73);

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

    fn session(playing: bool) -> ComparisonSession {
        let a = ComparisonSubject::new(source("a", 'c'), Some(identity('a', 'b', 'c'))).unwrap();
        let b = ComparisonSubject::new(source("b", 'f'), Some(identity('d', 'e', 'f'))).unwrap();
        let mut session = ComparisonSession::new(a, b).unwrap();
        session.set_musical_anchor(Some(MusicalComparisonAnchor::new(2, 1.0).unwrap()));
        if playing {
            session.set_transport_intent(ComparisonTransportIntent::Playing);
        }
        session
    }

    fn playing_state_for_active(session: &ComparisonSession) -> PlaybackState {
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
    fn blind_trial_requires_durable_subject_identity() {
        let source = |name: &str| PlaybackSource {
            rendition_id: None,
            audio_url: format!("/legacy/{name}"),
            duration_hint_seconds: None,
            advance_on_end: false,
            presentation: PlaybackPresentation::review(name.to_string()),
        };
        let a = ComparisonSubject::new(source("a"), None).unwrap();
        let b = ComparisonSubject::new(source("b"), None).unwrap();
        let session = ComparisonSession::new(a, b).unwrap();
        assert_eq!(
            BlindComparisonController::new(TRIAL, &session, false),
            Err(BlindControllerError::MissingDurableIdentity(
                ComparisonSide::A
            ))
        );
    }

    #[test]
    fn trial_rejects_subject_or_source_identity_substitution_after_binding() {
        let mut session = session(true);
        let controller = BlindComparisonController::new(TRIAL, &session, false).unwrap();
        session.a = ComparisonSubject::new(
            source("replacement", '7'),
            Some(identity('9', '8', '7')),
        )
        .unwrap();
        assert_eq!(
            controller.visible_subjects(&session),
            Err(BlindControllerError::SessionSubjectChanged(
                ComparisonSide::A
            ))
        );

        let mut session = session(true);
        let controller = BlindComparisonController::new(TRIAL, &session, false).unwrap();
        session.a.source.rendition_id = Some(RenditionArtifactId("7".repeat(64)));
        assert_eq!(
            controller.visible_subjects(&session),
            Err(BlindControllerError::SessionSubjectChanged(
                ComparisonSide::A
            ))
        );
    }

    #[test]
    fn admitted_active_playback_records_only_the_visible_label() {
        let session = session(true);
        let playback = playing_state_for_active(&session);
        let mut controller = BlindComparisonController::new(TRIAL, &session, true).unwrap();
        assert_eq!(
            controller
                .record_active_playback_started(&session, &playback)
                .unwrap(),
            BlindLabel::B
        );
        assert_eq!(controller.exposure().visible_a_auditions, 0);
        assert_eq!(controller.exposure().visible_b_auditions, 1);
    }

    #[test]
    fn source_mismatch_cannot_create_exposure() {
        let session = session(true);
        let mut playback = playing_state_for_active(&session);
        playback.source = Some(session.b.source.clone());
        let mut controller = BlindComparisonController::new(TRIAL, &session, false).unwrap();
        assert_eq!(
            controller.record_active_playback_started(&session, &playback),
            Err(BlindControllerError::PlaybackSourceMismatch)
        );
        assert_eq!(controller.exposure(), BlindListeningExposure::default());
    }

    #[test]
    fn switched_side_exposure_requires_committed_playing_transaction() {
        let mut session = session(true);
        let mut controller = BlindComparisonController::new(TRIAL, &session, false).unwrap();
        let mut playback = playing_state_for_active(&session);
        controller
            .record_active_playback_started(&session, &playback)
            .unwrap();

        let (mut transaction, _) =
            ComparisonTransportTransaction::begin(&session, &mut playback, plan(&session)).unwrap();
        assert_eq!(
            controller.record_committed_switch_playback(&transaction, &session, &playback),
            Err(BlindControllerError::TransactionNotCommitted)
        );
        transaction.metadata_loaded(&mut playback, 30.0).unwrap();
        transaction.seek_completed(&mut playback, 5.0).unwrap();
        transaction.playback_started(&mut playback).unwrap();
        transaction.commit(&mut session, &playback).unwrap();
        assert_eq!(
            controller
                .record_committed_switch_playback(&transaction, &session, &playback)
                .unwrap(),
            BlindLabel::B
        );
        assert!(controller.exposure().both_sides_auditioned());
    }

    #[test]
    fn verified_exposure_can_feed_one_blind_judgment_then_resolve_after_reveal() {
        let mut session = session(true);
        let mut controller = BlindComparisonController::new(TRIAL, &session, true).unwrap();
        let mut playback = playing_state_for_active(&session);
        controller
            .record_active_playback_started(&session, &playback)
            .unwrap();

        let (mut transaction, _) =
            ComparisonTransportTransaction::begin(&session, &mut playback, plan(&session)).unwrap();
        transaction.metadata_loaded(&mut playback, 30.0).unwrap();
        transaction.seek_completed(&mut playback, 5.0).unwrap();
        transaction.playback_started(&mut playback).unwrap();
        transaction.commit(&mut session, &playback).unwrap();
        controller
            .record_committed_switch_playback(&transaction, &session, &playback)
            .unwrap();

        let judgment = controller
            .record_judgment(
                &session,
                BlindComparisonChoice::Prefer(BlindLabel::A),
                session.musical_anchor,
                Some(0.7),
                "blind comparison note".into(),
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
