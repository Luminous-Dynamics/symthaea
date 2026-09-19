// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transactional admission of an already-planned semantic A/B switch into the
//! shared playback reducer.
//!
//! CMP-003A deliberately stops before browser effects. This layer advances one
//! step: it drives the pure `PlaybackState` reducer through load -> metadata ->
//! seek -> optional play while preserving the independent reversible-audition
//! bookmark. The comparison side is committed only after that sequence has been
//! admitted successfully and the session anchor has not changed underneath it.

use std::fmt;

use crate::comparison::{ComparisonSession, ComparisonSide, MusicalComparisonAnchor};
use crate::comparison_switch::ComparisonSwitchPlan;
use crate::playback::{
    PlaybackEffect, PlaybackEvent, PlaybackPhase, PlaybackSource, PlaybackState,
};

/// HTML media seeking is not sample-accurate. This tolerance is an admission
/// bound for the browser transport path, not a claim of gapless/sample-accurate
/// comparison. A future Web Audio executor can tighten it independently.
const SEEK_POSITION_EPS_SECONDS: f64 = 0.050;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonTransportPhase {
    Loading,
    Seeking,
    AwaitingPlaybackStart,
    ReadyToCommit,
    Committed,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComparisonTransportError {
    MissingMusicalAnchor,
    SessionChanged,
    InvalidPhase,
    PlaybackEpochChanged,
    PlaybackSourceChanged,
    UnexpectedReducerEffects,
    InvalidMediaDuration,
    TargetOutsideMedia,
    SeekPositionMismatch,
}

impl fmt::Display for ComparisonTransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingMusicalAnchor => {
                write!(f, "comparison transport requires a semantic musical anchor")
            }
            Self::SessionChanged => write!(f, "comparison session changed during transport switch"),
            Self::InvalidPhase => write!(f, "comparison transport event arrived in the wrong phase"),
            Self::PlaybackEpochChanged => {
                write!(f, "comparison playback epoch was superseded before commit")
            }
            Self::PlaybackSourceChanged => {
                write!(f, "comparison playback source changed before commit")
            }
            Self::UnexpectedReducerEffects => {
                write!(f, "playback reducer produced unexpected comparison effects")
            }
            Self::InvalidMediaDuration => write!(f, "browser media duration is invalid"),
            Self::TargetOutsideMedia => {
                write!(f, "semantic comparison target lies outside browser media duration")
            }
            Self::SeekPositionMismatch => {
                write!(f, "browser seek completed outside comparison tolerance")
            }
        }
    }
}

impl std::error::Error for ComparisonTransportError {}

#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonTransportTransaction {
    plan: ComparisonSwitchPlan,
    anchor: MusicalComparisonAnchor,
    load_epoch: u64,
    phase: ComparisonTransportPhase,
}

impl ComparisonTransportTransaction {
    pub fn phase(&self) -> ComparisonTransportPhase {
        self.phase
    }

    pub fn from_side(&self) -> ComparisonSide {
        self.plan.from_side
    }

    pub fn to_side(&self) -> ComparisonSide {
        self.plan.to_side
    }

    pub fn load_epoch(&self) -> u64 {
        self.load_epoch
    }

    pub fn target_seconds(&self) -> f64 {
        self.plan.target_seconds
    }

    /// Admit the source load into the playback reducer while preserving the
    /// reversible-audition bookmark byte-for-byte.
    pub fn begin(
        session: &ComparisonSession,
        playback: &mut PlaybackState,
        plan: ComparisonSwitchPlan,
    ) -> Result<(Self, Vec<PlaybackEffect>), ComparisonTransportError> {
        if session.active_side != plan.from_side {
            return Err(ComparisonTransportError::SessionChanged);
        }
        let anchor = session
            .musical_anchor
            .ok_or(ComparisonTransportError::MissingMusicalAnchor)?;

        let preserved_bookmark = playback.return_bookmark.clone();
        let effects = playback.reduce(PlaybackEvent::LoadRequested {
            source: plan.source.clone(),
            autoplay: false,
        });
        playback.return_bookmark = preserved_bookmark;

        let load_epoch = playback.load_epoch;
        let expected = vec![PlaybackEffect::Load {
            load_epoch,
            audio_url: plan.source.audio_url.clone(),
        }];
        if effects != expected {
            return Err(ComparisonTransportError::UnexpectedReducerEffects);
        }
        ensure_source(playback, load_epoch, &plan.source)?;

        Ok((
            Self {
                plan,
                anchor,
                load_epoch,
                phase: ComparisonTransportPhase::Loading,
            },
            effects,
        ))
    }

    /// Admit browser metadata and request the exact semantic seek target.
    ///
    /// The target is checked against the real media duration *before* the
    /// playback reducer sees `SeekRequested`, preventing the reducer's generic
    /// safety clamp from silently changing comparison semantics.
    pub fn metadata_loaded(
        &mut self,
        playback: &mut PlaybackState,
        duration_seconds: f64,
    ) -> Result<Vec<PlaybackEffect>, ComparisonTransportError> {
        self.require_phase(ComparisonTransportPhase::Loading)?;
        ensure_source(playback, self.load_epoch, &self.plan.source)?;
        if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
            return Err(ComparisonTransportError::InvalidMediaDuration);
        }
        if self.plan.target_seconds >= duration_seconds {
            return Err(ComparisonTransportError::TargetOutsideMedia);
        }

        let metadata_effects = playback.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: self.load_epoch,
            duration_seconds,
        });
        if !metadata_effects.is_empty() {
            return Err(ComparisonTransportError::UnexpectedReducerEffects);
        }

        let seek_effects = playback.reduce(PlaybackEvent::SeekRequested {
            seconds: self.plan.target_seconds,
        });
        let expected = vec![PlaybackEffect::Seek {
            load_epoch: self.load_epoch,
            seconds: self.plan.target_seconds,
        }];
        if seek_effects != expected {
            return Err(ComparisonTransportError::UnexpectedReducerEffects);
        }
        self.phase = ComparisonTransportPhase::Seeking;
        Ok(seek_effects)
    }

    /// Admit the browser's seek completion. Playing comparisons request play
    /// only after the seek has succeeded; paused comparisons become committable
    /// immediately after the aligned seek.
    pub fn seek_completed(
        &mut self,
        playback: &mut PlaybackState,
        seconds: f64,
    ) -> Result<Vec<PlaybackEffect>, ComparisonTransportError> {
        self.require_phase(ComparisonTransportPhase::Seeking)?;
        ensure_source(playback, self.load_epoch, &self.plan.source)?;
        if !seconds.is_finite()
            || (seconds - self.plan.target_seconds).abs() > SEEK_POSITION_EPS_SECONDS
        {
            return Err(ComparisonTransportError::SeekPositionMismatch);
        }

        let seek_completion_effects = playback.reduce(PlaybackEvent::SeekCompleted {
            load_epoch: self.load_epoch,
            seconds,
        });
        if !seek_completion_effects.is_empty() {
            return Err(ComparisonTransportError::UnexpectedReducerEffects);
        }

        if self.plan.play_after_seek {
            let play_effects = playback.reduce(PlaybackEvent::PlayRequested);
            let expected = vec![PlaybackEffect::Play {
                load_epoch: self.load_epoch,
            }];
            if play_effects != expected {
                return Err(ComparisonTransportError::UnexpectedReducerEffects);
            }
            self.phase = ComparisonTransportPhase::AwaitingPlaybackStart;
            Ok(play_effects)
        } else {
            self.phase = ComparisonTransportPhase::ReadyToCommit;
            Ok(Vec::new())
        }
    }

    /// Admit the browser's successful `play` transition for a comparison that
    /// was playing before the side switch.
    pub fn playback_started(
        &mut self,
        playback: &mut PlaybackState,
    ) -> Result<Vec<PlaybackEffect>, ComparisonTransportError> {
        self.require_phase(ComparisonTransportPhase::AwaitingPlaybackStart)?;
        ensure_source(playback, self.load_epoch, &self.plan.source)?;
        let effects = playback.reduce(PlaybackEvent::PlaybackStarted {
            load_epoch: self.load_epoch,
        });
        if !effects.is_empty() || playback.phase != PlaybackPhase::Playing {
            return Err(ComparisonTransportError::UnexpectedReducerEffects);
        }
        self.phase = ComparisonTransportPhase::ReadyToCommit;
        Ok(effects)
    }

    /// Record a browser playback failure against the transaction's exact load
    /// epoch. Failed transactions can never commit the comparison side.
    pub fn playback_failed(
        &mut self,
        playback: &mut PlaybackState,
        message: String,
    ) -> Result<Vec<PlaybackEffect>, ComparisonTransportError> {
        if matches!(
            self.phase,
            ComparisonTransportPhase::ReadyToCommit
                | ComparisonTransportPhase::Committed
                | ComparisonTransportPhase::Failed
        ) {
            return Err(ComparisonTransportError::InvalidPhase);
        }
        ensure_source(playback, self.load_epoch, &self.plan.source)?;
        let effects = playback.reduce(PlaybackEvent::PlaybackFailed {
            load_epoch: self.load_epoch,
            message,
        });
        if !effects.is_empty() || playback.phase != PlaybackPhase::Failed {
            return Err(ComparisonTransportError::UnexpectedReducerEffects);
        }
        self.phase = ComparisonTransportPhase::Failed;
        Ok(effects)
    }

    /// Commit the A/B state transition only after transport admission succeeded
    /// and only if the session still names the same source side and musical
    /// anchor that were used to prepare the switch.
    pub fn commit(
        &mut self,
        session: &mut ComparisonSession,
        playback: &PlaybackState,
    ) -> Result<(), ComparisonTransportError> {
        self.require_phase(ComparisonTransportPhase::ReadyToCommit)?;
        ensure_source(playback, self.load_epoch, &self.plan.source)?;
        if session.active_side != self.plan.from_side || session.musical_anchor != Some(self.anchor) {
            return Err(ComparisonTransportError::SessionChanged);
        }
        session.switch_to(self.plan.to_side);
        self.phase = ComparisonTransportPhase::Committed;
        Ok(())
    }

    fn require_phase(
        &self,
        expected: ComparisonTransportPhase,
    ) -> Result<(), ComparisonTransportError> {
        if self.phase != expected {
            return Err(ComparisonTransportError::InvalidPhase);
        }
        Ok(())
    }
}

fn ensure_source(
    playback: &PlaybackState,
    expected_epoch: u64,
    expected_source: &PlaybackSource,
) -> Result<(), ComparisonTransportError> {
    if playback.load_epoch != expected_epoch {
        return Err(ComparisonTransportError::PlaybackEpochChanged);
    }
    if playback.source.as_ref() != Some(expected_source) {
        return Err(ComparisonTransportError::PlaybackSourceChanged);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::{ComparisonSubject, ComparisonTransportIntent};
    use crate::playback::{PlaybackBookmark, PlaybackPresentation, PlaybackSubjectKind};

    fn source(url: &str, title: &str) -> PlaybackSource {
        PlaybackSource {
            rendition_id: None,
            audio_url: url.to_string(),
            duration_hint_seconds: Some(30.0),
            advance_on_end: false,
            presentation: PlaybackPresentation {
                kind: PlaybackSubjectKind::Review,
                title: title.to_string(),
                subtitle: None,
                style_hint: None,
            },
        }
    }

    fn session(playing: bool) -> ComparisonSession {
        let a = ComparisonSubject::new(source("/audio/a", "A"), None).unwrap();
        let b = ComparisonSubject::new(source("/audio/b", "B"), None).unwrap();
        let mut session = ComparisonSession::new(a, b).unwrap();
        session.set_musical_anchor(Some(MusicalComparisonAnchor::new(3, 1.5).unwrap()));
        if playing {
            session.set_transport_intent(ComparisonTransportIntent::Playing);
        }
        session
    }

    fn plan(session: &ComparisonSession) -> ComparisonSwitchPlan {
        ComparisonSwitchPlan {
            from_side: ComparisonSide::A,
            to_side: ComparisonSide::B,
            source: session.b.source.clone(),
            target_seconds: 6.75,
            play_after_seek: session.transport_intent == ComparisonTransportIntent::Playing,
        }
    }

    fn bookmark() -> PlaybackBookmark {
        PlaybackBookmark {
            source: source("/audio/original", "Original"),
            position_seconds: 12.25,
            resume_playing: true,
        }
    }

    #[test]
    fn begin_preserves_reversible_audition_bookmark_exactly() {
        let session = session(false);
        let mut playback = PlaybackState::default();
        playback.return_bookmark = Some(bookmark());
        let before = playback.return_bookmark.clone();

        let (transaction, effects) =
            ComparisonTransportTransaction::begin(&session, &mut playback, plan(&session)).unwrap();

        assert_eq!(transaction.phase(), ComparisonTransportPhase::Loading);
        assert_eq!(playback.return_bookmark, before);
        assert_eq!(playback.source, Some(session.b.source.clone()));
        assert_eq!(
            effects,
            vec![PlaybackEffect::Load {
                load_epoch: transaction.load_epoch(),
                audio_url: "/audio/b".into(),
            }]
        );
    }

    #[test]
    fn metadata_rejects_out_of_media_target_instead_of_clamping() {
        let session = session(false);
        let mut playback = PlaybackState::default();
        let (mut transaction, _) =
            ComparisonTransportTransaction::begin(&session, &mut playback, plan(&session)).unwrap();

        assert_eq!(
            transaction.metadata_loaded(&mut playback, 6.0),
            Err(ComparisonTransportError::TargetOutsideMedia)
        );
        assert_eq!(transaction.phase(), ComparisonTransportPhase::Loading);
    }

    #[test]
    fn paused_switch_commits_only_after_exact_seek_completion() {
        let mut session = session(false);
        let mut playback = PlaybackState::default();
        let (mut transaction, _) =
            ComparisonTransportTransaction::begin(&session, &mut playback, plan(&session)).unwrap();

        let effects = transaction.metadata_loaded(&mut playback, 30.0).unwrap();
        assert_eq!(
            effects,
            vec![PlaybackEffect::Seek {
                load_epoch: transaction.load_epoch(),
                seconds: 6.75,
            }]
        );
        assert_eq!(session.active_side, ComparisonSide::A);

        assert!(transaction
            .seek_completed(&mut playback, 6.75)
            .unwrap()
            .is_empty());
        assert_eq!(transaction.phase(), ComparisonTransportPhase::ReadyToCommit);
        assert_eq!(session.active_side, ComparisonSide::A);

        transaction.commit(&mut session, &playback).unwrap();
        assert_eq!(transaction.phase(), ComparisonTransportPhase::Committed);
        assert_eq!(session.active_side, ComparisonSide::B);
    }

    #[test]
    fn playing_switch_waits_for_playback_started_before_commit() {
        let mut session = session(true);
        let mut playback = PlaybackState::default();
        let (mut transaction, _) =
            ComparisonTransportTransaction::begin(&session, &mut playback, plan(&session)).unwrap();
        transaction.metadata_loaded(&mut playback, 30.0).unwrap();

        let play_effects = transaction.seek_completed(&mut playback, 6.75).unwrap();
        assert_eq!(
            play_effects,
            vec![PlaybackEffect::Play {
                load_epoch: transaction.load_epoch(),
            }]
        );
        assert_eq!(
            transaction.commit(&mut session, &playback),
            Err(ComparisonTransportError::InvalidPhase)
        );

        transaction.playback_started(&mut playback).unwrap();
        transaction.commit(&mut session, &playback).unwrap();
        assert_eq!(session.active_side, ComparisonSide::B);
        assert_eq!(playback.phase, PlaybackPhase::Playing);
    }

    #[test]
    fn stale_session_anchor_blocks_async_commit() {
        let mut session = session(false);
        let mut playback = PlaybackState::default();
        let (mut transaction, _) =
            ComparisonTransportTransaction::begin(&session, &mut playback, plan(&session)).unwrap();
        transaction.metadata_loaded(&mut playback, 30.0).unwrap();
        transaction.seek_completed(&mut playback, 6.75).unwrap();

        session.set_musical_anchor(Some(MusicalComparisonAnchor::new(4, 0.0).unwrap()));
        assert_eq!(
            transaction.commit(&mut session, &playback),
            Err(ComparisonTransportError::SessionChanged)
        );
        assert_eq!(session.active_side, ComparisonSide::A);
    }

    #[test]
    fn superseded_playback_epoch_blocks_late_metadata() {
        let session = session(false);
        let mut playback = PlaybackState::default();
        let (mut transaction, _) =
            ComparisonTransportTransaction::begin(&session, &mut playback, plan(&session)).unwrap();

        playback.reduce(PlaybackEvent::LoadRequested {
            source: source("/audio/other", "Other"),
            autoplay: false,
        });
        assert_eq!(
            transaction.metadata_loaded(&mut playback, 30.0),
            Err(ComparisonTransportError::PlaybackEpochChanged)
        );
    }

    #[test]
    fn seek_completion_outside_tolerance_does_not_commit_or_resume() {
        let session = session(true);
        let mut playback = PlaybackState::default();
        let (mut transaction, _) =
            ComparisonTransportTransaction::begin(&session, &mut playback, plan(&session)).unwrap();
        transaction.metadata_loaded(&mut playback, 30.0).unwrap();

        assert_eq!(
            transaction.seek_completed(&mut playback, 6.90),
            Err(ComparisonTransportError::SeekPositionMismatch)
        );
        assert_eq!(transaction.phase(), ComparisonTransportPhase::Seeking);
    }

    #[test]
    fn playback_failure_marks_transaction_terminal() {
        let session = session(true);
        let mut playback = PlaybackState::default();
        let (mut transaction, _) =
            ComparisonTransportTransaction::begin(&session, &mut playback, plan(&session)).unwrap();

        assert!(transaction
            .playback_failed(&mut playback, "decode failed".into())
            .unwrap()
            .is_empty());
        assert_eq!(transaction.phase(), ComparisonTransportPhase::Failed);
        assert_eq!(playback.phase, PlaybackPhase::Failed);
    }
}
