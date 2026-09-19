// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Workspace-local routing of browser media callbacks during A/B comparison.
//!
//! This is not a second player. It drives the same pure `PlaybackState` reducer
//! used by the rest of Melothaea, while giving an in-flight semantic comparison
//! switch first right of admission to metadata/seek/play/failure callbacks.
//! Returned `PlaybackEffect`s are still executed by the existing DOM adapter.

use std::fmt;

use crate::comparison::{ComparisonSession, ComparisonSide};
use crate::comparison_blind::BlindLabel;
use crate::comparison_duration_exposure::{
    BlindExposureError, DurationQualifiedBlindController,
};
use crate::comparison_switch::ComparisonSwitchPlan;
use crate::comparison_transport::{
    ComparisonTransportError, ComparisonTransportPhase, ComparisonTransportTransaction,
};
use crate::playback::{PlaybackEffect, PlaybackEvent, PlaybackPhase, PlaybackState};

#[derive(Clone, Debug, PartialEq)]
pub enum ComparisonBrowserEvent {
    MetadataLoaded {
        load_epoch: u64,
        duration_seconds: f64,
    },
    PlaybackStarted {
        load_epoch: u64,
    },
    PlaybackPaused {
        load_epoch: u64,
    },
    SeekCompleted {
        load_epoch: u64,
        seconds: f64,
    },
    TimeAdvanced {
        load_epoch: u64,
        seconds: f64,
    },
    PlaybackFailed {
        load_epoch: u64,
        message: String,
    },
    Ended {
        load_epoch: u64,
    },
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ComparisonBrowserUpdate {
    pub effects: Vec<PlaybackEffect>,
    pub switch_committed: Option<ComparisonSide>,
    pub exposure_started: Option<BlindLabel>,
    pub progress_delta_seconds: f64,
    pub switch_failed: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ComparisonBrowserError {
    SwitchAlreadyActive,
    StaleLoadEpoch,
    UnexpectedBrowserEvent,
    JourneyAdvancingSource(ComparisonSide),
    UnexpectedReducerEffects,
    Transport(ComparisonTransportError),
    Exposure(BlindExposureError),
}

impl fmt::Display for ComparisonBrowserError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SwitchAlreadyActive => write!(f, "a comparison switch is already in flight"),
            Self::StaleLoadEpoch => write!(f, "browser comparison callback belongs to a stale load epoch"),
            Self::UnexpectedBrowserEvent => write!(f, "browser callback is not admissible in the current comparison phase"),
            Self::JourneyAdvancingSource(side) => write!(
                f,
                "comparison side {side:?} carries journey auto-advance authority; compare subjects must be transport-isolated"
            ),
            Self::UnexpectedReducerEffects => write!(
                f,
                "comparison browser callback produced effects outside the comparison contract"
            ),
            Self::Transport(error) => error.fmt(f),
            Self::Exposure(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for ComparisonBrowserError {}

impl From<ComparisonTransportError> for ComparisonBrowserError {
    fn from(value: ComparisonTransportError) -> Self {
        Self::Transport(value)
    }
}

impl From<BlindExposureError> for ComparisonBrowserError {
    fn from(value: BlindExposureError) -> Self {
        Self::Exposure(value)
    }
}

/// Local browser-callback router for one comparison workspace.
#[derive(Debug, Default, PartialEq)]
pub struct ComparisonBrowserRouter {
    transaction: Option<ComparisonTransportTransaction>,
}

impl ComparisonBrowserRouter {
    pub fn has_active_switch(&self) -> bool {
        self.transaction.is_some()
    }

    pub fn begin_switch(
        &mut self,
        session: &ComparisonSession,
        playback: &mut PlaybackState,
        plan: ComparisonSwitchPlan,
    ) -> Result<ComparisonBrowserUpdate, ComparisonBrowserError> {
        self.validate_sources(session)?;
        if self.transaction.is_some() {
            return Err(ComparisonBrowserError::SwitchAlreadyActive);
        }
        let (transaction, effects) =
            ComparisonTransportTransaction::begin(session, playback, plan)?;
        self.transaction = Some(transaction);
        Ok(ComparisonBrowserUpdate {
            effects,
            ..Default::default()
        })
    }

    pub fn handle(
        &mut self,
        session: &mut ComparisonSession,
        blind: &mut DurationQualifiedBlindController,
        playback: &mut PlaybackState,
        event: ComparisonBrowserEvent,
    ) -> Result<ComparisonBrowserUpdate, ComparisonBrowserError> {
        self.validate_sources(session)?;
        match event {
            ComparisonBrowserEvent::MetadataLoaded {
                load_epoch,
                duration_seconds,
            } => self.metadata_loaded(playback, load_epoch, duration_seconds),
            ComparisonBrowserEvent::PlaybackStarted { load_epoch } => {
                self.playback_started(session, blind, playback, load_epoch)
            }
            ComparisonBrowserEvent::PlaybackPaused { load_epoch } => {
                self.playback_paused(playback, load_epoch)
            }
            ComparisonBrowserEvent::SeekCompleted {
                load_epoch,
                seconds,
            } => self.seek_completed(session, playback, load_epoch, seconds),
            ComparisonBrowserEvent::TimeAdvanced {
                load_epoch,
                seconds,
            } => self.time_advanced(session, blind, playback, load_epoch, seconds),
            ComparisonBrowserEvent::PlaybackFailed {
                load_epoch,
                message,
            } => self.playback_failed(playback, load_epoch, message),
            ComparisonBrowserEvent::Ended { load_epoch } => self.ended(playback, load_epoch),
        }
    }

    fn metadata_loaded(
        &mut self,
        playback: &mut PlaybackState,
        load_epoch: u64,
        duration_seconds: f64,
    ) -> Result<ComparisonBrowserUpdate, ComparisonBrowserError> {
        self.ensure_epoch(playback, load_epoch)?;
        if let Some(transaction) = self.transaction.as_mut() {
            if transaction.load_epoch() != load_epoch
                || transaction.phase() != ComparisonTransportPhase::Loading
            {
                return Err(ComparisonBrowserError::UnexpectedBrowserEvent);
            }
            return Ok(ComparisonBrowserUpdate {
                effects: transaction.metadata_loaded(playback, duration_seconds)?,
                ..Default::default()
            });
        }

        let effects = playback.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch,
            duration_seconds,
        });
        Ok(ComparisonBrowserUpdate {
            effects,
            ..Default::default()
        })
    }

    fn seek_completed(
        &mut self,
        session: &mut ComparisonSession,
        playback: &mut PlaybackState,
        load_epoch: u64,
        seconds: f64,
    ) -> Result<ComparisonBrowserUpdate, ComparisonBrowserError> {
        self.ensure_epoch(playback, load_epoch)?;
        if let Some(transaction) = self.transaction.as_mut() {
            if transaction.load_epoch() != load_epoch
                || transaction.phase() != ComparisonTransportPhase::Seeking
            {
                return Err(ComparisonBrowserError::UnexpectedBrowserEvent);
            }
            let effects = transaction.seek_completed(playback, seconds)?;
            if transaction.phase() == ComparisonTransportPhase::ReadyToCommit {
                let side = transaction.to_side();
                transaction.commit(session, playback)?;
                self.transaction = None;
                return Ok(ComparisonBrowserUpdate {
                    effects,
                    switch_committed: Some(side),
                    ..Default::default()
                });
            }
            return Ok(ComparisonBrowserUpdate {
                effects,
                ..Default::default()
            });
        }

        if playback.phase != PlaybackPhase::Seeking {
            return Err(ComparisonBrowserError::UnexpectedBrowserEvent);
        }
        let effects = playback.reduce(PlaybackEvent::SeekCompleted {
            load_epoch,
            seconds,
        });
        Ok(ComparisonBrowserUpdate {
            effects,
            ..Default::default()
        })
    }

    fn playback_started(
        &mut self,
        session: &mut ComparisonSession,
        blind: &mut DurationQualifiedBlindController,
        playback: &mut PlaybackState,
        load_epoch: u64,
    ) -> Result<ComparisonBrowserUpdate, ComparisonBrowserError> {
        self.ensure_epoch(playback, load_epoch)?;
        if let Some(transaction) = self.transaction.as_mut() {
            if transaction.load_epoch() != load_epoch
                || transaction.phase() != ComparisonTransportPhase::AwaitingPlaybackStart
            {
                return Err(ComparisonBrowserError::UnexpectedBrowserEvent);
            }
            let effects = transaction.playback_started(playback)?;
            let side = transaction.to_side();
            transaction.commit(session, playback)?;
            let label = blind.record_committed_switch_playback(transaction, session, playback)?;
            self.transaction = None;
            return Ok(ComparisonBrowserUpdate {
                effects,
                switch_committed: Some(side),
                exposure_started: Some(label),
                ..Default::default()
            });
        }

        let effects = playback.reduce(PlaybackEvent::PlaybackStarted { load_epoch });
        if !effects.is_empty() {
            return Err(ComparisonBrowserError::UnexpectedReducerEffects);
        }
        let label = blind.record_active_playback_started(session, playback)?;
        Ok(ComparisonBrowserUpdate {
            exposure_started: Some(label),
            ..Default::default()
        })
    }

    fn time_advanced(
        &mut self,
        session: &ComparisonSession,
        blind: &mut DurationQualifiedBlindController,
        playback: &mut PlaybackState,
        load_epoch: u64,
        seconds: f64,
    ) -> Result<ComparisonBrowserUpdate, ComparisonBrowserError> {
        self.ensure_epoch(playback, load_epoch)?;
        if self.transaction.is_some() {
            return Err(ComparisonBrowserError::UnexpectedBrowserEvent);
        }
        let effects = playback.reduce(PlaybackEvent::TimeAdvanced {
            load_epoch,
            seconds,
        });
        if !effects.is_empty() {
            return Err(ComparisonBrowserError::UnexpectedReducerEffects);
        }
        let delta = if playback.phase == PlaybackPhase::Playing {
            blind.record_time_advanced(session, playback, load_epoch, seconds)?
        } else {
            0.0
        };
        Ok(ComparisonBrowserUpdate {
            progress_delta_seconds: delta,
            ..Default::default()
        })
    }

    fn playback_paused(
        &mut self,
        playback: &mut PlaybackState,
        load_epoch: u64,
    ) -> Result<ComparisonBrowserUpdate, ComparisonBrowserError> {
        self.ensure_epoch(playback, load_epoch)?;
        if self.transaction.is_some()
            || !matches!(playback.phase, PlaybackPhase::Playing | PlaybackPhase::Seeking)
        {
            return Err(ComparisonBrowserError::UnexpectedBrowserEvent);
        }
        let effects = playback.reduce(PlaybackEvent::PlaybackPaused { load_epoch });
        if !effects.is_empty() {
            return Err(ComparisonBrowserError::UnexpectedReducerEffects);
        }
        Ok(ComparisonBrowserUpdate::default())
    }

    fn playback_failed(
        &mut self,
        playback: &mut PlaybackState,
        load_epoch: u64,
        message: String,
    ) -> Result<ComparisonBrowserUpdate, ComparisonBrowserError> {
        self.ensure_epoch(playback, load_epoch)?;
        if let Some(transaction) = self.transaction.as_mut() {
            if transaction.load_epoch() != load_epoch {
                return Err(ComparisonBrowserError::StaleLoadEpoch);
            }
            let effects = transaction.playback_failed(playback, message)?;
            self.transaction = None;
            return Ok(ComparisonBrowserUpdate {
                effects,
                switch_failed: true,
                ..Default::default()
            });
        }
        let effects = playback.reduce(PlaybackEvent::PlaybackFailed {
            load_epoch,
            message,
        });
        if !effects.is_empty() {
            return Err(ComparisonBrowserError::UnexpectedReducerEffects);
        }
        Ok(ComparisonBrowserUpdate::default())
    }

    fn ended(
        &mut self,
        playback: &mut PlaybackState,
        load_epoch: u64,
    ) -> Result<ComparisonBrowserUpdate, ComparisonBrowserError> {
        self.ensure_epoch(playback, load_epoch)?;
        if self.transaction.is_some() || playback.phase != PlaybackPhase::Playing {
            return Err(ComparisonBrowserError::UnexpectedBrowserEvent);
        }
        let effects = playback.reduce(PlaybackEvent::Ended { load_epoch });
        // A comparison runtime must never execute Journey auto-advance. Source
        // validation above should make this impossible; keep the effect check as
        // a second fail-closed boundary.
        if !effects.is_empty() {
            return Err(ComparisonBrowserError::UnexpectedReducerEffects);
        }
        Ok(ComparisonBrowserUpdate::default())
    }

    fn ensure_epoch(
        &self,
        playback: &PlaybackState,
        load_epoch: u64,
    ) -> Result<(), ComparisonBrowserError> {
        if playback.source.is_none() || playback.load_epoch != load_epoch {
            return Err(ComparisonBrowserError::StaleLoadEpoch);
        }
        Ok(())
    }

    fn validate_sources(
        &self,
        session: &ComparisonSession,
    ) -> Result<(), ComparisonBrowserError> {
        for side in [ComparisonSide::A, ComparisonSide::B] {
            if session.subject(side).source.advance_on_end {
                return Err(ComparisonBrowserError::JourneyAdvancingSource(side));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::{
        ComparisonSubject, ComparisonTransportIntent, MusicalComparisonAnchor,
    };
    use crate::comparison_blind::BlindTrialId;
    use crate::comparison_duration_exposure::BlindExposurePolicy;
    use crate::playback::{PlaybackPresentation, PlaybackSource, PlaybackSubjectKind};
    use symthaea_muse_protocol::{
        ArtifactIdentity, CompositionArtifactId, RenditionArtifactId, ScoreContentArtifactId,
    };

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
        session.set_musical_anchor(Some(MusicalComparisonAnchor::new(2, 0.5).unwrap()));
        session.set_transport_intent(ComparisonTransportIntent::Playing);
        session
    }

    fn blind(session: &ComparisonSession) -> DurationQualifiedBlindController {
        DurationQualifiedBlindController::new(
            BlindTrialId(101),
            session,
            false,
            BlindExposurePolicy::new(1.0, 0.6).unwrap(),
        )
        .unwrap()
    }

    fn initial_playback(session: &ComparisonSession) -> PlaybackState {
        let mut playback = PlaybackState::default();
        let effects = playback.reduce(PlaybackEvent::LoadRequested {
            source: session.active_subject().source.clone(),
            autoplay: false,
        });
        assert_eq!(effects.len(), 1);
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
    fn stale_browser_epoch_is_rejected_without_advancing_state() {
        let mut session = session();
        let mut blind = blind(&session);
        let mut playback = initial_playback(&session);
        let before = playback.clone();
        let mut router = ComparisonBrowserRouter::default();
        assert_eq!(
            router.handle(
                &mut session,
                &mut blind,
                &mut playback,
                ComparisonBrowserEvent::MetadataLoaded {
                    load_epoch: before.load_epoch + 1,
                    duration_seconds: 30.0,
                },
            ),
            Err(ComparisonBrowserError::StaleLoadEpoch)
        );
        assert_eq!(playback, before);
    }

    #[test]
    fn playing_switch_commits_only_after_transactional_browser_sequence() {
        let mut session = session();
        let mut blind = blind(&session);
        let mut playback = initial_playback(&session);
        let mut router = ComparisonBrowserRouter::default();
        let epoch = playback.load_epoch;
        router
            .handle(
                &mut session,
                &mut blind,
                &mut playback,
                ComparisonBrowserEvent::MetadataLoaded {
                    load_epoch: epoch,
                    duration_seconds: 30.0,
                },
            )
            .unwrap();
        let started = router
            .handle(
                &mut session,
                &mut blind,
                &mut playback,
                ComparisonBrowserEvent::PlaybackStarted { load_epoch: epoch },
            )
            .unwrap();
        assert_eq!(started.exposure_started, Some(BlindLabel::A));

        let begin = router
            .begin_switch(&session, &mut playback, plan(&session))
            .unwrap();
        assert!(matches!(begin.effects.as_slice(), [PlaybackEffect::Load { .. }]));
        let switch_epoch = playback.load_epoch;

        let metadata = router
            .handle(
                &mut session,
                &mut blind,
                &mut playback,
                ComparisonBrowserEvent::MetadataLoaded {
                    load_epoch: switch_epoch,
                    duration_seconds: 30.0,
                },
            )
            .unwrap();
        assert!(matches!(metadata.effects.as_slice(), [PlaybackEffect::Seek { .. }]));
        assert_eq!(session.active_side, ComparisonSide::A);

        let seeked = router
            .handle(
                &mut session,
                &mut blind,
                &mut playback,
                ComparisonBrowserEvent::SeekCompleted {
                    load_epoch: switch_epoch,
                    seconds: 5.0,
                },
            )
            .unwrap();
        assert!(matches!(seeked.effects.as_slice(), [PlaybackEffect::Play { .. }]));
        assert_eq!(session.active_side, ComparisonSide::A);

        let played = router
            .handle(
                &mut session,
                &mut blind,
                &mut playback,
                ComparisonBrowserEvent::PlaybackStarted {
                    load_epoch: switch_epoch,
                },
            )
            .unwrap();
        assert_eq!(played.switch_committed, Some(ComparisonSide::B));
        assert_eq!(played.exposure_started, Some(BlindLabel::B));
        assert_eq!(session.active_side, ComparisonSide::B);
        assert!(!router.has_active_switch());
    }

    #[test]
    fn admitted_timeupdate_flows_into_duration_qualification_after_commit() {
        let mut session = session();
        let mut blind = blind(&session);
        let mut playback = initial_playback(&session);
        let mut router = ComparisonBrowserRouter::default();
        let epoch = playback.load_epoch;
        router
            .handle(
                &mut session,
                &mut blind,
                &mut playback,
                ComparisonBrowserEvent::MetadataLoaded {
                    load_epoch: epoch,
                    duration_seconds: 30.0,
                },
            )
            .unwrap();
        router
            .handle(
                &mut session,
                &mut blind,
                &mut playback,
                ComparisonBrowserEvent::PlaybackStarted { load_epoch: epoch },
            )
            .unwrap();
        let progress = router
            .handle(
                &mut session,
                &mut blind,
                &mut playback,
                ComparisonBrowserEvent::TimeAdvanced {
                    load_epoch: epoch,
                    seconds: 0.5,
                },
            )
            .unwrap();
        assert!((progress.progress_delta_seconds - 0.5).abs() < 1e-9);
    }

    #[test]
    fn comparison_runtime_rejects_sources_with_journey_auto_advance_authority() {
        let mut session = session();
        session.a.source.advance_on_end = true;
        let mut playback = initial_playback(&session);
        let mut router = ComparisonBrowserRouter::default();
        assert_eq!(
            router.begin_switch(&session, &mut playback, plan(&session)),
            Err(ComparisonBrowserError::JourneyAdvancingSource(
                ComparisonSide::A
            ))
        );
    }
}
