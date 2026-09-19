// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Browser-independent playback state machine.
//!
//! Leptos and `<audio>` are adapters around this reducer. Browser events carry
//! the load epoch they observed; late events from a superseded source are
//! ignored deterministically.

use symthaea_muse_protocol::RenditionArtifactId;

/// What kind of musical subject the shared transport is currently auditioning.
/// This is deliberately transport/presentation authority, not canonical piece
/// identity: a review artifact can be audible without becoming `MuseState::current`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaybackSubjectKind {
    JourneyCandidate,
    CreatedCandidate,
    Review,
}

/// Human-facing metadata and action capabilities for the exact audible source.
/// Keeping this inside `PlaybackSource` makes the player/header change atomically
/// with the load-epoch-protected audio source instead of following stale
/// `MuseState::current` state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaybackPresentation {
    pub kind: PlaybackSubjectKind,
    pub title: String,
    pub subtitle: Option<String>,
    pub style_hint: Option<String>,
}

impl PlaybackPresentation {
    pub fn journey_candidate(title: String, subtitle: String, style: String) -> Self {
        Self {
            kind: PlaybackSubjectKind::JourneyCandidate,
            title,
            subtitle: Some(subtitle),
            style_hint: Some(style),
        }
    }

    pub fn created_candidate(title: String, subtitle: String, style: String) -> Self {
        Self {
            kind: PlaybackSubjectKind::CreatedCandidate,
            title,
            subtitle: Some(subtitle),
            style_hint: Some(style),
        }
    }

    pub fn review(title: String) -> Self {
        Self {
            kind: PlaybackSubjectKind::Review,
            title,
            subtitle: Some("Review audition · canonical piece unchanged".to_string()),
            style_hint: None,
        }
    }

    pub fn can_keep(&self) -> bool {
        matches!(
            self.kind,
            PlaybackSubjectKind::JourneyCandidate | PlaybackSubjectKind::CreatedCandidate
        )
    }

    pub fn can_advance_journey(&self) -> bool {
        self.kind == PlaybackSubjectKind::JourneyCandidate
    }

    pub fn can_change_renderer(&self) -> bool {
        matches!(
            self.kind,
            PlaybackSubjectKind::JourneyCandidate | PlaybackSubjectKind::CreatedCandidate
        )
    }

    pub fn palette_style(&self) -> &str {
        self.style_hint.as_deref().unwrap_or("Review")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlaybackSource {
    pub rendition_id: Option<RenditionArtifactId>,
    pub audio_url: String,
    pub duration_hint_seconds: Option<f64>,
    /// Listen journeys advance on completion. Review and authored-candidate
    /// auditions do not implicitly enter/advance the Listen journey.
    pub advance_on_end: bool,
    pub presentation: PlaybackPresentation,
}

/// The exact source/position being temporarily left behind for an audition.
/// This is transport state only: saving a bookmark never mutates canonical
/// piece identity or journey state.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaybackBookmark {
    pub source: PlaybackSource,
    pub position_seconds: f64,
    pub resume_playing: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlaybackPhase {
    Empty,
    Loading,
    Ready,
    Playing,
    Paused,
    Seeking,
    Ended,
    Failed,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlaybackState {
    pub phase: PlaybackPhase,
    pub source: Option<PlaybackSource>,
    /// One-level reversible audition bookmark. Nested review auditions preserve
    /// the original source instead of turning each review into a new canonical
    /// return point.
    pub return_bookmark: Option<PlaybackBookmark>,
    pub load_epoch: u64,
    pub position_seconds: f64,
    pub duration_seconds: Option<f64>,
    pub autoplay_pending: bool,
    resume_after_seek: bool,
    restore_position_after_metadata: Option<f64>,
    play_after_restore_seek: bool,
    pub error: Option<String>,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            phase: PlaybackPhase::Empty,
            source: None,
            return_bookmark: None,
            load_epoch: 0,
            position_seconds: 0.0,
            duration_seconds: None,
            autoplay_pending: false,
            resume_after_seek: false,
            restore_position_after_metadata: None,
            play_after_restore_seek: false,
            error: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PlaybackEvent {
    /// Replace the transport's authoritative source. This is used for normal
    /// candidate activation and intentionally invalidates any older audition
    /// bookmark.
    LoadRequested {
        source: PlaybackSource,
        autoplay: bool,
    },
    /// Temporarily audition a source while preserving the first source being
    /// left behind. Repeated/nested auditions keep the original bookmark.
    AuditionRequested {
        source: PlaybackSource,
        autoplay: bool,
    },
    /// Return to the source captured by `AuditionRequested`, restoring its
    /// position after metadata is available and resuming only if it had been
    /// playing before the audition.
    ReturnToBookmarkedSource,
    MetadataLoaded {
        load_epoch: u64,
        duration_seconds: f64,
    },
    PlayRequested,
    PlaybackStarted {
        load_epoch: u64,
    },
    PauseRequested,
    PlaybackPaused {
        load_epoch: u64,
    },
    SeekRequested {
        seconds: f64,
    },
    SeekCompleted {
        load_epoch: u64,
        seconds: f64,
    },
    TimeAdvanced {
        load_epoch: u64,
        seconds: f64,
    },
    Ended {
        load_epoch: u64,
    },
    PlaybackFailed {
        load_epoch: u64,
        message: String,
    },
    AutoplayRejected {
        load_epoch: u64,
    },
    SourceSuperseded,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PlaybackEffect {
    Load { load_epoch: u64, audio_url: String },
    Play { load_epoch: u64 },
    Pause { load_epoch: u64 },
    Seek { load_epoch: u64, seconds: f64 },
    AdvanceJourney,
}

impl PlaybackState {
    pub fn reduce(&mut self, event: PlaybackEvent) -> Vec<PlaybackEffect> {
        match event {
            PlaybackEvent::LoadRequested { source, autoplay } => {
                self.return_bookmark = None;
                self.load_source(source, autoplay)
            }
            PlaybackEvent::AuditionRequested { source, autoplay } => {
                if self.return_bookmark.is_none()
                    && let Some(current) = self.source.clone()
                {
                    self.return_bookmark = Some(PlaybackBookmark {
                        source: current,
                        position_seconds: self.position_seconds,
                        resume_playing: self.phase == PlaybackPhase::Playing,
                    });
                }
                self.load_source(source, autoplay)
            }
            PlaybackEvent::ReturnToBookmarkedSource => {
                let Some(bookmark) = self.return_bookmark.take() else {
                    return Vec::new();
                };
                let position_seconds = bookmark.position_seconds;
                let resume_playing = bookmark.resume_playing;
                let effects = self.load_source(bookmark.source, false);
                self.restore_position_after_metadata = Some(position_seconds);
                self.play_after_restore_seek = resume_playing;
                effects
            }
            PlaybackEvent::MetadataLoaded {
                load_epoch,
                duration_seconds,
            } if self.accepts(load_epoch) => {
                if duration_seconds.is_finite() && duration_seconds > 0.0 {
                    self.duration_seconds = Some(duration_seconds);
                }
                if let Some(seconds) = self.restore_position_after_metadata.take() {
                    let seconds = self.clamp_position(seconds);
                    self.phase = PlaybackPhase::Seeking;
                    self.position_seconds = seconds;
                    self.autoplay_pending = false;
                    self.resume_after_seek = false;
                    return vec![PlaybackEffect::Seek {
                        load_epoch,
                        seconds,
                    }];
                }
                self.phase = PlaybackPhase::Ready;
                if self.autoplay_pending {
                    vec![PlaybackEffect::Play { load_epoch }]
                } else {
                    Vec::new()
                }
            }
            PlaybackEvent::PlayRequested if self.source.is_some() => {
                self.autoplay_pending = false;
                vec![PlaybackEffect::Play {
                    load_epoch: self.load_epoch,
                }]
            }
            PlaybackEvent::PlaybackStarted { load_epoch } if self.accepts(load_epoch) => {
                self.phase = PlaybackPhase::Playing;
                self.autoplay_pending = false;
                self.error = None;
                Vec::new()
            }
            PlaybackEvent::PauseRequested if self.source.is_some() => {
                vec![PlaybackEffect::Pause {
                    load_epoch: self.load_epoch,
                }]
            }
            PlaybackEvent::PlaybackPaused { load_epoch } if self.accepts(load_epoch) => {
                if self.phase != PlaybackPhase::Ended {
                    self.phase = PlaybackPhase::Paused;
                }
                Vec::new()
            }
            PlaybackEvent::SeekRequested { seconds } if self.source.is_some() => {
                let seconds = self.clamp_position(seconds);
                self.restore_position_after_metadata = None;
                self.play_after_restore_seek = false;
                self.resume_after_seek = self.phase == PlaybackPhase::Playing;
                self.phase = PlaybackPhase::Seeking;
                self.position_seconds = seconds;
                vec![PlaybackEffect::Seek {
                    load_epoch: self.load_epoch,
                    seconds,
                }]
            }
            PlaybackEvent::SeekCompleted {
                load_epoch,
                seconds,
            } if self.accepts(load_epoch) => {
                self.position_seconds = self.clamp_position(seconds);
                if self.play_after_restore_seek {
                    self.play_after_restore_seek = false;
                    self.resume_after_seek = false;
                    self.phase = PlaybackPhase::Ready;
                    vec![PlaybackEffect::Play { load_epoch }]
                } else {
                    self.phase = if self.resume_after_seek {
                        PlaybackPhase::Playing
                    } else {
                        PlaybackPhase::Paused
                    };
                    self.resume_after_seek = false;
                    Vec::new()
                }
            }
            PlaybackEvent::TimeAdvanced {
                load_epoch,
                seconds,
            } if self.accepts(load_epoch) => {
                self.position_seconds = self.clamp_position(seconds);
                Vec::new()
            }
            PlaybackEvent::Ended { load_epoch }
                if self.accepts(load_epoch) && self.phase != PlaybackPhase::Ended =>
            {
                self.phase = PlaybackPhase::Ended;
                if let Some(duration) = self.duration_seconds {
                    self.position_seconds = duration;
                }
                if self
                    .source
                    .as_ref()
                    .is_some_and(|source| source.advance_on_end)
                {
                    vec![PlaybackEffect::AdvanceJourney]
                } else {
                    Vec::new()
                }
            }
            PlaybackEvent::PlaybackFailed {
                load_epoch,
                message,
            } if self.accepts(load_epoch) => {
                self.phase = PlaybackPhase::Failed;
                self.autoplay_pending = false;
                self.restore_position_after_metadata = None;
                self.play_after_restore_seek = false;
                self.error = Some(message);
                Vec::new()
            }
            PlaybackEvent::AutoplayRejected { load_epoch } if self.accepts(load_epoch) => {
                self.phase = PlaybackPhase::Ready;
                self.autoplay_pending = false;
                self.error = Some("browser blocked autoplay; press play to continue".into());
                Vec::new()
            }
            PlaybackEvent::SourceSuperseded => {
                self.load_epoch = self.load_epoch.wrapping_add(1).max(1);
                self.phase = PlaybackPhase::Empty;
                self.source = None;
                self.return_bookmark = None;
                self.position_seconds = 0.0;
                self.duration_seconds = None;
                self.autoplay_pending = false;
                self.resume_after_seek = false;
                self.restore_position_after_metadata = None;
                self.play_after_restore_seek = false;
                self.error = None;
                Vec::new()
            }
            _ => Vec::new(),
        }
    }

    fn load_source(&mut self, source: PlaybackSource, autoplay: bool) -> Vec<PlaybackEffect> {
        self.load_epoch = self.load_epoch.wrapping_add(1).max(1);
        self.phase = PlaybackPhase::Loading;
        self.position_seconds = 0.0;
        self.duration_seconds = source.duration_hint_seconds.filter(|v| *v > 0.0);
        self.autoplay_pending = autoplay;
        self.resume_after_seek = false;
        self.restore_position_after_metadata = None;
        self.play_after_restore_seek = false;
        self.error = None;
        let effect = PlaybackEffect::Load {
            load_epoch: self.load_epoch,
            audio_url: source.audio_url.clone(),
        };
        self.source = Some(source);
        vec![effect]
    }

    fn accepts(&self, load_epoch: u64) -> bool {
        self.source.is_some() && load_epoch == self.load_epoch
    }

    fn clamp_position(&self, seconds: f64) -> f64 {
        let seconds = if seconds.is_finite() {
            seconds.max(0.0)
        } else {
            0.0
        };
        self.duration_seconds
            .map(|duration| seconds.min(duration))
            .unwrap_or(seconds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(id: &str) -> PlaybackSource {
        PlaybackSource {
            rendition_id: Some(RenditionArtifactId(id.into())),
            audio_url: format!("/audio/{id}"),
            duration_hint_seconds: None,
            advance_on_end: true,
            presentation: PlaybackPresentation::journey_candidate(
                format!("Piece {id}"),
                "Classical · 4/4".into(),
                "Classical".into(),
            ),
        }
    }

    fn review(id: &str) -> PlaybackSource {
        PlaybackSource {
            rendition_id: None,
            audio_url: format!("/review/{id}"),
            duration_hint_seconds: None,
            advance_on_end: false,
            presentation: PlaybackPresentation::review(format!("Review {id}")),
        }
    }

    #[test]
    fn playback_capabilities_follow_audible_subject_kind() {
        let journey = PlaybackPresentation::journey_candidate(
            "Journey".into(),
            "Classical".into(),
            "Classical".into(),
        );
        assert!(journey.can_keep());
        assert!(journey.can_advance_journey());
        assert!(journey.can_change_renderer());

        let created = PlaybackPresentation::created_candidate(
            "Created".into(),
            "Sonata".into(),
            "Sonata".into(),
        );
        assert!(created.can_keep());
        assert!(!created.can_advance_journey());
        assert!(created.can_change_renderer());

        let review = PlaybackPresentation::review("Imported work".into());
        assert!(!review.can_keep());
        assert!(!review.can_advance_journey());
        assert!(!review.can_change_renderer());
        assert_eq!(review.palette_style(), "Review");
    }

    #[test]
    fn stale_events_from_replaced_source_are_ignored() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: source("a"),
            autoplay: true,
        });
        let old_epoch = state.load_epoch;
        state.reduce(PlaybackEvent::LoadRequested {
            source: source("b"),
            autoplay: false,
        });
        assert_ne!(old_epoch, state.load_epoch);
        assert!(
            state
                .reduce(PlaybackEvent::Ended {
                    load_epoch: old_epoch
                })
                .is_empty()
        );
        assert_eq!(state.phase, PlaybackPhase::Loading);
        assert_eq!(
            state
                .source
                .as_ref()
                .unwrap()
                .rendition_id
                .as_ref()
                .unwrap()
                .0,
            "b"
        );
    }

    #[test]
    fn duplicate_ended_advances_once() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: source("a"),
            autoplay: false,
        });
        let epoch = state.load_epoch;
        state.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: epoch,
            duration_seconds: 20.0,
        });
        state.reduce(PlaybackEvent::PlaybackStarted { load_epoch: epoch });
        assert_eq!(
            state.reduce(PlaybackEvent::Ended { load_epoch: epoch }),
            vec![PlaybackEffect::AdvanceJourney]
        );
        assert!(
            state
                .reduce(PlaybackEvent::Ended { load_epoch: epoch })
                .is_empty()
        );
    }

    #[test]
    fn review_audition_ends_without_advancing_the_listen_journey() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::AuditionRequested {
            source: review("review"),
            autoplay: true,
        });
        let epoch = state.load_epoch;
        state.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: epoch,
            duration_seconds: 30.0,
        });
        state.reduce(PlaybackEvent::PlaybackStarted { load_epoch: epoch });
        assert!(
            state
                .reduce(PlaybackEvent::Ended { load_epoch: epoch })
                .is_empty()
        );
        assert_eq!(state.phase, PlaybackPhase::Ended);
    }

    #[test]
    fn audition_bookmark_restores_source_position_and_play_state() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: source("original"),
            autoplay: false,
        });
        let original_epoch = state.load_epoch;
        state.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: original_epoch,
            duration_seconds: 30.0,
        });
        state.reduce(PlaybackEvent::PlaybackStarted {
            load_epoch: original_epoch,
        });
        state.reduce(PlaybackEvent::TimeAdvanced {
            load_epoch: original_epoch,
            seconds: 12.5,
        });

        state.reduce(PlaybackEvent::AuditionRequested {
            source: review("candidate"),
            autoplay: true,
        });
        assert_eq!(
            state.return_bookmark.as_ref().map(|b| b.position_seconds),
            Some(12.5)
        );
        assert!(state
            .return_bookmark
            .as_ref()
            .is_some_and(|b| b.resume_playing));
        assert_eq!(
            state.source.as_ref().map(|s| s.audio_url.as_str()),
            Some("/review/candidate")
        );

        let audition_epoch = state.load_epoch;
        assert!(
            state
                .reduce(PlaybackEvent::Ended {
                    load_epoch: original_epoch
                })
                .is_empty()
        );
        assert_eq!(state.load_epoch, audition_epoch);

        assert_eq!(
            state.reduce(PlaybackEvent::ReturnToBookmarkedSource),
            vec![PlaybackEffect::Load {
                load_epoch: audition_epoch + 1,
                audio_url: "/audio/original".into(),
            }]
        );
        let return_epoch = state.load_epoch;
        assert!(state.return_bookmark.is_none());
        assert_eq!(
            state.source.as_ref().map(|s| s.audio_url.as_str()),
            Some("/audio/original")
        );

        assert_eq!(
            state.reduce(PlaybackEvent::MetadataLoaded {
                load_epoch: return_epoch,
                duration_seconds: 30.0,
            }),
            vec![PlaybackEffect::Seek {
                load_epoch: return_epoch,
                seconds: 12.5,
            }]
        );
        assert_eq!(state.phase, PlaybackPhase::Seeking);
        assert_eq!(
            state.reduce(PlaybackEvent::SeekCompleted {
                load_epoch: return_epoch,
                seconds: 12.5,
            }),
            vec![PlaybackEffect::Play {
                load_epoch: return_epoch,
            }]
        );
        assert_eq!(state.phase, PlaybackPhase::Ready);
        state.reduce(PlaybackEvent::PlaybackStarted {
            load_epoch: return_epoch,
        });
        assert_eq!(state.phase, PlaybackPhase::Playing);
        assert_eq!(state.position_seconds, 12.5);
    }

    #[test]
    fn paused_source_returns_paused_at_saved_position() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: source("original"),
            autoplay: false,
        });
        let original_epoch = state.load_epoch;
        state.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: original_epoch,
            duration_seconds: 30.0,
        });
        state.reduce(PlaybackEvent::TimeAdvanced {
            load_epoch: original_epoch,
            seconds: 7.0,
        });
        state.reduce(PlaybackEvent::AuditionRequested {
            source: review("candidate"),
            autoplay: true,
        });
        state.reduce(PlaybackEvent::ReturnToBookmarkedSource);
        let return_epoch = state.load_epoch;
        state.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: return_epoch,
            duration_seconds: 30.0,
        });
        assert!(
            state
                .reduce(PlaybackEvent::SeekCompleted {
                    load_epoch: return_epoch,
                    seconds: 7.0,
                })
                .is_empty()
        );
        assert_eq!(state.phase, PlaybackPhase::Paused);
        assert_eq!(state.position_seconds, 7.0);
    }

    #[test]
    fn nested_auditions_preserve_the_original_return_point() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: source("original"),
            autoplay: false,
        });
        let original_epoch = state.load_epoch;
        state.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: original_epoch,
            duration_seconds: 30.0,
        });
        state.reduce(PlaybackEvent::TimeAdvanced {
            load_epoch: original_epoch,
            seconds: 9.0,
        });
        state.reduce(PlaybackEvent::AuditionRequested {
            source: review("one"),
            autoplay: true,
        });
        state.reduce(PlaybackEvent::AuditionRequested {
            source: review("two"),
            autoplay: true,
        });

        let bookmark = state.return_bookmark.as_ref().unwrap();
        assert_eq!(bookmark.source.audio_url, "/audio/original");
        assert_eq!(bookmark.position_seconds, 9.0);
    }

    #[test]
    fn authoritative_load_clears_an_old_audition_bookmark() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: source("original"),
            autoplay: false,
        });
        state.reduce(PlaybackEvent::AuditionRequested {
            source: review("candidate"),
            autoplay: true,
        });
        assert!(state.return_bookmark.is_some());

        state.reduce(PlaybackEvent::LoadRequested {
            source: source("replacement"),
            autoplay: false,
        });
        assert!(state.return_bookmark.is_none());
        assert!(state
            .reduce(PlaybackEvent::ReturnToBookmarkedSource)
            .is_empty());
        assert_eq!(
            state.source.as_ref().map(|s| s.audio_url.as_str()),
            Some("/audio/replacement")
        );
    }

    #[test]
    fn seeking_while_playing_preserves_playing_state() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: source("a"),
            autoplay: false,
        });
        let epoch = state.load_epoch;
        state.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: epoch,
            duration_seconds: 30.0,
        });
        state.reduce(PlaybackEvent::PlaybackStarted { load_epoch: epoch });
        state.reduce(PlaybackEvent::SeekRequested { seconds: 12.0 });
        state.reduce(PlaybackEvent::SeekCompleted {
            load_epoch: epoch,
            seconds: 12.0,
        });
        assert_eq!(state.phase, PlaybackPhase::Playing);
    }

    #[test]
    fn seeking_before_metadata_is_safe_and_later_clamped() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: source("a"),
            autoplay: false,
        });
        let epoch = state.load_epoch;
        assert_eq!(
            state.reduce(PlaybackEvent::SeekRequested { seconds: 30.0 }),
            vec![PlaybackEffect::Seek {
                load_epoch: epoch,
                seconds: 30.0
            }]
        );
        state.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: epoch,
            duration_seconds: 10.0,
        });
        state.reduce(PlaybackEvent::TimeAdvanced {
            load_epoch: epoch,
            seconds: 30.0,
        });
        assert_eq!(state.position_seconds, 10.0);
    }

    #[test]
    fn autoplay_rejection_recovers_to_explicit_play() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: source("a"),
            autoplay: true,
        });
        let epoch = state.load_epoch;
        state.reduce(PlaybackEvent::AutoplayRejected { load_epoch: epoch });
        assert_eq!(state.phase, PlaybackPhase::Ready);
        assert_eq!(
            state.reduce(PlaybackEvent::PlayRequested),
            vec![PlaybackEffect::Play { load_epoch: epoch }]
        );
        state.reduce(PlaybackEvent::PlaybackStarted { load_epoch: epoch });
        assert_eq!(state.phase, PlaybackPhase::Playing);
    }
}
