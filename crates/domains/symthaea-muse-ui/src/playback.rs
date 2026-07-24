// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Browser-independent playback state machine.
//!
//! Leptos and `<audio>` are adapters around this reducer. Browser events carry
//! the load epoch they observed; late events from a superseded source are
//! ignored deterministically.

use symthaea_muse_protocol::RenditionArtifactId;

#[derive(Clone, Debug, PartialEq)]
pub struct PlaybackSource {
    pub rendition_id: Option<RenditionArtifactId>,
    pub audio_url: String,
    pub duration_hint_seconds: Option<f64>,
    /// Listen journeys advance on completion. Review auditions do not.
    pub advance_on_end: bool,
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
    pub load_epoch: u64,
    pub position_seconds: f64,
    pub duration_seconds: Option<f64>,
    pub autoplay_pending: bool,
    resume_after_seek: bool,
    pub error: Option<String>,
}

impl Default for PlaybackState {
    fn default() -> Self {
        Self {
            phase: PlaybackPhase::Empty,
            source: None,
            load_epoch: 0,
            position_seconds: 0.0,
            duration_seconds: None,
            autoplay_pending: false,
            resume_after_seek: false,
            error: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PlaybackEvent {
    LoadRequested {
        source: PlaybackSource,
        autoplay: bool,
    },
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
                self.load_epoch = self.load_epoch.wrapping_add(1).max(1);
                self.phase = PlaybackPhase::Loading;
                self.position_seconds = 0.0;
                self.duration_seconds = source.duration_hint_seconds.filter(|v| *v > 0.0);
                self.autoplay_pending = autoplay;
                self.resume_after_seek = false;
                self.error = None;
                let effect = PlaybackEffect::Load {
                    load_epoch: self.load_epoch,
                    audio_url: source.audio_url.clone(),
                };
                self.source = Some(source);
                vec![effect]
            }
            PlaybackEvent::MetadataLoaded {
                load_epoch,
                duration_seconds,
            } if self.accepts(load_epoch) => {
                if duration_seconds.is_finite() && duration_seconds > 0.0 {
                    self.duration_seconds = Some(duration_seconds);
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
                self.phase = if self.resume_after_seek {
                    PlaybackPhase::Playing
                } else {
                    PlaybackPhase::Paused
                };
                self.resume_after_seek = false;
                Vec::new()
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
                self.position_seconds = 0.0;
                self.duration_seconds = None;
                self.autoplay_pending = false;
                self.resume_after_seek = false;
                self.error = None;
                Vec::new()
            }
            _ => Vec::new(),
        }
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
        }
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
        let mut review = source("review");
        review.advance_on_end = false;
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: review,
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
