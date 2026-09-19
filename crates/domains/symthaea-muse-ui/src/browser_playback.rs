// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Admission boundary between browser `<audio>` callbacks and the pure playback
//! reducer.
//!
//! Every browser callback must name the current load epoch. For callbacks whose
//! wrong-phase acceptance can corrupt transport semantics (`pause`, `seeked`,
//! and `ended`), this module additionally requires the corresponding playback
//! phase. Keeping those browser-specific rules outside `playback.rs` preserves
//! the reducer as a reusable state machine while making the DOM adapter fail
//! closed against same-epoch callback noise.

use leptos::prelude::*;

use crate::playback::{PlaybackEvent, PlaybackPhase, PlaybackState};
use crate::state::MuseState;

/// Dispatch one callback originating from the persistent browser media element
/// only when its epoch and, where required, transport phase are admissible.
pub fn dispatch_browser_event(muse: MuseState, event: PlaybackEvent) {
    if browser_event_is_admissible(&muse.playback.get_untracked(), &event) {
        muse.dispatch(event);
    }
}

fn browser_event_is_admissible(state: &PlaybackState, event: &PlaybackEvent) -> bool {
    let current_epoch = |epoch: u64| state.source.is_some() && epoch == state.load_epoch;

    match event {
        // These callbacks are already idempotent/source-scoped in the reducer.
        // Do not over-constrain them by phase: `play()` may be requested while
        // metadata is still loading, and the browser controls the exact order
        // in which the resulting metadata/play/time callbacks arrive.
        PlaybackEvent::MetadataLoaded { load_epoch, .. }
        | PlaybackEvent::PlaybackStarted { load_epoch }
        | PlaybackEvent::TimeAdvanced { load_epoch, .. }
        | PlaybackEvent::PlaybackFailed { load_epoch, .. } => current_epoch(*load_epoch),

        // A pause callback is meaningful only after playback actually began or
        // while a user-initiated seek from playing state is in flight. In
        // particular, a pause emitted around source replacement must not turn a
        // newly Loading source into Paused.
        PlaybackEvent::PlaybackPaused { load_epoch } => {
            current_epoch(*load_epoch)
                && matches!(state.phase, PlaybackPhase::Playing | PlaybackPhase::Seeking)
        }

        // `seeked` completes an explicit reducer-owned seek. Accepting it in
        // Ready/Paused/Loading would let an unsolicited browser callback mutate
        // phase and position as though the app had requested a seek.
        PlaybackEvent::SeekCompleted { load_epoch, .. } => {
            current_epoch(*load_epoch) && state.phase == PlaybackPhase::Seeking
        }

        // Journey auto-advance must represent actual playback reaching the end,
        // not a loading/paused/failed callback or a manual seek to the duration.
        PlaybackEvent::Ended { load_epoch } => {
            current_epoch(*load_epoch) && state.phase == PlaybackPhase::Playing
        }

        // These are application/reducer commands, not browser media callbacks.
        PlaybackEvent::LoadRequested { .. }
        | PlaybackEvent::AuditionRequested { .. }
        | PlaybackEvent::ReturnToBookmarkedSource
        | PlaybackEvent::PlayRequested
        | PlaybackEvent::PauseRequested
        | PlaybackEvent::SeekRequested { .. }
        | PlaybackEvent::AutoplayRejected { .. }
        | PlaybackEvent::SourceSuperseded => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playback::{PlaybackPresentation, PlaybackSource};
    use symthaea_muse_protocol::RenditionArtifactId;

    fn source() -> PlaybackSource {
        PlaybackSource {
            rendition_id: Some(RenditionArtifactId("a".repeat(64))),
            audio_url: "/audio/a".into(),
            duration_hint_seconds: None,
            advance_on_end: true,
            presentation: PlaybackPresentation::journey_candidate(
                "Journey".into(),
                "Classical · 4/4".into(),
                "Classical".into(),
            ),
        }
    }

    fn loading_state() -> PlaybackState {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: source(),
            autoplay: false,
        });
        state
    }

    #[test]
    fn every_browser_event_requires_the_current_epoch() {
        let state = loading_state();
        let stale = state.load_epoch + 1;
        assert!(!browser_event_is_admissible(
            &state,
            &PlaybackEvent::MetadataLoaded {
                load_epoch: stale,
                duration_seconds: 30.0,
            }
        ));
        assert!(!browser_event_is_admissible(
            &state,
            &PlaybackEvent::PlaybackStarted { load_epoch: stale }
        ));
        assert!(!browser_event_is_admissible(
            &state,
            &PlaybackEvent::TimeAdvanced {
                load_epoch: stale,
                seconds: 1.0,
            }
        ));
        assert!(!browser_event_is_admissible(
            &state,
            &PlaybackEvent::PlaybackFailed {
                load_epoch: stale,
                message: "stale".into(),
            }
        ));
    }

    #[test]
    fn pause_is_rejected_while_loading_and_accepted_while_playing() {
        let mut state = loading_state();
        let epoch = state.load_epoch;
        let paused = PlaybackEvent::PlaybackPaused { load_epoch: epoch };
        assert!(!browser_event_is_admissible(&state, &paused));

        state.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: epoch,
            duration_seconds: 30.0,
        });
        state.reduce(PlaybackEvent::PlaybackStarted { load_epoch: epoch });
        assert_eq!(state.phase, PlaybackPhase::Playing);
        assert!(browser_event_is_admissible(&state, &paused));
    }

    #[test]
    fn seek_completion_requires_an_active_seek() {
        let mut state = loading_state();
        let epoch = state.load_epoch;
        state.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: epoch,
            duration_seconds: 30.0,
        });
        let seeked = PlaybackEvent::SeekCompleted {
            load_epoch: epoch,
            seconds: 8.0,
        };
        assert!(!browser_event_is_admissible(&state, &seeked));

        state.reduce(PlaybackEvent::SeekRequested { seconds: 8.0 });
        assert_eq!(state.phase, PlaybackPhase::Seeking);
        assert!(browser_event_is_admissible(&state, &seeked));
    }

    #[test]
    fn ended_requires_actual_playback_and_current_epoch() {
        let mut state = loading_state();
        let epoch = state.load_epoch;
        state.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch: epoch,
            duration_seconds: 30.0,
        });
        let ended = PlaybackEvent::Ended { load_epoch: epoch };
        assert!(!browser_event_is_admissible(&state, &ended));

        state.reduce(PlaybackEvent::PlaybackStarted { load_epoch: epoch });
        assert_eq!(state.phase, PlaybackPhase::Playing);
        assert!(browser_event_is_admissible(&state, &ended));
        assert!(!browser_event_is_admissible(
            &state,
            &PlaybackEvent::Ended {
                load_epoch: epoch + 1,
            }
        ));
    }

    #[test]
    fn application_commands_never_cross_the_browser_boundary() {
        let state = loading_state();
        assert!(!browser_event_is_admissible(
            &state,
            &PlaybackEvent::PlayRequested
        ));
        assert!(!browser_event_is_admissible(
            &state,
            &PlaybackEvent::SourceSuperseded
        ));
    }
}
