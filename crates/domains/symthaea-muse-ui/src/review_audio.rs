// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Review-audio playback adapter for non-canonical audition surfaces.
//!
//! Imported works, teaching etudes, qualification artifacts, and future
//! Studio alternatives may be auditioned without becoming the current Listen
//! candidate. Review sources therefore carry no fabricated rendition identity
//! and never advance the Listen journey when playback ends.

use crate::playback::{PlaybackEvent, PlaybackSource};
use crate::state::MuseState;

fn review_source(audio_url: String) -> PlaybackSource {
    PlaybackSource {
        rendition_id: None,
        audio_url,
        duration_hint_seconds: None,
        advance_on_end: false,
    }
}

impl MuseState {
    /// Play an auxiliary/review artifact through the shared transport without
    /// replacing the current canonical candidate or advancing Listen when it
    /// finishes.
    ///
    /// `title` is intentionally accepted for the existing review call sites
    /// but is not promoted into global piece identity. A later audition-state
    /// tranche will give review subjects their own presentation metadata.
    pub fn play_review_audio(self, audio_url: String, _title: String) {
        self.dispatch(PlaybackEvent::LoadRequested {
            source: review_source(audio_url),
            autoplay: true,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playback::{PlaybackPhase, PlaybackState};

    #[test]
    fn review_source_has_no_fabricated_rendition_identity() {
        let source = review_source("/review/example.wav".into());
        assert!(source.rendition_id.is_none());
        assert_eq!(source.audio_url, "/review/example.wav");
        assert!(source.duration_hint_seconds.is_none());
        assert!(!source.advance_on_end);
    }

    #[test]
    fn review_source_ends_without_advancing_the_listen_journey() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: review_source("/review/example.wav".into()),
            autoplay: true,
        });
        let load_epoch = state.load_epoch;
        state.reduce(PlaybackEvent::MetadataLoaded {
            load_epoch,
            duration_seconds: 12.0,
        });
        state.reduce(PlaybackEvent::PlaybackStarted { load_epoch });

        assert!(state.reduce(PlaybackEvent::Ended { load_epoch }).is_empty());
        assert_eq!(state.phase, PlaybackPhase::Ended);
    }
}
