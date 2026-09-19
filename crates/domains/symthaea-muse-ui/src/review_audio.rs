// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Review-audio playback adapter for non-canonical audition surfaces.
//!
//! Imported works, teaching etudes, qualification artifacts, and future
//! Studio alternatives may be auditioned without becoming the current Listen
//! candidate. Review sources therefore carry no fabricated rendition identity,
//! never advance the Listen journey, and carry their own player presentation.

use crate::playback::{PlaybackEvent, PlaybackPresentation, PlaybackSource};
use crate::state::MuseState;

fn review_source(audio_url: String, title: String) -> PlaybackSource {
    PlaybackSource {
        rendition_id: None,
        audio_url,
        duration_hint_seconds: None,
        advance_on_end: false,
        presentation: PlaybackPresentation::review(title),
    }
}

impl MuseState {
    /// Temporarily audition an auxiliary/review artifact through the shared
    /// transport without replacing the canonical candidate or advancing Listen.
    ///
    /// `AuditionRequested` preserves the first source being left behind, along
    /// with its position/play state, so a later return can restore that exact
    /// transport context under a fresh load epoch.
    pub fn play_review_audio(self, audio_url: String, title: String) {
        self.dispatch(PlaybackEvent::AuditionRequested {
            source: review_source(audio_url, title),
            autoplay: true,
        });
    }

    /// Return from a temporary review audition to the transport source that was
    /// active before the first audition. No-op when there is no bookmark.
    pub fn return_from_audition(self) {
        self.dispatch(PlaybackEvent::ReturnToBookmarkedSource);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playback::{PlaybackPhase, PlaybackState, PlaybackSubjectKind};

    #[test]
    fn review_source_has_no_fabricated_rendition_identity_or_candidate_capabilities() {
        let source = review_source("/review/example.wav".into(), "Imported work".into());
        assert!(source.rendition_id.is_none());
        assert_eq!(source.audio_url, "/review/example.wav");
        assert!(source.duration_hint_seconds.is_none());
        assert!(!source.advance_on_end);
        assert_eq!(source.presentation.kind, PlaybackSubjectKind::Review);
        assert_eq!(source.presentation.title, "Imported work");
        assert!(!source.presentation.can_keep());
        assert!(!source.presentation.can_advance_journey());
        assert!(!source.presentation.can_change_renderer());
    }

    #[test]
    fn review_source_ends_without_advancing_the_listen_journey() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::AuditionRequested {
            source: review_source("/review/example.wav".into(), "Review".into()),
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
