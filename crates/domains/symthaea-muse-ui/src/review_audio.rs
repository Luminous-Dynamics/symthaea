// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Review-audio playback adapter for non-canonical audition surfaces.
//!
//! Imported works, teaching etudes, qualification artifacts, persisted keepers,
//! and future Studio alternatives may be auditioned without becoming the current
//! Listen candidate. Review sources never advance the Listen journey and carry
//! their own player presentation. A rendition identity is present only when a
//! caller has already established a real content-addressed hash; ordinary review
//! audio remains identity-less rather than fabricating one.

use leptos::prelude::*;
use symthaea_muse_protocol::RenditionArtifactId;
use web_sys::HtmlAudioElement;

use crate::playback::{
    PlaybackEvent, PlaybackPresentation, PlaybackSource, PlaybackState, PlaybackSubjectKind,
};
use crate::state::MuseState;

fn review_source(
    audio_url: String,
    title: String,
    rendition_id: Option<RenditionArtifactId>,
) -> PlaybackSource {
    PlaybackSource {
        rendition_id,
        audio_url,
        duration_hint_seconds: None,
        advance_on_end: false,
        presentation: PlaybackPresentation::review(title),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReviewExitAction {
    ReturnToBookmark,
    ClearUnbookmarkedReview,
}

fn review_exit_action(playback: &PlaybackState) -> Option<ReviewExitAction> {
    let is_review = playback
        .source
        .as_ref()
        .is_some_and(|source| source.presentation.kind == PlaybackSubjectKind::Review);
    if !is_review {
        return None;
    }
    Some(if playback.return_bookmark.is_some() {
        ReviewExitAction::ReturnToBookmark
    } else {
        ReviewExitAction::ClearUnbookmarkedReview
    })
}

impl MuseState {
    /// Temporarily audition auxiliary/review audio with no asserted rendition
    /// identity. This is the correct path for imported reconstructions,
    /// teaching etudes, and legacy persisted artifacts whose content hash has
    /// not been verified in the current client flow.
    pub fn play_review_audio(self, audio_url: String, title: String) {
        self.dispatch(PlaybackEvent::AuditionRequested {
            source: review_source(audio_url, title, None),
            autoplay: true,
        });
    }

    /// Temporarily audition review audio whose rendered bytes have already been
    /// bound to a verified content hash by an upstream evidence check (for
    /// example `keeper_api::fetch_verified_keeper_identity`). This does not make
    /// the artifact canonical; it only lets the transport carry the truthful
    /// rendition identity it has actually earned.
    pub fn play_identity_bound_review_audio(
        self,
        audio_url: String,
        title: String,
        rendition_id: RenditionArtifactId,
    ) {
        self.dispatch(PlaybackEvent::AuditionRequested {
            source: review_source(audio_url, title, Some(rendition_id)),
            autoplay: true,
        });
    }

    /// Return from a temporary review audition to the transport source that was
    /// active before the first audition. No-op when there is no bookmark.
    pub fn return_from_audition(self) {
        self.dispatch(PlaybackEvent::ReturnToBookmarkedSource);
    }

    /// Exit the exact audible review source without guessing what should become
    /// canonical. A bookmarked audition returns through the reducer's reversible
    /// path. An unbookmarked review is explicitly stopped and superseded while
    /// leaving `MuseState::current` untouched.
    pub fn exit_review_audition(self) {
        match review_exit_action(&self.playback.get_untracked()) {
            Some(ReviewExitAction::ReturnToBookmark) => self.return_from_audition(),
            Some(ReviewExitAction::ClearUnbookmarkedReview) => {
                if let Some(audio) = self.audio_ref.get_untracked() {
                    let audio: HtmlAudioElement = audio.into();
                    let _ = audio.pause();
                    let _ = audio.remove_attribute("src");
                    audio.load();
                }
                self.dispatch(PlaybackEvent::SourceSuperseded);
            }
            None => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playback::{PlaybackPhase, PlaybackState};

    fn journey_source() -> PlaybackSource {
        PlaybackSource {
            rendition_id: Some(RenditionArtifactId("a".repeat(64))),
            audio_url: "/audio/original".into(),
            duration_hint_seconds: Some(30.0),
            advance_on_end: true,
            presentation: PlaybackPresentation::journey_candidate(
                "Original".into(),
                "Classical · 4/4".into(),
                "Classical".into(),
            ),
        }
    }

    #[test]
    fn ordinary_review_source_has_no_fabricated_rendition_identity_or_candidate_capabilities() {
        let source = review_source("/review/example.wav".into(), "Imported work".into(), None);
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
    fn identity_bound_review_preserves_exact_verified_rendition_hash() {
        let expected = RenditionArtifactId("c".repeat(64));
        let source = review_source(
            "/api/keeper-audio/keeper-a".into(),
            "Keeper A".into(),
            Some(expected.clone()),
        );
        assert_eq!(source.rendition_id, Some(expected));
        assert_eq!(source.presentation.kind, PlaybackSubjectKind::Review);
        assert!(!source.advance_on_end);
    }

    #[test]
    fn review_source_ends_without_advancing_the_listen_journey() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::AuditionRequested {
            source: review_source("/review/example.wav".into(), "Review".into(), None),
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

    #[test]
    fn bookmarked_review_exits_by_returning() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: journey_source(),
            autoplay: false,
        });
        state.reduce(PlaybackEvent::AuditionRequested {
            source: review_source("/review/b.wav".into(), "Review B".into(), None),
            autoplay: true,
        });
        assert_eq!(
            review_exit_action(&state),
            Some(ReviewExitAction::ReturnToBookmark)
        );
    }

    #[test]
    fn unbookmarked_review_exits_by_clearing_only_the_review_source() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::AuditionRequested {
            source: review_source("/review/solo.wav".into(), "Solo review".into(), None),
            autoplay: true,
        });
        assert!(state.return_bookmark.is_none());
        assert_eq!(
            review_exit_action(&state),
            Some(ReviewExitAction::ClearUnbookmarkedReview)
        );
    }

    #[test]
    fn canonical_candidate_source_is_not_a_review_exit_target() {
        let mut state = PlaybackState::default();
        state.reduce(PlaybackEvent::LoadRequested {
            source: journey_source(),
            autoplay: false,
        });
        assert_eq!(review_exit_action(&state), None);
    }
}
