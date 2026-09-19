// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Browser-independent state for deliberate A/B comparison.
//!
//! This is intentionally separate from playback's one-level reversible
//! audition bookmark. An audition answers "temporarily hear this and return";
//! a comparison session answers "keep two explicitly named musical subjects
//! available for repeated evaluation." It owns no canonical-piece authority and
//! performs no browser/media operations.
//!
//! The session also does not equate equal wall-clock seconds with equal musical
//! position. A comparison may carry a bar/beat anchor; resolving that musical
//! coordinate into source-local seconds belongs to the later transport/timeline
//! adapter, where each subject's tempo and meter maps are available.

use std::fmt;

use symthaea_muse_protocol::ArtifactIdentity;

use crate::playback::PlaybackSource;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComparisonSide {
    #[default]
    A,
    B,
}

impl ComparisonSide {
    pub fn other(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ComparisonTransportIntent {
    #[default]
    Paused,
    Playing,
}

/// A musical comparison coordinate that deliberately avoids asserting a shared
/// second offset. `beat_offset` is zero-based within the named bar. Its upper
/// bound depends on that subject's meter map, so this type only validates the
/// context-free invariants (finite and non-negative); source-specific resolution
/// is a later responsibility.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MusicalComparisonAnchor {
    pub bar_index: u32,
    pub beat_offset: f64,
}

impl MusicalComparisonAnchor {
    pub fn new(bar_index: u32, beat_offset: f64) -> Result<Self, ComparisonSessionError> {
        if !beat_offset.is_finite() || beat_offset < 0.0 {
            return Err(ComparisonSessionError::InvalidMusicalAnchor);
        }
        Ok(Self {
            bar_index,
            beat_offset,
        })
    }
}

/// One comparison subject. `artifact_identity` is optional because imported,
/// teaching, or legacy material may be valid to hear before full content
/// identity is available. When both the source and full identity name a
/// rendition, they must agree; the comparison layer never resolves an identity
/// contradiction by choosing one side silently.
#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonSubject {
    pub source: PlaybackSource,
    pub artifact_identity: Option<ArtifactIdentity>,
}

impl ComparisonSubject {
    pub fn new(
        source: PlaybackSource,
        artifact_identity: Option<ArtifactIdentity>,
    ) -> Result<Self, ComparisonSessionError> {
        if let (Some(source_rendition), Some(identity)) =
            (source.rendition_id.as_ref(), artifact_identity.as_ref())
            && source_rendition != &identity.rendition
        {
            return Err(ComparisonSessionError::RenditionIdentityMismatch);
        }
        Ok(Self {
            source,
            artifact_identity,
        })
    }

    fn same_subject_as(&self, other: &Self) -> bool {
        // Any exact strong identity match is enough to establish that A/B would
        // not contain two distinct subjects. A different full identity does not
        // suppress a matching rendition check: two records that point at the
        // exact same rendered bytes are still not an audible A/B comparison.
        if let (Some(left), Some(right)) = (&self.artifact_identity, &other.artifact_identity)
            && left == right
        {
            return true;
        }
        if let (Some(left), Some(right)) = (&self.source.rendition_id, &other.source.rendition_id)
            && left == right
        {
            return true;
        }
        // Exact URL equality is the weakest fallback identity available for
        // legacy/unverified material. It is intentionally used only as a
        // duplicate guard, never promoted into ArtifactIdentity.
        self.source.audio_url == other.source.audio_url
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComparisonSessionError {
    SameSubject,
    RenditionIdentityMismatch,
    InvalidMusicalAnchor,
}

impl fmt::Display for ComparisonSessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SameSubject => write!(f, "comparison requires two distinct musical subjects"),
            Self::RenditionIdentityMismatch => write!(
                f,
                "playback source rendition conflicts with its supplied artifact identity"
            ),
            Self::InvalidMusicalAnchor => {
                write!(f, "comparison musical anchor must have a finite non-negative beat")
            }
        }
    }
}

impl std::error::Error for ComparisonSessionError {}

/// Pure session state for repeated A/B evaluation.
///
/// It deliberately contains neither `MuseState::current` nor a
/// `PlaybackBookmark`: comparison must not gain canonical-piece authority merely
/// because one side is currently audible, and Return semantics remain owned by
/// the separate temporary-audition flow.
#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonSession {
    pub a: ComparisonSubject,
    pub b: ComparisonSubject,
    pub active_side: ComparisonSide,
    pub transport_intent: ComparisonTransportIntent,
    pub musical_anchor: Option<MusicalComparisonAnchor>,
}

impl ComparisonSession {
    pub fn new(a: ComparisonSubject, b: ComparisonSubject) -> Result<Self, ComparisonSessionError> {
        if a.same_subject_as(&b) {
            return Err(ComparisonSessionError::SameSubject);
        }
        Ok(Self {
            a,
            b,
            active_side: ComparisonSide::A,
            transport_intent: ComparisonTransportIntent::Paused,
            musical_anchor: None,
        })
    }

    pub fn active_subject(&self) -> &ComparisonSubject {
        match self.active_side {
            ComparisonSide::A => &self.a,
            ComparisonSide::B => &self.b,
        }
    }

    pub fn subject(&self, side: ComparisonSide) -> &ComparisonSubject {
        match side {
            ComparisonSide::A => &self.a,
            ComparisonSide::B => &self.b,
        }
    }

    pub fn switch_to(&mut self, side: ComparisonSide) {
        self.active_side = side;
    }

    pub fn toggle_side(&mut self) {
        self.active_side = self.active_side.other();
    }

    pub fn set_transport_intent(&mut self, intent: ComparisonTransportIntent) {
        self.transport_intent = intent;
    }

    pub fn set_musical_anchor(&mut self, anchor: Option<MusicalComparisonAnchor>) {
        self.musical_anchor = anchor;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::playback::{PlaybackPresentation, PlaybackSubjectKind};
    use symthaea_muse_protocol::{
        CompositionArtifactId, RenditionArtifactId, ScoreContentArtifactId,
    };

    fn source(url: &str, rendition: Option<&str>) -> PlaybackSource {
        PlaybackSource {
            rendition_id: rendition.map(|hash| RenditionArtifactId(hash.to_string())),
            audio_url: url.to_string(),
            duration_hint_seconds: Some(30.0),
            advance_on_end: false,
            presentation: PlaybackPresentation {
                kind: PlaybackSubjectKind::Review,
                title: url.to_string(),
                subtitle: None,
                style_hint: None,
            },
        }
    }

    fn identity(score: char, composition: char, rendition: char) -> ArtifactIdentity {
        ArtifactIdentity {
            score_content: ScoreContentArtifactId(score.to_string().repeat(64)),
            composition: CompositionArtifactId(composition.to_string().repeat(64)),
            rendition: RenditionArtifactId(rendition.to_string().repeat(64)),
        }
    }

    #[test]
    fn exact_same_artifact_is_not_a_comparison() {
        let artifact = identity('a', 'b', 'c');
        let a = ComparisonSubject::new(
            source("/audio/a", Some(&artifact.rendition.0)),
            Some(artifact.clone()),
        )
        .unwrap();
        let b = ComparisonSubject::new(
            source("/mirrored/url", Some(&artifact.rendition.0)),
            Some(artifact),
        )
        .unwrap();

        assert_eq!(
            ComparisonSession::new(a, b),
            Err(ComparisonSessionError::SameSubject)
        );
    }

    #[test]
    fn same_verified_rendition_is_not_a_comparison_even_if_other_identity_axes_differ() {
        let shared_rendition = 'c';
        let a_identity = identity('a', 'b', shared_rendition);
        let b_identity = identity('d', 'e', shared_rendition);
        let a = ComparisonSubject::new(
            source("/audio/a", Some(&a_identity.rendition.0)),
            Some(a_identity),
        )
        .unwrap();
        let b = ComparisonSubject::new(
            source("/audio/b", Some(&b_identity.rendition.0)),
            Some(b_identity),
        )
        .unwrap();
        assert_eq!(
            ComparisonSession::new(a, b),
            Err(ComparisonSessionError::SameSubject)
        );
    }

    #[test]
    fn same_verified_rendition_without_full_identity_is_not_a_comparison() {
        let rendition = "c".repeat(64);
        let a = ComparisonSubject::new(source("/audio/a", Some(&rendition)), None).unwrap();
        let b = ComparisonSubject::new(source("/audio/b", Some(&rendition)), None).unwrap();
        assert_eq!(
            ComparisonSession::new(a, b),
            Err(ComparisonSessionError::SameSubject)
        );
    }

    #[test]
    fn different_renditions_of_same_composition_are_valid_comparison_subjects() {
        let a_identity = identity('a', 'b', 'c');
        let b_identity = identity('a', 'b', 'd');
        let a = ComparisonSubject::new(
            source("/audio/native", Some(&a_identity.rendition.0)),
            Some(a_identity),
        )
        .unwrap();
        let b = ComparisonSubject::new(
            source("/audio/fluidsynth", Some(&b_identity.rendition.0)),
            Some(b_identity),
        )
        .unwrap();

        assert!(ComparisonSession::new(a, b).is_ok());
    }

    #[test]
    fn contradictory_source_and_full_identity_fail_closed() {
        let artifact = identity('a', 'b', 'c');
        let mismatched_rendition = "d".repeat(64);
        let result = ComparisonSubject::new(
            source("/audio/a", Some(&mismatched_rendition)),
            Some(artifact),
        );
        assert_eq!(result, Err(ComparisonSessionError::RenditionIdentityMismatch));
    }

    #[test]
    fn exact_same_url_is_a_duplicate_guard_not_a_content_hash_claim() {
        let a = ComparisonSubject::new(source("/audio/legacy", None), None).unwrap();
        let b = ComparisonSubject::new(source("/audio/legacy", None), None).unwrap();
        assert_eq!(
            ComparisonSession::new(a, b),
            Err(ComparisonSessionError::SameSubject)
        );
    }

    #[test]
    fn switching_side_does_not_mutate_subjects_or_create_return_semantics() {
        let a = ComparisonSubject::new(source("/audio/a", None), None).unwrap();
        let b = ComparisonSubject::new(source("/audio/b", None), None).unwrap();
        let mut session = ComparisonSession::new(a.clone(), b.clone()).unwrap();

        assert_eq!(session.active_subject(), &a);
        session.toggle_side();
        assert_eq!(session.active_side, ComparisonSide::B);
        assert_eq!(session.active_subject(), &b);
        assert_eq!(session.a, a);
        assert_eq!(session.b, b);
    }

    #[test]
    fn musical_anchor_validates_only_context_free_invariants() {
        assert_eq!(
            MusicalComparisonAnchor::new(4, -0.1),
            Err(ComparisonSessionError::InvalidMusicalAnchor)
        );
        assert_eq!(
            MusicalComparisonAnchor::new(4, f64::NAN),
            Err(ComparisonSessionError::InvalidMusicalAnchor)
        );
        assert_eq!(
            MusicalComparisonAnchor::new(4, f64::INFINITY),
            Err(ComparisonSessionError::InvalidMusicalAnchor)
        );
        assert_eq!(
            MusicalComparisonAnchor::new(4, 2.5).unwrap(),
            MusicalComparisonAnchor {
                bar_index: 4,
                beat_offset: 2.5,
            }
        );
    }

    #[test]
    fn transport_intent_and_anchor_are_session_state_not_playback_effects() {
        let a = ComparisonSubject::new(source("/audio/a", None), None).unwrap();
        let b = ComparisonSubject::new(source("/audio/b", None), None).unwrap();
        let mut session = ComparisonSession::new(a, b).unwrap();
        let anchor = MusicalComparisonAnchor::new(2, 1.5).unwrap();

        session.set_transport_intent(ComparisonTransportIntent::Playing);
        session.set_musical_anchor(Some(anchor));

        assert_eq!(session.transport_intent, ComparisonTransportIntent::Playing);
        assert_eq!(session.musical_anchor, Some(anchor));
    }
}
