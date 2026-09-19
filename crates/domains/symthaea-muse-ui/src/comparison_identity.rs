// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Evidence-preserving identity comparison for deliberate A/B sessions.
//!
//! This module answers a structural question only: how do the two subjects'
//! content identities relate? It does not score musical quality, infer causality,
//! or treat missing identity as evidence that two artifacts differ.

use symthaea_muse_protocol::RenditionArtifactId;

use crate::comparison::ComparisonSubject;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IdentityAxisRelation {
    Same,
    Different,
    Unavailable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComparisonIdentityRelation {
    /// Both subjects resolve to the exact same score, recipe/composition, and
    /// rendition commitments. `ComparisonSession::new` normally rejects this
    /// before a session exists, but the diff remains total for diagnostics.
    SameArtifact,
    /// Same symbolic score and composition commitment, different rendered bytes.
    AlternateRendition,
    /// Different recipe/composition commitments produced the same symbolic score.
    SameScoreDifferentComposition,
    /// Both symbolic score and composition commitments differ.
    DifferentComposition,
    /// Two claims cannot both satisfy Melothaea's identity model; for example,
    /// the same composition commitment names different symbolic scores, or the
    /// same rendition commitment is attached to different upstream identities.
    EvidenceConflict,
    /// At least one upstream identity axis is unavailable. Known rendition
    /// commitments are still reported independently in the axis fields.
    InsufficientIdentity,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComparisonIdentityDiff {
    pub relation: ComparisonIdentityRelation,
    pub score: IdentityAxisRelation,
    pub composition: IdentityAxisRelation,
    pub rendition: IdentityAxisRelation,
}

/// Compare two subjects without manufacturing missing content identity.
///
/// Full `ArtifactIdentity` is authoritative for score/composition/rendition.
/// If full identity is unavailable, an independently verified rendition carried
/// by `PlaybackSource` may still establish only the rendition axis. URLs are not
/// content identity and never participate in this diff.
pub fn compare_subject_identity(
    a: &ComparisonSubject,
    b: &ComparisonSubject,
) -> ComparisonIdentityDiff {
    let score = compare_full_axis(
        a.artifact_identity
            .as_ref()
            .map(|identity| &identity.score_content.0),
        b.artifact_identity
            .as_ref()
            .map(|identity| &identity.score_content.0),
    );
    let composition = compare_full_axis(
        a.artifact_identity
            .as_ref()
            .map(|identity| &identity.composition.0),
        b.artifact_identity
            .as_ref()
            .map(|identity| &identity.composition.0),
    );
    let rendition = compare_optional_axis(effective_rendition(a), effective_rendition(b));

    let relation = classify(score, composition, rendition);
    ComparisonIdentityDiff {
        relation,
        score,
        composition,
        rendition,
    }
}

fn effective_rendition(subject: &ComparisonSubject) -> Option<&RenditionArtifactId> {
    subject
        .artifact_identity
        .as_ref()
        .map(|identity| &identity.rendition)
        .or(subject.source.rendition_id.as_ref())
}

fn compare_full_axis(a: Option<&String>, b: Option<&String>) -> IdentityAxisRelation {
    match (a, b) {
        (Some(a), Some(b)) if a == b => IdentityAxisRelation::Same,
        (Some(_), Some(_)) => IdentityAxisRelation::Different,
        _ => IdentityAxisRelation::Unavailable,
    }
}

fn compare_optional_axis<T: PartialEq>(a: Option<&T>, b: Option<&T>) -> IdentityAxisRelation {
    match (a, b) {
        (Some(a), Some(b)) if a == b => IdentityAxisRelation::Same,
        (Some(_), Some(_)) => IdentityAxisRelation::Different,
        _ => IdentityAxisRelation::Unavailable,
    }
}

fn classify(
    score: IdentityAxisRelation,
    composition: IdentityAxisRelation,
    rendition: IdentityAxisRelation,
) -> ComparisonIdentityRelation {
    use IdentityAxisRelation::{Different, Same, Unavailable};

    // Same rendered bytes cannot honestly carry conflicting upstream content
    // identities, and the same composition commitment cannot name two different
    // symbolic scores. Surface those as evidence conflicts rather than picking a
    // preferred axis or flattening them into an ordinary comparison.
    if rendition == Same && (score == Different || composition == Different) {
        return ComparisonIdentityRelation::EvidenceConflict;
    }
    if composition == Same && score == Different {
        return ComparisonIdentityRelation::EvidenceConflict;
    }

    match (score, composition, rendition) {
        (Same, Same, Same) => ComparisonIdentityRelation::SameArtifact,
        (Same, Same, Different) => ComparisonIdentityRelation::AlternateRendition,
        (Same, Different, Different) => {
            ComparisonIdentityRelation::SameScoreDifferentComposition
        }
        (Different, Different, Different) => ComparisonIdentityRelation::DifferentComposition,
        // Once any upstream axis is absent there is not enough evidence to make
        // a stronger score/composition relationship claim. A known rendition
        // difference remains visible in `rendition` above.
        (Unavailable, _, _) | (_, Unavailable, _) | (_, _, Unavailable) => {
            ComparisonIdentityRelation::InsufficientIdentity
        }
        // Remaining fully-known combinations are internally inconsistent with
        // the identity relationships represented by ArtifactIdentity.
        _ => ComparisonIdentityRelation::EvidenceConflict,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::comparison::ComparisonSubject;
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

    fn subject(
        identity: Option<ArtifactIdentity>,
        source_rendition: Option<char>,
    ) -> ComparisonSubject {
        let source = PlaybackSource {
            rendition_id: source_rendition
                .map(|value| RenditionArtifactId(value.to_string().repeat(64))),
            audio_url: format!("/audio/{}", source_rendition.unwrap_or('x')),
            duration_hint_seconds: Some(30.0),
            advance_on_end: false,
            presentation: PlaybackPresentation {
                kind: PlaybackSubjectKind::Review,
                title: "subject".into(),
                subtitle: None,
                style_hint: None,
            },
        };
        ComparisonSubject::new(source, identity).unwrap()
    }

    #[test]
    fn same_composition_different_rendition_is_described_not_ranked() {
        let a = subject(Some(identity('a', 'b', 'c')), Some('c'));
        let b = subject(Some(identity('a', 'b', 'd')), Some('d'));
        assert_eq!(
            compare_subject_identity(&a, &b),
            ComparisonIdentityDiff {
                relation: ComparisonIdentityRelation::AlternateRendition,
                score: IdentityAxisRelation::Same,
                composition: IdentityAxisRelation::Same,
                rendition: IdentityAxisRelation::Different,
            }
        );
    }

    #[test]
    fn different_recipes_can_converge_on_the_same_score_without_being_same_composition() {
        let a = subject(Some(identity('a', 'b', 'c')), Some('c'));
        let b = subject(Some(identity('a', 'd', 'e')), Some('e'));
        let diff = compare_subject_identity(&a, &b);
        assert_eq!(
            diff.relation,
            ComparisonIdentityRelation::SameScoreDifferentComposition
        );
        assert_eq!(diff.score, IdentityAxisRelation::Same);
        assert_eq!(diff.composition, IdentityAxisRelation::Different);
    }

    #[test]
    fn different_score_and_recipe_are_only_structurally_different_not_better_or_worse() {
        let a = subject(Some(identity('a', 'b', 'c')), Some('c'));
        let b = subject(Some(identity('d', 'e', 'f')), Some('f'));
        assert_eq!(
            compare_subject_identity(&a, &b).relation,
            ComparisonIdentityRelation::DifferentComposition
        );
    }

    #[test]
    fn missing_full_identity_never_becomes_a_score_or_recipe_claim() {
        let a = subject(None, Some('c'));
        let b = subject(None, Some('d'));
        assert_eq!(
            compare_subject_identity(&a, &b),
            ComparisonIdentityDiff {
                relation: ComparisonIdentityRelation::InsufficientIdentity,
                score: IdentityAxisRelation::Unavailable,
                composition: IdentityAxisRelation::Unavailable,
                rendition: IdentityAxisRelation::Different,
            }
        );
    }

    #[test]
    fn one_missing_rendition_is_unavailable_not_different() {
        let a = subject(None, Some('c'));
        let b = subject(None, None);
        assert_eq!(
            compare_subject_identity(&a, &b).rendition,
            IdentityAxisRelation::Unavailable
        );
    }

    #[test]
    fn conflicting_upstream_identity_on_same_rendered_bytes_is_an_evidence_conflict() {
        // Construct directly because ComparisonSubject::new correctly rejects the
        // same-rendition A/B session before it reaches this diagnostic layer.
        let a_identity = identity('a', 'b', 'c');
        let b_identity = identity('d', 'e', 'c');
        let a = ComparisonSubject {
            source: PlaybackSource {
                rendition_id: Some(a_identity.rendition.clone()),
                audio_url: "/audio/a".into(),
                duration_hint_seconds: None,
                advance_on_end: false,
                presentation: PlaybackPresentation::review("A".into()),
            },
            artifact_identity: Some(a_identity),
        };
        let b = ComparisonSubject {
            source: PlaybackSource {
                rendition_id: Some(b_identity.rendition.clone()),
                audio_url: "/audio/b".into(),
                duration_hint_seconds: None,
                advance_on_end: false,
                presentation: PlaybackPresentation::review("B".into()),
            },
            artifact_identity: Some(b_identity),
        };
        assert_eq!(
            compare_subject_identity(&a, &b).relation,
            ComparisonIdentityRelation::EvidenceConflict
        );
    }

    #[test]
    fn same_composition_commitment_cannot_silently_name_different_scores() {
        let a = subject(Some(identity('a', 'b', 'c')), Some('c'));
        let b = subject(Some(identity('d', 'b', 'e')), Some('e'));
        assert_eq!(
            compare_subject_identity(&a, &b).relation,
            ComparisonIdentityRelation::EvidenceConflict
        );
    }
}
