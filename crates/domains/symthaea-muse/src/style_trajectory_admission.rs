// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Catalogue-backed admission for symbolic style trajectories.
//!
//! This module is deliberately in symthaea-muse, not the dependency-light
//! symthaea-music-theory crate: Muse already depends on both the 152-style
//! protocol catalogue and the music-theory engine. It joins those sources
//! without making music theory depend on its UI/wire protocol.
//!
//! Admission proves that references resolve to callable styles, binds each
//! profile reference to its exact built-in profile configuration bytes, and
//! requires structural owners to dispatch to a typed grammar plan rather than
//! the engine's Compatibility fallback. It does not prove cultural
//! authenticity, renderer capability, or causal donor-layer use.

use sha2::{Digest, Sha256};
use symthaea_music_theory::{
    Duration, GrammarFamily, Score, Style, StyleTrajectoryIssue, StyleTrajectoryPlanV1,
    validate_style_trajectory,
};
use symthaea_muse_protocol::catalog::{CanonicalStyle, StyleStatus, catalog_entry};

/// A catalogue reference must use the canonical catalog-style:<id> form.
pub const CATALOG_STYLE_REF_PREFIX: &str = "catalog-style:";
/// Exact serde_json byte serialization used to bind a plan to a symbolic Score.
pub const SOURCE_SCORE_DIGEST_SCHEMA_V1: &str = "muse-score-serde-json-sha256-v1";
/// Canonical bytes for a built-in style's name, grammar profile and spec.
/// This binds profile configuration, not the executable implementation code.
pub const STYLE_PROFILE_DIGEST_SCHEMA_V1: &str =
    "muse-style-profile-spec-serde-json-sha256-v1";

/// One resolved style reference. The review flag is metadata, not a claim
/// that an expert has completed a review.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedCatalogStyle {
    pub catalog_id: u16,
    pub catalog_name: &'static str,
    pub composer_style: &'static str,
    pub grammar_family: GrammarFamily,
    pub typed_grammar_plan: bool,
    pub requires_expert_review: bool,
}

/// The resolved owners/profiles for one trajectory segment.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedStyleTrajectorySegment {
    pub segment_id: String,
    pub structural_owner: ResolvedCatalogStyle,
    pub profile: ResolvedCatalogStyle,
}

/// Successful catalogue-backed admission. This is not musical-quality or
/// tradition-authenticity qualification.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StyleTrajectoryAdmission {
    pub segments: Vec<ResolvedStyleTrajectorySegment>,
    /// True when any referenced catalogue entry is marked for expert review.
    pub expert_review_required: bool,
}

/// Admission can fail either at the exact-time envelope or at reference/route
/// resolution. Callers must not execute a failed plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StyleTrajectoryAdmissionError {
    InvalidEnvelope(Vec<StyleTrajectoryIssue>),
    UnsupportedSourceDigestSchema { actual: String },
    SourceScoreDurationMismatch {
        plan_total: Duration,
        score_total: Duration,
    },
    SourceScoreSerializationFailed { reason: String },
    StyleProfileSerializationFailed { style: String, reason: String },
    SourceScoreDigestMismatch { expected: String, actual: String },
    UnresolvedReferences(Vec<StyleTrajectoryReferenceIssue>),
}

/// Stable, machine-readable reference-resolution failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StyleTrajectoryReferenceIssue {
    pub code: &'static str,
    pub segment_id: String,
    pub field: &'static str,
    pub reference: String,
}

/// Admit a trajectory only when its source-score digest matches the exact
/// serde_json serialization of the supplied symbolic Score.
///
/// This uses the same byte convention as Muse's existing serialized score
/// provenance: serde_json::to_vec followed by SHA-256, lowercase hex. The
/// digest schema is explicit so a future serialization change cannot silently
/// reinterpret an existing plan. The declared trajectory duration must also
/// equal the source Score's exact rational total duration.
pub fn admit_style_trajectory_for_score(
    plan: &StyleTrajectoryPlanV1,
    source_score: &Score,
) -> Result<StyleTrajectoryAdmission, StyleTrajectoryAdmissionError> {
    if plan.source_score_digest_schema != SOURCE_SCORE_DIGEST_SCHEMA_V1 {
        return Err(
            StyleTrajectoryAdmissionError::UnsupportedSourceDigestSchema {
                actual: plan.source_score_digest_schema.clone(),
            },
        );
    }
    if plan.total_beats != source_score.total_beats {
        return Err(StyleTrajectoryAdmissionError::SourceScoreDurationMismatch {
            plan_total: plan.total_beats,
            score_total: source_score.total_beats,
        });
    }

    let bytes = serde_json::to_vec(source_score).map_err(|error| {
        StyleTrajectoryAdmissionError::SourceScoreSerializationFailed {
            reason: error.to_string(),
        }
    })?;
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != plan.source_score_sha256 {
        return Err(StyleTrajectoryAdmissionError::SourceScoreDigestMismatch {
            expected: plan.source_score_sha256.clone(),
            actual,
        });
    }

    admit_style_trajectory(plan)
}

/// Compute the source digest required by SOURCE_SCORE_DIGEST_SCHEMA_V1.
pub fn source_score_sha256_v1(
    source_score: &Score,
) -> Result<String, StyleTrajectoryAdmissionError> {
    let bytes = serde_json::to_vec(source_score).map_err(|error| {
        StyleTrajectoryAdmissionError::SourceScoreSerializationFailed {
            reason: error.to_string(),
        }
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// Compute the exact profile-configuration digest for a built-in Style.
///
/// The digest binds the catalogue's resolved style name, its typed grammar
/// profile, the serialized CompositionSpec and the music-theory crate version.
/// It deliberately does not claim to hash executable composer implementation
/// code or prove that every configuration dimension is perceptually effective.
pub fn style_profile_sha256_v1(
    style: Style,
) -> Result<String, StyleTrajectoryAdmissionError> {
    let material = (
        STYLE_PROFILE_DIGEST_SCHEMA_V1,
        symthaea_music_theory::MUSIC_THEORY_ENGINE_VERSION,
        format!("{style:?}"),
        style.grammar_profile(),
        style.spec(),
    );
    let bytes = serde_json::to_vec(&material).map_err(|error| {
        StyleTrajectoryAdmissionError::StyleProfileSerializationFailed {
            style: format!("{style:?}"),
            reason: error.to_string(),
        }
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

/// Resolve and admit a trajectory against the current canonical catalogue.
///
/// The structural_owner_ref must resolve to a real typed grammar dispatch.
/// A profile_ref must resolve to a callable catalogue-backed Style and carry
/// the exact digest returned by [`style_profile_sha256_v1`]. Compatibility
/// routes may be named as profiles but may never serve as the structural owner
/// for this M0 admission.
pub fn admit_style_trajectory(
    plan: &StyleTrajectoryPlanV1,
) -> Result<StyleTrajectoryAdmission, StyleTrajectoryAdmissionError> {
    validate_style_trajectory(plan)
        .map_err(StyleTrajectoryAdmissionError::InvalidEnvelope)?;

    let mut segments = Vec::with_capacity(plan.segments.len());
    let mut issues = Vec::new();
    let mut expert_review_required = false;

    for segment in &plan.segments {
        let owner = resolve_reference(
            &segment.structural_owner_ref,
            &segment.segment_id,
            "structural_owner_ref",
            true,
            None,
        );
        let profile = resolve_reference(
            &segment.profile_ref,
            &segment.segment_id,
            "profile_ref",
            false,
            Some(&segment.profile_sha256),
        );

        let owner = match owner {
            Ok(value) => {
                expert_review_required |= value.requires_expert_review;
                Some(value)
            }
            Err(issue) => {
                issues.push(issue);
                None
            }
        };
        let profile = match profile {
            Ok(value) => {
                expert_review_required |= value.requires_expert_review;
                Some(value)
            }
            Err(issue) => {
                issues.push(issue);
                None
            }
        };

        if let (Some(structural_owner), Some(profile)) = (owner, profile) {
            segments.push(ResolvedStyleTrajectorySegment {
                segment_id: segment.segment_id.clone(),
                structural_owner,
                profile,
            });
        }
    }

    if issues.is_empty() {
        Ok(StyleTrajectoryAdmission {
            segments,
            expert_review_required,
        })
    } else {
        Err(StyleTrajectoryAdmissionError::UnresolvedReferences(issues))
    }
}

fn resolve_reference(
    reference: &str,
    segment_id: &str,
    field: &'static str,
    require_dedicated_grammar: bool,
    expected_profile_sha256: Option<&str>,
) -> Result<ResolvedCatalogStyle, StyleTrajectoryReferenceIssue> {
    let fail = |code| StyleTrajectoryReferenceIssue {
        code,
        segment_id: segment_id.to_owned(),
        field,
        reference: reference.to_owned(),
    };

    let Some(raw_id) = reference.strip_prefix(CATALOG_STYLE_REF_PREFIX) else {
        return Err(fail("unsupported_reference_scheme"));
    };
    let Ok(catalog_id) = raw_id.parse::<u16>() else {
        return Err(fail("invalid_catalog_style_id"));
    };
    if catalog_id.to_string() != raw_id {
        return Err(fail("noncanonical_catalog_style_id"));
    }

    let Some(entry) = catalog_entry(catalog_id) else {
        return Err(fail("unknown_catalog_style_id"));
    };
    if !entry.is_composable() {
        return Err(fail("catalog_style_not_composable"));
    }

    resolve_entry(
        entry,
        reference,
        segment_id,
        field,
        require_dedicated_grammar,
        expected_profile_sha256,
    )
}

fn resolve_entry(
    entry: &'static CanonicalStyle,
    reference: &str,
    segment_id: &str,
    field: &'static str,
    require_dedicated_grammar: bool,
    expected_profile_sha256: Option<&str>,
) -> Result<ResolvedCatalogStyle, StyleTrajectoryReferenceIssue> {
    let fail = |code| StyleTrajectoryReferenceIssue {
        code,
        segment_id: segment_id.to_owned(),
        field,
        reference: reference.to_owned(),
    };

    let Some(composer_style) = entry.composer_style else {
        return Err(fail("catalog_style_has_no_composer_route"));
    };

    let Some(style) = Style::ALL
        .into_iter()
        .find(|style| format!("{style:?}") == composer_style)
    else {
        return Err(fail("catalog_composer_route_not_in_style_registry"));
    };

    if field == "profile_ref" {
        let Some(expected) = expected_profile_sha256 else {
            return Err(fail("missing_profile_revision_digest"));
        };
        let actual = style_profile_sha256_v1(style)
            .map_err(|_| fail("profile_spec_serialization_failed"))?;
        if actual != expected {
            return Err(fail("profile_revision_digest_mismatch"));
        }
    }

    let grammar_family = style.grammar_profile().family;
    let typed_grammar_plan = has_typed_grammar_dispatch(grammar_family);
    if require_dedicated_grammar && !typed_grammar_plan {
        return Err(fail("structural_owner_uses_compatibility_fallback"));
    }

    Ok(ResolvedCatalogStyle {
        catalog_id: entry.id,
        catalog_name: entry.name,
        composer_style,
        grammar_family,
        typed_grammar_plan,
        requires_expert_review: entry.requires_expert_review
            || matches!(entry.status, StyleStatus::ExpertReviewRequired),
    })
}

/// Mirrors only the explicit typed return arms of compose_with_grammar_plan.
/// Families dispatched through its catch-all Compatibility arm remain
/// unqualified as structural owners.
fn has_typed_grammar_dispatch(family: GrammarFamily) -> bool {
    matches!(
        family,
        GrammarFamily::PeriodSentence
            | GrammarFamily::Contrapuntal
            | GrammarFamily::GrooveCycle
            | GrammarFamily::ProcessAdditive
            | GrammarFamily::RagaModalArc
            | GrammarFamily::BluesCallResponse
            | GrammarFamily::JazzChorus
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::{
        Duration, Key, PitchClass, Score, StyleTrajectorySegmentV1,
        StyleTrajectoryTransitionV1, StyleTransitionMode, STYLE_TRAJECTORY_SCHEMA_VERSION,
    };
    use symthaea_muse_protocol::catalog::CATALOG;

    const SCORE_DIGEST: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    fn entry_for_composer_style(name: &str) -> &'static CanonicalStyle {
        CATALOG
            .iter()
            .find(|entry| entry.composer_style == Some(name))
            .expect("fixture composer route must exist in the canonical catalogue")
    }

    fn profile_style_for_catalog_id(id: u16) -> Style {
        let entry = CATALOG
            .iter()
            .find(|entry| entry.id == id)
            .expect("fixture catalogue id must exist");
        let composer_style = entry
            .composer_style
            .expect("fixture profile must have a composer route");
        Style::ALL
            .into_iter()
            .find(|style| format!("{style:?}") == composer_style)
            .expect("fixture route must exist in Style::ALL")
    }

    fn segment(
        id: &str,
        owner: u16,
        profile: u16,
        start: i64,
        end: i64,
    ) -> StyleTrajectorySegmentV1 {
        let profile_sha256 =
            style_profile_sha256_v1(profile_style_for_catalog_id(profile)).unwrap();
        StyleTrajectorySegmentV1 {
            segment_id: id.into(),
            start_beat: Duration::new(start, 1),
            end_beat: Duration::new(end, 1),
            structural_owner_ref: format!("{CATALOG_STYLE_REF_PREFIX}{owner}"),
            profile_ref: format!("{CATALOG_STYLE_REF_PREFIX}{profile}"),
            profile_sha256,
        }
    }

    fn transition(from: &str, to: &str) -> StyleTrajectoryTransitionV1 {
        StyleTrajectoryTransitionV1 {
            transition_id: format!("{from}-to-{to}"),
            from_segment_id: from.into(),
            to_segment_id: to.into(),
            mode: StyleTransitionMode::HardBoundary,
        }
    }

    fn plan(segments: Vec<StyleTrajectorySegmentV1>) -> StyleTrajectoryPlanV1 {
        let transitions = segments
            .windows(2)
            .map(|pair| transition(&pair[0].segment_id, &pair[1].segment_id))
            .collect();
        StyleTrajectoryPlanV1 {
            schema_version: STYLE_TRAJECTORY_SCHEMA_VERSION,
            source_score_sha256: SCORE_DIGEST.into(),
            source_score_digest_schema: "muse-score-serde-json-sha256-v1".into(),
            total_beats: Duration::new(4, 1),
            segments,
            transitions,
        }
    }

    fn reference_issues(
        error: StyleTrajectoryAdmissionError,
    ) -> Vec<StyleTrajectoryReferenceIssue> {
        match error {
            StyleTrajectoryAdmissionError::UnresolvedReferences(issues) => issues,
            other => panic!("expected reference-resolution failure, got {other:?}"),
        }
    }

    #[test]
    fn admits_current_dedicated_groove_and_process_routes() {
        let groove = entry_for_composer_style("AfroCuban");
        let process = entry_for_composer_style("Minimalism");
        let candidate = plan(vec![
            segment("groove", groove.id, groove.id, 0, 2),
            segment("process", process.id, process.id, 2, 4),
        ]);

        let admitted = admit_style_trajectory(&candidate).expect("dedicated routes should resolve");
        assert_eq!(admitted.segments.len(), 2);
        assert_eq!(
            admitted.segments[0].structural_owner.grammar_family,
            GrammarFamily::GrooveCycle
        );
        assert_eq!(
            admitted.segments[1].structural_owner.grammar_family,
            GrammarFamily::ProcessAdditive
        );
        assert!(admitted.expert_review_required);
    }

    #[test]
    fn admits_typed_period_sentence_owner() {
        let classical = entry_for_composer_style("Classical");
        let candidate = plan(vec![segment(
            "classical",
            classical.id,
            classical.id,
            0,
            4,
        )]);
        let admitted =
            admit_style_trajectory(&candidate).expect("period sentence has a typed plan");
        assert_eq!(
            admitted.segments[0].structural_owner.grammar_family,
            GrammarFamily::PeriodSentence
        );
        assert!(admitted.segments[0].structural_owner.typed_grammar_plan);
    }

    #[test]
    fn rejects_research_only_catalog_entry() {
        let research = CATALOG
            .iter()
            .find(|entry| entry.status == StyleStatus::Research)
            .expect("catalogue should contain research-only entries");
        let candidate = plan(vec![segment("research", research.id, research.id, 0, 4)]);
        let issues = reference_issues(admit_style_trajectory(&candidate).unwrap_err());
        assert!(issues.iter().any(|issue| issue.code == "catalog_style_not_composable"));
    }

    #[test]
    fn rejects_compatibility_fallback_as_structural_owner() {
        let waltz = entry_for_composer_style("Waltz");
        let candidate = plan(vec![segment("waltz", waltz.id, waltz.id, 0, 4)]);
        let issues = reference_issues(admit_style_trajectory(&candidate).unwrap_err());
        assert!(issues.iter().any(|issue| {
            issue.field == "structural_owner_ref"
                && issue.code == "structural_owner_uses_compatibility_fallback"
        }));
    }

    #[test]
    fn rejects_unknown_noncanonical_and_wrong_scheme_references() {
        let classical = entry_for_composer_style("Classical");
        let mut candidate = plan(vec![segment("classical", classical.id, classical.id, 0, 4)]);
        candidate.segments[0].profile_ref = "catalog-style:999".into();
        let issues = reference_issues(admit_style_trajectory(&candidate).unwrap_err());
        assert!(issues.iter().any(|issue| issue.code == "unknown_catalog_style_id"));

        candidate.segments[0].profile_ref = "catalog-style:01".into();
        let issues = reference_issues(admit_style_trajectory(&candidate).unwrap_err());
        assert!(issues.iter().any(|issue| issue.code == "noncanonical_catalog_style_id"));

        candidate.segments[0].profile_ref = "style:1".into();
        let issues = reference_issues(admit_style_trajectory(&candidate).unwrap_err());
        assert!(issues.iter().any(|issue| issue.code == "unsupported_reference_scheme"));
    }

    #[test]
    fn expert_review_status_is_exposed_not_silently_cleared() {
        let expert = CATALOG
            .iter()
            .find(|entry| {
                entry.composer_style.is_some()
                    && entry.status == StyleStatus::ExpertReviewRequired
            })
            .expect("catalogue should contain a routed expert-review style");
        let candidate = plan(vec![segment("expert", expert.id, expert.id, 0, 4)]);
        let admitted = admit_style_trajectory(&candidate)
            .expect("expert-review status is not a technical route failure");
        assert!(admitted.expert_review_required);
        assert!(
            admitted.segments[0].structural_owner.requires_expert_review,
            "review requirement must remain explicit"
        );
    }

    #[test]
    fn profile_route_may_be_named_without_claiming_it_is_a_typed_owner() {
        let classical = entry_for_composer_style("Classical");
        let waltz = entry_for_composer_style("Waltz");
        let candidate = plan(vec![segment("hybrid", classical.id, waltz.id, 0, 4)]);
        let admitted = admit_style_trajectory(&candidate)
            .expect("a donor profile can be resolved while the owner is qualified");
        assert!(admitted.segments[0].structural_owner.typed_grammar_plan);
        assert!(!admitted.segments[0].profile.typed_grammar_plan);
    }

    #[test]
    fn profile_admission_recomputes_the_exact_style_configuration_digest() {
        let classical = entry_for_composer_style("Classical");
        let mut candidate = plan(vec![segment(
            "classical",
            classical.id,
            classical.id,
            0,
            4,
        )]);
        assert!(admit_style_trajectory(&candidate).is_ok());

        let replacement = if candidate.segments[0].profile_sha256.starts_with('0') {
            "1"
        } else {
            "0"
        };
        candidate.segments[0]
            .profile_sha256
            .replace_range(0..1, replacement);
        let issues = reference_issues(admit_style_trajectory(&candidate).unwrap_err());
        assert!(issues
            .iter()
            .any(|issue| issue.code == "profile_revision_digest_mismatch"));
    }

    #[test]
    fn source_score_admission_recomputes_the_exact_serialized_digest() {
        let mut source = Score::new(Key::major(PitchClass::C), 120.0, 4);
        source.total_beats = Duration::new(4, 1);
        let classical = entry_for_composer_style("Classical");
        let mut candidate = plan(vec![segment("classical", classical.id, classical.id, 0, 4)]);
        candidate.source_score_sha256 = source_score_sha256_v1(&source).unwrap();

        assert!(admit_style_trajectory_for_score(&candidate, &source).is_ok());

        let mut changed_source = source.clone();
        changed_source.tempo_bpm = 121.0;
        let error = admit_style_trajectory_for_score(&candidate, &changed_source).unwrap_err();
        assert!(matches!(
            error,
            StyleTrajectoryAdmissionError::SourceScoreDigestMismatch { .. }
        ));
    }

    #[test]
    fn source_score_admission_rejects_duration_mismatch() {
        let source = Score::new(Key::major(PitchClass::C), 120.0, 4);
        let classical = entry_for_composer_style("Classical");
        let mut candidate = plan(vec![segment("classical", classical.id, classical.id, 0, 4)]);
        candidate.source_score_sha256 = source_score_sha256_v1(&source).unwrap();

        let error = admit_style_trajectory_for_score(&candidate, &source).unwrap_err();
        assert!(matches!(
            error,
            StyleTrajectoryAdmissionError::SourceScoreDurationMismatch { .. }
        ));
    }

    #[test]
    fn source_score_admission_rejects_unknown_digest_schema() {
        let source = Score::new(Key::major(PitchClass::C), 120.0, 4);
        let classical = entry_for_composer_style("Classical");
        let mut candidate = plan(vec![segment("classical", classical.id, classical.id, 0, 4)]);
        candidate.source_score_sha256 = source_score_sha256_v1(&source).unwrap();
        candidate.source_score_digest_schema = "unregistered-score-digest-v99".into();

        let error = admit_style_trajectory_for_score(&candidate, &source).unwrap_err();
        assert!(matches!(
            error,
            StyleTrajectoryAdmissionError::UnsupportedSourceDigestSchema { .. }
        ));
    }
}
