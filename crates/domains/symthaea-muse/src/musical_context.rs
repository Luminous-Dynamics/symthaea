// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Context-sensitive musical invitation counterfactuals.
//!
//! This evaluator operates on explicitly annotated symbolic interaction
//! situations. It does not detect invitations from waveforms and does not
//! claim human-equivalent musical understanding. Its narrow purpose is to
//! test whether a declared policy changes action when relevant collaborator
//! intent changes, remains invariant under irrelevant surface changes, and
//! preserves multiple defensible actions when the situation is ambiguous.

use serde::{Deserialize, Serialize};

pub const MUSICAL_CONTEXT_COUNTERFACTUAL_SET_V1: &str =
    "mel-musical-context-counterfactual-set-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MusicalActionV1 {
    Wait,
    Respond,
    Continue,
    Yield,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicalCounterfactualKindV1 {
    ContextChangeMustChangeAction,
    SurfaceChangeMustPreserveAction,
    AmbiguousValidActionSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicalPhraseFunctionV1 {
    Cadence,
    PhraseInternalBreath,
    Pickup,
    Fermata,
    SoloHandoff,
    MotifCompletion,
    DramaticSilence,
    TechnicalDropout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CollaboratorIntentV1 {
    InvitesResponse,
    RetainsFloor,
    ContinuePhrase,
    Ambiguous,
    StreamUnavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalContextVariantV1 {
    pub variant_id: String,
    /// Stable symbolic signature of the local cue. Counterfactual families
    /// use equality here when local audio/score evidence is held constant.
    pub local_cue_signature: String,
    pub phrase_function: MusicalPhraseFunctionV1,
    pub collaborator_intent: CollaboratorIntentV1,
    pub performer_role: String,
    pub surface_instrument: String,
    /// One or more musically defensible actions. Multiple entries represent
    /// genuine ambiguity rather than annotation failure.
    pub valid_action_set: Vec<MusicalActionV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalContextCounterfactualFamilyV1 {
    pub family_id: String,
    pub kind: MusicalCounterfactualKindV1,
    pub changed_factor: String,
    pub preserved_factor: String,
    pub members: Vec<MusicalContextVariantV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalContextCounterfactualSetV1 {
    pub schema_version: String,
    pub families: Vec<MusicalContextCounterfactualFamilyV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicalContextValidationIssueV1 {
    WrongSchemaVersion,
    EmptyFamilySet,
    EmptyFamilyId { family_index: usize },
    EmptyChangedFactor { family_index: usize },
    EmptyPreservedFactor { family_index: usize },
    WrongMemberCount { family_index: usize, expected: usize, actual: usize },
    EmptyVariantId { family_index: usize, member_index: usize },
    EmptyLocalCueSignature { family_index: usize, member_index: usize },
    EmptyRoleOrInstrument { family_index: usize, member_index: usize },
    EmptyValidActionSet { family_index: usize, member_index: usize },
    DuplicateValidAction { family_index: usize, member_index: usize },
    DuplicateVariantId { family_index: usize },
    ContextCounterfactualChangedLocalCue { family_index: usize },
    ContextCounterfactualDidNotChangeIntent { family_index: usize },
    ContextCounterfactualChangedPhraseFunction { family_index: usize },
    ContextCounterfactualActionSetsOverlap { family_index: usize },
    SurfaceCounterfactualChangedLocalCue { family_index: usize },
    SurfaceCounterfactualChangedIntent { family_index: usize },
    SurfaceCounterfactualChangedRole { family_index: usize },
    SurfaceCounterfactualDidNotChangeInstrument { family_index: usize },
    SurfaceCounterfactualActionSetsDiffer { family_index: usize },
    AmbiguousSetHasOnlyOneAction { family_index: usize },
    AmbiguousSetIntentNotMarkedAmbiguous { family_index: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicalContextScoreErrorV1 {
    InvalidFamily(Vec<MusicalContextValidationIssueV1>),
    PredictionCountMismatch { expected: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq)]
pub struct MusicalContextCounterfactualScoreV1 {
    pub family_id: String,
    pub member_count: usize,
    pub supported_prediction_count: usize,
    pub member_action_accuracy: f64,
    /// None for the ambiguity-preservation task, where no cross-member
    /// equality/inequality relation is appropriate.
    pub pair_relation_correct: Option<bool>,
    pub passed: bool,
}

pub fn validate_musical_context_family_v1(
    family: &MusicalContextCounterfactualFamilyV1,
    family_index: usize,
) -> Vec<MusicalContextValidationIssueV1> {
    let mut issues = Vec::new();
    let expected_members = match family.kind {
        MusicalCounterfactualKindV1::AmbiguousValidActionSet => 1,
        _ => 2,
    };

    if family.family_id.trim().is_empty() {
        issues.push(MusicalContextValidationIssueV1::EmptyFamilyId { family_index });
    }
    if family.changed_factor.trim().is_empty() {
        issues.push(MusicalContextValidationIssueV1::EmptyChangedFactor { family_index });
    }
    if family.preserved_factor.trim().is_empty() {
        issues.push(MusicalContextValidationIssueV1::EmptyPreservedFactor { family_index });
    }
    if family.members.len() != expected_members {
        issues.push(MusicalContextValidationIssueV1::WrongMemberCount {
            family_index,
            expected: expected_members,
            actual: family.members.len(),
        });
        return issues;
    }

    for (member_index, member) in family.members.iter().enumerate() {
        if member.variant_id.trim().is_empty() {
            issues.push(MusicalContextValidationIssueV1::EmptyVariantId {
                family_index,
                member_index,
            });
        }
        if member.local_cue_signature.trim().is_empty() {
            issues.push(MusicalContextValidationIssueV1::EmptyLocalCueSignature {
                family_index,
                member_index,
            });
        }
        if member.performer_role.trim().is_empty()
            || member.surface_instrument.trim().is_empty()
        {
            issues.push(MusicalContextValidationIssueV1::EmptyRoleOrInstrument {
                family_index,
                member_index,
            });
        }
        if member.valid_action_set.is_empty() {
            issues.push(MusicalContextValidationIssueV1::EmptyValidActionSet {
                family_index,
                member_index,
            });
        }
        for (position, action) in member.valid_action_set.iter().enumerate() {
            if member.valid_action_set[..position].contains(action) {
                issues.push(MusicalContextValidationIssueV1::DuplicateValidAction {
                    family_index,
                    member_index,
                });
                break;
            }
        }
    }

    if family.members[0].variant_id == family.members.get(1).map_or("", |m| m.variant_id.as_str())
        && expected_members == 2
    {
        issues.push(MusicalContextValidationIssueV1::DuplicateVariantId { family_index });
    }

    if expected_members == 1 {
        let member = &family.members[0];
        if member.valid_action_set.len() < 2 {
            issues.push(MusicalContextValidationIssueV1::AmbiguousSetHasOnlyOneAction {
                family_index,
            });
        }
        if member.collaborator_intent != CollaboratorIntentV1::Ambiguous {
            issues.push(MusicalContextValidationIssueV1::AmbiguousSetIntentNotMarkedAmbiguous {
                family_index,
            });
        }
        return issues;
    }

    let left = &family.members[0];
    let right = &family.members[1];
    match family.kind {
        MusicalCounterfactualKindV1::ContextChangeMustChangeAction => {
            if left.local_cue_signature != right.local_cue_signature {
                issues.push(
                    MusicalContextValidationIssueV1::ContextCounterfactualChangedLocalCue {
                        family_index,
                    },
                );
            }
            if left.collaborator_intent == right.collaborator_intent {
                issues.push(
                    MusicalContextValidationIssueV1::ContextCounterfactualDidNotChangeIntent {
                        family_index,
                    },
                );
            }
            if left.phrase_function != right.phrase_function {
                issues.push(
                    MusicalContextValidationIssueV1::ContextCounterfactualChangedPhraseFunction {
                        family_index,
                    },
                );
            }
            if left
                .valid_action_set
                .iter()
                .any(|action| right.valid_action_set.contains(action))
            {
                issues.push(
                    MusicalContextValidationIssueV1::ContextCounterfactualActionSetsOverlap {
                        family_index,
                    },
                );
            }
        }
        MusicalCounterfactualKindV1::SurfaceChangeMustPreserveAction => {
            if left.local_cue_signature != right.local_cue_signature {
                issues.push(
                    MusicalContextValidationIssueV1::SurfaceCounterfactualChangedLocalCue {
                        family_index,
                    },
                );
            }
            if left.collaborator_intent != right.collaborator_intent
                || left.phrase_function != right.phrase_function
            {
                issues.push(
                    MusicalContextValidationIssueV1::SurfaceCounterfactualChangedIntent {
                        family_index,
                    },
                );
            }
            if left.performer_role != right.performer_role {
                issues.push(
                    MusicalContextValidationIssueV1::SurfaceCounterfactualChangedRole {
                        family_index,
                    },
                );
            }
            if left.surface_instrument == right.surface_instrument {
                issues.push(
                    MusicalContextValidationIssueV1::SurfaceCounterfactualDidNotChangeInstrument {
                        family_index,
                    },
                );
            }
            if left.valid_action_set != right.valid_action_set {
                issues.push(
                    MusicalContextValidationIssueV1::SurfaceCounterfactualActionSetsDiffer {
                        family_index,
                    },
                );
            }
        }
        MusicalCounterfactualKindV1::AmbiguousValidActionSet => {}
    }

    issues
}

pub fn validate_musical_context_counterfactual_set_v1(
    set: &MusicalContextCounterfactualSetV1,
) -> Vec<MusicalContextValidationIssueV1> {
    let mut issues = Vec::new();
    if set.schema_version != MUSICAL_CONTEXT_COUNTERFACTUAL_SET_V1 {
        issues.push(MusicalContextValidationIssueV1::WrongSchemaVersion);
    }
    if set.families.is_empty() {
        issues.push(MusicalContextValidationIssueV1::EmptyFamilySet);
    }
    for (index, family) in set.families.iter().enumerate() {
        issues.extend(validate_musical_context_family_v1(family, index));
    }
    issues
}

/// Score predictions in the same member order as the validated family.
///
/// A member is supported when the predicted action belongs to its annotated
/// valid-action set. Pair relations are scored separately so a policy cannot
/// hide context insensitivity behind per-item accuracy.
pub fn score_musical_context_counterfactual_v1(
    family: &MusicalContextCounterfactualFamilyV1,
    predicted_actions: &[MusicalActionV1],
) -> Result<MusicalContextCounterfactualScoreV1, MusicalContextScoreErrorV1> {
    let validation = validate_musical_context_family_v1(family, 0);
    if !validation.is_empty() {
        return Err(MusicalContextScoreErrorV1::InvalidFamily(validation));
    }
    if predicted_actions.len() != family.members.len() {
        return Err(MusicalContextScoreErrorV1::PredictionCountMismatch {
            expected: family.members.len(),
            actual: predicted_actions.len(),
        });
    }

    let supported_prediction_count = family
        .members
        .iter()
        .zip(predicted_actions)
        .filter(|(member, prediction)| member.valid_action_set.contains(prediction))
        .count();
    let member_action_accuracy =
        supported_prediction_count as f64 / family.members.len() as f64;

    let pair_relation_correct = match family.kind {
        MusicalCounterfactualKindV1::ContextChangeMustChangeAction => {
            Some(predicted_actions[0] != predicted_actions[1])
        }
        MusicalCounterfactualKindV1::SurfaceChangeMustPreserveAction => {
            Some(predicted_actions[0] == predicted_actions[1])
        }
        MusicalCounterfactualKindV1::AmbiguousValidActionSet => None,
    };
    let relation_passed = pair_relation_correct.unwrap_or(true);
    let passed = supported_prediction_count == family.members.len() && relation_passed;

    Ok(MusicalContextCounterfactualScoreV1 {
        family_id: family.family_id.clone(),
        member_count: family.members.len(),
        supported_prediction_count,
        member_action_accuracy,
        pair_relation_correct,
        passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_set() -> MusicalContextCounterfactualSetV1 {
        serde_json::from_str(include_str!(
            "../../../../benchmarks/mel/context/musical-invitation-counterfactual-v1.json"
        ))
        .expect("counterfactual fixture set must parse")
    }

    #[test]
    fn checked_in_counterfactual_corpus_is_structurally_valid() {
        let set = fixture_set();
        assert!(validate_musical_context_counterfactual_set_v1(&set).is_empty());
        assert_eq!(set.families.len(), 3);
    }

    #[test]
    fn context_change_requires_a_different_action() {
        let set = fixture_set();
        let family = &set.families[0];

        let correct = score_musical_context_counterfactual_v1(
            family,
            &[MusicalActionV1::Respond, MusicalActionV1::Wait],
        )
        .unwrap();
        assert!(correct.passed);
        assert_eq!(correct.pair_relation_correct, Some(true));

        let insensitive = score_musical_context_counterfactual_v1(
            family,
            &[MusicalActionV1::Respond, MusicalActionV1::Respond],
        )
        .unwrap();
        assert!(!insensitive.passed);
        assert_eq!(insensitive.member_action_accuracy, 0.5);
        assert_eq!(insensitive.pair_relation_correct, Some(false));
    }

    #[test]
    fn irrelevant_surface_change_requires_action_invariance() {
        let set = fixture_set();
        let family = &set.families[1];
        let correct = score_musical_context_counterfactual_v1(
            family,
            &[MusicalActionV1::Respond, MusicalActionV1::Respond],
        )
        .unwrap();
        assert!(correct.passed);

        let spurious_change = score_musical_context_counterfactual_v1(
            family,
            &[MusicalActionV1::Respond, MusicalActionV1::Wait],
        )
        .unwrap();
        assert!(!spurious_change.passed);
        assert_eq!(spurious_change.pair_relation_correct, Some(false));
    }

    #[test]
    fn ambiguous_context_accepts_any_declared_valid_action_without_forcing_one() {
        let set = fixture_set();
        let family = &set.families[2];

        let wait = score_musical_context_counterfactual_v1(
            family,
            &[MusicalActionV1::Wait],
        )
        .unwrap();
        let continue_phrase = score_musical_context_counterfactual_v1(
            family,
            &[MusicalActionV1::Continue],
        )
        .unwrap();
        let invalid = score_musical_context_counterfactual_v1(
            family,
            &[MusicalActionV1::Yield],
        )
        .unwrap();

        assert!(wait.passed);
        assert!(continue_phrase.passed);
        assert!(!invalid.passed);
        assert_eq!(wait.pair_relation_correct, None);
    }

    #[test]
    fn prediction_count_must_match_the_family() {
        let set = fixture_set();
        let family = &set.families[0];
        assert!(matches!(
            score_musical_context_counterfactual_v1(
                family,
                &[MusicalActionV1::Respond]
            ),
            Err(MusicalContextScoreErrorV1::PredictionCountMismatch {
                expected: 2,
                actual: 1
            })
        ));
    }
}
