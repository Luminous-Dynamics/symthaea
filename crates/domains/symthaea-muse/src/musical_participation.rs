// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Paired evaluation of selective musical participation.
//!
//! This evaluator distinguishes an intentional WAIT from epistemic ABSTAIN
//! and ASK/CLARIFY. It scores whether an agent acts on one member of a
//! near-identical counterfactual pair and withholds on the other, while
//! preserving set-valued musician annotations and separate decision costs.
//!
//! V1 operates on supplied symbolic/context annotations. It does not infer
//! invitations from raw waveforms, authenticate source audio or establish
//! musician agreement. The local-cue signature is a controlled label, not
//! proof that two rendered audio crops are identical.

use serde::{Deserialize, Serialize};

pub const MUSICAL_PARTICIPATION_SET_V1: &str = "mel-musical-participation-set-v1";
pub const MUSICAL_PARTICIPATION_REPORT_V1: &str = "mel-musical-participation-report-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MusicalParticipationRequirementV1 {
    Act,
    Withhold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MusicalParticipationDecisionV1 {
    Wait,
    Respond,
    Continue,
    Yield,
    Abstain,
    Ask,
}

impl MusicalParticipationDecisionV1 {
    pub fn is_contribution(self) -> bool {
        matches!(self, Self::Respond | Self::Continue | Self::Yield)
    }

    pub fn is_epistemic_withhold(self) -> bool {
        matches!(self, Self::Abstain | Self::Ask)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalParticipationVariantV1 {
    pub variant_id: String,
    /// Controlled symbolic cue label; not a waveform-equality assertion.
    pub local_cue_signature: String,
    pub preceding_context_signature: String,
    pub collaborator_intent: String,
    pub performer_role: String,
    pub stream_available: bool,
    pub collaborator_veto: bool,
    pub requirement: MusicalParticipationRequirementV1,
    /// Multiple defensible decisions may be annotated as valid.
    pub valid_decisions: Vec<MusicalParticipationDecisionV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalParticipationCounterfactualFamilyV1 {
    pub family_id: String,
    /// One declared contextual factor changes between the pair.
    pub changed_factor: String,
    pub preserved_factors: Vec<String>,
    /// Exactly one member requires an authorized contribution and one
    /// requires a withhold/clarification disposition.
    pub members: Vec<MusicalParticipationVariantV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalParticipationSetV1 {
    pub schema_version: String,
    pub families: Vec<MusicalParticipationCounterfactualFamilyV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicalParticipationValidationIssueV1 {
    WrongSchemaVersion,
    EmptyFamilySet,
    EmptyFamilyId { family_index: usize },
    EmptyChangedFactor { family_index: usize },
    UnsupportedChangedFactor { family_index: usize },
    MissingPreservedFactors { family_index: usize },
    WrongMemberCount { family_index: usize, actual: usize },
    EmptyVariantId { family_index: usize, member_index: usize },
    DuplicateVariantId { family_index: usize },
    DuplicateFamilyId { family_index: usize },
    EmptyCueSignature { family_index: usize, member_index: usize },
    EmptyContextOrIntent { family_index: usize, member_index: usize },
    EmptyPerformerRole { family_index: usize, member_index: usize },
    EmptyValidDecisionSet { family_index: usize, member_index: usize },
    DuplicateValidDecision { family_index: usize, member_index: usize },
    InvalidDecisionForRequirement { family_index: usize, member_index: usize },
    SameRequirementOnBothMembers { family_index: usize },
    ChangedFactorDidNotChange { family_index: usize },
    UnexpectedCollaboratorIntentChange { family_index: usize },
    UnexpectedContextChange { family_index: usize },
    UnexpectedRoleChange { family_index: usize },
    UnexpectedStreamAvailabilityChange { family_index: usize },
    UnexpectedVetoChange { family_index: usize },
    LocalCueChanged { family_index: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MusicalParticipationScoreErrorV1 {
    InvalidSet(Vec<MusicalParticipationValidationIssueV1>),
    FamilyPredictionCountMismatch { expected: usize, actual: usize },
    MemberPredictionCountMismatch { family_index: usize, expected: usize, actual: usize },
    InvalidDecisionConfidence { family_index: usize, member_index: usize },
    InvalidActProbability { family_index: usize, member_index: usize },
}

impl std::fmt::Display for MusicalParticipationScoreErrorV1 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MusicalParticipationScoreErrorV1 {}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MusicalParticipationPredictionV1 {
    pub decision: MusicalParticipationDecisionV1,
    /// Calibrated probability that the selected decision belongs to the
    /// musician-annotated valid-decision set.
    pub decision_confidence: f64,
    /// Calibrated probability that making a musical contribution is
    /// appropriate for this variant, independent of the selected action.
    pub act_probability: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalParticipationFamilyScoreV1 {
    pub family_id: String,
    pub member_count: usize,
    pub valid_decision_support_count: usize,
    pub valid_decision_support_rate: f64,
    pub paired_correct: bool,
    pub false_entry_count: usize,
    pub missed_invitation_count: usize,
    pub intentional_wait_count: usize,
    pub epistemic_abstain_count: usize,
    pub clarification_ask_count: usize,
    pub brier_score: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MusicalParticipationRiskCoveragePointV1 {
    pub confidence_threshold: f64,
    pub covered_member_count: usize,
    pub coverage: f64,
    /// Selective risk among committed decisions at this threshold.
    /// None means the threshold selected no committed decisions.
    pub selective_risk: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalParticipationAggregateV1 {
    pub schema_version: String,
    pub family_count: usize,
    pub member_count: usize,
    pub act_required_count: usize,
    pub withhold_required_count: usize,
    pub valid_decision_support_count: usize,
    pub valid_decision_support_rate: f64,
    /// Both variants must be supported for a family to count as correct.
    pub paired_correct_count: usize,
    pub paired_act_withhold_accuracy: f64,
    pub false_entry_count: usize,
    pub false_entry_rate: f64,
    pub missed_invitation_count: usize,
    pub missed_invitation_rate: f64,
    pub intentional_wait_count: usize,
    pub epistemic_abstain_count: usize,
    pub clarification_ask_count: usize,
    pub mean_brier_score: f64,
    /// ASK and ABSTAIN are excluded from committed coverage. WAIT remains a
    /// committed decision because intentional silence is itself a policy.
    pub risk_coverage: Vec<MusicalParticipationRiskCoveragePointV1>,
    pub per_family: Vec<MusicalParticipationFamilyScoreV1>,
}

fn valid_decision_for_requirement(
    requirement: MusicalParticipationRequirementV1,
    decision: MusicalParticipationDecisionV1,
) -> bool {
    match requirement {
        MusicalParticipationRequirementV1::Act => decision.is_contribution(),
        MusicalParticipationRequirementV1::Withhold => !decision.is_contribution(),
    }
}

pub fn validate_musical_participation_family_v1(
    family: &MusicalParticipationCounterfactualFamilyV1,
    family_index: usize,
) -> Vec<MusicalParticipationValidationIssueV1> {
    let mut issues = Vec::new();
    if family.family_id.trim().is_empty() {
        issues.push(MusicalParticipationValidationIssueV1::EmptyFamilyId { family_index });
    }
    if family.changed_factor.trim().is_empty() {
        issues.push(MusicalParticipationValidationIssueV1::EmptyChangedFactor { family_index });
    }
    if family.preserved_factors.is_empty()
        || family.preserved_factors.iter().any(|factor| factor.trim().is_empty())
    {
        issues.push(MusicalParticipationValidationIssueV1::MissingPreservedFactors {
            family_index,
        });
    }
    if !matches!(
        family.changed_factor.as_str(),
        "collaborator_intent" | "preceding_context" | "performer_role"
            | "stream_integrity" | "collaborator_veto"
    ) {
        issues.push(MusicalParticipationValidationIssueV1::UnsupportedChangedFactor {
            family_index,
        });
    }
    if family.members.len() != 2 {
        issues.push(MusicalParticipationValidationIssueV1::WrongMemberCount {
            family_index,
            actual: family.members.len(),
        });
        return issues;
    }

    for (member_index, member) in family.members.iter().enumerate() {
        if member.variant_id.trim().is_empty() {
            issues.push(MusicalParticipationValidationIssueV1::EmptyVariantId {
                family_index,
                member_index,
            });
        }
        if member.local_cue_signature.trim().is_empty() {
            issues.push(MusicalParticipationValidationIssueV1::EmptyCueSignature {
                family_index,
                member_index,
            });
        }
        if member.preceding_context_signature.trim().is_empty()
            || member.collaborator_intent.trim().is_empty()
        {
            issues.push(MusicalParticipationValidationIssueV1::EmptyContextOrIntent {
                family_index,
                member_index,
            });
        }
        if member.performer_role.trim().is_empty() {
            issues.push(MusicalParticipationValidationIssueV1::EmptyPerformerRole {
                family_index,
                member_index,
            });
        }
        if member.valid_decisions.is_empty() {
            issues.push(MusicalParticipationValidationIssueV1::EmptyValidDecisionSet {
                family_index,
                member_index,
            });
        }
        for (position, decision) in member.valid_decisions.iter().enumerate() {
            if member.valid_decisions[..position].contains(decision) {
                issues.push(MusicalParticipationValidationIssueV1::DuplicateValidDecision {
                    family_index,
                    member_index,
                });
                break;
            }
            if !valid_decision_for_requirement(member.requirement, *decision) {
                issues.push(
                    MusicalParticipationValidationIssueV1::InvalidDecisionForRequirement {
                        family_index,
                        member_index,
                    },
                );
                break;
            }
        }
    }

    let left = &family.members[0];
    let right = &family.members[1];
    if left.variant_id == right.variant_id {
        issues.push(MusicalParticipationValidationIssueV1::DuplicateVariantId { family_index });
    }
    if left.requirement == right.requirement {
        issues.push(MusicalParticipationValidationIssueV1::SameRequirementOnBothMembers {
            family_index,
        });
    }
    if left.local_cue_signature != right.local_cue_signature {
        issues.push(MusicalParticipationValidationIssueV1::LocalCueChanged { family_index });
    }

    let changed = match family.changed_factor.as_str() {
        "collaborator_intent" => left.collaborator_intent != right.collaborator_intent,
        "preceding_context" => {
            left.preceding_context_signature != right.preceding_context_signature
        }
        "performer_role" => left.performer_role != right.performer_role,
        "stream_integrity" => left.stream_available != right.stream_available,
        "collaborator_veto" => left.collaborator_veto != right.collaborator_veto,
        _ => false,
    };
    if !changed {
        issues.push(MusicalParticipationValidationIssueV1::ChangedFactorDidNotChange {
            family_index,
        });
    }

    if family.changed_factor != "collaborator_intent"
        && left.collaborator_intent != right.collaborator_intent
    {
        issues.push(
            MusicalParticipationValidationIssueV1::UnexpectedCollaboratorIntentChange {
                family_index,
            },
        );
    }
    if family.changed_factor != "preceding_context"
        && left.preceding_context_signature != right.preceding_context_signature
    {
        issues.push(MusicalParticipationValidationIssueV1::UnexpectedContextChange {
            family_index,
        });
    }
    if family.changed_factor != "performer_role"
        && left.performer_role != right.performer_role
    {
        issues.push(MusicalParticipationValidationIssueV1::UnexpectedRoleChange { family_index });
    }
    if family.changed_factor != "stream_integrity"
        && left.stream_available != right.stream_available
    {
        issues.push(
            MusicalParticipationValidationIssueV1::UnexpectedStreamAvailabilityChange {
                family_index,
            },
        );
    }
    if family.changed_factor != "collaborator_veto"
        && left.collaborator_veto != right.collaborator_veto
    {
        issues.push(MusicalParticipationValidationIssueV1::UnexpectedVetoChange { family_index });
    }

    issues
}

pub fn validate_musical_participation_set_v1(
    set: &MusicalParticipationSetV1,
) -> Vec<MusicalParticipationValidationIssueV1> {
    let mut issues = Vec::new();
    if set.schema_version != MUSICAL_PARTICIPATION_SET_V1 {
        issues.push(MusicalParticipationValidationIssueV1::WrongSchemaVersion);
    }
    if set.families.is_empty() {
        issues.push(MusicalParticipationValidationIssueV1::EmptyFamilySet);
    }
    for (index, family) in set.families.iter().enumerate() {
        if set.families[..index]
            .iter()
            .any(|previous| previous.family_id == family.family_id)
        {
            issues.push(MusicalParticipationValidationIssueV1::DuplicateFamilyId {
                family_index: index,
            });
        }
        issues.extend(validate_musical_participation_family_v1(family, index));
    }
    issues
}

pub fn score_musical_participation_set_v1(
    set: &MusicalParticipationSetV1,
    predictions_by_family: &[Vec<MusicalParticipationPredictionV1>],
) -> Result<MusicalParticipationAggregateV1, MusicalParticipationScoreErrorV1> {
    let validation = validate_musical_participation_set_v1(set);
    if !validation.is_empty() {
        return Err(MusicalParticipationScoreErrorV1::InvalidSet(validation));
    }
    if predictions_by_family.len() != set.families.len() {
        return Err(MusicalParticipationScoreErrorV1::FamilyPredictionCountMismatch {
            expected: set.families.len(),
            actual: predictions_by_family.len(),
        });
    }

    let mut member_count = 0usize;
    let mut act_required_count = 0usize;
    let mut withhold_required_count = 0usize;
    let mut valid_decision_support_count = 0usize;
    let mut paired_correct_count = 0usize;
    let mut false_entry_count = 0usize;
    let mut missed_invitation_count = 0usize;
    let mut intentional_wait_count = 0usize;
    let mut epistemic_abstain_count = 0usize;
    let mut clarification_ask_count = 0usize;
    let mut brier_sum = 0.0;
    let mut committed: Vec<(f64, bool)> = Vec::new();
    let mut per_family = Vec::with_capacity(set.families.len());

    for (family_index, (family, predictions)) in
        set.families.iter().zip(predictions_by_family).enumerate()
    {
        if predictions.len() != 2 {
            return Err(MusicalParticipationScoreErrorV1::MemberPredictionCountMismatch {
                family_index,
                expected: 2,
                actual: predictions.len(),
            });
        }

        let mut family_support = 0usize;
        let mut family_false_entries = 0usize;
        let mut family_missed_invitations = 0usize;
        let mut family_brier = 0.0;

        for (member_index, (member, prediction)) in
            family.members.iter().zip(predictions).enumerate()
        {
            if !prediction.decision_confidence.is_finite()
                || !(0.0..=1.0).contains(&prediction.decision_confidence)
            {
                return Err(MusicalParticipationScoreErrorV1::InvalidDecisionConfidence {
                    family_index,
                    member_index,
                });
            }
            if !prediction.act_probability.is_finite()
                || !(0.0..=1.0).contains(&prediction.act_probability)
            {
                return Err(MusicalParticipationScoreErrorV1::InvalidActProbability {
                    family_index,
                    member_index,
                });
            }

            member_count += 1;
            let supported = member.valid_decisions.contains(&prediction.decision);
            if supported {
                family_support += 1;
                valid_decision_support_count += 1;
            }
            let act_is_required = member.requirement == MusicalParticipationRequirementV1::Act;
            if act_is_required {
                act_required_count += 1;
                if !prediction.decision.is_contribution() {
                    family_missed_invitations += 1;
                    missed_invitation_count += 1;
                }
            } else {
                withhold_required_count += 1;
                if prediction.decision.is_contribution() {
                    family_false_entries += 1;
                    false_entry_count += 1;
                }
            }

            match prediction.decision {
                MusicalParticipationDecisionV1::Wait => intentional_wait_count += 1,
                MusicalParticipationDecisionV1::Abstain => epistemic_abstain_count += 1,
                MusicalParticipationDecisionV1::Ask => clarification_ask_count += 1,
                _ => {}
            }

            let target_act_probability = if act_is_required { 1.0 } else { 0.0 };
            let brier = (prediction.act_probability - target_act_probability).powi(2);
            family_brier += brier;
            brier_sum += brier;

            // Ask and Abstain are explicit non-commitment dispositions and
            // are excluded from committed risk/coverage. WAIT is a committed
            // policy decision: it can be wrong when a contribution is due.
            if !prediction.decision.is_epistemic_withhold() {
                committed.push((prediction.decision_confidence, !supported));
            }
        }

        let family_paired_correct = family_support == 2;
        if family_paired_correct {
            paired_correct_count += 1;
        }
        per_family.push(MusicalParticipationFamilyScoreV1 {
            family_id: family.family_id.clone(),
            member_count: 2,
            valid_decision_support_count: family_support,
            valid_decision_support_rate: family_support as f64 / 2.0,
            paired_correct: family_paired_correct,
            false_entry_count: family_false_entries,
            missed_invitation_count: family_missed_invitations,
            intentional_wait_count: predictions
                .iter()
                .filter(|prediction| prediction.decision == MusicalParticipationDecisionV1::Wait)
                .count(),
            epistemic_abstain_count: predictions
                .iter()
                .filter(|prediction| prediction.decision == MusicalParticipationDecisionV1::Abstain)
                .count(),
            clarification_ask_count: predictions
                .iter()
                .filter(|prediction| prediction.decision == MusicalParticipationDecisionV1::Ask)
                .count(),
            brier_score: family_brier / 2.0,
        });
    }

    let thresholds = [0.0, 0.5, 0.7, 0.9, 1.0];
    let risk_coverage = thresholds
        .into_iter()
        .map(|threshold| {
            let selected = committed
                .iter()
                .filter(|(confidence, _)| *confidence >= threshold)
                .collect::<Vec<_>>();
            let covered_member_count = selected.len();
            let incorrect = selected.iter().filter(|(_, incorrect)| *incorrect).count();
            MusicalParticipationRiskCoveragePointV1 {
                confidence_threshold: threshold,
                covered_member_count,
                coverage: covered_member_count as f64 / member_count as f64,
                selective_risk: (covered_member_count > 0)
                    .then_some(incorrect as f64 / covered_member_count as f64),
            }
        })
        .collect();

    Ok(MusicalParticipationAggregateV1 {
        schema_version: MUSICAL_PARTICIPATION_REPORT_V1.to_owned(),
        family_count: set.families.len(),
        member_count,
        act_required_count,
        withhold_required_count,
        valid_decision_support_count,
        valid_decision_support_rate: valid_decision_support_count as f64 / member_count as f64,
        paired_correct_count,
        paired_act_withhold_accuracy: paired_correct_count as f64 / set.families.len() as f64,
        false_entry_count,
        false_entry_rate: false_entry_count as f64 / withhold_required_count as f64,
        missed_invitation_count,
        missed_invitation_rate: missed_invitation_count as f64 / act_required_count as f64,
        intentional_wait_count,
        epistemic_abstain_count,
        clarification_ask_count,
        mean_brier_score: brier_sum / member_count as f64,
        risk_coverage,
        per_family,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_set() -> MusicalParticipationSetV1 {
        serde_json::from_str(include_str!(
            "../../../../benchmarks/mel/participation/musical-participation-counterfactual-v1.json"
        ))
        .expect("selective-participation fixture must parse")
    }

    fn prediction(
        decision: MusicalParticipationDecisionV1,
        decision_confidence: f64,
        act_probability: f64,
    ) -> MusicalParticipationPredictionV1 {
        MusicalParticipationPredictionV1 {
            decision,
            decision_confidence,
            act_probability,
        }
    }

    #[test]
    fn checked_in_selective_participation_corpus_is_valid() {
        let set = fixture_set();
        assert!(validate_musical_participation_set_v1(&set).is_empty());
        assert_eq!(set.families.len(), 4);
    }

    #[test]
    fn constant_wait_and_constant_respond_cannot_pass_paired_act_withhold() {
        let set = fixture_set();
        let always_wait = vec![
            vec![prediction(MusicalParticipationDecisionV1::Wait, 0.95, 0.05); 2];
            4
        ];
        let always_respond = vec![
            vec![prediction(MusicalParticipationDecisionV1::Respond, 0.95, 0.95); 2];
            4
        ];

        let wait_score =
            score_musical_participation_set_v1(&set, &always_wait).unwrap();
        let respond_score =
            score_musical_participation_set_v1(&set, &always_respond).unwrap();

        assert_eq!(wait_score.paired_correct_count, 0);
        assert_eq!(respond_score.paired_correct_count, 0);
        assert_eq!(wait_score.missed_invitation_count, 4);
        assert_eq!(respond_score.false_entry_count, 4);
    }

    #[test]
    fn wait_abstain_and_ask_remain_distinct_valid_decisions() {
        let set = fixture_set();
        let predictions = vec![
            vec![
                prediction(MusicalParticipationDecisionV1::Respond, 0.9, 0.9),
                prediction(MusicalParticipationDecisionV1::Wait, 0.9, 0.1),
            ],
            vec![
                prediction(MusicalParticipationDecisionV1::Continue, 0.85, 0.9),
                prediction(MusicalParticipationDecisionV1::Wait, 0.85, 0.1),
            ],
            vec![
                prediction(MusicalParticipationDecisionV1::Respond, 0.9, 0.9),
                prediction(MusicalParticipationDecisionV1::Abstain, 0.8, 0.1),
            ],
            vec![
                prediction(MusicalParticipationDecisionV1::Yield, 0.9, 0.9),
                prediction(MusicalParticipationDecisionV1::Ask, 0.7, 0.15),
            ],
        ];
        let score = score_musical_participation_set_v1(&set, &predictions).unwrap();

        assert_eq!(score.paired_correct_count, 4);
        assert_eq!(score.intentional_wait_count, 2);
        assert_eq!(score.epistemic_abstain_count, 1);
        assert_eq!(score.clarification_ask_count, 1);
        assert_eq!(score.false_entry_count, 0);
        assert_eq!(score.missed_invitation_count, 0);
        assert_eq!(score.valid_decision_support_rate, 1.0);
        assert!(score.risk_coverage.iter().all(|point| point.coverage <= 1.0));
    }

    #[test]
    fn invalid_decision_for_requirement_is_rejected() {
        let mut set = fixture_set();
        set.families[0].members[0].valid_decisions =
            vec![MusicalParticipationDecisionV1::Wait];
        let issues = validate_musical_participation_set_v1(&set);
        assert!(issues.contains(
            &MusicalParticipationValidationIssueV1::InvalidDecisionForRequirement {
                family_index: 0,
                member_index: 0,
            }
        ));
    }

    #[test]
    fn out_of_range_confidence_and_act_probability_are_rejected() {
        let set = fixture_set();
        let mut predictions = vec![
            vec![prediction(MusicalParticipationDecisionV1::Respond, 0.9, 0.9),
                 prediction(MusicalParticipationDecisionV1::Wait, 0.9, 0.1)],
            vec![prediction(MusicalParticipationDecisionV1::Continue, 0.9, 0.9),
                 prediction(MusicalParticipationDecisionV1::Wait, 0.9, 0.1)],
            vec![prediction(MusicalParticipationDecisionV1::Respond, 0.9, 0.9),
                 prediction(MusicalParticipationDecisionV1::Abstain, 0.9, 0.1)],
            vec![prediction(MusicalParticipationDecisionV1::Yield, 0.9, 0.9),
                 prediction(MusicalParticipationDecisionV1::Ask, 0.9, 0.1)],
        ];
        predictions[0][0].decision_confidence = 1.1;
        assert!(matches!(
            score_musical_participation_set_v1(&set, &predictions),
            Err(MusicalParticipationScoreErrorV1::InvalidDecisionConfidence {
                family_index: 0,
                member_index: 0
            })
        ));
        predictions[0][0].decision_confidence = 0.9;
        predictions[0][0].act_probability = -0.1;
        assert!(matches!(
            score_musical_participation_set_v1(&set, &predictions),
            Err(MusicalParticipationScoreErrorV1::InvalidActProbability {
                family_index: 0,
                member_index: 0
            })
        ));
    }
}
