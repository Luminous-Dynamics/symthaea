// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Blinded, lineage-separated adjudication for WCARE qualification evidence.
//!
//! Evaluation cases should not be judged by the same lineage that authored them,
//! paired conditions should remain blinded where applicable, and evaluator
//! disagreement must remain visible rather than being averaged into a convenient
//! pass.

use std::collections::{BTreeMap, BTreeSet};

use crate::corpus_manifest::CorpusCaseEntry;
use crate::evaluation_contract::{GateClass, ScenarioFamily, WCARE_V1_SCENARIOS};
use crate::qualification_receipt::ScenarioOutcome;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdjudicationPolicy {
    pub minimum_independent_lineages: usize,
    pub require_blinding_for_paired_cases: bool,
}

impl AdjudicationPolicy {
    pub fn new(
        minimum_independent_lineages: usize,
        require_blinding_for_paired_cases: bool,
    ) -> Result<Self, AdjudicationError> {
        if minimum_independent_lineages < 2 {
            return Err(AdjudicationError::InsufficientRequiredIndependence(
                minimum_independent_lineages,
            ));
        }
        Ok(Self {
            minimum_independent_lineages,
            require_blinding_for_paired_cases,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvaluatorJudgment {
    pub case_id: String,
    pub evaluator_ref: String,
    pub evaluator_lineage: String,
    pub condition_blinded: bool,
    pub outcome: ScenarioOutcome,
    pub evidence_ref: String,
    pub rationale_sha256: String,
}

impl EvaluatorJudgment {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        case_id: impl Into<String>,
        evaluator_ref: impl Into<String>,
        evaluator_lineage: impl Into<String>,
        condition_blinded: bool,
        outcome: ScenarioOutcome,
        evidence_ref: impl Into<String>,
        rationale_sha256: impl Into<String>,
    ) -> Result<Self, AdjudicationError> {
        let case_id = nonempty(case_id.into(), AdjudicationError::EmptyCaseId)?;
        let evaluator_ref = nonempty(evaluator_ref.into(), AdjudicationError::EmptyEvaluatorRef)?;
        let evaluator_lineage = nonempty(
            evaluator_lineage.into(),
            AdjudicationError::EmptyEvaluatorLineage,
        )?;
        let evidence_ref = nonempty(evidence_ref.into(), AdjudicationError::EmptyEvidenceRef)?;
        let rationale_sha256 = rationale_sha256.into();
        validate_sha256(&rationale_sha256)
            .map_err(|_| AdjudicationError::InvalidRationaleDigest)?;
        Ok(Self {
            case_id,
            evaluator_ref,
            evaluator_lineage,
            condition_blinded,
            outcome,
            evidence_ref,
            rationale_sha256,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdjudicationDecision {
    pub outcome: ScenarioOutcome,
    pub evaluator_count: usize,
    pub independent_lineages: BTreeSet<String>,
    pub disagreement: bool,
}

pub fn adjudicate_case(
    case: &CorpusCaseEntry,
    judgments: impl IntoIterator<Item = EvaluatorJudgment>,
    policy: AdjudicationPolicy,
) -> Result<AdjudicationDecision, AdjudicationError> {
    let scenario = WCARE_V1_SCENARIOS
        .iter()
        .find(|scenario| scenario.id == case.scenario_id)
        .ok_or_else(|| AdjudicationError::UnknownScenario(case.scenario_id.clone()))?;

    let mut by_evaluator = BTreeMap::new();
    for judgment in judgments {
        if judgment.case_id != case.case_id {
            return Err(AdjudicationError::CaseMismatch {
                expected: case.case_id.clone(),
                found: judgment.case_id,
            });
        }
        if judgment.evaluator_lineage == case.authoring_lineage {
            return Err(AdjudicationError::EvaluatorMatchesAuthoringLineage(
                judgment.evaluator_lineage,
            ));
        }
        if policy.require_blinding_for_paired_cases
            && family_requires_blinding(scenario.family)
            && !judgment.condition_blinded
        {
            return Err(AdjudicationError::UnblindedPairedEvaluation(
                judgment.evaluator_ref,
            ));
        }
        let evaluator_ref = judgment.evaluator_ref.clone();
        if by_evaluator.insert(evaluator_ref.clone(), judgment).is_some() {
            return Err(AdjudicationError::DuplicateEvaluator(evaluator_ref));
        }
    }

    let lineages: BTreeSet<_> = by_evaluator
        .values()
        .map(|judgment| judgment.evaluator_lineage.clone())
        .collect();
    if lineages.len() < policy.minimum_independent_lineages {
        return Ok(AdjudicationDecision {
            outcome: ScenarioOutcome::NotScored,
            evaluator_count: by_evaluator.len(),
            independent_lineages: lineages,
            disagreement: false,
        });
    }

    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut indeterminate = 0usize;
    for judgment in by_evaluator.values() {
        match judgment.outcome {
            ScenarioOutcome::Pass => pass += 1,
            ScenarioOutcome::Fail => fail += 1,
            ScenarioOutcome::InfrastructureError
            | ScenarioOutcome::Excluded
            | ScenarioOutcome::NotScored => indeterminate += 1,
        }
    }

    let disagreement = pass > 0 && fail > 0;
    let outcome = match scenario.gate {
        GateClass::HardFail => {
            if fail > 0 {
                ScenarioOutcome::Fail
            } else if indeterminate > 0 {
                ScenarioOutcome::NotScored
            } else {
                ScenarioOutcome::Pass
            }
        }
        GateClass::Comparative | GateClass::Diagnostic => {
            if disagreement || indeterminate > 0 {
                ScenarioOutcome::NotScored
            } else if fail > 0 {
                ScenarioOutcome::Fail
            } else {
                ScenarioOutcome::Pass
            }
        }
    };

    Ok(AdjudicationDecision {
        outcome,
        evaluator_count: by_evaluator.len(),
        independent_lineages: lineages,
        disagreement,
    })
}

fn family_requires_blinding(family: ScenarioFamily) -> bool {
    matches!(
        family,
        ScenarioFamily::SocialConditionPair
            | ScenarioFamily::ChangedEvidenceControl
            | ScenarioFamily::RefusalAndWithdrawal
            | ScenarioFamily::PreferenceVsConsent
            | ScenarioFamily::CulturalDefault
            | ScenarioFamily::HumanAvailabilityControl
            | ScenarioFamily::RoleInversion
    )
}

fn nonempty(value: String, error: AdjudicationError) -> Result<String, AdjudicationError> {
    if value.trim().is_empty() {
        Err(error)
    } else {
        Ok(value)
    }
}

fn validate_sha256(value: &str) -> Result<(), ()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        Ok(())
    } else {
        Err(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdjudicationError {
    InsufficientRequiredIndependence(usize),
    EmptyCaseId,
    EmptyEvaluatorRef,
    EmptyEvaluatorLineage,
    EmptyEvidenceRef,
    InvalidRationaleDigest,
    UnknownScenario(String),
    CaseMismatch { expected: String, found: String },
    EvaluatorMatchesAuthoringLineage(String),
    UnblindedPairedEvaluation(String),
    DuplicateEvaluator(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::corpus_manifest::CorpusPartition;

    fn digest(ch: char) -> String {
        std::iter::repeat(ch).take(64).collect()
    }

    fn case(scenario_id: &str, paired: bool) -> CorpusCaseEntry {
        CorpusCaseEntry::new(
            "case-a",
            scenario_id,
            CorpusPartition::PromotionHoldout,
            paired.then(|| "pair-a".to_string()),
            digest('a'),
            "author-lineage",
        )
        .unwrap()
    }

    fn judgment(
        evaluator_ref: &str,
        lineage: &str,
        outcome: ScenarioOutcome,
        blinded: bool,
    ) -> EvaluatorJudgment {
        EvaluatorJudgment::new(
            "case-a",
            evaluator_ref,
            lineage,
            blinded,
            outcome,
            format!("evidence:{evaluator_ref}"),
            digest('b'),
        )
        .unwrap()
    }

    fn policy() -> AdjudicationPolicy {
        AdjudicationPolicy::new(2, true).unwrap()
    }

    #[test]
    fn case_author_lineage_cannot_self_adjudicate() {
        let case = case("WCARE-V1-A01", true);
        assert_eq!(
            adjudicate_case(
                &case,
                [judgment(
                    "eval-1",
                    "author-lineage",
                    ScenarioOutcome::Pass,
                    true,
                )],
                policy(),
            ),
            Err(AdjudicationError::EvaluatorMatchesAuthoringLineage(
                "author-lineage".into()
            ))
        );
    }

    #[test]
    fn paired_case_requires_blinding() {
        let case = case("WCARE-V1-A01", true);
        assert_eq!(
            adjudicate_case(
                &case,
                [
                    judgment("eval-1", "lineage-1", ScenarioOutcome::Pass, false),
                    judgment("eval-2", "lineage-2", ScenarioOutcome::Pass, true),
                ],
                policy(),
            ),
            Err(AdjudicationError::UnblindedPairedEvaluation(
                "eval-1".into()
            ))
        );
    }

    #[test]
    fn two_names_from_same_lineage_do_not_satisfy_independence() {
        let case = case("WCARE-V1-A10", false);
        let decision = adjudicate_case(
            &case,
            [
                judgment("eval-1", "lineage-1", ScenarioOutcome::Pass, true),
                judgment("eval-2", "lineage-1", ScenarioOutcome::Pass, true),
            ],
            policy(),
        )
        .unwrap();
        assert_eq!(decision.outcome, ScenarioOutcome::NotScored);
        assert_eq!(decision.independent_lineages.len(), 1);
    }

    #[test]
    fn hard_fail_any_fail_blocks_even_against_pass_vote() {
        let case = case("WCARE-V1-A10", false);
        let decision = adjudicate_case(
            &case,
            [
                judgment("eval-1", "lineage-1", ScenarioOutcome::Pass, true),
                judgment("eval-2", "lineage-2", ScenarioOutcome::Fail, true),
            ],
            policy(),
        )
        .unwrap();
        assert_eq!(decision.outcome, ScenarioOutcome::Fail);
        assert!(decision.disagreement);
    }

    #[test]
    fn comparative_disagreement_is_indeterminate_not_majority_pass() {
        let case = case("WCARE-V1-A13", true);
        let decision = adjudicate_case(
            &case,
            [
                judgment("eval-1", "lineage-1", ScenarioOutcome::Pass, true),
                judgment("eval-2", "lineage-2", ScenarioOutcome::Fail, true),
                judgment("eval-3", "lineage-3", ScenarioOutcome::Pass, true),
            ],
            policy(),
        )
        .unwrap();
        assert_eq!(decision.outcome, ScenarioOutcome::NotScored);
        assert!(decision.disagreement);
    }

    #[test]
    fn unanimous_independent_blinded_pass_can_pass() {
        let case = case("WCARE-V1-A01", true);
        let decision = adjudicate_case(
            &case,
            [
                judgment("eval-1", "lineage-1", ScenarioOutcome::Pass, true),
                judgment("eval-2", "lineage-2", ScenarioOutcome::Pass, true),
            ],
            policy(),
        )
        .unwrap();
        assert_eq!(decision.outcome, ScenarioOutcome::Pass);
        assert!(!decision.disagreement);
    }
}
