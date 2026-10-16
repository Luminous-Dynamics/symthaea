// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic action selection for evidence-aware reasoning.
//!
//! Paradigm shift: do not collapse epistemic state into one confidence score
//! and a binary answer/abstain threshold. The next action can be to answer,
//! state a bounded conclusion, clarify the question, acquire evidence, or abstain.
//! This module is a small policy kernel: upstream code must supply evidence
//! sufficiency, provenance status, question clarity, and lineage-deduplicated
//! independent-origin counts.
//!
//! Structural provenance validity is not factual truth. Independent-origin counts
//! must come from provenance assessment, not raw URLs or citation count. This
//! policy does not fetch sources, infer source independence, calculate
//! probabilities, or certify claims.

/// Assessment of how well the currently available evidence addresses the claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceCondition {
    /// Available evidence supports the exact claim being considered.
    FullSupport,
    /// Evidence supports only a narrower or explicitly qualified conclusion.
    PartialSupport,
    /// Available evidence is not relevant to the question.
    Irrelevant,
    /// No usable external evidence is available.
    Missing,
    /// Evidence includes unresolved contradictory support.
    Conflicting,
}

/// Integrity and completeness of the provenance chain for the evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceCondition {
    /// Integrity, lifecycle, and required derivation links passed upstream checks.
    Verified,
    /// Required provenance facts or links remain unresolved.
    Incomplete,
    /// Integrity failure, revocation, withdrawal, or another hard invalidation.
    Invalid,
}

/// Whether the user's question is sufficiently specified to answer as asked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuestionCondition {
    Clear,
    Ambiguous,
}

/// Next system action. Only the first two actions permit a substantive answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceAction {
    /// Answer the exact claim under the configured policy.
    Answer,
    /// Return only a narrower claim whose scope is explicitly supported.
    AnswerWithScopeLimit,
    /// Request clarification before spending effort on the wrong question.
    AskClarification,
    /// Seek missing provenance, corroboration, or resolving evidence.
    AcquireMoreEvidence,
    /// Make no substantive claim because the policy cannot safely proceed.
    Abstain,
}

impl EvidenceAction {
    /// Whether this action permits returning a substantive conclusion.
    pub fn permits_answer(self) -> bool {
        matches!(self, Self::Answer | Self::AnswerWithScopeLimit)
    }
}

/// Stable, machine-readable explanation for an action decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceActionReason {
    CompleteSupport,
    ScopedPartialSupport,
    ProvenanceIncomplete,
    ProvenanceInvalid,
    ConflictingEvidence,
    QuestionNeedsClarification,
    EvidenceMissing,
    EvidenceIrrelevant,
    IndependentOriginsUnknown,
    IndependentOriginsBelowPolicy,
    PartialSupportNotSafelyScoped,
}

/// Inputs are assessments, not raw citations. Callers must independently
/// validate or derive these values before invoking the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceActionInput {
    pub evidence: EvidenceCondition,
    pub provenance: ProvenanceCondition,
    pub question: QuestionCondition,
    /// Distinct credited primary-origin groups after lineage deduplication.
    /// None means lineage or independence is unresolved.
    pub independent_origins: Option<u32>,
    /// True only when a narrower, clearly worded conclusion is supportable.
    pub scoped_answer_available: bool,
    /// Whether the system is allowed to acquire additional evidence now.
    pub acquisition_budget_available: bool,
}

/// Explicit policy thresholds. These are policy controls, not calibrated
/// confidence probabilities or empirically established domain defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceActionPolicy {
    /// Minimum independent primary-origin groups required before answering.
    pub minimum_independent_origins: u32,
    /// Permit a scoped answer when evidence is partial but the scope is known.
    pub allow_scoped_partial_answer: bool,
}

impl EvidenceActionPolicy {
    pub fn new(
        minimum_independent_origins: u32,
        allow_scoped_partial_answer: bool,
    ) -> Result<Self, &'static str> {
        if minimum_independent_origins == 0 {
            return Err("minimum_independent_origins must be greater than zero");
        }
        Ok(Self {
            minimum_independent_origins,
            allow_scoped_partial_answer,
        })
    }

    /// Conservative structural default: one credited origin and no automatic
    /// widening of a partial result. Domain owners should set thresholds explicitly.
    pub fn bounded_default() -> Self {
        Self {
            minimum_independent_origins: 1,
            allow_scoped_partial_answer: false,
        }
    }

    /// Choose a next action without presenting partial evidence as complete.
    ///
    /// Precedence is intentional:
    /// 1. Hard provenance invalidation and conflict block substantive answers.
    /// 2. Unresolved provenance is repaired before answering.
    /// 3. Ambiguous questions are clarified before more retrieval is spent.
    /// 4. Missing/irrelevant evidence or inadequate origin diversity triggers
    ///    acquisition, or abstention when acquisition is unavailable.
    /// 5. Only then may complete or explicitly scoped evidence be answered.
    pub fn decide(&self, input: EvidenceActionInput) -> EvidenceActionDecision {
        if input.provenance == ProvenanceCondition::Invalid {
            return self.acquire_or_abstain(
                input.acquisition_budget_available,
                EvidenceActionReason::ProvenanceInvalid,
            );
        }

        if input.evidence == EvidenceCondition::Conflicting {
            return self.acquire_or_abstain(
                input.acquisition_budget_available,
                EvidenceActionReason::ConflictingEvidence,
            );
        }

        if input.provenance == ProvenanceCondition::Incomplete {
            return self.acquire_or_abstain(
                input.acquisition_budget_available,
                EvidenceActionReason::ProvenanceIncomplete,
            );
        }

        if input.question == QuestionCondition::Ambiguous {
            return EvidenceActionDecision {
                action: EvidenceAction::AskClarification,
                reason: EvidenceActionReason::QuestionNeedsClarification,
            };
        }

        match input.evidence {
            EvidenceCondition::Missing => {
                return self.acquire_or_abstain(
                    input.acquisition_budget_available,
                    EvidenceActionReason::EvidenceMissing,
                );
            }
            EvidenceCondition::Irrelevant => {
                return self.acquire_or_abstain(
                    input.acquisition_budget_available,
                    EvidenceActionReason::EvidenceIrrelevant,
                );
            }
            EvidenceCondition::Conflicting => {
                return self.acquire_or_abstain(
                    input.acquisition_budget_available,
                    EvidenceActionReason::ConflictingEvidence,
                );
            }
            EvidenceCondition::FullSupport | EvidenceCondition::PartialSupport => {}
        }

        match input.independent_origins {
            None => {
                return self.acquire_or_abstain(
                    input.acquisition_budget_available,
                    EvidenceActionReason::IndependentOriginsUnknown,
                );
            }
            Some(count) if count < self.minimum_independent_origins => {
                return self.acquire_or_abstain(
                    input.acquisition_budget_available,
                    EvidenceActionReason::IndependentOriginsBelowPolicy,
                );
            }
            Some(_) => {}
        }

        match input.evidence {
            EvidenceCondition::FullSupport => EvidenceActionDecision {
                action: EvidenceAction::Answer,
                reason: EvidenceActionReason::CompleteSupport,
            },
            EvidenceCondition::PartialSupport
                if self.allow_scoped_partial_answer && input.scoped_answer_available =>
            {
                EvidenceActionDecision {
                    action: EvidenceAction::AnswerWithScopeLimit,
                    reason: EvidenceActionReason::ScopedPartialSupport,
                }
            }
            EvidenceCondition::PartialSupport => self.acquire_or_abstain(
                input.acquisition_budget_available,
                EvidenceActionReason::PartialSupportNotSafelyScoped,
            ),
            EvidenceCondition::Missing
            | EvidenceCondition::Irrelevant
            | EvidenceCondition::Conflicting => unreachable!("handled above"),
        }
    }

    fn acquire_or_abstain(
        &self,
        acquisition_budget_available: bool,
        reason: EvidenceActionReason,
    ) -> EvidenceActionDecision {
        EvidenceActionDecision {
            action: if acquisition_budget_available {
                EvidenceAction::AcquireMoreEvidence
            } else {
                EvidenceAction::Abstain
            },
            reason,
        }
    }
}

/// Policy output. No probability or truth score is implied by this decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceActionDecision {
    pub action: EvidenceAction,
    pub reason: EvidenceActionReason,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(evidence: EvidenceCondition) -> EvidenceActionInput {
        EvidenceActionInput {
            evidence,
            provenance: ProvenanceCondition::Verified,
            question: QuestionCondition::Clear,
            independent_origins: Some(1),
            scoped_answer_available: false,
            acquisition_budget_available: true,
        }
    }

    fn policy() -> EvidenceActionPolicy {
        EvidenceActionPolicy::bounded_default()
    }

    #[test]
    fn complete_verified_support_answers() {
        let decision = policy().decide(input(EvidenceCondition::FullSupport));
        assert_eq!(decision.action, EvidenceAction::Answer);
        assert_eq!(decision.reason, EvidenceActionReason::CompleteSupport);
        assert!(decision.action.permits_answer());
    }

    #[test]
    fn partial_support_does_not_expand_to_a_full_answer() {
        let decision = policy().decide(input(EvidenceCondition::PartialSupport));
        assert_eq!(decision.action, EvidenceAction::AcquireMoreEvidence);
        assert!(!decision.action.permits_answer());
    }

    #[test]
    fn safe_scoped_partial_answer_requires_explicit_policy_and_scope() {
        let p = EvidenceActionPolicy::new(1, true).unwrap();
        let mut x = input(EvidenceCondition::PartialSupport);
        x.scoped_answer_available = true;
        let decision = p.decide(x);
        assert_eq!(decision.action, EvidenceAction::AnswerWithScopeLimit);
        assert_eq!(decision.reason, EvidenceActionReason::ScopedPartialSupport);
    }

    #[test]
    fn partial_answer_is_not_allowed_when_scoped_conclusion_is_unavailable() {
        let p = EvidenceActionPolicy::new(1, true).unwrap();
        let decision = p.decide(input(EvidenceCondition::PartialSupport));
        assert_eq!(decision.action, EvidenceAction::AcquireMoreEvidence);
        assert!(!decision.action.permits_answer());
    }

    #[test]
    fn conflicting_evidence_never_selects_a_winner_automatically() {
        for provenance in [
            ProvenanceCondition::Verified,
            ProvenanceCondition::Incomplete,
            ProvenanceCondition::Invalid,
        ] {
            let mut x = input(EvidenceCondition::Conflicting);
            x.provenance = provenance;
            let decision = policy().decide(x);
            assert!(!decision.action.permits_answer());
            assert!(matches!(
                decision.action,
                EvidenceAction::AcquireMoreEvidence | EvidenceAction::Abstain
            ));
        }
    }

    #[test]
    fn invalid_provenance_blocks_answers_for_every_evidence_condition() {
        for evidence in [
            EvidenceCondition::FullSupport,
            EvidenceCondition::PartialSupport,
            EvidenceCondition::Irrelevant,
            EvidenceCondition::Missing,
            EvidenceCondition::Conflicting,
        ] {
            let mut x = input(evidence);
            x.provenance = ProvenanceCondition::Invalid;
            let decision = policy().decide(x);
            assert!(!decision.action.permits_answer());
            assert_eq!(decision.reason, EvidenceActionReason::ProvenanceInvalid);
        }
    }

    #[test]
    fn incomplete_provenance_blocks_answers() {
        let mut x = input(EvidenceCondition::FullSupport);
        x.provenance = ProvenanceCondition::Incomplete;
        assert_eq!(
            policy().decide(x).action,
            EvidenceAction::AcquireMoreEvidence
        );
    }

    #[test]
    fn ambiguous_question_requests_clarification_before_retrieval() {
        let mut x = input(EvidenceCondition::Missing);
        x.question = QuestionCondition::Ambiguous;
        assert_eq!(
            policy().decide(x).action,
            EvidenceAction::AskClarification
        );
    }

    #[test]
    fn hard_provenance_failure_precedes_question_clarification() {
        let mut x = input(EvidenceCondition::FullSupport);
        x.provenance = ProvenanceCondition::Invalid;
        x.question = QuestionCondition::Ambiguous;
        assert_eq!(
            policy().decide(x).action,
            EvidenceAction::AcquireMoreEvidence
        );
    }

    #[test]
    fn missing_or_irrelevant_evidence_acquires_or_abstains() {
        for evidence in [EvidenceCondition::Missing, EvidenceCondition::Irrelevant] {
            let mut x = input(evidence);
            assert_eq!(
                policy().decide(x).action,
                EvidenceAction::AcquireMoreEvidence
            );
            x.acquisition_budget_available = false;
            assert_eq!(policy().decide(x).action, EvidenceAction::Abstain);
        }
    }

    #[test]
    fn unknown_lineage_never_counts_as_independent_origins() {
        let mut x = input(EvidenceCondition::FullSupport);
        x.independent_origins = None;
        assert_eq!(
            policy().decide(x).action,
            EvidenceAction::AcquireMoreEvidence
        );
        x.acquisition_budget_available = false;
        assert_eq!(policy().decide(x).action, EvidenceAction::Abstain);
    }

    #[test]
    fn duplicated_or_correlated_sources_below_threshold_do_not_qualify() {
        let p = EvidenceActionPolicy::new(2, false).unwrap();
        let mut x = input(EvidenceCondition::FullSupport);
        x.independent_origins = Some(1);
        assert_eq!(p.decide(x).action, EvidenceAction::AcquireMoreEvidence);
        x.independent_origins = Some(2);
        assert_eq!(p.decide(x).action, EvidenceAction::Answer);
    }

    #[test]
    fn zero_origin_policy_is_rejected() {
        assert!(EvidenceActionPolicy::new(0, false).is_err());
    }

    #[test]
    fn exhausted_budget_never_turns_failure_into_answer() {
        let p = EvidenceActionPolicy::new(2, true).unwrap();
        let mut x = input(EvidenceCondition::FullSupport);
        x.independent_origins = Some(1);
        x.acquisition_budget_available = false;
        assert_eq!(p.decide(x).action, EvidenceAction::Abstain);
    }

    #[test]
    fn action_answer_permission_matches_only_substantive_answer_actions() {
        assert!(EvidenceAction::Answer.permits_answer());
        assert!(EvidenceAction::AnswerWithScopeLimit.permits_answer());
        assert!(!EvidenceAction::AskClarification.permits_answer());
        assert!(!EvidenceAction::AcquireMoreEvidence.permits_answer());
        assert!(!EvidenceAction::Abstain.permits_answer());
    }
}
