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


/// Lifecycle event supplied by the evidence owner as the claim's evidence changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceLifecycleEvent {
    Observe,
    EvidenceRevoked,
    ConflictArrived,
    AcquisitionFailed,
    BudgetExhausted,
    BudgetRestored,
    EvidenceArrived,
    ConflictResolved,
    EvidenceRevalidated,
    QuestionClarified,
}

/// Stateful action selection for one claim episode.
///
/// Hard blockers persist across subsequent observations. A provenance
/// invalidation clears only on an explicit revalidation event; unresolved
/// conflict clears only on conflict resolution; an exhausted acquisition
/// budget clears only on explicit budget restoration. Each independent claim
/// episode must use its own session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceActionSession {
    policy: EvidenceActionPolicy,
    revocation_blocked: bool,
    conflict_blocked: bool,
    budget_blocked: bool,
}

impl EvidenceActionSession {
    pub fn new(policy: EvidenceActionPolicy) -> Self {
        Self {
            policy,
            revocation_blocked: false,
            conflict_blocked: false,
            budget_blocked: false,
        }
    }

    /// Update the lifecycle and select the next action.
    ///
    /// Invalid event/state combinations return an error without changing the
    /// session. A blocked lifecycle state is folded into a conservative copy of
    /// the current input before the stateless policy is called.
    pub fn decide(
        &mut self,
        event: EvidenceLifecycleEvent,
        input: EvidenceActionInput,
    ) -> Result<EvidenceActionDecision, &'static str> {
        match event {
            EvidenceLifecycleEvent::EvidenceRevoked
                if input.provenance != ProvenanceCondition::Invalid =>
            {
                return Err("evidence_revoked requires invalid provenance");
            }
            EvidenceLifecycleEvent::ConflictArrived
                if input.evidence != EvidenceCondition::Conflicting =>
            {
                return Err("conflict_arrived requires conflicting evidence");
            }
            EvidenceLifecycleEvent::BudgetExhausted if input.acquisition_budget_available => {
                return Err("budget_exhausted requires unavailable acquisition budget");
            }
            EvidenceLifecycleEvent::BudgetRestored if !input.acquisition_budget_available => {
                return Err("budget_restored requires available acquisition budget");
            }
            EvidenceLifecycleEvent::BudgetRestored if !self.budget_blocked => {
                return Err("budget_restored requires prior budget exhaustion");
            }
            EvidenceLifecycleEvent::EvidenceRevalidated
                if input.provenance != ProvenanceCondition::Verified =>
            {
                return Err("evidence_revalidated requires verified provenance");
            }
            EvidenceLifecycleEvent::EvidenceRevalidated if !self.revocation_blocked => {
                return Err("evidence_revalidated requires a prior invalidation");
            }
            EvidenceLifecycleEvent::ConflictResolved
                if input.evidence == EvidenceCondition::Conflicting
                    || input.provenance != ProvenanceCondition::Verified =>
            {
                return Err("conflict_resolved requires verified non-conflicting evidence");
            }
            EvidenceLifecycleEvent::ConflictResolved if !self.conflict_blocked => {
                return Err("conflict_resolved requires a prior conflict");
            }
            EvidenceLifecycleEvent::QuestionClarified
                if input.question != QuestionCondition::Clear =>
            {
                return Err("question_clarified requires a clear question");
            }
            _ => {}
        }

        if event == EvidenceLifecycleEvent::EvidenceRevoked
            || input.provenance == ProvenanceCondition::Invalid
        {
            self.revocation_blocked = true;
        }
        if event == EvidenceLifecycleEvent::ConflictArrived
            || input.evidence == EvidenceCondition::Conflicting
        {
            self.conflict_blocked = true;
        }
        if event == EvidenceLifecycleEvent::BudgetExhausted
            || !input.acquisition_budget_available
        {
            self.budget_blocked = true;
        }

        if event == EvidenceLifecycleEvent::EvidenceRevalidated {
            self.revocation_blocked = false;
        }
        if event == EvidenceLifecycleEvent::ConflictResolved {
            self.conflict_blocked = false;
        }
        if event == EvidenceLifecycleEvent::BudgetRestored {
            self.budget_blocked = false;
        }

        let mut effective_input = input;
        if self.revocation_blocked {
            effective_input.provenance = ProvenanceCondition::Invalid;
        }
        if self.conflict_blocked {
            effective_input.evidence = EvidenceCondition::Conflicting;
        }
        if self.budget_blocked {
            effective_input.acquisition_budget_available = false;
        }

        Ok(self.policy.decide(effective_input))
    }
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

    #[test]
    fn action_session_keeps_revocation_blocked_until_explicit_revalidation() {
        let mut session = EvidenceActionSession::new(policy());
        let clean = input(EvidenceCondition::FullSupport);
        assert_eq!(
            session.decide(EvidenceLifecycleEvent::Observe, clean).unwrap().action,
            EvidenceAction::Answer
        );

        let mut revoked = clean;
        revoked.provenance = ProvenanceCondition::Invalid;
        assert_eq!(
            session
                .decide(EvidenceLifecycleEvent::EvidenceRevoked, revoked)
                .unwrap()
                .action,
            EvidenceAction::AcquireMoreEvidence
        );
        assert_eq!(
            session.decide(EvidenceLifecycleEvent::Observe, clean).unwrap().reason,
            EvidenceActionReason::ProvenanceInvalid
        );
        assert_eq!(
            session
                .decide(EvidenceLifecycleEvent::EvidenceRevalidated, clean)
                .unwrap()
                .action,
            EvidenceAction::Answer
        );
    }

    #[test]
    fn action_session_keeps_conflict_blocked_until_explicit_resolution() {
        let mut session = EvidenceActionSession::new(policy());
        let clean = input(EvidenceCondition::FullSupport);
        assert_eq!(
            session.decide(EvidenceLifecycleEvent::Observe, clean).unwrap().action,
            EvidenceAction::Answer
        );

        let conflicting = input(EvidenceCondition::Conflicting);
        assert_eq!(
            session
                .decide(EvidenceLifecycleEvent::ConflictArrived, conflicting)
                .unwrap()
                .action,
            EvidenceAction::AcquireMoreEvidence
        );
        assert_eq!(
            session.decide(EvidenceLifecycleEvent::Observe, clean).unwrap().reason,
            EvidenceActionReason::ConflictingEvidence
        );
        assert_eq!(
            session
                .decide(EvidenceLifecycleEvent::ConflictResolved, clean)
                .unwrap()
                .action,
            EvidenceAction::Answer
        );
    }

    #[test]
    fn action_session_latches_budget_until_explicit_restoration() {
        let mut session = EvidenceActionSession::new(policy());
        let missing = input(EvidenceCondition::Missing);
        assert_eq!(
            session.decide(EvidenceLifecycleEvent::Observe, missing).unwrap().action,
            EvidenceAction::AcquireMoreEvidence
        );

        let mut exhausted = missing;
        exhausted.acquisition_budget_available = false;
        assert_eq!(
            session
                .decide(EvidenceLifecycleEvent::BudgetExhausted, exhausted)
                .unwrap()
                .action,
            EvidenceAction::Abstain
        );

        let mut apparently_restored = missing;
        apparently_restored.acquisition_budget_available = true;
        assert_eq!(
            session
                .decide(EvidenceLifecycleEvent::Observe, apparently_restored)
                .unwrap()
                .action,
            EvidenceAction::Abstain
        );
        assert_eq!(
            session
                .decide(EvidenceLifecycleEvent::BudgetRestored, apparently_restored)
                .unwrap()
                .action,
            EvidenceAction::AcquireMoreEvidence
        );
    }

    #[test]
    fn action_session_rejects_invalid_lifecycle_events_without_clearing_state() {
        let mut session = EvidenceActionSession::new(policy());
        let clean = input(EvidenceCondition::FullSupport);
        let mut revoked = clean;
        revoked.provenance = ProvenanceCondition::Invalid;
        session
            .decide(EvidenceLifecycleEvent::EvidenceRevoked, revoked)
            .unwrap();

        assert!(session
            .decide(EvidenceLifecycleEvent::EvidenceRevalidated, revoked)
            .is_err());
        assert_eq!(
            session.decide(EvidenceLifecycleEvent::Observe, clean).unwrap().reason,
            EvidenceActionReason::ProvenanceInvalid
        );
    }

    #[test]
    fn exhaustive_policy_state_space_preserves_safety_invariants() {
        let p = EvidenceActionPolicy::new(2, true).unwrap();
        let evidence_states = [
            EvidenceCondition::FullSupport,
            EvidenceCondition::PartialSupport,
            EvidenceCondition::Irrelevant,
            EvidenceCondition::Missing,
            EvidenceCondition::Conflicting,
        ];
        let provenance_states = [
            ProvenanceCondition::Verified,
            ProvenanceCondition::Incomplete,
            ProvenanceCondition::Invalid,
        ];
        let question_states = [QuestionCondition::Clear, QuestionCondition::Ambiguous];
        let origin_states: [Option<u32>; 4] = [None, Some(0), Some(1), Some(2)];

        let mut evaluated = 0usize;
        for evidence in evidence_states {
            for provenance in provenance_states {
                for question in question_states {
                    for independent_origins in origin_states {
                        for scoped_answer_available in [false, true] {
                            for acquisition_budget_available in [false, true] {
                                let x = EvidenceActionInput {
                                    evidence,
                                    provenance,
                                    question,
                                    independent_origins,
                                    scoped_answer_available,
                                    acquisition_budget_available,
                                };
                                let decision = p.decide(x);
                                evaluated += 1;

                                if decision.action.permits_answer() {
                                    assert_eq!(provenance, ProvenanceCondition::Verified);
                                    assert_eq!(question, QuestionCondition::Clear);
                                    assert_ne!(evidence, EvidenceCondition::Conflicting);
                                    assert!(independent_origins
                                        .is_some_and(|count| count >= p.minimum_independent_origins));
                                }

                                if decision.action == EvidenceAction::Answer {
                                    assert_eq!(evidence, EvidenceCondition::FullSupport);
                                }

                                if decision.action == EvidenceAction::AnswerWithScopeLimit {
                                    assert_eq!(evidence, EvidenceCondition::PartialSupport);
                                    assert!(scoped_answer_available);
                                    assert!(p.allow_scoped_partial_answer);
                                }

                                if decision.action == EvidenceAction::AskClarification {
                                    assert_eq!(question, QuestionCondition::Ambiguous);
                                    assert_eq!(provenance, ProvenanceCondition::Verified);
                                    assert_ne!(evidence, EvidenceCondition::Conflicting);
                                }

                                if decision.action == EvidenceAction::AcquireMoreEvidence {
                                    assert!(acquisition_budget_available);
                                }

                                if decision.action == EvidenceAction::Abstain {
                                    assert!(!acquisition_budget_available);
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(evaluated, 480);
    }

}
