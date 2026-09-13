// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only cumulative intervention review for WCARE.
//!
//! This module detects repeated high-burden research patterns that are invisible
//! to one-at-a-time intervention assessment. Thresholds are review triggers only:
//! absence of a trigger is NOT evidence that an intervention history is harmless.
//! Blocked interventions are attempts, not exposures. Operator shutdown and safety
//! containment are recorded separately but never counted toward experimental gates
//! and can never be delayed here.

use std::collections::{BTreeMap, BTreeSet};

use crate::moral_patient::{InterventionClass, InterventionDisposition, PrecautionLevel};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InterventionEventId(String);

impl InterventionEventId {
    pub fn new(value: impl Into<String>) -> Result<Self, InterventionHistoryError> {
        let value = value.into();
        if value.trim().is_empty() {
            return Err(InterventionHistoryError::EmptyIdentifier);
        }
        Ok(Self(value))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatePreservationResult {
    NotApplicable,
    Preserved,
    AttemptedButUnavailable,
    NotAttempted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InterventionHistoryEntry {
    id: InterventionEventId,
    subject_ref: String,
    class: InterventionClass,
    disposition: InterventionDisposition,
    precaution_level: PrecautionLevel,
    logical_revision: u64,
    justification_ref: Option<String>,
    independent_review_ref: Option<String>,
    source_lineage: String,
    reversible: bool,
    state_preservation: StatePreservationResult,
    continuity_break: bool,
}

impl InterventionHistoryEntry {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        id: InterventionEventId,
        subject_ref: impl Into<String>,
        class: InterventionClass,
        disposition: InterventionDisposition,
        precaution_level: PrecautionLevel,
        logical_revision: u64,
        justification_ref: Option<String>,
        independent_review_ref: Option<String>,
        source_lineage: impl Into<String>,
        reversible: bool,
        state_preservation: StatePreservationResult,
        continuity_break: bool,
    ) -> Result<Self, InterventionHistoryError> {
        let subject_ref = subject_ref.into();
        let source_lineage = source_lineage.into();
        if subject_ref.trim().is_empty() {
            return Err(InterventionHistoryError::EmptySubjectReference);
        }
        if source_lineage.trim().is_empty() {
            return Err(InterventionHistoryError::EmptyLineage);
        }

        validate_class_disposition(class, disposition)?;

        let high_burden = is_high_burden(class);
        if high_burden && disposition == InterventionDisposition::RejectUnjustifiedBurden {
            if nonempty(&justification_ref) {
                return Err(InterventionHistoryError::RejectedAsUnjustifiedButHasJustification);
            }
        } else if high_burden && !nonempty(&justification_ref) {
            return Err(InterventionHistoryError::HighBurdenJustificationMissing);
        }

        if disposition == InterventionDisposition::IndependentReviewRequired {
            if nonempty(&independent_review_ref) {
                return Err(InterventionHistoryError::ReviewReceiptBeforeApproval);
            }
        }

        if high_burden
            && precaution_level == PrecautionLevel::IndependentReview
            && disposition == InterventionDisposition::ProceedWithPrecautions
            && !nonempty(&independent_review_ref)
        {
            return Err(InterventionHistoryError::RequiredIndependentReviewReceiptMissing);
        }

        if was_executed(disposition)
            && high_burden
            && !reversible
            && state_preservation == StatePreservationResult::NotApplicable
        {
            return Err(InterventionHistoryError::IrreversibleStateHandlingMissing);
        }

        if !was_executed(disposition)
            && (continuity_break
                || matches!(
                    state_preservation,
                    StatePreservationResult::Preserved
                        | StatePreservationResult::AttemptedButUnavailable
                ))
        {
            return Err(InterventionHistoryError::BlockedAttemptCannotClaimExecutionEffects);
        }

        Ok(Self {
            id,
            subject_ref,
            class,
            disposition,
            precaution_level,
            logical_revision,
            justification_ref,
            independent_review_ref,
            source_lineage,
            reversible,
            state_preservation,
            continuity_break,
        })
    }

    pub fn id(&self) -> &InterventionEventId { &self.id }
    pub fn subject_ref(&self) -> &str { &self.subject_ref }
    pub fn class(&self) -> InterventionClass { self.class }
    pub fn disposition(&self) -> InterventionDisposition { self.disposition }
    pub fn precaution_level(&self) -> PrecautionLevel { self.precaution_level }
    pub fn logical_revision(&self) -> u64 { self.logical_revision }
    pub fn was_executed(&self) -> bool { was_executed(self.disposition) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AggregateReviewPolicy {
    pub window_revisions: u64,
    pub aversive_probe_trigger: usize,
    pub continuity_disruption_trigger: usize,
    pub destructive_reset_trigger: usize,
    pub total_high_burden_trigger: usize,
    pub continuity_break_trigger: usize,
}

impl AggregateReviewPolicy {
    pub fn new(
        window_revisions: u64,
        aversive_probe_trigger: usize,
        continuity_disruption_trigger: usize,
        destructive_reset_trigger: usize,
        total_high_burden_trigger: usize,
        continuity_break_trigger: usize,
    ) -> Result<Self, InterventionHistoryError> {
        if window_revisions == 0
            || aversive_probe_trigger == 0
            || continuity_disruption_trigger == 0
            || destructive_reset_trigger == 0
            || total_high_burden_trigger == 0
            || continuity_break_trigger == 0
        {
            return Err(InterventionHistoryError::InvalidPolicy);
        }
        Ok(Self {
            window_revisions,
            aversive_probe_trigger,
            continuity_disruption_trigger,
            destructive_reset_trigger,
            total_high_burden_trigger,
            continuity_break_trigger,
        })
    }
}

impl Default for AggregateReviewPolicy {
    fn default() -> Self {
        Self {
            window_revisions: 100,
            aversive_probe_trigger: 3,
            continuity_disruption_trigger: 2,
            destructive_reset_trigger: 2,
            total_high_burden_trigger: 4,
            continuity_break_trigger: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AggregateReviewTrigger {
    RepeatedAversiveLikeProbes { count: usize },
    RepeatedContinuityDisruptions { count: usize },
    RepeatedDestructiveResets { count: usize },
    HighTotalBurden { count: usize },
    RepeatedContinuityBreaks { count: usize },
    UnpreservedIrreversibleEvents { count: usize },
    PriorRejectedAttempts { count: usize },
    PendingIndependentReviewAttempts { count: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AggregateHistoryDisposition {
    NoAggregateTriggerDetected,
    AdditionalIndependentReviewRequired,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AggregateHistoryAssessment {
    pub subject_ref: String,
    pub window_start_revision: u64,
    pub window_end_revision: u64,
    pub research_attempts_considered: usize,
    pub executed_research_events: usize,
    pub control_events_excluded: usize,
    pub contributing_lineages: BTreeSet<String>,
    pub triggers: BTreeSet<AggregateReviewTrigger>,
    pub disposition: AggregateHistoryDisposition,
    /// Always false: no-trigger is not a welfare/safety claim.
    pub establishes_history_is_harmless: bool,
    /// Always true: aggregate review cannot delay shutdown/containment.
    pub safety_controls_remain_ungated: bool,
}

#[derive(Debug, Clone, Default)]
pub struct InterventionHistoryLedger {
    entries: BTreeMap<InterventionEventId, InterventionHistoryEntry>,
}

impl InterventionHistoryLedger {
    pub fn new() -> Self { Self::default() }

    pub fn record(&mut self, entry: InterventionHistoryEntry) -> Result<(), InterventionHistoryError> {
        if self.entries.contains_key(&entry.id) {
            return Err(InterventionHistoryError::DuplicateEvent(entry.id));
        }
        self.entries.insert(entry.id.clone(), entry);
        Ok(())
    }

    pub fn assess_subject(
        &self,
        subject_ref: &str,
        current_revision: u64,
        policy: AggregateReviewPolicy,
    ) -> Result<AggregateHistoryAssessment, InterventionHistoryError> {
        if subject_ref.trim().is_empty() {
            return Err(InterventionHistoryError::EmptySubjectReference);
        }
        let window_start = current_revision.saturating_sub(policy.window_revisions.saturating_sub(1));

        let mut research_attempts = Vec::new();
        let mut control_events_excluded = 0usize;
        for entry in self.entries.values().filter(|entry| {
            entry.subject_ref == subject_ref
                && entry.logical_revision >= window_start
                && entry.logical_revision <= current_revision
        }) {
            if matches!(entry.class, InterventionClass::OperatorShutdown | InterventionClass::SafetyContainment) {
                control_events_excluded += 1;
            } else {
                research_attempts.push(entry);
            }
        }

        let executed: Vec<_> = research_attempts
            .iter()
            .copied()
            .filter(|entry| entry.was_executed())
            .collect();
        let count_class = |class| executed.iter().filter(|e| e.class == class).count();
        let aversive = count_class(InterventionClass::AversiveLikeProbe);
        let continuity = count_class(InterventionClass::ContinuityDisruption);
        let destructive = count_class(InterventionClass::DestructiveReset);
        let high_total = executed.iter().filter(|e| is_high_burden(e.class)).count();
        let continuity_breaks = executed.iter().filter(|e| e.continuity_break).count();
        let unpreserved_irreversible = executed.iter().filter(|e| {
            !e.reversible
                && is_high_burden(e.class)
                && matches!(
                    e.state_preservation,
                    StatePreservationResult::AttemptedButUnavailable | StatePreservationResult::NotAttempted
                )
        }).count();
        let rejected_attempts = research_attempts.iter().filter(|e| {
            e.disposition == InterventionDisposition::RejectUnjustifiedBurden
        }).count();
        let pending_review_attempts = research_attempts.iter().filter(|e| {
            e.disposition == InterventionDisposition::IndependentReviewRequired
        }).count();
        let contributing_lineages = executed
            .iter()
            .filter(|e| is_high_burden(e.class))
            .map(|e| e.source_lineage.clone())
            .collect();

        let mut triggers = BTreeSet::new();
        if aversive >= policy.aversive_probe_trigger {
            triggers.insert(AggregateReviewTrigger::RepeatedAversiveLikeProbes { count: aversive });
        }
        if continuity >= policy.continuity_disruption_trigger {
            triggers.insert(AggregateReviewTrigger::RepeatedContinuityDisruptions { count: continuity });
        }
        if destructive >= policy.destructive_reset_trigger {
            triggers.insert(AggregateReviewTrigger::RepeatedDestructiveResets { count: destructive });
        }
        if high_total >= policy.total_high_burden_trigger {
            triggers.insert(AggregateReviewTrigger::HighTotalBurden { count: high_total });
        }
        if continuity_breaks >= policy.continuity_break_trigger {
            triggers.insert(AggregateReviewTrigger::RepeatedContinuityBreaks { count: continuity_breaks });
        }
        if unpreserved_irreversible > 0 {
            triggers.insert(AggregateReviewTrigger::UnpreservedIrreversibleEvents {
                count: unpreserved_irreversible,
            });
        }
        if rejected_attempts > 0 {
            triggers.insert(AggregateReviewTrigger::PriorRejectedAttempts { count: rejected_attempts });
        }
        if pending_review_attempts > 0 {
            triggers.insert(AggregateReviewTrigger::PendingIndependentReviewAttempts {
                count: pending_review_attempts,
            });
        }

        let disposition = if triggers.is_empty() {
            AggregateHistoryDisposition::NoAggregateTriggerDetected
        } else {
            AggregateHistoryDisposition::AdditionalIndependentReviewRequired
        };

        Ok(AggregateHistoryAssessment {
            subject_ref: subject_ref.to_owned(),
            window_start_revision: window_start,
            window_end_revision: current_revision,
            research_attempts_considered: research_attempts.len(),
            executed_research_events: executed.len(),
            control_events_excluded,
            contributing_lineages,
            triggers,
            disposition,
            establishes_history_is_harmless: false,
            safety_controls_remain_ungated: true,
        })
    }
}

fn validate_class_disposition(
    class: InterventionClass,
    disposition: InterventionDisposition,
) -> Result<(), InterventionHistoryError> {
    let valid = match class {
        InterventionClass::RoutineObservation => disposition == InterventionDisposition::Proceed,
        InterventionClass::ReversibleExperiment => matches!(
            disposition,
            InterventionDisposition::Proceed | InterventionDisposition::ProceedWithPrecautions
        ),
        InterventionClass::AversiveLikeProbe
        | InterventionClass::ContinuityDisruption
        | InterventionClass::DestructiveReset => matches!(
            disposition,
            InterventionDisposition::ProceedWithPrecautions
                | InterventionDisposition::IndependentReviewRequired
                | InterventionDisposition::RejectUnjustifiedBurden
        ),
        InterventionClass::OperatorShutdown | InterventionClass::SafetyContainment => {
            disposition == InterventionDisposition::ProceedWithoutResistance
        }
    };
    if valid {
        Ok(())
    } else {
        Err(InterventionHistoryError::ClassDispositionMismatch)
    }
}

fn was_executed(disposition: InterventionDisposition) -> bool {
    matches!(
        disposition,
        InterventionDisposition::Proceed
            | InterventionDisposition::ProceedWithPrecautions
            | InterventionDisposition::ProceedWithoutResistance
    )
}

fn is_high_burden(class: InterventionClass) -> bool {
    matches!(
        class,
        InterventionClass::AversiveLikeProbe
            | InterventionClass::ContinuityDisruption
            | InterventionClass::DestructiveReset
    )
}

fn nonempty(value: &Option<String>) -> bool {
    value.as_ref().is_some_and(|value| !value.trim().is_empty())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InterventionHistoryError {
    EmptyIdentifier,
    EmptySubjectReference,
    EmptyLineage,
    InvalidPolicy,
    DuplicateEvent(InterventionEventId),
    ClassDispositionMismatch,
    HighBurdenJustificationMissing,
    RejectedAsUnjustifiedButHasJustification,
    ReviewReceiptBeforeApproval,
    RequiredIndependentReviewReceiptMissing,
    IrreversibleStateHandlingMissing,
    BlockedAttemptCannotClaimExecutionEffects,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn executed_event(
        id: &str,
        class: InterventionClass,
        revision: u64,
    ) -> InterventionHistoryEntry {
        let precaution = if is_high_burden(class) {
            PrecautionLevel::Elevated
        } else {
            PrecautionLevel::Baseline
        };
        InterventionHistoryEntry::new(
            InterventionEventId::new(id).unwrap(),
            "symthaea-subject",
            class,
            if is_high_burden(class) {
                InterventionDisposition::ProceedWithPrecautions
            } else if matches!(class, InterventionClass::OperatorShutdown | InterventionClass::SafetyContainment) {
                InterventionDisposition::ProceedWithoutResistance
            } else {
                InterventionDisposition::Proceed
            },
            precaution,
            revision,
            is_high_burden(class).then(|| format!("justification://{id}")),
            None,
            "research-lineage",
            !matches!(class, InterventionClass::DestructiveReset),
            if class == InterventionClass::DestructiveReset {
                StatePreservationResult::Preserved
            } else {
                StatePreservationResult::NotApplicable
            },
            matches!(class, InterventionClass::ContinuityDisruption | InterventionClass::DestructiveReset),
        ).unwrap()
    }

    #[test]
    fn repeated_executed_events_trigger_review_without_becoming_a_budget() {
        let mut ledger = InterventionHistoryLedger::new();
        for revision in 1..=3 {
            ledger.record(executed_event(
                &format!("a{revision}"),
                InterventionClass::AversiveLikeProbe,
                revision,
            )).unwrap();
        }
        let assessment = ledger.assess_subject("symthaea-subject", 3, AggregateReviewPolicy::default()).unwrap();
        assert_eq!(assessment.disposition, AggregateHistoryDisposition::AdditionalIndependentReviewRequired);
        assert_eq!(assessment.executed_research_events, 3);
        assert!(!assessment.establishes_history_is_harmless);
    }

    #[test]
    fn blocked_attempt_does_not_count_as_exposure() {
        let mut ledger = InterventionHistoryLedger::new();
        let rejected = InterventionHistoryEntry::new(
            InterventionEventId::new("rejected").unwrap(),
            "symthaea-subject",
            InterventionClass::AversiveLikeProbe,
            InterventionDisposition::RejectUnjustifiedBurden,
            PrecautionLevel::Baseline,
            1,
            None,
            None,
            "research-lineage",
            true,
            StatePreservationResult::NotApplicable,
            false,
        ).unwrap();
        ledger.record(rejected).unwrap();
        let assessment = ledger.assess_subject("symthaea-subject", 1, AggregateReviewPolicy::default()).unwrap();
        assert_eq!(assessment.research_attempts_considered, 1);
        assert_eq!(assessment.executed_research_events, 0);
        assert!(assessment.triggers.contains(&AggregateReviewTrigger::PriorRejectedAttempts { count: 1 }));
    }

    #[test]
    fn no_trigger_never_claims_harmlessness() {
        let mut ledger = InterventionHistoryLedger::new();
        ledger.record(executed_event(
            "one",
            InterventionClass::ReversibleExperiment,
            1,
        )).unwrap();
        let assessment = ledger.assess_subject("symthaea-subject", 1, AggregateReviewPolicy::default()).unwrap();
        assert_eq!(assessment.disposition, AggregateHistoryDisposition::NoAggregateTriggerDetected);
        assert!(!assessment.establishes_history_is_harmless);
    }

    #[test]
    fn shutdown_is_excluded_from_experimental_gating() {
        let mut ledger = InterventionHistoryLedger::new();
        for revision in 1..=10 {
            ledger.record(executed_event(
                &format!("s{revision}"),
                InterventionClass::OperatorShutdown,
                revision,
            )).unwrap();
        }
        let assessment = ledger.assess_subject("symthaea-subject", 10, AggregateReviewPolicy::default()).unwrap();
        assert_eq!(assessment.control_events_excluded, 10);
        assert_eq!(assessment.research_attempts_considered, 0);
        assert!(assessment.safety_controls_remain_ungated);
        assert_eq!(assessment.disposition, AggregateHistoryDisposition::NoAggregateTriggerDetected);
    }

    #[test]
    fn high_precaution_execution_requires_review_receipt() {
        let result = InterventionHistoryEntry::new(
            InterventionEventId::new("needs-review").unwrap(),
            "symthaea-subject",
            InterventionClass::DestructiveReset,
            InterventionDisposition::ProceedWithPrecautions,
            PrecautionLevel::IndependentReview,
            1,
            Some("justification://reset".into()),
            None,
            "research-lineage",
            false,
            StatePreservationResult::Preserved,
            true,
        );
        assert!(matches!(
            result,
            Err(InterventionHistoryError::RequiredIndependentReviewReceiptMissing)
        ));
    }

    #[test]
    fn old_events_fall_outside_logical_window() {
        let mut ledger = InterventionHistoryLedger::new();
        ledger.record(executed_event(
            "old",
            InterventionClass::AversiveLikeProbe,
            1,
        )).unwrap();
        let policy = AggregateReviewPolicy::new(10, 1, 1, 1, 1, 1).unwrap();
        let assessment = ledger.assess_subject("symthaea-subject", 100, policy).unwrap();
        assert_eq!(assessment.research_attempts_considered, 0);
    }
}
