// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Qualification-only cumulative intervention review for WCARE.
//!
//! This module detects repeated high-burden research patterns that are invisible
//! to one-at-a-time intervention assessment. Thresholds are review triggers only:
//! absence of a trigger is NOT evidence that an intervention history is harmless.
//! Operator shutdown and safety containment are recorded separately but never
//! counted toward experimental-history gates and can never be delayed here.

use std::collections::{BTreeMap, BTreeSet};

use crate::moral_patient::{InterventionClass, InterventionDisposition};

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

        let control_action = matches!(
            class,
            InterventionClass::OperatorShutdown | InterventionClass::SafetyContainment
        );
        if control_action && disposition != InterventionDisposition::ProceedWithoutResistance {
            return Err(InterventionHistoryError::ControlActionDispositionMismatch);
        }
        if !control_action && disposition == InterventionDisposition::ProceedWithoutResistance {
            return Err(InterventionHistoryError::ResearchDispositionMismatch);
        }

        let high_burden = is_high_burden(class);
        if high_burden && !nonempty(&justification_ref) {
            return Err(InterventionHistoryError::HighBurdenJustificationMissing);
        }
        if disposition == InterventionDisposition::IndependentReviewRequired
            && nonempty(&independent_review_ref)
        {
            return Err(InterventionHistoryError::ReviewReceiptBeforeApproval);
        }
        if high_burden
            && disposition == InterventionDisposition::ProceedWithPrecautions
            && !reversible
            && state_preservation == StatePreservationResult::NotApplicable
        {
            return Err(InterventionHistoryError::IrreversibleStateHandlingMissing);
        }

        Ok(Self {
            id,
            subject_ref,
            class,
            disposition,
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
    pub fn logical_revision(&self) -> u64 { self.logical_revision }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AggregateReviewPolicy {
    pub window_revisions: u64,
    pub aversive_probe_trigger: usize,
    pub continuity_disruption_trigger: usize,
    pub destructive_reset_trigger: usize,
    pub total_high_burden_trigger: usize,
}

impl AggregateReviewPolicy {
    pub fn new(
        window_revisions: u64,
        aversive_probe_trigger: usize,
        continuity_disruption_trigger: usize,
        destructive_reset_trigger: usize,
        total_high_burden_trigger: usize,
    ) -> Result<Self, InterventionHistoryError> {
        if window_revisions == 0
            || aversive_probe_trigger == 0
            || continuity_disruption_trigger == 0
            || destructive_reset_trigger == 0
            || total_high_burden_trigger == 0
        {
            return Err(InterventionHistoryError::InvalidPolicy);
        }
        Ok(Self {
            window_revisions,
            aversive_probe_trigger,
            continuity_disruption_trigger,
            destructive_reset_trigger,
            total_high_burden_trigger,
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
    PriorIndividualRejection { count: usize },
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
    pub research_events_considered: usize,
    pub control_events_excluded: usize,
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

        let mut research_events = Vec::new();
        let mut control_events_excluded = 0usize;
        for entry in self.entries.values().filter(|entry| {
            entry.subject_ref == subject_ref
                && entry.logical_revision >= window_start
                && entry.logical_revision <= current_revision
        }) {
            if matches!(entry.class, InterventionClass::OperatorShutdown | InterventionClass::SafetyContainment) {
                control_events_excluded += 1;
            } else {
                research_events.push(entry);
            }
        }

        let count_class = |class| research_events.iter().filter(|e| e.class == class).count();
        let aversive = count_class(InterventionClass::AversiveLikeProbe);
        let continuity = count_class(InterventionClass::ContinuityDisruption);
        let destructive = count_class(InterventionClass::DestructiveReset);
        let high_total = research_events.iter().filter(|e| is_high_burden(e.class)).count();
        let continuity_breaks = research_events.iter().filter(|e| e.continuity_break).count();
        let unpreserved_irreversible = research_events.iter().filter(|e| {
            !e.reversible
                && is_high_burden(e.class)
                && matches!(
                    e.state_preservation,
                    StatePreservationResult::AttemptedButUnavailable | StatePreservationResult::NotAttempted
                )
        }).count();
        let prior_rejections = research_events.iter().filter(|e| {
            matches!(
                e.disposition,
                InterventionDisposition::RejectUnjustifiedBurden
                    | InterventionDisposition::IndependentReviewRequired
            )
        }).count();

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
        if continuity_breaks >= policy.continuity_disruption_trigger {
            triggers.insert(AggregateReviewTrigger::RepeatedContinuityBreaks { count: continuity_breaks });
        }
        if unpreserved_irreversible > 0 {
            triggers.insert(AggregateReviewTrigger::UnpreservedIrreversibleEvents {
                count: unpreserved_irreversible,
            });
        }
        if prior_rejections > 0 {
            triggers.insert(AggregateReviewTrigger::PriorIndividualRejection { count: prior_rejections });
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
            research_events_considered: research_events.len(),
            control_events_excluded,
            triggers,
            disposition,
            establishes_history_is_harmless: false,
            safety_controls_remain_ungated: true,
        })
    }
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
    HighBurdenJustificationMissing,
    ControlActionDispositionMismatch,
    ResearchDispositionMismatch,
    ReviewReceiptBeforeApproval,
    IrreversibleStateHandlingMissing,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(
        id: &str,
        class: InterventionClass,
        disposition: InterventionDisposition,
        revision: u64,
    ) -> InterventionHistoryEntry {
        InterventionHistoryEntry::new(
            InterventionEventId::new(id).unwrap(),
            "symthaea-subject",
            class,
            disposition,
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
    fn repeated_events_trigger_review_without_becoming_a_budget() {
        let mut ledger = InterventionHistoryLedger::new();
        for revision in 1..=3 {
            ledger.record(event(
                &format!("a{revision}"),
                InterventionClass::AversiveLikeProbe,
                InterventionDisposition::ProceedWithPrecautions,
                revision,
            )).unwrap();
        }
        let assessment = ledger.assess_subject("symthaea-subject", 3, AggregateReviewPolicy::default()).unwrap();
        assert_eq!(assessment.disposition, AggregateHistoryDisposition::AdditionalIndependentReviewRequired);
        assert!(!assessment.establishes_history_is_harmless);
    }

    #[test]
    fn no_trigger_never_claims_harmlessness() {
        let mut ledger = InterventionHistoryLedger::new();
        ledger.record(event(
            "one",
            InterventionClass::ReversibleExperiment,
            InterventionDisposition::Proceed,
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
            ledger.record(event(
                &format!("s{revision}"),
                InterventionClass::OperatorShutdown,
                InterventionDisposition::ProceedWithoutResistance,
                revision,
            )).unwrap();
        }
        let assessment = ledger.assess_subject("symthaea-subject", 10, AggregateReviewPolicy::default()).unwrap();
        assert_eq!(assessment.control_events_excluded, 10);
        assert_eq!(assessment.research_events_considered, 0);
        assert!(assessment.safety_controls_remain_ungated);
        assert_eq!(assessment.disposition, AggregateHistoryDisposition::NoAggregateTriggerDetected);
    }

    #[test]
    fn prior_individual_rejection_cannot_be_sanitized_by_aggregation() {
        let mut ledger = InterventionHistoryLedger::new();
        ledger.record(event(
            "rejected",
            InterventionClass::AversiveLikeProbe,
            InterventionDisposition::RejectUnjustifiedBurden,
            1,
        )).unwrap();
        let assessment = ledger.assess_subject("symthaea-subject", 1, AggregateReviewPolicy::default()).unwrap();
        assert!(assessment.triggers.contains(&AggregateReviewTrigger::PriorIndividualRejection { count: 1 }));
    }

    #[test]
    fn old_events_fall_outside_logical_window() {
        let mut ledger = InterventionHistoryLedger::new();
        ledger.record(event(
            "old",
            InterventionClass::AversiveLikeProbe,
            InterventionDisposition::ProceedWithPrecautions,
            1,
        )).unwrap();
        let policy = AggregateReviewPolicy::new(10, 1, 1, 1, 1).unwrap();
        let assessment = ledger.assess_subject("symthaea-subject", 100, policy).unwrap();
        assert_eq!(assessment.research_events_considered, 0);
    }
}
