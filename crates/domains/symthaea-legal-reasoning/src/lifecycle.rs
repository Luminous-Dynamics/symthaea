// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic lifecycle assessment for structured norms.
//!
//! This layer reports formal states from supplied event records. It does not
//! decide whether evidence is authentic, whether an act legally counts as
//! performance, or whether a waiver is valid; those are explicit upstream
//! formalization decisions.

use crate::context::{LegalDate, TemporalScope};
use crate::deontic::{DeonticProposition, Modality, StructuredNorm};
use crate::model::{ActionId, EventId, PartyId};
use std::error::Error;
use std::fmt;

/// A recorded performance or prohibited act.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ActionEvent {
    pub id: EventId,
    pub actor: PartyId,
    pub action: ActionId,
    pub beneficiary: Option<PartyId>,
    pub occurred_on: LegalDate,
}

impl ActionEvent {
    pub fn new(id: EventId, actor: PartyId, action: ActionId, occurred_on: LegalDate) -> Self {
        Self {
            id,
            actor,
            action,
            beneficiary: None,
            occurred_on,
        }
    }

    pub fn with_beneficiary(mut self, beneficiary: PartyId) -> Self {
        self.beneficiary = Some(beneficiary);
        self
    }

    pub fn matches(&self, proposition: &DeonticProposition) -> bool {
        self.actor == proposition.bearer
            && self.action == proposition.action
            && self.beneficiary == proposition.beneficiary
    }
}

/// A formalized waiver of one proposition by or for its bearer.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct WaiverEvent {
    pub id: EventId,
    pub proposition: DeonticProposition,
    pub occurred_on: LegalDate,
}

/// Events relevant to lifecycle calculation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NormEvent {
    Action(ActionEvent),
    Waiver(WaiverEvent),
}

impl NormEvent {
    pub fn id(&self) -> &EventId {
        match self {
            NormEvent::Action(event) => &event.id,
            NormEvent::Waiver(event) => &event.id,
        }
    }

    pub fn occurred_on(&self) -> LegalDate {
        match self {
            NormEvent::Action(event) => event.occurred_on,
            NormEvent::Waiver(event) => event.occurred_on,
        }
    }
}

/// A norm with activation, optional deadline, and optional reparation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimedNorm {
    pub norm: StructuredNorm,
    pub validity: TemporalScope,
    pub due_on: Option<LegalDate>,
    pub reparation: Option<StructuredNorm>,
}

/// Invalid temporal construction of a lifecycle-bearing norm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LifecycleError {
    DeadlineRequiresObligation,
    DueBeforeEffective { due_on: LegalDate, effective_from: LegalDate },
    DueAfterExpiry { due_on: LegalDate, effective_until: LegalDate },
}

impl fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LifecycleError::DeadlineRequiresObligation => {
                f.write_str("only an obligation may carry a performance deadline")
            }
            LifecycleError::DueBeforeEffective {
                due_on,
                effective_from,
            } => write!(
                f,
                "norm deadline {due_on} precedes effective date {effective_from}"
            ),
            LifecycleError::DueAfterExpiry {
                due_on,
                effective_until,
            } => write!(
                f,
                "norm deadline {due_on} follows expiry date {effective_until}"
            ),
        }
    }
}

impl Error for LifecycleError {}

impl TimedNorm {
    pub fn new(norm: StructuredNorm, validity: TemporalScope) -> Self {
        Self {
            norm,
            validity,
            due_on: None,
            reparation: None,
        }
    }

    pub fn with_deadline(mut self, due_on: LegalDate) -> Result<Self, LifecycleError> {
        if self.norm.modality != Modality::Obligatory {
            return Err(LifecycleError::DeadlineRequiresObligation);
        }
        if let Some(effective_from) = self.validity.effective_from {
            if due_on < effective_from {
                return Err(LifecycleError::DueBeforeEffective {
                    due_on,
                    effective_from,
                });
            }
        }
        if let Some(effective_until) = self.validity.effective_until {
            if due_on > effective_until {
                return Err(LifecycleError::DueAfterExpiry {
                    due_on,
                    effective_until,
                });
            }
        }
        self.due_on = Some(due_on);
        Ok(self)
    }

    pub fn with_reparation(mut self, reparation: StructuredNorm) -> Self {
        self.reparation = Some(reparation);
        self
    }
}

/// Formal lifecycle state at a query date.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NormState {
    NotYetEffective,
    Active,
    Fulfilled,
    FulfilledLate,
    Exercised,
    Violated,
    TemporallyAmbiguous,
    Waived,
    Expired,
}

/// Replayable evidence behind one lifecycle answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LifecycleAssessment {
    pub state: NormState,
    pub decisive_events: Vec<EventId>,
    pub activated_reparation: Option<StructuredNorm>,
}

/// Assess a norm using only events occurring on or before `as_of`.
pub fn assess_lifecycle(
    norm: &TimedNorm,
    events: &[NormEvent],
    as_of: LegalDate,
) -> LifecycleAssessment {
    if norm
        .validity
        .effective_from
        .is_some_and(|effective_from| as_of < effective_from)
    {
        return assessment(NormState::NotYetEffective, Vec::new(), None);
    }

    let proposition = &norm.norm.proposition;
    let mut relevant: Vec<&NormEvent> = events
        .iter()
        .filter(|event| event.occurred_on() <= as_of)
        .filter(|event| norm.validity.contains(event.occurred_on()))
        .filter(|event| match event {
            NormEvent::Action(action) => action.matches(proposition),
            NormEvent::Waiver(waiver) => &waiver.proposition == proposition,
        })
        .collect();
    relevant.sort_unstable_by(|left, right| {
        left.occurred_on()
            .cmp(&right.occurred_on())
            .then_with(|| left.id().cmp(right.id()))
    });

    let first_action = relevant.iter().find_map(|event| match event {
        NormEvent::Action(action) => Some(action),
        NormEvent::Waiver(_) => None,
    });
    let first_waiver = relevant.iter().find_map(|event| match event {
        NormEvent::Waiver(waiver) => Some(waiver),
        NormEvent::Action(_) => None,
    });
    let expired = norm
        .validity
        .effective_until
        .is_some_and(|effective_until| as_of > effective_until);

    match norm.norm.modality {
        Modality::Obligatory => assess_obligation(norm, first_action, first_waiver, as_of, expired),
        Modality::Forbidden => assess_action_or_waiver(
            first_action,
            first_waiver,
            NormState::Violated,
            norm.reparation.clone(),
            expired,
        ),
        Modality::Permitted => assess_action_or_waiver(
            first_action,
            first_waiver,
            NormState::Exercised,
            None,
            expired,
        ),
    }
}

fn assess_obligation(
    norm: &TimedNorm,
    action: Option<&ActionEvent>,
    waiver: Option<&WaiverEvent>,
    as_of: LegalDate,
    expired: bool,
) -> LifecycleAssessment {
    if let Some(due_on) = norm.due_on {
        let deadline_passed = as_of > due_on;
        let waiver_before_or_on_deadline = waiver.is_some_and(|event| event.occurred_on <= due_on);
        let action_before_or_on_deadline = action.is_some_and(|event| event.occurred_on <= due_on);

        if waiver_before_or_on_deadline || action_before_or_on_deadline {
            return resolve_action_waiver(
                action.filter(|event| event.occurred_on <= due_on),
                waiver.filter(|event| event.occurred_on <= due_on),
                NormState::Fulfilled,
                None,
            );
        }

        if deadline_passed {
            if let Some(action) = action {
                return assessment(
                    NormState::FulfilledLate,
                    vec![action.id.clone()],
                    norm.reparation.clone(),
                );
            }
            return assessment(
                NormState::Violated,
                Vec::new(),
                norm.reparation.clone(),
            );
        }
    }

    if action.is_some() || waiver.is_some() {
        return resolve_action_waiver(action, waiver, NormState::Fulfilled, None);
    }
    if expired {
        assessment(NormState::Expired, Vec::new(), None)
    } else {
        assessment(NormState::Active, Vec::new(), None)
    }
}

fn assess_action_or_waiver(
    action: Option<&ActionEvent>,
    waiver: Option<&WaiverEvent>,
    action_state: NormState,
    action_reparation: Option<StructuredNorm>,
    expired: bool,
) -> LifecycleAssessment {
    if action.is_some() || waiver.is_some() {
        resolve_action_waiver(action, waiver, action_state, action_reparation)
    } else if expired {
        assessment(NormState::Expired, Vec::new(), None)
    } else {
        assessment(NormState::Active, Vec::new(), None)
    }
}

fn resolve_action_waiver(
    action: Option<&ActionEvent>,
    waiver: Option<&WaiverEvent>,
    action_state: NormState,
    action_reparation: Option<StructuredNorm>,
) -> LifecycleAssessment {
    match (action, waiver) {
        (Some(action), Some(waiver)) if action.occurred_on < waiver.occurred_on => assessment(
            action_state,
            vec![action.id.clone()],
            action_reparation,
        ),
        (Some(action), Some(waiver)) if waiver.occurred_on < action.occurred_on => {
            assessment(NormState::Waived, vec![waiver.id.clone()], None)
        }
        (Some(action), Some(waiver)) => assessment(
            NormState::TemporallyAmbiguous,
            vec![action.id.clone(), waiver.id.clone()],
            None,
        ),
        (Some(action), None) => assessment(
            action_state,
            vec![action.id.clone()],
            action_reparation,
        ),
        (None, Some(waiver)) => {
            assessment(NormState::Waived, vec![waiver.id.clone()], None)
        }
        (None, None) => assessment(NormState::Active, Vec::new(), None),
    }
}

fn assessment(
    state: NormState,
    decisive_events: Vec<EventId>,
    activated_reparation: Option<StructuredNorm>,
) -> LifecycleAssessment {
    LifecycleAssessment {
        state,
        decisive_events,
        activated_reparation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proposition(action: &str) -> DeonticProposition {
        DeonticProposition::new(
            PartyId::new("debtor").unwrap(),
            ActionId::new(action).unwrap(),
        )
        .with_beneficiary(PartyId::new("creditor").unwrap())
    }

    fn action_event(id: &str, action: &str, date: LegalDate) -> NormEvent {
        NormEvent::Action(
            ActionEvent::new(
                EventId::new(id).unwrap(),
                PartyId::new("debtor").unwrap(),
                ActionId::new(action).unwrap(),
                date,
            )
            .with_beneficiary(PartyId::new("creditor").unwrap()),
        )
    }

    #[test]
    fn deadlines_are_reserved_for_obligations() {
        let permission = TimedNorm::new(
            StructuredNorm::new(Modality::Permitted, proposition("enter")),
            TemporalScope::unbounded(),
        );
        assert_eq!(
            permission.with_deadline(LegalDate::new(2026, 7, 31).unwrap()),
            Err(LifecycleError::DeadlineRequiresObligation)
        );
    }

    #[test]
    fn obligation_moves_from_active_to_fulfilled() {
        let start = LegalDate::new(2026, 7, 1).unwrap();
        let due = LegalDate::new(2026, 7, 31).unwrap();
        let paid = LegalDate::new(2026, 7, 20).unwrap();
        let norm = TimedNorm::new(
            StructuredNorm::new(Modality::Obligatory, proposition("pay")),
            TemporalScope::new(Some(start), Some(due)).unwrap(),
        )
        .with_deadline(due)
        .unwrap();

        assert_eq!(
            assess_lifecycle(&norm, &[], LegalDate::new(2026, 7, 10).unwrap()).state,
            NormState::Active
        );
        assert_eq!(
            assess_lifecycle(&norm, &[action_event("payment", "pay", paid)], due).state,
            NormState::Fulfilled
        );
    }

    #[test]
    fn late_performance_preserves_the_prior_violation_signal() {
        let due = LegalDate::new(2026, 7, 10).unwrap();
        let paid = LegalDate::new(2026, 7, 12).unwrap();
        let reparation = StructuredNorm::new(Modality::Obligatory, proposition("pay-interest"));
        let norm = TimedNorm::new(
            StructuredNorm::new(Modality::Obligatory, proposition("pay")),
            TemporalScope::unbounded(),
        )
        .with_deadline(due)
        .unwrap()
        .with_reparation(reparation.clone());
        let result = assess_lifecycle(
            &norm,
            &[action_event("late-payment", "pay", paid)],
            paid,
        );

        assert_eq!(result.state, NormState::FulfilledLate);
        assert_eq!(result.activated_reparation, Some(reparation));
    }

    #[test]
    fn events_outside_validity_do_not_change_the_norm() {
        let start = LegalDate::new(2026, 7, 1).unwrap();
        let end = LegalDate::new(2026, 7, 10).unwrap();
        let after = LegalDate::new(2026, 7, 11).unwrap();
        let norm = TimedNorm::new(
            StructuredNorm::new(Modality::Forbidden, proposition("disclose")),
            TemporalScope::new(Some(start), Some(end)).unwrap(),
        );
        let result = assess_lifecycle(
            &norm,
            &[action_event("late-disclosure", "disclose", after)],
            after,
        );

        assert_eq!(result.state, NormState::Expired);
        assert!(result.decisive_events.is_empty());
    }

    #[test]
    fn same_day_action_and_waiver_are_not_ordered_by_event_id() {
        let date = LegalDate::new(2026, 7, 21).unwrap();
        let proposition = proposition("disclose");
        let norm = TimedNorm::new(
            StructuredNorm::new(Modality::Forbidden, proposition.clone()),
            TemporalScope::unbounded(),
        );
        let events = vec![
            action_event("z-action", "disclose", date),
            NormEvent::Waiver(WaiverEvent {
                id: EventId::new("a-waiver").unwrap(),
                proposition,
                occurred_on: date,
            }),
        ];

        assert_eq!(
            assess_lifecycle(&norm, &events, date).state,
            NormState::TemporallyAmbiguous
        );
    }

    #[test]
    fn missed_deadline_activates_reparation() {
        let due = LegalDate::new(2026, 7, 10).unwrap();
        let reparation = StructuredNorm::new(Modality::Obligatory, proposition("pay-interest"));
        let norm = TimedNorm::new(
            StructuredNorm::new(Modality::Obligatory, proposition("pay")),
            TemporalScope::unbounded(),
        )
        .with_deadline(due)
        .unwrap()
        .with_reparation(reparation.clone());
        let result = assess_lifecycle(
            &norm,
            &[],
            LegalDate::new(2026, 7, 11).unwrap(),
        );

        assert_eq!(result.state, NormState::Violated);
        assert_eq!(result.activated_reparation, Some(reparation));
    }

    #[test]
    fn forbidden_action_is_a_violation_not_fulfillment() {
        let date = LegalDate::new(2026, 7, 21).unwrap();
        let norm = TimedNorm::new(
            StructuredNorm::new(Modality::Forbidden, proposition("disclose")),
            TemporalScope::unbounded(),
        );
        let result = assess_lifecycle(
            &norm,
            &[action_event("disclosure", "disclose", date)],
            date,
        );

        assert_eq!(result.state, NormState::Violated);
        assert_eq!(result.decisive_events, vec![EventId::new("disclosure").unwrap()]);
    }

    #[test]
    fn waiver_precedes_other_lifecycle_outcomes() {
        let date = LegalDate::new(2026, 7, 21).unwrap();
        let proposition = proposition("pay");
        let norm = TimedNorm::new(
            StructuredNorm::new(Modality::Obligatory, proposition.clone()),
            TemporalScope::unbounded(),
        );
        let waiver = NormEvent::Waiver(WaiverEvent {
            id: EventId::new("waiver").unwrap(),
            proposition,
            occurred_on: date,
        });

        assert_eq!(
            assess_lifecycle(&norm, &[waiver], date).state,
            NormState::Waived
        );
    }
}
