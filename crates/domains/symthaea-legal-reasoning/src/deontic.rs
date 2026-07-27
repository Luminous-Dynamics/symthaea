// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deontic logic: obligation, permission, prohibition, consistency, and
//! conflict-aware assessment of a named act.

use crate::model::{ActionId, PartyId};
use std::collections::BTreeSet;

/// A deontic status assigned to a named act.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Norm {
    /// O(a): the act is obligatory (must be done).
    Obligatory(String),
    /// P(a): the act is explicitly permitted (may be done).
    Permitted(String),
    /// F(a): the act is forbidden (must not be done); equivalently ¬P(a).
    Forbidden(String),
}

/// The modality of a structured legal norm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Modality {
    Obligatory,
    Permitted,
    Forbidden,
}

/// The party-bound content to which a deontic modality applies.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DeonticProposition {
    pub bearer: PartyId,
    pub action: ActionId,
    pub beneficiary: Option<PartyId>,
}

impl DeonticProposition {
    pub fn new(bearer: PartyId, action: ActionId) -> Self {
        Self {
            bearer,
            action,
            beneficiary: None,
        }
    }

    pub fn with_beneficiary(mut self, beneficiary: PartyId) -> Self {
        self.beneficiary = Some(beneficiary);
        self
    }
}

/// A typed deontic norm suitable for legal-kernel composition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StructuredNorm {
    pub modality: Modality,
    pub proposition: DeonticProposition,
}

impl StructuredNorm {
    pub fn new(modality: Modality, proposition: DeonticProposition) -> Self {
        Self {
            modality,
            proposition,
        }
    }
}

impl Norm {
    pub fn act(&self) -> &str {
        match self {
            Norm::Obligatory(a) | Norm::Permitted(a) | Norm::Forbidden(a) => a,
        }
    }
}

/// The complete deontic evidence found for one act.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NormAssessment {
    pub obligatory: bool,
    pub explicitly_permitted: bool,
    pub forbidden: bool,
}

impl NormAssessment {
    /// Whether positive permission support exists through P(a) or O(a) → P(a).
    pub fn has_permission_support(self) -> bool {
        self.explicitly_permitted || self.obligatory
    }

    /// Whether the evidence contains an internal deontic contradiction.
    pub fn is_conflicted(self) -> bool {
        self.forbidden && self.has_permission_support()
    }

    /// Reduce the evidence to a query-facing permission status.
    pub fn permission_status(self) -> PermissionStatus {
        if self.is_conflicted() {
            PermissionStatus::Conflicted
        } else if self.forbidden {
            PermissionStatus::Forbidden
        } else if self.explicitly_permitted {
            PermissionStatus::ExplicitlyPermitted
        } else if self.obligatory {
            PermissionStatus::ImpliedByObligation
        } else {
            PermissionStatus::Undetermined
        }
    }
}

/// A conflict-aware answer to a permission query.
///
/// In particular, [`Undetermined`](PermissionStatus::Undetermined) is distinct
/// from [`Forbidden`](PermissionStatus::Forbidden), and contradictory evidence
/// is surfaced rather than silently resolved by control-flow ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionStatus {
    ExplicitlyPermitted,
    ImpliedByObligation,
    Forbidden,
    Conflicted,
    Undetermined,
}

impl PermissionStatus {
    /// True only when permission is supported and not contradicted.
    pub fn is_permitted(self) -> bool {
        matches!(
            self,
            PermissionStatus::ExplicitlyPermitted | PermissionStatus::ImpliedByObligation
        )
    }
}

/// Collect all deontic evidence concerning `act`.
pub fn assess_act(norms: &[Norm], act: &str) -> NormAssessment {
    let mut assessment = NormAssessment::default();
    for norm in norms {
        match norm {
            Norm::Obligatory(candidate) if candidate == act => assessment.obligatory = true,
            Norm::Permitted(candidate) if candidate == act => {
                assessment.explicitly_permitted = true;
            }
            Norm::Forbidden(candidate) if candidate == act => assessment.forbidden = true,
            _ => {}
        }
    }
    assessment
}

/// Acts that are simultaneously forbidden and obligatory or permitted.
///
/// The returned acts are unique and lexicographically ordered, independent of
/// the order of the input norm slice.
pub fn conflicting_acts(norms: &[Norm]) -> Vec<String> {
    let mut candidates = BTreeSet::new();
    for norm in norms {
        candidates.insert(norm.act());
    }

    candidates
        .into_iter()
        .filter(|act| assess_act(norms, act).is_conflicted())
        .map(str::to_string)
        .collect()
}

/// Whether the norm set is deontically consistent.
pub fn is_consistent(norms: &[Norm]) -> bool {
    conflicting_acts(norms).is_empty()
}

/// Return a conflict-aware permission status for `act`.
pub fn permission_status(norms: &[Norm], act: &str) -> PermissionStatus {
    assess_act(norms, act).permission_status()
}

/// Compatibility Boolean permission query.
///
/// This returns `false` for forbidden, conflicted, and undetermined acts. New
/// code should prefer [`permission_status`] so those cases remain distinct.
pub fn is_permitted(norms: &[Norm], act: &str) -> bool {
    permission_status(norms, act).is_permitted()
}

/// Collect all modal evidence for one typed proposition.
pub fn assess_proposition(
    norms: &[StructuredNorm],
    proposition: &DeonticProposition,
) -> NormAssessment {
    let mut assessment = NormAssessment::default();
    for norm in norms.iter().filter(|norm| &norm.proposition == proposition) {
        match norm.modality {
            Modality::Obligatory => assessment.obligatory = true,
            Modality::Permitted => assessment.explicitly_permitted = true,
            Modality::Forbidden => assessment.forbidden = true,
        }
    }
    assessment
}

/// Return a conflict-aware permission status for a typed proposition.
pub fn proposition_permission_status(
    norms: &[StructuredNorm],
    proposition: &DeonticProposition,
) -> PermissionStatus {
    assess_proposition(norms, proposition).permission_status()
}

/// Return every typed proposition carrying contradictory modal evidence.
pub fn conflicting_propositions(norms: &[StructuredNorm]) -> Vec<DeonticProposition> {
    let propositions: BTreeSet<&DeonticProposition> =
        norms.iter().map(|norm| &norm.proposition).collect();
    propositions
        .into_iter()
        .filter(|proposition| assess_proposition(norms, proposition).is_conflicted())
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn consistent_norm_set() {
        let norms = vec![
            Norm::Obligatory("pay_tax".into()),
            Norm::Permitted("park_here".into()),
            Norm::Forbidden("steal".into()),
        ];
        assert!(is_consistent(&norms));
    }

    #[test]
    fn obligation_conflicts_with_prohibition() {
        let norms = vec![
            Norm::Obligatory("testify".into()),
            Norm::Forbidden("testify".into()),
        ];
        assert!(!is_consistent(&norms));
        assert_eq!(conflicting_acts(&norms), vec!["testify"]);
        assert_eq!(
            permission_status(&norms, "testify"),
            PermissionStatus::Conflicted
        );
    }

    #[test]
    fn permission_conflicts_with_prohibition() {
        let norms = vec![
            Norm::Permitted("enter".into()),
            Norm::Forbidden("enter".into()),
        ];
        assert!(!is_consistent(&norms));
        assert_eq!(
            permission_status(&norms, "enter"),
            PermissionStatus::Conflicted
        );
    }

    #[test]
    fn permission_sources_remain_distinct() {
        let norms = vec![
            Norm::Obligatory("vote".into()),
            Norm::Permitted("assemble".into()),
            Norm::Forbidden("bribe".into()),
        ];
        assert_eq!(
            permission_status(&norms, "vote"),
            PermissionStatus::ImpliedByObligation
        );
        assert_eq!(
            permission_status(&norms, "assemble"),
            PermissionStatus::ExplicitlyPermitted
        );
        assert_eq!(
            permission_status(&norms, "bribe"),
            PermissionStatus::Forbidden
        );
        assert_eq!(
            permission_status(&norms, "unlisted"),
            PermissionStatus::Undetermined
        );
    }

    #[test]
    fn compatibility_boolean_is_fail_closed() {
        let norms = vec![
            Norm::Obligatory("vote".into()),
            Norm::Forbidden("testify".into()),
            Norm::Permitted("testify".into()),
        ];
        assert!(is_permitted(&norms, "vote"));
        assert!(!is_permitted(&norms, "testify"));
        assert!(!is_permitted(&norms, "unlisted"));
    }

    #[test]
    fn conflict_listing_is_unique_and_order_invariant() {
        let norms = vec![
            Norm::Forbidden("zoning".into()),
            Norm::Permitted("appeal".into()),
            Norm::Permitted("zoning".into()),
            Norm::Forbidden("appeal".into()),
            Norm::Forbidden("zoning".into()),
        ];
        assert_eq!(conflicting_acts(&norms), vec!["appeal", "zoning"]);
    }

    #[test]
    fn structured_norms_distinguish_bearers_and_beneficiaries() {
        let employee = PartyId::new("employee").unwrap();
        let employer = PartyId::new("employer").unwrap();
        let pay_wage = ActionId::new("pay_wage").unwrap();
        let proposition =
            DeonticProposition::new(employer.clone(), pay_wage).with_beneficiary(employee.clone());
        let reversed = DeonticProposition::new(employee, ActionId::new("pay_wage").unwrap())
            .with_beneficiary(employer);
        let norms = vec![StructuredNorm::new(
            Modality::Obligatory,
            proposition.clone(),
        )];

        assert_eq!(
            proposition_permission_status(&norms, &proposition),
            PermissionStatus::ImpliedByObligation
        );
        assert_eq!(
            proposition_permission_status(&norms, &reversed),
            PermissionStatus::Undetermined
        );
    }

    #[test]
    fn structured_conflicts_are_reported_without_cross_party_collapse() {
        let court = PartyId::new("court").unwrap();
        let witness = PartyId::new("witness").unwrap();
        let testify = ActionId::new("testify").unwrap();
        let court_testifies = DeonticProposition::new(court, testify.clone());
        let witness_testifies = DeonticProposition::new(witness, testify);
        let norms = vec![
            StructuredNorm::new(Modality::Obligatory, witness_testifies.clone()),
            StructuredNorm::new(Modality::Forbidden, witness_testifies.clone()),
            StructuredNorm::new(Modality::Forbidden, court_testifies),
        ];

        assert_eq!(conflicting_propositions(&norms), vec![witness_testifies]);
    }
}
