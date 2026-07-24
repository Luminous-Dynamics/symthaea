// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Jurisdiction, authority, provenance, and legal-time context.

use crate::model::{AuthorityId, JurisdictionId, SourceRef};
use std::error::Error;
use std::fmt;

/// A dependency-free civil date used for legal validity intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LegalDate {
    pub year: i32,
    pub month: u8,
    pub day: u8,
}

/// Validation failure for a date or temporal interval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemporalError {
    InvalidMonth { month: u8 },
    InvalidDay { year: i32, month: u8, day: u8 },
    InvertedInterval { from: LegalDate, until: LegalDate },
}

impl fmt::Display for TemporalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemporalError::InvalidMonth { month } => {
                write!(f, "invalid civil month: {month}")
            }
            TemporalError::InvalidDay { year, month, day } => {
                write!(f, "invalid civil date: {year:04}-{month:02}-{day:02}")
            }
            TemporalError::InvertedInterval { from, until } => write!(
                f,
                "legal interval begins after it ends: {from} > {until}"
            ),
        }
    }
}

impl Error for TemporalError {}

impl fmt::Display for LegalDate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}-{:02}", self.year, self.month, self.day)
    }
}

impl LegalDate {
    pub fn new(year: i32, month: u8, day: u8) -> Result<Self, TemporalError> {
        if !(1..=12).contains(&month) {
            return Err(TemporalError::InvalidMonth { month });
        }
        let max_day = days_in_month(year, month);
        if day == 0 || day > max_day {
            return Err(TemporalError::InvalidDay { year, month, day });
        }
        Ok(Self { year, month, day })
    }
}

fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn is_leap_year(year: i32) -> bool {
    year.rem_euclid(4) == 0
        && (year.rem_euclid(100) != 0 || year.rem_euclid(400) == 0)
}

/// Inclusive legal-validity interval. An omitted bound is open-ended.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TemporalScope {
    pub effective_from: Option<LegalDate>,
    pub effective_until: Option<LegalDate>,
}

impl TemporalScope {
    pub fn new(
        effective_from: Option<LegalDate>,
        effective_until: Option<LegalDate>,
    ) -> Result<Self, TemporalError> {
        if let (Some(from), Some(until)) = (effective_from, effective_until) {
            if from > until {
                return Err(TemporalError::InvertedInterval { from, until });
            }
        }
        Ok(Self {
            effective_from,
            effective_until,
        })
    }

    pub fn unbounded() -> Self {
        Self::default()
    }

    pub fn contains(&self, date: LegalDate) -> bool {
        self.effective_from.map_or(true, |from| date >= from)
            && self.effective_until.map_or(true, |until| date <= until)
    }
}

/// Context required to determine where, under whose authority, from which
/// provision, and during which interval a formal legal object applies.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LegalContext {
    pub jurisdiction: JurisdictionId,
    pub authority: AuthorityId,
    pub source: SourceRef,
    pub validity: TemporalScope,
}

impl LegalContext {
    pub fn applies_on(&self, date: LegalDate) -> bool {
        self.validity.contains(date)
    }
}

/// Attach legal context to any formal object without coupling its core algebra
/// to a specific interchange format.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Contextual<T> {
    pub value: T,
    pub context: LegalContext,
}

impl<T> Contextual<T> {
    pub fn new(value: T, context: LegalContext) -> Self {
        Self { value, context }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Contextual<U> {
        Contextual {
            value: f(self.value),
            context: self.context,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_date_validation_handles_leap_years() {
        assert!(LegalDate::new(2024, 2, 29).is_ok());
        assert!(LegalDate::new(2100, 2, 29).is_err());
        assert!(LegalDate::new(2000, 2, 29).is_ok());
        assert!(LegalDate::new(2026, 13, 1).is_err());
    }

    #[test]
    fn temporal_scope_is_inclusive_and_rejects_inversion() {
        let from = LegalDate::new(2026, 1, 1).unwrap();
        let until = LegalDate::new(2026, 12, 31).unwrap();
        let scope = TemporalScope::new(Some(from), Some(until)).unwrap();

        assert!(scope.contains(from));
        assert!(scope.contains(until));
        assert!(!scope.contains(LegalDate::new(2025, 12, 31).unwrap()));
        assert!(TemporalScope::new(Some(until), Some(from)).is_err());
    }

    #[test]
    fn contextual_map_preserves_provenance() {
        let context = LegalContext {
            jurisdiction: JurisdictionId::new("ZA").unwrap(),
            authority: AuthorityId::new("parliament").unwrap(),
            source: SourceRef::new(
                crate::model::DocumentId::new("act-1").unwrap(),
                crate::model::ProvisionId::new("section-2").unwrap(),
            ),
            validity: TemporalScope::unbounded(),
        };
        let contextual = Contextual::new(2_u8, context.clone()).map(u16::from);
        assert_eq!(contextual.value, 2_u16);
        assert_eq!(contextual.context, context);
    }
}

/// Distinct legal-time dimensions for a formal object.
///
/// `effective` answers when the rule is in force. `applicability` answers which
/// underlying acts, transactions, or events it governs. The two intervals are
/// intentionally separate so retroactivity is represented rather than hidden.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TemporalDimensions {
    pub enacted_on: Option<LegalDate>,
    pub published_on: Option<LegalDate>,
    pub effective: TemporalScope,
    pub applicability: TemporalScope,
}

impl TemporalDimensions {
    pub fn new(effective: TemporalScope) -> Self {
        Self {
            enacted_on: None,
            published_on: None,
            applicability: effective.clone(),
            effective,
        }
    }

    pub fn enacted_on(mut self, date: LegalDate) -> Self {
        self.enacted_on = Some(date);
        self
    }

    pub fn published_on(mut self, date: LegalDate) -> Self {
        self.published_on = Some(date);
        self
    }

    pub fn with_applicability(mut self, applicability: TemporalScope) -> Self {
        self.applicability = applicability;
        self
    }

    pub fn is_effective_on(&self, decision_date: LegalDate) -> bool {
        self.effective.contains(decision_date)
    }

    pub fn applies_to(&self, event_date: LegalDate) -> bool {
        self.applicability.contains(event_date)
    }

    pub fn governs(&self, decision_date: LegalDate, event_date: LegalDate) -> bool {
        self.is_effective_on(decision_date) && self.applies_to(event_date)
    }

    /// Whether applicability begins before legal effect, an explicit signal of
    /// possible retroactive operation requiring review by the caller.
    pub fn is_potentially_retroactive(&self) -> bool {
        match (
            self.applicability.effective_from,
            self.effective.effective_from,
        ) {
            (Some(applies_from), Some(effective_from)) => applies_from < effective_from,
            (None, Some(_)) => true,
            _ => false,
        }
    }
}

/// One revision of a formal object with explicit legal-time dimensions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TemporalRevision<T> {
    pub revision: crate::model::RevisionId,
    pub value: T,
    pub time: TemporalDimensions,
}

impl<T> TemporalRevision<T> {
    pub fn new(
        revision: crate::model::RevisionId,
        value: T,
        time: TemporalDimensions,
    ) -> Self {
        Self {
            revision,
            value,
            time,
        }
    }
}

/// An ambiguous temporal selection where multiple revisions are effective.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemporalOverlap {
    pub revisions: Vec<crate::model::RevisionId>,
}

impl fmt::Display for TemporalOverlap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let revisions = self
            .revisions
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "multiple temporal revisions govern the query: {revisions}")
    }
}

impl Error for TemporalOverlap {}

/// Return every revision governing a decision date and underlying event date,
/// in canonical revision-id order.
pub fn governing_revisions<T>(
    revisions: &[TemporalRevision<T>],
    decision_date: LegalDate,
    event_date: LegalDate,
) -> Vec<&TemporalRevision<T>> {
    let mut selected: Vec<&TemporalRevision<T>> = revisions
        .iter()
        .filter(|revision| revision.time.governs(decision_date, event_date))
        .collect();
    selected.sort_unstable_by(|left, right| left.revision.cmp(&right.revision));
    selected
}

/// Select exactly one governing revision, failing explicitly on overlap.
pub fn unique_governing_revision<T>(
    revisions: &[TemporalRevision<T>],
    decision_date: LegalDate,
    event_date: LegalDate,
) -> Result<Option<&TemporalRevision<T>>, TemporalOverlap> {
    let selected = governing_revisions(revisions, decision_date, event_date);
    match selected.as_slice() {
        [] => Ok(None),
        [revision] => Ok(Some(*revision)),
        _ => Err(TemporalOverlap {
            revisions: selected
                .into_iter()
                .map(|revision| revision.revision.clone())
                .collect(),
        }),
    }
}

#[cfg(test)]
mod legal_time_tests {
    use super::*;

    #[test]
    fn legal_time_separates_effect_and_applicability() {
        let effective = TemporalScope::new(
            Some(LegalDate::new(2026, 7, 1).unwrap()),
            None,
        )
        .unwrap();
        let applicability = TemporalScope::new(
            Some(LegalDate::new(2026, 1, 1).unwrap()),
            None,
        )
        .unwrap();
        let time = TemporalDimensions::new(effective).with_applicability(applicability);

        assert!(time.is_potentially_retroactive());
        assert!(time.governs(
            LegalDate::new(2026, 7, 21).unwrap(),
            LegalDate::new(2026, 3, 1).unwrap()
        ));
        assert!(!time.governs(
            LegalDate::new(2026, 6, 30).unwrap(),
            LegalDate::new(2026, 3, 1).unwrap()
        ));
    }

    #[test]
    fn overlapping_revisions_fail_explicitly() {
        let date = LegalDate::new(2026, 7, 21).unwrap();
        let time = TemporalDimensions::new(TemporalScope::unbounded());
        let revisions = vec![
            TemporalRevision::new(
                crate::model::RevisionId::new("v1").unwrap(),
                "old",
                time.clone(),
            ),
            TemporalRevision::new(
                crate::model::RevisionId::new("v2").unwrap(),
                "new",
                time,
            ),
        ];

        let overlap = unique_governing_revision(&revisions, date, date).unwrap_err();
        assert_eq!(overlap.revisions.len(), 2);
    }

}
