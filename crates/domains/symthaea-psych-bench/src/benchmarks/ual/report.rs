// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared reporting types for the UAL (Unlimited Associative Learning) Phase-1
//! probe packet (P1/P2/P4a). See `symthaea/docs/SYMTHAEA_UAL_EXTENSION_DESIGN_2026-07-27.md`
//! and `symthaea/docs/SYMTHAEA_UAL_PHASE1_PROTOCOLS_2026-07-27.md` for the full
//! rationale — this module implements only the "mandatory three-field reporting
//! format" and the schedule-status qualifier those docs require.
//!
//! **This is deliberately NOT a 15th Butlin indicator.** It does not import or
//! extend `crate::benchmarks::butlin::report::IndicatorEvidence` — UAL is a
//! functional-capacity theory outside the frozen Butlin-14 denominator, and
//! must stay reportable as its own thing (design doc, "Why UAL sits outside
//! the frozen Butlin denominator").

use std::fmt;

/// A presentation-order schedule for a UAL probe's multi-schedule replication
/// requirement. Each probe interprets these two variants concretely in its own
/// module doc comment (e.g. P1: `Blocked` = abrupt reversal at a fixed trial,
/// `Interleaved` = probabilistic-hazard reversal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UalSchedule {
    Blocked,
    Interleaved,
}

/// Whether a signal was observed at all, independent of whether it counts as
/// UAL-functional support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Presence {
    Observed,
    NotObserved,
}

/// The probe-level functional-outcome verdict. Per the design doc's hard
/// rule, this may only be `Demonstrated` when `behavioral_expression` is
/// `Presence::Observed` — enforced structurally in `UalProbeReport::new`,
/// not left to the caller's discretion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionalOutcome {
    Demonstrated,
    NotDemonstrated,
}

/// Schedule-robustness qualifier (design doc, "Standing rule — schedule
/// robustness"). `ScheduleScoped` covers both "only one schedule has been run
/// yet" and "two schedules were run and they disagreed" — the two cases are
/// distinguished via `UalProbeReport::notes`, not via a separate enum variant,
/// since both cash out to the same actionable warning: don't treat this as a
/// portable, schedule-independent finding yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleStatus {
    ScheduleScoped,
    ReplicatedAcrossSchedules,
}

/// The mandatory three-field report for a single UAL probe run, plus the
/// schedule-status qualifier. See the design doc's "Learning-versus-expression
/// distinction" and "Standing rule — schedule robustness" sections.
#[derive(Debug, Clone)]
pub struct UalProbeReport {
    pub probe_id: &'static str,
    pub functional_outcome: FunctionalOutcome,
    pub schedule_status: ScheduleStatus,
    pub internal_association_formation: Presence,
    pub behavioral_expression: Presence,
    pub notes: Vec<String>,
}

impl UalProbeReport {
    /// Construct a report for a single schedule run. `functional_outcome` is
    /// derived, never passed in directly, so it is structurally impossible to
    /// report `Demonstrated` without `behavioral_expression == Observed`.
    pub fn new(
        probe_id: &'static str,
        behavioral_expression: Presence,
        internal_association_formation: Presence,
    ) -> Self {
        let functional_outcome = match behavioral_expression {
            Presence::Observed => FunctionalOutcome::Demonstrated,
            Presence::NotObserved => FunctionalOutcome::NotDemonstrated,
        };
        Self {
            probe_id,
            functional_outcome,
            schedule_status: ScheduleStatus::ScheduleScoped,
            internal_association_formation,
            behavioral_expression,
            notes: Vec::new(),
        }
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

impl fmt::Display for UalProbeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} functional outcome: {:?} — {:?}",
            self.probe_id, self.functional_outcome, self.schedule_status
        )?;
        writeln!(
            f,
            "Internal association formation: {:?}",
            self.internal_association_formation
        )?;
        write!(f, "Behavioral expression: {:?}", self.behavioral_expression)?;
        for note in &self.notes {
            write!(f, "\nNote: {note}")?;
        }
        Ok(())
    }
}

/// Combine two single-schedule reports (one per `UalSchedule` variant) into
/// one honest cross-schedule report. `ReplicatedAcrossSchedules` is granted
/// only when both schedules independently reached `Demonstrated` — a single
/// schedule showing the effect while the other doesn't is reported as
/// `ScheduleScoped` with an explicit disagreement note, never silently
/// dropped (design doc: "the explicit qualifier prevents a later summary from
/// silently dropping the schedule caveat").
pub fn combine_schedule_reports(
    probe_id: &'static str,
    blocked: &UalProbeReport,
    interleaved: &UalProbeReport,
) -> UalProbeReport {
    let both_demonstrated = blocked.functional_outcome == FunctionalOutcome::Demonstrated
        && interleaved.functional_outcome == FunctionalOutcome::Demonstrated;
    let behavioral_expression = if both_demonstrated {
        Presence::Observed
    } else {
        Presence::NotObserved
    };
    let internal_association_formation = if blocked.internal_association_formation
        == Presence::Observed
        || interleaved.internal_association_formation == Presence::Observed
    {
        Presence::Observed
    } else {
        Presence::NotObserved
    };
    let mut report = UalProbeReport::new(
        probe_id,
        behavioral_expression,
        internal_association_formation,
    );
    if both_demonstrated {
        report.schedule_status = ScheduleStatus::ReplicatedAcrossSchedules;
    } else if blocked.functional_outcome != interleaved.functional_outcome {
        report = report.with_note(format!(
            "schedules disagreed: blocked={:?}, interleaved={:?} — reporting as schedule-scoped, not replicated (this disagreement is itself a finding, not noise; see feedback_recall_harm_is_schedule_dependent)",
            blocked.functional_outcome, interleaved.functional_outcome
        ));
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn functional_outcome_cannot_be_demonstrated_without_behavioral_expression() {
        let r = UalProbeReport::new("UAL-TEST", Presence::NotObserved, Presence::Observed);
        assert_eq!(r.functional_outcome, FunctionalOutcome::NotDemonstrated);
        assert_eq!(r.internal_association_formation, Presence::Observed);
    }

    #[test]
    fn functional_outcome_demonstrated_requires_behavioral_expression_observed() {
        let r = UalProbeReport::new("UAL-TEST", Presence::Observed, Presence::Observed);
        assert_eq!(r.functional_outcome, FunctionalOutcome::Demonstrated);
    }

    #[test]
    fn combine_replicates_only_when_both_demonstrated() {
        let a = UalProbeReport::new("UAL-TEST", Presence::Observed, Presence::Observed);
        let b = UalProbeReport::new("UAL-TEST", Presence::Observed, Presence::Observed);
        let combined = combine_schedule_reports("UAL-TEST", &a, &b);
        assert_eq!(
            combined.schedule_status,
            ScheduleStatus::ReplicatedAcrossSchedules
        );
        assert_eq!(combined.functional_outcome, FunctionalOutcome::Demonstrated);
    }

    #[test]
    fn combine_reports_disagreement_as_schedule_scoped_not_silently_dropped() {
        let a = UalProbeReport::new("UAL-TEST", Presence::Observed, Presence::Observed);
        let b = UalProbeReport::new("UAL-TEST", Presence::NotObserved, Presence::NotObserved);
        let combined = combine_schedule_reports("UAL-TEST", &a, &b);
        assert_eq!(combined.schedule_status, ScheduleStatus::ScheduleScoped);
        assert_eq!(
            combined.functional_outcome,
            FunctionalOutcome::NotDemonstrated
        );
        assert!(
            !combined.notes.is_empty(),
            "disagreement must be recorded, not dropped"
        );
    }
}
