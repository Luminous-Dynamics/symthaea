// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared reporting types for the UAL (Unlimited Associative Learning) Phase-1
//! probe packet (P1/P2/P4a). See `symthaea/docs/SYMTHAEA_UAL_EXTENSION_DESIGN_2026-07-27.md`
//! and `symthaea/docs/SYMTHAEA_UAL_PHASE1_PROTOCOLS_2026-07-27.md` for the full
//! rationale — this module implements the "mandatory three-field reporting
//! format", the schedule-status qualifier, and (added in the claim-integrity
//! repair pass, 2026-07-27) an explicit system-under-test identity plus a
//! fail-closed `Inconclusive` outcome, after an independent review plus this
//! codebase's own direct verification found the original two-valued
//! Demonstrated/NotDemonstrated model could not represent an invalid
//! experiment (e.g. a manipulation that never actually applied) without
//! silently miscounting it as a real negative result.
//!
//! **This is deliberately NOT a 15th Butlin indicator.** It does not import or
//! extend `crate::benchmarks::butlin::report::IndicatorEvidence` — UAL is a
//! functional-capacity theory outside the frozen Butlin-14 denominator, and
//! must stay reportable as its own thing (design doc, "Why UAL sits outside
//! the frozen Butlin denominator").

use std::fmt;

/// A presentation-order schedule for a UAL probe's multi-schedule replication
/// requirement. Each probe interprets these two variants concretely in its own
/// module doc comment (e.g. P2/P4a: `Blocked`/`Interleaved` genuinely mean
/// blocked-vs-interleaved trial-type presentation). **P1 does NOT use this
/// shared type** — its "reversal timing" manipulation isn't interleaving of
/// trial types at all, and reusing this enum's `Interleaved` name for it was
/// a real naming trap (claim-integrity repair pass, 2026-07-27); P1 now
/// defines its own local `P1Schedule { FixedChangePoint, VariableChangePoint }`.
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

/// Identifies what actually produced a report's numbers. Added in the
/// claim-integrity repair pass after both an independent review and this
/// codebase's own direct verification found the original `FullSymthaea`
/// enum-variant name (in `p2_second_order.rs`/`p4a_recombination.rs`)
/// misleadingly implied the production Symthaea cognitive loop was being
/// exercised, when in fact none of P1/P2/P4a construct a
/// `CognitiveLoopService` or touch any live/production pathway at all —
/// every rung, including the "headline" one, is a benchmark-local reference
/// mechanism built from the crate's own HDC/scalar primitives (matching the
/// existing pattern in `reward_learning.rs`/`srtt.rs`). `LiveSymthaea` is
/// reserved for a future backend-gated adapter that does not exist yet;
/// using it before that adapter exists would itself be a claim-integrity
/// violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemUnderTest {
    ValueTable,
    FirstOrderLearner,
    GraphPropagation,
    StaticHdc,
    /// The benchmark's own candidate HDC mechanism (P2/P4a's former
    /// `BaselineRung::FullSymthaea`, renamed `CandidateHdcLearner`).
    BenchmarkLocalHdcLearner,
    /// Not used by any probe yet. Reserved for a real, backend-gated
    /// `CognitiveLoopService` adapter.
    LiveSymthaea,
}

impl fmt::Display for SystemUnderTest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SystemUnderTest::ValueTable => "value table (unlearned baseline)",
            SystemUnderTest::FirstOrderLearner => "first-order delta-rule learner",
            SystemUnderTest::GraphPropagation => "graph-propagation baseline",
            SystemUnderTest::StaticHdc => "static HDC binding, no value learning",
            SystemUnderTest::BenchmarkLocalHdcLearner => {
                "benchmark-local candidate HDC learner (NOT live Symthaea)"
            }
            SystemUnderTest::LiveSymthaea => "live Symthaea (production cognitive loop)",
        };
        write!(f, "{s}")
    }
}

/// The probe-level functional-outcome verdict. `Demonstrated`/`NotDemonstrated`
/// may only be reached when `UalRuntimeQualification::all_passed()` is true —
/// enforced structurally in `UalProbeReport::new`, not left to the caller's
/// discretion. An unqualified run (a manipulation that didn't apply, a
/// control that failed, an unusable signal) resolves to `Inconclusive`
/// regardless of what the raw behavioral/internal presence flags say — a
/// broken experiment must never be able to report a confident negative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FunctionalOutcome {
    Demonstrated,
    NotDemonstrated,
    Inconclusive,
}

/// Schedule-robustness qualifier (design doc, "Standing rule — schedule
/// robustness"). `ScheduleScoped` covers "only one schedule has been run
/// yet", "two schedules disagreed", and "at least one schedule arm was
/// itself inconclusive" — all three cash out to the same actionable warning:
/// don't treat this as a portable, schedule-independent finding yet. The
/// specific reason is always recorded in `UalProbeReport::notes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleStatus {
    ScheduleScoped,
    ReplicatedAcrossSchedules,
}

/// Fail-closed runtime-qualification record. Added in the claim-integrity
/// repair pass, importing the same discipline the Butlin qualification
/// pipeline already applies (`butlin::qualification_design`/
/// `qualification_runtime`) — a probe's functional outcome must not be
/// interpretable until the run itself is known to have been valid.
///
/// Each field is a real, computed check specific to the probe that
/// constructs it (see each probe module's `schedule_report` for exactly what
/// is checked); none of these are placeholders defaulted to `true`. Where a
/// probe's negative/leakage control is a structural invariant already
/// covered by a dedicated `#[test]` elsewhere in the same file (e.g. a
/// choice function with no ground-truth parameter to leak, checked once at
/// compile-call-signature level, not per aggregate run) that is noted
/// explicitly in the field-setting code, not silently assumed.
#[derive(Debug, Clone, Copy)]
pub struct UalRuntimeQualification {
    /// Did the intended manipulation actually occur in this run/aggregate
    /// (e.g. did a reversal actually happen; were pairing trials actually
    /// executed)? This is the field that P1's non-reversing hazard runs are
    /// meant to fail.
    pub manipulation_applied: bool,
    pub positive_control_passed: bool,
    pub negative_controls_passed: bool,
    pub leakage_checks_passed: bool,
    pub baseline_ladder_valid: bool,
    /// Is the behavioral signal itself finite/usable (not NaN, not from a
    /// zero-sample aggregate)?
    pub signal_usable: bool,
    /// Did the schedule manipulation itself behave as intended for this
    /// probe (distinct from `manipulation_applied`'s per-run reversal check
    /// — this is about the schedule mechanism, e.g. "did interleaving
    /// actually interleave", not the underlying task manipulation)?
    pub schedule_valid: bool,
}

impl UalRuntimeQualification {
    pub fn all_passed(&self) -> bool {
        self.manipulation_applied
            && self.positive_control_passed
            && self.negative_controls_passed
            && self.leakage_checks_passed
            && self.baseline_ladder_valid
            && self.signal_usable
            && self.schedule_valid
    }

    /// Which fields failed, for inclusion in report notes. Never silently
    /// drop this — an `Inconclusive` result with no stated reason is exactly
    /// the kind of unexplained gap this repair pass exists to prevent.
    pub fn failure_reasons(&self) -> Vec<&'static str> {
        let mut reasons = Vec::new();
        if !self.manipulation_applied {
            reasons.push("manipulation_applied=false");
        }
        if !self.positive_control_passed {
            reasons.push("positive_control_passed=false");
        }
        if !self.negative_controls_passed {
            reasons.push("negative_controls_passed=false");
        }
        if !self.leakage_checks_passed {
            reasons.push("leakage_checks_passed=false");
        }
        if !self.baseline_ladder_valid {
            reasons.push("baseline_ladder_valid=false");
        }
        if !self.signal_usable {
            reasons.push("signal_usable=false");
        }
        if !self.schedule_valid {
            reasons.push("schedule_valid=false");
        }
        reasons
    }
}

/// The mandatory reporting record for a single UAL probe run: functional
/// outcome, schedule-status qualifier, system-under-test identity, and the
/// internal-vs-behavioral split. See the design doc's "Learning-versus-
/// expression distinction" and "Standing rule — schedule robustness"
/// sections.
#[derive(Debug, Clone)]
pub struct UalProbeReport {
    pub probe_id: &'static str,
    pub system_under_test: SystemUnderTest,
    pub functional_outcome: FunctionalOutcome,
    pub schedule_status: ScheduleStatus,
    pub internal_association_formation: Presence,
    pub behavioral_expression: Presence,
    pub qualification: UalRuntimeQualification,
    pub notes: Vec<String>,
}

impl UalProbeReport {
    /// Construct a report for a single schedule run. `functional_outcome` is
    /// derived, never passed in directly: fail-closed to `Inconclusive` if
    /// `qualification` didn't fully pass, otherwise
    /// `Demonstrated`/`NotDemonstrated` from `behavioral_expression` alone —
    /// so it is structurally impossible to report `Demonstrated` either
    /// without `behavioral_expression == Observed` OR from an unqualified
    /// run.
    pub fn new(
        probe_id: &'static str,
        system_under_test: SystemUnderTest,
        qualification: UalRuntimeQualification,
        behavioral_expression: Presence,
        internal_association_formation: Presence,
    ) -> Self {
        let mut notes = Vec::new();
        let functional_outcome = if !qualification.all_passed() {
            for reason in qualification.failure_reasons() {
                notes.push(format!("inconclusive: {reason}"));
            }
            FunctionalOutcome::Inconclusive
        } else {
            match behavioral_expression {
                Presence::Observed => FunctionalOutcome::Demonstrated,
                Presence::NotObserved => FunctionalOutcome::NotDemonstrated,
            }
        };
        Self {
            probe_id,
            system_under_test,
            functional_outcome,
            schedule_status: ScheduleStatus::ScheduleScoped,
            internal_association_formation,
            behavioral_expression,
            qualification,
            notes,
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
        writeln!(f, "System under test: {}", self.system_under_test)?;
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

/// Combine two single-schedule reports into one honest cross-schedule
/// report. `ReplicatedAcrossSchedules` is granted only when both schedules
/// independently reached `Demonstrated` — an `Inconclusive` arm on either
/// side, a disagreement, or two consistent `NotDemonstrated`s all resolve to
/// `ScheduleScoped` with an explicit note, never silently dropped.
pub fn combine_schedule_reports(
    probe_id: &'static str,
    blocked: &UalProbeReport,
    interleaved: &UalProbeReport,
) -> UalProbeReport {
    let both_demonstrated = blocked.functional_outcome == FunctionalOutcome::Demonstrated
        && interleaved.functional_outcome == FunctionalOutcome::Demonstrated;
    let either_inconclusive = blocked.functional_outcome == FunctionalOutcome::Inconclusive
        || interleaved.functional_outcome == FunctionalOutcome::Inconclusive;

    let qualification = if either_inconclusive {
        blocked.qualification
    } else {
        interleaved.qualification
    };
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
        blocked.system_under_test,
        qualification,
        behavioral_expression,
        internal_association_formation,
    );
    // Always carry forward both individual schedule reports' own diagnostic
    // notes (bug found while wiring this in: the original version discarded
    // them and only conditionally added a combine-level note, so a
    // both-NotDemonstrated-with-no-disagreement result — a real, common
    // outcome — ended up with an empty notes list, silently dropping the
    // actual measured numbers each schedule arm recorded).
    for note in &blocked.notes {
        report = report.with_note(format!("[blocked] {note}"));
    }
    for note in &interleaved.notes {
        report = report.with_note(format!("[interleaved] {note}"));
    }
    if either_inconclusive {
        report = report.with_note(format!(
            "at least one schedule arm was inconclusive: blocked={:?}, interleaved={:?}",
            blocked.functional_outcome, interleaved.functional_outcome
        ));
    } else if both_demonstrated {
        report.schedule_status = ScheduleStatus::ReplicatedAcrossSchedules;
        report = report.with_note("both schedules independently reached Demonstrated");
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

    fn qualified() -> UalRuntimeQualification {
        UalRuntimeQualification {
            manipulation_applied: true,
            positive_control_passed: true,
            negative_controls_passed: true,
            leakage_checks_passed: true,
            baseline_ladder_valid: true,
            signal_usable: true,
            schedule_valid: true,
        }
    }

    #[test]
    fn functional_outcome_cannot_be_demonstrated_without_behavioral_expression() {
        let r = UalProbeReport::new(
            "UAL-TEST",
            SystemUnderTest::BenchmarkLocalHdcLearner,
            qualified(),
            Presence::NotObserved,
            Presence::Observed,
        );
        assert_eq!(r.functional_outcome, FunctionalOutcome::NotDemonstrated);
        assert_eq!(r.internal_association_formation, Presence::Observed);
    }

    #[test]
    fn functional_outcome_demonstrated_requires_behavioral_expression_observed() {
        let r = UalProbeReport::new(
            "UAL-TEST",
            SystemUnderTest::BenchmarkLocalHdcLearner,
            qualified(),
            Presence::Observed,
            Presence::Observed,
        );
        assert_eq!(r.functional_outcome, FunctionalOutcome::Demonstrated);
    }

    #[test]
    fn unqualified_run_is_inconclusive_even_with_behavioral_signal_observed() {
        // The core fail-closed guarantee: a broken experiment must not be
        // able to report a confident Demonstrated OR NotDemonstrated.
        let mut bad = qualified();
        bad.manipulation_applied = false;
        let r = UalProbeReport::new(
            "UAL-TEST",
            SystemUnderTest::BenchmarkLocalHdcLearner,
            bad,
            Presence::Observed,
            Presence::Observed,
        );
        assert_eq!(r.functional_outcome, FunctionalOutcome::Inconclusive);
        assert!(r.notes.iter().any(|n| n.contains("manipulation_applied")));
    }

    #[test]
    fn combine_replicates_only_when_both_demonstrated() {
        let a = UalProbeReport::new(
            "UAL-TEST",
            SystemUnderTest::BenchmarkLocalHdcLearner,
            qualified(),
            Presence::Observed,
            Presence::Observed,
        );
        let b = UalProbeReport::new(
            "UAL-TEST",
            SystemUnderTest::BenchmarkLocalHdcLearner,
            qualified(),
            Presence::Observed,
            Presence::Observed,
        );
        let combined = combine_schedule_reports("UAL-TEST", &a, &b);
        assert_eq!(
            combined.schedule_status,
            ScheduleStatus::ReplicatedAcrossSchedules
        );
        assert_eq!(combined.functional_outcome, FunctionalOutcome::Demonstrated);
    }

    #[test]
    fn combine_reports_disagreement_as_schedule_scoped_not_silently_dropped() {
        let a = UalProbeReport::new(
            "UAL-TEST",
            SystemUnderTest::BenchmarkLocalHdcLearner,
            qualified(),
            Presence::Observed,
            Presence::Observed,
        );
        let b = UalProbeReport::new(
            "UAL-TEST",
            SystemUnderTest::BenchmarkLocalHdcLearner,
            qualified(),
            Presence::NotObserved,
            Presence::NotObserved,
        );
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

    #[test]
    fn combine_reports_inconclusive_arm_as_inconclusive_not_averaged_away() {
        let mut bad_qual = qualified();
        bad_qual.manipulation_applied = false;
        let inconclusive = UalProbeReport::new(
            "UAL-TEST",
            SystemUnderTest::BenchmarkLocalHdcLearner,
            bad_qual,
            Presence::Observed,
            Presence::Observed,
        );
        let demonstrated = UalProbeReport::new(
            "UAL-TEST",
            SystemUnderTest::BenchmarkLocalHdcLearner,
            qualified(),
            Presence::Observed,
            Presence::Observed,
        );
        let combined = combine_schedule_reports("UAL-TEST", &inconclusive, &demonstrated);
        assert_eq!(combined.functional_outcome, FunctionalOutcome::Inconclusive);
        assert!(combined.notes.iter().any(|n| n.contains("inconclusive")));
    }
}
