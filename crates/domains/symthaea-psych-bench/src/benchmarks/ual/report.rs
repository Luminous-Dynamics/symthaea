// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared reporting types for the UAL (Unlimited Associative Learning) Phase-1
//! probe packet (P1/P2/P4a). See `symthaea/docs/SYMTHAEA_UAL_EXTENSION_DESIGN_2026-07-27.md`
//! and `symthaea/docs/SYMTHAEA_UAL_PHASE1_PROTOCOLS_2026-07-27.md` for the full
//! rationale — this module implements the "mandatory three-field reporting
//! format", the schedule-status qualifier, a system-under-test identity, and
//! a fail-closed `Inconclusive` outcome.
//!
//! **Second claim-integrity pass (2026-07-27, same day as the first)**: a
//! follow-up review plus this codebase's own direct verification found the
//! first pass's own `combine_schedule_reports` had an asymmetric bug —
//! `either_inconclusive` picked `blocked`'s qualification unconditionally,
//! which happened to be correct only when the inconclusive arm was passed
//! first, silently producing `NotDemonstrated` instead of `Inconclusive` in
//! the reverse ordering. Fixed by splitting `UalRuntimeQualification` into
//! `UalStaticQualification` (design-level facts verified once, by a
//! dedicated test elsewhere — never re-derived per run) and
//! `UalRunQualification` (computed fresh from this specific run's data),
//! each with a symmetric `.and()` combinator, so combining two arms'
//! qualifications is now commutative by construction rather than a
//! positional pick.
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
/// now used by `p1_live_diagnostic.rs`'s exploratory pilot (`symthaea-backend`
/// feature) — deliberately scoped as a pilot, not a formal preregistered
/// probe, since using this variant to license a `Demonstrated`/`NotDemonstrated`
/// verdict without the same rigor the rest of this module applies would
/// itself be a claim-integrity violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemUnderTest {
    ValueTable,
    FirstOrderLearner,
    GraphPropagation,
    StaticHdc,
    /// The benchmark's own candidate HDC mechanism (P2/P4a's former
    /// `BaselineRung::FullSymthaea`, renamed `CandidateHdcLearner`).
    BenchmarkLocalHdcLearner,
    /// A real, backend-gated `CognitiveLoopService` adapter (`p1_live_diagnostic.rs`,
    /// `symthaea-backend` feature). First use is deliberately an exploratory
    /// pilot, not a formal preregistered probe — see that file's module doc
    /// for why its results don't carry the same evidentiary weight as the
    /// rest of this module's `UalProbeReport`-backed verdicts.
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
/// control that failed, an unusable signal, or an unresolved construct-
/// validity question) resolves to `Inconclusive` regardless of what the raw
/// behavioral/internal presence flags say — a broken or not-yet-validated
/// experiment must never be able to report a confident positive OR negative.
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

/// Design-level facts verified ONCE, independent of any specific run — e.g.
/// "this negative control's design rules out the confound it claims to",
/// "the leakage path has no ground-truth channel by construction", "the
/// baseline ladder's calibration case passes by construction". Each field is
/// backed by a named, dedicated `#[test]` elsewhere in the same probe file
/// (cited in the field-setting code), not re-derived every aggregate call —
/// **this is NOT evidence about any particular empirical run**, only that
/// the experimental design itself is sound. Kept structurally separate from
/// `UalRunQualification` so a static design fact can never be mistaken for a
/// per-run measurement (the exact conflation a follow-up review flagged in
/// the first version of this type).
#[derive(Debug, Clone, Copy)]
pub struct UalStaticQualification {
    pub control_design_valid: bool,
    pub leakage_path_tested: bool,
    pub baseline_design_valid: bool,
    /// Does the readout/retrieval mechanism's *construct validity* itself
    /// hold — i.e. is it established that this mechanism measures what the
    /// probe claims to measure? Distinct from whether it produces a
    /// significant value. Set `false` when this is an open question (e.g.
    /// P4a's compositional-recombination interpretation pending the HDC
    /// binding/unbinding audit) — a probe cannot report `Demonstrated` while
    /// its own measurement's validity is unresolved, no matter how clean the
    /// numbers look.
    pub construct_validity_passed: bool,
}

impl UalStaticQualification {
    pub fn all_passed(&self) -> bool {
        self.control_design_valid
            && self.leakage_path_tested
            && self.baseline_design_valid
            && self.construct_validity_passed
    }

    pub fn and(&self, other: &Self) -> Self {
        Self {
            control_design_valid: self.control_design_valid && other.control_design_valid,
            leakage_path_tested: self.leakage_path_tested && other.leakage_path_tested,
            baseline_design_valid: self.baseline_design_valid && other.baseline_design_valid,
            construct_validity_passed: self.construct_validity_passed
                && other.construct_validity_passed,
        }
    }

    fn failure_reasons(&self) -> Vec<&'static str> {
        let mut reasons = Vec::new();
        if !self.control_design_valid {
            reasons.push("control_design_valid=false");
        }
        if !self.leakage_path_tested {
            reasons.push("leakage_path_tested=false");
        }
        if !self.baseline_design_valid {
            reasons.push("baseline_design_valid=false");
        }
        if !self.construct_validity_passed {
            reasons.push("construct_validity_passed=false");
        }
        reasons
    }
}

/// Facts computed FRESH from this specific run/aggregate's actual data —
/// the counterpart to `UalStaticQualification`. Field names are deliberately
/// "observed"/"executed", not "passed", where the underlying signal is an
/// internal-responsiveness observation rather than an independently-verified
/// control (see each probe's field-setting code for the honest scope of
/// what was actually checked this run).
#[derive(Debug, Clone, Copy)]
pub struct UalRunQualification {
    /// Did the intended manipulation actually occur in this run/aggregate
    /// (e.g. did a reversal actually happen; were pairing trials actually
    /// executed)? This is the field that P1's non-reversing hazard runs are
    /// meant to fail.
    pub manipulation_applied: bool,
    /// An internal-responsiveness observation this run (e.g. "did the
    /// learned value move at all"), NOT an independently-verified positive
    /// control in the classical sense unless the probe's own comment says
    /// otherwise.
    pub positive_control_observed: bool,
    pub negative_control_observed: bool,
    /// Is the behavioral signal itself finite/usable (not NaN, not from a
    /// zero-sample aggregate)?
    pub signal_usable: bool,
    /// Did the schedule mechanism itself execute as declared (e.g. did
    /// "interleaved" actually produce a different trial ordering than
    /// "blocked")? This checks the manipulation was *applied*, not that it
    /// *mattered* to the outcome — a probe whose mechanism is provably
    /// schedule-invariant (see P2's module doc) can have this field `true`
    /// while still needing an honest note that "replication" is trivial for
    /// that specific mechanism.
    pub schedule_executed_as_declared: bool,
}

impl UalRunQualification {
    pub fn all_passed(&self) -> bool {
        self.manipulation_applied
            && self.positive_control_observed
            && self.negative_control_observed
            && self.signal_usable
            && self.schedule_executed_as_declared
    }

    pub fn and(&self, other: &Self) -> Self {
        Self {
            manipulation_applied: self.manipulation_applied && other.manipulation_applied,
            positive_control_observed: self.positive_control_observed
                && other.positive_control_observed,
            negative_control_observed: self.negative_control_observed
                && other.negative_control_observed,
            signal_usable: self.signal_usable && other.signal_usable,
            schedule_executed_as_declared: self.schedule_executed_as_declared
                && other.schedule_executed_as_declared,
        }
    }

    fn failure_reasons(&self) -> Vec<&'static str> {
        let mut reasons = Vec::new();
        if !self.manipulation_applied {
            reasons.push("manipulation_applied=false");
        }
        if !self.positive_control_observed {
            reasons.push("positive_control_observed=false");
        }
        if !self.negative_control_observed {
            reasons.push("negative_control_observed=false");
        }
        if !self.signal_usable {
            reasons.push("signal_usable=false");
        }
        if !self.schedule_executed_as_declared {
            reasons.push("schedule_executed_as_declared=false");
        }
        reasons
    }
}

/// Fail-closed runtime-qualification record, composed of the static/run
/// split above. A probe's functional outcome must not be interpretable
/// until BOTH the experimental design is known to be sound AND this
/// specific run is known to have executed validly.
#[derive(Debug, Clone, Copy)]
pub struct UalRuntimeQualification {
    pub static_qual: UalStaticQualification,
    pub run_qual: UalRunQualification,
}

impl UalRuntimeQualification {
    pub fn all_passed(&self) -> bool {
        self.static_qual.all_passed() && self.run_qual.all_passed()
    }

    /// Symmetric combination for merging two schedule arms' qualifications.
    /// Field-wise AND in both sub-structs — commutative and associative, so
    /// which arm is "blocked" vs "interleaved" cannot affect the result
    /// (the asymmetry bug this replaces depended on argument order).
    pub fn and(&self, other: &Self) -> Self {
        Self {
            static_qual: self.static_qual.and(&other.static_qual),
            run_qual: self.run_qual.and(&other.run_qual),
        }
    }

    pub fn failure_reasons(&self) -> Vec<&'static str> {
        let mut reasons = self.static_qual.failure_reasons();
        reasons.extend(self.run_qual.failure_reasons());
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
///
/// Qualification is combined via `UalRuntimeQualification::and` — a
/// symmetric, field-wise AND across both arms — rather than picking one
/// arm's qualification by position. Panics if the two reports have different
/// `system_under_test` identities: combining reports for two different
/// mechanisms into one verdict would itself be a claim-integrity violation,
/// and this should never happen from a correctly-written `ual_report()`.
pub fn combine_schedule_reports(
    probe_id: &'static str,
    blocked: &UalProbeReport,
    interleaved: &UalProbeReport,
) -> UalProbeReport {
    assert_eq!(
        blocked.system_under_test, interleaved.system_under_test,
        "combine_schedule_reports: cannot combine reports for two different systems under test"
    );

    let both_demonstrated = blocked.functional_outcome == FunctionalOutcome::Demonstrated
        && interleaved.functional_outcome == FunctionalOutcome::Demonstrated;
    let either_inconclusive = blocked.functional_outcome == FunctionalOutcome::Inconclusive
        || interleaved.functional_outcome == FunctionalOutcome::Inconclusive;

    // Symmetric AND, not a positional pick: if either arm's qualification
    // failed anywhere, the combined qualification fails there too,
    // regardless of which arm is passed as `blocked` vs `interleaved`.
    let qualification = blocked.qualification.and(&interleaved.qualification);
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
    // notes (bug found in the first repair pass: the original version
    // discarded them and only conditionally added a combine-level note, so
    // a both-NotDemonstrated-with-no-disagreement result — a real, common
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
            static_qual: UalStaticQualification {
                control_design_valid: true,
                leakage_path_tested: true,
                baseline_design_valid: true,
                construct_validity_passed: true,
            },
            run_qual: UalRunQualification {
                manipulation_applied: true,
                positive_control_observed: true,
                negative_control_observed: true,
                signal_usable: true,
                schedule_executed_as_declared: true,
            },
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
        bad.run_qual.manipulation_applied = false;
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
    fn construct_validity_failure_forces_inconclusive_even_with_clean_data() {
        // The P4a fix this test guards: a mechanism whose construct validity
        // is unresolved must not report Demonstrated no matter how clean its
        // behavioral/internal signals look.
        let mut bad = qualified();
        bad.static_qual.construct_validity_passed = false;
        let r = UalProbeReport::new(
            "UAL-TEST",
            SystemUnderTest::BenchmarkLocalHdcLearner,
            bad,
            Presence::Observed,
            Presence::Observed,
        );
        assert_eq!(r.functional_outcome, FunctionalOutcome::Inconclusive);
        assert!(
            r.notes
                .iter()
                .any(|n| n.contains("construct_validity_passed"))
        );
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

    /// **Permutation 1** (this ordering happened to work even with the
    /// pre-fix asymmetric bug, which is exactly why it didn't catch the bug
    /// — see Permutation 2 below for the one that actually regresses it).
    #[test]
    fn combine_reports_inconclusive_as_blocked_arm_is_inconclusive() {
        let mut bad_qual = qualified();
        bad_qual.run_qual.manipulation_applied = false;
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

    /// **Permutation 2 — the one that regressed the pre-fix asymmetric bug**:
    /// with the qualification-selection bug, `either_inconclusive` was true
    /// here too, but the buggy code unconditionally picked `blocked`'s
    /// qualification (the QUALIFIED one, since `demonstrated` is now
    /// `blocked`), producing `NotDemonstrated` instead of `Inconclusive`.
    /// The symmetric `.and()` fix makes both orderings agree.
    #[test]
    fn combine_reports_inconclusive_as_interleaved_arm_is_also_inconclusive() {
        let mut bad_qual = qualified();
        bad_qual.run_qual.manipulation_applied = false;
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
        let combined = combine_schedule_reports("UAL-TEST", &demonstrated, &inconclusive);
        assert_eq!(combined.functional_outcome, FunctionalOutcome::Inconclusive);
        assert!(combined.notes.iter().any(|n| n.contains("inconclusive")));
    }

    #[test]
    #[should_panic(expected = "cannot combine reports for two different systems under test")]
    fn combine_reports_panics_on_mismatched_system_under_test() {
        let a = UalProbeReport::new(
            "UAL-TEST",
            SystemUnderTest::BenchmarkLocalHdcLearner,
            qualified(),
            Presence::Observed,
            Presence::Observed,
        );
        let b = UalProbeReport::new(
            "UAL-TEST",
            SystemUnderTest::FirstOrderLearner,
            qualified(),
            Presence::Observed,
            Presence::Observed,
        );
        let _ = combine_schedule_reports("UAL-TEST", &a, &b);
    }
}
