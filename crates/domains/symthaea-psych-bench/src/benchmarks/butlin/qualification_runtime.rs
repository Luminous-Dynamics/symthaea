// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Runtime qualification model for the Butlin Probe Qualification campaign
//! (see `symthaea/docs/BUTLIN_PR_B_RUNNER_PLAN_2026-07-27.md`) — steps 3-4 of
//! that plan's recommended sequencing. **This module still does not touch a
//! real `CognitiveLoopService`.** It defines the logic a future runner will
//! use to decide outcomes, and proves that logic correct via synthetic
//! *conformance fixtures* — deterministic, fabricated inputs that test
//! whether the mapping from qualification state to `EvidenceOutcome` is
//! right, not whether Symthaea exhibits any indicator.
//!
//! **This is not evidence about Symthaea.** Every test in this module is a
//! runner-conformance fixture (per the plan doc's explicit split) — it
//! proves the *contract* holds (broken controls, execution proxies, and
//! malformed interventions all resolve to `Inconclusive` or a hard error,
//! never an attractive false positive/negative), using fabricated
//! `RuntimeQualification` values, never a real cognitive-loop measurement.
//! The plan's step 5 (running `AE-2` for real) is deliberately **not**
//! implemented here — per the plan's own sequencing, that should only
//! happen once this contract layer is proven correct in isolation.
//!
//! Always compiled (no `symthaea-backend` feature needed) — same rationale
//! as `qualification_design.rs`: this is pure decision logic over already-
//! declared static data plus fabricated runtime-check results, not a call
//! into `symthaea::cognitive_loop`. A real runner (backend-gated, not built
//! yet) would populate `RuntimeQualification`'s fields from actual
//! measurements and call the same `resolve_outcome`/`check_identity_against_registry`
//! functions this module already tests against fixtures.

use super::qualification_design::QualificationDesign;
use super::report::{EvidenceOutcome, SupportTier};

/// Why a run's overall qualification failed, if it did. Named reasons, not
/// a single opaque `false` — see `BUTLIN_PR_B_RUNNER_PLAN_2026-07-27.md`'s
/// `RuntimeQualification` sketch for why this matters (diagnosability).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationFailure {
    /// The row's static design itself doesn't qualify (see
    /// `QualificationDesign::static_design_qualifies`) — no runtime check
    /// can rescue this; e.g. `GWT-3`'s `ExecutionProxy` probe.
    StaticDesignDoesNotQualify,
    /// The targeted ablation lever was supposed to run but didn't (or the
    /// run crashed/errored before it could).
    InterventionNotApplied,
    /// The intervention changed more than the intended state — e.g. it also
    /// perturbed an unrelated field, so any resulting effect can't be
    /// attributed cleanly to the targeted mechanism.
    InterventionNotSpecific,
    /// The positive control's manipulation was applied, but its claimed
    /// effect wasn't actually observed this run — an achievability/runtime
    /// failure of the exact kind `ControlReadiness::FormulaVerified` can't
    /// rule out on its own (see `qualification_design.rs`'s Finding B).
    PositiveControlEffectNotObserved,
    /// The sham *did* produce the targeted effect (or failed to run at
    /// all), undermining its use as a specificity control this run.
    ShamDidNotBehaveAsExpected,
    /// The probe's signal wasn't usable this run (e.g. non-finite values,
    /// or the same `ExecutionProxy`/`Unverified` concern re-confirmed at
    /// runtime rather than only assumed from the static design).
    ProbeSignalNotUsable,
    /// The row's declared identity doesn't match what the live registry
    /// says right now — see `check_identity_against_registry`. This is
    /// deliberately surfaced as a **hard error**, not folded into ordinary
    /// qualification failure, at the point it's detected (see
    /// `QualificationRunError`) — listed here too so a `RuntimeQualification`
    /// that was constructed *after* an identity check already passed can
    /// still represent "this would have failed the check" for reporting.
    IdentityOrConfigMismatch,
}

/// The full runtime qualification record for one experimental run. Stores
/// named components, not a single boolean, so a failure is diagnosable —
/// see `failure_reasons()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeQualification {
    pub static_design_qualifies: bool,
    pub intervention_applied: bool,
    pub intervention_specificity_passed: bool,
    pub positive_control_effect_observed: bool,
    pub sham_behaved_as_expected: bool,
    pub probe_signal_usable: bool,
    pub identity_and_config_match: bool,
}

impl RuntimeQualification {
    /// Start from a design's static layer, with every runtime-only field
    /// defaulted to `false` ("not yet checked") — deliberately does **not**
    /// inherit a pass from `static_design_qualifies`. A caller must
    /// explicitly confirm each runtime check; nothing here silently upgrades
    /// a static pass into a runtime one.
    pub fn from_static_design(design: &QualificationDesign) -> Self {
        Self {
            static_design_qualifies: design.static_design_qualifies(),
            intervention_applied: false,
            intervention_specificity_passed: false,
            positive_control_effect_observed: false,
            sham_behaved_as_expected: false,
            probe_signal_usable: false,
            identity_and_config_match: false,
        }
    }

    /// Every reason this run fails to qualify, in a fixed order. Empty iff
    /// `qualifies_run()` is `true`.
    pub fn failure_reasons(&self) -> Vec<QualificationFailure> {
        let mut reasons = Vec::new();
        if !self.static_design_qualifies {
            reasons.push(QualificationFailure::StaticDesignDoesNotQualify);
        }
        if !self.intervention_applied {
            reasons.push(QualificationFailure::InterventionNotApplied);
        }
        if !self.intervention_specificity_passed {
            reasons.push(QualificationFailure::InterventionNotSpecific);
        }
        if !self.positive_control_effect_observed {
            reasons.push(QualificationFailure::PositiveControlEffectNotObserved);
        }
        if !self.sham_behaved_as_expected {
            reasons.push(QualificationFailure::ShamDidNotBehaveAsExpected);
        }
        if !self.probe_signal_usable {
            reasons.push(QualificationFailure::ProbeSignalNotUsable);
        }
        if !self.identity_and_config_match {
            reasons.push(QualificationFailure::IdentityOrConfigMismatch);
        }
        reasons
    }

    pub fn qualifies_run(&self) -> bool {
        self.failure_reasons().is_empty()
    }
}

/// A hard failure distinct from ordinary qualification failure — the row's
/// declared identity doesn't match what the live registry says right now.
/// Mirrors `report.rs`'s `EvidenceMergeError::ClassificationMismatch`
/// philosophy at this layer: a self-contradictory or stale row must be
/// rejected outright, not silently downgraded to `Inconclusive`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualificationRunError {
    RegistryIdentityMismatch {
        indicator: &'static str,
        field: &'static str,
        declared: String,
        actual: String,
    },
}

impl std::fmt::Display for QualificationRunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QualificationRunError::RegistryIdentityMismatch {
                indicator,
                field,
                declared,
                actual,
            } => write!(
                f,
                "{indicator}: declared {field} {declared:?} does not match the live registry's \
                 {actual:?} — bundle rejected as stale/malformed, not silently downgraded"
            ),
        }
    }
}
impl std::error::Error for QualificationRunError {}

/// Check a design's declared identity against what the live registry
/// actually says right now. Pure and backend-free: the caller (a future
/// real runner) is responsible for fetching `live_target_lever`/
/// `live_functional_benchmark` from `ablation::ablation_specs()` (backend-
/// gated) and passing them in as plain strings — this function itself
/// never touches `symthaea::cognitive_loop`.
pub fn check_identity_against_registry(
    design: &QualificationDesign,
    live_target_lever: &str,
    live_functional_benchmark: &str,
) -> Result<(), QualificationRunError> {
    if design.target_lever != live_target_lever {
        return Err(QualificationRunError::RegistryIdentityMismatch {
            indicator: design.indicator,
            field: "target_lever",
            declared: design.target_lever.to_string(),
            actual: live_target_lever.to_string(),
        });
    }
    if design.functional_benchmark != live_functional_benchmark {
        return Err(QualificationRunError::RegistryIdentityMismatch {
            indicator: design.indicator,
            field: "functional_benchmark",
            declared: design.functional_benchmark.to_string(),
            actual: live_functional_benchmark.to_string(),
        });
    }
    Ok(())
}

/// Map a run's qualification state plus observed effects to an
/// `EvidenceOutcome` — reuses `report.rs`'s existing outcome vocabulary
/// rather than inventing a parallel one, so this stays consistent with
/// PR #30's evidence model. A run that doesn't qualify is `Inconclusive`
/// regardless of what the raw scores show; only a qualified run's actual
/// observed effect determines `NotDemonstrated` vs. `Supported`.
pub fn resolve_outcome(
    qualification: &RuntimeQualification,
    indicator_effect_observed: bool,
    functional_effect_observed: bool,
) -> EvidenceOutcome {
    if !qualification.qualifies_run() {
        return EvidenceOutcome::Inconclusive;
    }
    if !indicator_effect_observed {
        return EvidenceOutcome::NotDemonstrated;
    }
    if functional_effect_observed {
        EvidenceOutcome::Supported(SupportTier::FunctionallySupported)
    } else {
        EvidenceOutcome::Supported(SupportTier::CausallySupported)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::butlin::qualification_design::planned_designs;

    fn design(indicator: &str) -> QualificationDesign {
        *planned_designs()
            .iter()
            .find(|d| d.indicator == indicator)
            .unwrap()
    }

    fn fully_passing(design: &QualificationDesign) -> RuntimeQualification {
        RuntimeQualification {
            static_design_qualifies: design.static_design_qualifies(),
            intervention_applied: true,
            intervention_specificity_passed: true,
            positive_control_effect_observed: true,
            sham_behaved_as_expected: true,
            probe_signal_usable: true,
            identity_and_config_match: true,
        }
    }

    // ── RuntimeQualification / failure_reasons ──────────────────────────

    #[test]
    fn test_from_static_design_never_inherits_a_runtime_pass() {
        // Even for a row whose static design qualifies, every runtime-only
        // field must start false -- nothing here silently upgrades a static
        // pass into a runtime one.
        let ae2 = design("AE-2");
        assert!(ae2.static_design_qualifies());
        let rq = RuntimeQualification::from_static_design(&ae2);
        assert!(rq.static_design_qualifies);
        assert!(!rq.intervention_applied);
        assert!(!rq.intervention_specificity_passed);
        assert!(!rq.positive_control_effect_observed);
        assert!(!rq.sham_behaved_as_expected);
        assert!(!rq.probe_signal_usable);
        assert!(!rq.identity_and_config_match);
        assert!(!rq.qualifies_run(), "no runtime checks confirmed yet");
    }

    #[test]
    fn test_failure_reasons_lists_every_failing_dimension_not_just_first() {
        let rq = RuntimeQualification {
            static_design_qualifies: true,
            intervention_applied: false,
            intervention_specificity_passed: false,
            positive_control_effect_observed: true,
            sham_behaved_as_expected: true,
            probe_signal_usable: true,
            identity_and_config_match: true,
        };
        let reasons = rq.failure_reasons();
        assert_eq!(
            reasons,
            vec![
                QualificationFailure::InterventionNotApplied,
                QualificationFailure::InterventionNotSpecific,
            ]
        );
        assert!(!rq.qualifies_run());
    }

    #[test]
    fn test_fully_passing_runtime_qualification_qualifies() {
        let ae2 = design("AE-2");
        let rq = fully_passing(&ae2);
        assert!(rq.qualifies_run());
        assert!(rq.failure_reasons().is_empty());
    }

    // ── Conformance fixture: GWT-3 fails closed on probe validity alone ──
    // (Minimal-scope demonstration #4 -- this is already fully determined
    // by the STATIC design; no runtime check is even needed to reach
    // Inconclusive for a row that fails static_design_qualifies().)

    #[test]
    fn test_gwt3_static_design_alone_forces_inconclusive() {
        let gwt3 = design("GWT-3");
        assert!(
            !gwt3.static_design_qualifies(),
            "GWT-3 is ExecutionProxy + Unverified"
        );
        // Even with every OTHER runtime field fabricated as passing, the
        // static failure alone must force Inconclusive.
        let rq = RuntimeQualification {
            static_design_qualifies: false,
            intervention_applied: true,
            intervention_specificity_passed: true,
            positive_control_effect_observed: true,
            sham_behaved_as_expected: true,
            probe_signal_usable: true,
            identity_and_config_match: true,
        };
        assert!(!rq.qualifies_run());
        assert_eq!(
            resolve_outcome(&rq, true, true),
            EvidenceOutcome::Inconclusive,
            "an ExecutionProxy probe must stay Inconclusive even if every runtime check happens \
             to pass and the raw scores look like a hit"
        );
    }

    // ── Conformance fixture: control failed at runtime (demonstration #3) ─

    #[test]
    fn test_runtime_control_failure_forces_inconclusive_despite_qualifying_static_design() {
        let ae2 = design("AE-2");
        assert!(ae2.static_design_qualifies());
        let mut rq = fully_passing(&ae2);
        rq.positive_control_effect_observed = false; // the pin didn't actually take effect
        assert!(!rq.qualifies_run());
        assert_eq!(
            rq.failure_reasons(),
            vec![QualificationFailure::PositiveControlEffectNotObserved]
        );
        assert_eq!(
            resolve_outcome(&rq, true, true),
            EvidenceOutcome::Inconclusive,
            "a statically-eligible row must still fail closed when its runtime control check fails"
        );
    }

    // ── Conformance fixture: metric moved but the hook never fired ──────

    #[test]
    fn test_metric_moved_but_intervention_not_applied_is_not_a_qualified_pass() {
        // A fabricated result where the expected metric change is present
        // (indicator_effect_observed=true) but instrumentation shows the
        // mutation hook's counter stayed at zero -- must not be accepted as
        // proof the intended manipulation occurred.
        let ae2 = design("AE-2");
        let mut rq = fully_passing(&ae2);
        rq.intervention_applied = false;
        assert!(!rq.qualifies_run());
        assert_eq!(
            resolve_outcome(&rq, /* indicator_effect_observed = */ true, true),
            EvidenceOutcome::Inconclusive,
            "a coincidental-looking metric movement must not be accepted as a qualified pass \
             when the intervention itself is confirmed not to have executed"
        );
    }

    // ── Conformance fixture: qualified null (demonstration #2) ──────────

    #[test]
    fn test_qualified_probe_with_no_effect_is_not_demonstrated() {
        let ae2 = design("AE-2");
        let rq = fully_passing(&ae2);
        assert!(rq.qualifies_run());
        assert_eq!(
            resolve_outcome(&rq, /* indicator_effect_observed = */ false, false),
            EvidenceOutcome::NotDemonstrated,
            "a qualified probe that genuinely shows no effect is a real null result, not \
             Inconclusive"
        );
    }

    // ── Conformance fixture: qualified positive result (demonstration #1,
    // ── fabricated here -- NOT a claim about a real AE-2 measurement) ───

    #[test]
    fn test_qualified_probe_with_effect_resolves_to_supported() {
        let ae2 = design("AE-2");
        let rq = fully_passing(&ae2);
        assert_eq!(
            resolve_outcome(&rq, true, false),
            EvidenceOutcome::Supported(SupportTier::CausallySupported),
            "fixture only -- proves the mapping logic, not a real empirical finding"
        );
        assert_eq!(
            resolve_outcome(&rq, true, true),
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported),
            "fixture only -- proves the mapping logic, not a real empirical finding"
        );
    }

    // ── Conformance fixture: hard failure on registry mismatch
    // ── (demonstration #6) ───────────────────────────────────────────────

    #[test]
    fn test_identity_check_passes_when_registry_matches() {
        let ae2 = design("AE-2");
        assert!(
            check_identity_against_registry(&ae2, ae2.target_lever, ae2.functional_benchmark)
                .is_ok()
        );
    }

    #[test]
    fn test_identity_check_hard_fails_on_target_lever_mismatch() {
        let ae2 = design("AE-2");
        let err =
            check_identity_against_registry(&ae2, "some_other_lever", ae2.functional_benchmark)
                .unwrap_err();
        assert_eq!(
            err,
            QualificationRunError::RegistryIdentityMismatch {
                indicator: "AE-2",
                field: "target_lever",
                declared: "disable_embodied_cognition".to_string(),
                actual: "some_other_lever".to_string(),
            }
        );
    }

    #[test]
    fn test_identity_check_hard_fails_on_functional_benchmark_mismatch() {
        let ae2 = design("AE-2");
        let err =
            check_identity_against_registry(&ae2, ae2.target_lever, "SomeOtherBenchmark::Fake")
                .unwrap_err();
        assert_eq!(
            err,
            QualificationRunError::RegistryIdentityMismatch {
                indicator: "AE-2",
                field: "functional_benchmark",
                declared: "WorM::SpatialUpdating".to_string(),
                actual: "SomeOtherBenchmark::Fake".to_string(),
            }
        );
    }

    // ── Conformance fixture: dependency-aware reporting for HOT-3/PP-1
    // ── (demonstration #5) — reuses qualification_design.rs's already-
    // ── verified dependency methods; this fixture just proves a report
    // ── layer built on RuntimeQualification would actually check them. ──

    #[test]
    fn test_hot3_pp1_qualified_pair_must_not_be_reported_as_independent() {
        let hot3 = design("HOT-3");
        let pp1 = design("PP-1");
        assert!(hot3.static_design_qualifies());
        assert!(pp1.static_design_qualifies());
        let hot3_rq = fully_passing(&hot3);
        let pp1_rq = fully_passing(&pp1);
        assert!(hot3_rq.qualifies_run());
        assert!(pp1_rq.qualifies_run());
        // Both individually qualify and resolve to Supported -- but a report
        // built on this must still consult fully_independent_of() before
        // presenting them as two confirmations.
        assert_eq!(
            resolve_outcome(&hot3_rq, true, false),
            EvidenceOutcome::Supported(SupportTier::CausallySupported)
        );
        assert_eq!(
            resolve_outcome(&pp1_rq, true, false),
            EvidenceOutcome::Supported(SupportTier::CausallySupported)
        );
        assert!(
            !hot3.fully_independent_of(&pp1),
            "a runner must consult this before reporting HOT-3 and PP-1 as independent \
             corroboration, even when both individually resolve to Supported"
        );
        assert!(hot3.shares_probe_signal(&pp1));
    }
}
