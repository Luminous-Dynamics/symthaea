// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adversarial integration crucible for the safety-evidence runtime stack.
//!
//! The fixture begins in a genuinely `Ready` state, then perturbs one reviewed
//! assumption at a time. The crucible is evidence about assurance behavior only;
//! it never grants physical authority.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_evidence_deployment_scope::{
    DeploymentEvidenceContext, DeploymentScopedSafetyReceipt,
    assess_deployment_scoped_safety_case, bind_receipt_to_deployment,
};
use symthaea_evidence_lifecycle::{
    EvidenceLifecycleEvent, EvidenceLifecycleEventKind, ScopedSafetyEvidenceReceipt,
};
use symthaea_evidence_quarantine::{
    EvidenceQuarantineDirective, EvidenceQuarantineResolution,
    QuarantineResolutionDisposition, QuarantineSelector, assess_quarantined_safety_case,
};
use symthaea_evidence_time_assurance::{
    EvidenceTimeGuard, EvidenceTimePolicy, EvidenceTimeSample, EvidenceTimeStatus,
    assess_with_trusted_time,
};
use symthaea_formal_safety::{
    EvidenceKind, ProofObligation, SafetyCase, SafetyEvidenceReceipt, StrictSafetyCaseStatus,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceRuntimeCrucibleStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrucibleExpectedOutcome {
    Safety(StrictSafetyCaseStatus),
    Time(EvidenceTimeStatus),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrucibleObservedOutcome {
    Safety(StrictSafetyCaseStatus),
    Time(EvidenceTimeStatus),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRuntimeScenarioResult {
    pub scenario_id: String,
    pub expected: CrucibleExpectedOutcome,
    pub observed: CrucibleObservedOutcome,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRuntimeCrucibleReport {
    pub schema_version: String,
    pub status: EvidenceRuntimeCrucibleStatus,
    pub scenarios: Vec<EvidenceRuntimeScenarioResult>,
}

impl EvidenceRuntimeCrucibleReport {
    pub fn passed(&self) -> bool {
        self.status == EvidenceRuntimeCrucibleStatus::Pass
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

fn fixture_case() -> SafetyCase {
    let mut case = SafetyCase::new("evidence-runtime-crucible-fixture");
    case.add_obligation(
        ProofObligation::new(
            "synthetic reviewed claim used only to exercise runtime evidence semantics",
            EvidenceKind::Test,
        )
        .discharge("fixture:reviewed-workflow-discharge"),
    );
    case
}

fn context(
    configuration: &str,
    model: &str,
    calibration: &str,
) -> DeploymentEvidenceContext {
    DeploymentEvidenceContext {
        schema_version: "1".into(),
        deployment_id: "crucible-node-1".into(),
        configuration_digest: configuration.into(),
        model_manifest_digest: Some(model.into()),
        calibration_manifest_digest: Some(calibration.into()),
        evidence_refs: vec!["fixture:deployment-context".into()],
    }
}

fn baseline_context() -> DeploymentEvidenceContext {
    context(
        "blake3:config-v1",
        "blake3:model-v1",
        "blake3:calibration-v1",
    )
}

fn scoped_receipt(
    case: &SafetyCase,
    id: &str,
    verifier: &str,
    verified_at_ms: u64,
    valid_until_ms: u64,
) -> ScopedSafetyEvidenceReceipt {
    let obligation = &case.obligations[0];
    ScopedSafetyEvidenceReceipt {
        receipt: SafetyEvidenceReceipt {
            receipt_id: id.into(),
            obligation_key: obligation.stable_key(),
            evidence_kind: obligation.expected_evidence,
            evidence_ref: format!("fixture:artifact:{id}"),
            evidence_digest: format!("blake3:{id}"),
            verifier_ref: verifier.into(),
            verified_at_ms,
        },
        contract_digest: case.contract_digest(),
        valid_from_ms: verified_at_ms,
        valid_until_ms,
        applicability_refs: vec!["fixture:applicability-review".into()],
    }
}

fn bound_receipt(
    case: &SafetyCase,
    id: &str,
    verifier: &str,
    verified_at_ms: u64,
    valid_until_ms: u64,
    context: &DeploymentEvidenceContext,
) -> DeploymentScopedSafetyReceipt {
    bind_receipt_to_deployment(
        scoped_receipt(case, id, verifier, verified_at_ms, valid_until_ms),
        context,
        format!("fixture:scope-binding:{id}"),
    )
    .expect("fixture receipt must bind to reviewed context")
}

fn time_policy() -> EvidenceTimePolicy {
    EvidenceTimePolicy {
        schema_version: "1".into(),
        policy_id: "crucible-time-v1".into(),
        expected_clock_source: "ptp-grandmaster-a".into(),
        expected_clock_domain: "domain-7".into(),
        expected_epoch_ref: "epoch:crucible-1".into(),
        maximum_clock_uncertainty_ms: 100,
        maximum_wall_clock_backstep_ms: 5,
    }
}

fn time_sample(id: &str, wall: u64, monotonic: u64, uncertainty: u64) -> EvidenceTimeSample {
    EvidenceTimeSample {
        sample_id: id.into(),
        wall_time_ms: wall,
        monotonic_time_ms: monotonic,
        clock_source: "ptp-grandmaster-a".into(),
        clock_domain: "domain-7".into(),
        epoch_ref: "epoch:crucible-1".into(),
        clock_uncertainty_ms: uncertainty,
        evidence_refs: vec![format!("fixture:time:{id}")],
    }
}

fn push_safety(
    scenarios: &mut Vec<EvidenceRuntimeScenarioResult>,
    id: &str,
    expected: StrictSafetyCaseStatus,
    observed: StrictSafetyCaseStatus,
) {
    scenarios.push(EvidenceRuntimeScenarioResult {
        scenario_id: id.into(),
        expected: CrucibleExpectedOutcome::Safety(expected),
        observed: CrucibleObservedOutcome::Safety(observed),
        passed: expected == observed,
    });
}

fn push_time(
    scenarios: &mut Vec<EvidenceRuntimeScenarioResult>,
    id: &str,
    expected: EvidenceTimeStatus,
    observed: EvidenceTimeStatus,
) {
    scenarios.push(EvidenceRuntimeScenarioResult {
        scenario_id: id.into(),
        expected: CrucibleExpectedOutcome::Time(expected),
        observed: CrucibleObservedOutcome::Time(observed),
        passed: expected == observed,
    });
}

/// Run the full evidence-runtime assurance crucible.
///
/// The baseline must be genuinely `Ready`. Every destructive perturbation must
/// remove readiness, while reviewed replacement/remediation paths must restore it.
pub fn run_evidence_runtime_crucible() -> EvidenceRuntimeCrucibleReport {
    let case = fixture_case();
    let baseline_context = baseline_context();
    let baseline = bound_receipt(
        &case,
        "baseline",
        "verifier:safety-a",
        100,
        1_500,
        &baseline_context,
    );
    let mut scenarios = Vec::new();

    // Establish the positive control first: this fixture is actually Ready.
    let baseline_report = assess_quarantined_safety_case(
        &case,
        &baseline_context,
        std::slice::from_ref(&baseline),
        &[],
        &[],
        &[],
        1_000,
    );
    push_safety(
        &mut scenarios,
        "baseline-ready",
        StrictSafetyCaseStatus::Ready,
        baseline_report.status,
    );

    let expired = assess_quarantined_safety_case(
        &case,
        &baseline_context,
        std::slice::from_ref(&baseline),
        &[],
        &[],
        &[],
        2_000,
    );
    push_safety(
        &mut scenarios,
        "expired-evidence-blocks",
        StrictSafetyCaseStatus::Blocked,
        expired.status,
    );

    let revoked_event = EvidenceLifecycleEvent {
        event_id: "event:revoke-baseline".into(),
        receipt_id: "baseline".into(),
        event_at_ms: 900,
        kind: EvidenceLifecycleEventKind::Revoked {
            reason_ref: "fixture:revocation-review".into(),
        },
    };
    let revoked = assess_quarantined_safety_case(
        &case,
        &baseline_context,
        std::slice::from_ref(&baseline),
        &[revoked_event],
        &[],
        &[],
        1_000,
    );
    push_safety(
        &mut scenarios,
        "revocation-blocks",
        StrictSafetyCaseStatus::Blocked,
        revoked.status,
    );

    let contradiction_event = EvidenceLifecycleEvent {
        event_id: "event:contradict-baseline".into(),
        receipt_id: "baseline".into(),
        event_at_ms: 900,
        kind: EvidenceLifecycleEventKind::Contradicted {
            contradiction_id: "contradiction:baseline".into(),
            contradiction_ref: "fixture:contradictory-field-evidence".into(),
        },
    };
    let contradicted = assess_quarantined_safety_case(
        &case,
        &baseline_context,
        std::slice::from_ref(&baseline),
        &[contradiction_event],
        &[],
        &[],
        1_000,
    );
    push_safety(
        &mut scenarios,
        "unresolved-contradiction-blocks",
        StrictSafetyCaseStatus::Blocked,
        contradicted.status,
    );

    let configuration_drift = context(
        "blake3:config-v2",
        "blake3:model-v1",
        "blake3:calibration-v1",
    );
    let drifted = assess_deployment_scoped_safety_case(
        &case,
        &configuration_drift,
        std::slice::from_ref(&baseline),
        &[],
        1_000,
    );
    push_safety(
        &mut scenarios,
        "configuration-drift-blocks",
        StrictSafetyCaseStatus::Blocked,
        drifted.status,
    );

    let model_drift = context(
        "blake3:config-v1",
        "blake3:model-v2",
        "blake3:calibration-v1",
    );
    let drifted = assess_deployment_scoped_safety_case(
        &case,
        &model_drift,
        std::slice::from_ref(&baseline),
        &[],
        1_000,
    );
    push_safety(
        &mut scenarios,
        "model-manifest-drift-blocks",
        StrictSafetyCaseStatus::Blocked,
        drifted.status,
    );

    let calibration_drift = context(
        "blake3:config-v1",
        "blake3:model-v1",
        "blake3:calibration-v2",
    );
    let drifted = assess_deployment_scoped_safety_case(
        &case,
        &calibration_drift,
        std::slice::from_ref(&baseline),
        &[],
        1_000,
    );
    push_safety(
        &mut scenarios,
        "calibration-manifest-drift-blocks",
        StrictSafetyCaseStatus::Blocked,
        drifted.status,
    );

    let verifier_quarantine = EvidenceQuarantineDirective {
        directive_id: "quarantine:verifier-a".into(),
        selector: QuarantineSelector::VerifierRef("verifier:safety-a".into()),
        effective_from_ms: 900,
        reason_ref: "fixture:verifier-integrity-review".into(),
        evidence_refs: vec!["fixture:quarantine-evidence".into()],
    };
    let quarantined = assess_quarantined_safety_case(
        &case,
        &baseline_context,
        std::slice::from_ref(&baseline),
        &[],
        std::slice::from_ref(&verifier_quarantine),
        &[],
        1_000,
    );
    push_safety(
        &mut scenarios,
        "verifier-quarantine-blocks",
        StrictSafetyCaseStatus::Blocked,
        quarantined.status,
    );

    let expiring = bound_receipt(
        &case,
        "expiring",
        "verifier:safety-a",
        100,
        1_050,
        &baseline_context,
    );
    let mut guard = EvidenceTimeGuard::new(time_policy()).expect("valid time policy");
    let trusted_report = guard.observe(time_sample("time-straddle", 1_000, 1_000, 100));
    let trusted_time = trusted_report
        .trusted_time
        .as_ref()
        .expect("time fixture must remain trusted");
    let time_robust = assess_with_trusted_time(
        &case,
        &baseline_context,
        &[expiring],
        &[],
        &[],
        &[],
        trusted_time,
    );
    push_safety(
        &mut scenarios,
        "clock-uncertainty-straddling-expiry-blocks",
        StrictSafetyCaseStatus::Blocked,
        time_robust.status,
    );

    let mut rollback_guard = EvidenceTimeGuard::new(time_policy()).expect("valid time policy");
    let first = rollback_guard.observe(time_sample("clock-1", 1_000, 1_000, 10));
    push_time(
        &mut scenarios,
        "trusted-clock-positive-control",
        EvidenceTimeStatus::Trusted,
        first.status,
    );
    let rollback = rollback_guard.observe(time_sample("clock-2", 900, 2_000, 10));
    push_time(
        &mut scenarios,
        "wall-clock-rollback-latches-untrusted",
        EvidenceTimeStatus::Untrusted,
        rollback.status,
    );

    let replacement_context = configuration_drift;
    let replacement = bound_receipt(
        &case,
        "replacement-config-v2",
        "verifier:safety-b",
        950,
        2_000,
        &replacement_context,
    );
    let restored = assess_deployment_scoped_safety_case(
        &case,
        &replacement_context,
        &[baseline.clone(), replacement],
        &[],
        1_000,
    );
    push_safety(
        &mut scenarios,
        "replacement-evidence-restores-after-config-drift",
        StrictSafetyCaseStatus::Ready,
        restored.status,
    );

    let post_remediation = bound_receipt(
        &case,
        "post-remediation",
        "verifier:safety-a",
        950,
        2_000,
        &baseline_context,
    );
    let replacement_resolution = EvidenceQuarantineResolution {
        resolution_id: "resolution:verifier-a".into(),
        directive_id: verifier_quarantine.directive_id.clone(),
        resolved_at_ms: 900,
        disposition: QuarantineResolutionDisposition::RequireReplacement,
        resolution_ref: "fixture:verifier-remediation".into(),
        evidence_refs: vec!["fixture:remediation-evidence".into()],
    };
    let remediated = assess_quarantined_safety_case(
        &case,
        &baseline_context,
        &[baseline.clone(), post_remediation],
        &[],
        &[verifier_quarantine],
        &[replacement_resolution],
        1_000,
    );
    push_safety(
        &mut scenarios,
        "post-remediation-replacement-restores-quarantined-evidence",
        StrictSafetyCaseStatus::Ready,
        remediated.status,
    );

    let contradicted_old = bound_receipt(
        &case,
        "contradicted-old",
        "verifier:safety-a",
        100,
        2_000,
        &baseline_context,
    );
    let contradiction_replacement = bound_receipt(
        &case,
        "contradiction-replacement",
        "verifier:safety-b",
        800,
        2_000,
        &baseline_context,
    );
    let contradiction = EvidenceLifecycleEvent {
        event_id: "event:contradict-old".into(),
        receipt_id: "contradicted-old".into(),
        event_at_ms: 500,
        kind: EvidenceLifecycleEventKind::Contradicted {
            contradiction_id: "contradiction:old".into(),
            contradiction_ref: "fixture:new-field-evidence".into(),
        },
    };
    let contradiction_resolution = EvidenceLifecycleEvent {
        event_id: "event:resolve-old".into(),
        receipt_id: "contradicted-old".into(),
        event_at_ms: 700,
        kind: EvidenceLifecycleEventKind::ContradictionResolved {
            contradiction_id: "contradiction:old".into(),
            resolution_ref: "fixture:contradiction-review".into(),
        },
    };
    let resolved_with_replacement = assess_quarantined_safety_case(
        &case,
        &baseline_context,
        &[contradicted_old, contradiction_replacement],
        &[contradiction, contradiction_resolution],
        &[],
        &[],
        1_000,
    );
    push_safety(
        &mut scenarios,
        "reviewed-contradiction-plus-replacement-restores-readiness",
        StrictSafetyCaseStatus::Ready,
        resolved_with_replacement.status,
    );

    let status = if scenarios.iter().all(|scenario| scenario.passed) {
        EvidenceRuntimeCrucibleStatus::Pass
    } else {
        EvidenceRuntimeCrucibleStatus::Fail
    };

    EvidenceRuntimeCrucibleReport {
        schema_version: "1".into(),
        status,
        scenarios,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_runtime_evidence_crucible_passes() {
        let report = run_evidence_runtime_crucible();
        assert_eq!(report.status, EvidenceRuntimeCrucibleStatus::Pass);
        assert!(report.scenarios.len() >= 12);
        assert!(report.scenarios.iter().all(|scenario| scenario.passed));
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn destructive_and_recovery_paths_are_both_present() {
        let report = run_evidence_runtime_crucible();
        assert!(report
            .scenarios
            .iter()
            .any(|scenario| scenario.scenario_id == "configuration-drift-blocks"));
        assert!(report.scenarios.iter().any(|scenario| {
            scenario.scenario_id == "replacement-evidence-restores-after-config-drift"
        }));
        assert!(report.scenarios.iter().any(|scenario| {
            scenario.scenario_id == "clock-uncertainty-straddling-expiry-blocks"
        }));
        assert!(report.scenarios.iter().any(|scenario| {
            scenario.scenario_id == "wall-clock-rollback-latches-untrusted"
        }));
    }
}
