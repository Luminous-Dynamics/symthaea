use anyhow::bail;
use serde::Serialize;

use crate::autonomous_cargo_adapter_state::AutonomousAdapterRunReport;
use crate::autonomous_cargo_effect_binding::CommitmentBoundAutonomousRunReport;
use crate::autonomous_cargo_freshness_binding::FreshnessBoundAutonomousRunReport;
use crate::autonomous_cargo_persistence_binding::{
    CanonicalAdmissionPersistenceEvidence, ExpectedAdmissionPersistenceBinding,
    PersistenceBoundAutonomousRunReport,
};
use crate::cargo_adapter_commit::{
    AttemptCommitOutcome, CapturedSourcePostflightBoundary, commit_pipeline_attempt,
};
use crate::cargo_adapter_postflight::{
    CargoObservationVerificationBoundary, continue_with_observation,
};
use crate::cargo_adapter_state::AdapterRunReport;
use crate::cargo_execution_attempt::CargoExecutionAttemptResult;
use crate::cargo_execution_contract::{CargoExecutionIntent, validate_intent};

/// Outer composition result for the full autonomous pre-spawn lineage joined to
/// terminal-sensitive canonical Cargo attempt evidence.
///
/// The original autonomous report is retained intact so no established
/// eligibility/admission/persistence/freshness evidence is lost by projection.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum AutonomousAttemptCommitOutcome {
    NotCommitted {
        intent_id: String,
        reason: String,
        report: PersistenceBoundAutonomousRunReport,
    },
    Committed {
        intent_id: String,
        report: PersistenceBoundAutonomousRunReport,
        result: CargoExecutionAttemptResult,
    },
}

#[derive(Debug, Clone)]
enum BackendDisposition {
    NotReached { reason: String },
    Reached { adapter: AdapterRunReport },
}

/// Join the complete autonomous Cargo pre-spawn report to the existing
/// terminal-sensitive observation + canonical attempt-result machinery.
///
/// No Cargo observation verifier or postflight finalizer is reachable unless the
/// nested autonomous state actually reached `AdapterRunReport::BackendEntered`
/// and every accepted outer lineage is cross-bound to that exact backend report.
pub(crate) fn commit_persistence_bound_autonomous_attempt<V, F>(
    mut intent: CargoExecutionIntent,
    mut expected_persistence: ExpectedAdmissionPersistenceBinding,
    report: PersistenceBoundAutonomousRunReport,
    verifier: &mut V,
    finalizer: &mut F,
) -> anyhow::Result<AutonomousAttemptCommitOutcome>
where
    V: CargoObservationVerificationBoundary,
    F: CapturedSourcePostflightBoundary,
{
    validate_intent(&mut intent)?;
    normalize_digest(
        "expected_persistence.authority_snapshot_digest",
        &mut expected_persistence.authority_snapshot_digest,
    )?;
    normalize_digest(
        "expected_persistence.required_persistence_profile_id",
        &mut expected_persistence.required_persistence_profile_id,
    )?;

    let intent_id = intent.intent_id.clone();
    let report_intent = persistence_report_intent_id(&report);
    require_canonical_digest("autonomous_report.intent_id", report_intent)?;
    require_same_intent("autonomous persistence report", report_intent, &intent_id)?;

    match classify_backend_path(&intent, &expected_persistence, &report)? {
        BackendDisposition::NotReached { reason } => Ok(
            AutonomousAttemptCommitOutcome::NotCommitted {
                intent_id,
                reason,
                report,
            },
        ),
        BackendDisposition::Reached { adapter } => {
            let pipeline = continue_with_observation(intent.clone(), adapter, verifier)?;
            match commit_pipeline_attempt(intent, pipeline, finalizer)? {
                AttemptCommitOutcome::Committed { result } => {
                    Ok(AutonomousAttemptCommitOutcome::Committed {
                        intent_id,
                        report,
                        result,
                    })
                }
                AttemptCommitOutcome::NotCommitted { reason, .. } => bail!(
                    "backend-entered autonomous lineage unexpectedly produced no canonical attempt: {reason}"
                ),
            }
        }
    }
}

fn classify_backend_path(
    intent: &CargoExecutionIntent,
    expected_persistence: &ExpectedAdmissionPersistenceBinding,
    report: &PersistenceBoundAutonomousRunReport,
) -> anyhow::Result<BackendDisposition> {
    match report {
        PersistenceBoundAutonomousRunReport::PersistenceExpectationRejectedBeforeEligibility {
            intent_id,
            ..
        } => {
            require_same_intent("persistence expectation rejection", intent_id, &intent.intent_id)?;
            Ok(not_reached("persistence expectation rejected before eligibility"))
        }
        PersistenceBoundAutonomousRunReport::PersistenceBindingRejected {
            intent_id,
            inner,
            ..
        } => {
            require_same_intent("persistence binding rejection", intent_id, &intent.intent_id)?;
            ensure_no_backend_in_freshness(inner)?;
            Ok(not_reached("required persistence profile was not satisfied"))
        }
        PersistenceBoundAutonomousRunReport::AfterPersistenceBinding {
            intent_id,
            accepted_persistence,
            inner,
        } => {
            require_same_intent("persistence binding", intent_id, &intent.intent_id)?;
            classify_freshness_path(
                intent,
                expected_persistence,
                accepted_persistence.as_ref(),
                inner,
            )
        }
    }
}

fn classify_freshness_path(
    intent: &CargoExecutionIntent,
    expected_persistence: &ExpectedAdmissionPersistenceBinding,
    accepted_persistence: Option<&CanonicalAdmissionPersistenceEvidence>,
    report: &FreshnessBoundAutonomousRunReport,
) -> anyhow::Result<BackendDisposition> {
    match report {
        FreshnessBoundAutonomousRunReport::FreshnessBindingRejected {
            intent_id,
            inner,
            ..
        } => {
            require_same_intent("freshness binding rejection", intent_id, &intent.intent_id)?;
            ensure_no_backend_in_commitment(inner)?;
            Ok(not_reached("canonical freshness v2 targeted another prepared subject"))
        }
        FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
            intent_id,
            accepted_freshness,
            inner,
        } => {
            require_same_intent("freshness binding", intent_id, &intent.intent_id)?;
            classify_commitment_path(
                intent,
                expected_persistence,
                accepted_persistence,
                accepted_freshness.as_ref(),
                inner,
            )
        }
    }
}

fn classify_commitment_path(
    intent: &CargoExecutionIntent,
    expected_persistence: &ExpectedAdmissionPersistenceBinding,
    accepted_persistence: Option<&CanonicalAdmissionPersistenceEvidence>,
    accepted_freshness: Option<&crate::autonomous_cargo_freshness_binding::CanonicalFreshnessV2Evidence>,
    report: &CommitmentBoundAutonomousRunReport,
) -> anyhow::Result<BackendDisposition> {
    match report {
        CommitmentBoundAutonomousRunReport::ExpectationRejectedBeforeEligibility {
            intent_id,
            ..
        } => {
            require_same_intent("effect expectation rejection", intent_id, &intent.intent_id)?;
            Ok(not_reached("effect-admission expectation rejected before eligibility"))
        }
        CommitmentBoundAutonomousRunReport::EligibilityBindingRejected {
            intent_id,
            ..
        } => {
            require_same_intent("eligibility binding rejection", intent_id, &intent.intent_id)?;
            Ok(not_reached("eligibility lineage did not match host expectation"))
        }
        CommitmentBoundAutonomousRunReport::AdmissionBindingRejected {
            intent_id,
            ..
        } => {
            require_same_intent("admission binding rejection", intent_id, &intent.intent_id)?;
            Ok(not_reached("effect-admission commitment did not match host expectation"))
        }
        CommitmentBoundAutonomousRunReport::AfterCommitmentBinding {
            intent_id,
            expected,
            accepted_admission,
            inner,
        } => {
            require_same_intent("commitment binding", intent_id, &intent.intent_id)?;
            validate_expected_effect(intent, expected_persistence, expected)?;
            classify_eligibility_path(
                intent,
                expected_persistence,
                accepted_persistence,
                accepted_freshness,
                expected,
                accepted_admission.as_ref(),
                inner,
            )
        }
    }
}

fn validate_expected_effect(
    intent: &CargoExecutionIntent,
    expected_persistence: &ExpectedAdmissionPersistenceBinding,
    expected: &crate::autonomous_cargo_effect_binding::ExpectedEffectAdmissionBinding,
) -> anyhow::Result<()> {
    for (name, value) in [
        ("expected.execution_intent_id", expected.execution_intent_id.as_str()),
        ("expected.eligibility_id", expected.eligibility_id.as_str()),
        ("expected.prepared_intent_id", expected.prepared_intent_id.as_str()),
        ("expected.action_binding", expected.action_binding.as_str()),
        (
            "expected.authority_snapshot_digest",
            expected.authority_snapshot_digest.as_str(),
        ),
        (
            "expected.adapter_semantics_digest",
            expected.adapter_semantics_digest.as_str(),
        ),
        (
            "expected.effect_admission_commitment_digest",
            expected.effect_admission_commitment_digest.as_str(),
        ),
    ] {
        require_canonical_digest(name, value)?;
    }
    require_same_intent(
        "effect-admission expectation",
        &expected.execution_intent_id,
        &intent.intent_id,
    )?;
    if expected.adapter_semantics_digest != intent.adapter_semantics_digest {
        bail!("effect-admission expectation targets another adapter-semantics subject");
    }
    if expected.authority_snapshot_digest != expected_persistence.authority_snapshot_digest {
        bail!("persistence requirement belongs to another authority snapshot");
    }
    Ok(())
}

fn classify_eligibility_path(
    intent: &CargoExecutionIntent,
    expected_persistence: &ExpectedAdmissionPersistenceBinding,
    accepted_persistence: Option<&CanonicalAdmissionPersistenceEvidence>,
    accepted_freshness: Option<&crate::autonomous_cargo_freshness_binding::CanonicalFreshnessV2Evidence>,
    expected: &crate::autonomous_cargo_effect_binding::ExpectedEffectAdmissionBinding,
    accepted_admission: Option<&crate::autonomous_cargo_effect_binding::CommitmentAwareEffectAdmission>,
    report: &AutonomousAdapterRunReport,
) -> anyhow::Result<BackendDisposition> {
    match report {
        AutonomousAdapterRunReport::EligibilityRejected { intent_id, .. } => {
            require_same_intent("eligibility rejection", intent_id, &intent.intent_id)?;
            Ok(not_reached("autonomous eligibility was denied"))
        }
        AutonomousAdapterRunReport::EligibilityEvidenceRejected { intent_id, .. } => {
            require_same_intent("eligibility evidence rejection", intent_id, &intent.intent_id)?;
            Ok(not_reached("autonomous eligibility targeted another execution intent"))
        }
        AutonomousAdapterRunReport::AfterEligibility {
            intent_id,
            eligibility_id,
            prepared_intent_id,
            inner,
        } => {
            require_same_intent("eligibility binding", intent_id, &intent.intent_id)?;
            require_canonical_digest("eligibility_id", eligibility_id)?;
            require_canonical_digest("prepared_intent_id", prepared_intent_id)?;
            if eligibility_id != &expected.eligibility_id {
                bail!("accepted eligibility id does not match effect-admission expectation");
            }
            if prepared_intent_id != &expected.prepared_intent_id {
                bail!("accepted prepared-intent id does not match effect-admission expectation");
            }

            match inner {
                AdapterRunReport::BackendEntered {
                    intent_id: adapter_intent_id,
                    effect_admission_digest,
                    persistence_ack_digest,
                    pre_spawn_freshness_digest,
                    ..
                } => {
                    require_same_intent(
                        "backend adapter report",
                        adapter_intent_id,
                        &intent.intent_id,
                    )?;
                    let admission = accepted_admission.ok_or_else(|| {
                        anyhow::anyhow!(
                            "backend-entered lineage is missing accepted AI-Assurance admission evidence"
                        )
                    })?;
                    let persistence = accepted_persistence.ok_or_else(|| {
                        anyhow::anyhow!(
                            "backend-entered lineage is missing accepted persistence evidence"
                        )
                    })?;
                    let freshness = accepted_freshness.ok_or_else(|| {
                        anyhow::anyhow!(
                            "backend-entered lineage is missing accepted freshness-v2 evidence"
                        )
                    })?;

                    validate_backend_join(
                        intent,
                        expected_persistence,
                        expected,
                        admission,
                        persistence,
                        freshness,
                        effect_admission_digest,
                        persistence_ack_digest,
                        pre_spawn_freshness_digest,
                    )?;
                    Ok(BackendDisposition::Reached {
                        adapter: inner.clone(),
                    })
                }
                AdapterRunReport::RejectedBeforeEffect { .. } => {
                    Ok(not_reached("inner adapter rejected before effect entry"))
                }
                AdapterRunReport::EffectRejected { .. } => {
                    Ok(not_reached("effect entry was denied"))
                }
                AdapterRunReport::AdmissionPersistenceRejected { .. } => {
                    Ok(not_reached("admission persistence rejected before backend entry"))
                }
                AdapterRunReport::FreshnessRejected { .. } => {
                    Ok(not_reached("pre-spawn freshness rejected before backend entry"))
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_backend_join(
    intent: &CargoExecutionIntent,
    expected_persistence: &ExpectedAdmissionPersistenceBinding,
    expected: &crate::autonomous_cargo_effect_binding::ExpectedEffectAdmissionBinding,
    admission: &crate::autonomous_cargo_effect_binding::CommitmentAwareEffectAdmission,
    persistence: &CanonicalAdmissionPersistenceEvidence,
    freshness: &crate::autonomous_cargo_freshness_binding::CanonicalFreshnessV2Evidence,
    effect_admission_digest: &str,
    persistence_ack_digest: &str,
    pre_spawn_freshness_digest: &str,
) -> anyhow::Result<()> {
    for (name, value) in [
        ("admission.receipt_digest", admission.receipt_digest.as_str()),
        ("admission.action_binding", admission.action_binding.as_str()),
        (
            "admission.authority_snapshot_digest",
            admission.authority_snapshot_digest.as_str(),
        ),
        (
            "admission.adapter_semantics_digest",
            admission.adapter_semantics_digest.as_str(),
        ),
        (
            "admission.effect_admission_commitment_digest",
            admission.effect_admission_commitment_digest.as_str(),
        ),
        (
            "persistence.persistence_ack_digest",
            persistence.persistence_ack_digest.as_str(),
        ),
        (
            "persistence.persistence_profile_id",
            persistence.persistence_profile_id.as_str(),
        ),
        (
            "freshness.freshness_receipt_id",
            freshness.freshness_receipt_id.as_str(),
        ),
        (
            "freshness.prepared_intent_id",
            freshness.prepared_intent_id.as_str(),
        ),
        (
            "freshness.runtime_binding_id",
            freshness.runtime_binding_id.as_str(),
        ),
        (
            "freshness.materialization_receipt_id",
            freshness.materialization_receipt_id.as_str(),
        ),
        (
            "freshness.tool_attestation_id",
            freshness.tool_attestation_id.as_str(),
        ),
        (
            "freshness.git_worktree_state_id",
            freshness.git_worktree_state_id.as_str(),
        ),
        ("effect_admission_digest", effect_admission_digest),
        ("persistence_ack_digest", persistence_ack_digest),
        ("pre_spawn_freshness_digest", pre_spawn_freshness_digest),
    ] {
        require_canonical_digest(name, value)?;
    }
    require_optional_canonical_digest(
        "persistence.evidence_root_digest",
        persistence.evidence_root_digest.as_deref(),
    )?;
    require_optional_canonical_digest(
        "persistence.anti_rollback_evidence_digest",
        persistence.anti_rollback_evidence_digest.as_deref(),
    )?;
    require_canonical_text(
        "persistence.persistence_record_id",
        &persistence.persistence_record_id,
    )?;
    require_canonical_text("persistence.control_plane_id", &persistence.control_plane_id)?;

    if admission.receipt_digest != effect_admission_digest {
        bail!("inner backend report carries another effect-admission receipt");
    }
    if admission.action_binding != expected.action_binding
        || admission.authority_snapshot_digest != expected.authority_snapshot_digest
        || admission.adapter_semantics_digest != expected.adapter_semantics_digest
        || admission.effect_admission_commitment_digest
            != expected.effect_admission_commitment_digest
    {
        bail!("accepted AI-Assurance admission no longer matches host expectation");
    }
    if persistence.persistence_profile_id != expected_persistence.required_persistence_profile_id {
        bail!("accepted persistence evidence does not satisfy the required profile");
    }
    if persistence.persistence_ack_digest != persistence_ack_digest {
        bail!("inner backend report carries another persistence acknowledgement");
    }
    if freshness.prepared_intent_id != expected.prepared_intent_id {
        bail!("accepted freshness evidence targets another prepared intent");
    }
    if freshness.freshness_receipt_id != pre_spawn_freshness_digest {
        bail!("inner backend report carries another freshness-v2 receipt");
    }
    if intent.git_worktree_state_before.as_deref()
        != Some(freshness.git_worktree_state_id.as_str())
    {
        bail!("accepted freshness Git state does not match frozen execution intent");
    }
    Ok(())
}

fn ensure_no_backend_in_freshness(report: &FreshnessBoundAutonomousRunReport) -> anyhow::Result<()> {
    if freshness_contains_backend(report) {
        bail!("outer persistence rejection contains an impossible backend-entered inner report");
    }
    Ok(())
}

fn ensure_no_backend_in_commitment(
    report: &CommitmentBoundAutonomousRunReport,
) -> anyhow::Result<()> {
    if commitment_contains_backend(report) {
        bail!("outer freshness rejection contains an impossible backend-entered inner report");
    }
    Ok(())
}

fn freshness_contains_backend(report: &FreshnessBoundAutonomousRunReport) -> bool {
    match report {
        FreshnessBoundAutonomousRunReport::FreshnessBindingRejected { inner, .. }
        | FreshnessBoundAutonomousRunReport::AfterFreshnessBinding { inner, .. } => {
            commitment_contains_backend(inner)
        }
    }
}

fn commitment_contains_backend(report: &CommitmentBoundAutonomousRunReport) -> bool {
    match report {
        CommitmentBoundAutonomousRunReport::AfterCommitmentBinding { inner, .. } => {
            autonomous_contains_backend(inner)
        }
        CommitmentBoundAutonomousRunReport::ExpectationRejectedBeforeEligibility { .. }
        | CommitmentBoundAutonomousRunReport::EligibilityBindingRejected { .. }
        | CommitmentBoundAutonomousRunReport::AdmissionBindingRejected { .. } => false,
    }
}

fn autonomous_contains_backend(report: &AutonomousAdapterRunReport) -> bool {
    matches!(
        report,
        AutonomousAdapterRunReport::AfterEligibility {
            inner: AdapterRunReport::BackendEntered { .. },
            ..
        }
    )
}

fn persistence_report_intent_id(report: &PersistenceBoundAutonomousRunReport) -> &str {
    match report {
        PersistenceBoundAutonomousRunReport::PersistenceExpectationRejectedBeforeEligibility {
            intent_id,
            ..
        }
        | PersistenceBoundAutonomousRunReport::PersistenceBindingRejected {
            intent_id,
            ..
        }
        | PersistenceBoundAutonomousRunReport::AfterPersistenceBinding { intent_id, .. } => intent_id,
    }
}

fn require_same_intent(name: &str, actual: &str, expected: &str) -> anyhow::Result<()> {
    require_canonical_digest(name, actual)?;
    if actual != expected {
        bail!("{name} intent {actual} does not match frozen intent {expected}");
    }
    Ok(())
}

fn not_reached(reason: &str) -> BackendDisposition {
    BackendDisposition::NotReached {
        reason: reason.into(),
    }
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character hex digest");
    }
    value.make_ascii_lowercase();
    Ok(())
}

fn require_canonical_digest(name: &str, value: &str) -> anyhow::Result<()> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{name} must be a canonical lowercase 64-character hex digest");
    }
    Ok(())
}

fn require_optional_canonical_digest(name: &str, value: Option<&str>) -> anyhow::Result<()> {
    if let Some(value) = value {
        require_canonical_digest(name, value)?;
    }
    Ok(())
}

fn require_canonical_text(name: &str, value: &str) -> anyhow::Result<()> {
    if value.is_empty() || value.trim() != value {
        bail!("{name} must be non-empty canonical text without surrounding whitespace");
    }
    if value.bytes().any(|byte| matches!(byte, b'\r' | b'\n' | 0)) {
        bail!("{name} must not contain line separators or NUL");
    }
    if value.len() > 256 {
        bail!("{name} exceeds the 256-byte deterministic-model limit");
    }
    Ok(())
}
