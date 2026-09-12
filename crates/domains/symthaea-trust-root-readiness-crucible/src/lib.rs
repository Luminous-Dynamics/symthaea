// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end adversarial qualification of trust-root-backed readiness.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_assurance_policy_lineage::SignedAssurancePolicyRevision;
use symthaea_assurance_policy_lineage_anchor::PolicyLineageAnchor;
use symthaea_assurance_policy_manifest::{
    AssurancePolicyBinding, AssurancePolicyKind, AssurancePolicyManifest,
    ManifestSignatureVerificationReceipt, PolicyScopedSafetyReceipt,
    bind_receipt_to_assurance_manifest,
};
use symthaea_assurance_signing_authority::{
    ManifestSigningAuthorityPolicy, SigningAuthorityGovernance,
};
use symthaea_assurance_trust_store::{
    TrustStoreBackendKind, TrustStoreCheckpoint, TrustStoreProfile,
};
use symthaea_assurance_trust_store_attestation::CheckpointAttestationVerificationReceipt;
use symthaea_assurance_trust_store_recovery_ledger::RecoveryAcceptanceRecord;
use symthaea_evidence_atomic_coverage::{
    AtomicCoveragePolicy, AtomicEvidenceFacet, FacetEvidenceBinding,
};
use symthaea_evidence_deployment_scope::{
    DeploymentEvidenceContext, DeploymentScopedSafetyReceipt, bind_receipt_to_deployment,
};
use symthaea_evidence_lifecycle::ScopedSafetyEvidenceReceipt;
use symthaea_evidence_time_assurance::{
    EvidenceTimeGuard, EvidenceTimePolicy, EvidenceTimeSample, TrustedEvidenceTime,
};
use symthaea_evidence_trust_root_readiness::{
    TrustRootReadinessIssue, assess_trust_root_readiness,
};
use symthaea_evidence_verifier_diversity::{
    VerifierDiversityPolicy, VerifierDiversityRequirement, VerifierFaultDomainProfile,
};
use symthaea_formal_safety::{
    EvidenceKind, ProofObligation, SafetyCase, SafetyEvidenceReceipt, StrictSafetyCaseStatus,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrustRootReadinessCrucibleStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustRootReadinessScenario {
    pub scenario_id: String,
    pub passed: bool,
    pub observation: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustRootReadinessCrucibleReport {
    pub status: TrustRootReadinessCrucibleStatus,
    pub scenarios: Vec<TrustRootReadinessScenario>,
}

impl TrustRootReadinessCrucibleReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

fn scenario(
    id: &str,
    passed: bool,
    observation: impl Into<String>,
) -> TrustRootReadinessScenario {
    TrustRootReadinessScenario {
        scenario_id: id.into(),
        passed,
        observation: observation.into(),
    }
}

fn safety_case() -> SafetyCase {
    let mut case = SafetyCase::new("trust-root-readiness-fixture");
    case.add_obligation(
        ProofObligation::new("composite trust-root qualification claim", EvidenceKind::Test)
            .discharge("qualification:reviewed"),
    );
    case
}

fn context() -> DeploymentEvidenceContext {
    DeploymentEvidenceContext {
        schema_version: "1".into(),
        deployment_id: "node-1".into(),
        configuration_digest: "blake3:config-v1".into(),
        model_manifest_digest: Some("blake3:model-v1".into()),
        calibration_manifest_digest: Some("blake3:cal-v1".into()),
        evidence_refs: vec!["deployment:node-1".into()],
    }
}

fn policy_binding(kind: AssurancePolicyKind, revision: u64) -> AssurancePolicyBinding {
    AssurancePolicyBinding {
        kind,
        policy_id: format!("policy:{}", kind.code()),
        schema_version: "1".into(),
        content_digest: format!("blake3:{}-r{revision}", kind.code()),
        evidence_refs: vec![format!("review:{}:r{revision}", kind.code())],
    }
}

fn manifest(
    case: &SafetyCase,
    revision: u64,
    predecessor: Option<String>,
) -> AssurancePolicyManifest {
    AssurancePolicyManifest {
        schema_version: "1".into(),
        manifest_id: "assurance:node-1".into(),
        revision,
        safety_contract_digest: case.contract_digest(),
        deployment_id: "node-1".into(),
        configuration_digest: "blake3:config-v1".into(),
        model_manifest_digest: Some("blake3:model-v1".into()),
        calibration_manifest_digest: Some("blake3:cal-v1".into()),
        predecessor_manifest_digest: predecessor,
        change_ref: (revision > 1).then(|| format!("change:r{revision}")),
        policies: AssurancePolicyKind::REQUIRED
            .into_iter()
            .map(|kind| policy_binding(kind, revision))
            .collect(),
        evidence_refs: vec![format!("review:manifest-r{revision}")],
    }
}

fn signature(manifest: &AssurancePolicyManifest) -> ManifestSignatureVerificationReceipt {
    ManifestSignatureVerificationReceipt {
        receipt_id: format!("sig:r{}", manifest.revision),
        manifest_digest: manifest.manifest_digest(),
        signer_ref: "signer:a".into(),
        key_ref: "key:a1".into(),
        signature_algorithm: "ed25519".into(),
        signature_ref: format!("signature:r{}", manifest.revision),
        verified_by_ref: "verifier:manifest-signature".into(),
        verification_ref: format!("verification:r{}", manifest.revision),
        verified_at_ms: 500 + manifest.revision,
        evidence_refs: vec![format!("audit:r{}", manifest.revision)],
    }
}

fn signed_lineage(case: &SafetyCase) -> Vec<SignedAssurancePolicyRevision> {
    let r1 = manifest(case, 1, None);
    let r2 = manifest(case, 2, Some(r1.manifest_digest()));
    vec![
        SignedAssurancePolicyRevision {
            signature_receipt: signature(&r1),
            manifest: r1,
        },
        SignedAssurancePolicyRevision {
            signature_receipt: signature(&r2),
            manifest: r2,
        },
    ]
}

fn trusted_time() -> TrustedEvidenceTime {
    let policy = EvidenceTimePolicy {
        schema_version: "1".into(),
        policy_id: "time-v1".into(),
        expected_clock_source: "ptp-a".into(),
        expected_clock_domain: "domain-a".into(),
        expected_epoch_ref: "epoch-a".into(),
        maximum_clock_uncertainty_ms: 10,
        maximum_wall_clock_backstep_ms: 5,
    };
    let mut guard = EvidenceTimeGuard::new(policy).expect("valid time policy");
    guard
        .observe(EvidenceTimeSample {
            sample_id: "time:1".into(),
            wall_time_ms: 1_000,
            monotonic_time_ms: 5_000,
            clock_source: "ptp-a".into(),
            clock_domain: "domain-a".into(),
            epoch_ref: "epoch-a".into(),
            clock_uncertainty_ms: 10,
            evidence_refs: vec!["clock:1".into()],
        })
        .trusted_time
        .expect("trusted time")
}

fn deployment_receipt(
    case: &SafetyCase,
    id: &str,
    verifier: &str,
) -> DeploymentScopedSafetyReceipt {
    let obligation = &case.obligations[0];
    let scoped = ScopedSafetyEvidenceReceipt {
        receipt: SafetyEvidenceReceipt {
            receipt_id: id.into(),
            obligation_key: obligation.stable_key(),
            evidence_kind: obligation.expected_evidence,
            evidence_ref: format!("artifact:{id}"),
            evidence_digest: format!("blake3:{id}"),
            verifier_ref: verifier.into(),
            verified_at_ms: 100,
        },
        contract_digest: case.contract_digest(),
        valid_from_ms: 100,
        valid_until_ms: 5_000,
        applicability_refs: vec!["deployment:node-1".into()],
    };
    bind_receipt_to_deployment(scoped, &context(), format!("scope:{id}"))
        .expect("deployment scope")
}

fn policy_receipts(
    case: &SafetyCase,
    current: &SignedAssurancePolicyRevision,
) -> Vec<PolicyScopedSafetyReceipt> {
    [
        deployment_receipt(case, "r1", "verifier:a"),
        deployment_receipt(case, "r2", "verifier:b"),
    ]
    .into_iter()
    .map(|receipt| {
        let id = receipt.scoped_receipt.receipt.receipt_id.clone();
        bind_receipt_to_assurance_manifest(
            receipt,
            &current.manifest,
            &current.signature_receipt,
            format!("policy-scope:{id}"),
        )
        .expect("policy scope")
    })
    .collect()
}

fn verifier_profile(verifier: &str, suffix: &str) -> VerifierFaultDomainProfile {
    VerifierFaultDomainProfile {
        verifier_ref: verifier.into(),
        organization_domain: format!("org:{suffix}"),
        review_process_domain: format!("process:{suffix}"),
        toolchain_domain: format!("tool:{suffix}"),
        evidence_source_domain: format!("source:{suffix}"),
        evidence_refs: vec![format!("profile:{verifier}")],
    }
}

fn verifier_policy(case: &SafetyCase) -> VerifierDiversityPolicy {
    VerifierDiversityPolicy {
        schema_version: "1".into(),
        policy_id: "verifier-diversity-v1".into(),
        requirements: vec![VerifierDiversityRequirement {
            obligation_key: case.obligations[0].stable_key(),
            minimum_distinct_verifiers: 2,
            minimum_organization_domains: 2,
            minimum_review_process_domains: 2,
            minimum_toolchain_domains: 2,
            minimum_evidence_source_domains: 2,
            evidence_refs: vec!["review:verifier-diversity".into()],
        }],
        evidence_refs: vec!["policy:verifier-diversity".into()],
    }
}

fn atomic_policy(case: &SafetyCase) -> AtomicCoveragePolicy {
    AtomicCoveragePolicy {
        schema_version: "1".into(),
        policy_id: "atomic-v1".into(),
        facets: vec![AtomicEvidenceFacet {
            facet_id: "facet:a".into(),
            obligation_key: case.obligations[0].stable_key(),
            controlled_claim: "atomic facet a".into(),
            minimum_distinct_receipts: 2,
            minimum_distinct_evidence_objects: 2,
            evidence_refs: vec!["review:facet-a".into()],
        }],
        evidence_refs: vec!["policy:atomic".into()],
    }
}

fn facet_bindings() -> Vec<FacetEvidenceBinding> {
    vec![
        FacetEvidenceBinding {
            binding_id: "facet-binding:r1".into(),
            receipt_id: "r1".into(),
            facet_id: "facet:a".into(),
            facet_evidence_ref: "result:r1".into(),
            facet_evidence_digest: "blake3:facet-r1".into(),
            rationale_ref: "rationale:r1".into(),
        },
        FacetEvidenceBinding {
            binding_id: "facet-binding:r2".into(),
            receipt_id: "r2".into(),
            facet_id: "facet:a".into(),
            facet_evidence_ref: "result:r2".into(),
            facet_evidence_digest: "blake3:facet-r2".into(),
            rationale_ref: "rationale:r2".into(),
        },
    ]
}

fn signing_governance() -> SigningAuthorityGovernance {
    SigningAuthorityGovernance {
        policy: ManifestSigningAuthorityPolicy {
            schema_version: "1".into(),
            policy_id: "signing-authority-v1".into(),
            manifest_id: "assurance:node-1".into(),
            deployment_id: "node-1".into(),
            initial_signer_ref: "signer:a".into(),
            initial_key_ref: "key:a1".into(),
            allowed_signature_algorithms: vec!["ed25519".into()],
            trust_root_ref: "trust-root:safety-board".into(),
            evidence_refs: vec!["review:signing-authority".into()],
        },
        transitions: vec![],
    }
}

fn anchor(lineage: &[SignedAssurancePolicyRevision], trust_store_ref: &str) -> PolicyLineageAnchor {
    let tip = lineage.last().expect("lineage tip");
    PolicyLineageAnchor {
        schema_version: "1".into(),
        anchor_id: "anchor:node-1".into(),
        anchor_revision: 1,
        manifest_id: tip.manifest.manifest_id.clone(),
        deployment_id: tip.manifest.deployment_id.clone(),
        tip_revision: tip.manifest.revision,
        tip_manifest_digest: tip.manifest.manifest_digest(),
        predecessor_anchor_digest: None,
        recorded_at_ms: 800,
        trust_store_ref: trust_store_ref.into(),
        independent_verification_ref: "verification:anchor".into(),
        evidence_refs: vec!["audit:anchor".into()],
    }
}

fn trust_store_profile(trust_store_ref: &str, epoch: &str) -> TrustStoreProfile {
    TrustStoreProfile {
        schema_version: "1".into(),
        store_id: "policy-root".into(),
        trust_store_ref: trust_store_ref.into(),
        backend_kind: TrustStoreBackendKind::HardwareMonotonicCounter,
        hardware_instance_ref: Some(format!("hardware:{trust_store_ref}")),
        monotonic_counter_ref: format!("counter:{trust_store_ref}"),
        initial_counter_epoch: epoch.into(),
        minimum_independent_recovery_approvals: 2,
        evidence_refs: vec!["review:trust-store".into()],
    }
}

fn checkpoint_for_anchor(
    profile: &TrustStoreProfile,
    anchor: &PolicyLineageAnchor,
    store_revision: u64,
    counter_value: u64,
    predecessor: Option<String>,
    recorded_at_ms: u64,
) -> TrustStoreCheckpoint {
    TrustStoreCheckpoint {
        schema_version: "1".into(),
        checkpoint_id: format!("checkpoint:{store_revision}"),
        store_id: profile.store_id.clone(),
        store_revision,
        counter_epoch: profile.initial_counter_epoch.clone(),
        counter_value,
        anchor_revision: anchor.anchor_revision,
        anchor_digest: anchor.anchor_digest(),
        policy_tip_revision: anchor.tip_revision,
        predecessor_checkpoint_digest: predecessor,
        recorded_at_ms,
        attestation_ref: format!("attestation:checkpoint:{store_revision}"),
        independent_verification_ref: format!("verification:checkpoint:{store_revision}"),
        evidence_refs: vec![format!("audit:checkpoint:{store_revision}")],
    }
}

fn attestation_receipt(
    profile: &TrustStoreProfile,
    checkpoint: &TrustStoreCheckpoint,
    verified_at_ms: u64,
) -> CheckpointAttestationVerificationReceipt {
    CheckpointAttestationVerificationReceipt {
        receipt_id: format!("attestation-receipt:{}", checkpoint.store_revision),
        checkpoint_digest: checkpoint.checkpoint_digest(),
        logical_store_id: profile.store_id.clone(),
        trust_store_ref: profile.trust_store_ref.clone(),
        monotonic_counter_ref: profile.monotonic_counter_ref.clone(),
        counter_epoch: checkpoint.counter_epoch.clone(),
        counter_value: checkpoint.counter_value,
        attestation_ref: checkpoint.attestation_ref.clone(),
        verified_by_ref: "verifier:hardware-attestation".into(),
        verification_ref: format!("verification:attestation:{}", checkpoint.store_revision),
        verified_at_ms,
        evidence_refs: vec!["audit:attestation-verification".into()],
    }
}

#[allow(clippy::too_many_arguments)]
fn assess(
    case: &SafetyCase,
    lineage: &[SignedAssurancePolicyRevision],
    anchor: &PolicyLineageAnchor,
    profile: &TrustStoreProfile,
    current_segment: &[TrustStoreCheckpoint],
    recovery_ledger: &[RecoveryAcceptanceRecord],
    attestation: &CheckpointAttestationVerificationReceipt,
) -> symthaea_evidence_trust_root_readiness::TrustRootReadinessReport {
    let current = lineage.last().expect("lineage tip");
    let policy_receipts = policy_receipts(case, current);
    let signing = signing_governance();
    let signing_digest = signing.governance_digest();
    let anchor_digest = anchor.anchor_digest();
    assess_trust_root_readiness(
        case,
        &context(),
        &current.manifest,
        &current.signature_receipt,
        lineage,
        &policy_receipts,
        &[],
        &[],
        &[],
        &trusted_time(),
        &[
            verifier_profile("verifier:a", "a"),
            verifier_profile("verifier:b", "b"),
        ],
        &verifier_policy(case),
        &atomic_policy(case),
        &facet_bindings(),
        &signing,
        &signing_digest,
        std::slice::from_ref(anchor),
        &anchor_digest,
        profile,
        current_segment,
        recovery_ledger,
        attestation,
    )
}

fn accepted_recovery(
    profile: &TrustStoreProfile,
    first: &TrustStoreCheckpoint,
) -> RecoveryAcceptanceRecord {
    RecoveryAcceptanceRecord {
        schema_version: "1".into(),
        ledger_id: "recovery-ledger:policy-root".into(),
        ledger_revision: 1,
        logical_store_id: profile.store_id.clone(),
        authorization_id: "authorization:1".into(),
        recovery_commit_id: "commit:1".into(),
        previous_checkpoint_digest: "blake3:old-tip".into(),
        replacement_trust_store_ref: profile.trust_store_ref.clone(),
        replacement_counter_epoch: profile.initial_counter_epoch.clone(),
        first_replacement_checkpoint_digest: first.checkpoint_digest(),
        continuity_evidence_ref: "qualification:recovery-continuity".into(),
        continuity_evidence_digest: "blake3:recovery-continuity".into(),
        accepted_at_ms: 850,
        predecessor_acceptance_digest: None,
        evidence_refs: vec!["audit:recovery-acceptance".into()],
    }
}

pub fn run_trust_root_readiness_crucible() -> TrustRootReadinessCrucibleReport {
    let case = safety_case();
    let lineage = signed_lineage(&case);
    let anchor = anchor(&lineage, "trust-store:a");
    let profile = trust_store_profile("trust-store:a", "epoch:a");
    let checkpoint = checkpoint_for_anchor(&profile, &anchor, 1, 1, None, 850);
    let attestation = attestation_receipt(&profile, &checkpoint, 900);
    let mut scenarios = Vec::new();

    let baseline = assess(
        &case,
        &lineage,
        &anchor,
        &profile,
        std::slice::from_ref(&checkpoint),
        &[],
        &attestation,
    );
    scenarios.push(scenario(
        "exact_current_trust_root_ready",
        baseline.status == StrictSafetyCaseStatus::Ready && baseline.issues.is_empty(),
        format!("status={:?}", baseline.status),
    ));

    let mut wrong_checkpoint = checkpoint.clone();
    wrong_checkpoint.anchor_digest = "blake3:other-anchor".into();
    let wrong_attestation = attestation_receipt(&profile, &wrong_checkpoint, 900);
    let wrong_binding = assess(
        &case,
        &lineage,
        &anchor,
        &profile,
        std::slice::from_ref(&wrong_checkpoint),
        &[],
        &wrong_attestation,
    );
    scenarios.push(scenario(
        "checkpoint_anchor_substitution_invalid",
        wrong_binding.status == StrictSafetyCaseStatus::Invalid
            && wrong_binding
                .issues
                .contains(&TrustRootReadinessIssue::CheckpointDoesNotBindCurrentAnchor),
        format!("status={:?}", wrong_binding.status),
    ));

    let mut bad_attestation = attestation.clone();
    bad_attestation.checkpoint_digest = "blake3:other-checkpoint".into();
    let bad_attestation_report = assess(
        &case,
        &lineage,
        &anchor,
        &profile,
        std::slice::from_ref(&checkpoint),
        &[],
        &bad_attestation,
    );
    scenarios.push(scenario(
        "attestation_substitution_invalid",
        bad_attestation_report.status == StrictSafetyCaseStatus::Invalid
            && bad_attestation_report
                .issues
                .contains(&TrustRootReadinessIssue::InvalidCheckpointAttestationVerification),
        format!("status={:?}", bad_attestation_report.status),
    ));

    let late_checkpoint = checkpoint_for_anchor(&profile, &anchor, 1, 1, None, 995);
    let late_checkpoint_attestation = attestation_receipt(&profile, &late_checkpoint, 996);
    let late_checkpoint_report = assess(
        &case,
        &lineage,
        &anchor,
        &profile,
        std::slice::from_ref(&late_checkpoint),
        &[],
        &late_checkpoint_attestation,
    );
    scenarios.push(scenario(
        "late_checkpoint_cannot_retroactively_enable_interval",
        late_checkpoint_report.status == StrictSafetyCaseStatus::Blocked,
        format!("status={:?}", late_checkpoint_report.status),
    ));

    let late_verification = attestation_receipt(&profile, &checkpoint, 995);
    let late_verification_report = assess(
        &case,
        &lineage,
        &anchor,
        &profile,
        std::slice::from_ref(&checkpoint),
        &[],
        &late_verification,
    );
    scenarios.push(scenario(
        "late_attestation_verification_cannot_retroactively_enable_interval",
        late_verification_report.status == StrictSafetyCaseStatus::Blocked,
        format!("status={:?}", late_verification_report.status),
    ));

    let wrong_profile = trust_store_profile("trust-store:substituted", "epoch:a");
    let wrong_profile_checkpoint = checkpoint_for_anchor(&wrong_profile, &anchor, 1, 1, None, 850);
    let wrong_profile_attestation =
        attestation_receipt(&wrong_profile, &wrong_profile_checkpoint, 900);
    let wrong_profile_report = assess(
        &case,
        &lineage,
        &anchor,
        &wrong_profile,
        std::slice::from_ref(&wrong_profile_checkpoint),
        &[],
        &wrong_profile_attestation,
    );
    scenarios.push(scenario(
        "trust_store_identity_substitution_invalid",
        wrong_profile_report.status == StrictSafetyCaseStatus::Invalid,
        format!("status={:?}", wrong_profile_report.status),
    ));

    let first = checkpoint_for_anchor(&profile, &anchor, 1, 2, None, 840);
    let second = checkpoint_for_anchor(
        &profile,
        &anchor,
        2,
        2,
        Some(first.checkpoint_digest()),
        850,
    );
    let second_attestation = attestation_receipt(&profile, &second, 900);
    let regressed_segment_report = assess(
        &case,
        &lineage,
        &anchor,
        &profile,
        &[first, second],
        &[],
        &second_attestation,
    );
    scenarios.push(scenario(
        "current_counter_regression_invalid",
        regressed_segment_report.status == StrictSafetyCaseStatus::Invalid,
        format!("status={:?}", regressed_segment_report.status),
    ));

    let recovered_anchor = anchor(&lineage, "trust-store:b");
    let recovered_profile = trust_store_profile("trust-store:b", "epoch:b");
    let recovered_first = TrustStoreCheckpoint {
        schema_version: "1".into(),
        checkpoint_id: "checkpoint:4".into(),
        store_id: recovered_profile.store_id.clone(),
        store_revision: 4,
        counter_epoch: recovered_profile.initial_counter_epoch.clone(),
        counter_value: 1,
        anchor_revision: recovered_anchor.anchor_revision,
        anchor_digest: recovered_anchor.anchor_digest(),
        policy_tip_revision: recovered_anchor.tip_revision,
        predecessor_checkpoint_digest: Some("blake3:old-tip".into()),
        recorded_at_ms: 850,
        attestation_ref: "attestation:checkpoint:4".into(),
        independent_verification_ref: "verification:checkpoint:4".into(),
        evidence_refs: vec!["audit:checkpoint:4".into()],
    };
    let accepted = accepted_recovery(&recovered_profile, &recovered_first);
    let recovered_attestation =
        attestation_receipt(&recovered_profile, &recovered_first, 900);
    let recovered_report = assess(
        &case,
        &lineage,
        &recovered_anchor,
        &recovered_profile,
        std::slice::from_ref(&recovered_first),
        &[accepted],
        &recovered_attestation,
    );
    scenarios.push(scenario(
        "accepted_recovered_trust_root_can_retain_readiness",
        recovered_report.status == StrictSafetyCaseStatus::Ready,
        format!("status={:?}", recovered_report.status),
    ));

    let status = if scenarios.iter().all(|scenario| scenario.passed) {
        TrustRootReadinessCrucibleStatus::Pass
    } else {
        TrustRootReadinessCrucibleStatus::Fail
    };
    TrustRootReadinessCrucibleReport { status, scenarios }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn reviewed_end_to_end_matrix_passes() {
        let report = run_trust_root_readiness_crucible();
        assert_eq!(report.status, TrustRootReadinessCrucibleStatus::Pass);
        assert!(report.scenarios.iter().all(|scenario| scenario.passed));
        assert!(!report.grants_physical_authority());
    }

    #[test]
    fn scenario_ids_are_unique() {
        let report = run_trust_root_readiness_crucible();
        let ids = report
            .scenarios
            .iter()
            .map(|scenario| scenario.scenario_id.as_str())
            .collect::<BTreeSet<_>>();
        assert_eq!(ids.len(), report.scenarios.len());
    }
}
